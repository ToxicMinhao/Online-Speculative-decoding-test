"""Independent CPU review fixtures for incomplete and zero-round evaluation."""
import json

import pytest

from . import evaluation
from .analysis import reduce_requests
from .contract import METHODS, digest
from .test_analysis import fixture


def test_reduction_rejects_changed_endpoint_across_requests():
    rows = fixture()
    for row in rows:
        if row['request_id'] == 'a' and row['method'] == 'C_CODE58':
            row['checkpoint_id'] = 'unrelated_endpoint'
    with pytest.raises(ValueError, match='endpoint'):
        reduce_requests(rows, expected_requests=2)


def test_reduction_rejects_source_identity_mismatch_between_arms():
    rows = fixture()
    for row in rows:
        if row['request_id'] == 'a' and row['method'] == 'C_CODE58':
            row['source_id'] = 'another_input'
    with pytest.raises(ValueError, match='Source identity'):
        reduce_requests(rows, expected_requests=2)


def test_frozen_single_eos_with_zero_verifications_is_valid():
    request = dict(request_id='r', source_id='r', request_index=0, prompt_ids=[1, 2],
                   stop_token_ids=[99], max_new_tokens=2048)
    slot = dict(activation=0, phase='scored', request_ordinal=0, method='F', repeat=0)
    canonical = dict(request, response_ids=[99], finish_reason='eos')
    record = dict(record_id=evaluation.label_for(slot), session_id=evaluation.SESSION,
        request_index=0, checkpoint_id='frozen', purpose='evaluation',
        prompt_ids=[1, 2], response_ids=[99], finish_reason='eos',
        max_new_tokens=2048, stop_token_ids=[99], metadata=dict(
            request_latency_s=.2, native_generate_s=.1, block_size=16, context_limit=4096,
            verification_calls=0, accepted_draft_tokens=0, committed_progress=0,
            trace=[], cache_policy='idle_full_flush_before_every_request_and_publication',
            runtime='pinned_SGLang_FA3', feedback_capture_in_timed_request=False,
            capture=dict(collect_features=False, feature_bytes=0, native_capture_validated=True,
                         prompt_ids=[1, 2], response_ids=[99], blocks=[], cache_maintenance_s=.01)))
    row, errors = evaluation.measure(record, request, 'frozen', canonical, slot)
    assert errors == []
    assert row['verification_calls'] == row['native_accepted_draft_tokens'] == 0
    assert row['output_tokens'] == 1


def test_malformed_return_is_invalid_not_unknown(tmp_path, monkeypatch):
    requests = [dict(request_id=f'r{i}', source_id=f'r{i}', request_index=i,
        prompt_ids=[i + 1], stop_token_ids=[99], max_new_tokens=2048) for i in range(1024)]
    warmups = [dict(request_id=f'w{i}', source_id=f'w{i}', request_index=i,
        prompt_ids=[2000 + i], stop_token_ids=[99], max_new_tokens=2048) for i in range(2)]

    def reference(request):
        value = dict(request, response_ids=[99], finish_reason='eos',
                     reference_kind='target_only_AR', verified=True, provenance={})
        value['binding_digest'] = digest(value)
        return value

    canonicals = {row['request_id']: reference(row) for row in requests}
    warmup_canonicals = {row['request_id']: reference(row) for row in warmups}
    entry = dict(activation=0, repeat=0, group=0, method='AR', request_ids=['r0'])
    monkeypatch.setattr(evaluation, 'evaluation_schedule', lambda _: [entry])
    monkeypatch.setattr(evaluation, 'schedule_summary', lambda _: {})

    class Backend:
        def generate(self, *args, **kwargs):
            return dict(response_ids=[99], metadata={})

    class Ledger:
        def call(self, counts, label, function, *args, **kwargs):
            return function(*args, **kwargs)

    root = tmp_path / 'evaluation'
    with pytest.raises(ValueError, match='validity failure'):
        evaluation.execute(dict(execution_commit='fixture', devices={'A': 'GPUa'},
            canonical_provenance={}), root, requests, warmups, canonicals, warmup_canonicals,
            {method: method for method in METHODS}, activate=lambda *args: (Backend(), {}),
            boundary=lambda entry: {}, ledger=Ledger())
    status = json.loads((root / 'STATUS.json').read_text())
    assert status['returned'] == status['invalid'] == 1
    assert status['unknown'] == 0
    assert status['unattempted'] == 15599
    raw = list((root / 'raw').glob('*.json'))
    assert len(raw) == 1 and json.loads(raw[0].read_text())['response_ids'] == [99]
    saved = [json.loads(line) for line in (root / 'MEASUREMENTS.jsonl').read_text().splitlines()]
    assert saved[0]['valid_measurement'] is False
    assert 'malformed_returned_record' in saved[0]['errors'][0]
