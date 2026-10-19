"""CPU safeguards for the bounded diagnostic and interpretation validity."""
import json
from types import SimpleNamespace

import pytest

from . import retained_replay as replay


def records():
    case = dict(case_id='wildchat_00:t1:m0', stream='wildchat_00', turn=1, module=0,
                input_ids=[1, 2], output_ids=[3, 4, 5], matrix_sha256='matrix',
                frozen_metrics={'verification_calls': 2})
    frozen = dict(input_ids=[1, 2], output_ids=[3, 4, 5], verification_calls=2,
                  accepted_draft_tokens=1, verification_committed_tokens=3)
    singleton = dict(frozen, verification_calls=1, accepted_draft_tokens=2,
        read=dict(routing_bypassed=True, fixed_for_entire_request=True, matrix_sha256='matrix',
                  coefficients=[1., 0., 0., 0.], corrected_head_calls=1))
    return case, {'F': frozen, 'BANK': singleton}


def test_exact_scope_and_storage(monkeypatch, tmp_path):
    assert replay.CASES == (('wildchat_00', 1, 0), ('wildchat_01', 2, 1),
        ('wildchat_02', 6, 3), ('wildchat_19', 15, 2), ('personamem_persona42', 3, 2))
    monkeypatch.setattr(replay.shutil, 'disk_usage', lambda _: SimpleNamespace(free=11*2**30))
    plan = replay.storage_preflight(tmp_path/'out')
    assert plan['native_generations'] == 10
    assert plan['reserve_bytes'] == 8*2**30
    assert plan['optimizer_states'] == plan['model_exports'] == 0
    monkeypatch.setattr(replay.shutil, 'disk_usage', lambda _: SimpleNamespace(free=11*2**30-1))
    with pytest.raises(ValueError, match='reserve'):
        replay.storage_preflight(tmp_path/'out')


def test_invalid_outputs_or_baseline_never_become_gains():
    case, rows = records()
    result = replay.summarize(case, rows, {'status': 'unavailable'})
    assert result['gain_comparison_valid'] and result['calls_saved'] == 1
    rows['BANK']['output_ids'] = [3, 4, 6]
    result = replay.summarize(case, rows, {'status': 'unavailable'})
    assert not result['gain_comparison_valid'] and result['calls_saved'] is None
    rows['BANK']['output_ids'] = case['output_ids']
    rows['F']['verification_calls'] = 3
    result = replay.summarize(case, rows, {'status': 'unavailable'})
    assert not result['gain_comparison_valid'] and result['calls_saved'] is None


def test_report_retains_precision_limits_and_unavailable_evidence(tmp_path):
    case, rows = records()
    replay.write_report(tmp_path, [case], {case['case_id']: rows},
                        {case['case_id']: {'status': 'unavailable', 'reason': 'missing anchors'}})
    text = (tmp_path/'REPORT.md').read_text()
    assert replay.INTERPRETATION in text
    assert 'surviving BF16 rounding does not itself establish' in text
    assert 'not deployment-speedup claims' in text
    assert 'unavailable' in text
    result = json.loads((tmp_path/'COMPARISON.json').read_text())
    assert result['native_generations'] == 2  # Unit fixture, never native model execution.


def test_bypassed_head_is_not_evidence_against_corrections():
    case, rows = records()
    rows['BANK']['read']['corrected_head_calls'] = 0
    result = replay.summarize(case, rows, {'status': 'unavailable'})
    assert not result['gain_comparison_valid'] and result['calls_saved'] is None


def test_fresh_output_required_before_any_read_or_gpu(tmp_path):
    with pytest.raises(ValueError, match='Fresh diagnostic'):
        replay.main(['--campaign-root', str(tmp_path/'missing'), '--out', str(tmp_path), '--execute'])
