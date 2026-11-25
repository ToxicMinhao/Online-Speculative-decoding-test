"""CPU qualification of pilot scheduling, retained canonicals and failure records."""
from collections import Counter
import json

import tempfile
import unittest
from pathlib import Path

from experiments.opencode_native_comparison.contract import digest
from . import evaluation
from .analysis import summarize


def inputs():
    def request(i, role):
        return dict(request_id=f'{role}{i}', source_id=f'source-{role}{i}',
            request_index=i, panel_ordinal=i, source_ordinal=1000 + i,
            prompt_ids=[i + (1 if role == 'r' else 2000)],
            stop_token_ids=[151645, 151643], max_new_tokens=2048)

    def reference(request):
        value = dict(request, response_ids=[151645], finish_reason='eos',
            reference_kind='frozen_DFlash', verified=True, provenance={'source': 'PR56'})
        value['binding_digest'] = digest(value)
        return value

    panel = [request(i, 'r') for i in range(128)]
    warmups = [request(i, 'w') for i in range(2)]
    return panel, warmups, {r['request_id']: reference(r) for r in panel}, {
        r['request_id']: reference(r) for r in warmups}


class Backend:
    def __init__(self, *, malformed=False, transport_failure=False):
        self.checkpoint = None
        self.malformed, self.transport_failure = malformed, transport_failure

    def check_weights(self, checkpoint):
        self.checkpoint = checkpoint
        return dict(status='passed', all_equal=True, tensors=58, source_tensor_count=58,
            native_tensor_count=43, checkpoint_path=checkpoint,
            helper_matches_current_layers=True, helper_norm_matches_current_layers=True,
            fused_enabled=True)

    def generate(self, prompt_ids, **kwargs):
        if self.transport_failure:
            raise RuntimeError('unknown native operation')
        if self.malformed:
            return dict(response_ids=[151645], metadata={})
        return dict(record_id=kwargs['request_id'], session_id=kwargs['session_id'],
            request_index=kwargs['request_index'], checkpoint_id=self.checkpoint,
            purpose=kwargs['purpose'], prompt_ids=prompt_ids, response_ids=[151645],
            finish_reason='eos', stop_token_ids=[151645, 151643], max_new_tokens=2048,
            metadata=dict(request_latency_s=.2, native_generate_s=.1, block_size=16,
                context_limit=4096, verification_calls=0, accepted_draft_tokens=0,
                committed_progress=0, trace=[], runtime='pinned_SGLang_FA3',
                feedback_capture_in_timed_request=False,
                cache_policy='idle_full_flush_before_every_request_and_publication',
                capture=dict(collect_features=False, feature_bytes=0,
                    native_capture_validated=True, prompt_ids=prompt_ids,
                    response_ids=[151645], blocks=[], cache_maintenance_s=.01)))


class Ledger:
    def __init__(self):
        self.counts = Counter()

    def call(self, counts, label, function, *args, **kwargs):
        self.counts.update(counts)
        return function(*args, **kwargs)


def execute(tmp_path, backend, *, source_bundles=False):
    panel, warmups, references, warmup_references = inputs()
    if source_bundles:
        # The old AR member is intentionally unusable: only retained F is used.
        references = {key: {'AR': None, 'F': value} for key, value in references.items()}
        warmup_references = {key: {'AR': None, 'F': value} for key, value in warmup_references.items()}
    ledger = Ledger()

    def activate(method, checkpoint, entry):
        backend.checkpoint = checkpoint
        return backend, {}

    status = evaluation.execute(dict(execution_commit='fixture', devices={'A': 'GPUa'},
        canonical_provenance={'source': 'PR56'}), tmp_path / 'evaluation', panel, warmups,
        references, warmup_references, {'BASE': 'base-export', 'PAIR': 'pair-export'},
        activate=activate, boundary=lambda entry: dict(quiet=True, entry=entry), ledger=ledger)
    return status, ledger


class EvaluationTests(unittest.TestCase):
    def test_fixed_rotated_schedule_and_original_identities(self):
        panel, *_ = inputs()
        schedule = evaluation.evaluation_schedule(panel)
        summary = evaluation.schedule_summary(schedule)
        self.assertEqual(summary['scored_generations'], 768)
        self.assertEqual(summary['warmup_generations'], 24)
        self.assertEqual(summary['first_position_requests_per_repeat'], {'BASE': [64] * 3, 'PAIR': [64] * 3})
        self.assertEqual(schedule[0]['request_ids'], [r['request_id'] for r in panel[:64]])
        self.assertEqual(schedule[2]['method'], 'PAIR')
        self.assertEqual(panel[0]['source_ordinal'], 1000)

    def test_packaged_runner_reads_retained_f_and_keeps_all_792_records(self):
        for source_bundles in (False, True):
            with self.subTest(source_bundles=source_bundles), tempfile.TemporaryDirectory() as folder:
                tmp_path = Path(folder)
                status, ledger = execute(tmp_path, Backend(), source_bundles=source_bundles)
                for key in ('completed_valid', 'attempted', 'returned'):
                    self.assertEqual(status[key], 792)
                for key in ('invalid', 'unknown', 'unattempted'):
                    self.assertEqual(status[key], 0)
                self.assertEqual(ledger.counts, {'scored_generation': 768, 'warmup_generation': 24})
                rows = [json.loads(line) for line in (tmp_path / 'evaluation/MEASUREMENTS.jsonl').read_text().splitlines()]
                report = summarize(rows)
                self.assertEqual(report['requests'], 128)
                self.assertEqual(report['totals']['BASE']['verification_calls'], 0)
                self.assertEqual(report['totals']['PAIR']['output_tokens'], 128)
                self.assertEqual(report['contrasts']['PAIR_vs_BASE']['request_latency_s_reduction_pct']['value'], 0)

    def test_returned_invalid_is_persisted_and_not_counted_unknown(self):
        with tempfile.TemporaryDirectory() as folder:
            tmp_path = Path(folder)
            with self.assertRaisesRegex(ValueError, 'validity failure'):
                execute(tmp_path, Backend(malformed=True))
            status = json.loads((tmp_path / 'evaluation/STATUS.json').read_text())
            self.assertEqual(status['returned'], 1)
            self.assertEqual(status['invalid'], 1)
            self.assertEqual(status['unknown'], 0)
            self.assertEqual(status['unattempted'], 791)
            raw = list((tmp_path / 'evaluation/raw').glob('*.json'))
            self.assertEqual(len(raw), 1)
            self.assertEqual(json.loads(raw[0].read_text())['response_ids'], [151645])

    def test_transport_failure_retains_one_unknown_attempt(self):
        with tempfile.TemporaryDirectory() as folder:
            tmp_path = Path(folder)
            with self.assertRaisesRegex(RuntimeError, 'unknown native'):
                execute(tmp_path, Backend(transport_failure=True))
            status = json.loads((tmp_path / 'evaluation/STATUS.json').read_text())
            self.assertEqual(status['attempted'], 1)
            self.assertEqual(status['unknown'], 1)
            self.assertEqual(status['returned'], 0)
            self.assertEqual(status['invalid'], 0)

    def test_changed_f_reference_fails_before_generation(self):
        panel, warmups, references, warmup_references = inputs()
        references[panel[0]['request_id']]['response_ids'] = [151643]
        with tempfile.TemporaryDirectory() as folder, self.assertRaisesRegex(ValueError, 'reference identity'):
            evaluation.execute(dict(execution_commit='fixture', devices={'A': 'GPUa'},
                canonical_provenance={'source': 'PR56'}), Path(folder) / 'evaluation', panel, warmups,
                references, warmup_references, {'BASE': 'base-export', 'PAIR': 'pair-export'},
                activate=lambda *_: self.fail('must not load an endpoint'), boundary=lambda _: {}, ledger=Ledger())


if __name__ == '__main__':
    unittest.main()
