"""Fixed two-method endpoint order and retained K58 count gate."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.kplus_full_gsm8k import evaluation as E
from online_feedback.storage import write_json
from test_full_k_gsm8k_evaluation import requests, FullBackend


class RunRootNativeBackend(FullBackend):
    """Place actual sidecars outside the evaluation folder, as production does."""

    def __init__(self, inputs, run_root):
        super().__init__(inputs)
        self.run_root = run_root
        self.recorded_root = Path('/original_host/kplus/production_001')
        self.native_paths = []

    def generate(self, prompt_ids, **options):
        record = super().generate(prompt_ids, **options)
        native = self.run_root / 'campaign_001/final_native_attempt_001/native' / (
            f'request_{len(self.executions):06d}.json')
        native.parent.mkdir(parents=True, exist_ok=True)
        Path(record['metadata']['native_trace_path']).replace(native)
        record['metadata']['native_trace_path'] = str(self.recorded_root / native.relative_to(self.run_root))
        write_json(Path(options['save_to']), record)
        self.native_paths.append(native)
        return record


class EndpointTests(unittest.TestCase):
    def test_full_order_and_budget(self):
        slots = E.final_schedule(requests(1024))
        balance = E.schedule_balance(slots)
        self.assertEqual(len(slots), 6144)
        self.assertEqual(balance['method_positions'], {'K': [24, 24], 'KPLUS': [24, 24]})
        self.assertEqual(set(balance['pairwise_precedence'].values()), {24})
        self.assertEqual([s['panel_ordinal'] for s in slots[:64]], list(range(64)))
        self.assertEqual(E.method_order(0, 0), ('K', 'KPLUS'))
        self.assertEqual(E.method_order(0, 1), ('KPLUS', 'K'))
        self.assertEqual(E.method_order(1, 0), ('KPLUS', 'K'))

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.inputs = requests(8)
        self.canonicals = {r['request_id']: dict(response_ids=[10, 11, 12, 13, 151645],
            finish_reason='eos', source_record_path='/original/' + r['request_id']) for r in self.inputs}
        self.baseline = {r['request_id']: dict(verification_calls=2, accepted_draft_tokens=2,
            committed_progress=4, output_tokens=5) for r in self.inputs}
        self.checkpoints = dict(K='/K58', KPLUS='/KP58')

    def tiny(self):
        return patch.multiple(E, PANEL_SIZE=8, REQUEST_BLOCK=2,
                              HISTORICAL_K_TOTALS=dict(verification_calls=16, output_tokens=40))

    def test_raw_endpoint_all_repeats_and_exact_old_count_gate(self):
        backend = FullBackend(self.inputs)
        checks = []
        with self.tiny():
            result = E.evaluate_final(backend, self.inputs, self.checkpoints, self.root / 'final',
                canonicals=self.canonicals, baseline_counts=self.baseline,
                boundary_check=lambda: checks.append('outside request') or {'status': 'passed'})
            audit = E.audit_stage(self.root / 'final', self.inputs, self.checkpoints,
                                 canonicals=self.canonicals, baseline_counts=self.baseline)
        self.assertEqual(result['completed_valid_generations'], 48)
        self.assertEqual(audit['errors'], [])
        self.assertEqual(len(checks), 24)

    def test_baseline_discrepancy_stops_without_replacing_request(self):
        bad = deepcopy(self.baseline)
        bad['eval0']['verification_calls'] = 3
        bad['eval1']['verification_calls'] = 1
        with self.tiny(), self.assertRaisesRegex(RuntimeError, 'historical_K58_verification_calls'):
            E.evaluate_final(FullBackend(self.inputs), self.inputs, self.checkpoints,
                             self.root / 'failed', canonicals=self.canonicals, baseline_counts=bad)
        status = json.loads((self.root / 'failed/STATUS.json').read_text())
        self.assertEqual(status['attempted_generations'], 1)
        self.assertEqual(status['invalid_generations'], 1)
        self.assertEqual(status['unattempted_generations'], 47)

    def saved_native_endpoint(self):
        backend = RunRootNativeBackend(self.inputs, self.root)
        folder = self.root / 'campaign_001/evaluation/final'
        E.evaluate_final(backend, self.inputs, self.checkpoints, folder,
                         canonicals=self.canonicals, baseline_counts=self.baseline)
        return backend, folder

    def test_audit_rebases_saved_native_sidecars_from_run_root(self):
        with self.tiny():
            backend, folder = self.saved_native_endpoint()
            wrong_root = E.audit_stage(folder, self.inputs, self.checkpoints,
                canonicals=self.canonicals, baseline_counts=self.baseline)
            audit = E.audit_stage(folder, self.inputs, self.checkpoints,
                canonicals=self.canonicals, baseline_counts=self.baseline, evidence_root=self.root)
        self.assertEqual(wrong_root['status'], 'invalid_or_incomplete')
        self.assertIn('FileNotFoundError', wrong_root['errors'][0])
        self.assertEqual(audit['status'], 'passed')
        self.assertEqual(audit['errors'], [])
        self.assertEqual(len(audit['rows']), 48)
        self.assertTrue(all(path.is_file() for path in backend.native_paths))

    def test_audit_rejects_missing_saved_native_sidecar_with_correct_root(self):
        with self.tiny():
            backend, folder = self.saved_native_endpoint()
            backend.native_paths[0].unlink()
            audit = E.audit_stage(folder, self.inputs, self.checkpoints,
                canonicals=self.canonicals, baseline_counts=self.baseline, evidence_root=self.root)
        self.assertEqual(audit['status'], 'invalid_or_incomplete')
        self.assertIn('FileNotFoundError', audit['errors'][0])
        self.assertEqual(audit['rows'], [])
        self.assertTrue((folder / '0000_r0_K.json').is_file())
        self.assertTrue((folder / 'MEASUREMENTS.jsonl').is_file())

    def test_audit_rejects_changed_native_sidecar_even_when_embedded_capture_is_valid(self):
        with self.tiny():
            backend, folder = self.saved_native_endpoint()
            changed = json.loads(backend.native_paths[0].read_text())
            changed['feature_bytes'] = 1
            write_json(backend.native_paths[0], changed)
            audit = E.audit_stage(folder, self.inputs, self.checkpoints,
                canonicals=self.canonicals, baseline_counts=self.baseline, evidence_root=self.root)
        self.assertEqual(audit['status'], 'invalid_or_incomplete')
        self.assertIn('0000_r0_K: original_native_capture_changed', audit['errors'])
        raw = json.loads((folder / '0000_r0_K.json').read_text())
        self.assertEqual(raw['metadata']['capture']['feature_bytes'], 0)


if __name__ == '__main__':
    unittest.main()
