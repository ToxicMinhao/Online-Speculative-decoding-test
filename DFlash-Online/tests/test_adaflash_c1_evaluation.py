"""Fixed three-method endpoint, canonical validity and repeat consistency."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.adaflash_c1 import evaluation as E
from experiments.adaflash_c1 import contract as C
from online_feedback.storage import write_json
from test_full_k_gsm8k_evaluation import requests, FullBackend


class RunRootNativeBackend(FullBackend):
    """Place actual sidecars outside the evaluation folder, as production does."""

    def __init__(self, inputs, run_root):
        super().__init__(inputs)
        self.run_root = run_root
        self.recorded_root = Path('/original_host/adaflash/production_001')
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
        self.assertEqual(len(slots), 9216)
        self.assertEqual(len(balance['orders']), 6)
        self.assertEqual({row['count'] for row in balance['orders']}, {8})
        self.assertEqual([s['panel_ordinal'] for s in slots[:64]], list(range(64)))
        self.assertEqual(C.method_order(0, 0), ('O', 'KPLUS', 'A'))

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.inputs = requests(8)
        self.canonicals = {r['request_id']: dict(response_ids=[10, 11, 12, 13, 151645],
            finish_reason='eos', source_record_path='/original/' + r['request_id']) for r in self.inputs}
        self.checkpoints = dict(O='/O58', KPLUS='/KP58', A='/A58')

    def tiny(self):
        return patch.multiple(C, PANEL_SIZE=8, REQUEST_BLOCK=2)

    def test_raw_endpoint_all_repeats_without_historical_count_gate(self):
        backend = FullBackend(self.inputs)
        checks = []
        with self.tiny():
            result = E.evaluate_final(backend, self.inputs, self.checkpoints, self.root / 'final',
                canonicals=self.canonicals,
                boundary_check=lambda: checks.append('outside request') or {'status': 'passed'})
            audit = E.audit_stage(self.root / 'final', self.inputs, self.checkpoints,
                                 canonicals=self.canonicals)
        self.assertEqual(result['completed_valid_generations'], 72)
        self.assertEqual(audit['errors'], [])
        self.assertEqual(len(checks), 36)

    def test_output_discrepancy_stops_and_preserves_unattempted_work(self):
        def corrupt(record, options):
            record['response_ids'][0] = 999
        backend = FullBackend(self.inputs, mutation=corrupt)
        with self.tiny(), self.assertRaises(RuntimeError):
            E.evaluate_final(backend, self.inputs, self.checkpoints, self.root / 'failed',
                             canonicals=self.canonicals)
        status = json.loads((self.root / 'failed/STATUS.json').read_text())
        self.assertEqual(status['attempted_generations'], 1)
        self.assertEqual(status['invalid_generations'], 1)
        self.assertEqual(status['unattempted_generations'], 71)

    def saved_native_endpoint(self):
        backend = RunRootNativeBackend(self.inputs, self.root)
        folder = self.root / 'campaign_001/evaluation/final'
        E.evaluate_final(backend, self.inputs, self.checkpoints, folder,
                         canonicals=self.canonicals)
        return backend, folder

    def test_audit_rebases_saved_native_sidecars_from_run_root(self):
        with self.tiny():
            backend, folder = self.saved_native_endpoint()
            wrong_root = E.audit_stage(folder, self.inputs, self.checkpoints,
                canonicals=self.canonicals)
            audit = E.audit_stage(folder, self.inputs, self.checkpoints,
                canonicals=self.canonicals, evidence_root=self.root)
        self.assertEqual(wrong_root['status'], 'invalid_or_incomplete')
        self.assertIn('FileNotFoundError', wrong_root['errors'][0])
        self.assertEqual(audit['status'], 'passed')
        self.assertEqual(audit['errors'], [])
        self.assertEqual(len(audit['rows']), 72)
        self.assertTrue(all(path.is_file() for path in backend.native_paths))

    def test_audit_rejects_missing_saved_native_sidecar_with_correct_root(self):
        with self.tiny():
            backend, folder = self.saved_native_endpoint()
            backend.native_paths[0].unlink()
            audit = E.audit_stage(folder, self.inputs, self.checkpoints,
                canonicals=self.canonicals, evidence_root=self.root)
        self.assertEqual(audit['status'], 'invalid_or_incomplete')
        self.assertIn('FileNotFoundError', audit['errors'][0])
        self.assertEqual(audit['rows'], [])
        self.assertTrue((folder / '0000_r0_O.json').is_file())
        self.assertTrue((folder / 'MEASUREMENTS.jsonl').is_file())

    def test_audit_rejects_changed_native_sidecar_even_when_embedded_capture_is_valid(self):
        with self.tiny():
            backend, folder = self.saved_native_endpoint()
            changed = json.loads(backend.native_paths[0].read_text())
            changed['feature_bytes'] = 1
            write_json(backend.native_paths[0], changed)
            audit = E.audit_stage(folder, self.inputs, self.checkpoints,
                canonicals=self.canonicals, evidence_root=self.root)
        self.assertEqual(audit['status'], 'invalid_or_incomplete')
        self.assertIn('0000_r0_O: original_native_capture_changed', audit['errors'])
        raw = json.loads((folder / '0000_r0_O.json').read_text())
        self.assertEqual(raw['metadata']['capture']['feature_bytes'], 0)


if __name__ == '__main__':
    unittest.main()
