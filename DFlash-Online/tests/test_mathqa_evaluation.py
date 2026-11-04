"""CPU-only control, schedule, deferred-reference and failure-accounting checks."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from online_feedback.storage import write_json
from experiments.mathqa_transfer import contract as C, evaluation as E
from test_full_k_gsm8k_evaluation import FullBackend, requests
from test_kplus_full_evaluation import RunRootNativeBackend


class MathQAEvaluationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.requests = requests(4)
        self.checkpoints = dict(O='/O58', K='/K58', KPLUS='/KPLUS58')

    def test_exact_prespecified_order(self):
        slots = C.final_schedule(requests(1024))
        self.assertEqual(len(slots), 9216)
        counts = Counter(C.method_order(b, r) for b in range(16) for r in range(3))
        self.assertEqual(len(counts), 6)
        self.assertEqual(set(counts.values()), {8})
        for b in range(16):
            for method in C.METHODS:
                self.assertEqual(sorted(C.method_order(b, r).index(method) for r in range(3)), [0, 1, 2])
        self.assertEqual(C.method_order(0, 0), ('O', 'K', 'KPLUS'))
        self.assertEqual(C.method_order(1, 0), ('O', 'KPLUS', 'K'))
        self.assertEqual(C.method_order(2, 0), ('K', 'KPLUS', 'O'))

    def run_fixture(self, backend=None):
        backend = backend or FullBackend(self.requests)
        folder = self.root / 'campaign_001/evaluation/final'
        with patch.object(C, 'BLOCK', 1):
            status = E.evaluate(backend, self.requests, self.checkpoints, folder, production=False)
            audit = E.audit_stage(folder, self.requests, self.checkpoints,
                                  evidence_root=self.root, production=False)
        return folder, status, audit

    def test_all_three_arms_deferred_reference_and_native_reconstruction(self):
        backend = RunRootNativeBackend(self.requests, self.root)
        folder, status, audit = self.run_fixture(backend)
        self.assertEqual(status['completed_valid_generations'], 36)
        self.assertEqual(status['unattempted_generations'], 0)
        self.assertEqual(audit['errors'], [])
        self.assertEqual(len(backend.executions), 36)
        canonical = json.loads((folder / 'CANONICAL.json').read_text())
        self.assertTrue(all(c['source_method'] == 'O' and c['source_repeat'] == 0 for c in canonical.values()))
        events = [json.loads(line) for line in (folder / 'VALIDATION_EVENTS.jsonl').read_text().splitlines()]
        self.assertTrue(any(event['status'] == 'canonical_pending' for event in events))

    def test_exact_three_untimed_historical_controls(self):
        request = self.requests[0]
        control = dict(canonical=dict(response_ids=[10, 11, 12, 13, 151645], finish_reason='eos',
                                     source_record_path='/historical/0.json'),
            counts={m:dict(verification_calls=2, accepted_draft_tokens=2, committed_progress=4, output_tokens=5)
                    for m in C.METHODS})
        backend = FullBackend([request])
        folder = self.root / 'controls'
        def boundary():
            return dict(storage=dict(status='passed', free_bytes=10, required_bytes=8),
                        quiescence=dict(status='passed', training=[], transfers=[], foreign_gpu_processes=[]))
        result = E.evaluate(backend, [request], self.checkpoints, folder, control=control, boundary_check=boundary)
        audit = E.audit_stage(folder, [request], self.checkpoints, control=control)
        self.assertEqual(result['completed_valid_generations'], 3)
        self.assertEqual(audit['errors'], [])
        self.assertEqual([r['phase'] for r in audit['rows']], ['control'] * 3)
        boundaries = folder / 'BOUNDARY_CHECKS.jsonl'
        boundaries.write_text(boundaries.read_text().splitlines()[0] + '\n')
        failed = E.audit_stage(folder, [request], self.checkpoints, control=control)
        self.assertIn('boundary_check_schedule', failed['errors'])

    def test_output_disagreement_preserves_bad_record_and_stops(self):
        def mutate(record, options):
            if record['checkpoint_id'] == '/K58':
                record['response_ids'][0] = 99
        folder = self.root / 'failed'
        with self.assertRaisesRegex(RuntimeError, 'invalid'):
            E.evaluate(FullBackend(self.requests, mutate), self.requests, self.checkpoints, folder, production=False)
        status = json.loads((folder / 'STATUS.json').read_text())
        self.assertEqual(status['returned_generations'], 5)
        self.assertEqual(status['invalid_generations'], 1)
        self.assertEqual(status['unattempted_generations'], 31)
        self.assertTrue((folder / '0000_r0_K.json').is_file())

    def test_transport_failure_has_unknown_attempt_and_unattempted_work(self):
        backend = FullBackend(self.requests)
        backend.generate = lambda *a, **k: (_ for _ in ()).throw(RuntimeError('transport'))
        folder = self.root / 'failed'
        with self.assertRaisesRegex(RuntimeError, 'transport'):
            E.evaluate(backend, self.requests, self.checkpoints, folder, production=False)
        status = json.loads((folder / 'STATUS.json').read_text())
        self.assertEqual(status['unknown_generation_attempts'], 1)
        self.assertEqual(status['unattempted_generations'], 35)

    def test_raw_audit_rejects_changed_native_and_missing_repeats(self):
        folder, _, _ = self.run_fixture()
        raw = json.loads((folder / '0000_r0_O.json').read_text())
        native = Path(raw['metadata']['native_trace_path'])
        changed = json.loads(native.read_text())
        changed['feature_bytes'] = 1
        write_json(native, changed)
        with patch.object(C, 'BLOCK', 1):
            audit = E.audit_stage(folder, self.requests, self.checkpoints, evidence_root=self.root, production=False)
        self.assertNotEqual(audit['status'], 'passed')
        self.assertIn('0000_r0_O: native_capture_changed', audit['errors'])

    def test_production_rejects_small_panel_and_extra_arm(self):
        with self.assertRaises(ValueError):
            C.validate_inputs(self.requests, self.checkpoints)
        with self.assertRaises(ValueError):
            C.validate_inputs(requests(1024), {**self.checkpoints, 'F':'/F'})


if __name__ == '__main__':
    unittest.main()
