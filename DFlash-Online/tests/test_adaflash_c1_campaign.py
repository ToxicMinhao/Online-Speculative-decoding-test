"""CPU orchestration: exact one-arm work, cleanup and phase accounting."""
from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest
from unittest.mock import Mock, patch

from experiments.adaflash_c1 import campaign as C
from experiments.adaflash_c1 import contract


class CampaignTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_one_arm_budget_and_untrained_tail(self):
        self.assertEqual(C.expected_work(58, 49), C.ARM_LIMITS)
        self.assertEqual(C.expected_work(0), dict.fromkeys(C.ARM_LIMITS, 0))
        before, after = C.expected_work(58), C.expected_work(58, 49)
        self.assertEqual({key for key in before if before[key] != after[key]}, {'online_generations'})
        self.assertEqual(contract.ARMS, ('A',))
        budget = contract.NEW_BUDGET
        self.assertEqual(budget['online_generations'] + budget['correctness_generations'] + budget['final_generations'], 16693)
        self.assertEqual(budget['baseline_optimizer_steps'], 0)
        self.assertEqual(budget['intermediate_generations'], 0)
        with self.assertRaises(ValueError):
            contract.validate_plan(dict(schema='full_k_gsm8k_v1'))

    def test_preparation_retirement_occurs_only_after_successful_actual_prepare(self):
        calls = []
        backend = SimpleNamespace(generate=lambda *args, **kwargs: calls.append('generate') or {},
            prepare_feedback=lambda *args: calls.append('prepared') or dict(status='prepared'))
        ledger = C.WorkLedger(self.root / 'WORK.jsonl', C.ARM_LIMITS)
        progress = dict(generation_phase='online', event_root=str(self.root))
        with patch.object(C, 'retire_redundant_features', side_effect=lambda **kwargs: calls.append('retired')):
            C.counted_backend(backend, ledger, progress, self.root)
            backend.generate([1])
            progress['generation_phase'] = 'correctness'
            backend.generate([1])
            backend.prepare_feedback({'record_id': 'r'}, self.root / 'r.pt')
        self.assertEqual(calls, ['generate', 'generate', 'prepared', 'retired'])
        work = C.read_work(ledger.path, C.ARM_LIMITS)
        self.assertEqual(work['attempts'], work['completed'])
        self.assertEqual(work['completed']['terminal_preparation_forwards'], 1)
        self.assertEqual(work['completed']['optimizer_steps'], 0)

    def test_cleanup_attempts_all_owners_and_marks_failure(self):
        closed = []
        def fail():
            closed.append('trainer')
            raise RuntimeError('cleanup failed')
        progress = dict(status='completed')
        errors = C.close_owned(progress, SimpleNamespace(close=fail),
                               SimpleNamespace(close=lambda: closed.append('backend')))
        self.assertEqual(closed, ['trainer', 'backend'])
        self.assertEqual(progress['status'], 'failed')
        self.assertEqual(len(errors), 1)

    def test_deadline_handler_unwinds_owned_operation(self):
        handlers, cleaned = {}, []
        def run(_root):
            try:
                handlers[C.signal.SIGTERM](C.signal.SIGTERM, None)
            finally:
                cleaned.append(True)
        with patch('sys.argv', ['campaign', str(self.root)]), \
                patch.object(C.signal, 'signal', side_effect=lambda signal, handler: handlers.update({signal: handler})), \
                patch.object(C, 'run', side_effect=run), self.assertRaises(SystemExit) as failure:
            C.main()
        self.assertEqual(failure.exception.code, 143)
        self.assertEqual(cleaned, [True])

    def test_natural_anchor_comparison_never_supplies_old_schedule(self):
        self.assertEqual(C.observe_anchors({}, None, 1)['status'], 'unavailable')
        current = dict(records=[dict(record_id='a', **{'pass': 0}, sampled_anchors=[2])])
        prior = dict(records=[dict(record_id='a', **{'pass': 0}, sampled_anchors=[1])])
        with patch.object(C, 'read', return_value=prior):
            result = C.observe_anchors(current, '/original', 1)
        self.assertEqual(result['identity_matches'], 1)
        self.assertEqual(result['exact_anchor_matches'], 0)
        self.assertFalse(result['forced'])

    def test_numerical_failure_is_saved_before_any_native_model_work(self):
        plan = dict(binding='fixture', manifest='/manifest', storage={})
        evidence = dict(status='failed', actual_work={'synthetic_loss_forwards': 2},
                        failure={'gradient_absolute_error': .25})
        values = {str(self.root / 'sealed/PLAN.json'): plan, '/manifest': {'train_requests': []}}
        with patch.object(C, 'read', side_effect=lambda path: values[str(path)]), \
                patch.object(C, 'validate_plan'), patch.object(C, 'admit'), \
                patch('online_feedback.adaflash_opd.numerical_qualification', return_value=evidence), \
                patch('online_feedback.sglang_backend.SGLangDFlashBackend') as backend, \
                self.assertRaisesRegex(ValueError, 'numerical qualification failed'):
            C.run_arm(self.root)
        backend.assert_not_called()
        base = self.root / 'campaign_001/arms/A'
        self.assertEqual(json.loads((base / 'LOSS_QUALIFICATION.json').read_text()), evidence)
        status = json.loads((base / 'STATUS.json').read_text())
        self.assertEqual(status['status'], 'failed')
        self.assertEqual(status['actual_work']['attempts'], dict.fromkeys(C.ARM_LIMITS, 0))

    def test_final_audit_receives_run_root_for_native_sidecars(self):
        from experiments.adaflash_c1 import evaluation
        plan = dict(execution_source={'commit': 'fixture'}, storage={}, references={'O': '/retained/O', 'KPLUS': '/retained/KPLUS'},
                    target='/target', draft='/released', worker_python='/worker',
                    evaluation_requests=[], canonicals='/canonicals')
        values = {
            str(self.root / 'sealed/PLAN.json'): plan,
            str(self.root / 'campaign_001/arms/A/STATUS.json'): {'final_checkpoint': '/new/A'},
            '/canonicals': {},
        }
        backend = SimpleNamespace(loading_s=0., close=Mock())
        with patch.object(C, 'read', side_effect=lambda path: values[str(path)]), \
                patch.object(C, 'validate_plan'), patch.object(C, 'admit'), \
                patch.object(C, 'execute_training'), patch.object(C, 'evaluation_boundary', return_value={}), \
                patch('online_feedback.sglang_backend.SGLangDFlashBackend', return_value=backend), \
                patch.object(evaluation, 'evaluate_final', return_value={'status': 'completed'}), \
                patch.object(evaluation, 'audit_stage', return_value={'status': 'passed', 'rows': []}) as audit:
            result = C.run(self.root)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(audit.call_args.args[0], self.root / 'campaign_001/evaluation/final')
        self.assertEqual(audit.call_args.kwargs['evidence_root'], self.root)
        backend.close.assert_called_once_with()

    def test_final_boundary_checks_storage_and_quiescence_outside_requests(self):
        from experiments.adaflash_c1 import preflight
        calls = []
        with patch.object(C, 'resource_check', side_effect=lambda *a, **k: calls.append('storage') or {'status': 'passed'}), \
                patch.object(preflight, 'timing_barrier', side_effect=lambda owner: calls.append(owner) or {'status': 'passed'}):
            C.evaluation_boundary(self.root, {'storage': {}}, phase='before_final_backend')
            result = C.evaluation_boundary(self.root, {'storage': {}}, SimpleNamespace(worker=SimpleNamespace(pid=71)))
        self.assertEqual(calls, ['storage', None, 'storage', 71])
        self.assertEqual(set(result), {'resource', 'timing'})


if __name__ == '__main__':
    unittest.main()
