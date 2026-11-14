"""CPU-only orchestration, admission and honest incomplete-report checks."""
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
import os
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from experiments.threatened_repair_replay import campaign, probe_client
from experiments.verification_transition.budget import BudgetExceeded
from test_threat_prepare import prior_attempt_fixture


def fixture(folder):
    root = Path(folder)
    campaign.write_json(root/'parent.json', {'metadata': {'trace': [1, 2, 3]}})
    campaign.write_json(root/'sealed/PROBE_KEYS.json', [])
    campaign.write_json(root/'sealed/MEMORY_PRIVATE.json', [])
    (root/'sealed/PROBE_KEYS.csv').write_text('request_id,canonical_start,round_ordinal\n')
    (root/'sealed/MEMORY_REFERENCE.csv').write_text('request_id,canonical_start,proposal_slot\n')
    plan = dict(root=str(root), execution_commit='test-source', serving_gpu=0,
        probe_requests=[{'request_id': 'r'}], parent_records={'r': str(root/'parent.json')},
        probe_keys=str(root/'sealed/PROBE_KEYS.json'), memory_tags=str(root/'sealed/MEMORY_PRIVATE.json'))
    campaign.write_json(root/'sealed/PLAN.json', plan)
    return plan


class AdmissionTests(unittest.TestCase):
    def test_changed_sealed_carryover_or_removed_prior_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            prior = prior_attempt_fixture(Path(folder)/'prior')
            accounting = campaign.attempt_accounting(prior)
            for plan in (dict(prior_attempt=prior, budget_accounting=dict(accounting, prior_adam_steps=0)),
                         dict(budget_accounting=accounting)):
                with self.assertRaisesRegex(ValueError, 'accounting changed'):
                    campaign.checked_accounting(plan)

    def test_native_budget_uses_first_sealed_request_and_all_parent_rounds(self):
        with tempfile.TemporaryDirectory() as folder:
            plan = fixture(folder)
            path = Path(folder)/'lexically_first.json'
            campaign.write_json(path, {'metadata': {'trace': list(range(9))}})
            plan['parent_records']['a'] = str(path)
            limits = campaign.native_limits(plan)
            self.assertEqual(limits['draft_block'], 16384 + 256 + 12 + 3 + 4 + 256)
            self.assertEqual(limits['model_load'], 68)

    def test_shared_time_cap_is_exact_and_records_unfinished_work(self):
        with tempfile.TemporaryDirectory() as folder:
            clock = [0.]
            budget = campaign.CampaignBudget(Path(folder)/'budget.jsonl',
                limits={'draft_block': 5}, gpu_seconds=10800, clock=lambda: clock[0])
            budget.start_gpu(1)
            budget.attempt('draft_block', role='qualification')
            clock[0] = 10800.
            with self.assertRaisesRegex(BudgetExceeded, 'Three occupied'):
                budget.attempt('draft_block')
            self.assertEqual(budget.summary()['unknown'], {'draft_block': 1})
            self.assertEqual(budget.summary()['attempted'], {'draft_block': 1})

    def test_preflight_authenticates_controller_and_child_checkout_without_model_work(self):
        identity = dict(bytes=0, sha256='fixture')
        code = str(Path(campaign.__file__).resolve().parents[2])
        plan = dict(root='/data/sources/minhao/runs/cpu_fixture', constants=campaign.CONSTANTS,
            source_manifest='source.json', execution_commit='test-source', code_path=code,
            input_identities={'source.json': identity}, immutable_files={'population.json': identity},
            native_plan_path='native-plan.json', host_reserve_bytes=32*1024**3,
            storage={'maximum_additional_bytes': 240*1024**3, 'reserve_bytes': 8*1024**3})
        old = dict(checkpoints={name: {'export': '/'+name} for name in ('K4','K5','K6')},
                   pinned_files={'/K4/model.safetensors': identity, '/kept': identity})
        with patch.object(campaign, 'authenticate_source', return_value={'source_path': code}), \
             patch.object(campaign, 'identity', return_value=identity), \
             patch.object(campaign, 'read', return_value={'original_plan': old}), \
             patch('experiments.lost_repair.run.preflight', return_value={'status':'passed'}) as legacy, \
             patch('experiments.survival_confirmation.runtime.physical_gpu', return_value='GPU-fixture'), \
             patch.object(probe_client, 'resource_estimate', return_value={'conservative_precompaction_peak_bytes':1}), \
             patch.object(Path, 'read_text', return_value='MemAvailable: 200000000 kB\n'), \
             patch.object(campaign, 'capacity', return_value={'status':'passed'}), \
             patch.object(campaign, 'write_json'), patch.object(campaign.subprocess, 'Popen') as process:
            result = campaign.preflight(plan)
            self.assertTrue(result['no_model_work'])
            process.assert_not_called()
            supplied = legacy.call_args.args[0]
            self.assertEqual(set(supplied['checkpoints']), {'K5','K6'})
            self.assertNotIn('/K4/model.safetensors', supplied['pinned_files'])
            wrong = dict(plan, code_path='/different/checkout')
            with self.assertRaisesRegex(ValueError, 'authenticated execution source'):
                campaign.preflight(wrong)

    def test_training_environment_preserves_assignment_and_pins_interpreter_paths(self):
        plan = dict(serving_gpu=0, training_python='/env/training/bin/python', code_path='/source/DFlash-Online')
        with patch.dict(os.environ, {'PATH':'/old/bin', 'PYTHONPATH':'/stale/source'}, clear=True):
            env = campaign.training_environment(plan)
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '0')
            self.assertEqual(env['CONDA_PREFIX'], '/env/training')
            self.assertEqual(env['PATH'], '/env/training/bin:/old/bin')
            self.assertEqual(env['PYTHONPATH'], '/source/DFlash-Online:/source/DFlash-Online/training')
            os.environ['CUDA_VISIBLE_DEVICES'] = '1'
            with self.assertRaisesRegex(ValueError, 'Ambient CUDA assignment'):
                campaign.training_environment(plan)

    def test_child_launch_records_the_effective_environment(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {}, clear=True):
            plan = fixture(folder)
            plan.update(training_python='/env/training/bin/python', code_path='/source/DFlash-Online')
            campaign.write_json(Path(folder)/'training/RESULT.json', {'status':'completed'})
            process = SimpleNamespace(pid=111, returncode=0, wait=Mock(return_value=0))
            with patch.object(campaign.subprocess, 'Popen', return_value=process) as launch:
                result = campaign.launch_training(plan, time.monotonic()+60)
            self.assertEqual(result['status'], 'completed')
            self.assertEqual(launch.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'], '0')
            receipt = campaign.read(Path(folder)/'TRAINING_PROCESS.json')
            self.assertEqual(receipt['environment']['CONDA_PREFIX'], '/env/training')
            self.assertEqual(receipt['returncode'], 0)


class ReportingTests(unittest.TestCase):
    def test_retry_report_retains_failed_work_and_reports_cumulative_totals(self):
        with tempfile.TemporaryDirectory() as folder:
            plan = fixture(folder)
            plan['prior_attempt'] = prior_attempt_fixture(Path(folder)/'prior')
            local = prior_attempt_fixture(Path(folder)/'training', steps=4)
            # This helper creates a ledger only; the failed prior remains fixed
            # at three and local work is reported separately as four.
            Path(local['training_work_path']).rename(Path(folder)/'training/PHYSICAL_WORK.jsonl')
            clock = [0.]
            budget = campaign.CampaignBudget(Path(folder)/'work.jsonl', clock=lambda:clock[0])
            budget.start_gpu(1)
            clock[0] = 10.
            budget.end_gpu()
            result = campaign.report(plan, dict(status='failed', error='test stop', wall_s=10), budget)
            costs = campaign.read(Path(folder)/'WORK_AND_COSTS.json')
            self.assertEqual(costs['cumulative']['completed_adam_steps'], 7)
            self.assertEqual(costs['cumulative']['gpu_seconds'], 195.9867403432727)
            self.assertEqual(costs['cumulative']['adam_limit'], 171)
            self.assertEqual(costs['training']['completed']['optimizer_step'], 4)
            self.assertEqual(costs['budget_accounting']['prior_adam_steps'], 3)
            self.assertEqual(result['status'], 'incomplete')
            self.assertNotIn('primary', result)

    def test_incomplete_report_preserves_available_rows_without_efficacy_claim(self):
        with tempfile.TemporaryDirectory() as folder:
            plan = fixture(folder)
            root = Path(folder)
            (root/'probes').mkdir()
            (root/'probes/PROBE_OUTCOMES.jsonl').write_text(
                '{"request_id":"r","window":8,"state":"P","canonical_start":0,"round_ordinal":0,"valid":false}\n')
            budget = campaign.CampaignBudget(root/'work.jsonl')
            result = campaign.report(plan, dict(status='failed', error='native mismatch', wall_s=1), budget)
            self.assertEqual(result['status'], 'incomplete')
            self.assertEqual(result['observed_probe_rows'], 1)
            self.assertNotIn('primary', result)
            audit = campaign.read(root/'final_report/INDEPENDENT_REDUCTION.json')
            self.assertEqual(audit['status'], 'passed')
            self.assertFalse(audit['efficacy_audited'])

    def test_invalid_full_report_cannot_leave_campaign_completed(self):
        with tempfile.TemporaryDirectory() as folder:
            plan = fixture(folder)
            budget = campaign.CampaignBudget(Path(folder)/'work.jsonl')
            with self.assertRaisesRegex(ValueError, 'invalid scientific report'):
                campaign.report(plan, dict(status='completed', wall_s=1), budget)
            result = campaign.read(Path(folder)/'final_report/PRIMARY_RESULTS.json')
            self.assertEqual(result['status'], 'invalid')
            self.assertNotIn('primary', result)


class ControllerTests(unittest.TestCase):
    def mocks(self, stack):
        service = SimpleNamespace(connection={}, close=Mock())
        stack.enter_context(patch.object(campaign, 'BudgetService', return_value=service))
        stack.enter_context(patch.object(campaign.signal, 'setitimer'))
        stack.enter_context(patch.object(campaign.signal, 'signal'))
        preflight = stack.enter_context(patch.object(campaign, 'preflight', return_value={}))
        qualify = stack.enter_context(patch.object(probe_client, 'qualify', return_value={'status':'passed'}))
        training = stack.enter_context(patch.object(campaign, 'launch_training'))
        probes = stack.enter_context(patch.object(probe_client, 'run', return_value={'status':'completed'}))
        return service, preflight, qualify, training, probes

    def test_preflight_failure_runs_no_gpu_phase_and_returns_incomplete_report(self):
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            fixture(folder)
            service, preflight, qualify, training, probes = self.mocks(stack)
            preflight.side_effect = ValueError('admission failure')
            with self.assertRaisesRegex(ValueError, 'admission failure'):
                campaign.run(Path(folder)/'sealed/PLAN.json')
            for method in (qualify, training, probes):
                method.assert_not_called()
            result = campaign.read(Path(folder)/'final_report/PRIMARY_RESULTS.json')
            self.assertEqual(result['status'], 'incomplete')
            status = campaign.read(Path(folder)/'STATUS.json')
            self.assertEqual(status['budget']['gpu_seconds'], 0)
            service.close.assert_called_once()

    def test_retry_alarm_child_deadline_and_budget_deduct_prior_time(self):
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            plan = fixture(folder)
            plan['prior_attempt'] = prior_attempt_fixture(Path(folder)/'prior')
            plan['budget_accounting'] = campaign.attempt_accounting(plan['prior_attempt'])
            campaign.write_json(Path(folder)/'sealed/PLAN.json', plan)
            _, _, _, training, _ = self.mocks(stack)
            training.return_value = dict(probe_exports=[])
            stack.enter_context(patch.object(campaign, 'report', return_value={'status':'completed'}))
            stack.enter_context(patch.object(campaign.time, 'monotonic', return_value=100.))
            with patch.object(campaign.signal, 'setitimer') as alarm:
                result = campaign.run(Path(folder)/'sealed/PLAN.json')
            remaining = 10614.013259656727
            self.assertEqual(training.call_args.args[1], 100.+remaining)
            alarm.assert_any_call(campaign.signal.ITIMER_REAL, remaining)
            self.assertEqual(result['budget']['gpu_seconds_limit'], remaining)
            self.assertGreaterEqual(result['cumulative_gpu_seconds'], 185.9867403432727)

    def test_reporting_failure_does_not_mask_original_execution_failure(self):
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            fixture(folder)
            service, _, qualify, training, probes = self.mocks(stack)
            qualify.side_effect = ValueError('original native failure')
            stack.enter_context(patch.object(campaign, 'report', side_effect=RuntimeError('report failed')))
            with self.assertRaisesRegex(ValueError, 'original native failure'):
                campaign.run(Path(folder)/'sealed/PLAN.json')
            status = campaign.read(Path(folder)/'STATUS.json')
            self.assertEqual(status['status'], 'failed')
            self.assertIn('original native failure', status['error'])
            self.assertIn('report failed', status['report_error'])
            training.assert_not_called()
            probes.assert_not_called()
            service.close.assert_called_once()

    def test_success_orders_disposable_gate_training_and_all_export_probes(self):
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            plan = fixture(folder)
            root = Path(folder)
            weight = root/'checkpoint/model.safetensors'
            weight.parent.mkdir()
            weight.write_bytes(b'fixture weights')
            exports = [dict(window=w, state=s, checkpoint=str(weight.parent),
                            weights_identity=campaign.identity(weight),
                            reused_from='N' if s in ('U','T') else None)
                       for w in probe_client.WINDOWS for s in probe_client.STATES]
            probe_client.validate_exports(exports)
            service, _, qualify, training, probes = self.mocks(stack)
            order = []
            qualify.side_effect = lambda *a, **k: order.append('closed qualification') or {'status':'passed'}
            training.side_effect = lambda *a: order.append('closed training') or {'status':'completed','probe_exports':exports}
            probes.side_effect = lambda *a, **k: order.append('probes') or {'status':'completed'}
            stack.enter_context(patch.object(campaign, 'report', return_value={'status':'completed'}))
            result = campaign.run(root/'sealed/PLAN.json')
            self.assertEqual(order, ['closed qualification','closed training','probes'])
            self.assertEqual(probes.call_args.args[1], exports)
            self.assertEqual(result['status'], 'completed')
            self.assertEqual(result['phase'], 'complete')
            service.close.assert_called_once()

    def test_tampered_export_prevents_all_main_probes(self):
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            fixture(folder)
            path = Path(folder)/'checkpoint/model.safetensors'
            path.parent.mkdir()
            path.write_bytes(b'changed')
            service, _, _, training, probes = self.mocks(stack)
            training.return_value = dict(probe_exports=[dict(checkpoint=str(path.parent),
                weights_identity={'bytes':0,'sha256':'not the written weights'})])
            with self.assertRaisesRegex(ValueError, 'Window export changed'):
                campaign.run(Path(folder)/'sealed/PLAN.json')
            probes.assert_not_called()
            self.assertEqual(campaign.read(Path(folder)/'STATUS.json')['status'], 'failed')
            service.close.assert_called_once()

    def test_report_failure_after_completed_execution_marks_status_failed(self):
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            fixture(folder)
            service, _, _, training, _ = self.mocks(stack)
            training.return_value = dict(probe_exports=[])
            stack.enter_context(patch.object(campaign, 'report', side_effect=ValueError('invalid scientific report')))
            with self.assertRaisesRegex(ValueError, 'invalid scientific report'):
                campaign.run(Path(folder)/'sealed/PLAN.json')
            status = campaign.read(Path(folder)/'STATUS.json')
            self.assertEqual(status['status'], 'failed')
            self.assertIn('invalid scientific report', status['report_error'])
            service.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
