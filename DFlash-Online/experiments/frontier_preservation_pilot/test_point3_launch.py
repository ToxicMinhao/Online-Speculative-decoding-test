"""Point3 CPU work-accounting checks, including interrupted reservations."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from .point3_launch import Point3StepLedger, boundary, launch
from .point3_prepare import CASES, HISTORICAL_CARD_S, prospective_contract


class Point3LedgerTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/'ledger.json'
        self.ledger = Point3StepLedger(self.path, {'sha256':'test'})

    def qualify(self):
        self.ledger.start_attempt(1)
        for case in CASES:
            number = self.ledger.reserve(case)
            self.ledger.complete(number)
        for index in range(28):
            number = self.ledger.reserve_token(str(index))
            self.ledger.complete_token(number)
        self.ledger.finish_attempt(1, True)

    def test_qualification_then_both_events_have_exact_budgets(self):
        self.qualify()
        for arm in CASES[1:]:
            for window in range(1,129):
                number = self.ledger.reserve(arm, window=window)
                self.ledger.complete(number)
        self.assertEqual(len(self.ledger.value['calls']),259)
        self.assertEqual(len(self.ledger.value['token_backwards']),28)
        with self.assertRaises(RuntimeError):
            self.ledger.reserve('F_C_MAX',window=129)

    def test_unknown_native_call_stays_charged_after_reload(self):
        self.ledger.start_attempt(1)
        self.ledger.reserve('C')
        reloaded=Point3StepLedger(self.path,{'sha256':'test'})
        self.assertEqual(reloaded.value['calls'][0]['status'],'reserved')
        with self.assertRaises(RuntimeError):
            reloaded.reserve('C')

    def test_failed_production_window_cannot_be_retried(self):
        self.qualify()
        self.ledger.reserve('F_C_SUM',window=1)
        for window in (1,2):
            with self.assertRaises(RuntimeError):
                self.ledger.reserve('F_C_SUM',window=window)
        with self.assertRaises(ValueError):
            self.ledger.start_attempt(2,'cannot reset after production')

    def test_production_requires_completed_qualification(self):
        self.ledger.start_attempt(1)
        with self.assertRaises(ValueError):
            self.ledger.reserve('F_C_SUM',window=1)

    def test_retry_requires_failure_note_and_never_discards_work(self):
        self.ledger.start_attempt(1)
        self.ledger.reserve('C')
        self.ledger.finish_attempt(1,False)
        with self.assertRaises(ValueError):
            self.ledger.start_attempt(2)
        self.ledger.start_attempt(2,'identified implementation fix')
        self.ledger.reserve('C')
        self.assertEqual(len(self.ledger.value['calls']),2)
        self.ledger.finish_attempt(2,False)
        with self.assertRaises(ValueError):
            self.ledger.start_attempt(3,'not allowed')

    def test_oracle_cap_includes_uncompleted_calls(self):
        self.ledger.start_attempt(1)
        for index in range(30):
            self.ledger.reserve_token(str(index))
        with self.assertRaises(RuntimeError):
            self.ledger.reserve_token('extra')
        self.ledger.finish_attempt(1,False)
        self.ledger.start_attempt(2,'fix')
        for index in range(30):
            self.ledger.reserve_token(str(index))
        self.assertEqual(len(self.ledger.value['token_backwards']),60)
        with self.assertRaises(RuntimeError):
            self.ledger.reserve_token('extra')

    def test_missing_or_unknown_work_cannot_pass_qualification(self):
        self.ledger.start_attempt(1)
        for case in CASES:
            self.ledger.complete(self.ledger.reserve(case))
        for index in range(28):
            number=self.ledger.reserve_token(str(index))
            if index != 27:
                self.ledger.complete_token(number)
        with self.assertRaises(ValueError):
            self.ledger.finish_attempt(1,True)

    def test_changed_binding_cannot_reset_budget(self):
        with self.assertRaises(ValueError):
            Point3StepLedger(self.path,{'sha256':'different'})

    def test_blocked_plan_cannot_create_clock_or_launch_model(self):
        plan=prospective_contract()
        plan.update(production_admitted=False,resource_admission={'admitted':False})
        path=Path(self.temp.name)/'plan.json'
        path.write_text(json.dumps(plan))
        with self.assertRaises(ValueError):
            launch(path)
        self.assertFalse((Path(self.temp.name)/'CLOCK.json').exists())


class Point3SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.plan = prospective_contract()
        self.plan.update(root=str(self.root), source_root=str(self.root),
            source_manifest=str(self.root/'SOURCE.json'), training_python='unused-python',
            devices={'A':'gpu-a', 'B':'gpu-b'}, production_admitted=True,
            resource_admission={'admitted':True},
            compiler_baseline=dict(directories={'TRITON_CACHE_DIR':'/cache/triton'},
                                   allocated_bytes=2_000_000_000),
            forecast={'storage':{'remote_peak_bytes':20_000_000_000,
                                 'local_allowance_bytes':600_000_000}})
        self.path=self.root/'CONTRACT.json'
        self.path.write_text(json.dumps(self.plan))
        self.inventory=dict(processes='', cards=[
            dict(uuid=card,total_bytes=80_000_000_000,used_bytes=0)
            for card in ('gpu-a','gpu-b')])

    def environment(self):
        """Mock every external process/resource query; launch no Python child."""
        from contextlib import ExitStack
        stack=ExitStack()
        stack.enter_context(patch('experiments.frontier_preservation_pilot.qualify.verify_source'))
        stack.enter_context(patch('experiments.frontier_preservation_pilot.point1_prepare.gpu_inventory',
                                  return_value=self.inventory))
        stack.enter_context(patch('experiments.frontier_preservation_pilot.point1_prepare.host_memory',
                                  return_value={'available_bytes':100_000_000_000}))
        stack.enter_context(patch('experiments.frontier_preservation_pilot.point3_launch.shutil.disk_usage',
                                  return_value=SimpleNamespace(free=430_000_000_000)))
        stack.enter_context(patch('experiments.kplus_full_gsm8k.storage.compiler_snapshot',
                                  return_value=self.plan['compiler_baseline']))
        return stack

    def test_preflight_rejection_does_not_consume_qualification_or_clock(self):
        for attempt in range(2):
            with self.environment(), \
                 patch('experiments.frontier_preservation_pilot.point1_prepare.gpu_inventory',
                       return_value=dict(self.inventory, processes='another job')), \
                 patch('experiments.frontier_preservation_pilot.point3_launch.subprocess.Popen') as spawn:
                with self.assertRaisesRegex(RuntimeError, 'Both cards must be quiet'):
                    launch(self.path)
                spawn.assert_not_called()
            self.assertFalse((self.root/'STEP_LEDGER.json').exists())
            self.assertFalse((self.root/'CLOCK.json').exists())
            failures = [json.loads(line) for line in
                        (self.root/'PREFLIGHT_FAILURES.jsonl').read_text().splitlines()]
            self.assertEqual(len(failures), attempt+1)
            self.assertTrue(all(row['first_gpu_action_epoch'] is None
                                and row['completed_card_s'] == 0 for row in failures))
        # A later successful preflight still starts the original first attempt.
        with self.environment(), \
             patch('experiments.frontier_preservation_pilot.point3_launch.boundary',
                   side_effect=RuntimeError('stop before any child')):
            with self.assertRaisesRegex(RuntimeError, 'stop before any child'):
                launch(self.path)
        ledger = json.loads((self.root/'STEP_LEDGER.json').read_text())
        self.assertEqual([row['number'] for row in ledger['attempts']], [1])
        self.assertEqual(ledger['calls'], [])

    def test_later_preflight_block_preserves_finished_phase_cost(self):
        child = SimpleNamespace(pid=123, returncode=0, poll=lambda:0)
        with self.environment(), \
             patch('experiments.frontier_preservation_pilot.point3_launch.boundary', return_value={}), \
             patch('experiments.frontier_preservation_pilot.point3_launch.subprocess.Popen', return_value=child) as spawn, \
             patch.object(Point3StepLedger, 'finish_attempt'), \
             patch('experiments.frontier_preservation_pilot.point3_launch.phase_preflight',
                   side_effect=['B', RuntimeError('host memory changed')]):
            with self.assertRaisesRegex(RuntimeError, 'host memory changed'):
                launch(self.path)
        self.assertEqual(spawn.call_count, 1)
        clock = json.loads((self.root/'CLOCK.json').read_text())
        self.assertEqual(clock['status'], 'blocked_before_phase')
        self.assertEqual(clock['phases'][0]['status'], 'completed')
        self.assertEqual(clock['completed_card_s'], clock['phases'][0]['wall_s'])
        self.assertEqual(clock['preflight_failure']['phase'], 'train')

    def test_changed_cleanup_receipt_stops_before_work(self):
        receipt = self.root/'CLEANUP_RECEIPT.json'
        receipt.write_text('{}')
        self.plan['prelaunch_cleanup'] = dict(path=str(receipt), identity={'sha256':'changed'})
        self.path.write_text(json.dumps(self.plan))
        with self.environment(), \
             patch('experiments.frontier_preservation_pilot.point3_launch.subprocess.Popen') as spawn:
            with self.assertRaisesRegex(ValueError, 'cleanup receipt changed'):
                launch(self.path)
            spawn.assert_not_called()
        self.assertFalse((self.root/'STEP_LEDGER.json').exists())

    def test_live_storage_block_prevents_popen(self):
        with self.environment(), \
             patch('experiments.frontier_preservation_pilot.point3_launch.boundary',
                   side_effect=RuntimeError('Storage changed')), \
             patch('experiments.frontier_preservation_pilot.point3_launch.subprocess.Popen') as spawn:
            with self.assertRaisesRegex(RuntimeError,'Storage changed'):
                launch(self.path)
            spawn.assert_not_called()

    def test_qualification_finalization_failure_still_saves_clock_cost(self):
        child=SimpleNamespace(pid=123, returncode=0, poll=lambda:0)
        with self.environment(), \
             patch('experiments.frontier_preservation_pilot.point3_launch.boundary',return_value={}), \
             patch('experiments.frontier_preservation_pilot.point3_launch.subprocess.Popen',return_value=child):
            with self.assertRaisesRegex(ValueError,'missing/unknown work'):
                launch(self.path)
        clock=json.loads((self.root/'CLOCK.json').read_text())
        self.assertEqual(clock['status'],'failed')
        self.assertEqual(clock['phases'][0]['status'],'failed')
        self.assertIn('wall_s',clock['phases'][0])
        self.assertEqual(clock['completed_card_s'],clock['phases'][0]['wall_s'])
        self.assertEqual(clock['cumulative_card_s'],HISTORICAL_CARD_S+clock['completed_card_s'])
        ledger=json.loads((self.root/'STEP_LEDGER.json').read_text())
        self.assertEqual(ledger['attempts'][0]['status'],'failed')

    def test_unexpected_monitor_failure_stops_only_owned_child(self):
        child=SimpleNamespace(pid=123,returncode=None,poll=lambda:None)
        with self.environment(), \
             patch('experiments.frontier_preservation_pilot.point3_launch.boundary',return_value={}), \
             patch('experiments.frontier_preservation_pilot.point3_launch.subprocess.Popen',return_value=child), \
             patch('experiments.frontier_preservation_pilot.point3_launch.time.sleep',
                   side_effect=OSError('monitor failed')), \
             patch('experiments.repair_conditional_kl_pilot.launch.stop_owned') as stop:
            with self.assertRaisesRegex(OSError,'monitor failed'):
                launch(self.path)
            stop.assert_called_with(child)
        clock=json.loads((self.root/'CLOCK.json').read_text())
        self.assertEqual(clock['status'],'failed')

    def test_shutdown_error_does_not_hide_original_failure_or_clock(self):
        child=SimpleNamespace(pid=123,returncode=None,poll=lambda:None)
        with self.environment(), \
             patch('experiments.frontier_preservation_pilot.point3_launch.boundary',return_value={}), \
             patch('experiments.frontier_preservation_pilot.point3_launch.subprocess.Popen',return_value=child), \
             patch('experiments.frontier_preservation_pilot.point3_launch.time.sleep',
                   side_effect=OSError('monitor failed')), \
             patch('experiments.repair_conditional_kl_pilot.launch.stop_owned',
                   side_effect=RuntimeError('shutdown unknown')):
            with self.assertRaisesRegex(OSError,'monitor failed'):
                launch(self.path)
        clock=json.loads((self.root/'CLOCK.json').read_text())
        self.assertEqual(clock['status'],'failed')
        phase=clock['phases'][0]
        self.assertEqual(phase['status'],'failed')
        self.assertIn('monitor failed',phase['error'])
        self.assertIn('shutdown unknown',phase['shutdown_error'])
        self.assertEqual(clock['completed_card_s'],phase['wall_s'])

    def test_boundary_charges_active_phase_and_retains_historical_cost_separately(self):
        clock=dict(first_gpu_action_epoch=100.,completed_card_s=200.,
                   phases=[dict(status='running',started_epoch=900.)])
        (self.root/'CLOCK.json').write_text(json.dumps(clock))
        with patch('experiments.frontier_preservation_pilot.point3_launch.time.time',return_value=1000.), \
             patch('experiments.kplus_full_gsm8k.storage.compiler_snapshot',
                   return_value=self.plan['compiler_baseline']), \
             patch('experiments.bounded_prefix_allocation.runtime.allocated_bytes',return_value=500_000_000), \
             patch('experiments.frontier_preservation_pilot.point3_launch.shutil.disk_usage',
                   return_value=SimpleNamespace(free=420_000_000_000)):
            result=boundary(self.plan)
        self.assertEqual(result['elapsed_s'],900.)
        self.assertEqual(result['occupied_card_s'],300.)
        self.assertEqual(result['cumulative_card_s'],HISTORICAL_CARD_S+300.)
        self.assertEqual(result['charged_bytes'],1_100_000_000)

    def test_boundary_cannot_reset_clock_or_ignore_remaining_writes(self):
        cases=[(100.,14300.,420_000_000_000,TimeoutError),
               (-21000.,0.,420_000_000_000,TimeoutError),
               (100.,0.,407_000_000_000,RuntimeError)]
        for first,occupied,free,error in cases:
            with self.subTest(first=first,occupied=occupied,free=free):
                clock=dict(first_gpu_action_epoch=first,completed_card_s=occupied,
                           phases=[dict(status='running',started_epoch=900.)])
                (self.root/'CLOCK.json').write_text(json.dumps(clock))
                with patch('experiments.frontier_preservation_pilot.point3_launch.time.time',return_value=1000.), \
                     patch('experiments.kplus_full_gsm8k.storage.compiler_snapshot',
                           return_value=self.plan['compiler_baseline']), \
                     patch('experiments.bounded_prefix_allocation.runtime.allocated_bytes',return_value=500_000_000), \
                     patch('experiments.frontier_preservation_pilot.point3_launch.shutil.disk_usage',
                           return_value=SimpleNamespace(free=free)):
                    with self.assertRaises(error):
                        boundary(self.plan)

    def test_boundary_charges_external_compiler_growth_against_global_cap(self):
        clock = dict(first_gpu_action_epoch=900., completed_card_s=0.,
                     phases=[dict(status='running', started_epoch=900.)])
        (self.root/'CLOCK.json').write_text(json.dumps(clock))
        compiler = dict(self.plan['compiler_baseline'], allocated_bytes=7_000_000_001)
        with patch('experiments.frontier_preservation_pilot.point3_launch.time.time', return_value=1000.), \
             patch('experiments.bounded_prefix_allocation.runtime.allocated_bytes', return_value=19_400_000_000), \
             patch('experiments.kplus_full_gsm8k.storage.compiler_snapshot', return_value=compiler), \
             patch('experiments.frontier_preservation_pilot.point3_launch.shutil.disk_usage',
                   return_value=SimpleNamespace(free=450_000_000_000)):
            with self.assertRaisesRegex(RuntimeError, 'peak storage/reserve'):
                boundary(self.plan)

    def test_cache_growth_consumes_planned_allowance_but_shrinkage_gives_no_credit(self):
        clock = dict(first_gpu_action_epoch=900., completed_card_s=0.,
                     phases=[dict(status='running', started_epoch=900.)])
        (self.root/'CLOCK.json').write_text(json.dumps(clock))
        for allocated, growth in ((3_000_000_000, 1_000_000_000), (1, 0)):
            compiler = dict(self.plan['compiler_baseline'], allocated_bytes=allocated)
            with self.environment(), \
                 patch('experiments.frontier_preservation_pilot.point3_launch.time.time', return_value=1000.), \
                 patch('experiments.bounded_prefix_allocation.runtime.allocated_bytes', return_value=500_000_000), \
                 patch('experiments.kplus_full_gsm8k.storage.compiler_snapshot', return_value=compiler):
                result = boundary(self.plan)
            self.assertEqual(result['compiler_growth_bytes'], growth)
            self.assertEqual(result['charged_bytes'], 1_100_000_000+growth)
            self.assertEqual(result['remaining_peak_bytes'], 19_500_000_000-growth)

    def test_changed_compiler_locations_stop_instead_of_erasing_growth(self):
        clock = dict(first_gpu_action_epoch=900., completed_card_s=0.,
                     phases=[dict(status='running', started_epoch=900.)])
        (self.root/'CLOCK.json').write_text(json.dumps(clock))
        with self.environment(), \
             patch('experiments.frontier_preservation_pilot.point3_launch.time.time', return_value=1000.), \
             patch('experiments.kplus_full_gsm8k.storage.compiler_snapshot',
                   return_value=dict(directories={}, allocated_bytes=0)):
            with self.assertRaisesRegex(ValueError, 'cache locations changed'):
                boundary(self.plan)


if __name__=='__main__':
    unittest.main()
