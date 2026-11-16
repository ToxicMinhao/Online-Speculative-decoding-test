"""CPU launch gates and exact qualification carryover; no child model work."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from experiments.persistent_prefix_withholding import launch as L
from experiments.persistent_prefix_withholding import campaign as C
from experiments.persistent_prefix_withholding import qualification as Q
from experiments.persistent_prefix_withholding.resources import LIMITS, CARD_SECONDS
from online_feedback.storage import write_json


def training_receipts():
    result = {}
    for slot, steps in [('A', 5), ('B', 3)]:
        counts = dict(model_load=1, optimizer_step=steps,
                      response_forward=2 + 2*steps, response_backward=2 + 2*steps)
        result[slot] = dict(status='passed', qualification=dict(status='passed'),
                           physical_work=dict(attempted=counts, completed=dict(counts), failed={}))
    return result


def serving_receipts():
    return {slot: dict(status='passed', qualification=dict(status='passed',
        planned_generations=4, attempted_generations=4, returned_generations=4,
        valid_generations=4, invalid_generations=0, unknown_generations=0,
        unattempted_generations=0)) for slot in ('A', 'B')}


class LauncherTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.plan = dict(root=str(self.root), execution_commit='c'*40, source_root=str(self.root),
            training_python='/qualified/python', worker_python='/serving/python',
            compiler_root=str(self.root/'cache'), devices={'A':'GPU-A','B':'GPU-B'})
        self.plan_path = self.root/'PLAN.json'
        write_json(self.plan_path, self.plan)

    def clock(self):
        write_json(self.root/'CLOCK.json', dict(first_model_load_epoch=100., completed_card_s=10.,
                                              active={}, finished=[]))

    def test_exact_physical_work_carryover_includes_startup_and_no_main_work(self):
        L.seed_production_work(self.root, training_receipts(), serving_receipts())
        work = json.loads((self.root/'WORK.json').read_text())
        expected = {key: 0 for key in LIMITS}
        expected.update(adam=8, response_forward=20, response_backward=20,
                        training_initialization=2, qualification_generation=8)
        self.assertEqual(work['attempted'], expected)
        self.assertEqual(work['completed'], expected)
        self.assertEqual(work['pending'], {})
        self.assertEqual(work['next_operation'], 0)
        self.assertEqual(sum(expected[k] for k in ('feedback_generation','reference_generation',
                                                 'warmup_generation','scored_generation')), 0)

    def test_failed_unknown_and_wrong_budget_qualification_are_not_seeded(self):
        for mutation in ('failed', 'unknown', 'extra_step', 'missing_startup'):
            training = training_receipts()
            work = training['A']['physical_work']
            if mutation == 'failed':
                work['failed']['response_forward'] = 1
            elif mutation == 'unknown':
                work['completed']['response_forward'] -= 1
            else:
                key, delta = ('optimizer_step', 1) if mutation == 'extra_step' else ('response_forward', -2)
                work['attempted'][key] += delta
                work['completed'][key] += delta
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                L.seed_production_work(self.root, training, serving_receipts())
            self.assertFalse((self.root/'WORK.json').exists())
        serving = serving_receipts()
        serving['B']['qualification']['valid_generations'] = 3
        with self.assertRaises(ValueError):
            L.seed_production_work(self.root, training_receipts(), serving)

    def test_environment_keeps_single_uuid_and_removes_distributed_state(self):
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES='other', WORLD_SIZE='2', LOCAL_RANK='1',
                        RANK='1', MASTER_ADDR='external', MASTER_PORT='1234',
                        CC='unqualified-cc', CXX='unqualified-cxx', OMP_NUM_THREADS='7',
                        LD_LIBRARY_PATH='/unchanged/library'):
            cpu = L.child_environment(self.plan)
            gpu = L.child_environment(self.plan, 'B')
        self.assertEqual(cpu['CUDA_VISIBLE_DEVICES'], '')
        self.assertEqual(gpu['CUDA_VISIBLE_DEVICES'], 'GPU-B')
        self.assertTrue(gpu['PYTHONPATH'].startswith(str(self.root)+os.pathsep))
        self.assertFalse(set(('WORLD_SIZE','LOCAL_RANK','RANK','MASTER_ADDR','MASTER_PORT')) & set(gpu))
        for environment in (cpu, gpu):
            for name, path in L.QUALIFIED_COMPILERS.items():
                self.assertEqual(environment[name], path)
            self.assertEqual(environment['OMP_NUM_THREADS'], '7')
            self.assertEqual(environment['LD_LIBRARY_PATH'], '/unchanged/library')

    def test_qualified_compilers_must_exist_and_be_executable(self):
        with patch.object(L.Path, 'is_file', return_value=True), \
                patch.object(L.os, 'access', return_value=True) as access:
            self.assertEqual(L.check_compilers(), L.QUALIFIED_COMPILERS)
        self.assertEqual([call.args for call in access.call_args_list],
                         [(path, os.X_OK) for path in L.QUALIFIED_COMPILERS.values()])
        for missing, executable in ((False,True), (True,False)):
            with patch.object(L.Path, 'is_file', return_value=missing), \
                    patch.object(L.os, 'access', return_value=executable):
                with self.assertRaisesRegex(FileNotFoundError, 'Qualified compiler'):
                    L.check_compilers()

    def test_missing_compiler_stops_before_gpu_worker_or_card_lease(self):
        with patch.object(L, 'check_compilers', side_effect=FileNotFoundError('compiler')), \
                patch.object(L.subprocess, 'Popen') as worker, patch.object(L, 'occupied_card') as lease:
            with self.assertRaisesRegex(FileNotFoundError, 'compiler'):
                L.run_qualification_child(self.plan_path, self.plan, 'serving', 'A')
        worker.assert_not_called()
        lease.assert_not_called()

    def test_failed_cpu_preflight_launches_no_qualification_or_campaign(self):
        with patch.object(L.subprocess, 'run', side_effect=subprocess.CalledProcessError(1, 'cpu')) as cpu, \
             patch.object(L, 'run_qualification_child') as qualify, patch.object(C, 'run_campaign') as main:
            with self.assertRaises(subprocess.CalledProcessError):
                L.run(self.plan_path)
        cpu.assert_called_once()
        qualify.assert_not_called()
        main.assert_not_called()
        status = json.loads((self.root/'LAUNCH.json').read_text())
        self.assertEqual(status['status'], 'stopped')
        self.assertEqual(status['completed_phases'], [])
        self.assertFalse(status['automatic_retry'])
        self.assertFalse((self.root/'CLOCK.json').exists())
        self.assertFalse((self.root/'WORK.json').exists())

    def test_failed_native_gate_prevents_later_gpu_or_main_work(self):
        with patch.object(L.subprocess, 'run'), patch.object(L.subprocess, 'check_output', return_value='0 cache\n'), \
             patch.object(L, 'run_qualification_child', side_effect=RuntimeError('A gate failed')) as qualify, \
             patch.object(C, 'run_campaign') as main:
            with self.assertRaisesRegex(RuntimeError, 'A gate failed'):
                L.run(self.plan_path)
        qualify.assert_called_once()
        self.assertEqual(qualify.call_args.args[-2:], ('training', 'A'))
        main.assert_not_called()
        status = json.loads((self.root/'LAUNCH.json').read_text())
        self.assertEqual(status['completed_phases'], ['cpu_preflight'])
        self.assertEqual(status['status'], 'stopped')
        self.assertTrue((self.root/'CLOCK.json').exists())
        self.assertFalse((self.root/'WORK.json').exists())

    def test_cross_card_failure_prevents_serving_qualification_and_main(self):
        training = training_receipts()
        def child(path, plan, kind, slot):
            self.assertEqual(kind, 'training')
            return training[slot]
        with patch.object(L.subprocess, 'run'), patch.object(L.subprocess, 'check_output', return_value='0 cache\n'), \
             patch.object(L, 'run_qualification_child', side_effect=child) as qualify, \
             patch.object(Q, 'compare_cards', side_effect=ValueError('cross-card mismatch')), \
             patch.object(C, 'run_campaign') as main:
            with self.assertRaisesRegex(ValueError, 'cross-card mismatch'):
                L.run(self.plan_path)
        self.assertEqual(qualify.call_count, 2)
        main.assert_not_called()
        self.assertFalse((self.root/'WORK.json').exists())

    def test_deadline_terminates_only_owned_child_and_releases_card(self):
        (self.root/'logs').mkdir()
        self.clock()
        child = Mock(pid=987654, returncode=None)
        child.poll.return_value = None
        with patch.object(L.subprocess, 'Popen', return_value=child), \
             patch.object(L, 'check_compilers'), \
             patch.object(L, 'clock_status', return_value=dict(occupied_s=CARD_SECONDS, elapsed_s=0)), \
             patch.object(L.os, 'killpg') as kill:
            with self.assertRaisesRegex(TimeoutError, 'time cap'):
                L.run_qualification_child(self.plan_path, self.plan, 'training', 'A')
        kill.assert_called_once_with(child.pid, L.signal.SIGTERM)
        child.wait.assert_called_once_with(timeout=20)
        clock = json.loads((self.root/'CLOCK.json').read_text())
        self.assertEqual(clock['active'], {})
        self.assertEqual(clock['finished'][-1]['task'], 'training_A')
        self.assertEqual(clock['finished'][-1]['physical_gpu_uuid'], 'GPU-A')
        self.assertFalse((self.root/'WORK.json').exists())

    def test_return_code_zero_does_not_override_failed_receipt(self):
        (self.root/'logs').mkdir()
        self.clock()
        write_json(self.root/'qualification/A/STATUS.json', dict(status='failed'))
        child = Mock(pid=987654, returncode=0)
        child.poll.return_value = 0
        with patch.object(L.subprocess, 'Popen', return_value=child), patch.object(L, 'check_compilers'):
            with self.assertRaisesRegex(RuntimeError, 'Unpassed native'):
                L.run_qualification_child(self.plan_path, self.plan, 'training', 'A')

    def test_existing_launch_is_not_restarted(self):
        write_json(self.root/'LAUNCH.json', dict(status='stopped', evidence='preserved'))
        with patch.object(L.subprocess, 'run') as launch, self.assertRaises(FileExistsError):
            L.run(self.plan_path)
        launch.assert_not_called()
        self.assertEqual(json.loads((self.root/'LAUNCH.json').read_text())['evidence'], 'preserved')


if __name__ == '__main__':
    unittest.main()
