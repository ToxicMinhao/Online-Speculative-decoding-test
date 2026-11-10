"""CPU admission and watchdog fixtures; no subprocess runs a model."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from experiments.verification_transition import preflight
from experiments.verification_transition import run
from experiments.verification_transition.budget import WorkBudget
from experiments.verification_transition.contract import scientific_config
from experiments.verification_transition.data import identity, verify_seal


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.code = self.root / 'code'
        self.code.mkdir()
        (self.code / 'module.py').write_text('value = 1\n')
        target = self.root / 'target'
        target.mkdir()
        (target / 'config.json').write_text('{}\n')
        runtime = self.root / 'runtime'
        runtime_file = runtime / 'lib/python3.12/site-packages/sglang/srt/native.py'
        runtime_file.parent.mkdir(parents=True)
        runtime_file.write_text('# pinned CPU fixture\n')
        raw = self.root / 'native.json'
        raw.write_text('{}\n')
        checkpoints = {}
        for dataset in ('gsm8k', 'mathqa'):
            checkpoints[dataset] = {}
            for name in ('KPLUS4', 'KPLUS6'):
                export = self.root / dataset / name / 'export'
                export.mkdir(parents=True)
                weights = export / 'model.safetensors'
                weights.write_bytes((dataset + name).encode())
                checkpoints[dataset][name] = dict(export=str(export), weights=str(weights),
                                                  weights_identity=identity(weights))
        requests = [dict(dataset=d, request_id=f'{d}_{i}', selection_rank=i,
                         request=dict(prompt_ids=[1], max_new_tokens=2048),
                         canonical=dict(prompt_ids=[1], response_ids=[2, 3]))
                    for i in range(64) for d in ('gsm8k', 'mathqa')]
        self.plan = dict(config=scientific_config(), requests=requests,
                         checkpoints=checkpoints, bound_native_files={str(raw): identity(raw)},
                         shared=dict(target_path=str(target), target_files={'config.json': identity(target/'config.json')},
                                     runtime=dict(python=str(runtime/'bin/python'), python_version='3.12.14',
                                                  native_runtime_sha256={'native.py': identity(runtime_file)['sha256']})))

    def verify(self, plan=None, *, free=20 * 1024**3, dirty=''):
        def git(args, **kwargs):
            return dirty if 'status' in args else 'fixture-commit\n'
        with patch.object(preflight.shutil, 'disk_usage', return_value=SimpleNamespace(free=free)), \
                patch.object(preflight.subprocess, 'check_output', side_effect=git):
            return preflight.verify(self.plan if plan is None else plan,
                                    '/data/sources/minhao/runs/cpu-fixture', self.code)

    def test_preflight_authenticates_inputs_without_model_work(self):
        receipt = self.verify()
        self.assertEqual(receipt['status'], 'passed')
        self.assertEqual(receipt['new_model_work'], 0)
        self.assertEqual(len(receipt['artifact_checks']), 7)
        self.assertEqual(receipt['source_files']['module.py'], identity(self.code/'module.py'))

    def test_changed_weights_stop_before_admission(self):
        path = Path(self.plan['checkpoints']['gsm8k']['KPLUS4']['weights'])
        path.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Checkpoint missing or changed'):
            self.verify()

    def test_insufficient_peak_storage_and_dirty_source_fail(self):
        for free, dirty, error in ((10 * 1024**3 - 1, '', RuntimeError),
                                   (20 * 1024**3, ' M file.py\n', ValueError)):
            with self.subTest(free=free, dirty=dirty), self.assertRaises(error):
                self.verify(free=free, dirty=dirty)

    def test_request_order_and_configuration_are_fixed(self):
        reordered = deepcopy(self.plan)
        reordered['requests'][:2] = reversed(reordered['requests'][:2])
        changed = deepcopy(self.plan)
        changed['config']['limits']['draft_block'] += 1
        for candidate in (reordered, changed):
            with self.subTest(candidate=candidate['config']['limits']), self.assertRaises(ValueError):
                self.verify(candidate)

    def test_output_root_must_be_authorized(self):
        with self.assertRaisesRegex(ValueError, 'approved A800 run storage'):
            preflight.verify(self.plan, self.root/'unapproved', self.code)


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.inputs = self.root/'INPUTS.json'
        self.plan = dict(shared={'runtime': {'python': 'no-model-python'}}, requests=[],
                         checkpoints={d:{k:dict(weights='fixture', weights_identity={'bytes': 1, 'sha256': 'x'})
                                           for k in ('KPLUS4', 'KPLUS6')}
                                      for d in ('gsm8k', 'mathqa')})
        self.inputs.write_text(json.dumps(self.plan))

    def test_time_cap_stops_worker_and_preserves_incomplete_report(self):
        output = self.root/'out'
        child = Mock(pid=987654, returncode=-9)
        child.wait.side_effect = [subprocess.TimeoutExpired('cpu-fixture', 1), -9]
        child.poll.return_value = -9
        def launch(*args, **kwargs):
            ledger = [json.loads(line) for line in (output/'WORK.jsonl').read_text().splitlines()]
            self.assertEqual(ledger[-1]['kind'], 'gpu_start')
            self.assertTrue(kwargs['start_new_session'])
            return child
        with patch.object(run, 'verify_seal', return_value=deepcopy(self.plan)), \
                patch.object(run, 'active_owned_processes', return_value=[]), \
                patch.object(run, 'verify', return_value={'status':'passed','execution_commit':'fixture'}), \
                patch.object(run, 'identity', return_value={'bytes':1,'sha256':'x'}), \
                patch.object(run.subprocess, 'check_output', return_value=''), \
                patch.object(run.subprocess, 'run', return_value=SimpleNamespace(stdout='')), \
                patch.object(run.subprocess, 'Popen', side_effect=launch), \
                patch.object(run.os, 'killpg') as kill, \
                patch.object(run, 'record_interruption') as interrupted:
            code = run.execute(self.inputs, output)
        self.assertEqual(code, -9)
        kill.assert_called_with(child.pid, run.signal.SIGKILL)
        self.assertTrue(interrupted.call_args.args[-1].startswith('GPU time cap'))
        result = json.loads((output/'RESULT.json').read_text())
        self.assertTrue(result['timed_out'])
        self.assertEqual(result['work']['attempted'], {})
        kinds = [json.loads(line)['kind'] for line in (output/'WORK.jsonl').read_text().splitlines()]
        self.assertEqual(kinds.count('gpu_start'), 1)
        self.assertEqual(kinds.count('gpu_end'), 1)

    def test_existing_gpu_job_prevents_worker_launch(self):
        with patch.object(run, 'verify_seal', return_value=deepcopy(self.plan)), \
                patch.object(run, 'active_owned_processes', return_value=[]), \
                patch.object(run, 'verify', return_value={'status':'passed','execution_commit':'fixture'}), \
                patch.object(run, 'identity', return_value={'bytes':1,'sha256':'x'}), \
                patch.object(run.subprocess, 'check_output', return_value='1234\n'), \
                patch.object(run.subprocess, 'Popen') as launch:
            with self.assertRaisesRegex(RuntimeError, 'GPU work already active'):
                run.execute(self.inputs, self.root/'out')
        launch.assert_not_called()

    def test_nonzero_exit_cannot_preserve_a_completed_campaign_claim(self):
        output = self.root/'out'
        child = Mock(pid=987654, returncode=2)
        child.wait.return_value = 2
        child.poll.return_value = 2

        def launch(*args, **kwargs):
            campaign = output/'campaign'
            campaign.mkdir()
            (campaign/'STATUS.json').write_text(json.dumps({'status':'completed'}))
            return child

        with patch.object(run, 'verify_seal', return_value=deepcopy(self.plan)), \
                patch.object(run, 'active_owned_processes', return_value=[]), \
                patch.object(run, 'verify', return_value={'status':'passed','execution_commit':'fixture'}), \
                patch.object(run, 'identity', return_value={'bytes':1,'sha256':'x'}), \
                patch.object(run.subprocess, 'check_output', return_value=''), \
                patch.object(run.subprocess, 'run', return_value=SimpleNamespace(stdout='')), \
                patch.object(run.subprocess, 'Popen', side_effect=launch), \
                patch.object(run, 'record_interruption') as interrupted:
            self.assertEqual(run.execute(self.inputs, output), 2)
        self.assertEqual(interrupted.call_args.args[-1], 'worker did not finish cleanly')
        self.assertFalse(json.loads((output/'RESULT.json').read_text())['timed_out'])


class PreparedSealTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.order = [dict(dataset=d, request_id=f'{d}_{i}', selection_rank=i,
                           selection_sha256=str(i)) for i in range(64) for d in ('gsm8k', 'mathqa')]
        self.selected = self.root/'SELECTED_IDS.json'
        self.selected.write_text(json.dumps({'execution_order':self.order}))
        self.inputs = self.root/'INPUTS.json'
        self.inputs.write_text(json.dumps({'selected_ids_identity':identity(self.selected), 'requests':self.order}))
        self.seal = self.root/'SEAL.json'
        self.seal.write_text(json.dumps({'selected':identity(self.selected), 'private_inputs':identity(self.inputs)}))

    def test_exact_prepared_selection_is_returned(self):
        plan = verify_seal(self.inputs)
        self.assertEqual(plan['requests'], self.order)

    def test_input_or_selected_byte_change_fails(self):
        for path in (self.inputs, self.selected):
            before = path.read_bytes()
            path.write_bytes(before+b' ')
            with self.subTest(path=path.name), self.assertRaises(ValueError):
                verify_seal(self.inputs)
            path.write_bytes(before)

    def test_inputs_cannot_rebind_ids_even_with_updated_input_checksum(self):
        plan = json.loads(self.inputs.read_text())
        plan['requests'][0]['request_id'] = 'unselected'
        self.inputs.write_text(json.dumps(plan))
        seal = json.loads(self.seal.read_text())
        seal['private_inputs'] = identity(self.inputs)
        self.seal.write_text(json.dumps(seal))
        with self.assertRaisesRegex(ValueError, 'Execution requests differ'):
            verify_seal(self.inputs)


class OwnedProcessTests(unittest.TestCase):
    def test_only_actual_process_group_can_be_registered_and_is_durable(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory)/'WORK.jsonl'
            budget = WorkBudget(ledger)
            with patch('experiments.verification_transition.budget.os.getpgid', return_value=222):
                with self.assertRaisesRegex(ValueError, 'differs from the live process'):
                    budget.register_process(111, 333)
                self.assertEqual(budget.processes(), [])
                budget.register_process(111, 222)
            self.assertEqual(budget.processes(), [{'pid':111,'pgid':222}])
            rows = [json.loads(line) for line in ledger.read_text().splitlines()]
            self.assertEqual(rows[-1]['kind'], 'owned_process')
            self.assertEqual((rows[-1]['pid'], rows[-1]['pgid']), (111, 222))

    def test_cleanup_includes_native_detached_group_and_outer_worker(self):
        proxy = Mock()
        proxy.processes.return_value = [{'pid':222,'pgid':222}]
        child = Mock(pid=111)
        with patch.object(run.os, 'getpgrp', return_value=999), \
                patch.object(run.os, 'killpg') as kill, \
                patch.object(run, 'active_owned_processes', side_effect=[[{'pid':222,'pgid':222,'state':'S'}], []]):
            receipt = run.stop_owned_groups(proxy, child)
        self.assertEqual({call.args[0] for call in kill.call_args_list}, {111,222})
        self.assertEqual(receipt['registered_groups'], [111,222])
        self.assertEqual(receipt['remaining'], [])
        child.wait.assert_called_once()

    def test_cleanup_never_kills_supervisor_group(self):
        proxy = Mock()
        proxy.processes.return_value = [{'pid':999,'pgid':999}]
        with patch.object(run.os, 'getpgrp', return_value=999), patch.object(run.os,'killpg') as kill:
            with self.assertRaisesRegex(RuntimeError, 'supervisor process group'):
                run.stop_owned_groups(proxy, None)
        kill.assert_not_called()

    def test_surviving_native_process_is_not_successful_cleanup(self):
        proxy = Mock()
        proxy.processes.return_value = [{'pid':222,'pgid':222}]
        with patch.object(run.os, 'getpgrp', return_value=999), \
                patch.object(run.os, 'killpg'), \
                patch.object(run, 'active_owned_processes', return_value=[{'pid':222,'pgid':222,'state':'S'}]), \
                patch.object(run.time, 'monotonic', side_effect=[0,11]):
            with self.assertRaisesRegex(RuntimeError, 'survived termination'):
                run.stop_owned_groups(proxy, None)

    def test_process_listing_ignores_unowned_groups_and_zombies(self):
        listing = '10 100 S\n11 100 Z\n12 200 R\n'
        with patch.object(run.subprocess, 'check_output', return_value=listing):
            self.assertEqual(run.active_owned_processes({100}), [{'pid':10,'pgid':100,'state':'S'}])


if __name__ == '__main__':
    unittest.main()
