"""The localized I/O repair cannot reset work, retrain, or replace completed evidence."""
from copy import deepcopy
import json
import multiprocessing
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from experiments.continuation_repair_pilot import launch
from experiments.ok_coding.storage import identity


def writer(root):
    plan = dict(root=root, local_charged_bytes=0,
                limits=dict(elapsed_s=14400, occupied_card_s=14400,
                            reserve_bytes=400, allocated_bytes=2400))
    usage = dict(charged_bytes=10, free_bytes=1000)
    with mock.patch.object(launch, 'storage_usage', return_value=usage):
        for _ in range(25):
            launch.boundary(plan)


class ResumeTests(unittest.TestCase):
    def test_concurrent_resource_writers_have_separate_atomic_receipts(self):
        with tempfile.TemporaryDirectory() as directory:
            context = multiprocessing.get_context('fork')
            processes = [context.Process(target=writer, args=(directory,)) for _ in range(2)]
            for process in processes:
                process.start()
            for process in processes:
                process.join(timeout=10)
                self.assertEqual(process.exitcode, 0)
            for process in processes:
                receipt = Path(directory)/f'STORAGE.{process.pid}.json'
                self.assertEqual(json.loads(receipt.read_text())['writer_pid'], process.pid)
            self.assertFalse(list(Path(directory).glob('*.tmp')))
            self.assertFalse((Path(directory)/'STORAGE.json').exists())

    def fixture(self, directory):
        root = Path(directory)/'run'
        old_source, new_source = Path(directory)/'old_source', Path(directory)/'new_source'
        for path in (root/'sealed',root/'arms/TAIL',root/'mechanism',root/'qualification',old_source,new_source):
            path.mkdir(parents=True,exist_ok=True)
        old_manifest, new_manifest = old_source/'manifest.json',new_source/'manifest.json'
        entries = {'experiments/continuation_repair_pilot/launch.py':{'bytes':1,'sha256':'old'},
                   'online_feedback/continuation_repair.py':{'bytes':1,'sha256':'same'}}
        old_manifest.write_text(json.dumps(dict(commit='a'*40,clean=True,files=entries)))
        revised = deepcopy(entries)
        revised['experiments/continuation_repair_pilot/launch.py']['sha256']='new'
        new_manifest.write_text(json.dumps(dict(commit='b'*40,clean=True,files=revised)))
        plan = dict(root=str(root),source_root=str(old_source),source_manifest=str(old_manifest),
                    source_identity=identity(old_manifest),execution_commit='a'*40,binding='c'*64,
                    training_gpu='B',serving_gpu='A',training_python='python',
                    qualification_result=str(root/'qualification/RESULT.json'), prior_charged_roots=[],
                    resource_admission=dict(remote_remaining_peak_bytes=2200), local_charged_bytes=150,
                    limits=dict(elapsed_s=14400,occupied_card_s=14400,reserve_bytes=400,allocated_bytes=2400))
        original = root/'sealed/PLAN.json'
        original.write_text(json.dumps(plan))
        recovery=root/'arms/TAIL/recovery.pt'; recovery.write_bytes(b'recovery')
        export=root/'arms/TAIL/export';export.mkdir();(export/'model.safetensors').write_bytes(b'endpoint')
        training=dict(status='completed', global_step=256,presentations=512,completed_event=2,
                      recovery=str(recovery),recovery_identity=identity(recovery),checkpoint=str(export),
                      checkpoint_identity=identity(export/'model.safetensors'))
        processes=[dict(phase=name,start_epoch=100.+10*i,end_epoch=110.+10*i,exit_code=0 if i<2 else 1)
                   for i,name in enumerate(('qualification','training','readout'))]
        clock=dict(first_gpu_action_epoch=100.,occupied_card_s=30.,elapsed_s=30.,processes=processes)
        records={'STATUS.json':dict(status='blocked',phases=[dict(phase='qualification'),dict(phase='training')]),
                 'CLOCK.json':clock,'arms/TAIL/STATUS.json':training,
                 'mechanism/STATUS.json':dict(status='failed',completed_primary_responses=73),
                 'mechanism/ENDPOINT_STATE.json':dict(status='passed', unchanged=True),
                 'qualification/RESULT.json':dict(status='passed',binding='c'*64,base_disabled_exact=True,retained_BASE_reusable=True),
                 'ADAM_LEDGER.json':dict(calls=[dict(phase='qualification',status='completed')]*3+
                                         [dict(phase='training',status='completed')]*128)}
        for relative,value in records.items():
            (root/relative).write_text(json.dumps(value))
        (root/'readout.log').write_text('Original FileNotFoundError: STORAGE.json.tmp\n')
        (root/'EXIT').write_text('1\n')
        return root,original,new_source,new_manifest,clock

    def prepare(self, directory):
        root,original,source,manifest,clock=self.fixture(directory)
        original_bytes=original.read_bytes()
        with mock.patch.object(launch,'verify_source') as verify, \
             mock.patch.object(launch,'boundary'), \
             mock.patch.object(launch,'gpu_inventory',return_value={'processes':[]}):
            plan,continuation=launch.prepare_continuation(original,repair_source_root=source,
                repair_manifest=manifest,repair_commit='b'*40)
        self.assertEqual(verify.call_count,2)
        self.assertEqual(original.read_bytes(),original_bytes)
        return root,original,source,manifest,clock,plan,continuation

    def test_resume_preserves_original_plan_clock_failed_work_and_endpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root,original,source,manifest,clock,plan,continuation=self.prepare(directory)
            self.assertEqual(continuation['clock'],clock)
            self.assertEqual(plan['training_execution_commit'],'a'*40)
            self.assertEqual(plan['execution_commit'],'b'*40)
            self.assertEqual(plan['binding'],'c'*64)
            self.assertIn(str(Path(directory)/'old_source'),plan['prior_charged_roots'])
            receipt=json.loads((root/'resume_001/REPAIR.json').read_text())
            self.assertEqual(receipt['original_adam_applications'],131)
            self.assertEqual(receipt['additional_adam_applications'],0)
            self.assertEqual((root/'resume_001/prior/EXIT').read_text(),'1\n')
            self.assertEqual((root/'resume_001/prior/readout.log').read_text(),(root/'readout.log').read_text())
            with mock.patch.object(launch, 'verify_source'), self.assertRaises((FileExistsError, ValueError)):
                launch.prepare_continuation(original,repair_source_root=source,
                    repair_manifest=manifest,repair_commit='b'*40)

    def test_repair_rejects_algorithm_changes_before_any_gpu_work(self):
        with tempfile.TemporaryDirectory() as directory:
            root,original,source,manifest,_=self.fixture(directory)
            changed=json.loads(manifest.read_text())
            changed['files']['online_feedback/continuation_repair.py']['sha256']='algorithm-changed'
            manifest.write_text(json.dumps(changed))
            with mock.patch.object(launch,'verify_source'), \
                 mock.patch.object(launch,'boundary') as boundary:
                with self.assertRaisesRegex(ValueError,'outside orchestration'):
                    launch.prepare_continuation(original,repair_source_root=source,
                        repair_manifest=manifest,repair_commit='b'*40)
            boundary.assert_not_called()
            self.assertFalse((root/'resume_001').exists())

    def test_resume_runs_only_remaining_phases_and_appends_original_clock(self):
        with tempfile.TemporaryDirectory() as directory:
            root,original,source,manifest,clock,plan,continuation=self.prepare(directory)
            class Process:
                pid=123
                returncode=0
                def poll(self):return 0
            commands=[]
            def popen(command,**kwargs):
                commands.append((command,kwargs))
                return Process()
            with mock.patch.object(launch,'verify_source'), mock.patch.object(launch,'boundary'), \
                 mock.patch.object(launch,'gpu_inventory',return_value={'processes':[]}), \
                 mock.patch.object(launch.subprocess,'Popen',side_effect=popen), \
                 mock.patch.object(launch.time,'time',return_value=200.):
                result=launch.execute(plan,continuation=continuation)
            self.assertEqual(result['status'],'completed')
            self.assertEqual(len(commands),2)
            self.assertIn('experiments.continuation_repair_pilot.readout',commands[0][0])
            self.assertIn('--resume-unexecuted',commands[0][0])
            self.assertIn('experiments.continuation_repair_pilot.serving',commands[1][0])
            self.assertTrue(all('native' not in c[0] for c in commands))
            self.assertTrue(all(k['cwd']==str(source) for _,k in commands))
            after=json.loads((root/'CLOCK.json').read_text())
            self.assertEqual(after['first_gpu_action_epoch'],100.)
            self.assertEqual(after['processes'][:3],clock['processes'])
            self.assertEqual(after['occupied_card_s'],30.)
            self.assertEqual(len(after['processes']),5)
            self.assertEqual(json.loads((root/'ADAM_LEDGER.json').read_text())['calls'],
                             json.loads((root/'resume_001/prior/ADAM_LEDGER.json').read_text())['calls'])
            self.assertEqual((root/'readout.log').read_text(),'Original FileNotFoundError: STORAGE.json.tmp\n')

    def test_termination_error_keeps_unresolved_process_and_clock_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _, _, _, clock, plan, continuation = self.prepare(directory)
            process = mock.Mock(pid=123)
            process.poll.side_effect = [0, None]
            with mock.patch.object(launch, 'verify_source'), mock.patch.object(launch, 'boundary'), \
                 mock.patch.object(launch, 'gpu_inventory', return_value={'processes': []}), \
                 mock.patch.object(launch.subprocess, 'Popen', return_value=process), \
                 mock.patch.object(launch, 'stop', side_effect=RuntimeError('termination fixture')), \
                 mock.patch.object(launch.time, 'time', return_value=200.):
                with self.assertRaisesRegex(RuntimeError, 'termination fixture'):
                    launch.execute(plan, continuation=continuation)
            after = json.loads((root/'CLOCK.json').read_text())
            self.assertEqual(after['first_gpu_action_epoch'], clock['first_gpu_action_epoch'])
            self.assertEqual(after['processes'][:3], clock['processes'])
            self.assertIsNone(after['processes'][-1]['end_epoch'])
            self.assertTrue(after['processes'][-1]['active_process_unresolved'])
            self.assertIn('termination fixture', after['processes'][-1]['termination_error'])
            self.assertEqual(json.loads((root/'STATUS.json').read_text())['status'], 'blocked')

    def test_second_preflight_is_cpu_only_and_does_not_create_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            root, original, source, manifest, clock, plan, first = self.prepare(directory)
            attempt = dict(phase='readout', physical_gpu='B', start_epoch=200., end_epoch=209.,
                           exit_code=1, execution_commit='b'*40, continuation=True)
            clock['processes'].append(attempt)
            clock['occupied_card_s'] += 9.
            (root/'CLOCK.json').write_text(json.dumps(clock))
            (root/'resume_001/readout.log').write_text(
                '  File "readout.py", in resume_completed_readouts\n'
                'ValueError: Resume requires the identified storage-writer failure after73 primary readouts\n')
            (root/'mechanism/primary').mkdir()
            for index in range(73):
                path = root/'mechanism/primary'/f'{index:03d}.json'
                path.write_text('{}')
                os.utime(path, (150., 150.))
            path = root/'mechanism/qualification_repeat.json'
            path.write_text('{}')
            os.utime(path, (150., 150.))
            third = Path(directory)/'repair_three'
            third.mkdir()
            third_manifest = third/'manifest.json'
            updated = json.loads(manifest.read_text())
            updated['commit'] = 'd'*40
            updated['files']['experiments/continuation_repair_pilot/launch.py']['sha256'] = 'third'
            third_manifest.write_text(json.dumps(updated))
            with mock.patch.object(launch, 'verify_source'), mock.patch.object(launch, 'boundary'), \
                 mock.patch.object(launch, 'gpu_inventory', return_value={'processes': []}), \
                 mock.patch('experiments.continuation_repair_pilot.readout.authenticated_retained',
                            return_value=({}, {}, {}, {}, {})), \
                 mock.patch('experiments.continuation_repair_pilot.readout.resume_completed_readouts',
                            return_value=({}, {}, {'status':'passed'})) as readout:
                prepared, preflight = launch.prepare_continuation(original, repair_source_root=third,
                    repair_manifest=third_manifest, repair_commit='d'*40, preflight_only=True)
            self.assertIn(str(source), prepared['prior_charged_roots'])
            self.assertIn(str(Path(directory)/'old_source'), prepared['prior_charged_roots'])
            self.assertEqual(prepared['source_root'], str(third))
            self.assertTrue(preflight['preflight_only'])
            self.assertEqual(preflight['clock'], clock)
            self.assertFalse((root/'resume_002').exists())
            self.assertFalse(preflight['rejected_preparation']['historical_individual_byte_equality_proven'])
            self.assertEqual(len(preflight['rejected_preparation']['current_readout_identities']), 74)
            self.assertEqual(readout.call_args.kwargs, {'preserve':False})
            self.assertEqual(json.loads((root/'CLOCK.json').read_text()), clock)


if __name__=='__main__':
    unittest.main()
