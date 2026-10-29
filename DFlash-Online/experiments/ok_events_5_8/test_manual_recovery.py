"""CPU recovery fixtures: preserve event5, never repeat science or fake offload."""
from contextlib import ExitStack
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import manual_recovery as M
from .test_campaign import Harness, read, write
from . import test_storage


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.h=Harness(temporary.name)
        self.h.plan['storage']={'reserve_bytes':8*M.GIB}
        self.h.plan['execution_source']={'commit':M.SCIENTIFIC_COMMIT,'dirty':False}
        self.h.seal()
        base=self.h.storage_class()
        class StopAfterFive(base):
            def after_event(self,event,parents,pool,*,evaluated):
                super().after_event(event,parents,pool,evaluated=evaluated)
                raise InterruptedError('Supervisor signal 15')
        with self.h.patched(),patch('experiments.ok_events_5_8.storage.StorageLifecycle',StopAfterFive):
            with self.assertRaises(InterruptedError):M.C.run(self.h.root)
        self.output=self.h.root/'campaign_001'
        self.prior=read(self.output/'STATUS.json');self.costs=read(self.output/'COSTS.json')
        self.parents={arm:dict(state=str(self.output/arm/'training_state.pt'),
            export=str(self.output/arm/'events/event_005/export'),
            receipt=str(self.output/arm/'events/event_005/RESULT.json')) for arm in ('O','K')}
        self.pool=read(self.output/'event_005/POOL.json')
        self.source={'scientific_execution_commit':M.SCIENTIFIC_COMMIT,'recovery_execution_commit':'a'*40}

    def patched(self):
        stack=ExitStack();stack.enter_context(self.h.patched())
        stack.enter_context(patch.object(M,'verify_source',return_value=(self.h.plan,self.source,{'status':'passed'})))
        stack.enter_context(patch.object(M,'verify_boundary',return_value=(self.prior,self.costs,self.parents,self.pool,{'status':'passed'})))
        stack.enter_context(patch.object(M,'runtime_admission',return_value={'status':'passed'}))
        stack.enter_context(patch.object(M,'LocalRetention',self.h.storage_class()))
        return stack

    def test_resume_only_remaining_events_preserves_initial_segment_and_full_budget(self):
        before={str(p):p.read_bytes() for p in (self.output/'event_005').rglob('*') if p.is_file()}
        initial_calls=len(self.h.calls)
        original_status=(self.output/'STATUS.json').read_bytes()
        original_costs=(self.output/'COSTS.json').read_bytes()
        with self.patched():M.run(self.h.root)
        status=read(self.output/'STATUS.json');costs=read(self.output/'COSTS.json')
        self.assertEqual((status['optimizer_steps'],status['presentations'],status['feedback_returned'],
            status['evaluation_returned'],status['evaluation_valid']),(1024,2048,512,1920,1920))
        self.assertEqual(status['events_completed'],[5,6,7,8])
        new=self.h.calls[initial_calls:]
        self.assertEqual([(x[1],x[2]) for x in new if x[0]=='train'],
            [(a,n) for n in (6,7,8) for a in ('O','K')])
        self.assertEqual(sum(x[0]=='generate' for x in new),384)
        self.assertEqual([x[1:3] for x in new if x[0]=='evaluate'],
            [(6,'development'),(6,'additional'),(8,'development'),(8,'additional'),(8,'fresh')])
        self.assertEqual((self.h.root/'recovery_001/ORIGINAL_STATUS.json').read_bytes(),original_status)
        self.assertEqual((self.h.root/'recovery_001/ORIGINAL_COSTS.json').read_bytes(),original_costs)
        self.assertTrue(all(Path(p).read_bytes()==value for p,value in before.items()))
        self.assertEqual(len(costs['execution_segments']),2)
        self.assertEqual(costs['execution_segments'][0]['wall_s'],self.costs['campaign_wall_s'])
        self.assertEqual(costs['campaign_wall_s'],sum(x['wall_s'] for x in costs['execution_segments']))
        self.assertEqual(costs['operational_recovery']['scientific_execution_commit'],M.SCIENTIFIC_COMMIT)
        self.assertTrue(self.h.backend.closed)
        with self.patched(),self.assertRaises(FileExistsError):M.run(self.h.root)

    def test_new_failure_preserves_count_and_cannot_automatically_retry(self):
        self.h.fail_feedback=3
        with self.patched(),self.assertRaises(RuntimeError):M.run(self.h.root)
        status=read(self.output/'STATUS.json')
        self.assertEqual(status['feedback_returned'],130)
        self.assertEqual(status['feedback_unknown'],1)
        self.assertEqual(status['optimizer_steps'],256)
        self.assertEqual(status['evaluation_unattempted'],1920)
        self.assertEqual(read(self.h.root/'recovery_001/RECOVERY.json')['status'],'failed')
        with self.patched(),self.assertRaises(FileExistsError):M.run(self.h.root)

    def test_postflight_source_change_fails_without_repeating_completed_work(self):
        with self.patched(),patch.object(M,'verify_source',side_effect=[
            (self.h.plan,self.source,{'status':'passed'}),ValueError('source changed')]),self.assertRaises(ValueError):
            M.run(self.h.root)
        self.assertEqual(read(self.output/'STATUS.json')['status'],'failed')
        self.assertEqual(read(self.output/'STATUS.json')['evaluation_returned'],1920)
        self.assertEqual(read(self.h.root/'recovery_001/SOURCE_POSTFLIGHT.json')['status'],'failed')

    def boundary_fixture(self):
        write(self.h.root/'MANUAL_STOP.json',dict(status='stopped',phase='retention_005',signal=15,
            scientific_controller_stopped=True,transfer_processes_stopped=True))
        request=dict(event=5,campaign_root=str(self.output),phase='idle_after_training_and_evaluation',files={})
        for arm,parent in self.parents.items():
            audit=self.output/f'resume_005_{arm}'
            closed=read(audit/'CLOSED.json');closed['failed']=False;write(audit/'CLOSED.json',closed)
            for name in ('EVENT_CHECK.json','SAVED_ENDPOINT.json'):
                write(audit/name,{'status':'passed'})
            write(audit/'LIVE_BOUNDARY.json',dict(status='passed',exact_boundary=True,mismatches=[],
                global_step=640,presentations=1280))
            files=[Path(parent['state']),*Path(parent['export']).iterdir()]
            write(audit/'STATE_IDENTITIES.json',{str(p):M.identity(p) for p in files})
            for p in files[1:]:request['files'][str(p.relative_to(self.output))]=M.identity(p)
        for path in self.pool['features']:
            request['files'][str(Path(path).relative_to(self.output))]=M.identity(path)
        write(self.output/'retention/event_005_REQUEST.json',request)

    def test_boundary_rejects_wrong_stop_partial_work_later_paths_changed_bytes(self):
        self.boundary_fixture()
        with patch('experiments.ok_events_5_8.resume.validate_event_receipt'):
            result=M.verify_boundary(self.h.root,self.h.plan,decode=False)
            self.assertEqual(result[-1]['status'],'passed')
            original=read(self.output/'STATUS.json')
            for mutation in ({'phase':'training_K'},{'error':'TimeoutError()'},
                             {'feedback_returned':127},{'arm_counters':{'O':640,'K':639}}):
                write(self.output/'STATUS.json',dict(original,**mutation))
                with self.assertRaises(ValueError):M.verify_boundary(self.h.root,self.h.plan,decode=False)
            write(self.output/'STATUS.json',original)
            (self.output/'event_006').mkdir()
            with self.assertRaises(ValueError):M.verify_boundary(self.h.root,self.h.plan,decode=False)
            (self.output/'event_006').rmdir()
            Path(self.parents['O']['state']).write_bytes(b'changed')
            with self.assertRaises(ValueError):M.verify_boundary(self.h.root,self.h.plan,decode=False)


class SourceAndRetentionTests(unittest.TestCase):
    def test_source_allowlist_actual_plan_schema_and_original_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);folder=root/'source';folder.mkdir()
            for name in (*M.ADDITIONS,'campaign.py'):(folder/name).write_bytes(name.encode())
            originals=root/'original.pt';originals.write_bytes(b'original')
            plan={'execution_source':{'commit':M.SCIENTIFIC_COMMIT,'dirty':False}}
            write(root/'sealed/PLAN.json',plan)
            source=dict(scientific_execution_commit=M.SCIENTIFIC_COMMIT,recovery_execution_commit='a'*40,
                files={str(folder/name):M.identity(folder/name) for name in M.ADDITIONS})
            write(root/'sealed/RECOVERY_SOURCE.json',source)
            write(root/'sealed/REMOTE_INPUT_IDENTITIES.json',{str(originals):M.identity(originals)})
            write(root/'sealed/SEAL.json',dict(files={str(folder/'campaign.py'):M.identity(folder/'campaign.py')},
                exact_file_sets={str(folder):['campaign.py']}))
            with patch.object(M,'__file__',str(folder/'manual_recovery.py')),patch.object(M.C,'validate_plan'):
                self.assertEqual(M.verify_source(root)[-1]['additive_files'],2)
                (folder/'unexpected.py').write_bytes(b'change')
                with self.assertRaises(ValueError):M.verify_source(root)
                (folder/'unexpected.py').unlink();(folder/'campaign.py').write_bytes(b'changed')
                with self.assertRaises(ValueError):M.verify_source(root)

    def test_local_retention_never_deletes_or_fakes_ack_and_checks_required_endpoints(self):
        helper=test_storage.StorageTests()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);output=root/'campaign_001';output.mkdir();(output/'retention').mkdir()
            local=M.LocalRetention({'storage':{'reserve_bytes':8*M.GIB}},output)
            parents,pool,feature,capture=helper.artifacts(output,6)
            with patch('shutil.disk_usage') as disk:
                disk.return_value.free=100*M.GIB
                local.after_event(6,parents,pool,evaluated=True)
            original={p:p.read_bytes() for p in (feature,capture)}
            local.retire_obsolete_exports({})
            self.assertTrue(all(p.exists() and p.read_bytes()==value for p,value in original.items()))
            self.assertFalse((output/'retention/event_006_ACK.json').exists())
            for arm in ('O','K'):
                self.assertTrue((output/arm/'events/event_006/training_state.pt').exists())
            with self.assertRaises(ValueError):local.finalize(parents)
            with self.assertRaises(ValueError):local.after_event(6,parents,pool,evaluated=False)


if __name__=='__main__':unittest.main()
