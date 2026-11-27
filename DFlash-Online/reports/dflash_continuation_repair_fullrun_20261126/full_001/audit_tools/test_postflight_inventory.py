"""Synthetic stdlib-only tests; no GPU query, model, remote path or native files."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location('postflight_inventory', HERE/'postflight_inventory.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def put(path, data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data))
    return path


class PostflightTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.root=Path(self.temporary.name)
        self.plan=dict(root=str(self.root),binding='binding',execution_commit='commit',
                       limits=m.LIMITS.copy(),arms=list(m.ARMS),events=58,devices=dict(A='GPU-A',B='GPU-B'))
        self.audit=m.Postflight(self.plan)

    def tearDown(self):
        self.temporary.cleanup()

    def test_stream_hash_budget_and_mutation(self):
        path=self.root/'weights';path.write_bytes(b'abcd')
        with self.assertRaises(m.Incomplete):m.Budget(max_hash_bytes=3).hash(path)
        before=self.audit.budget.hash(path)
        path.write_bytes(b'efgh')
        self.assertNotEqual(before,self.audit.budget.hash(path))

    def test_allocation_deduplicates_nested_roots_and_hardlinks(self):
        directory=self.root/'data';directory.mkdir();(directory/'x').write_bytes(b'x'*9000)
        os.link(directory/'x',directory/'y')
        one=m.allocation([directory],m.Budget())
        both=m.allocation([directory,directory/'x'],m.Budget())
        self.assertEqual(one['allocated_bytes'],both['allocated_bytes'])
        self.assertEqual(one['entries'],2)

    def test_retirement_requires_exact_absence_and_preserved_evidence(self):
        proof=put(self.root/'proof.json',{'status':'passed'})
        obsolete=self.root/'campaign_001/old.pt'
        row=dict(path=str(obsolete),kind='older_state',bytes=3)
        receipt=dict(status='completed',files=[row],removed=[row],
                     compact_evidence={str(proof):self.audit.budget.hash(proof)})
        self.assertEqual(self.audit.retirement(receipt),{'older_state':1})
        obsolete.parent.mkdir();obsolete.write_bytes(b'old')
        with self.assertRaisesRegex(ValueError,'still exists'):self.audit.retirement(receipt)
        obsolete.unlink();proof.write_text('changed')
        with self.assertRaisesRegex(ValueError,'evidence differs'):self.audit.retirement(receipt)

    def test_retirement_outside_owned_root_rejected(self):
        with self.assertRaisesRegex(ValueError,'outside'):self.audit.absent_file(self.root/'historical.pt')

    def clock_fixture(self):
        put(self.root/'STATUS.json',dict(status='completed',execution_commit='commit',finished_epoch=120))
        put(self.root/'CLOCK.json',dict(first_model_load_epoch=100,completed_card_s=20,active={},finished=[
            dict(task='qualification_training_001_B',physical_gpu_uuid='GPU-B',started_epoch=100,
                 finished_epoch=120,occupied_s=20,pid=1)]))
        put(self.root/'WORK.json',dict(attempted={'qualification.adam':4},completed={'qualification.adam':4},pending={},failures=[]))
        put(self.root/'processes/qualification_training_001.json',dict(phase='qualification_training',
            termination_confirmed=True,finished_epoch=119,started_epoch=101,process_created=True,exit_code=0,pid=123))

    def test_clock_original_and_actual_idle(self):
        self.clock_fixture()
        result=self.audit.processes_and_clock(original_clock_epoch=100,
            gpu_probe=lambda:dict(gpu_uuids=['GPU-A','GPU-B'],compute_processes=[]),
            process_probe=lambda pid:dict(exists=False))
        self.assertEqual(result['occupied_card_s'],20)
        with self.assertRaisesRegex(ValueError,'clock changed'):
            self.audit.processes_and_clock(original_clock_epoch=99)

    def test_pending_or_live_gpu_is_incomplete(self):
        self.clock_fixture()
        with self.assertRaises(m.Incomplete):
            self.audit.processes_and_clock(original_clock_epoch=100,
                gpu_probe=lambda:dict(gpu_uuids=['GPU-A','GPU-B'],compute_processes=[['GPU-A','5','100']]),
                process_probe=lambda pid:dict(exists=False))
        work=read(self.root/'WORK.json');work['pending']={'1':{}};put(self.root/'WORK.json',work)
        with self.assertRaises(m.Incomplete):self.audit.processes_and_clock(original_clock_epoch=100)

    def test_live_or_reused_pid_is_not_silently_ignored(self):
        self.clock_fixture()
        with self.assertRaisesRegex(m.Incomplete,'live or reused'):
            self.audit.processes_and_clock(original_clock_epoch=100,
                process_probe=lambda pid:dict(exists=True))

    def test_stopped_campaign_is_incomplete_not_failed(self):
        self.clock_fixture();value=read(self.root/'STATUS.json');value['status']='blocked';put(self.root/'STATUS.json',value)
        self.audit.section('clock',lambda:self.audit.processes_and_clock(original_clock_epoch=100))
        self.assertEqual(self.audit.sections['clock']['status'],'incomplete')

    def storage_fixture(self):
        allocated=self.root/'run';allocated.mkdir()
        self.plan.update(allocation_roots=[str(allocated)],compiler_baseline=dict(directories={},allocated_bytes=0),
                         storage=dict(external_charge_bytes=1000,projected_peak_bytes=10000))
        put(self.root/'STORAGE.1.json',dict(charged_bytes=10000,free_bytes=500_000_000_000,projected_peak_bytes=10000))
        external=dict(status='passed',plan_binding='binding',execution_commit='commit',host='peer',checked_epoch=100,
                      all_project_allocations_included=True,measurements=[dict(allocated_bytes=4000,roots=['peer/study'])],
                      allocated_bytes=4000,remaining_write_allowance_bytes=1000,free_bytes=10000,reserve_bytes=5000)
        return external

    def test_storage_checks_nested_forecasts_and_all_locations(self):
        external=self.storage_fixture()
        with patch.object(m.shutil,'disk_usage',return_value=type('D',(),{'free':500_000_000_000})()):
            result=self.audit.storage(external,now=110)
            self.assertEqual(result['external_charge_bytes'],4000)
            bad=dict(storage=dict(charged_bytes=90_000_000_001,free_bytes=500_000_000_000,projected_peak_bytes=10000))
            put(self.root/'forecasts/BASE/event_001.json',bad)
            with self.assertRaisesRegex(ValueError,'admission violation'):self.audit.storage(external,now=110)

    def test_stale_external_storage_and_missing_allowance_rejected(self):
        external=self.storage_fixture()
        with self.assertRaisesRegex(ValueError,'stale'):self.audit.storage(external,now=4000)
        del external['remaining_write_allowance_bytes']
        self.audit.section('storage',lambda:self.audit.storage(external,now=110))
        self.assertEqual(self.audit.sections['storage']['status'],'failed')

    def test_missing_native_evidence_never_passes(self):
        plan=dict(self.plan);plan['binding']=m.digest({k:v for k,v in plan.items() if k!='binding'})
        path=put(self.root/'PLAN.json',plan)
        output=self.root/'audit'
        result=m.run(path,plan_sha256=m.Budget().hash(path)['sha256'],original_clock_epoch=100,
                     external_storage=self.root/'missing_peer.json',output=output,
                     gpu_probe=lambda:self.fail('GPU query must not run'))
        self.assertNotEqual(result['status'],'passed')
        self.assertTrue((output/'PRIVATE_DETAILS.json').is_file())
        self.assertFalse(read(output/'FINAL_TENSOR_INVENTORY.json')['files'])

    def test_exhausted_input_budget_still_writes_incomplete_receipt(self):
        plan=dict(self.plan);plan['binding']=m.digest({k:v for k,v in plan.items() if k!='binding'})
        path=put(self.root/'PLAN.json',plan);output=self.root/'bounded_audit'
        result=m.run(path,plan_sha256=m.Budget().hash(path)['sha256'],original_clock_epoch=100,
                     external_storage='none',output=output,budget=m.Budget(max_hash_bytes=0))
        self.assertEqual(result['status'],'incomplete')
        self.assertTrue((output/'POSTFLIGHT.json').is_file())
        self.assertEqual(result['tensor_inventory_identity'],m.Budget().hash(output/'FINAL_TENSOR_INVENTORY.json'))

    def test_bad_plan_does_not_follow_paths(self):
        plan=dict(self.plan);plan['binding']=m.digest({k:v for k,v in plan.items() if k!='binding'})
        path=put(self.root/'PLAN.json',plan)
        result=m.run(path,plan_sha256='wrong',original_clock_epoch=100,external_storage='none',output=self.root/'audit')
        self.assertEqual(result['status'],'failed')
        self.assertEqual(set(result['sections']),{'plan'})

    def test_complete_amendment_schema_and_preserved_prior_receipts(self):
        original_root=self.root/'original';repaired_root=self.root/'infrastructure_sources/repair_001/DFlash-Online'
        for folder,text in ((original_root,'old'),(repaired_root,'fixed')):
            path=folder/m.AMENDMENT_FILES[0];path.parent.mkdir(parents=True);path.write_text(text)
            (folder/'objective.py').write_text('scientific bytes')
        (repaired_root/m.AMENDMENT_FILES[-1]).write_text('source verifier')
        def manifest(folder,commit):
            return dict(commit=commit,clean=True,files={str(path.relative_to(folder)):m.Budget().hash(path) for path in folder.rglob('*.py')})
        original=manifest(original_root,'commit');new=manifest(repaired_root,'repaircommit')
        original_manifest=put(self.root/'SOURCE.json',original)
        repaired_manifest=put(repaired_root.parent/'SOURCE.json',new)
        self.plan.update(source_root=str(original_root),source_manifest=str(original_manifest),source_identity=m.Budget().hash(original_manifest))
        self.plan['binding']=m.digest({k:v for k,v in self.plan.items() if k!='binding'})
        plan_path=put(self.root/'sealed/PLAN.json',self.plan)
        proof=put(self.root/'CPU_RECOVERY.json',dict(status='passed',new_model_work=0))
        amendment=dict(schema='continuation_repair_fullrun_source_amendment_v1',original_plan_path=str(plan_path),
            original_plan_identity=m.Budget().hash(plan_path),plan_binding=self.plan['binding'],execution_commit='commit',
            actual_execution_commit='repaircommit',source_root=str(repaired_root),source_manifest=str(repaired_manifest),
            source_identity=m.Budget().hash(repaired_manifest),approved_production_changes=m.AMENDMENT_FILES,
            source_comparison=m.compare_sources(original,new),resume_evidence=dict(path=str(proof),identity=m.Budget().hash(proof)))
        amendment['binding']=m.digest(amendment)
        amendment_path=put(repaired_root.parent/'SOURCE_AMENDMENT.json',amendment)
        provenance=dict(execution_commit='commit',actual_execution_commit='repaircommit',actual_source_root=str(repaired_root),
            actual_source_identity=amendment['source_identity'],source_amendment_path=str(amendment_path),
            source_amendment_identity=m.Budget().hash(amendment_path),source_amendment_binding=amendment['binding'])
        old_clock=dict(first_model_load_epoch=100,active={},finished=[])
        old_work=dict(attempted={},completed={},pending={})
        put(self.root/'CLOCK.json',old_clock);put(self.root/'WORK.json',old_work)
        old_process=put(self.root/'processes/train_BASE_001.json',dict(phase='train',exit_code=1))
        resume_root=self.root/'infrastructure_resumes/resume_001'
        backups={}
        for name,data in (('CLOCK.json',old_clock),('WORK.json',old_work),('STATUS.json',{'status':'blocked'}),
                          ('processes/train_BASE_001.json',read(old_process))):
            backup=put(resume_root/'prior'/name,data);backups[name]=m.Budget().hash(backup)
        put(resume_root/'RESUME.json',dict(status='preserved',plan_binding=self.plan['binding'],new_model_work=0,
            original_model_load_epoch=100,prior_files=backups,**provenance))
        put(self.root/'STATUS.json',dict(status='completed',**provenance))
        result=self.audit.source_amendment(original)
        self.assertEqual(result['source_count'],2)
        self.assertEqual(result['actual_execution_commit'],'repaircommit')
        old_process.write_text('changed')
        with self.assertRaisesRegex(ValueError,'removed or overwritten'):self.audit.source_amendment(original)

    def test_source_amendment_rejects_scientific_changes(self):
        original=dict(files={'online_feedback/objective.py':{'sha256':'same'},m.AMENDMENT_FILES[0]:{'sha256':'old'}})
        repaired=dict(files={**original['files'],m.AMENDMENT_FILES[0]:{'sha256':'new'},m.AMENDMENT_FILES[-1]:{'sha256':'new'}})
        comparison=m.compare_sources(original,repaired)
        self.assertEqual(set(comparison['changed_production_paths']),{m.AMENDMENT_FILES[0],m.AMENDMENT_FILES[-1]})
        repaired['files']['online_feedback/objective.py']={'sha256':'changed'}
        with self.assertRaisesRegex(ValueError,'Unapproved'):m.compare_sources(original,repaired)

    def test_source_and_protected_full_identity(self):
        source=self.root/'source';source.mkdir()
        code=source/'objective.py';code.write_text('fixed')
        model=self.root/'model.pt';model.write_bytes(b'weights')
        source_receipt=put(self.root/'SOURCE.json',dict(commit='commit',clean=True,files={'objective.py':m.Budget().hash(code)}))
        expected=m.Budget().hash(model)
        self.plan.update(source_root=str(source),source_manifest=str(source_receipt),source_identity=m.Budget().hash(source_receipt),
                         formula_code_identities={'objective.py':m.Budget().hash(code)},input_identities={str(model):expected},
                         authenticated_files=[],retained_endpoints={})
        put(self.root/'PROTECTED_INPUTS.json',dict(status='sealed',files=[dict(path=str(model),**expected,mtime_ns=model.stat().st_mtime_ns)]))
        put(self.root/'STATUS.json',dict(execution_commit='commit'))
        self.assertEqual(self.audit.sources_and_protected()['protected_files'],1)
        model.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'Identity mismatch'):self.audit.sources_and_protected()

    def test_packaging_recovery_preserves_failure_and_all_cost(self):
        archive=self.root/'event.tar.zst'
        old=dict(status='failed',verification={'status':'passed'},error='Loose member changed after archive verification: x',
                 removed_paths=['a'],removed_allocated_bytes=100,wall_s=2)
        failed=put(str(archive)+'.retirement.failed_001.json',old)
        repair=dict(status='completed',new_model_work=0,failed_receipt=dict(path=str(failed),**m.Budget().hash(failed)),
                    verification=old['verification'],new_removed_paths=['b'],previous_removed_count=1,remaining_count=1,
                    new_removed_allocated_bytes=20,wall_s=1)
        put(str(archive)+'.retirement.recovery_001.json',repair)
        final=dict(recovery=repair,removed_paths=['a','b'],removed_allocated_bytes=120,wall_s=3)
        self.audit.archive_recovery(archive,final)
        self.assertEqual(self.audit.private['archive_recoveries'][0]['new_model_work'],0)
        final['wall_s']=1
        with self.assertRaisesRegex(ValueError,'time was dropped'):self.audit.archive_recovery(archive,final)

    def test_archive_stream_authenticates_without_extracting(self):
        native=self.root/'fake_zstd.py'
        native.write_text('#!'+sys.executable+'\nimport sys\nsys.stdout.buffer.write(open(sys.argv[-1],"rb").read())\n')
        native.chmod(0o755);self.plan['zstd']=str(native)
        archive=self.root/'compact/BASE/event_001.tar.zst';archive.parent.mkdir(parents=True)
        source=put(self.root/'member',{'status':'completed','files':[],'removed':[]})
        name='campaign_001/arms/BASE/events/event_001/RETIREMENT.json'
        with tarfile.open(archive,'w') as stream:stream.add(source,arcname=name)
        ai=m.Budget().hash(archive)
        index=dict(members={name:m.Budget().hash(source)},archive_bytes=ai['bytes'],archive_sha256=ai['sha256'])
        index['binding_digest']=m.digest(index)
        put(str(archive)+'.index.json',index)
        put(str(archive)+'.retirement.json',dict(status='completed',removed_paths=[name]))
        result=self.audit.archive('BASE',1)
        self.assertEqual(result[name]['status'],'completed')
        self.assertFalse((self.root/name).exists())
        self.assertEqual(self.audit.compact_identity(self.root/name),m.Budget().hash(source))


def read(path):return json.loads(Path(path).read_text())

if __name__=='__main__':unittest.main()
