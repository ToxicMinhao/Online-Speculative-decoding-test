"""Qualified-stop continuation reuses evidence and never resets model work."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from online_feedback.storage import write_json
from experiments.c_paired_confirmation import resume as R
from experiments.c_paired_confirmation.prepare import plan_binding,identity

AUTH=dict(policy='through_declared_campaign_completion',scope='elapsed_only',original_elapsed_cap_s=14400,
    user_statement='We will allow more time until experiment finished, how far are we from experiment done?')


class ResumeTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)/'run';self.root.mkdir()
        source=Path(temp.name)/'source';(source/'online_feedback').mkdir(parents=True)
        (source/'online_feedback/objective.py').write_text('qualified source\n')
        source_files={'online_feedback/objective.py':identity(source/'online_feedback/objective.py')}
        write_json(self.root/'SOURCE.json',dict(commit='a'*40,dirty=False,files=source_files))
        self.old=dict(root=str(self.root),binding='pending',source_root=str(source),source_manifest=str(self.root/'SOURCE.json'),
            execution_commit='a'*40,limits=dict(R.LIMITS),allocation_roots=[str(self.root),str(source)],
            compiler_baseline_allocated_bytes=123,estimated_seconds={},datasets={},input_identities={},
            methods=['N','C'],continuation_rng_seed=29,scientific_source_identities=source_files)
        for dataset in ('gsm8k','mathqa'):
            folder=self.root/'references'/dataset;folder.mkdir(parents=True)
            refs={}
            for i in range(2):
                path=self.root/'tasks'/f'{dataset}_warmup_references'/'records'/f'{i}.json'
                trace=path.with_suffix('.trace.json');write_json(trace,{})
                write_json(path,{'metadata':{'native_trace_path':str(trace)}})
                refs[str(i)]={'source_record_path':str(path)}
            write_json(folder/'WARMUP_CANONICALS.json',refs)
            self.old['datasets'][dataset]=dict(warmup_references=str(folder/'WARMUP_CANONICALS.json'),
                origin_export='/'+dataset+'/origin',origin_weights_identity={'sha256':dataset})
        dependency=self.root/'dependency.json';write_json(dependency,{'immutable':True})
        self.old['input_identities'][str(dependency)]=identity(dependency)
        self.old['binding']=plan_binding(self.old)
        write_json(self.root/'sealed/PLAN.json',self.old)
        write_json(self.root/'sealed/SEAL.json',dict(binding=self.old['binding']))
        write_json(self.root/'CPU_PREFLIGHT.json',dict(status='passed',execution_commit='a'*40))
        for dataset,sub in self.old['datasets'].items():
            write_json(self.root/'tasks'/f'{dataset}_warmup_references'/'RESULT.json',dict(status='completed',reference_count=2,
                binding=self.old['binding'],execution_commit='a'*40))
            write_json(Path(sub['warmup_references']).parent/'WARMUP_SEAL.json',dict(status='passed',reference_count=2,
                checkpoint=sub['origin_export'],checkpoint_identity=sub['origin_weights_identity'],execution_commit='a'*40,
                binding=self.old['binding'],files={sub['warmup_references']:identity(sub['warmup_references'])}))
            for kind,slot in (('training','A'),('serving','A'),('serving','B')):
                base=self.root/'qualification'/f'{dataset}_{kind}_{slot}'
                result={'cpu_export':{'status':'passed'}} if kind=='training' else {
                    'qualification':{'signatures':{'one':[1,2],'two':[3]}},'checkpoint_identity':{'sha256':dataset}}
                write_json(base/'STATUS.json',dict(status='passed',result=result))
                if kind=='training':write_json(base/'DETAILS.json',{'status':'passed'})
                else:
                    write_json(base/'RESULT.json',result);write_json(base/'calls/RESULT.json',result['qualification'])
        write_json(self.root/'QUALIFICATION.json',R.launch.qualification_gate(self.old))
        write_json(self.root/'CLOCK.json',dict(first_model_load_epoch=100.,completed_card_s=1433.,active={},finished=[]))
        write_json(self.root/'WORK.json',dict(limits=dict(R.LIMITS),attempted=dict(R.COMPLETED_QUALIFICATION),
            completed=dict(R.COMPLETED_QUALIFICATION),pending={}))
        write_json(self.root/'LAUNCH.json',dict(status='stopped',completed=['cpu_preflight','gsm8k_qualification','mathqa_qualification'],
            error='RuntimeError: Remaining fixed campaign does not fit original clocks: fixture',clock={'occupied_s':1433.}))
        folder=self.root/'resume_001';folder.mkdir()
        write_json(folder/'AUTHORIZATION.json',dict(elapsed_time_authorization=AUTH))
        write_json(folder/'SOURCE.json',dict(commit='b'*40,dirty=False,files=source_files))
        proof=dict(original_plan=dict(path=str(self.root/'sealed/PLAN.json'),identity=identity(self.root/'sealed/PLAN.json')),
            authorization=dict(path=str(folder/'AUTHORIZATION.json'),identity=identity(folder/'AUTHORIZATION.json')),
            retained_files={p:identity(p) for p in R.required_retained_files(self.old)})
        self.plan=dict(self.old,execution_commit='b'*40,source_manifest=str(folder/'SOURCE.json'),
            original_execution_commit='a'*40,original_scientific_binding=self.old['binding'],elapsed_time_authorization=AUTH,
            adopted_qualification_path=str(folder/'QUALIFICATION.json'),resume_from_qualified_stop=proof)
        self.path=folder/'PLAN.json';write_json(self.path,self.plan)
        self.input_patch=patch.object(R,'load_inputs',return_value={});self.input_patch.start();self.addCleanup(self.input_patch.stop)
        self.runtime_patch=patch.object(R,'runtime_inventory',return_value={'status':'passed','cuda_initialized':False})
        self.runtime=self.runtime_patch.start();self.addCleanup(self.runtime_patch.stop)

    def test_authentication_preserves_original_records_and_consumed_state(self):
        names=('LAUNCH.json','CLOCK.json','WORK.json','QUALIFICATION.json','CPU_PREFLIGHT.json','sealed/PLAN.json')
        before={n:(self.root/n).read_bytes() for n in names}
        result=R.authenticate_resume(self.plan)
        self.assertEqual(result['status'],'passed');self.assertEqual(result['consumed_work']['completed'],R.COMPLETED_QUALIFICATION)
        self.runtime.assert_called_once_with(self.plan)
        self.assertEqual(result['runtime'],{'status':'passed','cuda_initialized':False})
        adopted=R.adopt_qualification(self.plan,result)
        self.assertEqual(adopted['execution_commit'],'b'*40);self.assertEqual(adopted['original_execution_commit'],'a'*40)
        self.assertTrue(adopted['reused_without_new_model_work'])
        self.assertEqual(before,{n:(self.root/n).read_bytes() for n in names})

    def test_scientific_change_or_other_budget_change_rejected(self):
        changed=deepcopy(self.plan);changed['continuation_rng_seed']=30
        with self.assertRaisesRegex(ValueError,'scientific plan'):R.authenticate_resume(changed)
        changed=deepcopy(self.plan);changed['limits']['adam']+=1
        with self.assertRaisesRegex(ValueError,'scientific plan'):R.authenticate_resume(changed)
        changed=deepcopy(self.plan);changed['elapsed_time_authorization']['scope']='all_limits'
        with self.assertRaisesRegex(ValueError,'elapsed-only'):R.authenticate_resume(changed)

    def test_scientific_source_change_rejected_even_with_current_manifest(self):
        source=Path(self.plan['source_root'])/'online_feedback/objective.py';source.write_text('different formula\n')
        m=R.read(self.plan['source_manifest']);m['files']['online_feedback/objective.py']=identity(source)
        write_json(self.plan['source_manifest'],m)
        with self.assertRaisesRegex(ValueError,'scientific source'):R.authenticate_resume(self.plan)

    def test_missing_or_changed_retained_evidence_stops(self):
        changed=deepcopy(self.plan);del changed['resume_from_qualified_stop']['retained_files'][str(self.root/'CLOCK.json')]
        with self.assertRaisesRegex(ValueError,'Missing required'):R.authenticate_resume(changed)
        write_json(self.root/'CLOCK.json',{'changed':True})
        with self.assertRaisesRegex(ValueError,'Retained evidence changed'):R.authenticate_resume(self.plan)

    def test_pending_active_or_prior_production_stops_even_if_pins_updated(self):
        for field in ('pending','active','production'):
            with self.subTest(field=field):
                work=R.read(self.root/'WORK.json');clock=R.read(self.root/'CLOCK.json')
                if field=='pending':work['pending']={'1':{'unknown':True}}
                elif field=='active':clock['active']={'worker':{}}
                else:(self.root/'tasks/gsm8k_N_first').mkdir()
                write_json(self.root/'WORK.json',work);write_json(self.root/'CLOCK.json',clock)
                self.plan['resume_from_qualified_stop']['retained_files'].update({str(self.root/n):identity(self.root/n) for n in ('WORK.json','CLOCK.json')})
                with self.assertRaisesRegex(ValueError,'Unexpected|Production task'):R.authenticate_resume(self.plan)
                work['pending']={};clock['active']={};write_json(self.root/'WORK.json',work);write_json(self.root/'CLOCK.json',clock)

    def test_changed_origin_dependency_stops_before_production(self):
        Path(next(iter(self.old['input_identities']))).write_text('changed')
        with self.assertRaisesRegex(ValueError,'input dependency'):R.authenticate_resume(self.plan)

    def test_entry_skips_qualification_and_keeps_original_clock_work(self):
        work=(self.root/'WORK.json').read_bytes();clock=(self.root/'CLOCK.json').read_bytes()
        def execute(path,plan,receipt,status_path,resources):receipt.update(status='completed')
        resources={'status':'passed','groups':[]}
        with patch.object(R.launch,'check_compilers'),patch.object(R.launch,'admission',return_value=resources),\
                patch.object(R.launch,'execute_production',side_effect=execute) as production,\
                patch.object(R.launch,'initialize_work') as reset,patch.object(R.launch,'run_children') as children:
            result=R.run(self.path)
        self.assertEqual(result['status'],'completed');production.assert_called_once();reset.assert_not_called();children.assert_not_called()
        self.assertEqual((self.root/'WORK.json').read_bytes(),work);self.assertEqual((self.root/'CLOCK.json').read_bytes(),clock)
        self.assertEqual(R.read(self.root/'LAUNCH.json')['status'],'stopped')
        with self.assertRaisesRegex(FileExistsError,'automatic'):R.run(self.path)

    def test_failing_admission_preserves_receipt_and_starts_no_tasks(self):
        with patch.object(R.launch,'check_compilers'),patch.object(R.launch,'admission',side_effect=RuntimeError('card cap')),\
                patch.object(R.launch,'execute_production') as production:
            with self.assertRaisesRegex(RuntimeError,'card cap'):R.run(self.path)
        production.assert_not_called()
        status=R.read(self.path.parent/'RESUME_LAUNCH.json');self.assertEqual(status['status'],'stopped')
        self.assertFalse((self.root/'PRODUCTION_STARTED.json').exists())

    def test_runtime_drift_stops_before_adopting_qualification(self):
        self.runtime.side_effect=ValueError('Pinned runtime changed')
        with patch.object(R.launch,'check_compilers'),patch.object(R.launch,'execute_production') as production:
            with self.assertRaisesRegex(ValueError,'runtime changed'):R.run(self.path)
        production.assert_not_called()
        self.assertFalse(Path(self.plan['adopted_qualification_path']).exists())


if __name__=='__main__':unittest.main()
