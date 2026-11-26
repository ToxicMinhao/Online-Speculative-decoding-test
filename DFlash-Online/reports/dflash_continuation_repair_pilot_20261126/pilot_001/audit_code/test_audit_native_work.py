"""Independent audit regression fixtures; no production imports or real model."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from audit_native_work import identity, Audit, anchor_blocks, finite_tree, normalized_case, scientific_rows_equal


def receipt():
    return dict(rows=[dict(training_s=3., response_observation=dict(event_id='BASE', loss=1.))],
                gradient={'sha256':'g'},endpoint={'rng_sha256':'r','exp_avg':'m'},gradient_norm=2.)


def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def seed_resume(root):
    arms=['BASE','TAIL_DISABLED','TAIL']
    calls=[dict(token=i,phase='qualification',arm=arm,status='completed') for i,arm in enumerate(arms)]
    calls += [dict(token=i+3,phase='training',arm='TAIL',status='completed') for i in range(128)]
    write(root/'ADAM_LEDGER.json',{'calls':calls})
    write(root/'resume_001/prior/ADAM_LEDGER.json',{'calls':calls})
    phases=['qualification','training','readout','readout','evaluation']
    processes=[dict(phase=phase,start_epoch=1001+20*i,end_epoch=1011+20*i,exit_code=int(i==2)) for i,phase in enumerate(phases)]
    clock=dict(first_gpu_action_epoch=1000.,occupied_card_s=50.,elapsed_s=91.,processes=processes)
    write(root/'CLOCK.json',clock)
    write(root/'resume_001/prior/CLOCK.json',dict(clock,occupied_card_s=30.,elapsed_s=51.,processes=processes[:3]))
    write(root/'STATUS.json',{'status':'completed','execution_commit':'new','training_execution_commit':'old'})
    write(root/'mechanism/attempts/001/STATUS.json',{'status':'failed','work':{'response_reads_completed':74}})
    write(root/'mechanism/attempts/002/STATUS.json',{'attempt_work':{'response_reads_completed':63},
        'work':{'response_reads_completed':137},'qualification_response_forwards':0,'reused_qualification_response_forwards':1})
    write(root/'mechanism/RESUME_BINDING.json',{'retained_file_identities':{}})
    write(root/'resume_001/REPAIR.json',{'additional_adam_applications':0,'original_adam_applications':131})
    return {(0,arm):{} for arm in arms},clock


def seed_second_resume(root):
    cases,clock=seed_resume(root)
    processes=clock['processes'][:3]+[
        dict(phase='readout',start_epoch=1071.,end_epoch=1081.,exit_code=1),
        dict(phase='readout',start_epoch=35001.,end_epoch=35011.,exit_code=0),
        dict(phase='evaluation',start_epoch=35021.,end_epoch=35031.,exit_code=0)]
    clock.update(processes=processes,occupied_card_s=60.,elapsed_s=34031.)
    write(root/'CLOCK.json',clock)
    write(root/'resume_002/prior/CLOCK.json',dict(clock,processes=processes[:4],occupied_card_s=40.,elapsed_s=81.))
    write(root/'resume_002/prior/ADAM_LEDGER.json',json.loads((root/'ADAM_LEDGER.json').read_text()))
    error=root/'resume_001/readout.log'
    error.write_text('resume_completed_readouts\nValueError: Resume requires the identified storage-writer failure after73 primary readouts\n')
    evidence=dict(status='passed',new_forward_evidence=0,existing_readouts=74,
                  authenticated_traceback_before_adapter_load=True,retained_aggregate_hashes_unchanged=True,
                  historical_individual_byte_equality_proven=False,rejected_process_occupied_s=10.,
                  rejected_log_identity=identity(error),current_readout_identities={})
    write(root/'resume_002/REPAIR.json',dict(rejected_preparation=evidence))
    return cases,clock


class NativeAuditChecks(unittest.TestCase):
    def test_only_prespecified_metadata_is_removed(self):
        a=receipt();b=copy.deepcopy(a)
        b['rows'][0]['training_s']=8.;b['rows'][0]['response_observation']['event_id']='DISABLED'
        self.assertEqual(normalized_case(a),normalized_case(b))
        b['gradient']['sha256']='changed'
        self.assertNotEqual(normalized_case(a),normalized_case(b))
        b=copy.deepcopy(a);b['endpoint']['exp_avg']='changed'
        self.assertNotEqual(normalized_case(a),normalized_case(b))
        b=copy.deepcopy(a);b['endpoint']['rng_sha256']='changed'
        self.assertNotEqual(normalized_case(a),normalized_case(b))

    def test_natural_schedule_rejects_anchor_and_order_changes(self):
        a=[dict(record_id=str(i),pass_index=0,sampled_anchors=[{'positions':[[i,i+1]]}],
                optimizer_step_after=128+i//2,optimizer_stepped=bool(i%2)) for i in range(4)]
        for row in a:row['pass']=row.pop('pass_index')
        self.assertTrue(scientific_rows_equal(a,copy.deepcopy(a)))
        b=copy.deepcopy(a);b[0]['sampled_anchors'][0]['positions'][0][0]=999
        self.assertFalse(scientific_rows_equal(a,b))
        self.assertFalse(scientific_rows_equal(a,list(reversed(a))))
        self.assertFalse(scientific_rows_equal(a,a[:-1]))

    def test_anchor_geometry_counts_partial_chunks(self):
        self.assertEqual(anchor_blocks({'sampled_anchors':[{'positions':[list(range(17))]}]}),17)
        with self.assertRaises(ValueError):anchor_blocks({'sampled_anchors':[]})
        with self.assertRaises(ValueError):anchor_blocks({'sampled_anchors':[{'positions':[[1],[2]]}]})

    def test_nonfinite_preserved_as_failure(self):
        self.assertTrue(finite_tree({'count':128,'label':'BASE','values':[.1,2]}))
        self.assertFalse(finite_tree({'nested':[float('nan')]}))
        self.assertFalse(finite_tree({'nested':{'g':float('inf')}}))

    def test_failures_accumulate_without_overwrite(self):
        a=Audit();a.eq(1,2,'first');a.eq(3,3,'pass');a.eq(2,1,'second')
        self.assertEqual(a.checks,3);self.assertEqual(a.failures,['first','second'])

    def test_endpoint_micro_count_means_empty_remainder_not_zero_total(self):
        tensors=[dict(name=str(i),shape=[1 if i else 1048626432-57],dtype='torch.bfloat16') for i in range(58)]
        vector=dict(sha256='a'*64,tensors=tensors)
        endpoint=dict(parameters=vector,exp_avg=vector,exp_avg_sq=vector,steps=[129.]*58,
                      global_step=129,presentations=258,micro=258,gradients_cleared=True,
                      options=dict(lr=3e-4,weight_decay=.01),rng_sha256='b'*64)
        a=Audit();a.endpoint(endpoint,129,258,'endpoint');self.assertEqual(a.failures,[])
        endpoint['micro']=259;a=Audit();a.endpoint(endpoint,129,258,'endpoint')
        self.assertIn('endpoint.empty_accumulation',a.failures)
        endpoint['micro']=258;endpoint['steps'][7]=128
        a=Audit();a.endpoint(endpoint,129,258,'endpoint')
        self.assertIn('endpoint.Adam_counters',a.failures)

    def test_resume_preserves_failed_attempt_clock_and_Adam_ledger(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);cases,clock=seed_resume(root)
            a=Audit();a.ledger_and_clock(root,cases);self.assertEqual(a.failures,[])
            self.assertEqual(a.totals['readout_prior_completed'],74)
            self.assertEqual(a.totals['readout_resumed_completed'],63)
            clock['first_gpu_action_epoch']+=1;write(root/'CLOCK.json',clock)
            a=Audit();a.ledger_and_clock(root,cases)
            self.assertIn('clock.original_epoch_preserved',a.failures)
            clock['first_gpu_action_epoch']-=1;clock['processes'][2]['exit_code']=0;write(root/'CLOCK.json',clock)
            a=Audit();a.ledger_and_clock(root,cases)
            self.assertIn('clock.preserved_attempt_exit_codes',a.failures)

    def test_resume_rejects_repeated_forward_or_extra_optimizer_application(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);cases,_=seed_resume(root)
            current=root/'mechanism/attempts/002/STATUS.json';value=json.loads(current.read_text())
            value['attempt_work']['response_reads_completed']=64;write(current,value)
            ledger=json.loads((root/'ADAM_LEDGER.json').read_text())
            ledger['calls'].append(dict(token=131,phase='training',arm='TAIL',status='completed'))
            write(root/'ADAM_LEDGER.json',ledger)
            a=Audit();a.ledger_and_clock(root,cases)
            self.assertIn('resume.new63',a.failures)
            self.assertIn('ledger.committed_steps',a.failures)
            self.assertIn('resume.original_Adam_calls_retained',a.failures)

    def test_resume_provenance_allows_orchestration_only(self):
        with TemporaryDirectory() as directory:
            root=Path(directory)
            plan={'binding':'fixed','execution_commit':'old','source_identity':{'sha256':'oldsrc'},'lr':3e-4,'root':'remote'}
            write(root/'sealed/PLAN.json',plan)
            original=identity(root/'sealed/PLAN.json')
            execution=dict(plan,execution_commit='new',training_execution_commit='old',
                           source_identity={'sha256':'newsrc'},original_plan_identity=original)
            repair=dict(original_training_commit='old',orchestration_execution_commit='new',
                        original_plan_identity=original,original_settings_unchanged=True,
                        repair_source_identity=execution['source_identity'],original_source_identity=plan['source_identity'],
                        changed_files=['experiments/continuation_repair_pilot/readout.py'],preserved_failures={})
            write(root/'resume_001/EXECUTION_PLAN.json',execution);write(root/'resume_001/REPAIR.json',repair)
            for path in ('mechanism/STATUS.json','STATUS.json'):
                write(root/path,{'execution_commit':'new','training_execution_commit':'old'})
            write(root/'evaluation/PROTOCOL.json',{'execution_commit':'new'})
            a=Audit();a.resume_provenance(root,plan);self.assertEqual(a.failures,[])
            execution['lr']=1e-4;write(root/'resume_001/EXECUTION_PLAN.json',execution)
            repair['changed_files'].append('online_feedback/objective.py');write(root/'resume_001/REPAIR.json',repair)
            a=Audit();a.resume_provenance(root,plan)
            self.assertIn('resume.only_source_and_orchestration_plan_changes',a.failures)
            self.assertIn('resume.source_changes_within_narrow_repair',a.failures)

    def test_second_resume_keeps_both_failures_and_original_gpu_cap(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);cases,clock=seed_second_resume(root)
            a=Audit();a.ledger_and_clock(root,cases);self.assertEqual(a.failures,[])
            self.assertEqual(a.totals['readout_process_attempts'],3)
            self.assertEqual(a.totals['readout_attempts'],3)
            self.assertEqual(a.totals['readout_model_forward_attempts'],2)
            self.assertEqual(a.totals['readout_forward_counts_by_process'],[74,0,63])
            self.assertEqual(a.totals['pre_model_rejection_occupied_seconds'],10.)
            clock['occupied_card_s']=15000.
            clock['processes'][-1]['end_epoch']+=14940.
            clock['elapsed_s']=43000.
            write(root/'CLOCK.json',clock)
            a=Audit();a.ledger_and_clock(root,cases)
            self.assertIn('clock.time_caps',a.failures)

    def test_second_resume_rejects_hidden_model_work_or_omitted_failure(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);cases,clock=seed_second_resume(root)
            repair=json.loads((root/'resume_002/REPAIR.json').read_text())
            repair['rejected_preparation']['new_forward_evidence']=1
            write(root/'resume_002/REPAIR.json',repair)
            clock['processes'][3]['exit_code']=0;write(root/'CLOCK.json',clock)
            a=Audit();a.ledger_and_clock(root,cases)
            self.assertIn('resume2.pre_model_zero_new_forward_evidence',a.failures)
            self.assertIn('clock.preserved_attempt_exit_codes',a.failures)

    def test_second_resume_authorization_changes_elapsed_only_with_four_sources(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);_,clock=seed_second_resume(root)
            source_specs=[]
            for index in range(4):
                local=root/'audit_inputs'/f'source{index}.json'
                write(local,dict(commit=f'commit{index}',clean=True,files={}))
                source_specs.append((f'/remote/source{index}/SOURCE.json',identity(local),f'commit{index}'))
            limits=dict(elapsed_s=14400,occupied_card_s=14400,allocated_bytes=24000000000,reserve_bytes=400000000000)
            plan=dict(binding='fixed',root='/remote/run',execution_commit='commit0',source_manifest=source_specs[0][0],
                      source_identity=source_specs[0][1],limits=limits)
            write(root/'sealed/PLAN.json',plan);original=identity(root/'sealed/PLAN.json')
            authorization=dict(kind='elapsed_limit_extension',original_plan_binding='fixed',original_clock_epoch=1000.,
                previous_elapsed_s=14400,approved_elapsed_s=43200,occupied_card_s=14400,
                authorized_scope='remaining readouts and evaluation',user_answer='Extend elapsed limit to 12 hours',
                recorded_epoch=34000.,prior_preflight_source_manifest=source_specs[2][0],prior_preflight_source_identity=source_specs[2][1])
            auth_path=root/'audit_inputs/ELAPSED_EXTENSION_AUTHORIZATION.json';write(auth_path,authorization)
            for directory,index in [('resume_001',1),('resume_002',3)]:
                execution=dict(plan,source_manifest=source_specs[index][0],source_identity=source_specs[index][1],
                    execution_commit=source_specs[index][2],training_execution_commit='commit0',original_plan_identity=original)
                if index==3:
                    execution.update(limits=dict(limits,elapsed_s=43200),elapsed_extension_authorization='/remote/ELAPSED_EXTENSION_AUTHORIZATION.json',
                        elapsed_extension_authorization_identity=identity(auth_path))
                repair=dict(original_training_commit='commit0',orchestration_execution_commit=source_specs[index][2],
                    original_plan_identity=original,original_settings_unchanged=True,
                    repair_source_identity=source_specs[index][1],original_source_identity=source_specs[0][1],
                    changed_files=['experiments/continuation_repair_pilot/launch.py'],preserved_failures={})
                if index==3:
                    repair.update(elapsed_extension_authorization_identity=identity(auth_path),original_limits=limits,
                                  effective_limits=execution['limits'],elapsed_extension={'only_elapsed_limit_changed':True})
                write(root/directory/'EXECUTION_PLAN.json',execution);write(root/directory/'REPAIR.json',repair)
            for path in ('mechanism/STATUS.json','STATUS.json'):
                write(root/path,{'execution_commit':'commit3','training_execution_commit':'commit0'})
            write(root/'evaluation/PROTOCOL.json',{'execution_commit':'commit3'})
            a=Audit();a.resume_provenance(root,plan);self.assertEqual(a.failures,[])
            self.assertEqual(len(a.totals['source_identity_chain']),4)
            latest=root/'resume_002/EXECUTION_PLAN.json';value=json.loads(latest.read_text())
            value['limits']['occupied_card_s']=43200;write(latest,value)
            a=Audit();a.resume_provenance(root,plan)
            self.assertIn('resume_002.only_elapsed_limit_changed',a.failures)

    def test_serving_work_rejects_omitted_or_duplicate_execution(self):
        with TemporaryDirectory() as directory:
            root=Path(directory)
            status=dict(status='completed',planned=792,attempted=792,returned=792,completed_valid=792,
                        invalid=0,unknown=0,unattempted=0,pending=None)
            write(root/'evaluation/STATUS.json',status)
            counts=dict(scored_generation=768,warmup_generation=24)
            write(root/'evaluation_work.json',dict(attempted=counts,completed=counts,pending={}))
            rows=[dict(request_id=f'r{i}',method=arm,repeat=repeat,phase='scored',valid_measurement=True,errors=[])
                  for i in range(128) for arm in ('BASE','TAIL') for repeat in range(3)]
            rows += [dict(request_id=f'w{i%2}',method=('BASE','TAIL')[i%2],repeat=i//8,
                          phase='warmup',valid_measurement=True,errors=[]) for i in range(24)]
            path=root/'evaluation/MEASUREMENTS.jsonl'
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            a=Audit();a.serving_work(root);self.assertEqual(a.failures,[])
            rows[0]=copy.deepcopy(rows[1]);path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            a=Audit();a.serving_work(root)
            self.assertIn('serving.no_duplicate_repeats',a.failures)

    def test_loss_discrepancy_not_timing_metadata(self):
        a=receipt();b=copy.deepcopy(a);b['rows'][0]['response_observation']['loss']+=1e-12
        self.assertNotEqual(normalized_case(a),normalized_case(b))


if __name__=='__main__':unittest.main()
