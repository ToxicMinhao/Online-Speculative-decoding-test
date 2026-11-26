"""Synthetic, CPU-only tests of the standalone serving auditor."""
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

import audit_serving_independent as audit


def request(i, warm=False):
    return dict(request_id=('warm' if warm else 'request')+f'{i:03}', source_id=f'source{warm}{i}',
                request_index=i, prompt_ids=[1000+i+(1000 if warm else 0)],
                stop_token_ids=[151645,151643], max_new_tokens=2048)


def reference(q):
    value=dict(q,response_ids=[1,2,3,4,151645],finish_reason='eos',reference_kind='frozen_DFlash',
               verified=True,provenance={'test':'fixture'})
    value['binding_digest']=audit.digest(value)
    return value


def raw(q, slot, checkpoint):
    response=[1,2,3,4,151645]
    definitions=([(0,[2,9,9],[2,3,0,0],2),(2,[4,151645],[4,151645,0],2)]
                 if slot['method']=='BASE' else [(0,[2,3,4,151645],[2,3,4,151645,0],4)])
    trace=[]
    for offset,proposal,target,progress in definitions:
        accepted=next((i for i,token in enumerate(proposal) if token != target[i]),len(proposal))
        trace.append(dict(output_start=offset,anchor_token=response[offset],draft_tokens=proposal,
            target_argmax=target,verify_size=len(target),committed_progress=progress,
            committed_tokens=response[offset+1:offset+1+progress],positions=list(range(1+offset,1+offset+len(target))),
            accepted_draft_tokens=min(accepted,progress),native_accepted_draft_tokens=progress-1,
            raw_accepted_draft_tokens_before_stop=accepted,stop_clipped=progress<accepted+1))
    base_time=1+q['request_index']/100+slot['repeat']/10
    elapsed=base_time*(1 if slot['method']=='BASE' else .9)
    capture=dict(prompt_ids=q['prompt_ids'],response_ids=response,blocks=trace,collect_features=False,
                 feature_bytes=0,native_capture_validated=True,cache_maintenance_s=elapsed*.1)
    return dict(record_id=f"a{slot['activation']:03d}_{slot['phase']}_{slot['request_ordinal']:04d}_{slot['method']}",
        request_index=q['request_index'],session_id='opencode_final58',purpose='evaluation',checkpoint_id=checkpoint,
        prompt_ids=q['prompt_ids'],response_ids=response,finish_reason='eos',stop_token_ids=q['stop_token_ids'],max_new_tokens=2048,
        metadata=dict(trace=trace,capture=capture,block_size=16,context_limit=4096,runtime='pinned_SGLang_FA3',
            feedback_capture_in_timed_request=False,cache_policy='idle_full_flush_before_every_request_and_publication',
            request_latency_s=elapsed,native_generate_s=elapsed*.8,verification_calls=len(trace),committed_progress=4,
            accepted_draft_tokens=sum(row['accepted_draft_tokens'] for row in trace)))


def measurements(count=128):
    panel=[request(i) for i in range(count)]
    warmups=[request(i,True) for i in range(2)]
    if count==128:
        entries,slots=audit.expected_schedule(panel,warmups)
    else:
        entries=[]
        slots=[dict(activation=0,repeat=r,group=0,method=m,phase='scored',request_ordinal=q['request_index'],request_id=q['request_id'])
               for q in panel for m in audit.ARMS for r in range(3)]
    by_id={q['request_id']:q for q in panel+warmups}
    rows=[]
    for slot in slots:
        q=by_id[slot['request_id']]
        record=raw(q,slot,slot['method'])
        measured,_=audit.check_record(audit.Audit(),record,q,reference(q),slot,slot['method'],0)
        rows.append(dict(slot,**measured,valid_measurement=True,errors=[],physical_gpu_uuid='GPU-test',source_id=q['source_id'],
                         raw_path='/remote/evaluation/raw/'+record['record_id']+'.json'))
    return panel,warmups,entries,slots,rows


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value)+'\n')


def lines(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))


def csv_file(path,rows):
    with path.open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def file_identity(path):
    data=path.read_bytes()
    return dict(bytes=len(data),sha256=audit.hashlib.sha256(data).hexdigest())


def add_resume_fixture(root, public):
    plan_path=root/'sealed/PLAN.json'
    plan=json.loads(plan_path.read_text())
    plan.update(source_root='/remote/original/code', source_manifest='/remote/original/MANIFEST.json',
        source_identity={'sha256':'original-manifest','bytes':123}, binding='same-inputs',
        limits={'elapsed_s':14400,'occupied_card_s':14400}, prior_charged_roots=[])
    write(plan_path,plan)
    revised=copy.deepcopy(plan)
    revised.update(source_root='/remote/repaired/code', source_manifest='/remote/repaired/MANIFEST.json',
        source_identity={'sha256':'repaired-manifest','bytes':124}, execution_commit='repair',
        training_execution_commit='test', original_plan='/remote/run/sealed/PLAN.json',
        original_plan_identity=file_identity(plan_path),prior_charged_roots=['/remote/original'])
    directory=root/'resume_001'
    write(directory/'EXECUTION_PLAN.json',revised)
    old_clock={'first_gpu_action_epoch':100.,'occupied_card_s':30.}
    calls={'calls':[{'phase':'qualification','status':'completed'}]*3+
                   [{'phase':'training','status':'completed'}]*128}
    write(directory/'prior/CLOCK.json',old_clock)
    write(directory/'prior/ADAM_LEDGER.json',calls)
    write(root/'ADAM_LEDGER.json',calls)
    receipt=dict(original_plan_identity=file_identity(plan_path),original_plan=revised['original_plan'],
        original_training_commit='test',orchestration_execution_commit='repair',
        original_source_identity=plan['source_identity'],repair_source_identity=revised['source_identity'],
        binding=plan['binding'],additional_adam_applications=0,original_settings_unchanged=True,
        changed_files=['experiments/continuation_repair_pilot/launch.py'],
        preserved_failures={name:file_identity(directory/'prior'/name) for name in ('CLOCK.json','ADAM_LEDGER.json')},
        original_clock_epoch=100.,prior_occupied_card_s=30.,original_adam_applications=131)
    write(directory/'REPAIR.json',receipt)
    protocol=json.loads((root/'evaluation/PROTOCOL.json').read_text());protocol['execution_commit']='repair'
    write(root/'evaluation/PROTOCOL.json',protocol)
    final=json.loads((root/'FINAL_ANALYSIS.json').read_text())
    final['serving'].update(execution_commit='repair',training_execution_commit='test',readout_attempts=2)
    write(root/'FINAL_ANALYSIS.json',final)
    value=json.loads((public/'SERVING_SUMMARY.json').read_text())
    value.update(execution_commit='repair',training_execution_commit='test',readout_attempts=2)
    write(public/'SERVING_SUMMARY.json',value)


def add_second_resume_fixture(root,public):
    """Two recorded failures, one approved elapsed-only change, no repeated training."""
    add_resume_fixture(root,public)
    paths={name:root/'audit_inputs'/name for name in
           ('source.json','repair_source.json','preflight_source.json','execution_source.json')}
    for name,commit in zip(paths,('test','repair','preflight','repair2')):
        write(paths[name],dict(commit=commit,clean=True,files={}))
    original=json.loads((root/'sealed/PLAN.json').read_text())
    original['source_identity']=file_identity(paths['source.json'])
    write(root/'sealed/PLAN.json',original)
    original_identity=file_identity(root/'sealed/PLAN.json')
    first=json.loads((root/'resume_001/EXECUTION_PLAN.json').read_text())
    first.update(source_identity=file_identity(paths['repair_source.json']),original_plan_identity=original_identity)
    write(root/'resume_001/EXECUTION_PLAN.json',first)
    processes=[dict(phase=phase,start_epoch=100.+10*i,end_epoch=110.+10*i,exit_code=0 if i<2 else 1)
               for i,phase in enumerate(('qualification','training','readout'))]
    original_clock=dict(first_gpu_action_epoch=100.,occupied_card_s=30.,elapsed_s=30.,processes=processes)
    write(root/'resume_001/prior/CLOCK.json',original_clock)
    first_receipt=json.loads((root/'resume_001/REPAIR.json').read_text())
    first_receipt.update(original_plan_identity=original_identity,
        original_source_identity=original['source_identity'],repair_source_identity=first['source_identity'])
    first_receipt['preserved_failures']['CLOCK.json']=file_identity(root/'resume_001/prior/CLOCK.json')
    write(root/'resume_001/REPAIR.json',first_receipt)
    authorization=dict(kind='elapsed_limit_extension',original_plan_binding=original['binding'],
        original_clock_epoch=100.,previous_elapsed_s=14400,approved_elapsed_s=43200,occupied_card_s=14400,
        authorized_scope='remaining readouts and evaluation',user_answer='Extend elapsed limit to 12 hours',
        prior_preflight_source_manifest='/remote/preflight/MANIFEST.json',
        prior_preflight_source_root='/remote/preflight/code',
        prior_preflight_source_identity=file_identity(paths['preflight_source.json']),recorded_epoch=30000.)
    write(root/'sealed/ELAPSED_EXTENSION_AUTHORIZATION.json',authorization)
    authorization_identity=file_identity(root/'sealed/ELAPSED_EXTENSION_AUTHORIZATION.json')
    write(root/'sealed/RESUME2_AUTHORIZATION.json',dict(
        user_authorization='approve to resume the remaining readouts and evaluation',
        scope=dict(readouts=63,scored_generations=768,warmups=24,additional_training_steps=0,repeat_completed_readouts=False),
        source_commit='preflight',original_clock_preserved=True,recorded_epoch=29999.))
    second=copy.deepcopy(original)
    second.update(source_root='/remote/repair2/code',source_manifest='/remote/repair2/MANIFEST.json',
        source_identity=file_identity(paths['execution_source.json']),execution_commit='repair2',
        training_execution_commit='test',original_plan=first['original_plan'],original_plan_identity=original_identity,
        prior_charged_roots=['/remote/original','/remote/repaired','/remote/preflight'],
        limits=dict(original['limits'],elapsed_s=43200),
        elapsed_extension_authorization='/remote/run/sealed/ELAPSED_EXTENSION_AUTHORIZATION.json',
        elapsed_extension_authorization_identity=authorization_identity)
    write(root/'resume_002/EXECUTION_PLAN.json',second)
    prior=copy.deepcopy(original_clock)
    prior['processes'].append(dict(phase='readout',start_epoch=200.,end_epoch=209.,exit_code=1,execution_commit='repair',continuation=True))
    prior.update(occupied_card_s=39.,elapsed_s=109.)
    write(root/'resume_002/prior/CLOCK.json',prior)
    write(root/'resume_002/prior/ADAM_LEDGER.json',json.loads((root/'ADAM_LEDGER.json').read_text()))
    receipt=copy.deepcopy(first_receipt)
    receipt.update(orchestration_execution_commit='repair2',repair_source_identity=second['source_identity'],
        prior_occupied_card_s=39.,original_limits=original['limits'],effective_limits=second['limits'],
        elapsed_extension_authorization=second['elapsed_extension_authorization'],
        elapsed_extension_authorization_identity=authorization_identity,
        elapsed_extension=dict(authorization_path=second['elapsed_extension_authorization'],
            authorization_identity=authorization_identity,original_limits=original['limits'],effective_limits=second['limits'],
            original_clock_epoch=100.,prior_preflight_source_identity=authorization['prior_preflight_source_identity'],
            only_elapsed_limit_changed=True),
        rejected_preparation=dict(status='passed',original_clock_unchanged=True,retained_aggregate_hashes_unchanged=True,
            existing_readouts=74,new_forward_evidence=0,historical_individual_byte_equality_proven=False,
            rejected_execution_commit='repair'),
        preserved_failures={name:file_identity(root/'resume_002/prior'/name) for name in ('CLOCK.json','ADAM_LEDGER.json')})
    write(root/'resume_002/REPAIR.json',receipt)
    final_clock=copy.deepcopy(prior)
    final_clock['processes'] += [dict(phase='readout',start_epoch=31000.,end_epoch=31010.,exit_code=0,execution_commit='repair2'),
        dict(phase='evaluation',start_epoch=31010.,end_epoch=31100.,exit_code=0,execution_commit='repair2')]
    final_clock.update(occupied_card_s=139.,elapsed_s=31000.)
    write(root/'CLOCK.json',final_clock)
    protocol=json.loads((root/'evaluation/PROTOCOL.json').read_text());protocol['execution_commit']='repair2'
    write(root/'evaluation/PROTOCOL.json',protocol)
    final=json.loads((root/'FINAL_ANALYSIS.json').read_text())
    final['serving'].update(execution_commit='repair2',readout_attempts=3,readout_model_forward_attempts=2,
                            readout_forward_counts_by_process=[74,0,63])
    write(root/'FINAL_ANALYSIS.json',final)
    public_summary=json.loads((public/'SERVING_SUMMARY.json').read_text())
    public_summary.update(execution_commit='repair2',readout_attempts=3,readout_model_forward_attempts=2,
                          readout_forward_counts_by_process=[74,0,63])
    write(public/'SERVING_SUMMARY.json',public_summary)
    write(root/'mechanism/STATUS.json',dict(status='completed',work={'response_reads_completed':137},
                                           attempt_work={'response_reads_completed':63}))
    write(root/'mechanism/attempts/001/STATUS.json',dict(status='failed',work={'response_reads_completed':74}))


def full_fixture(root,public):
    panel,warmups,entries,slots,rows=measurements()
    inputs=dict(panel=panel,warmup=warmups)
    refs={name:{q['request_id']:reference(q) for q in values} for name,values in inputs.items()}
    by_id={q['request_id']:q for q in panel+warmups}
    write(root/'evaluation/INPUTS.json',inputs);write(root/'evaluation/REFERENCES.json',refs)
    write(root/'sealed/PLAN.json',dict(execution_commit='test',devices={'A':'GPU-test'},base_export_identity={'sha256':'base'}))
    write(root/'evaluation/PROTOCOL.json',dict(schedule=entries,execution_commit='test',gpu='GPU-test',
        canonical_provenance={'test':'fixture'},checkpoints={'BASE':'BASE','TAIL':'TAIL'}))
    lines(root/'evaluation/MEASUREMENTS.jsonl',rows)
    for slot in slots:
        record=raw(by_id[slot['request_id']],slot,slot['method'])
        write(root/'evaluation/raw'/(record['record_id']+'.json'),record)
    status=dict(status='completed',planned=792,attempted=792,returned=792,completed_valid=792,
                invalid=0,pending=None,automatic_retry=False,unknown=0,unattempted=0)
    write(root/'evaluation/STATUS.json',status)
    write(root/'evaluation_work.json',dict(attempted={'scored_generation':768,'warmup_generation':24},
        completed={'scored_generation':768,'warmup_generation':24},pending={},operations=792))
    activations=[]
    for entry in entries:
        check=entry['method']
        publication=dict(status='passed',checkpoint_path=check,cache_after_tokens=0,
            helper_matches_current_layers=True,helper_norm_matches_current_layers=True,fused_enabled=True)
        activations.append(dict(entry,loading_s=2.,publication=dict(checkpoint_structure_valid=True,checkpoint_id=check,
            activation_s=.1,native_status=dict(status='passed',poisoned=None,capture_pending=False,cache_tokens=0,
                attention_backend='fa3',block_size=16,publication=publication))))
    lines(root/'evaluation/ACTIVATIONS.jsonl',activations)
    for arm in audit.ARMS:
        write(root/f'evaluation/WEIGHTS_{arm}.json',dict(errors=[],receipt=dict(status='passed',all_equal=True,tensors=58,
            source_tensor_count=58,native_tensor_count=43,checkpoint_path=arm,
            helper_matches_current_layers=True,helper_norm_matches_current_layers=True,fused_enabled=True)))
    summary=audit.independent_reduction(rows,audit.Audit())
    summary['generation_accounting']=dict(returned=792,valid=792,scored=768,warmups=24,activations=12)
    summary['serving_costs']=dict(loading_s=24.,activation_s=1.2,
        warmup_request_latency_s=sum(row['request_latency_s'] for row in rows if row['phase']=='warmup'),
        scored_request_latency_s=sum(row['request_latency_s'] for row in rows if row['phase']=='scored'))
    summary['parameter_identities']={'BASE':{'sha256':'A'},'TAIL':{'sha256':'B'}}
    summary['distinct_serving_weights']=True
    write(root/'SERVING_PARAMETERS.json',summary['parameter_identities'])
    write(root/'evaluation/SUMMARY.json',summary)
    write(root/'evaluation/EXPORTS_UNCHANGED.json',{'BASE':{'sha256':'base'},'TAIL':{'sha256':'tail'}})
    write(root/'arms/TAIL/STATUS.json',dict(status='completed',checkpoint_identity={'sha256':'tail'},
        activity={'continuation_changed_high_positions':2},training={'records':[{'response_observation':{
            'continuation_repair':{'continuation_changed_high_positions':2}}}]}))
    write(root/'mechanism/STATUS.json',dict(status='completed'))
    summary['decision']=audit.independent_decision(summary,distinct=True,active=True,complete=True)
    summary['execution_commit']='test'
    write(root/'FINAL_ANALYSIS.json',dict(serving=summary))
    public.mkdir()
    write(public/'SERVING_SUMMARY.json',{k:v for k,v in summary.items() if k not in ('paired','bootstrap')})
    write(public/'BOOTSTRAP.json',{k:v for k,v in summary['bootstrap'].items() if k!='input_order'})
    ordinals={q['request_id']:i for i,q in enumerate(summary['paired'])}
    csv_file(public/'PER_REQUEST.csv',[dict(request_ordinal=i,**{arm+'_'+key:row[arm][key]
        for arm in audit.ARMS for key in (*audit.TIMES,*audit.COUNTS,'finish_reason')}) for i,row in enumerate(summary['paired'])])
    for phase,name in (('scored','ALL_REPEATS.csv'),('warmup','WARMUPS.csv')):
        lookup=ordinals if phase=='scored' else {q['request_id']:i for i,q in enumerate(warmups)}
        csv_file(public/name,[dict(request_ordinal=lookup[row['request_id']],**{key:row[key]
            for key in ('activation','repeat','group','method',*audit.TIMES,*audit.COUNTS,'finish_reason','valid_measurement')})
            for row in rows if row['phase']==phase])


class IndependentServingTests(unittest.TestCase):
    def test_schedule_balanced_and_exact_budget(self):
        panel,warmups,entries,slots,_=measurements()
        self.assertEqual((len(entries),len(slots)),(12,792))
        self.assertEqual(sum(slot['phase']=='scored' for slot in slots),768)
        for repeat in range(3):
            first=[entry['method'] for entry in entries if entry['repeat']==repeat and entry['position']==0]
            self.assertEqual(sorted(first),['BASE','TAIL'])

    def test_known_call_and_time_reductions(self):
        *_,rows=measurements(4)
        summary=audit.independent_reduction(rows,audit.Audit(),expected_requests=4)
        contrast=summary['contrasts']['TAIL_vs_BASE']
        self.assertAlmostEqual(contrast['verification_call_reduction_pct']['value'],50.)
        np.testing.assert_allclose(contrast['verification_call_reduction_pct']['ci95'],[50.,50.],atol=0,rtol=0)
        self.assertAlmostEqual(contrast['request_latency_s_reduction_pct']['value'],10.)
        np.testing.assert_allclose(contrast['request_latency_s_reduction_pct']['ci95'],[10.,10.],atol=1e-11,rtol=0)
        self.assertEqual(summary['totals']['BASE']['verification_calls'],8.)
        self.assertEqual(summary['totals']['TAIL']['verification_calls'],4.)
        self.assertEqual(contrast['wins'],4)

    def test_trace_rejects_rejection_boundary_and_canonical_change(self):
        q=request(0);slot=dict(activation=0,repeat=0,group=0,method='BASE',phase='scored',request_ordinal=0,request_id=q['request_id'])
        record=raw(q,slot,'BASE')
        for mutate in ('output','proposal','progress'):
            changed=copy.deepcopy(record)
            if mutate=='output': changed['response_ids'][1]=99
            elif mutate=='proposal': changed['metadata']['trace'][0]['draft_tokens'][0]=99
            else: changed['metadata']['trace'][0]['committed_progress']=3
            with self.assertRaises(ValueError): audit.check_record(audit.Audit(),changed,q,reference(q),slot,'BASE',0)

    def test_repeats_are_not_independent_requests_and_duplicates_fail(self):
        *_,rows=measurements(4)
        changed=copy.deepcopy(rows);changed[1]['repeat']=0
        with self.assertRaisesRegex(ValueError,'Missing/duplicate'): audit.independent_reduction(changed,audit.Audit(),expected_requests=4)
        changed=copy.deepcopy(rows);changed[1]['verification_calls']=3
        with self.assertRaisesRegex(ValueError,'Repeat identity/count'): audit.independent_reduction(changed,audit.Audit(),expected_requests=4)
        changed=copy.deepcopy(rows);changed[1]['valid_measurement']=False
        with self.assertRaisesRegex(ValueError,'Invalid row'): audit.independent_reduction(changed,audit.Audit(),expected_requests=4)

    def test_shared_bootstrap_draw_identity(self):
        *_,rows=measurements(4)
        report=audit.independent_reduction(rows,audit.Audit(),expected_requests=4)
        reference=np.random.Generator(np.random.PCG64(20261126)).integers(0,4,size=(10000,4))
        self.assertEqual(report['bootstrap']['draw_sha256'],audit.hashlib.sha256(reference.astype('<i8').tobytes()).hexdigest())

    def test_decision_fixed_thresholds_and_identical_parameters(self):
        def report(low,high=1.,point=.2,calls=.1):
            return {'contrasts':{'TAIL_vs_BASE':{'request_latency_s_reduction_pct':{'value':point,'ci95':[low,high]},
                'verification_call_reduction_pct':{'value':calls}}}}
        choose=lambda r,**k:audit.independent_decision(r,distinct=k.get('distinct',True),active=k.get('active',True),complete=k.get('complete',True))['label']
        self.assertEqual(choose(report(.01)),'SUPPORTED_FOR_CONFIRMATION')
        self.assertEqual(choose(report(-.49)),'TENTATIVE')
        self.assertEqual(choose(report(-.5)),'NO_DEMONSTRATED_BENEFIT')
        self.assertEqual(choose(report(-.49),active=False),'NO_DEMONSTRATED_BENEFIT')
        self.assertEqual(choose(report(.01),distinct=False),'NO_DEMONSTRATED_BENEFIT')
        self.assertEqual(choose(report(.01),complete=False),'BLOCKED')

    def test_complete_synthetic_artifact_audit(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'raw';public=Path(folder)/'public'
            full_fixture(root,public)
            checks=audit.Audit();result=audit.audit_run(root,public,checks)
            self.assertEqual(result['status'],'passed')
            self.assertEqual(result['native_blocks'],1188)
            self.assertGreater(checks.checks,50000)

    def test_public_table_corruption_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'raw';public=Path(folder)/'public'
            full_fixture(root,public)
            path=public/'PER_REQUEST.csv';text=path.read_text().splitlines()
            cells=text[1].split(',');cells[1]='999';text[1]=','.join(cells);path.write_text('\n'.join(text)+'\n')
            with self.assertRaisesRegex(ValueError,'PER_REQUEST.csv'): audit.audit_run(root,public,audit.Audit())

    def test_explicit_repair_provenance_and_original_clock(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'raw';public=Path(folder)/'public'
            full_fixture(root,public);add_resume_fixture(root,public)
            result=audit.audit_run(root,public,audit.Audit())
            self.assertEqual(result['source_provenance']['training_commit'],'test')
            self.assertEqual(result['source_provenance']['serving_commit'],'repair')
            self.assertEqual(result['source_provenance']['additional_adam_applications'],0)
            self.assertEqual(result['source_provenance']['original_clock_epoch'],100.)

    def test_resume_cannot_change_settings_or_add_adam_or_relabel_training(self):
        for changed in ('settings','adam','training-label'):
            with self.subTest(changed=changed),tempfile.TemporaryDirectory() as folder:
                root=Path(folder)/'raw';public=Path(folder)/'public'
                full_fixture(root,public);add_resume_fixture(root,public)
                if changed=='settings':
                    path=root/'resume_001/EXECUTION_PLAN.json';value=json.loads(path.read_text())
                    value['limits']['elapsed_s']=20000
                elif changed=='adam':
                    path=root/'ADAM_LEDGER.json';value=json.loads(path.read_text())
                    value['calls'].append({'phase':'training','status':'completed'})
                else:
                    path=root/'FINAL_ANALYSIS.json';value=json.loads(path.read_text())
                    value['serving']['training_execution_commit']='repair'
                write(path,value)
                with self.assertRaises(ValueError): audit.audit_run(root,public,audit.Audit())

    def test_two_failures_explicit_elapsed_extension_and_all_source_identities(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'raw';public=Path(folder)/'public'
            full_fixture(root,public);add_second_resume_fixture(root,public)
            result=audit.audit_run(root,public,audit.Audit())
            provenance=result['source_provenance']
            self.assertEqual(provenance['clock']['elapsed_s'],31000.)
            self.assertEqual(provenance['clock']['occupied_card_s'],139.)
            self.assertEqual(provenance['clock']['failed_pre_model_s'],9.)
            self.assertEqual(provenance['elapsed_extension']['approved_elapsed_s'],43200)
            self.assertEqual(len(provenance['source_manifests']),4)
            self.assertFalse(provenance['historical_individual_readout_byte_equality_proven'])

    def test_two_failure_audit_rejects_reset_clock_missing_approval_or_changed_gpu_budget(self):
        for mutation in ('clock_reset','discard_failure','missing_approval','card_budget','source_omitted','late_approval','hide_zero_forward_attempt'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as folder:
                root=Path(folder)/'raw';public=Path(folder)/'public'
                full_fixture(root,public);add_second_resume_fixture(root,public)
                if mutation in ('clock_reset','discard_failure'):
                    path=root/'CLOCK.json';value=json.loads(path.read_text())
                    if mutation=='clock_reset':value['first_gpu_action_epoch']=30000.
                    else:
                        value['processes'].pop(3);value['occupied_card_s']-=9.
                elif mutation=='missing_approval':
                    (root/'sealed/ELAPSED_EXTENSION_AUTHORIZATION.json').unlink()
                    with self.assertRaises(FileNotFoundError):audit.audit_run(root,public,audit.Audit())
                    continue
                elif mutation=='late_approval':
                    path=root/'sealed/RESUME2_AUTHORIZATION.json';value=json.loads(path.read_text())
                    value['recorded_epoch']=32000.
                elif mutation=='hide_zero_forward_attempt':
                    path=root/'FINAL_ANALYSIS.json';value=json.loads(path.read_text())
                    value['serving']['readout_attempts']=2
                else:
                    path=root/'resume_002/EXECUTION_PLAN.json';value=json.loads(path.read_text())
                    if mutation=='card_budget':value['limits']['occupied_card_s']=43200
                    else:value['prior_charged_roots'].remove('/remote/repaired')
                write(path,value)
                with self.assertRaises(ValueError):audit.audit_run(root,public,audit.Audit())

    def test_derived_cpu_analysis_has_separate_pinned_source_and_preserves_raw(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'raw';public=Path(folder)/'public';derived=Path(folder)/'analysis'
            full_fixture(root,public);add_second_resume_fixture(root,public)
            derived.mkdir()
            final=json.loads((root/'FINAL_ANALYSIS.json').read_text());(root/'FINAL_ANALYSIS.json').unlink()
            final['serving'].update(analysis_execution_commit='a'*40,serving_execution_commit='repair2')
            write(derived/'FINAL_ANALYSIS.json',final)
            public_summary=json.loads((public/'SERVING_SUMMARY.json').read_text())
            public_summary.update(analysis_execution_commit='a'*40,serving_execution_commit='repair2')
            write(public/'SERVING_SUMMARY.json',public_summary)
            view=json.loads((root/'resume_002/EXECUTION_PLAN.json').read_text());view['analysis_execution_commit']='a'*40
            write(derived/'ANALYSIS_PLAN.json',view)
            (derived/'analysis.py').write_bytes(b'# exact CPU analysis source fixture\n')
            write(derived/'ANALYSIS_SOURCE.json',dict(analysis_execution_commit='a'*40,
                serving_execution_commit='repair2',training_execution_commit='test',clean=True,
                analysis_py_sha256=file_identity(derived/'analysis.py')['sha256'],
                local_analysis_view_identity=file_identity(derived/'ANALYSIS_PLAN.json')))
            before={str(p.relative_to(root)):file_identity(p) for p in root.rglob('*') if p.is_file()}
            result=audit.audit_run(root,public,audit.Audit(),analysis_root=derived)
            self.assertEqual(result['source_provenance']['readout_attempts'],3)
            self.assertEqual(result['source_provenance']['readout_model_forward_attempts'],2)
            self.assertEqual(before,{str(p.relative_to(root)):file_identity(p) for p in root.rglob('*') if p.is_file()})
            (derived/'analysis.py').write_bytes(b'# changed\n')
            with self.assertRaisesRegex(ValueError,'analysis file identity'):
                audit.audit_run(root,public,audit.Audit(),analysis_root=derived)


if __name__=='__main__': unittest.main()
