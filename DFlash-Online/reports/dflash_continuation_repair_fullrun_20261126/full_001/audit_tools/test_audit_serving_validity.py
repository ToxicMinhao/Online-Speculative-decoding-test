import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import audit_serving_validity as a


def request(index):
    source=dict(dataset='nvidia/OpenCodeInstruct',revision='synthetic',raw_index=index)
    raw='input '+str(index);rendered='rendered '+raw;ids=[index+1]
    source_hash=a.digest(source)
    return dict(request_index=index,panel_ordinal=index,source_identity=source,source_identity_sha256=source_hash,
        source_id='opencode:'+source_hash,request_id='opencode:'+source_hash,raw_input=raw,
        raw_input_sha256=a.text_hash(raw),normalized_raw_sha256=a.text_hash(raw),rendered_prompt=rendered,
        rendered_prompt_sha256=a.text_hash(rendered),prompt_ids=ids,prompt_tokens=1,prompt_ids_sha256=a.digest(ids),
        stop_token_ids=[151645,151643],max_new_tokens=2048)


def native(request,checkpoint,slot,serial=0):
    output=[11,13,151645]
    block=dict(output_start=0,anchor_token=11,draft_tokens=[13],target_argmax=[13,151645],verify_size=2,
        committed_progress=2,committed_tokens=[13,151645],positions=[1,2],accepted_draft_tokens=1,
        native_accepted_draft_tokens=1,raw_accepted_draft_tokens_before_stop=1,stop_clipped=False,finished=True)
    capture=dict(status='passed',prompt_ids=request['prompt_ids'],response_ids=output,blocks=[block],
        target_argmax_ids=output,covered_token_ids=request['prompt_ids']+output[:-1],covered_rows=3,block_size=16,
        collect_features=False,feature_bytes=0,native_capture_validated=True,final_feature_logits_used_for_labels=False,
        final_emitted_token_feature_present=False,label_binding_scope='native_prefill_and_committed_verification',cache_maintenance_s=.1)
    return dict(record_id=a.label(slot),session_id=a.SESSION,request_index=request['request_index'],
        checkpoint_id=checkpoint,purpose='evaluation',prompt_ids=request['prompt_ids'],response_ids=output,
        stop_token_ids=request['stop_token_ids'],max_new_tokens=2048,finish_reason='eos',
        metadata=dict(block_size=16,context_limit=4096,runtime='pinned_SGLang_FA3',
            cache_policy='idle_full_flush_before_every_request_and_publication',feedback_capture_in_timed_request=False,
            trace_instrumentation_in_timed_request=True,capture=capture,trace=[block],verification_calls=1,
            committed_progress=2,accepted_draft_tokens=1,committed_progress_per_call=2.,calls_per_1000_output_tokens=1000/3,
            request_latency_s=1.,native_generate_s=.9,native_trace_path='/original/native/'+str(serial)+'.json'))


def reference(request,record,checkpoint='/F',commit='commit'):
    result=dict(request_id=request['request_id'],**{k:record[k] for k in ('prompt_ids','response_ids','stop_token_ids','max_new_tokens','finish_reason')},
        verified=True,reference_kind='frozen_DFlash',provenance={},source_checkpoint=checkpoint,
        source_record_id=record['record_id'],execution_commit=commit)
    result['binding_digest']=a.digest(result);return result


class NativeRecordTests(unittest.TestCase):
    def setUp(self):
        self.request=request(0)
        self.slot=dict(activation=0,repeat=0,group=0,method='F',phase='scored',request_ordinal=0)
        self.record=native(self.request,'/F',self.slot)
        self.ref=reference(self.request,self.record)
    def validate(self):return a.validate_record(self.record,self.request,self.ref,'/F',self.slot,a.Checks())
    def test_native_reconstruction_and_distinct_acceptance_counters(self):
        values=self.validate();self.assertEqual(values['verification_calls'],1);self.assertEqual(values['committed_progress'],2)
        self.record['response_ids']=[11,151645];block=self.record['metadata']['trace'][0]
        block.update(draft_tokens=[151645,15],target_argmax=[151645,15,16],verify_size=3,committed_progress=1,
            committed_tokens=[151645],positions=[1,2,3],accepted_draft_tokens=1,native_accepted_draft_tokens=0,
            raw_accepted_draft_tokens_before_stop=2,stop_clipped=True)
        meta=self.record['metadata'];meta.update(committed_progress=1,committed_progress_per_call=1.,calls_per_1000_output_tokens=500.)
        meta['capture'].update(response_ids=[11,151645],target_argmax_ids=[11,151645],covered_token_ids=[1,11],covered_rows=2)
        self.ref=reference(self.request,self.record);values=self.validate()
        self.assertEqual(values['accepted_draft_tokens'],1);self.assertEqual(values['native_accepted_draft_tokens'],0)
    def test_tampered_posterior_position_and_counts_are_rejected(self):
        for mutate in (lambda r:r['metadata']['trace'][0]['target_argmax'].__setitem__(0,99),
                       lambda r:r['metadata']['trace'][0]['positions'].__setitem__(0,99),
                       lambda r:r['metadata'].__setitem__('verification_calls',2)):
            saved=copy.deepcopy(self.record);mutate(self.record)
            with self.assertRaises(ValueError):self.validate()
            self.record=saved
    def test_zero_call_eos_is_valid_and_nonterminal_early_stop_is_not(self):
        r=self.record;r['response_ids']=[151645];m=r['metadata']
        m.update(trace=[],verification_calls=0,committed_progress=0,accepted_draft_tokens=0,
            committed_progress_per_call=None,calls_per_1000_output_tokens=0.)
        m['capture'].update(response_ids=[151645],target_argmax_ids=[151645],covered_token_ids=[1],covered_rows=1,blocks=[])
        self.ref=reference(self.request,r);self.assertEqual(self.validate()['verification_calls'],0)
        r['response_ids']=[11]
        with self.assertRaisesRegex(ValueError,'termination'):self.validate()
    def test_capture_and_canonical_mismatch_rejected(self):
        self.record['metadata']['capture']['collect_features']=True
        with self.assertRaisesRegex(ValueError,'capture'):self.validate()
        self.record['metadata']['capture']['collect_features']=False
        self.ref['response_ids']=[99]
        with self.assertRaisesRegex(ValueError,'agreement'):self.validate()
    def test_rendered_ids_authentication(self):
        a.validate_input(self.request,a.Checks())
        self.request['prompt_ids']=[99]
        with self.assertRaisesRegex(ValueError,'token hash'):a.validate_input(self.request,a.Checks())


class MemoryEvidence:
    def __init__(self):self.data={};self.inputs={}
    def key(self,name):return str(name).removeprefix('/original/')
    def json(self,name):
        key=self.key(name)
        if key not in self.data:raise a.MissingEvidence(key)
        return self.data[key]
    def lines(self,name):return self.json(name)
    def exists(self,name):return self.key(name) in self.data
    def authenticated(self,name,pin):
        value=self.json(name)
        if pin!=self.identity(value):raise ValueError('identity changed')
        return value
    @staticmethod
    def identity(value):
        data=json.dumps(value).encode();return dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())


def complete_fixture():
    e=MemoryEvidence();panel=[request(i) for i in range(1024)];warm=[request(2000),request(2001)]
    inputs=dict(evaluation_requests=panel,warmup_requests=warm)
    original=dict(train_requests=[request(3000)],evaluation_requests=[request(3001)],qualification_requests=[request(3002)],warmup_requests=warm)
    e.data.update({'/panel':inputs,'/original_manifest':original,'/source':dict(commit='commit',clean=True,files={})})
    paths={m:'/'+m for m in a.METHODS};identities=dict(a.PINNED_WEIGHTS,BASE={'sha256':'BASE','bytes':10},TAIL={'sha256':'TAIL','bytes':10})
    plan=dict(root='/original',methods=list(a.METHODS),execution_commit='commit',execution_source=dict(commit='commit',dirty=False),
        source_manifest='/source',source_identity=e.identity(e.data['/source']),evaluation_inputs='/panel',manifest='/original_manifest',
        devices={'A':'GPU-A'},canonical_provenance={},warmup_canonicals='/warmup',
        retained_endpoints={m:dict(checkpoint=paths[m],checkpoint_identity=identities[m]) for m in ('F','AREPO')},
        input_identities={p:e.identity(e.data[p]) for p in ('/panel','/original_manifest')})
    refs={};canonical_validity=[];serial=0
    for r in panel:
        slot=dict(activation=0,repeat=0,group=0,method='F',phase='canonical',request_ordinal=r['request_index'])
        record=native(r,'/F',slot,serial);serial+=1;refs[r['request_id']]=reference(r,record)
        path='canonicals/raw/'+a.label(slot)+'.json';e.data[path]=record;e.data[e.key(record['metadata']['native_trace_path'])]=record['metadata']['capture']
        canonical_validity.append(dict(slot,valid_measurement=True,errors=[],raw_path='/original/'+path))
    wr={r['request_id']:reference(r,native(r,'/F',dict(activation=0,phase='warmup',method='F',request_ordinal=r['request_index']))) for r in warm}
    e.data['/warmup']=wr;plan['input_identities']['/warmup']=e.identity(wr);plan['binding']=a.digest(plan)
    e.data.update({'canonicals/CANONICALS.json':refs,'canonicals/INPUTS.json':panel,
        'canonicals/PROTOCOL.json':dict(checkpoint='/F',execution_commit='commit',canonical_provenance={},gpu='GPU-A'),
        'canonicals/STATUS.json':dict(status='completed',planned=1024,attempted=1024,returned=1024,completed=1024,pending=None,invalid=0,unknown=0,unattempted=0),
        'canonicals/VALIDITY.jsonl':canonical_validity,'evaluation/EXPORTS_UNCHANGED.json':identities})
    for arm in ('BASE','TAIL'):
        e.data[f'campaign_001/arms/{arm}/STATUS.json']=dict(status='completed',binding=plan['binding'],execution_commit='commit',
            events_completed=58,global_step=7424,presentations=14848,checkpoint=paths[arm],checkpoint_identity=identities[arm])
    entries,slots=a.planned_schedule(panel,warm);rows=[]
    for slot,r in slots:
        record=native(r,paths[slot['method']],slot,serial);serial+=1
        path='evaluation/raw/'+a.label(slot)+'.json';e.data[path]=record;e.data[e.key(record['metadata']['native_trace_path'])]=record['metadata']['capture']
        values=a.validate_record(record,r,(wr if slot['phase']=='warmup' else refs)[r['request_id']],paths[slot['method']],slot,a.Checks())
        rows.append(dict(slot,request_id=r['request_id'],source_id=r['source_id'],checkpoint_id=paths[slot['method']],
            physical_gpu_uuid='GPU-A',valid_measurement=True,errors=[],raw_path='/original/'+path,**values))
    e.data.update({'evaluation/PROTOCOL.json':dict(schedule=entries,checkpoints=paths,execution_commit='commit',gpu='GPU-A',canonical_provenance={},native_record_session=a.SESSION,feedback_capture=False),
        'evaluation/INPUTS.json':dict(panel=panel,warmup=warm),'evaluation/REFERENCES.json':dict(panel=refs,warmup=wr),
        'evaluation/MEASUREMENTS.jsonl':rows,'evaluation/STATUS.json':dict(status='completed',planned=12384,attempted=12384,returned=12384,completed_valid=12384,pending=None,invalid=0,unknown=0,unattempted=0,resume_warmup_generations=0)})
    def activation(entry,method):
        publication=dict(status='passed',checkpoint_path=paths[method],cache_after_tokens=0,
            helper_matches_current_layers=True,helper_norm_matches_current_layers=True,fused_enabled=True)
        return dict(entry,weight_errors=[],loading_s=0.,publication=dict(checkpoint_structure_valid=True,checkpoint_id=paths[method],activation_s=.1,
            native_status=dict(status='passed',poisoned=None,capture_pending=False,cache_tokens=0,attention_backend='fa3',block_size=16,publication=publication)))
    e.data['canonicals/ACTIVATIONS.jsonl']=[activation(dict(activation='canonical'),'F')]
    e.data['evaluation/ACTIVATIONS.jsonl']=[activation(entry,entry['method']) for entry in entries]
    for phase,methods in (('canonicals',('F',)),('evaluation',a.METHODS)):
        for m in methods:e.data[f'{phase}/WEIGHTS_{m}.json']=dict(errors=[],receipt=dict(status='passed',all_equal=True,tensors=58,source_tensor_count=58,native_tensor_count=43,checkpoint_path=paths[m],helper_matches_current_layers=True,helper_norm_matches_current_layers=True,fused_enabled=True))
    quiet=dict(status='passed',physical_gpu_uuid='GPU-A',transfers=[],training=[],foreign_gpu_processes=[],active=[])
    e.data['evaluation/BOUNDARIES.jsonl']=[dict(slot=entry,quiet=quiet) for entry in entries]+[dict(slot={'activation':'finished'},quiet=quiet)]
    return plan,e


class CompleteAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.plan,cls.evidence=complete_fixture()
    def test_all_fixed_slots_and_canonicals(self):
        check=a.Checks();result=a.audit(self.plan,self.evidence,check)
        self.assertEqual((result['scored'],result['warmups'],result['canonical']),(12288,96,1024))
        self.assertEqual(result['actual_activations'],49);self.assertGreater(check.count,800000)
    def test_missing_and_invalid_work_never_pass(self):
        e=self.evidence;rows=e.data['evaluation/MEASUREMENTS.jsonl'];last=rows.pop()
        try:
            with self.assertRaisesRegex(ValueError,'complete planned'):a.audit(self.plan,e,a.Checks())
        finally:rows.append(last)
        old=rows[0]['valid_measurement'];rows[0]['valid_measurement']=False
        try:
            with self.assertRaisesRegex(ValueError,'measured validity'):a.audit(self.plan,e,a.Checks())
        finally:rows[0]['valid_measurement']=old
    def test_balance_and_source_binding_fail_closed(self):
        e=self.evidence;entries=e.data['evaluation/PROTOCOL.json']['schedule'];old=entries[0]['method'];entries[0]['method']='F'
        try:
            with self.assertRaisesRegex(ValueError,'schedule'):a.audit(self.plan,e,a.Checks())
        finally:entries[0]['method']=old
        changed=dict(self.plan,execution_commit='other')
        with self.assertRaisesRegex(ValueError,'plan binding'):a.audit(changed,e,a.Checks())
    def test_missing_raw_is_explicit(self):
        e=self.evidence;name='canonicals/raw/a000_canonical_0000_F.json';saved=e.data.pop(name)
        try:
            with self.assertRaises(a.MissingEvidence):a.audit(self.plan,e,a.Checks())
        finally:e.data[name]=saved


class RelocationTests(unittest.TestCase):
    def test_exact_external_bytes_and_no_unmapped_fallback(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'record.json';data=b'{"value":1}\n';path.write_bytes(data)
            pin=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
            e=a.Evidence(folder,'/original',{'/external/file':str(path)})
            self.assertEqual(e.authenticated('/external/file',pin),{'value':1})
            with self.assertRaises(a.MissingEvidence):e.json('/unmapped/file')
            path.write_text('{"value":2}\n')
            with self.assertRaisesRegex(ValueError,'identity'):e.authenticated('/external/file',pin)
    def test_incomplete_run_gets_no_pass(self):
        with tempfile.TemporaryDirectory() as folder:
            plan=dict(root='/original',execution_commit='commit',methods=list(a.METHODS),execution_source=dict(commit='commit',dirty=False),source_manifest='/missing',source_identity={})
            plan['binding']=a.digest(plan)
            result,_=a.run(plan,folder)
            self.assertEqual(result['status'],'incomplete');self.assertIn('missing',result)


if __name__=='__main__':unittest.main()
