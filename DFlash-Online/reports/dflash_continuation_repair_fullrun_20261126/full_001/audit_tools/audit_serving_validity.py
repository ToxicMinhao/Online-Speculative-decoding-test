#!/usr/bin/env python3
"""Independent CPU validation of the fixed PR66 native serving records.

No model, experimental validator or statistical reducer is imported. Agreement
is with native frozen DFlash, not an independent target-only autoregressive run.
Tensor bytes are covered by the companion live postflight, not reconstructed here.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import unicodedata

METHODS=('BASE','TAIL','AREPO','F')
ORDERS=(('BASE','TAIL','F','AREPO'),('TAIL','AREPO','BASE','F'),
        ('AREPO','F','TAIL','BASE'),('F','BASE','AREPO','TAIL'))
COUNTS=('output_tokens','verification_calls','accepted_draft_tokens','native_accepted_draft_tokens','committed_progress')
SESSION='opencode_final58'
PINNED_WEIGHTS={
    'AREPO':dict(bytes=2097259104,sha256='25540287dcb37baa2386fb7fe57c5b823f2976a84269bf8a018c5779a62b7c30'),
    'F':dict(bytes=2097259104,sha256='c702878094f38ad5843e6ac40b327720b3147ffbf669a5d9ca1974865ca6c080')}


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def text_hash(value):
    return hashlib.sha256(value.encode()).hexdigest()


class MissingEvidence(Exception):pass


class Checks:
    def __init__(self):self.count=0
    def require(self,condition,label):
        self.count+=1
        if not condition:raise ValueError(label)
    def finite_nonnegative(self,value,label):
        self.require(type(value) in (int,float) and math.isfinite(value) and value>=0,label)


class Evidence:
    """Read relocated retained files; outside-root locators need explicit mapping."""
    def __init__(self,root,original_root,external=None):
        self.root=Path(root).resolve();self.original_root=Path(original_root)
        self.external=external or {};self.inputs={}
    def path(self,name):
        value=Path(name)
        if str(value) in self.external:return Path(self.external[str(value)])
        if not value.is_absolute():return self.root/value
        if value.is_relative_to(self.original_root):return self.root/value.relative_to(self.original_root)
        if value.is_relative_to(self.root):return value
        raise MissingEvidence('Unmapped retained external evidence: '+str(value))
    def raw(self,name):
        path=self.path(name)
        if not path.is_file():raise MissingEvidence('Missing retained file: '+str(name))
        data=path.read_bytes()
        self.inputs[str(name)]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
        return data
    def json(self,name):return json.loads(self.raw(name))
    def lines(self,name):return [json.loads(line) for line in self.raw(name).splitlines() if line.strip()]
    def exists(self,name):return self.path(name).exists()
    def authenticated(self,name,pin):
        data=self.raw(name)
        if self.inputs[str(name)]!=pin:raise ValueError('Retained file identity changed: '+str(name))
        return json.loads(data)


def label(slot):
    return f"a{slot['activation']:03d}_{slot['phase']}_{slot['request_ordinal']:04d}_{slot['method']}"


def validate_input(row,check):
    ids=row['prompt_ids']
    check.require(isinstance(ids,list) and 1<=len(ids)<=2048 and all(type(x)is int and x>=0 for x in ids),'input token IDs/context')
    check.require(row['prompt_tokens']==len(ids) and row['prompt_ids_sha256']==digest(ids),'sealed rendered token hash')
    check.require(row['rendered_prompt_sha256']==text_hash(row['rendered_prompt']),'rendered text hash')
    check.require(row['raw_input_sha256']==text_hash(row['raw_input']),'raw input hash')
    normalized=' '.join(unicodedata.normalize('NFKC',row['raw_input']).split())
    check.require(row['normalized_raw_sha256']==text_hash(normalized),'normalized input hash')
    check.require(row['source_identity_sha256']==digest(row['source_identity']),'source coordinate hash')
    check.require(row['source_id']==row['request_id']=='opencode:'+row['source_identity_sha256'],'source/request binding')
    check.require(row['stop_token_ids']==[151645,151643] and row['max_new_tokens']==2048,'fixed stop/cap')


def validate_reference(reference,request,plan,check,*,fresh=False):
    check.require(reference['binding_digest']==digest({k:v for k,v in reference.items() if k!='binding_digest'}),'canonical binding')
    check.require(reference['verified'] is True and reference['reference_kind']=='frozen_DFlash','native F canonical kind')
    check.require(reference['provenance']==plan['canonical_provenance'],'canonical source provenance')
    for key in ('request_id','prompt_ids','stop_token_ids','max_new_tokens'):
        check.require(reference[key]==request[key],'canonical input '+key)
    check.require(reference['source_checkpoint']==plan['retained_endpoints']['F']['checkpoint'],'canonical F checkpoint')
    if fresh:check.require(reference['execution_commit']==plan['execution_commit'],'fresh canonical source commit')


def validate_record(record,request,reference,checkpoint,slot,check):
    """Reconstruct every committed token from native proposal/target decisions."""
    for key,expected in dict(record_id=label(slot),session_id=SESSION,request_index=request['request_index'],
            checkpoint_id=checkpoint,purpose='evaluation',prompt_ids=request['prompt_ids'],
            max_new_tokens=2048,stop_token_ids=request['stop_token_ids']).items():
        check.require(record[key]==expected,'native identity '+key)
    output=record['response_ids'];stops=request['stop_token_ids']
    check.require(isinstance(output,list) and 1<=len(output)<=2048 and all(type(x)is int and x>=0 for x in output),'native output token shape')
    check.require(not any(x in stops for x in output[:-1]),'no output after first stop')
    finish='eos' if output[-1] in stops else 'length'
    check.require(finish=='eos' or len(output)==2048,'native termination at EOS or cap')
    check.require(record['finish_reason']==finish,'semantic termination')
    for key in ('prompt_ids','response_ids','finish_reason','stop_token_ids','max_new_tokens'):
        check.require(record[key]==reference[key],'native F agreement '+key)
    meta=record['metadata'];capture=meta['capture'];blocks=meta['trace']
    check.require(meta['block_size']==16 and meta['context_limit']==4096,'native block/context contract')
    check.require(meta['runtime']=='pinned_SGLang_FA3' and meta['cache_policy']=='idle_full_flush_before_every_request_and_publication','native runtime/cache contract')
    check.require(meta['feedback_capture_in_timed_request'] is False and meta['trace_instrumentation_in_timed_request'] is True,'native feedback capture disabled')
    check.require(capture['status']=='passed' and capture['native_capture_validated'] is True and capture['collect_features'] is False and capture['feature_bytes']==0,'capture diagnostic contract')
    check.require(capture['prompt_ids']==request['prompt_ids'] and capture['response_ids']==output and capture['blocks']==blocks,'capture/raw binding')
    check.require(capture['target_argmax_ids']==output and capture['covered_token_ids']==request['prompt_ids']+output[:-1],'native label sequence reconstruction')
    check.require(capture['covered_rows']==len(request['prompt_ids'])+len(output)-1 and capture['block_size']==16,'capture row geometry')
    check.require(capture['final_feature_logits_used_for_labels'] is False and capture['final_emitted_token_feature_present'] is False,'terminal feature policy')
    check.require(capture['label_binding_scope']=='native_prefill_and_committed_verification','native conditioning scope')
    offset=0;clipped=0;native=0
    for ordinal,block in enumerate(blocks):
        width=block['verify_size'];proposal=block['draft_tokens'];target=block['target_argmax'];progress=block['committed_progress']
        check.require(type(width)is int and 1<=width<=16 and len(proposal)==width-1 and len(target)==width,'verification width')
        check.require(all(type(x)is int and x>=0 for x in proposal+target),'native proposal/posterior tokens')
        check.require(type(progress)is int and 1<=progress<=width,'committed progress bounds')
        accepted=0
        for proposed,verified in zip(proposal,target):
            if proposed!=verified:break
            accepted+=1
        check.require(progress<=accepted+1,'native prefix acceptance bound')
        proposed_commit=proposal[:accepted]+[target[accepted]]
        check.require(block['output_start']==offset and block['anchor_token']==output[offset],'native round anchor')
        check.require(block['committed_tokens']==proposed_commit[:progress]==target[:progress]==output[offset+1:offset+1+progress],'committed native token reconstruction')
        start=len(request['prompt_ids'])+offset
        check.require(block['positions']==list(range(start,start+width)),'native absolute verification positions')
        check.require(block['accepted_draft_tokens']==min(accepted,progress),'stop-clipped accepted tokens')
        check.require(block['native_accepted_draft_tokens']==progress-1,'native accepted-token accounting')
        check.require(block['raw_accepted_draft_tokens_before_stop']==accepted and block['stop_clipped']==(progress<accepted+1),'raw versus stop-clipped acceptance')
        check.require(block['finished'] is (ordinal==len(blocks)-1),'native terminal round')
        offset+=progress;clipped+=min(accepted,progress);native+=progress-1
    check.require(offset==len(output)-1,'trace covers entire response after prefill token')
    calls=len(blocks)
    check.require(meta['verification_calls']==calls and type(meta['verification_calls'])is int,'verification-call count')
    check.require(type(meta['committed_progress'])is int and type(meta['accepted_draft_tokens'])is int,'integer cumulative counters')
    check.require(meta['committed_progress']==offset and meta['accepted_draft_tokens']==clipped,'raw cumulative native counters')
    check.require(native==offset-calls,'native acceptance versus calls')
    check.require(meta['committed_progress_per_call']==(offset/calls if calls else None),'progress per verification')
    check.require(meta['calls_per_1000_output_tokens']==1000*calls/len(output),'calls per token ratio')
    for key in ('request_latency_s','native_generate_s'):check.finite_nonnegative(meta[key],key)
    check.require(meta['request_latency_s']>0 and meta['native_generate_s']>0,'positive native request timers')
    check.finite_nonnegative(capture['cache_maintenance_s'],'cache maintenance time')
    check.require(meta['request_latency_s']>=meta['native_generate_s'],'nested timing bounds')
    return dict(output_tokens=len(output),verification_calls=calls,accepted_draft_tokens=clipped,
        native_accepted_draft_tokens=native,committed_progress=offset,finish_reason=finish,
        request_latency_s=meta['request_latency_s'],native_generate_s=meta['native_generate_s'],
        cache_maintenance_s=capture['cache_maintenance_s'])


def planned_schedule(panel,warmups):
    entries=[];slots=[]
    for repeat in range(3):
        for group in range(4):
            selected=panel[group*256:(group+1)*256]
            for position,method in enumerate(ORDERS[(repeat+group)%4]):
                activation=len(entries)
                entries.append(dict(activation=activation,repeat=repeat,group=group,position=position,
                    method=method,warmups=2,request_ids=[r['request_id'] for r in selected]))
                for phase,requests in (('warmup',warmups),('scored',selected)):
                    for request in requests:
                        slots.append((dict(activation=activation,repeat=repeat,group=group,method=method,
                            phase=phase,request_ordinal=request['request_index']),request))
    return entries,slots


def validate_publication(row,checkpoint,check):
    receipt=row['publication'];native=receipt['native_status'];publication=native['publication']
    check.require(receipt['checkpoint_structure_valid'] is True and receipt['checkpoint_id']==checkpoint,'published checkpoint identity')
    check.require(native['status']=='passed' and native['poisoned'] is None and native['capture_pending'] is False,'publication worker healthy')
    check.require(native['cache_tokens']==publication['cache_after_tokens']==0,'publication cache empty')
    check.require(native['attention_backend']=='fa3' and native['block_size']==16,'publication runtime')
    check.require(publication['status']=='passed' and publication['checkpoint_path']==checkpoint,'publication validated path')
    for key in ('helper_matches_current_layers','helper_norm_matches_current_layers','fused_enabled'):
        check.require(publication[key] is True,'publication helper '+key)
    check.require(not row['weight_errors'],'publication weight errors absent')
    for value in (row['loading_s'],receipt['activation_s']):check.finite_nonnegative(value,'activation/loading cost')


def validate_weights(evidence,phase,method,checkpoint,check):
    outer=evidence.json(f'{phase}/WEIGHTS_{method}.json');row=outer['receipt']
    check.require(not outer['errors'] and row['status']=='passed' and row['all_equal'] is True,'loaded endpoint exact tensor receipt')
    check.require(row['tensors']==row['source_tensor_count']==58 and row['native_tensor_count']==43 and row['checkpoint_path']==checkpoint,'loaded tensor membership/checkpoint')
    for key in ('helper_matches_current_layers','helper_norm_matches_current_layers','fused_enabled'):
        check.require(row[key] is True,'loaded weight helper '+key)


def audit(plan,evidence,check):
    check.require(plan['binding']==digest({k:v for k,v in plan.items() if k!='binding'}),'sealed plan binding')
    check.require(plan['methods']==list(METHODS) and plan['execution_source']==dict(commit=plan['execution_commit'],dirty=False),'fixed endpoints/clean execution')
    source=evidence.authenticated(plan['source_manifest'],plan['source_identity'])
    check.require(source['commit']==plan['execution_commit'] and source['clean'] is True,'source manifest execution binding')
    inputs=evidence.authenticated(plan['evaluation_inputs'],plan['input_identities'][plan['evaluation_inputs']])
    original=evidence.authenticated(plan['manifest'],plan['input_identities'][plan['manifest']])
    panel,warmups=inputs['evaluation_requests'],inputs['warmup_requests']
    check.require(len(panel)==1024 and len(warmups)==2 and warmups==original['warmup_requests'],'exact sealed panel and retained warmups')
    check.require([r['request_index'] for r in panel]==list(range(1024)),'sealed chronological panel indices')
    for row in panel+warmups:validate_input(row,check)
    for key in ('request_id','source_id','normalized_raw_sha256','rendered_prompt_sha256','prompt_ids_sha256'):
        check.require(len({r[key] for r in panel+warmups})==1026,'panel/warmup uniqueness '+key)
    for role in ('train','evaluation','qualification','warmup'):
        for key in ('request_id','normalized_raw_sha256','rendered_prompt_sha256','prompt_ids_sha256'):
            check.require(not {r[key] for r in panel}&{r[key] for r in original[role+'_requests']},'original input exclusion '+role+'/'+key)
    checkpoints={m:plan['retained_endpoints'][m]['checkpoint'] for m in ('AREPO','F')}
    identities={m:plan['retained_endpoints'][m]['checkpoint_identity'] for m in ('AREPO','F')}
    check.require(identities==PINNED_WEIGHTS,'fixed retained AREPO/F tensor identities')
    for arm in ('BASE','TAIL'):
        state=evidence.json(f'campaign_001/arms/{arm}/STATUS.json')
        check.require(state['status']=='completed' and state['binding']==plan['binding'] and state['execution_commit']==plan['execution_commit'],'own completed endpoint provenance')
        check.require([state['events_completed'],state['global_step'],state['presentations']]==[58,7424,14848],'own full endpoint counters')
        checkpoints[arm]=state['checkpoint'];identities[arm]=state['checkpoint_identity']
    check.require(len(set(checkpoints.values()))==4,'four distinct checkpoint locators')
    check.require(evidence.json('evaluation/EXPORTS_UNCHANGED.json')==identities,'endpoint export identities unchanged')
    canonicals=evidence.json('canonicals/CANONICALS.json')
    warmup_refs=evidence.authenticated(plan['warmup_canonicals'],plan['input_identities'][plan['warmup_canonicals']])
    warmup_refs={key:(value['F'] if set(value)=={'AR','F'} else value) for key,value in warmup_refs.items()}
    check.require(set(canonicals)=={r['request_id'] for r in panel} and set(warmup_refs)=={r['request_id'] for r in warmups},'canonical coverage')
    check.require(evidence.json('canonicals/INPUTS.json')==panel,'canonical inputs sealed')
    check.require(evidence.json('canonicals/PROTOCOL.json')==dict(checkpoint=checkpoints['F'],execution_commit=plan['execution_commit'],canonical_provenance=plan['canonical_provenance'],gpu=plan['devices']['A']),'canonical protocol source/checkpoint/device')
    canonical_status=evidence.json('canonicals/STATUS.json')
    check.require(canonical_status['status']=='completed' and all(canonical_status[k]==1024 for k in ('planned','attempted','returned','completed')) and canonical_status['pending'] is None and all(canonical_status[k]==0 for k in ('invalid','unknown','unattempted')),'complete canonical execution status')
    canonical_validity=evidence.lines('canonicals/VALIDITY.jsonl')
    check.require(len(canonical_validity)==1024,'all canonical raw validity rows')
    for number,request in enumerate(panel):
        ref=canonicals[request['request_id']];validate_reference(ref,request,plan,check,fresh=True)
        slot=dict(activation=0,repeat=0,group=0,method='F',phase='canonical',request_ordinal=request['request_index'])
        record=evidence.json('canonicals/raw/'+label(slot)+'.json')
        validate_record(record,request,ref,checkpoints['F'],slot,check)
        check.require(ref['source_record_id']==label(slot),'fresh canonical source record')
        check.require(evidence.json(record['metadata']['native_trace_path'])==record['metadata']['capture'],'separate canonical native trace')
        row=canonical_validity[number]
        check.require(all(row[k]==v for k,v in slot.items()) and row['valid_measurement'] is True and not row['errors'],'canonical validity receipt')
        check.require(row['raw_path']==str(Path(plan['root'])/'canonicals/raw'/(label(slot)+'.json')),'canonical validity raw locator')
    for request in warmups:validate_reference(warmup_refs[request['request_id']],request,plan,check)
    expected,slots=planned_schedule(panel,warmups)
    protocol=evidence.json('evaluation/PROTOCOL.json')
    check.require(protocol['schedule']==expected and protocol['checkpoints']==checkpoints and protocol['execution_commit']==plan['execution_commit'],'exact native scoring schedule/source')
    check.require(protocol['gpu']==plan['devices']['A'] and protocol['canonical_provenance']==plan['canonical_provenance'] and protocol['native_record_session']==SESSION and protocol['feedback_capture'] is False,'scoring device/reference/capture contract')
    check.require(evidence.json('evaluation/INPUTS.json')==dict(panel=panel,warmup=warmups),'scoring sealed inputs')
    check.require(evidence.json('evaluation/REFERENCES.json')==dict(panel=canonicals,warmup=warmup_refs),'scoring canonical dictionary binding')
    rows=evidence.lines('evaluation/MEASUREMENTS.jsonl')
    check.require(len(rows)==len(slots)==12384,'complete planned serving rows')
    repeats=defaultdict(set);validations=Counter();native_blocks=0
    for row,(slot,request) in zip(rows,slots):
        check.require(all(row[k]==value for k,value in slot.items()),'scored/warmup chronological slot identity')
        check.require(row['request_id']==request['request_id'] and row['source_id']==request['source_id'] and row['checkpoint_id']==checkpoints[slot['method']],'measured input/endpoint binding')
        check.require(row['valid_measurement'] is True and not row['errors'] and row['physical_gpu_uuid']==plan['devices']['A'],'measured validity/device')
        path='evaluation/raw/'+label(slot)+'.json'
        check.require(row['raw_path']==str(Path(plan['root'])/path),'raw slot locator binding')
        record=evidence.json(path);reference=(warmup_refs if slot['phase']=='warmup' else canonicals)[request['request_id']]
        values=validate_record(record,request,reference,checkpoints[slot['method']],slot,check)
        for key,value in values.items():check.require(row[key]==value,'independently reconstructed metric '+key)
        check.require(evidence.json(record['metadata']['native_trace_path'])==record['metadata']['capture'],'separate native trace matches record')
        repeats[slot['phase'],slot['method'],request['request_id']].add(tuple(values[k] for k in (*COUNTS,'finish_reason')))
        validations[slot['phase']]+=1;native_blocks+=values['verification_calls']
    check.require(all(len(value)==1 for value in repeats.values()),'each endpoint repeat-consistent native counts')
    status=evidence.json('evaluation/STATUS.json')
    check.require(status['status']=='completed' and all(status[k]==12384 for k in ('planned','attempted','returned','completed_valid')) and status['pending'] is None and all(status[k]==0 for k in ('invalid','unknown','unattempted')),'complete valid/invalid/unknown classification')
    activation_count=0
    for phase in ('canonicals','evaluation'):
        activations=evidence.lines(phase+'/ACTIVATIONS.jsonl');activation_count+=len(activations)
        check.require(bool(activations),'activation evidence present')
        if phase=='evaluation':
            check.require([r['activation'] for r in activations]==sorted(r['activation'] for r in activations),'chronological activation order including resumes')
        seen=set()
        for row in activations:
            method='F' if phase=='canonicals' else row['method']
            if phase=='evaluation':
                entry=expected[row['activation']]
                check.require(all(row[k]==v for k,v in entry.items()),'balanced activation order')
                seen.add(row['activation'])
            validate_publication(row,checkpoints[method],check)
        if phase=='evaluation':check.require(seen==set(range(48)),'all48 successful planned activations')
        for method in (('F',) if phase=='canonicals' else METHODS):validate_weights(evidence,phase,method,checkpoints[method],check)
    boundaries=evidence.lines('evaluation/BOUNDARIES.jsonl')
    check.require(len(boundaries)>=49 and boundaries[-1]['slot']=={'activation':'finished'},'quiet timing boundaries cover final end')
    check.require({r['slot']['activation'] for r in boundaries if r['slot']['activation']!='finished'}==set(range(48)),'quiet checks cover each planned activation')
    for row in boundaries:
        if row['slot']['activation']!='finished':
            check.require(row['slot']==expected[row['slot']['activation']],'quiet check sealed activation identity')
        quiet=row['quiet']
        check.require(quiet['status']=='passed' and quiet['physical_gpu_uuid']==plan['devices']['A'],'quiet single-card timing receipt')
        for key in ('transfers','training','foreign_gpu_processes'):check.require(not quiet[key],'quiet timing '+key)
        check.require(all(x['physical_gpu_uuid']==plan['devices']['A'] for x in quiet['active']),'other physical card idle')
    retry_rows=evidence.lines('evaluation/RETRY_MEASUREMENTS.jsonl') if evidence.exists('evaluation/RETRY_MEASUREMENTS.jsonl') else []
    check.require(len(retry_rows)==status['resume_warmup_generations'],'explicit resumed warmup accounting')
    for row in retry_rows:
        check.require(row['phase']=='resume_warmup' and row['valid_measurement'] is True and not row['errors'],'valid extra resume warmup')
        request=next(r for r in warmups if r['request_id']==row['request_id'])
        slot={key:row[key] for key in ('activation','repeat','group','method','phase','request_ordinal')}
        record=evidence.json(row['raw_path'])
        values=validate_record(record,request,warmup_refs[request['request_id']],checkpoints[row['method']],slot,check)
        for key,value in values.items():check.require(row[key]==value,'extra warmup metric '+key)
        check.require(evidence.json(record['metadata']['native_trace_path'])==record['metadata']['capture'],'separate retry native trace')
    return dict(scored=validations['scored'],warmups=validations['warmup'],canonical=1024,
        extra_resume_warmups=len(retry_rows),planned_activations=48,actual_activations=activation_count,
        scored_and_warmup_native_blocks=native_blocks,invalid=0,unknown=0,unattempted=0)


def run(plan,root,external=None):
    evidence=Evidence(root,plan['root'],external);check=Checks()
    result=dict(status='running',plan_binding=plan['binding'],execution_commit=plan['execution_commit'],
        scope='Independent CPU native-F token/termination/trace/order/metric validation; no target-only AR or model re-execution',
        independence='Standard-library implementation without production validators; shared project context and serving implementation authorship disclosed',
        limitation='Rendered/token hashes and pinned input bytes are checked; tokenizer inference and endpoint tensor reads belong to source/input preflight and live postflight.',new_model_work=0)
    try:
        result['counts']=audit(plan,evidence,check);result['status']='passed'
    except MissingEvidence as error:result.update(status='incomplete',missing=str(error))
    except (AssertionError,KeyError,TypeError,ValueError,IndexError,OSError,StopIteration) as error:
        result.update(status='failed',error=repr(error))
    result['checks']=check.count
    result['input_manifest_digest']=digest(evidence.inputs)
    return result,evidence.inputs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',required=True);p.add_argument('--raw-root',required=True)
    p.add_argument('--external-map');p.add_argument('--output',required=True)
    args=p.parse_args();plan=json.loads(Path(args.plan).read_text())
    external=json.loads(Path(args.external_map).read_text()) if args.external_map else None
    result,inputs=run(plan,args.raw_root,external)
    destination=Path(args.output);destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    with destination.with_name(destination.stem+'_PRIVATE_INPUTS.json').open('x') as stream:
        json.dump(inputs,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status=result['status'],checks=result['checks'])))
    return 0 if result['status']=='passed' else 2


if __name__=='__main__':raise SystemExit(main())
