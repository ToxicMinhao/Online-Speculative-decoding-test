#!/usr/bin/env python3
"""Independent, CPU-only completed-run audit. No production reducer imports.

Accept an execution root (native path or faithful local copy). Heavy checks
refuse an incomplete launch, a running scored worker or any active card lease.
Only --output is written; evidence, weights and scientific source are read-only.
"""
from __future__ import annotations
import argparse, csv, hashlib, itertools, json, math, sys, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ARMS=('O58','AREPO58','KPLUS_C58')
CONTRASTS=(('KPLUS_C58','AREPO58'),('KPLUS_C58','O58'),('AREPO58','O58'))
COUNTS=('verification_calls','native_accepted_draft_tokens','accepted_draft_tokens','committed_progress','output_tokens')
TIMES=('request_latency_s','native_generate_s','cache_maintenance_s')
FIELDS=COUNTS+TIMES
SESSION='fresh_mathqa_c_arepo_osd_event58'
SCOPE='FRESH_MATHQA_C_AREPO_OSD_ENDPOINT'
LIMITS=dict(qualification_generation=24,canonical_generation=1024,warmup_generation=288,scored_generation=9216)
PROVENANCE=('target_identity','tokenizer_identity','runtime_identity')

class AuditFailure(ValueError):pass

def require(ok,message):
    if not ok:raise AuditFailure(message)

def encoded(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha(value):return hashlib.sha256(value).hexdigest()
def file_id(path):
    with Path(path).open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    return dict(bytes=Path(path).stat().st_size,sha256=digest)
def ratio(a,b):return float(a/b) if b>0 else None

def equal(expected,actual,label='value'):
    """Integer/string identities exact; floats permit summation-roundoff only."""
    if isinstance(expected,dict):
        require(isinstance(actual,dict),label+': expected object')
        for key,value in expected.items():
            require(key in actual,label+': missing '+key);equal(value,actual[key],label+'.'+key)
    elif isinstance(expected,(list,tuple)):
        require(isinstance(actual,(list,tuple)) and len(expected)==len(actual),label+': sequence size')
        for i,(a,b) in enumerate(zip(expected,actual)):equal(a,b,f'{label}[{i}]')
    elif isinstance(expected,float):
        require(type(actual) in (int,float) and math.isfinite(actual) and math.isclose(expected,actual,rel_tol=1e-11,abs_tol=1e-12),label+': numeric disagreement')
    else:
        require(type(expected) is type(actual) and expected==actual,label+': exact disagreement')

def interval(point,samples):
    samples=np.asarray(samples);bad=~np.isfinite(samples)
    return dict(point=point,ci95=None if bad.any() else np.quantile(samples,[.025,.975],method='linear').tolist(),undefined_draws=int(bad.sum()))

def evidence(metric):
    bounds=metric['ci95']
    return 'SUPPORTED_GAIN' if bounds is not None and bounds[0]>0 else 'SUPPORTED_LOSS' if bounds is not None and bounds[1]<0 else 'DIRECTION_UNRESOLVED'

def reduce_rows(rows,requests):
    """Three repeats are averaged within a request, never treated as extra n."""
    by_id={r['request_id']:r for r in requests};groups=defaultdict(list)
    require(len(by_id)==len(requests),'duplicate sealed request IDs')
    for row in rows:
        require(row['request_id'] in by_id and row['method'] in ARMS,'unknown scored identity')
        request=by_id[row['request_id']]
        equal({k:request[k] for k in ('source_id','panel_ordinal')},row,'scored input')
        groups[row['request_id'],row['method']].append(row)
    require(len(rows)==len(requests)*9,'scored repeat-table row count')
    paired=[]
    for request_id in sorted(by_id):
        request=by_id[request_id];result={k:request[k] for k in ('request_id','source_id','panel_ordinal')}
        for method in ARMS:
            values=sorted(groups[request_id,method],key=lambda r:r['repeat'])
            require([r['repeat'] for r in values]==[0,1,2],'missing/duplicate timing repeat')
            first=values[0]
            for key in COUNTS:
                require(type(first[key]) is int and first[key]>=0 and all(r[key]==first[key] for r in values),'count identity/repeat '+key)
            require(first['output_tokens']-1==first['committed_progress']==first['verification_calls']+first['native_accepted_draft_tokens'],'progress/native accounting')
            for row in values:
                for key in TIMES:require(type(row[key]) in (int,float) and math.isfinite(row[key]) and row[key]>=(0 if key=='cache_maintenance_s' else 1e-300),'invalid time '+key)
            for key in ('checkpoint_id','physical_gpu_uuid','finish_reason','cap_hit'):
                require(all(r[key]==first[key] for r in values),'repeat identity '+key)
            result[method]={k:first[k] for k in COUNTS}
            result[method].update({k:math.fsum(r[k] for r in values)/3 for k in TIMES})
            result[method].update(checkpoint_id=first['checkpoint_id'],finish_reason=first['finish_reason'],cap_hit=first['cap_hit'],repeat_request_latency_s=[r['request_latency_s'] for r in values])
            if 'native_acceptance_histogram' in first:
                require(all(r['native_acceptance_histogram']==first['native_acceptance_histogram'] for r in values),'repeat histogram')
                result[method]['native_acceptance_histogram']=first['native_acceptance_histogram']
        require(len({(result[m]['output_tokens'],result[m]['finish_reason']) for m in ARMS})==1,'paired output/termination differs')
        paired.append(result)
    return paired

def totals(paired,method):
    values=[row[method] for row in paired]
    result={k:sum(r[k] for r in values) for k in COUNTS}
    result.update({k:math.fsum(r[k] for r in values) for k in TIMES})
    calls=result['verification_calls'];hist=Counter()
    for row in values:hist.update(row.get('native_acceptance_histogram',{}))
    positive=[r for r in values if r['verification_calls']>0]
    result.update(native_accepted_tokens_per_call=ratio(result['native_accepted_draft_tokens'],calls),matched_accepted_tokens_per_call=ratio(result['accepted_draft_tokens'],calls),committed_progress_per_call=ratio(result['committed_progress'],calls),calls_per_1000_output_tokens=ratio(1000*calls,result['output_tokens']),equal_request_native_acceptance_mean=None if not positive else math.fsum(r['native_accepted_draft_tokens']/r['verification_calls'] for r in positive)/len(positive),equal_request_denominator=len(positive),zero_call_requests=len(values)-len(positive),cap_hit_requests=sum(r['cap_hit'] for r in values),native_acceptance_histogram=dict(sorted(hist.items(),key=lambda x:int(x[0]))))
    return result

def statistics(paired,draws=10000,seed=20261119,chunk=256):
    """Independent ratio-of-sums bootstrap; chunking preserves exact RNG stream."""
    require(paired and [r['request_id'] for r in paired]==sorted(r['request_id'] for r in paired),'bootstrap lexical request order')
    arrays={m:{k:np.array([r[m][k] for r in paired],dtype=np.float64) for k in FIELDS} for m in ARMS}
    estimates={m:{k:float(v.sum()) for k,v in fields.items()} for m,fields in arrays.items()}
    samples={a+'_vs_'+b:defaultdict(list) for a,b in CONTRASTS}
    rng=np.random.Generator(np.random.PCG64(seed));draw_hash=hashlib.sha256()
    for start in range(0,draws,chunk):
        indices=rng.integers(0,len(paired),size=(min(chunk,draws-start),len(paired)))
        draw_hash.update(indices.astype('<i8').tobytes())
        sums={m:{k:values[indices].sum(axis=1) for k,values in fields.items() if k in ('request_latency_s','verification_calls','native_accepted_draft_tokens','accepted_draft_tokens')} for m,fields in arrays.items()}
        with np.errstate(divide='ignore',invalid='ignore'):
            for a,b in CONTRASTS:
                x,y=sums[a],sums[b];sink=samples[a+'_vs_'+b]
                xn,yn=x['native_accepted_draft_tokens']/x['verification_calls'],y['native_accepted_draft_tokens']/y['verification_calls']
                xc,yc=x['accepted_draft_tokens']/x['verification_calls'],y['accepted_draft_tokens']/y['verification_calls']
                for key,value in dict(time_reduction=1-x['request_latency_s']/y['request_latency_s'],call_reduction=1-x['verification_calls']/y['verification_calls'],acceptance_improvement=xn/yn-1,acceptance_difference=xn-yn,clipped_absolute=xc-yc,clipped_relative=xc/yc-1).items():sink[key].append(value)
    contrasts={}
    for a,b in CONTRASTS:
        x,y=estimates[a],estimates[b];sink={k:np.concatenate(v) for k,v in samples[a+'_vs_'+b].items()}
        xn,yn=ratio(x['native_accepted_draft_tokens'],x['verification_calls']),ratio(y['native_accepted_draft_tokens'],y['verification_calls'])
        xc,yc=ratio(x['accepted_draft_tokens'],x['verification_calls']),ratio(y['accepted_draft_tokens'],y['verification_calls'])
        delta=arrays[b]['verification_calls']-arrays[a]['verification_calls']
        points=dict(time_reduction=1-x['request_latency_s']/y['request_latency_s'],call_reduction=None if y['verification_calls']==0 else 1-x['verification_calls']/y['verification_calls'],acceptance_improvement=None if xn is None or yn in (None,0) else xn/yn-1,acceptance_difference=None if xn is None or yn is None else xn-yn)
        result={key:interval(value,sink[key]) for key,value in points.items()}
        result.update(candidate=a,reference=b,speedup=y['request_latency_s']/x['request_latency_s'],repeat_time_reductions=[1-math.fsum(r[a]['repeat_request_latency_s'][j] for r in paired)/math.fsum(r[b]['repeat_request_latency_s'][j] for r in paired) for j in range(3)],requests_fewer_calls=int((delta>0).sum()),requests_more_calls=int((delta<0).sum()),request_call_ties=int((delta==0).sum()),gross_calls_saved=int(delta[delta>0].sum()),gross_calls_added=int(-delta[delta<0].sum()),net_calls_saved=int(delta.sum()))
        result['evidence']={k:evidence(result[k]) for k in points}
        result['clipped_acceptance']=dict(candidate_tokens_per_call=xc,reference_tokens_per_call=yc,absolute_difference=interval(None if xc is None or yc is None else xc-yc,sink['clipped_absolute']),relative_change=interval(None if xc is None or yc in (None,0) else xc/yc-1,sink['clipped_relative']))
        contrasts[a+'_vs_'+b]=result
    order=[dict(source_id=r['source_id'],request_id=r['request_id']) for r in paired]
    return dict(method_totals={m:totals(paired,m) for m in ARMS},comparisons=contrasts,bootstrap=dict(draws=draws,seed=seed,generator='PCG64',method='linear',order='lexicographic request_id',source_order=order,source_order_sha256=sha(encoded(order)),draw_matrix_sha256=draw_hash.hexdigest()))

def trace_record(record,request,reference,checkpoint,label,session):
    """Rebuild native and stop-clipped counters from proposal/posterior traces."""
    equal(dict(record_id=label,session_id=session,request_index=request['panel_ordinal'],checkpoint_id=checkpoint,purpose='evaluation',prompt_ids=request['prompt_ids'],stop_token_ids=request['stop_token_ids'],max_new_tokens=2048),record,'raw generation identity')
    output=record['response_ids'];stop=request['stop_token_ids'];finish=record['finish_reason']
    require(0<len(output)<=2048 and all(type(x) is int and x>=0 for x in output),'output tokens')
    require(not any(x in stop for x in output[:-1]),'early stop token')
    require((finish=='eos' and output[-1] in stop) or (finish=='length' and len(output)==2048 and output[-1] not in stop),'termination/cap')
    if reference is not None:equal({k:reference[k] for k in ('response_ids','finish_reason')},record,'canonical output equality')
    meta=record['metadata'];capture=meta['capture'];trace=meta['trace']
    equal(dict(block_size=16,context_limit=4096,runtime='pinned_SGLang_FA3',cache_policy='idle_full_flush_before_every_request_and_publication',feedback_capture_in_timed_request=False),meta,'native configuration')
    equal(dict(status='passed',collect_features=False,feature_bytes=0,native_capture_validated=True,
        label_binding_scope='native_prefill_and_committed_verification',
        target_argmax_ids=output,covered_token_ids=request['prompt_ids']+output[:-1],
        covered_rows=len(request['prompt_ids'])+len(output)-1,
        prompt_ids=request['prompt_ids'],response_ids=output,blocks=trace),capture,'native capture')
    progress=clipped=native=0;hist=Counter()
    for i,block in enumerate(trace):
        proposed,posterior=block['draft_tokens'],block['target_argmax'];width=block['verify_size'];advance=block['committed_progress']
        require(type(width) is int and 1<=width<=16 and len(proposed)+1==len(posterior)==width,'verification geometry')
        require(all(type(x) is int and x>=0 for x in proposed+posterior),'proposal/posterior token types')
        first=next((j for j in range(width-1) if proposed[j]!=posterior[j]),width-1)
        require(type(advance) is int and 1<=advance<=first+1,'committed prefix length')
        commit=(proposed[:first]+[posterior[first]])[:advance]
        equal(dict(output_start=progress,anchor_token=output[progress],positions=list(range(len(request['prompt_ids'])+progress,len(request['prompt_ids'])+progress+width)),committed_tokens=commit,accepted_draft_tokens=min(first,advance),native_accepted_draft_tokens=advance-1,raw_accepted_draft_tokens_before_stop=first,stop_clipped=advance<first+1),block,f'verification block {i}')
        require(commit==output[progress+1:progress+1+advance] and commit==posterior[:advance],'native committed response sequence')
        progress+=advance;clipped+=min(first,advance);native+=advance-1;hist[str(advance-1)]+=1
    require(progress==len(output)-1 and native==progress-len(trace),'complete trajectory accounting')
    equal(dict(verification_calls=len(trace),accepted_draft_tokens=clipped,committed_progress=progress),meta,'native aggregate counters')
    result=dict(verification_calls=len(trace),native_accepted_draft_tokens=native,accepted_draft_tokens=clipped,committed_progress=progress,output_tokens=len(output),finish_reason=finish,cap_hit=finish=='length',native_acceptance_histogram=dict(hist),request_latency_s=meta['request_latency_s'],native_generate_s=meta['native_generate_s'],cache_maintenance_s=capture['cache_maintenance_s'])
    for key in TIMES:require(type(result[key]) in (float,int) and math.isfinite(result[key]) and result[key]>=(0 if key=='cache_maintenance_s' else 1e-300),'finite measured time')
    return result

class Audit:
    def __init__(self,root):self.root=Path(root).absolute();self.reads={};self.raw_records=0;self.trace_blocks=0
    def json(self,path):
        path=Path(path);blob=path.read_bytes();self.reads[str(path)]=dict(bytes=len(blob),sha256=sha(blob));return json.loads(blob)
    def lines(self,path):return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
    def resolve(self,path):
        path=Path(path)
        if not path.is_absolute():return self.root/path
        try:return self.root/path.relative_to(self.plan['root'])
        except ValueError:return path
    def csv(self,path):
        with Path(path).open(newline='') as stream:rows=list(csv.DictReader(stream))
        for row in rows:
            for k in COUNTS+('panel_ordinal','repeat','block_ordinal','position','activation','verification_ordinal','output_start'):
                if k in row:row[k]=int(row[k])
            for k in TIMES:
                if k in row:row[k]=float(row[k])
            for k in ('cap_hit','stop_clipped'):
                if k in row:require(row[k] in ('True','False'),'CSV boolean');row[k]=row[k]=='True'
        return rows
    def native(self,path,request,reference,checkpoint,label,session):
        record=self.json(path);result=trace_record(record,request,reference,checkpoint,label,session)
        equal(record['metadata']['capture'],self.json(self.resolve(record['metadata']['native_trace_path'])),'separate native sidecar')
        self.raw_records+=1;self.trace_blocks+=len(record['metadata']['trace'])
        return record,result
    def ref(self,request,reference,producer,fresh):
        equal({k:request[k] for k in ('request_id','source_id','prompt_ids','stop_token_ids','max_new_tokens')},reference,'reference input')
        equal(dict(reference_kind='native_greedy_target_verifier',reference_checkpoint=producer,verified=True),reference,'reference producer')
        require(reference['binding_digest']==sha(encoded({k:v for k,v in reference.items() if k!='binding_digest'})),'canonical binding digest')
        if fresh:
            equal(dict(native_checkpoint=producer,producer_weight_identity=self.plan['checkpoint_identities']['O58'],**self.plan['canonical_provenance']),reference,'fresh canonical provenance')
            path=self.resolve(reference['source_record_path']);require(file_id(path)==reference['source_record_identity'],'canonical raw witness identity')
            self.native(path,request,reference,producer,'canonical_'+request['request_id'],SESSION+'_canonicals')
    def publication(self,value,checkpoint):
        status=value['native_status'];pub=status['publication']
        require(type(value['activation_s']) in (int,float) and math.isfinite(value['activation_s']) and value['activation_s']>=0,'publication activation cost')
        equal(dict(checkpoint_structure_valid=True,checkpoint_id=checkpoint),value,'publication checkpoint')
        equal(dict(status='passed',poisoned=None,capture_pending=False,cache_tokens=0,attention_backend='fa3',block_size=16),status,'publication native status')
        equal(dict(status='passed',checkpoint_path=checkpoint,cache_after_tokens=0,helper_matches_current_layers=True,helper_norm_matches_current_layers=True,fused_enabled=True),pub,'publication helpers')
    def weight(self,value,checkpoint):
        equal(dict(status='passed',all_equal=True,tensors=58,source_tensor_count=58,native_tensor_count=43,checkpoint_path=checkpoint,helper_matches_current_layers=True,helper_norm_matches_current_layers=True,fused_enabled=True),value,'native weight identity')

    def run(self):
        started=time.perf_counter();root=self.root;ev=root/'evaluation'
        # Four cheap reads are deliberately before bootstrap, bulk hashing or traces.
        self.plan=self.json(root/'PLAN.json');launch=self.json(root/'LAUNCH.json');clock=self.json(root/'CLOCK.json');scored=self.json(root/'phases/scored_A/STATUS.json')
        require(launch['status']=='completed' and scored['status']=='completed' and clock['active']=={},'Run is incomplete/active; do not audit during scored timing')
        scored_process=self.json(root/'processes/scored_A.json')
        require(scored_process['exit_code']==0 and scored_process['finished_epoch']>=scored_process['started_epoch'],'Scored worker has no successful exit receipt')
        plan=self.plan
        require(plan['binding']==sha(encoded({k:v for k,v in plan.items() if k!='binding'})),'plan binding')
        equal(dict(session=SESSION,automatic_retry=False,optimizer_steps=0,new_trained_checkpoints=0,
            serving=dict(block_size=16,max_new_tokens=2048,context_limit=4096,concurrency=1,greedy=True,
                precision='BF16',attention_backend='fa3',capture=False,
                cache_policy='idle_full_flush_before_every_request_and_publication'),
            limits=dict(elapsed_s=28800,occupied_card_s=43200,peak_bytes=30_000_000_000,
                shared_reserve_bytes=400_000_000_000)),plan,'fixed protocol plan')
        equal(dict(execution_commit=plan['execution_commit'],completed=['qualification','canonicals','scored'],automatic_retry=False),launch,'completed launch identity')
        preflight=self.json(root/'CPU_PREFLIGHT.json')
        equal(dict(status='passed',model_work=0,execution_commit=plan['execution_commit']),preflight,'preflight receipt')
        equal(preflight,launch['preflight'],'launch preflight copy')
        for path,pin in plan['input_identities'].items():require(file_id(self.resolve(path))==pin,'sealed input file '+path)
        panel=self.json(self.resolve(plan['panel']))['requests'];warm=self.json(self.resolve(plan['warmups']))['requests']
        require(len(panel)==1024 and len(warm)==2 and [r['panel_ordinal'] for r in panel]==list(range(1024)),'sealed input cardinalities/order')
        require(len({r['source_id'] for r in panel+warm})==1026 and len({r['request_id'] for r in panel+warm})==1026,'sealed input uniqueness')
        sealed_order=self.json(self.resolve(plan['panel']).parent/'ORDER.json')
        lexical_ids=sorted(r['request_id'] for r in panel)
        equal(dict(panel_order=[r['request_id'] for r in panel],source_id_order=[r['source_id'] for r in panel],
            lexicographic_request_id_order=lexical_ids,bootstrap_seed=20261119,
            bootstrap_order_sha256=sha(encoded(lexical_ids))),sealed_order,'prespecified bootstrap order')
        equal(panel,self.json(ev/'REQUESTS.json'),'evaluation panel exact copy')
        canonical=self.json(root/'CANONICALS.json');equal(canonical,self.json(ev/'CANONICAL.json'),'evaluation canonical copy')
        warm_canonical=self.json(self.resolve(plan['warmup_canonicals']));equal(dict(requests=warm,canonicals=warm_canonical),self.json(ev/'WARMUPS.json'),'warmup copies')
        require(set(canonical)=={r['request_id'] for r in panel},'exact fresh canonical coverage')
        for request in panel:self.ref(request,canonical[request['request_id']],plan['checkpoints']['O58'],True)
        for request in warm:self.ref(request,warm_canonical[request['request_id']],plan['reference_checkpoints']['warmup'],False)
        seal=self.json(root/'CANONICAL_SEAL.json');equal(dict(status='passed',count=1024,digest=sha(encoded(canonical)),producer='O58',producer_identity=plan['checkpoint_identities']['O58'],before_scored_generation=True,target_only=False),seal,'canonical population seal')
        orders=sorted(itertools.permutations(ARMS));schedule=[]
        for repeat in range(3):
            for group in range(16):
                for position,arm in enumerate(orders[(16*repeat+group)%6]):schedule.append(dict(activation=len(schedule),repeat=repeat,block_ordinal=group,position=position,method=arm,request_ordinals=list(range(group*64,(group+1)*64))))
        sealed_schedule=self.json(self.resolve(plan['schedule']));equal(schedule,sealed_schedule['activations'] if isinstance(sealed_schedule,dict) else sealed_schedule,'sealed schedule')
        protocol=self.json(ev/'PROTOCOL.json');equal(dict(schedule=schedule,checkpoints=plan['checkpoints'],checkpoint_identities=plan['checkpoint_identities'],canonical_provenance=plan['canonical_provenance'],physical_gpu_uuid=plan['devices']['A'],execution_commit=plan['execution_commit'],methods=list(ARMS),dataset='mathqa',event=58,scope=SCOPE,session=SESSION,capture=False,planned_scored=9216,planned_warmups=288,fresh_canonical_gate=True,canonical_producer='O58'),protocol,'evaluation protocol')
        activation_rows=self.lines(ev/'ACTIVATIONS.jsonl');require(len(activation_rows)==144,'144 activation records')
        for slot,actual in zip(schedule,activation_rows):equal(slot,actual,'activation schedule');require(not actual['errors'],'activation failure');self.publication(actual,plan['checkpoints'][slot['method']])
        weight_validation_s=0.
        for arm in ARMS:
            value=self.json(ev/f'WEIGHTS_{arm}.json');require(not value['errors'],'weight errors');self.weight(value['native_reply'],plan['checkpoints'][arm])
            weight_validation_s+=value['validation_s']
        boundaries=self.lines(ev/'BOUNDARY_CHECKS.jsonl');require([r['activation'] for r in boundaries]==[*range(144),'after_final'],'quiet-boundary coverage')
        for row in boundaries:
            require(not row['errors'],'boundary errors');equal(dict(status='passed',physical_gpu_uuid=plan['devices']['A'],both_cards_checked=True,other_gpu_idle=True,host_quiet=True,storage_admitted=True,time_admitted=True),row['check'],'quiet boundary')
        saved=self.lines(ev/'MEASUREMENTS.jsonl');require(len(saved)==9504,'9504 saved native rows')
        rebuilt=[];trace_count=0
        with (ev/'TRACE_COUNTS.csv').open(newline='') as trace_file:
            trace_csv=iter(csv.DictReader(trace_file))
            for slot in schedule:
                for phase,request in [('warmup',r) for r in warm]+[('scored',panel[i]) for i in slot['request_ordinals']]:
                    label=f"warmup_a{slot['activation']:02d}_{request['panel_ordinal']}_{slot['method']}" if phase=='warmup' else f"{request['panel_ordinal']:04d}_r{slot['repeat']}_{slot['method']}"
                    record,metrics=self.native(ev/(label+'.json'),request,(warm_canonical if phase=='warmup' else canonical)[request['request_id']],plan['checkpoints'][slot['method']],label,SESSION)
                    row=dict(metrics,**{k:v for k,v in slot.items() if k!='request_ordinals'},phase=phase,label=label,request_id=request['request_id'],source_id=request['source_id'],panel_ordinal=request['panel_ordinal'],physical_gpu_uuid=plan['devices']['A'],checkpoint_id=plan['checkpoints'][slot['method']],dataset='mathqa',event=58,scope=SCOPE)
                    actual=saved[len(rebuilt)];equal(row,actual,'raw scalar reconstruction');require(actual['valid_measurement'] is True and actual['validation_status']=='valid' and not actual['errors'],'invalid saved row');rebuilt.append(row)
                    require(self.resolve(actual['record_path'])==ev/(label+'.json'),'saved raw path identity')
                    equal(dict(status='valid',errors=[]),self.json(ev/(label+'_VALIDATION.json')),'per-generation validity receipt')
                    for i,b in enumerate(record['metadata']['trace']):
                        expected=dict(label=label,phase=phase,method=slot['method'],repeat=slot['repeat'],request_id=request['request_id'],verification_ordinal=i,**{k:b[k] for k in ('output_start','native_accepted_draft_tokens','accepted_draft_tokens','committed_progress','stop_clipped')})
                        actual_trace=next(trace_csv,None);require(actual_trace is not None,'truncated TRACE_COUNTS.csv');equal({k:str(v) for k,v in expected.items()},actual_trace,'trace-count CSV');trace_count+=1
            require(next(trace_csv,None) is None,'extra trace-count CSV rows')
        repeat_groups=defaultdict(list)
        for row in rebuilt:repeat_groups[row['phase'],row['method'],row['request_id']].append(row)
        for values in repeat_groups.values():
            first=values[0]
            for row in values[1:]:equal({k:first[k] for k in COUNTS+('native_acceptance_histogram','finish_reason','cap_hit')},row,'all scored/warmup deterministic trajectories')
        scores=[r for r in rebuilt if r['phase']=='scored'];warms=[r for r in rebuilt if r['phase']=='warmup']
        for name,rows in [('ALL_REPEATS.csv',scores),('WARMUPS.csv',warms)]:
            table=self.csv(ev/name);require(len(table)==len(rows),'repeat CSV coverage '+name)
            fields=set(('dataset','source_id','request_id','panel_ordinal','method','repeat','block_ordinal','position','activation','physical_gpu_uuid','checkpoint_id','finish_reason','cap_hit')+FIELDS)
            for expected,actual in zip(rows,table):
                require(set(actual)==fields,'repeat CSV field coverage '+name)
                equal({k:expected[k] for k in actual},actual,'repeat CSV '+name)
        paired=reduce_rows(scores,panel);result=statistics(paired)
        summary=self.json(ev/'SUMMARY.json');equal(dict(status='completed',request_count=1024,primary_comparison='KPLUS_C58_vs_AREPO58',execution_commit=plan['execution_commit'],automatic_followup_authorized=False,**result),summary,'independent statistics')
        equal(paired,summary['paired_requests'],'paired summary rows')
        paired_csv=self.csv(ev/'PER_REQUEST.csv');require(len(paired_csv)==3072,'paired CSV coverage')
        keyed={(r['request_id'],r['method']):r for r in paired_csv};require(len(keyed)==3072,'paired CSV unique keys')
        for request in paired:
            for arm in ARMS:equal({**{k:request[k] for k in ('source_id','request_id','panel_ordinal')},'method':arm,**{k:request[arm][k] for k in FIELDS}},keyed[request['request_id'],arm],'independent paired CSV')
        repeat_totals={str(i):{m:{k:sum(r[k] for r in scores if r['repeat']==i and r['method']==m) for k in FIELDS} for m in ARMS} for i in range(3)}
        equal(repeat_totals,summary['repeat_totals'],'repeat totals');rt=self.csv(ev/'REPEAT_TOTALS.csv');require(len(rt)==9,'repeat total CSV coverage')
        for row in rt:equal(repeat_totals[str(row['repeat'])][row['method']],row,'repeat total CSV')
        for i in range(3):
            for group in range(16):equal({m:{k:sum(r[k] for r in scores if r['repeat']==i and r['block_ordinal']==group and r['method']==m) for k in FIELDS} for m in ARMS},summary['block_views'][f'repeat_{i}_block_{group}'],'block totals')
        for a,b in CONTRASTS:equal({k:result['method_totals'][a][k]-result['method_totals'][b][k] for k in FIELDS},summary['absolute_differences'][a+'_minus_'+b],'absolute contrasts')
        status=self.json(ev/'STATUS.json');equal(dict(status='completed',planned_generations=9504,attempted_generations=9504,returned_generations=9504,completed_valid_generations=9504,invalid_generations=0,scored_generations=9216,warmup_generations=288,publications=144,logical_activations=144,unknown_generation_attempts=0,unattempted_generations=0,optimizer_steps=0,feedback_collections=0,teacher_preparations=0,automatic_retry=False),status,'evaluation completion')
        equal(dict(activation_s=math.fsum(r['activation_s'] for r in activation_rows),weight_validation_s=weight_validation_s),status,'independent activation and validation cost')
        validity=self.json(ev/'VALIDITY.json')
        equal(dict(status='passed',execution_commit=plan['execution_commit'],returned=9504,invalid=0,unknown=0,unattempted=0,private_raw_records_retained=True),validity,'validity coverage')
        equal([{k:r[k] for k in ('label','phase','method','request_id','repeat','validation_status','errors')} for r in saved],validity['rows'],'per-row final validity')
        costs=self.json(ev/'EVALUATION_COSTS.json');equal(status,costs,'evaluation costs/status')
        equal({phase:{k:math.fsum(r[k] for r in rows) for k in TIMES} for phase,rows in [('scored',scores),('warmup',warms)]},costs['measured_time_by_phase'],'cost phase totals')
        equal({m:result['method_totals'][m]['request_latency_s'] for m in ARMS},costs['serving_summed_request_means'],'cost request means')
        work=self.json(root/'WORK.json');equal(dict(limits=LIMITS,attempted=LIMITS,completed=LIMITS,pending={},optimizer_steps=0,trained_checkpoints=0,next_operation=10552),work,'complete generation ledger')
        require(work['pending']=={} and all(set(work[k])==set(LIMITS) for k in ('limits','attempted','completed')),'no pending or unrecognized generation category')
        equal(dict(status='passed',generations=24),self.json(root/'QUALIFICATION.json'),'qualification receipt')
        qual=0;qual_previous={}
        for device in ('A','B'):
            base=root/'phases'/f'qualification_{device}'
            require(self.json(base/'STATUS.json')['status']=='completed','qualification completion')
            for repeat in range(2):
                for arm in ARMS:
                    pub=self.json(base/f'{repeat}_{arm}_PUBLICATION.json');require(not pub['errors'],'qualification publication');self.publication(pub['receipt'],plan['checkpoints'][arm])
                    weights=self.json(base/f'{repeat}_{arm}_WEIGHTS.json');require(not weights['errors'],'qualification weight');self.weight(weights['receipt'],plan['checkpoints'][arm])
                    for request in warm:
                        label=f"{request['panel_ordinal']:04d}_r{repeat}_{arm}"
                        _,metrics=self.native(base/(label+'.json'),request,warm_canonical[request['request_id']],plan['checkpoints'][arm],label,SESSION)
                        key=(device,arm,request['request_id'])
                        if key in qual_previous:equal({k:qual_previous[key][k] for k in COUNTS+('native_acceptance_histogram',)},metrics,'qualification deterministic repeat')
                        qual_previous[key]=metrics;qual+=1
            shard=self.json(root/'phases'/f'canonical_{device}'/'CANONICALS.json');expected_ids={r['request_id'] for r in panel if r['panel_ordinal']%2==(device=='B')}
            require(set(shard)==expected_ids,'canonical shard exact coverage');equal({k:canonical[k] for k in expected_ids},shard,'shard-to-seal equality')
            base=root/'phases'/f'canonical_{device}'
            publication=self.json(base/'O58_PUBLICATION.json');require(not publication['errors'],'canonical publication');self.publication(publication['receipt'],plan['checkpoints']['O58'])
            weights=self.json(base/'O58_WEIGHTS.json');require(not weights['errors'],'canonical producer weights');self.weight(weights['receipt'],plan['checkpoints']['O58'])
        require(qual==24 and self.raw_records==10552,'all generation raw evidence accounted')
        phase_costs={}
        for name in ('qualification_A','qualification_B','canonical_A','canonical_B','scored_A'):
            phase=self.json(root/'phases'/name/'STATUS.json');require(phase['status']=='completed' and phase['execution_commit']==plan['execution_commit'],'phase identity')
            require(0<=phase['loading_s']<=phase['wall_s'],'phase loading/wall cost')
            phase_costs[name]={k:phase[k] for k in ('loading_s','wall_s')}
            process=self.json(root/'processes'/(name+'.json'))
            require(process['exit_code']==0 and process['finished_epoch']>=process['started_epoch'],'phase process did not exit successfully')
            physical=self.json(root/'phases'/name/'PHYSICAL_WORK.json');require(physical['extra_model_calls']==0,'unbudgeted native model calls')
        require(len(clock['finished'])==5 and {x['task'] for x in clock['finished']}=={'qualification_A','qualification_B','canonical_A','canonical_B','scored_A'},'card lease coverage')
        for lease in clock['finished']:require(lease['physical_gpu_uuid']==plan['devices'][lease['task'][-1]] and lease['occupied_card_s']>=0,'card lease physical identity')
        equal(float(math.fsum(x['occupied_card_s'] for x in clock['finished'])),clock['completed_card_s'],'occupied-card ledger sum')
        require(0<=clock['completed_card_s']<=43200,'occupied-card cap')
        final=launch['postflight'];require(final['clock']['elapsed_s']<=28800 and final['clock']['active']=={},'charged execution elapsed cap')
        require(final['storage']['charged_bytes']<=30_000_000_000 and final['storage']['free_bytes']>=final['storage']['required_free_bytes']>=400_000_000_000,'final allocation/free reserve')
        post=self.json(root/'POSTFLIGHT_EXPORT_IDENTITIES.json');require(post['status']=='passed' and not post['errors'],'frozen export postflight')
        for arm in ARMS:equal(dict(expected_identity=plan['checkpoint_identities'][arm],actual_identity=plan['checkpoint_identities'][arm],unchanged=True),post['exports'][arm],'retained export identity')
        primary=result['comparisons']['KPLUS_C58_vs_AREPO58'];time_label=primary['evidence']['time_reduction'];draft_label=primary['evidence']['call_reduction']
        interpretation='Supported lower complete-request C latency and drafting benefit on this fixed-endpoint fresh MathQA panel.' if time_label==draft_label=='SUPPORTED_GAIN' else 'Drafting benefit is supported while serving-time direction remains unresolved.' if draft_label=='SUPPORTED_GAIN' and time_label=='DIRECTION_UNRESOLVED' else 'Both primary time and drafting directions are unresolved; no demonstrated direct advantage or equivalence.' if time_label==draft_label=='DIRECTION_UNRESOLVED' else 'Report metric-specific directions, including any supported regression; do not select the favorable metric.'
        return dict(status='passed_independent_completed_evidence_audit',execution_commit=plan['execution_commit'],requests=1024,scored_rows=9216,warmup_rows=288,canonical_rows=1024,qualification_rows=24,raw_generations=self.raw_records,raw_verification_blocks=self.trace_blocks,trace_csv_blocks=trace_count,statistics=result,primary_interpretation=interpretation,primary_time_reduction_percent={k:None if v is None else [100*x for x in v] if isinstance(v,list) else 100*v for k,v in primary['time_reduction'].items() if k!='undefined_draws'},cost=dict(phase_costs=phase_costs,scored_activation_s=status['activation_s'],scored_weight_validation_s=status['weight_validation_s'],occupied_card_s=clock['completed_card_s'],charged_execution_elapsed_s=final['clock']['elapsed_s'],launch_wall_s=launch['wall_s'],analysis_s=launch['analysis_s'],evaluation_wall_s=status['evaluation_wall_s'],postflight_export_hash_s=post['wall_s'],nested_timers_not_additive=True),limits=['Conditional marginal 95% intervals for these fitted endpoints and this panel.','Primary C/AREPO complete-request latency; other intervals descriptive and unadjusted.','Three repeats are measurements within requests, not independent requests or seeds.','Native and stop-clipped acceptance are separately reconstructed; calls and native acceptance are coupled.','No independent training seed replication, causal cap/allocation isolation, equivalence/nondegradation or training-inclusive speedup claim.','Historical warmup source-record bytes and live model files are not reread; authenticated sealed witnesses and postflight identities are checked.','This audit cannot independently certify unknown project history or pretraining exposure.'],authorship='Reviewer previously authored panel audit and repaired runtime; statistical reducer and raw-trace reduction implemented separately without production summarize/compare/audit imports. No independent GPU replication.',model_work=0,network_operations=0,evidence_files_read=len(self.reads),audit_wall_s=time.perf_counter()-started)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('execution_root');parser.add_argument('--output',required=True);args=parser.parse_args()
    audit=Audit(args.execution_root)
    try:report=audit.run();code=0
    except (AuditFailure,OSError,KeyError,TypeError,ValueError,IndexError,StopIteration) as error:
        report=dict(status='failed_or_incomplete',error=repr(error),raw_generations_checked=audit.raw_records,no_full_study_conclusion=True,model_work=0,network_operations=0);code=1
    report['auditor_identity']=file_id(__file__)
    output=Path(args.output);require(not output.exists(),'Never overwrite an earlier audit result')
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ('status','error','raw_generations','raw_generations_checked','audit_wall_s') if k in report}));return code

if __name__=='__main__':sys.exit(main())
