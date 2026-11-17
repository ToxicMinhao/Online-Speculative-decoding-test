"""Independent CPU audit of PR50's complete native endpoint evidence.

Reads JSON/token traces only. No production metric or validation helper is
imported. The fixed audit comparison tolerance (not a scientific gate change)
is 1e-10 absolute/relative for independently reduced floating-point reports.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

ARMS=("N","C","S")
COUNTS=("verification_calls","native_accepted_draft_tokens","accepted_draft_tokens","committed_progress","output_tokens")
TIMES=("request_latency_s","native_generate_s","cache_maintenance_s")
CONTRASTS=(("C","N"),("C","S"),("S","N"))
TOLERANCE=1e-10


def identity(payload):
    return dict(bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest())


class Audit:
    def __init__(self,delivery):
        self.root=Path(delivery).resolve();self.count=0;self.errors=[];self.files={}
        self.manifest_name='RETRIEVAL_MANIFEST.json' if (self.root/'RETRIEVAL_MANIFEST.json').is_file() else 'DELIVERY_MANIFEST.json'
        self.manifest=json.loads((self.root/self.manifest_name).read_text())
        self.origin=None

    def check(self,condition,label):
        self.count+=1
        if not condition:self.errors.append(label)
        return bool(condition)

    def read(self,relative,lines=False):
        relative=str(relative);path=self.root/relative
        raw=path.read_bytes();observed=identity(raw)
        self.check(observed==self.manifest.get(relative),'manifest:'+relative)
        self.files[relative]=observed
        return [json.loads(line) for line in raw.splitlines() if line] if lines else json.loads(raw)

    def path(self,recorded):
        return Path(recorded).relative_to(self.origin)

    def equal(self,actual,expected,label):
        if isinstance(expected,float):
            good=isinstance(actual,(float,int)) and math.isfinite(actual) and math.isclose(actual,expected,rel_tol=TOLERANCE,abs_tol=TOLERANCE)
        else:good=actual==expected
        self.check(good,label)


def label(slot):
    if slot['phase']=='warmup':return f"warmup_a{slot['activation']:02d}_{slot['panel_ordinal']}_{slot['method']}"
    return f"{slot['panel_ordinal']:04d}_r{slot['repeat']}_{slot['method']}"


def schedule(permutations):
    result=[]
    for repeat in range(3):
        for block in range(4):
            base=ARMS if block%2==0 else ("N","S","C")
            shift=(repeat+block//2)%3;order=base[shift:]+base[:shift]
            for position,arm in enumerate(order):
                result.append(dict(activation=len(result),repeat=repeat,block_ordinal=block,position=position,
                    method=arm,request_ordinals=permutations[repeat][block]))
    return result


def raw_counts(audit,record,request,canonical,slot,protocol,persisted):
    name=label(slot);check=lambda condition,field:audit.check(condition,name+':'+field)
    prompt,response=record['prompt_ids'],record['response_ids'];meta=record['metadata'];capture=meta['capture'];trace=meta['trace']
    check(all(type(t) is int and t>=0 for t in [*prompt,*response]) and bool(response),'integer_tokens')
    check(prompt==request['prompt_ids']==canonical['prompt_ids'],'prompt')
    check(response==canonical['response_ids'] and record['finish_reason']==canonical['finish_reason'],'canonical_output')
    check(record['record_id']==name and record['request_index']==slot['panel_ordinal'],'request_identity')
    check(record['session_id']==protocol['session'] and record['purpose']=='evaluation','session_purpose')
    check(record['checkpoint_id']==protocol['checkpoints'][slot['method']],'checkpoint')
    stop=request['stop_token_ids']
    check(record['stop_token_ids']==stop and record['max_new_tokens']==2048,'stop_cap')
    check(len(prompt)+2048<=4096 and 0<len(response)<=2048,'context_output_budget')
    check(not any(t in stop for t in response[:-1]),'no_prior_eos')
    check((record['finish_reason']=='eos' and response[-1] in stop) or
        (record['finish_reason']=='length' and len(response)==2048 and response[-1] not in stop),'termination')
    check(meta['block_size']==16 and meta['context_limit']==4096,'native_geometry')
    check(meta['runtime']=='pinned_SGLang_FA3' and meta['cache_policy']=='idle_full_flush_before_every_request_and_publication','native_configuration')
    check(meta['feedback_capture_in_timed_request'] is False and capture['collect_features'] is False,'capture_disabled')
    check(capture['native_capture_validated'] is True and capture['feature_bytes']==0,'capture_validated_empty')
    check(capture==persisted and capture['blocks']==trace and capture['prompt_ids']==prompt and capture['response_ids']==response,'persisted_native_capture')
    position=0;native=0;matched=0;histogram=Counter()
    for b,block in enumerate(trace):
        prefix=f'block{b}:'
        proposed,target=block['draft_tokens'],block['target_argmax'];width=block['verify_size'];committed=block['committed_progress']
        valid=(type(width) is int and type(committed) is int and 1<=committed<=width<=16
            and all(type(block[k]) is int for k in ('output_start','anchor_token','raw_accepted_draft_tokens_before_stop','accepted_draft_tokens','native_accepted_draft_tokens'))
            and width==len(proposed)+1==len(target) and all(type(t) is int and t>=0 for t in [*proposed,*target]))
        check(valid,prefix+'geometry')
        if not valid:continue
        match=0
        while match<len(proposed) and proposed[match]==target[match]:match+=1
        commits=(proposed[:match]+[target[match]])[:committed]
        check(committed<=match+1,prefix+'reject_boundary')
        check(block['output_start']==position and block['anchor_token']==response[position],prefix+'seed_position')
        check(block['committed_tokens']==commits==target[:committed]==response[position+1:position+1+committed],prefix+'commits')
        check(block['positions']==list(range(len(prompt)+position,len(prompt)+position+width)),prefix+'absolute_positions')
        check(block['raw_accepted_draft_tokens_before_stop']==match and block['accepted_draft_tokens']==min(match,committed),prefix+'matching_acceptance')
        check(block['native_accepted_draft_tokens']==committed-1,prefix+'native_acceptance')
        check(block['stop_clipped']==(committed<match+1),prefix+'clipping')
        if 'finished' in block:check(block['finished']==(position+committed==len(response)-1),prefix+'finished')
        native+=committed-1;matched+=min(match,committed);position+=committed;histogram[str(committed-1)]+=1
    calls=len(trace)
    check(position==len(response)-1 and native==position-calls,'whole_progress')
    reconstructed=dict(verification_calls=calls,native_accepted_draft_tokens=native,
        accepted_draft_tokens=matched,committed_progress=position,output_tokens=len(response))
    for key in ('verification_calls','accepted_draft_tokens','committed_progress'):check(type(meta[key]) is int and meta[key]==reconstructed[key],'stored_'+key)
    values=dict(request_latency_s=meta['request_latency_s'],native_generate_s=meta['native_generate_s'],cache_maintenance_s=capture['cache_maintenance_s'])
    check(all(type(v) in (int,float) and math.isfinite(v) and (v>=0 if k=='cache_maintenance_s' else v>0) for k,v in values.items()),'finite_timing')
    return dict(**slot,**reconstructed,**values,source_id=request['source_id'],request_id=request['request_id'],
        finish_reason=record['finish_reason'],native_acceptance_histogram=dict(histogram),checkpoint_id=record['checkpoint_id'])


def statistics(rows):
    scored=[r for r in rows if r['phase']=='scored'];groups=defaultdict(list)
    for row in scored:groups[row['source_id'],row['method']].append(row)
    sources=sorted({r['source_id'] for r in scored});paired=[]
    for source in sources:
        initial=groups[source,ARMS[0]][0]
        item=dict(source_id=source,request_id=initial['request_id'],panel_ordinal=initial['panel_ordinal'])
        for arm in ARMS:
            values=sorted(groups[source,arm],key=lambda r:r['repeat'])
            if [r['repeat'] for r in values]!=[0,1,2]:raise ValueError('Missing or duplicated repeat')
            first=values[0]
            if any(any(r[k]!=first[k] for k in (*COUNTS,'finish_reason','checkpoint_id','native_acceptance_histogram')) for r in values):
                raise ValueError('Repeat count/output/checkpoint mismatch')
            item[arm]={k:first[k] for k in COUNTS}
            item[arm].update({k:sum(r[k] for r in values)/3 for k in TIMES})
            item[arm].update(checkpoint_id=first['checkpoint_id'],finish_reason=first['finish_reason'],
                cap_hit=first['finish_reason']=='length',native_acceptance_histogram=first['native_acceptance_histogram'],
                repeat_request_latency_s=[r['request_latency_s'] for r in values])
        paired.append(item)
    if len(paired)!=128:raise ValueError('Incomplete128pairedrequests')
    totals={arm:{k:sum(r[arm][k] for r in paired) for k in (*COUNTS,*TIMES)} for arm in ARMS}
    for arm,total in totals.items():
        histogram=Counter()
        for row in paired:histogram.update(row[arm]['native_acceptance_histogram'])
        positive=[r[arm] for r in paired if r[arm]['verification_calls']>0]
        total.update(calls_per_1000_output_tokens=1000*total['verification_calls']/total['output_tokens'],
            equal_request_native_acceptance_mean=sum(r['native_accepted_draft_tokens']/r['verification_calls'] for r in positive)/len(positive) if positive else None,
            equal_request_denominator=len(positive),zero_call_requests=128-len(positive),
            cap_hit_requests=sum(r[arm]['cap_hit'] for r in paired),
            native_acceptance_histogram=dict(sorted(histogram.items(),key=lambda item:int(item[0]))),native_accepted_tokens_per_call=total['native_accepted_draft_tokens']/total['verification_calls'] if total['verification_calls'] else None,
            matched_accepted_tokens_per_call=total['accepted_draft_tokens']/total['verification_calls'] if total['verification_calls'] else None,
            committed_progress_per_call=total['committed_progress']/total['verification_calls'] if total['verification_calls'] else None)
    draws=np.random.Generator(np.random.PCG64(1708)).integers(0,128,(10000,128))
    comparisons={}
    def array(arm,key):return np.asarray([r[arm][key] for r in paired],dtype=np.float64)
    for c,r in CONTRASTS:
        cv,rv=array(c,'verification_calls'),array(r,'verification_calls')
        ca,ra=array(c,'native_accepted_draft_tokens'),array(r,'native_accepted_draft_tokens')
        ct,rt=array(c,'request_latency_s'),array(r,'request_latency_s')
        with np.errstate(divide='ignore',invalid='ignore'):
            c_accept=ca[draws].sum(1)/cv[draws].sum(1);r_accept=ra[draws].sum(1)/rv[draws].sum(1)
            distributions=dict(time_reduction=1-ct[draws].sum(1)/rt[draws].sum(1),
                call_reduction=1-cv[draws].sum(1)/rv[draws].sum(1),
                acceptance_improvement=c_accept/r_accept-1,acceptance_difference=c_accept-r_accept)
        c_point=totals[c]['native_accepted_tokens_per_call'];r_point=totals[r]['native_accepted_tokens_per_call']
        points=dict(time_reduction=1-ct.sum()/rt.sum(),call_reduction=1-cv.sum()/rv.sum() if rv.sum() else None,
            acceptance_improvement=c_point/r_point-1 if c_point is not None and r_point else None,
            acceptance_difference=c_point-r_point if c_point is not None and r_point is not None else None)
        saved=rv-cv
        effects={key:dict(point=points[key],ci95=np.quantile(value,[.025,.975],method='linear').tolist() if np.isfinite(value).all() else None,
            undefined_draws=int((~np.isfinite(value)).sum())) for key,value in distributions.items()}
        comparisons[c+'_vs_'+r]=dict(**effects,speedup=float(rt.sum()/ct.sum()),
            repeat_time_reductions=[1-sum(x[c]['repeat_request_latency_s'][i] for x in paired)/sum(x[r]['repeat_request_latency_s'][i] for x in paired) for i in range(3)],
            requests_fewer_calls=int((saved>0).sum()),requests_more_calls=int((saved<0).sum()),request_call_ties=int((saved==0).sum()),
            gross_calls_saved=int(saved.clip(min=0).sum()),gross_calls_added=int((-saved).clip(min=0).sum()))
    repeat_totals={str(i):{arm:{k:sum(r[k] for r in scored if r['repeat']==i and r['method']==arm) for k in (*COUNTS,*TIMES)} for arm in ARMS} for i in range(3)}
    return dict(method_totals=totals,comparisons=comparisons,repeat_totals=repeat_totals,paired_requests=paired,
        draw_matrix_sha256=hashlib.sha256(draws.astype('<i8').tobytes()).hexdigest())


def compare_tree(audit,actual,expected,label):
    if isinstance(expected,dict):
        for key,value in expected.items():
            audit.check(key in actual,label+':field:'+key)
            if key in actual:compare_tree(audit,actual[key],value,label+'.'+key)
    elif isinstance(expected,list):
        audit.check(isinstance(actual,list) and len(actual)==len(expected),label+':length')
        if isinstance(actual,list):
            for i,(a,e) in enumerate(zip(actual,expected)):compare_tree(audit,a,e,label+f'[{i}]')
    else:audit.equal(actual,expected,label)


def run(delivery,output):
    began=time.perf_counter();audit=None;result=dict(status='INVALID_OR_INCOMPLETE',new_model_work=0,efficacy_claim=False)
    try:
        audit=Audit(delivery);plan=audit.read('sealed/PLAN.json');audit.origin=Path(plan['root'])
        final=Path('tasks/final/evaluation')
        protocol=audit.read(final/'PROTOCOL.json');requests=audit.read(final/'REQUESTS.json')
        canonicals=audit.read(final/'CANONICAL.json');warm=audit.read(final/'WARMUPS.json')
        panel=audit.read(audit.path(plan['data_panel']));fixed_references=audit.read(audit.path(plan['references']))
        fixed_warm=audit.read(audit.path(plan['warmups']));warm_refs=audit.read(audit.path(plan['warmup_references']))
        audit.check(requests==panel['requests'] and canonicals==fixed_references,'sealed_panel_and_references')
        audit.check(warm==dict(requests=fixed_warm['requests'],canonicals=warm_refs),'sealed_warmups')
        audit.check(len(requests)==128 and [r['panel_ordinal'] for r in requests]==list(range(128)),'complete128panel')
        audit.check(len({r['source_id'] for r in requests})==128 and len({r['request_id'] for r in requests})==128,'unique_panel')
        audit.check(protocol['methods']==list(ARMS) and protocol['session']=='bounded_prefix_allocation_event8','declared_three_methods_session')
        audit.check(protocol['capture'] is False and protocol['event']==8 and protocol['planned_scored']==1152 and protocol['planned_warmups']==72,'protocol_budget')
        audit.check(protocol['execution_commit']==plan['execution_commit'],'execution_commit')
        audit.check(protocol['physical_gpu_uuid']==plan['devices']['A'],'fixed_physical_timing_card')
        for arm in ('N','S'):audit.check(protocol['checkpoints'][arm]==plan['controls'][arm]['export'],'control_'+arm)
        c8=audit.read('tasks/C8/RESULT.json')
        audit.check(c8['status']=='completed' and c8['endpoint']['global_step']==1024 and c8['endpoint']['presentations']==2048,'C8_complete')
        audit.check(protocol['checkpoints']['C']==c8['endpoint']['export'],'candidate_C8')
        p=panel['within_block_permutations']
        audit.check(p==protocol['within_block_permutations'],'sealed_permutations')
        for rep in p:
            for b,values in enumerate(rep):audit.check(sorted(values)==list(range(32*b,32*(b+1))),'within_block_permutation')
        planned=schedule(p);audit.check(protocol['schedule']==planned,'balanced_schedule')
        status=audit.read(final/'STATUS.json')
        expected=dict(status='completed',planned_generations=1224,attempted_generations=1224,returned_generations=1224,
            completed_valid_generations=1224,scored_generations=1152,warmup_generations=72,publications=36,logical_activations=36,
            invalid_generations=0,unknown_generation_attempts=0,unattempted_generations=0,optimizer_steps=0,feedback_collections=0,teacher_preparations=0)
        compare_tree(audit,status,expected,'final_status')
        boundaries=audit.read(final/'BOUNDARY_CHECKS.jsonl',lines=True)
        audit.check([r['activation'] for r in boundaries]==[*range(36),'after_final'],'all_quiet_boundaries')
        for i,row in enumerate(boundaries):
            check=row['check'];audit.check(not row['errors'] and check['status']=='passed' and check['physical_gpu_uuid']==protocol['physical_gpu_uuid']
                and all(check.get(k) is True for k in ('both_cards_checked','other_gpu_idle','host_quiet','storage_admitted','time_admitted')),f'quiet:{i}')
        measurements=audit.read(final/'MEASUREMENTS.jsonl',lines=True);audit.check(len(measurements)==1224,'measurement_count')
        rows=[];offset=0
        for activation in planned:
            jobs=[('warmup',x,warm['canonicals'][x['request_id']]) for x in warm['requests']]
            jobs += [('scored',requests[i],canonicals[requests[i]['request_id']]) for i in activation['request_ordinals']]
            for phase,request,canonical in jobs:
                slot={k:v for k,v in activation.items() if k!='request_ordinals'}
                slot.update(phase=phase,panel_ordinal=request['panel_ordinal'])
                raw=audit.read(final/(label(slot)+'.json'))
                persisted=audit.read(audit.path(raw['metadata']['native_trace_path']))
                row=raw_counts(audit,raw,request,canonical,slot,protocol,persisted)
                saved=measurements[offset];offset+=1
                compare_tree(audit,saved,row,'measurement:'+label(slot))
                audit.check(saved['valid_measurement'] is True and saved['validation_status']=='valid' and not saved['errors'],'valid:'+label(slot))
                audit.check(saved['physical_gpu_uuid']==protocol['physical_gpu_uuid'],'measurement_card:'+label(slot))
                rows.append(row)
        audit.check(sum(r['phase']=='scored' for r in rows)==1152 and sum(r['phase']=='warmup' for r in rows)==72,'actual_generation_counts')
        if audit.errors:raise ValueError('Native or identity evidence failed; no efficacy reduction')
        independent=statistics(rows);production=audit.read(final/'SUMMARY.json')
        audit.check(production['status']=='completed' and production['request_count']==128,'summary_completion')
        for key in ('method_totals','comparisons','repeat_totals','paired_requests'):compare_tree(audit,production[key],independent[key],'summary:'+key)
        audit.check(production['bootstrap']['draw_matrix_sha256']==independent['draw_matrix_sha256'],'shared_bootstrap_draw_identity')
        cn=independent['comparisons']['C_vs_N'];cs=independent['comparisons']['C_vs_S']
        advance=(cn['time_reduction']['point']>=.005 and cn['time_reduction']['ci95'] is not None and cn['time_reduction']['ci95'][0]>0
            and cn['acceptance_improvement']['ci95'] is not None and cn['acceptance_improvement']['ci95'][0]>0
            and all(x>0 for x in cn['repeat_time_reductions'])
            and all(cs[k]['point'] is not None and cs[k]['point']>0 for k in ('time_reduction','acceptance_improvement')))
        expected_decision='ELIGIBLE_FOR_SEPARATE_CONFIRMATION' if advance else 'CLOSE_FIXED_CANDIDATE_AT_THIS_SCREEN'
        audit.check(production['decision']['outcome']==expected_decision,'prespecified_decision')
        result.update(status='passed' if not audit.errors else 'AUDIT_DISAGREEMENT',independent=independent,
            raw_generations=len(rows),scored_generations=1152,warmup_generations=72,paired_requests=128,decision=expected_decision)
    except (OSError,KeyError,ValueError,TypeError,IndexError,ZeroDivisionError) as error:
        result.update(error=repr(error),error_path=getattr(error,'filename',None))
    result.update(wall_s=time.perf_counter()-began,audit_tolerance=TOLERANCE,
        checks=0 if audit is None else audit.count,errors=[] if audit is None else audit.errors,
        consumed_files={} if audit is None else audit.files,
        input_manifest=None if audit is None else audit.manifest_name,
        independence='No production metric, schedule or native-validation helper imported; raw token/count calculation and statistics are local.')
    Path(output).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--delivery',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();result=run(args.delivery,args.output)
    print(json.dumps({k:result[k] for k in ('status','checks','wall_s')}))
