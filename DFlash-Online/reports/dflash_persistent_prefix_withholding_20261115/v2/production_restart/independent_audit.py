#!/usr/bin/env python3
"""Independent CPU audit of the completed authorized PR49 production restart.

No production analysis imports, Torch, remote calls, or model work. Reconstructs
all fresh final native trajectories and the fixed paired bootstrap independently.
Historical canonical authentication is reused explicitly from sealed evidence;
new canonical sources and all fresh final native records are reopened.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
import numpy as np

METHODS=('N','E','S')
COUNT_KEYS=('verification_calls','native_accepted_draft_tokens','accepted_draft_tokens','committed_progress','output_tokens')
TIME_KEYS=('request_latency_s','native_generate_s','cache_maintenance_s')
SCOPE='DEVELOPMENT_FEASIBILITY_REUSED_DATA'
OLD_COMMIT='db6e1221d8d1b234d9d07e246025d1afd5dc30d5'
EXECUTION_COMMIT='af708012e729437068fbef93cb11c9996a9109a9'
REUSED=('E7','S7','N_collect','E_collect','S_collect','E8','N8')
NEW=('S8','references','final')
class Audit:
    def __init__(self, root, mappings):
        self.root = Path(root).absolute()
        self.mappings = [(Path(old), Path(new)) for old, new in mappings]
        self.errors, self.checks, self.error_count, self.inputs = [], 0, 0, {}

    def check(self, condition, message):
        self.checks += 1
        if not condition:
            self.error_count += 1
            if len(self.errors) < 500:
                self.errors.append(message)
        return condition

    def resolve(self, value):
        path = Path(value)
        for old, new in sorted(self.mappings, key=lambda row: len(str(row[0])), reverse=True):
            if path.is_relative_to(old):
                return new / path.relative_to(old)
        return path if path.is_absolute() else self.root / path

    def read(self, value, lines=False):
        path = self.resolve(value)
        data = path.read_bytes()
        self.inputs[str(path)] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        return [json.loads(row) for row in data.splitlines() if row] if lines else json.loads(data)

    def equal(self, actual, expected, label):
        """Counts/identities exact; floating reductions permit only FP64 noise."""
        if isinstance(expected, dict):
            for key, value in expected.items():
                if self.check(isinstance(actual,dict) and key in actual, label+'/'+key+': missing'):
                    self.equal(actual[key],value,label+'/'+key)
        elif isinstance(expected, (list,tuple)):
            if self.check(isinstance(actual,(list,tuple)) and len(actual)==len(expected),label+': length'):
                for index,value in enumerate(expected):
                    self.equal(actual[index],value,label+'/'+str(index))
        elif isinstance(expected,float):
            self.check(type(actual) in (int,float) and math.isfinite(actual) and
                math.isclose(actual,expected,rel_tol=1e-10,abs_tol=1e-12),label+': FP64 reduction differs')
        else:
            self.check(type(actual) is type(expected) and actual==expected,label+': exact mismatch')

def native_counts(audit, record, request, reference, label, checkpoint=None, capture=False):
    prompt, output, meta = record['prompt_ids'],record['response_ids'],record['metadata']
    stops=request['stop_token_ids']
    audit.check(prompt==request['prompt_ids']==reference['prompt_ids'],label+': prompt binding')
    audit.check(output==reference['response_ids'],label+': exact canonical tokens')
    audit.check(record['finish_reason']==reference['finish_reason'],label+': exact termination')
    audit.check(reference.get('verified') is True and bool(reference.get('binding_digest')) and
        reference.get('reference_kind') in ('target_only_AR','native_greedy_target_verifier'),label+': authenticated reference')
    audit.check(0<len(output)<=2048 and all(type(t) is int and t>=0 for t in output),label+': tokens')
    audit.check(not any(t in stops for t in output[:-1]),label+': premature EOS')
    audit.check((record['finish_reason']=='eos' and output[-1] in stops) or
        (record['finish_reason']=='length' and len(output)==2048 and output[-1] not in stops),label+': EOS/cap')
    audit.check(record['max_new_tokens']==2048 and record['stop_token_ids']==stops and
        meta['block_size']==16 and meta['context_limit']==4096,label+': serving settings')
    if checkpoint is not None:
        audit.check(record['checkpoint_id']==checkpoint,label+': checkpoint')
    audit.check(record['purpose']==('serving' if capture else 'evaluation') and
        meta['feedback_capture_in_timed_request'] is capture,label+': capture mode')
    audit.check(meta['cache_policy']=='idle_full_flush_before_every_request_and_publication',label+': cache policy')
    observed=meta['capture']
    audit.check(observed['prompt_ids']==prompt and observed['response_ids']==output and
        observed['target_argmax_ids']==output and observed['covered_token_ids']==(prompt+output)[:-1],label+': target coverage')
    audit.check(observed['blocks']==meta['trace'] and observed['collect_features'] is capture and
        observed['native_capture_validated'] is True,label+': raw native binding')
    # Authenticate the independently persisted native record, not only its copy.
    audit.check(audit.read(meta['native_trace_path'])==observed,label+': native trace file')
    offset=0
    native=matched=0
    histogram=Counter()
    for number,block in enumerate(meta['trace']):
        tag=f'{label}: block{number}'
        proposed, target = block['draft_tokens'],block['target_argmax']
        width=block['verify_size']; progress=block['committed_progress']
        audit.check(type(width) is int and 1<=width<=16 and len(proposed)+1==width==len(target),tag+': geometry')
        audit.check(type(progress) is int and 1<=progress<=width,tag+': progress range')
        first=next((j for j,(draft,truth) in enumerate(zip(proposed,target)) if draft!=truth),len(proposed))
        expected=(proposed[:first]+[target[first]])[:progress]
        audit.check(progress<=first+1 and block['committed_tokens']==expected==target[:progress],tag+': first rejection/bonus')
        audit.check(block['output_start']==offset and block['anchor_token']==output[offset],tag+': canonical block start')
        audit.check(block['committed_tokens']==output[offset+1:offset+progress+1],tag+': actual commit')
        audit.check(block['positions']==list(range(len(prompt)+offset,len(prompt)+offset+width)),tag+': absolute positions')
        audit.check(block['accepted_draft_tokens']==min(first,progress) and
            block['native_accepted_draft_tokens']==progress-1 and
            block['raw_accepted_draft_tokens_before_stop']==first and
            block['stop_clipped']==(progress<first+1),tag+': acceptance counters')
        native+=progress-1; matched+=min(first,progress); histogram[str(progress-1)]+=1
        offset+=progress
    calls=len(meta['trace'])
    audit.check(offset==len(output)-1,label+': full trajectory coverage')
    audit.check(meta['verification_calls']==calls and meta['accepted_draft_tokens']==matched and
        meta['committed_progress']==offset and native==offset-calls,label+': aggregate counts')
    counts=dict(verification_calls=calls,native_accepted_draft_tokens=native,
        accepted_draft_tokens=matched,committed_progress=offset,output_tokens=len(output),
        native_acceptance_histogram=dict(histogram),finish_reason=record['finish_reason'],
        cap_hit=record['finish_reason']=='length')
    for key in TIME_KEYS:
        value=observed[key] if key=='cache_maintenance_s' else meta[key]
        audit.check(type(value) in (int,float) and math.isfinite(value) and
                    (value>=0 if key=='cache_maintenance_s' else value>0),label+': '+key)
        counts[key]=value
    return counts

def fixed_slots(requests,permutations,warmups):
    for repeat in range(3):
        for block in range(4):
            base=('N','E','S') if block%2==0 else ('N','S','E')
            offset=(repeat+block//2)%3
            order=base[offset:]+base[:offset]
            for position,method in enumerate(order):
                activation=repeat*12+block*3+position
                for phase,request in [('warmup',r) for r in warmups]+[('scored',requests[i]) for i in permutations[repeat][block]]:
                    label=(f"warmup_a{activation:02d}_{request['panel_ordinal']}_{method}" if phase=='warmup'
                           else f"{request['panel_ordinal']:04d}_r{repeat}_{method}")
                    yield dict(phase=phase,method=method,repeat=repeat,block_ordinal=block,
                        position=position,activation=activation,label=label,request=request)

def interval(point,samples):
    finite=np.isfinite(samples)
    return dict(point=point,ci95=np.quantile(samples,[.025,.975],method='linear').tolist() if finite.all() else None,
                undefined_draws=int((~finite).sum()))

def independent_summary(paired):
    ids=sorted(paired)
    totals={}
    for method in METHODS:
        data=[paired[key][method] for key in ids]
        total={field:sum(row[field] for row in data) for field in COUNT_KEYS+TIME_KEYS}
        a,v=total['native_accepted_draft_tokens'],total['verification_calls']
        total.update(native_accepted_tokens_per_call=a/v if v else None,
            matched_accepted_tokens_per_call=total['accepted_draft_tokens']/v if v else None,
            committed_progress_per_call=total['committed_progress']/v if v else None,
            calls_per_1000_output_tokens=1000*v/total['output_tokens'])
        histogram=Counter()
        for row in data: histogram.update(row['native_acceptance_histogram'])
        positive=[row for row in data if row['verification_calls']>0]
        total.update(equal_request_native_acceptance_mean=statistics.fmean(row['native_accepted_draft_tokens']/row['verification_calls'] for row in positive) if positive else None,
            equal_request_denominator=len(positive),zero_call_requests=len(data)-len(positive),
            cap_hit_requests=sum(row['cap_hit'] for row in data),native_acceptance_histogram=dict(histogram))
        totals[method]=total
    draws=np.random.Generator(np.random.PCG64(1708)).integers(0,len(ids),size=(10000,len(ids)))
    comparison={}
    for candidate,control in (('E','N'),('E','S'),('S','N')):
        arrays={m:{field:np.asarray([paired[key][m][field] for key in ids],dtype=np.float64)
                  for field in ('verification_calls','native_accepted_draft_tokens','request_latency_s')}
                for m in (candidate,control)}
        samples={m:{field:values[draws].sum(axis=1) for field,values in fields.items()}
                 for m,fields in arrays.items()}
        with np.errstate(divide='ignore',invalid='ignore'):
            acceptance={m:samples[m]['native_accepted_draft_tokens']/samples[m]['verification_calls'] for m in samples}
            timed=1-samples[candidate]['request_latency_s']/samples[control]['request_latency_s']
            called=1-samples[candidate]['verification_calls']/samples[control]['verification_calls']
            accepted=acceptance[candidate]/acceptance[control]-1
            absolute=acceptance[candidate]-acceptance[control]
        c,b=totals[candidate],totals[control]
        ca,ba=c['native_accepted_tokens_per_call'],b['native_accepted_tokens_per_call']
        savings=arrays[control]['verification_calls']-arrays[candidate]['verification_calls']
        repetitions=[1-sum(paired[key][candidate]['repeat_times'][r] for key in ids)/sum(paired[key][control]['repeat_times'][r] for key in ids) for r in range(3)]
        comparison[candidate+'_vs_'+control]=dict(candidate=candidate,reference=control,
            time_reduction=interval(1-c['request_latency_s']/b['request_latency_s'],timed),
            call_reduction=interval(1-c['verification_calls']/b['verification_calls'] if b['verification_calls'] else None,called),
            acceptance_improvement=interval(ca/ba-1 if ca is not None and ba else None,accepted),
            acceptance_difference=interval(ca-ba if ca is not None and ba is not None else None,absolute),
            speedup=b['request_latency_s']/c['request_latency_s'],repeat_time_reductions=repetitions,
            requests_fewer_calls=int((savings>0).sum()),requests_more_calls=int((savings<0).sum()),
            request_call_ties=int((savings==0).sum()),gross_calls_saved=int(savings[savings>0].sum()),gross_calls_added=int(-savings[savings<0].sum()))
    def positive(effect,key):
        bounds=effect[key]['ci95']
        return bounds is not None and bounds[0]>0
    def practical(effect):
        return (effect['time_reduction']['point']>=.005 and positive(effect,'time_reduction')
            and positive(effect,'acceptance_improvement') and all(v>0 for v in effect['repeat_time_reductions']))
    en,es=comparison['E_vs_N'],comparison['E_vs_S']
    clears=positive(es,'time_reduction') and positive(es,'acceptance_improvement')
    outcome=('FEASIBLE_E_WITH_ADDED_VALUE_OVER_FIXED_LR_CONTROL' if clears else 'FEASIBLE_E_LR_ALTERNATIVE_UNRESOLVED') if practical(en) else 'NO_PRACTICAL_ADVANCEMENT_THIS_PILOT'
    decision=dict(outcome=outcome,scope=SCOPE,feasibility_pass=practical(en),
        practical_checks=dict(time_point_at_least_half_percent=en['time_reduction']['point']>=.005,
            time_lower_bound_positive=positive(en,'time_reduction'),acceptance_lower_bound_positive=positive(en,'acceptance_improvement'),
            all_three_time_directions_positive=all(v>0 for v in en['repeat_time_reductions'])),
        s_vs_n_descriptive_practical_pass=practical(comparison['S_vs_N']),automatic_followup_authorized=False)
    return dict(method_totals=totals,comparisons=comparison,decision=decision,request_count=len(ids),
        bootstrap=dict(draws=10000,seed=1708,generator='PCG64',order='lexicographic source_id',
            unit='paired whole request, all arms and three repeat times together',percentiles=[2.5,97.5],method='linear'))

def audit_final(audit,plan):
    directory=audit.root/'tasks/final/evaluation'
    status=audit.read(directory/'STATUS.json')
    audit.check(status['status']=='completed','final status complete')
    audit.equal(status,dict(attempted_generations=1224,returned_generations=1224,
        completed_valid_generations=1224,invalid_generations=0,scored_generations=1152,
        warmup_generations=72,publications=36,logical_activations=36,
        unknown_generation_attempts=0,unattempted_generations=0),'final accounting')
    protocol=audit.read(directory/'PROTOCOL.json')
    panel=audit.read(plan['data_panel']);requests=panel['requests'];permutations=panel['within_block_permutations']
    warmup_package=audit.read(directory/'WARMUPS.json');warmups=warmup_package['requests']
    for name,reference in warmup_package['canonicals'].items(): authenticate_reference(audit,reference,'warmup reference '+name)
    refs=audit.read(directory/'CANONICAL.json')
    audit.equal(refs,audit.read(plan['references']),'merged canonical identity')
    for name,reference in refs.items(): authenticate_reference(audit,reference,'panel reference '+name)
    audit.check(len(requests)==128 and len(warmups)==2,'fixed panel cardinalities')
    audit.check(len({r['source_id'] for r in requests})==128,'distinct source identities')
    audit.check(not ({r['source_id'] for r in requests}&{r['source_id'] for r in warmups}),'warmups outside scored panel')
    audit.equal(audit.read(directory/'REQUESTS.json'),requests,'sealed request identity')
    audit.check(protocol['scope']==SCOPE and protocol['capture'] is False and protocol['event']==8,'protocol scope')
    audit.check(protocol['execution_commit']==plan['execution_commit'],'actual execution commit')
    # Recreate the specified PCG64 permutation stream independently.
    rng=np.random.Generator(np.random.PCG64(1708))
    expected=[[rng.permutation(np.arange(block*32,(block+1)*32)).tolist() for block in range(4)] for repeat in range(3)]
    audit.equal(permutations,expected,'sealed PCG64 permutations')
    saved=audit.read(directory/'MEASUREMENTS.jsonl',lines=True)
    slots=list(fixed_slots(requests,permutations,warmups))
    expected_activations=[]
    for slot in slots:
        if slot['phase']=='warmup' and slot['request']['panel_ordinal']==0:
            entry={k:slot[k] for k in ('activation','repeat','block_ordinal','position','method')}
            entry['request_ordinals']=permutations[slot['repeat']][slot['block_ordinal']]
            expected_activations.append(entry)
    audit.equal(protocol['schedule'],expected_activations,'protocol activation schedule')
    activations=audit.read(directory/'ACTIVATIONS.jsonl',lines=True)
    audit.check(len(activations)==36,'36 actual activations')
    for actual,expected_activation in zip(activations,expected_activations):
        audit.equal(actual,expected_activation,'actual activation')
        audit.check(not actual['errors'] and actual['checkpoint_id']==protocol['checkpoints'][actual['method']], 'activation checkpoint/validity')
    boundaries=audit.read(directory/'BOUNDARY_CHECKS.jsonl',lines=True)
    audit.check([r['activation'] for r in boundaries]==[*range(36),'after_final'],'37 quiet boundaries')
    for boundary in boundaries:
        observation=boundary['check']
        audit.check(not boundary['errors'] and observation['status']=='passed' and
            observation['physical_gpu_uuid']==protocol['physical_gpu_uuid'] and
            all(observation[k] is True for k in ('both_cards_checked','other_gpu_idle','host_quiet','storage_admitted','time_admitted')),
            'quiet/storage/time boundary')
    for method in METHODS:
        weights=audit.read(directory/('WEIGHTS_'+method+'.json'))
        reply=weights['native_reply']
        audit.check(not weights['errors'] and reply.get('all_equal') is True and reply.get('status')=='passed' and
            reply['checkpoint_path']==protocol['checkpoints'][method],'published exact weights '+method)
    audit.check(len(saved)==len(slots)==1224,'all raw final executions')
    grouped=defaultdict(list)
    for index,slot in enumerate(slots):
        label=slot['label'];request=slot['request']
        reference=(warmup_package['canonicals'] if slot['phase']=='warmup' else refs)[request['request_id']]
        record=audit.read(directory/(label+'.json'))
        audit.check(record['record_id']==label and record['request_index']==request['panel_ordinal'] and
            record['session_id']=='persistent_prefix_withholding_event8',label+': request identity')
        rebuilt=native_counts(audit,record,request,reference,label,protocol['checkpoints'][slot['method']])
        if index<len(saved):
            row=saved[index]
            audit.equal(row,rebuilt,label+'/saved measurements')
            audit.equal(row,{k:v for k,v in slot.items() if k!='request'},label+'/schedule')
            audit.check(row['source_id']==request['source_id'] and row['physical_gpu_uuid']==protocol['physical_gpu_uuid'] and
                row['valid_measurement'] is True and row['validation_status']=='valid' and not row['errors'],label+': validity/card')
        grouped[slot['phase'],request['source_id'],slot['method']].append((slot['repeat'],rebuilt))
    paired={}
    for (phase,source,method),rows in grouped.items():
        first=rows[0][1]
        for _,row in rows[1:]:
            audit.equal(row,{k:first[k] for k in COUNT_KEYS+('finish_reason','native_acceptance_histogram')},source+'/'+method+'/repeat counts')
        if phase=='scored':
            audit.check(sorted(r for r,_ in rows)==[0,1,2],source+'/'+method+'/three repeats')
            value={key:first[key] for key in COUNT_KEYS+('cap_hit','native_acceptance_histogram')}
            value.update({key:statistics.fmean(row[key] for _,row in rows) for key in TIME_KEYS})
            value['repeat_times']=[row['request_latency_s'] for _,row in sorted(rows)]
            paired.setdefault(source,{})[method]=value
    audit.check(len(paired)==128 and all(set(values)==set(METHODS) for values in paired.values()),'complete paired requests')
    if audit.error_count:
        return None
    rebuilt=independent_summary(paired)
    production=audit.read(directory/'SUMMARY.json')
    audit.equal(production,rebuilt,'production summary')
    return rebuilt

def bound(audit,node,label):
    value=audit.read(node.get('path',node.get('manifest')))
    audit.equal(audit.inputs[str(audit.resolve(node.get('path',node.get('manifest'))))],node['identity'],label+' identity')
    return value


def authenticate_reference(audit,reference,label):
    """Retain old source authentication; directly reopen every new source."""
    body={k:v for k,v in reference.items() if k!='binding_digest'}
    digest=hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    audit.check(digest==reference['binding_digest'],label+': binding digest')
    key=reference['binding_digest']
    if key in audit.historical_references:
        audit.equal(reference,audit.historical_references[key],label+': immutable historical reference')
        audit.historical_reference_reuses.add(key)
        return
    source=audit.read(reference['source_record_path'])
    audit.equal(audit.inputs[str(audit.resolve(reference['source_record_path']))],reference['source_record_identity'],label+': new source bytes')
    audit.check(source['checkpoint_id']==reference['native_checkpoint']==audit.plan['origin_export'],label+': fixed K6 canonical source')
    for field in ('prompt_ids','response_ids','finish_reason'):
        audit.equal(source[field],reference[field],label+'/'+field)
    audit.new_reference_reopens.add(key)


def expected_work():
    result=dict(adam=673,response_forward=1365,response_backward=1364,
        feedback_generation=384,reference_generation=24,qualification_generation=8,
        warmup_generation=72,scored_generation=1152,teacher_preparation=384,training_initialization=9)
    completed=dict(result,response_forward=1364)
    return result,completed


def audit_work(audit,plan,proof):
    work=audit.read(audit.root/'WORK.json')
    attempted,completed=expected_work()
    audit.equal(work['attempted'],attempted,'cumulative attempted work')
    audit.equal(work['completed'],completed,'cumulative completed work')
    pending={'3453':dict(task='S8',label='forward_loss',counts=dict(response_forward=1))}
    audit.equal(work['inherited_pending'],pending,'old interrupted forward retained')
    audit.check(not work['pending'],'no unresolved new work')
    audit.equal({k:work['attempted'][k]-work['completed'][k]-work.get('failed',{}).get(k,0) for k in attempted},
        dict.fromkeys(attempted,0)|{'response_forward':1},'exact unknown accounting')
    audit.check(not any(work.get('failed',{}).values()),'no silently failed scientific operations')
    initial=audit.read(audit.root/'INITIAL_WORK.json')
    audit.equal(initial,proof['initial_work'],'imported initial ledger')
    remaining={k:attempted[k]-initial['attempted'][k] for k in attempted}
    expected_remaining=dict.fromkeys(attempted,0)|dict(adam=128,response_forward=258,response_backward=258,
        training_initialization=1,reference_generation=24,warmup_generation=72,scored_generation=1152)
    audit.equal(remaining,expected_remaining,'restart-only new work')
    audit.equal(proof['expected_attempted'],attempted,'authenticated attempted endpoint')
    audit.equal(proof['expected_completed'],completed,'authenticated completed endpoint')
    audit.equal(proof['remaining'],expected_remaining,'authorized remaining work')
    clock=audit.read(audit.root/'CLOCK.json')
    old=proof['clock']
    audit.check(not clock['active'],'all leases closed')
    audit.check(clock['first_model_load_epoch']==old['first_model_load_epoch']==1794802603.911516,'original clock epoch')
    audit.equal(clock['finished'][:len(old['finished'])],old['finished'],'all prior occupied costs retained')
    audit.check(old['completed_card_s']==2849.2045271396637,'prior card seconds')
    audit.equal(clock['completed_card_s'],sum(row['occupied_card_s'] for row in clock['finished']),'summed occupied time')
    audit.check(clock['completed_card_s']<=14400,'four card-hour cap')
    campaign=audit.read(audit.root/'CAMPAIGN.json')
    audit.check(campaign['status']=='completed' and set(campaign['completed'])==set(REUSED+NEW),'all ten tasks completed')
    audit.check(campaign['execution_commit']==plan['execution_commit'],'campaign execution commit')
    audit.check(campaign['actual_clock']['elapsed_s']<=21600,'six elapsed-hour cap')
    audit.check([row['task'] for row in campaign['schedule']][-1:] == ['final'] and
        {row['task'] for row in campaign['schedule']}==set(NEW),'only three new tasks scheduled')
    statuses={};costs=[]
    for task in REUSED+NEW:
        base=audit.root/'tasks'/task
        status=audit.read(base/'STATUS.json');receipt=audit.read(base/'RESULT.json')
        commit=OLD_COMMIT if task in REUSED else EXECUTION_COMMIT
        audit.check(status['status']==receipt['status']=='completed' and status['execution_commit']==commit,task+': task state/source identity')
        statuses[task]=dict(execution_commit=commit,reused=task in REUSED,status='completed',physical_gpu_uuid=status['physical_gpu_uuid'])
        costs.append(dict(task=task,execution_commit=commit,reused=task in REUSED,wall_s=status['wall_s'],
            costs=receipt.get('costs'),loading_s=receipt.get('loading_s'),startup=receipt.get('startup')))
        if task in ('N8','E8','S8'):
            endpoint=receipt['endpoint'];boundary=audit.read(base/'BOUNDARY.json')
            audit.equal(endpoint,dict(global_step=1024,presentations=2048),task+': final counters')
            audit.check(boundary['status']=='passed' and boundary['export_exact'] is True and boundary['historical_receipts_unchanged'] is True,task+': final own-state boundary')
            audit.check(boundary['live']['exact_resume'] is True and not boundary['live']['mismatches'] and boundary['live']['no_rng_reset'] is True,task+': exact boundary')
            audit.equal(receipt['physical_work'],dict(draft_forwards=256,response_forwards=256),task+': intrinsic forwards')
    final=audit.read(audit.root/'tasks/final/evaluation/PROTOCOL.json')
    audit.check(final['physical_gpu_uuid']==plan['devices']['A'],'fixed final card A')
    for arm in METHODS:
        endpoint=audit.read(audit.root/'tasks'/f'{arm}8'/'RESULT.json')['endpoint']
        audit.check(final['checkpoints'][arm]==endpoint['export'],arm+': own final export')
    return dict(attempted=work['attempted'],completed=work['completed'],unknown_response_forward=1,
        inherited_pending=pending,new_work=remaining,clock=clock,task_provenance=statuses,task_costs=costs,
        cost_interpretation='Old and new costs remain distinct; component timers overlap task wall time.')


def run(root,mappings=()):
    began=time.perf_counter();audit=Audit(root,mappings)
    output=dict(status='running',extra_model_forwards=0,extra_optimizer_steps=0,extra_generations=0,
        production_analysis_helpers_used=False,efficacy_computed=False,
        scope=SCOPE,independent_recomputation_scope='Fresh native raw tokens/traces and paired whole-request statistics; no new model execution')
    try:
        plan=audit.read(audit.root/'sealed/PLAN.json');audit.plan=plan
        audit.mappings.append((Path(plan['root']),audit.root))
        audit.check(plan['execution_commit']==EXECUTION_COMMIT,'actual restart execution commit')
        proof=bound(audit,plan['restart_import'],'restart proof')
        manifest=bound(audit,plan['production_restart'],'restart manifest')
        audit.check(proof['execution_commit']==plan['execution_commit'] and proof['binding']==plan['binding'],'proof/source/scientific identity')
        audit.check(proof['reused_tasks']==list(REUSED) and proof['remaining_tasks']==list(NEW),'exact import/restart scope')
        audit.check(proof['qualification_repeated'] is False and proof['feedback_recollected'] is False,'no duplicate scientific work')
        for name,node in proof['receipts'].items():
            receipt=bound(audit,node,'retained qualification '+name)
            audit.check(receipt['status']=='passed' and receipt['execution_commit']==node['execution_commit'],'retained passed qualification '+name)
        cpu=audit.read(plan['cpu_gate']);admission=audit.read(plan['resource_admission'])
        audit.check(cpu['status']=='passed' and cpu['execution_commit']==EXECUTION_COMMIT,'current CPU preflight')
        audit.check(admission['status']=='passed','resource admission')
        source=audit.read(plan['source_manifest'])
        audit.check(not source['dirty'] and source['commit']==EXECUTION_COMMIT,'clean source receipt')
        audit.historical_references={};audit.historical_reference_reuses=set();audit.new_reference_reopens=set()
        for key,count in (('reused_references',104),('warmup_references',2)):
            refs=audit.read(plan[key])
            audit.equal(audit.inputs[str(audit.resolve(plan[key]))],plan['input_identities'][plan[key]],key+' immutable bytes')
            audit.check(len(refs)==count,key+' count')
            audit.historical_references.update({ref['binding_digest']:ref for ref in refs.values()})
        provenance=audit.read(plan['reference_provenance'])
        audit.equal(audit.inputs[str(audit.resolve(plan['reference_provenance']))],plan['input_identities'][plan['reference_provenance']],'canonical authentication receipt')
        audit.check(provenance['status']=='passed' and provenance['new_reference_route']=='fixed_original_K6_native_greedy_target_verifier' and provenance['no_target_only_AR_claim'] is True,'fixed reference route')
        output['work']=audit_work(audit,plan,proof)
        if not audit.error_count:
            summary=audit_final(audit,plan)
            audit.check(len(audit.new_reference_reopens)==24,'all24 new canonical sources reopened')
            audit.check(len(audit.historical_reference_reuses)==106,'104 historical scored and two warmup canonical bindings')
            if summary is not None and not audit.error_count:
                output.update(status='passed',efficacy_computed=True,independently_recomputed=summary)
        output['canonical_authentication']=dict(historical_bindings_reused=len(audit.historical_reference_reuses),
            new_source_records_reopened=len(audit.new_reference_reopens),
            limitation='The106 historical bindings reuse pinned prior source authentication; their Fenrir original source files are not reopened here.')
        if not output['efficacy_computed']:output['status']='invalid_or_incomplete'
    except (OSError,ValueError,TypeError,KeyError,IndexError) as error:
        audit.check(False,'missing or malformed evidence: '+repr(error));output['status']='invalid_or_incomplete'
    output.update(checks=audit.checks,error_count=audit.error_count,errors=audit.errors,input_identities=audit.inputs,
        audit_wall_s=time.perf_counter()-began)
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--map',action='append',default=[],metavar='REMOTE=LOCAL')
    args=parser.parse_args()
    report=run(args.root,[tuple(v.split('=',1)) for v in args.map])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ('status','checks','error_count','efficacy_computed','audit_wall_s')}))
    raise SystemExit(0 if report['status']=='passed' else 1)


if __name__=='__main__':main()
