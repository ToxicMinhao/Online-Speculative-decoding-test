"""CPU-only independent arithmetic from retained native MathQA records.

The qualified native-record validator is reused for structural/canonical checks.
Pair construction, totals, signs and bootstrap reductions below do not call the
production statistics implementation. No model tensors are loaded.
"""
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ANALYSIS=Path(__file__).resolve().parents[1]
CHECKOUT=ANALYSIS/'analysis_checkout/DFlash-Online'
sys.path.insert(0,str(CHECKOUT))
from experiments.o_half_matched_lr.archive import validate_record

METHODS=('KPLUS','FSG','SCALAR')
COUNTS=('verification_calls','accepted_draft_tokens','native_accepted_draft_tokens','committed_progress','output_tokens')
TIMES=('request_latency_s','native_generate_s','cache_maintenance_s')
COMPARED=(('FSG','KPLUS'),('FSG','SCALAR'),('SCALAR','KPLUS'))
INPUTS={}


def identity(path):
    data=Path(path).read_bytes()
    return dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())


def read(path):
    path=Path(path);INPUTS[str(path)]=identity(path)
    return json.loads(path.read_text())


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def csv_write(path,rows):
    with Path(path).open('w',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def check_raw(raw, request, canonical, checkpoint, *, label, session):
    errors=validate_record(raw,request,checkpoint,canonical,label=label,session=session)
    if errors:raise ValueError((label,errors))
    trace=raw['metadata']['trace'];metadata=raw['metadata']
    counts=dict(verification_calls=len(trace),accepted_draft_tokens=sum(t['accepted_draft_tokens'] for t in trace),
        native_accepted_draft_tokens=sum(t['native_accepted_draft_tokens'] for t in trace),
        committed_progress=sum(t['committed_progress'] for t in trace),output_tokens=len(raw['response_ids']))
    assert counts['committed_progress']==counts['output_tokens']-1
    assert counts['native_accepted_draft_tokens']==counts['committed_progress']-counts['verification_calls']
    for field in ('verification_calls','accepted_draft_tokens','committed_progress'):
        assert counts[field]==metadata[field],(label,field)
    assert metadata['block_size']==16 and metadata['context_limit']==4096
    assert metadata['runtime']=='pinned_SGLang_FA3'
    assert metadata['cache_policy']=='idle_full_flush_before_every_request_and_publication'
    assert metadata['feedback_capture_in_timed_request'] is False
    assert raw['max_new_tokens']==2048 and raw['purpose']=='evaluation'
    return counts


def origin_counts(current_root):
    base=Path('/home/minhao2/EAGLE_SPEC/research/dflash_role_loss_pilot_20261107')
    old=base/'result_review_001/retained'
    inventory=read(base/'result_review_001/REMOTE_INVENTORY.json')
    inventory={str(old/row['path']):{k:row[k] for k in ('bytes','sha256')} for row in inventory['files']}
    retrieval=read(base/'result_review_001/RETRIEVAL.json');assert retrieval['status']=='passed'
    sealed=current_root/'sealed';plan=read(sealed/'PLAN.json')
    requests=read(sealed/'mathqa/PANEL.json')['requests'];canonical=read(sealed/'mathqa/CANONICALS.json')
    prior_plan=read(base/'production_001/sealed/PLAN.json')
    assert prior_plan['serving']==plan['serving']
    for name in ('MODEL_PIN.json','RUNTIME_PIN.json'):
        assert identity(sealed/name)==identity(base/'production_001/sealed'/name)
        INPUTS[str(sealed/name)]=identity(sealed/name)
        INPUTS[str(base/'production_001/sealed'/name)]=identity(base/'production_001/sealed'/name)
    folder=old/'campaign_001/evaluation/mathqa/event_004'
    prior_requests=read(folder/'REQUESTS.json');assert requests==prior_requests
    origin=plan['datasets']['mathqa']['origin_export']
    status=read(old/'campaign_001/trajectories/mathqa/KPLUS/STATUS.json')
    assert status['final_checkpoint']==origin
    assert status['boundary']['weights_identity']==plan['datasets']['mathqa']['origin_weights_identity']
    totals=defaultdict(int);rows=[]
    for ordinal,request in enumerate(requests):
        pair=[]
        for repeat in range(2):
            path=folder/f'{ordinal:04d}_r{repeat}_KPLUS.json'
            assert identity(path)==inventory[str(path)],path
            raw=read(path)
            counts=check_raw(raw,request,canonical[request['request_id']],origin,
                label=f'{ordinal:04d}_r{repeat}_KPLUS',session=raw['session_id'])
            pair.append(counts)
        assert pair[0]==pair[1],request['request_id']
        rows.append(dict(panel_ordinal=ordinal,request_id=request['request_id'],**pair[0]))
        for field in COUNTS:totals[field]+=pair[0][field]
    for name in (folder/'AUDIT.json',folder/'STATUS.json',old/'campaign_001/trajectories/mathqa/KPLUS/STATUS.json'):
        assert identity(name)==inventory[str(name)],name
    receipt=dict(status='passed',requests=256,raw_repeat_records=512,
        origin_checkpoint=origin,origin_weights=plan['datasets']['mathqa']['origin_weights_identity'],
        original_execution_commit=retrieval['scientific_execution_commit'],
        exact_panel=True,exact_canonical_tokens_termination=True,exact_model_runtime_serving=True,
        repeats_consistent=True,historical_timings_used=False,counts=dict(totals))
    return rows,receipt


def endpoint(current_root):
    start=time.perf_counter();current_root=Path(current_root)
    plan=read(current_root/'sealed/PLAN.json');assert plan['execution_source']['commit']=='6440e8ff5af1a845f830d65c884f6711d7e4a687'
    folder=current_root/'campaign_001/evaluation/mathqa'
    requests=read(current_root/'sealed/mathqa/PANEL.json')['requests']
    canonical=read(current_root/'sealed/mathqa/CANONICALS.json')
    assert len(requests)==256
    assert read(folder/'REQUESTS.json')==requests
    assert read(folder/'CANONICAL.json')==canonical
    protocol=read(folder/'PROTOCOL.json')
    assert protocol['schedule']==plan['datasets']['mathqa']['evaluation_schedule']
    status=read(folder/'STATUS.json')
    for name in ('planned_generations','attempted_generations','returned_generations','completed_valid_generations'):
        assert status[name]==1536,(name,status[name])
    for name in ('invalid_generations','unknown_generation_attempts','unattempted_generations','optimizer_steps','feedback_collections','teacher_preparations','extra_generations'):
        assert status[name]==0,(name,status[name])
    assert status['status']=='completed'
    measurement_path=folder/'MEASUREMENTS.jsonl';INPUTS[str(measurement_path)]=identity(measurement_path)
    saved=[json.loads(line) for line in measurement_path.read_text().splitlines()]
    assert len(saved)==1536 and len(protocol['schedule'])==1536
    values={};repeat_rows=[];checkpoint_identities={}
    for method in METHODS:
        result=read(current_root/f'campaign_001/mathqa/event_006/{method}/RESULT.json')
        assert result['status']=='completed' and result['end_step']==768
        assert protocol['checkpoints'][method]==result['checkpoint_path']
        checkpoint_identities[method]=dict(checkpoint=result['checkpoint_path'],weights=result['weights_identity'])
    for saved_row,slot in zip(saved,protocol['schedule']):
        for key,value in slot.items():assert saved_row[key]==value,(key,value)
        assert saved_row['validation_status']=='valid' and saved_row['valid_measurement'] and not saved_row['errors']
        ordinal,method,repeat=slot['panel_ordinal'],slot['method'],slot['repeat']
        assert method in METHODS and repeat in (0,1)
        request=requests[ordinal];assert saved_row['request_id']==request['request_id']
        label=f'{ordinal:04d}_r{repeat}_{method}'
        raw=read(folder/(label+'.json'))
        counts=check_raw(raw,request,canonical[request['request_id']],protocol['checkpoints'][method],label=label,session='functional_step_geometry_evaluation')
        for field in COUNTS:assert counts[field]==saved_row[field],(label,field)
        timing={field:raw['metadata'][field] for field in TIMES[:2]}
        timing['cache_maintenance_s']=raw['metadata']['capture']['cache_maintenance_s']
        for field,value in timing.items():
            assert value==saved_row[field] and math.isfinite(value) and value>=0
        assert raw['finish_reason']==saved_row['finish_reason']
        assert saved_row['cap_hit']==(raw['finish_reason']=='length')
        key=(ordinal,method,repeat);assert key not in values
        values[key]=dict(**counts,**timing,cap_hit=saved_row['cap_hit'],finish_reason=raw['finish_reason'])
        repeat_rows.append(dict(panel_ordinal=ordinal,request_id=request['request_id'],method=method,repeat=repeat,**values[key]))
    paired=[];arrays={method:{field:[] for field in COUNTS+TIMES} for method in METHODS};totals={}
    for ordinal,request in enumerate(requests):
        for method in METHODS:
            a,b=(values[(ordinal,method,repeat)] for repeat in (0,1))
            for field in COUNTS+('cap_hit','finish_reason'):assert a[field]==b[field]
            row=dict(panel_ordinal=ordinal,request_id=request['request_id'],method=method,
                     **{field:a[field] for field in COUNTS},cap_hit=a['cap_hit'],finish_reason=a['finish_reason'],
                     **{field:(a[field]+b[field])/2 for field in TIMES})
            paired.append(row)
            for field in COUNTS+TIMES:arrays[method][field].append(row[field])
    for method in METHODS:
        arrays[method]={field:np.asarray(value,dtype=np.float64) for field,value in arrays[method].items()}
        row={field:int(sum(arrays[method][field])) for field in COUNTS}
        row.update({field:float(sum(arrays[method][field])) for field in TIMES})
        row.update(mean_request_time_s=row['request_latency_s']/256,
            matched_accepted_tokens_per_call=row['accepted_draft_tokens']/row['verification_calls'],
            native_accepted_tokens_per_call=row['native_accepted_draft_tokens']/row['verification_calls'],
            committed_progress_per_call=row['committed_progress']/row['verification_calls'],
            cap_hit_requests=sum(values[(i,method,0)]['cap_hit'] for i in range(256)))
        totals[method]=row
    indices=np.random.Generator(np.random.PCG64(20261108)).integers(0,256,size=(10000,256))
    comparisons={};comparison_rows=[]
    for candidate,reference in COMPARED:
        result={}
        for field in ('verification_calls',)+TIMES:
            c,r=arrays[candidate][field],arrays[reference][field]
            bootstrap=100*(1-c[indices].sum(axis=1)/r[indices].sum(axis=1))
            ci=np.quantile(bootstrap,[.025,.975],method='linear')
            point=100*(1-c.sum()/r.sum())
            metric=dict(reduction_pct=float(point),reduction_ci95_pct=ci.tolist(),
                increase_pct=float(-point),increase_ci95_pct=[float(-ci[1]),float(-ci[0])],
                candidate_sum=float(c.sum()),reference_sum=float(r.sum()),
                wins=int(np.count_nonzero(c<r)),regressions=int(np.count_nonzero(c>r)),ties=int(np.count_nonzero(c==r)))
            if field=='verification_calls':metric.update(gross_saved=int(np.maximum(r-c,0).sum()),gross_added=int(np.maximum(c-r,0).sum()))
            result[field]=metric
            comparison_rows.append(dict(candidate=candidate,reference=reference,metric=field,
                reduction_pct=metric['reduction_pct'],reduction_ci_low=ci[0],reduction_ci_high=ci[1],
                increase_pct=metric['increase_pct'],increase_ci_low=-ci[1],increase_ci_high=-ci[0],
                wins=metric['wins'],regressions=metric['regressions'],ties=metric['ties']))
        comparisons[candidate+'_vs_'+reference]=result
    sweeps={str(repeat):{method:{field:sum(values[(i,method,repeat)][field] for i in range(256)) for field in COUNTS+TIMES} for method in METHODS} for repeat in (0,1)}
    summary=dict(status='passed',source_commit=plan['execution_source']['commit'],binding=plan['binding'],
        requests=256,executions=1536,checkpoints=checkpoint_identities,method_totals=totals,
        comparisons=comparisons,sweep_totals=sweeps,bootstrap=dict(resamples=10000,seed=20261108,
        bit_generator='PCG64',percentile_method='linear',same_indices_across_contrasts=True,
        indices_identity=hashlib.sha256(indices.tobytes()).hexdigest()),
        model_work=dict(loads=0,forwards=0,backwards=0,optimizer_steps=0,generations=0))
    generated=read(folder/'SUMMARY.json');differences=[]
    for method,row in totals.items():
        for field,value in row.items():
            if field not in generated['method_totals'][method]:continue
            old=generated['method_totals'][method][field]
            if not math.isclose(value,old,rel_tol=1e-14,abs_tol=1e-13):differences.append([method,field,value,old])
    for name,result in comparisons.items():
        old=generated['comparisons'][name]
        for field,metric in result.items():
            target=old['call_reduction' if field=='verification_calls' else field+'_reduction']
            assert metric['reduction_pct']==target['point_pct'],(name,field)
            assert metric['reduction_ci95_pct']==target['ci95_pct'],(name,field)
    assert not differences,differences
    summary['saved_summary_reproduction']=dict(bootstrap_points_intervals_exact=True,totals_within_float_summation_roundoff=True)
    prior_rows,prior=origin_counts(current_root);summary['origin']=prior
    origin_array=np.asarray([row['verification_calls'] for row in prior_rows],dtype=np.int64)
    progress=[]
    for method in METHODS:
        c=arrays[method]['verification_calls'];delta=origin_array-c
        summary['origin'][method]=dict(calls=int(c.sum()),origin_calls=int(origin_array.sum()),
            reduction_pct=100*(1-c.sum()/origin_array.sum()),
            wins=int(np.count_nonzero(delta>0)),regressions=int(np.count_nonzero(delta<0)),ties=int(np.count_nonzero(delta==0)),
            gross_calls_saved=int(np.maximum(delta,0).sum()),gross_calls_added=int(np.maximum(-delta,0).sum()))
    for old in prior_rows:
        ordinal=old['panel_ordinal']
        progress.append(dict(panel_ordinal=ordinal,request_id=old['request_id'],KPLUS4=old['verification_calls'],
            **{m:int(arrays[m]['verification_calls'][ordinal]) for m in METHODS}))
    output=ANALYSIS/'endpoint_review';output.mkdir(exist_ok=True)
    summary['analysis_wall_s']=time.perf_counter()-start
    write(output/'ENDPOINT_INDEPENDENT.json',summary)
    write(output/'INPUT_IDENTITIES.json',INPUTS)
    csv_write(output/'PER_REQUEST.csv',paired);csv_write(output/'ALL_REPEATS.csv',repeat_rows)
    csv_write(output/'COMPARISONS.csv',comparison_rows);csv_write(output/'KPLUS4_COUNTS.csv',progress)
    print(json.dumps(dict(status=summary['status'],totals=totals,comparisons=comparisons,origin=summary['origin'],analysis_wall_s=summary['analysis_wall_s']),indent=2))


if __name__=='__main__':endpoint(ANALYSIS/'evidence/pilot')
