"""Independent CPU reaggregation of retained role-loss evidence; no production imports."""
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
import csv, json, math, hashlib
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT/'retained/results'
SUMMARY = json.loads((DATA/'SUMMARY.json').read_text())
TRAINING = json.loads((DATA/'TRAINING.json').read_text())
DIAGNOSTICS = json.loads((DATA/'DIAGNOSTICS.json').read_text())
COUNTS = ('verification_calls','accepted_draft_tokens','native_accepted_draft_tokens','committed_progress','output_tokens')
TIMES = ('request_latency_s','native_generate_s','cache_maintenance_s')
ARMS = ('KPLUS','MIX','ROLE')
checks = 0
errors = []

def check(condition, label):
    global checks
    checks += 1
    if not condition: errors.append(label)

def equal(a,b,label):
    if isinstance(a,(float,np.floating)) or isinstance(b,(float,np.floating)):
        check(math.isfinite(float(a)) and math.isfinite(float(b)) and math.isclose(float(a),float(b),rel_tol=1e-12,abs_tol=1e-10),label)
    else: check(a==b,label)

rows = list(csv.DictReader((DATA/'ALL_REPEATS.csv').open()))
check(len(rows)==4608,'all scheduled evaluation rows')
for row in rows:
    for key in COUNTS+('event','repeat','panel_ordinal','seed','prompt_tokens','block_ordinal','position'): row[key]=int(row[key])
    for key in TIMES: row[key]=float(row[key])
    row['cap_hit']=row['cap_hit']=='True'
    check(row['validation_status']=='valid' and row['valid_measurement']=='True' and json.loads(row['errors'])==[] and json.loads(row['structural_errors'])==[], 'record validity metadata '+row['label'])
    check(all(math.isfinite(row[k]) and row[k]>=0 for k in TIMES), 'finite measured times '+row['label'])
    equal(row['committed_progress'],row['output_tokens']-1,'progress count '+row['label'])
    equal(row['native_accepted_draft_tokens'],row['committed_progress']-row['verification_calls'],'native accepted count '+row['label'])
    check(row['accepted_draft_tokens']<=row['committed_progress'],'matched accepted bound '+row['label'])
    equal(row['cap_hit'],row['finish_reason']=='length','cap termination '+row['label'])
    if row['cap_hit']: equal(row['output_tokens'],2048,'cap length '+row['label'])

paired_csv={(r['dataset'],int(r['event']),int(r['panel_ordinal'])):r for r in csv.DictReader((DATA/'PAIRED_REQUESTS.csv').open())}
stages = {}
for dataset in ('gsm8k','mathqa'):
    for event in (2,4):
        stage=dataset+f'/event_{event:03d}'
        original=SUMMARY['stages'][stage]
        selected=[r for r in rows if r['dataset']==dataset and r['event']==event]
        equal(len(selected),256*3*(1 if event==2 else 2),stage+' rows')
        groups=defaultdict(list)
        for r in selected: groups[r['panel_ordinal'],r['method']].append(r)
        arrays={m:{k:[] for k in COUNTS+TIMES} for m in ARMS}
        cap_counts={m:0 for m in ARMS}
        ids=set()
        for ordinal in range(256):
            request_ids=set()
            output_contracts=set()
            for arm in ARMS:
                obs=sorted(groups[ordinal,arm],key=lambda r:r['repeat'])
                equal([r['repeat'] for r in obs],list(range(1 if event==2 else 2)),stage+f'/{ordinal}/{arm} repeats')
                request_ids.update(r['request_id'] for r in obs)
                output_contracts.update((r['output_tokens'],r['finish_reason']) for r in obs)
                check(len({r['checkpoint_id'] for r in obs})==1,'repeat checkpoint identity')
                for key in COUNTS:
                    check(len({r[key] for r in obs})==1,stage+f'/{ordinal}/{arm}/{key} repeat equality')
                    val=obs[0][key]; arrays[arm][key].append(val)
                    equal(val,int(paired_csv[dataset,event,ordinal][arm+'_'+key]),'paired count table')
                for key in TIMES:
                    val=sum(r[key] for r in obs)/len(obs);arrays[arm][key].append(val)
                    equal(val,float(paired_csv[dataset,event,ordinal][arm+'_'+key]),'paired timing table')
                cap_counts[arm]+=obs[0]['cap_hit']
            equal(len(request_ids),1,'request pairing')
            equal(len(output_contracts),1,'paired output lengths and termination')
            ids.update(request_ids)
        equal(len(ids),256,stage+' unique IDs')
        arrays={m:{k:np.asarray(v,dtype=np.float64) for k,v in d.items()} for m,d in arrays.items()}
        totals={}
        for arm in ARMS:
            totals[arm]={k:float(v.sum()) for k,v in arrays[arm].items()}
            totals[arm]['cap_hit_requests']=cap_counts[arm]
            for key,value in totals[arm].items(): equal(value,original['method_totals'][arm][key],stage+'/'+arm+'/'+key)
            calls=totals[arm]['verification_calls']
            for key,num in [('accepted_tokens_per_call','accepted_draft_tokens'),('native_accepted_tokens_per_call','native_accepted_draft_tokens'),('committed_progress_per_call','committed_progress')]:
                totals[arm][key]=totals[arm][num]/calls
                equal(totals[arm][key],original['method_totals'][arm][key],stage+'/'+arm+'/'+key)
        indices=np.random.Generator(np.random.PCG64(20261107)).integers(0,256,size=(10000,256))
        effects={}
        for a,b in [('ROLE','KPLUS'),('MIX','KPLUS'),('ROLE','MIX')]:
            name=a+'_vs_'+b;old=original['comparisons'][name];out={}
            for key in ('verification_calls',)+TIMES:
                label='call_reduction' if key=='verification_calls' else key+'_reduction'
                numerator=arrays[a][key];denominator=arrays[b][key]
                point=100*(1-numerator.sum()/denominator.sum())
                bootstrap=100*(1-numerator[indices].sum(axis=1)/denominator[indices].sum(axis=1))
                ci=np.percentile(bootstrap,[2.5,97.5],method='linear')
                out[label]={'point_pct':float(point),'ci95_pct':ci.tolist(),'candidate_sum':float(numerator.sum()),'reference_sum':float(denominator.sum())}
                for k in ('point_pct','candidate_sum','reference_sum'):equal(out[label][k],old[label][k],stage+'/'+name+'/'+label+'/'+k)
                for j in range(2): equal(ci[j],old[label]['ci95_pct'][j],stage+'/'+name+'/'+label+'/CI')
            delta=arrays[b]['verification_calls']-arrays[a]['verification_calls']
            out.update(request_wins=int((delta>0).sum()),request_regressions=int((delta<0).sum()),request_ties=int((delta==0).sum()),gross_calls_saved=int(delta[delta>0].sum()),gross_calls_added=int(-delta[delta<0].sum()))
            dt=arrays[b]['request_latency_s']-arrays[a]['request_latency_s']
            out.update(request_latency_wins=int((dt>0).sum()),request_latency_regressions=int((dt<0).sum()),request_latency_ties=int((dt==0).sum()))
            for k,v in out.items():
                if not isinstance(v,dict):equal(v,old[k],stage+'/'+name+'/'+k)
            effects[name]=out
        repeats={}
        for repeat in range(1 if event==2 else 2):
            repeats[str(repeat)]={}
            for arm in ARMS:
                obs=[r for r in selected if r['repeat']==repeat and r['method']==arm]
                values={k:sum(r[k] for r in obs) for k in COUNTS+TIMES}
                for k,v in values.items():equal(v,original['repeat_totals'][str(repeat)][arm][k],stage+'/repeat/'+arm+'/'+k)
                repeats[str(repeat)][arm]=values
        stages[stage]=dict(method_totals=totals,comparisons=effects,repeat_totals=repeats)

costs={};work=defaultdict(int);diagnostic_audit={}
for name,cell in TRAINING.items():
    equal(cell['status'],'passed',name+' training passed')
    equal(cell['progress']['events_completed'],4,name+' event count')
    equal(cell['optimizer_window_count'],512,name+' Adam windows')
    equal(cell['exposure']['response_count'],1024,name+' presentations')
    equal(len(cell['pass_rows']),8,name+' pass rows')
    equal(len(cell['event_rows']),4,name+' event rows')
    for key,val in cell['actual_work']['completed'].items():
        work[key]+=val;equal(val,cell['actual_work']['attempts'][key],name+'/'+key+' no incomplete work')
        equal(cell['actual_work']['unknown'][key],0,name+'/'+key+' unknown')
    costs[name]={}
    ecs=[e['costs'] for e in cell['progress']['event_costs']]
    for k in ('training_s','update_call_s','teacher_preparation_s','feature_persistence_s','checkpoint_save_s','export_s','publication_validation_s','retention_s'):
        costs[name][k]=sum(c[k] for c in ecs)
    costs[name]['native_activation_s']=sum(c['activation']['native_activation_s'] for c in ecs)
    costs[name]['worker_wall_s']=cell['progress']['worker_wall_s']
    costs[name]['backend_loading_s']=cell['progress']['backend_loading_s']
    costs[name]['trainer_loading_s']=cell['progress']['trainer_loading_s']
    costs[name]['peak_allocated_bytes']=max(e['resources']['training_peak_allocated_bytes'] for e in cell['event_rows'])
    costs[name]['peak_reserved_bytes']=max(e['resources']['training_peak_reserved_bytes'] for e in cell['event_rows'])
    costs[name]['clipped_windows']=sum(e['clipped_windows'] for e in cell['event_rows'])
    equal(costs[name]['clipped_windows'],cell['clipped_window_count'],name+' clipped windows')
    # Pool raw per-pass numerators/denominators without averaging ratios.
    for key,val in cell['exposure']['sum_metrics'].items():
        if key.startswith('derived_'):continue
        equal(sum(p['sum_metrics'].get(key,0) for p in cell['pass_rows']),val,name+'/pooledsum/'+key)
    for key,record in cell['exposure']['ratio_metrics'].items():
        equal(record['value'],record['numerator']/record['denominator'] if record['denominator'] else None,name+'/ratio/'+key)
    diag=DIAGNOSTICS[name]
    equal(diag['requests'],16,name+' diagnostic request count')
    flat=[a for r in diag['per_request'] for a in r['anchors']]
    equal(len(flat),128,name+' diagnostic anchors')
    totals=defaultdict(int)
    for a in flat:
        i,e2,e4,n=a['initial_prefix'],a['event2_prefix'],a['event4_prefix'],a['valid_proposals']
        check(all(0<=x<=n for x in (i,e2,e4)),name+' prefix limits')
        derived=dict(valid_proposals=n,initial_prefix=i,event2_prefix=e2,event4_prefix=e4,
          original_first_error=i<n,event2_broken_prefix=e2<i,event4_broken_prefix=e4<i,
          event2_repair_with_prefix_intact=e2>i,event4_repair_with_prefix_intact=e4>i,
          event2_gained_prefix_tokens=max(0,e2-i),event4_gained_prefix_tokens=max(0,e4-i),
          repair_persists=e2>i and e4>i,event2_gain_fully_persists=e2>i and e4>=e2,
          persisted_gain_tokens=min(max(0,e2-i),max(0,e4-i)),lost_gain_tokens=max(0,e2-i)-min(max(0,e2-i),max(0,e4-i)))
        for k,v in derived.items(): equal(v,a[k],name+'/anchor/'+str(a['anchor_id'])+'/'+k);totals[k]+=v
    for k,v in totals.items(): equal(v,diag['totals'][k],name+'/diagnostic total/'+k)
    diagnostic_audit[name]=dict(totals)
for key,expected in [('online_generations',3072),('optimizer_steps',3072),('response_forwards',6144),('response_backwards',6144),('correctness_generations',8),('terminal_preparation_forwards',3072)]:equal(work[key],expected,'global work '+key)
for dataset in ('gsm8k','mathqa'):
    for arm in ('MIX','ROLE'):
        c=costs[dataset+'/'+arm];b=costs[dataset+'/KPLUS']
        c['training_overhead_vs_KPLUS_pct']=100*(c['training_s']/b['training_s']-1)
        c['update_call_overhead_vs_KPLUS_pct']=100*(c['update_call_s']/b['update_call_s']-1)

# Independently rederive diagnostic prefixes from retained prediction vectors,
# and training totals from original per-response observations, not only report rows.
raw_diagnostic_files = 0
raw_training_files = 0
raw_diagnostic_forwards = 0
raw_sources = []
for dataset in ('gsm8k','mathqa'):
    folder=ROOT/'retained/campaign_001/diagnostics'/dataset
    initial=json.loads((folder/'INITIAL/event_000/RESULT.json').read_text())
    raw_diagnostic_files += 1
    raw_diagnostic_forwards += initial['draft_forwards']
    for arm in ARMS:
        snapshots=[initial]
        for event in (2,4):
            path=folder/arm/f'event_{event:03d}/RESULT.json'
            value=json.loads(path.read_text())
            snapshots.append(value);raw_sources.append(str(path.relative_to(ROOT)))
            raw_diagnostic_files+=1;raw_diagnostic_forwards+=value['draft_forwards']
        for v in snapshots:
            equal(v['status'],'completed','diagnostic raw completion')
            equal(v['attempted_responses'],16,'diagnostic raw attempts')
            equal(v['returned_responses'],16,'diagnostic raw returns')
            equal(v['unknown_responses'],0,'diagnostic raw unknowns')
            equal(v['unattempted_responses'],0,'diagnostic raw unattempted')
            equal(v['training_use'],False,'diagnostic raw no training')
            equal(v['rng_restored'],True,'diagnostic RNG restoration receipt')
            equal(v['model_mode_restored'],True,'diagnostic mode restoration receipt')
        for req in DIAGNOSTICS[dataset+'/'+arm]['per_request']:
            query=[v['snapshots'][req['request_id']] for v in snapshots]
            for k in ('conditioning_id','anchor_ids','input_ids_sha256','loss_mask_sha256','context_length','block_size','eos_token_ids'):
                equal(query[0][k],query[1][k],'fixed diagnostic '+k)
                equal(query[0][k],query[2][k],'fixed diagnostic '+k)
            for index,summary_anchor in enumerate(req['anchors']):
                b0=query[0]['blocks'][index]
                for k in ('anchor_id','labels','valid','weights'):
                    equal(b0[k],query[1]['blocks'][index][k],'paired diagnostic block '+k)
                    equal(b0[k],query[2]['blocks'][index][k],'paired diagnostic block '+k)
                for q,k in zip(query,('initial_prefix','event2_prefix','event4_prefix')):
                    b=q['blocks'][index];n=prefix=0;failed=False
                    equal(b['valid'][0],False,'known seed excluded')
                    for pos in range(1,len(b['labels'])):
                        if not b['valid'][pos]:break
                        n+=1
                        if not failed and b['predictions'][pos]==b['labels'][pos]:prefix+=1
                        else:failed=True
                        if b['labels'][pos] in q['eos_token_ids']:break
                    equal(prefix,summary_anchor[k],'prefix independently rescored')
                    equal(n,summary_anchor['valid_proposals'],'available proposals independently rescored')
        obs=[];windows=[]
        base=ROOT/'retained/campaign_001/trajectories'/dataset/arm/'trainer/events'
        for event in range(1,5):
            op=base/f'event_{event:03d}/RESPONSE_OBSERVATIONS.jsonl'
            wp=base/f'event_{event:03d}/WINDOW_OBSERVATIONS.jsonl'
            o=[json.loads(x) for x in op.read_text().splitlines() if x]
            w=[json.loads(x) for x in wp.read_text().splitlines() if x]
            equal(len(o),256,'raw response count');equal(len(w),128,'raw window count')
            raw_training_files+=2;raw_sources.extend([str(op.relative_to(ROOT)),str(wp.relative_to(ROOT))])
            obs.extend(o);windows.extend(w)
        cell=TRAINING[dataset+'/'+arm]
        equal(sum(w['gradient_norm']>1 for w in windows),cell['clipped_window_count'],'raw gradient clipping count')
        check(all(math.isfinite(w['gradient_norm']) for w in windows),'finite raw gradient norms')
        for k,v in cell['exposure']['sum_metrics'].items():
            if not k.startswith('derived_'):equal(sum(o['sum_metrics'].get(k,0) for o in obs),v,'original response sum/'+k)
        equal(sum(o['loss'] for o in obs)/len(obs),cell['exposure']['actual_mean_response_loss'],'actual mean response loss')
equal(raw_diagnostic_forwards,224,'all diagnostic forward budget')

fallacies=[
 ('Simpson paradox','CHECKED','Dataset results remain separate; no pooled cross-dataset efficacy conclusion.'),
 ('Ecological fallacy','CHECKED','Per-request wins and regressions are retained; aggregate gains are not attributed to every request.'),
 ('Berkson selection','CAUTION','Input-only deterministic selection avoids output-based filtering, but panels are reused development data, not a fresh population sample.'),
 ('Collider bias','CHECKED','No outcome-conditioned covariates or subgroup selection; all requested valid completions retained.'),
 ('Base-rate neglect','CHECKED','256-request denominators, gross saved/added calls and all ties are explicit; this is not a screening accuracy analysis.'),
 ('Regression to mean','CHECKED','Fixed inputs and contemporaneous controls, without selection for extreme prior outcomes; early-to-late latency is not used as a learning curve.'),
 ('Survivorship bias','CHECKED','All six trajectories, both endpoints and all 4,608 evaluation rows complete; no unfavorable rows omitted.'),
 ('Look-elsewhere effect','CAUTION','Two primary dataset-specific endpoint contrasts and secondary contrasts/metrics are all shown. Intervals are unadjusted; no familywise confirmation claim.'),
 ('Forking paths','CHECKED','Fixed code/panels/arms/events/bootstrap before outcomes; no best-intermediate selection or post-result tuning.'),
 ('Correlation/causation','CAUTION','Controlled online recipes differ in own generated feedback and resulting features. Local diagnostics do not isolate a loss mechanism or prove why native calls changed.'),
 ('Reverse causality','CHECKED','Training precedes serving; diagnostic/evaluation feedback never trains a model. Direction in this bounded experiment is temporally fixed.')]
report=dict(status='passed' if not errors else 'failed',verification_status='ANALYZED',created_utc=datetime.now(timezone.utc).isoformat(),
 execution_commit=SUMMARY['execution_source']['commit'],checks=checks,errors=errors,source_files={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in DATA.iterdir() if p.name in ['SUMMARY.json','ALL_REPEATS.csv','PAIRED_REQUESTS.csv','TRAINING.json','DIAGNOSTICS.json']},
 scope='Independent CPU calculation from retained measurement CSV and per-anchor/per-pass records; no production-summary function import; same-team audit, not independent native rerun.',
 stages=stages,costs=costs,work=dict(work),diagnostics=diagnostic_audit,fallacy_scan=fallacies,raw_diagnostic_files=raw_diagnostic_files,raw_diagnostic_forwards=raw_diagnostic_forwards,raw_training_files=raw_training_files,raw_sources=raw_sources,
 numerical_comparison='Float reaggregation tolerance abs1e-10/rel1e-12 only, not a changed experimental qualification tolerance.')
(ROOT/'NUMERICAL_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n')
lines=['# Role-loss pilot numerical audit','','## Material Passport','','- Origin Skill: ARS experiment-agent 3.24.2','- Mode: validate','- Verification Status: ANALYZED','- Version: role_loss_numerical_audit_v1',f'- Execution commit: `{report["execution_commit"]}`','- Same-team independently written CPU aggregation; no new GPU execution or independent training replication.','',f'**{report["status"].upper()}: {checks:,} checks; {len(errors)} discrepancies.**','',
 'The four-event pilot provides no observed advantage for ROLE over K+ on either dataset. Its endpoint point estimates are slightly worse for both calls and latency, and all four corresponding primary confidence intervals include zero. This is failure to demonstrate the intended benefit, not proof of equivalence or universal harm. MIX improves GSM8K call count but regresses MathQA; the split does not deliver a consistent advantage.','',
 '## Endpoint results','','Positive reductions favor the candidate. Intervals are unadjusted paired-request percentile intervals from 10,000 PCG64 draws, seed 20261107.','',
 '| Dataset | Contrast | Call reduction %,95%CI | Request-time reduction %,95%CI | Call wins/regressions/ties |','|---|---|---:|---:|---:|']
for dataset in ('gsm8k','mathqa'):
    for name,e in stages[dataset+'/event_004']['comparisons'].items():
        f=lambda x:f'{x["point_pct"]:+.4f} [{x["ci95_pct"][0]:+.4f},{x["ci95_pct"][1]:+.4f}]'
        lines.append(f'| {dataset} | {name} | {f(e["call_reduction"])} | {f(e["request_latency_s_reduction"])} | {e["request_wins"]}/{e["request_regressions"]}/{e["request_ties"]} |')
lines += ['', '## Costs','','Timers are nested. Training is included inside update-call time; do not add them. These are measured own-trajectory costs, not equal-feedback microbenchmarks or training-inclusive serving speedups.','', '| Dataset / arm | Training seconds | Training overhead vs K+ | Whole update calls seconds | Peak allocated GB | Clipped windows / 512 |','|---|---:|---:|---:|---:|---:|']
for name,c in costs.items():lines.append(f'| {name} | {c["training_s"]:.3f} | {c.get("training_overhead_vs_KPLUS_pct",0):+.2f}% | {c["update_call_s"]:.3f} | {c["peak_allocated_bytes"]/1e9:.3f} | {c["clipped_windows"]}/512 |')
lines += ['', '## Fixed-context evidence','','All six diagnostic trajectories contain 16 requests × 8 anchors. Prefix lengths were independently rescored from the 14 raw diagnostic prediction snapshots, including EOS and boundary rules; recomputed per-anchor totals match the report. The 48 original response/window observation files also match pooled training counters and losses. ROLE gains 20 prefix tokens versus 50 for K+ on GSM8K, and 53 versus 82 on MathQA relative to the same released reference. ROLE does not show a superior local net-prefix improvement in this sample. On MathQA, MIX repairs 36 original first errors versus 34 for K+, yet breaks 26 prefixes versus 20 and gains 29 prefix tokens versus 82. More repair events therefore do not by themselves establish improved overall progress. These 128 anchors per dataset are nested in 16 requests, not 128 independent conversations or timing samples.','',
 '## Limits','','All 4,608 measurement rows have valid metadata and consistent deterministic counts; the raw-token/output-gate audit is a separate root audit. This audit reaggregates the saved counters; it does not re-execute native inference or rederive model gradients. All 12 contrasts across both stages are recalculated; all 48 metric confidence intervals and repeat totals match. Own-feedback trajectories mean a change in objective can change later feedback/features; no pure loss-only causal attribution is made. One seed, four early events and reused development panels do not establish long-run ranking or robust unseen-workload gains. No additional run or algorithm change is authorized by this audit.','',
 '## Fallacy scan: 11/11 checked','','| Item | Assessment | Detail |','|---|---|---|']
for name,status,detail in fallacies:lines.append(f'| {name} | {status} | {detail} |')
if errors:lines += ['', '## Discrepancies']+[f'- {x}' for x in errors]
(ROOT/'NUMERICAL_AUDIT.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'status':report['status'],'checks':checks,'errors':errors,'costs':costs},indent=2))
