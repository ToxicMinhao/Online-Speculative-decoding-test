"""Independent, stdlib-only, read-only reconstruction of completed training records.

Run on A800 through stdin. Does not import project/model code or write remotely.
"""
import collections
import hashlib
import json
import math
from pathlib import Path
import time

ROOT = Path('/data/sources/minhao/runs/dflash_full_k_gsm8k_20261031/production_002')
COUNTS = collections.Counter()
FILES = {}
FAILURES = []
START = time.monotonic()

def check(ok, label, category='general'):
    COUNTS[category] += 1
    if not ok:
        FAILURES.append(label)
        if len(FAILURES) >= 50:
            raise RuntimeError('At least 50 failed conditions: ' + str(FAILURES))

def read(path, lines=False):
    path = Path(path)
    data = path.read_bytes()
    FILES[str(path)] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    return [json.loads(x) for x in data.splitlines()] if lines else json.loads(data)

def identity(path):
    path=Path(path)
    if str(path) not in FILES:
        data=path.read_bytes()
        FILES[str(path)]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
    return FILES[str(path)]

def near(a,b):
    return math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,rel_tol=2e-6,abs_tol=2e-5)

plan=read(ROOT/'sealed/PLAN.json')
manifest=read(plan['manifest'])
expected_work={'online_generations':7473,'correctness_generations':4,'intermediate_generations':384,
               'terminal_preparation_forwards':7424,'response_forwards':14848,
               'response_backwards':14848,'optimizer_steps':7424}
check(plan['execution_source']['commit']=='0698eeb2ddd97b9e2ac6953989dc82acbc0a1e34','execution source')
check(plan['seed']==17 and plan['events']==58 and plan['automatic_retry'] is False,'seed/horizon/retry')
check(len(manifest['train_requests'])==7473,'stream size')
check(plan['budget']['total_generations']==28010 and plan['budget']['disposable_optimizer_steps']==0,'budget')
report_audit=read(ROOT/'campaign_001/TRAINING_AUDIT.json')
output={'verification_status':'ANALYZED','scope':'Independent compact-record arithmetic and causal-order review; no tensor deserialization, model execution, GPU rerun or remote writes.',
        'execution_source':plan['execution_source'],'arms':{},'findings':FAILURES}

for arm,lr in [('K',.0003),('K_HALF',.00015)]:
    base=ROOT/'campaign_001/arms'/arm
    status=read(base/'STATUS.json')
    work=read(base/'WORK.jsonl',lines=True)
    started=collections.Counter(); done=collections.Counter()
    for row in work:
        kind,phase,ordinal=row['kind'],row['phase'],row['ordinal']
        check(kind in expected_work and phase in ('started','completed'),'work kind/phase','work')
        if phase=='started':
            check(ordinal==started[kind]+1 and started[kind]==done[kind],arm+' work start ordinal','work')
            steps=done['optimizer_steps']
            if kind=='online_generations': check(steps==min((ordinal-1)//128,58)*128,'own update before online response','causal')
            elif kind=='terminal_preparation_forwards':
                event=(ordinal-1)//128+1
                check(done['online_generations']==event*128 and steps==(event-1)*128,'prepare after own response pool','causal')
            elif kind in ('response_forwards','response_backwards'):
                check(done['terminal_preparation_forwards']==((ordinal-1)//256+1)*128,'training follows preparation','causal')
                if kind=='response_forwards': check(steps==(ordinal-1)//2,'forward exact optimizer boundary','causal')
                else: check(done['response_forwards']==ordinal,'backward original forward','causal')
            elif kind=='optimizer_steps': check(done['response_forwards']==done['response_backwards']==2*ordinal,'two responses per optimizer step','causal')
            elif kind=='correctness_generations': check(done['online_generations']==128 and steps==(0 if ordinal==1 else 128),'correctness inherited ordering','causal')
            elif kind=='intermediate_generations':
                endpoint=(14,29,44)[(ordinal-1)//128]
                check(steps==endpoint*128 and done['online_generations']==endpoint*128,'fixed intermediate endpoint','causal')
            started[kind]+=1
        else:
            check(ordinal==done[kind]+1 and started[kind]==ordinal,'work completion ordinal','work')
            done[kind]+=1
    check(dict(started)==dict(done)==expected_work,arm+' exact completed work','work')
    check(status['actual_work']['attempts']==status['actual_work']['completed']==expected_work and not any(status['actual_work']['unknown'].values()),arm+' status work reconciliation','work')
    check(len(status['segments'])==1 and status['segments'][0]['resume'] is False and len(status['worker_segments_s'])==1,'single fresh worker, no scientific restart','restart')
    check(status['events_completed']==58 and status['tail_completed']==49 and not status['invalid_online_records'] and status['status']=='completed','status endpoint')
    initial=read(base/'trainer/INITIAL_CHECK.json')
    for key,value in {'status':'passed','arm':arm,'completed_event':0,'global_step':0,'presentations':0,'adam_state_entries':0,'cuda_rng_states':1,'export_exact':True,'extra_model_passes':0,'extra_optimizer_steps':0}.items():
        check(initial.get(key)==value,'initial '+key,'state_receipt')
    if arm=='K_HALF':
        paired=initial['paired_initialization']
        check(paired['status']=='passed' and paired['exact_released_weights_empty_adam_rng'] is True and paired['rng_overwritten'] is False,'independent paired initialization','state_receipt')
    retained=[{'path':initial['state_path'],'bytes':Path(initial['state_path']).stat().st_size,'inspection':'exists/size only; equality from recorded initial validation'}]
    total_cost=collections.Counter(); exposure=collections.Counter(); native_work=collections.Counter()
    retired_kinds=collections.Counter(); retired_bytes=0; compact_bindings=0; seen_requests=[]; native_paths=set()
    event_rows=[]; peaks=[]
    for event in range(1,59):
        eid=f'event_{event:03d}'
        eroot=base/'events'/eid
        troot=base/'trainer/events'/eid
        result=read(troot/'RESULT.json')
        controller=read(eroot/'loop/events'/f'{eid}.json')
        observations=read(troot/'RESPONSE_OBSERVATIONS.jsonl',lines=True)
        windows=read(troot/'WINDOW_OBSERVATIONS.jsonl',lines=True)
        expected_ids=[r['request_id'] for r in manifest['train_requests'][(event-1)*128:event*128]]
        features=result['features']; ids=[x['record_id'] for x in features]
        check(ids==expected_ids and len(set(ids))==128,'event original stream order','feedback')
        seen_requests.extend(ids)
        check(controller['training']==result,'controller/native receipt equality','event')
        check(len(result['records'])==len(observations)==256 and len(windows)==128,'response/window counts','event')
        for key,value in {'event_id':eid,'objective':'osd_first_error','status':'completed','passes':2,'response_presentations':256,'optimizer_steps':128,'start_step':(event-1)*128,'end_step':event*128,'reduction':'response_mean'}.items():
            check(result.get(key)==value,'event '+key,'event')
        check(not (troot/'FAILED.json').exists(),'no hidden event failure','event')
        check(result['selection_totals_source']=='optimizer_boundary_accumulation_windows_once','exposure reducer','exposure')
        sums=collections.Counter(); anchors=0
        for index,(row,obs) in enumerate(zip(result['records'],observations)):
            stepped=index%2==1
            check(row['response_observation']==obs,'response observation copy','response')
            check(row['record_id']==obs['record_id'] and row['pass']==obs['pass_index']==index//128 and obs['presentation_ordinal']==index and obs['event_id']==eid,'response identity','response')
            check(row['optimizer_stepped'] is stepped and row['optimizer_step_after']==(event-1)*128+(index+1)//2 and row['learning_rate']==(lr if stepped else None),'response optimizer accounting','response')
            check(math.isfinite(obs['loss']) and all(math.isfinite(v) for v in row['metrics'].values()),'finite response data','response')
            draws=row['sampled_anchors']
            check(draws==obs['sampled_anchors'] and len(draws)==1,'one native anchor draw','anchors')
            draw=draws[0]
            positions,keep=draw['positions'][0],draw['keep_mask'][0]
            check(len(draw['positions'])==len(draw['keep_mask'])==1 and 0<len(positions)==len(keep)<=512,'anchor shape','anchors')
            check(all(type(x) is int and x>=0 for x in positions) and all(type(x) is bool for x in keep),'anchor values','anchors')
            anchors+=sum(keep)
            for name,value in obs['sum_metrics'].items():
                check(math.isfinite(value),'finite sum metric','exposure')
                sums[name]+=value
            check(0<=obs['sum_metrics']['selected_position_occurrences']<=obs['sum_metrics']['available_position_occurrences'],'selection bounds','exposure')
        for pi in (0,1): check(sorted(x['record_id'] for x in result['records'][pi*128:(pi+1)*128])==sorted(ids),'each own response once per pass','response')
        for i,w in enumerate(windows):
            pair=observations[2*i:2*i+2]
            check(w['scope']=='optimizer_window' and w['event_id']==eid and w['window']==i+1 and w['global_step']==(event-1)*128+i+1,'window identity','window')
            check(w['record_ids']==[r['record_id'] for r in pair] and w['learning_rate']==lr,'window response/LR','window')
            check(w['mean_response_loss']==sum(r['loss'] for r in pair)/2 and math.isfinite(w['gradient_norm']) and w['gradient_norm']>=0,'response-mean loss and finite gradient','window')
        for name,value in result['selection_totals'].items():
            check(value==sum(r['metrics'][name] for r in result['records'] if r['optimizer_stepped']),'sum optimizer boundary once '+name,'exposure')
            if name in sums: check(near(value,sums[name]),'sum original responses '+name,'exposure')
            exposure[name]+=value
        exposure['sampled_anchor_occurrences']+=anchors
        check(result['objective_work']['reference_operations']==0 and result['objective_work']['work']['response_forwards']==result['objective_work']['work']['draft_forwards']==256,'no extra reference/current forward','work')
        native_work.update(result['objective_work']['work'])
        event_cost=dict(result['costs'])
        check(near(event_cost['training_s'],sum(row['training_s'] for row in result['records'])),'training cost sum','cost')
        event_cost['blocking_update_s']=read(eroot/'loop/events'/f'{eid}.costs.json')['update_call_s']
        event_cost['teacher_preparation_s']=sum(x['teacher_preparation_s'] for x in controller['preparation'])
        event_cost['feature_persistence_s']=sum(x['feature_persistence_s'] for x in controller['preparation'])
        event_cost['activation_s']=controller['activation']['activation_s']
        expected_checkpoint=plan['draft'] if event==1 else str(base/'trainer/events'/f'event_{event-1:03d}'/'export')
        expected_retired={}
        for request,feature,prepared in zip(manifest['train_requests'][(event-1)*128:event*128],features,controller['preparation']):
            rec=read(eroot/'loop/serving'/f"{request['request_id']}.json")
            meta=rec['metadata']; capture=meta['capture']; combined=rec['prompt_ids']+rec['response_ids']
            check(rec['record_id']==request['request_id'] and rec['request_index']==request['request_index'] and rec['prompt_ids']==request['prompt_ids'] and rec['checkpoint_id']==expected_checkpoint and rec['purpose']=='serving','own feedback preceding checkpoint','feedback')
            check(meta['feedback_capture_in_timed_request'] is True and capture['collect_features'] is True and capture['native_capture_validated'] is True,'native feedback capture','feedback')
            check(capture['target_argmax_ids']==rec['response_ids'] and capture['covered_token_ids']==combined[:-1],'native label/feature binding','feedback')
            check(prepared['input_ids']==combined and prepared['target_argmax_ids']==rec['response_ids'] and prepared['label_mismatches']==[],'exact own preparation','feedback')
            check(prepared['record_id']==feature['record_id']==rec['record_id'] and prepared['feature_path']==feature['path'] and feature['length']==len(combined)<=2272 and prepared['response_only_loss'] is True and prepared['final_feature_logits_used_for_labels'] is False,'response-only no-truncation preparation','feedback')
            check(prepared['prompt_length']==len(rec['prompt_ids']) and prepared['response_length']==len(rec['response_ids'])==prepared['supervised_positions'],'response exposure','feedback')
            terminal=prepared['terminal_feature']
            check(terminal['input_ids']==combined and terminal['positions']==[len(combined)-1] and terminal['final_feature_logits_used_for_labels'] is False,'terminal feature does not supply label','feedback')
            check(meta['verification_calls']==len(meta['trace']) and meta['accepted_draft_tokens']==sum(x['accepted_draft_tokens'] for x in meta['trace']),'native scalar training trace','feedback')
            np=meta['native_trace_path']; check(np not in native_paths,'unique native generation','work'); native_paths.add(np)
            for path,kind in [(capture['feature_path'],'native_capture'),(feature['path'],'prepared_feature'),(terminal['tensor_path'],'terminal_feature')]: expected_retired[path]=kind
            timing=read(eroot/'loop/serving'/f"{request['request_id']}.costs.json")
            for name,value in {'controller_serving_s':timing['serve_call_s'],'native_request_s':meta['request_latency_s'],'native_generate_s':meta['native_generate_s'],'cache_maintenance_s':capture['cache_maintenance_s'],'capture_host_s':capture['capture_host_s'],'capture_persistence_s':capture['feature_persistence_s']}.items(): event_cost[name]=event_cost.get(name,0)+value
        for key,value in result['objective_work'].items():
            if key.endswith('_s'): event_cost['objective_'+key]=value
        for name,value in event_cost.items(): check(math.isfinite(value) and value>=0,'finite component cost '+name,'cost')
        total_cost.update(event_cost)
        for key,value in event_cost.items(): check(near(value,report_audit['arms'][arm]['event_audits'][event-1]['costs'][key]),'report cost '+key,'cost')
        boundary=read(base/'boundaries'/f'{eid}.json'); validation=read(base/'boundaries'/f'{eid}_validation.json')
        check(boundary['scientific_check']==validation and boundary['status']=='passed' and boundary['arm']==arm and boundary['event']==event,'boundary identity','state_receipt')
        for key,value in {'status':'passed','arm':arm,'completed_event':event,'global_step':event*128,'presentations':event*256,'adam_state_entries':58,'cuda_rng_states':1,'export_exact':True,'extra_model_passes':0,'extra_optimizer_steps':0}.items(): check(validation.get(key)==value,'boundary '+key,'state_receipt')
        live=validation['live']; check(live['status']=='passed' and live['exact_resume'] is True and live['mismatches']==[] and live['no_rng_reset'] is True and live['global_step']==event*128 and live['presentations']==event*256,'live exact resume receipt','state_receipt')
        for key in ('state_identity','weights_identity'): check(boundary[key]['bytes']>0 and len(boundary[key]['sha256'])==64,'tensor identity receipt','state_receipt')
        if event>1:
            prior=read(base/'boundaries'/f'event_{event-1:03d}.json')
            retired_state=read(base/'boundaries'/f'event_{event-1:03d}_retirement.json')
            check(retired_state['status']=='completed' and retired_state['retired_state']==prior['state'] and retired_state['identity']==prior['state_identity'] and retired_state['successor']==boundary['state'],'state successor retirement','retention')
            check(not Path(prior['state']).exists(),'retired state absent','retention')
            retired_kinds['previous_optimizer_state']+=1
            retired_bytes+=prior['state_identity']['bytes']
            expected_retired[str(Path(expected_checkpoint)/'model.safetensors')]='previous_export_weights'
        retirement=read(eroot/'RETIREMENT.json'); retirement_plan=read(eroot/'RETIREMENT_PLAN.json')
        check(retirement['status']=='completed' and retirement['files']==retirement['removed']==retirement_plan['files'] and retirement['compact_evidence']==retirement_plan['compact_evidence'],'retirement plan exactly committed','retention')
        check({x['path']:x['kind'] for x in retirement['removed']}==expected_retired,'only own qualified tensors retired','retention')
        check(retirement['removed_bytes']==sum(x['bytes'] for x in retirement['removed']),'retirement byte sum','retention')
        retired_bytes+=retirement['removed_bytes']
        for item in retirement['removed']:
            retired_kinds[item['kind']]+=1
            check(not Path(item['path']).exists(),'retired artifact absent','retention')
        for item in retirement['compact_evidence']:
            ident=identity(item['path'])
            check(ident=={k:item[k] for k in ('bytes','sha256')},'retained compact identity','retention')
            compact_bindings+=1
        resource=read(eroot/'RESOURCE_COSTS.json')
        check(all(resource[k]==0 for k in ('extra_forwards','extra_backwards','extra_optimizer_steps')),'passive resource recording','resources')
        check(0<resource['training_peak_allocated_bytes']<=resource['training_peak_reserved_bytes'],'memory scope','resources')
        for timer in resource['training_compiler_timers'].values(): check(timer['calls']==len(timer['durations_s']) and timer['total_s']==sum(timer['durations_s']) and all(math.isfinite(v) and v>=0 for v in timer['durations_s']),'compiler timers separately accounted','resources')
        peaks.append(resource)
        if event in (14,29,44):
            unchanged=read(base/'evaluation'/eid/'TRAINING_STATE_UNCHANGED.json')
            check(unchanged['status']=='passed' and unchanged['export_exact'] is True and unchanged['live']['exact_resume'] is True and unchanged['live']['mismatches']==[] and unchanged['global_step']==event*128,'intermediate state preserved','state_receipt')
        event_rows.append({'event':event,'costs':event_cost,'sampled_anchor_occurrences':anchors,'selection_totals':result['selection_totals']})
    check(len(set(seen_requests))==7424 and seen_requests==[r['request_id'] for r in manifest['train_requests'][:7424]],'no response reused across events','feedback')
    for request in manifest['train_requests'][7424:]:
        rec=read(base/'tail'/f"{request['request_id']}.json")
        check(rec['record_id']==request['request_id'] and rec['prompt_ids']==request['prompt_ids'] and rec['checkpoint_id']==plan['checkpoints'][arm] and rec['metadata']['feedback_capture_in_timed_request'] is False,'tail final checkpoint, no training capture','feedback')
        native_paths.add(rec['metadata']['native_trace_path'])
    native_inventory=list(base.glob('native_attempt_*/native/request_[0-9]*.json'))
    check(len(native_inventory)==7861 and len(list(base.glob('native_attempt_*')))==1,'exact native execution inventory, single attempt','work')
    check(all(Path(x).exists() for x in native_paths),'native training/tail traces retained','retention')
    check(len(native_paths)==7473,'exact own online native identities','work')
    final=read(base/'FINAL_STATE_CHECK.json')
    check(final['status']=='passed' and final['completed_event']==58 and final['global_step']==7424 and final['presentations']==14848 and final['export_exact'] is True and final['live']['exact_resume'] is True and final['live']['mismatches']==[],'complete own final state','state_receipt')
    for name,key in [('state','state_identity'),('weights','weights_identity')]:
        path=Path(boundary[name]); check(path.is_file() and path.stat().st_size==boundary[key]['bytes'],'retained final identity size','retention')
        retained.append({'path':str(path),**boundary[key],'inspection':'exists/size matched; original SHA receipt, not rehashed in this review'})
    pointer=base/'trainer/training_state.pt'
    check(pointer.exists() and pointer.samefile(Path(boundary['state'])),'latest state points to retained final','retention')
    for key,value in exposure.items(): check(near(value,report_audit['arms'][arm]['exposure'][key]),'reported full exposure '+key,'exposure')
    check(dict(native_work)==report_audit['arms'][arm]['actual_native_work'],'reported measured native work','work')
    for key,value in total_cost.items(): check(near(value,report_audit['arms'][arm]['costs'][key]),'reported arm cost '+key,'cost')
    output['arms'][arm]={'work_ledger_rows':len(work),'completed_work':dict(done),'events':58,'native_inventory':len(native_inventory),
        'single_fresh_worker':True,'retained_initial_final':retained,'retired_paths_by_kind':dict(retired_kinds),'retired_bytes':retired_bytes,
        'compact_evidence_bindings_rehashed':compact_bindings,'exposure':dict(exposure),'actual_native_work':dict(native_work),
        'summed_event_costs_s':dict(total_cost),'worker_wall_s':sum(status['worker_segments_s']),
        'event_peak_allocated_bytes':max(x['training_peak_allocated_bytes'] for x in peaks),
        'event_peak_reserved_bytes':max(x['training_peak_reserved_bytes'] for x in peaks), 'event_rows':event_rows}
    print(json.dumps({'progress':arm,'checks':sum(COUNTS.values()),'errors':len(FAILURES)}),flush=True)

final_native=list((ROOT/'campaign_001').glob('final_native*/native/request_[0-9]*.json'))
output['checks_by_category']=dict(COUNTS)
output['checks_total']=sum(COUNTS.values())
output['unique_compact_files_read_or_rehashed']=len(FILES)
output['input_identity_manifest']=FILES
output['wall_s']=time.monotonic()-START
output['status']='passed' if not FAILURES else 'findings'
print('REVIEW_RESULT_JSON_BEGIN',flush=True)
print(json.dumps(output,indent=2),flush=True)
