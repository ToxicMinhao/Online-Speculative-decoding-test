"""Read-only accounting audit; no production validation/reduction functions imported."""
from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import json
import math

HERE = Path(__file__).resolve().parent
ROOT = HERE / 'retained'
REMOTE = '/data/sources/minhao/runs/dflash_role_loss_pilot_20261107/production_001/'
checks = Counter()
errors = []
sources = {}


def read(path):
    path = Path(path)
    raw = path.read_bytes()
    sources[str(path.relative_to(ROOT))] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    return json.loads(raw)


def lines(path):
    path = Path(path)
    raw = path.read_bytes()
    sources[str(path.relative_to(ROOT))] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    return [json.loads(row) for row in raw.splitlines() if row]


def check(condition, group, description):
    checks[group] += 1
    if not condition:
        errors.append(dict(group=group, description=description))


def local(path):
    return ROOT / path.removeprefix(REMOTE)


def finite(value, context):
    if isinstance(value, float):
        check(math.isfinite(value), 'finite_numeric', context)
    elif isinstance(value, dict):
        for key, child in value.items():
            finite(child, context + '/' + key)
    elif isinstance(value, list):
        for child in value:
            finite(child, context)


def prefix(block):
    result = 0
    for valid, predicted, label in zip(block['valid'][1:], block['predictions'][1:], block['labels'][1:]):
        if not valid or predicted != label:
            break
        result += 1
    return result


plan = read(ROOT / 'sealed/PLAN.json')
reported = read(ROOT / 'results/TRAINING.json')
reported_diagnostics = read(ROOT / 'results/DIAGNOSTICS.json')
summaries, feedback, diagnostic_summaries, retained = {}, {}, {}, Counter()
sampling = {}
total_work = Counter()
for dataset in ('gsm8k', 'mathqa'):
    training_requests = read(local(plan['datasets'][dataset]['training_manifest']))['train_requests']
    evaluation_requests = read(local(plan['datasets'][dataset]['panel_manifest']))['requests']
    check(len(training_requests) == 512 and len(evaluation_requests) == 256, 'input', dataset)
    check(not ({r['source_id'] for r in training_requests} & {r['source_id'] for r in evaluation_requests}), 'input', dataset+' source disjoint')
    check(not ({r['normalized_stem'] for r in training_requests} & {r['normalized_stem'] for r in evaluation_requests}), 'input', dataset+' normalized disjoint')
    for arm in ('KPLUS', 'MIX', 'ROLE'):
        key = dataset + '/' + arm
        base = ROOT / 'campaign_001/trajectories' / key
        status = read(base / 'STATUS.json')
        initial = read(base / 'trainer/INITIAL_CHECK.json')
        check(status['status'] == 'completed' and status['events_completed'] == 4, 'trajectory', key)
        check(initial['global_step'] == initial['presentations'] == initial['adam_state_entries'] == initial['completed_event'] == 0,
              'initialization', key+' fresh counters')
        check(initial['original_K_initialization']['exact_released_weights_empty_adam_rng'] is True
              and initial['original_K_initialization']['rng_overwritten'] is False
              and initial['original_weights_loaded_into_candidate'] is False
              and initial['export_exact'] is True, 'initialization', key+' exact saved-state receipt')
        work = defaultdict(Counter)
        ordinals = defaultdict(list)
        for row in lines(base/'WORK.jsonl'):
            work[row['phase']][row['kind']] += 1
            ordinals[(row['phase'], row['kind'])].append(row['ordinal'])
        required = dict(online_generations=512, correctness_generations=4 if arm == 'KPLUS' else 0,
                        terminal_preparation_forwards=512, response_forwards=1024, response_backwards=1024, optimizer_steps=512)
        for name, count in required.items():
            check(work['started'][name] == work['completed'][name] == count, 'work', key+' '+name)
            check(ordinals[('started',name)] == ordinals[('completed',name)] == list(range(1,count+1)), 'work', key+' ordered '+name)
            check(status['actual_work']['attempts'][name] == status['actual_work']['completed'][name] == count
                  and status['actual_work']['unknown'][name] == 0, 'work', key+' status '+name)
            total_work[name] += count
        check(set(work).issubset({'started','completed'}), 'work', key+' no failure ledger')
        metrics = Counter()
        sampling[key] = []
        ratio_totals = defaultdict(lambda: [0.,0.])
        gradnorms, all_feedback, active_anchors, pass_counts = [], [], 0, []
        costs = Counter()
        peak_allocated = peak_reserved = 0
        for event in range(1,5):
            name = f'event_{event:03d}'
            event_root = base / 'trainer/events' / name
            result = read(event_root/'RESULT.json')
            obs = lines(event_root/'RESPONSE_OBSERVATIONS.jsonl')
            windows = lines(event_root/'WINDOW_OBSERVATIONS.jsonl')
            boundary = read(base/'boundaries'/f'{name}.json')
            publication = read(base/'events'/name/'PUBLICATION_CHECK.json')
            buffer = read(base/'events'/name/'loop/feedback_buffer/state.json')
            resources = read(base/'events'/name/'RESOURCE_COSTS.json')
            retirement = read(base/'events'/name/'RETIREMENT.json')
            retired = Counter(row['kind'] for row in retirement['files'])
            check(retirement['status'] == 'completed', 'retention', key+'/'+name)
            retained.update(retired)
            peak_allocated = max(peak_allocated, resources['training_peak_allocated_bytes'])
            peak_reserved = max(peak_reserved, resources['training_peak_reserved_bytes'])
            check(result['status'] == 'completed' and result['method'] == arm and result['dataset'] == dataset, 'event', key+'/'+name)
            check(result['passes'] == 2 and result['response_presentations'] == 256 and result['optimizer_steps'] == 128,
                  'event', key+'/'+name+' accounting')
            check(result['start_step'] == (event-1)*128 and result['end_step'] == event*128,
                  'event', key+'/'+name+' step continuity')
            check(result['counters'] == dict(global_step=event*128,presentations=event*256,completed_events=event),
                  'event', key+'/'+name+' global counters')
            check(result['reduction'] == 'response_mean' and result['objective'] == 'osd_first_error', 'event', key+'/'+name+' K settings')
            check(len(result['features']) == 128 and len(result['records']) == len(obs) == 256 and len(windows) == 128,
                  'event', key+'/'+name+' source record counts')
            check(obs == [row['response_observation'] for row in result['records']], 'observations', key+'/'+name+' original-forward equality')
            check(publication['all_equal'] is True and publication['status'] == 'passed'
                  and publication['trainer_status']['accumulation_remainder'] == 0
                  and publication['trainer_status']['horizon'] == 7424, 'publication', key+'/'+name)
            scientific = boundary['scientific_check']
            check(boundary['status'] == scientific['status'] == scientific['live']['status'] == 'passed'
                  and scientific['global_step'] == event*128 and scientific['presentations'] == event*256
                  and scientific['adam_state_entries'] == 58 and scientific['export_exact'] is True
                  and scientific['live']['exact_resume'] is True and scientific['live']['no_rng_reset'] is True
                  and scientific['live']['mismatches'] == [], 'boundary', key+'/'+name+' exact receipt')
            check(scientific['extra_model_passes'] == scientific['extra_optimizer_steps'] == 0,
                  'boundary', key+'/'+name+' no work in checks')
            expected_requests = training_requests[(event-1)*128:event*128]
            records = buffer['records']
            check([row['record_id'] for row in records] == [row['request_id'] for row in expected_requests],
                  'feedback', key+'/'+name+' order')
            record_by_id = {row['record_id']:row for row in records}
            check(len(record_by_id) == 128, 'feedback', key+'/'+name+' uniqueness')
            expected_checkpoint = plan['draft'] if event == 1 else str(base/'trainer/events'/f'event_{event-1:03d}'/'export').replace(str(ROOT),REMOTE[:-1])
            # The plan's draft pin is an artifact object, while records use its configured path.
            if isinstance(expected_checkpoint,dict):
                expected_checkpoint = expected_checkpoint.get('path') or expected_checkpoint.get('checkpoint_path')
            for row, request in zip(records,expected_requests):
                identity = key+'/'+name+'/'+row['record_id']
                check(row['prompt_ids'] == request['prompt_ids'], 'feedback', identity+' exact prompt')
                check(row['purpose'] == 'serving' and row['max_new_tokens'] == 2048, 'feedback', identity+' canonical generation')
                check(row['checkpoint_id'] == expected_checkpoint, 'feedback', identity+' own pre-event checkpoint')
                check(0 < len(row['response_ids']) <= 2048 and row['finish_reason'] in ('eos','length'), 'feedback', identity+' bounds')
                check(row['metadata']['feedback_capture_in_timed_request'] is True, 'feedback', identity+' actual capture')
                all_feedback.append((row['record_id'],row['response_ids'],row['finish_reason']))
            features = {row['record_id']:row for row in result['features']}
            for pass_index in (0,1):
                pass_rows = [row for row in result['records'] if row['pass']==pass_index]
                check(len(pass_rows) == 128 and set(row['record_id'] for row in pass_rows) == set(record_by_id), 'presentation', key+'/'+name+' pass coverage')
            for ordinal,(row,observation) in enumerate(zip(result['records'],obs)):
                identity = key+'/'+name+f'/presentation_{ordinal}'
                check(row['optimizer_step_after'] == (event-1)*128+(ordinal+1)//2
                      and row['optimizer_stepped'] == bool(ordinal%2), 'presentation', identity+' native accumulation')
                if row['optimizer_stepped']:
                    check(row['learning_rate']==0.0003, 'presentation', identity+' learning rate')
                check(observation['record_id']==row['record_id'] and observation['pass_index']==row['pass']
                      and observation['scope']=='one_original_response_forward', 'observations', identity+' identity')
                check(row['sampled_anchors']==observation['sampled_anchors'], 'anchors', identity+' captured once')
                feature=features[row['record_id']]
                response=record_by_id[row['record_id']]
                check(feature['prompt_length']==len(response['prompt_ids']) and feature['response_tokens']==len(response['response_ids'])
                      and feature['length']==len(response['prompt_ids'])+len(response['response_ids'])
                      and feature['length']<=2272 and feature['adjacent_response_positions']==len(response['response_ids'])-1,
                      'features',identity+' context/response geometry')
                check(len(row['sampled_anchors'])==1, 'anchors',identity+' one native sampling call')
                draw=row['sampled_anchors'][0]
                positions=draw['positions'][0];keep=draw['keep_mask'][0]
                chosen=[p for p,k in zip(positions,keep) if k]
                check(len(positions)==len(keep) and len(chosen)==row['active_anchors']==min(512,feature['adjacent_response_positions'])
                      and chosen==sorted(set(chosen)) and all(feature['prompt_length']<=p<feature['length']-1 for p in chosen),
                      'anchors',identity+' eligible unique sorted count')
                active_anchors += len(chosen)
                sampling[key].append((row['record_id'], row['pass'], row['sampled_anchors']))
                finite(observation,identity)
                sm=observation['sum_metrics']
                check(sm['kplus_response_count']==1 and sm['role_loss_response_count']==1, 'observations',identity+' no recomputation duplication')
                check(sm['supervised_position_exposure']==sm['selected_position_occurrences']
                      and 0<=sm['selected_position_occurrences']<=sm['available_position_occurrences'], 'exposure',identity)
                original_mass=sm['kplus_W1']+sm['kplus_W2']
                check(abs(sm['kplus_first_error_weight_after']-original_mass)<=2e-7+2e-6*original_mass,'mass',identity)
                metrics.update(sm)
                for metric,pair in observation['ratio_metrics'].items():
                    ratio_totals[metric][0]+=pair[0];ratio_totals[metric][1]+=pair[1]
            for index,window in enumerate(windows):
                pair=obs[2*index:2*index+2]
                check(window['window']==index+1 and window['global_step']==(event-1)*128+index+1
                      and window['record_ids']==[item['record_id'] for item in pair], 'window',key+'/'+name+str(index))
                check(window['learning_rate']==0.0003 and math.isclose(window['mean_response_loss'],sum(item['loss'] for item in pair)/2,rel_tol=1e-12,abs_tol=1e-12),
                      'window',key+'/'+name+str(index)+' response average')
                check(math.isfinite(window['gradient_norm']), 'window',key+'/'+name+str(index)+' finite gradient')
                gradnorms.append(window['gradient_norm'])
            counts=result['objective_work']['work']
            check(counts['draft_forwards']==counts['response_forwards']==256 and counts['layer_forwards']==1280,
                  'physical_work', key+'/'+name+' no objective extra backbone pass')
            check(counts['vocabulary_head_calls']==2*counts['objective_chunks'], 'physical_work',key+'/'+name+' checkpoint recomputation head accounting')
            check(result['objective_work']['reference_operations']==0, 'physical_work',key+'/'+name+' no reference model')
            for cost,value in result['costs'].items(): costs[cost]+=value
        final=read(base/'FINAL_STATE_CHECK.json')
        check(final['global_step']==512 and final['presentations']==1024 and final['completed_event']==4
              and final['live']['mismatches']==[], 'boundary',key+' final')
        check(len(all_feedback)==512, 'feedback',key+' total')
        feedback[key]=all_feedback
        summaries[key]=dict(steps=512,presentations=1024,feedback=512,active_anchors=active_anchors,
                           clipped_steps=sum(x>1 for x in gradnorms),gradient_norm_mean=sum(gradnorms)/512,
                           gradient_norm_max=max(gradnorms),sum_metrics=dict(metrics),
                           pooled_ratios={name:dict(numerator=n,denominator=d,value=n/d if d else None) for name,(n,d) in ratio_totals.items()},
                           peak_allocated_bytes=peak_allocated,peak_reserved_bytes=peak_reserved,costs=dict(costs))
        check(reported[key]['optimizer_window_count']==512 and reported[key]['clipped_window_count']==summaries[key]['clipped_steps'], 'report_match',key+' windows')
        for metric,value in reported[key]['exposure']['sum_metrics'].items():
            if metric in metrics:
                check(math.isclose(metrics[metric],value,rel_tol=1e-12,abs_tol=1e-8),'report_match',key+' exposure/'+metric)

feedback_differences={}
sampling_differences={}
for dataset in ('gsm8k','mathqa'):
    for left,right in [('MIX','KPLUS'),('ROLE','KPLUS'),('ROLE','MIX')]:
        rows=[]
        for a,b in zip(feedback[dataset+'/'+left],feedback[dataset+'/'+right]):
            check(a[0]==b[0], 'feedback_pairing',dataset+'/'+left+'/'+right)
            if a[1:]!=b[1:]:rows.append(a[0])
        feedback_differences[dataset+'/'+left+'_vs_'+right]=dict(requests=512,token_or_termination_differences=len(rows),request_ids=rows)
        sampling_differences[dataset+'/'+left+'_vs_'+right]=dict(presentations=1024,
            different_order_or_anchors=sum(a!=b for a,b in zip(sampling[dataset+'/'+left],sampling[dataset+'/'+right])))

diagnostic_work=Counter()
for dataset in ('gsm8k','mathqa'):
    diagnostic_root=ROOT/'campaign_001/diagnostics'/dataset
    pool=read(diagnostic_root/'pool/POOL.json')
    check(pool['status']=='completed' and pool['completed_preparations']==pool['attempted_preparations']==16
          and pool['unknown_preparations']==pool['unattempted_preparations']==pool['new_generations']==0,
          'diagnostics',dataset+' preparation')
    for field in ('completed_preparations','target_prefill_forwards','target_verification_replay_forwards','terminal_target_forwards'):
        diagnostic_work[field]+=pool[field]
    panels={}
    for arm,event in [('INITIAL',0)]+[(a,e) for a in ('KPLUS','MIX','ROLE') for e in (2,4)]:
        result=read(diagnostic_root/arm/f'event_{event:03d}/RESULT.json')
        check(result['status']=='completed' and result['attempted_responses']==result['returned_responses']==16
              and result['unknown_responses']==result['unattempted_responses']==0
              and result['training_use'] is False and result['rng_restored'] is True and result['model_mode_restored'] is True,
              'diagnostics',dataset+'/'+arm+str(event)+' isolation')
        diagnostic_work['draft_response_forwards']+=result['draft_forwards']
        diagnostic_work['vocabulary_projection_calls']+=result['projection_calls']
        panels[arm,event]=result['snapshots']
    initial=panels['INITIAL',0]
    for arm in ('KPLUS','MIX','ROLE'):
        totals=Counter()
        for request_id,first in initial.items():
            second,last=panels[arm,2][request_id],panels[arm,4][request_id]
            for snapshot in (first,second,last):
                check(snapshot['evaluation_mode'] is True and snapshot['no_backward'] is True,
                      'diagnostics',dataset+'/'+arm+'/'+request_id+' eval/no_backward')
            for field in ('conditioning_id','anchor_ids','input_ids_sha256','loss_mask_sha256','context_length','block_size','eos_token_ids'):
                check(first[field]==second[field]==last[field],'diagnostic_pairing',dataset+'/'+arm+'/'+request_id+'/'+field)
            check(len(first['blocks'])==len(second['blocks'])==len(last['blocks'])==8,'diagnostic_pairing',dataset+'/'+arm+'/'+request_id+' block counts')
            for b0,b2,b4 in zip(first['blocks'],second['blocks'],last['blocks']):
                for field in ('anchor_id','labels','valid','weights'):
                    check(b0[field]==b2[field]==b4[field],'diagnostic_pairing',dataset+'/'+arm+'/'+request_id+'/'+field)
                l0,l2,l4=map(prefix,(b0,b2,b4));n=sum(b0['valid'])
                gain2=max(l2-l0,0);gain4=max(l4-l0,0)
                totals.update(dict(valid_proposals=n,initial_prefix=l0,event2_prefix=l2,event4_prefix=l4,
                                   original_first_error=int(l0<n),event2_broken_prefix=int(l2<l0),event4_broken_prefix=int(l4<l0),
                                   event2_repair_with_prefix_intact=int(l2>l0),event4_repair_with_prefix_intact=int(l4>l0),
                                   event2_gained_prefix_tokens=gain2,event4_gained_prefix_tokens=gain4,
                                   repair_persists=int(l2>l0 and l4>l0),event2_gain_fully_persists=int(gain2>0 and l4>=l2),
                                   persisted_gain_tokens=min(gain2,gain4),lost_gain_tokens=max(gain2-gain4,0)))
        diagnostic_summaries[dataset+'/'+arm]=dict(totals)
        check(dict(totals)==reported_diagnostics[dataset+'/'+arm]['totals'],'report_match',dataset+'/'+arm+' independent diagnostic tally')

check(total_work['online_generations']==3072 and total_work['optimizer_steps']==3072
      and total_work['response_forwards']==total_work['response_backwards']==6144
      and total_work['correctness_generations']==8, 'total', 'production budget')
check(diagnostic_work['completed_preparations']==32 and diagnostic_work['draft_response_forwards']==224,'total','diagnostic budget')
output=dict(status='passed' if not errors else 'failed',verification_status='ANALYZED',checks=sum(checks.values()),
            check_groups=dict(checks),errors=errors,total_work=dict(total_work),diagnostic_work=dict(diagnostic_work),
            trajectories=summaries,feedback_differences=feedback_differences,sampling_differences=sampling_differences,diagnostics=diagnostic_summaries,
            retirement_kinds=dict(retained),source_files=sources,
            limitations=['Saved tensor-equality and RNG-restoration checks are receipts, not independently rerun tensor checks.',
                         'Prepared training tensors and rolling intermediate recoveries were retired under the admitted policy; they are not archived.',
                         'This read-only CPU audit is not a new GPU execution or independent training replication.',
                         'Identical feedback token IDs do not establish identical native hidden features.',
                         'The 128 anchors per dataset are local fixed-context measurements, not an estimator of serving speedup.'])
(HERE/'WORK_AUDIT.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps({k:output[k] for k in ('status','checks','check_groups','errors','total_work','diagnostic_work','feedback_differences')},indent=2))
