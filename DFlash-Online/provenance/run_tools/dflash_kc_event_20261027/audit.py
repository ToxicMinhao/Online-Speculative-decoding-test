"""Independent KC evidence audit: compact files only, no model or remote work."""
import csv
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
SPEC = importlib.util.spec_from_file_location(
    'retained_trace_audit', RESEARCH / 'dflash_ok_continuation_20261027/audit_results.py')
TRACE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRACE)
load, lines, check, compare, near = TRACE.load, TRACE.lines, TRACE.check, TRACE.compare, TRACE.near
ARMS = ('K', 'KC')


def f32(value):
    return struct.unpack('f', struct.pack('f', value))[0]


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def valid_length(labels, weights, eos):
    if labels[0] in eos:
        return 0
    length = 0
    for label, weight in zip(labels[1:], weights[1:]):
        if weight <= 0:
            break
        length += 1
        if label in eos:
            break
    return length


def prefix_length(labels, predictions, available):
    length = 0
    for offset in range(1, available + 1):
        if labels[offset] != predictions[offset]:
            break
        length += 1
    return length


def eos_ids(evidence):
    metadata = RESEARCH / 'dflash_prefix_correction_step_20261027/model_metadata'
    generation = load(metadata / 'target_generation_config.json')
    tokenizer = load(metadata / 'target_tokenizer_config.json')
    ids = generation['eos_token_id']
    result = set(ids if isinstance(ids, list) else [ids])
    token = tokenizer['eos_token']
    token = token['content'] if isinstance(token, dict) else token
    result.update(int(index) for index, value in tokenizer['added_tokens_decoder'].items()
                  if value['content'] == token)
    pin = evidence.read(evidence.raw / 'sealed/MODEL_PIN.json')
    if pin:
        for name in ('generation_config.json', 'tokenizer_config.json'):
            actual = TRACE.identity(metadata / ('target_' + name))
            check(actual['sha256'] == pin['target']['files'][name]['sha256'],
                  evidence.issues, 'Pinned EOS metadata ' + name)
    return result


def score_window(details, step, eos, issues, *, collect_blocks):
    """Reconstruct prefixes, hinge weights and repaired block identities."""
    posts = [('K', details['post_k'])]
    if details['post_c'] is not None:
        posts.append(('KC', details['post_c']))
    all_totals = {name: {} for name, _ in posts}
    table = []
    for response_index, source in enumerate(details['responses']):
        response_totals = {name: {} for name, _ in posts}
        hinge_terms = {name: [] for name, _ in posts}
        transitions = dict(original_first_errors=0, K_useful_repairs=0, C_useful_repairs=0,
            useful_repairs_retained=0, useful_repairs_lost=0, useful_repairs_newly_obtained=0)
        for block, labels in enumerate(source['labels'][0]):
            validity = source['validity_weights'][0][block]
            available = valid_length(labels, validity, eos)
            anchor = source['anchors'][0][block]
            for offset in range(1, available + 1):
                check(labels[offset] == source['input_ids'][0][anchor + offset], issues,
                      f'Step {step} canonical supervised token identity')
            old = prefix_length(labels, source['original_proposals'][0][block], available)
            weights = [0] + [max(old - offset + 1, 0) for offset in range(1, len(labels))]
            check(weights == source['protected_weights'][0][block], issues,
                  f'Step {step} original triangular weights')
            row = dict(step=step, response_id=source['record_id'], block=block,
                anchor=source['anchors'][0][block], anchor_kept=source['keep'][0][block],
                available_positions=available, old_prefix_length=old)
            for name, post in posts:
                saved = post['responses'][response_index]
                predictions = saved['proposals'][0][block]
                margins = saved['protected_margins'][0][block]
                hinge_terms[name].extend(w * max(0, -margin) for w, margin in zip(weights, margins))
                new = prefix_length(labels, predictions, available)
                wrong_ties = sum(w > 0 and predictions[i] != labels[i] and margins[i] == 0
                                 for i, w in enumerate(weights))
                negative = sum(w > 0 and margins[i] < 0 for i, w in enumerate(weights))
                wrong = sum(w > 0 and predictions[i] != labels[i] for i, w in enumerate(weights))
                check(wrong == negative + wrong_ties, issues,
                      f'Step {step} {name} protected errors = negative margins + wrong ties')
                metrics = dict(blocks=1, old_matching_prefix_tokens=old,
                    new_matching_prefix_tokens=new, net_matching_prefix_change=new-old,
                    old_first_errors=int(old < available),
                    first_errors_repaired_with_prefix_intact=int(old < available and new > old),
                    new_earlier_errors=int(new < old), old_prefix_tokens_lost=max(old-new, 0),
                    protected_positions=old, protected_weight=sum(weights),
                    strict_margin_violations=negative,
                    weighted_strict_margin_violations=sum(w for i,w in enumerate(weights) if margins[i] < 0),
                    wrong_at_tie_positions=wrong_ties,
                    weighted_wrong_at_tie_positions=sum(w for i,w in enumerate(weights)
                        if predictions[i] != labels[i] and margins[i] == 0),
                    protected_argmax_errors=wrong)
                check(saved['matching_prefix_lengths'][0][block] == new, issues,
                      f'Step {step} {name} saved matching length')
                for key, value in metrics.items():
                    response_totals[name][key] = response_totals[name].get(key, 0) + value
                row.update({name+'_prefix_length': new,
                    name+'_lost_old_prefix_tokens': max(old-new, 0), name+'_wrong_ties': wrong_ties})
            if len(posts) == 2:
                repaired_k = old < available and row['K_prefix_length'] > old
                repaired_c = old < available and row['KC_prefix_length'] > old
                changes = dict(original_first_errors=int(old < available),
                    K_useful_repairs=int(repaired_k), C_useful_repairs=int(repaired_c),
                    useful_repairs_retained=int(repaired_k and repaired_c),
                    useful_repairs_lost=int(repaired_k and not repaired_c),
                    useful_repairs_newly_obtained=int(not repaired_k and repaired_c))
                for key, value in changes.items():
                    transitions[key] += value
                row.update(changes)
            if collect_blocks:
                table.append(row)
        for name, post in posts:
            saved = post['responses'][response_index]
            check(all(saved[k] == v for k,v in response_totals[name].items()), issues,
                  f'Step {step} {name} per-response counts')
            for key, value in response_totals[name].items():
                all_totals[name][key] = all_totals[name].get(key, 0) + value
            # FP32 chunk reductions need not equal a new FP64 summation exactly.
            # Check finiteness and the exact zero case without creating a new gate.
            saved_hinge = saved['violation_numerator']
            reconstructed_hinge = math.fsum(hinge_terms[name])
            check(math.isfinite(saved_hinge) and saved_hinge >= 0, issues,
                  f'Step {step} {name} finite nonnegative hinge numerator')
            if reconstructed_hinge == 0:
                check(saved_hinge == 0, issues, f'Step {step} {name} exactly empty hinge')
        if len(posts) == 2:
            check(transitions == posts[1][1]['repair_transitions'][response_index], issues,
                  f'Step {step} repair identities')
    for name, post in posts:
        check(all_totals[name] == post['totals'], issues, f'Step {step} {name} totals')
        check(post['denominator'] == all_totals[name]['protected_weight'], issues,
              f'Step {step} {name} global denominator')
        numerator = f32(sum(row['violation_numerator'] for row in post['responses']))
        expected_v = f32(numerator/post['denominator']) if post['denominator'] else numerator
        check(post['numerator'] == numerator and post['V'] == expected_v, issues,
              f'Step {step} {name} FP32 reduction')
    correction = details['correction']
    if correction['status'] == 'applied':
        check(correction['V'] > 0 and correction['gradient_norm_squared_fp32'] > 0, issues,
              f'Step {step} active branch')
        check(correction['coefficient_fp32'] ==
              f32(-correction['V']/correction['gradient_norm_squared_fp32']), issues,
              f'Step {step} prescribed FP32 coefficient')
    else:
        check(correction['realized_correction_norm'] == correction['proposed_correction_norm'] == 0,
              issues, f'Step {step} no-op norm')
    return all_totals, table


def training(evidence, output):
    root = evidence.raw / 'campaign_001'
    event = evidence.read(root / 'KC/events/event_001/RESULT.json')
    report = evidence.read(root / 'CORRECTION_EVIDENCE.json')
    if event is None or report is None:
        return {'status': 'unavailable'}
    ordinary = load(RESEARCH / 'dflash_osd_one_event_20261025/final_review/raw/campaign_001/trainer/events/event_001/RESULT.json')
    result = TRACE.training_exposure(event, 1, 'K', evidence.issues, ordinary)
    result['arm'] = 'KC (unchanged K objective, tested post-K correction)'
    expected_counts = dict(K_backward_attempted=256, K_backward_completed=256, K_backward_failed=0,
        Adam_attempted=128, Adam_completed=128, Adam_failed=0,
        presentations_completed=256, windows_completed=128)
    compare(report['counts'], expected_counts, evidence.issues, 'Exact main counts')
    compare(report['work'], dict(original_response_forwards=256, post_k_response_forwards=256,
        post_c_response_forwards=6, post_k_response_attempted=256,
        post_c_response_attempted=6, violation_backward_failed=0), evidence.issues, 'Exact main forwards')
    check(report['work']['violation_backward_completed'] <= 256 and
          report['work']['direct_corrections'] <= 128, evidence.issues, 'Bounded correction work')
    check([s['step'] for s in report['steps']] == list(range(1,129)), evidence.issues, '128 decisions')
    check([p['step'] for p in report['probes']] == [1,64,128], evidence.issues, 'Only three probes')
    eos = eos_ids(evidence)
    steps, blocks, probes = [], [], []
    summed_work, summed_costs = {}, {}
    for row in report['steps']:
        step = row['step']
        saved = evidence.read(root / f'correction/window_{step:03d}.json')
        if saved is None:
            continue
        check(saved['summary'] == row, evidence.issues, f'Step {step} persisted summary')
        details = saved['details']
        check(row['status'] == 'completed' and row['actual_Adam_steps_completed'] == step and
              row['trainer_step_before_return'] == step-1, evidence.issues, f'Step {step} native accounting')
        for phase, checks in row['integrity'].items():
            check(checks is True or all(checks.values()), evidence.issues, f'Step {step} {phase} integrity')
        presentations = event['records'][(step-1)*2:step*2]
        check([r['record_id'] for r in details['responses']] == [r['record_id'] for r in presentations],
              evidence.issues, f'Step {step} response identity')
        for source, presentation in zip(details['responses'], presentations):
            schedule = presentation['sampled_anchors']
            check(len(schedule) == 1 and source['anchors'] == schedule[0]['positions'] and
                  source['keep'] == schedule[0]['keep_mask'], evidence.issues,
                  f'Step {step} original anchors')
        totals, table = score_window(details, step, eos, evidence.issues, collect_blocks=step in (1,64,128))
        blocks.extend(table)
        steps.append(dict(step=step, **details['correction'], **row['costs'], integrity_s=row['integrity_s'],
                          violation_backwards=details['work']['violation_backward_completed']))
        for key, value in details['work'].items():
            summed_work[key] = summed_work.get(key,0) + value
        for key, value in row['costs'].items():
            summed_costs[key] = summed_costs.get(key,0) + value
        summed_costs['integrity_s'] = summed_costs.get('integrity_s',0) + row['integrity_s']
        if step in (1,64,128):
            persisted_probe = next(p for p in report['probes'] if p['step'] == step)
            check(persisted_probe == {'step': step, **details}, evidence.issues,
                  f'Step {step} probe detail identity')
            deltas = [r['KC_prefix_length']-r['K_prefix_length'] for r in table]
            probes.append(dict(step=step, K=totals['K'], KC=totals['KC'],
                K_V=details['post_k']['V'], KC_V=details['post_c']['V'],
                repair_retention=details['post_c']['repair_totals'],
                improved=sum(x>0 for x in deltas), regressed=sum(x<0 for x in deltas),
                tied=sum(x==0 for x in deltas), added_prefix_tokens=sum(max(x,0) for x in deltas),
                lost_post_K_prefix_tokens=sum(max(-x,0) for x in deltas)))
        if step == 1:
            reference = load(RESEARCH / 'dflash_prefix_correction_step_20261027/raw/execution_001/RESULT.json')
            keys = ('K_update_norm','responses','post_k','correction','post_c')
            check({key:details[key] for key in keys} == {key:reference['core'][key] for key in keys},
                  evidence.issues, 'Entire retained first-window evidence reproduces exactly')
    check(summed_work == report['work'], evidence.issues, 'Window work sums')
    check(all(near(value, report['costs'][key]) for key,value in summed_costs.items()),
          evidence.issues, 'Window nested cost sums')
    write_csv(output / 'CORRECTION_STEPS.csv', steps)
    write_csv(output / 'PROBE_BLOCKS.csv', blocks)
    result.update(correction_steps=len(steps), active=sum(r['status']=='applied' for r in steps),
        work=report['work'], counts=report['counts'], correction_costs=report['costs'],
        probes=probes, event_costs=event['costs'], eos_token_ids=sorted(eos))
    return result


def evaluation(evidence):
    root = evidence.raw / 'campaign_001/evaluation'
    requests = evidence.read(root / 'REQUESTS.json')
    protocol = evidence.read(root / 'PROTOCOL.json')
    if requests is None or protocol is None:
        return {'status':'unavailable'}
    original=load(RESEARCH/'dflash_osd_one_event_20261025/sealed/MANIFEST.json')['eval_requests']
    check(requests == [dict(r,panel_ordinal=r.get('panel_ordinal',i)) for i,r in enumerate(original)],evidence.issues,'Exact original64 requests/canonicalF')
    checkpoints=protocol['checkpoints']
    retained_k=load(RESEARCH/'dflash_first_error_execution_20261026/raw/campaign_001/K/events/event_001/RESULT.json')
    trained_kc=evidence.read(evidence.raw/'campaign_001/KC/events/event_001/RESULT.json')
    if trained_kc:
        check(checkpoints=={'K':retained_k['checkpoint_path'],'KC':trained_kc['checkpoint_path']},
              evidence.issues,'Exact retained K1/new KC export identities')
    expected=[(r,n,a) for r in requests for n in range(3)
              for a in (ARMS if (r['panel_ordinal']+n)%2==0 else ARMS[::-1])]
    rows=lines(root/'MEASUREMENTS.jsonl')
    check(len(rows)==384 and [(r['request_id'],r['repeat'],r['method']) for r in rows] ==
          [(r['request_id'],n,a) for r,n,a in expected],evidence.issues,'Exact384 alternating slots')
    valid_rows=[]
    for serial,(row,(request,repeat,arm)) in enumerate(zip(rows,expected),1):
        label=f"{request['panel_ordinal']:04d}_r{repeat}_{arm}"
        record=evidence.read(root/(label+'.json'))
        if record is None:continue
        before=len(evidence.issues)
        counts=TRACE.native_record(record,request,checkpoints[arm],capture=False,
            session='prefix_correction_evaluation',label=label,index=request['panel_ordinal'],issues=evidence.issues)
        TRACE.bind_native(evidence,record,serial)
        compare(row,dict(**counts,label=label,validation_status='valid',valid_measurement=True,
                         structural_errors=[],errors=[],finish_reason=request['finish_reason']),evidence.issues,label+' row')
        for name in ('request_latency_s','native_generate_s'):
            check(row[name]==record['metadata'][name],evidence.issues,label+' '+name)
        if len(evidence.issues)==before:valid_rows.append(row)
    summary=evidence.read(root/'report/SUMMARY.json')
    cohort=[]
    for request in requests:
        own=[r for r in valid_rows if r['request_id']==request['request_id']]
        complete=len(own)==6 and {(r['method'],r['repeat']) for r in own}=={(a,n) for a in ARMS for n in range(3)}
        consistent=all(len({tuple(r[k] for k in TRACE.COUNTS) for r in own if r['method']==a})<=1 for a in ARMS)
        check(consistent,evidence.issues,request['request_id']+' repeat counts')
        if complete and consistent:cohort.extend(own)
    totals={}
    for arm in ARMS:
        own=[r for r in cohort if r['method']==arm]
        totals[arm]={k:sum(r[k] for r in own if r['repeat']==0) for k in TRACE.COUNTS}
        totals[arm]['summed_mean_request_latency_s']=sum(r['request_latency_s'] for r in own)/3
        if summary:
            for name,value in totals[arm].items():check(near(value,summary[arm][name]),evidence.issues,'Pooled '+arm+'/'+name)
    pairs=[(next(r for r in cohort if r['request_id']==req['request_id'] and r['method']=='K' and r['repeat']==0),
            next(r for r in cohort if r['request_id']==req['request_id'] and r['method']=='KC' and r['repeat']==0))
           for req in requests if sum(r['request_id']==req['request_id'] for r in cohort)==6]
    directions={name:sum((k['verification_calls']>g['verification_calls'] if name=='improved' else
                         k['verification_calls']<g['verification_calls'] if name=='regressed' else
                         k['verification_calls']==g['verification_calls']) for k,g in pairs)
                for name in ('improved','regressed','tied')}
    ratios={name:100*(1-totals['KC'][key]/totals['K'][key]) if totals['K'][key] else None
            for name,key in [('call_reduction_pct','verification_calls'),('request_time_reduction_pct','summed_mean_request_latency_s')]}
    if summary:
        for name,value in ratios.items():check(near(value,summary['comparisons']['KC_vs_K'][name]),evidence.issues,'Pooled ratio '+name)
        check(directions==summary['comparisons']['KC_vs_K']['call_request_directions'],evidence.issues,'Request directions')
    activations = lines(root / 'ACTIVATIONS.jsonl')
    last = None
    expected_activation = []
    for request,repeat,arm in expected:
        if arm != last:
            expected_activation.append((request['panel_ordinal'], repeat, arm))
            last = arm
    check([(r['panel_ordinal'],r['repeat'],r['method']) for r in activations]==expected_activation,evidence.issues,'Publication schedule')
    for row in activations:
        check(not row['errors'],evidence.issues,'Publication errors')
        TRACE.publication(row,checkpoints[row['method']],evidence.issues,'Publication')
    for arm in ARMS:
        record=evidence.read(root/f'WEIGHTS_{arm}.json')
        if record:TRACE.tensor_equality(record['native_reply'],checkpoints[arm],evidence.issues,arm+' weights')
    status=evidence.read(root/'STATUS.json')
    if status:
        compare(status,dict(status='completed',planned_generations=384,attempted_generations=384,
            returned_generations=384,completed_valid_generations=384,invalid_generations=0,
            unknown_generation_attempts=0,unattempted_generations=0,optimizer_steps=0,feedback_collections=0),evidence.issues,'Evaluation accounting')
        check(near(status['activation_s'],sum(r['activation_s'] for r in activations)),evidence.issues,'Activation cost once')
    return dict(valid_requests=len(pairs),returned=len(rows),native_blocks=evidence.native_blocks,
                totals=totals,comparisons=ratios,directions=directions,publications=len(activations),execution=status)


def audit(root=HERE):
    root = Path(root)
    output = root / 'analysis'
    output.mkdir(exist_ok=True)
    evidence = TRACE.Evidence(root / 'raw')
    raw = evidence.raw
    retrieval = evidence.read(raw / 'RETRIEVAL_MANIFEST.json')
    if retrieval:
        for name, expected in retrieval.items():
            path = raw / name
            if not path.is_file():
                evidence.missing.append(str(path))
                continue
            actual = TRACE.identity(path)
            check(actual['bytes'] == expected.get('bytes', expected.get('size')) and
                  actual['sha256'] == expected['sha256'], evidence.issues, 'Retrieval ' + name)
    seal = evidence.read(raw / 'sealed/SEAL.json')
    plan = evidence.read(raw / 'sealed/PLAN.json')
    if seal and plan:
        bindings = {
            plan['ordinary_receipt']: RESEARCH/'dflash_osd_one_event_20261025/final_review/raw/campaign_001/trainer/events/event_001/RESULT.json',
            plan['manifest']: RESEARCH/'dflash_osd_one_event_20261025/sealed/MANIFEST.json',
            plan['k_receipt']: RESEARCH/'dflash_first_error_execution_20261026/raw/campaign_001/K/events/event_001/RESULT.json',
            plan['reference_diagnostic']: RESEARCH/'dflash_prefix_correction_step_20261027/raw/execution_001/RESULT.json',
        }
        for remote, local in bindings.items():
            check(TRACE.identity(local) == seal['files'][remote], evidence.issues,
                  'Original input binding ' + remote)
    for name in ('PREFLIGHT.json', 'POSTFLIGHT.json'):
        receipt = evidence.read(raw / name)
        if receipt and seal:
            check(receipt['status'] == 'passed' and receipt['files_checked'] == len(seal['files']),
                  evidence.issues, name + ' source integrity')
    status = evidence.read(raw / 'campaign_001/STATUS.json')
    costs = evidence.read(raw / 'campaign_001/COSTS.json')
    completed = status and status['status'] == 'completed'
    if completed:
        compare(status, dict(main_optimizer_steps=128, main_presentations=256,
            native_adam_steps_completed=128, native_K_presentations_completed=256,
            new_qualification_optimizer_steps=0, new_qualification_presentations=0,
            new_feedback_generations=0), evidence.issues, 'Main budget')
        first = evidence.read(raw / 'campaign_001/FIRST_WINDOW_REPRODUCTION.json')
        check(first and first['exact_equal'], evidence.issues, 'First-window gate passed')
        for name in ('INITIAL_VS_PIN', 'FRESH_CORE'):
            receipt = evidence.read(raw / f'campaign_001/initialization/{name}.json')
            check(receipt and receipt['exact_equal'], evidence.issues, 'Exact initialization ' + name)
    if costs:
        check(costs['new_qualification_optimizer_steps'] == 0, evidence.issues, 'No extra qualification steps')
        if 'startup' in costs:
            compare(costs['startup'], dict(status='passed', optimizer_steps=0, response_evaluations=2,
                backwards=2, exact_initial_state_restored=True), evidence.issues, 'Separate startup')
        if 'reference_diagnostic' in costs:
            check(costs['reference_diagnostic']['historical_only'] and
                  not costs['reference_diagnostic']['included_in_current_campaign_wall'],
                  evidence.issues, 'Prior diagnostic not recharged')
    train = evidence.section('Training', lambda: training(evidence, output)) if completed else {}
    serving = evidence.section('Evaluation', lambda: evaluation(evidence)) if completed else {}
    if costs and train.get('event_costs'):
        check(train['event_costs'] == costs['event'], evidence.issues, 'Event costs match')
        nested = train['correction_costs']
        check(sum(nested.get(k, 0) for k in ('post_k_s','correction_s','post_c_s','integrity_s'))
              <= costs['event']['training_s'] <= costs['event_wall_s'] <= costs['actual_campaign_wall_s'],
              evidence.issues, 'Nested adaptation costs within training/event/campaign')
    result = dict(status=('passed' if completed and not evidence.issues and not evidence.missing
                          else 'failed_or_incomplete'), experiment_status=status,
        issues=evidence.issues, missing=evidence.missing, verified_retrieved_files=len(retrieval or {}),
        training=train, evaluation=serving, costs=costs,
        scope='Independent compact-evidence reconstruction only; no model execution or new experiment.',
        interpretation='Reused development panel. Probes at 64/128 share current KC pre-step state; they are not independently trained K arms.')
    (root / 'FINAL_AUDIT.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


if __name__ == '__main__':
    result = audit(sys.argv[1] if len(sys.argv) > 1 else HERE)
    print(json.dumps({k:v for k,v in result.items() if k not in ('training','evaluation','costs')}, indent=2))
    sys.exit(0 if result['status'] == 'passed' else 1)
