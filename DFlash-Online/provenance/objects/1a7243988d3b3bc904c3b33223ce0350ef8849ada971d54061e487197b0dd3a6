"""Independent local compact-evidence audit of O/K continuation events 2--4.

Standard library only. No Torch/model imports, network, inference, checkpoint
reload or raw-evidence mutation. Missing/partial evidence cannot pass as complete.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ARMS = ('O', 'K')
OBJECTIVES = {'O': 'osd', 'K': 'osd_first_error'}
COUNTS = ('verification_calls', 'accepted_draft_tokens', 'committed_progress', 'output_tokens')
HIST = tuple(f'first_error_position_{n}' for n in range(1, 16))
SUMS = ('available_position_occurrences', 'selected_position_occurrences',
        'available_position_weight', 'selected_position_weight', 'no_error_blocks',
        'empty_blocks', 'main_loss_numerator', 'loss_denominator',
        'response_presentations', 'supervised_position_exposure') + HIST


def load(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def identity(path):
    path = Path(path)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(bytes=path.stat().st_size, sha256=digest)


def check(condition, issues, message):
    if not condition:
        issues.append(message)


def near(actual, expected, *, rel=1e-12, absolute=1e-9):
    return (type(actual) in (int, float) and type(expected) in (int, float)
            and math.isfinite(actual) and math.isfinite(expected)
            and math.isclose(actual, expected, rel_tol=rel, abs_tol=absolute))


def compare(actual, expected, issues, label):
    for key, value in expected.items():
        check(actual.get(key) == value, issues, label + ': ' + key)


class Evidence:
    def __init__(self, raw):
        self.raw = Path(raw)
        self.issues, self.missing, self.used_native = [], [], set()
        self.native_blocks = 0

    def read(self, path):
        path = Path(path)
        if not path.is_file():
            self.missing.append(str(path))
            return None
        return load(path)

    def section(self, label, operation):
        try:
            return operation()
        except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError, ZeroDivisionError) as error:
            self.issues.append(label + ': ' + type(error).__name__ + ': ' + str(error))
            return {'status': 'malformed_or_incomplete'}


def native_record(record, request, checkpoint, *, capture, session, label, index, issues):
    """Reconstruct greedy first-mismatch commits, EOS clipping and all counters."""
    meta, response = record['metadata'], record['response_ids']
    blocks, observed = meta['trace'], meta['capture']
    compare(record, dict(record_id=label, session_id=session, request_index=index,
        checkpoint_id=checkpoint, purpose='serving' if capture else 'evaluation',
        prompt_ids=request['prompt_ids'], max_new_tokens=2048,
        stop_token_ids=request['stop_token_ids']), issues, label)
    if not capture:
        compare(record, dict(response_ids=request['response_ids'], finish_reason=request['finish_reason']), issues, label + ' retained F')
    check(isinstance(response, list) and bool(response) and len(response) <= 2048
          and all(type(token) is int and token >= 0 for token in response), issues, label + ': response tokens')
    stops = [i for i, token in enumerate(response) if token in request['stop_token_ids']]
    check((stops == [len(response)-1] and record['finish_reason'] == 'eos') or
          (not stops and len(response) == 2048 and record['finish_reason'] == 'length'), issues, label + ': termination')
    compare(meta, dict(block_size=16, context_limit=4096, runtime='pinned_SGLang_FA3',
        cache_policy='idle_full_flush_before_every_request_and_publication',
        feedback_capture_in_timed_request=capture), issues, label + ' metadata')
    compare(observed, dict(status='passed', prompt_ids=request['prompt_ids'], response_ids=response,
        target_argmax_ids=response, blocks=blocks, covered_token_ids=request['prompt_ids'] + response[:-1],
        covered_rows=len(request['prompt_ids'])+len(response)-1, block_size=16,
        native_capture_validated=True, collect_features=capture,
        label_binding_scope='native_prefill_and_committed_verification',
        final_feature_logits_used_for_labels=False), issues, label + ' capture')
    if capture:
        check(observed['feature_bytes'] > 0 and bool(observed['feature_path']), issues, label + ': persisted feedback')
    else:
        check(observed['feature_bytes'] == observed['feature_persistence_s'] == 0
              and not observed.get('feature_path'), issues, label + ': capture-disabled evaluation')
    offset, accepted = 0, 0
    for number, block in enumerate(blocks):
        scope = label + f': block{number}'
        drafts, teacher = block['draft_tokens'], block['target_argmax']
        check(block['output_start'] == offset and block['anchor_token'] == response[offset], issues, scope + ': known seed')
        check(len(drafts) == 15 and len(teacher) == block['verify_size'] == 16, issues, scope + ': block geometry')
        first_error = 0
        while first_error < 15 and drafts[first_error] == teacher[first_error]:
            first_error += 1
        raw = drafts[:first_error] + [teacher[first_error]]
        committed = raw[:max(2048-1-offset, 0)]
        for position, token in enumerate(committed):
            if token in request['stop_token_ids']:
                committed = committed[:position+1]
                break
        progress = len(committed)
        compare(block, dict(raw_accepted_draft_tokens_before_stop=first_error,
            committed_tokens=committed, committed_progress=progress,
            accepted_draft_tokens=min(first_error, progress),
            native_accepted_draft_tokens=max(progress-1, 0),
            positions=list(range(len(request['prompt_ids'])+offset, len(request['prompt_ids'])+offset+16)),
            stop_clipped=progress < first_error+1, finished=number == len(blocks)-1), issues, scope)
        check(response[offset+1:offset+1+progress] == committed, issues, scope + ': canonical commit')
        offset += progress
        accepted += min(first_error, progress)
    compare(meta, dict(committed_progress=offset, accepted_draft_tokens=accepted,
                       verification_calls=len(blocks)), issues, label + ' summed native counters')
    check(offset == len(response)-1, issues, label + ': complete committed coverage')
    for name in ('request_latency_s', 'native_generate_s'):
        check(type(meta[name]) in (int, float) and math.isfinite(meta[name]) and meta[name] > 0, issues, label + ': finite ' + name)
    return dict(verification_calls=len(blocks), accepted_draft_tokens=accepted,
                committed_progress=offset, output_tokens=len(response))


def bind_native(evidence, record, expected_serial):
    name = Path(record['metadata']['native_trace_path']).name
    check(name == f'request_{expected_serial:06d}.json', evidence.issues, 'Native serial schedule: ' + name)
    check(name not in evidence.used_native, evidence.issues, 'Duplicate native generation: ' + name)
    evidence.used_native.add(name)
    raw = evidence.read(evidence.raw/'campaign_001/native'/name)
    if raw is not None:
        check(raw == record['metadata']['capture'], evidence.issues, 'Native/wrapper capture disagreement: ' + name)
    evidence.native_blocks += len(record['metadata']['trace'])


def training_exposure(receipt, event, arm, issues, ordinary=None):
    label = f'event{event}/{arm}'
    compare(receipt, dict(status='completed', event_id=f'event_{event:03d}', objective=OBJECTIVES[arm],
        passes=2, response_presentations=256, optimizer_steps=128, start_step=(event-1)*128,
        end_step=event*128, reduction='response_mean', gradient_denominator='responses per accumulation window'), issues, label)
    features, rows = receipt['features'], receipt['records']
    by_id = {r['record_id']: r for r in features}
    check(len(features) == len(by_id) == 128 and len(rows) == 256, issues, label + ': event size')
    for pass_number in (0, 1):
        subset = rows[pass_number*128:(pass_number+1)*128]
        check(sorted(r['record_id'] for r in subset) == sorted(by_id)
              and all(r['pass'] == pass_number for r in subset), issues, label + ': complete response pass')
    if ordinary is not None:
        check(features == ordinary['features'], issues, label + ': same feature metadata')
    slots = anchors = available = 0
    total_weight, pending = 0., []
    for number, row in enumerate(rows):
        scope = label + f'/presentation{number}'
        compare(row, dict(optimizer_stepped=number % 2 == 1,
            optimizer_step_after=(event-1)*128+(number+1)//2), issues, scope)
        check(row['learning_rate'] == (3e-4 if row['optimizer_stepped'] else None), issues, scope + ': unchanged constant learning rate')
        if ordinary is not None:
            check(all(row[k] == ordinary['records'][number][k] for k in
                ('record_id','pass','sampled_anchors','optimizer_stepped','optimizer_step_after','learning_rate')),
                issues, scope + ': matched normal sampling/accumulation/LR')
        feature, nslots, nanchors, count, weight = by_id[row['record_id']], 0, 0, 0, 0.
        check(len(row['sampled_anchors']) == 1, issues, scope + ': one current-forward anchor draw')
        for draw in row['sampled_anchors']:
            check(len(draw['positions']) == len(draw['keep_mask']) == 1, issues, scope + ': one response')
            for positions, mask in zip(draw['positions'], draw['keep_mask']):
                check(len(positions) == len(mask), issues, scope + ': mask shape')
                nslots += len(positions)
                for anchor, keep in zip(positions, mask):
                    check(type(anchor) is int and type(keep) is bool, issues, scope + ': anchor types')
                    if keep:
                        check(feature['prompt_length'] <= anchor < feature['length']-1, issues, scope + ': eligible anchor')
                        nanchors += 1
                        offsets = [j for j in range(1,16) if feature['prompt_length'] <= anchor+j < feature['length']]
                        count += len(offsets)
                        weight += sum(math.exp(-(j-1)/7.) for j in offsets)
        slots += nslots; anchors += nanchors; available += count; total_weight += weight
        pending.append((nslots, count, weight))
        metrics = row['metrics']
        check(all(type(v) in (int,float) and math.isfinite(v) for v in metrics.values()), issues, scope + ': finite metrics')
        check(metrics['response_presentations'] == len(pending), issues, scope + ': response count')
        if arm == 'K':
            compare(row, dict(metrics_scope='accumulation_window' if number % 2 else 'presentation'), issues, scope)
            check(metrics['available_position_occurrences'] == sum(p[1] for p in pending), issues, scope + ': available occurrence count')
            check(near(metrics['available_position_weight'],sum(p[2] for p in pending),rel=1.3e-6,absolute=1e-5), issues, scope + ': exponential weights')
            check(sum(metrics[name] for name in HIST)+metrics['no_error_blocks']+metrics['empty_blocks'] == sum(p[0] for p in pending), issues, scope + ': histogram partition')
            for name in ('available_position_occurrences','selected_position_occurrences','no_error_blocks','empty_blocks')+HIST:
                check(metrics[name] >= 0 and metrics[name] == int(metrics[name]), issues, scope + ': integral ' + name)
            chosen, weighted = metrics['selected_position_occurrences'], metrics['selected_position_weight']
            check(0 <= chosen <= metrics['available_position_occurrences'] and chosen == metrics['supervised_position_exposure']
                  and 0 <= weighted <= metrics['available_position_weight'], issues, scope + ': selection bounds')
            error_positions = sum(i*metrics[name] for i,name in enumerate(HIST,1))
            check(error_positions+metrics['no_error_blocks'] <= chosen <= error_positions+15*metrics['no_error_blocks'], issues, scope + ': includes first error')
            check(near(metrics['loss_denominator'],weighted+len(pending)*1e-6,rel=2.4e-7,absolute=2e-6), issues, scope + ': one epsilon per response')
            ratio = metrics['main_loss_numerator']/metrics['loss_denominator']
            check(all(near(metrics[name],ratio,rel=2.4e-7,absolute=2e-7) for name in ('objective_loss','forward_kl')), issues, scope + ': numerator/denominator ratio')
        else:
            check(metrics['supervised_position_exposure'] == sum(p[1] for p in pending), issues, scope + ': O all valid exposure')
        if row['optimizer_stepped']:
            pending = []
    check(not pending, issues, label + ': no incomplete accumulation')
    boundary = [row for row in rows if row['optimizer_stepped']]
    if arm == 'K':
        totals = {name:sum(row['metrics'][name] for row in boundary) for name in SUMS}
        check(receipt['selection_totals_source'] == 'optimizer_boundary_accumulation_windows_once'
              and receipt['selection_totals'] == totals, issues, label + ': boundary-only logged totals')
    else:
        totals = dict(selected_position_occurrences=available, selected_position_weight=total_weight)
    check(len(boundary) == 128 and sum(row['metrics']['response_presentations'] for row in boundary) == 256, issues, label + ': accumulation count')
    work = receipt['objective_work']
    check(all(work[k] == 0 for k in ('reference_setup_s','reference_forward_s','reference_head_s','projection_s')), issues, label + ': no reference/auxiliary work')
    check(all(type(v) in (int,float) and math.isfinite(v) and v >= 0 for v in work.values()), issues, label + ': finite objective costs')
    return dict(status=receipt['status'], sampled_slots=slots, kept_anchors=anchors,
        available_occurrences=available, available_weight=total_weight, selection=totals,
        objective_work=work, optimizer_steps=128, presentations=256)


def resume_receipts(evidence, event, arm, expected_parent, seal):
    root = evidence.raw/f'campaign_001/resume_{event:03d}_{arm}'
    saved, live, closed, checked = [evidence.read(root/name) for name in ('SAVED.json','LIVE.json','CLOSED.json','EVENT_CHECK.json')]
    label = f'event{event}/{arm}'
    if any(value is None for value in (saved,live,closed,checked)):
        return dict(status='incomplete', saved=saved, live=live, closed=closed)
    compare(saved, dict(status='passed', completed_event=event-1, objective=OBJECTIVES[arm],
        global_step=(event-1)*128, presentations=(event-1)*256, horizon=7424,
        draft_tensors=58, optimizer_parameters=58, adam_step=(event-1)*128,
        cuda_rng_states=1, scientific_extra_steps=0, export_exact=True,
        state_path=expected_parent['state'], export_path=expected_parent['export']), evidence.issues, label + ' saved')
    if event == 2:
        check(saved['state_identity'] == seal['files'][expected_parent['state']], evidence.issues, label + ': initial sealed full state')
    compare(live, dict(status='passed', exact_resume=True, mismatches=[], mismatch_count=0,
        global_step=(event-1)*128, presentations=(event-1)*256, no_rng_reset=True,
        scientific_extra_steps=0), evidence.issues, label + ' exact live restore')
    compare(closed, dict(closed=True, failed=False, global_step=event*128, presentations=event*256,
        completed_events=[f'event_{n:03d}' for n in range(1,event+1)], objective=OBJECTIVES[arm],
        reduction='response_mean', accumulation_remainder=0, horizon=7424), evidence.issues, label + ' closed')
    compare(checked, dict(status='passed', event_id=f'event_{event:03d}', objective=OBJECTIVES[arm],
        start_step=(event-1)*128, end_step=event*128, optimizer_steps=128,
        response_presentations=256, passes=2, reduction='response_mean',
        schedule_matched=arm=='K', anchor_draws_checked=256, scientific_extra_steps=0), evidence.issues, label + ' event gate')
    costs = evidence.read(root/'COSTS.json')
    if costs is not None:
        check(all(type(costs[k]) in (int,float) and math.isfinite(costs[k]) and costs[k] >= 0 for k in
            ('loading_and_resume_validation_s','event_wall_s','loading_peak_allocated_bytes',
             'loading_peak_reserved_bytes','peak_allocated_bytes','peak_reserved_bytes')), evidence.issues, label + ': costs')
    return dict(status='passed', saved=saved, live=live, closed=closed, costs=costs,
        evidence_limit='Exact tensor/Adam/RNG equality is supported by sealed runtime receipts, not a local tensor replay.')


def publication(receipt, checkpoint, issues, label):
    native = receipt['native_status']
    published = native['publication']
    compare(receipt, dict(checkpoint_id=checkpoint, checkpoint_structure_valid=True), issues, label)
    compare(native, dict(status='passed', poisoned=None, capture_pending=False, cache_tokens=0,
                         attention_backend='fa3', block_size=16), issues, label + ' native')
    compare(published, dict(status='passed', checkpoint_path=checkpoint, cache_after_tokens=0,
        helper_matches_current_layers=True, helper_norm_matches_current_layers=True,
        fused_enabled=True), issues, label + ' publication')


def tensor_equality(receipt, checkpoint, issues, label):
    compare(receipt, dict(status='passed', all_equal=True, checkpoint_path=checkpoint,
        tensors=58, source_tensor_count=58, native_tensor_count=43,
        helper_matches_current_layers=True, helper_norm_matches_current_layers=True,
        fused_enabled=True), issues, label)


def storage_receipt(path, evidence, *, rows, first=None):
    row = evidence.read(path)
    if row is None:
        return
    compare(row, dict(rows=rows, admitted=True, tensor_margin=.10,
        compact_allowance_bytes=2*2**30, work_bytes=2*2**30, reserve_bytes=8*2**30), evidence.issues, str(path))
    if first is not None:
        compare(row, dict(new_states=2 if first else 0, exports=2,
                          overlap_states=1 if first else 2), evidence.issues, str(path))
    tensors = rows*(4096*6*2+12)+(3*(row['new_states']+row['overlap_states'])+row['exports'])*2097259104
    check(row['tensor_bytes'] == tensors and row['required_bytes'] == int(tensors*1.1)+12*2**30
          and row['free_bytes'] >= row['required_bytes'], evidence.issues, str(path) + ': peak admission')


def feedback_pool(evidence, window, checkpoint, serial_start):
    event = window['event_index']
    root = evidence.raw/'campaign_001'/window['event_id']
    requests = window['requests']
    pool, prepared = evidence.read(root/'POOL.json'), evidence.read(root/'PREPARATIONS.json')
    public = evidence.read(root/'PUBLICATION.json')
    if public is not None:
        check(public['errors'] == [], evidence.issues, f'event{event}: feedback publication errors')
        publication(public['activation'], checkpoint, evidence.issues, f'event{event} feedback')
        tensor_equality(public['equality'], checkpoint, evidence.issues, f'event{event} feedback loaded weights')
    storage_receipt(root/'collection_STORAGE.json', evidence,
                    rows=sum(len(r['prompt_ids'])+2048 for r in requests))
    records = []
    paths = sorted((root/'responses').glob('*.json'))
    expected_names = [r['request_id']+'.json' for r in requests]
    check([p.name for p in paths] == expected_names[:len(paths)], evidence.issues, f'event{event}: chronological response file prefix')
    for position, path in enumerate(paths):
        request, record = requests[position], load(path)
        native_record(record,request,checkpoint,capture=True,session=window['event_id'],
            label=request['request_id'],index=request['request_index'],issues=evidence.issues)
        check(len(record['prompt_ids'])+len(record['response_ids']) <= 2272
              and len(record['response_ids']) >= 2, evidence.issues, f'event{event}: eligible complete response')
        bind_native(evidence,record,serial_start+position)
        records.append(record)
    if pool is None or prepared is None:
        return dict(status='incomplete', responses=len(records), preparation_receipts=len(prepared or []))
    compare(pool, dict(event_id=window['event_id'],checkpoint=checkpoint,immutable=True,
        response_generations=128,teacher_terminal_forwards=128,
        record_ids=[r['request_id'] for r in requests]), evidence.issues, f'event{event} pool')
    check(len(records) == len(prepared) == len(pool['features']) == len(pool['identities']) == 128,
          evidence.issues, f'event{event}: exactly128 shared features and responses')
    actual_rows = sum(len(r['prompt_ids'])+len(r['response_ids']) for r in records)
    storage_receipt(root/'preparation_training_STORAGE.json',evidence,rows=actual_rows,first=event==2)
    terminal_forwards = 0
    for position,(record,prep) in enumerate(zip(records,prepared)):
        ids = record['prompt_ids']+record['response_ids']
        label = f'event{event}/'+record['record_id']
        compare(prep,dict(status='prepared',record_id=record['record_id'],input_ids=ids,
            prompt_length=len(record['prompt_ids']),response_length=len(record['response_ids']),
            target_argmax_ids=record['response_ids'],label_mismatches=[],target_layer_ids=[1,9,17,25,33],
            feature_shape=[1,len(ids),20480],capture_backend='same_frozen_SGLang_native_capture',
            native_capture_validated=True,label_binding_scope='native_prefill_and_committed_verification',
            final_feature_logits_used_for_labels=False,response_only_loss=True,
            supervised_positions=len(record['response_ids']),feature_path=pool['features'][position]), evidence.issues,label+' preparation')
        check(Path(prep['feature_path']).name == record['record_id']+'.pt', evidence.issues,label+': feature basename')
        header=evidence.read(root/'features'/(record['record_id']+'.json'))
        check(header == prep,evidence.issues,label+': durable preparation header')
        terminal=prep['terminal_feature']
        compare(terminal,dict(status='passed',input_ids=ids,positions=[len(ids)-1],target_forward_count=1,
            forward_input_rows=len(ids),retained_feature_rows=1,final_feature_logits_used_for_labels=False,
            capture_backend='same_frozen_SGLang_full_context_final_row'),evidence.issues,label+' terminal feature')
        terminal_file=evidence.read(root/'features'/(record['record_id']+'.terminal.json'))
        check(terminal_file == terminal,evidence.issues,label+': durable terminal receipt')
        terminal_forwards += terminal['target_forward_count']
        info=pool['identities'][prep['feature_path']]
        check(type(info['bytes']) is int and info['bytes'] > 0 and re.fullmatch(r'[0-9a-f]{64}',info['sha256']) is not None,
              evidence.issues,label+': feature identity declaration')
    check(len(list((root/'features').glob('*.terminal.json'))) == 128, evidence.issues,f'event{event}: terminal forward receipt count')
    costs=evidence.read(root/'COSTS.json')
    if costs is not None:
        for key in ('teacher_preparation_s','feature_persistence_s'):
            check(near(costs[key],sum(p[key] for p in prepared)),evidence.issues,f'event{event}: preparation cost '+key)
        check(near(costs['request_latency_s'],sum(r['metadata']['request_latency_s'] for r in records)),evidence.issues,f'event{event}: feedback latency sum')
        for key,value in costs['capture_nested'].items():
            check(near(value,sum(r['metadata']['capture'][key] for r in records)),evidence.issues,f'event{event}: nested capture cost '+key)
    return dict(status='complete',responses=len(records),terminal_forwards=terminal_forwards,
        sequence_rows=actual_rows,pool=pool,costs=costs,
        feature_metadata=[dict(record_id=r['record_id'],length=len(r['prompt_ids'])+len(r['response_ids']),
            prompt_length=len(r['prompt_ids']),response_tokens=len(r['response_ids']),
            adjacent_response_positions=len(r['response_ids'])-1,path=pool['features'][i]) for i,r in enumerate(records)],
        evidence_limit='Feature identities and canonical-label headers checked; bulky tensors are not reloaded locally. Shared bytes are enforced by the sealed driver before and after each arm.')


def slots(requests):
    return [(r,repeat,arm) for r in requests for repeat in range(3)
            for arm in (ARMS if (r['panel_ordinal']+repeat)%2==0 else ARMS[::-1])]


def evaluation(evidence,event,plan,checkpoints,serial_start):
    root=evidence.raw/f'campaign_001/evaluation_{event:03d}'
    status=evidence.read(root/'STATUS.json')
    path=root/'MEASUREMENTS.jsonl'
    if not path.is_file():
        evidence.missing.append(str(path))
        return dict(status='incomplete',returned=0)
    rows=lines(path)
    requests=[dict(r,panel_ordinal=r.get('panel_ordinal',ordinal)) for ordinal,r in enumerate(plan['eval_requests'])]
    expected=slots(requests)
    check(len(rows)<=384 and [(r['request_id'],r['repeat'],r['method']) for r in rows] ==
          [(r['request_id'],repeat,arm) for r,repeat,arm in expected[:len(rows)]],evidence.issues,f'event{event}: exact alternating evaluation prefix')
    protocol=evidence.read(root/'PROTOCOL.json')
    if protocol is not None:
        compare(protocol,dict(arms=['O','K'],comparisons=[['K','O']],repeats=3,checkpoints=checkpoints,
            rotation='(panel_ordinal + repeat) % 2',count_trajectory_repeat=0,historical_count_gate=False,
            capture=False,extra_warmup_generations=0),evidence.issues,f'event{event} protocol')
    check(evidence.read(root/'FROZEN_COUNTS.json') == plan['frozen_counts'],evidence.issues,f'event{event}: historical F counts')
    for position,row in enumerate(rows):
        request,repeat,arm=expected[position]
        label=f"{request['panel_ordinal']:04d}_r{repeat}_{arm}"
        check(Path(row['record_path']).name == label+'.json',evidence.issues,label+': raw basename')
        record=evidence.read(root/(label+'.json'))
        if record is None:continue
        counts=native_record(record,request,checkpoints[arm],capture=False,session='ok_continuation_evaluation',
            label=label,index=request['panel_ordinal'],issues=evidence.issues)
        bind_native(evidence,record,serial_start+position)
        compare(row,dict(**counts,label=label,panel_ordinal=request['panel_ordinal'],
            original_index=request['original_index'],validation_status='valid',valid_measurement=True,
            structural_errors=[],errors=[],finish_reason=request['finish_reason'],
            cap_hit=request['finish_reason']=='length',prompt_tokens=len(request['prompt_ids'])),evidence.issues,label+' measurement')
        for key in ('request_latency_s','native_generate_s'):
            check(near(row[key],record['metadata'][key]),evidence.issues,label+': '+key)
        check(near(row['cache_maintenance_s'],record['metadata']['capture']['cache_maintenance_s']),evidence.issues,label+': cache timer')
        validation=evidence.read(root/(label+'_VALIDATION.json'))
        if validation is not None:
            compare(validation,dict(label=label,request_id=request['request_id'],method=arm,repeat=repeat,
                status='valid',valid=True,errors=[],canonical_source='retained_manifest_eval_requests',canonical_method='F'),evidence.issues,label+' validation')
    for request in requests:
        for arm in ARMS:
            own=[row for row in rows if row['request_id']==request['request_id'] and row['method']==arm]
            check(len({tuple(row[key] for key in COUNTS) for row in own})<=1,evidence.issues,f'event{event}/{request["request_id"]}/{arm}: repeat counts')
    if status is None or status.get('status')!='completed':
        return dict(status='incomplete',returned=len(rows),execution=status)
    compare(status,dict(planned_generations=384,attempted_generations=384,returned_generations=384,
        completed_valid_generations=384,invalid_generations=0,pending_canonical_generations=0,
        unknown_generation_attempts=0,unattempted_generations=0,optimizer_steps=0,
        feedback_collections=0,automatic_retry=False,qualification='passed',qualification_requests=8,
        interrupted_attempt=None),evidence.issues,f'event{event} evaluation completion')
    gate=evidence.read(root/'QUALIFICATION.json')
    if gate is not None:
        compare(gate,dict(status='passed',requests=8,generations=48,arms_with_native_equality=['K','O'],
            performance_win_required=False,extra_generations=0),evidence.issues,f'event{event} included gate')
    raw_files={p.name for p in root.glob('*.json') if re.fullmatch(r'\d{4}_r[012]_[OK]\.json',p.name)}
    check(len(rows)==len(raw_files)==384 and raw_files=={r['label']+'.json' for r in rows},evidence.issues,f'event{event}: exact raw evaluation set')
    events=lines(root/'VALIDATION_EVENTS.jsonl')
    check(len(events)==384 and all(r['status']=='valid' and not r['errors'] for r in events)
          and {r['label'] for r in events}=={r['label'] for r in rows},evidence.issues,f'event{event}: durable validation events')
    activations=lines(root/'ACTIVATIONS.jsonl'); last=None; expected_activations=[]
    for request,repeat,arm in expected:
        if arm!=last:
            expected_activations.append((request['panel_ordinal'],repeat,arm));last=arm
    check([(r['panel_ordinal'],r['repeat'],r['method']) for r in activations]==expected_activations,evidence.issues,f'event{event}: activation schedule')
    for row in activations:
        check(not row['errors'],evidence.issues,f'event{event}: publication errors')
        publication(row,checkpoints[row['method']],evidence.issues,f'event{event} publication')
    weight_s=0
    for arm in ARMS:
        weights=evidence.read(root/f'WEIGHTS_{arm}.json')
        if weights is not None:
            check(not weights['errors'],evidence.issues,f'event{event}/{arm}: weight gate errors')
            tensor_equality(weights['native_reply'],checkpoints[arm],evidence.issues,f'event{event}/{arm} loaded weights')
            weight_s+=weights['validation_s']
    check(near(status['activation_s'],sum(r['activation_s'] for r in activations))
          and near(status['weight_validation_s'],weight_s),evidence.issues,f'event{event}: activation/weight costs')
    summary=evidence.read(root/'report/SUMMARY.json'); totals={}
    for arm in ARMS:
        own=[r for r in rows if r['method']==arm]
        primary=[r for r in own if r['repeat']==0]
        totals[arm]={key:sum(r[key] for r in primary) for key in COUNTS}
        totals[arm]['summed_mean_request_latency_s']=sum(r['request_latency_s'] for r in own)/3
        if summary is not None:
            for key,value in totals[arm].items():
                check(near(summary[arm][key],value),evidence.issues,f'event{event}/{arm}: report total '+key)
    return dict(status='complete',returned=len(rows),totals=totals,execution=status,
                activations=len(activations),included_qualification_generations=48)


def provenance(evidence, plan, seal):
    """Rehash locally present compact evidence; preserve remote tensor-check limits."""
    prefixes = [(f'/data/sources/minhao/runs/{HERE.name}/', evidence.raw),
                ('/data/sources/minhao/DFlash-Online/', HERE/'worktree')]
    checked, remote_only = [], []
    for name, expected in seal['files'].items():
        local = None
        for prefix, folder in prefixes:
            if name.startswith(prefix):
                local = folder/name[len(prefix):]
                if not local.is_file() and folder == evidence.raw:
                    local = HERE/name[len(prefix):]
                break
        if local is None or not local.is_file():
            remote_only.append(name)
            continue
        check(identity(local) == expected, evidence.issues, 'Sealed local bytes: '+name)
        checked.append(name)
    plan_names=[name for name in checked if name.endswith('/sealed/PLAN.json')]
    check(len(plan_names)==1,evidence.issues,'Exactly one locally verified sealed PLAN')
    pin=evidence.read(evidence.raw/'sealed/RUNTIME_PIN.json')
    pre=evidence.read(evidence.raw/'PREFLIGHT.json')
    post=evidence.read(evidence.raw/'POSTFLIGHT.json')
    for label, row in (('preflight',pre),('postflight',post)):
        if row is None:continue
        compare(row,dict(status='passed',files_checked=len(seal['files'])),evidence.issues,label)
        if pin is not None:
            compare(row['runtime'],{k:pin[k] for k in ('python_version','torch','transformers','cuda','native_runtime_sha256')},evidence.issues,label+' runtime')
    if pre is not None:
        for arm in ARMS:
            compare(pre['parents'][arm],dict(status='passed',objective=OBJECTIVES[arm],
                completed_event=1,global_step=128,presentations=256,horizon=7424,
                export_exact=True,state_path=plan['initial'][arm]['state'],
                export_path=plan['initial'][arm]['export']),evidence.issues,'preflight parent '+arm)
    windows_path=evidence.raw/'data_preparation/WINDOWS.json'
    if not windows_path.is_file():windows_path=HERE/'data_preparation/WINDOWS.json'
    check(load(windows_path)==plan['windows'],evidence.issues,'Plan uses independently prepared exact windows')
    selection_path=windows_path.parent/'SELECTION_AUDIT.json'
    selection=load(selection_path)
    check(selection['selection_complete'] and selection['selected_count']==384,evidence.issues,'Complete selection audit')
    for name,digest in selection['output_sha256'].items():
        check(identity(windows_path.parent/name)['sha256']==digest,evidence.issues,'Prepared manifest digest '+name)
    selected=[r for w in plan['windows'] for r in w['requests']]
    check([r['request_id'] for r in selected]==selection['selected_request_ids'],evidence.issues,'Exact selection order')
    check([r['dataset_index'] for r in selected]==list(range(128,512)) and
        all(r['older_project_execution_verified'] for r in selected),evidence.issues,'Original chronological indices128--511 and disclosed older exposure')
    check(len({tuple(r['prompt_ids']) for r in selected})==384 and
        not {tuple(r['prompt_ids']) for r in selected}&{tuple(r['prompt_ids']) for r in plan['eval_requests']},evidence.issues,'Unique training inputs disjoint from development panel')
    original=load(HERE.parent/'dflash_osd_one_event_20261025/sealed/MANIFEST.json')
    check(plan['eval_requests']==original['eval_requests'],evidence.issues,'Exact retained original development manifest including F outputs')
    history=lines(HERE.parent/'dflash_osd_one_event_20261025/final_review/raw/campaign_001/evaluation/MEASUREMENTS.jsonl')
    check(plan['frozen_counts']=={row['request_id']:row['verification_calls'] for row in history if row['method']=='F' and row['repeat']==0},evidence.issues,'Retained F count context')
    settings=evidence.read(evidence.raw/'campaign_001/native/engine_settings.json')
    if settings is not None:
        compare(settings,dict(model_path=plan['target'],speculative_draft_model_path=plan['draft'],
            speculative_algorithm='DFLASH',speculative_num_draft_tokens=16,dtype='bfloat16',
            attention_backend='fa3',tp_size=1,max_running_requests=1,mem_fraction_static=.65,
            context_length=4112,chunked_prefill_size=8192,cuda_graph_max_bs=8,random_seed=17,
            skip_server_warmup=True),evidence.issues,'Native engine configuration')
    native_status=evidence.read(evidence.raw/'campaign_001/native/native_status.json')
    if native_status is not None:
        compare(native_status,dict(status='passed',poisoned=None,capture_pending=False,
            cache_tokens=0,attention_backend='fa3',block_size=16),evidence.issues,'Final native status')
    final_path=evidence.raw/'FINAL_ARTIFACTS.json'
    final=load(final_path) if final_path.is_file() else None
    return dict(locally_rehashed=len(checked),remote_only_count=len(remote_only),
        remote_only=remote_only,seal_files=len(seal['files']),preflight=pre,postflight=post,
        final_artifacts_inventory=final,
        limits='Local compact files are rehashed. Model/restart/feature tensor bytes and external runtime checks rely on sealed pre/postflight and live runtime receipts; this audit performs no tensor reload.')


def execution(evidence, plan, seal):
    compare(plan,dict(stop_step=512,inherited_horizon=7424,evaluation_events=[2,4],
        shared_feedback_source='O pre-update checkpoint',fresh_F_latency=False,
        additional_optimizer_qualification_steps=0,generation_warmups=0,
        new_budget=dict(feedback=384,evaluation=768,generations=1152,presentations=1536,optimizer_steps=768)),evidence.issues,'Authorized plan')
    check([w['event_index'] for w in plan['windows']]==[2,3,4] and
        all(len(w['requests'])==128 for w in plan['windows']) and len(plan['eval_requests'])==64,
        evidence.issues,'Exact three training windows and original64 panel')
    compare(plan['serving'],dict(block=16,output_cap=2048,context=4096,greedy=True,concurrency=1),evidence.issues,'Serving plan')
    parents={arm:dict(plan['initial'][arm]) for arm in ARMS}
    events={};serial=1
    for window in plan['windows']:
        n=window['event_index']; event={}
        event['pool']=evidence.section(f'event{n} feedback',lambda:feedback_pool(evidence,window,parents['O']['export'],serial))
        serial+=128
        ordinary=None
        for arm in ARMS:
            event[arm]={}
            resumed=evidence.section(f'event{n}/{arm} resume',lambda:resume_receipts(evidence,n,arm,parents[arm],seal))
            event[arm]['resume']=resumed
            path=evidence.raw/f'campaign_001/{arm}/events/event_{n:03d}/RESULT.json'
            result=evidence.read(path)
            if result is None:continue
            event[arm]['training']=evidence.section(f'event{n}/{arm} exposure',lambda:training_exposure(result,n,arm,evidence.issues,ordinary))
            if event['pool'].get('status')=='complete':
                check(result['features']==event['pool']['feature_metadata'],evidence.issues,f'event{n}/{arm}: exact shared immutable feature list')
            if resumed.get('closed') is not None:
                check(resumed['closed']['full_state_path']==result['full_state_path'],evidence.issues,f'event{n}/{arm}: final state path')
            root_prefix=f'/data/sources/minhao/runs/{HERE.name}/campaign_001/{arm}/'
            check(result['full_state_path']==root_prefix+'training_state.pt' and
                result['export_path']==root_prefix+f'events/event_{n:03d}/export',evidence.issues,f'event{n}/{arm}: own isolated rolling state and event export')
            if arm=='O':ordinary=result
            parents[arm]=dict(state=result['full_state_path'],export=result['export_path'])
        if n in (2,4):
            checks={arm:parents[arm]['export'] for arm in ARMS}
            event['evaluation']=evidence.section(f'event{n} evaluation',lambda:evaluation(evidence,n,plan,checks,serial))
            serial+=384
            if event['evaluation'].get('returned',0):
                check(all((event[arm]['resume'].get('closed') or {}).get('closed') is True and
                    (event[arm]['resume'].get('closed') or {}).get('global_step')==n*128 for arm in ARMS),evidence.issues,f'event{n}: both trainer closures precede evaluation in sealed sequential driver')
        events[str(n)]=event
    root=evidence.raw/'campaign_001';status=evidence.read(root/'STATUS.json')
    costs=evidence.read(root/'COSTS.json')
    if costs is not None:
        for key in ('campaign_wall_s','backend_loading_s'):
            check(type(costs.get(key)) in (int,float) and math.isfinite(costs[key]) and costs[key]>0,evidence.issues,'Campaign cost '+key)
        for n,event in events.items():
            pool_cost=event['pool'].get('costs')
            if pool_cost is not None:
                check(costs.get(f'event_{int(n):03d}_feedback')==pool_cost,evidence.issues,'Root feedback cost agreement event'+n)
            for arm in ARMS:
                arm_cost=event[arm]['resume'].get('costs')
                if arm_cost is not None:
                    check(costs.get(f'event_{int(n):03d}_{arm}')==arm_cost,evidence.issues,f'Root training cost agreement event{n}/{arm}')
    actual_native={p.name for p in (root/'native').glob('request_*.json')}
    unassigned=sorted(actual_native-evidence.used_native)
    check(evidence.used_native<=actual_native,evidence.issues,'Every wrapper has retained native evidence')
    check(not (root/'evaluation_003').exists() and not (root/'evaluation_001').exists(),evidence.issues,'No event1 repetition or event3 evaluation')
    for arm in ARMS:
        check(not (root/arm/'events/event_001').exists(),evidence.issues,'No repeated event1 training: '+arm)
    steps={arm:128 for arm in ARMS};presentations={arm:256 for arm in ARMS}
    interrupted=False
    for n in (2,3,4):
        for arm in ARMS:
            folder=root/f'resume_{n:03d}_{arm}'
            closed=folder/'CLOSED.json';failed=folder/'FAILED.json'
            value=load(closed) if closed.is_file() else load(failed).get('status') if failed.is_file() else None
            if value is not None:
                steps[arm]=value['global_step'];presentations[arm]=value['presentations']
            interrupted=interrupted or failed.is_file() or (folder/'CLEANUP_ERROR.json').is_file()
    feedback=sum(e['pool'].get('responses',0) for e in events.values())
    eval_returned=sum(e.get('evaluation',{}).get('returned',0) for e in events.values())
    terminal=sum(e['pool'].get('terminal_forwards',0) for e in events.values())
    work=dict(feedback_response_records=feedback,evaluation_response_records=eval_returned,
        native_response_records=len(actual_native),native_blocks=evidence.native_blocks,
        unassigned_native_records=unassigned,terminal_feature_forwards=terminal,
        completed_optimizer_steps_observed=sum(steps.values())-256,
        presentations_observed=sum(presentations.values())-512,arm_counters=steps,
        training_interrupted=interrupted or bool((status or {}).get('training_interrupted')),
        optimizer_counter_scope='Completed calls reported by trainer; an interrupted in-flight call is not excluded.',
        no_extra_work_limit='Exact output inventories and sealed serial orchestration support bounded work. No claim of detecting unrecorded external processes.')
    if status is not None:
        compare(status,dict(arm_counters=steps,optimizer_steps=work['completed_optimizer_steps_observed'],
            presentations=work['presentations_observed'],automatic_retry=False,extra_optimizer_steps=0,
            extra_generations=0),evidence.issues,'Root reconciled work')
        if status.get('status')=='completed':
            compare(status,dict(phase='finished',feedback_attempted=384,feedback_returned=384,
                evaluation_returned=768,evaluation_attempted=768,evaluation_unknown=0,feedback_unknown=0,
                evaluation_unattempted=0,feedback_unattempted=0,active_feedback=None,
                arm_counters=dict(O=512,K=512),optimizer_steps=768,presentations=1536,
                events_completed=[2,3,4],training_interrupted=False),evidence.issues,'Completed campaign budget')
            check((feedback,eval_returned,len(actual_native),terminal)==(384,768,1152,384),evidence.issues,'Independent exact work inventory')
            check(actual_native=={f'request_{i:06d}.json' for i in range(1,1153)} and not unassigned,evidence.issues,'Exact1152 serials with no extra captures')
            check(not interrupted,evidence.issues,'No failed training or cleanup receipts in completed campaign')
    return dict(status=status,events=events,work=work,costs=costs,
        ordering_evidence='Sealed single-process driver closes each trainer before calling evaluation; six closure receipts and serial native identities support this order. Compact receipts have no independent overlapping-activity clock trace.')


def audit(raw):
    evidence=Evidence(raw)
    plan=evidence.read(evidence.raw/'sealed/PLAN.json')
    seal=evidence.read(evidence.raw/'sealed/SEAL.json')
    result=dict(schema='ok_continuation_independent_compact_audit_v1',raw_root=str(evidence.raw),
        audit_source=identity(__file__),new_model_forwards=0,new_optimizer_steps=0)
    if plan is not None and seal is not None:
        result['provenance']=evidence.section('provenance',lambda:provenance(evidence,plan,seal))
        result['execution']=evidence.section('execution',lambda:execution(evidence,plan,seal))
    exit_path=evidence.raw/'EXIT_CODE'
    exit_code=exit_path.read_text().strip() if exit_path.is_file() else None
    if exit_code is None:evidence.missing.append(str(exit_path))
    if exit_code not in (None,'0'):evidence.issues.append('Supervisor exit code '+exit_code)
    campaign=result.get('execution',{}).get('status') or {}
    child_complete=True
    if campaign.get('status')=='completed':
        events=result.get('execution',{}).get('events',{})
        for event_number in (2,3,4):
            event=events.get(str(event_number),{})
            checks=[('feedback pool',event.get('pool',{}).get('status')=='complete')]
            for arm in ARMS:
                own=event.get(arm,{})
                checks.extend([(arm+' resume',own.get('resume',{}).get('status')=='passed'),
                    (arm+' training',own.get('training',{}).get('status')=='completed')])
            if event_number in (2,4):
                evaluated=event.get('evaluation',{})
                checks.append(('evaluation',evaluated.get('status')=='complete' and
                    (evaluated.get('execution') or {}).get('status')=='completed' and
                    evaluated.get('returned')==384))
            for label,valid in checks:
                check(valid,evidence.issues,f'Completed root requires completed event{event_number} '+label)
                child_complete=child_complete and valid
    complete=campaign.get('status')=='completed' and exit_code=='0' and child_complete
    result.update(status='passed' if complete and not evidence.issues and not evidence.missing else
                  'failed' if evidence.issues else 'incomplete',completed_campaign=complete,
        supervisor_exit_code=exit_code,issues=evidence.issues,missing=evidence.missing,
        verification_status='ANALYZED; compact evidence recalculated, no scientific experiment rerun',
        methodological_scan={
            'coverage':'11/11 design and accounting checks; numerical direction and efficacy interpretation are in the separate analysis report.',
            'Simpsons_paradox':'Keep events and timing repeats separate; do not pool event1/2/4 latency.',
            'ecological_fallacy':'Aggregate improvement does not imply every request improves; repeats do not create independent trained seeds.',
            'Berksons_paradox':'Original development panel and chronological train selection are fixed; older-project exposure is disclosed.',
            'collider_bias':'Validity gate covers all returned rows; no filtering by favorable performance is permitted.',
            'base_rate_neglect':'Report all64 requests and full384 evaluation slots per evaluated event, with incomplete/unknown work visible.',
            'regression_to_mean':'Continuation follows an earlier positive screening result; use concurrent O/K within each event, not an unadjusted before/after latency claim.',
            'survivorship_bias':'Missing, failed and unknown work prevent a complete passed audit; no replacement or automatic retry.',
            'look_elsewhere_effect':'Report both prespecified evaluation events2/4 and retained event1 context; no significance testing in this auditor.',
            'forking_paths':'Sealed schedule, checkpoints, fixed budgets and unchanged source constrain analysis choices; broader study remains exploratory.',
            'correlation_causation':'Controlled objective comparison is conditional on shared O feedback and one trajectory; it does not establish net deployment acceleration or a unique mechanism.',
            'reverse_causality':'O checkpoint is bound before feedback collection; own-arm updates precede evaluation by sealed serial orchestration.'},
        scope_limits=[
            'One continued training trajectory per arm; repeat generations are serving repeats, not independent training seeds.',
            'Evaluation uses the original64 development inputs and retained F canonical outputs. F provides no fresh latency baseline.',
            'The next384 training inputs exclude both64 evaluation panels and original128 training inputs. All384 have disclosed older-project exposure.',
            'Shared O-generated feedback controls this objective comparison; it does not estimate independent K deployment feedback or adaptation-inclusive speedup.',
            'Natural K masks are audited through bounded native qualification and current-forward aggregate diagnostics; this compact continuation audit cannot reconstruct unsaved training logits.',
            'Masking and selected-weight normalization change together. Training losses with different selected distributions are not direct efficacy comparisons.'
        ])
    return result


def markdown(result):
    work=result.get('execution',{}).get('work',{})
    lines=['# Independent O/K continuation execution audit','',
        f"Status: **{result['status']}**. Completed campaign: {result['completed_campaign']}. "
        f"Issues: {len(result['issues'])}; missing files: {len(result['missing'])}.",'',
        'Material Passport: ARS validate; local compact execution evidence; Verification Status: ANALYZED. '
        'This audit loads no model or optimizer and performs no inference.','',
        f"Observed feedback/evaluation/native records: {work.get('feedback_response_records')}/"
        f"{work.get('evaluation_response_records')}/{work.get('native_response_records')}. "
        f"Reconstructed native verification blocks: {work.get('native_blocks')}.",
        f"Completed optimizer calls reported by trainer: {work.get('completed_optimizer_steps_observed')}; "
        f"presentations: {work.get('presentations_observed')}; arm counters: {work.get('arm_counters')}. "
        f"Interrupted training: {work.get('training_interrupted')}.",'',
        'Checks cover own-arm complete-state resume receipts, exact live Adam/RNG/counter restoration receipts, '
        'causal O-only feedback checkpoint identity, shared immutable feature metadata, matched response/anchor schedules, '
        'boundary-only exposure totals, first-error inclusion/count bounds, full native token/termination/call traces, '
        'rotation/repeat schedules, included correctness gates, source/runtime pre/postflight, and phase-specific storage admission.','',
        'Exact tensor restore is evidenced by sealed runtime receipts, not repeated locally. '
        'Sequential source and closure receipts establish the intended training/evaluation order; '
        'there is no independently timestamped activity-overlap trace. Missing/incomplete runs cannot pass.','',
        'Interpretation limits:','']
    lines += ['- '+value for value in result['scope_limits']]
    lines += ['','Methodological scan: '+result['methodological_scan']['coverage'],'']
    lines += ['- '+key.replace('_',' ')+': '+value for key,value in result['methodological_scan'].items() if key!='coverage']
    if result['issues']:
        lines += ['','Issues:','']+['- '+value for value in result['issues']]
    if result['missing']:
        lines += ['','Missing evidence:','']+['- '+value for value in result['missing']]
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root',type=Path,default=HERE/'raw')
    parser.add_argument('--output',type=Path,default=HERE)
    args=parser.parse_args();result=audit(args.raw_root)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'FINAL_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    (args.output/'FINAL_AUDIT.md').write_text(markdown(result))
    print(json.dumps(dict(status=result['status'],issues=len(result['issues']),missing=len(result['missing']))))
    return 0 if result['status']=='passed' else 1


if __name__=='__main__':
    raise SystemExit(main())
