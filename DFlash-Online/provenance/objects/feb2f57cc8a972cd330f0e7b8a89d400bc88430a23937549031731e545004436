"""Independent local compact-evidence audit of the O/C/K/A campaign.

No model imports, tensors, GPU, remote access or experiment execution. Negative
and null outcomes pass when complete and correct. Missing evidence fails closed.
Native longest-prefix validation is reused from the separately hashed retained
independent auditor. Tensor equality is supported by retained exact-check receipts;
this compact audit does not rerun tensor comparisons or reconstruct unsaved logits.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path

ARMS = ('O', 'C', 'K', 'A')
OBJECTIVES = {'C': 'osd_ce', 'K': 'osd_first_error', 'A': 'osd_ce_first_error'}
PAIRS = (('K', 'O'), ('A', 'C'), ('C', 'O'), ('A', 'K'))
COUNTS = ('verification_calls', 'accepted_draft_tokens', 'committed_progress', 'output_tokens')
HISTOGRAM = tuple('first_error_position_' + str(i) for i in range(1, 16))
SUMS = ('available_position_occurrences', 'selected_position_occurrences',
        'available_position_weight', 'selected_position_weight', 'no_error_blocks', 'empty_blocks',
        'main_loss_numerator', 'loss_denominator', 'response_presentations',
        'supervised_position_exposure') + HISTOGRAM
SCHEDULE = ('record_id', 'pass', 'sampled_anchors', 'optimizer_stepped', 'optimizer_step_after')
TRIALS = (('warmup', 'osd', False, None), ('O', 'osd', False, None),
          ('C', 'osd_ce', False, None), ('K_all', 'osd_first_error', True, 'O'),
          ('A_all', 'osd_ce_first_error', True, 'C'),
          ('K_natural', 'osd_first_error', False, None),
          ('A_natural', 'osd_ce_first_error', False, None))
GATES = ('ordinary_compatibility', 'ce_probability_arithmetic', 'first_error_alignment',
         'all_selected_loss_and_gradients', 'excluded_prediction_gradients',
         'fresh_state_rng_schedule', 'memory_fit')
WORK = dict(current_training_forward=14, temporary_optimizer_step=7, backward=14,
            detached_logit_gradient=30, reference_training_forward=0, detached_comparison_forward=0)
PLAN = dict(temporary_optimizer_steps=7, response_presentations=14, current_training_forwards=14,
            backwards=14, detached_logit_gradient_calls=30, reference_training_forwards=0,
            detached_comparison_forwards=0, teacher_model_forwards=0, generations=0,
            main_optimizer_steps=0, captured_logit_rows_per_presentation=8)
DEFAULT_TRACE_AUDITOR = Path(__file__).resolve().parents[1] / 'dflash_prefix_ranking_one_event_20261025/audit_results.py'

def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def check(condition, issues, message):
    if not condition:
        issues.append(message)


def close(a, b):
    return a is None and b is None or (type(a) in (int, float) and type(b) in (int, float)
        and math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-9))


def compare(actual, expected, issues, label):
    """Compare expected fields recursively; prose/additional fields are not metrics."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            issues.append(label + ' is not a mapping')
            return
        for key, value in expected.items():
            compare(actual.get(key), value, issues, label + '.' + key)
    else:
        check(close(actual, expected) if type(expected) in (int, float) or expected is None
              else actual == expected, issues, label + ' differs')


def load_trace_auditor(path):
    spec = importlib.util.spec_from_file_location('retained_independent_trace_auditor', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def exact_gate(path, issues):
    gate = read(path)
    check(gate['exact_equal'] is True and not any(gate[key] for key in
          ('missing_left', 'missing_right', 'metadata_differences', 'tensor_differences')),
          issues, 'Exact state gate failed: ' + str(path))
    return dict(path=str(path), sha256=digest(path), equal_tensors=gate.get('equal_tensors'),
                compared_coordinates=gate.get('compared_coordinates'))


def qualification(root, ordinary, issues):
    directory = root / 'qualification'
    result, work = read(directory / 'RESULT.json'), lines(directory / 'WORK.jsonl')
    compare(result, dict(scope='first_error_OCKA', status='passed', plan=PLAN,
        checks={name: True for name in GATES}, baseline_tolerance='exact equality',
        generations=0, main_optimizer_steps=0, main_campaign_started=False,
        all_temporary_tensors_retired=True), issues, 'Qualification contract')
    counts = {}
    for kind, maximum in WORK.items():
        counts[kind] = {state: sum(row['kind'] == kind and row.get('status') == state for row in work)
                       for state in ('started', 'completed', 'failed')}
        check(counts[kind] == dict(started=maximum, completed=maximum, failed=0),
              issues, 'Qualification budget/completion: ' + kind)
        for state, suffix in (('started', 'attempted'), ('completed', 'completed')):
            check(result['work_counts'].get(kind + '_' + suffix, 0) == counts[kind][state],
                  issues, 'Qualification work count: ' + kind + '_' + suffix)
    presentations = [row for row in work if row['kind'] == 'presentation_complete']
    check(len(presentations) == result['work_counts']['response_presentations'] == 14
          and result['work_counts']['current_draft_forwards_observed'] == 14
          and result['work_counts'].get('reference_draft_forwards_observed', 0) == 0,
          issues, 'Qualification observed forward/presentation counts')
    trials = result['trials']
    check([row['label'] for row in trials] == [row[0] for row in TRIALS], issues, 'Seven ordered trials')
    by_label, gates = {row['label']: row for row in trials}, []
    for label, objective, force_all, baseline in TRIALS:
        trial = by_label[label]
        check(trial == read(directory / label / 'TRIAL.json'), issues, label + ': duplicate trial receipt')
        compare(trial, dict(objective=objective, force_all=force_all, baseline=baseline,
            status='discarded_warmup' if label == 'warmup' else 'passed'), issues, label + ' contract')
        check(len(trial['records']) == 2, issues, label + ': two presentations')
        for index, row in enumerate(trial['records']):
            check(all(row[key] == ordinary['records'][index][key] for key in SCHEDULE),
                  issues, label + ': retained schedule at ' + str(index))
        shuffles = [row for row in work if row['kind'] == 'full_first_pass_shuffle' and row['trial'] == label]
        check(len(shuffles) == 1 and shuffles[0]['n'] == 128 and shuffles[0]['exact_equal'] is True
              and shuffles[0]['record_ids'] == [row['record_id'] for row in ordinary['records'][:128]],
              issues, label + ': full retained first-pass shuffle')
        logged = [row for row in presentations if row['trial'] == label]
        check([row['presentation'] for row in logged] == [0, 1]
              and [row['row'] for row in logged] == trial['records'], issues, label + ': durable presentations')
        for kind in WORK:
            expected = ([0, 1] if kind in ('current_training_forward', 'backward') else
                        [1] if kind == 'temporary_optimizer_step' else
                        [0, 0, 0, 1, 1, 1] if kind == 'detached_logit_gradient' and objective != 'osd' else [])
            for state in ('started', 'completed'):
                actual = [row['presentation'] for row in work if row['kind'] == kind
                          and row['trial'] == label and row.get('status') == state]
                check(actual == expected, issues, label + ': work/presentation binding ' + kind + '/' + state)
                suffix = 'attempted' if state == 'started' else 'completed'
                check(trial['work_counts'].get(kind + '_' + suffix, 0) == len(expected),
                      issues, label + ': per-trial work counter ' + kind + '/' + state)
        check([s['stage'] for s in trial['gradient_snapshots']] == ['after_backward_0', 'after_backward_1', 'pre_optimizer']
              and len({s['path'] for s in trial['gradient_snapshots']}) == 3
              and all(s['bytes'] > 0 for s in trial['gradient_snapshots']), issues, label + ': gradient boundaries')
        gates.append(exact_gate(directory / label / 'initial_vs_pinned.json', issues))
        if label not in ('warmup', 'O'):
            gates.append(exact_gate(directory / label / 'rng_vs_O.json', issues))
        if baseline:
            check(trial['exact_baseline_compatibility'] is True, issues, label + ': exact all-selected compatibility')
            for stage in ('after_backward_0', 'after_backward_1', 'pre_optimizer', 'final', 'records'):
                gates.append(exact_gate(directory / label / (stage + '_vs_baseline.json'), issues))
        check(all(trial['objective_work'][name] == 0 for name in
                  ('reference_setup_s', 'reference_forward_s', 'reference_head_s', 'projection_s')),
              issues, label + ': no reference or projection work')
        if objective in ('osd_ce', 'osd_ce_first_error'):
            check(trial['objective_work']['teacher_head_s'] == 0, issues, label + ': CE teacher head cost')
        if objective == 'osd':
            check(not trial['loss_checks'] and not trial['gradient_checks'], issues, label + ': no extra logit VJPs')
        else:
            for field, kind in (('loss_checks', 'first_error_loss_check'), ('gradient_checks', 'prediction_gradient_check')):
                check([row['presentation'] for row in trial[field]] == [0, 1], issues, label + ': ' + field + ' coverage')
                logged = [{k: v for k, v in row.items() if k not in ('kind', 'trial')}
                          for row in work if row['kind'] == kind and row['trial'] == label]
                check(logged == trial[field], issues, label + ': durable ' + field)
            for row in trial['loss_checks']:
                check(all(row[k] is True for k in ('exact_loss', 'exact_mask', 'exact_logging'))
                      and row['probability_dtype'] == 'torch.bfloat16' and row['forced_all_selected'] == force_all
                      and 0 <= row['selected'] <= row['available'] and row['chunks'] > 0,
                      issues, label + ': current mask/loss/logging checks')
            for row in trial['gradient_checks']:
                check(all(row[k] is True for k in ('actual_excluded_gradient_zero',
                          'detached_excluded_gradient_zero', 'all_selected_gradient_exact'))
                      and 0 < row['rows'] <= 8 and row['detached_autograd_calls'] == 3
                      and row['dtype'] == 'torch.bfloat16', issues, label + ': prediction gradients')
    check([row['metrics'] for row in by_label['O']['records']] == [row['metrics'] for row in ordinary['records'][:2]],
          issues, 'Retained O first-window scalar metrics')
    storage = result['storage']
    check(storage['required_bytes'] == 28 * storage['draft_bytes'] + 10 * 2**30
          and storage['free_bytes_at_admission'] >= storage['required_bytes'], issues, 'Qualification storage admission')
    reclamation = read(directory / 'RECLAMATION.json')
    expected_paths = {(label, name + '.pt') for label, *_ in TRIALS
                      for name in ('initial', 'after_backward_0', 'after_backward_1', 'pre_optimizer', 'final')}
    entries = reclamation['entries']
    check(len(entries) == 35 and {(Path(e['path']).parent.name, Path(e['path']).name) for e in entries} == expected_paths,
          issues, 'Exactly35 run-local temporary tensor retirements')
    for entry in entries:
        check(entry['removed'] is True and entry['bytes'] > 0 and len(entry['sha256']) == 64
              and entry['retained_evidence'] and entry['lost_capability'], issues, 'Tensor retirement evidence')
        check(set(entry['retained_evidence_sha256']) == set(entry['retained_evidence']), issues, 'Retirement hash coverage')
        for path, expected in entry['retained_evidence_sha256'].items():
            parts = Path(path).parts
            if 'qualification' in parts:
                local = directory.joinpath(*parts[parts.index('qualification') + 1:])
                check(digest(local) == expected, issues, 'Retained compact retirement evidence changed: ' + path)
            else:
                check(result['input_sha256'].get(path) == expected, issues, 'Retirement retained pin binding')
    for index, batch in enumerate(reclamation['batches']):
        check(batch['batch'] == index and batch['status'] == 'completed'
              and batch['free_bytes_before'] >= 0 and batch['free_bytes_after'] >= 0
              and batch['paths'] == [e['path'] for e in entries if e['batch'] == index], issues, 'Reclamation batch receipt')
    check(not list(directory.rglob('*.pt')), issues, 'Compact qualification should have no temporary tensors')
    return dict(status=result['status'], work=counts, presentation_count=14, exact_gates=gates,
                wall_s=result['wall_s'], storage=storage, retired_temporary_tensors=len(entries),
                historical_O_limit=result['historical_O_limit'])


def diagnostic_totals(rows):
    boundary = [row for row in rows if row['optimizer_stepped']]
    sums = {key: sum(row['metrics'][key] for row in boundary) for key in SUMS}
    return dict(boundary_rows=len(boundary), sums=sums,
        occurrence_selection_rate=sums['selected_position_occurrences'] / sums['available_position_occurrences']
            if sums['available_position_occurrences'] else None,
        weight_selection_rate=sums['selected_position_weight'] / sums['available_position_weight']
            if sums['available_position_weight'] else None,
        pooled_weighted_main_loss=sums['main_loss_numerator'] / sums['loss_denominator']
            if sums['loss_denominator'] else None)


def training(root, ordinary, issues):
    results = {}
    features = {row['record_id']: row for row in ordinary['features']}
    for arm, objective in OBJECTIVES.items():
        receipt = read(root / arm / 'events/event_001/RESULT.json')
        compare(receipt, dict(status='completed', objective=objective, reduction='response_mean',
            passes=2, optimizer_steps=128, response_presentations=256, start_step=0, end_step=128,
            selection_totals_source='optimizer_boundary_accumulation_windows_once',
            gradient_denominator='responses per accumulation window'), issues, arm + ' receipt')
        check(receipt['features'] == ordinary['features'] and len(receipt['records']) == 256,
              issues, arm + ': retained feature pool/presentation count')
        slots, anchors, positions, weight_sum, pending = 0, 0, 0, 0., []
        for index, (row, original) in enumerate(zip(receipt['records'], ordinary['records'])):
            label = arm + ': presentation ' + str(index)
            check(all(row[key] == original[key] for key in SCHEDULE + ('learning_rate',)), issues, label + ': schedule/LR')
            check(row['metrics_scope'] == ('accumulation_window' if index % 2 else 'presentation'), issues, label + ': metric scope')
            feature = features[row['record_id']]
            nslots, nanchors, npositions, nweight = 0, 0, 0, 0.
            check(len(row['sampled_anchors']) == 1, issues, label + ': single retained anchor draw')
            for draw in row['sampled_anchors']:
                check(len(draw['positions']) == len(draw['keep_mask']), issues, label + ': anchor batch shape')
                for sampled, keep in zip(draw['positions'], draw['keep_mask']):
                    check(len(sampled) == len(keep), issues, label + ': anchor keep shape')
                    nslots += len(sampled)
                    for anchor, active in zip(sampled, keep):
                        if active:
                            nanchors += 1
                            check(feature['prompt_length'] <= anchor < feature['length'] - 1, issues, label + ': eligible anchor')
                            offsets = [j for j in range(1, 16) if feature['prompt_length'] <= anchor + j < feature['length']]
                            npositions += len(offsets)
                            nweight += sum(math.exp(-(j-1)/7.) for j in offsets)
            slots += nslots
            anchors += nanchors
            positions += npositions
            weight_sum += nweight
            pending.append((nslots, npositions, nweight))
            m = row['metrics']
            check(all(type(value) in (int, float) and math.isfinite(value) for value in m.values()), issues, label + ': finite metrics')
            check(m['response_presentations'] == len(pending)
                  and m['available_position_occurrences'] == sum(v[1] for v in pending), issues, label + ': available occurrence/response counts')
            check(m['no_error_blocks'] + m['empty_blocks'] + sum(m[k] for k in HISTOGRAM) == sum(v[0] for v in pending),
                  issues, label + ': first-error histogram partitions sampled blocks')
            for name in ('available_position_occurrences', 'selected_position_occurrences', 'no_error_blocks', 'empty_blocks') + HISTOGRAM:
                check(m[name] >= 0 and m[name] == int(m[name]), issues, label + ': integral count ' + name)
            check(0 <= m['selected_position_occurrences'] <= m['available_position_occurrences']
                  and m['selected_position_occurrences'] == m['supervised_position_exposure']
                  and 0 <= m['selected_position_weight'] <= m['available_position_weight'], issues, label + ': selection/exposure bounds')
            check(math.isclose(m['available_position_weight'], sum(v[2] for v in pending), rel_tol=1.3e-6, abs_tol=1e-5),
                  issues, label + ': reconstructed exponential weights')
            check(math.isclose(m['loss_denominator'], m['selected_position_weight'] + len(pending)*1e-6,
                               rel_tol=2.4e-7, abs_tol=2e-6), issues, label + ': one epsilon per response')
            # Ratios use FP32 scalar reductions; reported ratios may round once more.
            ratio = m['main_loss_numerator'] / m['loss_denominator']
            for key in ('objective_loss', 'canonical_ce' if arm in ('C', 'A') else 'forward_kl'):
                check(math.isclose(m[key], ratio, rel_tol=2.4e-7, abs_tol=2e-7), issues, label + ': pooled numerator/denominator ' + key)
            if arm == 'C':
                check(m['selected_position_occurrences'] == m['available_position_occurrences']
                      and m['selected_position_weight'] == m['available_position_weight'], issues, label + ': C selects all valid')
            else:
                error_selected = sum(offset*m[name] for offset, name in enumerate(HISTOGRAM, 1))
                check(error_selected + m['no_error_blocks'] <= m['selected_position_occurrences']
                      <= error_selected + 15*m['no_error_blocks'], issues, label + ': include-first-error count bounds')
            if row['optimizer_stepped']:
                pending = []
        check(not pending, issues, arm + ': no uncommitted accumulation')
        diagnostics = diagnostic_totals(receipt['records'])
        compare(receipt['selection_totals'], diagnostics['sums'], issues, arm + ': boundary-only selection totals')
        check(diagnostics['boundary_rows'] == 128 and diagnostics['sums']['response_presentations'] == 256
              and diagnostics['sums']['available_position_occurrences'] == positions, issues, arm + ': accumulated schedule/exposure')
        initial, final = read(root / (arm + '_INITIAL.json')), read(root / (arm + '_CLOSED.json'))
        compare(initial, dict(global_step=0, presentations=0, accumulation_remainder=0, completed_events=[],
            closed=False, failed=False, objective=objective, reduction='response_mean'), issues, arm + ' initial')
        compare(initial['controlled_initialization'], dict(exact_pinned_initial_state=True,
            full_rng_restored_after_construction=True, inherited_warmup_state=False, reference_created=False), issues, arm + ' startup')
        compare(final, dict(global_step=128, presentations=256, accumulation_remainder=0,
            completed_events=['event_001'], closed=True, failed=False), issues, arm + ' closed')
        gate = exact_gate(root / arm / 'initialization/initial_vs_pinned.json', issues)
        work = receipt['objective_work']
        check(all(work[key] == 0 for key in ('reference_setup_s', 'reference_forward_s', 'reference_head_s', 'projection_s')),
              issues, arm + ': no reference/auxiliary cost')
        if arm in ('C', 'A'):
            check(work['teacher_head_s'] == 0, issues, arm + ': no CE teacher projection')
        check(all(type(value) in (int, float) and math.isfinite(value) and value >= 0 for value in work.values()),
              issues, arm + ': finite nonnegative objective work')
        results[arm] = dict(status=receipt['status'], optimizer_steps=128, response_presentations=256,
            sampled_block_slots=slots, kept_anchors=anchors, reconstructed_available_occurrences=positions,
            reconstructed_available_weight=weight_sum, diagnostics=diagnostics, initial_gate=gate, objective_work=work)
    results['evidence_limit'] = ('Schedule, exposure, histogram, normalization and logging consistency are checked from compact logs. '
        'Unsaved main-training logits cannot be reconstructed here; current-mask and gradient correctness use the separate qualification oracle. '
        'FP32 weight/ratio checks allow rounding only; exact qualification gates remain exact.')
    return results


def trace_extras(record, request, row, issues):
    """Bind raw identity and duplicated trace fields omitted by runner validators."""
    label = f"{request['panel_ordinal']:04d}_r{row['repeat']}_{row['method']}"
    meta, response = record['metadata'], record['response_ids']
    capture, blocks = meta['capture'], meta['trace']
    check(record['record_id'] == label and record['session_id'] == 'prefix_ranking_evaluation'
          and record['request_index'] == request['panel_ordinal']
          and Path(row['record_path']).name == label + '.json', issues, label + ': raw identity/path')
    check(row['panel_ordinal'] == request['panel_ordinal'] and row['original_index'] == request['original_index'],
          issues, label + ': panel binding')
    check(capture['blocks'] == blocks and capture['prompt_ids'] == request['prompt_ids']
          and capture['response_ids'] == capture['target_argmax_ids'] == response
          and capture['covered_token_ids'] == request['prompt_ids'] + response[:-1]
          and capture['covered_rows'] == len(capture['covered_token_ids']), issues, label + ': capture token/trace duplication')
    check(capture['status'] == 'passed' and capture['block_size'] == 16
          and capture['feature_persistence_s'] == 0 and not capture.get('feature_path')
          and not meta.get('feature_path'), issues, label + ': capture-disabled persistence')
    for index, block in enumerate(blocks):
        check(block['native_accepted_draft_tokens'] == max(block['committed_progress'] - 1, 0),
              issues, label + ': native progress counter')
        check(block['stop_clipped'] == (block['committed_progress'] < block['raw_accepted_draft_tokens_before_stop'] + 1)
              and block['finished'] == (index == len(blocks) - 1), issues, label + ': terminal trace flags')
    check(close(row['native_generate_s'], meta['native_generate_s'])
          and close(row['cache_maintenance_s'], capture['cache_maintenance_s'])
          and row['prompt_tokens'] == len(request['prompt_ids'])
          and row['cap_hit'] == (request['finish_reason'] == 'length')
          and row['finish_reason'] == request['finish_reason'], issues, label + ': measurement metadata')


def independent_totals(rows, valid_ids):
    grouped = {(request, arm): [r for r in rows if r['request_id'] == request and r['method'] == arm]
               for request in valid_ids for arm in ARMS}
    values, totals = {}, {}
    for (request, arm), repeats in grouped.items():
        primary = next(row for row in repeats if row['repeat'] == 0)
        values[request, arm] = {**{key: primary[key] for key in COUNTS},
                              'latency': sum(row['request_latency_s'] for row in repeats) / 3}
    for arm in ARMS:
        total = {key: sum(values[r, arm][key] for r in valid_ids) for key in COUNTS}
        total['summed_mean_request_latency_s'] = sum(values[r, arm]['latency'] for r in valid_ids)
        total['all_repeats_request_latency_s'] = sum(row['request_latency_s'] for row in rows
                                                    if row['method'] == arm and row['request_id'] in valid_ids)
        calls, tokens = total['verification_calls'], total['output_tokens']
        total['calls_per_1000_output_tokens'] = 1000 * calls / tokens if tokens else None
        total['accepted_draft_tokens_per_call'] = total['accepted_draft_tokens'] / calls if calls else None
        total['committed_progress_per_call'] = total['committed_progress'] / calls if calls else None
        totals[arm] = total
    comparisons = {}
    for candidate, reference in PAIRS:
        comparison = dict(candidate=candidate, reference=reference)
        for metric, key, reduction_name in (('call', 'verification_calls', 'call_reduction_pct'),
                                            ('latency', 'latency', 'request_time_reduction_pct')):
            before = [values[r, reference][key] for r in valid_ids]
            after = [values[r, candidate][key] for r in valid_ids]
            differences = [a - b for a, b in zip(before, after)]
            comparison[reduction_name] = 100 * (1 - sum(after) / sum(before)) if sum(before) else None
            comparison[metric + '_request_directions'] = dict(improved=sum(d > 0 for d in differences),
                regressed=sum(d < 0 for d in differences), tied=sum(d == 0 for d in differences))
            comparison[metric + '_gross_saved'] = sum(max(d, 0) for d in differences)
            comparison[metric + '_gross_extra'] = sum(max(-d, 0) for d in differences)
            comparison[metric + '_net_saved'] = sum(differences)
        comparisons[candidate + '_vs_' + reference] = comparison
    return totals, comparisons


def validate_slots(rows, requests, issues):
    expected = [(r['request_id'], repeat, arm) for r in requests for repeat in range(3)
                for arm in ARMS[(r['panel_ordinal'] + repeat) % 4:] + ARMS[:(r['panel_ordinal'] + repeat) % 4]]
    actual = [(r['request_id'], r['repeat'], r['method']) for r in rows]
    check(actual == expected and len(rows) == len(set(actual)) == 768,
          issues, 'Exactly 768 unique ordered evaluation slots required')
    return expected


def evaluation(root, requests, ordinary_rows, ordinary, trace_auditor, issues):
    directory = root / 'evaluation'
    rows, protocol = lines(directory / 'MEASUREMENTS.jsonl'), read(directory / 'PROTOCOL.json')
    status, saved = read(directory / 'STATUS.json'), read(directory / 'report/SUMMARY.json')
    expected_order = validate_slots(rows, requests, issues)
    checkpoints = protocol['checkpoints']
    check(protocol['arms'] == list(ARMS) and protocol['comparisons'] == [list(p) for p in PAIRS]
          and protocol['repeats'] == 3 and protocol['capture'] is False
          and protocol['required_arms'] == ['O'] and protocol['count_trajectory_repeat'] == 0
          and set(checkpoints) == set(ARMS) and len(set(checkpoints.values())) == 4,
          issues, 'Evaluation protocol/checkpoints')
    check(checkpoints['O'] == ordinary['checkpoint_path'] and all(
          checkpoints[a] == read(root / a / 'events/event_001/RESULT.json')['checkpoint_path'] for a in OBJECTIVES),
          issues, 'Evaluation retained/new checkpoint identities')
    by_id = {r['request_id']: r for r in requests}
    retained = {}
    check(len(ordinary_rows) == 384, issues, 'Original F/O measurements must have 384 rows')
    for request in requests:
        selected = [r for r in ordinary_rows if r['request_id'] == request['request_id']]
        check(len(selected) == 6 and {(r['method'], r['repeat']) for r in selected}
              == {(a, i) for a in ('F', 'O') for i in range(3)}, issues, 'Original F/O slots')
        retained[request['request_id']] = {}
        for arm in ('F', 'O'):
            group = [r for r in selected if r['method'] == arm]
            check(all(r['valid_measurement'] is True and not r['errors'] for r in group)
                  and len({r['verification_calls'] for r in group}) == 1, issues, 'Original F/O validity/repeat counts')
            retained[request['request_id']][arm] = group[0]['verification_calls']
    check(protocol['ordinary_counts'] == retained, issues, 'Retained first-event F/O count binding')
    checked, block_count = {}, 0
    for row in rows:
        before = len(issues)
        request, arm, repeat = by_id[row['request_id']], row['method'], row['repeat']
        label = f"{request['panel_ordinal']:04d}_r{repeat}_{arm}"
        record = read(directory / (label + '.json'))
        result = trace_auditor.trace_record(record, request, checkpoints[arm], issues, label)
        trace_extras(record, request, row, issues)
        for key in COUNTS + ('request_latency_s',):
            check(row[key] == result[key], issues, label + ': raw-derived ' + key)
        check(row['checkpoint_id'] == checkpoints[arm] and row['valid_measurement'] is True and not row['errors'],
              issues, label + ': execution validity/checkpoint')
        if arm == 'O':
            check(result['verification_calls'] == retained[row['request_id']][arm], issues, label + ': retained F/O calls')
        result['valid'] = len(issues) == before
        checked[row['request_id'], arm, repeat] = result
        block_count += result['verification_calls']
    valid_ids = []
    for request in requests:
        rid = request['request_id']
        group = [checked.get((rid, arm, repeat)) for arm in ARMS for repeat in range(3)]
        if all(row and row['valid'] for row in group):
            consistent = all(len({tuple(checked[rid, arm, repeat][k] for k in COUNTS) for repeat in range(3)}) == 1 for arm in ARMS)
            check(consistent, issues, rid + ': repeated counts differ')
            if consistent:
                valid_ids.append(rid)
    check(len(valid_ids) == 64, issues, 'Complete common valid cohort must contain 64 requests')
    totals, comparisons = independent_totals(rows, valid_ids)
    compare(saved, dict(planned_requests=64, valid_requests=len(valid_ids), invalid_or_incomplete_requests=64-len(valid_ids),
        returned_measurements=len(rows), valid_timed_generations=12*len(valid_ids),
        comparison_axes=dict(position_selection=['K_vs_O', 'A_vs_C'], loss_choice=['C_vs_O', 'A_vs_K']),
        timing_repeats_per_method_request=3, **totals, comparisons=comparisons), issues, 'Independent report')
    check('F' not in saved and 'primary_comparison' not in saved, issues, 'No fresh F or replacement primary comparison')
    compare(read(directory / 'report/HISTORICAL_F_COUNTS.json'), dict(
        source='retained first-event F measurements; not fresh latency', valid_requests=len(valid_ids),
        verification_calls=sum(retained[r]['F'] for r in valid_ids),
        per_request={r: retained[r]['F'] for r in valid_ids}), issues, 'Historical F common-cohort context')
    audit_csv_reports(directory, rows, valid_ids, issues)
    compare(status, dict(status='completed', planned_generations=768, attempted_generations=768,
        returned_generations=768, unknown_generation_attempts=0, unattempted_generations=0,
        qualification='passed', qualification_requests=8, optimizer_steps=0, feedback_collections=0), issues, 'Evaluation status')
    compare(read(directory / 'QUALIFICATION.json'), dict(status='passed', requests=8, generations=96,
        arms_with_native_equality=sorted(ARMS), performance_win_required=False), issues, 'Embedded evaluation gate')
    check(saved['execution'] == status, issues, 'Summary execution receipt')
    weights = []
    for arm in ARMS:
        receipt = read(directory / ('WEIGHTS_' + arm + '.json'))
        reply = receipt['native_reply']
        check(not receipt['errors'] and reply['status'] == 'passed' and reply['all_equal'] is True
              and reply['tensors'] == reply['source_tensor_count'] == 58 and reply['native_tensor_count'] == 43
              and reply['checkpoint_path'] == checkpoints[arm]
              and all(reply[k] is True for k in ('helper_matches_current_layers', 'helper_norm_matches_current_layers', 'fused_enabled')),
              issues, arm + ': native weight publication')
        weights.append(receipt)
    activations = lines(directory / 'ACTIVATIONS.jsonl')
    transitions, previous = [], None
    for rid, repeat, arm in expected_order:
        if arm != previous:
            transitions.append((by_id[rid]['panel_ordinal'], repeat, arm))
            previous = arm
    check([(a['panel_ordinal'], a['repeat'], a['method']) for a in activations] == transitions,
          issues, 'Native activation order/count')
    for receipt in activations:
        native, checkpoint = receipt['native_status'], checkpoints[receipt['method']]
        publication = native['publication']
        check(not receipt['errors'] and receipt['checkpoint_structure_valid'] is True
              and receipt['checkpoint_id'] == checkpoint and native['status'] == 'passed' and native['poisoned'] is None
              and native['capture_pending'] is False and native['cache_tokens'] == publication['cache_after_tokens'] == 0
              and native['attention_backend'] == 'fa3' and native['block_size'] == 16
              and publication['status'] == 'passed' and publication['checkpoint_path'] == checkpoint
              and all(publication[k] is True for k in ('helper_matches_current_layers', 'helper_norm_matches_current_layers', 'fused_enabled')),
              issues, 'Native publication/cache contract')
    check(close(sum(r['activation_s'] for r in activations), status['activation_s'])
          and close(sum(r['validation_s'] for r in weights), status['weight_validation_s']), issues, 'Evaluation setup cost sums')
    return dict(status=status['status'], raw_measurements=len(rows), native_blocks=block_count,
                valid_common_requests=len(valid_ids), totals=totals, comparisons=comparisons,
                historical_F_calls=sum(retained[r]['F'] for r in valid_ids),
                publication_receipts=len(activations), activation_s=status['activation_s'],
                weight_validation_s=status['weight_validation_s'], evaluation_wall_s=status['evaluation_wall_s'])


def csv_scalar(actual, expected):
    if type(expected) in (int, float):
        try:
            return close(float(actual), expected)
        except (TypeError, ValueError):
            return False
    return actual == ('' if expected is None else str(expected))


def audit_csv_reports(directory, rows, valid_ids, issues):
    def table(name):
        with (directory / 'report' / name).open() as stream:
            return list(csv.DictReader(stream))
    per_request = table('PER_REQUEST.csv')
    check([r['request_id'] for r in per_request if r['valid'] == 'True'] == valid_ids
          and len(per_request) == 64 and all(r['complete'] == r['valid'] == 'True' and not r['errors'] for r in per_request),
          issues, 'CSV common valid request cohort')
    for result in per_request:
        rid = result['request_id']
        for arm in ARMS:
            group = [row for row in rows if row['request_id'] == rid and row['method'] == arm]
            primary = next(row for row in group if row['repeat'] == 0)
            values = {**{key: primary[key] for key in COUNTS},
                'mean_request_latency_s': sum(row['request_latency_s'] for row in group)/3,
                'calls_per_1000_output_tokens': 1000*primary['verification_calls']/primary['output_tokens'],
                'accepted_draft_tokens_per_call': primary['accepted_draft_tokens']/primary['verification_calls'],
                'committed_progress_per_call': primary['committed_progress']/primary['verification_calls']}
            for name, value in values.items():
                check(csv_scalar(result[arm + '_' + name], value), issues, rid + ': per-request CSV ' + arm + '_' + name)
        for candidate, reference in PAIRS:
            label = candidate + '_vs_' + reference
            for metric, key, reduction in (('call', 'verification_calls', 'call_reduction_pct'),
                    ('latency', 'mean_request_latency_s', 'request_time_reduction_pct')):
                before, after = float(result[reference + '_' + key]), float(result[candidate + '_' + key])
                expected = 'improved' if after < before else 'regressed' if after > before else 'tied'
                check(result[label + '_' + metric + '_direction'] == expected
                      and csv_scalar(result[label + '_' + reduction], 100*(1-after/before)),
                      issues, rid + ': per-request CSV ' + label + '/' + metric)
    checked = table('ALL_REPEATS.csv')
    check(len(checked) == len(rows) == 768, issues, 'CSV all-repeat completeness')
    for row, saved in zip(rows, checked):
        check(all(csv_scalar(saved[key], row[key]) for key in row)
              and saved['report_valid'] == 'True' and saved['report_errors'] == '[]', issues, 'CSV raw-measurement binding')
    totals = table('REPEAT_TOTALS.csv')
    check([(int(row['repeat']), row['method']) for row in totals] == [(r, a) for r in range(3) for a in ARMS],
          issues, 'CSV per-repeat aggregate slots')
    for row in totals:
        group = [r for r in rows if r['repeat'] == int(row['repeat']) and r['method'] == row['method'] and r['request_id'] in valid_ids]
        check(csv_scalar(row['requests'], len(group)) and csv_scalar(row['verification_calls'], sum(r['verification_calls'] for r in group))
              and csv_scalar(row['request_latency_s'], sum(r['request_latency_s'] for r in group)), issues, 'CSV repeat totals')


def costs(root, issues):
    receipt, arms = read(root / 'COSTS.json'), {}
    qualification_wall = read(root / 'qualification/RESULT.json')['wall_s']
    check(close(receipt['qualification_driver_wall_s'], qualification_wall)
          and receipt['qualification_wall_s'] >= qualification_wall, issues, 'Full versus nested qualification wall cost')
    evaluation_wall = read(root / 'evaluation/STATUS.json')['evaluation_wall_s']
    counted = receipt['qualification_wall_s'] + receipt['backend_loading_s'] + evaluation_wall
    for arm in OBJECTIVES:
        value = receipt[arm]
        event = read(root / arm / 'events/event_001/RESULT.json')
        check(value == read(root / (arm + '_COSTS.json')), issues, arm + ': duplicate costs')
        compare(value, event['costs'], issues, arm + ': event costs')
        check(value['objective_work'] == event['objective_work'] and value['selection_totals'] == event['selection_totals']
              and value['new_feedback_generations'] == value['new_teacher_preparation_s'] == 0,
              issues, arm + ': reused feedback and nested objective costs')
        check(close(value['training_s'], sum(row['training_s'] for row in event['records']))
              and value['event_wall_s'] + 1e-6 >= sum(value[key] for key in ('training_s', 'checkpoint_save_s', 'export_s')),
              issues, arm + ': presentation/event cost sums')
        check(close(value['initialization_s'], read(root / (arm + '_INITIAL.json'))['controlled_initialization']['wall_s']),
              issues, arm + ': startup costs')
        phase_keys = ('loading_s', 'initialization_s', 'event_wall_s', 'training_s', 'checkpoint_save_s', 'export_s')
        check(all(type(value[k]) in (int, float) and math.isfinite(value[k]) and value[k] >= 0 for k in phase_keys),
              issues, arm + ': finite nonnegative phase costs')
        counted += sum(value[k] for k in ('loading_s', 'initialization_s', 'event_wall_s'))
        arms[arm] = value
    check(receipt['actual_campaign_wall_s'] + 1e-6 >= counted, issues, 'Disjoint phase costs exceed campaign wall')
    return dict(arms=arms, qualification_wall_s=receipt['qualification_wall_s'],
        qualification_driver_wall_s=qualification_wall, backend_loading_s=receipt['backend_loading_s'],
        evaluation_wall_s=evaluation_wall, nonoverlapping_phase_sum_s=counted,
        actual_campaign_wall_s=receipt['actual_campaign_wall_s'],
        note='Selection/teacher timers are nested inside training/event; activation/weight checks inside evaluation. Historical feedback costs are not zero. No adaptation-inclusive speedup inferred.')


def provenance(root, args, issues):
    q, seal, postflight = read(root / 'qualification/RESULT.json'), read(args.seal), read(args.postflight)
    check(q['input_sha256'][q['ordinary_receipt']] == digest(args.ordinary_receipt), issues, 'Qualification retained O receipt hash')
    check(q['input_sha256'][q['config']] == digest(root / 'config.yaml'), issues, 'Qualification copied configuration hash')
    bindings = dict(seal['dependency_sha256'])
    for name, value in seal['execution_sha256'].items():
        for path in q['input_sha256']:
            if path.endswith('/' + name):
                bindings[path] = value
    for path, value in q['input_sha256'].items():
        if path not in (q['ordinary_receipt'], q['config']):
            check(bindings.get(path) == value, issues, 'Qualification input/source seal: ' + path)
    check(postflight['status'] == 'passed' and not postflight['errors'] and postflight['supervisor_exit_code'] == 0,
          issues, 'Postflight preservation AND supervisor completion')
    for category in ('source_sha256', 'execution_sha256', 'dependency_sha256'):
        check(postflight['checked'][category] == len(seal[category]), issues, 'Postflight seal coverage: ' + category)
    parent = Path(args.postflight).parent
    preflight, retained = read(parent / 'PREFLIGHT.json'), read(parent / 'RETAINED_ARTIFACTS.json')
    check(preflight['status'] == 'passed' and preflight['original_inputs_unchanged'] is True
          and preflight['storage']['admitted'] is True
          and postflight['checked']['retained_artifacts'] == preflight['retained_artifacts'] == len(retained),
          issues, 'Preflight/postflight retained artifact coverage')
    check(retained[q['ordinary_receipt']]['sha256'] == digest(args.ordinary_receipt), issues, 'Retained O artifact identity')
    return dict(status=postflight['status'], postflight=postflight, seal_sha256=digest(args.seal),
        postflight_sha256=digest(args.postflight), preflight_sha256=digest(parent / 'PREFLIGHT.json'),
        retained_artifacts=len(retained), evidence_scope='Remote provenance assertions from the retained preflight/postflight; no remote paths accessed locally.')


def validate_scope(manifest, ordinary, issues):
    requests, training_requests = manifest['eval_requests'], manifest['train_requests']
    check(len(requests) == len({r['request_id'] for r in requests}) == 64
          and [r['panel_ordinal'] for r in requests] == list(range(64))
          and all(r['max_new_tokens'] == 2048 for r in requests), issues, 'Sealed64-request development panel')
    compare(ordinary, dict(status='completed', objective='osd', passes=2, optimizer_steps=128,
        response_presentations=256, start_step=0, end_step=128), issues, 'Original O event scope')
    identities = {r['record_id'] for r in ordinary['features']}
    check(len(training_requests) == len({r['request_id'] for r in training_requests}) == 128
          and len(ordinary['features']) == len(identities) == len({r['path'] for r in ordinary['features']}) == 128
          and identities == {r['request_id'] for r in training_requests}
          and not identities.intersection(r['request_id'] for r in requests), issues, 'Distinct128 retained responses versus64 panel requests')
    check(len(ordinary['records']) == 256, issues, 'Original256 presentation schedule')
    for pass_index in range(2):
        rows = ordinary['records'][128*pass_index:128*(pass_index+1)]
        check(len(rows) == len({r['record_id'] for r in rows}) == 128
              and {r['record_id'] for r in rows} == identities and all(r['pass'] == pass_index for r in rows),
              issues, 'Original full feedback pass ' + str(pass_index))
    for index, row in enumerate(ordinary['records']):
        check(row['optimizer_stepped'] == bool(index % 2) and row['optimizer_step_after'] == (index+1)//2
              and len(row['sampled_anchors']) == 1, issues, 'Original accumulation/single anchor draw ' + str(index))


def audit(args):
    root, issues = Path(args.campaign), []
    ordinary, manifest = read(args.ordinary_receipt), read(args.manifest)
    validate_scope(manifest, ordinary, issues)
    check(read(root / 'MANIFEST.json') == manifest, issues, 'Campaign manifest copy')
    status = read(root / 'STATUS.json')
    compare(status, dict(status='completed', phase='finished', main_optimizer_steps=384,
        main_response_presentations=768, main_steps_remaining=0, unknown_arm_counters=None,
        qualification_status='passed', planned_evaluation_generations=768,
        new_feedback_generations=0, new_teacher_preparation_forwards=0, automatic_retry=False), issues, 'Campaign completion/scope')
    check(not (root / 'O').exists(), issues, 'No new O training directory')
    prior = load_trace_auditor(args.trace_auditor)
    result = dict(campaign=str(root.resolve()), auditor_sha256=digest(__file__),
        trace_auditor=str(Path(args.trace_auditor).resolve()), trace_auditor_sha256=digest(args.trace_auditor),
        scope='Local compact evidence only; no remote operations, tensors, model execution or new generations.')
    for name, operation in (
        ('qualification', lambda: qualification(root, ordinary, issues)),
        ('training', lambda: training(root, ordinary, issues)),
        ('evaluation', lambda: evaluation(root, manifest['eval_requests'], lines(args.ordinary_measurements), ordinary, prior, issues)),
        ('costs', lambda: costs(root, issues)),
        ('provenance', lambda: provenance(root, args, issues)),
    ):
        try:
            result[name] = operation()
        except (OSError, KeyError, TypeError, ValueError, IndexError, ZeroDivisionError, StopIteration) as error:
            issues.append(name + ': incomplete or malformed evidence: ' + repr(error))
            result[name] = dict(status='unavailable_or_malformed', error=repr(error))
    try:
        check(status['evaluation'] == read(root / 'evaluation/STATUS.json'), issues, 'Campaign duplicate evaluation receipt')
        for arm in OBJECTIVES:
            check(status['arm_counters'][arm] == read(root / (arm + '_CLOSED.json')), issues, arm + ': campaign duplicate counters')
    except (OSError, KeyError, TypeError, ValueError) as error:
        issues.append('Duplicate completion evidence: ' + repr(error))
    result.update(status='passed' if not issues else 'failed', issues=issues,
        interpretation='Passing establishes compact-evidence consistency and completed bounded work, not efficacy, independent GPU reproduction or historical O full-trajectory equality. F counts are historical context only; no fresh F latency comparison.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('campaign', 'manifest', 'ordinary-receipt', 'ordinary-measurements', 'seal', 'postflight', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--trace-auditor', default=str(DEFAULT_TRACE_AUDITOR))
    args = parser.parse_args()
    try:
        result = audit(args)
    except (OSError, KeyError, TypeError, ValueError, IndexError, ZeroDivisionError, StopIteration) as error:
        result = dict(status='failed', issues=['Required inputs unavailable or malformed: ' + repr(error)])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status=result['status'], issues=len(result['issues']), output=str(output))))
    return int(result['status'] != 'passed')


if __name__ == '__main__':
    raise SystemExit(main())
