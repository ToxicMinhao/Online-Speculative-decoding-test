"""Independent local audit of retrieved R campaign JSON evidence (no model imports).

This checks completeness/correctness, not whether R improves performance. A valid
negative result passes. Missing evidence fails closed. Tensor equality is checked
through retained exact-comparison receipts, not by loading state tensors again.
The retained independent longest-prefix auditor is reused and its hash recorded.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

ARMS = ('F', 'O', 'P', 'R')
PAIRS = (('R', 'O'), ('R', 'P'), ('R', 'F'), ('O', 'F'), ('P', 'O'))
COUNTS = ('verification_calls', 'accepted_draft_tokens', 'committed_progress', 'output_tokens')
SUMS = ('kl_loss_numerator', 'auxiliary_loss_numerator', 'loss_denominator',
        'protected_active_positions', 'protected_positions', 'reference_correct_positions',
        'reference_correct_current_incorrect_positions')
SCHEDULE = ('record_id', 'pass', 'sampled_anchors', 'optimizer_stepped', 'optimizer_step_after')
WORK = dict(current_training_forward=10, reference_training_forward=4,
            detached_comparison_forward=4, temporary_optimizer_step=5,
            backward=10, detached_logit_gradient=20)
PLAN = dict(temporary_optimizer_steps=5, response_presentations=10,
            current_training_forwards=10, reference_training_forwards=4,
            detached_comparison_forwards=4, backwards=10,
            detached_logit_gradient_calls=20, generations=0,
            main_optimizer_steps=0, teacher_model_forwards=0)
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
    check(result['status'] == 'passed' and result['plan'] == PLAN, issues, 'Qualification status/plan')
    check(result['generations'] == result['main_optimizer_steps'] == 0
          and result['main_campaign_started'] is False and result['baseline_tolerance'] == 'exact equality',
          issues, 'Qualification scope/tolerance')
    counts = {}
    for kind, maximum in WORK.items():
        counts[kind] = {state: sum(row['kind'] == kind and row.get('status') == state for row in work)
                        for state in ('started', 'completed', 'failed')}
        check(counts[kind] == dict(started=maximum, completed=maximum, failed=0),
              issues, 'Qualification budget/completion: ' + kind)
        for state, suffix in (('started', 'attempted'), ('completed', 'completed')):
            check(result['work_counts'].get(kind + '_' + suffix, 0) == counts[kind][state],
                  issues, 'Qualification recorded count: ' + kind + '_' + suffix)
    presentations = [row for row in work if row['kind'] == 'presentation_complete']
    check(len(presentations) == result['work_counts']['response_presentations'] == 10,
          issues, 'Qualification presentation count')
    check(result['work_counts']['current_draft_forwards_observed'] == 10
          and result['work_counts']['reference_draft_forwards_observed'] == 4,
          issues, 'Observed qualification forwards')
    trials = result['trials']
    check([t['label'] for t in trials] == ['warmup', 'Oa', 'R0', 'R1'], issues, 'Qualification trial order')
    gates, by_label = [], {t['label']: t for t in trials}
    for label, objective, coefficient, n in (('warmup', 'osd', 0, 2), ('Oa', 'osd', 0, 2),
            ('R0', 'osd_reference_margin', 0, 2), ('R1', 'osd_reference_margin', 1, 4)):
        trial = by_label[label]
        check(trial == read(directory / label / 'TRIAL.json'), issues, label + ': duplicate trial receipt')
        check(trial['objective'] == objective and trial['ranking_lambda'] == coefficient
              and trial['status'] == ('discarded_warmup' if label == 'warmup' else 'passed')
              and trial['frozen_target_version_layout_unchanged'] is True, issues, label + ': trial contract')
        check(len(trial['records']) == n, issues, label + ': presentation budget')
        for index, row in enumerate(trial['records']):
            check(all(row[key] == ordinary['records'][index][key] for key in SCHEDULE),
                  issues, label + ': exact retained schedule at ' + str(index))
        shuffles = [r for r in work if r['kind'] == 'full_first_pass_shuffle' and r['trial'] == label]
        check(len(shuffles) == 1 and shuffles[0]['n'] == 128 and shuffles[0]['exact_equal'] is True
              and shuffles[0]['record_ids'] == [r['record_id'] for r in ordinary['records'][:128]],
              issues, label + ': full retained first-pass shuffle')
        logged = [r for r in presentations if r['trial'] == label]
        check([r['presentation'] for r in logged] == list(range(n))
              and [r['row'] for r in logged] == trial['records'], issues, label + ': durable presentation binding')
        snapshots = [r for r in work if r['kind'] == 'gradient_snapshot' and r['trial'] == label]
        check([{k: v for k, v in r.items() if k not in ('kind', 'trial', 'presentation')} for r in snapshots] == trial['gradient_snapshots'],
              issues, label + ': durable gradient snapshot binding')
        for kind in WORK:
            expected = (list(range(n)) if kind in ('current_training_forward', 'backward') else
                        list(range(1, n, 2)) if kind == 'temporary_optimizer_step' else
                        list(range(n)) if kind == 'reference_training_forward' and label == 'R1' else
                        [None] * 4 if kind == 'detached_comparison_forward' and label == 'R1' else
                        [i for i in range(n) for _ in range(5)] if kind == 'detached_logit_gradient' and label == 'R1' else [])
            for state in ('started', 'completed'):
                actual = [r['presentation'] for r in work if r['kind'] == kind
                          and r['trial'] == label and r.get('status') == state]
                check(actual == expected, issues, label + ': work/presentation binding ' + kind + '/' + state)
        gates.append(exact_gate(directory / label / 'initial_vs_pinned.json', issues))
    for filename in ('initial_vs_Oa', 'after_backward_0_vs_Oa', 'after_backward_1_vs_Oa',
                     'pre_optimizer_vs_Oa', 'final_vs_Oa', 'records_vs_Oa'):
        gates.append(exact_gate(directory / 'R0' / (filename + '.json'), issues))
    gates.append(exact_gate(directory / 'R1/first_window_rng_vs_Oa.json', issues))
    check(by_label['R0']['exact_baseline_compatibility'] is True, issues, 'R0 exact compatibility')
    for index in range(2):
        check(by_label['Oa']['records'][index]['metrics'] == ordinary['records'][index]['metrics']
              == by_label['R0']['records'][index]['metrics'], issues, 'Oa/R0 retained scalar metrics')
    active = by_label['R1']
    check(active['reference_rng_isolation'] is True and active['post_update_presentations'] == 2,
          issues, 'R1 post-update coverage/reference RNG')
    expected_stages = ['after_backward_0', 'after_backward_1', 'pre_optimizer_step_1',
                       'after_backward_2', 'after_backward_3', 'pre_optimizer_step_2']
    check([s['stage'] for s in active['gradient_snapshots']] == expected_stages
          and len({s['path'] for s in active['gradient_snapshots']}) == 6
          and all(s['bytes'] > 0 for s in active['gradient_snapshots']), issues, 'R1 gradient evidence boundaries')
    for kind, fields in (('prefix_checks', ('prefix_equal', 'reference_own_rivals_equal', 'reference_rng_equal')),
            ('ranking_gradient_checks', ('exact_original_P_gradient', 'reference_detached',
                                        'equality_value_and_gradient_zero', 'synthetic_changed_rivals_checked'))):
        checks = active[kind]
        check([r['presentation'] for r in checks] == list(range(4))
              and all(all(r[key] is True for key in fields) for r in checks), issues, 'R1 ' + kind)
    detached = [r for r in work if r['kind'] == 'detached_block_check']
    check(len(detached) == 2 and [r['record_id'] for r in detached]
          == [r['record_id'] for r in ordinary['records'][:2]]
          and all(r['trial'] == 'R1' and r['presentation'] is None
                  and r['exact_hidden_equality'] is True for r in detached),
          issues, 'R1 two independent full-block alignment receipts')
    logging = [r for r in work if r['kind'] == 'reference_margin_loss_logging_check']
    check(len(logging) == 4 and all(all(r[k] is True for k in
          ('exact_chunk_terms', 'exact_denominator', 'exact_logging')) for r in logging),
          issues, 'R1 direct loss/diagnostic checks')
    storage = result['storage']
    check(storage['required_bytes'] == 38 * storage['draft_bytes'] + 10 * 2**30
          and storage['free_bytes_at_admission'] >= storage['required_bytes'], issues, 'Qualification storage admission')
    return dict(status=result['status'], work=counts, presentation_count=len(presentations),
                exact_gates=gates, wall_s=result['wall_s'], storage=storage,
                observed_active_rows=[r['observed_active_rows'] for r in active['ranking_gradient_checks']],
                historical_O_limit=result['historical_O_limit'])


def diagnostic_totals(rows):
    boundary = [row for row in rows if row['optimizer_stepped']]
    sums = {key: sum(row['metrics'][key] for row in boundary) for key in SUMS}
    ratios = {name: sums[numerator] / sums[denominator] if sums[denominator] else None
              for name, numerator, denominator in (
                  ('protected_activation_rate', 'protected_active_positions', 'protected_positions'),
                  ('reference_correct_to_incorrect_rate', 'reference_correct_current_incorrect_positions', 'reference_correct_positions'),
                  ('pooled_weighted_forward_kl', 'kl_loss_numerator', 'loss_denominator'),
                  ('pooled_weighted_auxiliary_loss', 'auxiliary_loss_numerator', 'loss_denominator'))}
    return dict(boundary_rows=len(boundary), sums=sums, ratios=ratios)


def training(root, ordinary, issues):
    receipt = read(root / 'R/events/event_001/RESULT.json')
    compare(receipt, dict(status='completed', objective='osd_reference_margin', reduction='response_mean',
        passes=2, optimizer_steps=128, response_presentations=256, start_step=0, end_step=128), issues, 'R receipt')
    check(len(ordinary['features']) == 128 and len(ordinary['records']) == 256
          and receipt['features'] == ordinary['features'] and len(receipt['records']) == 256,
          issues, 'R original feature pool/presentation count')
    anchors, positions = 0, 0
    features = {r['record_id']: r for r in ordinary['features']}
    for index, (row, original) in enumerate(zip(receipt['records'], ordinary['records'])):
        check(all(row[key] == original[key] for key in SCHEDULE + ('learning_rate',)),
              issues, 'R original schedule/LR: ' + str(index))
        feature = features[row['record_id']]
        for draw in row['sampled_anchors']:
            check(len(draw['positions']) == len(draw['keep_mask']), issues, 'R anchor batch shape')
            for sampled, keep in zip(draw['positions'], draw['keep_mask']):
                check(len(sampled) == len(keep), issues, 'R anchor keep shape')
                for anchor, active in zip(sampled, keep):
                    if active:
                        anchors += 1
                        check(feature['prompt_length'] <= anchor < feature['length'] - 1,
                              issues, 'R ineligible anchor')
                        positions += sum(feature['prompt_length'] <= anchor + j < feature['length'] for j in range(1, 16))
        m = row['metrics']
        check(all(type(v) in (int, float) and math.isfinite(v) for v in m.values()), issues, 'R finite metrics')
        check(0 <= m['protected_active_positions'] <= m['protected_positions'] <= m['supervised_position_exposure']
              and 0 <= m['reference_correct_current_incorrect_positions'] <= m['reference_correct_positions']
              <= m['supervised_position_exposure'] and m['auxiliary_loss_numerator'] >= 0
              and m['loss_denominator'] > 0, issues, 'R diagnostic bounds')
        check(close(m['protected_positions'], m['reference_prefix_positions']), issues, 'R protected/prefix count')
    boundary = [r for r in receipt['records'] if r['optimizer_stepped']]
    exposure = {key: sum(r['metrics'][key] for r in boundary) for key in
                ('supervised_position_exposure', 'reference_prefix_positions', 'reference_prefix_weight', 'ranking_coefficient_weight')}
    check(len(boundary) == 128 and exposure['supervised_position_exposure'] == positions,
          issues, 'R boundary count/direct exposure')
    check(math.isclose(exposure['reference_prefix_weight'], exposure['ranking_coefficient_weight'],
                       rel_tol=1.3e-6, abs_tol=1e-5), issues, 'R P allocation coefficient mass')
    initial, final = read(root / 'R_INITIAL.json'), read(root / 'R_CLOSED.json')
    compare(initial, dict(global_step=0, presentations=0, accumulation_remainder=0, completed_events=[],
                         closed=False, failed=False, objective='osd_reference_margin', reduction='response_mean'), issues, 'R initial')
    compare(initial['controlled_initialization'], dict(exact_pinned_initial_state=True,
        full_rng_restored_after_construction=True, inherited_warmup_state=False, reference_created=False), issues, 'R startup')
    compare(final, dict(global_step=128, presentations=256, accumulation_remainder=0,
                       completed_events=['event_001'], closed=True, failed=False), issues, 'R closed')
    gate = exact_gate(root / 'R/initialization/initial_vs_pinned.json', issues)
    costs = read(root / 'R_COSTS.json')
    diagnostics = diagnostic_totals(receipt['records'])
    compare(costs['diagnostics'], diagnostics, issues, 'R boundary diagnostics')
    compare(costs['exposure'], dict(anchors=anchors, optimizer_boundary_metric_sums=exposure), issues, 'R exposure')
    return dict(status=receipt['status'], optimizer_steps=len(boundary), response_presentations=len(receipt['records']),
                anchors=anchors, reconstructed_supervised_positions=positions, exposure=exposure,
                diagnostics=diagnostics, initial_gate=gate, objective_work=receipt['objective_work'], costs=receipt['costs'])


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


def evaluation(root, requests, ordinary_rows, prefix_receipt, ordinary, trace_auditor, issues):
    directory = root / 'evaluation'
    rows, protocol = lines(directory / 'MEASUREMENTS.jsonl'), read(directory / 'PROTOCOL.json')
    status, saved = read(directory / 'STATUS.json'), read(directory / 'report/SUMMARY.json')
    expected_order = validate_slots(rows, requests, issues)
    checkpoints = protocol['checkpoints']
    check(protocol['arms'] == list(ARMS) and protocol['comparisons'] == [list(p) for p in PAIRS]
          and protocol['repeats'] == 3 and protocol['capture'] is False
          and set(checkpoints) == set(ARMS) and len(set(checkpoints.values())) == 4,
          issues, 'Evaluation protocol/checkpoints')
    check(checkpoints['O'] == ordinary['checkpoint_path'] and checkpoints['P'] == prefix_receipt['checkpoint_path']
          and checkpoints['R'] == read(root / 'R/events/event_001/RESULT.json')['checkpoint_path'],
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
        if arm in ('F', 'O'):
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
        returned_measurements=len(rows), valid_timed_generations=12*len(valid_ids), primary_comparison='R_vs_O',
        timing_repeats_per_method_request=3, **totals, comparisons=comparisons), issues, 'Independent report')
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
                publication_receipts=len(activations), activation_s=status['activation_s'],
                weight_validation_s=status['weight_validation_s'], evaluation_wall_s=status['evaluation_wall_s'])


def costs(root, issues):
    costs, event = read(root / 'COSTS.json'), read(root / 'R/events/event_001/RESULT.json')
    r = costs['R']
    check(r == read(root / 'R_COSTS.json'), issues, 'Duplicate R costs')
    compare(r, event['costs'], issues, 'R event costs')
    check(r['objective_work'] == event['objective_work'] and r['teacher_preparation_s'] == r['new_feedback_generations'] == 0
          and r['reference_cost_is_nested_in_event_wall'] is True, issues, 'R feedback/reference costs')
    check(close(r['training_s'], sum(row['training_s'] for row in event['records']))
          and r['event_wall_s'] + 1e-6 >= sum(r[k] for k in ('training_s', 'checkpoint_save_s', 'export_s')),
          issues, 'R presentation/event wall sums')
    check(close(r['initialization_control_s'], read(root / 'R_INITIAL.json')['controlled_initialization']['wall_s'])
          and close(costs['qualification_wall_s'], read(root / 'qualification/RESULT.json')['wall_s']), issues, 'Startup costs')
    counted = costs['qualification_wall_s'] + costs['backend_loading_s'] + read(root / 'evaluation/STATUS.json')['evaluation_wall_s']
    counted += sum(r[k] for k in ('loading_s', 'initialization_control_s', 'event_wall_s'))
    check(costs['actual_campaign_wall_s'] + 1e-6 >= counted, issues, 'Disjoint costs exceed campaign wall')
    check(all(type(r[k]) in (int, float) and math.isfinite(r[k]) and r[k] >= 0 for k in
          ('loading_s', 'initialization_control_s', 'event_wall_s', 'training_s', 'checkpoint_save_s', 'export_s')),
          issues, 'Finite nonnegative phase costs')
    return dict(R=r, qualification_wall_s=costs['qualification_wall_s'], backend_loading_s=costs['backend_loading_s'],
                evaluation_wall_s=read(root / 'evaluation/STATUS.json')['evaluation_wall_s'],
                nonoverlapping_phase_sum_s=counted, actual_campaign_wall_s=costs['actual_campaign_wall_s'],
                note='Reference work is nested in the R event; activation/weight checks are nested in evaluation. No net online speedup inference.')


def provenance(root, args, issues):
    result = {'status': 'not_supplied', 'note': 'Pass --seal and --postflight for remote preservation receipt checks; this auditor never accesses remote paths.'}
    q = read(root / 'qualification/RESULT.json')
    check(q['input_sha256'][q['ordinary_receipt']] == digest(args.ordinary_receipt), issues, 'Qualification ordinary receipt hash')
    check(q['input_sha256'][q['config']] == digest(root / 'config.yaml'), issues, 'Qualification copied config hash')
    if args.seal:
        seal = read(args.seal)
        bindings = {**seal['dependency_sha256']}
        for name, value in seal['execution_sha256'].items():
            matches = [path for path in q['input_sha256'] if path.endswith('/' + name)]
            for path in matches:
                bindings[path] = value
        for path, value in q['input_sha256'].items():
            if path not in (q['ordinary_receipt'], q['config']):
                check(bindings.get(path) == value, issues, 'Qualification input/seal binding: ' + path)
        result.update(seal_sha256=digest(args.seal))
    if args.postflight:
        postflight = read(args.postflight)
        check(postflight['status'] == 'passed' and not postflight['errors']
              and postflight['supervisor_exit_code'] == 0, issues, 'Postflight preservation/completion')
        if args.seal:
            for category in ('source_sha256', 'execution_sha256', 'dependency_sha256'):
                check(postflight['checked'][category] == len(seal[category]), issues, 'Postflight seal coverage: ' + category)
        parent = Path(args.postflight).parent
        preflight, retained = read(parent / 'PREFLIGHT.json'), read(parent / 'RETAINED_ARTIFACTS.json')
        check(preflight['status'] == 'passed' and preflight['original_inputs_unchanged'] is True
              and preflight['storage']['admitted'] is True
              and postflight['checked']['retained_artifacts'] == preflight['retained_artifacts'] == len(retained),
              issues, 'Preflight/postflight retained artifact coverage')
        check(retained[q['ordinary_receipt']]['sha256'] == digest(args.ordinary_receipt),
              issues, 'Retained O receipt identity')
        prefix_export = read(args.prefix_receipt)['checkpoint_path']
        prefix_receipt_path = str(Path(prefix_export).parent / 'RESULT.json')
        check(retained[prefix_receipt_path]['sha256'] == digest(args.prefix_receipt),
              issues, 'Retained P receipt identity')
        result.update(status=postflight['status'], postflight=postflight, postflight_sha256=digest(args.postflight),
                      preflight_sha256=digest(parent / 'PREFLIGHT.json'), retained_artifacts=len(retained))
    return result


def audit(args):
    root, issues = Path(args.campaign), []
    ordinary, manifest, prefix = read(args.ordinary_receipt), read(args.manifest), read(args.prefix_receipt)
    requests = manifest['eval_requests']
    check(len(requests) == 64 and len({r['request_id'] for r in requests}) == 64
          and [r['panel_ordinal'] for r in requests] == list(range(64))
          and all(r['max_new_tokens'] == 2048 for r in requests), issues, 'Sealed 64-request panel')
    check(read(root / 'MANIFEST.json') == manifest, issues, 'Campaign manifest copy')
    status = read(root / 'STATUS.json')
    compare(status, dict(status='completed', phase='finished', main_optimizer_steps=128,
        main_response_presentations=256, new_feedback_generations=0, new_teacher_preparation_forwards=0,
        automatic_retry=False), issues, 'Campaign completion/scope')
    prior = load_trace_auditor(args.trace_auditor)
    result = {'campaign': str(root.resolve()), 'trace_auditor': str(Path(args.trace_auditor).resolve()),
              'trace_auditor_sha256': digest(args.trace_auditor), 'auditor_sha256': digest(__file__),
              'scope': 'Local compact evidence; no remote operations, tensors, model execution or new generations.'}
    sections = (
        ('qualification', lambda: qualification(root, ordinary, issues)),
        ('training', lambda: training(root, ordinary, issues)),
        ('evaluation', lambda: evaluation(root, requests, lines(args.ordinary_measurements), prefix, ordinary, prior, issues)),
        ('costs', lambda: costs(root, issues)),
        ('provenance', lambda: provenance(root, args, issues)),
    )
    for name, operation in sections:
        try:
            result[name] = operation()
        except (OSError, KeyError, TypeError, ValueError, IndexError) as error:
            issues.append(name + ': incomplete or malformed evidence: ' + repr(error))
            result[name] = {'status': 'unavailable_or_malformed', 'error': repr(error)}
    for key, path in (('qualification', 'qualification/RESULT.json'), ('evaluation', 'evaluation/STATUS.json')):
        try:
            check(status.get(key) == read(root / path), issues, 'Campaign duplicate ' + key + ' receipt')
        except (OSError, ValueError) as error:
            issues.append('Campaign duplicate ' + key + ': ' + repr(error))
    result.update(status='passed' if not issues else 'failed', issues=issues,
                  interpretation='Passing establishes compact-evidence consistency and completion, not an R gain, GPU rerun, or historical O full-trajectory equivalence.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('campaign', 'manifest', 'ordinary-receipt', 'ordinary-measurements', 'prefix-receipt', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--trace-auditor', default=str(DEFAULT_TRACE_AUDITOR))
    parser.add_argument('--seal')
    parser.add_argument('--postflight')
    args = parser.parse_args()
    try:
        result = audit(args)
    except (OSError, KeyError, TypeError, ValueError, IndexError) as error:
        result = dict(status='failed', issues=['Required inputs unavailable or malformed: ' + repr(error)])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status=result['status'], issues=len(result['issues']), output=str(output))))
    return int(result['status'] != 'passed')


if __name__ == '__main__':
    raise SystemExit(main())
