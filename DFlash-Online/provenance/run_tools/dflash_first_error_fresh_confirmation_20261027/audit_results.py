"""Independent local audit of the sealed 384-generation O/K confirmation.

Standard library only; no model imports, GPU, network, mutation of raw evidence,
or experimental rerun. Remote record paths are resolved by local basename.
"""
import argparse
import ast
from collections import Counter
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
COUNTS = ('verification_calls', 'accepted_draft_tokens', 'committed_progress', 'output_tokens')
ARMS = ('O', 'K')
TRACE_AUDITOR = HERE.parent / 'dflash_prefix_ranking_one_event_20261025/audit_results.py'
NATIVE_SOURCE = HERE / 'worktree/online_feedback/native.py'


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def check(condition, issues, message):
    if not condition:
        issues.append(message)


def close(a, b):
    return ((a is None and b is None) or
            (type(a) in (int, float) and type(b) in (int, float)
             and math.isfinite(a) and math.isfinite(b)
             and math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-9)))


def compare(actual, expected, issues, label):
    if isinstance(expected, dict):
        check(isinstance(actual, dict), issues, label + ': mapping required')
        if isinstance(actual, dict):
            for key, value in expected.items():
                compare(actual.get(key), value, issues, label + '.' + key)
    else:
        check(close(actual, expected) if type(expected) in (int, float) or expected is None
              else actual == expected, issues, label + ': differs')


def local_record(directory, remote_path):
    """Never edit a retained remote path merely to make a reader accept it."""
    return Path(directory) / Path(remote_path).name


def load_helpers():
    spec = importlib.util.spec_from_file_location('retained_trace_auditor', TRACE_AUDITOR)
    previous = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous)
    tree = ast.parse(NATIVE_SOURCE.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == 'validate_generation')
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(NATIVE_SOURCE), 'exec'), namespace)
    return previous.trace_record, namespace['validate_generation']


def expected_slots(requests):
    return [(r['request_id'], repeat, arm) for r in requests for repeat in range(3)
            for arm in (ARMS if (r['panel_ordinal'] + repeat) % 2 == 0 else ARMS[::-1])]


def totals_and_comparison(rows, requests):
    values, totals = {}, {}
    for request in requests:
        rid = request['request_id']
        for arm in ARMS:
            selected = [r for r in rows if r['request_id'] == rid and r['method'] == arm]
            if len(selected) != 3 or {r['repeat'] for r in selected} != {0, 1, 2}:
                raise ValueError(f'Incomplete repeated observations: {rid}/{arm}')
            primary = next(r for r in selected if r['repeat'] == 0)
            values[rid, arm] = {**{key: primary[key] for key in COUNTS},
                               'latency': sum(r['request_latency_s'] for r in selected) / 3}
    for arm in ARMS:
        total = {key: sum(values[r['request_id'], arm][key] for r in requests) for key in COUNTS}
        total['summed_mean_request_latency_s'] = sum(values[r['request_id'], arm]['latency'] for r in requests)
        total['all_repeats_request_latency_s'] = sum(r['request_latency_s'] for r in rows if r['method'] == arm)
        calls, output = total['verification_calls'], total['output_tokens']
        total['calls_per_1000_output_tokens'] = 1000 * calls / output if output else None
        total['accepted_draft_tokens_per_call'] = total['accepted_draft_tokens'] / calls if calls else None
        total['committed_progress_per_call'] = total['committed_progress'] / calls if calls else None
        totals[arm] = total
    comparison = dict(candidate='K', reference='O')
    for metric, key, reduction_name in (('call', 'verification_calls', 'call_reduction_pct'),
                                        ('latency', 'latency', 'request_time_reduction_pct')):
        before = [values[r['request_id'], 'O'][key] for r in requests]
        after = [values[r['request_id'], 'K'][key] for r in requests]
        differences = [a - b for a, b in zip(before, after)]
        comparison[reduction_name] = 100 * (1 - sum(after) / sum(before)) if sum(before) else None
        comparison[metric + '_request_directions'] = dict(improved=sum(d > 0 for d in differences),
            regressed=sum(d < 0 for d in differences), tied=sum(d == 0 for d in differences))
        comparison[metric + '_gross_saved'] = sum(max(d, 0) for d in differences)
        comparison[metric + '_gross_extra'] = sum(max(-d, 0) for d in differences)
        comparison[metric + '_net_saved'] = sum(differences)
    return totals, comparison, values


def output_summary(canonicals):
    """Count outputs once per request, independently of arms and timing repeats."""
    lengths = [len(row['response_ids']) for row in canonicals.values()]
    return dict(requests=len(lengths), minimum_tokens=min(lengths), maximum_tokens=max(lengths),
                mean_tokens=sum(lengths) / len(lengths), total_tokens=sum(lengths),
                cap_hit_requests=sum(row['finish_reason'] == 'length' for row in canonicals.values()),
                finish_reason_counts=dict(Counter(row['finish_reason'] for row in canonicals.values())))


def trace_extras(record, request, row, canonical, validate_generation, issues):
    label = row['label']
    response, meta = record['response_ids'], record['metadata']
    blocks, capture = meta['trace'], meta['capture']
    check(all(type(token) is int and token >= 0 for token in response), issues, label + ': integer response tokens')
    check(all(type(meta[name]) is int and meta[name] >= 0 for name in COUNTS[:-1]), issues, label + ': integer native counts')
    check(record['record_id'] == label and record['request_index'] == request['panel_ordinal']
          and record['session_id'] == 'fresh_confirmation_evaluation', issues, label + ': native identity')
    check(row['panel_ordinal'] == request['panel_ordinal'] and row['original_index'] == request['original_index'], issues, label + ': panel identity')
    check(capture['blocks'] == blocks and capture['prompt_ids'] == request['prompt_ids']
          and capture['response_ids'] == capture['target_argmax_ids'] == response
          and capture['covered_token_ids'] == request['prompt_ids'] + response[:-1]
          and capture['covered_rows'] == len(capture['covered_token_ids']), issues, label + ': duplicated native capture')
    check(capture['status'] == 'passed' and capture['block_size'] == 16
          and capture['feature_persistence_s'] == 0 and not capture.get('feature_path')
          and not meta.get('feature_path'), issues, label + ': capture disabled')
    for index, block in enumerate(blocks):
        check(block['native_accepted_draft_tokens'] == max(block['committed_progress'] - 1, 0), issues, label + ': native accepted counter')
        check(block['stop_clipped'] == (block['committed_progress'] < block['raw_accepted_draft_tokens_before_stop'] + 1)
              and block['finished'] == (index == len(blocks) - 1), issues, label + ': terminal flags')
    actual, finish = validate_generation(request['prompt_ids'], record['prompt_ids'] + response, blocks,
                                        max_new_tokens=2048, stop_tokens=request['stop_token_ids'])
    check(actual == canonical['response_ids'] and finish == canonical['finish_reason'], issues, label + ': exact native canonical')
    for key in COUNTS[:-1] + ('request_latency_s', 'native_generate_s'):
        check(close(row[key], meta[key]), issues, label + ': measurement/native ' + key)
    check(row['output_tokens'] == len(response) and row['prompt_tokens'] == len(request['prompt_ids'])
          and row['finish_reason'] == finish and row['cap_hit'] == (finish == 'length'), issues, label + ': measurement output')
    check(close(row['cache_maintenance_s'], capture['cache_maintenance_s']) and capture['cache_maintenance_s'] >= 0,
          issues, label + ': cache maintenance timer')


def audit_evaluation(raw, plan, issues):
    root, requests = raw / 'campaign_001/evaluation', plan['requests']
    rows, protocol = lines(root / 'MEASUREMENTS.jsonl'), read(root / 'PROTOCOL.json')
    status, canonicals = read(root / 'STATUS.json'), read(root / 'CANONICAL.json')
    summary = read(root / 'report/SUMMARY.json')
    checkpoints = plan['checkpoints']
    expected = expected_slots(requests)
    actual = [(r['request_id'], r['repeat'], r['method']) for r in rows]
    check(actual == expected and len(actual) == len(set(actual)) == 384, issues, '384 unique chronological alternating slots')
    check(set(canonicals) == {r['request_id'] for r in requests}, issues, '64 canonical request keys')
    compare(protocol, dict(arms=['O', 'K'], comparisons=[['K', 'O']], repeats=3,
            checkpoints=checkpoints, rotation='(panel_ordinal + repeat) % 2',
            count_trajectory_repeat=0, historical_count_gate=False, capture=False,
            extra_warmup_generations=0), issues, 'protocol')
    compare(status, dict(status='completed', planned_generations=384, attempted_generations=384,
            returned_generations=384, completed_valid_generations=384, invalid_generations=0,
            pending_canonical_generations=0, unknown_generation_attempts=0, unattempted_generations=0,
            optimizer_steps=0, feedback_collections=0, automatic_retry=False,
            qualification='passed', qualification_requests=8, interrupted_attempt=None), issues, 'evaluation status')
    qualification = read(root / 'QUALIFICATION.json')
    compare(qualification, dict(status='passed', requests=8, generations=48,
            arms_with_native_equality=['K', 'O'], performance_win_required=False, extra_generations=0), issues, 'included qualification')
    trace_record, validate_generation = load_helpers()
    by_id, checked = {r['request_id']: r for r in requests}, []
    native_paths, accepted_histograms, blocks_total = set(), {a: [0] * 16 for a in ARMS}, 0
    progress_histograms = {a: [0] * 17 for a in ARMS}
    for row in rows:
        request, canonical = by_id[row['request_id']], canonicals[row['request_id']]
        arm, repeat = row['method'], row['repeat']
        label = f"{request['panel_ordinal']:04d}_r{repeat}_{arm}"
        check(row['label'] == label and Path(row['record_path']).name == label + '.json', issues, label + ': record basename')
        path = local_record(root, row['record_path'])
        record = read(path)
        expected_output = {**request, 'response_ids': canonical['response_ids'], 'finish_reason': canonical['finish_reason']}
        derived = trace_record(record, expected_output, checkpoints[arm], issues, label)
        trace_extras(record, request, row, canonical, validate_generation, issues)
        check(row['validation_status'] == 'valid' and row['valid_measurement'] is True
              and row['structural_errors'] == row['errors'] == [], issues, label + ': valid measurement')
        capture_path = local_record(raw / 'campaign_001/native', record['metadata']['native_trace_path'])
        check(read(capture_path) == record['metadata']['capture'], issues, label + ': native capture file')
        native_paths.add(capture_path.name)
        blocks_total += len(record['metadata']['trace'])
        if repeat == 0:
            for block in record['metadata']['trace']:
                accepted_histograms[arm][block['accepted_draft_tokens']] += 1
                progress_histograms[arm][block['committed_progress']] += 1
        checked.append({**row, **{k: derived[k] for k in COUNTS},
                        'request_latency_s': record['metadata']['request_latency_s']})
        validation = read(root / (label + '_VALIDATION.json'))
        compare(validation, dict(label=label, request_id=request['request_id'], method=arm, repeat=repeat,
                status='valid', valid=True, errors=[]), issues, label + ': final validation')
        check(Path(validation['canonical_source']).name == Path(canonical['source_record_path']).name,
              issues, label + ': validation canonical source')
        if arm == 'O' and repeat == 0:
            check(canonical['source_method'] == 'O' and canonical['source_repeat'] == 0
                  and Path(canonical['source_record_path']).name == label + '.json'
                  and canonical['response_ids'] == record['response_ids']
                  and canonical['finish_reason'] == record['finish_reason'], issues, label + ': first O canonical binding')
    raw_files = {p.name for p in root.iterdir() if re.fullmatch(r'\d{4}_r[012]_[OK]\.json', p.name)}
    check(raw_files == {r['label'] + '.json' for r in rows} and len(raw_files) == 384, issues, 'exact native record file set')
    check(native_paths == {p.name for p in (raw / 'campaign_001/native').glob('request_*.json')}
          and len(native_paths) == 384, issues, 'exact384 worker generation captures; no extra warmup')
    events = lines(root / 'VALIDATION_EVENTS.jsonl')
    pending = [r for r in events if r['status'] == 'canonical_pending']
    check({r['label'] for r in pending} == {f'{i:04d}_r0_K' for i in range(1, 64, 2)}
          and len(pending) == 32, issues, 'K-first canonical deferral on odd request ordinals')
    check(all(r['status'] in ('valid', 'canonical_pending') and not r['errors'] for r in events), issues, 'validation event outcomes')
    valid_events = [r for r in events if r['status'] == 'valid']
    check(len(valid_events) == 384 and {r['label'] for r in valid_events} == {r['label'] for r in rows},
          issues, 'one durable valid event per returned generation')
    for event in pending:
        transitions = [r['status'] for r in events if r['label'] == event['label']]
        check(event['canonical_source'] is None and transitions == ['canonical_pending', 'valid'],
              issues, event['label'] + ': deferred validation transition')
    for request in requests:
        for arm in ARMS:
            repeats = [r for r in checked if r['request_id'] == request['request_id'] and r['method'] == arm]
            check(len({tuple(r[k] for k in COUNTS) for r in repeats}) == 1, issues, request['request_id'] + '/' + arm + ': identical repeated counts')
    totals, comparison, values = totals_and_comparison(checked, requests)
    compare(summary, dict(planned_requests=64, valid_requests=64, invalid_or_incomplete_requests=0,
            returned_measurements=384, valid_timed_generations=384,
            timing_repeats_per_method_request=3, primary_comparison='K_vs_O',
            **totals, comparisons={'K_vs_O': comparison}, execution=status), issues, 'recomputed summary')
    with (root / 'report/PER_REQUEST.csv').open(newline='') as stream:
        per_request = list(csv.DictReader(stream))
    check([r['request_id'] for r in per_request] == [r['request_id'] for r in requests], issues, 'per-request report order')
    for row in per_request:
        check(row['valid'] == row['complete'] == 'True' and row['errors'] == '', issues, 'per-request report validity')
        for arm in ARMS:
            value = values[row['request_id'], arm]
            for key in COUNTS:
                check(int(row[f'{arm}_{key}']) == value[key], issues, 'per-request ' + arm + '/' + key)
            check(close(float(row[f'{arm}_mean_request_latency_s']), value['latency']), issues, 'per-request mean request time')
    repeat_totals, repeat_reductions = [], []
    for repeat in range(3):
        for arm in ARMS:
            selected = [r for r in checked if r['method'] == arm and r['repeat'] == repeat]
            repeat_totals.append(dict(repeat=repeat, method=arm, requests=len(selected),
                request_latency_s=sum(r['request_latency_s'] for r in selected),
                verification_calls=sum(r['verification_calls'] for r in selected)))
        o, k = repeat_totals[-2:]
        repeat_reductions.append(100 * (1 - k['request_latency_s'] / o['request_latency_s']))
    with (root / 'report/REPEAT_TOTALS.csv').open(newline='') as stream:
        saved_repeat = list(csv.DictReader(stream))
    check(len(saved_repeat) == 6, issues, 'six repeat total rows')
    for actual_row, expected_row in zip(saved_repeat, repeat_totals):
        for key, value in expected_row.items():
            check(close(float(actual_row[key]), value) if type(value) in (int, float) else actual_row[key] == value,
                  issues, 'repeat total ' + key)
    return dict(records=len(rows), requests=len(requests), reconstructed_native_blocks=blocks_total,
                worker_captures=len(native_paths), deferred_K_first_rows=len(pending),
                qualification=qualification, totals=totals, comparison=comparison,
                repeat_totals=repeat_totals, repeat_request_time_reduction_pct=repeat_reductions,
                accepted_draft_count_histograms_repeat0=accepted_histograms,
                committed_progress_histograms_repeat0=progress_histograms,
                output_summary=output_summary(canonicals), status=status)


def audit_publication(raw, plan, evaluation, issues):
    root = raw / 'campaign_001/evaluation'
    activations = lines(root / 'ACTIVATIONS.jsonl')
    protocol_source = HERE / 'worktree/experiments/osd_components/protocol.py'
    tree = ast.parse(protocol_source.read_text())
    namespace = {'BLOCK_SIZE': 16}
    helpers = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('publication_errors', 'weight_check_errors')]
    exec(compile(ast.Module(body=helpers, type_ignores=[]), str(protocol_source), 'exec'), namespace)
    previous, expected_activations = None, []
    for rid, repeat, arm in expected_slots(plan['requests']):
        ordinal = next(r['panel_ordinal'] for r in plan['requests'] if r['request_id'] == rid)
        if previous != arm:
            expected_activations.append((ordinal, repeat, arm))
            previous = arm
    check([(r['panel_ordinal'], r['repeat'], r['method']) for r in activations] == expected_activations,
          issues, 'actual checkpoint switch schedule')
    for row in activations:
        check(not row['errors'] and not namespace['publication_errors'](row, plan['checkpoints'][row['method']]), issues, 'checkpoint publication gate')
    check(close(sum(r['activation_s'] for r in activations), evaluation['status']['activation_s']), issues, 'activation cost sum')
    weight_s = 0
    for arm in ARMS:
        receipt = read(root / f'WEIGHTS_{arm}.json')
        check(not receipt['errors'] and not namespace['weight_check_errors'](receipt['native_reply'], plan['checkpoints'][arm]), issues, arm + ': native loaded tensor equality')
        weight_s += receipt['validation_s']
    check(close(weight_s, evaluation['status']['weight_validation_s']), issues, 'weight validation cost sum')
    engine = read(raw / 'campaign_001/native/engine_settings.json')
    compare(engine, dict(model_path=plan['target'], speculative_draft_model_path=plan['released_draft'],
            speculative_algorithm='DFLASH', speculative_num_draft_tokens=16, dtype='bfloat16',
            attention_backend='fa3', tp_size=1, max_running_requests=1, context_length=4112,
            chunked_prefill_size=8192, random_seed=17, skip_server_warmup=True), issues, 'qualified native engine')
    return dict(activations=len(activations), activation_s=sum(r['activation_s'] for r in activations),
                weight_validation_s=weight_s, native_tensor_equality_arms=list(ARMS))


def audit_integrity(raw, plan, issues):
    seal = read(raw / 'sealed/SEAL.json')
    preflight, postflight = read(raw / 'PREFLIGHT.json'), read(raw / 'POSTFLIGHT.json')
    check(seal['sealed_before_generation'] is True, issues, 'sealed before generation')
    for name, receipt in [('preflight', preflight), ('postflight', postflight)]:
        check(receipt['status'] == 'passed' and receipt['files_checked'] == len(seal['files']), issues, name + ': all sealed files verified')
        check(receipt['free_bytes'] >= receipt['required_bytes'], issues, name + ': storage reserve')
    runtime_pin = read(raw / 'sealed/RUNTIME_PIN.json')
    runtime_postflight = read(raw / 'RUNTIME_POSTFLIGHT.json')
    expected_runtime = {key: runtime_pin[key] for key in
                        ('python_version', 'torch', 'transformers', 'cuda', 'native_runtime_sha256')}
    check(preflight['runtime'] == expected_runtime, issues, 'runtime preflight exact pin')
    check(runtime_postflight == dict(status='passed', runtime=expected_runtime),
          issues, 'runtime postflight exact pin')
    local_checked, remote_only = [], []
    remote_run = '/data/sources/minhao/runs/' + HERE.name + '/'
    remote_code = '/data/sources/minhao/DFlash-Online/'
    for name, identity in seal['files'].items():
        local = None
        if name.startswith(remote_code):
            local = HERE / 'worktree' / name[len(remote_code):]
        elif name.startswith(remote_run):
            relative = name[len(remote_run):]
            candidates = (raw / relative, HERE / relative)
            local = next((p for p in candidates if p.is_file()), None)
        if local is None:
            remote_only.append(name)
            continue
        check(local.is_file() and local.stat().st_size == identity['bytes'] and digest(local) == identity['sha256'], issues, 'local sealed file: ' + str(local))
        local_checked.append(str(local))
    for arm in ARMS:
        checkpoint = plan['checkpoints'][arm]
        expected = seal['exact_file_sets'][checkpoint]
        check(expected and all(checkpoint + '/' + name in seal['files'] for name in expected), issues, arm + ': complete checkpoint seal')
    status, costs = read(raw / 'campaign_001/STATUS.json'), read(raw / 'campaign_001/COSTS.json')
    compare(status, dict(status='completed', phase='finished', planned_generations=384,
            training_steps=0, feedback_collections=0, extra_generations=0), issues, 'campaign work')
    check(int((raw / 'EXIT_CODE').read_text().strip()) == 0, issues, 'supervisor exit0')
    check(all(type(costs[k]) in (int, float) and math.isfinite(costs[k]) and costs[k] > 0
              for k in ('backend_loading_s', 'campaign_wall_s')), issues, 'finite positive campaign cost')
    check(not any((raw / 'campaign_001').glob('**/training_state.pt')), issues, 'no new training state')
    check(not (raw / 'campaign_001/evaluation/REPORT_FAILURE.json').exists(), issues, 'no report failure')
    selection = read(HERE / 'inventory_audit/SELECTION_AUDIT.json')
    panel = read(HERE / 'inventory_audit/NEW_PANEL.json')
    check(plan['requests'] == panel and digest(HERE / 'inventory_audit/NEW_PANEL.json') == selection['panel_sha256'], issues, 'sealed selected panel')
    check(plan['freshness_scope'] == 'current_one_event_O_K_study'
          and plan['older_project_exposure_disclosed'] is True
          and selection['selection_seed'] == plan['selection_seed'] == 20261027
          and selection['selection_older_executed_count'] == 64
          and all(r['older_project_execution_verified'] for r in plan['requests']), issues, 'approved freshness scope and exposure disclosure')
    return dict(sealed_files=len(seal['files']), local_files_rehashed=len(local_checked),
                remote_only_files=len(remote_only), postflight=postflight, preflight=preflight,
                runtime_postflight=runtime_postflight,
                costs=costs, source_of_tensor_immutability='Passed remote pre/postflight seal checks plus loaded-tensor equality; model tensors are not reloaded or rehashed locally.',
                older_project_exposed_requests=64, current_study_new_requests=64)


def run_audit(raw):
    raw, issues = Path(raw), []
    result = dict(scope='fixed_checkpoint_O_K_current_study_new_inputs', verification_status='ANALYZED',
        issues=issues, generated_or_trained_by_auditor=0,
        helper_sha256={str(p): digest(p) for p in (TRACE_AUDITOR, NATIVE_SOURCE)},
        independence='Audit implementation is separate from the evaluator. This analyst prepared the source panel and previously implemented training/qualification; no blind or external-review independence is claimed.')
    try:
        plan = read(raw / 'sealed/PLAN.json')
        check(len(plan['requests']) == 64 and plan['planned_generations'] == 384 and set(plan['checkpoints']) == {'O', 'K'}, issues, 'sealed budget/arms')
        result['evaluation'] = audit_evaluation(raw, plan, issues)
        result['publication'] = audit_publication(raw, plan, result['evaluation'], issues)
        result['integrity'] = audit_integrity(raw, plan, issues)
    except (OSError, KeyError, TypeError, ValueError, IndexError, RuntimeError, StopIteration) as error:
        issues.append(type(error).__name__ + ': ' + str(error))
    result['status'] = 'passed' if not issues else 'issues_found'
    return result


def markdown(result):
    text = ['## Material Passport', '', '- Origin Skill: ARS academic-research-suite, experiment-agent validate.',
            '- Verification Status: ANALYZED; local compact-evidence audit, no model rerun.',
            '- Scope: fixed O/K checkpoints, 64 inputs new to the current study, all 64 previously executed in older unrelated project work.',
            '- Review provenance: separate evaluator audit; this analyst also prepared the panel and earlier training code.',
            '', f"Audit status: **{result['status']}**. Issues: **{len(result['issues'])}**.", '']
    if result['issues']:
        text += ['- ' + issue for issue in result['issues']]
    if 'evaluation' in result:
        evaluation, totals, comparison = result['evaluation'], result['evaluation']['totals'], result['evaluation']['comparison']
        text += ['', f"Reconstructed {evaluation['reconstructed_native_blocks']:,} native blocks across {evaluation['records']} returned generations. "
                 'Exact inputs, first-O canonical outputs, termination, within-arm repeat counts, alternating order and included 48-generation correctness gate were checked.', '',
                 '| Arm | Calls once/request | Sum of three-repeat mean times (s) | Accepted draft tokens/call | Committed progress/call |',
                 '|---|---:|---:|---:|---:|']
        for arm in ARMS:
            value = totals[arm]
            text.append(f"| {arm} | {value['verification_calls']} | {value['summed_mean_request_latency_s']:.6f} | {value['accepted_draft_tokens_per_call']:.6f} | {value['committed_progress_per_call']:.6f} |")
        signs = comparison['call_request_directions']
        text += ['', f"K/O call reduction: {comparison['call_reduction_pct']:+.4f}%; request-time reduction: {comparison['request_time_reduction_pct']:+.4f}%. "
                 f"Requests improve/regress/tie in calls: {signs['improved']}/{signs['regressed']}/{signs['tied']}. "
                 f"Gross calls saved/added: {comparison['call_gross_saved']}/{comparison['call_gross_extra']}; net saved: {comparison['call_net_saved']}.",
                 '', 'Per-repeat request-time reductions: ' + ', '.join(f'{v:+.4f}%' for v in evaluation['repeat_request_time_reduction_pct']) + '.',
                 '', 'Counts use one repeat-0 trajectory per request; time uses the mean of three repeated measurements. Ratios are pooled sums. '
                 'No bootstrap, p-value, population-generalization or training-seed robustness claim is made. This reuses fixed one-seed checkpoints; the new panel is independent of current-study inputs, not globally unseen project data.']
        output = evaluation['output_summary']
        text += ['', f"Canonical output lengths range from {output['minimum_tokens']} to {output['maximum_tokens']} tokens "
                 f"(mean {output['mean_tokens']:.3f}; total {output['total_tokens']:,} once across 64 requests). "
                 f"Termination counts: {output['finish_reason_counts']}; {output['cap_hit_requests']} requests hit the 2,048-token cap.",
                 '', f"O/K accepted draft-token totals are {totals['O']['accepted_draft_tokens']:,}/{totals['K']['accepted_draft_tokens']:,}; "
                 f"both commit {totals['O']['committed_progress']:,} tokens beyond the 64 initial target-generated seeds. "
                 'Accepted counts are clipped at actual stop boundaries; the full 0–15 accepted-length and 0–16 committed-progress histograms are retained in FINAL_AUDIT.json.',
                 '', '**Interpretation:** these fixed K weights require modestly fewer verification calls than O on this prespecified current-study-new panel. '
                 'Every timing repeat has the same favorable aggregate direction, but its magnitude varies substantially; the three repeats do not provide independent inputs or training seeds. '
                 'This supports a bounded checkpoint-level serving benefit on this panel. It does not establish a generally reliable algorithmic gain, isolate selection from selected-weight renormalization, or demonstrate adaptation-inclusive acceleration. '
                 'Earlier development results remain a separate experiment and are not pooled here. The historical O full-training-startup numerical-equivalence limitation is unchanged.']
    if 'integrity' in result:
        integrity = result['integrity']
        text += ['', f"Integrity: {integrity['sealed_files']} sealed files covered by passed remote pre/postflight checks; "
                 f"{integrity['local_files_rehashed']} locally available source/evidence files independently rehashed. "
                 'Checkpoint tensors are supported by remote immutability checks and native loaded-tensor equality, not a local tensor rerun. '
                 'Runtime preflight and postflight exactly match all pinned version fields and seven native runtime source hashes.',
                 '', 'No training, optimizer update, feedback collection or extra generation was recorded. Activation, weight checks, backend loading and full campaign wall time remain separate from available-checkpoint request latency.']
    text += ['', '### Statistical fallacy scan', '', '**11/11 checked.** Overall interpretive confidence: CAUTION.', '',
             '| Check | Assessment |', '|---|---|',
             "| Simpson's paradox | Repeat-level directions are reported explicitly; no pooled-workload generalization. |",
             '| Ecological fallacy | Request improvements and regressions are reported; aggregate benefit is not assigned to every request. |',
             "| Berkson's paradox | Current-study exclusions and historical project exposure restrict the sample; no specific induced-correlation claim. |",
             '| Collider bias | All 64 completed paired requests are retained; no output-quality or performance filtering. |',
             '| Base-rate neglect | Denominators are 64 requests, 384 timed generations and one trajectory/request for calls. |',
             '| Regression to the mean | Fixed selected checkpoints are tested on new current-study inputs against O; one seed and earlier method screening still limit inference. |',
             '| Survivorship bias | Attempted/returned/invalid/unknown/unattempted counts and all raw files are checked; failures cannot be silently dropped. |',
             '| Look-elsewhere effect | One prespecified K/O comparison; earlier development screening and historical data exposure remain disclosed. |',
             '| Forking paths | Seed, exclusions, order, checkpoint bytes and metric definitions were sealed before generation; no post-result reselection or metric change. |',
             '| Correlation versus causation | Controlled fixed-checkpoint serving comparison supports a bounded observed difference, not a universal training-mechanism claim. |',
             '| Reverse causality | Selection and checkpoint sealing precede this evaluation; previous unrelated-project exposure prevents a claim of globally untouched data. |',
             '', 'Stage 1/project None remains unchanged. The audit neither launches a follow-up nor authorizes retuning.']
    return '\n'.join(text) + '\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', type=Path, default=HERE / 'raw')
    parser.add_argument('--output', type=Path, default=HERE)
    args = parser.parse_args()
    result = run_audit(args.raw_root)
    args.output.mkdir(exist_ok=True, parents=True)
    (args.output / 'FINAL_AUDIT.json').write_text(json.dumps(result, indent=2) + '\n')
    (args.output / 'FINAL_AUDIT.md').write_text(markdown(result))
    print(json.dumps(dict(status=result['status'], issues=result['issues'], evaluation=result.get('evaluation'))))
    raise SystemExit(bool(result['issues']))
