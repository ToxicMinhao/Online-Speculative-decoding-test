"""Read-only, stdlib reconstruction of the completed C1 native endpoint.

No experiment modules are imported; no model or GPU work is performed.
Output contains identities, totals and bounded errors, never prompt/answer text.
"""
import collections
import datetime
import hashlib
import json
import math
from pathlib import Path
import time

ROOT = Path('/data/sources/minhao/runs/dflash_adaflash_c1_20261105/production_001')
FINAL = ROOT / 'campaign_001/evaluation/final'
METHODS = ('O', 'KPLUS', 'A')
TIMES = ('request_latency_s', 'native_generate_s', 'cache_maintenance_s')
COUNTS = ('verification_calls', 'accepted_draft_tokens', 'committed_progress', 'output_tokens')
started = time.monotonic()
checks = collections.Counter()
failures = collections.Counter()
examples = []
inputs = {}


def check(condition, name, identity=''):
    checks[name] += 1
    if not condition:
        failures[name] += 1
        if len(examples) < 100:
            examples.append({'check': name, 'identity': str(identity)})


def read(path, *, lines=False):
    path = Path(path)
    data = path.read_bytes()
    inputs[str(path)] = (len(data), hashlib.sha256(data).hexdigest())
    return [json.loads(line) for line in data.splitlines() if line] if lines else json.loads(data)


def near(a, b):
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-9)


plan = read(ROOT / 'sealed/PLAN.json')
requests = plan['evaluation_requests']
canonicals = read(ROOT / 'sealed/CANONICALS.json')
protocol = read(FINAL / 'PROTOCOL.json')
status = read(FINAL / 'STATUS.json')
rows = read(FINAL / 'MEASUREMENTS.jsonl', lines=True)
reported = read(ROOT / 'results/SUMMARY.json')
checkpoints = dict(plan['references'], A=str(ROOT / 'campaign_001/arms/A/trainer/events/event_058/export'))
check(len(requests) == 1024 and len(canonicals) == 1024, 'complete_panel')
check(len({r['request_id'] for r in requests}) == 1024, 'unique_request_ids')
check(read(FINAL / 'REQUESTS.json') == requests, 'sealed_request_binding')
check(read(FINAL / 'CANONICAL_INPUT.json') == canonicals, 'sealed_canonical_binding')
check(protocol['checkpoints'] == checkpoints, 'sealed_checkpoint_binding')
check(plan['execution_source']['commit'] == '681336c5e5c5caa0aa67560e46ffe83ce81e9351'
      and plan['execution_source']['dirty'] is False, 'execution_revision')
for name in ('capture', 'historical_call_gate', 'performance_win_required'):
    check(protocol[name] is False, 'evaluation_' + name)
check(protocol['extra_warmup_generations'] == 0, 'no_generation_warmup')

expected = []
orders = collections.Counter()
for block in range(16):
    for repeat in range(3):
        base = METHODS if block % 2 == 0 else ('O', 'A', 'KPLUS')
        shift = (block // 2 + repeat) % 3
        order = base[shift:] + base[:shift]
        orders[order] += 1
        for position, method in enumerate(order):
            for ordinal in range(block * 64, (block + 1) * 64):
                expected.append(dict(seed=17, event=58, phase='final', block_ordinal=block,
                    panel_ordinal=ordinal, request_id=requests[ordinal]['request_id'],
                    repeat=repeat, method=method, position=position))
check(len(orders) == 6 and set(orders.values()) == {8}, 'balanced_six_orders')
check(protocol['schedule'] == expected == plan['evaluation_schedule'], 'sealed_full_order')
check(len(rows) == len(expected) == 9216, 'complete_measurement_count')
expected_names = {f"{s['panel_ordinal']:04d}_r{s['repeat']}_{s['method']}.json" for s in expected}
actual_names = {p.name for p in FINAL.glob('[0-9][0-9][0-9][0-9]_r[0-2]_*.json')
                if not p.name.endswith('_VALIDATION.json')}
check(actual_names == expected_names, 'exact_raw_record_inventory')

# Bind the sealed reference to its original frozen serving evidence as well.
for request in requests:
    canonical = canonicals[request['request_id']]
    original = read(canonical['source_record_path'])
    check(original['prompt_ids'] == request['prompt_ids'], 'original_frozen_prompt', request['request_id'])
    check(original['response_ids'] == canonical['response_ids']
          and original['finish_reason'] == canonical['finish_reason'], 'original_frozen_reference', request['request_id'])
    check(original['checkpoint_id'] == canonical['F_checkpoint'], 'original_frozen_checkpoint', request['request_id'])

by_request = collections.defaultdict(list)
trace_blocks = 0
native_paths = set()
round_endings = collections.Counter()
for row_number, (slot, row) in enumerate(zip(expected, rows)):
    ordinal, repeat, method = slot['panel_ordinal'], slot['repeat'], slot['method']
    label = f'{ordinal:04d}_r{repeat}_{method}'
    request = requests[ordinal]
    canonical = canonicals[request['request_id']]
    raw = read(FINAL / (label + '.json'))
    meta = raw['metadata']
    capture = meta['capture']
    trace = meta['trace']
    response = raw['response_ids']
    prompt = raw['prompt_ids']
    check(all(row[k] == v for k, v in slot.items()), 'observed_execution_order', label)
    check(raw['record_id'] == label and raw['request_index'] == ordinal
          and raw['session_id'] == 'adaflash_c1_evaluation', 'raw_identity', label)
    check(raw['checkpoint_id'] == checkpoints[method], 'raw_checkpoint', label)
    check(prompt == request['prompt_ids'], 'exact_prompt_tokens', label)
    check(response == canonical['response_ids'], 'exact_output_tokens', label)
    check(raw['finish_reason'] == canonical['finish_reason'], 'exact_termination', label)
    check(raw['stop_token_ids'] == request['stop_token_ids']
          and raw['max_new_tokens'] == request['max_new_tokens'] == 2048
          and len(prompt) + 2048 <= 4096, 'serving_boundaries', label)
    check(0 < len(response) <= 2048 and all(type(t) is int and t >= 0 for t in response), 'valid_output_tokens', label)
    check(not any(t in raw['stop_token_ids'] for t in response[:-1]), 'first_eos_boundary', label)
    expected_end = 'eos' if response[-1] in raw['stop_token_ids'] else 'length'
    check(raw['finish_reason'] == expected_end
          and (expected_end == 'eos' or len(response) == 2048), 'native_termination', label)
    check(raw['purpose'] == 'evaluation' and meta['feedback_capture_in_timed_request'] is False
          and capture['collect_features'] is False and capture['feature_bytes'] == 0
          and capture['feature_persistence_s'] == 0, 'capture_disabled', label)
    check(meta['block_size'] == capture['block_size'] == 16 and meta['context_limit'] == 4096
          and meta['runtime'] == 'pinned_SGLang_FA3'
          and meta['cache_policy'] == 'idle_full_flush_before_every_request_and_publication', 'qualified_runtime_cache', label)
    check(capture['status'] == 'passed' and capture['native_capture_validated'] is True
          and capture['prompt_ids'] == prompt and capture['response_ids'] == response
          and capture['blocks'] == trace and capture['target_argmax_ids'] == response, 'native_capture_binding', label)
    native_path = Path(meta['native_trace_path'])
    check(native_path.is_relative_to(ROOT / 'campaign_001') and str(native_path) not in native_paths, 'unique_native_sidecar', label)
    native_paths.add(str(native_path))
    check(read(native_path) == capture, 'exact_native_sidecar', label)

    offset = 0
    accepted_sum = native_accepted_sum = raw_accepted_sum = 0
    for index, block in enumerate(trace):
        identity = f'{label}:block{index}'
        proposals, posterior = block['draft_tokens'], block['target_argmax']
        width, progress = block['verify_size'], block['committed_progress']
        check(type(width) is int and 1 <= width <= 16 and len(proposals) + 1 == width == len(posterior)
              and type(progress) is int and 1 <= progress <= width, 'block_geometry', identity)
        first_error = next((i for i, (a, b) in enumerate(zip(proposals, posterior)) if a != b), len(proposals))
        check(block['output_start'] == offset and block['anchor_token'] == response[offset], 'round_start_anchor', identity)
        check(block['positions'] == list(range(len(prompt) + offset, len(prompt) + offset + width)), 'round_positions', identity)
        committed = proposals[:first_error] + [posterior[first_error]]
        check(progress <= len(committed) and block['committed_tokens'] == committed[:progress]
              == posterior[:progress] == response[offset + 1:offset + 1 + progress], 'canonical_commit_reconstruction', identity)
        check(block['accepted_draft_tokens'] == min(first_error, progress)
              and block['native_accepted_draft_tokens'] == progress - 1
              and block['raw_accepted_draft_tokens_before_stop'] == first_error, 'accepted_counter_definitions', identity)
        check(block['stop_clipped'] == (progress < first_error + 1), 'stop_clipping', identity)
        check(block['finished'] is (index == len(trace) - 1), 'round_finished', identity)
        round_endings['stop_clipped' if block['stop_clipped'] else 'not_stop_clipped'] += 1
        accepted_sum += block['accepted_draft_tokens']
        native_accepted_sum += block['native_accepted_draft_tokens']
        raw_accepted_sum += first_error
        offset += progress
        trace_blocks += 1
    check(offset == len(response) - 1 == meta['committed_progress'], 'initial_seed_excluded_progress', label)
    check(len(trace) == meta['verification_calls'] and accepted_sum == meta['accepted_draft_tokens']
          and 0 <= accepted_sum <= offset, 'record_counter_totals', label)
    check(near(meta['committed_progress_per_call'], offset / len(trace))
          and near(meta['calls_per_1000_output_tokens'], 1000 * len(trace) / len(response)), 'record_ratios', label)
    observed = dict(verification_calls=len(trace), accepted_draft_tokens=accepted_sum,
        native_accepted_draft_tokens=native_accepted_sum, raw_accepted_before_stop=raw_accepted_sum,
        committed_progress=offset, output_tokens=len(response),
        request_latency_s=meta['request_latency_s'], native_generate_s=meta['native_generate_s'],
        cache_maintenance_s=capture['cache_maintenance_s'])
    check(all(row[k] == observed[k] for k in (*COUNTS, *TIMES)), 'measurement_raw_equality', label)
    check(row['checkpoint_id'] == checkpoints[method] and row['prompt_tokens'] == len(prompt)
          and row['finish_reason'] == raw['finish_reason'] and row['cap_hit'] == (expected_end == 'length')
          and row['record_path'] == str(FINAL / (label + '.json')) and row['label'] == label
          and row['original_index'] == request['original_index'], 'measurement_identity', label)
    check(all(math.isfinite(observed[t]) and observed[t] >= 0 for t in TIMES)
          and observed['request_latency_s'] > 0 and observed['native_generate_s'] > 0, 'finite_timing', label)
    check(row['errors'] == row['structural_errors'] == [] and row['valid_measurement'] is True
          and row['validation_status'] == 'valid', 'record_validation_receipt', label)
    validation = read(FINAL / (label + '_VALIDATION.json'))
    check(validation['status'] == 'valid' and validation['errors'] == [], 'durable_validation_receipt', label)
    by_request[ordinal, method].append((repeat, observed))

totals = {m: collections.defaultdict(float) for m in METHODS}
repeat_totals = {r: {m: collections.defaultdict(float) for m in METHODS} for r in range(3)}
acceptance_rows = []
for (ordinal, method), values in sorted(by_request.items()):
    values.sort()
    check([r for r, _ in values] == [0, 1, 2], 'three_distinct_repeats', f'{ordinal}:{method}')
    count_fields = (*COUNTS, 'native_accepted_draft_tokens', 'raw_accepted_before_stop')
    check(len({tuple(v[k] for k in count_fields) for _, v in values}) == 1, 'repeat_consistent_counts', f'{ordinal}:{method}')
    acceptance_rows.append(dict(panel_ordinal=ordinal, request_id=requests[ordinal]['request_id'],
        method=method, checkpoint_id=checkpoints[method], count_trajectory_repeat=0,
        **{key: values[0][1][key] for key in count_fields}))
    for key in count_fields:
        totals[method][key] += values[0][1][key]
    for key in TIMES:
        totals[method][key] += sum(v[key] for _, v in values) / 3
    for repeat, value in values:
        for key in (*count_fields, *TIMES):
            repeat_totals[repeat][method][key] += value[key]
for method, total in totals.items():
    for key in (*COUNTS, 'native_accepted_draft_tokens', 'raw_accepted_before_stop'):
        total[key] = int(total[key])
    for key, value in reported['endpoint']['method_totals'][method].items():
        if key in total:
            check(near(value, total[key]), 'reported_total_' + key, method)
    total['accepted_tokens_per_call'] = total['accepted_draft_tokens'] / total['verification_calls']
    total['native_accepted_tokens_per_call'] = total['native_accepted_draft_tokens'] / total['verification_calls']
    total['committed_progress_per_call'] = total['committed_progress'] / total['verification_calls']
    total['total_output_progress_per_call'] = total['output_tokens'] / total['verification_calls']
    total['calls_per_1000_output_tokens'] = total['verification_calls'] * 1000 / total['output_tokens']
    for repeat in range(3):
        for key, value in reported['endpoint']['repeat_totals'][str(repeat)][method].items():
            check(near(value, repeat_totals[repeat][method][key]), 'reported_repeat_total', f'{repeat}:{method}:{key}')

comparisons = {}
for candidate, baseline in (('KPLUS', 'A'), ('KPLUS', 'O'), ('A', 'O')):
    diffs = [by_request[i, baseline][0][1]['verification_calls'] - by_request[i, candidate][0][1]['verification_calls'] for i in range(1024)]
    comparisons[f'{candidate}_vs_{baseline}'] = dict(
        call_reduction_pct=100 * (1 - totals[candidate]['verification_calls'] / totals[baseline]['verification_calls']),
        request_time_reduction_pct=100 * (1 - totals[candidate]['request_latency_s'] / totals[baseline]['request_latency_s']),
        wins=sum(d > 0 for d in diffs), regressions=sum(d < 0 for d in diffs), ties=sum(d == 0 for d in diffs),
        gross_calls_saved=sum(d for d in diffs if d > 0), gross_calls_added=-sum(d for d in diffs if d < 0))

boundaries = read(FINAL / 'BOUNDARY_CHECKS.jsonl', lines=True)
expected_batches = [[b, r, p] for b in range(16) for r in range(3) for p in range(3)]
check([row['batch'] for row in boundaries] == expected_batches, 'all_144_timing_boundaries')
campaign = read(ROOT / 'campaign_001/STATUS.json')
for identity, receipt in [(str(row['batch']), row['check']) for row in boundaries] + [
        ('before_final', campaign['training_quiescence']), ('after_final', campaign['post_evaluation_boundary'])]:
    timing, resource = receipt['timing'], receipt['resource']
    check(timing['status'] == 'passed' and timing['transfers'] == timing['training'] == timing['foreign_gpu_processes'] == [], 'no_reported_timing_interference', identity)
    check(resource['status'] == 'passed' and resource['free_bytes'] >= resource['required_additional_and_reserve_bytes'], 'resource_admission', identity)

activations = read(FINAL / 'ACTIVATIONS.jsonl', lines=True)
switches = [s for i, s in enumerate(expected) if i == 0 or s['method'] != expected[i - 1]['method']]
check(len(activations) == len(switches), 'activation_count')
for slot, row in zip(switches, activations):
    method = slot['method']; native = row['native_status']; publication = native['publication']
    check(all(row[k] == v for k, v in slot.items()), 'activation_order', method)
    check(row['errors'] == [] and row['checkpoint_structure_valid'] is True
          and row['checkpoint_id'] == publication['checkpoint_path'] == checkpoints[method], 'activation_identity', method)
    check(native['status'] == publication['status'] == 'passed' and native['poisoned'] is None
          and native['capture_pending'] is False and native['cache_tokens'] == publication['cache_after_tokens'] == 0
          and native['attention_backend'] == 'fa3' and native['block_size'] == 16
          and all(publication[k] is True for k in ('helper_matches_current_layers', 'helper_norm_matches_current_layers', 'fused_enabled')), 'activation_native_state', method)
for method in METHODS:
    row = read(FINAL / f'WEIGHTS_{method}.json'); native = row['native_reply']
    check(row['errors'] == [] and native['status'] == 'passed' and native['all_equal'] is True
          and native['tensors'] == native['source_tensor_count'] == 58 and native['native_tensor_count'] == 43
          and native['checkpoint_path'] == checkpoints[method]
          and all(native[k] is True for k in ('helper_matches_current_layers', 'helper_norm_matches_current_layers', 'fused_enabled')), 'native_weight_equality_receipt', method)

for key in ('planned_generations', 'attempted_generations', 'returned_generations', 'completed_valid_generations'):
    check(status[key] == 9216, 'status_' + key)
for key in ('invalid_generations', 'pending_canonical_generations', 'unknown_generation_attempts',
            'unattempted_generations', 'extra_generations', 'optimizer_steps', 'feedback_collections'):
    check(status[key] == 0, 'status_' + key)
check(status['status'] == campaign['status'] == 'completed' and status['automatic_retry'] is False
      and status['interrupted_attempt'] is None, 'completed_without_retry')
for name in ('EXIT_CODE', 'CAMPAIGN_EXIT_CODE'):
    check((ROOT / name).read_text().strip() == '0', 'zero_exit', name)
postflight = read(ROOT / 'POSTFLIGHT.json')
check(postflight['status'] == postflight['seal']['status'] == 'passed'
      and postflight['seal']['execution_source'] == plan['execution_source'], 'postflight_source_identity')
original_audit = read(ROOT / 'campaign_001/FINAL_AUDIT.json')
check(original_audit['status'] == 'passed' and original_audit['errors'] == [] and original_audit['rows'] == 9216, 'production_audit_receipt')

manifest = json.dumps(inputs, sort_keys=True, separators=(',', ':')).encode()
result = dict(status='passed' if not failures else 'failed',
    material_passport=dict(workflow='ARS experiment-agent validate',
        evidence='Existing native raw outputs and sidecars; no model execution',
        authorship='Separate same-team raw reconstruction; auditor authored A loss component; not independent replication'),
    created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), execution_source=plan['execution_source'],
    root=str(ROOT), extra_gpu_work=0, extra_optimizer_steps=0, source_code_changes=0,
    raw_generations=len(expected), original_frozen_references=1024, native_sidecars=len(native_paths),
    native_verification_blocks=trace_blocks, total_checks=sum(checks.values()), checks=dict(checks),
    failure_count=sum(failures.values()), failures=dict(failures), failure_examples=examples,
    method_totals=totals, comparisons=comparisons, repeat_totals=repeat_totals,
    per_request_acceptance=acceptance_rows,
    order_balance=[dict(order=list(order), count=count) for order, count in sorted(orders.items())],
    round_endings=dict(round_endings), timing_boundary_receipts=len(boundaries), activation_receipts=len(activations),
    counters=dict(accepted_draft_tokens='Matched proposal tokens committed after EOS/cap clipping',
        native_accepted_draft_tokens='Native per-round counter: committed_progress minus one; differs when stopping clips bonus',
        committed_progress='Sum committed after verification, excludes one initial prefill-produced response token per request'),
    source_validator_reuse='None. Production FINAL_AUDIT receipt checked only as corroborating evidence.',
    limitations=['Uses retained verifier evidence; no independent target-only regeneration.',
                 'Interference claims are based on recorded before/after and 144 batch-boundary checks, not continuous external monitoring.',
                 'No training effectiveness, causal mechanism, full author reproduction or deployment amortization inference.',
                 'Confidence interval reconstruction delegated separately; this audit independently recomputes raw counts and point estimates.'],
    files_read=len(inputs), bytes_read=sum(v[0] for v in inputs.values()),
    input_manifest_sha256=hashlib.sha256(manifest).hexdigest(),
    elapsed_s=time.monotonic() - started)
print(json.dumps(result, indent=2, allow_nan=False))
