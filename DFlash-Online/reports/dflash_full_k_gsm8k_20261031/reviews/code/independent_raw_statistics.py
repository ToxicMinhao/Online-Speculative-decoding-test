"""Read-only CPU audit; imports no campaign or reporting modules.

Run through SSH stdin with the approved environment helper. All output is JSON
on stdout; no remote file is created or modified. Native logits/weights are not
recomputed. This checks retained records and independently recomputes statistics.
"""
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from collections import Counter, defaultdict

import numpy as np

ROOT = Path('/data/sources/minhao/runs/dflash_full_k_gsm8k_20261031/production_002')
COMMIT = '0698eeb2ddd97b9e2ac6953989dc82acbc0a1e34'
METHODS = ('F', 'O', 'K', 'K_HALF')
COUNTS = ('verification_calls', 'accepted_draft_tokens', 'committed_progress', 'output_tokens')
TIMES = ('request_latency_s', 'native_generate_s', 'cache_maintenance_s')
COMPARISONS = (('K_HALF', 'O'), ('K', 'O'), ('K_HALF', 'K'), ('K_HALF', 'F'), ('K', 'F'))
checks = Counter()
files = {}
native_paths = set()
totals = Counter()
result = {'status': 'running', 'execution_commit': COMMIT, 'errors': [],
          'verification_status': 'ANALYZED', 'scope': 'Independent CPU record/statistical reconstruction; no GPU reproduction',
          'numpy_version': np.__version__, 'python': sys.executable, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}
started = time.time()


def check(ok, category, detail=''):
    checks[category] += 1
    if not ok:
        raise AssertionError(category + ': ' + detail)


def read(path, lines=False):
    path = Path(path)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if str(path) in files:
        check(files[str(path)]['sha256'] == digest, 'unchanged_during_audit', str(path))
    files[str(path)] = {'bytes': len(raw), 'sha256': digest}
    return [json.loads(x) for x in raw.splitlines() if x] if lines else json.loads(raw)


def eq_number(a, b, name):
    check(type(a) in (int, float) and type(b) in (int, float) and math.isfinite(a) and math.isfinite(b), 'finite_statistics', name)
    check(math.isclose(a, b, rel_tol=2e-13, abs_tol=2e-11), 'reported_numeric_equality', name + f' {a} != {b}')


def trace_check(record, request, *, canonical, checkpoint, label, session, historical=False):
    prompt, response, stop = record['prompt_ids'], record['response_ids'], record['stop_token_ids']
    meta, cap = record['metadata'], record['metadata']['capture']
    check(prompt == request['prompt_ids'] and stop == request['stop_token_ids'], 'input_tokens', label)
    check(record['max_new_tokens'] == 2048 and 0 < len(response) <= 2048 and len(prompt) + 2048 <= 4096,
          'context_and_output_bounds', label)
    check(all(type(v) is int and v >= 0 for v in prompt + response + stop), 'token_integer_domain', label)
    check(not any(v in stop for v in response[:-1]), 'first_eos', label)
    finish = 'eos' if response[-1] in stop else 'length'
    check(finish == record['finish_reason'] and (finish == 'eos' or len(response) == 2048), 'termination', label)
    check(response == canonical['response_ids'] and finish == canonical['finish_reason'], 'frozen_output_equality', label)
    check(record['checkpoint_id'] == checkpoint and record['purpose'] == 'evaluation', 'checkpoint_and_purpose', label)
    if not historical:
        check(record['record_id'] == label and record['session_id'] == session and record['request_index'] == request['panel_ordinal'], 'native_identity', label)
    check(meta['runtime'] == 'pinned_SGLang_FA3' and meta['block_size'] == 16 and meta['context_limit'] == 4096,
          'native_runtime', label)
    check(meta['cache_policy'] == 'idle_full_flush_before_every_request_and_publication' and meta['feedback_capture_in_timed_request'] is False,
          'cache_and_capture_policy', label)
    check(cap['collect_features'] is False and cap['feature_bytes'] == 0 and cap['native_capture_validated'] is True,
          'capture_disabled', label)
    check(cap['prompt_ids'] == prompt and cap['response_ids'] == response and cap['blocks'] == meta['trace'], 'capture_binding', label)
    check(cap['covered_token_ids'] == prompt + response[:-1] and cap['target_argmax_ids'] == response, 'native_target_label_binding', label)
    check(read(meta['native_trace_path']) == cap, 'native_duplicate_evidence', label)
    if not historical:
        check(meta['native_trace_path'] not in native_paths, 'unique_native_trace', label)
        native_paths.add(meta['native_trace_path'])
    offset = accepted_total = 0
    for block in meta['trace']:
        totals['historical_blocks' if historical else 'new_evaluation_blocks'] += 1
        draft, posterior = block['draft_tokens'], block['target_argmax']
        width, progress = block['verify_size'], block['committed_progress']
        check(type(width) is int and width == len(draft) + 1 == len(posterior) and 1 <= width <= 16,
              'block_geometry', label)
        check(type(progress) is int and 1 <= progress <= width and all(type(v) is int and v >= 0 for v in draft + posterior),
              'block_token_progress_domain', label)
        prefix = 0
        while prefix < len(draft) and draft[prefix] == posterior[prefix]:
            prefix += 1
        decision = draft[:prefix] + [posterior[prefix]]
        check(progress <= len(decision) and block['committed_tokens'] == decision[:progress] == posterior[:progress],
              'first_error_commit', label)
        check(block['output_start'] == offset and block['anchor_token'] == response[offset] and
              block['committed_tokens'] == response[offset + 1:offset + progress + 1], 'trace_output_reconstruction', label)
        check(block['positions'] == list(range(len(prompt) + offset, len(prompt) + offset + width)), 'position_geometry', label)
        check(block['accepted_draft_tokens'] == min(prefix, progress) and block['native_accepted_draft_tokens'] == progress - 1
              and block['raw_accepted_draft_tokens_before_stop'] == prefix and block['stop_clipped'] == (progress < prefix + 1),
              'acceptance_counts', label)
        offset += progress
        accepted_total += min(prefix, progress)
    check(offset == len(response) - 1, 'whole_trace_progress', label)
    counts = dict(verification_calls=len(meta['trace']), accepted_draft_tokens=accepted_total,
                  committed_progress=offset, output_tokens=len(response))
    check(0 <= accepted_total <= offset and counts['verification_calls'] > 0, 'count_bounds', label)
    for key in COUNTS[:-1]:
        check(meta[key] == counts[key], 'native_scalar_counts', key + ' ' + label)
    for key, value in {'calls_per_1000_output_tokens': 1000 * counts['verification_calls'] / len(response),
                       'committed_progress_per_call': offset / counts['verification_calls']}.items():
        check(meta[key] == value, 'native_normalized_counts', key + ' ' + label)
    times = {key: (cap[key] if key == 'cache_maintenance_s' else meta[key]) for key in TIMES}
    check(all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in times.values()), 'timing_domain', label)
    check(times['request_latency_s'] >= times['native_generate_s'] > 0, 'timing_nesting', label)
    totals['historical_records' if historical else 'new_evaluation_records'] += 1
    return {**counts, **times, 'finish_reason': finish, 'cap_hit': len(response) == 2048,
            'prompt_tokens': len(prompt), 'checkpoint_id': checkpoint}


def audit_stage(folder, slots, requests, canonicals, checkpoints):
    protocol, status = read(folder / 'PROTOCOL.json'), read(folder / 'STATUS.json')
    check(protocol['schedule'] == slots and protocol['checkpoints'] == checkpoints, 'stage_protocol', str(folder))
    check(protocol['capture'] is False and protocol['extra_warmup_generations'] == 0 and protocol['historical_call_gate'] is False
          and protocol['performance_win_required'] is False, 'stage_fixed_scope', str(folder))
    check(read(folder / 'REQUESTS.json') == requests and read(folder / 'CANONICAL_INPUT.json') == canonicals, 'stage_sealed_inputs', str(folder))
    rows = read(folder / 'MEASUREMENTS.jsonl', lines=True)
    check(len(rows) == len(slots), 'stage_completeness', str(folder))
    for key in ('planned_generations', 'attempted_generations', 'returned_generations', 'completed_valid_generations'):
        check(status[key] == len(slots), 'stage_status_counts', key)
    for key in ('invalid_generations', 'pending_canonical_generations', 'unknown_generation_attempts', 'unattempted_generations', 'extra_generations', 'optimizer_steps', 'feedback_collections'):
        check(status[key] == 0, 'stage_no_extra_work', key)
    check(status['status'] == 'completed' and status['automatic_retry'] is False and status['interrupted_attempt'] is None,
          'stage_terminal_status', str(folder))
    for method, checkpoint in checkpoints.items():
        weights = read(folder / ('WEIGHTS_' + method + '.json'))
        w = weights['native_reply']
        check(weights['errors'] == [] and w['status'] == 'passed' and w['all_equal'] is True and w['tensors'] == w['source_tensor_count'] == 58
              and w['native_tensor_count'] == 43 and w['checkpoint_path'] == checkpoint,
              'recorded_weight_equality', method)
        check(all(w[k] is True for k in ('helper_matches_current_layers', 'helper_norm_matches_current_layers', 'fused_enabled')),
              'recorded_fused_helpers', method)
    expected_activations = [slot for i, slot in enumerate(slots) if i == 0 or slots[i-1]['method'] != slot['method']]
    acts = read(folder / 'ACTIVATIONS.jsonl', lines=True)
    check(len(acts) == len(expected_activations), 'activation_count', str(folder))
    for act, slot in zip(acts, expected_activations):
        check(all(act[k] == v for k, v in slot.items()), 'activation_order', str(folder))
        checkpoint = checkpoints[slot['method']]
        ns, pub = act['native_status'], act['native_status']['publication']
        check(act['errors'] == [] and act['checkpoint_structure_valid'] is True and act['checkpoint_id'] == checkpoint
              and ns['status'] == pub['status'] == 'passed' and ns['poisoned'] is None and ns['capture_pending'] is False
              and ns['cache_tokens'] == pub['cache_after_tokens'] == 0 and pub['checkpoint_path'] == checkpoint
              and ns['attention_backend'] == 'fa3' and ns['block_size'] == 16,
              'recorded_publication_and_cache_flush', str(folder))
    for row, slot in zip(rows, slots):
        ordinal = slot['panel_ordinal']; request = requests[ordinal]
        label = f"{ordinal:04d}_r{slot['repeat']}_{slot['method']}"
        check(all(row[k] == v for k, v in slot.items()) and row['original_index'] == request['original_index'], 'measurement_order_identity', label)
        check(row['validation_status'] == 'valid' and row['valid_measurement'] is True and row['errors'] == row['structural_errors'] == [],
              'saved_validation_status', label)
        path = folder / (label + '.json')
        check(row['record_path'] == str(path), 'raw_path_binding', label)
        rec = read(path)
        reconstructed = trace_check(rec, request, canonical=canonicals[request['request_id']], checkpoint=checkpoints[slot['method']],
                                    label=label, session='full_k_gsm8k_evaluation')
        for k, v in reconstructed.items():
            check(row[k] == v, 'measurement_reconstruction', label + ' ' + k)
        validation = read(folder / (label + '_VALIDATION.json'))
        check(validation['status'] == 'valid' and validation['errors'] == [] and validation['canonical_source'] == canonicals[request['request_id']]['source_record_path'],
              'validation_binding', label)
    return rows


def main():
    plan = read(ROOT / 'sealed/PLAN.json')
    check(plan['execution_source']['commit'] == COMMIT and plan['execution_source']['dirty'] is False, 'execution_source')
    requests = plan['evaluation_requests']
    canonical = read(plan['canonicals'])
    check(len(requests) == len(canonical) == 1024 and [r['panel_ordinal'] for r in requests] == list(range(1024))
          and len({r['request_id'] for r in requests}) == 1024, 'complete_unique_panel')
    for request in requests:
        c = canonical[request['request_id']]
        original = read(c['source_record_path'])
        values = trace_check(original, request, canonical=c, checkpoint=c['F_checkpoint'], label='historical_' + request['request_id'], session=None, historical=True)
        check({k: values[k] for k in COUNTS} == c['F_counts'], 'frozen_canonical_counts')
    slots = []
    positions, precedence = Counter(), Counter()
    for block in range(16):
        for repeat in range(3):
            order = list(METHODS if (block // 4) % 2 == 0 else ('F', 'K_HALF', 'K', 'O'))
            shift = (block + repeat) % 4
            order = order[shift:] + order[:shift]
            for left, right in itertools.combinations(order, 2):
                precedence[left + '_before_' + right] += 1
            for position, method in enumerate(order):
                positions[method, position] += 1
                for ordinal in range(64 * block, 64 * (block + 1)):
                    slots.append(dict(seed=17, event=58, phase='final', block_ordinal=block,
                                      panel_ordinal=ordinal, request_id=requests[ordinal]['request_id'],
                                      repeat=repeat, method=method, position=position))
    check(slots == plan['evaluation_schedule'], 'sealed_final_schedule')
    check(set(positions.values()) == {12} and set(precedence.values()) == {24} and len(precedence) == 12, 'position_and_precedence_balance')
    final_rows = audit_stage(ROOT / 'campaign_001/evaluation/final', slots, requests, canonical, plan['checkpoints'])
    result['schedule_balance'] = {'generations': len(slots), 'block_repeat_orders': 48,
        'method_positions': {m: [positions[m, p] for p in range(4)] for m in METHODS}, 'pairwise_precedence': dict(precedence)}
    grouped = defaultdict(list)
    for row in final_rows:
        grouped[row['panel_ordinal'], row['method']].append(row)
    vectors = {m: {k: [] for k in COUNTS + TIMES} for m in METHODS}
    for ordinal in range(1024):
        for m in METHODS:
            records = sorted(grouped[ordinal, m], key=lambda r:r['repeat'])
            check([r['repeat'] for r in records] == [0, 1, 2], 'three_distinct_repeats')
            for k in COUNTS + ('finish_reason', 'cap_hit', 'checkpoint_id'):
                check(all(r[k] == records[0][k] for r in records), 'repeat_trajectory_consistency')
            for k in COUNTS:
                vectors[m][k].append(records[0][k])
            for k in TIMES:
                vectors[m][k].append(sum(r[k] for r in records) / 3)
    vectors = {m: {k: np.asarray(v, dtype=np.float64) for k, v in vals.items()} for m, vals in vectors.items()}
    rng = np.random.Generator(np.random.PCG64(20261102))
    indices = rng.integers(0, 1024, size=(10000, 1024))
    effects = {}
    for a, b in COMPARISONS:
        diff = vectors[b]['verification_calls'] - vectors[a]['verification_calls']
        effect = dict(request_wins=int((diff > 0).sum()), request_regressions=int((diff < 0).sum()), request_ties=int((diff == 0).sum()),
                      request_denominator=1024, gross_calls_saved=int(np.maximum(diff, 0).sum()), gross_calls_added=int(np.maximum(-diff, 0).sum()))
        for key in ('verification_calls',) + TIMES:
            x, y = vectors[a][key], vectors[b][key]
            draws = 100 * (1 - x[indices].sum(axis=1) / y[indices].sum(axis=1))
            field = 'call_reduction' if key == 'verification_calls' else key + '_reduction'
            effect[field] = dict(point_pct=100 * (1 - x.sum() / y.sum()), ci95_pct=np.quantile(draws, [.025, .975], method='linear').tolist(),
                                 candidate_sum=float(x.sum()), reference_sum=float(y.sum()), undefined_bootstrap_draws=0)
        effects[a + '_vs_' + b] = effect
    result['comparisons'] = effects
    result['method_totals'] = {m: {k: (int(v.sum()) if k in COUNTS else float(v.sum())) for k, v in vals.items()} for m, vals in vectors.items()}
    result['bootstrap'] = dict(seed=20261102, generator='PCG64', resamples=10000, method='linear', percentiles=[2.5, 97.5],
                               unit='one paired request; timing uses three-repeat per-request mean', indices_sha256=hashlib.sha256(indices.tobytes()).hexdigest())
    result['gain_harm_vs_F'] = {}
    for m in METHODS[1:]:
        d = vectors['F']['verification_calls'] - vectors[m]['verification_calls']
        result['gain_harm_vs_F'][m] = dict(gross_gain_calls=int(np.maximum(d, 0).sum()), gross_harm_calls=int(np.maximum(-d, 0).sum()), net_calls_saved=int(d.sum()))
    result['repeat_totals'] = {str(repeat): {m: {k: sum(r[k] for r in final_rows if r['method'] == m and r['repeat'] == repeat) for k in COUNTS + TIMES} for m in METHODS} for repeat in range(3)}
    result['learning_curve'] = []
    for arm in ('K', 'K_HALF'):
        for event in (14, 29, 44, 58):
            if event == 58:
                rows = [r for r in final_rows if r['method'] == arm and r['repeat'] == 0 and r['panel_ordinal'] < 128]
            else:
                schedule = [dict(seed=17, event=event, phase='intermediate', block_ordinal=0, panel_ordinal=i,
                                 request_id=requests[i]['request_id'], repeat=0, method=arm, position=0) for i in range(128)]
                folder = ROOT / f'campaign_001/arms/{arm}/evaluation/event_{event:03d}'
                cp = str(ROOT / f'campaign_001/arms/{arm}/trainer/events/event_{event:03d}/export')
                rows = audit_stage(folder, schedule, requests, canonical, {arm: cp})
            deltas = [canonical[r['request_id']]['F_counts']['verification_calls'] - r['verification_calls'] for r in rows]
            baseline = sum(canonical[r['request_id']]['F_counts']['verification_calls'] for r in rows)
            calls = sum(r['verification_calls'] for r in rows)
            result['learning_curve'].append(dict(arm=arm, event=event, requests=len(rows), verification_calls=calls,
                frozen_verification_calls=baseline, call_reduction_vs_F_pct=100*(1-calls/baseline), request_wins=sum(v>0 for v in deltas),
                request_regressions=sum(v<0 for v in deltas), request_ties=sum(v==0 for v in deltas), output_tokens=sum(r['output_tokens'] for r in rows)))
    report = read(ROOT / 'final_report/SUMMARY.json')
    check(report['status'] == 'valid' and report['errors'] == [] and report['execution_source']['commit'] == COMMIT, 'report_status_identity')
    analysis = report['analysis']
    for name in ('comparisons', 'method_totals', 'gain_harm_vs_F', 'repeat_totals'):
        def compare(a, b, context):
            if isinstance(a, dict):
                for k, v in a.items():
                    compare(v, b[k], context + '/' + k)
            elif isinstance(a, list):
                check(len(a) == len(b), 'report_array_length', context)
                for i, v in enumerate(a):
                    compare(v, b[i], context + '/' + str(i))
            else:
                eq_number(a, b, context)
        compare(result[name], analysis[name], name)
    check(result['schedule_balance'] == report['schedule_balance'], 'report_schedule_equality')
    for derived, recorded in zip(result['learning_curve'], report['learning_curve']):
        check(all(recorded[k] == v for k, v in derived.items()), 'report_learning_curve_equality')
    check(totals['new_evaluation_records'] == 13056 and totals['historical_records'] == 1024, 'complete_record_audit')
    result['lengths_and_termination'] = report['lengths_and_termination']
    result['status'] = 'passed'


try:
    main()
except BaseException as exc:
    result['status'] = 'failed'
    result['errors'].append(repr(exc))
    result['traceback'] = traceback.format_exc()
finally:
    result['check_counts'] = dict(checks)
    result['total_checks'] = sum(checks.values())
    result['audit_totals'] = dict(totals)
    result['audited_file_count'] = len(files)
    result['audited_unique_bytes'] = sum(v['bytes'] for v in files.values())
    result['source_manifest_sha256'] = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    result['elapsed_s'] = time.time() - started
    result['source_file_identities'] = files
    print(json.dumps(result, indent=2, allow_nan=False))
sys.exit(0 if result['status'] == 'passed' else 1)
