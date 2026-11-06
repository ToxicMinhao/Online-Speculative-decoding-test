"""Independent scalar/statistical audit; no project imports or model operations."""
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parents[2]
SOURCE = BASE / 'completion_review_001/results'
OUT = Path(__file__).resolve().parent
METHODS = ('O', 'KPLUS', 'A')
COUNTS = ('verification_calls', 'accepted_draft_tokens', 'committed_progress', 'output_tokens')
TIMES = ('request_latency_s', 'native_generate_s', 'cache_maintenance_s')
COMPARISONS = (('KPLUS', 'A'), ('KPLUS', 'O'), ('A', 'O'))
checks, errors = 0, []

def check(condition, label):
    global checks
    checks += 1
    if not condition:
        errors.append(label)

def close(actual, expected, label):
    check(math.isfinite(float(actual)) and math.isfinite(float(expected)) and
          math.isclose(float(actual), float(expected), rel_tol=1e-11, abs_tol=1e-9), label)

def compare_tree(actual, expected, label):
    if isinstance(actual, dict):
        check(set(actual) == set(expected), label + '.keys')
        for key in actual.keys() & expected.keys():
            compare_tree(actual[key], expected[key], label + '.' + key)
    elif isinstance(actual, list):
        check(len(actual) == len(expected), label + '.length')
        for index, (a, b) in enumerate(zip(actual, expected)):
            compare_tree(a, b, f'{label}[{index}]')
    elif isinstance(actual, (int, float)) and not isinstance(actual, bool):
        close(actual, expected, label)
    else:
        check(actual == expected, label)

summary = json.loads((SOURCE / 'SUMMARY.json').read_text())
with (SOURCE / 'ALL_REPEATS.csv').open(newline='') as stream:
    raw = list(csv.DictReader(stream))
check(summary['status'] == 'valid_complete' and summary['errors'] == [], 'complete_summary')
check(summary['execution_source']['commit'] == '681336c5e5c5caa0aa67560e46ffe83ce81e9351', 'execution_commit')
check(len(raw) == 9216, '9216_rows')
groups = defaultdict(list)
request_ids, checkpoints, orders = {}, defaultdict(set), defaultdict(list)
for line, row in enumerate(raw):
    label = f'row{line}'
    for field in (*COUNTS, 'panel_ordinal', 'repeat', 'event', 'seed', 'block_ordinal', 'position', 'prompt_tokens'):
        row[field] = int(row[field])
    for field in TIMES:
        row[field] = float(row[field])
        check(math.isfinite(row[field]) and row[field] >= 0, label + '.' + field)
    check(row['method'] in METHODS and row['seed'] == 17 and row['event'] == 58 and row['phase'] == 'final', label + '.identity')
    check(row['validation_status'] == 'valid' and row['valid_measurement'] == 'True', label + '.valid')
    check(json.loads(row['errors']) == [] and json.loads(row['structural_errors']) == [], label + '.errors')
    check(row['finish_reason'] in ('eos', 'length') and row['cap_hit'] in ('True', 'False'), label + '.termination')
    check((row['finish_reason'] == 'length') == (row['cap_hit'] == 'True'), label + '.cap')
    check(0 < row['verification_calls'] <= row['output_tokens'] <= 2048, label + '.calls_output')
    check(0 <= row['accepted_draft_tokens'] <= row['output_tokens'], label + '.accepted')
    check(0 <= row['committed_progress'] <= row['output_tokens'], label + '.progress')
    check(row['native_generate_s'] <= row['request_latency_s'] + 1e-9, label + '.native_latency')
    check(row['cache_maintenance_s'] <= row['request_latency_s'] + 1e-9, label + '.cache_latency')
    ordinal, method = row['panel_ordinal'], row['method']
    if ordinal in request_ids:
        check(request_ids[ordinal] == row['request_id'], label + '.same_request')
    request_ids[ordinal] = row['request_id']
    checkpoints[method].add(row['checkpoint_id'])
    groups[ordinal, method].append(row)
    order_key = (row['block_ordinal'], row['repeat'])
    if row['panel_ordinal'] % 64 == 0:
        orders[order_key].append(method)
check(sorted(request_ids) == list(range(1024)) and len(set(request_ids.values())) == 1024, '1024_distinct_requests')
for method in METHODS:
    check(checkpoints[method] == {summary['checkpoints'][method]}, method + '.checkpoint_identity')
check(len(orders) == 48 and set(Counter(tuple(order) for order in orders.values()).values()) == {8}, 'six_orders_eight_each')
# Independently reconstruct the exact predeclared block/repeat order.
expected_sequence = []
for block in range(16):
    base = ('O', 'KPLUS', 'A') if block % 2 == 0 else ('O', 'A', 'KPLUS')
    for repeat in range(3):
        shift = (block // 2 + repeat) % 3
        sequence = base[shift:] + base[:shift]
        for position, method in enumerate(sequence):
            for ordinal in range(block * 64, (block + 1) * 64):
                expected_sequence.append((ordinal, method, repeat, block, position))
for line, (row, expected) in enumerate(zip(raw, expected_sequence)):
    check(tuple(row[k] for k in ('panel_ordinal', 'method', 'repeat', 'block_ordinal', 'position')) == expected,
          f'order_row{line}')

arrays = {method: {key: [] for key in (*COUNTS, *TIMES)} for method in METHODS}
paired = []
for ordinal in range(1024):
    saved = {'panel_ordinal': ordinal, 'request_id': request_ids[ordinal]}
    outputs = []
    for method in METHODS:
        records = sorted(groups[ordinal, method], key=lambda r: r['repeat'])
        check([r['repeat'] for r in records] == [0, 1, 2], f'{ordinal}.{method}.repeats')
        first = records[0]
        for field in (*COUNTS, 'finish_reason', 'cap_hit', 'checkpoint_id', 'prompt_tokens'):
            check(all(r[field] == first[field] for r in records), f'{ordinal}.{method}.{field}.consistent')
        values = {field: first[field] for field in COUNTS}
        values.update({field: math.fsum(r[field] for r in records) / 3 for field in TIMES})
        values.update(cap_hit=first['cap_hit'] == 'True', finish_reason=first['finish_reason'],
                      checkpoint_id=first['checkpoint_id'], count_trajectory_repeat=0, timing_repeats=3)
        saved[method] = values
        for field in arrays[method]:
            arrays[method][field].append(values[field])
        outputs.append((first['output_tokens'], first['finish_reason'], first['cap_hit']))
    check(len(set(outputs)) == 1, f'{ordinal}.matched_lengths_termination')
    paired.append(saved)
arrays = {m: {f: np.asarray(v, dtype=np.float64) for f, v in fields.items()} for m, fields in arrays.items()}
with (SOURCE / 'PAIRED_REQUESTS.csv').open(newline='') as stream:
    reported_paired = list(csv.DictReader(stream))
check(len(reported_paired) == 1024, 'paired_csv_count')
for expected, saved in zip(paired, reported_paired):
    check(int(saved['panel_ordinal']) == expected['panel_ordinal'] and saved['request_id'] == expected['request_id'], 'paired_csv_identity')
    for method in METHODS:
        for field, value in expected[method].items():
            encoded = saved[method + '_' + field]
            if isinstance(value, bool):
                check(encoded == str(value), f'paired.{method}.{field}')
            elif isinstance(value, (int, float)):
                close(float(encoded), value, f'paired.{method}.{field}')
            else:
                check(encoded == value, f'paired.{method}.{field}')

totals = {}
for method in METHODS:
    v = arrays[method]
    t = {field: int(v[field].sum()) for field in COUNTS}
    t.update({field: float(v[field].sum()) for field in TIMES})
    t['calls_per_1000_output_tokens'] = 1000 * t['verification_calls'] / t['output_tokens']
    t['accepted_tokens_per_call'] = t['accepted_draft_tokens'] / t['verification_calls']
    t['committed_progress_per_call'] = t['committed_progress'] / t['verification_calls']
    totals[method] = t
compare_tree(totals, summary['endpoint']['method_totals'], 'method_totals')

indices = np.random.Generator(np.random.PCG64(20261102)).integers(0, 1024, size=(10000, 1024))
comparisons = {}
for candidate, reference in COMPARISONS:
    c, r = arrays[candidate], arrays[reference]
    delta = r['verification_calls'] - c['verification_calls']
    effect = dict(request_wins=int(np.count_nonzero(delta > 0)), request_regressions=int(np.count_nonzero(delta < 0)),
                  request_ties=int(np.count_nonzero(delta == 0)), request_denominator=1024,
                  gross_calls_saved=int(delta[delta > 0].sum()), gross_calls_added=int(-delta[delta < 0].sum()))
    for field, name in [('verification_calls', 'call_reduction'), *[(f, f + '_reduction') for f in TIMES]]:
        numerator, denominator = float(c[field].sum()), float(r[field].sum())
        draw_reference = r[field][indices].sum(axis=1)
        check(bool((draw_reference > 0).all()), candidate + reference + field + '.positive_draw_denominators')
        draws = 100 * (1 - c[field][indices].sum(axis=1) / draw_reference)
        effect[name] = dict(point_pct=100 * (1 - numerator / denominator),
                            ci95_pct=np.quantile(draws, [.025, .975], method='linear').tolist(),
                            undefined_bootstrap_draws=int((~np.isfinite(draws)).sum()),
                            candidate_sum=numerator, reference_sum=denominator)
    comparisons[candidate + '_vs_' + reference] = effect
compare_tree(comparisons, summary['endpoint']['comparisons'], 'comparisons')
check(summary['endpoint']['interpretation']['endpoint_label'] == 'favors_KPLUS' and
      comparisons['KPLUS_vs_A']['request_latency_s_reduction']['ci95_pct'][0] > 0, 'primary_interpretation')
repeat_totals = {}
for repeat in range(3):
    repeat_totals[str(repeat)] = {}
    for method in METHODS:
        selected = [row for row in raw if row['repeat'] == repeat and row['method'] == method]
        check(len(selected) == 1024, f'repeat{repeat}.{method}.count')
        repeat_totals[str(repeat)][method] = {field: sum(row[field] for row in selected) for field in (*COUNTS, *TIMES)}
compare_tree(repeat_totals, summary['endpoint']['repeat_totals'], 'repeat_totals')
compare_tree(repeat_totals, json.loads((SOURCE / 'REPEAT_TOTALS.json').read_text()), 'repeat_totals_file')

# Descriptive order/repeat checks only; no additional hypothesis test or selection.
description = dict(repeat_request_time_reductions={}, block_request_time_reductions={},
                   cap_requests=sum(row['A']['cap_hit'] for row in paired),
                   output_token_min=int(arrays['A']['output_tokens'].min()), output_token_max=int(arrays['A']['output_tokens'].max()))
for c, r in COMPARISONS:
    key = c + '_vs_' + r
    description['repeat_request_time_reductions'][key] = [100 * (1 - repeat_totals[str(rep)][c]['request_latency_s'] /
              repeat_totals[str(rep)][r]['request_latency_s']) for rep in range(3)]
    description['block_request_time_reductions'][key] = [100 * (1 - arrays[c]['request_latency_s'][start:start+64].sum() /
              arrays[r]['request_latency_s'][start:start+64].sum()) for start in range(0, 1024, 64)]

v = json.loads((SOURCE / 'VALIDITY.json').read_text())
check(v['status'] == 'passed' and v['errors'] == [], 'validity_receipt')
for field in ('planned_generations', 'attempted_generations', 'returned_generations', 'completed_valid_generations'):
    check(v['progress'][field] == 9216, field)
for field in ('extra_generations', 'invalid_generations', 'pending_canonical_generations', 'unknown_generation_attempts', 'unattempted_generations', 'optimizer_steps', 'feedback_collections'):
    check(v['progress'][field] == 0, field)
expected_work = dict(online_generations=7473, correctness_generations=4, terminal_preparation_forwards=7424,
                     response_forwards=14848, response_backwards=14848, optimizer_steps=7424)
compare_tree(summary['arm_progress']['actual_work']['completed'], expected_work, 'completed_main_work')
compare_tree(summary['arm_progress']['actual_work']['attempts'], expected_work, 'attempted_main_work')
check(not any(summary['arm_progress']['actual_work']['unknown'].values()), 'unknown_main_work')
check(summary['arm_progress']['events_completed'] == 58 and summary['arm_progress']['tail_completed'] == 49, 'events_tail')

input_identity = {p.name: dict(bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                  for p in (SOURCE/'ALL_REPEATS.csv', SOURCE/'PAIRED_REQUESTS.csv', SOURCE/'SUMMARY.json', SOURCE/'REPEAT_TOTALS.json', SOURCE/'VALIDITY.json')}
result = dict(status='passed' if not errors else 'failed', checks=checks, discrepancies=errors,
              execution_commit=summary['execution_source']['commit'], source_files=input_identity,
              method_totals=totals, comparisons=comparisons, repeat_totals=repeat_totals,
              descriptive_checks=description, scope='Independent CPU arithmetic on retained report tables; no model rerun or raw-token revalidation.')
(OUT/'STATISTICS_AUDIT.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(dict(status=result['status'], checks=checks, discrepancies=errors,
                     primary=comparisons['KPLUS_vs_A'], descriptive_checks=description), indent=2))
raise SystemExit(bool(errors))
