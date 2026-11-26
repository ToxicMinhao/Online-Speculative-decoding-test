"""Independent compact-record audit; no production imports or model execution."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import struct

import numpy as np

# Declared before reading outcomes. This checks a different host tanh
# implementation, not the already-required exact native parity comparison.
SUPPORT_AUDIT_ULPS = 4


class Audit:
    def __init__(self):
        self.checked = 0
        self.failures = []
        self.support = {}
        self.max_area_identity_error = 0.

    def equal(self, actual, expected, label):
        self.checked += 1
        if actual != expected:
            self.failures.append(dict(label=label, actual=actual, expected=expected))

    def require(self, condition, label):
        self.equal(bool(condition), True, label)

    def support_value(self, margin, value, source, label):
        self.equal(float(np.float32(value)), value, label + '.support_is_FP32')
        self.require(math.isfinite(value) and 0 <= value <= 1, label + '.bounded_support')
        expected = float(np.float32(max(0., math.tanh(float(np.float32(margin / 2))))))
        bits = lambda x: struct.unpack('<I', struct.pack('<f', x))[0]
        distance = 0 if value == expected else abs(bits(value) - bits(expected))
        record = self.support.setdefault(source, dict(positions=0, differing=0, maximum_ULPs=0,
                                                      maximum_absolute_error=0., allowance_ULPs=SUPPORT_AUDIT_ULPS))
        record['positions'] += 1
        record['differing'] += int(value != expected)
        record['maximum_ULPs'] = max(record['maximum_ULPs'], distance)
        record['maximum_absolute_error'] = max(record['maximum_absolute_error'], abs(value - expected))
        self.require(distance <= SUPPORT_AUDIT_ULPS, label + '.independent_tanh_ULPs')
        if margin <= 0:
            self.equal(value, 0., label + '.nonpositive_margin_support')


def roles(block):
    valid = block['valid']
    if len(valid) != 16 or valid[0]:
        raise ValueError('Invalid block geometry')
    count = 0
    for index in range(1, 16):
        if not valid[index]:
            break
        count += 1
    if valid != [False] + [True] * count + [False] * (15 - count):
        raise ValueError('Validity is not contiguous')
    matching = [block['labels'][i] == block['proposals'][i] for i in range(1, count + 1)]
    first = next((i + 1 for i, matched in enumerate(matching) if not matched), -1)
    length = count if first == -1 else first - 1
    run = 0
    if first != -1:
        for index in range(first + 1, count + 1):
            if block['labels'][index] != block['proposals'][index]:
                break
            run += 1
    endpoint = count if first == -1 else first + run
    return dict(labels=block['labels'], valid=valid,
                prefix=[0 < i <= length for i in range(16)],
                repair=[i == first for i in range(16)], m=count, L=length, e=first,
                K=endpoint, H=run + 1 if first != -1 else 0)


def audit_compact(block, audit, label, *, legacy=False):
    role = roles(block)
    audit.equal(block['roles'], role, label + '.roles')
    audit.equal(block['m'], role['m'], label + '.m')
    positions = {p['position']: p for p in block['positions']}
    audit.equal(len(positions), len(block['positions']), label + '.unique_positions')
    if not legacy:
        audit.equal(sorted(positions), list(range(1, role['m'] + 1)), label + '.all_valid_positions')
    for index, position in positions.items():
        here = f'{label}.position{index}'
        audit.require(1 <= index <= role['m'], here + '.position_valid')
        canonical, prediction = block['labels'][index], block['proposals'][index]
        margin = position['margin']
        audit.equal(position['canonical'], canonical, here + '.canonical')
        audit.equal(position['prediction'], prediction, here + '.prediction')
        audit.require(position['rival'] != canonical, here + '.noncanonical_rival')
        audit.require(math.isfinite(margin) and float(np.float32(margin)) == margin, here + '.finite_FP32_margin')
        audit.require((prediction == canonical and margin >= 0) or (prediction != canonical and margin <= 0),
                      here + '.native_margin_sign')
        audit.equal(position['margin_tie'], margin == 0, here + '.tie_flag')
        if 'correct' in position:
            audit.equal(position['correct'], prediction == canonical, here + '.correct_flag')
        if prediction != canonical:
            audit.equal(position['rival'], prediction, here + '.wrong_winner_is_rival')
        if margin == 0:
            audit.equal(prediction, min(canonical, position['rival']), here + '.lowest_ID_tie')
        audit.support_value(margin, position['support_fp32'], block['support_source'], here)
    return role, positions


def expected_metrics(parent, child):
    origin, current = roles(parent), roles(child)
    error, length, endpoint, count = (origin[k] for k in ('e', 'L', 'K', 'm'))
    run = endpoint - error if error != -1 else 0
    available = count - error if error != -1 else 0
    positions = {p['position']: p for p in parent['positions']}
    followers = [positions.get(i, {}).get('support_fp32') for i in range(error + 1, count + 1)] if error != -1 else []
    supports = followers[:run]
    if any(value is None for value in supports):
        raise ValueError('Missing within-run support')
    minima = [min(supports[:end]) for end in range(1, run + 1)]
    indices = [supports[:end].index(minimum) for end, minimum in enumerate(minima, 1)]
    ties = [supports[:end].count(minimum) > 1 for end, minimum in enumerate(minima, 1)]
    first_support = followers[0] if followers and followers[0] is not None else 0.
    area = math.fsum(minima)
    deficit = math.fsum(first_support - value for value in minima) / 14
    before = [positions[j]['margin'] for j in range(1, length + 1)]
    error_margin = positions[error]['margin'] if error != -1 else 0.
    signed = min(current['L'], endpoint) - length
    positive = max(0, signed)
    expected = dict(m=count, L=length, e=error, r=run, n=available, K=endpoint,
                    s=bool(run), high=error != -1 and run > 0,
                    beta=math.exp(-(error - 1) / 7) if error != -1 else 0.,
                    A=area, u=area / 14, D=deficit, h1=first_support,
                    error_margin=error_margin, d=max(0., -error_margin),
                    minimum_original_prefix_margin=min(before, default=0.),
                    empty_prefix=length == 0, error_tie=error != -1 and error_margin == 0,
                    prefix_tie=0 in before, follower_supports_fp32=followers,
                    running_minima_fp64=minima + [0.] * (available - run),
                    run_bottleneck_indices=indices, run_bottleneck_tied_terms=ties,
                    run_immediate_controlled_terms=indices.count(0),
                    run_later_controlled_terms=run - indices.count(0),
                    run_immediate_controlled_fraction=indices.count(0) / run if run else None,
                    run_later_controlled_fraction=(run - indices.count(0)) / run if run else None,
                    child_L=current['L'], Y=signed, raw_repair=error != -1 and child['proposals'][error] == parent['labels'][error],
                    useful_repair=signed > 0, prefix_damaged=signed < 0,
                    positive_progress=positive, lost_prefix=max(0, -signed),
                    reachable_origin_followers=max(positive - 1, 0),
                    damaged_prefix_positions=sum(child['proposals'][j] != parent['labels'][j] for j in range(1, length + 1)),
                    surviving_origin_followers=sum(child['proposals'][j] == parent['labels'][j] for j in range(error + 1, endpoint + 1)) if error != -1 else 0,
                    uncapped_progress=current['L'] - length,
                    inferred_zero_area_terms=available - run if len(positions) < count else 0)
    complete = len(positions) == count and error != -1
    expected['full_suffix_ties_observed'] = complete
    if complete:
        all_minima = [min(followers[:end]) for end in range(1, available + 1)]
        all_indices = [followers[:end].index(value) for end, value in enumerate(all_minima, 1)]
        expected.update(all_follower_bottleneck_indices=all_indices,
                        all_follower_bottleneck_tied_terms=[followers[:end].count(value) > 1 for end, value in enumerate(all_minima, 1)],
                        all_follower_immediate_controlled_fraction=all_indices.count(0) / available if available else None,
                        all_follower_later_controlled_fraction=(available - all_indices.count(0)) / available if available else None)
    else:
        expected.update(all_follower_bottleneck_indices=None, all_follower_bottleneck_tied_terms=None,
                        all_follower_immediate_controlled_fraction=None, all_follower_later_controlled_fraction=None)
    return expected


def audit_pair(row, audit, label, *, legacy=False):
    parent, child = row['parent'], row['child']
    audit.equal(parent['labels'], child['labels'], label + '.fixed_labels')
    audit.equal(parent['valid'], child['valid'], label + '.fixed_validity')
    audit_compact(parent, audit, label + '.parent', legacy=legacy)
    audit_compact(child, audit, label + '.child', legacy=legacy)
    expected = expected_metrics(parent, child)
    for key, value in expected.items():
        audit.equal(row[key], value, label + '.' + key)
    identity_error = abs(row['D'] - (row['r'] * row['h1'] / 14 - row['u']))
    audit.max_area_identity_error = max(audit.max_area_identity_error, identity_error)
    audit.require(identity_error <= 1e-14, label + '.area_identity_roundoff')
    audit.require(row['D'] >= 0, label + '.nonnegative_D')
    if row['e'] != -1:
        audit.require(-(row['e'] - 1) <= row['Y'] <= row['r'] + 1, label + '.signed_outcome_bounds')
    return expected


def audit_raw_binding(row, parent, child, audit, label):
    for endpoint, raw in (('parent', parent), ('child', child)):
        audit.equal(row[endpoint], raw, label + '.' + endpoint + '.raw_binding')


def legacy_binding(row, original_parent, original_child, audit, label):
    for endpoint, old in (('parent', original_parent), ('child', original_child)):
        compact = row[endpoint]
        audit.equal(compact['labels'], old['current']['labels'], label + '.' + endpoint + '.labels')
        audit.equal(compact['valid'], old['current']['valid'], label + '.' + endpoint + '.valid')
        audit.equal(compact['proposals'], old['proposals'], label + '.' + endpoint + '.proposals')
        for old_position, new_position in zip(old['positions'], compact['positions']):
            for key, value in old_position.items():
                audit.equal(new_position[key], value, label + '.' + endpoint + '.' + key)
        result = expected_metrics(row['parent'], compact)
        integer_metrics = dict(raw_repair=int(result['raw_repair']), useful_repair=int(result['useful_repair']),
            damaged_prefix_positions=result['damaged_prefix_positions'], damaged_prefix_blocks=int(result['prefix_damaged']),
            surviving_followers=result['surviving_origin_followers'], reachable_followers=result['reachable_origin_followers'],
            matching_prefix_length=result['child_L'], fixed_K_shortfall=max(0, row['K'] - result['child_L']))
        for key, value in integer_metrics.items():
            audit.equal(old['metrics'][key], value, label + '.' + endpoint + '.historical.' + key)
        audit.equal(old['origin'], row['parent']['roles'], label + '.' + endpoint + '.fixed_parent_origin')


def cohort_summary(rows):
    fields = ('raw_repair', 'useful_repair', 'prefix_damaged', 'positive_progress', 'lost_prefix',
              'reachable_origin_followers', 'damaged_prefix_positions', 'surviving_origin_followers',
              'uncapped_progress', 'Y')
    return dict(responses=len({r['record_id'] for r in rows}), blocks=len(rows),
                origin_errors=sum(r['e'] >= 1 for r in rows), HIGH=sum(r['high'] for r in rows),
                LOW=sum(r['e'] >= 1 and not r['high'] for r in rows),
                no_error=sum(r['e'] == -1 and r['m'] > 0 for r in rows), empty=sum(r['m'] == 0 for r in rows),
                **{field: sum(r[field] for r in rows) for field in fields})


def run(root, output):
    audit = Audit()
    contract = json.loads((root / 'CONTRACT.json').read_text())
    rows = json.loads((root / 'BLOCK_READOUTS.json').read_text())
    index = {(r['record_id'], r['anchor']): r for r in rows}
    audit.equal(len(index), len(rows), 'primary.unique_response_anchors')
    for binding in contract['features']:
        path = root / 'primary' / f'gpu{binding["physical_shard"]}'
        parents = json.loads((path / 'parent' / f'{binding["ordinal"]:03d}.json').read_text())
        children = json.loads((path / 'child' / f'{binding["ordinal"]:03d}.json').read_text())
        for record in (parents, children):
            audit.equal(record['anchors'], binding['anchors'], 'primary.sealed_anchors')
            audit.equal(record['record_id'], binding['record_id'], 'primary.sealed_response')
            audit.equal(len(record['blocks']), len(binding['anchors']), 'primary.blocks')
        for position, anchor in enumerate(binding['anchors']):
            row = index[binding['record_id'], anchor]
            label = f'primary.{binding["ordinal"]}.{position}'
            audit_raw_binding(row, parents['blocks'][position], children['blocks'][position], audit, label)
            audit_pair(row, audit, label)
    primary = cohort_summary(rows)
    primary_high = cohort_summary([r for r in rows if r['high']])
    audit.equal(primary['blocks'], contract['budget']['primary_paired_blocks'], 'primary.full_block_budget')
    audit.equal(primary['responses'], len(contract['features']), 'primary.full_response_cohort')
    legacy = json.loads((root / 'LEGACY_BLOCKS.json').read_text())
    source_parent = json.loads((root / 'legacy/parent.json').read_text())
    source_child = json.loads((root / 'legacy/child.json').read_text())
    summaries = {}
    for cohort in ('new_buffer', 'parent_training'):
        indexed = {(r['record_id'], r['anchor']): r for r in legacy[cohort]}
        audit.equal(len(indexed), 64, cohort + '.legacy_blocks')
        for before, after in zip(source_parent['sets'][cohort], source_child['sets'][cohort]):
            audit.equal(before['record_id'], after['record_id'], cohort + '.legacy_response')
            audit.equal(before['anchors'], after['anchors'], cohort + '.legacy_anchors')
            for ordinal, anchor in enumerate(before['anchors']):
                row = indexed[before['record_id'], anchor]
                label = cohort + f'.block{ordinal}'
                legacy_binding(row, before['blocks'][ordinal], after['blocks'][ordinal], audit, label)
                audit_pair(row, audit, label, legacy=True)
        summaries[cohort] = cohort_summary(legacy[cohort])
        audit.equal(summaries[cohort]['responses'], 8, cohort + '.legacy_responses')
        audit.equal(summaries[cohort]['origin_errors'], 44 if cohort == 'new_buffer' else 31,
                    cohort + '.historical_origin_errors')
    output.mkdir(parents=True, exist_ok=True)
    # Keep failure values out of the publication directory: they can contain IDs.
    private_failures = output.parent / 'METRICS_AUDIT_FAILURES_PRIVATE.json'
    private_failures.write_text(json.dumps(audit.failures, indent=2) + '\n')
    result = dict(status='passed' if not audit.failures else 'failed', checked_scalar_or_binding_claims=audit.checked,
                  failures=len(audit.failures), execution_commit=contract['execution_commit'],
                  auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  production_modules_imported=False, gpu_work=0, model_execution=False,
                  cohorts=dict(primary_all=primary, primary_HIGH=primary_high,
                               legacy_new_buffer=summaries['new_buffer'], legacy_parent_training=summaries['parent_training']),
                  support_operator_checks=audit.support, maximum_area_identity_error=audit.max_area_identity_error,
                  exact_checks='Native role reconstruction, source bindings, saved-FP32-support FP64 running-minimum area/D, outcomes and decompositions',
                  limitations=[
                      'The independent tanh reference uses FP64 libm rounded to FP32; its prospective allowance is four FP32 ULPs. Saved native supports, not recomputed supports, determine audited scores.',
                      'This arithmetic audit allowance does not relax native legacy parity or change model precision.',
                      'Full vocabulary logits and the two gathered logit values were not saved; strongest-rival optimality and native subtraction cannot be independently recomputed from compact margins alone.',
                      'Rival exclusion, native sign/tie consistency and retained margins are checked; native source/qualification evidence covers the unsaved head operation.',
                      'Legacy next-mismatch margins beyond origin K are unavailable. Their contribution to the area is exactly zero; missing margins are not fabricated.',
                      'These are fixed-block training-buffer outcomes, not native acceptance, verification calls, latency or causal weight-allocation gains.',
                  ])
    (output / 'AUDIT_METRICS.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    lines = ['# Independent continuation metric audit', '',
             'Verification status: ANALYZED. Independent CPU reconstruction of retained compact evidence; no new model execution.', '',
             f"Execution commit: `{contract['execution_commit']}`.", '',
             f"Status: **{result['status']}**; {audit.checked:,} scalar/binding checks, {len(audit.failures)} discrepancies.", '',
             '| Cohort | Responses | Blocks | Origin errors | HIGH | Useful repairs | Lost old-prefix tokens | Signed fixed-K progress |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, group in result['cohorts'].items():
        lines.append(f"| {name} | {group['responses']} | {group['blocks']} | {group['origin_errors']} | {group['HIGH']} | {group['useful_repair']} | {group['lost_prefix']} | {group['Y']} |")
    lines += ['', 'Native FP32 supports were retained for the exact FP64 area, D and outcome checks. The independent rounded-libm tanh check is reported separately:', '']
    for source, values in audit.support.items():
        lines.append(f"- `{source}`: {values['positions']:,} values; {values['differing']:,} differ from the independent rounded reference; maximum {values['maximum_ULPs']} ULP, absolute error {values['maximum_absolute_error']:.10g}.")
    lines += ['', *['- ' + text for text in result['limitations']], '',
              'The eight-response new-buffer witness overlaps the primary cohort. The eight-response parent-training witness is a separate small retention description. Neither is pooled into additional inferential responses.', '']
    (output / 'AUDIT_METRICS.md').write_text('\n'.join(lines))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    result = run(arguments.root, arguments.output)
    print(json.dumps(dict(status=result['status'], checks=result['checked_scalar_or_binding_claims'], failures=result['failures'])))
    raise SystemExit(result['status'] != 'passed')
