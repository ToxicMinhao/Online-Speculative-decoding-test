"""Independent CPU-only audit of retained Point3 mechanism and cost records.

No project model/assessment/report helper is imported. Public tables use only
set/response/block ordinals, never private IDs, anchors, labels or proposals.
"""
import csv
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = RAW = OUT = HISTORICAL_C = None
ARMS = ('F_C_SUM', 'F_C_MAX')
ENDPOINTS = ('parent', 'C', *ARMS)
SETS = ('new_buffer', 'parent_training')
COUNTS = ('raw_repair', 'useful_repair', 'damaged_prefix_positions',
          'damaged_prefix_blocks', 'surviving_followers', 'reachable_followers',
          'matching_prefix_length', 'fixed_K_shortfall')
CHECKS, SOURCES = [], {}


def check(value, name):
    if not value:
        raise AssertionError(name)
    CHECKS.append(name)


def read(path):
    data = path.read_bytes()
    SOURCES[str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)] = dict(
        bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    return json.loads(data)


def close(a, b):
    return math.isclose(a, b, rel_tol=2e-6, abs_tol=2e-7)


def softplus(value):
    return max(value, 0.) + math.log1p(math.exp(-abs(value)))


def write_csv(name, rows):
    with (OUT / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def flat(value):
    if isinstance(value, list):
        return [item for row in value for item in flat(row)]
    return [value]


def audit_block(block):
    origin, proposals = block['origin'], block['proposals']
    labels, valid = origin['labels'], origin['valid']
    length = 0
    for j in range(1, len(labels)):
        if not valid[j] or proposals[j] != labels[j]:
            break
        length += 1
    check(length == block['current']['L'], 'current consecutive agreement')
    damage = sum(flag and proposals[j] != labels[j] for j, flag in enumerate(origin['prefix']))
    error, endpoint = origin['e'], origin['K']
    repaired = error >= 1 and proposals[error] == labels[error]
    followers = range(error + 1, endpoint + 1) if error >= 1 else ()
    metrics = dict(raw_repair=int(repaired), useful_repair=int(repaired and damage == 0),
        damaged_prefix_positions=damage, damaged_prefix_blocks=int(damage > 0),
        surviving_followers=sum(proposals[j] == labels[j] for j in followers),
        reachable_followers=sum(length >= j for j in followers),
        matching_prefix_length=length, fixed_K_shortfall=max(0, endpoint - length))
    check(all(block['metrics'][key] == value for key, value in metrics.items()), 'native discrete metrics')
    positions = block['positions']
    check([row['position'] for row in positions] == list(range(1, endpoint + 1)), 'fixed K support')
    selected, multiplicity = [], [0] * len(labels)
    bottleneck = None
    for index, row in enumerate(positions):
        if bottleneck is None or row['margin'] < positions[bottleneck]['margin']:
            bottleneck = index
        selected.append(bottleneck + 1)
        multiplicity[bottleneck + 1] += 1
    for index, row in enumerate(positions):
        check(row['bottleneck_position'] == selected[index], 'earliest minimum branch')
        check(row['bottleneck_multiplicity'] == multiplicity[index + 1], 'branch multiplicity')
        phi = softplus(-row['margin']) / math.log(2)
        maximum = softplus(-positions[selected[index] - 1]['margin']) / math.log(2)
        check(close(row['assessment_contribution_fp32'], (endpoint - index) * phi), 'SUM contribution formula')
        check(close(row['max_assessment_contribution_fp32'], maximum), 'MAX contribution formula')
    summation = math.fsum(row['assessment_contribution_fp32'] for row in positions)
    maximum = math.fsum(row['max_assessment_contribution_fp32'] for row in positions)
    check(close(summation, block['assessment_sum_fp64']), 'SUM contribution reduction')
    check(close(maximum, block['max_assessment_sum_fp64']), 'MAX contribution reduction')
    check(metrics['fixed_K_shortfall'] <= maximum + 1e-6 and maximum <= summation + 1e-6,
          'fixed K shortfall bound')
    return {**metrics, 'SUM_contribution_fp64': summation, 'MAX_contribution_fp64': maximum,
            'origin_prefix_positions': origin['L'], 'origin_error_blocks': int(error >= 1),
            'origin_followers': len(followers), 'origin_K': endpoint,
            'valid_positions': origin['m']}


def witnesses():
    paths = {endpoint: (RAW / 'qualification/native' / f'WITNESS_{endpoint}.json'
                        if endpoint in ('parent', 'C') else RAW / 'arms' / endpoint / 'WITNESS.json')
             for endpoint in ENDPOINTS}
    data = {endpoint: read(path) for endpoint, path in paths.items()}
    rows, response_rows, block_rows, measurements = [], [], [], {}
    for endpoint, evidence in data.items():
        check(evidence['status'] == 'passed' and set(evidence['sets']) == set(SETS), 'witness sets complete')
        for name, values in evidence['sets'].items():
            parent = data['parent']['sets'][name]
            check(len(values['responses']) == 8, 'eight witness responses')
            total = {key: 0 for key in (*COUNTS, 'origin_prefix_positions', 'origin_error_blocks',
                                        'origin_followers', 'origin_K', 'valid_positions')}
            assessments = {'SUM': [], 'MAX': []}
            for ordinal, (response, first) in enumerate(zip(values['responses'], parent['responses'], strict=True)):
                check(response['record_id'] == first['record_id'] and response['anchors'] == first['anchors'],
                      'common parent response and anchors')
                check(len(response['blocks']) == len(first['blocks']) == 8, 'eight witness blocks')
                local = dict(total.fromkeys(total, 0))
                numerator = dict(SUM=0., MAX=0.)
                for block_index, (block, initial) in enumerate(zip(response['blocks'], first['blocks'], strict=True)):
                    check(block['origin'] == initial['origin'], 'fixed parent roles')
                    metrics = audit_block(block)
                    measurements[endpoint, name, ordinal, block_index] = metrics
                    block_rows.append(dict(endpoint=endpoint, witness_set=name, response_ordinal=ordinal,
                                           block_ordinal=block_index, **metrics))
                    for key in total:
                        total[key] += metrics[key]
                        local[key] += metrics[key]
                    for kind in assessments:
                        numerator[kind] += metrics[kind + '_contribution_fp64']
                count = response['nonempty_blocks']
                check(count == sum(b['origin']['m'] > 0 for b in response['blocks']), 'nonempty response denominator')
                means = {kind: numerator[kind] / count if count else 0. for kind in assessments}
                for kind, value in means.items():
                    check(close(value, response['assessments'][kind]['fp64']), kind + ' response mean')
                    assessments[kind].append(value)
                response_rows.append(dict(endpoint=endpoint, witness_set=name, response_ordinal=ordinal,
                    blocks=8, nonempty_blocks=count, **local,
                    R_SUM_fp64=means['SUM'], R_MAX_fp64=means['MAX']))
            row = dict(endpoint=endpoint, witness_set=name, responses=8, blocks=64, **total)
            for kind, values_by_response in assessments.items():
                value = math.fsum(values_by_response) / 8
                check(close(value, values['assessments'][kind]['fp64']), kind + ' equal response mean')
                row['R_' + kind + '_fp64'] = value
                row['R_' + kind + '_fp32'] = values['assessments'][kind]['fp32']
            rows.append(row)
    differences, response_differences, paired_summary = [], [], []
    for candidate, reference in ((*[(arm, 'parent') for arm in ENDPOINTS[1:]],
                                   ('F_C_MAX', 'F_C_SUM'), ('F_C_SUM', 'C'), ('F_C_MAX', 'C'))):
        for name in SETS:
            paired_responses = []
            for ordinal in range(8):
                paired_blocks = []
                for index in range(8):
                    first, second = measurements[reference, name, ordinal, index], measurements[candidate, name, ordinal, index]
                    # Subtract matching represented contributions before FP64
                    # reduction, not already-rounded endpoint totals.
                    old = data[reference]['sets'][name]['responses'][ordinal]['blocks'][index]
                    new = data[candidate]['sets'][name]['responses'][ordinal]['blocks'][index]
                    benefits = {kind: math.fsum(a[field] - b[field] for a, b in
                        zip(old['positions'], new['positions'], strict=True)) for kind, field in
                        (('SUM', 'assessment_contribution_fp32'), ('MAX', 'max_assessment_contribution_fp32'))}
                    change = dict(comparison=candidate + '_vs_' + reference,
                        witness_set=name, response_ordinal=ordinal, block_ordinal=index,
                        **{key + '_change': second[key] - first[key] for key in COUNTS},
                        SUM_contribution_benefit_fp64=benefits['SUM'],
                        MAX_contribution_benefit_fp64=benefits['MAX'])
                    differences.append(change)
                    paired_blocks.append(change)
                count = data[reference]['sets'][name]['responses'][ordinal]['nonempty_blocks']
                response_change = dict(comparison=candidate + '_vs_' + reference,
                    witness_set=name, response_ordinal=ordinal, blocks=8, nonempty_blocks=count,
                    **{key + '_change': sum(row[key + '_change'] for row in paired_blocks) for key in COUNTS},
                    **{kind + '_benefit_fp64': math.fsum(row[kind + '_contribution_benefit_fp64']
                        for row in paired_blocks) / count if count else 0. for kind in ('SUM', 'MAX')})
                response_differences.append(response_change)
                paired_responses.append(response_change)
            paired_summary.append(dict(comparison=candidate + '_vs_' + reference,
                witness_set=name, responses=8, blocks=64,
                **{key + '_change': sum(row[key + '_change'] for row in paired_responses) for key in COUNTS},
                **{kind + '_benefit_fp64': math.fsum(row[kind + '_benefit_fp64'] for row in paired_responses) / 8
                   for kind in ('SUM', 'MAX')},
                prefix_response_wins=sum(row['matching_prefix_length_change'] > 0 for row in paired_responses),
                prefix_response_losses=sum(row['matching_prefix_length_change'] < 0 for row in paired_responses),
                prefix_response_ties=sum(row['matching_prefix_length_change'] == 0 for row in paired_responses)))
    write_csv('WITNESS_SUMMARY.csv', rows)
    write_csv('WITNESS_RESPONSES.csv', response_rows)
    write_csv('WITNESS_BLOCKS.csv', block_rows)
    write_csv('WITNESS_PAIRED_BLOCK_CHANGES.csv', differences)
    write_csv('WITNESS_PAIRED_RESPONSE_CHANGES.csv', response_differences)
    write_csv('WITNESS_PAIRED_SUMMARY.csv', paired_summary)
    return rows


def training():
    costs, sparse, physical, accounting = [], [], {}, {}
    for arm in ARMS:
        path = RAW / 'arms' / arm
        state = read(path / 'STATUS.json')
        event = state['training']
        check(state['status'] == 'completed' and event['optimizer_steps'] == 128
              and event['response_presentations'] == 256, 'complete own event')
        reference_s = scoring_s = window_s = 0.
        clipped = 0
        work_counts = {}
        for window in range(1, 129):
            base = path / 'trainer/events/event_002' / f'allocation_{window:03d}'
            result, work = read(base / 'RESULT.json'), read(base / 'WORK.json')
            check(result['window'] == window and result['native']['native_adam_steps'] == 1, 'one native update')
            check(not work['pending'] and work['attempted'] == work['completed'], 'complete physical work')
            ref = result['reference']
            check(ref['native_adam_steps'] == ref['committed_presentations'] == ref['sampling_calls'] == 0
                  and ref['response_forwards'] == ref['response_backwards'] == 2
                  and ref['rng_restored'] and ref['gradients_cleared'], 'isolated native C reference')
            check(result['coefficients_sealed_before_commit'], 'sealed coefficients')
            clipped += int(result['native_clipping_active'])
            reference_s += ref['wall_s']
            scoring_s += result['score_s']
            window_s += result['total_window_s']
            for key, value in work['completed'].items():
                work_counts[key] = work_counts.get(key, 0) + value
            for allocation in result['allocation']:
                before, after = flat(allocation['mass_before']), flat(allocation['mass_after'])
                check(all(math.isclose(a, b, abs_tol=2e-5, rel_tol=2e-5) for a, b in zip(before, after, strict=True)),
                      'reported allocator mass conservation')
            if window in (1, 64, 128):
                chunks = [item for response in result['preview']['bottlenecks'] for item in response.values()]
                allocation = result['allocation']
                responses = result['responses']
                row = dict(arm=arm, window=window,
                    assessment_kind=result['preview']['assessment_kind'],
                    blocks=sum(c['blocks'] for c in chunks),
                    prefix_positions=sum(r['prefix_positions'] for r in responses),
                    prefix_beta_mass=sum(sum(flat(a['mass_before'])) for a in allocation),
                    margin_ties=sum(c['margin_ties'] for c in chunks),
                    wrong_rival_ties=sum(c['wrong_rival_ties'] for c in chunks),
                    tied_prefixes=sum(c['tied_prefixes'] for c in chunks),
                    reference_pools=sum(sum(flat(a['reference_branch'])) for a in allocation),
                    zero_centered_positions=sum(r['zero_centered_positions'] for r in responses),
                    centered_score_min=min(r['centered_score_min'] for r in responses if r['centered_score_min'] is not None),
                    centered_score_max=max(r['centered_score_max'] for r in responses if r['centered_score_max'] is not None),
                    half_l1_movement_beta=sum(r['allocation_mass_moved_from_beta'] for r in responses),
                    half_l1_movement_current_C=sum(r['allocation_mass_moved_from_C'] for r in responses),
                    predicted_alignment=sum(r['predicted_score_alignment'] for r in responses),
                    native_preclip_norm=result['native_preclip_norm'],
                    native_clipping_active=result['native_clipping_active'])
                for role in ('P', 'E', 'C'):
                    for field in ('distinct_positions', 'multiplicity'):
                        row['bottleneck_' + role + '_' + field] = sum(c['roles'][role][field] for c in chunks)
                row['bottleneck_total_distinct_positions'] = sum(row['bottleneck_' + role + '_distinct_positions'] for role in ('P', 'E', 'C'))
                row['bottleneck_total_multiplicity'] = sum(row['bottleneck_' + role + '_multiplicity'] for role in ('P', 'E', 'C'))
                sparse.append(row)
        observed = [row['response_observation'] for row in event['records']]
        def average(key):
            return math.fsum(n / d for row in observed for n, d in [row['ratio_metrics'][key]]) / len(observed)
        costs.append(dict(arm=arm, **event['costs'], scoring_s_nested_in_training=scoring_s,
            native_C_reference_s_nested_in_scoring=reference_s, total_window_s=window_s,
            candidate_training_s_training_minus_scoring=event['costs']['training_s'] - scoring_s,
            startup_s=state['startup']['wall_s'], whole_arm_wall_s=state['wall_s'],
            process_peak_allocated_bytes=state['peak_allocated_bytes'],
            process_peak_reserved_bytes=state['peak_reserved_bytes'],
            mean_response_FKL=average('forward_kl'), mean_response_PAIR=average('repair_pair_auxiliary'),
            mean_response_objective=average('objective_loss'),
            committed_presentations=256, committed_Adam=128,
            reference_response_forwards=256, reference_response_backwards=256,
            reference_Adam=0, final_witness_response_forwards=16, native_clipped_windows=clipped))
        physical[arm] = work_counts
        ordinary = event['objective_work']['work']
        reference = event['point3_reference_work']['physical']
        combined = {key: ordinary.get(key, 0) + reference.get(key, 0)
                    for key in set(ordinary) | set(reference)}
        check(combined['draft_forwards'] == 1280
              and combined['layer_forwards'] == 6400, 'full event physical forward accounting')
        accounting[arm] = dict(
            scope='Event physical instrumentation; excludes startup and final witnesses',
            event_counters_excluding_isolated_C_reference=ordinary,
            isolated_C_reference_counters=reference,
            combined_instrumented_event_counters=combined,
            startup=state['startup']['physical_work'],
            startup_response_forwards=2, startup_response_backwards=2, startup_Adam=0,
            final_witness_response_forwards=16,
            nested_counter_notice='WORK operation counts overlap these counters; do not add them again. '
                'response_forwards is committed-observation instrumentation and does not count all physical forwards. '
                'Vocabulary-head/objective chunks include recorded checkpoint recomputation.',
            event_objective_subtimers=event['objective_work'])
    write_csv('TRAINING_COSTS.csv', costs)
    historical = read(HISTORICAL_C)
    check(historical['status'] == 'completed' and historical['training']['optimizer_steps'] == 128,
          'historical native C event complete')
    retained_cost = dict(arm='C', scope='historical PR60 measurement; not a fresh matched timing',
        execution_commit='225802657d630c0985abd68d6e7453f3e47c5b74',
        **historical['training']['costs'], startup_s=historical['startup']['wall_s'])
    (OUT / 'HISTORICAL_C_COST.json').write_text(json.dumps(retained_cost, indent=2) + '\n')
    comparisons = []
    for item in costs:
        baseline = retained_cost['training_s']
        comparisons.append(dict(candidate=item['arm'], reference='C_historical',
            candidate_training_s=item['training_s'], reference_training_s=baseline,
            extra_training_s=item['training_s'] - baseline,
            training_time_ratio=item['training_s'] / baseline,
            extra_training_percent=100 * (item['training_s'] / baseline - 1),
            comparison_scope='historical context; not contemporaneous matched overhead'))
    comparisons.append(dict(candidate='F_C_MAX', reference='F_C_SUM',
        candidate_training_s=costs[1]['training_s'], reference_training_s=costs[0]['training_s'],
        extra_training_s=costs[1]['training_s'] - costs[0]['training_s'],
        training_time_ratio=costs[1]['training_s'] / costs[0]['training_s'],
        extra_training_percent=100 * (costs[1]['training_s'] / costs[0]['training_s'] - 1),
        comparison_scope='observed serial event timers; not isolated causal kernel overhead'))
    write_csv('TRAINING_COST_COMPARISONS.csv', comparisons)
    write_csv('SPARSE_DIAGNOSTICS.csv', sparse)
    (OUT / 'PHYSICAL_WORK.json').write_text(json.dumps(physical, indent=2) + '\n')
    (OUT / 'PHYSICAL_ACCOUNTING.json').write_text(json.dumps(accounting, indent=2) + '\n')
    return costs, sparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', required=True, type=Path,
                        help='Retained Point3 directory containing raw/')
    parser.add_argument('--historical-c-status', required=True, type=Path,
                        help='Authenticated PR60 CONTROL STATUS.json (historical cost context)')
    args = parser.parse_args()
    ROOT, HISTORICAL_C = args.run_root.resolve(), args.historical_c_status.resolve()
    RAW, OUT = ROOT / 'raw', ROOT / 'public_mechanism'
    OUT.mkdir(exist_ok=True)
    summary = witnesses()
    costs, sparse = training()
    audit = dict(status='passed', scope='independent CPU scalar and native-proposal recomputation',
        execution_commit='6f7f3a5d1769a630e4ec852205eb06ca55047341',
        checks=len(CHECKS), source_files=len(SOURCES),
        limitations=['No original model tensors or full gradient vectors reconstructed',
            'Native head-selected vocabulary rivals are retained evidence, not independently recomputed from full logits',
            'Training witnesses are neither held-out requests nor native serving measurements',
            'Process GPU peaks include startup; host-memory peak not recorded here'],
        no_model_forwards=True, no_training=True, no_GPU=True)
    (OUT / 'MECHANISM_AUDIT.json').write_text(json.dumps(audit, indent=2) + '\n')
    (ROOT / 'MECHANISM_SOURCE_IDENTITIES_PRIVATE.json').write_text(json.dumps(SOURCES, indent=2) + '\n')
    print(json.dumps(dict(summary=summary, costs=costs, sparse=sparse, audit=audit), indent=2))
