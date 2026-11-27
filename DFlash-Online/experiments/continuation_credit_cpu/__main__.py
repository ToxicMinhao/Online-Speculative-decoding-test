"""Run the prespecified CPU analysis and publish derived, prompt-free tables."""

import argparse
import csv
import gzip
import json
import os
from pathlib import Path
import platform
import resource
import time

import numpy as np
import scipy

from .allocation import attach_oof_scores, compare_allocations, sensitivity
from .association import run_association
from .prediction import run_prediction
from .records import load_records


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def write_csv(path, rows):
    if not rows:
        return
    opener = gzip.open if path.suffix == '.gz' else open
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with opener(path, 'wt', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def summarize_outcomes(rows, endpoint):
    result = []
    for group in ('ALL', 'HIGH', 'LOW', 'NO_ERROR'):
        selected = [row for row in rows if group == 'ALL'
                    or group == ('HIGH' if row['high'] else 'LOW' if row['has_error'] else 'NO_ERROR')]
        errors = [row for row in selected if row['has_error']]
        barriers = [row for row in selected if row['K'] < row['m']]
        result.append(dict(endpoint=endpoint, group=group, blocks=len(selected),
            responses=len({row['response_ordinal'] for row in selected}),
            Y_full=sum(row['Y_full'] for row in selected),
            Y_old=sum(row['Y_old'] for row in selected),
            beyond_K=sum(row['beyond_K'] for row in selected),
            crossed_next=sum(row['crossed_next'] for row in barriers),
            crossed_next_denominator=len(barriers),
            useful_repairs=sum(row['useful_repair'] for row in errors),
            repair_denominator=len(errors),
            lost_prefix=sum(row['lost_prefix'] for row in selected),
            blocks_losing_prefix=sum(row['lost_prefix'] > 0 for row in selected),
            old_prefix_occurrences=sum(row['L'] for row in selected),
            positive_shared_increment=sum((row['delta_u'] or 0) > 0 for row in selected),
            mean_u_old=float(np.mean([row['u_old'] for row in errors])) if errors else None,
            mean_u_shared=float(np.mean([row['u_shared'] for row in errors])) if errors else None))
    return result


def summarize_allocations(pools):
    results = []
    for score in sorted({row['score'] for row in pools}):
        group = [row for row in pools if row['score'] == score]
        moved = sum(row['moved_vs_old'] for row in group)
        repair_mass = sum(row['repair_mass'] for row in group)
        high_mass = sum(row['native_high_mass'] for row in group)
        energy = sum(row['increment_energy'] for row in group)
        results.append(dict(score=score, pools=len(group), active=sum(row['active'] for row in group),
            pools_changed_vs_old=sum(row['moved_vs_old'] > 1e-14 for row in group),
            changed_repairs=sum(row['changed_repairs'] for row in group),
            moved_coefficient_mass=moved, repair_mass_denominator=repair_mass,
            native_high_mass_denominator=high_mass,
            moved_fraction_all_repairs=moved / repair_mass if repair_mass else None,
            moved_fraction_native_high=moved / high_mass if high_mass else None,
            effective_half_l1=sum(row['effective_half_l1_vs_old'] for row in group),
            increment_energy=energy,
            increment_energy_fraction_removed=(1 - sum(row['centered_increment_energy'] for row in group) / energy) if energy else None,
            max_absolute_high_mass_error=max(abs(row['high_mass_error']) for row in group)))
    return results


def summarize_sensitivity(pools):
    keys = sorted({(row['parameter'], row['value'], row['score']) for row in pools})
    return [dict(parameter=parameter, value=value, score=score,
        pools=len(group),
        coefficient_half_l1=sum(row['coefficient_half_l1'] for row in group),
        effective_half_l1=sum(row['effective_half_l1'] for row in group),
        mean_denominator=float(np.mean([row['denominator'] for row in group])),
        max_absolute_effective_change=max(row['max_absolute_effective_change'] for row in group))
        for parameter, value, score in keys
        for group in [[row for row in pools if (row['parameter'], row['value'], row['score']) == (parameter, value, score)]]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--primary-root', type=Path, required=True)
    parser.add_argument('--tail-root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-revision', required=True)
    args = parser.parse_args()
    # These are the host-resource contract, not scientific acceptance gates.
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Run this retained-record analysis with CUDA_VISIBLE_DEVICES=""')
    thread_limits = {name: os.environ.get(name) for name in
                     ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')}
    if any(value not in ('1', '2') for value in thread_limits.values()) or len(os.sched_getaffinity(0)) > 2:
        raise RuntimeError('Set BLAS/OpenMP limits and process affinity to at most two CPUs')
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    timings = {}
    print('Loading compact native JSON records', flush=True)
    blocks, followers, secondary, inputs = load_records(args.primary_root, args.tail_root)
    # Raw response IDs, tokens and original anchor locations stay in retained inputs.
    ordinals = {(row['response_ordinal'], row['anchor']): index % 64 for index, row in enumerate(blocks)}
    def public(rows):
        return [{('block_ordinal' if key == 'anchor' else key):
                 (ordinals[(row['response_ordinal'], value)] if key == 'anchor' else value)
                 for key, value in row.items() if key != 'record_id'} for row in rows]

    write_json(args.output / 'INPUTS.json', dict(source_revision=args.source_revision,
        source_base='ab7d8231a70387c7e02f0d9d1feeedd1e80dd258', **inputs,
        preservation_sensitivity='deferred: canonical probabilities not retained',
        canonical_probability_evidence='PR64 position records retain margins/support only; PR65 C prefix_allocation receipts contain aggregate masses and by-slot summaries, not per-position q(y).',
        new_model_forwards=0, new_optimizer_steps=0, new_generations=0))
    outcomes = summarize_outcomes(blocks, 'native')
    if secondary:
        outcomes += summarize_outcomes(secondary, 'TAIL_secondary')
    write_csv(args.output / 'OUTCOMES.csv', outcomes)
    per_response = []
    for endpoint, rows in (('native', blocks), ('TAIL_secondary', secondary)):
        for ordinal in sorted({row['response_ordinal'] for row in rows}):
            per_response.extend(dict(response_ordinal=ordinal, **summary) for summary in
                summarize_outcomes([row for row in rows if row['response_ordinal'] == ordinal], endpoint))
    write_csv(args.output / 'RESPONSE_OUTCOMES.csv', per_response)
    timings['load_and_outcomes_s'] = time.perf_counter() - started

    print('Fitting fixed HIGH association and 2,000 response bootstrap draws', flush=True)
    phase = time.perf_counter()
    association = run_association(blocks)
    write_json(args.output / 'ASSOCIATION.json', association)
    timings['association_s'] = time.perf_counter() - phase
    print('Fitting grouped OOF reachability models and fixed-OOF bootstrap', flush=True)
    phase = time.perf_counter()
    prediction = run_prediction(followers)
    write_csv(args.output / 'OOF_PREDICTIONS.csv', public(prediction['predictions']))
    write_csv(args.output / 'PREDICTIVE_METRICS.csv', prediction['metrics'])
    write_json(args.output / 'PREDICTION.json', {key: value for key, value in prediction.items() if key != 'predictions'})
    timings['prediction_s'] = time.perf_counter() - phase

    print('Replaying response-pool allocations and one-at-a-time sensitivity', flush=True)
    phase = time.perf_counter()
    scores = ('u_old', 'u_shared')
    if prediction['status'] == 'complete':
        blocks = attach_oof_scores(blocks, followers, prediction['predictions'])
        scores += ('u_const', 'u_adapt')
    coefficients, pools = compare_allocations(blocks, scores)
    sensitivity_rows, sensitivity_pools = sensitivity(blocks, scores)
    write_csv(args.output / 'BLOCK_SCORES.csv', public(blocks))
    write_csv(args.output / 'TAIL_SECONDARY_BLOCKS.csv', public(secondary))
    write_csv(args.output / 'ALLOCATION.csv.gz', public(coefficients))
    write_csv(args.output / 'RESPONSE_ALLOCATION.csv', pools)
    allocation_summary = summarize_allocations(pools)
    sensitivity_summary = summarize_sensitivity(sensitivity_pools)
    write_csv(args.output / 'ALLOCATION_SUMMARY.csv', allocation_summary)
    write_csv(args.output / 'SENSITIVITY.csv.gz', public(sensitivity_rows))
    write_csv(args.output / 'RESPONSE_SENSITIVITY.csv', sensitivity_pools)
    write_csv(args.output / 'SENSITIVITY_SUMMARY.csv', sensitivity_summary)
    timings['allocation_and_sensitivity_s'] = time.perf_counter() - phase
    timings['total_wall_s'] = time.perf_counter() - started
    usage = resource.getrusage(resource.RUSAGE_SELF)
    write_json(args.output / 'SUMMARY.json', dict(status='complete', outcomes=outcomes,
        association_primary=association['primary'], association_secondary=association['secondary'],
        prediction_status=prediction['status'], prediction_comparisons=prediction['comparisons'],
        allocations=allocation_summary, sensitivity=sensitivity_summary,
        adaptive_allocation_status='complete' if prediction['status'] == 'complete' else 'deferred: incomplete OOF fits',
        preservation_sensitivity='deferred: canonical probabilities not retained'))
    write_json(args.output / 'RUNTIME.json', dict(timings=timings,
        user_cpu_s=usage.ru_utime, system_cpu_s=usage.ru_stime, peak_rss_kib=usage.ru_maxrss,
        host=platform.node(), python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
        pid=os.getpid(), cpu_affinity=sorted(os.sched_getaffinity(0)), thread_limits=thread_limits,
        cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES']))
    print(json.dumps(dict(status='complete', **timings)), flush=True)


if __name__ == '__main__':
    main()
