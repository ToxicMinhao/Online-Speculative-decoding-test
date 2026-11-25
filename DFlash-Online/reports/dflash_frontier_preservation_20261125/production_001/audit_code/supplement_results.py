#!/usr/bin/env python3
"""Offline reductions of retained sparse diagnostics, timings and public tables."""

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from export_results import ARMS, COMMIT, COUNTS, TIMES, require

ROOT = Path(__file__).resolve().parent
RAW = ROOT / 'raw'
OUT = ROOT / 'export_001'
IDENTITIES = {}


def read(path, lines=False):
    path = Path(path)
    payload = path.read_bytes()
    IDENTITIES[str(path.relative_to(ROOT))] = dict(
        bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest()
    )
    return (
        [json.loads(row) for row in payload.splitlines() if row]
        if lines
        else json.loads(payload)
    )


def write(name, value):
    path = OUT / name
    payload = json.dumps(value, indent=2, allow_nan=False) + '\n'
    if path.exists():
        require(
            path.read_text() == payload,
            'Preserve differing previous supplementary output: ' + name,
        )
        return
    path.write_text(payload)


def numeric_range(values):
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    require(bool(np.isfinite(array).all()), 'Nonfinite retained sparse scalar')
    return (
        dict(minimum=float(array.min()), maximum=float(array.max()))
        if len(array)
        else None
    )


def sparse_allocations():
    observations = read(
        RAW / 'arms/FRONTIER_P/trainer/events/event_002/WINDOW_OBSERVATIONS.jsonl',
        lines=True,
    )
    require(
        [row['window'] for row in observations] == list(range(1, 129)),
        'Native window order differs',
    )
    result = dict(
        execution_commit=COMMIT,
        scope='Retained production windows 1,64,128; scalar response/block reductions only',
        contrast_definition='Per block maximum absolute centered score. The full signed per-token contrast range was not retained.',
        predicted_alignment_definition='Sum over blocks of (assigned-beta)*score; centered form and variance identity retained separately. This is a local prediction, not measured endpoint benefit.',
        step_displacement_norms=dict(
            status='unavailable',
            reason='Not instrumented in production; no rerun or substitution',
        ),
        windows=[],
    )
    for window in (1, 64, 128):
        value = read(
            RAW
            / f'arms/FRONTIER_P/trainer/events/event_002/allocation_{window:03d}/RESULT.json'
        )
        native = observations[window - 1]
        require(
            value['window'] == window and native['global_step'] == window + 128,
            'Sparse step binding changed',
        )
        preview = value['preview']
        row = dict(
            window=window,
            global_step=native['global_step'],
            g0_global_norm_fp32=preview['gradient_norm'],
            g0_clip_factor=preview['clip'],
            g0_clipping_active=preview['clipping_active'],
            g0_norm_scope='FP32 global norm of native BF16 reference-gradient values',
            final_native_gradient_norm=native['gradient_norm'],
            final_native_clipped=native['clipped'],
            final_native_clip_factor=1.0 if not native['clipped'] else None,
            final_native_clip_factor_scope='Exact unity when retained native clipped flag is false; no clipped factor was separately logged',
            response_assessments=preview['response_assessments'],
            window_assessment=preview['window_assessment'],
            score_s=value['score_s'],
            total_window_s=value['total_window_s'],
            responses=[],
        )
        require(
            len(value['allocation']) == len(value['response_scores']) == 2,
            'Sparse window lost a response',
        )
        for ordinal, (allocation, scores) in enumerate(
            zip(value['allocation'], value['response_scores'])
        ):
            arrays = {
                key: np.asarray(numbers).reshape(-1)
                for key, numbers in allocation.items()
            }
            n = len(arrays['prefix_positions'])
            require(
                all(len(a) == n and np.isfinite(a).all() for a in arrays.values()),
                'Sparse block dimensions/nonfinite differ',
            )
            before, after = arrays['mass_before'], arrays['mass_after']
            error = after - before
            require(
                bool((np.abs(error) <= 2e-7 + 2e-6 * np.abs(before)).all()),
                'Retained prefix mass failed inherited tolerance',
            )
            branch = arrays['reference_branch'].astype(bool)
            center = arrays['maximum_centered_score']
            require(
                bool((branch == (center == 0)).all()),
                'Reference branch diagnostic differs',
            )
            raw, centered, rhs = (
                arrays[key]
                for key in (
                    'signed_benefit',
                    'centered_signed_benefit',
                    'variance_identity_rhs',
                )
            )
            require(
                bool((np.abs(centered - rhs) <= 2e-7 + 2e-6 * np.abs(rhs)).all()),
                'Retained signed variance identity failed',
            )
            row['responses'].append(
                dict(
                    response_ordinal=ordinal,
                    blocks=n,
                    prefix_positions=int(arrays['prefix_positions'].sum()),
                    prefix_pools=int((arrays['prefix_positions'] > 0).sum()),
                    multi_position_prefix_pools=int(
                        (arrays['prefix_positions'] >= 2).sum()
                    ),
                    score_range=dict(
                        minimum=scores['minimum'], maximum=scores['maximum']
                    ),
                    weighted_mean_score_range=numeric_range(
                        arrays['weighted_mean_score']
                    ),
                    block_maximum_absolute_centered_score_range=numeric_range(center),
                    block_maximum_absolute_score_range=numeric_range(
                        arrays['maximum_absolute_score']
                    ),
                    reference_branch_blocks=int(branch.sum()),
                    active_allocation_blocks=int((~branch).sum()),
                    reference_branch_multi_position_pools=int(
                        (branch & (arrays['prefix_positions'] >= 2)).sum()
                    ),
                    prefix_mass_before=float(before.sum()),
                    prefix_mass_after=float(after.sum()),
                    prefix_mass_signed_error=float(error.sum()),
                    prefix_mass_max_absolute_error=float(np.abs(error).max()),
                    prefix_mass_l1_error=float(np.abs(error).sum()),
                    mass_moved_from_beta=scores['mass_moved_from_beta'],
                    mass_moved_from_old_C=scores['mass_moved_from_old_C'],
                    sum_of_retained_block_half_l1_mass_moved=float(
                        arrays['half_l1_mass_moved'].sum()
                    ),
                    predicted_signed_alignment=float(raw.sum()),
                    predicted_centered_alignment=float(centered.sum()),
                    variance_identity_rhs_sum=float(rhs.sum()),
                    block_signed_alignment_range=numeric_range(raw),
                    block_centered_alignment_range=numeric_range(centered),
                    raw_centered_alignment_difference_sum=float(
                        arrays['raw_centered_benefit_difference'].sum()
                    ),
                )
            )
        result['windows'].append(row)
    return result


def timing_breakdown():
    plan = read(RAW / 'sealed/PLAN.json')
    features = read(RAW / 'shared/FEATURE_IDENTITIES.json')
    selected = read(RAW / 'shared/INPUTS.json')
    costs = read(ROOT / 'COLLECTION_COST_RECORDS_PRIVATE.json')
    require(
        costs['status'] == 'completed' and len(costs['files']) == 128,
        'Collection cost set incomplete',
    )
    expected = {
        'shared/event_002/loop/serving/' + row['request']['request_id'] + '.costs.json'
        for row in selected
    }
    require(
        {row['path'] for row in costs['files']} == expected,
        'Collection timing source set differs',
    )
    keys = (
        'serve_call_s',
        'buffer_persistence_s',
        'generation_validation_and_recording_s',
        'native_request_latency_s',
    )
    totals = {key: 0.0 for key in keys}
    for record in costs['files']:
        data = record['content']
        require(
            record['bytes'] > 0 and len(record['sha256']) == 64,
            'Collection cost lacks source identity',
        )
        require(
            all(
                type(data[key]) in (int, float)
                and math.isfinite(data[key])
                and data[key] >= 0
                for key in keys
            ),
            'Invalid collection timing',
        )
        require(
            math.isclose(
                data['serve_call_s'],
                data['buffer_persistence_s']
                + data['generation_validation_and_recording_s'],
                abs_tol=1e-9,
            ),
            'Collection scopes do not reconcile',
        )
        for key in keys:
            totals[key] += data[key]
    totals['teacher_preparation_s'] = sum(
        row['preparation']['teacher_preparation_s'] for row in features
    )
    totals['feature_persistence_s'] = sum(
        row['preparation']['feature_persistence_s'] for row in features
    )
    init = read(ROOT / 'INITIALIZATION_PRIVATE.json')
    initialization = {}
    for name, relative in (
        ('shared_worker_loading_s', 'shared/native/ready.json'),
        ('evaluation_worker_loading_s', 'evaluation_native/ready.json'),
    ):
        records = [row for row in init if row['path'] == relative]
        require(len(records) == 1, 'Initialization timing binding differs')
        initialization[name] = records[0]['content']['loading_s']
    activations = read(RAW / 'evaluation/ACTIVATIONS.jsonl', lines=True)
    require(
        [a['activation'] for a in activations] == list(range(12)),
        'Activation order changed',
    )
    eval_cost = dict(
        backend_loading_s=sum(row['loading_s'] for row in activations),
        publication_activation_s=sum(
            row['publication']['activation_s'] for row in activations
        ),
        native_activation_s=sum(
            row['publication']['native_activation_s'] for row in activations
        ),
        activations=[
            dict(
                activation=row['activation'],
                arm=row['method'],
                loading_s=row['loading_s'],
                activation_s=row['publication']['activation_s'],
                native_activation_s=row['publication']['native_activation_s'],
            )
            for row in activations
        ],
    )
    return dict(
        execution_commit=COMMIT,
        status='reconciled',
        shared_responses=128,
        scopes='Nested timing scopes overlap. Worker loading is included in backend loading; native activation is included in publication activation; generation and buffer persistence sum to serve_call. Do not add nested subtotals.',
        shared_collection=totals,
        initialization=initialization,
        evaluation=eval_cost,
        trainer_loading=dict(
            status='unavailable',
            reason='No separate production trainer-loading timer was retained; phase residuals are not substituted',
        ),
    )


def table_reductions():
    summary = read(OUT / 'SERVING_SUMMARY.json')
    with (OUT / 'PER_REQUEST.csv').open() as stream:
        requests = list(csv.DictReader(stream))
    with (OUT / 'ALL_REPEATS.csv').open() as stream:
        repeats = list(csv.DictReader(stream))
    require(
        len(requests) == 128 and len(repeats) == 792, 'Public table row count differs'
    )
    require(
        [int(row['request_ordinal']) for row in requests] == list(range(128)),
        'Public ordinal order differs',
    )
    for arm in ARMS:
        for name in (*COUNTS, *TIMES):
            total = sum(float(row[arm + '_' + name]) for row in requests)
            require(
                math.isclose(
                    total, summary['totals'][arm][name], rel_tol=1e-12, abs_tol=1e-10
                ),
                'Public request total differs',
            )
        for repeat in range(3):
            rows = [
                r
                for r in repeats
                if r['phase'] == 'scored'
                and r['arm'] == arm
                and int(r['repeat']) == repeat
            ]
            require(
                len(rows) == 128 and all(r['valid'] == 'True' for r in rows),
                'Public repeated panel incomplete',
            )
            for name in TIMES:
                require(
                    math.isclose(
                        sum(float(r[name]) for r in rows),
                        summary['totals'][arm]['repeat_totals'][name][repeat],
                        rel_tol=1e-12,
                        abs_tol=1e-10,
                    ),
                    'Public repeat total differs',
                )
    contrast = summary['contrasts']['FRONTIER_P_vs_CONTROL']
    savings = [int(r['verification_calls_saved']) for r in requests]
    require(
        sum(savings) == contrast['net_calls_saved']
        and sum(x > 0 for x in savings) == contrast['wins']
        and sum(x < 0 for x in savings) == contrast['regressions'],
        'Public paired call changes differ',
    )
    return dict(
        status='passed',
        requests=128,
        repeated_rows=792,
        warmup_rows=sum(r['phase'] == 'warmup' for r in repeats),
        exact_count_reductions=True,
        float_reductions='relative1e-12/absolute1e-10 roundoff only',
    )


if __name__ == '__main__':
    require(
        read(OUT / 'PRIVATE_AUDIT.json')['status'] == 'passed',
        'Primary offline audit must pass first',
    )
    write('ALLOCATION_SPARSE_DIAGNOSTICS.json', sparse_allocations())
    write('TIMING_BREAKDOWN.json', timing_breakdown())
    write('TABLE_REDUCTION_AUDIT.json', table_reductions())
    write(
        'PRIVATE_SUPPLEMENT_AUDIT.json',
        dict(status='passed', input_identities=IDENTITIES, extra_model_work=0),
    )
    print(
        json.dumps(
            dict(status='passed', extra_model_work=0, input_files=len(IDENTITIES))
        )
    )
