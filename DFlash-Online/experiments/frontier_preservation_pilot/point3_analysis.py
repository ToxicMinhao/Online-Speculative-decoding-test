"""Prespecified Point3 comparisons and development freeze rule."""

from collections import defaultdict
import hashlib
import json

import numpy as np

from experiments.opencode_native_comparison.analysis import _interval
from experiments.opencode_native_comparison.contract import digest
from experiments.repair_conditional_kl_pilot.analysis import _reduce_requests, TIMES
from experiments.repair_conditional_kl_pilot.evaluation import COUNT_FIELDS
from .point3_evaluation import METHODS, REQUESTS, REPEATS

BOOTSTRAP_SEED, BOOTSTRAP_DRAWS = 20261125, 10000
CONTRASTS = (('F_C_MAX', 'F_C_SUM'), ('F_C_SUM', 'C'), ('F_C_MAX', 'C'))


def reduce_requests(rows, *, expected_requests=REQUESTS):
    return _reduce_requests(rows, expected_requests=expected_requests, methods=METHODS)


def complete_generation_accounting(rows):
    """A complete report includes every included warmup as well as scored work."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['activation']].append(row)
    if len(rows) != 1188 or set(grouped) != set(range(18)):
        raise ValueError('Point3 requires all 1,188 returned measurements and eighteen activations')
    warmup_ids = {row['request_id'] for row in rows if row['phase'] == 'warmup'}
    if len(warmup_ids) != 2:
        raise ValueError('Point3 requires the same two warmup identities throughout')
    group_ids = {}
    for activation, records in grouped.items():
        repeat, group = divmod(activation // 3, 2)
        shift, position = (repeat + group) % 3, activation % 3
        method = (METHODS[shift:] + METHODS[:shift])[position]
        if any((row['repeat'], row['group'], row['method']) != (repeat, group, method) for row in records):
            raise ValueError('Point3 measurement activation identity differs from its schedule')
        warmup = [row for row in records if row['phase'] == 'warmup']
        scored = [row for row in records if row['phase'] == 'scored']
        if (len(warmup) != 2 or len(scored) != 64
                or len({row['request_id'] for row in scored}) != 64):
            raise ValueError('Point3 activation requires its two warmups and sixty-four scored requests')
        if {row['request_id'] for row in warmup} != warmup_ids:
            raise ValueError('Every Point3 activation must include both warmup identities exactly once')
        selected = frozenset(row['request_id'] for row in scored)
        if group in group_ids and selected != group_ids[group]:
            raise ValueError('Point3 scored group identities changed across methods or repeats')
        group_ids[group] = selected
    panel_ids = group_ids[0] | group_ids[1]
    if group_ids[0] & group_ids[1] or len(panel_ids) != REQUESTS or panel_ids & warmup_ids:
        raise ValueError('Point3 requires disjoint scored groups and separate warmups')
    return dict(returned=1188, valid=1188, scored=1152, warmups=36, activations=18)


def summarize(rows, *, expected_requests=REQUESTS):
    paired = reduce_requests(rows, expected_requests=expected_requests)
    accounting = complete_generation_accounting(rows) if expected_requests == REQUESTS else None
    draws = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED)).integers(
        0, len(paired), size=(BOOTSTRAP_DRAWS, len(paired)),
    )
    values = {
        method: {key: np.asarray([row[method][key] for row in paired], dtype=np.float64)
                 for key in (*TIMES, *COUNT_FIELDS)}
        for method in METHODS
    }
    totals = {}
    for method, metrics in values.items():
        sums = {key: float(value.sum()) for key, value in metrics.items()}
        calls = sums['verification_calls']
        for name, key in (('native_accepted_per_call', 'native_accepted_draft_tokens'),
                          ('stop_clipped_accepted_per_call', 'accepted_draft_tokens'),
                          ('committed_per_call', 'committed_progress')):
            sums[name] = sums[key] / calls if calls else None
        sums['tokens_per_s'] = sums['output_tokens'] / sums['request_latency_s']
        sums['repeat_totals'] = {
            key: [sum(row[method]['repeat_times'][key][r] for row in paired)
                  for r in range(REPEATS)] for key in TIMES
        }
        sums['termination_counts'] = {
            kind: sum(row[method]['finish_reason'] == kind for row in paired)
            for kind in ('eos', 'length')
        }
        sums['output_length'] = dict(
            min=int(metrics['output_tokens'].min()), max=int(metrics['output_tokens'].max()),
            mean=float(metrics['output_tokens'].mean()), median=float(np.median(metrics['output_tokens'])),
        )
        totals[method] = sums
    contrasts = {}
    for candidate, reference in CONTRASTS:
        comparison = dict(candidate=candidate, reference=reference)
        for name, key in [(key + '_reduction_pct', key) for key in TIMES] + [
                ('verification_call_reduction_pct', 'verification_calls')]:
            current, baseline = values[candidate][key], values[reference][key]
            point = 100 * (1 - current.sum() / baseline.sum()) if baseline.sum() else float('nan')
            with np.errstate(divide='ignore', invalid='ignore'):
                samples = 100 * (1 - current[draws].sum(1) / baseline[draws].sum(1))
            comparison[name] = _interval(point, samples)
        primary = comparison['request_latency_s_reduction_pct']
        if primary['status'] != 'undefined':
            primary['status'] = ('prespecified_primary_component' if reference == 'F_C_SUM'
                                 else 'required_whole_method')
        difference = values[reference]['verification_calls'] - values[candidate]['verification_calls']
        comparison.update(
            wins=int((difference > 0).sum()), regressions=int((difference < 0).sum()),
            ties=int((difference == 0).sum()), gross_calls_saved=int(np.maximum(difference, 0).sum()),
            gross_calls_added=int(np.maximum(-difference, 0).sum()), net_calls_saved=int(difference.sum()),
        )
        contrasts[candidate + '_vs_' + reference] = comparison
    ids = [row['request_id'] for row in paired]
    return dict(
        requests=len(paired), totals=totals, contrasts=contrasts, paired=paired,
        generation_accounting=accounting,
        bootstrap=dict(
            draws=BOOTSTRAP_DRAWS, generator='PCG64', seed=BOOTSTRAP_SEED,
            input_order=ids, input_order_digest=digest(ids),
            draw_sha256=hashlib.sha256(draws.astype('<i8', copy=False).tobytes()).hexdigest(),
            shared_indices_across_all_arms=True,
            percentiles='linear 2.5/97.5; paired whole requests; timing repeats averaged first',
            secondary_intervals='descriptive, unadjusted',
            scope='fixed endpoints and reused development panel; not fresh confirmation or seed replication',
        ),
    )


def exported_parameters_identity(path):
    """Hash named tensor values, independent of safetensors metadata/header order.

    One tensor is mapped at a time on CPU; this check belongs outside timing.
    File-level identities remain separate provenance/authentication evidence.
    """
    import torch
    from safetensors import safe_open
    digest = hashlib.sha256()
    count, elements = 0, 0
    with safe_open(str(path), framework='pt', device='cpu') as handle:
        for name in sorted(handle.keys()):
            value = handle.get_tensor(name)
            metadata = dict(name=name, shape=list(value.shape), dtype=str(value.dtype))
            digest.update(json.dumps(metadata, sort_keys=True).encode())
            flat = value.detach().reshape(-1)
            for start in range(0, flat.numel(), 1_048_576):
                chunk = flat[start:start + 1_048_576].contiguous().view(torch.uint8)
                digest.update(memoryview(chunk.numpy()))
            count += 1
            elements += value.numel()
    if count == 0:
        raise ValueError('Serving export contains no parameters')
    return dict(sha256=digest.hexdigest(), tensors=count, elements=elements,
                representation='sorted names/shapes/dtypes/native tensor bytes; header metadata excluded')


def freeze_decision(report, serving_identities, *, authenticated_complete):
    """Apply §9 to unrounded values; model identity overrides noisy timing ties.

    Each serving identity contains ``parameters`` and ``inference`` identities.
    Optimizer-state identities may be recorded separately; they are deliberately
    not evidence of a different serving model.
    """
    if not authenticated_complete:
        return dict(selected='C', status='incomplete', eligible=[],
                    reason='Keep the working configuration until all required evidence is complete')
    if set(serving_identities) != set(METHODS):
        raise ValueError('Point3 needs all three authenticated serving identities')
    identities = {}
    for method in METHODS:
        identity = serving_identities[method]
        if not identity.get('parameters') or not identity.get('inference'):
            raise ValueError('Missing exported-parameter or inference identity')
        identities[method] = (identity['parameters'], identity['inference'])
    totals = report['totals']
    candidates = list(METHODS[1:])
    equivalent_candidates = identities['F_C_SUM'] == identities['F_C_MAX']
    if equivalent_candidates:
        candidates = ['F_C_SUM']
    eligible = [
        method for method in candidates
        if identities[method] != identities['C']
        and totals[method]['request_latency_s'] < totals['C']['request_latency_s']
        and totals[method]['verification_calls'] <= totals['C']['verification_calls']
    ]
    selected = min(eligible, key=lambda method: (
        totals[method]['request_latency_s'], totals[method]['verification_calls'], METHODS.index(method),
    )) if eligible else 'C'
    interval = (None if selected == 'C' else
                report['contrasts'][selected + '_vs_C']['request_latency_s_reduction_pct']['ci95'])
    provisional = selected != 'C' and (interval is None or interval[0] <= 0)
    return dict(
        selected=selected, status='provisional_development_selection' if provisional else 'development_selection',
        eligible=eligible, candidates_have_identical_serving_models=equivalent_candidates,
        model_identity_ignores_optimizer_state=True,
        selected_latency_ci95=interval,
        no_demonstrated_speedup=selected == 'C' or provisional,
        reason=('No distinct candidate meets latency-better and calls-not-worse criteria'
                if selected == 'C' else 'Prespecified unrounded latency/call selection on a reused development panel'),
        further_mechanism_search=False,
    )
