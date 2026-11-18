"""Price the complete retained trajectory before any model load.

Logical bytes provide an upper bound, with serialization allowance. An atomic
rename of a new immutable checkpoint has one live payload, not two. Historical
assets are reused in place and cannot be retired to make this plan fit.
"""
from __future__ import annotations

import math

from .runtime import ALLOCATION_BYTES, RESERVE_BYTES


def storage_plan(datasets, *, retain_final_exports=True):
    """Price every phase from sealed prompt lengths and actual model geometry."""
    if set(datasets) != {'gsm8k', 'mathqa'}:
        raise ValueError('Both complete dataset dependencies are required')
    geometry, pools, states, exports, temporary = {}, {}, {}, {}, {}
    for dataset, row in datasets.items():
        lengths = row['next_prompt_lengths']
        if len(lengths) != 128 or any(type(n) is not int or not 0 < n <= 2048 for n in lengths):
            raise ValueError('Each next-event manifest must contain 128 eligible prompts')
        row_bytes = row['hidden_size'] * (row['captured_layers'] + 1) * 2 + 12
        geometry[dataset] = row_bytes
        pools[dataset] = sum(n + 2048 for n in lengths) * row_bytes
        states[dataset] = row['recovery_bytes'] + 16 * 2**20
        exports[dataset] = row['export_bytes'] + 2**20
        # Prepared packet plus capture/terminal overlap for only one request.
        temporary[dataset] = 2 * (max(lengths) + 2049) * row_bytes
    common = dict(source_and_workspace_bytes=1_000_000_000,
                  compact_evidence_bytes=1_000_000_000,
                  compiler_bytes=4 * 2**30)
    fixed = sum(common.values())
    phases = []
    retained_states = retained_features = retained_exports = 0

    def phase(name, state_bytes, feature_bytes, export_bytes, scratch_bytes=0):
        subtotal = fixed + state_bytes + feature_bytes + export_bytes + scratch_bytes
        margin = math.ceil(subtotal * .05)
        phases.append(dict(name=name, recovery_bytes=state_bytes,
            feature_bytes=feature_bytes, export_bytes=export_bytes,
            transient_bytes=scratch_bytes, serialization_allowance_bytes=margin,
            peak_new_bytes=subtotal + margin))

    # One trainer's qualification payload at a time; it is retired only after
    # both serving-card checks and compact restoration evidence are sealed.
    for dataset in ('gsm8k', 'mathqa'):
        phase(dataset+'_qualification', 2*states[dataset], 0, 2*exports[dataset])
    for dataset in ('gsm8k', 'mathqa'):
        # All four event states survive. At most three exports overlap during
        # this dataset: both first events and one completed second event.
        phase(dataset+'_production', retained_states + 4*states[dataset],
              retained_features + 2*pools[dataset],
              retained_exports + 3*exports[dataset], temporary[dataset])
        retained_states += 4*states[dataset]
        retained_features += 2*pools[dataset]
        if retain_final_exports or dataset == 'mathqa':
            retained_exports += 2*exports[dataset]
        phase(dataset+'_complete', retained_states, retained_features, retained_exports)
    for dataset in ('gsm8k', 'mathqa'):
        live_exports = 2*sum(exports.values())  # reconstruct GSM endpoints before quiet final timing
        phase(dataset+'_timing', retained_states, retained_features, live_exports)
    peak = max(row['peak_new_bytes'] for row in phases)
    return dict(status='passed' if peak <= ALLOCATION_BYTES else 'blocked',
        peak_new_bytes=peak, allocation_cap_bytes=ALLOCATION_BYTES,
        reserve_bytes=RESERVE_BYTES, required_initial_free_bytes=peak+RESERVE_BYTES,
        geometry_row_bytes=geometry, bank_bound_bytes=pools, recovery_bound_bytes=states,
        export_bound_bytes=exports, temporary_feature_bytes=max(temporary.values()),
        atomic_publication_bytes=0, phases=phases, retain_final_exports=retain_final_exports,
        retained_complete_recoveries=8, retained_unique_feature_banks=4,
        original_dependencies_copied=False, historical_retirement=False,
        qualified_derived_export_retirement_required=True,
        gsm_final_exports_temporarily_retired=not retain_final_exports,
        all_four_final_exports_retained_after_timing=True, **common)


def production_groups():
    """Dependency-safe overlap; both arms train on physical A, collectors on B."""
    groups = []
    for dataset in ('gsm8k', 'mathqa'):
        groups.extend([
            [(dataset, 'N_first', 'A'), (dataset, 'references', 'B')],
            [(dataset, 'C_first', 'A'), (dataset, 'N_collect', 'B')],
            [(dataset, 'N_second', 'A'), (dataset, 'C_collect', 'B')],
            [(dataset, 'C_second', 'A')],
        ])
    groups.extend([[(dataset, 'final', 'A')] for dataset in ('gsm8k', 'mathqa')])
    return groups


def price_groups(groups, estimates):
    costs = [[estimates[dataset][task] for dataset, task, _ in group] for group in groups]
    return dict(elapsed_seconds=sum(max(group) for group in costs),
                occupied_card_seconds=sum(sum(group) for group in costs))
