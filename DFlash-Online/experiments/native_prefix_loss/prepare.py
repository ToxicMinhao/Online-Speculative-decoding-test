"""Seal a new prefix-role investigation from authenticated PR46 input bindings.

Only populations and scientific source are inherited. Prior intervention states,
outcomes and retry budgets are not used as new scientific evidence.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path

from experiments.threatened_repair_replay.prepare import (
    authenticate_source, checked_identity, identity, read, write,
)

WINDOWS = tuple(range(8, 129, 8))
STATES = ('P', 'N', 'E', 'C', 'H')
CONSTANTS = dict(
    windows=list(WINDOWS), states=list(STATES), source_step=768, destination_step=896,
    native_steps=128, qualification_adam=6, main_adam=176, maximum_adam=182,
    ordinary_response_forwards=320, ordinary_response_backwards=320,
    qualification_response_forwards_max=12, qualification_response_backwards_max=12,
    source_scoring_response_forwards_max=160, independent_conceptual_blocks=20480,
    qualification_blocks_max=32, new_complete_response_generations=0,
    head_chunk_blocks=8, block_size=16, bootstrap_draws=10000,
    bootstrap_seed=1707, bootstrap_rng='PCG64', bootstrap_quantile='linear',
    card_seconds_limit=10800, primary='E_minus_N',
    primary_unit='accepted draft proposals per fixed block',
    role_gate='torch.where(active_role, token_loss, token_loss.detach())',
    denominator='unchanged native K+ selected-weight denominator',
    native_backward='authoritative full backward; never sum or subtract role gradients',
    history='explicit zero tensors for native-present gradients; preserve None',
    source_reuse=False, historical_probe_outcome_reuse=False,
)


def attempt_accounting(prior=None):
    """Charge the approved retry for authenticated work from execution_001.

    Scientific work and the new process's 182-Adam limit stay unchanged. Only
    the approved cumulative Adam limit increases; the three-hour clock does not.
    """
    accounting = dict(prior_adam_steps=0, prior_gpu_seconds=0.0,
        current_attempt_adam_limit=CONSTANTS['maximum_adam'],
        cumulative_adam_limit=CONSTANTS['maximum_adam'],
        cumulative_gpu_seconds_limit=CONSTANTS['card_seconds_limit'],
        current_attempt_gpu_seconds_limit=float(CONSTANTS['card_seconds_limit']))
    if prior is None:
        return accounting
    if prior['approved_cumulative_adam_limit'] != 187:
        raise ValueError('This retry requires the approved cumulative 187-Adam limit')
    for stem in ('status', 'work_costs', 'training_work'):
        checked_identity(prior[stem+'_path'], prior.get(stem+'_identity'))
    status, costs = read(prior['status_path']), read(prior['work_costs_path'])
    if (status['status'] != 'failed'
            or status['phase'] != 'training_qualification_and_native_trajectory'
            or status['execution_commit'] != 'fdccdf5482f2125f212476547522068f953c564a'):
        raise ValueError('Prior attempt must be the retained execution_001 qualification failure')
    from experiments.threatened_repair_replay.work import read_ledger
    work = read_ledger(prior['training_work_path'])
    records = [json.loads(line) for line in Path(prior['training_work_path']).read_text().splitlines()]
    starts = {row['serial']:row for row in records if row['phase'] == 'started'}
    for row in records:
        if row['phase'] == 'ledger_closed':
            continue
        if (row['phase'] not in ('started', 'completed', 'failed')
                or row.get('category') not in ('startup', 'qualification')):
            raise ValueError('Prior attempt contains scientific or malformed work')
        start = starts[row['serial']]
        if (row['kind'], row['category']) != (start['kind'], start['category']):
            raise ValueError('Prior work completion differs from its recorded start')
        if row['kind'] == 'optimizer_step' and row['category'] != 'qualification':
            raise ValueError('Prior Adam applications must be disposable qualification steps')
    if (not records or records[-1]['phase'] != 'ledger_closed'
            or sum(row['phase'] == 'ledger_closed' for row in records) != 1
            or work['unknown'] or work['failed']
            or work['attempted'].get('optimizer_step', 0) != 5
            or work['completed'].get('optimizer_step', 0) != 5):
        raise ValueError('Prior attempt must contain exactly five completed, known qualification Adam steps')
    seconds = costs['native_budget']['gpu_seconds']
    if (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
            or not math.isfinite(seconds) or not 0 < seconds < CONSTANTS['card_seconds_limit']
            or costs['native_budget']['gpu_seconds_limit'] != CONSTANTS['card_seconds_limit']):
        raise ValueError('Prior occupied GPU time must be finite and below the original cap')
    if (costs['training'] != work or costs['native_budget'] != status['budget']
            or costs['prior_attempt'] is not None
            or costs['budget_accounting'] != accounting or status['budget_accounting'] != accounting
            or costs['cumulative'] != dict(completed_adam_steps=5, attempted_adam_steps=5,
                gpu_seconds=seconds, adam_limit=182, gpu_seconds_limit=10800)
            or status['cumulative_gpu_seconds'] != seconds):
        raise ValueError('Prior cost, status and physical-work receipts do not reconcile')
    accounting.update(prior_adam_steps=work['completed']['optimizer_step'], prior_gpu_seconds=seconds,
        cumulative_adam_limit=187,
        current_attempt_gpu_seconds_limit=CONSTANTS['card_seconds_limit']-seconds,
        prior_execution_commit=status['execution_commit'], prior_training_work=work,
        prior_native_budget=status['budget'])
    return accounting


def prepare(spec):
    output = Path(spec['output'])
    if output.exists():
        raise FileExistsError('Use a new sealed destination; preserve every old attempt')
    checked_identity(spec['previous_plan'], spec['previous_plan_identity'])
    previous = read(spec['previous_plan'])
    if previous['execution_commit'] != 'd51718a7fa993a1494e593d15a8539090227d8b8':
        raise ValueError('Require the authenticated completed PR46 input plan')
    source = authenticate_source(spec['source_manifest'], spec['source_manifest_identity'], spec['execution_commit'])
    prior = spec.get('prior_attempt')
    accounting = attempt_accounting(prior)
    if (output.parent.name == 'execution_002') != (prior is not None):
        raise ValueError('The approved retry requires execution_002 and authenticated prior-attempt evidence')
    if prior is not None and Path(prior['status_path']).parent.resolve() == output.parent.resolve():
        raise ValueError('Retry must preserve the failed attempt in a separate destination')
    inputs = deepcopy(previous['input_identities'])
    inputs.pop(previous['source_manifest'], None)
    inputs[str(spec['previous_plan'])] = spec['previous_plan_identity']
    inputs[spec['source_manifest']] = spec['source_manifest_identity']
    if prior is not None:
        for stem in ('status', 'work_costs', 'training_work'):
            inputs[str(Path(prior[stem+'_path']))] = prior[stem+'_identity']
    for name, expected in previous['immutable_files'].items():
        path = Path(spec['previous_plan']).parent / name
        checked_identity(path, expected)
        inputs[str(path)] = expected
    for path, expected in inputs.items():
        checked_identity(path, expected)
    if len(previous['feature_paths']) != 128 or len(previous['probe_requests']) != 64:
        raise ValueError('Retained feedback or fixed development panel is incomplete')
    output.mkdir(parents=True)
    for name in ('PROBE_KEYS.json', 'PROBE_KEYS.csv'):
        original = Path(spec['previous_plan']).parent / name
        (output / name).write_bytes(original.read_bytes())
    plan = deepcopy(previous)
    for name in ('memory_tags', 'memory_layouts', 'population', 'prior_attempt', 'budget_accounting'):
        plan.pop(name, None)
    weight_bytes = (Path(plan['origin_export']) / 'model.safetensors').stat().st_size
    recovery_bytes = Path(plan['origin_state']).stat().st_size
    # 80 fresh exports, two boundary/failure witnesses, temporary serialization,
    # and a generous compact-evidence allowance. No historical deletion needed.
    additional = 82 * weight_bytes + 4 * recovery_bytes + 4 * 1024**3
    plan.update(schema='native-prefix-loss-v1', root=str(output.parent),
        code_path=source['source_path'], execution_commit=spec['execution_commit'], constants=CONSTANTS,
        source_manifest=spec['source_manifest'], source_verification=source,
        probe_keys=str(output/'PROBE_KEYS.json'), input_identities=inputs,
        immutable_files={p.name:identity(p) for p in output.iterdir()},
        previous_plan=spec['previous_plan'],
        prior_attempt=deepcopy(prior), budget_accounting=accounting,
        storage=dict(maximum_additional_bytes=additional, reserve_bytes=8*1024**3),
        storage_calculation=dict(weight_bytes=weight_bytes, recovery_bytes=recovery_bytes,
            weight_copies=82, recovery_copies=4, compact_allowance_bytes=4*1024**3),
        duration_prior_s=[3600,7200],
        remaining_probe_estimate_s=3600, source_forward_estimate_s=1.0,
        source_forward_estimate_basis='PR45 fitted scorer: 161.698 seconds/512 response forwards (~0.316); allow 1.0 second per full response',
        probe_estimate_basis='PR46 2243.254 seconds/16128 blocks including reconstruction; conservatively allow 3600 for 20480',
        qualified_source_contract='PR46 pins plus narrow gradient-role gate; no DKD implementation',
    )
    write(output/'INPUT_IDENTITIES.json', inputs)
    write(output/'PLAN.json', plan)
    return plan


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True)
    args = parser.parse_args()
    result = prepare(read(args.spec))
    print(result['schema'], result['execution_commit'], result['storage'])
