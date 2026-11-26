"""Point3's three real endpoints, balanced serving and retained F references."""

from collections import Counter

from experiments.opencode_native_comparison.contract import validate_requests
from experiments.repair_conditional_kl_pilot.evaluation import _execute_schedule

METHODS = ('C', 'F_C_SUM', 'F_C_MAX')
REQUESTS, REPEATS, GROUP_SIZE = 128, 3, 64
ACTIVATIONS, SCORED_GENERATIONS, WARMUP_GENERATIONS = 18, 1152, 36


def evaluation_schedule(requests):
    validate_requests(requests, count=REQUESTS, role='evaluation')
    schedule = []
    for repeat in range(REPEATS):
        for group in range(REQUESTS // GROUP_SIZE):
            shift = (repeat + group) % len(METHODS)
            for position, method in enumerate(METHODS[shift:] + METHODS[:shift]):
                schedule.append(dict(
                    activation=len(schedule), repeat=repeat, group=group,
                    position=position, method=method, warmups=2,
                    request_ids=[row['request_id'] for row in
                                 requests[group * GROUP_SIZE:(group + 1) * GROUP_SIZE]],
                ))
    return schedule


def schedule_summary(schedule):
    """Every request occupies every execution position once for each method."""
    if len(schedule) != ACTIVATIONS:
        raise ValueError('Point3 requires eighteen activations')
    coverage, positions, first = Counter(), Counter(), Counter()
    ordered_groups = {}
    for start in range(0, ACTIVATIONS, len(METHODS)):
        batch = schedule[start:start + len(METHODS)]
        repeat, group = divmod(start // len(METHODS), REQUESTS // GROUP_SIZE)
        shift = (repeat + group) % len(METHODS)
        expected_order = METHODS[shift:] + METHODS[:shift]
        ids = batch[0]['request_ids']
        if len(ids) != GROUP_SIZE or len(set(ids)) != GROUP_SIZE:
            raise ValueError('Point3 requires distinct complete sixty-four-request groups')
        if group in ordered_groups and ids != ordered_groups[group]:
            raise ValueError('Point3 request ordering changed across repeats')
        ordered_groups[group] = ids
        for position, entry in enumerate(batch):
            expected = dict(activation=start + position, repeat=repeat, group=group,
                            position=position, method=expected_order[position], warmups=2,
                            request_ids=ids)
            if entry != expected:
                raise ValueError('Point3 activation identity, order or panel changed')
            for request_id in ids:
                coverage[request_id, repeat, entry['method']] += 1
                positions[request_id, entry['method'], position] += 1
                if position == 0:
                    first[repeat, entry['method']] += 1
    if set(ordered_groups[0]) & set(ordered_groups[1]):
        raise ValueError('Point3 groups must be disjoint')
    if (len(coverage) != SCORED_GENERATIONS or any(n != 1 for n in coverage.values())
            or len(positions) != SCORED_GENERATIONS or any(n != 1 for n in positions.values())):
        raise ValueError('Point3 coverage or per-request execution-position balance changed')
    return dict(
        activations=ACTIVATIONS, scored_generations=SCORED_GENERATIONS,
        warmup_generations=WARMUP_GENERATIONS, total_generations=1188,
        groups=6, requests=REQUESTS, repeats=REPEATS, group_size=GROUP_SIZE,
        each_request_each_method_each_execution_position=1,
        first_position_requests_per_repeat={
            method: [first[repeat, method] for repeat in range(REPEATS)] for method in METHODS
        },
    )


def execute(plan, root, requests, warmups, canonicals, warmup_canonicals,
            checkpoints, *, activate, boundary, ledger):
    """Use the qualified native measurement path without two-arm impersonation."""
    schedule = evaluation_schedule(requests)
    return _execute_schedule(
        plan, root, requests, warmups, canonicals, warmup_canonicals, checkpoints,
        activate=activate, boundary=boundary, ledger=ledger, methods=METHODS,
        schedule=schedule, balance=schedule_summary(schedule),
        schema='point3_cumulative_max_evaluation_v1',
    )


def run_evaluation(plan):
    """Activate retained C and both completed Point3 endpoints on quiet GPU A."""
    import json
    import os
    from pathlib import Path

    from experiments.ok_coding.storage import identity
    from experiments.repair_conditional_kl_pilot.campaign import create_backend
    from experiments.repair_conditional_kl_pilot.production import quiet_evaluation
    from experiments.repair_conditional_kl_pilot.resources import Ledger, read
    from online_feedback.storage import write_json
    from .allocation_campaign import selected_inputs
    from .point3_analysis import summarize, freeze_decision, exported_parameters_identity
    from .point3_launch import boundary
    from .point3_prepare import SCHEMA

    if (plan.get('schema') != SCHEMA or plan.get('production_admitted') is not True
            or os.environ.get('CUDA_VISIBLE_DEVICES') != plan['devices']['A']):
        raise ValueError('Point3 serving requires admitted scope and its physical GPU A')
    root = Path(plan['root'])
    qualification = read(plan['qualification_result'])
    if (qualification.get('status') != 'completed'
            or qualification.get('native_reference', {}).get('status') != 'passed'
            or qualification.get('qualification', {}).get('status') != 'passed'):
        raise ValueError('Point3 qualification must pass before serving')
    control_path = Path(plan['retained_C_status'])
    if identity(control_path) != plan['retained_C_status_identity']:
        raise ValueError('Retained C status changed')
    receipts = {'C': read(control_path)}
    receipts.update({method: read(root / 'arms' / method / 'STATUS.json') for method in METHODS[1:]})
    checkpoints, export_identities = {}, {}
    for method, receipt in receipts.items():
        if (receipt.get('status') != 'completed' or receipt.get('global_step') != 256
                or receipt.get('presentations') != 512 or receipt.get('completed_event') != 2):
            raise ValueError('Point3 requires three complete event-two endpoint receipts')
        checkpoint = receipt['checkpoint']
        export = Path(checkpoint) / 'model.safetensors'
        if identity(export) != receipt['checkpoint_identity']:
            raise ValueError('Point3 endpoint export changed: ' + method)
        if identity(receipt['recovery']) != receipt['recovery_identity']:
            raise ValueError('Point3 endpoint recovery changed: ' + method)
        checkpoints[method], export_identities[method] = checkpoint, receipt['checkpoint_identity']
    production = read(plan['production_plan'])
    inputs, _ = selected_inputs(production)
    backend = None
    ledger = Ledger(root / 'evaluation_work.json')

    def checked_boundary(slot):
        return dict(slot=slot, resources=boundary(plan), quiet=quiet_evaluation(plan, backend))

    def activate(method, checkpoint, slot):
        nonlocal backend
        loading_s = 0.0
        if backend is None:
            backend = create_backend(plan, root / 'evaluation_native', card='A')
            loading_s = backend.loading_s
        return backend, dict(publication=backend.publish(checkpoint), loading_s=loading_s)

    try:
        result = execute(
            plan, root / 'evaluation', inputs['evaluation_requests'][:128], inputs['warmup_requests'],
            read(plan['canonicals']), read(plan['warmup_canonicals']), checkpoints,
            activate=activate, boundary=checked_boundary, ledger=ledger,
        )
        actual = read(root / 'evaluation_work.json')
        expected = dict(scored_generation=SCORED_GENERATIONS, warmup_generation=WARMUP_GENERATIONS)
        if actual['pending'] or actual['attempted'] != expected or actual['completed'] != expected:
            raise ValueError('Point3 fixed serving work does not reconcile')
    finally:
        if backend is not None:
            try:
                if not backend.failed:
                    write_json(root / 'evaluation_physical_work.json', backend._call('work_receipt'))
            finally:
                backend.close()
    for method, checkpoint in checkpoints.items():
        if identity(Path(checkpoint) / 'model.safetensors') != export_identities[method]:
            raise ValueError('Point3 serving changed an endpoint export: ' + method)
    write_json(root / 'evaluation/EXPORTS_UNCHANGED.json', export_identities)
    rows = [json.loads(line) for line in (root / 'evaluation/MEASUREMENTS.jsonl').read_text().splitlines()]
    report = summarize(rows)
    activations = [json.loads(line) for line in
                   (root / 'evaluation/ACTIVATIONS.jsonl').read_text().splitlines()]
    report['serving_costs'] = dict(
        loading_s=sum(row['loading_s'] for row in activations),
        activation_s=sum((row.get('publication') or {}).get('activation_s', 0.) for row in activations),
        warmup_request_latency_s=sum(row['request_latency_s'] for row in rows if row['phase'] == 'warmup'),
        scored_request_latency_s=sum(row['request_latency_s'] for row in rows if row['phase'] == 'scored'),
        primary_latency_uses='summed per-request means over three scored repeats only',
        loading_activation_and_warmups_excluded_from_primary=True,
    )
    # Common serving metadata is explicit; parameter identity ignores optimizer state.
    parameter_identities = {
        method: exported_parameters_identity(Path(checkpoint) / 'model.safetensors')
        for method, checkpoint in checkpoints.items()
    }
    serving_identities = {method: dict(parameters=parameter_identities[method]['sha256'],
        inference=plan['serving']) for method in METHODS}
    report['parameter_identities'] = parameter_identities
    report['freeze'] = freeze_decision(report, serving_identities, authenticated_complete=True)
    report['serving_identities'] = serving_identities
    write_json(root / 'evaluation/SUMMARY.json', report)
    return result


if __name__ == '__main__':
    import argparse
    import json
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    plan['qualification_result'] = str(Path(plan['root']) / 'qualification/native/RESULT.json')
    run_evaluation(plan)
