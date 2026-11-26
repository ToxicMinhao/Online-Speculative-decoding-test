"""Fresh BASE/TAIL serving, using the qualified native two-arm execution path."""

import argparse
import json
import os
from pathlib import Path

from experiments.ok_coding.storage import identity
from experiments.repair_conditional_kl_pilot import evaluation
from experiments.repair_conditional_kl_pilot.resources import read, Ledger
from online_feedback.storage import write_json
from .analysis import METHODS, summarize, exported_parameters_identity


def schedule(requests):
    result = evaluation.evaluation_schedule(requests, methods=METHODS)
    evaluation.schedule_summary(result, methods=METHODS)
    return result


def execute(plan, root, requests, warmups, canonicals, warmup_canonicals,
            checkpoints, *, activate, boundary, ledger):
    slots = schedule(requests)
    return evaluation._execute_schedule(
        plan, root, requests, warmups, canonicals, warmup_canonicals, checkpoints,
        activate=activate, boundary=boundary, ledger=ledger, methods=METHODS,
        schedule=slots, balance=evaluation.schedule_summary(slots, methods=METHODS),
        schema='continuation_repair_serving_v1',
    )


def require_qualification(plan):
    qualification = read(plan['qualification_result'])
    if (qualification.get('status') != 'passed' or qualification.get('binding') != plan['binding']
            or qualification.get('base_disabled_exact') is not True
            or qualification.get('retained_BASE_reusable') is not True):
        raise ValueError('Require exact disabled neutrality and authenticated BASE reuse for this pilot')
    return qualification


def endpoint_receipts(plan):
    """Use the named retained BASE and new TAIL; validate each final export once."""
    root = Path(plan['root'])
    base_status = Path(plan['base_status'])
    if identity(base_status) != plan['base_status_identity']:
        raise ValueError('Authenticated BASE status changed')
    receipts = dict(BASE=read(base_status), TAIL=read(root / 'arms/TAIL/STATUS.json'))
    for method, receipt in receipts.items():
        if (receipt.get('status') != 'completed' or receipt.get('global_step') != 256
                or receipt.get('presentations') != 512 or receipt.get('completed_event') != 2):
            raise ValueError('Require complete event-two endpoint: ' + method)
        export = Path(receipt['checkpoint']) / 'model.safetensors'
        if identity(export) != receipt['checkpoint_identity']:
            raise ValueError('Endpoint export changed: ' + method)
    return receipts


def run_evaluation(plan):
    from experiments.frontier_preservation_pilot.allocation_campaign import selected_inputs
    from experiments.repair_conditional_kl_pilot.campaign import create_backend
    from experiments.repair_conditional_kl_pilot.production import quiet_evaluation
    from .launch import boundary

    if not plan.get('production_admitted') or os.environ.get('CUDA_VISIBLE_DEVICES') != plan['devices']['A']:
        raise ValueError('Serving requires admitted scope and its designated physical card')
    root = Path(plan['root'])
    require_qualification(plan)
    receipts = endpoint_receipts(plan)
    checkpoints = {method: receipt['checkpoint'] for method, receipt in receipts.items()}
    parameter_identities = {method: exported_parameters_identity(Path(path) / 'model.safetensors')
                            for method, path in checkpoints.items()}
    write_json(root / 'SERVING_PARAMETERS.json', parameter_identities)
    inputs, _ = selected_inputs(plan)
    ledger = Ledger(root / 'evaluation_work.json')
    backend = None

    def checked_boundary(slot):
        return dict(slot=slot, resources=boundary(plan), quiet=quiet_evaluation(plan, backend))

    def activate(method, checkpoint, slot):
        nonlocal backend
        loading_s = 0.
        if backend is None:
            backend = create_backend(plan, root / 'evaluation_native', card='A')
            loading_s = backend.loading_s
        return backend, dict(publication=backend.publish(checkpoint), loading_s=loading_s)

    try:
        result = execute(plan, root / 'evaluation', inputs['evaluation_requests'][:128],
            inputs['warmup_requests'], read(plan['canonicals']), read(plan['warmup_canonicals']),
            checkpoints, activate=activate, boundary=checked_boundary, ledger=ledger)
        work = read(root / 'evaluation_work.json')
        expected = dict(scored_generation=768, warmup_generation=24)
        if work['pending'] or work['attempted'] != expected or work['completed'] != expected:
            raise ValueError('The792 fixed serving attempts do not reconcile')
    finally:
        if backend is not None:
            try:
                if not backend.failed:
                    write_json(root / 'evaluation_physical_work.json', backend._call('work_receipt'))
            finally:
                backend.close()
    unchanged = {}
    for method, receipt in receipts.items():
        actual = identity(Path(receipt['checkpoint']) / 'model.safetensors')
        if actual != receipt['checkpoint_identity']:
            raise ValueError('Serving changed an endpoint export: ' + method)
        unchanged[method] = actual
    write_json(root / 'evaluation/EXPORTS_UNCHANGED.json', unchanged)
    rows = [json.loads(line) for line in (root / 'evaluation/MEASUREMENTS.jsonl').read_text().splitlines()]
    report = summarize(rows)
    activations = [json.loads(line) for line in (root / 'evaluation/ACTIVATIONS.jsonl').read_text().splitlines()]
    report['parameter_identities'] = parameter_identities
    report['distinct_serving_weights'] = parameter_identities['BASE']['sha256'] != parameter_identities['TAIL']['sha256']
    report['serving_costs'] = dict(
        loading_s=sum(row['loading_s'] for row in activations),
        activation_s=sum((row.get('publication') or {}).get('activation_s', 0.) for row in activations),
        warmup_request_latency_s=sum(row['request_latency_s'] for row in rows if row['phase'] == 'warmup'),
        scored_request_latency_s=sum(row['request_latency_s'] for row in rows if row['phase'] == 'scored'),
        primary_uses='Sum of per-request means across three scored repeats; excludes loading/switching/warmups')
    write_json(root / 'evaluation/SUMMARY.json', report)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    run_evaluation(read(parser.parse_args().plan))
