"""Strict serving-only confirmation around the unchanged PR37 raw evaluator.

The shared engine's ``production=False`` disables its original fixed-128 panel
size. This adapter explicitly restores every other production admission/audit
check, adds the sealed confirmation size, and binds all blocks to the pinned
physical GPU. It never patches the pilot's module globals or loss/training paths.
"""
import json
from pathlib import Path

from online_feedback.storage import write_json
from experiments.survival_head import evaluation as pilot

METHODS = pilot.METHODS
EVENT = pilot.EVENT
BLOCK_SIZE = pilot.BLOCK_SIZE
ENDPOINT_REPEATS = pilot.ENDPOINT_REPEATS
SESSION = pilot.SESSION
MIN_PANEL_SIZE, MAX_PANEL_SIZE = 136, 768
SCHEMA = 'survival_confirmation_evaluation_v1'
schedule_balance = pilot.schedule_balance
first_round = pilot.first_round


def validate_panel_size(sealed_n, request_count):
    if (type(sealed_n) is not int or not MIN_PANEL_SIZE <= sealed_n <= MAX_PANEL_SIZE
            or sealed_n % BLOCK_SIZE or request_count != sealed_n):
        raise ValueError('Require the sealed N matching inputs, a multiple of eight in 136..768')
    return sealed_n


def evaluation_schedule(requests, dataset='gsm8k', event=EVENT, *, sealed_n):
    """Rotate complete eight-request blocks; reverse each order in sweep two."""
    validate_panel_size(sealed_n, len(requests))
    return pilot.evaluation_schedule(requests, dataset, event)


def validate_evaluation_inputs(requests, checkpoints, canonicals, *, sealed_n,
                               schedule, dataset='gsm8k', event=EVENT):
    validate_panel_size(sealed_n, len(requests))
    if schedule is None:
        raise ValueError('Confirmation requires its presealed timing schedule')
    planned = pilot.validate_evaluation_inputs(requests, checkpoints, canonicals,
        dataset=dataset, event=event, schedule=schedule, production=False)
    if len(planned) != sealed_n * len(METHODS) * ENDPOINT_REPEATS:
        raise ValueError('Confirmation schedule does not cover all six executions per request')
    return planned


def _physical_gpu(value):
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError('Require the nonempty authenticated physical GPU identity from the sealed plan')
    return value


def confirmation_protocol(sealed_n, physical_gpu):
    validate_panel_size(sealed_n, sealed_n)
    return dict(schema=SCHEMA, sealed_n=sealed_n, physical_gpu=_physical_gpu(physical_gpu),
        dataset='gsm8k', event=EVENT, methods=list(METHODS), repeats=ENDPOINT_REPEATS,
        block_size=BLOCK_SIZE, planned_generations=sealed_n*len(METHODS)*ENDPOINT_REPEATS,
        optimizer_steps=0, feedback_collections=0, teacher_preparations=0,
        extra_warmup_generations=0, canonical_generation_scope='separate_before_timing',
        first8_correctness='included_in_main_timed_budget',
        shared_engine='experiments.survival_head.evaluation',
        production_size_override='strict_sealed_confirmation_N_only')


def boundary_errors(check, physical_gpu):
    """Retain native quiet-host/storage checks and reject changed physical cards."""
    _physical_gpu(physical_gpu)
    try:
        errors = list(pilot.boundary_errors(check))
        if check.get('physical_gpu') != physical_gpu:
            errors.append('physical_gpu_changed_or_unbound')
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        errors = ['missing_or_malformed_boundary: ' + repr(error)]
    return errors


def _batches(planned):
    result = []
    for slot in planned:
        batch = [slot['block_ordinal'], slot['repeat'], slot['position']]
        if not result or batch != result[-1]:
            result.append(batch)
    return result


def _confirmation_work(progress, rows, sealed_n):
    returned = progress.get('returned_generations', 0)
    return dict(schema=SCHEMA, status=progress.get('status', 'unavailable'),
        planned_generations=6*sealed_n,
        attempted_generations=progress.get('attempted_generations', 0),
        returned_generations=returned, persisted_measurements=len(rows),
        returned_without_measurement=max(0, returned-len(rows)),
        unknown_generation_attempts=progress.get('unknown_generation_attempts', 0),
        unattempted_generations=progress.get('unattempted_generations', 6*sealed_n),
        extra_generations=progress.get('extra_generations', 0),
        automatic_retry=False)


def _read_rows(folder):
    path = Path(folder)/'MEASUREMENTS.jsonl'
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def evaluate_stage(backend, requests, checkpoints, root, *, canonicals, sealed_n,
                   schedule, boundary_check, physical_gpu, dataset='gsm8k', event=EVENT):
    """Execute exactly the presealed 6N schedule with production boundary gates.

    Canonical preparation and frozen-export identity pinning occur in the caller's
    preflight. The native evaluator rechecks exact output/termination, weight
    publication, repeat counts and every saved native record.
    """
    planned = validate_evaluation_inputs(requests, checkpoints, canonicals,
        sealed_n=sealed_n, schedule=schedule, dataset=dataset, event=event)
    contract = confirmation_protocol(sealed_n, physical_gpu)
    if not callable(boundary_check):
        raise ValueError('Confirmation requires a callable quiet-host/storage boundary check')
    root = Path(root)
    if root.exists():
        raise FileExistsError('Preserve existing confirmation evidence; no automatic retry: '+str(root))
    batches = _batches(planned)
    checked = 0

    def authenticated_boundary():
        nonlocal checked
        if not (root/'CONFIRMATION_PROTOCOL.json').exists():
            write_json(root/'CONFIRMATION_PROTOCOL.json', contract)
        expected = batches[checked] if checked < len(batches) else None
        observation = None
        try:
            if expected is None:
                raise RuntimeError('Unexpected extra confirmation timing boundary')
            observation = boundary_check()
            errors = boundary_errors(observation, physical_gpu)
            if errors:
                raise RuntimeError('Failed confirmation timing boundary: '+repr(errors))
        except BaseException as error:
            # The shared engine only saves a boundary after this callback returns.
            # Preserve the rejected receipt before propagating into its failure path.
            write_json(root/'CONFIRMATION_BOUNDARY_FAILURE.json', dict(status='failed',
                boundary_ordinal=checked, expected_batch=expected,
                expected_physical_gpu=physical_gpu, observed=observation, error=repr(error),
                generation_attempted_at_this_boundary=False, automatic_retry=False))
            raise
        checked += 1
        return observation

    try:
        return pilot.evaluate_stage(backend, requests, checkpoints, root, canonicals=canonicals,
            dataset=dataset, event=event, schedule=planned,
            boundary_check=authenticated_boundary, production=False)
    finally:
        # Do not rewrite the inherited STATUS or reinterpret an unvalidated return
        # as a valid execution. The supplemental counter exposes that failure gap.
        status = root/'STATUS.json'
        if status.exists():
            progress = json.loads(status.read_text())
            write_json(root/'CONFIRMATION_STATUS.json',
                _confirmation_work(progress, _read_rows(root), sealed_n))


def audit_stage(folder, requests, checkpoints, *, canonicals, sealed_n, schedule,
                physical_gpu, dataset='gsm8k', event=EVENT, evidence_root=None):
    """Independently replay native audits and all strict confirmation admissions."""
    folder = Path(folder)
    errors, native = [], dict(rows=[], progress={}, model_work=0)
    contract = None
    try:
        planned = validate_evaluation_inputs(requests, checkpoints, canonicals,
            sealed_n=sealed_n, schedule=schedule, dataset=dataset, event=event)
        contract = confirmation_protocol(sealed_n, physical_gpu)
        native = pilot.audit_stage(folder, requests, checkpoints, canonicals=canonicals,
            dataset=dataset, event=event, schedule=planned, evidence_root=evidence_root, production=False)
        errors.extend(native['errors'])
        if json.loads((folder/'CONFIRMATION_PROTOCOL.json').read_text()) != contract:
            errors.append('sealed_confirmation_contract_changed')
        boundaries = [json.loads(line) for line in (folder/'BOUNDARY_CHECKS.jsonl').read_text().splitlines()]
        if [row['batch'] for row in boundaries] != _batches(planned):
            errors.append('confirmation_boundary_check_schedule')
        for row in boundaries:
            errors.extend(boundary_errors(row['check'], physical_gpu))
        if (folder/'CONFIRMATION_BOUNDARY_FAILURE.json').exists():
            errors.append('failed_confirmation_boundary_retained')
        work = json.loads((folder/'CONFIRMATION_STATUS.json').read_text())
        expected_work = _confirmation_work(native['progress'], _read_rows(folder), sealed_n)
        if work != expected_work:
            errors.append('confirmation_work_accounting_changed')
        if expected_work['returned_without_measurement']:
            errors.append('returned_native_response_missing_measurement')
    except (OSError, KeyError, ValueError, TypeError, IndexError, RuntimeError, AttributeError) as error:
        errors.append('missing_or_malformed_confirmation_evidence: '+repr(error))
    return dict(native, status='passed' if not errors else 'invalid_or_incomplete',
        errors=sorted(set(errors)), confirmation=contract, model_work=0)
