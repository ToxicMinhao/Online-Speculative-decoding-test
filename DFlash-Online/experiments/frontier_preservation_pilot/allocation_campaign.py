"""Conditional event-two continuation using one immutable parent feedback pool.

Collection, each warm fork, and endpoint evaluation own separate processes.
The existing allocation phase clock and all historical occupied-card cost carry
forward. Numerical qualification alone never admits this entry.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from experiments.full_k_gsm8k.campaign import online_record_check
from experiments.kplus_full_gsm8k.preflight import timing_barrier
from experiments.kplus_full_gsm8k.storage import retire_redundant_features
from experiments.ok_coding.storage import identity
from experiments.opencode_native_comparison.contract import validate_manifest
from experiments.opencode_native_comparison.training import validate_recovery_file
from experiments.prefix_guard.qualify import pin_retained_graph_runtime
from experiments.repair_conditional_kl_pilot import analysis, evaluation
from experiments.repair_conditional_kl_pilot.campaign import (
    create_backend,
    check_publication,
)
from experiments.repair_conditional_kl_pilot.production import quiet_evaluation
from experiments.repair_conditional_kl_pilot.launch import stop_owned
from experiments.repair_conditional_kl_pilot.resources import (
    Ledger,
    read,
    verify_inputs,
)
from online_feedback.controller import OnlineController
from online_feedback.storage import write_json

from .allocation_launch import total_storage
from .fork import ARMS
from .prepare import INPUT_IDENTITY
from .qualify import STOP_GRACE_S

MODULE = 'experiments.frontier_preservation_pilot.allocation_campaign'
PHASES = (
    ('collect', None),
    ('train', 'CONTROL'),
    ('train', 'FRONTIER_P'),
    ('evaluate', None),
)
BOOTSTRAP_SEED = 20261125


def selected_inputs(plan):
    """Keep source identities 128..255; a local ordinal is separate metadata."""
    if identity(plan['manifest']) != INPUT_IDENTITY:
        raise ValueError('Require the full authenticated PR56 manifest')
    manifest = read(plan['manifest'])
    validate_manifest(manifest)
    requests = manifest['train_requests'][128:256]
    if len(requests) != 128 or [row['request_index'] for row in requests] != list(
        range(128, 256)
    ):
        raise ValueError('Require the original contiguous train[128:256] identities')
    return manifest, requests


def validate_production(plan):
    if (
        plan.get('schema') != 'frontier_allocation_production_v1'
        or plan.get('production_admitted') is not True
        or tuple(plan.get('arms', ())) != ARMS
        or plan.get('bootstrap_seed') != BOOTSTRAP_SEED
        or plan.get('event_id') != 'event_002'
    ):
        raise ValueError(
            'Require the admitted fixed CONTROL/FRONTIER_P event-two contract'
        )
    if set(plan['configs']) != set(ARMS) or set(plan['devices']) != {'A', 'B'}:
        raise ValueError('Require both fork configurations and physical cards')
    if plan['devices']['A'] == plan['devices']['B']:
        raise ValueError('Serving and training cards must be distinct')
    for key in ('numerical_result', 'integration_result'):
        if identity(plan[key]) != plan[key + '_identity']:
            raise ValueError('An admission receipt changed: ' + key)
    numerical = read(plan['numerical_result'])
    if (
        numerical.get('status') != 'numerically_qualified'
        or numerical.get('passed') is not True
        or numerical.get('restoration', {}).get('exact_resume') is not True
    ):
        raise ValueError('Numerical qualification must pass before continuation')
    integration = read(plan['integration_result'])
    if (
        integration.get('status') != 'passed'
        or integration.get('passed') is not True
        or integration.get('production_qualified') is not True
        or integration.get('restoration', {}).get('exact_resume') is not True
        or integration.get('candidate_active') is not True
    ):
        raise ValueError('Native integration and attributable BF16 activity must pass')
    clock = read(plan['phase_clock'])
    if clock['elapsed_limit_s'] != 14400 or clock['occupied_card_limit_s'] != 28800:
        raise ValueError('The original allocation phase limits changed')
    if identity(plan['phase_clock']) != plan['phase_clock_identity']:
        raise ValueError('The original allocation phase clock changed')
    if plan['prior_occupied_card_s'] < clock['prior_occupied_card_s']:
        raise ValueError('Historical occupied-card work cannot be discarded')
    if (
        min(
            plan['collection_write_bound_bytes'], plan['continuation_write_bound_bytes']
        )
        <= 0
    ):
        raise ValueError(
            'Both storage stages require explicit positive write forecasts'
        )
    if min(plan['collection_wall_bound_s'], plan['continuation_wall_bound_s']) <= 0:
        raise ValueError('Both stages require explicit positive time forecasts')


def phase_name(phase, arm=None):
    if (phase, arm) not in PHASES:
        raise ValueError(
            'Only shared collection, two serial forks, and evaluation are permitted'
        )
    return phase if arm is None else phase + '_' + arm


def phase_card(phase, arm=None):
    phase_name(phase, arm)
    return 'B' if phase == 'train' else 'A'


def child_command(plan_path, plan, phase, arm=None):
    card = phase_card(phase, arm)
    source = Path(plan['source_root'])
    environment = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=plan['devices'][card],
        PYTHONDONTWRITEBYTECODE='1',
    )
    environment['PYTHONPATH'] = os.pathsep.join(
        str(source / part) for part in ('.', 'training', 'inference')
    )
    command = [
        plan['training_python'],
        '-u',
        '-m',
        MODULE,
        '--plan',
        str(plan_path),
        '--phase',
        phase,
    ]
    if arm is not None:
        command.extend(['--arm', arm])
    return command, environment


def clock_status(plan):
    clock = read(plan['phase_clock'])
    production = read(Path(plan['root']) / 'OCCUPIED_CARDS.json')
    active = production['active']
    active_s = 0.0 if active is None else time.time() - active['started_epoch']
    return dict(
        first_gpu_action_epoch=clock['first_gpu_action_epoch'],
        new_phase_elapsed_s=time.time() - clock['first_gpu_action_epoch'],
        prior_occupied_card_s=plan['prior_occupied_card_s'],
        production_occupied_card_s=production['completed_card_s'] + active_s,
        cumulative_occupied_card_s=plan['prior_occupied_card_s']
        + production['completed_card_s']
        + active_s,
        active=active,
    )


def boundary(plan):
    """Check the shared clock and the current stage's measured absolute peak."""
    root = Path(plan['root'])
    clock = clock_status(plan)
    if (
        clock['new_phase_elapsed_s'] >= 14400 - STOP_GRACE_S
        or clock['cumulative_occupied_card_s'] >= 28800 - STOP_GRACE_S
    ):
        raise TimeoutError(
            'The original allocation phase time or total card budget is exhausted'
        )
    usage = total_storage(plan)
    admission = read(root / 'STORAGE_STAGE.json')
    charged = usage['all_locations_charged_bytes']
    peak = admission['projected_peak_allocated_bytes']
    remaining = max(0, peak - charged)
    if (
        charged > peak
        or peak > 40_000_000_000
        or usage['free_bytes'] - remaining < 400_000_000_000
    ):
        raise RuntimeError(
            'Continuation exceeded its admitted study storage or free-space reserve'
        )
    return dict(
        clock=clock, storage=usage, admission=admission, remaining_write_bytes=remaining
    )


def admit_stage(plan, stage):
    """Collection can be admitted before its actual feature size is known."""
    usage = total_storage(plan)
    extra = plan[
        (
            'collection_write_bound_bytes'
            if stage == 'collection'
            else 'continuation_write_bound_bytes'
        )
    ]
    peak = usage['all_locations_charged_bytes'] + extra
    clock = clock_status(plan)
    seconds = plan[
        (
            'collection_wall_bound_s'
            if stage == 'collection'
            else 'continuation_wall_bound_s'
        )
    ]
    remaining_seconds = min(
        14400 - clock['new_phase_elapsed_s'],
        28800 - clock['cumulative_occupied_card_s'],
    )
    receipt = dict(
        stage=stage,
        current_allocated_bytes=usage['all_locations_charged_bytes'],
        additional_write_bound_bytes=extra,
        projected_peak_allocated_bytes=peak,
        free_bytes=usage['free_bytes'],
        forecast_wall_s=seconds,
        remaining_time_budget_s=remaining_seconds,
        passed=(
            peak <= 40_000_000_000
            and usage['free_bytes'] - extra >= 400_000_000_000
            and seconds + STOP_GRACE_S < remaining_seconds
        ),
    )
    root = Path(plan['root'])
    write_json(root / (stage.upper() + '_ADMISSION.json'), receipt)
    if not receipt['passed']:
        raise RuntimeError(
            'Measured ' + stage + ' storage cannot admit the fixed remaining work'
        )
    write_json(root / 'STORAGE_STAGE.json', receipt)
    return receipt


def collect_shared(plan):
    root = Path(plan['root'])
    base = root / 'shared'
    base.mkdir(parents=True, exist_ok=False)
    event_root = base / 'event_002'
    event_root.mkdir()
    _, requests = selected_inputs(plan)
    write_json(
        base / 'INPUTS.json',
        [dict(local_ordinal=i, request=row) for i, row in enumerate(requests)],
    )
    ledger = Ledger(base / 'WORK.json')
    status = dict(
        status='running',
        completed=0,
        parent_checkpoint=plan['parent_export'],
        source_indices=list(range(128, 256)),
        feedback_generations=128,
        automatic_retry=False,
    )
    write_json(base / 'STATUS.json', status)
    backend = None
    features = []
    try:
        backend = create_backend(plan, base / 'native', card='A')
        status['publication'] = check_publication(
            backend, plan['parent_export'], base / 'PARENT_PUBLICATION.json'
        )
        controller = OnlineController(
            backend, None, event_root / 'loop', buffer_capacity=128
        )
        generate = backend.generate
        backend.generate = lambda *args, **kwargs: ledger.call(
            {'feedback_generation': 1},
            'shared_parent_feedback',
            generate,
            *args,
            **kwargs,
        )
        for ordinal, request in enumerate(requests):
            boundary(plan)
            record = controller.serve(
                request['prompt_ids'],
                request_id=request['request_id'],
                session_id='frontier_common_parent_event_002',
                request_index=request['request_index'],
                max_new_tokens=2048,
            )
            online_record_check(record, request, plan['parent_export'], capture=True)
            if len(record['response_ids']) < 2:
                raise ValueError(
                    'Shared response has no eligible anchors; retain it without replacement'
                )
            path = event_root / 'loop/features' / (record['record_id'] + '.pt')
            prepared = ledger.call(
                {'teacher_preparation': 1},
                'native_teacher',
                backend.prepare_feedback,
                record,
                path,
            )
            if (
                prepared.get('status') != 'prepared'
                or prepared.get('eligible') is False
            ):
                raise ValueError('Shared native preparation failed without replacement')
            retire_redundant_features(record, prepared, root, event_root)
            features.append(
                dict(
                    path=str(path),
                    identity=identity(path),
                    record_id=record['record_id'],
                    source_request_id=request['request_id'],
                    source_index=request['request_index'],
                    local_ordinal=ordinal,
                    preparation=prepared,
                )
            )
            write_json(base / 'FEATURE_IDENTITIES.json', features)
            status['completed'] = len(features)
            write_json(base / 'STATUS.json', status)
        status['sealed_event'] = controller.buffer.seal_event('event_002', 128)
        expected = dict(feedback_generation=128, teacher_preparation=128)
        work = read(base / 'WORK.json')
        if (
            work['pending']
            or work['attempted'] != expected
            or work['completed'] != expected
        ):
            raise ValueError('Shared feedback work does not reconcile')
        status['status'] = 'completed'
    except BaseException as error:
        status.update(
            status='failed', error=repr(error), traceback=traceback.format_exc()
        )
        raise
    finally:
        if backend is not None:
            try:
                if not backend.failed:
                    write_json(
                        base / 'PHYSICAL_WORK.json', backend._call('work_receipt')
                    )
            finally:
                backend.close()
        write_json(base / 'STATUS.json', status)
    return status


def shared_features(plan):
    base = Path(plan['root']) / 'shared'
    if read(base / 'STATUS.json').get('status') != 'completed':
        raise ValueError('Both forks require the one completed common-parent buffer')
    rows = read(base / 'FEATURE_IDENTITIES.json')
    _, requests = selected_inputs(plan)
    if len(rows) != 128:
        raise ValueError('The common training buffer must have exactly 128 responses')
    for ordinal, (row, request) in enumerate(zip(rows, requests)):
        if (
            row['local_ordinal'] != ordinal
            or row['source_index'] != request['request_index']
            or row['source_request_id'] != request['request_id']
            or identity(row['path']) != row['identity']
        ):
            raise ValueError(
                'A shared feature or its unmodified source identity changed'
            )
    return [row['path'] for row in rows]


def train_arm(plan, arm):
    from .allocation_training import ForkTrainer
    from .allocation_witness import freeze_sets, evaluate_sets

    if arm not in ARMS:
        raise ValueError('Unknown fork arm')
    root = Path(plan['root'])
    base = root / 'arms' / arm
    base.mkdir(parents=True, exist_ok=False)
    paths = shared_features(plan)
    status = dict(status='running', arm=arm, event_id='event_002')
    write_json(base / 'STATUS.json', status)
    trainer = None
    ledger = Ledger(base / 'WORK.json')
    try:
        runtime = pin_retained_graph_runtime(base)
        runtime.update(
            scope='warm frontier event-two fork', applied_before='parent restoration'
        )
        write_json(base / 'RUNTIME_COMPATIBILITY.json', runtime)
        trainer = ForkTrainer(
            plan['configs'][arm],
            base / 'trainer',
            arm=arm,
            binding=plan['binding'],
            parent_plan=plan['parent_plan'],
        )
        status['startup'] = trainer.discarded_startup(paths, base / 'startup')
        witness = root / 'witness'
        if arm == 'CONTROL':
            parent = read(plan['parent_plan'])
            old = read(Path(parent['root']) / 'arms/PAIR/FEATURE_IDENTITIES.json')[:8]
            for row in old:
                if identity(row['path']) != row['identity']:
                    raise ValueError('An original parent witness packet changed')
            status['witness_seal'] = freeze_sets(
                trainer, paths, [row['path'] for row in old], witness
            )
            status['parent_witness'] = evaluate_sets(trainer, witness, 'parent')
        for owner, name, kind in (
            (trainer.strategy, 'forward_loss', 'response_forward'),
            (trainer.backend, 'backward', 'response_backward'),
            (trainer.backend.optimizer.optimizer, 'step', 'adam'),
        ):
            ledger.wrap(owner, name, kind)
        status['training'] = trainer.train_event('event_002', paths, passes=2)
        checkpoint = status['training']['checkpoint_path']
        status['boundary'] = trainer.verify_boundary(
            checkpoint, base / 'FINAL_STATE_CHECK.json'
        )
        if (
            trainer.global_step != 256
            or trainer.presentations != 512
            or len(trainer.events) != 2
        ):
            raise ValueError('The fork endpoint is not the fixed event-two boundary')
        status['recovery_size'] = validate_recovery_file(trainer.latest)
        status['endpoint_witness'] = evaluate_sets(trainer, witness, arm)
        work = read(base / 'WORK.json')
        expected = dict(response_forward=256, response_backward=256, adam=128)
        if (
            work['pending']
            or work['attempted'] != expected
            or work['completed'] != expected
        ):
            raise ValueError('Committed fork teaching work does not reconcile')
        status.update(
            status='completed',
            checkpoint=checkpoint,
            recovery=str(trainer.latest),
            checkpoint_identity=identity(Path(checkpoint) / 'model.safetensors'),
            recovery_identity=identity(trainer.latest),
            global_step=256,
            presentations=512,
            completed_event=2,
        )
    except BaseException as error:
        status.update(
            status='failed', error=repr(error), traceback=traceback.format_exc()
        )
        raise
    finally:
        write_json(base / 'STATUS.json', status)
        if trainer is not None:
            trainer.close()
    return status


def evaluate_endpoints(plan):
    root = Path(plan['root'])
    inputs, _ = selected_inputs(plan)
    checkpoints = {}
    for arm in ARMS:
        receipt = read(root / 'arms' / arm / 'STATUS.json')
        if receipt.get('status') != 'completed' or receipt.get('global_step') != 256:
            raise ValueError('Require both newly completed event-two endpoints')
        path = receipt['checkpoint']
        if identity(Path(path) / 'model.safetensors') != receipt['checkpoint_identity']:
            raise ValueError('An endpoint export changed before evaluation')
        checkpoints[arm] = path
    ledger = Ledger(root / 'evaluation_work.json')
    backend = None

    def checked_boundary(slot):
        return dict(
            slot=slot, resources=boundary(plan), quiet=quiet_evaluation(plan, backend)
        )

    def activate(arm, checkpoint, slot):
        nonlocal backend
        loading_s = 0.0
        if backend is None:
            backend = create_backend(plan, root / 'evaluation_native', card='A')
            loading_s = backend.loading_s
        return backend, dict(
            publication=backend.publish(checkpoint), loading_s=loading_s
        )

    try:
        result = evaluation.execute(
            plan,
            root / 'evaluation',
            inputs['evaluation_requests'][:128],
            inputs['warmup_requests'],
            read(plan['canonicals']),
            read(plan['warmup_canonicals']),
            checkpoints,
            activate=activate,
            boundary=checked_boundary,
            ledger=ledger,
            methods=ARMS,
        )
        actual = read(root / 'evaluation_work.json')
        expected = dict(scored_generation=768, warmup_generation=24)
        if (
            actual['pending']
            or actual['attempted'] != expected
            or actual['completed'] != expected
        ):
            raise ValueError('Fixed endpoint evaluation work does not reconcile')
        return result
    finally:
        if backend is not None:
            try:
                if not backend.failed:
                    write_json(
                        root / 'evaluation_physical_work.json',
                        backend._call('work_receipt'),
                    )
            finally:
                backend.close()


def run_phase(plan_path, phase, arm=None):
    plan = read(plan_path)
    verify_inputs(plan)
    validate_production(plan)
    name = phase_name(phase, arm)
    if (
        os.environ.get('CUDA_VISIBLE_DEVICES')
        != plan['devices'][phase_card(phase, arm)]
    ):
        raise ValueError('Phase child is not isolated on its sealed physical GPU')
    index = PHASES.index((phase, arm))
    for previous in PHASES[:index]:
        receipt = Path(plan['root']) / 'phases' / (phase_name(*previous) + '.json')
        if not receipt.is_file() or read(receipt).get('status') != 'completed':
            raise ValueError(
                'A preceding shared-collection or fork phase is incomplete'
            )
    path = Path(plan['root']) / 'phases' / (name + '.json')
    if path.exists():
        raise FileExistsError('No automatic restart of an attempted phase')
    result = dict(status='running', phase=phase, arm=arm, started_epoch=time.time())
    write_json(path, result)
    try:
        boundary(plan)
        if phase == 'collect':
            result['result'] = collect_shared(plan)
        elif phase == 'train':
            result['result'] = train_arm(plan, arm)
        else:
            result['result'] = evaluate_endpoints(plan)
        result['status'] = 'completed'
    except BaseException as error:
        result.update(
            status='failed', error=repr(error), traceback=traceback.format_exc()
        )
        raise
    finally:
        result['wall_s'] = time.time() - result['started_epoch']
        write_json(path, result)
    return result


def run_child(plan_path, plan, phase, arm=None):
    root = Path(plan['root'])
    name = phase_name(phase, arm)
    path = root / 'phases' / (name + '.launch.json')
    if path.exists():
        raise FileExistsError('This child was already attempted')
    command, environment = child_command(plan_path, plan, phase, arm)
    result = dict(
        status='starting',
        command=command,
        phase=phase,
        arm=arm,
        started_epoch=time.time(),
        automatic_retry=False,
    )
    write_json(path, result)
    accounting = root / 'OCCUPIED_CARDS.json'
    charge = read(accounting)
    if charge['active'] is not None:
        raise ValueError('An unresolved occupied-card phase already exists')
    charge['active'] = dict(
        phase=name,
        started_epoch=time.time(),
        physical_gpu_uuid=plan['devices'][phase_card(phase, arm)],
    )
    write_json(accounting, charge)
    process = None
    try:
        with (root / (name + '.log')).open('x') as log:
            process = subprocess.Popen(
                command,
                cwd=plan['source_root'],
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            result.update(status='running', pid=process.pid)
            write_json(path, result)
            while process.poll() is None:
                boundary(plan)
                time.sleep(5)
        result['exit_code'] = process.returncode
        if (
            process.returncode
            or read(root / 'phases' / (name + '.json')).get('status') != 'completed'
        ):
            raise RuntimeError('The attempted phase did not complete: ' + name)
        result['status'] = 'completed'
    except BaseException as error:
        result.update(status='failed', error=repr(error))
        raise
    finally:
        closure_error = None
        try:
            if process is not None:
                stop_owned(process)
                result['exit_code'] = process.returncode
                timing_barrier()
            result['clean_process_closure'] = True
        except BaseException as error:
            closure_error = error
            result.update(
                status='failed',
                clean_process_closure=False,
                closure_error=repr(error),
                occupied_card_status='unresolved_still_charged',
            )
        elapsed = time.time() - charge['active']['started_epoch']
        if closure_error is None:
            charge['completed_card_s'] += elapsed
            charge['finished'].append(
                dict(**charge['active'], elapsed_s=elapsed, status=result['status'])
            )
            charge['active'] = None
        write_json(accounting, charge)
        result['wall_s'] = time.time() - result['started_epoch']
        write_json(path, result)
        if closure_error is not None:
            raise closure_error
    return result


def summarize(plan):
    from .allocation_witness import summarize_sets

    root = Path(plan['root'])
    status = read(root / 'evaluation/STATUS.json')
    if (
        status.get('status') != 'completed'
        or status.get('completed_valid') != 792
        or any(status.get(key) for key in ('invalid', 'unknown', 'unattempted'))
    ):
        raise ValueError(
            'Incomplete endpoint evidence cannot produce an efficacy comparison'
        )
    rows = [
        json.loads(line)
        for line in (root / 'evaluation/MEASUREMENTS.jsonl').read_text().splitlines()
    ]
    destination = root / 'analysis'
    destination.mkdir(exist_ok=False)
    result = analysis.summarize(rows, methods=ARMS, bootstrap_seed=BOOTSTRAP_SEED)
    write_json(destination / 'SERVING.json', result)
    write_json(destination / 'WITNESSES.json', summarize_sets(root / 'witness'))
    write_json(
        destination / 'COSTS_AND_WORK.json',
        dict(
            clock=clock_status(plan),
            shared_work=read(root / 'shared/WORK.json'),
            arm_status={arm: read(root / 'arms' / arm / 'STATUS.json') for arm in ARMS},
            arm_work={arm: read(root / 'arms' / arm / 'WORK.json') for arm in ARMS},
            evaluation_work=read(root / 'evaluation_work.json'),
            historical_hidden_state_gate='failed_retained_diagnostic_not_production_veto',
        ),
    )
    return dict(
        status='completed',
        serving=str(destination / 'SERVING.json'),
        witnesses=str(destination / 'WITNESSES.json'),
    )


def launch(plan_path):
    plan = read(plan_path)
    verify_inputs(plan)
    validate_production(plan)
    root = Path(plan['root'])
    if (root / 'LAUNCH.json').exists() or (root / 'OCCUPIED_CARDS.json').exists():
        raise FileExistsError('No automatic continuation restart or cost reset')
    write_json(
        root / 'OCCUPIED_CARDS.json',
        dict(completed_card_s=0.0, active=None, finished=[]),
    )
    admit_stage(plan, 'collection')
    result = dict(
        status='running',
        phases=[],
        pending=None,
        unattempted=[phase_name(*phase) for phase in PHASES],
        automatic_retry=False,
    )
    path = root / 'LAUNCH.json'
    write_json(path, result)
    try:
        for phase, arm in PHASES:
            if phase == 'train' and arm == 'CONTROL':
                admit_stage(plan, 'continuation')
            boundary(plan)
            timing_barrier()
            name = phase_name(phase, arm)
            result['pending'] = name
            result['unattempted'].remove(name)
            write_json(path, result)
            child = run_child(plan_path, plan, phase, arm)
            timing_barrier()
            result['phases'].append(child)
            result['pending'] = None
            write_json(path, result)
        shared_features(plan)
        result['analysis'] = summarize(plan)
        result['status'] = 'completed'
    except BaseException as error:
        result.update(
            status='failed', error=repr(error), traceback=traceback.format_exc()
        )
        raise
    finally:
        result['clock'] = clock_status(plan)
        write_json(path, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument(
        '--phase', choices=('launch', 'collect', 'train', 'evaluate'), default='launch'
    )
    parser.add_argument('--arm', choices=ARMS)
    args = parser.parse_args()
    if args.phase == 'launch':
        if args.arm is not None:
            parser.error('The coordinator always runs both forks')
        launch(args.plan)
    else:
        run_phase(args.plan, args.phase, args.arm)
