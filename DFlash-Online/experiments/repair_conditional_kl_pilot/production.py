"""One BASE event, one PAIR event, then quiet-card evaluation and analysis.

Every model phase runs in a fresh child process. This coordinator never invokes
the historical disposable qualification and never resumes a failed phase.
"""

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from online_feedback.storage import write_json
from experiments.kplus_full_gsm8k.preflight import timing_barrier
from experiments.ok_coding.storage import identity
from . import analysis, evaluation
from .campaign import create_backend, qualification_reuse, run_arm
from .launch import stop_owned
from .resources import (
    boundary,
    clock_status,
    initialize,
    Ledger,
    occupied_card,
    read,
    verify_inputs,
)

PHASES = (('train', 'BASE'), ('train', 'PAIR'), ('evaluate', None))
MODULE = 'experiments.repair_conditional_kl_pilot.production'


def validate_production(plan):
    """Reject blocked, foreign or expanded plans before constructing a runtime."""
    if (
        plan.get('execution_mode') != 'one_event_production'
        or plan.get('production_admitted') is not True
        or plan.get('maximum_events') != 1
        or plan.get('arms') != ['BASE', 'PAIR']
    ):
        raise ValueError('Require the newly admitted, fixed one-event production plan')
    if plan.get('feature_strategy') not in ('local_single', 'local_split'):
        raise ValueError('A verified offload route has not been implemented or admitted')
    if set(plan.get('feature_roots', {})) != {'BASE', 'PAIR'}:
        raise ValueError('Both own-feedback feature locations must be explicitly sealed')
    roots = [Path(plan['feature_roots'][arm]) for arm in ('BASE', 'PAIR')]
    if any(not path.is_absolute() for path in roots) or roots[0] == roots[1]:
        raise ValueError('Feature roots must be distinct absolute locations')
    devices = plan.get('devices', {})
    if set(devices) != {'A', 'B'} or devices['A'] == devices['B']:
        raise ValueError('Seal two distinct physical GPU UUIDs for serving A and training B')
    qualification_reuse(plan)


def phase_name(phase, arm=None):
    if (phase, arm) not in PHASES:
        raise ValueError('Only BASE1, PAIR1 and their fixed evaluation are permitted')
    return phase if arm is None else phase + '_' + arm


def phase_cards(phase, arm=None):
    phase_name(phase, arm)
    return ('A', 'B') if phase == 'train' else ('A',)


def child_command(plan_path, plan, phase, arm=None):
    """Both training children see B; their native serving workers use sealed A."""
    phase_name(phase, arm)
    card = 'B' if phase == 'train' else 'A'
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=plan['devices'][card])
    source = Path(plan['source_root'])
    environment['PYTHONPATH'] = os.pathsep.join(
        str(path) for path in (source, source / 'training', source / 'inference')
    )
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
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


def quiet_evaluation(plan, backend=None):
    """Allow only the serving worker tree; the other physical card stays idle."""
    worker_pid = None if backend is None else backend.worker.pid
    quiet = timing_barrier(worker_pid)
    devices = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader,nounits'],
        text=True,
    )
    active = []
    for line in devices.splitlines():
        fields = [field.strip() for field in line.split(',')]
        if len(fields) == 2 and fields[0].isdigit():
            active.append(dict(pid=int(fields[0]), physical_gpu_uuid=fields[1]))
    if any(row['physical_gpu_uuid'] != plan['devices']['A'] for row in active):
        raise ValueError('Scored evaluation requires the second card idle')
    return dict(quiet, physical_gpu_uuid=plan['devices']['A'], active=active)


def evaluate_endpoints(plan):
    """Connect the existing 768-score/24-warmup schedule to final own exports."""
    root = Path(plan['root'])
    inputs = read(plan['manifest'])
    checkpoints = {}
    for arm in ('BASE', 'PAIR'):
        receipt = read(root / 'arms' / arm / 'STATUS.json')
        if receipt.get('status') != 'completed' or receipt.get('phase') != 'completed':
            raise ValueError('Both complete event-one endpoints are required before evaluation')
        checkpoint = receipt['checkpoint']
        if identity(Path(checkpoint) / 'model.safetensors') != receipt['checkpoint_identity']:
            raise ValueError('A published event-one export changed before evaluation')
        checkpoints[arm] = checkpoint
    ledger = Ledger(root / 'evaluation_work.json')
    backend = None

    def evaluation_boundary(slot):
        return dict(slot=slot, resources=boundary(plan), quiet=quiet_evaluation(plan, backend))

    def activate(arm, checkpoint, slot):
        nonlocal backend
        loading_s = 0.0
        if backend is None:
            backend = create_backend(plan, root / 'evaluation_native', card='A')
            loading_s = backend.loading_s
        publication = backend.publish(checkpoint)
        return backend, dict(publication=publication, loading_s=loading_s)

    try:
        result = evaluation.execute(
            plan,
            root / 'evaluation',
            inputs['evaluation_requests'],
            inputs['warmup_requests'],
            read(root / 'sealed' / 'CANONICALS.json'),
            read(root / 'sealed' / 'WARMUP_CANONICALS.json'),
            checkpoints,
            activate=activate,
            boundary=evaluation_boundary,
            ledger=ledger,
        )
        work = read(root / 'evaluation_work.json')
        expected = dict(scored_generation=768, warmup_generation=24)
        if work['pending'] or work['attempted'] != expected or work['completed'] != expected:
            raise ValueError('Fixed endpoint evaluation work does not reconcile')
        return result
    finally:
        if backend is not None:
            try:
                if not backend.failed:
                    write_json(
                        root / 'evaluation_physical_work.json', backend._call('work_receipt')
                    )
            finally:
                backend.close()


def run_phase(plan_path, phase, arm=None):
    """A child owns exactly its declared phase, with durable attempted status."""
    plan = read(plan_path)
    verify_inputs(plan)
    validate_production(plan)
    name = phase_name(phase, arm)
    expected_device = plan['devices']['B' if phase == 'train' else 'A']
    if os.environ.get('CUDA_VISIBLE_DEVICES') != expected_device:
        raise ValueError('Production child is not isolated on its declared physical card')
    root = Path(plan['root'])
    path = root / 'phases' / (name + '.json')
    if path.exists():
        raise FileExistsError('No automatic repeat of an attempted production phase')
    result = dict(
        status='running',
        phase=phase,
        arm=arm,
        execution_commit=plan['execution_commit'],
        started_epoch=time.time(),
        automatic_retry=False,
    )
    write_json(path, result)
    try:
        boundary(plan)
        result['result'] = run_arm(plan, arm) if phase == 'train' else evaluate_endpoints(plan)
        result['status'] = 'completed'
    except BaseException as error:
        result.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        result['wall_s'] = time.time() - result['started_epoch']
        write_json(path, result)
    return result


def run_child(plan_path, plan, phase, arm=None):
    """Watch the owned process and stop it if the cumulative limit is crossed."""
    root = Path(plan['root'])
    name = phase_name(phase, arm)
    path = root / 'phases' / (name + '.launch.json')
    if path.exists():
        raise FileExistsError('This production child was already attempted')
    command, environment = child_command(plan_path, plan, phase, arm)
    log_path = root / (name + '.log')
    result = dict(
        status='starting',
        phase=phase,
        arm=arm,
        started_epoch=time.time(),
        command=command,
        log=str(log_path),
        automatic_retry=False,
    )
    write_json(path, result)
    process = None
    try:
        with log_path.open('x') as log:
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
                result['wall_s'] = time.time() - result['started_epoch']
                write_json(path, result)
                time.sleep(5)
        result['exit_code'] = process.returncode
        if process.returncode:
            raise RuntimeError(name + ' failed; retain its work ledger and do not retry')
        child_result = read(root / 'phases' / (name + '.json'))
        if child_result.get('status') != 'completed':
            raise ValueError('Child exit did not establish a completed production phase')
        result['status'] = 'completed'
    except BaseException as error:
        result.update(status='failed', error=repr(error))
        raise
    finally:
        if process is not None and process.poll() is None:
            stop_owned(process)
            result['terminated_by_watchdog'] = True
            result['exit_code'] = process.returncode
        result['wall_s'] = time.time() - result['started_epoch']
        write_json(path, result)
    return result


def summarize_production(plan):
    """Read all fixed endpoints once; retain matched requests and block observations."""
    from .witness import compare_matched_witnesses

    root = Path(plan['root'])
    status = read(root / 'evaluation' / 'STATUS.json')
    if (
        status.get('status') != 'completed'
        or status.get('completed_valid') != 792
        or any(status.get(key) for key in ('invalid', 'unknown', 'unattempted'))
    ):
        raise ValueError('Incomplete evaluation cannot produce an efficacy comparison')
    rows = [
        json.loads(line)
        for line in (root / 'evaluation' / 'MEASUREMENTS.jsonl').read_text().splitlines()
    ]
    output = root / 'analysis'
    output.mkdir(exist_ok=False)
    summary = analysis.summarize(rows)
    write_json(output / 'SERVING.json', summary)
    witness = root / 'witness'
    matched = compare_matched_witnesses(
        read(witness / 'origin.json'), read(witness / 'BASE1.json'), read(witness / 'PAIR1.json')
    )
    write_json(output / 'WITNESS_COMPARISON.json', matched)
    costs = dict(
        qualification=qualification_reuse(plan),
        production_arms={
            arm: read(root / 'arms' / arm / 'STATUS.json') for arm in ('BASE', 'PAIR')
        },
        evaluation=status,
        new_clock=clock_status(root),
        work={arm: read(root / 'arms' / arm / 'WORK.json') for arm in ('BASE', 'PAIR')},
        evaluation_work=read(root / 'evaluation_work.json'),
        historical_qualification_reexecuted=False,
    )
    write_json(output / 'COSTS_AND_WORK.json', costs)
    return dict(
        status='completed',
        serving=str(output / 'SERVING.json'),
        witness=str(output / 'WITNESS_COMPARISON.json'),
        costs=str(output / 'COSTS_AND_WORK.json'),
    )


def launch(plan_path):
    """Admit once, train serial arms, wait for clean exits, score, and stop."""
    plan = read(plan_path)
    verify_inputs(plan)
    validate_production(plan)
    root = Path(plan['root'])
    if read(root / 'CPU_PREFLIGHT.json').get('status') != 'passed':
        raise ValueError('The new production source/input/resource preflight must pass')
    if (root / 'LAUNCH.json').exists() or (root / 'CLOCK.json').exists():
        raise FileExistsError('A production attempt already exists; no automatic restart')
    boundary(plan)
    timing_barrier()
    initialize(plan)
    path = root / 'LAUNCH.json'
    result = dict(
        status='running',
        execution_commit=plan['execution_commit'],
        phases=[],
        pending=None,
        unattempted=[phase_name(*item) for item in PHASES],
        qualification_reused=True,
        qualification_executed=False,
        automatic_retry=False,
    )
    write_json(path, result)
    try:
        for phase, arm in PHASES:
            name = phase_name(phase, arm)
            boundary(plan)
            timing_barrier()
            result['pending'] = name
            result['unattempted'].remove(name)
            write_json(path, result)
            with ExitStack() as stack:
                for card in phase_cards(phase, arm):
                    stack.enter_context(
                        occupied_card(root, name + '_' + card, plan['devices'][card])
                    )
                child = run_child(plan_path, plan, phase, arm)
            # The child and all its owned runtimes have exited before the next
            # phase's idle barrier. No training or preparation overlaps timing.
            timing_barrier()
            result['phases'].append(child)
            result['pending'] = None
            write_json(path, result)
        result['analysis'] = summarize_production(plan)
        boundary(plan)
        result['status'] = 'completed'
    except BaseException as error:
        result.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        result['clock'] = clock_status(root)
        write_json(path, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--phase', choices=('launch', 'train', 'evaluate'), default='launch')
    parser.add_argument('--arm', choices=('BASE', 'PAIR'))
    args = parser.parse_args()
    if args.phase == 'launch':
        if args.arm is not None:
            parser.error('The coordinator always runs both declared arms')
        launch(args.plan)
    else:
        run_phase(args.plan, args.phase, args.arm)
