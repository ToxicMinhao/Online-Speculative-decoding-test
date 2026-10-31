"""Compatibility first, then two ordinary O_HALF trajectories and count evaluation."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import traceback

from online_feedback.storage import write_json
from experiments.ok_coding.attempts import reserve_directory, reserve_receipt
from experiments.ok_coding.campaign import fresh_worker_environment, read_work, require_complete_work
from experiments.ok_events_5_8.storage import identity
from .contract import EVENT_WORK, SEEDS, validate_plan
from .storage import admit, retire_odd_boundary, validate_retired


def read(path):
    return json.loads(Path(path).read_text())


def job_paths(root, seed, event):
    base = Path(root) / 'campaign_001' / f'seed_{seed}'
    return base / f'resume_{event:03d}_O_HALF', base / 'O_HALF/events' / f'event_{event:03d}'


def evaluation_path(root, seed, phase):
    base = Path(root) / 'campaign_001'
    return base / f'bridge_seed_{seed}' if phase == 'bridge' else base / f'seed_{seed}/event_004/evaluation'


def completed_job(root, plan, seed, event, *, decode=False):
    from experiments.ok_coding.training import validate_event_receipt, _export_equal
    audit, endpoint = job_paths(root, seed, event)
    worker = read(audit / 'WORKER.json')
    if worker['status'] != 'completed' or read(audit / 'EXEC.json')['returncode'] != 0:
        raise ValueError('Partial training cannot be retried: ' + str(audit))
    require_complete_work(read_work(audit / 'WORK.jsonl', EVENT_WORK), EVENT_WORK)
    result = read(endpoint / 'RESULT.json')
    schedule = read(plan['original_schedules'][str(seed)][str(event)]['path'])
    validate_event_receipt(result, event_number=event, objective='osd', reference_schedule=schedule)
    saved, live, closed = (read(audit / name) for name in ('SAVED_ENDPOINT.json', 'LIVE_BOUNDARY.json', 'CLOSED.json'))
    if (saved['status'] != 'passed' or not saved['export_exact'] or live['exact_resume'] is not True
            or closed['global_step'] != 128 * event or closed['presentations'] != 256 * event):
        raise ValueError('Saved/live/closed O_HALF boundary differs')
    for filename, expected in read(audit / 'STATE_IDENTITIES.json').items():
        if Path(filename).exists():
            if identity(filename) != expected:
                raise ValueError('Own retained endpoint changed: ' + filename)
        else:
            validate_retired(root, seed, event, filename, expected)
    if decode and (endpoint / 'training_state.pt').exists():
        import torch
        from .training_worker import validate_own
        state = torch.load(endpoint / 'training_state.pt', map_location='cpu', weights_only=False)
        validate_own(state, plan, plan['configs'][str(seed)]['O_HALF'], seed, event)
        _export_equal(state, endpoint / 'export')
    return result, read(audit / 'COSTS.json')


def validate_evaluation(root, plan, seed, phase):
    from .evaluation import audit_stage
    path = evaluation_path(root, seed, phase)
    archive = read(plan['archive']['path'])
    checkpoints = archive['checkpoints'][str(seed)] if phase == 'bridge' else {
        'O': plan['original_endpoints'][str(seed)]['O']['export'],
        'O_HALF': str(job_paths(root, seed, 4)[1] / 'export')}
    checked = audit_stage(path, plan['evaluation_requests'], checkpoints,
                          seed=seed, phase=phase, archive=archive)
    rows, errors = checked['rows'], checked['errors']
    expected = 24 if phase == 'bridge' else 512
    if errors or len(rows) != expected or read(path / 'STATUS.json')['status'] != 'completed':
        raise ValueError('Incomplete/invalid evaluation cannot be retried: ' + str(path) + str(errors[:5]))
    return rows


def inspect_boundary(root, plan):
    """Validate the whole chronological prefix before computing remaining writes."""
    bridge, completed, evaluations = [], [], []
    gap = False
    for seed in SEEDS:
        path = evaluation_path(root, seed, 'bridge')
        if not path.exists():
            gap = True
        else:
            if gap:
                raise ValueError('Bridge exists after a chronology gap')
            validate_evaluation(root, plan, seed, 'bridge')
            bridge.append(seed)
    gap = False
    for seed in SEEDS:
        for event in range(1, 5):
            audit, endpoint = job_paths(root, seed, event)
            if not (audit.exists() or endpoint.exists()):
                gap = True
                continue
            if gap or bridge != list(SEEDS):
                raise ValueError('Training exists before complete bridge or after a gap')
            if seed == 29 and 17 not in evaluations:
                raise ValueError('Seed29 training precedes seed17 evaluation')
            completed_job(root, plan, seed, event, decode=True)
            completed.append([seed, event])
        path = evaluation_path(root, seed, 'final')
        if path.exists():
            if [seed, 4] not in completed:
                raise ValueError('Final evaluation precedes its own endpoint')
            validate_evaluation(root, plan, seed, 'final')
            evaluations.append(seed)
    return dict(bridge_complete=bridge, completed_jobs=completed,
                completed_initials=len({seed for seed, _ in completed}), evaluation_complete=evaluations)


def dispatch_training(root, plan, seed, event):
    audit, _ = job_paths(root, seed, event)
    audit.mkdir(parents=True, exist_ok=False)
    environment = fresh_worker_environment(audit / 'caches')
    command = [plan['training_python'], '-u', '-m', 'experiments.o_half_matched_lr.training_worker',
               str(root), '--seed', str(seed), '--event', str(event)]
    write_json(audit / 'JOB.json', dict(command=command, seed=seed, event=event,
               arm='O_HALF', execution_source=plan['execution_source']))
    started = time.perf_counter()
    with (audit / 'WORKER.log').open('w') as stream:
        result = subprocess.run(command, env=environment, stdout=stream, stderr=subprocess.STDOUT)
    write_json(audit / 'EXEC.json', dict(command=command, returncode=result.returncode,
               wall_s=time.perf_counter() - started, automatic_retry=False))
    if result.returncode:
        raise RuntimeError('Training worker failed: ' + str(audit))
    return completed_job(root, plan, seed, event)


def run_evaluation(root, plan, seed, phase):
    from online_feedback.sglang_backend import SGLangDFlashBackend
    from .evaluation import evaluate_bridge, evaluate_final
    attempt = reserve_directory(Path(root) / 'campaign_001/native_attempts')
    receipt = dict(status='starting', seed=seed, phase=phase, native_path=str(attempt / 'native'))
    write_json(attempt / 'STATUS.json', receipt)
    archive = read(plan['archive']['path'])
    checkpoints = archive['checkpoints'][str(seed)] if phase == 'bridge' else {
        'O': plan['original_endpoints'][str(seed)]['O']['export'],
        'O_HALF': str(job_paths(root, seed, 4)[1] / 'export')}
    backend = None
    try:
        backend = SGLangDFlashBackend(plan['target'], plan['draft'], worker_python=plan['worker_python'],
                    output_root=attempt / 'native', serving_gpu='0', context_limit=4096, block_size=16)
        evaluate = evaluate_bridge if phase == 'bridge' else evaluate_final
        costs = evaluate(backend, plan['evaluation_requests'], checkpoints,
                         evaluation_path(root, seed, phase), seed=seed, archive=archive)
        costs['native_loading_s'] = backend.loading_s
        receipt['status'] = 'completed'
        validate_evaluation(root, plan, seed, phase)
        return costs
    except BaseException as error:
        receipt.update(status='failed', error=repr(error))
        raise
    finally:
        if backend is not None:
            backend.close()
        write_json(attempt / 'STATUS.json', receipt)


def reconcile(root):
    totals = {phase: {kind: 0 for kind in EVENT_WORK} for phase in ('attempts', 'completed', 'unknown')}
    counters, presentations, evaluations = {}, {}, {}
    for seed in SEEDS:
        key = f'{seed}/O_HALF'
        counters[key] = presentations[key] = 0
        for event in range(1, 5):
            audit, _ = job_paths(root, seed, event)
            for phase, values in read_work(audit / 'WORK.jsonl', EVENT_WORK).items():
                for kind, value in values.items():
                    totals[phase][kind] += value
            if (audit / 'CLOSED.json').exists():
                row = read(audit / 'CLOSED.json')
                counters[key], presentations[key] = row['global_step'], row['presentations']
        for phase in ('bridge', 'final'):
            path = evaluation_path(root, seed, phase) / 'STATUS.json'
            if path.exists():
                evaluations[f'{seed}/{phase}'] = read(path)
    result = dict(work=totals, arm_counters=counters, arm_presentations=presentations,
                  evaluations=evaluations, optimizer_steps=totals['completed']['optimizer_steps'],
                  presentations=sum(presentations.values()))
    for suffix, key in (('attempted', 'attempted_generations'), ('returned', 'returned_generations'),
                       ('valid', 'completed_valid_generations'), ('unknown', 'unknown_generation_attempts'),
                       ('invalid', 'invalid_generations'), ('pending', 'pending_canonical_generations')):
        result['evaluation_' + suffix] = sum(row.get(key, 0) for row in evaluations.values())
    return result


def require_complete(progress):
    require_complete_work(progress['work'], dict(forwards=2048, backwards=2048, optimizer_steps=1024))
    if any(v != 512 for v in progress['arm_counters'].values()) or any(v != 1024 for v in progress['arm_presentations'].values()):
        raise ValueError('Incomplete O_HALF own-state trajectories')
    for key, value in dict(attempted=1072, returned=1072, valid=1072, unknown=0, invalid=0, pending=0).items():
        if progress['evaluation_' + key] != value:
            raise ValueError('Incomplete or invalid count-focused serving work')


def run(root, *, resume=False):
    from .preflight import verify
    root = Path(root).resolve()
    plan = read(root / 'sealed/PLAN.json')
    validate_plan(plan)
    verify(root, postflight=True, receipt_path=reserve_receipt(root / 'source_verification'))
    output = root / 'campaign_001'
    if output.exists() and not resume:
        raise FileExistsError('Explicit complete-boundary recovery is required')
    boundary = inspect_boundary(root, plan) if resume else dict(
        bridge_complete=[], completed_jobs=[], completed_initials=0, evaluation_complete=[])
    admit(root, completed_jobs=len(boundary['completed_jobs']), completed_initials=boundary['completed_initials'],
          receipt=str(reserve_receipt(root / 'production_admission').relative_to(root)))
    output.mkdir(exist_ok=resume)
    progress = dict(status='running', phase='bridge', automatic_retry=False,
                    execution_source=plan['execution_source'], explicit_resume=resume)
    costs = read(output / 'COSTS.json') if (output / 'COSTS.json').exists() else {}
    started = time.perf_counter()
    try:
        for seed in SEEDS:
            if seed not in boundary['bridge_complete']:
                progress.update(seed=seed, phase='bridge')
                write_json(output / 'STATUS.json', {**progress, **reconcile(root)})
                costs[f'{seed}/bridge'] = run_evaluation(root, plan, seed, 'bridge')
                write_json(output / 'COSTS.json', costs)
        write_json(output / 'BRIDGE_PASSED.json', dict(status='passed', generations=48,
                   seeds=list(SEEDS), extra_generations=0, before_all_new_training=True))
        completed = len(boundary['completed_jobs'])
        for seed in SEEDS:
            for event in range(1, 5):
                if [seed, event] in boundary['completed_jobs']:
                    # Idempotently finish any declared retirement after a closed successor.
                    retire_odd_boundary(root, seed, event)
                    continue
                progress.update(seed=seed, event=event, phase='training')
                write_json(output / 'STATUS.json', {**progress, **reconcile(root)})
                initials = 0 if completed == 0 else (1 if completed <= 4 else 2)
                admit(root, completed_jobs=completed, completed_initials=initials,
                      receipt=str(reserve_receipt(root / 'event_admission').relative_to(root)))
                _, costs[f'{seed}/{event}/O_HALF'] = dispatch_training(root, plan, seed, event)
                completed += 1
                retire_odd_boundary(root, seed, event)
                write_json(output / 'COSTS.json', costs)
            if seed not in boundary['evaluation_complete']:
                progress.update(seed=seed, phase='evaluation', event=4)
                write_json(output / 'STATUS.json', {**progress, **reconcile(root)})
                costs[f'{seed}/evaluation'] = run_evaluation(root, plan, seed, 'final')
                write_json(output / 'COSTS.json', costs)
        require_complete(reconcile(root))
        progress.update(status='completed', phase='finished')
    except BaseException as error:
        progress.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        progress.update(reconcile(root), wall_s=time.perf_counter() - started)
        write_json(output / 'STATUS.json', progress)
        write_json(output / 'COSTS.json', costs)
    return progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run(args.root, resume=args.resume)


if __name__ == '__main__':
    main()
