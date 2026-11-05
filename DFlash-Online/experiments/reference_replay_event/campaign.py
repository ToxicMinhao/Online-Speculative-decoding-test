"""One fresh ordinary-K reference-replay event, then two fixed KR/KRC panels."""
import json
import math
from pathlib import Path
import signal
import sys
import time
import traceback

from online_feedback.storage import write_json
from .evaluation_protocol import validate_plan

FIRST_A_FIELDS = ('record_id', 'pass', 'sampled_anchors', 'optimizer_stepped',
                  'optimizer_step_after', 'learning_rate', 'metrics', 'metrics_scope')


def validate_retained(ordinary, retained_k):
    expected = dict(status='completed', passes=2, optimizer_steps=128,
                    response_presentations=256, start_step=0, end_step=128)
    for receipt, objective in ((ordinary, 'osd'), (retained_k, 'osd_first_error')):
        if receipt['objective'] != objective or any(receipt[k] != v for k, v in expected.items()):
            raise ValueError('Require original completed one-event O/K receipts')
        if len(receipt['features']) != 128 or len(receipt['records']) != 256:
            raise ValueError('Keep original 128 responses and 256 retained presentations')
    if ordinary['features'] != retained_k['features']:
        raise ValueError('Original K and O feedback differ')
    for left, right in zip(ordinary['records'], retained_k['records'], strict=True):
        for key in ('record_id', 'pass', 'sampled_anchors', 'optimizer_stepped', 'optimizer_step_after'):
            if left[key] != right[key]:
                raise ValueError('Original K/O schedule differs: ' + key)


def first_update_gate(rows, retained_k, destination):
    """Reuse the two real first-A forwards; no post-update tensor claim."""
    from experiments.prefix_guard.qualify import persist_comparison
    if len(rows) != 2:
        raise ValueError('First-A gate requires exactly two presentation records')
    select = lambda values: [{key: row[key] for key in FIRST_A_FIELDS} for row in values]
    report = persist_comparison(destination, select(rows), select(retained_k['records'][:2]))
    report['scope'] = ('Exact first-A metrics, anchors, learning rate and counters; fresh weights/Adam/RNG '
                       'are checked separately. No retained post-update tensor equivalence is claimed.')
    write_json(destination, report)
    return report


def diagnostic_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: diagnostic_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [diagnostic_json(item) for item in value]
    return value


def unattempted_panel():
    return dict(status='unattempted', planned_generations=384, attempted_generations=0,
                returned_generations=0, completed_valid_generations=0, invalid_generations=0,
                unknown_generation_attempts=0, unattempted_generations=384)


def evaluation_totals(panels):
    keys = ('planned_generations', 'attempted_generations', 'returned_generations',
            'completed_valid_generations', 'invalid_generations',
            'unknown_generation_attempts', 'unattempted_generations')
    return {key: sum(panel.get(key, 0) for panel in panels.values()) for key in keys}


def evaluate_panels(backend, panels, checkpoints, root, progress):
    """Only development's first eight requests form the included native gate."""
    from .evaluate import evaluate
    root = Path(root)
    for index, panel in enumerate(panels):
        name = panel['name']
        destination = root / 'evaluation' / name
        try:
            progress['panels'][name] = evaluate(
                backend, panel['requests'], checkpoints, destination,
                retained_krc_counts=panel['retained_krc_counts'],
                qualification_requests=8 if index == 0 else 0)
        finally:
            status_path = destination / 'STATUS.json'
            if status_path.exists():
                progress['panels'][name] = json.loads(status_path.read_text())
            progress['evaluation_totals'] = evaluation_totals(progress['panels'])
            write_json(root / 'STATUS.json', progress)


def run(root):
    import torch
    from online_feedback.reference_replay_event import ReferenceReplayEventTrainer
    from online_feedback.sglang_backend import SGLangDFlashBackend
    from experiments.prefix_guard.qualify import (
        WorkLog, pin_retained_graph_runtime, require_runtime_pin, run_startup, prepare_main)
    from .protocol import validate_replay_schedule, check_training

    root = Path(root)
    plan = json.loads((root / 'sealed/PLAN.json').read_text())
    plan_check = validate_plan(plan)
    ordinary = json.loads(Path(plan['ordinary_receipt']).read_text())
    retained_k = json.loads(Path(plan['k_receipt']).read_text())
    validate_retained(ordinary, retained_k)
    schedule = json.loads(Path(plan['replay_schedule']).read_text())
    schedule_check = validate_replay_schedule(schedule, ordinary,
        evaluation_ids=[r['request_id'] for panel in plan['panels'] for r in panel['requests']])
    output = root / 'campaign_001'
    output.mkdir(exist_ok=False)
    started = time.perf_counter()
    progress = dict(status='running', phase='startup', main_optimizer_steps=0,
        main_presentations=0, outer_windows_completed=0, new_feedback_generations=0,
        new_qualification_optimizer_steps=0, new_qualification_presentations=0,
        automatic_retry=False, panels={panel['name']: unattempted_panel() for panel in plan['panels']})
    progress['evaluation_totals'] = evaluation_totals(progress['panels'])
    costs = dict(new_qualification_optimizer_steps=0, new_qualification_presentations=0)
    trainer = backend = startup_log = None
    failure = None
    write_json(output / 'STATUS.json', progress)
    write_json(output / 'PLAN_CHECK.json', plan_check)
    write_json(output / 'REPLAY_SCHEDULE_CHECK.json', schedule_check)
    try:
        costs['runtime_compatibility'] = pin_retained_graph_runtime(output)
        startup_log = WorkLog(output / 'startup')
        startup_log.trial = 'main_process_startup'
        pinned = torch.load(plan['pinned_initial'], map_location='cpu', weights_only=False, mmap=True)
        require_runtime_pin()
        costs['startup'] = run_startup(plan['config'], ordinary, pinned, startup_log, output / 'startup')
        del pinned
        expected = dict(status='passed', response_evaluations=2, backwards=2, optimizer_steps=0,
                        exact_initial_state_restored=True, separate_disposable_runtime=True)
        if any(costs['startup'][key] != value for key, value in expected.items()):
            raise ValueError('Startup requires two disposable forward/backwards and zero Adam steps')
        costs['startup_work_counts'] = dict(startup_log.counts)
        progress['phase'] = 'loading_KR'
        write_json(output / 'STATUS.json', progress)
        torch.cuda.reset_peak_memory_stats()
        loading_started = time.perf_counter()
        require_runtime_pin()
        trainer = ReferenceReplayEventTrainer(plan['config'], output / 'KR',
            objective='osd_first_error', reduction='response_mean', ranking_lambda=0)
        initialization = prepare_main(trainer, plan['pinned_initial'], output / 'initialization')
        require_runtime_pin()
        costs['loading_and_initialization_s'] = time.perf_counter() - loading_started
        write_json(output / 'INITIALIZATION.json', initialization)
        progress['phase'] = 'training_KR'
        write_json(output / 'STATUS.json', progress)
        training_started = time.perf_counter()
        receipt = trainer.train_replay_event('event_001',
            [Path(row['path']) for row in ordinary['features']], schedule,
            first_update_check=lambda rows: first_update_gate(
                rows, retained_k, output / 'FIRST_A_REPRODUCTION.json'))
        costs['event_wall_s'] = time.perf_counter() - training_started
        training_check = check_training(receipt, schedule)
        write_json(output / 'TRAINING_CHECK.json', training_check)
        require_runtime_pin()
        costs['event'] = receipt['costs']
        costs['objective_work'] = receipt['objective_work']
        costs['reference_replay'] = receipt['reference_replay']
        costs['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        costs['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        progress.update(main_optimizer_steps=trainer.global_step, main_presentations=trainer.presentations,
                        outer_windows_completed=receipt['reference_replay']['counts']['outer_windows_completed'])
        checkpoint = receipt['checkpoint_path']
        trainer.close()
        write_json(output / 'TRAINER_CLOSED.json', trainer.status())
        trainer = None
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        progress['phase'] = 'loading_native'
        write_json(output / 'STATUS.json', progress)
        backend = SGLangDFlashBackend(plan['target'], plan['draft'],
            worker_python=plan['worker_python'], output_root=output / 'native',
            serving_gpu='0', context_limit=4096, block_size=16)
        costs['backend_loading_s'] = backend.loading_s
        progress['phase'] = 'evaluation'
        write_json(output / 'STATUS.json', progress)
        evaluate_panels(backend, plan['panels'],
                        {'KR': checkpoint, 'KRC': plan['retained_krc_checkpoint']}, output, progress)
        progress.update(status='completed', phase='finished')
    except BaseException as error:
        failure = error
        progress.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        if trainer is not None:
            progress['trainer_at_failure'] = trainer.status()
            progress.update(main_optimizer_steps=trainer.global_step, main_presentations=trainer.presentations)
            trainer.failed = True
        raise
    finally:
        def cleanup(action, label):
            try:
                action()
            except BaseException as error:
                progress.setdefault('cleanup_errors', []).append(dict(operation=label, error=repr(error)))
                if failure is not None:
                    failure.add_note(label + ': ' + repr(error))
                else:
                    progress.update(status='failed', phase='cleanup', error=repr(error))
        if startup_log is not None:
            costs['startup_work_counts'] = dict(startup_log.counts)
        if trainer is not None:
            costs['reference_replay'] = trainer.replay_report()
            cleanup(trainer.close, 'close_trainer')
            write_json(output / 'TRAINER_CLOSED.json', trainer.status())
        if 'reference_replay' in costs:
            counts = costs['reference_replay']['counts']
            progress['actual_Adam_steps_completed'] = counts.get('Adam_completed', 0)
            progress['actual_response_forwards_completed'] = counts.get('response_forwards_completed', 0)
        if backend is not None:
            cleanup(backend.close, 'close_backend')
        costs['actual_campaign_wall_s'] = time.perf_counter() - started
        costs['accounting'] = (
            'A/R train_step times are nested in event wall time; startup is two separate forward/backwards with zero Adam. '
            'No new feedback or teacher preparation is performed; prior preparation cost is reused, not erased. '
            'KR changes optimizer steps/state and exposure: practical replay control, not equal-compute isolation. '
            'Historical KRC training time is not a fresh matched cost measurement. '
            'Serving is timed only after training closes. Native generation and cache maintenance are nested in complete request latency. '
            'Publication, weight checks, backend loading and benchmark repeats remain separate from deployment adaptation cost.')
        progress['evaluation_totals'] = evaluation_totals(progress['panels'])
        write_json(output / 'COSTS.json', diagnostic_json(costs))
        write_json(output / 'STATUS.json', progress)
        if failure is None and progress.get('cleanup_errors'):
            raise RuntimeError('Campaign cleanup failed; see STATUS.json')


if __name__ == '__main__':
    def interrupted(signum, frame):
        raise InterruptedError('Supervisor signal ' + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    run(sys.argv[1])
