"""Continue qualified KRC2 through events 3/4, then evaluate O4/K4/KRC4."""
import hashlib
import json
from pathlib import Path
import signal
import sys
import time
import traceback

from online_feedback.storage import write_json
from experiments.krc_continuation.campaign import diagnostic_json, evaluation_totals
from .evaluation_protocol import validate_plan, GENERATIONS


def read(path):
    return json.loads(Path(path).read_text())


def artifact_identity(path):
    path = Path(path)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest)


def unattempted_panel():
    return dict(status='unattempted', planned_generations=GENERATIONS,
        attempted_generations=0, returned_generations=0, completed_valid_generations=0,
        invalid_generations=0, unknown_generation_attempts=0, unattempted_generations=GENERATIONS)


def evaluate_panels(backend, panels, checkpoints, root, progress):
    from .evaluate import evaluate
    for index, panel in enumerate(panels):
        destination = root / 'evaluation' / panel['name']
        try:
            progress['panels'][panel['name']] = evaluate(backend, panel['requests'], checkpoints,
                destination, expected_control_counts=panel['expected_control_counts'],
                historical_counts=panel['historical_counts'], qualification_requests=8 if index == 0 else 0)
        finally:
            status_path = destination / 'STATUS.json'
            if status_path.exists():
                progress['panels'][panel['name']] = read(status_path)
            progress['evaluation_totals'] = evaluation_totals(progress['panels'])
            write_json(root / 'STATUS.json', progress)


def verify_saved_boundary(trainer, state, receipt_path):
    """Saving/exporting must preserve the live trajectory used by the next event.

    Unlike initial restoration, a completed event may retain its last anchor
    observation for reporting. Check scientific state without clearing it.
    """
    from experiments.ok_continuation.resume import differences
    from online_feedback.rng import capture_rng
    actual = dict(weights=dict(trainer.bundle.draft_model.state_dict()),
        optimizer=trainer.backend.optimizer.state_dict(), rng=capture_rng(), events=trainer.events,
        global_step=trainer.global_step, presentations=trainer.presentations,
        contract=trainer.contract, config=trainer.config_text, horizon=trainer.horizon,
        accumulation=trainer.accumulation, reduction=trainer.reduction)
    expected = dict(weights=state['draft_state_dict'], optimizer=state['backend']['optimizer'],
        rng=state['rng'], events=state['events'], global_step=state['global_step'],
        presentations=state['presentations'], contract=state['algorithm_contract'], config=state['config_text'],
        horizon=state['horizon'], accumulation=state['accumulation_steps'], reduction=state['reduction'])
    mismatch = differences(actual, expected)
    if (trainer.core.accumulation_remainder or trainer.core._micro != trainer.presentations
            or trainer.core._ratio_totals or trainer.core._sum_totals
            or trainer._presentation_schedule is not None or trainer._krc_late_event is not None
            or trainer.bundle.model.reference is not None or trainer.bundle.model.ranking_enabled
            or any(p.grad is not None for p in trainer.bundle.draft_model.parameters())
            or not trainer.bundle.model.training or trainer.failed or trainer.closed):
        mismatch.append('live_training_boundary')
    result = dict(status='passed' if not mismatch else 'failed', exact_boundary=not mismatch,
        mismatches=mismatch, global_step=trainer.global_step, presentations=trainer.presentations,
        no_rng_restore=True, extra_model_passes=0, extra_optimizer_steps=0)
    write_json(receipt_path, result)
    if mismatch:
        raise ValueError('State/save boundary changed the trajectory: ' + str(mismatch[:8]))
    return result


def run(root):
    import torch
    from online_feedback.krc_late_continuation import KRCLateContinuationTrainer, KRCLateContinuationEvent
    from online_feedback.sglang_backend import SGLangDFlashBackend
    from experiments.ok_continuation.resume import load_snapshot, verify_live_resume, validate_event_receipt
    from experiments.prefix_guard.qualify import WorkLog, pin_retained_graph_runtime, require_runtime_pin, run_startup
    from .protocol import validate_retained, validate_schedule, check_correction_report

    root = Path(root)
    plan = read(root / 'sealed/PLAN.json')
    plan_check = validate_plan(plan)
    entries = [(item, read(item['ordinary_receipt']), read(item['k_receipt'])) for item in plan['events']]
    retained = validate_retained(read(plan['krc2_receipt']),
        [(item['event_number'], ordinary, k) for item, ordinary, k in entries])
    evaluation_ids = [r['request_id'] for panel in plan['panels'] for r in panel['requests']]
    schedules = {item['event_number']: read(item['reference_schedule']) for item, _, _ in entries}
    schedule_checks = [validate_schedule(schedules[item['event_number']], ordinary,
        item['event_number'], evaluation_ids=evaluation_ids) for item, ordinary, _ in entries]
    output = root / 'campaign_001'
    output.mkdir(exist_ok=False)
    started = time.perf_counter()
    progress = dict(status='running', phase='resume_preflight', global_optimizer_step=256,
        new_optimizer_steps=0, new_presentations=0, new_feedback_generations=0,
        new_qualification_optimizer_steps=0, automatic_retry=False, events={},
        panels={panel['name']: unattempted_panel() for panel in plan['panels']},
        execution_source=plan['execution_source'])
    progress['evaluation_totals'] = evaluation_totals(progress['panels'])
    costs = dict(events={}, new_qualification_optimizer_steps=0,
                 execution_source=plan['execution_source'])
    retained_states = {}
    trainer = backend = manager = startup_log = None
    failure = None
    write_json(output / 'STATUS.json', progress)
    write_json(output / 'PLAN_CHECK.json', plan_check)
    write_json(output / 'RETAINED_CHECK.json', retained)
    write_json(output / 'REFERENCE_SCHEDULE_CHECK.json', schedule_checks)
    try:
        costs['runtime_compatibility'] = pin_retained_graph_runtime(output)
        state, snapshot_check = load_snapshot(plan['krc2_state'], plan['krc2_checkpoint'],
            objective='osd_first_error', config_text=Path(plan['config']).read_text(), completed_event=2)
        write_json(output / 'RESUME_SNAPSHOT.json', snapshot_check)
        costs['snapshot_validation_s'] = snapshot_check['wall_s']
        progress['phase'] = 'startup'
        write_json(output / 'STATUS.json', progress)
        startup_log = WorkLog(output / 'startup')
        startup_log.trial = 'continuation_process_startup'
        pinned = torch.load(plan['pinned_initial'], map_location='cpu', weights_only=False, mmap=True)
        require_runtime_pin()
        costs['startup'] = run_startup(plan['config'], read(plan['startup_ordinary_receipt']),
            pinned, startup_log, output / 'startup')
        del pinned
        expected = dict(status='passed', response_evaluations=2, backwards=2, optimizer_steps=0,
                        exact_initial_state_restored=True, separate_disposable_runtime=True)
        if any(costs['startup'][k] != v for k, v in expected.items()):
            raise ValueError('Require exactly two disposable forwards/backwards and zero Adam steps')
        progress['phase'] = 'resuming_KRC2'
        write_json(output / 'STATUS.json', progress)
        loading_started = time.perf_counter()
        trainer = KRCLateContinuationTrainer(plan['config'], output / 'KRC', resume_state=plan['krc2_state'])
        verify_live_resume(trainer, state, receipt_path=output / 'LIVE_RESUME.json')
        del state
        costs['loading_and_live_resume_s'] = time.perf_counter() - loading_started
        for item, ordinary, _ in entries:
            number = item['event_number']
            event_id = f'event_{number:03d}'
            event_root = output / 'events' / event_id
            progress['phase'] = 'training_' + event_id
            write_json(output / 'STATUS.json', progress)
            require_runtime_pin()
            torch.cuda.reset_peak_memory_stats()
            event_started = time.perf_counter()
            manager = KRCLateContinuationEvent(trainer, schedule=schedules[number],
                feature_paths={row['record_id']: Path(row['path']) for row in ordinary['features']},
                output_dir=event_root / 'correction')
            with manager:
                receipt = trainer.train_event(event_id, [Path(row['path']) for row in ordinary['features']],
                    passes=2, presentation_schedule=ordinary)
            report = manager.report()
            write_json(event_root / 'CORRECTION_EVIDENCE.json', diagnostic_json(report))
            write_json(event_root / 'CORRECTION_CHECK.json', check_correction_report(report, number))
            write_json(event_root / 'TRAINING_CHECK.json', validate_event_receipt(receipt,
                event_number=number, objective='osd_first_error', reference_schedule=ordinary,
                trainer_status=trainer.status()))
            costs['events'][event_id] = dict(event_wall_s=time.perf_counter() - event_started,
                event=receipt['costs'], objective_work=receipt['objective_work'], correction=report,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved())
            # Verify complete state/export and the still-live next-event state;
            # this reads state only, without restoring RNG or rebuilding trainer.
            saved, check = load_snapshot(receipt['full_state_path'], receipt['checkpoint_path'],
                objective='osd_first_error', config_text=Path(plan['config']).read_text(), completed_event=number)
            write_json(event_root / 'SAVED_STATE_CHECK.json', check)
            verify_saved_boundary(trainer, saved, event_root / 'LIVE_BOUNDARY.json')
            del saved
            costs['events'][event_id]['saved_state_validation_s'] = check['wall_s']
            identity_started = time.perf_counter()
            retained_states[event_id] = [artifact_identity(receipt['full_state_path'])] + [
                artifact_identity(path) for path in sorted(Path(receipt['checkpoint_path']).iterdir())
                if path.is_file()]
            write_json(event_root / 'STATE_IDENTITIES.json', retained_states[event_id])
            costs['events'][event_id]['artifact_identity_s'] = time.perf_counter() - identity_started
            progress['events'][event_id] = dict(status='completed', checkpoint=receipt['checkpoint_path'],
                full_state_path=receipt['full_state_path'], global_step=trainer.global_step)
            progress.update(global_optimizer_step=trainer.global_step, new_optimizer_steps=trainer.global_step-256,
                new_presentations=trainer.presentations-512)
            write_json(output / 'STATUS.json', progress)
            manager = None
        checkpoint = receipt['checkpoint_path']
        trainer.close()
        write_json(output / 'TRAINER_CLOSED.json', trainer.status())
        trainer = None
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        retention_started = time.perf_counter()
        retained_after = {event: [artifact_identity(row['path']) for row in identities]
                          for event, identities in retained_states.items()}
        retention_check = dict(status='passed' if retained_after == retained_states else 'failed',
            before=retained_states, after=retained_after,
            preserved_both_complete_states=retained_after == retained_states)
        write_json(output / 'STATE_RETENTION.json', retention_check)
        costs['final_state_retention_validation_s'] = time.perf_counter() - retention_started
        if retained_after != retained_states:
            raise ValueError('A complete KRC3/KRC4 state or export changed after retention')
        require_runtime_pin()
        progress['phase'] = 'loading_native'
        write_json(output / 'STATUS.json', progress)
        backend = SGLangDFlashBackend(plan['target'], plan['draft'], worker_python=plan['worker_python'],
            output_root=output / 'native', serving_gpu='0', context_limit=4096, block_size=16)
        costs['backend_loading_s'] = backend.loading_s
        progress['phase'] = 'evaluation'
        write_json(output / 'STATUS.json', progress)
        evaluate_panels(backend, plan['panels'], dict(O4=plan['retained_o4_checkpoint'],
            K4=plan['retained_k4_checkpoint'], KRC4=checkpoint), output, progress)
        progress.update(status='completed', phase='finished')
    except BaseException as error:
        failure = error
        progress.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        if trainer is not None:
            progress['trainer_at_failure'] = trainer.status()
            progress.update(global_optimizer_step=trainer.global_step,
                new_optimizer_steps=trainer.global_step-256, new_presentations=trainer.presentations-512)
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
        if manager is not None:
            cleanup(manager.close, 'close_correction_controller')
            report = manager.report()
            costs['interrupted_event'] = diagnostic_json(report)
            write_json(manager.output_dir.parent / 'CORRECTION_EVIDENCE.json', diagnostic_json(report))
        reports = [event['correction'] for event in costs['events'].values()]
        if manager is not None:
            # Replace any completed copy of this event; do not double-count a
            # save/export or validation failure after its final native step.
            reports = [report for report in reports if report['start_step'] != manager.start_step]
            reports.append(manager.report())
        if reports:
            progress.update(
                actual_new_Adam_steps_completed=sum(r['counts']['Adam_completed'] for r in reports),
                actual_global_Adam_step=max(r['native_global_step'] for r in reports),
                actual_K_presentations_completed=sum(r['counts']['presentations_completed'] for r in reports))
        if trainer is not None:
            cleanup(trainer.close, 'close_trainer')
            write_json(output / 'TRAINER_CLOSED.json', trainer.status())
        if backend is not None:
            cleanup(backend.close, 'close_backend')
        costs['actual_campaign_wall_s'] = time.perf_counter() - started
        costs['accounting'] = (
            'Only unchanged KRC events3/4 are trained from complete KRC2 own state. '
            '512 ordinary presentations,256 Adam steps,1536 core forwards and at most1024 backwards; '
            'optional probes disabled. Startup separate:2 disposable forward/backwards,zero Adam. '
            'States and exports retained at384 and512. Fresh three-arm serving begins after trainer closure. '
            'Native/cache timers nest within request time. Historical O/K training and serving times '
            'are not fresh matched costs or a latency learning curve. Additional reference work is not compute matched.')
        progress['evaluation_totals'] = evaluation_totals(progress['panels'])
        write_json(output / 'COSTS.json', diagnostic_json(costs))
        write_json(output / 'STATUS.json', progress)
        if failure is None and progress.get('cleanup_errors'):
            raise RuntimeError('Cleanup failed; see STATUS.json')


if __name__ == '__main__':
    def interrupted(signum, frame):
        raise InterruptedError('Supervisor signal ' + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    run(sys.argv[1])
