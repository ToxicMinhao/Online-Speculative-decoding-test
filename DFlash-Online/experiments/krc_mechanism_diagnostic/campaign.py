"""Run exactly two independent diagnostic replays; no serving or follow-up arm."""
import gc
import json
from pathlib import Path
import signal
import sys
import time
import traceback

from experiments.krc_continuation.campaign import diagnostic_json
from experiments.krc_events_3_4.campaign import artifact_identity
from online_feedback.storage import write_json
from .protocol import (BUDGET, validate_plan, reference_schedule, validate_window_plan,
                       validate_original_report, validate_diagnostic_report)


def read(path):
    return json.loads(Path(path).read_text())


def controller_types():
    from online_feedback.krc_continuation import KRCContinuationTrainer, KRCContinuationEvent
    from online_feedback.krc_mechanism import MechanismDiagnosticMixin
    from online_feedback.krc_mechanism_replay import KRCMechanismEvent4Trainer, KRCMechanismEvent4Base

    class Event2(MechanismDiagnosticMixin, KRCContinuationEvent):
        pass

    class Event4(MechanismDiagnosticMixin, KRCMechanismEvent4Base):
        pass

    return {2: (KRCContinuationTrainer, Event2), 4: (KRCMechanismEvent4Trainer, Event4)}


def run(root):
    import torch
    from experiments.ok_continuation.resume import load_snapshot, verify_live_resume, validate_event_receipt
    from experiments.prefix_guard.qualify import WorkLog, pin_retained_graph_runtime, require_runtime_pin, run_startup
    from .state import compare_endpoints, verify_live_endpoint, retire_verified_replay

    root = Path(root)
    plan = read(root / 'sealed/PLAN.json')
    sealed_files = read(root / 'sealed/SEAL.json')['files']
    plan_check = validate_plan(plan)
    types = controller_types()
    # All checks precede creation of the non-retryable campaign destination.
    episode_inputs = []
    for item in plan['episodes']:
        number = item['event_number']
        ordinary = read(item['ordinary_receipt'])
        validate_event_receipt(read(item['parent_receipt']), event_number=number - 1,
                               objective='osd_first_error')
        validate_event_receipt(read(item['endpoint_receipt']), event_number=number,
                               objective='osd_first_error', reference_schedule=ordinary)
        schedule = read(item['reference_schedule'])
        if schedule != reference_schedule(ordinary, number, evaluation_ids=plan['evaluation_ids']):
            raise ValueError('Sealed reference schedule changed')
        windows = read(item['window_plan'])
        check = validate_window_plan(windows, ordinary, number, evaluation_ids=plan['evaluation_ids'])
        episode_inputs.append((item, ordinary, schedule, windows, check))
    output = root / 'campaign_001'
    output.mkdir(exist_ok=False)
    progress = dict(status='running', phase='runtime_preflight', episodes={},
        actual_new_Adam_steps_completed=0, actual_K_presentations_completed=0,
        serving_generations=0, feedback_generations=0, startup_Adam_steps=0,
        automatic_retry=False, execution_source=plan['execution_source'])
    costs = dict(episodes={}, execution_source=plan['execution_source'])
    reports = {}
    started = time.perf_counter()
    trainer = manager = startup_log = None
    active_number = None
    failure = None
    write_json(output / 'PLAN_CHECK.json', plan_check)
    write_json(output / 'STATUS.json', progress)
    try:
        costs['runtime_compatibility'] = pin_retained_graph_runtime(output)
        for item, ordinary, schedule, windows, check in episode_inputs:
            number = active_number = item['event_number']
            event_id = f'event_{number:03d}'
            event_root = output / event_id
            event_root.mkdir()
            progress['phase'] = 'restore_preflight_' + event_id
            write_json(output / 'STATUS.json', progress)
            write_json(event_root / 'WINDOW_PLAN_CHECK.json', check)
            episode_cost = costs['episodes'][event_id] = {}
            episode_started = time.perf_counter()
            state, state_check = load_snapshot(item['parent_state'], item['parent_checkpoint'],
                objective='osd_first_error', config_text=Path(plan['config']).read_text(),
                completed_event=number - 1)
            write_json(event_root / 'PARENT_STATE_CHECK.json', state_check)
            episode_cost['parent_validation_s'] = state_check['wall_s']
            startup_log = WorkLog(event_root / 'startup')
            startup_log.trial = f'mechanism_event_{number}_disposable_startup'
            pinned = torch.load(plan['pinned_initial'], map_location='cpu', weights_only=False, mmap=True)
            require_runtime_pin()
            startup = run_startup(plan['config'], read(plan['startup_ordinary_receipt']),
                                  pinned, startup_log, event_root / 'startup')
            del pinned
            expected = dict(status='passed', response_evaluations=2, backwards=2, optimizer_steps=0,
                            exact_initial_state_restored=True, separate_disposable_runtime=True)
            if any(startup.get(key) != value for key, value in expected.items()):
                raise ValueError('Per-episode startup requires two disposable forward/backwards and zero Adam')
            episode_cost['startup'] = startup
            episode_cost['startup_work_counts'] = dict(startup_log.counts)
            startup_log = None
            trainer_type, manager_type = types[number]
            require_runtime_pin()
            loading_started = time.perf_counter()
            trainer = trainer_type(plan['config'], event_root / 'replay', resume_state=item['parent_state'])
            verify_live_resume(trainer, state, receipt_path=event_root / 'LIVE_RESUME.json')
            del state
            episode_cost['loading_and_live_resume_s'] = time.perf_counter() - loading_started
            torch.cuda.reset_peak_memory_stats()
            manager = manager_type(trainer, schedule=schedule,
                feature_paths={row['record_id']: Path(row['path']) for row in ordinary['features']},
                output_dir=event_root / 'original_correction', window_plan=windows,
                mechanism_dir=event_root / 'mechanism')
            progress['phase'] = 'replaying_' + event_id
            write_json(output / 'STATUS.json', progress)
            with manager:
                receipt = trainer.train_event(event_id, [Path(row['path']) for row in ordinary['features']],
                    passes=2, presentation_schedule=ordinary)
            report = reports[number] = manager.report()
            write_json(event_root / 'CORRECTION_EVIDENCE.json', diagnostic_json(report))
            write_json(event_root / 'ORIGINAL_WORK_CHECK.json', validate_original_report(report, number))
            write_json(event_root / 'MECHANISM_WORK_CHECK.json', validate_diagnostic_report(report))
            write_json(event_root / 'TRAINING_CHECK.json', validate_event_receipt(receipt,
                event_number=number, objective='osd_first_error', reference_schedule=ordinary,
                trainer_status=trainer.status()))
            episode_cost.update(event=receipt['costs'], objective_work=receipt['objective_work'],
                correction=report, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved())
            progress['phase'] = 'endpoint_check_' + event_id
            write_json(output / 'STATUS.json', progress)
            saved, saved_check = load_snapshot(receipt['full_state_path'], receipt['checkpoint_path'],
                objective='osd_first_error', config_text=Path(plan['config']).read_text(), completed_event=number)
            write_json(event_root / 'SAVED_STATE_CHECK.json', saved_check)
            verify_live_endpoint(trainer, saved, event_root / 'LIVE_ENDPOINT.json')
            trainer.close()
            write_json(event_root / 'TRAINER_CLOSED.json', trainer.status())
            trainer = manager = None
            gc.collect()
            torch.cuda.empty_cache()
            original, original_check = load_snapshot(item['endpoint_state'], item['endpoint_checkpoint'],
                objective='osd_first_error', config_text=Path(plan['config']).read_text(), completed_event=number)
            write_json(event_root / 'ORIGINAL_ENDPOINT_CHECK.json', original_check)
            comparison = compare_endpoints(saved, original, event_number=number,
                config_text=Path(plan['config']).read_text(), receipt_path=event_root / 'ENDPOINT_EQUALITY.json')
            episode_cost['endpoint_comparison_s'] = comparison['wall_s']
            episode_cost['episode_wall_s'] = time.perf_counter() - episode_started
            identities = [artifact_identity(receipt['full_state_path'])] + [artifact_identity(path)
                for path in sorted(Path(receipt['checkpoint_path']).iterdir()) if path.is_file()]
            write_json(event_root / 'REPLAY_STATE_IDENTITIES.json', identities)
            if number == 2:
                retention = retire_verified_replay(event_root, receipt, item, saved,
                                                   comparison, sealed_files)
                episode_cost['verified_replay_retirement_s'] = retention['wall_s']
            del saved, original
            progress['episodes'][event_id] = dict(status='completed', independent_parent=item['parent_state'],
                exact_decoded_endpoint=True, full_state_path=receipt['full_state_path'],
                checkpoint_path=receipt['checkpoint_path'], full_state_retained=number != 2,
                original_endpoint_state=item['endpoint_state'])
            progress.update(actual_new_Adam_steps_completed=sum(r['counts']['Adam_completed'] for r in reports.values()),
                actual_K_presentations_completed=sum(r['counts']['presentations_completed'] for r in reports.values()))
            write_json(output / 'STATUS.json', progress)
            gc.collect()
        progress.update(status='completed', phase='finished')
    except BaseException as error:
        failure = error
        progress.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        if trainer is not None:
            progress['trainer_at_failure'] = trainer.status()
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
        if manager is not None:
            cleanup(manager.close, 'close_diagnostic_controller')
            reports[active_number] = manager.report()
            write_json(manager.output_dir.parent / 'CORRECTION_EVIDENCE.json',
                       diagnostic_json(reports[active_number]))
        if trainer is not None:
            cleanup(trainer.close, 'close_trainer')
        if startup_log is not None:
            costs['interrupted_startup_work'] = dict(startup_log.counts)
        if reports:
            progress.update(actual_new_Adam_steps_completed=sum(r['counts']['Adam_completed'] for r in reports.values()),
                actual_K_presentations_completed=sum(r['counts']['presentations_completed'] for r in reports.values()))
        costs['actual_campaign_wall_s'] = time.perf_counter() - started
        costs['work_ledger'] = work_ledger(reports, costs)
        costs['accounting'] = ('Two independently restored original episodes only; no serving or feedback. '
            'Original R_FULL continues after every sampled window, irrespective of candidate measurements. '
            'Diagnostic and intrinsic component timers nest in event time; startup is separate. '
            'Head and layer recomputation counters are internal work, separate from response equivalents.')
        write_json(output / 'COSTS.json', diagnostic_json(costs))
        write_json(output / 'STATUS.json', progress)
        if failure is None and progress.get('cleanup_errors'):
            raise RuntimeError('Cleanup failed; see STATUS.json')


def work_ledger(reports, costs):
    baseline_forwards = baseline_backwards = diagnostic_forwards = diagnostic_backwards = 0
    for report in reports.values():
        work = report['work']
        baseline_forwards += sum(work.get(key, 0) for key in ('original_response_forwards',
            'pre_reference_response_forwards', 'post_k_reference_response_forwards'))
        baseline_backwards += report['counts']['K_backward_completed'] + work.get('violation_backward_completed', 0)
        extra = report.get('mechanism', {}).get('work', {})
        diagnostic_forwards += sum(extra.get(key, 0) for key in ('pre_capture_response_forwards',
            'candidate_response_forwards', 'd_gradient_response_forwards'))
        diagnostic_backwards += extra.get('d_gradient_backwards', 0)
    startups = [row.get('startup', {}) for row in costs.get('episodes', {}).values()]
    result = dict(baseline_response_forwards=baseline_forwards,
        diagnostic_response_forwards=diagnostic_forwards,
        core_response_forwards=baseline_forwards + diagnostic_forwards,
        core_response_backwards=baseline_backwards + diagnostic_backwards,
        startup_response_forwards=sum(row.get('response_evaluations', 0) for row in startups),
        startup_response_backwards=sum(row.get('backwards', 0) for row in startups),
        startup_Adam_steps=sum(row.get('optimizer_steps', 0) for row in startups),
        ordinary_Adam_steps=sum(r['counts']['Adam_completed'] for r in reports.values()),
        ordinary_presentations=sum(r['counts']['presentations_completed'] for r in reports.values()),
        serving_generations=0, feedback_generations=0, authorized_budget=BUDGET)
    result['within_budget'] = (result['core_response_forwards'] <= 2136
        and result['core_response_backwards'] <= 1064 and result['ordinary_Adam_steps'] <= 256
        and result['startup_response_forwards'] <= 4 and result['startup_response_backwards'] <= 4
        and result['startup_Adam_steps'] == 0)
    return result


if __name__ == '__main__':
    def interrupted(signum, frame):
        raise InterruptedError('Supervisor signal ' + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    run(sys.argv[1])
