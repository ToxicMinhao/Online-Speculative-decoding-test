"""Train one KC event on retained feedback, then compare retained K and final KC."""
import json
import math
from pathlib import Path
import signal
import sys
import time
import traceback

from online_feedback.storage import write_json


def validate_scope(manifest, ordinary, retained_k):
    from experiments.osd_one_event.protocol import validate_manifest
    validate_manifest(manifest)
    expected = dict(status='completed', passes=2, optimizer_steps=128,
                    response_presentations=256, start_step=0, end_step=128)
    for receipt, objective in ((ordinary, 'osd'), (retained_k, 'osd_first_error')):
        if receipt['objective'] != objective or any(receipt[k] != v for k, v in expected.items()):
            raise ValueError('Require the completed original one-event O/K receipts')
        if len(receipt['features']) != 128 or len(receipt['records']) != 256:
            raise ValueError('Original feedback and presentation counts differ')
    if retained_k['features'] != ordinary['features']:
        raise ValueError('K must have used the original feedback pool')
    for left, right in zip(ordinary['records'], retained_k['records'], strict=True):
        for key in ('record_id', 'pass', 'sampled_anchors', 'optimizer_stepped', 'optimizer_step_after'):
            if left[key] != right[key]:
                raise ValueError('Retained schedule differs: ' + key)
    if len(manifest['eval_requests']) != 64:
        raise ValueError('Keep the original 64-request development panel')


def check_event(receipt, ordinary):
    expected = dict(status='completed', objective='osd_first_error', passes=2,
                    response_presentations=256, optimizer_steps=128, start_step=0, end_step=128)
    if any(receipt[k] != v for k, v in expected.items()):
        raise ValueError('KC must finish exactly one original K-objective event')
    if receipt['features'] != ordinary['features']:
        raise ValueError('KC feedback changed')
    for left, right in zip(receipt['records'], ordinary['records'], strict=True):
        for key in ('record_id', 'pass', 'sampled_anchors', 'optimizer_stepped', 'optimizer_step_after'):
            if left[key] != right[key]:
                raise ValueError('KC presentation/anchor trajectory changed: ' + key)



def first_window_gate(report, reference, destination):
    """Compare the event's own first window before allowing the next window."""
    from experiments.prefix_guard.qualify import persist_comparison
    keys = ('K_update_norm', 'responses', 'post_k', 'correction', 'post_c')
    actual = {key: report[key] for key in keys}
    expected = {key: reference['core'][key] for key in keys}
    persist_comparison(destination, actual, expected)


def check_correction_report(report):
    work = report['work']
    expected_counts = dict(K_backward_attempted=256, K_backward_completed=256, K_backward_failed=0,
                           Adam_attempted=128, Adam_completed=128, Adam_failed=0,
                           presentations_completed=256, windows_completed=128)
    if report['status'] != 'completed' or any(report['counts'][key] != value
                                              for key, value in expected_counts.items()):
        raise ValueError('KC must complete exactly 128 native Adam steps and 256 K backwards')
    expected = dict(original_response_forwards=256, post_k_response_forwards=256,
                    post_c_response_forwards=6)
    if any(work[key] != value for key, value in expected.items()):
        raise ValueError('KC response-forward budget must total 518')
    if (work['violation_backward_completed'] > 256 or work['direct_corrections'] > 128
            or work['violation_backward_failed'] != 0):
        raise ValueError('KC exceeded its backward or correction budget')
    if [row['step'] for row in report['steps']] != list(range(1,129)):
        raise ValueError('Require one correction decision per native K step')
    if [row['step'] for row in report['probes']] != [1,64,128]:
        raise ValueError('Only steps 1, 64 and 128 receive corrected-state scoring')


def diagnostic_json(value):
    """Preserve nonfinite failure evidence without masking the original error."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: diagnostic_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [diagnostic_json(item) for item in value]
    return value


def run(root):
    import torch
    from online_feedback.training import SpecForgeEventTrainer
    from online_feedback.prefix_correction_event import PrefixCorrectionEvent
    from online_feedback.sglang_backend import SGLangDFlashBackend
    from experiments.prefix_guard.qualify import (
        WorkLog, pin_retained_graph_runtime, require_runtime_pin, run_startup, prepare_main)
    from .evaluate import evaluate

    root = Path(root)
    plan = json.loads((root / 'sealed/PLAN.json').read_text())
    manifest = json.loads(Path(plan['manifest']).read_text())
    ordinary = json.loads(Path(plan['ordinary_receipt']).read_text())
    retained_k = json.loads(Path(plan['k_receipt']).read_text())
    validate_scope(manifest, ordinary, retained_k)
    output = root / 'campaign_001'
    output.mkdir(exist_ok=False)
    started = time.perf_counter()
    progress = dict(status='running', phase='checking_retained_diagnostic', main_optimizer_steps=0,
                    main_presentations=0, new_feedback_generations=0, automatic_retry=False,
                    new_qualification_optimizer_steps=0, new_qualification_presentations=0,
                    native_adam_steps_completed=0, native_K_presentations_completed=0,
                    main_counter_scope='trainer counters after completed wrapped train_step returns')
    costs = dict(new_qualification_optimizer_steps=0, new_qualification_presentations=0)
    trainer = backend = manager = startup_log = None
    failure = None
    write_json(output / 'STATUS.json', progress)
    try:
        reference = json.loads(Path(plan['reference_diagnostic']).read_text())
        if (reference['status'] != 'completed' or reference['completed_optimizer_steps'] != 1
                or reference['completed_presentations'] != 2):
            raise ValueError('Require the completed retained single-step correction diagnostic')
        costs['reference_diagnostic'] = dict(path=plan['reference_diagnostic'], historical_only=True,
                                             included_in_current_campaign_wall=False)
        runtime_pin = pin_retained_graph_runtime(output)
        runtime_pin.update(applied_before='disposable startup and fresh main KC construction',
                           scope='process-wide ordinary startup and main KC')
        costs['runtime_compatibility'] = runtime_pin
        write_json(output / 'RUNTIME_COMPATIBILITY.json', runtime_pin)
        progress.update(qualification='retained_diagnostic_checked', phase='startup')
        write_json(output / 'STATUS.json', progress)

        # Warm compiled process caches in a disposable ordinary runtime. No Adam
        # step or correction runs here; the subsequent main trainer is constructed fresh.
        startup_log = WorkLog(output / 'startup')
        startup_log.trial = 'main_process_startup'
        pinned = torch.load(plan['pinned_initial'], map_location='cpu', weights_only=False, mmap=True)
        require_runtime_pin()
        costs['startup'] = run_startup(plan['config'], ordinary, pinned, startup_log, output / 'startup')
        del pinned
        expected_startup = dict(status='passed', response_evaluations=2, backwards=2,
                                optimizer_steps=0, exact_initial_state_restored=True,
                                separate_disposable_runtime=True)
        if any(costs['startup'][key] != value for key, value in expected_startup.items()):
            raise ValueError('Main process startup must restore exactly without optimizer steps')
        costs['startup_work_counts'] = dict(startup_log.counts)
        progress['phase'] = 'loading_KC'
        write_json(output / 'STATUS.json', progress)
        torch.cuda.reset_peak_memory_stats()
        loading_started = time.perf_counter()
        require_runtime_pin()
        trainer = SpecForgeEventTrainer(plan['config'], output / 'KC', objective='osd_first_error',
                                        reduction='response_mean', ranking_lambda=0)
        initialization = prepare_main(trainer, plan['pinned_initial'], output / 'initialization')
        require_runtime_pin()
        costs['loading_and_initialization_s'] = time.perf_counter() - loading_started
        write_json(output / 'INITIALIZATION.json', initialization)
        manager = PrefixCorrectionEvent(trainer, probe_steps=(1, 64, 128),
            output_dir=output / 'correction', first_window_check=lambda report:
                first_window_gate(report, reference, output / 'FIRST_WINDOW_REPRODUCTION.json'))
        manager.__enter__()
        progress['phase'] = 'training_KC'
        write_json(output / 'STATUS.json', progress)
        training_started = time.perf_counter()
        receipt = trainer.train_event('event_001', [Path(row['path']) for row in ordinary['features']],
                                      passes=2, presentation_schedule=ordinary)
        costs['event_wall_s'] = time.perf_counter() - training_started
        check_event(receipt, ordinary)
        manager.close()
        correction_report = manager.report()
        check_correction_report(correction_report)
        require_runtime_pin()
        costs['event'] = receipt['costs']
        costs['objective_work'] = receipt['objective_work']
        costs['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        costs['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        progress.update(main_optimizer_steps=trainer.global_step,
                        main_presentations=trainer.presentations)
        checkpoint = receipt['checkpoint_path']
        costs['correction'] = correction_report
        write_json(output / 'CORRECTION_EVIDENCE.json', correction_report)
        trainer.close()
        write_json(output / 'TRAINER_CLOSED.json', trainer.status())
        trainer = None
        manager = None
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
        progress['evaluation'] = evaluate(backend, manifest['eval_requests'],
            {'K': retained_k['checkpoint_path'], 'KC': checkpoint}, output / 'evaluation')
        progress.update(status='completed', phase='finished')
    except BaseException as error:
        failure = error
        progress.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        if trainer is not None:
            progress['trainer_at_failure'] = trainer.status()
            progress.update(main_optimizer_steps=trainer.global_step,
                            main_presentations=trainer.presentations)
            trainer.failed = True
        raise
    finally:
        # Preserve the original failure if cleanup also fails. The manager keeps
        # compact diagnostics after close, even though its heavy model refs clear.
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
            costs['correction'] = manager.report()
            write_json(output / 'CORRECTION_EVIDENCE.json', diagnostic_json(costs['correction']))
            cleanup(manager.close, 'close_correction')
        if 'correction' in costs:
            native_counts = costs['correction']['counts']
            progress.update(native_adam_steps_completed=native_counts['Adam_completed'],
                            native_K_presentations_completed=native_counts['presentations_completed'])
        if trainer is not None:
            cleanup(trainer.close, 'close_trainer')
            write_json(output / 'TRAINER_CLOSED.json', trainer.status())
        if backend is not None:
            cleanup(backend.close, 'close_backend')
        costs['actual_campaign_wall_s'] = time.perf_counter() - started
        costs['accounting'] = ('The retained single-step diagnostic is historical and excluded from current campaign wall time. '
            'Two new disposable startup forward/backward evaluations use zero optimizer steps. '
            'K training, intrinsic corrections and six corrected-state rechecks are inside event_wall_s; nested costs are not added twice. '
            'Serving latency excludes loading/training/publication and all probes. '
            'Feedback/features are reused, with zero new collection/preparation, not zero historical cost.')
        write_json(output / 'COSTS.json', diagnostic_json(costs))
        write_json(output / 'STATUS.json', progress)
        if failure is None and progress.get('cleanup_errors'):
            raise RuntimeError('Campaign cleanup failed; see STATUS.json')


if __name__ == '__main__':
    def interrupted(signum, frame):
        raise InterruptedError('Supervisor signal ' + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    run(sys.argv[1])
