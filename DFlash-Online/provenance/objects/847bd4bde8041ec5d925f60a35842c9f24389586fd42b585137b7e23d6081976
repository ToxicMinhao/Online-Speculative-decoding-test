"""Qualify G, train one original-feedback event, then evaluate retained K and G."""
import json
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
        raise ValueError('G must finish exactly one original K-objective event')
    if receipt['features'] != ordinary['features']:
        raise ValueError('G feedback changed')
    for left, right in zip(receipt['records'], ordinary['records'], strict=True):
        for key in ('record_id', 'pass', 'sampled_anchors', 'optimizer_stepped', 'optimizer_step_after'):
            if left[key] != right[key]:
                raise ValueError('G presentation/anchor trajectory changed: ' + key)


def check_guard_report(report):
    work = report['work']
    expected = dict(current_response_forwards=256, h_backward_calls=256,
                    k_backward_calls=256, probe_response_forwards=18)
    if any(work[key] != value for key, value in expected.items()):
        raise ValueError('G forward/backward or local probe budget differs')
    if [row['optimizer_step'] for row in report['steps']] != list(range(1, 129)):
        raise ValueError('Require one global guard decision per optimizer step')
    expected_probes = {(step, candidate) for step in (1, 64, 128)
                       for candidate in ('ordinary_K', 'projected_G', 'norm_scaled_K')}
    if (len(report['probes']) != 9 or
            {(row['optimizer_step'], row['candidate']) for row in report['probes']} != expected_probes or
            any(len(row['responses']) != 2 for row in report['probes'])):
        raise ValueError('Require exactly the declared 18 local response probes')


def run(root):
    import torch
    from online_feedback.training import SpecForgeEventTrainer
    from online_feedback.prefix_guard import install_guard
    from online_feedback.sglang_backend import SGLangDFlashBackend
    from experiments.prefix_guard.qualify import run_qualification, prepare_main
    from experiments.prefix_guard.evaluator import evaluate

    root = Path(root)
    plan = json.loads((root / 'sealed/PLAN.json').read_text())
    manifest = json.loads(Path(plan['manifest']).read_text())
    ordinary = json.loads(Path(plan['ordinary_receipt']).read_text())
    retained_k = json.loads(Path(plan['k_receipt']).read_text())
    validate_scope(manifest, ordinary, retained_k)
    output = root / 'campaign_001'
    output.mkdir(exist_ok=False)
    started = time.perf_counter()
    progress = dict(status='running', phase='qualification', main_optimizer_steps=0,
                    main_presentations=0, new_feedback_generations=0, automatic_retry=False)
    costs = {}
    trainer = backend = guard = None
    write_json(output / 'STATUS.json', progress)
    try:
        qualification = run_qualification(plan['config'], plan['ordinary_receipt'],
                                         plan['pinned_initial'], output / 'qualification')
        if qualification['status'] != 'passed':
            raise RuntimeError('G qualification did not pass')
        costs['qualification'] = qualification
        progress.update(qualification='passed', phase='loading_G')
        write_json(output / 'STATUS.json', progress)
        torch.cuda.reset_peak_memory_stats()
        loading_started = time.perf_counter()
        trainer = SpecForgeEventTrainer(plan['config'], output / 'G', objective='osd_first_error',
                                        reduction='response_mean', ranking_lambda=0)
        initialization = prepare_main(trainer, plan['pinned_initial'], output / 'initialization')
        costs['loading_and_initialization_s'] = time.perf_counter() - loading_started
        write_json(output / 'INITIALIZATION.json', initialization)
        guard = install_guard(trainer, enabled=True, probe_steps=(1, 64, 128),
                              output_dir=output / 'guard')
        progress['phase'] = 'training_G'
        write_json(output / 'STATUS.json', progress)
        training_started = time.perf_counter()
        receipt = trainer.train_event('event_001', [Path(row['path']) for row in ordinary['features']],
                                      passes=2, presentation_schedule=ordinary)
        costs['event_wall_s'] = time.perf_counter() - training_started
        check_event(receipt, ordinary)
        check_guard_report(receipt['prefix_guard'])
        costs['event'] = receipt['costs']
        costs['objective_work'] = receipt['objective_work']
        costs['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        costs['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        progress.update(main_optimizer_steps=trainer.global_step,
                        main_presentations=trainer.presentations)
        checkpoint = receipt['checkpoint_path']
        costs['guard'] = guard.report()
        write_json(output / 'GUARD_EVIDENCE.json', costs['guard'])
        guard.close()
        trainer.close()
        write_json(output / 'TRAINER_CLOSED.json', trainer.status())
        trainer = None
        guard = None
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
            {'K': retained_k['checkpoint_path'], 'G': checkpoint}, output / 'evaluation')
        progress.update(status='completed', phase='finished')
    except BaseException as error:
        progress.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        if trainer is not None:
            progress['trainer_at_failure'] = trainer.status()
            trainer.failed = True
        raise
    finally:
        if guard is not None:
            costs['guard'] = guard.report()
            write_json(output / 'GUARD_EVIDENCE.json', costs['guard'])
            guard.close()
        if trainer is not None:
            try:
                trainer.close()
            except BaseException as error:
                progress['cleanup_error'] = repr(error)
            write_json(output / 'TRAINER_CLOSED.json', trainer.status())
        if backend is not None:
            backend.close()
        costs['actual_campaign_wall_s'] = time.perf_counter() - started
        costs['accounting'] = ('Training, guard and local probe work are inside event_wall_s; '
            'nested costs are not added twice. Qualification is discarded diagnostic work. '
            'Serving latency excludes loading/training/publication and all probes. '
            'Feedback/features are reused, with zero new collection/preparation, not zero historical cost.')
        write_json(output / 'COSTS.json', costs)
        write_json(output / 'STATUS.json', progress)


if __name__ == '__main__':
    def interrupted(signum, frame):
        raise InterruptedError('Supervisor signal ' + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    run(sys.argv[1])
