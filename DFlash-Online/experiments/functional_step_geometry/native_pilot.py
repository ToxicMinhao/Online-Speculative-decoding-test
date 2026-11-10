"""Native events-five/six workers behind the fixed FSG pilot controller.

Collection closes its serving process before training. Each training process
restores one complete arm boundary, trains one event, and exits before serving.
The shared pool is immutable; baseline schedules assert native draws rather
than replacing the sampler. No path here initializes released weights.
"""
import argparse
from copy import deepcopy
import importlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from online_feedback.storage import write_json
from experiments.ok_coding.campaign import WorkLedger, read_work, require_complete_work
from experiments.ok_coding.storage import identity
from .contract import ARMS, DATASETS, EVENTS, REALIZATION_R1
from .data import seal_shared_packets, verify_shared_packets
from .pilot import diagnostic_pairs, read, validate_update, pilot_authorization

DIAGNOSTIC_FACTORY = 'experiments.functional_step_geometry.pilot_diagnostics:run_window_diagnostics'
PILOT_STARTUP_BUDGET = dict(response_forwards=24, backwards=24, optimizer_steps=0)

TRAINING_WORK = dict(response_forwards=256, response_backwards=256, optimizer_steps=128)
STARTUP_WORK = dict(response_forwards=2, response_backwards=2, optimizer_steps=0)


def require_capacity(plan, receipt_path):
    from .pilot_storage import check_current_capacity
    receipt = check_current_capacity(plan['root'], plan['main_storage'])
    write_json(receipt_path, receipt)
    if receipt['status'] != 'passed':
        raise ValueError('Pilot resource admission failed before additional work')
    return receipt


def event_root(root, dataset, event, arm=None):
    base = Path(root) / 'campaign_001' / dataset / f'event_{event:03d}'
    return base / arm if arm else base


def result_path(root, dataset, event, arm):
    return event_root(root, dataset, event, arm) / 'RESULT.json'


def feedback_requests(plan, dataset, event):
    rows = read(plan['datasets'][dataset]['training_manifest'])['train_requests']
    selected = [row for row in rows if row['event_id'] == f'event_{event:03d}']
    if (dataset not in DATASETS or event not in EVENTS or len(selected) != 128
            or len({row['request_id'] for row in selected}) != 128):
        raise ValueError('Require the prescribed 128 distinct requests for this shared event')
    return selected


def retained_packets(plan, dataset, requests):
    """Reuse the two original qualification responses without recollecting them."""
    preparation = read(plan['shared_feedback_preparation'])
    rows = preparation['datasets'][dataset]
    if (preparation['status'] != 'completed' or len(rows) != 2
            or [row['request_id'] for row in rows] != [r['request_id'] for r in requests[:2]]):
        raise ValueError('The four qualifying packets must be the prescribed event-five prefix')
    for row in rows:
        if row['status'] != 'completed':
            raise ValueError('Retained qualification feedback is incomplete')
        for field in ('record', 'feature'):
            if identity(row[field + '_path']) != row[field + '_identity']:
                raise ValueError('Retained shared packet changed')
    return deepcopy(rows)


def collect_shared(plan, dataset, event, producer=None):
    from online_feedback.sglang_backend import SGLangDFlashBackend
    from experiments.full_k_gsm8k.campaign import online_record_check
    from experiments.ok_events_5_8.campaign import publish_checked
    from .qualify import cleanup

    root = Path(plan['root'])
    output = event_root(root, dataset, event) / 'shared'
    output.mkdir(parents=True, exist_ok=False)
    require_capacity(plan, output/'STORAGE_ADMISSION.json')
    requests = feedback_requests(plan, dataset, event)
    checkpoint = (plan['datasets'][dataset]['origin_export'] if event == 5
                  else producer['checkpoint_path'])
    rows = retained_packets(plan, dataset, requests) if event == 5 else []
    progress = dict(status='running', dataset=dataset, event=event, producer=f'KPLUS{event-1}',
        reused_feedback=len(rows), records=rows, automatic_retry=False)
    ledger = WorkLedger(output / 'WORK.jsonl',
        dict(feedback_generations=128-len(rows), teacher_preparations=128-len(rows)))
    backend = None
    failure = None
    started = time.perf_counter()
    try:
        for row, request in zip(rows, requests):
            online_record_check(read(row['record_path']), request, checkpoint, capture=True)
        backend = SGLangDFlashBackend(plan['target'], checkpoint,
            worker_python=plan['worker_python'], output_root=output / 'native',
            serving_gpu='0', context_limit=4096, block_size=16)
        progress['loading_s'] = backend.loading_s
        progress['publication'] = publish_checked(backend, checkpoint, output/'PUBLICATION.json')
        for request in requests[len(rows):]:
            record_path = output / 'records' / (request['request_id']+'.json')
            row = dict(request_id=request['request_id'], record_path=str(record_path),
                       status='generation_attempted')
            rows.append(row)
            write_json(output/'COLLECTION.json', progress)
            record = ledger.call('feedback_generations', backend.generate,
                request['prompt_ids'], max_new_tokens=2048,
                request_id=request['request_id'], session_id=dataset+'_fsg_shared',
                request_index=request['request_index'], purpose='serving', save_to=record_path)
            online_record_check(record, request, checkpoint, capture=True)
            feature = output / 'loop/features' / (request['request_id']+'.pt')
            row.update(status='teacher_preparation_attempted', feature_path=str(feature))
            write_json(output/'COLLECTION.json', progress)
            prepared = ledger.call('teacher_preparations', backend.prepare_feedback, record, feature)
            if prepared.get('eligible') is False or prepared.get('status') != 'prepared':
                raise ValueError('Ineligible feedback is retained; no replacement is permitted')
            from experiments.kplus_full_gsm8k.storage import retire_redundant_features
            retirement = retire_redundant_features(record, prepared, root, output)
            row.update(status='completed', preparation=prepared, preparation_retirement=retirement, feature_identity=identity(feature),
                record_identity=identity(record_path), output_tokens=len(record['response_ids']),
                finish_reason=record['finish_reason'])
            write_json(output/'COLLECTION.json', progress)
        pool = seal_shared_packets([row['feature_path'] for row in rows],
            [row['record_path'] for row in rows], dataset=dataset, event=event,
            output=output/'POOL.json')
        require_complete_work(read_work(output/'WORK.jsonl', ledger.limits), ledger.limits)
        progress.update(status='completed', pool=str(output/'POOL.json'))
    except BaseException as error:
        failure = error
        progress.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        if backend is not None:
            cleanup(backend.close, failure, progress, 'shared serving backend close')
        progress.update(wall_s=time.perf_counter()-started,
                        work=read_work(output/'WORK.jsonl', ledger.limits))
        cleanup(lambda: write_json(output/'COLLECTION.json', progress), failure, progress, 'collection receipt')
    return pool


def diagnostic_observer(trainer, pool, baseline, *, dataset, event, output, factory):
    """Bind only sixteen predetermined windows; D sampling lives in the runner."""
    import torch
    from online_feedback.rng import preserve_rng
    module, name = factory.split(':', 1)
    execute = getattr(importlib.import_module(module), name)
    bindings = diagnostic_pairs(baseline['records'], dataset=dataset, event=event)
    files = {row['record_id']: row['feature']['path'] for row in pool['records']}
    by_window = {row['window']: row for row in bindings}
    receipts = []
    output = Path(output)
    write_json(output/'BINDINGS.json', bindings)

    def observe(phase, hook):
        window = trainer.global_step - (event-1)*128 + 1
        if phase != 'committed' or window not in by_window:
            return
        started = time.perf_counter()
        binding = by_window[window]
        actual_ids = [row['record_id'] for row in trainer.response_records[-2:]]
        if actual_ids != binding['fitting_ids']:
            raise ValueError('Current native fitting pair differs from its sealed window binding')
        with preserve_rng():
            batches = {}
            for record_id in binding['fitting_ids'] + binding['off_fit_ids']:
                raw = torch.load(files[record_id], map_location='cpu', weights_only=True)
                if raw['record_id'] != record_id:
                    raise ValueError('Diagnostic feature identity differs from the shared pool')
                batches[record_id] = dict(trainer._batch(raw).tensors, record_id=record_id)
            result = execute(hook, [batches[key] for key in binding['fitting_ids']],
                [batches[key] for key in binding['off_fit_ids']],
                output=output/f'window_{window:03d}', binding=deepcopy(binding))
        result['observer_wall_s'] = time.perf_counter()-started
        result['observer_scope'] = 'feature loading, fixed-window diagnostics and state restoration; nested in training time'
        receipts.append(result)
        write_json(output/'RESULTS.json', receipts)
    return observe, receipts


def discarded_startup(trainer, features, output):
    """Warm two backwards without Adam, then restore exact scientific state."""
    import torch
    from specforge.training.strategies.base import StepContext
    from online_feedback.rng import restore_rng
    from experiments.prefix_guard.qualify import persist_comparison
    from .qualify import scientific_state, runtime_state, restore_runtime, physical_work, cleanup

    output = Path(output)
    output.mkdir()
    ledger = WorkLedger(output/'WORK.jsonl', STARTUP_WORK)
    runtime = runtime_state(trainer.bundle.model)
    before = scientific_state(trainer)
    original_active = getattr(trainer, '_active_event', None)
    trainer._active_event = 'disposable_startup'
    (trainer.events_root/'disposable_startup').mkdir()
    trainer.bundle.model.begin_event()
    receipt = dict(status='running', optimizer_steps=0)
    started = time.perf_counter()
    failure = None
    try:
        for index, feature in enumerate(features[:2]):
            raw = torch.load(feature, map_location='cpu', weights_only=True)
            trainer._anchor_observations.clear()
            result = ledger.call('response_forwards', trainer.strategy.forward_loss,
                trainer._batch(raw), StepContext(global_step=trainer.global_step,
                    total_steps=trainer.horizon, collect_detailed_metrics=True))
            ledger.call('response_backwards', trainer.backend.backward,
                result.loss/2, is_boundary=index == 1)
            del result, raw
        receipt['status'] = 'completed'
    except BaseException as error:
        failure = error
        receipt.update(status='failed', error=repr(error))
        raise
    finally:
        receipt['physical_work'] = physical_work(trainer)
        cleanup(trainer.backend.optimizer.zero_grad, failure, receipt, 'startup gradient cleanup')
        cleanup(lambda: restore_rng(before['rng']), failure, receipt, 'startup RNG restoration')
        cleanup(lambda: restore_runtime(trainer.bundle.model, runtime), failure, receipt, 'startup runtime restoration')
        cleanup(lambda: persist_comparison(output/'EXACT_RESTORATION.json',
                    scientific_state(trainer), before), failure, receipt, 'startup exact-state check')
        trainer.response_records.clear()
        trainer.window_records.clear()
        trainer.work.clear()
        trainer._anchor_observations.clear()
        trainer._active_event = original_active
        receipt.update(wall_s=time.perf_counter()-started,
                       work=read_work(output/'WORK.jsonl', STARTUP_WORK))
        cleanup(lambda: write_json(output/'RESULT.json', receipt), failure, receipt, 'startup receipt')
    require_complete_work(receipt['work'], STARTUP_WORK)
    return receipt


def train_native_event(root, dataset, event, arm):
    """A native subprocess: complete resume, actual training, decoded publication."""
    import torch
    from experiments.prefix_guard.qualify import pin_retained_graph_runtime
    from experiments.full_k_gsm8k.costs import begin_event_profile, finish_event_profile
    from online_feedback.functional_geometry import install_functional_geometry
    from .training import FunctionalGeometryTrainer
    from .qualify import cleanup

    root = Path(root)
    plan = read(root/'sealed/PLAN.json')
    pilot_authorization(plan)
    if (arm not in ARMS or dataset not in DATASETS or event not in EVENTS
            or plan['realization_revision'] != REALIZATION_R1):
        raise ValueError('Require the sealed R1 dataset/arm/event')
    output = event_root(root, dataset, event, arm)
    output.mkdir(parents=True, exist_ok=False)
    require_capacity(plan, output/'STORAGE_ADMISSION.json')
    pool = read(event_root(root, dataset, event)/'shared/POOL.json')
    verify_shared_packets(pool)
    feature_paths = [Path(row['feature']['path']) for row in pool['records']]
    item = plan['datasets'][dataset]
    previous = read(result_path(root, dataset, event-1, arm)) if event == 6 else None
    baseline = read(result_path(root, dataset, event, 'KPLUS')) if arm != 'KPLUS' else None
    trainer = None
    failure = None
    ledger = WorkLedger(output/'WORK.jsonl', TRAINING_WORK)
    started = time.perf_counter()
    progress = dict(status='running', dataset=dataset, event=event, arm=arm,
                    realization_revision=REALIZATION_R1, automatic_retry=False)
    try:
        pin_retained_graph_runtime(output)
        loading = time.perf_counter()
        trainer = FunctionalGeometryTrainer(item['configs'][arm], output/'trainer', arm=arm,
            dataset=dataset, binding=plan['binding'], original_config_text=Path(plan['original_config']).read_text(),
            origin_state=item['origin_state'], origin_export=item['origin_export'],
            resume_state=None if previous is None else previous['state_path'],
            completed_event=event-1, realization_revision=REALIZATION_R1)
        progress['loading_s'] = time.perf_counter()-loading
        progress['startup'] = discarded_startup(trainer, feature_paths, output/'startup')
        diagnostic_results = []
        if arm != 'KPLUS':
            observer = None
            if arm == 'FSG':
                observer, diagnostic_results = diagnostic_observer(trainer, pool, baseline,
                    dataset=dataset, event=event, output=output/'diagnostics',
                    factory=plan['diagnostic_factory'])
            hook = install_functional_geometry(trainer, arm=arm, strength=1.,
                output_dir=output/'geometry', observer=observer, realization_revision=REALIZATION_R1)
        ledger.wrap(trainer.core.strategy, 'forward_loss', 'response_forwards')
        ledger.wrap(trainer.backend, 'backward', 'response_backwards')
        ledger.wrap(trainer.backend.optimizer.optimizer, 'step', 'optimizer_steps')
        profile = begin_event_profile()
        try:
            result = trainer.train_event(f'event_{event:03d}', feature_paths, passes=2,
                                         presentation_schedule=baseline)
        finally:
            write_json(output/'RESOURCE_COSTS.json', finish_event_profile(profile))
        result['diagnostic_windows'] = diagnostic_results
        diagnostic_s = sum(row['observer_wall_s'] for row in diagnostic_results)
        result['costs'].update(diagnostic_observer_s=diagnostic_s,
            training_excluding_diagnostic_observer_s=result['costs']['training_s']-diagnostic_s,
            timing_scope='training_s includes ordinary + geometry + diagnostic observer; do not sum nested timers')
        validate_update(result, dataset=dataset, arm=arm, event=event, baseline=baseline)
        require_complete_work(read_work(output/'WORK.jsonl', TRAINING_WORK), TRAINING_WORK)
        if arm != 'KPLUS':
            work = hook.report()['work']
            if (work['additional_response_forwards_attempted'] > 768
                    or work['additional_backwards_attempted'] > 256
                    or work['additional_response_forwards_failed'] or work['additional_backwards_failed']):
                raise ValueError('Main geometry work exceeds the separate event budget')
        boundary = trainer.verify_boundary(export_path=result['checkpoint_path'],
                                          receipt_path=output/'BOUNDARY_CHECK.json')
        result.update(state_path=str(trainer.latest), state_identity=identity(trainer.latest),
            weights_identity=identity(Path(result['checkpoint_path'])/'model.safetensors'),
            boundary=boundary, realization_revision=REALIZATION_R1)
        verify_shared_packets(pool)
        write_json(output/'RESULT.json', result)
        if previous is not None:
            from .pilot_storage import retire_previous_boundary
            progress['retirement'] = retire_previous_boundary(root, previous, result, output/'PREVIOUS_RETIREMENT.json')
        progress.update(status='completed', result=str(output/'RESULT.json'),
                        final_step=trainer.global_step, final_presentations=trainer.presentations)
    except BaseException as error:
        failure = error
        if trainer is not None:
            trainer.failed = True
        progress.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        if trainer is not None:
            cleanup(trainer.close, failure, progress, 'training close')
        progress.update(wall_s=time.perf_counter()-started, work=read_work(output/'WORK.jsonl', TRAINING_WORK))
        cleanup(lambda: write_json(output/'STATUS.json', progress), failure, progress, 'training receipt')
    return result


def launch_training(plan, dataset, event, arm, pool, previous=None):
    """Wait for a real native worker; failed jobs never receive automatic retries."""
    from experiments.role_loss_pilot.campaign import install_signal_handlers
    install_signal_handlers()
    root = Path(plan['root'])
    output = event_root(root, dataset, event)
    command = [plan['training_python'], '-u', '-m',
               'experiments.functional_step_geometry.native_pilot', str(root),
               '--train', dataset, str(event), arm]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(plan['training_gpu']))
    receipt = dict(command=command, dataset=dataset, event=event, arm=arm, automatic_retry=False)
    started = time.perf_counter()
    require_capacity(plan, output/(arm+'_STORAGE_ADMISSION.json'))
    with (output/(arm+'.log')).open('x') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
            env=environment, cwd=plan['execution_source']['source_path'])
        receipt['pid'] = process.pid
        write_json(output/(arm+'_EXEC.json'), receipt)
        try:
            code = process.wait()
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait()
            receipt.update(returncode=process.returncode, wall_s=time.perf_counter()-started)
            write_json(output/(arm+'_EXEC.json'), receipt)
    if code:
        raise RuntimeError('Native training failed; no retry: '+str(result_path(root,dataset,event,arm)))
    return read(result_path(root,dataset,event,arm))


def evaluate_dataset(plan, dataset, final):
    from online_feedback.sglang_backend import SGLangDFlashBackend
    from experiments.kplus_full_gsm8k.preflight import timing_barrier
    from .evaluation import evaluate_stage, audit_stage
    from .qualify import cleanup

    root = Path(plan['root'])
    item = plan['datasets'][dataset]
    output = root/'campaign_001/evaluation'/dataset
    checkpoints = {arm: final[arm]['checkpoint_path'] for arm in ARMS}
    requests = read(item['panel_manifest'])['requests']
    canonicals = read(item['references'])
    schedule = item['evaluation_schedule']
    backend = None
    failure = None
    receipt = dict(status='running')
    started = time.perf_counter()
    try:
        timing_barrier()
        backend = SGLangDFlashBackend(plan['target'], checkpoints['KPLUS'],
            worker_python=plan['worker_python'], output_root=output.parent/(dataset+'_native'),
            serving_gpu='0', context_limit=4096, block_size=16)
        receipt['loading_s'] = backend.loading_s
        def boundary():
            from .pilot_storage import check_current_capacity
            storage = check_current_capacity(root, plan['main_storage'])
            return dict(storage=storage, quiescence=timing_barrier(backend.worker.pid))
        evaluate_stage(backend, requests, checkpoints, output, canonicals=canonicals,
            dataset=dataset, event=6, schedule=schedule, boundary_check=boundary)
        for arm in ARMS:
            if identity(Path(checkpoints[arm])/'model.safetensors') != final[arm]['weights_identity']:
                raise ValueError('Final checkpoint changed during paired serving')
        audit = audit_stage(output, requests, checkpoints, canonicals=canonicals,
            dataset=dataset, event=6, schedule=schedule, evidence_root=root)
        write_json(output/'AUDIT.json', audit)
        if audit['status'] != 'passed':
            raise ValueError('Endpoint raw-evidence audit failed')
        from .statistics import summarize
        write_json(output/'SUMMARY.json', summarize(audit['rows'], dataset=dataset, event=6))
        receipt['status'] = 'completed'
    except BaseException as error:
        failure = error
        receipt.update(status='failed', error=repr(error))
        raise
    finally:
        if backend is not None:
            cleanup(backend.close, failure, receipt, 'evaluation close')
        receipt['wall_s'] = time.perf_counter()-started
        cleanup(lambda: write_json(output.parent/(dataset+'_WORKER.json'),receipt),failure,receipt,'evaluation receipt')
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--train', nargs=3, metavar=('DATASET','EVENT','ARM'))
    args = parser.parse_args()
    if args.train:
        dataset,event,arm = args.train
        train_native_event(args.root,dataset,int(event),arm)
    else:
        from .pilot import run
        from .pilot_storage import retire_shared_pool
        run(args.root, collect_shared=collect_shared, train_arm=launch_training,
            evaluate_dataset=evaluate_dataset, retire_shared=retire_shared_pool)


if __name__ == '__main__':
    main()
