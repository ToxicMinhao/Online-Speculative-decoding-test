"""One persistent synchronous C stream using the inherited PR25 controller."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

from online_feedback.storage import write_json
from experiments.ok_coding.campaign import WorkLedger, read_work
from experiments.full_k_gsm8k.campaign import online_record_check, save_boundary
from experiments.full_k_gsm8k.costs import begin_event_profile, finish_event_profile
from experiments.osd_persistent.run import check_event, check_first_updated_request
from experiments.o_half_matched_lr.archive import full_trace_errors
from .contract import EVENTS, validate_plan
from .runtime import WorkLedger as SharedWork, boundary as campaign_boundary
from .storage import admit, check_growth, retire_redundant_features, retire_event

ARM = 'KPLUS_C'
ARM_LIMITS = dict(online_generations=7473, correctness_generations=4,
    terminal_preparation_forwards=7424, response_forwards=14848,
    response_backwards=14848, optimizer_steps=7424)


def restart_boundary(root,plan):
    """Only an authenticated, self-contained completed event can be resumed.

    Partial collections/optimizer windows remain evidence requiring explicit
    reconciliation. A weights-only restart is never an available branch.
    """
    import torch
    from .training import validate_snapshot
    from experiments.ok_coding.training import _export_equal
    from experiments.ok_coding.storage import identity
    base=root/'campaign_001/arms'/ARM
    progress=read(base/'STATUS.json')
    event=progress['events_completed']
    if progress['phase'] not in ('idle_boundary','tail','completed') or not 1<=event<=58:
        raise ValueError('Partial scientific work must be reconciled before continuation')
    if progress['binding']!=plan['binding'] or progress['arm']!=ARM:
        raise ValueError('Resume belongs to a different scientific trajectory')
    work=read_work(base/'WORK.jsonl',ARM_LIMITS)
    expected=expected_work(event,progress['tail_completed'])
    if work['completed']!=expected or work['attempts']!=expected or any(work['unknown'].values()):
        raise ValueError('Unknown or partial event work cannot be replayed')
    marker=read(base/'boundaries'/f'event_{event:03d}.json')
    state=torch.load(marker['state'],map_location='cpu',weights_only=False,mmap=True)
    initial=torch.load(base/'trainer/initial_state.pt',map_location='cpu',weights_only=False,mmap=True)
    if marker['status']!='passed' or identity(marker['state'])!=marker['state_identity']:
        raise ValueError('Latest complete recovery identity changed')
    checked=validate_snapshot(state,binding=plan['binding'],config_text=Path(plan['config_path']).read_text(),
        original_config_text=Path(plan['original_config']).read_text(),completed_event=event,
        expected_contract=initial['algorithm_contract'],expected_recipe=initial['backend']['optimizer']['recipe'])
    _export_equal(state,Path(marker['weights']).parent)
    if not os.path.samefile(base/'trainer/training_state.pt',marker['state']):
        raise ValueError('Latest committed recovery is not the authenticated boundary')
    for number,history in enumerate(state['events'],1):
        result=read(base/'trainer/events'/f'event_{number:03d}/RESULT.json')
        if any(result.get(k)!=v for k,v in history.items()):
            raise ValueError('Chronological event history changed')
    return progress,marker,dict(status='passed',boundary=marker,scientific_state=checked,work=work)


def read(path):
    return json.loads(Path(path).read_text())


def close_owned(progress, *owners):
    """Attempt every owned cleanup and make any failure visible in status."""
    errors = []
    for owner in owners:
        if owner is not None:
            try:
                owner.close()
            except BaseException as error:
                errors.append(error)
    if errors:
        progress.update(status='failed', cleanup_errors=[repr(error) for error in errors])
    return errors


def expected_work(events, tail=0):
    return dict(online_generations=events * 128 + tail,
        correctness_generations=4 if events else 0, terminal_preparation_forwards=events * 128,
        response_forwards=events * 256, response_backwards=events * 256, optimizer_steps=events * 128)


def observe_anchors(result, original_root, event):
    """Compare saved natural draws only; never supply a historical schedule."""
    if not original_root:
        return dict(status='unavailable', reason='No authenticated original event-result path supplied', forced=False)
    path = Path(original_root) / f'event_{event:03d}/RESULT.json'
    original = read(path)
    before = {(row['pass'], row['record_id']): row for row in original['records']}
    identity_matches = sum((row['pass'], row['record_id']) in before for row in result['records'])
    anchor_matches = sum(before.get((row['pass'], row['record_id']), {}).get('sampled_anchors') == row['sampled_anchors']
                         for row in result['records'])
    order_matches = sum(a['record_id'] == b['record_id'] and a['pass'] == b['pass']
                        for a, b in zip(original['records'], result['records']))
    return dict(status='compared', source=str(path), presentations=len(result['records']),
        identity_matches=identity_matches, exact_anchor_matches=anchor_matches,
        presentation_order_matches=order_matches, forced=False,
        scope='Natural draw equality only; generated-token equality does not establish hidden-feature equality.')


def resource_check(root, storage, destination, phase, event=None):
    from experiments.kw_v1.evaluate import append_json
    receipt = dict(phase=phase, event=event, **check_growth(root, storage))
    append_json(destination, receipt)
    return receipt


def evaluation_boundary(root, plan, backend=None, *, phase='final_batch'):
    """Check storage and quiescence outside every timed request batch."""
    from experiments.kplus_full_gsm8k.preflight import timing_barrier
    resource = resource_check(root, plan['storage'], root / 'campaign_001/RESOURCE_CHECKS.jsonl', phase)
    owner = backend.worker.pid if backend is not None else None
    return dict(resource=resource, timing=timing_barrier(owner))


def counted_backend(backend, ledger, progress, root):
    generate, prepare = backend.generate, backend.prepare_feedback

    def generate_once(*args, **kwargs):
        kind = 'correctness_generations' if progress['generation_phase'] == 'correctness' else 'online_generations'
        return ledger.call(kind, generate, *args, **kwargs)

    def prepare_once(record, path):
        prepared = ledger.call('terminal_preparation_forwards', prepare, record, path)
        if prepared.get('eligible') is not False and prepared.get('status') == 'prepared':
            retire_redundant_features(record=record, prepared_receipt=prepared, root=root,
                                     event_root=Path(progress['event_root']))
        return prepared

    backend.generate, backend.prepare_feedback = generate_once, prepare_once


def run_arm(root,*,resume=False):
    from online_feedback.controller import OnlineController
    from online_feedback.sglang_backend import SGLangDFlashBackend
    from .training import FullCTrainer
    root = Path(root)
    plan = read(root / 'sealed/PLAN.json')
    validate_plan(plan)
    base = root / 'campaign_001/arms' / ARM
    base.mkdir(parents=True, exist_ok=resume)
    shared = SharedWork(root, 'production_training')
    if resume:
        progress,previous_boundary,resume_check=restart_boundary(root,plan)
        from experiments.kw_v1.evaluate import append_json
        append_json(base/'RESUME_CHECKS.jsonl',resume_check)
        ledger=WorkLedger.__new__(WorkLedger)
        ledger.path,ledger.limits=base/'WORK.jsonl',dict(ARM_LIMITS)
        work=resume_check['work'];ledger.attempts=work['attempts'];ledger.completed=work['completed']
        progress.update(status='running',generation_phase='online')
        previous_export=Path(previous_boundary['weights']).parent
    else:
        progress = dict(status='running', phase='initializing', arm=ARM, binding=plan['binding'],
            events_completed=0, tail_completed=0, generation_phase='online', event_costs=[], automatic_retry=False)
        ledger = WorkLedger(base / 'WORK.jsonl', ARM_LIMITS)
        previous_export = previous_boundary = None
    backend = trainer = None
    started = time.perf_counter()
    requests = read(plan['manifest'])['train_requests']
    try:
        admit(root, plan['storage'], receipt_path=base / 'STORAGE_ADMISSION.json')
        attempt=len(list(base.glob('native_attempt_*')))+1
        backend = SGLangDFlashBackend(plan['target'], str(previous_export or plan['draft']), worker_python=plan['worker_python'],
            output_root=base / f'native_attempt_{attempt:03d}/native', serving_gpu=plan['devices']['A'], context_limit=4096, block_size=16,
            worker_module='experiments.persistent_prefix_withholding.worker')
        original_generate = backend.generate
        def generate(*args, **kwargs):
            kind = 'correctness_generation' if progress['generation_phase']=='correctness' else 'feedback_generation'
            return shared.call({kind:1}, kind, original_generate, *args, **kwargs)
        backend.generate = generate
        shared.wrap(backend, 'prepare_feedback', 'teacher_preparation')
        counted_backend(backend, ledger, progress, root)
        loading = time.perf_counter()
        trainer = shared.call({'training_initialization':1}, 'production_constructor', FullCTrainer,
            plan['configs'][ARM], base / 'trainer', binding=plan['binding'],
            original_config_text=Path(plan['original_config']).read_text(),
            resume_state=previous_boundary['state'] if resume else None,
            completed_event=progress['events_completed'])
        progress['backend_loading_s'] = backend.loading_s
        progress['trainer_loading_s'] = time.perf_counter() - loading
        initial_started = time.perf_counter()
        if not resume:
            trainer.save_initial(plan['original_initial_state'])
        progress['initial_validation_s'] = time.perf_counter() - initial_started
        ledger.wrap(trainer.core.strategy, 'forward_loss', 'response_forwards')
        ledger.wrap(trainer.backend, 'backward', 'response_backwards')
        ledger.wrap(trainer.backend.optimizer.optimizer, 'step', 'optimizer_steps')
        shared.wrap(trainer.core.strategy, 'forward_loss', 'response_forward')
        shared.wrap(trainer.backend, 'backward', 'response_backward')
        shared.wrap(trainer.backend.optimizer.optimizer, 'step', 'adam')
        for event in range(progress['events_completed']+1, EVENTS + 1):
            campaign_boundary(plan)
            event_id = f'event_{event:03d}'
            event_root = base / 'events' / event_id
            if event_root.exists() and any(p.name!='STORAGE_ADMISSION.json' for p in event_root.iterdir()):
                raise ValueError('An incomplete future event has evidence requiring reconciliation')
            event_root.mkdir(parents=True, exist_ok=True)
            admit(root, plan['storage'], receipt_path=event_root / 'STORAGE_ADMISSION.json')
            progress.update(phase='collecting', generation_phase='online', current_event=event,
                            event_root=str(event_root))
            resource_check(root, plan['storage'], base / 'RESOURCE_CHECKS.jsonl', 'before_event', event)
            write_json(base / 'STATUS.json', progress)
            controller = OnlineController(backend, trainer, event_root / 'loop')
            records = []
            for request in requests[(event - 1) * 128:event * 128]:
                record = controller.serve(request['prompt_ids'], request_id=request['request_id'],
                    session_id='gsm8k_train', request_index=request['request_index'], max_new_tokens=2048)
                online_record_check(record, request, previous_export or plan['draft'], capture=True)
                records.append(record)
            if event == 1:
                progress['generation_phase'] = 'correctness'
                probe = records[0]
                replay = controller.evaluate(probe['prompt_ids'], canonical_response_ids=probe['response_ids'],
                                             label='first_capture_disabled', max_new_tokens=2048)
                if (replay['record']['finish_reason'] != probe['finish_reason'] or
                        replay['record']['metadata']['verification_calls'] != probe['metadata']['verification_calls']):
                    raise ValueError('Inherited capture-disabled correctness check differs')
            progress.update(phase='training', generation_phase='online')
            write_json(base / 'STATUS.json', progress)
            compiler_before = begin_event_profile()
            try:
                receipt = controller.update(event_id)
            finally:
                write_json(event_root / 'RESOURCE_COSTS.json', finish_event_profile(compiler_before))
            validation_started = time.perf_counter()
            check_event(receipt, trainer, backend, event_root, event, 128)
            boundary = save_boundary(root, ARM, trainer, receipt)
            validation_s = time.perf_counter() - validation_started
            export = Path(receipt['training']['checkpoint_path'])
            if event == 1:
                progress['generation_phase'] = 'correctness'
                correctness = check_first_updated_request(backend, requests[128], plan['draft'], str(export), event_root)
                checks = [read(event_root / f'qualification_{name}.json') for name in ('updated', 'frozen', 'reloaded')]
                if len({row['finish_reason'] for row in checks}) != 1 or any(full_trace_errors(row) for row in checks):
                    raise ValueError('Inherited updated/frozen/reloaded correctness differs')
                write_json(event_root / 'INHERITED_CORRECTNESS.json', dict(correctness, generations=4,
                           first_capture_disabled_in_loop=True, disposable_optimizer_steps=0))
            write_json(event_root / 'SAMPLING_POLICY.json', dict(natural=True, forced=False,
                historical_anchor_comparison=False, own_generated_responses=True))
            retention_started = time.perf_counter()
            retire_event(root=root, event_root=event_root, records=records, receipt=receipt,
                previous_export=previous_export, previous_boundary=previous_boundary, current_boundary=boundary)
            costs = dict(read(event_root / f'loop/events/{event_id}.costs.json'),
                **receipt['training']['costs'],
                teacher_preparation_s=sum(row['teacher_preparation_s'] for row in receipt['preparation']),
                feature_persistence_s=sum(row['feature_persistence_s'] for row in receipt['preparation']),
                activation=receipt['activation'], publication_validation_s=validation_s,
                retention_s=time.perf_counter() - retention_started)
            progress['event_costs'].append(dict(event=event, costs=costs))
            progress.update(events_completed=event, phase='idle_boundary', generation_phase='online',
                final_checkpoint=str(export), boundary=boundary, actual_work=read_work(base / 'WORK.jsonl', ARM_LIMITS))
            expected = expected_work(event)
            if (progress['actual_work']['completed'] != expected or progress['actual_work']['attempts'] != expected
                    or any(progress['actual_work']['unknown'].values())):
                raise ValueError('Complete event work differs from the fixed budget')
            resource_check(root, plan['storage'], base / 'RESOURCE_CHECKS.jsonl', 'after_retention', event)
            write_json(base / 'STATUS.json', progress)
            previous_export, previous_boundary = export, boundary
            print(json.dumps(dict(arm=ARM, event=event, steps=trainer.global_step)), flush=True)
        progress.update(phase='tail', generation_phase='online')
        for request in requests[7424+progress['tail_completed']:]:
            record = backend.generate(request['prompt_ids'], request_id=request['request_id'],
                session_id='gsm8k_tail', request_index=request['request_index'], max_new_tokens=2048,
                purpose='evaluation', save_to=base / 'tail' / (request['request_id'] + '.json'))
            online_record_check(record, request, previous_export, capture=False)
            progress['tail_completed'] += 1
            write_json(base / 'STATUS.json', progress)
        progress['final_state_check'] = trainer.verify_boundary(export_path=previous_export,
                                                               receipt_path=base / 'FINAL_STATE_CHECK.json')
        progress['actual_work'] = read_work(base / 'WORK.jsonl', ARM_LIMITS)
        if (progress['actual_work']['completed'] != ARM_LIMITS or progress['actual_work']['attempts'] != ARM_LIMITS
                or any(progress['actual_work']['unknown'].values())):
            raise ValueError('Full one-arm accounting does not match the budget')
        progress.update(status='completed', phase='completed')
    except BaseException as error:
        progress.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        # Save the scientific failure before asking a possibly failed worker for
        # optional physical counters. Cleanup must not erase the original cause.
        write_json(base/'STATUS.json', progress)
        receipt_error = None
        if backend is not None and not backend.failed:
            try:
                write_json(base/'PHYSICAL_WORK.json', backend._call('work_receipt'))
            except BaseException as error:
                receipt_error = error
                progress.update(status='failed', physical_work_error=repr(error))
        cleanup_errors = close_owned(progress, trainer, backend)
        progress['worker_wall_s'] = time.perf_counter() - started
        progress['actual_work'] = read_work(base / 'WORK.jsonl', ARM_LIMITS)
        write_json(base / 'STATUS.json', progress)
        if cleanup_errors:
            raise cleanup_errors[0]
        if receipt_error is not None:
            raise receipt_error
    return progress
