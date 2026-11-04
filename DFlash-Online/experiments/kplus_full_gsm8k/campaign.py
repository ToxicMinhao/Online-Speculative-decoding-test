"""One persistent synchronous K+ stream, followed by the frozen two-arm endpoint."""
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
from .storage import admit, check_growth, retire_redundant_features, retire_event

ARM = 'KPLUS'
ARM_LIMITS = dict(online_generations=7473, correctness_generations=4,
    terminal_preparation_forwards=7424, response_forwards=14848,
    response_backwards=14848, optimizer_steps=7424)


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
    from .preflight import timing_barrier
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


def run_arm(root):
    from online_feedback.controller import OnlineController
    from online_feedback.sglang_backend import SGLangDFlashBackend
    from .training import KPlusTrainer
    root = Path(root)
    plan = read(root / 'sealed/PLAN.json')
    validate_plan(plan)
    base = root / 'campaign_001/arms' / ARM
    base.mkdir(parents=True, exist_ok=False)
    progress = dict(status='running', phase='initializing', arm=ARM, binding=plan['binding'],
        events_completed=0, tail_completed=0, generation_phase='online', event_costs=[], automatic_retry=False)
    ledger = WorkLedger(base / 'WORK.jsonl', ARM_LIMITS)
    backend = trainer = None
    started = time.perf_counter()
    requests = read(plan['manifest'])['train_requests']
    try:
        admit(root, plan['storage'], receipt_path=base / 'STORAGE_ADMISSION.json')
        backend = SGLangDFlashBackend(plan['target'], plan['draft'], worker_python=plan['worker_python'],
            output_root=base / 'native_attempt_001/native', serving_gpu='0', context_limit=4096, block_size=16)
        counted_backend(backend, ledger, progress, root)
        loading = time.perf_counter()
        trainer = KPlusTrainer(plan['configs'][ARM], base / 'trainer', binding=plan['binding'],
                              original_config_text=Path(plan['original_config']).read_text())
        progress['backend_loading_s'] = backend.loading_s
        progress['trainer_loading_s'] = time.perf_counter() - loading
        initial_started = time.perf_counter()
        trainer.save_initial(plan['original_initial_state'])
        progress['initial_validation_s'] = time.perf_counter() - initial_started
        ledger.wrap(trainer.core.strategy, 'forward_loss', 'response_forwards')
        ledger.wrap(trainer.backend, 'backward', 'response_backwards')
        ledger.wrap(trainer.backend.optimizer.optimizer, 'step', 'optimizer_steps')
        previous_export = previous_boundary = None
        for event in range(1, EVENTS + 1):
            event_id = f'event_{event:03d}'
            event_root = base / 'events' / event_id
            event_root.mkdir(parents=True, exist_ok=False)
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
            write_json(event_root / 'ORIGINAL_ANCHOR_COMPARISON.json', observe_anchors(
                receipt['training'], plan.get('original_event_results'), event))
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
        for request in requests[7424:]:
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
        cleanup_errors = close_owned(progress, trainer, backend)
        progress['worker_wall_s'] = time.perf_counter() - started
        progress['actual_work'] = read_work(base / 'WORK.jsonl', ARM_LIMITS)
        write_json(base / 'STATUS.json', progress)
        if cleanup_errors:
            raise cleanup_errors[0]
    return progress


def execute_training(root, plan):
    command = [plan['training_python'], '-u', '-m', 'experiments.kplus_full_gsm8k.campaign', str(root), '--train']
    campaign = root / 'campaign_001'
    started = time.perf_counter()
    with (campaign / 'KPLUS.log').open('x') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        receipt = dict(command=command, pid=process.pid, automatic_retry=False)
        write_json(campaign / 'EXEC_KPLUS.json', receipt)
        try:
            code = process.wait()
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=45)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
        finally:
            receipt.update(returncode=process.poll(), wall_s=time.perf_counter() - started)
            write_json(campaign / 'EXEC_KPLUS.json', receipt)
    if code or read(campaign / 'arms/KPLUS/STATUS.json')['status'] != 'completed':
        raise RuntimeError('K+ training failed; preserve evidence without retry')


def run(root):
    from .evaluation import evaluate_final, audit_stage
    from online_feedback.sglang_backend import SGLangDFlashBackend
    root = Path(root).absolute()
    plan = read(root / 'sealed/PLAN.json')
    validate_plan(plan)
    campaign = root / 'campaign_001'
    campaign.mkdir(exist_ok=False)
    progress = dict(status='running', execution_source=plan['execution_source'], automatic_retry=False)
    started = time.perf_counter()
    backend = None
    try:
        admit(root, plan['storage'], receipt_path=campaign / 'STORAGE_ADMISSION.json')
        execute_training(root, plan)
        progress['training_quiescence'] = evaluation_boundary(root, plan, phase='before_final_backend')
        checkpoint = read(campaign / 'arms/KPLUS/STATUS.json')['final_checkpoint']
        checkpoints = dict(K=plan['references']['K'], KPLUS=checkpoint)
        backend = SGLangDFlashBackend(plan['target'], plan['draft'], worker_python=plan['worker_python'],
            output_root=campaign / 'final_native_attempt_001/native', serving_gpu='0', context_limit=4096, block_size=16)
        progress['final_loading_s'] = backend.loading_s
        final = campaign / 'evaluation/final'
        evaluation = evaluate_final(backend, plan['evaluation_requests'], checkpoints, final,
            canonicals=read(plan['canonicals']), baseline_counts=read(plan['baseline_counts']),
            boundary_check=lambda: evaluation_boundary(root, plan, backend))
        progress['evaluation'] = evaluation
        progress['post_evaluation_boundary'] = evaluation_boundary(root, plan, backend, phase='after_final_evaluation')
        audit = audit_stage(final, plan['evaluation_requests'], checkpoints,
            canonicals=read(plan['canonicals']), baseline_counts=read(plan['baseline_counts']),
            evidence_root=root)
        write_json(campaign / 'FINAL_AUDIT.json', audit)
        if audit['status'] != 'passed':
            raise ValueError('Complete final endpoint failed raw-evidence validation')
        progress['status'] = 'completed'
    except BaseException as error:
        progress.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        cleanup_errors = close_owned(progress, backend)
        progress['campaign_wall_s'] = time.perf_counter() - started
        write_json(campaign / 'STATUS.json', progress)
        if cleanup_errors:
            raise cleanup_errors[0]
    return progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root')
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()
    def terminate(_signum, _frame):
        raise SystemExit(128 + signal.SIGTERM)
    signal.signal(signal.SIGTERM, terminate)
    print(json.dumps(run_arm(args.root) if args.train else run(args.root), indent=2))


if __name__ == '__main__':
    main()
