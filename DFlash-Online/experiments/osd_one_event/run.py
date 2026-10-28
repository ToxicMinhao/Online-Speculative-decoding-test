"""Collect with F, perform one unchanged OSD event, close training, compare F/O.

There are exactly 128 serving generations and 384 evaluation generations.
The first eight evaluation requests double as the native correctness gate.
"""

import argparse
import json
from pathlib import Path
import shutil
import signal
import time
import traceback

from online_feedback.storage import write_json
from experiments.osd_components.run import append_json
from experiments.osd_persistent.run import check_event, score_record
from .protocol import (
    ARMS, OUTPUT_REFERENCE_ERRORS, QUALIFICATION_REQUESTS, REPEATS,
    method_order, publication_errors, record_errors, validate_manifest,
    validate_training_accounting, weight_check_errors,
)
from .report import write_results
from .storage import admit_collection, admit_update


def evaluate(backend, requests, checkpoints, root, *, qualification_requests=QUALIFICATION_REQUESTS):
    """No trainer or feedback buffer is accessible to these timed replays."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    rows = []
    checked_arms = set()
    published = False
    active_attempt = None
    progress = {
        'status': 'running', 'planned_generations': len(requests) * len(ARMS) * REPEATS,
        'attempted_generations': 0, 'returned_generations': 0,
        'unknown_generation_attempts': 0, 'qualification': 'pending',
        'activation_s': 0.0, 'weight_validation_s': 0.0,
        'optimizer_steps': 0, 'feedback_collections': 0,
    }
    try:
        for request in requests:
            ordinal = request['panel_ordinal']
            request_rows = []
            for repeat in range(REPEATS):
                for method in method_order(ordinal, repeat):
                    checkpoint = checkpoints[method]
                    if not published or backend.checkpoint_id != checkpoint:
                        receipt = backend.publish(checkpoint)
                        errors = publication_errors(receipt, checkpoint)
                        append_json(root / 'ACTIVATIONS.jsonl', {
                            'panel_ordinal': ordinal, 'repeat': repeat, 'method': method,
                            **receipt, 'errors': errors})
                        progress['activation_s'] += receipt['activation_s']
                        if errors:
                            raise RuntimeError('Native publication failed: ' + str(errors))
                        published = True
                    if method not in checked_arms:
                        started = time.perf_counter()
                        equality = backend.check_weights(checkpoint)
                        elapsed = time.perf_counter() - started
                        errors = weight_check_errors(equality, checkpoint)
                        write_json(root / f'WEIGHTS_{method}.json', {
                            'native_reply': equality, 'errors': errors, 'validation_s': elapsed})
                        progress['weight_validation_s'] += elapsed
                        if errors:
                            raise RuntimeError('Native tensor equality failed: ' + str(errors))
                        checked_arms.add(method)
                    label = f'{ordinal:04d}_r{repeat}_{method}'
                    active_attempt = {'label': label, 'request_id': request['request_id']}
                    progress['attempted_generations'] += 1
                    write_json(root / 'STATUS.json', {**progress, 'active_attempt': active_attempt})
                    record_path = root / (label + '.json')
                    record = backend.generate(
                        request['prompt_ids'], max_new_tokens=request['max_new_tokens'],
                        request_id=label, session_id='osd_one_event_evaluation', request_index=ordinal,
                        purpose='evaluation', save_to=record_path)
                    progress['returned_generations'] += 1
                    active_attempt = None
                    errors = record_errors(record, request, method, checkpoint)
                    row = score_record(record, request_id=request['request_id'], repeat=repeat, method=method)
                    row.update(panel_ordinal=ordinal, original_index=request['original_index'],
                               record_path=str(record_path), finish_reason=record['finish_reason'],
                               valid_measurement=not errors, errors=errors,
                               cache_maintenance_s=record['metadata']['capture']['cache_maintenance_s'])
                    append_json(root / 'MEASUREMENTS.jsonl', row)
                    rows.append(row)
                    request_rows.append(row)
                    structural_errors = [error for error in errors if error not in OUTPUT_REFERENCE_ERRORS]
                    if structural_errors:
                        raise RuntimeError('Structural measurement validation failed: ' + str(structural_errors))
            repeated = [method for method in ARMS if len({
                row['verification_calls'] for row in request_rows if row['method'] == method}) != 1]
            for row in request_rows:
                if row['method'] in repeated:
                    row['valid_measurement'] = False
                    row['errors'].append('repeat_count_mismatch')
            write_json(root / f'{ordinal:04d}_VALIDATION.json', {
                'request_id': request['request_id'], 'repeat_count_mismatch': repeated,
                'measurements': [{key: row[key] for key in ('method', 'repeat', 'valid_measurement', 'errors')}
                                 for row in request_rows]})
            if repeated:
                raise RuntimeError('Verification counts do not reproduce; raw evidence retained')
            if ordinal + 1 == qualification_requests:
                qualified = all(row['valid_measurement'] for row in rows)
                progress['qualification'] = 'passed' if qualified else 'failed'
                write_json(root / 'QUALIFICATION.json', {
                    'status': progress['qualification'], 'requests': qualification_requests,
                    'generations': progress['returned_generations'],
                    'arms_with_native_equality': sorted(checked_arms), 'performance_win_required': False})
                if not qualified:
                    raise RuntimeError('First-group correctness qualification failed')
            write_json(root / 'STATUS.json', progress)
        progress['status'] = 'completed'
    except BaseException as error:
        progress.update(status='failed', error_type=type(error).__name__, error=str(error))
        if progress['qualification'] == 'pending':
            progress['qualification'] = 'incomplete'
        progress['unknown_generation_attempts'] = int(active_attempt is not None)
        raise
    finally:
        progress['unattempted_generations'] = progress['planned_generations'] - progress['attempted_generations']
        progress['interrupted_attempt'] = active_attempt
        write_json(root / 'STATUS.json', progress)
        write_results(rows, requests, root / 'report')
    return progress


def collect_and_update(controller, requests, root, draft, progress):
    """The only learning call in this experiment, after all 128 F responses."""
    records = []
    for request in requests:
        progress['feedback_attempted'] += 1
        write_json(root / 'STATUS.json', progress)
        record = controller.serve(
            request['prompt_ids'], request_id=request['request_id'], session_id='gsm8k_feedback',
            request_index=request['request_index'], max_new_tokens=request['max_new_tokens'])
        records.append(record)
        progress['feedback_returned'] += 1
        if record['checkpoint_id'] != str(draft) or record['prompt_ids'] != request['prompt_ids']:
            raise RuntimeError('Feedback was not generated by released F on the sealed input')
        costs = json.loads((controller.root / 'serving' / f"{request['request_id']}.costs.json").read_text())
        progress['feedback_controller_s'] += costs['serve_call_s']
        progress['feedback_request_latency_s'] += record['metadata']['request_latency_s']
        capture = record['metadata']['capture']
        for key in ('capture_host_s', 'feature_persistence_s', 'cache_maintenance_s'):
            progress['feedback_nested_costs'][key] += capture[key]
        write_json(root / 'STATUS.json', progress)
    admit_update(root, draft, records)
    progress['phase'] = 'single_update'
    write_json(root / 'STATUS.json', progress)
    receipt = controller.update('event_001')
    validate_training_accounting(receipt, controller.trainer.status())
    progress['updates_completed'] = 1
    progress['optimizer_steps'] = 128
    return receipt


def run(args):
    from online_feedback.controller import OnlineController
    from online_feedback.qualify_sglang import check_recipe
    from online_feedback.recipe import load_event_config
    from online_feedback.sglang_backend import SGLangDFlashBackend
    from online_feedback.training import SpecForgeEventTrainer

    started = time.perf_counter()
    manifest = json.loads(Path(args.manifest).read_text())
    validate_manifest(manifest)
    config, recipe = load_event_config(args.config)
    selected_recipe = check_recipe(args.config)
    if (config.data.max_length != 2272 or config.training.objective_chunk_blocks != 8
            or config.training.max_grad_norm != 1 or config.training.loss_decay_gamma != 7
            or config.training.lr_scheduler != 'constant'):
        raise ValueError('The qualified OSD loss and sequence settings must remain unchanged')
    for supplied, configured in ((args.target, config.model.target_model_path),
                                 (args.draft, config.model.draft_checkpoint_path)):
        if Path(supplied).resolve() != Path(configured).resolve():
            raise ValueError('Serving and training model paths disagree')
    root = Path(args.output).absolute()
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / 'MANIFEST.json', manifest)
    write_json(root / 'RECIPE.json', {
        **selected_recipe, 'execution_event_limit': 1,
        'schedule_note': 'Qualified 7424-step constant-LR horizon retained; execution stops after step 128.'})
    shutil.copy2(args.config, root / 'config.yaml')
    admit_collection(root, args.draft, manifest['train_requests'])
    backend = trainer = None
    progress = {'status': 'running', 'phase': 'loading', 'feedback_attempted': 0,
                'feedback_returned': 0, 'updates_completed': 0, 'optimizer_steps': 0,
                'feedback_controller_s': 0.0, 'feedback_request_latency_s': 0.0,
                'feedback_nested_costs': {'capture_host_s': 0.0, 'feature_persistence_s': 0.0,
                                          'cache_maintenance_s': 0.0},
                'extra_qualification_generations': 0, 'automatic_retry': False}
    costs = {}
    write_json(root / 'STATUS.json', progress)
    try:
        backend = SGLangDFlashBackend(
            args.target, args.draft, worker_python=args.worker_python,
            output_root=root / 'native', serving_gpu=args.serving_gpu,
            context_limit=4096, block_size=16)
        costs['backend_loading_s'] = backend.loading_s
        trainer_started = time.perf_counter()
        trainer = SpecForgeEventTrainer(args.config, root / 'trainer', objective='osd', reduction='response_mean')
        costs['trainer_loading_s'] = time.perf_counter() - trainer_started
        initial = trainer.status()
        if initial['global_step'] != 0 or initial['presentations'] != 0 or initial['completed_events']:
            raise RuntimeError('O must start fresh from released DFlash')
        write_json(root / 'INITIAL_TRAINER.json', initial)
        validation_started = time.perf_counter()
        equality = backend.check_weights(args.draft)
        errors = weight_check_errors(equality, args.draft)
        write_json(root / 'INITIAL_F_WEIGHTS.json', {'native_reply': equality, 'errors': errors})
        if errors:
            raise RuntimeError('Released F does not match native draft weights')
        costs['initial_weight_validation_s'] = time.perf_counter() - validation_started
        event_root = root / 'event_001'
        controller = OnlineController(backend, trainer, event_root / 'loop')
        progress['phase'] = 'frozen_feedback'
        receipt = collect_and_update(controller, manifest['train_requests'], root, args.draft, progress)
        publication_started = time.perf_counter()
        check_event(receipt, trainer, backend, event_root, 1, recipe.responses_per_event)
        publication_errors_found = publication_errors(receipt['activation'], receipt['training']['checkpoint_path'])
        write_json(event_root / 'PUBLICATION_CONTRACT.json', {'errors': publication_errors_found})
        if publication_errors_found:
            raise RuntimeError('Updated checkpoint publication contract failed')
        costs['publication_validation_s'] = time.perf_counter() - publication_started
        update_costs = json.loads((controller.root / 'events/event_001.costs.json').read_text())
        costs.update(
            blocking_update_s=update_costs['update_call_s'],
            teacher_preparation_s=sum(row['teacher_preparation_s'] for row in receipt['preparation']),
            feature_persistence_s=sum(row['feature_persistence_s'] for row in receipt['preparation']),
            teacher_terminal_target_forwards=len(receipt['preparation']),
            training=receipt['training']['costs'], activation=receipt['activation'],
            checkpoint_path=receipt['training']['checkpoint_path'],
            full_state_path=receipt['training']['full_state_path'])
        progress['final_trainer_status'] = trainer.status()
        trainer.close()
        progress['closed_trainer_status'] = trainer.status()
        trainer = None
        controller.trainer = None
        progress['phase'] = 'capture_disabled_evaluation'
        write_json(root / 'STATUS.json', progress)
        write_json(root / 'COSTS.json', costs)
        print('Training closed. Beginning 384 capture-disabled replays; no transfers during timing.', flush=True)
        evaluation_started = time.perf_counter()
        evaluation = evaluate(backend, manifest['eval_requests'],
                              {'F': args.draft, 'O': receipt['training']['checkpoint_path']}, root / 'evaluation')
        costs['evaluation_wall_s'] = time.perf_counter() - evaluation_started
        costs['benchmark_activation_s'] = evaluation['activation_s']
        costs['benchmark_weight_validation_s'] = evaluation['weight_validation_s']
        progress.update(status='completed', phase='finished', evaluation=evaluation)
    except BaseException as error:
        progress.update(status='failed', error_type=type(error).__name__, error=str(error),
                        traceback=traceback.format_exc())
        if trainer is not None:
            progress['trainer_status_at_failure'] = trainer.status()
        raise
    finally:
        if trainer is not None:
            trainer.close()
        if backend is not None:
            backend.close()
        costs['actual_campaign_wall_s'] = time.perf_counter() - started
        costs['feedback_controller_s'] = progress['feedback_controller_s']
        costs['feedback_request_latency_s'] = progress['feedback_request_latency_s']
        costs['feedback_nested_costs'] = progress['feedback_nested_costs']
        costs['reconstructed_feedback_and_update_s'] = (
            progress['feedback_controller_s'] + costs.get('blocking_update_s', 0.0))
        costs['scope'] = (
            'Feedback collection includes capture. Blocking update includes teacher preparation, training, '
            'checkpoint save/export and activation exactly once; nested components are not added again. '
            'Evaluation is capture-disabled checkpoint-ready serving. Loading, validation, benchmark switches '
            'and actual campaign wall time remain separate.')
        write_json(root / 'COSTS.json', costs)
        write_json(root / 'STATUS.json', progress)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('target', 'draft', 'config', 'worker-python', 'manifest', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--serving-gpu', default='0')
    def stop_on_timeout(signum, frame):
        raise TimeoutError('Experiment supervisor requested termination')
    signal.signal(signal.SIGTERM, stop_on_timeout)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
