"""Serve fresh feedback, update every buffer, then compare final and released drafts.

The experiment owns request order, evaluation and artifact retention. Inference,
feedback conditioning and all learning operations stay in online_feedback.
"""

import argparse
import csv
import json
from pathlib import Path
import shutil
import time
import traceback

from online_feedback.storage import write_json
from .data import validate_manifest
from .retention import admit_storage, checked_retire


def log(message):
    print(time.strftime('%Y-%m-%dT%H:%M:%S'), message, flush=True)


def score_record(record, *, request_id, repeat, method):
    metrics = record['metadata']
    return {
        'request_id': request_id, 'repeat': repeat, 'method': method,
        'checkpoint_id': record['checkpoint_id'],
        'prompt_tokens': len(record['prompt_ids']),
        'output_tokens': len(record['response_ids']),
        'cap_hit': record['finish_reason'] == 'length',
        'verification_calls': metrics['verification_calls'],
        'accepted_draft_tokens': metrics['accepted_draft_tokens'],
        'committed_progress': metrics['committed_progress'],
        'request_latency_s': metrics['request_latency_s'],
        'native_generate_s': metrics['native_generate_s'],
    }


def summarize_pairs(rows):
    """Pool identical-input/output pairs; keep invalid results in the raw table."""
    valid = [row for row in rows if row['valid_pair']]
    total = {'pair_count': len(rows) // 2, 'valid_pairs': len(valid) // 2,
             'invalid_pairs': (len(rows) - len(valid)) // 2}
    for method in ('F', 'O'):
        selected = [row for row in valid if row['method'] == method]
        sums = {key: sum(row[key] for row in selected) for key in (
            'output_tokens', 'verification_calls', 'accepted_draft_tokens',
            'committed_progress', 'request_latency_s')}
        calls, tokens = sums['verification_calls'], sums['output_tokens']
        sums['calls_per_1000_output_tokens'] = 1000 * calls / tokens if tokens else None
        sums['committed_progress_per_call'] = sums['committed_progress'] / calls if calls else None
        sums['output_tokens_per_s'] = tokens / sums['request_latency_s'] if selected else None
        total[method] = sums
    for name, key in [('call_reduction_pct', 'verification_calls'),
                      ('request_time_reduction_pct', 'request_latency_s')]:
        base = total['F'][key]
        total[name] = 100 * (1 - total['O'][key] / base) if base else None
    total['distinct_requests'] = len({row['request_id'] for row in valid})
    total['timing_scope'] = 'Capture-disabled complete request; checkpoint already available.'
    return total


def evaluate(backend, requests, released, trained, root, *, repeats=3):
    """Fresh paired measurements; no trainer or feedback buffer is passed here."""
    root.mkdir()
    rows, switches = [], []
    for index, request in enumerate(requests):
        reference_ids = None
        for repeat in range(repeats):
            order = ('F', 'O') if (index + repeat) % 2 == 0 else ('O', 'F')
            pair = {}
            for method in order:
                checkpoint = released if method == 'F' else trained
                if backend.checkpoint_id != str(checkpoint):
                    switches.append(backend.publish(checkpoint))
                label = f'{index:04d}_r{repeat}_{method}'
                record = backend.generate(
                    request['prompt_ids'], max_new_tokens=request['max_new_tokens'],
                    request_id=label, session_id='heldout', request_index=index,
                    purpose='evaluation', save_to=root / f'{label}.json')
                pair[method] = record
            if reference_ids is None:
                reference_ids = pair['F']['response_ids']
            errors = []
            for method, record in pair.items():
                if record['prompt_ids'] != request['prompt_ids']:
                    errors.append(f'{method}: input token mismatch')
                if record['response_ids'] != reference_ids:
                    errors.append(f'{method}: canonical frozen output mismatch')
                if record['metadata']['feedback_capture_in_timed_request']:
                    errors.append(f'{method}: feedback capture enabled')
            comparison = {'request_id': request['request_id'], 'repeat': repeat,
                          'order': order, 'valid': not errors, 'errors': errors,
                          'canonical_response_ids': reference_ids}
            write_json(root / f'{index:04d}_r{repeat}_comparison.json', comparison)
            for method in ('F', 'O'):
                row = score_record(pair[method], request_id=request['request_id'],
                                   repeat=repeat, method=method)
                rows.append({**row, 'valid_pair': not errors, 'errors': '; '.join(errors)})
            with (root / 'PER_REQUEST.csv').open('w', newline='') as output:
                writer = csv.DictWriter(output, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            write_json(root / 'SUMMARY.json', summarize_pairs(rows))
        log(f'heldout {index + 1}/{len(requests)} complete')
    write_json(root / 'BENCHMARK_ACTIVATIONS.json', switches)
    return summarize_pairs(rows)


def check_event(receipt, trainer, backend, event_root, event_number, event_size):
    """Correctness gate only; a worse checkpoint still continues unchanged."""
    training = receipt['training']
    expected = {'optimizer_steps': event_size,
                'response_presentations': 2 * event_size,
                'end_step': event_number * event_size}
    for key, value in expected.items():
        if training[key] != value:
            raise AssertionError(f'{key}: expected {value}, got {training[key]}')
    publication = backend.check_weights(training['checkpoint_path'])
    if not publication['all_equal']:
        raise AssertionError('Published native draft differs from trained export')
    write_json(event_root / 'PUBLICATION_CHECK.json', {
        **publication, 'status': 'passed', 'event_id': training['event_id'],
        'checkpoint_path': training['checkpoint_path'], 'expected_accounting': expected,
        'trainer_status': trainer.status(),
    })


def check_first_updated_request(backend, request, released, trained, root):
    """Exercise updated/frozen/reloaded serving without collecting any feedback."""
    started = time.perf_counter()
    records = {}
    for label, checkpoint in (('updated', trained), ('frozen', released), ('reloaded', trained)):
        if backend.checkpoint_id != str(checkpoint):
            backend.publish(checkpoint)
        records[label] = backend.generate(
            request['prompt_ids'], max_new_tokens=request['max_new_tokens'],
            request_id='qualification_' + label, session_id='qualification', request_index=0,
            purpose='evaluation', save_to=root / f'qualification_{label}.json')
    same_input = all(row['prompt_ids'] == request['prompt_ids'] for row in records.values())
    same_output = all(row['response_ids'] == records['frozen']['response_ids'] for row in records.values())
    reload_calls = (records['updated']['metadata']['verification_calls']
                    == records['reloaded']['metadata']['verification_calls'])
    capture_disabled = all(not row['metadata']['feedback_capture_in_timed_request']
                           for row in records.values())
    result = {'same_input': same_input, 'same_output': same_output,
              'reloaded_calls_equal': reload_calls, 'wall_s': time.perf_counter() - started,
              'feedback_collected': not capture_disabled, 'performance_win_required': False}
    write_json(root / 'NATIVE_REPLAY_CHECK.json', result)
    if not (same_input and same_output and reload_calls and capture_disabled):
        raise AssertionError('First full-event output/switching check failed; evidence preserved')
    return result


def run(args):
    from online_feedback.controller import OnlineController
    from online_feedback.qualify_sglang import check_recipe
    from online_feedback.recipe import load_event_config
    from online_feedback.sglang_backend import SGLangDFlashBackend
    from online_feedback.training import SpecForgeEventTrainer

    root = Path(args.output).absolute()
    root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    manifest = json.loads(Path(args.manifest).read_text())
    validate_manifest(manifest)
    config, recipe = load_event_config(args.config)
    actual_recipe = check_recipe(args.config)
    for supplied, configured in ((args.target, config.model.target_model_path),
                                 (args.draft, config.model.draft_checkpoint_path)):
        if Path(supplied).resolve() != Path(configured).resolve():
            raise ValueError('Serving and training model paths disagree')
    if (config.training.objective_chunk_blocks != 8 or config.training.max_grad_norm != 1
            or config.training.loss_decay_gamma != 7 or config.training.lr_scheduler != 'constant'):
        raise ValueError('Selected loss/optimizer settings changed')
    write_json(root / 'MANIFEST.json', manifest)
    write_json(root / 'RECIPE.json', actual_recipe)
    shutil.copy2(args.config, root / 'config.yaml')
    if config.data.max_length != manifest['policy']['train_context_limit']:
        raise ValueError('Manifest and configured training length disagree')
    admission = dict(max_sequence_length=config.data.max_length,
                     responses_per_event=recipe.responses_per_event)
    admit_storage(root, args.draft, **admission, receipt_path=root / 'STORAGE_ADMISSION.json')

    backend = trainer = None
    progress = {'status': 'running', 'serving_completed': 0, 'updates_completed': 0,
                'optimizer_steps': 0, 'training_costs': [], 'serving_latency_s': 0.0,
                'controller_serving_s': 0.0, 'blocking_update_s': 0.0,
                'qualification_s': 0.0, 'publication_check_s': 0.0, 'retention_s': 0.0,
                'note': 'Training is synchronous; historical outcomes do not gate continuation.'}
    write_json(root / 'STATUS.json', progress)
    try:
        backend = SGLangDFlashBackend(
            args.target, args.draft, worker_python=args.worker_python,
            output_root=root / 'native', serving_gpu=args.serving_gpu,
            context_limit=manifest['policy']['serving_context_limit'], block_size=16)
        trainer_started = time.perf_counter()
        trainer = SpecForgeEventTrainer(args.config, root / 'trainer', objective='osd',
                                        reduction='response_mean')
        progress['loading_s'] = {'backend': backend.loading_s,
                                'trainer': time.perf_counter() - trainer_started}
        previous_export = None
        requests = manifest['train_requests']
        event_size = recipe.responses_per_event
        for number in range(1, recipe.max_events + 1):
            event_id = f'event_{number:03d}'
            event_root = root / 'events' / event_id
            event_root.mkdir(parents=True)
            admit_storage(root, args.draft, **admission,
                          current_state=trainer.latest if trainer.latest.exists() else None,
                          current_export=previous_export,
                          receipt_path=event_root / 'STORAGE_ADMISSION.json')
            controller = OnlineController(backend, trainer, event_root / 'loop')
            records = []
            group = requests[(number - 1) * event_size:number * event_size]
            for request in group:
                record = controller.serve(
                    request['prompt_ids'], request_id=request['request_id'],
                    session_id='gsm8k_train', request_index=request['request_index'],
                    max_new_tokens=request['max_new_tokens'])
                records.append(record)
                serve_cost = json.loads((event_root / 'loop/serving' /
                                        f"{request['request_id']}.costs.json").read_text())
                progress['controller_serving_s'] += serve_cost['serve_call_s']
                progress['serving_completed'] += 1
                progress['serving_latency_s'] += record['metadata']['request_latency_s']
                write_json(root / 'STATUS.json', progress)
                log(f"serve {progress['serving_completed']}/{len(requests)}; "
                    f"tokens={len(record['response_ids'])}, calls={record['metadata']['verification_calls']}")
            if number == 1:
                # Frozen captured serving must reproduce with capture disabled.
                qualification_started = time.perf_counter()
                probe = records[0]
                replay = controller.evaluate(
                    probe['prompt_ids'], canonical_response_ids=probe['response_ids'],
                    label='first_capture_disabled', max_new_tokens=probe['max_new_tokens'])
                if replay['record']['metadata']['verification_calls'] != probe['metadata']['verification_calls']:
                    raise AssertionError('Capture-disabled frozen verification count differs')
                progress['qualification_s'] += time.perf_counter() - qualification_started
            log(f'{event_id}: starting 128-step update')
            progress.update(phase='training', current_event=event_id)
            write_json(root / 'STATUS.json', progress)
            receipt = controller.update(event_id)
            publication_check_started = time.perf_counter()
            check_event(receipt, trainer, backend, event_root, number, event_size)
            progress['publication_check_s'] += time.perf_counter() - publication_check_started
            current_export = Path(receipt['training']['checkpoint_path'])
            if number == 1:
                native_check = check_first_updated_request(
                    backend, requests[event_size], args.draft, str(current_export), event_root)
                progress['qualification_s'] += native_check['wall_s']
            progress.update(phase='serving', updates_completed=number, optimizer_steps=trainer.global_step,
                            latest_checkpoint=str(current_export),
                            current_state=str(trainer.latest))
            progress['training_costs'].append({
                'event_id': event_id,
                'costs': json.loads((event_root / 'loop/events' / f'{event_id}.costs.json').read_text()),
                'training': receipt['training']['costs'],
                'activation': receipt['activation'],
                'teacher_preparation_s': sum(row['teacher_preparation_s'] for row in receipt['preparation']),
                'feature_persistence_s': sum(row['feature_persistence_s'] for row in receipt['preparation']),
            })
            progress['blocking_update_s'] += progress['training_costs'][-1]['costs']['update_call_s']
            progress['reconstructed_episode_s'] = progress['controller_serving_s'] + progress['blocking_update_s']
            write_json(root / 'STATUS.json', progress)
            retirement_started = time.perf_counter()
            checked_retire(event_root, records, receipt, previous_export, root)
            progress['retention_s'] += time.perf_counter() - retirement_started
            previous_export = current_export
            log(f'{event_id}: committed; total steps={trainer.global_step}')

        for request in requests[recipe.max_events * event_size:]:
            # The reference's final 49 requests do not fill a new training event.
            tail_started = time.perf_counter()
            record = backend.generate(
                request['prompt_ids'], request_id=request['request_id'],
                session_id='gsm8k_tail', request_index=request['request_index'],
                max_new_tokens=request['max_new_tokens'], purpose='evaluation',
                save_to=root / 'tail' / f"{request['request_id']}.json")
            progress['serving_completed'] += 1
            progress['controller_serving_s'] += time.perf_counter() - tail_started
            progress['serving_latency_s'] += record['metadata']['request_latency_s']
            progress['reconstructed_episode_s'] = progress['controller_serving_s'] + progress['blocking_update_s']
            write_json(root / 'STATUS.json', progress)
        progress['online_episode_wall_s'] = time.perf_counter() - started
        progress['cost_scope'] = (
            'reconstructed_episode_s = controller serving + blocking controller update, each once; '
            'capture is already inside serving. Training/checkpoint/export/activation are nested '
            'update components, not additional charges. Loading, qualification, publication checks, '
            'retirement and orchestration account for other online wall time. Final replays are separate.')
        progress['status'] = 'evaluating'
        progress['final_trainer_status'] = trainer.status()
        trainer.close()
        trainer = None
        write_json(root / 'STATUS.json', progress)
        evaluation_started = time.perf_counter()
        summary = evaluate(backend, manifest['eval_requests'], args.draft,
                           str(previous_export), root / 'evaluation')
        progress.update(status='completed', evaluation_wall_s=time.perf_counter() - evaluation_started,
                        evaluation=summary)
    except BaseException as error:
        progress.update(status='failed', error_type=type(error).__name__, error=str(error),
                        traceback=traceback.format_exc(), automatic_retry=False)
        raise
    finally:
        progress['actual_campaign_wall_s'] = time.perf_counter() - started
        write_json(root / 'STATUS.json', progress)
        if trainer is not None:
            trainer.close()
        if backend is not None:
            backend.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('target', 'draft', 'config', 'worker-python', 'manifest', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--serving-gpu', default='0')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
