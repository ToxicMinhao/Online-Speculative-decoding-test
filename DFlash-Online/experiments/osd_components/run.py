"""Replay the four fixed checkpoints; this module never creates a trainer."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import signal
import time

from online_feedback.storage import write_json
from experiments.osd_persistent.run import score_record
from .analysis import summarize, write_report
from .prepare import verify_source, verify_native_runtime, verify_artifacts
from .protocol import (ARMS, GENERATIONS, OUTPUT_REFERENCE_ERRORS, QUALIFICATION_REQUESTS, REPEATS, REQUESTS,
                       method_order, publication_errors, record_errors, weight_check_errors)


def append_json(path, value):
    """Append one durable measurement before checking its scientific validity."""
    import os

    with Path(path).open('a') as output:
        output.write(json.dumps(value, allow_nan=False) + '\n')
        output.flush()
        os.fsync(output.fileno())


def evaluate(backend, requests, checkpoints, root, *, qualification_requests=8,
             deadline=None):
    """The reading path: publish, qualify once per arm, generate, validate, record.

    The first eight requests must pass qualification. Later isolated canonical
    output/termination mismatches remain explicit. Structural errors and
    historical/repeated count failures always stop for investigation. No
    performance result is a continuation condition.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    rows, checked_arms = [], set()
    published = False
    progress = {'status': 'running', 'planned_generations': len(requests) * 12,
                'attempted_generations': 0, 'returned_generations': 0,
                'unknown_generation_attempts': 0, 'qualification': 'pending',
                'activation_s': 0.0, 'weight_validation_s': 0.0,
                'optimizer_steps': 0, 'feedback_collections': 0}
    active_attempt = None
    try:
        for request in requests:
            ordinal = request['panel_ordinal']
            request_rows = []
            for repeat in range(REPEATS):
                for method in method_order(ordinal, repeat):
                    if deadline is not None and time.monotonic() >= deadline:
                        raise TimeoutError('Two-hour experiment wall bound reached')
                    checkpoint = checkpoints[method]
                    # Even initial F uses the same publication/helper gate as
                    # every later arm; this adds no model generation.
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
                            'method': method, 'checkpoint_path': checkpoint,
                            'native_reply': equality, 'validation_s': elapsed, 'errors': errors})
                        progress['weight_validation_s'] += elapsed
                        if errors:
                            raise RuntimeError('Native tensor equality failed: ' + str(errors))
                        checked_arms.add(method)
                    label = f'{ordinal:04d}_r{repeat}_{method}'
                    active_attempt = {'label': label, 'method': method, 'repeat': repeat,
                                      'request_id': request['request_id']}
                    progress['attempted_generations'] += 1
                    write_json(root / 'STATUS.json', {**progress, 'active_attempt': active_attempt})
                    record_path = root / (label + '.json')
                    record = backend.generate(
                        request['prompt_ids'], max_new_tokens=request['max_new_tokens'],
                        request_id=label, session_id='osd_components', request_index=ordinal,
                        purpose='evaluation', save_to=record_path)
                    progress['returned_generations'] += 1
                    active_attempt = None
                    errors = record_errors(record, request, method, checkpoint)
                    row = score_record(record, request_id=request['request_id'],
                                       repeat=repeat, method=method)
                    row.update(panel_ordinal=ordinal, original_index=request['original_index'],
                               finish_reason=record['finish_reason'], record_path=str(record_path),
                               valid_measurement=not errors, errors=errors,
                               cache_maintenance_s=record['metadata']['capture']['cache_maintenance_s'])
                    append_json(root / 'MEASUREMENTS.jsonl', row)
                    rows.append(row)
                    request_rows.append(row)
                    structural_errors = [error for error in errors if error not in OUTPUT_REFERENCE_ERRORS]
                    if structural_errors:
                        raise RuntimeError('Structural measurement validation failed; evidence retained: '
                                           + str(structural_errors))
            repeated = [method for method in ARMS if len({
                row['verification_calls'] for row in request_rows if row['method'] == method}) != 1]
            if repeated:
                for row in request_rows:
                    if row['method'] in repeated:
                        row['valid_measurement'] = False
                        row['errors'].append('repeat_count_mismatch')
            write_json(root / f'{ordinal:04d}_VALIDATION.json', {
                'request_id': request['request_id'], 'valid': all(r['valid_measurement'] for r in request_rows),
                'repeat_count_mismatch': repeated,
                'measurements': [{k: r[k] for k in ('method', 'repeat', 'valid_measurement', 'errors')}
                                 for r in request_rows]})
            if repeated:
                raise RuntimeError('Verification counts do not reproduce; evidence retained')
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
            print(f'Completed request {ordinal + 1}/{len(requests)}; '
                  f'{progress["returned_generations"]} generations', flush=True)
        progress['status'] = 'completed'
    except BaseException as error:
        progress.update(status='failed', error_type=type(error).__name__, error=str(error))
        if progress['qualification'] == 'pending':
            progress['qualification'] = 'incomplete'
        if active_attempt is not None:
            progress['unknown_generation_attempts'] = 1
        raise
    finally:
        progress['unattempted_generations'] = progress['planned_generations'] - progress['attempted_generations']
        progress['interrupted_attempt'] = active_attempt
        write_json(root / 'STATUS.json', progress)
        # No new model computation: includes missing slots and invalid raw rows.
        summary = summarize(rows, requests)
        summary['execution'] = progress
        write_report(summary, root / 'report')
    return progress


def load_prepared(root):
    root = Path(root)
    for name, digest in json.loads((root / 'INPUT_SEAL.json').read_text()).items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Prepared input changed: ' + name)
    plan = json.loads((root / 'PLAN.json').read_text())
    requests = json.loads((root / 'PANEL.json').read_text())['requests']
    if (len(requests) != REQUESTS or plan['generations'] != GENERATIONS
            or plan['model_arms'] != list(ARMS) or plan['repeats'] != REPEATS
            or plan['qualification_requests'] != QUALIFICATION_REQUESTS):
        raise ValueError('Prepared execution budget differs from the fixed design')
    return plan, requests


def run(args):
    from online_feedback.sglang_backend import SGLangDFlashBackend

    started = time.perf_counter()
    deadline = time.monotonic() + 7200
    prepared = Path(args.prepared)
    plan, requests = load_prepared(prepared)
    code = Path(__file__).resolve().parents[2]
    verify_source(plan['prior_root'], code)
    verify_artifacts(prepared, plan, code)
    verify_native_runtime(plan['worker_python'], json.loads(Path(plan['runtime_pin']).read_text()))
    root = Path(args.output).absolute()
    root.mkdir(parents=True, exist_ok=False)
    free = shutil.disk_usage(root).free
    admission = {'free_bytes': free, 'reserve_bytes': 8 * 2**30,
                 'record_allowance_bytes': 2_000_000_000,
                 'required_bytes': 8 * 2**30 + 2_000_000_000}
    write_json(root / 'STORAGE_ADMISSION.json', admission)
    if free < admission['required_bytes']:
        raise ValueError('Insufficient record workspace plus reserve; do not start')
    backend = None
    costs = {'preparation_s': plan['preparation_s'], 'loading_s': None,
             'timing_scope': 'Request latency includes prefill, drafting, verification and nested cache maintenance.'}
    write_json(root / 'PLAN.json', plan)
    shutil.copy2(plan['panel'], root / 'PANEL.json')
    try:
        backend = SGLangDFlashBackend(
            plan['target'], plan['arms']['F'], worker_python=plan['worker_python'],
            output_root=root / 'native', serving_gpu=args.serving_gpu,
            context_limit=4096, block_size=16)
        costs['loading_s'] = backend.loading_s
        evaluate(backend, requests, plan['arms'], root / 'evaluation', deadline=deadline)
    finally:
        if backend is not None:
            backend.close()
        costs['actual_experiment_wall_s'] = time.perf_counter() - started
        costs['training_s'] = 0.0
        write_json(root / 'COSTS.json', costs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--serving-gpu', default='0')
    def stop_on_timeout(signum, frame):
        raise TimeoutError('Experiment supervisor requested termination')
    signal.signal(signal.SIGTERM, stop_on_timeout)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
