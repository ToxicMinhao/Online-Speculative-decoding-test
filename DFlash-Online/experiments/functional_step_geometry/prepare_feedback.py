"""Collect only the first two sealed event-5 inputs per dataset for qualification.

These four answers count toward the prescribed 512 shared feedback generations.
They are retained for the future event, not discarded or collected again. This
entry point performs no optimizer work and cannot launch the main pilot.
"""
import argparse
import json
from pathlib import Path
import time
import traceback

from experiments.full_k_gsm8k.campaign import online_record_check
from experiments.ok_coding.storage import identity
from online_feedback.sglang_backend import SGLangDFlashBackend
from online_feedback.storage import write_json
from .qualify import cleanup


def prepare(root):
    root = Path(root)
    plan = json.loads((root / 'sealed/PLAN.json').read_text())
    output = root / 'qualification_feedback'
    output.mkdir(exist_ok=False)
    receipt = dict(status='running', datasets={}, shared_feedback_generations=0,
                   teacher_preparations=0, extra_qualification_generations=0,
                   main_updates=0, automatic_retry=False,
                   execution_commit=plan['execution_source']['commit'])
    started = time.perf_counter()
    failure = None
    try:
        for dataset, item in plan['datasets'].items():
            directory = output / dataset
            directory.mkdir()
            backend = None
            generation_failure = None
            rows = []
            receipt['datasets'][dataset] = rows
            try:
                backend = SGLangDFlashBackend(
                    plan['target'], item['origin_export'],
                    worker_python=plan['worker_python'],
                    output_root=directory / 'native', serving_gpu='0',
                    context_limit=4096, block_size=16)
                requests = json.loads(Path(item['training_manifest']).read_text())['train_requests']
                for request in requests[:2]:
                    record_path = directory / 'records' / (request['request_id'] + '.json')
                    row = dict(request_id=request['request_id'],
                               record_path=str(record_path), status='generation_attempted')
                    rows.append(row)
                    receipt['shared_feedback_generations'] += 1
                    write_json(output / 'PREPARATION.json', receipt)
                    record = backend.generate(
                        request['prompt_ids'], max_new_tokens=2048,
                        request_id=request['request_id'], session_id=dataset + '_fsg_shared',
                        request_index=request['request_index'], purpose='serving',
                        save_to=record_path)
                    online_record_check(record, request, item['origin_export'], capture=True)
                    feature = directory / 'features' / (request['request_id'] + '.pt')
                    row.update(status='teacher_preparation_attempted', feature_path=str(feature))
                    receipt['teacher_preparations'] += 1
                    write_json(output / 'PREPARATION.json', receipt)
                    prepared = backend.prepare_feedback(record, feature)
                    if prepared.get('eligible') is False or prepared.get('status') != 'prepared':
                        raise ValueError('No eligible feedback: ' + request['request_id'])
                    row.update(status='completed', feature_identity=identity(feature),
                               record_identity=identity(record_path), preparation=prepared,
                               output_tokens=len(record['response_ids']),
                               finish_reason=record['finish_reason'])
                    write_json(output / 'PREPARATION.json', receipt)
                write_json(directory / 'LOADING.json', dict(loading_s=backend.loading_s))
            except BaseException as error:
                generation_failure = error
                raise
            finally:
                if backend is not None:
                    cleanup(backend.close, generation_failure, receipt, 'feedback backend close')
        receipt['status'] = 'completed'
    except BaseException as error:
        failure = error
        receipt.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        receipt['wall_s'] = time.perf_counter() - started
        cleanup(lambda: write_json(output / 'PREPARATION.json', receipt), failure, receipt,
                'feedback result write')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root')
    arguments = parser.parse_args()
    result = prepare(arguments.root)
    print(json.dumps({key: result[key] for key in ('status', 'wall_s', 'shared_feedback_generations')}))
