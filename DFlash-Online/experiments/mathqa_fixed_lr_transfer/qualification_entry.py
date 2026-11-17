"""Disposable native checks on each physical card; never serving checkpoints."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import time
import traceback

from online_feedback.storage import write_json
from experiments.fresh_confirmation.protocol import weight_check_errors
from experiments.persistent_prefix_withholding.qualification_entry import HostMemorySample, install_interrupt_handlers
from experiments.persistent_prefix_withholding.serving_qualification import evaluate, gpu_memory
from experiments.persistent_prefix_withholding.task_runner import create_backend, close_backend
from .runtime import WorkLedger, boundary, occupied_card, read


def serving(plan, slot, base):
    """Two sealed warmup inputs, twice, at the disposable S endpoint."""
    from .task_runner import verify_reference_seal
    verify_reference_seal(plan)
    training = read(Path(plan['root'])/'qualification'/('training_'+slot)/'STATUS.json')
    if training['status'] != 'passed':
        raise ValueError('Serving checks require completed native training qualification')
    checkpoint = training['result']['export']
    uuid = plan['devices'][slot]
    requests = read(plan['warmups'])['requests']
    references = read(plan['warmup_references'])
    from .evaluation import validate_references
    validate_references(requests, references, origin_checkpoint=plan['origin_export'])
    ledger = WorkLedger(plan['root'], 'qualification_serving_'+slot)
    result = dict(status='running', checkpoint=checkpoint, role='disposable_S',
                  execution_commit=plan['execution_commit'])
    backend, failure = None, None
    write_json(base/'RESULT.json', result)
    try:
        backend = create_backend(plan, base, checkpoint, uuid)
        result['loading_s'] = backend.loading_s
        ledger.wrap(backend, 'generate', 'qualification_generation')
        for phase in ('before', 'after'):
            check = backend.check_weights(checkpoint)
            errors = weight_check_errors(check, str(checkpoint))
            write_json(base/('WEIGHTS_'+phase.upper()+'.json'), dict(check=check, errors=errors))
            if errors:
                raise ValueError('Disposable S serving weights differ from export')
            if phase == 'before':
                result['qualification'] = evaluate(backend, requests, references, checkpoint,
                    base/'calls', slot=slot, uuid=uuid)
                for row in result['qualification']['rows']:
                    row['method'] = 'disposable_S'
                result['qualification']['inherited_record_session'] = 'persistent_prefix_withholding_serving_qualification'
                write_json(base/'calls/RESULT.json',result['qualification'])
        result['graph_and_physical_work'] = backend._call('work_receipt')
        result['gpu_memory'] = gpu_memory(uuid)
        result['status'] = 'passed'
    except BaseException as error:
        failure = error
        result.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        write_json(base/'RESULT.json', result)
        try:
            close_backend(backend, base)
        except BaseException as error:
            result.update(status='failed', cleanup_error=repr(error))
            write_json(base/'RESULT.json', result)
            if failure is None:
                raise
            failure.add_note('Serving cleanup also failed: '+repr(error))
    return result


def run(plan, kind, slot):
    uuid = plan['devices'][slot]
    if os.environ.get('CUDA_VISIBLE_DEVICES') != uuid:
        raise ValueError('Qualification requires its declared physical GPU')
    base = Path(plan['root'])/'qualification'/(kind+'_'+slot)
    base.mkdir(parents=True, exist_ok=False)
    receipt = dict(status='running', physical_gpu_uuid=uuid,
        execution_commit=plan['execution_commit'], started_epoch=time.time())
    monitor = HostMemorySample()
    write_json(base/'STATUS.json', receipt)
    try:
        with occupied_card(plan['root'], kind+'_'+slot, uuid), monitor:
            boundary(plan)
            if kind == 'training':
                from .qualification import run_training_qualification
                receipt['result'] = run_training_qualification(plan, slot)
            else:
                receipt['result'] = serving(plan, slot, base)
            receipt['postflight'] = boundary(plan, compiler=True)
            receipt['status'] = 'passed'
    except BaseException as error:
        receipt.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        receipt.update(wall_s=time.time()-receipt['started_epoch'],
            sampled_process_tree_peak_rss_bytes=monitor.maximum_bytes,
            rss_sample_interval_s=.25, rss_not_continuous_peak=True)
        write_json(base/'STATUS.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--kind', choices=('training', 'serving'), required=True)
    parser.add_argument('--slot', choices=('A', 'B'), required=True)
    args = parser.parse_args()
    install_interrupt_handlers()
    run(read(args.plan), args.kind, args.slot)


if __name__ == '__main__':
    main()
