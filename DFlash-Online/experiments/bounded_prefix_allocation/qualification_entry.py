"""Disposable training gate and eight native calls, all charged to one ledger."""
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
from .runtime import WorkLedger, boundary, occupied_card, read


def serving(plan, slot, base):
    from online_feedback.sglang_backend import SGLangDFlashBackend
    uuid = plan['devices'][slot]
    checkpoint = (plan['origin_export'] if slot == 'B' else
                  read(Path(plan['root'])/'qualification/training/STATUS.json')['export'])
    requests, references = read(plan['warmups'])['requests'], read(plan['warmup_references'])
    ledger = WorkLedger(plan['root'], 'qualification_serving_'+slot)
    result = dict(status='running', checkpoint=checkpoint, role='native_K6' if slot == 'B' else 'disposable_active_C',
                  execution_commit=plan['execution_commit'])
    backend = None
    write_json(base/'RESULT.json', result)
    try:
        began = time.perf_counter()
        backend = SGLangDFlashBackend(plan['target'], checkpoint,
            worker_python=plan['worker_python'], output_root=base/'runtime', serving_gpu=uuid,
            context_limit=4096, block_size=16, reasoning='off',
            worker_module='experiments.persistent_prefix_withholding.worker')
        result['loading_s'] = time.perf_counter()-began
        result['gpu_memory_after_loading'] = gpu_memory(uuid)
        ledger.wrap(backend, 'generate', 'qualification_generation')
        for phase in ('before', 'after'):
            check = backend.check_weights(checkpoint)
            errors = weight_check_errors(check, str(checkpoint))
            write_json(base/('WEIGHTS_'+phase.upper()+'.json'), dict(check=check, errors=errors))
            if errors:
                raise ValueError('Serving checkpoint mismatch: '+repr(errors))
            if phase == 'before':
                result['qualification'] = evaluate(backend, requests, references, checkpoint,
                    base/'calls', slot=slot, uuid=uuid)
                for row in result['qualification']['rows']:
                    row['method'] = 'K6' if slot == 'B' else 'disposable_C'
        result['graph_and_physical_work'] = backend._call('work_receipt')
        result['gpu_memory_after_generations'] = gpu_memory(uuid)
        result['status'] = 'passed'
    except BaseException as error:
        result.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        if backend is not None:
            try:
                backend.close()
            except BaseException as error:
                result.update(status='failed', cleanup_error=repr(error))
        write_json(base/'RESULT.json', result)
    if result.get('cleanup_error'):
        raise RuntimeError('Serving process did not close cleanly')
    return result


def run(plan, kind, slot):
    uuid = plan['devices'][slot]
    if os.environ.get('CUDA_VISIBLE_DEVICES') != uuid:
        raise ValueError('Qualification must use its declared physical GPU')
    name = kind+'_'+slot
    base = Path(plan['root'])/'qualification'/name
    base.mkdir(parents=True, exist_ok=False)
    receipt = dict(status='running', physical_gpu_uuid=uuid, execution_commit=plan['execution_commit'],
                   started_epoch=time.time())
    monitor = HostMemorySample()
    write_json(base/'STATUS.json', receipt)
    try:
        with occupied_card(plan['root'], name, uuid), monitor:
            boundary(plan)
            if kind == 'training':
                if slot != 'A':
                    raise ValueError('Only one disposable training trajectory is planned')
                from .qualification import qualify
                receipt['result'] = qualify(plan)
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
    parser.add_argument('--kind', choices=('training','serving'), required=True)
    parser.add_argument('--slot', choices=('A','B'), required=True)
    args = parser.parse_args()
    install_interrupt_handlers()
    run(read(args.plan), args.kind, args.slot)


if __name__ == '__main__':
    main()
