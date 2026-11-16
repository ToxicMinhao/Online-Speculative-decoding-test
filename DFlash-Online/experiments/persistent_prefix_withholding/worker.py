"""Native serving worker with bounded graph-initialization accounting only."""
from __future__ import annotations

import argparse
from collections import Counter
import functools
import json
import os
from multiprocessing.connection import Listener
from pathlib import Path
import time
import traceback

from online_feedback import sglang_hooks, sglang_worker
from online_feedback.storage import write_json

_INITIALIZATION = []
_PHYSICAL = Counter()
_PEAK_MEMORY = {}


_WORK_PATH_ENV = "DFLASH_PERSISTENT_WORK_RECEIPT"


def _save_work(path):
    """The scheduler writes this before its control RPC becomes available."""
    write_json(path, dict(initialization=_INITIALIZATION,
        physical=dict(_PHYSICAL), **_PEAK_MEMORY, extra_model_calls=0,
        scope='Existing CUDA graph forward calls; graph replay counted by native request traces'))


def _observe_capture(original, persist):
    """Observe the existing routine while preserving its forward signature."""
    @functools.wraps(original)
    def capture(graph, bs, forward, *positional, **options):
        row = dict(kind='graph_capture', batch_size=int(bs),
                   draft=bool(graph.model_runner.is_draft_worker),
                   tokens_per_batch=int(options.get('num_tokens_per_bs', graph.num_tokens_per_bs)),
                   status='started', model_forward_calls=0,
                   model_forward_completions=0, model_forward_failures=0)
        _INITIALIZATION.append(row)
        began = time.perf_counter()
        persist()

        @functools.wraps(forward)
        def counted(*inputs, **settings):
            row['model_forward_calls'] += 1
            persist()
            try:
                value = forward(*inputs, **settings)
                row['model_forward_completions'] += 1
                return value
            except BaseException:
                row['model_forward_failures'] += 1
                raise
            finally:
                persist()

        try:
            if bs > 8 or len(_INITIALIZATION) > 408:
                raise RuntimeError('Installed finite graph-initialization bound exceeded')
            result = original(graph, bs, counted, *positional, **options)
            if row['model_forward_calls'] != 3:
                raise RuntimeError('Expected two graph warmup forwards and one capture forward')
            row['status'] = 'completed'
            return result
        except BaseException as error:
            row.update(status='failed', error=repr(error))
            raise
        finally:
            row['wall_s'] = time.perf_counter()-began
            persist()
    return capture


def scheduler_process(*args, **kwargs):
    """Observe installed initialization; do not change its shapes or calls."""
    from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner
    from sglang.srt.managers.scheduler import Scheduler, run_scheduler_process

    path = Path(os.environ[_WORK_PATH_ENV])
    persist = lambda: _save_work(path)
    persist()

    def work_receipt(scheduler, outputfile):
        import torch
        sglang_hooks.strict_idle(scheduler)
        _PEAK_MEMORY.update(
            peak_gpu_allocated_bytes=int(torch.cuda.max_memory_allocated()),
            peak_gpu_reserved_bytes=int(torch.cuda.max_memory_reserved()))
        _save_work(outputfile)
        return True

    CudaGraphRunner.capture_one_batch_size = _observe_capture(
        CudaGraphRunner.capture_one_batch_size, persist)
    Scheduler.persistent_work_receipt = work_receipt
    sglang_hooks.install()
    try:
        return run_scheduler_process(*args, **kwargs)
    finally:
        persist()


class Worker(sglang_worker.NativeWorker):
    def __init__(self, settings, root):
        sglang_hooks.scheduler_process = scheduler_process
        path = Path(root).absolute() / 'PHYSICAL_WORK.json'
        _save_work(path)
        previous_path = os.environ.get(_WORK_PATH_ENV)
        os.environ[_WORK_PATH_ENV] = str(path)
        try:
            super().__init__(settings, root)
            self.work_receipt({})
        finally:
            if previous_path is None:
                os.environ.pop(_WORK_PATH_ENV, None)
            else:
                os.environ[_WORK_PATH_ENV] = previous_path

    def work_receipt(self, request):
        path = self.root/'PHYSICAL_WORK.json'
        self.engine.collective_rpc('persistent_work_receipt', outputfile=str(path))
        return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    worker = None
    try:
        began = time.perf_counter()
        worker = Worker(json.loads(args.settings.read_text()), args.output)
        with Listener(('127.0.0.1', 0), authkey=b'dflash-online-local') as listener:
            write_json(args.output/'ready.json', dict(address=list(listener.address),
                loading_s=time.perf_counter()-began, status=worker.status()))
            with listener.accept() as connection:
                methods = dict(generate=worker.generate, teacher=worker.teacher,
                               publish=worker.publish, check_weights=worker.check_weights,
                               work_receipt=worker.work_receipt)
                while True:
                    request = connection.recv()
                    if request['operation'] == 'close':
                        receipt = worker.work_receipt({})
                        connection.send(dict(ok=True, result=receipt))
                        break
                    try:
                        if worker.blocked:
                            raise RuntimeError('Native worker is blocked after an earlier failure')
                        result = methods[request['operation']](request)
                        connection.send(dict(ok=True, result=result))
                    except BaseException as error:
                        worker.blocked = True
                        failure = dict(type=type(error).__name__, message=str(error),
                                       traceback=traceback.format_exc(), operation=request['operation'])
                        write_json(args.output/'last_failure.json', failure)
                        connection.send(dict(ok=False, error=failure))
    except BaseException as error:
        # Initialization may fail before Worker is assigned or RPC is reachable.
        # The scheduler-owned PHYSICAL_WORK.json remains separate and untouched.
        write_json(args.output/'last_failure.json', dict(type=type(error).__name__,
            message=str(error), traceback=traceback.format_exc(),
            operation='initialization' if worker is None else 'worker_lifetime'))
        raise
    finally:
        if worker is not None:
            worker.close()


if __name__ == '__main__':
    main()
