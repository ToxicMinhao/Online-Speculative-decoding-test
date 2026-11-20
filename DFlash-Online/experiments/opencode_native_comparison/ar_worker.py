"""Genuine target-only SGLang worker; no DFlash capture or verification hooks.

Only initialization counting and idle/cache inspection wrap native execution.
The Engine's supported flush_cache API implements cold-cache requests.
"""
from __future__ import annotations

import argparse
import functools
import json
from multiprocessing.connection import Listener
import os
from pathlib import Path
import time
import traceback

from online_feedback.storage import write_json

_INITIALIZATION = []
_WORK_PATH_ENV = 'DFLASH_CODE_AR_WORK_RECEIPT'


def observe_capture(original, persist):
    """Count existing graph warmup/capture calls without adding a model call."""
    @functools.wraps(original)
    def capture(graph, bs, forward, *args, **kwargs):
        row = dict(kind='graph_capture', batch_size=int(bs),
                   draft=bool(graph.model_runner.is_draft_worker),
                   tokens_per_batch=int(kwargs.get('num_tokens_per_bs', graph.num_tokens_per_bs)),
                   status='started', model_forward_calls=0,
                   model_forward_completions=0, model_forward_failures=0)
        _INITIALIZATION.append(row)
        started = time.perf_counter()
        persist()

        @functools.wraps(forward)
        def counted(*inputs, **options):
            row['model_forward_calls'] += 1
            persist()
            try:
                value = forward(*inputs, **options)
                row['model_forward_completions'] += 1
                return value
            except BaseException:
                row['model_forward_failures'] += 1
                raise
            finally:
                persist()

        try:
            if row['draft'] or bs > 8 or len(_INITIALIZATION) > 408:
                raise RuntimeError('AR graph initialization violates the declared engine bound')
            result = original(graph, bs, counted, *args, **kwargs)
            if row['model_forward_calls'] != 3:
                raise RuntimeError('Expected two graph warmup forwards and one capture forward')
            row['status'] = 'completed'
            return result
        except BaseException as error:
            row.update(status='failed', error=repr(error))
            raise
        finally:
            row['wall_s'] = time.perf_counter() - started
            persist()
    return capture


def cache_state(scheduler):
    """Inspection supports native overlapping AR scheduling without altering it."""
    return dict(
        idle=bool(scheduler._is_no_request() and not scheduler.waiting_queue
                  and scheduler.chunked_req is None
                  and not scheduler.grammar_manager.grammar_queue),
        cached_tokens=int(scheduler.tree_cache.total_size()),
        speculation_disabled=bool(scheduler.spec_algorithm.is_none()),
        draft_worker_absent=scheduler.draft_worker is None,
        overlap_enabled=bool(scheduler.enable_overlap))


def scheduler_process(*args, **kwargs):
    """Run the normal scheduler with generic accounting, never install DFlash hooks."""
    from sglang.srt.managers.scheduler import Scheduler, run_scheduler_process
    from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner

    def persist():
        write_json(Path(os.environ[_WORK_PATH_ENV]), dict(
            initialization=_INITIALIZATION, extra_model_calls=0,
            scope='Native target-only graph initialization; no DFlash observer'))

    def inspect_cache(scheduler, outputfile):
        state = cache_state(scheduler)
        write_json(outputfile, state)
        return True

    def work_receipt(scheduler, outputfile):
        import torch
        state = cache_state(scheduler)
        if not state['idle']:
            raise RuntimeError('AR work receipt requested before the scheduler became idle')
        write_json(outputfile, dict(
            initialization=_INITIALIZATION, extra_model_calls=0, state=state,
            peak_gpu_allocated_bytes=int(torch.cuda.max_memory_allocated()),
            peak_gpu_reserved_bytes=int(torch.cuda.max_memory_reserved()),
            scope='Native target-only graph initialization; no DFlash observer'))
        return True

    Scheduler.code_ar_cache_state = inspect_cache
    Scheduler.code_ar_work_receipt = work_receipt
    CudaGraphRunner.capture_one_batch_size = observe_capture(
        CudaGraphRunner.capture_one_batch_size, persist)
    persist()
    return run_scheduler_process(*args, **kwargs)


class ARWorker:
    def __init__(self, settings, root):
        from sglang.srt.entrypoints.engine import Engine

        if (settings.get('speculative_algorithm') is not None
                or settings.get('speculative_draft_model_path') is not None):
            raise ValueError('AR requires a target-only engine with speculation disabled')
        self.root = Path(root)
        self.blocked = False
        os.environ[_WORK_PATH_ENV] = str(self.root / 'PHYSICAL_WORK.json')
        Engine.run_scheduler_process_func = staticmethod(scheduler_process)
        self.engine = Engine(**settings)

    def _cache_state(self, path):
        self.engine.collective_rpc('code_ar_cache_state', outputfile=str(path))
        state = json.loads(Path(path).read_text())
        if not state['idle'] or not state['speculation_disabled'] or not state['draft_worker_absent']:
            raise RuntimeError('AR scheduler is not idle and genuinely target-only')
        return state

    def status(self):
        info = self.engine.get_server_info()
        keys = ('model_path', 'speculative_algorithm', 'speculative_draft_model_path',
                'context_length', 'dtype', 'attention_backend', 'max_running_requests',
                'disable_cuda_graph', 'disable_piecewise_cuda_graph',
                'disable_overlap_schedule', 'random_seed', 'version')
        return dict(engine={key: info[key] for key in keys},
                    cache=self._cache_state(self.root / 'startup_cache.json'))

    def generate(self, request):
        path = Path(request['trace_path'])
        cache_started = time.perf_counter()
        flush = self.engine.flush_cache()
        if flush.success is not True:
            raise RuntimeError('Native AR cache flush failed')
        before = self._cache_state(path.with_suffix('.cache.json'))
        if before['cached_tokens'] != 0:
            raise RuntimeError('AR cache remains populated after a successful native flush')
        cache_s = time.perf_counter() - cache_started
        started = time.perf_counter()
        result = self.engine.generate(input_ids=request['prompt_ids'], sampling_params={
            'temperature': 0, 'top_p': 1, 'top_k': 1,
            'max_new_tokens': request['max_new_tokens'], 'ignore_eos': False,
            'stop_token_ids': request['stop_token_ids']})
        native_s = time.perf_counter() - started
        # Save returned tokens before any post-generation correctness check.
        observation = dict(output_ids=result['output_ids'], meta_info=result['meta_info'],
                           native_generate_s=native_s, cache_maintenance_s=cache_s,
                           cache_before=before, speculation_disabled=True,
                           feedback_capture=False, speculative_observer=False)
        write_json(path, observation)
        after = self._cache_state(path.with_suffix('.finish.json'))
        observation['cache_after'] = after
        write_json(path, observation)
        return observation

    def work_receipt(self, request):
        path = self.root / 'PHYSICAL_WORK.json'
        self.engine.collective_rpc('code_ar_work_receipt', outputfile=str(path))
        return json.loads(path.read_text())

    def close(self):
        self.engine.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    worker = None
    try:
        started = time.perf_counter()
        worker = ARWorker(json.loads(args.settings.read_text()), args.output)
        with Listener(('127.0.0.1', 0), authkey=b'dflash-online-local') as listener:
            write_json(args.output / 'ready.json', dict(address=list(listener.address),
                loading_s=time.perf_counter() - started, status=worker.status(),
                work=worker.work_receipt({})))
            with listener.accept() as connection:
                methods = dict(generate=worker.generate, work_receipt=worker.work_receipt)
                while True:
                    request = connection.recv()
                    if request['operation'] == 'close':
                        connection.send(dict(ok=True, result=worker.work_receipt({})))
                        break
                    try:
                        if worker.blocked:
                            raise RuntimeError('AR worker is blocked after an earlier failure')
                        result = methods[request['operation']](request)
                        connection.send(dict(ok=True, result=result))
                    except BaseException as error:
                        worker.blocked = True
                        failure = dict(type=type(error).__name__, message=str(error),
                            traceback=traceback.format_exc(), operation=request['operation'])
                        write_json(args.output / 'last_failure.json', failure)
                        connection.send(dict(ok=False, error=failure))
    except BaseException as error:
        write_json(args.output / 'last_failure.json', dict(type=type(error).__name__,
            message=str(error), traceback=traceback.format_exc(),
            operation='initialization' if worker is None else 'worker_lifetime'))
        raise
    finally:
        if worker is not None:
            worker.close()


if __name__ == '__main__':
    main()
