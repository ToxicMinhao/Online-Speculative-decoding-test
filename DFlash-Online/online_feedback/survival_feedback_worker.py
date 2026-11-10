"""Isolated saved-path teacher worker with no answer-generation operation."""
from __future__ import annotations

import argparse
import functools
import json
from multiprocessing.connection import Listener
from pathlib import Path
import time
import traceback

from . import sglang_hooks, sglang_worker
from .storage import write_json


OPERATIONS = ('reconstruct', 'teacher', 'check_weights')
_STARTUP = {}
_STARTUP_ACTIVE = True


def install_startup_accounting():
    """Separate observed graph-setup/model forwards from request reconstruction."""
    from sglang.srt.models.dflash import DFlashDraftModel
    from sglang.srt.models.qwen3 import Qwen3ForCausalLM

    def observed(original, family):
        @functools.wraps(original)
        def forward(model, *args, **kwargs):
            if not _STARTUP_ACTIVE:
                return original(model, *args, **kwargs)
            batch = kwargs.get('forward_batch')
            if batch is None:
                batch = next((item for item in args if hasattr(item, 'forward_mode')), None)
            if batch is None:
                raise RuntimeError('Startup forward lacks its native batch for accounting')
            phase = 'verify' if batch.forward_mode.is_target_verify() else 'context'
            counts = _STARTUP.setdefault(family + '_' + phase, dict(attempted=0, completed=0))
            counts['attempted'] += 1
            result = original(model, *args, **kwargs)
            counts['completed'] += 1
            return result
        return forward

    DFlashDraftModel.forward = observed(DFlashDraftModel.forward, 'draft')
    Qwen3ForCausalLM.forward = observed(Qwen3ForCausalLM.forward, 'target')


def survival_startup(self, outputfile):
    global _STARTUP_ACTIVE
    sglang_hooks.strict_idle(self)
    _STARTUP_ACTIVE = False
    result = dict(status='passed', observed_python_model_calls=_STARTUP,
        scope='engine_initialization_before_first_RPC; separate from saved-path target forwards',
        new_generations=0, startup_is_not_qualification=True)
    write_json(outputfile, result)
    return result


def survival_reconstruct(self, record, outputfile):
    from experiments.survival_head.feedback_replay import reconstruct

    return reconstruct(self, record, outputfile)


def survival_teacher(self, input_ids, outputfile):
    """Keep ordinary terminal semantics, with durable forward accounting.

    Persist completion immediately after runner.forward returns, before feature
    checks and tensor persistence, so a later failure cannot erase consumed work.
    """
    path = Path(outputfile)
    work_path = path.with_suffix('.work.json')
    if path.exists() or path.with_suffix('.pt').exists() or work_path.exists():
        raise FileExistsError('Terminal work and tensors are immutable')
    runner = self.tp_worker.model_runner
    original = runner.forward
    work = dict(status='running', target_forward_attempts=0,
                target_forward_completions=0, input_rows=len(input_ids),
                retained_rows=1, final_feature_logits_used_for_labels=False)
    write_json(work_path, work)

    def counted(*args, **kwargs):
        if work['target_forward_attempts']:
            raise RuntimeError('Terminal preparation exceeded its one-forward allowance')
        work['target_forward_attempts'] += 1
        write_json(work_path, work)
        result = original(*args, **kwargs)
        work['target_forward_completions'] += 1
        write_json(work_path, work)
        return result

    runner.forward = counted
    try:
        result = sglang_hooks.online_teacher(self, input_ids=input_ids, outputfile=outputfile)
        if work['target_forward_completions'] != 1:
            raise RuntimeError('Terminal preparation did not execute exactly one target forward')
        work['status'] = 'passed'
        result['work_receipt_path'] = str(work_path)
        write_json(path, result)
        return result
    except BaseException as error:
        work.update(status='failed', error=repr(error))
        raise
    finally:
        runner.forward = original
        write_json(work_path, work)


def scheduler_process(*args, **kwargs):
    install_startup_accounting()
    sglang_hooks.install()
    from sglang.srt.managers.scheduler import Scheduler, run_scheduler_process

    for function in (survival_startup, survival_reconstruct, survival_teacher):
        setattr(Scheduler, function.__name__, sglang_hooks._rpc(function))
    return run_scheduler_process(*args, **kwargs)


class SurvivalFeedbackWorker(sglang_worker.NativeWorker):
    def __init__(self, settings, root):
        # This module runs in its own new process. Existing author/serving source
        # is unchanged; NativeWorker resolves this entry before engine startup.
        sglang_hooks.scheduler_process = scheduler_process
        super().__init__(settings, root)
        path = root / 'STARTUP_WORK.json'
        self.engine.collective_rpc('survival_startup', outputfile=str(path))
        self.startup = json.loads(path.read_text())

    def reconstruct(self, request):
        path = Path(request['outputfile'])
        self.engine.collective_rpc('survival_reconstruct', record=request['record'], outputfile=str(path))
        return json.loads(path.read_text())

    def teacher(self, request):
        path = Path(request['outputfile'])
        self.engine.collective_rpc('survival_teacher', input_ids=request['input_ids'], outputfile=str(path))
        return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    worker = None
    try:
        started = time.perf_counter()
        worker = SurvivalFeedbackWorker(json.loads(args.settings.read_text()), root)
        with Listener(('127.0.0.1', 0), authkey=b'dflash-online-local') as listener:
            write_json(root / 'ready.json', dict(address=list(listener.address),
                loading_s=time.perf_counter() - started, status=worker.status(),
                startup_work=worker.startup))
            with listener.accept() as connection:
                methods = {name: getattr(worker, name) for name in OPERATIONS}
                while True:
                    request = connection.recv()
                    if request['operation'] == 'close':
                        connection.send(dict(ok=True, result=None))
                        break
                    try:
                        if worker.blocked:
                            raise RuntimeError('Reconstruction worker blocked after an earlier failure')
                        if request['operation'] not in methods:
                            raise ValueError('This worker only reconstructs saved feedback; generation is unavailable')
                        result = methods[request['operation']](request)
                        connection.send(dict(ok=True, result=result))
                    except Exception as error:
                        worker.blocked = True
                        failure = dict(type=type(error).__name__, message=str(error),
                            traceback=traceback.format_exc(), operation=request['operation'])
                        write_json(root / 'last_failure.json', failure)
                        connection.send(dict(ok=False, error=failure))
    finally:
        if worker is not None:
            worker.close()


if __name__ == '__main__':
    main()
