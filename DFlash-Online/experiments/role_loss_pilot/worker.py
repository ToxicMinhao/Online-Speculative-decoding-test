"""Native online worker plus one idle retained-feedback reconstruction RPC.

Serving, publication, teacher preparation and weight checks inherit NativeWorker
unchanged. The experiment scheduler installs only the existing online hooks and
the reconstruction operation; it installs no depth controller or decode wrapper.
"""
from __future__ import annotations

import argparse
import json
from multiprocessing.connection import Listener
from pathlib import Path
import time
import traceback

from online_feedback import sglang_hooks, sglang_worker
from online_feedback.storage import write_json


def roleloss_reconstruct(self, record, outputfile):
    from experiments.block_depth.reconstruct import reconstruct

    sglang_hooks.strict_idle(self)
    return reconstruct(self, record, outputfile)


def scheduler_process(*args, **kwargs):
    sglang_hooks.install()
    from sglang.srt.managers.scheduler import Scheduler, run_scheduler_process

    Scheduler.roleloss_reconstruct = roleloss_reconstruct
    return run_scheduler_process(*args, **kwargs)


class RoleLossWorker(sglang_worker.NativeWorker):
    def __init__(self, settings, root):
        # NativeWorker resolves this entry point before spawning its scheduler.
        # This assignment is confined to the opt-in worker process.
        sglang_hooks.scheduler_process = scheduler_process
        super().__init__(settings, root)

    def roleloss_reconstruct(self, request):
        path = Path(request['outputfile'])
        self.engine.collective_rpc('roleloss_reconstruct', record=request['record'],
                                   outputfile=str(path))
        # Return the reconstruction receipt directly, including its `record`.
        # The scheduler's persisted receipt is the authoritative RPC result.
        return json.loads(path.read_text())


def serve_connection(worker, connection, root):
    """Keep the native serialized transport and fail-closed operation policy."""
    methods = dict(generate=worker.generate, teacher=worker.teacher, publish=worker.publish,
                   check_weights=worker.check_weights,
                   roleloss_reconstruct=worker.roleloss_reconstruct)
    while True:
        request = connection.recv()
        if request['operation'] == 'close':
            connection.send(dict(ok=True, result=None))
            break
        try:
            if worker.blocked:
                raise RuntimeError('Native worker blocked after an earlier failure')
            result = methods[request['operation']](request)
            connection.send(dict(ok=True, result=result))
        except Exception as error:
            worker.blocked = True
            failure = dict(type=type(error).__name__, message=str(error),
                           traceback=traceback.format_exc(), operation=request['operation'])
            write_json(Path(root) / 'last_failure.json', failure)
            connection.send(dict(ok=False, error=failure))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    worker = None
    try:
        started = time.perf_counter()
        worker = RoleLossWorker(json.loads(args.settings.read_text()), args.output)
        with Listener(('127.0.0.1', 0), authkey=b'dflash-online-local') as listener:
            write_json(args.output / 'ready.json', dict(
                address=list(listener.address), loading_s=time.perf_counter() - started,
                status=worker.status(),
            ))
            with listener.accept() as connection:
                serve_connection(worker, connection, args.output)
    finally:
        if worker is not None:
            worker.close()


if __name__ == '__main__':
    main()
