"""Opt-in scheduler RPCs on the established SGLang process boundary.

The ordinary backend/worker remain the default. Only this experiment selects
this entry point; its scheduler retains all existing verification and cache
hooks. Profiling happens at idle, before the next request flushes its caches.
"""

from pathlib import Path
import time

from online_feedback import sglang_worker, sglang_hooks
from online_feedback.storage import write_json


def controller(scheduler):
    from .native_routes import DepthRoutes

    if not hasattr(scheduler, "depth_routes"):
        scheduler.depth_routes = DepthRoutes(scheduler.draft_worker, seed=17)
    return scheduler.depth_routes


def depth_mode(self, mode, outputfile, audit=False):
    sglang_hooks.strict_idle(self)
    routes = controller(self)
    routes.audit_output_path = Path(outputfile).with_suffix('.rounds.json')
    routes.set_mode(mode, audit=audit)
    result = routes.status()
    write_json(outputfile, result)
    return result


def depth_profile(self, record, prompt_ordinal, outputfile):
    sglang_hooks.strict_idle(self)
    routes = controller(self)
    return routes.profile_completed_request(
        record, prompt_ordinal=prompt_ordinal, output_path=Path(outputfile),
    )


def depth_status(self, outputfile):
    sglang_hooks.strict_idle(self)
    result = controller(self).status()
    write_json(outputfile, result)
    return result


def scheduler_process(*args, **kwargs):
    sglang_hooks.install()
    from sglang.srt.managers.scheduler import Scheduler, run_scheduler_process

    for function in (depth_mode, depth_profile, depth_status):
        setattr(Scheduler, function.__name__, function)
    return run_scheduler_process(*args, **kwargs)


class DepthWorker(sglang_worker.NativeWorker):
    def __init__(self, settings, root):
        # NativeWorker resolves this named entry point before spawning its
        # scheduler. No author source or installed runtime file is overwritten.
        sglang_hooks.scheduler_process = scheduler_process
        super().__init__(settings, root)

    def depth_rpc(self, request):
        name = request["name"]
        if name not in ("depth_mode", "depth_profile", "depth_status"):
            raise ValueError("Unknown depth qualification operation")
        started = time.perf_counter()
        self.engine.collective_rpc(name, **request["arguments"])
        import json

        result = json.loads(Path(request["arguments"]["outputfile"]).read_text())
        return dict(result=result, rpc_wall_s=time.perf_counter() - started)


def main():
    """Reuse worker startup/transport, adding one experiment-specific operation."""
    import argparse
    import json
    from multiprocessing.connection import Listener
    import traceback

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    worker = None
    try:
        started = time.perf_counter()
        worker = DepthWorker(json.loads(args.settings.read_text()), args.output)
        with Listener(("127.0.0.1", 0), authkey=b"dflash-online-local") as listener:
            write_json(args.output / "ready.json", dict(
                address=list(listener.address), loading_s=time.perf_counter() - started,
                status=worker.status(),
            ))
            with listener.accept() as connection:
                methods = dict(generate=worker.generate, publish=worker.publish,
                               check_weights=worker.check_weights, depth_rpc=worker.depth_rpc)
                while True:
                    request = connection.recv()
                    if request["operation"] == "close":
                        connection.send(dict(ok=True, result=None))
                        break
                    try:
                        if worker.blocked:
                            raise RuntimeError("Native worker blocked after an earlier failure")
                        result = methods[request["operation"]](request)
                        connection.send(dict(ok=True, result=result))
                    except Exception as error:
                        worker.blocked = True
                        failure = dict(type=type(error).__name__, message=str(error),
                                       traceback=traceback.format_exc(), operation=request["operation"])
                        write_json(args.output / "last_failure.json", failure)
                        connection.send(dict(ok=False, error=failure))
    finally:
        if worker is not None:
            worker.close()


if __name__ == "__main__":
    main()
