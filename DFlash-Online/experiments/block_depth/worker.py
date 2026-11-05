"""Opt-in scheduler RPCs on the established SGLang process boundary.

The ordinary backend/worker remain the default. Only this experiment selects
this entry point; its scheduler retains all existing verification and cache
hooks. Profiling happens at idle, before the next request flushes its caches.
"""

from pathlib import Path
import functools
import json
import time

from online_feedback import sglang_worker, sglang_hooks
from online_feedback.storage import write_json


def controller(scheduler):
    from .native_routes import DepthRoutes

    if not hasattr(scheduler, "depth_routes"):
        scheduler.depth_routes = DepthRoutes(scheduler.draft_worker, seed=17)
        scheduler.draft_worker._depth_routes = scheduler.depth_routes
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



def depth_load(self, sidecar_path, gate_path, outputfile):
    import torch

    sglang_hooks.strict_idle(self)
    if Path(sidecar_path).suffix == ".safetensors":
        from safetensors.torch import load_file
        sidecar = load_file(sidecar_path, device="cpu")
    else:
        sidecar = torch.load(sidecar_path, map_location="cpu", weights_only=True)
    gate = json.loads(Path(gate_path).read_text()) if gate_path is not None else None
    result = controller(self).load_fitted(sidecar, gate)
    write_json(outputfile, result)
    return result


def depth_reconstruct(self, record, outputfile):
    from .reconstruct import reconstruct

    return reconstruct(self, record, outputfile)


def depth_calibrate(self, record, outputfile):
    sglang_hooks.strict_idle(self)
    return controller(self).calibrate_completed_request(record, output_path=Path(outputfile))


class DecodeMeter:
    """Shared lightweight round accounting; only GATE retains proposal traces.

    GATE synchronizes complete round work, including trailing committed-KV work,
    to calibrate kappa. FINAL adds no synchronization: its subcounter is host
    submission wall time, while complete Engine.generate latency remains primary.
    """

    def __init__(self, prompt_ids, mode, trace_mode, outputfile):
        if trace_mode not in {"none", "gate"}:
            raise ValueError("Only GATE may retain per-round proposal evidence")
        self.prompt_ids, self.mode = list(prompt_ids), mode
        self.trace_mode, self.outputfile = trace_mode, Path(outputfile)
        self.calls = self.accepted = self.progress = 0
        self.shallow = self.deep = 0
        self.decode_wall_s = 0.0
        self.trace = []
        self.cache_maintenance_s = 0.0

    def record(self, worker, req, before_length, elapsed, result):
        progress = len(req.output_ids) - before_length
        accepted = int(result.num_accepted_tokens)
        if not 1 <= progress <= 16 or accepted != max(0, progress - 1):
            raise RuntimeError("Native round commit/accepted counters disagree")
        depth = 5 if self.mode == "FIVE" else worker._depth_routes.last_route
        if depth not in (3, 5):
            raise RuntimeError("Native live route did not record its selected depth")
        self.calls += 1
        self.progress += progress
        self.accepted += accepted
        self.decode_wall_s += elapsed
        self.shallow += depth == 3
        self.deep += depth == 5
        if self.trace_mode == "gate":
            # The native rectangular draft buffer is not changed by verification.
            # Copy after stopping the round timer; no copy exists in FINAL.
            candidates = worker._draft_block_tokens_buf[0].detach().cpu().tolist()
            positions = worker._draft_block_positions_buf[0].detach().cpu().tolist()
            start = before_length - 1
            anchor = len(self.prompt_ids) + start
            if len(candidates) != 16 or positions != list(range(anchor, anchor + 16)):
                raise RuntimeError("GATE trace changed its native block geometry")
            if candidates[0] != req.output_ids[start]:
                raise RuntimeError("GATE trace seed differs from canonical response")
            self.trace.append(dict(output_start=start, positions=positions,
                                   draft_tokens=candidates[1:], committed_progress=progress,
                                   accepted_draft_tokens=accepted))

    def result(self):
        result = dict(status="passed", behavior=self.mode, verification_calls=self.calls,
                      observed_rounds=self.calls, accepted_draft_tokens=self.accepted,
                      committed_progress=self.progress, decode_round_wall_s=self.decode_wall_s,
                      shallow_rounds=self.shallow, deep_rounds=self.deep,
                      cache_maintenance_s=self.cache_maintenance_s,
                      feedback_capture=False, trace_instrumentation=self.trace_mode == "gate",
                      accepted_draft_tokens_convention="native max(committed progress - 1, 0) per round; "
                          "may undercount a correctly accepted terminal EOS/cap token",
                      decode_timer=("synchronized native worker wall including trailing committed KV; "
                          "excludes prefill, cleanup and GATE trace copies" if self.trace_mode == "gate" else
                          "native worker host wall without additional synchronization; may omit trailing "
                          "asynchronous KV completion; complete request timer remains primary"))
        if self.trace_mode == "gate":
            result["trace"] = self.trace
        return result


def observe_decode(original):
    """Wrap the worker equally for FIVE/THREE/ADAPTIVE without changing model work."""
    @functools.wraps(original)
    def measured(worker, batch, **kwargs):
        meter = getattr(worker, "_depth_meter", None)
        if meter is None:
            return original(worker, batch, **kwargs)
        if batch.forward_mode.is_idle():
            return original(worker, batch, **kwargs)
        if len(batch.reqs) != 1:
            raise RuntimeError("Depth serving requires concurrency one")
        req = batch.reqs[0]
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            if list(req.origin_input_ids) != meter.prompt_ids:
                raise RuntimeError("Depth serving prompt changed")
            return original(worker, batch, **kwargs)
        before = len(req.output_ids)
        if meter.trace_mode == "gate":
            import torch
            torch.cuda.synchronize()
        started = time.perf_counter()
        try:
            result = original(worker, batch, **kwargs)
            if meter.trace_mode == "gate":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            meter.record(worker, req, before, elapsed, result)
            return result
        except BaseException as error:
            partial = meter.result()
            partial.update(status="failed", error=repr(error), active_round_unknown=True)
            write_json(meter.outputfile, partial)
            raise
    return measured


def depth_begin_lean(self, prompt_ids, mode, trace_mode, outputfile):
    sglang_hooks.strict_idle(self)
    if sglang_hooks._ACTIVE is not None or getattr(self.draft_worker, "_depth_meter", None) is not None:
        raise RuntimeError("A prior capture or lean request is unfinished")
    if Path(outputfile).exists():
        raise FileExistsError("Lean execution evidence already exists")
    routes = controller(self)
    routes.set_mode(mode, audit=False, record_routes=False)
    meter = DecodeMeter(prompt_ids, mode, trace_mode, outputfile)
    meter.cache_maintenance_s = sglang_hooks.flush_idle(self)
    self.draft_worker._depth_meter = meter


def depth_finish_lean(self, output_ids):
    sglang_hooks.strict_idle(self)
    meter = self.draft_worker._depth_meter
    result = meter.result()
    result["output_ids"] = list(output_ids)
    result["output_length"] = len(output_ids)
    valid = meter.progress == max(0, len(output_ids) - 1)
    result["status"] = "passed" if valid else "failed"
    write_json(meter.outputfile, result)
    self.draft_worker._depth_meter = None
    if not valid:
        raise RuntimeError("Lean round progress does not account for the canonical output")
    return result

def scheduler_process(*args, **kwargs):
    sglang_hooks.install()
    from sglang.srt.managers.scheduler import Scheduler, run_scheduler_process

    from sglang.srt.speculative.dflash_worker import DFlashWorker
    DFlashWorker.forward_batch_generation = observe_decode(DFlashWorker.forward_batch_generation)
    for function in (depth_mode, depth_profile, depth_status, depth_load, depth_calibrate,
                     depth_begin_lean, depth_finish_lean, depth_reconstruct):
        setattr(Scheduler, function.__name__, function)
    return run_scheduler_process(*args, **kwargs)


class DepthWorker(sglang_worker.NativeWorker):
    def __init__(self, settings, root):
        # NativeWorker resolves this named entry point before spawning its
        # scheduler. No author source or installed runtime file is overwritten.
        sglang_hooks.scheduler_process = scheduler_process
        super().__init__(settings, root)

    def generate_lean(self, request):
        path = Path(request["trace_path"])
        started = time.perf_counter()
        self.engine.collective_rpc("depth_begin_lean", prompt_ids=request["prompt_ids"],
                                   mode=request["mode"], trace_mode=request.get("trace_mode", "none"),
                                   outputfile=str(path))
        generation_started = time.perf_counter()
        result = self.engine.generate(input_ids=request["prompt_ids"], sampling_params={
            "temperature": 0, "top_p": 1, "top_k": 1,
            "max_new_tokens": request["max_new_tokens"], "ignore_eos": False,
            "stop_token_ids": request["stop_token_ids"]})
        native_s = time.perf_counter() - generation_started
        self.engine.collective_rpc("depth_finish_lean", output_ids=result["output_ids"])
        latency = time.perf_counter() - started
        observation = json.loads(path.read_text())
        return dict(output_ids=result["output_ids"], meta_info=result["meta_info"],
                    native_generate_s=native_s, request_latency_s=latency, trace=observation)

    def depth_rpc(self, request):
        name = request["name"]
        if name not in ("depth_mode", "depth_profile", "depth_status", "depth_load", "depth_calibrate", "depth_reconstruct"):
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
                methods = dict(generate=worker.generate, generate_lean=worker.generate_lean,
                               teacher=worker.teacher, publish=worker.publish,
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
