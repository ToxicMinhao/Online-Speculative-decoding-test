"""One-block exit atop PR34's qualified native state constructor.

The original output cap stays on the restored request. We stop the diagnostic
driver after one native round; we never shorten the request's sampling budget.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import time

from experiments.verification_transition import native_worker as base
from online_feedback import sglang_hooks
from online_feedback.storage import write_json


def transition_one_block(self, handle, checkpoint, outputfile):
    import torch

    sglang_hooks.strict_idle(self)
    if base._SESSION is not None:
        raise ValueError("Another native capture is active")
    matches = [
        (s, v)
        for s in base._SESSIONS
        for v in s.successors.values()
        if v.handle == handle
    ]
    if len(matches) != 1 or str(checkpoint) != str(
        self._online_publication["checkpoint_path"]
    ):
        raise ValueError("Missing live successor or wrong published checkpoint")
    session, state = matches[0]
    if state.terminal:
        raise ValueError("A sealed later start must have a nonterminal preceding round")
    base._SESSION = session
    request = session.request
    recorder = sglang_hooks.RequestCapture(
        request["prompt_ids"], str(Path(outputfile).with_suffix(".trace.json")), False
    )
    recorder.prefill_rows = len(request["prompt_ids"])
    recorder.covered_ids = request["prompt_ids"] + state.output_ids[:-1]
    recorder.teacher_ids = state.output_ids
    recorder.block_size = 16
    sglang_hooks._ACTIVE = recorder
    session.capture_enabled = False
    session.install()
    started = time.perf_counter()
    failure = None
    try:
        with torch.inference_mode():
            batch = session.rebuild(state, checkpoint)
            base._BUDGET.check_time()
            session.worker.forward_batch_generation(batch)
            torch.cuda.synchronize()
        req = batch.reqs[0]
        response = list(req.output_ids)
        observation = recorder.finish(response)
        receipt = dict(
            status="completed",
            blocks=observation["blocks"],
            response_ids=response,
            output_matches_canonical_prefix=response
            == request["canonical_response_ids"][: len(response)],
            native_finished=bool(req.finished()),
            original_cap=state.original_max_new_tokens,
            original_remaining_budget=state.original_max_new_tokens
            - len(state.output_ids),
            cap_unchanged=req.sampling_params.max_new_tokens
            == state.original_max_new_tokens,
            reconstruction=session.reconstruction,
            wall_s=time.perf_counter() - started,
        )
        write_json(outputfile, receipt)
        if (
            len(observation["blocks"]) != 1
            or not receipt["output_matches_canonical_prefix"]
            or not receipt["cap_unchanged"]
        ):
            raise ValueError("One-block native output/cap contract failed")
        if receipt["native_finished"] != (
            len(response) == len(request["canonical_response_ids"])
        ):
            raise ValueError("One-block termination differs from canonical boundary")
        return receipt
    except BaseException as error:
        failure = error
        write_json(
            outputfile,
            dict(
                status="failed",
                error=repr(error),
                completed_trace=recorder.trace,
                emitted_prefix=recorder.teacher_ids,
                reconstruction=session.reconstruction,
                wall_s=time.perf_counter() - started,
            ),
        )
        raise
    finally:
        session.close()
        base._SESSION = None
        sglang_hooks._ACTIVE = None
        try:
            sglang_hooks.flush_idle(self)
        except BaseException as error:
            if failure is None:
                raise
            saved = json.loads(Path(outputfile).read_text())
            saved["cleanup_error"] = repr(error)
            write_json(outputfile, saved)


def scheduler_process(*args, **kwargs):
    base.install_accounting()
    sglang_hooks.install()
    from sglang.srt.managers.scheduler import Scheduler, run_scheduler_process

    for fn in (
        base.transition_begin,
        base.transition_walk_finished,
        base.transition_release,
        transition_one_block,
    ):
        setattr(Scheduler, fn.__name__, sglang_hooks._rpc(fn))
    return run_scheduler_process(*args, **kwargs)


class HeadroomBackend(base.NativeTransitionBackend):
    def __init__(
        self, target_path, draft_path, *, worker_python, output_root, budget_connection
    ):
        from online_feedback.sglang_backend import SGLangDFlashBackend

        key = "DFLASH_TRANSITION_BUDGET"
        previous = os.environ.get(key)
        os.environ[key] = json.dumps(budget_connection)
        try:
            self.backend = SGLangDFlashBackend(
                target_path,
                draft_path,
                worker_python=worker_python,
                output_root=output_root,
                serving_gpu="0",
                context_limit=4096,
                block_size=16,
                worker_module="experiments.longer_run_headroom.native_worker",
            )
        finally:
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous
        self.loading_s = self.backend.loading_s
        self.serial = 0

    def select_checkpoint(self, checkpoint):
        if self.backend.checkpoint_id != str(checkpoint):
            self.backend.publish(checkpoint)

    def one_block(self, successor, checkpoint):
        self.select_checkpoint(checkpoint)
        return self._rpc(
            "transition_one_block",
            handle=successor["handle"],
            checkpoint=str(checkpoint),
        )


if __name__ == "__main__":
    base.scheduler_process = scheduler_process
    base.main()
