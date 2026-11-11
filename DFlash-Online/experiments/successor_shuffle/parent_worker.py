"""Untimed parent probe: execute normal prefill and exactly one native round.

The requested decoding cap stays 2,048. Only after the entire first native
round returns do we stop scheduling another round. The diagnostic stop is not
reported as a model EOS or a complete-response generation. Timed child serving
uses the unmodified worker, never this module.
"""

import functools

from online_feedback import sglang_hooks, sglang_worker


def stop_after_first_round(recorder, request, finish_factory):
    if len(recorder.trace) != 1:
        raise ValueError("Parent diagnostic must execute exactly one draft block")
    if not request.finished():
        request.finished_reason = finish_factory(len(request.output_ids))


def scheduler_process(*args, **kwargs):
    sglang_hooks.install()
    from sglang.srt.speculative.dflash_worker import DFlashWorker
    from sglang.srt.managers.schedule_batch import FINISH_LENGTH
    from sglang.srt.managers.scheduler import run_scheduler_process

    ordinary = DFlashWorker.forward_batch_generation

    @functools.wraps(ordinary)
    def first_round(worker, batch, **kw):
        recorder = sglang_hooks._ACTIVE
        live = recorder is not None
        prefill = batch.forward_mode.is_extend() or batch.is_extend_in_batch
        if live and not prefill and recorder.trace:
            raise ValueError("Refuse a second parent diagnostic block")
        result = ordinary(worker, batch, **kw)
        if live and not prefill:
            stop_after_first_round(recorder, batch.reqs[0], FINISH_LENGTH)
        return result

    DFlashWorker.forward_batch_generation = first_round
    return run_scheduler_process(*args, **kwargs)


if __name__ == "__main__":
    # NativeWorker imports this scheduler function at initialization. The patch
    # exists only in this separate diagnostic worker process.
    sglang_hooks.scheduler_process = scheduler_process
    sglang_worker.main()
