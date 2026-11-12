"""PR40 native reconstruction extended to the opening verification start.

All starts use the unchanged qualified constructor. Native prefill has already
ingested prompt features into draft KV at the opening boundary, so that context
is reconstructed through its original ingestion shapes too. Every start must
pass neutral K+6 proposal/commit parity.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from experiments.verification_transition import native_worker as base
from experiments.verification_transition.native import NativeSuccessors
from experiments.longer_run_headroom.native_worker import transition_one_block
from online_feedback import sglang_hooks
from online_feedback.storage import write_json


class CensusSuccessors(NativeSuccessors):
    def install(self):
        super().install()
        ordinary = self.worker.forward_batch_generation

        def forward(batch, **kwargs):
            prefill = batch.forward_mode.is_extend() or batch.is_extend_in_batch
            if self.capture_enabled and not prefill and self.round_index == 0:
                self.capture_opening(batch)
            return ordinary(batch, **kwargs)

        self.worker.forward_batch_generation = forward

    def capture(self, batch):
        # A full census owns many CPU KV snapshots within one request. Stop
        # explicitly before memory exhaustion; never drop expensive starts.
        if os.path.exists('/proc/meminfo'):
            available = next(int(line.split()[1]) * 1024 for line in
                             Path('/proc/meminfo').read_text().splitlines()
                             if line.startswith('MemAvailable:'))
            if available < 8 * 1024 ** 3:
                raise MemoryError('All-start snapshot capture reached the 8 GiB host reserve')
        if not batch.reqs[0].finished():
            super().capture(batch)

    def capture_opening(self, batch):
        req = batch.reqs[0]
        prefix = int(batch.seq_lens[0].item())
        if (len(batch.reqs) != 1 or len(req.output_ids) != 1
                or prefix != len(req.origin_input_ids)
                or list(req.origin_input_ids) != self.request['prompt_ids']
                or not self.ingestions
                or int(batch.spec_info.draft_seq_lens[0]) != prefix):
            raise ValueError('Opening state differs from native ingested prefill')
        super().capture(batch)


def transition_finish_capture(self, canonical_response_ids, canonical_finish_reason, outputfile):
    session = base._SESSION
    session.request['canonical_response_ids'] = list(canonical_response_ids)
    session.request['canonical_finish_reason'] = canonical_finish_reason
    return base.transition_walk_finished(self, outputfile)


def online_teacher(self, input_ids, outputfile):
    """Count the captured packet's terminal feature as preparation, not startup."""
    def forward():
        base._TARGET_ADMITTED += 1
        try:
            return sglang_hooks.online_teacher(self, input_ids, outputfile)
        finally:
            base._TARGET_ADMITTED -= 1
    def target():
        return base.run_counted(base._BUDGET, 'target_call', forward,
                                metadata=dict(operation='terminal_teacher_preparation', startup=False))
    return base.run_counted(base._BUDGET, 'context_forward', target,
                            metadata=dict(operation='terminal_teacher_preparation',
                                          tokens=len(input_ids), startup=False))


def scheduler_process(*args, **kwargs):
    base.NativeSuccessors = CensusSuccessors
    base.install_accounting()
    sglang_hooks.install()
    from sglang.srt.managers.scheduler import Scheduler, run_scheduler_process
    Scheduler.online_teacher = online_teacher
    for function in (base.transition_begin, transition_finish_capture, base.transition_release,
                     transition_one_block):
        setattr(Scheduler, function.__name__, sglang_hooks._rpc(function))
    return run_scheduler_process(*args, **kwargs)


class CensusBackend(base.NativeTransitionBackend):
    def __init__(self, target_path, draft_path, *, worker_python, output_root,
                 budget_connection):
        from online_feedback.sglang_backend import SGLangDFlashBackend
        key = 'DFLASH_TRANSITION_BUDGET'
        previous = os.environ.get(key)
        os.environ[key] = json.dumps(budget_connection)
        try:
            self.backend = SGLangDFlashBackend(
                target_path, draft_path, worker_python=worker_python,
                output_root=output_root, serving_gpu='0', context_limit=4096,
                block_size=16, worker_module='experiments.lost_repair.native_worker',
            )
        finally:
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous
        self.loading_s = self.backend.loading_s
        self.serial = 0

    def capture(self, request, checkpoint, folder):
        if request['stop_token_ids'] != self.backend.stop_tokens:
            raise ValueError('Sealed EOS IDs differ from native serving')
        self.backend.publish(checkpoint)
        native = dict(request, canonical_response_ids=[], canonical_finish_reason=None)
        self._rpc('transition_begin', request=native, checkpoint=str(checkpoint),
                  capture_rounds=list(range(1, request['max_new_tokens'] + 1)))
        record = self.backend.generate(
            request['prompt_ids'], max_new_tokens=request['max_new_tokens'],
            request_id=request['request_id'], session_id='lost_repair_event007',
            request_index=request['request_index'], purpose='serving',
            save_to=Path(folder) / 'K6_CAPTURE.json',
        )
        receipt = self._rpc('transition_finish_capture', canonical_response_ids=record['response_ids'],
                            canonical_finish_reason=record['finish_reason'])
        if receipt['native_rounds'] != record['metadata']['verification_calls']:
            raise ValueError('Captured rounds differ from native trace')
        return record, receipt['successors']

    def one_block(self, state, checkpoint):
        if self.backend.checkpoint_id != str(checkpoint):
            self.backend.publish(checkpoint)
        return self._rpc('transition_one_block', handle=state['handle'],
                         checkpoint=str(checkpoint))


if __name__ == '__main__':
    base.scheduler_process = scheduler_process
    base.main()
