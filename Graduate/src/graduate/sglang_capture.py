"""Request-local observation for pinned SGLang DFlash V2, installed in its worker.

Generation and verification remain native. The terminal feature uses one extra
full-context target forward; only its last row is retained, never its logits.
"""

from array import array
from copy import copy
from pathlib import Path
import time

import torch

from graduate.collection import FeedbackCapture
from graduate.feedback import ResponseRecord
from graduate.storage import write_json


_TEACHER = "_graduate_normalized_teacher"
_ACTIVE = None
_FORWARD = None
_INSTALLED = False
_original_logits = _original_trim = _original_target = _original_worker = None


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _ids(value):
    return value.detach().reshape(-1).cpu().tolist() if isinstance(value, torch.Tensor) else list(value)


def _take_pair(output):
    """Remove the private bridge without mutating a graph's stored dictionary."""
    info = output.customized_info or {}
    teacher = info.get(_TEACHER)
    output.customized_info = {key: value for key, value in info.items() if key != _TEACHER} or None
    auxiliary = output.hidden_states
    if teacher is not None and auxiliary is not None:
        teacher = teacher[:auxiliary.shape[0]]  # Graph outputs may include padded rows.
    return auxiliary, teacher


def _cpu_pair(auxiliary, teacher, rows):
    _require(auxiliary is not None and teacher is not None,
             "Native target did not expose auxiliary and normalized teacher features.")
    _require(auxiliary.ndim == teacher.ndim == 2 and auxiliary.shape[0] == teacher.shape[0] == rows
             and auxiliary.dtype == teacher.dtype == torch.bfloat16,
             "Native target feature shape or BF16 dtype changed.")
    return (auxiliary.detach().to("cpu", copy=True).contiguous(),
            teacher.detach().to("cpu", copy=True).contiguous())


def _logits_forward(self, input_ids, hidden_states, lm_head, logits_metadata,
                    aux_hidden_states=None, hidden_states_before_norm=None, **kwargs):
    output = _original_logits(self, input_ids, hidden_states, lm_head, logits_metadata,
                              aux_hidden_states, hidden_states_before_norm, **kwargs)
    # This bridge must also exist during graph capture, before a request is armed.
    if aux_hidden_states is not None and logits_metadata.capture_hidden_mode.is_full():
        output.customized_info = {**(output.customized_info or {}), _TEACHER: hidden_states.detach()}
    return output


def _trim_output(self, output):
    trimmed = _original_trim(self, output)
    trimmed.customized_info = output.customized_info
    return trimmed


def _target_forward(self, batch=None, forward_batch=None, **kwargs):
    observing = _FORWARD is not None and self is _FORWARD["worker"].target_worker
    if observing:
        actual = batch if batch is not None else forward_batch
        _FORWARD["tokens"] = _ids(actual.input_ids)
        if forward_batch is not None:
            _FORWARD["positions"] = _ids(forward_batch.positions)
    result = _original_target(self, batch=batch, forward_batch=forward_batch, **kwargs)
    result.logits_output = copy(result.logits_output)
    auxiliary, teacher = _take_pair(result.logits_output)
    if observing:
        _FORWARD["target_cuda_graph"] = result.can_run_cuda_graph
    if observing and _ACTIVE.capture:
        _FORWARD["features"] = _cpu_pair(auxiliary, teacher, len(_FORWARD["tokens"]))
    return result


def _worker_forward(self, batch, **kwargs):
    global _FORWARD
    if _ACTIVE is None or batch.forward_mode.is_idle():
        return _original_worker(self, batch, **kwargs)
    _require(batch.batch_size() == 1 and not self.use_compact_draft_cache,
             "Capture requires one request and full causal draft context.")
    _require(_ids(batch.reqs[0].origin_input_ids) == _ACTIVE.prompt_ids,
             "Armed prompt differs from the actual native request.")
    prefill = batch.forward_mode.is_extend() or batch.is_extend_in_batch
    observation = {"worker": self}
    if prefill:
        observation["start"] = int(batch.prefix_lens[0])
    _FORWARD = observation
    try:
        result = _original_worker(self, batch, **kwargs)
    finally:
        _FORWARD = None
    _require("tokens" in observation, "Native DFlash call did not expose its target forward.")
    if _ACTIVE.layer_ids is None:
        config = self.draft_model_runner.model_config.hf_config
        _ACTIVE.layer_ids = list(config.dflash_config["target_layer_ids"])
        for role, runner in (("target", self.target_worker.model_runner),
                             ("draft", self.draft_model_runner)):
            _ACTIVE.runtime[f"{role}_prefill_backend"] = runner.prefill_attention_backend_str
            _ACTIVE.runtime[f"{role}_decode_backend"] = runner.decode_attention_backend_str
            _ACTIVE.runtime[f"{role}_attention_class"] = type(runner.attn_backend).__name__
    if prefill:
        _ACTIVE.prefills.append({"start": observation["start"], "tokens": observation["tokens"],
                                 "first_token": _ids(result.next_token_ids)[0],
                                 "target_cuda_graph": observation["target_cuda_graph"],
                                 "features": observation.get("features")})
    else:
        _require(batch.sampling_info.is_all_greedy, "Capture supports native greedy verification only.")
        candidates = observation["tokens"]
        posterior = _ids(result.logits_output.next_token_logits.argmax(dim=-1))
        progress = _ids(result.accept_lens)[0]
        accepted_end = next((slot for slot in range(1, len(candidates))
                             if candidates[slot] != posterior[slot - 1]), len(candidates))
        _require(progress == accepted_end, "Native acceptance differs from its greedy posterior.")
        features = observation.get("features")
        if features is not None:
            # Drop rejected histories now; later finish clipping may discard more.
            features = tuple(value[:progress].clone() for value in features)
        _ACTIVE.blocks.append({"candidates": candidates, "posterior": posterior,
                               "positions": observation["positions"], "progress": progress,
                               "target_cuda_graph": observation["target_cuda_graph"],
                               "features": features})
    return result


class CaptureSession:
    def __init__(self, prompt_ids, capture):
        _require(type(capture) is bool and prompt_ids and
                 all(type(token) is int and token >= 0 for token in prompt_ids),
                 "Capture requires exact prompt IDs and an explicit capture flag.")
        self.prompt_ids = list(prompt_ids)
        self.capture = capture
        self.prefills, self.blocks = [], []
        self.layer_ids = None
        self.runtime = {}
        self.feedback = None
        self.finished = False

    def finish(self, response_ids, trace_path):
        _require(not self.finished and response_ids and
                 all(type(token) is int and token >= 0 for token in response_ids),
                 "Capture must finish once with exact nonempty output IDs.")
        # Preserve observations before validation, including failed reconciliation.
        document = {"response_ids": list(response_ids), "trace": [],
                    "runtime": self.runtime,
                    "target_graph_forward_flags": [row.get("target_cuda_graph")
                                                   for row in self.prefills + self.blocks],
                    "raw_blocks": [{key: value for key, value in block.items() if key != "features"}
                                   for block in self.blocks],
                    "peak_cuda_bytes": torch.cuda.max_memory_allocated()}
        write_json(Path(trace_path), document)
        covered = []
        for prefill in self.prefills:
            _require(prefill["start"] == len(covered), "Prefill feature positions have a gap or overlap.")
            covered.extend(prefill["tokens"])
        _require(covered == self.prompt_ids and self.prefills[-1]["first_token"] == response_ids[0],
                 "Native prefill does not explain the prompt and first emitted token.")
        if self.capture:
            self.feedback = FeedbackCapture(self.prompt_ids, self.layer_ids, "sglang_live_capture")
            auxiliary = torch.cat([chunk["features"][0] for chunk in self.prefills])
            teacher = torch.cat([chunk["features"][1] for chunk in self.prefills])
            self.feedback.prefill(response_ids[0], auxiliary, teacher)
        offset = 0
        for index, block in enumerate(self.blocks):
            candidates, posterior = block["candidates"], block["posterior"]
            progress = min(block["progress"], len(response_ids) - 1 - offset)
            _require(progress > 0 and (progress == block["progress"] or index == len(self.blocks) - 1),
                     "Native verification continues beyond the emitted response.")
            start = len(self.prompt_ids) + offset
            _require(block["positions"] == list(range(start, start + len(candidates)))
                     and candidates[:progress] == response_ids[offset:offset + progress]
                     and posterior[:progress] == response_ids[offset + 1:offset + progress + 1],
                     "Verification rows include a rejected or otherwise noncanonical history.")
            document["trace"].append({
                "output_start": offset, "verify_size": len(candidates),
                "anchor_token": candidates[0], "draft_tokens": candidates[1:],
                "target_argmax": posterior,
                "accepted_draft_tokens": min(block["progress"] - 1, progress),
                "committed_progress": progress, "committed_tokens": posterior[:progress],
            })
            if self.capture:
                # Raw acceptance was checked against the complete native block.
                # The shared collector sees only those canonical input rows.
                raw_progress = block["progress"]
                self.feedback.verify(candidates[:raw_progress], posterior[:raw_progress], progress,
                                     *block["features"])
            offset += progress
        _require(offset == len(response_ids) - 1, "Native verification coverage is incomplete.")
        self.finished = True
        self.prefills.clear()
        self.blocks.clear()
        write_json(Path(trace_path), document)


def _flush_idle(scheduler):
    _require(not scheduler.enable_overlap and scheduler.is_fully_idle(),
             "Collection RPC requires an idle scheduler with overlap disabled.")
    _require(scheduler.flush_cache() is True, "Native cache flush failed.")


def graduate_begin(self, prompt_ids, capture):
    global _ACTIVE
    _require(_ACTIVE is None, "Prior capture must be finished and saved before another request.")
    _flush_idle(self)
    _ACTIVE = CaptureSession(prompt_ids, capture)


def graduate_finish(self, response_ids, trace_path):
    global _ACTIVE
    _require(_ACTIVE is not None and self.is_fully_idle(), "No completed native request is ready.")
    _ACTIVE.finish(response_ids, trace_path)
    if not _ACTIVE.capture:
        _ACTIVE = None


def graduate_save_feedback(self, record, outputfile):
    """An extra full-prefix target forward supplies only the final feature row."""
    global _ACTIVE
    from sglang.srt.managers.overlap_utils import resolve_forward_inputs
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch
    from sglang.srt.sampling.sampling_params import SamplingParams
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    _require(_ACTIVE is not None and _ACTIVE.finished and _ACTIVE.capture,
             "Saving feedback requires a finished feature-capturing request.")
    response = ResponseRecord.model_validate(record)
    _require(list(response.prompt_ids) == _ACTIVE.prompt_ids
             and list(response.response_ids) == _ACTIVE.feedback.response_ids,
             "Feedback record differs from native captured outputs.")
    _require(self.server_args.tp_size == self.server_args.pp_size == 1,
             "Terminal feature collection requires the qualified single-rank target.")
    _flush_idle(self)
    runner = self.tp_worker.model_runner
    tokens = list(response.input_ids)
    started = time.perf_counter()
    try:
        req = Req(rid="graduate_terminal_feature", origin_input_text="",
                  origin_input_ids=array("q", tokens),
                  sampling_params=SamplingParams(temperature=0, max_new_tokens=1))
        req.init_next_round_input()
        req.set_extend_range(0, len(tokens))
        batch = ScheduleBatch.init_new(
            reqs=[req], req_to_token_pool=runner.req_to_token_pool,
            token_to_kv_pool_allocator=runner.token_to_kv_pool_allocator,
            tree_cache=self.tree_cache, model_config=runner.model_config,
            enable_overlap=False, spec_algorithm=SpeculativeAlgorithm.NONE,
        )
        batch.prepare_for_extend()
        resolve_forward_inputs(batch, None)  # Full prefill: no FutureMap lookup.
        forward = ForwardBatch.init_new(batch, runner, capture_hidden_mode=CaptureHiddenMode.FULL,
                                        return_hidden_states_before_norm=False)
        with torch.inference_mode():
            result = runner.forward(forward)
            auxiliary, teacher = _take_pair(copy(result.logits_output))
            auxiliary, teacher = _cpu_pair(auxiliary, teacher, len(tokens))
            _ACTIVE.feedback.final_token(tokens[-1], auxiliary[-1:], teacher[-1:])
        binding = _ACTIVE.feedback.save(response, Path(outputfile))
        receipt = {"label_binding": binding, "terminal_forward_count": 1,
                   "terminal_input_rows": len(tokens), "retained_terminal_rows": 1,
                   "teacher_forward_count": 1, "teacher_forward_input_rows": len(tokens),
                   "retained_feature_rows": 1, "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
                   "teacher_preparation_seconds": time.perf_counter() - started,
                   "final_feature_logits_used_for_labels": False}
        write_json(Path(outputfile).with_suffix(".json"), receipt)
    finally:
        # The synthetic forward is outside scheduler request ownership. Clear its
        # allocations even on failure, while the scheduler is demonstrably idle.
        _flush_idle(self)
    _ACTIVE = None


def install():
    global _INSTALLED, _original_logits, _original_trim, _original_target, _original_worker
    from sglang.srt.layers.logits_processor import LogitsProcessor
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import PrefillCudaGraphRunner
    from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

    _require(not _INSTALLED, "SGLang collection hooks are already installed.")
    _original_logits, _original_trim = LogitsProcessor.forward, PrefillCudaGraphRunner._trim_logits_output
    _original_target, _original_worker = TpModelWorker.forward_batch_generation, DFlashWorkerV2.forward_batch_generation
    LogitsProcessor.forward = _logits_forward
    PrefillCudaGraphRunner._trim_logits_output = _trim_output
    TpModelWorker.forward_batch_generation = _target_forward
    DFlashWorkerV2.forward_batch_generation = _worker_forward
    for function in (graduate_begin, graduate_finish, graduate_save_feedback):
        setattr(Scheduler, function.__name__, function)
    _INSTALLED = True


def run_scheduler_process(*args, **kwargs):
    install()
    from sglang.srt.managers.scheduler import run_scheduler_process as upstream
    return upstream(*args, **kwargs)
