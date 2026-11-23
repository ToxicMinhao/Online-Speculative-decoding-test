"""Synchronous requests to native SGLang DFlash/FA3 and its feedback worker.

SGLang owns drafting, verification, KV caches and termination. Graduate's
process-local hooks observe canonical rows; installed upstream files stay intact.
"""

from dataclasses import dataclass
from importlib.metadata import version
import inspect
import json
import os
from pathlib import Path


SGLANG_VERSION = "0.5.18"
CUDA_HOME = "/usr/local/cuda-13.0"
CAPTURE_BACKEND = "sglang_live_capture"


@dataclass(frozen=True)
class GenerationResult:
    response_ids: list[int]
    finish_reason: str
    trace: list[dict]
    capture: "SavedFeedback | None"


class SavedFeedback:
    """Save the worker's completed capture before another request replaces it."""

    def __init__(self, generator):
        self.generator = generator

    def save(self, record, path):
        generator = self.generator
        if generator._pending_feedback is not self:
            raise RuntimeError("This feedback capture has already been saved or closed.")
        path = Path(path).resolve()
        generator.engine.collective_rpc(
            "graduate_save_feedback", record=record.model_dump(mode="json"),
            outputfile=str(path),
        )
        preparation = json.loads(path.with_suffix(".json").read_text())
        generator.peak_cuda_bytes = max(generator.peak_cuda_bytes, preparation["peak_cuda_bytes"])
        generator._pending_feedback = None
        return preparation["label_binding"]


class SGLangGenerator:
    def __init__(self, engine, config, stop_token_ids, runtime, control_dir):
        self.engine = engine
        self.config = config
        self.stop_token_ids = stop_token_ids
        self.runtime = runtime
        self.peak_cuda_bytes = 0
        self.control_dir = Path(control_dir)
        self.control_dir.mkdir(parents=True, exist_ok=True)
        self._request_index = 0
        self._pending_feedback = None
        self._closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        if not self._closed:
            self._closed = True
            try:
                self.engine.shutdown()
            finally:
                self._pending_feedback = None

    def generate(self, prompt_ids, max_new_tokens, stop_token_ids, block_size, capture=True):
        if self._closed:
            raise RuntimeError("The native generation engine is closed.")
        if self._pending_feedback is not None:
            raise RuntimeError("Save the pending feedback before generating another request.")
        prompt, stops = list(prompt_ids), list(stop_token_ids)
        if (not prompt or not stops
                or any(type(token) is not int or token < 0 for token in prompt + stops)):
            raise ValueError("Generation requires exact nonnegative integer token IDs.")
        if type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer.")
        if len(prompt) + max_new_tokens > self.config.serving.context_limit:
            raise ValueError("Prompt plus output cap exceeds the full context limit.")
        if type(block_size) is not int or block_size != self.config.serving.block_size:
            raise ValueError("Request block_size differs from the native engine configuration.")

        trace_path = self.control_dir / f"{self._request_index:06d}.json"
        raw_output_path = self.control_dir / f"{self._request_index:06d}-engine.json"
        self._request_index += 1
        self.engine.collective_rpc("graduate_begin", prompt_ids=prompt, capture=capture)
        output = self.engine.generate(input_ids=prompt, sampling_params={
            "temperature": 0.0, "top_p": 1.0, "top_k": 1,
            "max_new_tokens": max_new_tokens, "stop_token_ids": stops, "ignore_eos": False,
        })
        # Persist the original response before any validity check can fail. Close
        # stops the owned engine; it never deletes these raw observations.
        raw_output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
        response_ids = output["output_ids"]
        native_reason = output["meta_info"]["finish_reason"]["type"]
        if (not response_ids or any(type(token) is not int or token < 0 for token in response_ids)
                or len(response_ids) > max_new_tokens
                or any(token in stops for token in response_ids[:-1])):
            raise ValueError("Native generation returned an invalid completed token sequence.")
        ended_at_eos = response_ids[-1] in stops
        if not ((native_reason == "stop" and ended_at_eos)
                or (native_reason == "length" and not ended_at_eos and len(response_ids) == max_new_tokens)):
            raise ValueError("Native termination disagrees with the exact EOS/output-cap boundary.")

        self.engine.collective_rpc(
            "graduate_finish", response_ids=response_ids, trace_path=str(trace_path),
        )
        observations = json.loads(trace_path.read_text())
        if observations["response_ids"] != response_ids:
            raise ValueError("Native capture and the engine's returned token IDs differ.")
        self.peak_cuda_bytes = max(self.peak_cuda_bytes, observations["peak_cuda_bytes"])
        self.runtime.update(observations.get("runtime", {}))
        saved = SavedFeedback(self) if capture else None
        self._pending_feedback = saved
        return GenerationResult(response_ids, "eos" if ended_at_eos else "length",
                                observations["trace"], saved)


def load_generator(config, control_dir):
    """Start one native serving worker; no training or dependency installation."""
    if version("sglang") != SGLANG_VERSION:
        raise ValueError(f"The local capture hooks require SGLang {SGLANG_VERSION}.")
    serving = config.serving
    if (serving.runtime != "sglang" or serving.attention_backend != "fa3"
            or not serving.greedy or serving.concurrency != 1 or not serving.cuda_graphs):
        raise ValueError("Collection requires native greedy SGLang/FA3, concurrency one and CUDA graphs.")
    target = json.loads((config.models.target / "config.json").read_text())
    draft = json.loads((config.models.draft / "config.json").read_text())
    if (target["model_type"] != "qwen3" or draft["architectures"] != ["DFlashDraftModel"]
            or draft["block_size"] != serving.block_size):
        raise ValueError("Native collection requires the configured Qwen3/original DFlash pair and block size.")
    stop_ids = json.loads((config.models.target / "generation_config.json").read_text())["eos_token_id"]
    stop_ids = [stop_ids] if isinstance(stop_ids, int) else list(stop_ids)

    # Both CLI and F5 compile against the CUDA 13 runtime already installed here,
    # instead of inheriting the older nvcc at the front of the shell PATH.
    compiler_home = os.environ.setdefault("CUDA_HOME", CUDA_HOME)
    os.environ.setdefault("CUDA_PATH", compiler_home)
    compiler_bin = str(Path(compiler_home) / "bin")
    if not (Path(compiler_bin) / "nvcc").is_file():
        raise FileNotFoundError(f"CUDA compiler is missing from {compiler_bin}.")
    os.environ["PATH"] = compiler_bin + os.pathsep + os.environ.get("PATH", "")

    from sglang.srt.entrypoints.engine import Engine
    from graduate.sglang_capture import run_scheduler_process

    class CollectionEngine(Engine):
        run_scheduler_process_func = staticmethod(run_scheduler_process)

    max_total_tokens = serving.context_limit + serving.block_size * serving.cuda_graph_max_batch_size
    runtime = {
        "backend": "sglang_fa3", "capture_backend": CAPTURE_BACKEND,
        "sglang_version": version("sglang"), "sglang_kernel_version": version("sglang-kernel"),
        "torch_version": version("torch"), "transformers_version": version("transformers"),
        "engine_source_file": inspect.getfile(Engine), "cuda_home": compiler_home,
        "memory_fraction": serving.memory_fraction, "max_total_tokens": max_total_tokens,
        "draft_block_size": serving.block_size,
        "target_attention_backend": "fa3", "draft_attention_backend": "fa3",
        "cuda_graph_max_batch_size": serving.cuda_graph_max_batch_size,
        "terminal_feature": "separate full-context native target forward, final row retained",
    }
    control_dir = Path(control_dir).resolve()
    control_dir.mkdir(parents=True, exist_ok=True)
    engine = CollectionEngine(
        model_path=str(config.models.target), speculative_draft_model_path=str(config.models.draft),
        speculative_algorithm="DFLASH", speculative_num_draft_tokens=serving.block_size,
        speculative_num_steps=1, speculative_eagle_topk=1,
        attention_backend="fa3", speculative_draft_attention_backend="fa3", dtype="bfloat16",
        context_length=serving.context_limit, tp_size=1, pp_size=1, max_running_requests=1,
        disable_overlap_schedule=True, disable_radix_cache=True, max_total_tokens=max_total_tokens,
        mem_fraction_static=serving.memory_fraction, chunked_prefill_size=serving.chunked_prefill_tokens,
        cuda_graph_backend_decode="full", cuda_graph_max_bs_decode=serving.cuda_graph_max_batch_size,
    )
    return SGLangGenerator(engine, config, stop_ids, runtime, control_dir)
