"""CPU native-engine boundary tests; no CUDA imports, model loads or workers."""

from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from graduate.feedback import ResponseRecord
from graduate.generation import SGLangGenerator, load_generator


class EngineDouble:
    created = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []
        self.output = {"output_ids": [5, 6, 9], "meta_info": {"finish_reason": {"type": "stop"}}}
        self.trace = [{"accepted_draft_tokens": 1, "committed_progress": 2}]
        self.generate_error = None
        self.save_error = None
        self.shutdown_error = None
        self.capture_mismatch = False
        self.shutdown_count = 0
        self.created.append(self)

    def generate(self, **kwargs):
        self.calls.append(("generate", deepcopy(kwargs)))
        if self.generate_error:
            raise self.generate_error
        return deepcopy(self.output)

    def collective_rpc(self, method, **kwargs):
        self.calls.append((method, deepcopy(kwargs)))
        if method == "graduate_finish":
            Path(kwargs["trace_path"]).write_text(json.dumps({
                "response_ids": [8] if self.capture_mismatch else kwargs["response_ids"],
                "trace": self.trace, "peak_cuda_bytes": 1024,
                "runtime": {"target_attention_backend": "fa3"},
            }))
        elif method == "graduate_save_feedback":
            if self.save_error:
                raise self.save_error
            path = Path(kwargs["outputfile"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"CPU test packet")
            path.with_suffix(".json").write_text(json.dumps({
                "peak_cuda_bytes": 2048, "label_binding": {"record_id": "response"},
                "terminal_forward_count": 1, "terminal_input_rows": 5,
                "retained_terminal_rows": 1,
            }))

    def shutdown(self):
        self.shutdown_count += 1
        if self.shutdown_error:
            raise self.shutdown_error


def scheduler_double(*args, **kwargs):
    pass


def configuration(root):
    return SimpleNamespace(
        models=SimpleNamespace(target=root / "target", draft=root / "draft"),
        serving=SimpleNamespace(runtime="sglang", attention_backend="fa3", greedy=True,
                                concurrency=1, cuda_graphs=True, block_size=16,
                                context_limit=4096, memory_fraction=0.88,
                                chunked_prefill_tokens=8192, cuda_graph_max_batch_size=8),
    )


def response_record():
    return ResponseRecord(
        record_id="response", session_id="session", request_index=0, checkpoint_id="draft",
        purpose="serving", prompt_ids=[2, 3], response_ids=[5, 6, 9],
        finish_reason="eos", stop_token_ids=[9], max_new_tokens=3,
    )


class GenerationBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="graduate-native-boundary-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = configuration(self.root)
        self.engine = EngineDouble()
        self.generator = SGLangGenerator(self.engine, self.config, [9], {}, self.root / "native")
        self.addCleanup(self.generator.close)

    def generate(self, capture=True):
        return self.generator.generate([2, 3], 3, [9], 16, capture=capture)

    def test_exact_tokens_greedy_parameters_and_capture_rpc_order(self):
        prompt = [2, 3]
        result = self.generator.generate(prompt, 3, [9], 16)
        self.assertEqual(prompt, [2, 3])
        self.assertEqual(result.response_ids, [5, 6, 9])
        self.assertEqual(result.finish_reason, "eos")
        self.assertEqual(result.trace, self.engine.trace)
        self.assertEqual([name for name, _ in self.engine.calls],
                         ["graduate_begin", "generate", "graduate_finish"])
        self.assertEqual(self.engine.calls[0][1], {"prompt_ids": prompt, "capture": True})
        self.assertEqual(self.engine.calls[1][1], {
            "input_ids": prompt, "sampling_params": {
                "temperature": 0.0, "top_p": 1.0, "top_k": 1, "max_new_tokens": 3,
                "stop_token_ids": [9], "ignore_eos": False,
            },
        })
        self.assertEqual(self.generator.peak_cuda_bytes, 1024)
        binding = result.capture.save(response_record(), self.root / "features.pt")
        self.assertEqual(binding, {"record_id": "response"})
        self.assertEqual(self.generator.peak_cuda_bytes, 2048)
        self.assertEqual(self.engine.calls[-1][0], "graduate_save_feedback")
        self.assertEqual(self.engine.calls[-1][1]["record"], response_record().model_dump(mode="json"))

    def test_evaluation_has_no_save_rpc_and_allows_next_request(self):
        first, second = self.generate(capture=False), self.generate(capture=False)
        self.assertIsNone(first.capture)
        self.assertIsNone(second.capture)
        self.assertEqual(first.response_ids, second.response_ids)
        finishes = [args for name, args in self.engine.calls if name == "graduate_finish"]
        self.assertNotEqual(finishes[0]["trace_path"], finishes[1]["trace_path"])
        self.assertNotIn("graduate_save_feedback", [name for name, _ in self.engine.calls])

    def test_pending_capture_cannot_be_overwritten_or_saved_twice(self):
        result = self.generate()
        with self.assertRaisesRegex(RuntimeError, "pending feedback"):
            self.generate()
        result.capture.save(response_record(), self.root / "features.pt")
        with self.assertRaisesRegex(RuntimeError, "already been saved"):
            result.capture.save(response_record(), self.root / "other.pt")
        self.generate(capture=False)

    def test_native_failure_propagates_and_context_closes_owned_engine(self):
        self.engine.generate_error = RuntimeError("native forward failed")
        control_directory = self.generator.control_dir
        with self.assertRaisesRegex(RuntimeError, "native forward failed"):
            with self.generator:
                self.generate()
        self.assertEqual(self.engine.shutdown_count, 1)
        self.assertTrue(control_directory.exists())
        self.assertNotIn("graduate_finish", [name for name, _ in self.engine.calls])

    def test_capture_mismatch_fails_instead_of_rewriting_engine_output(self):
        self.engine.capture_mismatch = True
        with self.assertRaisesRegex(ValueError, "token IDs differ"):
            self.generate()
        self.generator.close()
        raw = json.loads((self.generator.control_dir / "000000-engine.json").read_text())
        self.assertEqual(raw, self.engine.output)
        self.assertEqual(json.loads((self.generator.control_dir / "000000.json").read_text())["response_ids"], [8])

    def test_only_native_eos_or_complete_cap_are_accepted(self):
        for ids, reason in [([], "length"), ([5], "stop"), ([5], "length"),
                            ([5, 9, 6], "stop"), ([5, -1, 9], "stop"),
                            ([5, 6, 9], "length"), ([5, 6, 7, 8], "length")]:
            with self.subTest(ids=ids, reason=reason):
                self.engine.output = {"output_ids": ids, "meta_info": {"finish_reason": {"type": reason}}}
                with self.assertRaises(ValueError):
                    self.generate(capture=False)
        self.engine.output = {"output_ids": [5, 6, 7], "meta_info": {"finish_reason": {"type": "length"}}}
        self.assertEqual(self.generate(capture=False).finish_reason, "length")

    def test_invalid_inputs_fail_before_native_work(self):
        for prompt, cap, stops, block in [([], 3, [9], 16), ([True], 3, [9], 16),
                                         ([2], 0, [9], 16), ([2], 4096, [9], 16),
                                         ([2], 3, [9], 8), ([2], 3, [9], 16.0),
                                         ([2], 3, [True], 16)]:
            with self.subTest(prompt=prompt, cap=cap, block=block):
                with self.assertRaises(ValueError):
                    self.generator.generate(prompt, cap, stops, block)
        self.assertEqual(self.engine.calls, [])

    def test_save_failure_remains_visible_and_capture_is_not_reused(self):
        result = self.generate()
        self.engine.save_error = RuntimeError("native teacher failed")
        with self.assertRaisesRegex(RuntimeError, "native teacher failed"):
            result.capture.save(response_record(), self.root / "features.pt")
        with self.assertRaisesRegex(RuntimeError, "pending feedback"):
            self.generate()

    def test_shutdown_failure_preserves_raw_control_files(self):
        self.generate(capture=False)
        control_directory = self.generator.control_dir
        self.engine.shutdown_error = RuntimeError("shutdown failed")
        with self.assertRaisesRegex(RuntimeError, "shutdown failed"):
            self.generator.close()
        self.assertTrue((control_directory / "000000.json").is_file())
        self.assertTrue((control_directory / "000000-engine.json").is_file())
        self.generator.close()
        self.assertEqual(self.engine.shutdown_count, 1)


class NativeLoaderTests(unittest.TestCase):
    def test_loader_pins_native_worker_memory_and_compiler_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = configuration(root)
            config.models.target.mkdir()
            config.models.draft.mkdir()
            (config.models.target / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
            (config.models.target / "generation_config.json").write_text(json.dumps({"eos_token_id": [9, 10]}))
            (config.models.draft / "config.json").write_text(json.dumps({
                "architectures": ["DFlashDraftModel"], "block_size": 16,
            }))
            compiler = root / "cuda"
            (compiler / "bin").mkdir(parents=True)
            (compiler / "bin" / "nvcc").touch()
            modules = {name: ModuleType(name) for name in [
                "sglang", "sglang.srt", "sglang.srt.entrypoints", "sglang.srt.entrypoints.engine",
                "graduate.sglang_capture",
            ]}
            modules["sglang.srt.entrypoints.engine"].Engine = EngineDouble
            modules["graduate.sglang_capture"].run_scheduler_process = scheduler_double
            versions = {"sglang": "0.5.18", "sglang-kernel": "0.4.6.post1",
                        "torch": "2.13.0", "transformers": "5.12.1"}
            with patch.dict(sys.modules, modules), patch("graduate.generation.version", side_effect=versions.__getitem__):
                with patch.dict(os.environ, {"CUDA_HOME": str(compiler), "CUDA_PATH": str(compiler), "PATH": "/old/bin"}):
                    with load_generator(config, root / "native") as generator:
                        engine = generator.engine
                        self.assertIs(engine.run_scheduler_process_func, scheduler_double)
                        self.assertEqual(generator.stop_token_ids, [9, 10])
                        self.assertEqual(os.environ["PATH"].split(os.pathsep)[0], str(compiler / "bin"))
                        self.assertEqual(engine.kwargs["attention_backend"], "fa3")
                        self.assertEqual(engine.kwargs["speculative_draft_attention_backend"], "fa3")
                        self.assertEqual(engine.kwargs["speculative_algorithm"], "DFLASH")
                        self.assertEqual(engine.kwargs["speculative_num_draft_tokens"], 16)
                        self.assertEqual(engine.kwargs["max_running_requests"], 1)
                        self.assertTrue(engine.kwargs["disable_overlap_schedule"])
                        self.assertTrue(engine.kwargs["disable_radix_cache"])
                        self.assertEqual(engine.kwargs["mem_fraction_static"], 0.88)
                        self.assertEqual(engine.kwargs["max_total_tokens"], 4224)
                        self.assertEqual(engine.kwargs["cuda_graph_backend_decode"], "full")
                        self.assertEqual(engine.kwargs["cuda_graph_max_bs_decode"], 8)
                        self.assertEqual(engine.kwargs["chunked_prefill_size"], 8192)
                        self.assertEqual(engine.kwargs["dtype"], "bfloat16")
                        self.assertEqual(generator.runtime["max_total_tokens"], 4224)
                    self.assertEqual(engine.shutdown_count, 1)

    def test_unqualified_runtime_version_fails_before_startup(self):
        with patch("graduate.generation.version", return_value="0.5.17"):
            with self.assertRaisesRegex(ValueError, "0.5.18"):
                load_generator(None, Path("unused"))


if __name__ == "__main__":
    unittest.main()
