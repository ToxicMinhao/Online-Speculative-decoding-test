"""CPU wrapper/reconciliation checks; these do not qualify the SGLang GPU runtime."""

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from graduate import sglang_capture as hooks
from graduate.feedback import ResponseRecord


def pair(values):
    return (torch.tensor([[value] * 4 for value in values], dtype=torch.bfloat16),
            torch.tensor([[100 + value] * 2 for value in values], dtype=torch.bfloat16))


def session(capture=True):
    result = hooks.CaptureSession([11, 12], capture)
    result.layer_ids = [1, 3]
    result.prefills = [{"start": 0, "tokens": [11], "first_token": 99,
                        "features": pair([0]) if capture else None},
                       {"start": 1, "tokens": [12], "first_token": 21,
                        "features": pair([1]) if capture else None}]
    result.blocks = [{"candidates": [21, 22, 23], "posterior": [22, 23, 2],
                      "positions": [2, 3, 4], "progress": 3,
                      "features": pair([2, 3, 4]) if capture else None}]
    return result


class SGLangCaptureTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.object(hooks.torch.cuda, "max_memory_allocated", return_value=123))
        self.enterContext(patch.object(hooks, "_ACTIVE", None))
        self.enterContext(patch.object(hooks, "_FORWARD", None))
        temporary = tempfile.TemporaryDirectory(prefix="graduate-sglang-cpu-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_chunked_prefill_and_cap_clipping_keep_only_canonical_features(self):
        observed = session()
        observed.finish([21, 22, 23], self.root / "trace.json")
        trace = json.loads((self.root / "trace.json").read_text())
        self.assertEqual(trace["peak_cuda_bytes"], 123)
        self.assertEqual(trace["trace"][0]["verify_size"], 3)
        self.assertEqual(trace["trace"][0]["committed_progress"], 2)
        self.assertEqual(trace["trace"][0]["target_argmax"], [22, 23, 2])
        observed.feedback.final_token(23, *pair([8]))
        record = ResponseRecord(record_id="cpu", session_id="cpu", request_index=0,
                                checkpoint_id="draft", purpose="serving", prompt_ids=[11, 12],
                                response_ids=[21, 22, 23], stop_token_ids=[2],
                                finish_reason="length", max_new_tokens=3)
        observed.feedback.save(record, self.root / "features.pt")
        packet = torch.load(self.root / "features.pt", weights_only=True)
        self.assertEqual(packet["hidden_states"][0, :, 0].tolist(), [0, 1, 2, 3, 8])
        self.assertEqual(packet["label_binding"]["capture_backend"], "sglang_live_capture")

    def test_eos_clipping_and_trace_only_evaluation_need_no_teacher_rows(self):
        observed = session(False)
        observed.blocks[0].update(candidates=[21, 2, 23], posterior=[2, 23, 24])
        observed.finish([21, 2], self.root / "evaluation.json")
        document = json.loads((self.root / "evaluation.json").read_text())
        self.assertEqual(document["trace"][0]["committed_progress"], 1)
        self.assertIsNone(observed.feedback)
        self.assertEqual({item.name for item in self.root.iterdir()}, {"evaluation.json"})

    def test_rejected_history_preserves_raw_observations_before_failure(self):
        observed = session()
        observed.blocks[0]["candidates"][1] = 77
        with self.assertRaisesRegex(ValueError, "noncanonical"):
            observed.finish([21, 22, 23, 2], self.root / "failure.json")
        raw = json.loads((self.root / "failure.json").read_text())
        self.assertEqual(raw["raw_blocks"][0]["candidates"], [21, 77, 23])
        self.assertEqual(raw["response_ids"], [21, 22, 23, 2])
        self.assertFalse(observed.finished)

    def test_first_prefill_eos_has_no_verification_blocks(self):
        observed = session()
        observed.prefills[-1]["first_token"] = 2
        observed.blocks = []
        observed.finish([2], self.root / "short.json")
        self.assertEqual(observed.feedback.response_ids, [2])
        self.assertEqual(json.loads((self.root / "short.json").read_text())["trace"], [])

    def test_graph_bridge_survives_trim_without_mutating_graph_dictionary(self):
        auxiliary, teacher = pair([0, 1, 2])
        metadata = SimpleNamespace(capture_hidden_mode=SimpleNamespace(is_full=lambda: True))
        original_output = SimpleNamespace(hidden_states=auxiliary, customized_info={"user": "kept"})
        with patch.object(hooks, "_original_logits", return_value=original_output):
            graph_output = hooks._logits_forward(None, None, teacher, None, metadata, auxiliary)
        graph_dictionary = graph_output.customized_info
        trimmed_output = SimpleNamespace(hidden_states=auxiliary[:2], customized_info=None)
        with patch.object(hooks, "_original_trim", return_value=trimmed_output):
            result = hooks._trim_output(None, graph_output)
        aux_rows, teacher_rows = hooks._take_pair(result)
        self.assertEqual(teacher_rows.shape, (2, 2))
        self.assertEqual(aux_rows.shape, (2, 4))
        self.assertEqual(result.customized_info, {"user": "kept"})
        self.assertIn(hooks._TEACHER, graph_dictionary)
        self.assertIs(graph_dictionary[hooks._TEACHER], graph_output.customized_info[hooks._TEACHER])

    def test_target_wrapper_owns_features_before_native_buffer_reuse(self):
        auxiliary, teacher = pair([0, 1])
        target = object()
        hooks._ACTIVE = SimpleNamespace(capture=True)
        hooks._FORWARD = {"worker": SimpleNamespace(target_worker=target)}
        output = SimpleNamespace(hidden_states=auxiliary,
                                 customized_info={hooks._TEACHER: teacher})
        forward = SimpleNamespace(input_ids=torch.tensor([21, 22]), positions=torch.tensor([2, 3]))
        with patch.object(hooks, "_original_target", return_value=SimpleNamespace(
                logits_output=output, can_run_cuda_graph=True)):
            returned = hooks._target_forward(target, forward_batch=forward)
        auxiliary.fill_(90)
        teacher.fill_(90)
        self.assertEqual(hooks._FORWARD["features"][0][:, 0].tolist(), [0, 1])
        self.assertEqual(hooks._FORWARD["features"][1][:, 0].tolist(), [100, 101])
        self.assertEqual(hooks._FORWARD["tokens"], [21, 22])
        self.assertEqual(hooks._FORWARD["positions"], [2, 3])
        self.assertIs(hooks._FORWARD["target_cuda_graph"], True)
        self.assertIsNot(returned.logits_output, output)
        self.assertIn(hooks._TEACHER, output.customized_info)
        self.assertIsNone(returned.logits_output.customized_info)

    def test_outer_worker_discards_rejected_rows_but_keeps_native_full_trace(self):
        observed = hooks.CaptureSession([11, 12], True)
        hooks._ACTIVE = observed
        mode = SimpleNamespace(is_idle=lambda: False, is_extend=lambda: False)
        batch = SimpleNamespace(forward_mode=mode, is_extend_in_batch=False,
                                batch_size=lambda: 1, reqs=[SimpleNamespace(origin_input_ids=[11, 12])],
                                sampling_info=SimpleNamespace(is_all_greedy=True))
        config = SimpleNamespace(dflash_config={"target_layer_ids": [1, 3]})
        worker = SimpleNamespace(use_compact_draft_cache=False,
                                 draft_model_runner=SimpleNamespace(model_config=SimpleNamespace(hf_config=config)))
        runner = worker.draft_model_runner
        runner.prefill_attention_backend_str = runner.decode_attention_backend_str = "fa3"
        runner.attn_backend = object()
        worker.target_worker = SimpleNamespace(model_runner=runner)
        logits = torch.zeros((3, 128))
        logits[range(3), [22, 31, 32]] = 1

        def original(instance, actual_batch):
            hooks._FORWARD.update(tokens=[21, 77, 78], positions=[2, 3, 4], features=pair([2, 98, 99]),
                                  target_cuda_graph=True)
            return SimpleNamespace(logits_output=SimpleNamespace(next_token_logits=logits),
                                   accept_lens=torch.tensor([1]))

        with patch.object(hooks, "_original_worker", side_effect=original):
            hooks._worker_forward(worker, batch)
        self.assertEqual(observed.blocks[0]["features"][0][:, 0].tolist(), [2])
        self.assertEqual(observed.blocks[0]["candidates"], [21, 77, 78])
        self.assertEqual(observed.blocks[0]["posterior"], [22, 31, 32])
        self.assertIsNone(hooks._FORWARD)

    def test_rpc_rejects_overlap_and_releases_trace_only_session(self):
        scheduler = SimpleNamespace(enable_overlap=True, is_fully_idle=lambda: True,
                                    flush_cache=Mock(return_value=True))
        with self.assertRaisesRegex(ValueError, "overlap disabled"):
            hooks.graduate_begin(scheduler, [11, 12], False)
        scheduler.flush_cache.assert_not_called()
        scheduler.enable_overlap = False
        hooks.graduate_begin(scheduler, [11, 12], False)
        hooks._ACTIVE = session(False)
        hooks.graduate_finish(scheduler, [21, 22, 23, 2], str(self.root / "trace.json"))
        self.assertIsNone(hooks._ACTIVE)

    def test_terminal_rpc_uses_full_context_without_sampling_and_flushes_failures(self):
        class Request:
            def __init__(self, **values):
                self.__dict__.update(values)

            def init_next_round_input(self):
                self.initialized = True

            def set_extend_range(self, start, end):
                self.extend_range = (start, end)

        def make_batch(**values):
            return SimpleNamespace(**values, prepare_for_extend=Mock())

        def resolve(batch, future_map):
            self.assertIsNone(future_map)
            request = batch.reqs[0]
            self.assertTrue(request.initialized)
            self.assertEqual(request.extend_range, (0, 6))
            batch.input_ids = torch.tensor(list(request.origin_input_ids))

        forward_inputs = []

        def make_forward(batch, runner, **options):
            self.assertFalse(options["return_hidden_states_before_norm"])
            forward_inputs.append(batch.input_ids.tolist())
            return SimpleNamespace(input_ids=batch.input_ids)

        modules = {
            "sglang.srt.managers.overlap_utils": SimpleNamespace(resolve_forward_inputs=resolve),
            "sglang.srt.managers.schedule_batch": SimpleNamespace(
                Req=Request, ScheduleBatch=SimpleNamespace(init_new=make_batch)),
            "sglang.srt.model_executor.forward_batch_info": SimpleNamespace(
                CaptureHiddenMode=SimpleNamespace(FULL="full"),
                ForwardBatch=SimpleNamespace(init_new=make_forward)),
            "sglang.srt.sampling.sampling_params": SimpleNamespace(SamplingParams=lambda **values: values),
            "sglang.srt.speculative.spec_info": SimpleNamespace(SpeculativeAlgorithm=SimpleNamespace(NONE="none")),
        }
        response = ResponseRecord(record_id="cpu", session_id="cpu", request_index=0,
                                  checkpoint_id="draft", purpose="serving", prompt_ids=[11, 12],
                                  response_ids=[21, 22, 23, 2], stop_token_ids=[2],
                                  finish_reason="eos", max_new_tokens=8)
        for failure in (False, True):
            with self.subTest(failure=failure), patch.dict(sys.modules, modules):
                observed = session()
                observed.finish(list(response.response_ids), self.root / "trace.json")
                hooks._ACTIVE = observed
                auxiliary, teacher = pair([0, 1, 2, 3, 4, 9])
                runner = SimpleNamespace(
                    req_to_token_pool=object(), token_to_kv_pool_allocator=object(), model_config=object(),
                    forward=Mock(return_value=SimpleNamespace(logits_output=SimpleNamespace(
                        hidden_states=auxiliary, customized_info={hooks._TEACHER: teacher}))))
                scheduler = SimpleNamespace(
                    enable_overlap=False, is_fully_idle=lambda: True, flush_cache=Mock(return_value=True),
                    server_args=SimpleNamespace(tp_size=1, pp_size=1), tree_cache=object(),
                    tp_worker=SimpleNamespace(model_runner=runner))
                path = self.root / ("failure.pt" if failure else "terminal.pt")
                if failure:
                    runner.forward.side_effect = RuntimeError("native target failed")
                    with self.assertRaisesRegex(RuntimeError, "native target failed"):
                        hooks.graduate_save_feedback(scheduler, response.model_dump(mode="json"), str(path))
                    self.assertFalse(path.exists())
                    self.assertIs(hooks._ACTIVE, observed)
                else:
                    hooks.graduate_save_feedback(scheduler, response.model_dump(mode="json"), str(path))
                    saved = torch.load(path, weights_only=True)
                    self.assertEqual(saved["hidden_states"][0, :, 0].tolist(), [0, 1, 2, 3, 4, 9])
                    receipt = json.loads(path.with_suffix(".json").read_text())
                    self.assertEqual(receipt["terminal_input_rows"], 6)
                    self.assertEqual(receipt["terminal_forward_count"], 1)
                    self.assertEqual(receipt["retained_terminal_rows"], 1)
                    self.assertFalse(receipt["final_feature_logits_used_for_labels"])
                    self.assertIsNone(hooks._ACTIVE)
                self.assertEqual(runner.forward.call_count, 1)
                self.assertEqual(scheduler.flush_cache.call_count, 2)
        self.assertEqual(forward_inputs, [[11, 12, 21, 22, 23, 2]] * 2)


if __name__ == "__main__":
    unittest.main()
