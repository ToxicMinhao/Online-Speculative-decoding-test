"""CPU boundaries for the native SGLang adapter; no server or GPU is launched."""
from contextlib import nullcontext
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from online_feedback import sglang_backend as B
from online_feedback import sglang_worker as W


def backend(root):
    result = object.__new__(B.SGLangDFlashBackend)
    result.root = Path(root)
    result.context_limit = 16
    result.block_size = 16
    result.checkpoint_id = "/released"
    result.stop_tokens = [9]
    result.target_layer_ids = [0, 2]
    result.serial = 0
    result.connection = None
    result.worker = None
    result._call = Mock()
    return result


def generated():
    return {
        "output_ids": [4, 5, 9],
        "native_generate_s": 0.5,
        "trace": {
            "blocks": [{
                "output_start": 0,
                "anchor_token": 4,
                "committed_progress": 2,
                "committed_tokens": [5, 9],
                "accepted_draft_tokens": 1,
            }],
        },
    }


class GenerationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.backend = backend(temporary.name)
        self.backend._call.return_value = generated()
        rng = patch.object(B, "preserve_rng", side_effect=nullcontext)
        rng.start()
        self.addCleanup(rng.stop)

    def generate(self, **changes):
        arguments = {
            "max_new_tokens": 8, "request_id": "r1", "session_id": "s1",
            "request_index": 1, "purpose": "serving",
        }
        prompt = changes.pop("prompt_ids", [1, 2])
        arguments.update(changes)
        return self.backend.generate(prompt, **arguments)

    def test_invalid_requests_never_reach_worker(self):
        for changes in ({"prompt_ids": []}, {"max_new_tokens": 0},
                        {"max_new_tokens": 15}, {"purpose": "training"}):
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    self.generate(**changes)
        self.backend._call.assert_not_called()

    def test_evaluation_disables_feature_capture_and_preserves_exact_ids(self):
        with patch.object(B.time, "perf_counter", side_effect=[10.0, 12.0]):
            record = self.generate(purpose="evaluation")
        kwargs = self.backend._call.call_args.kwargs
        self.assertFalse(kwargs["capture"])
        self.assertEqual(kwargs["prompt_ids"], [1, 2])
        self.assertEqual(record["response_ids"], [4, 5, 9])
        self.assertEqual(record["finish_reason"], "eos")
        self.assertFalse(record["metadata"]["feedback_capture_in_timed_request"])
        self.assertEqual(record["metadata"]["request_latency_s"], 2.0)
        self.assertEqual(record["metadata"]["native_generate_s"], 0.5)
        self.assertEqual(record["metadata"]["verification_calls"], 1)

    def test_serving_marks_native_capture_cost_inside_request(self):
        record = self.generate()
        self.assertTrue(self.backend._call.call_args.kwargs["capture"])
        self.assertTrue(record["metadata"]["feedback_capture_in_timed_request"])

    def test_malformed_committed_response_is_rejected(self):
        for response in ([4, 9, 5], [4, 5], []):
            with self.subTest(response=response):
                result = generated()
                result["output_ids"] = response
                self.backend._call.return_value = result
                with self.assertRaises(RuntimeError):
                    self.generate()

    def test_publication_failure_preserves_checkpoint_identity(self):
        self.backend._call.side_effect = RuntimeError("failed native validation")
        with self.assertRaises(RuntimeError):
            self.backend.publish("/new")
        self.assertEqual(self.backend.checkpoint_id, "/released")

    def test_activation_time_includes_whole_validated_worker_call(self):
        self.backend._call.return_value = {"native_activation_s": 0.25}
        with patch.object(B.time, "perf_counter", side_effect=[20.0, 23.0]):
            receipt = self.backend.publish("/new")
        self.assertEqual(receipt["activation_s"], 3.0)
        self.assertEqual(receipt["native_activation_s"], 0.25)
        self.assertEqual(self.backend.checkpoint_id, "/new")


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.worker = object.__new__(W.NativeWorker)
        self.worker.layout = {"fc.weight": ([2, 6], "BF16")}
        self.worker.config = {"hidden_size": 2}
        self.worker.blocked = False
        self.worker.engine = Mock()
        self.worker.engine.update_weights_from_disk.return_value = (True, "updated")
        self.worker.status = Mock(return_value={
            "status": "passed", "publication": {"status": "passed"},
        })
        self.worker.check_weights = Mock(return_value={"all_equal": True})
        patches = (
            patch.object(W, "checkpoint_layout", return_value=self.worker.layout),
            patch.object(W, "semantic_config", return_value=self.worker.config),
        )
        self.layout = patches[0].start()
        self.semantic = patches[1].start()
        for item in patches:
            self.addCleanup(item.stop)

    def test_structure_or_config_mismatch_rejected_before_weight_mutation(self):
        for field in ("layout", "config"):
            with self.subTest(field=field):
                self.layout.return_value = self.worker.layout if field != "layout" else {}
                self.semantic.return_value = self.worker.config if field != "config" else {}
                with self.assertRaises(ValueError):
                    self.worker.publish({"checkpoint": "/new"})
                self.assertFalse(self.worker.blocked)
        self.worker.engine.update_weights_from_disk.assert_not_called()

    def test_failed_native_update_poisons_worker(self):
        self.worker.engine.update_weights_from_disk.return_value = (False, "partial load")
        with self.assertRaises(RuntimeError):
            self.worker.publish({"checkpoint": "/new"})
        self.assertTrue(self.worker.blocked)

    def test_failed_helper_validation_poisons_worker(self):
        self.worker.status.return_value = {
            "status": "failed", "publication": {"status": "failed"},
        }
        with self.assertRaises(RuntimeError):
            self.worker.publish({"checkpoint": "/new"})
        self.assertTrue(self.worker.blocked)

    def test_successful_publication_validates_helper_before_unblocking(self):
        def checked():
            self.assertTrue(self.worker.blocked)
            return {"status": "passed", "publication": {"status": "passed"}}

        self.worker.status.side_effect = checked
        self.worker.publish({"checkpoint": "/new"})
        self.worker.status.assert_called_once()
        self.worker.check_weights.assert_not_called()
        self.worker.engine.update_weights_from_disk.assert_called_once_with(
            model_path="/new", update_speculative_draft=True)
        self.assertFalse(self.worker.blocked)

    def test_weight_equality_remains_explicit_qualification_operation(self):
        with tempfile.TemporaryDirectory() as directory:
            self.worker.root = Path(directory)

            def write_check(operation, **kwargs):
                self.assertEqual(operation, "online_check_weights")
                self.assertEqual(kwargs["checkpoint"], "/new")
                Path(kwargs["outputfile"]).write_text(json.dumps({"all_equal": False}))

            self.worker.engine.collective_rpc.side_effect = write_check
            result = W.NativeWorker.check_weights(self.worker, {"checkpoint": "/new"})
            self.assertIs(result["all_equal"], False)


class PublicationConfigTests(unittest.TestCase):
    """Reproduce the pinned loader's observed old/new config dictionaries."""

    def setUp(self):
        import torch
        from safetensors.torch import save_file

        self.torch, self.save_file = torch, save_file
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.released, self.export = self.root / "released", self.root / "export"
        self.released.mkdir()
        self.export.mkdir()
        self.base = {
            "model_type": "qwen3", "vocab_size": 32,
            "rope_theta": 1000000, "rope_scaling": None,
        }
        self.export_config = {
            **self.base, "rope_parameters": {"rope_theta": 1000000, "rope_type": "default"},
        }
        self.weights = {"fc.weight": torch.ones(2, 6, dtype=torch.bfloat16)}
        for path, config in ((self.released, self.base), (self.export, self.export_config)):
            (path / "config.json").write_text(json.dumps(config))
            save_file(self.weights, path / "model.safetensors")

        def load_config(path):
            config = json.loads((Path(path) / "config.json").read_text())
            return Mock(to_dict=Mock(return_value=config))

        loader = patch("transformers.AutoConfig.from_pretrained", side_effect=load_config)
        loader.start()
        self.addCleanup(loader.stop)
        self.worker = object.__new__(W.NativeWorker)
        self.worker.layout = W.checkpoint_layout(self.released)
        self.worker.config = W.semantic_config(self.released)
        self.worker.blocked = False
        self.worker.engine = Mock()
        self.worker.engine.update_weights_from_disk.return_value = (True, "updated")
        self.worker.status = Mock(return_value={
            "status": "passed", "publication": {"status": "passed"},
        })

    def test_exact_redundant_hf5_default_rope_alias_can_be_published(self):
        self.assertEqual(W.semantic_config(self.export), self.worker.config)
        self.worker.publish({"checkpoint": str(self.export)})
        self.worker.engine.update_weights_from_disk.assert_called_once_with(
            model_path=str(self.export), update_speculative_draft=True)
        self.assertFalse(self.worker.blocked)

    def test_changed_rope_parameters_are_not_ignored(self):
        changes = (
            {"rope_parameters": {"rope_theta": 2000000, "rope_type": "default"}},
            {"rope_theta": 2000000,
             "rope_parameters": {"rope_theta": 2000000, "rope_type": "default"}},
            {"rope_parameters": {"rope_theta": 1000000, "rope_type": "linear"}},
            {"rope_scaling": {"rope_type": "linear", "factor": 2.0}},
            {"rope_parameters": {"rope_theta": 1000000, "rope_type": "default",
                                 "partial_rotary_factor": 0.5}},
            {"rope_parameters": {"rope_type": "default"}},
        )
        for changed in changes:
            with self.subTest(changed=changed):
                (self.export / "config.json").write_text(json.dumps({**self.export_config, **changed}))
                with self.assertRaisesRegex(ValueError, "semantic draft configuration"):
                    self.worker.publish({"checkpoint": str(self.export)})
                self.assertFalse(self.worker.blocked)
        self.worker.engine.update_weights_from_disk.assert_not_called()

    def test_tensor_key_shape_and_dtype_checks_remain_strict(self):
        changes = (
            {"other.weight": self.weights["fc.weight"]},
            {"fc.weight": self.torch.ones(2, 5, dtype=self.torch.bfloat16)},
            {"fc.weight": self.torch.ones(2, 6, dtype=self.torch.float32)},
        )
        for weights in changes:
            with self.subTest(keys=list(weights), tensor=next(iter(weights.values())).shape):
                self.save_file(weights, self.export / "model.safetensors")
                with self.assertRaisesRegex(ValueError, "tensor keys, shapes or dtypes"):
                    self.worker.publish({"checkpoint": str(self.export)})
                self.assertFalse(self.worker.blocked)
        self.worker.engine.update_weights_from_disk.assert_not_called()

    def test_nonfinite_weights_are_rejected_before_native_loading(self):
        for value in (float("nan"), float("inf")):
            with self.subTest(value=value):
                weights = {"fc.weight": self.torch.full((2, 6), value, dtype=self.torch.bfloat16)}
                self.save_file(weights, self.export / "model.safetensors")
                with self.assertRaisesRegex(ValueError, "Nonfinite checkpoint tensor"):
                    self.worker.publish({"checkpoint": str(self.export)})
                self.assertFalse(self.worker.blocked)
        self.worker.engine.update_weights_from_disk.assert_not_called()


class FeedbackTests(unittest.TestCase):
    def setUp(self):
        import torch

        self.torch = torch
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.backend = backend(self.root)
        self.path = self.root / "feedback.pt"
        self.capture = {
            "status": "passed", "prompt_ids": [1, 2], "response_ids": [4, 5, 9],
            "target_argmax_ids": [4, 5, 9], "covered_token_ids": [1, 2, 4, 5],
            "covered_rows": 4, "block_size": 16,
            "native_capture_validated": True,
            "label_binding_scope": "native_prefill_and_committed_verification",
            "final_feature_logits_used_for_labels": False,
            "final_emitted_token_feature_present": False, "collect_features": True,
            "feature_path": str(self.root / "capture.pt"),
        }
        self.packet = {
            **self.capture, "input_ids": [1, 2, 4, 5],
            "hidden_states": torch.arange(24, dtype=torch.bfloat16).reshape(4, 6),
            "target_last_hidden_states": torch.arange(12, dtype=torch.bfloat16).reshape(4, 3),
        }
        self.terminal = {
            "status": "passed", "input_ids": [1, 2, 4, 5, 9], "positions": [4],
            "retained_feature_rows": 1, "forward_input_rows": 5,
            "target_forward_count": 1, "final_feature_logits_used_for_labels": False,
            "capture_backend": "same_frozen_SGLang_full_context_final_row",
            "tensor_path": str(self.root / "terminal.pt"),
        }
        self.last = {
            **self.terminal,
            "hidden_states": torch.full((1, 6), 42, dtype=torch.bfloat16),
            "target_last_hidden_states": torch.full((1, 3), 43, dtype=torch.bfloat16),
        }
        self.record = {
            "record_id": "r1", "purpose": "serving", "prompt_ids": [1, 2],
            "response_ids": [4, 5, 9], "metadata": {"capture": self.capture},
        }
        self.backend._call.return_value = self.terminal

    def write_inputs(self):
        self.torch.save(self.packet, self.capture["feature_path"])
        self.torch.save(self.last, self.terminal["tensor_path"])

    def test_payload_preserves_native_rows_and_appends_only_terminal_row(self):
        from online_feedback.training import validate_feature, validate_target_binding

        self.write_inputs()
        receipt = self.backend.prepare_feedback(self.record, self.path)
        payload = self.torch.load(self.path, map_location="cpu", weights_only=True)
        self.assertEqual(payload["input_ids"].tolist(), [[1, 2, 4, 5, 9]])
        self.assertEqual(payload["loss_mask"].tolist(), [[0, 0, 1, 1, 1]])
        for key in ("hidden_states", "target_last_hidden_states"):
            self.assertTrue(self.torch.equal(payload[key][0, :-1], self.packet[key]))
            self.assertTrue(self.torch.equal(payload[key][0, -1:], self.last[key]))
        validate_feature(payload, max_length=16, hidden_width=6, vocab_size=32)
        validate_target_binding(payload)
        self.assertEqual(receipt["target_argmax_ids"], [4, 5, 9])
        self.assertFalse(receipt["final_feature_logits_used_for_labels"])
        self.backend._call.assert_called_once_with(
            "teacher", input_ids=[1, 2, 4, 5, 9],
            outputfile=str(self.path.with_suffix(".terminal.json")))

    def test_evaluation_short_response_and_existing_features_do_not_call_teacher(self):
        with self.assertRaises(ValueError):
            self.backend.prepare_feedback({**self.record, "purpose": "evaluation"}, self.path)
        short = self.backend.prepare_feedback({**self.record, "response_ids": [9]}, self.path)
        self.assertEqual(short["status"], "ineligible_short_response")
        self.path.write_bytes(b"immutable existing evidence")
        with self.assertRaises(FileExistsError):
            self.backend.prepare_feedback(self.record, self.path)
        self.backend._call.assert_not_called()

    def test_unverified_or_wrong_capture_cannot_be_relabelled_valid(self):
        changes = (
            ("native_capture_validated", False),
            ("label_binding_scope", "other_prefix"),
            ("target_argmax_ids", [4, 7, 9]),
            ("covered_token_ids", [1, 3, 4, 5]),
            ("collect_features", False),
        )
        for field, value in changes:
            with self.subTest(field=field):
                capture = copy.deepcopy(self.capture)
                capture[field] = value
                record = {**self.record, "metadata": {"capture": capture}}
                self.write_inputs()
                with self.assertRaises((ValueError, RuntimeError)):
                    self.backend.prepare_feedback(record, self.root / f"bad_{field}.pt")
                self.assertFalse((self.root / f"bad_{field}.pt").exists())

    def test_terminal_receipt_cannot_supply_another_prefix_or_position(self):
        changes = (
            ("input_ids", [1, 3, 4, 5, 9]),
            ("positions", [3]),
            ("final_feature_logits_used_for_labels", True),
        )
        for field, value in changes:
            with self.subTest(field=field):
                self.write_inputs()
                terminal = {**self.terminal, field: value}
                self.backend._call.return_value = terminal
                destination = self.root / f"bad_terminal_{field}.pt"
                with self.assertRaises((ValueError, RuntimeError)):
                    self.backend.prepare_feedback(self.record, destination)
                self.assertFalse(destination.exists())


class WorkerLifetimeTests(unittest.TestCase):
    def test_failed_startup_connection_cleans_isolated_worker_group(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target, draft, output = root / "target", root / "draft", root / "worker"
            target.mkdir()
            draft.mkdir()
            (target / "config.json").write_text(json.dumps({"eos_token_id": 9}))
            (draft / "config.json").write_text(json.dumps({
                "block_size": 16, "dflash_config": {"target_layer_ids": [0, 2]},
            }))
            process = Mock(pid=12345)
            process.wait.return_value = 0

            def started(*args, **kwargs):
                self.assertTrue(kwargs["start_new_session"])
                (output / "ready.json").write_text(json.dumps({"address": ["127.0.0.1", 1234]}))
                return process

            instance = object.__new__(B.SGLangDFlashBackend)
            with patch("transformers.AutoTokenizer.from_pretrained", return_value=Mock(eos_token_id=9)), \
                    patch.object(B.subprocess, "Popen", side_effect=started), \
                    patch.object(B, "Client", side_effect=ConnectionRefusedError("worker died")), \
                    patch.object(B.os, "killpg") as kill:
                with self.assertRaises(ConnectionRefusedError):
                    instance.__init__(target, draft, worker_python="/unused/python", output_root=output)
            kill.assert_called_once_with(12345, B.signal.SIGTERM)
            process.wait.assert_called_once()
            self.assertTrue(instance.log.closed)

    def test_rpc_timeout_does_not_block_in_recv(self):
        instance = backend("/unused")
        instance.connection = Mock()
        instance.connection.poll.return_value = False
        with self.assertRaises(TimeoutError):
            B.SGLangDFlashBackend._call(instance, "generate")
        instance.connection.recv.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, "blocked"):
            B.SGLangDFlashBackend._call(instance, "generate")
        instance.connection.send.assert_called_once()

    def test_close_still_cleans_owned_process_when_connection_fails(self):
        instance = backend("/unused")
        connection = Mock()
        instance.connection = connection
        instance.worker = Mock()
        instance.worker.poll.return_value = None
        connection.send.side_effect = BrokenPipeError("worker disconnected")
        instance._stop_process = Mock()
        try:
            instance.close()
        except BrokenPipeError:
            pass
        connection.close.assert_called_once()
        self.assertIsNone(instance.connection)
        instance._stop_process.assert_called_once()

    def test_owned_process_group_is_escalated_if_shutdown_times_out(self):
        instance = backend("/unused")
        instance.worker = Mock(pid=12345)
        instance.worker.wait.side_effect = [B.subprocess.TimeoutExpired("worker", 10), 0]
        instance.log = Mock()
        with patch.object(B.os, "killpg") as kill:
            instance._stop_process()
        self.assertEqual(kill.call_args_list[0].args, (12345, B.signal.SIGTERM))
        self.assertEqual(kill.call_args_list[1].args, (12345, B.signal.SIGKILL))
        instance.log.close.assert_called_once()

    def test_client_disconnect_closes_engine(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = root / "settings.json"
            settings.write_text("{}")
            worker = Mock()
            worker.status.return_value = {"backend": "fa3"}
            connection = Mock()
            connection.recv.side_effect = EOFError("parent disconnected")
            listener = Mock()
            listener.address = ("127.0.0.1", 1234)
            accepted = Mock()
            accepted.__enter__ = Mock(return_value=connection)
            accepted.__exit__ = Mock(return_value=False)
            listener.accept.return_value = accepted
            context = Mock()
            context.__enter__ = Mock(return_value=listener)
            context.__exit__ = Mock(return_value=False)
            with patch.object(W, "NativeWorker", return_value=worker), \
                    patch.object(W, "Listener", return_value=context), \
                    patch("sys.argv", ["worker", "--settings", str(settings), "--output", str(root)]):
                try:
                    W.main()
                except EOFError:
                    pass
            worker.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
