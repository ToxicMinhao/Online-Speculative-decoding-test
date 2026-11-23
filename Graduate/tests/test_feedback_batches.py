"""CPU structural checks with tiny invented features, not native model qualification."""

from contextlib import chdir, redirect_stderr, redirect_stdout
from copy import deepcopy
import io
from dataclasses import replace
import json
from pathlib import Path
import random
import tempfile
import tomllib
import unittest
from unittest.mock import Mock, patch, sentinel

import torch

from graduate.__main__ import main
from graduate.batches import FeatureLayout, load_feedback
from graduate.feedback import ResponseRecord, read_response


ROOT = Path(__file__).resolve().parents[1]


def response_values():
    return {
        "record_id": "response-7",
        "session_id": "session-3",
        "request_index": 7,
        "checkpoint_id": "draft-before-update",
        "purpose": "serving",
        "prompt_ids": [11, 12, 13],
        "response_ids": [21, 22, 2],
        "finish_reason": "eos",
        "stop_token_ids": [2],
        "max_new_tokens": 8,
        "metadata": {"fixture": "cpu-structural"},
    }


def feature_fixture():
    """Every feature row/column is distinct, making shifts and reordering visible."""
    layout = FeatureLayout(
        context_limit=6, vocab_size=64, target_hidden_size=4, target_layer_ids=(1, 3)
    )
    record = ResponseRecord.model_validate(response_values())
    auxiliary = torch.tensor(
        [[row * 8 + column for column in range(8)] for row in range(6)],
        dtype=torch.bfloat16,
    )
    teacher = torch.tensor(
        [[100 + row * 4 + column for column in range(4)] for row in range(6)],
        dtype=torch.bfloat16,
    )
    # Assemble the saved packet independently of the production reader.
    prepared = {
        "record_id": "response-7",
        "prompt_length": 3,
        "input_ids": torch.tensor([[11, 12, 13, 21, 22, 2]], dtype=torch.int64),
        "loss_mask": torch.tensor([[0, 0, 0, 1, 1, 1]], dtype=torch.float32),
        "hidden_states": auxiliary.unsqueeze(0),
        "target_last_hidden_states": teacher.unsqueeze(0),
        "label_binding": {
            "record_id": "response-7",
            "prompt_length": 3,
            "input_ids": [11, 12, 13, 21, 22, 2],
            "target_argmax_ids": [21, 22, 2],
            "target_layer_ids": [1, 3],
            "label_mismatches": [],
            "capture_backend": "same_frozen_SGLang_native_capture",
            "native_capture_validated": True,
            "label_binding_scope": "native_prefill_and_committed_verification",
            "final_feature_logits_used_for_labels": False,
            "response_only_loss": True,
        },
    }
    return record, layout, prepared


def load_saved_fixture(record, prepared, layout):
    """Exercise the public loader with an independently assembled saved packet."""
    with tempfile.TemporaryDirectory(prefix="graduate-saved-feedback-") as directory:
        path = Path(directory) / "prepared.pt"
        torch.save(prepared, path)
        return load_feedback(record, path, layout)


class ResponseRecordTests(unittest.TestCase):
    def test_saved_response_keeps_exact_ids_and_rejects_evaluation(self):
        with tempfile.TemporaryDirectory(prefix="graduate-response-test-") as directory:
            path = Path(directory) / "response.json"
            values = response_values()
            path.write_text(json.dumps(values))
            record = read_response(path)
            self.assertEqual(record.input_ids, (11, 12, 13, 21, 22, 2))
            self.assertEqual(record.eligible_anchor_count, 2)
            values["purpose"] = "evaluation"
            path.write_text(json.dumps(values))
            with self.assertRaises(ValueError):
                read_response(path)

    def test_length_termination_requires_the_original_cap(self):
        values = response_values()
        values.update(response_ids=[21, 22, 23], finish_reason="length", max_new_tokens=3)
        self.assertEqual(ResponseRecord.model_validate(values).response_ids, (21, 22, 23))
        values["max_new_tokens"] = 4
        with self.assertRaises(ValueError):
            ResponseRecord.model_validate(values)

    def test_invalid_termination_is_not_repaired(self):
        changes = (
            {"response_ids": [21, 2, 22]},
            {"response_ids": [21, 22, 23]},
            {"finish_reason": "length"},
            {"max_new_tokens": 2},
        )
        for change in changes:
            with self.subTest(change=change):
                values = response_values()
                values.update(change)
                with self.assertRaises(ValueError):
                    ResponseRecord.model_validate(values)

    def test_token_ids_require_nonnegative_integers(self):
        with tempfile.TemporaryDirectory(prefix="graduate-invalid-response-") as directory:
            path = Path(directory) / "response.json"
            for token in (-1, True, 21.0, "21"):
                with self.subTest(token=token):
                    values = response_values()
                    values["response_ids"] = [token, 22, 2]
                    with self.assertRaises(ValueError):
                        ResponseRecord.model_validate(values)
                    path.write_text(json.dumps(values))
                    with self.assertRaises(ValueError):
                        read_response(path)

    def test_one_token_response_is_evidence_but_not_a_training_sample(self):
        _, layout, prepared = feature_fixture()
        for token, reason in ((2, "eos"), (21, "length")):
            with self.subTest(reason=reason):
                values = response_values()
                values.update(response_ids=[token], finish_reason=reason, max_new_tokens=1)
                record = ResponseRecord.model_validate(values)
                original = record.model_dump()
                self.assertEqual(record.eligible_anchor_count, 0)
                with self.assertRaisesRegex(ValueError, "no eligible adjacent tokens"):
                    load_saved_fixture(record, prepared, layout)
                self.assertEqual(record.model_dump(), original)


class FeedbackBatchTests(unittest.TestCase):
    def setUp(self):
        self.record, self.layout, self.prepared = feature_fixture()

    def test_batch_keeps_full_context_without_padding_or_premature_shift(self):
        batch = load_saved_fixture(self.record, self.prepared, self.layout)
        self.assertEqual(batch.record_id, "response-7")
        self.assertEqual(batch.input_ids.tolist(), [[11, 12, 13, 21, 22, 2]])
        self.assertEqual(batch.loss_mask.tolist(), [[0, 0, 0, 1, 1, 1]])
        self.assertEqual(batch.hidden_states.shape, (1, 6, 8))
        self.assertEqual(batch.target_last_hidden_states.shape, (1, 6, 4))
        self.assertEqual(batch.hidden_states[0, 0].tolist(), list(range(8)))
        self.assertEqual(batch.hidden_states[0, 5].tolist(), list(range(40, 48)))

    def test_unshifted_teacher_rows_support_the_model_predecessor_alignment(self):
        batch = load_saved_fixture(self.record, self.prepared, self.layout)
        # The later model uses j-1 for token j. Batch preparation must keep this
        # mapping possible; it must not apply the shift a second time.
        self.assertEqual(batch.input_ids[0, 4:6].tolist(), [22, 2])
        self.assertEqual(batch.target_last_hidden_states[0, 3:5].tolist(), [
            [112, 113, 114, 115],
            [116, 117, 118, 119],
        ])
        self.assertEqual(batch.target_last_hidden_states[0, 5].tolist(), [120, 121, 122, 123])

    def test_context_cap_rejects_the_entire_sample_without_truncation(self):
        too_short = replace(self.layout, context_limit=5)
        with self.assertRaisesRegex(ValueError, "truncation is forbidden"):
            load_saved_fixture(self.record, self.prepared, too_short)
        self.assertEqual(self.prepared["input_ids"].shape, (1, 6))

    def test_tokens_outside_vocabulary_are_rejected(self):
        layout = replace(self.layout, vocab_size=22)
        with self.assertRaisesRegex(ValueError, "outside the model vocabulary"):
            load_saved_fixture(self.record, self.prepared, layout)

    def test_native_binding_ids_do_not_accept_numerically_equal_float_or_bool(self):
        for name, value in (("input_ids", [11.0, 12, 13, 21, 22, 2]),
                            ("target_layer_ids", [True, 3])):
            with self.subTest(name=name, value=value):
                prepared = deepcopy(self.prepared)
                prepared["label_binding"][name] = value
                with self.assertRaises(ValueError):
                    load_saved_fixture(self.record, prepared, self.layout)

    def test_prepared_identity_and_exact_input_ids_are_bound_to_response(self):
        changes = {
            "record_id": "another-response",
            "prompt_length": 2,
            "input_ids": torch.tensor([[10, 12, 13, 21, 22, 2]], dtype=torch.int64),
        }
        for name, value in changes.items():
            with self.subTest(name=name):
                prepared = deepcopy(self.prepared)
                prepared[name] = value
                with self.assertRaises(ValueError):
                    load_saved_fixture(self.record, prepared, self.layout)

    def test_incomplete_prepared_packets_keep_the_original_key_error(self):
        for name in ("label_binding", "target_last_hidden_states"):
            with self.subTest(missing=name):
                prepared = deepcopy(self.prepared)
                del prepared[name]
                with self.assertRaises(KeyError):
                    load_saved_fixture(self.record, prepared, self.layout)

    def test_prepared_binding_preserves_identity_full_prompt_and_layer_order(self):
        changes = {
            "record_id": "another-response",
            "prompt_length": 2,
            "input_ids": [10, 12, 13, 21, 22, 2],
            "target_argmax_ids": [21, 23, 2],
            "target_layer_ids": [3, 1],
            "label_mismatches": [1],
            "native_capture_validated": False,
            "response_only_loss": False,
            "final_feature_logits_used_for_labels": True,
            "capture_backend": "same_frozen_HF_target_canonical_replay",
            "label_binding_scope": "unverified_replay",
        }
        for name, value in changes.items():
            with self.subTest(name=name):
                prepared = deepcopy(self.prepared)
                prepared["label_binding"][name] = value
                with self.assertRaises(ValueError):
                    load_saved_fixture(self.record, prepared, self.layout)

    def test_masks_cannot_supervise_prompt_omit_eos_or_skip_response_tokens(self):
        for mask in ([1, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 0], [0, 0, 0, 1, 0, 1]):
            with self.subTest(mask=mask):
                prepared = deepcopy(self.prepared)
                prepared["loss_mask"] = torch.tensor([mask], dtype=torch.float32)
                with self.assertRaises(ValueError):
                    load_saved_fixture(self.record, prepared, self.layout)

    def test_ids_and_mask_require_the_native_batch_shape_and_dtype(self):
        changes = (
            ("input_ids", self.prepared["input_ids"].to(torch.float32)),
            ("input_ids", self.prepared["input_ids"].squeeze(0)),
            ("loss_mask", self.prepared["loss_mask"].to(torch.bool)),
            ("loss_mask", self.prepared["loss_mask"].squeeze(0)),
        )
        for name, value in changes:
            with self.subTest(name=name, representation=repr(value)):
                prepared = deepcopy(self.prepared)
                prepared[name] = value
                with self.assertRaises(ValueError):
                    load_saved_fixture(self.record, prepared, self.layout)

    def test_prepared_features_require_finite_detached_cpu_bf16_shapes(self):
        for name in ("hidden_states", "target_last_hidden_states"):
            for problem in ("dtype", "width", "rows", "nan", "infinity", "meta", "gradient"):
                with self.subTest(name=name, problem=problem):
                    prepared = deepcopy(self.prepared)
                    value = prepared[name]
                    if problem == "dtype":
                        prepared[name] = value.float()
                    elif problem == "width":
                        prepared[name] = value[..., :-1]
                    elif problem == "rows":
                        prepared[name] = value[..., :-1, :]
                    elif problem == "meta":
                        # A saved meta tensor has no data to map to CPU and
                        # remains meta after loading; no GPU is initialized.
                        prepared[name] = torch.empty_like(value, device="meta")
                    elif problem == "gradient":
                        # torch.save/load preserves this flag, so the real
                        # serialized boundary must reject the non-frozen input.
                        prepared[name] = value.requires_grad_()
                    else:
                        value.reshape(-1)[0] = float("nan" if problem == "nan" else "inf")
                    with self.assertRaises(ValueError):
                        load_saved_fixture(self.record, prepared, self.layout)

    def test_serialized_packet_roundtrip_retains_exact_cpu_tensors(self):
        with tempfile.TemporaryDirectory(prefix="graduate-feature-test-") as directory:
            path = Path(directory) / "prepared.pt"
            torch.save(self.prepared, path)
            saved_bytes = path.read_bytes()
            batch = load_feedback(self.record, path, self.layout)
            for name in ("input_ids", "loss_mask", "hidden_states", "target_last_hidden_states"):
                with self.subTest(name=name):
                    actual = getattr(batch, name)
                    self.assertEqual(actual.device.type, "cpu")
                    self.assertEqual(actual.dtype, self.prepared[name].dtype)
                    self.assertTrue(torch.equal(actual, self.prepared[name]))
            self.assertEqual(batch.stop_token_ids, self.record.stop_token_ids)
            self.assertEqual(path.read_bytes(), saved_bytes)

    def test_reading_and_failure_do_not_consume_rng(self):
        with tempfile.TemporaryDirectory(prefix="graduate-rng-test-") as directory:
            path = Path(directory) / "prepared.pt"
            torch.save(self.prepared, path)
            python_state = random.getstate()
            torch_state = torch.get_rng_state().clone()
            load_feedback(self.record, path, self.layout)
            with self.assertRaises(ValueError):
                load_feedback(self.record, path, replace(self.layout, context_limit=5))
            self.assertEqual(random.getstate(), python_state)
            self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))


class TrainingEntryTests(unittest.TestCase):
    """Real input/persistence path, with model execution and training mocked."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="graduate-training-entry-test-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.packet_directory = self.directory / "feedback-data"
        self.packet_directory.mkdir()
        self.records = []
        self.entries = []
        for index in (7, 8):
            record, _, prepared = feature_fixture()
            record = record.model_copy(update={"record_id": f"response-{index}", "request_index": index})
            prepared["record_id"] = record.record_id
            prepared["label_binding"]["record_id"] = record.record_id
            response_path = self.packet_directory / f"response-{index}.json"
            features_path = self.packet_directory / f"prepared-{index}.pt"
            response_path.write_text(record.model_dump_json())
            torch.save(prepared, features_path)
            self.records.append(record)
            self.entries.append({"response": response_path.name, "features": features_path.name})
        self.manifest_path = self.packet_directory / "manifest.json"
        self.manifest_path.write_text(json.dumps(self.entries))

        target = self.directory / "target"
        draft = self.directory / "draft"
        for path in (target, draft):
            path.mkdir()
        (target / "config.json").write_text(json.dumps({
            "hidden_size": 4, "vocab_size": 64, "num_hidden_layers": 4,
        }))
        (draft / "config.json").write_text(json.dumps({
            "hidden_size": 4, "vocab_size": 64, "num_hidden_layers": 2,
            "block_size": 16, "dflash_config": {"target_layer_ids": [1, 3]},
        }))
        text = (ROOT / "configs" / "fenrir.toml").read_text()
        original = tomllib.loads(text)
        for section, field, value in (
            ("models", "target", "target"), ("models", "draft", "draft"),
            ("feedback", "manifest", "feedback-data/manifest.json"),
            ("feedback", "buffer_dir", "buffer"), ("training", "output_dir", "training-output"),
        ):
            text = text.replace(json.dumps(original[section][field]), json.dumps(value))
        self.config_path = self.directory / "fixture.toml"
        self.config_path.write_text(text)
        self.buffer_directory = self.directory / "buffer"
        self.training_directory = self.directory / "training-output"

        self.parameter = torch.nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))
        self.frozen = torch.nn.Parameter(torch.zeros(1, dtype=torch.bfloat16), requires_grad=False)
        self.model = Mock()
        self.model.parameters.return_value = [self.parameter, self.frozen]
        self.seed = self.enterContext(patch("graduate.training.seed_training"))
        self.load_model = self.enterContext(patch("graduate.model.load_model", return_value=self.model))
        self.adam = self.enterContext(patch("torch.optim.AdamW", return_value=sentinel.optimizer))
        self.train_event = self.enterContext(patch("graduate.training.train_event", side_effect=lambda *args: {
            "responses": 2, "presentations": 4, "optimizer_steps": 2,
        }))
        self.enterContext(patch("torch.cuda.reset_peak_memory_stats"))
        self.enterContext(patch("torch.cuda.max_memory_allocated", return_value=0))

    def run_main(self, debug_responses=2):
        arguments = ["train", "--config", str(self.config_path)]
        if debug_responses is not None:
            arguments += ["--debug-responses", str(debug_responses)]
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(arguments), 0)
        return json.loads(output.getvalue())

    def test_training_entry_passes_exact_feedback_and_persists_inputs(self):
        source_files = {path: path.read_bytes() for path in self.directory.rglob("*") if path.is_file()}
        result = self.run_main()
        self.seed.assert_called_once_with(17)
        self.load_model.assert_called_once()
        config, device = self.load_model.call_args.args
        self.assertEqual(device, "cuda")
        self.assertEqual(list(self.adam.call_args.args[0]), [self.parameter])
        self.assertEqual(self.adam.call_args.kwargs, {"lr": 3e-4, "weight_decay": 0.01})
        model, adam, prepared, training_config, log_path = self.train_event.call_args.args
        self.assertIs(model, self.model)
        self.assertIs(adam, sentinel.optimizer)
        self.assertIs(training_config, config)
        self.assertEqual([batch.record_id for batch in prepared], [record.record_id for record in self.records])
        for batch, record in zip(prepared, self.records):
            self.assertEqual(batch.input_ids.tolist(), [list(record.input_ids)])
        self.assertEqual(result["status"], "debug_training_completed")
        run_directory = Path(result["output_directory"])
        self.assertEqual(log_path, run_directory / "presentations.jsonl")
        self.assertEqual(run_directory.parent, self.training_directory)
        self.assertEqual(json.loads((run_directory / "result.json").read_text()), result)
        inputs = json.loads((run_directory / "inputs.json").read_text())
        self.assertEqual(inputs["record_ids"], [record.record_id for record in self.records])
        self.assertEqual(inputs["debug_responses"], 2)
        self.assertEqual(inputs["manifest"], str(self.manifest_path))
        state = json.loads((self.buffer_directory / "state.json").read_text())
        self.assertEqual(state["records"], [record.model_dump(mode="json") for record in self.records])
        self.assertEqual(state["events"], [])
        for path, original in source_files.items():
            self.assertEqual(path.read_bytes(), original)

    def test_repeated_import_is_unchanged_but_changed_record_identity_is_rejected(self):
        self.run_main()
        state_path = self.buffer_directory / "state.json"
        saved = state_path.read_bytes()
        self.run_main()
        self.assertEqual(state_path.read_bytes(), saved)
        values = self.records[0].model_dump(mode="json")
        values["checkpoint_id"] = "changed-draft"
        (self.packet_directory / self.entries[0]["response"]).write_text(json.dumps(values))
        with self.assertRaisesRegex(ValueError, "Changed content"):
            self.run_main()
        self.assertEqual(self.train_event.call_count, 2)
        self.assertEqual(state_path.read_bytes(), saved)

    def test_full_event_and_invalid_debug_counts_fail_before_any_state_write(self):
        for count in (None, 0, -1, 3):
            with self.subTest(count=count), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    self.run_main(count)
                self.assertEqual(error.exception.code, 2)
        self.assertFalse(self.buffer_directory.exists())
        self.assertFalse(self.training_directory.exists())
        self.load_model.assert_not_called()
        self.train_event.assert_not_called()

    def test_duplicate_manifest_records_fail_before_any_state_write(self):
        self.manifest_path.write_text(json.dumps([self.entries[0], self.entries[0]]))
        with self.assertRaisesRegex(ValueError, "duplicate response identities"):
            self.run_main()
        self.assertFalse(self.buffer_directory.exists())
        self.assertFalse(self.training_directory.exists())
        self.train_event.assert_not_called()

    def test_invalid_second_packet_never_creates_or_mutates_the_buffer(self):
        self.run_main()
        state_path = self.buffer_directory / "state.json"
        saved = state_path.read_bytes()
        _, _, prepared = feature_fixture()
        prepared["record_id"] = self.records[1].record_id
        prepared["label_binding"]["record_id"] = self.records[1].record_id
        prepared["loss_mask"][0, -1] = 0
        torch.save(prepared, self.packet_directory / self.entries[1]["features"])
        with self.assertRaisesRegex(ValueError, "Loss mask must supervise exactly the complete response"):
            self.run_main()
        self.assertEqual(state_path.read_bytes(), saved)
        self.config_path.write_text(self.config_path.read_text().replace('buffer_dir = "buffer"',
                                                                       'buffer_dir = "new-buffer"'))
        with self.assertRaisesRegex(ValueError, "Loss mask must supervise exactly the complete response"):
            self.run_main()
        self.assertFalse((self.directory / "new-buffer").exists())
        self.assertEqual(self.train_event.call_count, 1)

    def test_mismatched_model_pair_is_rejected_before_any_state_write(self):
        path = self.directory / "draft" / "config.json"
        metadata = json.loads(path.read_text())
        metadata["hidden_size"] = 8
        path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "Target and draft disagree on hidden_size"):
            self.run_main()
        self.assertFalse(self.buffer_directory.exists())
        self.assertFalse(self.training_directory.exists())
        self.train_event.assert_not_called()

    def test_manifest_and_packet_paths_resolve_from_their_own_parent_directories(self):
        other_directory = self.directory / "unrelated-working-directory"
        other_directory.mkdir()
        for working_directory in (self.directory, other_directory):
            with self.subTest(working_directory=working_directory.name), chdir(working_directory):
                self.run_main()
                prepared = self.train_event.call_args.args[2]
                self.assertEqual([batch.record_id for batch in prepared],
                                 [record.record_id for record in self.records])
        self.assertEqual(list(other_directory.iterdir()), [])
        state = json.loads((self.buffer_directory / "state.json").read_text())
        self.assertEqual(state["records"], [record.model_dump(mode="json") for record in self.records])


if __name__ == "__main__":
    unittest.main()
