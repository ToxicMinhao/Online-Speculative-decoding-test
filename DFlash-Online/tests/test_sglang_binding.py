"""Admission checks for features captured from actual SGLang serving forwards."""
import copy
import unittest

import torch

from online_feedback.training import validate_feature, validate_target_binding


def captured_feature():
    tokens = [1, 2, 4, 5, 6]
    return {
        "record_id": "serving_1",
        "prompt_length": 2,
        "input_ids": torch.tensor([tokens], dtype=torch.int64),
        "loss_mask": torch.tensor([[0, 0, 1, 1, 1]], dtype=torch.float32),
        "hidden_states": torch.ones(1, len(tokens), 6),
        "target_last_hidden_states": torch.ones(1, len(tokens), 3),
        "label_binding": {
            "capture_backend": "same_frozen_SGLang_native_capture",
            "native_capture_validated": True,
            "label_binding_scope": "native_prefill_and_committed_verification",
            "final_feature_logits_used_for_labels": False,
            "input_ids": list(tokens),
            "target_argmax_ids": tokens[2:],
            "prompt_length": 2,
            "record_id": "serving_1",
            "label_mismatches": [],
        },
    }


class SGLangBindingTests(unittest.TestCase):
    def admit(self, raw):
        # The event trainer validates tensor/mask contracts before provenance.
        validate_feature(raw, max_length=5, hidden_width=6, vocab_size=32)
        validate_target_binding(raw)

    def test_native_capture_accepts_complete_sequence_without_mutation(self):
        raw = captured_feature()
        original = copy.deepcopy(raw)
        self.admit(raw)
        for key in ("input_ids", "loss_mask", "hidden_states", "target_last_hidden_states"):
            self.assertTrue(torch.equal(raw[key], original[key]), key)
        self.assertEqual(raw["label_binding"], original["label_binding"])

    def test_each_provenance_field_is_required(self):
        for field in captured_feature()["label_binding"]:
            with self.subTest(field=field):
                raw = captured_feature()
                del raw["label_binding"][field]
                with self.assertRaisesRegex(ValueError, "canonical target-greedy"):
                    self.admit(raw)

    def test_replay_claim_or_unvalidated_capture_is_rejected(self):
        changes = (
            ("capture_backend", "SGLang"),
            ("native_capture_validated", False),
            ("native_capture_validated", 1),
            ("label_binding_scope", "full_sequence_canonical_replay"),
            ("final_feature_logits_used_for_labels", True),
            ("final_feature_logits_used_for_labels", 0),
            ("label_mismatches", [{"position": 3}]),
        )
        for field, value in changes:
            with self.subTest(field=field, value=value):
                raw = captured_feature()
                raw["label_binding"][field] = value
                with self.assertRaisesRegex(ValueError, "canonical target-greedy"):
                    self.admit(raw)

    def test_capture_cannot_be_reassigned_to_other_prompt_or_record(self):
        for change in ("prompt_tokens", "record_id", "prompt_length"):
            with self.subTest(change=change):
                raw = captured_feature()
                if change == "prompt_tokens":
                    raw["input_ids"][0, 0] = 7
                elif change == "record_id":
                    raw["record_id"] = "serving_2"
                else:
                    raw["label_binding"]["prompt_length"] = 1
                with self.assertRaisesRegex(ValueError, "canonical target-greedy"):
                    self.admit(raw)

    def test_changed_response_or_teacher_argmax_is_rejected(self):
        for change in ("response", "argmax"):
            with self.subTest(change=change):
                raw = captured_feature()
                if change == "response":
                    raw["input_ids"][0, -1] = 7
                    raw["label_binding"]["input_ids"][-1] = 7
                else:
                    raw["label_binding"]["target_argmax_ids"][-1] = 7
                with self.assertRaisesRegex(ValueError, "canonical target-greedy"):
                    self.admit(raw)

    def test_complete_capture_ids_are_required_including_final_token(self):
        changes = (
            [1, 2, 4, 5],
            [1, 2, 4, 5, 6, 7],
            [True, 2, 4, 5, 6],
            (1, 2, 4, 5, 6),
        )
        for value in changes:
            with self.subTest(value=value):
                raw = captured_feature()
                raw["label_binding"]["input_ids"] = value
                with self.assertRaisesRegex(ValueError, "canonical target-greedy"):
                    self.admit(raw)

    def test_argmax_ids_must_be_integer_list(self):
        for value in ([4.0, 5, 6], (4, 5, 6)):
            with self.subTest(value=value):
                raw = captured_feature()
                raw["label_binding"]["target_argmax_ids"] = value
                with self.assertRaisesRegex(ValueError, "canonical target-greedy"):
                    self.admit(raw)

    def test_incomplete_feature_tensor_or_response_mask_is_rejected(self):
        for change in ("last_hidden", "masked_tail", "supervised_prompt"):
            with self.subTest(change=change):
                raw = captured_feature()
                if change == "last_hidden":
                    raw["target_last_hidden_states"] = raw["target_last_hidden_states"][:, :-1]
                else:
                    position = -1 if change == "masked_tail" else 0
                    raw["loss_mask"][0, position] = 1 - raw["loss_mask"][0, position]
                with self.assertRaises(ValueError):
                    self.admit(raw)

    def test_existing_hf_capture_contracts_remain_accepted(self):
        for backend in (
            "same_frozen_HF_target_canonical_replay",
            "same_frozen_HF_target_saved_native_block_replay",
        ):
            with self.subTest(backend=backend):
                raw = captured_feature()
                raw["label_binding"] = {
                    "capture_backend": backend,
                    "record_id": raw["record_id"],
                    "prompt_length": raw["prompt_length"],
                    "target_argmax_ids": [4, 5, 6],
                    "label_mismatches": [],
                }
                if backend.endswith("saved_native_block_replay"):
                    raw["label_binding"]["native_replay"] = {
                        "full_posterior_match": True,
                        "label_binding_scope": "exact_saved_native_prefill_and_verification_blocks",
                        "final_feature_logits_used_for_labels": False,
                    }
                self.admit(raw)

    def test_existing_hf_replay_requirements_remain_enforced(self):
        raw = captured_feature()
        raw["label_binding"]["capture_backend"] = "same_frozen_HF_target_saved_native_block_replay"
        with self.assertRaisesRegex(ValueError, "canonical target-greedy"):
            self.admit(raw)


if __name__ == "__main__":
    unittest.main()
