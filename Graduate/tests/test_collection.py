"""Tiny CPU capture fixtures verify alignment, not native model execution."""

import json
from pathlib import Path
import tempfile
import unittest

import torch

from graduate.batches import FeatureLayout, load_feedback
from graduate.collection import FeedbackCapture, GenerationRequest, read_requests
from graduate.feedback import ResponseRecord


def features(rows):
    auxiliary = torch.tensor([[row, row + 1, row + 2, row + 3] for row in rows], dtype=torch.bfloat16)
    final = torch.tensor([[100 + row, 150 + row] for row in rows], dtype=torch.bfloat16)
    return auxiliary, final


def record(response, *, cap=8):
    return ResponseRecord(
        record_id="request-1", session_id="capture-test", request_index=0,
        checkpoint_id="released-draft", purpose="serving", prompt_ids=(11, 12),
        response_ids=tuple(response), finish_reason="eos" if response[-1] == 2 else "length",
        stop_token_ids=(2,), max_new_tokens=cap,
    )


class RequestTests(unittest.TestCase):
    def read(self, values):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "requests.json"
            path.write_text(json.dumps(values))
            return read_requests(path)

    def test_request_preserves_text_and_explicit_evaluation_role(self):
        text = "  Explain this.\nKeep the formatting."
        requests = self.read([{"record_id": "eval-1", "prompt": text, "role": "evaluation"}])
        self.assertEqual(requests[0].prompt, text)
        self.assertEqual(requests[0].role, "evaluation")
        with self.assertRaises(ValueError):
            GenerationRequest(record_id="bad", prompt="text", role="training")

    def test_input_rejects_duplicate_ids_cross_role_overlap_and_empty_text(self):
        invalid = [
            [],
            [{"record_id": "same", "prompt": "one"}, {"record_id": "same", "prompt": "two"}],
            [{"record_id": "a", "prompt": "The  prompt"},
             {"record_id": "b", "prompt": " the\nPROMPT ", "role": "evaluation"}],
            [{"record_id": "blank", "prompt": " \n "}],
        ]
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.read(values)


class CaptureTests(unittest.TestCase):
    def capture(self, first=21, backend="sglang_live_capture"):
        capture = FeedbackCapture([11, 12], [1, 3], backend)
        capture.prefill(first, *features([0, 1]))
        return capture

    def save(self, capture, response_record):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prepared.pt"
            capture.save(response_record, path)
            return torch.load(path, map_location="cpu", weights_only=True)

    def test_zero_acceptance_then_all_accepted_preserves_exact_unshifted_rows(self):
        capture = self.capture()
        auxiliary, final = features([2, 80, 90])
        # The last two input histories are rejected, including their feature NaNs.
        auxiliary[1:] = float("nan")
        final[1:] = float("nan")
        capture.verify([21, 41, 42], [22, 88, 89], 1, auxiliary, final)
        capture.verify([22, 23, 24], [23, 24, 2], 3, *features([3, 4, 5]))
        capture.final_token(2, *features([6]))
        packet = self.save(capture, record([21, 22, 23, 24, 2]))

        self.assertEqual(packet["input_ids"].tolist(), [[11, 12, 21, 22, 23, 24, 2]])
        self.assertEqual(packet["loss_mask"].tolist(), [[0, 0, 1, 1, 1, 1, 1]])
        expected_auxiliary, expected_final = features([0, 1, 2, 3, 4, 5, 6])
        self.assertTrue(torch.equal(packet["hidden_states"][0], expected_auxiliary))
        self.assertTrue(torch.equal(packet["target_last_hidden_states"][0], expected_final))
        self.assertEqual(packet["label_binding"]["target_argmax_ids"], [21, 22, 23, 24, 2])

    def test_first_prefill_eos_is_retained_without_inventing_training_anchors(self):
        capture = self.capture(first=2)
        capture.final_token(2, *features([7]))
        response_record = record([2])
        packet = self.save(capture, response_record)
        self.assertEqual(response_record.eligible_anchor_count, 0)
        self.assertEqual(packet["input_ids"].tolist(), [[11, 12, 2]])
        self.assertEqual(packet["label_binding"]["target_argmax_ids"], [2])
        self.assertEqual(packet["hidden_states"].shape, (1, 3, 4))

    def test_eos_and_output_cap_clip_before_feature_selection(self):
        cases = [
            ([21, 2, 41], [2, 41, 42], 1, [21, 2], 8),
            ([21, 22, 23], [22, 23, 24], 2, [21, 22, 23], 3),
        ]
        for candidates, posterior, progress, response, cap in cases:
            with self.subTest(response=response):
                capture = self.capture()
                capture.verify(candidates, posterior, progress, *features([2, 3, 99]))
                capture.final_token(response[-1], *features([10]))
                packet = self.save(capture, record(response, cap=cap))
                self.assertEqual(capture.response_ids, response)
                self.assertEqual(packet["hidden_states"][0, :, 0].tolist(),
                                 [0, 1] + [2, 3][:progress] + [10])
                self.assertEqual(packet["label_binding"]["target_argmax_ids"], response)
                self.assertFalse(packet["label_binding"]["final_feature_logits_used_for_labels"])

    def test_capture_owns_detached_feature_storage_before_next_runtime_call(self):
        capture = FeedbackCapture([11, 12], [1, 3], "sglang_live_capture")
        auxiliary, final = features([0, 1])
        auxiliary.requires_grad_()
        final.requires_grad_()
        capture.prefill(2, auxiliary, final)
        with torch.no_grad():
            auxiliary.fill_(99)
            final.fill_(99)
        capture.final_token(2, *features([2]))
        packet = self.save(capture, record([2]))
        self.assertEqual(packet["hidden_states"][0, :2, 0].tolist(), [0, 1])
        self.assertFalse(packet["hidden_states"].requires_grad)
        self.assertFalse(packet["target_last_hidden_states"].requires_grad)

    def test_invalid_token_geometry_rejected_suffix_and_incomplete_tail_fail(self):
        for token in (True, 11.0, -1):
            with self.subTest(token=token), self.assertRaises(ValueError):
                FeedbackCapture([token], [1, 3], "sglang_live_capture")
        capture = self.capture()
        for candidates, posterior, progress in (([22, 23], [23, 2], 2),
                                                ([21, 41], [22, 2], 2),
                                                ([21, 22], [22, 2], True)):
            with self.subTest(candidates=candidates, progress=progress), self.assertRaises(ValueError):
                capture.verify(candidates, posterior, progress, *features([2, 3]))
        with self.assertRaises(ValueError):
            capture.final_token(22, *features([4]))
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
            capture.save(record([21, 2]), Path(directory) / "incomplete.pt")

    def test_feature_dtype_width_and_nonfinite_committed_rows_are_not_converted(self):
        auxiliary, final = features([0, 1])
        bad_pairs = [(auxiliary.float(), final), (auxiliary, final[:, :1]),
                     (auxiliary[:1], final), (auxiliary, final * float("nan"))]
        for pair in bad_pairs:
            with self.subTest(shapes=[item.shape for item in pair]):
                capture = FeedbackCapture([11, 12], [1, 3], "sglang_live_capture")
                with self.assertRaises(ValueError):
                    capture.prefill(2, *pair)

    def test_saved_packet_preserves_backend_identity_and_is_immutable(self):
        capture = self.capture()
        capture.verify([21, 42], [2, 99], 1, *features([2, 99]))
        capture.final_token(2, *features([3]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prepared.pt"
            receipt = capture.save(record([21, 2]), path)
            original = path.read_bytes()
            self.assertEqual(receipt["capture_backend"], "sglang_live_capture")
            with self.assertRaises(FileExistsError):
                capture.save(record([21, 2]), path)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual([item.name for item in path.parent.iterdir()], ["prepared.pt"])
            with self.assertRaises(ValueError):
                capture.save(record([21, 22, 2]), path.parent / "wrong.pt")
            self.assertFalse((path.parent / "wrong.pt").exists())

    def test_existing_sglang_packet_contract_roundtrips_through_step4_loader(self):
        capture = self.capture(backend="same_frozen_SGLang_native_capture")
        capture.verify([21, 42], [2, 99], 1, *features([2, 99]))
        capture.final_token(2, *features([3]))
        response_record = record([21, 2])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prepared.pt"
            capture.save(response_record, path)
            batch = load_feedback(response_record, path, FeatureLayout(16, 128, 2, (1, 3)))
            self.assertEqual(batch.input_ids.tolist(), [[11, 12, 21, 2]])
            self.assertEqual(batch.hidden_states[0, :, 0].tolist(), [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
