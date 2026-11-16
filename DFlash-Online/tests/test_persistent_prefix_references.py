"""Reference validity checks use verifier observations, not candidate agreement."""
from copy import deepcopy
import unittest

from experiments.persistent_prefix_withholding.references import canonical_record, record_errors


def fixture():
    block = dict(draft_tokens=[2, 8], target_argmax=[2, 99, 4], verify_size=3,
        committed_progress=2, output_start=0, anchor_token=1,
        committed_tokens=[2, 99], positions=[1, 2, 3], accepted_draft_tokens=1,
        native_accepted_draft_tokens=1, raw_accepted_draft_tokens_before_stop=1,
        stop_clipped=False)
    capture = dict(status="passed", native_capture_validated=True,
        label_binding_scope="native_prefill_and_committed_verification",
        prompt_ids=[90], response_ids=[1, 2, 99], blocks=[block],
        target_argmax_ids=[1, 2, 99], covered_token_ids=[90, 1, 2], covered_rows=3,
        collect_features=False, feature_bytes=0)
    record = dict(prompt_ids=[90], response_ids=[1, 2, 99], finish_reason="eos",
        max_new_tokens=2048, stop_token_ids=[99], checkpoint_id="original-F",
        metadata=dict(capture=capture, trace=[block], block_size=16, context_limit=4096,
            runtime="pinned_SGLang_FA3", cache_policy="idle_full_flush_before_every_request_and_publication",
            feedback_capture_in_timed_request=False, verification_calls=1, committed_progress=2))
    request = dict(prompt_ids=[90], stop_token_ids=[99], source_id="gsm8k:main:test:1")
    return record, request


class PersistentReferenceTests(unittest.TestCase):
    def test_complete_target_commit_trace_passes(self):
        record, request = fixture()
        self.assertEqual(record_errors(record, request), [])

    def test_first_output_must_match_observed_target_prefill(self):
        record, request = fixture()
        record["metadata"]["capture"]["target_argmax_ids"][0] = 17
        self.assertIn("captured_target_argmax_binding", record_errors(record, request))

    def test_late_draft_match_cannot_override_first_error(self):
        record, request = fixture()
        record["metadata"]["trace"][0]["draft_tokens"] = [7, 99]
        self.assertIn("block_0_native_decision", record_errors(record, request))

    def test_stop_and_progress_mismatch_is_invalid(self):
        record, request = fixture()
        record["finish_reason"] = "length"
        self.assertIn("termination", record_errors(record, request))
        record["metadata"]["committed_progress"] = 3
        self.assertIn("native_count_identity", record_errors(record, request))

    def test_capture_disabled_and_native_cache_policy_required(self):
        record, request = fixture()
        record["metadata"]["capture"]["collect_features"] = True
        record["metadata"]["cache_policy"] = "different"
        errors = record_errors(record, request)
        self.assertIn("feedback_capture_enabled", errors)
        self.assertIn("native_serving_contract", errors)

    def test_single_first_token_eos_with_zero_calls_is_valid(self):
        record, request = fixture()
        record["response_ids"] = [99]
        record["metadata"].update(trace=[], verification_calls=0, committed_progress=0)
        record["metadata"]["capture"].update(response_ids=[99], blocks=[],
            target_argmax_ids=[99], covered_token_ids=[90], covered_rows=1)
        self.assertEqual(record_errors(record, request), [])

    def test_reference_binding_is_native_and_does_not_change_raw_record(self):
        record, request = fixture()
        before = deepcopy(record)
        reference = canonical_record(request, record, "/retained/F.json",
                                     {"bytes": 10, "sha256": "a" * 64}, {"receipt": "proof.json"})
        self.assertEqual(record, before)
        self.assertEqual(reference["reference_kind"], "native_greedy_target_verifier")
        self.assertTrue(reference["verified"])
        self.assertEqual(len(reference["binding_digest"]), 64)
        changed = canonical_record(request, record, "/different/F.json",
                                   {"bytes": 10, "sha256": "a" * 64}, {"receipt": "proof.json"})
        self.assertNotEqual(reference["binding_digest"], changed["binding_digest"])


if __name__ == "__main__":
    unittest.main()
