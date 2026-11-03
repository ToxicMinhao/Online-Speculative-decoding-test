"""Standalone CPU qualification of the separate auditor; no experiment imports."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from audit_math import admit_fresh, selected_records, endpoint_statistics
from audit_track2 import Audit, run


def records(event, vector=(1.0, 0.0)):
    return [dict(record_id=f"r{i}", global_ordinal=i, event_index=event, descriptor=list(vector))
            for i in range((event - 1) * 128, event * 128)]


class AuditorTests(unittest.TestCase):
    def test_hand_computed_facility_coverage_and_exact_zero_gain_ties(self):
        old, fresh = records(1), records(2)
        old[0]["descriptor"] = [0.6, 0.8]
        old[2]["descriptor"] = [0.0, 1.0]
        for row in fresh[64:]:
            row["descriptor"] = [0.0, 1.0]
        result = selected_records(2, fresh, list(reversed(old)))
        self.assertEqual(result["coverage"]["record_ids"], [f"r{i}" for i in range(32)])
        np.testing.assert_allclose([step["coverage"] for step in result["coverage"]["steps"][:3]],
                                   [.7, .9, 1.0], rtol=0, atol=5e-16)
        self.assertTrue(all(step["marginal_gain"] == 0 for step in result["coverage"]["steps"][3:]))
        zero = selected_records(2, records(2), records(1, (-1.0, 0.0)))
        self.assertEqual(zero["coverage"]["coverage"], 0)
        self.assertEqual(zero["coverage"]["record_ids"], [f"r{i}" for i in range(32)])

    def test_reservoir_uses_one_continuing_stream_and_exact_unique_admissions(self):
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([20261102, 0, 2])))
        slots, seen, steps = admit_fresh([], records(1), rng, 0)
        self.assertEqual([r["global_ordinal"] for r in slots], list(range(128)))
        self.assertTrue(all(step["draw"] is None for step in steps))
        reference = np.random.Generator(np.random.PCG64(np.random.SeedSequence([20261102, 0, 2])))
        expected = list(range(128))
        for ordinal in range(128, 1024):
            choice = int(reference.integers(ordinal + 1))
            if choice < 128:
                expected[choice] = ordinal
        for event in range(2, 9):
            slots, seen, _ = admit_fresh(slots, records(event), rng, seen)
        self.assertEqual(seen, 1024)
        self.assertEqual([r["global_ordinal"] for r in slots], expected)
        self.assertEqual(rng.bit_generator.state, reference.bit_generator.state)

    def test_stratified_bootstrap_preserves_opposite_domain_effects(self):
        paired = []
        for ordinal in range(256):
            calls = {"K": 10, "U": 9, "R": 8 if ordinal % 2 == 0 else 10}
            paired.append({arm: dict(verification_calls=count, accepted_draft_tokens=10, committed_progress=19,
                output_tokens=20, request_latency_s=float(count), native_generate_s=float(count), cache_maintenance_s=0.0)
                for arm, count in calls.items()})
        before = np.random.get_state()
        summary = endpoint_statistics(paired)
        effect = summary["comparisons"]["R_vs_U"]
        self.assertAlmostEqual(effect["call_reduction"]["point_pct"], 0)
        np.testing.assert_allclose(effect["call_reduction"]["ci95_pct"], [0, 0], rtol=0, atol=1e-12)
        self.assertEqual(effect["request_wins"], 128)
        self.assertEqual(effect["request_regressions"], 128)
        self.assertEqual(effect["gross_calls_saved"], 128)
        self.assertEqual(effect["gross_calls_added"], 128)
        self.assertGreater(summary["domains"]["math"]["comparisons"]["R_vs_U"]["call_reduction"]["point_pct"], 0)
        self.assertLess(summary["domains"]["coding"]["comparisons"]["R_vs_U"]["call_reduction"]["point_pct"], 0)
        np.testing.assert_array_equal(before[1], np.random.get_state()[1])

    def test_native_raw_trace_reconstructs_counts_and_rejects_posterior_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            block = dict(draft_tokens=[11], target_argmax=[11, 12], verify_size=2, committed_progress=2,
                anchor_token=10, output_start=0, committed_tokens=[11, 12], positions=[2, 3],
                accepted_draft_tokens=1, native_accepted_draft_tokens=1,
                raw_accepted_draft_tokens_before_stop=1, stop_clipped=False)
            capture = dict(collect_features=False, native_capture_validated=True, feature_bytes=0,
                           cache_maintenance_s=.1, prompt_ids=[1, 2], response_ids=[10, 11, 12], blocks=[block])
            metadata = dict(feedback_capture_in_timed_request=False, capture=capture, block_size=16, context_limit=4096,
                runtime="pinned_SGLang_FA3", cache_policy="idle_full_flush_before_every_request_and_publication",
                trace=[block], verification_calls=1, accepted_draft_tokens=1, committed_progress=2,
                request_latency_s=1.0, native_generate_s=.9, native_trace_path=str(root / "native.json"))
            raw = dict(metadata=metadata, response_ids=[10, 11, 12], prompt_ids=[1, 2], stop_token_ids=[12],
                       max_new_tokens=2048, purpose="evaluation", finish_reason="eos")
            request = dict(prompt_ids=[1, 2], stop_token_ids=[12])
            (root / "native.json").write_text(json.dumps(capture))
            audit = Audit(root)
            counts = audit.native_record(raw, request, capture=False, canonical=raw)
            self.assertEqual(audit.errors, [])
            self.assertEqual(counts["verification_calls"], 1)
            self.assertEqual(counts["accepted_draft_tokens"], 1)
            corrupted = deepcopy(raw)
            corrupted["metadata"]["trace"][0]["target_argmax"][0] = 99
            corrupted["metadata"]["capture"]["blocks"] = corrupted["metadata"]["trace"]
            (root / "native.json").write_text(json.dumps(corrupted["metadata"]["capture"]))
            broken = Audit(root)
            broken.native_record(corrupted, request, capture=False)
            self.assertTrue(any("posterior commit" in error for error in broken.errors))

    def test_unknown_work_is_retained_and_no_efficacy_on_missing_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "evidence"
            root.mkdir()
            (root / "work.jsonl").write_text(json.dumps(dict(kind="step", ordinal=1, phase="started")) + "\n")
            audit = Audit(root)
            work = audit.work_ledger("work.jsonl", {"step": 1})
            self.assertEqual(work["unknown"], {"step": 1})
            self.assertTrue(audit.errors)
            result = run(root, Path(directory) / "review")
            self.assertEqual(result["status"], "invalid_or_incomplete")
            self.assertFalse(result["efficacy_interpretation_allowed"])
            self.assertEqual(len(result["fallacies"]), 11)
            self.assertTrue((Path(directory) / "review/FINAL_AUDIT_INPUTS.json").is_file())


if __name__ == "__main__":
    unittest.main()
