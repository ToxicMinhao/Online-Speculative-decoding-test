"""Storage accounting tests use metadata only, never model work."""
import unittest
import json
from pathlib import Path
import tempfile

from experiments.ok_coding.storage import (
    GIB, WEIGHTS, FEATURE_ROW_BYTES, COMPILER_CACHE_ALLOWANCE, full_campaign_budget,
    StorageLifecycle, identity,
)


def plan_fixture():
    return {"windows": [{"event_index": event,
        "requests": [{"prompt_ids": list(range(224)), "max_new_tokens": 2048}
                     for _ in range(128)]} for event in range(1, 5)]}


class CodingStorageTest(unittest.TestCase):
    def test_two_seeds_keep_eight_endpoints_and_all_pools(self):
        budget = full_campaign_budget(plan_fixture())
        pools = 8 * 128 * 2272 * FEATURE_ROW_BYTES * 2
        endpoints = 8 * 4 * WEIGHTS
        initials = 4 * WEIGHTS
        expected = pools + endpoints + initials + COMPILER_CACHE_ALLOWANCE
        self.assertEqual(budget["final_incremental_payload_bytes"], expected)
        self.assertEqual(budget["feedback_pools"], 8)
        self.assertEqual(budget["retained_endpoint_pairs"], 8)
        self.assertEqual(budget["reserve_bytes"], 8 * GIB)
        self.assertGreater(budget["peak_required_bytes"], expected)

    def test_peak_includes_rolling_states_and_export_overlap(self):
        budget = full_campaign_budget(plan_fixture())
        rows = {row["phase"]: row for row in budget["phases"]}
        pending = rows["seed_29_event_4_K_pending"]["payload_bytes"]
        saved = rows["seed_29_event_4_K_saved"]["payload_bytes"]
        retired = rows["seed_29_event_4_rolling_retired"]["payload_bytes"]
        self.assertEqual(pending - saved, WEIGHTS)
        self.assertEqual(saved - retired, 8 * WEIGHTS)
        self.assertEqual(budget["peak_phase"], "seed_29_event_4_K_pending")

    def test_shorter_inputs_change_only_feature_bound(self):
        plan = plan_fixture()
        full = full_campaign_budget(plan)
        for window in plan["windows"]:
            for request in window["requests"]:
                request["prompt_ids"] = [1]
        short = full_campaign_budget(plan)
        difference = full["final_incremental_payload_bytes"] - short["final_incremental_payload_bytes"]
        self.assertEqual(difference, 8 * 128 * 223 * FEATURE_ROW_BYTES * 2)

    def test_retirement_requires_successor_evidence_and_preserves_current(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = {"storage": {"mode": "retain_local", "offload_assumed": False,
                                "reserve_bytes": 8 * GIB}}
            storage = StorageLifecycle(plan, root, seed=17)
            parents = {}
            for arm in ("O", "K"):
                for event in (1, 2):
                    folder = root / arm / "events" / f"event_{event:03d}"
                    (folder / "export").mkdir(parents=True)
                    state = folder / "training_state.pt"
                    export = folder / "export" / "model.safetensors"
                    state.write_bytes(b"state")
                    export.write_bytes(b"weights")
                    audit = root / f"resume_{event:03d}_{arm}"
                    audit.mkdir()
                    (audit / "STATE_IDENTITIES.json").write_text(json.dumps({
                        str(state): identity(state), str(export): identity(export)}))
                    (audit / "SAVED_ENDPOINT.json").write_text(json.dumps({
                        "status": "passed", "export_exact": True,
                        "completed_event": event, "global_step": event * 128}))
                    (audit / "LIVE_BOUNDARY.json").write_text('{"exact_boundary": true}')
                    parents[arm] = {"state": str(state)}
            with self.assertRaises(ValueError):
                storage.after_event(2, parents, {}, evaluated=False)
            old = root / "O/events/event_001/training_state.pt"
            self.assertTrue(old.exists())
            receipt = storage.after_event(2, parents, {}, evaluated=True)
            self.assertEqual(len(receipt["retired"]), 4)
            self.assertFalse(old.exists())
            self.assertTrue(Path(parents["O"]["state"]).exists())
            self.assertTrue((root / "retention/event_002_RETIREMENT_INTENT.json").exists())


if __name__ == "__main__":
    unittest.main()
