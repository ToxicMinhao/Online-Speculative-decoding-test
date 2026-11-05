"""Remaining storage uses tiny metadata/files, never scientific model work."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.ok_coding.preflight import storage_admission
from experiments.ok_coding.resume_storage import (
    FIXED_ALLOWANCE, NO_EXISTING_RELEASE_PROOF, _physical_reclamation, allocation_envelope,
    inventory_allocations, remaining_campaign_budget,
)
from experiments.ok_coding.storage import (
    ARMS, COMPILER_CACHE_ALLOWANCE, FEATURE_ROW_BYTES, GIB, SEEDS, WEIGHTS,
    full_campaign_budget, identity,
)


def sealed_length_plan():
    # Public preparation's aggregate prompt lengths; no private IDs or text.
    windows = []
    for event, total in enumerate((5211, 4985, 4538, 5067), 1):
        quotient, remainder = divmod(total, 128)
        requests = [{"prompt_ids": [1] * (quotient + (index < remainder)),
                     "max_new_tokens": 2048} for index in range(128)]
        windows.append({"event_index": event, "requests": requests})
    plan = {"windows": windows}
    plan["storage"] = {"mode": "retain_local", "offload_assumed": False,
                       "reserve_bytes": 8 * GIB,
                       "campaign_capacity": full_campaign_budget(plan)}
    return plan


def boundary_through(seed=None, event=0):
    boundary = {"completed_pools": [], "completed_jobs": [],
                "completed_evaluations": [], "completed_retention": [],
                "next_phase": "fixture_validated_next_phase"}
    for current_seed in SEEDS:
        for current_event in range(1, 5):
            if seed is None or (current_seed, current_event) > (seed, event):
                return boundary
            pair = [current_seed, current_event]
            boundary["completed_pools"].append(pair)
            boundary["completed_jobs"].extend([*pair, arm] for arm in ARMS)
            boundary["completed_retention"].append(pair)
            if current_event in (2, 4):
                boundary["completed_evaluations"].append(pair)
    return boundary


def rolling_files(root, *, arm="O", latest=True):
    folder = root / "seed_17" / arm / "events/event_001"
    (folder / "export").mkdir(parents=True)
    state = folder / "training_state.pt"
    export = folder / "export/model.safetensors"
    state.write_bytes(b"s" * 8192)
    export.write_bytes(b"e" * 4096)
    audit = root / f"seed_17/resume_001_{arm}"
    audit.mkdir(parents=True)
    (audit / "STATE_IDENTITIES.json").write_text(json.dumps({
        str(state): identity(state), str(export): identity(export)}))
    if latest:
        os.link(state, root / "seed_17" / arm / "training_state.pt")
    return state, export


class ResumeStorageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.plan = sealed_length_plan()

    def tearDown(self):
        self.temporary.cleanup()

    def budget(self, boundary):
        return remaining_campaign_budget(self.plan, self.root, boundary)

    def test_empty_boundary_preserves_exact_fresh_threshold(self):
        fresh = full_campaign_budget(self.plan)
        boundary = boundary_through()
        resumed = self.budget(boundary)
        self.assertEqual(fresh["peak_required_bytes"], 366_710_105_681)
        self.assertEqual(resumed["peak_required_bytes"], fresh["peak_required_bytes"])
        self.assertEqual(resumed["full_fresh_bound_bytes"], fresh["peak_required_bytes"])
        required = fresh["peak_required_bytes"]
        for keywords in ({}, {"campaign_root": self.root, "boundary": boundary}):
            self.assertTrue(storage_admission(self.plan, required, 80 * GIB, **keywords)["admitted"])
            self.assertFalse(storage_admission(self.plan, required - 1, 80 * GIB, **keywords)["admitted"])
            self.assertFalse(storage_admission(self.plan, 150_373_031_936, 80 * GIB, **keywords)["admitted"])

    def test_complete_first_seed_admits_only_remaining_headroom(self):
        boundary = boundary_through(17, 4)
        # Existing retained allocations reduce current free already. Their size
        # neither adds new writes nor supplies a retirement credit.
        (self.root / "retained_evidence").write_bytes(b"r" * 8192)
        capacity = self.budget(boundary)
        required = capacity["peak_required_bytes"]
        self.assertEqual(required, 209_627_864_400)
        self.assertLess(required, capacity["full_fresh_bound_bytes"])
        self.assertGreater(capacity["current_baseline_allocated_bytes"], 0)
        self.assertEqual(capacity["admitted_retirement_credits"], [])
        for offset, expected in ((0, True), (-1, False)):
            receipt = storage_admission(self.plan, required + offset, 80 * GIB,
                campaign_root=self.root, boundary=boundary)
            self.assertEqual(receipt["admitted"], expected)
        self.assertFalse(storage_admission(self.plan, required, 80 * GIB)["admitted"])
        self.assertFalse(storage_admission(self.plan, required, 80 * GIB - 1,
            campaign_root=self.root, boundary=boundary)["admitted"])

    def test_pending_export_overlap_future_retirement_caps_and_reserves(self):
        capacity = self.budget(boundary_through(17, 4))
        rows = {row["phase"]: row for row in capacity["phases"]}
        pending = rows["seed_29_event_4_K_pending"]
        saved = rows["seed_29_event_4_K_saved"]
        retired = rows["seed_29_event_4_rolling_retired"]
        self.assertEqual(pending["temporary_bytes"], 5 * WEIGHTS)
        self.assertEqual(pending["new_live_payload_bytes"] - saved["new_live_payload_bytes"], WEIGHTS)
        self.assertEqual(saved["new_live_payload_bytes"] - retired["new_live_payload_bytes"], 8 * WEIGHTS)
        before_capture = rows["seed_29_K_initial_saved"]["new_live_payload_bytes"]
        capture = rows["seed_29_event_1_capture"]["new_live_payload_bytes"]
        prepared = rows["seed_29_event_1_prepared"]["new_live_payload_bytes"]
        expected_pool = (5211 + 128 * 2048) * FEATURE_ROW_BYTES
        self.assertEqual(capture - before_capture, expected_pool)
        self.assertEqual(prepared - capture, expected_pool)
        self.assertEqual(capacity["compiler_cache_allowance_bytes"], 16 * GIB)
        self.assertEqual(rows["resume_current"]["new_live_payload_bytes"], COMPILER_CACHE_ALLOWANCE)
        self.assertEqual(capacity["compact_and_workspace_bytes"], 4 * GIB)
        self.assertEqual(capacity["reserve_bytes"], 8 * GIB)
        self.assertEqual(pending["required_bytes"], int(pending["new_live_payload_bytes"] * 1.10) + 12 * GIB)

    def test_lower_future_output_cap_is_rejected(self):
        self.plan["windows"][0]["requests"][0]["max_new_tokens"] = 16
        with self.assertRaisesRegex(ValueError, "2048-token"):
            self.budget(boundary_through(17, 4))

    def test_only_missing_initial_states_and_jobs_are_new_writes(self):
        boundary = boundary_through()
        boundary["completed_pools"] = [[17, 1]]
        boundary["completed_jobs"] = [[17, 1, "O"]]
        capacity = self.budget(boundary)
        names = {row["phase"] for row in capacity["phases"]}
        self.assertNotIn("seed_17_O_initial_pending", names)
        self.assertNotIn("seed_17_event_1_capture", names)
        self.assertNotIn("seed_17_event_1_prepared", names)
        self.assertNotIn("seed_17_event_1_O_pending", names)
        self.assertIn("seed_17_K_initial_pending", names)
        self.assertIn("seed_17_event_1_K_pending", names)

    def test_hardlinks_count_once_without_assumed_release_after_retirement(self):
        state, export = rolling_files(self.root)
        before = {path: path.read_bytes() for path in (state, export)}
        capacity = self.budget(boundary_through(17, 1))
        inventory = capacity["inventory"]
        state_row = next(row for row in inventory["inodes"] if str(state) in row["aliases"])
        self.assertEqual(len(state_row["aliases"]), 2)
        self.assertEqual(state_row["allocated_bytes"], state.stat().st_blocks * 512)
        self.assertEqual(capacity["admitted_retirement_credits"], [])
        self.assertEqual(state_row["retirement_credit_reason"], NO_EXISTING_RELEASE_PROOF)
        rows = {row["phase"]: row for row in capacity["phases"]}
        self.assertEqual(rows["seed_17_event_2_K_pending"]["existing_reclaimed_bytes"], 0)
        self.assertEqual(rows["seed_17_event_2_evaluated_before_retirement"]["existing_reclaimed_bytes"], 0)
        retired = rows["seed_17_event_2_rolling_retired"]
        self.assertEqual(retired["existing_reclaimed_bytes"], 0)
        self.assertEqual(retired["required_bytes"], FIXED_ALLOWANCE
                         + retired["slack_adjusted_new_bytes"])
        self.assertEqual({path: path.read_bytes() for path in (state, export)}, before)

    @patch("experiments.ok_coding.resume_storage._physical_reclamation", wraps=_physical_reclamation)
    def test_outside_or_retained_alias_receives_no_credit(self, proof):
        state, export = rolling_files(self.root)
        with tempfile.TemporaryDirectory() as outside:
            os.link(state, Path(outside) / "outside_state")
            os.link(export, self.root / "retained_export")
            capacity = self.budget(boundary_through(17, 1))
        self.assertEqual(capacity["admitted_retirement_credits"], [])
        proof.assert_not_called()
        reasons = [row["retirement_credit_reason"] for row in capacity["inventory"]["inodes"]]
        self.assertIn("Outside or unaccounted hard-link alias remains", reasons)
        self.assertIn("Inventoried alias is not scheduled for final unlink", reasons)

    def test_stale_latest_is_not_credited_after_successor_save(self):
        state, _ = rolling_files(self.root)
        boundary = boundary_through(17, 1)
        boundary["completed_pools"].append([17, 2])
        boundary["completed_jobs"].append([17, 2, "O"])
        capacity = self.budget(boundary)
        self.assertFalse(any(str(state) in row["aliases"] for row in capacity["admitted_retirement_credits"]))

    def test_sparse_logical_size_cannot_supply_physical_credit(self):
        state, _ = rolling_files(self.root, latest=False)
        with state.open("wb") as stream:
            stream.seek(32 * 1024 * 1024)
            stream.write(b"x")
        bindings_path = self.root / "seed_17/resume_001_O/STATE_IDENTITIES.json"
        bindings = json.loads(bindings_path.read_text())
        bindings[str(state)] = identity(state)
        bindings_path.write_text(json.dumps(bindings))
        capacity = self.budget(boundary_through(17, 1))
        row = next(row for row in capacity["inventory"]["inodes"] if str(state) in row["aliases"])
        self.assertLess(row["allocated_bytes"], row["logical_bytes"])
        self.assertEqual(row["allocated_bytes"], state.stat().st_blocks * 512)
        self.assertEqual(row["retirement_credit_bytes"], 0)

    @patch("experiments.ok_coding.resume_storage._physical_reclamation", wraps=_physical_reclamation)
    def test_other_filesystem_and_unbound_files_receive_no_credit(self, proof):
        state, export = rolling_files(self.root, latest=False)
        extra = export.parent / "unbound_weights"
        extra.write_bytes(b"x" * 4096)
        inventory = inventory_allocations(self.root)
        for row in inventory["inodes"]:
            if str(state) in row["aliases"] or str(export) in row["aliases"]:
                row["same_filesystem"] = False
                row["device"] += 1
        inventory["unique_allocated_bytes"] = sum(row["allocated_bytes"] for row in inventory["inodes"]
                                                  if row["same_filesystem"])
        with patch("experiments.ok_coding.resume_storage.inventory_allocations", return_value=inventory):
            capacity = self.budget(boundary_through(17, 1))
        self.assertEqual(capacity["admitted_retirement_credits"], [])
        proof.assert_not_called()

    def test_slack_precedes_retirement_and_fixed_reserve_does_not_cancel(self):
        # Pure accounting example. Production admission proves no release and
        # never supplies a nonzero existing-file credit to this calculation.
        self.assertEqual(allocation_envelope(100, 90), 20)
        self.assertNotEqual(allocation_envelope(100, 90), int((100 - 90) * 1.10))
        self.assertEqual(FIXED_ALLOWANCE + allocation_envelope(100, 1000), 12 * GIB)

    def test_existing_rolling_files_do_not_reduce_remaining_threshold(self):
        boundary = boundary_through(17, 1)
        empty = self.budget(boundary)
        state, export = rolling_files(self.root, latest=False)
        retained = self.budget(boundary)
        self.assertEqual(state.stat().st_nlink, 1)
        self.assertEqual(export.stat().st_nlink, 1)
        self.assertEqual(retained["peak_required_bytes"], empty["peak_required_bytes"])
        self.assertGreater(retained["current_baseline_allocated_bytes"], 0)
        self.assertEqual(retained["admitted_retirement_credits"], [])
        required = retained["peak_required_bytes"]
        for offset, expected in ((0, True), (-1, False)):
            receipt = storage_admission(self.plan, required + offset, 80 * GIB,
                campaign_root=self.root, boundary=boundary)
            self.assertEqual(receipt["admitted"], expected)

    def test_symlink_is_not_counted_as_target_or_reclamation(self):
        state, _ = rolling_files(self.root)
        link = self.root / "alias_symlink"
        link.symlink_to(state)
        inventory = inventory_allocations(self.root)
        self.assertIn(str(link), inventory["excluded_nonregular_paths"])
        self.assertFalse(any(str(link) in row["aliases"] for row in inventory["inodes"]))

    def test_open_handles_and_unknown_release_always_receive_zero_credit(self):
        state, _ = rolling_files(self.root, latest=False)
        row = next(row for row in inventory_allocations(self.root)["inodes"] if str(state) in row["aliases"])
        self.assertGreater(row["allocated_bytes"], 0)
        self.assertEqual(row["link_count"], 1)
        with state.open("rb") as holder:
            self.assertFalse(holder.closed)
            proof = _physical_reclamation(state, row)
            capacity = self.budget(boundary_through(17, 1))
        self.assertEqual(proof, {"bytes": 0, "reason": NO_EXISTING_RELEASE_PROOF})
        self.assertEqual(capacity["admitted_retirement_credits"], [])
        self.assertTrue(all(phase["existing_reclaimed_bytes"] == 0 for phase in capacity["phases"]))
        self.assertEqual(_physical_reclamation(state, row)["bytes"], 0)


if __name__ == "__main__":
    unittest.main()
