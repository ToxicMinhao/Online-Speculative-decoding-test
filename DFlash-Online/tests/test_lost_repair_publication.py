"""Small CPU fixtures for publication privacy, receipts and partial evidence."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

from experiments.lost_repair.publish_supplement import publish, starts, work_accounting


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(root):
    campaign, sealed = root / "campaign", root / "sealed"
    secret = "PRIVATE_PAYLOAD_CANARY_982341"
    plan = dict(execution_commit="abc123", requests=[{"request_id": "r0", "prompt": secret}],
                original_row_range=[768, 896], original_rows_canonical_json_sha256="batchhash",
                recovery_path="/private/state.pt", pinned_files={"/private/state.pt": {"sha256": "statehash"}},
                checkpoints={"K6": {"export": "/private/export", "weights_identity": {"sha256": "weighthash"}}})
    save(campaign / "PLAN.json", plan)
    save(sealed / "PLAN.json", plan)
    save(sealed / "SEAL.json", {"status": "sealed"})
    (sealed / "config_KPLUS.yaml").write_text("model: KPLUS\n")
    source = root / "SOURCE.json"
    save(source, {"commit": "abc123", "files": {}})
    row = dict(request_id="r0", canonical_start=0, round_ordinal=1, valid_proposals=15,
               a4=8, a5=15, a6=8, genuine_k4_rejection=True, genuine_k5_rejection=False,
               genuine_k6_rejection=True, parity_valid=True, terminal=False, stop_clipped=False,
               private_tokens=[982341, 912311], prompt=secret)
    save(campaign / "census/CENSUS.json", [row])
    for arm in ("K6", "K4", "K5"):
        save(campaign / f"census/r0/0000_{arm}.json", dict(
            status="completed", cap_unchanged=True, output_matches_canonical_prefix=True,
            private_tokens=[982341, 912311], prompt=secret,
            reconstruction=dict(status="passed", target_prefix_cache_exact=True,
                                neutral_draft_prefix_cache_exact=True if arm == "K6" else None,
                                destination_draft_rebuilt=True)))
    save(campaign / "census/r0/0000_PARITY.json", dict(status="passed", differences=[],
            all_15_proposals_exact=True, expected_sha256="e", observed_sha256="o"))
    save(campaign / "STATUS.json", dict(status="completed", census_complete=True, training_complete=True))
    save(campaign / "POSTFLIGHT.json", dict(status="passed", inputs_and_checkpoints_unchanged=True, source_unchanged=True))
    save(campaign / "RESULT.json", dict(status="completed", wall_s=10,
            phase_costs={"census": {"wall_s": 7, "exit_code": 0}, "training": {"wall_s": 2, "exit_code": 0}},
            optimizer_steps=0, backward_calls=0, timed_endpoints=0, prompt=secret))
    save(campaign / "BUDGET.json", dict(gpu_seconds=9, gpu_seconds_limit=14400, attempted={}, completed={}, failed={}, unknown={}))
    save(campaign / "BUDGET_CONNECTION.json", dict(auth=secret))
    save(campaign / "training/INPUTS.json", dict(inputs=[dict(path="/private/state.pt", sha256="statehash")], config_matches_recovery=True))
    save(campaign / "training/SCHEDULE.json", {"records": [{"request_id": "r0", "private_tokens": [982341]}]})
    save(campaign / "training/RESULT.json", dict(initialization_s=0.5, schedule_s=0.1, wall_s=1.4,
            phase_wall_s=2, response_forwards=256, projection_chunks=1024, optimizer_steps=0,
            backwards=0, peak_allocated_bytes=1024, caller_rng_and_modes_restored=True,
            parameters_and_gradients_unchanged=True, private_tokens=[982341], prompt=secret))
    work = [dict(kind="draft_block", status="attempted", ticket=1, gpu_seconds=0,
                 metadata={"startup": True, "operation": "startup_draft", "prompt": secret}),
            dict(kind="draft_block", status="completed", ticket=1, gpu_seconds=0.4),
            dict(kind="draft_block", status="attempted", ticket=2, gpu_seconds=1,
                 metadata={"operation": "native_proposal_plus_verification"}),
            dict(kind="draft_block", status="completed", ticket=2, gpu_seconds=1.5)]
    (campaign / "WORK.jsonl").write_text("\n".join(map(json.dumps, work)) + "\n")
    return campaign, sealed, source, secret


class SupplementTests(unittest.TestCase):
    def test_complete_receipts_and_private_payloads(self):
        with tempfile.TemporaryDirectory() as folder:
            campaign, sealed, source, secret = fixture(Path(folder))
            before = (campaign / "RESULT.json").read_bytes()
            output = publish(campaign, sealed, source)
            destination = Path(output["output"])
            self.assertEqual({p.name for p in destination.iterdir()}, {
                "ALL_STARTS.csv", "PARITY_RECEIPTS.csv", "PARITY_SUMMARY.json",
                "IDENTITIES.json", "COST_ACCOUNTING.json", "supplement.md"})
            for path in destination.iterdir():
                self.assertNotIn(secret, path.read_text())
                self.assertNotIn("912311", path.read_text())
            self.assertEqual(before, (campaign / "RESULT.json").read_bytes())
            identities = json.loads((destination / "IDENTITIES.json").read_text())
            self.assertIsNotNone(identities["actual_run_plan"])
            self.assertIsNotNone(identities["schedule_observed"])
            self.assertEqual(identities["recovery"]["declared_identity"]["sha256"], "statehash")
            parity = json.loads((destination / "PARITY_SUMMARY.json").read_text())
            self.assertEqual(parity["arms"]["K6"]["checks"]["acceptance_exact"]["passed"], 1)
            self.assertEqual(parity["arms"]["K4"]["checks"]["acceptance_exact"]["unavailable"], 1)

    def test_unknown_partial_receipts_are_not_zero_or_success(self):
        with tempfile.TemporaryDirectory() as folder:
            campaign, sealed, source, _ = fixture(Path(folder))
            save(campaign / "STATUS.json", dict(status="failed", census_complete=False, training_complete=False))
            (campaign / "training/RESULT.json").unlink()
            (campaign / "census/r0/0000_K5.json").unlink()
            output = publish(campaign, sealed, source)
            destination = Path(output["output"])
            parity = json.loads((destination / "PARITY_SUMMARY.json").read_text())
            self.assertIsNone(parity["full_census_denominator"])
            self.assertEqual(parity["arms"]["K5"]["receipt_status_counts"], {"unavailable": 1})
            costs = json.loads((destination / "COST_ACCOUNTING.json").read_text())
            self.assertIsNone(costs["training"]["response_forwards"])

    def test_startup_is_subset_and_unfinished_work_remains_unknown(self):
        with tempfile.TemporaryDirectory() as folder:
            campaign, _, _, _ = fixture(Path(folder))
            with (campaign / "WORK.jsonl").open("a") as stream:
                stream.write(json.dumps(dict(kind="target_call", status="attempted", ticket=3,
                                             gpu_seconds=2, metadata={"operation": "native_target_verify"})) + "\n")
            result = work_accounting(campaign / "WORK.jsonl", [], [])
            self.assertEqual(result["startup_attempts_by_kind"], {"draft_block": 1})
            self.assertEqual(sum(r["attempted"] for r in result["by_operation"] if r["kind"] == "draft_block"), 2)
            self.assertEqual(sum(r["unknown"] for r in result["by_operation"]), 1)

    def test_history_categories_keep_terminal_and_converse_separate(self):
        base = dict(request_id="r0", canonical_start=0, valid_proposals=15, parity_valid=True,
                    a4=8, a5=15, a6=8, genuine_k4_rejection=True,
                    genuine_k5_rejection=False, genuine_k6_rejection=True)
        rows = starts([base, dict(base, a5=8, genuine_k5_rejection=True),
                       dict(base, a4=4, a5=4, a6=15, genuine_k5_rejection=True, genuine_k6_rejection=False),
                       dict(base, a4=None)])
        self.assertEqual([r["history"] for r in rows], ["H", "M", "all_correct_or_terminal", "missing_replay"])
        self.assertTrue(rows[2]["converse"])
        self.assertEqual(rows[2]["converse_prefix_gain"], 11)

    def test_destination_cannot_replace_original_report(self):
        with tempfile.TemporaryDirectory() as folder:
            campaign, sealed, source, _ = fixture(Path(folder))
            with self.assertRaisesRegex(ValueError, "separate output"):
                publish(campaign, sealed, source, output=campaign / "final_report")


if __name__ == "__main__":
    unittest.main()
