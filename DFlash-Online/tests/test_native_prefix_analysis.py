"""Denominator, validity and decision checks for the fixed role-loss screen."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.native_prefix_loss import analysis as a


def fixture(windows=(8, 16), requests=3, starts=2):
    sealed = [{"request_id": f"r{r}", "canonical_start": s * 16,
               "round_ordinal": s, "valid_proposals": 15}
              for r in range(requests) for s in range(starts)]
    rows = []
    for window in windows:
        for probe in sealed:
            counts = {"P": 3, "N": 5, "E": 6, "C": 2, "H": 4}
            for state, count in counts.items():
                rows.append({**probe, "window": window, "state": state,
                             "accepted_prefix": count, "first_error": count + 1, "valid": True})
    return sealed, rows


def source_fixture(windows=(8, 16)):
    rows = []
    for window in windows:
        for rid in ("fit-a", "fit-b"):
            for anchor in range(2):
                for state, count in {"P": 3, "N": 5, "E": 6, "C": 2, "H": 4}.items():
                    rows.append({"window": window, "request_id": rid,
                                 "canonical_start": anchor * 16, "anchor_ordinal": anchor,
                                 "state": state, "accepted_prefix": count, "keep": True,
                                 "valid_proposals": 15, "first_error": count + 1, "valid": True})
    return rows


class ReductionTests(unittest.TestCase):
    def reduce(self, rows=None, sealed=None, **kwargs):
        if rows is None:
            sealed, rows = fixture()
        return a.reduce_probes(rows, sealed, windows=(8, 16), require_full_population=False, **kwargs)

    def test_exact_grid_and_primary_denominator(self):
        summary, requests, windows, blocks = self.reduce()
        self.assertEqual(summary["primary_contrast"], "E_N")
        self.assertEqual(summary["paired_block_windows"], 12)
        self.assertEqual(summary["conceptual_observations"], 60)
        self.assertEqual(summary["contrasts"]["E_N"]["exact"], {"numerator": 1, "denominator": 1})
        self.assertEqual(summary["contrasts"]["E_N"]["lower"], 1)
        self.assertEqual(len(requests), 3)
        self.assertEqual(len(windows), 2)
        self.assertEqual(len(blocks), 12)

    def test_equal_requests_not_pooled_starts(self):
        sealed, rows = fixture(requests=2, starts=1)
        extra = {"request_id": "r1", "canonical_start": 16, "round_ordinal": 1, "valid_proposals": 15}
        sealed.append(extra)
        for w in (8, 16):
            for state in a.STATES:
                count = 15 if state == "E" else 5
                rows.append({**extra, "window": w, "state": state,
                             "accepted_prefix": count, "first_error": None if count == 15 else count + 1,
                             "valid": True})
        summary, request_rows, _, _ = self.reduce(rows, sealed)
        self.assertEqual(request_rows[0]["E_N"], 1)
        self.assertEqual(request_rows[1]["E_N"], 5.5)
        self.assertEqual(summary["contrasts"]["E_N"]["estimate"], 3.25)
        self.assertNotEqual(summary["contrasts"]["E_N"]["estimate"], 4)

    def test_whole_request_bootstrap_independent_reference(self):
        rows = [{"request_id": f"r{i}", **{f"{x}_{y}": value for x, y in a.CONTRASTS}}
                for i, value in enumerate((-1, .5, 2))]
        observed = a.bootstrap_requests(rows)
        rng = np.random.Generator(np.random.PCG64(1707))
        draws = rng.integers(0, 3, size=(10000, 3))
        samples = np.array([-1, .5, 2])[draws].mean(axis=1)
        self.assertEqual(observed["intervals"]["E_N"]["lower"], np.quantile(samples, .025, method="linear"))
        self.assertEqual(observed["intervals"]["E_N"]["upper"], np.quantile(samples, .975, method="linear"))
        self.assertEqual(observed["draws"], 10000)

    def test_h_is_a_real_update_not_aliased_to_p(self):
        summary, _, _, blocks = self.reduce()
        self.assertEqual(summary["contrasts"]["H_P"]["estimate"], 1)
        self.assertEqual(blocks[0]["H_net"], 1)
        self.assertEqual(blocks[0]["H_acquired"], 1)

    def test_trades_and_blocker_counts_have_separate_units(self):
        row = a.tradeoff({"P": 5, "N": 8, "E": 4, "C": 6, "H": 5}, 15)
        self.assertEqual(row["N_acquired"], 3)
        self.assertEqual(row["N_repairs_pre_blocker"], 1)
        self.assertEqual(row["E_lost"], 1)
        self.assertEqual(row["E_earlier_error"], 1)
        self.assertEqual(row["E_displaced_native_blocker_repair"], 1)
        self.assertEqual(row["E_N_net"], -4)
        self.assertEqual(row["E_N_acquisition_difference"] - row["E_N_loss_difference"], -4)

    def test_full_censored_boundary_has_no_blocker(self):
        row = a.tradeoff({state: 4 for state in a.STATES}, 4)
        self.assertEqual(row["pre_has_blocker"], 0)
        self.assertTrue(all(row[f"{state}_repairs_pre_blocker"] == 0 for state in a.CANDIDATES))

    def test_zero_effect_windows_are_not_omitted(self):
        sealed, rows = fixture()
        for row in rows:
            if row["window"] == 16:
                row.update(accepted_prefix=3, first_error=4)
        summary, _, windows, _ = self.reduce(rows, sealed)
        self.assertEqual(summary["contrasts"]["E_N"]["estimate"], .5)
        self.assertEqual(windows[1]["E_N"], 0)

    def test_all_production_population_and_seed_guards(self):
        sealed, rows = fixture()
        with self.assertRaisesRegex(ValueError, "Production population"):
            a.reduce_probes(rows, sealed)
        sealed, rows = fixture(windows=a.WINDOWS, requests=64, starts=4)
        with self.assertRaisesRegex(ValueError, "bootstrap"):
            a.reduce_probes(rows, sealed, seed=1)

    def test_invalid_missing_duplicate_and_wrong_boundaries_fail(self):
        sealed, rows = fixture()
        variants = [rows[:-1], rows + [rows[0]]]
        for field, value in (("valid", False), ("first_error", 13), ("accepted_prefix", True), ("valid_proposals", 14)):
            changed = copy.deepcopy(rows)
            changed[0][field] = value
            variants.append(changed)
        for variant in variants:
            with self.subTest(last=variant[0]):
                with self.assertRaises(ValueError):
                    self.reduce(variant, sealed)

    def test_reused_state_must_match(self):
        sealed, rows = fixture()
        rows[2]["reused_from"] = "N"
        with self.assertRaisesRegex(ValueError, "Aliased"):
            self.reduce(rows, sealed)

    def test_source_fitting_kept_separate(self):
        result, blocks = a.reduce_source(source_fixture(), windows=(8, 16))
        self.assertEqual(result["paired_block_count"], 8)
        self.assertEqual(result["gross_counts"]["E_N_net"], 8)
        self.assertEqual(result["response_window_count"], 4)
        self.assertNotIn("bootstrap", result)
        self.assertEqual(len(blocks), 8)

    def test_source_duplicate_anchor_starts_preserve_occurrences(self):
        rows = source_fixture()
        for row in rows:
            row["canonical_start"] = 0
        summary, blocks = a.reduce_source(rows, windows=(8, 16))
        self.assertEqual(summary["paired_block_count"], 8)
        self.assertEqual(len(blocks), 8)

    def test_source_requires_pair_and_all_states(self):
        with self.assertRaises(ValueError):
            a.reduce_source(source_fixture()[:-1], windows=(8, 16))
        with self.assertRaises(ValueError):
            a.reduce_source([row for row in source_fixture() if row["request_id"] == "fit-a"], windows=(8, 16))

    def test_source_padding_and_zero_supervision_denominators(self):
        rows = source_fixture()
        # Of eight paired anchors: two padded, two kept-but-empty, four eligible.
        for row in rows:
            if row["anchor_ordinal"] == 0:
                row.update(accepted_prefix=0, valid_proposals=0, first_error=None,
                           keep=row["request_id"] == "fit-a")
        result, blocks = a.reduce_source(rows, windows=(8, 16))
        counts = result["denominators"]
        self.assertEqual(counts["sampled_anchor_slots"], 8)
        self.assertEqual(counts["kept_anchor_occurrences"], 6)
        self.assertEqual(counts["dropped_or_padded_anchor_slots"], 2)
        self.assertEqual(counts["eligible_blocks"], 4)
        self.assertEqual(counts["zero_supervision_anchor_slots"], 4)
        self.assertEqual(counts["kept_but_zero_supervision_anchor_occurrences"], 2)
        self.assertEqual(counts["scored_proposal_occurrences"], 60)
        self.assertEqual(result["mean_levels_all_sampled_slots"]["N"], 2.5)
        self.assertEqual(result["mean_levels_eligible_blocks"]["N"], 5)
        self.assertEqual(len(blocks), 8)

    def test_source_entirely_empty_kept_visible_without_nan(self):
        rows = source_fixture()
        for row in rows:
            row.update(keep=False, accepted_prefix=0, valid_proposals=0, first_error=None)
        result, blocks = a.reduce_source(rows, windows=(8, 16))
        self.assertEqual(result["denominators"]["eligible_blocks"], 0)
        self.assertEqual(result["denominators"]["zero_supervision_anchor_slots"], 8)
        self.assertTrue(all(value is None for value in result["mean_levels_eligible_blocks"].values()))
        self.assertEqual(result["gross_counts"]["E_N_net"], 0)
        json.dumps(result, allow_nan=False)

    def test_source_keep_flag_must_match_states_and_validity(self):
        rows = source_fixture()
        rows[0]["keep"] = False
        with self.assertRaisesRegex(ValueError, "keep flag"):
            a.reduce_source(rows, windows=(8, 16))
        for row in rows:
            row["keep"] = False
        with self.assertRaisesRegex(ValueError, "cannot contribute"):
            a.reduce_source(rows, windows=(8, 16))
        for row in rows:
            row.pop("keep")
        with self.assertRaisesRegex(ValueError, "keep flag"):
            a.reduce_source(rows, windows=(8, 16))

    def test_decision_retains_primary_and_no_equivalence_claim(self):
        valid = {"complete": True, "all_required_checks_passed": True}
        for lower, upper, mean, decision in ((.1, .3, .2, "LOCAL_PREFIX_WITHHOLDING_SCREEN_PASSED"),
                                           (-.3, -.1, -.2, "NATIVE_TRANSFERS_BETTER"),
                                           (-.1, .1, 0, "UNRESOLVED_STOP_THIS_SCREEN")):
            self.assertEqual(a.decide({"contrasts": {"E_N": {"lower": lower, "upper": upper, "estimate": mean}}}, valid), decision)
        self.assertEqual(a.decide({}, {"complete": False}), "INCOMPLETE")
        self.assertEqual(a.decide({}, {"complete": True}), "INVALID")

    def test_report_preserves_invalid_records_and_missing_accounting(self):
        sealed, rows = fixture()
        rows = rows[:-2]
        rows[0]["valid"] = False
        with tempfile.TemporaryDirectory() as directory:
            result = a.write_report(directory, probe_rows=rows, sealed_probes=sealed,
                validity={"complete": False}, work={"adam_steps": 1}, execution_commit="test",
                windows=(8, 16), require_full_population=False)
            self.assertEqual(result["grid"]["unattempted"], 2)
            self.assertEqual(result["grid"]["invalid"], 1)
            self.assertNotIn("primary", result)
            self.assertEqual(len((Path(directory) / "PROBE_OUTCOMES.csv").read_text().splitlines()), len(rows) + 1)
            self.assertEqual(json.loads((Path(directory) / "SUMMARY.json").read_text())["work"]["adam_steps"], 1)

    def test_claimed_valid_report_rechecks_rows(self):
        sealed, rows = fixture()
        rows[0]["accepted_prefix"] = 50
        with tempfile.TemporaryDirectory() as directory:
            result = a.write_report(directory, probe_rows=rows, sealed_probes=sealed,
                source_rows=source_fixture(), validity={"complete": True, "all_required_checks_passed": True},
                work={}, execution_commit="test", windows=(8, 16), require_full_population=False)
            self.assertEqual(result["decision"], "INVALID")
            self.assertNotIn("primary", result)

    def test_unknown_and_unattempted_remain_distinct(self):
        sealed, rows = fixture()
        rows[0].update(status="unknown", valid=False)
        rows[1].update(status="unattempted", valid=False)
        result = a.grid_accounting(rows[:-1], sealed, windows=(8, 16))
        self.assertEqual(result["unknown"], 1)
        self.assertEqual(result["unattempted"], 2)
        self.assertEqual(result["valid"], 57)

    def test_completed_report_requires_source_and_separates_h(self):
        sealed, rows = fixture()
        with tempfile.TemporaryDirectory() as directory:
            result = a.write_report(directory, probe_rows=rows, sealed_probes=sealed,
                source_rows=source_fixture(), validity={"complete": True, "all_required_checks_passed": True},
                work={}, execution_commit="test", windows=(8, 16), require_full_population=False)
            self.assertEqual(result["decision"], "LOCAL_PREFIX_WITHHOLDING_SCREEN_PASSED")
            self.assertIn("persistent moments", (Path(directory) / "REPORT.md").read_text())
            self.assertTrue((Path(directory) / "SOURCE_TRADEOFF.csv").exists())


if __name__ == "__main__":
    unittest.main()
