"""Fixed stratified endpoint arithmetic, uncertainty and incomplete reports."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from online_feedback.storage import write_json
from experiments.k_workload_replay_v1 import statistics as S
from experiments.k_workload_replay_v1.contract import BUDGET
from experiments.k_workload_replay_v1.report import _evidence_index, _work_accounting, build_report
from experiments.k_workload_replay_v1.selection import ReservoirMemory, build_event_plan
from test_k_workload_replay_evaluation import panel
from test_k_workload_replay_selection import event_records


def rows():
    result = []
    for ordinal in range(256):
        baseline = 10 + ordinal % 11
        calls = {"K": baseline, "U": baseline - 1, "R": baseline - (2 if ordinal % 2 == 0 else 0)}
        for method in S.ARMS:
            for repeat in range(3):
                result.append(dict(seed=17, event=8, phase="final", capture=False, panel_ordinal=ordinal,
                    request_id=f"eval{ordinal}", domain="math" if ordinal % 2 == 0 else "coding", repeat=repeat,
                    method=method, verification_calls=calls[method], accepted_draft_tokens=20,
                    committed_progress=39, output_tokens=40, checkpoint_id="/" + method,
                    finish_reason="eos", cap_hit=False, validation_status="valid", valid_measurement=True, errors=[],
                    request_latency_s=float(calls[method] + repeat), native_generate_s=calls[method] - .1 + repeat,
                    cache_maintenance_s=0.0))
    return result


class ReplayStatisticsTests(unittest.TestCase):
    def test_count_once_time_means_gross_harm_and_domain_tradeoff(self):
        with patch.object(S, "RESAMPLES", 50):
            summary = S.summarize(rows())
        baseline = sum(10 + i % 11 for i in range(256))
        self.assertEqual(summary["method_totals"]["K"]["verification_calls"], baseline)
        self.assertEqual(summary["method_totals"]["K"]["request_latency_s"], baseline + 256)
        effect = summary["comparisons"]["R_vs_U"]
        self.assertAlmostEqual(effect["call_reduction"]["point_pct"], 0)
        self.assertEqual((effect["request_wins"], effect["request_regressions"], effect["request_ties"]), (128, 128, 0))
        self.assertEqual((effect["gross_calls_saved"], effect["gross_calls_added"]), (128, 128))
        self.assertGreater(summary["domains"]["math"]["comparisons"]["R_vs_U"]["call_reduction"]["point_pct"], 0)
        self.assertLess(summary["domains"]["coding"]["comparisons"]["R_vs_U"]["call_reduction"]["point_pct"], 0)
        self.assertEqual(effect["cache_maintenance_s_reduction"]["point_pct"], None)
        self.assertEqual(effect["cache_maintenance_s_reduction"]["undefined_bootstrap_draws"], 50)
        self.assertEqual(summary["repeat_totals"]["2"]["K"]["verification_calls"], baseline)

    def test_exact_stratified_pcg64_10000_linear_shared_draws_and_rng_isolation(self):
        before = np.random.get_state()
        summary = S.summarize(rows())
        rng = np.random.Generator(np.random.PCG64(20261102))
        sampled = []
        for _ in range(10000):
            math_indices = 2 * rng.integers(0, 128, 128)
            coding_indices = 2 * rng.integers(0, 128, 128) + 1
            sampled.append(np.concatenate([math_indices, coding_indices]))
        sampled = np.array(sampled)
        original = np.array([10 + i % 11 for i in range(256)])
        candidate = original - np.array([2 if i % 2 == 0 else 0 for i in range(256)])
        expected = 100 * (1 - candidate[sampled].sum(axis=1) / original[sampled].sum(axis=1))
        interval = np.quantile(expected, [.025, .975], method="linear").tolist()
        self.assertEqual(summary["comparisons"]["R_vs_K"]["call_reduction"]["ci95_pct"], interval)
        self.assertEqual(summary["bootstrap"]["strata"], {"math": 128, "coding": 128})
        after = np.random.get_state()
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

    def test_repeated_identity_count_domain_timing_and_validity_fail_closed(self):
        mutations = [lambda r: r[0].update(valid_measurement=False),
                     lambda r: r[0].update(verification_calls=r[0]["verification_calls"] + 1),
                     lambda r: r[0].update(domain="coding"), lambda r: r[0].update(request_latency_s=float("nan")),
                     lambda r: r.pop(), lambda r: r.append(deepcopy(r[0])),
                     lambda r: r[0].update(capture=True)]
        for mutation in mutations:
            records = rows()
            mutation(records)
            with self.assertRaises(ValueError):
                S.reduce_final(records)

    def test_future_event_summary_has_no_intervals_or_capture_time_claim(self):
        event_rows = []
        for event in range(1, 9):
            for ordinal in range(128):
                for method in S.ARMS:
                    event_rows.append(dict(event=event, request_id=f"e{event}_{ordinal}", method=method,
                        domain="math" if event in (1, 2, 5, 6) else "coding", verification_calls=10 if method != "R" else 9,
                        accepted_draft_tokens=20, committed_progress=39, output_tokens=40,
                        valid_measurement=True, capture=method == "K"))
        summary = S.summarize_events(event_rows)
        self.assertIsNone(summary["intervals"])
        self.assertEqual(summary["future_events_3_to_8"]["expected_requests"], 768)
        self.assertTrue(summary["future_events_3_to_8"]["valid_complete"])
        self.assertAlmostEqual(summary["future_events_3_to_8"]["comparisons"]["R_vs_K"]["call_reduction_pct"], 10)

    def test_incomplete_report_preserves_source_and_every_unattempted_slot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sealed").mkdir()
            (root / "campaign_001").mkdir()
            requests = panel()
            canonical = {r["request_id"]: dict(response_ids=[10, 151645], finish_reason="eos", source_record_path="/original")
                         for r in requests}
            write_json(root / "sealed/REQUESTS.json", {"evaluation": requests})
            write_json(root / "sealed/CANONICALS.json", canonical)
            write_json(root / "sealed/PLAN.json", dict(execution_source={"commit": "a" * 40, "working_tree_status": ""},
                budget=BUDGET, requests={"path": "sealed/REQUESTS.json"}, canonicals={"path": "sealed/CANONICALS.json"}))
            write_json(root / "campaign_001/STATUS.json", dict(status="failed", costs={"observed_wall_s": 1.25}))
            summary = build_report(root)
            self.assertEqual(summary["status"], "invalid_or_incomplete")
            self.assertIsNone(summary["endpoint"])
            self.assertEqual(summary["campaign_status"]["costs"], {"observed_wall_s": 1.25})
            text = (root / "final_report/REPORT.md").read_text()
            self.assertIn("a" * 40, text)
            self.assertEqual(len((root / "final_report/ALL_FINAL_ROWS.csv").read_text().splitlines()), 2305)
            self.assertFalse(summary["recommendation"]["automatic_followup_authorized"])
            self.assertEqual(summary["material_passport"]["verificationStatus"], "raw-evidence-analyzed")
            self.assertIn("not an independent GPU reproduction", text)

    def test_evidence_index_joins_domains_after_selection_and_preserves_retirement_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign = Path(directory)
            folder = campaign / "events/event_002"
            folder.mkdir(parents=True)
            memory = ReservoirMemory()
            memory.insert_after_event(1, event_records(1), verified_arms=S.ARMS)
            write_json(folder / "MEMORY_BEFORE.json", memory.state_dict())
            plan = build_event_plan(2, event_records(2), memory.records)
            write_json(folder / "SELECTION.json", plan)
            memory.insert_after_event(2, event_records(2), verified_arms=S.ARMS)
            write_json(folder / "MEMORY_AFTER.json", memory.state_dict())
            write_json(campaign / "RETIREMENTS/one.json", dict(status="retired", path="/run/old.pt",
                                                               archived=False, retained_evidence=["/run/receipt.json"]))
            requests = {"training": [dict(request_id=f"original_{i:04d}", domain="math" if i < 128 else "coding")
                                     for i in range(256)]}
            index = _evidence_index(campaign, requests)
            self.assertEqual(len(index["selection_steps"]), 64)
            self.assertEqual({r["reporting_domain"] for r in index["selection_steps"]}, {"math"})
            before = next(r for r in index["membership_and_selection"] if r["kind"] == "MEMORY_BEFORE.json")
            self.assertEqual(before["reporting_domain_counts"], {"math": 128})
            self.assertFalse(index["retirement_receipts"][0]["receipt"]["archived"])
            self.assertEqual(json.loads((folder / "SELECTION.json").read_text()), plan)

    def test_work_ledger_audit_preserves_unknown_partial_operations(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign = Path(directory)
            rows = [dict(kind="pre_update_generations", phase="started", ordinal=1),
                    dict(kind="pre_update_generations", phase="completed", ordinal=1),
                    dict(kind="terminal_preparation_forwards", phase="started", ordinal=1)]
            (campaign / "WORK.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            worker = campaign / "arms/K/events/event_001/WORK.jsonl"
            worker.parent.mkdir(parents=True)
            worker.write_text(json.dumps(dict(kind="response_forwards", phase="started", ordinal=1)) + "\n")
            result = _work_accounting(campaign)
            self.assertEqual(result["serving"]["observed"]["completed"]["pre_update_generations"], 1)
            self.assertEqual(result["serving"]["observed"]["unknown"]["terminal_preparation_forwards"], 1)
            self.assertEqual(result["observed_training_totals"]["unknown"]["response_forwards"], 1)
            self.assertEqual(result["expected_training_totals"]["optimizer_steps"], 3072)
            self.assertEqual(len(result["training_workers"]), 24)
            self.assertTrue(result["errors"])


if __name__ == "__main__":
    unittest.main()
