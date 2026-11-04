"""Reporting refuses missing evidence and reconstructs shared work accounting."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from online_feedback.storage import write_json
from experiments.vfd_second_pass import report as R
from experiments.vfd_second_pass.contract import ARMS, PASS2, BUDGET


def pass_receipt(event, arm, number, start, order):
    return dict(
        event=event,
        pass_index=number,
        start_step=start,
        end_step=start + len(order) // 2,
        optimizer_steps=len(order) // 2,
        response_presentations=len(order),
        executed_order=order,
        records=[
            dict(record_id=identity, optimizer_stepped=bool(i % 2))
            for i, identity in enumerate(order)
        ],
        training_s=float(len(order)),
        pass_wall_s=float(len(order) + 1),
    )


def event_fixture(root):
    def ledger(folder, presentations, scores, result):
        limits = dict(
            training_presentations=presentations,
            backwards=presentations,
            optimizer_steps=presentations // 2,
            scoring_forwards=scores,
        )
        rows = [
            dict(kind=kind, ordinal=i, phase=phase)
            for kind, count in limits.items()
            for i in range(1, count + 1)
            for phase in ("started", "completed")
        ]
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "WORK.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
        result.update(
            status="completed",
            work=dict(
                attempts=limits, completed=limits, unknown={key: 0 for key in limits}
            ),
        )
        write_json(folder / "RESULT.json", result)

    previous = {arm: 0 for arm in ARMS}
    status = dict(
        status="completed",
        work={
            key: value
            for key, value in BUDGET.items()
            if not key.startswith("qualification_")
        },
        qualification_work=dict(
            scoring_forwards=2, training_presentations=4, optimizer_steps=2
        ),
        logical_updater_seconds={arm: 0.0 for arm in ARMS},
        costs={},
    )
    status["work"]["canonical_generations"] = 58
    for event in range(1, 5):
        order = [f"e{event}_{i}" for i in range(128)]
        second = list(reversed(order))
        shared = None
        for arm in ARMS:
            initial = dict(
                status="passed",
                exact_resume=True,
                global_step=64 if event == 1 else previous[arm],
            )
            first = pass_receipt(event, arm, 0, previous[arm], order)
            if event == 1:
                if shared is None:
                    shared = first
                    write_json(root / "events/event_001/SHARED/pass_0.json", shared)
                    ledger(
                        root / "events/event_001/SHARED", 128, 256, dict(pass1=first)
                    )
                first = deepcopy(shared)
            selected = second[: PASS2[arm]]
            last = (
                pass_receipt(event, arm, 1, first["end_step"], selected)
                if selected
                else None
            )
            previous[arm] = (last or first)["end_step"]
            result = dict(
                initial_state=initial,
                pass1=first,
                pass2=last,
                selection=dict(selected_ids=selected, training_order=selected),
                logical_updater_s=first["pass_wall_s"]
                + (last["pass_wall_s"] if last else 0.0),
                costs=dict(
                    training_s=first["pass_wall_s"]
                    + (last["pass_wall_s"] if last else 0.0),
                    training_compute_s=float(128 + PASS2[arm]),
                    scoring_s=0.0,
                    selection_s=0.0,
                    scoring_before_s=0.0,
                    scoring_after_s=0.0,
                    scoring_reduction_s=0.0,
                ),
            )
            status["logical_updater_seconds"][arm] += result["logical_updater_s"]
            result["state"] = dict(
                status="passed",
                global_step=previous[arm],
                presentations=2 * previous[arm],
            )
            ledger(
                root / f"events/event_{event:03d}" / arm,
                PASS2[arm] if event == 1 else 128 + PASS2[arm],
                0 if event == 1 else {"STATIC64": 128, "PROGRESS64": 256}.get(arm, 0),
                result,
            )
    return status


class ReportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        patched = patch.object(R, "score_evidence", return_value=[])
        patched.start()
        self.addCleanup(patched.stop)

    def test_shared_work_reconciles_without_counting_forked_pass_four_more_times(self):
        status = event_fixture(self.root)
        result = R.audit_training(self.root, {}, status)
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["physical_optimizer_steps"], 1664)
        self.assertEqual(result["physical_training_presentations"], 3328)
        self.assertEqual(result["logical_optimizer_steps"], 1920)
        self.assertEqual(result["logical_training_presentations"], 3840)
        self.assertEqual(len(result["selections"]), 20)
        self.assertEqual(
            result["event_selection_overlap"][0]["static_progress_selection_overlap"],
            64,
        )

    def test_missing_or_changed_pass_never_admits_incomplete_training(self):
        status = event_fixture(self.root)
        path = self.root / "events/event_002/RANDOM64/RESULT.json"
        result = json.loads(path.read_text())
        result["pass2"]["optimizer_steps"] = 33
        write_json(path, result)
        audit = R.audit_training(self.root, {}, status)
        self.assertNotEqual(audit["errors"], [])

    def test_native_subset_order_must_follow_full_permutation(self):
        status = event_fixture(self.root)
        path = self.root / "events/event_003/PROGRESS64/RESULT.json"
        result = json.loads(path.read_text())
        result["selection"]["training_order"].reverse()
        write_json(path, result)
        self.assertNotEqual(R.audit_training(self.root, {}, status)["status"], "passed")

    def test_budget_and_qualification_are_separate(self):
        status = event_fixture(self.root)
        self.assertEqual(R.budget_errors(status, 58), [])
        changed = deepcopy(status)
        changed["work"]["optimizer_steps"] += 2
        self.assertIn("work_optimizer_steps", R.budget_errors(changed, 58))
        changed = deepcopy(status)
        changed["qualification_work"]["scoring_forwards"] = 9
        self.assertIn("qualification_scoring_forwards", R.budget_errors(changed, 58))
        changed = deepcopy(status)
        changed["work"]["canonical_generations"] = 59
        self.assertIn("work_canonical_generations", R.budget_errors(changed, 58))

    def test_complete_report_writes_all_tables_without_prompt_or_token_columns(self):
        from test_vfd_second_pass_endpoint import measures

        rows = measures()
        for row in rows:
            row.update(
                block_ordinal=row["panel_ordinal"] // 20,
                position=ARMS.index(row["method"]),
            )
        requests = [
            dict(
                panel_ordinal=i,
                request_id=f"req:{i}",
                question="PRIVATE_PROMPT_SENTINEL",
                prompt_ids=[123456],
            )
            for i in range(200)
        ]
        status = event_fixture(self.root / "campaign_001")
        training = R.audit_training(self.root / "campaign_001", {}, status)
        write_json(
            self.root / "sealed/PLAN.json",
            dict(
                evaluation_requests=requests,
                final_checkpoints={a: "/" + a for a in ARMS},
            ),
        )
        write_json(self.root / "sealed/REFERENCE_MISSING.json", dict(count=58))
        write_json(
            self.root / "sealed/REFERENCE_REUSE_AUDIT.json", dict(binding_digest="test")
        )
        write_json(self.root / "campaign_001/canonicals/CANONICALS.json", {})
        write_json(self.root / "campaign_001/STATUS.json", status)
        with patch.object(
            R, "audit_qualification", return_value={"status": "passed"}
        ), patch.object(R, "audit_training", return_value=training), patch.object(
            R, "audit_canonicals", return_value={"status": "passed"}
        ), patch.object(
            R,
            "audit_stage",
            return_value=dict(status="passed", errors=[], rows=rows, progress={}),
        ):
            result = R.report(self.root, self.root / "report")
        self.assertEqual(result["status"], "passed", result)
        summary = json.loads((self.root / "report/SUMMARY.json").read_text())
        self.assertEqual(summary["execution_count"], 2000)
        for name in (
            "ALL_REPEATS.csv",
            "PAIRED_REQUESTS.csv",
            "SCORES.csv",
            "SCORE_BLOCKS.csv",
            "SELECTIONS.csv",
            "TRAINING_PASSES.csv",
        ):
            self.assertTrue((self.root / "report" / name).exists())
        text = "".join(p.read_text() for p in (self.root / "report").iterdir())
        self.assertNotIn("PRIVATE_PROMPT_SENTINEL", text)
        self.assertNotIn("prompt_ids", text)

    def test_missing_endpoint_writes_failure_receipt_without_statistics(self):
        status = event_fixture(self.root / "campaign_001")
        write_json(
            self.root / "sealed/PLAN.json",
            dict(evaluation_requests=[], final_checkpoints={}),
        )
        write_json(self.root / "sealed/REFERENCE_MISSING.json", dict(count=58))
        write_json(self.root / "campaign_001/STATUS.json", status)
        with patch.object(R, "summarize", side_effect=AssertionError("must never run")):
            result = R.report(self.root, self.root / "report")
        self.assertEqual(result["status"], "invalid_or_incomplete")
        self.assertTrue((self.root / "report/REPORT_STATUS.json").is_file())
        self.assertFalse((self.root / "report/SUMMARY.json").exists())
        self.assertIn(
            "No efficacy estimate", (self.root / "report/REPORT.md").read_text()
        )


if __name__ == "__main__":
    unittest.main()
