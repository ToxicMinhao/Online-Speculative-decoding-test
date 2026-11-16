"""Offline reporting never promotes partial work into an efficacy claim."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.persistent_prefix_withholding import report as R
from online_feedback.storage import write_json


class ReportTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.plan = dict(root=str(self.root), execution_commit="a"*40, binding="sealed",
                         missing_references=str(self.root/"missing.json"))
        write_json(self.plan["missing_references"], [dict(request_id=str(i)) for i in range(24)])
        self.path = self.root/"PLAN.json"
        write_json(self.path, self.plan)

    def complete(self):
        self.plan["qualification_receipts"] = {}
        for label in ("A","B","cross_card","serving","cpu_gate","reference_provenance","resource_admission"):
            path = self.root/(label+".json")
            write_json(path, dict(status="passed",execution_commit="a"*40))
            if label in ("A","B","cross_card","serving"):
                self.plan["qualification_receipts"][label] = str(path)
            else:
                self.plan[label] = str(path)
        for task in R.TASKS:
            write_json(self.root/"tasks"/task/"STATUS.json", dict(status="completed",wall_s=1.))
            write_json(self.root/"tasks"/task/"RESULT.json", dict(status="completed"))
        counts = R.planned_work(self.plan)
        write_json(self.root/"WORK.json",dict(attempted=counts,completed=counts,pending={}))
        write_json(self.root/"CAMPAIGN.json",dict(status="completed",execution_commit="a"*40))
        write_json(self.path,self.plan)

    def test_failed_qualification_preserves_unknown_and_does_not_analyze(self):
        counts = {name:0 for name in R.LIMITS}
        attempted = dict(counts,response_forward=1)
        write_json(self.root/"WORK.json",dict(attempted=attempted,completed=counts,pending={"0":{"kind":"response_forward"}}))
        write_json(self.root/"qualification/A/STATUS.json",dict(status="failed",error="fixed failure"))
        with patch.object(R.analysis,"write_report") as analyze:
            result = R.assemble_report(self.path)
        analyze.assert_not_called()
        self.assertIsNone(result["efficacy"])
        row = next(row for row in result["work"]["rows"] if row["kind"] == "response_forward")
        self.assertEqual((row["completed"],row["unknown"],row["unattempted"]),(0,1,1309))
        self.assertEqual(result["qualification"]["qualification"]["A"]["error"],"fixed failure")
        self.assertTrue((self.root/"TASK_COSTS.csv").exists())

    def test_absent_ledger_is_unavailable_not_zero(self):
        result = R.assemble_report(self.path)
        self.assertFalse(result["work"]["accounting_complete"])
        self.assertTrue(all(row["completed"] is None for row in result["work"]["rows"]))

    def test_preproduction_failure_uses_retained_native_ledger_without_inventing_success(self):
        from experiments.threatened_repair_replay.work import Ledger
        ledger=Ledger(self.root/"qualification/A/PHYSICAL_WORK.jsonl")
        with ledger.operation("model_load"):
            pass
        with self.assertRaisesRegex(RuntimeError,"native error"):
            with ledger.operation("response_forward"):
                raise RuntimeError("native error")
        ledger.close()
        result=R.assemble_report(self.path)
        self.assertEqual(result["work"]["source"],"retained qualification ledgers")
        row=next(row for row in result["work"]["rows"] if row["kind"]=="response_forward")
        self.assertEqual((row["completed"],row["failed"],row["unknown"],row["unattempted"]),(0,1,0,1309))
        self.assertIsNone(result["efficacy"])

    def test_complete_campaign_uses_existing_raw_audit_and_checks_execution_commit(self):
        self.complete()
        scientific = dict(execution_commit="a"*40,decision={"outcome":"NO_PRACTICAL_ADVANCEMENT_THIS_PILOT"},
                          comparisons={},paired_requests=["private rows omitted"])
        with patch.object(R.analysis,"write_report",return_value=scientific) as analyze:
            result = R.assemble_report(self.path)
        analyze.assert_called_once_with(self.root/"tasks/final/evaluation",native_trace_resolver=None)
        self.assertEqual(result["status"],"COMPLETED_VALID_COMPARISON")
        self.assertNotIn("paired_requests",result["efficacy"])
        scientific["execution_commit"] = "b"*40
        with patch.object(R.analysis,"write_report",return_value=scientific):
            result = R.assemble_report(self.path)
        self.assertIsNone(result["efficacy"])

    def test_unknown_work_blocks_scientific_reduction_even_when_nominally_complete(self):
        self.complete()
        counts=R.planned_work(self.plan)
        write_json(self.root/"WORK.json",dict(attempted=counts,completed=dict(counts,adam=647),pending={"4":{}}))
        with patch.object(R.analysis,"write_report") as analyze:
            result=R.assemble_report(self.path)
        analyze.assert_not_called()
        self.assertIsNone(result["efficacy"])


if __name__ == "__main__":
    unittest.main()
