"""Independent CPU review of interrupted-job accounting at the campaign boundary."""

import json
from pathlib import Path
import tempfile
import unittest
from online_feedback.storage import write_json
from experiments.vfd_second_pass.campaign import accounting
from experiments.vfd_second_pass.contract import ARMS


class InterruptedAccountingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.plan = dict(final_checkpoints={arm: "/" + arm for arm in ARMS})

    def test_killed_worker_without_result_keeps_started_unknown_actions(self):
        write_json(
            self.root / "campaign_001/worker_logs/event_001_FULL.json",
            dict(
                command=["python", "worker", "root", "1", "FULL"],
                status="failed",
                exit_code=-9,
            ),
        )
        folder = self.root / "campaign_001/events/event_001/FULL"
        folder.mkdir(parents=True)
        (folder / "WORK.jsonl").write_text(
            "\n".join(
                json.dumps(row)
                for row in [
                    dict(kind="training_presentations", phase="started", ordinal=1),
                    dict(kind="training_presentations", phase="completed", ordinal=1),
                    dict(kind="optimizer_steps", phase="started", ordinal=1),
                ]
            )
            + "\n"
        )
        status = accounting(self.root, self.plan)
        self.assertEqual(status["work"]["training_presentations"], 1)
        self.assertEqual(status["work"]["optimizer_steps"], 0)
        job = status["physical_work_detail"]["jobs"][0]
        self.assertFalse(job["result_available"])
        self.assertEqual(job["work"]["unknown"]["optimizer_steps"], 1)
        self.assertEqual(status["logical_updater_seconds"]["FULL"], 0.0)

    def test_qualification_unknown_does_not_enter_main_science_counts(self):
        folder = self.root / "campaign_001/qualification"
        folder.mkdir(parents=True)
        (folder / "WORK.jsonl").write_text(
            json.dumps(dict(kind="scoring_forwards", phase="started", ordinal=1)) + "\n"
        )
        status = accounting(self.root, self.plan)
        self.assertEqual(status["work"]["scoring_forwards"], 0)
        self.assertEqual(status["qualification_work"]["scoring_forwards"], 0)
        self.assertEqual(
            status["physical_work_detail"]["qualification"]["unknown"][
                "scoring_forwards"
            ],
            1,
        )

    def test_result_cannot_disagree_with_original_work(self):
        write_json(
            self.root / "campaign_001/worker_logs/event_001_FULL.json",
            dict(
                command=["python", "worker", "root", "1", "FULL"],
                status="completed",
                exit_code=0,
            ),
        )
        write_json(
            self.root / "campaign_001/events/event_001/FULL/RESULT.json", dict(work={})
        )
        with self.assertRaisesRegex(ValueError, "original work ledger"):
            accounting(self.root, self.plan)


if __name__ == "__main__":
    unittest.main()
