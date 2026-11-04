"""Orchestration failures must preserve counted work without model execution."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from online_feedback.storage import write_json
from experiments.ok_coding.campaign import WorkLedger
from experiments.vfd_second_pass.campaign import accounting, execute_worker
from experiments.vfd_second_pass.contract import ARMS, physical_training
from experiments.vfd_second_pass.storage import capacity_plan, admit


class CampaignAccountingTests(unittest.TestCase):
    def test_physical_and_logical_budgets(self):
        self.assertEqual(
            physical_training(1), dict(optimizer_steps=224, training_presentations=448)
        )
        self.assertEqual(
            physical_training(4),
            dict(optimizer_steps=1664, training_presentations=3328),
        )
        self.assertEqual(
            capacity_plan(2097259104)["required_initial_free_bytes"], 104597327235
        )

    def test_killed_worker_keeps_unfinished_action_without_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "campaign_001/events/event_001/SHARED"
            folder.mkdir(parents=True)
            write_json(
                root / "campaign_001/worker_logs/event_001_SHARED.json",
                dict(
                    command=[
                        "python",
                        "-m",
                        "experiments.vfd_second_pass.worker",
                        str(root),
                        "1",
                        "SHARED",
                    ],
                    pid=123,
                    status="failed",
                    exit_code=-9,
                ),
            )
            limits = dict(
                training_presentations=128,
                backwards=128,
                optimizer_steps=64,
                scoring_forwards=256,
            )
            ledger = WorkLedger(folder / "WORK.jsonl", limits)
            ledger.call("scoring_forwards", lambda: None)

            def fail():
                raise RuntimeError("interrupted native action")

            with self.assertRaises(RuntimeError):
                ledger.call("training_presentations", fail)
            got = accounting(root, dict(final_checkpoints={arm: arm for arm in ARMS}))
            self.assertEqual(got["work"]["scoring_forwards"], 1)
            self.assertEqual(got["work"]["training_presentations"], 0)
            job = got["physical_work_detail"]["jobs"][0]
            self.assertFalse(job["result_available"])
            self.assertEqual(job["work"]["attempts"]["training_presentations"], 1)
            self.assertEqual(job["work"]["unknown"]["training_presentations"], 1)

    def test_failed_worker_is_not_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "campaign_001").mkdir()
            plan = dict(training_python="/bin/false", training_gpu="1")
            with self.assertRaises(RuntimeError):
                execute_worker(root, plan, 1, "FULL")
            receipt = json.loads(
                (root / "campaign_001/worker_logs/event_001_FULL.json").read_text()
            )
            self.assertEqual(receipt["exit_code"], 1)
            self.assertFalse(receipt["automatic_retry"])
            with self.assertRaises(FileExistsError):
                execute_worker(root, plan, 1, "FULL")

    def test_launch_receipt_failure_stops_owned_child(self):
        from unittest.mock import Mock

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "campaign_001").mkdir()
            process = Mock(pid=12345, returncode=None)
            process.poll.return_value = None
            process.wait.return_value = -15

            def save(path, receipt):
                if receipt["status"] == "running":
                    raise OSError("receipt persistence failed")

            with patch(
                "experiments.vfd_second_pass.campaign.subprocess.Popen",
                return_value=process,
            ), patch(
                "experiments.vfd_second_pass.campaign.write_json", side_effect=save
            ), patch(
                "experiments.vfd_second_pass.campaign.os.killpg"
            ) as kill:
                with self.assertRaises(OSError):
                    execute_worker(
                        root,
                        dict(training_python="python", training_gpu="1"),
                        1,
                        "FULL",
                    )
                kill.assert_called_once()
                process.wait.assert_called_once_with(timeout=45)

    def test_missing_compiler_baseline_after_new_layout_work_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "campaign_001/events/event_001/SHARED/WORK.jsonl"
            path.parent.mkdir(parents=True)
            path.write_text("")
            with self.assertRaisesRegex(ValueError, "Compiler baseline"):
                admit(root, capacity_plan(2097259104))

    def test_reserve_cannot_be_lowered_or_future_offload_credited(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = capacity_plan(2097259104)
            usage = type(
                "Usage", (), dict(free=plan["required_initial_free_bytes"] - 1_000_000)
            )()
            with patch(
                "experiments.vfd_second_pass.storage.shutil.disk_usage",
                return_value=usage,
            ), patch(
                "experiments.vfd_second_pass.storage.compiler_growth",
                return_value=dict(compiler_growth_bytes=0),
            ):
                with self.assertRaises(ValueError):
                    admit(Path(tmp), plan)
            self.assertFalse(plan["offload_assumed"])


if __name__ == "__main__":
    unittest.main()
