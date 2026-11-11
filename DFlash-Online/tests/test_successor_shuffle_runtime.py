import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.successor_shuffle import runtime


class ControllerBudgetTests(unittest.TestCase):
    def fixture(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        (root / "sealed").mkdir()
        plan = dict(
            root=str(root),
            execution_source={"commit": "1" * 40},
            checkpoints={
                "TRUE": {
                    "checkpoint_path": "true",
                    "weights_identity": {"sha256": "true", "bytes": 1},
                }
            },
        )
        (root / "sealed/PLAN.json").write_text(json.dumps(plan))
        return root, plan

    def test_failed_qualification_preserved_and_never_trains(self):
        root, plan = self.fixture()
        with patch.object(
            runtime, "preflight", return_value={"status": "passed"}
        ), patch.object(runtime, "launch", return_value={}) as launch, patch(
            "experiments.successor_shuffle.qualification.summarize",
            return_value={"status": "failed"},
        ):
            with self.assertRaisesRegex(ValueError, "qualification"):
                runtime.run(root)
        self.assertEqual(len(launch.call_args_list), 3)
        self.assertTrue(
            all(c.args[1].endswith(".qualification") for c in launch.call_args_list)
        )
        status = json.loads((root / "STATUS.json").read_text())
        self.assertEqual(status["status"], "failed")
        self.assertIn("qualification", status["error"])
        self.assertFalse((root / "CHECKPOINTS.json").exists())

    def test_no_actuation_finishes_training_without_generation(self):
        root, plan = self.fixture()
        for arm in ("K", "SHUFFLED"):
            folder = root / "campaign_001/gsm8k/event_005" / arm
            folder.mkdir(parents=True)
            (folder / "RESULT.json").write_text(
                json.dumps(
                    dict(
                        checkpoint_path=arm,
                        weights_identity={"sha256": arm, "bytes": 1},
                    )
                )
            )
        with patch.object(
            runtime, "preflight", return_value={"status": "passed"}
        ), patch.object(runtime, "launch", return_value={}) as launch, patch(
            "experiments.successor_shuffle.qualification.summarize",
            return_value={"status": "passed"},
        ), patch(
            "experiments.successor_shuffle.campaign.manipulation_summary",
            return_value={"labels_changed": False, "final_export_changed": True},
        ), patch.object(
            runtime, "prepare_references"
        ) as refs, patch.object(
            runtime, "parent_probes"
        ) as probes, patch.object(
            runtime, "evaluate"
        ) as evaluate:
            runtime.run(root)
        self.assertEqual(len(launch.call_args_list), 5)
        refs.assert_not_called()
        probes.assert_not_called()
        evaluate.assert_not_called()
        status = json.loads((root / "STATUS.json").read_text())
        self.assertEqual(status["status"], "completed_no_actuation")

    def test_existing_execution_cannot_be_overwritten(self):
        root, plan = self.fixture()
        (root / "STATUS.json").write_text('{"status":"failed"}')
        with self.assertRaises(FileExistsError):
            runtime.run(root)
        self.assertEqual(
            json.loads((root / "STATUS.json").read_text())["status"], "failed"
        )
