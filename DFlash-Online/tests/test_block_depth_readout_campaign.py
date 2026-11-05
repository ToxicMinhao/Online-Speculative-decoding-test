"""CPU process/result/retention contracts at the readout campaign boundary."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from experiments.block_depth import full_campaign as campaign
from online_feedback.storage import write_json


class CompletedProcess:
    pid = 12001
    returncode = 0

    def __init__(self, command, *, result, **kwargs):
        self.command, self.environment = command, kwargs["env"]
        output = Path(command[command.index("--output") + 1])
        write_json(output / "RESULT.json", result)

    def wait(self, timeout=None):
        return self.returncode

    def poll(self):
        return self.returncode


def passed_result():
    return {"optimizer_steps": 128, "response_presentations": 256,
            "frozen_parameter_versions_unchanged": True,
            "boundary_check": {"status": "passed"}}


class ReadoutCampaignTests(unittest.TestCase):
    def test_group_subprocess_binds_own_resume_and_gpu_and_accepts_validated_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calls = []
            def launch(command, **kwargs):
                process = CompletedProcess(command, result=passed_result(), **kwargs)
                calls.append(process)
                return process
            previous = root / "group1/training_state.pt"
            with mock.patch.object(campaign.subprocess, "Popen", side_effect=launch):
                folder, result = campaign.train_group(root, {"training_python": "/python", "training_config": "/config"},
                    2, root / "pool.json", previous)
            self.assertEqual(calls[0].environment["CUDA_VISIBLE_DEVICES"], "1")
            command = calls[0].command
            self.assertEqual(command[command.index("--group") + 1], "2")
            self.assertEqual(command[command.index("--resume-state") + 1], str(previous))
            self.assertEqual(result["boundary_check"]["status"], "passed")
            self.assertEqual(json.loads((folder / "LAUNCH.json").read_text())["status"], "completed")

    def test_zero_exit_with_incorrect_work_is_failed_not_completed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = {**passed_result(), "optimizer_steps": 127}
            with mock.patch.object(campaign.subprocess, "Popen", side_effect=lambda command, **kwargs:
                    CompletedProcess(command, result=result, **kwargs)):
                with self.assertRaises(ValueError):
                    campaign.train_group(root, {"training_python": "/python", "training_config": "/config"},
                        1, root / "pool.json", None)
            receipt = json.loads((root / "readout/group_001/trainer/LAUNCH.json").read_text())
            self.assertEqual(receipt["exit_code"], 0)
            self.assertEqual(receipt["status"], "failed")

    def make_retention(self, root, *, boundary="passed"):
        folder = root / "readout/group_001"
        trainer = folder / "trainer"
        feature = folder / "loop/features/request.pt"
        feature.parent.mkdir(parents=True)
        feature.write_bytes(b"consumed teacher tensor fixture")
        write_json(feature.with_suffix(".json"), {"status": "prepared", "record_id": "request"})
        pool = {"features": [str(feature)], "identities": {str(feature): campaign.identity(feature)}}
        write_json(folder / "POOL.json", pool)
        result = passed_result()
        result["boundary_check"]["status"] = boundary
        write_json(trainer / "RESULT.json", result)
        state = trainer / "training_state.pt"
        state.write_bytes(b"validated compact recovery fixture")
        export = trainer / "events/group_001/export/readout.safetensors"
        export.parent.mkdir(parents=True)
        export.write_bytes(b"validated compact export fixture")
        return folder, trainer, feature, pool, state, export

    def test_retirement_preserves_compact_successor_and_rejects_unvalidated_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder, trainer, feature, pool, state, export = self.make_retention(root, boundary="failed")
            with self.assertRaises(ValueError):
                campaign.retire_consumed_group(root, folder, pool, trainer)
            self.assertTrue(feature.exists())
            result = passed_result()
            write_json(trainer / "RESULT.json", result)
            receipt = campaign.retire_consumed_group(root, folder, pool, trainer)
            self.assertEqual(receipt["status"], "completed")
            self.assertFalse(feature.exists())
            self.assertTrue(feature.with_suffix(".json").exists())
            self.assertTrue(state.exists())
            self.assertTrue(export.exists())

    def test_changed_feature_blocks_retirement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder, trainer, feature, pool, _, _ = self.make_retention(root)
            feature.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                campaign.retire_consumed_group(root, folder, pool, trainer)
            self.assertTrue(feature.exists())


if __name__ == "__main__":
    unittest.main()
