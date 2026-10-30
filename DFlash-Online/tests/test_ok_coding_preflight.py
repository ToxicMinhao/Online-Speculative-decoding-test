"""Admission and execution-seal checks do not construct a model."""
import json
from pathlib import Path
import tempfile
import unittest

from experiments.ok_coding.preflight import storage_admission, verify_seal
from experiments.ok_coding.storage import GIB, full_campaign_budget, identity
from test_ok_coding_storage import plan_fixture


class CodingPreflightTests(unittest.TestCase):
    def test_full_capacity_and_host_reserve_are_required(self):
        plan = plan_fixture()
        capacity = full_campaign_budget(plan)
        plan["storage"] = {"mode": "retain_local", "offload_assumed": False,
                           "reserve_bytes": 8 * GIB, "campaign_capacity": capacity}
        required = capacity["peak_required_bytes"]
        self.assertTrue(storage_admission(plan, required, 80 * GIB)["admitted"])
        self.assertFalse(storage_admission(plan, required - 1, 80 * GIB)["admitted"])
        self.assertFalse(storage_admission(plan, required, 80 * GIB - 1)["admitted"])
        plan["storage"]["reserve_bytes"] = 1
        with self.assertRaises(ValueError):
            storage_admission(plan, required, 80 * GIB)

    def test_source_change_is_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sealed").mkdir()
            code = root / "code"
            code.mkdir()
            path = code / "experiment.py"
            path.write_text("# original\n")
            seal = {"execution_source": {"commit": "a" * 40, "dirty": False},
                    "files": {str(path): identity(path)},
                    "exact_file_sets": {str(code): [path.name]}}
            (root / "sealed/SEAL.json").write_text(json.dumps(seal))
            self.assertEqual(verify_seal(root)["files_checked"], 1)
            path.write_text("# changed\n")
            with self.assertRaisesRegex(ValueError, "changed"):
                verify_seal(root)


if __name__ == "__main__":
    unittest.main()
