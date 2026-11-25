"""CPU checks for the sealed panel, projected storage and cumulative accounting."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from experiments.repair_conditional_kl_pilot import contract as C, resources as R
from online_feedback.storage import write_json


def inputs_fixture():
    inputs = {}
    offset = 0
    for name, count in (("train_requests", 128), ("evaluation_requests", 128),
                        ("warmup_requests", 2), ("qualification_requests", 2)):
        inputs[name] = [dict(request_id=f"sealed_{offset+i}", request_index=i,
            prompt_ids=[offset+i+1], max_new_tokens=2048, stop_token_ids=[151645, 151643])
            for i in range(count)]
        offset += count
    return inputs


class ContractTests(unittest.TestCase):
    def test_exact_one_event_panel_retains_native_serving_and_fixed_limits(self):
        inputs = inputs_fixture()
        before = deepcopy(inputs)
        self.assertEqual(C.validate_inputs(inputs), dict(status="passed", distinct_inputs=260))
        self.assertEqual(inputs, before)
        self.assertEqual(C.ARMS, ("BASE", "PAIR"))
        self.assertEqual((C.WINDOW, C.PANEL, C.REPEATS), (128, 128, 3))
        self.assertEqual(C.LIMITS, dict(elapsed_s=10800, occupied_card_s=18000,
            new_allocated_bytes=90_000_000_000, free_reserve_bytes=400_000_000_000))

    def test_cross_role_leakage_counts_order_and_native_stopping_are_rejected(self):
        for mutate in (
            lambda x: x["evaluation_requests"].pop(),
            lambda x: x["evaluation_requests"][0].update(request_id=x["train_requests"][0]["request_id"]),
            lambda x: x["qualification_requests"][0].update(prompt_ids=x["train_requests"][0]["prompt_ids"]),
            lambda x: x["train_requests"][0].update(request_index=1),
            lambda x: x["train_requests"][0].update(max_new_tokens=1024),
            lambda x: x["train_requests"][0].update(stop_token_ids=[151645]),
            lambda x: x["train_requests"][0].update(prompt_ids=[]),
            lambda x: x["train_requests"][0].update(prompt_ids=list(range(2049))),
        ):
            inputs = inputs_fixture()
            mutate(inputs)
            with self.assertRaises(ValueError):
                C.validate_inputs(inputs)


class ResourcesTests(unittest.TestCase):
    def test_capacity_includes_full_cap_witness_replacement_and_both_recoveries(self):
        requests = [dict(prompt_ids=list(range(i+1))) for i in range(128)]
        result = R.capacity(requests)
        feature = sum(i+1+2048 for i in range(128))*49164
        witness = sum(i+1+2048 for i in range(8))*49164
        overlap = (128+2049)*49164
        tensors = feature+witness+overlap+2*6_791_777_312+2*2_097_259_104
        self.assertEqual(result["active_feature_bound_bytes"], feature)
        self.assertEqual(result["retained_witness_bound_bytes"], witness)
        self.assertEqual(result["replacement_overlap_bytes"], overlap)
        self.assertEqual(result["projected_peak_bytes"], (tensors*105+99)//100+4_000_000_000)
        self.assertIn("Verified BASE feature offload", result["prerequisite"])
        longer = [dict(prompt_ids=list(range(2048))) for _ in range(128)]
        self.assertGreater(R.capacity(longer)["projected_peak_bytes"], result["projected_peak_bytes"])

    def test_projected_reserve_accounts_for_already_written_data_and_compiler_growth(self):
        with tempfile.TemporaryDirectory() as folder:
            plan = dict(root=folder, allocation_roots=[folder],
                compiler_baseline={"allocated_bytes": 10}, storage={"projected_peak_bytes": 1000})
            # 300 charged + 100 compiler leaves another 600 to write. A current
            # free-space check alone would miss this admission boundary.
            with patch.object(R, "allocated_bytes", return_value=300), \
                 patch.object(R, "compiler_snapshot", return_value={"allocated_bytes": 110}), \
                 patch.object(R.shutil, "disk_usage", return_value=SimpleNamespace(free=400_000_000_600)):
                result = R.boundary(plan)
                self.assertEqual(result["remaining_peak_bytes"], 600)
                self.assertEqual(result["clock"], dict(elapsed_s=0., occupied_s=0.))
            with patch.object(R, "allocated_bytes", return_value=300), \
                 patch.object(R, "compiler_snapshot", return_value={"allocated_bytes": 110}), \
                 patch.object(R.shutil, "disk_usage", return_value=SimpleNamespace(free=400_000_000_599)):
                with self.assertRaisesRegex(RuntimeError, "projected peak"):
                    R.boundary(plan)

    def test_actual_peak_and_cumulative_time_are_hard_boundaries(self):
        with tempfile.TemporaryDirectory() as folder:
            plan = dict(root=folder, allocation_roots=[folder],
                compiler_baseline={"allocated_bytes": 0}, storage={"projected_peak_bytes": 1000})
            Path(folder, "CLOCK.json").touch()
            with patch.object(R.shutil, "disk_usage", return_value=SimpleNamespace(free=500_000_000_000)), \
                 patch.object(R, "compiler_snapshot", return_value={"allocated_bytes": 0}), \
                 patch.object(R, "clock_status", return_value=dict(elapsed_s=0., occupied_s=0.)), \
                 patch.object(R, "allocated_bytes", return_value=1001):
                with self.assertRaisesRegex(RuntimeError, "admitted peak"):
                    R.boundary(plan)
            for clock in (dict(elapsed_s=10800, occupied_s=0), dict(elapsed_s=0, occupied_s=18000)):
                with patch.object(R.shutil, "disk_usage", return_value=SimpleNamespace(free=500_000_000_000)), \
                     patch.object(R, "compiler_snapshot", return_value={"allocated_bytes": 0}), \
                     patch.object(R, "allocated_bytes", return_value=0), \
                     patch.object(R, "clock_status", return_value=clock):
                    with self.assertRaisesRegex(TimeoutError, "cumulative time"):
                        R.boundary(plan)

    def test_clock_cannot_restart_and_failed_work_stays_pending(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(R.time, "time", return_value=123.):
                R.initialize({"root": folder})
            before = Path(folder, "CLOCK.json").read_bytes()
            with self.assertRaisesRegex(FileExistsError, "no automatic retry"):
                R.initialize({"root": folder})
            self.assertEqual(Path(folder, "CLOCK.json").read_bytes(), before)
            path = Path(folder, "WORK.json")
            ledger = R.Ledger(path)
            self.assertEqual(ledger.call({"generation": 2}, "success", lambda: 7), 7)
            def fail():
                state = R.read(path)
                self.assertEqual(state["attempted"]["generation"], 3)
                self.assertEqual(state["completed"]["generation"], 2)
                self.assertEqual(state["pending"]["1"]["label"], "failure")
                raise RuntimeError("retained failure")
            with self.assertRaisesRegex(RuntimeError, "retained failure"):
                ledger.call({"generation": 1}, "failure", fail)
            state = R.read(path)
            self.assertEqual(state["operations"], 2)
            self.assertEqual(state["attempted"], {"generation": 3})
            self.assertEqual(state["completed"], {"generation": 2})
            self.assertEqual(state["pending"]["1"], dict(label="failure", counts={"generation": 1}))
            R.Ledger(path)
            self.assertEqual(R.read(path), state)


if __name__ == "__main__":
    unittest.main()
