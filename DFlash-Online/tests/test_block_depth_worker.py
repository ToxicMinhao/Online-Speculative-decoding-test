"""Lean serving counters do not turn FINAL into trace or feedback capture."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from experiments.block_depth.worker import DecodeMeter, observe_decode, depth_load


class Mode:
    def __init__(self, extend=False, idle=False):
        self.extend, self.idle = extend, idle

    def is_extend(self):
        return self.extend

    def is_idle(self):
        return self.idle


class LeanWorkerTests(unittest.TestCase):
    def test_prefill_is_excluded_and_final_keeps_only_counters(self):
        with tempfile.TemporaryDirectory() as directory:
            meter = DecodeMeter([7, 8], "FIVE", "none", Path(directory) / "rounds.json")
            req = SimpleNamespace(origin_input_ids=[7, 8], output_ids=[])
            batch = SimpleNamespace(reqs=[req], forward_mode=Mode(extend=True), is_extend_in_batch=False)
            # Deliberately no draft buffers: FINAL must not access/copy them.
            worker = SimpleNamespace(_depth_meter=meter)
            def native(worker, batch):
                req.output_ids.extend([9] if batch.forward_mode.is_extend() else [10, 11, 12])
                return SimpleNamespace(num_accepted_tokens=0 if batch.forward_mode.is_extend() else 2)
            call = observe_decode(native)
            with mock.patch("experiments.block_depth.worker.time.perf_counter", side_effect=[2., 2.25]), \
                 mock.patch("torch.cuda.synchronize", side_effect=AssertionError("FINAL must not synchronize")):
                call(worker, batch)
                self.assertEqual(meter.calls, 0)
                batch.forward_mode = Mode()
                call(worker, batch)
            receipt = meter.result()
            self.assertEqual(receipt["verification_calls"], 1)
            self.assertEqual(receipt["committed_progress"], 3)
            self.assertEqual(receipt["accepted_draft_tokens"], 2)
            self.assertEqual(receipt["decode_round_wall_s"], 0.25)
            self.assertEqual(receipt["deep_rounds"], 1)
            self.assertNotIn("trace", receipt)
            self.assertFalse(receipt["feedback_capture"])
            self.assertEqual(meter.trace, [])

    def test_gate_trace_retains_exact_proposals_seed_and_absolute_start(self):
        with tempfile.TemporaryDirectory() as directory:
            meter = DecodeMeter([7, 8], "THREE", "gate", Path(directory) / "rounds.json")
            req = SimpleNamespace(origin_input_ids=[7, 8], output_ids=[9])
            batch = SimpleNamespace(reqs=[req], forward_mode=Mode(), is_extend_in_batch=False)
            worker = SimpleNamespace(_depth_meter=meter, _depth_routes=SimpleNamespace(last_route=3),
                                     _draft_block_tokens_buf=torch.arange(9, 25).reshape(1, 16),
                                     _draft_block_positions_buf=torch.arange(2, 18).reshape(1, 16))
            def native(worker, batch):
                req.output_ids.extend([10, 11, 12])
                return SimpleNamespace(num_accepted_tokens=2)
            with mock.patch("torch.cuda.synchronize") as synchronize:
                observe_decode(native)(worker, batch)
                self.assertEqual(synchronize.call_count, 2)
            receipt = meter.result()
            self.assertEqual(receipt["shallow_rounds"], 1)
            self.assertEqual(receipt["trace"][0], dict(output_start=0, positions=list(range(2, 18)),
                draft_tokens=list(range(10, 25)), committed_progress=3, accepted_draft_tokens=2))

    def test_native_failure_preserves_compact_unknown_round_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "failed.json"
            meter = DecodeMeter([7], "FIVE", "none", path)
            req = SimpleNamespace(output_ids=[9])
            batch = SimpleNamespace(reqs=[req], forward_mode=Mode(), is_extend_in_batch=False)
            worker = SimpleNamespace(_depth_meter=meter)
            def failed(worker, batch):
                raise RuntimeError("native interruption")
            with self.assertRaisesRegex(RuntimeError, "native interruption"):
                observe_decode(failed)(worker, batch)
            receipt = json.loads(path.read_text())
            self.assertEqual(receipt["status"], "failed")
            self.assertTrue(receipt["active_round_unknown"])
            self.assertEqual(receipt["verification_calls"], 0)

    def test_load_accepts_compact_safetensors_without_replacing_values(self):
        from safetensors.torch import save_file
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "readout.safetensors"
            state = {"norm.weight": torch.ones(8, dtype=torch.bfloat16),
                     "A": torch.ones(32, 8, dtype=torch.bfloat16),
                     "B": torch.zeros(8, 32, dtype=torch.bfloat16)}
            save_file(state, path)
            routes = mock.Mock()
            routes.load_fitted.return_value = {"status": "passed"}
            with mock.patch("experiments.block_depth.worker.sglang_hooks.strict_idle"), \
                 mock.patch("experiments.block_depth.worker.controller", return_value=routes):
                result = depth_load(object(), str(path), None, str(Path(directory) / "receipt.json"))
            loaded, gate = routes.load_fitted.call_args.args
            self.assertIsNone(gate)
            self.assertEqual(set(loaded), set(state))
            self.assertTrue(all(torch.equal(loaded[key], value) for key, value in state.items()))
            self.assertEqual(result["status"], "passed")

    def test_existing_capture_path_is_unchanged_when_meter_is_absent(self):
        seen = []
        def ordinary(worker, batch):
            seen.append(batch)
            return 123
        self.assertEqual(observe_decode(ordinary)(SimpleNamespace(), "batch"), 123)
        self.assertEqual(seen, ["batch"])


if __name__ == "__main__":
    unittest.main()
