import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock

from experiments.longer_run_headroom import native_worker as native


class OneBlockTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "receipt.json"
        self.state = SimpleNamespace(
            handle="h", terminal=False, output_ids=[10], original_max_new_tokens=2048
        )
        self.req = SimpleNamespace(
            output_ids=[10], sampling_params=SimpleNamespace(max_new_tokens=2048)
        )
        self.req.finished = lambda: len(self.req.output_ids) == 4
        self.record = SimpleNamespace(trace=[], teacher_ids=[10])
        self.record.finish = lambda ids: {"blocks": self.record.trace}
        self.batch = SimpleNamespace(reqs=[self.req])
        self.session = SimpleNamespace(
            request={"prompt_ids": [7, 8], "canonical_response_ids": [10, 11, 12, 99]},
            reconstruction={"status": "passed"},
            capture_enabled=True,
            install=Mock(),
            close=Mock(),
            rebuild=Mock(return_value=self.batch),
            successors={1: self.state},
        )

        def forward(batch):
            batch.reqs[0].output_ids.extend([11])
            self.record.trace.append({"test_block": True})

        self.session.worker = SimpleNamespace(
            forward_batch_generation=Mock(side_effect=forward)
        )
        self.scheduler = SimpleNamespace(
            _online_publication={"checkpoint_path": "child"}
        )
        self.stack = []
        for ctx in [
            patch.object(native.base, "_SESSION", None),
            patch.object(native.base, "_SESSIONS", [self.session]),
            patch.object(native.base, "_BUDGET", SimpleNamespace(check_time=Mock())),
            patch.object(native.sglang_hooks, "strict_idle"),
            patch.object(native.sglang_hooks, "flush_idle"),
            patch.object(
                native.sglang_hooks, "RequestCapture", return_value=self.record
            ),
            patch("torch.cuda.synchronize"),
        ]:
            ctx.start()
            self.addCleanup(ctx.stop)

    def test_exactly_one_round_keeps_original_cap_and_exits_before_eos(self):
        r = native.transition_one_block(self.scheduler, "h", "child", str(self.path))
        self.assertEqual(self.session.worker.forward_batch_generation.call_count, 1)
        self.assertEqual(self.req.sampling_params.max_new_tokens, 2048)
        self.assertFalse(r["native_finished"])
        self.assertTrue(r["cap_unchanged"])
        self.assertEqual(r["original_remaining_budget"], 2047)
        self.assertIsNone(native.base._SESSION)

    def test_terminal_round_exits_once_without_shortening_budget(self):
        self.session.worker.forward_batch_generation.side_effect = lambda batch: (
            self.req.output_ids.extend([11, 12, 99]),
            self.record.trace.append({"test_block": True}),
        )
        r = native.transition_one_block(self.scheduler, "h", "child", str(self.path))
        self.assertTrue(r["native_finished"])
        self.assertEqual(r["original_cap"], 2048)

    def test_failure_is_saved_and_global_capture_is_removed(self):
        self.session.worker.forward_batch_generation.side_effect = lambda batch: (
            self.req.output_ids.append(999),
            self.record.trace.append({"bad_block": True}),
        )
        with self.assertRaisesRegex(ValueError, "output/cap"):
            native.transition_one_block(self.scheduler, "h", "child", str(self.path))
        r = json.loads(self.path.read_text())
        self.assertEqual(r["status"], "failed")
        self.assertEqual(len(r["completed_trace"]), 1)
        self.assertIsNone(native.base._SESSION)
        self.assertIsNone(native.sglang_hooks._ACTIVE)


if __name__ == "__main__":
    unittest.main()
