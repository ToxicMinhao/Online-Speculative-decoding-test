"""CPU-only native adapter tests; no model import, GPU query or worker launch."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


_spec = importlib.util.spec_from_file_location("followup_native_test", Path(__file__).with_name("native.py"))
native = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(native)


def checkpoint(path, label):
    path.mkdir(parents=True)
    (path / "config.json").write_text('{"synthetic":true}')
    (path / "model.safetensors").write_bytes(b"synthetic bytes, never loaded")
    (path / "version.txt").write_text("1")
    files = {p.name: {"bytes": p.stat().st_size, "sha256": native.sha(p)} for p in sorted(path.iterdir())}
    return {"path": str(path), "files": files, "checkpoint_identity_sha256": native.digest(files), "checkpoint_id": label}


class Clock:
    def __init__(self):
        self.now = 0.

    def __call__(self):
        return self.now


class MockWorker:
    kind = "sd"

    def __init__(self, clock, identity):
        self.clock, self.identity, self.calls = clock, identity, []

    def call(self, action, body):
        self.calls.append((action, copy.deepcopy(body)))
        if action == "publish":
            self.clock.now += 5.
            return {"status": "passed", "activation_s": 1.5}
        if action == "flush":
            self.clock.now += 7.
            return {"status": "passed", "seconds": 7.}
        if action == "generate":
            self.clock.now += 2.
            return {"status": "passed", "output_ids": [5, 6], "text": "synthetic", "feedback": None,
                    "capture_costs": None, "accepted_draft_tokens": 1, "verification_calls": 1,
                    "checkpoint_identity_sha256": self.identity, "input_ids_sha256": native.digest(body["input_ids"])}
        raise AssertionError(action)


class NativeAdapterTests(unittest.TestCase):
    def test_publication_and_cache_delay_excluded_from_request_timer(self):
        clock = Clock()
        backend = native.NativeBackend({})
        backend.checkpoints = {"C1": {"checkpoint_id": "C1", "checkpoint_identity_sha256": "1" * 64}}
        backend.worker, backend.active_checkpoint = MockWorker(clock, "1" * 64), "F"
        with patch.object(native.base.time, "perf_counter", clock):
            result = backend.serve([1, 2], "C1", capture=False, turn_index=25, repeat=0)
        self.assertEqual(result["request_s"], 2.)
        self.assertEqual(result["activation_s"], 5.)
        self.assertEqual(result["native_activation_s"], 1.5)
        self.assertEqual(result["cache_maintenance_s"], 7.)
        self.assertEqual(clock.now, 14.)
        self.assertEqual([action for action, _ in backend.worker.calls], ["publish", "flush", "generate"])
        self.assertFalse(backend.worker.calls[-1][1]["capture"])
        self.assertEqual(backend.worker.calls[-1][1]["checkpoint_id"], "C1")

    def test_same_available_checkpoint_has_no_publication_charge(self):
        clock = Clock()
        backend = native.NativeBackend({})
        backend.checkpoints = {"A1": {"checkpoint_id": "A1", "checkpoint_identity_sha256": "1" * 64}}
        backend.worker, backend.active_checkpoint = MockWorker(clock, "1" * 64), "A1"
        with patch.object(native.base.time, "perf_counter", clock):
            result = backend.serve([1], "A1", capture=False, turn_index=17)
        self.assertEqual(result["activation_s"], 0.)
        self.assertEqual(result["request_s"], 2.)
        self.assertEqual([action for action, _ in backend.worker.calls], ["flush", "generate"])

    def test_capture_and_unknown_arm_fail_before_any_worker_call(self):
        backend = native.NativeBackend({})
        backend.worker = MockWorker(Clock(), "1" * 64)
        for label, capture in (("F", True), ("AR", False), ("U1", False)):
            with self.assertRaises(ValueError):
                backend.serve([1], label, capture=capture, turn_index=17)
        self.assertEqual(backend.worker.calls, [])

    def test_retained_weights_require_exact_reference_binding_and_intact_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = checkpoint(root / "old" / "U1", "C1")
            config = {"external_checkpoints": {"C1": copy.deepcopy(old)}, "episode_out": str(root / "new")}
            self.assertEqual(native.admit_checkpoint(config, old), "C1")
            alternate = checkpoint(root / "other" / "U1", "C1")
            with self.assertRaisesRegex(ValueError, "Unbound retained"):
                native.admit_checkpoint(config, alternate)
            (Path(old["path"]) / "model.safetensors").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "bytes changed"):
                native.admit_checkpoint(config, old)

    def test_author_export_must_belong_to_exact_current_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {"episode_out": str(root / "new")}
            right = checkpoint(root / "new" / "checkpoints" / "U1", "A1")
            self.assertEqual(native.admit_checkpoint(config, right), "A1")
            foreign = checkpoint(root / "other" / "checkpoints" / "U1", "A1")
            with self.assertRaisesRegex(ValueError, "another episode"):
                native.admit_checkpoint(config, foreign)

    def test_retained_checkpoint_is_validated_before_base_starts_any_worker(self):
        backend = native.NativeBackend({})
        reference = {"episode_id": "R1_STABLE_CODE", "updates": [
            {"checkpoint": {"path": "/missing/followup-test-weights", "checkpoint_id": "U1",
                            "files": {}, "checkpoint_identity_sha256": "0" * 64}}]}
        with patch.object(native.base.NativeBackend, "start_episode") as start:
            with self.assertRaises(ValueError):
                backend.start_episode("R1_STABLE_CODE", reference=reference)
            start.assert_not_called()

    def test_support_contains_exact_own_response_tokens_and_no_proposed_blocks(self):
        with tempfile.TemporaryDirectory() as directory:
            backend = native.NativeBackend({})
            backend.out, backend.episode_id = Path(directory), "R1_STABLE_CODE"
            backend.target = {"identity_sha256": "2" * 64}
            backend.checkpoints = {"A1": {"checkpoint_identity_sha256": "1" * 64}}
            # turn17's response tokens are in turn18's conditioning prefix.
            item = {"turn": 18, "checkpoint_id": "A1", "checkpoint_identity_sha256": "1" * 64,
                    "input_ids": [1, 1017, 5017, 518], "output_ids": [1018, 5018]}
            result = backend.prepare_support(item)
            saved = json.loads(Path(result["metadata_binding"]["path"]).read_text())
            self.assertEqual(saved["support_turn"], 18)
            self.assertEqual(saved["input_ids"], item["input_ids"])
            self.assertEqual(saved["output_ids"], item["output_ids"])
            self.assertEqual(saved["input_token_count"], 4)
            self.assertFalse(saved["feature_capture"])
            self.assertEqual(saved["collection_kind"], "committed_token_metadata_only")
            self.assertNotIn("tensors", saved)
            self.assertNotIn("feedback", saved)
            self.assertNotIn(5017, saved["output_ids"])
            self.assertGreaterEqual(result["collection_s"], 0.)

    def test_support_rejects_wrong_chronology_and_weight_identity_before_write(self):
        with tempfile.TemporaryDirectory() as directory:
            backend = native.NativeBackend({})
            backend.out, backend.episode_id = Path(directory), "R1_STABLE_CODE"
            backend.checkpoints = {"F": {"checkpoint_identity_sha256": "1" * 64}}
            for turn, label, identity in ((25, "A2", "1" * 64), (18, "F", "1" * 64), (10, "F", "3" * 64)):
                with self.assertRaises(ValueError):
                    backend.prepare_support({"turn": turn, "checkpoint_id": label,
                                             "checkpoint_identity_sha256": identity})
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
