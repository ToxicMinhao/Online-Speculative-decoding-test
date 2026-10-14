"""CPU regressions for complete activation cost versus request timing."""
import copy
import unittest
from unittest import mock

try:
    from . import native, report
    from .test_report import episode
except ImportError:
    import native
    import report
    from test_report import episode


class Clock:
    def __init__(self):
        self.seconds = 0.0

    def __call__(self):
        return self.seconds

    def advance(self, seconds):
        self.seconds += seconds


class TimedWorker:
    """Model-free RPC fixture with distinct validation and native timers."""
    validation_s = 7.0
    publication_s = 2.0
    flush_s = 3.0
    generation_s = {"F": 10.0, "U1": 4.0, "U2": 4.0, "AR": 6.0}

    def __init__(self, clock, kind="sd"):
        self.clock = clock
        self.kind = kind
        self.calls = []

    def call(self, operation, body):
        self.calls.append((operation, copy.deepcopy(body)))
        if operation == "publish":
            # Checkpoint hashing precedes the worker's native-publication timer.
            self.clock.advance(self.validation_s)
            self.clock.advance(self.publication_s)
            return {"status": "passed", "checkpoint_id": body["checkpoint_id"],
                    "activation_s": self.publication_s,
                    "native_receipt": {"status": "passed"}}
        if operation == "flush":
            self.clock.advance(self.flush_s)
            return {"seconds": self.flush_s}
        if operation == "generate":
            checkpoint = body["checkpoint_id"]
            self.clock.advance(self.generation_s[checkpoint])
            return {"status": "passed", "output_ids": [5, 99],
                    "input_ids_sha256": native.object_hash(body["input_ids"]),
                    "verification_calls": 0 if checkpoint == "AR" else 2,
                    "accepted_draft_tokens": 0 if checkpoint == "AR" else 1}
        raise AssertionError("Unexpected worker operation: " + operation)


def backend(clock, kind="sd"):
    value = native.NativeBackend({})
    value.worker = TimedWorker(clock, kind)
    value.checkpoints = {name: {"checkpoint_id": name} for name in ("F", "U1", "U2")}
    value.active_checkpoint = "F" if kind == "sd" else "AR"
    return value


class ActivationAccountingTests(unittest.TestCase):
    def test_publication_includes_validation_and_preserves_native_subtimer(self):
        clock = Clock()
        engine = backend(clock)
        with mock.patch.object(native.time, "perf_counter", clock):
            result = engine.publish("U1")

        self.assertEqual(result["activation_s"], 9.0)
        self.assertEqual(result["native_activation_s"], 2.0)
        self.assertEqual(engine.totals["activation_s"], 9.0)
        self.assertEqual(engine.active_checkpoint, "U1")
        self.assertEqual(clock.seconds, 9.0)
        self.assertEqual([name for name, _ in engine.worker.calls], ["publish"])

    def test_serving_keeps_validation_publication_and_flush_outside_request(self):
        clock = Clock()
        engine = backend(clock)
        with mock.patch.object(native.time, "perf_counter", clock):
            result = engine.serve([1, 2], "U1", capture=True, turn_index=17)

        self.assertEqual(result["activation_s"], 9.0)
        self.assertEqual(result["native_activation_s"], 2.0)
        self.assertEqual(result["request_s"], 4.0)
        self.assertEqual(result["cache_maintenance_s"], 3.0)
        self.assertEqual(result["startup_s"], 0.0)
        self.assertEqual(clock.seconds, 16.0)
        self.assertEqual(result["request_s"] + result["activation_s"] +
                         result["cache_maintenance_s"], clock.seconds)
        self.assertEqual([name for name, _ in engine.worker.calls],
                         ["publish", "flush", "generate"])

    def test_same_checkpoint_does_not_repeat_activation_or_native_subtimer(self):
        clock = Clock()
        engine = backend(clock)
        with mock.patch.object(native.time, "perf_counter", clock):
            engine.publish("U1")
            unchanged = engine.publish("U1")
            result = engine.serve([1, 2], "U1", capture=True, turn_index=18)

        self.assertEqual(unchanged["activation_s"], 0.0)
        self.assertEqual(unchanged["native_activation_s"], 0.0)
        self.assertEqual(result["activation_s"], 0.0)
        self.assertEqual(result["native_activation_s"], 0.0)
        self.assertEqual(result["request_s"], 4.0)
        self.assertEqual(engine.totals["activation_s"], 9.0)
        self.assertEqual(clock.seconds, 16.0)
        self.assertEqual([name for name, _ in engine.worker.calls],
                         ["publish", "flush", "generate"])

    def test_ar_request_has_no_checkpoint_activation(self):
        clock = Clock()
        engine = backend(clock, "ar")
        with mock.patch.object(native.time, "perf_counter", clock):
            result = engine.serve([1, 2], "AR", turn_index=17, repeat=0)

        self.assertEqual(result["activation_s"], 0.0)
        self.assertEqual(result["native_activation_s"], 0.0)
        self.assertEqual(result["request_s"], 6.0)
        self.assertEqual(result["cache_maintenance_s"], 3.0)
        self.assertEqual(engine.totals["activation_s"], 0.0)
        self.assertEqual(clock.seconds, 9.0)
        self.assertEqual([name for name, _ in engine.worker.calls], ["flush", "generate"])

    def test_complete_episode_counts_full_activation_once_outside_paired_timing(self):
        source = episode()
        clock = Clock()
        canonical, diagnostic, ar = backend(clock), backend(clock), backend(clock, "ar")
        with mock.patch.object(native.time, "perf_counter", clock):
            for row in source["canonical"]:
                row.update(canonical.serve([1, row["turn"]], row["checkpoint_id"],
                                           turn_index=row["turn"]))
            source["baseline_early"] = copy.deepcopy(source["canonical"][:16])
            for row in source["scores"]:
                engine = ar if row["checkpoint_id"] == "AR" else diagnostic
                row.update(engine.serve([1, row["turn"]], row["checkpoint_id"],
                                        turn_index=row["turn"], repeat=row["repeat"]))

        # Cleanup is aggregate diagnostic bookkeeping, not another episode cost.
        source["cleanup"]["activation_s"] = (canonical.totals["activation_s"] +
                                               diagnostic.totals["activation_s"])
        result = report.summarize([source])["episodes"][0]

        self.assertEqual(canonical.totals["activation_s"], 18.0)
        self.assertGreater(diagnostic.totals["activation_s"], 0.0)
        self.assertEqual(sum(row["native_activation_s"] for row in source["canonical"]), 4.0)
        self.assertEqual(result["canonical_serving_s"], 224.0)
        self.assertEqual(result["canonical_cache_maintenance_s"], 96.0)
        self.assertEqual(result["canonical_publication_s"], 18.0)
        self.assertEqual(result["observed_synchronous_update_wait_s"], 30.0)
        self.assertEqual(result["accounted_adaptive_episode_s"], 368.0)
        self.assertEqual(result["frozen_episode_reference_s_by_repeat"], [416.0] * 3)
        self.assertEqual(len(result["windows"]), 6)
        for window in result["windows"]:
            learned = "U1" if window["turns"] == "17–24" else "U2"
            self.assertTrue(window["paired_same_generation_valid"])
            self.assertTrue(window["AR_same_generation_valid"])
            self.assertEqual(window["metrics"]["F"]["complete_request_s"], 80.0)
            self.assertEqual(window["metrics"][learned]["complete_request_s"], 32.0)
            self.assertEqual(window["metrics"]["AR"]["complete_request_s"], 48.0)
            self.assertAlmostEqual(window["latency_reduction_percent"], 60.0)


if __name__ == "__main__":
    unittest.main()
