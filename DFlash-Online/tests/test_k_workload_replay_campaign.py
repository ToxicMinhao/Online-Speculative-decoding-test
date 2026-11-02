"""CPU campaign fixtures for capture parity, fixed work and failure ordering."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from online_feedback.storage import write_json
from experiments.k_workload_replay_v1 import campaign as C
from experiments.k_workload_replay_v1 import worker as W
from experiments.k_workload_replay_v1.selection import ReplayRecord
from test_full_k_gsm8k_evaluation import FullBackend, harmful


def requests(event):
    return [dict(request_id=f"fresh_{ordinal:04d}", request_index=ordinal, global_ordinal=ordinal,
        prompt_ids=[1, 2000 + ordinal], stop_token_ids=[151645], max_new_tokens=2048,
        event=event, domain="math" if event in (1, 2, 5, 6) else "coding")
        for ordinal in range((event - 1) * 128, event * 128)]


class CaptureBackend(FullBackend):
    def __init__(self, inputs, mutation=None):
        super().__init__({row["request_index"]: row for row in inputs}, mutation)

    def generate(self, prompt_ids, **options):
        record = super().generate(prompt_ids, **options)
        if options["purpose"] == "serving":
            record["metadata"]["feedback_capture_in_timed_request"] = True
            record["metadata"]["capture"].update(collect_features=True, feature_bytes=16,
                feature_path=str(Path(options["save_to"]).with_suffix(".pt")))
        write_json(options["save_to"], record)
        return record


def fake_prepare(backend, ledger, record, request, event_root, root, plan):
    ledger.call("terminal_preparation_forwards", lambda: None)
    binding = dict(record_id=request["request_id"], global_ordinal=request["global_ordinal"],
                   path=str(event_root / "features" / (request["request_id"] + ".pt")), sha256="0" * 64)
    descriptor = ReplayRecord(request["request_id"], request["global_ordinal"], request["event"], (1.0, 0.0))
    receipt = dict(preparation=dict(teacher_preparation_s=.01, feature_persistence_s=.02), descriptor_s=.001)
    return binding, descriptor, receipt


class ReplayCampaignTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        (self.root / "campaign_001").mkdir()

    def record(self, event=1, capture=True):
        request = requests(event)[0]
        backend = CaptureBackend([request])
        backend.checkpoint_id = "/K"
        record = backend.generate(request["prompt_ids"], request_id=request["request_id"],
            request_index=request["request_index"], session_id=f"event_{event:03d}",
            purpose="serving" if capture else "evaluation", save_to=self.root / "record.json")
        return request, record

    def test_capture_contract_only_relaxes_declared_feature_mode(self):
        request, captured = self.record(capture=True)
        self.assertEqual(C.check_stream_record(captured, request, "/K", capture=True), [])
        self.assertIn("feedback_capture_or_purpose", C.check_stream_record(captured, request, "/K", capture=False))
        for key, value in (("collect_features", False), ("feature_bytes", 0), ("native_capture_validated", False)):
            bad = deepcopy(captured)
            bad["metadata"]["capture"][key] = value
            self.assertIn("required_native_capture", C.check_stream_record(bad, request, "/K", capture=True))
        bad = deepcopy(captured)
        bad["request_index"] += 1
        self.assertIn("original_request_identity", C.check_stream_record(bad, request, "/K", capture=True))
        bad = deepcopy(captured)
        bad["metadata"]["trace"][0]["draft_tokens"][0] += 99
        self.assertTrue(C.check_stream_record(bad, request, "/K", capture=True))
        _, plain = self.record(capture=False)
        self.assertEqual(C.check_stream_record(plain, request, "/K", capture=False, canonical=captured), [])

    def test_stream_outputs_always_match_counts_only_neutral_events(self):
        request, record = self.record()
        group = {request["request_id"]: {arm: deepcopy(record) for arm in C.ARMS}}
        self.assertEqual(C.compare_stream(group, 1), [])
        group[request["request_id"]]["R"]["metadata"]["verification_calls"] += 1
        for event in (1, 2):
            self.assertIn((request["request_id"], "R_neutral_count_mismatch"), C.compare_stream(group, event))
        self.assertEqual(C.compare_stream(group, 3), [])
        group[request["request_id"]]["R"]["response_ids"][0] += 1
        self.assertIn((request["request_id"], "R_canonical_mismatch"), C.compare_stream(group, 3))
        del group[request["request_id"]]["U"]
        self.assertEqual(C.compare_stream(group, 3), [(request["request_id"], "incomplete_methods")])

    def serve(self, event, backend):
        ledger = C.WorkLedger(self.root / "campaign_001/WORK.jsonl", C.SERVING_LIMITS)
        costs = {name: 0.0 for name in ("activation_s", "teacher_preparation_s", "feature_persistence_s", "descriptor_s")}
        progress = dict(costs=costs)
        folder = self.root / f"campaign_001/events/event_{event:03d}"
        with patch.object(C, "prepare_packet", side_effect=fake_prepare), patch.object(C, "require_write_space"), \
                patch.object(C, "check_growth", return_value={"status": "passed", "resource_check_wall_s": 0.01}):
            result = C.serve_event(backend, ledger, requests(event), {arm: "/" + arm for arm in C.ARMS},
                                   folder, self.root, {"storage": {}}, progress)
        return result, folder, progress

    def test_event_one_fixed_budget_only_k_prepares_and_two_controls_are_included(self):
        backend = CaptureBackend(requests(1))
        (bindings, descriptors), folder, progress = self.serve(1, backend)
        work = C.read_work(self.root / "campaign_001/WORK.jsonl", C.SERVING_LIMITS)
        self.assertEqual(work["completed"], dict(pre_update_generations=384, capture_parity_generations=2,
                                                terminal_preparation_forwards=128))
        self.assertEqual(work["unknown"], {key: 0 for key in C.SERVING_LIMITS})
        self.assertEqual(len(backend.executions), 386)
        self.assertEqual(len(bindings), 128)
        self.assertEqual([r.global_ordinal for r in descriptors], list(range(128)))
        rows = [json.loads(line) for line in (folder / "MEASUREMENTS.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 384)
        self.assertEqual([row["method"] for row in rows[::128]], ["K", "U", "R"])
        self.assertEqual(sum(row["capture"] for row in rows), 128)
        self.assertEqual(len(json.loads((folder / "PARITY.json").read_text())), 2)
        self.assertEqual(json.loads((folder / "POOL.json").read_text())["teacher_terminal_forwards"], 128)
        self.assertAlmostEqual(progress["costs"]["teacher_preparation_s"], 1.28)

    def test_event_three_rotates_and_contains_exactly_two_coding_parity_controls(self):
        backend = CaptureBackend(requests(3))
        (_, descriptors), folder, _ = self.serve(3, backend)
        rows = [json.loads(line) for line in (folder / "MEASUREMENTS.jsonl").read_text().splitlines()]
        self.assertEqual([row["method"] for row in rows[::128]], ["R", "K", "U"])
        self.assertEqual([r.global_ordinal for r in descriptors], list(range(256, 384)))
        parity = json.loads((folder / "PARITY.json").read_text())
        self.assertEqual([row["request_id"] for row in parity], ["fresh_0256", "fresh_0257"])
        self.assertEqual(len(backend.executions), 386)

    def test_capture_parity_failure_stops_before_other_arms_without_extra_retry(self):
        def mutation(record, options):
            if options["purpose"] == "evaluation":
                harmful(record)
        backend = CaptureBackend(requests(1), mutation)
        with self.assertRaisesRegex(ValueError, "Capture-mode parity"):
            self.serve(1, backend)
        work = C.read_work(self.root / "campaign_001/WORK.jsonl", C.SERVING_LIMITS)
        self.assertEqual(work["completed"], dict(pre_update_generations=1, capture_parity_generations=1,
                                                terminal_preparation_forwards=1))
        self.assertEqual(len(backend.executions), 2)
        folder = self.root / "campaign_001/events/event_001"
        parity = json.loads((folder / "PARITY.json").read_text())
        self.assertEqual(parity[0]["errors"], ["capture_mode_verification_count"])
        self.assertEqual(len((folder / "MEASUREMENTS.jsonl").read_text().splitlines()), 1)
        self.assertFalse((folder / "POOL.json").exists())

    def test_malformed_return_keeps_raw_invalid_row_and_completed_attempt(self):
        def mutation(record, options):
            del record["checkpoint_id"]
        backend = CaptureBackend(requests(1), mutation)
        with self.assertRaisesRegex(ValueError, "malformed_native_record"):
            self.serve(1, backend)
        folder = self.root / "campaign_001/events/event_001"
        rows = [json.loads(line) for line in (folder / "MEASUREMENTS.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0]["valid_measurement"])
        self.assertTrue(Path(rows[0]["record_path"]).is_file())
        raw = json.loads(Path(rows[0]["record_path"]).read_text())
        self.assertNotIn("checkpoint_id", raw)
        work = C.read_work(self.root / "campaign_001/WORK.jsonl", C.SERVING_LIMITS)
        self.assertEqual(work["completed"]["pre_update_generations"], 1)
        self.assertEqual(work["attempts"]["terminal_preparation_forwards"], 0)
        self.assertEqual(len(backend.executions), 1)

    def test_initialization_failure_retains_zero_work_accounting_without_backend_start(self):
        root = self.root / "whole"
        (root / "sealed").mkdir(parents=True)
        write_json(root / "inputs.json", dict(training=[{}] * 1024, evaluation=[{}] * 256))
        write_json(root / "canonicals.json", {})
        plan = dict(requests=str(root / "inputs.json"), canonicals=str(root / "canonicals.json"),
                    budget=C.BUDGET, storage={}, execution_source={"commit": "a" * 40}, draft="/released")
        write_json(root / "sealed/PLAN.json", plan)
        with patch.object(C, "admit"), patch.object(C, "execute_worker", side_effect=ValueError("initial state differs")), \
                patch("online_feedback.sglang_backend.SGLangDFlashBackend") as backend, \
                patch("experiments.k_workload_replay_v1.report.build_report") as report:
            with self.assertRaisesRegex(ValueError, "initial state differs"):
                C.run(root)
        backend.assert_not_called()
        report.assert_called_once_with(root)
        status = json.loads((root / "campaign_001/STATUS.json").read_text())
        self.assertEqual(status["status"], "failed")
        self.assertEqual(status["events_completed"], 0)
        self.assertEqual(status["serving_work"]["attempts"], {key: 0 for key in C.SERVING_LIMITS})

    def test_initial_worker_never_trains_and_close_failure_preserves_original_error(self):
        (self.root / "sealed").mkdir()
        original = self.root / "original.yaml"
        original.write_text("run_id: fixture\n")
        write_json(self.root / "sealed/PLAN.json", dict(configs={arm: "/config_" + arm for arm in C.ARMS},
            binding="0" * 64, original_config=str(original)))
        with patch.object(W, "WorkloadReplayTrainer") as trainer_class:
            trainer = trainer_class.return_value
            trainer.save_initial.return_value = {"state_path": "/immutable_initial"}
            result = W.run(self.root, "U", "initialize")
            trainer.save_initial.assert_called_once_with(
                comparator_path=self.root / "campaign_001/arms/K/trainer/initial_state.pt", shared_snapshot=True)
            trainer.train_event.assert_not_called()
            trainer.close.assert_called_once()
        self.assertEqual(result["extra_forwards"], 0)
        self.assertEqual(result["extra_optimizer_steps"], 0)
        self.assertEqual(result["status"], "passed")
        with patch.object(W, "WorkloadReplayTrainer") as trainer_class:
            trainer = trainer_class.return_value
            trainer.save_initial.side_effect = ValueError("initial qualification failed")
            trainer.close.side_effect = RuntimeError("cleanup failed")
            with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
                W.run(self.root, "R", "initialize")
        failure = json.loads((self.root / "campaign_001/arms/R/INITIAL_WORKER.json").read_text())
        self.assertEqual(failure["status"], "failed")
        self.assertIn("initial qualification failed", failure["error"])
        self.assertIn("cleanup failed", failure["cleanup_error"])


if __name__ == "__main__":
    unittest.main()
