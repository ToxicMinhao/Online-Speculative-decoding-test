"""Balanced endpoint execution and durable failure evidence, without models."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.k_workload_replay_v1 import evaluation as E
from experiments.k_workload_replay_v1.contract import final_schedule
from test_full_k_gsm8k_evaluation import FullBackend, harmful, requests


def panel():
    return [dict(row, domain="math" if i % 2 == 0 else "coding") for i, row in enumerate(requests(256))]


class ReplayEvaluationTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.inputs = panel()
        self.canonicals = {row["request_id"]: dict(response_ids=[10, 11, 12, 13, 151645],
            finish_reason="eos", source_record_path="/original/" + row["request_id"] + ".json") for row in self.inputs}
        self.checkpoints = {method: "/" + method for method in E.ARMS}

    def tiny_schedule(self):
        slots = [slot for slot in final_schedule(self.inputs) if slot["panel_ordinal"] < 2]
        return patch.object(E, "final_schedule", return_value=slots)

    def test_fixed_full_schedule_domain_position_permutation_precedence_balance(self):
        slots = final_schedule(self.inputs)
        self.assertEqual(len(slots), 2304)
        groups = {}
        for row in slots:
            groups.setdefault((row["block_ordinal"], row["repeat"]), {})[row["method"]] = row["position"]
        orders = Counter(tuple(sorted(order, key=order.get)) for order in groups.values())
        self.assertEqual(len(orders), 6)
        self.assertEqual(set(orders.values()), {2})
        for arm in E.ARMS:
            self.assertEqual(Counter(g[arm] for g in groups.values()), {0: 4, 1: 4, 2: 4})
            for domain in ("math", "coding"):
                selected = [r for r in slots if r["method"] == arm and r["domain"] == domain]
                self.assertEqual(Counter(r["position"] for r in selected), {0: 128, 1: 128, 2: 128})
        for a, b in (("K", "U"), ("K", "R"), ("U", "R")):
            self.assertEqual(sum(g[a] < g[b] for g in groups.values()), 6)

    def test_complete_repeats_raw_audit_and_no_warmup_or_reference_mutation(self):
        backend = FullBackend(self.inputs)
        before = deepcopy(self.canonicals)
        with self.tiny_schedule():
            result = E.evaluate_final(backend, self.inputs, self.checkpoints, self.root / "final", canonicals=self.canonicals)
            audit = E.audit_final(self.root / "final", self.inputs, self.checkpoints, canonicals=self.canonicals)
        self.assertEqual(result["completed_valid_generations"], 18)
        self.assertEqual(len(backend.executions), 18)
        self.assertEqual(audit["errors"], [])
        self.assertEqual(self.canonicals, before)
        self.assertEqual(set(backend.weight_checks), set(self.checkpoints.values()))
        self.assertEqual(result["extra_generations"], 0)
        self.assertAlmostEqual(sum(r["request_latency_s"] for r in audit["rows"]), 18)
        self.assertGreater(result["activation_s"], 0)

    def test_wrong_complete_output_keeps_raw_invalid_and_unattempted_evidence(self):
        def mutate(record, options):
            record["response_ids"][0] += 1
        backend = FullBackend(self.inputs, mutate)
        with self.tiny_schedule(), self.assertRaisesRegex(RuntimeError, "correctness"):
            E.evaluate_final(backend, self.inputs, self.checkpoints, self.root / "failed", canonicals=self.canonicals)
        status = json.loads((self.root / "failed/STATUS.json").read_text())
        self.assertEqual(status["invalid_generations"], 1)
        self.assertEqual(status["unattempted_generations"], 17)
        self.assertTrue((self.root / "failed/0000_r0_K.json").exists())
        rows = (self.root / "failed/MEASUREMENTS.jsonl").read_text().splitlines()
        self.assertEqual(len(rows), 1)
        self.assertFalse(json.loads(rows[0])["valid_measurement"])

    def test_generation_exception_preserves_unknown_attempt_and_refuses_same_root_retry(self):
        backend = FullBackend(self.inputs)
        with self.tiny_schedule(), patch.object(backend, "generate", side_effect=RuntimeError("native failed")):
            with self.assertRaisesRegex(RuntimeError, "native failed"):
                E.evaluate_final(backend, self.inputs, self.checkpoints, self.root / "failed", canonicals=self.canonicals)
            with self.assertRaises(FileExistsError):
                E.evaluate_final(backend, self.inputs, self.checkpoints, self.root / "failed", canonicals=self.canonicals)
        status = json.loads((self.root / "failed/STATUS.json").read_text())
        self.assertEqual(status["unknown_generation_attempts"], 1)
        self.assertEqual(status["returned_generations"], 0)
        self.assertEqual(status["unattempted_generations"], 17)
        self.assertFalse(status["automatic_retry"])

    def test_repeat_count_mismatch_stops_and_raw_reinspection_detects_tampering(self):
        def mutate(record, options):
            if options["request_id"] == "0000_r1_U":
                harmful(record)
        backend = FullBackend(self.inputs, mutate)
        with self.tiny_schedule(), self.assertRaisesRegex(RuntimeError, "repeat_count_mismatch"):
            E.evaluate_final(backend, self.inputs, self.checkpoints, self.root / "failed", canonicals=self.canonicals)
        backend = FullBackend(self.inputs)
        with self.tiny_schedule():
            E.evaluate_final(backend, self.inputs, self.checkpoints, self.root / "complete", canonicals=self.canonicals)
            path = self.root / "complete/0000_r0_K.json"
            raw = json.loads(path.read_text())
            raw["metadata"]["trace"][0]["draft_tokens"][0] += 101
            path.write_text(json.dumps(raw))
            audit = E.audit_final(self.root / "complete", self.inputs, self.checkpoints, canonicals=self.canonicals)
        self.assertEqual(audit["status"], "invalid_or_incomplete")
        self.assertTrue(any("0000_r0_K" in error for error in audit["errors"]))

    def test_bad_loaded_weights_stop_before_any_generation(self):
        backend = FullBackend(self.inputs)
        with self.tiny_schedule(), patch.object(backend, "check_weights", return_value=dict(
                status="passed", all_equal=False, tensors=58, source_tensor_count=58, native_tensor_count=43,
                checkpoint_path="/K", helper_matches_current_layers=True, helper_norm_matches_current_layers=True,
                fused_enabled=True)), self.assertRaisesRegex(RuntimeError, "weight mismatch"):
            E.evaluate_final(backend, self.inputs, self.checkpoints, self.root / "failed", canonicals=self.canonicals)
        self.assertEqual(backend.executions, [])
        status = json.loads((self.root / "failed/STATUS.json").read_text())
        self.assertEqual(status["unattempted_generations"], 18)

    def test_boundary_checks_surround_batches_and_gate_before_generation(self):
        backend = FullBackend(self.inputs)
        calls = []
        def check():
            calls.append(len(backend.executions))
            return {"status": "passed", "observed_generations": calls[-1]}
        with self.tiny_schedule():
            result = E.evaluate_final(backend, self.inputs, self.checkpoints, self.root / "checked",
                                      canonicals=self.canonicals, boundary_check=check)
            audit = E.audit_final(self.root / "checked", self.inputs, self.checkpoints, canonicals=self.canonicals)
        self.assertEqual(calls, list(range(0, 19, 2)))
        self.assertEqual(len(result["boundary_checks"]), 10)
        self.assertEqual(result["boundary_checks"][-1]["boundary"], "after_final_batch")
        self.assertEqual(audit["errors"], [])
        backend = FullBackend(self.inputs)
        with self.tiny_schedule(), self.assertRaisesRegex(RuntimeError, "resource gate"):
            E.evaluate_final(backend, self.inputs, self.checkpoints, self.root / "blocked", canonicals=self.canonicals,
                             boundary_check=lambda: (_ for _ in ()).throw(RuntimeError("resource gate")))
        self.assertEqual(backend.executions, [])
        status = json.loads((self.root / "blocked/STATUS.json").read_text())
        self.assertEqual(status["unattempted_generations"], 18)
        backend = FullBackend(self.inputs)
        def final_gate():
            if len(backend.executions) == 18:
                raise RuntimeError("final resource gate")
            return {"status": "passed"}
        with self.tiny_schedule(), self.assertRaisesRegex(RuntimeError, "final resource gate"):
            E.evaluate_final(backend, self.inputs, self.checkpoints, self.root / "last_gate",
                             canonicals=self.canonicals, boundary_check=final_gate)
        status = json.loads((self.root / "last_gate/STATUS.json").read_text())
        self.assertEqual(status["status"], "failed")
        self.assertEqual(status["completed_valid_generations"], 18)
        self.assertEqual(status["unattempted_generations"], 0)


if __name__ == "__main__":
    unittest.main()
