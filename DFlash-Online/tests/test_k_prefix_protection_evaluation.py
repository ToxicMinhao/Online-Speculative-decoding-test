"""Fake-native CPU checks for global F reference and all included serving gates."""

import json
from pathlib import Path
import tempfile
import unittest

from online_feedback.storage import write_json
from experiments.fresh_confirmation.test_evaluate import Backend as NativeFixture
from experiments.k_prefix_protection_v1.evaluation import evaluate
from experiments.k_prefix_protection_v1.contract import CELLS, METHODS, evaluation_schedule
from test_ok_coding_campaign import requests
from test_prefix_ranking_evaluation import harmful


class Backend(NativeFixture):
    def __init__(self, inputs, mutation=None):
        super().__init__(inputs)
        self.mutation_after_native = mutation
        for expected, request in zip(self.requests, inputs):
            expected["prompt_ids"] = list(request["prompt_ids"])
            expected["stop_token_ids"] = list(request["stop_token_ids"])
            expected["response_ids"][-1] = 151645

    def generate(self, prompt_ids, **kwargs):
        record = super().generate(prompt_ids, **kwargs)
        record["metadata"]["trace"][-1]["committed_tokens"][-1] = 151645
        if self.mutation_after_native:
            self.mutation_after_native(record, kwargs)
        record["metadata"]["trace"][-1]["committed_tokens"][-1] = 151645
        write_json(kwargs["save_to"], record)
        return record


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.inputs = requests(8)

    def run_cell(self, seed=17, event=4, mutation=None):
        backend = Backend(self.inputs, mutation)
        status = evaluate(backend, self.inputs, {"F": "/F", "K": f"/K{seed}_{event}", "K_HALF": f"/KH{seed}_{event}", "KP": f"/KP{seed}_{event}"},
            self.root / f"{seed}_{event}", seed=seed, event=event, canonical_path=self.root / "canonical.json")
        return backend, status

    def test_two_cells_reuse_global_F_and_each_gate_is_included96(self):
        for seed, event in CELLS:
            backend, status = self.run_cell(seed, event)
            self.assertEqual(status["completed_valid_generations"], 96)
            self.assertEqual(status["attempted_generations"], 96)
            gate = json.loads((self.root / f"{seed}_{event}" / "QUALIFICATION.json").read_text())
            self.assertEqual(gate["generations"], 96)
            self.assertEqual(len(backend.executions), 96)
            self.assertEqual(backend.switches[0], "/F")
        canonical = json.loads((self.root / "canonical.json").read_text())
        self.assertEqual(len(canonical), 8)
        self.assertTrue(all(value["source_cell"] == [17, 4] and value["source_method"] == "F"
                            for value in canonical.values()))

    def test_first_cell_defers_all_candidates_when_F_later(self):
        self.run_cell()
        events = [json.loads(line) for line in (self.root / "17_4" / "VALIDATION_EVENTS.jsonl").read_text().splitlines()]
        for method in ("K", "K_HALF", "KP"):
            self.assertEqual([row["status"] for row in events if row["label"] == "0001_r0_" + method],
                             ["canonical_pending", "valid"])

    def test_changed_F_counts_across_cells_stop_and_preserve_record(self):
        self.run_cell()

        def change(record, options):
            if record["checkpoint_id"] == "/F":
                harmful(record)

        with self.assertRaisesRegex(RuntimeError, "global_F_count_mismatch"):
            self.run_cell(29, 4, change)
        status = json.loads((self.root / "29_4" / "STATUS.json").read_text())
        self.assertEqual(status["invalid_generations"], 1)
        self.assertEqual(status["returned_generations"], 1)
        self.assertTrue((self.root / "29_4" / "0000_r0_F.json").exists())

    def test_harmful_K_is_valid_outcome_not_failure_gate(self):
        def change(record, options):
            if record["checkpoint_id"].startswith("/K"):
                harmful(record)
        _, status = self.run_cell(mutation=change)
        self.assertEqual(status["status"], "completed")

    def test_exact_full_rotation_has_position_balance_and192_pair_precedence(self):
        schedule = evaluation_schedule(requests(), seed=17, event=4)
        for method in METHODS:
            self.assertEqual([sum(row["method"] == method and row["position"] == position for row in schedule)
                              for position in range(4)], [96, 96, 96, 96])
        for left, right in (("F", "K"), ("F", "K_HALF"), ("K_HALF", "KP")):
            before = 0
            for index in range(0, len(schedule), 4):
                order = [row["method"] for row in schedule[index:index + 4]]
                before += order.index(left) < order.index(right)
            self.assertEqual(before, 192)

    def test_no_first_cell_retry_or_later_cell_without_complete_canonical(self):
        with self.assertRaises(FileNotFoundError):
            self.run_cell(29, 4)
        self.run_cell()
        with self.assertRaisesRegex(ValueError, "retry"):
            self.run_cell()


if __name__ == "__main__":
    unittest.main()
