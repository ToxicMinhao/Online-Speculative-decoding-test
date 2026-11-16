"""CPU fixtures for fixed ordering, durable failures and conditional inference."""

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.persistent_prefix_withholding import evaluation as E, analysis as A
from test_full_k_gsm8k_evaluation import FullBackend, requests


def inputs():
    rows = requests(128)
    for i, row in enumerate(rows):
        row["source_id"] = f"gsm8k:main:test:{i}"
    return rows


def permutations():
    rng = np.random.Generator(np.random.PCG64(1708))
    return [[rng.permutation(np.arange(b * 32, (b + 1) * 32)).tolist() for b in range(4)] for _ in range(3)]


def references(rows):
    return {r["request_id"]: dict(prompt_ids=r["prompt_ids"], response_ids=[10, 11, 12, 13, 151645],
        finish_reason="eos", source_record_path="/retained/" + r["request_id"],
        verified=True, binding_digest="authenticated", reference_kind="native_greedy_target_verifier") for r in rows}


def quiet():
    return dict(status="passed", physical_gpu_uuid="GPU-fixed", both_cards_checked=True,
        other_gpu_idle=True, host_quiet=True, storage_admitted=True, time_admitted=True)


def scalar_rows():
    rows = []
    for ordinal in range(128):
        for repeat in range(3):
            for method in E.METHODS:
                calls = 10 + ordinal % 7 - (method == "E")
                native = 99 - calls
                rows.append(dict(phase="scored", event=8, scope=E.SCOPE,
                    panel_ordinal=ordinal, block_ordinal=ordinal // 32,
                    request_id=f"eval{ordinal}", source_id=f"gsm8k:main:test:{ordinal}",
                    repeat=repeat, method=method, physical_gpu_uuid="GPU-fixed",
                    checkpoint_id="/" + method, verification_calls=calls,
                    native_accepted_draft_tokens=native, accepted_draft_tokens=native,
                    committed_progress=99, output_tokens=100,
                    native_acceptance_histogram={"0": calls - 1, str(native): 1},
                    finish_reason="eos", cap_hit=False, validation_status="valid",
                    valid_measurement=True, errors=[], request_latency_s=float(calls + repeat),
                    native_generate_s=calls + repeat - .1, cache_maintenance_s=.1))
    return rows


class ScheduleTests(unittest.TestCase):
    def test_exact_six_orders_twice_and_request_permutations(self):
        p = permutations()
        schedule = E.evaluation_schedule(inputs(), p)
        self.assertEqual(len(schedule), 36)
        self.assertEqual(sum(len(r["request_ordinals"]) for r in schedule), 1152)
        self.assertEqual(E.schedule_balance(schedule)["warmups"], 72)
        expected = [("N", "E", "S"), ("N", "S", "E"), ("E", "S", "N"), ("S", "E", "N"),
            ("E", "S", "N"), ("S", "E", "N"), ("S", "N", "E"), ("E", "N", "S"),
            ("S", "N", "E"), ("E", "N", "S"), ("N", "E", "S"), ("N", "S", "E")]
        orders = [tuple(r["method"] for r in schedule[i:i+3]) for i in range(0, 36, 3)]
        self.assertEqual(orders, expected)
        self.assertEqual(set(Counter(orders).values()), {2})
        for entry in schedule:
            self.assertEqual(entry["request_ordinals"], p[entry["repeat"]][entry["block_ordinal"]])
        self.assertEqual(E.schedule_balance(schedule)["positions"], {m: [4, 4, 4] for m in E.METHODS})

    def test_cross_block_permutation_and_unverified_reference_rejected(self):
        p = permutations()
        p[0][0][0] = 127
        with self.assertRaisesRegex(ValueError, "fixed block"):
            E.evaluation_schedule(inputs(), p)
        refs = references(inputs())
        refs["eval0"]["reference_kind"] = "candidate_agreement"
        with self.assertRaisesRegex(ValueError, "provenance"):
            E.validate_references(inputs(), refs)

    def test_boundary_requires_both_cards_and_host(self):
        self.assertEqual(E.boundary_errors(quiet(), "GPU-fixed"), [])
        for key in ("both_cards_checked", "other_gpu_idle", "host_quiet"):
            record = quiet()
            record[key] = False
            self.assertIn(key, E.boundary_errors(record, "GPU-fixed"))


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.requests = inputs()
        self.refs = references(self.requests)
        self.warmups = deepcopy(self.requests[:2])
        for i, row in enumerate(self.warmups):
            row.update(request_id=f"warm{i}", source_id=f"old:{i}", prompt_ids=[1, 9000+i])
        self.warmrefs = references(self.warmups)
        self.paths = {m: "/" + m for m in E.METHODS}
        self.p = permutations()

    def run_final(self, backend, folder):
        return E.evaluate_final(backend, self.requests, self.paths, self.root/folder,
            canonicals=self.refs, warmup_requests=self.warmups, warmup_canonicals=self.warmrefs,
            schedule=E.evaluation_schedule(self.requests, self.p), within_block_permutations=self.p,
            physical_gpu_uuid="GPU-fixed", boundary_check=quiet, execution_commit="c" * 40)

    def test_complete_fixed_work_including_adjacent_same_arm_activation(self):
        backend = FullBackend(self.requests)
        value = self.run_final(backend, "complete")
        self.assertEqual(value["attempted_generations"], 1224)
        self.assertEqual(value["scored_generations"], 1152)
        self.assertEqual(value["warmup_generations"], 72)
        self.assertEqual(value["publications"], 36)
        self.assertEqual(len(backend.executions), 1224)
        self.assertEqual(len(backend.switches), 36)
        self.assertEqual(len(backend.weight_checks), 3)
        audit = E.audit_raw_records(self.root/"complete")
        self.assertEqual(audit["status"], "passed", audit["errors"][:3])
        self.assertEqual(len(audit["rows"]), 1224)
        summary = A.summarize(audit["rows"])
        self.assertEqual(summary["method_totals"]["N"]["verification_calls"], 256)
        report = A.write_report(self.root/"complete")
        self.assertEqual(report["execution_commit"], "c" * 40)
        self.assertEqual(len((self.root/"complete/ALL_REPEATS.csv").read_text().splitlines()), 1153)
        self.assertEqual(len((self.root/"complete/WARMUPS.csv").read_text().splitlines()), 73)
        path = self.root/"complete/0000_r0_N.json"
        raw = json.loads(path.read_text())
        raw["metadata"]["trace"][0]["draft_tokens"][0] += 1
        path.write_text(json.dumps(raw))
        self.assertEqual(E.audit_raw_records(self.root/"complete")["status"], "invalid_or_incomplete")

    def test_invalid_first_warmup_is_saved_and_stops(self):
        def mutation(record, options):
            record["response_ids"][0] += 1
        with self.assertRaisesRegex(RuntimeError, "correctness"):
            self.run_final(FullBackend(self.requests, mutation), "invalid")
        status = json.loads((self.root/"invalid/STATUS.json").read_text())
        self.assertEqual(status["invalid_generations"], 1)
        self.assertEqual(status["unknown_generation_attempts"], 0)
        self.assertEqual(status["unattempted_generations"], 1223)
        self.assertTrue((self.root/"invalid/warmup_a00_0_N.json").exists())
        self.assertEqual(E.audit_raw_records(self.root/"invalid")["status"], "invalid_or_incomplete")

    def test_unknown_generation_is_counted_without_retry(self):
        backend = FullBackend(self.requests)
        with patch.object(backend, "generate", side_effect=RuntimeError("failed before return")):
            with self.assertRaisesRegex(RuntimeError, "before return"):
                self.run_final(backend, "unknown")
        status = json.loads((self.root/"unknown/STATUS.json").read_text())
        self.assertEqual(status["unknown_generation_attempts"], 1)
        self.assertEqual(status["returned_generations"], 0)
        self.assertEqual(status["unattempted_generations"], 1223)

    def test_first_token_eos_is_valid_with_zero_calls(self):
        def mutation(record, options):
            record["response_ids"] = [151645]
            record["metadata"].update(trace=[], verification_calls=0, accepted_draft_tokens=0, committed_progress=0)
        backend = FullBackend(self.requests, mutation)
        slot = dict(phase="scored", method="N", repeat=0, event=8, panel_ordinal=0,
            activation=0, block_ordinal=0, position=0, request_id="eval0")
        record = backend.generate(self.requests[0]["prompt_ids"], request_id=E.label_for(slot),
            session_id=E.SESSION, request_index=0, purpose="evaluation", save_to=self.root/"zero.json")
        ref = dict(self.refs["eval0"], response_ids=[151645])
        row, errors = E.measure(record, self.requests[0], "/F", ref, slot, "GPU-fixed")
        self.assertEqual(errors, [])
        self.assertEqual(row["native_accepted_draft_tokens"], 0)


class StatisticalTests(unittest.TestCase):
    def test_bootstrap_direct_oracle_and_one_trajectory_count(self):
        rows = scalar_rows()
        summary = A.summarize(rows)
        paired = A.reduce_requests(rows)
        ordinary = np.asarray([r["N"]["verification_calls"] for r in paired], dtype=float)
        candidate = ordinary - 1
        indices = np.random.Generator(np.random.PCG64(1708)).integers(0, 128, (10000, 128))
        expected = ((99-candidate)[indices].sum(1)/candidate[indices].sum(1)) / ((99-ordinary)[indices].sum(1)/ordinary[indices].sum(1))-1
        self.assertEqual(summary["comparisons"]["E_vs_N"]["acceptance_improvement"]["ci95"],
            np.quantile(expected, [.025,.975], method="linear").tolist())
        self.assertEqual(summary["method_totals"]["N"]["verification_calls"], int(ordinary.sum()))
        self.assertEqual(summary["method_totals"]["N"]["request_latency_s"], ordinary.sum()+128)
        self.assertTrue(summary["decision"]["feasibility_pass"])
        self.assertEqual(summary["scope"], E.SCOPE)

    def test_missing_invalid_or_different_device_rows_are_not_dropped(self):
        for problem in ("missing", "invalid", "counts", "device", "duplicate_repeat"):
            rows = scalar_rows()
            if problem == "missing": rows.pop()
            elif problem == "invalid": rows[-1]["valid_measurement"] = False
            elif problem == "counts": rows[-1]["verification_calls"] += 1
            elif problem == "device": rows[-1]["physical_gpu_uuid"] = "GPU-other"
            else: rows[-1]["repeat"] = 0
            with self.assertRaises(ValueError, msg=problem):
                A.summarize(rows)

    def test_zero_call_requests_retained_and_equal_request_mean_denominator(self):
        rows = scalar_rows()
        for row in rows:
            if row["panel_ordinal"] == 0:
                row.update(verification_calls=0, native_accepted_draft_tokens=0, accepted_draft_tokens=0,
                    committed_progress=0, output_tokens=1, native_acceptance_histogram={})
        summary = A.summarize(rows)
        self.assertEqual(summary["request_count"], 128)
        self.assertEqual(summary["method_totals"]["N"]["zero_call_requests"], 1)
        self.assertEqual(summary["method_totals"]["N"]["equal_request_denominator"], 127)
        for row in rows:
            row.update(verification_calls=0, native_accepted_draft_tokens=0, accepted_draft_tokens=0,
                committed_progress=0, output_tokens=1, native_acceptance_histogram={})
        summary = A.summarize(rows)
        self.assertIsNone(summary["comparisons"]["E_vs_N"]["acceptance_improvement"]["point"])
        self.assertFalse(summary["decision"]["feasibility_pass"])

    def test_repeat_direction_and_strict_joint_inferiority(self):
        summary = A.summarize(scalar_rows())
        effects = deepcopy(summary["comparisons"])
        effects["E_vs_N"]["repeat_time_reductions"][2] = 0
        self.assertFalse(A.decision(effects)["feasibility_pass"])
        effects["E_vs_S"]["time_reduction"]["ci95"] = [-.2, -.1]
        effects["E_vs_S"]["acceptance_improvement"]["ci95"] = [-.1, .1]
        self.assertNotEqual(A.decision(effects)["e_vs_s"], "E_INFERIOR_TO_S")
        effects["E_vs_S"]["acceptance_improvement"]["ci95"] = [-.2, -.1]
        self.assertEqual(A.decision(effects)["e_vs_s"], "E_INFERIOR_TO_S")


if __name__ == "__main__":
    unittest.main()
