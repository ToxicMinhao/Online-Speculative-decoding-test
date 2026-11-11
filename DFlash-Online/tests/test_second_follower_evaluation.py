"""Strict second_follower admissions and complete raw-record audit; no model work."""

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.second_follower import evaluation as E
from experiments.second_follower import analysis as A
import numpy as np

pilot = E
from test_full_k_gsm8k_evaluation import FullBackend, requests

GPU = "GPU-second_follower-card-0"


def inputs(n=256):
    rows = requests(n)
    refs = {
        r["request_id"]: dict(
            prompt_ids=r["prompt_ids"],
            response_ids=[10, 11, 12, 13, 151645],
            finish_reason="eos",
            verified=True,
            binding_digest="a" * 64,
        )
        for r in rows
    }
    return rows, {m: "/" + m for m in E.METHODS}, refs


def boundary(gpu=GPU):
    return dict(
        physical_gpu=gpu,
        storage=dict(status="passed", free_bytes=1000, required_bytes=500),
        quiescence=dict(
            status="passed", training=[], transfers=[], foreign_gpu_processes=[]
        ),
    )


class AdmissionTests(unittest.TestCase):
    def test_exact_1024_coverage_and_reversed_blocks(self):
        slots = E.evaluation_schedule(requests(256), sealed_n=256)
        self.assertEqual(E.METHODS, ("NATIVE", "SECOND"))
        self.assertEqual(len(slots), 1024)
        self.assertEqual(len({(s["panel_ordinal"], s["repeat"], s["method"]) for s in slots}), 1024)
        self.assertEqual(set(Counter(s["request_id"] for s in slots).values()), {4})
        orders = {}
        for s in slots:
            if s["panel_ordinal"] % 8 == 0:
                orders.setdefault((s["block_ordinal"], s["repeat"]), []).append(s["method"])
        for block in range(32):
            shift = block % 2
            self.assertEqual(orders[block, 0], list(E.METHODS[shift:] + E.METHODS[:shift]))
            self.assertEqual(orders[block, 0], orders[block, 1][::-1])
        balance = E.schedule_balance(slots)
        self.assertEqual(balance["positions"], {m: [32, 32] for m in E.METHODS})
        self.assertEqual(set(balance["pair_precedence"].values()), {32})

    def test_bad_size_mismatch_dataset_event_rejected(self):
        for n in (None, True, 256.0, "256", 0, 8, 64, 248, 255, 257, 264):
            with self.subTest(n=n), self.assertRaises(ValueError):
                E.evaluation_schedule(requests(256), sealed_n=n)
        for kwargs in ({"sealed_n": 144}, {"sealed_n": 256, "dataset": "mathqa"}, {"sealed_n": 256, "event": 6}):
            with self.assertRaises(ValueError):
                E.evaluation_schedule(requests(256), **kwargs)
        with self.assertRaises(ValueError):
            E.evaluation_schedule(requests(248), sealed_n=256)

    def test_canonical_and_sealed_order_checks_remain(self):
        rows, paths, refs = inputs()
        slots = E.evaluation_schedule(rows, sealed_n=256)
        E.validate_evaluation_inputs(rows, paths, refs, sealed_n=256, schedule=slots)
        for schedule in (None, list(reversed(slots))):
            with self.assertRaises(ValueError):
                E.validate_evaluation_inputs(
                    rows, paths, refs, sealed_n=256, schedule=schedule
                )
        for change in (
            {"verified": False},
            {"prompt_ids": [999]},
            {"binding_digest": "other"},
        ):
            bad = dict(refs, eval0=dict(refs["eval0"], **change))
            with self.assertRaises(ValueError):
                E.validate_evaluation_inputs(
                    rows, paths, bad, sealed_n=256, schedule=slots
                )
        for badpaths, badrefs in (
            ({k: v for k, v in paths.items() if k != "SECOND"}, refs),
            (paths, {k: v for k, v in refs.items() if k != "eval0"}),
        ):
            with self.assertRaises(ValueError):
                E.validate_evaluation_inputs(
                    rows, badpaths, badrefs, sealed_n=256, schedule=slots
                )

    def test_immediate_seed_eos_zero_calls_passes_unchanged_raw_validator(self):
        rows, paths, refs = inputs()
        refs["eval0"] = dict(refs["eval0"], response_ids=[151645])
        slots = E.evaluation_schedule(rows, sealed_n=256)
        E.validate_evaluation_inputs(rows, paths, refs, sealed_n=256, schedule=slots)

        def seed_eos(record, _options):
            record["response_ids"] = [151645]
            record["metadata"].update(
                verification_calls=0,
                accepted_draft_tokens=0,
                committed_progress=0,
                trace=[],
            )

        backend = FullBackend(rows, seed_eos)
        backend.publish(paths["NATIVE"])
        with tempfile.TemporaryDirectory() as temp:
            record = backend.generate(
                rows[0]["prompt_ids"],
                request_id="0000_r0_NATIVE",
                session_id=E.SESSION,
                request_index=0,
                purpose="evaluation",
                max_new_tokens=2048,
                save_to=Path(temp) / "0000_r0_NATIVE.json",
            )
            measured, errors = pilot.measure(
                record, rows[0], paths["NATIVE"], refs["eval0"], slots[0]
            )
        self.assertEqual(errors, [])
        self.assertEqual(
            (
                measured["verification_calls"],
                measured["committed_progress"],
                measured["output_tokens"],
            ),
            (0, 0, 1),
        )
        self.assertFalse(measured["first_round_present"])
        self.assertEqual(
            pilot.repeat_errors([measured, dict(measured, repeat=1)], "NATIVE"), []
        )

    def test_boundary_storage_quiet_host_and_physical_pin(self):
        self.assertEqual(E.boundary_errors(boundary(), GPU), [])
        for check in (
            {},
            dict(boundary(), physical_gpu="other"),
            dict(boundary(), physical_gpu=None),
            dict(
                boundary(),
                storage=dict(status="passed", free_bytes=1, required_bytes=500),
            ),
            dict(
                boundary(),
                quiescence=dict(
                    status="passed",
                    training=[],
                    transfers=["busy"],
                    foreign_gpu_processes=[],
                ),
            ),
        ):
            self.assertTrue(E.boundary_errors(check, GPU))
        for gpu in (None, "", " card", 1):
            with self.assertRaises(ValueError):
                E.second_follower_protocol(256, gpu)


class FailureTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name) / "evaluation"
        self.rows, self.paths, self.refs = inputs()
        self.slots = E.evaluation_schedule(self.rows, sealed_n=256)

    def run_stage(self, backend=None, check=boundary):
        return E.evaluate_stage(
            backend or FullBackend(self.rows),
            self.rows,
            self.paths,
            self.root,
            canonicals=self.refs,
            sealed_n=256,
            schedule=self.slots,
            boundary_check=check,
            physical_gpu=GPU,
        )

    def status(self):
        return json.loads((self.root / "STATUS.json").read_text())

    def test_no_boundary_callback_stops_before_native(self):
        backend = FullBackend(self.rows)
        with self.assertRaisesRegex(ValueError, "callable"):
            self.run_stage(backend, None)
        self.assertEqual(backend.executions, [])
        self.assertFalse(self.root.exists())

    def test_first_wrong_gpu_preserves_rejected_boundary_zero_work(self):
        backend = FullBackend(self.rows)
        with self.assertRaisesRegex(RuntimeError, "physical_gpu"):
            self.run_stage(backend, lambda: boundary("other"))
        self.assertEqual(backend.executions, [])
        self.assertEqual(
            (
                self.status()["attempted_generations"],
                self.status()["unattempted_generations"],
            ),
            (0, 1024),
        )
        saved = json.loads((self.root / "SECOND_FOLLOWER_BOUNDARY_FAILURE.json").read_text())
        self.assertEqual(saved["observed"]["physical_gpu"], "other")
        self.assertEqual(saved["expected_batch"], [0, 0, 0])

    def test_later_gpu_change_stops_before_next_block(self):
        calls = []
        backend = FullBackend(self.rows)

        def switch():
            calls.append(1)
            return boundary(GPU if len(calls) == 1 else "other")

        with self.assertRaisesRegex(RuntimeError, "physical_gpu"):
            self.run_stage(backend, switch)
        self.assertEqual(len(backend.executions), 8)
        self.assertEqual(
            (
                self.status()["completed_valid_generations"],
                self.status()["unattempted_generations"],
            ),
            (8, 1016),
        )
        saved = json.loads((self.root / "SECOND_FOLLOWER_BOUNDARY_FAILURE.json").read_text())
        self.assertEqual(saved["expected_batch"], [0, 0, 1])

    def test_boundary_exception_is_not_generation_attempt(self):
        def unavailable():
            raise OSError("hardware query failed")

        with self.assertRaises(OSError):
            self.run_stage(check=unavailable)
        saved = json.loads((self.root / "SECOND_FOLLOWER_BOUNDARY_FAILURE.json").read_text())
        self.assertIsNone(saved["observed"])
        self.assertIn("hardware query failed", saved["error"])
        self.assertEqual(self.status()["attempted_generations"], 0)

    def test_invalid_output_keeps_raw_invalid_unattempted_and_no_retry(self):
        def mutate(record, _options):
            record["response_ids"][0] += 1

        with self.assertRaises(RuntimeError):
            self.run_stage(FullBackend(self.rows, mutate))
        s = self.status()
        self.assertEqual(
            (
                s["invalid_generations"],
                s["unknown_generation_attempts"],
                s["unattempted_generations"],
            ),
            (1, 0, 1023),
        )
        self.assertTrue((self.root / "0000_r0_NATIVE.json").exists())
        self.assertEqual(
            json.loads((self.root / "SECOND_FOLLOWER_STATUS.json").read_text())[
                "returned_without_measurement"
            ],
            0,
        )
        with self.assertRaises(FileExistsError):
            self.run_stage()

    def test_unknown_native_attempt_is_not_retried(self):
        backend = FullBackend(self.rows)
        with patch.object(
            backend, "generate", side_effect=RuntimeError("native transport")
        ):
            with self.assertRaises(RuntimeError):
                self.run_stage(backend)
        s = self.status()
        self.assertEqual(
            (
                s["unknown_generation_attempts"],
                s["returned_generations"],
                s["unattempted_generations"],
            ),
            (1, 0, 1023),
        )

    def test_postreturn_failure_reports_unvalidated_response_separately(self):
        with patch.object(
            pilot, "measure", side_effect=RuntimeError("postreturn failure")
        ):
            with self.assertRaises(RuntimeError):
                self.run_stage()
        s = self.status()
        work = json.loads((self.root / "SECOND_FOLLOWER_STATUS.json").read_text())
        self.assertEqual(
            (s["returned_generations"], s["unknown_generation_attempts"]), (1, 0)
        )
        self.assertEqual(
            (work["returned_without_measurement"], work["persisted_measurements"]),
            (1, 0),
        )


class CompleteAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name) / "evaluation"
        cls.rows, cls.paths, cls.refs = inputs()
        cls.slots = E.evaluation_schedule(cls.rows, sealed_n=256)
        cls.backend = FullBackend(cls.rows)
        cls.result = E.evaluate_stage(
            cls.backend,
            cls.rows,
            cls.paths,
            cls.root,
            canonicals=cls.refs,
            sealed_n=256,
            schedule=cls.slots,
            boundary_check=boundary,
            physical_gpu=GPU,
        )

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def audit(self, **changes):
        values = dict(
            canonicals=self.refs, sealed_n=256, schedule=self.slots, physical_gpu=GPU
        )
        values.update(changes)
        return E.audit_stage(self.root, self.rows, self.paths, **values)

    def replace_json(self, name, mutate):
        path = self.root / name
        saved = path.read_bytes()
        self.addCleanup(path.write_bytes, saved)
        value = json.loads(saved)
        mutate(value)
        path.write_text(json.dumps(value))

    def replace_boundaries(self, mutate):
        path = self.root / "BOUNDARY_CHECKS.jsonl"
        saved = path.read_bytes()
        self.addCleanup(path.write_bytes, saved)
        rows = [json.loads(line) for line in saved.splitlines()]
        mutate(rows)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def test_1024_completions_exact_raw_audit_and_first8_inside_budget(self):
        audit = self.audit()
        self.assertEqual(audit["errors"], [])
        self.assertEqual(audit["status"], "passed")
        self.assertEqual(len(audit["rows"]), 1024)
        self.assertEqual(self.result["completed_valid_generations"], 1024)
        self.assertEqual(len(self.backend.executions), 1024)
        self.assertEqual(audit["second_follower"]["physical_gpu"], GPU)
        self.assertEqual(audit["second_follower"]["optimizer_steps"], 0)
        gate = json.loads((self.root / "QUALIFICATION.json").read_text())
        self.assertEqual(
            (gate["status"], gate["generations"], gate["extra_generations"]),
            ("passed", 32, 0),
        )
        self.assertTrue(all(r["first_round_present"] for r in audit["rows"]))
        self.assertEqual(
            len((self.root / "BOUNDARY_CHECKS.jsonl").read_text().splitlines()), 128
        )

    def test_missing_boundary_file_cannot_bypass_engine(self):
        path = self.root / "BOUNDARY_CHECKS.jsonl"
        saved = path.read_bytes()
        path.unlink()
        self.addCleanup(path.write_bytes, saved)
        self.assertEqual(self.audit()["status"], "invalid_or_incomplete")

    def test_changed_physical_card_fails_even_all_raw_rows_valid(self):
        self.replace_boundaries(
            lambda rows: rows[1]["check"].update(physical_gpu="other")
        )
        self.assertIn("physical_gpu_changed_or_unbound", self.audit()["errors"])

    def test_removed_boundary_slot_fails_audit(self):
        self.replace_boundaries(lambda rows: rows.pop())
        self.assertIn(
            "second_follower_boundary_check_schedule", self.audit()["errors"]
        )

    def test_wrong_external_pin_and_metadata_tamper_fail(self):
        self.assertIn(
            "physical_gpu_changed_or_unbound",
            self.audit(physical_gpu="other")["errors"],
        )
        self.replace_json(
            "SECOND_FOLLOWER_PROTOCOL.json", lambda row: row.update(sealed_n=144)
        )
        self.assertIn(
            "sealed_second_follower_contract_changed", self.audit()["errors"]
        )

    def test_raw_sidecar_tamper_detected_despite_valid_flags(self):
        self.replace_json(
            "0000_r0_NATIVE_NATIVE.json", lambda row: row.update(feature_bytes=999)
        )
        self.assertIn("0000_r0_NATIVE: native_capture_changed", self.audit()["errors"])

    def test_missing_canonical_or_reordered_schedule_is_invalid_no_modelwork(self):
        refs = dict(self.refs)
        refs.pop("eval0")
        audit = self.audit(canonicals=refs)
        self.assertEqual(audit["status"], "invalid_or_incomplete")
        self.assertEqual(audit["model_work"], 0)
        self.assertEqual(
            self.audit(schedule=list(reversed(self.slots)))["status"],
            "invalid_or_incomplete",
        )

    def test_supplemental_work_receipt_bound_to_saved_rows(self):
        self.replace_json(
            "SECOND_FOLLOWER_STATUS.json", lambda row: row.update(returned_generations=1025)
        )
        self.assertIn(
            "second_follower_work_accounting_changed", self.audit()["errors"]
        )


def statistical_rows():
    """Unequal request baselines and repeats distinguish the prescribed estimator."""
    rows = []
    for ordinal in range(256):
        native_calls = 4 + ordinal % 7
        saved = (1, -1, 0)[ordinal % 3]
        for method in E.METHODS:
            calls = native_calls - saved if method == "SECOND" else native_calls
            output = 2048 if ordinal == 255 else 21
            for repeat in (0, 1):
                native_time = 1 + ordinal / 80 + repeat / 3
                time_saved = .03 * saved + (.2 if repeat == 0 else -.1)
                elapsed = native_time - time_saved if method == "SECOND" else native_time
                rows.append(dict(
                    phase="final", dataset="gsm8k", event=5, seed=17,
                    method=method, panel_ordinal=ordinal, request_id=f"eval{ordinal}",
                    repeat=repeat, checkpoint_id="/" + method,
                    verification_calls=calls, output_tokens=output,
                    committed_progress=output - 1,
                    accepted_draft_tokens=output - 1 - calls,
                    native_accepted_draft_tokens=output - 1 - calls,
                    request_latency_s=elapsed, native_generate_s=elapsed - .1,
                    cache_maintenance_s=.1, cap_hit=ordinal == 255,
                    finish_reason="length" if ordinal == 255 else "eos",
                    first_round_present=True,
                    validation_status="valid", valid_measurement=True,
                    errors=[], structural_errors=[],
                ))
    return rows


class AnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = statistical_rows()
        cls.paired = A.reduce_stage(cls.rows)
        cls.summary = A.summarize(cls.rows)

    def test_counts_once_and_times_mean_before_ratio_of_sums(self):
        totals = self.summary["method_totals"]
        for method in E.METHODS:
            self.assertEqual(totals[method]["verification_calls"], sum(row["verification_calls"] for row in self.rows if row["method"] == method and row["repeat"] == 0))
            self.assertAlmostEqual(totals[method]["request_latency_s"], sum(row["request_latency_s"] for row in self.rows if row["method"] == method) / 2)
            self.assertEqual(totals[method]["cap_hit_requests"], 1)
        effect = self.summary["comparisons"][A.CONTRAST]
        native = totals["NATIVE"]["verification_calls"]
        second = totals["SECOND"]["verification_calls"]
        self.assertEqual(effect["call_reduction"]["point_pct"], 100 * (native - second) / native)
        self.assertEqual((effect["request_wins"], effect["request_regressions"], effect["request_ties"]), (86, 85, 85))
        self.assertEqual((effect["gross_calls_saved"], effect["gross_calls_added"], effect["net_calls_saved"]), (86, 85, 1))
        flat = A.request_table(self.paired)
        self.assertEqual(len(flat), 256)
        self.assertEqual([row["call_outcome"] for row in flat[:3]], ["improved", "worse", "tied"])
        sweeps = self.summary["sweep_reductions"]
        self.assertGreater(sweeps["0"]["request_latency_s_reduction_pct"], 0)
        self.assertLess(sweeps["1"]["request_latency_s_reduction_pct"], 0)

    def test_independent_paired_pcg64_linear_percentile_oracle(self):
        indices = np.random.Generator(np.random.PCG64(20261113)).integers(0, 256, size=(10000, 256))
        effect = self.summary["comparisons"][A.CONTRAST]
        for field in ("verification_calls", *A.TIMINGS):
            native = np.array([row["NATIVE"][field] for row in self.paired])
            second = np.array([row["SECOND"][field] for row in self.paired])
            draws = 100 * (native[indices].sum(axis=1) - second[indices].sum(axis=1)) / native[indices].sum(axis=1)
            expected = np.percentile(draws, [2.5, 97.5], method="linear")
            result = effect["call_reduction" if field == "verification_calls" else field + "_reduction"]
            np.testing.assert_allclose(result["ci95_pct"], expected, rtol=0, atol=1e-12)
            self.assertEqual(result["undefined_bootstrap_draws"], 0)
        self.assertEqual(self.summary["bootstrap"]["seed"], 20261113)
        self.assertEqual(self.summary["bootstrap"]["resamples"], 10000)

    def test_incomplete_duplicate_invalid_and_changed_checkpoint_refuse_inference(self):
        bad_sets = [self.rows[:-1], self.rows + [self.rows[0]]]
        for changes in (
            dict(repeat=1), dict(valid_measurement=False), dict(errors=["bad"]),
            dict(structural_errors=["bad"]), dict(checkpoint_id="/changed"),
            dict(verification_calls=7), dict(event=4), dict(request_latency_s=float("nan")),
            dict(panel_ordinal=True), dict(cap_hit=0),
        ):
            changed = deepcopy(self.rows)
            changed[0].update(changes)
            bad_sets.append(changed)
        for rows in bad_sets:
            with self.subTest(first=rows[0]), self.assertRaises(ValueError):
                A.reduce_stage(rows)
            result, paired = A.endpoint_from_audit(dict(status="passed", errors=[], model_work=0, rows=rows))
            self.assertFalse(result["inference_permitted"])
            self.assertEqual(paired, [])
        result, _ = A.endpoint_from_audit(dict(status="invalid_or_incomplete", errors=["raw tamper"], model_work=0, rows=self.rows))
        self.assertFalse(result["inference_permitted"])

    def test_seed_eos_kept_and_zero_cache_component_is_only_descriptive(self):
        rows = deepcopy(self.rows)
        for row in rows:
            row["cache_maintenance_s"] = 0.
            if row["panel_ordinal"] == 0:
                row.update(verification_calls=0, output_tokens=1, committed_progress=0,
                           accepted_draft_tokens=0, native_accepted_draft_tokens=0,
                           finish_reason="eos", first_round_present=False)
        result, paired = A.endpoint_from_audit(dict(status="passed", errors=[], model_work=0, rows=rows))
        self.assertTrue(result["inference_permitted"])
        self.assertEqual(len(paired), 256)
        effect = result["summary"]["comparisons"][A.CONTRAST]
        self.assertEqual(effect["cache_maintenance_s_reduction"]["undefined_bootstrap_draws"], 10000)
        self.assertIsNone(effect["cache_maintenance_s_reduction"]["ci95_pct"])
        rows[0]["first_round_present"] = True
        with self.assertRaises(ValueError):
            A.reduce_stage(rows)

    def test_undefined_primary_draws_do_not_support_claims(self):
        rows = deepcopy(self.rows)
        for row in rows:
            if row["panel_ordinal"] != 0:
                row.update(verification_calls=0, output_tokens=1, committed_progress=0,
                           accepted_draft_tokens=0, native_accepted_draft_tokens=0,
                           finish_reason="eos", first_round_present=False, cap_hit=False)
        result, paired = A.endpoint_from_audit(dict(status="passed", errors=[], model_work=0, rows=rows))
        self.assertEqual(result["status"], "undefined_primary_bootstrap")
        self.assertFalse(result["inference_permitted"])
        self.assertEqual(len(paired), 256)
        self.assertGreater(result["summary"]["comparisons"][A.CONTRAST]["call_reduction"]["undefined_bootstrap_draws"], 0)
        self.assertFalse(A.decision(result["summary"])["call_benefit_supported"])

    def test_fixed_decision_preserves_call_gain_but_stops_latency_regression(self):
        summary = deepcopy(self.summary)
        effect = summary["comparisons"][A.CONTRAST]
        effect["call_reduction"]["ci95_pct"] = [1., 2.]
        effect["request_latency_s_reduction"]["ci95_pct"] = [-2., -1.]
        result = A.decision(summary)
        self.assertEqual(result["outcome"], "call_benefit_supported")
        self.assertTrue(result["stop_serving_promotion"])
        self.assertTrue(result["latency_regression_supported"])
        effect["request_latency_s_reduction"]["ci95_pct"] = [1., 2.]
        self.assertFalse(A.decision(summary)["latency_benefit_supported"])
        summary["sweep_reductions"]["1"]["request_latency_s_reduction_pct"] = 1.
        self.assertTrue(A.decision(summary)["latency_benefit_supported"])
        effect["call_reduction"]["ci95_pct"] = [0., 2.]
        self.assertEqual(A.decision(summary)["outcome"], "call_difference_unresolved")
        effect["call_reduction"]["ci95_pct"] = [-2., 0.]
        self.assertEqual(A.decision(summary)["outcome"], "call_difference_unresolved")
        effect["call_reduction"]["ci95_pct"] = [-2., -1.]
        self.assertEqual(A.decision(summary)["outcome"], "call_harm_supported")
        self.assertFalse(A.decision(summary)["automatic_followup_authorized"])


if __name__ == "__main__":
    unittest.main()
