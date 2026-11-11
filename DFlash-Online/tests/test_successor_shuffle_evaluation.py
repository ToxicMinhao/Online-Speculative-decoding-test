"""Strict successor_shuffle admissions and complete raw-record audit; no model work."""

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.successor_shuffle import evaluation as E

pilot = E
from test_full_k_gsm8k_evaluation import FullBackend, requests

GPU = "GPU-successor_shuffle-card-0"


def inputs(n=64):
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
    def test_all_allowed_sizes_exact6N_coverage_reversed_blocks(self):
        for n in range(64, 257, 8):
            slots = E.evaluation_schedule(requests(n), sealed_n=n)
            self.assertEqual(len(slots), 6 * n)
            self.assertEqual(
                len({(s["panel_ordinal"], s["repeat"], s["method"]) for s in slots}),
                6 * n,
            )
            self.assertEqual(set(Counter(s["request_id"] for s in slots).values()), {6})
            orders = {}
            for s in slots:
                if s["panel_ordinal"] % 8 == 0:
                    orders.setdefault((s["block_ordinal"], s["repeat"]), []).append(
                        s["method"]
                    )
            for block in range(n // 8):
                shift = block % 3
                self.assertEqual(
                    orders[block, 0], list(E.METHODS[shift:] + E.METHODS[:shift])
                )
                self.assertEqual(orders[block, 0], orders[block, 1][::-1])
            self.assertEqual(
                set(E.schedule_balance(slots)["pair_precedence"].values()), {n // 8}
            )

    def test_bad_size_mismatch_dataset_event_rejected(self):
        for n in (None, True, 64.0, "64", 0, 63, 65, 255, 257, 264):
            with self.subTest(n=n), self.assertRaises(ValueError):
                E.evaluation_schedule(requests(64), sealed_n=n)
        for kwargs in (
            {"sealed_n": 144},
            {"sealed_n": 64, "dataset": "mathqa"},
            {"sealed_n": 64, "event": 6},
        ):
            with self.assertRaises(ValueError):
                E.evaluation_schedule(requests(64), **kwargs)

    def test_canonical_and_sealed_order_checks_remain(self):
        rows, paths, refs = inputs()
        slots = E.evaluation_schedule(rows, sealed_n=64)
        E.validate_evaluation_inputs(rows, paths, refs, sealed_n=64, schedule=slots)
        for schedule in (None, list(reversed(slots))):
            with self.assertRaises(ValueError):
                E.validate_evaluation_inputs(
                    rows, paths, refs, sealed_n=64, schedule=schedule
                )
        for change in (
            {"verified": False},
            {"prompt_ids": [999]},
            {"binding_digest": "other"},
        ):
            bad = dict(refs, eval0=dict(refs["eval0"], **change))
            with self.assertRaises(ValueError):
                E.validate_evaluation_inputs(
                    rows, paths, bad, sealed_n=64, schedule=slots
                )
        for badpaths, badrefs in (
            ({k: v for k, v in paths.items() if k != "SHUFFLED"}, refs),
            (paths, {k: v for k, v in refs.items() if k != "eval0"}),
        ):
            with self.assertRaises(ValueError):
                E.validate_evaluation_inputs(
                    rows, badpaths, badrefs, sealed_n=64, schedule=slots
                )

    def test_immediate_seed_eos_zero_calls_passes_unchanged_raw_validator(self):
        rows, paths, refs = inputs()
        refs["eval0"] = dict(refs["eval0"], response_ids=[151645])
        slots = E.evaluation_schedule(rows, sealed_n=64)
        E.validate_evaluation_inputs(rows, paths, refs, sealed_n=64, schedule=slots)

        def seed_eos(record, _options):
            record["response_ids"] = [151645]
            record["metadata"].update(
                verification_calls=0,
                accepted_draft_tokens=0,
                committed_progress=0,
                trace=[],
            )

        backend = FullBackend(rows, seed_eos)
        backend.publish(paths["K"])
        with tempfile.TemporaryDirectory() as temp:
            record = backend.generate(
                rows[0]["prompt_ids"],
                request_id="0000_r0_K",
                session_id=E.SESSION,
                request_index=0,
                purpose="evaluation",
                max_new_tokens=2048,
                save_to=Path(temp) / "0000_r0_K.json",
            )
            measured, errors = pilot.measure(
                record, rows[0], paths["K"], refs["eval0"], slots[0]
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
            pilot.repeat_errors([measured, dict(measured, repeat=1)], "K"), []
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
                E.successor_shuffle_protocol(64, gpu)


class FailureTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name) / "evaluation"
        self.rows, self.paths, self.refs = inputs()
        self.slots = E.evaluation_schedule(self.rows, sealed_n=64)

    def run_stage(self, backend=None, check=boundary):
        return E.evaluate_stage(
            backend or FullBackend(self.rows),
            self.rows,
            self.paths,
            self.root,
            canonicals=self.refs,
            sealed_n=64,
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
            (0, 384),
        )
        saved = json.loads((self.root / "SUCCESSOR_BOUNDARY_FAILURE.json").read_text())
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
            (8, 376),
        )
        saved = json.loads((self.root / "SUCCESSOR_BOUNDARY_FAILURE.json").read_text())
        self.assertEqual(saved["expected_batch"], [0, 0, 1])

    def test_boundary_exception_is_not_generation_attempt(self):
        def unavailable():
            raise OSError("hardware query failed")

        with self.assertRaises(OSError):
            self.run_stage(check=unavailable)
        saved = json.loads((self.root / "SUCCESSOR_BOUNDARY_FAILURE.json").read_text())
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
            (1, 0, 383),
        )
        self.assertTrue((self.root / "0000_r0_K.json").exists())
        self.assertEqual(
            json.loads((self.root / "SUCCESSOR_STATUS.json").read_text())[
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
            (1, 0, 383),
        )

    def test_postreturn_failure_reports_unvalidated_response_separately(self):
        with patch.object(
            pilot, "measure", side_effect=RuntimeError("postreturn failure")
        ):
            with self.assertRaises(RuntimeError):
                self.run_stage()
        s = self.status()
        work = json.loads((self.root / "SUCCESSOR_STATUS.json").read_text())
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
        cls.slots = E.evaluation_schedule(cls.rows, sealed_n=64)
        cls.backend = FullBackend(cls.rows)
        cls.result = E.evaluate_stage(
            cls.backend,
            cls.rows,
            cls.paths,
            cls.root,
            canonicals=cls.refs,
            sealed_n=64,
            schedule=cls.slots,
            boundary_check=boundary,
            physical_gpu=GPU,
        )

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def audit(self, **changes):
        values = dict(
            canonicals=self.refs, sealed_n=64, schedule=self.slots, physical_gpu=GPU
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

    def test_384_completions_exact_raw_audit_and_first8_inside_budget(self):
        audit = self.audit()
        self.assertEqual(audit["errors"], [])
        self.assertEqual(audit["status"], "passed")
        self.assertEqual(len(audit["rows"]), 384)
        self.assertEqual(self.result["completed_valid_generations"], 384)
        self.assertEqual(len(self.backend.executions), 384)
        self.assertEqual(audit["successor_shuffle"]["physical_gpu"], GPU)
        self.assertEqual(audit["successor_shuffle"]["optimizer_steps"], 0)
        gate = json.loads((self.root / "QUALIFICATION.json").read_text())
        self.assertEqual(
            (gate["status"], gate["generations"], gate["extra_generations"]),
            ("passed", 48, 0),
        )
        self.assertTrue(all(r["first_round_present"] for r in audit["rows"]))
        self.assertEqual(
            len((self.root / "BOUNDARY_CHECKS.jsonl").read_text().splitlines()), 48
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
            "successor_shuffle_boundary_check_schedule", self.audit()["errors"]
        )

    def test_wrong_external_pin_and_metadata_tamper_fail(self):
        self.assertIn(
            "physical_gpu_changed_or_unbound",
            self.audit(physical_gpu="other")["errors"],
        )
        self.replace_json(
            "SUCCESSOR_PROTOCOL.json", lambda row: row.update(sealed_n=144)
        )
        self.assertIn(
            "sealed_successor_shuffle_contract_changed", self.audit()["errors"]
        )

    def test_raw_sidecar_tamper_detected_despite_valid_flags(self):
        self.replace_json(
            "0000_r0_K_NATIVE.json", lambda row: row.update(feature_bytes=999)
        )
        self.assertIn("0000_r0_K: native_capture_changed", self.audit()["errors"])

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
            "SUCCESSOR_STATUS.json", lambda row: row.update(returned_generations=385)
        )
        self.assertIn(
            "successor_shuffle_work_accounting_changed", self.audit()["errors"]
        )


if __name__ == "__main__":
    unittest.main()
