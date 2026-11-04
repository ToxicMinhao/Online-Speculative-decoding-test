"""CPU evidence tests; exercise budget, reference isolation, and paired analysis."""

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from experiments.vfd_second_pass import data as D, evaluation as E, statistics as S
from test_full_k_gsm8k_evaluation import FullBackend, requests
from test_kplus_full_evaluation import RunRootNativeBackend


def inputs(count):
    result = requests(count)
    for row in result:
        row["stop_token_ids"] = [151645, 151643]
    return result


def canonicals(panel):
    return {
        r["request_id"]: dict(
            prompt_ids=r["prompt_ids"],
            response_ids=[10, 11, 12, 13, 151645],
            finish_reason="eos",
            verified=True,
            binding_digest="test-binding",
        )
        for r in panel
    }


def measures():
    result = []
    scales = dict(FULL=1.1, ONCE=1.05, RANDOM64=1.02, STATIC64=1.0, PROGRESS64=0.9)
    for ordinal in range(200):
        for repeat in range(2):
            for method in S.METHODS:
                latency = scales[method] * (2 + ordinal % 11) + repeat * 0.2
                result.append(
                    dict(
                        seed=17,
                        event=4,
                        phase="final",
                        panel_ordinal=ordinal,
                        request_id=f"req:{ordinal}",
                        repeat=repeat,
                        method=method,
                        verification_calls=10 + ordinal % 7 + (method == "PROGRESS64"),
                        accepted_draft_tokens=50,
                        committed_progress=99,
                        output_tokens=100,
                        checkpoint_id="/" + method,
                        finish_reason="length" if ordinal == 0 else "eos",
                        cap_hit=ordinal == 0,
                        valid_measurement=True,
                        validation_status="valid",
                        errors=[],
                        request_latency_s=latency,
                        native_generate_s=latency - 0.05,
                        cache_maintenance_s=0.02,
                    )
                )
    return result


class EndpointTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.panel = inputs(3)
        self.canonical = canonicals(self.panel)
        self.checkpoints = {m: "/" + m for m in E.METHODS}

    def test_exact_two_sweeps_and_reverse_corresponding_block_order(self):
        schedule = E.final_schedule(inputs(200))
        self.assertEqual(len(schedule), 2000)
        self.assertEqual(
            len({(r["request_id"], r["method"], r["repeat"]) for r in schedule}), 2000
        )
        positions = Counter()
        for block in range(10):
            orders = []
            for repeat in range(2):
                rows = [
                    r
                    for r in schedule
                    if r["block_ordinal"] == block and r["repeat"] == repeat
                ]
                orders.append([r["method"] for r in rows[::20]])
                self.assertEqual(
                    [r["panel_ordinal"] for r in rows[:20]],
                    list(range(block * 20, block * 20 + 20)),
                )
                positions.update(
                    (method, index) for index, method in enumerate(orders[-1])
                )
            self.assertEqual(orders[1], list(reversed(orders[0])))
        self.assertEqual(set(positions.values()), {4})

    def test_native_roundtrip_five_real_exports_and_relocated_sidecars(self):
        backend = RunRootNativeBackend(self.panel, self.root)
        folder = self.root / "campaign_001/evaluation/final"
        status = E.evaluate(
            backend,
            self.panel,
            self.checkpoints,
            folder,
            canonicals=self.canonical,
            production=False,
        )
        audit = E.audit_stage(
            folder,
            self.panel,
            self.checkpoints,
            canonicals=self.canonical,
            evidence_root=self.root,
            production=False,
        )
        self.assertEqual(status["completed_valid_generations"], 30)
        self.assertEqual(audit["errors"], [])
        self.assertEqual(
            set(x[0] for x in backend.executions), set(self.checkpoints.values())
        )
        backend.native_paths[0].write_text("{}")
        audit = E.audit_stage(
            folder,
            self.panel,
            self.checkpoints,
            canonicals=self.canonical,
            evidence_root=self.root,
            production=False,
        )
        self.assertNotEqual(audit["errors"], [])

    def test_parity_failure_stops_and_preserves_record(self):
        def mutate(record, options):
            if record["checkpoint_id"] == "/ONCE":
                record["response_ids"][0] = 99

        folder = self.root / "failed"
        with self.assertRaisesRegex(RuntimeError, "invalid"):
            E.evaluate(
                FullBackend(self.panel, mutate),
                self.panel,
                self.checkpoints,
                folder,
                canonicals=self.canonical,
                production=False,
            )
        status = json.loads((folder / "STATUS.json").read_text())
        self.assertEqual(status["returned_generations"], 4)
        self.assertEqual(status["invalid_generations"], 1)
        self.assertTrue((folder / "0000_r0_ONCE.json").exists())

    def test_reference_once_only_and_binding_required(self):
        backend = FullBackend(self.panel)
        result = E.prepare_canonicals(
            backend,
            self.panel,
            self.root / "references",
            canonicals={
                self.panel[0]["request_id"]: self.canonical[self.panel[0]["request_id"]]
            },
            binding_digest="test-binding",
        )
        self.assertEqual(len(backend.executions), 2)
        self.assertEqual(set(result), set(self.canonical))
        E.validate_inputs(self.panel, self.checkpoints, result, production=False)
        result[self.panel[0]["request_id"]]["prompt_ids"] = [99]
        with self.assertRaises(ValueError):
            E.validate_inputs(self.panel, self.checkpoints, result, production=False)

    def test_pending_reference_extra_arm_or_production_short_panel_fail(self):
        for cp, canonical, production in [
            (self.checkpoints, {}, False),
            ({**self.checkpoints, "FAKE": "/fake"}, self.canonical, False),
            (self.checkpoints, self.canonical, True),
        ]:
            with self.assertRaises(ValueError):
                E.validate_inputs(self.panel, cp, canonical, production=production)


class StatisticsTests(unittest.TestCase):
    def test_shared_paired_draws_and_two_repeat_means(self):
        with patch.object(S, "RESAMPLES", 80):
            result = S.summarize(measures())
        basis = 2 + np.arange(200) % 11
        indices = np.random.Generator(np.random.PCG64(20261104)).integers(
            0, 200, (80, 200)
        )
        for method, scale in [
            ("FULL", 1.1),
            ("STATIC64", 1.0),
            ("RANDOM64", 1.02),
            ("ONCE", 1.05),
        ]:
            draws = 100 * (
                1
                - (0.9 * basis + 0.1)[indices].sum(1)
                / (scale * basis + 0.1)[indices].sum(1)
            )
            actual = result["comparisons"]["PROGRESS64_vs_" + method][
                "request_latency_s_reduction"
            ]
            np.testing.assert_allclose(
                actual["ci95_pct"], np.quantile(draws, [0.025, 0.975]), rtol=1e-12
            )
        self.assertEqual(result["request_count"], 200)
        self.assertEqual(result["execution_count"], 2000)
        self.assertAlmostEqual(
            result["method_totals"]["STATIC64"]["request_latency_s"], basis.sum() + 20
        )
        self.assertLess(
            result["comparisons"]["PROGRESS64_vs_FULL"]["call_reduction"]["point_pct"],
            0,
        )

    def test_incomplete_duplicate_inconsistent_and_nonfinite_rejected(self):
        original = measures()
        for changed in [original[:-1], original + [original[0]]]:
            with self.assertRaises(ValueError):
                S.reduce_final(changed)
        for field, value in [
            ("verification_calls", 999),
            ("request_latency_s", float("nan")),
            ("finish_reason", "other"),
            ("checkpoint_id", "/changed"),
            ("valid_measurement", False),
        ]:
            changed = deepcopy(original)
            changed[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                S.reduce_final(changed)

    def test_decision_boundaries_simpler_preference_and_crossing_zero_caution(self):
        def effect(point, low, high):
            return {
                "request_latency_s_reduction": dict(
                    point_pct=point, ci95_pct=[low, high]
                )
            }

        effects = {
            "PROGRESS64_vs_" + m: effect(1.0, 0.1, 2.0)
            for m in ("FULL", "STATIC64", "RANDOM64", "ONCE")
        }
        effects.update(
            {m + "_vs_FULL": effect(0.2, -0.4, 0.8) for m in ("ONCE", "RANDOM64")}
        )
        costs = dict(
            FULL=100.0, ONCE=50.0, RANDOM64=75.0, STATIC64=80.0, PROGRESS64=85.0
        )
        effects["PROGRESS64_vs_STATIC64"] = effect(0.1, -0.1, 0.3)
        result = S.interpret(effects, costs)
        self.assertEqual(result["decision"], "advance_to_longer_in_domain_run")
        self.assertFalse(result["scoring_mechanism_established_against_both_selectors"])
        effects["PROGRESS64_vs_FULL"] = effect(0.1, -0.49, 0.8)
        self.assertEqual(S.interpret(effects, costs)["preferred_policy"], "ONCE")
        effects["PROGRESS64_vs_FULL"] = effect(0.1, -0.5, 0.8)
        self.assertEqual(S.interpret(effects, costs)["decision"], "otherwise")
        effects["PROGRESS64_vs_FULL"] = effect(1.0, 0.1, 2.0)
        effects["PROGRESS64_vs_ONCE"] = effect(-1.0, -2.0, -0.1)
        self.assertEqual(S.interpret(effects, costs)["decision"], "efficiency_only")


class DataTests(unittest.TestCase):
    def test_unused_first_then_exact_hash_and_train_text_disjoint(self):
        rows = []
        for i in range(5):
            stable = f"gsm8k:test:{i}"
            rows.append(
                dict(
                    source_row=i,
                    stable_id=stable,
                    question="Q" + str(i),
                    prompt_ids=[i + 10],
                    prompt_ids_sha256=D.token_digest([i + 10]),
                    selection_key=hashlib.sha256(
                        ("vfd-s2-v1|eval|" + stable).encode()
                    ).hexdigest(),
                )
            )
        train = [dict(dataset_id="train", question=" Ｑ０ ", prompt_ids=[99])]
        result = D.select_panel(rows, train, {1: []}, count=3)
        self.assertNotIn("gsm8k:test:0", [r["dataset_id"] for r in result["requests"]])
        self.assertTrue(all(not r["prior_project_use"] for r in result["requests"]))
        self.assertEqual(
            [r["source_row"] for r in result["requests"]],
            [
                r["source_row"]
                for r in sorted(rows[2:], key=lambda r: r["selection_key"])
            ],
        )
        self.assertEqual(result["freshness_claim"], "train-disjoint")

    def test_history_ignores_answer_and_catalog_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "panel.json"
            path.write_text(
                json.dumps(
                    dict(
                        question="old",
                        answer="new",
                        sources=[dict(question="new")],
                        messages=[dict(role="assistant", content="new")],
                    )
                )
            )
            rows = [
                dict(
                    source_row=0,
                    question="old",
                    user_message="old",
                    rendered_prompt="old",
                    prompt_ids_sha256="x",
                ),
                dict(
                    source_row=1,
                    question="new",
                    user_message="new",
                    rendered_prompt="new",
                    prompt_ids_sha256="y",
                ),
            ]
            overlap, receipt = D.prior_use([path], rows)
            self.assertEqual(set(overlap), {0})
            self.assertEqual(receipt["unreadable_or_unavailable"], [])


if __name__ == "__main__":
    unittest.main()
