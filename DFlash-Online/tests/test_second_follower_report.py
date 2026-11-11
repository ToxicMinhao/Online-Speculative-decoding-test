"""Independent scalar audit and report evidence boundaries; CPU only."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments.second_follower import report as R
from experiments.second_follower.evaluation import evaluation_schedule
from online_feedback.second_follower import reduce_second_follower
from online_feedback.storage import write_json
from test_second_follower import fixture, objective_terms
from test_second_follower_evaluation import inputs, statistical_rows


def event_fixture():
    proposals, labels, weights = fixture()
    losses = torch.linspace(.01, 3., labels.numel(), dtype=torch.bfloat16).reshape_as(weights)
    terms = objective_terms(losses, proposals, labels, weights)
    numerator, metrics, _ = reduce_second_follower(terms)
    metrics = {key: float(value) for key, value in metrics.items()}
    denominator = float(terms[1] + 1e-6)
    metrics.update(selected_position_weight=float(terms[1]), loss_denominator=denominator,
                   main_loss_numerator=float(numerator))
    records = []
    for ordinal in range(256):
        record_id = f"train{ordinal % 128}"
        records.append(dict(
            record_id=record_id, **{"pass": ordinal // 128},
            optimizer_stepped=ordinal % 2 == 1, optimizer_step_after=512 + (ordinal + 1) // 2,
            # Deliberately unrelated window metrics: the audit must use response scalars.
            metrics=dict(second_follower_transferred_weight=99999.),
            response_observation=dict(record_id=record_id, pass_index=ordinal // 128,
                                      loss=float(numerator / (terms[1] + 1e-6)), sum_metrics=deepcopy(metrics)),
        ))
    return dict(status="completed", start_step=512, end_step=640, records=records,
                costs=dict(training_s=12., checkpoint_save_s=2., export_s=1., event_wall_s_before_commit=16.))


class ExposureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.event = event_fixture()

    def test_independent_scalar_audit_accepts_fp32_cancellation_and_counts_once(self):
        summary, responses, depths = R.exposure_audit(self.event)
        self.assertEqual(summary["errors"], [])
        self.assertEqual(summary["status"], "passed")
        self.assertEqual((len(responses), len(depths)), (256, 4096))
        self.assertEqual((summary["active_responses"], summary["active_steps"]), (256, 128))
        moved = self.event["records"][0]["response_observation"]["sum_metrics"]["second_follower_transferred_weight"]
        self.assertAlmostEqual(summary["totals"]["transferred_weight"], 256 * moved)
        self.assertAlmostEqual(summary["transferred_over_selected"], moved / responses[0]["selected_weight"])
        self.assertAlmostEqual(summary["mean_response_transferred_over_selected"], moved / responses[0]["selected_denominator"])
        self.assertGreater(summary["checks"], 1000)

    def test_tampered_mass_denominator_and_disabled_reducer_fail(self):
        for key, value in (
            ("second_follower_depth_1_transferred_weight", 999.),
            ("second_follower_depth_1_short_weight_after", 0.),
            ("second_follower_selected_denominator", 999.),
            ("second_follower_prefix_weight_change", 1.),
            ("second_follower_disabled", 1.),
            ("main_loss_numerator", 999.),
        ):
            event = deepcopy(self.event)
            event["records"][0]["response_observation"]["sum_metrics"][key] = value
            result, _, _ = R.exposure_audit(event)
            self.assertEqual(result["status"], "invalid_or_incomplete", key)
        partial = deepcopy(self.event)
        partial["records"].pop()
        result, _, _ = R.exposure_audit(partial)
        self.assertIn("require_256_presentations", result["errors"])

    def test_missing_or_nonfinite_metrics_are_not_silent_zero_exposure(self):
        for change in (lambda m: m.pop("second_follower_depth_1_short_count"),
                       lambda m: m.update(second_follower_transferred_weight=float("nan"))):
            event = deepcopy(self.event)
            change(event["records"][0]["response_observation"]["sum_metrics"])
            result, _, _ = R.exposure_audit(event)
            self.assertEqual(result["status"], "invalid_or_incomplete")
        result, _, _ = R.exposure_audit(None)
        self.assertEqual(result["status"], "unavailable")


class ReportTests(unittest.TestCase):
    def test_qualification_requires_realized_coordinates_and_bound_case_receipts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            identities = {}
            cases = {}
            for name in ("NATIVE", "ZERO", "ACTIVE"):
                moved = .5 if name == "ACTIVE" else 0.
                value = dict(status="completed", finite=dict(status="passed"),
                             exact=dict(exact_equal=True), response_forwards=2,
                             response_backwards=2, optimizer_steps=1, next_anchor_draws=1,
                             transferred_weight=2*moved, active_intervention=moved > 0,
                             observations=[dict(sum_metrics={R.PREFIX + "transferred_weight": moved,
                                                             R.PREFIX + "disabled": int(name == "ZERO")})] * 2)
                if name != "NATIVE":
                    coordinates = dict(changed=3 if moved else 0, coordinates=8, parameters=[
                        dict(dtype="torch.bfloat16", coordinates=4, changed=1 if moved else 0),
                        dict(dtype="torch.float32", coordinates=4, changed=2 if moved else 0),
                    ])
                    value.update(gradient_changes=coordinates, weight_changes=coordinates)
                cases[name] = value
                path = root / f"qualification/{name}/RESULT.json"
                write_json(path, value)
                identities[name] = R.file_identity(path)
            write_json(root / "qualification/ADMISSION.json", dict(status="passed", cases=identities))
            exposure = dict(status="completed", response_forwards=16, response_backwards=0,
                            optimizer_steps=0, parent_weights_unchanged=dict(exact_equal=True),
                            exposure_is_parent_state_no_update=True, first_active_window=0,
                            windows=[dict(window=i, responses=cases["ACTIVE"]["observations"]) for i in range(8)])
            write_json(root / "qualification/exposure/RESULT.json", exposure)
            result = R.qualification(R.Evidence(root, root))
            self.assertEqual(result["errors"], [])
            self.assertTrue(result["positive_transfer_checked"])
            self.assertEqual(result["cases"]["ACTIVE"]["weight_changes"]["bf16_changed"], 1)
            # Missing coordinate metadata must fail even under a newly valid hash.
            cases["ACTIVE"].pop("weight_changes")
            path = root / "qualification/ACTIVE/RESULT.json"
            write_json(path, cases["ACTIVE"])
            identities["ACTIVE"] = R.file_identity(path)
            write_json(root / "qualification/ADMISSION.json", dict(status="passed", cases=identities))
            result = R.qualification(R.Evidence(root, root))
            self.assertIn("ACTIVE_weight_changes_dtype_evidence_missing", result["errors"])
            cases["ZERO"]["next_anchor_draws"] = 0
            write_json(root / "qualification/ZERO/RESULT.json", cases["ZERO"])
            result = R.qualification(R.Evidence(root, root))
            self.assertIn("ZERO_admission_identity_changed", result["errors"])
            self.assertIn("ZERO_next_anchor_check_missing", result["errors"])

    def test_integrity_recounts_raw_work_and_requires_postflight(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            limits = dict(response_forwards=256, response_backwards=256, optimizer_steps=128)
            work = dict(attempts=limits, completed=limits, unknown={key: 0 for key in limits})
            write_json(root / "PREFLIGHT.json", dict(status="passed"))
            write_json(root / "POSTFLIGHT.json", dict(status="passed"))
            ledger = root / "training/SECOND/WORK.jsonl"
            ledger.parent.mkdir(parents=True)
            rows = [dict(kind=kind, ordinal=i, phase=phase) for kind, count in limits.items()
                    for i in range(1, count+1) for phase in ("started", "completed")]
            ledger.write_text("".join(json.dumps(row) + "\n" for row in rows))
            training = dict(boundary=dict(status="passed", export_exact=True), work_ledger=work,
                            weights_identity=dict(sha256="candidate"))
            checkpoints = dict(SECOND=dict(weights_identity=training["weights_identity"]))
            cost = dict(training_status=dict(status="completed", final_step=640, work=work))
            def check():
                return R.integrity(R.Evidence(root, root), {}, training, checkpoints, cost)
            self.assertEqual(check()["errors"], [])
            ledger.write_text("".join(json.dumps(row) + "\n" for row in rows[:-1]))
            self.assertEqual(check()["status"], "invalid_or_incomplete")
            (root / "POSTFLIGHT.json").unlink()
            self.assertIn("postflight_missing_or_failed", check()["errors"])

    def test_matched_saved_summary_cannot_override_scored_rows_or_failed_phase(self):
        from experiments.second_follower.matched_start import summarize
        from test_second_follower_matched_start import analysis_rows
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows, clusters = analysis_rows(), [f"r{i}" for i in range(64)]
            saved = summarize(rows, clusters)
            for name in ("RESULT", "STATUS"):
                write_json(root / f"matched_start/{name}.json", dict(status="completed"))
            write_json(root / "matched_start/POSTFLIGHT.json", dict(status="passed"))
            write_json(root / "matched_start/ROWS.json", rows)
            write_json(root / "matched_start/SUMMARY.json", saved)
            write_json(root / "sealed/MATCHED.json", dict(request_clusters=clusters))
            plan = dict(matched_start="sealed/MATCHED.json")
            self.assertEqual(R.matched_summary(R.Evidence(root, root), plan)["status"], "complete")
            saved["census_weighted_SECOND_minus_NATIVE"]["per_later_parent_start"] += 1
            write_json(root / "matched_start/SUMMARY.json", saved)
            result = R.matched_summary(R.Evidence(root, root), plan)
            self.assertIn("matched_saved_summary_disagrees_with_reduction", result["errors"])
            write_json(root / "matched_start/POSTFLIGHT.json", dict(status="failed"))
            result = R.matched_summary(R.Evidence(root, root), plan)
            self.assertIn("matched_postflight_incomplete", result["errors"])

    def test_prior_attempt_paths_and_costs_keep_same_total_ceiling(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, prior = Path(temporary) / "current", Path(temporary) / "prior"
            root.mkdir()
            write_json(prior / "STATUS.json", dict(status="failed", phase="qualification_exposure"))
            previous = dict(files={"/remote/attempt001/STATUS.json": R.file_identity(prior / "STATUS.json")},
                            gpu_seconds=74., response_forwards=18, response_backwards=2, optimizer_steps=0)
            evidence = R.Evidence(root, "/remote/attempt002", [("/remote/attempt001", prior)])
            result = R.costs(evidence, dict(previous_attempts=[previous], prior_gpu_seconds=74.,
                                           budget=dict(gpu_seconds=4500)),
                             dict(gpu_clock_elapsed_s=174., this_attempt_gpu_seconds=100.),
                             dict(costs=dict(training_s=42.)))
            self.assertEqual(result["previous_attempts"][0]["local_identity_check"], "passed")
            self.assertEqual(result["remaining_gpu_ceiling_s"], 4326.)
            self.assertEqual(result["this_attempt_gpu_seconds"], 100.)
            self.assertEqual(result["training_event_clocks"]["training_s"], 42.)

    def test_rebound_raw_audit_uses_real_native_arm_record_and_explicit_trace_index(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            requests, paths, refs = inputs()
            plan = dict(root="/original/run", panel_manifest="/original/run/sealed/PANEL.json",
                        references="/original/run/sealed/CANONICAL.json",
                        evaluation_schedule=evaluation_schedule(requests, sealed_n=256),
                        serving_gpu_identity="GPU-fixture")
            write_json(root / "sealed/PANEL.json", dict(requests=requests))
            write_json(root / "sealed/CANONICAL.json", refs)
            raw_path = root / "evaluation/0000_r0_NATIVE.json"
            trace_path = root / "native/capture.json"
            write_json(trace_path, dict(tokens="private fixture"))
            write_json(raw_path, dict(metadata=dict(native_trace_path="/original/run/native/capture.json")))
            (root / "evaluation/MEASUREMENTS.jsonl").write_text("")
            observed = {}

            def audit(folder, panel, checkpoints, **kwargs):
                index = Path(kwargs["evidence_root"]) / "audit_inputs/NATIVE_TRACE_REFERENCES.json"
                observed.update(json.loads(index.read_text()))
                self.assertEqual(checkpoints, paths)
                self.assertEqual(Path(folder), root / "evaluation")
                return dict(status="invalid_or_incomplete", errors=["fixture incomplete"], rows=[], model_work=0)

            with patch.object(R, "audit_stage", side_effect=audit) as called:
                result, paired, _ = R.endpoint(R.Evidence(root, plan["root"]), plan,
                    {key: dict(checkpoint_path=value) for key, value in paths.items()})
            called.assert_called_once()
            self.assertFalse(result["inference_permitted"])
            self.assertEqual(paired, [])
            self.assertIn(dict(recorded_native_path="/original/run/native/capture.json",
                               local_relative_path=str(trace_path)), observed["references"])

    def test_missing_run_results_emit_all_artifacts_without_efficacy_claims(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            output = Path(temporary) / "report"
            requests, _, refs = inputs()
            source = dict(commit="execution_fixture", dirty=False)
            plan = dict(root="/original/run", execution_source=source, source_manifest="/original/run/SOURCE.json",
                        panel_manifest="/original/run/sealed/PANEL.json", references="/original/run/sealed/CANONICAL.json",
                        evaluation_schedule=evaluation_schedule(requests, sealed_n=256), serving_gpu_identity="GPU-fixture",
                        budget=dict(gpu_seconds=4500))
            write_json(root / "sealed/PLAN.json", plan)
            write_json(root / "SOURCE.json", source)
            write_json(root / "sealed/PANEL.json", dict(requests=requests))
            write_json(root / "sealed/CANONICAL.json", refs)
            write_json(root / "sealed/SEAL.json", dict(files={"/original/run/sealed/PLAN.json": R.file_identity(root / "sealed/PLAN.json")}))
            write_json(root / "STATUS.json", dict(status="failed", gpu_clock_elapsed_s=10.))
            result = R.generate(root, output)
            self.assertFalse(result["inference_permitted"])
            self.assertEqual(result["decision"]["outcome"], "invalid_or_incomplete")
            self.assertEqual(result["costs"]["remaining_gpu_ceiling_s"], 4490.)
            self.assertIsNone(result["costs"]["unattempted_main_optimizer_steps"])
            self.assertEqual({p.name for p in output.iterdir()}, {
                "REPORT.md", "SUMMARY.json", "PER_REQUEST.csv", "REPEATS.csv", "EXPOSURE.csv",
                "DEPTH_EXPOSURE.csv", "EXPOSURE_AUDIT.json", "INPUT_MANIFEST.json",
            })
            self.assertIn("execution_fixture", (output / "REPORT.md").read_text())
            self.assertIn("No endpoint benefit or speed claim", (output / "REPORT.md").read_text())
            with self.assertRaises(FileExistsError):
                R.generate(root, output)

    def test_endpoint_decision_does_not_claim_speed_for_unresolved_time(self):
        summary = R.analysis.summarize(statistical_rows())
        effect = summary["comparisons"]["SECOND_vs_NATIVE"]
        effect["call_reduction"]["ci95_pct"] = [1., 2.]
        effect["request_latency_s_reduction"]["ci95_pct"] = [-1., 2.]
        decision = R.analysis.decision(summary)
        self.assertTrue(decision["call_benefit_supported"])
        self.assertFalse(decision["latency_benefit_supported"])

    def test_completed_markdown_renders_endpoint_donor_losses_and_qualified_scope(self):
        from experiments.second_follower.matched_start import summarize
        from test_second_follower_matched_start import analysis_rows
        with tempfile.TemporaryDirectory() as temporary:
            cost = R.costs(R.Evidence(temporary, temporary), {}, {}, None)
            endpoint = R.analysis.summarize(statistical_rows())
            exposure, _, _ = R.exposure_audit(event_fixture())
            value = dict(endpoint=dict(status="passed", summary=endpoint), exposure=exposure,
                         decision=R.analysis.decision(endpoint), execution_source=dict(commit="fixture"),
                         inference_permitted=True, qualification=dict(status="passed", positive_transfer_checked=True, cases={}),
                         matched_start=summarize(analysis_rows(), [f"r{i}" for i in range(64)]),
                         costs=cost, run_status=dict(status="completed"))
            document = R.markdown(value)
            self.assertIn("SHORT donor starts", document)
            self.assertIn("single inherited seed does not establish generalization", document)
            self.assertIn("Complete request time", document)


if __name__ == "__main__":
    unittest.main()
