"""CPU integration checks for admission, immutable inputs and orchestration.

Native model work is replaced by explicit process receipts. These tests do not
qualify GPU numerics, resource fit or output preservation.
"""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import unittest
from unittest.mock import patch

from online_feedback.storage import write_json
from experiments.recent_repair_retention import prepare_run
from experiments.recent_repair_retention.analysis import ANALYSIS
from experiments.recent_repair_retention.panel import select_panel
from experiments.survival_confirmation.data import digest, REVISION, STOP_IDS
from experiments.verification_transition.data import identity, read

try:
    from experiments.recent_repair_retention import runtime, fixed_starts
except ModuleNotFoundError as error:
    if error.name not in ("torch", "pydantic", "safetensors"):
        raise
    runtime = fixed_starts = None


def panel_fixture(count=3):
    rows = [dict(request_id=f"r{i}", source_id=f"gsm8k:main:test:{i}",
        dataset_id=f"gsm8k:main:test:{i}", source_row=i, source_revision=REVISION,
        prompt_ids=[i + 10], prompt_ids_sha256=digest([i + 10]),
        normalized_question_sha256=f"question-{i}", rendered_prompt_sha256=f"render-{i}",
        stop_token_ids=list(STOP_IDS), max_new_tokens=2048, collect_feedback=False)
        for i in range(count)]
    return select_panel(rows, [], {})


class PreparationIntegrationTests(unittest.TestCase):
    def fixture(self, root):
        task, prior = root / "task", root / "prior"
        origin = "/data/sources/minhao/parent/training_state.pt"
        feature_path = "/data/sources/minhao/runs/dflash_lost_repair_20261114/execution_001/campaign_001/census/FEATURES.json"
        contract = dict(serving=dict(block_size=16, max_new_tokens=2048),
            native_runtime_sha256={"models/dflash.py": "native-hash"},
            python_version="3.12.14", torch="2.9.1+cu130", transformers="4.57.1",
            target_files={"model.safetensors": dict(bytes=4, sha256="target-hash")})
        panel = panel_fixture()
        requests = panel["requests"]
        references = {row["request_id"]: dict(prompt_ids=row["prompt_ids"], response_ids=[5],
            finish_reason="eos", verified=True) for row in requests[:-1]}
        inputs = dict(panel=panel, references=references, canonical_reuse_audit={"verified": 2},
            source_files={}, panel_identity={"sha256": "panel-source"}, canonical_binding="canonical-binding",
            target_runtime_binding=contract, missing_reference_ids=[requests[-1]["request_id"]])
        old = dict(recovery_path=origin, config_path="/data/sources/minhao/parent/config.yaml",
            target_path="/data/sources/minhao/target", worker_python="/data/sources/minhao/env/bin/python",
            training_python="/data/sources/minhao/train-env/bin/python", serving=contract["serving"],
            checkpoints={key: dict(export=f"/data/sources/minhao/{key}") for key in ("K4", "K5", "K6")},
            runtime_contract={key: contract[key] for key in ("serving", "native_runtime_sha256")},
            versions={"serving": dict(python="3.12.14", torch="2.9.1+cu130", transformers="4.57.1")},
            pinned_files={"/data/sources/minhao/target/model.safetensors": contract["target_files"]["model.safetensors"]})
        write_json(task / "input_bundle_001/INPUTS.json", inputs)
        write_json(prior / "campaign_001/PLAN.json", old)
        write_json(prior / "campaign_001/training/INPUTS.json", {"inputs": [
            dict(path=origin, bytes=42, sha256="15ac28e1235851a15158e789af1480fd2d82535903b1dec5d992ec843c9ae3a1"),
            dict(path=feature_path, bytes=12, sha256="feature-hash")]})
        return task, prior, inputs, old

    def prepare(self, root, *, modify=None):
        task, prior, inputs, old = self.fixture(root)
        if modify:
            modify(inputs)
            write_json(task / "input_bundle_001/INPUTS.json", inputs)
        source = dict(commit="published-commit", dirty=False, source_path="/local/checkout", files={}, symlinks={})
        def copy_record(source_path, destination):
            Path(destination).write_text("retained input fixture\n")
        with patch.object(prepare_run, "source_manifest", return_value=source), \
             patch.object(prepare_run.shutil, "copyfile", side_effect=copy_record):
            plan = prepare_run.prepare(task, prior, root / "execution",
                Path("/data/sources/minhao/runs/retention-test/execution"), Path("/data/sources/minhao/code/retention"))
        return plan, inputs, old

    def test_prepared_paths_and_budget_bind_actual_input_schema(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            plan, inputs, old = self.prepare(root)
            self.assertEqual(plan["origin_state"], old["recovery_path"])
            self.assertEqual(plan["origin_export"], old["checkpoints"]["K6"]["export"])
            self.assertEqual(plan["configs"], {arm: old["config_path"] for arm in ("NATIVE", "RECENT", "CONTROL")})
            self.assertEqual(set(plan["checkpoints"]), {"K5", "K6"})
            self.assertEqual(plan["missing_reference_ids"], inputs["missing_reference_ids"])
            self.assertEqual(plan["serving_budget"], dict(timed_completions=18, parent_walks=3, missing_canonicals=1))
            self.assertEqual(plan["production_budget"], dict(trajectories=3, adam_steps=384, response_presentations=768))
            self.assertEqual(plan["qualification_budget"]["temporary_adam_steps"], 3)
            self.assertEqual(plan["analysis"], ANALYSIS)
            self.assertIn(plan["feature_manifest"], plan["input_identities"])
            for path in (root / "execution").rglob("*"):
                if path.is_file() and path.name not in ("PLAN.json", "SEAL.json"):
                    remote = str(Path(plan["root"]) / path.relative_to(root / "execution"))
                    self.assertEqual(plan["input_identities"][remote], identity(path))
            seal = read(root / "execution/sealed/SEAL.json")
            self.assertEqual(seal["files"][str(Path(plan["root"]) / "sealed/PLAN.json")],
                             identity(root / "execution/sealed/PLAN.json"))

    def test_rejects_modified_panel_before_sealing(self):
        with TemporaryDirectory() as folder:
            def change(inputs):
                inputs["panel"]["requests"][0]["prompt_ids"] = [999]
            with self.assertRaisesRegex(ValueError, "panel|Panel|seal"):
                self.prepare(Path(folder), modify=change)

    def test_rejects_canonical_runtime_different_from_planned_runtime(self):
        with TemporaryDirectory() as folder:
            def change(inputs):
                inputs["target_runtime_binding"]["native_runtime_sha256"]["models/dflash.py"] = "other-runtime"
            with self.assertRaisesRegex(ValueError, "runtime|Runtime|canonical|Canonical"):
                self.prepare(Path(folder), modify=change)


@unittest.skipIf(runtime is None, "Installed CPU training dependencies required for inherited preflight imports")
class RuntimeIntegrationTests(unittest.TestCase):
    def plan(self, root):
        origin = str(root / "origin.pt")
        Path(origin).write_text("unchanged state")
        plan = dict(root=str(root), execution_commit="published", panel_size=3,
            origin_state=origin, source_manifest=str(root / "SOURCE.json"), analysis=deepcopy(ANALYSIS),
            production_budget=dict(trajectories=3, adam_steps=384, response_presentations=768),
            original_plan=dict(recovery_path=origin,
                checkpoints={key: dict(export="/retained/" + key) for key in ("K4", "K5", "K6")},
                pinned_files={"/retained/K4/model.safetensors": {}, "/retained/K5/model.safetensors": {}}),
            storage=dict(reserve_bytes=10, maximum_additional_bytes=20), input_identities={origin: identity(origin)},
            host_memory_required_bytes=0, qualification_root=str(root / "qualification"))
        write_json(root / "sealed/PLAN.json", plan)
        write_json(root / "sealed/SEAL.json", dict(execution_commit=plan["execution_commit"],
            files={str(root / "sealed/PLAN.json"): identity(root / "sealed/PLAN.json")}))
        return plan

    def test_plan_seal_covers_runtime_fields_and_allows_only_pretraining_additions(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            plan = self.plan(root)
            runtime.verify_execution_plan(plan)
            trained = dict(plan, assignments=str(root / "ASSIGNMENTS.json"), binding="pretraining-seal")
            runtime.verify_execution_plan(trained)
            altered = dict(plan, origin_state="/another/weights-only.pt")
            with self.assertRaisesRegex(ValueError, "sealed configuration"):
                runtime.verify_execution_plan(altered)
            write_json(root / "sealed/PLAN.json", altered)
            with self.assertRaisesRegex(ValueError, "execution package changed"):
                runtime.verify_execution_plan(altered)

    def test_preflight_checks_fixed_design_and_exact_inputs_without_loading_model(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            plan = self.plan(root)
            with patch("experiments.lost_repair.run.preflight", return_value={"status": "passed"}) as original, \
                 patch.object(runtime, "physical_gpu", return_value="expected-card"), \
                 patch.object(runtime, "capacity", return_value={"status": "passed"}):
                runtime.preflight(plan)
                legacy = original.call_args.args[0]
                self.assertEqual(set(legacy["checkpoints"]), {"K5", "K6"})
                self.assertNotIn("/retained/K4/model.safetensors", legacy["pinned_files"])
                self.assertIn("/retained/K5/model.safetensors", legacy["pinned_files"])
                altered = deepcopy(plan)
                altered["production_budget"]["adam_steps"] += 1
                with self.assertRaisesRegex(ValueError, "design changed"):
                    runtime.preflight(altered)
                Path(plan["origin_state"]).write_text("changed state")
                with self.assertRaisesRegex(ValueError, "Sealed input changed"):
                    runtime.preflight(plan)

    def test_capacity_requires_peak_overlap_plus_reserve(self):
        plan = dict(root="/unused", storage=dict(reserve_bytes=10, maximum_additional_bytes=20))
        with patch.object(runtime.shutil, "disk_usage", return_value=SimpleNamespace(free=29)):
            with self.assertRaisesRegex(RuntimeError, "reserve unavailable"):
                runtime.capacity(plan)
        with patch.object(runtime.shutil, "disk_usage", return_value=SimpleNamespace(free=30)):
            self.assertEqual(runtime.capacity(plan)["status"], "passed")

    def test_complete_authenticated_reference_pool_needs_no_model_or_generation(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            plan = self.plan(root)
            panel = panel_fixture()
            references = {row["request_id"]: dict(prompt_ids=row["prompt_ids"], response_ids=[5],
                finish_reason="eos", verified=True) for row in panel["requests"]}
            write_json(root / "PANEL.json", panel)
            write_json(root / "REFERENCES.json", references)
            plan.update(panel_manifest=str(root / "PANEL.json"), reused_references=str(root / "REFERENCES.json"),
                missing_reference_ids=[], checkpoints={"K6": {"export": "/retained/K6"}})
            with patch("online_feedback.sglang_backend.SGLangDFlashBackend") as backend:
                actual = runtime.prepare_canonicals(plan)
            backend.assert_not_called()
            self.assertEqual(actual, references)
            receipt = read(root / "canonicals/STATUS.json")
            self.assertEqual(receipt["planned_generations"], 0)
            self.assertEqual(receipt["returned"], 0)
            self.assertEqual(receipt["reused"], 3)
            self.assertEqual(receipt["status"], "completed")

    def test_canonical_inventory_change_fails_before_model_creation(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            plan = self.plan(root)
            write_json(root / "PANEL.json", panel_fixture())
            write_json(root / "REFERENCES.json", {})
            plan.update(panel_manifest=str(root / "PANEL.json"), reused_references=str(root / "REFERENCES.json"),
                missing_reference_ids=[])
            with patch("online_feedback.sglang_backend.SGLangDFlashBackend") as backend:
                with self.assertRaisesRegex(ValueError, "inventory changed"):
                    runtime.prepare_canonicals(plan)
            backend.assert_not_called()

    def test_pretraining_seal_authenticates_assignments_and_parent_observations(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            plan = self.plan(root)
            assignments, panel, observations = root / "ASSIGNMENTS.json", root / "PANEL.json", root / "parent.json"
            for path in (assignments, panel, observations):
                write_json(path, {"frozen": True})
            seal = dict(inputs=plan["input_identities"], parent_observations={str(observations): identity(observations)},
                assignments=identity(assignments), panel=identity(panel))
            write_json(root / "PRETRAINING_SEAL.json", seal)
            plan.update(assignments=str(assignments), panel_manifest=str(panel),
                        binding=identity(root / "PRETRAINING_SEAL.json")["sha256"])
            runtime.verify_pretraining(plan)
            write_json(observations, {"changed": True})
            with self.assertRaisesRegex(ValueError, "sealed content changed"):
                runtime.verify_pretraining(plan)

    def test_zero_eligible_stops_before_qualification_training_or_serving(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            plan = self.plan(root)
            write_json(root / "PLAN.json", plan)
            assignments = dict(summary=dict(training_feasible=False, control_identical=True))
            with patch.object(runtime, "preflight"), patch.object(runtime, "launch") as launch, \
                 patch.object(runtime, "prepare_assignments", return_value=(plan, assignments)), \
                 patch.object(runtime, "serve") as serve:
                result = runtime.run(root / "PLAN.json")
            self.assertEqual(result["status"], "completed_no_treatment")
            self.assertEqual(result["scientific_steps"], 0)
            self.assertEqual(result["timed_generations"], 0)
            self.assertEqual(launch.call_count, 1)
            self.assertEqual(launch.call_args.args[1], "experiments.recent_repair_retention.reference")
            serve.assert_not_called()
            self.assertEqual(read(root / "STATUS.json")["status"], "completed_no_treatment")

    def test_identical_control_reuses_result_and_never_trains_third_arm(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            plan = self.plan(root)
            write_json(root / "PLAN.json", plan)
            assignments = dict(summary=dict(training_feasible=True, control_identical=True))
            phases = []
            def launch(plan, module, arguments, name, **kwargs):
                phases.append(name)
                if name.startswith("qualification_"):
                    case = name.removeprefix("qualification_")
                    write_json(root / "qualification" / case / "RESULT.json", {"status": "passed"})
                if name.startswith("train_"):
                    arm = name.removeprefix("train_")
                    write_json(root / "training" / arm / "RESULT.json", dict(status="completed", checkpoint_path="/export/" + arm))
            with patch.object(runtime, "preflight"), patch.object(runtime, "launch", side_effect=launch), \
                 patch.object(runtime, "prepare_assignments", return_value=(plan, assignments)), \
                 patch.object(runtime, "verify_pretraining"), \
                 patch.object(runtime, "prepare_canonicals", return_value={}), \
                 patch("experiments.recent_repair_retention.endpoint_difference.compare_weights", return_value={"different": 1}), \
                 patch.object(runtime, "serve") as serve:
                result = runtime.run(root / "PLAN.json")
            self.assertEqual(result["status"], "completed")
            self.assertEqual([p for p in phases if p.startswith("train_")], ["train_NATIVE", "train_RECENT"])
            exports = serve.call_args.args[1]
            self.assertEqual(exports["CONTROL"], exports["RECENT"])
            receipt = read(root / "CONTROL_REUSE.json")
            self.assertEqual(receipt["scientific_optimizer_steps"], 0)
            self.assertFalse(receipt["placement_identifiable"])

    def test_failed_qualification_preserves_failure_and_never_starts_scientific_work(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            plan = self.plan(root)
            write_json(root / "PLAN.json", plan)
            assignments = dict(summary=dict(training_feasible=True, control_identical=False))
            phases = []
            def launch(plan, module, arguments, name, **kwargs):
                phases.append(name)
                if name == "qualification_NATIVE":
                    write_json(root / "qualification/NATIVE/RESULT.json", {"status": "failed"})
            with patch.object(runtime, "preflight"), patch.object(runtime, "launch", side_effect=launch), \
                 patch.object(runtime, "prepare_assignments", return_value=(plan, assignments)), \
                 patch.object(runtime, "serve") as serve:
                with self.assertRaisesRegex(ValueError, "qualification did not pass"):
                    runtime.run(root / "PLAN.json")
            self.assertEqual(phases, ["reference", "qualification_NATIVE"])
            serve.assert_not_called()
            self.assertEqual(read(root / "STATUS.json")["status"], "failed")

    def test_fixed_start_outcomes_keep_earlier_break_and_late_match_separate(self):
        start = dict(request_id="r0", canonical_start=5, repair_slot=3, a6=4, genuine_k6_rejection=True)
        observation = dict(accepted=1, valid_proposals=5, genuine_rejection=True)
        row = fixed_starts.outcome_row(start, observation, {"draft_tokens": [10, 99, 12]}, [10, 11, 12], "RECENT")
        self.assertTrue(row["earlier_break"])
        self.assertFalse(row["exact_slot_relapse"])
        self.assertFalse(row["accepts_through"])
        self.assertTrue(row["direct_canonical_agreement"])
        self.assertEqual(row["lost_prefix_occurrences"], 3)
        self.assertEqual(row["net_prefix_occurrences"], -3)

    def test_fixed_start_subsequence_and_tag_seal_precede_all_endpoint_replays(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            panel = panel_fixture()
            requests = {row["request_id"]: row for row in panel["requests"]}
            write_json(root / "PANEL.json", panel)
            order = list(reversed([row["request_id"] for row in panel["requests"]]))
            plan = dict(panel_manifest=str(root / "PANEL.json"), path_request_ids=order,
                fixed_start_seconds_limit=5, target_path="target", worker_python="python",
                checkpoints={name: {"export": name} for name in ("K5", "K6")})
            exports = {name: name for name in ("NATIVE", "RECENT", "CONTROL")}
            references = {rid: dict(response_ids=[100, 11, 12, 13, 14], finish_reason="length",
                prompt_ids=row["prompt_ids"], verified=True) for rid, row in requests.items()}
            output = root / "fixed"
            captures, endpoints = [], []
            def block(accepted):
                proposals = [11, 12, 13, 14]
                proposals[accepted] = 999
                return dict(output_start=0, anchor_token=100, positions=list(range(16)),
                    draft_tokens=proposals + [999] * 11, accepted_draft_tokens=accepted,
                    finished=False, stop_clipped=False)
            class Backend:
                def __init__(self, *args, **kwargs):
                    pass
                def capture(self, request, checkpoint, directory):
                    captures.append(request["request_id"])
                    record = dict(references[request["request_id"]], stop_token_ids=[],
                                  metadata={"trace": [block(2)]})
                    return record, {"0": {"request_id": request["request_id"]}}
                def compact_request(self, request_id):
                    return {"status": "compacted"}
                def one_block(self, state, checkpoint):
                    if checkpoint in exports:
                        self.seal = read(output / "TAG_SEAL.json")
                        if self.seal["endpoint_loaded"] is not False or len(captures) != 3:
                            raise AssertionError("Endpoint loaded before all parent labels were sealed")
                        endpoints.append((checkpoint, state["request_id"]))
                    return {"blocks": [block(1 if checkpoint == "K5" else 2)]}
                def close(self):
                    pass
            budget = SimpleNamespace(start_gpu=lambda n: None, end_gpu=lambda: None,
                                     summary=lambda: {"mocked": True})
            service = SimpleNamespace(connection={}, close=lambda: None)
            with patch("experiments.repair_history_margin.native_worker.MarginBackend", Backend), \
                 patch.object(fixed_starts, "WorkBudget", return_value=budget), \
                 patch.object(fixed_starts, "BudgetService", return_value=service), \
                 patch.object(fixed_starts, "counted", side_effect=lambda budget, kind, fn, **kwargs: fn()), \
                 patch.object(fixed_starts, "validate_parent", return_value={"status": "passed"}), \
                 patch.object(fixed_starts, "validate_commits"), \
                 patch.object(fixed_starts, "neutral_parity", return_value={"status": "passed"}):
                result = fixed_starts.run(plan, exports, references, output)
            self.assertEqual(captures, order)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(len(endpoints), 9)
            starts = read(output / "PRE_ENDPOINT_STARTS.json")
            self.assertEqual([row["repair_slot"] for row in starts], [2, 2, 2])
            self.assertEqual(len(read(output / "OUTCOMES.json")), 9)


if __name__ == "__main__":
    unittest.main()
