"""Qualification-reuse accounting units; staging has separate end-to-end tests."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.persistent_prefix_withholding import production_continue as M
from experiments.persistent_prefix_withholding import campaign as C, admission as A, launch as L, report as R
from experiments.persistent_prefix_withholding import staging as S
from experiments.persistent_prefix_withholding.resources import LIMITS
from online_feedback.storage import write_json

PRODUCTION_AMENDED_LIMITS = dict(LIMITS, adam=650, response_forward=1316,
                                response_backward=1316, training_initialization=8)


class ProductionContinuationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.root = self.base/"production"
        self.root.mkdir()
        self.plan = dict(root=str(self.root), source_root=str(self.base), execution_commit="new", binding="training-binding",
            prior_source_manifests=[], allocation_roots=[str(self.base/"old")], input_identities={},
            cpu_gate=str(self.root/"CPU_PREFLIGHT.json"), resource_admission=str(self.root/"RESOURCE_ADMISSION.json"))
        self.plan["limits"] = dict(LIMITS)
        self.plan.update({key: "same-"+key for key in M.SCIENTIFIC_VALUES})
        self.plan["devices"] = {"A":"GPU-A", "B":"GPU-B"}
        self.plan["source_root"] = str(self.base)
        for key in M.SCIENTIFIC_FILES:
            path = self.base/(key+".json")
            write_json(path, {"requests":[{}]*24} if key == "missing_references" else {"fixed":key})
            self.plan[key] = str(path)
        evidence = self.save("reference_evidence", dict(authenticated=True))
        provenance = self.save("reference_provenance", dict(status="passed",
            new_reference_route="fixed_original_K6_native_greedy_target_verifier", no_target_only_AR_claim=True,
            reference_reuse_receipt=evidence["path"], reference_provenance=evidence["path"]))
        self.plan["reference_provenance"] = provenance["path"]
        self.plan["input_identities"][provenance["path"]] = provenance["identity"]
        self.plan["configs"] = {}
        for arm in ("N", "E", "S"):
            path = self.base/(arm+".yaml")
            path.write_text("qualified: true\n")
            self.plan["configs"][arm] = str(path)
        original = dict(self.plan, execution_commit="training")
        serving = dict(self.plan, execution_commit="serving", binding="serving-relocation-binding")
        self.manifest = dict(scope="production_only", execution_commit="new", binding=self.plan["binding"],
            original_plan=self.save("original_plan", original), serving_plan=self.save("serving_plan", serving), receipts={})
        self.plan["qualification_receipts"] = {}
        for label in ("A", "B", "cross_card", "serving", "serving_A", "serving_B"):
            commit = "serving" if label.startswith("serving") else "training"
            name = ("qualification/"+label+"/STATUS" if label in ("A","B") else
                    "serving_qualification/"+label[-1]+"/STATUS" if label in ("serving_A","serving_B") else
                    "qualification/CROSS_CARD" if label == "cross_card" else "serving_qualification/CROSS_CARD")
            self.manifest["receipts"][label] = dict(self.save(name, dict(status="passed", execution_commit=commit)), execution_commit=commit)
            if label in ("A", "B", "cross_card", "serving"):
                self.plan["qualification_receipts"][label] = self.manifest["receipts"][label]["path"]
        self.manifest["retained_evidence"] = []
        for slot in ("A","B"):
            base = self.base/"qualification"/slot
            paths = [base/name for name in ("PHYSICAL_WORK.jsonl", "cases/RESULT.json", "startup/EXACT_RESTORATION.json")]
            base = self.base/"serving_qualification"/slot/"calls"
            paths += [base/"RESULT.json"]+[base/f"q{slot}_r{repeat}_w{request}.json" for repeat in range(2) for request in range(2)]
            for path in paths:
                write_json(path, dict(raw_fixture=True))
                self.manifest["retained_evidence"].append(dict(path=str(path), identity=M.file_identity(path)))
        self.amendment = dict(authority="PR49_PRODUCTION_CONTINUATION_20261115",
            scientific_binding=self.plan["binding"], original_limits=dict(LIMITS), effective_limits=dict(PRODUCTION_AMENDED_LIMITS))
        self.plan["operational_amendment"] = self.amendment
        self.manifest["amendment"] = self.save("amendment", self.amendment)
        counts = {key:0 for key in LIMITS}
        counts.update(adam=10, response_forward=26, response_backward=26, training_initialization=3, qualification_generation=8)
        self.work = dict(cumulative=dict(attempted=counts, completed=dict(counts),
            failed={key:0 for key in LIMITS}, unknown={key:0 for key in LIMITS}), unresolved_sources=[], counts_are_lower_bounds=False)
        self.manifest["work"] = self.save("work", self.work)
        self.clock = dict(first_model_load_epoch=1794802603.911516, completed_card_s=435.09486651420593,
                         finished=[dict(occupied_card_s=435.09486651420593)], active={})
        self.manifest["clock"] = self.save("clock", self.clock)
        self.old_source = {"experiments/persistent_prefix_withholding/training.py":{"sha256":"fixed"},
                           "experiments/persistent_prefix_withholding/campaign.py":{"sha256":"old"}}
        self.new_source = deepcopy(self.old_source)
        self.new_source["experiments/persistent_prefix_withholding/campaign.py"] = {"sha256":"new"}
        self.seal_source()
        self.seal()
        self.storage = patch.object(M, "storage_boundary", return_value=dict(status="passed"))
        self.storage.start()
        self.addCleanup(self.storage.stop)
        reports = patch.object(R, "assemble_report")
        reports.start()
        self.addCleanup(reports.stop)

    def save(self, name, value):
        path = self.base/(name+".json")
        write_json(path, value)
        return dict(path=str(path), identity=M.file_identity(path))

    def seal_source(self):
        self.plan["source_manifest"] = self.save("current_source", dict(commit="new", dirty=False, files=self.new_source))["path"]
        diff = dict(status="reviewed", execution_commit="new", sources={}, changes={})
        for role in ("training", "serving"):
            diff["sources"][role] = dict(execution_commit=role,
                manifest=self.save(role+"_source", dict(commit=role, dirty=False, files=self.old_source)))
            diff["changes"][role] = {key:dict(before=self.old_source.get(key), after=self.new_source.get(key))
                for key in set(self.old_source)|set(self.new_source) if self.old_source.get(key) != self.new_source.get(key)}
        self.manifest["source_differences"] = self.save("source_differences", diff)

    def seal(self):
        bound = self.save("continuation", self.manifest)
        self.plan["production_continuation"] = dict(manifest=bound["path"], identity=bound["identity"])
        self.plan_path = self.root/"sealed/PLAN.json"
        write_json(self.plan_path, self.plan)

    def cpu_pass(self, *args, **kwargs):
        self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")
        write_json(self.plan["cpu_gate"], dict(status="passed", execution_commit="new"))

    def test_imports_complete_work_and_clock_without_qualification(self):
        with patch.object(S, "reference_readiness", return_value=dict(status="passed")) as readiness, \
                patch.object(M.subprocess, "run", side_effect=self.cpu_pass), \
                patch.object(A, "qualify_resources", return_value=dict(status="passed", costs={}, task_memory_bytes={}, task_vram_bytes={})), \
                patch.object(C, "run_campaign", return_value=dict(status="completed")) as campaign, \
                patch.object(L, "run_qualification_child") as qualification:
            result = M.run(self.plan_path)
        qualification.assert_not_called()
        readiness.assert_called_once()
        campaign.assert_called_once()
        self.assertEqual(result["status"], "completed")
        imported = M.read(self.root/"INITIAL_WORK.json")
        self.assertEqual(imported["attempted"], self.work["cumulative"]["attempted"])
        self.assertEqual(imported["completed"], self.work["cumulative"]["completed"])
        self.assertEqual(imported["limits"], PRODUCTION_AMENDED_LIMITS)
        self.assertEqual(imported["scientific_binding"], "training-binding")
        self.assertEqual(M.read(self.root/"CLOCK.json"), self.clock)
        updated = M.read(self.plan_path)
        self.assertEqual(updated["compiler_baseline_allocated_bytes"], self.plan["compiler_baseline_allocated_bytes"])
        self.assertTrue(M.qualified_receipt_matches(updated, "A", M.read(self.manifest["receipts"]["A"]["path"])))
        with self.assertRaises(FileExistsError):
            M.run(self.plan_path)

    def test_altered_qualification_receipt_and_missing_evidence_fail(self):
        path = Path(self.manifest["receipts"]["A"]["path"])
        path.write_text("{}")
        with self.assertRaisesRegex(ValueError, "evidence changed"):
            M.authenticate_continuation(self.plan)
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            M.authenticate_continuation(self.plan)

    def test_reviewed_manifest_cannot_allow_changed_numerical_component(self):
        self.new_source["experiments/persistent_prefix_withholding/training.py"] = {"sha256":"changed"}
        self.seal_source()
        self.seal()
        with self.assertRaisesRegex(ValueError, "scientific component"):
            M.authenticate_continuation(self.plan)

    def test_unknown_work_is_not_discarded(self):
        self.work["cumulative"]["unknown"]["adam"] = 1
        self.manifest["work"] = self.save("work", self.work)
        self.seal()
        with self.assertRaisesRegex(ValueError, "cannot be discarded"):
            M.authenticate_continuation(self.plan)

    def test_missing_or_altered_raw_receipt_blocks_reuse(self):
        node = self.manifest["retained_evidence"].pop()
        self.seal()
        with self.assertRaisesRegex(ValueError, "raw qualification receipts"):
            M.authenticate_continuation(self.plan)
        self.manifest["retained_evidence"].append(node)
        self.seal()
        Path(node["path"]).write_text("changed")
        with self.assertRaisesRegex(ValueError, "raw qualification evidence changed"):
            M.authenticate_continuation(self.plan)

    def test_original_clock_and_binding_cannot_be_reset(self):
        self.plan["binding"] = "new-binding"
        with self.assertRaisesRegex(ValueError, "identity differs"):
            M.authenticate_continuation(self.plan)
        self.plan["binding"] = "training-binding"
        self.clock["first_model_load_epoch"] += 1
        self.manifest["clock"] = self.save("clock", self.clock)
        self.seal()
        with self.assertRaisesRegex(ValueError, "clock"):
            M.authenticate_continuation(self.plan)

    def test_blocked_fresh_admission_starts_no_worker(self):
        with patch.object(S, "reference_readiness", return_value=dict(status="passed")), \
                patch.object(M.subprocess, "run", side_effect=self.cpu_pass), \
                patch.object(A, "qualify_resources", return_value=dict(status="blocked")), \
                patch.object(C, "run_campaign") as campaign, patch.object(L, "run_qualification_child") as qualification:
            with self.assertRaisesRegex(RuntimeError, "Fresh full-work"):
                M.run(self.plan_path)
        campaign.assert_not_called()
        qualification.assert_not_called()
        self.assertEqual(M.read(self.root/"CLOCK.json"), self.clock)

    def test_reference_readiness_failure_precedes_preflight_and_workers(self):
        with patch.object(S,"reference_readiness",side_effect=FileNotFoundError("canonical seed")) as readiness, \
                patch.object(M.subprocess,"run") as preflight, \
                patch.object(C,"run_campaign") as campaign:
            with self.assertRaises(FileNotFoundError):
                M.run(self.plan_path)
        readiness.assert_called_once()
        preflight.assert_not_called()
        campaign.assert_not_called()
        self.assertFalse((self.root/"WORK.json").exists())
        self.assertFalse((self.root/"CLOCK.json").exists())


if __name__ == "__main__":
    unittest.main()
