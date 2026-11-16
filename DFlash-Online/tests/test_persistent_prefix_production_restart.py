"""CPU restart admission: preserve consumed work, task identities and old state."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.persistent_prefix_withholding import production_restart as M
from experiments.persistent_prefix_withholding.resources import LIMITS, RESTART_AMENDMENT
from experiments.persistent_prefix_withholding.staging import seed_reference_destination
from online_feedback.storage import write_json
from test_persistent_prefix_reference_join import audit_schema


class ProductionRestartTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.old, self.root = self.base/"failed", self.base/"restart"
        self.old.mkdir(); self.root.mkdir()
        self.plan = dict(root=str(self.root), execution_commit="a"*40, binding="science",
            limits=dict(LIMITS), input_identities={}, prior_source_manifests=[],
            allocation_roots=[str(self.old)], configs={}, qualification_receipts={},
            references=str(self.root/"references/CANONICALS.json"))
        self.plan.update({key:"fixed-"+key for key in M.continuation.SCIENTIFIC_VALUES})
        self.plan["devices"] = dict(A="GPU-A", B="GPU-B")
        self.plan["origin_export"] = "fixed-K6"
        panel, missing, reused = audit_schema()
        fixtures = dict(data_panel=panel, missing_references=missing, reused_references=reused)
        for key in M.continuation.SCIENTIFIC_FILES:
            node = self.save("inputs/"+key, fixtures.get(key, {"fixed":key}))
            self.plan[key] = node["path"]
            self.plan["input_identities"][node["path"]] = node["identity"]
        for arm in ("N","E","S"):
            self.plan["configs"][arm] = self.save("configs/"+arm, {"arm":arm})["path"]
        self.old_source = dict(commit=M.FAILED_COMMIT, dirty=False, files={
            "experiments/persistent_prefix_withholding/training.py": {"sha256":"fixed"},
            "experiments/persistent_prefix_withholding/production_restart.py": None})
        self.new_source = deepcopy(self.old_source)
        self.new_source["commit"] = self.plan["execution_commit"]
        self.new_source["files"]["experiments/persistent_prefix_withholding/production_restart.py"] = {"sha256":"new"}
        self.plan["source_manifest"] = self.save("new_source",self.new_source)["path"]
        failed_source = self.save("old_source",self.old_source)
        self.failed_plan = deepcopy(self.plan)
        self.failed_plan.update(root=str(self.old),execution_commit=M.FAILED_COMMIT,source_manifest=failed_source["path"])
        self.old_clock = dict(first_model_load_epoch=1794802603.911516,
            completed_card_s=435.09486651420593,active={},finished=[dict(task="qualification",occupied_card_s=435.09486651420593)])
        self.proof = dict(status="passed",execution_commit=M.FAILED_COMMIT,clock=self.old_clock,receipts={})
        self.receipts = {}
        for label in ("A","B","cross_card","serving","serving_A","serving_B"):
            node=self.save("qualification/"+label,dict(status="passed",execution_commit="q"*40))
            self.proof["receipts"][label]=dict(node,execution_commit="q"*40)
            self.receipts[label]=M.read(node["path"])
            self.plan["qualification_receipts"][label]=node["path"]
        expected=M.restart_expected_work()
        self.work=dict(attempted=expected["prior_attempted"],completed=expected["prior_completed"],
            failed=dict.fromkeys(LIMITS,0),unknown=dict.fromkeys(LIMITS,0),
            scientific_binding=self.plan["binding"],next_operation=3454,
            pending={"3453":dict(task="S8",label="forward_loss",counts=dict(response_forward=1))})
        self.clock=deepcopy(self.old_clock)
        self.clock["finished"].append(dict(task="production",occupied_card_s=2414.1096606254578))
        self.clock["completed_card_s"]=2849.2045271396637
        limits=dict(LIMITS,adam=673,response_forward=1365,response_backward=1364,training_initialization=9)
        self.plan["operational_amendment"]=dict(authority=RESTART_AMENDMENT,scientific_binding="science",original_limits=dict(LIMITS),effective_limits=limits)
        self.plan["elapsed_time_amendment"]=dict(authority=RESTART_AMENDMENT,scientific_binding="science",original_elapsed_s=14400,effective_elapsed_s=21600,occupied_card_cap_s=14400,first_model_load_epoch=1794802603.911516)
        authority=dict(scope="one_explicit_packaging_repair_restart",failed_execution_commit=M.FAILED_COMMIT,
            first_model_load_epoch=1794802603.911516,elapsed_cap_s=21600,occupied_card_cap_s=14400,
            retry_tasks=list(M.REMAINING_TASKS),reused_tasks=list(M.REUSED_TASKS),failed_forward_remains_unknown=1,
            qualification_repeated=False,scientific_changes=False,additional_retry_authorized=False)
        self.manifest=dict(scope=M.SCOPE,execution_commit=self.plan["execution_commit"],binding="science",
            failed_plan=self.save("failed_plan",self.failed_plan),failed_source=failed_source,
            failed_campaign=self.save("failed_campaign",dict(status="failed",completed=list(M.REUSED_TASKS))),
            failed_qualification_reuse=self.save("failed_qualification_reuse",self.proof),
            failed_work=self.save("failed_work",self.work),failed_clock=self.save("failed_clock",self.clock),
            amendment=self.save("amendment",self.plan["operational_amendment"]),
            elapsed_amendment=self.save("elapsed",self.plan["elapsed_time_amendment"]),
            authorization=self.save("authority",authority),completed_tasks={},retained_artifacts={})
        self.seal_sources()
        for name in M.REUSED_TASKS:
            root=self.old/"tasks"/name
            write_json(root/"STATUS.json",dict(status="completed",execution_commit=M.FAILED_COMMIT))
            result=dict(status="completed")
            if name in ("E7","S7","E8","N8"):
                state=root/"trainer/training_state.pt"; weights=root/"export/model.safetensors"
                state.parent.mkdir(parents=True);weights.parent.mkdir(parents=True)
                state.write_bytes((name+"state").encode());weights.write_bytes((name+"weights").encode())
                result["endpoint"]=dict(state=str(state),export=str(weights.parent))
                if name != "E7":
                    self.manifest["retained_artifacts"].update({str(p):M.file_identity(p) for p in (state,weights)})
            else:
                paths=[]
                for i in range(128):
                    path=root/"features"/(str(i)+".pt");path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(str(i).encode());paths.append(str(path))
                    if name=="S_collect":self.manifest["retained_artifacts"][str(path)]=M.file_identity(path)
                result["feature_paths"]=paths
                write_json(root/"FEATURES.json",result)
            write_json(root/"RESULT.json",result)
            self.manifest["completed_tasks"][name]=dict(root=str(root),files=M.compact_files(root))
        self.seal()
        seed_reference_destination(self.plan)
        patcher=patch.object(M.continuation,"authenticate_continuation",return_value=(self.proof,self.receipts))
        self.authenticate_prior=patcher.start();self.addCleanup(patcher.stop)

    def save(self,name,value):
        path=self.base/(name+".json");write_json(path,value)
        return dict(path=str(path),identity=M.file_identity(path))

    def seal_sources(self):
        self.plan["source_manifest"]=self.save("new_source",self.new_source)["path"]
        old,new=self.old_source["files"],self.new_source["files"]
        changes={k:dict(before=old.get(k),after=new.get(k)) for k in sorted(set(old)|set(new)) if old.get(k)!=new.get(k)}
        self.manifest["source_differences"]=self.save("diff",dict(status="reviewed",execution_commit=self.plan["execution_commit"],failed_execution_commit=M.FAILED_COMMIT,changes=changes))

    def seal(self):
        self.plan["production_restart"]=self.save("restart_manifest",self.manifest)

    def test_import_preserves_old_unknown_work_and_reuses_only_completed_tasks(self):
        old_work=Path(self.manifest["failed_work"]["path"]).read_bytes()
        proof=M.import_restart(self.plan)
        work=M.read(self.root/"WORK.json")
        self.assertEqual(work["attempted"],self.work["attempted"])
        self.assertEqual(work["completed"],self.work["completed"])
        self.assertEqual(work["inherited_pending"],self.work["pending"])
        self.assertEqual(work["unknown"]["response_forward"],1)
        self.assertEqual(work["pending"],{})
        self.assertEqual(work["next_operation"],3454)
        self.assertEqual(M.read(self.root/"CLOCK.json"),self.clock)
        self.assertEqual(proof["initial_work"],work)
        self.assertEqual(set(p.name for p in (self.root/"tasks").iterdir()),set(M.REUSED_TASKS))
        self.assertFalse(list((self.root/"tasks").rglob("*.pt")))
        self.assertEqual(Path(self.manifest["failed_work"]["path"]).read_bytes(),old_work)
        self.assertEqual(proof["remaining"]["adam"],128)
        self.assertEqual(proof["expected_attempted"]["response_forward"],1365)
        self.assertEqual(proof["expected_completed"]["response_forward"],1364)
        self.assertEqual(proof["remaining"]["feedback_generation"],0)
        self.authenticate_prior.assert_called_once_with(self.failed_plan)
        self.assertTrue(M.qualified_receipt_matches(self.plan,"A",self.receipts["A"]))

    def test_second_import_cannot_repeat_work_or_overwrite_evidence(self):
        M.import_restart(self.plan); before=(self.root/"WORK.json").read_bytes()
        with self.assertRaises(FileExistsError):M.import_restart(self.plan)
        self.assertEqual((self.root/"WORK.json").read_bytes(),before)

    def test_missing_reference_seed_fails_before_import_or_backend(self):
        Path(self.plan["references"]).unlink()
        with patch("experiments.persistent_prefix_withholding.task_runner.create_backend") as backend:
            with self.assertRaises(FileNotFoundError):M.import_restart(self.plan)
        backend.assert_not_called()
        self.assertFalse((self.root/"WORK.json").exists())
        self.assertFalse((self.root/"tasks").exists())

    def test_changed_pending_work_cannot_be_silently_reset(self):
        self.work["pending"]={};self.manifest["failed_work"]=self.save("failed_work",self.work);self.seal()
        with self.assertRaisesRegex(ValueError,"work or completed-task"):M.import_restart(self.plan)

    def test_clock_reset_is_rejected(self):
        self.clock["first_model_load_epoch"]+=60;self.manifest["failed_clock"]=self.save("failed_clock",self.clock);self.seal()
        with self.assertRaisesRegex(ValueError,"original epoch"):M.import_restart(self.plan)

    def test_training_code_change_is_rejected_even_with_reviewed_diff(self):
        self.new_source["files"]["experiments/persistent_prefix_withholding/training.py"]={"sha256":"changed"}
        self.seal_sources();self.seal()
        with self.assertRaisesRegex(ValueError,"scientific component"):M.import_restart(self.plan)

    def test_changed_compact_task_receipt_is_rejected(self):
        write_json(self.old/"tasks/S7/RESULT.json",dict(status="failed"))
        with self.assertRaisesRegex(ValueError,"compact evidence"):M.import_restart(self.plan)

    def test_changed_own_reference_feature_is_rejected(self):
        path=next(p for p in self.manifest["retained_artifacts"] if "/features/" in p)
        Path(path).write_bytes(b"different")
        with self.assertRaisesRegex(ValueError,"tensor changed"):M.import_restart(self.plan)

    def test_incomplete_task_cannot_be_relabelled_complete(self):
        node=self.manifest["completed_tasks"]["S7"];write_json(Path(node["root"])/"STATUS.json",dict(status="failed",execution_commit=M.FAILED_COMMIT));node["files"]=M.compact_files(node["root"]);self.seal()
        with self.assertRaisesRegex(ValueError,"cannot be imported"):M.import_restart(self.plan)

    def test_import_refuses_preexisting_unfinished_S8(self):
        (self.root/"tasks/S8").mkdir(parents=True)
        with self.assertRaises(FileExistsError):M.import_restart(self.plan)

    def test_missing_retained_tensor_binding_is_rejected(self):
        self.manifest["retained_artifacts"].pop(next(iter(self.manifest["retained_artifacts"])));self.seal()
        with self.assertRaisesRegex(ValueError,"unbound"):M.import_restart(self.plan)

    def test_absent_elapsed_authorization_cannot_extend_clock(self):
        self.plan["elapsed_time_amendment"]["effective_elapsed_s"]=14400
        with self.assertRaisesRegex(ValueError,"elapsed"):M.import_restart(self.plan)

    def prepare_entry(self):
        self.plan.update(source_root=str(self.base),cpu_gate=str(self.root/"CPU_PREFLIGHT.json"),
                         resource_admission=str(self.root/"RESOURCE_ADMISSION.json"))
        path=self.root/"sealed/PLAN.json";write_json(path,self.plan)
        return path

    def cpu_pass(self,*args,**kwargs):
        self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"],"")
        write_json(self.plan["cpu_gate"],dict(status="passed",execution_commit=self.plan["execution_commit"]))

    def test_tracked_entry_imports_before_admission_and_launches_remaining_scope(self):
        path=self.prepare_entry(); order=[]
        def admit(plan,*args):
            order.append("admission")
            self.assertTrue((self.root/"RESTART_IMPORT.json").exists())
            self.assertEqual(M.read(self.root/"WORK.json")["unknown"]["response_forward"],1)
            return dict(status="passed",costs={},task_memory_bytes={},task_vram_bytes={})
        def campaign(actual):
            order.append("campaign")
            staged=M.read(actual);proof=M.continuation.bound_json(staged["restart_import"])
            self.assertEqual(proof["remaining_tasks"],["S8","references","final"])
            self.assertEqual(staged["gpu_deadline_monotonic"]>0,True)
            return dict(status="completed")
        with patch.object(M.subprocess,"run",side_effect=self.cpu_pass), \
             patch("experiments.persistent_prefix_withholding.task_runner.storage_boundary",return_value=dict(status="passed")), \
             patch("experiments.persistent_prefix_withholding.admission.qualify_resources",side_effect=admit), \
             patch("experiments.persistent_prefix_withholding.campaign.run_campaign",side_effect=campaign), \
             patch("experiments.persistent_prefix_withholding.report.assemble_report"), \
             patch("experiments.persistent_prefix_withholding.launch.run_qualification_child") as qualification:
            result=M.run(path)
        self.assertEqual(result["status"],"completed");self.assertEqual(order,["admission","campaign"])
        qualification.assert_not_called()

    def test_entry_missing_seed_stops_before_preflight_import_or_task_launch(self):
        path=self.prepare_entry();Path(self.plan["references"]).unlink()
        with patch.object(M.subprocess,"run") as subprocess, \
             patch("experiments.persistent_prefix_withholding.campaign.run_campaign") as campaign, \
             patch("experiments.persistent_prefix_withholding.report.assemble_report"):
            with self.assertRaises(FileNotFoundError):M.run(path)
        subprocess.assert_not_called();campaign.assert_not_called()
        self.assertFalse((self.root/"tasks").exists());self.assertFalse((self.root/"WORK.json").exists())
        self.assertEqual(M.read(self.root/"RESTART_STATUS.json")["status"],"stopped")

    def test_failed_admission_preserves_import_but_does_not_launch_or_reset_clock(self):
        path=self.prepare_entry()
        with patch.object(M.subprocess,"run",side_effect=self.cpu_pass), \
             patch("experiments.persistent_prefix_withholding.task_runner.storage_boundary",return_value=dict(status="passed")), \
             patch("experiments.persistent_prefix_withholding.admission.qualify_resources",return_value=dict(status="blocked")), \
             patch("experiments.persistent_prefix_withholding.campaign.run_campaign") as campaign, \
             patch("experiments.persistent_prefix_withholding.report.assemble_report"):
            with self.assertRaisesRegex(RuntimeError,"resource admission"):M.run(path)
        campaign.assert_not_called();self.assertEqual(M.read(self.root/"CLOCK.json"),self.clock)
        with self.assertRaises(FileExistsError):M.run(path)
