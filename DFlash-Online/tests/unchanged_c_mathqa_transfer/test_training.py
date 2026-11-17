"""Real tiny DFlash integration of the unchanged C controller on MathQA state."""
from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest

import torch

from experiments.unchanged_c_mathqa_transfer import training as T
from experiments.mathqa_fixed_lr_transfer.training import MathQAFixedLRTrainer
from experiments.ok_continuation.resume import differences
from online_feedback.prefix_allocation import PrefixAllocation
from online_feedback.kplus import KPlusRedistribution
import test_mathqa_fixed_lr_transfer_training as baseline_fixtures

BINDING = baseline_fixtures.BINDING


class ControllerTests(unittest.TestCase):
    def make_trainer(self):
        trainer=object.__new__(T.MathQACandidateTrainer)
        trainer.relative_cap=trainer._qualification_cap=.25
        trainer.qualification_only=False
        model=SimpleNamespace(objective="osd_first_error",loss_dtype=torch.bfloat16,
            first_error_redistribution=KPlusRedistribution(),prefix_allocation=PrefixAllocation())
        trainer.bundle=SimpleNamespace(model=model)
        return trainer

    def test_original_native_guard_still_rejects_C_and_only_C_is_allowed(self):
        trainer=self.make_trainer();trainer._check_native_path()
        with self.assertRaises(ValueError):MathQAFixedLRTrainer._check_native_path(trainer)
        trainer.bundle.model.prefix_guard=object()
        with self.assertRaises(ValueError):trainer._check_native_path()
        trainer.bundle.model.prefix_guard=None
        trainer.bundle.model.prefix_allocation.contract["relative_cap"]=0.
        with self.assertRaises(ValueError):trainer._check_native_path()

    def test_production_never_switches_or_resets(self):
        trainer=self.make_trainer()
        with self.assertRaises(ValueError):trainer.qualification_allocation(0.)
        with self.assertRaises(ValueError):trainer.reset_qualification()
        with self.assertRaises(ValueError):trainer.run_pair([])
        with self.assertRaises(ValueError):trainer.restore_qualification("unused")
        with self.assertRaises(AttributeError):trainer.arm="N"


class TinyIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get("MATHQA_C_TINY_CHILD")=="1":
            from test_full_k_training import FullKTinyIntegrationTests
            FullKTinyIntegrationTests.setUpClass.__func__(cls)

    @classmethod
    def tearDownClass(cls):
        if os.environ.get("MATHQA_C_TINY_CHILD")=="1":
            from test_full_k_training import FullKTinyIntegrationTests
            FullKTinyIntegrationTests.tearDownClass.__func__(cls)

    def setUp(self):
        if os.environ.get("MATHQA_C_TINY_CHILD")=="1":
            from test_full_k_training import FullKTinyIntegrationTests
            FullKTinyIntegrationTests.setUp(self)

    make_parent=baseline_fixtures.MathQATinyIntegrationTests.make_parent

    def subprocess_or_continue(self):
        if os.environ.get("MATHQA_C_TINY_CHILD")=="1":return False
        root=Path(__file__).resolve().parents[2]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES="",MATHQA_C_TINY_CHILD="1",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
        env["PYTHONPATH"]=os.pathsep.join(str(root/p) for p in (".","tests","tests/unchanged_c_mathqa_transfer","training","inference"))
        result=subprocess.run([sys.executable,"-m","unittest","-v","unchanged_c_mathqa_transfer.test_training."+type(self).__name__+"."+self._testMethodName],
            cwd=root,env=env,text=True,capture_output=True,timeout=240)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        return True

    def build_candidate(self,name,path,export,**kwargs):
        trainer=T.MathQACandidateTrainer(self.config,self.root/name,arm="C",binding=BINDING,
            origin_state=path,origin_export=export,original_config_text=self.original_config_text,
            expected_cuda_rng_states=0,**kwargs)
        self.addCleanup(trainer.close)
        return trainer

    def test_two_events_own_resume_controller_and_origin_counter_isolation(self):
        if self.subprocess_or_continue():return
        path,export,source=self.make_parent()
        trainer=self.build_candidate("C5",path,export)
        fifth=trainer.train_event("event_005",self.features)
        trainer.verify_boundary(fifth["export_path"])
        saved=self.root/"C5.pt";saved.write_bytes(trainer.latest.read_bytes())
        self.assertEqual((trainer.global_step,trainer.presentations),(640,1280))
        self.assertEqual(differences(source,trainer.origin),[])
        self.assertTrue(all("prefix_allocation" in row["response_observation"] for row in fifth["records"]))
        features=[]
        for index,old in enumerate(self.features):
            raw=torch.load(old,weights_only=True)
            raw["record_id"]=f"own_C5_{index}";raw["label_binding"]["record_id"]=raw["record_id"]
            destination=self.root/f"own_{index}.pt";torch.save(raw,destination);features.append(destination)
        uninterrupted=trainer.train_event("event_006",features)
        final=torch.load(trainer.latest,weights_only=False);trainer.close()
        restored=self.build_candidate("C6_reloaded",path,export,completed_event=5,resume_state=saved,resume_export=fifth["export_path"])
        self.assertEqual(restored.bundle.model.prefix_allocation.relative_cap,.25)
        repeated=restored.train_event("event_006",features)
        restored.verify_boundary(repeated["export_path"])
        actual=torch.load(restored.latest,weights_only=False)
        for key in ("draft_state_dict","backend","rng","global_step","presentations"):
            self.assertEqual(differences(final[key],actual[key]),[],key)
        for key in ("record_id","pass","sampled_anchors","metrics"):
            self.assertEqual([r[key] for r in uninterrupted["records"]],[r[key] for r in repeated["records"]])
        self.assertEqual((restored.global_step,restored.presentations),(768,1536))
        self.assertEqual(differences(source,restored.origin),[])
        with self.assertRaises(ValueError):restored.train_event("event_007",features)
        self.assertFalse(torch.cuda.is_initialized())

    def test_native_cap_zero_and_own_active_next_update_reload(self):
        if self.subprocess_or_continue():return
        from experiments.functional_step_geometry.qualify import runtime_state
        path,export,source=self.make_parent()
        trainer=self.build_candidate("qualification",path,export,qualification_only=True)
        features=T.SpecForgeEventTrainer._prepare_event(trainer,"event_005",self.features,2)
        runtime=runtime_state(trainer.bundle.model)
        endpoints=[]
        for label,cap in (("native",None),("cap_zero",0.)):
            trainer.reset_qualification(relative_cap=cap,runtime=runtime)
            trainer._active_event=label;(trainer.events_root/label).mkdir()
            trainer.bundle.model.begin_event();order=torch.randperm(len(features)).tolist()
            forward=trainer._native_forward
            if cap is None:trainer._native_forward=trainer.pr51_native_forward
            trainer.run_pair([features[i] for i in order[:2]])
            trainer._native_forward=forward
            endpoints.append(deepcopy(T.native.live_state(trainer)))
        self.assertEqual(differences(*endpoints),[])
        self.assertEqual(differences(source,trainer.origin),[])
        trainer.reset_qualification(relative_cap=.25,runtime=runtime)
        trainer._active_event="active";(trainer.events_root/"active").mkdir()
        trainer.bundle.model.begin_event();order=torch.randperm(len(features)).tolist()
        trainer.run_pair([features[i] for i in order[:2]])
        snapshot=self.root/"temporary.pt";trainer.save_qualification(snapshot)
        retained=torch.load(snapshot,weights_only=False)
        runtime=runtime_state(trainer.bundle.model)
        rows=trainer.run_pair([features[i] for i in order[2:4]],ordinal=2)
        endpoint=deepcopy(T.native.live_state(trainer));work=deepcopy(trainer.work)
        trainer.restore_qualification(snapshot,runtime=runtime)
        self.assertEqual(trainer.bundle.model.prefix_allocation.relative_cap,.25)
        self.assertEqual(trainer.work,work)
        repeated=trainer.run_pair([features[i] for i in order[2:4]],ordinal=2)
        self.assertEqual(differences(endpoint,T.native.live_state(trainer)),[])
        for key in ("metrics","sampled_anchors","response_observation"):
            self.assertEqual([r[key] for r in rows],[r[key] for r in repeated])
        self.assertEqual((trainer.global_step,trainer.presentations),(514,1028))
        self.assertTrue(all(s["step"].item()==513 for s in retained["backend"]["optimizer"]["optimizer_state_dict"]["state"].values()))
        self.assertEqual(differences(source,trainer.origin),[])
        self.assertFalse(torch.cuda.is_initialized())


if __name__=="__main__":unittest.main()
