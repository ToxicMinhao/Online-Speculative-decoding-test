"""CPU checks for complete MathQA forks and ordinary constant-rate continuation."""
from copy import deepcopy
import io
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import yaml

from experiments.mathqa_fixed_lr_transfer import state as S
from experiments.mathqa_fixed_lr_transfer import training as T
from experiments.ok_continuation.resume import differences
from experiments.role_loss_pilot import training as historical
from online_feedback.rng import capture_rng
from test_role_loss_campaign_training import pilot_state

BINDING = "a" * 64


def origin():
    with patch.object(torch.cuda, "is_initialized", return_value=False):
        return pilot_state(4, "KPLUS", "mathqa")


def arm_config(source, arm):
    raw = yaml.safe_load(source["config_text"])
    raw["training"]["learning_rate"] = S.arm_lr(arm)
    return yaml.safe_dump(raw)


def own_state(arm="S", event=4):
    source, historical_args = origin()
    config = arm_config(source, arm)
    own, receipt = S.fork_state(source, arm=arm, binding=BINDING,
        config_text=config, run_id=source["run_id"],
        original_config_text=historical_args["original_config_text"], expected_cuda_rng_states=0)
    if event > 4:
        later, _ = pilot_state(event, "KPLUS", "mathqa")
        own = deepcopy(own)
        own.update(global_step=128*event, presentations=256*event)
        for key, value in own["backend"]["optimizer"]["optimizer_state_dict"]["state"].items():
            value["step"] = torch.tensor(float(128*event))
        own["events"].extend(later["events"][4:])
        for row in own["events"][4:]:
            row.update(method=arm, dataset="mathqa", kplus_loss=T.kplus_contract(),
                       **{S.CONTRACT_KEY:S.experiment_contract(arm, BINDING)})
            row.pop("pilot_loss")
            for record in row["records"]:
                record["learning_rate"] = S.arm_lr(arm) if record["optimizer_stepped"] else None
                record["response_observation"]["sampled_anchors"] = deepcopy(record["sampled_anchors"])
    args = dict(origin=source, arm=arm, binding=BINDING, config_text=config,
        original_config_text=historical_args["original_config_text"], completed_event=event,
        expected_recipe=deepcopy(own["backend"]["optimizer"]["recipe"]), expected_cuda_rng_states=0)
    return source, own, receipt, args


class MathQAStateTests(unittest.TestCase):
    def test_literal_lr_and_exact_configuration_pin(self):
        self.assertEqual(S.arm_lr("S"), 2.88e-4)
        self.assertNotEqual(S.arm_lr("S"), .96*3e-4)
        source, own, receipt, args = own_state()
        self.assertEqual(receipt["learning_rate_hex"], (2.88e-4).hex())
        for section, field, value in [("training","num_anchors",32), ("training","seed",23),
                ("training","max_steps",256), ("online_feedback","weight_decay",0.)]:
            raw = yaml.safe_load(own["config_text"])
            raw[section][field] = value
            with self.assertRaises(ValueError):
                S.validate_config_text(yaml.safe_dump(raw), "S", args["original_config_text"])

    def test_origin_and_two_arm_forks_do_not_change_history_rng_or_tensors(self):
        source, args = origin()
        before = deepcopy(source)
        rng = capture_rng()
        for arm in S.ARMS:
            own, receipt = S.fork_state(source, arm=arm, binding=BINDING,
                config_text=arm_config(source, arm), run_id=source["run_id"],
                original_config_text=args["original_config_text"], expected_cuda_rng_states=0)
            self.assertEqual(own["global_step"],512)
            self.assertEqual(own["presentations"],1024)
            self.assertEqual(own["horizon"],7424)
            for field in ("draft_state_dict","rng","events"):
                self.assertIs(own[field],source[field])
            self.assertEqual(own["backend"]["optimizer"]["recipe"]["optimizer_options"]["lr"],S.arm_lr(arm))
            self.assertEqual(own["backend"]["optimizer"]["optimizer_state_dict"]["param_groups"][0]["lr"],S.arm_lr(arm))
            self.assertFalse(receipt["seed_reset"])
        self.assertEqual(differences(source,before),[])
        self.assertEqual(differences(rng,capture_rng()),[])

    def test_complete_boundaries_and_serialization_are_read_only(self):
        for arm in S.ARMS:
            for number in (4,5,6):
                _, own, _, args = own_state(arm,number)
                before = deepcopy(own)
                result = S.validate_snapshot(own,**args)
                self.assertEqual(result["global_step"],number*128)
                stream = io.BytesIO();torch.save(own,stream);stream.seek(0)
                decoded = torch.load(stream,weights_only=False)
                S.validate_snapshot(decoded,**args)
                self.assertEqual(differences(before,own),[])
                self.assertEqual(differences(before,decoded),[])

    def test_changed_lr_history_foreign_arm_and_reset_counters_fail(self):
        _, own, _, args = own_state("S",5)
        mutations = [
            lambda s:s["algorithm_contract"][S.CONTRACT_KEY].update(arm="N"),
            lambda s:s["events"][0]["records"][1].update(learning_rate=S.SCALAR_LR),
            lambda s:s["backend"]["optimizer"]["optimizer_state_dict"]["param_groups"][0].update(lr=3e-4),
            lambda s:s["backend"]["optimizer"]["recipe"]["optimizer_options"].update(lr=3e-4),
            lambda s:s["backend"]["optimizer"]["optimizer_state_dict"]["state"].clear(),
            lambda s:s.update(global_step=128),
            lambda s:s.update(horizon=256),
            lambda s:s["algorithm_contract"].update(prefix_allocation={}),
        ]
        for mutate in mutations:
            bad=deepcopy(own);mutate(bad)
            with self.assertRaises(ValueError):S.validate_snapshot(bad,**args)
        with self.assertRaises(ValueError):S.validate_snapshot(own,**dict(args,completed_event=7))
        with self.assertRaises(ValueError):S.validate_origin(own,args["original_config_text"],expected_cuda_rng_states=0)
        with self.assertRaises(ValueError):
            historical.validate_snapshot(own,arm="KPLUS",dataset="mathqa",binding=BINDING,
                config_text=own["config_text"],original_config_text=args["original_config_text"],
                completed_event=5,expected_contract=own["algorithm_contract"],
                expected_recipe=args["expected_recipe"],expected_cuda_rng_states=0)

    def test_wrong_dataset_or_role_cannot_seed_a_fork(self):
        source,args=origin()
        for field,value in [("dataset","gsm8k"),("arm","ROLE")]:
            bad=deepcopy(source);bad["algorithm_contract"][historical.CONTRACT_KEY][field]=value
            with self.assertRaises(ValueError):S.validate_origin(bad,args["original_config_text"],expected_cuda_rng_states=0)

    def test_live_group_recipe_and_config_are_all_checked(self):
        from online_feedback.optimizer import DirectBF16AdamW
        trainer=object.__new__(T.MathQAFixedLRTrainer)
        trainer._arm="S"
        model=torch.nn.Linear(2,1,bias=False).to(torch.bfloat16)
        optimizer=DirectBF16AdamW(model,learning_rate=S.SCALAR_LR)
        trainer.backend=SimpleNamespace(optimizer=optimizer)
        trainer.cfg=SimpleNamespace(training=SimpleNamespace(learning_rate=S.SCALAR_LR))
        trainer._check_actual_lr()
        for destination in (optimizer.recipe["optimizer_options"],optimizer.optimizer.param_groups[0]):
            destination["lr"]=3e-4
            with self.assertRaises(ValueError):trainer._check_actual_lr()
            destination["lr"]=S.SCALAR_LR
        trainer.cfg.training.learning_rate=3e-4
        with self.assertRaises(ValueError):trainer._check_actual_lr()
        with self.assertRaises(AttributeError):trainer.arm="N"

    def test_optional_controllers_and_production_reset_are_rejected(self):
        from online_feedback.kplus import KPlusRedistribution
        trainer=object.__new__(T.MathQAFixedLRTrainer)
        trainer.bundle=SimpleNamespace(model=SimpleNamespace(first_error_redistribution=KPlusRedistribution()))
        trainer._check_native_path()
        for name in T.FORBIDDEN_CONTROLLERS:
            setattr(trainer.bundle.model,name,object())
            with self.assertRaises(ValueError):trainer._check_native_path()
            setattr(trainer.bundle.model,name,None)
        trainer.qualification_only=False
        with self.assertRaises(ValueError):trainer.reset_qualification()
        with self.assertRaises(ValueError):trainer.run_pair([])
        with self.assertRaises(ValueError):trainer.restore_qualification("unused.pt")


class MathQATinyIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get("MATHQA_LR_TINY_CHILD")=="1":
            from test_full_k_training import FullKTinyIntegrationTests
            FullKTinyIntegrationTests.setUpClass.__func__(cls)

    @classmethod
    def tearDownClass(cls):
        if os.environ.get("MATHQA_LR_TINY_CHILD")=="1":
            from test_full_k_training import FullKTinyIntegrationTests
            FullKTinyIntegrationTests.tearDownClass.__func__(cls)

    def setUp(self):
        if os.environ.get("MATHQA_LR_TINY_CHILD")=="1":
            from test_full_k_training import FullKTinyIntegrationTests
            FullKTinyIntegrationTests.setUp(self)

    def subprocess_or_continue(self):
        if os.environ.get("MATHQA_LR_TINY_CHILD")=="1":return False
        root=Path(__file__).resolve().parents[1]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES="",MATHQA_LR_TINY_CHILD="1",OMP_NUM_THREADS="1")
        env["PYTHONPATH"]=os.pathsep.join(str(root/p) for p in (".","tests","training","inference"))
        result=subprocess.run([sys.executable,"-m","unittest","-v",__name__+"."+type(self).__name__+"."+self._testMethodName],
            cwd=root,env=env,text=True,capture_output=True,timeout=240)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        return True

    def make_parent(self):
        from test_full_k_training import FullKTinyIntegrationTests
        from safetensors.torch import save_file
        base=FullKTinyIntegrationTests.build(self,"base")
        T.install_kplus(base.bundle.model)
        saved=deepcopy(T.native.live_state(base))
        old,_=origin()
        saved.update(events=old["events"],global_step=512,presentations=1024)
        saved["algorithm_contract"].pop(T.native.CONTRACT_KEY)
        saved["algorithm_contract"][historical.CONTRACT_KEY]=old["algorithm_contract"][historical.CONTRACT_KEY]
        optimizer=saved["backend"]["optimizer"]["optimizer_state_dict"]
        for index,parameter in zip(optimizer["param_groups"][0]["params"],base.backend.optimizer.model_params):
            optimizer["state"][index]=dict(step=torch.tensor(512.),exp_avg=torch.zeros_like(parameter),exp_avg_sq=torch.zeros_like(parameter))
        path=self.root/"parent.pt";T.save_torch(path,saved)
        export=self.root/"parent_export";export.mkdir();save_file(saved["draft_state_dict"],export/"model.safetensors")
        base.close()
        return path,export,saved

    def build_arm(self,name,arm,path,export,**kwargs):
        config=self.root/(arm+".yaml")
        raw=yaml.safe_load(self.original_config_text);raw["training"]["learning_rate"]=S.arm_lr(arm)
        config.write_text(yaml.safe_dump(raw))
        trainer=T.MathQAFixedLRTrainer(config,self.root/name,arm=arm,binding=BINDING,
            origin_state=path,origin_export=export,original_config_text=self.original_config_text,
            expected_cuda_rng_states=0,**kwargs)
        self.addCleanup(trainer.close)
        return trainer

    def test_two_events_own_resume_and_origin_counter_isolation(self):
        if self.subprocess_or_continue():return
        path,export,source=self.make_parent()
        trainer=self.build_arm("S5","S",path,export)
        fifth=trainer.train_event("event_005",self.features)
        trainer.verify_boundary(fifth["export_path"])
        saved=self.root/"S5.pt";saved.write_bytes(trainer.latest.read_bytes())
        self.assertEqual((trainer.global_step,trainer.presentations),(640,1280))
        self.assertEqual(differences(source,trainer.origin),[])
        features=[]
        for index,old in enumerate(self.features):
            raw=torch.load(old,weights_only=True)
            raw["record_id"]=f"own_S5_{index}";raw["label_binding"]["record_id"]=raw["record_id"]
            destination=self.root/f"own_{index}.pt";torch.save(raw,destination);features.append(destination)
        uninterrupted=trainer.train_event("event_006",features)
        final=torch.load(trainer.latest,weights_only=False);trainer.close()
        restored=self.build_arm("S6_reloaded","S",path,export,completed_event=5,resume_state=saved,resume_export=fifth["export_path"])
        repeated=restored.train_event("event_006",features)
        restored.verify_boundary(repeated["export_path"])
        actual=torch.load(restored.latest,weights_only=False)
        for key in ("draft_state_dict","backend","rng","global_step","presentations"):
            self.assertEqual(differences(final[key],actual[key]),[],key)
        for key in ("record_id","pass","sampled_anchors","metrics"):
            self.assertEqual([r[key] for r in uninterrupted["records"]],[r[key] for r in repeated["records"]])
        self.assertEqual((restored.global_step,restored.presentations),(768,1536))
        with self.assertRaises(ValueError):restored.train_event("event_007",features)
        self.assertEqual(differences(source,restored.origin),[])
        self.assertFalse(torch.cuda.is_initialized())

    def test_disposable_private_forks_and_own_next_step_serialization(self):
        if self.subprocess_or_continue():return
        from experiments.functional_step_geometry.qualify import runtime_state
        path,export,source=self.make_parent()
        trainer=self.build_arm("qualification","N",path,export,qualification_only=True)
        features=T.SpecForgeEventTrainer._prepare_event(trainer,"event_005",self.features,2)
        runtime=runtime_state(trainer.bundle.model)
        endpoints=[]
        for label in ("native","factor_one"):
            trainer.reset_qualification(arm="N",runtime=runtime)
            trainer._active_event=label;(trainer.events_root/label).mkdir()
            trainer.bundle.model.begin_event()
            order=torch.randperm(len(features)).tolist()
            forward=trainer._native_forward
            if label=="native":trainer._native_forward=trainer._ordinary_forward
            rows=trainer.run_pair([features[i] for i in order[:2]])
            trainer._native_forward=forward
            endpoints.append(deepcopy(T.native.live_state(trainer)))
        self.assertEqual(differences(*endpoints),[])
        self.assertEqual(differences(source,trainer.origin),[])
        trainer.reset_qualification(arm="S",runtime=runtime)
        trainer._active_event="scaled";(trainer.events_root/"scaled").mkdir()
        trainer.bundle.model.begin_event();order=torch.randperm(len(features)).tolist()
        trainer.run_pair([features[i] for i in order[:2]])
        snapshot=self.root/"temporary.pt";trainer.save_qualification(snapshot)
        retained=torch.load(snapshot,weights_only=False)
        runtime=runtime_state(trainer.bundle.model)
        rows=trainer.run_pair([features[i] for i in order[2:4]],ordinal=2)
        endpoint=deepcopy(T.native.live_state(trainer));work=deepcopy(trainer.work)
        trainer.restore_qualification(snapshot,runtime=runtime)
        self.assertEqual(trainer.work,work)
        repeated=trainer.run_pair([features[i] for i in order[2:4]],ordinal=2)
        self.assertEqual(differences(endpoint,T.native.live_state(trainer)),[])
        for key in ("metrics","sampled_anchors"):
            self.assertEqual([r[key] for r in rows],[r[key] for r in repeated])
        self.assertEqual((trainer.global_step,trainer.presentations),(514,1028))
        self.assertTrue(all(s["step"].item()==513 for s in retained["backend"]["optimizer"]["optimizer_state_dict"]["state"].values()))
        self.assertEqual(differences(source,trainer.origin),[])
        self.assertFalse(torch.cuda.is_initialized())


if __name__=="__main__":unittest.main()
