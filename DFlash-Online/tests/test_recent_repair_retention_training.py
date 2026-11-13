"""Pure schedule/state gates and native-shaped frozen margin observations."""
from copy import deepcopy
import unittest

import torch

from experiments.recent_repair_retention.reference import slot_measurements, validate_schedule
from experiments.recent_repair_retention.training import (
    assignment_index, continuation_contract, difference_norm, schedule_adapter, native_view,
)
from experiments.recent_repair_retention.qualification import QUALIFICATION_BUDGET
from online_feedback.rng import capture_rng
from experiments.ok_continuation.resume import differences


def schedule():
    return dict(passes=2, records=[dict(presentation_id=i, pass_index=i//128,
        record_id=f"r{i%128:03d}", sampled_anchors=[dict(positions=[[7]],keep_mask=[[True]])])
        for i in range(256)])


class ScheduleTests(unittest.TestCase):
    def test_translation_checks_but_never_draws_rng(self):
        s=schedule(); before=deepcopy(s); rng=capture_rng()
        features=[dict(record_id=f"r{i:03d}") for i in range(128)]
        result=schedule_adapter(s,features)
        self.assertEqual(result["records"][128]["pass"],1)
        self.assertEqual(result["records"][128]["sampled_anchors"],s["records"][128]["sampled_anchors"])
        self.assertEqual(s,before)
        self.assertEqual(differences(rng,capture_rng()),[])

    def test_duplicate_or_changed_pass_fails(self):
        ids=[f"r{i:03d}" for i in range(128)]
        bad=schedule();bad["records"][128]["record_id"]="r001"
        with self.assertRaises(ValueError): validate_schedule(bad,ids)
        bad=schedule();bad["records"][128]["pass_index"]=0
        with self.assertRaises(ValueError): validate_schedule(bad,ids)

    def test_assignment_is_presentation_aligned_and_copied(self):
        row=dict(presentation_id=2,margin=.25,weight=.5,canonical_start=4,proposal_slot=3)
        original=deepcopy(row)
        index=assignment_index(dict(arms=dict(RECENT=[row])),"RECENT")
        self.assertEqual(index[2][0]["parent_margin"],.25)
        self.assertEqual(index[2][0]["position_weight"],.5)
        self.assertEqual(index[0],[])
        self.assertEqual(row,original)
        self.assertEqual(sum(map(len,assignment_index({},"NATIVE").values())),0)

    def test_temporary_work_is_separate(self):
        self.assertEqual(QUALIFICATION_BUDGET["temporary_optimizer_steps"],3)
        self.assertEqual(QUALIFICATION_BUDGET["generations"],0)
        self.assertEqual(sum(QUALIFICATION_BUDGET[k] for k in QUALIFICATION_BUDGET if k.endswith("forwards")),14)


class ReferenceTests(unittest.TestCase):
    def fixture(self):
        labels=torch.tensor([[[0,1,2,3,4]]])
        logits=torch.full((1,1,5,6),-3.,dtype=torch.bfloat16)
        logits.scatter_(-1,labels.unsqueeze(-1),2.)
        weights=torch.tensor([[[0.,1.,.75,.5,.25]]],dtype=torch.float32)
        return logits,labels,weights

    def observe(self,logits,labels,weights,**kwargs):
        return slot_measurements(logits,labels,weights,anchors=torch.tensor([[10]]),keep=torch.tensor([[True]]),
            prompt_length=7,request_id="r",presentation_id=0,pass_index=0,**kwargs)

    def test_strongest_rival_and_response_offset(self):
        logits,labels,weights=self.fixture();logits[0,0,2,5]=1.75
        rows=self.observe(logits,labels,weights)
        self.assertEqual(rows[1]["margin"],.25)
        self.assertEqual(rows[1]["canonical_start"],3)
        self.assertEqual(rows[1]["rival_token"],5)
        self.assertEqual(rows[1]["correct_prefix"],4)
        self.assertTrue(rows[1]["label_match"])

    def test_wrong_tie_and_earlier_error_do_not_extend_prefix(self):
        logits,labels,weights=self.fixture();logits[0,0,2,0]=2.
        rows=self.observe(logits,labels,weights)
        self.assertEqual(rows[1]["margin"],0)
        self.assertFalse(rows[1]["native_correct"])
        self.assertEqual(rows[1]["correct_prefix"],1)
        self.assertTrue(rows[2]["native_correct"])
        self.assertEqual(rows[2]["correct_prefix"],1)

    def test_eos_is_valid_and_later_slots_invalid(self):
        logits,labels,weights=self.fixture()
        rows=self.observe(logits,labels,weights,eos_token_ids=(2,))
        self.assertTrue(rows[1]["valid"])
        self.assertFalse(rows[2]["valid"])
        self.assertEqual(rows[0]["correct_prefix"],2)

    def test_private_label_mismatch_fails_and_candidate_keys_are_exact(self):
        logits,labels,weights=self.fixture()
        key=("r",3,2)
        rows=self.observe(logits,labels,weights,candidates={key},expected_tokens={key:2})
        self.assertEqual(len(rows),1)
        with self.assertRaises(ValueError):
            self.observe(logits,labels,weights,candidates={key},expected_tokens={key:4})
        self.assertEqual(self.observe(logits,labels,weights,candidates={("r",4,2)}),[])

    def test_seed_never_emitted_and_padding_invalid(self):
        logits,labels,weights=self.fixture();weights[...,3:]=0
        rows=self.observe(logits,labels,weights)
        self.assertEqual([r["proposal_slot"] for r in rows],[1,2,3,4])
        self.assertEqual([r["valid"] for r in rows],[True,True,False,False])

    def test_difference_norm_counts_actual_bfloat_entries(self):
        a={"x":torch.tensor([1.,2.],dtype=torch.bfloat16)}
        b={"x":torch.tensor([1.,3.],dtype=torch.bfloat16)}
        result=difference_norm(a,b)
        self.assertEqual(result["difference_norm"],1.)
        self.assertEqual(result["different_entries"],1)



class CompleteStateTests(unittest.TestCase):
    def origin(self):
        from test_role_loss_campaign_training import pilot_state
        from experiments.functional_step_geometry.contract import experiment_contract,REALIZATION_R1
        from experiments.role_loss_pilot.training import CONTRACT_KEY
        from experiments.functional_step_geometry.training import CONTRACT_KEY as FSG
        from online_feedback.kplus import kplus_contract
        state,kwargs=pilot_state(6,"KPLUS","gsm8k")
        state["algorithm_contract"][FSG]=experiment_contract("b"*64,"KPLUS","gsm8k",
            origin_binding=state["algorithm_contract"][CONTRACT_KEY]["binding"],realization_revision=REALIZATION_R1)
        for row in state["events"][4:]:
            row.pop("pilot_loss")
            row["kplus_loss"]=kplus_contract()
        return state,kwargs["original_config_text"]

    def test_full_parent_validation_is_read_only(self):
        from experiments.recent_repair_retention.training import validate_origin
        state,text=self.origin();before=deepcopy(state);rng=capture_rng()
        value=validate_origin(state,text,expected_cuda_rng_states=0)
        self.assertEqual(value["start_step"],768)
        self.assertEqual(differences(state,before),[])
        self.assertEqual(differences(capture_rng(),rng),[])

    def test_missing_adam_or_foreign_geometry_fails(self):
        from experiments.recent_repair_retention.training import validate_origin,geometry
        state,text=self.origin()
        state["algorithm_contract"][geometry.CONTRACT_KEY]["arm"]="FSG"
        with self.assertRaises(ValueError): validate_origin(state,text,expected_cuda_rng_states=0)
        state,text=self.origin()
        state["backend"]["optimizer"]["optimizer_state_dict"]["state"].clear()
        with self.assertRaises(ValueError): validate_origin(state,text,expected_cuda_rng_states=0)


class TinyContinuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import os
        if os.environ.get("RETENTION_TINY_CHILD")=="1":
            from test_full_k_training import FullKTinyIntegrationTests
            FullKTinyIntegrationTests.setUpClass.__func__(cls)

    @classmethod
    def tearDownClass(cls):
        import os
        if os.environ.get("RETENTION_TINY_CHILD")=="1":
            from test_full_k_training import FullKTinyIntegrationTests
            FullKTinyIntegrationTests.tearDownClass.__func__(cls)

    def setUp(self):
        import os
        if os.environ.get("RETENTION_TINY_CHILD")=="1":
            from test_full_k_training import FullKTinyIntegrationTests
            FullKTinyIntegrationTests.setUp(self)

    def test_complete_empty_tag_continuation_is_exact_native(self):
        import os,sys,subprocess
        from pathlib import Path
        if os.environ.get("RETENTION_TINY_CHILD")!="1":
            root=Path(__file__).resolve().parents[1]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES="",RETENTION_TINY_CHILD="1")
            env["PYTHONPATH"]=os.pathsep.join(str(root/p) for p in (".","tests","training","inference"))
            result=subprocess.run([sys.executable,"-m","unittest","-v",__name__+".TinyContinuationTests."+self._testMethodName],cwd=root,env=env,text=True,capture_output=True,timeout=120)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            return
        from test_full_k_training import FullKTinyIntegrationTests
        from experiments.recent_repair_retention import training as T
        from experiments.lost_repair.training_probe import sample_schedule
        from online_feedback.kplus import install_kplus
        from online_feedback.storage import save_torch
        from safetensors.torch import save_file
        base=FullKTinyIntegrationTests.build(self,"base")
        install_kplus(base.bundle.model)
        state=deepcopy(T.native.live_state(base))
        fake,_=CompleteStateTests().origin()
        state.update(events=fake["events"],global_step=768,presentations=1536)
        state["algorithm_contract"].pop(T.native.CONTRACT_KEY)
        state["algorithm_contract"].update({k:v for k,v in fake["algorithm_contract"].items() if k in (T.historical.CONTRACT_KEY,T.geometry.CONTRACT_KEY)})
        optimizer=state["backend"]["optimizer"]["optimizer_state_dict"]
        for index,p in zip(optimizer["param_groups"][0]["params"],base.backend.optimizer.model_params):
            optimizer["state"][index]=dict(step=torch.tensor(768.),exp_avg=torch.zeros_like(p),exp_avg_sq=torch.zeros_like(p))
        schedule=sample_schedule(base.bundle.model,self.features,state["rng"],max_length=base.max_length)
        schedule.pop("final_rng")
        origin=self.root/"origin.pt";save_torch(origin,state)
        export=self.root/"parent_export";export.mkdir();save_file(state["draft_state_dict"],export/"model.safetensors")
        base.close()
        endpoints=[]
        for arm in ("NATIVE","RECENT"):
            trainer=T.RecentRetentionTrainer(self.config,self.root/arm,arm=arm,binding="c"*64,
                origin_state=origin,origin_export=export,original_config_text=self.original_config_text,
                schedule=schedule,assignments={"arms":{"RECENT":[]}},expected_cuda_rng_states=0)
            self.addCleanup(trainer.close)
            if arm=="RECENT":
                trainer.install_retention()
                # Exercise reversible first-window machinery without inventing
                # a claimed natural hinge in this empty-tag CPU fixture.
                trainer.window_fires=lambda: len(trainer.response_records)==4
            result=trainer.train_event("event_007",self.features)
            self.assertEqual(result["optimizer_steps"],128)
            self.assertEqual(result["end_step"],896)
            if arm=="NATIVE":
                self.assertIsNone(trainer.firing_capture)
            else:
                replay=T.first_firing_replays(trainer,self.root/"synthetic_capture_replay")
                self.assertEqual(replay["status"],"passed")
                self.assertEqual(replay["temporary_optimizer_steps"],2)
                self.assertEqual(replay["bf16_weights"]["different_entries"],0)
            trainer.verify_boundary(result["checkpoint_path"])
            endpoints.append(deepcopy(T.native.live_state(trainer)))
            trainer.close()
        for key in ("draft_state_dict","backend","rng","global_step","presentations"):
            self.assertEqual(differences(endpoints[0][key],endpoints[1][key]),[],key)


if __name__ == "__main__":
    unittest.main()
