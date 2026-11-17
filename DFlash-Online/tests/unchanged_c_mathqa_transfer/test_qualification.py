"""Disposable orchestration accounts attempts and preserves failure evidence."""
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from experiments.unchanged_c_mathqa_transfer import qualification as Q
from experiments.unchanged_c_mathqa_transfer import training as T
from experiments.unchanged_c_mathqa_transfer.runtime import initialize_work, read
from experiments.functional_step_geometry.qualify import restore_runtime
from online_feedback.optimizer import DirectBF16AdamW
from online_feedback.rng import capture_rng, restore_rng
import test_mathqa_fixed_lr_transfer_qualification as baseline_fixtures


class TinyTrainer(baseline_fixtures.TinyTrainer):
    instances=[]
    corrupt_cap_zero=False
    fail_adam=False

    def reset_qualification(self, *, relative_cap=.25, runtime=None):
        self.arm="C";self._qualification_cap=relative_cap
        self.backend.optimizer.zero_grad()
        self.load(deepcopy(self.origin))
        self.response_records.clear();self.window_records.clear()
        if runtime is not None:restore_runtime(self.bundle.model,runtime)
        return dict(status="passed",arm="C")

    def checked_forward(self,feature):
        loss,row=self.forward(feature)
        if self.corrupt_cap_zero and self._qualification_cap==0:
            loss=loss*2
            if row is not None:
                row["response_observation"]["loss"]=float(loss.detach())
        return loss,row

    def pr51_native_forward(self,feature):
        assert self._qualification_cap is None
        return self.forward(feature)

    def forward(self,feature):
        if feature is None:return self.bundle.model.weight.sum(),None
        draw=dict(position=int(torch.randint(100,(1,)).item()))
        active=self._qualification_cap==.25
        coefficients=[.5,.25] if active else [.5,-.25]
        loss=(self.bundle.model.weight*self.bundle.model.weight.new_tensor(coefficients)).sum()
        self.bundle.model.cache_marker.add_(1);self.bundle.model._cached_seq_len+=1
        self.work["response_forwards"]=self.work.get("response_forwards",0)+1
        if getattr(self.bundle.model,"block_observer",None) is not None:
            labels=torch.tensor([[1,2,3]])
            self.bundle.model.block_observer(labels,labels,torch.ones(1,3),torch.ones(1,3,dtype=torch.bool))
        summary=(dict(enabled=False) if self._qualification_cap is None else dict(
            changed_positions=2 if active else 0,changed_blocks=int(active),weight_moved=.25 if active else 0.,
            prefix_weight_before=3.,prefix_weight_after=3.,mass_error_abs_sum=0.,delta_numerator=0.,
            mass_violations=0,bounds_violations=0,support_violations=0,nonfinite_values=0))
        sums=dict(weight=2.,main_loss_numerator=float(loss.detach()))
        if self._qualification_cap is not None:sums["prefix_allocation_changed_positions"]=summary["changed_positions"]
        observation=dict(loss=float(loss.detach()),sum_metrics=sums,ratio_metrics={},prefix_allocation=summary)
        row=dict(record_id=feature["record_id"],**{"pass":0},sampled_anchors=[draw],
                 response_observation=observation,metrics={"loss":float(loss.detach())})
        self.response_records.append(observation)
        return loss,row


class QualificationTests(unittest.TestCase):
    def setUp(self):
        before=capture_rng();self.addCleanup(lambda:restore_rng(before))
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);initialize_work(self.root)
        TinyTrainer.instances.clear();TinyTrainer.corrupt_cap_zero=False;TinyTrainer.fail_adam=False
        model=baseline_fixtures.TinyModel();optimizer=DirectBF16AdamW(model)
        saved=optimizer.state_dict()
        for index,p in zip(saved["optimizer_state_dict"]["param_groups"][0]["params"],model.parameters()):
            saved["optimizer_state_dict"]["state"][index]=dict(step=torch.tensor(512.),exp_avg=torch.zeros_like(p),exp_avg_sq=torch.zeros_like(p))
        source=dict(draft_state_dict={name:p.detach().clone() for name,p in model.named_parameters()},
                    backend={"optimizer":saved},rng=capture_rng(),global_step=512,presentations=1024)
        torch.save(source,self.root/"origin.pt")
        (self.root/"config.yaml").write_text("CPU orchestration fixture\n")
        self.features=[dict(record_id=f"r{i}",path=f"feature{i}") for i in range(4)]
        self.plan=dict(root=str(self.root),binding="b"*64,execution_commit="fixture",devices={"A":"GPU-A","B":"GPU-B"},
            configs={"C":str(self.root/"config.yaml")},origin_state=str(self.root/"origin.pt"),
            origin_export="unused",origin_config=str(self.root/"config.yaml"))

    def execute(self,slot="A",startup_override=None):
        def startup(trainer,features,output):
            for _ in range(2):
                loss,_=trainer.strategy.forward_loss(None);trainer.backend.backward(loss/2)
            trainer.backend.optimizer.zero_grad()
            return dict(wall_s=.1,optimizer_steps=0)
        with ExitStack() as stack:
            stack.enter_context(patch.object(T,"MathQACandidateTrainer",TinyTrainer))
            stack.enter_context(patch("experiments.unchanged_c_mathqa_transfer.task_runner.feature_pool",return_value=["unused"]*4))
            stack.enter_context(patch("online_feedback.training.SpecForgeEventTrainer._prepare_event",return_value=self.features))
            stack.enter_context(patch("experiments.prefix_guard.qualify.pin_retained_graph_runtime",return_value={"status":"passed"}))
            stack.enter_context(patch("experiments.functional_step_geometry.native_pilot.discarded_startup",side_effect=startup_override or startup))
            export=stack.enter_context(patch("specforge.export.to_hf.export_to_hf"))
            result=Q.run_training_qualification(self.plan,slot)
            self.assertEqual(export.call_count,1)
            return result

    def test_two_cards_exact_native_cap_zero_active_and_own_resume(self):
        a=self.execute("A");b=self.execute("B")
        self.assertEqual(Q.compare_training_cards(a,b)["status"],"passed")
        for result in (a,b):
            self.assertEqual([row["case"] for row in result["cases"]],list(Q.CASES))
            self.assertTrue(all(row["checks"]=="passed" for row in result["cases"]))
            self.assertTrue(result["actual_intervention"])
            self.assertEqual(result["origin_before"],result["origin_after"])
            self.assertEqual(result["cases"][3]["actual_learning_rate"],3e-4)
            self.assertEqual(result["cases"][5]["endpoint"]["global_step"],514)
        work=read(self.root/"WORK.json")
        self.assertEqual(work["completed"]["adam"],12)
        self.assertEqual(work["completed"]["response_forward"],28)
        self.assertEqual(work["completed"]["response_backward"],28)
        self.assertEqual(work["completed"]["training_initialization"],2)
        self.assertEqual(work["pending"],{})
        self.assertFalse(torch.cuda.is_initialized())

    def test_parity_failure_saves_endpoint_and_does_not_continue(self):
        TinyTrainer.corrupt_cap_zero=True
        with self.assertRaisesRegex(ValueError,"cap_zero equivalence"):self.execute()
        result=read(self.root/"qualification/training_A/DETAILS.json")
        self.assertEqual(result["status"],"failed")
        self.assertEqual(len(result["cases"]),3)
        row=result["cases"][2]
        self.assertEqual(row["status"],"completed")
        self.assertEqual(row["checks"],"pending")
        self.assertIn("preclip",row);self.assertIn("endpoint",row)
        self.assertEqual(read(self.root/"WORK.json")["completed"]["adam"],3)

    def test_startup_and_adam_failures_retain_exact_attempt_accounting(self):
        def broken(trainer,features,output):
            trainer.strategy.forward_loss(None)
            raise RuntimeError("startup before backward")
        with self.assertRaisesRegex(RuntimeError,"startup before backward"):self.execute(startup_override=broken)
        work=read(self.root/"WORK.json")
        self.assertEqual(work["completed"]["response_forward"],1)
        self.assertEqual(work["attempted"]["response_backward"],0)
        self.assertEqual(work["attempted"]["adam"],0)

    def test_failed_adam_keeps_preclip_and_pending_work(self):
        TinyTrainer.fail_adam=True
        with self.assertRaisesRegex(RuntimeError,"injected Adam"):self.execute()
        work=read(self.root/"WORK.json")
        self.assertEqual(work["attempted"]["adam"],1)
        self.assertEqual(work["completed"]["adam"],0)
        self.assertEqual(len(work["pending"]),1)
        result=read(self.root/"qualification/training_A/DETAILS.json")
        self.assertEqual(result["cases"][0]["status"],"attempted")
        self.assertIn("preclip",result["cases"][0])

    def test_cross_card_mismatch_visible_and_no_silent_overwrite(self):
        a=self.execute();b=deepcopy(a)
        b["cases"][3]["endpoint"]["weights"]["weight"]["sha256"]="different"
        result=Q.compare_training_cards(a,b)
        self.assertEqual(result["status"],"failed")
        self.assertTrue(result["same_card_checks_passed"])
        self.assertIn("active: cross-card endpoint",result["errors"])
        with self.assertRaises(FileExistsError):self.execute()

    def test_serving_failure_saved_before_cleanup(self):
        from experiments.unchanged_c_mathqa_transfer import qualification_entry as E
        from online_feedback.storage import write_json
        base=self.root/"qualification/serving_A";base.mkdir(parents=True)
        write_json(self.root/"qualification/training_A/STATUS.json",dict(status="passed",result=dict(export="/temporary_C")))
        self.plan.update(warmups=str(self.root/"warmups.json"),warmup_references=str(self.root/"references.json"))
        write_json(self.plan["warmups"],{"requests":[]});write_json(self.plan["warmup_references"],{})
        backend=SimpleNamespace(loading_s=.1,generate=lambda *a,**k:None,check_weights=lambda checkpoint:{})
        observed=[]
        def close(*unused):
            observed.append(read(base/"RESULT.json"));raise RuntimeError("cleanup failed")
        with patch.object(E,"create_backend",return_value=backend), patch.object(E,"weight_check_errors",return_value=[]), \
             patch("experiments.unchanged_c_mathqa_transfer.evaluation.validate_references"), \
             patch("experiments.unchanged_c_mathqa_transfer.task_runner.verify_reference_seal"), \
             patch.object(E,"evaluate",side_effect=ValueError("original serving mismatch")), patch.object(E,"close_backend",side_effect=close):
            with self.assertRaisesRegex(ValueError,"original serving mismatch"):E.serving(self.plan,"A",base)
        self.assertEqual(observed[0]["status"],"failed")
        saved=read(base/"RESULT.json")
        self.assertIn("original serving mismatch",saved["error"]);self.assertIn("cleanup failed",saved["cleanup_error"])


if __name__=="__main__":unittest.main()
