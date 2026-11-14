"""CPU integration of disposable role windows and persistent native Adam."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import time
import unittest
from unittest.mock import patch

import torch

from experiments.native_prefix_loss.training import NativePrefixTrainer, observe_source_pair
from experiments.native_prefix_loss.qualification import zero_adam_reference, qualify, gradient_role_check
from experiments.native_prefix_loss.state import capture_state, restore_state, verify_state
from experiments.native_prefix_loss.runner import price_remaining, WindowController
from experiments.threatened_repair_replay.work import Ledger, TrainingInstrumentation
from experiments.ok_continuation.resume import differences
from online_feedback.native_prefix_roles import NativePrefixRoles
from online_feedback.optimizer import DirectBF16AdamW
from specforge.training.strategies.base import StepOutput
from test_threat_state import tiny_trainer, first_pair
import test_threat_replay as replay_fixtures


def prefix_fixture(root, *, count=128):
    trainer, features = tiny_trainer(root, count=count)
    trainer.__class__ = NativePrefixTrainer
    trainer.roles = NativePrefixRoles()
    trainer.bundle.model.native_prefix_roles = None

    def forward(batch, ctx=None):
        raw = batch.tensors
        observation = trainer.bundle.model._sample_anchor_positions()
        trainer._anchor_observations.append(observation)
        trainer._presentation_schedule.check_anchors(observation)
        prediction = trainer.bundle.draft_model(raw["x"])
        token_loss = (prediction.float()-raw["y"]).square().reshape(1,1,2)
        role = trainer.bundle.model.native_prefix_roles
        loss_rows = token_loss
        if role is not None:
            role.begin_response()
            loss_rows = role.control_rows(token_loss,
                torch.tensor([[[1,0]]]), torch.tensor([[[1,2]]]),
                torch.ones_like(token_loss), torch.ones_like(token_loss,dtype=torch.bool), chunk_key=0)
        denominator = raw["denominator"]
        numerator = loss_rows.sum()
        loss = numerator / denominator
        if role is not None:
            z = torch.tensor(0.)
            role.finish_response(loss, denominator, dict(kplus_S=denominator,kplus_W1=z,kplus_W2=z,
                kplus_priority1_error_count=z,kplus_priority2_error_count=z))
        return StepOutput(loss=loss,metrics={},loss_terms=(numerator,denominator),
            ratio_metrics=dict(fit=(numerator.detach(),denominator)),sum_metrics=dict(positions=denominator))
    trainer._native_forward = forward
    return trainer, features


class HistoryTests(unittest.TestCase):
    def test_zero_recurrence_matches_both_native_paths_and_preserves_none(self):
        for foreach in (False, True):
            with self.subTest(foreach=foreach):
                torch.manual_seed(17)
                model = torch.nn.Linear(3,2,dtype=torch.bfloat16)
                backend = DirectBF16AdamW(model)
                backend.optimizer.param_groups[0]["foreach"] = foreach
                for parameter in model.parameters():
                    backend.optimizer.state[parameter] = dict(step=torch.tensor(768.),
                        exp_avg=torch.randn_like(parameter)*.2, exp_avg_sq=torch.rand_like(parameter)*.3+.01)
                trainer = SimpleNamespace(bundle=SimpleNamespace(draft_model=model),
                    backend=SimpleNamespace(optimizer=backend,
                        state_dict=lambda: dict(optimizer=backend.state_dict())))
                gradients = dict(weight=torch.ones_like(model.weight), bias=None)
                expected, receipt = zero_adam_reference(trainer, gradients)
                original_bias = deepcopy(backend.optimizer.state[model.bias])
                bias = model.bias.detach().clone()
                model.weight.grad = torch.zeros_like(model.weight)
                model.bias.grad = None
                backend.step()
                self.assertEqual(differences(model.state_dict(), expected["draft_state_dict"]), [])
                self.assertEqual(differences(dict(optimizer=backend.state_dict()),expected["backend"]), [])
                self.assertTrue(torch.equal(bias,model.bias))
                self.assertEqual(differences(backend.optimizer.state[model.bias],original_bias), [])
                self.assertEqual(receipt["optimizer_applications"],0)

    def test_h_consumes_one_step_no_presentations_or_sampling_and_restores(self):
        with tempfile.TemporaryDirectory() as root:
            trainer,features = prefix_fixture(root,count=4)
            for value in trainer.backend.optimizer.optimizer.state.values():
                value["exp_avg"].fill_(1.)
            pair=first_pair(trainer,features)
            pre=capture_state(trainer)
            native=trainer.run_pair(pair,0,0,role="N",capture_gradients=True)
            restore_state(trainer,pre)
            expected,_=zero_adam_reference(trainer,native["preclip_gradients"]["native"])
            result=trainer.run_history_step(native["preclip_gradients"]["native"],window=1)
            self.assertEqual(result["update"]["optimizer_applications"],1)
            self.assertEqual(trainer.presentations,1536)
            self.assertEqual(trainer.core._micro,1536)
            self.assertEqual(trainer.global_step,769)
            self.assertEqual(trainer._presentation_schedule.position,0)
            self.assertEqual(differences(trainer.bundle.draft_model.state_dict(),expected["draft_state_dict"]),[])
            self.assertGreater(sum(int(not torch.equal(pre["scientific"]["draft_state_dict"][name],value))
                for name,value in expected["draft_state_dict"].items()),0)
            restore_state(trainer,pre)
            verify_state(trainer,pre)


class QualificationTests(unittest.TestCase):
    def test_six_case_qualification_and_complete_origin_restoration(self):
        with tempfile.TemporaryDirectory() as root:
            trainer,features=prefix_fixture(Path(root)/"trainer")
            before=capture_state(trainer)
            result=qualify(trainer,[Path(row["path"]) for row in features],Path(root)/"qualification")
            self.assertEqual(result["status"],"passed")
            self.assertEqual(result["optimizer_steps"],6)
            self.assertEqual(result["native_response_presentations"],10)
            self.assertEqual(result["response_forwards"],10)
            verify_state(trainer,before)
            self.assertFalse(list((Path(root)/"qualification").rglob("*.pt")))

    def test_active_role_support_failure_is_not_called_a_pass(self):
        receipt=dict(responses=[dict(chunks=[dict(prefix_occurrences=2,error_occurrences=0,
            gradient=dict(finite=True,excluded_nonzero=0,prefix_nonzero=2,error_nonzero=0))])])
        result=gradient_role_check(receipt,"E")
        self.assertEqual(result["status"],"failed")
        self.assertIn("fixed_pair_lacks_both_required_roles",result["failures"])

    def test_forward_failure_still_saves_role_evidence(self):
        with tempfile.TemporaryDirectory() as root:
            trainer,features=prefix_fixture(root,count=4)
            pair=first_pair(trainer,features)
            def failing(*_args,**_kwargs):
                trainer.roles.begin_response()
                raise ValueError("fixture failure")
            trainer._native_forward=failing
            with self.assertRaisesRegex(ValueError,"fixture failure"):
                trainer.run_pair(pair,0,0,role="E",label="E",category="qualification")
            path=Path(root)/"role_observations/qualification_001_E.json"
            self.assertTrue(path.exists())
            self.assertEqual(json.loads(path.read_text())["role"],"E")

    def test_duration_admission_does_not_shrink_scope(self):
        qual=dict(cases=[dict(case=case,update=dict(wall_s=2. if case!="H" else .2))
                         for case in ("N","ALL","E","C","H","N_repeat")])
        plan=dict(remaining_probe_estimate_s=3600.)
        full=price_remaining(plan,qual,time.monotonic()+10000)
        blocked=price_remaining(plan,qual,time.monotonic()+10)
        self.assertEqual(full["status"],"passed")
        self.assertEqual(blocked["status"],"blocked")
        self.assertEqual(full["estimated_remaining_s"],blocked["estimated_remaining_s"])


class ObserverControllerTests(unittest.TestCase):
    def test_source_observer_replays_full_shapes_and_restores_state(self):
        from online_feedback.rng import capture_rng
        with tempfile.TemporaryDirectory() as folder:
            fixture=replay_fixtures.ReplayIntegrationTests()
            model,trainer,raw,layout,_=fixture.saved_capture_fixture(folder)
            trainer.work={"original":3}
            binding=deepcopy(layout["binding"])
            binding["sampled_anchors"][0]["keep_mask"][0][-1]=False
            trainer.schedule=dict(records=[binding,dict(deepcopy(binding),presentation_id=1)])
            feature=dict(record_id="capture",path=layout["feature_path"])
            model.train()
            original_grad=model.hidden.grad.clone()
            rng=capture_rng()
            shapes=[]
            handle=model.lm_head.register_forward_pre_hook(lambda _m, inputs:shapes.append(tuple(inputs[0].shape)))
            try:
                rows,work=observe_source_pair(trainer,[feature,feature],0,0,window=8,state="P")
            finally:
                handle.remove()
            self.assertEqual(shapes,[(1,8,16,4),(1,1,16,4)]*2)
            self.assertEqual(len(rows),18)
            self.assertEqual(sum(row["eligible_block"] for row in rows),16)
            self.assertEqual(sum(row["response_forwards"] for row in work),2)
            self.assertEqual(trainer.work,{"original":3})
            self.assertEqual(model.costs,{"existing":9.})
            self.assertEqual(model._cached_seq_len,17)
            self.assertTrue(model.training)
            self.assertTrue(torch.equal(model.hidden.grad,original_grad))
            self.assertEqual(differences(rng,capture_rng()),[])

    def test_complete_window_has_five_states_four_steps_and_only_native_continues(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            trainer,features=prefix_fixture(root/"trainer")
            order=torch.randperm(128).tolist()
            for ordinal in range(0,14,2):
                trainer.run_pair([features[index] for index in order[ordinal:ordinal+2]],0,ordinal)
            pair=[features[index] for index in order[14:16]]
            origin=root/"origin.pt"; origin.write_bytes(b"fixture")
            ledger=Ledger(root/"ledger.jsonl",limits={"optimizer_step":4})
            trainer.ledger=ledger
            instrument=TrainingInstrumentation(trainer,ledger)
            def observer(current,pair,pass_index,ordinal,*,window,state,ledger):
                return [dict(window=window,state=state,request_id=feature["record_id"],
                    accepted_prefix=2,valid_proposals=3,first_error=3,valid=True,
                    canonical_start=0,anchor_ordinal=0) for feature in pair], [dict(response_forwards=1) for _ in pair]
            def exporter(weights,template,destination):
                destination.mkdir(parents=True)
                return dict(checkpoint=str(destination),export_s=0.,bytes=0)
            controller=WindowController(dict(origin_state=str(origin),origin_export=folder),root/"run",ledger,
                observer=observer,exporter=exporter,admit=lambda _:dict(status="fixture"))
            controller.output.mkdir()
            try:
                result=controller(trainer,8,pair,0,14)
                self.assertEqual(ledger.completed["optimizer_step"],4)
                self.assertEqual(len(controller.probe_exports),5)
                self.assertEqual({row["state"] for row in controller.probe_exports},{"P","N","E","C","H"})
                self.assertEqual(len(controller.source_work),10)
                self.assertEqual(trainer.global_step,776)
                self.assertEqual(trainer.presentations,1552)
                self.assertEqual(trainer._presentation_schedule.position,16)
                self.assertEqual(len(trainer.response_records),16)
                self.assertEqual(len(trainer.window_records),8)
                self.assertEqual(result["preclip_gradients"],{})
                self.assertTrue(all(row["status"]=="passed" for row in controller.parity))
            finally:
                instrument.close();ledger.close()


if __name__ == "__main__":
    unittest.main()
