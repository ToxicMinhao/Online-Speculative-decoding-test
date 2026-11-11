"""Tiny CPU trainer doubles verify state isolation and work separation."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch

from experiments.functional_step_geometry.qualify import scientific_state
from experiments.prefix_guard.qualify import compare_states
from experiments.second_follower.deferred import install_deferred_check
from online_feedback.optimizer import DirectBF16AdamW
from online_feedback.second_follower import SecondFollowerRedistribution
from specforge.runtime.contracts import TrainBatch
from specforge.training.controller import TrainerCore
from specforge.training.strategies.base import StepContext, StepOutput


class TinyDraft(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([.4, -.7, 1.3], dtype=torch.bfloat16))
        self.register_buffer("calls", torch.zeros(()))
        self.first_error_redistribution = SecondFollowerRedistribution()
        self.costs = {"ordinary": 7.}


def trainer_double(root, *, fail_native=False):
    model = TinyDraft()
    t = SimpleNamespace(bundle=SimpleNamespace(model=model, draft_model=model),
                        global_step=512, presentations=1024, response_records=[],
                        window_records=[], work={}, _anchor_observations=[],
                        _presentation_schedule=SimpleNamespace(position=0, checked_draws=0),
                        _active_event="event_005", events_root=Path(root),
                        main_counts=dict(response_forwards=0,response_backwards=0,optimizer_steps=0))

    class Strategy:
        def forward_loss(self, batch, ctx=None):
            if not getattr(t, "_second_follower_diagnostic", False):
                t.main_counts["response_forwards"] += 1
            elif fail_native:
                raise RuntimeError("injected disposable forward failure")
            is_candidate = isinstance(model.first_error_redistribution, SecondFollowerRedistribution)
            active = is_candidate and batch.metadata["active"]
            draw = torch.rand(4).tolist()
            t._anchor_observations.append(dict(positions=[draw], keep_mask=[[True]*4]))
            if t._presentation_schedule is not None:
                t._presentation_schedule.checked_draws += 1
            with torch.no_grad():
                model.calls.add_(1)
            model.costs["ordinary"] += 1
            loss = (model.weight * batch.tensors["x"]).sum()
            if active:
                loss = loss + 2 * model.weight[0]
            t.response_records.append(dict(record_id=batch.sample_ids[0], loss=float(loss.detach()),
                sampled_anchors=deepcopy(t._anchor_observations),
                sum_metrics={"second_follower_transferred_weight": .125 if active else 0.}))
            t.work["response_forwards"] = t.work.get("response_forwards", 0) + 1
            return StepOutput(loss=loss, metrics={}, loss_terms=(loss, loss.detach().new_ones(())))

    class Backend:
        def __init__(self):
            self.module = model
            self.parallel_config = None
            self.optimizer = DirectBF16AdamW(model, learning_rate=.01, weight_decay=.01)
            ordinary = self.optimizer.step
            def observed_step():
                result = ordinary()
                t.window_records.append(dict(global_step=t.global_step+1,
                    record_ids=[r["record_id"] for r in t.response_records[-2:]]))
                return result
            self.optimizer.step = observed_step
            adam = self.optimizer.optimizer.step
            def counted_adam(*args, **kwargs):
                if not getattr(t, "_second_follower_diagnostic", False):
                    t.main_counts["optimizer_steps"] += 1
                return adam(*args, **kwargs)
            self.optimizer.optimizer.step = counted_adam

        def backward(self, loss, *, is_boundary=True):
            if not getattr(t, "_second_follower_diagnostic", False):
                t.main_counts["response_backwards"] += 1
            t.work["response_backwards"] = t.work.get("response_backwards", 0) + 1
            loss.backward()

        def scale_gradients(self, factor):
            with torch.no_grad():
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(factor)

        def step(self):
            return self.optimizer.step()

    t.backend = Backend()
    t.core = TrainerCore(Strategy(), t.backend, accumulation_steps=2)
    t.core._micro = 1024
    return t


def run_responses(trainer, activations):
    results=[]
    for ordinal, active in enumerate(activations):
        batch=TrainBatch(sample_ids=[f"response_{ordinal}"], strategy="dflash",
                         tensors={"x":torch.tensor([-1.,.5,.3],dtype=torch.bfloat16)},
                         metadata={"active":active})
        trainer._anchor_observations.clear()
        trainer._presentation_schedule.checked_draws=0
        result=trainer.core.train_step(batch,StepContext(global_step=trainer.global_step,total_steps=7424))
        assert trainer._presentation_schedule.checked_draws == 1
        trainer._presentation_schedule.position += 1
        result.materialize_metrics()
        trainer.presentations += 1
        trainer.global_step += int(result.optimizer_stepped)
        results.append(result)
    return results


class DeferredTests(unittest.TestCase):
    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def test_first_later_active_window_restores_candidate_and_counts_once(self):
        with tempfile.TemporaryDirectory() as directory:
            outcomes=[]
            for enabled in (False,True):
                torch.manual_seed(17)
                trainer=trainer_double(directory)
                reducer=trainer.bundle.model.first_error_redistribution
                schedule=trainer._presentation_schedule
                handle=install_deferred_check(trainer,Path(directory)/"deferred") if enabled else None
                results=run_responses(trainer,[False,False,True,True,True,True])
                if handle:
                    receipt=handle.close()
                    self.assertEqual(receipt["status"],"completed")
                    self.assertEqual(receipt["windows_examined"],2)
                    self.assertEqual(receipt["record_ids"],["response_2","response_3"])
                    self.assertEqual(receipt["extra_work"]["completed"],dict(response_forwards=2,response_backwards=2,optimizer_steps=1))
                    self.assertTrue(receipt["exact_candidate_restoration"]["exact_equal"])
                    self.assertTrue(receipt["same_inputs_rng"]["exact_equal"])
                    self.assertGreater(receipt["gradient_changes"]["changed"],0)
                    self.assertGreater(receipt["weight_changes"]["changed"],0)
                    self.assertIs(trainer.bundle.model.first_error_redistribution,reducer)
                    self.assertIs(trainer._presentation_schedule,schedule)
                    self.assertFalse(trainer._second_follower_diagnostic)
                    self.assertIs(handle.close(),receipt)
                self.assertEqual(trainer.main_counts,dict(response_forwards=6,response_backwards=6,optimizer_steps=3))
                self.assertEqual((trainer.global_step,trainer.presentations,trainer.core._micro),(515,1030,1030))
                outcomes.append(dict(state=scientific_state(trainer),records=trainer.response_records,
                    windows=trainer.window_records,work=trainer.work,costs=trainer.bundle.model.costs,
                    schedule=vars(trainer._presentation_schedule),metrics=[r.materialize_metrics() for r in results],
                    next_draw=torch.rand(4)))
            self.assertTrue(compare_states(*outcomes)["exact_equal"])

    def test_no_active_window_adds_no_model_work(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer=trainer_double(directory)
            original=trainer.core.train_step
            handle=install_deferred_check(trainer,Path(directory)/"deferred")
            run_responses(trainer,[False]*4)
            receipt=handle.close()
            self.assertEqual(receipt["status"],"no_active_main_window")
            self.assertEqual(receipt["windows_examined"],2)
            self.assertEqual(receipt["extra_work"]["attempts"],dict(response_forwards=0,response_backwards=0,optimizer_steps=0))
            self.assertEqual(trainer.core.train_step,original)
            self.assertFalse((Path(directory)/"deferred/WORK.jsonl").exists())

    def test_disposable_failure_restores_actual_poststate_and_preserves_failure_work(self):
        with tempfile.TemporaryDirectory() as directory:
            torch.manual_seed(31)
            baseline=trainer_double(directory)
            run_responses(baseline,[True,True])
            # Hook restores before the outer second-response counter commit.
            baseline.presentations-=1
            baseline.global_step-=1
            expected=scientific_state(baseline)
            torch.manual_seed(31)
            trainer=trainer_double(directory,fail_native=True)
            handle=install_deferred_check(trainer,Path(directory)/"deferred")
            with self.assertRaisesRegex(RuntimeError,"injected disposable forward failure"):
                run_responses(trainer,[True,True])
            receipt=handle.close()
            self.assertEqual(receipt["status"],"failed")
            self.assertEqual(receipt["extra_work"]["attempts"]["response_forwards"],1)
            self.assertEqual(receipt["extra_work"]["unknown"]["response_forwards"],1)
            self.assertTrue(receipt["exact_candidate_restoration"]["exact_equal"])
            self.assertFalse(trainer._second_follower_diagnostic)
            self.assertTrue(compare_states(scientific_state(trainer),expected)["exact_equal"])
            self.assertEqual(trainer.main_counts,dict(response_forwards=2,response_backwards=2,optimizer_steps=1))

    def test_disabled_reducer_and_partial_windows_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer=trainer_double(directory)
            trainer.bundle.model.first_error_redistribution=SecondFollowerRedistribution(disabled=True)
            with self.assertRaises(ValueError):
                install_deferred_check(trainer,Path(directory)/"disabled")
            trainer.bundle.model.first_error_redistribution=SecondFollowerRedistribution()
            trainer.core._micro+=1
            with self.assertRaises(ValueError):
                install_deferred_check(trainer,Path(directory)/"partial")


if __name__=="__main__":
    unittest.main()
