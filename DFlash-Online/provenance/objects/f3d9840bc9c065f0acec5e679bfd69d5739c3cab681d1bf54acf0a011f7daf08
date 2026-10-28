"""CPU controls for exact, fail-closed qualification receipts and state isolation."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

from experiments.prefix_guard import qualify as Q
from online_feedback.optimizer import DirectBF16AdamW
from online_feedback.rng import capture_rng


def fixture_trainer():
    model = torch.nn.Linear(2, 1, bias=False).to(torch.bfloat16)
    model.weight.data.fill_(.05)
    optimizer = DirectBF16AdamW(model)
    backend = SimpleNamespace(optimizer=optimizer, backward=lambda loss, **kw: loss.backward())
    core = SimpleNamespace(_micro=0, _ratio_totals={}, _sum_totals={}, accumulation_remainder=0)
    return SimpleNamespace(bundle=SimpleNamespace(model=model, draft_model=model), backend=backend,
                           global_step=0, presentations=0, core=core, events=[],
                           _presentation_schedule=None, _anchor_observations=[])


class QualificationTests(unittest.TestCase):
    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def test_declared_budget_has_three_steps_and_no_optimizer_warmup(self):
        self.assertEqual(Q.TRIALS, (("K", None), ("G_disabled", False), ("G_enabled", True)))
        self.assertEqual(Q.PLAN["temporary_optimizer_steps"], 3)
        self.assertEqual(Q.PLAN["response_presentations"], 6)
        self.assertEqual(Q.PLAN["startup_response_evaluations"], 2)
        self.assertEqual(Q.PLAN["startup_optimizer_steps"], 0)
        self.assertEqual(Q.PLAN["generations"], 0)

    def test_donation_pin_precedes_startup_and_is_recorded_on_failure(self):
        import torch._functorch.config as autograd_config
        with TemporaryDirectory() as directory, mock.patch.object(autograd_config, "donated_buffer", True):
            root = Path(directory)
            receipt = root / "receipt.json"
            receipt.write_text(json.dumps(dict(features=[{}] * 128, records=[{}] * 256)))
            pin = root / "pin.pt"
            torch.save({}, pin)
            error = RuntimeError("stop at startup fixture")

            def startup(*args):
                self.assertIs(autograd_config.donated_buffer, False)
                raise error

            with mock.patch.object(Q, "run_startup", side_effect=startup), self.assertRaises(RuntimeError) as caught:
                Q.run_qualification("fixture", receipt, pin, root / "qualification")
            self.assertIs(caught.exception, error)
            result = json.loads((root / "qualification/RESULT.json").read_text())
            settings = result["runtime_compatibility"]
            self.assertIs(settings["before"], True)
            self.assertIs(settings["requested"], False)
            self.assertIs(settings["effective"], False)
            self.assertIs(settings["effective_at_exit"], False)
            self.assertEqual(result["status"], "failed")

    def test_failed_trial_persists_status_counters_and_keeps_original_error(self):
        import torch._functorch.config as autograd_config
        with TemporaryDirectory() as directory, mock.patch.object(autograd_config, "donated_buffer", True):
            root = Path(directory)
            receipt = root / "receipt.json"
            receipt.write_text(json.dumps(dict(features=[{}] * 128, records=[{}] * 256)))
            pin = root / "pin.pt"
            torch.save({}, pin)
            trainer = fixture_trainer()
            trainer.bundle.model.begin_event = lambda: None
            trainer.bundle.model.reference = None
            trainer.close = mock.Mock(side_effect=OSError("secondary cleanup failure"))
            failure = RuntimeError("original backward failure")
            from contextlib import nullcontext
            with mock.patch.object(Q, "run_startup", return_value=dict(status="passed")), \
                    mock.patch.object(Q, "SpecForgeEventTrainer", return_value=trainer), \
                    mock.patch.object(Q, "prepare_main"), \
                    mock.patch.object(Q, "observe_forwards", return_value=nullcontext()), \
                    mock.patch.object(Q, "run_window", side_effect=failure), \
                    self.assertRaises(RuntimeError) as caught:
                Q.run_qualification("fixture", receipt, pin, root / "qualification")
            self.assertIs(caught.exception, failure)
            result = json.loads((root / "qualification/RESULT.json").read_text())
            trial = json.loads((root / "qualification/K/TRIAL.json").read_text())
            self.assertEqual(trial["status"], "failed")
            self.assertEqual(trial["trainer_counters"]["global_step"], 0)
            self.assertEqual(trial["peak_cuda_allocated_bytes"], 0)
            self.assertEqual(result["error"], "original backward failure")
            self.assertIn("secondary cleanup failure", result["diagnostic_errors"][0])
            self.assertTrue(trainer.failed)

    def test_failed_H_attempt_and_memory_are_present_in_trial_diagnostics(self):
        guard_report = dict(work=dict(h_backward_attempted=1, h_backward_completed=0, h_backward_failed=1))
        guard = SimpleNamespace(report=lambda: deepcopy(guard_report), work=guard_report["work"],
                                forward_calls={"h_backward:layer": 1}, costs={}, step_records=[], response_records=[])
        with TemporaryDirectory() as directory:
            trial = {}
            with mock.patch("torch.cuda.is_initialized", return_value=True), \
                    mock.patch("torch.cuda.max_memory_allocated", return_value=123), \
                    mock.patch("torch.cuda.max_memory_reserved", return_value=456):
                Q.trial_diagnostics(trial, fixture_trainer(), guard, Q.WorkLog(directory))
            self.assertEqual(trial["guard_report"], guard_report)
            self.assertEqual(trial["guard_work"]["h_backward_failed"], 1)
            self.assertEqual(trial["peak_cuda_allocated_bytes"], 123)
            self.assertEqual(trial["peak_cuda_reserved_bytes"], 456)

    def test_compiled_ordinary_backward_then_retained_H_and_K(self):
        """Exercise the native failure ordering with CPU Inductor, not eager."""
        import torch._functorch.config as autograd_config

        def forward(x, weight):
            hidden = torch.sin(x @ weight)
            return hidden @ weight.T

        with TemporaryDirectory() as directory, mock.patch.object(autograd_config, "donated_buffer", True):
            Q.pin_retained_graph_runtime(directory)
            torch._dynamo.reset()
            compiled = torch.compile(forward, backend="inductor", fullgraph=True)
            x = torch.linspace(-.5, .5, 64).reshape(8, 8).requires_grad_()
            weight = torch.linspace(-.2, .2, 64).reshape(8, 8).requires_grad_()
            # The ordinary backward must happen first so a cached backward is reused.
            compiled(x, weight).sum().backward()
            x.grad = weight.grad = None
            output = compiled(x, weight)
            h = torch.autograd.grad(output.square().mean(), (x, weight), retain_graph=True)
            self.assertIsNone(x.grad)
            self.assertIsNone(weight.grad)
            output.sum().backward()
            k = (x.grad.clone(), weight.grad.clone())
            eager_x = x.detach().clone().requires_grad_()
            eager_w = weight.detach().clone().requires_grad_()
            eager = forward(eager_x, eager_w)
            expected_h = torch.autograd.grad(eager.square().mean(), (eager_x, eager_w), retain_graph=True)
            eager.sum().backward()
            torch.testing.assert_close(h, expected_h)
            torch.testing.assert_close(k, (eager_x.grad, eager_w.grad))
            Q.require_runtime_pin()
            torch._dynamo.reset()

    def test_failing_comparison_keeps_magnitude_samples_and_raises(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "failed.json"
            a, b = torch.tensor([1., 3.]), torch.tensor([1., 2.])
            with self.assertRaisesRegex(AssertionError, "Exact qualification gate failed"):
                Q.persist_comparison(path, {"gradient": a}, {"gradient": b})
            report = json.loads(path.read_text())
            row, = report["tensor_differences"]
            self.assertFalse(report["exact_equal"])
            self.assertEqual(row["differing_coordinates"], 1)
            self.assertEqual(row["finite_max_absolute_error"], 1.)
            self.assertEqual(row["samples"], [dict(index=1, left=3., right=2., error=1.)])

    def test_nonfinite_is_failure_even_when_same_tensor(self):
        for value in (float("inf"), float("nan"), -float("inf")):
            tensor = torch.tensor([value])
            result = Q.compare_states(tensor, tensor)
            self.assertFalse(result["exact_equal"])
            self.assertEqual(result["tensor_differences"][0]["nonfinite_coordinates"], 1)
            json.dumps(result, allow_nan=False)

    def test_nested_rng_dtype_and_empty_container_differences_are_checked(self):
        state = dict(rng=capture_rng(), empty={}, rows=(1, None))
        self.assertTrue(Q.compare_states(state, deepcopy(state))["exact_equal"])
        changed = deepcopy(state)
        changed["rows"] = [1, None]
        self.assertFalse(Q.compare_states(state, changed)["exact_equal"])
        self.assertFalse(Q.compare_states(torch.ones(2), torch.ones(2).double())["exact_equal"])
        self.assertTrue(Q.compare_states(np.arange(3, dtype=np.uint32), np.arange(3, dtype=np.uint32))["exact_equal"])

    def test_prepare_main_restores_rng_without_changing_parameters_or_adam(self):
        trainer = fixture_trainer()
        pin = Q.scientific_state(trainer)
        torch.rand(3)
        with TemporaryDirectory() as directory:
            result = Q.prepare_main(trainer, pin, directory)
            self.assertTrue(result["fresh_optimizer"])
            self.assertTrue(Q.compare_states(Q.scientific_state(trainer), pin)["exact_equal"])
            self.assertEqual(trainer.backend.optimizer.optimizer.state, {})
            trainer.bundle.draft_model.weight.data.add_(1)
            with self.assertRaises(AssertionError):
                Q.prepare_main(trainer, pin, directory)

    def test_observer_captures_native_after_clip_and_before_correction(self):
        with TemporaryDirectory() as directory:
            trainer = fixture_trainer()
            root = Path(directory)
            log = Q.WorkLog(root)
            observer = Q.ExactObserver(trainer, root, log)
            with observer.installed():
                trainer.backend.backward(trainer.bundle.model.weight.float().sum() * 4)
                trainer.backend.backward(trainer.bundle.model.weight.float().sum() * 3)
                trainer.backend.optimizer.step()
                native_weight = trainer.bundle.model.weight.detach().clone()
                trainer.bundle.model.weight.data.add_(.5)
            self.assertEqual(log.counts["temporary_optimizer_step_completed"], 1)
            self.assertEqual(log.counts["qualification_k_backward_completed"], 2)
            pre = observer.saved["pre_clip_gradients"]["gradients"]["weight"]
            post = observer.saved["post_clip_gradients"]["gradients"]["weight"]
            self.assertGreater(float(pre.norm()), float(post.norm()))
            self.assertTrue(torch.equal(observer.saved["native_post_adam"]["draft"]["weight"], native_weight))
            self.assertEqual(len(observer.saved["native_post_adam"]["optimizer"]["optimizer_state_dict"]["state"]), 1)

    def test_changed_gradient_stops_before_optimizer_and_preserves_receipt(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            trainer = fixture_trainer()
            expected = dict(after_k_backward_0=dict(
                gradients={"weight": torch.zeros_like(trainer.bundle.model.weight)}, rng=capture_rng()))
            observer = Q.ExactObserver(trainer, root, Q.WorkLog(root), expected)
            with observer.installed(), self.assertRaises(AssertionError):
                trainer.backend.backward(trainer.bundle.model.weight.float().sum())
            self.assertTrue((root / "after_k_backward_0.json").is_file())
            self.assertEqual(trainer.backend.optimizer.optimizer.state, {})


class TinyTrainerQualificationTests(unittest.TestCase):
    """Use the actual response strategy/optimizer with the existing CPU fixture."""
    @classmethod
    def setUpClass(cls):
        import test_osd_recipe as legacy
        cls.legacy = legacy.OSDRecipeTests
        cls.legacy.setUpClass.__func__(cls)

    @classmethod
    def tearDownClass(cls):
        cls.legacy.tearDownClass.__func__(cls)

    def setUp(self):
        from online_feedback import training
        self.legacy.setUp(self)
        patch = mock.patch.object(training, "ranking_eos_token_ids", return_value=(28,))
        patch.start()
        self.addCleanup(patch.stop)

    def trainer(self, *args, **kwargs):
        return self.legacy.trainer(self, *args, **kwargs)

    def test_startup_evaluations_restore_exact_state_without_optimizer_calls(self):
        reference = self.trainer("reference", objective="osd_first_error")
        pin = Q.scientific_state(reference)
        receipt = reference.train_event("fixture", self.features[:2])
        current = self.trainer("startup", objective="osd_first_error")
        root = self.root / "startup_evidence"
        Q.prepare_main(current, pin, root / "before")
        log = Q.WorkLog(root)
        with mock.patch.object(current.backend.optimizer.optimizer, "step",
                               side_effect=AssertionError("Startup must never update Adam")), \
                Q.observe_forwards(current, log):
            Q.warm_without_optimizer(current, receipt, pin, log, root)
        self.assertTrue(Q.compare_states(Q.scientific_state(current), pin)["exact_equal"])
        self.assertEqual(log.counts["startup_backward_completed"], 2)
        self.assertEqual(log.counts["startup_response_evaluation_completed"], 2)
        self.assertEqual(log.counts["response_model_forwards"], 2)
        self.assertGreater(log.counts["head_forwards_including_recompute"], 2)
        self.assertTrue(all(value is None for value in Q.named_gradients(current).values()))

    def test_startup_runtime_is_closed_before_constructing_fresh_K(self):
        reference = self.trainer("reference", objective="osd_first_error")
        pin = Q.scientific_state(reference)
        receipt = reference.train_event("fixture", self.features[:2])
        created = []

        def construct(*args, **kwargs):
            trainer = self.trainer("disposable_startup", objective="osd_first_error")
            trainer.backend.optimizer.optimizer.step = mock.Mock(
                side_effect=AssertionError("Startup must not call Adam"))
            created.append(trainer)
            return trainer

        root = self.root / "disposable_evidence"
        with mock.patch.object(Q, "SpecForgeEventTrainer", side_effect=construct):
            result = Q.run_startup(self.config, receipt, pin, Q.WorkLog(root), root)
        self.assertEqual(len(created), 1)
        self.assertTrue(created[0].closed)
        self.assertFalse(hasattr(created[0], "bundle"))
        self.assertTrue(result["runtime_closed_before_K_construction"])
        self.assertEqual(result["optimizer_steps"], 0)
        ordinary = self.trainer("fresh_K", objective="osd_first_error")
        self.assertIsNot(ordinary, created[0])
        self.assertEqual(ordinary.global_step, 0)
        self.assertEqual(ordinary.backend.optimizer.optimizer.state, {})

    def test_real_guard_wrappers_preserve_all_native_boundaries(self):
        from contextlib import nullcontext
        from online_feedback.prefix_guard import install_guard

        reference = self.trainer("reference", objective="osd_first_error")
        pin = Q.scientific_state(reference)
        receipt = reference.train_event("fixture", self.features[:2])
        baseline = None
        log = Q.WorkLog(self.root / "trials")
        for label, enabled in Q.TRIALS:
            trainer = self.trainer(label, objective="osd_first_error")
            root = self.root / "trials" / label
            Q.prepare_main(trainer, pin, root / "initial")
            trainer.bundle.model.begin_event()
            guard = nullcontext(None) if enabled is None else install_guard(trainer, enabled=enabled, probe_steps=())
            observer = Q.ExactObserver(trainer, root, log, baseline)
            with guard, observer.installed():
                rows = list(Q.run_window(trainer, receipt, log))
            self.assertEqual(len(rows), 2)
            if enabled is None:
                baseline = observer.saved
            else:
                self.assertTrue((root / "native_post_adam.json").is_file())
            self.assertEqual(trainer.global_step, 1)
            self.assertIsNone(getattr(trainer.bundle.model, "prefix_guard", None))
        self.assertEqual(log.counts["temporary_optimizer_step_completed"], 3)
        self.assertEqual(log.counts["qualification_k_backward_completed"], 6)
        self.assertEqual(log.counts["response_presentations"], 6)


if __name__ == "__main__":
    unittest.main()
