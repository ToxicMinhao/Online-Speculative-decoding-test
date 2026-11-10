"""Tiny CPU integration checks; these do not qualify the native GPU runtime."""

from copy import deepcopy
import unittest
from unittest.mock import patch

import torch

from specforge.runtime.contracts import TrainBatch
from specforge.training.controller import TrainerCore
from online_feedback.adversarial_block_runtime import install_adversarial_block_core
from online_feedback.first_error import first_error_mask
from online_feedback.kplus import install_kplus
from online_feedback.objective import OSDTrainStrategy, TargetObjectiveModel
from online_feedback.optimizer import DirectBF16AdamW
from online_feedback.rng import capture_rng, restore_rng
from online_feedback.training import ReductionAdapter
from test_objective_integration import tiny_author, response_batch


class CPUBackend:
    checks_loss_denominator = False

    def __init__(self, model):
        self.module = model
        self.optimizer = DirectBF16AdamW(model.draft_model)
        self.backward_boundaries = []
        self.steps = 0
        self.scalings = []

    def backward(self, loss, *, is_boundary=True):
        self.backward_boundaries.append(is_boundary)
        loss.backward()

    def scale_gradients(self, factor):
        self.scalings.append(float(factor))
        for parameter in self.module.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(factor)

    def step(self, **kwargs):
        self.steps += 1
        return self.optimizer.step()


def build():
    model = TargetObjectiveModel(tiny_author(chunk_blocks=2), "osd_first_error",
                                 loss_precision="bfloat16")
    model.num_anchors = 5  # Exercises two full chunks and a partial final chunk.
    install_kplus(model)
    model.begin_event()
    raw = response_batch(model)
    raw["target_last_hidden_states"] = torch.randn(
        1, 12, 16, dtype=torch.bfloat16, generator=torch.Generator().manual_seed(9))
    batch = TrainBatch(sample_ids=["fixture"], strategy="dflash", tensors=raw)
    author = OSDTrainStrategy(model)
    backend = CPUBackend(model)
    native = TrainerCore(ReductionAdapter(author, "response_mean"), backend, accumulation_steps=2)
    return model, batch, author, backend, native


class AdversarialBlockRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def install(self, model, author, native, arm, **kwargs):
        return install_adversarial_block_core(
            native, model, arm=arm, forward_loss=author.forward_loss,
            response_key=lambda batch, ctx: ("fixture", 5, 0, native._micro), **kwargs)

    def test_neutral_installer_returns_original_core_without_model_hooks(self):
        model, _, author, _, native = build()
        original_noise = model._create_noise_embed
        for arm, rho in (("KPLUS", .01), ("ADV", 0), ("RANDOM", 0)):
            actual = self.install(model, author, native, arm, rho=rho)
            self.assertIs(actual, native)
            self.assertEqual(model._create_noise_embed, original_noise)
            self.assertFalse(hasattr(model, "adversarial_block"))

    def test_optional_observer_keeps_native_loss_gradient_and_projection_work(self):
        model, batch, author, backend, native = build()
        before = capture_rng()
        original = author.forward_loss(batch)
        original.loss.backward()
        gradients = {name: p.grad.clone() for name, p in model.draft_model.named_parameters()}
        backend.optimizer.zero_grad()
        restore_rng(before)
        rows = {}

        def observer(proposals, labels, weights, selection):
            rows.setdefault(labels.storage_offset(), tuple(value.detach().clone()
                for value in (proposals, labels, weights, selection)))

        model.block_observer = observer
        observed = author.forward_loss(batch)
        observed.loss.backward()
        torch.testing.assert_close(observed.loss, original.loss, rtol=0, atol=0)
        for name, parameter in model.draft_model.named_parameters():
            torch.testing.assert_close(parameter.grad, gradients[name], rtol=0, atol=0)
        self.assertEqual(sorted(rows), [0, 8, 16])
        self.assertEqual([rows[key][0].shape[1] for key in sorted(rows)], [2, 2, 1])

    def test_two_responses_four_views_one_native_bf16_adam_boundary(self):
        model, batch, author, backend, native = build()
        observations, work = [], []
        original = deepcopy(model.state_dict())
        core = self.install(model, author, native, "ADV",
            observe_response=lambda batch, out, diag: observations.append((out, diag)),
            work_callback=lambda *values: work.append(values))
        with patch.object(torch.nn.utils, "clip_grad_norm_", wraps=torch.nn.utils.clip_grad_norm_) as clip:
            first = core.train_step(batch)
            self.assertFalse(first.optimizer_stepped)
            self.assertEqual(core._micro, 1)
            self.assertEqual(backend.steps, 0)
            self.assertFalse(backend.optimizer.optimizer.state)
            for name, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, original[name]))
            self.assertTrue(all(p.grad.dtype == torch.bfloat16 for p in model.draft_model.parameters()))
            second = core.train_step(batch)
            self.assertTrue(second.optimizer_stepped)
            self.assertEqual(clip.call_count, 1)
        self.assertEqual(core._micro, 2)
        self.assertEqual(backend.steps, 1)
        self.assertEqual(backend.scalings, [1.])
        self.assertEqual(backend.backward_boundaries, [False, False, False, True])
        self.assertEqual(len(observations), 2)
        self.assertEqual(len(work), 16)
        self.assertEqual(core._ratio_totals, {})
        self.assertEqual(core._sum_totals, {})
        for parameter in model.draft_model.parameters():
            state = backend.optimizer.optimizer.state[parameter]
            self.assertEqual(state["step"].item(), 1)
            self.assertEqual(state["exp_avg"].dtype, torch.bfloat16)
            self.assertEqual(state["exp_avg_sq"].dtype, torch.bfloat16)
        for output, diagnostic in observations:
            self.assertAlmostEqual(float(output.loss),
                .5 * diagnostic["clean_loss"] + .5 * diagnostic["second_loss"], places=7)
            self.assertEqual(output.loss_terms[1].item(), 1)
            self.assertEqual(output.sum_metrics["response_presentations"].item(), 1)
            self.assertEqual(output.sum_metrics["abt_forward_views"].item(), 2)
            snapshots = diagnostic["snapshots"]
            self.assertEqual(snapshots["anchors"].shape, (1, 5))
            self.assertTrue(torch.equal(snapshots["selection"], first_error_mask(
                snapshots["clean_proposals"], snapshots["labels"], snapshots["weights"])))
            self.assertTrue(all(value.device.type == "cpu" for value in snapshots.values()
                                if isinstance(value, torch.Tensor)))
            self.assertEqual(set(snapshots["model_rng"]), {"python", "numpy", "torch", "cuda"})
        self.assertFalse(hasattr(model, "adversarial_block"))
        self.assertTrue(all(p.grad is None for p in model.embed_tokens.parameters()))
        self.assertTrue(all(p.grad is None for p in model.lm_head.parameters()))

    def test_clean2_is_two_real_passes_and_replays_stochastic_realization(self):
        model, batch, author, backend, native = build()
        calls, draws, observations = [], [], []
        original_forward = model.draft_model.forward

        def stochastic_forward(*args, **kwargs):
            draw = torch.rand(())
            draws.append(draw.clone())
            calls.append(1)
            return original_forward(*args, **kwargs) * (.5 + draw)

        model.draft_model.forward = stochastic_forward
        core = self.install(model, author, native, "CLEAN2", rho=0,
            observe_response=lambda batch, out, diag: observations.append(diag))
        before = capture_rng()
        # Original native response establishes the ordinary RNG endpoint.
        native.train_step(batch)
        expected_rng = capture_rng()
        backend.optimizer.zero_grad()
        restore_rng(before)
        draws.clear()
        calls.clear()
        backend.backward_boundaries.clear()
        core.train_step(batch)
        self.assertEqual(len(calls), 2)
        self.assertEqual(backend.backward_boundaries, [False, False])
        self.assertTrue(torch.equal(draws[0], draws[1]))
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng["torch"]))
        self.assertEqual(observations[0]["clean_loss"], observations[0]["second_loss"])
        self.assertTrue(torch.equal(observations[0]["snapshots"]["clean_proposals"],
                                    observations[0]["snapshots"]["second_proposals"]))

    def test_random_and_adv_leave_ordinary_rng_and_anchor_schedule_equal(self):
        snapshots = []
        for arm in ("KPLUS", "RANDOM", "ADV"):
            model, batch, author, backend, native = build()
            sampled, diagnostics = [], []
            original_sampler = model._sample_anchor_positions

            def sampler(*args, **kwargs):
                values = original_sampler(*args, **kwargs)
                sampled.append(values[0].clone())
                return values

            model._sample_anchor_positions = sampler
            core = self.install(model, author, native, arm,
                observe_response=lambda batch, out, diag: diagnostics.append(diag))
            core.train_step(batch)
            self.assertEqual(len(sampled), 1)
            snapshots.append((sampled[0], torch.get_rng_state().clone()))
            if arm != "KPLUS":
                self.assertGreater(int(diagnostics[0]["snapshots"]["changed_slots"].sum()), 0)
        for anchors, rng in snapshots[1:]:
            self.assertTrue(torch.equal(anchors, snapshots[0][0]))
            self.assertTrue(torch.equal(rng, snapshots[0][1]))

    def test_both_graphs_complete_sequentially_and_observation_precedes_step(self):
        model, batch, author, backend, native = build()
        events = []
        raw_forward = author.forward_loss

        def forward(batch, ctx):
            events.append("forward")
            output = raw_forward(batch, ctx)
            output.loss.register_hook(lambda gradient: events.append("loss_backward"))
            return output

        core = install_adversarial_block_core(native, model, arm="CLEAN2", forward_loss=forward,
            response_key=lambda batch, ctx: ("fixture", 5, 0, len(events)),
            observe_response=lambda batch, out, diag: events.append("observation"))
        old_step = backend.step
        backend.step = lambda **kwargs: (events.append("step"), old_step(**kwargs))[1]
        core.train_step(batch)
        core.train_step(batch)
        self.assertEqual(events, ["forward", "loss_backward", "forward", "loss_backward", "observation",
                                  "forward", "loss_backward", "forward", "loss_backward", "observation", "step"])

    def test_clean2_four_view_gradients_and_adam_match_explicit_quarter_weight_reference(self):
        model, batch, author, backend, native = build()
        core = self.install(model, author, native, "CLEAN2")
        before = capture_rng()
        core.train_step(batch)
        core.train_step(batch)
        actual_weights = deepcopy(model.draft_model.state_dict())
        actual_optimizer = deepcopy(backend.optimizer.state_dict())
        expected_model, expected_batch, expected_author, expected_backend, _ = build()
        restore_rng(before)
        for _ in range(2):
            before_response = capture_rng()
            output = expected_author.forward_loss(expected_batch)
            (.25 * output.loss).backward()
            after_clean = capture_rng()
            restore_rng(before_response)
            output = expected_author.forward_loss(expected_batch)
            (.25 * output.loss).backward()
            restore_rng(after_clean)
        expected_backend.optimizer.step()
        for name, value in expected_model.draft_model.state_dict().items():
            torch.testing.assert_close(actual_weights[name], value, rtol=0, atol=0)
        expected_state = expected_backend.optimizer.state_dict()["optimizer_state_dict"]["state"]
        for index, state in actual_optimizer["optimizer_state_dict"]["state"].items():
            for name, value in state.items():
                torch.testing.assert_close(value, expected_state[index][name], rtol=0, atol=0)

    def test_second_view_failure_restores_hooks_rng_and_never_steps(self):
        model, batch, author, backend, native = build()
        calls = []
        raw_forward = author.forward_loss
        original_noise = model._create_noise_embed
        original_sampler = model._sample_anchor_positions
        post_clean = []

        def forward(batch, ctx):
            calls.append(1)
            if len(calls) == 2:
                torch.rand(19)
                raise RuntimeError("second fixture")
            output = raw_forward(batch, ctx)
            post_clean.append(torch.get_rng_state().clone())
            return output

        core = install_adversarial_block_core(native, model, arm="ADV", forward_loss=forward,
                                              response_key=lambda batch, ctx: ("fixture", 5, 0, 0))
        with self.assertRaisesRegex(RuntimeError, "second fixture"):
            core.train_step(batch)
        self.assertEqual(backend.steps, 0)
        self.assertEqual(model._create_noise_embed, original_noise)
        self.assertEqual(model._sample_anchor_positions, original_sampler)
        self.assertFalse(hasattr(model, "adversarial_block"))
        self.assertTrue(torch.equal(torch.get_rng_state(), post_clean[0]))
        self.assertIsNone(core.runtime.delta)


if __name__ == "__main__":
    unittest.main()
