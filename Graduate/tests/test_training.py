"""CPU optimizer/control-flow checks, not native model or GPU qualification."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from graduate.training import seed_training, train_event


class AnalyticModel(nn.Module):
    """Expose two BF16 parameters and observable fresh draws to the real loop."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([0.001, -0.002], dtype=torch.bfloat16))
        self.lm_head = nn.Identity()
        self.observations = []

    def forward(self, response):
        # This draw checks loop RNG continuity only, not the native anchor rule.
        anchors = torch.rand(5).argsort()[:2].sort().values.unsqueeze(0)
        self.observations.append((response.record_id, anchors.tolist()[0]))
        return SimpleNamespace(
            hidden=self.weight, teacher_hidden=None, labels=response.coefficients,
            weights=response.denominator, anchors=anchors,
        )


def analytic_response_loss(hidden, teacher_hidden, labels, weights, *unused):
    """Unequal response denominators distinguish response and pooled means."""
    numerator = (hidden.float() * labels).sum()
    return numerator / weights, {
        "numerator": numerator.detach(), "denominator": weights.detach(),
    }


def responses():
    return [
        SimpleNamespace(record_id="a", stop_token_ids=(2,),
                        coefficients=torch.tensor([2.0, 0.0]), denominator=torch.tensor(2.0)),
        SimpleNamespace(record_id="b", stop_token_ids=(2,),
                        coefficients=torch.tensor([0.0, 12.0]), denominator=torch.tensor(3.0)),
    ]


def configuration(passes=2):
    return SimpleNamespace(
        training=SimpleNamespace(passes_per_event=passes, accumulation_steps=2,
                                 max_grad_norm=1.0, projection_chunk_blocks=8),
        objective=object(),
    )


def optimizer(parameter):
    return torch.optim.AdamW([parameter], lr=3e-4, weight_decay=0.01)


class SeedTrainingTests(unittest.TestCase):
    def test_initializes_all_generators_once_without_consuming_draws(self):
        self.addCleanup(random.setstate, random.getstate())
        self.addCleanup(np.random.set_state, np.random.get_state())
        self.enterContext(torch.random.fork_rng(devices=[]))
        seed = 17
        # Exercise torch.manual_seed's real delegation without initializing CUDA.
        with patch("torch.cuda.manual_seed_all") as seed_cuda:
            seed_training(seed)
        seed_cuda.assert_called_once_with(seed)
        self.assertEqual(random.getstate(), random.Random(seed).getstate())
        np.testing.assert_equal(np.random.get_state(), np.random.RandomState(seed).get_state())
        expected_torch = torch.Generator().manual_seed(seed).get_state()
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_torch))


class TrainingLoopTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="graduate-training-test-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.enterContext(torch.random.fork_rng(devices=[]))

    def run_event(self, model, adam, feedback, passes=2, name="presentations.jsonl"):
        path = self.directory / name
        with patch("graduate.training.response_loss", side_effect=analytic_response_loss):
            with redirect_stdout(io.StringIO()):
                result = train_event(model, adam, feedback, configuration(passes), path)
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        return result, rows

    def assert_matches_analytic_steps(self, model, adam, steps):
        reference = nn.Parameter(torch.tensor([0.001, -0.002], dtype=torch.bfloat16))
        reference_adam = optimizer(reference)
        # d[(2*x/2 + 12*y/3)/2] = [0.5, 2]. Pooling denominators
        # would instead give [0.4, 2.4]; clipping each response also differs.
        for _ in range(steps):
            reference.grad = torch.tensor([0.5, 2.0], dtype=torch.bfloat16)
            expected_norm = torch.nn.utils.clip_grad_norm_([reference], 1.0)
            reference_adam.step()
            reference_adam.zero_grad(set_to_none=True)
        self.assertTrue(torch.equal(model.weight, reference))
        actual_state, expected_state = adam.state[model.weight], reference_adam.state[reference]
        for name in ("step", "exp_avg", "exp_avg_sq"):
            with self.subTest(optimizer_state=name):
                self.assertTrue(torch.equal(actual_state[name], expected_state[name]))
        self.assertEqual(actual_state["exp_avg"].dtype, torch.bfloat16)
        self.assertEqual(actual_state["exp_avg_sq"].dtype, torch.bfloat16)
        self.assertIsNone(model.weight.grad)
        return expected_norm.item()

    def test_separate_response_means_accumulate_then_clip_once_before_bf16_adam(self):
        model = AnalyticModel().eval()
        adam = optimizer(model.weight)
        result, rows = self.run_event(model, adam, responses(), passes=1)
        self.assertTrue(model.training)
        expected_norm = self.assert_matches_analytic_steps(model, adam, steps=1)
        self.assertEqual(result["presentations"], 2)
        self.assertEqual(result["optimizer_steps"], 1)
        self.assertNotIn("optimizer_step", rows[0])
        self.assertEqual(rows[1]["optimizer_step"], 1)
        self.assertEqual(rows[1]["gradient_norm"], expected_norm)

    def test_two_shuffled_passes_preserve_interleaved_model_draws(self):
        seed = 17
        generator = torch.Generator().manual_seed(seed)
        expected = []
        feedback = responses()
        for _ in range(2):
            order = torch.randperm(len(feedback), generator=generator).tolist()
            for index in order:
                anchors = torch.rand(5, generator=generator).argsort()[:2].sort().values.tolist()
                expected.append((feedback[index].record_id, anchors))

        model = AnalyticModel()
        adam = optimizer(model.weight)
        torch.manual_seed(seed)
        result, rows = self.run_event(model, adam, feedback)
        self.assertEqual(model.observations, expected)
        self.assertTrue(torch.equal(torch.get_rng_state(), generator.get_state()))
        self.assertEqual([row["pass"] for row in rows], [1, 1, 2, 2])
        self.assertEqual([row["presentation"] for row in rows], [1, 2, 3, 4])
        self.assertEqual([row.get("optimizer_step") for row in rows], [None, 1, None, 2])
        self.assertEqual(result["responses"], 2)
        self.assertEqual(result["presentations"], 4)
        self.assertEqual(result["optimizer_steps"], 2)
        self.assert_matches_analytic_steps(model, adam, steps=2)

    def test_successive_events_keep_the_same_adam_moments_and_step_count(self):
        model = AnalyticModel()
        adam = optimizer(model.weight)
        self.run_event(model, adam, responses(), passes=1, name="first.jsonl")
        second, _ = self.run_event(model, adam, responses(), passes=1, name="second.jsonl")
        self.assertEqual(second["optimizer_steps"], 1)
        self.assertEqual(adam.state[model.weight]["step"].item(), 2)
        self.assert_matches_analytic_steps(model, adam, steps=2)

    def test_incomplete_window_fails_before_forward_or_optimizer_work(self):
        model = AnalyticModel()
        adam = optimizer(model.weight)
        before = model.weight.detach().clone()
        path = self.directory / "incomplete.jsonl"
        with self.assertRaisesRegex(ValueError, "complete accumulation window"):
            train_event(model, adam, responses()[:1], configuration(passes=1), path)
        self.assertEqual(model.observations, [])
        self.assertEqual(len(adam.state), 0)
        self.assertTrue(torch.equal(model.weight, before))
        self.assertFalse(path.exists())

    def test_forward_failure_propagates_without_retry_or_optimizer_step(self):
        model = AnalyticModel()
        adam = optimizer(model.weight)
        before = model.weight.detach().clone()
        calls = 0

        def fail_second_response(*args):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("response forward failed")
            return analytic_response_loss(*args)

        path = self.directory / "failed.jsonl"
        with patch("graduate.training.response_loss", side_effect=fail_second_response):
            with redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "response forward failed"):
                    train_event(model, adam, responses(), configuration(), path)
        self.assertEqual(calls, 2)
        self.assertEqual(len(model.observations), 2)
        self.assertEqual(len(adam.state), 0)
        self.assertTrue(torch.equal(model.weight, before))
        self.assertEqual(len(path.read_text().splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
