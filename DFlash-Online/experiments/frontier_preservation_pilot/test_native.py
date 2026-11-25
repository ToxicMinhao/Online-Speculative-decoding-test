"""CPU-only derivative composition and durable failure accounting."""

import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from online_feedback.rng import capture_rng
from .native import (
    WorkReceipt,
    first_order_control,
    forward_ad_jvp,
    rng_digest,
    support_attempt,
    trainable_parameters,
    transposed_reverse_jvp,
    unit_direction,
)


class SharedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([[0.3, -0.4], [0.2, 0.5]], dtype=torch.float32))
        self.bias = nn.Parameter(torch.tensor([0.1, -0.2], dtype=torch.float32))

    def forward(self, response):
        x = torch.tensor([[1.0, 2.0], [-0.5, 1.0]], dtype=self.weight.dtype)
        hidden = torch.tanh(x @ self.weight + self.bias)
        return hidden @ self.weight.T


class NativeSupportTests(unittest.TestCase):
    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def test_ordinary_first_order_control_is_separate_and_does_not_step(self):
        model = SharedModel()
        parameters = trainable_parameters(model)
        with tempfile.TemporaryDirectory() as directory:
            work = WorkReceipt(Path(directory) / 'control.json')
            result = first_order_control(model, None, parameters, unit_direction(parameters), work)
        self.assertGreater(result['gradient_l2'], 0)
        self.assertEqual(result['gradient_tensors'], 2)
        self.assertEqual(work.value['attempted'], work.value['completed'])
        self.assertEqual(work.value['native_adam_steps'], 0)
        self.assertFalse(work.value['pending'])
        self.assertTrue(all(parameter.grad is None for parameter in parameters.values()))

    def test_matrix_free_routes_match_shared_parameter_direction(self):
        model = SharedModel()
        parameters = trainable_parameters(model)
        direction = unit_direction(parameters)
        before = {name: value.detach().clone() for name, value in parameters.items()}
        with tempfile.TemporaryDirectory() as directory:
            forward = forward_ad_jvp(
                model, None, parameters, direction, WorkReceipt(Path(directory) / 'forward.json')
            )
            reverse = transposed_reverse_jvp(
                model, None, parameters, direction, WorkReceipt(Path(directory) / 'reverse.json')
            )
        self.assertGreater(forward['tangent_norm'], 0)
        self.assertAlmostEqual(forward['tangent_norm'], reverse['tangent_norm'], places=6)
        for name, parameter in parameters.items():
            self.assertTrue(torch.equal(before[name], parameter))
            self.assertIsNone(parameter.grad)

    def test_failed_primitive_keeps_attempt_and_traceback_without_claiming_completion(self):
        class Response:
            rng_after_sampling = capture_rng()

        def unavailable(module, response, parameters, direction, work):
            work.start('native_grad_output_vjp')
            raise NotImplementedError('fixture missing native grad-output derivative')

        with tempfile.TemporaryDirectory() as directory:
            work = WorkReceipt(Path(directory) / 'work.json')
            result = support_attempt(
                unavailable, None, Response(), {}, {}, work, Path(directory) / 'attempt.json'
            )
            stored = json.loads((Path(directory) / 'attempt.json').read_text())
            self.assertEqual(result['status'], 'unsupported_or_failed')
            self.assertIn('NotImplementedError', stored['traceback'])
            self.assertEqual(work.value['attempted'], {'native_grad_output_vjp': 1})
            self.assertEqual(work.value['completed'], {})
            self.assertEqual(work.value['pending'], ['native_grad_output_vjp'])
            self.assertEqual(work.value['native_adam_steps'], 0)

    def test_rng_receipt_is_stable_without_consuming_draws(self):
        first = capture_rng()
        self.assertEqual(rng_digest(first), rng_digest(capture_rng()))


if __name__ == '__main__':
    unittest.main()
