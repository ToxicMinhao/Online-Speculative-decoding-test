"""CPU checks of diagnostic direction, immutable window and numerical gates."""

from copy import deepcopy
import unittest

import torch

from online_feedback.rng import capture_rng
from .native import FrozenResponse, rng_digest
from .repair_native import (
    DUALITY_ATOL,
    DUALITY_RTOL,
    aligned_direction,
    duality_receipt,
    fixed_cotangent,
    require_original_window,
)


class RepairNativeTests(unittest.TestCase):
    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def test_gradient_aligned_direction_matches_shared_parameter_jvp(self):
        parameter = torch.tensor([0.3, -0.7, 1.1], dtype=torch.float64, requires_grad=True)
        matrix = torch.tensor([[1.0, 2.0, -1.0], [-0.2, 0.8, 1.5]], dtype=torch.float64)

        def function(value):
            return (matrix @ value).sin()

        output = function(parameter)
        cotangent = fixed_cotangent(output)
        gradients = torch.autograd.grad(output, parameter, grad_outputs=cotangent)
        direction = aligned_direction(gradients)
        _, tangent = torch.func.jvp(function, (parameter.detach(),), tuple(direction))
        receipt = duality_receipt(tangent, cotangent, gradients, direction)
        self.assertTrue(receipt['passed'])
        self.assertGreater(receipt['native_backward_dot'], 0)
        self.assertLess(receipt['absolute_error'], 1e-12)
        self.assertIsNone(parameter.grad)

    def test_predeclared_tolerance_rejects_inaccurate_tangent(self):
        gradient = torch.tensor([1.0], dtype=torch.float64)
        cotangent = torch.tensor([1.0], dtype=torch.float64)
        for scale, expected in ((1.015, True), (1.02, False), (float('nan'), False)):
            receipt = duality_receipt(gradient * scale, cotangent, [gradient], [gradient])
            self.assertEqual(receipt['passed'], expected)
            self.assertEqual(receipt['atol'], DUALITY_ATOL)
            self.assertEqual(receipt['rtol'], DUALITY_RTOL)
        with self.assertRaisesRegex(ValueError, 'nonzero'):
            aligned_direction([torch.zeros(2)])
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            aligned_direction([torch.tensor([float('nan')])])

    def test_private_cotangent_generator_preserves_global_rng(self):
        before = rng_digest(capture_rng())
        first = fixed_cotangent(torch.empty(3, 4, dtype=torch.bfloat16))
        second = fixed_cotangent(torch.empty(3, 4, dtype=torch.bfloat16))
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(rng_digest(capture_rng()), before)
        self.assertTrue(bool((first != 0).all()))

    def test_original_two_response_window_must_match_exactly(self):
        state = capture_rng()
        response = FrozenResponse(
            'training-only',
            {
                'input_ids': torch.ones(1, 3, dtype=torch.long),
                'hidden_states': torch.ones(1, 3, 2, dtype=torch.bfloat16),
            },
            torch.tensor([[1]]),
            torch.tensor([[True]]),
            state,
            state,
        )
        window = dict(
            responses=[response.receipt(), response.receipt()],
            accumulation_responses=2,
            native_factor=0.5,
            next_rng_sha256=rng_digest(state),
        )
        self.assertEqual(require_original_window([response, response], window), window)
        changed = deepcopy(window)
        changed['responses'][1]['anchors'] = [[2]]
        with self.assertRaisesRegex(ValueError, 'original selected'):
            require_original_window([response, response], changed)


if __name__ == '__main__':
    unittest.main()
