"""CPU checks of diagnostic direction, immutable window and numerical gates."""

from copy import deepcopy
from contextlib import nullcontext
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from online_feedback.rng import capture_rng
from .native import FrozenResponse, WorkReceipt, rng_digest
from .repair_native import (
    DUALITY_ATOL,
    DUALITY_RTOL,
    aligned_direction,
    duality_receipt,
    fixed_cotangent,
    first_vjp_comparison,
    native_model_vjp,
    qualify_transposed_model,
    require_original_window,
    vector_dot,
)


class RepairNativeTests(unittest.TestCase):
    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def test_first_vjp_mismatch_retains_scalar_parameter_evidence(self):
        reference = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
        actual = [reference[0].clone(), torch.tensor([6.0, 8.0])]
        rows = first_vjp_comparison(('equal_weight', 'changed_weight'), actual, reference)
        self.assertTrue(rows[0]['equal'])
        self.assertEqual(rows[0]['l2_difference'], 0.0)
        self.assertFalse(rows[1]['equal'])
        self.assertEqual(rows[1]['max_absolute_difference'], 4.0)
        self.assertEqual(rows[1]['l2_difference'], 5.0)

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

    def test_transposed_probe_preserves_first_vjp_and_fixed_reference(self):
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([0.2, 0.7], dtype=torch.float64))

            def forward(self, response):
                value = self.weight
                for _ in range(5):
                    value = torch.nn.functional.silu(value)
                return value + self.weight.square()

        model = TinyModel()
        response = SimpleNamespace(rng_after_sampling=capture_rng())
        parameters = dict(model.named_parameters())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            work = WorkReceipt(root / 'WORK.json')
            baseline, cotangent, gradients = native_model_vjp(
                model, response, parameters, work, 'ordinary_native'
            )
            reference = vector_dot(gradients, aligned_direction(gradients))
            old = root / 'old'
            old.mkdir()
            (old / 'MODEL.json').write_text(
                json.dumps(
                    dict(
                        native_primal_bit_equal=True,
                        duality=dict(passed=False, native_backward_dot=reference),
                    )
                )
            )
            with mock.patch(
                'experiments.frontier_preservation_pilot.attention_reverse_jvp.native_attention_reverse_jvp',
                side_effect=lambda: nullcontext({}),
            ):
                receipt = qualify_transposed_model(
                    model,
                    response,
                    parameters,
                    baseline,
                    cotangent,
                    gradients,
                    work,
                    root / 'MODEL.json',
                    old,
                )
            self.assertTrue(receipt['native_primal_bit_equal'])
            self.assertTrue(receipt['ordinary_backward_bit_equal'])
            self.assertTrue(receipt['duality']['passed'])
            self.assertEqual(receipt['duality']['native_backward_dot'], reference)
            self.assertFalse(receipt['parameter_hessian_requested'])
            self.assertEqual(receipt['silu_scope_accounting']['native_silu_forwards'], 5)
            self.assertEqual(receipt['silu_scope_accounting']['native_silu_first_vjps'], 5)
            self.assertEqual(receipt['silu_scope_accounting']['silu_cotangent_transposes'], 5)
            self.assertEqual(sum(work.value['completed'].values()), 5)
            self.assertEqual(work.value['pending'], [])
            self.assertEqual(work.value['native_adam_steps'], 0)
            (old / 'MODEL.json').write_text(
                json.dumps(
                    dict(
                        native_primal_bit_equal=True,
                        duality=dict(passed=False, native_backward_dot=reference + 0.1),
                    )
                )
            )
            with self.assertRaisesRegex(ValueError, 'reference direction'):
                qualify_transposed_model(
                    model,
                    response,
                    parameters,
                    baseline,
                    cotangent,
                    gradients,
                    work,
                    root / 'MISMATCH.json',
                    old,
                )
            self.assertEqual(sum(work.value['completed'].values()), 5)
        self.assertIsNone(model.weight.grad)


if __name__ == '__main__':
    unittest.main()
