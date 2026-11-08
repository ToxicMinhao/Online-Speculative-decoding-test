"""CPU tests for zero-update head isolation; no native GPU equivalence claim."""
from copy import deepcopy
import json
import unittest

import torch

from online_feedback.fsg_gradient_diagnostic import (
    GRADIENT_TOLERANCES, HeadDiagnosticError, REFERENCE, VARIANTS,
    run_head_diagnostic, tensor_comparison,
)
from online_feedback.functional_geometry_math import geometry_terms


def fixture():
    head = torch.nn.Linear(3, 5, bias=False).bfloat16()
    with torch.no_grad():
        head.weight.copy_(torch.arange(15).reshape(5, 3).bfloat16()/32)
    before = torch.arange(24).reshape(1, 2, 4, 3).bfloat16()/16
    native = before + torch.tensor([.125, -.25, .25]).bfloat16()
    weights = torch.tensor([[[0., .125, .125, .25], [0., .125, .125, .25]]])
    return head, before, native, weights


class TensorComparisonTests(unittest.TestCase):
    def test_fp64_energy_difference_is_not_rounded_away(self):
        report = tensor_comparison(torch.tensor([1.+1e-11], dtype=torch.float64),
                                   torch.ones(1, dtype=torch.float64))
        self.assertFalse(report['exact'])
        self.assertGreater(report['maximum_absolute_error'], 0.)
        self.assertEqual(report['differing_coordinates'], 1)

    def test_threshold_and_global_norm_are_both_required(self):
        reference = torch.ones(4, dtype=torch.bfloat16)
        same = tensor_comparison(reference, reference)
        self.assertTrue(same['exact'])
        self.assertTrue(same['matches_pinned_tolerance'])
        # Coordinate absolute tolerance alone admits this, global norm does not.
        tiny = tensor_comparison(torch.full((4,), 1e-6), torch.zeros(4))
        self.assertEqual(tiny['violations'], 0)
        self.assertFalse(tiny['matches_pinned_tolerance'])
        self.assertEqual(tiny['relative_l2'], 'inf')

    def test_nonfinite_evidence_is_explicit_json_safe_and_bounded(self):
        report = tensor_comparison(torch.tensor([float('nan'), float('inf'), 3.]),
                                   torch.zeros(3), sample_size=2)
        self.assertEqual(report['nonfinite_coordinates'], 2)
        self.assertEqual(report['violations'], 3)
        self.assertEqual(len(report['failing_sample']), 2)
        self.assertFalse(report['matches_pinned_tolerance'])
        json.dumps(report, allow_nan=False)


class HeadDiagnosticTests(unittest.TestCase):
    def test_three_layouts_preserve_model_rng_inputs_and_existing_gradients(self):
        head, before, native, weights = fixture()
        head.train()
        head.weight.grad = torch.ones_like(head.weight)
        existing_gradient = head.weight.grad
        saved = deepcopy(head.state_dict())
        inputs = [x.clone() for x in (before, native, weights)]
        rng = torch.get_rng_state().clone()
        receipt, tensors = run_head_diagnostic(head, before, native, weights,
            return_tensors=True, include_fixed_cotangent=False)
        self.assertEqual(receipt['status'], 'completed')
        self.assertEqual(receipt['work']['head_backward_completed'], 3)
        self.assertFalse(receipt['fixed_cotangent_executed'])
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(torch.equal(head.weight, saved['weight']))
        self.assertIs(head.weight.grad, existing_gradient)
        self.assertTrue(head.training)
        self.assertTrue(receipt['head_parameters_and_gradients_unchanged'])
        for value, old in zip((before, native, weights), inputs):
            self.assertTrue(torch.equal(value, old))
        self.assertEqual(set(tensors['adjoints']), set(VARIANTS))
        for adjoint in tensors['adjoints'].values():
            self.assertEqual(adjoint.dtype, torch.bfloat16)
            self.assertFalse(adjoint.requires_grad)
        json.dumps(receipt, allow_nan=False)

    def test_full_layout_adjoint_matches_direct_detached_proxy(self):
        head, before, native, weights = fixture()
        receipt, tensors = run_head_diagnostic(head, before, native, weights,
            return_tensors=True, include_fixed_cotangent=False)
        live = before.clone().requires_grad_(True)
        logits = head(live)
        cotangent, _ = geometry_terms(logits, head(native), weights)
        expected, = torch.autograd.grad((logits.float()*cotangent).sum(), live)
        self.assertTrue(torch.equal(tensors['adjoints'][REFERENCE], expected))
        self.assertEqual(receipt['gradient_tolerances'], GRADIENT_TOLERANCES)
        self.assertTrue(receipt['checkpoint_effect']['initial']['adjoint']['exact'])
        for comparison in receipt['comparisons']['initial'].values():
            self.assertIn('autograd_entry_cotangent', comparison)

    def test_layout_dependent_forward_triggers_only_three_fixed_cotangent_backwards(self):
        class ShapeSensitiveHead(torch.nn.Linear):
            def forward(self, hidden):
                logits = super().forward(hidden)
                # Deliberate fixture discrepancy; production head is unchanged.
                if hidden.shape[1] == 1:
                    logits = logits + torch.arange(5, device=hidden.device).bfloat16()/4
                return logits
        ordinary, before, native, weights = fixture()
        head = ShapeSensitiveHead(3, 5, bias=False).bfloat16()
        head.load_state_dict(ordinary.state_dict())
        receipt, tensors = run_head_diagnostic(head, before, native, weights, return_tensors=True)
        self.assertTrue(receipt['fixed_cotangent_executed'])
        self.assertEqual(receipt['work']['head_backward_completed'], 6)
        self.assertEqual(set(tensors['fixed_adjoints']), set(VARIANTS))
        for comparison in receipt['comparisons']['fixed_cotangent'].values():
            self.assertTrue(comparison['cotangent']['exact'])
            self.assertTrue(comparison['autograd_entry_cotangent']['exact'])
        self.assertTrue(receipt['head_parameters_and_gradients_unchanged'])

    def test_zero_weights_have_zero_adjoint_without_extra_fixed_trial(self):
        head, before, native, weights = fixture()
        receipt, tensors = run_head_diagnostic(head, before, native, weights*0, return_tensors=True)
        self.assertEqual(receipt['work']['head_backward_completed'], 3)
        self.assertFalse(receipt['fixed_cotangent_executed'])
        self.assertTrue(all(not bool(x.any()) for x in tensors['adjoints'].values()))

    def test_only_local_head_graph_enables_grad_under_scoring_context(self):
        head, before, native, weights = fixture()
        with torch.no_grad():
            receipt = run_head_diagnostic(head, before, native, weights, include_fixed_cotangent=False)
            self.assertFalse(torch.is_grad_enabled())
        self.assertEqual(receipt['work']['head_backward_completed'], 3)
        self.assertTrue(all(parameter.grad is None for parameter in head.parameters()))

    def test_nonfinite_head_stops_with_partial_work_and_preserves_state(self):
        head, before, native, weights = fixture()
        with torch.no_grad():
            head.weight[0, 0] = float('nan')
        with self.assertRaises(HeadDiagnosticError) as caught:
            run_head_diagnostic(head, before, native, weights)
        receipt = caught.exception.receipt
        self.assertEqual(receipt['status'], 'failed')
        self.assertEqual(receipt['work'].get('head_backward_attempted', 0), 0)
        self.assertTrue(receipt['head_parameters_and_gradients_unchanged'])
        self.assertIn('Nonfinite', receipt['error'])
        self.assertGreater(receipt['nonfinite']['summary']['nonfinite_coordinates'], 0)
        json.dumps(receipt, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
