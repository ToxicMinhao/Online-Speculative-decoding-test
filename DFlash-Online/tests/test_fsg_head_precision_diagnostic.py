"""CPU contracts for the retained-case head-precision diagnostic."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from experiments.functional_step_geometry import diagnose_head_precision as precision
from experiments.functional_step_geometry.diagnose_gradients import DiagnosticWork
from experiments.ok_continuation.resume import differences
from online_feedback.rng import capture_rng, preserve_rng


def fixture():
    head = torch.nn.Linear(3, 5, bias=False).bfloat16().requires_grad_(False)
    with torch.no_grad():
        head.weight.copy_(torch.tensor([
            [.5, -.75, .125], [.25, .5, -.5], [-.25, .75, 1.],
            [.125, -.25, .5], [.75, .25, -.125]], dtype=torch.bfloat16))
    hidden = torch.linspace(-.8, 1.2, 96).reshape(1, 2, 16, 3).bfloat16()
    native = hidden + torch.tensor([.125, -.25, .0625], dtype=torch.bfloat16)
    return head, hidden, native


def reference_hook(head, *, vary_weights=False):
    """Record every branch's reference scan; no borrowed branch state."""
    calls = []

    def scan(hidden, response, chunk_blocks):
        calls.append((chunk_blocks, hidden.detach().clone(), response))
        weights = torch.zeros(hidden.shape[:-1], dtype=torch.float32)
        first_weight = .75 if vary_weights and chunk_blocks == 1 else .25
        weights[:, 0, 1] = first_weight
        weights[:, 1, 1] = 1 - first_weight
        with torch.no_grad():
            proposals = head(hidden).argmax(-1)
        mask = weights > 0
        margins = weights.double() * .1
        return weights, proposals, mask, margins

    hook = SimpleNamespace(model=SimpleNamespace(lm_head=head), _scan_reference=scan)
    return hook, calls


class OperatorControlTests(unittest.TestCase):
    def test_seven_products_use_original_operands_and_leave_inputs_unchanged(self):
        with preserve_rng(), tempfile.TemporaryDirectory() as directory:
            head, hidden, _ = fixture()
            incoming = torch.linspace(-.7, .9, 160).reshape(1, 2, 16, 5).bfloat16()
            saved_incoming = incoming.clone()
            saved_weight = head.weight.clone()
            before_rng = capture_rng()
            before_settings = precision.precision_settings()
            work = DiagnosticWork(directory, precision.BUDGET)
            costs, products = {}, []
            original_matmul = torch.matmul

            def capture_product(left, right):
                products.append((left.detach().clone(), right.detach().clone()))
                return original_matmul(left, right)

            with mock.patch.object(torch, 'matmul', side_effect=capture_product):
                summary, adjoints = precision.operator_controls(incoming, head.weight, work, costs)
            self.assertEqual(len(products), 7)
            self.assertEqual([x.dtype for x, _ in products],
                [torch.bfloat16] * 3 + [torch.float32] * 3 + [torch.float64])
            self.assertEqual([x.shape[1] for x, _ in products], [1, 1, 2, 1, 1, 2, 2])
            for index, (left, right) in enumerate(products):
                start = 1 if index in (1, 4) else 0
                expected = incoming[:, start:start + left.shape[1]].to(left.dtype)
                self.assertTrue(torch.equal(left, expected))
                self.assertTrue(torch.equal(right, saved_weight.to(right.dtype)))
            expected_reference = original_matmul(incoming.double(), saved_weight.double()).bfloat16()
            self.assertTrue(torch.equal(adjoints['reference'], expected_reference))
            self.assertTrue(torch.equal(incoming, saved_incoming))
            self.assertTrue(torch.equal(head.weight, saved_weight))
            self.assertIsNone(head.weight.grad)
            self.assertFalse(differences(before_rng, capture_rng()))
            self.assertEqual(precision.precision_settings(), before_settings)
            self.assertEqual(work.counts, {'operator_gemm_attempted': 7,
                                           'operator_gemm_completed': 7})
            self.assertEqual(set(costs), {'native_bf16/chunked', 'native_bf16/full',
                'ieee_fp32/chunked', 'ieee_fp32/full', 'fp64_reference'})
            self.assertTrue(all(value >= 0 for value in costs.values()))
            self.assertEqual(summary['transient_fp32_weight_bytes'], head.weight.numel() * 4)
            self.assertEqual(summary['transient_fp64_weight_bytes'], head.weight.numel() * 8)

    def test_fp64_reference_does_not_reuse_a_biased_fp32_product(self):
        with tempfile.TemporaryDirectory() as directory:
            head, _, _ = fixture()
            incoming = torch.ones(1, 2, 16, 5, dtype=torch.bfloat16)
            expected = torch.matmul(incoming.double(), head.weight.double()).bfloat16()

            def biased_fp32(cotangent, weight, *, return_precast):
                precast = torch.matmul(cotangent.float(), weight.float()) + 1
                return precast.bfloat16(), precast

            work = DiagnosticWork(directory, precision.BUDGET)
            with mock.patch.object(precision, 'fp32_head_vjp', side_effect=biased_fp32):
                summary, adjoints = precision.operator_controls(incoming, head.weight, work, {})
            self.assertTrue(torch.equal(adjoints['reference'], expected))
            self.assertFalse(torch.equal(adjoints['full'], expected))
            self.assertFalse(summary['accuracy_check_passed'])

    def test_operator_failure_is_durable_and_stops_remaining_products(self):
        with tempfile.TemporaryDirectory() as directory:
            head, _, _ = fixture()
            work = DiagnosticWork(directory, precision.BUDGET)
            costs = {}
            with mock.patch.object(torch, 'matmul', side_effect=RuntimeError('operator failed')):
                with self.assertRaisesRegex(RuntimeError, 'operator failed'):
                    precision.operator_controls(torch.ones(1, 2, 16, 5, dtype=torch.bfloat16),
                                                head.weight, work, costs)
            self.assertEqual(json.loads(work.path.read_text())['counts'],
                {'operator_gemm_attempted': 1, 'operator_gemm_failed': 1})
            self.assertEqual(set(costs), {'native_bf16/chunked'})


class BranchAndBudgetTests(unittest.TestCase):
    def test_complete_small_case_uses_five_parameter_vjps_without_training(self):
        with preserve_rng(), tempfile.TemporaryDirectory() as directory:
            head, source_hidden, native = fixture()
            parameter = torch.nn.Parameter(source_hidden.clone())
            parameter.grad = torch.full_like(parameter, .5)
            ordinary_gradient = parameter.grad
            saved_gradient = ordinary_gradient.clone()
            saved_parameter = parameter.detach().clone()
            hook, scans = reference_hook(head)
            work = DiagnosticWork(directory, precision.BUDGET)
            case = SimpleNamespace(hook=hook, root=Path(directory), work=work,
                hidden=parameter * 1, native_hidden=native, response={'id': 'retained'},
                parameters=(parameter,), names=['fixture_parameter'])
            before_rng = capture_rng()
            before_settings = precision.precision_settings()
            result = precision.precision_case(case)
            self.assertEqual(work.counts, {
                'head_backward_attempted': 2, 'head_backward_completed': 2,
                'draft_backward_attempted': 5, 'draft_backward_completed': 5,
                'operator_gemm_attempted': 7, 'operator_gemm_completed': 7})
            self.assertEqual([row[0] for row in scans], [1, 0])
            self.assertTrue(torch.equal(parameter, saved_parameter))
            self.assertIs(parameter.grad, ordinary_gradient)
            self.assertTrue(torch.equal(parameter.grad, saved_gradient))
            self.assertFalse(differences(before_rng, capture_rng()))
            self.assertEqual(precision.precision_settings(), before_settings)
            self.assertFalse(result['original_gate_overridden'])
            self.assertFalse(result['full_qualification_completed'])
            self.assertTrue(result['fixed_control_does_not_substitute_for_complete_gradient'])
            self.assertTrue((case.root / 'END_TO_END.json').exists())
            self.assertTrue((case.root / 'FIXED_INPUT_PARAMETERS.json').exists())

    def test_each_branch_recomputes_weights_and_sensitivity_and_preserves_bf16_entry(self):
        with preserve_rng(), tempfile.TemporaryDirectory() as directory:
            head, hidden, native = fixture()
            hook, scans = reference_hook(head, vary_weights=True)
            response = {'identity': 'same-retained-response'}
            work = DiagnosticWork(directory, precision.BUDGET)
            before = capture_rng()
            chunked = precision.branch(hook, response, hidden, native, work, chunk_blocks=1)
            full = precision.branch(hook, response, hidden, native, work, chunk_blocks=0)
            self.assertEqual([call[0] for call in scans], [1, 0])
            self.assertTrue(all(call[2] is response for call in scans))
            self.assertFalse(torch.equal(chunked['weights'], full['weights']))
            self.assertFalse(torch.equal(chunked['sensitivity'], full['sensitivity']))
            for result in (chunked, full):
                self.assertEqual(result['incoming'].dtype, torch.bfloat16)
                self.assertTrue(torch.equal(result['incoming'], result['sensitivity'].bfloat16()))
                self.assertEqual(result['adjoint'].dtype, torch.bfloat16)
                self.assertEqual(float(result['weights'].sum()), 1.)
                self.assertTrue(torch.equal(result['pre_logits'], head(hidden)))
            self.assertFalse(differences(before, capture_rng()))
            self.assertIsNone(head.weight.grad)
            self.assertEqual(work.counts, {'head_backward_attempted': 2,
                                           'head_backward_completed': 2})

    def test_work_limits_are_exact_and_zero_update_categories_stay_closed(self):
        self.assertEqual(precision.BUDGET, dict(response_forward=2, head_backward=2,
            draft_backward=5, operator_gemm=7, optimizer_step=0, feedback_generation=0,
            teacher_preparation=0, serving_generation=0))
        with tempfile.TemporaryDirectory() as directory:
            work = DiagnosticWork(directory, precision.BUDGET)
            for kind, limit in precision.BUDGET.items():
                with self.subTest(kind=kind):
                    for _ in range(limit):
                        with work.operation(kind):
                            pass
                    previous = dict(work.counts)
                    with self.assertRaisesRegex(ValueError, 'work limit exceeded'):
                        with work.operation(kind):
                            self.fail('Exceeded the sealed work cap')
                    self.assertEqual(work.counts, previous)


if __name__ == '__main__':
    unittest.main()
