"""CPU-only tests for the prototype's restricted cotangent transpose."""

from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from .attention_reverse_jvp import (
    attention_reverse_jvp_wrapper,
    native_attention_reverse_jvp,
    native_silu_reverse_jvp,
)
from .test_attention_jvp import dense_attention


class RejectDoubleBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value.clone()

    @staticmethod
    def backward(ctx, gradient):
        raise RuntimeError('fixture native double backward is unsupported')


class OpaqueAttention(torch.autograd.Function):
    """A first-order native fixture with the same missing higher derivative."""

    @staticmethod
    def forward(ctx, query, key, value):
        ctx.save_for_backward(query, key, value)
        return dense_attention(query, key, value)

    @staticmethod
    def backward(ctx, incoming):
        higher_order = torch.is_grad_enabled()
        with torch.enable_grad():
            leaves = tuple(value.detach().requires_grad_() for value in ctx.saved_tensors)
            output = dense_attention(*leaves)
            gradients = torch.autograd.grad(output, leaves, incoming, create_graph=higher_order)
        return tuple(RejectDoubleBackward.apply(gradient) for gradient in gradients)


def opaque_attention(query, key, value, *, return_aux=None, **kwargs):
    output = OpaqueAttention.apply(query, key, value)
    if return_aux is None:
        return output
    _, auxiliary = dense_attention(query, key, value, return_aux=True)
    return output, auxiliary


class OpaqueSquare(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        ctx.save_for_backward(value)
        return value.square()

    @staticmethod
    def backward(ctx, incoming):
        (value,) = ctx.saved_tensors
        return RejectDoubleBackward.apply(2 * value * incoming)


class SingleUseAttention(OpaqueAttention):
    """Model the compiled donated-buffer policy at the native VJP boundary."""

    @staticmethod
    def backward(ctx, incoming):
        if torch.is_grad_enabled() or torch._C._autograd._get_current_graph_task_keep_graph():
            raise RuntimeError('fixture native buffers require a single-use first backward')
        return OpaqueAttention.backward(ctx, incoming)


class CotangentTransposeTests(unittest.TestCase):
    def setUp(self):
        self.generator = torch.Generator().manual_seed(3501)

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized(), 'CPU prototype initialized CUDA')

    def inputs(self, dtype=torch.float64):
        shapes = ((2, 4, 5, 4), (2, 2, 7, 4), (2, 2, 7, 3))
        values = [
            torch.randn(shape, generator=self.generator, dtype=dtype).requires_grad_()
            for shape in shapes
        ]
        directions = [
            torch.randn(shape, generator=self.generator, dtype=dtype) for shape in shapes
        ]
        return values, directions

    def transpose(self, output, parameters, directions):
        dummy = torch.zeros_like(output, requires_grad=True)
        gradients = torch.autograd.grad(
            output, parameters, dummy, create_graph=True, retain_graph=True
        )
        contraction = sum(
            (gradient * direction).sum() for gradient, direction in zip(gradients, directions)
        )
        return torch.autograd.grad(contraction, dummy)[0]

    def test_exact_primal_and_first_vjp_then_continuous_transpose(self):
        for dtype, tolerance in ((torch.float64, 1e-10), (torch.float32, 2e-5)):
            with self.subTest(dtype=dtype):
                values, directions = self.inputs(dtype)
                reference = dense_attention(*values)
                accounting = {}
                wrapped = attention_reverse_jvp_wrapper(dense_attention, accounting)
                output = wrapped(*values, enable_gqa=True)
                torch.testing.assert_close(output, reference, rtol=0, atol=0)
                cotangent = torch.randn(output.shape, generator=self.generator, dtype=dtype)
                expected = torch.autograd.grad(reference, values, cotangent, retain_graph=True)
                actual = torch.autograd.grad(output, values, cotangent, retain_graph=True)
                for left, right in zip(actual, expected):
                    torch.testing.assert_close(left, right, rtol=0, atol=0)
                # First-VJP identity and transpose are independent checks; each
                # owns a fresh native forward whose inner graph is used once.
                transpose_output = wrapped(*values, enable_gqa=True)
                tangent = self.transpose(transpose_output, values, directions)
                _, expected_tangent = torch.autograd.functional.jvp(
                    dense_attention, tuple(values), tuple(directions)
                )
                torch.testing.assert_close(
                    tangent, expected_tangent, rtol=tolerance, atol=tolerance
                )
                self.assertEqual(accounting['native_primal_forwards'], 2)
                self.assertEqual(accounting['native_first_vjps'], 2)
                self.assertEqual(accounting['cotangent_transposes'], 1)
                self.assertEqual(accounting['derivative_fp32_precision'], 'ieee')
                self.assertTrue(accounting['precision_restored'])

    def test_transpose_consumes_native_inner_graph_once_without_retention(self):
        values, directions = self.inputs()
        saved_values = []

        def native(query, key, value, *, return_aux=None, **kwargs):
            output = SingleUseAttention.apply(query, key, value)
            _, auxiliary = dense_attention(query, key, value, return_aux=True)
            for tensor in (query, key, value, output, auxiliary.lse):
                saved_values.append((tensor, tensor.detach().clone()))
            return (output, auxiliary) if return_aux else output

        accounting = {}
        wrapper = attention_reverse_jvp_wrapper(native, accounting)
        actual = self.transpose(wrapper(*values, enable_gqa=True), values, directions)
        _, expected = torch.autograd.functional.jvp(
            dense_attention, tuple(values), tuple(directions)
        )
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
        for tensor, before in saved_values:
            torch.testing.assert_close(tensor, before, rtol=0, atol=0)
        self.assertEqual(accounting['native_primal_forwards'], 1)
        self.assertEqual(accounting['native_first_vjps'], 1)
        self.assertEqual(accounting['cotangent_transposes'], 1)

    def test_repeated_native_vjp_requires_a_fresh_outer_forward(self):
        values, _ = self.inputs()
        accounting = {}
        output = attention_reverse_jvp_wrapper(dense_attention, accounting)(
            *values, enable_gqa=True
        )
        incoming = torch.ones_like(output)
        torch.autograd.grad(output, values, incoming, retain_graph=True)
        with self.assertRaisesRegex(RuntimeError, 'already consumed; run a fresh forward'):
            torch.autograd.grad(output, values, incoming, retain_graph=True)
        self.assertEqual(accounting['native_first_vjps'], 1)

    def test_native_bf16_primal_and_first_vjp_are_exact(self):
        values, _ = self.inputs(torch.bfloat16)
        reference = dense_attention(*values)
        output = attention_reverse_jvp_wrapper(dense_attention, {})(*values, enable_gqa=True)
        torch.testing.assert_close(output, reference, rtol=0, atol=0)
        cotangent = torch.randn(output.shape, generator=self.generator, dtype=torch.bfloat16)
        expected = torch.autograd.grad(reference, values, cotangent)
        actual = torch.autograd.grad(output, values, cotangent)
        for left, right in zip(actual, expected):
            torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_opaque_native_double_backward_is_bypassed_only_at_attention(self):
        values, directions = self.inputs()
        with self.assertRaisesRegex(RuntimeError, 'native double backward'):
            self.transpose(opaque_attention(*values), values, directions)
        accounting = {}
        wrapper = attention_reverse_jvp_wrapper(opaque_attention, accounting)
        output = wrapper(*values, enable_gqa=True)
        actual = self.transpose(output, values, directions)
        _, expected = torch.autograd.functional.jvp(
            dense_attention, tuple(values), tuple(directions)
        )
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
        self.assertEqual(accounting['native_primal_forwards'], 1)
        self.assertEqual(accounting['native_first_vjps'], 1)
        self.assertEqual(accounting['cotangent_transposes'], 1)

    def test_shared_parameter_composition_matches_direct_jvp_and_preserves_state(self):
        weight = torch.randn(
            4, 4, generator=self.generator, dtype=torch.float64
        ).requires_grad_()
        direction = torch.randn(4, 4, generator=self.generator, dtype=torch.float64)
        query_input = torch.randn(2, 5, 4, generator=self.generator, dtype=torch.float64)
        context = torch.randn(2, 7, 4, generator=self.generator, dtype=torch.float64)
        weight.grad = torch.ones_like(weight)
        before_weight, before_grad = weight.detach().clone(), weight.grad.clone()
        before_rng = torch.get_rng_state().clone()
        accounting = {}
        wrapped = attention_reverse_jvp_wrapper(opaque_attention, accounting)

        def model(parameter, attention):
            hidden = torch.tanh(query_input @ parameter)
            keys = torch.sigmoid(context @ parameter)
            values = context @ parameter
            attended = attention(hidden.unsqueeze(1), keys.unsqueeze(1), values.unsqueeze(1))
            residual = attended.squeeze(1) + hidden
            normalized = residual * torch.rsqrt(residual.square().mean(-1, keepdim=True) + 1e-6)
            return torch.nn.functional.silu(normalized) @ parameter

        output = model(weight, wrapped)
        actual = self.transpose(output, (weight,), (direction,))
        _, expected = torch.autograd.functional.jvp(
            lambda value: model(value, dense_attention), weight, direction
        )
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(weight, before_weight, rtol=0, atol=0)
        torch.testing.assert_close(weight.grad, before_grad, rtol=0, atol=0)
        self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))
        self.assertEqual(accounting['cotangent_transposes'], 1)

    def test_unrelated_unsupported_double_backward_remains_visible(self):
        values, directions = self.inputs()
        wrapped = attention_reverse_jvp_wrapper(dense_attention, {})
        output = OpaqueSquare.apply(wrapped(*values, enable_gqa=True))
        with self.assertRaisesRegex(RuntimeError, 'native double backward'):
            self.transpose(output, values, directions)

    def test_scope_restores_alias_and_no_grad_calls_delegate(self):
        values, directions = self.inputs()
        native = mock.Mock(side_effect=dense_attention)
        module = SimpleNamespace(compile_friendly_flex_attention=native)
        with mock.patch(
            'experiments.frontier_preservation_pilot.attention_reverse_jvp.importlib.import_module',
            return_value=module,
        ):
            with self.assertRaisesRegex(RuntimeError, 'fixture exit'):
                with native_attention_reverse_jvp() as accounting:
                    with torch.no_grad():
                        output = module.compile_friendly_flex_attention(*values)
                    torch.testing.assert_close(output, dense_attention(*values), rtol=0, atol=0)
                    self.assertEqual(accounting['passthrough_calls'], 1)
                    self.assertEqual(accounting['native_primal_forwards'], 0)
                    raise RuntimeError('fixture exit')
            self.assertIs(module.compile_friendly_flex_attention, native)
        native.assert_called_once_with(*values)
        wrapper = attention_reverse_jvp_wrapper(dense_attention, {})
        with torch.autograd.forward_ad.dual_level():
            duals = [
                torch.autograd.forward_ad.make_dual(value, direction)
                for value, direction in zip(values, directions)
            ]
            with self.assertRaisesRegex(ValueError, 'combined with forward AD'):
                wrapper(*duals)


class SiLUCotangentTests(unittest.TestCase):
    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized(), 'CPU SiLU qualification initialized CUDA')

    def test_bf16_before_discrepancy_and_exact_native_first_vjp_after_fix(self):
        generator = torch.Generator().manual_seed(20261125)
        value = torch.randn(8192, generator=generator).bfloat16().requires_grad_()
        incoming = torch.randn(8192, generator=generator).bfloat16().requires_grad_()
        direction = torch.randn(8192, generator=generator).bfloat16()
        original = torch.nn.functional.silu
        native_output = original(value)
        ordinary = torch.autograd.grad(native_output, value, incoming)[0]
        higher_order = torch.autograd.grad(original(value), value, incoming, create_graph=True)[
            0
        ]
        mismatch_count = int(torch.count_nonzero(ordinary != higher_order))
        self.assertGreater(mismatch_count, 0)
        native_kernel = torch.ops.aten.silu_backward.default(incoming.detach(), value.detach())
        torch.testing.assert_close(native_kernel, ordinary, rtol=0, atol=0)
        with native_silu_reverse_jvp() as accounting:
            output = torch.nn.functional.silu(value)
            first = torch.autograd.grad(output, value, incoming, create_graph=True)[0]
            torch.testing.assert_close(output, native_output, rtol=0, atol=0)
            torch.testing.assert_close(first, ordinary, rtol=0, atol=0)
            contraction = (first.float() * direction.float()).sum()
            transpose = torch.autograd.grad(contraction, incoming)[0]
            expected = torch.ops.aten.silu_backward.default(direction, value.detach())
            torch.testing.assert_close(transpose, expected, rtol=0, atol=0)
        self.assertIs(torch.nn.functional.silu, original)
        self.assertEqual(accounting['native_silu_forwards'], 1)
        self.assertEqual(accounting['native_silu_first_vjps'], 1)
        self.assertEqual(accounting['silu_cotangent_transposes'], 1)
        difference = ordinary.float() - higher_order.detach().float()
        print(
            f'SiLU BF16 fixture: native create_graph mismatch {mismatch_count}/8192; '
            f'max_abs={float(difference.abs().max()):.9g}; '
            f'relative_l2={float(difference.norm() / ordinary.float().norm()):.9g}; '
            'shim first-VJP and cotangent-transpose native equality=True'
        )

    def test_continuous_silu_transpose_and_shared_swiglu_first_vjp(self):
        generator = torch.Generator().manual_seed(610)
        for dtype, tolerance in ((torch.float64, 1e-10), (torch.float32, 2e-5)):
            value = torch.randn(31, generator=generator, dtype=dtype).requires_grad_()
            direction = torch.randn(31, generator=generator, dtype=dtype)
            _, expected = torch.autograd.functional.jvp(
                torch.nn.functional.silu, value, direction
            )
            incoming = torch.randn(31, generator=generator, dtype=dtype).requires_grad_()
            with native_silu_reverse_jvp():
                output = torch.nn.functional.silu(value)
                gradient = torch.autograd.grad(output, value, incoming, create_graph=True)[0]
                actual = torch.autograd.grad((gradient * direction).sum(), incoming)[0]
            torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)

        hidden = torch.randn(2, 17, 16, generator=generator).bfloat16()
        parameters = [
            torch.randn(shape, generator=generator).bfloat16().requires_grad_()
            for shape in ((16, 48), (16, 48), (48, 16))
        ]

        def swiglu():
            gate, up, down = parameters
            return (torch.nn.functional.silu(hidden @ gate) * (hidden @ up)) @ down

        native_output = swiglu()
        incoming = (
            torch.randn(native_output.shape, generator=generator).bfloat16().requires_grad_()
        )
        reference = torch.autograd.grad(native_output, parameters, incoming)
        with native_silu_reverse_jvp() as accounting:
            output = swiglu()
            actual = torch.autograd.grad(output, parameters, incoming, create_graph=True)
        torch.testing.assert_close(output, native_output, rtol=0, atol=0)
        for left, right in zip(actual, reference):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        self.assertEqual(accounting['native_silu_first_vjps'], 1)

    def test_silu_scope_restoration_passthrough_and_inplace_rejection(self):
        original = torch.nn.functional.silu
        value = torch.tensor([-1.0, 0.0, 1.0], requires_grad=True)
        before = value.detach().clone()
        with self.assertRaisesRegex(RuntimeError, 'fixture exit'):
            with native_silu_reverse_jvp() as accounting:
                with torch.no_grad():
                    output = torch.nn.SiLU()(value)
                torch.testing.assert_close(output, original(value), rtol=0, atol=0)
                self.assertEqual(accounting['passthrough_calls'], 1)
                with self.assertRaisesRegex(ValueError, 'In-place SiLU'):
                    torch.nn.functional.silu(value, inplace=True)
                raise RuntimeError('fixture exit')
        self.assertIs(torch.nn.functional.silu, original)
        torch.testing.assert_close(value, before, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
