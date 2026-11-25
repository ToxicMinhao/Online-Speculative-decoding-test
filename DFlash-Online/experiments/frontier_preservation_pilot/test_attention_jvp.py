"""Continuous attention algebra and the derivative-only dispatch boundary."""

from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from torch.nn.attention.flex_attention import AuxOutput, create_block_mask

from .attention_jvp import attention_jvp_wrapper, attention_tangent, native_attention_jvp


def dense_attention(
    query,
    key,
    value,
    *,
    block_mask=None,
    scale=None,
    enable_gqa=False,
    return_aux=None,
    **kwargs
):
    """Independent small ordinary-attention reference, including empty rows."""
    groups = query.shape[1] // key.shape[1]
    key = key.repeat_interleave(groups, 1)
    value = value.repeat_interleave(groups, 1)
    scores = query @ key.transpose(-2, -1)
    scores = scores * (query.shape[-1] ** -0.5 if scale is None else scale)
    if block_mask is not None:
        b = torch.arange(query.shape[0]).view(-1, 1, 1, 1)
        h = torch.arange(query.shape[1]).view(1, -1, 1, 1)
        q = torch.arange(query.shape[2]).view(1, 1, -1, 1)
        k = torch.arange(key.shape[2]).view(1, 1, 1, -1)
        scores = scores.masked_fill(~block_mask.mask_mod(b, h, q, k), -torch.inf)
    lse = torch.logsumexp(scores, dim=-1)
    empty = torch.isneginf(lse)
    safe_scores = torch.where(empty.unsqueeze(-1), 0.0, scores)
    probabilities = safe_scores.softmax(-1).masked_fill(empty.unsqueeze(-1), 0.0)
    output = probabilities @ value
    return (output, AuxOutput(lse=lse, max_scores=None)) if return_aux else output


class AttentionDerivativeTests(unittest.TestCase):
    def setUp(self):
        self.generator = torch.Generator().manual_seed(835)

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized(), 'CPU qualification initialized CUDA')

    def tensors(self, dtype, *, batch=2, query_length=7, key_length=11):
        shapes = (
            (batch, 4, query_length, 5),
            (batch, 2, key_length, 5),
            (batch, 2, key_length, 3),
        )
        primals = [
            torch.randn(shape, dtype=dtype, generator=self.generator) for shape in shapes
        ]
        tangents = [
            torch.randn(shape, dtype=dtype, generator=self.generator) for shape in shapes
        ]
        return primals, tangents

    def mask(self, batch=2, query_length=7, key_length=11):
        def mask_mod(b, h, q, k):
            # Head/batch dependent support, and one entirely masked query row.
            return (q != query_length - 1) & ((k + h + b) % 3 != 0) & (k <= q + 3)

        return create_block_mask(mask_mod, batch, 4, query_length, key_length, device='cpu')

    def test_fp64_and_fp32_match_dense_autograd_and_adjoint(self):
        for dtype, tolerance in ((torch.float64, 1e-10), (torch.float32, 2e-5)):
            with self.subTest(dtype=dtype):
                primals, tangents = self.tensors(dtype)
                block_mask = self.mask()

                def forward(*values):
                    return dense_attention(
                        *values, block_mask=block_mask, scale=0.37, enable_gqa=True
                    )

                output, expected = torch.autograd.functional.jvp(
                    forward, tuple(primals), tuple(tangents)
                )
                _, auxiliary = dense_attention(
                    *primals, block_mask=block_mask, scale=0.37, return_aux=True
                )
                accounting = {}
                actual = attention_tangent(
                    *primals,
                    *tangents,
                    output=output,
                    lse=auxiliary.lse,
                    block_mask=block_mask,
                    scale=0.37,
                    enable_gqa=True,
                    query_tile=3,
                    key_tile=4,
                    accounting=accounting,
                )
                torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
                self.assertEqual(int(torch.count_nonzero(actual[:, :, -1])), 0)
                self.assertLessEqual(accounting['max_score_elements'], 2 * 4 * 3 * 4)
                leaves = [value.clone().requires_grad_() for value in primals]
                cotangent = torch.randn(output.shape, dtype=dtype, generator=self.generator)
                gradients = torch.autograd.grad(forward(*leaves), leaves, cotangent)
                reference_dot = sum(
                    (g * direction).sum() for g, direction in zip(gradients, tangents)
                )
                actual_dot = (actual * cotangent).sum()
                torch.testing.assert_close(
                    actual_dot, reference_dot, rtol=tolerance, atol=tolerance
                )

    def test_each_direction_branch_and_broadcast_gqa(self):
        primals, directions = self.tensors(torch.float64)
        primals[1:] = [value[:1] for value in primals[1:]]
        directions[1:] = [value[:1] for value in directions[1:]]
        output, auxiliary = dense_attention(*primals, return_aux=True)
        for branch in range(3):
            selected = [
                direction if index == branch else None
                for index, direction in enumerate(directions)
            ]
            zeros = [
                direction if index == branch else torch.zeros_like(direction)
                for index, direction in enumerate(directions)
            ]
            _, expected = torch.autograd.functional.jvp(
                dense_attention, tuple(primals), tuple(zeros)
            )
            actual = attention_tangent(
                *primals,
                *selected,
                output=output,
                lse=auxiliary.lse,
                enable_gqa=True,
                query_tile=3,
                key_tile=4,
            )
            torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)

    def test_sparse_metadata_skips_disallowed_blocks_without_changing_support(self):
        primals, tangents = self.tensors(
            torch.float32, batch=1, query_length=129, key_length=260
        )

        def mask_mod(b, h, q, k):
            return (q < 128) & (q // 128 == k // 128)

        block_mask = create_block_mask(mask_mod, 1, 1, 129, 260, device='cpu')
        output, auxiliary = dense_attention(*primals, block_mask=block_mask, return_aux=True)
        accounting = {}
        actual = attention_tangent(
            *primals,
            *tangents,
            output=output,
            lse=auxiliary.lse,
            block_mask=block_mask,
            enable_gqa=True,
            accounting=accounting,
        )
        _, expected = torch.autograd.functional.jvp(
            lambda *values: dense_attention(*values, block_mask=block_mask),
            tuple(primals),
            tuple(tangents),
        )
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
        self.assertEqual(accounting['query_tiles'], 2)
        self.assertEqual(accounting['key_tiles'], 1)
        self.assertLessEqual(accounting['max_score_elements'], 4 * 128 * 128)

    def test_native_bf16_policy_dtype_and_fully_masked_rows(self):
        primals, tangents = self.tensors(torch.bfloat16)
        block_mask = self.mask()
        output, auxiliary = dense_attention(
            *[value.float() for value in primals], block_mask=block_mask, return_aux=True
        )
        tangent = attention_tangent(
            *primals,
            *tangents,
            output=output.bfloat16(),
            lse=auxiliary.lse,
            block_mask=block_mask,
            enable_gqa=True,
            query_tile=3,
            key_tile=4,
        )
        self.assertEqual(tangent.dtype, torch.bfloat16)
        self.assertTrue(bool(torch.isfinite(tangent).all()))
        self.assertEqual(int(torch.count_nonzero(tangent[:, :, -1])), 0)

    def test_mixed_full_partial_blocks_and_nonsquare_sparse_metadata(self):
        primals, tangents = self.tensors(
            torch.float64, batch=2, query_length=139, key_length=259
        )

        def mask_mod(b, h, q, k):
            return (q < 138) & ((k < 64) | ((k < q + b + 5) & (k % 3 != 0)))

        block_mask = create_block_mask(
            mask_mod, 2, 1, 139, 259, device='cpu', BLOCK_SIZE=(32, 64)
        )
        self.assertGreater(int(block_mask.full_kv_num_blocks.sum()), 0)
        self.assertGreater(int(block_mask.kv_num_blocks.sum()), 0)
        output, auxiliary = dense_attention(*primals, block_mask=block_mask, return_aux=True)
        actual = attention_tangent(
            *primals,
            *tangents,
            output=output,
            lse=auxiliary.lse,
            block_mask=block_mask,
            enable_gqa=True,
            query_tile=37,
            key_tile=31,
        )
        _, expected = torch.autograd.functional.jvp(
            lambda *values: dense_attention(*values, block_mask=block_mask),
            tuple(primals),
            tuple(tangents),
        )
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)

    def test_derivative_precision_restores_ambient_settings_even_on_error(self):
        primals, tangents = self.tensors(torch.float32)
        output, auxiliary = dense_attention(*primals, return_aux=True)
        arguments = dict(output=output, lse=auxiliary.lse, enable_gqa=True)
        expected = attention_tangent(*primals, *tangents, **arguments)
        previous = torch.backends.cuda.matmul.fp32_precision
        try:
            torch.backends.cuda.matmul.fp32_precision = 'tf32'
            with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
                actual = attention_tangent(*primals, *tangents, **arguments)
                self.assertTrue(torch.is_autocast_enabled('cpu'))
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                with self.assertRaisesRegex(ValueError, 'tile sizes'):
                    attention_tangent(*primals, *tangents, query_tile=0, **arguments)
                self.assertTrue(torch.is_autocast_enabled('cpu'))
            self.assertEqual(torch.backends.cuda.matmul.fp32_precision, 'tf32')
        finally:
            torch.backends.cuda.matmul.fp32_precision = previous

    def test_wrapper_ordinary_outputs_and_gradients_are_bit_identical(self):
        primals, _ = self.tensors(torch.float64)
        leaves = [value.requires_grad_() for value in primals]
        accounting = {}
        native = mock.Mock(side_effect=dense_attention)
        wrapper = attention_jvp_wrapper(native, accounting)
        reference = dense_attention(*leaves, enable_gqa=True)
        output = wrapper(*leaves, enable_gqa=True)
        torch.testing.assert_close(output, reference, rtol=0, atol=0)
        expected = torch.autograd.grad(reference.sum(), leaves)
        actual = torch.autograd.grad(output.sum(), leaves)
        for left, right in zip(actual, expected):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        native.assert_called_once_with(*leaves, enable_gqa=True)
        self.assertEqual(accounting, {'ordinary_calls': 1})

    def test_dual_wrapper_uses_native_primal_auxiliary_once_and_preserves_rng(self):
        primals, tangents = self.tensors(torch.float64)
        block_mask = self.mask()
        accounting = {}
        native = mock.Mock(side_effect=dense_attention)
        wrapper = attention_jvp_wrapper(native, accounting)
        expected, expected_tangent = torch.autograd.functional.jvp(
            lambda *values: dense_attention(*values, block_mask=block_mask),
            tuple(primals),
            tuple(tangents),
        )
        before = torch.get_rng_state().clone()
        with torch.autograd.forward_ad.dual_level():
            duals = [
                torch.autograd.forward_ad.make_dual(primal, tangent)
                for primal, tangent in zip(primals, tangents)
            ]
            dual_output = wrapper(*duals, block_mask=block_mask, enable_gqa=True)
            output, tangent = torch.autograd.forward_ad.unpack_dual(dual_output)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        torch.testing.assert_close(tangent, expected_tangent, rtol=1e-10, atol=1e-10)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(native.call_count, 1)
        self.assertTrue(native.call_args.kwargs['return_aux'].lse)
        self.assertEqual(accounting['dual_calls'], 1)
        self.assertEqual(accounting['native_aux_forwards'], 1)
        self.assertEqual(accounting['derivative_fp32_precision'], 'ieee')
        self.assertFalse(accounting['derivative_autocast'])
        self.assertTrue(accounting['precision_restored'])

    def test_scope_restores_original_after_error_and_rejects_modified_scores(self):
        primals, tangents = self.tensors(torch.float32)
        module = SimpleNamespace(compile_friendly_flex_attention=dense_attention)
        with mock.patch(
            'experiments.frontier_preservation_pilot.attention_jvp.importlib.import_module',
            return_value=module,
        ):
            with self.assertRaisesRegex(RuntimeError, 'fixture'):
                with native_attention_jvp():
                    self.assertIsNot(module.compile_friendly_flex_attention, dense_attention)
                    raise RuntimeError('fixture')
            self.assertIs(module.compile_friendly_flex_attention, dense_attention)
        wrapper = attention_jvp_wrapper(dense_attention, {})
        with torch.autograd.forward_ad.dual_level():
            duals = [
                torch.autograd.forward_ad.make_dual(primal, tangent)
                for primal, tangent in zip(primals, tangents)
            ]
            with self.assertRaisesRegex(ValueError, 'vanilla'):
                wrapper(*duals, score_mod=lambda score, *args: score + 1)
            with self.assertRaisesRegex(ValueError, 'Prescaled'):
                wrapper(*duals, kernel_options={'PRESCALE_QK': True})


if __name__ == '__main__':
    unittest.main()
