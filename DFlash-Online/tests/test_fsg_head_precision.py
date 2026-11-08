"""Small CPU contracts for the opt-in head VJP; native qualification is separate."""
import unittest
from unittest.mock import patch

import torch

from online_feedback.fsg_head_precision import auxiliary_head, fp32_head_vjp


def fixture():
    head = torch.nn.Linear(3, 5, bias=False).bfloat16().requires_grad_(False)
    with torch.no_grad():
        head.weight.copy_(torch.arange(15).reshape(5, 3).bfloat16()/16)
    hidden = (torch.arange(24).reshape(1, 2, 4, 3).bfloat16()/8).requires_grad_(True)
    return head, hidden


class HeadPrecisionTests(unittest.TestCase):
    def test_forward_is_bit_exact_native_and_does_not_mutate_head(self):
        head, hidden = fixture()
        head.train()
        before = head.weight.clone()
        version = head.weight._version
        rng = torch.get_rng_state().clone()
        actual = auxiliary_head(head, hidden)
        self.assertTrue(torch.equal(actual, head(hidden)))
        self.assertEqual(actual.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(before, head.weight))
        self.assertEqual(version, head.weight._version)
        self.assertTrue(head.training)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_custom_backward_keeps_bf16_entry_rounding_and_no_weight_gradient(self):
        head, hidden = fixture()
        cotangent = torch.linspace(-1.003, 1.003, 40).reshape(1, 2, 4, 5)
        logits = auxiliary_head(head, hidden)
        actual, = torch.autograd.grad((logits.float()*cotangent).sum(), hidden)
        expected = (cotangent.bfloat16().float() @ head.weight.float()).bfloat16()
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(actual.dtype, torch.bfloat16)
        self.assertIsNone(head.weight.grad)
        self.assertIsNone(hidden.grad)

    def test_diagnostic_precast_uses_rounded_not_raw_cotangent(self):
        weight = torch.tensor([[1., 2.], [1., 2.]], dtype=torch.bfloat16)
        cotangent = torch.tensor([[1.003, -.997]])
        actual, precast = fp32_head_vjp(cotangent, weight, return_precast=True)
        expected = cotangent.bfloat16().float() @ weight.float()
        self.assertTrue(torch.equal(precast, expected))
        self.assertFalse(torch.equal(precast, cotangent @ weight.float()))
        self.assertTrue(torch.equal(actual, expected.bfloat16()))
        self.assertEqual(precast.dtype, torch.float32)
        self.assertFalse(precast.requires_grad)

    def test_local_ieee_and_autocast_restored_after_success(self):
        weight = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.bfloat16)
        cotangent = torch.tensor([[1.003, -.997]])
        previous = torch.backends.cuda.matmul.fp32_precision
        matmul = torch.matmul
        observed = []
        def inspected(left, right):
            observed.append((torch.backends.cuda.matmul.fp32_precision,
                torch.is_autocast_enabled('cpu'), left.dtype, right.dtype))
            return matmul(left, right)
        try:
            torch.backends.cuda.matmul.fp32_precision = 'tf32'
            with torch.autocast('cpu', dtype=torch.bfloat16):
                with patch('torch.matmul', side_effect=inspected):
                    _, precast = fp32_head_vjp(cotangent, weight, return_precast=True)
                self.assertTrue(torch.is_autocast_enabled('cpu'))
                self.assertEqual(torch.backends.cuda.matmul.fp32_precision, 'tf32')
            self.assertEqual(precast.dtype, torch.float32)
            self.assertEqual(observed, [('ieee', False, torch.float32, torch.float32)])
        finally:
            torch.backends.cuda.matmul.fp32_precision = previous

    def test_local_precision_and_autocast_restored_after_product_exception(self):
        head, _ = fixture()
        previous = torch.backends.cuda.matmul.fp32_precision
        def fail(*_):
            self.assertEqual(torch.backends.cuda.matmul.fp32_precision, 'ieee')
            self.assertFalse(torch.is_autocast_enabled('cpu'))
            raise RuntimeError('product failed')
        try:
            torch.backends.cuda.matmul.fp32_precision = 'tf32'
            with torch.autocast('cpu', dtype=torch.bfloat16):
                with patch('torch.matmul', side_effect=fail):
                    with self.assertRaisesRegex(RuntimeError, 'product failed'):
                        fp32_head_vjp(torch.ones(2, 5), head.weight)
                self.assertTrue(torch.is_autocast_enabled('cpu'))
                self.assertEqual(torch.backends.cuda.matmul.fp32_precision, 'tf32')
        finally:
            torch.backends.cuda.matmul.fp32_precision = previous

    def test_nonfinite_cotangent_and_product_fail_explicitly(self):
        head, hidden = fixture()
        for value in (float('nan'), float('inf')):
            with self.subTest(value=value):
                with self.assertRaisesRegex(FloatingPointError, 'cotangent'):
                    fp32_head_vjp(torch.full((1, 5), value), head.weight)
        with torch.no_grad():
            head.weight[0, 0] = float('nan')
        with self.assertRaisesRegex(FloatingPointError, 'FP32 auxiliary head adjoint'):
            fp32_head_vjp(torch.ones(1, 5), head.weight)
        with self.assertRaisesRegex(FloatingPointError, 'native auxiliary head logits'):
            auxiliary_head(head, hidden)

    def test_frozen_bias_free_native_dtype_contract(self):
        head, hidden = fixture()
        head.weight.requires_grad_(True)
        with self.assertRaisesRegex(ValueError, 'frozen BF16'):
            auxiliary_head(head, hidden)
        with self.assertRaisesRegex(ValueError, 'frozen BF16'):
            fp32_head_vjp(torch.ones(1, 5), head.weight)
        head.requires_grad_(False)
        with self.assertRaisesRegex(ValueError, 'BF16 hidden'):
            auxiliary_head(head, hidden.float())
        biased = torch.nn.Linear(3, 5).bfloat16().requires_grad_(False)
        with self.assertRaisesRegex(ValueError, 'bias-free'):
            auxiliary_head(biased, hidden)

    def test_first_order_vjp_does_not_retain_fp32_weight_copy(self):
        head, hidden = fixture()
        saved = []
        with torch.autograd.graph.saved_tensors_hooks(lambda x: saved.append(x) or x,
                                                     lambda x: x):
            logits = auxiliary_head(head, hidden)
            torch.autograd.grad(logits.float().sum(), hidden)
        self.assertEqual(len(saved), 1)
        self.assertIs(saved[0], head.weight)
        self.assertEqual(saved[0].dtype, torch.bfloat16)


if __name__ == '__main__':
    unittest.main()
