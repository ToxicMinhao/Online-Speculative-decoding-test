"""Numerical contracts for the detached, fixed projected-target constructor."""

import unittest

import torch

from online_feedback.projected_target import MARGIN_RATIO, projected_targets


PROBABILITY_ATOL = 2e-6
ORACLE_ATOL = 2e-7
ORACLE_RTOL = 2e-5


def oracle_targets(logits, labels):
    """Independent FP64 solution by bisection of the conserved total mass."""
    probabilities = logits.double().softmax(dim=-1)
    rows = probabilities.reshape(-1, logits.shape[-1]).clone()
    predictions = logits.argmax(dim=-1).reshape(-1)
    for row, label, prediction in zip(rows, labels.reshape(-1), predictions):
        label = int(label)
        if int(prediction) == label:
            continue
        competitors = torch.cat((row[:label], row[label + 1 :]))
        lower, upper = 0.0, 1.0 / MARGIN_RATIO
        for _ in range(100):
            threshold = (lower + upper) / 2
            mass = MARGIN_RATIO * threshold + competitors.clamp(max=threshold).sum()
            if float(mass) < 1.0:
                lower = threshold
            else:
                upper = threshold
        threshold = (lower + upper) / 2
        row.clamp_(max=threshold)
        row[label] = MARGIN_RATIO * threshold
    return rows.reshape_as(probabilities)


class ProjectedTargetTests(unittest.TestCase):
    def assert_distribution(self, result):
        self.assertEqual(result.dtype, torch.float32)
        self.assertTrue(torch.isfinite(result).all())
        self.assertTrue((result >= 0).all())
        torch.testing.assert_close(
            result.sum(dim=-1), torch.ones_like(result[..., 0]),
            atol=PROBABILITY_ATOL, rtol=0,
        )

    def assert_repair_margin(self, result, labels):
        target = result.gather(-1, labels.unsqueeze(-1))
        competitors = result.scatter(-1, labels.unsqueeze(-1), 0)
        violation = MARGIN_RATIO * competitors.amax(dim=-1, keepdim=True) - target
        self.assertTrue((violation <= PROBABILITY_ATOL).all())
        self.assertTrue(torch.equal(result.argmax(dim=-1), labels))

    def test_worked_example(self):
        logits = torch.tensor([[0.40, 0.35, 0.05, 0.20]]).log()
        result = projected_targets(logits, torch.tensor([2]))
        expected = torch.tensor([[0.258064516, 0.258064516, 0.283870968, 0.20]])
        torch.testing.assert_close(result, expected, atol=ORACLE_ATOL, rtol=ORACLE_RTOL)
        self.assert_distribution(result)

    def test_correct_rows_are_unchanged_even_without_a_margin(self):
        logits = torch.tensor([[1.0, 0.999, -4.0], [2.0, 2.0, -1.0]])
        result = projected_targets(logits, torch.tensor([0, 0]))
        self.assertTrue(torch.equal(result, logits.softmax(dim=-1)))

    def test_lowest_id_tie_is_preserved_and_other_tied_label_is_repaired(self):
        logits = torch.zeros(3, 4)
        labels = torch.tensor([0, 1, 3])
        result = projected_targets(logits, labels)
        self.assertTrue(torch.equal(result[0], logits[0].softmax(dim=-1)))
        self.assertTrue(torch.equal(result[1:].argmax(dim=-1), labels[1:]))
        self.assert_repair_margin(result[1:], labels[1:])
        self.assert_distribution(result)

    def test_batched_shapes_preserve_correct_rows_and_repair_only_errors(self):
        generator = torch.Generator().manual_seed(17)
        logits = torch.randn(2, 3, 19, generator=generator)
        labels = torch.tensor([[0, 5, 11], [12, 3, 8]])
        labels[0, 0] = logits[0, 0].argmax()
        result = projected_targets(logits, labels)
        correct = logits.argmax(dim=-1) == labels
        self.assertTrue(torch.equal(result[correct], logits.softmax(dim=-1)[correct]))
        self.assert_repair_margin(result[~correct], labels[~correct])
        self.assert_distribution(result)
        torch.testing.assert_close(
            result.double(), oracle_targets(logits, labels),
            atol=ORACLE_ATOL, rtol=ORACLE_RTOL,
        )

    def test_targets_are_detached_and_current_logit_gradient_is_q_minus_r(self):
        reference = torch.tensor([[3.0, 2.0, 0.0]], requires_grad=True)
        targets = projected_targets(reference, torch.tensor([2]))
        self.assertFalse(targets.requires_grad)
        self.assertIsNone(targets.grad_fn)
        current = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        loss = -(targets * current.float().log_softmax(dim=-1)).sum()
        loss.backward()
        self.assertIsNone(reference.grad)
        torch.testing.assert_close(
            current.grad, current.detach().softmax(dim=-1) - targets,
            atol=ORACLE_ATOL, rtol=ORACLE_RTOL,
        )

    def test_bfloat16_logits_are_upcast_before_probability_arithmetic(self):
        logits = torch.tensor([[2.3, 1.1, -0.4, -6.0]], dtype=torch.bfloat16)
        labels = torch.tensor([2])
        result = projected_targets(logits, labels)
        self.assertTrue(torch.equal(result, projected_targets(logits.float(), labels)))
        self.assert_distribution(result)
        torch.testing.assert_close(
            result.double(), oracle_targets(logits, labels),
            atol=ORACLE_ATOL, rtol=ORACLE_RTOL,
        )

    def test_softmax_underflow_is_not_floored_or_renormalized(self):
        logits = torch.tensor([[1000.0, 0.0, -1000.0], [1000.0, 0.0, -1000.0]])
        labels = torch.tensor([0, 1])
        result = projected_targets(logits, labels)
        self.assertTrue(torch.equal(result[0], torch.tensor([1.0, 0.0, 0.0])))
        self.assertEqual(float(result[1, 2]), 0.0)
        torch.testing.assert_close(
            result[1], torch.tensor([1 / 2.1, 1.1 / 2.1, 0.0]),
            atol=ORACLE_ATOL, rtol=ORACLE_RTOL,
        )
        self.assert_distribution(result)
        self.assert_repair_margin(result[1:], labels[1:])

    def test_extreme_finite_logits_remain_finite(self):
        limit = torch.finfo(torch.float32).max
        logits = torch.tensor([[limit, -limit, 0.0], [-limit, -limit, -limit]])
        labels = torch.tensor([1, 2])
        result = projected_targets(logits, labels)
        self.assert_distribution(result)
        self.assert_repair_margin(result, labels)
        torch.testing.assert_close(
            result.double(), oracle_targets(logits, labels),
            atol=ORACLE_ATOL, rtol=ORACLE_RTOL,
        )

    def test_stress_matches_independent_fp64_oracle(self):
        generator = torch.Generator().manual_seed(20261022)
        for scale in (0.01, 1.0, 20.0, 200.0):
            logits = torch.randn(12, 257, generator=generator) * scale
            labels = torch.randint(257, (12,), generator=generator)
            with self.subTest(scale=scale):
                result = projected_targets(logits, labels)
                incorrect = logits.argmax(dim=-1) != labels
                self.assert_distribution(result)
                self.assert_repair_margin(result[incorrect], labels[incorrect])
                torch.testing.assert_close(
                    result.double(), oracle_targets(logits, labels),
                    atol=ORACLE_ATOL, rtol=ORACLE_RTOL,
                )

    def test_full_vocabulary_mass_and_margin(self):
        generator = torch.Generator().manual_seed(19)
        logits = torch.randn(2, 151936, generator=generator)
        labels = torch.tensor([17, 151935])
        result = projected_targets(logits, labels)
        self.assert_distribution(result)
        self.assert_repair_margin(result, labels)
        torch.testing.assert_close(
            result.double(), oracle_targets(logits, labels),
            atol=ORACLE_ATOL, rtol=ORACLE_RTOL,
        )

    def test_nonfinite_reference_is_rejected_without_repair(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "finite"):
                projected_targets(torch.tensor([[value, 0.0]]), torch.tensor([1]))


if __name__ == "__main__":
    unittest.main()
