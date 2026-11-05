"""Focused CPU checks of diagnostic arithmetic and censored block prefixes."""

import math
import unittest

import torch

from fit_metrics import block_prefix_metrics, summarize_positions


def metrics(reference, current, target, labels, weights=None):
    reference = torch.as_tensor(reference).reshape(1, 1, -1, len(reference[0]))
    current = torch.as_tensor(current).reshape_as(reference)
    target = torch.as_tensor(target).reshape_as(reference)
    labels = torch.as_tensor(labels).reshape(reference.shape[:-1])
    weights = torch.ones_like(labels) if weights is None else torch.as_tensor(weights).reshape_as(labels)
    return summarize_positions(reference, current, target, labels, weights)


class PositionMetricsTests(unittest.TestCase):
    def test_known_distribution_and_kl_identity(self):
        r = [0.4, 0.3, 0.2, 0.1]
        q = [0.1, 0.2, 0.3, 0.4]
        logits = [math.log(v) for v in q]
        result = metrics([logits], [logits], [r], [0])
        expected_ce = -sum(a * math.log(b) for a, b in zip(r, q))
        expected_entropy = -sum(a * math.log(a) for a in r)
        self.assertAlmostEqual(result["soft_ce"].item(), expected_ce, places=6)
        self.assertAlmostEqual(result["target_entropy"].item(), expected_entropy, places=6)
        self.assertAlmostEqual(result["kl_to_r"].item(), expected_ce - expected_entropy, places=6)
        self.assertAlmostEqual(result["target_token_logit_margin_vs_best_other"].item(), math.log(0.25), places=6)
        self.assertAlmostEqual(result["projected_probability_margin"].item(), 0.1, places=6)
        self.assertAlmostEqual(result["projected_log_margin"].item(), math.log(4 / 3), places=6)
        for value in result.values():
            self.assertEqual(value.shape, (1, 1, 1))
            self.assertEqual(value.device.type, "cpu")
            self.assertFalse(value.requires_grad)

    def test_repair_damage_and_zero_weight_rows_are_not_silently_removed(self):
        result = metrics([[3., 1.], [1., 3.]], [[1., 3.], [3., 1.]], [[0.7, 0.3]] * 2, [0, 0], [0, 1])
        self.assertEqual(result["reference_correct"].flatten().tolist(), [True, False])
        self.assertEqual(result["repaired"].flatten().tolist(), [False, True])
        self.assertEqual(result["damaged"].flatten().tolist(), [True, False])

    def test_native_bfloat16_argmax_tie_uses_lowest_id(self):
        logits = torch.tensor([[25.125, 25.125, 25.0]], dtype=torch.bfloat16)
        result = metrics(logits, logits, [[0.4, 0.4, 0.2]], [1])
        self.assertEqual(result["predicted_id"].item(), 0)
        self.assertFalse(result["reference_correct"].item())
        self.assertEqual(result["target_token_logit_margin_vs_best_other"].item(), 0.)

    def test_underflow_q_and_zero_r_still_have_finite_forward_kl(self):
        result = metrics([[0., -1000., -2000.]], [[0., -1000., -2000.]], [[0., 1., 0.]], [1])
        self.assertEqual(result["target_entropy"].item(), 0.)
        self.assertEqual(result["kl_to_r"].item(), 1000.)
        self.assertEqual(result["soft_ce"].item(), 1000.)
        self.assertEqual(result["projected_log_margin"].item(), math.inf)

    def test_native_argmax_is_not_replaced_by_rounded_softmax_argmax(self):
        logits = torch.tensor([[0., 1e-8]], dtype=torch.bfloat16)
        self.assertEqual(logits.float().softmax(-1).argmax(-1).item(), 0)
        result = metrics(logits, logits, [[0.5, 0.5]], [1])
        self.assertEqual(result["predicted_id"].item(), 1)
        self.assertTrue(result["reference_correct"].item())

    def test_negative_roundoff_is_preserved(self):
        # A one-ULP mass drift is not silently normalized or clipped away.
        result = metrics([[0., 0.]], [[0., 0.]], [[0.5 - 2**-24, 0.5]], [0])
        self.assertLess(result["kl_to_r"].item(), 0.)

    def test_no_autograd_and_inputs_unchanged(self):
        logits = torch.tensor([[[[1., 2., 3.]]]], requires_grad=True)
        target = logits.detach().softmax(-1)
        before = logits.detach().clone()
        result = summarize_positions(logits, logits, target, torch.tensor([[[2]]]), torch.ones(1, 1, 1))
        self.assertTrue(torch.equal(logits, before))
        self.assertTrue(all(not value.requires_grad for value in result.values()))
        self.assertAlmostEqual(result["kl_to_r"].item(), 0., places=6)


class PrefixMetricsTests(unittest.TestCase):
    def test_later_repair_cannot_pass_early_error(self):
        before = block_prefix_metrics([1, 9, 9, 4], [1, 2, 3, 4], [1] * 4, [], "eos")
        after = block_prefix_metrics([1, 9, 3, 4], [1, 2, 3, 4], [1] * 4, [], "eos")
        self.assertEqual(before["matching_prefix_length"], 1)
        self.assertEqual(after["matching_prefix_length"], 1)
        self.assertEqual(after["first_mismatch_slot"], 2)

    def test_matched_eos_is_counted_once_and_stops_prefix(self):
        result = block_prefix_metrics([1, 99, 8], [1, 99, 7], [1, 1, 1], [99], "eos")
        self.assertEqual(result["matching_prefix_length"], 2)
        self.assertEqual(result["matching_non_eos_prefix_length"], 1)
        self.assertTrue(result["matched_eos"])
        self.assertIsNone(result["first_mismatch_slot"])
        self.assertFalse(result["cap_boundary_censored"])

    def test_wrong_eos_is_a_mismatch(self):
        result = block_prefix_metrics([1, 9], [1, 99], [1, 1], [99], "eos")
        self.assertEqual(result["matching_prefix_length"], 1)
        self.assertFalse(result["matched_eos"])
        self.assertEqual(result["first_mismatch_slot"], 2)

    def test_cap_tail_is_censored_only_when_all_available_slots_match(self):
        result = block_prefix_metrics([1, 2, 9], [1, 2, 0], [1, 1, 0], [], "length", truncated_by_cap=True)
        self.assertEqual(result["matching_prefix_length"], 2)
        self.assertTrue(result["cap_boundary_censored"])
        result = block_prefix_metrics([1, 9, 9], [1, 2, 0], [1, 1, 0], [], "length", truncated_by_cap=True)
        self.assertFalse(result["cap_boundary_censored"])

    def test_internal_full_block_is_not_censored_by_response_finish_reason(self):
        result = block_prefix_metrics([1, 2], [1, 2], [1, 1], [], "length")
        self.assertFalse(result["cap_boundary_censored"])

    def test_eligibility_gap_is_not_skipped(self):
        result = block_prefix_metrics([1, 2, 3], [1, 2, 3], [1, 0, 1], [], "length", truncated_by_cap=True)
        self.assertEqual(result["matching_prefix_length"], 1)
        self.assertEqual(result["eligible_count"], 2)
        self.assertTrue(result["eligibility_gap"])
        self.assertFalse(result["cap_boundary_censored"])

    def test_no_eligible_positions_are_not_censored(self):
        result = block_prefix_metrics([1, 2], [1, 2], [0, 0], [], "length", truncated_by_cap=True)
        self.assertEqual(result["matching_prefix_length"], 0)
        self.assertFalse(result["cap_boundary_censored"])


if __name__ == "__main__":
    unittest.main()
