"""Pure CPU checks for the fixed, response-normalized repair hinge."""

import unittest

import torch

from online_feedback.first_error import first_error_mask
from online_feedback.recent_repair_retention import (
    RecentRepairRetention, STAT_NAMES, parent_slot_measurements,
    retention_chunk_terms,
)
from specforge.core.chunking import checkpointed_chunk_reduce


def fixture():
    labels = torch.tensor([[[0, 1, 2, 3]]])
    logits = torch.full((1, 1, 4, 5), -2.0)
    logits.scatter_(-1, labels[..., None], 1.0)
    weights = torch.tensor([[[0.0, 1.0, 0.8, 0.6]]])
    assigned = torch.tensor([[[False, False, True, False]]])
    floors = torch.full_like(weights, 0.5)
    return logits, labels, weights, assigned, floors


def terms(logits, labels, weights, assigned, floors, eos=()):
    selected = first_error_mask(logits.detach().argmax(-1), labels, weights, eos)
    return retention_chunk_terms(logits, labels, weights, selected, assigned, floors)


class RetentionArithmeticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_prescribed_margin_examples_and_gradients(self):
        for margin, expected in ((1.4, 0.0), (0.2, 0.3), (-0.1, 0.6), (0.5, 0.0)):
            with self.subTest(margin=margin):
                logits, labels, weights, assigned, floors = fixture()
                logits[0, 0, 2, 2] = margin
                logits[0, 0, 2, 4] = 0
                logits.requires_grad_()
                numerator, stats = terms(logits, labels, weights, assigned, floors)
                numerator.backward()
                self.assertAlmostEqual(float(numerator), 0.8 * expected, places=6)
                gradient = logits.grad[0, 0, 2]
                active = expected > 0
                self.assertAlmostEqual(float(gradient[2]), -0.8 if active else 0, places=6)
                self.assertAlmostEqual(float(gradient[4]), 0.8 if active else 0, places=6)
                self.assertEqual(int(torch.count_nonzero(logits.grad)), 2 if active else 0)
                self.assertEqual(int(stats[4]), int(active))
                self.assertFalse(stats.requires_grad)

    def test_first_error_at_tag_is_protected_but_earlier_error_blocks_it(self):
        logits, labels, weights, assigned, floors = fixture()
        logits[0, 0, 2, 4] = 3
        first, stats = terms(logits, labels, weights, assigned, floors)
        self.assertGreater(float(first), 0)
        self.assertEqual(int(stats[6]), 1)
        logits[0, 0, 1, 4] = 3
        blocked, stats = terms(logits, labels, weights, assigned, floors)
        self.assertEqual(float(blocked), 0)
        self.assertEqual(int(stats[2]), 0)

    def test_rival_changes_and_is_selected_over_full_vocabulary(self):
        logits, labels, weights, assigned, floors = fixture()
        for rival in (0, 4):
            scores = logits.clone()
            scores[0, 0, 2, rival] = 2
            scores.requires_grad_()
            numerator, _ = terms(scores, labels, weights, assigned, floors)
            numerator.backward()
            self.assertAlmostEqual(float(scores.grad[0, 0, 2, rival]), 0.8, places=6)
            self.assertAlmostEqual(float(scores.grad[0, 0, 2, 2]), -0.8, places=6)

    def test_zero_floor_wrong_tie_is_logged_without_hinge(self):
        logits, labels, weights, assigned, floors = fixture()
        floors.zero_()
        logits[0, 0, 2, 0] = logits[0, 0, 2, 2]
        logits.requires_grad_()
        numerator, stats = terms(logits, labels, weights, assigned, floors)
        numerator.backward()
        self.assertEqual(float(numerator), 0)
        self.assertEqual(int(stats[STAT_NAMES.index("wrong_tie_count")]), 1)
        self.assertEqual(int(torch.count_nonzero(logits.grad)), 0)
        logits = logits.detach()
        logits[0, 0, 2, 0] = -2
        logits[0, 0, 2, 4] = logits[0, 0, 2, 2]
        _, stats = terms(logits, labels, weights, assigned, floors)
        self.assertEqual(int(stats[STAT_NAMES.index("correct_tie_count")]), 1)

    def test_fixed_metadata_and_mask_receive_no_gradient(self):
        logits, labels, weights, assigned, floors = fixture()
        logits[0, 0, 2, 4] = 3
        logits.requires_grad_()
        weights.requires_grad_()
        floors.requires_grad_()
        numerator, _ = terms(logits, labels, weights, assigned, floors)
        numerator.backward()
        self.assertIsNone(weights.grad)
        self.assertIsNone(floors.grad)
        self.assertGreater(float(logits.grad.abs().sum()), 0)

    def test_zero_tags_and_zero_hinge_keep_native_graph(self):
        logits, labels, weights, assigned, floors = fixture()
        native = torch.tensor(1.0, requires_grad=True)
        extension = RecentRepairRetention()
        for mask in (assigned, torch.zeros_like(assigned)):
            extras = terms(logits, labels, weights, mask, floors)
            actual, sums, _ = extension.augment_response(native, torch.tensor(9.0), extras)
            self.assertIs(actual, native)
            self.assertEqual(float(sums["retention_auxiliary_loss"]), 0)

    def test_response_normalization_is_native_mass_not_tag_mass(self):
        logits, labels, weights, assigned, floors = fixture()
        logits[0, 0, 2, 4] = 3
        extras = terms(logits, labels, weights, assigned, floors)
        native = torch.tensor(2.0)
        denominator = torch.tensor(100.0 + 1e-6)
        actual, sums, _ = RecentRepairRetention().augment_response(native, denominator, extras)
        expected = native + 0.1 * extras[0] / denominator
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertAlmostEqual(float(sums["retention_hinge_numerator"]), 2.0, places=6)

    def test_chunk_recomputation_and_partial_chunk_are_additive_and_idempotent(self):
        logits, labels, weights, assigned, floors = fixture()
        logits[0, 0, 2, 4] = 3
        values = [tensor.repeat(1, 17, *([1] * (tensor.ndim - 2)))
                  for tensor in (logits, labels, weights, assigned, floors)]
        selection = first_error_mask(values[0].argmax(-1), values[1], values[2])
        chunked_logits = values[0].clone().requires_grad_()
        direct_logits = values[0].clone().requires_grad_()
        chunked = checkpointed_chunk_reduce(
            retention_chunk_terms, chunked_logits, values[1], values[2], selection,
            values[3], values[4], chunk_size=8, dim=1, checkpoint=True,
        )
        direct = retention_chunk_terms(direct_logits, values[1], values[2], selection,
                                       values[3], values[4])
        saved_stats = chunked[1].clone()
        chunked[0].backward()
        direct[0].backward()
        torch.testing.assert_close(chunked[0], direct[0])
        torch.testing.assert_close(chunked[1], direct[1])
        torch.testing.assert_close(chunked_logits.grad, direct_logits.grad, rtol=0, atol=0)
        torch.testing.assert_close(chunked[1], saved_stats, rtol=0, atol=0)

    def test_parent_collector_has_exact_native_ties_and_boundaries(self):
        logits, labels, weights, _, _ = fixture()
        logits = logits.to(torch.bfloat16)
        weights[0, 0, 3] = 0  # cap boundary
        result = parent_slot_measurements(logits, labels, weights)
        self.assertEqual(int(result["correct_prefix"]), 2)
        self.assertFalse(bool(result["valid"][0, 0, 0]))
        self.assertFalse(bool(result["valid"][0, 0, 3]))
        result = parent_slot_measurements(logits, labels, weights, (1,))
        self.assertEqual(int(result["correct_prefix"]), 1)  # EOS itself included
        self.assertFalse(bool(result["valid"][0, 0, 2]))
        result = parent_slot_measurements(logits, labels, weights, (0,))
        self.assertEqual(int(result["correct_prefix"]), 0)  # EOS seed


class RetentionBindingTests(unittest.TestCase):
    def setUp(self):
        self.labels = torch.tensor([[[4, 5, 6, 7], [8, 9, 10, 11]]])
        self.weights = torch.tensor([[[0., 1., .8, .6], [0., 1., .8, .6]]])
        self.anchors = torch.tensor([[104, 108]])
        self.keep = torch.tensor([[True, True]])
        self.row = dict(canonical_start=4, proposal_slot=2, canonical_token=6,
                        parent_margin=2.4, position_weight=float(self.weights[0, 0, 2]))

    def bind(self, rows, **kwargs):
        extension = RecentRepairRetention(**kwargs)
        extension.set_presentation(rows, request_id="r", presentation_id=1,
                                   prompt_length=100, expected_anchors=self.anchors,
                                   expected_keep=self.keep)
        return extension

    def test_offset_identity_and_capped_floor(self):
        mask, floors = self.bind([self.row]).prepare_response(
            self.anchors, self.keep, self.labels, self.weights)
        self.assertEqual(torch.nonzero(mask).tolist(), [[0, 0, 2]])
        self.assertEqual(float(floors[0, 0, 2]), 0.5)
        self.assertFalse(mask.requires_grad or floors.requires_grad)

    def test_input_metadata_is_copied_and_next_presentation_replaces_it(self):
        extension = self.bind([self.row])
        self.row["parent_margin"] = 0
        _, floor = extension.prepare_response(self.anchors, self.keep, self.labels, self.weights)
        self.assertEqual(float(floor[0, 0, 2]), 0.5)
        extension.set_presentation([], request_id="r", presentation_id=2, prompt_length=100,
                                   expected_anchors=self.anchors, expected_keep=self.keep)
        mask, _ = extension.prepare_response(self.anchors, self.keep, self.labels, self.weights)
        self.assertFalse(bool(mask.any()))

    def test_seed_eos_cap_gap_duplicate_and_wrong_identity_rejected(self):
        for change, eos in (({"proposal_slot": 0}, ()), ({"proposal_slot": 4}, ()),
                            ({"canonical_start": 5}, ()), ({"canonical_token": 7}, ()),
                            ({"position_weight": 0.7}, ()), ({}, (5,)),
                            ({"parent_margin": -0.1}, ()), ({"parent_margin": float("nan")}, ())):
            with self.subTest(change=change, eos=eos):
                with self.assertRaises(ValueError):
                    self.bind([{**self.row, **change}]).prepare_response(
                        self.anchors, self.keep, self.labels, self.weights, eos)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.bind([self.row, self.row]).prepare_response(
                self.anchors, self.keep, self.labels, self.weights)
        weights = self.weights.clone()
        weights[0, 0, 1] = 0
        with self.assertRaisesRegex(ValueError, "valid"):
            self.bind([self.row]).prepare_response(self.anchors, self.keep, self.labels, weights)

    def test_sampler_changes_are_not_hidden_even_for_disabled_extension(self):
        for kwargs in ({"enabled": False}, {"coefficient": 0.0}, {}):
            extension = self.bind([], **kwargs)
            with self.assertRaisesRegex(ValueError, "anchors differ"):
                extension.prepare_response(self.anchors + 1, self.keep, self.labels, self.weights)
            with self.assertRaisesRegex(ValueError, "keep mask"):
                extension.prepare_response(self.anchors, ~self.keep, self.labels, self.weights)


if __name__ == "__main__":
    unittest.main()
