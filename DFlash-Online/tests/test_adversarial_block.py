"""CPU tensor checks only: no DFlash loads, native training, or serving runs."""

import random
import unittest

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from online_feedback.adversarial_block import (
    FrozenKPlusChunk, FrozenKPlusResponse, apply_mask_delta, clean_delta,
    make_perturbation, replay_model_rng, uses_two_views,
)
from online_feedback.first_error import first_error_mask
from online_feedback.kplus import group_terms, redistribute_response
from online_feedback.rng import capture_rng


def fixture():
    labels = torch.tensor([[[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]])
    proposals = labels.clone()
    proposals[0, 0, 2:] = 9  # Priority one.
    proposals[0, 1, 1] = 9  # Priority two: immediate successor is correct.
    weights = torch.tensor([[[0., 1., .7, .5, .3], [0., .9, .6, .4, .2],
                             [0., .8, .5, .3, .1]]])
    return labels, proposals, weights


def native_terms(rows, proposals, labels, weights):
    selected = first_error_mask(proposals, labels, weights)
    return ((rows * (weights * selected)).sum(), (weights * selected).sum(),
            rows.new_zeros(()), rows.new_zeros(()), rows.new_zeros(8),
            *group_terms(rows, proposals, labels, weights, selected))


def frozen_terms(chunk, rows):
    numerator, selected, *extra = chunk.terms(rows)
    return (numerator, selected, rows.new_zeros(()), rows.new_zeros(()),
            rows.new_zeros(8), *extra)


def sum_chunks(chunks):
    total = chunks[0]
    for terms in chunks[1:]:
        total = tuple(left + right for left, right in zip(total, terms))
    return total


class AdversarialBlockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def test_disabled_attack_bypasses_but_clean2_requires_real_views(self):
        for arm, rho in (("KPLUS", .01), ("ADV", 0), ("RANDOM", 0)):
            self.assertFalse(uses_two_views(arm, rho))
        for arm in ("CLEAN2", "ADV", "RANDOM"):
            self.assertTrue(uses_two_views(arm))
        self.assertTrue(uses_two_views("CLEAN2", 0))
        for arm, rho in (("unknown", .01), ("ADV", -1), ("ADV", float("nan"))):
            with self.assertRaises(ValueError):
                uses_two_views(arm, rho)

    def test_eligibility_includes_unselected_suffix_but_respects_gap_eos_seed(self):
        labels, proposals, weights = fixture()
        labels[0, 1, 3] = 20
        weights[0, 2, 2] = 0
        saved = FrozenKPlusChunk.capture(proposals, labels, weights, (20,))
        self.assertEqual(saved.eligible.tolist(), [[[False, True, True, True, True],
                                                   [False, True, True, True, False],
                                                   [False, True, False, False, False]]])
        self.assertTrue(bool((saved.eligible & ~saved.selection).any()))
        labels[0, 0, 0] = 20
        ended = FrozenKPlusChunk.capture(proposals, labels, weights, (20,))
        self.assertFalse(bool(ended.eligible[0, 0].any()))

    def test_clean_capture_owns_immutable_detached_input_copies(self):
        labels, proposals, weights = fixture()
        weights.requires_grad_()
        selected = first_error_mask(proposals, labels, weights)
        saved = FrozenKPlusChunk.capture(proposals, labels, weights, selection=selected)
        expected = saved.weights.clone()
        with torch.no_grad():
            weights.zero_()
            selected.zero_()
            proposals.zero_()
        self.assertTrue(torch.equal(saved.weights, expected))
        self.assertTrue(bool(saved.selection.any()))
        self.assertFalse(saved.weights.requires_grad)

    def test_bf16_native_chunk_order_loss_and_gradient_are_exact(self):
        labels, proposals, weights = fixture()
        generator = torch.Generator().manual_seed(41)
        logits = torch.randn(1, 3, 5, 13, generator=generator).bfloat16()
        teacher = torch.randn(logits.shape, generator=generator).bfloat16()
        left = logits.clone().requires_grad_()
        right = logits.clone().requires_grad_()

        def kl(values):
            return F.kl_div(F.log_softmax(values, -1), F.softmax(teacher, -1),
                            reduction="none").sum(-1)

        original_rows, copied_rows = kl(left), kl(right)
        clean_chunks, original_parts, copied_parts = [], [], []
        for start, stop in ((0, 2), (2, 3)):
            selected_slice = slice(start, stop)
            saved = FrozenKPlusChunk.capture(proposals[:, selected_slice],
                                             labels[:, selected_slice], weights[:, selected_slice])
            clean_chunks.append(saved)
            original_parts.append(native_terms(original_rows[:, selected_slice],
                proposals[:, selected_slice], labels[:, selected_slice], weights[:, selected_slice]))
            copied_parts.append(frozen_terms(saved, copied_rows[:, selected_slice]))
        original = sum_chunks(original_parts)
        copied = sum_chunks(copied_parts)
        denominator = original[1] + 1e-6
        response = FrozenKPlusResponse.capture(clean_chunks, original, denominator)
        reference = redistribute_response(original[0], original[1], original[5:])[0] / denominator
        actual, _ = response.loss(copied)
        reference.backward()
        actual.backward()
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        torch.testing.assert_close(right.grad, left.grad, rtol=0, atol=0)
        self.assertEqual(response.denominator.item(), denominator.item())

    def test_fixed_selection_does_not_disappear_when_perturbed_predictions_change(self):
        labels, proposals, weights = fixture()
        saved = FrozenKPlusChunk.capture(proposals, labels, weights)
        rows = torch.ones_like(weights)
        original = native_terms(rows, proposals, labels, weights)
        response = FrozenKPlusResponse.capture((saved,), original, original[1] + 1e-6)
        changed_proposals = torch.full_like(proposals, 10)
        self.assertFalse(torch.equal(saved.selection,
                                    first_error_mask(changed_proposals, labels, weights)))
        changed_rows = torch.arange(weights.numel()).reshape_as(weights).float().requires_grad_()
        terms = frozen_terms(saved, changed_rows)
        # Bogus second-view masses cannot replace the clean pooled values.
        wrong = list(terms)
        for index in (1, 5, 6, 9, 10):
            wrong[index] = wrong[index] + 100
        actual, _ = response.loss(wrong)
        expected, _ = response.loss(terms)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.backward()
        self.assertTrue(torch.equal(changed_rows.grad != 0, saved.selection))

    def test_neutral_response_returns_exact_native_numerator_object(self):
        labels, proposals, weights = fixture()
        proposals.copy_(labels)
        rows = torch.randn(weights.shape, requires_grad=True)
        terms = native_terms(rows, proposals, labels, weights)
        chunk = FrozenKPlusChunk.capture(proposals, labels, weights)
        response = FrozenKPlusResponse.capture((chunk,), terms, terms[1] + 1e-6)
        numerator, _ = response.reduce_response(terms)
        self.assertIs(numerator, terms[0])

    def test_checkpoint_recomputation_reuses_indexed_metadata_without_mutation(self):
        labels, proposals, weights = fixture()
        saved = FrozenKPlusChunk.capture(proposals, labels, weights)
        before = saved.selection.clone()
        leaf = torch.randn(weights.shape, requires_grad=True)
        calls = []

        def objective(values):
            calls.append(1)
            return saved.terms(values.square())[0]

        checkpoint(objective, leaf, use_reentrant=False).backward()
        self.assertGreaterEqual(len(calls), 2)
        self.assertTrue(torch.equal(saved.selection, before))
        expected = 2 * leaf.detach() * (weights * saved.selection)
        torch.testing.assert_close(leaf.grad, expected, rtol=0, atol=0)

    def test_one_clean_backward_supplies_bf16_parameter_and_fp32_delta_gradients(self):
        labels, proposals, weights = fixture()
        eligible = FrozenKPlusChunk.capture(proposals, labels, weights).eligible
        embeddings = torch.ones(1, 3, 5, 8, dtype=torch.bfloat16, requires_grad=True)
        delta = clean_delta(embeddings)
        parameter = torch.ones(8, dtype=torch.bfloat16, requires_grad=True)
        actual = apply_mask_delta(embeddings, delta, eligible)
        # One clean backward has the declared .5 view / 2 response scaling.
        loss = (actual * parameter).square().float().sum()
        (loss * .5 / 2).backward()
        self.assertEqual(parameter.grad.dtype, torch.bfloat16)
        self.assertEqual(delta.grad.dtype, torch.float32)
        self.assertIsNone(embeddings.grad)
        self.assertTrue(bool((delta.grad[eligible] != 0).all()))
        self.assertFalse(bool((delta.grad[~eligible] != 0).any()))
        self.assertTrue(torch.equal(actual, embeddings))

    def attack_fixture(self):
        labels, proposals, weights = fixture()
        eligible = FrozenKPlusChunk.capture(proposals, labels, weights).eligible
        embeddings = torch.full((1, 3, 5, 8), 2, dtype=torch.bfloat16)
        gradient = torch.arange(embeddings.numel()).reshape_as(embeddings).float() + 1
        return embeddings, eligible, gradient

    def test_adversarial_radius_support_detachment_and_realized_norm(self):
        embeddings, eligible, gradient = self.attack_fixture()
        delta, metrics = make_perturbation(embeddings, eligible, gradient, arm="ADV", key=("gsm", 5, 0, 1))
        torch.testing.assert_close(metrics["intended_norm"], metrics["nominal_radius"],
                                   rtol=2e-6, atol=1e-8)
        self.assertFalse(delta.requires_grad)
        self.assertFalse(bool((delta[~eligible] != 0).any()))
        changed = apply_mask_delta(embeddings, delta, eligible)
        self.assertEqual(changed.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(changed[~eligible], embeddings[~eligible]))
        realized = torch.linalg.vector_norm(changed.float() - embeddings.float(), dim=(-2, -1))
        torch.testing.assert_close(metrics["realized_norm"], realized, rtol=0, atol=0)
        self.assertTrue(bool((metrics["changed_slots"] > 0).all()))
        self.assertFalse(torch.equal(metrics["realized_norm"], metrics["nominal_radius"]))

    def test_random_is_keyed_and_does_not_consume_ordinary_rng(self):
        embeddings, eligible, gradient = self.attack_fixture()
        before = torch.get_rng_state().clone()
        first, random_metrics = make_perturbation(embeddings, eligible, gradient,
                                               arm="RANDOM", key=("gsm", 5, 0, 1))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        second, _ = make_perturbation(embeddings, eligible, gradient,
                                      arm="RANDOM", key=("gsm", 5, 0, 1))
        other, _ = make_perturbation(embeddings, eligible, gradient,
                                    arm="RANDOM", key=("gsm", 5, 0, 2))
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, other))
        adv, adv_metrics = make_perturbation(embeddings, eligible, gradient,
                                            arm="ADV", key=("gsm", 5, 0, 1))
        self.assertFalse(torch.equal(first, adv))
        torch.testing.assert_close(random_metrics["intended_norm"], adv_metrics["intended_norm"],
                                   rtol=2e-6, atol=1e-8)

    def test_random_and_adv_share_zero_gradient_and_no_eligible_convention(self):
        embeddings, eligible, gradient = self.attack_fixture()
        gradient[:, 0] = 0
        eligible[:, 1] = False
        for arm in ("RANDOM", "ADV"):
            delta, metrics = make_perturbation(embeddings, eligible, gradient,
                                              arm=arm, key=("mathqa", 6, 1, 17))
            self.assertTrue(torch.equal(delta[:, :2], torch.zeros_like(delta[:, :2])))
            self.assertEqual(metrics["zero_gradient_blocks"].tolist(), [[True, False, False]])
            self.assertEqual(metrics["no_eligible_blocks"].tolist(), [[False, True, False]])

    def test_absent_graph_and_nonfinite_eligible_gradient_fail_visibly(self):
        embeddings, eligible, gradient = self.attack_fixture()
        with self.assertRaisesRegex(RuntimeError, "did not reach"):
            make_perturbation(embeddings, eligible, None, arm="ADV", key=(1,))
        gradient[0, 0, 1, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            make_perturbation(embeddings, eligible, gradient, arm="ADV", key=(1,))

    def test_tiny_positive_gradient_normalizes_before_radius_without_overflow(self):
        embeddings = torch.tensor([[[[2.]]]], dtype=torch.bfloat16)
        eligible = torch.ones(1, 1, 1, dtype=torch.bool)
        gradient = torch.tensor([[[[1e-39]]]], dtype=torch.float32)
        delta, metrics = make_perturbation(embeddings, eligible, gradient, arm="ADV", key=(1,))
        self.assertGreater(metrics["gradient_norm"].item(), 0)
        torch.testing.assert_close(delta, torch.full_like(delta, .02), rtol=0, atol=0)
        self.assertTrue(bool(torch.isfinite(delta).all()))

    def test_block_direction_ignores_invalid_slots_and_gradient_scale(self):
        embeddings, eligible, gradient = self.attack_fixture()
        first, _ = make_perturbation(embeddings, eligible, gradient, arm="ADV", key=(1,))
        gradient[~eligible] = float("nan")
        second, _ = make_perturbation(embeddings, eligible, gradient * .25, arm="ADV", key=(1,))
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_pair_replays_stochastic_draws_and_restores_post_clean_rng(self):
        torch.manual_seed(71)
        np.random.seed(71)
        random.seed(71)
        torch.rand(5)  # Represents the ordinary anchor sampling before capture.
        pre_clean = capture_rng()
        clean = (torch.rand(7), np.random.rand(3), random.random())
        post_clean = capture_rng()
        with replay_model_rng(pre_clean):
            second = (torch.rand(7), np.random.rand(3), random.random())
        self.assertTrue(torch.equal(clean[0], second[0]))
        self.assertTrue(np.array_equal(clean[1], second[1]))
        self.assertEqual(clean[2], second[2])
        self.assertTrue(torch.equal(torch.get_rng_state(), post_clean["torch"]))
        self.assertEqual(random.getstate(), post_clean["python"])
        self.assertTrue(np.array_equal(np.random.get_state()[1], post_clean["numpy"][1]))

    def test_pair_restores_rng_even_if_second_view_fails(self):
        pre_clean = capture_rng()
        torch.rand(3)
        after_clean = torch.get_rng_state().clone()
        with self.assertRaisesRegex(RuntimeError, "fixture"):
            with replay_model_rng(pre_clean):
                torch.rand(19)
                raise RuntimeError("fixture")
        self.assertTrue(torch.equal(torch.get_rng_state(), after_clean))


if __name__ == "__main__":
    unittest.main()
