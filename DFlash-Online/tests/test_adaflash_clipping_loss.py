"""CPU checks for clipping placement under the common probability arithmetic."""

import unittest
from unittest import mock

import torch

from experiments.adaflash_clipping.loss import (
    ClippingComparisonLoss, clipping_contract, position_clipped_rows,
)
from online_feedback.adaflash_opd import (
    PaperAdaFlashOPD, chunk_terms as original_paper_terms,
)
from online_feedback.objective import TargetObjectiveModel
from specforge.core.chunking import checkpointed_chunk_reduce
from test_objective_integration import tiny_author, parameter_gradients
from test_osd_objective import osd_batch


def direct_response(logits, teacher, weights, mode):
    """Unchunked mathematical oracle with clipping expressed by where."""
    log_q = torch.log_softmax(logits, -1)
    log_p = torch.log_softmax(teacher.detach(), -1)
    entries = log_q.exp() * (log_q - log_p)
    ce = -torch.take_along_dim(
        log_q, teacher.detach().argmax(-1, keepdim=True), -1,
    ).squeeze(-1)
    denominator = weights.sum() + 1e-6
    if mode == "paper_entry":
        clipped = torch.where(entries > .01, entries.new_tensor(.01), entries)
        return ((.8 * ce + .2 * clipped.sum(-1)) * weights).sum() / denominator
    rows = entries.sum(-1) / denominator
    clipped = torch.where(rows > .01, rows.new_tensor(.01), rows)
    return .8 * (ce * weights).sum() / denominator + .2 * (clipped * weights).sum()


def evaluate(extension, logits, teacher, weights, *, checkpoint=True):
    extension.prepare_response(weights)
    terms = checkpointed_chunk_reduce(
        extension.chunk_terms, logits, teacher, teacher.detach().argmax(-1), weights,
        chunk_size=extension.chunk_blocks, dim=1, checkpoint=checkpoint,
    )
    return terms[0] / (terms[1] + 1e-6), terms


class ClippingLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def fixture(self, dtype=torch.float64):
        generator = torch.Generator().manual_seed(511)
        logits = torch.randn(1, 9, 4, 7, generator=generator, dtype=dtype)
        teacher = torch.randn(logits.shape, generator=generator, dtype=dtype)
        weights = torch.tensor([0., 1., .5, .25], dtype=torch.float64 if dtype == torch.float64 else torch.float32)
        weights = weights.expand(1, 9, 4).clone()
        weights[:, -1, 2:] = 0
        return logits, teacher, weights

    def test_both_rules_match_direct_fp64_values_and_gradients(self):
        for mode in ("paper_entry", "repo_position"):
            logits, teacher, weights = self.fixture()
            actual_leaf = logits.clone().requires_grad_()
            teacher.requires_grad_()
            actual, _ = evaluate(ClippingComparisonLoss(mode, chunk_blocks=3), actual_leaf, teacher, weights)
            actual.backward()
            expected_leaf = logits.clone().requires_grad_()
            expected = direct_response(expected_leaf, teacher, weights, mode)
            expected.backward()
            torch.testing.assert_close(actual, expected, atol=2e-15, rtol=2e-15)
            torch.testing.assert_close(actual_leaf.grad, expected_leaf.grad, atol=2e-16, rtol=2e-14)
            self.assertIsNone(teacher.grad)

    def test_paper_branch_exactly_preserves_a58_bf16_terms_and_gradient(self):
        logits, teacher, weights = self.fixture(torch.bfloat16)
        old_leaf = logits.clone().requires_grad_()
        old = checkpointed_chunk_reduce(
            original_paper_terms, old_leaf, teacher, teacher.argmax(-1), weights,
            chunk_size=8, dim=1, checkpoint=True,
        )
        old_loss = old[0] / (old[1] + 1e-6)
        old_loss.backward()
        new_leaf = logits.clone().requires_grad_()
        new_loss, new = evaluate(ClippingComparisonLoss("paper_entry"), new_leaf, teacher, weights)
        new_loss.backward()
        for expected, actual in zip(old, new[:5]):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(new_loss, old_loss, rtol=0, atol=0)
        torch.testing.assert_close(new_leaf.grad, old_leaf.grad, rtol=0, atol=0)

    def test_normalization_uses_entire_response_for_every_chunk(self):
        logits, teacher, weights = self.fixture()
        outcomes = []
        for size in (1, 3, 8, 64):
            leaf = logits.clone().requires_grad_()
            extension = ClippingComparisonLoss("repo_position", chunk_blocks=size)
            loss, terms = evaluate(extension, leaf, teacher, weights)
            loss.backward()
            torch.testing.assert_close(extension.denominator, terms[1] + 1e-6, rtol=0, atol=0)
            outcomes.append((loss.detach(), leaf.grad))
        for actual in outcomes[1:]:
            for value, expected in zip(actual, outcomes[0]):
                torch.testing.assert_close(value, expected, atol=2e-15, rtol=2e-13)
        wrong = sum(direct_response(logits[:, start:start+1], teacher[:, start:start+1],
                                   weights[:, start:start+1], "repo_position")
                    * weights[:, start:start+1].sum() for start in range(9)) / weights.sum()
        self.assertGreater(abs(float(outcomes[0][0] - wrong)), 1e-3)

    def test_fp32_denominator_matches_existing_chunk_reduction_order(self):
        logits, teacher, weights = self.fixture(torch.bfloat16)
        weights = torch.rand(weights.shape, generator=torch.Generator().manual_seed(22))
        extension = ClippingComparisonLoss("repo_position", chunk_blocks=3)
        _, terms = evaluate(extension, logits, teacher, weights)
        torch.testing.assert_close(extension.denominator, terms[1] + 1e-6, rtol=0, atol=0)

    def test_position_cap_depends_on_response_denominator(self):
        raw = torch.tensor([.5, .001, -.02], dtype=torch.float64, requires_grad=True)
        small, _ = position_clipped_rows(raw, torch.tensor(1., dtype=torch.float64))
        small.sum().backward()
        torch.testing.assert_close(small, torch.tensor([.01, .001, -.02], dtype=torch.float64))
        torch.testing.assert_close(raw.grad, torch.tensor([0., 1., 1.], dtype=torch.float64))
        raw.grad = None
        large, _ = position_clipped_rows(raw, torch.tensor(100., dtype=torch.float64))
        large.sum().backward()
        torch.testing.assert_close(large, raw.detach(), rtol=0, atol=0)
        torch.testing.assert_close(raw.grad, torch.ones_like(raw), rtol=0, atol=0)

    def test_uncapped_bf16_rows_recover_common_raw_rkl_exactly(self):
        raw = torch.linspace(-.5, .5, 257).to(torch.bfloat16).requires_grad_()
        for denominator in (64.5, 239.817, 1017.55, 3000.25):
            effective, _ = position_clipped_rows(raw, torch.tensor(denominator))
            torch.testing.assert_close(effective, raw, rtol=0, atol=0)
            gradient, = torch.autograd.grad(effective.sum(), raw)
            torch.testing.assert_close(gradient, torch.ones_like(raw), rtol=0, atol=0)

    def test_inactive_positions_have_zero_gradient_and_no_cap_counts(self):
        logits, teacher, weights = self.fixture()
        for mode in ("paper_entry", "repo_position"):
            leaf = logits.clone().requires_grad_()
            extension = ClippingComparisonLoss(mode, chunk_blocks=3)
            loss, terms = evaluate(extension, leaf, teacher, weights)
            before = terms[5].clone()
            loss.backward()
            self.assertTrue(torch.equal(leaf.grad[weights == 0], torch.zeros_like(leaf.grad[weights == 0])))
            torch.testing.assert_close(terms[5], before, rtol=0, atol=0)
            self.assertEqual(int(terms[5][5]), int((weights > 0).sum()))
            self.assertEqual(int(terms[5][4]), int((weights > 0).sum()) * logits.shape[-1])

    def test_zero_supervision_has_zero_loss_gradient_and_metrics(self):
        logits, teacher, weights = self.fixture(torch.bfloat16)
        weights.zero_()
        for mode in ("paper_entry", "repo_position"):
            leaf = logits.clone().requires_grad_()
            extension = ClippingComparisonLoss(mode)
            loss, terms = evaluate(extension, leaf, teacher, weights)
            loss.backward()
            self.assertEqual(float(loss.detach()), 0)
            self.assertTrue(torch.equal(leaf.grad, torch.zeros_like(leaf)))
            self.assertTrue(torch.equal(terms[4], torch.zeros_like(terms[4])))
            self.assertTrue(torch.equal(terms[5], torch.zeros_like(terms[5])))

    def test_consecutive_responses_finish_checkpoint_backward_before_rebinding(self):
        extension = ClippingComparisonLoss("repo_position", chunk_blocks=3)
        logits, teacher, weights = self.fixture()
        for scale in (1., 50.):
            current_weights = weights * scale
            leaf = logits.clone().requires_grad_()
            loss, _ = evaluate(extension, leaf, teacher, current_weights)
            loss.backward()  # The runner must complete this before the next prepare.
            reference = logits.clone().requires_grad_()
            expected = direct_response(reference, teacher, current_weights, "repo_position")
            expected.backward()
            torch.testing.assert_close(loss, expected, atol=2e-15, rtol=2e-15)
            torch.testing.assert_close(leaf.grad, reference.grad, atol=2e-16, rtol=2e-14)

    def test_bound_denominator_is_detached_and_not_a_mask_view(self):
        weights = torch.ones(1, 2, 4, requires_grad=True)
        extension = ClippingComparisonLoss("repo_position")
        denominator = extension.prepare_response(weights)
        self.assertFalse(denominator.requires_grad)
        with torch.no_grad():
            weights.zero_()
        torch.testing.assert_close(denominator, torch.tensor(8. + 1e-6), rtol=0, atol=0)

    def test_contract_and_metrics_label_the_common_arithmetic_scope(self):
        self.assertFalse(clipping_contract("repo_position")["official_kernel_reproduction"])
        logits, teacher, weights = self.fixture()
        extension = ClippingComparisonLoss("repo_position")
        _, terms = evaluate(extension, logits, teacher, weights)
        sums, ratios = extension.response_metrics(terms, terms[1] + 1e-6)
        self.assertEqual(int(sums["adaflash_response_count"]), 1)
        self.assertIn("clipping_effective_reverse_kl_loss", ratios)
        torch.testing.assert_close(ratios["clipping_effective_reverse_kl_loss"][1], extension.denominator)
        with self.assertRaisesRegex(ValueError, "Unknown clipping rule"):
            ClippingComparisonLoss("other")
        with self.assertRaisesRegex(RuntimeError, "complete response"):
            ClippingComparisonLoss("repo_position").chunk_terms(logits, teacher, teacher.argmax(-1), weights)

    def test_native_bf16_position_cap_matches_explicit_common_arithmetic(self):
        logits, teacher, weights = self.fixture(torch.bfloat16)
        leaf = logits.clone().requires_grad_()
        extension = ClippingComparisonLoss("repo_position", chunk_blocks=64)
        actual, _ = evaluate(extension, leaf, teacher, weights)
        actual.backward()
        oracle = logits.clone().requires_grad_()
        log_q, log_p = oracle.log_softmax(-1), teacher.log_softmax(-1)
        raw = (log_q.exp() * (log_q - log_p)).sum(-1)
        denominator = weights.sum() + 1e-6
        normalized = raw.float() / denominator
        capped = torch.where(normalized > .01, normalized.new_tensor(.01), normalized)
        effective = (capped * denominator).to(torch.bfloat16)
        ce = -log_q.gather(-1, teacher.argmax(-1, keepdim=True)).squeeze(-1)
        expected = ((.8 * ce + .2 * effective) * weights).sum() / denominator
        expected.backward()
        self.assertGreater(int((normalized > .01).sum()), 0)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(leaf.grad, oracle.grad, rtol=0, atol=0)

    def test_tiny_dflash_paper_hook_preserves_historical_loss_and_model_gradients(self):
        anchors = torch.tensor([[4, 8, 10, 0]])
        keep = torch.tensor([[True, True, True, False]])
        for chunk_blocks in (2, 8):
            model = TargetObjectiveModel(
                tiny_author(chunk_blocks=chunk_blocks), "osd", loss_precision="bfloat16",
            )
            model.begin_event()
            batch = osd_batch(model)
            batch["loss_mask"][0, 6] = 0
            model.point_loss_extension = PaperAdaFlashOPD()
            with mock.patch.object(model, "_sample_anchor_positions", return_value=(anchors, keep)):
                expected, _, old_metrics = model(**batch)
                expected.backward()
                old_gradients = parameter_gradients(model)
                model.zero_grad(set_to_none=True)
                extension = ClippingComparisonLoss("paper_entry", chunk_blocks=chunk_blocks)
                model.point_loss_extension = extension
                actual, _, metrics = model(**batch)
                actual.backward()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for name, gradient in parameter_gradients(model).items():
                torch.testing.assert_close(gradient, old_gradients[name], rtol=0, atol=0)
            torch.testing.assert_close(extension.denominator, metrics["loss_terms"][1], rtol=0, atol=0)
            self.assertEqual(int(metrics["sum_metrics"]["adaflash_available_positions"]), 6)
            for name in old_metrics["sum_metrics"]:
                torch.testing.assert_close(metrics["sum_metrics"][name], old_metrics["sum_metrics"][name], rtol=0, atol=0)

    def test_tiny_dflash_position_cap_rebinds_full_denominator_between_responses(self):
        model = TargetObjectiveModel(tiny_author(chunk_blocks=2), "osd", loss_precision="bfloat16")
        model.begin_event()
        extension = ClippingComparisonLoss("repo_position", chunk_blocks=2)
        model.point_loss_extension = extension
        batch = osd_batch(model)
        anchors = torch.tensor([[4, 8, 10, 0]])
        keep = torch.tensor([[True, True, True, False]])
        denominators = []
        for masked_position in (None, 9):
            if masked_position is not None:
                batch["loss_mask"][0, masked_position] = 0
            with mock.patch.object(model, "_sample_anchor_positions", return_value=(anchors, keep)):
                loss, _, metrics = model(**batch)
                bound = extension.denominator.clone()
                loss.backward()
            torch.testing.assert_close(extension.denominator, bound, rtol=0, atol=0)
            torch.testing.assert_close(bound, metrics["loss_terms"][1], rtol=0, atol=0)
            self.assertTrue(all(torch.isfinite(value).all() for value in parameter_gradients(model).values()))
            denominators.append(float(bound))
            model.zero_grad(set_to_none=True)
        self.assertLess(denominators[1], denominators[0])


if __name__ == "__main__":
    unittest.main()
