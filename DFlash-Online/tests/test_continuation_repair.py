"""CPU qualification of the fixed within-HIGH continuation allocation.

Literal FP64 coefficient/gradient fixtures are independent of the factorized
reducer. Native tests preserve the original BF16 graph and chunk interfaces.
"""

import copy
import unittest
from unittest import mock

import torch

from online_feedback.continuation_repair import (
    ContinuationRepair, continuation_scores, high_pool_coefficients,
    install_continuation_repair,
)
from online_feedback.first_error import first_error_mask
from online_feedback.kplus import redistribute_response
from online_feedback.repair_conditional_kl import repair_pair_reduce_response
from specforge.core.chunking import checkpointed_chunk_reduce
from test_kplus_objective import mixed_logits
from test_repair_conditional_kl import make_model
from test_osd_objective import osd_batch


def literal_coefficients(beta, high, scores):
    """Direct FP64 definition, without production reductions/helpers."""
    beta, scores = beta.detach().double(), scores.detach().double()
    total = sum(beta.tolist())
    high_mass = sum(float(b) for b, h in zip(beta, high) if h)
    result = beta.clone()
    if not high_mass or high_mass == total:
        return result
    fraction = high_mass / total
    c = .5 * (1-fraction) / (1+fraction)
    kappa = min(c, .5-c)
    mean = sum(float(b*u) for b, h, u in zip(beta, high, scores) if h) / high_mass
    for index, (b, h, u) in enumerate(zip(beta, high, scores)):
        native = float(b) * (.5 + .5 * (2 if h else 1) / (1+fraction))
        result[index] = native + (kappa * float(b) * (float(u)-mean) if h else 0)
    return result


def native_pool(beta, high):
    zero = beta.new_zeros(())
    _, metrics = redistribute_response(zero, beta.sum(), (
        beta[~high].sum(), beta[high].sum(), zero, zero,
        (~high).sum().float(), high.sum().float()))
    return metrics


def reduce_fixture(beta, high, scores, fkl, pair):
    """Build the existing HIGH moments and call only the new response reducer."""
    metrics = native_pool(beta, high)
    native = beta * torch.where(high, metrics['kplus_coefficient2'], metrics['kplus_coefficient1'])
    prefix = fkl.new_tensor(.8)
    native_numerator = prefix + (native * (fkl+pair)).sum()
    native_terms = [fkl.new_zeros(()) for _ in range(11)]
    native_terms[8] = (beta[high] * fkl[high]).sum()
    pair_terms = (fkl.new_zeros(()), (beta[high] * pair[high]).sum(), fkl.new_zeros(1))
    moments = ((beta[high]*scores[high]).sum().detach(),
               (beta[high]*scores[high].detach()*fkl[high]).sum(),
               (beta[high]*scores[high].detach()*pair[high]).sum())
    extension = ContinuationRepair()
    extension._rows = [(beta, high, scores)]
    result, summary = extension.reduce_response(
        native_numerator, moments, native_terms, pair_terms, metrics)
    return result, native_numerator, summary


def reduced_native(model, logits, teacher, labels, weights, chunk_size=8, checkpoint=True):
    extension = getattr(model, 'continuation_repair', None)
    if extension is None:
        terms = checkpointed_chunk_reduce(model._chunk_loss, logits, None, labels, weights, teacher,
                                          chunk_size=chunk_size, dim=1, checkpoint=checkpoint)
    else:
        with extension.collect_response():
            terms = checkpointed_chunk_reduce(model._chunk_loss, logits, None, labels, weights, teacher,
                                              chunk_size=chunk_size, dim=1, checkpoint=checkpoint)
    numerator, metrics = model.first_error_redistribution.reduce_response(terms[:11])
    numerator, _ = model.prefix_allocation.reduce_response(numerator, terms[11:15])
    pair, pair_metrics = repair_pair_reduce_response(terms[15:18], metrics)
    if bool(pair_metrics['repair_pair_repair_rows'] > 0):
        numerator = numerator + pair
    if extension is not None:
        numerator, _ = extension.reduce_response(
            numerator, terms[18:], terms[:11], terms[15:18], metrics, terms[11:15])
    return numerator/(terms[1]+1e-6), terms


class ContinuationRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def test_prespecified_six_repair_fixture(self):
        beta = torch.ones(6)
        high = torch.tensor([True, True, False, False, False, False])
        scores = torch.tensor([0., 1., 0., 0., 0., 0.])
        actual, native, pool = high_pool_coefficients(
            beta, high, scores, native_pool(beta, high), (beta[high]*scores[high]).sum())
        torch.testing.assert_close(actual, torch.tensor([1.125, 1.375, .875, .875, .875, .875]),
                                   rtol=0, atol=0)
        self.assertEqual(float(actual.sum()), 6.)
        self.assertAlmostEqual(float(pool['kappa']), .25, places=7)
        self.assertEqual(float(pool['changed_high_positions']), 2.)
        self.assertTrue(torch.equal(actual[~high], native[~high]))

    def test_unequal_weights_match_fp64_oracle_mass_and_relative_order(self):
        generator = torch.Generator().manual_seed(811)
        for count in (3, 17, 512):
            beta = torch.rand(count, generator=generator)*.9+.1
            high = torch.arange(count) % 3 != 0
            scores = torch.rand(count, generator=generator)
            actual, native, pool = high_pool_coefficients(
                beta, high, scores, native_pool(beta, high), (beta[high]*scores[high]).sum())
            expected = literal_coefficients(beta, high, scores)
            tolerance = 2e-7+5e-6*beta.double().abs()
            self.assertTrue(bool(((actual.double()-expected).abs() <= tolerance).all()))
            self.assertLessEqual(float((actual[high].sum()-native[high].sum()).abs()),
                                 float(tolerance[high].sum()))
            self.assertTrue(torch.equal(actual[~high], native[~high]))
            order = scores[high].argsort()
            self.assertTrue(bool(((actual[high]/beta[high])[order].diff() >= 0).all()))
            self.assertTrue(bool((actual[high] >= beta[high]-tolerance[high]).all()))
            self.assertTrue(bool((actual[high] <= 1.5*beta[high]+tolerance[high]).all()))
            self.assertTrue(pool['active'])

    def test_all_neutral_pools_return_native_numerator_object(self):
        fixtures = [([], []), ([False, False], [0., 0.]), ([True, True], [.1, .8]),
                    ([True, False], [.7, 0.]), ([True, False, True], [.4, 0., .4])]
        for high, scores in fixtures:
            high = torch.tensor(high, dtype=torch.bool)
            scores = torch.tensor(scores)
            beta = torch.ones(len(high))
            fkl = torch.arange(len(high), dtype=torch.float32, requires_grad=True)
            pair = torch.ones(len(high), requires_grad=True)
            actual, original, summary = reduce_fixture(beta, high, scores, fkl, pair)
            self.assertIs(actual, original)
            self.assertEqual(float(summary['continuation_active_response']), 0.)
            self.assertEqual(float(summary['continuation_changed_high_positions']), 0.)

    def test_factorization_matches_literal_loss_and_both_gradients(self):
        beta = torch.tensor([.7, .2, .5, .3, .9, .4])
        high = torch.tensor([True, False, True, True, False, True])
        scores = torch.tensor([.2, 0., .8, .1, 0., .4], requires_grad=True)
        fkl = torch.tensor([.2, 1.1, .5, .8, .4, .7], requires_grad=True)
        pair = torch.tensor([.7, .1, .6, .1, .4, .3], requires_grad=True)
        actual, _, summary = reduce_fixture(beta, high, scores, fkl, pair)
        gradients = torch.autograd.grad(actual, (fkl, pair, scores), allow_unused=True)
        coefficients = literal_coefficients(beta, high, scores)
        expected = .8 + (coefficients * (fkl.detach().double()+pair.detach().double())).sum()
        torch.testing.assert_close(actual.double(), expected, rtol=2e-6, atol=2e-7)
        for gradient in gradients[:2]:
            torch.testing.assert_close(gradient.double(), coefficients, rtol=2e-6, atol=2e-7)
        self.assertIsNone(gradients[2])
        self.assertNotEqual(float(summary['continuation_fk_correction']), 0.)
        self.assertNotEqual(float(summary['continuation_pair_correction']), 0.)

    def test_enabled_neutral_pools_keep_native_bf16_loss_and_gradients_exact(self):
        labels = torch.zeros((1, 6, 16), dtype=torch.long)
        weights = torch.ones_like(labels, dtype=torch.float32)
        weights[..., 0] = 0
        teacher = torch.ones((1, 6, 16, 3), dtype=torch.bfloat16)
        teacher[..., 1] = -.5
        for scenario in ('empty', 'no_error', 'all_low', 'all_high', 'singleton', 'constant'):
            logits = torch.zeros_like(teacher)
            logits[..., 0] = 2
            local = weights.clone()
            if scenario == 'empty':
                local.zero_()
            if scenario not in ('empty', 'no_error'):
                logits[:, :, 1] = torch.tensor([0., 2., 0.])
                low_start = {'all_low': 0, 'all_high': 6, 'singleton': 1, 'constant': 2}[scenario]
                logits[:, low_start:, 2] = torch.tensor([0., 2., 0.])
            results = []
            for enabled in (False, True):
                model = make_model(True)
                install_continuation_repair(model, enabled=enabled)
                current = logits.clone().requires_grad_()
                loss, _ = reduced_native(model, current, teacher, labels, local, chunk_size=2)
                loss.backward()
                results.append((loss.detach(), current.grad))
                if enabled:
                    self.assertEqual(float(model.continuation_repair.last_summary['continuation_active_response']), 0.)
            for actual, native in zip(results[1], results[0]):
                torch.testing.assert_close(actual, native, rtol=0, atol=0)

    def test_zero_scalar_correction_still_changes_gradient(self):
        beta = torch.ones(6)
        high = torch.tensor([True, True, False, False, False, False])
        scores = torch.tensor([0., 1., 0., 0., 0., 0.])
        fkl = torch.ones(6, requires_grad=True)
        pair = torch.ones(6, requires_grad=True)
        actual, native, summary = reduce_fixture(beta, high, scores, fkl, pair)
        self.assertEqual(float(summary['continuation_total_correction']), 0.)
        self.assertEqual(float(actual.detach()), float(native.detach()))
        self.assertIsNot(actual, native)
        observed = torch.autograd.grad(actual, (fkl, pair), retain_graph=True)
        baseline = torch.autograd.grad(native, (fkl, pair))
        expected_change = torch.tensor([-.125, .125, 0., 0., 0., 0.])
        for current, original in zip(observed, baseline):
            torch.testing.assert_close(current-original, expected_change, rtol=0, atol=0)
        self.assertEqual(float(summary['continuation_changed_high_positions']), 2.)

    def test_score_native_ties_seed_eos_gap_and_later_error(self):
        labels = torch.zeros((1, 10, 16), dtype=torch.long)
        logits = torch.zeros((1, 10, 16, 3), dtype=torch.bfloat16)
        logits[..., 0] = 2
        weights = torch.ones_like(labels, dtype=torch.float32)
        weights[..., 0] = 0
        def wrong(block, offset):
            logits[0, block, offset] = torch.tensor([0., 2., -2.])
        for block in (0, 1, 2, 4, 5, 7):
            wrong(block, 1)
        logits[0, 0, 3, 0] = 4
        wrong(0, 4)  # Later correct positions cannot restart running support.
        logits[0, 1, 2] = torch.tensor([2., 2., 0.])  # Correct native tie: HIGH, u=0.
        labels[0, 2, 2] = 1
        logits[0, 2, 2] = torch.tensor([2., 2., 0.])  # Wrong native tie: LOW.
        wrong(3, 14)
        for block, offset in ((3, 15), (4, 2)):
            labels[0, block, offset] = 2
            logits[0, block, offset] = torch.tensor([0., 0., 2.])
        weights[0, 5, 2] = 0
        labels[0, 6, 0] = 2
        wrong(6, 1)
        labels[0, 7, 1] = 2  # First error is canonical EOS: no followers.
        wrong(9, 15)
        proposals = logits.argmax(-1)
        selection = first_error_mask(proposals, labels, weights, (2,))
        score, low, high = continuation_scores(logits, proposals, labels, weights, selection, (2,))
        one = torch.tanh(torch.tensor(1.))/14
        torch.testing.assert_close(score[0], torch.tensor([2*one, 0, 0, one, one, 0, 0, 0, 0, 0]))
        self.assertEqual(high[0].any(-1).tolist(), [True, True, False, True, True, False, False, False, False, False])
        self.assertEqual(low[0].any(-1).tolist(), [False, False, True, False, False, True, False, True, False, True])
        self.assertFalse(score.requires_grad)

    def test_native_terms_chunk_pooling_checkpoint_and_no_direct_suffix_gradient(self):
        logits, teacher, labels, weights = mixed_logits()
        observed = []
        for chunk, checkpoint in ((1, False), (8, False), (8, True), (64, True)):
            model = make_model(True)
            install_continuation_repair(model)
            captures = []
            model.continuation_repair.coefficient_probe = captures.append
            current = logits.clone().requires_grad_()
            loss, terms = reduced_native(model, current, teacher, labels, weights, chunk, checkpoint)
            self.assertEqual(len(terms), 21)
            self.assertEqual(len(captures), 1)
            capture = captures[0]
            self.assertEqual(len(capture['beta']), 16)
            self.assertEqual(int(capture['high'].sum()), 8)
            self.assertGreater(float(capture['summary']['continuation_changed_high_positions']), 0)
            expected = literal_coefficients(capture['beta'], capture['high'], capture['scores'])
            tolerance = 2e-7+5e-6*capture['beta'].double()
            self.assertTrue(bool(((capture['candidate'].double()-expected).abs() <= tolerance).all()))
            before = copy.deepcopy(model.continuation_repair.last_summary)
            loss.backward()
            self.assertEqual(len(captures), 1)
            self.assertEqual(model.continuation_repair._rows, [])
            for key, value in before.items():
                self.assertTrue(torch.equal(value, model.continuation_repair.last_summary[key]))
            selection = first_error_mask(logits.argmax(-1), labels, weights)
            self.assertTrue(bool((current.grad[~selection] == 0).all()))
            observed.append((loss.detach(), current.grad, capture['candidate']))
            baseline = make_model(True)
            baseline_prefix = []
            baseline.prefix_allocation.coefficient_probe = baseline_prefix.append
            _, old_terms = reduced_native(baseline, logits.clone().requires_grad_(), teacher,
                                           labels, weights, chunk, checkpoint)
            old_coefficients = torch.cat([row['candidate'].reshape(-1, 16) for row in baseline_prefix])
            old_prefix = torch.cat([row['prefix'].reshape(-1, 16) for row in baseline_prefix])
            self.assertTrue(torch.equal(capture['preservation_coefficients'], old_coefficients))
            self.assertTrue(torch.equal(capture['preservation_mask'], old_prefix))
            self.assertEqual(model.continuation_repair._prefix_rows, [])
            for actual, old in zip(terms[:18], old_terms):
                torch.testing.assert_close(actual, old, rtol=0, atol=0)
        for result in observed[1:]:
            torch.testing.assert_close(result[0], observed[0][0], rtol=2e-6, atol=2e-7)
            torch.testing.assert_close(result[2], observed[0][2], rtol=2e-6, atol=2e-7)
        # Checkpointing alone does not change this identical physical head layout.
        torch.testing.assert_close(observed[1][1], observed[2][1], rtol=0, atol=0)

    def test_full_forward_disabled_parity_projection_count_rng_and_response_reset(self):
        logits, teacher, labels, weights = mixed_logits()
        anchors = torch.arange(labels.shape[1]).reshape(1, -1)*16
        keep = torch.ones_like(anchors, dtype=torch.bool)
        results = []
        for enabled in (None, False, True):
            model = make_model(True)
            if enabled is not None:
                install_continuation_repair(model, enabled=enabled)
            current = logits.clone().requires_grad_()
            with mock.patch.object(model, '_forward_draft_blocks', return_value=(anchors, keep, current.reshape(1, -1, 31))), \
                 mock.patch.object(model, '_block_labels_and_weights', return_value=(labels, weights, labels)), \
                 mock.patch.object(model, '_aligned_target_hidden', return_value=teacher), \
                 mock.patch.object(model.lm_head, 'forward', wraps=model.lm_head.forward) as head:
                rng = torch.get_rng_state().clone()
                loss, _, metrics = model(labels.reshape(1, -1), torch.zeros(1), torch.ones_like(labels.reshape(1, -1)), teacher)
                calls = head.call_count
                loss.backward()
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            results.append((loss.detach(), current.grad, metrics, calls))
            if enabled:
                old_summary = model.continuation_repair.last_summary
                all_correct = logits.argmax(-1)
                reduced_native(model, logits.clone().requires_grad_(), teacher, all_correct, weights)
                self.assertIsNot(old_summary, model.continuation_repair.last_summary)
                self.assertEqual(float(model.continuation_repair.last_summary['continuation_high_count']), 0.)
        for actual, expected in zip(results[1][:2], results[0][:2]):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual([result[3] for result in results], [6, 6, 6])
        for key, value in results[0][2]['sum_metrics'].items():
            torch.testing.assert_close(results[2][2]['sum_metrics'][key], value, rtol=0, atol=0)

    def test_two_response_normalization_stays_separate(self):
        logits, teacher, labels, weights = mixed_logits()
        losses, numerators, denominators, gradients = [], [], [], []
        for multiplier in (.1, 2.):
            model = make_model(True)
            install_continuation_repair(model)
            current = logits.clone().requires_grad_()
            local = weights.clone()*multiplier
            if multiplier == 2.:
                local[:, 8:] *= .1
            loss, terms = reduced_native(model, current, teacher, labels, local)
            losses.append(loss)
            denominators.append(terms[1]+1e-6)
            numerators.append(loss*denominators[-1])
            gradients.append(torch.autograd.grad(loss, current, retain_graph=True)[0])
        averaged = sum(losses)/2
        pooled = sum(numerators)/sum(denominators)
        self.assertGreater(float((averaged-pooled).detach().abs()), 1e-5)
        for loss, gradient in zip(losses, gradients):
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(bool(torch.isfinite(gradient).all()))

    def test_actual_tiny_backbone_preserves_frozen_head_and_teacher(self):
        model = make_model(True, identity_head=False)
        model.block_size = 16
        install_continuation_repair(model)
        batch = osd_batch(model)
        batch['target_last_hidden_states'].requires_grad_(True)
        loss, _, metrics = model(**batch)
        loss.backward()
        self.assertGreater(float(metrics['sum_metrics']['repair_pair_repair_rows']), 0.)
        self.assertTrue(any(parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)
                            for parameter in model.draft_model.parameters()))
        self.assertIsNone(batch['target_last_hidden_states'].grad)
        for module in (model.lm_head, model.embed_tokens):
            self.assertTrue(all(not parameter.requires_grad and parameter.grad is None
                                for parameter in module.parameters()))

    def test_score_nonfinite_and_wrong_block_size_fail(self):
        labels = torch.zeros(1, 1, 16, dtype=torch.long)
        logits = torch.zeros(1, 1, 16, 3, dtype=torch.bfloat16)
        logits[..., 0] = 2
        logits[0, 0, 1] = torch.tensor([0., 2., 0.])
        logits[0, 0, 2, 0] = torch.inf
        weights = torch.ones_like(labels, dtype=torch.float32)
        weights[..., 0] = 0
        proposals = logits.argmax(-1)
        selection = first_error_mask(proposals, labels, weights)
        with self.assertRaisesRegex(ValueError, 'Nonfinite continuation margin'):
            continuation_scores(logits, proposals, labels, weights, selection)
        with self.assertRaisesRegex(ValueError, 'sixteen-slot'):
            continuation_scores(logits[..., :4, :], proposals[..., :4], labels[..., :4],
                                weights[..., :4], selection[..., :4])

    def test_collection_exception_discards_partial_rows_and_rejects_invalid_installer(self):
        extension = ContinuationRepair()
        with self.assertRaisesRegex(RuntimeError, 'fixture'):
            with extension.collect_response():
                extension._rows.append((torch.ones(1), torch.ones(1, dtype=torch.bool), torch.zeros(1)))
                raise RuntimeError('fixture')
        self.assertEqual(extension._rows, [])
        self.assertFalse(extension._collecting)
        with self.assertRaisesRegex(ValueError, 'boolean'):
            install_continuation_repair(make_model(True), enabled=1)
        with self.assertRaisesRegex(ValueError, r'C\+K\+PAIR'):
            install_continuation_repair(make_model(False))


if __name__ == '__main__':
    unittest.main()
