"""Focused CPU qualification for repair-only conditional KL, not pilot efficacy."""

import copy
import unittest
from unittest import mock

import torch
import torch.nn.functional as F

from online_feedback.first_error import first_error_mask, valid_prefix_mask
from online_feedback.kplus import install_kplus
from online_feedback.objective import TargetObjectiveModel
from online_feedback.prefix_allocation import install_prefix_allocation
from online_feedback.repair_conditional_kl import (
    SUMMARY_FIELDS, conditional_pair_values, install_repair_conditional_kl,
    repair_conditional_kl_contract, repair_pair_chunk_terms, repair_pair_reduce_response,
)
from specforge.core.chunking import checkpointed_chunk_reduce
from test_kplus_objective import explicit_weights, mixed_logits, ordinary_kl
from test_objective_integration import tiny_author
from test_osd_objective import osd_batch
from test_prefix_allocation import scalar_coefficients


def make_model(enabled=None, chunk_blocks=8, identity_head=True):
    model = TargetObjectiveModel(
        tiny_author(chunk_blocks=chunk_blocks), "osd_first_error", loss_precision="bfloat16")
    if identity_head:
        model.block_size = 16
        model.lm_head = torch.nn.Identity()
    install_kplus(model)
    install_prefix_allocation(model)
    if enabled is not None:
        install_repair_conditional_kl(model, enabled=enabled)
    model.begin_event()
    return model


def reduced_loss(model, logits, teacher, labels, weights, *, chunk_size=8, checkpoint=True):
    terms = checkpointed_chunk_reduce(
        model._chunk_loss, logits, None, labels, weights, teacher,
        chunk_size=chunk_size, dim=1, checkpoint=checkpoint)
    native, native_metrics = model.first_error_redistribution.reduce_response(terms[:11])
    base, allocation_metrics = model.prefix_allocation.reduce_response(native, terms[11:15])
    if getattr(model, "repair_conditional_kl", False):
        extra, pair_metrics = repair_pair_reduce_response(terms[15:], native_metrics)
    else:
        extra, pair_metrics = base.new_zeros(()), {}
    return (base + extra) / (terms[1] + 1e-6), terms, native_metrics, allocation_metrics, pair_metrics


def explicit_objective(logits, teacher, labels, weights):
    """Scalar role/allocation oracle with an independently computed FP32 pair KL."""
    proposals = logits.detach().argmax(-1)
    selected, final, errors = explicit_weights(proposals, labels, weights)
    valid = valid_prefix_mask(labels, weights)
    prefix = (selected > 0) & (proposals == labels)
    probability = F.log_softmax(logits.detach(), -1).gather(
        -1, labels.unsqueeze(-1)).squeeze(-1).float().exp()
    for block in range(labels.shape[1]):
        coefficients = scalar_coefficients(
            probability[0, block].tolist(), weights[0, block].tolist(),
            valid[0, block].tolist(), prefix[0, block].tolist())
        for offset in range(labels.shape[-1]):
            if prefix[0, block, offset]:
                final[0, block, offset] = coefficients[offset]
    base = (ordinary_kl(logits, teacher) * final).sum()
    auxiliary = logits.new_zeros((), dtype=torch.float32)
    for block, offset, _ in errors:
        y, r = labels[0, block, offset], proposals[0, block, offset]
        target = teacher[0, block, offset].detach().float()
        pair = torch.stack((y, r))
        p = target.softmax(-1)
        target_pair = p[pair] / p[pair].sum()
        draft_log_pair = logits[0, block, offset, pair].float().log_softmax(-1)
        conditional_kl = (target_pair * (target_pair.log() - draft_log_pair)).sum()
        auxiliary = auxiliary + final[0, block, offset] * (1 - p[pair].sum()) * conditional_kl
    denominator = selected.sum() + 1e-6
    return (base + auxiliary) / denominator, auxiliary, denominator, final


class RepairConditionalKLTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def test_fp64_decomposition_and_extra_gradient(self):
        target = torch.tensor([.15, -.4, 1.2, .8, -.9], dtype=torch.float64)
        draft = torch.tensor([-.2, .7, -.1, .3, 1.], dtype=torch.float64, requires_grad=True)
        pair = torch.tensor([2, 4])
        other = torch.tensor([0, 1, 3])
        p, q = target.softmax(-1), draft.softmax(-1)
        mu, nu = p[pair].sum(), q[pair].sum()
        pair_kl = (p[pair] / mu * ((p[pair] / mu).log() - (q[pair] / nu).log())).sum()
        other_kl = (p[other] / (1-mu) * (
            (p[other] / (1-mu)).log() - (q[other] / (1-nu)).log())).sum()
        binary = mu * (mu / nu).log() + (1-mu) * ((1-mu) / (1-nu)).log()
        full = (p * (p.log() - q.log())).sum()
        torch.testing.assert_close(full, binary + mu * pair_kl + (1-mu) * other_kl,
                                   rtol=1e-13, atol=1e-13)
        extra = (1-mu) * pair_kl
        extra.backward()
        delta = (1-mu) * (p[pair][0] / mu - q[pair][0] / nu)
        expected = torch.zeros_like(draft)
        expected[pair[0]], expected[pair[1]] = -delta, delta
        torch.testing.assert_close(draft.grad, expected, rtol=1e-13, atol=1e-13)
        current = draft.detach().float().requires_grad_()
        actual = conditional_pair_values(current[pair][None], target[pair][None], target[None])[0]
        actual.sum().backward()
        torch.testing.assert_close(actual[0].double(), extra.detach(), rtol=2e-6, atol=2e-7)
        torch.testing.assert_close(current.grad.double(), expected, rtol=2e-6, atol=2e-7)
        self.assertTrue(torch.equal(current.grad[other], torch.zeros(3)))

    def test_fp32_small_and_near_one_pair_mass(self):
        for target in ([-30., -31., 0.], [0., -1., -15.], [0., -1., -100.]):
            target = torch.tensor([target])
            draft = torch.tensor([[-.7, .3]], requires_grad=True)
            extra, mu, complement, _, _ = conditional_pair_values(draft, target[:, :2], target)
            p = target.double().softmax(-1)
            reference_mu = p[:, :2].sum(-1)
            conditional = p[:, :2] / reference_mu[:, None]
            expected = (1-reference_mu) * (
                conditional * (conditional.log() - draft.double().log_softmax(-1))).sum(-1)
            torch.testing.assert_close(mu.double(), reference_mu, rtol=2e-6, atol=2e-7)
            torch.testing.assert_close(extra.double(), expected, rtol=2e-6, atol=2e-7)
            self.assertTrue(bool((complement >= 0).all()))
            extra.sum().backward()
            self.assertTrue(bool(torch.isfinite(draft.grad).all()))

    def test_roundoff_clamp_count_and_invalid_mass(self):
        pair = torch.tensor([[0., 0.]])
        with mock.patch("online_feedback.repair_conditional_kl.torch.logsumexp",
                        side_effect=[torch.tensor([1.]), torch.tensor([1.-2e-7])]):
            extra, mu, complement, _, count = conditional_pair_values(pair, pair, pair)
        self.assertEqual(float(count), 1.)
        self.assertEqual(float(extra), 0.)
        self.assertEqual(float(mu), 1.)
        self.assertEqual(float(complement), 0.)
        with mock.patch("online_feedback.repair_conditional_kl.torch.logsumexp",
                        side_effect=[torch.tensor([1.]), torch.tensor([.99])]):
            with self.assertRaisesRegex(ValueError, "mass exceeds one"):
                conditional_pair_values(pair, pair, pair)

    def test_disabled_native_loss_gradients_and_no_error_neutrality(self):
        logits, teacher, labels, weights = mixed_logits()
        for no_error in (False, True):
            targets = logits.argmax(-1) if no_error else labels
            outputs = []
            for enabled in ((None, False, True) if no_error else (None, False)):
                current = logits.clone().requires_grad_()
                loss, terms, _, _, metrics = reduced_loss(
                    make_model(enabled), current, teacher, targets, weights)
                loss.backward()
                outputs.append((loss.detach(), current.grad, terms))
                if enabled:
                    self.assertEqual(float(metrics["repair_pair_repair_rows"]), 0.)
                    self.assertEqual(float(metrics["repair_pair_auxiliary_numerator"]), 0.)
            for result in outputs[1:]:
                for actual, expected in zip(result[:2], outputs[0][:2]):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                for actual, expected in zip(result[2][:15], outputs[0][2]):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_native_roles_eos_gap_padding_and_pair_gradient_support(self):
        labels = torch.tensor([[[0, 1, 2, 3, 4, 5]] * 6])
        proposals = labels.clone()
        weights = torch.ones_like(labels, dtype=torch.float32)
        weights[..., 0] = 0
        proposals[0, 0, 0] = 9       # Anchor never receives a direct term.
        proposals[0, 0, 2:] = 9     # First mismatch only; prefix remains base KL.
        weights[0, 1, 2] = 0
        proposals[0, 1, 3] = 9      # Cannot cross a padding/gap boundary.
        labels[0, 2, 2] = 10       # Valid EOS can be the repair token.
        proposals[0, 2, 2:] = 9
        labels[0, 3, 2] = 10
        proposals[0, 3, 2] = 10
        proposals[0, 3, 3:] = 9     # Post-EOS excluded.
        labels[0, 4, 0] = 10       # EOS anchor leaves no continuation.
        proposals[0, 4, 1:] = 9
        proposals[0, 5, 5] = 9     # Last valid position has no successor.
        logits = F.one_hot(proposals, num_classes=11).float().requires_grad_()
        teacher = torch.randn(logits.shape, generator=torch.Generator().manual_seed(18),
                              requires_grad=True)
        selection = first_error_mask(proposals, labels, weights, (10,))
        with mock.patch("online_feedback.repair_conditional_kl.conditional_pair_values",
                        wraps=conditional_pair_values) as values:
            n1, n2, summary = repair_pair_chunk_terms(
                logits, teacher, proposals, labels, weights, selection, (10,))
        self.assertEqual(tuple(values.call_args.args[2].shape), (3, 11))
        (n1+n2).backward()
        self.assertEqual(float(summary[0]), 3.)
        expected_support = torch.zeros_like(logits, dtype=torch.bool)
        for block, offset in ((0, 2), (2, 2), (5, 5)):
            expected_support[0, block, offset, labels[0, block, offset]] = True
            expected_support[0, block, offset, proposals[0, block, offset]] = True
        self.assertTrue(bool((logits.grad[~expected_support] == 0).all()))
        self.assertTrue(bool((logits.grad[expected_support] != 0).all()))
        self.assertIsNone(teacher.grad)

    def test_teacher_preference_ties_reversals_and_native_argmax_tie(self):
        logits = torch.tensor([[[[0., 0., 0.], [2., 2., 0.]]] * 3], requires_grad=True)
        labels = torch.tensor([[[0, 1]] * 3])
        weights = torch.tensor([[[0., 1.]] * 3])
        teacher = torch.tensor([[[[0., 0., 0.], [0., 1., 0.]],
                                 [[0., 0., 0.], [1., 1., 0.]],
                                 [[0., 0., 0.], [1., 0., 0.]]]])
        proposals = logits.detach().argmax(-1)
        self.assertEqual(proposals[0, :, 1].tolist(), [0, 0, 0])
        terms = repair_pair_chunk_terms(logits, teacher, proposals, labels, weights,
                                       first_error_mask(proposals, labels, weights))
        metrics = dict(zip(SUMMARY_FIELDS, terms[-1].tolist()))
        self.assertEqual(metrics["teacher_y_preferred_rows"], 1.)
        self.assertEqual(metrics["teacher_tie_rows"], 1.)
        self.assertEqual(metrics["teacher_r_preferred_rows"], 1.)

    def test_mixed_groups_chunks_and_two_response_weighted_reference(self):
        logits, teacher, labels, weights = mixed_logits()
        response_results = []
        for second in (False, True):
            local_weights = weights.clone()
            if second:
                local_weights[:, :8] *= .07
                local_weights[:, 8:] *= 1.7
            reference = logits.clone().requires_grad_()
            expected, expected_extra, denominator, _ = explicit_objective(
                reference, teacher, labels, local_weights)
            expected.backward()
            outputs = []
            for chunk_size, checkpoint in ((1, False), (8, False), (8, True), (64, True)):
                leaf = logits.clone().requires_grad_()
                model = make_model(True)
                loss, terms, native, allocation, pair = reduced_loss(
                    model, leaf, teacher, labels, local_weights,
                    chunk_size=chunk_size, checkpoint=checkpoint)
                before = {key: value.clone() for key, value in pair.items()}
                loss.backward()
                torch.testing.assert_close(loss, expected, rtol=2e-6, atol=3e-7)
                torch.testing.assert_close(pair["repair_pair_auxiliary_numerator"], expected_extra,
                                           rtol=2e-6, atol=3e-7)
                torch.testing.assert_close(terms[1]+1e-6, denominator, rtol=2e-6, atol=3e-7)
                # Native BF16 graph changes reduction order in the explicit oracle.
                torch.testing.assert_close(leaf.grad, reference.grad, rtol=.03, atol=2e-4)
                self.assertEqual(float(pair["repair_pair_repair_rows"]), 16.)
                self.assertEqual(float(native["kplus_mixed_response"]), 1.)
                for key in pair:
                    torch.testing.assert_close(pair[key], before[key], rtol=0, atol=0)
                outputs.append((loss.detach(), leaf.grad))
            torch.testing.assert_close(outputs[1][0], outputs[2][0], rtol=0, atol=0)
            torch.testing.assert_close(outputs[1][1], outputs[2][1], rtol=0, atol=0)
            response_results.append((outputs[2][0], expected.detach(), denominator))
        torch.testing.assert_close(sum(row[0] for row in response_results)/2,
                                   sum(row[1] for row in response_results)/2)
        pooled = sum(row[1]*row[2] for row in response_results) / sum(row[2] for row in response_results)
        self.assertGreater(float(abs(pooled-sum(row[0] for row in response_results)/2)), 1e-5)

    def test_base_terms_prefix_coefficients_fkl_metrics_and_projection_count_unchanged(self):
        logits, teacher, labels, weights = mixed_logits()
        blocks = labels.shape[1]
        anchors = torch.arange(blocks).reshape(1, -1)*16
        keep = torch.ones_like(anchors, dtype=torch.bool)
        outputs = []
        for enabled in (False, True):
            model = make_model(enabled)
            leaf = logits.clone().requires_grad_()
            with mock.patch.object(model, "_forward_draft_blocks",
                                   return_value=(anchors, keep, leaf.reshape(1, -1, 31))), \
                 mock.patch.object(model, "_block_labels_and_weights", return_value=(labels, weights, labels)), \
                 mock.patch.object(model, "_aligned_target_hidden", return_value=teacher), \
                 mock.patch.object(model.lm_head, "forward", wraps=model.lm_head.forward) as head:
                rng = torch.get_rng_state().clone()
                loss, _, metrics = model(labels.reshape(1, -1), torch.zeros(1),
                                         torch.ones_like(labels.reshape(1, -1)), teacher)
                forward_calls = head.call_count
                summary = copy.deepcopy(model.prefix_allocation.last_summary)
                loss.backward()
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                self.assertEqual(summary, model.prefix_allocation.last_summary)
            outputs.append((loss.detach(), metrics, summary, forward_calls))
        self.assertEqual(outputs[0][2:], outputs[1][2:])
        self.assertEqual(outputs[1][3], 6)
        base_metrics, pair_metrics = outputs[0][1], outputs[1][1]
        for key, value in base_metrics["sum_metrics"].items():
            torch.testing.assert_close(pair_metrics["sum_metrics"][key], value, rtol=0, atol=0)
        for actual, expected in zip(pair_metrics["ratio_metrics"]["forward_kl"],
                                    base_metrics["ratio_metrics"]["forward_kl"]):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        extra, denominator = pair_metrics["ratio_metrics"]["repair_pair_auxiliary"]
        torch.testing.assert_close(outputs[1][0], outputs[0][0] + extra/denominator)
        self.assertEqual(float(pair_metrics["sum_metrics"]["repair_pair_repair_rows"]), 16.)

    def test_actual_tiny_backbone_gradients_and_frozen_teacher_head_embedding(self):
        model = make_model(True, identity_head=False)
        batch = osd_batch(model)
        batch["target_last_hidden_states"].requires_grad_(True)
        loss, _, metrics = model(**batch)
        loss.backward()
        self.assertGreater(float(metrics["sum_metrics"]["repair_pair_repair_rows"]), 0.)
        self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum() > 0)
                            for p in model.draft_model.parameters()))
        self.assertIsNone(batch["target_last_hidden_states"].grad)
        for module in (model.lm_head, model.embed_tokens):
            self.assertTrue(all(not p.requires_grad and p.grad is None for p in module.parameters()))

    def test_incompatible_hooks_missing_teacher_and_nonfinite_fail_visibly(self):
        for name in ("point_loss_extension", "role_loss_extension", "successor_shuffle"):
            model = make_model(False)
            setattr(model, name, object())
            with self.assertRaisesRegex(ValueError, "other objective hooks"):
                install_repair_conditional_kl(model)
        model = make_model(True, identity_head=False)
        batch = osd_batch(model)
        batch.pop("target_last_hidden_states")
        with self.assertRaisesRegex(ValueError, "requires frozen target_last_hidden_states"):
            model(**batch)
        model.point_loss_extension = object()
        with self.assertRaisesRegex(ValueError, "other objective hooks"):
            model(**batch)
        for which in (0, 1, 2):
            args = [torch.zeros(1, 2), torch.zeros(1, 2), torch.zeros(1, 3)]
            args[which][0, 0] = float("nan")
            with self.assertRaisesRegex(ValueError, "nonfinite selected logits"):
                conditional_pair_values(*args)
        with self.assertRaisesRegex(ValueError, "nonfinite auxiliary"):
            conditional_pair_values(torch.tensor([[3e38, -3e38]]),
                                    torch.zeros(1, 2), torch.zeros(1, 3))
        with self.assertRaisesRegex(ValueError, "boolean"):
            repair_conditional_kl_contract(1)


if __name__ == "__main__":
    unittest.main()
