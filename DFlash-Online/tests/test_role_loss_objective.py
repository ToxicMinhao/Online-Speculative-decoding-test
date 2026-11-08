"""CPU qualification of the fixed K+ role-loss contract; no Adam or generation.

The expression oracle independently traverses blocks and repeats the specified
row operations. It imports no selection, redistribution or loss helper. Scalar
2e-6/2e-6 and exact BF16 gradients/caps retain the PR31 numerical policy.
"""

import unittest
from unittest import mock

import torch
import torch.nn.functional as F

from online_feedback import role_loss
from online_feedback.kplus import install_kplus
from online_feedback.objective import TargetObjectiveModel
from online_feedback.role_loss import RoleLossReducer, install_role_loss, role_loss_contract
from specforge.core.chunking import checkpointed_chunk_reduce
from test_kplus_objective import mixed_logits
from test_objective_integration import tiny_author


SCALAR_TOLERANCE = dict(rtol=2e-6, atol=2e-6)


def manual_roles(logits, labels, weights, eos=()):
    """Independent scalar traversal; successor affects priority, not selection."""
    proposals = logits.detach().argmax(-1)
    prefix = torch.zeros_like(labels, dtype=torch.bool)
    low = torch.zeros_like(prefix)
    high = torch.zeros_like(prefix)
    for batch in range(labels.shape[0]):
        for block in range(labels.shape[1]):
            if int(labels[batch, block, 0]) in eos:
                continue
            for offset in range(1, labels.shape[2]):
                if float(weights[batch, block, offset]) <= 0:
                    break
                terminal = int(labels[batch, block, offset]) in eos
                if int(proposals[batch, block, offset]) != int(labels[batch, block, offset]):
                    next_offset = offset + 1
                    priority2 = (
                        not terminal and next_offset < labels.shape[2]
                        and float(weights[batch, block, next_offset]) > 0
                        and int(proposals[batch, block, next_offset]) ==
                        int(labels[batch, block, next_offset])
                    )
                    (high if priority2 else low)[batch, block, offset] = True
                    break
                prefix[batch, block, offset] = True
                if terminal:
                    break
    return prefix, low, high


def expression_oracle(logits, teacher, labels, weights, arm, chunk_size=8, eos=()):
    """Independent response expression, including the same native chunk sums."""
    full_d = weights.reshape(-1).sum() + 1e-6
    totals = None
    caps = []
    for start in range(0, labels.shape[1], chunk_size):
        sl = slice(start, start + chunk_size)
        student = logits[:, sl]
        target = teacher[:, sl].detach()
        canonical = labels[:, sl]
        native_weight = weights[:, sl]
        prefix, low, high = manual_roles(student, canonical, native_weight, eos)
        errors = low | high
        selected = prefix | errors
        mixture = selected if arm == 'MIX' else errors
        selected_weights = native_weight * selected
        fkl = F.kl_div(F.log_softmax(student, -1), F.softmax(target, -1),
                       reduction='none').sum(-1)
        native = (fkl * selected_weights).sum()
        fprefix = native if not bool(errors.any()) else (fkl * (native_weight * prefix)).sum()
        zero = native_weight.new_zeros(())
        ce = rkl = ce1 = ce2 = rkl1 = rkl2 = zero
        if bool(mixture.any()):
            s = student[mixture]
            t = target[mixture]
            target_tokens = torch.argmax(t, dim=-1)
            ce_rows = F.cross_entropy(s, target_tokens, reduction='none')
            q = F.softmax(s / 1.0, dim=-1)
            log_p = F.log_softmax(t.float() / 1.0, dim=-1).to(s.dtype)
            entries = F.kl_div(log_p.detach(), q, reduction='none')
            r = entries.sum(dim=-1) * (1.0 * 1.0)
            r /= full_d
            caps.append((r > 0.01).detach())
            r = torch.clamp(r, max=0.01)
            w = native_weight[mixture]
            low_w, high_w = w * low[mixture], w * high[mixture]
            ce, rkl = (ce_rows * w).sum(), (r * w).sum()
            ce1, ce2 = (ce_rows * low_w).sum(), (ce_rows * high_w).sum()
            rkl1, rkl2 = (r * low_w).sum(), (r * high_w).sum()
        current = (native, selected_weights.sum(), fprefix, ce, rkl, ce1, ce2, rkl1, rkl2,
                   (native_weight * low).sum(), (native_weight * high).sum(),
                   errors.sum().float())
        totals = current if totals is None else tuple(a + b for a, b in zip(totals, current))
    native, mass, prefix_n, ce, rkl, ce1, ce2, rkl1, rkl2, w1, w2, count = totals
    z = mass + 1e-6
    if not bool(mass > 0) or (arm == 'ROLE' and not bool(count > 0)):
        return native / z.clamp_min(torch.finfo(torch.float32).tiny), caps
    if bool((w1 > 0) & (w2 > 0)):
        v = w1 + 2 * w2
        ce = ce + 0.5 * (w1 * ce2 - w2 * ce1) / v
        rkl = rkl + 0.5 * (w1 * rkl2 - w2 * rkl1) / v
    if arm == 'MIX':
        return 0.8 * ce / z + 0.2 * (full_d / z) * rkl, caps
    return prefix_n / z + 0.8 * ce / z + 0.2 * (full_d / z) * rkl, caps


class RoleLossObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized(), 'CPU fixture initialized CUDA')

    def model(self, arm, chunk_size=8, eos=()):
        model = TargetObjectiveModel(tiny_author(chunk_blocks=chunk_size), 'osd_first_error',
                                     loss_precision='bfloat16', eos_token_ids=eos)
        model.block_size = 16
        model.lm_head = torch.nn.Identity()
        if arm == 'KPLUS':
            install_kplus(model)
        elif arm is not None:
            self.assertEqual(install_role_loss(model, arm), role_loss_contract(arm))
        model.begin_event()
        return model

    def run_chunks(self, model, logits, teacher, labels, weights, chunk_size=8, checkpoint=True):
        extension = getattr(model, 'role_loss_extension', None)
        if extension is not None:
            extension.prepare_response(weights)
        terms = checkpointed_chunk_reduce(
            model._chunk_loss, logits, None, labels, weights, teacher,
            chunk_size=chunk_size, dim=1, checkpoint=checkpoint,
        )
        if extension is not None:
            loss = extension.reduce_response(terms, terms[1] + 1e-6)
        else:
            numerator = terms[0]
            if getattr(model, 'first_error_redistribution', None) is not None:
                numerator, _ = model.first_error_redistribution.reduce_response(terms)
            loss = numerator / (terms[1] + 1e-6).clamp_min(torch.finfo(torch.float32).tiny)
        return loss, terms

    def test_independent_expression_and_exact_bf16_gradients_for_both_arms(self):
        logits, teacher, labels, weights = mixed_logits()
        teacher.requires_grad_()
        for arm in ('MIX', 'ROLE'):
            for checkpoint in (False, True):
                with self.subTest(arm=arm, checkpoint=checkpoint):
                    actual_logits = logits.clone().requires_grad_()
                    expected_logits = logits.clone().requires_grad_()
                    expected, expected_caps = expression_oracle(
                        expected_logits, teacher, labels, weights, arm)
                    expected.backward()
                    model = self.model(arm)
                    actual, terms = self.run_chunks(
                        model, actual_logits, teacher, labels, weights, checkpoint=checkpoint)
                    actual.backward()
                    torch.testing.assert_close(actual, expected, **SCALAR_TOLERANCE)
                    torch.testing.assert_close(actual_logits.grad, expected_logits.grad, rtol=0, atol=0)
                    self.assertIsNone(teacher.grad)
                    prefix, low, high = manual_roles(logits, labels, weights)
                    selected = prefix | low | high
                    self.assertTrue(torch.equal(actual_logits.grad[~selected],
                                                torch.zeros_like(actual_logits.grad[~selected])))
                    self.assertEqual(int(terms[18][6] + terms[18][7]),
                                     sum(int(cap.sum()) for cap in expected_caps))
                    self.assertEqual(terms[12].dtype, torch.float32)
                    self.assertEqual(terms[13].dtype, torch.float32)

    def test_response_wide_priority_classes_span_chunks_and_conserve_mass(self):
        logits, teacher, labels, weights = mixed_logits()
        for arm in ('MIX', 'ROLE'):
            model = self.model(arm)
            loss, terms = self.run_chunks(model, logits.requires_grad_(), teacher, labels, weights)
            metrics, ratios = model.role_loss_extension.response_metrics(terms)
            self.assertEqual(float(metrics['kplus_priority1_error_count']), 8.)
            self.assertEqual(float(metrics['kplus_priority2_error_count']), 8.)
            self.assertEqual(float(metrics['kplus_mixed_response']), 1.)
            torch.testing.assert_close(metrics['kplus_first_error_weight_after'],
                                       terms[5] + terms[6], **SCALAR_TOLERANCE)
            self.assertLess(float(metrics['kplus_coefficient1']), 1.)
            self.assertGreater(float(metrics['kplus_coefficient2']), 1.)
            # Incorrect chunk-local redistribution sees only one priority class
            # in each error-bearing chunk, and therefore returns another loss.
            local = []
            model.role_loss_extension.prepare_response(weights)
            for start in range(0, 17, 8):
                sl = slice(start, start + 8)
                t = model._chunk_loss(logits[:, sl], None, labels[:, sl], weights[:, sl], teacher[:, sl])
                local.append(model.role_loss_extension._mixture_numerators(t))
            pooled_ce, pooled_rkl = model.role_loss_extension._mixture_numerators(terms)
            self.assertNotEqual(float(pooled_ce), float(sum(row[0] for row in local)))
            self.assertTrue(all(not value.requires_grad for value in metrics.values()))
            self.assertTrue(all(not value.requires_grad for pair in ratios.values() for value in pair))
            self.assertTrue(torch.isfinite(loss) and torch.isfinite(pooled_rkl))

    def test_no_error_role_is_exact_original_kplus_and_never_evaluates_rkl(self):
        logits, teacher, _, weights = mixed_logits()
        labels = logits.argmax(-1)
        observations = []
        for arm in ('KPLUS', 'ROLE'):
            model = self.model(arm)
            leaf = logits.clone().requires_grad_()
            with mock.patch.object(role_loss, 'repo_position_terms',
                                   side_effect=AssertionError('empty RKL group evaluated')):
                loss, terms = self.run_chunks(model, leaf, teacher, labels, weights)
                loss.backward()
            observations.append((loss.detach(), leaf.grad))
            if arm == 'ROLE':
                self.assertEqual(float(terms[9] + terms[10]), 0.)
                self.assertEqual(float(terms[18][9]), 0.)
                self.assertEqual(float(terms[18][4] + terms[18][5]), 0.)
        for native, actual in zip(*observations):
            torch.testing.assert_close(native, actual, rtol=0, atol=0)

    def test_empty_response_keeps_zero_backward_without_evaluating_mixture(self):
        logits, teacher, labels, weights = mixed_logits()
        weights.zero_()
        for arm in ('MIX', 'ROLE'):
            leaf = logits.clone().requires_grad_()
            model = self.model(arm)
            with mock.patch.object(role_loss, 'repo_position_terms',
                                   side_effect=AssertionError('empty RKL group evaluated')):
                loss, _ = self.run_chunks(model, leaf, teacher, labels, weights)
                loss.backward()
            self.assertEqual(float(loss.detach()), 0.)
            self.assertTrue(torch.equal(leaf.grad, torch.zeros_like(leaf.grad)))

    def test_disabled_extension_has_exact_native_kplus_behavior(self):
        logits, teacher, labels, weights = mixed_logits()
        observations = []
        for explicit_none in (False, True):
            model = self.model('KPLUS')
            if explicit_none:
                model.role_loss_extension = None
            leaf = logits.clone().requires_grad_()
            loss, _ = self.run_chunks(model, leaf, teacher, labels, weights)
            loss.backward()
            observations.append((loss.detach(), leaf.grad))
        for a, b in zip(*observations):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_no_error_mix_still_uses_teacher_mixture(self):
        logits, teacher, _, weights = mixed_logits()
        labels = logits.argmax(-1)
        model = self.model('MIX')
        loss, terms = self.run_chunks(model, logits.requires_grad_(), teacher, labels, weights)
        expected, _ = expression_oracle(logits, teacher, labels, weights, 'MIX')
        torch.testing.assert_close(loss, expected, **SCALAR_TOLERANCE)
        self.assertGreater(float(terms[18][4]), 0.)
        self.assertEqual(float(terms[18][5]), 0.)

    def test_one_priority_class_keeps_native_component_numerators(self):
        logits, teacher, _, weights = mixed_logits()
        for priority in (1, 2):
            labels = logits.argmax(-1)
            labels[..., 1] = (labels[..., 1] + 1) % 31
            if priority == 1:
                labels[..., 2] = (labels[..., 2] + 1) % 31
            for arm in ('MIX', 'ROLE'):
                with self.subTest(priority=priority, arm=arm):
                    model = self.model(arm)
                    leaf = logits.clone().requires_grad_()
                    loss, terms = self.run_chunks(model, leaf, teacher, labels, weights)
                    ce, rkl = model.role_loss_extension._mixture_numerators(terms)
                    self.assertIs(ce, terms[12])
                    self.assertIs(rkl, terms[13])
                    reference = logits.clone().requires_grad_()
                    expected, _ = expression_oracle(reference, teacher, labels, weights, arm)
                    loss.backward()
                    expected.backward()
                    torch.testing.assert_close(loss, expected, **SCALAR_TOLERANCE)
                    torch.testing.assert_close(leaf.grad, reference.grad, rtol=0, atol=0)

    def test_eos_validity_gaps_first_and_last_errors_match_independent_roles(self):
        logits, teacher, labels, weights = mixed_logits()
        labels = logits.argmax(-1)
        labels[0, 0, 1] = (labels[0, 0, 1] + 1) % 31
        labels[0, 1, -1] = (labels[0, 1, -1] + 1) % 31
        weights[0, 2, 2] = 0
        labels[0, 2, 3] = (labels[0, 2, 3] + 1) % 31
        labels[0, 3, 2] = 30  # EOS mismatch cannot get successor priority.
        logits[0, 3, 2, 30] = -10
        labels[0, 4, 0] = 30  # EOS seed suppresses the block.
        for arm in ('MIX', 'ROLE'):
            model = self.model(arm, eos=(30,))
            leaf = logits.clone().requires_grad_()
            loss, terms = self.run_chunks(model, leaf, teacher, labels, weights)
            expected_leaf = logits.clone().requires_grad_()
            expected, _ = expression_oracle(expected_leaf, teacher, labels, weights, arm, eos=(30,))
            loss.backward()
            expected.backward()
            torch.testing.assert_close(loss, expected, **SCALAR_TOLERANCE)
            torch.testing.assert_close(leaf.grad, expected_leaf.grad, rtol=0, atol=0)
            prefix, low, high = manual_roles(logits, labels, weights, (30,))
            torch.testing.assert_close(terms[1], (weights * (prefix | low | high)).sum(),
                                       **SCALAR_TOLERANCE)
            self.assertTrue(bool(low[0, 3, 2]))
            self.assertFalse(bool((prefix | low | high)[0, 4].any()))

    def test_full_native_mass_counts_occurrences_and_retains_positions_after_selection(self):
        logits, teacher, labels, weights = mixed_logits()
        model = self.model('ROLE')
        _, terms = self.run_chunks(model, logits.requires_grad_(), teacher, labels, weights)
        d = model.role_loss_extension.denominator
        z = terms[1] + 1e-6
        torch.testing.assert_close(d, weights.reshape(-1).sum() + 1e-6, rtol=0, atol=0)
        self.assertGreater(float(d), float(z))
        metrics, _ = model.role_loss_extension.response_metrics(terms)
        torch.testing.assert_close(metrics['role_loss_full_to_selected_ratio'], d / z, rtol=0, atol=0)
        # Duplicate sampled occurrences contribute again; they are never deduplicated.
        repeated = weights.repeat(1, 2, 1)
        doubled = RoleLossReducer('ROLE').prepare_response(repeated)
        torch.testing.assert_close(doubled, 2 * (d - 1e-6) + 1e-6, **SCALAR_TOLERANCE)

    def test_cap_uses_full_D_not_selected_Z_and_native_row_dtype(self):
        logits = torch.zeros(1, 2, 16, 5, dtype=torch.bfloat16)
        teacher = torch.zeros_like(logits)
        logits[..., 0] = 1.0
        teacher[..., 1] = 1.0
        labels = torch.zeros(1, 2, 16, dtype=torch.long)
        labels[..., 1] = 1
        weights = torch.ones_like(labels, dtype=torch.float32)
        weights[..., 0] = 0
        model = self.model('ROLE')
        captured = []
        native_rows = role_loss.repo_position_terms

        def capture(*args):
            rows = native_rows(*args)
            captured.append(rows)
            return rows

        with mock.patch.object(role_loss, 'repo_position_terms', side_effect=capture):
            _, terms = self.run_chunks(model, logits.requires_grad_(), teacher, labels, weights,
                                       checkpoint=False)
        rows = captured[0]
        self.assertEqual(rows['normalized_rkl'].dtype, torch.bfloat16)
        self.assertEqual(rows['ce'].dtype, torch.bfloat16)
        self.assertEqual(rows['teacher_log_probabilities'].dtype, torch.bfloat16)
        z = terms[1] + 1e-6
        wrong = native_rows(logits[..., 1, :], teacher[..., 1, :], z)
        self.assertFalse(bool((rows['normalized_rkl'] > .01).any()))
        self.assertTrue(bool((wrong['normalized_rkl'] > .01).all()))

    def test_cap_both_sides_and_clipped_rkl_gradient(self):
        logits, teacher, labels, weights = mixed_logits()
        weights *= .02
        for arm in ('MIX', 'ROLE'):
            model = self.model(arm)
            leaf = logits.clone().requires_grad_()
            loss, terms = self.run_chunks(model, leaf, teacher, labels, weights)
            ref = logits.clone().requires_grad_()
            expected, caps = expression_oracle(ref, teacher, labels, weights, arm)
            loss.backward()
            expected.backward()
            self.assertGreater(sum(int(cap.sum()) for cap in caps), 0)
            torch.testing.assert_close(loss, expected, **SCALAR_TOLERANCE)
            torch.testing.assert_close(leaf.grad, ref.grad, rtol=0, atol=0)
            self.assertEqual(float(terms[18][6] + terms[18][7]),
                             sum(int(cap.sum()) for cap in caps))

    def test_teacher_argmax_is_ce_target_and_mismatches_are_logged_by_role(self):
        logits, teacher, labels, weights = mixed_logits()
        teacher.zero_()
        teacher[..., 30] = 5
        model = self.model('MIX')
        loss, terms = self.run_chunks(model, logits.requires_grad_(), teacher, labels, weights)
        prefix, low, high = manual_roles(logits, labels, weights)
        mismatch = labels != 30
        self.assertEqual(float(terms[18][11]), float((mismatch & (prefix | low | high)).sum()))
        self.assertEqual(float(terms[18][12]), float((mismatch & prefix).sum()))
        self.assertEqual(float(terms[18][13]), float((mismatch & (low | high)).sum()))
        expected, _ = expression_oracle(logits, teacher, labels, weights, 'MIX')
        torch.testing.assert_close(loss, expected, **SCALAR_TOLERANCE)

    def test_native_zero_probability_backward_failure_is_preserved(self):
        logits, teacher, labels, weights = mixed_logits()
        logits[0, 0, 1, 0] = -1000
        labels[0, 0, 1] = (logits[0, 0, 1].argmax() + 1) % 31
        model = self.model('ROLE')
        leaf = logits.clone().requires_grad_()
        loss, terms = self.run_chunks(model, leaf, teacher, labels, weights)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(terms[18][8]), 0.)
        loss.backward()
        self.assertFalse(bool(torch.isfinite(leaf.grad).all()))

    def test_excluded_underflow_does_not_enter_nominal_zero_rkl_branch(self):
        logits, teacher, labels, weights = mixed_logits()
        for block in range(17):
            logits[0, block, -1, 0] = -1000
            labels[0, block, 1] = (logits[0, block, 1].argmax() + 1) % 31
        for arm in ('MIX', 'ROLE'):
            model = self.model(arm)
            leaf = logits.clone().requires_grad_()
            loss, terms = self.run_chunks(model, leaf, teacher, labels, weights)
            loss.backward()
            self.assertTrue(bool(torch.isfinite(leaf.grad).all()))
            self.assertEqual(float(terms[18][8]), 0.)
            self.assertTrue(torch.equal(leaf.grad[..., -1, :], torch.zeros_like(leaf.grad[..., -1, :])))

    def test_nonfinite_logits_remain_failure_not_silently_stabilized(self):
        logits, teacher, labels, weights = mixed_logits()
        logits[0, 0, 1, :] = float('nan')
        labels[0, 0, 1] = 1
        model = self.model('ROLE')
        loss, terms = self.run_chunks(model, logits.requires_grad_(), teacher, labels, weights)
        self.assertFalse(bool(torch.isfinite(loss)))
        self.assertGreater(float(terms[18][10]), 0.)

    def test_full_forward_one_backbone_and_only_shared_vocabulary_projections(self):
        logits, teacher, labels, weights = mixed_logits()
        for arm in ('MIX', 'ROLE'):
            model = self.model(arm)
            leaf = logits.clone().requires_grad_()
            anchors = torch.arange(17).reshape(1, -1) * 16
            keep = torch.ones_like(anchors, dtype=torch.bool)
            input_ids = labels.reshape(1, -1)
            with mock.patch.object(model, '_forward_draft_blocks',
                                   return_value=(anchors, keep, leaf.reshape(1, -1, 31))) as backbone:
                with mock.patch.object(model, '_block_labels_and_weights',
                                       return_value=(labels, weights, labels)):
                    with mock.patch.object(model, '_aligned_target_hidden', return_value=teacher):
                        with mock.patch.object(model.lm_head, 'forward', wraps=model.lm_head.forward) as head:
                            rng = torch.get_rng_state().clone()
                            loss, _, metrics = model(input_ids, torch.zeros(1),
                                                     torch.ones_like(input_ids), teacher)
                            self.assertEqual(head.call_count, 6)
                            saved = {k: v.clone() for k, v in metrics['sum_metrics'].items()}
                            loss.backward()
                            self.assertEqual(head.call_count, 12)
            self.assertEqual(backbone.call_count, 1)
            self.assertTrue(torch.equal(torch.get_rng_state(), rng))
            self.assertEqual(float(metrics['loss_terms'][1]), 1.)
            self.assertEqual(float(metrics['sum_metrics']['role_loss_response_count']), 1.)
            self.assertEqual(float(metrics['sum_metrics']['kplus_response_count']), 1.)
            self.assertIn('native_selected_forward_kl', metrics['ratio_metrics'])
            for key, value in saved.items():
                torch.testing.assert_close(metrics['sum_metrics'][key], value, rtol=0, atol=0)

    def test_saved_D_changes_only_after_previous_backward_and_recomputes_exactly(self):
        logits, teacher, labels, weights = mixed_logits()
        model = self.model('ROLE')
        for factor in (1., .03):
            leaf = logits.clone().requires_grad_()
            response_weights = weights * factor
            loss, _ = self.run_chunks(model, leaf, teacher, labels, response_weights)
            saved_d = model.role_loss_extension.denominator.clone()
            loss.backward()
            torch.testing.assert_close(model.role_loss_extension.denominator, saved_d, rtol=0, atol=0)
            reference = logits.clone().requires_grad_()
            expected, _ = expression_oracle(reference, teacher, labels, response_weights, 'ROLE')
            expected.backward()
            torch.testing.assert_close(leaf.grad, reference.grad, rtol=0, atol=0)

    def test_installer_isolated_and_rejects_non_native_configuration(self):
        for objective, precision in (('osd', 'bfloat16'), ('osd_first_error', 'float32')):
            model = TargetObjectiveModel(tiny_author(), objective, loss_precision=precision)
            with self.assertRaisesRegex(ValueError, 'native BF16'):
                install_role_loss(model, 'ROLE')
        for name in ('prefix_guard', 'first_error_redistribution', 'point_loss_extension',
                     'role_loss_extension'):
            model = self.model(None)
            setattr(model, name, object())
            with self.assertRaisesRegex(ValueError, 'cannot combine'):
                install_role_loss(model, 'ROLE')
        model = self.model(None)
        parameters = tuple(model.parameters())
        install_role_loss(model, 'ROLE')
        self.assertEqual(tuple(model.parameters()), parameters)
        with self.assertRaisesRegex(ValueError, 'one response'):
            model.role_loss_extension.validate_response(torch.zeros(2, 16))
        with self.assertRaisesRegex(ValueError, 'MIX or ROLE'):
            role_loss_contract('KPLUS')

    def test_full_forward_rejects_later_extension_stacking(self):
        model = self.model('ROLE')
        install_kplus(model)  # The historical installer itself is unchanged.
        with self.assertRaisesRegex(ValueError, 'cannot combine'):
            model(torch.zeros(1, 16, dtype=torch.long), torch.zeros(1), torch.ones(1, 16))

    def test_prepare_rejects_bad_weights_and_unprepared_loss(self):
        reducer = RoleLossReducer('ROLE')
        for weights in (torch.ones(1, 2, 16, dtype=torch.bfloat16),
                        torch.ones(1, 2, 16, requires_grad=True)):
            with self.assertRaisesRegex(ValueError, 'detached native FP32'):
                reducer.prepare_response(weights)
        for value in (-1., float('nan'), float('inf')):
            with self.assertRaisesRegex(ValueError, 'finite nonnegative'):
                reducer.prepare_response(torch.full((1, 2, 16), value))
        model = self.model('ROLE')
        logits, teacher, labels, weights = mixed_logits()
        with self.assertRaisesRegex(RuntimeError, 'Prepare the full response'):
            model._chunk_loss(logits, None, labels, weights, teacher)


if __name__ == '__main__':
    unittest.main()
