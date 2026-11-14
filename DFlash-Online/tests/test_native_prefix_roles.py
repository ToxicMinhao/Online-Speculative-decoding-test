"""CPU controls for the disposable K+ role-gradient experiment."""

import copy
import json
import unittest
from unittest import mock

import torch

from online_feedback.first_error import first_error_mask
from online_feedback.kplus import group_terms, install_kplus, redistribute_response
from online_feedback.native_prefix_roles import (
    NativePrefixRoles, controlled_rows, install_native_prefix_roles, role_masks,
)
from online_feedback.objective import TargetObjectiveModel
from specforge.core.chunking import checkpointed_chunk_reduce
from test_kplus_objective import explicit_weights, mixed_logits, ordinary_kl
from test_objective_integration import tiny_author, response_batch


class NativePrefixRoleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def test_native_and_all_return_original_tensor(self):
        values = torch.tensor([1., 2.], requires_grad=True)
        prefix = torch.tensor([True, False])
        for role in ('N', 'ALL'):
            self.assertIs(controlled_rows(values, prefix, ~prefix, role), values)

    def test_first_slot_last_slot_all_correct_eos_gap_and_seed(self):
        labels = torch.tensor([[[0, 1, 2, 3, 4]] * 7])
        proposals = labels.clone()
        weights = torch.ones_like(labels, dtype=torch.float64)
        weights[..., 0] = 0
        proposals[0, 0, 1] = 9
        proposals[0, 1, 4] = 9
        labels[0, 3, 2] = 8
        proposals[0, 3, 2] = 8  # Correct EOS, later tokens cannot become roles.
        weights[0, 4, 2] = 0  # Gap cannot be bridged.
        labels[0, 5, 0] = 8  # EOS seed leaves no proposals.
        labels[0, 6, 1] = 8  # First proposal is wrong at EOS.
        selection = first_error_mask(proposals, labels, weights, (8,))
        prefix, errors = role_masks(proposals, labels, weights, selection)
        self.assertEqual(prefix.sum(-1).tolist(), [[0, 3, 4, 2, 1, 0, 0]])
        self.assertEqual(errors.sum(-1).tolist(), [[1, 1, 0, 0, 0, 0, 1]])
        self.assertFalse(bool(prefix[..., 0].any() | errors[..., 0].any()))
        self.assertTrue(torch.equal(prefix | errors, selection))
        self.assertFalse(prefix.requires_grad or errors.requires_grad)

    def test_complete_redistribution_and_unchanged_denominator_against_row_oracle(self):
        for pattern in ('mixed', 'priority1', 'priority2', 'correct'):
            logits, _, labels, weights = mixed_logits(torch.float64)
            proposals = logits.argmax(-1)
            if pattern == 'correct':
                labels = proposals.clone()
            elif pattern in ('priority1', 'priority2'):
                for block in range(16):
                    offset = 1 + block % 5
                    labels[0, block, offset + 1] = proposals[0, block, offset + 1]
                    if pattern == 'priority1':
                        labels[0, block, offset + 1] = (labels[0, block, offset + 1] + 1) % 31
            selected, modified, _ = explicit_weights(proposals, labels, weights.double())
            selection = selected > 0
            prefix, errors = role_masks(proposals, labels, weights, selection)
            for role in ('N', 'ALL', 'C', 'E'):
                with self.subTest(pattern=pattern, role=role):
                    rows = torch.linspace(.01, 3, labels.numel(), dtype=torch.float64).reshape_as(weights)
                    rows = rows.requires_grad_()
                    controlled = controlled_rows(rows, prefix, errors, role)
                    terms = group_terms(controlled, proposals, labels, weights.double(), selection)
                    actual_numerator, metrics = redistribute_response(
                        (controlled * selected).sum(), selected.sum(), terms)
                    loss = actual_numerator / (selected.sum() + 1e-6)
                    loss.backward()
                    active = prefix | errors if role in ('N', 'ALL') else (prefix if role == 'C' else errors)
                    expected_gradient = modified * active / (selected.sum() + 1e-6)
                    torch.testing.assert_close(rows.grad, expected_gradient, rtol=1e-12, atol=1e-12)
                    torch.testing.assert_close(loss, (rows.detach() * modified).sum() / (selected.sum()+1e-6),
                                               rtol=1e-12, atol=1e-12)
                    self.assertEqual(float(metrics['kplus_S']), float(selected.sum()))
                    self.assertTrue(torch.equal(rows.grad[~active], torch.zeros_like(rows.grad[~active])))

    def model(self):
        model = TargetObjectiveModel(tiny_author(chunk_blocks=8), 'osd_first_error', loss_precision='bfloat16')
        model.block_size = 16
        model.lm_head = torch.nn.Identity()
        install_kplus(model)
        model.begin_event()
        return model

    def evaluate_chunks(self, model, logits, teacher, labels, weights, controller=None):
        if controller is not None:
            controller.begin_response()
        terms = checkpointed_chunk_reduce(
            model._chunk_loss, logits, None, labels, weights, teacher,
            chunk_size=8, dim=1, checkpoint=True,
        )
        numerator, metrics = model.first_error_redistribution.reduce_response(terms)
        denominator = terms[1] + 1e-6
        loss = numerator / denominator
        if controller is not None:
            controller.finish_response(loss, denominator, metrics)
        return loss, metrics

    def test_bf16_native_all_enabled_loss_gradients_and_chunk_receipts_exact(self):
        logits, teacher, labels, weights = mixed_logits()
        native_model = self.model()
        native = logits.clone().requires_grad_()
        native_loss, native_metrics = self.evaluate_chunks(native_model, native, teacher, labels, weights)
        native_loss.backward()
        for role in ('N', 'ALL'):
            model = self.model()
            controller = install_native_prefix_roles(model, role, capture_gradients=True)
            current = logits.clone().requires_grad_()
            loss, metrics = self.evaluate_chunks(model, current, teacher, labels, weights, controller)
            loss.backward()
            self.assertTrue(torch.equal(loss, native_loss))
            self.assertTrue(torch.equal(current.grad, native.grad))
            for name in metrics:
                self.assertTrue(torch.equal(metrics[name], native_metrics[name]), name)
            receipt = controller.receipt()
            self.assertEqual(len(receipt['responses']), 1)
            self.assertEqual(receipt['unique_projection_chunks'], 3)
            self.assertTrue(all(chunk['gradient']['finite'] for chunk in receipt['responses'][0]['chunks']))
            self.assertTrue(all(chunk['gradient']['excluded_nonzero'] == 0
                                for chunk in receipt['responses'][0]['chunks']))
            self.assertEqual(receipt['responses'][0]['native_full_objective'],
                             receipt['responses'][0]['gradient_enabled_objective'])

    def test_bf16_role_controls_have_no_excluded_row_or_logit_gradients(self):
        logits, teacher, labels, weights = mixed_logits()
        baseline, _ = self.evaluate_chunks(self.model(), logits, teacher, labels, weights)
        selected = first_error_mask(logits.argmax(-1), labels, weights)
        prefix, errors = role_masks(logits.argmax(-1), labels, weights, selected)
        for role in ('E', 'C'):
            model = self.model()
            controller = install_native_prefix_roles(model, role, capture_gradients=True)
            current = logits.clone().requires_grad_()
            loss, _ = self.evaluate_chunks(model, current, teacher, labels, weights, controller)
            loss.backward()
            active = errors if role == 'E' else prefix
            self.assertTrue(torch.equal(loss, baseline))
            self.assertTrue(torch.equal(current.grad[~active], torch.zeros_like(current.grad[~active])))
            self.assertGreater(int(torch.count_nonzero(current.grad[active])), 0)
            response = controller.receipt()['responses'][0]
            self.assertGreater(response['gradient_enabled_objective'], 0)
            self.assertLess(response['gradient_enabled_objective'], response['native_full_objective'])
            chunks = response['chunks']
            self.assertTrue(all(chunk['gradient']['excluded_nonzero'] == 0 for chunk in chunks))
            inactive_name = 'prefix_nonzero' if role == 'E' else 'error_nonzero'
            self.assertEqual(sum(chunk['gradient'][inactive_name] for chunk in chunks), 0)

    def test_two_response_normalization_and_reset_do_not_mix_occurrences(self):
        model = self.model()
        controller = install_native_prefix_roles(model, 'E', capture_gradients=True)
        logits, teacher, labels, weights = mixed_logits()
        responses = []
        for scale in (1., 3.):
            current = logits.clone().requires_grad_()
            loss, _ = self.evaluate_chunks(model, current, teacher, labels, weights * scale, controller)
            (loss / 2).backward()
            responses.append((loss.detach(), current.grad))
        receipt = controller.receipt()
        self.assertEqual(len(receipt['responses']), 2)
        self.assertEqual(receipt['unique_projection_chunks'], 6)
        self.assertNotEqual(receipt['responses'][0]['denominator'], receipt['responses'][1]['denominator'])
        for index, scale in enumerate((1., 3.)):
            independent = self.model()
            other_controller = install_native_prefix_roles(independent, 'E')
            current = logits.clone().requires_grad_()
            loss, _ = self.evaluate_chunks(independent, current, teacher, labels, weights * scale, other_controller)
            loss.backward()
            torch.testing.assert_close(responses[index][1].float() * 2, current.grad.float(), rtol=0, atol=0)
        controller.reset(role='C')
        self.assertEqual(controller.receipt()['responses'], [])
        self.assertEqual(controller.receipt()['observed_chunk_calls'], 0)

    def test_real_tiny_forward_preserves_native_parameters_rng_and_gradient_patterns(self):
        author = tiny_author(chunk_blocks=2)
        baseline = TargetObjectiveModel(copy.deepcopy(author), 'osd_first_error', loss_precision='bfloat16')
        wrapped = TargetObjectiveModel(copy.deepcopy(author), 'osd_first_error', loss_precision='bfloat16')
        for model in (baseline, wrapped):
            install_kplus(model)
            model.begin_event()
        controller = install_native_prefix_roles(wrapped, 'ALL', capture_gradients=True)
        batch = response_batch(baseline)
        batch['target_last_hidden_states'] = torch.randn(1, 12, 16).to(torch.bfloat16)
        initial_rng = torch.get_rng_state()
        expected, _, _ = baseline(**batch)
        expected.backward()
        expected_rng = torch.get_rng_state()
        torch.set_rng_state(initial_rng)
        actual, _, _ = wrapped(**batch)
        actual.backward()
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
        for (name, left), (_, right) in zip(baseline.named_parameters(), wrapped.named_parameters()):
            self.assertEqual(left.grad is None, right.grad is None, name)
            if left.grad is not None:
                self.assertTrue(torch.equal(left.grad, right.grad), name)
        self.assertEqual(len(controller.receipt()['responses']), 1)

    def test_partial_and_nonfinite_receipts_remain_serializable(self):
        controller = NativePrefixRoles('C', capture_gradients=True)
        controller.begin_response()
        rows = torch.tensor([[[0., float('nan'), 2.]]], requires_grad=True)
        labels = torch.tensor([[[0, 1, 2]]])
        proposals = torch.tensor([[[0, 1, 9]]])
        weights = torch.tensor([[[0., 1., 1.]]])
        selected = first_error_mask(proposals, labels, weights)
        controlled = controller.control_rows(rows, proposals, labels, weights, selected, chunk_key=0)
        controlled.sum().backward()
        receipt = controller.receipt()
        self.assertFalse(receipt['responses'][0]['finished'])
        self.assertEqual(receipt['responses'][0]['chunks'][0]['_active_terms'][0], 'nan')
        json.dumps(receipt, allow_nan=False)

    def test_explicit_zero_current_gradients_are_not_none_under_adam_history(self):
        parameter = torch.nn.Parameter(torch.tensor([1., 2.], dtype=torch.bfloat16))
        unused = torch.nn.Parameter(torch.tensor([3.], dtype=torch.bfloat16))
        optimizer = torch.optim.AdamW([parameter, unused], lr=.01, weight_decay=.01)
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        before = parameter.detach().clone()
        untouched = unused.detach().clone()
        parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
        self.assertFalse(torch.equal(parameter, before))
        self.assertEqual(float(optimizer.state[parameter]['step']), 2.)
        self.assertTrue(torch.equal(unused, untouched))
        self.assertNotIn(unused, optimizer.state)


if __name__ == '__main__':
    unittest.main()
