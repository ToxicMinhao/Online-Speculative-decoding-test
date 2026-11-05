"""Teacher alignment and forward-KL on actual tiny DFlash training blocks."""
import copy
import unittest
from unittest import mock

import torch
import torch.nn.functional as F

from online_feedback.objective import TargetObjectiveModel
from test_objective_integration import tiny_author, response_batch, parameter_gradients


def osd_batch(model):
    batch = response_batch(model)
    generator = torch.Generator().manual_seed(117)
    batch['target_last_hidden_states'] = torch.randn(
        1, batch['input_ids'].shape[1], 16, generator=generator,
        dtype=next(model.draft_model.parameters()).dtype,
    )
    return batch


def direct_kl(model, batch, anchors, keep, hidden):
    """Independent per-position oracle, without the wrapper's gather or masks."""
    hidden = hidden.reshape(1, anchors.shape[1], model.block_size, -1)
    numerator = hidden.new_zeros((), dtype=torch.float32)
    denominator = numerator.clone()
    positions = []
    for block in range(anchors.shape[1]):
        if not keep[0, block]:
            continue
        for offset in range(1, model.block_size):
            position = int(anchors[0, block]) + offset
            if position >= batch['input_ids'].shape[1] or not batch['loss_mask'][0, position]:
                continue
            logits = model.lm_head(hidden[0, block, offset]).float()
            with torch.no_grad():
                teacher_logits = model.lm_head(batch['target_last_hidden_states'][0, position - 1]).float()
                teacher = teacher_logits.softmax(-1)
                log_teacher = teacher_logits.log_softmax(-1)
            weight = torch.exp(torch.tensor(-(offset - 1) / model.loss_decay_gamma))
            numerator = numerator + (teacher * (log_teacher - logits.log_softmax(-1))).sum() * weight
            denominator = denominator + weight
            positions.append(position)
    return numerator, denominator + 1e-6, positions


class OSDObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_exact_teacher_predecessor_alignment_masks_loss_and_gradients(self):
        model = TargetObjectiveModel(tiny_author(dtype=torch.float32), 'osd')
        batch = osd_batch(model)
        batch['loss_mask'][0, 6] = 0
        anchors = torch.tensor([[4, 8, 10, 0]])
        keep = torch.tensor([[True, True, True, False]])
        draft_inputs = {name: batch[name] for name in ('input_ids', 'hidden_states', 'loss_mask')}
        with mock.patch.object(model, '_sample_anchor_positions', return_value=(anchors, keep)):
            _, _, hidden = model._forward_draft_blocks(**draft_inputs)
            numerator, denominator, positions = direct_kl(model, batch, anchors, keep, hidden)
            expected = numerator / denominator
            expected.backward()
            expected_gradients = parameter_gradients(model)
            model.zero_grad(set_to_none=True)
            actual, _, metrics = model(**batch)
            actual.backward()
        self.assertEqual(positions, [5, 7, 9, 10, 11, 11])
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(metrics['loss_terms'][0], numerator, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(metrics['loss_terms'][1], denominator, atol=2e-6, rtol=2e-6)
        self.assertEqual(int(metrics['sum_metrics']['supervised_position_exposure']), len(positions))
        for name, gradient in parameter_gradients(model).items():
            torch.testing.assert_close(gradient, expected_gradients[name], atol=2e-6, rtol=2e-5)

    def test_chunk_sizes_preserve_values_gradients_and_rng(self):
        author = tiny_author(dtype=torch.float32)
        outputs = []
        for chunk_blocks in (1, 2, 64):
            model = TargetObjectiveModel(copy.deepcopy(author), 'osd')
            model.objective_chunk_blocks = chunk_blocks
            torch.manual_seed(17)
            loss, _, metrics = model(**osd_batch(model))
            loss.backward()
            outputs.append((loss.detach(), parameter_gradients(model), torch.get_rng_state(), metrics))
        for actual in outputs[1:]:
            torch.testing.assert_close(actual[0], outputs[0][0], atol=2e-6, rtol=2e-6)
            self.assertTrue(torch.equal(actual[2], outputs[0][2]))
            torch.testing.assert_close(actual[3]['loss_terms'][1], outputs[0][3]['loss_terms'][1])
            for name, gradient in actual[1].items():
                torch.testing.assert_close(gradient, outputs[0][1][name], atol=2e-6, rtol=2e-5)

    def test_teacher_features_and_target_weights_never_receive_gradients(self):
        model = TargetObjectiveModel(tiny_author(), 'osd')
        batch = osd_batch(model)
        batch['target_last_hidden_states'].requires_grad_(True)
        model.begin_event()
        loss, _, _ = model(**batch)
        loss.backward()
        self.assertIsNone(batch['target_last_hidden_states'].grad)
        for module in (model.lm_head, model.embed_tokens):
            self.assertTrue(all(not p.requires_grad and p.grad is None for p in module.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in model.draft_model.parameters()))
        self.assertIsNone(model.reference)
        self.assertGreater(model.end_event()['teacher_head_s'], 0.0)

    def test_identical_distributions_give_zero_kl(self):
        model = TargetObjectiveModel(tiny_author(dtype=torch.float32), 'osd')
        hidden = torch.randn(1, 2, 4, 16, requires_grad=True)
        labels = torch.zeros(1, 2, 4, dtype=torch.long)
        weights = torch.ones_like(labels, dtype=torch.float32)
        numerator, _, _, _ = model._chunk_loss(hidden, None, labels, weights, hidden.detach())
        self.assertLess(abs(float(numerator.detach())), 2e-6)
        numerator.backward()
        self.assertLess(float(hidden.grad.abs().max()), 2e-7)

    def test_one_hot_teacher_is_hard_ce_and_zero_weight_positions_have_no_gradient(self):
        model = TargetObjectiveModel(tiny_author(dtype=torch.float32), 'osd')
        model.lm_head = torch.nn.Identity()
        hidden = torch.tensor([[[[0.2, -0.1, 0.3], [0.4, 0.1, -0.2]]]], requires_grad=True)
        teacher = torch.tensor([[[[-1000., 1000., -1000.], [1000., -1000., -1000.]]]])
        labels = torch.tensor([[[1, 0]]])
        weights = torch.tensor([[[1., 0.]]])
        numerator, _, _, _ = model._chunk_loss(hidden, None, labels, weights, teacher)
        expected = F.cross_entropy(hidden[0, 0, :1], labels[0, 0, :1])
        torch.testing.assert_close(numerator, expected)
        numerator.backward()
        self.assertTrue(torch.equal(hidden.grad[0, 0, 1], torch.zeros(3)))

    def test_missing_teacher_is_an_explicit_error(self):
        model = TargetObjectiveModel(tiny_author(), 'osd')
        with self.assertRaisesRegex(ValueError, 'requires frozen target_last_hidden_states'):
            model(**response_batch(model))


if __name__ == '__main__':
    unittest.main()
