"""Ranking through real tiny DFlash blocks, with CPU-only synthetic inputs."""
import copy
import random
import unittest
import weakref
from unittest import mock

import numpy as np
import torch
import torch.nn.functional as F

from online_feedback.objective import TargetObjectiveModel
from online_feedback.ranking import matching_prefix_mask
from online_feedback.rng import capture_rng, restore_rng
from test_objective_integration import tiny_author, parameter_gradients
from test_osd_objective import osd_batch
from test_training_contract import trees_equal


class RankingObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def model(self, arm='osd_prefix', dtype=torch.float32, **kwargs):
        return TargetObjectiveModel(tiny_author(dtype=dtype), arm, **kwargs)

    def test_direct_loss_and_gradients_match_both_arms(self):
        for arm in ('osd_prefix', 'osd_uniform'):
            model = self.model(arm)
            model.begin_event()
            batch = osd_batch(model)
            anchors = torch.tensor([[4, 6, 8, 10]])
            keep = torch.ones_like(anchors, dtype=torch.bool)
            # Controlled valid reference prefixes exercise nonzero ranking mass.
            prefix = torch.tensor([[[0, 1, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0]]],
                                  dtype=torch.bool)
            with mock.patch.object(model, '_sample_anchor_positions', return_value=(anchors, keep)), \
                    mock.patch.object(model, '_reference_prefix', return_value=prefix):
                loss, _, metrics = model(**batch)
                loss.backward()
                actual_gradients = parameter_gradients(model)
                model.zero_grad(set_to_none=True)
                _, _, hidden = model._forward_draft_blocks(
                    **{key: batch[key] for key in ('input_ids', 'hidden_states', 'loss_mask')})
                hidden = hidden.reshape(1, 4, 4, -1)
                terms = []
                weights = []
                flags = []
                for block, anchor in enumerate(anchors[0].tolist()):
                    for offset in range(1, 4):
                        position = anchor + offset
                        if position >= batch['input_ids'].shape[1]:
                            continue
                        weight = torch.exp(torch.tensor(-(offset - 1) / 2.))
                        label = int(batch['input_ids'][0, position])
                        logits = model.lm_head(hidden[0, block, offset])
                        teacher = model.lm_head(batch['target_last_hidden_states'][0, position - 1]).softmax(-1)
                        kl = F.kl_div(logits.log_softmax(-1), teacher, reduction='sum')
                        competing_indices = [i for i in range(logits.numel()) if i != label]
                        rival = competing_indices[int(logits.detach()[competing_indices].argmax())]
                        ranking = F.softplus(logits[rival] - logits[label])
                        terms.append((kl, ranking))
                        weights.append(weight)
                        flags.append(float(prefix[0, block, offset]))
                denominator = sum(weights) + 1e-6
                rho = sum(w * g for w, g in zip(weights, flags)) / sum(weights)
                numerator = sum(w * (kl + (g if arm == 'osd_prefix' else rho) * rank)
                                for w, g, (kl, rank) in zip(weights, flags, terms))
                expected = numerator / denominator
                expected.backward()
            torch.testing.assert_close(loss, expected, atol=2e-6, rtol=2e-6)
            torch.testing.assert_close(metrics['loss_terms'][1], denominator)
            for name, gradient in parameter_gradients(model).items():
                torch.testing.assert_close(actual_gradients[name], gradient, atol=2e-6, rtol=2e-5)

    def test_zero_lambda_is_bitwise_baseline_loss_gradients_and_rng(self):
        author = tiny_author()
        initial_rng = capture_rng()
        outputs = []
        for arm in ('osd', 'osd_uniform', 'osd_prefix'):
            model = TargetObjectiveModel(copy.deepcopy(author), arm, loss_precision='bfloat16', ranking_lambda=0)
            restore_rng(initial_rng)
            model.begin_event()
            loss, _, metrics = model(**osd_batch(model))
            loss.backward()
            outputs.append((loss.detach(), metrics['loss_terms'][1], parameter_gradients(model), capture_rng()))
            self.assertIsNone(model.reference)
        self.assertTrue(trees_equal(outputs[0], outputs[1]))
        self.assertTrue(trees_equal(outputs[0], outputs[2]))

    def test_reference_is_independent_fixed_and_does_not_change_current_modes(self):
        model = self.model(dtype=torch.bfloat16)
        model.train()
        model.draft_model.layers[0].eval()  # Preserve a mixed current mode tree.
        modes = {name: module.training for name, module in model.named_modules()}
        rng = capture_rng()
        model.begin_event()
        self.assertTrue(trees_equal(rng, capture_rng()))
        self.assertEqual(modes, {name: module.training for name, module in model.named_modules()})
        reference = model.reference.model.draft_model
        state = copy.deepcopy(reference.state_dict())
        for name, parameter in reference.named_parameters():
            current = dict(model.draft_model.named_parameters())[name]
            self.assertNotEqual(parameter.data_ptr(), current.data_ptr())
            self.assertFalse(parameter.requires_grad)
        self.assertTrue(all(not module.training for module in reference.modules()))
        with torch.no_grad():
            next(model.draft_model.parameters()).add_(.25)
        self.assertTrue(trees_equal(state, reference.state_dict()))
        self.assertFalse(any('reference' in name for name in model.state_dict()))
        model.end_event()
        self.assertIsNone(model.reference)
        model.begin_event()
        self.assertTrue(trees_equal(model.draft_model.state_dict(), model.reference.model.draft_model.state_dict()))

    def test_reference_runs_first_without_rng_effect_or_retained_activations(self):
        model = self.model()
        model.begin_event()
        batch = osd_batch(model)
        reference_forward = model.reference.model.draft_model.forward
        current_forward = model.draft_model.forward
        hidden_refs = []
        phases = []

        def reference_call(*args, **kwargs):
            self.assertFalse(torch.is_grad_enabled())
            phases.append('reference')
            random.random(); np.random.rand(); torch.rand(4)
            result = reference_forward(*args, **kwargs)
            hidden_refs.append(weakref.ref(result))
            return result

        def current_call(*args, **kwargs):
            phases.append('current')
            self.assertTrue(torch.is_grad_enabled())
            self.assertTrue(all(ref() is None for ref in hidden_refs))
            return current_forward(*args, **kwargs)

        rng = capture_rng()
        with mock.patch.object(model.reference.model.draft_model, 'forward', side_effect=reference_call), \
                mock.patch.object(model.draft_model, 'forward', side_effect=current_call), \
                mock.patch.object(model, '_sample_anchor_positions', wraps=model._sample_anchor_positions) as sampler:
            loss, _, _ = model(**batch)
            self.assertEqual(sampler.call_count, 1)
            sampled_rng = capture_rng()
            loss.backward()
        self.assertEqual(phases, ['reference', 'current'])
        restore_rng(rng)
        model._sample_anchor_positions(12, batch['loss_mask'], torch.device('cpu'))
        self.assertTrue(trees_equal(sampled_rng, capture_rng()))
        self.assertTrue(all(p.grad is None for p in model.reference.model.draft_model.parameters()))

    def test_reference_mask_matches_same_weight_current_on_identical_full_blocks(self):
        model = self.model(dtype=torch.bfloat16)
        model.eval(); model.begin_event()
        batch = osd_batch(model)
        anchors, keep = model._sample_anchor_positions(12, batch['loss_mask'], torch.device('cpu'))
        labels, weights, _ = model._block_labels_and_weights(batch['input_ids'], batch['loss_mask'], anchors, keep)
        with torch.no_grad(), mock.patch.object(model, '_sample_anchor_positions', return_value=(anchors, keep)):
            _, _, hidden = model._forward_draft_blocks(
                **{key: batch[key] for key in ('input_ids', 'hidden_states', 'loss_mask')})
            proposals = model.lm_head(hidden.reshape(1, 4, 4, -1)).argmax(-1)
            expected = matching_prefix_mask(proposals, labels, weights)
            actual = model._reference_prefix(batch['input_ids'], batch['hidden_states'], batch['loss_mask'],
                                             anchors, keep, labels, weights)
        self.assertTrue(torch.equal(expected, actual))

    def test_chunking_keeps_response_rho_losses_gradients_and_rng(self):
        author = tiny_author(dtype=torch.float32)
        for arm in ('osd_uniform', 'osd_prefix'):
            outputs = []
            for chunk_size in (1, 2, 64):
                model = TargetObjectiveModel(copy.deepcopy(author), arm)
                model.objective_chunk_blocks = chunk_size
                model.begin_event()
                # Prefix selection is fixture-controlled to ensure nonzero auxiliary gradients.
                def selected(*args):
                    weights = args[-1]
                    prefix = weights > 0
                    prefix[:, 1::2] = False
                    return prefix
                torch.manual_seed(17)
                with mock.patch.object(model, '_reference_prefix', side_effect=selected):
                    loss, _, metrics = model(**osd_batch(model))
                    loss.backward()
                outputs.append((loss.detach(), parameter_gradients(model), metrics, torch.get_rng_state()))
            for actual in outputs[1:]:
                torch.testing.assert_close(actual[0], outputs[0][0], atol=2e-6, rtol=2e-6)
                self.assertTrue(torch.equal(actual[3], outputs[0][3]))
                torch.testing.assert_close(actual[2]['sum_metrics']['ranking_rho'],
                                           outputs[0][2]['sum_metrics']['ranking_rho'])
                for name, gradient in actual[1].items():
                    torch.testing.assert_close(gradient, outputs[0][1][name], atol=2e-6, rtol=2e-5)

    def test_reference_projection_chunks_do_not_shrink_attention_blocks(self):
        model = self.model(); model.objective_chunk_blocks = 1; model.begin_event()
        batch = osd_batch(model)
        reference = model.reference.model.draft_model
        shapes = []
        handle = reference.register_forward_pre_hook(lambda module, args, kwargs: shapes.append(
            (kwargs['position_ids'].shape, kwargs['noise_embedding'].shape)), with_kwargs=True)
        try:
            loss, _, _ = model(**batch)
            loss.backward()
        finally:
            handle.remove()
        self.assertEqual(len(shapes), 1)
        self.assertEqual(shapes[0][0][-1], 12 + 4 * 4)
        self.assertEqual(shapes[0][1][-2], 4 * 4)

    def test_empty_response_keeps_original_error_without_resampling(self):
        for arm in ('osd', 'osd_uniform', 'osd_prefix'):
            model = self.model(arm); model.begin_event(); batch = osd_batch(model)
            batch['loss_mask'].zero_()
            with mock.patch.object(model, '_sample_anchor_positions', wraps=model._sample_anchor_positions) as sampler:
                with self.assertRaisesRegex(ValueError, 'two consecutive supervised tokens'):
                    model(**batch)
            self.assertEqual(sampler.call_count, 1)

    def test_reference_conditioning_excludes_features_at_or_after_anchor(self):
        model = self.model(); model.eval(); model.begin_event()
        batch = osd_batch(model)
        anchors = torch.tensor([[4]])
        keep = torch.tensor([[True]])
        original = model.reference.hidden(batch['input_ids'], batch['hidden_states'], batch['loss_mask'],
                                          anchors, keep)
        future = batch['hidden_states'].clone()
        future[:, 4:] += 100
        blocked = model.reference.hidden(batch['input_ids'], future, batch['loss_mask'], anchors, keep)
        torch.testing.assert_close(original, blocked, atol=0, rtol=0)
        earlier = batch['hidden_states'].clone()
        earlier[:, :4] += 100
        visible = model.reference.hidden(batch['input_ids'], earlier, batch['loss_mask'], anchors, keep)
        self.assertGreater(float((original - visible).abs().max()), 1e-5)

    def test_reference_buffer_storage_is_independent(self):
        model = self.model()
        model.draft_model.register_buffer('test_reference_buffer', torch.tensor([3.]))
        model.begin_event()
        current = model.draft_model.test_reference_buffer
        reference = model.reference.model.draft_model.test_reference_buffer
        self.assertNotEqual(current.data_ptr(), reference.data_ptr())
        current.add_(7)
        self.assertEqual(float(reference[0]), 3.)


if __name__ == '__main__':
    unittest.main()
