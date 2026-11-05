"""KP numerical contracts on CPU, including actual tiny DFlash forwards."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from safetensors.torch import save_file

from online_feedback.first_error import first_error_mask
from online_feedback.objective import TargetObjectiveModel
from online_feedback.prefix_protection import (
    ReleasedPrefixObjective, canonical_margin, protection_terms,
    combine_response_losses, REFERENCE_FRACTION, REFERENCE_MARGIN_CAP,
)
from online_feedback.ranking import matching_prefix_mask
from online_feedback.rng import capture_rng
from specforge.core.chunking import checkpointed_chunk_reduce
from test_objective_integration import tiny_author, parameter_gradients
from test_osd_objective import osd_batch


def assert_rng_equal(test, left, right):
    test.assertEqual(left['python'], right['python'])
    test.assertEqual(left['numpy'][0], right['numpy'][0])
    np.testing.assert_array_equal(left['numpy'][1], right['numpy'][1])
    test.assertEqual(left['numpy'][2:], right['numpy'][2:])
    test.assertTrue(torch.equal(left['torch'], right['torch']))
    test.assertEqual(len(left['cuda']), len(right['cuda']))
    for a, b in zip(left['cuda'], right['cuda']):
        test.assertTrue(torch.equal(a, b))


class PrefixProtectionFormulaTests(unittest.TestCase):
    def test_prefix_eos_gap_boundary_seed_and_first_error(self):
        labels = torch.tensor([[[4, 1, 2, 3, 0], [4, 1, 2, 3, 0],
                                [4, 1, 9, 3, 0], [4, 1, 2, 3, 0]]])
        predictions = labels.clone()
        predictions[0, 0, 2] = 8
        weights = torch.tensor([[[0., 1, 1, 1, 0], [0, 1, 0, 1, 1],
                                  [0, 1, 1, 1, 0], [0, 0, 0, 0, 0]]])
        protected = matching_prefix_mask(predictions, labels, weights, (9,))
        selected = first_error_mask(predictions, labels, weights, (9,))
        self.assertEqual(protected.tolist(), [[[False, True, False, False, False],
                                               [False, True, False, False, False],
                                               [False, True, True, False, False],
                                               [False] * 5]])
        self.assertTrue(selected[0, 0, 2])
        self.assertFalse(protected[0, 0, 2])

    def test_fully_correct_and_first_wrong_prefix(self):
        labels = torch.tensor([[[0, 1, 2, 3], [0, 1, 2, 3]]])
        predictions = labels.clone()
        predictions[0, 1, 1] = 8
        weights = torch.tensor([[[0., 1, 1, 1], [0., 1, 1, 1]]])
        self.assertEqual(matching_prefix_mask(predictions, labels, weights).tolist(),
                         [[[False, True, True, True], [False] * 4]])

    def test_native_rival_ties_and_differentiable_gathers(self):
        logits = torch.tensor([[2., 2., 1., 2.], [3., 2., 3., 1.]],
                              dtype=torch.bfloat16, requires_grad=True)
        margins = canonical_margin(logits, torch.tensor([1, 2]))
        self.assertEqual(margins.dtype, torch.float32)
        torch.testing.assert_close(margins, torch.tensor([0., 0.]))
        margins.sum().backward()
        torch.testing.assert_close(logits.grad.float(),
                                   torch.tensor([[-1., 1., 0., 0.], [-1., 0., 1., 0.]]))

    def test_reference_threshold_cap_has_no_floor(self):
        margins = torch.tensor([0., .25, 1., 4.])
        threshold = REFERENCE_FRACTION * margins.clamp(max=REFERENCE_MARGIN_CAP)
        torch.testing.assert_close(threshold, torch.tensor([0., .125, .5, .5]))
        values = margins.clone().requires_grad_()
        numerator, denominator = protection_terms(values, torch.ones(4),
                                                   torch.ones(4, dtype=torch.bool), threshold)
        numerator.backward()
        self.assertEqual(float(numerator), 0.)
        self.assertEqual(float(denominator), 4.)
        self.assertTrue(torch.equal(values.grad, torch.zeros(4)))

    def test_hinge_active_inactive_equality_empty_and_detachment(self):
        margins = torch.tensor([-.5, .5, 1.], requires_grad=True)
        threshold = torch.tensor([.5, .5, .5], requires_grad=True)
        numerator, weight = protection_terms(margins, torch.tensor([1., 2., 3.]),
                                             torch.ones(3, dtype=torch.bool), threshold)
        numerator.backward()
        self.assertEqual(float(numerator), 1.)
        self.assertEqual(float(weight), 6.)
        torch.testing.assert_close(margins.grad, torch.tensor([-1., 0., 0.]))
        self.assertIsNone(threshold.grad)
        empty, empty_weight = protection_terms(margins, torch.ones(3),
                                               torch.zeros(3, dtype=torch.bool), threshold)
        total, k_loss, penalty = combine_response_losses(torch.tensor(4.), torch.tensor(2.),
                                                         empty, empty_weight)
        self.assertEqual(float(penalty), 0.)
        self.assertEqual(float(total), float(k_loss))

    def test_current_rival_is_not_reference_rival(self):
        labels = torch.tensor([0])
        reference = torch.tensor([[2., 1., -1.]])
        current = torch.tensor([[0., -1., 3.]], requires_grad=True)
        threshold = .5 * canonical_margin(reference, labels)
        numerator, _ = protection_terms(canonical_margin(current, labels), torch.ones(1),
                                         torch.ones(1, dtype=torch.bool), threshold)
        numerator.backward()
        torch.testing.assert_close(current.grad, torch.tensor([[-1., 0., 1.]]))

    def test_separate_denominators_and_response_mean(self):
        first, k_first, p_first = combine_response_losses(
            torch.tensor(8.), torch.tensor(4.), torch.tensor(10.), torch.tensor(2.))
        second, _, _ = combine_response_losses(
            torch.tensor(2.), torch.tensor(2.), torch.tensor(0.), torch.tensor(0.))
        expected = 8 / (4 + 1e-6) + .1 * 10 / (2 + 1e-6)
        self.assertAlmostEqual(float(first), expected, places=6)
        self.assertAlmostEqual(float((first + second) / 2), (expected + 2 / (2 + 1e-6)) / 2, places=6)
        self.assertNotAlmostEqual(float(first), (8 + .1 * 10) / (4 + 1e-6), places=3)
        torch.testing.assert_close(first, k_first + .1 * p_first)

    def test_chunked_fp32_loss_and_gradient_match_whole_response(self):
        outputs = []
        for chunk_size in (0, 1, 3):
            logits = torch.tensor([[.3, -1., 1.2], [-.7, .4, 1.1],
                                   [1.2, .1, -.8], [-.9, .4, .2]], requires_grad=True)
            labels = torch.tensor([0, 1, 0, 2])
            weights = torch.tensor([.8, 1., .4, .2])
            protected = torch.tensor([True, True, False, True])
            thresholds = torch.tensor([.4, .25, .2, .1])

            def terms(z, y, w, p, threshold):
                main = -z.log_softmax(-1).gather(-1, y[:, None]).squeeze(-1)
                p_num, p_weight = protection_terms(canonical_margin(z, y), w, p, threshold)
                return (main * w).sum(), w.sum(), p_num, p_weight

            reductions = checkpointed_chunk_reduce(terms, logits, labels, weights,
                                                     protected, thresholds, chunk_size=chunk_size)
            total, _, _ = combine_response_losses(*reductions)
            total.backward()
            outputs.append((total.detach(), logits.grad))
        for loss, gradient in outputs[1:]:
            torch.testing.assert_close(loss, outputs[0][0], atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(gradient, outputs[0][1], atol=1e-6, rtol=1e-5)


class PrefixProtectionIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.author = tiny_author()
        self.released = {name: value.clone() for name, value in self.author.draft_model.state_dict().items()}
        save_file(self.released, str(Path(self.directory.name) / 'model.safetensors'))

    def model(self, **kwargs):
        result = ReleasedPrefixObjective(copy.deepcopy(self.author), self.directory.name, **kwargs)
        self.addCleanup(result.close_counters)
        return result

    def test_production_rejects_qualification_override_and_changed_constants(self):
        for kwargs in ({'forced_active': True}, {'coefficient': 0}, {'loss_precision': 'float32'},
                       {'coefficient': .2, 'qualification': True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.model(**kwargs)

    def test_reference_comes_from_released_checkpoint_not_current_or_next_event(self):
        with torch.no_grad():
            for parameter in self.author.draft_model.parameters():
                parameter.add_(.25)
        current_modes = (self.author.lm_head.training, self.author.embed_tokens.training)
        rng_before = capture_rng()
        model = self.model()
        assert_rng_equal(self, rng_before, capture_rng())
        self.assertEqual(current_modes, (model.lm_head.training, model.embed_tokens.training))
        reference = model.released_reference.model.draft_model
        self.assertFalse(reference.training)
        self.assertTrue(model.draft_model.training)
        for name, value in reference.state_dict().items():
            self.assertTrue(torch.equal(value, self.released[name]), name)
        self.assertTrue(all(not p.requires_grad for p in reference.parameters()))
        reference_ids = {id(p) for p in reference.parameters()}
        self.assertFalse(reference_ids & {id(p) for p in model.parameters()})
        self.assertFalse(any('released_reference' in name for name in model.state_dict()))
        model.end_event()
        model.begin_event()
        self.assertIs(model.released_reference.model.draft_model, reference)

    def test_neutral_reference_is_exercised_and_native_k_is_exact(self):
        ordinary = TargetObjectiveModel(copy.deepcopy(self.author), 'osd_first_error', loss_precision='bfloat16')
        ordinary.begin_event()
        neutral = self.model(coefficient=0, qualification=True)
        observations = []
        for model in (ordinary, neutral):
            batch = osd_batch(model)
            torch.manual_seed(17)
            sampler = model._sample_anchor_positions
            sampled = []

            def sample(*args, **kwargs):
                result = sampler(*args, **kwargs)
                sampled.append(tuple(value.clone() for value in result))
                return result

            with mock.patch.object(model, '_sample_anchor_positions', side_effect=sample):
                loss, accuracy, metrics = model(**batch)
                loss.backward()
            observations.append((loss.detach(), accuracy, metrics, parameter_gradients(model),
                                 capture_rng(), sampled))
            self.assertEqual(len(sampled), 1)
        a, b = observations
        self.assertTrue(torch.equal(a[0], b[0]))
        self.assertTrue(torch.equal(a[1], b[1]))
        self.assertEqual(a[2].keys(), b[2].keys())
        for key in a[3]:
            self.assertTrue(torch.equal(a[3][key], b[3][key]), key)
        assert_rng_equal(self, a[4], b[4])
        for left, right in zip(a[5][0], b[5][0]):
            self.assertTrue(torch.equal(left, right))
        self.assertEqual(neutral.work['reference_response_forwards'], 1)
        self.assertEqual(neutral.work['current_response_forwards'], 1)
        self.assertTrue(neutral.last_response_record['neutral_reference_exercised'])

    def test_forced_active_path_adds_gradient_and_records_each_response_once(self):
        normal = self.model()
        forced = self.model(qualification=True, forced_active=True)
        results = []
        for model in (normal, forced):
            batch = osd_batch(model)
            batch['target_last_hidden_states'].requires_grad_()
            records = []
            model.response_observer = records.append
            torch.manual_seed(17)
            loss, _, metrics = model(**batch)
            forward_chunks = model.work['current_objective_chunks']
            loss.backward()
            self.assertEqual(len(records), 1)
            self.assertEqual(model.work['reference_response_forwards'], 1)
            self.assertGreater(model.work['current_objective_chunks'], forward_chunks)
            self.assertEqual(sum(records[0]['protected_prefix_histogram']),
                             records[0]['protected_prefix_histogram_denominator'])
            self.assertTrue(records[0]['finite'])
            self.assertIsNone(batch['target_last_hidden_states'].grad)
            self.assertTrue(all(p.grad is None for p in model.released_reference.model.draft_model.parameters()))
            self.assertTrue(torch.equal(metrics['loss_terms'][0], loss))
            self.assertEqual(float(metrics['loss_terms'][1]), 1.)
            self.assertFalse(metrics['loss_terms'][1].requires_grad)
            results.append((parameter_gradients(model), records[0]))
        self.assertEqual(results[0][1]['protection_loss'], 0.)
        self.assertGreater(results[1][1]['protection_loss'], .99)
        self.assertTrue(any(not torch.equal(value, results[1][0][name]) for name, value in results[0][0].items()))

    def test_checkpointed_response_shape_is_local_not_mutable_model_state(self):
        model = self.model(qualification=True, forced_active=True)
        losses = []
        for anchors in (torch.tensor([[4, 7, 10]]), torch.tensor([[4, 8]])):
            keep = torch.ones_like(anchors, dtype=torch.bool)
            with mock.patch.object(model, '_sample_anchor_positions', return_value=(anchors, keep)):
                losses.append(model(**osd_batch(model))[0])
        ((losses[0] + losses[1]) / 2).backward()
        self.assertTrue(all(torch.isfinite(g).all() for g in parameter_gradients(model).values()))

    def test_reference_protection_contributes_beyond_current_k_first_error(self):
        model = self.model()
        model.lm_head = torch.nn.Identity()
        labels = torch.tensor([[[0, 0, 1, 2]]])
        weights = torch.tensor([[[0., 1., .5, .25]]])
        teacher = torch.zeros(1, 1, 4, 3, dtype=torch.bfloat16)
        thresholds = torch.full_like(weights, .5)
        results = []
        for protected in (torch.zeros_like(labels, dtype=torch.bool), weights > 0):
            logits = torch.tensor([[[[1., 0., 0.], [-1., 1., 0.],
                                     [-1., 0., 1.], [0., 0., 1.]]]],
                                  dtype=torch.bfloat16, requires_grad=True)
            terms = model._protected_chunk(logits, labels, weights, teacher,
                                            protected, thresholds, torch.tensor([[0]]), total_blocks=1)
            total, main, _ = combine_response_losses(terms[0], terms[1], terms[5], terms[6])
            total.backward()
            results.append((main.detach(), logits.grad.clone()))
        self.assertTrue(torch.equal(results[0][0], results[1][0]))
        self.assertTrue(torch.equal(results[0][1][0, 0, 2], torch.zeros(3, dtype=torch.bfloat16)))
        self.assertGreater(float(results[1][1][0, 0, 2].abs().sum()), 0.)

    def test_reference_projection_restores_rng_even_when_head_consumes_it(self):
        model = self.model()
        batch = osd_batch(model)
        anchors = torch.tensor([[4, 7, 10]])
        keep = torch.ones_like(anchors, dtype=torch.bool)
        labels, weights, _ = model._block_labels_and_weights(batch['input_ids'], batch['loss_mask'], anchors, keep)
        original_head = model.lm_head.forward

        def draw_then_project(hidden):
            torch.rand(3)
            return original_head(hidden)

        before = capture_rng()
        with mock.patch.object(model.lm_head, 'forward', side_effect=draw_then_project):
            protected, thresholds, _ = model._reference_values(
                batch['input_ids'], batch['hidden_states'], batch['loss_mask'],
                anchors, keep, labels, weights)
        assert_rng_equal(self, before, capture_rng())
        self.assertFalse(protected.requires_grad)
        self.assertFalse(thresholds.requires_grad)


if __name__ == '__main__':
    unittest.main()
