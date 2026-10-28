"""CPU oracles for detached, current-model selection within each DFlash block."""

import unittest

import torch

from online_feedback.first_error import first_error_mask, selection_statistics


def loop_selection(proposals, labels, weights, eos_token_ids=()):
    """Independent scalar specification: seed, boundary, match or first error."""
    result = torch.zeros_like(labels, dtype=torch.bool)
    flat_proposals = proposals.reshape(-1, proposals.shape[-1])
    flat_labels = labels.reshape_as(flat_proposals)
    flat_weights = weights.reshape_as(flat_proposals)
    flat_result = result.reshape_as(flat_proposals)
    for block in range(flat_labels.shape[0]):
        if int(flat_labels[block, 0]) in eos_token_ids:
            continue
        for offset in range(1, flat_labels.shape[1]):
            if flat_weights[block, offset] <= 0:
                break
            flat_result[block, offset] = True
            if (flat_proposals[block, offset] != flat_labels[block, offset]
                    or int(flat_labels[block, offset]) in eos_token_ids):
                break
    return result


class FirstErrorMaskTests(unittest.TestCase):
    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized(), "CPU test initialized CUDA")

    def test_first_last_no_error_are_independent_and_include_first_error(self):
        labels = torch.tensor([[[4, 1, 2, 3], [4, 1, 2, 3], [4, 1, 2, 3]]])
        proposals = torch.tensor([[[0, 0, 2, 3], [0, 1, 2, 0], [0, 1, 2, 3]]])
        weights = torch.tensor([[[0., 1., .6, .3]]] * 1).expand_as(labels)
        expected = torch.tensor([[[False, True, False, False],
                                  [False, True, True, True],
                                  [False, True, True, True]]])
        actual = first_error_mask(proposals, labels, weights)
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(actual, loop_selection(proposals, labels, weights)))

    def test_padding_invalid_holes_and_eos_never_reopen_selection(self):
        labels = torch.tensor([[[1, 2, 9, 4, 5], [1, 2, 3, 4, 5],
                                [9, 2, 3, 4, 5], [1, 9, 3, 4, 5],
                                [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]])
        proposals = labels.clone()
        proposals[0, 3, 1] = 8  # The wrong proposal at canonical EOS still learns.
        weights = torch.tensor([[[0., 1., .6, .3, .2], [0., 1., 0., .3, .2],
                                 [0., 1., .6, .3, .2], [0., 1., .6, .3, .2],
                                 [0., 0., .6, .3, .2], [0., 1., .6, 0., 0.]]])
        expected = torch.tensor([[[0, 1, 1, 0, 0], [0, 1, 0, 0, 0],
                                  [0, 0, 0, 0, 0], [0, 1, 0, 0, 0],
                                  [0, 0, 0, 0, 0], [0, 1, 1, 0, 0]]], dtype=torch.bool)
        actual = first_error_mask(proposals, labels, weights, (9,))
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(actual, loop_selection(proposals, labels, weights, (9,))))

    def test_predicted_eos_error_receives_loss_and_stops_at_that_mismatch(self):
        labels = torch.tensor([[[1, 2, 3, 4]]])
        predictions = torch.tensor([[[1, 2, 9, 4]]])
        weights = torch.tensor([[[0., 1., .6, .3]]])
        self.assertEqual(first_error_mask(predictions, labels, weights, (9,)).tolist(),
                         [[[False, True, True, False]]])

    def test_changed_current_predictions_recompute_mask_without_cache(self):
        labels = torch.tensor([[[0, 1, 2, 3]]])
        weights = torch.tensor([[[0., 1., .6, .3]]])
        predictions = labels.clone()
        predictions[..., 1] = 0
        first = first_error_mask(predictions, labels, weights)
        predictions.copy_(labels)
        second = first_error_mask(predictions, labels, weights)
        predictions[..., 2] = 0
        third = first_error_mask(predictions, labels, weights)
        self.assertEqual([int(mask.sum()) for mask in (first, second, third)], [1, 3, 2])

    def test_mask_and_statistics_detach_every_input(self):
        labels = torch.tensor([[[0, 1, 2, 3]]])
        logits = torch.randn(1, 1, 4, 5, requires_grad=True)
        weights = torch.tensor([[[0., 1., .6, .3]]], requires_grad=True)
        predictions = logits.argmax(-1)
        selected = first_error_mask(predictions, labels, weights)
        statistics = selection_statistics(predictions, labels, weights, selected)
        self.assertEqual(selected.dtype, torch.bool)
        for tensor in (selected, statistics):
            self.assertFalse(tensor.requires_grad)
            self.assertIsNone(tensor.grad_fn)
        self.assertIsNone(logits.grad)
        self.assertIsNone(weights.grad)

    def test_first_error_histogram_counts_blocks_once_and_ignores_post_boundary_errors(self):
        labels = torch.tensor([[[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3],
                                [0, 1, 2, 3], [9, 1, 2, 3]]])
        predictions = torch.tensor([[[0, 4, 2, 3], [0, 1, 2, 4], [0, 1, 2, 3],
                                     [0, 1, 4, 3], [9, 4, 2, 3]]])
        weights = torch.tensor([[[0., 1., .5, .25], [0., 1., .5, .25],
                                 [0., 1., .5, .25], [0., 1., 0., .25],
                                 [0., 1., .5, .25]]])
        selection = first_error_mask(predictions, labels, weights, (9,))
        actual = selection_statistics(predictions, labels, weights, selection, (9,))
        expected = torch.tensor([14., 8., float(weights.sum()),
                                 float((weights * selection).sum()), 2., 1., 1., 0., 1.])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_randomized_multiple_response_blocks_match_scalar_oracle(self):
        generator = torch.Generator().manual_seed(17)
        for _ in range(40):
            labels = torch.randint(0, 7, (3, 5, 8), generator=generator)
            predictions = labels.clone()
            wrong = torch.rand(labels.shape, generator=generator) < .3
            predictions[wrong] = (predictions[wrong] + 1) % 7
            weights = (torch.rand(labels.shape, generator=generator) > .15).float()
            # Deliberately positive seeds: selection must always exclude them.
            actual = first_error_mask(predictions, labels, weights, (5, 6))
            expected = loop_selection(predictions, labels, weights, (5, 6))
            self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
