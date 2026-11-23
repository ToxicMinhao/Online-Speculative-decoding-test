"""CPU checks of block alignment and native sampling; no model weights are loaded."""

import math
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import Mock

import torch
from specforge.algorithms.common.dflash_family_model import OnlineDFlashModel

from graduate.batches import PreparedFeedback
from graduate.model import DraftModel


class DraftBlockTests(unittest.TestCase):
    def setUp(self):
        self.feedback = PreparedFeedback(
            record_id="alignment",
            input_ids=torch.arange(100, 120).reshape(1, 20),
            loss_mask=torch.tensor([[0.0] * 3 + [1.0] * 17]),
            hidden_states=torch.arange(80, dtype=torch.bfloat16).reshape(1, 20, 4),
            target_last_hidden_states=torch.tensor(
                [[[row * 10, row * 10 + 1] for row in range(20)]], dtype=torch.bfloat16),
            stop_token_ids=(119,),
        )
        self.anchors = torch.tensor([[0, 3, 18, 0]])
        self.keep = torch.tensor([[True, True, True, False]])
        self.native_hidden = torch.arange(128, dtype=torch.bfloat16).reshape(1, 64, 2)
        self.native_hidden.requires_grad_()
        self.wrapper = SimpleNamespace(
            draft_model=SimpleNamespace(device=torch.device("cpu")),
            block_size=16, loss_decay_gamma=7.0,
            _forward_draft_blocks=Mock(return_value=(self.anchors, self.keep, self.native_hidden)),
        )
        self.wrapper._aligned_target_hidden = MethodType(
            OnlineDFlashModel._aligned_target_hidden, self.wrapper)

    def test_labels_teacher_predecessors_and_weights_match_scalar_positions(self):
        blocks = DraftModel.forward(self.wrapper, self.feedback)
        expected_labels, expected_teacher, expected_weights = [], [], []
        for anchor, keep in zip(self.anchors[0].tolist(), self.keep[0].tolist()):
            labels, teacher, weights = [], [], []
            for offset in range(16):
                position = anchor + offset
                safe_position = min(position, 19)
                predecessor = max(safe_position - 1, 0)
                labels.append(100 + safe_position)
                teacher.append([predecessor * 10, predecessor * 10 + 1])
                supervised = keep and offset > 0 and 3 <= position < 20
                weights.append(math.exp(-(offset - 1) / 7) if supervised else 0)
            expected_labels.append(labels)
            expected_teacher.append(teacher)
            expected_weights.append(weights)
        torch.testing.assert_close(blocks.labels, torch.tensor([expected_labels]))
        torch.testing.assert_close(blocks.teacher_hidden,
                                   torch.tensor([expected_teacher], dtype=torch.bfloat16))
        torch.testing.assert_close(blocks.weights, torch.tensor([expected_weights]),
                                   rtol=1e-6, atol=1e-7)
        self.assertEqual(blocks.weights.dtype, torch.float32)
        self.assertEqual(blocks.hidden.shape, (1, 4, 16, 2))
        self.assertIs(blocks.anchors, self.anchors)
        # The final EOS token is supervised; slots beyond it and unused blocks are not.
        self.assertEqual(blocks.labels[0, 2, 1].item(), 119)
        self.assertEqual(blocks.weights[0, 2, 1].item(), 1.0)
        self.assertEqual(blocks.weights[0, 2, 2:].count_nonzero().item(), 0)
        self.assertEqual(blocks.weights[0, 3].count_nonzero().item(), 0)

    def test_forward_keeps_native_gradients_and_calls_the_sampler_path_once(self):
        rng_before = torch.get_rng_state()
        blocks = DraftModel.forward(self.wrapper, self.feedback)
        self.wrapper._forward_draft_blocks.assert_called_once()
        inputs, features, mask, maximum = self.wrapper._forward_draft_blocks.call_args.args
        torch.testing.assert_close(inputs, self.feedback.input_ids)
        torch.testing.assert_close(features, self.feedback.hidden_states)
        torch.testing.assert_close(mask, self.feedback.loss_mask)
        self.assertEqual(maximum, 16)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng_before))
        blocks.hidden.sum().backward()
        torch.testing.assert_close(self.native_hidden.grad, torch.ones_like(self.native_hidden))
        self.assertFalse(blocks.teacher_hidden.requires_grad)


class NativeAnchorSamplerTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        self.sampler = MethodType(OnlineDFlashModel._sample_anchor_positions,
                                  SimpleNamespace(num_anchors=512))

    def test_seed_reproducibility_512_limit_sorted_starts_and_overlapping_blocks(self):
        mask = torch.tensor([[0.0] * 50 + [1.0] * 750])
        torch.manual_seed(17)
        anchors, keep = self.sampler(800, mask, torch.device("cpu"), max_valid_anchors=749)
        torch.manual_seed(17)
        repeated, repeated_keep = self.sampler(800, mask, torch.device("cpu"), max_valid_anchors=749)
        self.assertTrue(torch.equal(anchors, repeated))
        self.assertTrue(torch.equal(keep, repeated_keep))
        self.assertEqual(anchors.shape, (1, 512))
        self.assertTrue(keep.all().item())
        starts = anchors[0].tolist()
        self.assertEqual(starts, sorted(set(starts)))
        self.assertGreaterEqual(min(starts), 50)
        self.assertLessEqual(max(starts), 798)
        self.assertTrue(any(right - left < 16 for left, right in zip(starts, starts[1:])))
        torch.manual_seed(18)
        changed, _ = self.sampler(800, mask, torch.device("cpu"), max_valid_anchors=749)
        self.assertFalse(torch.equal(anchors, changed))

    def test_all_eligible_anchors_still_consume_full_candidate_rng_draw(self):
        mask = torch.tensor([[0.0] * 16 + [1.0] * 4])
        torch.manual_seed(17)
        anchors, keep = self.sampler(20, mask, torch.device("cpu"), max_valid_anchors=3)
        after_sampling = torch.get_rng_state()
        self.assertEqual(anchors.tolist(), [[16, 17, 18]])
        self.assertTrue(keep.all().item())
        torch.manual_seed(17)
        torch.rand((1, 19))  # Native sampling draws over all L-1 slots, including prompt.
        self.assertTrue(torch.equal(torch.get_rng_state(), after_sampling))


if __name__ == "__main__":
    unittest.main()
