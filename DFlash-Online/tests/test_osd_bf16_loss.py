"""BF16 probability/KL rows with FP32 position weighting and reduction."""

import unittest
from unittest import mock

import torch
import torch.nn.functional as F

from online_feedback.objective import TargetObjectiveModel
from test_objective_integration import tiny_author, parameter_gradients
from test_osd_objective import osd_batch


def reference_rows(model, draft_hidden, teacher_hidden, dtype):
    """Reference head and KL calculation, independent of the objective wrapper."""
    draft_logits = F.linear(draft_hidden, model.lm_head.weight.detach()).to(dtype)
    with torch.no_grad():
        teacher_logits = F.linear(teacher_hidden, model.lm_head.weight.detach()).to(dtype)
        teacher_probabilities = torch.softmax(teacher_logits, dim=-1)
    log_draft = torch.log_softmax(draft_logits, dim=-1)
    rows = F.kl_div(log_draft, teacher_probabilities, reduction="none").sum(dim=-1)
    return rows, teacher_probabilities, log_draft


class BF16OSDLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_bf16_rows_and_fp32_reduction_match_values_and_gradients(self):
        model = TargetObjectiveModel(tiny_author(), "osd", loss_precision="bfloat16")
        generator = torch.Generator().manual_seed(117)
        hidden = torch.randn(1, 2, 4, 16, generator=generator, dtype=torch.bfloat16).requires_grad_()
        teacher = torch.randn(1, 2, 4, 16, generator=generator, dtype=torch.bfloat16).requires_grad_()
        reference_hidden = hidden.detach().clone().requires_grad_()
        labels = torch.tensor([[[0, 7, 8, 9], [0, 10, 11, 12]]])
        weights = torch.tensor([[[0., 1., 0.60653067, 0.36787945], [0., 1., 0.60653067, 0.]]])

        rows, teacher_probability, log_draft = reference_rows(
            model, reference_hidden, teacher, torch.bfloat16,
        )
        self.assertEqual(teacher_probability.dtype, torch.bfloat16)
        self.assertEqual(log_draft.dtype, torch.bfloat16)
        self.assertEqual(rows.dtype, torch.bfloat16)
        expected = (rows.float() * weights).sum() / (weights.sum() + 1e-6)
        expected.backward()

        numerator, denominator, _, positions = model._chunk_loss(hidden, None, labels, weights, teacher)
        actual = numerator / (denominator + 1e-6)
        actual.backward()
        self.assertEqual(numerator.dtype, torch.float32)
        self.assertEqual(actual.dtype, torch.float32)
        self.assertEqual(int(positions), 5)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(hidden.grad, reference_hidden.grad, rtol=0, atol=0)
        self.assertIsNone(teacher.grad)
        self.assertIsNone(model.lm_head.weight.grad)
        self.assertTrue(torch.equal(hidden.grad[weights == 0], torch.zeros_like(hidden.grad[weights == 0])))

        # The historical/default FP32 objective remains different. A silent
        # cast of probability/KL rows to FP32 would fail the comparison above.
        fp32_rows, _, _ = reference_rows(model, hidden.detach(), teacher.detach(), torch.float32)
        fp32_loss = (fp32_rows * weights).sum() / (weights.sum() + 1e-6)
        self.assertGreater(abs(float(actual.detach() - fp32_loss)), 1e-5)
        self.assertEqual(TargetObjectiveModel(tiny_author(), "osd").loss_dtype, torch.float32)

    def test_full_blocks_preserve_teacher_alignment_masks_and_bf16_gradients(self):
        model = TargetObjectiveModel(tiny_author(chunk_blocks=64), "osd", loss_precision="bfloat16")
        batch = osd_batch(model)
        batch["loss_mask"][0, 6] = 0
        batch["target_last_hidden_states"].requires_grad_(True)
        anchors = torch.tensor([[4, 8, 10, 0]])
        keep = torch.tensor([[True, True, True, False]])
        inputs = {key: batch[key] for key in ("input_ids", "hidden_states", "loss_mask")}
        with mock.patch.object(model, "_sample_anchor_positions", return_value=(anchors, keep)):
            _, _, hidden = model._forward_draft_blocks(**inputs)
            hidden = hidden.reshape(1, 4, 4, 16)
            teacher_rows = torch.empty_like(hidden)
            weights = torch.zeros(1, 4, 4, dtype=torch.float32)
            positions = []
            length = batch["input_ids"].shape[1]
            for block, anchor in enumerate(anchors[0].tolist()):
                for offset in range(4):
                    position = anchor + offset
                    predecessor = max(0, min(position, length - 1) - 1)
                    teacher_rows[0, block, offset] = batch["target_last_hidden_states"][0, predecessor].detach()
                    if keep[0, block] and offset > 0 and position < length and batch["loss_mask"][0, position]:
                        weights[0, block, offset] = torch.exp(torch.tensor(-(offset - 1) / 2.0))
                        positions.append(position)
            rows, _, _ = reference_rows(model, hidden, teacher_rows, torch.bfloat16)
            numerator = (rows.float() * weights).sum()
            denominator = weights.sum() + 1e-6
            expected = numerator / denominator
            expected.backward()
            expected_gradients = parameter_gradients(model)
            model.zero_grad(set_to_none=True)
            actual, _, metrics = model(**batch)
            actual.backward()

        self.assertEqual(positions, [5, 7, 9, 10, 11, 11])
        self.assertEqual(int(metrics["sum_metrics"]["supervised_position_exposure"]), 6)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(metrics["loss_terms"][0], numerator, rtol=0, atol=0)
        torch.testing.assert_close(metrics["loss_terms"][1], denominator, rtol=0, atol=0)
        for name, gradient in parameter_gradients(model).items():
            torch.testing.assert_close(gradient, expected_gradients[name], rtol=0, atol=0)
        self.assertIsNone(batch["target_last_hidden_states"].grad)
        self.assertIsNone(model.lm_head.weight.grad)


if __name__ == "__main__":
    unittest.main()
