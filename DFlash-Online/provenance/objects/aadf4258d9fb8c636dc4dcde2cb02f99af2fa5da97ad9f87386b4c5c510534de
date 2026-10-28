"""Independent probability, gradient and complete-block checks for C/K/A."""

import copy
import unittest
from unittest import mock

import torch
import torch.nn.functional as F

from online_feedback.objective import TargetObjectiveModel
from test_first_error_mask import loop_selection
from test_objective_integration import tiny_author
from test_osd_objective import osd_batch


NEW_ARMS = ("osd_ce", "osd_first_error", "osd_ce_first_error")
SELECTED_ARMS = ("osd_first_error", "osd_ce_first_error")
CE_ARMS = ("osd_ce", "osd_ce_first_error")


def probability_oracle(logits, teacher, labels, weights, arm, eos=()):
    """BF16 probabilities followed by FP32 weighted response sums."""
    selection = (loop_selection(logits.detach().argmax(-1), labels, weights, eos)
                 if arm in SELECTED_ARMS else torch.ones_like(labels, dtype=torch.bool))
    log_probability = F.log_softmax(logits, dim=-1)
    if arm in CE_ARMS:
        rows = -log_probability.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    else:
        probability = F.softmax(teacher.detach(), dim=-1)
        rows = F.kl_div(log_probability, probability, reduction="none").sum(-1)
    effective_weights = weights * selection
    numerator = (rows.float() * effective_weights).sum()
    denominator = effective_weights.sum() + 1e-6
    return numerator / denominator, numerator, denominator, selection


def logit_fixture():
    generator = torch.Generator().manual_seed(17)
    logits = torch.randn(1, 3, 4, 7, generator=generator, dtype=torch.bfloat16)
    labels = logits.argmax(-1)
    labels[0, 0, 1] = (labels[0, 0, 1] + 1) % 7
    labels[0, 1, 3] = (labels[0, 1, 3] + 1) % 7
    teacher = torch.randn(logits.shape, generator=generator, dtype=torch.bfloat16)
    weights = torch.tensor([[[0., 1., .6, .3], [0., .2, .1, .05], [0., 1., 0., 0.]]])
    return logits, teacher, labels, weights


class FirstErrorObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized(), "CPU test initialized CUDA")

    def model(self, arm):
        model = TargetObjectiveModel(tiny_author(), arm, loss_precision="bfloat16")
        model.lm_head = torch.nn.Identity()
        return model

    def test_selected_and_full_losses_match_independent_bf16_values_and_direct_gradients(self):
        for arm in NEW_ARMS:
            with self.subTest(arm=arm):
                model = self.model(arm)
                logits, teacher, labels, weights = logit_fixture()
                teacher.requires_grad_()
                actual_logits = logits.clone().requires_grad_()
                expected_logits = logits.clone().requires_grad_()
                expected, numerator, denominator, selection = probability_oracle(
                    expected_logits, teacher, labels, weights, arm)
                expected.backward()
                terms = model._chunk_loss(actual_logits, None, labels, weights, teacher)
                actual = terms[0] / (terms[1] + 1e-6)
                actual.backward()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(terms[0], numerator, rtol=0, atol=0)
                torch.testing.assert_close(terms[1] + 1e-6, denominator, rtol=0, atol=0)
                torch.testing.assert_close(actual_logits.grad, expected_logits.grad, rtol=0, atol=0)
                self.assertEqual(terms[0].dtype, torch.float32)
                self.assertIsNone(teacher.grad)
                excluded = ~(selection & (weights > 0))
                self.assertTrue(torch.equal(actual_logits.grad[excluded],
                                            torch.zeros_like(actual_logits.grad[excluded])))
                if arm in SELECTED_ARMS:
                    self.assertGreater(float(actual_logits.grad[0, 0, 1].abs().sum()), 0)

    def test_all_valid_positions_selected_recovers_full_kl_or_ce_and_gradients_exactly(self):
        for full, selected in (("osd", "osd_first_error"), ("osd_ce", "osd_ce_first_error")):
            with self.subTest(full=full, selected=selected):
                logits, teacher, _, weights = logit_fixture()
                labels = logits.argmax(-1)
                values = []
                for arm in (full, selected):
                    leaf = logits.clone().requires_grad_()
                    terms = self.model(arm)._chunk_loss(leaf, None, labels, weights, teacher)
                    loss = terms[0] / (terms[1] + 1e-6)
                    loss.backward()
                    values.append((loss, leaf.grad))
                torch.testing.assert_close(values[0][0], values[1][0], rtol=0, atol=0)
                torch.testing.assert_close(values[0][1], values[1][1], rtol=0, atol=0)

    def test_ce_uses_saved_canonical_token_and_qualified_bf16_log_probability(self):
        logits, teacher, labels, weights = logit_fixture()
        model = self.model("osd_ce")
        actual = model._chunk_loss(logits, None, labels, weights, teacher)[0]
        bf16 = -(logits.log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
                 * weights).sum()
        legacy = (F.cross_entropy(logits.float().reshape(-1, 7), labels.reshape(-1),
                                  reduction="none").reshape_as(labels) * weights).sum()
        torch.testing.assert_close(actual, bf16, rtol=0, atol=0)
        self.assertGreater(abs(float(actual - legacy)), 1e-4)
        altered_teacher = teacher.flip(-1) * 40
        unchanged = model._chunk_loss(logits, None, labels, weights, altered_teacher)[0]
        torch.testing.assert_close(actual, unchanged, rtol=0, atol=0)
        changed_labels = (labels + 1) % 7
        changed = model._chunk_loss(logits, None, changed_labels, weights, teacher)[0]
        self.assertNotEqual(float(actual), float(changed))

    def test_zero_available_weight_is_finite_zero_loss_and_zero_direct_gradients(self):
        for arm in NEW_ARMS:
            with self.subTest(arm=arm):
                logits, teacher, labels, weights = logit_fixture()
                logits.requires_grad_()
                terms = self.model(arm)._chunk_loss(logits, None, labels, weights * 0, teacher)
                loss = terms[0] / (terms[1] + 1e-6)
                loss.backward()
                self.assertEqual(float(loss.detach()), 0)
                self.assertEqual(int(terms[3]), 0)
                self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits)))

    def test_each_presentation_uses_its_current_training_logits(self):
        for arm in SELECTED_ARMS:
            with self.subTest(arm=arm):
                model = self.model(arm)
                logits, teacher, labels, weights = logit_fixture()
                first = model._chunk_loss(logits, None, labels, weights, teacher)
                improved = logits.clone()
                improved.scatter_(-1, labels.unsqueeze(-1), 10)
                second = model._chunk_loss(improved, None, labels, weights, teacher)
                self.assertEqual(int(first[3]), 5)
                self.assertEqual(int(second[3]), 7)
                self.assertIsNone(model.reference)

    def test_full_block_attention_inputs_sampling_and_rng_are_identical_across_four_arms(self):
        author = tiny_author()
        captured = []
        for arm in ("osd", *NEW_ARMS):
            model = TargetObjectiveModel(copy.deepcopy(author), arm, loss_precision="bfloat16")
            model.begin_event()
            torch.manual_seed(17)
            with mock.patch.object(model.draft_model, "forward", wraps=model.draft_model.forward) as forward:
                loss, _, _ = model(**osd_batch(model))
                loss.backward()
            self.assertEqual(forward.call_count, 1)
            captured.append((forward.call_args.kwargs, torch.get_rng_state()))
            self.assertIsNone(model.reference)
            costs = model.end_event()
            for name in ("reference_setup_s", "reference_forward_s", "reference_head_s"):
                self.assertEqual(costs[name], 0)
        for kwargs, rng in captured[1:]:
            self.assertEqual(kwargs.keys(), captured[0][0].keys())
            for key in kwargs:
                torch.testing.assert_close(kwargs[key], captured[0][0][key], rtol=0, atol=0)
            self.assertTrue(torch.equal(rng, captured[0][1]))

    def test_response_normalization_sums_blocks_before_single_epsilon_and_backward(self):
        anchors = torch.tensor([[4, 8, 10, 0]])
        keep = torch.tensor([[True, True, True, False]])
        for arm in NEW_ARMS:
            with self.subTest(arm=arm):
                author = tiny_author(chunk_blocks=1)
                model = TargetObjectiveModel(author, arm, loss_precision="bfloat16")
                batch = osd_batch(model)
                generator = torch.Generator().manual_seed(106)
                hidden = torch.randn(1, 4, 4, 16, generator=generator,
                                     dtype=torch.bfloat16).requires_grad_()
                expected_hidden = hidden.detach().clone().requires_grad_()
                labels = torch.zeros(1, 4, 4, dtype=torch.long)
                weights = torch.zeros(1, 4, 4)
                teachers = torch.empty_like(hidden)
                for block, anchor in enumerate(anchors[0].tolist()):
                    for offset in range(4):
                        position = min(anchor + offset, 11)
                        labels[0, block, offset] = batch["input_ids"][0, position]
                        teachers[0, block, offset] = batch["target_last_hidden_states"][0, max(position - 1, 0)]
                        if keep[0, block] and offset and anchor + offset < 12:
                            weights[0, block, offset] = torch.exp(torch.tensor(-(offset - 1) / 2.))
                expected, numerator, denominator, selected = probability_oracle(
                    model.lm_head(expected_hidden), model.lm_head(teachers), labels, weights, arm)
                expected.backward()
                with mock.patch.object(model, "_forward_draft_blocks",
                                       return_value=(anchors, keep, hidden.reshape(1, 16, 16))):
                    actual, _, metrics = model(**batch)
                    before = {key: value.clone() for key, value in metrics["sum_metrics"].items()}
                    actual.backward()
                # FP32 chunk reduction may reorder additions; BF16 rows and gradients remain exact here.
                torch.testing.assert_close(actual, expected, rtol=1e-7, atol=1e-7)
                torch.testing.assert_close(metrics["loss_terms"][0], numerator, rtol=1e-7, atol=1e-7)
                torch.testing.assert_close(metrics["loss_terms"][1], denominator, rtol=1e-7, atol=1e-7)
                torch.testing.assert_close(hidden.grad, expected_hidden.grad, rtol=0, atol=0)
                self.assertEqual(int(metrics["sum_metrics"]["available_position_occurrences"]), 7)
                self.assertEqual(int(metrics["sum_metrics"]["selected_position_occurrences"]),
                                 int((selected & (weights > 0)).sum()))
                for name, old in before.items():
                    torch.testing.assert_close(metrics["sum_metrics"][name], old, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
