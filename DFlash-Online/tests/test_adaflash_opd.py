"""Paper clipping, arithmetic, gradients and native OSD integration on CPU."""

import math
import json
import unittest
from unittest import mock

import torch

from online_feedback.adaflash_opd import (
    CE_FRACTION, RKL_FRACTION, ENTRY_CAP,
    adaflash_opd_contract, chunk_terms, install_adaflash_opd,
    numerical_qualification, paper_position_loss,
)
from online_feedback.objective import TargetObjectiveModel
from specforge.core.chunking import checkpointed_chunk_reduce
from test_objective_integration import tiny_author, parameter_gradients
from test_osd_objective import osd_batch


def direct_rows(logits, teacher):
    """Qualification oracle separate from the implementation's clamp."""
    log_q = logits.log_softmax(-1)
    log_p = teacher.detach().log_softmax(-1)
    entries = log_q.exp() * (log_q - log_p)
    entries = torch.where(entries > .01, entries.new_tensor(.01), entries)
    ce = -torch.take_along_dim(log_q, teacher.detach().argmax(-1, keepdim=True), -1).squeeze(-1)
    return .8 * ce + .2 * entries.sum(-1)


class AdaFlashOPDTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized(), "CPU tests initialized CUDA")

    def model(self, *, chunk_blocks=8):
        model = TargetObjectiveModel(tiny_author(chunk_blocks=chunk_blocks), "osd",
                                     loss_precision="bfloat16")
        install_adaflash_opd(model)
        model.begin_event()
        return model

    def test_fp64_fixture_matches_analytic_values_and_gradient(self):
        q = torch.tensor([[.6, .3, .1]], dtype=torch.float64)
        p = torch.tensor([[.1, .4, .5]], dtype=torch.float64)
        logits = q.log().requires_grad_()
        teacher = p.log().requires_grad_()
        loss, ce, rkl, entries, target = paper_position_loss(logits, teacher)
        loss.sum().backward()
        self.assertEqual(target.item(), 2)
        torch.testing.assert_close(rkl, rkl.new_tensor([-.23724841297894436]), rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(loss, loss.new_tensor([1.7946183917994476]), rtol=1e-12, atol=1e-12)
        gradient = torch.tensor([[.4616698095574733, .2735739804316298, -.7352437899891032]],
                                dtype=torch.float64)
        torch.testing.assert_close(logits.grad, gradient, rtol=1e-12, atol=1e-12)
        # Clipping the final vocabulary sum is a different objective.
        wrong = CE_FRACTION * ce + RKL_FRACTION * entries.sum(-1).clamp_max(ENTRY_CAP)
        self.assertGreater(abs(float(loss.detach() - wrong.detach())), .04)
        self.assertIsNone(teacher.grad)
        # A capped entry's logit still affects other probabilities through softmax.
        self.assertNotEqual(float(logits.grad[0, 0]), 0)

    def test_unclipped_reverse_kl_matches_closed_form_gradient(self):
        logits = torch.tensor([[.01, .0, -.01]], dtype=torch.float64, requires_grad=True)
        teacher = torch.zeros_like(logits)
        loss, _, rkl, entries, target = paper_position_loss(logits, teacher)
        q = logits.softmax(-1).detach()
        log_ratio = logits.log_softmax(-1).detach() - teacher.log_softmax(-1)
        expected_gradient = .8 * (q - torch.nn.functional.one_hot(target, 3))
        expected_gradient += .2 * q * (log_ratio - (q * log_ratio).sum(-1, keepdim=True))
        self.assertFalse(bool((entries > ENTRY_CAP).any()))
        torch.testing.assert_close(rkl, (q * log_ratio).sum(-1), rtol=1e-12, atol=1e-12)
        loss.sum().backward()
        torch.testing.assert_close(logits.grad, expected_gradient, rtol=1e-12, atol=1e-12)

    def test_equal_distribution_has_zero_rkl_but_keeps_ce(self):
        logits = torch.tensor([[.4, -.1, .2]], dtype=torch.float64, requires_grad=True)
        loss, ce, rkl, _, _ = paper_position_loss(logits, logits.detach())
        self.assertEqual(float(rkl.detach()), 0)
        torch.testing.assert_close(loss, .8 * ce, rtol=0, atol=0)
        self.assertGreater(float(loss.detach()), 0)

    def test_bf16_row_arithmetic_and_fp32_weighting_are_exact(self):
        model = self.model()
        model.lm_head = torch.nn.Identity()
        logits = torch.tensor([[[[.4, .1, -.2], [-.2, .3, .7], [.5, -.5, .2]]]],
                              dtype=torch.bfloat16, requires_grad=True)
        teacher = torch.tensor([[[[1., -1., .2], [-.7, .2, 1.], [.8, -.3, .1]]]],
                               dtype=torch.bfloat16, requires_grad=True)
        labels = torch.tensor([[[0, 0, 2]]])
        weights = torch.tensor([[[0., 1., .3]]], dtype=torch.float32)
        reference_logits = logits.detach().clone().requires_grad_()
        expected_rows = direct_rows(reference_logits, teacher)
        expected = (expected_rows * weights).sum() / (weights.sum() + 1e-6)
        expected.backward()
        terms = model._chunk_loss(logits, None, labels, weights, teacher)
        actual = terms[0] / (terms[1] + 1e-6)
        actual.backward()
        self.assertEqual(expected_rows.dtype, torch.bfloat16)
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(logits.grad, reference_logits.grad, rtol=0, atol=0)
        self.assertIsNone(teacher.grad)
        self.assertEqual(int(terms[3]), 2)
        self.assertEqual(int(terms[4][4]), 6)
        self.assertEqual(int(terms[4][7]), 2)  # CE follows teacher argmax, not saved label.

    def test_zero_supervision_is_zero_and_does_not_resample(self):
        logits = torch.tensor([[[.6, .3, -.1], [.2, .4, -.8]]], requires_grad=True)
        teacher = -logits.detach()
        weights = torch.zeros(1, 2)
        terms = chunk_terms(logits, teacher, torch.zeros(1, 2, dtype=torch.long), weights)
        loss = terms[0] / (terms[1] + 1e-6)
        loss.backward()
        self.assertEqual(float(loss.detach()), 0)
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits)))
        self.assertTrue(torch.equal(terms[4], torch.zeros_like(terms[4])))

    def test_teacher_argmax_tie_is_deterministic_first_index(self):
        logits = torch.tensor([[.5, -.5, .2]], dtype=torch.float64)
        teacher = torch.tensor([[1., 1., -1.]], dtype=torch.float64)
        _, ce, _, _, labels = paper_position_loss(logits, teacher)
        self.assertEqual(labels.item(), 0)
        torch.testing.assert_close(ce, -logits.log_softmax(-1)[:, 0])

    def test_complete_response_matches_explicit_teacher_alignment_and_weights(self):
        model = self.model(chunk_blocks=64)
        batch = osd_batch(model)
        batch["loss_mask"][0, 6] = 0
        batch["target_last_hidden_states"].requires_grad_()
        anchors = torch.tensor([[4, 8, 10, 0]])
        keep = torch.tensor([[True, True, True, False]])
        inputs = {key: batch[key] for key in ("input_ids", "hidden_states", "loss_mask")}
        with mock.patch.object(model, "_sample_anchor_positions", return_value=(anchors, keep)):
            _, _, hidden = model._forward_draft_blocks(**inputs)
            hidden = hidden.reshape(1, 4, 4, 16)
            teacher = torch.empty_like(hidden)
            weights = torch.zeros(1, 4, 4)
            positions = []
            length = batch["input_ids"].shape[1]
            for block, anchor in enumerate(anchors[0].tolist()):
                for offset in range(4):
                    position = anchor + offset
                    predecessor = max(0, min(position, length - 1) - 1)
                    teacher[0, block, offset] = batch["target_last_hidden_states"][0, predecessor].detach()
                    if keep[0, block] and offset > 0 and position < length and batch["loss_mask"][0, position]:
                        weights[0, block, offset] = math.exp(-(offset - 1) / 2.0)
                        positions.append(position)
            logits = model.lm_head(hidden).to(torch.bfloat16)
            teacher_logits = model.lm_head(teacher).to(torch.bfloat16)
            expected = (direct_rows(logits, teacher_logits) * weights).sum() / (weights.sum() + 1e-6)
            expected.backward()
            expected_gradients = parameter_gradients(model)
            model.zero_grad(set_to_none=True)
            actual, _, metrics = model(**batch)
            original_diagnostics = {key: value.clone() for key, value in metrics["sum_metrics"].items()}
            actual.backward()

        self.assertEqual(positions, [5, 7, 9, 10, 11, 11])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for name, gradient in parameter_gradients(model).items():
            torch.testing.assert_close(gradient, expected_gradients[name], rtol=0, atol=0)
        self.assertEqual(int(metrics["sum_metrics"]["supervised_position_exposure"]), 6)
        self.assertEqual(int(metrics["sum_metrics"]["adaflash_response_count"]), 1)
        self.assertEqual(int(metrics["sum_metrics"]["adaflash_available_positions"]), 6)
        for name, value in original_diagnostics.items():
            torch.testing.assert_close(metrics["sum_metrics"][name], value, rtol=0, atol=0)
        self.assertIsNone(batch["target_last_hidden_states"].grad)
        self.assertIsNone(model.lm_head.weight.grad)
        self.assertIsNone(model.reference)

    def test_chunk_reduction_uses_one_response_denominator_and_counts_once(self):
        generator = torch.Generator().manual_seed(87)
        logits = torch.randn(1, 17, 4, 9, generator=generator, dtype=torch.float64)
        teacher = torch.randn(logits.shape, generator=generator, dtype=torch.float64)
        labels = teacher.argmax(-1)
        weights = torch.tensor([0., 1., .6, .2]).expand(1, 17, 4).clone()
        results = []
        for chunk_size in (1, 8, 64):
            leaf = logits.clone().requires_grad_()
            terms = checkpointed_chunk_reduce(chunk_terms, leaf, teacher, labels, weights,
                                              chunk_size=chunk_size, dim=1, checkpoint=True)
            loss = terms[0] / (terms[1] + 1e-6)
            loss.backward()
            results.append((loss.detach(), leaf.grad, terms[4]))
            self.assertEqual(int(terms[4][4]), 51 * 9)
            self.assertEqual(int(terms[4][6]), 51)
            self.assertTrue(torch.equal(leaf.grad[:, :, 0], torch.zeros_like(leaf.grad[:, :, 0])))
        for loss, gradient, diagnostics in results[1:]:
            torch.testing.assert_close(loss, results[0][0], rtol=2e-7, atol=2e-7)
            torch.testing.assert_close(gradient, results[0][1], rtol=2e-7, atol=2e-7)
            torch.testing.assert_close(diagnostics, results[0][2], rtol=2e-6, atol=2e-6)

    def test_all_valid_positions_get_loss_even_after_first_error(self):
        model = self.model()
        model.lm_head = torch.nn.Identity()
        logits = torch.zeros(1, 1, 4, 3, dtype=torch.bfloat16, requires_grad=True)
        teacher = torch.tensor([0., 1., 0.], dtype=torch.bfloat16).expand_as(logits)
        labels = teacher.argmax(-1)
        weights = torch.tensor([[[0., 1., .6, .2]]])
        terms = model._chunk_loss(logits, None, labels, weights, teacher)
        terms[0].backward()
        self.assertEqual(int(terms[3]), 3)
        for offset in (1, 2, 3):
            self.assertGreater(float(logits.grad[0, 0, offset].abs().sum()), 0)

    def test_installer_preserves_model_parameters_and_rejects_other_mechanisms(self):
        model = TargetObjectiveModel(tiny_author(), "osd", loss_precision="bfloat16")
        parameters = dict(model.named_parameters())
        before = {name: value.detach().clone() for name, value in parameters.items()}
        self.assertEqual(install_adaflash_opd(model), adaflash_opd_contract())
        for name, value in model.named_parameters():
            self.assertIs(value, parameters[name])
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        for objective, precision in (("osd_first_error", "bfloat16"), ("osd", "float32")):
            with self.assertRaisesRegex(ValueError, "ordinary all-position BF16"):
                install_adaflash_opd(TargetObjectiveModel(tiny_author(), objective, loss_precision=precision))
        with self.assertRaisesRegex(ValueError, "another objective"):
            install_adaflash_opd(model)

    def test_installation_and_qualification_consume_no_rng(self):
        state = torch.get_rng_state().clone()
        receipt = numerical_qualification("cpu")
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertEqual(receipt["status"], "passed")
        self.assertEqual(receipt["work"]["optimizer_steps"], 0)
        self.assertEqual(receipt["work"]["model_forwards"], 0)
        self.assertEqual(len(receipt["cases"]), 6)
        self.assertTrue(any(case["negative_entries"] for case in receipt["cases"]))
        self.assertTrue(any(case["clipped_entries"] for case in receipt["cases"]))

    def test_failed_qualification_returns_evidence_before_caller_stops(self):
        original = paper_position_loss

        def wrong_loss(*args):
            result = original(*args)
            return (result[0] + 1, *result[1:])

        with mock.patch("online_feedback.adaflash_opd.paper_position_loss", side_effect=wrong_loss):
            receipt = numerical_qualification("cpu")
        self.assertEqual(receipt["status"], "failed")
        self.assertEqual(len(receipt["cases"]), 1)
        self.assertFalse(receipt["cases"][0]["value_matches"])
        self.assertEqual(receipt["work"]["synthetic_loss_forwards"], 2)
        self.assertEqual(receipt["work"]["synthetic_loss_backwards"], 2)
        json.dumps(receipt, allow_nan=False)

    def test_nonfinite_qualification_is_explicit_and_serializable(self):
        original = paper_position_loss

        def nonfinite_loss(*args):
            result = original(*args)
            return (result[0] * float("nan"), *result[1:])

        with mock.patch("online_feedback.adaflash_opd.paper_position_loss", side_effect=nonfinite_loss):
            receipt = numerical_qualification("cpu")
        self.assertEqual(receipt["status"], "failed")
        self.assertFalse(receipt["cases"][0]["finite"])
        self.assertEqual(receipt["cases"][0]["actual_loss"], "nan")
        json.dumps(receipt, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
