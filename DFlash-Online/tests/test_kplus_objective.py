"""K+ CPU arithmetic qualification against an independent explicit-weight oracle.

Tolerances were declared before fixture execution in
research/dflash_kplus_20261103/reviews/OBJECTIVE_TOLERANCES_PREDECLARED.json.
No optimizer step, GPU execution or efficacy measurement occurs here.
"""

import unittest
from unittest import mock

import torch
import torch.nn.functional as F

from online_feedback.first_error import first_error_mask
from online_feedback.kplus import (
    first_error_groups, group_terms, install_kplus, kplus_contract,
    redistribute_response,
)
from online_feedback.objective import TargetObjectiveModel
from specforge.core.chunking import checkpointed_chunk_reduce
from test_objective_integration import tiny_author


FP32_TOLERANCE = {"rtol": 2e-6, "atol": 2e-7}
BF16_GRADIENT_TOLERANCE = {"rtol": 0.02, "atol": 2e-4}
BF16_RELATIVE_L2_MAX = 0.01


def explicit_weights(proposals, labels, weights, eos=()):
    """Scalar block traversal and direct w*(.5+.5*v/vbar) reference."""
    selected = torch.zeros_like(weights)
    errors = []
    for block in range(labels.shape[1]):
        if int(labels[0, block, 0]) in eos:
            continue
        for offset in range(1, labels.shape[2]):
            if float(weights[0, block, offset]) <= 0:
                break
            selected[0, block, offset] = weights[0, block, offset]
            terminal = int(labels[0, block, offset]) in eos
            if proposals[0, block, offset] != labels[0, block, offset]:
                following = offset + 1
                has_correct_successor = (
                    not terminal and following < labels.shape[2]
                    and float(weights[0, block, following]) > 0
                    and proposals[0, block, following] == labels[0, block, following]
                )
                errors.append((block, offset, 2 if has_correct_successor else 1))
                break
            if terminal:
                break
    modified = selected.clone()
    if {row[2] for row in errors} == {1, 2}:
        mass = sum(float(weights[0, block, offset]) for block, offset, _ in errors)
        vbar = sum(float(weights[0, block, offset]) * priority
                   for block, offset, priority in errors) / mass
        for block, offset, priority in errors:
            modified[0, block, offset] *= 0.5 + 0.5 * priority / vbar
    return selected, modified, errors


def mixed_logits(dtype=torch.bfloat16):
    """Seventeen blocks: low-priority chunk, high-priority chunk, all-correct."""
    generator = torch.Generator().manual_seed(20261103)
    logits = torch.randn(1, 17, 16, 31, generator=generator).to(dtype)
    teacher = torch.randn(logits.shape, generator=generator).to(dtype)
    labels = logits.argmax(-1)
    offsets = torch.arange(16).float()
    weights = torch.exp(-(offsets - 1).clamp_min(0) / 7).expand(1, 17, 16).clone()
    weights[..., 0] = 0
    for block in range(16):
        offset = 1 + block % 5
        labels[0, block, offset] = (labels[0, block, offset] + 1) % 31
        weights[0, block] *= 0.15 + 0.06 * block
        if block < 8:
            labels[0, block, offset + 1] = (labels[0, block, offset + 1] + 1) % 31
    return logits, teacher, labels, weights


def ordinary_kl(logits, teacher):
    return F.kl_div(
        F.log_softmax(logits, dim=-1), F.softmax(teacher.detach(), dim=-1),
        reduction="none",
    ).sum(-1)


class KPlusObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized(), "CPU fixture initialized CUDA")

    def model(self, installed=True):
        model = TargetObjectiveModel(
            tiny_author(chunk_blocks=8), "osd_first_error", loss_precision="bfloat16",
        )
        model.block_size = 16
        model.lm_head = torch.nn.Identity()
        if installed:
            self.assertEqual(install_kplus(model), kplus_contract())
        model.begin_event()
        return model

    def test_equal_error_weights_give_seven_sixths_and_five_sixths(self):
        proposals = torch.tensor([[[0, 9, 2, 3], [0, 9, 8, 3]]])
        labels = torch.tensor([[[0, 1, 2, 3], [0, 1, 2, 3]]])
        weights = torch.tensor([[[0., 1., .5, .25], [0., 1., .5, .25]]], dtype=torch.float64)
        token_loss = torch.tensor(
            [[[1., 3., 7., 9.], [1., 11., 13., 17.]]],
            dtype=torch.float64, requires_grad=True,
        )
        selected = first_error_mask(proposals, labels, weights)
        native = (token_loss * weights * selected).sum()
        terms = group_terms(token_loss, proposals, labels, weights, selected)
        numerator, metrics = redistribute_response(native, (weights * selected).sum(), terms)
        numerator.backward()
        self.assertAlmostEqual(float(metrics["kplus_coefficient1"]), 5 / 6, places=12)
        self.assertAlmostEqual(float(metrics["kplus_coefficient2"]), 7 / 6, places=12)
        torch.testing.assert_close(
            token_loss.grad[0, :, 1], torch.tensor([7 / 6, 5 / 6], dtype=torch.float64),
            rtol=1e-12, atol=1e-12,
        )
        self.assertEqual(float(metrics["kplus_first_error_weight_after"]), 2.)
        self.assertAlmostEqual(float(metrics["kplus_error_weight_moved_share"]), 1 / 12, places=12)
        self.assertTrue(all(not value.requires_grad for value in metrics.values()))

    def test_priority_never_crosses_gap_end_eos_or_anchor(self):
        labels = torch.tensor([[[0, 1, 2, 3, 4, 5]] * 7])
        proposals = labels.clone()
        weights = torch.ones_like(labels, dtype=torch.float32)
        weights[..., 0] = 0
        proposals[0, :, 1] = 9
        weights[0, 0, 2] = 0  # A later correct token cannot bridge this gap.
        weights[0, 1, 2:] = 0  # Last valid proposal.
        labels[0, 2, 1] = 30  # Wrong draft token at canonical EOS.
        labels[0, 3, 2] = 30  # Correct EOS immediately after the first error.
        proposals[0, 3, 2] = 30
        labels[0, 4, 0] = 30  # EOS seed excludes the whole block.
        proposals[0, 5] = labels[0, 5]
        proposals[0, 5, 0] = 9  # Anchor mismatch is never an error.
        proposals[0, 6] = labels[0, 6]
        proposals[0, 6, -1] = 9  # Last block slot has no successor.
        selection = first_error_mask(proposals, labels, weights, (30,))
        low, high = first_error_groups(proposals, labels, weights, selection, (30,))
        self.assertEqual(
            torch.nonzero(low, as_tuple=False).tolist(),
            [[0, 0, 1], [0, 1, 1], [0, 2, 1], [0, 6, 5]],
        )
        self.assertEqual(torch.nonzero(high, as_tuple=False).tolist(), [[0, 3, 1]])
        selected, modified, errors = explicit_weights(proposals, labels, weights, (30,))
        self.assertEqual(len(errors), 5)
        torch.testing.assert_close(selected > 0, selection)
        torch.testing.assert_close(modified.sum(), selected.sum(), **FP32_TOLERANCE)
        self.assertFalse(bool(selection[0, 3, 2]))  # Successor is only read, never selected.

    def test_mixed_unequal_mass_across_eight_block_chunks_matches_explicit_rows(self):
        logits, _, labels, weights = mixed_logits(torch.float64)
        proposals = logits.argmax(-1)
        for dtype, tolerance in ((torch.float64, {"rtol": 1e-12, "atol": 1e-12}),
                                 (torch.float32, FP32_TOLERANCE)):
            with self.subTest(dtype=dtype):
                local_weights = weights.to(dtype)
                rows = torch.linspace(.05, 9, labels.numel(), dtype=dtype)
                rows = rows.reshape_as(local_weights).requires_grad_()
                reference_rows = rows.detach().clone().requires_grad_()
                selected, modified, errors = explicit_weights(proposals, labels, local_weights)
                expected = (reference_rows * modified).sum() / (selected.sum() + 1e-6)
                expected.backward()

                def chunk(values, chosen, target, coefficients):
                    mask = first_error_mask(chosen, target, coefficients)
                    return ((values * coefficients * mask).sum(), (coefficients * mask).sum(),
                            *group_terms(values, chosen, target, coefficients, mask))

                pooled = checkpointed_chunk_reduce(
                    chunk, rows, proposals, labels, local_weights,
                    chunk_size=8, dim=1, checkpoint=True,
                )
                numerator, metrics = redistribute_response(pooled[0], pooled[1], pooled[2:])
                actual = numerator / (pooled[1] + 1e-6)
                actual.backward()
                torch.testing.assert_close(actual, expected, **tolerance)
                torch.testing.assert_close(rows.grad, reference_rows.grad, **tolerance)
                self.assertEqual(int(metrics["kplus_priority1_error_count"]), 8)
                self.assertEqual(int(metrics["kplus_priority2_error_count"]), 8)
                self.assertEqual(float(metrics["kplus_mixed_response"]), 1.)
                self.assertGreater(abs(float(numerator.detach() - pooled[0].detach())), 0.01)
                first_errors = torch.zeros_like(selected, dtype=torch.bool)
                for block, offset, _ in errors:
                    first_errors[0, block, offset] = True
                torch.testing.assert_close(modified[~first_errors], selected[~first_errors], rtol=0, atol=0)
                self.assertTrue(pooled[4].requires_grad and pooled[5].requires_grad)
                self.assertFalse(pooled[2].requires_grad or pooled[3].requires_grad)

    def test_bf16_ordinary_kl_loss_and_gradients_match_explicit_response_weights(self):
        model = self.model()
        logits, teacher, labels, weights = mixed_logits()
        teacher.requires_grad_()
        actual_logits = logits.clone().requires_grad_()
        reference_logits = logits.clone().requires_grad_()
        selected, modified, _ = explicit_weights(logits.argmax(-1), labels, weights)
        expected = (ordinary_kl(reference_logits, teacher) * modified).sum() / (selected.sum() + 1e-6)
        expected.backward()
        with mock.patch.object(model.lm_head, "forward", wraps=model.lm_head.forward) as head:
            terms = checkpointed_chunk_reduce(
                model._chunk_loss, actual_logits, None, labels, weights, teacher,
                chunk_size=8, dim=1, checkpoint=True,
            )
            self.assertEqual(head.call_count, 6)  # Exactly current/teacher for each of three chunks.
            numerator, metrics = model.first_error_redistribution.reduce_response(terms)
            actual = numerator / (terms[1] + 1e-6)
            actual.backward()
        torch.testing.assert_close(actual, expected, **FP32_TOLERANCE)
        torch.testing.assert_close(actual_logits.grad, reference_logits.grad, **BF16_GRADIENT_TOLERANCE)
        difference = (actual_logits.grad - reference_logits.grad).float()
        relative = torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(
            reference_logits.grad.float(),
        )
        self.assertLessEqual(float(relative), BF16_RELATIVE_L2_MAX)
        self.assertIsNone(teacher.grad)
        excluded_gradients = actual_logits.grad[selected == 0]
        self.assertTrue(torch.equal(excluded_gradients, torch.zeros_like(excluded_gradients)))
        self.assertEqual(terms[0].dtype, torch.float32)
        self.assertEqual(terms[7].dtype, torch.float32)
        self.assertTrue(terms[7].requires_grad and terms[8].requires_grad)
        self.assertTrue(all(not value.requires_grad for value in metrics.values()))

    def test_neutral_classes_return_same_native_numerator_and_exact_gradients(self):
        for case in ("no_errors", "only_priority1", "only_priority2", "empty"):
            with self.subTest(case=case):
                logits, teacher, labels, weights = mixed_logits()
                labels = logits.argmax(-1)
                if case.startswith("only_"):
                    labels[..., 1] = (labels[..., 1] + 1) % 31
                    if case == "only_priority1":
                        labels[..., 2] = (labels[..., 2] + 1) % 31
                if case == "empty":
                    weights.zero_()
                outputs = []
                for installed in (False, True):
                    model = self.model(installed)
                    leaf = logits.clone().requires_grad_()
                    terms = checkpointed_chunk_reduce(
                        model._chunk_loss, leaf, None, labels, weights, teacher,
                        chunk_size=8, dim=1, checkpoint=True,
                    )
                    numerator = terms[0]
                    if installed:
                        numerator, diagnostics = model.first_error_redistribution.reduce_response(terms)
                        self.assertIs(numerator, terms[0])
                        self.assertEqual(float(diagnostics["kplus_mixed_response"]), 0.)
                        self.assertEqual(float(diagnostics["kplus_error_weight_moved_share"]), 0.)
                    loss = numerator / (terms[1] + 1e-6)
                    loss.backward()
                    outputs.append((loss, leaf.grad))
                torch.testing.assert_close(outputs[0][0], outputs[1][0], rtol=0, atol=0)
                torch.testing.assert_close(outputs[0][1], outputs[1][1], rtol=0, atol=0)

    def test_full_forward_pools_once_and_diagnostics_do_not_count_backward_replay(self):
        model = self.model()
        logits, teacher, labels, weights = mixed_logits()
        logits.requires_grad_()
        blocks = labels.shape[1]
        anchors = torch.arange(blocks).reshape(1, -1) * 16
        keep = torch.ones_like(anchors, dtype=torch.bool)
        input_ids = labels.reshape(1, -1)
        blocks_result = (anchors, keep, logits.reshape(1, -1, 31))
        with mock.patch.object(model, "_forward_draft_blocks", return_value=blocks_result) as backbone:
            with mock.patch.object(model, "_block_labels_and_weights", return_value=(labels, weights, labels)):
                with mock.patch.object(model, "_aligned_target_hidden", return_value=teacher):
                    rng = torch.get_rng_state().clone()
                    loss, _, metrics = model(input_ids, torch.zeros(1), torch.ones_like(input_ids), teacher)
                    observations = {name: value.clone() for name, value in metrics["sum_metrics"].items()}
                    loss.backward()
        self.assertEqual(backbone.call_count, 1)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        selected, modified, _ = explicit_weights(logits.detach().argmax(-1), labels, weights)
        expected = (ordinary_kl(logits.detach(), teacher) * modified).sum() / (selected.sum() + 1e-6)
        torch.testing.assert_close(loss, expected, **FP32_TOLERANCE)
        torch.testing.assert_close(metrics["loss_terms"][1], selected.sum() + 1e-6, **FP32_TOLERANCE)
        self.assertEqual(float(metrics["sum_metrics"]["kplus_response_count"]), 1.)
        self.assertEqual(float(metrics["sum_metrics"]["kplus_priority1_error_count"]), 8.)
        self.assertEqual(float(metrics["sum_metrics"]["kplus_priority2_error_count"]), 8.)
        for name, value in observations.items():
            torch.testing.assert_close(metrics["sum_metrics"][name], value, rtol=0, atol=0)
        costs = model.end_event()
        self.assertGreater(costs["kplus_chunk_s"], 0.)
        self.assertGreater(costs["kplus_response_s"], 0.)

    def test_installation_and_response_scope_reject_scientific_mismatches(self):
        model = self.model(False)
        parameters = tuple(model.parameters())
        install_kplus(model)
        self.assertEqual(tuple(model.parameters()), parameters)
        with self.assertRaisesRegex(ValueError, "already installed"):
            install_kplus(model)
        with self.assertRaisesRegex(ValueError, "one response"):
            model(torch.zeros(2, 4, dtype=torch.long), torch.zeros(1), torch.zeros(2, 4))
        for objective, precision in (("osd", "bfloat16"), ("osd_first_error", "float32")):
            incompatible = TargetObjectiveModel(tiny_author(), objective, loss_precision=precision)
            with self.assertRaisesRegex(ValueError, "native BF16"):
                install_kplus(incompatible)


if __name__ == "__main__":
    unittest.main()
