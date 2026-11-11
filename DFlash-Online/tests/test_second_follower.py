"""CPU scalar-row checks; no model loading or serving qualification claim."""

import copy
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from online_feedback.first_error import first_error_mask
from online_feedback.kplus import KPlusRedistribution, group_terms, redistribute_response
from online_feedback.second_follower import (
    SecondFollowerRedistribution, install_second_follower,
    reduce_second_follower, second_follower_contract, second_follower_groups,
    second_follower_terms,
)
from specforge.core.chunking import checkpointed_chunk_reduce


def fixture():
    labels = torch.arange(16).reshape(1, 1, 16).expand(1, 17, 16).clone()
    proposals = labels.clone()
    weights = torch.exp(-(torch.arange(16).float() - 1).clamp_min(0) / 7)
    weights = weights.reshape(1, 1, 16).expand_as(labels).clone()
    weights[..., 0] = 0
    # Unequal fixture masses exercise weighted means, not just class averages.
    weights *= torch.linspace(.2, 1.8, 17).reshape(1, -1, 1)
    for block, position, kind in (
        (0, 1, "short"), (1, 2, "long"), (2, 2, "short"),
        (3, 1, "low"), (4, 4, "long"), (5, 14, "censored"),
        (7, 3, "short"), (8, 1, "long"), (9, 2, "short"),
        (10, 3, "long"), (11, 1, "censored"), (12, 1, "short"),
        (13, 1, "censored"), (15, 1, "low"), (16, 1, "low"),
    ):
        proposals[0, block, position] = 29
        if kind == "short":
            proposals[0, block, position + 2] = 28
        if kind == "low":
            proposals[0, block, position + 1] = 28
    labels[0, 11, 2] = proposals[0, 11, 2] = 30  # Correct first follower EOS.
    labels[0, 12, 3] = 30  # Incorrect, valid second follower EOS is SHORT.
    weights[0, 13, 3] = 0  # Correct first follower, second slot unavailable.
    labels[0, 14, 0] = proposals[0, 14, 0] = 30  # EOS seed empties the block.
    weights[0, 15, 2] = 0  # First follower gap cannot reopen.
    labels[0, 16, 1] = 30  # First error at EOS is low priority.
    return proposals, labels, weights


def scalar_oracle(proposals, labels, weights, eos=(30,)):
    """Independent sequential scan and Python-float mass allocation."""
    selected = torch.zeros_like(weights)
    errors = []
    classes = {name: [] for name in ("short", "long", "censored")}
    for block in range(labels.shape[1]):
        valid = []
        if int(labels[0, block, 0]) not in eos:
            for position in range(1, labels.shape[-1]):
                if float(weights[0, block, position]) <= 0:
                    break
                valid.append(position)
                if int(labels[0, block, position]) in eos:
                    break
        for position in valid:
            selected[0, block, position] = weights[0, block, position]
            if int(proposals[0, block, position]) != int(labels[0, block, position]):
                first_correct = (position + 1 in valid and
                    int(proposals[0, block, position + 1]) == int(labels[0, block, position + 1]))
                errors.append((block, position, 2 if first_correct else 1))
                if first_correct:
                    name = ("censored" if position + 2 not in valid else
                            "long" if int(proposals[0, block, position + 2]) == int(labels[0, block, position + 2])
                            else "short")
                    classes[name].append((block, position))
                break
    W1 = sum(float(weights[0, b, j]) for b, j, priority in errors if priority == 1)
    W2 = sum(float(weights[0, b, j]) for b, j, priority in errors if priority == 2)
    c1 = 1 - .5 * W2 / (W1 + 2 * W2) if W1 > 0 and W2 > 0 else 1.
    c2 = 1 + .5 * W1 / (W1 + 2 * W2) if W1 > 0 and W2 > 0 else 1.
    native = selected.double()
    for block, position, priority in errors:
        native[0, block, position] *= c1 if priority == 1 else c2
    candidate = native.clone()
    moved = {}
    for position in range(labels.shape[-1]):
        short = [b for b, j in classes["short"] if j == position]
        long = [b for b, j in classes["long"] if j == position]
        bs = sum(float(weights[0, b, position]) for b in short)
        bl = sum(float(weights[0, b, position]) for b in long)
        transfer = min(c2 * bs / 8, c2 * bl / 8, (c2 - 1) * bs) if bs and bl and c2 > 1 else 0.
        moved[position] = transfer
        if transfer:
            for block in short:
                candidate[0, block, position] -= transfer * float(weights[0, block, position]) / bs
            for block in long:
                candidate[0, block, position] += transfer * float(weights[0, block, position]) / bl
    return selected, native, candidate, classes, moved


def objective_terms(rows, proposals, labels, weights, *, chunk=8):
    def reduce_chunk(local_rows, chosen, canonical, local_weights):
        selection = first_error_mask(chosen, canonical, local_weights, (30,))
        return ((local_rows * local_weights * selection).sum(),
                (local_weights * selection).sum(), local_weights.new_zeros(()),
                selection.sum().float(), local_weights.new_zeros(21),
                *second_follower_terms(local_rows, chosen, canonical, local_weights, selection, (30,)))
    return checkpointed_chunk_reduce(reduce_chunk, rows, proposals, labels, weights,
                                     chunk_size=chunk, dim=1, checkpoint=True)


class SecondFollowerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def test_detached_classes_match_independent_eos_gap_boundary_walk(self):
        proposals, labels, weights = fixture()
        before = [x.clone() for x in (proposals, labels, weights)]
        rng = torch.get_rng_state().clone()
        selection = first_error_mask(proposals, labels, weights, (30,))
        actual = second_follower_groups(proposals, labels, weights, selection, (30,))
        selected, _, _, expected, _ = scalar_oracle(proposals, labels, weights)
        self.assertTrue(torch.equal(selection, selected > 0))
        for name, group in zip(("short", "long", "censored"), actual):
            self.assertEqual(torch.nonzero(group, as_tuple=False).tolist(),
                             [[0, b, j] for b, j in expected[name]])
            self.assertFalse(group.requires_grad)
        self.assertIn((11, 1), expected["censored"])
        self.assertIn((12, 1), expected["short"])
        self.assertIn((13, 1), expected["censored"])
        self.assertIn((5, 14), expected["censored"])
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, (proposals, labels, weights))))

    def test_mixed_depths_across_chunks_match_dense_scalar_loss_and_gradient(self):
        proposals, labels, weights = fixture()
        selected, native, expected_weights, _, expected_transfer = scalar_oracle(proposals, labels, weights)
        rows = torch.linspace(.01, 3., labels.numel(), dtype=torch.float64).reshape_as(weights).requires_grad_()
        terms = objective_terms(rows, proposals, labels, weights)
        actual, metrics, receipt = reduce_second_follower(terms)
        loss = actual / (terms[1] + 1e-6)
        loss.backward()
        expected = (rows.detach() * expected_weights).sum() / (float(selected.sum()) + 1e-6)
        torch.testing.assert_close(loss, expected, rtol=2e-6, atol=2e-7)
        torch.testing.assert_close(rows.grad, expected_weights / (float(selected.sum()) + 1e-6), rtol=2e-6, atol=2e-7)
        self.assertEqual(float(metrics["second_follower_active_strata"]), 3.)
        self.assertEqual(float(metrics["second_follower_depth_4_transferred_weight"]), 0.)
        for position, expected in expected_transfer.items():
            self.assertAlmostEqual(float(metrics[f"second_follower_depth_{position}_transferred_weight"]), expected, delta=2e-7)
        self.assertTrue(all(not value.requires_grad for value in metrics.values()))
        self.assertTrue(all(not value.requires_grad for value in receipt["depths"].values()))
        self.assertTrue(terms[17].requires_grad and terms[18].requires_grad)
        self.assertTrue(all(not value.requires_grad for value in terms[11:17]))
        # Followers only label classes; all originally excluded rows stay loss-free.
        self.assertTrue(torch.equal(rows.grad[selected == 0], torch.zeros_like(rows.grad[selected == 0])))
        torch.testing.assert_close(expected_weights.sum(), native.sum(), rtol=1e-12, atol=1e-12)

    def test_all_three_caps_and_donor_floor(self):
        for short_mass, long_mass, low_mass, binding in ((1., 2., 100., "short"),
                (2., 1., 100., "long"), (1., 1., .01, "floor")):
            with self.subTest(binding=binding):
                labels = torch.arange(5).reshape(1, 1, 5).expand(1, 3, 5).clone()
                proposals = labels.clone()
                proposals[..., 1] = 9
                proposals[0, 0, 3] = 8
                proposals[0, 2, 2] = 8
                weights = torch.ones_like(labels, dtype=torch.float32)
                weights[..., 0] = 0
                weights *= torch.tensor([short_mass, long_mass, low_mass]).reshape(1, 3, 1)
                rows = torch.arange(15.).reshape_as(weights).requires_grad_()
                _, metrics, _ = reduce_second_follower(objective_terms(rows, proposals, labels, weights))
                c = float(metrics["kplus_coefficient2"])
                choices = {"short": c * short_mass / 8, "long": c * long_mass / 8, "floor": (c - 1) * short_mass}
                self.assertAlmostEqual(float(metrics["second_follower_depth_1_transferred_weight"]), choices[binding], delta=2e-7)
                self.assertGreaterEqual(float(metrics["second_follower_depth_1_short_coefficient"]), 1. - 2e-7)
                self.assertGreaterEqual(float(metrics["second_follower_depth_1_short_relative_scale"]), .875 - 2e-7)
                self.assertLessEqual(float(metrics["second_follower_depth_1_long_relative_scale"]), 1.125 + 2e-7)

    def test_zero_transfer_returns_exact_native_tensor(self):
        for case in ("disabled", "all_correct", "all_high", "only_short", "only_long", "empty"):
            with self.subTest(case=case):
                proposals, labels, weights = fixture()
                if case != "disabled":
                    if case == "all_high":
                        # Both follower classes, but no native low-priority mass:
                        # native c is exactly one, so donor-floor transfer is zero.
                        labels = torch.arange(16).reshape(1, 1, 16).expand_as(labels).clone()
                        weights.fill_(1.)
                        weights[..., 0] = 0
                    proposals = labels.clone()
                    if case in ("all_high", "only_short", "only_long"):
                        proposals[0, :, 1] = 29
                        if case == "all_high":
                            proposals[0, :8, 3] = 28
                        if case == "only_short":
                            proposals[0, :, 3] = 28
                        if case != "all_high":
                            proposals[0, 3, 2] = 28
                    if case == "empty":
                        weights.zero_()
                rows = torch.linspace(.01, 1., labels.numel(), dtype=torch.bfloat16).reshape_as(weights).requires_grad_()
                terms = objective_terms(rows, proposals, labels, weights)
                native, _ = redistribute_response(terms[0], terms[1], terms[5:11])
                # Verify object identity to the actual native invocation inside reducer.
                with mock.patch("online_feedback.second_follower.redistribute_response", return_value=(native, redistribute_response(terms[0], terms[1], terms[5:11])[1])):
                    actual, metrics, _ = reduce_second_follower(terms, disabled=case == "disabled")
                self.assertIs(actual, native)
                self.assertEqual(float(metrics["second_follower_transferred_weight"]), 0.)
                if case == "all_high":
                    self.assertEqual(float(metrics["kplus_coefficient2"]), 1.)
                    self.assertEqual(float(metrics["second_follower_mixed_strata"]), 1.)
                first = torch.autograd.grad(actual, rows, retain_graph=True)[0]
                second = torch.autograd.grad(native, rows)[0]
                self.assertTrue(torch.equal(first, second))

    def test_disabled_two_response_bf16_grad_adam_rng_and_next_draw_equal(self):
        proposals, labels, weights = fixture()
        initial = torch.linspace(.01, 1., labels.numel(), dtype=torch.bfloat16).reshape_as(weights)
        initial_rng = torch.get_rng_state().clone()
        outcomes = []
        for disabled in (False, True):
            torch.set_rng_state(initial_rng)
            parameter = initial.clone().requires_grad_()
            optimizer = torch.optim.AdamW([parameter], lr=3e-4, weight_decay=.01)
            for response in range(2):
                rows = parameter * (response + 1)
                terms = objective_terms(rows, proposals, labels, weights)
                if disabled:
                    numerator, _, _ = reduce_second_follower(terms, disabled=True)
                else:
                    numerator, _ = redistribute_response(terms[0], terms[1], terms[5:11])
                (numerator / (terms[1] + 1e-6) / 2).backward()
            gradient = parameter.grad.clone()
            torch.nn.utils.clip_grad_norm_([parameter], 1.)
            clipped = parameter.grad.clone()
            optimizer.step()
            outcomes.append((gradient, clipped, parameter.detach().clone(), copy.deepcopy(optimizer.state_dict()), torch.get_rng_state(), torch.rand(8)))
        for index in (0, 1, 2, 4, 5):
            self.assertTrue(torch.equal(outcomes[0][index], outcomes[1][index]))
        left, right = outcomes[0][3], outcomes[1][3]
        self.assertEqual(left["param_groups"], right["param_groups"])
        for key in left["state"][0]:
            self.assertTrue(torch.equal(left["state"][0][key], right["state"][0][key]))

    def test_chunk_partition_changes_no_membership_or_sufficient_statistic_meaning(self):
        proposals, labels, weights = fixture()
        rows = torch.linspace(.1, .9, labels.numel()).reshape_as(weights)
        totals = [objective_terms(rows, proposals, labels, weights, chunk=size) for size in (1, 8, 16, 0)]
        for other in totals[1:]:
            for expected, actual in zip(totals[0], other):
                torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)
        values = [reduce_second_follower(terms)[0] for terms in totals]
        for value in values[1:]:
            torch.testing.assert_close(value, values[0], rtol=2e-6, atol=2e-7)

    def test_installer_replaces_only_exact_native_reducer_and_exposes_receipt(self):
        for existing in (None, KPlusRedistribution()):
            model = SimpleNamespace(objective="osd_first_error", loss_dtype=torch.bfloat16,
                                    first_error_redistribution=existing)
            self.assertEqual(install_second_follower(model), second_follower_contract())
            reducer = model.first_error_redistribution
            self.assertIsInstance(reducer, SecondFollowerRedistribution)
            proposals, labels, weights = fixture()
            reducer.reduce_response(objective_terms(torch.ones_like(weights), proposals, labels, weights))
            self.assertEqual(reducer.last_receipt["contract"], second_follower_contract())
            with self.assertRaises(ValueError):
                install_second_follower(model)
        for attribute in ("prefix_guard", "point_loss_extension", "role_loss_extension",
                          "survival_head_extension", "successor_shuffle", "adversarial_block"):
            model = SimpleNamespace(objective="osd_first_error", loss_dtype=torch.bfloat16)
            setattr(model, attribute, object())
            with self.assertRaises(ValueError):
                install_second_follower(model)
        model = SimpleNamespace(objective="osd", loss_dtype=torch.bfloat16)
        with self.assertRaises(ValueError):
            install_second_follower(model)
        model.objective, model.loss_dtype = "osd_first_error", torch.float32
        with self.assertRaises(ValueError):
            install_second_follower(model)
        model.loss_dtype = torch.bfloat16
        self.assertTrue(install_second_follower(model, disabled=True)["disabled"])
        with self.assertRaises(ValueError):
            second_follower_contract(disabled=1)


if __name__ == "__main__":
    unittest.main()
