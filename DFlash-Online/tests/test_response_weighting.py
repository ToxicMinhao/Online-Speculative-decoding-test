"""Small CPU arithmetic fixtures; no model or campaign qualification budget."""

import copy
import io
import unittest

import torch

from online_feedback.first_error import selection_statistics
from online_feedback.response_weighting import (
    HalfGradientCapture, KW_CONTRACT, MAX_CHUNK_ELEMENTS, NativeAdamWPreview,
    ResponseWeightController, apply_weighted_gradients_, chunked_dot,
    clone_native_gradients, optimizer_secant, response_features, weighted_gradients,
)


def metrics(**overrides):
    result = {"available_position_weight": 4.0, "selected_position_weight": 2.0,
              "no_error_blocks": 1.0, "empty_blocks": 9.0}
    result.update({f"first_error_position_{i}": 0.0 for i in range(1, 16)})
    result["first_error_position_4"] = 1.0
    result.update(overrides)
    return result


def optimizer_fixture(*, foreach=None):
    parameters = [torch.nn.Parameter(torch.tensor(values, dtype=torch.bfloat16))
                  for values in ([0.015625, -0.03125, 0.0001, 0.125],
                                 [0.0625, 0.0002], [0.25, -0.5])]
    optimizer = torch.optim.AdamW([
        {"params": [parameters[1], parameters[0]], "lr": 0.03, "weight_decay": 0.01},
        {"params": [parameters[2]], "lr": 0.02, "weight_decay": 0.03},
    ], foreach=foreach)
    for index, parameter in enumerate(parameters):
        parameter.grad = torch.full_like(parameter, (index + 1) * 0.1)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    native = [torch.tensor([0.9, -0.6, 0.2, 0.8], dtype=torch.bfloat16),
              torch.tensor([0.3, -0.4], dtype=torch.bfloat16), None]
    first = [torch.tensor([0.8, 0.2, -0.3, 0.7], dtype=torch.bfloat16),
             torch.tensor([-0.5, 0.4], dtype=torch.bfloat16), None]
    second = [torch.tensor([0.1, -0.8, 0.5, 0.1], dtype=torch.bfloat16), None, None]
    for parameter, gradient in zip(parameters, native):
        parameter.grad = gradient
    return tuple(zip(("first", "second", "inactive"), parameters)), optimizer, native, first, second


class ResponseWeightingTests(unittest.TestCase):
    def assert_tree_equal(self, actual, expected):
        if isinstance(expected, torch.Tensor):
            self.assertEqual(actual.dtype, expected.dtype)
            self.assertEqual(actual.device, expected.device)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        elif isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in expected:
                self.assert_tree_equal(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(type(actual), type(expected))
            self.assertEqual(len(actual), len(expected))
            for one, two in zip(actual, expected):
                self.assert_tree_equal(one, two)
        else:
            self.assertEqual(actual, expected)

    def test_features_exact_definitions_empty_blocks_loss_and_length_clamps(self):
        actual = response_features(torch.tensor(3.0, requires_grad=True), metrics(), 2049)
        expected = torch.tensor([2 / (4 + 1e-6), 20 / 32, 0.5, 0.75, 1.0])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(actual.dtype, torch.float32)
        self.assertEqual(actual.device.type, "cpu")
        self.assertFalse(actual.requires_grad)
        empty = metrics(no_error_blocks=0, first_error_position_4=0,
                        selected_position_weight=0, available_position_weight=0)
        torch.testing.assert_close(response_features(-3, empty, 1024),
                                   torch.tensor([0, 0, 0, 0, 0.5]), rtol=0, atol=0)

    def test_eos_seed_is_empty_but_eos_continuation_is_nonempty(self):
        labels = torch.zeros((2, 16), dtype=torch.long)
        labels[0, 0] = 7
        labels[1, 1] = 7
        weights = torch.ones((2, 16))
        weights[:, 0] = 0
        observed = selection_statistics(labels, labels, weights, weights.bool(), (7,))
        names = ["available_position_occurrences", "selected_position_occurrences",
                 "available_position_weight", "selected_position_weight",
                 "no_error_blocks", "empty_blocks"]
        names += [f"first_error_position_{i}" for i in range(1, 16)]
        values = dict(zip(names, observed))
        self.assertEqual(float(values["no_error_blocks"]), 1)
        self.assertEqual(float(values["empty_blocks"]), 1)
        actual = response_features(0, values, 2)
        self.assertEqual(float(actual[1]), 1)
        self.assertEqual(float(actual[2]), 1)
        self.assertEqual(float(actual[-1]), 2 / 2048)

    def test_features_reject_invalid_raw_values_before_clamping(self):
        for key, value in (("selected_position_weight", -1), ("no_error_blocks", -1),
                           ("first_error_position_15", float("nan")),
                           ("empty_blocks", float("inf"))):
            with self.subTest(key=key), self.assertRaises(ValueError):
                response_features(1, metrics(**{key: value}), 10)
        for loss, length in ((float("inf"), 10), (1, 0), (1, 1.5), (1, float("nan"))):
            with self.subTest(loss=loss, length=length), self.assertRaises(ValueError):
                response_features(loss, metrics(), length)

    def test_response_swap_and_descent_direction(self):
        controller = ResponseWeightController()
        controller.phi.copy_(torch.tensor([0.2, -0.4, 0.3, 0.1, -0.2]))
        first = torch.tensor([0.8, 0.2, 0.4, 0.9, 0.6])
        second = torch.tensor([0.1, 0.5, 0.3, 0.2, 0.5])
        allocation = controller.allocate(first, second)
        swapped = controller.allocate(second, first)
        self.assertEqual(swapped.a, -allocation.a)
        self.assert_tree_equal(swapped.b, -allocation.b)
        self.assertEqual(swapped.weights, tuple(reversed(allocation.weights)))
        before = controller.phi.clone()
        row = controller.update_credit(2.0, allocation.b)
        self.assertLess(float(torch.dot(controller.phi - before, allocation.b)), 0)
        self.assertEqual(row["v"], 0.2)
        self.assertEqual(row["n"], 1)

    def test_delayed_allocation_uses_entry_phi_and_clears_each_pass(self):
        controller = ResponseWeightController()
        first, second = torch.ones(5), torch.zeros(5)
        allocations = []
        names = ("parameter",)
        for window in range(3):
            entry = controller.phi.clone()
            allocation = controller.allocate(first, second, phi_entry=entry)
            row = controller.credit([torch.ones(2, dtype=torch.bfloat16)], names)
            allocations.append(allocation.a)
            if window > 0:
                self.assertEqual(row["producing_identity"], {"window": window - 1})
            controller.save_pending([torch.ones(2)], allocation.b, {"window": window},
                                    names=names, end_of_pass=window == 2)
        self.assertEqual(allocations[:2], [0.0, 0.0])
        self.assertLess(allocations[2], 0)
        self.assertEqual((controller.n, controller.windows, controller.previews), (2, 3, 6))
        self.assertIsNone(controller.pending)
        self.assertIsNone(controller.credit([torch.ones(2)], names))

    def test_zero_credit_counts_and_skips_projection_bitwise(self):
        controller = ResponseWeightController()
        controller.phi[:] = torch.tensor([-0.0, 0.0, 0.0, 0.0, 0.0])
        controller.phi[2] = 2.0
        initial_bytes = controller.phi.view(torch.int64).clone()
        controller.v = 4.0
        controller.n = 1
        row = controller.update_credit(0, torch.ones(5))
        self.assert_tree_equal(controller.phi.view(torch.int64), initial_bytes)
        self.assertEqual((controller.n, controller.v), (2, 3.8))
        self.assertEqual(row["controller_update_norm"], 0)
        self.assertFalse(row["controller_projected"])
        row = controller.update_credit(2, torch.zeros(5))
        self.assert_tree_equal(controller.phi.view(torch.int64), initial_bytes)
        self.assertEqual(row["controller_update_norm"], 0)

    def test_controller_projection_and_checkpoint_roundtrip(self):
        controller = ResponseWeightController()
        controller.phi[0] = 2
        controller.update_credit(-1, [1, 0, 0, 0, 0])
        self.assertEqual(float(torch.linalg.vector_norm(controller.phi)), 2)
        controller.save_pending([torch.zeros(2)], torch.zeros(5), "last",
                                names=["x"], end_of_pass=True)
        saved = controller.state_dict()
        data = io.BytesIO()
        torch.save(saved, data)
        data.seek(0)
        restored = ResponseWeightController()
        restored.load_state_dict(torch.load(data, weights_only=False))
        self.assert_tree_equal(restored.state_dict(), saved)
        self.assertEqual(saved["contract"], KW_CONTRACT)
        restored.save_pending([torch.zeros(2)], torch.zeros(5), "pending", names=["x"])
        with self.assertRaisesRegex(ValueError, "complete event"):
            restored.state_dict()
        malformed = copy.deepcopy(saved)
        malformed["contract"]["secant_half_width"] = 0.1
        with self.assertRaisesRegex(ValueError, "contract"):
            controller.load_state_dict(malformed)
        self.assert_tree_equal(controller.state_dict(), saved)

    def test_raw_hook_capture_preserves_small_second_contribution(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.bfloat16))
        capture = HalfGradientCapture([("p", parameter)])
        try:
            capture.begin_response()
            (parameter * 512 / 2).sum().backward()
            first = capture.end_response()
            accumulated_first = parameter.grad.clone()
            capture.begin_response()
            (parameter * 0.5 / 2).sum().backward()
            second = capture.end_response()
        finally:
            capture.close()
        self.assertEqual(float(first[0]), 256)
        self.assertEqual(float(second[0]), 0.25)
        self.assertEqual(float(parameter.grad - accumulated_first), 0)

    def test_weighting_preserves_neutral_native_storage_and_none(self):
        named, _, native, first, second = optimizer_fixture()
        actual = weighted_gradients(native, first, second, 0)
        self.assertIs(actual[0], native[0])
        self.assertIs(actual[1], native[1])
        self.assertIsNone(actual[2])
        saved = clone_native_gradients(named)
        pointers = [None if p.grad is None else p.grad.data_ptr() for _, p in named]
        apply_weighted_gradients_(named, native, first, second, 0)
        self.assert_tree_equal(clone_native_gradients(named), saved)
        expected = []
        for gradient, one, two in zip(native[:2], first[:2], second[:2]):
            difference = one.float() - (torch.zeros_like(one).float() if two is None else two.float())
            expected.append((gradient.float() + torch.tensor(0.4) * difference).bfloat16())
        apply_weighted_gradients_(named, native, first, second, 0.2)
        for (_, parameter), result in zip(named, expected):
            self.assert_tree_equal(parameter.grad, result)
        self.assertIsNone(named[2][1].grad)
        self.assertEqual(pointers, [None if p.grad is None else p.grad.data_ptr() for _, p in named])

    def test_preview_matches_independent_native_step_and_all_state(self):
        for foreach in (None, False, True):
            for allocation in (0, -0.25, 0.137, 0.25):
                with self.subTest(foreach=foreach, allocation=allocation):
                    named, optimizer, native, first, second = optimizer_fixture(foreach=foreach)
                    weights = [p.detach().clone() for _, p in named]
                    state = copy.deepcopy(optimizer.state_dict())
                    gradients = clone_native_gradients(named)
                    grad_ids = [id(p.grad) for _, p in named]
                    rng = torch.get_rng_state().clone()
                    scratch = NativeAdamWPreview(optimizer, named)
                    observed = []
                    preview_norm = scratch.apply(native, first, second, allocation,
                                                 on_optimizer_step=observed.append)
                    self.assertEqual(observed, [allocation])
                    self.assert_tree_equal([p.detach() for _, p in named], weights)
                    self.assert_tree_equal(optimizer.state_dict(), state)
                    self.assert_tree_equal(clone_native_gradients(named), gradients)
                    self.assertEqual([id(p.grad) for _, p in named], grad_ids)
                    self.assert_tree_equal(torch.get_rng_state(), rng)
                    # Independent perturbation, then the live native BF16 clip/Adam.
                    for (_, parameter), gradient, one, two in zip(named, native, first, second):
                        if gradient is None:
                            parameter.grad = None
                        elif allocation == 0:
                            parameter.grad = gradient.clone()
                        else:
                            difference = one.float() - (torch.zeros_like(one).float() if two is None else two.float())
                            parameter.grad = (gradient.float() + torch.tensor(2 * allocation) * difference).bfloat16()
                    norm = torch.nn.utils.clip_grad_norm_([p for _, p in named], 1.0,
                                                         error_if_nonfinite=True)
                    optimizer.step()
                    self.assert_tree_equal(preview_norm, norm)
                    self.assert_tree_equal([p.detach() for p in scratch.parameters],
                                           [p.detach() for _, p in named])
                    self.assert_tree_equal(scratch.optimizer.state_dict(), optimizer.state_dict())
                    self.assert_tree_equal(named[2][1].detach(), weights[2])
                    self.assertEqual(float(optimizer.state[named[2][1]]["step"]), 1)

    def test_secant_clipped_denominator_zero_case_and_swap(self):
        named, optimizer, native, first, second = optimizer_fixture()
        observed = []
        actual = optimizer_secant(optimizer, named, native, first, second, 0.24,
                                  on_optimizer_step=observed.append)
        self.assertEqual(observed, [0.25, 0.1775])
        self.assertEqual(actual.stats["secant_denominator"], 0.07250000000000001)
        scratch = NativeAdamWPreview(optimizer, named)
        scratch.apply(native, first, second, 0.25)
        plus = [p.detach().float() for p in scratch.parameters]
        scratch.apply(native, first, second, 0.1775)
        expected = [(one - p.detach().float()) / torch.tensor(0.07250000000000001)
                    for one, p in zip(plus, scratch.parameters)]
        self.assert_tree_equal(actual.eligibility, expected)
        self.assertFalse(actual.stats["zero_secant"])
        opposite = optimizer_secant(optimizer, named, native, second, first, -0.24)
        self.assert_tree_equal(opposite.eligibility, [-value for value in actual.eligibility])
        zeros = optimizer_secant(optimizer, named, native, first, first, 0)
        self.assertTrue(zeros.stats["zero_secant"])
        self.assertEqual(zeros.stats["secant_norm"], 0)
        self.assertEqual(chunked_dot(native, zeros.eligibility), 0)

    def test_scratch_restarts_from_identical_state_and_keeps_fresh_none_inactive(self):
        named, optimizer, native, first, second = optimizer_fixture()
        inactive = named[2][1]
        del optimizer.state[inactive]
        original = inactive.detach().clone()
        scratch = NativeAdamWPreview(optimizer, named)
        scratch.apply(native, first, second, 0.12)
        first_parameters = [p.detach().clone() for p in scratch.parameters]
        first_state = copy.deepcopy(scratch.optimizer.state_dict())
        scratch.apply(native, first, second, -0.2)
        scratch.apply(native, first, second, 0.12)
        self.assert_tree_equal([p.detach() for p in scratch.parameters], first_parameters)
        self.assert_tree_equal(scratch.optimizer.state_dict(), first_state)
        self.assert_tree_equal(scratch.parameters[2].detach(), original)
        self.assertNotIn(scratch.parameters[2], scratch.optimizer.state)
        self.assertNotIn(inactive, optimizer.state)

    def test_all_none_secant_and_zero_credit_still_advance_rms_count(self):
        named, optimizer, _, _, _ = optimizer_fixture()
        missing = [None] * len(named)
        for _, parameter in named:
            parameter.grad = None
        result = optimizer_secant(optimizer, named, missing, missing, missing, 0.25)
        self.assertTrue(result.stats["zero_secant"])
        controller = ResponseWeightController()
        controller.save_pending(result.eligibility, torch.ones(5), "previous",
                                names=[name for name, _ in named])
        row = controller.credit(missing, [name for name, _ in named])
        self.assertEqual((row["credit"], row["n"], row["v"]), (0.0, 1, 0.0))
        self.assertEqual(row["controller_update_norm"], 0)

    def test_callback_stops_before_scratch_application_without_touching_live(self):
        named, optimizer, native, first, second = optimizer_fixture()
        saved_weights = [p.detach().clone() for _, p in named]
        saved_state = copy.deepcopy(optimizer.state_dict())
        def exhausted(_allocation):
            raise RuntimeError("qualification budget exhausted")
        with self.assertRaisesRegex(RuntimeError, "budget exhausted"):
            optimizer_secant(optimizer, named, native, first, second, 0,
                              on_optimizer_step=exhausted)
        self.assert_tree_equal([p.detach() for _, p in named], saved_weights)
        self.assert_tree_equal(optimizer.state_dict(), saved_state)

    def test_chunked_dot_uses_fixed_cpu_fp64_accumulation_and_none(self):
        length = MAX_CHUNK_ELEMENTS + 9
        one = (torch.arange(length, dtype=torch.float32) % 17 - 8).bfloat16()
        two = torch.arange(length, dtype=torch.float32) % 13 / 10
        expected = torch.zeros((), dtype=torch.float64)
        for start in range(0, length, MAX_CHUNK_ELEMENTS):
            expected.add_((one[start:start + MAX_CHUNK_ELEMENTS].double()
                           * two[start:start + MAX_CHUNK_ELEMENTS].double()).sum())
        self.assertEqual(chunked_dot([one, None], [two, torch.ones(7)]), float(expected))
        with self.assertRaises(ValueError):
            chunked_dot([one], [two], chunk_elements=MAX_CHUNK_ELEMENTS + 1)

    def test_nonfinite_gradients_credit_and_proposals_fail(self):
        named, optimizer, native, first, second = optimizer_fixture()
        invalid = [value.clone() if value is not None else None for value in native]
        invalid[0][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            optimizer_secant(optimizer, named, invalid, first, second, 0)
        controller = ResponseWeightController()
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            controller.update_credit(float("inf"), torch.ones(5))
        self.assertEqual(controller.n, 0)
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            controller.update_credit(1e200, torch.ones(5))
        self.assertEqual(controller.n, 0)
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            chunked_dot([None], [torch.tensor([float("nan")])])


if __name__ == "__main__":
    unittest.main()
