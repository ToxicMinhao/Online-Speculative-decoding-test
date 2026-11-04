"""CPU qualification of fixed VFD selection and author-forward isolation."""

import copy
import hashlib
import math
import random
import unittest
from unittest import mock

import numpy as np
import torch

from online_feedback.first_error import valid_prefix_mask
from online_feedback.objective import TargetObjectiveModel
from online_feedback.rng import capture_rng
from online_feedback.vfd_scoring import (
    POLICY_VERSION,
    native_eligible_anchors,
    native_predictions_and_margins,
    score_response_pair,
    score_response_snapshot,
    select_responses,
    select_score_anchors,
    selection_hash,
)
from test_objective_integration import tiny_author, response_batch


def assert_rng_equal(test, left, right):
    test.assertEqual(left["python"], right["python"])
    test.assertEqual(left["numpy"][0], right["numpy"][0])
    np.testing.assert_array_equal(left["numpy"][1], right["numpy"][1])
    test.assertEqual(left["numpy"][2:], right["numpy"][2:])
    test.assertTrue(torch.equal(left["torch"], right["torch"]))
    test.assertEqual(len(left["cuda"]), len(right["cuda"]))
    for first, second in zip(left["cuda"], right["cuda"]):
        test.assertTrue(torch.equal(first, second))


def block(predictions, margins=None, labels=None, weights=None, eos=()):
    labels = list(range(len(predictions))) if labels is None else labels
    weights = [0.0] + [1.0] * (len(labels) - 1) if weights is None else weights
    valid = valid_prefix_mask(torch.tensor(labels), torch.tensor(weights), eos).tolist()
    return dict(
        anchor_id=0,
        predictions=list(predictions),
        labels=list(labels),
        weights=list(weights),
        valid=valid,
        margins=list(
            margins or [1.0 if p == y else -2.0 for p, y in zip(predictions, labels)]
        ),
    )


def snapshot(blocks, response_id="response", event_id=1):
    blocks = copy.deepcopy(blocks)
    for index, row in enumerate(blocks):
        row["anchor_id"] = index
    return dict(
        policy_version=POLICY_VERSION,
        event_id=event_id,
        response_id=response_id,
        anchor_ids=list(range(len(blocks))),
        eligible_anchors=len(blocks),
        sampled_anchors=len(blocks),
        context_length=80,
        response_length=64,
        conditioning_id="fixture-packet",
        input_ids_sha256="fixture-tokens",
        loss_mask_sha256="fixture-mask",
        feature_shape=[1, 80, 16],
        feature_dtype="torch.bfloat16",
        block_size=5,
        eos_token_ids=[],
        evaluation_mode=True,
        native_logits_dtype="torch.bfloat16",
        blocks=blocks,
    )


class ScoringFormulaTests(unittest.TestCase):
    def test_credit_and_all_valid_blocks_denominator_match_scalar_oracle(self):
        first = block([0, 1, 8, 3, 4])
        second = block([0, 1, 2, 3, 4])
        after = copy.deepcopy(first)
        after["margins"][2] = -1.0
        result = score_response_pair(
            snapshot([first, second]), snapshot([after, second])
        )
        # d=1, g=1; fully matching block is included in the denominator.
        expected = (2 / (1 + math.exp(1)) * 1 / (1 + 1)) / 2
        self.assertAlmostEqual(result["progress_score"], expected, places=7)
        self.assertEqual(result["static_score"], 0.5)
        self.assertEqual(result["counts"]["valid_blocks"], 2)
        self.assertEqual(result["counts"]["fully_matching_blocks"], 1)
        self.assertEqual(result["counts"]["qualifying_opportunities"], 1)

    def test_gates_repaired_prefix_broken_successor_broken_and_no_movement(self):
        pre = block([0, 1, 8, 3, 4])
        cases = [
            ("repaired", [0, 1, 2, 3, 4], 1.0, "repaired_errors"),
            ("prefix", [0, 8, 8, 3, 4], -1.0, "broken_prefixes"),
            ("successor", [0, 1, 8, 8, 4], -1.0, "broken_successors"),
            ("unchanged", [0, 1, 8, 3, 4], -2.0, "persistent_errors"),
            ("worse", [0, 1, 8, 3, 4], -3.0, "persistent_errors"),
        ]
        for name, predictions, margin, count in cases:
            with self.subTest(name=name):
                post = block(predictions)
                post["margins"][2] = margin
                result = score_response_pair(snapshot([pre]), snapshot([post]))
                self.assertEqual(result["progress_score"], 0.0)
                self.assertEqual(result["counts"][count], 1)
        repaired_and_broken = block([0, 8, 2, 3, 4])
        result = score_response_pair(snapshot([pre]), snapshot([repaired_and_broken]))
        self.assertEqual(result["counts"]["repaired_errors"], 1)
        self.assertEqual(result["counts"]["broken_prefixes"], 1)

    def test_same_error_position_does_not_require_same_wrong_token(self):
        before, after = block([0, 1, 8, 3, 4]), block([0, 1, 9, 3, 4])
        after["margins"][2] = -1.0
        result = score_response_pair(snapshot([before]), snapshot([after]))
        self.assertGreater(result["progress_score"], 0)

    def test_no_successor_no_opportunity_last_slot_or_wrong_successor(self):
        blocks = [block([0, 1, 2, 3, 8]), block([0, 1, 8, 9, 4])]
        result = score_response_pair(snapshot(blocks), snapshot(blocks))
        self.assertEqual(result["static_score"], 0.0)
        self.assertEqual(result["counts"]["initial_errors"], 2)

    def test_eos_and_gap_stop_validity_including_eos_error(self):
        blocks = [
            block([0, 8, 2, 3, 4], labels=[0, 9, 2, 3, 4], eos=(9,)),
            block([0, 8, 2, 3, 4], weights=[0, 1, 0, 1, 1]),
            block([9, 8, 2, 3, 4], labels=[9, 1, 2, 3, 4], eos=(9,)),
        ]
        result = score_response_pair(snapshot(blocks), snapshot(blocks))
        self.assertEqual(result["counts"]["valid_blocks"], 2)
        self.assertEqual(result["static_score"], 0.0)
        self.assertEqual(blocks[0]["valid"], [False, True, False, False, False])
        self.assertEqual(blocks[2]["valid"], [False] * 5)

    def test_matching_eos_can_be_immediate_successor(self):
        before = block([0, 8, 9, 3, 4], labels=[0, 1, 9, 3, 4], eos=(9,))
        after = copy.deepcopy(before)
        after["margins"][1] = -1
        result = score_response_pair(snapshot([before]), snapshot([after]))
        self.assertEqual(result["static_score"], 1)
        self.assertGreater(result["progress_score"], 0)

    def test_native_bf16_argmax_ties_and_fp32_margin(self):
        logits = torch.tensor(
            [[2, 2, 1], [2, 2, 1], [1.007, 1.005, -1]], dtype=torch.bfloat16
        )
        labels = torch.tensor([1, 0, 1])
        predictions, margins = native_predictions_and_margins(logits, labels)
        self.assertEqual(predictions.tolist(), [0, 0, 0])
        self.assertEqual(margins.dtype, torch.float32)
        self.assertEqual(margins.tolist(), [0, 0, 0])

    def test_wrong_tie_has_unit_credit_if_gap_closed_and_zero_if_no_movement(self):
        before = block([0, 0, 2, 3, 4])
        after = copy.deepcopy(before)
        after["margins"][1] = 0.0
        result = score_response_pair(snapshot([before]), snapshot([after]))
        self.assertEqual(result["progress_score"], 1.0)
        result = score_response_pair(snapshot([after]), snapshot([after]))
        self.assertEqual(result["progress_score"], 0.0)
        after["predictions"][1] = 1
        result = score_response_pair(snapshot([before]), snapshot([after]))
        self.assertEqual(result["progress_score"], 0.0)
        self.assertEqual(result["counts"]["repaired_errors"], 1)

    def test_empty_snapshot_and_empty_validity_have_zero_scores(self):
        for blocks in ([], [block([0, 1, 2, 3, 4], weights=[0] * 5)]):
            result = score_response_pair(snapshot(blocks), snapshot(blocks))
            self.assertEqual(result["static_score"], 0)
            self.assertEqual(result["progress_score"], 0)
            self.assertEqual(result["counts"]["valid_blocks"], 0)

    def test_pre_post_binding_and_nonfinite_fail_closed(self):
        before = snapshot([block([0, 1, 8, 3, 4])])
        for field, changed in (
            ("conditioning_id", "other"),
            ("evaluation_mode", False),
            ("native_logits_dtype", "torch.float32"),
            ("event_id", 2),
        ):
            with self.subTest(field=field):
                after = copy.deepcopy(before)
                after[field] = changed
                with self.assertRaises(ValueError):
                    score_response_pair(before, after)
        after = copy.deepcopy(before)
        after["blocks"][0]["weights"][1] = 0.5
        with self.assertRaisesRegex(ValueError, "validity"):
            score_response_pair(before, after)
        after = copy.deepcopy(before)
        after["blocks"][0]["margins"][1] = float("nan")
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            score_response_pair(before, after)


class HashSelectionTests(unittest.TestCase):
    def test_native_eligibility_threshold_and_exact_hash_cap(self):
        mask = torch.ones(1, 48)
        mask[0, :3] = 0
        mask[0, 9] = 0.5
        mask[0, 20] = 0
        expected = [
            i
            for i in range(47)
            if float(mask[0, i]) > 0.5 and float(mask[0, i + 1]) > 0.5
        ]
        self.assertEqual(native_eligible_anchors(mask), expected)
        ranked = sorted(
            expected,
            key=lambda i: hashlib.sha256(
                f"vfd-s2-v1|anchor|2|r1|{i}".encode()
            ).hexdigest(),
        )
        self.assertEqual(select_score_anchors(mask, 2, "r1"), sorted(ranked[:16]))
        self.assertEqual(select_score_anchors(torch.ones(1, 4), 2, "r1"), [0, 1, 2])
        for cap in (0, 17, 1.5, True):
            with self.assertRaises(ValueError):
                select_score_anchors(mask, 2, "r1", cap=cap)

    def test_exact64_tie_order_matches_both_selectors_and_induced_order(self):
        ids = [f"r{i:03d}" for i in range(128)]
        scores = [
            score_response_pair(snapshot([], item), snapshot([], item)) for item in ids
        ]
        full_order = list(reversed(ids))
        expected = sorted(
            ids,
            key=lambda item: hashlib.sha256(
                f"vfd-s2-v1|tie|1|{item}".encode()
            ).hexdigest(),
        )[:64]
        static = select_responses(scores, full_order, event_id=1, policy="STATIC64")
        progress = select_responses(scores, full_order, event_id=1, policy="PROGRESS64")
        self.assertEqual(static["selected_ids"], expected)
        self.assertEqual(progress["selected_ids"], expected)
        self.assertEqual(
            static["training_order"], [item for item in full_order if item in expected]
        )
        self.assertEqual(static["zero_rate"], 1)
        self.assertEqual(static["tie_rate"], 1)
        self.assertEqual(static["cutoff_tie_count"], 128)

    def test_random_hash_is_independent_and_does_not_consume_rng(self):
        ids = [f"r{i}" for i in range(128)]
        before = capture_rng()
        selected = select_responses([], ids, event_id=3, policy="RANDOM64")
        expected = sorted(
            ids,
            key=lambda item: hashlib.sha256(
                f"vfd-s2-v1|random|3|{item}".encode()
            ).hexdigest(),
        )[:64]
        self.assertEqual(selected["selected_ids"], expected)
        self.assertEqual(len(set(expected)), 64)
        assert_rng_equal(self, before, capture_rng())

    def test_scores_rank_descending_without_changing_training_order(self):
        ids = ["a", "b", "c", "d"]
        scores = [
            score_response_pair(snapshot([], item), snapshot([], item)) for item in ids
        ]
        for row, value in zip(scores, [0.1, 0.4, 0.3, 0.2]):
            row["progress_score"] = value
        selected = select_responses(
            scores, ids, event_id=1, policy="PROGRESS64", count=2
        )
        self.assertEqual(selected["selected_ids"], ["b", "c"])
        self.assertEqual(selected["training_order"], ["b", "c"])
        scores[1]["has_after_snapshot"] = False
        with self.assertRaisesRegex(ValueError, "completed"):
            select_responses(scores, ids, event_id=1, policy="PROGRESS64", count=2)


class AuthorSnapshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.model = TargetObjectiveModel(
            tiny_author(), "osd_first_error", loss_precision="bfloat16"
        )
        self.batch = response_batch(self.model)

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def test_real_author_snapshot_preserves_rng_mode_sampler_weights_gradients(self):
        model = self.model
        model.train()
        model.draft_model.layers[0].eval()
        observations = []
        original_sampler = model._sample_anchor_positions

        def recorder(*args, **kwargs):
            observations.append("called")
            return original_sampler(*args, **kwargs)

        model._sample_anchor_positions = recorder
        modes = [(module, module.training) for module in model.modules()]
        parameters = [
            (parameter, parameter.detach().clone()) for parameter in model.parameters()
        ]
        parameter = next(model.draft_model.parameters())
        parameter.grad = torch.ones_like(parameter)
        gradient = parameter.grad.clone()
        state = capture_rng()
        score = score_response_snapshot(
            model,
            self.batch,
            event_id=1,
            response_id="r",
            conditioning_id="authenticated-packet",
        )
        assert_rng_equal(self, state, capture_rng())
        self.assertIs(model._sample_anchor_positions, recorder)
        self.assertEqual(observations, [])
        self.assertTrue(all(module.training == mode for module, mode in modes))
        self.assertTrue(
            all(torch.equal(parameter, saved) for parameter, saved in parameters)
        )
        self.assertTrue(torch.equal(parameter.grad, gradient))
        self.assertEqual(score["anchor_ids"], list(range(4, 11)))
        self.assertEqual(score["costs"]["response_forwards"], 1)
        self.assertEqual(score["costs"]["projection_calls"], 4)
        self.assertEqual(score["native_logits_dtype"], "torch.bfloat16")
        self.assertFalse(
            any(
                isinstance(value, torch.Tensor) for value in score["blocks"][0].values()
            )
        )
        repeated = score_response_snapshot(
            model,
            self.batch,
            event_id=1,
            response_id="r",
            conditioning_id="authenticated-packet",
        )
        self.assertEqual(score["blocks"], repeated["blocks"])
        self.assertEqual(score_response_pair(score, repeated)["progress_score"], 0)
        model._sample_anchor_positions = original_sampler

    def test_snapshot_matches_explicit_author_forward_at_identical_anchors(self):
        model = self.model
        actual = score_response_snapshot(model, self.batch, event_id=1, response_id="r")
        anchors = torch.tensor([actual["anchor_ids"]])
        keep = torch.ones_like(anchors, dtype=torch.bool)
        model.eval()
        with torch.no_grad(), mock.patch.object(
            model, "_sample_anchor_positions", return_value=(anchors, keep)
        ):
            _, _, hidden = model._forward_draft_blocks(**self.batch)
            hidden = hidden.reshape(1, len(actual["anchor_ids"]), model.block_size, -1)
            labels, weights, _ = model._block_labels_and_weights(
                self.batch["input_ids"], self.batch["loss_mask"], anchors, keep
            )
            logits = model.lm_head(hidden)
            for index, row in enumerate(actual["blocks"]):
                self.assertEqual(row["labels"], labels[0, index].tolist())
                self.assertEqual(row["weights"], weights[0, index].tolist())
                self.assertEqual(
                    row["predictions"], logits[0, index].argmax(-1).tolist()
                )
                for offset, label in enumerate(row["labels"]):
                    scores = logits[0, index, offset].float().tolist()
                    expected = scores[label] - max(
                        value for token, value in enumerate(scores) if token != label
                    )
                    self.assertEqual(row["margins"][offset], expected)

    def test_exception_restores_runtime_and_existing_cache_bindings(self):
        model = self.model
        model.train()
        cache = object()
        model._cached_block_mask = cache
        sampler_was_instance_attribute = "_sample_anchor_positions" in model.__dict__
        state = capture_rng()
        modes = [(module, module.training) for module in model.modules()]
        rotary_buffer = model.draft_model.rotary_emb.inv_freq
        old_buffer = rotary_buffer.clone()

        def fail(*args, **kwargs):
            self.assertFalse(torch.is_grad_enabled())
            self.assertTrue(all(not module.training for module in model.modules()))
            self.assertIsNone(model._cached_block_mask)
            torch.rand(3)
            random.random()
            np.random.random()
            rotary_buffer.add_(1)
            model._cached_seq_len = 999
            raise RuntimeError("qualification failure")

        with mock.patch.object(model, "_forward_draft_blocks", side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, "qualification failure"):
                score_response_snapshot(model, self.batch, event_id=1, response_id="r")
        assert_rng_equal(self, state, capture_rng())
        self.assertTrue(all(module.training == mode for module, mode in modes))
        self.assertIs(model._cached_block_mask, cache)
        self.assertIsNone(model._cached_seq_len)
        self.assertIs(model.draft_model.rotary_emb.inv_freq, rotary_buffer)
        self.assertTrue(torch.equal(rotary_buffer, old_buffer))
        self.assertEqual(
            "_sample_anchor_positions" in model.__dict__, sampler_was_instance_attribute
        )

    def test_empty_anchors_do_not_forward_and_wrong_anchors_fail(self):
        batch = dict(self.batch)
        batch["loss_mask"] = torch.zeros_like(batch["loss_mask"])
        with mock.patch.object(
            self.model, "_forward_draft_blocks", side_effect=AssertionError("forward")
        ):
            result = score_response_snapshot(
                self.model, batch, event_id=1, response_id="r"
            )
        self.assertEqual(result["costs"]["response_forwards"], 0)
        self.assertEqual(result["blocks"], [])
        with self.assertRaisesRegex(ValueError, "fixed event hash"):
            score_response_snapshot(
                self.model, self.batch, event_id=1, response_id="r", anchor_ids=[4]
            )


if __name__ == "__main__":
    unittest.main()
