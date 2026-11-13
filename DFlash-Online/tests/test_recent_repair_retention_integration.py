"""Shared native K+ forwards and actual tiny-draft backward/optimizer checks."""

import copy
import unittest
from unittest import mock

import torch

from online_feedback.kplus import install_kplus
from online_feedback.objective import TargetObjectiveModel
from online_feedback.recent_repair_retention import install_recent_repair_retention
from test_kplus_objective import mixed_logits
from test_objective_integration import tiny_author, response_batch


class RetentionIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_full_chunked_forward_preserves_eleven_native_terms_and_reduction(self):
        model = TargetObjectiveModel(tiny_author(chunk_blocks=8), "osd_first_error",
                                     loss_precision="bfloat16")
        model.block_size = 16
        model.lm_head = torch.nn.Identity()
        install_kplus(model)
        extension = install_recent_repair_retention(model)
        logits, teacher, labels, weights = mixed_logits()
        anchors = torch.arange(17).reshape(1, -1) * 16
        keep = torch.ones_like(anchors, dtype=torch.bool)
        # Synthetic assigned first-error rows exercise both K+ classes and the
        # final partial head chunk without changing full-block attention.
        rows = [dict(canonical_start=int(anchors[0, block]), proposal_slot=1 + block % 5,
                     canonical_token=int(labels[0, block, 1 + block % 5]),
                     parent_margin=2.4,
                     position_weight=float(weights[0, block, 1 + block % 5]))
                for block in (0, 8)]
        extension.set_presentation(rows, request_id="r", presentation_id=0,
                                   prompt_length=0, expected_anchors=anchors, expected_keep=keep)
        logits = logits.requires_grad_()
        model.begin_event()
        original_reduce = model.first_error_redistribution.reduce_response
        seen = []

        def reducer(terms):
            seen.append(len(terms))
            return original_reduce(terms)

        with mock.patch.object(model, "_forward_draft_blocks", return_value=(
                anchors, keep, logits.reshape(1, -1, 31))):
            with mock.patch.object(model, "_block_labels_and_weights", return_value=(labels, weights, labels)):
                with mock.patch.object(model, "_aligned_target_hidden", return_value=teacher):
                    with mock.patch.object(model.first_error_redistribution, "reduce_response", side_effect=reducer):
                        loss, _, metrics = model(labels.reshape(1, -1), torch.zeros(1),
                                                 torch.ones_like(labels.reshape(1, -1)), teacher)
                        saved = {k: v.clone() for k, v in metrics["sum_metrics"].items()}
                        loss.backward()
        self.assertEqual(seen, [11])
        sums = metrics["sum_metrics"]
        self.assertFalse(set(sums) & set(metrics["ratio_metrics"]))
        self.assertEqual(int(sums["retention_assigned_count"]), 2)
        self.assertEqual(int(sums["retention_firing_count"]), 2)
        self.assertEqual(int(sums["retention_response_count"]), 1)
        expected = sums["kplus_N_new"] / sums["loss_denominator"] + (
            0.1 * sums["retention_hinge_numerator"] / sums["loss_denominator"])
        torch.testing.assert_close(loss, expected, rtol=0, atol=0)
        self.assertGreater(float(logits.grad.abs().sum()), 0)
        for key in saved:
            torch.testing.assert_close(saved[key], sums[key], rtol=0, atol=0)

    def test_disabled_and_zero_coefficient_match_native_tiny_draft_and_adam_exactly(self):
        author = tiny_author()
        baseline = TargetObjectiveModel(author, "osd_first_error", loss_precision="bfloat16")
        install_kplus(baseline)
        batch = response_batch(baseline)
        batch["target_last_hidden_states"] = torch.randn(1, 12, 16, dtype=torch.bfloat16)
        anchors = torch.tensor([[4, 7, 9, 10]])
        keep = torch.ones_like(anchors, dtype=torch.bool)
        results = []
        for setting in (None, {"enabled": False}, {"coefficient": 0.0}):
            model = copy.deepcopy(baseline)
            if setting is not None:
                extension = install_recent_repair_retention(model, **setting)
                extension.set_presentation([
                    dict(canonical_start=0, proposal_slot=1, canonical_token=6,
                         parent_margin=2.4, position_weight=1.0)], request_id="r", presentation_id=0,
                                           prompt_length=4, expected_anchors=anchors,
                                           expected_keep=keep)
            optimizer = torch.optim.AdamW(model.draft_model.parameters(), lr=3e-4,
                                          weight_decay=.01, foreach=False)
            model.begin_event()
            torch.manual_seed(17)
            with mock.patch.object(model, "_sample_anchor_positions", return_value=(anchors, keep)):
                loss, _, metrics = model(**batch)
                loss.backward()
            gradients = {n: p.grad.clone() for n, p in model.draft_model.named_parameters()}
            torch.nn.utils.clip_grad_norm_(model.draft_model.parameters(), 1.0)
            optimizer.step()
            results.append((loss.detach(), metrics["sum_metrics"], gradients,
                            copy.deepcopy(model.draft_model.state_dict()),
                            copy.deepcopy(optimizer.state_dict()), torch.get_rng_state()))
        for result in results[1:]:
            self.assert_nested_equal(results[0], result)

    def assert_nested_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_nested_equal(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for x, y in zip(left, right):
                self.assert_nested_equal(x, y)
        else:
            self.assertEqual(left, right)

    def test_active_hinge_reaches_actual_trainable_draft(self):
        model = TargetObjectiveModel(tiny_author(), "osd_first_error", loss_precision="bfloat16")
        install_kplus(model)
        extension = install_recent_repair_retention(model)
        batch = response_batch(model)
        batch["target_last_hidden_states"] = torch.randn(1, 12, 16, dtype=torch.bfloat16)
        anchors = torch.tensor([[4, 7, 9, 10]])
        keep = torch.ones_like(anchors, dtype=torch.bool)
        extension.set_presentation([
            dict(canonical_start=0, proposal_slot=1, canonical_token=6,
                 parent_margin=2.4, position_weight=1.0)],
            request_id="fixture", presentation_id=0, prompt_length=4,
            expected_anchors=anchors, expected_keep=keep)
        model.begin_event()
        captured = []
        original = extension.chunk_terms

        def capture(*args):
            output = original(*args)
            captured.append(output[0])
            return output

        with mock.patch.object(model, "_sample_anchor_positions", return_value=(anchors, keep)):
            with mock.patch.object(extension, "chunk_terms", side_effect=capture):
                _, _, metrics = model(**batch)
                hinge = sum(captured)
                self.assertGreater(float(hinge.detach()), 0)
                hinge.backward()
        self.assertGreater(int(metrics["sum_metrics"]["retention_firing_count"]), 0)
        total = sum(float(p.grad.float().square().sum()) for p in model.draft_model.parameters()
                    if p.grad is not None)
        self.assertGreater(total, 0)
        self.assertIsNone(model.lm_head.weight.grad)
        self.assertIsNone(model.embed_tokens.weight.grad)


if __name__ == "__main__":
    unittest.main()
