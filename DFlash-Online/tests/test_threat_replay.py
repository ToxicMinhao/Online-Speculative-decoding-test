"""CPU numerical tests; native Flex/DDP parity remains a separate GPU gate."""
from copy import deepcopy
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
import torch.nn.functional as F

from experiments.threatened_repair_replay import replay
from online_feedback.rng import capture_rng
from experiments.lost_repair.training_probe import _rng_equal
from test_threat_core import tag


class SparseObjectiveTests(unittest.TestCase):
    def test_work_context_can_label_memory_without_duplicate_keywords(self):
        from experiments.threatened_repair_replay.work import Ledger
        with tempfile.TemporaryDirectory() as folder:
            ledger = Ledger(Path(folder)/"work.jsonl")
            with ledger.scope(category="disposable", state="U"):
                with replay._work(ledger, "response_forward", {"category": "memory", "window": 8},
                                  layout_id="a:0") as receipt:
                    self.assertEqual(receipt["category"], "memory")
                    ledger.tick("vocabulary_head_calls")
                self.assertEqual(ledger.context, {"category": "disposable", "state": "U"})
            self.assertEqual(ledger.summary()["by_category"], {"memory": {"vocabulary_head_calls": 1}})
            self.assertEqual(ledger.completed["response_forward"], 1)
            ledger.close()

    def test_forward_kl_teacher_detached_and_unselected_direct_gradient_zero(self):
        logits = torch.tensor([[[[.5, -.5], [1., 0.], [0., 2.]]]],
                              dtype=torch.bfloat16, requires_grad=True)
        teacher = torch.tensor([[[[0., 1.], [0., 1.], [1., 0.]]]],
                               dtype=torch.bfloat16, requires_grad=True)
        weights = torch.tensor([[[0., 1., 0.]]])
        numerator = replay.teacher_kl_terms(logits, teacher, weights)
        loss = .1 * numerator / (weights.sum() + 1e-6)
        expected = .1 * F.kl_div(F.log_softmax(logits, -1), F.softmax(teacher.detach(), -1),
            reduction="none").sum(-1)[0, 0, 1].float() / (1 + 1e-6)
        self.assertEqual(float(loss.detach()), float(expected.detach()))
        loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertGreater(float(logits.grad[0, 0, 1].norm()), 0)
        self.assertEqual(float(logits.grad[0, 0, [0, 2]].abs().sum()), 0)

    def test_global_normalization_is_not_response_average(self):
        logits = [torch.tensor([[[[0., 2.]]]], requires_grad=True),
                  torch.tensor([[[[2., 0.]]]], requires_grad=True)]
        teacher = torch.tensor([[[[2., 0.]]]])
        nums = [replay.teacher_kl_terms(x, teacher, torch.tensor([[[w]]]))
                for x, w in zip(logits, (1., .1))]
        loss = sum(nums) / 1.100001
        wrong = (nums[0]/1.000001 + nums[1]/.100001)/2
        self.assertNotEqual(float(loss.detach()), float(wrong.detach()))

    def test_masks_exclude_seed_gap_and_after_eos(self):
        labels = torch.tensor([[[0, 2, 5, 2, 2]]])
        weights = torch.tensor([[[0., 1., .8, .7, 0.]]])
        tags = [dict(tag("r", slot=1), canonical_token=2),
                dict(tag("r", slot=2), canonical_token=5)]
        selected = {replay.tag_key(tags[1])}
        mask = replay.sparse_weights(weights, labels, tags, selected, eos_token_ids=(5,))
        self.assertEqual(mask.tolist(), [[[0., 0., float(weights[0, 0, 2]), 0., 0.]]])
        self.assertFalse(mask.requires_grad)
        for slot in (0, 3, 4):
            invalid = dict(tag("r", slot=slot), canonical_token=int(labels[0, 0, slot]))
            with self.assertRaises(ValueError):
                replay.sparse_weights(weights, labels, [invalid], {replay.tag_key(invalid)}, eos_token_ids=(5,))

    def test_native_argmax_wrong_tie_and_matched_eos_boundary(self):
        labels = torch.tensor([[[0, 2, 3, 2]]])
        weights = torch.tensor([[[0., 1., .8, .7]]])
        logits = torch.full((1, 1, 4, 4), -1., dtype=torch.bfloat16)
        logits[0, 0, 1, 2] = 1
        logits[0, 0, 2, 3] = 1
        tags = [dict(tag("r", slot=2), canonical_token=3)]
        row, = replay.observation_rows(logits, labels, weights, tags, anchor_offset=0, eos_token_ids=(3,))
        self.assertEqual((row["accepted_prefix"], row["valid_proposals"]), (2, 2))
        logits[0, 0, 2, 0] = 1
        row, = replay.observation_rows(logits, labels, weights, tags, anchor_offset=0, eos_token_ids=(3,))
        self.assertEqual(row["accepted_prefix"], 1)
        self.assertTrue(row["wrong_tie"])

    def test_shared_denominator_is_exact_for_adversarial_permutations(self):
        tags = [tag(str(i), slot=j) for i, j in enumerate((1, 15, 2, 14))]
        expected = replay.replay_denominator(tags, device="cpu")
        for order in (tags[::-1], tags[1:] + tags[:1], [tags[i] for i in (0, 2, 1, 3)]):
            self.assertTrue(torch.equal(expected, replay.replay_denominator(order, device="cpu")))

    def test_zero_addition_leaves_native_bucket_and_version_untouched(self):
        p = torch.nn.Parameter(torch.ones(3, dtype=torch.bfloat16))
        p.grad = torch.tensor([1., 2., 3.], dtype=torch.bfloat16)
        original, version = p.grad, p.grad._version
        result = replay.add_gradient_sum([p], [torch.zeros_like(p)])
        self.assertIs(p.grad, original)
        self.assertEqual(p.grad._version, version)
        self.assertEqual(result["realized_added_gradient_norm"], 0)


class TinyLayout(torch.nn.Module):
    block_size, objective_chunk_blocks, loss_dtype, eos_token_ids = 16, 8, torch.bfloat16, ()

    def __init__(self):
        super().__init__()
        self.hidden = torch.nn.Parameter(torch.randn(1, 9, 16, 4, dtype=torch.bfloat16))
        self.lm_head = torch.nn.Linear(4, 6, bias=False, dtype=torch.bfloat16)
        self.lm_head.weight.requires_grad_(False)
        self.costs = {"existing": 9.}
        self._cached_seq_len = 17

    def _sample_anchor_positions(self, *args, **kwargs):
        raise AssertionError("Replay must not draw anchors")

    def _forward_draft_blocks(self, *args, **kwargs):
        anchors, keep = self._sample_anchor_positions()
        torch.rand(1)
        self._cached_seq_len = 99
        self.costs["temporary"] = 5
        return anchors, keep, self.hidden * 1

    def _aligned_target_hidden(self, target, indices):
        # Same predecessor-row alignment as the native objective.
        return target[:, (indices - 1).clamp_min(0).reshape(-1)].reshape(1, 9, 16, 4)


class ReplayIntegrationTests(unittest.TestCase):
    def fixture(self):
        torch.manual_seed(4)
        model = TinyLayout()
        p = model.hidden
        p.grad = torch.full_like(p, .02)
        trainer = SimpleNamespace(bundle=SimpleNamespace(model=model),
            backend=SimpleNamespace(optimizer=SimpleNamespace(model_params=[p])),
            cfg=SimpleNamespace(data=SimpleNamespace(max_length=4096)))
        a, b = tag("a", slot=1), tag("b", slot=1, start=8, ordinal=8)
        # CPU fixture weights come from the same FP32 tensor as metadata.
        labels = torch.full((1, 9, 16), 2, dtype=torch.long)
        weights = torch.exp(-torch.arange(16).sub(1).clamp_min(0).float()/7).reshape(1, 1, 16).expand(1, 9, 16).clone()
        weights[..., 0] = 0
        anchors = torch.arange(9).reshape(1, 9)
        keep = torch.ones_like(anchors, dtype=torch.bool)
        indices = torch.arange(9*16).reshape(1, 9, 16)
        batch = dict(input_ids=labels.reshape(1, -1), loss_mask=torch.ones(1, 144),
            hidden_states=torch.ones(1, 144, 4))
        raw = dict(prompt_length=0,
            target_last_hidden_states=torch.randn(1, 144, 4, dtype=torch.bfloat16))
        inputs = (raw, batch, anchors, keep, labels, weights, indices)
        layouts = [dict(layout_id=t["layout_id"], request_id=t["request_id"],
            reference_presentation_id=0, candidate_tags=[t], binding={"pass_index": 0}) for t in (a, b)]
        return model, trainer, inputs, layouts, a, b

    def saved_capture_fixture(self, directory):
        """Real capture schema and normalizer; no mocked feature preparation."""
        from online_feedback.objective import TargetObjectiveModel

        model, trainer, _, _, _, _ = self.fixture()
        model.loss_decay_gamma = 7
        model._block_labels_and_weights = TargetObjectiveModel._block_labels_and_weights.__get__(model)
        prompt, length = 2, 32
        ids = torch.full((1, length), 2, dtype=torch.long)
        mask = torch.ones_like(ids)
        mask[:, :prompt] = 0
        # Distinct rows make an off-by-one teacher alignment detectable.
        teacher = (torch.arange(length*4).reshape(1, length, 4).float()/32).bfloat16()
        raw = dict(record_id="capture", prompt_length=prompt, input_ids=ids,
            loss_mask=mask, hidden_states=torch.zeros(1, length, 4, dtype=torch.bfloat16),
            target_last_hidden_states=teacher,
            label_binding=dict(capture_backend="same_frozen_SGLang_native_capture",
                native_capture_validated=True,
                label_binding_scope="native_prefill_and_committed_verification",
                final_feature_logits_used_for_labels=False,
                input_ids=ids[0].tolist(), target_argmax_ids=ids[0, prompt:].tolist(),
                prompt_length=prompt, record_id="capture", label_mismatches=[]))
        feature_path = Path(directory)/"capture.pt"
        torch.save(raw, feature_path)
        anchors = torch.arange(prompt, prompt+9).reshape(1, 9)
        binding = dict(record_id="capture", presentation_id=0, pass_index=0,
            sampled_anchors=[dict(positions=anchors.tolist(), keep_mask=torch.ones_like(
                anchors, dtype=torch.bool).tolist())])
        selected = tag("capture", slot=1, start=0, ordinal=0)
        layout = dict(layout_id=selected["layout_id"], request_id="capture",
            reference_presentation_id=0, candidate_tags=[selected], binding=binding,
            feature_path=str(feature_path))
        return model, trainer, raw, layout, selected

    def test_real_capture_normalizer_keeps_teacher_separate_and_replay_aligns_it(self):
        with tempfile.TemporaryDirectory() as directory:
            model, trainer, raw, layout, selected = self.saved_capture_fixture(directory)
            _, batch, _, _, _, _, indices = replay._layout_inputs(model, layout, 4096)
            self.assertEqual(set(batch), {"input_ids", "loss_mask", "hidden_states"})
            self.assertNotIn("target_last_hidden_states", batch)
            native_before, rng = model.hidden.grad.clone(), capture_rng()
            teacher_inputs = []
            original = model._aligned_target_hidden

            def record_teacher(states, label_indices):
                aligned = original(states, label_indices)
                teacher_inputs.append((states.detach().clone(), aligned.detach().clone()))
                return aligned

            with mock.patch.object(model, "_aligned_target_hidden", side_effect=record_teacher):
                receipt = replay.add_replay_gradients(trainer, [layout], [selected])
            states, aligned = teacher_inputs[0]
            self.assertTrue(torch.equal(states, raw["target_last_hidden_states"]))
            self.assertEqual(states.dtype, torch.bfloat16)
            self.assertFalse(states.requires_grad)
            expected = raw["target_last_hidden_states"][:, (indices-1).clamp_min(0).reshape(-1)]
            self.assertTrue(torch.equal(aligned, expected.reshape(1, 9, 16, 4)))
            self.assertEqual((receipt["response_forwards"], receipt["response_backwards"]), (1, 1))
            self.assertGreater(receipt["realized_added_gradient_norm"], 0)
            self.assertTrue(torch.equal(model.hidden.grad[:, 1:], native_before[:, 1:]))
            self.assertTrue(_rng_equal(rng, capture_rng()))

    def test_real_capture_padded_zero_executes_teacher_path_without_changing_native_grad(self):
        with tempfile.TemporaryDirectory() as directory:
            model, trainer, _, layout, _ = self.saved_capture_fixture(directory)
            original, values, version = model.hidden.grad, model.hidden.grad.clone(), model.hidden.grad._version
            receipt = replay.add_replay_gradients(trainer, [layout], [], force_padded_zero=True)
            self.assertEqual((receipt["response_forwards"], receipt["response_backwards"]), (1, 1))
            self.assertEqual(receipt["teacher_head_calls"], 4)  # two chunks and recomputation
            self.assertEqual(receipt["auxiliary_loss"], 0)
            self.assertIs(model.hidden.grad, original)
            self.assertEqual(model.hidden.grad._version, version)
            self.assertTrue(torch.equal(model.hidden.grad, values))

    def test_independent_observer_uses_same_full_head_tensors_without_extra_forwards(self):
        model, _, inputs, layouts, _, _ = self.fixture()
        head_calls = []
        existing = model.lm_head.register_forward_hook(lambda *args: head_calls.append(1))
        rng = capture_rng()
        hooks = len(model.lm_head._forward_hooks)
        with mock.patch.object(replay, "_layout_inputs", return_value=inputs):
            result = replay.qualify_memory_observer(model, layouts, max_length=4096)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(len(head_calls), 4)
        self.assertEqual(result["response_forwards"], 2)
        self.assertEqual(result["additional_head_forwards"], 0)
        self.assertEqual(len(model.lm_head._forward_hooks), hooks)
        self.assertTrue(_rng_equal(rng, capture_rng()))
        existing.remove()

    def test_independent_observer_detects_wrong_new_scalar_and_removes_hook(self):
        model, _, inputs, layouts, _, _ = self.fixture()
        original = replay.observation_rows
        def wrong(*args, **kwargs):
            rows = original(*args, **kwargs)
            for row in rows:
                row["margin"] += 1
            return rows
        with mock.patch.object(replay, "_layout_inputs", return_value=inputs), \
                mock.patch.object(replay, "observation_rows", side_effect=wrong):
            result = replay.qualify_memory_observer(model, layouts, max_length=4096)
        self.assertEqual(result["status"], "failed")
        self.assertEqual({row["field"] for row in result["mismatches"]}, {"margin"})
        self.assertEqual(len(model.lm_head._forward_hooks), 0)

    def test_full_union_backward_zero_layout_rng_and_head_partial_chunk(self):
        model, trainer, inputs, layouts, a, b = self.fixture()
        before = model.hidden.grad.clone()
        rng, costs, mode = capture_rng(), deepcopy(model.costs), model.training
        shapes = []
        hook = model.lm_head.register_forward_pre_hook(lambda m, args: shapes.append(tuple(args[0].shape)))
        with mock.patch.object(replay, "_layout_inputs", return_value=inputs):
            receipt = replay.add_replay_gradients(trainer, layouts, [a])
        hook.remove()
        self.assertEqual((receipt["response_forwards"], receipt["response_backwards"]), (2, 2))
        self.assertEqual(receipt["logical_head_chunks"], 4)
        self.assertTrue(all(shape[1] in (8, 1) for shape in shapes))
        self.assertGreater(receipt["realized_added_gradient_norm"], 0)
        self.assertTrue(torch.equal(model.hidden.grad[:, 8], before[:, 8]))
        self.assertTrue(_rng_equal(rng, capture_rng()))
        self.assertEqual((model.costs, model.training, model._cached_seq_len), (costs, mode, 17))

    def test_padded_zero_runs_every_layout_without_mutating_native_gradients(self):
        model, trainer, inputs, layouts, a, b = self.fixture()
        before, version = model.hidden.grad.clone(), model.hidden.grad._version
        with mock.patch.object(replay, "_layout_inputs", return_value=inputs):
            receipt = replay.add_replay_gradients(trainer, layouts, [], force_padded_zero=True)
        self.assertEqual(receipt["response_backwards"], 2)
        self.assertEqual(receipt["auxiliary_loss"], 0)
        self.assertTrue(torch.equal(model.hidden.grad, before))
        self.assertEqual(model.hidden.grad._version, version)

    def test_disabled_paths_do_not_touch_model_or_reconstruct_gradient(self):
        model, trainer, inputs, layouts, a, b = self.fixture()
        with mock.patch.object(replay, "_layout_inputs", side_effect=AssertionError("forbidden")):
            for tags, coefficient in (([], .1), ([a], 0.)):
                result = replay.add_replay_gradients(trainer, layouts, tags, coefficient=coefficient)
                self.assertEqual(result["status"], "direct_native")

    def test_shared_denominator_and_immutable_masks_survive_recomputation(self):
        model, trainer, inputs, layouts, a, b = self.fixture()
        denominator = replay.replay_denominator([a], device="cpu")
        original = deepcopy(layouts)
        with mock.patch.object(replay, "_layout_inputs", return_value=inputs):
            result = replay.add_replay_gradients(trainer, layouts, [b], denominator=denominator)
        self.assertEqual(layouts, original)
        self.assertEqual(result["denominator"], float(denominator))
        self.assertEqual(result["recomputed_head_chunks"], 4)


if __name__ == "__main__":
    unittest.main()
