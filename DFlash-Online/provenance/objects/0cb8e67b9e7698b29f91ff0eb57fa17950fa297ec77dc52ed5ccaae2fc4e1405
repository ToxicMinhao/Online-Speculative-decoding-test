"""Real tiny DFlash forwards for the opt-in objective and fixed event reference.

These CPU tests use upstream anchor sampling, attention geometry and trainable
draft blocks. They qualify numerical/learning contracts, not GPU generation.
"""

import copy
import unittest
from unittest import mock

import torch
import torch.nn.functional as F
from transformers import Qwen3Config

from specforge.algorithms.common.dflash_family_model import OnlineDFlashModel
from specforge.modeling.draft.dflash import DFlashDraftModel

from online_feedback.objective import TargetObjectiveModel
from online_feedback.projected_target import projected_targets
from online_feedback.rng import capture_rng


def tiny_author(dtype=torch.bfloat16, chunk_blocks=2):
    """Use actual Qwen3/DFlash blocks, with a small frozen head/embedding."""
    torch.manual_seed(41)
    config = Qwen3Config(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=29,
        max_position_embeddings=128,
        rms_norm_eps=1e-5,
        tie_word_embeddings=False,
    )
    config.layer_types = ["full_attention"]
    config.block_size = 4
    config.num_target_layers = 2
    config.dflash_config = {"mask_token_id": 0}
    config._attn_implementation = "eager"
    draft = DFlashDraftModel(config).to(dtype=dtype)
    head = torch.nn.Linear(16, 29, bias=False).to(dtype=dtype).requires_grad_(False)
    embedding = torch.nn.Embedding(29, 16).to(dtype=dtype).requires_grad_(False)
    return OnlineDFlashModel(
        draft_model=draft,
        target_lm_head=head,
        target_embed_tokens=embedding,
        mask_token_id=0,
        block_size=4,
        attention_backend="eager",
        num_anchors=4,
        loss_decay_gamma=2.0,
        objective_chunk_blocks=chunk_blocks,
        teacher_metrics=False,
    )


def response_batch(model):
    """A four-token prompt followed by eight supervised answer tokens."""
    width = len(model.draft_model.target_layer_ids) * 16
    generator = torch.Generator().manual_seed(83)
    return {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]),
        "hidden_states": torch.randn(
            1, 12, width, generator=generator,
            dtype=next(model.draft_model.parameters()).dtype,
        ),
        "loss_mask": torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]]),
    }


def direct_hard_ce(model, batch, anchors, keep, hidden):
    """Independent position loop: preserve upstream masks and weighted sum."""
    hidden = hidden.reshape(1, anchors.shape[1], model.block_size, -1)
    numerator = hidden.new_zeros((), dtype=torch.float32)
    denominator = numerator.clone()
    supervised = []
    for block in range(anchors.shape[1]):
        if not keep[0, block]:
            continue
        for offset in range(1, model.block_size):
            position = int(anchors[0, block]) + offset
            if position >= batch["input_ids"].shape[1]:
                continue
            if batch["loss_mask"][0, position] == 0:
                continue
            logits = F.linear(hidden[0, block, offset].float(), model.lm_head.weight.float())
            token = batch["input_ids"][0, position]
            weight = torch.exp(torch.tensor(-(offset - 1) / 2.0))
            numerator = numerator + F.cross_entropy(logits[None], token[None]) * weight
            denominator = denominator + weight
            supervised.append(position)
    return numerator, denominator, supervised


def parameter_gradients(model):
    return {
        name: parameter.grad.detach().float().clone()
        for name, parameter in model.draft_model.named_parameters()
        if parameter.grad is not None
    }


class ObjectiveIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_hard_ce_matches_independent_fp32_loss_and_author_masks(self):
        model = TargetObjectiveModel(tiny_author(dtype=torch.float32), "hard_ce")
        batch = response_batch(model)
        # Exercise a partially clipped tail, a masked source position and a
        # disabled block independently of the random sampler.
        batch["loss_mask"][0, 6] = 0
        anchors = torch.tensor([[4, 8, 10, 0]])
        keep = torch.tensor([[True, True, True, False]])
        with mock.patch.object(model, "_sample_anchor_positions", return_value=(anchors, keep)):
            _, _, hidden = model._forward_draft_blocks(**batch)
            numerator, denominator, supervised = direct_hard_ce(model, batch, anchors, keep, hidden)
            expected = numerator / denominator
            expected.backward()
            expected_gradients = parameter_gradients(model)
            model.zero_grad(set_to_none=True)
            actual, _, metrics = model(**batch)
            actual.backward()

        self.assertEqual(supervised, [5, 7, 9, 10, 11, 11])
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
        actual_numerator, actual_denominator = metrics["loss_terms"]
        torch.testing.assert_close(actual_numerator, numerator, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(actual_denominator, denominator, atol=2e-6, rtol=2e-6)
        for name, gradient in parameter_gradients(model).items():
            torch.testing.assert_close(gradient, expected_gradients[name], atol=2e-6, rtol=2e-5)

    def test_projected_and_control_share_anchors_and_training_rng(self):
        author = tiny_author()
        control = TargetObjectiveModel(copy.deepcopy(author), "hard_ce")
        projected = TargetObjectiveModel(copy.deepcopy(author), "projected")
        projected.begin_event()
        batch = response_batch(control)
        recorded = []
        rng_states = []
        for model in (control, projected):
            sampled = []
            sampler = model._sample_anchor_positions

            def record_sample(*args, **kwargs):
                result = sampler(*args, **kwargs)
                sampled.append(tuple(value.clone() for value in result))
                return result

            torch.manual_seed(17)
            with mock.patch.object(model, "_sample_anchor_positions", side_effect=record_sample):
                loss, _, _ = model(**batch)
                loss.backward()
            self.assertEqual(len(sampled), 1)
            recorded.append(sampled[0])
            rng_states.append(torch.get_rng_state())

        for control_value, projected_value in zip(*recorded):
            self.assertTrue(torch.equal(control_value, projected_value))
        self.assertTrue(torch.equal(*rng_states))

    def test_reference_is_fixed_within_event_and_recaptured_next_event(self):
        model = TargetObjectiveModel(tiny_author(), "projected")
        batch = response_batch(model)
        model.begin_event()
        first_reference = model.reference
        reference_parameters = {
            name: value.detach().clone()
            for name, value in first_reference.model.draft_model.state_dict().items()
        }
        with torch.no_grad():
            for parameter in model.draft_model.parameters():
                parameter.add_(0.125)
        for name, value in first_reference.model.draft_model.state_dict().items():
            self.assertTrue(torch.equal(value, reference_parameters[name]), name)
        model(**batch)
        self.assertIs(model.reference, first_reference)
        model.end_event()
        self.assertIsNone(model.reference)
        model.begin_event()
        self.assertIsNot(model.reference, first_reference)
        for name, value in model.reference.model.draft_model.state_dict().items():
            self.assertTrue(torch.equal(value, model.draft_model.state_dict()[name]), name)

    def test_projected_loss_still_uses_pre_event_targets_after_current_weights_change(self):
        model = TargetObjectiveModel(tiny_author(dtype=torch.float32), "projected")
        batch = response_batch(model)
        anchors = torch.tensor([[4, 7, 10]])
        keep = torch.ones_like(anchors, dtype=torch.bool)
        model.begin_event()
        reference_hidden = model.reference.hidden(
            batch["input_ids"], batch["hidden_states"], batch["loss_mask"], anchors, keep
        ).reshape(1, 3, 4, -1)
        fixed_targets = {}
        for block, anchor in enumerate(anchors[0].tolist()):
            for offset in range(1, 4):
                position = anchor + offset
                if position < batch["input_ids"].shape[1]:
                    reference_logits = model.lm_head(reference_hidden[0, block, offset])
                    fixed_targets[block, offset] = projected_targets(
                        reference_logits, batch["input_ids"][0, position]
                    )

        # Simulate a preceding optimizer step in the same event. The target
        # should remain tied to the event's old draft, not these new weights.
        with torch.no_grad():
            model.draft_model.fc.weight.add_(0.15)
        with mock.patch.object(model, "_sample_anchor_positions", return_value=(anchors, keep)):
            _, _, hidden = model._forward_draft_blocks(**batch)
            hidden = hidden.reshape(1, 3, 4, -1)
            numerator = torch.zeros(())
            denominator = torch.zeros(())
            for (block, offset), targets in fixed_targets.items():
                logits = model.lm_head(hidden[0, block, offset]).float()
                weight = torch.exp(torch.tensor(-(offset - 1) / 2.0))
                numerator = numerator - (targets * logits.log_softmax(-1)).sum() * weight
                denominator = denominator + weight
            actual, _, _ = model(**batch)
        torch.testing.assert_close(actual, numerator / denominator, atol=2e-6, rtol=2e-6)

    def test_reference_uses_explicit_anchors_without_rng_or_gradients(self):
        model = TargetObjectiveModel(tiny_author(), "projected")
        batch = response_batch(model)
        model.begin_event()
        anchors = torch.tensor([[4, 7, 9]])
        keep = torch.ones_like(anchors, dtype=torch.bool)
        before = capture_rng()
        hidden = model.reference.hidden(
            batch["input_ids"], batch["hidden_states"], batch["loss_mask"], anchors, keep
        )
        after = capture_rng()
        self.assertFalse(hidden.requires_grad)
        self.assertEqual(hidden.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(before["torch"], after["torch"]))
        self.assertEqual(before["python"], after["python"])
        self.assertEqual(before["numpy"][1].tolist(), after["numpy"][1].tolist())

    def test_projected_gradients_reach_draft_without_training_reference_or_head(self):
        model = TargetObjectiveModel(tiny_author(), "projected")
        model.begin_event()
        reference_before = {
            name: value.clone() for name, value in model.reference.model.state_dict().items()
        }
        loss, _, _ = model(**response_batch(model))
        loss.backward()
        gradients = parameter_gradients(model)
        self.assertTrue(gradients)
        self.assertGreater(sum(value.abs().sum().item() for value in gradients.values()), 0.0)
        self.assertTrue(any("layers." in name and value.abs().sum() > 0 for name, value in gradients.items()))
        self.assertIsNone(model.lm_head.weight.grad)
        self.assertIsNone(model.embed_tokens.weight.grad)
        for name, parameter in model.reference.model.named_parameters():
            self.assertFalse(parameter.requires_grad, name)
            self.assertIsNone(parameter.grad, name)
        for name, value in model.reference.model.state_dict().items():
            self.assertTrue(torch.equal(value, reference_before[name]), name)

    def test_reference_is_excluded_from_optimizer_and_checkpoint_names(self):
        model = TargetObjectiveModel(tiny_author(), "projected")
        state_keys = set(model.state_dict())
        parameter_ids = {id(value) for value in model.parameters()}
        model.begin_event()
        self.assertEqual(state_keys, set(model.state_dict()))
        self.assertEqual(parameter_ids, {id(value) for value in model.parameters()})
        draft_reference_ids = {id(value) for value in model.reference.model.draft_model.parameters()}
        self.assertFalse(parameter_ids & draft_reference_ids)

    def test_chunking_preserves_loss_and_gradients(self):
        for objective in ("hard_ce", "projected"):
            for dtype in (torch.float32, torch.bfloat16):
                with self.subTest(objective=objective, dtype=dtype):
                    author = tiny_author(dtype=dtype, chunk_blocks=1)
                    chunked = TargetObjectiveModel(copy.deepcopy(author), objective)
                    unchunked = TargetObjectiveModel(copy.deepcopy(author), objective)
                    unchunked.objective_chunk_blocks = 128
                    if objective == "projected":
                        chunked.begin_event()
                        unchunked.begin_event()
                    losses, gradients = [], []
                    for model in (chunked, unchunked):
                        torch.manual_seed(17)
                        loss, _, _ = model(**response_batch(model))
                        loss.backward()
                        losses.append(loss.detach())
                        gradients.append(parameter_gradients(model))
                    torch.testing.assert_close(*losses, atol=2e-6, rtol=2e-6)
                    tolerance = 2e-3 if dtype == torch.bfloat16 else 2e-6
                    for name in gradients[0]:
                        torch.testing.assert_close(
                            gradients[0][name], gradients[1][name], atol=tolerance, rtol=tolerance
                        )


if __name__ == "__main__":
    unittest.main()
