"""Metadata and CPU numerical differential checks; no model download or GPU use."""
import copy
import importlib.util
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

try:
    from . import author_training as training
except ImportError:
    import author_training as training

HAS_TORCH = importlib.util.find_spec("torch") is not None
RELEASED, UPDATED, TARGET = "a" * 64, "b" * 64, "c" * 64


def response(turn, checkpoint=RELEASED, output_length=14, prefix_length=20):
    return dict(episode_id="episode", record_id=f"episode:turn{turn}", turn=turn,
                input_ids=[i % 19 for i in range(prefix_length)], input_token_count=prefix_length,
                output_ids=[(i + turn) % 19 for i in range(output_length)], output_cap=256,
                capture_checkpoint_sha256=checkpoint, target_id="target", target_identity_sha256=TARGET)


def upstream_path():
    value = os.environ.get("ADAFLASH_SOURCE_ROOT")
    if value:
        return Path(value) / "specforge/core/dflash.py"
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "research/adaflash_review_20260929/vendor/AdaFlash/specforge/core/dflash.py"
        if candidate.is_file():
            return candidate
    return None


def load_upstream():
    """Import unedited author core; only its annotation-only model import is stubbed."""
    import torch
    path = upstream_path()
    if path is None or not path.is_file():
        raise unittest.SkipTest("Set ADAFLASH_SOURCE_ROOT to pinned AdaFlash checkout for differential checks")
    # File identity is checked against the pinned source; do not silently compare
    # an arbitrary newer author implementation supplied by the environment.
    import hashlib
    expected = "d6a16d8ea8f9401e1e523f227945f228bc40d5feb021baff5fca620386416024"
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise AssertionError("AdaFlash core source differs from pinned 507b7aba")
    stub = types.ModuleType("specforge.modeling.draft.dflash")
    stub.DFlashDraftModel = torch.nn.Module
    spec = importlib.util.spec_from_file_location("_pinned_author_dflash", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict("sys.modules", {"specforge.modeling.draft.dflash": stub}):
        spec.loader.exec_module(module)
    return module


class MetadataTests(unittest.TestCase):
    def test_full_context_above_2048_preserved(self):
        r = response(9, prefix_length=10000, output_length=256)
        audit = training.audit_response(r)
        self.assertEqual(audit["real_sequence_length"], 10256)
        self.assertEqual(audit["padded_sequence_length"], 10265)
        self.assertEqual(audit["sampled_anchor_count"], 255)
        self.assertEqual(audit["sequence_sha256"], training.digest(r["input_ids"] + r["output_ids"]))
        r = response(9, prefix_length=31744, output_length=256)
        self.assertEqual(training.audit_response(r)["padded_sequence_length"], 32009)
        for r in (response(9, prefix_length=31745), response(9, output_length=257)):
            with self.assertRaises(ValueError):
                training.audit_response(r)

    def test_short_response_is_explicit_failure_not_silent_skip(self):
        for length in (0, 1):
            with self.assertRaisesRegex(ValueError, "no response was skipped"):
                training.audit_response(response(9, output_length=length))
        self.assertEqual(training.audit_response(response(9, output_length=2))["sampled_anchor_count"], 1)

    def test_ownership_masks_earlier_within_group_history(self):
        a = training.audit_response(response(17, UPDATED, prefix_length=20))
        b = training.audit_response(response(18, UPDATED, prefix_length=40))
        self.assertEqual(a["loss_mask_sha256"], training.digest([0] * 20 + [1] * 14 + [0] * 9))
        self.assertEqual(b["loss_mask_sha256"], training.digest([0] * 40 + [1] * 14 + [0] * 9))
        self.assertGreater(b["response_start"], a["response_end_exclusive"])

    def test_exact_chronology_and_tokens_are_required(self):
        rs = [response(t) for t in range(9, 17)]
        schedule = training.event_schedule(1, rs)
        self.assertEqual([r["turn"] for _, pair in schedule for r in pair], list(range(9, 17)) * 2)
        for invalid in (rs[:-1], rs[::-1], rs[:1] * 8):
            with self.assertRaises(ValueError):
                training.event_schedule(1, invalid)
        r = response(9)
        r["input_ids_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "hash differs"):
            training.audit_response(r)


@unittest.skipUnless(HAS_TORCH, "CPU torch environment required for numerical checks")
class NumericalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)

    def fixture(self):
        import torch
        from torch import nn
        import torch.nn.functional as F
        torch.manual_seed(41)

        class Draft(nn.Module):
            use_prob_head = False
            use_thresh_head = False

            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(8, 4)
                self.mix = nn.Linear(4, 4)
                self.forward_calls = 0

            def forward(self, *, position_ids, noise_embedding, target_hidden, attention_mask):
                self.forward_calls += 1
                context = self.fc(target_hidden)
                kv = torch.cat((context, noise_embedding), dim=1).unsqueeze(1)
                hidden = F.scaled_dot_product_attention(noise_embedding.unsqueeze(1), kv, kv,
                                                        attn_mask=attention_mask).squeeze(1)
                return self.mix(hidden + noise_embedding), None

        draft = Draft()
        embed = nn.Embedding(20, 4).requires_grad_(False).eval()
        head = nn.Linear(4, 20, bias=False).requires_grad_(False).eval()
        teacher_calls = []

        def teacher(ids):
            teacher_calls.append(ids.detach().clone())
            base = embed(ids)
            # Causal and nonconstant features: a current token cannot alter an
            # earlier teacher row. The final feature already has its own scale.
            causal = base.cumsum(1) / torch.arange(1, ids.shape[1] + 1, device=ids.device).view(1, -1, 1)
            return torch.cat((base, causal), dim=-1), causal

        return draft, embed, head, teacher, teacher_calls

    def make_trainer(self):
        draft, embed, head, teacher, calls = self.fixture()
        trainer = training.PersistentAuthorOSDTrainer(draft, embed, head, teacher,
            episode_id="episode", target_id="target", target_identity_sha256=TARGET,
            released_checkpoint_sha256=RELEASED, mask_token_id=19,
            attention_backend="sdpa", projection_chunk=7)
        return trainer, calls

    def tensors(self):
        import torch
        draft, embed, head, teacher, _ = self.fixture()
        ids = torch.tensor([[i % 19 for i in range(31)] + [19] * 9])
        hidden, final = teacher(ids)
        mask = torch.zeros_like(ids, dtype=torch.float32)
        mask[:, 8:31] = 1
        return draft, embed, head, ids, hidden, final, mask

    def test_pinned_author_sampler_mask_loss_and_gradients(self):
        import torch
        source = load_upstream()
        draft, embed, head, ids, hidden, final, mask = self.tensors()
        author_draft = copy.deepcopy(draft)
        author = source.OnlineDFlashModel(author_draft, head, embed, mask_token_id=19,
                                          block_size=10, attention_backend="sdpa", num_anchors=512,
                                          loss_decay_gamma=7.0, teacher_kd_temperature=1.0, teacher_kd_alpha=0.0)
        torch.manual_seed(17)
        want_anchors, want_keep = author._sample_anchor_positions(ids.shape[1], mask, ids.device)
        torch.manual_seed(17)
        anchors, keep = training.sample_anchor_positions(ids.shape[1], mask)
        self.assertTrue(torch.equal(anchors, want_anchors))
        self.assertTrue(torch.equal(keep, want_keep))
        self.assertTrue(torch.equal(training.create_attention_mask(anchors, keep, ids.shape[1]),
                                   source.create_dflash_sdpa_mask(anchors, keep, ids.shape[1], 10, ids.device)))
        torch.manual_seed(17)
        expected = author(ids, hidden, mask, teacher_hidden_states=final)[0]
        expected.backward()
        torch.manual_seed(17)
        actual, metadata = training.author_response_loss(draft, embed, head, ids, hidden, final, mask,
                                                          mask_token_id=19, projection_chunk=7)
        actual.backward()
        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)
        for (name, parameter), (other_name, other) in zip(draft.named_parameters(), author_draft.named_parameters()):
            self.assertEqual(name, other_name)
            torch.testing.assert_close(parameter.grad, other.grad, rtol=3e-6, atol=3e-7)
        self.assertEqual(draft.forward_calls, 1, "Head checkpointing must never recompute the backbone")
        self.assertEqual(metadata["anchor_positions"], want_anchors[0].tolist())
        self.assertTrue(all(8 < pos < 31 for pos in metadata["unique_supervised_position_ids"]))
        for labels, teacher_positions, binary, weights in zip(metadata["label_positions"], metadata["teacher_positions"],
                                                               metadata["binary_loss_mask"], metadata["position_weights"]):
            self.assertEqual(binary[0], 0)
            for k, (pos, ptr, active, weight) in enumerate(zip(labels, teacher_positions, binary, weights)):
                self.assertEqual(ptr, pos - 1)
                self.assertEqual(bool(active), k > 0 and 8 <= pos < 31)
                self.assertAlmostEqual(weight, float(torch.exp(torch.tensor(-(k - 1) / 7))) if active else 0, places=6)

    def test_bfloat16_port_preserves_native_kl_dtype(self):
        import torch
        source = load_upstream()
        draft, embed, head, ids, hidden, final, mask = self.tensors()
        draft, embed, head = draft.to(torch.bfloat16), embed.to(torch.bfloat16), head.to(torch.bfloat16)
        hidden, final = hidden.to(torch.bfloat16), final.to(torch.bfloat16)
        author_draft = copy.deepcopy(draft)
        author = source.OnlineDFlashModel(author_draft, head, embed, mask_token_id=19,
                                          block_size=10, attention_backend="sdpa", num_anchors=512,
                                          loss_decay_gamma=7.0, teacher_kd_temperature=1.0, teacher_kd_alpha=0.0)
        torch.manual_seed(19)
        expected = author(ids, hidden, mask, teacher_hidden_states=final)[0]
        expected.backward()
        torch.manual_seed(19)
        actual, _ = training.author_response_loss(draft, embed, head, ids, hidden, final, mask,
                                                   mask_token_id=19, projection_chunk=7)
        actual.backward()
        # Row chunks preserve the native bf16 KL, though differently shaped GEMMs
        # and gradient accumulation can differ at bf16 rounding precision.
        torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.001)
        for p, q in zip(draft.parameters(), author_draft.parameters()):
            torch.testing.assert_close(p.grad, q.grad, rtol=0.03, atol=0.0005)

    def test_mask_blocks_future_history_other_blocks_and_padding(self):
        import torch
        anchors, keep = torch.tensor([[5, 12]]), torch.tensor([[True, True]])
        mask = training.create_attention_mask(anchors, keep, 30)
        for block, anchor in enumerate((5, 12)):
            for q in range(block * 10, (block + 1) * 10):
                self.assertEqual(mask[0, 0, q].nonzero().flatten().tolist(),
                                 list(range(anchor)) + list(range(30 + block * 10, 40 + block * 10)))

    def test_teacher_target_is_previous_position_and_no_target_gradient(self):
        import torch
        draft, embed, head, ids, hidden, final, mask = self.tensors()
        torch.manual_seed(17)
        first, _ = training.author_response_loss(draft, embed, head, ids, hidden, final, mask, mask_token_id=19)
        wrong_final = final.roll(1, dims=1)
        torch.manual_seed(17)
        wrong, _ = training.author_response_loss(draft, embed, head, ids, hidden, wrong_final, mask, mask_token_id=19)
        self.assertNotAlmostEqual(float(first.detach()), float(wrong.detach()), places=5)
        first.backward()
        self.assertTrue(all(p.grad is None for p in list(embed.parameters()) + list(head.parameters())))

    def test_padding_features_never_influence_supervised_loss(self):
        import torch
        draft, embed, head, ids, hidden, final, mask = self.tensors()
        torch.manual_seed(17)
        a, _ = training.author_response_loss(draft, embed, head, ids, hidden, final, mask, mask_token_id=19)
        changed_hidden, changed_final = hidden.clone(), final.clone()
        changed_hidden[:, 31:] = 1000
        changed_final[:, 31:] = -1000
        torch.manual_seed(17)
        b, _ = training.author_response_loss(draft, embed, head, ids, changed_hidden, changed_final, mask, mask_token_id=19)
        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_persistent_optimizer_rng_eight_steps_and_new_episode_reset(self):
        import torch
        trainer, calls = self.make_trainer()
        initial_rng = trainer.generator.get_state().clone()
        u1 = trainer.train_event(1, [response(t) for t in range(9, 17)])
        optimizer = trainer.optimizer
        self.assertEqual(u1["optimizer_steps_this_event"], 8)
        self.assertEqual(len(calls), 16)
        self.assertTrue(all(c.shape[1] == 34 for c in calls), "Teacher must see real IDs only, without synthetic padding")
        self.assertEqual(trainer.draft.forward_calls, 16)
        self.assertFalse(torch.equal(initial_rng, trainer.generator.get_state()))
        rng_after_u1 = trainer.generator.get_state().clone()
        with self.assertRaisesRegex(ValueError, "Bind actual"):
            trainer.train_event(2, [response(t, UPDATED) for t in range(17, 25)])
        trainer.bind_checkpoint(1, UPDATED)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "state.pt"
            receipt = trainer.save_checkpoint_state(path, include_draft=False)
            state = torch.load(path, weights_only=True)
            self.assertFalse(receipt["includes_draft"])
            self.assertNotIn("draft", state)
            self.assertEqual(state["optimizer_steps"], 8)
            self.assertTrue(torch.equal(state["anchor_rng_state"], rng_after_u1))
            with self.assertRaisesRegex(ValueError, "fresh path"):
                trainer.save_checkpoint_state(path)
        u2 = trainer.train_event(2, [response(t, UPDATED) for t in range(17, 25)])
        self.assertIs(trainer.optimizer, optimizer)
        self.assertEqual(u2["optimizer_steps_total"], 16)
        self.assertEqual({int(s["step"]) for s in optimizer.state.values()}, {16})
        self.assertEqual(u2["response_order"], [f"episode:turn{t}" for t in range(17, 25)] * 2)
        fresh, _ = self.make_trainer()
        self.assertEqual(fresh.optimizer.state, {})
        self.assertTrue(torch.equal(fresh.generator.get_state(), initial_rng))
        self.assertEqual(u1["batches"][0]["presentations"][0]["anchor_positions"],
                         fresh.train_event(1, [response(t) for t in range(9, 17)])["batches"][0]["presentations"][0]["anchor_positions"])

    def test_preflight_rejects_wrong_generation_or_short_response_before_mutation(self):
        for change in ("checkpoint", "episode", "short"):
            trainer, calls = self.make_trainer()
            rs = [response(t) for t in range(9, 17)]
            if change == "checkpoint":
                rs[-1]["capture_checkpoint_sha256"] = UPDATED
            elif change == "episode":
                rs[-1]["episode_id"] = "other"
            else:
                rs[-1]["output_ids"] = [1]
            with self.subTest(change=change), self.assertRaises(ValueError):
                trainer.train_event(1, rs)
            self.assertEqual(calls, [])
            self.assertEqual(trainer.optimizer_steps, 0)
            self.assertFalse(trainer.failed)

    def test_failure_poison_and_optimizer_reset_detection(self):
        trainer, _ = self.make_trainer()
        trainer.teacher_features = lambda ids: (_ for _ in ()).throw(RuntimeError("teacher failure"))
        with self.assertRaisesRegex(RuntimeError, "teacher failure"):
            trainer.train_event(1, [response(t) for t in range(9, 17)])
        self.assertTrue(trainer.failed)
        with self.assertRaisesRegex(ValueError, "Prior failure"):
            trainer.train_event(1, [response(t) for t in range(9, 17)])
        trainer, _ = self.make_trainer()
        trainer.train_event(1, [response(t) for t in range(9, 17)])
        trainer.optimizer.state.clear()
        with self.assertRaisesRegex(ValueError, "reset or changed"):
            trainer.bind_checkpoint(1, UPDATED)

    def test_all_author_adaptive_head_variants_rejected(self):
        for name in ("use_prob_head", "use_thresh_head_subsequent", "use_thresh_head_two_model"):
            trainer, _ = self.make_trainer()
            setattr(trainer.draft, name, True)
            with self.subTest(head=name), self.assertRaisesRegex(ValueError, "head became active"):
                trainer.train_event(1, [response(t) for t in range(9, 17)])

    def test_short_two_token_zero_exposure_is_recorded_not_skipped(self):
        import torch
        draft, embed, head, teacher, _ = self.fixture()
        ids = torch.tensor([[1, 2, 3, 4] + [19] * 9])
        hidden, final = teacher(ids)
        mask = torch.zeros_like(ids, dtype=torch.float32)
        mask[:, 2:4] = 1
        found = False
        for seed in range(10):
            generator = torch.Generator().manual_seed(seed)
            loss, meta = training.author_response_loss(draft, embed, head, ids, hidden, final, mask,
                                                        mask_token_id=19, generator=generator)
            if meta["zero_supervision"]:
                self.assertEqual(meta["eligible_positions"], 0)
                self.assertEqual(float(loss.detach()), 0.0)
                loss.backward()
                self.assertTrue(all(p.grad is not None and torch.count_nonzero(p.grad) == 0 for p in draft.parameters()))
                found = True
                break
        self.assertTrue(found, "Fixture must exercise terminal-only sampled anchor")


if __name__ == "__main__":
    unittest.main()
