"""CPU exposure/sampler checks; these do not qualify native GPU execution."""

from pathlib import Path
import tempfile
import unittest

import torch

from experiments.lost_repair.training_probe import (
    frozen_training_state, observe_logits, observe_response, run_readonly_probe, sample_schedule,
    validate_recovery,
)
from online_feedback.rng import capture_rng


def packet(record_id="a", shift=0):
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9]]) + shift
    prompt = 2
    mask = torch.ones_like(ids)
    mask[:, :prompt] = 0
    return dict(
        record_id=record_id, input_ids=ids, loss_mask=mask, prompt_length=prompt,
        hidden_states=torch.zeros((1, ids.shape[1], 2), dtype=torch.bfloat16),
        label_binding=dict(
            capture_backend="same_frozen_SGLang_native_capture", native_capture_validated=True,
            label_binding_scope="native_prefill_and_committed_verification",
            final_feature_logits_used_for_labels=False, input_ids=ids[0].tolist(),
            target_argmax_ids=ids[0, prompt:].tolist(), prompt_length=prompt,
            record_id=record_id, label_mismatches=[],
        ),
    )


class TinyModel(torch.nn.Module):
    """Deterministic draft with the native sampler's random draw/shape/order."""

    block_size = 4
    objective_chunk_blocks = 2
    num_anchors = 5
    eos_token_ids = (12,)
    loss_dtype = torch.bfloat16

    def __init__(self):
        super().__init__()
        self.parameter = torch.nn.Parameter(torch.zeros((), dtype=torch.bfloat16))
        self.lm_head = torch.nn.Identity()
        self._cached_seq_len = None
        self.forward_calls = 0

    def _sample_anchor_positions(self, length, loss_mask, device, max_valid_anchors=None):
        valid = (loss_mask[:, :-1] > .5) & (loss_mask[:, 1:] > .5)
        random_values = torch.rand(valid.shape, device=device)
        random_values.masked_fill_(~valid, 2.)
        width = min(self.num_anchors, max_valid_anchors)
        anchors = random_values.argsort(1)[:, :width].sort(1).values
        return anchors, torch.ones_like(anchors, dtype=torch.bool)

    def _block_labels_and_weights(self, ids, mask, anchors, keep):
        offsets = torch.arange(self.block_size).reshape(1, 1, -1)
        indices = anchors.unsqueeze(-1) + offsets
        safe = indices.clamp(max=ids.shape[1] - 1)
        labels = ids.unsqueeze(1).expand(-1, anchors.shape[1], -1).gather(2, safe)
        weights = keep.unsqueeze(-1).float() * (indices < ids.shape[1]).float()
        weights *= (offsets > 0).float()
        weights *= mask.unsqueeze(1).expand(-1, anchors.shape[1], -1).gather(2, safe)
        weights *= torch.exp(-(offsets - 1).clamp(min=0).float() / 7)
        return labels, weights, safe

    def _forward_draft_blocks(self, ids, hidden, mask, max_valid_anchors=None):
        self.forward_calls += 1
        self._cached_seq_len = ids.shape[1]
        anchors, keep = self._sample_anchor_positions(ids.shape[1], mask, ids.device, max_valid_anchors)
        labels, _, _ = self._block_labels_and_weights(ids, mask, anchors, keep)
        logits = torch.zeros((*labels.shape, 32), dtype=torch.bfloat16)
        logits.scatter_(-1, labels.unsqueeze(-1), 2.)
        for b in range(labels.shape[1]):
            logits[0, b, 1, 31] = 3.
            if int(anchors[0, b]) % 2:
                logits[0, b, 2, 31] = 3.
        return anchors, keep, logits.reshape(1, -1, 32)


class TrainingProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_exact_k6_recovery_configuration_and_complete_window(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text("retained: config\n")
            state = dict(config_text=path.read_text(), global_step=768, presentations=1536,
                         horizon=7424, accumulation_steps=2, reduction="response_mean")
            receipt = validate_recovery(path, state)
            self.assertTrue(receipt["config_matches_recovery"])
            self.assertEqual(receipt["accumulation_remainder"], 0)
            self.assertEqual(receipt["global_step"], 768)
            for key, value in (('global_step', 767), ('presentations', 1535),
                               ('horizon', 7425), ('accumulation_steps', 1),
                               ('reduction', 'token_mean')):
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                    validate_recovery(path, dict(state, **{key: value}))
            # Even a whitespace-only difference is not the copied source config.
            path.write_text(state["config_text"] + "\n")
            with self.assertRaisesRegex(ValueError, "configuration differs"):
                validate_recovery(path, state)

    def test_first_error_selection_eos_gap_tie_and_ineligible(self):
        labels = torch.tensor([[[0, 2, 3, 4], [0, 2, 3, 4], [0, 2, 12, 4],
                                [0, 12, 3, 4], [12, 2, 3, 4], [0, 2, 3, 4]]])
        logits = torch.zeros((*labels.shape, 16), dtype=torch.bfloat16)
        logits.scatter_(-1, labels.unsqueeze(-1), 2.)
        logits[0, 0, 1, 15] = 3.  # high: next proposal correct
        logits[0, 1, 1:3, 15] = 3.  # low: next proposal wrong
        logits[0, 2, 2, 15] = 3.  # EOS mismatch: no eligible successor
        logits[0, 3, 2, 15] = 3.  # after matched EOS ignored
        logits[0, 5, 1, 1] = 2.  # wrong tie wins lower token index
        weights = torch.ones_like(labels, dtype=torch.float32)
        weights[..., 0] = 0
        rows, terms = observe_logits(logits, labels, weights, eos_token_ids=(12,))
        self.assertEqual([r["j"] for r in rows], [1, 1, 2, None, None, 1])
        self.assertEqual([r["priority"] for r in rows],
                         ["high", "low", "ineligible", "ineligible", "ineligible", "high"])
        self.assertEqual(rows[2]["native_priority"], "low")
        self.assertEqual(rows[5]["wrong_margin"], 0.)
        self.assertTrue(rows[5]["wrong_at_tie"])
        self.assertEqual(float(terms[0]), 6.)
        self.assertEqual(float(terms[1]), 2.)
        self.assertEqual(float(terms[2]), 2.)

    def test_validity_gap_cannot_reopen_prefix(self):
        labels = torch.tensor([[[0, 2, 3, 4]]])
        logits = torch.zeros((1, 1, 4, 6), dtype=torch.bfloat16)
        logits.scatter_(-1, labels.unsqueeze(-1), 2.)
        logits[0, 0, 3, 5] = 3.
        weights = torch.tensor([[[0., 1., 0., 1.]]])
        rows, terms = observe_logits(logits, labels, weights)
        self.assertIsNone(rows[0]["j"])
        self.assertEqual(rows[0]["valid_proposals"], 1)
        self.assertEqual(float(terms[0]), 1.)

    def test_response_coefficients_are_global_across_chunks(self):
        model = TinyModel()
        rows, summary = observe_response(model, packet(), presentation_id=0, pass_index=0, max_length=32)
        self.assertEqual(summary["projection_chunks"], 3)
        self.assertEqual(summary["expected_anchor_count"], 5)
        self.assertEqual(summary["response_forwards"], 1)
        self.assertEqual(summary["backwards"], 0)
        w1, w2 = summary["ordinary_low_weight"], summary["ordinary_high_weight"]
        self.assertGreater(w1, 0)
        self.assertGreater(w2, 0)
        self.assertAlmostEqual(summary["low_coefficient"], 1 - .5 * w2 / (w1 + 2 * w2))
        self.assertAlmostEqual(summary["high_coefficient"], 1 + .5 * w1 / (w1 + 2 * w2))
        self.assertAlmostEqual(sum(r["first_error_weight"] for r in rows), summary["first_error_weight"], places=5)
        self.assertTrue(all(r["canonical_start"] == r["sequence_start"] - 2 for r in rows))

    def test_schedule_and_two_pass_forward_preserve_rng_mode_and_parameters(self):
        model = TinyModel().eval()
        model.lm_head.train()  # Preserve heterogeneous child flags too.
        model.parameter.grad = torch.ones_like(model.parameter)
        before = model.parameter.detach().clone()
        gradient = model.parameter.grad
        state = capture_rng()
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for record_id in ("a", "b"):
                path = Path(directory) / (record_id + ".pt")
                torch.save(packet(record_id), path)
                paths.append(path)
            schedule = sample_schedule(model, paths, state, max_length=32)
            self.assertEqual(model.forward_calls, 0)
            self.assertTrue(torch.equal(torch.get_rng_state(), state["torch"]))
            emitted = []
            result = run_readonly_probe(model, paths, state, max_length=32, schedule=schedule,
                                        sink=lambda rows, summary: emitted.append(summary))
        self.assertEqual(result["response_forwards"], 4)
        self.assertEqual(result["projection_chunks"], 12)
        self.assertEqual(result["optimizer_steps"], 0)
        self.assertEqual(result["backwards"], 0)
        self.assertEqual(len(emitted), 4)
        self.assertFalse(model.training)
        self.assertTrue(model.lm_head.training)
        self.assertIsNone(model._cached_seq_len)
        self.assertIs(model.parameter.grad, gradient)
        self.assertTrue(torch.equal(model.parameter, before))
        self.assertTrue(torch.equal(torch.get_rng_state(), state["torch"]))

    def test_schedule_mismatch_fails_before_draft_and_restores_state(self):
        model = TinyModel().eval()
        state = capture_rng()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "a.pt"
            torch.save(packet(), path)
            schedule = sample_schedule(model, [path], state, max_length=32)
            schedule["records"][0]["sampled_anchors"][0]["positions"][0][0] = 99
            with self.assertRaisesRegex(ValueError, "Native anchor draw"):
                run_readonly_probe(model, [path], state, max_length=32, schedule=schedule)
        self.assertTrue(torch.equal(torch.get_rng_state(), state["torch"]))
        self.assertFalse(model.training)
        self.assertIsNone(model._cached_seq_len)

    def test_nonfinite_logits_and_stochastic_draft_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            observe_logits(torch.full((1, 1, 4, 6), float("nan"), dtype=torch.bfloat16),
                           torch.zeros((1, 1, 4), dtype=torch.long), torch.ones((1, 1, 4)))
        model = TinyModel()
        model.dropout = torch.nn.Dropout(.1)
        with self.assertRaisesRegex(ValueError, "zero-dropout"):
            with frozen_training_state(model, capture_rng()):
                pass

    def test_unkept_slots_do_not_become_support_anchors(self):
        model = TinyModel()
        ordinary = model._sample_anchor_positions

        def padded(*args, **kwargs):
            anchors, keep = ordinary(*args, **kwargs)
            anchors[0, -1], keep[0, -1] = 0, False
            return anchors, keep

        model._sample_anchor_positions = padded
        rows, summary = observe_response(model, packet(), presentation_id=0, pass_index=0, max_length=32)
        self.assertEqual(len(rows), 4)
        self.assertEqual(summary["expected_anchor_count"], 4)
        self.assertEqual(summary["sampled_anchor_slots"], 5)
        self.assertTrue(all(row["kept"] for row in rows))
        self.assertTrue(all(row["canonical_start"] >= 0 for row in rows))

    def test_budget_records_each_completed_readonly_forward(self):
        class Budget:
            def __init__(self):
                self.attempted, self.completed, self.failed = [], [], []

            def attempt(self, kind, **metadata):
                self.attempted.append((kind, metadata))
                return len(self.attempted)

            def complete(self, ticket, **measurements):
                self.completed.append((ticket, measurements))

            def fail(self, ticket, error):
                self.failed.append((ticket, error))

        model, budget, state = TinyModel(), Budget(), capture_rng()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "a.pt"
            torch.save(packet(), path)
            schedule = sample_schedule(model, [path], state, max_length=32)
            result = run_readonly_probe(model, [path], state, max_length=32,
                                        schedule=schedule, budget=budget)
        self.assertEqual(len(budget.attempted), 2)
        self.assertEqual(len(budget.completed), 2)
        self.assertEqual(budget.failed, [])
        self.assertTrue(all(kind == "training_forward" for kind, _ in budget.attempted))
        self.assertEqual(result["optimizer_steps"], 0)

    def test_parameter_and_gradient_mutation_are_detected(self):
        model = TinyModel()
        with self.assertRaisesRegex(RuntimeError, "modified a parameter"):
            with frozen_training_state(model, capture_rng()):
                model.parameter.add_(1)
        model.parameter.grad = torch.zeros_like(model.parameter)
        with self.assertRaisesRegex(RuntimeError, "modified a parameter"):
            with frozen_training_state(model, capture_rng()):
                model.parameter.grad.add_(1)


if __name__ == "__main__":
    unittest.main()
