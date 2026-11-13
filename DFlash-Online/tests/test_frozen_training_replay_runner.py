"""CPU contracts for frozen endpoint installation, occurrence binding and replay."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from experiments.frozen_training_repair_replay.runner import (
    attach_sealed_rows, file_identity, index_occurrences, score_presentations,
    shared_piece_identities, slot_key, verify_endpoint, verify_inputs,
)
from experiments.recent_repair_retention.reference import slot_measurements
from online_feedback.rng import capture_rng
from test_lost_repair_training_probe import TinyModel, packet
from experiments.lost_repair.training_probe import sample_schedule


def sealed(presentation=0, **kwargs):
    return dict(presentation_id=presentation, pass_index=presentation // 128,
                request_id="a", canonical_start=0, proposal_slot=2, anchor_ordinal=0,
                canonical_token=3, parent_margin=.75, floor=.375, weight=1., **kwargs)


def observed(prefix=2, *, native_correct=True, margin=.5):
    return dict(presentation_id=0, pass_index=0, request_id="a", canonical_start=0,
                proposal_slot=2, anchor_ordinal=0, canonical_token=3, weight=1.,
                valid=True, label_match=True, correct_prefix=prefix,
                native_correct=native_correct, margin=margin)


class FrozenReplayRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_input_binding_covers_every_dependency_and_sealed_occurrence(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = {name: Path(directory) / name for name in ("config", "state", "schedule", "feature", "slots")}
            for path in paths.values():
                path.write_text("retained")
            plan = dict(origin_config=str(paths["config"]), origin_state=str(paths["state"]),
                        schedule=str(paths["schedule"]), feature_paths=[str(paths["feature"])],
                        sealed_occurrences=str(paths["slots"]),
                        sealed_occurrences_identity=file_identity(paths["slots"]),
                        input_identities={str(path): file_identity(path) for path in paths.values()})
            self.assertTrue(verify_inputs(plan)["all_identities_match"])
            missing = dict(plan, input_identities=dict(plan["input_identities"]))
            del missing["input_identities"][str(paths["feature"])]
            with self.assertRaisesRegex(ValueError, "Missing bound"):
                verify_inputs(missing)
            paths["feature"].write_text("changed")
            with self.assertRaisesRegex(ValueError, "identity differs"):
                verify_inputs(plan)

    def test_occurrence_index_preserves_singleton_and_pass_identity(self):
        rows = [sealed(0), sealed(128), dict(sealed(129), canonical_start=1)]
        indexed = index_occurrences(rows, enforce_population=False)
        self.assertNotIn(("a", 1, 2), indexed[0])
        self.assertIn(("a", 1, 2), indexed[129])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            index_occurrences(rows + [rows[0]], enforce_population=False)
        with self.assertRaisesRegex(ValueError, "pass"):
            index_occurrences([dict(rows[0], pass_index=1)], enforce_population=False)
        with self.assertRaisesRegex(ValueError, "population"):
            index_occurrences(rows)

    def test_index_checks_original_floor_and_excludes_seed(self):
        for change in (dict(floor=.1), dict(parent_margin=float("nan")),
                       dict(proposal_slot=0), dict(proposal_slot=16)):
            with self.subTest(change=change), self.assertRaises(ValueError):
                index_occurrences([dict(sealed(), **change)], enforce_population=False)

    def test_all_outcomes_retained_including_direct_correct_after_earlier_error(self):
        row = sealed()
        for prefix, outcome in ((0, "earlier_break"), (1, "first_mismatch_at_j"), (2, "accepts_through")):
            result = attach_sealed_rows([observed(prefix, native_correct=prefix != 1)],
                                        {slot_key(row): row}, "RECENT7")[0]
            self.assertEqual(result["outcome"], outcome)
            self.assertEqual(result["native_correct"], prefix != 1)
            self.assertEqual(result["accepts_through"], prefix >= 2)
        tied = attach_sealed_rows([observed(1, native_correct=False, margin=0.)], {slot_key(row): row}, "NATIVE7")[0]
        self.assertTrue(tied["wrong_at_tie"])
        self.assertTrue(tied["floor_breach"])
        self.assertEqual(tied["parent_margin"], .75)

    def test_missing_extra_duplicate_and_binding_errors_fail(self):
        binding = {slot_key(sealed()): sealed()}
        cases = ([], [observed(), observed()], [dict(observed(), canonical_start=3)],
                 [dict(observed(), canonical_token=4)], [dict(observed(), anchor_ordinal=4)],
                 [dict(observed(), valid=False)], [dict(observed(), weight=.5)],
                 [observed(1, native_correct=True)], [observed(2, native_correct=False)])
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                attach_sealed_rows(case, binding, "NATIVE7")

    def test_native_bf16_ties_eos_and_validity_boundaries(self):
        labels = torch.tensor([[[0, 2, 3, 5], [0, 2, 3, 5], [0, 2, 3, 5]]])
        logits = torch.zeros((1, 3, 4, 8), dtype=torch.bfloat16)
        logits.scatter_(-1, labels.unsqueeze(-1), 2.)
        logits[0, 0, 1, 1] = 2.  # Wrong tie blocks a later direct match.
        logits[0, 1, 2, 1] = 2.  # Wrong tie at the tag.
        weights = torch.ones((1, 3, 4), dtype=torch.float32)
        weights[..., 0] = 0
        rows = slot_measurements(logits, labels, weights,
            anchors=torch.tensor([[0, 1, 2]]), keep=torch.ones((1, 3), dtype=torch.bool),
            prompt_length=0, request_id="a", presentation_id=0, pass_index=0,
            candidates={("a", 0, 2), ("a", 1, 2), ("a", 2, 2)}, eos_token_ids=(3,))
        self.assertEqual([r["correct_prefix"] for r in rows], [0, 1, 2])
        self.assertEqual([r["native_correct"] for r in rows], [True, False, True])
        self.assertEqual(rows[1]["margin"], 0.)
        # EOS itself remains scorable; positions after it cannot qualify.
        self.assertTrue(all(r["valid"] for r in rows))

    def test_export_equality_detects_wrong_tensor_dtype_and_file(self):
        draft = torch.nn.Linear(3, 2, bias=False, dtype=torch.bfloat16)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.safetensors"
            save_file(draft.state_dict(), str(path))
            digest = file_identity(path)["sha256"]
            receipt = verify_endpoint(draft, path, digest)
            self.assertEqual(receipt["coordinates"], 6)
            with torch.no_grad():
                draft.weight[0, 0] += 1
            with self.assertRaisesRegex(ValueError, "tensor differs"):
                verify_endpoint(draft, path, digest)
            with self.assertRaisesRegex(ValueError, "identity"):
                verify_endpoint(draft, path, "0" * 64)
            draft.float()
            with self.assertRaisesRegex(ValueError, "tensor differs"):
                verify_endpoint(draft, path, digest)

    def test_shared_head_embedding_fingerprint_detects_modification(self):
        model = torch.nn.Module()
        model.embed_tokens = torch.nn.Embedding(4, 3, dtype=torch.bfloat16).requires_grad_(False)
        model.lm_head = torch.nn.Linear(3, 4, bias=False, dtype=torch.bfloat16).requires_grad_(False)
        before = shared_piece_identities(model)
        with torch.no_grad():
            model.lm_head.weight[0, 0] += 1
        after = shared_piece_identities(model)
        self.assertEqual(before["embed_tokens"], after["embed_tokens"])
        self.assertNotEqual(before["lm_head"], after["lm_head"])

    def test_presentation_specific_candidates_and_empty_tags_keep_full_geometry(self):
        model = TinyModel().eval()
        origin_rng = capture_rng()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "feature.pt"
            torch.save(packet(), source)
            schedule = sample_schedule(model, [source], origin_rng, max_length=32)
            features = [dict(record_id="a", path=str(source))]
            rows = {0: {("a", 0, 2): sealed()}}  # No eligible occurrence in pass 1.
            calls = []
            def observer(model, raw, binding, **kwargs):
                calls.append(kwargs["candidates"])
                # Exercise normal native sampling even when no tags are emitted.
                model._sample_anchor_positions(raw["input_ids"].shape[1], raw["loss_mask"], "cpu",
                                               max_valid_anchors=6)
                return ([observed()] if kwargs["candidates"] else []), dict(projection_chunks=3)
            status = dict(response_forwards=0, projection_chunks=0, emitted_occurrences=0,
                          synchronized_forward_wall_s=0.)
            with patch("experiments.frozen_training_repair_replay.runner.observe_response", observer):
                score_presentations(model, features, schedule["records"], rows, origin_rng,
                                    max_length=32, arm="NATIVE7", output=directory, status=status)
            self.assertEqual(calls, [{("a", 0, 2)}, set()])
            self.assertEqual(status["response_forwards"], 2)
            self.assertEqual(status["projection_chunks"], 6)
            self.assertEqual(status["emitted_occurrences"], 1)
            self.assertFalse(model.training)
            self.assertTrue(torch.equal(origin_rng["torch"], torch.get_rng_state()))
            self.assertTrue((directory / "presentations/001.json").exists())

    def test_completed_forward_is_counted_before_binding_failure(self):
        model = TinyModel().eval()
        origin_rng = capture_rng()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "feature.pt"
            torch.save(packet(), source)
            schedule = sample_schedule(model, [source], origin_rng, max_length=32)
            status = dict(response_forwards=0, projection_chunks=0, emitted_occurrences=0,
                          synchronized_forward_wall_s=0.)
            with patch("experiments.frozen_training_repair_replay.runner.observe_response",
                       return_value=([observed()], dict(projection_chunks=3))):
                with self.assertRaisesRegex(ValueError, "unsealed"):
                    score_presentations(model, [dict(record_id="a", path=str(source))],
                        schedule["records"], {}, origin_rng, max_length=32, arm="NATIVE7",
                        output=directory, status=status)
            import json
            saved = json.loads((directory / "STATUS.json").read_text())
            self.assertEqual(saved["response_forwards"], 1)
            self.assertEqual(saved["projection_chunks"], 3)
            self.assertTrue((directory / "raw_presentations/000.json").exists())
            self.assertFalse((directory / "presentations/000.json").exists())

    def test_original_projection_geometry_includes_final_partial_chunk_without_tags(self):
        from experiments.recent_repair_retention.reference import observe_response
        from experiments.lost_repair.training_probe import frozen_training_state
        model = TinyModel()
        model.block_size, model.objective_chunk_blocks, model.num_anchors = 16, 8, 10
        model.eos_token_ids = ()
        raw = packet()
        ids = torch.arange(1, 25).reshape(1, -1)
        mask = torch.ones_like(ids)
        mask[:, :2] = 0
        raw.update(input_ids=ids, loss_mask=mask,
                   hidden_states=torch.zeros((1, 24, 2), dtype=torch.bfloat16))
        raw["label_binding"].update(input_ids=ids[0].tolist(), target_argmax_ids=ids[0, 2:].tolist())
        shapes = []
        hook = model.lm_head.register_forward_pre_hook(
            lambda module, args: shapes.append(tuple(args[0].shape)))
        origin_rng = capture_rng()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "feature.pt"
            torch.save(raw, source)
            binding = sample_schedule(model, [source], origin_rng, max_length=32)["records"][0]
            with frozen_training_state(model, origin_rng):
                torch.randperm(1)
                rows, receipt = observe_response(model, raw, binding, max_length=32,
                                                 candidates=set(), expected_tokens={})
        hook.remove()
        self.assertEqual(shapes, [(1, 8, 16, 32), (1, 2, 16, 32)])
        self.assertEqual(receipt["response_forwards"], 1)
        self.assertEqual(receipt["projection_chunks"], 2)
        self.assertEqual(rows, [])

    def test_failed_observation_records_attempt_before_raising_and_restores_state(self):
        model = TinyModel().eval()
        origin_rng = capture_rng()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "feature.pt"
            torch.save(packet(), source)
            schedule = sample_schedule(model, [source], origin_rng, max_length=32)
            status = dict(response_forwards=0, projection_chunks=0, emitted_occurrences=0,
                          synchronized_forward_wall_s=0.)
            with patch("experiments.frozen_training_repair_replay.runner.observe_response", side_effect=ValueError("bad layout")):
                with self.assertRaisesRegex(ValueError, "bad layout"):
                    score_presentations(model, [dict(record_id="a", path=str(source))],
                        schedule["records"], {}, origin_rng, max_length=32, arm="NATIVE7",
                        output=directory, status=status)
            import json
            saved = json.loads((directory / "STATUS.json").read_text())
            self.assertEqual(saved["response_forwards_attempted"], 1)
            self.assertEqual(saved["response_forwards"], 0)
            self.assertFalse(model.training)
            self.assertTrue(torch.equal(origin_rng["torch"], torch.get_rng_state()))


if __name__ == "__main__":
    unittest.main()
