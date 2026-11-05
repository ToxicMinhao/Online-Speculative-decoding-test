"""CPU readout integration on real tiny DFlash blocks and native K+ loss."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from transformers import Qwen3Config
from specforge.algorithms.common.dflash_family_model import OnlineDFlashModel
from specforge.modeling.draft.dflash import DFlashDraftModel

from experiments.block_depth.readout_training import (
    FrozenThreeDraft, TrainingReadout, ReadoutTrainer, SCHEMA,
    require_sidecar_optimizer, check_frozen_versions,
)
from experiments.ok_continuation.resume import differences
from online_feedback.kplus import install_kplus
from online_feedback.objective import TargetObjectiveModel
from online_feedback.optimizer import DirectBF16AdamW
from online_feedback.rng import capture_rng
from online_feedback.training import SpecForgeEventTrainer


def tiny_backbone():
    config = Qwen3Config(hidden_size=16, intermediate_size=32, num_hidden_layers=5,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8, vocab_size=29,
        max_position_embeddings=128, rms_norm_eps=1e-5, tie_word_embeddings=False)
    config.layer_types = ["full_attention"] * 5
    config.block_size = 16
    config.num_target_layers = 35
    config.dflash_config = {"mask_token_id": 0}
    config._attn_implementation = "eager"
    return DFlashDraftModel(config).to(dtype=torch.bfloat16)


def tiny_objective():
    draft = FrozenThreeDraft(tiny_backbone())
    head = torch.nn.Linear(16, 29, bias=False).to(dtype=torch.bfloat16).requires_grad_(False)
    embedding = torch.nn.Embedding(29, 16).to(dtype=torch.bfloat16).requires_grad_(False)
    author = OnlineDFlashModel(draft_model=draft, target_lm_head=head,
        target_embed_tokens=embedding, mask_token_id=0, block_size=16,
        attention_backend="eager", num_anchors=4, loss_decay_gamma=7.,
        objective_chunk_blocks=2, teacher_metrics=False)
    model = TargetObjectiveModel(author, "osd_first_error", loss_precision="bfloat16", eos_token_ids=(28,))
    install_kplus(model)
    model.begin_event()
    generator = torch.Generator().manual_seed(59)
    batch = dict(input_ids=torch.randint(1, 28, (1, 36), generator=generator),
        hidden_states=torch.randn(1, 36, 80, generator=generator, dtype=torch.bfloat16),
        target_last_hidden_states=torch.randn(1, 36, 16, generator=generator, dtype=torch.bfloat16),
        loss_mask=torch.tensor([[0] * 4 + [1] * 32]))
    return model, batch


class ReadoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def test_sidecar_initialization_preserves_rng_and_original_norm(self):
        backbone = tiny_backbone()
        state = capture_rng()
        original = backbone.norm.weight.detach().clone()
        one = TrainingReadout(backbone.norm, 16)
        two = TrainingReadout(backbone.norm, 16)
        self.assertFalse(differences(capture_rng(), state))
        self.assertTrue(torch.equal(one.A, two.A))
        self.assertEqual(tuple(one.A.shape), (32, 16))
        self.assertEqual(tuple(one.B.shape), (16, 32))
        self.assertEqual(int(torch.count_nonzero(one.B)), 0)
        with torch.no_grad():
            one.norm.weight.add_(1)
        self.assertTrue(torch.equal(backbone.norm.weight, original))

    def test_prefix_view_leaves_original_five_layers_and_norm_intact(self):
        backbone = tiny_backbone()
        norm = backbone.norm
        shallow = FrozenThreeDraft(backbone)
        self.assertEqual(len(backbone.layers), 5)
        self.assertEqual(len(shallow.prefix.layers), 3)
        self.assertIs(backbone.norm, norm)
        self.assertIsInstance(shallow.prefix.norm, torch.nn.Identity)
        self.assertTrue(all(shallow.prefix.layers[i] is backbone.layers[i] for i in range(3)))
        self.assertTrue(all(not p.requires_grad for p in backbone.parameters()))
        self.assertEqual({name for name, p in shallow.named_parameters() if p.requires_grad},
                         {"readout.A", "readout.B", "readout.norm.weight"})

    def test_zero_readout_equals_norm_of_residual_complete_layer_three(self):
        model, batch = tiny_objective()
        shallow = model.draft_model
        seen = []
        handle = shallow.backbone.layers[2].register_forward_hook(lambda module, args, result: seen.append(result.detach()))
        with mock.patch.object(model, "_sample_anchor_positions", return_value=(
                torch.tensor([[4, 10, 20, 33]]), torch.ones(1, 4, dtype=torch.bool))):
            _, _, output = model._forward_draft_blocks(batch["input_ids"], batch["hidden_states"], batch["loss_mask"])
        handle.remove()
        expected = shallow.backbone.norm(seen[0])
        self.assertTrue(torch.equal(output, expected))

    def test_actual_kplus_loss_updates_only_sidecar_and_never_executes_layers_four_five(self):
        model, batch = tiny_objective()
        shallow = model.draft_model
        original = {name: value.detach().clone() for name, value in shallow.backbone.state_dict().items()}
        optimizer = DirectBF16AdamW(shallow)
        require_sidecar_optimizer(shallow, optimizer)
        calls = [0] * 5
        handles = []
        for index, layer in enumerate(shallow.backbone.layers):
            def count(module, args, index=index):
                calls[index] += 1
            handles.append(layer.register_forward_pre_hook(count))
        losses = []
        for _ in range(2):
            loss, _, metrics = model(**batch)
            losses.append(loss.detach())
            self.assertEqual(float(metrics["sum_metrics"]["kplus_response_count"]), 1.)
            (loss / 2).backward()
        before = shallow.readout.B.detach().clone()
        norm = optimizer.step()
        require_sidecar_optimizer(shallow, optimizer)
        self.assertTrue(torch.isfinite(torch.stack(losses)).all())
        self.assertGreater(float(norm), 0)
        self.assertFalse(torch.equal(before, shallow.readout.B))
        self.assertEqual(calls, [2, 2, 2, 0, 0])
        self.assertTrue(all(torch.equal(value, original[name]) for name, value in shallow.backbone.state_dict().items()))
        self.assertTrue(all(parameter.grad is None for parameter in model.lm_head.parameters()))
        for handle in handles:
            handle.remove()

    def test_optimizer_membership_and_frozen_mutation_checks_fail_explicitly(self):
        shallow = FrozenThreeDraft(tiny_backbone())
        optimizer = DirectBF16AdamW(shallow)
        optimizer.model_params.append(next(shallow.backbone.parameters()))
        with self.assertRaises(RuntimeError):
            require_sidecar_optimizer(shallow, optimizer)
        params = dict(shallow.backbone.named_parameters())
        versions = {name: p._version for name, p in params.items()}
        with torch.no_grad():
            next(shallow.backbone.parameters()).add_(1)
        with self.assertRaises(RuntimeError):
            check_frozen_versions(params, versions)

    def test_compact_recovery_restores_own_sidecar_adam_rng_and_counters(self):
        model, batch = tiny_objective()
        optimizer = DirectBF16AdamW(model.draft_model)
        loss, _, _ = model(**batch)
        loss.backward()
        optimizer.step()
        with tempfile.TemporaryDirectory() as root:
            trainer = object.__new__(ReadoutTrainer)
            trainer.bundle = SimpleNamespace(draft_model=model.draft_model)
            trainer.backend = SimpleNamespace(optimizer=optimizer)
            trainer.config_text = "same-config"
            trainer.contract = {"schema": SCHEMA}
            trainer.output_root = Path(root)
            trainer.core = SimpleNamespace(_micro=0)
            state = dict(schema=SCHEMA, readout=copy.deepcopy(model.draft_model.readout.state_dict()),
                optimizer=copy.deepcopy(optimizer.state_dict()), rng=capture_rng(), global_step=128,
                presentations=256, events=[{}], config_text=trainer.config_text,
                contract=trainer.contract, accumulation_remainder=0)
            for moments in state["optimizer"]["optimizer_state_dict"]["state"].values():
                moments["step"].fill_(128)
            path = Path(root) / "state.pt"
            torch.save(state, path)
            with torch.no_grad():
                model.draft_model.readout.B.add_(1)
            torch.rand(7)
            trainer._restore(path)
            self.assertFalse(differences(model.draft_model.readout.state_dict(), state["readout"]))
            self.assertFalse(differences(optimizer.state_dict(), state["optimizer"]))
            self.assertFalse(differences(capture_rng(), state["rng"]))
            self.assertEqual(trainer.core._micro, 256)
            self.assertTrue(trainer.resume_receipt["optimizer_exact"])
            self.assertNotIn("backbone", state)
            bad = copy.deepcopy(state)
            next(iter(bad["optimizer"]["optimizer_state_dict"]["state"].values()))["step"].fill_(127)
            torch.save(bad, path)
            with self.assertRaisesRegex(ValueError, "Adam counter"):
                trainer._restore(path)

    def test_group_boundaries_reject_overlap_and_extra_work(self):
        trainer = object.__new__(ReadoutTrainer)
        trainer.events = [{"features": [{"record_id": "already_used"}]}]
        with self.assertRaises(ValueError):
            trainer.train_group("bad", ["feature"] * 127)
        with mock.patch.object(SpecForgeEventTrainer, "_prepare_event", return_value=[{"record_id": "already_used"}]):
            with self.assertRaisesRegex(ValueError, "disjoint"):
                trainer._prepare_event("group2", ["path"], 2)
        with mock.patch.object(SpecForgeEventTrainer, "_prepare_event", return_value=[{"record_id": "new"}]):
            self.assertEqual(trainer._prepare_event("group2", ["path"], 2), [{"record_id": "new"}])

    def test_decoded_boundary_binds_export_recovery_optimizer_and_rng(self):
        from safetensors.torch import save_file
        with tempfile.TemporaryDirectory() as root:
            trainer = object.__new__(ReadoutTrainer)
            shallow = FrozenThreeDraft(tiny_backbone())
            trainer.bundle = SimpleNamespace(draft_model=shallow)
            trainer.backend = SimpleNamespace(optimizer=DirectBF16AdamW(shallow))
            trainer.output_root = Path(root)
            trainer.latest = Path(root) / "training_state.pt"
            trainer.global_step = trainer.presentations = 0
            trainer.core = SimpleNamespace(accumulation_remainder=0)
            trainer.frozen_parameters = dict(shallow.backbone.named_parameters())
            trainer.frozen_versions = {name: p._version for name, p in trainer.frozen_parameters.items()}
            readout = {name: value.detach().clone() for name, value in shallow.readout.state_dict().items()}
            state = dict(readout=readout, optimizer=trainer.backend.optimizer.state_dict(), rng=capture_rng(),
                         global_step=0, presentations=0, accumulation_remainder=0)
            torch.save(state, trainer.latest)
            export = Path(root) / "export"
            export.mkdir()
            save_file(readout, str(export / "readout.safetensors"))
            self.assertEqual(trainer.verify_boundary(export)["status"], "passed")
            readout["B"].add_(1)
            save_file(readout, str(export / "readout.safetensors"))
            with self.assertRaisesRegex(RuntimeError, "decoded boundary"):
                trainer.verify_boundary(export)

    def test_checkpoint_rejects_incorrect_or_missing_work_before_publication(self):
        trainer = object.__new__(ReadoutTrainer)
        trainer.step_observations = [{"changed_coordinates": {"B": 1}}]
        trainer.work = {"frozen_prefix_forwards": 256, "sidecar_forwards": 256,
                        "backbone_layer_1_forwards": 256, "backbone_layer_2_forwards": 256,
                        "backbone_layer_3_forwards": 256, "backbone_layer_4_forwards": 1}
        with self.assertRaisesRegex(RuntimeError, "different number"):
            trainer._checkpoint_state({"optimizer_steps": 128, "response_presentations": 256})
        with self.assertRaisesRegex(RuntimeError, "fixed budget"):
            trainer._checkpoint_state({"optimizer_steps": 127, "response_presentations": 256})
        trainer.step_observations = [{"changed_coordinates": {"B": 0}}]
        with self.assertRaisesRegex(RuntimeError, "no BF16"):
            trainer._checkpoint_state({"optimizer_steps": 128, "response_presentations": 256})

    def test_freeze_requires_complete_fit_and_stops_all_sidecar_gradients(self):
        with tempfile.TemporaryDirectory() as root:
            trainer = object.__new__(ReadoutTrainer)
            trainer.bundle = SimpleNamespace(draft_model=FrozenThreeDraft(tiny_backbone()))
            trainer.global_step, trainer.presentations, trainer.events = 128, 256, [{}]
            trainer.core = SimpleNamespace(accumulation_remainder=0)
            trainer.frozen_parameters = dict(trainer.bundle.draft_model.backbone.named_parameters())
            trainer.frozen_versions = {name: p._version for name, p in trainer.frozen_parameters.items()}
            trainer.output_root, trainer.contract = Path(root), {"schema": SCHEMA}
            with self.assertRaises(ValueError):
                trainer.freeze_readout()
            trainer.global_step, trainer.presentations, trainer.events = 384, 768, [{}, {}, {}]
            result = trainer.freeze_readout()
            self.assertTrue(result["readout_frozen_before_gate"])
            self.assertTrue(all(not p.requires_grad for p in trainer.bundle.draft_model.readout.parameters()))


if __name__ == "__main__":
    unittest.main()
