"""Fit a shallow readout with the existing K+ objective and training machinery.

The SpecForge forward remains responsible for conditioning, attention and
sampling. A separate view stops its frozen draft after layer three; the original
five-layer module is preserved. Only the copied norm and rank-32 factors enter
the ordinary BF16 AdamW optimizer. Recovery and exports contain only this
sidecar, never another copy of the frozen backbone or teacher logits.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from online_feedback.kplus import install_kplus, kplus_contract
from online_feedback.recipe import load_event_config
from online_feedback.rng import capture_rng, preserve_rng, restore_rng
from online_feedback.storage import write_json
from online_feedback.training import ReductionAdapter, SpecForgeEventTrainer, ranking_eos_token_ids


SCHEMA = "depth35_readout_v1"


class TrainingReadout(nn.Module):
    """Same readout formula/tensors, using qualified SpecForge training kernels.

    Serving retains SGLang's fused residual-add/RMSNorm. Training receives the
    SpecForge layer's BF16 residual-complete output. This numerical distinction
    is disclosed; bitwise training/native hidden-state equality is not claimed.
    """

    def __init__(self, final_norm, hidden_size, *, seed=17):
        super().__init__()
        self.norm = copy.deepcopy(final_norm).requires_grad_(True)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        initial = torch.empty(32, hidden_size, dtype=torch.float32)
        initial.uniform_(-1 / math.sqrt(hidden_size), 1 / math.sqrt(hidden_size), generator=generator)
        self.A = nn.Parameter(initial.to(final_norm.weight))
        self.B = nn.Parameter(torch.zeros(hidden_size, 32, device=final_norm.weight.device,
                                         dtype=final_norm.weight.dtype))

    def forward(self, hidden):
        normalized = self.norm(hidden)
        return normalized + F.linear(F.linear(normalized, self.A), self.B)


class FrozenThreeDraft(nn.Module):
    """Reuse the author forward over three unchanged residual-complete layers.

    SpecForge layers return the residual-complete tensor. Native SGLang carries
    that residual separately; its readout adds it through fused RMSNorm. These
    represent the same mathematical input, with different intermediate rounding.
    Do not add the residual again to the SpecForge output.
    """

    def __init__(self, backbone):
        super().__init__()
        if len(backbone.layers) != 5:
            raise ValueError("The readout requires the authenticated five-layer K+58 draft")
        self.backbone = backbone.requires_grad_(False)
        # The view owns a module dictionary but shares the frozen parameter
        # tensors. Changing its layer list/norm cannot change original FIVE.
        self.prefix = copy.copy(backbone)
        self.prefix._modules = backbone._modules.copy()
        self.prefix.layers = nn.ModuleList(list(backbone.layers[:3]))
        self.prefix.norm = nn.Identity()
        self.readout = TrainingReadout(backbone.norm, backbone.config.hidden_size)
        self.config = backbone.config
        self.target_layer_ids = backbone.target_layer_ids
        self.sliding_window = backbone.sliding_window
        self.layer_types = backbone.layer_types

    def forward(self, *args, **kwargs):
        with torch.no_grad():
            hidden = self.prefix(*args, **kwargs)
        return self.readout(hidden)


def require_sidecar_optimizer(model, optimizer):
    """Check the optimizer contains all and only the three sidecar tensors."""
    expected = {id(parameter) for parameter in model.readout.parameters()}
    actual = {id(parameter) for parameter in optimizer.model_params}
    trainable = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if expected != actual or expected != trainable or len(expected) != 3:
        raise RuntimeError("Readout optimizer includes frozen weights or omits a sidecar tensor")
    if any(parameter.grad is not None for parameter in model.backbone.parameters()):
        raise RuntimeError("Frozen K+ backbone received gradients")


def check_frozen_versions(parameters, versions):
    """Version counters catch parameter writes without copying the backbone."""
    changed = [name for name, parameter in parameters.items() if parameter._version != versions[name]]
    if changed:
        raise RuntimeError("Frozen parameter changed during readout fitting: " + ", ".join(changed))


class ReadoutTrainer(SpecForgeEventTrainer):
    """Three fixed 128-response groups, one continuous sidecar optimizer/RNG."""

    def __init__(self, config_path, output_root, resume_state=None):
        self.response_observations = []
        self.step_observations = []
        super().__init__(config_path, output_root, resume_state=resume_state,
                         objective="osd_first_error", reduction="response_mean", ranking_lambda=0.)

    def status(self):
        return {**super().status(), "schema_version": SCHEMA, "backbone_optimizer_steps": 0}

    def _validate_config(self):
        super()._validate_config()
        training = self.cfg.training
        expected = (training.seed, training.learning_rate, training.max_grad_norm,
                    self.accumulation, self.horizon, self.max_length)
        if expected != (17, 3e-4, 1.0, 2, 384, 2272):
            raise ValueError("Readout seed/LR/clipping/accumulation/horizon/context changed")
        if training.loss_decay_gamma != 7.0 or training.objective_chunk_blocks != 8:
            raise ValueError("Readout must preserve native decay7 and eight-block projections")
        if (self.recipe.optimizer != "bf16_adamw" or self.recipe.weight_decay != .01
                or not self.recipe.shuffle_responses or self.recipe.kl_precision != "bfloat16"
                or self.recipe.responses_per_event != 128 or self.recipe.passes_per_event != 2
                or self.recipe.max_events != 3):
            raise ValueError("Readout recipe differs from the three fixed native K+ groups")

    def _build_runtime(self):
        """Use the shared author assembly, objective, backend and TrainerCore.

        This small assembly override inserts the readout before optimizer/DDP
        creation; it does not replace the inherited event loop or loss reduction.
        """
        from accelerate.utils import set_seed
        from specforge.application import resolve_run
        from specforge.training.assembly import build_model_bundle
        from specforge.training.backend import FSDPTrainingBackend, ParallelConfig
        from specforge.training.controller import TrainerCore
        from torch._functorch import config as functorch_config
        from online_feedback.objective import TargetObjectiveModel, OSDTrainStrategy
        from online_feedback.optimizer import DirectBF16AdamW

        self.cfg, self.recipe = load_event_config(self.config_path)
        self._validate_config()
        functorch_config.donated_buffer = False
        set_seed(17)
        self._initialize_distributed()
        cfg = self.cfg
        resolved = resolve_run(cfg)
        self.bundle = build_model_bundle(cfg, algorithm=resolved.algorithm)
        shallow = FrozenThreeDraft(self.bundle.draft_model)
        self.bundle.draft_model = shallow
        self.bundle.model.draft_model = shallow
        self.bundle.model.lm_head.requires_grad_(False)
        self.bundle.model.embed_tokens.requires_grad_(False)
        with preserve_rng():
            self.eos_token_ids = ranking_eos_token_ids(cfg.model.target_model_path)
        self.bundle.model = TargetObjectiveModel(self.bundle.model, "osd_first_error",
            loss_precision="bfloat16", eos_token_ids=self.eos_token_ids)
        install_kplus(self.bundle.model)
        if (self.bundle.model.num_anchors != 64 or self.bundle.model.block_size != 16
                or self.bundle.model.attention_backend != "flex_attention"):
            raise ValueError("Readout requires native FlexAttention, block16 and at most64 anchors")

        def optimizer_factory(model):
            return DirectBF16AdamW(model, learning_rate=3e-4, weight_decay=.01, max_grad_norm=1.)

        parallel = ParallelConfig.from_distributed(tp_size=1, sp_ulysses_size=1,
            sp_ring_size=1, sharding_strategy=cfg.training.fsdp_sharding)
        self.backend = FSDPTrainingBackend(parallel, optimizer_factory=optimizer_factory)
        wrapped = self.backend.prepare_model(self.bundle.model, optimizer_target=shallow)
        self.author_strategy = OSDTrainStrategy(wrapped)
        self.strategy = ReductionAdapter(self.author_strategy, "response_mean")
        self.core = TrainerCore(self.strategy, self.backend, accumulation_steps=2)
        self.hidden_width = self.bundle.target_hidden_size * len(shallow.target_layer_ids)
        self.vocab_size = self.bundle.target_vocab_size
        self.contract = {"schema": SCHEMA, "frozen_checkpoint": cfg.model.draft_checkpoint_path,
                         "rank": 32, "depth": 3, "kplus": kplus_contract(),
                         "loss_decay_gamma": self.bundle.model.loss_decay_gamma,
                         "eos_token_ids": list(self.eos_token_ids),
                         "training_readout_norm": "SpecForge_BF16_residual_complete_then_copied_norm",
                         "serving_readout_norm": "SGLang_fused_residual_add_RMSNorm",
                         "training_native_hidden_bitwise_parity_claimed": False,
                         "trainable": ["readout.norm.weight", "readout.A", "readout.B"]}
        self._original_sampler = self.bundle.model._sample_anchor_positions
        self.bundle.model._sample_anchor_positions = self._sample_and_record_anchors
        self.bundle.model.train()
        self.frozen_parameters = {name: parameter for name, parameter in
                                  self.bundle.model.named_parameters() if not parameter.requires_grad}
        self.frozen_versions = {name: parameter._version for name, parameter in self.frozen_parameters.items()}
        require_sidecar_optimizer(shallow, self.backend.optimizer)
        self._install_observations()

    def _install_observations(self):
        original_forward = self.author_strategy.forward_loss
        original_step = self.backend.optimizer.step
        self.work = {}
        self._work_handles = []
        modules = [(self.bundle.draft_model.prefix, "frozen_prefix_forwards"),
                   (self.bundle.draft_model.readout, "sidecar_forwards"),
                   (self.bundle.model.lm_head, "vocabulary_projections")]
        modules.extend((layer, f"backbone_layer_{index + 1}_forwards")
                       for index, layer in enumerate(self.bundle.draft_model.backbone.layers))
        for module, name in modules:
            def count(_module, _inputs, name=name):
                self.work[name] = self.work.get(name, 0) + 1
            self._work_handles.append(module.register_forward_pre_hook(count))

        def forward(batch, ctx=None):
            output = original_forward(batch, ctx)
            row = {"record_id": str(batch.sample_ids[0]), "loss": float(output.loss.detach()),
                   "sum_metrics": {key: float(value.detach()) for key, value in output.sum_metrics.items()},
                   "presentation": self.presentations + 1,
                   "sampled_anchors": list(self._anchor_observations)}
            self.last_response = row
            if not math.isfinite(row["loss"]) or not all(math.isfinite(x) for x in row["sum_metrics"].values()):
                raise RuntimeError("Nonfinite readout response loss or exposure")
            self.response_observations.append(row)
            return output

        def step():
            shallow = self.bundle.draft_model
            before = {name: parameter.detach().clone() for name, parameter in shallow.readout.named_parameters()}
            require_sidecar_optimizer(shallow, self.backend.optimizer)
            norm = original_step()  # Existing clipping rejects nonfinite gradients.
            check_frozen_versions(self.frozen_parameters, self.frozen_versions)
            changed = {name: int((parameter.detach() != before[name]).sum())
                       for name, parameter in shallow.readout.named_parameters()}
            if any(not bool(torch.isfinite(parameter).all()) for parameter in shallow.readout.parameters()):
                raise RuntimeError("Nonfinite fitted readout parameter")
            self.step_observations.append({"global_step": self.global_step + 1,
                                           "gradient_norm": float(norm), "changed_coordinates": changed})
            write_json(self.output_root / "steps" / f"step_{self.global_step + 1:03d}.json",
                       {**self.step_observations[-1], "responses": self.response_observations[-2:]})
            return norm

        self.author_strategy.forward_loss = forward
        self.backend.optimizer.step = step

    def _train_passes(self, features, passes):
        self.response_observations.clear()
        self.step_observations.clear()
        self.work.clear()
        rows = super()._train_passes(features, passes)
        if len(rows) != len(self.response_observations):
            raise RuntimeError("Readout observations double-counted or omitted a response forward")
        for row, observation in zip(rows, self.response_observations):
            if row["record_id"] != observation["record_id"]:
                raise RuntimeError("Readout response observation identity changed")
            row["response_observation"] = observation
        return rows

    def _checkpoint_state(self, event):
        if event["optimizer_steps"] != 128 or event["response_presentations"] != 256:
            raise RuntimeError("Readout group work differs from its fixed budget")
        if not any(sum(row["changed_coordinates"].values()) for row in self.step_observations):
            raise RuntimeError("Readout fitting produced no BF16 parameter update")
        if (self.work.get("frozen_prefix_forwards") != 256 or self.work.get("sidecar_forwards") != 256
                or any(self.work.get(f"backbone_layer_{i}_forwards", 0) != (256 if i <= 3 else 0)
                       for i in range(1, 6))):
            raise RuntimeError("Readout executed a different number of frozen/shallow layers")
        check_frozen_versions(self.frozen_parameters, self.frozen_versions)
        event["readout_steps"] = list(self.step_observations)
        event["actual_module_work"] = dict(self.work)
        event["frozen_parameter_versions_unchanged"] = True
        return {"schema": SCHEMA, "readout": {name: value.detach().cpu() for name, value in
                    self.bundle.draft_model.readout.state_dict().items()},
                "optimizer": self.backend.optimizer.state_dict(), "rng": capture_rng(),
                "global_step": self.global_step, "presentations": self.presentations,
                "events": self.events + [event], "config_text": self.config_text,
                "contract": self.contract, "accumulation_remainder": self.core.accumulation_remainder}

    def _restore(self, path):
        state = torch.load(path, map_location="cpu", weights_only=False)
        if (state["schema"] != SCHEMA or state["config_text"] != self.config_text
                or state["contract"] != self.contract or state["accumulation_remainder"] != 0
                or state["presentations"] != 2 * state["global_step"]
                or state["global_step"] != 128 * len(state["events"])):
            raise ValueError("Readout recovery does not describe the same complete group boundary")
        optimizer_state = state["optimizer"]["optimizer_state_dict"]
        parameter_ids = optimizer_state["param_groups"][0]["params"]
        if set(parameter_ids) != set(optimizer_state["state"]):
            raise ValueError("Readout recovery is missing an Adam state")
        for moments in optimizer_state["state"].values():
            if int(moments["step"].item()) != state["global_step"]:
                raise ValueError("Readout Adam counter differs from the group boundary")
            if any(not bool(torch.isfinite(moments[name]).all()) for name in ("exp_avg", "exp_avg_sq")):
                raise ValueError("Readout recovery contains nonfinite Adam moments")
        if any(not bool(torch.isfinite(value).all()) for value in state["readout"].values()):
            raise ValueError("Readout recovery contains nonfinite weights")
        self.bundle.draft_model.readout.load_state_dict(state["readout"], strict=True)
        self.backend.optimizer.load_state_dict(state["optimizer"])
        self.events, self.global_step, self.presentations = state["events"], state["global_step"], state["presentations"]
        self.core._micro = self.presentations
        restore_rng(state["rng"])
        from experiments.ok_continuation.resume import differences
        optimizer_diff = differences(self.backend.optimizer.state_dict(), state["optimizer"])
        rng_diff = differences(capture_rng(), state["rng"])
        own = {name: bool(torch.equal(value.cpu(), state["readout"][name])) for name, value in
               self.bundle.draft_model.readout.state_dict().items()}
        if not all(own.values()) or optimizer_diff or rng_diff:
            raise RuntimeError("Readout recovery weights/optimizer/RNG differ after restoration")
        self.resume_receipt = {"status": "passed", "state_path": str(path), "readout_exact": own,
                               "global_step": self.global_step, "presentations": self.presentations,
                               "optimizer_exact": True, "rng_exact": True,
                               "optimizer_loaded_without_reset": True, "rng_restored_without_reseed": True}
        write_json(self.output_root / "RESUME_CHECK.json", self.resume_receipt)

    def _export(self, export_path):
        from safetensors.torch import save_file
        export_path = Path(export_path)
        export_path.mkdir()
        save_file({name: tensor.detach().cpu().contiguous() for name, tensor in
                   self.bundle.draft_model.readout.state_dict().items()}, str(export_path / "readout.safetensors"))
        write_json(export_path / "readout_config.json", {**self.contract, "steps": self.global_step,
                   "presentations": self.presentations, "precision": "bfloat16"})

    def train_group(self, group_id, feature_paths):
        if len(feature_paths) != 128 or len(self.events) >= 3:
            raise ValueError("Readout fitting is exactly three groups of 128 responses")
        result = self.train_event(group_id, feature_paths, passes=2)
        result["boundary_check"] = self.verify_boundary(result["export_path"])
        return result

    def verify_boundary(self, export_path):
        """Authenticate decoded recovery and serving sidecar before retirement."""
        from safetensors.torch import load_file
        from experiments.ok_continuation.resume import differences

        with preserve_rng():
            state = torch.load(self.latest, map_location="cpu", weights_only=False)
            live = dict(self.bundle.draft_model.readout.state_dict())
            exported = load_file(str(Path(export_path) / "readout.safetensors"))
            problems = {
                "readout": differences(live, state["readout"]),
                "export": differences(live, exported),
                "optimizer": differences(self.backend.optimizer.state_dict(), state["optimizer"]),
                "rng": differences(capture_rng(), state["rng"]),
                "counters": differences((self.global_step, self.presentations, self.core.accumulation_remainder),
                                        (state["global_step"], state["presentations"], state["accumulation_remainder"])),
            }
            check_frozen_versions(self.frozen_parameters, self.frozen_versions)
        receipt = {"status": "failed" if any(problems.values()) else "passed", "differences": problems,
                   "state_path": str(self.latest), "export_path": str(export_path),
                   "global_step": self.global_step, "presentations": self.presentations,
                   "frozen_parameter_versions_unchanged": True}
        write_json(self.output_root / "BOUNDARY_CHECK.json", receipt)
        if receipt["status"] != "passed":
            raise RuntimeError("Readout successor state/export failed decoded boundary checks")
        return receipt

    def _prepare_event(self, event_id, feature_paths, passes):
        current = super()._prepare_event(event_id, feature_paths, passes)
        prior_ids = {feature["record_id"] for event in self.events for feature in event["features"]}
        if prior_ids.intersection(feature["record_id"] for feature in current):
            raise ValueError("READOUT groups must contain disjoint response identities")
        return current

    def freeze_readout(self):
        if (self.global_step, self.presentations, len(self.events), self.core.accumulation_remainder) != (384, 768, 3, 0):
            raise ValueError("Freeze only after the prescribed complete readout fit")
        check_frozen_versions(self.frozen_parameters, self.frozen_versions)
        self.bundle.draft_model.readout.eval().requires_grad_(False)
        receipt = {"status": "passed", "schema": SCHEMA, "global_step": 384, "presentations": 768,
                   "backbone_optimizer_steps": 0, "frozen_parameters_unchanged": True,
                   "readout_frozen_before_gate": True, "contract": self.contract}
        write_json(self.output_root / "FROZEN_READOUT.json", receipt)
        return receipt

    def close(self):
        for handle in getattr(self, "_work_handles", []):
            handle.remove()
        self.frozen_parameters = {}
        super().close()


def main():
    """One group per training subprocess; restore its compact own state next."""
    import argparse
    import time
    import traceback
    import torch._functorch.config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--group", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--resume-state", type=Path)
    args = parser.parse_args()
    if (args.group > 1) != (args.resume_state is not None):
        raise ValueError("Groups2/3 resume the preceding readout state; group1 starts fresh")
    # Same retained-graph compiler contract as qualified native K+ training.
    torch._functorch.config.donated_buffer = False
    args.output.mkdir(parents=True, exist_ok=True)
    payload = json.loads(args.pool.read_text())
    paths = payload["features"] if isinstance(payload, dict) else payload
    trainer = None
    started = time.perf_counter()
    try:
        trainer = ReadoutTrainer(args.config, args.output, resume_state=args.resume_state)
        initialization_s = time.perf_counter() - started
        if len(trainer.events) != args.group - 1:
            raise ValueError("Readout group number differs from the restored event history")
        result = trainer.train_group(f"group_{args.group:03d}", paths)
        result["initialization_s"] = initialization_s
        result["resume_check"] = getattr(trainer, "resume_receipt", {"status": "fresh_seed17"})
        result["startup_work"] = {"extra_optimizer_steps": 0, "extra_response_forwards": 0,
                                   "compiler_warmup": "first_actual_forward_included_in_training"}
        if args.group == 3:
            result["frozen_readout"] = trainer.freeze_readout()
        if torch.cuda.is_initialized():
            result["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
            result["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved()
        result["worker_wall_s"] = time.perf_counter() - started
        write_json(args.output / "RESULT.json", result)
    except BaseException as error:
        write_json(args.output / "FAILED.json", {"status": "failed", "error": repr(error),
                   "traceback": traceback.format_exc(), "automatic_retry": False,
                   "trainer_status": trainer.status() if trainer is not None else None,
                   "last_response_repr": repr(getattr(trainer, "last_response", None))})
        raise
    finally:
        if trainer is not None:
            trainer.close()


if __name__ == "__main__":
    main()
