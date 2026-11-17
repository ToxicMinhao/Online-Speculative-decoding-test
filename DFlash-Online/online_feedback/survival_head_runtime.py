"""Opt-in auxiliary head on the native single-forward K+ training path.

The plain extension object keeps its head outside the model's registered
modules, native optimizer, recovery payload and serving export. The campaign
must save its explicit state separately from the native backbone state.
"""

from copy import deepcopy

import torch

from specforge.training.controller import TrainerCore

from .optimizer import DirectBF16AdamW
from .survival_head import (
    ARMS, ScalarLinearHead, bce_terms, proposal_targets, response_bce,
    survival_head_contract,
)


class SurvivalHeadExtension:
    """Pure chunk losses plus an independent head/optimizer state owner."""

    def __init__(self, model, *, arm, learning_rate, weight_decay, max_grad_norm):
        if arm not in {"LOCAL", "SURVIVAL"}:
            raise ValueError("Active head requires LOCAL or SURVIVAL")
        if (model.objective != "osd_first_error" or model.loss_dtype != torch.bfloat16
                or getattr(model, "first_error_redistribution", None) is None):
            raise ValueError("Survival head requires installed native BF16 K+")
        if any(getattr(model, name, None) is not None for name in (
                "prefix_guard", "role_loss_extension", "point_loss_extension",
                "adversarial_block", "survival_head_extension")):
            raise ValueError("Survival head cannot combine objective extensions")
        parameter = next(model.draft_model.parameters())
        if parameter.dtype != torch.bfloat16:
            raise ValueError("Survival head requires a BF16 draft")
        self.arm = arm
        self.contract = survival_head_contract(arm)
        self.head = ScalarLinearHead(model.draft_model.config.hidden_size, device=parameter.device)
        self.optimizer = DirectBF16AdamW(
            self.head, learning_rate=learning_rate, weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
        )
        self.steps = 0

    def chunk_terms(self, hidden, proposals, labels, weights, eos_token_ids=()):
        targets, valid = proposal_targets(
            proposals, labels, weights, arm=self.arm, eos_token_ids=eos_token_ids,
        )
        return bce_terms(self.head(hidden), targets, valid)

    def augment_response(self, native_loss, terms):
        auxiliary = response_bce(terms)
        total = native_loss + auxiliary
        numerator, count = (value.detach() for value in terms)
        one = native_loss.detach().new_ones(())
        sums = {
            "head_bce_numerator": numerator,
            "head_valid_proposals": count,
            "head_response_bce_sum": auxiliary.detach(),
            "head_response_count": one,
            "head_native_kplus_loss_sum": native_loss.detach(),
        }
        ratios = {
            "head_bce_pooled": (numerator, count),
            "head_bce_response_mean": (auxiliary.detach(), one),
            "head_total_objective": (total.detach(), one),
        }
        return total, sums, ratios

    def step(self):
        norm = self.optimizer.step()
        self.steps += 1
        return norm

    def state_dict(self):
        return {
            "contract": deepcopy(self.contract),
            "head": self.head.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "steps": self.steps,
        }

    def load_state_dict(self, state):
        if state["contract"] != self.contract:
            raise ValueError("Saved survival-head contract differs")
        steps = state["steps"]
        if type(steps) is not int or steps < 0:
            raise ValueError("Saved head step count must be a nonnegative integer")
        self.optimizer.load_state_dict(state["optimizer"])
        self.head.load_state_dict(state["head"], strict=True)
        self.steps = steps


class SurvivalHeadTrainerCore(TrainerCore):
    """Keep the native core math and add one independently clipped head step."""

    def __init__(self, native_core, extension):
        super().__init__(native_core.strategy, native_core.backend, accumulation_steps=2)
        self._micro = native_core._micro
        self.extension = extension

    def train_step(self, batch, ctx=None):
        # response_mean makes native /2 backward the complete normalization
        # for both parameter sets. Native boundary scaling is exactly one;
        # only backbone parameters are owned by the native backend.
        result = super().train_step(batch, ctx)
        if result.optimizer_stepped:
            head_norm = self.extension.step()
            result._metric_values["head_grad_norm"] = head_norm
            result._metric_values["head_optimizer_steps"] = head_norm.new_ones(())
        return result


def install_survival_head_core(native_core, model, *, arm, learning_rate=None,
                               weight_decay=None, max_grad_norm=None):
    """Install after native restoration; disabled KPLUS returns it untouched."""
    if arm not in ARMS:
        raise ValueError(f"Unknown survival-head arm: {arm}")
    if arm == "KPLUS":
        return native_core
    if native_core.accumulation_steps != 2 or native_core.accumulation_remainder:
        raise ValueError("Install head at a complete two-response boundary")
    if getattr(native_core.strategy, "reduction", None) != "response_mean":
        raise ValueError("Survival head requires native response_mean reduction")
    if native_core._ratio_totals or native_core._sum_totals:
        raise ValueError("Native core has uncommitted metric accumulation")
    native_optimizer = native_core.backend.optimizer
    if not isinstance(native_optimizer, DirectBF16AdamW):
        raise ValueError("Survival head requires the native DirectBF16AdamW optimizer")
    parallel = getattr(native_core.backend, "parallel_config", None)
    if getattr(parallel, "world_size", 1) != 1:
        raise ValueError("Survival head pilot requires one training process")
    group = native_optimizer.optimizer.param_groups[0]
    expected = (group["lr"], group["weight_decay"], native_optimizer.max_grad_norm)
    requested = (learning_rate, weight_decay, max_grad_norm)
    if any(actual is not None and actual != native for actual, native in zip(requested, expected)):
        raise ValueError("Head LR, decay and clipping must match the native optimizer")
    extension = SurvivalHeadExtension(
        model, arm=arm, learning_rate=expected[0], weight_decay=expected[1],
        max_grad_norm=expected[2],
    )
    # A plain object is intentional: nn.Module registration would contaminate
    # backbone state/export and could bring the head into native clipping.
    model.survival_head_extension = extension
    return SurvivalHeadTrainerCore(native_core, extension)
