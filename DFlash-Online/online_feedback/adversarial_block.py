"""Pure primitives for the bounded two-view K+ block-training pilot.

The native objective supplies clean chunk metadata and its already-reduced
scalars. These helpers do not sample anchors, run a model, or step an optimizer.
In particular, they never replace K+'s aggregate numerator with weighted rows.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math

import torch

from .first_error import first_error_mask, valid_prefix_mask
from .kplus import first_error_groups, redistribute_response
from .rng import capture_rng, restore_rng


ARMS = ("KPLUS", "CLEAN2", "RANDOM", "ADV")
PILOT_RHO = 0.01


def uses_two_views(arm, rho=PILOT_RHO):
    """CLEAN2 always takes two real passes; disabled attacks use native K+."""
    if arm not in ARMS or not math.isfinite(rho) or rho < 0:
        raise ValueError("Expected a pilot arm and finite nonnegative radius")
    return arm == "CLEAN2" or (arm in ("RANDOM", "ADV") and rho > 0)


@dataclass(frozen=True)
class FrozenKPlusChunk:
    """Detached clean decisions, indexed explicitly across checkpoint replays."""

    weights: torch.Tensor
    selection: torch.Tensor
    priority1: torch.Tensor
    priority2: torch.Tensor
    eligible: torch.Tensor

    @classmethod
    @torch.no_grad()
    def capture(cls, proposals, labels, weights, eos_token_ids=(), selection=None):
        if selection is None:
            selection = first_error_mask(proposals, labels, weights, eos_token_ids)
        priority1, priority2 = first_error_groups(
            proposals, labels, weights, selection, eos_token_ids,
        )
        eligible = valid_prefix_mask(labels, weights, eos_token_ids)
        return cls(*(value.detach().clone() for value in
                     (weights, selection, priority1, priority2, eligible)))

    def terms(self, token_loss):
        """Native N_K, selected weight, then W1/W2/N1/N2/count1/count2.

        The caller reduces chunks in the original native order. Both the
        ordinary and priority numerators share the same BF16 token-loss graph.
        """
        selected_weights = self.weights * self.selection
        weights1 = self.weights * self.priority1
        weights2 = self.weights * self.priority2
        return (
            (token_loss * selected_weights).sum(), selected_weights.sum(),
            weights1.sum(), weights2.sum(),
            (token_loss * weights1).sum(), (token_loss * weights2).sum(),
            self.priority1.sum().float(), self.priority2.sum().float(),
        )


@dataclass(frozen=True)
class FrozenKPlusResponse:
    """Keep native pooled scalar values, including the once-added epsilon.

    `native_terms` has TargetObjectiveModel's existing eleven-field K+ layout.
    Capturing these values after its chunk reduction preserves that reduction's
    floating-point order; recomputing W1/W2 from concatenated rows would not.
    """

    chunks: tuple
    selected_weight: torch.Tensor
    denominator: torch.Tensor
    weight1: torch.Tensor
    weight2: torch.Tensor
    count1: torch.Tensor
    count2: torch.Tensor

    @classmethod
    def capture(cls, chunks, native_terms, denominator):
        if len(native_terms) != 11:
            raise ValueError("Expected the native eleven-field K+ reduction")
        values = (native_terms[1], denominator, native_terms[5], native_terms[6],
                  native_terms[9], native_terms[10])
        return cls(tuple(chunks), *(value.detach().clone() for value in values))

    def reduce_response(self, native_terms):
        """Reuse the aggregate formula and clean masses with current KL rows."""
        return redistribute_response(
            native_terms[0], self.selected_weight,
            (self.weight1, self.weight2, native_terms[7], native_terms[8],
             self.count1, self.count2),
        )

    def loss(self, native_terms):
        numerator, metrics = self.reduce_response(native_terms)
        loss = numerator / self.denominator.clamp_min(torch.finfo(torch.float32).tiny)
        return loss, metrics


def apply_mask_delta(embeddings, delta, eligible):
    """Sum in FP32, cast once to the original model-input dtype.

    Shapes are [batch, blocks, slots, hidden] and [batch, blocks, slots].
    Masking is part of the graph, so seed/invalid-slot delta gradients are zero.
    Frozen teacher embeddings cannot acquire gradients through this operation.
    """
    if embeddings.shape != delta.shape or embeddings.shape[:-1] != eligible.shape:
        raise ValueError("Perturbation and eligibility must match block embeddings")
    masked = delta.masked_fill(~eligible.unsqueeze(-1), 0)
    return (embeddings.detach().float() + masked).to(embeddings.dtype)


def clean_delta(embeddings):
    """The only extra differentiable leaf; read its grad after clean backward.

    The intended integration backpropagates 0.5 * L / 2 per response/view.
    That same backward supplies native parameter gradients and this delta grad;
    no extra backward or higher-order graph is used to construct the direction.
    """
    return torch.zeros_like(embeddings, dtype=torch.float32, requires_grad=True)


def _block_generator(device, key, block_index):
    encoded = json.dumps([list(key), block_index], separators=(",", ":"),
                         ensure_ascii=True).encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") % (2**63)
    return torch.Generator(device=device).manual_seed(seed)


@torch.no_grad()
def make_perturbation(embeddings, eligible, gradient, *, arm, key, rho=PILOT_RHO):
    """One detached block-normalized direction and its actual cast diagnostics.

    RANDOM uses the ADV zero-gradient convention and a separate generator for
    each block, keyed by dataset/event/pass/presentation plus flat block index.
    All valid slots participate, including the unselected clean suffix.
    """
    if arm not in ("ADV", "RANDOM") or not math.isfinite(rho) or rho < 0:
        raise ValueError("Expected ADV/RANDOM and finite nonnegative radius")
    if gradient is None:
        raise RuntimeError("Clean backward did not reach the perturbation leaf")
    if embeddings.shape != gradient.shape or embeddings.shape[:-1] != eligible.shape:
        raise ValueError("Gradient and eligibility must match block embeddings")
    support = eligible.unsqueeze(-1)
    base = embeddings.detach().float().masked_fill(~support, 0)
    direction = gradient.detach().float().masked_fill(~support, 0)
    gradient_norm = torch.linalg.vector_norm(direction, dim=(-2, -1))
    embedding_norm = torch.linalg.vector_norm(base, dim=(-2, -1))
    if not bool(torch.isfinite(gradient_norm).all() & torch.isfinite(embedding_norm).all()):
        raise ValueError("Nonfinite eligible embedding or perturbation gradient")
    active = (gradient_norm > 0) & eligible.any(-1)
    if arm == "RANDOM":
        direction = torch.empty_like(base)
        flat = direction.reshape(-1, *direction.shape[-2:])
        for block_index, block in enumerate(flat):
            block.copy_(torch.randn(
                block.shape, generator=_block_generator(block.device, key, block_index),
                device=block.device, dtype=torch.float32,
            ))
        direction.masked_fill_(~support, 0)
    direction_norm = torch.linalg.vector_norm(direction, dim=(-2, -1))
    nominal_radius = rho * embedding_norm
    active &= direction_norm > 0
    safe_norm = torch.where(active, direction_norm, 1)
    delta = direction / safe_norm[..., None, None]
    delta *= torch.where(active, nominal_radius, 0)[..., None, None]
    intended_norm = torch.linalg.vector_norm(delta, dim=(-2, -1))
    if not bool(torch.isfinite(delta).all() & torch.isfinite(intended_norm).all()):
        raise ValueError("Nonfinite proposed perturbation")
    summed = apply_mask_delta(embeddings, delta, eligible)
    realized = summed.float() - embeddings.detach().float()
    changed_slots = (realized != 0).any(-1) & eligible
    metrics = {
        "eligible_slots": eligible.sum(-1),
        "gradient_norm": gradient_norm,
        "nominal_radius": nominal_radius,
        "intended_norm": intended_norm,
        "realized_norm": torch.linalg.vector_norm(realized, dim=(-2, -1)),
        "changed_slots": changed_slots.sum(-1),
        "zero_gradient_blocks": (gradient_norm == 0) & eligible.any(-1),
        "no_eligible_blocks": ~eligible.any(-1),
    }
    return delta.detach(), metrics


@contextmanager
def replay_model_rng(pre_clean_rng):
    """Replay the post-anchor/pre-draft state; restore post-clean RNG afterward.

    Enter only once the clean backward has finished. Checkpoint recomputation
    stays within each view's lifetime. This helper does not alter module modes.
    """
    post_clean_rng = capture_rng()
    restore_rng(pre_clean_rng)
    try:
        yield
    finally:
        restore_rng(post_clean_rng)
