"""Detached reference-prefix selection and differentiable ranking pressure.

These helpers change only the auxiliary objective. The author sampler, attention,
position weights and forward-KL denominator remain owned by the existing model.
"""
from contextlib import contextmanager

import torch
import torch.nn.functional as F


RANKING_OBJECTIVES = {"osd_uniform", "osd_prefix", "osd_reference_margin"}
OSD_OBJECTIVES = {"osd", *RANKING_OBJECTIVES}


@contextmanager
def fixed_anchors(model, anchors, keep):
    """Run the author's complete blocks without drawing anchors a second time."""
    sampler = model._sample_anchor_positions
    model._sample_anchor_positions = lambda *args, **kwargs: (anchors, keep)
    try:
        yield
    finally:
        model._sample_anchor_positions = sampler


@torch.no_grad()
def matching_prefix_mask(proposals, labels, weights, eos_token_ids=()):
    """Protect consecutive matches after the seed, including a valid matched EOS.

    A mismatch, missing supervision or sequence boundary ends the prefix. Later
    matches cannot reopen it. The known seed never receives auxiliary loss.
    """
    terminal = torch.zeros_like(labels, dtype=torch.bool)
    for token_id in eos_token_ids:
        terminal |= labels == token_id
    after_terminal = terminal.long().cumsum(-1) - terminal.long() > 0
    valid_matches = (proposals == labels) & (weights > 0) & ~after_terminal
    prefix = torch.zeros_like(valid_matches)
    prefix[..., 1:] = valid_matches[..., 1:].long().cumprod(-1).bool()
    return prefix


@torch.no_grad()
def allocation_coefficients(prefix, weights, objective):
    """Return coefficients and one detached rho for the whole response presentation."""
    weight_sum = weights.sum()
    protected_weight = (weights * prefix).sum()
    rho = torch.where(
        weight_sum > 0,
        protected_weight / weight_sum.clamp_min(torch.finfo(weights.dtype).tiny),
        weight_sum.new_zeros(()),
    )
    if objective in {"osd_prefix", "osd_reference_margin"}:
        coefficients = prefix.to(weights.dtype)
    elif objective == "osd_uniform":
        coefficients = torch.ones_like(weights) * rho
    else:
        raise ValueError(
            "Ranking allocation must be osd_prefix, osd_uniform or osd_reference_margin"
        )
    return coefficients, rho


def rival_softplus(logits, labels):
    """Use the exact strongest non-label token; both gathered logits retain gradients.

    Argmax uses the projected logits' native precision and first-index tie rule.
    FP32 subtraction happens only after gathering, so no full FP32 vocabulary
    tensor or detached-logit gradient path is needed.
    """
    with torch.no_grad():
        competitors = logits.detach().clone()
        competitors.scatter_(-1, labels.unsqueeze(-1), -torch.inf)
        rivals = competitors.argmax(-1)
    desired = logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1).float()
    competing = logits.gather(-1, rivals.unsqueeze(-1)).squeeze(-1).float()
    return F.softplus(competing - desired)


def reference_margin_penalty(current_softplus, reference_softplus):
    """Penalize each position only when its margin worsens from the reference.

    Inputs are the FP32 outputs of separate ``rival_softplus`` calls, so each
    model uses its own strongest non-label rival. ReLU precedes any weighting
    or reduction and has zero derivative at equality. The reference is fixed.
    """
    return torch.relu(current_softplus - reference_softplus.detach())
