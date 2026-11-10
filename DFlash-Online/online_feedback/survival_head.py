"""The fixed token-correctness and prefix-survival auxiliary objectives.

Both heads use the current clean draft's detached greedy proposals. Only the
target differs; all valid proposals contribute, including the error suffix.
"""

import math

import torch
import torch.nn.functional as F

from .first_error import valid_prefix_mask


ARMS = {"KPLUS", "LOCAL", "SURVIVAL"}
HEAD_SEED = 17


def survival_head_contract(arm):
    if arm not in ARMS:
        raise ValueError(f"Unknown survival-head arm: {arm}")
    return {
        "name": "kplus_prefix_survival_head_v1",
        "arm": arm,
        "coefficient": 1.0,
        "head_seed": HEAD_SEED,
        "head_initialization": "CPU_generator_uniform_+-1/sqrt(hidden_size)_then_BF16",
        "head_bias_initialization": "zero",
        "head_projection_precision": "bfloat16",
        "bce_precision": "float32",
        "target": "current_detached_argmax_correctness" if arm == "LOCAL" else
                  "current_detached_argmax_prefix_correctness_from_slot_one" if arm == "SURVIVAL" else None,
        "mask": "native_valid_prefix_mask_including_canonical_EOS",
        "reduction": "response_BCE_sum_divided_once_by_valid_proposal_count",
        "empty_mask_loss": 0.0,
        "response_accumulation": 2,
        "head_optimizer": "separate_native_direct_BF16_AdamW",
        "head_clipping": "independent_with_native_threshold",
        "additional_draft_forwards": 0,
        "additional_vocabulary_projections": 0,
        "export_head": False,
    }


@torch.no_grad()
def proposal_targets(proposals, labels, weights, *, arm, eos_token_ids=()):
    """Return detached FP32 binary targets and the unchanged validity mask."""
    if arm not in {"LOCAL", "SURVIVAL"}:
        raise ValueError("Auxiliary targets require LOCAL or SURVIVAL")
    if proposals.shape != labels.shape or weights.shape != labels.shape:
        raise ValueError("Proposals, canonical labels and weights must align")
    valid = valid_prefix_mask(labels, weights, eos_token_ids)
    matches = proposals.detach() == labels.detach()
    targets = torch.zeros_like(weights, dtype=torch.float32)
    if arm == "LOCAL":
        targets[..., 1:] = matches[..., 1:].float()
    else:
        # The known seed is deliberately absent from the cumulative product.
        targets[..., 1:] = matches[..., 1:].long().cumprod(-1).float()
    targets.masked_fill_(~valid, 0.0)
    return targets, valid


def bce_terms(logits, targets, valid):
    """Additive FP32 numerator/count; no per-chunk or decay-weight averaging."""
    if logits.shape != targets.shape or valid.shape != targets.shape:
        raise ValueError("Head logits, targets and validity mask must align")
    rows = F.binary_cross_entropy_with_logits(
        logits.float(), targets.detach().float(), reduction="none",
    )
    return rows.masked_fill(~valid, 0.0).sum(), valid.sum().float()


def response_bce(terms):
    """Divide once after all chunks, preserving a differentiable empty zero."""
    numerator, count = terms
    return numerator / count.clamp_min(1.0)


class ScalarLinearHead(torch.nn.Module):
    """One BF16 scalar projection initialized without touching global RNG."""

    def __init__(self, hidden_size, *, device):
        super().__init__()
        if not isinstance(hidden_size, int) or hidden_size <= 0:
            raise ValueError("Head hidden size must be a positive integer")
        generator = torch.Generator(device="cpu").manual_seed(HEAD_SEED)
        bound = 1.0 / math.sqrt(hidden_size)
        initial = torch.empty((1, hidden_size), dtype=torch.float32).uniform_(
            -bound, bound, generator=generator,
        )
        self.weight = torch.nn.Parameter(initial.to(device=device, dtype=torch.bfloat16))
        self.bias = torch.nn.Parameter(torch.zeros(1, device=device, dtype=torch.bfloat16))

    def forward(self, hidden):
        return F.linear(hidden, self.weight, self.bias).squeeze(-1)
