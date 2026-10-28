"""Pure, unweighted diagnostics for already-computed matched draft logits."""

import torch
from torch import Tensor


def _label_and_best_other(values: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
    top = values.topk(2, dim=-1)
    best_other = torch.where(
        top.indices[..., 0] == labels, top.values[..., 1], top.values[..., 0]
    )
    return values.gather(-1, labels.unsqueeze(-1)).squeeze(-1), best_other


@torch.no_grad()
def summarize_positions(
    reference_logits: Tensor,
    current_logits: Tensor,
    targets: Tensor,
    labels: Tensor,
    weights: Tensor,
) -> dict[str, Tensor]:
    """Return CPU tensors with shape [B,N,K], without reduction or masking.

    Inputs are [B,N,K,V] logits and fixed projected targets r, with [B,N,K]
    labels/weights. Original logits determine argmax, including native ties;
    distribution arithmetic is FP32. Weights validate layout only: the caller
    must exclude zero-weight rows and retain original response normalization.
    All label IDs must be in vocabulary, including zero-weight rows.

    kl_to_r is KL(r || q_current), despite its short historical field name.
    It is evaluated directly, with no clipping of small negative roundoff.
    Projected margins are r(y)-max_other(r) and log(r(y)/max_other(r)).
    Exact target zeros stay zero; an infinite log margin is retained.
    """
    if reference_logits.shape != current_logits.shape or targets.shape != current_logits.shape:
        raise ValueError("Reference, current logits and targets must have identical shapes.")
    if current_logits.ndim != 4 or current_logits.shape[-1] < 2:
        raise ValueError("Expected [B,N,K,V] logits with at least two vocabulary entries.")
    if labels.shape != current_logits.shape[:-1] or weights.shape != labels.shape:
        raise ValueError("Labels and weights must match the leading logit dimensions.")

    predicted = current_logits.argmax(-1)
    reference_correct = reference_logits.argmax(-1) == labels
    correct = predicted == labels
    logq = current_logits.float().log_softmax(-1)
    r = targets.float()
    rlogr = torch.xlogy(r, r)
    rlogq = r * logq
    current_y, current_other = _label_and_best_other(current_logits.float(), labels)
    target_y, target_other = _label_and_best_other(r, labels)
    values = {
        "predicted_id": predicted,
        "reference_correct": reference_correct,
        "correct": correct,
        "repaired": ~reference_correct & correct,
        "damaged": reference_correct & ~correct,
        "soft_ce": -rlogq.sum(-1),
        "target_entropy": -rlogr.sum(-1),
        "kl_to_r": (rlogr - rlogq).sum(-1),
        "target_token_logit_margin_vs_best_other": current_y - current_other,
        "projected_probability_margin": target_y - target_other,
        "projected_log_margin": target_y.log() - target_other.log(),
    }
    return {name: value.cpu() for name, value in values.items()}


def block_prefix_metrics(
    predicted_ids,
    labels,
    eligible,
    stop_ids,
    finish_reason,
    *,
    truncated_by_cap=False,
) -> dict:
    """Score one block's slots 1..15, with the known anchor already removed.

    Inputs are equal-length 1D tensors/sequences. Eligibility ends a contiguous
    prefix: masked slots are never skipped. first_mismatch_slot is 1-based and
    null when no eligible mismatch was observed. A correctly matched EOS counts
    in matching_prefix_length, but not matching_non_eos_prefix_length.

    Set truncated_by_cap only when this block reaches the saved cap boundary;
    finish_reason='length' alone cannot identify a block at that boundary.
    No unseen continuation is inferred, and this is not native acceptance.
    """
    predicted_ids = torch.as_tensor(predicted_ids).tolist()
    labels = torch.as_tensor(labels).tolist()
    eligible = torch.as_tensor(eligible, dtype=torch.bool).tolist()
    if not (len(predicted_ids) == len(labels) == len(eligible)):
        raise ValueError("Predictions, labels and eligibility must have equal lengths.")
    if any(isinstance(value, list) for value in predicted_ids + labels + eligible):
        raise ValueError("Prefix inputs must be one-dimensional.")
    if truncated_by_cap and finish_reason != "length":
        raise ValueError("Cap truncation requires the saved finish_reason='length'.")
    stop_ids = set(stop_ids)
    prefix = non_eos = 0
    matched_eos = False
    first_mismatch = None
    for slot, (predicted, label, valid) in enumerate(zip(predicted_ids, labels, eligible), 1):
        if not valid:
            break
        if predicted != label:
            first_mismatch = slot
            break
        prefix += 1
        if label in stop_ids:
            matched_eos = True
            break
        non_eos += 1
    eligible_count = sum(eligible)
    first_ineligible = eligible.index(False) if False in eligible else len(eligible)
    eligibility_gap = any(eligible[first_ineligible + 1 :])
    return {
        "eligible_count": eligible_count,
        "matching_prefix_length": prefix,
        "matching_non_eos_prefix_length": non_eos,
        "matched_eos": matched_eos,
        "first_mismatch_slot": first_mismatch,
        "cap_boundary_censored": bool(
            truncated_by_cap and prefix > 0 and prefix == eligible_count
            and not matched_eos and first_mismatch is None and not eligibility_gap
        ),
        "eligibility_gap": eligibility_gap,
    }
