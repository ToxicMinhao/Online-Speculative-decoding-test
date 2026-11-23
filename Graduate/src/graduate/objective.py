"""Native K+ with bounded correct-prefix allocation, adapted from DFlash-Online.

The source is online_feedback/{objective,kplus,first_error,prefix_allocation}.py.
Vocabulary arithmetic stays BF16; detached weights and response sums stay FP32.
"""

from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from graduate.config import ObjectiveSettings


class ResponseTerms(NamedTuple):
    """Pool chunk scalars before redistribution; successor groups hold first errors.

    An absent or incorrect next proposal counts as an unmatched successor.
    """

    native_numerator: torch.Tensor
    selected_weight: torch.Tensor
    unmatched_successor_weight: torch.Tensor
    matched_successor_weight: torch.Tensor
    unmatched_successor_numerator: torch.Tensor
    matched_successor_numerator: torch.Tensor
    prefix_delta: torch.Tensor
    changed_prefix_positions: torch.Tensor


@torch.no_grad()
def _prefix_weights(probability, weights, valid, prefix, settings):
    """Move at most 25% of each prefix weight while preserving its block's mass."""
    smoothing = settings.prefix_probability_smoothing
    factors = torch.where(valid, smoothing + (1 - smoothing) * probability, 1.0)
    products = factors.cumprod(-1) * valid
    # Score every valid proposal, including the tail after the first error.
    # Only the correct prefix receives redistributed weight.
    scores = products.flip(-1).cumsum(-1).flip(-1)

    prefix_mass = (weights * prefix).sum(-1, keepdim=True)
    score_mass = (scores * prefix).sum(-1, keepdim=True)
    desired_weights = prefix_mass * scores / torch.where(score_mass > 0, score_mass, 1.0)
    relative_change = torch.where(
        prefix, desired_weights / torch.where(prefix, weights, 1.0) - 1, 0.0,
    )
    max_change = relative_change.abs().amax(-1, keepdim=True)
    safe_max_change = torch.where(max_change > 0, max_change, 1.0)
    bounded_scale = torch.where(
        max_change > 0,
        (settings.prefix_relative_cap / safe_max_change).clamp(max=1),
        0.0,
    )
    allocated_weights = weights * (1 + bounded_scale * relative_change)
    movable = prefix & (prefix.sum(-1, keepdim=True) > 1)
    return torch.where(movable, allocated_weights, weights)


def compute_chunk(draft_logits, teacher_logits, labels, weights, stop_token_ids,
                  settings: ObjectiveSettings) -> ResponseTerms:
    """Consume [1, blocks, slots, vocabulary] logits and [1, blocks, slots] labels.

    Slot zero is the known seed. Weights already include response bounds and
    the native position decay. Nothing here samples anchors or shifts labels.
    """
    draft_logits = draft_logits.to(torch.bfloat16)
    with torch.no_grad():
        teacher_probability = F.softmax(teacher_logits.to(torch.bfloat16), dim=-1)
    log_probability = F.log_softmax(draft_logits, dim=-1)
    token_loss = F.kl_div(log_probability, teacher_probability, reduction="none").sum(-1)

    with torch.no_grad():
        # These coefficients also appear outside no_grad in the loss reduction.
        base_weights = weights.detach().float()
        matches = draft_logits.argmax(-1) == labels

        # Keep consecutive proposals after the seed, through canonical EOS.
        terminal = torch.zeros_like(labels, dtype=torch.bool)
        for token_id in stop_token_ids:
            terminal |= labels == token_id
        after_terminal = terminal.long().cumsum(-1) - terminal.long() > 0
        available = (base_weights > 0) & ~after_terminal
        valid = torch.zeros_like(available)
        valid[..., 1:] = available[..., 1:].long().cumprod(-1).bool()

        # K selects the correct prefix AND its first error. C uses only the prefix.
        errors = valid & ~matches
        earlier_errors = errors.long().cumsum(-1) - errors.long()
        selected = valid & (earlier_errors == 0)
        first_error = selected & errors
        prefix = selected & matches
        selected_weights = base_weights * selected

        # K+ prioritizes first errors followed by a valid, already-correct token.
        # A missing or incorrect successor belongs to the other group.
        successor_matches = torch.zeros_like(valid)
        successor_matches[..., :-1] = valid[..., 1:] & matches[..., 1:]
        unmatched_successor_weights = base_weights * (first_error & ~successor_matches)
        matched_successor_weights = base_weights * (first_error & successor_matches)

        canonical_log_probability = log_probability.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        # Exponentiate AFTER converting the native BF16 log-probability to FP32.
        canonical_probability = canonical_log_probability.float().exp()
        allocated_weights = _prefix_weights(canonical_probability, base_weights, valid, prefix, settings)
        changed_positions = ((allocated_weights != base_weights) & prefix).sum().float()

    return ResponseTerms(
        native_numerator=(token_loss * selected_weights).sum(),
        selected_weight=selected_weights.sum(),
        unmatched_successor_weight=unmatched_successor_weights.sum(),
        matched_successor_weight=matched_successor_weights.sum(),
        unmatched_successor_numerator=(token_loss * unmatched_successor_weights).sum(),
        matched_successor_numerator=(token_loss * matched_successor_weights).sum(),
        prefix_delta=((allocated_weights - base_weights) * token_loss).sum(),
        changed_prefix_positions=changed_positions,
    )


def reduce_response(terms: ResponseTerms, settings: ObjectiveSettings):
    """Apply response-wide K+, then the block-local prefix correction, once."""
    numerator = terms.native_numerator
    unmatched_weight = terms.unmatched_successor_weight
    matched_weight = terms.matched_successor_weight
    if bool((unmatched_weight > 0) & (matched_weight > 0)):
        numerator = numerator + settings.first_error_redistribution_fraction * (
            unmatched_weight * terms.matched_successor_numerator
            - matched_weight * terms.unmatched_successor_numerator
        ) / (unmatched_weight + 2 * matched_weight)
    kplus_numerator = numerator
    # Branch on coefficient movement, never the delta's value: a zero-valued
    # correction may still have a nonzero derivative. Neutral graphs stay exact.
    if bool(terms.changed_prefix_positions > 0):
        numerator = numerator + terms.prefix_delta
    denominator = terms.selected_weight + settings.denominator_epsilon
    loss = numerator / denominator.clamp_min(torch.finfo(torch.float32).tiny)
    metrics = {
        "native_numerator": terms.native_numerator.detach(),
        "kplus_numerator": kplus_numerator.detach(),
        "prefix_delta": terms.prefix_delta.detach(),
        "numerator": numerator.detach(),
        "denominator": denominator.detach(),
        # Preserve the established log names: priorities are 1 and 2 respectively.
        "priority1_weight": unmatched_weight.detach(),
        "priority2_weight": matched_weight.detach(),
        "changed_prefix_positions": terms.changed_prefix_positions.detach(),
    }
    return loss, metrics


def response_loss(draft_hidden, teacher_hidden, labels, weights, lm_head,
                  stop_token_ids, settings: ObjectiveSettings, chunk_blocks: int):
    """Project [1, blocks, slots, hidden] tensors in bounded checkpointed chunks."""
    def project_chunk(hidden, target_hidden, target_ids, position_weights):
        # compute_chunk owns the BF16 loss boundary; the native head is frozen.
        draft_logits = lm_head(hidden)
        with torch.no_grad():
            teacher_logits = lm_head(target_hidden)
        return compute_chunk(draft_logits, teacher_logits, target_ids, position_weights,
                             stop_token_ids, settings)

    totals = None
    for start in range(0, draft_hidden.shape[1], chunk_blocks):
        stop = start + chunk_blocks
        chunk = (draft_hidden[:, start:stop], teacher_hidden[:, start:stop],
                 labels[:, start:stop], weights[:, start:stop])
        if torch.is_grad_enabled() and draft_hidden.requires_grad:
            terms = checkpoint(project_chunk, *chunk, use_reentrant=False)
        else:
            terms = project_chunk(*chunk)
        totals = terms if totals is None else ResponseTerms(*(a + b for a, b in zip(totals, terms)))
    return reduce_response(totals, settings)
