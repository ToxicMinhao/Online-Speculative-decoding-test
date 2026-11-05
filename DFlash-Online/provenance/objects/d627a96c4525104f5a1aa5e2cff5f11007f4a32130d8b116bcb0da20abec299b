"""Current-forward proposal selection; no reference, auxiliary loss or RNG use."""

import torch


CE_OBJECTIVES = {"osd_ce", "osd_ce_first_error"}
FIRST_ERROR_OBJECTIVES = {"osd_first_error", "osd_ce_first_error"}
NEW_OBJECTIVES = CE_OBJECTIVES | FIRST_ERROR_OBJECTIVES


@torch.no_grad()
def valid_prefix_mask(labels, weights, eos_token_ids=()):
    """Return consecutive valid proposals, through canonical EOS inclusively.

    Slot zero is the known seed. A missing proposal or sequence boundary ends
    the block; a valid position after that gap cannot reopen it. An EOS seed
    therefore leaves no available continuation in that block.
    """
    terminal = torch.zeros_like(labels, dtype=torch.bool)
    for token_id in eos_token_ids:
        terminal |= labels == token_id
    after_terminal = terminal.long().cumsum(-1) - terminal.long() > 0
    valid = (weights > 0) & ~after_terminal
    prefix = torch.zeros_like(valid)
    prefix[..., 1:] = valid[..., 1:].long().cumprod(-1).bool()
    return prefix


@torch.no_grad()
def first_error_mask(proposals, labels, weights, eos_token_ids=()):
    """Select the current correct prefix and its first mismatch in each block."""
    valid = valid_prefix_mask(labels, weights, eos_token_ids)
    errors = valid & (proposals.detach() != labels)
    earlier_errors = errors.long().cumsum(-1) - errors.long()
    return valid & (earlier_errors == 0)


@torch.no_grad()
def selection_statistics(proposals, labels, weights, selection, eos_token_ids=()):
    """Return additive counts/weights for one observed forward.

    The entries are available/selected occurrences, available/selected weights,
    nonempty no-error/empty blocks, then first-error counts at offsets 1..B-1.
    Available occurrences use the existing validity weights; no-error and empty
    blocks use the contiguous proposal chain ending at padding or canonical EOS.
    Returning values rather than changing counters prevents checkpoint replay
    from counting occurrences twice.
    """
    eligible = weights > 0
    selected = eligible & selection
    valid = valid_prefix_mask(labels, weights, eos_token_ids)
    errors = valid & (proposals.detach() != labels)
    first_errors = errors & (errors.long().cumsum(-1) == 1)
    nonempty = valid.any(-1)
    summary = torch.stack((
        eligible.sum().float(), selected.sum().float(), weights.sum().float(),
        (weights * selection).sum().float(),
        (nonempty & ~errors.any(-1)).sum().float(), (~nonempty).sum().float(),
    ))
    histogram = first_errors.reshape(-1, labels.shape[-1]).sum(0)[1:].float()
    return torch.cat((summary, histogram))
