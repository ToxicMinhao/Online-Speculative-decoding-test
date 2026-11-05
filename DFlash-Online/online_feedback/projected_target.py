"""Fixed preservation-and-repair targets for the first objective component.

All arithmetic is FP32. Softmax underflow remains zero: no probability floor,
post-projection normalization, or failed-projection repair is applied. For
finite logits the water-filling formula is also defined on those numerical
zeros. The returned targets never participate in autograd.

Qualification tolerances are fixed in the tests: 2e-6 absolute for probability
mass and the margin inequality; 2e-7 absolute / 2e-5 relative for the independent
FP64 oracle and the soft-CE gradient identity.
"""

import torch
from torch import Tensor


MARGIN_RATIO = 1.1


@torch.no_grad()
def projected_targets(reference_logits: Tensor, target_ids: Tensor) -> Tensor:
    """Return fixed targets of shape ``[..., vocabulary]`` at temperature one.

    ``target_ids`` has the reference's leading dimensions. The reference's
    argmax uses PyTorch's first (lowest token-ID) tie rule on the original
    logits, matching native argmax rather than rounded softmax probabilities.
    Correct decisions preserve the whole reference distribution. Other rows
    transfer mass from the largest competitors until the target has 1.1 times
    every competing probability.

    Call on a bounded position chunk; no response-sized vocabulary tensor needs
    to be retained by the caller.
    """
    if not torch.isfinite(reference_logits).all():
        raise ValueError("Projected targets require finite reference logits.")

    vocabulary_size = reference_logits.shape[-1]
    probabilities = reference_logits.float().softmax(dim=-1)
    rows = probabilities.reshape(-1, vocabulary_size)
    labels = target_ids.reshape(-1)
    incorrect = reference_logits.argmax(dim=-1).reshape(-1) != labels
    if not incorrect.any():
        return probabilities

    repair_rows = rows[incorrect]
    repair_labels = labels[incorrect, None]
    target_mass = repair_rows.gather(1, repair_labels)

    # Put the target after all competitors, then remove it from the sorted list.
    competitors = repair_rows.scatter(1, repair_labels, -torch.inf)
    competitors = competitors.sort(dim=-1, descending=True).values[:, :-1]
    active_counts = torch.arange(
        1, vocabulary_size, device=rows.device, dtype=torch.float32
    )
    thresholds = (target_mass + competitors.cumsum(dim=-1)) / (
        MARGIN_RATIO + active_counts
    )
    next_competitor = torch.cat(
        (competitors[:, 1:], torch.zeros_like(target_mass)), dim=-1
    )
    first_complete = (next_competitor <= thresholds).to(torch.int64).argmax(dim=-1)
    threshold = thresholds.gather(1, first_complete[:, None])

    repaired = torch.minimum(repair_rows, threshold)
    repaired.scatter_(1, repair_labels, MARGIN_RATIO * threshold)
    rows[incorrect] = repaired
    return probabilities
