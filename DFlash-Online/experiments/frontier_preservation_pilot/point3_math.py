"""Point3's fixed-K assessment; teaching and its allocator remain unchanged.

The cumulative maximum chooses the earliest lowest-margin position in each
prefix. Selection is detached, but the selected margin remains differentiable.
No cumulative operation crosses a block boundary.
"""

import math

import torch
import torch.nn.functional as F


def max_terms_from_margins(margins, assessment_mask):
    """Return one obstruction term per assessed prefix and its selected branch.

    This small algebra helper preserves the input precision for smooth FP64
    tests. Native callers provide FP32 margins gathered from BF16 logits.
    Unassessed and empty positions have a graph-connected zero and are never
    gathered. Strict comparison implements earliest-position ties explicitly.
    """
    positions = margins.shape[-1]
    if not bool(torch.isfinite(margins[assessment_mask]).all()):
        raise ValueError('Nonfinite assessed margins')
    flat_margins = margins.reshape(-1, positions)
    flat_mask = assessment_mask.reshape(-1, positions)
    selected_indices = torch.full_like(flat_mask, -1, dtype=torch.long)
    tied_prefixes = torch.zeros_like(flat_mask)
    running_margin = margins.new_full((flat_margins.shape[0],), torch.inf)
    running_index = torch.full_like(running_margin, -1, dtype=torch.long)
    tie_count = torch.zeros_like(running_index)
    detached = flat_margins.detach()
    for position in range(positions):
        eligible = flat_mask[:, position]
        smaller = eligible & (detached[:, position] < running_margin)
        equal = eligible & (detached[:, position] == running_margin)
        running_index = torch.where(smaller, position, running_index)
        running_margin = torch.where(smaller, detached[:, position], running_margin)
        tie_count = torch.where(smaller, 1, tie_count + equal.long())
        selected_indices[:, position] = torch.where(eligible, running_index, -1)
        tied_prefixes[:, position] = eligible & (tie_count > 1)

    # Index only actual prefixes: an empty block never selects its seed slot.
    block_indices, prefix_indices = flat_mask.nonzero(as_tuple=True)
    bottlenecks = selected_indices[block_indices, prefix_indices]
    selected_margins = flat_margins[block_indices, bottlenecks]
    terms = torch.zeros_like(flat_margins) + margins.reshape(-1)[:0].sum()
    terms[block_indices, prefix_indices] = F.softplus(-selected_margins) / math.log(2.0)
    multiplicity = torch.zeros_like(flat_mask, dtype=torch.long)
    multiplicity.index_put_(
        (block_indices, bottlenecks), torch.ones_like(bottlenecks), accumulate=True
    )
    return terms.reshape_as(margins), dict(
        bottleneck_indices=selected_indices.reshape_as(assessment_mask),
        bottleneck_multiplicity=multiplicity.reshape_as(assessment_mask),
        tied_bottleneck_prefixes=tied_prefixes.reshape_as(assessment_mask),
    )


def _native_inputs(logits, labels, assessment_mask, collect_diagnostics):
    """Select both native vocabulary branches before promoting to FP32."""
    selected = logits[assessment_mask]
    selected_labels = labels[assessment_mask]
    if not bool(torch.isfinite(selected).all()):
        raise ValueError('Nonfinite assessed logits')
    wrong_logits = selected.detach().clone()
    wrong_logits.scatter_(-1, selected_labels.unsqueeze(-1), -torch.inf)
    rivals = wrong_logits.argmax(-1)
    canonical = selected.gather(-1, selected_labels.unsqueeze(-1)).squeeze(-1).float()
    competitor = selected.gather(-1, rivals.unsqueeze(-1)).squeeze(-1).float()
    margins = canonical - competitor
    if not collect_diagnostics:
        return margins, {}
    predictions = selected.argmax(-1)
    wrong_max = wrong_logits.gather(-1, rivals.unsqueeze(-1))
    diagnostics = dict(
        assessed_positions=int(assessment_mask.sum()),
        margin_ties=int((margins == 0).sum()),
        wrong_rival_ties=int(((wrong_logits == wrong_max).sum(-1) > 1).sum()),
        rivals=rivals.detach(),
        predictions=predictions.detach(),
        correct=(predictions == selected_labels).detach(),
        margins=margins.detach(),
    )
    return margins, diagnostics


def _numerators(logits, labels, assessment_mask, assessment_weight, collect_diagnostics):
    margins, diagnostics = _native_inputs(logits, labels, assessment_mask, collect_diagnostics)
    # Preserve the historical SUM operation order, including its FP32 reduction.
    weights = assessment_weight[assessment_mask].detach()
    sum_terms = weights * F.softplus(-margins) / math.log(2.0)
    dense_margins = torch.zeros_like(assessment_weight).float().masked_scatter(
        assessment_mask, margins
    )
    max_terms, bottlenecks = max_terms_from_margins(dense_margins, assessment_mask)
    diagnostics.update(bottlenecks)
    if collect_diagnostics:
        diagnostics.update(
            sum_numerator_fp64=float(sum_terms.detach().double().sum()),
            max_numerator_fp64=float(max_terms.detach()[assessment_mask].double().sum()),
        )
    return {'SUM': sum_terms.sum(), 'MAX': max_terms[assessment_mask].sum()}, diagnostics


def native_assessment_numerator(
    logits, labels, assessment_mask, assessment_weight, *, assessment='MAX',
    collect_diagnostics=False,
):
    """A complete-block head chunk, without a response or window denominator.

    The caller accumulates chunk numerators before applying the response's
    fixed count of nonempty blocks, then the two-response factor one half.
    """
    if assessment not in ('SUM', 'MAX'):
        raise ValueError('Point3 assessment must be SUM or MAX')
    numerators, diagnostics = _numerators(
        logits, labels, assessment_mask, assessment_weight, collect_diagnostics
    )
    return numerators[assessment], diagnostics


def assessment_pair(logits, roles):
    """Both response means from the same native margins, with branch readouts."""
    if logits.shape[:-1] != roles['labels'].shape:
        raise ValueError('Assessment logits lost their role binding')
    numerators, diagnostics = _numerators(
        logits, roles['labels'], roles['assessment_mask'], roles['assessment_weight'], True
    )
    count = int(roles['valid'].any(-1).sum())
    diagnostics['nonempty_blocks'] = count
    active = diagnostics['bottleneck_multiplicity'] > 0
    continuation = roles['assessment_mask'] & ~(roles['prefix'] | roles['repair'])
    diagnostics['active_bottlenecks_by_role'] = {
        role: dict(
            distinct_positions=int((active & mask).sum()),
            multiplicity=int((diagnostics['bottleneck_multiplicity'] * mask).sum()),
        )
        for role, mask in (
            ('P', roles['prefix']), ('E', roles['repair']), ('C', continuation)
        )
    }
    if count == 0:
        return numerators, diagnostics
    return {name: value / count for name, value in numerators.items()}, diagnostics


def cumulative_max_assessment(logits, roles):
    """Response-mean MAX assessment, matching math.assessment's return contract."""
    values, diagnostics = assessment_pair(logits, roles)
    return values['MAX'], diagnostics
