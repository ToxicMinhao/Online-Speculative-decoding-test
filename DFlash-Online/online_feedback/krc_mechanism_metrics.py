"""Pure, compact accounting for the fixed KRC mechanism diagnostic.

Inputs are native argmax predictions and canonical margins already measured by
the qualified forward path. Nothing here changes model parameters or logits.
Prefix means use the same nonempty valid blocks for every candidate. Hinge and
gradient reductions here are descriptive FP64 reductions, not update arithmetic.
"""
from __future__ import annotations

import math

import torch

from .first_error import valid_prefix_mask
from .ranking import matching_prefix_mask


_COUNT_FIELDS = (
    'valid_blocks', 'pre_K_prefix_tokens', 'K_ONLY_prefix_tokens',
    'candidate_prefix_tokens', 'prefix_change_sum', 'gross_prefix_gains',
    'gross_prefix_losses', 'improved_blocks', 'regressed_blocks', 'tied_blocks',
    'old_prefix_tokens_lost', 'K_new_prefix_gain', 'K_prefix_gain_retained',
    'K_prefix_gain_lost', 'original_first_errors', 'K_useful_repairs',
    'C_useful_repairs', 'useful_repairs_retained', 'useful_repairs_lost',
    'useful_repairs_newly_obtained', 'protected_positions',
    'K_protected_errors', 'candidate_protected_errors',
    'protected_errors_repaired', 'new_protected_errors',
    'K_strict_negative_margin_errors', 'K_wrong_native_tie_errors',
    'candidate_strict_negative_margin_errors', 'candidate_wrong_native_tie_errors',
    'protected_weight', 'K_hinge_numerator_fp64', 'candidate_hinge_numerator_fp64',
)


def _cpu(value):
    return torch.as_tensor(value).detach().cpu()


def _ratio(numerator, denominator, empty_reason):
    if denominator == 0:
        return None, empty_reason
    return numerator / denominator, None


def _finish_summary(summary):
    summary['prefix_change_mean'], summary['prefix_change_mean_null_reason'] = _ratio(
        summary['prefix_change_sum'], summary['valid_blocks'], 'no_valid_blocks',
    )
    for phase in ('K', 'candidate'):
        summary[f'{phase}_hinge_fp64'], summary[f'{phase}_hinge_null_reason'] = _ratio(
            summary[f'{phase}_hinge_numerator_fp64'], summary['protected_weight'],
            'empty_protection',
        )
    summary['prefix_mean_denominator'] = 'fixed_nonempty_valid_blocks'
    summary['hinge_denominator'] = 'pooled_frozen_protected_weight'
    summary['reporting_reduction_dtype'] = 'float64'
    return summary


def _error_masks(predictions, labels, margins, valid):
    wrong = predictions != labels
    if bool((valid & ((wrong & (margins > 0)) | (~wrong & (margins < 0)))).any()):
        raise ValueError('Native predictions and canonical margins disagree')
    return wrong, wrong & (margins < 0), wrong & (margins == 0)


@torch.no_grad()
def response_metrics(response, post_k_predictions, candidate_predictions,
                     post_k_margins, candidate_margins, *, eos_token_ids=(),
                     identity=None):
    """Compare a candidate with K_ONLY on one captured response.

    ``response`` uses PrefixCorrection's labels/original_proposals/validity_weights/
    protected_weights/anchors/keep/record_id fields. Margins must contain the
    CURRENT strongest-rival margin at every position, not only protected slots.
    ``identity`` supplies event, pass, local/global step, role and candidate tags.
    Empty blocks remain in ``blocks`` but do not enter any block mean or count.
    Protected-error repair/new-error counts are relative to post-K predictions.
    """
    labels = _cpu(response['labels'])
    original = _cpu(response['original_proposals'])
    validity = _cpu(response['validity_weights'])
    weights = _cpu(response['protected_weights'])
    ordinary = _cpu(post_k_predictions)
    candidate = _cpu(candidate_predictions)
    k_margin = _cpu(post_k_margins)
    c_margin = _cpu(candidate_margins)
    if labels.ndim < 2 or labels.shape[-1] < 1:
        raise ValueError('Expected captured blocks with a seed position')
    tensors = (original, validity, weights, ordinary, candidate, k_margin, c_margin)
    if any(tensor.shape != labels.shape for tensor in tensors):
        raise ValueError('Predictions, margins and captured masks must share the labels shape')
    if not all(bool(torch.isfinite(tensor).all()) for tensor in tensors):
        raise ValueError('Nonfinite captured metric input')
    if bool((weights < 0).any()) or bool((validity < 0).any()):
        raise ValueError('Negative protection or validity weight')
    anchors = _cpu(response['anchors'])
    keep = _cpu(response['keep']).bool()
    if anchors.shape != labels.shape[:-1] or keep.shape != anchors.shape:
        raise ValueError('Anchors and keep mask must identify the captured blocks')

    valid = valid_prefix_mask(labels, validity, eos_token_ids)
    old_prefix = matching_prefix_mask(original, labels, validity, eos_token_ids)
    protected = weights > 0
    if not torch.equal(protected, old_prefix):
        raise ValueError('Frozen protected membership differs from the pre-K native prefix')
    if bool((valid.any(-1) & ~keep).any()):
        raise ValueError('Dropped anchor unexpectedly contains valid proposals')
    k_wrong, k_strict, k_ties = _error_masks(ordinary, labels, k_margin, valid)
    c_wrong, c_strict, c_ties = _error_masks(candidate, labels, c_margin, valid)
    protected_k_wrong = protected & k_wrong
    first_mask = protected_k_wrong & (protected_k_wrong.long().cumsum(-1) == 1)
    first_oracle = torch.where(first_mask, labels, ordinary)
    all_oracle = torch.where(protected_k_wrong, labels, ordinary)
    predictions = (original, ordinary, candidate, first_oracle, all_oracle)
    lengths = [matching_prefix_mask(p, labels, validity, eos_token_ids).sum(-1)
               for p in predictions]
    available = valid.sum(-1)
    shape = labels.shape
    width = shape[-1]
    flat = lambda tensor: tensor.reshape(-1, width)
    labels, original, ordinary, candidate, valid, protected, weights, k_margin, c_margin = (
        flat(tensor) for tensor in
        (labels, original, ordinary, candidate, valid, protected, weights, k_margin, c_margin)
    )
    k_wrong, c_wrong, k_strict, k_ties, c_strict, c_ties, first_mask = (
        flat(tensor) for tensor in
        (k_wrong, c_wrong, k_strict, k_ties, c_strict, c_ties, first_mask)
    )
    lengths = [tensor.reshape(-1).tolist() for tensor in lengths]
    summary = {key: 0 for key in _COUNT_FIELDS}
    blocks = []
    for index in range(labels.shape[0]):
        pre, k, current, first, all_fixed = [values[index] for values in lengths]
        n_valid = int(available.reshape(-1)[index])
        is_valid = n_valid > 0
        difference = current - k
        new_gain = max(0, k - pre)
        retained = min(new_gain, max(0, current - pre))
        old_error = pre < n_valid
        repaired_k, repaired_c = old_error and k > pre, old_error and current > pre
        membership = protected[index]
        first_positions = first_mask[index].nonzero().flatten().tolist()
        first_position = first_positions[0] if first_positions else None
        first_kind = None
        if first_position is not None:
            first_kind = 'wrong_native_tie' if k_margin[index, first_position] == 0 else 'strict_negative_margin'
        counts = {
            'valid_blocks': int(is_valid), 'pre_K_prefix_tokens': pre,
            'K_ONLY_prefix_tokens': k, 'candidate_prefix_tokens': current,
            'prefix_change_sum': difference, 'gross_prefix_gains': max(0, difference),
            'gross_prefix_losses': max(0, -difference),
            'improved_blocks': int(is_valid and difference > 0),
            'regressed_blocks': int(is_valid and difference < 0),
            'tied_blocks': int(is_valid and difference == 0),
            'old_prefix_tokens_lost': max(0, pre - current),
            'K_new_prefix_gain': new_gain, 'K_prefix_gain_retained': retained,
            'K_prefix_gain_lost': new_gain - retained,
            'original_first_errors': int(old_error), 'K_useful_repairs': int(repaired_k),
            'C_useful_repairs': int(repaired_c),
            'useful_repairs_retained': int(repaired_k and repaired_c),
            'useful_repairs_lost': int(repaired_k and not repaired_c),
            'useful_repairs_newly_obtained': int(not repaired_k and repaired_c),
            'protected_positions': int(membership.sum()),
            'K_protected_errors': int((membership & k_wrong[index]).sum()),
            'candidate_protected_errors': int((membership & c_wrong[index]).sum()),
            'protected_errors_repaired': int((membership & k_wrong[index] & ~c_wrong[index]).sum()),
            'new_protected_errors': int((membership & ~k_wrong[index] & c_wrong[index]).sum()),
            'K_strict_negative_margin_errors': int((membership & k_strict[index]).sum()),
            'K_wrong_native_tie_errors': int((membership & k_ties[index]).sum()),
            'candidate_strict_negative_margin_errors': int((membership & c_strict[index]).sum()),
            'candidate_wrong_native_tie_errors': int((membership & c_ties[index]).sum()),
            'protected_weight': float(weights[index].double().sum()),
            'K_hinge_numerator_fp64': float((weights[index].double() * (-k_margin[index].double()).clamp_min(0)).sum()),
            'candidate_hinge_numerator_fp64': float((weights[index].double() * (-c_margin[index].double()).clamp_min(0)).sum()),
        }
        for key, value in counts.items():
            summary[key] += value
        blocks.append({
            **(identity or {}), 'record_id': response['record_id'],
            'block_index': index, 'anchor': int(anchors.reshape(-1)[index]),
            'kept_anchor': bool(keep.reshape(-1)[index]),
            'available_prefix_tokens': n_valid, **counts,
            'original_first_error_offset': pre + 1 if old_error else None,
            'original_first_error_canonical_token': int(labels[index, pre + 1]) if old_error else None,
            'K_prediction_at_original_first_error': int(ordinary[index, pre + 1]) if old_error else None,
            'candidate_prediction_at_original_first_error': int(candidate[index, pre + 1]) if old_error else None,
            'canonical_tokens': labels[index].tolist(),
            'valid_positions': valid[index].tolist(),
            'protected_weights': weights[index].tolist(),
            'pre_K_predictions': original[index].tolist(),
            'K_ONLY_predictions': ordinary[index].tolist(),
            'candidate_predictions': candidate[index].tolist(),
            'K_ONLY_canonical_margins': k_margin[index].tolist(),
            'candidate_canonical_margins': c_margin[index].tolist(),
            'oracle_first_protected_error_offset': first_position,
            'oracle_first_protected_error_kind': first_kind,
            'oracle_first_prefix_length': first,
            'oracle_all_prefix_length': all_fixed,
            'oracle_first_prefix_gain': first - k,
            'oracle_all_prefix_gain': all_fixed - k,
        })
    result = {'blocks': blocks, 'summary': _finish_summary(summary)}
    result['oracles'] = _oracle_summary(blocks)
    return result


def _oracle_summary(blocks):
    valid = [block for block in blocks if block['valid_blocks']]
    results = {}
    for name, field in (('FIRST_PROTECTED_ERROR', 'first'), ('ALL_PROTECTED_ERRORS', 'all')):
        gain_key = f'oracle_{field}_prefix_gain'
        total = sum(block[gain_key] for block in valid)
        mean, reason = _ratio(total, len(valid), 'no_valid_blocks')
        results[name] = {
            'prefix_gain_sum': total, 'prefix_gain_mean': mean,
            'prefix_gain_mean_null_reason': reason, 'valid_blocks': len(valid),
            'improved_blocks': sum(block[gain_key] > 0 for block in valid),
            'zero_opportunity_blocks': sum(block[gain_key] == 0 for block in valid),
            'blocks_with_protected_errors': sum(block['K_protected_errors'] > 0 for block in valid),
        }
    first = results['FIRST_PROTECTED_ERROR']
    for kind in ('wrong_native_tie', 'strict_negative_margin'):
        selected = [block for block in valid if block['oracle_first_protected_error_kind'] == kind]
        first[f'{kind}_blocking_errors'] = len(selected)
        first[f'{kind}_prefix_gain_sum'] = sum(block['oracle_first_prefix_gain'] for block in selected)
    results['all_minus_first_prefix_gain_sum'] = sum(
        block['oracle_all_prefix_gain'] - block['oracle_first_prefix_gain'] for block in valid
    )
    results['interpretation'] = 'hypothetical_token_replacements_on_fixed_training_anchors'
    return results


def pooled_summary(response_results):
    """Pool pair numerators/denominators; never average response hinge values."""
    responses = list(response_results)
    summary = {key: sum(result['summary'][key] for result in responses) for key in _COUNT_FIELDS}
    return {
        'summary': _finish_summary(summary),
        'oracles': _oracle_summary([block for result in responses for block in result['blocks']]),
        'responses': len(responses),
    }


def fitted_objective_predictions(value, alpha, branch_status, proposed_dot,
                                 realized_dot, actual_value):
    """Separate a formal projection ideal from measured linearized predictions.

    The ideal applies only to an active fitted correction. Inactive positive-V
    zero-gradient branches do not have a justified zero-objective prediction.
    Actual-minus-linearized residuals do not uniquely diagnose nonlinearity:
    nonsmoothness and finite-precision gradient/forward effects remain possible.
    """
    numbers = tuple(float(number) for number in
                    (value, alpha, proposed_dot, realized_dot, actual_value))
    if not all(math.isfinite(number) for number in numbers):
        raise ValueError('Nonfinite fitted-objective prediction input')
    value, alpha, proposed_dot, realized_dot, actual_value = numbers
    if alpha < 0 or value < 0 or actual_value < 0:
        raise ValueError('Negative dose or hinge value')
    active = branch_status == 'applied'
    formal_ideal = (1 - alpha) * value
    return {
        'V_before': value, 'alpha': alpha, 'branch_status': branch_status,
        'formal_ideal': formal_ideal,
        'ideal': formal_ideal if active else None,
        'ideal_null_reason': None if active else f'inactive_correction:{branch_status}',
        'g_dot_proposed_fp32_displacement': proposed_dot,
        'g_dot_realized_bf16_displacement': realized_dot,
        'proposed_linearized_V': value + proposed_dot,
        'realized_linearized_V': value + realized_dot,
        'actual_V': actual_value,
        'proposed_minus_formal_ideal': value + proposed_dot - formal_ideal,
        'realized_minus_proposed': realized_dot - proposed_dot,
        'actual_minus_realized_linearized': actual_value - value - realized_dot,
        'reporting_reduction_dtype': 'float64',
    }


@torch.no_grad()
def gradient_alignment(reference_gradients, diagnostic_gradients):
    """Compare already-computed gradient lists, allowing exact CPU staging."""
    reference_gradients = tuple(reference_gradients)
    diagnostic_gradients = tuple(diagnostic_gradients)
    if len(reference_gradients) != len(diagnostic_gradients):
        raise ValueError('Gradient lists have different lengths')
    dot = norm_r = norm_d = 0.0
    for reference, diagnostic in zip(reference_gradients, diagnostic_gradients):
        if reference.shape != diagnostic.shape:
            raise ValueError('Corresponding gradient shapes differ')
        reference = reference.detach().double()
        diagnostic = diagnostic.detach().to(device=reference.device, dtype=torch.float64)
        if not bool(torch.isfinite(reference).all() & torch.isfinite(diagnostic).all()):
            raise ValueError('Nonfinite diagnostic gradient')
        dot += float((reference * diagnostic).sum())
        norm_r += float(reference.square().sum())
        norm_d += float(diagnostic.square().sum())
    cosine, reason = _ratio(dot, math.sqrt(norm_r) * math.sqrt(norm_d), 'zero_gradient_norm')
    return {
        'g_R_dot_g_D': dot, 'g_R_norm_fp64': math.sqrt(norm_r),
        'g_D_norm_fp64': math.sqrt(norm_d), 'cosine': cosine,
        'cosine_null_reason': reason, 'reporting_reduction_dtype': 'float64',
    }
