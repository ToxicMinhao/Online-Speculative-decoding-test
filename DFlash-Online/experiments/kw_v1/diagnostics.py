"""Compact evidence for exact qualification checks, without changing their gate.

Tensor pairs are compared at native bit precision. On failure we summarize every
different tensor in bounded chunks and retain a few coordinate examples, rather
than truncating the list of failed parameters or dumping their full contents.
"""
import math

import torch

from experiments.ok_continuation.resume import differences
from online_feedback.storage import write_json


CHUNK = 1_048_576
SAMPLE_COORDINATES = 6


def _number(value):
    value = float(value)
    return value if math.isfinite(value) else str(value)


def tensor_difference(left, right):
    """Count bit differences, including signed zero, and preserve nonfinites."""
    row = dict(shape=list(left.shape), dtype=str(left.dtype),
               other_shape=list(right.shape), other_dtype=str(right.dtype))
    if left.shape != right.shape or left.dtype != right.dtype:
        return {**row, 'comparable': False}
    a, b = left.detach().reshape(-1), right.detach().reshape(-1)
    changed = nonfinite_a = nonfinite_b = 0
    norm_a = norm_b = error_square = maximum = 0.0
    samples = []
    for offset in range(0, a.numel(), CHUNK):
        aa = a[offset:offset + CHUNK].cpu().contiguous()
        bb = b[offset:offset + CHUNK].cpu().contiguous()
        unequal = (aa.view(torch.uint8).reshape(-1, aa.element_size()) !=
                   bb.view(torch.uint8).reshape(-1, bb.element_size())).any(dim=1)
        changed += int(unequal.sum())
        x, y = aa.double(), bb.double()
        finite_x, finite_y = torch.isfinite(x), torch.isfinite(y)
        finite_pair = finite_x & finite_y
        nonfinite_a += int((~finite_x).sum())
        nonfinite_b += int((~finite_y).sum())
        norm_a += float(x[finite_x].square().sum())
        norm_b += float(y[finite_y].square().sum())
        error = x[finite_pair] - y[finite_pair]
        error_square += float(error.square().sum())
        if error.numel():
            maximum = max(maximum, float(error.abs().max()))
        if len(samples) < SAMPLE_COORDINATES:
            indices = unequal.nonzero().reshape(-1)[:SAMPLE_COORDINATES - len(samples)]
            for index in indices.tolist():
                samples.append(dict(flat_index=offset + index, left=_number(x[index]),
                                    right=_number(y[index]), error=_number(x[index] - y[index])))
    row.update(comparable=True, coordinates=a.numel(), differing_coordinates=changed,
               differing_fraction=changed / a.numel() if a.numel() else 0.0,
               nonfinite_left=nonfinite_a, nonfinite_right=nonfinite_b,
               max_abs_error_finite_pairs=_number(maximum),
               discrepancy_l2_finite_pairs=_number(math.sqrt(error_square)),
               left_l2_finite_coordinates=_number(math.sqrt(norm_a)),
               right_l2_finite_coordinates=_number(math.sqrt(norm_b)), samples=samples)
    return row


def _tensor_pairs(left, right, prefix='state'):
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        yield prefix, left, right
    elif isinstance(left, dict) and isinstance(right, dict):
        for key in left:
            if key in right:
                yield from _tensor_pairs(left[key], right[key], prefix + '.' + str(key))
    elif isinstance(left, (tuple, list)) and type(left) is type(right):
        for index, (a, b) in enumerate(zip(left, right)):
            yield from _tensor_pairs(a, b, prefix + '.' + str(index))


def compare_exact(left, right, path, checks, *, receipt_path=None):
    mismatch = differences(left, right)
    row = dict(check=path, status='failed' if mismatch else 'passed',
               criterion='native bit equality', differences=mismatch,
               differing_path_count=len(mismatch))
    if mismatch:
        failed = set(mismatch)
        row['tensor_differences'] = {
            name: tensor_difference(a, b) for name, a, b in _tensor_pairs(left, right)
            if name in failed}
    checks.append(row)
    if receipt_path is not None:
        write_json(receipt_path, checks)
    if mismatch:
        raise ValueError(path + ': ' + ', '.join(mismatch[:8]))
