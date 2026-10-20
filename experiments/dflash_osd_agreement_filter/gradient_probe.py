"""Bounded retained-graph probes for the failed gradient-sum qualification.

These helpers perform no optimization, sampling of their own, file writes or
acceptance tests. The caller restores the exact weights, RNG and fixed forward
arguments before every probe. Existing qualification tolerances are untouched.
Returned gradients are CPU FP32 copies; native precision already lost during
backward is not reconstructed by that conversion.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
import re

CHUNK_ELEMENTS = 1_048_576


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _parameters(parameters):
    import torch
    _require(isinstance(parameters, Mapping) and bool(parameters), 'Nonempty ordered parameter mapping required')
    _require(all(isinstance(name, str) and name for name in parameters), 'Named parameters required')
    values = tuple(parameters.values())
    _require(len({id(p) for p in values}) == len(values), 'Duplicate parameter identity')
    _require(all(isinstance(p, torch.Tensor) and p.is_leaf and p.requires_grad and p.numel() > 0
                 for p in values), 'Trainable nonempty leaf parameters required')
    return values


def _loss_label(label):
    _require(isinstance(label, str), 'Backward labels must be strings')
    if label in ('A', 'B', 'F'):
        return label
    if re.fullmatch(r'F[1-9][0-9]*', label):
        return 'F'
    raise ValueError('Supported backward labels are A, B, F or numbered full-loss repeats F1/F2')


def run_backward_probe(forward_factory, parameters, *, order=('A', 'B', 'F')):
    """Run one sampled forward and two/three requested backwards on its graph.

    ``forward_factory()`` returns ``(full_loss, loss_a, loss_b, metadata)``.
    Numbered F labels reuse the exact full-loss tensor, e.g. F1,F2 diagnoses
    repeated backward on one retained graph. ``autograd.grad`` does not write
    parameter ``.grad`` buffers. The caller owns state/RNG restoration; this
    function neither reruns the forward nor resamples between backward calls.
    """
    import torch
    _require(callable(forward_factory), 'A fixed forward factory is required')
    values = _parameters(parameters)
    _require(isinstance(order, (tuple, list)) and 1 <= len(order) <= 3 and len(set(order)) == len(order),
             'Use one to three distinct backward labels')
    labels = [_loss_label(label) for label in order]
    versions = tuple(p._version for p in values)
    result = forward_factory()  # Exactly one call; never silently retry.
    _require(isinstance(result, (tuple, list)) and len(result) == 4,
             'Forward factory must return full, A, B and metadata')
    full, a, b, metadata = result
    losses = {'F': full, 'A': a, 'B': b}
    _require(all(isinstance(loss, torch.Tensor) and loss.ndim == 0 and loss.requires_grad
                 for loss in losses.values()), 'All three losses must be scalar differentiable tensors')
    scalar_losses = {name: float(loss.detach()) for name, loss in losses.items()}
    gradients, native_dtypes = {}, {}
    for index, (label, selection) in enumerate(zip(order, labels)):
        computed = torch.autograd.grad(losses[selection], values, retain_graph=index < len(order)-1,
                                       create_graph=False, allow_unused=False)
        gradients[label], native_dtypes[label] = {}, {}
        for (name, parameter), gradient in zip(parameters.items(), computed):
            _require(gradient.shape == parameter.shape and not gradient.is_sparse,
                     'Native gradient shape/layout differs from its parameter')
            native_dtypes[label][name] = str(gradient.dtype)
            # Blocking copies ensure the result no longer depends on CUDA
            # storage before the next backward. copy=True also protects CPU
            # fixtures from aliasing retained graph buffers.
            gradients[label][name] = gradient.detach().to(device='cpu', dtype=torch.float32, copy=True)
        del computed, gradient
        _require(tuple(p._version for p in values) == versions,
                 'Forward/backward probe mutated trainable weights')
    return {
        'gradients': gradients, 'losses': scalar_losses, 'metadata': metadata,
        'backward_order': list(order), 'selected_loss_order': labels,
        'forward_calls': 1, 'backward_calls': len(order),
        'native_gradient_dtypes': native_dtypes,
        'returned_gradient_precision': 'CPU torch.float32 copied from native autograd results',
        'optimizer_steps': 0, 'file_writes': 0,
    }


def _finish(totals):
    finite = totals['nonfinite_left'] == totals['nonfinite_right'] == 0
    left_norm = math.sqrt(totals['left_sq']) if finite else None
    right_norm = math.sqrt(totals['right_sq']) if finite else None
    difference_norm = math.sqrt(totals['difference_sq']) if finite else None
    relative = (difference_norm / right_norm if right_norm else (0.0 if difference_norm == 0.0 else None)) if finite else None
    cosine = totals['dot'] / (left_norm * right_norm) if finite and left_norm and right_norm else None
    return {
        'elements': totals['elements'], 'different_coordinates': totals['different_coordinates'],
        'different_fraction': totals['different_coordinates'] / totals['elements'],
        'nonfinite_left': totals['nonfinite_left'], 'nonfinite_right': totals['nonfinite_right'],
        'nonfinite_pairs': totals['elements'] - totals['finite_pairs'], 'finite_pairs': totals['finite_pairs'],
        'all_values_finite': finite, 'exactly_equal': finite and totals['different_coordinates'] == 0,
        'left_norm': left_norm, 'right_norm': right_norm, 'difference_norm': difference_norm,
        'relative_l2_error_vs_right': relative,
        'relative_l2_error_defined': relative is not None,
        'cosine_similarity': cosine,
        'max_abs_difference': totals['max_abs_difference'] if finite else None,
        'finite_pairs_max_abs_difference': totals['max_abs_difference'] if totals['finite_pairs'] else None,
        'finite_pairs_difference_norm': math.sqrt(totals['difference_sq']),
        # Finite-pair quantities are explicitly named; excluded nonfinite values
        # must never look like a successful numerical comparison.
    }


def compare_gradients(left, right, parameters):
    """Descriptive FP64 comparison, without changing or applying any tolerance.

    Input maps may contain native or FP32 gradients on CPU/GPU. Only one 1M
    coordinate chunk per side is promoted to CPU FP64 at a time. ``parameters``
    provides the required ordered names/shapes and native precision; its weight
    values are not read. Nonfinite entries are counted and make aggregate norms
    undefined, while explicitly labeled finite-pair statistics remain available.
    """
    import torch
    _parameters(parameters)
    _require(isinstance(left, Mapping) and isinstance(right, Mapping) and
             list(left) == list(parameters) == list(right), 'Gradient inventories/order differ')
    per_parameter, all_totals = {}, []
    for name, parameter in parameters.items():
        l, r = left[name], right[name]
        _require(all(isinstance(x, torch.Tensor) and x.is_floating_point() and not x.is_sparse and
                     x.shape == parameter.shape for x in (l, r)), 'Gradient shape/precision/layout mismatch')
        l, r = l.detach().reshape(-1), r.detach().reshape(-1)
        totals = dict(elements=l.numel(), different_coordinates=0, nonfinite_left=0, nonfinite_right=0,
                      finite_pairs=0, left_sq=0.0, right_sq=0.0, difference_sq=0.0, dot=0.0,
                      max_abs_difference=0.0)
        reductions = {'left_sq': [], 'right_sq': [], 'difference_sq': [], 'dot': []}
        for start in range(0, l.numel(), CHUNK_ELEMENTS):
            lc = l[start:start+CHUNK_ELEMENTS].to(device='cpu', dtype=torch.float64)
            rc = r[start:start+CHUNK_ELEMENTS].to(device='cpu', dtype=torch.float64)
            lf, rf = torch.isfinite(lc), torch.isfinite(rc)
            valid = lf & rf
            totals['nonfinite_left'] += int((~lf).sum())
            totals['nonfinite_right'] += int((~rf).sum())
            totals['different_coordinates'] += int((lc != rc).sum())
            totals['finite_pairs'] += int(valid.sum())
            # BF16/FP32 inputs are exactly representable in FP64. Multiplication
            # here is diagnostic norm arithmetic, not an agreement sign test.
            lv, rv = lc[valid], rc[valid]
            difference = lv - rv
            reductions['left_sq'].append(float(lv.dot(lv)))
            reductions['right_sq'].append(float(rv.dot(rv)))
            reductions['difference_sq'].append(float(difference.dot(difference)))
            reductions['dot'].append(float(lv.dot(rv)))
            if difference.numel():
                totals['max_abs_difference'] = max(totals['max_abs_difference'], float(difference.abs().max()))
        for key, terms in reductions.items():
            totals[key] = math.fsum(terms)
        row = _finish(totals)
        row.update(name=name, shape=list(parameter.shape), native_parameter_dtype=str(parameter.dtype),
                   left_dtype=str(left[name].dtype), right_dtype=str(right[name].dtype))
        per_parameter[name] = row
        all_totals.append(totals)
    summed = {key: sum(row[key] for row in all_totals) for key in (
        'elements', 'different_coordinates', 'nonfinite_left', 'nonfinite_right', 'finite_pairs')}
    summed.update({key: math.fsum(row[key] for row in all_totals)
                   for key in ('left_sq', 'right_sq', 'difference_sq', 'dot')})
    summed['max_abs_difference'] = max(row['max_abs_difference'] for row in all_totals)
    return {
        'schema': 'agreement_gradient_comparison_v1', 'global': _finish(summed),
        'per_parameter': per_parameter, 'reduction_precision': 'CPU torch.float64; chunk reductions plus math.fsum',
        'chunk_elements': CHUNK_ELEMENTS, 'acceptance_tolerance_applied': False,
        'interpretation': 'Descriptive comparison only; existing qualification thresholds remain unchanged.',
    }
