"""Bounded, durable evidence for the unchanged grouped-gradient qualification.

Statistics never alter gradients, precision, or the acceptance threshold. The
caller records the complete summary *before* enforcing the original failure.
Only a bounded coordinate sample leaves device memory; no complete gradient
array is serialized. FP64 reductions are chunked independently of the FP32
arithmetic used by the existing tolerance check.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import struct
import tempfile

CHUNK_ELEMENTS = 1_048_576
SUM_RULE = 'abs(sum-reference) <= 8*eps_native*(abs(A)+abs(B)+abs(reference))+1e-8'
NONFINITE_ERROR = 'Nonfinite gradient qualification'
SUM_ERROR = 'Grouped raw gradients differ beyond declared sum tolerance'


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return {'nonfinite_float': repr(value),
                'ieee754_binary64_hex': struct.pack('>d', value).hex()}
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


def _maximum(previous, value):
    # Do not let Python's order-sensitive max silently hide a NaN.
    return float('nan') if math.isnan(previous) or math.isnan(value) else max(previous, value)


def _index(flat_index, shape):
    result = []
    for size in reversed(shape):
        result.append(flat_index % size)
        flat_index //= size
    return list(reversed(result))


def summarize_gradient_sum(raw_a, raw_b, reference, parameters, *, context=None, sample_limit=32):
    """Measure all parameters without raising on numerical failure.

    Ordered mappings must cover the same nonempty parameter inventory. A/B and
    reference values are evaluated in FP32, exactly as in the original check.
    ``context`` is caller-provided event/pass/trial, response/state/anchor/group
    identity, losses and denominator metadata. It contains no tensor payloads.
    Samples are the first failing coordinates in parameter/flattened order;
    ``sample_limit`` is a *global* bound, not a per-parameter allowance.
    """
    import torch

    if (not parameters or list(raw_a) != list(parameters) or
            list(raw_b) != list(parameters) or list(reference) != list(parameters)):
        raise ValueError('Gradient diagnostic parameter inventories differ')
    if type(sample_limit) is not int or sample_limit < 0:
        raise ValueError('Gradient diagnostic sample_limit must be a nonnegative integer')
    norm_labels = ('a', 'b', 'sum', 'reference', 'discrepancy', 'native_cast_discrepancy')
    global_sq = dict.fromkeys(norm_labels, 0.0)
    counts = dict.fromkeys(('coordinates', 'differing_coordinates',
                            'native_cast_differing_coordinates', 'tolerance_violation_coordinates',
                            'nonfinite_input_coordinates', 'rejected_coordinates'), 0)
    nonfinite = dict.fromkeys(('a', 'b', 'sum', 'reference', 'error', 'tolerance', 'error_over_tolerance'), 0)
    max_error = max_scaled = native_max_error = 0.0
    per_parameter, samples = [], []
    first_failure = None
    with torch.no_grad():
        for name, parameter in parameters.items():
            originals = [raw_a[name], raw_b[name], reference[name]]
            if any(value.shape != parameter.shape or value.device != parameter.device for value in originals):
                raise ValueError(f'Gradient diagnostic shape/device differs for {name}')
            if parameter.numel() == 0:
                raise ValueError(f'Gradient diagnostic empty parameter: {name}')
            arrays = [value.detach().reshape(-1) for value in originals]
            item = dict(name=name, shape=list(parameter.shape), parameter_dtype=str(parameter.dtype),
                        raw_a_dtype=str(raw_a[name].dtype), raw_b_dtype=str(raw_b[name].dtype),
                        reference_dtype=str(reference[name].dtype), comparison_dtype='torch.float32',
                        reduction_dtype='torch.float64', device=str(parameter.device),
                        native_epsilon=torch.finfo(parameter.dtype).eps,
                        max_abs_error=0.0, max_error_over_tolerance=0.0,
                        native_cast_sum_max_abs_error=0.0)
            local_counts = dict.fromkeys(counts, 0)
            local_nonfinite = dict.fromkeys(nonfinite, 0)
            local_sq = dict.fromkeys(norm_labels, 0.0)
            for start in range(0, parameter.numel(), CHUNK_ELEMENTS):
                a, b, ref = (value[start:start + CHUNK_ELEMENTS].float() for value in arrays)
                aggregate = a + b
                error = (aggregate - ref).abs()
                native_error = (aggregate.to(parameter.dtype).float() - ref).abs()
                tolerance = 8 * torch.finfo(parameter.dtype).eps * (a.abs() + b.abs() + ref.abs()) + 1e-8
                ratio = error / tolerance
                raw_finite = torch.isfinite(a) & torch.isfinite(b) & torch.isfinite(ref)
                # Identical acceptance expression to the original check. NaN
                # fails <=. Nonfinite *input* values separately fail as before.
                tolerance_failure = ~(error <= tolerance)
                rejected = ~raw_finite | tolerance_failure
                numbers = dict(coordinates=error.numel(), differing_coordinates=int((error != 0).sum()),
                               native_cast_differing_coordinates=int((native_error != 0).sum()),
                               tolerance_violation_coordinates=int(tolerance_failure.sum()),
                               nonfinite_input_coordinates=int((~raw_finite).sum()),
                               rejected_coordinates=int(rejected.sum()))
                for key, count in numbers.items():
                    local_counts[key] += count
                for key, value in (('a', a), ('b', b), ('sum', aggregate), ('reference', ref),
                                   ('error', error), ('tolerance', tolerance), ('error_over_tolerance', ratio)):
                    local_nonfinite[key] += int((~torch.isfinite(value)).sum())
                for key, value in zip(norm_labels, (a, b, aggregate, ref, error, native_error)):
                    local_sq[key] += float(value.double().square().sum())
                item['max_abs_error'] = _maximum(item['max_abs_error'], float(error.max()))
                item['max_error_over_tolerance'] = _maximum(item['max_error_over_tolerance'], float(ratio.max()))
                item['native_cast_sum_max_abs_error'] = _maximum(item['native_cast_sum_max_abs_error'], float(native_error.max()))
                if numbers['rejected_coordinates']:
                    if first_failure is None:
                        first_failure = dict(parameter=name, chunk_start=start,
                                             error=NONFINITE_ERROR if numbers['nonfinite_input_coordinates'] else SUM_ERROR)
                    remaining = sample_limit - len(samples)
                    if remaining > 0:
                        selected = torch.nonzero(rejected, as_tuple=False).flatten()[:remaining]
                        selected_indices = selected.cpu().tolist()
                        selected_values = torch.stack((a[selected], b[selected], aggregate[selected], ref[selected],
                                                       error[selected], tolerance[selected], ratio[selected]), dim=1).cpu().tolist()
                        for offset, values in zip(selected_indices, selected_values):
                            flat = start + offset
                            samples.append(dict(parameter=name, flat_index=flat, index=_index(flat, parameter.shape),
                                                **dict(zip(('g_a', 'g_b', 'sum', 'full_gradient', 'error', 'tolerance',
                                                            'error_over_tolerance'), values))))
            for key, count in local_counts.items():
                counts[key] += count
            for key, count in local_nonfinite.items():
                nonfinite[key] += count
            for key, value in local_sq.items():
                global_sq[key] += value
            item.update(local_counts)
            item['tolerance_violation_fraction'] = local_counts['tolerance_violation_coordinates'] / parameter.numel()
            item['rejected_fraction'] = local_counts['rejected_coordinates'] / parameter.numel()
            item['nonfinite_counts'] = local_nonfinite
            item['l2_norms'] = {key: math.sqrt(value) for key, value in local_sq.items()}
            item['status'] = 'failed' if local_counts['rejected_coordinates'] else 'passed'
            per_parameter.append(item)
            max_error = _maximum(max_error, item['max_abs_error'])
            max_scaled = _maximum(max_scaled, item['max_error_over_tolerance'])
            native_max_error = _maximum(native_max_error, item['native_cast_sum_max_abs_error'])
    norms = {key: math.sqrt(value) for key, value in global_sq.items()}
    total = counts['coordinates']
    summary = dict(schema='agreement_gradient_sum_diagnostic_v1',
                   status='failed' if counts['rejected_coordinates'] else 'passed',
                   context={} if context is None else context,
                   rule=SUM_RULE, max_abs_error=max_error, max_error_over_tolerance=max_scaled,
                   reference_l2_norm=norms['reference'], sum_error_l2_norm=norms['discrepancy'],
                   native_cast_sum_error_l2_norm=norms['native_cast_discrepancy'],
                   native_cast_sum_max_abs_error=native_max_error,
                   differing_fraction=counts['differing_coordinates'] / total,
                   native_cast_differing_fraction=counts['native_cast_differing_coordinates'] / total,
                   tolerance_violation_fraction=counts['tolerance_violation_coordinates'] / total,
                   rejected_fraction=counts['rejected_coordinates'] / total,
                   nonfinite_counts=nonfinite, l2_norms=norms, per_parameter=per_parameter,
                   first_failure=first_failure, samples=samples, sample_limit=sample_limit,
                   sample_selection='first rejected coordinates in parameter then flattened-index order; globally bounded',
                   reduction_chunk_elements=CHUNK_ELEMENTS,
                   reference='same sampled forward and current weights; no optimizer step or new anchor draw',
                   **counts)
    return _json_safe(summary)


def enforce_gradient_sum(summary):
    """Raise the original error after the caller has saved the summary."""
    if summary['status'] == 'failed':
        raise ValueError(summary['first_failure']['error'])
    if summary['status'] != 'passed':
        raise ValueError('Malformed gradient qualification summary status')
    return summary


def durable_write(path, summary):
    """Atomic strict-JSON replacement with file and parent-directory fsync."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(_json_safe(summary), indent=2, sort_keys=True, allow_nan=False) + '\n'
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix='.' + path.name + '.', suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path
