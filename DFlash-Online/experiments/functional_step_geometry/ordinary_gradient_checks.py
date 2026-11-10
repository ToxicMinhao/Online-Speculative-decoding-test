"""Exact ordinary-gradient checks with bounded, JSON-safe diagnostics.

The right-hand gradient is the reference for relative error. This helper does
not apply numerical tolerances: matching metadata, finite values and exact
coordinate equality are all required. FP64 diagnostic arithmetic is confined
to small CPU chunks, never a model-sized conversion.
"""
import math

import torch


CHUNK_COORDINATES = 262_144
SAMPLE_LIMIT = 8


def _number(value):
    value = float(value)
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "Infinity" if value > 0 else "-Infinity"
    return value


def _value_totals():
    return dict(coordinates=0, finite_coordinates=0, nan_coordinates=0,
                positive_infinity_coordinates=0, negative_infinity_coordinates=0,
                finite_l2=0.0)


def _add_values(total, values):
    finite = torch.isfinite(values)
    total['coordinates'] += values.numel()
    total['finite_coordinates'] += int(finite.sum())
    total['nan_coordinates'] += int(torch.isnan(values).sum())
    total['positive_infinity_coordinates'] += int(torch.isposinf(values).sum())
    total['negative_infinity_coordinates'] += int(torch.isneginf(values).sum())
    total['finite_l2'] = math.hypot(
        total['finite_l2'], float(torch.linalg.vector_norm(values[finite])))
    return finite


def _finish_values(total):
    result = dict(total)
    if total['nan_coordinates']:
        result['l2'] = "NaN"
    elif total['positive_infinity_coordinates'] or total['negative_infinity_coordinates']:
        result['l2'] = "Infinity"
    else:
        result['l2'] = _number(total['finite_l2'])
    result['finite_l2'] = _number(total['finite_l2'])
    return result


def _flatten(tensor):
    # Inputs are retained CPU snapshots. Detaching preserves both values and
    # the caller's autograd state; a noncontiguous input is copied only in its
    # original dtype by reshape, not expanded to FP64 as a whole.
    if tensor.device.type != 'cpu':
        raise ValueError('Compare retained CPU gradients, not live GPU tensors')
    return tensor.detach().reshape(-1)


def _single_values(tensor):
    total = _value_totals()
    if tensor is not None:
        values = _flatten(tensor)
        for start in range(0, values.numel(), CHUNK_COORDINATES):
            _add_values(total, values[start:start + CHUNK_COORDINATES].double())
    return _finish_values(total)


def _relative(discrepancy, reference):
    if not isinstance(discrepancy, (int, float)) or not isinstance(reference, (int, float)):
        return 'undefined'
    if reference == 0:
        return 0.0 if discrepancy == 0 else 'Infinity'
    return _number(discrepancy / reference)


def _paired(left, right):
    left_values, right_values = _flatten(left), _flatten(right)
    left_total, right_total = _value_totals(), _value_totals()
    different = nonfinite = finite_pairs = nonfinite_errors = 0
    discrepancy = maximum = 0.0
    samples = []
    for start in range(0, left_values.numel(), CHUNK_COORDINATES):
        a = left_values[start:start + CHUNK_COORDINATES].double()
        b = right_values[start:start + CHUNK_COORDINATES].double()
        finite = _add_values(left_total, a) & _add_values(right_total, b)
        bad = (a != b) | ~finite
        different += int(bad.sum())
        nonfinite += int((~finite).sum())
        finite_pairs += int(finite.sum())
        error = a - b
        finite_error = finite & torch.isfinite(error)
        nonfinite_errors += int((finite & ~torch.isfinite(error)).sum())
        differences = error[finite_error]
        discrepancy = math.hypot(discrepancy, float(torch.linalg.vector_norm(differences)))
        if differences.numel():
            maximum = max(maximum, float(differences.abs().max()))
        if len(samples) < SAMPLE_LIMIT:
            indices = bad.nonzero().flatten()[:SAMPLE_LIMIT - len(samples)]
            for index in indices.tolist():
                samples.append(dict(index=start + index, left=_number(a[index]),
                                    right=_number(b[index]), error=_number(error[index])))
    full_discrepancy = 'undefined' if nonfinite else (
        'Infinity' if nonfinite_errors else _number(discrepancy))
    left_summary, right_summary = _finish_values(left_total), _finish_values(right_total)
    return dict(compared_coordinates=left_values.numel(),
                differing_coordinates=different, nonfinite_coordinates=nonfinite,
                finite_pair_coordinates=finite_pairs,
                nonfinite_difference_coordinates=nonfinite_errors,
                finite_discrepancy_l2=_number(discrepancy),
                discrepancy_l2=full_discrepancy,
                finite_maximum_absolute_error=_number(maximum),
                maximum_absolute_error='undefined' if nonfinite else (
                    'Infinity' if nonfinite_errors else _number(maximum)),
                relative_l2=_relative(full_discrepancy, right_summary['l2']),
                left=left_summary, right=right_summary, samples=samples)


def _parameter(name, left, right, left_present, right_present):
    row = dict(name=name, left_present=left_present, right_present=right_present,
               left_shape=None if left is None else list(left.shape),
               right_shape=None if right is None else list(right.shape),
               left_dtype=None if left is None else str(left.dtype),
               right_dtype=None if right is None else str(right.dtype))
    comparable = left is not None and right is not None and left.shape == right.shape
    layout_matches = (left_present == right_present and row['left_shape'] == row['right_shape']
                      and row['left_dtype'] == row['right_dtype'])
    row['layout_matches'] = layout_matches
    if comparable:
        row.update(_paired(left, right))
        row['exact_equal'] = layout_matches and row['differing_coordinates'] == 0
    else:
        left_summary, right_summary = _single_values(left), _single_values(right)
        # Unpaired entries have no shared coordinate system. Count each side's
        # nonfinite values so a missing/malformed counterpart cannot hide them.
        nonfinite = sum(side['coordinates'] - side['finite_coordinates']
                        for side in (left_summary, right_summary))
        row.update(compared_coordinates=0, differing_coordinates=0, nonfinite_coordinates=nonfinite,
                   finite_pair_coordinates=0, nonfinite_difference_coordinates=0,
                   finite_discrepancy_l2=0.0, finite_maximum_absolute_error=0.0,
                   discrepancy_l2=0.0 if layout_matches else 'undefined',
                   maximum_absolute_error=0.0 if layout_matches else 'undefined',
                   relative_l2=0.0 if layout_matches else 'undefined',
                   left=left_summary, right=right_summary, samples=[],
                   exact_equal=layout_matches)
    return row


def compare_gradients(left, right):
    """Return exact parity plus compact per-parameter and global diagnostics.

    A missing parameter differs from a present ``None``. Two ``None`` values
    match; absent/shape-incompatible gradients have no paired coordinates and
    make the global discrepancy undefined. Nonfinite values always fail.
    Samples are the first eight failing coordinates in sorted parameter order.
    ``nonfinite_coordinates`` counts the union within paired tensors and the
    sum of side-specific nonfinite counts where coordinates cannot be paired.
    """
    parameters = [_parameter(name, left.get(name), right.get(name), name in left, name in right)
                  for name in sorted(left.keys() | right.keys())]
    total = dict(compared_tensors=sum(row['compared_coordinates'] > 0 for row in parameters),
                 parameter_names=len(parameters),
                 metadata_mismatches=sum(not row['layout_matches'] for row in parameters),
                 differing_tensors=sum(not row['exact_equal'] for row in parameters))
    for key in ('compared_coordinates', 'differing_coordinates', 'nonfinite_coordinates',
                'finite_pair_coordinates', 'nonfinite_difference_coordinates'):
        total[key] = sum(row[key] for row in parameters)
    for side in ('left', 'right'):
        values = _value_totals()
        for row in parameters:
            for key in values.keys() - {'finite_l2'}:
                values[key] += row[side][key]
            values['finite_l2'] = math.hypot(values['finite_l2'], float(row[side]['finite_l2']))
        total[side] = _finish_values(values)
        total[side + '_l2'] = total[side]['l2']
    discrepancy = math.hypot(*(float(row['finite_discrepancy_l2']) for row in parameters))
    maximum = max((float(row['finite_maximum_absolute_error']) for row in parameters), default=0.0)
    undefined = any(row['discrepancy_l2'] == 'undefined' for row in parameters)
    total.update(finite_discrepancy_l2=_number(discrepancy),
                 discrepancy_l2='undefined' if undefined else (
                     'Infinity' if total['nonfinite_difference_coordinates'] else _number(discrepancy)),
                 finite_maximum_absolute_error=_number(maximum),
                 maximum_absolute_error='undefined' if undefined else (
                     'Infinity' if total['nonfinite_difference_coordinates'] else _number(maximum)))
    total['relative_l2'] = _relative(total['discrepancy_l2'], total['right_l2'])
    total['samples'] = [dict(parameter=row['name'], **sample)
                        for row in parameters for sample in row['samples']][:SAMPLE_LIMIT]
    return dict(exact_equal=all(row['exact_equal'] for row in parameters),
                comparison='exact finite coordinates, shapes, dtypes and parameter membership',
                relative_l2_reference='right', chunk_coordinates=CHUNK_COORDINATES,
                sample_limit=SAMPLE_LIMIT, parameters=parameters, **{'global': total})
