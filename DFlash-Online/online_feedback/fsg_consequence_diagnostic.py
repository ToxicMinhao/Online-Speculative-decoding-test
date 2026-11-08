"""Read-only consequence metrics for two FSG proposals on one retained window.

These reducers do not decide acceptance. The existing geometry gate owns that
rule; raw and committed comparisons remain separate even when both fall back.
"""
import math

import torch

from . import functional_geometry_math as geometry
from .first_error import valid_prefix_mask
from .prefix_correction import repair_transitions
from .prefix_guard import prefix_changes


def _regions(theta0, theta_native, chunk_size):
    if chunk_size <= 0 or len(theta0) != len(theta_native):
        raise ValueError('Consequence comparison requires aligned states and positive chunk size')
    for index, (old, native) in enumerate(zip(theta0, theta_native)):
        if old.shape != native.shape:
            raise ValueError('Consequence state shapes differ')
        old, native = old.detach().reshape(-1), native.detach().reshape(-1)
        for start in range(0, old.numel(), chunk_size):
            region = slice(start, min(start+chunk_size, old.numel()))
            yield index, region, old[region].float(), native[region].to(old.device).float()


def _proposal(sketch, index, region, old, native):
    # Match write_candidate's FP32 operation order, before its final BF16 cast.
    delta = native-old
    if sketch.strength != 0 and sketch.sigma != 0:
        for gradients, normalizer, coefficient in zip(
                sketch.gradients, sketch.normalizers, sketch.coefficients):
            gradient = gradients[index]
            if gradient is not None:
                piece = gradient.detach().reshape(-1)[region].to(old.device).float()
                delta = delta-piece*float(normalizer*coefficient)
    return delta


def _ratio(value, denominator):
    return value/denominator if denominator else None


def _summarize(chunks):
    sums = counts = maximum = None
    coordinates = 0
    for native, left, right, endpoints in chunks:
        if sums is None:
            sums = native.new_zeros(12, dtype=torch.float64)
            counts = native.new_zeros(8, dtype=torch.int64)
            maximum = native.new_zeros(())
        u, a, b = native.double(), left.double(), right.double()
        difference = a-b
        left_native, right_native = a-u, b-u
        old_values = torch.zeros_like(native)
        native_values, left_values, right_values = native, left, right
        if endpoints is not None:
            old_values, native_values, left_values, right_values = endpoints
            difference = left_values.double()-right_values.double()
            left_native = left_values.double()-native_values.double()
            right_native = right_values.double()-native_values.double()
        zero = native_values == old_values
        sums += torch.stack((u.square().sum(), a.square().sum(), b.square().sum(),
            difference.square().sum(), (a*b).sum(), left_native.square().sum(),
            right_native.square().sum(), (a*u).sum(), (b*u).sum(),
            difference[zero].square().sum(), a[zero].square().sum(), b[zero].square().sum()))
        counts += torch.stack(((left_values != right_values).sum(),
            (left_values != old_values).sum(), (right_values != old_values).sum(),
            (zero & (left_values != old_values)).sum(), (zero & (right_values != old_values)).sum(),
            (zero & (left_values != right_values)).sum(),
            (left_values != native_values).sum(), (right_values != native_values).sum()))
        maximum = torch.maximum(maximum, difference.abs().max())
        coordinates += native.numel()
    if sums is None:
        values, totals, max_difference = [0.]*12, [0]*8, 0.
    else:
        geometry.finite(sums, 'consequence vector reductions')
        geometry.finite(maximum, 'consequence maximum difference')
        values, totals, max_difference = sums.tolist(), counts.tolist(), float(maximum)
    native_norm, left_norm, right_norm, difference_norm = map(math.sqrt, values[:4])
    return dict(coordinates=coordinates, native_norm=native_norm,
        left_norm=left_norm, right_norm=right_norm, difference_norm=difference_norm,
        relative_difference_to_native_norm=_ratio(difference_norm, native_norm),
        direction_cosine=_ratio(values[4], left_norm*right_norm),
        left_deviation_from_native_norm=math.sqrt(values[5]),
        right_deviation_from_native_norm=math.sqrt(values[6]),
        left_deviation_relative_to_native=_ratio(math.sqrt(values[5]), native_norm),
        right_deviation_relative_to_native=_ratio(math.sqrt(values[6]), native_norm),
        left_native_cosine=_ratio(values[7], left_norm*native_norm),
        right_native_cosine=_ratio(values[8], right_norm*native_norm),
        difference_on_zero_native_norm=math.sqrt(values[9]),
        left_on_zero_native_norm=math.sqrt(values[10]), right_on_zero_native_norm=math.sqrt(values[11]),
        differing_coordinates=totals[0], identical=totals[0] == 0,
        left_changed_coordinates=totals[1], right_changed_coordinates=totals[2],
        left_changed_zero_native_coordinates=totals[3], right_changed_zero_native_coordinates=totals[4],
        differing_zero_native_coordinates=totals[5],
        left_identical_native=totals[6] == 0, right_identical_native=totals[7] == 0,
        maximum_absolute_difference=max_difference)


@torch.no_grad()
def compare_sketches(left, right, theta0, theta_native, *, chunk_size=geometry.VECTOR_CHUNK_SIZE):
    """Compare continuous FP32 proposals with streamed FP64 scalar reductions."""
    def chunks():
        for index, region, old, native in _regions(theta0, theta_native, chunk_size):
            yield (native-old, _proposal(left, index, region, old, native),
                   _proposal(right, index, region, old, native), None)
    return dict(_summarize(chunks()), quantity='continuous_fp32_proposals')


@torch.no_grad()
def compare_endpoints(left, right, theta0, theta_native, *, chunk_size=geometry.VECTOR_CHUNK_SIZE):
    """Compare existing rounded states; CPU snapshots are moved one slice at a time."""
    if not (len(left) == len(right) == len(theta0)):
        raise ValueError('Consequence endpoints must align with the origin')
    for a, b, old in zip(left, right, theta0):
        if a.shape != old.shape or b.shape != old.shape:
            raise ValueError('Consequence endpoint shapes differ')
    def chunks():
        for index, region, old, native in _regions(theta0, theta_native, chunk_size):
            a = left[index].detach().reshape(-1)[region].to(old.device).float()
            b = right[index].detach().reshape(-1)[region].to(old.device).float()
            yield native-old, a-old, b-old, (old, native, a, b)
    return dict(_summarize(chunks()), quantity='realized_endpoint_displacements')


def decision_summary(left_gate, right_gate, raw, committed):
    """Describe unchanged gate outputs without hiding distinct rejected proposals."""
    both_fallback = not left_gate['accepted'] and not right_gate['accepted']
    return dict(left=dict(left_gate), right=dict(right_gate),
        acceptance_disagrees=left_gate['accepted'] != right_gate['accepted'],
        reason_disagrees=left_gate['reason'] != right_gate['reason'], both_fallback=both_fallback,
        raw_endpoints_identical=raw['identical'], committed_endpoints_identical=committed['identical'],
        both_fallback_with_distinct_raw_endpoints=both_fallback and not raw['identical'])


@torch.no_grad()
def prefix_consequences(before, native, left, right, labels, validity_weights, eos_token_ids=()):
    """Per-response native proposal metrics with the established EOS/seed rules.

    Pass predictions from the common production-chunk scorer. These training-
    conditioned blocks do not establish complete-generation or held-out benefit.
    """
    valid = valid_prefix_mask(labels, validity_weights, eos_token_ids)
    arguments = (labels, validity_weights, eos_token_ids)
    def repairs(candidate):
        rows = repair_transitions(before, native, candidate, *arguments)
        return {('native_useful_repairs' if key == 'K_useful_repairs' else
                 'candidate_useful_repairs' if key == 'C_useful_repairs' else key): value
                for key, value in rows.items()}
    return dict(blocks=labels.numel()//labels.shape[-1], valid_proposal_occurrences=int(valid.sum()),
        differing_valid_proposals=int(((left != right) & valid).sum()),
        native_from_pre=prefix_changes(before, native, *arguments),
        left_from_pre=prefix_changes(before, left, *arguments),
        right_from_pre=prefix_changes(before, right, *arguments),
        right_vs_left=prefix_changes(left, right, *arguments),
        left_repair_identities_vs_native=repairs(left),
        right_repair_identities_vs_native=repairs(right))
