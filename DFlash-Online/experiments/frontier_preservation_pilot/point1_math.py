"""Small derivative and streamed arithmetic helpers for the Point 1 diagnostic.

These functions do not change training. The JVP uses the existing shadow's
represented arithmetic; reported contractions and norms use FP64 products and
reductions. Large vectors are consumed one parameter, then one chunk, at a time.
"""

import math
import re

import torch

from .math import require


class DifferenceVectors:
    """Repeatable parameter-vector view; only the yielded difference is allocated."""

    def __init__(self, candidate, reference):
        require(len(candidate) == len(reference), 'Difference vector lengths differ')
        self.candidate = candidate
        self.reference = reference

    def __len__(self):
        return len(self.candidate)

    def __iter__(self):
        for candidate, reference in zip(self.candidate, self.reference, strict=True):
            require(candidate.shape == reference.shape, 'Difference parameter shapes differ')
            yield fp32_difference(candidate, reference)


class _OptimizerJVP:
    """Repeatable JVP view with the coupled clipping contraction computed once."""

    @torch.no_grad()
    def __init__(self, context, tangents):
        gradients = context['gradients']
        diagonals = context['adam_diagonal']
        require(len(tangents) == len(gradients) == len(diagonals), 'Optimizer tangent list differs')
        self.context, self.tangents = context, tangents
        contractions = []
        for tangent, gradient, diagonal in zip(tangents, gradients, diagonals, strict=True):
            require(tangent.shape == gradient.shape == diagonal.shape, 'Optimizer tangent shape differs')
            direction = tangent.detach().to(diagonal.dtype)
            require(torch.isfinite(direction).all(), 'Nonfinite optimizer tangent')
            if context['clipping_active']:
                contractions.append((gradient * direction).sum())
        self.radial = None
        if context['clipping_active']:
            norm, clip = context['norm'], context['clip']
            contraction = torch.stack(contractions).sum()
            self.radial = clip * contraction / (norm * (norm + context['clip_epsilon']))

    def __len__(self):
        return len(self.tangents)

    def __iter__(self):
        context = self.context
        for tangent, gradient, diagonal in zip(
            self.tangents, context['gradients'], context['adam_diagonal'], strict=True
        ):
            with torch.no_grad():
                direction = tangent.detach().to(diagonal.dtype)
                if context['clipping_active']:
                    direction = context['clip'] * direction - self.radial * gradient
                result = diagonal * direction
                require(torch.isfinite(result).all(), 'Nonfinite optimizer JVP')
            yield result


def optimizer_jvp_view(context, tangents):
    """Push a gradient difference through global clipping, then Adam.

    The existing VJP applies these operations in the opposite order. Reusing it
    here would be wrong whenever clipping couples coordinates with unequal Adam
    diagonals. The boundary branch is inherited from ``adamw_shadow``.
    """
    return _OptimizerJVP(context, tangents)


def optimizer_jvp(context, tangents):
    """Materialized convenience for small tests; full-model callers use the view."""
    return list(optimizer_jvp_view(context, tangents))


def fp32_difference(candidate, reference):
    """Subtract represented operands after promotion, never in BF16."""
    return candidate.detach().float() - reference.detach().float()


def _layer_name(parameter_name):
    match = re.search(r'(?:^|\.)layers\.(\d+)(?:\.|$)', parameter_name)
    return 'layers.' + match.group(1) if match else parameter_name.rsplit('.', 1)[0]


def _zero_status(reference_squared, candidate_squared):
    if reference_squared == candidate_squared == 0:
        return 'both_zero'
    if reference_squared == 0:
        return 'reference_zero'
    if candidate_squared == 0:
        return 'candidate_zero'
    return 'both_nonzero'


def _finish_vector_totals(totals, reference_name, candidate_name, has_cotangent):
    reference_norm = math.sqrt(totals['reference_squared'])
    candidate_norm = math.sqrt(totals['candidate_squared'])
    difference_norm = math.sqrt(totals['difference_squared'])
    result = dict(
        reference=reference_name,
        candidate=candidate_name,
        reference_norm=reference_norm,
        candidate_norm=candidate_norm,
        difference_norm=difference_norm,
        dot=totals['dot'],
        cosine=(totals['dot'] / (reference_norm * candidate_norm)
                if reference_norm > 0 and candidate_norm > 0 else None),
        relative_difference=(difference_norm / reference_norm if reference_norm > 0 else None),
        relative_difference_denominator=f'norm({reference_name})',
        zero_status=_zero_status(totals['reference_squared'], totals['candidate_squared']),
        differing_coordinates=totals['differing_coordinates'],
        coordinates=totals['coordinates'],
    )
    if has_cotangent:
        result.update(
            reference_signed_projection=totals['reference_signed_projection'],
            candidate_signed_projection=totals['candidate_signed_projection'],
            difference_signed_projection=totals['difference_signed_projection'],
            projection_definition='-dot_FP64(cotangent, vector)',
        )
    return result


@torch.no_grad()
def vector_metrics(reference, candidate, *, reference_name, candidate_name,
                   cotangent=None, parameter_names=None, chunk_elements=1_048_576):
    """Compare already formed vectors without materializing full FP64 copies.

    ``reference`` and ``candidate`` may be generators of parameter tensors. Form
    endpoint/gradient differences with :func:`fp32_difference` before supplying
    them. FP64 fixture vectors retain their precision. Per-layer entries expose
    whether a global mismatch is concentrated in particular parameters.
    """
    require(chunk_elements > 0, 'Vector chunk size must be positive')
    reference = iter(reference)
    candidate = iter(candidate)
    names = iter(parameter_names) if parameter_names is not None else None
    cotangents = iter(cotangent) if cotangent is not None else None
    keys = ('reference_squared', 'candidate_squared', 'difference_squared', 'dot',
            'reference_signed_projection', 'candidate_signed_projection',
            'difference_signed_projection')
    terms = {key: [] for key in keys}
    totals = dict(coordinates=0, differing_coordinates=0)
    layers = {}
    parameter_count = 0
    for parameter_count, (left, right) in enumerate(zip(reference, candidate, strict=True), 1):
        require(left.shape == right.shape, 'Vector parameter shapes differ')
        name = next(names) if names is not None else f'parameter_{parameter_count - 1}'
        layer = layers.setdefault(_layer_name(name), {key: [] for key in keys})
        layer.setdefault('coordinates', 0)
        layer.setdefault('differing_coordinates', 0)
        incoming = next(cotangents) if cotangents is not None else None
        if incoming is not None:
            require(incoming.shape == left.shape, 'Vector cotangent shape differs')
            incoming = incoming.detach().reshape(-1)
        left, right = left.detach().reshape(-1), right.detach().reshape(-1)
        for start in range(0, left.numel(), chunk_elements):
            part = slice(start, start + chunk_elements)
            a, b = left[part].double(), right[part].double()
            require(torch.isfinite(a).all() and torch.isfinite(b).all(), 'Nonfinite diagnostic vector')
            error = b - a
            values = dict(
                reference_squared=float(a.square().sum()),
                candidate_squared=float(b.square().sum()),
                difference_squared=float(error.square().sum()),
                dot=float((a * b).sum()),
            )
            if incoming is not None:
                r = incoming[part].double()
                require(torch.isfinite(r).all(), 'Nonfinite diagnostic cotangent')
                values.update(
                    reference_signed_projection=float(-(r * a).sum()),
                    candidate_signed_projection=float(-(r * b).sum()),
                    difference_signed_projection=float(-(r * error).sum()),
                )
            for key, value in values.items():
                terms[key].append(value)
                layer[key].append(value)
            count, different = a.numel(), int((a != b).sum())
            totals['coordinates'] += count
            totals['differing_coordinates'] += different
            layer['coordinates'] += count
            layer['differing_coordinates'] += different
    require(parameter_count > 0, 'Diagnostic vector is empty')
    if names is not None:
        require(next(names, None) is None, 'Extra diagnostic parameter names')
    if cotangents is not None:
        require(next(cotangents, None) is None, 'Extra diagnostic cotangents')
    totals.update({key: math.fsum(values) for key, values in terms.items()})
    result = _finish_vector_totals(totals, reference_name, candidate_name, cotangent is not None)
    result['per_layer'] = {}
    for name, values in layers.items():
        values = {key: math.fsum(value) if isinstance(value, list) else value
                  for key, value in values.items()}
        result['per_layer'][name] = _finish_vector_totals(
            values, reference_name, candidate_name, cotangent is not None
        )
    result['products_and_reductions'] = 'FP64'
    return result


def _signed_sum(values):
    return dict(
        total=float(values.sum()),
        positive=float(values.clamp_min(0).sum()),
        negative=float(values.clamp_max(0).sum()),
        absolute=float(values.abs().sum()),
    )


@torch.no_grad()
def score_contrast(candidate_coefficients, reference_coefficients, scores, beta, prefix):
    """Report the signed score contrast and cancellation for one response.

    Center scores separately within every frozen block, with beta weights and a
    stable offset. Raw versus centered differences expose the finite coefficient
    mass residual; no favorable sign is required for the F-minus-C comparison.
    """
    require(candidate_coefficients.shape == reference_coefficients.shape == scores.shape
            == beta.shape == prefix.shape, 'Score contrast shapes differ')
    require(prefix.dtype == torch.bool, 'Prefix support must be boolean')
    values = (candidate_coefficients, reference_coefficients, scores, beta)
    require(all(torch.isfinite(value).all() for value in values), 'Nonfinite score contrast')
    difference = fp32_difference(candidate_coefficients, reference_coefficients).double()
    score, base = scores.detach().double(), beta.detach().double()
    mass = (base * prefix).sum(-1, keepdim=True)
    lowest = torch.where(prefix, score, torch.inf).amin(-1, keepdim=True)
    origin = torch.where(prefix.any(-1, keepdim=True), lowest, 0.0)
    offsets = torch.where(prefix, score - origin, 0.0)
    mean_offset = (base * offsets).sum(-1, keepdim=True) / torch.where(mass > 0, mass, 1.0)
    centered = torch.where(prefix, offsets - mean_offset, 0.0)
    mean = origin + mean_offset
    difference = torch.where(prefix, difference, 0.0)
    raw = difference * score
    centered_terms = difference * centered
    mass_residual = difference.sum(-1, keepdim=True)
    offset_contribution = mass_residual * mean
    return dict(
        raw=_signed_sum(raw),
        beta_centered=_signed_sum(centered_terms),
        raw_minus_centered=float(raw.sum() - centered_terms.sum()),
        mass_offset_contribution=float(offset_contribution.sum()),
        max_absolute_pool_mass_residual=float(mass_residual.abs().max()),
        prefix_occurrences=int(prefix.sum()),
        nonempty_pools=int(prefix.any(-1).sum()),
        products_and_reductions='FP64 after FP32 coefficient subtraction',
        per_pool=dict(
            raw=raw.sum(-1).cpu().tolist(),
            beta_centered=centered_terms.sum(-1).cpu().tolist(),
            mass_residual=mass_residual.squeeze(-1).cpu().tolist(),
            weighted_mean_score=mean.squeeze(-1).cpu().tolist(),
        ),
    )


@torch.no_grad()
def matched_assessment_contrast(reference_contributions, candidate_contributions):
    """Subtract matched represented contributions before their FP64 reduction.

    Inputs already include the authored position weights, block denominator and
    response factor. This readout never replaces the scoring assessment's FP32
    arithmetic. Positive benefit means the candidate has lower assessment.
    """
    require(reference_contributions.shape == candidate_contributions.shape,
            'Assessment contribution identities or shapes differ')
    reference = reference_contributions.detach().double()
    candidate = candidate_contributions.detach().double()
    require(torch.isfinite(reference).all() and torch.isfinite(candidate).all(),
            'Nonfinite assessment contribution')
    return dict(reference_sum=float(reference.sum()), candidate_sum=float(candidate.sum()),
                benefit=_signed_sum(reference - candidate), occurrences=reference.numel(),
                products_and_reductions='FP64 on matched represented contributions')


def prediction_diagnostics(prediction, finite_benefit, *, prediction_name):
    """Describe sign and prediction error without inventing a fidelity gate."""
    prediction, finite_benefit = float(prediction), float(finite_benefit)
    require(math.isfinite(prediction) and math.isfinite(finite_benefit),
            'Nonfinite functional prediction')

    def sign(value):
        return 'positive' if value > 0 else 'negative' if value < 0 else 'zero'

    error = prediction - finite_benefit
    return dict(
        prediction_name=prediction_name,
        prediction=prediction,
        finite_benefit=finite_benefit,
        prediction_sign=sign(prediction),
        finite_benefit_sign=sign(finite_benefit),
        signs_match=sign(prediction) == sign(finite_benefit),
        signed_prediction_error=error,
        absolute_prediction_error=abs(error),
        relative_absolute_error=(abs(error) / abs(finite_benefit) if finite_benefit else None),
        relative_error_denominator='abs(finite_native_assessment_benefit)',
        zero_benefit=finite_benefit == 0,
        admission_gate=False,
    )


def adjoint_reconciliation(gradient_projection, linear_projection):
    """Expose the JVP/VJP projection identity error and its numerical scale.

    Both inputs have already used the required FP64 diagnostic contraction.
    The JVP/VJP themselves retain the shadow's existing FP32 arithmetic. The
    independent tiny FP64 tests qualify the formula; this full-model receipt
    does not create a post-hoc tolerance or conflate prediction with efficacy.
    """
    gradient_projection, linear_projection = float(gradient_projection), float(linear_projection)
    require(math.isfinite(gradient_projection) and math.isfinite(linear_projection),
            'Nonfinite adjoint projection')
    error = linear_projection - gradient_projection
    return dict(
        gradient_projection=gradient_projection,
        linear_projection=linear_projection,
        difference=error,
        absolute_difference=abs(error),
        relative_difference=(abs(error) / abs(gradient_projection) if gradient_projection else None),
        relative_difference_denominator='abs(-dot_FP64(w, delta_g))',
        both_zero=gradient_projection == linear_projection == 0,
        zero_reference=gradient_projection == 0,
        admission_gate=False,
    )
