"""Numerical reducers for the sealed K+ functional-step-geometry experiment.

Model-sized quantities stay in their native dtype. Only a parameter chunk is
promoted for vector algebra; small Gram matrices and scalar sums use FP64.
Geometry cotangents use FP32, while the actual risk check recomputes both
distributions with FP64 log-softmax. These are deliberately distinct paths.
"""
from dataclasses import dataclass
import math

import torch

from .first_error import valid_prefix_mask


MARGIN_FLOOR = 0.05
STRENGTH = 1.0
SCALAR_ITERATIONS = 24
VECTOR_CHUNK_SIZE = 262144
GRAM_ROUNDOFF_MULTIPLIER = 64.0
RISK_ROUNDOFF_MULTIPLIER = 16.0
EXPANDED_RELATIVE_CUTOFF = 1e-6


def finite(value, name):
    """Nonfinite evidence is a failure, never a bypass or a repaired number."""
    if isinstance(value, torch.Tensor):
        valid = bool(torch.isfinite(value).all())
    else:
        valid = math.isfinite(float(value))
    if not valid:
        raise ValueError('Nonfinite functional geometry ' + name)


@torch.no_grad()
def prefix_statistics(proposals, labels, validity_weights, canonical_prob,
                      rival_prob, eos_token_ids=()):
    """Return unnormalized per-response weights from complete independent blocks.

    Chunks may contain different blocks. Their denominators must be summed
    before ``normalize_weights`` is called; a chunk mean is not a response mean.
    Argmax membership comes from native logits, not recomputed probabilities.
    """
    finite(canonical_prob, 'canonical probability')
    finite(rival_prob, 'rival probability')
    valid = valid_prefix_mask(labels, validity_weights, eos_token_ids)
    mismatch = valid & (proposals.detach() != labels)
    mask = valid & (mismatch.long().cumsum(-1) == 0)
    lengths = mask.sum(-1, keepdim=True)
    positions = torch.arange(labels.shape[-1], device=labels.device)
    triangular = torch.where(mask, lengths - positions + 1, 0)
    margin = canonical_prob.detach().double() - rival_prob.detach().double()
    raw = triangular.double() / margin.clamp_min(MARGIN_FLOOR).square()
    return dict(mask=mask, triangular=triangular, margin=margin,
                raw_weights=raw, denominator=raw.sum(dtype=torch.float64))


@torch.no_grad()
def normalize_weights(raw_weights, denominator):
    finite(denominator, 'weight denominator')
    if float(denominator) < 0:
        raise ValueError('Negative functional geometry weight denominator')
    if float(denominator) == 0:
        return torch.zeros_like(raw_weights, dtype=torch.float32)
    return (raw_weights.double() / denominator).float().detach()


@torch.no_grad()
def geometry_terms(z0, z_native, weights):
    """Detached FP32 cotangent and centered-squared FP64 scalar reduction.

    No two-response factor enters the cotangent: the factor one half is supplied
    by each conceptual column's ``sqrt(2*c)`` normalizer.
    """
    finite(z0, 'pre-state logits')
    finite(z_native, 'native logits')
    q0 = z0.detach().float().softmax(-1)
    displacement = z_native.detach().float() - z0.detach().float()
    center = displacement - (q0 * displacement).sum(-1, keepdim=True)
    weighted_prob = weights.detach().float().unsqueeze(-1) * q0
    cotangent = (weighted_prob * center).detach()
    c = (weighted_prob.double() * center.double().square()).sum(dtype=torch.float64)
    finite(c, 'centered output energy')
    finite(cotangent, 'output cotangent')
    return cotangent, c


@torch.no_grad()
def risk_terms(z0, z_candidate, weights):
    """Unaveraged risk/absolute-sum terms for one response or projection chunk.

    Both distributions are recomputed from native logits in FP64. Reusing FP32
    geometry probabilities would be inconsistent with the declared deadband.
    """
    finite(z0, 'risk reference logits')
    finite(z_candidate, 'risk candidate logits')
    log0 = z0.detach().double().log_softmax(-1)
    log_candidate = z_candidate.detach().double().log_softmax(-1)
    weighted_prob = weights.detach().double().unsqueeze(-1) * log0.exp()
    risk = (weighted_prob * (log0 - log_candidate)).sum(dtype=torch.float64)
    absolute_sum = (weighted_prob * (log0.abs() + log_candidate.abs())).sum(dtype=torch.float64)
    finite(risk, 'risk reduction')
    finite(absolute_sum, 'risk absolute reduction')
    return risk, absolute_sum


@dataclass(frozen=True)
class Risk:
    value: float
    raw: float
    eta: float
    absolute_sum: float

    def as_dict(self):
        return dict(value=self.value, raw=self.raw, eta=self.eta,
                    absolute_sum=self.absolute_sum, negative_roundoff=self.raw < 0)


def pooled_risk(parts, vocab_size):
    """Pool additive chunk/response terms with the fixed two-response factor."""
    risk = 0.5 * math.fsum(float(part[0]) for part in parts)
    absolute_sum = 0.5 * math.fsum(float(part[1]) for part in parts)
    finite(risk, 'pooled risk')
    finite(absolute_sum, 'pooled absolute sum')
    eps = torch.finfo(torch.float64).eps
    gamma = (vocab_size + 4) * eps / (1 - (vocab_size + 4) * eps)
    eta = RISK_ROUNDOFF_MULTIPLIER * gamma * (1 + absolute_sum)
    if risk < -eta:
        raise ValueError('Negative functional geometry KL exceeds declared deadband')
    return Risk(max(0., risk), risk, eta, absolute_sum)


def gate(candidate_risk, native_risk, candidate_norm, native_norm, *, identical=False):
    """Decide only the stated finite/norm/actual-risk checks, without retuning."""
    for name, value in (('candidate norm', candidate_norm), ('native norm', native_norm),
                        ('candidate risk', candidate_risk.value), ('native risk', native_risk.value),
                        ('candidate deadband', candidate_risk.eta), ('native deadband', native_risk.eta)):
        finite(value, name)
    if candidate_norm > native_norm:
        return dict(accepted=False, reason='rounded_norm_exceeds_native')
    if identical:
        return dict(accepted=True, reason='identical_native_endpoint')
    if candidate_risk.value + candidate_risk.eta <= native_risk.value - native_risk.eta:
        return dict(accepted=True, reason='actual_risk_decreased')
    overlap = (candidate_risk.value - candidate_risk.eta <= native_risk.value + native_risk.eta
               and native_risk.value - native_risk.eta <= candidate_risk.value + candidate_risk.eta)
    return dict(accepted=False, reason='numerical_ambiguity' if overlap else 'actual_risk_increased')


def _slices(size, chunk_size):
    for start in range(0, size, chunk_size):
        yield slice(start, min(start + chunk_size, size))


@torch.no_grad()
def vector_dot(left, right, chunk_size=VECTOR_CHUNK_SIZE):
    """FP64 dot product without materializing a promoted parameter vector."""
    total = None
    for a, b in zip(left, right):
        if a is None or b is None:
            continue
        if a.shape != b.shape:
            raise ValueError('Functional geometry vector shape mismatch')
        a, b = a.detach().reshape(-1), b.detach().reshape(-1)
        if total is None:
            total = torch.zeros((), dtype=torch.float64, device=a.device)
        for region in _slices(a.numel(), chunk_size):
            total += (a[region].double() * b[region].double()).sum()
    result = 0. if total is None else float(total)
    finite(result, 'parameter dot product')
    return result


@torch.no_grad()
def displacement_dot(gradient, theta0, theta_native, chunk_size=VECTOR_CHUNK_SIZE):
    total = None
    for g, old, new in zip(gradient, theta0, theta_native):
        if g is None:
            continue
        old, new, g = old.reshape(-1), new.reshape(-1), g.reshape(-1)
        if total is None:
            total = torch.zeros((), dtype=torch.float64, device=old.device)
        for region in _slices(old.numel(), chunk_size):
            u = new[region].float() - old[region].float()
            total += (g[region].double() * u.double()).sum()
    result = 0. if total is None else float(total)
    finite(result, 'gradient/native displacement dot product')
    return result


@dataclass
class Sketch:
    gradients: tuple
    normalizers: tuple
    gram: torch.Tensor
    coefficients: torch.Tensor
    sigma: float
    strength: float
    statistics: dict


def _solve_gram(gram, rhs, strength, sigma=None):
    """Small CPU FP64 symmetric solve; arithmetic allowance is sealed, not tuned."""
    gram = (gram.double().cpu() + gram.double().cpu().T) * 0.5
    finite(gram, 'Gram matrix')
    eigenvalues = torch.linalg.eigvalsh(gram)
    scale = float(gram.abs().max()) if gram.numel() else 0.
    allowance = GRAM_ROUNDOFF_MULTIPLIER * torch.finfo(torch.float64).eps * max(1, len(gram)) * scale
    if len(eigenvalues) and float(eigenvalues[0]) < -allowance:
        raise ValueError('Invalid negative functional geometry Gram eigenvalue')
    observed_sigma = max(0., float(eigenvalues[-1])) if len(eigenvalues) else 0.
    sigma = observed_sigma if sigma is None else float(sigma)
    finite(sigma, 'curvature scale')
    if sigma < 0:
        raise ValueError('Negative functional geometry curvature scale')
    if sigma == 0 or strength == 0:
        return torch.zeros_like(rhs), sigma, dict(eigenvalues=eigenvalues.tolist(),
            gram_eigenvalue_roundoff_allowance=allowance, solve_residual=0., condition_number=1.)
    matrix = torch.eye(len(gram), dtype=torch.float64) + strength / sigma * gram
    target = strength / sigma * rhs
    coefficients = torch.linalg.solve(matrix, target)
    finite(coefficients, 'small solve')
    return coefficients, sigma, dict(eigenvalues=eigenvalues.tolist(),
        gram_eigenvalue_roundoff_allowance=allowance,
        solve_residual=float(torch.linalg.vector_norm(matrix @ coefficients - target)),
        condition_number=float(torch.linalg.cond(matrix)))


@torch.no_grad()
def make_sketch(gradients, c_values, theta0, theta_native, *, strength=STRENGTH,
                chunk_size=VECTOR_CHUNK_SIZE):
    """Build the at-most-two-column production sketch from native VJP tensors."""
    if len(gradients) != len(c_values) or len(gradients) > 2:
        raise ValueError('Production sketch requires one column per response, at most two')
    if strength not in (0., STRENGTH):
        raise ValueError('Functional geometry strength is sealed to zero or one')
    usable, normalizers, rows = [], [], []
    for index, (gradient, c) in enumerate(zip(gradients, c_values)):
        c = float(c)
        finite(c, 'response output energy')
        if c < 0:
            raise ValueError('Negative centered output energy')
        norm_sq = vector_dot(gradient, gradient, chunk_size)
        dot_u = displacement_dot(gradient, theta0, theta_native, chunk_size)
        row = dict(response=index, c=c, gradient_norm=math.sqrt(norm_sq),
                   gradient_dot_native=dot_u, gradient_dot_native_over_c=dot_u/c if c else None)
        if c == 0 or norm_sq == 0:
            row['bypass'] = 'zero_output_energy' if c == 0 else 'zero_gradient'
        else:
            usable.append(tuple(gradient))
            normalizers.append(1 / math.sqrt(2*c))
        rows.append(row)
    count = len(usable)
    gram = torch.zeros((count, count), dtype=torch.float64)
    rhs = torch.zeros(count, dtype=torch.float64)
    for i in range(count):
        rhs[i] = displacement_dot(usable[i], theta0, theta_native, chunk_size) * normalizers[i]
        for j in range(i + 1):
            gram[i, j] = gram[j, i] = vector_dot(usable[i], usable[j], chunk_size) * normalizers[i] * normalizers[j]
    if count:
        coefficients, sigma, solved = _solve_gram(gram, rhs, strength)
    else:
        coefficients, sigma, solved = rhs, 0., dict(eigenvalues=[], solve_residual=0., condition_number=1.,
            gram_eigenvalue_roundoff_allowance=0.)
    statistics = dict(responses=rows, usable_columns=count, gram=gram.tolist(),
        rhs=rhs.tolist(), sigma=sigma, coefficients=coefficients.tolist(), strength=strength,
        native_projected_energy=float(rhs @ rhs), **solved)
    if count == 2:
        denom = math.sqrt(float(gram[0, 0] * gram[1, 1]))
        statistics['cross_response_cosine'] = float(gram[0, 1]) / denom if denom else None
    return Sketch(tuple(usable), tuple(normalizers), gram, coefficients, sigma, strength, statistics)


@torch.no_grad()
def expanded_terms(z0, z_native, z_geometry, weights):
    """Two detached output directions with their joint weighted covariance."""
    finite(z0, 'expanded reference logits')
    finite(z_native, 'expanded native logits')
    finite(z_geometry, 'expanded geometry logits')
    q0 = z0.detach().float().softmax(-1)
    vectors = torch.stack((z_native.detach().float()-z0.detach().float(),
                           z_geometry.detach().float()-z0.detach().float()), dim=-1)
    centered = vectors - (q0.unsqueeze(-1)*vectors).sum(-2, keepdim=True)
    probability = weights.detach().float().unsqueeze(-1)*q0
    cotangents = (probability.unsqueeze(-1)*centered).detach()
    covariance = torch.einsum('...v,...vi,...vj->ij', probability.double(),
                              centered.double(), centered.double())
    finite(covariance, 'expanded output covariance')
    return (cotangents[..., 0], cotangents[..., 1]), covariance


@torch.no_grad()
def make_expanded_sketch(response_gradients, output_covariances, theta0, theta_native,
                         *, production_sigma, chunk_size=VECTOR_CHUNK_SIZE):
    """Whiten joint output directions without materializing new parameter vectors.

    The returned coefficients operate directly on the original native gradients.
    The solve uses the production sigma, not the expanded largest eigenvalue.
    """
    raw_gradients, transforms, discarded = [], [], []
    for gradients, covariance in zip(response_gradients, output_covariances):
        covariance = covariance.double().cpu()
        finite(covariance, 'expanded covariance')
        covariance = (covariance+covariance.T)*.5
        eigenvalues, vectors = torch.linalg.eigh(covariance)
        scale = float(covariance.abs().max())
        # Cotangents are FP32; this numerical cutoff is specified for this
        # diagnostic only. It does not alter the production two-column sketch.
        allowance = GRAM_ROUNDOFF_MULTIPLIER*torch.finfo(torch.float64).eps*len(covariance)*scale
        if float(eigenvalues[0]) < -allowance:
            raise ValueError('Invalid negative expanded covariance eigenvalue')
        largest = max(0., float(eigenvalues[-1]))
        keep = eigenvalues > EXPANDED_RELATIVE_CUTOFF*largest
        transform = vectors[:, keep] / eigenvalues[keep].sqrt().unsqueeze(0) / math.sqrt(2.)
        transforms.append(transform)
        raw_gradients.extend(tuple(g) for g in gradients)
        discarded.append(dict(eigenvalues=eigenvalues.tolist(), retained=int(keep.sum()),
                              cutoff=EXPANDED_RELATIVE_CUTOFF*largest))
    count = len(raw_gradients)
    transform = torch.block_diag(*transforms) if transforms else torch.zeros((0, 0), dtype=torch.float64)
    raw_gram = torch.zeros((count, count), dtype=torch.float64)
    raw_rhs = torch.zeros(count, dtype=torch.float64)
    for i in range(count):
        raw_rhs[i] = displacement_dot(raw_gradients[i], theta0, theta_native, chunk_size)
        for j in range(i+1):
            raw_gram[i, j] = raw_gram[j, i] = vector_dot(raw_gradients[i], raw_gradients[j], chunk_size)
    gram = transform.T @ raw_gram @ transform
    rhs = transform.T @ raw_rhs
    if len(gram):
        coefficients, sigma, solved = _solve_gram(gram, rhs, STRENGTH, production_sigma)
    else:
        coefficients, sigma = rhs, float(production_sigma)
        solved = dict(eigenvalues=[], solve_residual=0., condition_number=1., gram_eigenvalue_roundoff_allowance=0.)
    expanded_max = max(solved['eigenvalues'], default=0.)
    stats = dict(kind='expanded_diagnostic', output_covariances=discarded,
        usable_columns=len(gram), production_sigma=float(production_sigma), sigma=sigma,
        expanded_largest_eigenvalue=expanded_max,
        expanded_relative_strength=expanded_max/sigma if sigma else None,
        gram=gram.tolist(), raw_gram=raw_gram.tolist(), coefficients=(transform@coefficients).tolist(),
        **solved)
    return Sketch(tuple(raw_gradients), tuple(1. for _ in raw_gradients), gram,
                  transform@coefficients, sigma, STRENGTH, stats)


@torch.no_grad()
def endpoint_statistics(parameters, theta0, theta_native, chunk_size=VECTOR_CHUNK_SIZE):
    """Measure the actual rounded endpoint over native-active tensors only."""
    sums = None
    counts = None
    identical = None
    for parameter, old, native in zip(parameters, theta0, theta_native):
        current, old, native = parameter.detach().reshape(-1), old.reshape(-1), native.reshape(-1)
        if sums is None:
            sums = torch.zeros(3, dtype=torch.float64, device=current.device)
            counts = torch.zeros(2, dtype=torch.int64, device=current.device)
            identical = torch.ones((), dtype=torch.bool, device=current.device)
        for region in _slices(current.numel(), chunk_size):
            delta = current[region].float() - old[region].float()
            u = native[region].float() - old[region].float()
            sums[0] += delta.double().square().sum()
            sums[1] += u.double().square().sum()
            sums[2] += (delta.double() * u.double()).sum()
            counts[0] += (delta != 0).sum()
            counts[1] += ((u == 0) & (delta != 0)).sum()
            identical &= (current[region] == native[region]).all()
    values = [0., 0., 0.] if sums is None else sums.tolist()
    for value in values:
        finite(value, 'endpoint vector statistic')
    counts = [0, 0] if counts is None else counts.tolist()
    identical = True if identical is None else bool(identical)
    norm, native_norm = math.sqrt(values[0]), math.sqrt(values[1])
    return dict(norm=norm, native_norm=native_norm, identical_native=identical,
        changed_coordinates=counts[0], changed_zero_native_coordinates=counts[1],
        native_dot=values[2], direction_cosine=values[2]/(norm*native_norm) if norm*native_norm else None)


@torch.no_grad()
def write_candidate(parameters, theta0, theta_native, sketch, chunk_size=VECTOR_CHUNK_SIZE):
    """Write one FP32-then-native-dtype endpoint into existing Parameter objects."""
    if sketch.strength == 0 or sketch.sigma == 0:
        for parameter, native in zip(parameters, theta_native):
            parameter.copy_(native)
        stats = endpoint_statistics(parameters, theta0, theta_native, chunk_size)
        return dict(stats, continuous_norm=stats['native_norm'],
                    bypass='lambda_zero' if sketch.strength == 0 else 'empty_sketch')
    continuous_square = None
    for index, (parameter, old, native) in enumerate(zip(parameters, theta0, theta_native)):
        current, old, native = parameter.reshape(-1), old.reshape(-1), native.reshape(-1)
        if continuous_square is None:
            continuous_square = torch.zeros((), dtype=torch.float64, device=current.device)
        for region in _slices(current.numel(), chunk_size):
            delta = native[region].float() - old[region].float()
            for gradients, normalizer, coefficient in zip(sketch.gradients, sketch.normalizers, sketch.coefficients):
                gradient = gradients[index]
                if gradient is not None:
                    delta = delta - gradient.reshape(-1)[region].float() * float(normalizer * coefficient)
            continuous_square += delta.double().square().sum()
            current[region].copy_((old[region].float() + delta).to(current.dtype))
    finite(continuous_square, 'continuous candidate')
    return dict(endpoint_statistics(parameters, theta0, theta_native, chunk_size),
                continuous_norm=math.sqrt(float(continuous_square)), bypass=None)


@torch.no_grad()
def _scalar_norm(theta0, theta_native, alpha, chunk_size):
    total = None
    for old, native in zip(theta0, theta_native):
        old, native = old.reshape(-1), native.reshape(-1)
        if total is None:
            total = torch.zeros((), dtype=torch.float64, device=old.device)
        for region in _slices(old.numel(), chunk_size):
            if alpha == 0:
                endpoint = old[region]
            elif alpha == 1:
                endpoint = native[region]
            else:
                endpoint = (old[region].float() + alpha*(native[region].float()-old[region].float())).to(old.dtype)
            total += (endpoint.float()-old[region].float()).double().square().sum()
    value = 0. if total is None else math.sqrt(float(total))
    finite(value, 'scalar rounded norm')
    return value


@torch.no_grad()
def scalar_match(parameters, theta0, theta_native, target_norm, *, iterations=SCALAR_ITERATIONS,
                 chunk_size=VECTOR_CHUNK_SIZE):
    """Choose the closer of the final 24-bisection bracket's rounded endpoints."""
    if iterations != SCALAR_ITERATIONS:
        raise ValueError('Scalar bisection count is sealed to 24')
    finite(target_norm, 'scalar target norm')
    native_norm = _scalar_norm(theta0, theta_native, 1., chunk_size)
    if target_norm < 0 or target_norm > native_norm:
        raise ValueError('Scalar target norm must lie within the native norm')
    lo, hi, lo_norm, hi_norm = 0., 1., 0., native_norm
    for _ in range(iterations):
        middle = (lo + hi) * .5
        middle_norm = _scalar_norm(theta0, theta_native, middle, chunk_size)
        if middle_norm < target_norm:
            lo, lo_norm = middle, middle_norm
        else:
            hi, hi_norm = middle, middle_norm
    alpha, norm = (lo, lo_norm) if abs(lo_norm-target_norm) <= abs(hi_norm-target_norm) else (hi, hi_norm)
    for parameter, old, native in zip(parameters, theta0, theta_native):
        if alpha == 0:
            parameter.copy_(old)
        elif alpha == 1:
            parameter.copy_(native)
        else:
            flat, old, native = parameter.reshape(-1), old.reshape(-1), native.reshape(-1)
            for region in _slices(flat.numel(), chunk_size):
                flat[region].copy_((old[region].float() + alpha*(native[region].float()-old[region].float())).to(flat.dtype))
    return dict(endpoint_statistics(parameters, theta0, theta_native, chunk_size), alpha=alpha,
        target_norm=float(target_norm), absolute_norm_mismatch=abs(norm-target_norm),
        relative_norm_mismatch=abs(norm-target_norm)/target_norm if target_norm else (0. if norm==0 else None),
        bracket=[lo, hi], bracket_norms=[lo_norm, hi_norm], iterations=iterations,
        norm_scan_passes=iterations+1)
