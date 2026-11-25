"""Frozen frontier roles, signed allocation and the declared optimizer shadow.

The shadow is continuous FP32 algebra on promoted native state. It is not an
exact derivative of the discrete BF16 update. No live state is mutated here.
"""

import math

import torch
import torch.nn.functional as F

from online_feedback.first_error import valid_prefix_mask
from online_feedback.kplus import MASS_ATOL, MASS_RTOL

RELATIVE_CAP = 0.25
CLIP_EPSILON = 1e-6  # torch.nn.utils.clip_grad_norm_, not an algorithmic floor.


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


@torch.no_grad()
def current_roles(proposals, labels, weights, eos_token_ids=()):
    """Freeze each current block's prefix, first error and reachable endpoint.

    Slot zero is the native seed. e=-1 and H=0 denote an all-correct or empty
    block; K remains the valid length. All returned teaching metadata is detached.
    """
    require(proposals.shape == labels.shape == weights.shape, 'Role tensor shapes differ')
    require(
        labels.ndim >= 1 and labels.shape[-1] >= 2, 'Expected native seed and proposal slots'
    )
    valid = valid_prefix_mask(labels, weights, eos_token_ids)
    matches = proposals.detach() == labels
    errors = valid & ~matches
    earlier = errors.long().cumsum(-1)
    prefix = valid & (earlier == 0)
    length = prefix.sum(-1)
    valid_length = valid.sum(-1)
    has_error = length < valid_length
    position = torch.arange(labels.shape[-1], device=labels.device)
    error_position = torch.where(has_error, length + 1, -1)
    repair = valid & (position == error_position.unsqueeze(-1))
    later_error = errors & (position > error_position.unsqueeze(-1))
    next_error = torch.where(later_error, position, valid_length.unsqueeze(-1) + 1).amin(-1)
    endpoint = torch.where(has_error, next_error - 1, valid_length)
    horizon = torch.where(has_error, endpoint - length, 0)
    assessment_mask = valid & (position <= endpoint.unsqueeze(-1))
    assessment_weight = torch.where(
        assessment_mask, endpoint.unsqueeze(-1) - position + 1, 0
    ).float()
    beta = weights.detach().float().clone()
    denominator = (beta * (prefix | repair)).sum() + 1e-6
    return dict(
        labels=labels.detach().clone(),
        valid=valid,
        prefix=prefix,
        repair=repair,
        assessment_mask=assessment_mask,
        assessment_weight=assessment_weight,
        L=length,
        e=error_position,
        H=horizon,
        K=endpoint,
        m=valid_length,
        beta=beta,
        denominator=denominator.detach(),
    )


def assessment(logits, roles):
    """Response-mean R with a fresh deterministic strongest-wrong active branch.

    Select native logits before FP32 promotion. The detached argmax chooses one
    lowest-ID rival at ties; gather preserves the chosen branch's derivative.
    """
    require(
        logits.shape[:-1] == roles['labels'].shape, 'Assessment logits lost their role binding'
    )
    mask = roles['assessment_mask']
    count = int(roles['valid'].any(-1).sum())
    if count == 0:
        zero = logits.reshape(-1)[:0].float().sum()
        return zero, dict(
            nonempty_blocks=0, assessed_positions=0, margin_ties=0, wrong_rival_ties=0
        )
    selected = logits[mask]
    require(torch.isfinite(selected).all(), 'Nonfinite assessed logits')
    labels = roles['labels'][mask]
    wrong_logits = selected.detach().clone()
    wrong_logits.scatter_(-1, labels.unsqueeze(-1), -torch.inf)
    rivals = wrong_logits.argmax(-1)
    canonical = selected.gather(-1, labels.unsqueeze(-1)).squeeze(-1).float()
    competitor = selected.gather(-1, rivals.unsqueeze(-1)).squeeze(-1).float()
    margins = canonical - competitor
    weights = roles['assessment_weight'][mask].detach()
    value = (weights * F.softplus(-margins) / math.log(2.0)).sum() / count
    with torch.no_grad():
        predictions = selected.argmax(-1)
        wrong_max = wrong_logits.gather(-1, rivals.unsqueeze(-1))
        rival_ties = ((wrong_logits == wrong_max).sum(-1) > 1).sum()
        diagnostics = dict(
            nonempty_blocks=count,
            assessed_positions=int(mask.sum()),
            margin_ties=int((margins == 0).sum()),
            wrong_rival_ties=int(rival_ties),
            rivals=rivals.detach(),
            predictions=predictions.detach(),
            correct=(predictions == labels).detach(),
            margins=margins.detach(),
        )
    return value, diagnostics


def window_mean(response_values):
    """Two responses always have factor one half, including an empty response."""
    require(len(response_values) == 2, 'Require the native two-response accumulation window')
    return (response_values[0] + response_values[1]) * 0.5


def response_loss(
    token_losses, prefix_coefficients, repair_coefficients, roles, *, scoring=False
):
    """Reference x stays differentiable; final a and repair coefficients detach.

    token_losses already contains native FKL, plus PAIR only at current repair
    positions. Assessment continuation positions have no direct committed loss.
    """
    require(
        token_losses.shape
        == prefix_coefficients.shape
        == repair_coefficients.shape
        == roles['prefix'].shape,
        'Teaching tensors lost their role binding',
    )
    prefix = prefix_coefficients if scoring else prefix_coefficients.detach()
    coefficients = prefix * roles['prefix'] + repair_coefficients.detach() * roles['repair']
    return (token_losses * coefficients).sum() / roles['denominator'].detach()


@torch.no_grad()
def coefficient_sensitivities(token_loss_directional, roles):
    """Contract current token gradients with the optimizer VJP, once per response."""
    direction = token_loss_directional
    if direction.dtype != torch.float64:  # FP64 is qualification-only.
        direction = direction.float()
    return -torch.where(roles['prefix'], direction, 0.0) / (2 * roles['denominator'])


@torch.no_grad()
def allocate_prefix(scores, beta, prefix, *, relative_cap=RELATIVE_CAP):
    """Exact signed block-local centering; no positive-score rectification."""
    require(
        relative_cap in (0.0, RELATIVE_CAP),
        'Only cap 0.25 or qualification-only cap zero is supported',
    )
    require(scores.shape == beta.shape == prefix.shape, 'Allocation tensor shapes differ')
    scores, base = scores.detach().float(), beta.detach().float()
    require(
        torch.isfinite(scores).all() and torch.isfinite(base).all(),
        'Nonfinite scores or native coefficients',
    )
    require((base[prefix] > 0).all(), 'Prefix coefficients must be positive')
    count = prefix.sum(-1, keepdim=True)
    mass = (base * prefix).sum(-1, keepdim=True)
    lowest = torch.where(prefix, scores, torch.inf).amin(-1, keepdim=True)
    origin = torch.where(count > 0, lowest, 0.0)
    # Algebraically identical weighted centering, evaluated after subtracting
    # one pool reference. This preserves tiny nonconstant FP32 contrasts without
    # a floor, threshold or altered cap. Exact constant/singleton pools give 0.
    offsets = torch.where(prefix, scores - origin, 0.0)
    mean_offset = (base * offsets).sum(-1, keepdim=True) / torch.where(mass > 0, mass, 1.0)
    mean = origin + mean_offset
    center = torch.where(prefix, offsets - mean_offset, 0.0)
    maximum = center.abs().amax(-1, keepdim=True)
    ratio = center / torch.where(maximum > 0, maximum, 1.0)
    candidate = torch.where(prefix, base * (1 + relative_cap * ratio), base)
    require(
        torch.isfinite(mean).all() and torch.isfinite(candidate).all(),
        'Nonfinite coefficient arithmetic',
    )
    after = (candidate * prefix).sum(-1, keepdim=True)
    tolerance = MASS_ATOL + MASS_RTOL * base.abs()
    require(
        torch.isclose(after, mass, rtol=MASS_RTOL, atol=MASS_ATOL).all(),
        'Prefix mass conservation failed',
    )
    require(
        (
            (candidate >= (1 - relative_cap) * base - tolerance)
            & (candidate <= (1 + relative_cap) * base + tolerance)
        ).all(),
        'Prefix relative cap failed',
    )
    require(
        (candidate[~prefix] == base[~prefix]).all(), 'Allocation changed non-prefix support'
    )
    benefit = ((candidate - base) * scores).sum(-1, keepdim=True)
    variance_mass = (base * center.square()).sum(-1, keepdim=True)
    expected = relative_cap * variance_mass / torch.where(maximum > 0, maximum, 1.0)
    require(
        torch.isclose(benefit, expected, rtol=MASS_RTOL, atol=MASS_ATOL).all()
        and (benefit >= -MASS_ATOL).all(),
        'Signed variance identity failed',
    )
    return candidate, dict(
        prefix_positions=count,
        mass_before=mass,
        mass_after=after,
        weighted_mean_score=mean,
        maximum_centered_score=maximum,
        maximum_absolute_score=torch.where(prefix, scores.abs(), 0.0).amax(-1, keepdim=True),
        reference_branch=(maximum == 0) | (relative_cap == 0),
        half_l1_mass_moved=(candidate - base).abs().sum(-1, keepdim=True) / 2,
        signed_benefit=benefit,
        centered_signed_benefit=((candidate - base) * center).sum(-1, keepdim=True),
        variance_identity_rhs=expected,
    )


def _optimizer_options(options, max_grad_norm):
    require(
        not any(
            options.get(key, False)
            for key in ('amsgrad', 'maximize', 'capturable', 'differentiable')
        ),
        'Unsupported native AdamW option',
    )
    require(options.get('decoupled_weight_decay', True), 'Require decoupled AdamW decay')
    lr, decay, epsilon = (float(options[key]) for key in ('lr', 'weight_decay', 'eps'))
    beta1, beta2 = (float(value) for value in options['betas'])
    require(
        all(math.isfinite(value) for value in (lr, decay, epsilon, beta1, beta2, max_grad_norm))
        and lr >= 0
        and decay >= 0
        and epsilon > 0
        and max_grad_norm > 0
        and 0 <= beta1 < 1
        and 0 <= beta2 < 1,
        'Invalid native AdamW hyperparameters',
    )
    return lr, decay, epsilon, beta1, beta2


@torch.no_grad()
def adamw_shadow(
    parameters,
    gradients,
    exp_avg,
    exp_avg_sq,
    steps,
    options,
    *,
    max_grad_norm,
    dtype=torch.float32
):
    """Promoted native one-step preview plus compact analytic derivative context.

    FP64 is exposed only for continuous-algebra qualification. Production uses
    FP32. Inputs are the already accumulated/normalized reference gradients;
    this helper applies no additional response or accumulation scaling.
    """
    require(
        dtype in (torch.float32, torch.float64),
        'Shadow precision must be FP32 or qualification FP64',
    )
    require(
        len(parameters) == len(gradients) == len(exp_avg) == len(exp_avg_sq) == len(steps) > 0,
        'Shadow parameter/state lists differ',
    )
    lr, decay, epsilon, beta1, beta2 = _optimizer_options(options, max_grad_norm)
    promoted = [gradient.detach().to(dtype) for gradient in gradients]
    require(
        all(torch.isfinite(gradient).all() for gradient in promoted),
        'Nonfinite reference gradient',
    )
    norms = torch.stack([torch.linalg.vector_norm(gradient) for gradient in promoted])
    norm = torch.linalg.vector_norm(norms)
    require(torch.isfinite(norm), 'Nonfinite global gradient norm')
    raw_clip = max_grad_norm / (norm + CLIP_EPSILON)
    clip = raw_clip.clamp(max=1.0)
    preview, diagonal = [], []
    for parameter, gradient, old_mean, old_square, old_step in zip(
        parameters, promoted, exp_avg, exp_avg_sq, steps
    ):
        require(
            parameter.shape == gradient.shape == old_mean.shape == old_square.shape,
            'Shadow state shape differs',
        )
        step = float(old_step)
        require(
            math.isfinite(step) and step >= 0 and step.is_integer(),
            'Invalid native Adam step counter',
        )
        parameter = parameter.detach().to(dtype)
        mean, square = old_mean.detach().to(dtype), old_square.detach().to(dtype)
        require(
            torch.isfinite(parameter).all()
            and torch.isfinite(mean).all()
            and torch.isfinite(square).all()
            and (square >= 0).all(),
            'Invalid native Adam state',
        )
        clipped = gradient * clip
        next_mean = torch.lerp(mean, clipped, 1 - beta1)
        next_square = square * beta2 + clipped.square() * (1 - beta2)
        correction1 = 1 - beta1 ** (step + 1)
        correction2_sqrt = math.sqrt(1 - beta2 ** (step + 1))
        root = next_square.sqrt()
        denominator = root / correction2_sqrt + epsilon
        step_size = lr / correction1
        updated = parameter * (1 - lr * decay) - step_size * next_mean / denominator
        require(
            ((root > 0) | (next_mean == 0)).all(),
            'Zero Adam second moment with nonzero first moment has no unique derivative',
        )
        # At m_new=v_new=g=0 the complete Adam ratio has a removable first
        # derivative, although separately differentiating sqrt(0) yields NaN.
        root_safe = torch.where(root > 0, root, 1.0)
        denominator_derivative = (1 - beta2) * clipped / (correction2_sqrt * root_safe)
        derivative = -step_size * (
            (1 - beta1) / denominator
            - next_mean * denominator_derivative / denominator.square()
        )
        require(
            torch.isfinite(updated).all() and torch.isfinite(derivative).all(),
            'Nonfinite optimizer shadow',
        )
        preview.append(updated.detach().requires_grad_())
        diagonal.append(derivative.detach())
    context = dict(
        gradients=promoted,
        adam_diagonal=diagonal,
        norm=norm,
        clip=clip,
        clipping_active=bool(raw_clip <= 1),
        clip_epsilon=CLIP_EPSILON,
    )
    return preview, context


@torch.no_grad()
def optimizer_vjp(context, cotangents):
    """Pull an assessment cotangent through Adam and coupled global clipping."""
    require(
        len(cotangents) == len(context['adam_diagonal']), 'Optimizer cotangent list differs'
    )
    require(
        all(
            cotangent.shape == diagonal.shape
            for cotangent, diagonal in zip(cotangents, context['adam_diagonal'])
        ),
        'Optimizer cotangent shape differs',
    )
    adjoints = [
        cotangent.detach().to(diagonal.dtype) * diagonal
        for cotangent, diagonal in zip(cotangents, context['adam_diagonal'])
    ]
    if context['clipping_active']:
        norm, clip = context['norm'], context['clip']
        contraction = torch.stack(
            [
                (adjoint * gradient).sum()
                for adjoint, gradient in zip(adjoints, context['gradients'])
            ]
        ).sum()
        correction = contraction * clip / (norm * (norm + context['clip_epsilon']))
        adjoints = [
            clip * adjoint - correction * gradient
            for adjoint, gradient in zip(adjoints, context['gradients'])
        ]
    require(
        all(torch.isfinite(adjoint).all() for adjoint in adjoints), 'Nonfinite optimizer VJP'
    )
    return adjoints
