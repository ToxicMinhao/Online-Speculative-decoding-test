"""Filter the actual qualified AdamW displacement, without altering its history.

Native training has BF16 parameters and BF16 Adam moments, not FP32 master
weights. Raw group gradients are FP32 copies of the parameter gradients produced
by autograd; promoting a BF16 gradient does not recover precision already lost.
The separately computed native full-loss gradient supplies the ordinary one-time
global clip and AdamW step. A+B is never substituted for that gradient: separate
low-precision backwards need not add to the direct full-loss backward. Raw A/B
gradients are used only for filtering and its diagnostics. The resulting signed
displacement is measured in FP32 from native pre/post weights. No gradient/
displacement products decide the mask.

FP64 reductions determine N's single global alpha and diagnostics. N's result is
rounded back to the native parameter dtype; the actual applied norm is therefore
reported separately from the ideal scaled norm. This function deliberately uses
one ordinary optimizer.step(), preserving moments, decay and step counters even
when the eventual applied displacement is zero. Errors after that call are fatal
for the owning trainer; this is not a transactional optimizer rollback.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
import importlib.util
from pathlib import Path

_control_spec = importlib.util.spec_from_file_location('_agreement_norm_controls', Path(__file__).with_name('controls.py'))
controls = importlib.util.module_from_spec(_control_spec)
_control_spec.loader.exec_module(controls)

NUMERICS = {
    'version': 'agreement_direct_full_loss_gradient_v2',
    'native_parameter_dtype': 'torch.bfloat16',
    'native_optimizer_moment_dtype': 'torch.bfloat16',
    'raw_group_gradient_buffer_dtype': 'torch.float32',
    'full_loss_gradient_dtype': 'native parameter dtype; BF16 in the qualified runtime',
    'aggregate': 'direct full-response loss gradient; never reconstructed from A+B',
    'proposal_gradient_source': 'explicit full_loss_gradient mapping from the same current weights and sampled forward',
    'clipping': 'torch.nn.utils.clip_grad_norm_; max_norm=1; foreach=None; once',
    'displacement': 'native AdamW post weight promoted to FP32 minus native pre weight promoted to FP32',
    'sign_check': 'non-strict sign comparisons on finite raw FP32 gradients and FP32 displacement; no products',
    'sign_tolerance': 0.0,
    'reductions': 'FP64 global reductions over all trainable parameters',
    'uniform_control_policy': dict(controls.N_NORM_POLICY),
    'uniform_scale_application': 'FP32 alpha*d added to FP32 pre weight, then native dtype rounding',
    'filter_application': 'retain exact ordinary proposed native weight or restore exact native pre weight',
    'reduction_chunk_elements': 1_048_576,
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def agreement_mask(raw_a, raw_b, displacement):
    """Return exactly (a*d <= 0) & (b*d <= 0), without underflowing products.

    Zeros (including signed zero) satisfy the non-strict inequality. Precision
    conversion is the caller's responsibility; inputs must already be FP32.
    """
    import torch
    _require(all(x.dtype == torch.float32 for x in (raw_a, raw_b, displacement)),
             'Sign checks require explicitly pinned raw FP32 tensors')
    _require(raw_a.shape == raw_b.shape == displacement.shape and
             raw_a.device == raw_b.device == displacement.device,
             'Sign-check tensors differ in shape/device')
    _require(all(bool(torch.isfinite(x).all()) for x in (raw_a, raw_b, displacement)),
             'Nonfinite sign-check input')
    def permitted(g):
        return ((displacement == 0) | (g == 0) |
                ((displacement < 0) & (g > 0)) | ((displacement > 0) & (g < 0)))
    return permitted(raw_a) & permitted(raw_b)


def _state_steps(optimizer, parameters, *, require_existing=False):
    import torch
    steps = {}
    for name, parameter in parameters.items():
        state = optimizer.state.get(parameter, {})
        if not state:
            _require(not require_existing, 'AdamW did not create complete state')
            steps[name] = 0
            continue
        _require('step' in state and 'exp_avg' in state and 'exp_avg_sq' in state,
                 'Incomplete existing AdamW state')
        step = float(state['step'])
        _require(math.isfinite(step) and step >= 0 and step.is_integer(), 'Invalid AdamW step counter')
        steps[name] = int(step)
        for key in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
            if key not in state:
                continue
            value = state[key]
            _require(value.dtype == parameter.dtype and value.shape == parameter.shape and
                     value.device == parameter.device and bool(torch.isfinite(value).all()),
                     f'Invalid/nonfinite AdamW {key} for {name}')
    _require(len(set(steps.values())) == 1, 'Trainable parameter Adam counters differ')
    return steps


def _validate(parameters, optimizer, gradients_a, gradients_b, full_loss_gradient, mode, clip_norm):
    import torch
    _require(isinstance(parameters, Mapping) and bool(parameters), 'Ordered trainable parameters required')
    _require(isinstance(optimizer, torch.optim.AdamW), 'Qualified torch AdamW required')
    _require(mode in ('G', 'N', 'H', 'L', 'unfiltered'), 'Unsupported filter mode')
    _require(clip_norm == 1.0, 'Keep the qualified global gradient clip of 1.0')
    _require(list(gradients_a) == list(parameters) == list(gradients_b),
             'Group gradients must cover the complete ordered parameter inventory')
    _require(isinstance(full_loss_gradient, Mapping) and list(full_loss_gradient) == list(parameters),
             'Explicit full-loss gradients must cover the complete ordered parameter inventory')
    values = list(parameters.values())
    _require(len({id(p) for p in values}) == len(values), 'Duplicate trainable parameter')
    owned = [p for group in optimizer.param_groups for p in group['params']]
    _require([id(p) for p in owned] == [id(p) for p in values], 'AdamW parameter ownership/order differs')
    _require(len({p.device for p in values}) == len({p.dtype for p in values}) == 1,
             'Qualified training uses one device and one native dtype')
    _require(values[0].dtype in (torch.bfloat16, torch.float32), 'Unsupported parameter precision')
    # FP32 is accepted for deterministic CPU fixtures. Native startup separately
    # requires the original 58 BF16 parameters and the original model inventory.
    for name, parameter in parameters.items():
        _require(parameter.requires_grad and parameter.is_leaf and parameter.numel() > 0 and
                 bool(torch.isfinite(parameter).all()), f'Invalid/nonfinite trainable parameter {name}')
        for label, gradients in (('A', gradients_a), ('B', gradients_b)):
            gradient = gradients[name]
            _require(isinstance(gradient, torch.Tensor) and not gradient.is_sparse and
                     gradient.dtype == torch.float32 and gradient.shape == parameter.shape and
                     gradient.device == parameter.device and bool(torch.isfinite(gradient).all()),
                     f'Invalid/nonfinite raw FP32 group {label} gradient for {name}')
        full = full_loss_gradient[name]
        _require(isinstance(full, torch.Tensor) and not full.is_sparse and
                 full.dtype == parameter.dtype and full.shape == parameter.shape and
                 full.device == parameter.device and bool(torch.isfinite(full).all()),
                 f'Invalid/nonfinite native full-loss gradient for {name}')
    return _state_steps(optimizer, parameters)


def _empty_totals(device):
    import torch
    return {key: torch.zeros((), device=device, dtype=torch.float64) for key in (
        'proposed_sq', 'filtered_sq', 'applied_sq', 'proposed_a', 'proposed_b',
        'filtered_a', 'filtered_b', 'applied_a', 'applied_b', 'coordinates',
        'keep_coordinates', 'proposed_nonzero', 'kept_nonzero', 'clipped_gradient_sq')}


def _accumulate(totals, displacement, a, b, mask, *, applied_only=False):
    """Bound FP64 temporaries instead of promoting a whole billion-weight model."""
    flat = [x.reshape(-1) for x in (displacement, a, b)]
    flat_mask = None if mask is None else mask.reshape(-1)
    chunk = NUMERICS['reduction_chunk_elements']
    for start in range(0, displacement.numel(), chunk):
        end = start + chunk
        d, ga, gb = (x[start:end].double() for x in flat)
        if applied_only:
            totals['applied_sq'].add_(d.dot(d))
            totals['applied_a'].add_(ga.dot(d))
            totals['applied_b'].add_(gb.dot(d))
            continue
        keep = flat_mask[start:end]
        filtered = d * keep  # Products are diagnostic FP64 arithmetic, never mask tests.
        totals['proposed_sq'].add_(d.dot(d))
        totals['filtered_sq'].add_(filtered.dot(filtered))
        totals['proposed_a'].add_(ga.dot(d))
        totals['proposed_b'].add_(gb.dot(d))
        totals['filtered_a'].add_(ga.dot(filtered))
        totals['filtered_b'].add_(gb.dot(filtered))
        totals['coordinates'].add_(d.numel())
        totals['keep_coordinates'].add_(keep.sum())
        totals['proposed_nonzero'].add_((d != 0).sum())
        totals['kept_nonzero'].add_(((d != 0) & keep).sum())


def apply_adamw_step(parameters, optimizer, gradients_a, gradients_b, *, full_loss_gradient,
                     mode='G', clip_norm=1.0):
    """Apply one step and return JSON-safe numerical/optimizer diagnostics.

    A/B gradients must come from the same current weights and use the original
    full-loss denominator. ``full_loss_gradient`` is mandatory and must be the
    direct full-loss gradient in the native parameter dtype. Caller owns matching
    its sampled forward/current weights to A/B and zero_grad afterward. Existing
    .grad buffers are replaced with independent clones of that full-loss gradient
    and left containing its clipped value. Neither the full-loss input nor raw
    group input buffers are modified. There is no A+B fallback. H/L/unfiltered
    are fixtures/qualification modes; live H/L uses the unchanged ordinary trainer.
    """
    import torch
    before_steps = _validate(parameters, optimizer, gradients_a, gradients_b,
                             full_loss_gradient, mode, clip_norm)
    with torch.no_grad():
        for name, parameter in parameters.items():
            # Clipping mutates .grad in place; never alias the recorded direct
            # gradient or reconstruct it from separately differentiated groups.
            parameter.grad = full_loss_gradient[name].detach().clone()
        # Match the qualified clip exactly, including its native dtype and foreach
        # dispatch. Do not clip the two group gradients individually.
        clip_return = torch.nn.utils.clip_grad_norm_(list(parameters.values()), clip_norm,
                                                    error_if_nonfinite=True, foreach=None)
        _require(bool(torch.isfinite(clip_return)), 'Nonfinite aggregate gradient norm')
        previous = {name: parameter.detach().clone() for name, parameter in parameters.items()}
        optimizer.step()  # Exactly one advancement, never repeated for A/B or N.
        after_steps = _state_steps(optimizer, parameters, require_existing=True)
        _require(all(after_steps[name] == before_steps[name] + 1 for name in parameters),
                 'AdamW must advance every parameter counter exactly once')
        totals = _empty_totals(next(iter(parameters.values())).device)
        for name, parameter in parameters.items():
            _require(bool(torch.isfinite(parameter).all()), 'Nonfinite ordinary AdamW proposal')
            d = parameter.detach().float() - previous[name].float()
            a, b = gradients_a[name], gradients_b[name]
            mask = agreement_mask(a, b, d)
            _accumulate(totals, d, a, b, mask)
            for chunk in parameter.grad.detach().reshape(-1).split(NUMERICS['reduction_chunk_elements']):
                value = chunk.double()
                totals['clipped_gradient_sq'].add_(value.dot(value))
            if mode == 'G':
                parameter.copy_(torch.where(mask, parameter, previous[name]))
        scalars = {key: float(value) for key, value in totals.items()}
        _require(all(math.isfinite(x) for x in scalars.values()), 'Nonfinite FP64 step diagnostics')
        proposed_norm = math.sqrt(scalars['proposed_sq'])
        filtered_norm = math.sqrt(scalars['filtered_sq'])
        alpha = filtered_norm / proposed_norm if proposed_norm else 0.0
        _require(0.0 <= alpha <= 1.0 + 1e-12, 'Filtered global norm exceeds proposed norm')
        # A ratio barely above one can only be reduction rounding; never silently
        # clamp the applied scale. The audit exposes the exact computed value.
        if mode == 'N':
            for name, parameter in parameters.items():
                d = parameter.detach().float() - previous[name].float()
                if alpha == 0.0:
                    parameter.copy_(previous[name])
                elif alpha != 1.0:
                    parameter.copy_((previous[name].float() + d * alpha).to(parameter.dtype))
                # alpha=1 preserves the exact ordinary proposed native value.
        for name, parameter in parameters.items():
            _require(bool(torch.isfinite(parameter).all()), 'Nonfinite applied update')
            actual = parameter.detach().float() - previous[name].float()
            _accumulate(totals, actual, gradients_a[name], gradients_b[name], None, applied_only=True)
        scalars = {key: float(value) for key, value in totals.items()}
        _require(all(math.isfinite(x) for x in scalars.values()), 'Nonfinite applied-step diagnostics')
        # In exact arithmetic the mask enforces both inequalities coordinatewise.
        # FP64 accumulation of nonpositive terms cannot turn this into a positive
        # sum. This is a source-gradient statement, not a future-loss guarantee.
        _require(scalars['filtered_a'] <= 0 and scalars['filtered_b'] <= 0,
                 'Agreement-filter directional derivative became positive')
        applied_norm = math.sqrt(scalars['applied_sq'])
        return {
            'mode': mode, 'numerics': dict(NUMERICS), 'actual_parameter_dtype': str(next(iter(parameters.values())).dtype),
            'proposal_gradient_source': 'direct_full_loss_gradient',
            'optimizer_steps_before': next(iter(before_steps.values())),
            'optimizer_steps_after': next(iter(after_steps.values())), 'optimizer_step_calls': 1,
            'gradient_norm_before_clip': float(clip_return),
            'gradient_norm_after_clip_fp64': math.sqrt(scalars['clipped_gradient_sq']),
            'proposed_step_norm': proposed_norm, 'hypothetical_filtered_step_norm': filtered_norm,
            'uniform_scale_alpha': alpha if mode == 'N' else None,
            'uniform_control': controls.norm_match(applied_norm, filtered_norm) if mode == 'N' else None,
            'own_hypothetical_uniform_scale': alpha,
            'ideal_applied_step_norm': filtered_norm if mode in ('G', 'N') else proposed_norm,
            'applied_step_norm': applied_norm,
            'native_rounding_norm_difference': applied_norm - (filtered_norm if mode in ('G', 'N') else proposed_norm),
            'coordinates': int(scalars['coordinates']),
            'mask_kept_coordinates': int(scalars['keep_coordinates']),
            'mask_keep_fraction': scalars['keep_coordinates'] / scalars['coordinates'],
            'proposed_nonzero_coordinates': int(scalars['proposed_nonzero']),
            'kept_nonzero_coordinates': int(scalars['kept_nonzero']),
            'nonzero_keep_fraction': (scalars['kept_nonzero'] / scalars['proposed_nonzero']
                                      if scalars['proposed_nonzero'] else 0.0),
            'applied_zero': applied_norm == 0.0,
            'group_directional_derivatives': {
                'proposed': {'A': scalars['proposed_a'], 'B': scalars['proposed_b']},
                'hypothetical_filtered': {'A': scalars['filtered_a'], 'B': scalars['filtered_b']},
                'applied': {'A': scalars['applied_a'], 'B': scalars['applied_b']},
            },
        }
