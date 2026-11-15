"""Native cached-gradient Adam trials and bounded, outcome-free norm calibration."""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from copy import copy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import time

from experiments.full_k_gsm8k.training import live_state
from experiments.ok_continuation.resume import differences
from experiments.threatened_repair_replay.state import (
    capture_state, copy_tree, named_gradients, restore_state, tensor_change, verify_state,
)
from online_feedback.storage import write_json

RELATIVE_BRACKET = 0.0025
MAX_TRIALS = 10
NORM_ABSOLUTE_TOLERANCE = 1e-12
NORM_RELATIVE_TOLERANCE = 1e-10


class CalibrationUnresolved(RuntimeError):
    """A fixed numerical calibration could not meet its prespecified contract."""


def append_receipt(path, row):
    """Persist each attempted operation before a later check can fail."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        stream.write(json.dumps(row, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def norm_agreement(actual, recorded):
    if not math.isfinite(actual) or not math.isfinite(recorded):
        return False
    return abs(actual-recorded) <= NORM_ABSOLUTE_TOLERANCE + NORM_RELATIVE_TOLERANCE*abs(recorded)


def gradient_identity(trainer):
    """Exact clipped-gradient identity, including native BF16 bits and None."""
    import torch
    result = {}
    for name, parameter in trainer.bundle.draft_model.named_parameters():
        if not parameter.requires_grad:
            continue
        gradient = parameter.grad
        if gradient is None:
            result[name] = None
            continue
        digest = hashlib.sha256()
        flat = gradient.detach().flatten()
        for start in range(0, flat.numel(), 1 << 20):
            raw = flat[start:start+(1 << 20)].cpu().contiguous().view(torch.uint8)
            digest.update(raw.numpy().tobytes())
        result[name] = dict(dtype=str(gradient.dtype), shape=list(gradient.shape),
            bytes=gradient.numel()*gradient.element_size(), sha256=digest.hexdigest())
    return result


@contextmanager
def observe_adam(trainer, expected_lrs=None):
    """Observe the actual inner application without replacing its arithmetic."""
    optimizer = trainer.backend.optimizer.optimizer
    original = optimizer.step
    observation = dict(applications=0)

    def observed(*args, **kwargs):
        lrs = [float(group['lr']) for group in optimizer.param_groups]
        observation['effective_lrs_before'] = lrs
        if expected_lrs is not None and lrs != expected_lrs:
            raise ValueError('Wrapper overwrote the declared effective learning rate')
        observation['clipped_gradients'] = gradient_identity(trainer)
        observation['applications'] += 1
        result = original(*args, **kwargs)
        observation['effective_lrs_after'] = [float(group['lr']) for group in optimizer.param_groups]
        if observation['effective_lrs_after'] != lrs:
            raise ValueError('Native Adam changed the effective learning-rate groups')
        return result

    optimizer.step = observed
    try:
        yield observation
    finally:
        optimizer.step = original


def ordinary_pair(trainer, pair, pass_index, ordinal, *, category, label):
    """The unchanged full native forward/backward, with read-only step receipts."""
    lrs = [float(group['lr']) for group in trainer.backend.optimizer.optimizer.param_groups]
    with observe_adam(trainer, lrs) as observation:
        result = trainer.run_pair(pair, pass_index, ordinal, label=label,
            category=category, capture_gradients=True)
    if observation['applications'] != 1:
        raise ValueError('A native pair must apply Adam exactly once')
    result['update']['adam_observation'] = observation
    return result


def cached_trial(trainer, pre, native, gradients, native_update, alpha, *, window,
                 trial_index, output, category='calibration'):
    """Apply native Adam once to fresh preclip copies, then restore all of N.

    Only weights, Adam moments and the optimizer-step counter may advance in the
    temporary branch. Its sampler, presentations, modes, caches and RNG stay at P.
    The immutable gradient cache and the physical-work ledger are never restored.
    """
    import torch
    from experiments.functional_step_geometry.qualify import cleanup

    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError('Scalar learning-rate multiplier must lie in [0,1]')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f'TRIAL_{trial_index:02d}.json'
    receipt = dict(status='started', window=window, trial_index=trial_index, alpha=float(alpha),
        optimizer_applications=0, response_forwards=0, response_backwards=0,
        response_presentations=0, sampling_draws=0)
    write_json(path, receipt)
    started, failure = time.perf_counter(), None
    weights = None
    try:
        restore_state(trainer, pre)
        receipt['restored_pre'] = verify_state(trainer, pre)
        parameters = {name: parameter for name, parameter in trainer.bundle.draft_model.named_parameters()
                      if parameter.requires_grad}
        if parameters.keys() != gradients.keys() or trainer.core.accumulation_remainder:
            raise ValueError('Cached trial requires native gradient membership and a complete window')
        optimizer = trainer.backend.optimizer.optimizer
        base_lrs = [float(group['lr']) for group in optimizer.param_groups]
        scaled_lrs = [alpha*value for value in base_lrs]
        receipt.update(original_lrs=base_lrs, requested_lrs=scaled_lrs)
        for name, parameter in parameters.items():
            source = gradients[name]
            parameter.grad = None if source is None else source.to(parameter.device).clone()
        # Confirm the installed tensors before clipping can mutate them.
        mismatch = differences(named_gradients(trainer), gradients, 'installed_preclip')
        if mismatch:
            raise ValueError('Fresh preclip gradient installation differs: ' + str(mismatch[:8]))
        for group, lr in zip(optimizer.param_groups, scaled_lrs):
            group['lr'] = lr
        scope = trainer.ledger.scope(category=category, window=window, alpha=alpha,
                                     trial_index=trial_index) if trainer.ledger else nullcontext()
        observation = None
        try:
            with scope, observe_adam(trainer, scaled_lrs) as observation:
                norm = trainer._native_step()
            trainer.global_step += 1
            receipt['optimizer_applications'] = observation['applications']
            receipt['clip_norm'] = float(norm)
            factor = torch.clamp(trainer.backend.optimizer.max_grad_norm/(norm.detach()+1e-6), max=1.)
            receipt.update(clip_factor=float(factor), clip_factor_dtype=str(factor.dtype),
                           adam_observation=observation)
        finally:
            for group, lr in zip(optimizer.param_groups, base_lrs):
                group['lr'] = lr
            if observation is not None:
                receipt['adam_observation'] = observation
                receipt['optimizer_applications'] = observation['applications']
        expected_clip = native_update['combined_native_clip_norm']
        if (receipt['clip_norm'] != expected_clip
                or receipt['clip_factor'] != native_update['clip_factor']
                or receipt['clip_factor_dtype'] != native_update['clip_factor_dtype']):
            raise ValueError('Cached trial changed native clipping')
        mismatch = differences(observation['clipped_gradients'],
            native_update['adam_observation']['clipped_gradients'], 'clipped_gradients')
        current = live_state(trainer)
        mismatch += differences(current['backend'], native['scientific']['backend'], 'native_adam_state')
        receipt['moment_clip_option_parity'] = dict(status='failed' if mismatch else 'passed',
                                                  mismatches=mismatch[:64])
        if mismatch:
            raise ValueError('Cached trial changed gradients, Adam moments/counters or options: '+str(mismatch[:8]))
        weights = copy_tree(current['draft_state_dict'])
        expected = copy(pre)
        expected['scientific'] = dict(pre['scientific'], draft_state_dict=weights,
            backend=native['scientific']['backend'], global_step=pre['scientific']['global_step']+1)
        expected['gradients'] = {name: None for name in gradients}
        receipt['cached_state_semantics'] = verify_state(trainer, expected)
        receipt['displacement'] = tensor_change(pre['scientific']['draft_state_dict'], weights)
        receipt.update(status='completed', end_step=trainer.global_step,
            end_presentations=trainer.presentations, restored_group_lrs=[float(g['lr']) for g in optimizer.param_groups])
        return weights, receipt
    except BaseException as error:
        failure = error
        receipt.update(status='failed', error=repr(error))
        raise
    finally:
        cleanup(lambda: restore_state(trainer, native), failure, receipt, 'restore authoritative N')
        cleanup(lambda: receipt.update(restored_native=verify_state(trainer, native)),
                failure, receipt, 'verify authoritative N')
        receipt['wall_s'] = time.perf_counter()-started
        cleanup(lambda: write_json(path, receipt), failure, receipt, 'save cached-trial evidence')


@dataclass
class Point:
    alpha: float
    norm: float
    trial_index: int
    weights: object = None
    receipt: dict | None = None


def tight_bracket(lower, upper, target):
    return (lower.norm <= target <= upper.norm
            and (target-lower.norm)/target <= RELATIVE_BRACKET
            and (upper.norm-target)/target <= RELATIVE_BRACKET)


def calibrate(target, native_norm, native_weights, trial, *, on_attempt=None):
    """Bisect observed alpha bounds; select by norm only, with at most ten calls."""
    if not (math.isfinite(target) and math.isfinite(native_norm) and 0 < target <= native_norm):
        raise CalibrationUnresolved('Historical E norm must be finite, positive and no larger than native N')
    lower = Point(0., 0., -1)
    upper = Point(1., native_norm, 0, native_weights)
    if target == native_norm:
        return upper, upper, []
    alpha_low, alpha_high = 0., 1.
    observations = [(0., 0.), (1., native_norm)]
    attempts = []
    for index in range(1, MAX_TRIALS+1):
        alpha = (alpha_low+alpha_high)/2
        attempted = dict(trial_index=index, alpha=alpha, status='started')
        if on_attempt is not None:
            on_attempt(attempted)
        try:
            weights, receipt = trial(alpha, index)
            norm = float(receipt['displacement']['difference_norm'])
            attempted = dict(receipt, trial_index=index, alpha=alpha, actual_norm=norm)
            if not math.isfinite(norm):
                raise CalibrationUnresolved('Nonfinite measured calibration norm')
            if any((old_alpha < alpha and old_norm > norm)
                   or (old_alpha > alpha and old_norm < norm)
                   for old_alpha, old_norm in observations):
                raise CalibrationUnresolved('Measured norms contradict observed monotone alpha ordering')
            observations.append((alpha,norm))
            point = Point(alpha,norm,index,weights,receipt)
            if norm == target:
                attempted['selection_status'] = 'exact_match_both_controls'
                attempts.append(attempted)
                if on_attempt is not None:
                    on_attempt(attempted)
                return point, point, attempts
            if norm < target:
                alpha_low = alpha
                if norm > lower.norm or (norm == lower.norm and (alpha,index) < (lower.alpha,lower.trial_index)):
                    lower = point
            else:
                alpha_high = alpha
                if norm < upper.norm or (norm == upper.norm and (alpha,index) < (upper.alpha,upper.trial_index)):
                    upper = point
            attempted['selection_status'] = 'measured_norm_only'
            attempts.append(attempted)
            if on_attempt is not None:
                on_attempt(attempted)
            if tight_bracket(lower,upper,target):
                return lower,upper,attempts
            # Only the two best endpoint snapshots survive to the next trial.
            del weights, point
        except BaseException as error:
            attempted.update(status='failed',error=repr(error))
            if 'actual_norm' in attempted and not math.isfinite(attempted['actual_norm']):
                attempted['actual_norm'] = repr(attempted['actual_norm'])
                attempted['displacement'] = dict(attempted['displacement'],
                    difference_norm=attempted['actual_norm'])
            attempts.append(attempted)
            if on_attempt is not None:
                on_attempt(attempted)
            raise
    raise CalibrationUnresolved('Ten native scalar trials did not attain both fixed 0.25% norm bounds')
