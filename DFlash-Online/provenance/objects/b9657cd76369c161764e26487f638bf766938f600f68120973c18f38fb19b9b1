"""Apply the tested post-K correction once per ordinary accumulation window.

The trainer, its optimizer and sampling stream remain alive for the whole event.
Only the two-response prefix recorder and its correction buffers are replaced.
"""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import time

import numpy as np
import torch

from .prefix_correction import PrefixCorrection
from .rng import capture_rng
from .storage import write_json


def _equal(left, right):
    """Exact equality for the small runtime tree and native Adam tensors."""
    if torch.is_tensor(left):
        return torch.is_tensor(right) and left.dtype == right.dtype and torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(
            _equal(a, b) for a, b in zip(left, right))
    return left == right


def _versions(parameters):
    return tuple((id(parameter), parameter._version) for parameter in parameters)


def _runtime(trainer):
    model = trainer.bundle.model
    cache_names = ('_cached_block_mask', '_cached_seq_len', '_cached_bsz')
    return {
        'rng': capture_rng(),
        'modes': tuple((id(module), module.training) for module in model.modules()),
        'caches': tuple((name, id(getattr(model, name))) for name in cache_names
                        if hasattr(model, name)),
        'counters': (trainer.global_step, trainer.presentations, trainer.core._micro),
    }


class AuxiliaryIntegrity:
    """One device-local Adam copy; no model-sized CPU transfers or master weights.

    Parameters are checked by identity/version during read-only passes. Tests
    additionally compare their actual values. Adam and existing gradients are
    compared exactly; all copies are discarded before the next window.
    """

    def __init__(self, trainer):
        self.trainer = trainer
        self.parameters = tuple(trainer.backend.optimizer.model_params)
        self.optimizer = deepcopy(trainer.backend.optimizer.optimizer.state_dict())
        self.gradients = [None if parameter.grad is None else parameter.grad.detach().clone()
                          for parameter in self.parameters]
        self.runtime = _runtime(trainer)
        self.parameter_versions = _versions(self.parameters)

    def check(self, phase, *, parameters_read_only):
        comparisons = {
            'optimizer': _equal(self.optimizer, self.trainer.backend.optimizer.optimizer.state_dict()),
            'gradients': _equal(self.gradients, [parameter.grad for parameter in self.parameters]),
            'runtime': _equal(self.runtime, _runtime(self.trainer)),
            'parameter_identity': tuple(id(parameter) for parameter in self.parameters)
                                  == tuple(id(parameter) for parameter in self.trainer.backend.optimizer.model_params),
        }
        if parameters_read_only:
            comparisons['parameter_versions'] = self.parameter_versions == _versions(self.parameters)
        if not all(comparisons.values()):
            failed = ', '.join(name for name, equal in comparisons.items() if not equal)
            raise ValueError(f'{phase} changed protected training state: {failed}')
        self.parameter_versions = _versions(self.parameters)
        return comparisons


def _add_counts(destination, source):
    for name, value in source.items():
        destination[name] = destination.get(name, 0) + value


def _failure_json(value):
    """Keep nonfinite failure evidence explicit without invalid JSON numbers."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {name: _failure_json(item) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_failure_json(item) for item in value]
    return value


class PrefixCorrectionEvent:
    """Wrap existing train_step calls; finish correction before returning step 2.

    ``first_window_check`` receives the unmodified detailed PrefixCorrection
    report after the step-1 corrected probe. It must raise on a discrepancy.
    """

    def __init__(self, trainer, *, output_dir=None, probe_steps=(1, 64, 128),
                 first_window_check=None):
        if trainer.core.accumulation_remainder:
            raise ValueError('KC must start at an ordinary optimizer boundary')
        self.trainer = trainer
        self.output_dir = Path(output_dir) if output_dir is not None else None
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.probe_steps = frozenset(probe_steps)
        if first_window_check is not None and 1 not in self.probe_steps:
            raise ValueError('The first-window reproduction gate needs the step-1 probe')
        self.first_window_check = first_window_check
        self.start_step = trainer.global_step
        self.steps, self.probes = [], []
        self.work, self.costs, self.forward_calls = {}, {}, {}
        self.counts = dict(K_backward_attempted=0, K_backward_completed=0,
                           K_backward_failed=0, Adam_attempted=0, Adam_completed=0,
                           Adam_failed=0, presentations_attempted=0,
                           presentations_completed=0, presentations_failed=0, windows_completed=0)
        self.status = 'ready'
        self.active = self.capture = None
        self.closed = False
        self.original_train_step = trainer.core.train_step
        self.original_backward = trainer.backend.backward
        self.original_adam_step = trainer.backend.optimizer.optimizer.step
        self.contract = {
            'arm': 'KC', 'version': 1, 'base_objective': 'osd_first_error',
            'capture': 'current_KC_strict_prefix_per_two_response_window',
            'weights': 'descending_prefix_length',
            'violation': 'sum(weight*relu(-native_margin))/global_two_response_weight_sum',
            'correction': 'one_post_K_negative_V_over_gradient_squared_norm_step',
            'decision_precision': 'float32', 'parameter_cast': 'bfloat16',
            'optimizer': 'unchanged_native_K_AdamW', 'probe_steps': sorted(self.probe_steps),
            'epsilon': None, 'extra_optimizer_steps': 0,
        }
        trainer.contract['prefix_correction_event'] = deepcopy(self.contract)

    def __enter__(self):
        self.trainer.core.train_step = self._train_step
        self.trainer.backend.backward = self._backward
        self.trainer.backend.optimizer.optimizer.step = self._adam_step
        self.status = 'running'
        return self

    def _backward(self, *args, **kwargs):
        self.counts['K_backward_attempted'] += 1
        try:
            result = self.original_backward(*args, **kwargs)
        except BaseException:
            self.counts['K_backward_failed'] += 1
            raise
        self.counts['K_backward_completed'] += 1
        return result

    def _adam_step(self, *args, **kwargs):
        self.counts['Adam_attempted'] += 1
        try:
            result = self.original_adam_step(*args, **kwargs)
        except BaseException:
            self.counts['Adam_failed'] += 1
            raise
        self.counts['Adam_completed'] += 1
        return result

    def _begin_window(self):
        self.window_step = self.trainer.global_step - self.start_step + 1
        self.window_started = time.perf_counter()
        self.window_integrity_s = 0.0
        self.window_integrity = {}
        self.active = PrefixCorrection(self.trainer)
        self.capture = self.active.capture_k()
        self.capture.__enter__()

    def _train_step(self, batch, context=None):
        try:
            if self.active is None:
                self._begin_window()
            self.active.record_ids += tuple(batch.sample_ids)
            self.counts['presentations_attempted'] += 1
            result = self.original_train_step(batch, context)
            self.counts['presentations_completed'] += 1
            if result.optimizer_stepped:
                capture = self.capture
                self.capture = None
                capture.__exit__(None, None, None)
                self._finish_window()
            return result
        except BaseException as error:
            self.status = 'failed'
            self.counts['presentations_failed'] = (
                self.counts['presentations_attempted'] - self.counts['presentations_completed'])
            try:
                if self.capture is not None:
                    capture = self.capture
                    self.capture = None
                    capture.__exit__(type(error), error, error.__traceback__)
                self._record_window('failed', error)
            except BaseException as evidence_error:
                error.add_note(f'KC failure recording also failed: {evidence_error}')
            raise

    def _checked(self, snapshot, phase, *, parameters_read_only):
        started = time.perf_counter()
        try:
            self.window_integrity[phase] = snapshot.check(
                phase, parameters_read_only=parameters_read_only)
        finally:
            self.window_integrity_s += time.perf_counter() - started

    def _finish_window(self):
        started = time.perf_counter()
        snapshot = AuxiliaryIntegrity(self.trainer)
        self.window_integrity_s += time.perf_counter() - started
        self.active.measure_post_k()
        self._checked(snapshot, 'post_K_replay', parameters_read_only=True)
        correction = self.active.apply_once()
        self._checked(snapshot, 'correction', parameters_read_only=correction['status'] != 'applied')
        if self.window_step in self.probe_steps:
            self.active.measure_post_c()
            self._checked(snapshot, 'corrected_probe', parameters_read_only=True)
        del snapshot
        if self.window_step == 1 and self.first_window_check is not None:
            started = time.perf_counter()
            self.first_window_check(self.active.report())
            self.window_integrity_s += time.perf_counter() - started
            self.window_integrity['retained_first_window'] = True
        self._record_window('completed')
        self.counts['windows_completed'] += 1

    def _record_window(self, status, error=None):
        if self.active is None:
            if error is not None:
                failure = {'step': self.window_step, 'status': status,
                           'actual_Adam_steps_completed': self.counts['Adam_completed'],
                           'error': {'type': type(error).__name__, 'message': str(error)}}
                if self.steps and self.steps[-1]['step'] == self.window_step:
                    self.steps[-1].update(failure)
                else:
                    self.steps.append(failure)
            return
        details = self.active.report()
        if error is not None:
            details = _failure_json(details)
        record = {
            'step': self.window_step, 'status': status,
            'record_ids': list(self.active.record_ids),
            'trainer_step_before_return': self.trainer.global_step,
            'actual_Adam_steps_completed': self.counts['Adam_completed'],
            'correction': details['correction'],
            'post_K_totals': details['post_k']['totals'] if details['post_k'] else None,
            'integrity': self.window_integrity, 'integrity_s': self.window_integrity_s,
            'window_wall_s': time.perf_counter() - self.window_started,
            'work': details['work'], 'costs': details['costs'],
        }
        if error is not None:
            record['error'] = {'type': type(error).__name__, 'message': str(error)}
        self.steps.append(record)
        _add_counts(self.work, details['work'])
        _add_counts(self.costs, details['costs'])
        _add_counts(self.forward_calls, details['forward_calls'])
        self.costs['integrity_s'] = self.costs.get('integrity_s', 0.0) + self.window_integrity_s
        if self.window_step in self.probe_steps:
            self.probes.append({'step': self.window_step, **details})
        try:
            if self.output_dir is not None:
                write_json(self.output_dir / f'window_{self.window_step:03d}.json',
                           {'summary': record, 'details': details})
        finally:
            self.active.close()
            self.active = None

    def report(self):
        return {
            'arm': 'KC', 'status': self.status, 'contract': deepcopy(self.contract),
            'counts': dict(self.counts), 'work': dict(self.work), 'costs': dict(self.costs),
            'forward_calls': dict(self.forward_calls), 'steps': list(self.steps),
            'probes': list(self.probes),
            'cost_scopes': {
                'post_k_s_plus_correction_s': 'intrinsic added correction work',
                'post_c_s': 'diagnostic corrected-state probes only',
                'integrity_s': 'state snapshots, exact Adam/RNG checks and reproduction gate',
                'prefix_chunk_capture_s': 'partial capture cost, nested in ordinary K work',
                'window_wall_s': 'complete window, including inter-presentation input loading',
            },
            'integrity_scope': 'exact device-local Adam/gradient values and RNG/runtime; read-only parameter identity/version checks',
        }

    def close(self):
        if self.closed:
            return
        self.trainer.core.train_step = self.original_train_step
        self.trainer.backend.backward = self.original_backward
        self.trainer.backend.optimizer.optimizer.step = self.original_adam_step
        if self.active is not None:
            self.status = 'incomplete'
            if self.capture is not None:
                self.capture.__exit__(RuntimeError, RuntimeError('Incomplete KC window'), None)
                self.capture = None
            self._record_window('incomplete')
        if self.status == 'running':
            self.status = 'completed'
        self.original_train_step = self.original_backward = self.original_adam_step = None
        self.trainer = self.first_window_check = None
        self.closed = True

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_value is not None:
            self.status = 'failed'
        try:
            self.close()
        except BaseException as cleanup_error:
            if exc_value is None:
                raise
            exc_value.add_note(f'KC cleanup also failed: {cleanup_error}')
        return False
