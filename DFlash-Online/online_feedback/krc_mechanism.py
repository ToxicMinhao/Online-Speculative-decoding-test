"""Temporary same-state interventions during an unchanged historical KRC replay.

Only the original full R correction continues. Diagnostic records own their
buffers; the qualified one-shot correction controller is never reset or reused.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import math
from pathlib import Path
import time

import torch

from .prefix_correction import canonical_margin, correction_direction, apply_correction, _sync
from .prefix_correction_event import AuxiliaryIntegrity, _equal
from .prefix_guard import preserved_probe_runtime
from .reference_correction_event import ReferenceCorrectionWindow, ReferencePrefixCorrection, CACHE_NAMES
from .rng import capture_rng, restore_rng
from .storage import write_json
from .krc_mechanism_metrics import response_metrics, pooled_summary, fitted_objective_predictions, gradient_alignment

CANDIDATES = ('R_FULL', 'R_HALF', 'D_FULL', 'K_ONLY')


def _cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_cpu_tree(item) for item in value)
    return deepcopy(value)


class ContinuationSnapshot:
    """Exact transient CPU snapshot, including buffers and absent cache fields.

    Frozen parameter values are never written by this diagnostic; their version
    counters are checked. In the pinned implementation the three cache fields remain unpopulated;
    candidate replays nevertheless invalidate them and bind each saved response.
    """
    def __init__(self, trainer):
        self.trainer = trainer
        self.model = trainer.bundle.model
        self.parameters = tuple(trainer.backend.optimizer.model_params)
        self.weights = [p.detach().cpu().clone() for p in self.parameters]
        trained = {id(p) for p in self.parameters}
        self.frozen_versions = [(p, p._version) for p in self.model.parameters() if id(p) not in trained]
        self.buffers = {name: value.detach().cpu().clone() for name, value in self.model.named_buffers()}
        self.rng = capture_rng()
        self.modes = [(module, module.training) for module in self.model.modules()]
        self.caches = {name: getattr(self.model, name) for name in CACHE_NAMES if hasattr(self.model, name)}
        self.cache_tensor_values = {name: value.detach().cpu().clone() for name, value in self.caches.items()
                                    if torch.is_tensor(value)}
        self.gradients = [None if p.grad is None else p.grad.detach().cpu().clone() for p in self.parameters]
        self.optimizer = _cpu_tree(trainer.backend.optimizer.optimizer.state_dict())
        self.counters = (trainer.global_step, trainer.presentations, trainer.core._micro)
        self.accumulation = {name: _cpu_tree(getattr(trainer.core, name))
                             for name in ('_ratio_totals', '_sum_totals', 'accumulation_remainder')
                             if hasattr(trainer.core, name)}
        self.anchor_observations = _cpu_tree(getattr(trainer, '_anchor_observations', None))
        self.sampler = self.model._sample_anchor_positions
        schedule = getattr(trainer, '_presentation_schedule', None)
        self.schedule = schedule
        self.schedule_state = _cpu_tree(vars(schedule)) if schedule is not None else None

    @torch.no_grad()
    def restore(self, *, invalidate_caches=False):
        for parameter, value in zip(self.parameters, self.weights):
            parameter.copy_(value)
        current_buffers = dict(self.model.named_buffers())
        if current_buffers.keys() != self.buffers.keys():
            raise ValueError('Diagnostic changed model buffer registration')
        for name, value in self.buffers.items():
            current_buffers[name].copy_(value)
        for parameter, value in zip(self.parameters, self.gradients):
            if value is None:
                parameter.grad = None
            elif parameter.grad is None:
                parameter.grad = value.to(parameter.device)
            else:
                parameter.grad.copy_(value)
        restore_rng(self.rng)
        for module, mode in self.modes:
            module.training = mode
        for name in CACHE_NAMES:
            if name in self.caches:
                value = self.caches[name]
                if name in self.cache_tensor_values:
                    value.copy_(self.cache_tensor_values[name])
                setattr(self.model, name, None if invalidate_caches else value)
            elif hasattr(self.model, name):
                delattr(self.model, name)
        # No diagnostic operation is permitted to advance Adam or counters.
        if not _equal(self.optimizer, _cpu_tree(self.trainer.backend.optimizer.optimizer.state_dict())):
            raise ValueError('Diagnostic modified Adam state')
        if self.counters != (self.trainer.global_step, self.trainer.presentations, self.trainer.core._micro):
            raise ValueError('Diagnostic modified training counters')

    def check(self):
        checks = {
            'parameters': all(torch.equal(p.detach().cpu(), v) for p, v in zip(self.parameters, self.weights)),
            'frozen_parameter_versions': all(p._version == version for p, version in self.frozen_versions),
            'buffers': _equal(self.buffers, {n: b.detach().cpu() for n, b in self.model.named_buffers()}),
            'gradients': _equal(self.gradients, [None if p.grad is None else p.grad.detach().cpu() for p in self.parameters]),
            'optimizer': _equal(self.optimizer, _cpu_tree(self.trainer.backend.optimizer.optimizer.state_dict())),
            'counters': self.counters == (self.trainer.global_step, self.trainer.presentations, self.trainer.core._micro),
            'rng': _equal(self.rng, capture_rng()),
            'accumulation': all(_equal(value, _cpu_tree(getattr(self.trainer.core, name)))
                                for name, value in self.accumulation.items()),
            'anchor_observations': _equal(self.anchor_observations, getattr(self.trainer, '_anchor_observations', None)),
            'sampler_binding': self.model._sample_anchor_positions == self.sampler,
            'presentation_schedule': getattr(self.trainer, '_presentation_schedule', None) is self.schedule
                and _equal(self.schedule_state, _cpu_tree(vars(self.schedule)) if self.schedule is not None else None),
            'modes': all(module.training == mode for module, mode in self.modes),
            'cache_bindings': all(hasattr(self.model, n) == (n in self.caches) and
                                  (n not in self.caches or getattr(self.model, n) is self.caches[n]) for n in CACHE_NAMES),
            'cache_tensor_values': all(torch.equal(self.caches[n].cpu(), v) for n, v in self.cache_tensor_values.items()),
        }
        if not all(checks.values()):
            raise ValueError(f'Diagnostic continuation restoration failed: {checks}')
        return checks


def _weight_identity(parameters):
    digest = hashlib.sha256()
    for parameter in parameters:
        value = parameter.detach().cpu().contiguous().view(torch.uint8)
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class DetailedReference(ReferencePrefixCorrection):
    """Same qualified forwards; retain all native margins for compact analysis."""
    @torch.no_grad()
    def chunk(self, logits, labels, validity_weights, chunk_key):
        result = super().chunk(logits, labels, validity_weights, chunk_key)
        if self.collecting:
            self.current.setdefault('margin_chunks', []).append(canonical_margin(logits, labels).cpu())
        return result

    def finish_response(self, numerator, denominator):
        result = super().finish_response(numerator, denominator)
        self.responses[-1]['pre_margins'] = torch.cat(self.responses[-1].pop('margin_chunks'), 1)
        return result

    def _score_response(self, response, predictions, margins):
        return {**super()._score_response(response, predictions, margins), 'all_margins': margins.tolist()}


def diagnostic_recorder(trainer, *, record_ids=()):
    recorder = DetailedReference(trainer, record_ids=record_ids)
    # The original recorder counts every real module forward, with a distinct
    # diagnostic phase. Avoid installing duplicate hooks for temporary scorers.
    for hook in recorder.hooks:
        hook.remove()
    recorder.hooks.clear()
    return recorder


@torch.no_grad()
def score_pair(recorder):
    recorder.phase = 'post_c'
    device = recorder.parameters[0].device
    denominator = sum((r['protected_weights'].sum().to(device) for r in recorder.responses),
                      torch.zeros((), dtype=torch.float32, device=device))
    numerator = denominator.new_zeros(())
    rows = []
    with preserved_probe_runtime(recorder.model):
        for response in recorder.responses:
            value, row, predictions = recorder._forward_response(response)
            numerator.add_(value)
            rows.append(row)
    value = numerator / denominator if float(denominator) else numerator
    if not bool(torch.isfinite(value)):
        raise ValueError('Nonfinite candidate hinge')
    return {'V': float(value), 'numerator': float(numerator), 'denominator': float(denominator), 'responses': rows}


@torch.no_grad()
def apply_dose(parameters, gradients, value, denominator, alpha):
    """Scale the qualified FP32 proposal before its one BF16 parameter cast."""
    if denominator == 0:
        status, squared, coefficient = 'empty_protection', value.new_zeros(()), None
    else:
        status, squared, coefficient = correction_direction(value, gradients)
    if coefficient is not None:
        for parameter, gradient in zip(parameters, gradients):
            delta = (coefficient * gradient) * alpha
            rounded = (parameter.float() + delta).to(parameter.dtype)
            if not bool(torch.isfinite(delta).all() & torch.isfinite(rounded).all()):
                raise ValueError('Nonfinite dose candidate')
        for parameter, gradient in zip(parameters, gradients):
            parameter.copy_((parameter.float() + (coefficient * gradient) * alpha).to(parameter.dtype))
    return {'status': status, 'coefficient_fp32': None if coefficient is None else float(coefficient),
            'gradient_norm_squared_fp32': float(squared), 'alpha': alpha}


@torch.no_grad()
def direction_evidence(parameters, before, gradients, correction, alpha):
    """FP64 reporting on the actual FP32 proposed and BF16-realized vectors."""
    coefficient = correction['coefficient_fp32']
    device = parameters[0].device
    sums = torch.zeros(4, device=device, dtype=torch.float64)
    for parameter, old, gradient in zip(parameters, before, gradients):
        realized = parameter.detach().float() - old.to(device).float()
        proposed = torch.zeros_like(gradient) if coefficient is None else (gradient * coefficient) * alpha
        sums[0] += proposed.double().square().sum()
        sums[1] += realized.double().square().sum()
        sums[2] += (gradient.double() * proposed.double()).sum()
        sums[3] += (gradient.double() * realized.double()).sum()
    a, b, c, d = map(float, sums)
    return dict(proposed_norm=math.sqrt(a), realized_norm=math.sqrt(b),
                proposed_dot=c, realized_dot=d, reporting_precision='float64 reductions; vectors float32')


class MechanismWindow(ReferenceCorrectionWindow):
    def __init__(self, trainer, window, feature_paths, *, plan, output_dir):
        super().__init__(trainer, window, feature_paths, probe=False)
        self.reference.close()
        self.reference = DetailedReference(trainer, record_ids=window['reference_record_ids'])
        self.plan, self.output_dir = plan, Path(output_dir)
        self.extra = {role: diagnostic_recorder(trainer, record_ids=[r['record_id'] for r in plan[role]])
                      for role in ('A', 'D')}
        self.diagnostic = {'step': plan['step'], 'plan': plan, 'candidate_order': list(CANDIDATES),
                           'status': 'pending', 'candidates': {}, 'work': {}, 'costs': {}}

    @contextmanager
    def capture_k(self):
        with super().capture_k():
            copy_started = time.perf_counter()
            snapshot = ContinuationSnapshot(self.trainer)
            self.diagnostic['costs']['pre_capture_snapshot_s'] = time.perf_counter() - copy_started
            started = time.perf_counter()
            try:
                for role in ('A', 'D'):
                    self.reference.phase = f'mechanism_pre_{role}'
                    self.extra[role].capture_reference(self.plan[role], self.feature_paths)
                self.diagnostic['pre_capture_integrity'] = snapshot.check()
            finally:
                self.diagnostic['costs']['pre_capture_s'] = time.perf_counter() - started
            self.reference.phase = 'learning_K'
            del snapshot
            yield self

    def apply_once(self):
        parameters = self.reference.parameters
        copy_started = time.perf_counter()
        post_k = [p.detach().cpu().clone() for p in parameters]
        self.diagnostic['costs']['post_K_parameter_snapshot_s'] = time.perf_counter() - copy_started
        correction = super().apply_once()
        copy_started = time.perf_counter()
        continuation = ContinuationSnapshot(self.trainer)
        self.diagnostic['costs']['full_continuation_snapshot_s'] = time.perf_counter() - copy_started
        phase = self.reference.phase
        started = time.perf_counter()
        r_scorer = diagnostic_recorder(self.trainer)
        r_scorer.responses = list(self.reference.responses)
        scorers = {'A': self.extra['A'], 'R': r_scorer, 'D': self.extra['D']}
        self.diagnostic['original_R_post_K'] = self.reference.post_k
        try:
            for candidate in CANDIDATES:
                candidate_started = time.perf_counter()
                if candidate != 'R_FULL':
                    continuation.restore(invalidate_caches=True)
                    with torch.no_grad():
                        for p, value in zip(parameters, post_k):
                            p.copy_(value)
                self.reference.phase = f'mechanism_{candidate}'
                restore_elapsed = time.perf_counter() - candidate_started
                construction_started = time.perf_counter()
                try:
                    fitted, alpha, info = self.reference, 1.0, correction
                    if candidate == 'R_HALF':
                        alpha = 0.5
                        info = apply_dose(parameters, fitted.gradients, fitted.value,
                                          float(fitted.global_denominator), alpha)
                    elif candidate == 'D_FULL':
                        fitted = self.extra['D']
                        fitted.k_update_norm = self.reference.k_update_norm
                        fitted.measure_post_k()
                        info = fitted.apply_once()
                        self.diagnostic['gradient_alignment'] = gradient_alignment(self.reference.gradients, fitted.gradients)
                        self.diagnostic['D_post_K'] = fitted.post_k
                    elif candidate == 'K_ONLY':
                        alpha = 0.0
                        info = {'status': 'unchanged_post_K', 'coefficient_fp32': None}
                    construction_s = time.perf_counter() - construction_started
                    identity_started = time.perf_counter()
                    identity = _weight_identity(parameters)
                    identity_s = time.perf_counter() - identity_started
                    entry = {'status': 'valid', 'branch': info, 'alpha': alpha,
                             'state_identity': identity, 'roles': {},
                             'costs': {'restore_s': restore_elapsed, 'construction_and_gradient_s': construction_s,
                                       'state_identity_s': identity_s}}
                    if candidate != 'K_ONLY':
                        entry['direction'] = direction_evidence(parameters, post_k, fitted.gradients, info, alpha)
                        entry['fitted_role'] = 'D' if candidate == 'D_FULL' else 'R'
                    scoring_started = time.perf_counter()
                    for role, scorer in scorers.items():
                        entry['roles'][role] = score_pair(scorer)
                    _sync(parameters)
                    entry['costs']['scoring_s'] = time.perf_counter() - scoring_started
                    if candidate != 'K_ONLY':
                        ev = entry['direction']
                        entry['prediction'] = fitted_objective_predictions(
                            float(fitted.value), alpha, info['status'], ev['proposed_dot'], ev['realized_dot'],
                            entry['roles'][entry['fitted_role']]['V'])
                    self.diagnostic['candidates'][candidate] = entry
                    self._save()
                except ValueError as error:
                    # Only a candidate numerical failure is recoverable here.
                    # Conditioning/state mismatches and the final witness remain
                    # fatal; there is no retry or replacement candidate.
                    if 'Nonfinite' not in str(error) or candidate == 'K_ONLY':
                        raise
                    continuation.restore()
                    checks = continuation.check()
                    self.diagnostic['candidates'][candidate] = {
                        'status': 'invalid', 'error': str(error),
                        'restoration_after_failure': checks, 'roles': {},
                    }
                    self._save()
            validation_started = time.perf_counter()
            self._validate_witness()
            self._summarize(scorers)
            self.diagnostic['costs']['validation_and_block_metrics_s'] = time.perf_counter() - validation_started
            self.diagnostic['status'] = 'completed'
        except BaseException as error:
            self.diagnostic['status'] = 'failed'
            self.diagnostic['error'] = {'type': type(error).__name__, 'message': str(error)}
            raise
        finally:
            self.diagnostic['costs']['candidate_sequence_s'] = time.perf_counter() - started
            restore_started = time.perf_counter()
            continuation.restore()
            self.diagnostic['continuation_integrity'] = continuation.check()
            self.diagnostic['restoration_passed'] = True
            self.diagnostic['costs']['final_restore_and_check_s'] = time.perf_counter() - restore_started
            self.reference.phase = phase
            self.diagnostic['work'] = {
                'pre_capture_response_forwards': sum(s.work['original_response_forwards'] for s in self.extra.values()),
                'candidate_response_forwards': sum(s.work['post_c_response_forwards'] for s in scorers.values()),
                'd_gradient_response_forwards': self.extra['D'].work['post_k_response_forwards'],
                'd_gradient_backwards': self.extra['D'].work['violation_backward_completed'],
                'candidate_response_attempted': sum(s.work['post_c_response_attempted'] for s in scorers.values()),
                'd_gradient_response_attempted': self.extra['D'].work['post_k_response_attempted'],
                'd_gradient_backwards_attempted': self.extra['D'].work['violation_backward_attempted'],
                'candidate_response_unattempted': 24 - sum(s.work['post_c_response_attempted'] for s in scorers.values()),
                'd_gradient_response_unattempted': 2 - self.extra['D'].work['post_k_response_attempted'],
                'projection_chunks_and_recomputation': {role: dict(s.work) for role, s in scorers.items()},
            }
            self._save()
            # Shared R response records belong to the original one-shot controller.
            r_scorer.responses = []
            r_scorer.close()
        return correction

    def _validate_witness(self):
        original = self.reference.post_k
        witness = self.diagnostic['candidates']['K_ONLY']['roles']['R']
        keys = ('V', 'numerator', 'denominator', 'responses')
        if any(not _equal(original[key], witness[key]) for key in keys):
            raise ValueError('Final physical K_ONLY replay differs from original post-K R evidence')
        self.diagnostic['k_only_witness_passed'] = True

    def _summarize(self, scorers):
        baseline = self.diagnostic['candidates']['K_ONLY']
        identities = {}
        blocks = []
        for candidate, entry in self.diagnostic['candidates'].items():
            if entry['status'] != 'valid':
                continue
            identity = entry['state_identity']
            entry['aliases'] = [name for name, other in self.diagnostic['candidates'].items()
                                if name != candidate and other.get('state_identity') == identity]
            for role, scorer in scorers.items():
                rows = []
                for response, ordinary, current in zip(scorer.responses, baseline['roles'][role]['responses'], entry['roles'][role]['responses']):
                    result = response_metrics(
                        response, torch.tensor(ordinary['proposals']), torch.tensor(current['proposals']),
                        torch.tensor(ordinary['all_margins']), torch.tensor(current['all_margins']),
                        eos_token_ids=scorer.model.eos_token_ids,
                        identity={'event': self.plan.get('event_number'), 'step': self.plan['step'],
                                  'pass': self.plan.get('pass_index'), 'candidate': candidate, 'role': role,
                                  'event_local_step': self.plan['step'],
                                  'global_step_after_native_Adam': self.trainer.global_step + 1})
                    rows.append(result)
                    blocks.extend(result['blocks'])
                entry['roles'][role]['metrics'] = pooled_summary(rows)
        self.diagnostic['block_rows'] = len(blocks)
        path = self.output_dir / f'window_{self.plan["step"]:03d}_BLOCKS.jsonl'
        import json
        with path.open('w') as stream:
            for row in blocks:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
        self.diagnostic['blocks_path'] = str(path)

    def _save(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        write_json(self.output_dir / f'window_{self.plan["step"]:03d}.json', self.diagnostic)

    def close(self):
        for recorder in self.extra.values():
            recorder.close()
        super().close()


class MechanismDiagnosticMixin:
    """Add sampled instrumentation without altering KRC's retained contract."""
    def __init__(self, trainer, *, window_plan, mechanism_dir, **kwargs):
        self.window_plan = {row['step']: row for row in window_plan}
        self.mechanism_dir = Path(mechanism_dir)
        self.mechanism_rows = []
        super().__init__(trainer, **kwargs)

    def _begin_window(self):
        step = self.trainer.global_step - self.start_step + 1
        if step not in self.window_plan:
            return super()._begin_window()
        self.window_step = step
        self.window_started = time.perf_counter()
        self.window_integrity_s, self.window_integrity = 0., {}
        window = self.schedule['windows'][step - 1]
        self.active = MechanismWindow(self.trainer, window, self.feature_paths,
                                      plan=self.window_plan[step], output_dir=self.mechanism_dir)
        self.capture = self.active.capture_k()
        self.capture.__enter__()

    def _checked(self, snapshot, phase, *, parameters_read_only):
        # A restored no-op has identical values but copy_ necessarily advances
        # PyTorch version counters. The sampled controller checks exact values.
        if phase == 'correction' and isinstance(self.active, MechanismWindow):
            if not self.active.diagnostic.get('restoration_passed'):
                raise ValueError('Candidate sequence has no exact restoration witness')
            parameters_read_only = False
        return super()._checked(snapshot, phase, parameters_read_only=parameters_read_only)

    def _record_window(self, status, error=None):
        if isinstance(self.active, MechanismWindow):
            self.mechanism_rows.append(self.active.diagnostic)
        return super()._record_window(status, error)

    def report(self):
        result = super().report()
        work = {key: sum(row.get('work', {}).get(key, 0) for row in self.mechanism_rows)
                for key in ('pre_capture_response_forwards', 'candidate_response_forwards',
                            'd_gradient_response_forwards', 'd_gradient_backwards')}
        result['mechanism'] = {'rows': self.mechanism_rows, 'work': work,
                               'candidate_order': list(CANDIDATES)}
        return result
