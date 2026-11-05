"""Protect current matching prefixes after the unchanged native K/AdamW step.

The guard owns a separate FP32 gradient buffer. It never supplies a gradient to
Adam, and the optimizer's moments are not projected. Native BF16 rounding is
measured rather than repaired by backtracking or rollback.
"""
from __future__ import annotations

from contextlib import contextmanager
import math
from pathlib import Path
import time

import torch

from .first_error import valid_prefix_mask
from .ranking import fixed_anchors, matching_prefix_mask, rival_softplus
from .rng import preserve_rng
from .storage import write_json


@torch.no_grad()
def preservation_weights(proposals, labels, validity_weights, eos_token_ids=()):
    """Give a strict matching prefix of length ell weights ell,...,1."""
    prefix = matching_prefix_mask(proposals, labels, validity_weights, eos_token_ids)
    lengths = prefix.sum(-1, keepdim=True)
    offsets = torch.arange(labels.shape[-1], device=labels.device)
    return ((lengths - offsets + 1).clamp_min(0) * prefix).float()


def preservation_terms(logits, labels, weights):
    """Additive FP32 terms; normalize once after all response chunks."""
    penalties = rival_softplus(logits, labels)
    return (penalties * weights).sum(), weights.sum()


def normalized_preservation(numerator, weight_sum):
    # clamp avoids 0/0 while retaining an exactly zero empty-set gradient.
    return numerator / (math.log(2.0) * weight_sum.clamp_min(1.0))


@torch.no_grad()
def prefix_changes(before, after, labels, validity_weights, eos_token_ids=()):
    """Useful repairs require that the preceding canonical prefix stays intact."""
    valid = valid_prefix_mask(labels, validity_weights, eos_token_ids)
    old = matching_prefix_mask(before, labels, validity_weights, eos_token_ids)
    new = matching_prefix_mask(after, labels, validity_weights, eos_token_ids)
    old_length, new_length = old.sum(-1), new.sum(-1)
    available = valid.sum(-1)
    has_old_error = old_length < available
    return {
        'blocks': old_length.numel(),
        'old_matching_prefix_tokens': int(old_length.sum()),
        'new_matching_prefix_tokens': int(new_length.sum()),
        'net_matching_prefix_change': int((new_length - old_length).sum()),
        'old_first_errors': int(has_old_error.sum()),
        'first_errors_repaired_with_prefix_intact': int((has_old_error & (new_length > old_length)).sum()),
        'new_earlier_errors': int((new_length < old_length).sum()),
        'old_prefix_tokens_lost': int((old_length - new_length).clamp_min(0).sum()),
    }


def _sync(parameters):
    if parameters[0].is_cuda:
        torch.cuda.synchronize(parameters[0].device)


@torch.no_grad()
def displacement_statistics(parameters, before, h):
    """FP32 displacements, with FP64 scalar reductions across all parameters."""
    device = parameters[0].device
    totals = torch.zeros(3, dtype=torch.float64, device=device)
    finite = torch.ones((), dtype=torch.bool, device=device)
    for parameter, previous, gradient in zip(parameters, before, h):
        delta = parameter.float() - previous.float()
        finite &= torch.isfinite(delta).all() & torch.isfinite(gradient).all()
        totals[0] += (delta.double() * gradient.double()).sum()
        totals[1] += gradient.double().square().sum()
        totals[2] += delta.double().square().sum()
    if not bool(finite) or not bool(torch.isfinite(totals).all()):
        raise ValueError('Nonfinite prefix-guard displacement or H gradient')
    dot, h_squared, delta_squared = totals.tolist()
    return dot, h_squared, delta_squared


@torch.no_grad()
def project_native_displacement(parameters, before, h, *, enabled=True):
    """Apply one global half-space projection; leave inactive weights untouched."""
    dot, h_squared, native_squared = displacement_statistics(parameters, before, h)
    active = bool(enabled and h_squared > 0 and dot > 0)
    coefficient = dot / h_squared if active else 0.0
    ideal_squared = 0.0
    ideal_dot = 0.0
    realized_correction_squared = 0.0
    if active:
        for parameter, previous, gradient in zip(parameters, before, h):
            delta = parameter.float() - previous.float()
            projected = delta - coefficient * gradient
            ideal_squared += float(projected.double().square().sum())
            ideal_dot += float((projected.double() * gradient.double()).sum())
            corrected = (previous.float() + projected).to(parameter.dtype)
            realized_correction_squared += float((corrected.float() - parameter.float()).double().square().sum())
            parameter.copy_(corrected)
    else:
        ideal_squared = native_squared
        ideal_dot = dot
    realized_dot, _, realized_squared = displacement_statistics(parameters, before, h)
    return {
        'enabled': bool(enabled), 'active': active, 'projection_coefficient': coefficient,
        'h_norm': math.sqrt(h_squared), 'native_directional_derivative': dot,
        'ideal_directional_derivative': ideal_dot,
        'analytic_projected_directional_derivative': dot - coefficient * h_squared,
        'realized_directional_derivative': realized_dot,
        'native_update_norm': math.sqrt(native_squared),
        'projected_fp32_update_norm': math.sqrt(ideal_squared),
        'realized_update_norm': math.sqrt(realized_squared),
        'ideal_correction_norm': coefficient * math.sqrt(h_squared),
        'realized_correction_norm': math.sqrt(realized_correction_squared),
        'rounding_violates_halfspace': realized_dot > 0,
        'scalar_reduction_precision': 'float64',
    }


@contextmanager
def preserved_probe_runtime(model):
    """Restore training modes, RNG and the author's reusable mask cache exactly."""
    modes = [(module, module.training) for module in model.modules()]
    cache_names = ('_cached_block_mask', '_cached_seq_len', '_cached_bsz')
    caches = {name: getattr(model, name) for name in cache_names if hasattr(model, name)}
    with preserve_rng():
        try:
            model.eval()
            yield
        finally:
            for module, training in modes:
                module.training = training
            for name, value in caches.items():
                setattr(model, name, value)


class PrefixGuard:
    """A small optional session around the existing K trainer.

    ``observer(phase, trainer, guard)`` is for bounded qualification only. Normal
    training uses compact scalar records, not complete gradient snapshots.
    """

    def __init__(self, trainer, *, enabled, probe_steps, output_dir, observer=None):
        self.trainer = trainer
        self.model = trainer.bundle.model
        self.parameters = tuple(trainer.backend.optimizer.model_params)
        self.enabled = bool(enabled)
        self.closed = False
        self.probe_steps = frozenset(probe_steps)
        self.output_dir = Path(output_dir) if output_dir is not None else None
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.observer = observer
        self.h = []
        self.pending_h = None
        self.collecting = False
        self.current = None
        self.chunk_weights = {}
        self.window = []
        self.response_records = []
        self.step_records = []
        self.probe_records = []
        self.phase = 'current_forward'
        self.forward_calls = {}
        self.hooks = []
        for name, module in [('draft', self.model.draft_model), ('vocabulary_head', self.model.lm_head),
                             *((f'draft_layer_{index}', layer) for index, layer in
                               enumerate(getattr(self.model.draft_model, 'layers', ())))]:
            def count_forward(module, args, name=name):
                key = f'{self.phase}/{name}'
                self.forward_calls[key] = self.forward_calls.get(key, 0) + 1
            self.hooks.append(module.register_forward_pre_hook(count_forward))
        self.work = dict(current_response_forwards=0, current_objective_chunks=0,
                         preservation_chunk_evaluations=0,
                         h_backward_calls=0, k_backward_calls=0,
                         h_backward_attempted=0, h_backward_completed=0, h_backward_failed=0,
                         k_backward_attempted=0, k_backward_completed=0, k_backward_failed=0,
                         probe_response_forwards=0, probe_head_chunks=0)
        self.costs = dict(h_terms_s=0.0, h_backward_s=0.0, guard_projection_s=0.0, probes_s=0.0)

    @property
    def next_step(self):
        return self.trainer.global_step + 1

    def observe(self, phase):
        if self.observer is not None:
            self.observer(phase, self.trainer, self)

    def begin_response(self, input_ids, hidden_states, loss_mask, anchors, keep, labels, weights):
        self.work['current_response_forwards'] += 1
        self.phase = 'current_forward'
        self.collecting = True
        self.current = {'chunks': []}
        self.chunk_weights = {}
        if self.next_step in self.probe_steps:
            # Keep only the two existing response tensors for this probe window.
            self.current['probe'] = {
                key: value.detach().cpu().clone() for key, value in dict(
                    input_ids=input_ids, hidden_states=hidden_states, loss_mask=loss_mask,
                    anchors=anchors, keep=keep, labels=labels, validity_weights=weights,
                ).items()
            }

    def chunk(self, logits, labels, validity_weights, chunk_key):
        self.work['preservation_chunk_evaluations'] += 1
        _sync(self.parameters)
        started = time.perf_counter()
        if self.collecting:
            proposals = logits.detach().argmax(-1)
            weights = preservation_weights(proposals, labels, validity_weights, self.model.eos_token_ids)
            self.chunk_weights[chunk_key] = weights
        else:
            # Checkpoint recomputation uses the original detached membership.
            weights = self.chunk_weights[chunk_key]
        if self.collecting:
            summary = {'protected_positions': int((weights > 0).sum())}
            if 'probe' in self.current:
                summary.update(weights=weights.cpu(), proposals=proposals.cpu())
            self.current['chunks'].append(summary)
        terms = preservation_terms(logits, labels, weights)
        _sync(self.parameters)
        self.costs['h_terms_s'] += time.perf_counter() - started
        return terms

    def finish_response(self, numerator, weight_sum):
        self.collecting = False
        self.pending_h = normalized_preservation(numerator, weight_sum)
        if not bool(torch.isfinite(self.pending_h)):
            if self.output_dir is not None:
                write_json(self.output_dir / 'NONFINITE_H.json', {
                    'optimizer_step': self.next_step, 'presentation': len(self.response_records) + 1,
                    'H': str(float(self.pending_h.detach())), 'numerator': str(float(numerator.detach())),
                    'weight_sum': str(float(weight_sum)),
                })
            raise ValueError('Nonfinite current-prefix preservation quantity')
        row = dict(presentation=len(self.response_records) + 1, optimizer_step=self.next_step,
                   h_before=float(self.pending_h.detach()), h_numerator=float(numerator.detach()),
                   h_weight_sum=float(weight_sum),
                   protected_positions=sum(chunk['protected_positions'] for chunk in self.current['chunks']))
        self.response_records.append(row)
        if 'probe' in self.current:
            probe = self.current['probe']
            probe['protection_weights'] = torch.cat([chunk['weights'] for chunk in self.current['chunks']], 1)
            probe['before_proposals'] = torch.cat([chunk['proposals'] for chunk in self.current['chunks']], 1)
            probe['h_before'] = row['h_before']
            self.window.append(probe)
        self.current = None
        return self.pending_h.detach()

    def _call_backward(self, kind, function, *args, **kwargs):
        """Count attempted work even when the underlying backward raises."""
        self.work[f'{kind}_backward_attempted'] += 1
        try:
            result = function(*args, **kwargs)
        except BaseException:
            self.work[f'{kind}_backward_failed'] += 1
            raise
        self.work[f'{kind}_backward_completed'] += 1
        # Retain the original successful-call field for existing budget readers.
        self.work[f'{kind}_backward_calls'] += 1
        return result

    def backward(self, original_backward, loss, *, is_boundary=True):
        if self.enabled:
            _sync(self.parameters)
            started = time.perf_counter()
            self.phase = 'h_backward'
            try:
                gradients = self._call_backward(
                    'h', torch.autograd.grad, self.pending_h, self.parameters,
                    retain_graph=True, allow_unused=True,
                )
                finite = [torch.isfinite(gradient).all() for gradient in gradients if gradient is not None]
                if finite and not bool(torch.stack(finite).all()):
                    failures = []
                    names = [name for name, parameter in self.model.draft_model.named_parameters()
                             if parameter.requires_grad]
                    for index, gradient in enumerate(gradients):
                        if gradient is not None and not bool(torch.isfinite(gradient).all()):
                            failures.append({
                                'parameter_index': index, 'parameter_name': names[index],
                                'shape': list(gradient.shape), 'dtype': str(gradient.dtype),
                                'nonfinite_count': int((~torch.isfinite(gradient)).sum()),
                            })
                    if self.output_dir is not None:
                        write_json(self.output_dir / 'NONFINITE_H_GRADIENT.json', {
                            'optimizer_step': self.next_step, 'presentation': len(self.response_records),
                            'parameters': failures,
                        })
                    raise ValueError('Nonfinite H gradient before K backward or optimizer step')
                if not self.h:
                    self.h = [torch.zeros_like(p, dtype=torch.float32) for p in self.parameters]
                for total, gradient in zip(self.h, gradients):
                    if gradient is not None:
                        total.add_(gradient.detach().float(), alpha=1.0 / self.trainer.accumulation)
                self.pending_h = None
                del gradients
                _sync(self.parameters)
            finally:
                self.costs['h_backward_s'] += time.perf_counter() - started
        self.phase = 'k_backward'
        self.observe('before_k_backward')
        self._call_backward('k', original_backward, loss, is_boundary=is_boundary)
        self.observe('after_k_backward')
        self.chunk_weights.clear()
        self.phase = 'current_forward'

    def step(self, original_step):
        self.observe('before_native_step')
        before = [parameter.detach().clone() for parameter in self.parameters] if self.enabled else None
        grad_norm = original_step()
        self.observe('after_native_step')
        if self.enabled:
            native = ([parameter.detach().clone() for parameter in self.parameters]
                      if self.next_step in self.probe_steps else None)
            _sync(self.parameters)
            started = time.perf_counter()
            row = project_native_displacement(self.parameters, before, self.h)
            _sync(self.parameters)
            self.costs['guard_projection_s'] += time.perf_counter() - started
            row['optimizer_step'] = self.next_step
            row['h_before_response_mean'] = sum(
                item['h_before'] for item in self.response_records[-self.trainer.accumulation:]
            ) / self.trainer.accumulation
            self.step_records.append(row)
            if native is not None:
                self.run_probes(before, native, row)
            self.h.clear()
            self.window.clear()
            if self.output_dir is not None:
                write_json(self.output_dir / f'step_{self.next_step:03d}.json', row)
        self.observe('after_guard_step')
        return grad_norm

    @torch.no_grad()
    def evaluate_probe_response(self, probe):
        device = self.parameters[0].device
        tensors = {name: probe[name].to(device) for name in
                   ('input_ids', 'hidden_states', 'loss_mask', 'anchors', 'keep', 'labels',
                    'validity_weights', 'protection_weights')}
        with fixed_anchors(self.model, tensors['anchors'], tensors['keep']):
            hidden = self.model._forward_draft_blocks(
                tensors['input_ids'], tensors['hidden_states'], tensors['loss_mask'],
            )[2]
        self.work['probe_response_forwards'] += 1
        hidden = hidden.reshape(1, tensors['anchors'].shape[1], self.model.block_size, -1)
        numerator = torch.zeros((), device=device)
        weight_sum = tensors['protection_weights'].sum()
        predictions = []
        for start in range(0, hidden.shape[1], self.model.objective_chunk_blocks):
            stop = start + self.model.objective_chunk_blocks
            logits = self.model.lm_head(hidden[:, start:stop])
            self.work['probe_head_chunks'] += 1
            if not bool(torch.isfinite(logits).all()):
                if self.output_dir is not None:
                    write_json(self.output_dir / 'NONFINITE_PROBE.json', {
                        'optimizer_step': self.next_step, 'candidate': self.probe_candidate,
                        'chunk_start': start, 'nan_count': int(torch.isnan(logits).sum()),
                        'inf_count': int(torch.isinf(logits).sum()),
                    })
                raise ValueError('Nonfinite draft logits in prefix-guard probe')
            terms = preservation_terms(logits, tensors['labels'][:, start:stop],
                                       tensors['protection_weights'][:, start:stop])
            numerator += terms[0]
            predictions.append(logits.argmax(-1).cpu())
        if not bool(torch.isfinite(numerator)):
            raise ValueError('Nonfinite prefix-guard probe H')
        return {
            'h_before': probe['h_before'],
            'h_after': float(normalized_preservation(numerator, weight_sum)),
            **prefix_changes(probe['before_proposals'], torch.cat(predictions, 1),
                             probe['labels'], probe['validity_weights'], self.model.eos_token_ids),
        }

    @torch.no_grad()
    def run_probes(self, before, native, step_record):
        final = [parameter.detach().clone() for parameter in self.parameters]
        native_norm = step_record['native_update_norm']
        scale = step_record['projected_fp32_update_norm'] / native_norm if native_norm else 1.0
        _sync(self.parameters)
        started = time.perf_counter()
        try:
            with preserved_probe_runtime(self.model):
                self.phase = 'probe'
                for candidate in ('ordinary_K', 'projected_G', 'norm_scaled_K'):
                    self.probe_candidate = candidate
                    for parameter, old, ordinary, guarded in zip(self.parameters, before, native, final):
                        if candidate == 'ordinary_K':
                            parameter.copy_(ordinary)
                        elif candidate == 'projected_G':
                            parameter.copy_(guarded)
                        else:
                            delta = ordinary.float() - old.float()
                            parameter.copy_((old.float() + scale * delta).to(parameter.dtype))
                    dot, _, squared_norm = displacement_statistics(self.parameters, before, self.h)
                    record = {
                        'optimizer_step': self.next_step, 'candidate': candidate,
                        'native_update_norm': native_norm,
                        'projected_fp32_update_norm': step_record['projected_fp32_update_norm'],
                        'realized_update_norm': math.sqrt(squared_norm),
                        'realized_directional_derivative': dot,
                        'uniform_scale': scale if candidate == 'norm_scaled_K' else None,
                        'responses': [self.evaluate_probe_response(probe) for probe in self.window],
                    }
                    self.probe_records.append(record)
                    if self.output_dir is not None:
                        write_json(self.output_dir / f'probe_{self.next_step:03d}_{candidate}.json', record)
        finally:
            for parameter, saved in zip(self.parameters, final):
                parameter.copy_(saved)
        _sync(self.parameters)
        self.costs['probes_s'] += time.perf_counter() - started
        self.phase = 'current_forward'

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        """Detach hooks and release GPU owners; retain only compact evidence."""
        if self.closed:
            return
        self.trainer.backend.backward = self.original_backward
        self.trainer.backend.optimizer.step = self.original_step
        if getattr(self.model, 'prefix_guard', None) is self:
            del self.model.prefix_guard
        if getattr(self.trainer, 'prefix_guard', None) is self:
            del self.trainer.prefix_guard
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.pending_h = None
        self.h.clear()
        self.window.clear()
        self.chunk_weights.clear()
        self.current = None
        self.parameters = ()
        self.model = self.trainer = None
        self.original_backward = self.original_step = None
        self.observer = None
        self.closed = True

    def summary(self):
        return self.report()

    def report(self):
        return dict(enabled=self.enabled, work=dict(self.work), forward_calls=dict(self.forward_calls),
                    costs=dict(self.costs),
                    responses=list(self.response_records), steps=list(self.step_records),
                    probes=list(self.probe_records))


def install_guard(trainer, *, enabled=True, probe_steps=(1, 64, 128), output_dir=None, observer=None):
    """Attach G to an already constructed fresh K trainer without replacing Adam."""
    if trainer.objective != 'osd_first_error' or trainer.reduction != 'response_mean':
        raise ValueError('Prefix guard requires the qualified K response-mean objective')
    guard = PrefixGuard(trainer, enabled=enabled, probe_steps=probe_steps,
                        output_dir=output_dir, observer=observer)
    trainer.prefix_guard = guard
    if enabled:
        trainer.bundle.model.prefix_guard = guard
    original_backward = trainer.backend.backward
    original_step = trainer.backend.optimizer.step
    guard.original_backward = original_backward
    guard.original_step = original_step
    trainer.backend.backward = lambda loss, is_boundary=True: guard.backward(
        original_backward, loss, is_boundary=is_boundary)
    trainer.backend.optimizer.step = lambda: guard.step(original_step)
    trainer.contract['prefix_preservation_guard'] = {
        'version': 1, 'enabled': bool(enabled), 'prefix': 'current_strict_canonical_matching',
        'weights': 'descending_prefix_length', 'normalization': 'response_sum_div_log2_weight_sum',
        'gradient_buffer': 'float32', 'projection_scalar_reduction': 'float64',
        'projection': 'global_native_adamw_displacement_halfspace', 'parameter_dtype': 'bfloat16',
        'optimizer_state': 'unchanged_K', 'probe_steps': list(probe_steps),
        'rounding_repair': None,
    }
    return guard
