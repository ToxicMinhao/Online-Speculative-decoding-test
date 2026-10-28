"""One post-K correction of negative margins on the original matching prefixes.

K's loss, backward and optimizer stay unchanged. The two saved training blocks
are replayed after that step; only their original correct-prefix positions enter
one globally normalized hinge. This diagnostic never publishes a checkpoint.
"""
from __future__ import annotations

from contextlib import contextmanager
import math
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from specforge.core.chunking import checkpointed_chunk_reduce

from .first_error import valid_prefix_mask
from .prefix_guard import preservation_weights, prefix_changes, preserved_probe_runtime
from .ranking import fixed_anchors, matching_prefix_mask
from .storage import write_json


def canonical_margin(logits, labels):
    """Native deterministic rival selection, differentiable FP32 subtraction."""
    with torch.no_grad():
        competitors = logits.detach().clone()
        competitors.scatter_(-1, labels.unsqueeze(-1), -torch.inf)
        rivals = competitors.argmax(-1)
    canonical = logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1).float()
    competitor = logits.gather(-1, rivals.unsqueeze(-1)).squeeze(-1).float()
    return canonical - competitor


def violation_terms(logits, labels, protected_weights):
    """One additive numerator; the caller owns the shared two-response divisor."""
    margin = canonical_margin(logits, labels)
    return (protected_weights * F.relu(-margin)).sum(), margin


@torch.no_grad()
def repair_transitions(original, post_k, post_c, labels, validity, eos_token_ids=()):
    """Match repair identities per block, retaining the native EOS/tie rules."""
    lengths = [matching_prefix_mask(predictions, labels, validity, eos_token_ids).sum(-1)
               for predictions in (original, post_k, post_c)]
    before, ordinary, corrected = lengths
    available = valid_prefix_mask(labels, validity, eos_token_ids).sum(-1)
    has_error = before < available
    repaired_k = has_error & (ordinary > before)
    repaired_c = has_error & (corrected > before)
    return {
        'original_first_errors': int(has_error.sum()),
        'K_useful_repairs': int(repaired_k.sum()),
        'C_useful_repairs': int(repaired_c.sum()),
        'useful_repairs_retained': int((repaired_k & repaired_c).sum()),
        'useful_repairs_lost': int((repaired_k & ~repaired_c).sum()),
        'useful_repairs_newly_obtained': int((~repaired_k & repaired_c).sum()),
    }


def _sync(parameters):
    if parameters[0].is_cuda:
        torch.cuda.synchronize(parameters[0].device)


@torch.no_grad()
def correction_direction(value, gradients):
    """Return the single FP32 coefficient; zero norm has no epsilon fallback."""
    squared_norm = torch.zeros((), dtype=torch.float32, device=value.device)
    finite = torch.isfinite(value)
    for gradient in gradients:
        finite &= torch.isfinite(gradient).all()
        squared_norm.add_(gradient.square().sum())
    if not bool(finite & torch.isfinite(squared_norm)):
        raise ValueError('Nonfinite violation value or correction gradient norm')
    if float(value) == 0:
        return 'zero_violation', squared_norm, None
    if float(squared_norm) == 0:
        return 'unresolved_zero_gradient', squared_norm, None
    coefficient = -value / squared_norm
    if not bool(torch.isfinite(coefficient)):
        raise ValueError('Nonfinite correction coefficient')
    return 'applied', squared_norm, coefficient


@torch.no_grad()
def apply_correction(parameters, gradients, value, *, protected_weight_sum, k_update_norm):
    """One FP32 displacement followed by BF16 rounding, with unchanged Adam."""
    if protected_weight_sum == 0:
        status, squared_norm, coefficient = 'empty_protection', value.new_zeros(()), None
    else:
        status, squared_norm, coefficient = correction_direction(value, gradients)
    result = {
        'status': status, 'V': float(value), 'global_protected_weight': float(protected_weight_sum),
        'gradient_norm_squared_fp32': float(squared_norm),
        'coefficient_fp32': float(coefficient) if coefficient is not None else None,
        'proposed_correction_norm': 0.0, 'realized_correction_norm': 0.0,
        'K_update_norm': k_update_norm,
        'norm_reporting_precision': 'float64; correction decisions and vectors are float32',
    }
    proposed_squared = value.new_zeros((), dtype=torch.float64)
    realized_squared = value.new_zeros((), dtype=torch.float64)
    if coefficient is not None:
        # Reject overflow before changing any parameter. This is a finite check,
        # not a second correction, line search or clipping operation.
        finite = torch.ones((), device=value.device, dtype=torch.bool)
        for parameter, gradient in zip(parameters, gradients):
            delta = coefficient * gradient
            candidate = (parameter.float() + delta).to(parameter.dtype)
            finite &= torch.isfinite(delta).all() & torch.isfinite(candidate).all()
        if not bool(finite):
            raise ValueError('Nonfinite proposed or BF16-rounded correction')
        for parameter, gradient in zip(parameters, gradients):
            before = parameter.float()
            delta = coefficient * gradient
            corrected = (before + delta).to(parameter.dtype)
            proposed_squared += delta.double().square().sum()
            realized_squared += (corrected.float() - before).double().square().sum()
            parameter.copy_(corrected)
        result['proposed_correction_norm'] = math.sqrt(float(proposed_squared))
        result['realized_correction_norm'] = math.sqrt(float(realized_squared))
    for kind in ('proposed', 'realized'):
        result[f'{kind}_correction_to_K_norm_ratio'] = (
            result[f'{kind}_correction_norm'] / k_update_norm if k_update_norm else None
        )
    return result


class PrefixCorrection:
    """Record K's first window, then measure/apply/recheck one correction."""

    def __init__(self, trainer, *, output_dir=None, record_ids=()):
        if trainer.objective != 'osd_first_error' or trainer.accumulation != 2:
            raise ValueError('This diagnostic requires the qualified two-response K window')
        self.trainer = trainer
        self.model = trainer.bundle.model
        self.parameters = tuple(trainer.backend.optimizer.model_params)
        self.output_dir = Path(output_dir) if output_dir is not None else None
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.record_ids = tuple(record_ids)
        self.responses = []
        self.current = None
        self.collecting = False
        self.phase = 'K'
        self.gradients = []
        self.global_denominator = None
        self.value = None
        self.post_k = None
        self.post_c = None
        self.correction = None
        self.k_update_norm = None
        self.closed = False
        self.work = dict(original_response_forwards=0, current_objective_chunks=0,
                         post_k_response_forwards=0, post_c_response_forwards=0,
                         post_k_response_attempted=0, post_c_response_attempted=0,
                         violation_backward_attempted=0, violation_backward_completed=0,
                         violation_backward_failed=0, direct_corrections=0,
                         post_k_objective_chunks=0, post_c_objective_chunks=0)
        self.costs = dict(prefix_chunk_capture_s=0.0, post_k_s=0.0, correction_s=0.0, post_c_s=0.0)
        self.forward_calls = {}
        self.hooks = []
        modules = [('draft', self.model.draft_model), ('vocabulary_head', self.model.lm_head)]
        modules += [(f'draft_layer_{index}', layer)
                    for index, layer in enumerate(getattr(self.model.draft_model, 'layers', ()))]
        for name, module in modules:
            def count(module, inputs, name=name):
                key = f'{self.phase}/{name}'
                self.forward_calls[key] = self.forward_calls.get(key, 0) + 1
            self.hooks.append(module.register_forward_pre_hook(count))

    @contextmanager
    def capture_k(self):
        """Observe existing K logits; no extra gradient or model forward."""
        if hasattr(self.model, 'prefix_guard') or self.responses:
            raise ValueError('Use one fresh, unguarded K window for this diagnostic')
        before = [parameter.detach().clone() for parameter in self.parameters]
        self.model.prefix_guard = self
        try:
            yield self
            squared = torch.zeros((), dtype=torch.float64, device=self.parameters[0].device)
            for parameter, previous in zip(self.parameters, before):
                squared += (parameter.detach().float() - previous.float()).double().square().sum()
            self.k_update_norm = math.sqrt(float(squared))
        finally:
            del self.model.prefix_guard
            del before

    def begin_response(self, input_ids, hidden_states, loss_mask, anchors, keep, labels, weights):
        self.work['original_response_forwards'] += 1
        self.collecting = True
        self.current = {key: value.detach().cpu().clone() for key, value in dict(
            input_ids=input_ids, hidden_states=hidden_states, loss_mask=loss_mask,
            anchors=anchors, keep=keep, labels=labels, validity_weights=weights,
        ).items()}
        self.current['chunks'] = []

    @torch.no_grad()
    def chunk(self, logits, labels, validity_weights, chunk_key):
        # The existing objective calls this again during K recomputation. Only
        # the original forward is recorded; detached zeros never enter K loss.
        if self.collecting:
            started = time.perf_counter()
            if not bool(torch.isfinite(logits).all()):
                raise ValueError('Nonfinite original K logits')
            proposals = logits.detach().argmax(-1)
            weights = preservation_weights(proposals, labels, validity_weights, self.model.eos_token_ids)
            self.current['chunks'].append((proposals.cpu(), weights.cpu()))
            self.costs['prefix_chunk_capture_s'] += time.perf_counter() - started
        zero = logits.new_zeros((), dtype=torch.float32)
        return zero, zero

    def finish_response(self, numerator, denominator):
        self.collecting = False
        current = self.current
        chunks = current.pop('chunks')
        current['original_proposals'] = torch.cat([chunk[0] for chunk in chunks], 1)
        current['protected_weights'] = torch.cat([chunk[1] for chunk in chunks], 1)
        ordinal = len(self.responses)
        current['record_id'] = self.record_ids[ordinal] if self.record_ids else str(ordinal)
        self.responses.append(current)
        self.current = None
        return numerator.detach()

    def _objective_chunk(self, hidden, labels, protected_weights):
        self.work[f'{self.phase}_objective_chunks'] += 1
        logits = self.model.lm_head(hidden)
        if not bool(torch.isfinite(logits).all()):
            raise ValueError(f'Nonfinite {self.phase} logits')
        numerator, margins = violation_terms(logits, labels, protected_weights)
        if not bool(torch.isfinite(margins).all() & torch.isfinite(numerator)):
            self._save('NONFINITE_VIOLATION.json', {
                'phase': self.phase, 'numerator': str(float(numerator.detach())),
                'nonfinite_margin_count': int((~torch.isfinite(margins)).sum()),
            })
            raise ValueError(f'Nonfinite {self.phase} margin or hinge numerator')
        if self.collecting:
            self.current_predictions.append(logits.detach().argmax(-1).cpu())
            self.current_margins.append(margins.detach().cpu())
        return (numerator,)

    def _score_response(self, response, predictions, margins):
        protected = response['protected_weights'] > 0
        wrong = predictions != response['labels']
        strict = protected & (margins < 0)
        wrong_ties = protected & wrong & (margins == 0)
        result = {
            'record_id': response['record_id'],
            **prefix_changes(response['original_proposals'], predictions, response['labels'],
                             response['validity_weights'], self.model.eos_token_ids),
            'protected_positions': int(protected.sum()),
            'protected_weight': float(response['protected_weights'].sum()),
            'strict_margin_violations': int(strict.sum()),
            'weighted_strict_margin_violations': float((strict * response['protected_weights']).sum()),
            'wrong_at_tie_positions': int(wrong_ties.sum()),
            'weighted_wrong_at_tie_positions': float((wrong_ties * response['protected_weights']).sum()),
            'protected_argmax_errors': int((protected & wrong).sum()),
            'matching_prefix_lengths': matching_prefix_mask(
                predictions, response['labels'], response['validity_weights'],
                self.model.eos_token_ids,
            ).sum(-1).tolist(),
            'proposals': predictions.tolist(),
            'protected_margins': torch.where(protected, margins, 0).tolist(),
        }
        return result

    def _forward_response(self, response):
        device = self.parameters[0].device
        names = ('input_ids', 'hidden_states', 'loss_mask', 'anchors', 'keep', 'labels',
                 'protected_weights')
        tensors = {name: response[name].to(device) for name in names}
        self.work[f'{self.phase}_response_attempted'] += 1
        with fixed_anchors(self.model, tensors['anchors'], tensors['keep']):
            hidden = self.model._forward_draft_blocks(
                tensors['input_ids'], tensors['hidden_states'], tensors['loss_mask'],
            )[2]
        self.work[f'{self.phase}_response_forwards'] += 1
        hidden = hidden.reshape(1, tensors['anchors'].shape[1], self.model.block_size, -1)
        self.current_predictions, self.current_margins = [], []
        self.collecting = True
        try:
            numerator, = checkpointed_chunk_reduce(
                self._objective_chunk, hidden, tensors['labels'], tensors['protected_weights'],
                chunk_size=self.model.objective_chunk_blocks, dim=1, checkpoint=True,
            )
        finally:
            self.collecting = False
        predictions = torch.cat(self.current_predictions, 1)
        margins = torch.cat(self.current_margins, 1)
        row = self._score_response(response, predictions, margins)
        row['violation_numerator'] = float(numerator.detach())
        return numerator, row, predictions

    def _accumulate_gradient(self, value):
        self.work['violation_backward_attempted'] += 1
        try:
            gradients = torch.autograd.grad(value, self.parameters, allow_unused=True)
        except BaseException:
            self.work['violation_backward_failed'] += 1
            raise
        self.work['violation_backward_completed'] += 1
        finite = [torch.isfinite(gradient).all() for gradient in gradients if gradient is not None]
        if finite and not bool(torch.stack(finite).all()):
            raise ValueError('Nonfinite correction gradient')
        for total, gradient in zip(self.gradients, gradients):
            if gradient is not None:
                total.add_(gradient.detach().float())

    @staticmethod
    def _aggregate(rows):
        names = ('blocks', 'old_matching_prefix_tokens', 'new_matching_prefix_tokens',
                 'net_matching_prefix_change', 'old_first_errors',
                 'first_errors_repaired_with_prefix_intact', 'new_earlier_errors',
                 'old_prefix_tokens_lost', 'protected_positions', 'protected_weight',
                 'strict_margin_violations', 'weighted_strict_margin_violations',
                 'wrong_at_tie_positions', 'weighted_wrong_at_tie_positions',
                 'protected_argmax_errors')
        return {name: sum(row[name] for row in rows) for name in names}

    def measure_post_k(self):
        """Two fixed draft replays, with a shared divisor and at most two VJPs."""
        if len(self.responses) != 2 or self.post_k is not None:
            raise ValueError('Measure the single completed two-response K window once')
        device = self.parameters[0].device
        self.global_denominator = sum(
            (response['protected_weights'].sum().to(device) for response in self.responses),
            torch.zeros((), dtype=torch.float32, device=device),
        )
        self.gradients = [torch.zeros_like(parameter, dtype=torch.float32) for parameter in self.parameters]
        total_numerator = torch.zeros((), dtype=torch.float32, device=device)
        rows = []
        _sync(self.parameters)
        started = time.perf_counter()
        try:
            with preserved_probe_runtime(self.model), torch.enable_grad():
                self.phase = 'post_k'
                for response in self.responses:
                    numerator, row, predictions = self._forward_response(response)
                    total_numerator.add_(numerator.detach())
                    response['post_k_proposals'] = predictions
                    rows.append(row)
                    if float(numerator.detach()) > 0:
                        self._accumulate_gradient(numerator / self.global_denominator)
                self.value = (total_numerator / self.global_denominator
                              if float(self.global_denominator) else total_numerator)
                if not bool(torch.isfinite(self.value)):
                    raise ValueError('Nonfinite aggregate violation value')
            _sync(self.parameters)
        finally:
            self.costs['post_k_s'] += time.perf_counter() - started
        self.post_k = dict(V=float(self.value), numerator=float(total_numerator),
                           denominator=float(self.global_denominator), responses=rows,
                           totals=self._aggregate(rows))
        self._save('POST_K.json', self.post_k)
        return self.post_k

    def apply_once(self):
        """Apply only the prescribed scalar/vector, leaving optimizer state alone."""
        if self.post_k is None or self.correction is not None:
            raise ValueError('Complete post-K measurement before the single correction')
        _sync(self.parameters)
        started = time.perf_counter()
        self.correction = apply_correction(
            self.parameters, self.gradients, self.value,
            protected_weight_sum=float(self.global_denominator), k_update_norm=self.k_update_norm,
        )
        self.work['direct_corrections'] += self.correction['status'] == 'applied'
        _sync(self.parameters)
        self.costs['correction_s'] += time.perf_counter() - started
        self._save('CORRECTION.json', self.correction)
        return self.correction

    @torch.no_grad()
    def measure_post_c(self):
        if self.correction is None or self.post_c is not None:
            raise ValueError('Apply or explicitly skip the correction before one recheck')
        rows, transition_rows = [], []
        total_numerator = self.value.new_zeros(())
        _sync(self.parameters)
        started = time.perf_counter()
        try:
            with preserved_probe_runtime(self.model):
                self.phase = 'post_c'
                for response in self.responses:
                    numerator, row, predictions = self._forward_response(response)
                    total_numerator.add_(numerator)
                    rows.append(row)
                    transition_rows.append(repair_transitions(
                        response['original_proposals'], response['post_k_proposals'], predictions,
                        response['labels'], response['validity_weights'], self.model.eos_token_ids,
                    ))
            _sync(self.parameters)
        finally:
            self.costs['post_c_s'] += time.perf_counter() - started
        value = total_numerator / self.global_denominator if float(self.global_denominator) else total_numerator
        if not bool(torch.isfinite(value)):
            raise ValueError('Nonfinite post-correction aggregate violation')
        self.post_c = dict(V=float(value), numerator=float(total_numerator),
                           denominator=float(self.global_denominator), responses=rows,
                           totals=self._aggregate(rows), repair_transitions=transition_rows,
                           repair_totals={name: sum(row[name] for row in transition_rows)
                                          for name in transition_rows[0]})
        self._save('POST_C.json', self.post_c)
        return self.post_c

    def _save(self, name, record):
        if self.output_dir is not None:
            write_json(self.output_dir / name, record)

    def report(self):
        keys = ('record_id', 'input_ids', 'loss_mask', 'anchors', 'keep', 'labels',
                'validity_weights', 'original_proposals', 'protected_weights')
        metadata = [{name: (response[name].tolist() if torch.is_tensor(response[name]) else response[name])
                     for name in keys} for response in self.responses]
        return dict(work=dict(self.work), costs=dict(self.costs), forward_calls=dict(self.forward_calls),
                    cost_scopes={'prefix_chunk_capture_s': 'chunk-local prefix capture only; input copies and parameter snapshots are in outer runtime'},
                    K_update_norm=self.k_update_norm, responses=metadata,
                    post_k=self.post_k, correction=self.correction, post_c=self.post_c)

    def close(self):
        if self.closed:
            return
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        for response in self.responses:
            response.pop('hidden_states', None)
        self.gradients.clear()
        self.parameters = ()
        self.model = self.trainer = None
        self.closed = True
