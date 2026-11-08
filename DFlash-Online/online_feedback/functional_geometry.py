"""Filter a native K+ AdamW displacement using fixed pre-step distributions.

The ordinary loss, backward and Adam step are unchanged. The hook observes the
two existing response forwards, then runs isolated full-block replays. It never
installs a loss extension or changes the ordinary sampler.
"""
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import time

import torch

from specforge.core.chunking import checkpointed_chunk_reduce

from . import functional_geometry_math as geometry
from .ranking import fixed_anchors
from .rng import preserve_rng
from .storage import write_json


def _cleanup(action, failure, record, label):
    """Keep the first model failure authoritative if restoration also fails."""
    if failure is None:
        action()
        return
    try:
        action()
    except BaseException as secondary:
        message = f'{label}: {type(secondary).__name__}: {secondary}'
        if record is not None:
            record.setdefault('diagnostic_errors', []).append(message)
        failure.add_note(message)


@contextmanager
def isolated_geometry_runtime(model):
    """Restore RNG, module modes, registered buffers and reusable cache fields.

    Buffer identities matter: rotary implementations may replace a buffer when
    its cached length grows. Restore the original objects and their contents.
    Parameter endpoints are managed separately by the optimizer hook.
    """
    modules = tuple(model.modules())
    modes = [(module, module.training) for module in modules]
    buffers = []
    fields = []
    cache_names = ('_cached_block_mask', '_cached_seq_len', '_cached_bsz',
                   'max_seq_len_cached', 'original_max_seq_len')
    for module in modules:
        original = dict(module._buffers)
        contents = {name: value.detach().clone() for name, value in original.items()
                    if value is not None}
        buffers.append((module, original, contents, set(module._non_persistent_buffers_set)))
        fields.append((module, {name: getattr(module, name) for name in cache_names
                                if hasattr(module, name)}))
    def restore():
        with torch.no_grad():
            for module, original, contents, nonpersistent in buffers:
                module._buffers.clear()
                module._buffers.update(original)
                module._non_persistent_buffers_set = nonpersistent
                for name, contents_value in contents.items():
                    original[name].copy_(contents_value)
            for module, saved in fields:
                for name in cache_names:
                    if name in saved:
                        setattr(module, name, saved[name])
                    elif hasattr(module, name):
                        delattr(module, name)
            for module, training in modes:
                module.training = training
    failure = None
    with preserve_rng():
        try:
            model.eval()
            yield
        except BaseException as error:
            failure = error
            raise
        finally:
            _cleanup(restore, failure, None, 'auxiliary runtime restoration')


@dataclass
class ResponseReference:
    """Only hidden states and per-position scalars survive a projection chunk."""
    response: dict
    hidden: torch.Tensor
    weights: torch.Tensor
    proposals: torch.Tensor
    prefix_mask: torch.Tensor
    margins: torch.Tensor
    c: float
    gradients: tuple
    native_risk_terms: tuple
    statistics: dict


class FunctionalGeometry:
    """One native optimizer step, one geometry proposal and one actual check."""

    def __init__(self, trainer, *, arm='FSG', strength=1.0, output_dir=None,
                 observer=None, work_callback=None):
        if arm not in ('FSG', 'SCALAR') or strength not in (0., 1.):
            raise ValueError('FSG uses FSG/SCALAR and sealed strength one or neutral zero')
        if (trainer.objective != 'osd_first_error' or trainer.accumulation != 2
                or trainer.reduction != 'response_mean'):
            raise ValueError('FSG requires the qualified two-response K objective')
        self.trainer = trainer
        self.model = trainer.bundle.model
        if getattr(self.model, 'first_error_redistribution', None) is None:
            raise ValueError('Install ordinary K+ before the functional geometry hook')
        self.parameters = tuple(trainer.backend.optimizer.model_params)
        self.parameter_names = {id(parameter): name for name, parameter in
                                trainer.bundle.draft_model.named_parameters()}
        self.arm = arm
        self.strength = float(strength)
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.observer = observer
        self.work_callback = work_callback
        self.responses = []
        self.active_parameters = ()
        self.theta0 = ()
        self.theta_native = ()
        self.phase = 'ordinary'
        self.closed = False
        self.vocab_size = None
        self.original_forward = self.model._forward_draft_blocks
        self.original_step = trainer.backend.optimizer.step
        self.hooks = []
        self.clear_event()
        for name, module in [('draft', self.model.draft_model),
                             ('vocabulary_head', self.model.lm_head),
                             *((f'draft_layer_{i}', layer) for i, layer in
                               enumerate(getattr(self.model.draft_model, 'layers', ())))]:
            def count(_module, _inputs, name=name):
                key = f'{self.phase}/{name}'
                self.forward_calls[key] = self.forward_calls.get(key, 0) + 1
                if name == 'vocabulary_head':
                    self.work['head_projection_calls'] += 1
            self.hooks.append(module.register_forward_pre_hook(count))

    def clear_event(self):
        if self.responses:
            raise ValueError('Cannot clear FSG accounting inside an accumulation window')
        self.work = dict(captured_ordinary_responses=0, head_projection_calls=0,
                         checkpoint_proxy_calls=0, state_copy_passes=0)
        for kind in ('additional_response_forwards', 'additional_backwards', 'optimizer_steps'):
            for status in ('attempted', 'completed', 'failed'):
                self.work[f'{kind}_{status}'] = 0
        self.costs = {}
        self.steps = []
        self.forward_calls = {}

    def _sync(self):
        if self.parameters and self.parameters[0].is_cuda:
            torch.cuda.synchronize(self.parameters[0].device)

    @contextmanager
    def _timed(self, name):
        self._sync()
        started = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            self.costs[name] = self.costs.get(name, 0.) + time.perf_counter() - started

    @contextmanager
    def _work(self, kind):
        if self.work_callback is not None:
            self.work_callback(kind, 'attempted')
        self.work[f'{kind}_attempted'] += 1
        self._write_work()
        try:
            yield
        except BaseException as error:
            self.work[f'{kind}_failed'] += 1
            if self.work_callback is not None:
                _cleanup(lambda: self.work_callback(kind, 'failed'), error, None,
                         'failed-work callback')
            _cleanup(self._write_work, error, None, 'failed-work write')
            raise
        else:
            self.work[f'{kind}_completed'] += 1
            if self.work_callback is not None:
                self.work_callback(kind, 'completed')
            self._write_work()

    def _write_work(self):
        if self.output_dir is not None:
            write_json(self.output_dir / 'WORK.json', dict(work=self.work, phase=self.phase))

    def _notify(self, phase):
        if self.observer is not None:
            self.observer(phase, self)

    def capture_forward(self, input_ids, hidden_states, loss_mask, *args, **kwargs):
        """Capture original blocks after their native forward, without its graph."""
        result = self.original_forward(input_ids, hidden_states, loss_mask, *args, **kwargs)
        anchors, keep, _ = result
        with self._timed('ordinary_capture_s'):
            labels, weights, _ = self.model._block_labels_and_weights(
                input_ids, loss_mask, anchors, keep)
            tensors = dict(input_ids=input_ids, hidden_states=hidden_states,
                           loss_mask=loss_mask, anchors=anchors, keep=keep,
                           labels=labels, validity_weights=weights)
            response = {name: value.detach().cpu().clone() for name, value in tensors.items()}
            response['presentation'] = self.trainer.presentations
            self.responses.append(response)
            self.work['captured_ordinary_responses'] += 1
        return result

    def forward_hidden(self, response, phase, *, grad=False):
        """One complete-block replay using retained anchors, never new sampling."""
        self.phase = phase
        device = self.parameters[0].device
        tensors = {name: response[name].to(device) for name in
                   ('input_ids', 'hidden_states', 'loss_mask', 'anchors', 'keep')}
        with self._work('additional_response_forwards'), self._timed(phase + '_forward_s'):
            with torch.set_grad_enabled(grad), fixed_anchors(
                    self.model, tensors['anchors'], tensors['keep']):
                hidden = self.original_forward(tensors['input_ids'], tensors['hidden_states'],
                                               tensors['loss_mask'])[2]
        return hidden.reshape(1, tensors['anchors'].shape[1], self.model.block_size, -1)

    @torch.no_grad()
    def _scan_reference(self, hidden, response, chunk_blocks):
        """Normalize across the response before the BF16 VJP is constructed."""
        labels = response['labels'].to(hidden.device)
        valid = response['validity_weights'].to(hidden.device)
        raw_weights, proposals, masks, margins = [], [], [], []
        size = chunk_blocks or hidden.shape[1]
        with self._timed('reference_head_scan_s'):
            for start in range(0, hidden.shape[1], size):
                stop = start + size
                logits = self.model.lm_head(hidden[:, start:stop])
                geometry.finite(logits, 'reference head logits')
                self.vocab_size = logits.shape[-1]
                probabilities = logits.float().softmax(-1)
                canonical = probabilities.gather(-1, labels[:, start:stop, :, None]).squeeze(-1)
                probabilities.scatter_(-1, labels[:, start:stop, :, None], -torch.inf)
                rival = probabilities.max(-1).values
                predicted = logits.argmax(-1)
                values = geometry.prefix_statistics(predicted, labels[:, start:stop],
                    valid[:, start:stop], canonical, rival, self.model.eos_token_ids)
                raw_weights.append(values['raw_weights'])
                proposals.append(predicted)
                masks.append(values['mask'])
                margins.append(values['margin'])
        raw_weights = torch.cat(raw_weights, 1)
        weights = geometry.normalize_weights(raw_weights, raw_weights.sum(dtype=torch.float64))
        return weights, torch.cat(proposals, 1), torch.cat(masks, 1), torch.cat(margins, 1)

    def _proxy_chunk(self, hidden, native_hidden, weights):
        """Checkpointed vocabulary proxy; all cotangents are detached."""
        self.work['checkpoint_proxy_calls'] += 1
        logits = self.model.lm_head(hidden)
        with torch.no_grad():
            native_logits = self.model.lm_head(native_hidden)
            cotangent, energy = geometry.geometry_terms(logits, native_logits, weights)
            risk, absolute = geometry.risk_terms(logits, native_logits, weights)
        proxy = (logits.float() * cotangent).sum(dtype=torch.float32)
        return proxy, energy, risk, absolute

    def build_reference(self, response, native_hidden, active_parameters, *, chunk_blocks=None):
        """One pre-state forward and at most one VJP, with the live graph reused.

        The caller must already have restored theta0 and isolated runtime state.
        ``chunk_blocks=0`` is reserved for a small native qualification oracle.
        """
        chunk_blocks = self.model.objective_chunk_blocks if chunk_blocks is None else chunk_blocks
        hidden = self.forward_hidden(response, 'prestate', grad=True)
        weights, proposals, mask, margins = self._scan_reference(hidden.detach(), response, chunk_blocks)
        if bool(mask.any()):
            self.phase = 'prestate_proxy'
            with self._timed('proxy_forward_s'):
                proxy, energy, risk, absolute = checkpointed_chunk_reduce(
                    self._proxy_chunk, hidden, native_hidden.detach(), weights,
                    chunk_size=chunk_blocks, dim=1, checkpoint=True)
            if float(energy) > 0:
                self.phase = 'prestate_vjp'
                with self._work('additional_backwards'), self._timed('auxiliary_backward_s'):
                    gradients = torch.autograd.grad(proxy, active_parameters, allow_unused=True)
                for gradient in gradients:
                    if gradient is not None:
                        geometry.finite(gradient, 'native VJP gradient')
                gradients = tuple(None if value is None else value.detach() for value in gradients)
            else:
                gradients = tuple(None for _ in active_parameters)
        else:
            energy = risk = absolute = hidden.new_zeros((), dtype=torch.float64)
            gradients = tuple(None for _ in active_parameters)
        statistics = dict(protected_occurrences=int(mask.sum()),
            weight_sum=float(weights.double().sum()),
            margin_floor_occurrences=int((mask & (margins < geometry.MARGIN_FLOOR)).sum()),
            margin_floor_weight=float((weights * (margins < geometry.MARGIN_FLOOR)).double().sum()),
            tie_occurrences=int((mask & (margins == 0)).sum()),
            tie_weight=float((weights * (margins == 0)).double().sum()),
            native_risk_unaveraged=float(risk), native_absolute_sum_unaveraged=float(absolute),
            anchors=int(response['anchors'].numel()))
        return ResponseReference(response, hidden.detach(), weights.detach(), proposals.detach(),
                                 mask.detach(), margins.detach(), float(energy), gradients,
                                 (float(risk), float(absolute)), statistics)

    @torch.no_grad()
    def score_candidate(self, references):
        """Score actual BF16 parameters against the fixed pre-state references."""
        parts = []
        rows = []
        size = self.model.objective_chunk_blocks
        for reference in references:
            hidden = self.forward_hidden(reference.response, 'candidate', grad=False)
            self.phase = 'candidate_head'
            risk = hidden.new_zeros((), dtype=torch.float64)
            absolute = hidden.new_zeros((), dtype=torch.float64)
            with self._timed('candidate_head_s'):
                for start in range(0, hidden.shape[1], size):
                    stop = start + size
                    old_logits = self.model.lm_head(reference.hidden[:, start:stop])
                    logits = self.model.lm_head(hidden[:, start:stop])
                    values = geometry.risk_terms(old_logits, logits, reference.weights[:, start:stop])
                    risk += values[0]
                    absolute += values[1]
            parts.append((float(risk), float(absolute)))
            rows.append(dict(risk_unaveraged=float(risk), absolute_sum_unaveraged=float(absolute)))
        return geometry.pooled_risk(parts, self.vocab_size), rows

    @torch.no_grad()
    def restore_endpoint(self, tensors):
        with self._timed('parameter_restore_s'):
            for parameter, saved in zip(self.active_parameters, tensors):
                parameter.copy_(saved)
        self.work['state_copy_passes'] += 1

    def _filter(self):
        """Build references once, propose one endpoint, then measure its risk."""
        native_hidden = [self.forward_hidden(response, 'native_candidate', grad=False).detach()
                         for response in self.responses]
        self.restore_endpoint(self.theta0)
        references = []
        for response, hidden in zip(self.responses, native_hidden):
            references.append(self.build_reference(response, hidden, self.active_parameters))
        del native_hidden
        with self._timed('sketch_solve_s'):
            sketch = geometry.make_sketch([row.gradients for row in references],
                [row.c for row in references], self.theta0, self.theta_native, strength=self.strength)
        native_risk = geometry.pooled_risk([row.native_risk_terms for row in references], self.vocab_size)
        with self._timed('candidate_formation_s'):
            candidate = geometry.write_candidate(self.active_parameters, self.theta0,
                                                  self.theta_native, sketch)
        row = dict(sketch=sketch.statistics, responses=[ref.statistics for ref in references],
                   raw_fsg=candidate, native_risk=native_risk.as_dict())
        if self.strength == 0 or sketch.sigma == 0:
            row.update(decision=dict(accepted=False, reason='lambda_zero' if self.strength == 0
                                     else 'empty_sketch'), candidate_risk=native_risk.as_dict(),
                       candidate=candidate, committed=candidate)
            return row
        if candidate['norm'] > candidate['native_norm']:
            self.restore_endpoint(self.theta_native)
            row.update(decision=dict(accepted=False, reason='rounded_norm_exceeds_native'),
                       candidate=candidate, candidate_risk=None,
                       committed=geometry.endpoint_statistics(self.active_parameters, self.theta0,
                                                              self.theta_native))
            return row
        if self.arm == 'SCALAR':
            with self._timed('scalar_bisection_s'):
                candidate = geometry.scalar_match(self.active_parameters, self.theta0,
                                                  self.theta_native, candidate['norm'])
            row['scalar'] = candidate
        if candidate['identical_native']:
            candidate_risk = native_risk
            candidate_rows = [dict(risk_unaveraged=ref.native_risk_terms[0],
                                   absolute_sum_unaveraged=ref.native_risk_terms[1]) for ref in references]
        else:
            candidate_risk, candidate_rows = self.score_candidate(references)
        decision = geometry.gate(candidate_risk, native_risk, candidate['norm'],
                                 candidate['native_norm'], identical=candidate['identical_native'])
        if not decision['accepted']:
            self.restore_endpoint(self.theta_native)
        row.update(candidate=candidate, candidate_risk=candidate_risk.as_dict(),
                   candidate_response_risks=candidate_rows, decision=decision,
                   committed=geometry.endpoint_statistics(self.active_parameters, self.theta0,
                                                          self.theta_native))
        return row

    def step(self):
        """Advance Adam once; TrainerCore retains ownership of global counters."""
        if len(self.responses) != 2:
            raise ValueError('FSG optimizer boundary requires exactly two captured responses')
        self._sync()
        started = time.perf_counter()
        costs_before = dict(self.costs)
        self.active_parameters = tuple(parameter for parameter in self.parameters if parameter.grad is not None)
        active_names = [self.parameter_names[id(parameter)] for parameter in self.active_parameters]
        with self._timed('pre_snapshot_s'):
            self.theta0 = tuple(parameter.detach().clone() for parameter in self.active_parameters)
        row = dict(global_step=self.trainer.global_step + 1, arm=self.arm, strength=self.strength,
                   active_parameter_names=active_names, active_tensors=len(active_names),
                   active_coordinates=sum(parameter.numel() for parameter in self.active_parameters))
        native_completed = False
        failure = None
        try:
            self._notify('pre')
            self.phase = 'ordinary'
            with self._work('optimizer_steps'), self._timed('native_optimizer_s'):
                result = self.original_step()
            native_completed = True
            with self._timed('native_snapshot_s'):
                self.theta_native = tuple(parameter.detach().clone() for parameter in self.active_parameters)
            native_stats = geometry.endpoint_statistics(self.active_parameters, self.theta0, self.theta_native)
            row['native'] = native_stats
            row['preclip_gradient_norm'] = float(result)
            self._notify('native')
            if native_stats['native_norm'] == 0:
                row.update(decision=dict(accepted=False, reason='zero_native_displacement'), committed=native_stats)
            else:
                with isolated_geometry_runtime(self.model):
                    row.update(self._filter())
            if any(parameter.grad is not None for parameter in self.parameters):
                raise ValueError('Auxiliary geometry changed the cleared native gradient pattern')
            self._notify('committed')
            row['status'] = 'completed'
        except BaseException as error:
            failure = error
            row.update(status='failed', error_type=type(error).__name__, error=str(error))
            if native_completed and self.theta_native:
                _cleanup(lambda: self.restore_endpoint(self.theta_native), failure, row,
                         'failed-step native restoration')
            raise
        finally:
            _cleanup(self._sync, failure, row, 'final synchronization')
            row['hook_total_s'] = time.perf_counter() - started
            row['costs'] = {name: value - costs_before.get(name, 0.)
                            for name, value in self.costs.items()}
            row['cost_scope'] = 'hook_total_includes_native_optimizer_and_nested_component_times'
            row['work_cumulative'] = dict(self.work)
            self.steps.append(row)
            if self.output_dir is not None:
                _cleanup(lambda: write_json(self.output_dir / f"step_{row['global_step']:06d}.json", row),
                         failure, row, 'step result write')
            self.responses.clear()
            self.active_parameters = self.theta0 = self.theta_native = ()
            self.phase = 'ordinary'
        return result

    def report(self):
        work = dict(self.work)
        work['additional_response_forwards'] = work['additional_response_forwards_completed']
        work['additional_backwards'] = work['additional_backwards_completed']
        return dict(arm=self.arm, strength=self.strength, work=work, costs=dict(self.costs),
                    forward_calls=dict(self.forward_calls), steps=list(self.steps))

    def close(self):
        if self.closed:
            return
        self.model._forward_draft_blocks = self.original_forward
        self.trainer.backend.optimizer.step = self.original_step
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()
        self.responses.clear()
        self.active_parameters = self.theta0 = self.theta_native = ()
        self.closed = True


def install_functional_geometry(trainer, *, arm='FSG', strength=1., output_dir=None,
                                observer=None, work_callback=None):
    """Attach after exact recovery; keep existing loss/observation wrappers."""
    existing = getattr(trainer, 'functional_geometry', None)
    if existing is not None and not existing.closed:
        raise ValueError('Functional geometry is already installed')
    hook = FunctionalGeometry(trainer, arm=arm, strength=strength, output_dir=output_dir,
                              observer=observer, work_callback=work_callback)
    trainer.functional_geometry = hook
    hook.model._forward_draft_blocks = hook.capture_forward
    trainer.backend.optimizer.step = hook.step
    return hook
