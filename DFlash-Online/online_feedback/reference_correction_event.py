"""KRC: the tested post-K correction fitted to a separate saved response pair."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import math
import time

import torch

from .prefix_correction import PrefixCorrection, _sync
from .prefix_correction_event import AuxiliaryIntegrity, PrefixCorrectionEvent
from .prefix_guard import preserved_probe_runtime
from .ranking import fixed_anchors
from .rng import capture_rng, restore_rng


CACHE_NAMES = ('_cached_block_mask', '_cached_seq_len', '_cached_bsz')


class ReferencePrefixCorrection(PrefixCorrection):
    """Capture fresh reference prefixes and replay their exact stochastic state."""

    @torch.no_grad()
    def capture_reference(self, bindings, feature_paths):
        self.phase = 'pre_reference'
        self.work['pre_reference_attempted'] = 0
        self.work['pre_reference_chunks'] = 0
        self.pre_rows = []
        _sync(self.parameters)
        started = time.perf_counter()
        try:
            with preserved_probe_runtime(self.model):
                for binding in bindings:
                    self._capture_response(binding, feature_paths[binding['record_id']])
            _sync(self.parameters)
        finally:
            self.costs['pre_reference_s'] = time.perf_counter() - started

    def _capture_response(self, binding, path):
        raw = torch.load(path, map_location='cpu', weights_only=True)
        if raw['record_id'] != binding['record_id']:
            raise ValueError('Reference feature identity differs from its sealed binding')
        tensors = self.trainer._batch(raw).tensors
        device = self.parameters[0].device
        inputs, features, mask = [tensors[name].to(device) for name in
                                 ('input_ids', 'hidden_states', 'loss_mask')]
        observation, = binding['sampled_anchors']
        anchors = torch.tensor(observation['positions'], device=device, dtype=torch.long)
        keep = torch.tensor(observation['keep_mask'], device=device, dtype=torch.bool)
        replay_rng = capture_rng()
        replay_caches = {name: getattr(self.model, name) for name in CACHE_NAMES
                         if hasattr(self.model, name)}
        self.work['pre_reference_attempted'] += 1
        with fixed_anchors(self.model, anchors, keep):
            hidden = self.model._forward_draft_blocks(inputs, features, mask)[2]
        labels, weights, _ = self.model._block_labels_and_weights(inputs, mask, anchors, keep)
        hidden = hidden.reshape(1, anchors.shape[1], self.model.block_size, -1)
        self.begin_response(inputs, features, mask, anchors, keep, labels, weights)
        for start in range(0, anchors.shape[1], self.model.objective_chunk_blocks):
            stop = start + self.model.objective_chunk_blocks
            logits = self.model.lm_head(hidden[:, start:stop])
            self.work['pre_reference_chunks'] += 1
            self.chunk(logits, labels[:, start:stop], weights[:, start:stop], start)
        self.finish_response(hidden.new_zeros(()), hidden.new_zeros(()))
        response = self.responses[-1]
        response['_replay_rng'] = replay_rng
        response['_replay_caches'] = replay_caches
        self.pre_rows.append({
            'record_id': response['record_id'],
            'protected_positions': int((response['protected_weights'] > 0).sum()),
            'protected_weight': float(response['protected_weights'].sum()),
        })

    def _forward_response(self, response):
        # Both reference phases use the same eval mode, RNG and cache starting
        # state. Their temporary stochastic draws never advance ordinary K RNG.
        with preserved_probe_runtime(self.model):
            restore_rng(response['_replay_rng'])
            for name, value in response['_replay_caches'].items():
                setattr(self.model, name, value)
            return super()._forward_response(response)

    def close(self):
        for response in self.responses:
            response.pop('_replay_rng', None)
            response.pop('_replay_caches', None)
        super().close()


class LearningPrefixProbe(PrefixCorrection):
    """Score A without obtaining a violation gradient or applying a correction."""

    def __init__(self, trainer):
        super().__init__(trainer)
        # The reference recorder counts actual forwards for both roles. Avoid
        # installing a second set of accounting hooks on the same modules.
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    @torch.no_grad()
    def measure_post_k(self):
        self.phase = 'post_k'
        device = self.parameters[0].device
        self.global_denominator = sum(
            (response['protected_weights'].sum().to(device) for response in self.responses),
            torch.zeros((), dtype=torch.float32, device=device),
        )
        numerator = torch.zeros((), dtype=torch.float32, device=device)
        rows = []
        _sync(self.parameters)
        started = time.perf_counter()
        try:
            with preserved_probe_runtime(self.model):
                for response in self.responses:
                    value, row, predictions = self._forward_response(response)
                    numerator.add_(value)
                    response['post_k_proposals'] = predictions
                    rows.append(row)
            _sync(self.parameters)
        finally:
            self.costs['post_k_s'] += time.perf_counter() - started
        value = numerator / self.global_denominator if float(self.global_denominator) else numerator
        if not bool(torch.isfinite(value)):
            raise ValueError('Nonfinite learning-probe violation')
        self.value = value
        self.post_k = dict(V=float(value), numerator=float(numerator),
                           denominator=float(self.global_denominator), responses=rows,
                           totals=self._aggregate(rows))
        return self.post_k


class ReferenceCorrectionWindow:
    """Two-role adapter for the existing continuous KC event manager."""

    def __init__(self, trainer, window, feature_paths, *, probe, first_window_check=None):
        self.trainer = trainer
        self.window = window
        self.feature_paths = feature_paths
        self.reference = ReferencePrefixCorrection(trainer, record_ids=window['reference_record_ids'])
        self.learning = LearningPrefixProbe(trainer) if probe else None
        self.first_window_check = first_window_check
        self._record_ids = ()
        self.pre_integrity_s = 0.0
        self.pre_integrity = None
        self.gate_passed = False
        self.gate_s = 0.0

    @property
    def record_ids(self):
        return self._record_ids

    @record_ids.setter
    def record_ids(self, values):
        expected = tuple(self.window['learning_record_ids'][:len(values)])
        if tuple(values) != expected:
            raise ValueError('Learning identities differ from the sealed reference schedule')
        self._record_ids = tuple(values)
        if self.learning is not None:
            self.learning.record_ids = tuple(values)

    @contextmanager
    def capture_k(self):
        started = time.perf_counter()
        snapshot = AuxiliaryIntegrity(self.trainer)
        self.pre_integrity_s += time.perf_counter() - started
        self.reference.capture_reference(self.window['reference'], self.feature_paths)
        started = time.perf_counter()
        self.pre_integrity = snapshot.check('reference_prefix_capture', parameters_read_only=True)
        self.pre_integrity_s += time.perf_counter() - started
        del snapshot
        self.reference.phase = 'learning_K'
        if self.learning is not None:
            with self.learning.capture_k():
                yield self
            self.reference.k_update_norm = self.learning.k_update_norm
        else:
            before = [parameter.detach().clone() for parameter in self.reference.parameters]
            try:
                yield self
                squared = torch.zeros((), dtype=torch.float64, device=before[0].device)
                for parameter, previous in zip(self.reference.parameters, before):
                    squared += (parameter.detach().float() - previous.float()).double().square().sum()
                self.reference.k_update_norm = math.sqrt(float(squared))
            finally:
                del before

    def measure_post_k(self):
        result = self.reference.measure_post_k()
        if self.learning is not None:
            self.reference.phase = 'post_k_learning'
            self.learning.measure_post_k()
            if self.first_window_check is not None:
                started = time.perf_counter()
                self.first_window_check(self.learning.report())
                self.gate_s += time.perf_counter() - started
                self.gate_passed = True
        return result

    def apply_once(self):
        return self.reference.apply_once()

    def measure_post_c(self):
        result = self.reference.measure_post_c()
        if self.learning is not None:
            self.reference.phase = 'post_c_learning'
            # This is the one already-applied R correction, never an A update.
            self.learning.correction = self.reference.correction
            self.learning.measure_post_c()
        return result

    def report(self):
        reference = self.reference.report()
        learning = self.learning.report() if self.learning is not None else None
        if learning is not None:
            learning['diagnostic_only'] = True
            learning['correction_role'] = 'the same parameter correction fitted only on reference R'
        rw = reference['work']
        lw = learning['work'] if learning is not None else {}
        work = {
            'original_response_forwards': reference['forward_calls'].get('learning_K/draft', 0),
            'pre_reference_response_forwards': rw['original_response_forwards'],
            'post_k_reference_response_forwards': rw['post_k_response_forwards'],
            'post_c_reference_response_forwards': rw['post_c_response_forwards'],
            'post_k_learning_response_forwards': lw.get('post_k_response_forwards', 0),
            'post_c_learning_response_forwards': lw.get('post_c_response_forwards', 0),
            'violation_backward_attempted': rw['violation_backward_attempted'],
            'violation_backward_completed': rw['violation_backward_completed'],
            'violation_backward_failed': rw['violation_backward_failed'],
            'direct_corrections': rw['direct_corrections'],
            'pre_reference_attempted': rw.get('pre_reference_attempted', 0),
            'post_k_reference_attempted': rw['post_k_response_attempted'],
            'post_c_reference_attempted': rw['post_c_response_attempted'],
            'pre_reference_chunks': rw.get('pre_reference_chunks', 0),
            'post_k_reference_chunks': rw['post_k_objective_chunks'],
            'post_c_reference_chunks': rw['post_c_objective_chunks'],
            'learning_objective_chunks_at_probe_steps': lw.get('current_objective_chunks', 0),
            'post_k_learning_chunks': lw.get('post_k_objective_chunks', 0),
            'post_c_learning_chunks': lw.get('post_c_objective_chunks', 0),
        }
        costs = {
            'pre_reference_s': reference['costs'].get('pre_reference_s', 0.0),
            'post_k_s': reference['costs']['post_k_s'],
            'correction_s': reference['costs']['correction_s'],
            'post_c_reference_s': reference['costs']['post_c_s'],
            'post_k_learning_s': learning['costs']['post_k_s'] if learning else 0.0,
            'post_c_learning_s': learning['costs']['post_c_s'] if learning else 0.0,
            'pre_reference_integrity_s': self.pre_integrity_s,
            'first_learning_gate_s': self.gate_s,
        }
        return {
            **reference, 'work': work, 'costs': costs,
            'learning': learning, 'reference': reference,
            'learning_record_ids': self.window['learning_record_ids'],
            'reference_record_ids': self.window['reference_record_ids'],
            'schedule_binding': deepcopy(self.window),
            'reference_pre_integrity': self.pre_integrity,
            'first_learning_gate_passed': self.gate_passed,
        }

    def close(self):
        self.reference.close()
        if self.learning is not None:
            self.learning.close()
        self.trainer = self.feature_paths = self.first_window_check = None


class ReferenceCorrectionEvent(PrefixCorrectionEvent):
    """Use one fresh reference controller per existing two-response K window."""

    def __init__(self, trainer, *, schedule, feature_paths, output_dir=None,
                 probe_steps=(1, 64, 128), first_window_check=None):
        if first_window_check is not None and 1 not in probe_steps:
            raise ValueError('The learning reproduction gate requires the step-1 probe')
        super().__init__(trainer, output_dir=output_dir, probe_steps=probe_steps)
        self.schedule = schedule
        self.feature_paths = feature_paths
        self.learning_gate = first_window_check
        self.contract.update(arm='KRC', capture='current_reference_strict_prefix',
                             reference_selection='next_disjoint_pair_cyclic_within_pass',
                             replay='fixed_anchors_and_pre_reference_rng_cache_in_eval_mode',
                             reference_exposure='additional_training_exposure')
        del trainer.contract['prefix_correction_event']
        trainer.contract['reference_correction_event'] = deepcopy(self.contract)

    def _begin_window(self):
        self.window_step = self.trainer.global_step - self.start_step + 1
        self.window_started = time.perf_counter()
        self.window_integrity_s = 0.0
        self.window_integrity = {}
        window = self.schedule['windows'][self.window_step - 1]
        if window['step'] != self.window_step:
            raise ValueError('Reference schedule step differs from native K progress')
        self.active = ReferenceCorrectionWindow(
            self.trainer, window, self.feature_paths,
            probe=self.window_step in self.probe_steps,
            first_window_check=self.learning_gate if self.window_step == 1 else None,
        )
        self.capture = self.active.capture_k()
        self.capture.__enter__()

    def report(self):
        report = super().report()
        report['arm'] = 'KRC'
        report['cost_scopes'] = {
            'pre_reference_s_plus_post_k_s_plus_correction_s': 'intrinsic reference correction work',
            'post_c_reference_s_plus_post_k_learning_s_plus_post_c_learning_s': '18 diagnostic response forwards',
            'integrity_s_plus_pre_reference_integrity_s_plus_first_learning_gate_s': 'state-copy, preservation checks and first-step reproduction',
            'window_wall_s': 'complete window, including inter-presentation input loading',
        }
        return report

    def close(self):
        super().close()
        self.learning_gate = self.feature_paths = self.schedule = None
