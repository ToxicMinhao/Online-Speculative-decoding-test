"""Ordinary K event 2 from KRC1's exact state, with correction removed.

The qualified trainer still owns sampling, loss, accumulation, clipping, Adam
and publication. This adapter changes the recorded correction contract only
after exact restoration, and observes existing work without adding forwards.
"""
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
import json
import math
from pathlib import Path
import time

from . import krc_continuation as continuation
from .storage import write_json


ZERO_REFERENCE_WORK = {
    'pre_reference_response_forwards': 0,
    'post_k_reference_response_forwards': 0,
    'reference_backwards': 0,
    'protected_prefix_captures': 0,
    'direct_corrections': 0,
    'replay_updates': 0,
    'diagnostic_response_forwards': 0,
}


def off_contract():
    return {
        'arm': 'KRC2_OFF', 'version': 1, 'base_objective': 'osd_first_error',
        'source': 'complete_KRC1_state', 'reference_processing': False,
        'protected_prefix_capture': False, 'reference_gradients': False,
        'direct_correction': False, 'probe_steps': [],
        'optimizer': 'unchanged_native_K_AdamW', 'extra_optimizer_steps': 0,
    }


def diagnostic_json(value):
    """Preserve nonfinite failure evidence without emitting invalid JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: diagnostic_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [diagnostic_json(item) for item in value]
    return value


class KRCOffTrainer(continuation.KRCContinuationTrainer):
    """Strict KRC1 restoration followed by one explicitly disabled event."""

    def __init__(self, *args, **kwargs):
        self.off_status = 'not_enabled'
        self.off_counts = Counter()
        self.off_costs = Counter()
        self.off_rows = []
        self.first_A = None
        self.off_migration = None
        super().__init__(*args, **kwargs)

    def enable_off_continuation(self, *, resume_check, first_A_records, output_dir):
        """Record the sole contract change after the campaign's live-state gate."""
        if (resume_check.get('status') != 'passed'
                or resume_check.get('exact_resume') is not True
                or resume_check.get('global_step') != self.global_step
                or resume_check.get('presentations') != self.presentations):
            raise ValueError('Exact live KRC1 restoration must pass before disabling correction')
        if (self.off_status != 'not_enabled'
                or self.contract.get('reference_correction_event')
                != continuation.krc1_reference_contract()
                or self.global_step != continuation.RESUME_STEP
                or self.core.accumulation_remainder):
            raise ValueError('OFF migration requires the unchanged restored KRC1 boundary')
        if len(first_A_records) != 2:
            raise ValueError('Require the two retained KRC2 first-A presentation records')
        self._first_A_reference = deepcopy(first_A_records)
        self._off_directory = Path(output_dir)
        self._off_directory.mkdir(parents=True, exist_ok=True)
        before = deepcopy(self.contract)
        self.contract['reference_correction_event'] = off_contract()
        self.off_migration = {
            'scope': 'Disable event-2 reference processing, direct correction and probes after exact restoration',
            'before': before, 'after': deepcopy(self.contract),
            'changed_contract_keys': ['reference_correction_event'],
            'resume_check': deepcopy(resume_check), 'reseeded': False,
            'optimizer_reset': False, 'ordinary_K_settings_changed': False,
        }
        write_json(self._off_directory / 'CONTRACT_MIGRATION.json', self.off_migration)
        self._krc_continuation_active = True
        self.off_status = 'ready'

    @contextmanager
    def _observe_work(self):
        """Count existing calls; retain no features, prefix masks or gradients."""
        model = self.bundle.model
        native_step = self.core.train_step
        native_backward = self.backend.backward
        optimizer = self.backend.optimizer.optimizer
        native_adam = optimizer.step
        handles = []

        def counted(operation, name, *args, **kwargs):
            self.off_counts[name + '_attempted'] += 1
            try:
                result = operation(*args, **kwargs)
            except BaseException:
                self.off_counts[name + '_failed'] += 1
                raise
            self.off_counts[name + '_completed'] += 1
            return result

        def backward(*args, **kwargs):
            return counted(native_backward, 'backward', *args, **kwargs)

        def adam_step(*args, **kwargs):
            return counted(native_adam, 'Adam', *args, **kwargs)

        def observed_step(batch, context=None):
            learning_rate = self.backend.optimizer.get_learning_rate()
            result = native_step(batch, context)
            self.off_counts['presentations_completed'] += 1
            row = {
                'record_id': batch.sample_ids[0],
                'pass': self._presentation_schedule.rows[self._presentation_schedule.position]['pass'],
                'optimizer_step_after': self.global_step + int(result.optimizer_stepped),
                'optimizer_stepped': result.optimizer_stepped,
                'learning_rate': learning_rate if result.optimizer_stepped else None,
                'metrics': deepcopy(result.materialize_metrics()),
                'sampled_anchors': deepcopy(self._anchor_observations),
                'metrics_scope': 'accumulation_window' if result.optimizer_stepped else 'presentation',
            }
            self.off_rows.append(row)
            started = time.perf_counter()
            with (self._off_directory / 'PRESENTATIONS.jsonl').open('a') as stream:
                stream.write(json.dumps(diagnostic_json(row), allow_nan=False) + '\n')
                stream.flush()
            self.off_costs['journal_s'] += time.perf_counter() - started
            if len(self.off_rows) == 2:
                started = time.perf_counter()
                try:
                    self._check_first_A(optimizer)
                finally:
                    self.off_costs['first_A_check_s'] += time.perf_counter() - started
            return result

        def before_response(*args):
            self.off_counts['response_forwards_attempted'] += 1

        def after_response(*args):
            self.off_counts['response_forwards_completed'] += 1

        self.core.train_step = observed_step
        self.backend.backward = backward
        optimizer.step = adam_step
        handles.append(model.register_forward_pre_hook(before_response))
        handles.append(model.register_forward_hook(after_response))
        for module, name in [
            (model.draft_model, 'draft_forwards'),
            (model.lm_head, 'head_forwards_including_recompute'),
            *((layer, 'layer_forwards_including_recompute') for layer in model.draft_model.layers),
        ]:
            def count_module(*args, key=name):
                self.off_counts[key] += 1
            handles.append(module.register_forward_pre_hook(count_module))
        try:
            yield
        finally:
            self.core.train_step = native_step
            self.backend.backward = native_backward
            optimizer.step = native_adam
            for handle in handles:
                handle.remove()

    def _check_first_A(self, optimizer):
        """Use the actual first update; no historical tensors or extra work."""
        fields = ('record_id', 'pass', 'sampled_anchors', 'optimizer_step_after',
                  'optimizer_stepped', 'learning_rate', 'metrics', 'metrics_scope')
        differences = []
        for index, (actual, retained) in enumerate(zip(self.off_rows, self._first_A_reference)):
            for field in fields:
                if actual[field] != retained[field]:
                    differences.append(f'presentation_{index + 1}.{field}')
        counts = sorted({float(state['step']) for state in optimizer.state.values()})
        expected_work = {'response_forwards_completed': 2, 'backward_completed': 2,
                         'Adam_completed': 1, 'presentations_completed': 2}
        checks = {
            'retained_first_A_records_match': not differences,
            'work_matches': all(self.off_counts[key] == count for key, count in expected_work.items()),
            'Adam_counters_match': counts == [float(continuation.RESUME_STEP + 1)],
            'continuous_optimizer': optimizer is self.backend.optimizer.optimizer,
            'empty_accumulation': self.core.accumulation_remainder == 0,
            'micro_counter': self.core._micro == 2 * (continuation.RESUME_STEP + 1),
            'no_reference_or_capture': (self.bundle.model.reference is None
                                        and not hasattr(self.bundle.model, 'prefix_guard')
                                        and not hasattr(self, 'prefix_guard')),
        }
        self.first_A = {
            'status': 'passed' if all(checks.values()) else 'failed', 'checks': checks,
            'differences': differences, 'records': deepcopy(self.off_rows),
            'retained_records': deepcopy(self._first_A_reference),
            'event_local_step': 1, 'global_step_before': continuation.RESUME_STEP,
            'global_step_after': continuation.RESUME_STEP + 1, 'optimizer_counters': counts,
            'actual_work': dict(self.off_counts), 'zero_reference_work': dict(ZERO_REFERENCE_WORK),
            'extra_forwards': 0, 'extra_optimizer_steps': 0,
            'scope': 'KRC2 first ordinary A update before its first correction; no later equality claim',
        }
        write_json(self._off_directory / 'FIRST_A.json', diagnostic_json(self.first_A))
        if not all(checks.values()):
            raise ValueError('KRC2_OFF first-A check failed: ' + ', '.join(
                key for key, value in checks.items() if not value))

    def _train_passes(self, features, passes):
        if self._presentation_schedule is None:
            raise ValueError('KRC2_OFF requires retained event-2 order and anchor assertions')
        self.off_status = 'running'
        try:
            with self._observe_work():
                rows = super()._train_passes(features, passes)
            expected = {'response_forwards_completed': 2 * continuation.EVENT_WINDOWS,
                        'backward_completed': 2 * continuation.EVENT_WINDOWS,
                        'presentations_completed': 2 * continuation.EVENT_WINDOWS,
                        'Adam_completed': continuation.EVENT_WINDOWS}
            if (any(self.off_counts[key] != count for key, count in expected.items())
                    or self.global_step != continuation.RESUME_STEP + continuation.EVENT_WINDOWS
                    or self.core.accumulation_remainder):
                raise ValueError('KRC2_OFF ordinary event accounting differs')
            self.off_status = 'completed'
            return rows
        except BaseException:
            self.off_status = 'failed'
            raise
        finally:
            write_json(self._off_directory / 'WORK.json', diagnostic_json(self.off_report()))

    def off_report(self):
        return {
            'arm': 'KRC2_OFF', 'status': self.off_status,
            'start_step': continuation.RESUME_STEP, 'global_step': self.global_step,
            'start_global_step': continuation.RESUME_STEP, 'end_global_step': self.global_step,
            'extra_qualification_optimizer_steps': 0,
            'event_local_windows': self.off_counts['Adam_completed'],
            'counts': dict(self.off_counts), 'zero_reference_work': dict(ZERO_REFERENCE_WORK),
            'work': {
                'response_forwards': self.off_counts['response_forwards_completed'],
                'K_backwards': self.off_counts['backward_completed'],
                'Adam_steps': self.off_counts['Adam_completed'],
                'presentations': self.off_counts['presentations_completed'],
                'reference_forwards': 0, 'reference_backwards': 0,
                'protected_prefix_captures': 0, 'direct_corrections': 0,
                'probe_forwards': 0,
            },
            'costs': dict(self.off_costs), 'first_A': deepcopy(self.first_A),
            'migration': deepcopy(self.off_migration),
            'sampling': 'normal RNG draws asserted against retained event-2 bindings',
            'replay_updates': 0, 'direct_corrections': 0,
        }

    def _checkpoint_state(self, event):
        event['training_variant'] = 'KRC2_OFF'
        event['correction_removal'] = self.off_report()
        return super()._checkpoint_state(event)
