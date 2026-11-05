"""Bounded KRC2 -> KRC3 -> KRC4 continuation using the qualified KRC path.

Only event orchestration changes. The ordinary trainer, reference capture,
correction arithmetic, optimizer and random sampling remain inherited.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import time

from .krc_continuation import KRCContinuationEvent, krc1_reference_contract
from .reference_correction_event import ReferenceCorrectionEvent
from .storage import write_json
from .training import SpecForgeEventTrainer


EVENT_WINDOWS = 128
RESUME_STEP = 256
FINAL_STEP = 512
HORIZON = 7424


def krc2_reference_contract():
    """The exact qualified KRC2 contract; optional probes are already absent."""
    contract = krc1_reference_contract()
    contract['probe_steps'] = []
    return contract


def _validate_history(trainer, completed_events):
    if len(trainer.events) != completed_events:
        raise ValueError('KRC late continuation requires the complete own event history')
    for number, event in enumerate(trainer.events, 1):
        expected = {
            'event_id': f'event_{number:03d}', 'start_step': (number - 1) * EVENT_WINDOWS,
            'end_step': number * EVENT_WINDOWS, 'optimizer_steps': EVENT_WINDOWS,
            'response_presentations': 2 * EVENT_WINDOWS, 'passes': 2,
            'objective': 'osd_first_error', 'reduction': 'response_mean',
        }
        # Atomic state serialization precedes the external completed receipt.
        # Do not assume that a saved event has its later receipt's status key.
        if (any(event.get(key) != value for key, value in expected.items())
                or len(event.get('features', [])) != EVENT_WINDOWS
                or len(event.get('records', [])) != 2 * EVENT_WINDOWS):
            raise ValueError('KRC late continuation requires the complete own event history')


def _validate_schedule_binding(schedule, ordinary, event_id, start_step):
    """Bind both roles to this event's retained ordinary draws, without sampling."""
    if (ordinary.get('event_id') != event_id or ordinary.get('start_step') != start_step
            or ordinary.get('end_step') != start_step + EVENT_WINDOWS
            or ordinary.get('optimizer_steps') != EVENT_WINDOWS
            or ordinary.get('passes') != 2
            or ordinary.get('response_presentations') != 2 * EVENT_WINDOWS
            or ordinary.get('status') != 'completed'
            or len(ordinary.get('records', [])) != 2 * EVENT_WINDOWS):
        raise ValueError('Retained ordinary schedule has a different event boundary')
    records = ordinary['records']
    for index, row in enumerate(records):
        if (row.get('optimizer_step_after') != start_step + (index + 1) // 2
                or row.get('optimizer_stepped') is not bool(index % 2)
                or row.get('pass') != index // EVENT_WINDOWS):
            raise ValueError('Retained ordinary schedule has different cumulative counters')
    keys = ('record_id', 'pass', 'sampled_anchors')
    for index, window in enumerate(schedule['windows']):
        pass_index, pair_index = divmod(index, EVENT_WINDOWS // 2)
        reference_pair = (pair_index + 1) % (EVENT_WINDOWS // 2)
        reference_start = pass_index * EVENT_WINDOWS + 2 * reference_pair
        learning = [{key: row[key] for key in keys} for row in records[2 * index:2 * index + 2]]
        reference = [{key: row[key] for key in keys}
                     for row in records[reference_start:reference_start + 2]]
        expected = dict(step=index + 1, pass_index=pass_index, batch_index=pair_index,
                        reference_batch_index=reference_pair,
                        learning_record_ids=[row['record_id'] for row in learning],
                        reference_record_ids=[row['record_id'] for row in reference])
        roles_match = all(
            [{key: row[key] for key in keys} for row in window[role]] == rows
            for role, rows in (('learning', learning), ('reference', reference)))
        if (any(window.get(key) != value for key, value in expected.items())
                or not roles_match
                or set(expected['learning_record_ids']) & set(expected['reference_record_ids'])):
            raise ValueError('Reference schedule differs from the retained same-pass next pair')


class KRCLateContinuationTrainer(SpecForgeEventTrainer):
    """Strictly restore KRC2, then allow only two continuous events to step512."""

    def __init__(self, config_path, output_root, *, resume_state,
                 reduction='response_mean', objective='osd_first_error'):
        if resume_state is None:
            raise ValueError('KRC late continuation requires its complete KRC2 recovery state')
        if objective != 'osd_first_error' or reduction != 'response_mean':
            raise ValueError('KRC late continuation keeps the qualified K objective and reduction')
        root = Path(output_root)
        if root.exists() and any(root.iterdir()):
            raise ValueError('KRC late continuation requires a fresh output directory')
        self._krc_late_event = None
        super().__init__(config_path, output_root, resume_state=resume_state,
                         reduction=reduction, objective=objective)
        if (self.global_step != RESUME_STEP or self.presentations != 2 * RESUME_STEP
                or self.accumulation != 2 or self.core.accumulation_remainder):
            raise ValueError('KRC late continuation must restore the complete event-2 boundary')
        _validate_history(self, 2)

    def _validate_config(self):
        super()._validate_config()
        if self.horizon != HORIZON:
            raise ValueError('KRC late continuation preserves the inherited 7424-step horizon')

    def _build_runtime(self):
        super()._build_runtime()
        # The inherited strict loader compares the entire saved contract after
        # this hook. No KRC1 probe migration or correction-removal state is accepted.
        self.contract['reference_correction_event'] = krc2_reference_contract()

    def _prepare_event(self, event_id, feature_paths, passes):
        manager = self._krc_late_event
        if manager is None or manager.closed or manager.status != 'running':
            raise ValueError('Install the KRC late continuation controller before training')
        number = len(self.events) + 1
        if (number not in (3, 4) or event_id != f'event_{number:03d}'
                or self.global_step != (number - 1) * EVENT_WINDOWS
                or self.global_step != manager.start_step
                or self.presentations != 2 * self.global_step
                or len(feature_paths) != EVENT_WINDOWS or type(passes) is not int or passes != 2):
            raise ValueError('Only consecutive sealed events 3 and 4 are authorized')
        _validate_history(self, number - 1)
        return super()._prepare_event(event_id, feature_paths, passes)

    def train_event(self, event_id, feature_paths, passes=2, *, reference_check=None,
                    presentation_schedule=None):
        if reference_check is not None:
            raise ValueError('Optional learning and post-correction probes remain disabled')
        if presentation_schedule is None:
            raise ValueError('The retained ordinary presentation schedule is required')
        manager = self._krc_late_event
        if manager is None or manager.closed or manager.status != 'running':
            raise ValueError('Install the KRC late continuation controller before training')
        ordinary = presentation_schedule
        if isinstance(ordinary, (str, Path)):
            ordinary = json.loads(Path(ordinary).read_text())
        _validate_schedule_binding(manager.schedule, ordinary, event_id, self.global_step)
        receipt = super().train_event(event_id, feature_paths, passes,
                                      presentation_schedule=ordinary)
        # The inherited serializer writes a new pending inode and os.replace's
        # latest. Thus this hard link remains a complete immutable event state
        # when event4 replaces latest; it consumes no second tensor copy.
        started = time.perf_counter()
        event_dir = self.events_root / event_id
        retained_state = event_dir / 'training_state.pt'
        try:
            os.link(self.latest, retained_state)
            receipt['full_state_path'] = str(retained_state)
            receipt['latest_state_path'] = str(self.latest)
            receipt['costs']['state_retention_s'] = time.perf_counter() - started
            receipt['state_retention'] = 'hardlink of complete atomic state; latest is replaced, never mutated'
            write_json(event_dir / 'RESULT.json', receipt)
        except BaseException as error:
            self.failed = True
            write_json(event_dir / 'STATE_RETENTION_FAILED.json', {
                'error_type': type(error).__name__, 'message': str(error),
                'global_step': self.global_step, 'presentations': self.presentations,
                'latest_state_path': str(self.latest), 'retained_state_path': str(retained_state),
                'automatic_retry': False,
            })
            raise
        return receipt


class KRCLateContinuationEvent(KRCContinuationEvent):
    """Use the qualified controller at event-local1..128 and global257..512."""

    def __init__(self, trainer, *, schedule, feature_paths, output_dir=None):
        previous = deepcopy(trainer.contract.get('reference_correction_event'))
        if previous != krc2_reference_contract():
            raise ValueError('Validate the exact restored no-probe KRC2 contract')
        if (trainer.global_step not in (RESUME_STEP, RESUME_STEP + EVENT_WINDOWS)
                or trainer.presentations != 2 * trainer.global_step
                or len(schedule['windows']) != EVENT_WINDOWS):
            raise ValueError('KRC late continuation boundary or local window count differs')
        if getattr(trainer, '_krc_late_event', None) is not None:
            raise ValueError('A KRC late continuation controller is already installed')
        # Skip the old event-2 constructor, whose strict128->256 guards remain
        # unchanged. All scientific operations still use its qualified parent.
        ReferenceCorrectionEvent.__init__(
            self, trainer, schedule=schedule, feature_paths=feature_paths,
            output_dir=output_dir, probe_steps=(), first_window_check=None)
        if self.contract != previous:
            raise ValueError('Late continuation changed the qualified KRC2 contract')
        self.migration = {
            'scope': 'no algorithm or diagnostic migration from qualified KRC2',
            'before': previous, 'after': deepcopy(self.contract),
            'changed_fields': [], 'reseeded': False,
            'normal_learning_sampling': 'draw normally and require retained event-3/4 schedules',
        }
        self.first_window = None
        self.first_window_gate_s = 0.0
        self.optimizer_identity = id(trainer.backend.optimizer.optimizer)
        if self.output_dir is not None:
            write_json(self.output_dir / 'CONTINUATION_CONTRACT.json', self.migration)

    def __enter__(self):
        if getattr(self.trainer, '_krc_late_event', None) is not None:
            raise ValueError('A KRC late continuation controller is already installed')
        ReferenceCorrectionEvent.__enter__(self)
        self.trainer._krc_late_event = self
        return self

    def _begin_window(self):
        if (self.trainer.global_step >= self.start_step + EVENT_WINDOWS
                or self.trainer.global_step >= FINAL_STEP):
            raise ValueError('The authorized KRC late continuation event is complete')
        ReferenceCorrectionEvent._begin_window(self)

    def _check_first_window(self):
        """Observe this event's actual first update, without extra model work."""
        expected_work = {
            'original_response_forwards': 2, 'pre_reference_response_forwards': 2,
            'post_k_reference_response_forwards': 2, 'post_c_reference_response_forwards': 0,
            'post_k_learning_response_forwards': 0, 'post_c_learning_response_forwards': 0,
        }
        checks = {name: self.work.get(name, 0) == count for name, count in expected_work.items()}
        checks.update({
            'two_K_backwards': self.counts['K_backward_completed'] == 2,
            'one_native_Adam': self.counts['Adam_completed'] == 1,
            'two_presentations': self.counts['presentations_completed'] == 2,
            'one_complete_window': self.counts['windows_completed'] == 1,
            'bounded_reference_backwards': 0 <= self.work['violation_backward_completed'] <= 2,
            'no_failed_backward': self.counts['K_backward_failed'] == 0
                                  and self.work['violation_backward_failed'] == 0,
            'no_probes_or_live_recorder': not self.probes and self.active is None and self.capture is None,
            'same_optimizer': id(self.trainer.backend.optimizer.optimizer) == self.optimizer_identity,
            'micro_counter': self.trainer.core._micro == 2 * (self.start_step + 1),
        })
        for phase in ('pre_reference_capture', 'post_K_replay', 'correction'):
            values = self.steps[-1]['integrity'].get(phase)
            checks[phase + '_preserved_state'] = bool(values) and all(values.values())
        counters = sorted({float(state['step'])
                           for state in self.trainer.backend.optimizer.optimizer.state.values()})
        checks['continued_Adam_counters'] = counters == [float(self.start_step + 1)]
        self.first_window = {
            'status': 'passed' if all(checks.values()) else 'failed', 'checks': checks,
            'event_local_step': 1, 'global_step_before': self.start_step,
            'global_step_after_native_Adam': self.start_step + 1,
            'optimizer_counters': counters, 'counts': dict(self.counts), 'work': dict(self.work),
            'extra_optimizer_steps': 0, 'extra_forwards': 0,
            'scope': 'actual first-window integrity and accounting; no historical loss or weight reproduction claim',
        }
        if self.output_dir is not None:
            write_json(self.output_dir / 'FIRST_WINDOW.json', self.first_window)
        if not all(checks.values()):
            failed = [name for name, passed in checks.items() if not passed]
            raise ValueError('KRC late continuation first-window check failed: ' + ', '.join(failed))

    def close(self):
        if not self.closed and getattr(self.trainer, '_krc_late_event', None) is self:
            self.trainer._krc_late_event = None
        ReferenceCorrectionEvent.close(self)
