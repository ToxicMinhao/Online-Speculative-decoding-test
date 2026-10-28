"""Continue the qualified KRC state for one event, without optional probes.

Restoration uses the original trainer's strict contract check. Only after that
check does the event controller disable the diagnostic forwards; ordinary K
sampling, the optimizer and the reference correction keep their existing paths.
"""
from copy import deepcopy
import time

from .reference_correction_event import ReferenceCorrectionEvent
from .storage import write_json
from .training import SpecForgeEventTrainer


RESUME_STEP = 128
EVENT_WINDOWS = 128


def krc1_reference_contract():
    """The saved KRC1 contract, including its original diagnostic schedule."""
    return {
        'arm': 'KRC', 'version': 1, 'base_objective': 'osd_first_error',
        'capture': 'current_reference_strict_prefix',
        'weights': 'descending_prefix_length',
        'violation': 'sum(weight*relu(-native_margin))/global_two_response_weight_sum',
        'correction': 'one_post_K_negative_V_over_gradient_squared_norm_step',
        'decision_precision': 'float32', 'parameter_cast': 'bfloat16',
        'optimizer': 'unchanged_native_K_AdamW', 'probe_steps': [1, 64, 128],
        'epsilon': None, 'extra_optimizer_steps': 0,
        'reference_selection': 'next_disjoint_pair_cyclic_within_pass',
        'replay': 'fixed_anchors_and_pre_reference_rng_cache_in_eval_mode',
        'reference_exposure': 'additional_training_exposure',
    }


class KRCContinuationTrainer(SpecForgeEventTrainer):
    """Restore KRC1 directly from its read-only state into a new output root."""

    def __init__(self, config_path, output_root, *, resume_state,
                 reduction='response_mean', objective='osd_first_error'):
        if resume_state is None:
            raise ValueError('KRC continuation requires its complete recovery state')
        if objective != 'osd_first_error' or reduction != 'response_mean':
            raise ValueError('KRC continuation keeps the qualified K objective and reduction')
        self._krc_continuation_active = False
        super().__init__(config_path, output_root, resume_state=resume_state,
                         reduction=reduction, objective=objective)
        if (self.global_step != RESUME_STEP
                or self.presentations != 2 * RESUME_STEP
                or self.accumulation != 2 or self.core.accumulation_remainder):
            raise ValueError('KRC continuation must restore the complete event-1 boundary')
        # The atomic checkpoint stores its event before the external RESULT
        # receipt adds status='completed'. The campaign binds that receipt and
        # exported weights separately; here validate the saved event boundary.
        if (len(self.events) != 1 or self.events[0]['event_id'] != 'event_001'
                or self.events[0]['start_step'] != 0
                or self.events[0]['end_step'] != RESUME_STEP
                or self.events[0]['optimizer_steps'] != RESUME_STEP
                or self.events[0]['response_presentations'] != 2 * RESUME_STEP):
            raise ValueError('KRC continuation requires the completed KRC1 event history')

    def _build_runtime(self):
        super()._build_runtime()
        # Parent __init__ invokes the unchanged _restore only after this method.
        # This is the old contract, not a relaxation accepting arbitrary state.
        self.contract['reference_correction_event'] = krc1_reference_contract()

    def _prepare_event(self, event_id, feature_paths, passes):
        if not self._krc_continuation_active:
            raise ValueError('Install the KRC continuation controller before training')
        if (event_id != 'event_002' or self.global_step != RESUME_STEP
                or len(feature_paths) != EVENT_WINDOWS or passes != 2):
            raise ValueError('Only the sealed event-2 pool and two-pass continuation are authorized')
        return super()._prepare_event(event_id, feature_paths, passes)


class KRCContinuationEvent(ReferenceCorrectionEvent):
    """Run local windows 1..128 at global steps 129..256, with no probes."""

    def __init__(self, trainer, *, schedule, feature_paths, output_dir=None):
        previous = deepcopy(trainer.contract.get('reference_correction_event'))
        if previous != krc1_reference_contract():
            raise ValueError('Validate the exact restored KRC1 contract before disabling probes')
        if (trainer.global_step != RESUME_STEP or trainer.presentations != 2 * RESUME_STEP
                or len(schedule['windows']) != EVENT_WINDOWS):
            raise ValueError('KRC continuation boundary or local window count differs')
        super().__init__(trainer, schedule=schedule, feature_paths=feature_paths,
                         output_dir=output_dir, probe_steps=(), first_window_check=None)
        self.migration = {
            'scope': 'diagnostic-only change after exact KRC1 restoration',
            'before': previous, 'after': deepcopy(self.contract),
            'changed_fields': ['probe_steps'], 'reseeded': False,
            'normal_learning_sampling': 'draw normally and require the retained event-2 schedule',
        }
        expected = deepcopy(previous)
        expected['probe_steps'] = []
        if self.contract != expected:
            raise ValueError('Continuation changed more than the optional probe schedule')
        self.first_window = None
        self.first_window_gate_s = 0.0
        self.optimizer_identity = id(trainer.backend.optimizer.optimizer)
        if self.output_dir is not None:
            write_json(self.output_dir / 'DIAGNOSTIC_MIGRATION.json', self.migration)

    def __enter__(self):
        super().__enter__()
        self.trainer._krc_continuation_active = True
        return self

    def _begin_window(self):
        if self.trainer.global_step >= RESUME_STEP + EVENT_WINDOWS:
            raise ValueError('The authorized KRC continuation event is complete')
        super()._begin_window()

    def _record_window(self, status, error=None):
        # Attach local/global coordinates before the inherited recorder writes
        # its detailed evidence and frees all model-sized temporary buffers.
        if self.active is not None:
            self.window_integrity['pre_reference_capture'] = self.active.pre_integrity
            self.window_integrity['step_coordinates'] = {
                'event_local_step': self.window_step,
                'global_step_before': self.start_step + self.window_step - 1,
                'global_step_after_native_Adam': self.start_step + self.counts['Adam_completed'],
            }
        super()._record_window(status, error)

    def _finish_window(self):
        super()._finish_window()
        if self.window_step == 1:
            started = time.perf_counter()
            try:
                self._check_first_window()
            finally:
                self.first_window_gate_s += time.perf_counter() - started

    def _check_first_window(self):
        """Check work already executed, without a loss or historical-weight gate."""
        expected_work = {
            'original_response_forwards': 2,
            'pre_reference_response_forwards': 2,
            'post_k_reference_response_forwards': 2,
            'post_c_reference_response_forwards': 0,
            'post_k_learning_response_forwards': 0,
            'post_c_learning_response_forwards': 0,
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
            'micro_counter': self.trainer.core._micro == 2 * (RESUME_STEP + 1),
        })
        for phase in ('pre_reference_capture', 'post_K_replay', 'correction'):
            values = self.steps[-1]['integrity'].get(phase)
            checks[phase + '_preserved_state'] = bool(values) and all(values.values())
        states = self.trainer.backend.optimizer.optimizer.state.values()
        counters = sorted({float(state['step']) for state in states})
        checks['continued_Adam_counters'] = counters == [float(RESUME_STEP + 1)]
        self.first_window = {
            'status': 'passed' if all(checks.values()) else 'failed', 'checks': checks,
            'event_local_step': 1, 'global_step_before': RESUME_STEP,
            'global_step_after_native_Adam': RESUME_STEP + 1,
            'optimizer_counters': counters, 'counts': dict(self.counts), 'work': dict(self.work),
            'extra_optimizer_steps': 0, 'extra_forwards': 0,
            'scope': 'actual first-window integrity and accounting; no K2 loss or weight reproduction claim',
        }
        if self.output_dir is not None:
            write_json(self.output_dir / 'FIRST_WINDOW.json', self.first_window)
        if not all(checks.values()):
            failed = [name for name, passed in checks.items() if not passed]
            raise ValueError('KRC continuation first-window check failed: ' + ', '.join(failed))

    def report(self):
        report = super().report()
        report.update(start_step=self.start_step,
                      native_global_step=self.start_step + self.counts['Adam_completed'],
                      diagnostic_migration=deepcopy(self.migration),
                      first_window=deepcopy(self.first_window))
        report['costs']['first_window_gate_s'] = self.first_window_gate_s
        report['cost_scopes']['post_c_reference_s_plus_post_k_learning_s_plus_post_c_learning_s'] = (
            'disabled: zero diagnostic response forwards')
        report['cost_scopes']['integrity_s_plus_pre_reference_integrity_s_plus_first_learning_gate_s'] = (
            'state copies and preservation checks; no historical K1 reproduction gate')
        report['cost_scopes']['first_window_gate_s'] = 'accounting and first-window receipt; no model forward'
        return report

    def close(self):
        if not self.closed:
            self.trainer._krc_continuation_active = False
        super().close()
