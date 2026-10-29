"""Additive, event-4-only restoration for the bounded mechanism diagnostic.

The historical KRC2-only late-continuation constructor remains unchanged.  This
wrapper admits exactly the independently retained KRC3 state and event 4; all
scientific training, correction, schedule checking and persistence use the
qualified parent implementations.
"""
from pathlib import Path

from .krc_late_continuation import (
    KRCLateContinuationEvent, KRCLateContinuationTrainer, _validate_history,
)
from .training import SpecForgeEventTrainer


class KRCMechanismEvent4Trainer(KRCLateContinuationTrainer):
    def __init__(self, config_path, output_root, *, resume_state,
                 reduction='response_mean', objective='osd_first_error'):
        if resume_state is None:
            raise ValueError('Event-4 diagnostic requires the complete own KRC3 state')
        if objective != 'osd_first_error' or reduction != 'response_mean':
            raise ValueError('Event-4 diagnostic preserves the qualified K objective and reduction')
        root = Path(output_root)
        if root.exists() and any(root.iterdir()):
            raise ValueError('Event-4 diagnostic requires a fresh output directory')
        self._krc_late_event = None
        # Deliberately do not call the KRC2-only constructor. Its guards are
        # historical evidence and are not relaxed to admit a new boundary.
        SpecForgeEventTrainer.__init__(self, config_path, output_root,
            resume_state=resume_state, reduction=reduction, objective=objective)
        if (self.global_step != 384 or self.presentations != 768
                or self.accumulation != 2 or self.core.accumulation_remainder):
            raise ValueError('Event-4 diagnostic must restore the complete KRC3 boundary at 384/768')
        _validate_history(self, 3)

    def _prepare_event(self, event_id, feature_paths, passes):
        if (event_id != 'event_004' or self.global_step != 384
                or self.presentations != 768 or len(self.events) != 3):
            raise ValueError('Event-4 diagnostic permits only event 4 from independent KRC3')
        return super()._prepare_event(event_id, feature_paths, passes)


class KRCMechanismEvent4Base(KRCLateContinuationEvent):
    """Qualified event controller with the strict KRC3-only entry boundary."""

    def __init__(self, trainer, *, schedule, feature_paths, output_dir=None):
        if trainer.global_step != 384 or trainer.presentations != 768:
            raise ValueError('Event-4 diagnostic controller requires KRC3 at 384/768')
        _validate_history(trainer, 3)
        super().__init__(trainer, schedule=schedule, feature_paths=feature_paths,
                         output_dir=output_dir)
