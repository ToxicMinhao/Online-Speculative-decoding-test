"""Add events 5–6 after validating an unchanged PR32 KPLUS4 recovery.

The old pilot validator still accepts only events 0–4. This wrapper first
authenticates that historical boundary, preserves its receipts, and validates
new events under a separate additive contract. All ordinary training stays in
the existing K+ trainer and native optimizer.
"""
from copy import deepcopy
import math
from pathlib import Path
import time

from experiments.full_k_gsm8k import training as native
from experiments.ok_coding.training import _export_equal
from experiments.ok_continuation.resume import differences
from experiments.role_loss_pilot import training as historical
from online_feedback.kplus import install_kplus, kplus_contract
from online_feedback.rng import preserve_rng, restore_rng
from online_feedback.storage import write_json
from online_feedback.training import SpecForgeEventTrainer
from .contract import ARMS, DATASETS, EVENTS, SCHEMA, LEGACY_REALIZATION, experiment_contract

CONTRACT_KEY = SCHEMA
OBJECTIVE = 'osd_first_error'


def validate_origin(state, *, dataset, original_config_text, expected_cuda_rng_states=1):
    """Require the original complete KPLUS4 state, never weights-only recovery."""
    if dataset not in DATASETS:
        raise ValueError('Unknown continuation dataset')
    old = state['algorithm_contract'].get(historical.CONTRACT_KEY, {})
    if (old.get('arm') != 'KPLUS' or old.get('dataset') != dataset
            or CONTRACT_KEY in state['algorithm_contract']):
        raise ValueError('Origin must be this dataset\'s PR32 KPLUS4 state')
    return historical.validate_snapshot(state, arm='KPLUS', dataset=dataset,
        binding=old['binding'], config_text=state['config_text'],
        original_config_text=original_config_text, completed_event=4,
        expected_contract=state['algorithm_contract'],
        expected_recipe=state['backend']['optimizer']['recipe'],
        expected_cuda_rng_states=expected_cuda_rng_states)


def continuation_contract(origin, *, binding, arm, dataset, realization_revision=LEGACY_REALIZATION):
    old = origin['algorithm_contract'][historical.CONTRACT_KEY]
    result = deepcopy(origin['algorithm_contract'])
    result[CONTRACT_KEY] = experiment_contract(binding, arm, dataset, origin_binding=old['binding'],
        realization_revision=realization_revision)
    return result


def validate_event_receipt(receipt, *, arm, dataset, event_number,
                           require_completed=True, trainer_status=None):
    if (arm not in ARMS or dataset not in DATASETS or event_number not in EVENTS
            or receipt.get('method') != arm or receipt.get('dataset') != dataset
            or receipt.get('kplus_loss') != kplus_contract()):
        raise ValueError('New receipt must identify the declared FSG continuation arm and event')
    checked = native.validate_event_receipt(receipt, arm='K', event_number=event_number,
        require_completed=require_completed, trainer_status=trainer_status)
    for row in receipt['records']:
        observation = row['response_observation']
        sums = observation['sum_metrics']
        if (observation['record_id'] != row['record_id'] or observation['pass_index'] != row['pass']
                or sums['kplus_response_count'] != 1
                or not all(math.isfinite(float(v)) for v in sums.values())):
            raise ValueError('Ordinary K+ observations changed or were counted more than once')
        total = sums['kplus_W1'] + sums['kplus_W2']
        tolerance = kplus_contract()['mass_conservation_atol'] + kplus_contract()['mass_conservation_rtol'] * total
        if abs(sums['kplus_first_error_weight_after'] - total) > tolerance:
            raise ValueError('Native K+ first-error mass conservation failed')
    return dict(checked, arm=arm, dataset=dataset)


def validate_snapshot(state, *, origin, binding, arm, dataset, config_text,
                      original_config_text, completed_event, expected_recipe,
                      expected_cuda_rng_states=1, realization_revision=LEGACY_REALIZATION):
    """Validate new boundaries with an explicitly checked metadata-only view.

    This does not turn event 5 into an old event 4, alter scientific tensors, or
    modify historical receipts. The existing 58-event native validator checks
    moments, sampling records, finite tensors, RNG and actual global counters.
    """
    if type(completed_event) is not int or not 4 <= completed_event <= 6:
        raise ValueError('FSG complete boundary must be event 4, 5 or 6')
    expected = continuation_contract(origin, binding=binding, arm=arm, dataset=dataset,
        realization_revision=realization_revision)
    if differences(state['algorithm_contract'], expected):
        raise ValueError('Continuation scientific contract differs')
    if differences(state['events'][:4], origin['events']):
        raise ValueError('Retained PR32 event receipts were changed')
    old_binding = origin['algorithm_contract'][historical.CONTRACT_KEY]['binding']
    old_native = historical.native_contract_view(origin['algorithm_contract'], old_binding, 'KPLUS', dataset)
    view = dict(state, algorithm_contract=old_native)
    checked = native.validate_snapshot(view, arm='K', binding=old_binding,
        config_text=config_text, original_config_text=original_config_text,
        completed_event=completed_event, expected_contract=old_native,
        expected_recipe=expected_recipe, expected_cuda_rng_states=expected_cuda_rng_states)
    for number, receipt in enumerate(state['events'][4:], 5):
        validate_event_receipt(receipt, arm=arm, dataset=dataset,
            event_number=number, require_completed=False)
    return dict(checked, arm=arm, dataset=dataset, historical_receipts_unchanged=True,
                start_adam_step=512, maximum_adam_step=768)


def migrate_origin(origin, *, binding, arm, dataset, config_text, run_id,
                   original_config_text, expected_recipe, expected_cuda_rng_states=1,
                   realization_revision=LEGACY_REALIZATION):
    """Change only run metadata after validating the original complete recovery."""
    validate_origin(origin, dataset=dataset, original_config_text=original_config_text,
                    expected_cuda_rng_states=expected_cuda_rng_states)
    state = dict(origin, config_text=config_text, run_id=run_id,
        algorithm_contract=continuation_contract(origin, binding=binding, arm=arm, dataset=dataset,
            realization_revision=realization_revision))
    validate_snapshot(state, origin=origin, binding=binding, arm=arm, dataset=dataset,
        config_text=config_text, original_config_text=original_config_text,
        completed_event=4, expected_recipe=expected_recipe,
        expected_cuda_rng_states=expected_cuda_rng_states, realization_revision=realization_revision)
    # Tensor, Adam and RNG payloads are deliberately shared read-only views until
    # loaded into the new runtime. No tensor arithmetic is part of migration.
    return state


class FunctionalGeometryTrainer(native.FullKTrainer):
    """One continuous events-5–6 trajectory, with an optional external FSG hook."""

    def __init__(self, config_path, output_root, *, arm, dataset, binding,
                 original_config_text, origin_state, origin_export,
                 resume_state=None, completed_event=4, expected_cuda_rng_states=1,
                 realization_revision=LEGACY_REALIZATION):
        import torch
        if (arm not in ARMS or dataset not in DATASETS or completed_event not in (4, 5, 6)
                or (completed_event > 4) != (resume_state is not None)):
            raise ValueError('Require KPLUS4 origin, or own complete FSG event-5/6 recovery')
        self.arm, self.dataset, self.binding = arm, dataset, binding
        self.realization_revision = realization_revision
        self.original_config_text = original_config_text
        self.expected_cuda_rng_states = expected_cuda_rng_states
        self.completed_event = completed_event
        self.origin_path, self.origin_export = Path(origin_state), Path(origin_export)
        self.origin = torch.load(self.origin_path, map_location='cpu', weights_only=False, mmap=True)
        validate_origin(self.origin, dataset=dataset, original_config_text=original_config_text,
                        expected_cuda_rng_states=expected_cuda_rng_states)
        _export_equal(self.origin, self.origin_export)
        self.response_records, self.window_records, self.work, self._work_handles = [], [], {}, []
        SpecForgeEventTrainer.__init__(self, config_path, output_root,
            resume_state=resume_state or origin_state, objective=OBJECTIVE,
            reduction='response_mean', ranking_lambda=0.)

    def _validate_config(self):
        SpecForgeEventTrainer._validate_config(self)
        native.validate_config_text(self.config_text, 'K', self.original_config_text)

    def _build_runtime(self):
        SpecForgeEventTrainer._build_runtime(self)
        install_kplus(self.bundle.model)
        source_base = dict(self.origin['algorithm_contract'])
        del source_base[historical.CONTRACT_KEY]
        if differences(self.contract, source_base):
            raise ValueError('Constructed native runtime differs from retained KPLUS4')
        self.contract = continuation_contract(self.origin, binding=self.binding, arm=self.arm, dataset=self.dataset,
            realization_revision=self.realization_revision)
        self._install_observations()
        ordinary_forward, ordinary_step = self._native_forward, self._native_step

        def finite_forward(*args, **kwargs):
            import torch
            result = ordinary_forward(*args, **kwargs)
            if not bool(torch.isfinite(result.loss).all()):
                raise ValueError('Nonfinite ordinary K+ loss before optimizer application')
            return result

        def finite_step():
            from experiments.adaflash_arepo.qualify import finite_gradient_check
            finite_gradient_check({n:p for n,p in self.bundle.draft_model.named_parameters() if p.requires_grad},
                                  self.output_root, self.presentations)
            return ordinary_step()

        self._native_forward, self._native_step = finite_forward, finite_step
        observed_end = self.bundle.model.end_event

        def end_event():
            result = observed_end()
            hook = getattr(self, 'functional_geometry', None)
            if hook is not None:
                report = hook.report()
                result['functional_geometry'] = report
                result['reference_operations'] = report['work'].get('additional_response_forwards_completed', 0)
            return result
        self.bundle.model.end_event = end_event

    def validation_arguments(self):
        return dict(origin=self.origin, binding=self.binding, arm=self.arm, dataset=self.dataset,
            config_text=self.config_text, original_config_text=self.original_config_text,
            expected_recipe=self.backend.optimizer.recipe,
            expected_cuda_rng_states=self.expected_cuda_rng_states,
            realization_revision=self.realization_revision)

    def _restore(self, path):
        import torch
        if self.completed_event == 4:
            state = migrate_origin(self.origin, binding=self.binding, arm=self.arm, dataset=self.dataset,
                config_text=self.config_text, run_id=self.cfg.run_id,
                original_config_text=self.original_config_text,
                expected_recipe=self.backend.optimizer.recipe,
                expected_cuda_rng_states=self.expected_cuda_rng_states,
                realization_revision=self.realization_revision)
        else:
            state = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
            validate_snapshot(state, **self.validation_arguments(), completed_event=self.completed_event)
        self.bundle.draft_model.load_state_dict(state['draft_state_dict'], strict=True)
        # AdamW may reuse same-device checkpoint tensor storage. In particular,
        # its CPU step counters would otherwise mutate the retained origin on
        # the first GPU step; CPU execution can also alias the BF16 moments.
        # Give the live optimizer its own complete payload before loading it.
        self.backend.load_state_dict(deepcopy(state['backend']))
        self.global_step, self.presentations = state['global_step'], state['presentations']
        self.events = deepcopy(state['events'])
        self.core._micro = self.presentations
        restore_rng(state['rng'])
        receipt = native.verify_live(self, state)
        receipt.update(origin_state=str(self.origin_path), origin_export=str(self.origin_export),
            historical_receipts_unchanged=True, source_step=512, seed_reset=False,
            changes=['additive FSG contract', 'run/output metadata'])
        write_json(self.output_root/'RESUME_CHECK.json', receipt)

    def save_initial(self, *args, **kwargs):
        raise ValueError('FSG resumes complete KPLUS4; it has no released initialization path')

    def _prepare_event(self, event_id, feature_paths, passes):
        if len(self.events) + 1 not in EVENTS:
            raise ValueError('FSG is limited to events 5 and 6')
        hook = getattr(self, 'functional_geometry', None)
        if self.arm != 'KPLUS' and hook is None:
            raise ValueError('Install the declared geometry hook before candidate training')
        if self.arm == 'KPLUS' and hook is not None:
            raise ValueError('The production KPLUS control has no geometry hook')
        if hook is not None:
            if (hook.arm != self.arm or hook.strength != 1.0 or hook.closed
                    or hook.realization_revision != self.realization_revision):
                raise ValueError('Production requires its own active geometry arm at fixed strength 1')
            hook.clear_event()
        return native.FullKTrainer._prepare_event(self, event_id, feature_paths, passes)

    def train_event(self, event_id, feature_paths, passes=2, *, reference_check=None, presentation_schedule=None):
        if reference_check is not None:
            raise ValueError('Use only the prespecified isolated geometry diagnostics')
        result = SpecForgeEventTrainer.train_event(self, event_id, feature_paths, passes,
                                                  presentation_schedule=presentation_schedule)
        validate_event_receipt(result, arm=self.arm, dataset=self.dataset,
            event_number=len(self.events), trainer_status=self.status())
        self.completed_event = len(self.events)
        return result

    def _train_passes(self, features, passes):
        return historical.RoleLossTrainer._train_passes(self, features, passes)

    def _checkpoint_state(self, event):
        event.update(method=self.arm, dataset=self.dataset, kplus_loss=kplus_contract())
        validate_event_receipt(event, arm=self.arm, dataset=self.dataset,
            event_number=len(self.events)+1, require_completed=False)
        return SpecForgeEventTrainer._checkpoint_state(self, event)

    def verify_boundary(self, export_path=None, receipt_path=None):
        import torch
        started = time.perf_counter()
        with preserve_rng():
            state = torch.load(self.latest, map_location='cpu', weights_only=False, mmap=True)
            receipt = validate_snapshot(state, **self.validation_arguments(), completed_event=len(self.events))
            if export_path is not None:
                _export_equal(state, export_path)
                receipt['export_exact'] = True
            receipt['live'] = native.verify_live(self, state)
        receipt['wall_s'] = time.perf_counter() - started
        if receipt_path is not None:
            write_json(receipt_path, receipt)
        return receipt

    def close(self):
        if not self.closed:
            hook = getattr(self, 'functional_geometry', None)
            if hook is not None:
                hook.close()
        return native.FullKTrainer.close(self)
