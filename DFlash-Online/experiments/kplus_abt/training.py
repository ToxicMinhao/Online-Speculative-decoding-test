"""Add events 5–6 after validating an unchanged PR32 KPLUS4 recovery.

The old pilot validator still accepts only events 0–4. This wrapper first
authenticates that historical boundary, preserves its receipts, and validates
new events under a separate additive contract. All ordinary training stays in
the existing K+ trainer and native optimizer.
"""
from copy import deepcopy
import math
import json
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
from .contract import ARMS, DATASETS, EVENTS, SCHEMA, experiment_contract

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


def continuation_contract(origin, *, binding, arm, dataset):
    old = origin['algorithm_contract'][historical.CONTRACT_KEY]
    result = deepcopy(origin['algorithm_contract'])
    result[CONTRACT_KEY] = experiment_contract(binding, arm, dataset, origin_binding=old['binding'])
    return result


def validate_event_receipt(receipt, *, arm, dataset, event_number,
                           require_completed=True, trainer_status=None):
    if (arm not in ARMS or dataset not in DATASETS or event_number not in EVENTS
            or receipt.get('method') != arm or receipt.get('dataset') != dataset
            or receipt.get('kplus_loss') != kplus_contract()):
        raise ValueError('New receipt must identify the declared ABT continuation arm and event')
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
                      expected_cuda_rng_states=1):
    """Validate new boundaries with an explicitly checked metadata-only view.

    This does not turn event 5 into an old event 4, alter scientific tensors, or
    modify historical receipts. The existing 58-event native validator checks
    moments, sampling records, finite tensors, RNG and actual global counters.
    """
    if type(completed_event) is not int or not 4 <= completed_event <= 6:
        raise ValueError('ABT complete boundary must be event 4, 5 or 6')
    expected = continuation_contract(origin, binding=binding, arm=arm, dataset=dataset)
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
                   original_config_text, expected_recipe, expected_cuda_rng_states=1):
    """Change only run metadata after validating the original complete recovery."""
    validate_origin(origin, dataset=dataset, original_config_text=original_config_text,
                    expected_cuda_rng_states=expected_cuda_rng_states)
    state = dict(origin, config_text=config_text, run_id=run_id,
        algorithm_contract=continuation_contract(origin, binding=binding, arm=arm, dataset=dataset))
    validate_snapshot(state, origin=origin, binding=binding, arm=arm, dataset=dataset,
        config_text=config_text, original_config_text=original_config_text,
        completed_event=4, expected_recipe=expected_recipe,
        expected_cuda_rng_states=expected_cuda_rng_states)
    # Tensor, Adam and RNG payloads are deliberately shared read-only views until
    # loaded into the new runtime. No tensor arithmetic is part of migration.
    return state


class AdversarialBlockTrainer(native.FullKTrainer):
    """One continuous events-5–6 trajectory, with an isolated two-view training core."""

    def __init__(self, config_path, output_root, *, arm, dataset, binding,
                 original_config_text, origin_state, origin_export,
                 resume_state=None, completed_event=4, expected_cuda_rng_states=1):
        import torch
        if (arm not in ARMS or dataset not in DATASETS or completed_event not in (4, 5, 6)
                or (completed_event > 4) != (resume_state is not None)):
            raise ValueError('Require KPLUS4 origin, or own complete ABT event-5/6 recovery')
        self.arm, self.dataset, self.binding = arm, dataset, binding
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
        self.contract = continuation_contract(self.origin, binding=self.binding, arm=self.arm, dataset=self.dataset)
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
    def install_two_views(self, *, enabled=True, rho=.01, work_callback=None,
                          response_observer=None):
        """Install after discarded native startup, at an empty Adam boundary."""
        from online_feedback.adversarial_block_runtime import install_adversarial_block_core
        self.view_records = []
        self.response_observer = response_observer
        original = self.core
        self.core = install_adversarial_block_core(original, self.bundle.model,
            arm=self.arm if enabled else 'KPLUS', rho=rho,
            forward_loss=self._native_forward,
            response_key=lambda batch, ctx: (self.dataset, len(self.events)+1,
                len(self.response_records)//128, len(self.response_records)),
            observe_response=self._observe_views, work_callback=work_callback)
        self.two_view_installed = self.core is not original
        return self.two_view_installed

    def _observe_views(self, batch, output, diagnostic):
        """One logical presentation, although the loss used two real views."""
        def scalar(value):
            return float(value.detach().item()) if hasattr(value, 'detach') else float(value)
        index = len(self.response_records)
        snapshots = diagnostic.pop('snapshots')
        record = dict(scope='one_response_two_views', record_id=str(batch.sample_ids[0]),
            event_id=self._active_event, pass_index=index//128, presentation_ordinal=index,
            sampled_anchors=deepcopy(self._anchor_observations), loss=scalar(output.loss),
            ratio_metrics={key:[scalar(a),scalar(b)] for key,(a,b) in output.ratio_metrics.items()},
            sum_metrics={key:scalar(value) for key,value in output.sum_metrics.items()})
        self.response_records.append(record)
        self._append_observation('RESPONSE_OBSERVATIONS.jsonl', record)
        self.work['response_forwards'] = self.work.get('response_forwards', 0)+2
        radii = {key:snapshots[key].tolist() for key in ('gradient_norm', 'nominal_radius',
            'intended_norm', 'realized_norm', 'changed_slots', 'zero_gradient_blocks',
            'no_eligible_blocks', 'eligible_slots') if key in snapshots}
        row = dict(diagnostic, record_id=record['record_id'], radii=radii,
                   ordinary_presentation=index)
        self.view_records.append(row)
        self._append_observation('VIEW_OBSERVATIONS.jsonl', row)
        if self.response_observer is not None:
            self.response_observer(batch, output, snapshots, row)

    def validation_arguments(self):
        return dict(origin=self.origin, binding=self.binding, arm=self.arm, dataset=self.dataset,
            config_text=self.config_text, original_config_text=self.original_config_text,
            expected_recipe=self.backend.optimizer.recipe,
            expected_cuda_rng_states=self.expected_cuda_rng_states)

    def _restore(self, path):
        import torch
        if self.completed_event == 4:
            state = migrate_origin(self.origin, binding=self.binding, arm=self.arm, dataset=self.dataset,
                config_text=self.config_text, run_id=self.cfg.run_id,
                original_config_text=self.original_config_text,
                expected_recipe=self.backend.optimizer.recipe,
                expected_cuda_rng_states=self.expected_cuda_rng_states)
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
            changes=['additive ABT contract', 'run/output metadata'])
        write_json(self.output_root/'RESUME_CHECK.json', receipt)

    def save_initial(self, *args, **kwargs):
        raise ValueError('ABT resumes complete KPLUS4; it has no released initialization path')

    def _prepare_event(self, event_id, feature_paths, passes):
        if len(self.events) + 1 not in EVENTS:
            raise ValueError('ABT is limited to events 5 and 6')
        if self.arm != 'KPLUS' and not getattr(self, 'two_view_installed', False):
            raise ValueError('Install the declared two-view core after disposable startup')
        return native.FullKTrainer._prepare_event(self, event_id, feature_paths, passes)

    def train_event(self, event_id, feature_paths, passes=2, *, reference_check=None, presentation_schedule=None):
        if reference_check is not None:
            raise ValueError('Use only the prespecified isolated two-view diagnostics')
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
        return native.FullKTrainer.close(self)
