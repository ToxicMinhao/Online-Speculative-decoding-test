"""Native N and unchanged C, sharing restoration, sampling and event publication."""
from pathlib import Path
import time

from experiments.bounded_prefix_allocation.training import CandidateTrainer
from experiments.full_k_gsm8k import training as native
from experiments.mathqa_fixed_lr_transfer.training import MathQAFixedLRTrainer, FORBIDDEN_CONTROLLERS
from experiments.ok_coding.training import _export_equal
from experiments.ok_continuation.resume import differences
from experiments.role_loss_pilot import training as historical
from experiments.persistent_prefix_withholding.qualification import tree_identity
from online_feedback.kplus import install_kplus, KPlusRedistribution, kplus_contract
from online_feedback.rng import capture_rng, preserve_rng, restore_rng
from online_feedback.storage import write_json
from online_feedback.training import SpecForgeEventTrainer
from . import state as S


class PairedContinuationTrainer(MathQAFixedLRTrainer):
    """The fork happens after disposable startup, once at the first natural pass."""

    def __init__(self, config_path, output_root, *, dataset, arm, binding, origin_state,
                 origin_export, original_config_text, completed_event, origin_identity,
                 physical_gpu_uuid, resume_state=None, resume_export=None,
                 expected_cuda_rng_states=1, qualification_only=False):
        import torch
        first = S.ORIGIN_EVENT[dataset]
        if completed_event not in (first, first+1) or (completed_event > first) != (resume_state is not None):
            raise ValueError('Start from the dataset origin or resume the same arm first event')
        if (resume_state is None) != (resume_export is None):
            raise ValueError('Own recovery requires its verified matching serving export')
        self._arm, self.dataset, self.binding = arm, dataset, binding
        self.completed_event, self.original_config_text = completed_event, original_config_text
        self.expected_cuda_rng_states, self.qualification_only = expected_cuda_rng_states, qualification_only
        self.origin_path, self.origin_export = Path(origin_state), Path(origin_export)
        self.origin = torch.load(origin_state, map_location='cpu', weights_only=False, mmap=True)
        S.validate_origin(self.origin, original_config_text, dataset=dataset,
                          expected_cuda_rng_states=expected_cuda_rng_states)
        _export_equal(self.origin, origin_export)
        self.origin_identity, self.physical_gpu_uuid = origin_identity, physical_gpu_uuid
        self._source = self.origin if resume_state is None else torch.load(
            resume_state, map_location='cpu', weights_only=False, mmap=True)
        if resume_export is not None:
            _export_equal(self._source, resume_export)
        self.relative_cap = S.RELATIVE_CAP if arm == 'C' else None
        self._qualification_cap = self.relative_cap
        self._fork_applied = completed_event > first
        self.response_records, self.window_records, self.work, self._work_handles = [], [], {}, []
        self.step_observer = None
        SpecForgeEventTrainer.__init__(self, config_path, output_root,
            resume_state=resume_state or origin_state, objective='osd_first_error',
            reduction='response_mean', ranking_lambda=0.)
        self._source = None

    def _validate_config(self):
        SpecForgeEventTrainer._validate_config(self)
        native.validate_config_text(self.config_text, 'K', self.original_config_text)

    def _build_runtime(self):
        import torch
        SpecForgeEventTrainer._build_runtime(self)
        install_kplus(self.bundle.model)
        if differences(self.contract, S.native_base_contract(self.origin['algorithm_contract'])):
            raise ValueError('New wrapper changed the authenticated native training contract')
        device = next(self.bundle.draft_model.parameters()).device
        logical_index = torch.cuda.current_device()
        self._fork_rng = S.fork_rng_state(self.origin['rng'], device=device, logical_cuda_index=logical_index)
        self.lineage = S.fork_record(self.origin, self._fork_rng, dataset=self.dataset,
            origin_identity=self.origin_identity, physical_gpu_uuid=self.physical_gpu_uuid,
            logical_cuda_index=logical_index)
        self.contract = S.continuation_contract(self.origin, dataset=self.dataset, arm=self.arm,
                                               binding=self.binding, lineage=self.lineage)
        self._install_allocation()
        self._check_native_path()
        self._install_observations()
        self._ordinary_forward, self._ordinary_step = self._native_forward, self._native_step

        def checked_forward(*args, **kwargs):
            self._check_native_path()
            result = self._ordinary_forward(*args, **kwargs)
            if not bool(torch.isfinite(result.loss).all()):
                raise ValueError('Nonfinite paired native response loss')
            return result

        def checked_step():
            self._check_actual_lr()
            if self.step_observer is not None:
                if not self.qualification_only:
                    raise ValueError('Gradient inspection is disposable qualification only')
                self.step_observer(self)
            result = self._ordinary_step()
            self._check_actual_lr()
            return result

        self._native_forward, self._native_step = checked_forward, checked_step

    _install_allocation = CandidateTrainer._install_allocation
    _check_allocation = CandidateTrainer._check_allocation
    qualification_allocation = CandidateTrainer.qualification_allocation
    _append_observation = CandidateTrainer._append_observation

    def _check_native_path(self):
        model = self.bundle.model
        forbidden = tuple(name for name in FORBIDDEN_CONTROLLERS if name != 'prefix_allocation')
        if (not isinstance(getattr(model, 'first_error_redistribution', None), KPlusRedistribution)
                or any(getattr(model, name, None) is not None for name in forbidden)):
            raise ValueError('Only ordinary K+ and the unchanged C allocation are permitted')
        self._check_allocation()
        controller = getattr(model, 'prefix_allocation', None)
        if controller is not None:
            from online_feedback.prefix_allocation import prefix_allocation_contract
            if controller.contract != prefix_allocation_contract(controller.relative_cap):
                raise ValueError('Allocation controller contract changed')

    def _check_actual_lr(self):
        optimizer = self.backend.optimizer
        if (self.cfg.training.learning_rate != S.NATIVE_LR
                or optimizer.recipe['optimizer_options']['lr'] != S.NATIVE_LR
                or len(optimizer.optimizer.param_groups) != 1
                or optimizer.optimizer.param_groups[0]['lr'] != S.NATIVE_LR
                or optimizer.get_learning_rate() != S.NATIVE_LR):
            raise ValueError('Actual native 3e-4 learning rate changed')

    def validation_arguments(self):
        return dict(origin=self.origin, dataset=self.dataset, arm=self.arm, binding=self.binding,
            lineage=self.lineage, config_text=self.config_text, original_config_text=self.original_config_text,
            expected_recipe=self.backend.optimizer.recipe,
            expected_cuda_rng_states=self.expected_cuda_rng_states)

    def _initial_state(self):
        return S.fork_state(self.origin, dataset=self.dataset, arm=self.arm, binding=self.binding,
            lineage=self.lineage, config_text=self.config_text, run_id=self.cfg.run_id,
            original_config_text=self.original_config_text,
            expected_cuda_rng_states=self.expected_cuda_rng_states)

    def _load_scientific_state(self, state):
        self._install_allocation()
        MathQAFixedLRTrainer._load_scientific_state(self, state)

    def _restore(self, path):
        state = self._initial_state() if self.completed_event == S.ORIGIN_EVENT[self.dataset] else self._source
        S.validate_snapshot(state, **self.validation_arguments(), completed_event=self.completed_event)
        self._load_scientific_state(state)
        receipt = native.verify_live(self, state)
        receipt.update(origin_state=str(self.origin_path), resume_state=str(path), dataset=self.dataset,
            arm=self.arm, continuation_rng_fork=self.lineage,
            fork_pending=not self._fork_applied, optimizer_reset=False)
        write_json(self.output_root/'RESUME_CHECK.json', receipt)

    def apply_continuation_fork(self):
        """No constructors, warmup or metadata loads occur after this live check."""
        if self._fork_applied:
            return dict(status='already_applied', reseeded=False)
        if self.completed_event != S.ORIGIN_EVENT[self.dataset] or self.core.accumulation_remainder:
            raise ValueError('Continuation RNG may change only at the restored origin boundary')
        if differences(capture_rng(), self.origin['rng']):
            raise ValueError('Startup did not restore the original RNG before the declared fork')
        restore_rng(self._fork_rng)
        if differences(capture_rng(), self._fork_rng):
            raise ValueError('Patched RNG was not live before native sampling')
        self._fork_applied = True
        result = dict(status='passed', reseeded=True, **self.lineage)
        write_json(self.output_root/'CONTINUATION_RNG_FORK.json', result)
        return result

    def _prepare_event(self, event_id, feature_paths, passes):
        first = S.ORIGIN_EVENT[self.dataset]
        if self.qualification_only or len(self.events)+1 not in (first+1, first+2):
            raise ValueError('Production is restricted to the two authorized paired events')
        return native.FullKTrainer._prepare_event(self, event_id, feature_paths, passes)

    def _train_passes(self, features, passes):
        self.apply_continuation_fork()
        before = tree_identity(capture_rng())
        rows = historical.RoleLossTrainer._train_passes(self, features, passes)
        write_json(self.events_root/self._active_event/'NATURAL_SAMPLING.json', dict(
            status='passed', rng_before=before, rng_after=tree_identity(capture_rng()),
            historical_schedule_forced=False, fork_reapplied=False,
            pass_orders=[[row['record_id'] for row in rows if row['pass']==p] for p in range(passes)],
            anchor_receipts='event.json records[*].sampled_anchors'))
        return rows

    def _checkpoint_state(self, event):
        if not self._fork_applied:
            raise ValueError('Cannot publish a trajectory without applying its declared fork')
        event.update(method=self.arm, dataset=self.dataset, kplus_loss=kplus_contract(),
                     **{S.CONTRACT_KEY:self.contract[S.CONTRACT_KEY]})
        S.validate_event_receipt(event, dataset=self.dataset, arm=self.arm, binding=self.binding,
            lineage=self.lineage, event_number=len(self.events)+1, require_completed=False)
        return SpecForgeEventTrainer._checkpoint_state(self, event)

    def train_event(self, event_id, feature_paths, passes=2, *, reference_check=None, presentation_schedule=None):
        if reference_check is not None or presentation_schedule is not None:
            raise ValueError('Paired continuation uses natural sampling, without forced schedules')
        result = SpecForgeEventTrainer.train_event(self, event_id, feature_paths, passes)
        S.validate_event_receipt(result, dataset=self.dataset, arm=self.arm, binding=self.binding,
            lineage=self.lineage, event_number=len(self.events), trainer_status=self.status())
        self.completed_event = len(self.events)
        return result

    def verify_boundary(self, export_path=None, receipt_path=None):
        import torch
        began = time.perf_counter()
        with preserve_rng():
            saved = torch.load(self.latest, map_location='cpu', weights_only=False, mmap=True)
            result = S.validate_snapshot(saved, **self.validation_arguments(), completed_event=len(self.events))
            if export_path is not None:
                _export_equal(saved, export_path)
            result.update(live=native.verify_live(self, saved), export_exact=export_path is not None)
        result['wall_s'] = time.perf_counter()-began
        if receipt_path is not None:
            write_json(receipt_path, result)
        return result

    def reset_qualification(self, *, relative_cap=.25, runtime=None):
        if not self.qualification_only:
            raise ValueError('Disposable reset is forbidden in production')
        self.qualification_allocation(relative_cap)
        self._clear_qualification_accumulation()
        self._load_scientific_state(self._initial_state())
        self.completed_event = S.ORIGIN_EVENT[self.dataset]
        self._fork_applied = False
        self.response_records.clear()
        self.window_records.clear()
        self._anchor_observations.clear()
        self._presentation_schedule = None
        if runtime is not None:
            from experiments.functional_step_geometry.qualify import restore_runtime
            restore_runtime(self.bundle.model, runtime)
        return dict(status='passed', fork_pending=True, extra_optimizer_steps=0)

    def save_qualification(self, path):
        if not self._fork_applied:
            raise ValueError('Temporary endpoint must include the live continuation fork')
        return CandidateTrainer.save_qualification(self, path)

    def restore_qualification(self, path, *, runtime=None):
        result = CandidateTrainer.restore_qualification(self, path, runtime=runtime)
        self._fork_applied = True
        return result
