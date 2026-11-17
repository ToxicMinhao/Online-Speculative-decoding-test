"""Unchanged PR50 C on complete MathQA KPLUS4 -> C5 -> C6 state.

The objective is the original prefix-allocation module, installed after K+.
PR51 supplies ordinary training, sampling, observations and private optimizer
restoration; this adapter owns only C identity and its saved controller contract.
"""
from pathlib import Path
import time

from experiments.bounded_prefix_allocation.training import CandidateTrainer
from experiments.full_k_gsm8k import training as native
from experiments.mathqa_fixed_lr_transfer.training import MathQAFixedLRTrainer, FORBIDDEN_CONTROLLERS
from experiments.ok_coding.training import _export_equal
from experiments.ok_continuation.resume import differences
from online_feedback.kplus import install_kplus, kplus_contract, KPlusRedistribution
from online_feedback.prefix_allocation import prefix_allocation_contract
from online_feedback.rng import preserve_rng
from online_feedback.storage import write_json
from online_feedback.training import SpecForgeEventTrainer
from . import state as S


class MathQACandidateTrainer(MathQAFixedLRTrainer):
    """Persistent C; native/cap-zero switching exists only in disposable checks."""

    def __init__(self, config_path, output_root, *, arm="C", binding, origin_state,
                 origin_export, original_config_text, completed_event=4,
                 resume_state=None, resume_export=None, expected_cuda_rng_states=1,
                 qualification_only=False):
        import torch
        if completed_event not in (4, 5) or (completed_event == 5) != (resume_state is not None):
            raise ValueError("C starts at original KPLUS4 or resumes complete own C5")
        if (resume_state is None) != (resume_export is None):
            raise ValueError("C recovery requires its matching export")
        S.experiment_contract(arm, binding)
        self._arm, self.dataset, self.binding = arm, "mathqa", binding
        self.relative_cap = self._qualification_cap = S.RELATIVE_CAP
        self.original_config_text = original_config_text
        self.completed_event = completed_event
        self.expected_cuda_rng_states = expected_cuda_rng_states
        self.qualification_only = qualification_only
        self.origin_path, self.origin_export = Path(origin_state), Path(origin_export)
        self.origin = torch.load(origin_state, map_location="cpu", weights_only=False, mmap=True)
        S.validate_origin(self.origin, original_config_text, expected_cuda_rng_states=expected_cuda_rng_states)
        _export_equal(self.origin, origin_export)
        self._source = (self.origin if resume_state is None else
                        torch.load(resume_state, map_location="cpu", weights_only=False, mmap=True))
        if resume_export is not None:
            _export_equal(self._source, resume_export)
        self.response_records, self.window_records, self.work, self._work_handles = [], [], {}, []
        self.step_observer = None
        SpecForgeEventTrainer.__init__(self, config_path, output_root,
            resume_state=resume_state or origin_state, objective="osd_first_error",
            reduction="response_mean", ranking_lambda=0.)
        self._source = None
        write_json(self.output_root/"CANDIDATE_CONFIG.json", dict(
            native_config_text=self.config_text, candidate_contract=self.contract[S.CONTRACT_KEY]))

    def _validate_config(self):
        SpecForgeEventTrainer._validate_config(self)
        S.validate_config_text(self.config_text, self.arm, self.original_config_text)

    def _build_runtime(self):
        SpecForgeEventTrainer._build_runtime(self)
        install_kplus(self.bundle.model)
        if differences(self.contract, S.native_base_contract(self.origin["algorithm_contract"])):
            raise ValueError("Constructed C base differs from authenticated native MathQA runtime")
        self.contract = S.continuation_contract(self.origin, arm=self.arm, binding=self.binding)
        self._install_allocation()
        self._check_native_path()
        self._install_observations()
        self._ordinary_forward, self._ordinary_step = self._native_forward, self._native_step

        def checked_forward(*args, **kwargs):
            import torch
            self._check_native_path()
            result = self._ordinary_forward(*args, **kwargs)
            if not bool(torch.isfinite(result.loss).all()):
                raise ValueError("Nonfinite MathQA C response loss")
            return result

        def checked_step():
            self._check_actual_lr()
            if self.step_observer is not None:
                if not self.qualification_only:
                    raise ValueError("Gradient inspection is disposable qualification only")
                self.step_observer(self)
            result = self._ordinary_step()
            self._check_actual_lr()
            return result

        self._native_forward, self._native_step = checked_forward, checked_step

    def _check_native_path(self):
        model = self.bundle.model
        forbidden = tuple(name for name in FORBIDDEN_CONTROLLERS if name != "prefix_allocation")
        if (not isinstance(getattr(model, "first_error_redistribution", None), KPlusRedistribution)
                or any(getattr(model, name, None) is not None for name in forbidden)):
            raise ValueError("C requires native K+ with only its original allocation controller")
        self._check_allocation()
        controller = getattr(model, "prefix_allocation", None)
        if controller is not None and controller.contract != prefix_allocation_contract(controller.relative_cap):
            raise ValueError("C controller contract changed")

    def _check_actual_lr(self):
        optimizer = self.backend.optimizer
        if (self.cfg.training.learning_rate != S.NATIVE_LR
                or optimizer.recipe["optimizer_options"]["lr"] != S.NATIVE_LR
                or len(optimizer.optimizer.param_groups) != 1
                or optimizer.optimizer.param_groups[0]["lr"] != S.NATIVE_LR
                or optimizer.get_learning_rate() != S.NATIVE_LR):
            raise ValueError("C must preserve the actual native 3e-4 optimizer rate")

    # These are the exact PR50 controller/summary/disposable-state operations.
    # They contain no GSM8K or event-seven training dependency.
    _install_allocation = CandidateTrainer._install_allocation
    qualification_allocation = CandidateTrainer.qualification_allocation
    _check_allocation = CandidateTrainer._check_allocation
    _append_observation = CandidateTrainer._append_observation
    save_qualification = CandidateTrainer.save_qualification
    restore_qualification = CandidateTrainer.restore_qualification

    def pr51_native_forward(self, *args, **kwargs):
        """Use the unchanged PR51 native-path guard for the disposable baseline."""
        MathQAFixedLRTrainer._check_native_path(self)
        return self._ordinary_forward(*args, **kwargs)

    def validation_arguments(self):
        return dict(origin=self.origin, arm=self.arm, binding=self.binding,
            config_text=self.config_text, original_config_text=self.original_config_text,
            expected_recipe=self.backend.optimizer.recipe,
            expected_cuda_rng_states=self.expected_cuda_rng_states)

    def _load_scientific_state(self, state):
        # Restoring own C never drops the controller. The PR51 loader deep-copies
        # backend state before loading so CPU Adam counters cannot alias origin.
        self._install_allocation()
        MathQAFixedLRTrainer._load_scientific_state(self, state)

    def _restore(self, path):
        state, migration = self._source, None
        if self.completed_event == 4:
            state, migration = S.fork_state(state, arm=self.arm, binding=self.binding,
                config_text=self.config_text, run_id=self.cfg.run_id,
                original_config_text=self.original_config_text,
                expected_cuda_rng_states=self.expected_cuda_rng_states)
        S.validate_snapshot(state, **self.validation_arguments(), completed_event=self.completed_event)
        self._load_scientific_state(state)
        receipt = native.verify_live(self, state)
        receipt.update(origin_state=str(self.origin_path), origin_export=str(self.origin_export),
            resume_state=str(path), migration=migration, arm=self.arm,
            actual_learning_rate=S.NATIVE_LR, candidate_contract=self.contract[S.CONTRACT_KEY], seed_reset=False)
        write_json(self.output_root/"RESUME_CHECK.json", receipt)

    def _checkpoint_state(self, event):
        self._check_native_path()
        event.update(method="C", dataset="mathqa", kplus_loss=kplus_contract(),
                     **{S.CONTRACT_KEY: self.contract[S.CONTRACT_KEY]})
        S.validate_event_receipt(event, arm=self.arm, binding=self.binding,
            event_number=len(self.events)+1, require_completed=False)
        return SpecForgeEventTrainer._checkpoint_state(self, event)

    def train_event(self, event_id, feature_paths, passes=2, *, reference_check=None, presentation_schedule=None):
        if reference_check is not None or presentation_schedule is not None:
            raise ValueError("C uses natural sampling without additional reference probes")
        result = SpecForgeEventTrainer.train_event(self, event_id, feature_paths, passes)
        S.validate_event_receipt(result, arm=self.arm, binding=self.binding,
            event_number=len(self.events), trainer_status=self.status())
        self.completed_event = len(self.events)
        return result

    def verify_boundary(self, export_path=None, receipt_path=None):
        import torch
        began = time.perf_counter()
        with preserve_rng():
            saved = torch.load(self.latest, map_location="cpu", weights_only=False, mmap=True)
            result = S.validate_snapshot(saved, **self.validation_arguments(), completed_event=len(self.events))
            if export_path is not None:
                _export_equal(saved, export_path)
            result.update(live=native.verify_live(self, saved), export_exact=export_path is not None)
        result["wall_s"] = time.perf_counter()-began
        if receipt_path is not None:
            write_json(receipt_path, result)
        return result

    def reset_qualification(self, *, relative_cap=.25, runtime=None):
        if not self.qualification_only:
            raise ValueError("Disposable origin reset is forbidden in production")
        self.qualification_allocation(relative_cap)
        state, receipt = S.fork_state(self.origin, arm=self.arm, binding=self.binding,
            config_text=self.config_text, run_id=self.cfg.run_id,
            original_config_text=self.original_config_text,
            expected_cuda_rng_states=self.expected_cuda_rng_states)
        self._clear_qualification_accumulation()
        self._load_scientific_state(state)
        self.completed_event = 4
        self.response_records.clear()
        self.window_records.clear()
        self._anchor_observations.clear()
        self._presentation_schedule = None
        if runtime is not None:
            from experiments.functional_step_geometry.qualify import restore_runtime
            restore_runtime(self.bundle.model, runtime)
        return receipt
