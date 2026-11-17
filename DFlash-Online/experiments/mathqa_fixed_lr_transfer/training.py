"""Two native K+ continuations differing only in their fixed AdamW group LR.

The sampler, response loss, accumulation, clipping, AdamW and exporter are the
existing qualified implementations. Qualification can explicitly restore a
private fork; production restores only on construction and never per step.
"""
from copy import deepcopy
from pathlib import Path
import math
import time

from experiments.full_k_gsm8k import training as native
from experiments.ok_coding.training import _export_equal
from experiments.ok_continuation.resume import differences
from experiments.role_loss_pilot import training as historical
from online_feedback.kplus import install_kplus, kplus_contract, KPlusRedistribution
from online_feedback.presentation_schedule import PresentationSchedule
from online_feedback.rng import preserve_rng, restore_rng
from online_feedback.storage import save_torch, write_json
from online_feedback.training import SpecForgeEventTrainer
from . import state as S

QUALIFICATION_SCHEMA = "mathqa_fixed_lr_disposable_state_v1"
FORBIDDEN_CONTROLLERS = ("prefix_guard", "adversarial_block", "native_prefix_roles",
    "point_loss_extension", "prefix_allocation", "recent_repair_retention",
    "role_loss_extension", "successor_shuffle", "survival_head_extension")


class MathQAFixedLRTrainer(native.FullKTrainer):
    """Complete KPLUS4 -> own event5 -> own event6, with no objective hooks."""

    @property
    def arm(self):
        return self._arm

    def __init__(self, config_path, output_root, *, arm, binding, origin_state,
                 origin_export, original_config_text, completed_event=4,
                 resume_state=None, resume_export=None, expected_cuda_rng_states=1,
                 qualification_only=False):
        import torch
        if completed_event not in (4, 5) or (completed_event == 5) != (resume_state is not None):
            raise ValueError("Continue original KPLUS4 or a complete own event5 state")
        if (resume_state is None) != (resume_export is None):
            raise ValueError("An own recovery requires its matching published export")
        S.experiment_contract(arm, binding)
        self._arm, self.dataset, self.binding = arm, "mathqa", binding
        self.original_config_text = original_config_text
        self.completed_event = completed_event
        self.expected_cuda_rng_states = expected_cuda_rng_states
        self.qualification_only = qualification_only
        self.origin_path, self.origin_export = Path(origin_state), Path(origin_export)
        self.origin = torch.load(origin_state, map_location="cpu", weights_only=False, mmap=True)
        S.validate_origin(self.origin, original_config_text,
                          expected_cuda_rng_states=expected_cuda_rng_states)
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

    def _validate_config(self):
        SpecForgeEventTrainer._validate_config(self)
        S.validate_config_text(self.config_text, self.arm, self.original_config_text)

    def _build_runtime(self):
        SpecForgeEventTrainer._build_runtime(self)
        install_kplus(self.bundle.model)
        if differences(self.contract, S.native_base_contract(self.origin["algorithm_contract"])):
            raise ValueError("Constructed native runtime differs from the original MathQA state")
        self.contract = S.continuation_contract(self.origin, arm=self.arm, binding=self.binding)
        self._check_native_path()
        self._install_observations()
        self._ordinary_forward, self._ordinary_step = self._native_forward, self._native_step

        def checked_forward(*args, **kwargs):
            import torch
            self._check_native_path()
            result = self._ordinary_forward(*args, **kwargs)
            if not bool(torch.isfinite(result.loss).all()):
                raise ValueError("Nonfinite native MathQA K+ loss")
            return result

        def checked_step():
            self._check_actual_lr()
            if self.step_observer is not None:
                if not self.qualification_only:
                    raise ValueError("Gradient snapshots are disposable qualification only")
                self.step_observer(self)
            result = self._ordinary_step()
            self._check_actual_lr()
            return result

        self._native_forward, self._native_step = checked_forward, checked_step

    def _check_native_path(self):
        model = self.bundle.model
        if (not isinstance(getattr(model, "first_error_redistribution", None), KPlusRedistribution)
                or any(getattr(model, name, None) is not None for name in FORBIDDEN_CONTROLLERS)):
            raise ValueError("Fixed-LR continuation requires native K+ without another controller")

    def _check_actual_lr(self):
        optimizer = self.backend.optimizer
        expected = S.arm_lr(self.arm)
        if (self.cfg.training.learning_rate != expected
                or optimizer.recipe["optimizer_options"]["lr"] != expected
                or len(optimizer.optimizer.param_groups) != 1
                or any(group["lr"] != expected for group in optimizer.optimizer.param_groups)
                or optimizer.get_learning_rate() != expected):
            raise ValueError("Actual Adam group LR changed from the fixed arm rate")

    def validation_arguments(self):
        return dict(origin=self.origin, arm=self.arm, binding=self.binding,
            config_text=self.config_text, original_config_text=self.original_config_text,
            expected_recipe=self.backend.optimizer.recipe,
            expected_cuda_rng_states=self.expected_cuda_rng_states)

    def _load_scientific_state(self, state):
        self.bundle.draft_model.load_state_dict(state["draft_state_dict"], strict=True)
        # Adam can alias same-device state (especially CPU step counters). The
        # origin and saved qualification oracle must never be mutated by a fork.
        self.backend.load_state_dict(deepcopy(state["backend"]))
        self.global_step, self.presentations = state["global_step"], state["presentations"]
        self.events = deepcopy(state["events"])
        self.core._micro = self.presentations
        restore_rng(state["rng"])
        self._check_actual_lr()
        self._check_native_path()

    def _restore(self, path):
        state = self._source
        migration = None
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
            actual_learning_rate=S.arm_lr(self.arm), seed_reset=False)
        write_json(self.output_root / "RESUME_CHECK.json", receipt)

    def save_initial(self, *args, **kwargs):
        raise ValueError("Fixed-LR MathQA resumes complete KPLUS4; no released-state reset")

    def _prepare_event(self, event_id, feature_paths, passes):
        if self.qualification_only or len(self.events) + 1 not in (5, 6):
            raise ValueError("Production permits only the next complete MathQA event5/6")
        return native.FullKTrainer._prepare_event(self, event_id, feature_paths, passes)

    def _train_passes(self, features, passes):
        return historical.RoleLossTrainer._train_passes(self, features, passes)

    def _checkpoint_state(self, event):
        event.update(method=self.arm, dataset="mathqa", kplus_loss=kplus_contract(),
                     **{S.CONTRACT_KEY: self.contract[S.CONTRACT_KEY]})
        S.validate_event_receipt(event, arm=self.arm, binding=self.binding,
            event_number=len(self.events)+1, require_completed=False)
        return SpecForgeEventTrainer._checkpoint_state(self, event)

    def train_event(self, event_id, feature_paths, passes=2, *, reference_check=None, presentation_schedule=None):
        if reference_check is not None or presentation_schedule is not None:
            raise ValueError("Fresh native continuation has no probes or forced historical schedule")
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

    def reset_qualification(self, *, arm=None, runtime=None):
        """Restore a private origin fork in a disposable runtime; never production."""
        import yaml
        if not self.qualification_only:
            raise ValueError("Qualification reset is forbidden in production")
        self._arm = self.arm if arm is None else arm
        rate = S.arm_lr(self.arm)
        config = yaml.safe_load(self.config_text)
        config["training"]["learning_rate"] = rate
        self.config_text = yaml.safe_dump(config)
        self.cfg.training.learning_rate = rate
        self.backend.optimizer.recipe["optimizer_options"]["lr"] = rate
        for group in self.backend.optimizer.optimizer.param_groups:
            group["lr"] = rate
        self.contract = S.continuation_contract(self.origin, arm=self.arm, binding=self.binding)
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

    def _clear_qualification_accumulation(self):
        self.backend.optimizer.zero_grad()
        self.core._ratio_totals.clear()
        self.core._sum_totals.clear()

    def run_pair(self, pair, pass_index=0, ordinal=0):
        """Two natural presentations, one native Adam step, for bounded checks."""
        import torch
        from specforge.training.strategies.base import StepContext
        if not self.qualification_only or len(pair) != 2 or self.core.accumulation_remainder:
            raise ValueError("Disposable pair requires two responses and empty accumulation")
        before = (self.global_step, self.presentations)
        rows = []
        for feature in pair:
            if self._presentation_schedule is not None:
                self._presentation_schedule.begin_presentation(pass_index, feature["record_id"])
            raw = torch.load(feature["path"], map_location="cpu", weights_only=True)
            self._anchor_observations.clear()
            began = time.perf_counter()
            result = self.core.train_step(self._batch(raw), StepContext(global_step=self.global_step,
                total_steps=self.horizon, collect_detailed_metrics=True))
            if self._presentation_schedule is not None:
                self._presentation_schedule.finish_presentation()
            metrics = result.materialize_metrics()
            self.presentations += 1
            self.global_step += int(result.optimizer_stepped)
            if not all(math.isfinite(float(value)) for value in metrics.values()):
                raise ValueError("Nonfinite disposable native metrics")
            row = dict(record_id=feature["record_id"], **{"pass": pass_index},
                optimizer_step_after=self.global_step, optimizer_stepped=result.optimizer_stepped,
                learning_rate=self.backend.optimizer.get_learning_rate() if result.optimizer_stepped else None,
                metrics=metrics, training_s=time.perf_counter()-began,
                sampled_anchors=deepcopy(self._anchor_observations),
                metrics_scope="accumulation_window" if result.optimizer_stepped else "presentation",
                response_observation=self.response_records[-1])
            rows.append(row)
            self._append_observation("PRESENTATIONS.jsonl", row)
        if (self.global_step, self.presentations) != (before[0]+1, before[1]+2):
            raise ValueError("Disposable pair changed ordinary accumulation accounting")
        return rows

    def save_qualification(self, path):
        """Native export-compatible temporary state; no extra forward or Adam."""
        from experiments.prefix_ranking.qualify import cpu_copy
        if not self.qualification_only or self.core.accumulation_remainder:
            raise ValueError("Temporary snapshot requires disposable empty accumulation")
        self._check_actual_lr()
        self._check_native_path()
        state = cpu_copy(native.live_state(self))
        metadata = dict(schema=QUALIFICATION_SCHEMA, arm=self.arm,
            schedule=None if self._presentation_schedule is None else deepcopy(vars(self._presentation_schedule)),
            response_records=deepcopy(self.response_records), window_records=deepcopy(self.window_records),
            anchor_observations=deepcopy(self._anchor_observations), active_event=self._active_event,
            model_costs=deepcopy(self.bundle.model.costs))
        save_torch(path, dict(state, qualification_state=metadata))
        return dict(global_step=self.global_step, presentations=self.presentations, extra_optimizer_steps=0)

    def restore_qualification(self, path, *, runtime=None):
        import torch
        if not self.qualification_only:
            raise ValueError("Temporary restore is forbidden in production")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        metadata = payload.pop("qualification_state")
        if (metadata["schema"] != QUALIFICATION_SCHEMA or metadata["arm"] != self.arm
                or differences(payload["algorithm_contract"], self.contract)
                or payload["config_text"] != self.config_text or payload["run_id"] != self.cfg.run_id
                or differences(payload["events"], self.origin["events"])
                or payload["presentations"] != 2*payload["global_step"]
                or not 512 <= payload["global_step"] <= 640):
            raise ValueError("Foreign or incomplete disposable MathQA checkpoint")
        S.require_optimizer_lr(payload["backend"]["optimizer"], self.arm)
        for moments in payload["backend"]["optimizer"]["optimizer_state_dict"]["state"].values():
            if moments["step"].item() != payload["global_step"]:
                raise ValueError("Temporary Adam counters differ")
        self._clear_qualification_accumulation()
        self._load_scientific_state(payload)
        self._presentation_schedule = None
        if metadata["schedule"] is not None:
            self._presentation_schedule = object.__new__(PresentationSchedule)
            vars(self._presentation_schedule).update(deepcopy(metadata["schedule"]))
        self.response_records = deepcopy(metadata["response_records"])
        self.window_records = deepcopy(metadata["window_records"])
        self._anchor_observations[:] = deepcopy(metadata["anchor_observations"])
        self._active_event = metadata["active_event"]
        self.bundle.model.costs = deepcopy(metadata["model_costs"])
        if runtime is not None:
            from experiments.functional_step_geometry.qualify import restore_runtime
            restore_runtime(self.bundle.model, runtime)
        mismatch = differences(native.live_state(self), payload)
        if mismatch:
            raise ValueError("Disposable own state did not restore exactly: " + str(mismatch[:8]))
        return dict(status="passed", global_step=self.global_step, presentations=self.presentations,
                    exact_scientific_state=True, extra_optimizer_steps=0)
