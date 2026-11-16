"""Persistent two-event adapter around the unchanged SpecForge/K+ trainer.

Each response uses this arm's current weights and natural sampling. The E row
controller is reinstalled before every forward; it cannot fall back to N through
an inherited disposable-window default. No model or optimizer snapshot is kept
per window in production.
"""
from copy import deepcopy
from pathlib import Path
import math
import time

from experiments.full_k_gsm8k import training as native
from experiments.ok_coding.training import _export_equal
from experiments.ok_continuation.resume import differences
from experiments.recent_repair_retention.training import schedule_adapter
from online_feedback.kplus import install_kplus, kplus_contract
from online_feedback.native_prefix_roles import install_native_prefix_roles
from online_feedback.rng import preserve_rng, restore_rng
from online_feedback.storage import write_json
from online_feedback.training import SpecForgeEventTrainer
from . import state as contracts

ROLE_BOUNDARIES = frozenset((0, 1, 126, 127, 254, 255))


def boundary_gradient_check(receipt):
    """Check observed E rows without requiring an error in every response."""
    chunks = [chunk for response in receipt["responses"] for chunk in response["chunks"]]
    errors = []
    if receipt["role"] != "E" or not chunks:
        errors.append("missing_current_E_role_evidence")
    for chunk in chunks:
        gradient = chunk["gradient"]
        if gradient is None or gradient["finite"] is not True:
            errors.append("missing_or_nonfinite_row_gradient")
        elif gradient["excluded_nonzero"] or gradient["prefix_nonzero"]:
            errors.append("withheld_prefix_row_has_direct_gradient")
    return dict(status="failed" if errors else "passed", errors=sorted(set(errors)),
        observed_chunks=len(chunks), prefix_occurrences=sum(c["prefix_occurrences"] for c in chunks),
        error_occurrences=sum(c["error_occurrences"] for c in chunks),
        all_correct_is_valid=True)


class PersistentPrefixTrainer(native.FullKTrainer):
    """One immutable arm and one own-state lineage, ending after event eight."""

    def __init__(self, config_path, output_root, *, arm, binding, origin_state,
                 origin_export, original_config_text, completed_event,
                 schedule=None, expected_cuda_rng_states=1,
                 qualification_only=False, qualification_role=None):
        import torch
        contracts.experiment_contract(arm, binding)
        if qualification_role not in (None, "ALL") or (qualification_role == "ALL" and arm != "N"):
            raise ValueError("Only qualification N may install the all-enabled controller")
        if qualification_role is not None and not qualification_only:
            raise ValueError("Qualification role cannot alter a production arm")
        self._arm = arm
        self.binding = binding
        self.completed_event = completed_event
        self.original_config_text = original_config_text
        self.expected_cuda_rng_states = expected_cuda_rng_states
        self.qualification_only = bool(qualification_only)
        self.qualification_role = qualification_role
        self.schedule = schedule
        self.origin_path = Path(origin_state)
        self.origin_export = Path(origin_export)
        self._source = torch.load(origin_state, map_location="cpu", weights_only=False, mmap=True)
        self._is_fork = contracts.CONTRACT_KEY not in self._source["algorithm_contract"]
        if self._is_fork:
            contracts.validate_legacy_origin(self._source, original_config_text, completed_event,
                                             expected_cuda_rng_states)
        elif completed_event != 7:
            raise ValueError("Own-state continuation starts at the complete event-seven boundary")
        _export_equal(self._source, origin_export)
        self.response_records, self.window_records, self.work, self._work_handles = [], [], {}, []
        self.role_controller = None
        self.presentation_ordinal = None
        self.role_receipts = []
        self.step_observer = None  # Qualification-only preclip observation, never an update.
        SpecForgeEventTrainer.__init__(self, config_path, output_root, resume_state=origin_state,
            objective="osd_first_error", reduction="response_mean", ranking_lambda=0.)
        # The live GPU owns its tensors; no historical CPU model-sized archive
        # remains resident through the event. Compact histories are retained.
        self._source = None

    @property
    def arm(self):
        return self._arm

    def _validate_config(self):
        SpecForgeEventTrainer._validate_config(self)
        contracts.validate_config_text(self.config_text, self.arm, self.original_config_text)

    def _build_runtime(self):
        SpecForgeEventTrainer._build_runtime(self)
        install_kplus(self.bundle.model)
        if differences(contracts.native_base_contract(self._source["algorithm_contract"]), self.contract):
            raise ValueError("Constructed native runtime differs from the authenticated origin")
        if self._is_fork:
            self._source, self.migration_receipt = contracts.fork_state(self._source,
                arm=self.arm, binding=self.binding, config_text=self.config_text,
                run_id=self.cfg.run_id, qualification_only=self.qualification_only)
        else:
            self.migration_receipt = None
        self.contract = deepcopy(self._source["algorithm_contract"])
        role = self.qualification_role or ("E" if self.arm == "E" else None)
        if role is not None:
            self.role_controller = install_native_prefix_roles(self.bundle.model, role)
        self._install_observations()
        ordinary_forward, ordinary_step = self._native_forward, self._native_step
        self._uncontrolled_forward = ordinary_forward

        def current_arm_forward(*args, **kwargs):
            import torch
            self._install_current_role()
            result = ordinary_forward(*args, **kwargs)
            if not bool(torch.isfinite(result.loss).all()):
                raise ValueError("Nonfinite persistent response loss")
            return result

        def native_step():
            # DirectBF16AdamW performs the qualified single finite clip, AdamW,
            # and zero_grad. Observer may inspect preclip gradients, not replace
            # them or perform an additional optimizer application.
            if self.step_observer is not None:
                if not self.qualification_only:
                    raise ValueError("Model-sized gradient observation is qualification-only")
                self.step_observer(self)
            return ordinary_step()

        self._native_forward, self._native_step = current_arm_forward, native_step

    def _install_current_role(self):
        model = self.bundle.model
        if self.role_controller is None:
            if getattr(model, "native_prefix_roles", None) is not None:
                raise ValueError("Native N/S unexpectedly acquired a row controller")
            return
        expected = self.qualification_role or "E"
        # Reinstall the same declared controller, then start fresh observations.
        model.native_prefix_roles = self.role_controller
        self.role_controller.reset(role=expected,
            capture_gradients=self.qualification_only or self.presentation_ordinal in ROLE_BOUNDARIES)
        if model.native_prefix_roles.role != expected:
            raise ValueError("Persistent arm role was not reinstated")

    def validation_arguments(self):
        return dict(arm=self.arm, binding=self.binding, config_text=self.config_text,
            original_config_text=self.original_config_text, expected_contract=self.contract,
            expected_recipe=self.backend.optimizer.recipe,
            expected_cuda_rng_states=self.expected_cuda_rng_states,
            qualification_only=self.qualification_only)

    def _restore(self, path):
        state = self._source
        contracts.validate_snapshot(state, **self.validation_arguments(), completed_event=self.completed_event)
        self.bundle.draft_model.load_state_dict(state["draft_state_dict"], strict=True)
        self.backend.load_state_dict(state["backend"])
        self.global_step, self.presentations = state["global_step"], state["presentations"]
        self.events = deepcopy(state["events"])
        self.core._micro = self.presentations
        self._install_current_role()
        restore_rng(state["rng"])
        receipt = native.verify_live(self, state)
        receipt.update(origin_state=str(self.origin_path), origin_export=str(self.origin_export),
                       immutable_arm=self.arm, migration=self.migration_receipt)
        write_json(self.output_root / "RESUME_CHECK.json", receipt)

    def _prepare_event(self, event_id, feature_paths, passes):
        if self.qualification_only:
            raise ValueError("Disposable qualification cannot publish a production event")
        number = len(self.events) + 1
        if number not in (7, 8) or (self.arm == "N" and number != 8):
            raise ValueError("Persistent campaign only authorizes E/S7 and N/E/S8")
        if (number == 7) != (self.schedule is not None):
            raise ValueError("Event seven requires its retained schedule; event eight uses own natural sampling")
        self.role_receipts.clear()
        features = native.FullKTrainer._prepare_event(self, event_id, feature_paths, passes)
        if self.schedule is not None:
            from online_feedback.presentation_schedule import PresentationSchedule
            self._presentation_schedule = PresentationSchedule(schedule_adapter(self.schedule, features),
                [feature["record_id"] for feature in features], passes, features=features)
        return features

    def run_pair(self, pair, pass_index, ordinal):
        """Two actual native presentations, one Adam step, continuing this arm."""
        import torch
        from specforge.training.strategies.base import StepContext
        if len(pair) != 2 or self.core.accumulation_remainder:
            raise ValueError("Require one complete native two-response window")
        start_step, start_presentations = self.global_step, self.presentations
        rows = []
        for offset, feature in enumerate(pair):
            self.presentation_ordinal = ordinal + offset
            schedule = self._presentation_schedule
            if schedule is not None:
                schedule.begin_presentation(pass_index, feature["record_id"])
            raw = torch.load(feature["path"], map_location="cpu", weights_only=True)
            self._anchor_observations.clear()
            began = time.perf_counter()
            result = self.core.train_step(self._batch(raw), StepContext(global_step=self.global_step,
                total_steps=self.horizon, collect_detailed_metrics=True))
            if schedule is not None:
                schedule.finish_presentation()
            metrics = result.materialize_metrics()
            elapsed = time.perf_counter() - began
            self.presentations += 1
            self.global_step += int(result.optimizer_stepped)
            if not all(math.isfinite(float(value)) for value in metrics.values()):
                raise ValueError("Nonfinite persistent training metrics")
            roles = self.role_controller.receipt() if self.role_controller is not None else dict(role="N")
            self.role_receipts.append(roles)
            row = dict(record_id=feature["record_id"], **{"pass": pass_index},
                optimizer_step_after=self.global_step, optimizer_stepped=result.optimizer_stepped,
                learning_rate=self.backend.optimizer.get_learning_rate() if result.optimizer_stepped else None,
                metrics=metrics, training_s=elapsed, sampled_anchors=deepcopy(self._anchor_observations),
                metrics_scope="accumulation_window" if result.optimizer_stepped else "presentation",
                response_observation=self.response_records[-1], prefix_roles=roles,
                prompt_tokens=feature["prompt_length"], response_tokens=feature["response_tokens"])
            if self.arm == "E" and (self.qualification_only or self.presentation_ordinal in ROLE_BOUNDARIES):
                row["boundary_gradient_check"] = boundary_gradient_check(roles)
            rows.append(row)
            self._append_observation("PRESENTATIONS.jsonl", row)
            if row.get("boundary_gradient_check", {}).get("status") == "failed":
                raise ValueError("Current E boundary row-gradient check failed; evidence retained")
        if (self.global_step, self.presentations) != (start_step + 1, start_presentations + 2):
            raise ValueError("Persistent pair changed the native optimizer/accumulation accounting")
        return rows

    def _train_passes(self, features, passes):
        import torch
        rows = []
        for pass_index in range(passes):
            order = torch.randperm(len(features)).tolist()
            for offset in range(0, len(order), 2):
                pair = [features[index] for index in order[offset:offset + 2]]
                rows.extend(self.run_pair(pair, pass_index, len(rows)))
        self.presentation_ordinal = None
        if len(rows) != 256 or len(self.response_records) != 256 or len(self.window_records) != 128:
            raise ValueError("Persistent event must have 256 original forwards and 128 native updates")
        return rows

    def _checkpoint_state(self, event):
        event.update(method=self.arm, dataset="gsm8k", kplus_loss=kplus_contract(),
                     **{contracts.CONTRACT_KEY: self.contract[contracts.CONTRACT_KEY]})
        contracts.validate_event_receipt(event, arm=self.arm, binding=self.binding,
            event_number=len(self.events) + 1, require_completed=False)
        return SpecForgeEventTrainer._checkpoint_state(self, event)

    def train_event(self, event_id, feature_paths, passes=2, *, reference_check=None, presentation_schedule=None):
        if reference_check is not None or presentation_schedule is not None:
            raise ValueError("No external probe or schedule substitution is authorized")
        result = SpecForgeEventTrainer.train_event(self, event_id, feature_paths, passes)
        contracts.validate_event_receipt(result, arm=self.arm, binding=self.binding,
            event_number=len(self.events), trainer_status=self.status())
        self.completed_event = len(self.events)
        self.schedule = None
        return result

    def verify_boundary(self, export_path=None, receipt_path=None):
        import torch
        began = time.perf_counter()
        with preserve_rng():
            saved = torch.load(self.latest, map_location="cpu", weights_only=False, mmap=True)
            result = contracts.validate_snapshot(saved, **self.validation_arguments(),
                                                 completed_event=len(self.events))
            if export_path is not None:
                _export_equal(saved, export_path)
            result.update(live=native.verify_live(self, saved), export_exact=export_path is not None)
        result["wall_s"] = time.perf_counter() - began
        if receipt_path is not None:
            write_json(receipt_path, result)
        return result
