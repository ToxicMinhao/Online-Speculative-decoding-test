"""Persistent C7/C8 training on the qualified native K+ execution path.

Only the detached prefix allocation is new. The native sampler, response mean,
accumulation, clipping, AdamW, feature reader and exporter remain shared. The
complete own C7 recovery, including RNG and moments, is the only C8 origin.
"""
from copy import deepcopy
from pathlib import Path
import math
import time

from experiments.full_k_gsm8k import training as native
from experiments.functional_step_geometry import training as geometry
from experiments.ok_coding.training import _export_equal
from experiments.ok_continuation.resume import differences
from experiments.persistent_prefix_withholding.qualification import isolated_backend_for_load
from experiments.recent_repair_retention import training as retention
from experiments.role_loss_pilot import training as historical
from online_feedback.kplus import install_kplus, kplus_contract
from online_feedback.prefix_allocation import install_prefix_allocation, prefix_allocation_contract
from online_feedback.presentation_schedule import PresentationSchedule
from online_feedback.rng import preserve_rng, restore_rng
from online_feedback.storage import save_torch, write_json
from online_feedback.training import SpecForgeEventTrainer

CONTRACT_KEY = "bounded_prefix_allocation_v1"
QUALIFICATION_SCHEMA = "bounded_prefix_allocation_disposable_state_v1"


def experiment_contract(binding, relative_cap=.25):
    native.experiment_contract("K", binding)  # Retain the shared binding check.
    return dict(schema=CONTRACT_KEY, arm="C", binding=binding,
        allocation=prefix_allocation_contract(relative_cap=relative_cap),
        native_loss=kplus_contract(), first_event=7, final_event=8,
        inherited_horizon=7424, learning_rate=3e-4, seed_reset=False,
        initialization="exact_complete_KPLUS6_then_own_complete_C7",
        reference=None, additional_model_forwards=0)


def native_base_contract(contract):
    result = dict(contract)
    for key in (historical.CONTRACT_KEY, geometry.CONTRACT_KEY, CONTRACT_KEY):
        result.pop(key, None)
    return result


def native_view(contract):
    origin_binding = contract[historical.CONTRACT_KEY]["binding"]
    return dict(native_base_contract(contract),
                **{native.CONTRACT_KEY: native.experiment_contract("K", origin_binding)})


def fork_state(state, *, binding, config_text, run_id, relative_cap=.25):
    """Add an experiment identity without changing any scientific payload.

    The caller authenticates K6 first. Live optimizer loading isolates its
    mutable counters; this metadata-only view never reconstructs the origin.
    """
    if len(state["events"]) != 6 or CONTRACT_KEY in state["algorithm_contract"]:
        raise ValueError("Candidate may fork only from complete native K6")
    return dict(state, config_text=config_text, run_id=run_id,
        algorithm_contract=dict(state["algorithm_contract"],
            **{CONTRACT_KEY: experiment_contract(binding, relative_cap)}))


def validate_event_receipt(event, *, binding, event_number, relative_cap=.25,
                           require_completed=True, trainer_status=None):
    if event_number not in (7, 8):
        raise ValueError("Candidate production ends after its own event eight")
    if (event.get("method") != "C" or event.get("kplus_loss") != kplus_contract()
            or event.get(CONTRACT_KEY) != experiment_contract(binding, relative_cap)):
        raise ValueError("Candidate event lost its fixed allocation contract")
    result = native.validate_event_receipt(event, arm="K", event_number=event_number,
        require_completed=require_completed, trainer_status=trainer_status)
    for row in event["records"]:
        observed = row["response_observation"]
        if (observed["record_id"] != row["record_id"]
                or observed["sampled_anchors"] != row["sampled_anchors"]
                or observed["sum_metrics"]["kplus_response_count"] != 1
                or "prefix_allocation" not in observed):
            raise ValueError("Candidate diagnostics do not bind the original response forward")
    return dict(result, arm="C")


def validate_snapshot(state, *, binding, config_text, original_config_text,
                      completed_event, expected_contract, expected_recipe,
                      relative_cap=.25, expected_cuda_rng_states=1):
    """Validate full event boundaries without weakening the native validator."""
    if completed_event not in (6, 7, 8):
        raise ValueError("Candidate recovery is outside its complete event boundaries")
    native.validate_config_text(config_text, "K", original_config_text)
    contract = state["algorithm_contract"]
    if (contract.get(CONTRACT_KEY) != experiment_contract(binding, relative_cap)
            or differences(contract, expected_contract)):
        raise ValueError("Changed candidate contract or foreign recovery")
    for number, event in enumerate(state["events"], 1):
        if number <= 4:
            historical.validate_event_receipt(event, arm="KPLUS", dataset="gsm8k",
                                              event_number=number, require_completed=False)
        elif number <= 6:
            geometry.validate_event_receipt(event, arm="KPLUS", dataset="gsm8k",
                                            event_number=number, require_completed=False)
        else:
            validate_event_receipt(event, binding=binding, event_number=number,
                relative_cap=relative_cap, require_completed=False)
    view = native_view(contract)
    result = native.validate_snapshot(dict(state, algorithm_contract=view), arm="K",
        binding=view[native.CONTRACT_KEY]["binding"], config_text=config_text,
        original_config_text=original_config_text, completed_event=completed_event,
        expected_contract=view, expected_recipe=expected_recipe,
        expected_cuda_rng_states=expected_cuda_rng_states)
    return dict(result, arm="C", candidate_contract_restored=True)


class CandidateTrainer(native.FullKTrainer):
    """One persistent candidate; the optional cap-zero path is disposable only."""

    def __init__(self, config_path, output_root, *, binding, origin_state,
                 origin_export, original_config_text, completed_event,
                 schedule=None, relative_cap=.25, expected_cuda_rng_states=1,
                 qualification_only=False):
        import torch
        if completed_event not in (6, 7) or (relative_cap != .25 and not qualification_only):
            raise ValueError("Production requires fixed C from K6 or complete own C7")
        experiment_contract(binding, relative_cap)
        self.arm, self.binding = "C", binding
        self.relative_cap = relative_cap
        self.completed_event = completed_event
        self.original_config_text = original_config_text
        self.expected_cuda_rng_states = expected_cuda_rng_states
        self.qualification_only = qualification_only
        self._qualification_cap = relative_cap
        self.schedule = schedule
        self.origin_path, self.origin_export = Path(origin_state), Path(origin_export)
        self._source = torch.load(origin_state, map_location="cpu", weights_only=False, mmap=True)
        self._is_fork = CONTRACT_KEY not in self._source["algorithm_contract"]
        if self._is_fork:
            if completed_event != 6:
                raise ValueError("Candidate initialization must use exact complete K6")
            retention.validate_origin(self._source, original_config_text,
                                      expected_cuda_rng_states=expected_cuda_rng_states)
        elif completed_event != 7:
            raise ValueError("Production continuation requires complete own C7")
        _export_equal(self._source, origin_export)
        self.response_records, self.window_records, self.work, self._work_handles = [], [], {}, []
        self.step_observer = None
        SpecForgeEventTrainer.__init__(self, config_path, output_root, resume_state=origin_state,
            objective="osd_first_error", reduction="response_mean", ranking_lambda=0.)
        self._source = None
        write_json(self.output_root / "CANDIDATE_CONFIG.json", dict(
            native_config_text=self.config_text, candidate_contract=self.contract[CONTRACT_KEY]))

    def _validate_config(self):
        SpecForgeEventTrainer._validate_config(self)
        native.validate_config_text(self.config_text, "K", self.original_config_text)

    def _build_runtime(self):
        SpecForgeEventTrainer._build_runtime(self)
        install_kplus(self.bundle.model)
        if differences(native_base_contract(self._source["algorithm_contract"]), self.contract):
            raise ValueError("Constructed native training settings differ from authenticated origin")
        if self._is_fork:
            self._source = fork_state(self._source, binding=self.binding,
                config_text=self.config_text, run_id=self.cfg.run_id, relative_cap=self.relative_cap)
        self.contract = deepcopy(self._source["algorithm_contract"])
        self._install_allocation()
        self._install_observations()
        ordinary_forward, ordinary_step = self._native_forward, self._native_step

        def candidate_forward(*args, **kwargs):
            import torch
            self._check_allocation()
            result = ordinary_forward(*args, **kwargs)
            if not bool(torch.isfinite(result.loss).all()):
                raise ValueError("Nonfinite candidate response loss")
            return result

        def observed_native_step():
            if self.step_observer is not None:
                if not self.qualification_only:
                    raise ValueError("Preclip gradient inspection is qualification-only")
                self.step_observer(self)
            return ordinary_step()

        self._native_forward, self._native_step = candidate_forward, observed_native_step

    def _install_allocation(self):
        model = self.bundle.model
        if getattr(model, "native_prefix_roles", None) is not None:
            raise ValueError("Candidate cannot activate the historical prefix-withholding role")
        from online_feedback.prefix_allocation import PrefixAllocation
        existing = getattr(model, "prefix_allocation", None)
        if existing is not None:
            allowed = (0., .25) if self.qualification_only else (self.relative_cap,)
            if (not isinstance(existing, PrefixAllocation) or existing.relative_cap not in allowed
                    or existing.contract != prefix_allocation_contract(existing.relative_cap)):
                raise ValueError("Refuse to replace a foreign allocation controller")
        cap = self._qualification_cap if self.qualification_only else self.relative_cap
        # Only the verified own rule is replaced on an actual restore. Ordinary
        # forwards check it in place and never reconstruct parameters or hooks.
        model.prefix_allocation = None
        if cap is not None:
            install_prefix_allocation(model, relative_cap=cap)

    def qualification_allocation(self, relative_cap):
        """Select absent/cap-zero/active code for disposable parity checks only."""
        if not self.qualification_only or relative_cap not in (None, 0., .25):
            raise ValueError("Only bounded qualification may switch allocation paths")
        self._qualification_cap = relative_cap
        self._install_allocation()

    def _check_allocation(self):
        model = self.bundle.model
        cap = self._qualification_cap if self.qualification_only else self.relative_cap
        installed = getattr(model, "prefix_allocation", None)
        wrong = installed is not None if cap is None else (installed is None or installed.relative_cap != cap)
        if getattr(model, "native_prefix_roles", None) is not None or wrong:
            raise ValueError("Candidate allocation changed during a persistent trajectory")

    def _append_observation(self, name, value):
        if name == "RESPONSE_OBSERVATIONS.jsonl":
            # This runs once after the original response reduction, never during
            # activation-checkpoint recomputation or optimizer accumulation.
            controller = getattr(self.bundle.model, "prefix_allocation", None)
            value["prefix_allocation"] = deepcopy(controller.last_summary) if controller is not None else {"enabled": False}
        return native.FullKTrainer._append_observation(self, name, value)

    def validation_arguments(self):
        return dict(binding=self.binding, config_text=self.config_text,
            original_config_text=self.original_config_text, expected_contract=self.contract,
            expected_recipe=self.backend.optimizer.recipe, relative_cap=self.relative_cap,
            expected_cuda_rng_states=self.expected_cuda_rng_states)

    def _load_scientific_state(self, state):
        """The one shared restore operation; production calls it only on startup."""
        self.bundle.draft_model.load_state_dict(state["draft_state_dict"], strict=True)
        device = next(self.bundle.draft_model.parameters()).device
        self.backend.load_state_dict(isolated_backend_for_load(state["backend"], device))
        self.global_step, self.presentations = state["global_step"], state["presentations"]
        self.events = deepcopy(state["events"])
        self.core._micro = self.presentations
        self._install_allocation()
        restore_rng(state["rng"])

    def _restore(self, path):
        state = self._source
        validate_snapshot(state, **self.validation_arguments(), completed_event=self.completed_event)
        self._load_scientific_state(state)
        receipt = native.verify_live(self, state)
        receipt.update(origin_state=str(self.origin_path), origin_export=str(self.origin_export),
                       candidate_contract=self.contract[CONTRACT_KEY])
        write_json(self.output_root / "RESUME_CHECK.json", receipt)

    def _prepare_event(self, event_id, feature_paths, passes):
        if self.qualification_only:
            raise ValueError("Disposable qualification cannot publish a production event")
        number = len(self.events) + 1
        if number not in (7, 8):
            raise ValueError("Candidate campaign ends after event eight")
        if (number == 7) != (self.schedule is not None):
            raise ValueError("C7 checks retained anchors; C8 samples its own feedback naturally")
        features = native.FullKTrainer._prepare_event(self, event_id, feature_paths, passes)
        if self.schedule is not None:
            self._presentation_schedule = PresentationSchedule(retention.schedule_adapter(self.schedule, features),
                [feature["record_id"] for feature in features], passes, features=features)
        return features

    def run_pair(self, pair, pass_index, ordinal):
        """Expose the native two-presentation window to bounded qualification."""
        import torch
        from specforge.training.strategies.base import StepContext
        if len(pair) != 2 or self.core.accumulation_remainder:
            raise ValueError("Require a native two-response window at empty accumulation")
        start_step, start_presentations = self.global_step, self.presentations
        rows = []
        for feature in pair:
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
            self.presentations += 1
            self.global_step += int(result.optimizer_stepped)
            if not all(math.isfinite(float(value)) for value in metrics.values()):
                raise ValueError("Nonfinite candidate training metrics")
            row = dict(record_id=feature["record_id"], **{"pass": pass_index},
                optimizer_step_after=self.global_step, optimizer_stepped=result.optimizer_stepped,
                learning_rate=self.backend.optimizer.get_learning_rate() if result.optimizer_stepped else None,
                metrics=metrics, training_s=time.perf_counter()-began,
                sampled_anchors=deepcopy(self._anchor_observations),
                metrics_scope="accumulation_window" if result.optimizer_stepped else "presentation",
                response_observation=self.response_records[-1])
            rows.append(row)
            self._append_observation("PRESENTATIONS.jsonl", row)
        if (self.global_step, self.presentations) != (start_step+1, start_presentations+2):
            raise ValueError("Candidate window changed native accumulation accounting")
        return rows

    def _checkpoint_state(self, event):
        event.update(method="C", dataset="gsm8k", kplus_loss=kplus_contract(),
                     **{CONTRACT_KEY: self.contract[CONTRACT_KEY]})
        validate_event_receipt(event, binding=self.binding, event_number=len(self.events)+1,
                              relative_cap=self.relative_cap, require_completed=False)
        return SpecForgeEventTrainer._checkpoint_state(self, event)

    def train_event(self, event_id, feature_paths, passes=2, *, reference_check=None, presentation_schedule=None):
        if reference_check is not None or presentation_schedule is not None:
            raise ValueError("No additional reference or substituted sampler is authorized")
        result = SpecForgeEventTrainer.train_event(self, event_id, feature_paths, passes)
        validate_event_receipt(result, binding=self.binding, event_number=len(self.events),
            relative_cap=self.relative_cap, trainer_status=self.status())
        self.completed_event = len(self.events)
        self.schedule = None
        return result

    def verify_boundary(self, export_path=None, receipt_path=None):
        import torch
        began = time.perf_counter()
        with preserve_rng():
            saved = torch.load(self.latest, map_location="cpu", weights_only=False, mmap=True)
            result = validate_snapshot(saved, **self.validation_arguments(), completed_event=len(self.events))
            if export_path is not None:
                _export_equal(saved, export_path)
            result.update(live=native.verify_live(self, saved), export_exact=export_path is not None)
        result["wall_s"] = time.perf_counter()-began
        if receipt_path is not None:
            write_json(receipt_path, result)
        return result

    def save_qualification(self, path):
        """Serialize an actual temporary endpoint, without calling it an event."""
        from experiments.prefix_ranking.qualify import cpu_copy
        if not self.qualification_only or self.core.accumulation_remainder:
            raise ValueError("Temporary checkpoint requires disposable empty accumulation")
        self._check_allocation()
        state = cpu_copy(native.live_state(self))
        controller = getattr(self.bundle.model, "prefix_allocation", None)
        metadata = dict(schema=QUALIFICATION_SCHEMA, allocation_cap=self._qualification_cap,
            schedule=None if self._presentation_schedule is None else deepcopy(vars(self._presentation_schedule)),
            response_records=deepcopy(self.response_records), window_records=deepcopy(self.window_records),
            anchor_observations=deepcopy(self._anchor_observations), active_event=self._active_event,
            model_costs=deepcopy(self.bundle.model.costs),
            allocation_summary=deepcopy(controller.last_summary) if controller is not None else None)
        save_torch(path, dict(state, qualification_state=metadata))
        return dict(global_step=self.global_step, presentations=self.presentations,
                    extra_forward=0, extra_backward=0, extra_optimizer_steps=0)

    def restore_qualification(self, path, *, runtime=None):
        """Restore own temporary weights, Adam, RNG and schedule, not native K6."""
        import torch
        if not self.qualification_only:
            raise ValueError("Temporary restoration is forbidden in production")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        metadata = payload.pop("qualification_state")
        state = payload
        if (metadata["schema"] != QUALIFICATION_SCHEMA
                or differences(state["algorithm_contract"], self.contract)
                or state["config_text"] != self.config_text or state["run_id"] != self.cfg.run_id
                or state["presentations"] != 2*state["global_step"]
                or len(state["events"]) != self.completed_event
                or not 128*self.completed_event <= state["global_step"] <= 128*(self.completed_event+1)):
            raise ValueError("Foreign or incomplete disposable candidate checkpoint")
        for moments in state["backend"]["optimizer"]["optimizer_state_dict"]["state"].values():
            if moments["step"].item() != state["global_step"]:
                raise ValueError("Disposable Adam counter differs")
        self.backend.optimizer.zero_grad()
        self.core._ratio_totals.clear()
        self.core._sum_totals.clear()
        self._qualification_cap = metadata["allocation_cap"]
        self._load_scientific_state(state)
        schedule = metadata["schedule"]
        self._presentation_schedule = None
        if schedule is not None:
            self._presentation_schedule = object.__new__(PresentationSchedule)
            vars(self._presentation_schedule).update(deepcopy(schedule))
        self.response_records = deepcopy(metadata["response_records"])
        self.window_records = deepcopy(metadata["window_records"])
        self._anchor_observations[:] = deepcopy(metadata["anchor_observations"])
        self._active_event = metadata["active_event"]
        self.bundle.model.costs = deepcopy(metadata["model_costs"])
        if self.bundle.model.prefix_allocation is not None:
            self.bundle.model.prefix_allocation.last_summary = deepcopy(metadata["allocation_summary"])
        if runtime is not None:
            from experiments.functional_step_geometry.qualify import restore_runtime
            restore_runtime(self.bundle.model, runtime)
        self._check_allocation()
        mismatch = differences(native.live_state(self), state)
        if mismatch:
            raise ValueError("Temporary candidate state did not restore exactly: " + str(mismatch[:8]))
        return dict(status="passed", global_step=self.global_step, presentations=self.presentations,
                    exact_scientific_state=True, extra_optimizer_steps=0)
