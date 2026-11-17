"""One event-five continuation from the complete original PR32 KPLUS4 state."""

from copy import deepcopy
import math
from pathlib import Path
import time

from experiments.full_k_gsm8k import training as native
from experiments.ok_coding.storage import identity
from experiments.ok_coding.training import _export_equal
from experiments.ok_continuation.resume import differences
from experiments.role_loss_pilot import training as historical
from online_feedback.kplus import install_kplus, kplus_contract
from online_feedback.rng import preserve_rng, restore_rng
from online_feedback.storage import save_torch, write_json
from online_feedback.survival_head import survival_head_contract
from online_feedback.training import SpecForgeEventTrainer
from .contract import ARMS, DATASETS, EVENTS, SCHEMA, experiment_contract, expected_work


CONTRACT_KEY = SCHEMA
OBJECTIVE = "osd_first_error"


def validate_origin(state, *, dataset, original_config_text, expected_cuda_rng_states=1):
    if dataset not in DATASETS:
        raise ValueError("Require the GSM8K continuation dataset")
    old = state["algorithm_contract"].get(historical.CONTRACT_KEY, {})
    if (old.get("arm") != "KPLUS" or old.get("dataset") != dataset
            or CONTRACT_KEY in state["algorithm_contract"]):
        raise ValueError("Origin must be the original GSM8K PR32 KPLUS4 state")
    return historical.validate_snapshot(
        state, arm="KPLUS", dataset=dataset, binding=old["binding"],
        config_text=state["config_text"], original_config_text=original_config_text,
        completed_event=4, expected_contract=state["algorithm_contract"],
        expected_recipe=state["backend"]["optimizer"]["recipe"],
        expected_cuda_rng_states=expected_cuda_rng_states,
    )


def continuation_contract(origin, *, binding, arm, dataset):
    old = origin["algorithm_contract"][historical.CONTRACT_KEY]
    result = deepcopy(origin["algorithm_contract"])
    result[CONTRACT_KEY] = experiment_contract(binding, arm, dataset, origin_binding=old["binding"])
    return result


def validate_head_state(state, *, arm, expected_steps, native_recipe):
    """Validate the separate BF16 head and both Adam moments at a full boundary."""
    import torch

    if arm == "KPLUS" or state["contract"] != survival_head_contract(arm):
        raise ValueError("Separate head scientific contract differs")
    if type(state["steps"]) is not int or state["steps"] != expected_steps:
        raise ValueError("Separate head optimizer step count differs")
    head = state["head"]
    if set(head) != {"weight", "bias"}:
        raise ValueError("Require only scalar linear head weight and bias")
    weight, bias = head["weight"], head["bias"]
    if weight.ndim != 2 or weight.shape[0] != 1 or weight.shape[1] <= 0 or bias.shape != (1,):
        raise ValueError("Malformed scalar head shapes")
    if any(tensor.dtype != torch.bfloat16 or not torch.isfinite(tensor).all() for tensor in head.values()):
        raise ValueError("Head tensors must be finite BF16")
    optimizer = state["optimizer"]
    recipe, saved = optimizer["recipe"], optimizer["optimizer_state_dict"]
    expected_parameters = tuple((name, tuple(tensor.shape), str(tensor.dtype))
                                for name, tensor in head.items())
    if (recipe["type"] != "direct_bf16_adamw" or recipe["version"] != 1
            or recipe["parameters"] != expected_parameters
            or recipe["max_grad_norm"] != native_recipe["max_grad_norm"]
            or differences(recipe["optimizer_options"], native_recipe["optimizer_options"])):
        raise ValueError("Head optimizer recipe differs from native numerical settings")
    if len(saved["param_groups"]) != 1:
        raise ValueError("Head requires one independent Adam parameter group")
    group = saved["param_groups"][0]
    options = {name: value for name, value in group.items() if name != "params"}
    ids = group["params"]
    if (len(ids) != 2 or len(set(ids)) != 2 or differences(options, recipe["optimizer_options"])
            or set(saved["state"]) != (set(ids) if expected_steps else set())):
        raise ValueError("Head Adam parameter identities or moments are incomplete")
    for parameter_id, tensor in zip(ids, head.values()):
        if not expected_steps:
            continue
        moments = saved["state"][parameter_id]
        if set(moments) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("Head Adam moment fields differ")
        step = moments["step"]
        if step.numel() != 1 or not torch.isfinite(step).all() or step.item() != expected_steps:
            raise ValueError("Head Adam moment step differs")
        for name in ("exp_avg", "exp_avg_sq"):
            value = moments[name]
            if value.shape != tensor.shape or value.dtype != torch.bfloat16 or not torch.isfinite(value).all():
                raise ValueError("Head Adam moment dtype, shape or finiteness differs")
        if (moments["exp_avg_sq"] < 0).any():
            raise ValueError("Head Adam squared moment is negative")
    return dict(status="passed", steps=expected_steps, head_width=weight.shape[1],
                separate_optimizer=True, head_exported=False)


def validate_event_receipt(receipt, *, arm, dataset, event_number,
                           require_completed=True, trainer_status=None):
    if (arm not in ARMS or dataset not in DATASETS or event_number not in EVENTS
            or receipt.get("method") != arm or receipt.get("dataset") != dataset
            or receipt.get("kplus_loss") != kplus_contract()
            or receipt.get("survival_work") != expected_work(arm, event_number)):
        raise ValueError("Require the declared event-five survival continuation receipt")
    checked = native.validate_event_receipt(
        receipt, arm="K", event_number=event_number,
        require_completed=require_completed, trainer_status=trainer_status,
    )
    head = receipt.get("survival_head")
    if arm == "KPLUS":
        if head is not None:
            raise ValueError("KPLUS must not create a head")
    elif (not isinstance(head, dict) or head.get("contract") != survival_head_contract(arm)
          or head.get("steps") != 128 or not head.get("state_path")
          or not head.get("state_identity")):
        raise ValueError("Active arm requires a separately saved complete head boundary")
    for row in receipt["records"]:
        observation = row["response_observation"]
        sums = observation["sum_metrics"]
        if (observation["record_id"] != row["record_id"] or observation["pass_index"] != row["pass"]
                or sums["kplus_response_count"] != 1
                or not all(math.isfinite(float(value)) for value in sums.values())):
            raise ValueError("Native K+ observation identity or counting differs")
        mass = sums["kplus_W1"] + sums["kplus_W2"]
        tolerance = kplus_contract()["mass_conservation_atol"] + kplus_contract()["mass_conservation_rtol"] * mass
        if abs(sums["kplus_first_error_weight_after"] - mass) > tolerance:
            raise ValueError("Native K+ first-error mass conservation failed")
        if arm != "KPLUS":
            count = sums.get("head_valid_proposals", -1)
            if sums.get("head_response_count") != 1 or count < 0 or count != int(count):
                raise ValueError("Head response observations are absent or counted twice")
            if (sums["head_bce_numerator"] < 0 or sums["head_response_bce_sum"] < 0
                    or not math.isclose(sums["head_response_bce_sum"],
                                        sums["head_bce_numerator"] / max(count, 1),
                                        rel_tol=2e-7, abs_tol=1e-7)):
                raise ValueError("Head BCE must divide once by all valid proposals")
            expected_loss = sums["head_native_kplus_loss_sum"] + sums["head_response_bce_sum"]
            if not math.isclose(observation["loss"], expected_loss, rel_tol=2e-7, abs_tol=1e-7):
                raise ValueError("Response is not native K+ plus one mean BCE")
            if row["optimizer_stepped"] and row["metrics"].get("head_optimizer_steps") != 1:
                raise ValueError("Each backbone boundary must have one separate head step")
        elif any(name.startswith("head_") for name in sums):
            raise ValueError("Disabled KPLUS unexpectedly has auxiliary observations")
    return dict(checked, arm=arm, dataset=dataset)


def validate_snapshot(state, *, origin, binding, arm, dataset, config_text,
                      original_config_text, completed_event, expected_recipe,
                      expected_cuda_rng_states=1):
    import torch

    if type(completed_event) is not int or completed_event not in (4, 5):
        raise ValueError("Survival continuation has only event-four and event-five boundaries")
    expected = continuation_contract(origin, binding=binding, arm=arm, dataset=dataset)
    if differences(state["algorithm_contract"], expected):
        raise ValueError("Continuation scientific contract differs")
    if differences(state["events"][:4], origin["events"]):
        raise ValueError("Retained PR32 event receipts changed")
    old_binding = origin["algorithm_contract"][historical.CONTRACT_KEY]["binding"]
    old_native = historical.native_contract_view(origin["algorithm_contract"], old_binding, "KPLUS", dataset)
    view = dict(state, algorithm_contract=old_native)
    checked = native.validate_snapshot(
        view, arm="K", binding=old_binding, config_text=config_text,
        original_config_text=original_config_text, completed_event=completed_event,
        expected_contract=old_native, expected_recipe=expected_recipe,
        expected_cuda_rng_states=expected_cuda_rng_states,
    )
    if completed_event == 5:
        event = state["events"][4]
        validate_event_receipt(event, arm=arm, dataset=dataset, event_number=5, require_completed=False)
        if arm != "KPLUS":
            head = event["survival_head"]
            if identity(head["state_path"]) != head["state_identity"]:
                raise ValueError("Separate head checkpoint bytes changed")
            head_state = torch.load(head["state_path"], map_location="cpu", weights_only=False)
            checked["head"] = validate_head_state(head_state, arm=arm, expected_steps=128,
                                                   native_recipe=expected_recipe)
    return dict(checked, arm=arm, dataset=dataset, historical_receipts_unchanged=True,
                start_adam_step=512, maximum_adam_step=640)


def migrate_origin(origin, *, binding, arm, dataset, config_text, run_id,
                   original_config_text, expected_recipe, expected_cuda_rng_states=1):
    validate_origin(origin, dataset=dataset, original_config_text=original_config_text,
                    expected_cuda_rng_states=expected_cuda_rng_states)
    state = dict(origin, config_text=config_text, run_id=run_id,
                 algorithm_contract=continuation_contract(origin, binding=binding, arm=arm, dataset=dataset))
    validate_snapshot(state, origin=origin, binding=binding, arm=arm, dataset=dataset,
        config_text=config_text, original_config_text=original_config_text, completed_event=4,
        expected_recipe=expected_recipe, expected_cuda_rng_states=expected_cuda_rng_states)
    return state


class SurvivalHeadTrainer(native.FullKTrainer):
    def __init__(self, config_path, output_root, *, arm, binding,
                 original_config_text, origin_state, origin_export, dataset="gsm8k",
                 resume_state=None, completed_event=4, expected_cuda_rng_states=1):
        import torch

        if (arm not in ARMS or dataset not in DATASETS or completed_event not in (4, 5)
                or (completed_event == 5) != (resume_state is not None)):
            raise ValueError("Require the KPLUS4 origin or own completed event-five recovery")
        self.arm, self.dataset, self.binding = arm, dataset, binding
        self.original_config_text = original_config_text
        self.expected_cuda_rng_states = expected_cuda_rng_states
        self.completed_event = completed_event
        self.origin_path, self.origin_export = Path(origin_state), Path(origin_export)
        self.origin = torch.load(self.origin_path, map_location="cpu", weights_only=False, mmap=True)
        validate_origin(self.origin, dataset=dataset, original_config_text=original_config_text,
                        expected_cuda_rng_states=expected_cuda_rng_states)
        _export_equal(self.origin, self.origin_export)
        self.response_records, self.window_records, self.work, self._work_handles = [], [], {}, []
        self.head_installed = False
        self._saved_head_receipt = None
        SpecForgeEventTrainer.__init__(self, config_path, output_root,
            resume_state=resume_state or origin_state, objective=OBJECTIVE,
            reduction="response_mean", ranking_lambda=0.0)

    def _validate_config(self):
        SpecForgeEventTrainer._validate_config(self)
        native.validate_config_text(self.config_text, "K", self.original_config_text)

    def _build_runtime(self):
        SpecForgeEventTrainer._build_runtime(self)
        install_kplus(self.bundle.model)
        source_base = dict(self.origin["algorithm_contract"])
        del source_base[historical.CONTRACT_KEY]
        if differences(self.contract, source_base):
            raise ValueError("Constructed native runtime differs from original KPLUS4")
        self.contract = continuation_contract(self.origin, binding=self.binding,
                                               arm=self.arm, dataset=self.dataset)
        self._install_observations()
        ordinary_forward, ordinary_step = self._native_forward, self._native_step

        def finite_forward(*args, **kwargs):
            import torch
            result = ordinary_forward(*args, **kwargs)
            if not bool(torch.isfinite(result.loss).all()):
                raise ValueError("Nonfinite K+ plus head loss before optimizer application")
            return result

        def finite_step():
            from experiments.adaflash_arepo.qualify import finite_gradient_check
            parameters = {name: p for name, p in self.bundle.draft_model.named_parameters() if p.requires_grad}
            finite_gradient_check(parameters, self.output_root, self.presentations)
            extension = getattr(self.bundle.model, "survival_head_extension", None)
            if extension is not None:
                finite_gradient_check(dict(extension.head.named_parameters()), self.output_root / "head", self.presentations)
            return ordinary_step()

        self._native_forward, self._native_step = finite_forward, finite_step

    def install_head(self, *, enabled=True):
        """Call after native restoration and discarded startup have finished."""
        import torch
        from online_feedback.survival_head_runtime import install_survival_head_core

        if self.head_installed:
            raise ValueError("Survival head has already been installed")
        original = self.core
        self.core = install_survival_head_core(original, self.bundle.model,
                                               arm=self.arm if enabled else "KPLUS")
        self.head_installed = self.core is not original
        if self.head_installed:
            if self._saved_head_receipt is not None:
                state = torch.load(self._saved_head_receipt["state_path"], map_location="cpu", weights_only=False)
                self.core.extension.load_state_dict(state)
            else:
                save_torch(self.output_root / "head_initial_state.pt", self.core.extension.state_dict())
        return self.head_installed

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
            state = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
            validate_snapshot(state, **self.validation_arguments(), completed_event=5)
            self._saved_head_receipt = state["events"][4]["survival_head"]
        self.bundle.draft_model.load_state_dict(state["draft_state_dict"], strict=True)
        # Adam's same-device CPU counters must not alias retained origin tensors.
        self.backend.load_state_dict(deepcopy(state["backend"]))
        self.global_step, self.presentations = state["global_step"], state["presentations"]
        self.events = deepcopy(state["events"])
        self.core._micro = self.presentations
        restore_rng(state["rng"])
        receipt = native.verify_live(self, state)
        receipt.update(origin_state=str(self.origin_path), origin_export=str(self.origin_export),
            historical_receipts_unchanged=True, source_step=512, seed_reset=False,
            changes=["additive survival-head contract", "run/output metadata"])
        write_json(self.output_root / "RESUME_CHECK.json", receipt)

    def save_initial(self, *args, **kwargs):
        raise ValueError("Survival pilot resumes complete KPLUS4; released initialization is forbidden")

    def _prepare_event(self, event_id, feature_paths, passes):
        if len(self.events) + 1 not in EVENTS:
            raise ValueError("Survival pilot stops after event five")
        if self.arm != "KPLUS" and not self.head_installed:
            raise ValueError("Install the head after discarded native startup")
        features = native.FullKTrainer._prepare_event(self, event_id, feature_paths, passes)
        return features

    def train_event(self, event_id, feature_paths, passes=2, *, reference_check=None, presentation_schedule=None):
        if reference_check is not None:
            raise ValueError("No extra model probes are part of the main survival event")
        ordinary_backward = self.backend.backward

        def observed_backward(*args, **kwargs):
            result = ordinary_backward(*args, **kwargs)
            self.work["response_backwards"] = self.work.get("response_backwards", 0) + 1
            return result

        self.backend.backward = observed_backward
        try:
            result = SpecForgeEventTrainer.train_event(self, event_id, feature_paths, passes,
                                                       presentation_schedule=presentation_schedule)
        finally:
            self.backend.backward = ordinary_backward
        validate_event_receipt(result, arm=self.arm, dataset=self.dataset,
                               event_number=len(self.events), trainer_status=self.status())
        self.completed_event = len(self.events)
        return result

    def _train_passes(self, features, passes):
        return historical.RoleLossTrainer._train_passes(self, features, passes)

    def _checkpoint_state(self, event):
        import torch

        head = None
        if self.arm != "KPLUS":
            state = self.core.extension.state_dict()
            validate_head_state(state, arm=self.arm, expected_steps=128,
                                native_recipe=self.backend.optimizer.recipe)
            path = self.events_root / event["event_id"] / "head_state.pt"
            save_torch(path, state)
            head = dict(contract=survival_head_contract(self.arm), state_path=str(path),
                        state_identity=identity(path), steps=state["steps"])
        windows = [row for row in event["records"] if row["optimizer_stepped"]]
        sums = [row["response_observation"]["sum_metrics"] for row in event["records"]]
        numerator = sum(row.get("head_bce_numerator", 0) for row in sums)
        count = sum(row.get("head_valid_proposals", 0) for row in sums)
        work = dict(optimizer_steps=len(windows), ordinary_presentations=len(event["records"]),
            response_forwards=self.work.get("response_forwards", 0),
            response_backwards=self.work.get("response_backwards", 0),
            head_optimizer_steps=0 if head is None else head["steps"])
        if work != expected_work(self.arm, 5):
            raise ValueError("Actual survival event work differs from the one-event budget")
        device = next(self.bundle.draft_model.parameters()).device
        costs = dict(
            head_bce_response_mean=sum(row.get("head_response_bce_sum", 0) for row in sums) / 256,
            head_bce_pooled=numerator / max(count, 1),
            backbone_clipped_windows=sum(row["metrics"]["grad_norm"] > self.backend.optimizer.max_grad_norm for row in windows),
            backbone_optimizer_windows=len(windows),
            head_clipped_windows=sum(row["metrics"].get("head_grad_norm", 0) > self.backend.optimizer.max_grad_norm for row in windows),
            head_optimizer_windows=0 if head is None else head["steps"],
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
            peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0,
        )
        event.update(method=self.arm, dataset=self.dataset, kplus_loss=kplus_contract(),
                     survival_head=head, survival_work=work, survival_costs=costs)
        validate_event_receipt(event, arm=self.arm, dataset=self.dataset,
                               event_number=5, require_completed=False)
        return SpecForgeEventTrainer._checkpoint_state(self, event)

    def verify_boundary(self, export_path=None, receipt_path=None):
        import torch

        started = time.perf_counter()
        with preserve_rng():
            state = torch.load(self.latest, map_location="cpu", weights_only=False, mmap=True)
            receipt = validate_snapshot(state, **self.validation_arguments(), completed_event=len(self.events))
            if export_path is not None:
                _export_equal(state, export_path)
                receipt["export_exact"] = True
            receipt["live"] = native.verify_live(self, state)
            if self.arm != "KPLUS":
                if not self.head_installed:
                    raise ValueError("Restore separate head before live endpoint validation")
                head_path = state["events"][4]["survival_head"]["state_path"]
                saved = torch.load(head_path, map_location="cpu", weights_only=False)
                if differences(self.core.extension.state_dict(), saved):
                    raise ValueError("Live head differs from its separate endpoint")
                receipt["head_live_exact"] = True
        receipt["wall_s"] = time.perf_counter() - started
        if receipt_path is not None:
            write_json(receipt_path, receipt)
        return receipt


# Short campaign-facing name retained alongside the descriptive class name.
SurvivalTrainer = SurvivalHeadTrainer
