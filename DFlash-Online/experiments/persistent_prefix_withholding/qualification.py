"""Eight fixed first-pair applications across two separately initialized cards.

Only compact complete tensor identities are written. The restored K6 CPU payload
is reused read-only; no model-sized qualification gradient archive is created.
Call after the separately counted, qualified two-backward startup.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
import traceback

import torch

from experiments.full_k_gsm8k import training as native
from experiments.functional_step_geometry.qualify import runtime_state, restore_runtime
from experiments.native_prefix_loss.qualification import gradient_role_check
from experiments.ok_continuation.resume import differences
from experiments.recent_repair_retention.training import schedule_adapter
from experiments.threatened_repair_replay.training import ThreatReplayTrainer
from online_feedback.native_prefix_roles import install_native_prefix_roles
from online_feedback.presentation_schedule import PresentationSchedule
from online_feedback.rng import capture_rng, restore_rng
from online_feedback.storage import write_json
from online_feedback.training import SpecForgeEventTrainer
from . import state as contracts

CASES = {"A": (("legacy_N", "N", None), ("ALL", "N", "ALL"), ("E", "E", None),
               ("E_repeat", "E", None), ("S", "S", None)),
         "B": (("N", "N", None), ("E", "E", None), ("S", "S", None))}


def tensor_identity(value):
    if value is None:
        return None
    if not torch.is_tensor(value):
        raise TypeError("Expected tensor or absent gradient")
    digest = hashlib.sha256()
    flat = value.detach().reshape(-1)
    finite = True
    for start in range(0, flat.numel(), 1 << 20):
        part = flat[start:start + (1 << 20)].to(device="cpu").contiguous()
        if part.is_floating_point():
            finite = finite and bool(torch.isfinite(part).all())
        digest.update(part.view(torch.uint8).numpy().tobytes())
    if not finite:
        raise ValueError("Nonfinite tensor in complete scientific identity")
    return dict(shape=list(value.shape), dtype=str(value.dtype), elements=value.numel(),
                sha256=digest.hexdigest(), finite=finite)


def tree_identity(value):
    import numpy as np
    if torch.is_tensor(value):
        return tensor_identity(value)
    if isinstance(value, np.ndarray):
        return dict(dtype=str(value.dtype), shape=list(value.shape), sha256=hashlib.sha256(value.tobytes()).hexdigest())
    if isinstance(value, dict):
        return {str(key): tree_identity(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [tree_identity(item) for item in value]
    return value


def scientific_identity(trainer):
    optimizer = trainer.backend.optimizer.state_dict()
    return dict(weights={name: tensor_identity(parameter) for name, parameter in trainer.bundle.draft_model.named_parameters()
                         if parameter.requires_grad}, optimizer=tree_identity(optimizer),
                rng=tree_identity(capture_rng()), global_step=trainer.global_step,
                presentations=trainer.presentations, micro=trainer.core._micro,
                accumulation_remainder=trainer.core.accumulation_remainder)


def forward_identity(rows):
    # The gradient/clip norm is an update result, not an equal-forward criterion.
    return [dict(record_id=row["record_id"], pass_index=row["pass"], sampled_anchors=row["sampled_anchors"],
                 loss=row["response_observation"]["loss"],
                 sum_metrics=row["response_observation"]["sum_metrics"],
                 ratio_metrics=row["response_observation"]["ratio_metrics"],
                 metrics={key:value for key,value in row["metrics"].items() if key != "grad_norm"})
            for row in rows]


def _restore_arm(trainer, source, runtime, arm, role):
    """Explicit disposable migration; never used in production continuation."""
    import yaml
    if not trainer.qualification_only:
        raise ValueError("Qualification state switching is forbidden in production")
    trainer.backend.optimizer.zero_grad()
    trainer.core._ratio_totals.clear()
    trainer.core._sum_totals.clear()
    config = yaml.safe_load(source["config_text"])
    config["training"]["learning_rate"] = contracts.arm_lr(arm)
    trainer.config_text = yaml.safe_dump(config)
    trainer.cfg.training.learning_rate = contracts.arm_lr(arm)
    trainer._arm, trainer.qualification_role = arm, role
    candidate, migration = contracts.fork_state(source, arm=arm, binding=trainer.binding,
        config_text=trainer.config_text, run_id=trainer.cfg.run_id, qualification_only=True)
    trainer.contract = candidate["algorithm_contract"]
    trainer.backend.optimizer.recipe["optimizer_options"]["lr"] = contracts.arm_lr(arm)
    trainer.backend.optimizer.optimizer.param_groups[0]["lr"] = contracts.arm_lr(arm)
    trainer.bundle.draft_model.load_state_dict(candidate["draft_state_dict"], strict=True)
    saved_backend = candidate["backend"]
    # CPU fixture Adam may retain CPU tensor aliases. Production GPU loading
    # converts these CPU tensors into distinct device allocations itself.
    if next(trainer.bundle.draft_model.parameters()).device.type == "cpu":
        saved_backend = deepcopy(saved_backend)
    trainer.backend.load_state_dict(saved_backend)
    trainer.global_step, trainer.presentations = source["global_step"], source["presentations"]
    trainer.core._micro = trainer.presentations
    trainer.events = deepcopy(source["events"])
    trainer.response_records.clear()
    trainer.window_records.clear()
    trainer.role_receipts.clear()
    trainer._anchor_observations.clear()
    trainer._presentation_schedule = None
    trainer.presentation_ordinal = None
    trainer.bundle.model.native_prefix_roles = None
    selected = role or ("E" if arm == "E" else None)
    trainer.role_controller = (install_native_prefix_roles(trainer.bundle.model, selected, capture_gradients=True)
                               if selected is not None else None)
    restore_runtime(trainer.bundle.model, runtime)
    restore_rng(source["rng"])
    trainer._install_current_role()
    return migration


def _assert_equal(left, right, label):
    mismatch = differences(left, right)
    if mismatch:
        raise ValueError(label + ": " + str(mismatch[:16]))


def qualify(trainer, feature_paths, output, *, device_slot):
    """Run only this card's declared cases; output persisted before each gate."""
    if device_slot not in CASES or not trainer.qualification_only:
        raise ValueError("Require one bounded qualification worker on card A or B")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    source = torch.load(trainer.origin_path, map_location="cpu", weights_only=False, mmap=True)
    runtime = runtime_state(trainer.bundle.model)
    started = time.perf_counter()
    report = dict(status="running", slot=device_slot, cases=[], optimizer_applications_attempted=0,
                  optimizer_applications_completed=0, response_presentations_completed=0, response_presentations_attempted=0,
                  complete_generations=0, work_before=dict(trainer.work), startup_separate=True)
    original_forward = trainer._native_forward
    try:
        features = SpecForgeEventTrainer._prepare_event(trainer, "event_007", feature_paths, 2)
        for label, arm, role in CASES[device_slot]:
            migration = _restore_arm(trainer, source, runtime, arm, role)
            trainer._active_event = "qualification_" + label
            (trainer.events_root / trainer._active_event).mkdir()
            trainer._presentation_schedule = PresentationSchedule(schedule_adapter(trainer.schedule, features),
                [row["record_id"] for row in features], 2, features=features)
            trainer.bundle.model.begin_event()
            order = torch.randperm(len(features)).tolist()
            pair = [features[index] for index in order[:2]]
            row = dict(case=label, arm=arm, status="attempted", migration=migration,
                       fixed_pair=[feature["record_id"] for feature in pair])
            report["cases"].append(row)
            report["optimizer_applications_attempted"] += 1
            report["response_presentations_attempted"] += 2
            write_json(output / "RESULT.json", report)
            observation_started = 0.
            def observe(current):
                nonlocal observation_started
                began = time.perf_counter()
                row["preclip"] = {name: tensor_identity(parameter.grad)
                                  for name, parameter in current.bundle.draft_model.named_parameters()
                                  if parameter.requires_grad}
                if any(value is not None and not value["finite"] for value in row["preclip"].values()):
                    raise ValueError("Nonfinite qualification preclip gradients")
                observation_started += time.perf_counter() - began
            trainer.step_observer = observe
            before_steps = trainer.global_step
            pair_started = time.perf_counter()
            if label == "legacy_N":
                # Use the retained native pair implementation, without the new
                # role forward wrapper. It reaches the identical native core.
                trainer._native_forward = trainer._uncontrolled_forward
                trainer.ledger = None
                bundle = ThreatReplayTrainer.run_pair(trainer, pair, 0, 0)
                rows = bundle["rows"]
            else:
                rows = trainer.run_pair(pair, 0, 0)
            pair_elapsed = time.perf_counter() - pair_started
            trainer._native_forward = original_forward
            trainer.step_observer = None
            report["optimizer_applications_completed"] += trainer.global_step - before_steps
            report["response_presentations_completed"] += len(rows)
            row.update(status="completed", forward=forward_identity(rows),
                       endpoint=scientific_identity(trainer),
                       pair_wall_s=pair_elapsed, preclip_observation_s=observation_started,
                       pair_wall_excluding_preclip_observer_s=pair_elapsed-observation_started,
                       work_cumulative=dict(trainer.work))
            if arm == "E":
                roles = dict(responses=[response for receipt in trainer.role_receipts for response in receipt["responses"]])
                row["gradient_roles"] = gradient_role_check(roles, "E")
            write_json(output / "RESULT.json", report)
            if arm == "E" and row["gradient_roles"]["status"] != "passed":
                raise ValueError("The fixed pair failed E role-gradient qualification")
            baseline = report["cases"][0]
            if arm == "E":
                _assert_equal(row["forward"], baseline["forward"], "E native forward/selection/weights")
            if label == "ALL":
                for field in ("forward", "preclip", "endpoint"):
                    _assert_equal(row[field], baseline[field], "ALL versus legacy N " + field)
            if label == "E_repeat":
                first = next(item for item in report["cases"] if item["case"] == "E")
                for field in ("forward", "preclip", "endpoint", "gradient_roles"):
                    _assert_equal(row[field], first[field], "Repeated E " + field)
            if label == "S":
                _assert_equal(row["forward"], baseline["forward"], "S native forward")
                _assert_equal(row["preclip"], baseline["preclip"], "S native preclip")
                for field in ("state",):
                    _assert_equal(row["endpoint"]["optimizer"]["optimizer_state_dict"][field],
                                  baseline["endpoint"]["optimizer"]["optimizer_state_dict"][field], "S native Adam moments/counters")
            row["checks"] = "passed"
            write_json(output / "RESULT.json", report)
        _restore_arm(trainer, source, runtime, "N", None)
        restored = scientific_identity(trainer)
        expected = dict(weights={name:tensor_identity(value) for name,value in source["draft_state_dict"].items()},
            optimizer=tree_identity(source["backend"]["optimizer"]), rng=tree_identity(source["rng"]),
            global_step=768, presentations=1536, micro=1536, accumulation_remainder=0)
        _assert_equal(restored, expected, "Final exact K6 restoration")
        report.update(status="passed", exact_origin_restoration=True)
    except BaseException as error:
        report.update(status="failed", error=repr(error), traceback=traceback.format_exc(), automatic_retry=False)
        raise
    finally:
        trainer._native_forward = original_forward
        trainer.step_observer = None
        report.update(wall_s=time.perf_counter()-started, work_after=dict(trainer.work),
            live_global_step=trainer.global_step, live_presentations=trainer.presentations,
            optimizer_applications_unknown=report["optimizer_applications_attempted"]-report["optimizer_applications_completed"],
            response_presentations_unknown=report["response_presentations_attempted"]-report["response_presentations_completed"])
        write_json(output / "RESULT.json", report)
    return report


def compare_cards(first, second):
    """The same scientific endpoints/gradients/RNG, independent physical cards."""
    if first["status"] != "passed" or second["status"] != "passed":
        raise ValueError("Both native card gates must pass before cross-card comparison")
    a = {row["case"]: row for row in first["cases"]}
    b = {row["case"]: row for row in second["cases"]}
    comparisons = []
    for left, right in (("ALL", "N"), ("E", "E"), ("S", "S")):
        for field in ("fixed_pair", "forward", "preclip", "endpoint"):
            _assert_equal(a[left][field], b[right][field], f"Cross-card {left}/{right} {field}")
        comparisons.append(dict(card_A=left, card_B=right, exact_scientific_parity=True))
    return dict(status="passed", comparisons=comparisons, attempted_optimizer_applications=8,
                complete_response_generations=0, physical_device_metadata_compared_separately=True)
