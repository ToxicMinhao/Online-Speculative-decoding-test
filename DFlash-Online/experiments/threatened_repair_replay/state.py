"""Exact transient state for disposable branches inside an active event.

The physical-work ledger deliberately does not belong to these snapshots.
Neither a diagnostic restore nor a failed branch can erase executed work.
"""
from copy import deepcopy
import math

from experiments.full_k_gsm8k import training as native
from experiments.ok_continuation.resume import differences
from online_feedback.rng import restore_rng
from online_feedback.storage import write_json


BOOKKEEPING = (
    "response_records", "window_records", "work", "_anchor_observations",
    "presentation_ordinal", "_active_event", "_pending_capture", "_pending_gradient",
    "firing_capture", "_replaying", "diagnostic_capture_s", "capture_first_firing",
)


def copy_tree(value):
    import torch
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return type(value)((name, copy_tree(item)) for name, item in value.items())
    if isinstance(value, (list, tuple)):
        return type(value)(copy_tree(item) for item in value)
    return deepcopy(value)


def named_gradients(trainer):
    return {name: None if parameter.grad is None else parameter.grad.detach().cpu().clone()
            for name, parameter in trainer.bundle.draft_model.named_parameters() if parameter.requires_grad}


def capture_state(trainer):
    """Capture a clean pair boundary while retaining its active schedule cursor."""
    from experiments.functional_step_geometry.qualify import runtime_state
    if trainer.core.accumulation_remainder or trainer.core._micro != trainer.presentations:
        raise ValueError("A threatened-replay snapshot requires a complete native optimizer window")
    model = trainer.bundle.model
    schedule = trainer._presentation_schedule
    trained = {id(p) for p in trainer.bundle.draft_model.parameters() if p.requires_grad}
    return dict(
        scientific=copy_tree(native.live_state(trainer)),
        gradients=named_gradients(trainer),
        core=copy_tree(dict(micro=trainer.core._micro, ratio_totals=trainer.core._ratio_totals,
                            sum_totals=trainer.core._sum_totals)),
        runtime=runtime_state(model),
        bookkeeping={name: copy_tree(getattr(trainer, name)) for name in BOOKKEEPING if hasattr(trainer, name)},
        costs=copy_tree(model.costs),
        schedule_object=schedule,
        schedule_state=None if schedule is None else copy_tree(vars(schedule)),
        sampler=model._sample_anchor_positions,
        original_sampler=trainer._original_sampler,
        frozen_versions=[(p, p._version) for p in model.parameters() if id(p) not in trained],
    )


def restore_state(trainer, snapshot):
    """Restore tensors, Adam, counters, observations, schedule, modes, then RNG."""
    import torch
    from experiments.functional_step_geometry.qualify import restore_runtime
    state = snapshot["scientific"]
    trainer.bundle.draft_model.load_state_dict(state["draft_state_dict"], strict=True)
    trainer.backend.load_state_dict(copy_tree(state["backend"]))
    trainer.global_step, trainer.presentations = state["global_step"], state["presentations"]
    trainer.events = copy_tree(state["events"])
    trainer.core._micro = snapshot["core"]["micro"]
    trainer.core._ratio_totals = copy_tree(snapshot["core"]["ratio_totals"])
    trainer.core._sum_totals = copy_tree(snapshot["core"]["sum_totals"])
    with torch.no_grad():
        for name, parameter in trainer.bundle.draft_model.named_parameters():
            if parameter.requires_grad:
                gradient = snapshot["gradients"][name]
                parameter.grad = None if gradient is None else gradient.to(parameter.device).clone()
    for name, value in snapshot["bookkeeping"].items():
        setattr(trainer, name, copy_tree(value))
    trainer.bundle.model.costs = copy_tree(snapshot["costs"])
    schedule = snapshot["schedule_object"]
    trainer._presentation_schedule = schedule
    if schedule is not None:
        vars(schedule).clear()
        vars(schedule).update(copy_tree(snapshot["schedule_state"]))
    trainer.bundle.model._sample_anchor_positions = snapshot["sampler"]
    trainer._original_sampler = snapshot["original_sampler"]
    restore_runtime(trainer.bundle.model, snapshot["runtime"])
    restore_rng(state["rng"])


def _runtime_differences(model, expected):
    """Compare values plus cache bindings without copying opaque attention masks."""
    from experiments.functional_step_geometry.qualify import CACHE_FIELDS
    modules = dict(model.named_modules())
    mismatch = differences({name: module.training for name, module in modules.items()}, expected["modes"], "modes")
    mismatch += differences({name: value.detach().cpu() for name, value in model.named_buffers()}, expected["buffers"], "buffers")
    mismatch += differences({name: set(module._non_persistent_buffers_set) for name, module in modules.items()}, expected["nonpersistent"], "nonpersistent")
    for name, fields in expected["caches"].items():
        module = modules[name]
        for field in CACHE_FIELDS:
            if hasattr(module, field) != (field in fields):
                mismatch.append(f"cache.{name}.{field}.presence")
            elif field in fields:
                current, old = getattr(module, field), fields[field]
                if isinstance(old, (type(None), bool, int, float, str)):
                    if current != old:
                        mismatch.append(f"cache.{name}.{field}.value")
                elif current is not old:
                    mismatch.append(f"cache.{name}.{field}.binding")
    return mismatch


def verify_state(trainer, snapshot, receipt_path=None):
    """An in-event checker: a non-null, exact schedule is valid here."""
    mismatch = differences(native.live_state(trainer), snapshot["scientific"], "scientific")
    mismatch += differences(named_gradients(trainer), snapshot["gradients"], "gradients")
    mismatch += differences(dict(micro=trainer.core._micro, ratio_totals=trainer.core._ratio_totals,
                                 sum_totals=trainer.core._sum_totals), snapshot["core"], "core")
    mismatch += differences({name: getattr(trainer, name) for name in snapshot["bookkeeping"]}, snapshot["bookkeeping"], "bookkeeping")
    mismatch += differences(trainer.bundle.model.costs, snapshot["costs"], "costs")
    schedule = trainer._presentation_schedule
    if schedule is not snapshot["schedule_object"]:
        mismatch.append("schedule.object")
    mismatch += differences(None if schedule is None else vars(schedule), snapshot["schedule_state"], "schedule")
    if trainer.bundle.model._sample_anchor_positions != snapshot["sampler"] or trainer._original_sampler != snapshot["original_sampler"]:
        mismatch.append("sampler.binding")
    if any(parameter._version != version for parameter, version in snapshot["frozen_versions"]):
        mismatch.append("frozen_parameter_version")
    mismatch += _runtime_differences(trainer.bundle.model, snapshot["runtime"])
    if trainer.closed or trainer.failed or trainer.core.accumulation_remainder:
        mismatch.append("unclean_boundary")
    receipt = dict(status="passed" if not mismatch else "failed", mismatches=mismatch[:64],
                   global_step=trainer.global_step, presentations=trainer.presentations,
                   schedule_position=None if schedule is None else schedule.position,
                   exact_weights_adam_rng_gradients_runtime=True if not mismatch else False)
    if receipt_path is not None:
        write_json(receipt_path, receipt)
    if mismatch:
        raise ValueError("Transient native state differs: " + str(mismatch[:8]))
    return receipt


def tensor_change(left, right):
    """FP64 scalar accounting in bounded CPU slices, with native bit comparisons."""
    import torch
    left_sq = right_sq = difference_sq = dot = 0.
    changed = elements = 0
    if left.keys() != right.keys():
        raise ValueError("Parameter/gradient membership changed")
    for name, a in left.items():
        b = right[name]
        if a is None or b is None:
            if a is not b:
                raise ValueError("Gradient presence changed: " + name)
            continue
        if a.dtype != b.dtype or a.shape != b.shape:
            raise ValueError("Tensor shape or dtype changed: " + name)
        aa, bb = a.detach().flatten(), b.detach().flatten()
        for offset in range(0, aa.numel(), 1 << 20):
            x = aa[offset:offset + (1 << 20)].to(device="cpu", dtype=torch.float64)
            y = bb[offset:offset + (1 << 20)].to(device="cpu", dtype=torch.float64)
            if not bool(torch.isfinite(x).all() and torch.isfinite(y).all()):
                bad = ~(torch.isfinite(x) & torch.isfinite(y))
                indices = bad.nonzero().flatten()[:8]
                sample = [(offset + int(index), float(x[index]), float(y[index])) for index in indices]
                raise ValueError(f"Nonfinite update/gradient accounting: {name}, "
                                 f"slice_bad_count={int(bad.sum())}, index/left/right={sample}")
            delta = y - x
            left_sq += float(x.square().sum())
            right_sq += float(y.square().sum())
            difference_sq += float(delta.square().sum())
            dot += float((x * delta).sum())
            changed += int((x != y).sum())
            elements += x.numel()
    return dict(left_norm=math.sqrt(left_sq), right_norm=math.sqrt(right_sq),
                difference_norm=math.sqrt(difference_sq), left_dot_difference=dot,
                changed_entries=changed, elements=elements)


def compare_final_native(trainer, retained_path, receipt_path):
    """Compare scientific tensors/counters/RNG; allow only additive receipt metadata."""
    import torch
    from online_feedback.rng import preserve_rng
    with preserve_rng():
        retained = torch.load(retained_path, map_location="cpu", weights_only=False, mmap=True)
        actual = native.live_state(trainer)
        fields = ("draft_state_dict", "backend", "rng", "global_step", "presentations",
                  "strategy", "world_size", "reduction", "accumulation_steps", "horizon")
        mismatch = differences({name: actual[name] for name in fields}, {name: retained[name] for name in fields})
        for current, old in zip(actual["events"][-1]["records"], retained["events"][-1]["records"]):
            for name in ("record_id", "pass", "optimizer_step_after", "optimizer_stepped", "sampled_anchors"):
                mismatch += differences(current[name], old[name], "event_007." + name)
        if len(actual["events"]) != 7 or len(actual["events"][-1]["records"]) != 256:
            mismatch.append("event_007.accounting")
        if trainer.global_step != 896 or trainer.presentations != 1792:
            mismatch.append("event_007.counters")
        receipt = dict(status="passed" if not mismatch else "failed", mismatches=mismatch[:64],
                       global_step=trainer.global_step, presentations=trainer.presentations,
                       weights_optimizer_counters_rng_exact=not mismatch,
                       metadata_scope="run/output identity and additive diagnostics may differ")
        write_json(receipt_path, receipt)
        if mismatch:
            raise ValueError("Final native continuation differs from retained NATIVE7: " + str(mismatch[:8]))
        return receipt
