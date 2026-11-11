"""Scheduled exposure and disposable identity/active windows from the exact parent."""

import argparse
from copy import deepcopy
from pathlib import Path
import time
import traceback

from online_feedback.storage import write_json, save_torch
from experiments.survival_head.data import read
from experiments.successor_shuffle.campaign import shared_inputs
from experiments.successor_shuffle.qualification import next_anchor_draw
from experiments.ok_coding.storage import identity


def build_trainer(plan, output, arm="SECOND"):
    from experiments.prefix_guard.qualify import pin_retained_graph_runtime
    from experiments.functional_step_geometry.native_pilot import discarded_startup
    from .training import SecondFollowerTrainer

    pool, baseline = shared_inputs(plan)
    features = {r["record_id"]: Path(r["feature"]["path"]) for r in pool["records"]}
    pin_retained_graph_runtime(output)
    trainer = SecondFollowerTrainer(
        plan["configs"][arm],
        output / "trainer",
        arm=arm,
        binding=plan["binding"],
        original_config_text=Path(plan["original_config"]).read_text(),
        origin_state=plan["origin_state"],
        origin_export=plan["origin_export"],
    )
    startup = discarded_startup(
        trainer, list(features.values())[:2], output / "startup"
    )
    return trainer, features, baseline, startup


def compare_weight_mappings(path, actual, expected):
    """State dictionaries may use different mapping classes; tensors stay exact."""
    from experiments.prefix_guard.qualify import persist_comparison

    return persist_comparison(path, dict(actual), dict(expected))


def inspect_exposure(root):
    import torch
    from specforge.training.strategies.base import StepContext
    from online_feedback.rng import capture_rng
    from experiments.prefix_guard.qualify import persist_comparison

    root = Path(root)
    plan = read(root / "sealed/PLAN.json")
    output = root / "qualification/exposure"
    output.mkdir(parents=True, exist_ok=False)
    receipt = dict(
        status="running", response_forwards=0, optimizer_steps=0, response_backwards=0
    )
    trainer = None
    started = time.perf_counter()
    try:
        trainer, features, baseline, receipt["startup"] = build_trainer(plan, output)
        trainer.install_second()
        trainer._active_event = "exposure"
        (trainer.events_root / "exposure").mkdir()
        trainer.bundle.model.begin_event()
        order = torch.randperm(len(features)).tolist()
        ordered = [list(features)[i] for i in order]
        if ordered != [r["record_id"] for r in baseline["records"][:128]]:
            raise ValueError("Exposure order differs from retained first pass")
        receipts = []
        first_active = None
        for window in range(8):
            save_torch(output / f"window_{window:02d}_rng.pt", capture_rng())
            pair = []
            for ordinal in range(2 * window, 2 * window + 2):
                raw = torch.load(
                    features[ordered[ordinal]], map_location="cpu", weights_only=True
                )
                trainer._anchor_observations.clear()
                with torch.no_grad():
                    result = trainer.core.strategy.forward_loss(
                        trainer._batch(raw),
                        StepContext(
                            global_step=512,
                            total_steps=trainer.horizon,
                            collect_detailed_metrics=True,
                        ),
                    )
                receipt["response_forwards"] += 1
                float(result.loss.detach())
                expected = baseline["records"][ordinal]
                if trainer._anchor_observations != expected["sampled_anchors"]:
                    raise ValueError("Exposure anchors differ from retained schedule")
                pair.append(deepcopy(trainer.response_records[-1]))
                del result, raw
            receipts.append(
                dict(
                    window=window,
                    record_ids=ordered[2 * window : 2 * window + 2],
                    responses=pair,
                )
            )
            if first_active is None and any(
                r["sum_metrics"]["second_follower_transferred_weight"] > 0 for r in pair
            ):
                first_active = window
            write_json(output / "OBSERVATIONS.json", receipts)
        receipt.update(
            status="completed",
            first_active_window=first_active,
            windows=receipts,
            exposure_is_parent_state_no_update=True,
            next_anchor_state_identity=identity(output / "window_00_rng.pt"),
        )
        # Never let no-grad screening become the production state.
        receipt["parent_weights_unchanged"] = compare_weight_mappings(
            output / "PARENT_WEIGHTS.json",
            trainer.bundle.draft_model.state_dict(),
            trainer.origin["draft_state_dict"],
        )
    except BaseException as error:
        receipt.update(
            status="failed", error=repr(error), traceback=traceback.format_exc()
        )
        raise
    finally:
        if trainer is not None:
            trainer.close()
        receipt["wall_s"] = time.perf_counter() - started
        if torch.cuda.is_initialized():
            receipt["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        write_json(output / "RESULT.json", receipt)
    return receipt


def coordinate_changes(actual, reference):
    import torch

    rows = []
    for name, a in actual.items():
        b = reference[name]
        if a is None or b is None:
            if a is not b:
                raise ValueError("Gradient presence differs")
            continue
        delta = a.float() - b.float()
        rows.append(
            dict(
                name=name,
                shape=list(a.shape),
                dtype=str(a.dtype),
                coordinates=a.numel(),
                changed=int(torch.count_nonzero(a != b)),
                max_abs=float(delta.abs().max()),
                l2=float(torch.linalg.vector_norm(delta)),
            )
        )
    return dict(
        changed=sum(r["changed"] for r in rows),
        coordinates=sum(r["coordinates"] for r in rows),
        parameters=rows,
    )


def run_case(root, case):
    import torch
    from specforge.training.strategies.base import StepContext
    from experiments.kplus_abt.qualification import endpoint, finite_endpoint
    from experiments.prefix_guard.qualify import persist_comparison
    from online_feedback.rng import restore_rng

    root = Path(root)
    plan = read(root / "sealed/PLAN.json")
    exposure = read(root / "qualification/exposure/RESULT.json")
    # Native and disabled cases use the same positive window where available.
    window = (
        exposure["first_active_window"]
        if exposure["first_active_window"] is not None
        else 0
    )
    output = root / "qualification" / case
    output.mkdir(parents=True, exist_ok=False)
    receipt = dict(
        status="running",
        case=case,
        window=window,
        response_forwards=0,
        response_backwards=0,
        optimizer_steps=0,
    )
    trainer = None
    started = time.perf_counter()
    gradients = {}
    clipped = {}
    try:
        trainer, features, baseline, receipt["startup"] = build_trainer(
            plan, output, arm="NATIVE" if case == "NATIVE" else "SECOND"
        )
        if case != "NATIVE":
            trainer.install_second(disabled=case == "ZERO")
        trainer._active_event = "qualification"
        (trainer.events_root / "qualification").mkdir()
        trainer.bundle.model.begin_event()
        restore_rng(
            torch.load(
                root / f"qualification/exposure/window_{window:02d}_rng.pt",
                weights_only=False,
            )
        )
        original = trainer.backend.optimizer.step

        def step():
            for n, p in trainer.bundle.draft_model.named_parameters():
                gradients[n] = None if p.grad is None else p.grad.detach().cpu().clone()
            return original()

        trainer.backend.optimizer.step = step
        adam = trainer.backend.optimizer.optimizer.step

        def adam_step(*args, **kwargs):
            for n, p in trainer.bundle.draft_model.named_parameters():
                clipped[n] = None if p.grad is None else p.grad.detach().cpu().clone()
            result = adam(*args, **kwargs)
            receipt["optimizer_steps"] += 1
            return result

        trainer.backend.optimizer.optimizer.step = adam_step
        for record in baseline["records"][2 * window : 2 * window + 2]:
            raw = torch.load(
                features[record["record_id"]], map_location="cpu", weights_only=True
            )
            trainer._anchor_observations.clear()
            result = trainer.core.train_step(
                trainer._batch(raw),
                StepContext(
                    global_step=trainer.global_step,
                    total_steps=trainer.horizon,
                    collect_detailed_metrics=True,
                ),
            )
            receipt["response_forwards"] += 1
            receipt["response_backwards"] += 1
            result.materialize_metrics()
            trainer.presentations += 1
            trainer.global_step += int(result.optimizer_stepped)
            if trainer._anchor_observations != record["sampled_anchors"]:
                raise ValueError("Qualification anchor mismatch")
        saved = endpoint(trainer, gradients)
        saved["clipped_gradients"] = clipped
        raw = torch.load(
            next(iter(features.values())), map_location="cpu", weights_only=True
        )
        saved["next_anchor_draw"] = next_anchor_draw(trainer, raw)
        save_torch(output / "ENDPOINT.pt", saved)
        receipt.update(
            endpoint_identity=identity(output / "ENDPOINT.pt"),
            finite=finite_endpoint(saved),
            observations=trainer.response_records,
            physical_work=trainer.work,
            next_anchor_draws=1,
            resume=read(trainer.output_root / "RESUME_CHECK.json"),
        )
        if case != "NATIVE":
            ref = torch.load(
                root / "qualification/NATIVE/ENDPOINT.pt",
                map_location="cpu",
                weights_only=False,
                mmap=True,
            )
            fields = (
                tuple(saved)
                if case == "ZERO"
                else (
                    "rng",
                    "anchors",
                    "global_step",
                    "presentations",
                    "micro",
                    "remainder",
                    "next_anchor_draw",
                )
            )
            receipt["exact"] = persist_comparison(
                output / "EXACT.json",
                {k: saved[k] for k in fields},
                {k: ref[k] for k in fields},
            )
            receipt["gradient_changes"] = coordinate_changes(
                saved["gradients"], ref["gradients"]
            )
            receipt["weight_changes"] = coordinate_changes(
                saved["weights"], ref["weights"]
            )
        if (
            receipt["optimizer_steps"],
            trainer.global_step,
            trainer.presentations,
            trainer.core.accumulation_remainder,
        ) != (1, 513, 1026, 0):
            raise ValueError("Disposable window counters differ")
        receipt["transferred_weight"] = sum(
            r["sum_metrics"].get("second_follower_transferred_weight", 0.0)
            for r in trainer.response_records
        )
        receipt["active_intervention"] = receipt["transferred_weight"] > 0
        receipt["status"] = "completed"
    except BaseException as error:
        receipt.update(
            status="failed", error=repr(error), traceback=traceback.format_exc()
        )
        raise
    finally:
        if trainer is not None:
            trainer.close()
        receipt["wall_s"] = time.perf_counter() - started
        if torch.cuda.is_initialized():
            receipt["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        write_json(output / "RESULT.json", receipt)
    return receipt


if __name__ == "__main__":
    from experiments.role_loss_pilot.campaign import install_signal_handlers

    install_signal_handlers()
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument(
        "--case", choices=["exposure", "NATIVE", "ZERO", "ACTIVE"], required=True
    )
    a = p.parse_args()
    inspect_exposure(a.root) if a.case == "exposure" else run_case(a.root, a.case)
