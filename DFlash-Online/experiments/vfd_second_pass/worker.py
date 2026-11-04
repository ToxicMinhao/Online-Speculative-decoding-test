"""One bounded GPU training job; the parent owns serving and physical work totals."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import time
import traceback

from experiments.ok_coding.campaign import WorkLedger, read_work, require_complete_work
from experiments.full_k_gsm8k.training import live_state
from experiments.ok_continuation.resume import differences
from online_feedback.rng import capture_rng
from online_feedback.storage import write_json
from online_feedback.vfd_scoring import (
    score_response_snapshot,
    score_response_pair,
    select_responses,
)
from .contract import ARMS, EVENTS, PASS2, validate_plan, validate_subset
from .training import VFDTrainer, scientific_state


def read(path):
    return json.loads(Path(path).read_text())


def scoring(trainer, features, pool, event, ledger):
    import torch

    result = {}
    for feature in features:
        started = time.perf_counter()
        raw = torch.load(feature["path"], map_location="cpu", weights_only=True)
        result[feature["record_id"]] = ledger.call(
            "scoring_forwards",
            score_response_snapshot,
            trainer.bundle.model,
            trainer._batch(raw),
            event_id=event,
            response_id=feature["record_id"],
            conditioning_id=pool["identities"][feature["path"]]["sha256"],
        )
        result[feature["record_id"]]["costs"]["worker_total_s"] = (
            time.perf_counter() - started
        )
    return result


def scoring_seconds(snapshots):
    return sum(row["costs"]["worker_total_s"] for row in snapshots.values())


def scoring_compute_seconds(snapshots):
    return sum(row["costs"]["total_s"] for row in snapshots.values())


def verify_boundary_source(path, receipt):
    """Bind a rolling state to the previous completed worker before loading it."""
    from experiments.ok_events_5_8.campaign import identity

    path = Path(path)
    if (
        path.resolve() != Path(receipt["state_path"]).resolve()
        or identity(path) != receipt["identity"]
    ):
        raise ValueError(
            "Complete predecessor state differs from its committed identity"
        )
    return dict(status="passed", path=str(path), identity=receipt["identity"])


def qualification(root, plan, pool):
    """Two disposable native steps, eight bounded score invocations, no generation."""
    output = root / "campaign_001/qualification"
    output.mkdir(exist_ok=False)
    limits = dict(
        training_presentations=4, backwards=4, optimizer_steps=2, scoring_forwards=8
    )
    ledger = WorkLedger(output / "WORK.jsonl", limits)
    trainer = None
    result = dict(status="running", extra_generations=0)
    started = time.perf_counter()
    try:
        trainer = VFDTrainer(
            plan["config"],
            output / "trainer",
            original_config_text=Path(plan["original_config"]).read_text(),
        )
        result["initial"] = trainer.restore_original(plan["original_initial_state"])
        features = trainer.features(1, pool["features"])[:4]
        before_rng = capture_rng()
        before = scoring(trainer, features, pool, 1, ledger)
        if differences(before_rng, capture_rng()):
            raise ValueError("Qualification scoring changed training RNG")
        ledger.wrap(trainer.core.strategy, "forward_loss", "training_presentations")
        ledger.wrap(trainer.backend, "backward", "backwards")
        ledger.wrap(trainer.backend.optimizer.optimizer, "step", "optimizer_steps")
        trained = trainer.run_pass(features, event=1, pass_index=0)
        before_rng = capture_rng()
        after = scoring(trainer, features, pool, 1, ledger)
        if differences(before_rng, capture_rng()):
            raise ValueError("Qualification post-score changed training RNG")
        examples = [score_response_pair(before[k], after[k]) for k in before]
        state = scientific_state(live_state(trainer))
        # Exact tensor/RNG fork semantics, without additional updates or forwards.
        for arm in plan["arms"]:
            copy = deepcopy(state)
            if differences(state, copy):
                raise ValueError("Disposable complete-state fork differs: " + arm)
            del copy
        result.update(
            training=trained,
            examples=examples,
            before=before,
            after=after,
            fork_equality=True,
            policy_selection_cpu_tests_required=True,
        )
        result["restoration_after_discard"] = trainer.restore_original(
            plan["original_initial_state"]
        )
        require_complete_work(read_work(output / "WORK.jsonl", limits), limits)
        result["status"] = "passed"
    except BaseException as error:
        result.update(
            status="failed", error=repr(error), traceback=traceback.format_exc()
        )
        raise
    finally:
        if trainer is not None:
            if result["status"] != "passed":
                trainer.failed = True
            result["peak_allocated_bytes"] = __import__(
                "torch"
            ).cuda.max_memory_allocated()
            result["peak_reserved_bytes"] = __import__(
                "torch"
            ).cuda.max_memory_reserved()
            trainer.close()
        result["wall_s"] = time.perf_counter() - started
        result["work"] = read_work(output / "WORK.jsonl", limits)
        write_json(output / "RESULT.json", result)
    return result


def run(root, event, arm):
    from experiments.ok_events_5_8.campaign import verify_pool

    root = Path(root)
    plan = read(root / "sealed/PLAN.json")
    validate_plan(plan)
    if (
        type(event) is not int
        or not 1 <= event <= EVENTS
        or arm not in (*ARMS, "SHARED", "QUALIFICATION")
        or (arm in ("SHARED", "QUALIFICATION") and event != 1)
    ):
        raise ValueError("Worker identity is outside the fixed four-event screen")
    event_root = root / f"campaign_001/events/event_{event:03d}"
    pool = read(event_root / "POOL.json")
    verify_pool(pool)
    if arm == "QUALIFICATION":
        return qualification(root, plan, pool)
    output = event_root / arm
    output.mkdir(exist_ok=False)
    shared = arm == "SHARED"
    presentations = 128 if shared else (PASS2[arm] if event == 1 else 128 + PASS2[arm])
    score_limit = (
        256
        if shared
        else (0 if event == 1 else {"STATIC64": 128, "PROGRESS64": 256}.get(arm, 0))
    )
    limits = dict(
        training_presentations=presentations,
        backwards=presentations,
        optimizer_steps=presentations // 2,
        scoring_forwards=score_limit,
    )
    ledger = WorkLedger(output / "WORK.jsonl", limits)
    result = dict(status="running", event=event, arm=arm, initial_state=None)
    trainer = None
    started = time.perf_counter()
    try:
        if not shared:
            previous_result = (
                root / "campaign_001/events/event_001/SHARED/RESULT.json"
                if event == 1
                else root / f"campaign_001/events/event_{event-1:03d}/{arm}/RESULT.json"
            )
            predecessor = read(previous_result)
            if predecessor["status"] != "completed":
                raise ValueError("Cannot load an incomplete predecessor worker")
            state_path = (
                root / f'campaign_001/states/{"SHARED" if event == 1 else arm}.pt'
            )
            result["predecessor_state"] = verify_boundary_source(
                state_path, predecessor["state"]
            )
        trainer = VFDTrainer(
            plan["config"],
            output / "trainer",
            original_config_text=Path(plan["original_config"]).read_text(),
        )
        if shared:
            result["initial_state"] = trainer.restore_original(
                plan["original_initial_state"]
            )
        elif event == 1:
            result["initial_state"] = trainer.restore_boundary(
                root / "campaign_001/states/SHARED.pt",
                arm=arm,
                event=0,
                shared_fork=True,
            )
        else:
            result["initial_state"] = trainer.restore_boundary(
                root / f"campaign_001/states/{arm}.pt", arm=arm, event=event - 1
            )
        ledger.wrap(trainer.core.strategy, "forward_loss", "training_presentations")
        ledger.wrap(trainer.backend, "backward", "backwards")
        ledger.wrap(trainer.backend.optimizer.optimizer, "step", "optimizer_steps")
        features = trainer.features(event, pool["features"])
        snapshots_before = snapshots_after = {}
        scores = []
        score_reduction_s = 0.0
        if shared or (event > 1 and arm in ("STATIC64", "PROGRESS64")):
            snapshots_before = scoring(trainer, features, pool, event, ledger)
            write_json(output / "SCORES_BEFORE.json", snapshots_before)
        if shared or event > 1:
            order = (
                None
                if arm in ("SHARED", "FULL")
                else read(event_root / "FULL/RESULT.json")["pass1"]["full_order"]
            )
            pass1 = trainer.run_pass(
                features, event=event, pass_index=0, full_order=order
            )
        else:
            pass1 = read(event_root / "SHARED/RESULT.json")["pass1"]
        if shared or (event > 1 and arm == "PROGRESS64"):
            snapshots_after = scoring(trainer, features, pool, event, ledger)
            write_json(output / "SCORES_AFTER.json", snapshots_after)
        if snapshots_before:
            reduction_started = time.perf_counter()
            scores = [
                score_response_pair(snapshots_before[k], snapshots_after.get(k))
                for k in snapshots_before
            ]
            score_reduction_s = time.perf_counter() - reduction_started
            write_json(output / "SCORING.json", scores)
        if shared:
            result.update(
                pass1=pass1,
                pass2=None,
                selection=None,
                scoring_before_s=scoring_seconds(snapshots_before),
                scoring_after_s=scoring_seconds(snapshots_after),
                scoring_before_compute_s=scoring_compute_seconds(snapshots_before),
                scoring_after_compute_s=scoring_compute_seconds(snapshots_after),
                scoring_reduction_s=score_reduction_s,
            )
            result["physical_scoring_wall_s"] = (
                result["scoring_before_s"]
                + result["scoring_after_s"]
                + score_reduction_s
            )
            require_complete_work(read_work(output / "WORK.jsonl", limits), limits)
            verify_pool(pool)
            result["state"] = trainer.save_boundary(
                root / "campaign_001/states/SHARED.pt",
                arm="SHARED",
                event=0,
                shared_fork=True,
            )
            # Native pass files belong to the trainer, with an explicit copy for accounting.
            write_json(output / "pass_0.json", pass1)
        else:
            if event == 1 and arm in ("STATIC64", "PROGRESS64"):
                scores = read(event_root / "SHARED/SCORING.json")
                if arm == "STATIC64":
                    # STATIC never uses after-score information, even in the shared first event.
                    before = read(event_root / "SHARED/SCORES_BEFORE.json")
                    reduction_started = time.perf_counter()
                    scores = [score_response_pair(before[k]) for k in before]
                    score_reduction_s = time.perf_counter() - reduction_started
                write_json(output / "SCORING.json", scores)
            pass2 = None
            selection = None
            if arm != "ONCE":
                if arm == "FULL":
                    pass2 = trainer.run_pass(features, event=event, pass_index=1)
                    selection_started = time.perf_counter()
                    selection = select_responses(
                        [], pass2["full_order"], event_id=event, policy=arm
                    )
                    selection_s = time.perf_counter() - selection_started
                else:
                    selection_started = time.perf_counter()
                    full_order = read(event_root / "FULL/RESULT.json")["pass2"][
                        "full_order"
                    ]
                    selection = select_responses(
                        scores, full_order, event_id=event, policy=arm
                    )
                    validate_subset(full_order, selection["selected_ids"], arm)
                    selection_s = time.perf_counter() - selection_started
                    pass2 = trainer.run_pass(
                        features,
                        event=event,
                        pass_index=1,
                        full_order=full_order,
                        selected=selection["selected_ids"],
                    )
            else:
                selection_started = time.perf_counter()
                full_order = read(event_root / "FULL/RESULT.json")["pass2"][
                    "full_order"
                ]
                selection = select_responses([], full_order, event_id=event, policy=arm)
                selection_s = time.perf_counter() - selection_started
            by_id = {feature["record_id"]: feature for feature in features}
            selection["selected_response_lengths"] = [
                by_id[k]["response_tokens"] for k in selection["training_order"]
            ]
            result.update(pass1=pass1, pass2=pass2, selection=selection)
            before_s, after_s = scoring_seconds(snapshots_before), scoring_seconds(
                snapshots_after
            )
            scoring_compute_s = scoring_compute_seconds(
                snapshots_before
            ) + scoring_compute_seconds(snapshots_after)
            result["physical_scoring_wall_s"] = (
                before_s + after_s + score_reduction_s if snapshots_before else 0.0
            )
            result["physical_selection_s"] = selection_s + (
                score_reduction_s if not snapshots_before else 0.0
            )
            if event == 1 and arm in ("STATIC64", "PROGRESS64"):
                shared_receipt = read(event_root / "SHARED/RESULT.json")
                before_s = shared_receipt["scoring_before_s"]
                after_s = (
                    shared_receipt["scoring_after_s"] if arm == "PROGRESS64" else 0.0
                )
                scoring_compute_s = shared_receipt["scoring_before_compute_s"]
                if arm == "PROGRESS64":
                    scoring_compute_s += shared_receipt["scoring_after_compute_s"]
                    score_reduction_s = shared_receipt["scoring_reduction_s"]
            training_s = pass1["pass_wall_s"] + (pass2["pass_wall_s"] if pass2 else 0.0)
            training_compute_s = pass1["training_s"] + (
                pass2["training_s"] if pass2 else 0.0
            )
            scoring_s = before_s + after_s + score_reduction_s
            result.update(
                logical_updater_s=training_s + scoring_s + selection_s,
                costs=dict(
                    training_s=training_s,
                    training_compute_s=training_compute_s,
                    scoring_s=scoring_s,
                    scoring_compute_s=scoring_compute_s,
                    scoring_before_s=before_s,
                    scoring_after_s=after_s,
                    scoring_reduction_s=score_reduction_s,
                    selection_s=selection_s,
                ),
            )
            event_record = dict(
                event_id=f"event_{event:03d}",
                arm=arm,
                features=features,
                selected_ids=selection["selected_ids"],
                optimizer_steps=(128 + PASS2[arm]) // 2,
                response_presentations=128 + PASS2[arm],
                pass1=pass1,
                pass2=pass2,
            )
            require_complete_work(read_work(output / "WORK.jsonl", limits), limits)
            verify_pool(pool)
            # Export before state replacement keeps the declared transient peak bounded.
            if arm == "FULL" or event == 4:
                export = root / f"campaign_001/exports/{arm}_event_{event:03d}"
                result["publication"] = trainer.export(export)
            result["state"] = trainer.save_boundary(
                root / f"campaign_001/states/{arm}.pt",
                arm=arm,
                event=event,
                event_record=event_record,
            )
            result["state"]["replaced_predecessor_identity"] = (
                result["predecessor_state"]["identity"] if event > 1 else None
            )
            result["state"]["retired_direct_resume_boundary"] = (
                dict(
                    arm=arm,
                    completed_event=event - 1,
                    path=result["predecessor_state"]["path"],
                )
                if event > 1
                else None
            )
            result["costs"].update(
                persistence_s=result["state"]["persistence_s"],
                export_s=result.get("publication", {}).get("export_s", 0.0),
            )
        result["status"] = "completed"
    except BaseException as error:
        result.update(
            status="failed", error=repr(error), traceback=traceback.format_exc()
        )
        raise
    finally:
        if trainer is not None:
            if result["status"] != "completed":
                trainer.failed = True
            result["peak_allocated_bytes"] = __import__(
                "torch"
            ).cuda.max_memory_allocated()
            result["peak_reserved_bytes"] = __import__(
                "torch"
            ).cuda.max_memory_reserved()
            trainer.close()
        result["work"] = read_work(output / "WORK.jsonl", limits)
        result["wall_s"] = time.perf_counter() - started
        write_json(output / "RESULT.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("event", type=int)
    parser.add_argument("arm")
    args = parser.parse_args()
    run(args.root, args.event, args.arm)


if __name__ == "__main__":
    main()
