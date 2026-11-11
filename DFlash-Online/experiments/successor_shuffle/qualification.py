"""Three fixed disposable native windows; no qualification state trains an arm."""

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import time
import traceback

from experiments.ok_coding.campaign import WorkLedger, read_work, require_complete_work
from experiments.ok_coding.storage import identity
from online_feedback.storage import write_json
from .campaign import read, shared_inputs, validate_manipulation
from .contract import QUALIFICATION_CASES


def case_limits(case):
    if case not in QUALIFICATION_CASES:
        raise ValueError("Unknown fixed qualification case")
    return dict(
        response_forwards=2,
        response_backwards=2,
        optimizer_steps=1,
        next_anchor_draws=1,
    )


def snapshot_check(snapshot, row):
    """Reconstruct the donor map and invariants from saved scalar masks."""
    validate_manipulation(row, identity_assignment=row["identity"])
    anchors = snapshot["anchors"][0]
    fields = ("selection", "errors", "eligible", "true_high", "assigned_high")
    masks = {name: snapshot[name][0] for name in fields}
    weights = snapshot["weights"][0]
    labels = snapshot["labels"][0]
    eos = set(snapshot["eos_token_ids"])
    if anchors != row["anchor_offsets"] or any(
        len(masks[k]) != len(anchors) for k in fields
    ):
        raise ValueError("Snapshot and receipt anchor membership differs")
    by_anchor = {anchor: index for index, anchor in enumerate(anchors)}
    for s in row["strata"]:
        p = s["position"]
        recipients = s["recipients"]
        expected = recipients
        if not row["identity"] and s["n"] > 1:

            def donor_key(offset):
                payload = json.dumps(
                    [20261111, row["response_id"], p, offset],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
                return hashlib.sha256(payload).digest(), offset

            expected = sorted(recipients, key=donor_key)
        if expected != s["donors"]:
            raise ValueError("Donor hash order differs from the sealed serialization")
        for recipient, donor in zip(recipients, expected):
            i, j = by_anchor[recipient], by_anchor[donor]
            if (
                not masks["eligible"][i][p]
                or not masks["eligible"][j][p]
                or masks["assigned_high"][i][p] != masks["true_high"][j][p]
                or weights[i][p] != weights[j][p]
            ):
                raise ValueError(
                    "Assigned label is not the declared same-position donor"
                )
    counts = dict(flipped_bits=0, eligible_errors=0, ineligible_errors=0)
    true_w1 = true_w2 = assigned_w1 = assigned_w2 = selected = 0.0
    for i in range(len(anchors)):
        error_positions = [p for p, value in enumerate(masks["errors"][i]) if value]
        if len(error_positions) > 1:
            raise ValueError("More than one selected first error per block")
        valid = [False] * len(weights[i])
        for p in range(1, len(valid)) if labels[i][0] not in eos else ():
            if weights[i][p] <= 0:
                break
            valid[p] = True
            if labels[i][p] in eos:
                break
        error_position = error_positions[0] if error_positions else len(valid)
        if masks["selection"][i] != [
            keep and p <= error_position for p, keep in enumerate(valid)
        ]:
            raise ValueError("Native selected prefix/first-error mask changed")
        for p, w in enumerate(weights[i]):
            selection, error, eligible, truth, assigned = [
                masks[k][i][p] for k in fields
            ]
            if eligible != (error and p + 1 < len(valid) and valid[p + 1]):
                raise ValueError(
                    "Successor eligibility changed across gap or canonical EOS"
                )
            if (
                (error and not selection)
                or (eligible and not error)
                or (truth and not eligible)
                or (assigned and not eligible)
            ):
                raise ValueError(
                    "Assignment escaped native first-error/valid-successor scope"
                )
            counts["flipped_bits"] += truth != assigned
            counts["eligible_errors"] += eligible
            counts["ineligible_errors"] += error and not eligible
            selected += w * selection
            true_w1 += w * (error and not truth)
            true_w2 += w * truth
            assigned_w1 += w * (error and not assigned)
            assigned_w2 += w * assigned
    if any(counts[k] != row[k] for k in counts):
        raise ValueError("Saved masks and manipulation counts differ")
    for key, value in dict(
        true_W1=true_w1,
        true_W2=true_w2,
        assigned_W1=assigned_w1,
        assigned_W2=assigned_w2,
        selected_weight=selected,
    ).items():
        if not math.isclose(value, row[key], rel_tol=2e-6, abs_tol=2e-7):
            raise ValueError(
                "Saved-mask mass differs from native FP32 reduction: " + key
            )
    return dict(
        status="passed", independent_donor_map=True, mask_mass_checked=True, **counts
    )


def next_anchor_draw(trainer, raw):
    """One sampler-only diagnostic; preserve the live training RNG exactly."""
    from online_feedback.rng import preserve_rng

    batch = trainer._batch(raw)
    device = next(trainer.bundle.draft_model.parameters()).device
    mask = batch.tensors["loss_mask"].to(device)
    with preserve_rng():
        anchors, keep = trainer._original_sampler(
            batch.tensors["input_ids"].shape[-1], mask, device
        )
    return dict(
        record_id=raw["record_id"],
        positions=anchors.cpu().tolist(),
        keep_mask=keep.cpu().tolist(),
    )


def run_case(root, case):
    import torch
    from specforge.training.strategies.base import StepContext
    from experiments.kplus_abt.qualification import endpoint, finite_endpoint
    from experiments.functional_step_geometry.native_pilot import discarded_startup
    from experiments.functional_step_geometry.qualify import cleanup
    from experiments.prefix_guard.qualify import (
        pin_retained_graph_runtime,
        persist_comparison,
        compare_states,
    )
    from online_feedback.storage import save_torch
    from .training import SuccessorShuffleTrainer

    root = Path(root)
    plan = read(root / "sealed/PLAN.json")
    pool, _ = shared_inputs(plan)
    if read(root / "PREFLIGHT.json").get("status") != "passed":
        raise ValueError(
            "Fresh source/input/storage preflight required before native qualification"
        )
    output = root / "qualification" / case
    output.mkdir(parents=True, exist_ok=False)
    limits = case_limits(case)
    ledger = WorkLedger(output / "WORK.jsonl", limits)
    selected = pool["records"][:3]
    features = [Path(row["feature"]["path"]) for row in selected]
    receipt = dict(
        status="running",
        case=case,
        binding=plan["binding"],
        execution_commit=plan["execution_source"]["commit"],
        automatic_retry=False,
        inputs=dict(pool_identity=identity(plan["pool"]), packets=selected),
        manipulations=[],
    )
    trainer = None
    failure = None
    started = time.perf_counter()
    gradients = {}
    clipped = {}
    snapshots = []
    try:
        pin_retained_graph_runtime(output)
        arm = "TRUE" if case == "TRUE" else "SHUFFLED"
        trainer = SuccessorShuffleTrainer(
            plan["configs"][arm],
            output / "trainer",
            arm=arm,
            binding=plan["binding"],
            original_config_text=Path(plan["original_config"]).read_text(),
            origin_state=plan["origin_state"],
            origin_export=plan["origin_export"],
        )
        receipt["complete_state"] = read(trainer.output_root / "RESUME_CHECK.json")
        receipt["startup"] = discarded_startup(
            trainer, features[:2], output / "startup"
        )
        if case != "TRUE":
            trainer.install_shuffle(identity_assignment=case == "IDENTITY")
        trainer._active_event = "qualification_" + case
        (trainer.events_root / trainer._active_event).mkdir()
        trainer.bundle.model.begin_event()
        ledger.wrap(trainer.core.strategy, "forward_loss", "response_forwards")
        ledger.wrap(trainer.backend, "backward", "response_backwards")
        ordinary_step = trainer.backend.optimizer.step

        def observe_unclipped():
            for name, p in trainer.bundle.draft_model.named_parameters():
                gradients[name] = (
                    None if p.grad is None else p.grad.detach().cpu().clone()
                )
            return ordinary_step()

        trainer.backend.optimizer.step = observe_unclipped
        adam_step = trainer.backend.optimizer.optimizer.step

        def observe_clipped(*args, **kwargs):
            for name, p in trainer.bundle.draft_model.named_parameters():
                clipped[name] = (
                    None if p.grad is None else p.grad.detach().cpu().clone()
                )
            return ledger.call("optimizer_steps", adam_step, *args, **kwargs)

        trainer.backend.optimizer.optimizer.step = observe_clipped
        torch.cuda.reset_peak_memory_stats()
        for feature in features[:2]:
            raw = torch.load(feature, map_location="cpu", weights_only=True)
            trainer._anchor_observations.clear()
            result = trainer.core.train_step(
                trainer._batch(raw),
                StepContext(
                    global_step=trainer.global_step,
                    total_steps=trainer.horizon,
                    collect_detailed_metrics=True,
                ),
            )
            result.materialize_metrics()
            trainer.presentations += 1
            trainer.global_step += int(result.optimizer_stepped)
            if case != "TRUE":
                row = deepcopy(trainer.bundle.model.successor_shuffle.last_receipt)
                snapshot = (
                    trainer.bundle.model.successor_shuffle.qualification_snapshot()
                )
                snapshot["eos_token_ids"] = list(trainer.bundle.model.eos_token_ids)
                receipt["manipulations"].append(
                    dict(receipt=row, check=snapshot_check(snapshot, row))
                )
                snapshots.append(snapshot)
                write_json(output / "SNAPSHOTS.json", snapshots)
                write_json(output / "MANIPULATION.json", receipt["manipulations"])
        saved = endpoint(trainer, gradients)
        saved["clipped_gradients"] = clipped
        raw = torch.load(features[2], map_location="cpu", weights_only=True)
        saved["next_anchor_draw"] = ledger.call(
            "next_anchor_draws", next_anchor_draw, trainer, raw
        )
        save_torch(output / "ENDPOINT.pt", saved)
        compact_fields = (
            "rng",
            "anchors",
            "global_step",
            "presentations",
            "micro",
            "remainder",
            "response_losses",
            "next_anchor_draw",
        )
        save_torch(
            output / "COMPACT_ENDPOINT.pt", {k: saved[k] for k in compact_fields}
        )
        receipt.update(
            endpoint_identity=identity(output / "ENDPOINT.pt"),
            compact_endpoint_identity=identity(output / "COMPACT_ENDPOINT.pt"),
            observations=deepcopy(trainer.response_records),
            windows=deepcopy(trainer.window_records),
            physical_work=deepcopy(trainer.work),
            finite=finite_endpoint(saved),
        )
        require_complete_work(read_work(ledger.path, limits), limits)
        if (
            trainer.global_step != 513
            or trainer.presentations != 1026
            or trainer.core.accumulation_remainder
            or len(trainer.response_records) != 2
            or len(trainer.window_records) != 1
        ):
            raise ValueError("Disposable two-response window work differs")
        if case != "TRUE":
            reference = torch.load(
                root / "qualification/TRUE/ENDPOINT.pt",
                map_location="cpu",
                weights_only=False,
                mmap=True,
            )
            fields = (
                tuple(saved)
                if case == "IDENTITY"
                else tuple(k for k in compact_fields if k != "response_losses")
            )
            receipt["comparison"] = persist_comparison(
                output / "EXACT.json",
                {k: saved[k] for k in fields},
                {k: reference[k] for k in fields},
            )
            receipt["native_difference"] = compare_states(saved, reference)
            write_json(output / "NATIVE_DIFFERENCE.json", receipt["native_difference"])
            if [
                r["receipt"]["true_loss"] for r in receipt["manipulations"]
            ] != reference["response_losses"]:
                raise ValueError(
                    "Unpermuted losses do not match native TRUE on the same two-response state"
                )
            if case == "SHUFFLED":
                neutral = read(root / "qualification/IDENTITY/SNAPSHOTS.json")
                keep = (
                    "anchors",
                    "labels",
                    "weights",
                    "selection",
                    "errors",
                    "eligible",
                    "true_high",
                    "eos_token_ids",
                )
                receipt["same_state_inputs"] = persist_comparison(
                    output / "SAME_STATE_INPUTS.json",
                    [{k: s[k] for k in keep} for s in snapshots],
                    [{k: s[k] for k in keep} for s in neutral],
                )
                if (
                    sum(r["receipt"]["flipped_bits"] for r in receipt["manipulations"])
                    <= 0
                    or sum(
                        r["receipt"]["reassigned_weight"]
                        for r in receipt["manipulations"]
                    )
                    <= 0
                ):
                    raise ValueError(
                        "Fixed native mixed-window qualification did not actuate; no extra windows"
                    )
            receipt["snapshots_identity"] = identity(output / "SNAPSHOTS.json")
            receipt["manipulation_identity"] = identity(output / "MANIPULATION.json")
        receipt["status"] = "completed"
    except BaseException as error:
        failure = error
        receipt.update(
            status="failed", error=repr(error), traceback=traceback.format_exc()
        )
        if trainer is not None:
            trainer.failed = True
        raise
    finally:
        receipt.update(
            wall_s=time.perf_counter() - started,
            work=read_work(ledger.path, limits),
            peak_allocated_bytes=(
                torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0
            ),
            peak_reserved_bytes=(
                torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0
            ),
        )
        cleanup(
            lambda: write_json(output / "RESULT.json", receipt),
            failure,
            receipt,
            "qualification receipt",
        )
        if trainer is not None:
            cleanup(trainer.close, failure, receipt, "disposable trainer close")
    return receipt


def summarize(root):
    root = Path(root)
    plan = read(root / "sealed/PLAN.json")
    checks = {}
    results = {}
    for case in QUALIFICATION_CASES:
        folder = root / "qualification" / case
        row = read(folder / "RESULT.json")
        require_complete_work(row["work"], case_limits(case))
        require_complete_work(
            row["startup"]["work"],
            dict(response_forwards=2, response_backwards=2, optimizer_steps=0),
        )
        if (
            row["status"] != "completed"
            or row["binding"] != plan["binding"]
            or row["execution_commit"] != plan["execution_source"]["commit"]
            or row["complete_state"]["status"] != "passed"
            or row["finite"]["status"] != "passed"
            or read(folder / "startup/EXACT_RESTORATION.json").get("exact_equal")
            is not True
            or identity(folder / "ENDPOINT.pt") != row["endpoint_identity"]
            or identity(folder / "COMPACT_ENDPOINT.pt")
            != row["compact_endpoint_identity"]
            or row["inputs"]["pool_identity"] != identity(plan["pool"])
            or row["inputs"]["packets"] != read(plan["pool"])["records"][:3]
        ):
            raise ValueError(
                "Qualification source, work, input or endpoint identity failed: " + case
            )
        if case != "TRUE":
            if (
                row["comparison"].get("exact_equal") is not True
                or read(folder / "EXACT.json") != row["comparison"]
                or identity(folder / "SNAPSHOTS.json") != row["snapshots_identity"]
                or identity(folder / "MANIPULATION.json")
                != row["manipulation_identity"]
                or read(folder / "MANIPULATION.json") != row["manipulations"]
                or len(row["manipulations"]) != 2
            ):
                raise ValueError(
                    "Exact identity/sampling gate or manipulation evidence failed"
                )
            for snap, part in zip(
                read(folder / "SNAPSHOTS.json"), row["manipulations"]
            ):
                validate_manipulation(
                    part["receipt"], identity_assignment=case == "IDENTITY"
                )
                if snapshot_check(snap, part["receipt"]) != part["check"]:
                    raise ValueError("Manipulation masks no longer reproduce")
        if case == "SHUFFLED":
            if (
                read(folder / "SAME_STATE_INPUTS.json") != row["same_state_inputs"]
                or row["same_state_inputs"]["exact_equal"] is not True
                or sum(r["receipt"]["flipped_bits"] for r in row["manipulations"]) <= 0
            ):
                raise ValueError(
                    "Real permutation same-state manipulation qualification failed"
                )
        checks[case] = True
        results[case] = identity(folder / "RESULT.json")
    return dict(
        status="passed",
        binding=plan["binding"],
        execution_commit=plan["execution_source"]["commit"],
        checks=checks,
        result_identities=results,
        optimizer_steps=3,
        response_forwards=12,
        response_backwards=12,
        next_anchor_draws=3,
        qualification_states_not_used_for_training=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("--case", choices=QUALIFICATION_CASES, required=True)
    args = parser.parse_args()
    run_case(args.root, args.case)
