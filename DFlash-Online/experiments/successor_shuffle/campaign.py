"""Small fail-closed gates shared by the native workers and root controller."""

import math
import hashlib
import json
import struct
from pathlib import Path

from experiments.kplus_abt.campaign import schedule_bindings
from experiments.ok_coding.storage import identity
from experiments.survival_head.data import read, verify_shared_packets
from online_feedback.storage import write_json
from .contract import ARMS, SCHEMA, TRUE_EXPORT_SHA256, expected_work, training_arms


def validate_manipulation(row, *, identity_assignment=False):
    flags = (
        "class_counts_exact",
        "mass_preserved",
        "selection_preserved",
        "denominator_preserved",
    )
    if (
        row.get("schema") != SCHEMA
        or row.get("status") != "passed"
        or row.get("seed") != 20261111
        or row.get("identity") is not identity_assignment
        or not row.get("response_id")
        or any(row.get(key) is not True for key in flags)
    ):
        raise ValueError("Successor assignment identity or invariants failed")
    for key in (
        "eligible_errors",
        "ineligible_errors",
        "mixed_strata",
        "flipped_bits",
        "projection_calls",
    ):
        if type(row.get(key)) is not int or row[key] < 0:
            raise ValueError("Malformed manipulation count: " + key)
    if not 1 <= row["projection_calls"] <= 64:
        raise ValueError("Extra detached projection budget exceeded")
    if row.get("projection_completed") != row["projection_calls"]:
        raise ValueError("Incomplete detached projection work")
    for key in (
        "uniform_expected_flips",
        "reassigned_weight",
        "coefficient_l1_weight",
        "projection_s",
        "assignment_s",
        "selected_weight",
        "denominator",
        "true_loss",
        "assigned_loss",
    ):
        if (
            isinstance(row.get(key), bool)
            or not isinstance(row.get(key), (int, float))
            or not math.isfinite(row[key])
        ):
            raise ValueError("Nonfinite manipulation scalar: " + key)
    for key in (
        "uniform_expected_flips",
        "reassigned_weight",
        "coefficient_l1_weight",
        "projection_s",
        "assignment_s",
        "selected_weight",
        "denominator",
    ):
        if row[key] < 0:
            raise ValueError("Negative manipulation quantity: " + key)
    strata = row["strata"]
    if len({s["position"] for s in strata}) != len(strata):
        raise ValueError("Duplicated first-error stratum")
    expected = 0.0
    for s in strata:
        n, k = s["n"], s["k"]
        if (
            type(n) is not int
            or type(k) is not int
            or n <= 0
            or not 0 <= k <= n
            or s["true_high_count"] != k
            or s["assigned_high_count"] != k
            or s["recipients"] != sorted(s["recipients"])
            or len(set(s["recipients"])) != n
            or sorted(s["donors"]) != s["recipients"]
            or not 0 <= s["flipped_bits"] <= n
            or s["flipped_bits"] % 2
        ):
            raise ValueError("Per-position class or donor/recipient membership changed")
        benchmark = 2.0 * k * (n - k) / n
        if s["uniform_expected_flips"] != benchmark:
            raise ValueError("Uniform permutation benchmark differs")
        expected += benchmark
        donors = s["recipients"]
        if not identity_assignment and n > 1:

            def donor_key(offset):
                payload = json.dumps(
                    [20261111, row["response_id"], s["position"], offset],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
                return hashlib.sha256(payload).digest(), offset

            donors = sorted(s["recipients"], key=donor_key)
        if s["donors"] != donors:
            raise ValueError("Donors differ from the pinned canonical hash order")
    if (
        sum(s["n"] for s in strata) != row["eligible_errors"]
        or sum(0 < s["k"] < s["n"] for s in strata) != row["mixed_strata"]
        or sum(s["flipped_bits"] for s in strata) != row["flipped_bits"]
        or not math.isclose(
            expected, row["uniform_expected_flips"], rel_tol=1e-12, abs_tol=1e-12
        )
    ):
        raise ValueError("Response and stratum totals disagree")
    for name in ("W1", "W2", "coefficient1", "coefficient2", "error_mass"):
        if not math.isclose(
            row["true_" + name], row["assigned_" + name], rel_tol=2e-6, abs_tol=2e-7
        ):
            raise ValueError("Native FP32 class mass/coefficient mismatch: " + name)
    if identity_assignment and (
        row["flipped_bits"]
        or row["reassigned_weight"]
        or row["true_loss"] != row["assigned_loss"]
    ):
        raise ValueError("Identity assignment changed native TRUE loss or labels")
    return dict(
        status="passed",
        response_id=row["response_id"],
        flipped_bits=row["flipped_bits"],
    )


def validate_observation(observation):
    row = observation["successor_shuffle"]
    sums = observation["sum_metrics"]
    validate_manipulation(row)
    numeric = (
        "eligible_errors",
        "ineligible_errors",
        "mixed_strata",
        "flipped_bits",
        "uniform_expected_flips",
        "reassigned_weight",
        "coefficient_l1_weight",
        "projection_calls",
        "projection_s",
        "assignment_s",
        "true_loss",
        "assigned_loss",
    )
    # Output metrics are explicitly FP32 tensors; receipt wall clocks are Python doubles.
    f32 = lambda value: struct.unpack("f", struct.pack("f", value))[0]
    if sums.get("shuffle_response_count") != 1:
        raise ValueError("Duplicate or absent shuffled response metric")
    if any(sums.get("shuffle_" + key) != f32(row[key]) for key in numeric):
        raise ValueError(
            "Assignment receipt disagrees with same-forward scalar metrics"
        )
    for name in ("W1", "W2", "coefficient1", "coefficient2"):
        if sums["kplus_" + name] != row["assigned_" + name]:
            raise ValueError("K+ assigned class mass/coefficient observation disagrees")
    if (
        sums["kplus_S"] != row["selected_weight"]
        or sums["kplus_first_error_weight_after"] != row["assigned_error_mass"]
        or observation["loss"] != row["assigned_loss"]
    ):
        raise ValueError("Assigned loss/denominator/mass differs from actual forward")
    return dict(status="passed", same_forward_metrics=True)


def shared_inputs(plan):
    for key in ("pool", "original_schedule", "baseline_result"):
        if identity(plan[key]) != plan[key + "_identity"]:
            raise ValueError("Sealed original training metadata changed: " + key)
    pool = read(plan["pool"])
    verify_shared_packets(pool)
    baseline, original = read(plan["original_schedule"]), read(plan["baseline_result"])
    if schedule_bindings(baseline) != schedule_bindings(original):
        raise ValueError("Prepared schedule differs from original TRUE event")
    if [r["record_id"] for r in pool["records"]] != [
        r["record_id"] for r in baseline["features"]
    ]:
        raise ValueError("Shared packet order differs from original schedule")
    if [r["feature"]["path"] for r in pool["records"]] != [
        r["path"] for r in baseline["features"]
    ]:
        raise ValueError("Schedule does not use the authenticated PR37 teacher packets")
    return pool, baseline


def authenticate_true(plan):
    item = plan["true_reuse"]
    if not item["enabled"]:
        if plan["budget"]["optimizer_steps"] != 384:
            raise ValueError("Fallback TRUE training requires sealed384-step budget")
        return dict(status="not_reused")
    receipt = read(item["authentication_path"])
    if (
        receipt.get("status") != "passed"
        or receipt.get("binding") != plan["binding"]
        or receipt.get("execution_commit") != plan["execution_source"]["commit"]
        or any(
            receipt.get(k) is not True
            for k in (
                "origin",
                "packets",
                "schedule",
                "optimizer",
                "runtime",
                "step640",
                "export",
            )
        )
        or identity(item["result_path"]) != item["result_identity"]
        or identity(Path(item["checkpoint_path"]) / "model.safetensors")
        != item["weights_identity"]
        or item["weights_identity"]["sha256"] != TRUE_EXPORT_SHA256
    ):
        raise ValueError("TRUE export cannot be authenticated for reuse")
    result = read(item["result_path"])
    if (
        result["status"] != "completed"
        or result["method"] != "KPLUS"
        or result["start_step"] != 512
        or result["end_step"] != 640
        or result["weights_identity"] != item["weights_identity"]
        or result["boundary"]["status"] != "passed"
        or result["boundary"]["export_exact"] is not True
    ):
        raise ValueError("Retained TRUE result has no exact step640 export boundary")
    return receipt


def require_admission(plan):
    from .qualification import summarize
    from .runtime import capacity

    root = Path(plan["root"])
    if read(root / "sealed/PLAN.json") != plan or plan["automatic_retry"] is not False:
        raise ValueError("Plan changed or automatic retry enabled")
    if read(root / "PREFLIGHT.json").get("status") != "passed":
        raise ValueError("Require current completed preflight before native training")
    seal = read(root / "sealed/SEAL.json")
    if seal["binding"] != plan["binding"] or any(
        identity(path) != expected for path, expected in seal["files"].items()
    ):
        raise ValueError("Sealed source/input metadata changed")
    count = len(training_arms(plan["true_reuse"]["enabled"]))
    if (
        plan["budget"]["optimizer_steps"] != 128 * count
        or plan["budget"]["training_presentations"] != 256 * count
    ):
        raise ValueError("Training budget differs from declared reuse decision")
    authenticate_true(plan)
    admission = read(root / "qualification/ADMISSION.json")
    if admission != summarize(root) or admission.get("status") != "passed":
        raise ValueError("Native qualification admission differs from raw evidence")
    capacity(plan)
    return admission


def validate_update(result, arm, baseline):
    expected = dict(
        status="completed",
        method=arm,
        dataset="gsm8k",
        event_id="event_005",
        start_step=512,
        end_step=640,
        optimizer_steps=128,
        response_presentations=256,
        passes=2,
    )
    if arm not in ARMS or any(result.get(k) != v for k, v in expected.items()):
        raise ValueError("One-event native accounting differs")
    if (
        schedule_bindings(result) != schedule_bindings(baseline)
        or result["features"] != baseline["features"]
    ):
        raise ValueError(
            "Original response order, packet bindings or native anchor draws changed"
        )
    if result.get("successor_work") != expected_work(arm):
        raise ValueError("Unexpected main forward/backward/update work")
    if arm == "SHUFFLED":
        for row in result["records"]:
            manipulation = row["response_observation"]["successor_shuffle"]
            if manipulation["response_id"] != row["record_id"]:
                raise ValueError("Assignment receipt refers to another response")
            validate_observation(row["response_observation"])
    return dict(status="passed", original_schedule_exact=True)


def manipulation_summary(plan, shuffled_result, true_weights_identity):
    _, baseline = shared_inputs(plan)
    validate_update(shuffled_result, "SHUFFLED", baseline)
    rows = [
        row["response_observation"]["successor_shuffle"]
        for row in shuffled_result["records"]
    ]
    names = (
        "eligible_errors",
        "ineligible_errors",
        "mixed_strata",
        "flipped_bits",
        "uniform_expected_flips",
        "reassigned_weight",
        "projection_calls",
        "projection_s",
        "assignment_s",
    )
    totals = {name: sum(row[name] for row in rows) for name in names}
    changed = shuffled_result["weights_identity"] != true_weights_identity
    result = dict(
        status="passed",
        presentations=len(rows),
        **totals,
        labels_changed=totals["flipped_bits"] > 0,
        final_export_changed=changed,
        all_invariants_passed=True,
        binding=plan["binding"],
        actuation="active" if totals["flipped_bits"] and changed else "no_actuation",
        limitation="One fixed deterministic permutation, not a uniform-permutation experiment; benchmark is descriptive.",
        rows=rows
    )
    write_json(Path(plan["root"]) / "manipulation/RESULT.json", result)
    return result
