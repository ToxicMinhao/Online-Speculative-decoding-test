"""Read-only final report from private raw pilot evidence; never starts model work."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile

from . import analysis
from .evaluation import audit_stage, label_for, METHODS

PREFIX = "second_follower_"
ATOL, RTOL = 2e-7, 2e-6
DEPTH_FIELDS = (
    "short_count", "long_count", "censored_count", "short_weight", "long_weight",
    "censored_weight", "short_numerator", "long_numerator", "native_short_weight",
    "native_long_weight", "native_censored_weight", "rule_transfer_weight",
    "transferred_weight", "short_weight_after", "long_weight_after",
    "short_relative_scale", "long_relative_scale", "short_coefficient",
    "long_coefficient", "mixed", "active", "mass_balance_error",
)


def file_identity(path):
    raw = Path(path).read_bytes()
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


class Evidence:
    """Resolve a retrieved run without changing recorded scientific identities."""

    def __init__(self, root, original_root, path_maps=()):
        self.root, self.original_root = Path(root).resolve(), Path(original_root)
        self.path_maps = [(self.original_root, self.root)] + [
            (Path(original), Path(local).resolve()) for original, local in path_maps
        ]
        self.files, self.missing, self.errors = {}, {}, []

    def local(self, path):
        path = Path(path)
        if path.is_absolute():
            for original, local in sorted(self.path_maps, key=lambda pair: len(pair[0].parts), reverse=True):
                try:
                    return local / path.relative_to(original)
                except ValueError:
                    continue
            return path
        return self.root / path

    def read(self, path, *, lines=False, optional=False):
        path = self.local(path)
        if not path.is_file():
            if not optional:
                self.missing[str(path)] = "Not retained; never imputed as successful or zero work."
            return None
        raw = path.read_bytes()
        observed = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        previous = self.files.setdefault(str(path), observed)
        if previous != observed:
            self.errors.append("Changed between reads: " + str(path))
        try:
            return [json.loads(row) for row in raw.splitlines() if row.strip()] if lines else json.loads(raw)
        except (TypeError, ValueError) as error:
            self.errors.append(str(path) + ": " + repr(error))
            return None

    def remember_tree(self, path):
        for child in sorted(self.local(path).rglob("*")):
            if child.is_file() and child.suffix in (".json", ".jsonl", ".csv"):
                observed = file_identity(child)
                if self.files.setdefault(str(child), observed) != observed:
                    self.errors.append("Changed during audit: " + str(child))

    def stable(self):
        return all(Path(path).is_file() and file_identity(path) == value for path, value in self.files.items())


def dump_json(path, value):
    def safe(item):
        if isinstance(item, float) and not math.isfinite(item):
            return {"nonfinite_saved_value": repr(item)}
        if isinstance(item, dict):
            return {key: safe(value) for key, value in item.items()}
        if isinstance(item, (list, tuple)):
            return [safe(value) for value in item]
        return item
    Path(path).write_text(json.dumps(safe(value), indent=2, allow_nan=False) + "\n")


def dump_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or ["status"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def exposure_audit(result):
    """Independent Python scalar recomputation; no calls to the live reducer.

    This verifies saved sufficient-statistic arithmetic, not unsaved per-anchor
    proposal membership. FP32 association/cancellation uses the native tolerance.
    """
    errors, response_rows, depth_rows = [], [], []
    checks = 0

    def require(condition, label):
        nonlocal checks
        checks += 1
        if not condition:
            errors.append(label)

    def close(actual, expected, label, scale=None):
        bound = ATOL + RTOL * max(abs(expected), abs(actual), abs(scale or 0.))
        require(math.isfinite(actual) and math.isfinite(expected) and abs(actual - expected) <= bound, label)

    if not isinstance(result, dict):
        return dict(status="unavailable", errors=["No candidate event result"], model_work=0), [], []
    records = result.get("records", [])
    require(result.get("status") == "completed", "event_not_completed")
    require(result.get("start_step") == 512 and result.get("end_step") == 640, "event_step_boundary")
    require(len(records) == 256, "require_256_presentations")
    for ordinal, record in enumerate(records):
        label = f"presentation_{ordinal}"
        try:
            observation = record["response_observation"]
            metrics = observation["sum_metrics"]
            require(observation["record_id"] == record["record_id"] and observation["pass_index"] == record["pass"] == ordinal // 128, label + ":identity")
            require(record["optimizer_stepped"] is (ordinal % 2 == 1), label + ":accumulation")
            require(record["optimizer_step_after"] == 512 + (ordinal + 1) // 2, label + ":step")
            require(all(type(value) in (int, float) and math.isfinite(value) for value in metrics.values()), label + ":finite_metrics")
            scalar = {name[len(PREFIX):]: float(value) for name, value in metrics.items() if name.startswith(PREFIX) and not name.startswith(PREFIX + "depth_")}
            require(scalar["disabled"] == 0 and scalar["response_count"] == 1, label + ":enabled_single_response")
            w1, w2, selected = (float(metrics[name]) for name in ("kplus_W1", "kplus_W2", "kplus_S"))
            require(min(w1, w2, selected) >= 0, label + ":nonnegative_weights")
            mixed_native = w1 > 0 and w2 > 0
            denominator = w1 + 2 * w2
            c1 = 1 - .5 * w2 / denominator if mixed_native else 1.
            c2 = 1 + .5 * w1 / denominator if mixed_native else 1.
            close(float(metrics["kplus_coefficient1"]), c1, label + ":native_low_coefficient")
            close(float(metrics["kplus_coefficient2"]), c2, label + ":native_high_coefficient")
            # Saved c is the actual FP32 coefficient used by this response.
            c = float(metrics["kplus_coefficient2"])
            own_depths, correction = [], 0.
            for depth in range(16):
                d = {name: float(metrics[f"{PREFIX}depth_{depth}_{name}"]) for name in DEPTH_FIELDS}
                tag = label + f":depth_{depth}"
                for kind in ("short", "long", "censored"):
                    require(d[kind + "_count"] >= 0 and d[kind + "_count"].is_integer(), tag + ":count")
                    require(d[kind + "_weight"] >= 0, tag + ":weight")
                    close(d["native_" + kind + "_weight"], c * d[kind + "_weight"], tag + ":native_mass")
                if depth in (0, 15):
                    require(sum(d[k + "count"] for k in ("short_", "long_", "censored_")) == 0, tag + ":seed_or_no_follower")
                if depth == 14:
                    require(d["short_count"] == d["long_count"] == 0, tag + ":unavailable_second_slot")
                bs, bl = d["short_weight"], d["long_weight"]
                short, long = c * bs, c * bl
                mixed = bs > 0 and bl > 0
                transfer = min(short / 8, long / 8, (c - 1) * bs) if mixed and c > 1 else 0.
                tol = ATOL + RTOL * (short + long)
                for key in ("rule_transfer_weight", "transferred_weight"):
                    close(d[key], transfer, tag + ":" + key, short + long)
                close(d["short_weight_after"], short - transfer, tag + ":donor_mass")
                close(d["long_weight_after"], long + transfer, tag + ":recipient_mass")
                require(-tol <= d["transferred_weight"] <= min(short / 8, long / 8) + tol, tag + ":relative_cap")
                require(d["short_weight_after"] >= bs - tol, tag + ":donor_floor")
                close(d["short_weight_after"] + d["long_weight_after"], short + long, tag + ":mass_balance")
                close(d["mass_balance_error"], 0., tag + ":saved_mass_error", short + long)
                short_scale = 1 - transfer / short if transfer else 1.
                long_scale = 1 + transfer / long if transfer else 1.
                for key, expected in (("short_relative_scale", short_scale), ("long_relative_scale", long_scale), ("short_coefficient", c * short_scale), ("long_coefficient", c * long_scale)):
                    close(d[key], expected, tag + ":" + key)
                require(d["mixed"] == int(mixed) and d["active"] == int(transfer > 0), tag + ":stratum_flags")
                if transfer:
                    correction += transfer * (d["long_numerator"] / bl - d["short_numerator"] / bs)
                own_depths.append(d)
                depth_rows.append(dict(presentation_ordinal=ordinal, record_id=record["record_id"], pass_index=record["pass"], depth=depth, native_high_coefficient=c, recomputed_transfer=transfer, **d))
            for key in ("short_count", "long_count", "censored_count", "short_weight", "long_weight", "censored_weight", "native_short_weight", "native_long_weight", "native_censored_weight", "transferred_weight", "rule_transfer_weight"):
                close(scalar[key], math.fsum(d[key] for d in own_depths), label + ":sum_" + key)
            close(scalar["short_weight"] + scalar["long_weight"] + scalar["censored_weight"], w2, label + ":high_weight_partition")
            close(scalar["short_count"] + scalar["long_count"] + scalar["censored_count"], float(metrics["kplus_priority2_error_count"]), label + ":high_count_partition")
            require(scalar["active_response"] == int(any(d["active"] for d in own_depths)), label + ":active_response")
            close(scalar["active_strata"], sum(d["active"] for d in own_depths), label + ":active_strata")
            close(scalar["mixed_strata"], sum(d["mixed"] for d in own_depths), label + ":mixed_strata")
            close(scalar["selected_weight"], selected, label + ":selection")
            close(scalar["selected_denominator"], selected + 1e-6, label + ":epsilon_once")
            close(float(metrics["selected_position_weight"]), selected, label + ":native_selection")
            close(float(metrics["loss_denominator"]), scalar["selected_denominator"], label + ":actual_denominator")
            close(scalar["native_high_weight"], c * w2, label + ":native_high")
            close(scalar["native_low_weight"], c1 * w1, label + ":native_low")
            close(scalar["prefix_weight"], selected - w1 - w2, label + ":prefix")
            for field in ("prefix_weight_change", "low_weight_change", "selected_weight_change"):
                require(scalar[field] == 0, label + ":" + field)
            close(scalar["mass_balance_error"], 0., label + ":mass_error", selected)
            nk, n1, n2 = (float(metrics[name]) for name in ("kplus_N_K", "kplus_N1", "kplus_N2"))
            native = nk + .5 * (w1 * n2 - w2 * n1) / denominator if mixed_native else nk
            close(scalar["native_numerator"], native, label + ":native_numerator", nk)
            close(scalar["native_numerator"], float(metrics["kplus_N_new"]), label + ":native_metric")
            close(scalar["candidate_numerator"], native + correction, label + ":candidate_numerator", native)
            close(float(metrics["main_loss_numerator"]), scalar["candidate_numerator"], label + ":actual_numerator")
            close(scalar["numerator_correction"], correction, label + ":correction", native)
            close(float(observation["loss"]), scalar["candidate_numerator"] / (selected + 1e-6), label + ":actual_loss")
            close(scalar["moved_share_selected"], scalar["transferred_weight"] / (selected + 1e-6), label + ":selected_rate")
            high = scalar["native_high_weight"]
            close(scalar["moved_share_native_high"], scalar["transferred_weight"] / high if high else 0., label + ":high_rate")
            response_rows.append(dict(presentation_ordinal=ordinal, record_id=record["record_id"], pass_index=record["pass"], window=ordinal // 2, optimizer_stepped=record["optimizer_stepped"], optimizer_step_after=record["optimizer_step_after"], actual_loss=observation["loss"], native_low_coefficient=c1, native_high_coefficient=c, **scalar))
        except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError) as error:
            errors.append(label + ":malformed:" + repr(error))
    if len(records) == 256:
        first = [row["record_id"] for row in records[:128]]
        second = [row["record_id"] for row in records[128:]]
        require(len(set(first)) == 128 and sorted(first) == sorted(second), "two_passes_of_128_original_responses")
    totals = {key: math.fsum(row[key] for row in response_rows) for key in (
        "short_count", "long_count", "censored_count", "short_weight", "long_weight", "censored_weight",
        "native_short_weight", "native_long_weight", "native_censored_weight", "transferred_weight",
        "selected_weight", "native_high_weight", "mixed_strata", "active_strata",
    )} if response_rows else {}
    active_steps = len({row["window"] for row in response_rows if row["transferred_weight"] > 0})
    by_depth = []
    for depth in range(16):
        own = [row for row in depth_rows if row["depth"] == depth]
        by_depth.append(dict(depth=depth, **{key: math.fsum(row[key] for row in own) for key in (
            "short_count", "long_count", "censored_count", "short_weight", "long_weight", "censored_weight",
            "native_short_weight", "native_long_weight", "transferred_weight", "mixed", "active",
        )}))
    moved = totals.get("transferred_weight", 0.)
    total_selected, total_high = totals.get("selected_weight", 0.), totals.get("native_high_weight", 0.)
    summary = dict(
        status="passed" if not errors else "invalid_or_incomplete", errors=errors, checks=checks,
        presentations_observed=len(response_rows), planned_presentations=256,
        optimizer_steps_observed=sum(r.get("optimizer_stepped") is True for r in records), planned_optimizer_steps=128,
        active_responses=sum(row["transferred_weight"] > 0 for row in response_rows), active_steps=active_steps,
        active_response_fraction=sum(row["transferred_weight"] > 0 for row in response_rows) / 256,
        active_step_fraction=active_steps / 128, totals=totals, by_depth=by_depth,
        transferred_over_selected=moved / total_selected if total_selected else None,
        transferred_over_native_high=moved / total_high if total_high else None,
        mean_response_transferred_over_selected=math.fsum(row["transferred_weight"] / row["selected_denominator"] for row in response_rows) / len(response_rows) if response_rows else None,
        largest_response_transfer_share=max((row["transferred_weight"] for row in response_rows), default=0.) / moved if moved else None,
        largest_depth_transfer_share=max((row["transferred_weight"] for row in by_depth), default=0.) / moved if moved else None,
        tolerance=dict(atol=ATOL, rtol=RTOL, cancellation_scale="native mass/numerator"), model_work=0,
        scope="Independent scalar audit of saved response-wide sufficient statistics; unsaved proposal membership/EOS/gap classification is not independently reconstructed.",
    )
    return summary, response_rows, depth_rows


def endpoint(evidence, plan, checkpoints):
    panel = evidence.read(plan["panel_manifest"])
    refs = evidence.read(plan["references"])
    if not panel or not refs or not checkpoints or set(checkpoints) != set(METHODS):
        return dict(status="invalid_or_incomplete", inference_permitted=False, errors=["Missing sealed panel, references or both checkpoints"]), [], []
    folder = evidence.root / "evaluation"
    if not folder.is_dir():
        return dict(status="unattempted", inference_permitted=False, planned_generations=1024, unattempted_generations=1024), [], []
    measurements = evidence.read(folder / "MEASUREMENTS.jsonl", lines=True) or []
    evidence.remember_tree(folder)
    # The inherited raw reader supports an explicit relocation index. Construct
    # it privately and temporarily; do not rewrite raw IDs or monkeypatch tools.
    references = []
    for slot in plan["evaluation_schedule"]:
        path = folder / (label_for(slot) + ".json")
        raw = evidence.read(path)
        recorded = (raw or {}).get("metadata", {}).get("native_trace_path")
        if recorded:
            local = evidence.local(recorded)
            evidence.read(local)
            references.append(dict(recorded_native_path=recorded, local_relative_path=str(local)))
    with tempfile.TemporaryDirectory(prefix="second_follower_raw_audit_") as temporary:
        index = Path(temporary) / "audit_inputs/NATIVE_TRACE_REFERENCES.json"
        index.parent.mkdir()
        dump_json(index, dict(references=references))
        audit = audit_stage(
            folder, panel["requests"], {arm: value["checkpoint_path"] for arm, value in checkpoints.items()},
            canonicals=refs, sealed_n=256, schedule=plan["evaluation_schedule"],
            physical_gpu=plan["serving_gpu_identity"], evidence_root=temporary,
        )
    result, paired = analysis.endpoint_from_audit(audit)
    result["raw_audit"] = {key: value for key, value in audit.items() if key != "rows"}
    return result, paired, measurements


def coordinate_summary(value, label, errors):
    """Keep BF16 realization distinct from aggregate changes in other dtypes."""
    parameters = value.get("parameters", []) if isinstance(value, dict) else []
    if not parameters:
        errors.append(label + "_dtype_evidence_missing")
        return None
    try:
        if (sum(p["changed"] for p in parameters) != value["changed"]
                or sum(p["coordinates"] for p in parameters) != value["coordinates"]
                or any(not 0 <= p["changed"] <= p["coordinates"] for p in parameters)):
            errors.append(label + "_coordinate_totals")
        return dict(
            **{key: item for key, item in value.items() if key != "parameters"},
            bf16_changed=sum(p["changed"] for p in parameters if p.get("dtype") == "torch.bfloat16"),
            dtype_evidence_available=True,
        )
    except (KeyError, TypeError, ValueError):
        errors.append(label + "_coordinate_metadata_invalid")
        return None


def qualification(evidence):
    admission = evidence.read("qualification/ADMISSION.json")
    exposure = evidence.read("qualification/exposure/RESULT.json")
    cases, errors = {}, []
    for name in ("NATIVE", "ZERO", "ACTIVE"):
        row = evidence.read(f"qualification/{name}/RESULT.json")
        if row is None:
            errors.append("missing_" + name)
            continue
        expected_identity = (admission or {}).get("cases", {}).get(name)
        observed_identity = evidence.files.get(str(evidence.local(f"qualification/{name}/RESULT.json")))
        if expected_identity != observed_identity:
            errors.append(name + "_admission_identity_changed")
        cases[name] = {key: row.get(key) for key in (
            "status", "window", "response_forwards", "response_backwards", "optimizer_steps",
            "transferred_weight", "active_intervention", "finite", "exact", "resume",
            "next_anchor_draws", "gradient_changes", "weight_changes", "wall_s", "peak_allocated_bytes",
        )}
        if name != "NATIVE":
            for key in ("gradient_changes", "weight_changes"):
                cases[name][key] = coordinate_summary(row.get(key), name + "_" + key, errors)
        if row.get("status") != "completed" or row.get("finite", {}).get("status") != "passed":
            errors.append(name + "_not_finite_complete")
        if name != "NATIVE" and row.get("exact", {}).get("exact_equal") is not True:
            errors.append(name + "_identity_or_rng_failed")
        if (row.get("response_forwards"), row.get("response_backwards"), row.get("optimizer_steps")) != (2, 2, 1):
            errors.append(name + "_work_changed")
        if row.get("next_anchor_draws") != 1:
            errors.append(name + "_next_anchor_check_missing")
        observations = row.get("observations", [])
        if len(observations) != 2:
            errors.append(name + "_response_observations_missing")
        else:
            moved = sum(o.get("sum_metrics", {}).get(PREFIX + "transferred_weight", 0.) for o in observations)
            if not math.isfinite(moved) or row.get("transferred_weight") != moved or row.get("active_intervention") != (moved > 0):
                errors.append(name + "_actual_transfer_disagrees")
            if name != "NATIVE" and any(o.get("sum_metrics", {}).get(PREFIX + "disabled") != int(name == "ZERO") for o in observations):
                errors.append(name + "_reducer_mode_changed")
            if name == "ZERO" and moved != 0:
                errors.append("ZERO_transferred_weight")
    deferred = evidence.read("training/SECOND/deferred_qualification/RESULT.json", optional=True)
    deferred_errors = []
    if deferred and deferred.get("status") == "completed":
        for key in ("gradient_changes", "weight_changes"):
            deferred[key] = coordinate_summary(deferred.get(key), "deferred_" + key, deferred_errors)
        candidate = deferred.get("candidate_observations", [])
        moved = sum(row.get("sum_metrics", {}).get(PREFIX + "transferred_weight", 0) for row in candidate)
        if len(candidate) != 2 or not math.isfinite(moved) or deferred.get("transferred_weight") != moved:
            deferred_errors.append("deferred_actual_transfer_disagrees")
        errors.extend(deferred_errors)
    positive = cases.get("ACTIVE", {}).get("transferred_weight", 0) or 0
    late_work = (deferred or {}).get("extra_work", {})
    late_limits = dict(response_forwards=2, response_backwards=2, optimizer_steps=1)
    late_valid = bool(deferred and not deferred_errors and deferred.get("status") == "completed"
                      and deferred.get("same_inputs_rng", {}).get("exact_equal") is True
                      and deferred.get("exact_candidate_restoration", {}).get("exact_equal") is True
                      and deferred.get("transferred_weight", 0) > 0
                      and late_work.get("attempts") == late_work.get("completed") == late_limits
                      and late_work.get("unknown") == {key: 0 for key in late_limits})
    if deferred and deferred.get("status") == "completed" and not late_valid:
        errors.append("deferred_comparison_or_work_failed")
    if not admission or admission.get("status") != "passed":
        errors.append("missing_or_failed_admission")
    if not exposure or exposure.get("status") != "completed":
        errors.append("missing_or_failed_early_exposure")
    elif (
        (exposure.get("response_forwards"), exposure.get("response_backwards"), exposure.get("optimizer_steps")) != (16, 0, 0)
        or exposure.get("parent_weights_unchanged", {}).get("exact_equal") is not True
        or exposure.get("exposure_is_parent_state_no_update") is not True
        or len(exposure.get("windows", [])) != 8
    ):
        errors.append("early_exposure_budget_or_parent_state")
    else:
        active_windows = []
        for ordinal, window in enumerate(exposure["windows"]):
            if window.get("window") != ordinal or len(window.get("responses", [])) != 2:
                errors.append("early_exposure_window_identity")
                continue
            if any(row.get("sum_metrics", {}).get(PREFIX + "transferred_weight", 0) > 0 for row in window["responses"]):
                active_windows.append(ordinal)
        if exposure.get("first_active_window") != (active_windows[0] if active_windows else None):
            errors.append("early_exposure_first_positive_changed")
    return dict(
        status="passed" if not errors else "invalid_or_incomplete", errors=errors, cases=cases,
        early_exposure={key: (exposure or {}).get(key) for key in ("status", "first_active_window", "response_forwards", "response_backwards", "optimizer_steps", "exposure_is_parent_state_no_update", "parent_weights_unchanged")},
        positive_transfer_checked=positive > 0 or late_valid,
        deferred={key: deferred.get(key) for key in ("status", "transferred_weight", "global_step_before", "record_ids", "gradient_changes", "weight_changes", "same_inputs_rng", "exact_candidate_restoration", "extra_work", "diagnostic_wall_s")} if deferred else None,
        comparison_scope="Numerical equality and restoration are execution receipts; report generation does not reload/recompute qualification tensors.",
    )


def integrity(evidence, plan, training, checkpoints, cost):
    """Required execution gates; matched-start availability is separate."""
    errors = []
    preflight = evidence.read("PREFLIGHT.json")
    postflight = evidence.read("POSTFLIGHT.json")
    if not preflight or preflight.get("status") != "passed":
        errors.append("preflight_missing_or_failed")
    if not postflight or postflight.get("status") != "passed":
        errors.append("postflight_missing_or_failed")
    state = cost.get("training_status") or {}
    if state.get("status") != "completed" or state.get("final_step") != 640:
        errors.append("training_status_incomplete")
    boundary = (training or {}).get("boundary", {})
    if boundary.get("status") != "passed" or boundary.get("export_exact") is not True:
        errors.append("training_state_export_boundary_missing")
    if checkpoints and training and checkpoints.get("SECOND", {}).get("weights_identity") != training.get("weights_identity"):
        errors.append("candidate_checkpoint_result_identity_changed")
    limits = dict(response_forwards=256, response_backwards=256, optimizer_steps=128)
    ledger = evidence.read("training/SECOND/WORK.jsonl", lines=True)
    if ledger is None:
        errors.append("training_work_ledger_missing")
    else:
        from experiments.ok_coding.campaign import read_work, require_complete_work
        try:
            work = read_work(evidence.local("training/SECOND/WORK.jsonl"), limits)
            require_complete_work(work, limits)
            if work != (training or {}).get("work_ledger") or work != state.get("work"):
                errors.append("training_work_receipts_disagree")
        except (KeyError, TypeError, ValueError) as error:
            errors.append("training_work_invalid: " + repr(error))
    schedule_check = "Retained execution pre/postflight and training schedule check; original schedule not locally available."
    original = evidence.read(plan["original_schedule"], optional=True) if plan.get("original_schedule") else None
    if original is not None and training:
        from experiments.kplus_abt.campaign import schedule_bindings
        try:
            if file_identity(evidence.local(plan["original_schedule"])) != plan["original_schedule_identity"] or schedule_bindings(training) != schedule_bindings(original):
                errors.append("inherited_presentation_or_anchor_schedule_changed")
            else:
                schedule_check = "Recomputed exact inherited input/pass/anchor schedule binding."
        except (KeyError, TypeError, ValueError) as error:
            errors.append("inherited_schedule_invalid: " + repr(error))
    return dict(status="passed" if not errors else "invalid_or_incomplete", errors=errors,
                preflight=preflight, postflight=postflight, schedule_check=schedule_check)


def matched_summary(evidence, plan):
    retained = evidence.read("matched_start/SUMMARY.json", optional=True)
    status = evidence.read("matched_start/STATUS.json", optional=True)
    if not retained:
        return dict(status="unavailable", retained_status=status, planned_starts=128, planned_child_blocks=256)
    if retained.get("status") != "complete":
        return retained
    result = evidence.read("matched_start/RESULT.json")
    postflight = evidence.read("matched_start/POSTFLIGHT.json")
    rows = evidence.read("matched_start/ROWS.json")
    inputs = evidence.read(plan["matched_start"]) if plan.get("matched_start") else None
    errors = []
    if not result or result.get("status") != "completed" or not status or status.get("status") != "completed":
        errors.append("matched_phase_incomplete")
    if not postflight or postflight.get("status") != "passed":
        errors.append("matched_postflight_incomplete")
    recomputed = None
    if rows is None or inputs is None:
        errors.append("matched_raw_scored_rows_or_clusters_missing")
    else:
        from .matched_start import summarize
        try:
            recomputed = summarize(rows, inputs["request_clusters"])
            if recomputed != retained:
                errors.append("matched_saved_summary_disagrees_with_reduction")
        except (KeyError, TypeError, ValueError) as error:
            errors.append("matched_reduction_failed: " + repr(error))
    if errors:
        return dict(status="invalid_or_incomplete", errors=errors, retained_summary=retained,
                    scope="No local-mechanism support claim from invalid or incomplete evidence.")
    return dict(recomputed, report_reduction="Recomputed from retained scored rows; native parent/child validity remains execution evidence.")


def costs(evidence, plan, status, training):
    resources = evidence.read("RESOURCES.json")
    training_status = evidence.read("training/SECOND/STATUS.json")
    profile = evidence.read("training/SECOND/RESOURCE_COSTS.json")
    eval_status = evidence.read("evaluation/STATUS.json", optional=True)
    worker = evidence.read("evaluation/WORKER.json", optional=True)
    matched = evidence.read("matched_start/RESULT.json", optional=True)
    phases = {}
    for path in sorted(evidence.root.rglob("*_EXEC.json")):
        phases[str(path.relative_to(evidence.root))] = evidence.read(path)
    startups = {}
    for path in sorted(evidence.root.rglob("startup/RESULT.json")):
        startups[str(path.relative_to(evidence.root))] = evidence.read(path)
    deferred = evidence.read("training/SECOND/deferred_qualification/RESULT.json", optional=True)
    qualification_work = {}
    for name in ("exposure", "NATIVE", "ZERO", "ACTIVE"):
        row = evidence.read(f"qualification/{name}/RESULT.json", optional=True)
        qualification_work[name] = {
            key: (row or {}).get(key)
            for key in ("status", "response_forwards", "response_backwards", "optimizer_steps", "next_anchor_draws")
        }
    budget = plan.get("budget", {})
    main_work = (training or {}).get("work_ledger") or (training_status or {}).get("work")
    attempted = (main_work or {}).get("attempts", {})
    elapsed = (status or {}).get("gpu_clock_elapsed_s")
    previous_attempts = []
    for previous in plan.get("previous_attempts", []):
        retained, errors = {}, []
        for path, expected in previous.get("files", {}).items():
            value = evidence.read(path, optional=True)
            if value is not None:
                retained[path] = value
                if file_identity(evidence.local(path)) != expected:
                    errors.append("Previous-attempt identity differs: " + path)
                    evidence.errors.append(errors[-1])
        previous_attempts.append(dict(
            sealed_accounting=previous, retained_local_evidence=retained, errors=errors,
            local_identity_check="passed" if len(retained) == len(previous.get("files", {})) and not errors else "incomplete_or_invalid",
        ))
    return dict(
        resources=resources, durable_phase_receipts=phases,
        training_status=training_status, training_resource_profile=profile,
        training_event_clocks={key: (training or {}).get("costs", {}).get(key) for key in ("training_s", "checkpoint_save_s", "export_s", "event_wall_s_before_commit")},
        main_work=main_work, discarded_startups=startups,
        qualification_returned_work=qualification_work,
        deferred_extra_work=(deferred or {}).get("extra_work"),
        deferred_nested_wall_s=(deferred or {}).get("diagnostic_wall_s"),
        evaluation_status=eval_status, evaluation_worker=worker, matched_start=matched,
        budget=budget, gpu_clock_elapsed_s=elapsed,
        prior_gpu_seconds=plan.get("prior_gpu_seconds", 0),
        this_attempt_gpu_seconds=(status or {}).get("this_attempt_gpu_seconds"),
        previous_attempts=previous_attempts,
        remaining_gpu_ceiling_s=max(0., budget.get("gpu_seconds", 4500) - elapsed) if elapsed is not None else None,
        unattempted_main_optimizer_steps=max(0, 128 - attempted["optimizer_steps"]) if "optimizer_steps" in attempted else None,
        unattempted_timed_completions=(eval_status or {}).get("unattempted_generations", 1024 if (status or {}).get("status") == "completed_inactive" else None),
        clocks="Phase, event, compiler, startup and deferred clocks nest. They are retained separately, never summed into request latency or subtracted as intrinsic treatment overhead.",
        warning="Generic RESOURCE_COSTS extra-work zeros describe its passive profiler, not any deferred native twin. Its actual ledger is reported separately. Missing failed-phase receipts imply unknown work, never zero.",
    )


def _effect_text(effect):
    if not effect or effect.get("point_pct") is None:
        return "unavailable"
    interval = effect.get("ci95_pct")
    suffix = f" [{interval[0]:+.4f}%, {interval[1]:+.4f}%]" if interval else " [interval undefined]"
    return f"{effect['point_pct']:+.4f}%" + suffix


def markdown(summary):
    endpoint_result, exposure = summary["endpoint"], summary["exposure"]
    decision = summary["decision"]
    lines = ["# Second-follower training pilot", "",
        "Decision: **" + decision["outcome"] + "**.", "",
        f"Execution commit: `{summary['execution_source'].get('commit', 'unavailable')}`. "
        "One GSM8K continuation from step512 to640; native KPLUS5 weights are reused as NATIVE. "
        "Both serving arms use the same native algorithm. The exact PR39 panel has known prior project evaluation; this single inherited seed does not establish generalization.", ""]
    lines += ["| Sealed input / checkpoint | SHA256 |", "|---|---|"]
    for name, value in summary.get("input_identities", {}).items():
        lines.append(f"| {name} | `{(value or {}).get('sha256', 'unavailable')}` |")
    for arm, value in (summary.get("checkpoints") or {}).items():
        lines.append(f"| {arm} serving weights | `{value['weights_identity']['sha256']}` |")
    lines.append("")
    if endpoint_result.get("summary"):
        report = endpoint_result["summary"]
        effect = report["comparisons"][analysis.CONTRAST]
        lines += [
            "| Endpoint, SECOND versus NATIVE | Reduction and paired 95% interval |",
            "|---|---:|",
            f"| Verification calls | {_effect_text(effect['call_reduction'])} |",
            f"| Complete request time | {_effect_text(effect['request_latency_s_reduction'])} |",
            f"| Native generation time | {_effect_text(effect['native_generate_s_reduction'])} |",
            f"| Cache maintenance time | {_effect_text(effect['cache_maintenance_s_reduction'])} |", "",
            f"Gross calls saved/added: {effect['gross_calls_saved']}/{effect['gross_calls_added']}; "
            f"net saved {effect['net_calls_saved']}. Improved/worse/tied requests: "
            f"{effect['request_wins']}/{effect['request_regressions']}/{effect['request_ties']}. "
            "Counts use one validated trajectory; times average the two repeats within each request. "
            "Intervals use10,000 paired whole-request PCG64 draws, seed20261113, linear percentiles.", "",
            "Complete-time sweep reductions: " + ", ".join(f"sweep{int(k)+1}: {v['request_latency_s_reduction_pct']:+.4f}%" for k, v in report["sweep_reductions"].items()) + ".",
            "A latency benefit requires a positive lower confidence bound and both favorable sweeps. "
            "A call interval spanning zero is unresolved; it does not establish equivalence.", "",
            "| Arm | Calls, one trajectory per request | Mean-repeat request-time sum (s) | Capped requests |",
            "|---|---:|---:|---:|",
        ]
        for arm in METHODS:
            total = report["method_totals"][arm]
            lines.append(f"| {arm} | {total['verification_calls']} | {total['request_latency_s']:.6f} | {total['cap_hit_requests']} |")
        lines.append("")
        if not summary["inference_permitted"]:
            lines += ["**Endpoint numbers are descriptive only: one or more required validity/qualification gates are incomplete or invalid.**", ""]
    else:
        lines += ["Complete-request endpoint: " + endpoint_result["status"] + ". No endpoint benefit or speed claim is available.", ""]
    totals = exposure.get("totals", {})
    lines += [
        f"Exposure audit: **{exposure['status']}**. Observed presentations {exposure.get('presentations_observed', 0)}/256; "
        f"optimizer steps {exposure.get('optimizer_steps_observed', 0)}/128. "
        f"Active responses/steps: {exposure.get('active_responses', 0)}/{exposure.get('active_steps', 0)}. "
        f"SHORT/LONG/censored occurrences: {totals.get('short_count', 'unavailable')}/{totals.get('long_count', 'unavailable')}/{totals.get('censored_count', 'unavailable')}.", "",
        f"Transferred mass: {totals.get('transferred_weight', 'unavailable')}; "
        f"sum transfer / sum selected weight: {exposure.get('transferred_over_selected')}; "
        f"sum transfer / sum native high weight: {exposure.get('transferred_over_native_high')}; "
        f"mean response transfer / selected denominator: {exposure.get('mean_response_transferred_over_selected')}.",
        f"Largest response/depth shares of total transfer: {exposure.get('largest_response_transfer_share')}/"
        f"{exposure.get('largest_depth_transfer_share')}; mixed/active depth strata: "
        f"{totals.get('mixed_strata', 'unavailable')}/{totals.get('active_strata', 'unavailable')}.",
        "The scalar audit independently recomputes the capped depth transfer, donor floor, mass, native coefficients, numerator correction and unchanged denominator. "
        "It cannot reconstruct unsaved per-anchor proposals or class membership. Per-response and depth metrics are in EXPOSURE.csv and DEPTH_EXPOSURE.csv.", "",
        "Qualification: " + summary["qualification"]["status"] + "; actual positive-transfer window checked: " + str(summary["qualification"]["positive_transfer_checked"]) + ". "
        "Exact state/gradient/RNG comparisons are retained execution receipts, not a second numerical replay by this report.", "",
    ]
    for name, row in summary["qualification"]["cases"].items():
        lines.append(f"- {name}: transferred mass {row.get('transferred_weight')}; changed BF16 gradient/weight coordinates "
                     f"{(row.get('gradient_changes') or {}).get('bf16_changed')}/{(row.get('weight_changes') or {}).get('bf16_changed')}; "
                     f"exact comparison {(row.get('exact') or {}).get('exact_equal')}; next anchor draws {row.get('next_anchor_draws')}.")
    deferred = summary["qualification"].get("deferred")
    if deferred:
        lines += ["", f"Deferred first-positive qualification: {deferred['status']}; transferred mass {deferred.get('transferred_weight')}; "
                  f"changed BF16 gradient/weight coordinates {(deferred.get('gradient_changes') or {}).get('bf16_changed')}/"
                  f"{(deferred.get('weight_changes') or {}).get('bf16_changed')}; exact candidate restoration "
                  f"{(deferred.get('exact_candidate_restoration') or {}).get('exact_equal')}."]
    matched = summary["matched_start"]
    lines += ["", "Matched later-parent-start diagnostic: **" + matched.get("status", "unavailable") + "**. "
              "These local accepted-prefix changes are separate from full-request calls and latency; other parent starts remain unmeasured.", ""]
    if matched.get("status") == "complete":
        lines += ["| Class | NATIVE repairs / earlier breaks | SECOND repairs / earlier breaks | Paired net prefix tokens |", "|---|---:|---:|---:|"]
        for group, values in matched["groups"].items():
            native, second = (values["arms"][arm]["metrics"] for arm in METHODS)
            delta = values["paired_SECOND_minus_NATIVE"]["native_prefix_change"]
            lines.append(f"| {group} | {native['repair']['total']}/{native['earlier_prefix_break']['total']} | {second['repair']['total']}/{second['earlier_prefix_break']['total']} | {delta['total']:+g} |")
        weighted = matched["census_weighted_SECOND_minus_NATIVE"]
        lines += ["", f"Census-weighted LONG+SHORT candidate-minus-control change per later parent start: {weighted['per_later_parent_start']:+.6f}; "
                  f"request-cluster 95% interval {weighted.get('ci95')}. LONG weight403/2697; SHORT263/2697. "
                  "Full follower acceptance, donor losses, paired changes and contributing request counts are retained in SUMMARY.json.", ""]
        short = matched["groups"]["SHORT"]
        lines += [f"SHORT donor starts, candidate versus control: gross prefix tokens gained/lost "
                  f"{short['paired_gross_gained']}/{short['paired_gross_lost']}; improved/worse/tied "
                  f"{short['improved']}/{short['worse']}/{short['tied']}. These losses remain reported regardless of endpoint direction.", ""]
    resource = summary["costs"]
    event_cost = resource["training_event_clocks"]
    lines += [
        f"Campaign status: {summary['run_status'].get('status', 'unavailable')}. Conservative occupied-GPU clock: "
        f"{resource['gpu_clock_elapsed_s']} s of4500; unused ceiling {resource['remaining_gpu_ceiling_s']} s. "
        f"Unattempted main steps/timed completions: {resource['unattempted_main_optimizer_steps']}/{resource['unattempted_timed_completions']}.", "",
        "Training, qualification/startup, publication/loading, endpoint requests and matched replays have separate recorded clocks. "
        "Nested timers are not added. Native control training is historical reuse, so this run provides no contemporaneous NATIVE training-cost comparison. "
        "Unknown or missing work is not imputed as zero. See SUMMARY.json for phase receipts and planned versus observed work.", "",
        f"Candidate event clocks (seconds): training {event_cost['training_s']}; checkpoint save {event_cost['checkpoint_save_s']}; "
        f"export {event_cost['export_s']}; enclosing event wall {event_cost['event_wall_s_before_commit']}.", "",
        "| Recorded phase | Status | Wrapper wall seconds |",
        "|---|---|---:|",
    ]
    for name, receipt in resource["durable_phase_receipts"].items():
        lines.append(f"| {name} | {(receipt or {}).get('status', 'unavailable')} | {(receipt or {}).get('wall_s', 'unavailable')} |")
    lines += ["", "| Qualification phase (startup excluded) | Returned forwards | Returned backwards | Disposable updates |",
              "|---|---:|---:|---:|"]
    for name, work in resource["qualification_returned_work"].items():
        lines.append(f"| {name} | {work['response_forwards']} | {work['response_backwards']} | {work['optimizer_steps']} |")
    lines += ["", f"Discarded-startup receipts retained: {len(resource['discarded_startups'])}. Their work ledgers are separate from the table. "
              "Next-anchor checks are sampler operations, not model forwards. Failed qualification returned counters do not bound unknown in-flight work. "
              f"Deferred native-twin extra work: {resource['deferred_extra_work']}; its nested diagnostic wall is {resource['deferred_nested_wall_s']} s.", "",
        "Source/input/checkpoint identities, raw-audit validity failures and evidence stability are retained in SUMMARY.json and INPUT_MANIFEST.json. "
        "Raw token records and tensors remain in the private run directory; exported tables contain request identifiers and metrics only. "
        "No new seed, tuning, configuration, training event or follow-up is authorized by a favorable result.", "",
    ]
    for attempt in resource["previous_attempts"]:
        saved = attempt["sealed_accounting"]
        lines += [f"Preserved prior attempt `{saved.get('execution_commit')}`: {saved.get('reason')}. "
                  f"Charged {saved.get('response_forwards')} forwards, {saved.get('response_backwards')} backwards, "
                  f"{saved.get('optimizer_steps')} Adam steps and {saved.get('gpu_seconds')} occupied-GPU seconds. "
                  "This cost is included in the same4500-second ceiling, not added to its remaining allowance. "
                  f"Retrieved identity check: {attempt['local_identity_check']}.", ""]
    return "\n".join(lines)


def generate(root, output, *, path_maps=()):
    root, output = Path(root).resolve(), Path(output).resolve()
    plan = json.loads((root / "sealed/PLAN.json").read_text())
    evidence = Evidence(root, plan["root"], path_maps)
    evidence.read("sealed/PLAN.json")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Preserve existing report; choose a fresh output directory")
    output.mkdir(parents=True, exist_ok=True)
    status = evidence.read("STATUS.json") or {}
    source = evidence.read(plan.get("source_manifest", "SOURCE.json")) or {}
    seal = evidence.read("sealed/SEAL.json") or {}
    provenance_errors = []
    for path, expected in seal.get("files", {}).items():
        local = evidence.local(path)
        if not local.is_file() or file_identity(local) != expected:
            provenance_errors.append("Sealed identity differs or missing: " + str(path))
        elif local.suffix in (".json", ".jsonl"):
            evidence.read(local, lines=local.suffix == ".jsonl")
        else:
            evidence.files[str(local)] = expected
    if not seal or source.get("commit") != plan["execution_source"]["commit"]:
        provenance_errors.append("Missing seal or mismatched execution source")
    checkpoints = evidence.read("CHECKPOINTS.json", optional=True)
    training = evidence.read("training/SECOND/RESULT.json")
    exposure, exposure_rows, depth_rows = exposure_audit(training)
    qual = qualification(evidence)
    endpoint_result, paired, repeats = endpoint(evidence, plan, checkpoints)
    matched = matched_summary(evidence, plan)
    cost = costs(evidence, plan, status, training)
    checks = integrity(evidence, plan, training, checkpoints, cost)
    for folder in ("qualification", "training/SECOND", "matched_start"):
        evidence.remember_tree(folder)
    if training and exposure["status"] == "passed" and exposure["totals"]["transferred_weight"] > 0 and not qual["positive_transfer_checked"]:
        qual["status"] = "invalid_or_incomplete"
        qual["errors"].append("Positive event exposure lacks a completed same-state qualification")
    stable = evidence.stable()
    permitted = bool(endpoint_result.get("inference_permitted") and exposure["status"] == "passed"
                     and qual["status"] == "passed" and checks["status"] == "passed"
                     and not provenance_errors and not evidence.errors and stable)
    inactive_condition = bool(
        exposure.get("totals", {}).get("transferred_weight") == 0
        or (checkpoints and set(checkpoints) == set(METHODS)
            and checkpoints["NATIVE"]["weights_identity"] == checkpoints["SECOND"]["weights_identity"])
    )
    inactive = bool(status.get("status") == "completed_inactive" and inactive_condition
                    and exposure["status"] == "passed" and qual["status"] == "passed"
                    and checks["status"] == "passed" and not provenance_errors and not evidence.errors and stable)
    decision = analysis.decision(endpoint_result.get("summary")) if permitted else dict(
        outcome="inactive_intervention" if inactive else "invalid_or_incomplete",
        automatic_followup_authorized=False,
    )
    if permitted and matched.get("status") == "complete":
        long = matched["groups"]["LONG"]["paired_SECOND_minus_NATIVE"]["native_prefix_change"]
        weighted = matched["census_weighted_SECOND_minus_NATIVE"]
        decision["local_explanation_direction_supported"] = long["total"] > 0 and weighted["per_later_parent_start"] > 0
        decision["local_explanation_scope"] = "Positive directions only; request-cluster intervals and donor loss reported separately."
    summary = dict(
        schema="second_follower_report_v1", status="complete" if permitted or inactive else "invalid_or_incomplete",
        inference_permitted=permitted, decision=decision, execution_source=plan["execution_source"],
        run_status=status, binding=plan.get("binding"), checkpoints=checkpoints,
        source_authentication=dict(errors=provenance_errors, evidence_stable=stable, source_manifest_identity=evidence.files.get(str(evidence.local(plan.get("source_manifest", "SOURCE.json"))))),
        endpoint=endpoint_result, exposure=exposure, qualification=qual, matched_start=matched, costs=cost,
        execution_integrity=checks,
        evidence_errors=evidence.errors, missing_evidence=evidence.missing, model_work=0,
    )
    summary["input_identities"] = {
        key: evidence.files.get(str(evidence.local(plan[key])))
        for key in ("panel_manifest", "references", "matched_start") if key in plan
    }
    summary["input_identities"].update({
        key: plan[key] for key in ("origin_state_identity", "origin_weights_identity", "original_schedule_identity")
        if key in plan
    })
    summary["input_identities"]["sealed_plan"] = evidence.files.get(str(root / "sealed/PLAN.json"))
    code = Path(__file__).resolve().parent
    analysis_source = dict(files={name: file_identity(code / name) for name in ("report.py", "analysis.py", "evaluation.py")})
    try:
        analysis_source["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=code, text=True, stderr=subprocess.DEVNULL
        ).strip()
        analysis_source["dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=code, text=True, stderr=subprocess.DEVNULL
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        analysis_source["commit"] = None
        analysis_source["dirty"] = None
    summary["analysis_source"] = analysis_source
    dump_json(output / "SUMMARY.json", summary)
    dump_json(output / "EXPOSURE_AUDIT.json", exposure)
    dump_csv(output / "PER_REQUEST.csv", analysis.request_table(paired))
    repeat_fields = ("request_id", "panel_ordinal", "method", "repeat", "event", "dataset", "checkpoint_id", *analysis.COUNT_FIELDS, *analysis.TIMINGS, "finish_reason", "cap_hit", "validation_status", "valid_measurement", "errors")
    dump_csv(output / "REPEATS.csv", [{field: row.get(field) for field in repeat_fields} for row in repeats])
    dump_csv(output / "EXPOSURE.csv", exposure_rows)
    dump_csv(output / "DEPTH_EXPOSURE.csv", depth_rows)
    dump_json(output / "INPUT_MANIFEST.json", dict(
        source_root=plan["root"], analysis_root=str(root), files=evidence.files,
        missing=evidence.missing, errors=evidence.errors, stable=stable,
        execution_commit=plan["execution_source"]["commit"], model_work=0,
        path_maps=[dict(original=str(original), local=str(local)) for original, local in evidence.path_maps],
    ))
    (output / "REPORT.md").write_text(markdown(summary))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--path-map", action="append", default=[], metavar="ORIGINAL=LOCAL",
                        help="Optional retained external dependency or prior-run path prefix mapping")
    args = parser.parse_args()
    maps = []
    for value in args.path_map:
        if "=" not in value:
            parser.error("--path-map requires ORIGINAL=LOCAL")
        original, local = value.split("=", 1)
        if not Path(original).is_absolute() or not local:
            parser.error("--path-map original must be absolute and local nonempty")
        maps.append((original, local))
    result = generate(args.root, args.output, path_maps=maps)
    print(json.dumps(dict(status=result["status"], decision=result["decision"], output=str(args.output))))


if __name__ == "__main__":
    main()
