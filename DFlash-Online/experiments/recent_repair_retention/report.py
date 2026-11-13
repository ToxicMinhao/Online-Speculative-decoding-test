"""CPU-only, redacted reporting for the prespecified retention event.

Raw serving files are revalidated before any gain is calculated. Incomplete
runs retain accounting and exposure but receive no endpoint performance claim.
Only explicit scalar/identity fields enter the public report directory.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path

from . import analysis, evaluation

ARMS = analysis.ARMS
REPEAT_FIELDS = ("request_id", "method", "repeat", "panel_ordinal", "original_index",
    "prompt_tokens", "output_tokens", "verification_calls", "accepted_draft_tokens",
    "native_accepted_draft_tokens", "committed_progress", "finish_reason", "cap_hit",
    "request_latency_s", "native_generate_s", "cache_maintenance_s", "valid_measurement")
SLOT_FIELDS = ("request_id", "canonical_start", "proposal_slot", "presentation_id",
    "pass_index", "anchor_ordinal", "a5", "a6", "valid_proposals", "kind", "status",
    "slot_id", "margin", "weight", "floor", "placement_switched", "source_slot_id")
OUTCOME_FIELDS = ("request_id", "canonical_start", "method", "accepted", "parent_accepted",
    "valid_proposals", "repair_slot", "genuine_rejection", "exact_slot_relapse",
    "earlier_break", "accepts_through", "direct_canonical_agreement", "parent_first_blocker",
    "passes_parent_first_blocker", "acquired_prefix_occurrences", "lost_prefix_occurrences",
    "net_prefix_occurrences", "valid")
START_FIELDS = ("request_id", "canonical_start", "round_ordinal", "a5", "a6",
    "valid_proposals", "genuine_k5_rejection", "genuine_k6_rejection", "repair_slot", "parity_valid")
SAFE_STATUS_FIELDS = ("status", "phase", "execution_commit", "arm", "wall_s", "loading_s",
    "planned_generations", "attempted_generations", "returned_generations",
    "completed_valid_generations", "invalid_generations", "unknown_generation_attempts",
    "unattempted_generations", "pending_canonical_generations", "evaluation_wall_s",
    "activation_s", "weight_validation_s", "optimizer_steps", "feedback_collections",
    "teacher_preparations", "extra_generations", "automatic_retry", "attempted", "returned",
    "valid", "reused", "generation_s", "scientific_response_forwards", "scientific_adam_steps",
    "global_step", "final_global_step", "presentations", "response_forwards", "projection_chunks",
    "backward_calls", "peak_allocated_bytes", "peak_reserved_bytes", "gpu_seconds",
    "gpu_seconds_limit", "parent_requests", "planned_parent_requests", "endpoint_blocks", "combined_campaign_wall_s")


def read(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def rows(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def select(row, names):
    return {name: row[name] for name in names if name in row and
            (row[name] is None or isinstance(row[name], (str, bool, int, float)))}


def csv_table(path, records, fields=()):
    names = list(fields) or sorted({key for row in records for key in row})
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names or ["status"])
        writer.writeheader()
        writer.writerows(records)


def numeric_leaves(value, prefix=""):
    """Cost receipts have private paths too; emit only numeric leaves."""
    if isinstance(value, dict):
        for key, child in value.items():
            yield from numeric_leaves(child, f"{prefix}.{key}" if prefix else key)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield prefix, value


def response_exposure(records, arm):
    """Use each original response observation once, never accumulated metrics."""
    output, seen = [], set()
    for ordinal, row in enumerate(records):
        observed = row["response_observation"]
        pid = int(observed.get("presentation_ordinal", ordinal))
        if pid in seen:
            raise ValueError("Duplicate response presentation in exposure accounting")
        seen.add(pid)
        if observed["record_id"] != row["record_id"]:
            raise ValueError("Response observation identity differs from its presentation")
        values = observed["sum_metrics"]
        if values["kplus_response_count"] != 1:
            raise ValueError("Exposure observation must describe exactly one response")
        selected = {key: float(value) for key, value in values.items()
            if key.startswith(("retention_", "kplus_", "first_error_position_")) or key in (
                "available_position_occurrences", "selected_position_occurrences",
                "available_position_weight", "selected_position_weight", "main_loss_numerator",
                "loss_denominator", "supervised_position_exposure", "no_error_blocks", "empty_blocks")}
        if not all(math.isfinite(value) for value in selected.values()):
            raise ValueError("Nonfinite observed training exposure")
        output.append(dict(arm=arm, presentation_id=pid, request_id=row["record_id"],
            pass_index=int(row["pass"]), optimizer_step_after=row["optimizer_step_after"],
            optimizer_stepped=row["optimizer_stepped"], training_s=row["training_s"], **selected))
        if "kplus_N_K" in selected and "kplus_S" in selected:
            output[-1]["ordinary_selected_kl"] = selected["kplus_N_K"] / (selected["kplus_S"] + 1e-6)
    totals = {key: math.fsum(float(row.get(key, 0)) for row in output)
        for key in {key for row in output for key in row if key.startswith(("retention_", "kplus_")) or
                    key in ("available_position_occurrences", "selected_position_occurrences",
                            "available_position_weight", "selected_position_weight", "loss_denominator",
                            "main_loss_numerator", "supervised_position_exposure")}}
    totals["response_presentations"] = len(output)
    totals["adam_steps"] = sum(row["optimizer_stepped"] for row in output) if all(
        row["optimizer_stepped"] is not None for row in output) else None
    totals["firing_presentations"] = sum(row.get("retention_firing_count", 0) > 0 for row in output)
    totals["firing_windows"] = len({row["presentation_id"] // 2 for row in output
                                     if row.get("retention_firing_count", 0) > 0})
    # Coefficient is a setting, not additive exposure.
    coefficients = {row["retention_coefficient"] for row in output if "retention_coefficient" in row}
    totals.pop("retention_coefficient", None)
    totals["retention_coefficient"] = next(iter(coefficients)) if len(coefficients) == 1 else None
    return output, totals


def matching_rows(assignments):
    output = []
    for pair in assignments.get("matches", []):
        for index, pid in enumerate(pair["presentation_ids"]):
            output.append(dict(select(pair, ("source_slot_id", "destination_slot_id", "request_id",
                "source_start", "destination_start", "proposal_slot", "a6", "total_margin_distance",
                "switched_raw_weight", "tie_sha256")), presentation_id=pid,
                source_margin=pair["source_margins"][index],
                destination_margin=pair["destination_margins"][index]))
    return output


def training_evidence(root):
    presentations, costs, summaries, receipts, endpoints = [], [], {}, [], []
    differences = read(root / "ENDPOINT_DIFFERENCES.json", {})
    for arm in ARMS:
        folder = root / "training" / arm
        status = read(folder / "STATUS.json", {})
        result = read(folder / "RESULT.json")
        if result is None:
            result = read(folder / "trainer/events/event_007/RESULT.json", {})
        original = result.get("records", [])
        if not original:
            observed = rows(folder / "trainer/events/event_007/RESPONSE_OBSERVATIONS.jsonl")
            # Interrupted response observations remain visible, but no optimizer
            # completion is inferred from them.
            original = [dict(record_id=row["record_id"], **{"pass":row["pass_index"]},
                response_observation=row, optimizer_step_after=None, optimizer_stepped=None,
                training_s=None) for row in observed]
        if original:
            active, total = response_exposure(original, arm)
            presentations.extend(active)
            summaries[arm] = dict(total, status=status.get("status", result.get("status", "unknown")))
        else:
            summaries[arm] = dict(status=status.get("status", "unattempted"),
                                 response_presentations=0, adam_steps=0)
        summaries[arm]["work_ledger"] = status.get("work", {})
        summaries[arm]["costs"] = select(result.get("costs", {}), (
            "training_s", "checkpoint_save_s", "export_s", "event_wall_s_before_commit"))
        summaries[arm]["diagnostic_capture_s"] = result.get("diagnostic_capture_s")
        sources = {"event": result.get("costs", {}), "startup": status.get("startup", {}),
            "resource": read(folder / "RESOURCE_COSTS.json", {}),
            "first_firing": result.get("first_firing", read(folder / "first_firing_replay/RESULT.json", {})),
            "physical_work": status.get("physical_work", {}), "work_ledger": status.get("work", {}),
            "process": select(status, ("loading_s", "wall_s")),
            "diagnostic_capture": {"wall_s": result.get("diagnostic_capture_s", 0)}}
        for component, receipt in sources.items():
            costs.extend(dict(phase="training", arm=arm, component=component, metric=key, value=value)
                         for key, value in numeric_leaves(receipt))
        first = sources["first_firing"]
        first_row = dict(arm=arm, status=first.get("status", "unavailable"),
                         natural_firing=first.get("natural_firing"))
        first_row.update({key: value for key, value in numeric_leaves(first)
                          if not key.startswith(("work.", "active_physical_work.", "off_physical_work."))})
        for key in ("active_exact", "final_exact_restoration"):
            first_row[key + "_equal"] = first.get(key, {}).get("exact_equal")
        receipts.append(first_row)
        summaries[arm]["first_firing"] = first_row
        if arm in differences:
            summaries[arm]["final_vs_native"] = select(differences[arm], (
                "status", "different_coordinates", "coordinates", "difference_norm"))
            for parameter in differences[arm].get("parameters", []):
                endpoints.append(dict(arm=arm, **select(parameter, ("parameter", "dtype", "coordinates",
                    "different", "difference_squared_norm", "max_absolute_difference"))))
    return presentations, costs, summaries, receipts, endpoints


def _cost_rows(root, plan):
    result = []
    paths = {"reference_preparation": root / "reference/STATUS.json",
        "canonical_preparation": root / "canonicals/STATUS.json",
        "fixed_starts": root / "fixed_starts/STATUS.json",
        "serving": root / "SERVING_RESULT.json",
        "evaluation": root / "evaluation/STATUS.json", "campaign": root / "STATUS.json"}
    for case in ("NATIVE", "DISABLED", "ZERO", "ACTIVE"):
        paths["qualification_" + case] = Path(plan.get("qualification_root", root / "qualification")) / case / "RESULT.json"
    for phase, path in paths.items():
        for metric, value in numeric_leaves(read(path, {})):
            result.append(dict(phase=phase, arm="", component="receipt", metric=metric, value=value))
    for path in sorted((root / "processes").glob("*.json")):
        value = read(path)
        if "wall_s" in value:
            result.append(dict(phase=path.stem, arm="", component="allocated_process_window",
                               metric="wall_s", value=value["wall_s"]))
    return result


def pass_summaries(exposure):
    grouped = defaultdict(list)
    for row in exposure:
        grouped[row["arm"], row["pass_index"]].append(row)
    fields = ("retention_assigned_count", "retention_assigned_weight", "retention_reachable_count",
        "retention_reachable_weight", "retention_firing_count", "retention_firing_weight",
        "retention_firing_incorrect_count", "retention_firing_correct_count", "retention_auxiliary_loss",
        "retention_native_kl_loss", "retention_hinge_numerator", "retention_native_denominator")
    return [dict(arm=arm, pass_index=pass_index, presentations=len(values),
        firing_presentations=sum(row.get("retention_firing_count",0)>0 for row in values),
        **{key:math.fsum(row.get(key,0) for row in values) for key in fields})
        for (arm,pass_index),values in sorted(grouped.items())]


def sweep_totals(validated):
    output=[]
    for arm in ARMS:
        for repeat in (0,1):
            selected=[row for row in validated if row["method"]==arm and row["repeat"]==repeat]
            if selected:
                output.append(dict(arm=arm,repeat=repeat,requests=len(selected),
                    **{key:math.fsum(row[key] for row in selected) for key in (
                        "request_latency_s","native_generate_s","cache_maintenance_s")},
                    verification_calls=sum(row["verification_calls"] for row in selected),
                    accepted_draft_tokens=sum(row["accepted_draft_tokens"] for row in selected)))
    return output


def _plan(root):
    for path in (root / "TRAINING_PLAN.json", root / "sealed/PLAN.json", root / "PLAN.json"):
        if path.exists():
            return read(path)
    raise FileNotFoundError("No sealed experiment plan is available")


def generate_report(root, output=None):
    root = Path(root)
    output = Path(output) if output else root / "final_report"
    output.mkdir(parents=True, exist_ok=True)
    plan = _plan(root)
    status = read(root / "STATUS.json", {})
    continuation = read(root / "CONTINUATION.json", {})
    assignments = read(root / "ASSIGNMENTS.json", {})
    assignment_summary = assignments.get("summary", {})
    panel = read(plan["panel_manifest"])
    requests = panel["requests"]
    errors, paired, comparisons, fixed = [], [], {}, None
    audit = dict(status="unavailable", errors=[], rows=[], progress=read(root / "evaluation/STATUS.json", {}))
    if all(path.exists() for path in (root / "EXPORTS.json", root / "canonicals/CANONICAL.json",
                                     root / "evaluation/STATUS.json")):
        audit = evaluation.audit_stage(root / "evaluation", requests, read(root / "EXPORTS.json"),
            canonicals=read(root / "canonicals/CANONICAL.json"), dataset="gsm8k", event=7,
            schedule=read(plan["timing_schedule"]))
    raw = rows(root / "evaluation/MEASUREMENTS.jsonl")
    repeat_rows = []
    for row in raw:
        public = select(row, REPEAT_FIELDS)
        public["recorded_error_count"] = len(row.get("errors", []))
        public["raw_audit_passed"] = audit["status"] == "passed"
        repeat_rows.append(public)
    complete = status.get("status") == "completed" and audit["status"] == "passed"
    if complete:
        try:
            paired = analysis.paired_requests(audit["rows"])
            if len(paired) != 3 * len(requests):
                raise ValueError("Paired population differs from sealed panel")
            comparisons = {other: analysis.compare(paired, other) for other in ("NATIVE", "CONTROL")}
        except (ValueError, KeyError) as error:
            errors.append("serving_reduction_" + type(error).__name__)
            complete, comparisons = False, {}
    starts = read(root / "fixed_starts/STARTS.json", [])
    outcomes = read(root / "fixed_starts/OUTCOMES.json", rows(root / "fixed_starts/OUTCOMES.jsonl"))
    if complete:
        try:
            if read(root / "fixed_starts/STATUS.json", {}).get("status") != "completed":
                raise ValueError("Fixed-start phase is incomplete")
            fixed = analysis.fixed_start_summary(starts, outcomes, request_ids=plan["path_request_ids"])
        except (ValueError, KeyError) as error:
            errors.append("fixed_start_reduction_" + type(error).__name__)
            complete = False
    try:
        exposure, training_costs, training, gradient_receipts, endpoint_rows = training_evidence(root)
    except (ValueError, KeyError) as error:
        errors.append("training_accounting_" + type(error).__name__)
        complete = False
        exposure, training_costs, training, gradient_receipts, endpoint_rows = [], [], {}, [], []
    reused = read(root / "CONTROL_REUSE.json") is not None
    qualification = {case:select(read(Path(plan.get("qualification_root", root / "qualification")) /
        case / "RESULT.json", {}), SAFE_STATUS_FIELDS) for case in ("NATIVE", "DISABLED", "ZERO", "ACTIVE")}
    if complete:
        for case, receipt in qualification.items():
            if receipt.get("status") != "passed":
                errors.append("missing_or_failed_qualification_" + case)
        for arm in ARMS:
            if arm == "CONTROL" and reused:
                continue
            observed = training.get(arm, {})
            if (observed.get("status") != "completed" or observed.get("response_presentations") != 256
                    or observed.get("adam_steps") != 128):
                errors.append("incomplete_training_" + arm)
            if arm != "NATIVE":
                firing = observed.get("first_firing", {})
                if observed.get("firing_presentations", 0) and not (
                    firing.get("status") == "passed" and firing.get("active_exact_equal")
                    and firing.get("final_exact_restoration_equal")):
                    errors.append("unqualified_first_firing_" + arm)
                if observed.get("final_vs_native", {}).get("status") != "passed":
                    errors.append("unavailable_final_weight_difference_" + arm)
        if errors:
            complete = False
    if not complete:
        # Do not turn a valid-looking subset into the prespecified endpoint.
        comparisons, paired = {}, []
    verdict = analysis.decision(comparisons, fixed,
        assignment_summary.get("placement_screen_passed", False)) if complete else dict(
            decision="no_training_eligible_slots" if status.get("status") == "completed_no_treatment"
            else "incomplete_or_invalid_no_gain_claim", automatic_followup=False)
    cost_rows = training_costs + _cost_rows(root, plan)
    if continuation:
        cost_rows.extend(dict(phase="previous_failed_fixed_start",arm="",component="retained_failed_attempt",
                              metric=key,value=value) for key,value in numeric_leaves({
            "gpu_seconds":continuation["prior_fixed_start_gpu_s"],
            "work":continuation["failed_startup_work"]}))
    passes = pass_summaries(exposure)
    sweeps = sweep_totals(audit["rows"]) if audit["status"] == "passed" else []
    process_seconds = math.fsum(row["value"] for row in cost_rows if
        row["component"] == "allocated_process_window" or
        (row["phase"] in (("serving",) if continuation else ("canonical_preparation", "serving"))
         and row["metric"] == "wall_s"))
    prior_process_seconds = 0.
    if continuation:
        prior = read(Path(continuation["previous_execution_root"])/"final_report/SUMMARY.json", {})
        prior_process_seconds = continuation.get("prior_allocated_phase_window_s",
            prior.get("resources", {}).get("measured_allocated_phase_window_s"))
    combined_process_seconds = process_seconds + prior_process_seconds if prior_process_seconds is not None else None
    combined_wall = status.get("combined_campaign_wall_s", (status.get("wall_s", 0) +
        continuation.get("prior_campaign_wall_s", 0)) if "wall_s" in status else None)
    peak_allocated = max((row["value"] for row in cost_rows
        if row["metric"].endswith("peak_allocated_bytes")), default=None)
    peak_reserved = max((row["value"] for row in cost_rows
        if row["metric"].endswith("peak_reserved_bytes")), default=None)
    summary = dict(schema="recent-retention-event7-report-v1", verification_status="ANALYZED",
        execution_commit=plan["execution_commit"], source_status=select(status, SAFE_STATUS_FIELDS),
        training_execution_commit=continuation.get("training_execution_commit",plan["execution_commit"]),
        evaluation_execution_commit=continuation.get("evaluation_execution_commit",plan["execution_commit"]),
        continuation=select(continuation,("training_execution_commit","evaluation_execution_commit",
            "prior_campaign_wall_s","prior_fixed_start_gpu_s","actual_parent_generations_before",
            "pending_parent_generations","pending_timed_completions","new_optimizer_steps")),
        combined_campaign_wall_s=combined_wall,
        complete_valid_comparison=complete, report_errors=errors, panel_requests=len(requests),
        requested_panel_size=256, panel_shortfall=256-len(requests),
        public_redaction="Source identities and scalar measurements only; prompts/token IDs/tensors remain private.",
        planned=dict(scientific_adam_steps=384, scientific_response_presentations=768,
            qualification_temporary_adam_steps=3, qualification_response_presentations=6,
            first_firing_max_temporary_adam_steps=4, first_firing_max_response_presentations=8,
            timed_generations=6*len(requests), fixed_start_parent_requests=len(plan["path_request_ids"])),
        control_reused_recent=reused, eligibility=assignment_summary,
        dataset_revision=panel.get("source_revision"),
        selected_exposure_tiers=panel.get("selected_tier_counts", {}),
        qualification=qualification, pass_exposure=passes, sweep_totals=sweeps,
        resources=dict(physical_gpu=plan.get("serving_gpu_identity"),
            measured_allocated_phase_window_s=combined_process_seconds,
            measured_allocated_phase_window_card_hours=combined_process_seconds/3600 if combined_process_seconds is not None else None,
            current_execution_allocated_phase_window_s=process_seconds,
            prior_execution_allocated_phase_window_s=prior_process_seconds,
            phase_window_scope="Serialized child-process windows plus canonical/serving driver windows; includes CPU loading/cleanup, excludes gaps; not kernel-busy time.",
            peak_allocated_bytes=peak_allocated,peak_reserved_bytes=peak_reserved),
        frozen_selected_weight=assignments.get("frozen_selected_weight"),
        frozen_assigned_weight_fraction=assignments.get("frozen_assigned_weight_fraction"),
        training=training, comparisons=comparisons, fixed_starts=fixed,
        serving_audit=dict(status=audit["status"], error_count=len(audit["errors"]),
                           progress=select(audit.get("progress", {}), SAFE_STATUS_FIELDS)),
        decision=verdict, inference="Conditional paired request uncertainty on this one seed/shared-feedback fork.")
    csv_table(output / "per_request.csv", paired)
    csv_table(output / "all_repeats.csv", repeat_rows)
    csv_table(output / "fixed_start_origins.csv", [select(r, START_FIELDS) for r in starts])
    csv_table(output / "fixed_start_outcomes.csv", [select(r, OUTCOME_FIELDS) for r in outcomes])
    csv_table(output / "eligibility.csv", [select(r, SLOT_FIELDS) for r in assignments.get("eligibility", [])])
    csv_table(output / "matching.csv", matching_rows(assignments))
    csv_table(output / "assigned_slots.csv", [dict(arm=arm, **select(row, SLOT_FIELDS))
        for arm, values in assignments.get("arms", {}).items() for row in values])
    csv_table(output / "activation.csv", exposure)
    csv_table(output / "pass_activation.csv", passes)
    csv_table(output / "sweep_totals.csv", sweeps)
    csv_table(output / "gradient_receipts.csv", gradient_receipts)
    csv_table(output / "endpoint_differences.csv", endpoint_rows)
    csv_table(output / "costs.csv", cost_rows)
    csv_table(output / "panel.csv", [select(r, ("request_id", "source_id", "source_row", "original_index",
        "panel_ordinal", "prompt_ids_sha256", "normalized_question_sha256", "prior_endpoint_exposure",
        "exposure_tier", "selection_sha256")) for r in requests])
    csv_table(output / "checkpoint_identities.csv", [dict(arm=arm, artifact=kind, **select(value, ("sha256", "bytes")))
        for arm in ARMS for kind, value in read(root / "training" / arm / "RESULT.json", {}).items()
        if kind in ("weights_identity", "state_identity")])
    write(output / "SUMMARY.json", summary)
    write(output / "REPORT_SELF_CHECKS.json", self_checks(summary, paired, exposure))
    (output / "REPORT.md").write_text(markdown_report(summary))
    return summary


def self_checks(summary, paired, exposure):
    """A separate scalar summation checks reduction plumbing, not an independent replication."""
    checks = {}
    for arm in ARMS:
        observed = [r for r in exposure if r["arm"] == arm]
        checks[arm + "_unique_presentations"] = len(observed) == len({r["presentation_id"] for r in observed})
    if summary["complete_valid_comparison"]:
        for other, result in summary["comparisons"].items():
            for prefix, arm in (("recent", "RECENT"), ("reference", other)):
                selected = [r for r in paired if r["method"] == arm]
                checks[other + "_" + prefix + "_calls"] = sum(r["verification_calls"] for r in selected) == result[prefix+"_calls"]
                checks[other + "_" + prefix + "_accepted"] = sum(r["accepted_draft_tokens"] for r in selected) == result[prefix+"_accepted"]
    return dict(status="passed" if all(checks.values()) else "failed", checks=checks,
                scope="Separate scalar reduction and raw audit; same-team implementation, not independent GPU replication.")


def _number(value, digits=4):
    return "unavailable" if value is None else f"{value:.{digits}f}"


def _ci(value):
    return f"[{_number(value['lower'])}, {_number(value['upper'])}]"


def markdown_report(s):
    lines = ["## Material Passport", "", "- Type: bounded experimental result and evidence audit.",
        "- Verification: ANALYZED; raw native serving records rechecked when available.",
        f"- Training execution commit: `{s['training_execution_commit']}`.",
        f"- Evaluation/report execution commit: `{s['evaluation_execution_commit']}`.",
        f"- Run status: `{s['source_status'].get('status', 'unavailable')}`; decision: `{s['decision']['decision']}`.",
        "- Private prompt text, canonical/rival token IDs and tensors are excluded from this package.", "",
        "## Question and fixed design", "",
        "Does a margin-floor penalty on recent repairs improve one K+6→K+7 continuation? "
        "The auxiliary coefficient is 0.10; each frozen floor is 0.50*min(parent margin,1.0). "
        "NATIVE continues ordinary K+; RECENT protects prospective K5→K6 repair slots; CONTROL moves matched "
        "assignments to established slots and retains unmatched assignments. All start from the same complete "
        "K+6 state and use the same 128 feedback responses, two-pass order and anchors. No hyperparameter was tuned.",
        f"Sealed panel: {s['panel_requests']} requests; shortfall from 256: {s['panel_shortfall']}. "
        f"Recorded prior-exposure tiers: {s['selected_exposure_tiers']}. "
        "This is one seed and a shared-feedback fork, not independent online deployments.", "",
        "## Completion and validity", "",
        f"Raw serving audit: **{s['serving_audit']['status']}**; audit errors: {s['serving_audit']['error_count']}; "
        f"report reduction errors: {len(s['report_errors'])}.",
        "Budgets: at most 384 scientific Adam steps / 768 presentations, three disposable qualification "
        "Adam steps / six presentations, and at most four first-firing diagnostic Adam steps / eight presentations. "
        f"Timed serving budget: {s['planned']['timed_generations']} completions; fixed-start parent walks: "
        f"{s['planned']['fixed_start_parent_requests']}. Startup and canonical preparation are separate."]
    if s["continuation"]:
        c=s["continuation"]
        lines += ["", "Evaluation-only recovery reused the completed training, qualification and canonical "
            "preparation without repeating them. The earlier failed attempt remains preserved: "
            f"{_number(c['prior_fixed_start_gpu_s'],6)} occupied GPU seconds, one failed parent-walk "
            "wrapper, 51 startup draft blocks and 48 startup target calls; no parent generation began. "
            f"Recovery adds {c['new_optimizer_steps']} optimizer steps. Prior and new costs are identified "
            "separately; linked evidence is counted only once."]
    progress = s["serving_audit"]["progress"]
    if progress:
        lines += ["", "| Serving work | Count |", "|---|---:|"]
        for key in ("planned_generations", "attempted_generations", "returned_generations", "completed_valid_generations",
                    "invalid_generations", "unknown_generation_attempts", "unattempted_generations"):
            if key in progress: lines.append(f"| {key} | {progress[key]} |")
    if not s["complete_valid_comparison"]:
        lines += ["", "**No endpoint gain claim is made.** The run is incomplete, invalid, or stopped at the "
            "pretraining feasibility gate. Completed measurements and failure accounting remain in the tables; "
            "unattempted work is not imputed. Correctness/resource failure requires resolution before interpretation."]
        if s["report_errors"]:
            lines += ["", "Unmet report checks: " + ", ".join(s["report_errors"]) + "."]
    else:
        lines += ["", "## Prespecified serving comparison", "",
            "Accepted length is clipped accepted draft tokens divided by verification calls. Counts use one "
            "validated trajectory per request; serving time uses each request's two-sweep mean.", "",
            "| RECENT versus | Calls (RECENT / comparator) | Accepted-length gain % [95% CI] | Time reduction % [95% CI] | Call wins / losses / ties |",
            "|---|---:|---:|---:|---:|"]
        for other, r in s["comparisons"].items():
            lines.append(f"| {other} | {r['recent_calls']} / {r['reference_calls']} | "
                f"{_number(r['accepted_length_relative_gain_pct'])} {_ci(r['accepted_length_interval'])} | "
                f"{_number(r['time_reduction_pct'])} {_ci(r['time_reduction_interval'])} | "
                f"{r['requests_improved']} / {r['requests_regressed']} / {r['requests_tied']} |")
        lines += ["", "Intervals use 10,000 paired whole-request PCG64 draws, seed 1707, linear percentiles. "
            "They are conditional on these fixed trained checkpoints. Undefined bootstrap draws are recorded "
            "in SUMMARY.json. A timing benefit is demonstrated only when its lower confidence bound is positive.",
            "", "## Fixed-start mechanism evidence", "",
            "| Arm | Repairs | Exact-slot relapses | Earlier breaks | Accepts through repair | Blockers passed | Net prefix occurrences |",
            "|---|---:|---:|---:|---:|---:|---:|"]
        for arm, r in s["fixed_starts"]["totals"].items():
            lines.append(f"| {arm} | {r['repairs']} | {r['exact_slot_relapse']} | {r['earlier_break']} | "
                f"{r['accepts_through']} | {r['passes_parent_first_blocker']} / {r['blockers']} | {r['net_prefix_occurrences']} |")
        lines += ["", "Earlier breaks remain in the repair denominator. Fixed-start confidence intervals cluster "
            "the entire sealed request population, including requests with no eligible start. These local "
            "prediction changes cannot be converted into a fraction of time saved."]
    if s["sweep_totals"]:
        lines += ["", "| Arm / sweep | Request latency s | Native generation s | Cache maintenance s | Calls (repeated trajectory) |",
            "|---|---:|---:|---:|---:|"]
        for r in s["sweep_totals"]:
            lines.append(f"| {r['arm']} / {r['repeat']} | {_number(r['request_latency_s'],2)} | "
                f"{_number(r['native_generate_s'],2)} | {_number(r['cache_maintenance_s'],2)} | {r['verification_calls']} |")
        lines += ["", "Sweep call totals repeat the same validated requests; they are not additional statistical samples."]
    e = s["eligibility"]
    lines += ["", "## Eligibility, placement and actual treatment", "",
        f"Eligible distinct recent slots: {e.get('recent_distinct_slots', 'unavailable')}; occurrences: "
        f"{e.get('recent_occurrences', 'unavailable')}; matched slots: {e.get('matched_distinct_slots', 'unavailable')}; "
        f"matched requests: {e.get('matched_requests', 'unavailable')}. Switched weight fraction: "
        f"{_number(e.get('switched_weight_fraction'))}; absolute margin SMD: "
        f"{_number(e.get('absolute_standardized_margin_difference'))}; placement screen: "
        f"{e.get('placement_screen_passed', 'unavailable')}.",
        f"Frozen assigned/native selected weight: {_number(100*s['frozen_assigned_weight_fraction'] if s['frozen_assigned_weight_fraction'] is not None else None, 6)}%. "
        "Raw weight is exposure, not gradient influence. Final parameter differences and first-natural-firing "
        "ACTIVE/OFF gradient, moment and BF16 displacement receipts are separate evidence.", "",
        "| Arm | Presentations | Adam steps | Firing presentations / windows | Final coordinates differing from NATIVE |",
        "|---|---:|---:|---:|---:|"]
    for arm, t in s["training"].items():
        lines.append(f"| {arm} | {t.get('response_presentations', 0)} | {t.get('adam_steps', 0)} | "
            f"{t.get('firing_presentations', 0)} / {t.get('firing_windows', 0)} | "
            f"{t.get('final_vs_native', {}).get('different_coordinates', 'not measured')} |")
    if s["control_reused_recent"]:
        lines += ["", "CONTROL assignments were exactly identical to RECENT before training. CONTROL reuses "
            "that endpoint, with zero additional training; history-specific placement is unidentifiable."]
    if s["pass_exposure"]:
        lines += ["", "| Arm / pass | Assigned / reachable / firing occurrences | Firing wrong / still correct | Assigned / firing raw weight | Auxiliary loss sum |",
            "|---|---:|---:|---:|---:|"]
        for r in s["pass_exposure"]:
            lines.append(f"| {r['arm']} / {r['pass_index']} | "
                f"{r['retention_assigned_count']:g} / {r['retention_reachable_count']:g} / {r['retention_firing_count']:g} | "
                f"{r['retention_firing_incorrect_count']:g} / {r['retention_firing_correct_count']:g} | "
                f"{_number(r['retention_assigned_weight'])} / {_number(r['retention_firing_weight'])} | "
                f"{_number(r['retention_auxiliary_loss'],6)} |")
    lines += ["", "Activation/exposure sums use each presentation's original response observation exactly once; "
        "accumulation-boundary metrics are not counted a second time. `ordinary_selected_kl` is "
        "kplus_N_K/(kplus_S+1e-6); `retention_native_kl_loss` is the redistributed K+ objective, "
        "not that ordinary selected KL. `retention_auxiliary_loss` remains separately reported.", "",
        "Native numerical qualification: " + "; ".join(f"{case}={r.get('status', 'unavailable')}"
            for case, r in s["qualification"].items()) + "."]
    lines += ["", "## Costs and interpretation", "",
        "| Arm | Main training s | Checkpoint persistence s | Export s | Diagnostic state capture s | First-firing replay s |",
        "|---|---:|---:|---:|---:|---:|"]
    for arm, t in s["training"].items():
        costs = t.get("costs", {})
        lines.append(f"| {arm} | {_number(costs.get('training_s'), 2)} | "
            f"{_number(costs.get('checkpoint_save_s'), 2)} | {_number(costs.get('export_s'), 2)} | "
            f"{_number(t.get('diagnostic_capture_s'), 2)} | {_number(t.get('first_firing', {}).get('wall_s'), 2)} |")
    lines += ["",
        "costs.csv separates frozen preparation, disposable qualification, startup, main training, first-firing "
        "replays, diagnostic state capture, persistence/export, fixed-start work, canonical preparation, loading, "
        "activation/cache maintenance and timed serving. Parent/child timers overlap and must not be added as "
        "independent costs; compiler timers may also nest. Training includes the realized objective workload, "
        "while diagnostic replays and snapshot capture are recorded separately. The bounded first-firing "
        "preclip-gradient CPU copy remains included in training_s; other reversible state captures are timed "
        "separately. These are measured process costs, not exact intrinsic-kernel costs. No kernel-time "
        "breakdown is inferred.",
        f"Summed execution wall time: {_number(s['combined_campaign_wall_s'], 2)} s. "
        "For a recovery this adds prior and new execution walls, excluding repair/user-wait gaps; "
        "it is not calendar elapsed time. This does not establish training-inclusive speedup.", "",
        "Feedback, teacher packets and historical K5/K6 label trajectories were reused. The present clocks "
        "do not newly measure their collection, teacher preparation or label-replay overhead. Future events "
        "require that work; it is not free. Full online deployment cost remains unestablished.", "",
        f"Recorded allocated phase windows: {_number(s['resources']['measured_allocated_phase_window_card_hours'])} "
        f"card-hours on {s['resources']['physical_gpu'] or 'unrecorded GPU identity'}; "
        f"peak allocated / reserved bytes: {s['resources']['peak_allocated_bytes']} / "
        f"{s['resources']['peak_reserved_bytes']}. " + s['resources']['phase_window_scope'], "",
        f"**Decision: {s['decision']['decision']}.** "
        "Advance requires the prespecified native accepted-length gain and interval, an identifiable placement "
        "advantage, fewer exact held-out repair relapses and more accept-through decisions versus both controls, "
        "and no demonstrated slowdown. Lower protected loss alone is insufficient. An unresolved interval is "
        "not equivalence. No tuning or additional experiment follows automatically; reopening requires a "
        "separately authorized study addressing the failed or unresolved link.", "",
        "Files: SUMMARY.json, per_request.csv, all_repeats.csv, fixed_start_origins.csv, fixed_start_outcomes.csv, "
        "eligibility.csv, matching.csv, assigned_slots.csv, activation.csv, gradient_receipts.csv, "
        "endpoint_differences.csv, costs.csv and REPORT_SELF_CHECKS.json.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = generate_report(args.root, args.output)
    print(json.dumps(dict(decision=result["decision"]["decision"], complete=result["complete_valid_comparison"])))
