"""Fresh, paired, serving-only baseline comparisons with explicit validity gates."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from .experiment import (CHECKPOINTS, STEPS, METRIC_DEFINITIONS, digest, measurement_validation,
                         preserve_invalid_numbers, validate_training)

SCOPE = ("Small fixed-feedback baseline diagnostic, not an AdaFlash reproduction or broad-generalization claim. "
         "Each checkpoint/request has one fresh timed execution. Latencies are preliminary. Training, source-loss "
         "diagnostics, warmup and checkpoint loading are excluded from request latency and reported separately. "
         "Serving-only reductions do not establish net online speedup.")


def _number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _ratio(numerator, denominator, scale=1):
    return scale * numerator / denominator if numerator is not None and denominator else None


def _reduction(candidate, control):
    return 100 * (1 - candidate / control) if candidate is not None and control else None


def totals(rows):
    rows = list(rows)
    def add(key):
        return sum(row[key] for row in rows) if rows and all(_number(row.get(key)) for row in rows) else None
    calls, tokens = add("verification_calls"), add("output_tokens")
    accepted, committed = add("accepted_draft_tokens"), add("verification_committed_tokens")
    return dict(requests=len(rows), request_s=add("request_s"), output_tokens=tokens, verification_calls=calls,
                accepted_draft_tokens=accepted, verification_committed_tokens=committed,
                calls_per_1000_output_tokens=_ratio(calls, tokens, 1000),
                accepted_draft_tokens_per_call=_ratio(accepted, calls),
                committed_progress_per_call=_ratio(committed, calls),
                emitted_tokens_per_verification_call=_ratio(tokens, calls))


def _invalid(result, row, frozen, expected):
    validation = measurement_validation(row)
    problems = ["measurement:" + message for message in validation["errors"]]
    if row.get("measurement_validation") != {"status": "passed", "errors": []}:
        problems.append("recorded_measurement_validation_failed_or_missing")
    if row.get("role") != "evaluation" or row.get("training_data") is not False or row.get("timing_repeats") != 1:
        problems.append("not_single_untrained_heldout_execution")
    guard = row.get("training_quiescence", {})
    if guard.get("status") != "passed" or guard.get("training_quiescent") is not True or guard.get("trainer_processes") != []:
        problems.append("background_training_not_excluded")
    if expected is None or row.get("input_ids") != expected.get("input_ids"):
        problems.append("manifest_input_mismatch")
    if expected is not None and row.get("reference_input_ids_sha256") != digest(expected["input_ids"]):
        problems.append("manifest_input_identity_mismatch")
    if frozen is None or row.get("output_ids") != frozen.get("output_ids"):
        problems.append("frozen_output_mismatch_or_missing")
    if frozen is not None and row.get("reference_output_ids_sha256") != digest(frozen.get("output_ids")):
        problems.append("frozen_output_identity_mismatch")
    bound = result.get("checkpoints", {}).get(row.get("checkpoint_id"), {})
    if row.get("checkpoint_identity_sha256") != bound.get("checkpoint_identity_sha256") or not bound:
        problems.append("checkpoint_identity_mismatch")
    for key in ("same_generation_valid", "input_match", "canonical_output_match", "checkpoint_identity_match"):
        if row.get(key) is not True:
            problems.append(key)
    return sorted(set(problems))


def comparison(result, request_id, candidate, control):
    selected = [[row for row in result.get("scores", []) if row.get("request_id") == request_id and
                 row.get("checkpoint_id") == checkpoint] for checkpoint in (candidate, control)]
    frozen_rows = [row for row in result.get("scores", []) if row.get("request_id") == request_id and row.get("checkpoint_id") == "F"]
    frozen = frozen_rows[0] if len(frozen_rows) == 1 else None
    expected = next((row for row in result.get("evaluation_requests", []) if row["request_id"] == request_id), None)
    errors = []
    if result.get("training_valid") is not True:
        errors.append("training:recorded_qualification_missing_or_failed")
    try:
        validate_training(result.get("training") or {}, result.get("support", []))
    except (ValueError, TypeError, KeyError, OverflowError) as error:
        errors.append("training:" + str(error))
    for checkpoint, rows in zip((candidate, control), selected):
        if len(rows) != 1:
            errors.append(checkpoint + ":missing_or_duplicate_measurement")
        for row in rows:
            errors.extend(checkpoint + ":" + error for error in _invalid(result, row, frozen, expected))
    # Even S8/S2 needs a valid independently measured frozen canonical reference.
    if frozen is None:
        errors.append("F:missing_or_duplicate_reference")
    else:
        errors.extend("F:" + error for error in _invalid(result, frozen, frozen, expected))
    valid = not errors
    new, old = [totals(rows) for rows in selected]
    saved = old["verification_calls"] - new["verification_calls"] if valid else None
    row = dict(group_id=result["group_id"], domain=result["domain"], request_id=request_id,
               candidate=candidate, control=control, candidate_step=STEPS[candidate], control_step=STEPS[control],
               valid=valid, invalid_reasons=sorted(set(errors)), candidate_metrics=new, control_metrics=old,
               verification_calls_saved=saved,
               request_time_saved_s=old["request_s"] - new["request_s"] if valid else None,
               verification_call_reduction_percent=_reduction(new["verification_calls"], old["verification_calls"]) if valid else None,
               request_time_reduction_percent=_reduction(new["request_s"], old["request_s"]) if valid else None,
               drafting_result=("improvement" if saved > 0 else "regression" if saved < 0 else "tie") if valid else "invalid")
    return row, selected


def _pool(pairs):
    usable = [(row, groups) for row, groups in pairs if row["valid"]]
    new = totals(row for _, groups in usable for row in groups[0])
    old = totals(row for _, groups in usable for row in groups[1])
    return dict(expected_requests=len(pairs), valid_requests=len(usable), invalid_requests=len(pairs) - len(usable),
                improved_requests=sum(row["drafting_result"] == "improvement" for row, _ in usable),
                regressed_requests=sum(row["drafting_result"] == "regression" for row, _ in usable),
                tied_requests=sum(row["drafting_result"] == "tie" for row, _ in usable),
                candidate_metrics=new, control_metrics=old,
                verification_calls_saved=old["verification_calls"] - new["verification_calls"] if usable else None,
                request_time_saved_s=old["request_s"] - new["request_s"] if usable else None,
                verification_call_reduction_percent=_reduction(new["verification_calls"], old["verification_calls"]),
                request_time_reduction_percent=_reduction(new["request_s"], old["request_s"]))


def _costs(result):
    training = result.get("training") or {}
    declared = training.get("costs", {})
    publications = result.get("publications", [])
    feedback, scores, warmups = (result.get(key, []) for key in ("feedback", "scores", "warmups"))
    def total(rows, key):
        return sum(row[key] for row in rows) if rows and all(_number(row.get(key)) for row in rows) else None
    return dict(group_id=result["group_id"], domain=result["domain"], status=result["status"],
        feedback_collection_request_s=total(feedback, "request_s"),
        evaluation_request_s=total(scores, "request_s"), warmup_request_s=total(warmups, "request_s"),
        support_prepare_wall_s=result.get("support_prepare_wall_s"),
        capture_s=0.0 if all(r.get("capture") is False for r in feedback + scores) else None,
        teacher_preparation_s=declared.get("teacher_preparation_s"), training_s=declared.get("training_s"),
        source_loss_diagnostic_s=declared.get("source_loss_diagnostic_s"), export_s=declared.get("export_s"),
        optimizer_state_save_s=declared.get("optimizer_state_save_s"),
        checkpoint_transfer_s=declared.get("checkpoint_transfer_s"),
        checkpoint_loading_s=total(publications, "activation_s"),
        native_checkpoint_activation_s=total(publications, "native_activation_s"),
        cache_maintenance_s=total(feedback + scores + warmups, "cache_maintenance_s"),
        training_call_wall_s=result.get("training_call_wall_s"),
        trainer_startup_s=training.get("trainer_startup_s"), startup=result.get("startup"),
        peak_allocated_gpu_bytes=declared.get("peak_allocated_gpu_bytes"),
        peak_reserved_gpu_bytes=declared.get("peak_reserved_gpu_bytes"), actual_group_wall_s=result.get("actual_wall_s"),
        accounting="Teacher preparation is nested in training; native activation is nested in loading. Wall times overlap components; do not sum blindly. No net-online-speedup claim.")


def summarize(results):
    campaign = results if isinstance(results, dict) and "groups" in results else None
    groups = list(campaign["groups"] if campaign is not None else results)
    pairs = []
    for result in groups:
        for request in result.get("evaluation_requests", []):
            for candidate, control in (("S2", "F"), ("S8", "F"), ("S8", "S2")):
                pairs.append(comparison(result, request["request_id"], candidate, control))
    pooled = []
    scopes = [("overall", "all", groups)]
    scopes += [("domain", domain, [g for g in groups if g["domain"] == domain]) for domain in sorted({g["domain"] for g in groups})]
    scopes += [("group", group["group_id"], [group]) for group in groups]
    for scope, name, selected_groups in scopes:
        ids = {group["group_id"] for group in selected_groups}
        for candidate, control in (("S2", "F"), ("S8", "F"), ("S8", "S2")):
            selected = [(row, rows) for row, rows in pairs if row["group_id"] in ids and row["candidate"] == candidate and row["control"] == control]
            pooled.append(dict(scope=scope, scope_id=name, candidate=candidate, control=control, **_pool(selected)))
    per_request, per_round, source_losses, exposure = [], [], [], []
    for result in groups:
        for original in result.get("feedback", []) + result.get("scores", []):
            row = {key: value for key, value in original.items() if key not in ("input_ids", "output_ids", "rounds", "text")}
            row["measurement_validation_recomputed"] = measurement_validation(original)
            row["context_tokens"] = len(original.get("input_ids", []))
            row["output_ids_sha256"] = digest(original.get("output_ids"))
            per_request.append(row)
            for event in original.get("rounds", []):
                per_round.append(dict(group_id=result["group_id"], domain=result["domain"],
                    request_id=original["request_id"], role=original["role"], checkpoint_id=original["checkpoint_id"], **event))
        training = result.get("training") or {}
        for source in training.get("source_losses", []):
            source_losses.append(dict(group_id=result["group_id"], domain=result["domain"], **source))
        for batch in training.get("accounting", {}).get("batches", []):
            for row in batch.get("presentations", []):
                exposure.append(dict(group_id=result["group_id"], domain=result["domain"],
                    optimizer_step=batch.get("step", batch.get("optimizer_steps_after")),
                    pass_index=batch.get("pass_index"), **row))
    status = ("not_executed" if not groups else "complete" if len(groups) == 6 and
              all(g.get("implementation_checks", {}).get("status") == "passed" for g in groups) else "partial_or_failed")
    return dict(schema_version=1, status=status, scope=SCOPE, metric_definitions=METRIC_DEFINITIONS,
                groups=[dict(group_id=g["group_id"], domain=g["domain"], status=g["status"],
                             error=g.get("error"), implementation_checks=g.get("implementation_checks")) for g in groups],
                expected_groups=6, completed_groups=sum(g.get("status") == "complete" for g in groups),
                unexecuted_groups=campaign.get("unexecuted_groups", []) if campaign else None,
                comparisons=[row for row, _ in pairs], pooled=pooled, per_request=per_request, per_round=per_round,
                source_losses=source_losses, supervised_exposure=exposure, costs=[_costs(g) for g in groups],
                training_accounting=[dict(group_id=g["group_id"], training=g.get("training")) for g in groups],
                actual_campaign_wall_s=campaign.get("actual_wall_s") if campaign else None,
                training_accounting_incomplete_groups=[g["group_id"] for g in groups
                    if g.get("training_started") and not g.get("training")],
                core_generations=sum(len(g.get("feedback", [])) + len(g.get("scores", [])) for g in groups),
                explicit_warmups=sum(len(g.get("warmups", [])) for g in groups),
                optimizer_steps=sum((g.get("training") or {}).get("accounting", {}).get("optimizer_steps", 0) for g in groups),
                training_response_presentations=len(exposure),
                zero_supervision_presentations=sum(row.get("zero_supervision") is True and
                    row.get("eligible_positions") == 0 for row in exposure))


def _write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True, allow_nan=False) if isinstance(value, (dict, list, tuple)) else value
                             for key, value in row.items()})


def _fmt(value, suffix=""):
    return "unavailable" if value is None else f"{value:+.3f}{suffix}"


def write_reports(results, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    summary = preserve_invalid_numbers(summarize(results))
    (out / "SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    for name, key in (("PER_REQUEST.csv", "per_request"), ("PER_ROUND.csv", "per_round"),
                      ("COMPARISONS.csv", "comparisons"), ("POOLED.csv", "pooled"),
                      ("SOURCE_LOSSES.csv", "source_losses"), ("SUPERVISED_EXPOSURE.csv", "supervised_exposure"), ("COSTS.csv", "costs")):
        _write_csv(out / name, summary[key])
    (out / "TRAINING_ACCOUNTING.json").write_text(json.dumps(summary["training_accounting"], indent=2, allow_nan=False) + "\n")
    lines = ["# Fixed-feedback DFlash baseline learning curve", "", f"Status: **{summary['status']}**.", "", SCOPE, "",
             f"Completed groups: {summary['completed_groups']}/6. Core generations: {summary['core_generations']}/96; "
             f"documented optimizer steps: {summary['optimizer_steps']}/48; documented training presentations: {summary['training_response_presentations']}/96. "
             f"Explicit warmup generations: {summary['explicit_warmups']} (separate).", ""]
    if summary["status"] == "not_executed":
        lines += ["No native measurements have been executed. Implementation tests are not evidence of source fitting or serving improvement.", ""]
    else:
        if summary['training_accounting_incomplete_groups']:
            lines += ["Training failed before a complete controller receipt for: " +
                      ", ".join(summary['training_accounting_incomplete_groups']) +
                      ". Their actual partial steps are not imputed as zero; inspect retained worker step/failure records.", ""]
        lines += ["## Held-out serving comparisons", "", "Positive reduction means less time/work. Ratios use summed paired measurements, not averages of per-request percentages.", "",
                  "| Scope | Checkpoints | Valid requests | Request-time reduction | Verification-call reduction | Harmful / unchanged |",
                  "|---|---|---:|---:|---:|---:|"]
        for row in summary["pooled"]:
            lines.append(f"| {row['scope']}:{row['scope_id']} | {row['candidate']} vs {row['control']} | {row['valid_requests']}/{row['expected_requests']} | "
                         f"{_fmt(row['request_time_reduction_percent'], '%')} | {_fmt(row['verification_call_reduction_percent'], '%')} | "
                         f"{row['regressed_requests']} / {row['tied_requests']} |")
        lines += ["", "## Source fitting and training exposure", "", "Source losses use a separate fixed anchor set, with training RNG restored. Lower source loss is an in-sample result and does not establish held-out transfer.", "",
                  "| Group | F source loss | S2 source loss | S8 source loss | S2−F | S8−S2 |", "|---|---:|---:|---:|---:|---:|"]
        for group in summary["groups"]:
            losses = {row["checkpoint_id"]: row.get("loss") for row in summary["source_losses"] if row["group_id"] == group["group_id"]}
            values = [losses.get(key) for key in CHECKPOINTS]
            d2 = values[1] - values[0] if all(_number(v) for v in values[:2]) else None
            d8 = values[2] - values[1] if all(_number(v) for v in values[1:]) else None
            lines.append(f"| {group['group_id']} | {_fmt(values[0])} | {_fmt(values[1])} | {_fmt(values[2])} | {_fmt(d2)} | {_fmt(d8)} |")
        lines += ["", f"Zero-supervision presentations: {summary['zero_supervision_presentations']}. These retain the qualified sampled anchors and scheduled optimizer steps; actual exposure remains explicit in SUPERVISED_EXPOSURE.csv.", "",
                  "## Cost and interpretation", "",
                  "Teacher preparation, training, fixed-anchor source diagnostics, checkpoint export/persistence, loading, warmups, cache maintenance and observed wall time remain separate in COSTS.csv. Teacher preparation is already inside training; native activation is already inside loading. No net online speedup is computed from these overlapping timers.", "",
                  "S2/F and S8/F measure transfer from the four frozen-generated feedback responses. S8/S2 measures extra training on that same pool and workload. Source improvement with absent or harmful held-out gains is retained. Fewer verification calls with unchanged or worse latency is drafting improvement without established serving acceleration.", "",
                  "Null or harmful results do not fail qualification. Output mismatches, invalid counts/nonfinite values, wrong bindings or concurrent training do. Invalid comparisons remain in the tables and cannot supply gain claims."]
        failures = [g for g in summary["groups"] if g["status"] != "complete"]
        if failures:
            lines += ["", "## Incomplete groups", ""] + [f"- {g['group_id']}: {g.get('error') or g['status']}" for g in failures]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    return summary
