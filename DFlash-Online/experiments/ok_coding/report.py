"""Reconstruct the fixed coding comparison from raw native records and work receipts."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from online_feedback.storage import write_json
from experiments.fresh_confirmation.protocol import (
    COUNT_FIELDS, measurement_errors, record_errors, repeat_errors,
    publication_errors, weight_check_errors,
)
from experiments.osd_persistent.run import score_record
from .contract import ARMS, CELLS, METHODS, SEEDS, evaluation_schedule, validate_plan

TIMINGS = ("request_latency_s", "native_generate_s", "cache_maintenance_s")
COMPARISONS = (("K", "O"), ("K", "F"), ("O", "F"))


def read_json(path):
    return json.loads(Path(path).read_text())


def local_artifact(root, recorded):
    """Resolve a retained raw path after compact retrieval without changing its identity."""
    path = Path(recorded)
    if path.exists():
        return path
    marker = "/campaign_001/"
    if marker not in str(path):
        raise FileNotFoundError(recorded)
    return Path(root) / "campaign_001" / str(path).split(marker, 1)[1]


def audit_cell(root, plan, seed, event):
    """Independently validate raw outputs/counts and reduce repeats by request."""
    root = Path(root)
    folder = root / "campaign_001" / f"seed_{seed}" / f"event_{event:03d}" / "evaluation"
    requests = plan["evaluation_requests"]
    errors, rows = [], []
    try:
        protocol = read_json(folder / "PROTOCOL.json")
        status = read_json(folder / "STATUS.json")
        canonical = read_json(root / "campaign_001" / "EVALUATION_CANONICAL.json")
        rows = [json.loads(line) for line in (folder / "MEASUREMENTS.jsonl").read_text().splitlines()]
        expected_schedule = evaluation_schedule(requests, seed=seed, event=event)
        schedule_keys = ("seed", "event", "request_id", "panel_ordinal", "repeat", "method", "position")
        if (protocol["schedule"] != expected_schedule
                or [{key: row.get(key) for key in schedule_keys} for row in rows] != expected_schedule):
            errors.append("schedule_or_chronological_rows")
        expected_count = len(requests) * 9
        for key, expected in {"status": "completed", "qualification": "passed",
                "attempted_generations": expected_count, "returned_generations": expected_count,
                "completed_valid_generations": expected_count, "invalid_generations": 0,
                "pending_canonical_generations": 0, "unknown_generation_attempts": 0,
                "unattempted_generations": 0}.items():
            if status.get(key) != expected:
                errors.append("status_" + key)
        gate = read_json(folder / "QUALIFICATION.json")
        if gate != {"status": "passed", "requests": 8, "generations": 72,
                    "methods_with_native_equality": ["F", "K", "O"],
                    "performance_win_required": False, "extra_generations": 0}:
            errors.append("included_gate")
        checkpoints = {"F": plan["draft"]}
        for arm in ARMS:
            endpoint = read_json(root / "campaign_001" / f"seed_{seed}" / arm / "events"
                                 / f"event_{event:03d}" / "RESULT.json")
            checkpoints[arm] = endpoint["export_path"]
        if protocol["checkpoints"] != checkpoints:
            errors.append("checkpoint_binding")
        for method in METHODS:
            equality = read_json(folder / ("WEIGHTS_" + method + ".json"))
            if equality["errors"] or weight_check_errors(equality["native_reply"], checkpoints[method]):
                errors.append("weights_" + method)
        for line in (folder / "ACTIVATIONS.jsonl").read_text().splitlines():
            activation = json.loads(line)
            if activation["errors"] or publication_errors(activation, checkpoints[activation["method"]]):
                errors.append("publication")
        if set(canonical) != {request["request_id"] for request in requests}:
            errors.append("canonical_panel")
        for request in requests:
            rid = request["request_id"]
            reference = canonical[rid]
            ordinal = request["panel_ordinal"]
            expected_source = (root / "campaign_001" / "seed_17" / "event_002" / "evaluation"
                               / f"{ordinal:04d}_r0_F.json")
            if (reference["source_method"] != "F" or reference["source_repeat"] != 0
                    or reference["source_cell"] != [17, 2]
                    or reference["F_checkpoint"] != plan["draft"]
                    or local_artifact(root, reference["source_record_path"]).resolve() != expected_source.resolve()):
                errors.append(rid + ":canonical_source")
            source = read_json(expected_source)
            if (source["response_ids"] != reference["response_ids"]
                    or source["finish_reason"] != reference["finish_reason"]):
                errors.append(rid + ":canonical_source_output")
            source_score = score_record(source, request_id=rid, repeat=0, method="F")
            if any(source_score[name] != reference["F_counts"][name] for name in COUNT_FIELDS):
                errors.append(rid + ":canonical_F_counts")
            selected = [row for row in rows if row["request_id"] == rid]
            for row in selected:
                path = folder / (row["label"] + ".json")
                if local_artifact(root, row["record_path"]).resolve() != path.resolve():
                    errors.append(row["label"] + ":raw_path")
                record = read_json(path)
                invalid = record_errors(record, request, checkpoints[row["method"]], reference)
                if (record["record_id"] != row["label"] or record["request_index"] != ordinal
                        or record["session_id"] != "ok_coding_evaluation"):
                    invalid.append("native_identity")
                recalculated = score_record(record, request_id=rid, repeat=row["repeat"], method=row["method"])
                recalculated.update(finish_reason=record["finish_reason"],
                                    cache_maintenance_s=record["metadata"]["capture"]["cache_maintenance_s"])
                if any(row.get(key) != value for key, value in recalculated.items()):
                    invalid.append("raw_measurement_disagreement")
                invalid.extend(measurement_errors(row, request, reference))
                invalid.extend(repeat_errors(selected, row["method"]))
                if row["method"] == "F" and any(row[name] != reference["F_counts"][name] for name in COUNT_FIELDS):
                    invalid.append("global_F_counts")
                if row.get("valid_measurement") is not True or row.get("errors"):
                    invalid.append("measurement_not_valid")
                errors.extend(row["label"] + ":" + error for error in sorted(set(invalid)))
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        errors.append("missing_or_malformed_evidence: " + repr(error))
    if errors:
        return [], errors
    return reduce_requests(rows, requests), []


def reduce_requests(rows, requests):
    reduced = []
    for request in requests:
        value = {"request_id": request["request_id"], "panel_ordinal": request["panel_ordinal"]}
        for method in METHODS:
            selected = [row for row in rows if row["request_id"] == request["request_id"] and row["method"] == method]
            selected.sort(key=lambda row: row["repeat"])
            if [row["repeat"] for row in selected] != [0, 1, 2] or repeat_errors(selected, method):
                raise ValueError("Three identical-count timing repeats are required")
            value[method] = {name: selected[0][name] for name in COUNT_FIELDS}
            value[method].update({name: sum(row[name] for row in selected) / 3 for name in TIMINGS})
            value[method]["cap_hit"] = selected[0]["cap_hit"]
            value[method]["finish_reason"] = selected[0]["finish_reason"]
        reduced.append(value)
    return reduced


def bootstrap_indices(request_count=128):
    generator = np.random.Generator(np.random.PCG64(20261030))
    return generator.integers(0, request_count, size=(10000, request_count))


def summarize_cell(rows, indices):
    summary = {"requests": len(rows), "methods": {}, "comparisons": {}}
    for method in METHODS:
        totals = {name: sum(row[method][name] for row in rows) for name in (*COUNT_FIELDS, *TIMINGS)}
        totals["calls_per_1000_output_tokens"] = 1000 * totals["verification_calls"] / totals["output_tokens"]
        totals["committed_progress_per_call"] = totals["committed_progress"] / totals["verification_calls"]
        totals["output_cap_requests"] = sum(row[method]["cap_hit"] for row in rows)
        totals["output_cap_frequency"] = totals["output_cap_requests"] / len(rows)
        summary["methods"][method] = totals
    for candidate, baseline in COMPARISONS:
        base = np.asarray([row[baseline]["verification_calls"] for row in rows], dtype=np.float64)
        treatment = np.asarray([row[candidate]["verification_calls"] for row in rows], dtype=np.float64)
        difference = base - treatment
        sampled_base = base[indices].sum(axis=1)
        sampled_treatment = treatment[indices].sum(axis=1)
        bootstrapped = 100 * (sampled_base - sampled_treatment) / sampled_base
        result = {"call_reduction_pct": float(100 * (base.sum() - treatment.sum()) / base.sum()),
                  "bootstrap_95pct": np.quantile(bootstrapped, [0.025, 0.975], method="linear").tolist(),
                  "wins": int((difference > 0).sum()), "losses": int((difference < 0).sum()),
                  "ties": int((difference == 0).sum()),
                  "gross_saved_calls": int(difference[difference > 0].sum()),
                  "gross_added_calls": int(-difference[difference < 0].sum())}
        for metric in TIMINGS:
            baseline_time = summary["methods"][baseline][metric]
            candidate_time = summary["methods"][candidate][metric]
            result[metric + "_reduction_pct"] = 100 * (baseline_time - candidate_time) / baseline_time if baseline_time else None
        summary["comparisons"][candidate + "_vs_" + baseline] = result
    return summary


def audit_training(root, plan):
    from .campaign import TRAINING_LIMITS, read_work, reconcile_work, require_complete_campaign, require_complete_work
    from .training import validate_event_receipt

    output = root / "campaign_001"
    errors, records, pids, cache_paths, devices = [], [], [], [], []
    try:
        reported = read_json(output / "STATUS.json")
        reconstructed = {"status": reported["status"]}
        reconcile_work(output, reconstructed)
        require_complete_campaign(reconstructed)
        if reported["status"] != "completed":
            errors.append("campaign_not_completed")
        for key in reconstructed:
            if reported.get(key) != reconstructed[key]:
                errors.append("campaign_counter_" + key)
        for key in ("extra_generations", "optimizer_previews", "qualification_model_work", "extra_optimizer_steps"):
            if reported.get(key) != 0:
                errors.append("extra_work_" + key)
        for seed in SEEDS:
            seed_root = output / f"seed_{seed}"
            for event in range(1, 5):
                ordinary = read_json(seed_root / "O" / "events" / f"event_{event:03d}" / "RESULT.json")
                for arm in ARMS:
                    folder = seed_root / arm / "events" / f"event_{event:03d}"
                    result = read_json(folder / "RESULT.json")
                    validate_event_receipt(result, event_number=event,
                        objective="osd" if arm == "O" else "osd_first_error",
                        reference_schedule=ordinary if arm == "K" else None)
                    audit = seed_root / f"resume_{event:03d}_{arm}"
                    worker = read_json(audit / "WORKER.json")
                    require_complete_work(read_work(audit / "WORK.jsonl", TRAINING_LIMITS), TRAINING_LIMITS)
                    if worker["status"] != "completed":
                        errors.append(f"{seed}/{event}/{arm}:worker")
                    startup = worker["startup"]
                    pids.append(startup["pid"])
                    devices.append((startup["cuda_visible_devices"], startup["device_uuid"]))
                    if startup["cuda_initialized_on_entry"] or startup["warmup_model_work"] != 0:
                        errors.append(f"{seed}/{event}/{arm}:startup")
                    for cache in startup["caches"].values():
                        cache_paths.append(cache["path"])
                        if cache["initially_empty"] is not True:
                            errors.append("nonempty_startup_cache")
                    if event == 1:
                        initial = read_json(audit / "INITIAL.json")
                        if (initial["status"] != "passed" or initial["global_step"] != 0
                                or not initial["released_weights_exact"]):
                            errors.append("initial_state")
                        if arm == "K" and not read_json(audit / "INITIAL_PARITY.json")["exact_initial_state"]:
                            errors.append("initial_parity")
                    elif not read_json(audit / "LIVE.json")["exact_resume"]:
                        errors.append("live_resume")
                    if not read_json(audit / "LIVE_BOUNDARY.json")["exact_boundary"]:
                        errors.append("saved_boundary")
                    records.append({"seed": seed, "event": event, "arm": arm,
                        "costs": read_json(audit / "COSTS.json"),
                        "selection_totals": result.get("selection_totals"),
                        "response_records": result["records"]})
        if len(pids) != 16 or len(set(pids)) != 16 or len(set(devices)) != 1:
            errors.append("fresh_process_or_same_updater")
        if len(cache_paths) != 96 or len(set(cache_paths)) != 96:
            errors.append("separate_empty_compiler_caches")
    except (OSError, ValueError, KeyError, TypeError) as error:
        errors.append("training_evidence: " + repr(error))
    return records, errors


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_campaign_report(root, output, *, execution_commit=None):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plan = read_json(root / "sealed" / "PLAN.json")
    errors = []
    try:
        validate_plan(plan)
    except (ValueError, KeyError, TypeError) as error:
        errors.append("plan: " + repr(error))
    if execution_commit is not None and execution_commit != plan["execution_source"]["commit"]:
        errors.append("execution_source_binding")
    training, training_errors = audit_training(root, plan)
    errors.extend(training_errors)
    summaries, reduced = {}, {}
    indices = bootstrap_indices(len(plan["evaluation_requests"]))
    for seed, event in CELLS:
        rows, cell_errors = audit_cell(root, plan, seed, event)
        key = f"{seed}/{event}"
        errors.extend(key + ":" + error for error in cell_errors)
        if not cell_errors:
            reduced[key] = rows
            summaries[key] = summarize_cell(rows, indices)
    summary = {"status": "valid" if not errors else "invalid_or_incomplete", "errors": errors,
               "execution_source": plan["execution_source"], "cells": summaries,
               "cell_scope": "Raw-valid completed cells; overall interpretation requires complete campaign validity",
               "bootstrap": {"resamples": 10000, "generator": "PCG64", "seed": 20261030,
                   "percentile_method": "linear", "shared_request_draws_across_seeds": True,
                   "scope": "Conditional on the two fitted trajectories and selected128-request panel"}}
    if not errors:
        final = [summaries[f"{seed}/4"]["comparisons"] for seed in SEEDS]
        if all(row["K_vs_O"]["call_reduction_pct"] > 0 for row in final):
            if all(row["K_vs_F"]["call_reduction_pct"] > 0 and
                   row["K_vs_O"]["request_latency_s_reduction_pct"] > 0 for row in final):
                outcome = "Promising transfer at this fixed early-adaptation budget; inspect costs and interval uncertainty"
            elif all(row["K_vs_F"]["call_reduction_pct"] <= 0 for row in final):
                outcome = "Relative advantage only"
            else:
                outcome = "Mixed/seed-sensitive"
        elif any(row["K_vs_O"]["call_reduction_pct"] > 0 for row in final):
            outcome = "Mixed/seed-sensitive"
        else:
            outcome = "Unsupported at this budget"
        summary["outcome"] = outcome
    else:
        summary["outcome"] = "Invalid/incomplete; no efficacy interpretation"
    write_json(output / "SUMMARY.json", summary)
    write_json(output / "TRAINING_AUDIT.json", {"errors": training_errors, "events": training})
    flat = []
    for key, rows in reduced.items():
        for row in rows:
            flat.append({"seed": int(key.split("/")[0]), "event": int(key.split("/")[1]),
                "request_id": row["request_id"], "panel_ordinal": row["panel_ordinal"],
                **{method + "_" + field: value for method in METHODS for field, value in row[method].items()}})
    write_csv(output / "PAIRED_REQUESTS.csv", flat)
    all_repeats, repeat_totals = [], []
    for seed, event in CELLS:
        measurements = root / "campaign_001" / f"seed_{seed}" / f"event_{event:03d}" / "evaluation" / "MEASUREMENTS.jsonl"
        if not measurements.exists():
            continue
        raw_rows = [json.loads(line) for line in measurements.read_text().splitlines()]
        all_repeats.extend({key: json.dumps(value) if isinstance(value, (dict, list)) else value
                            for key, value in row.items()} for row in raw_rows)
        for method in METHODS:
            for repeat in range(3):
                selected = [row for row in raw_rows if row["method"] == method and row["repeat"] == repeat]
                valid = selected if f"{seed}/{event}" in reduced else []
                repeat_totals.append({"seed": seed, "event": event, "method": method, "repeat": repeat,
                    "returned_rows": len(selected), "valid_rows": len(valid),
                    "summary_scope": "Only independently valid complete cells; raw rows retain their original flags",
                    **{name: sum(row[name] for row in valid) for name in (*COUNT_FIELDS, *TIMINGS)}})
    write_csv(output / "ALL_REPEATS.csv", all_repeats)
    write_csv(output / "REPEAT_TOTALS.csv", repeat_totals)
    write_csv(output / "TRAINING_COSTS.csv", [
        {"seed": row["seed"], "event": row["event"], "arm": row["arm"],
         **{key: value for key, value in row["costs"].items() if not isinstance(value, (dict, list))}}
        for row in training])
    lines = ["# Fixed two-seed O/K coding confirmation", "",
             "Execution source: `" + plan["execution_source"]["commit"] + "`.", "",
             summary["outcome"] + ".", ""]
    if not errors:
        lines += ["| Seed | Event | F calls | O calls | K calls | K/O reduction | Paired95% interval | K/F reduction | K/O request-time reduction |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for seed, event in CELLS:
            cell = summaries[f"{seed}/{event}"]
            pair = cell["comparisons"]["K_vs_O"]
            interval = pair["bootstrap_95pct"]
            lines.append(f"| {seed} | {event} | {cell['methods']['F']['verification_calls']} | "
                f"{cell['methods']['O']['verification_calls']} | {cell['methods']['K']['verification_calls']} | "
                f"{pair['call_reduction_pct']:.4f}% | [{interval[0]:.4f}, {interval[1]:.4f}]% | "
                f"{cell['comparisons']['K_vs_F']['call_reduction_pct']:.4f}% | "
                f"{pair['request_latency_s_reduction_pct']:.4f}% |")
        lines += ["", "Primary comparisons are event4 K/O separately for each seed. Event2 is secondary. "
                  "Counts use one exact trajectory; timing uses each request's three-repeat mean. "
                  "F is remeasured contemporaneously in every cell with globally identical outputs/counts.", "",
                  "All1,024 feedback generations,1,024 terminal preparation forwards,2,048 native Adam updates, "
                  "4,096 ordinary presentations and4,608 evaluation generations are complete. "
                  "The288 included gate executions are part of the evaluation total. No added qualification, "
                  "preview, warmup or reference generation is included."]
    else:
        lines += ["Evidence failures:", "", *["- " + error for error in errors]]
    lines += ["", "The selected population is short-prompt CodeAlpaca under the retained project's raw partition "
              "and inventoried historical exclusions. Source outputs were not labels. Valid output caps are retained. "
              "Freshness does not establish absence from pretraining or inaccessible history.", "",
              "Bootstrap intervals resample the128 identities using the same10,000 PCG64 draws for both seeds; "
              "timing repeats and repeated F cells are not independent observations. Intervals are conditional "
              "on the fitted trajectories and selected panel. Two seeds do not support population-level seed robustness "
              "or multiplicity-adjusted claims.", "",
              "Training costs and observed memory scopes are in TRAINING_COSTS.csv; raw nested timers and loss/selection "
              "records are in TRAINING_AUDIT.json and campaign COSTS.json. Update time includes cold compilation and "
              "native checkpoint recomputation. These costs are separate from serving latency; no asynchronous, "
              "queueing or training-inclusive deployment speedup was measured. Four events assess early adaptation, "
              "not convergence, novelty, stochastic decoding or multi-turn domain shift.", "",
              "This report performs CPU reconstruction from retained receipts and native traces. Source authorship "
              "is not an independent experimental rerun. No additional seed, event or tuning follows automatically."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("output")
    parser.add_argument("--execution-commit")
    args = parser.parse_args()
    result = write_campaign_report(args.root, args.output, execution_commit=args.execution_commit)
    print(json.dumps({"status": result["status"], "errors": result["errors"]}))
