"""Reconstruct endpoint evidence and expose all attempted work and measured costs."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

from online_feedback.storage import write_json
from experiments.ok_coding.campaign import read_work
from .contract import ARMS, BUDGET, final_schedule
from .evaluation import audit_final
from .statistics import COMPARISONS, reduce_final, summarize, summarize_events


def read_json(path):
    return json.loads(Path(path).read_text())


def read_lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def _bound_path(root, value):
    path = Path(value["path"] if isinstance(value, dict) else value)
    return path if path.is_absolute() else root / path


def _work_accounting(campaign):
    """Audit durable work starts/completions once, including partial failures."""
    serving_limits = dict(pre_update_generations=3072, capture_parity_generations=4,
                          terminal_preparation_forwards=1024)
    worker_limits = dict(response_forwards=256, response_backwards=256, optimizer_steps=128)
    errors, workers = [], []
    totals = {phase: {kind: 0 for kind in worker_limits} for phase in ("attempts", "completed", "unknown")}
    def inspect(path, expected):
        item = dict(path=str(path), present=path.is_file(), expected=expected)
        try:
            work = read_work(path, expected)
            item["observed"] = work
            item["complete"] = (path.is_file() and work["attempts"] == expected and work["completed"] == expected
                                and work["unknown"] == {key: 0 for key in expected})
            if not item["complete"]:
                errors.append("incomplete_work_ledger: " + str(path))
        except (OSError, KeyError, TypeError, ValueError) as error:
            item.update(complete=False, error=repr(error))
            errors.append("malformed_work_ledger: " + str(path) + ": " + repr(error))
        return item
    serving = inspect(campaign / "WORK.jsonl", serving_limits)
    for arm in ARMS:
        for event in range(1, 9):
            item = inspect(campaign / f"arms/{arm}/events/event_{event:03d}/WORK.jsonl", worker_limits)
            item.update(arm=arm, event=event)
            if "observed" in item:
                for phase in totals:
                    for kind in worker_limits:
                        totals[phase][kind] += item["observed"][phase][kind]
            workers.append(item)
    return dict(serving=serving, training_workers=workers, observed_training_totals=totals,
                expected_training_totals={key: value * 24 for key, value in worker_limits.items()}, errors=errors,
                scope="Durable starts, completions and unknown partial work; absent/malformed ledgers prevent a complete verdict.")


def _evidence_index(campaign, requests_document):
    """Index compact provenance and join domain labels only after selection."""
    training_requests = requests_document.get("training", []) if isinstance(requests_document, dict) else []
    domains = {row["request_id"]: row["domain"] for row in training_requests}
    receipts, selection_rows, composition = [], [], []
    for event in range(1, 9):
        folder = campaign / f"events/event_{event:03d}"
        names = ["SELECTION.json", "MEMORY_BEFORE.json", "MEMORY_AFTER.json"]
        if event in (1, 3):
            names.append("PARITY.json")
        if event == 1:
            names.append("NEUTRAL_STATE.json")
        for name in names:
            path = folder / name
            receipts.append(dict(event=event, kind=name, path=str(path), present=path.is_file()))
        selection_path = folder / "SELECTION.json"
        if selection_path.is_file():
            selection = read_json(selection_path)
            for arm, key in (("U", "uniform"), ("R", "coverage")):
                chosen = selection.get(key)
                if chosen is not None:
                    for step in chosen["steps"]:
                        selection_rows.append(dict(event=event, arm=arm, **step,
                            reporting_domain=domains.get(step["record_id"])))
                    composition.append(dict(event=event, kind="selected_" + arm, records=len(chosen["record_ids"]),
                        reporting_domain_counts=dict(Counter(domains.get(record, "unavailable")
                                                             for record in chosen["record_ids"])),
                        coverage=chosen["coverage"], similarity_distribution=chosen["similarity_distribution"],
                        concentration=chosen["concentration"], ages_events=chosen["ages_events"],
                        ages_fresh_records=chosen["ages_fresh_records"]))
        for name in ("MEMORY_BEFORE.json", "MEMORY_AFTER.json"):
            path = folder / name
            if path.is_file():
                memory = read_json(path)
                slots = memory.get("slots")
                if slots is not None:
                    composition.append(dict(event=event, kind=name, records=len(slots),
                        reporting_domain_counts=dict(Counter(domains.get(row["record_id"], "unavailable") for row in slots))))
    descriptors = campaign / "ALL_DESCRIPTORS.json"
    receipts.append(dict(kind="ALL_DESCRIPTORS.json", path=str(descriptors), present=descriptors.is_file()))
    retirement_root = campaign / "RETIREMENTS"
    retirements = [dict(path=str(path), receipt=read_json(path)) for path in sorted(retirement_root.rglob("*.json"))]
    return dict(compact_receipts=receipts, membership_and_selection=composition,
                selection_steps=selection_rows, retirement_root=str(retirement_root), retirement_receipts=retirements,
                domain_scope="Reporting-only join after deterministic selection; domains were not selector inputs.",
                retirement_scope="Retired tensor files are not archived; receipts identify exact paths and retained evidence.")


def _training_costs(campaign):
    from .training import validate_event_receipt

    errors, events = [], []
    totals = dict(optimizer_steps=0, response_presentations=0, replay_presentations=0,
                  prompt_tokens=0, response_tokens=0, active_anchors=0)
    for arm in ARMS:
        previous = []
        for event in range(1, 9):
            path = campaign / f"arms/{arm}/trainer/events/event_{event:03d}/RESULT.json"
            if not path.exists():
                errors.append(f"{arm}/event_{event:03d}: missing training receipt")
                continue
            result = read_json(path)
            item = dict(arm=arm, event=event, source_path=str(path), status=result.get("status"),
                        costs=result.get("costs"), runtime_costs=result.get("runtime_costs"),
                        pass_loop_costs=result.get("pass_loop_costs"), pass_exposure=result.get("pass_exposure"))
            try:
                item["validation"] = validate_event_receipt(result, arm=arm, event_number=event,
                                                             previous_records=previous)
                previous.extend(result["features"])
            except (KeyError, TypeError, ValueError) as error:
                errors.append(f"{arm}/event_{event:03d}: {error!r}")
                item["validation"] = dict(status="invalid", error=repr(error))
            # Totals reflect recorded observations, even when a later gate fails.
            records = result.get("records", [])
            totals["response_presentations"] += len(records)
            totals["optimizer_steps"] += sum(row.get("optimizer_stepped") is True for row in records)
            for field in ("replay_presentations", "prompt_tokens", "response_tokens", "active_anchors"):
                source_field = "replay" if field == "replay_presentations" else field
                if all(source_field in row for row in records):
                    totals[field] += sum(row[source_field] for row in records)
                else:
                    errors.append(f"{arm}/event_{event:03d}: missing exposure {source_field}")
            events.append(item)
    return dict(events=events, observed_totals=totals, errors=errors,
                timing_scope="Nested recorded costs are retained separately and must not be summed as disjoint time.")


def _complete_row_ledger(rows, planned, progress):
    by_key = {(r["panel_ordinal"], r["repeat"], r["method"]): r for r in rows}
    interrupted = progress.get("interrupted_attempt")
    result = list(rows)
    for slot in planned:
        key = (slot["panel_ordinal"], slot["repeat"], slot["method"])
        if key in by_key:
            continue
        unknown = interrupted is not None and all(interrupted.get(k) == v for k, v in slot.items())
        result.append({**slot, "validation_status": "unknown_attempt" if unknown else "unattempted",
                       "valid_measurement": False, "errors": ["missing_returned_measurement"] if unknown else []})
    return result


def _recommendation(endpoint, events, complete):
    if not complete or endpoint is None:
        return dict(outcome="invalid_or_incomplete", next_action="Resolve the recorded implementation or evidence failure; no efficacy conclusion.",
                    automatic_followup_authorized=False)
    effects = endpoint["comparisons"]
    ru, rk, uk = [effects[name]["call_reduction"] for name in ("R_vs_U", "R_vs_K", "U_vs_K")]
    timing = effects["R_vs_K"]["request_latency_s_reduction"]["point_pct"]
    future = events["future_events_3_to_8"]
    compatible = (future["valid_complete"] and
                  all(future["comparisons"][name]["call_reduction_pct"] > 0 for name in ("R_vs_U", "R_vs_K")))
    if ru["ci95_pct"][0] > 0 and rk["ci95_pct"][0] > 0:
        code = "A_endpoint_conditions_met" if compatible and timing > 0 else "A_endpoint_only_or_F"
        action = ("Review measured selector/I/O/updater costs and future-domain records. Only if costs are acceptable, "
                  "propose unchanged longer validation with another seed and workload order; no automatic run.")
    elif ru["point_pct"] > 0 and rk["point_pct"] <= 0:
        code = "C"
        action = ("Do not call this an upgrade to K. Propose 16 old plus 112 fresh only if recurrence gains and lost fresh adaptation "
                  "are specifically supported; otherwise stop this allocation. Do not execute a changed dose here.")
    elif uk["point_pct"] > 0 and rk["point_pct"] > 0 and ru["ci95_pct"][0] <= 0:
        code = "B"
        action = "Retain uniform replay as the practical candidate; R-over-U remains unsupported or uncertain. No selector search is authorized."
    elif rk["point_pct"] <= 0 and uk["point_pct"] <= 0:
        code = "E_or_G_pending_recurrence_review"
        action = "Retain K; stop this allocation absent supported recurrence benefit. Do not retune the descriptor on this panel."
    else:
        code = "G"
        action = "Report uncertainty; propose at most an unchanged replication only if domain and future-serving directions are practically useful."
    return dict(outcome=code, next_action=action, future_direction_compatible=compatible,
                calls_without_time_gain=rk["point_pct"] > 0 and timing <= 0,
                domain_tradeoffs={d: endpoint["domains"][d]["comparisons"]["R_vs_K"]["call_reduction"]["point_pct"]
                                 for d in ("math", "coding")},
                automatic_followup_authorized=False,
                qualification="Endpoint rules do not replace review of per-domain tradeoffs and measured adaptation costs.")


def build_report(run_root):
    root = Path(run_root)
    campaign = root / "campaign_001"
    plan = read_json(root / "sealed/PLAN.json")
    requests_document = read_json(_bound_path(root, plan["requests"]))
    requests = requests_document["evaluation"] if isinstance(requests_document, dict) else requests_document
    canonicals = read_json(_bound_path(root, plan["canonicals"]))
    status_path = campaign / "STATUS.json"
    status = read_json(status_path) if status_path.exists() else {}
    final = campaign / "final_evaluation"
    protocol_path = final / "PROTOCOL.json"
    protocol = read_json(protocol_path) if protocol_path.exists() else {}
    audit = audit_final(final, requests, protocol.get("checkpoints", {}), canonicals=canonicals)
    returned = read_lines(final / "MEASUREMENTS.jsonl") if (final / "MEASUREMENTS.jsonl").exists() else []
    errors = list(audit["errors"])
    if plan.get("budget") != BUDGET:
        errors.append("sealed_budget_differs")
    if not plan.get("execution_source", {}).get("commit"):
        errors.append("missing_execution_commit")
    if status.get("status") != "completed":
        errors.append("campaign_not_completed")
    endpoint, paired = None, []
    if audit["status"] == "passed":
        try:
            endpoint = summarize(audit["rows"])
            paired = reduce_final(audit["rows"])
        except (KeyError, TypeError, ValueError) as error:
            errors.append("endpoint_statistics: " + repr(error))
    event_rows = []
    for event in range(1, 9):
        path = campaign / f"events/event_{event:03d}/MEASUREMENTS.jsonl"
        if path.exists():
            event_rows.extend(read_lines(path))
        else:
            errors.append(f"missing_event_{event:03d}_serving")
    try:
        event_summary = summarize_events(event_rows)
        if len(event_summary["events"]) != 8 or any(not e["valid_complete"] for e in event_summary["events"]):
            errors.append("event_serving_incomplete_or_invalid")
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        errors.append("event_statistics: " + repr(error))
        event_summary = dict(events=[], future_events_3_to_8={"valid_complete": False}, error=repr(error))
    training = _training_costs(campaign)
    errors.extend(training["errors"])
    evidence = _evidence_index(campaign, requests_document)
    errors.extend("missing_compact_evidence: " + row["path"] for row in evidence["compact_receipts"] if not row["present"])
    work_accounting = _work_accounting(campaign)
    errors.extend(work_accounting["errors"])
    observed = training["observed_totals"]
    for key, expected in (("optimizer_steps", 3072), ("response_presentations", 6144), ("replay_presentations", 448)):
        if observed[key] != expected:
            errors.append("training_budget_" + key)
    summary = dict(status="valid_complete" if not errors else "invalid_or_incomplete", errors=errors,
        execution_source=plan.get("execution_source"), declared_budget=plan.get("budget"),
        endpoint=endpoint, event_summary=event_summary, training=training,
        evidence=evidence,
        work_accounting=work_accounting,
        campaign_status=status, final_evaluation_status=audit["progress"],
        validity_audit={key: value for key, value in audit.items() if key not in ("rows", "saved_rows")},
        costs_source=str(status_path), cost_scope="Observed fields only; absent costs are not inferred or treated as zero.",
        limitations=["One seed and intentionally recurrent workload; no natural-traffic claim.",
                     "Previously used development panel; no fresh confirmatory-panel claim.",
                     "Common K-generated supervision; not three independent self-generated streams.",
                     "Cannot add these gains to the separately trained PR23 K58 gain.",
                     "Equal presentations do not establish equal token FLOPs or training-inclusive speedup.",
                     "Coverage is a proxy; its increase alone does not demonstrate serving utility."])
    summary["material_passport"] = dict(mode="run→validate", verificationStatus="raw-evidence-analyzed",
        execution_commit=plan.get("execution_source", {}).get("commit"),
        source_handoff=plan.get("source_handoff", "TRACK2_K_REPLAY_EXECUTOR_INSTRUCTIONS_20261102.txt; user handoff dated 2026-11-02"),
        scope="CPU reconstruction of retained raw/native/work records; not an independent GPU reproduction.")
    summary["recommendation"] = _recommendation(endpoint, event_summary, not errors)
    destination = root / "final_report"
    destination.mkdir(exist_ok=True)
    write_json(destination / "SUMMARY.json", summary)
    write_json(destination / "EVENT_SUMMARY.json", event_summary)
    write_json(destination / "TRAINING_COSTS.json", training)
    write_json(destination / "EVIDENCE_INDEX.json", evidence)
    write_json(destination / "WORK_ACCOUNTING.json", work_accounting)
    write_csv(destination / "SELECTION_STEPS.csv", evidence["selection_steps"])
    write_csv(destination / "ALL_FINAL_ROWS.csv", _complete_row_ledger(returned, final_schedule(requests), audit["progress"]))
    write_csv(destination / "EVENT_ROWS.csv", event_rows)
    flat = []
    for row in paired:
        output = {key: row[key] for key in ("panel_ordinal", "request_id", "domain")}
        for arm in ARMS:
            output.update({arm + "_" + key: value for key, value in row[arm].items()})
        flat.append(output)
    write_csv(destination / "FINAL_PAIRED.csv", flat)
    passport = summary["material_passport"]
    lines = ["# Workload-conditioned replay with K", "", "## Material Passport", "",
             "Mode: run→validate. Verification status: raw-evidence-analyzed.", "",
             passport["scope"], "", "Source handoff: " + passport["source_handoff"] + ".", "",
             "Status: " + summary["status"], "",
             "Execution source: `" + str(plan.get("execution_source", {}).get("commit")) + "`.", "",
             "Source receipt: " + json.dumps(plan.get("execution_source"), sort_keys=True), "",
             "Primary contrast: R/U; required usefulness contrast: R/K. Counts use 256 distinct requests; "
             "time uses the per-request mean of three repeats. All 2,304 planned endpoint slots remain in ALL_FINAL_ROWS.csv.", ""]
    if endpoint is not None:
        lines.extend(["| Panel | Contrast | Call reduction % [95% interval] | Wins / regressions / ties | Gross saved / added | Time reduction % |",
                      "|---|---|---:|---:|---:|---:|"])
        for label, value in [("pooled", endpoint), *endpoint["domains"].items()]:
            for a, b in COMPARISONS:
                effect = value["comparisons"][a + "_vs_" + b]
                point = effect["call_reduction"]
                low, high = point["ci95_pct"]
                lines.append(f"| {label} | {a}/{b} | {point['point_pct']:.4f} [{low:.4f}, {high:.4f}] | "
                    f"{effect['request_wins']} / {effect['request_regressions']} / {effect['request_ties']} | "
                    f"{effect['gross_calls_saved']} / {effect['gross_calls_added']} | "
                    f"{effect['request_latency_s_reduction']['point_pct']:.4f} |")
        lines.extend(["", endpoint["scope"], "", endpoint["secondary_intervals"], "",
                      "Raw totals, throughput ratios, accepted draft tokens/call, committed progress/call and every repeat "
                      "are retained in SUMMARY.json; FINAL_PAIRED.csv preserves each request's counts and mean times.", ""])
    lines.extend(["Future-serving events 3–8 are descriptive only. Capture-on K and capture-off U/R times are not speedup comparators.", "",
                  "Observed training exposure: " + json.dumps(observed, sort_keys=True) + ".", "",
                  "Training forward/backward, token/anchor exposure, per-pass loading/bookkeeping, updater and runtime costs "
                  "remain in TRAINING_COSTS.json. Campaign and final evaluation statuses preserve recorded preparation, "
                  "publication, resource and total-time fields separately; missing values are not fabricated. Nested timers are not additive.", "",
                  "[Work accounting](WORK_ACCOUNTING.json) audits the root generation/preparation ledger and all 24 training-worker ledgers, "
                  "including attempted, completed and unknown partial operations. Required totals are 3,072 pre-update generations, "
                  "4 parity generations, 1,024 preparations, 6,144 forwards, 6,144 backwards and 3,072 optimizer steps; "
                  "the final evaluator separately accounts for its 2,304 generations.", "",
                  "[Evidence index](EVIDENCE_INDEX.json) links every event's selection, before/after memory, capture parity and neutral-state receipts, "
                  "the full descriptor ledger and exact retirement receipts. [Selection steps](SELECTION_STEPS.csv) preserve marginal gains and "
                  "reporting-only domain composition; memory mixture, similarity distributions, concentration and ages are in the evidence index.", "",
                  "Recommendation: " + summary["recommendation"]["next_action"], "",
                  *["- " + value for value in summary["limitations"]]])
    costs = status.get("costs", {})
    cost_names = ("initialization_s", "backend_loading_s", "training_worker_wall_s", "activation_s",
                  "teacher_preparation_s", "feature_persistence_s", "descriptor_s", "selection_s",
                  "memory_maintenance_s", "export_s", "resource_check_s", "adaptation_ready_wall_s",
                  "evaluation_wall_s", "campaign_wall_s")
    lines.extend(["", "Recorded component wall times (seconds; nested scopes may overlap):", "",
                  "| Component | Seconds |", "|---|---:|"])
    for name in cost_names:
        value = costs.get(name, status.get(name))
        lines.append(f"| {name} | {value if value is not None else 'unavailable'} |")
    lines.extend(["", "Resource checks remain at their original granularity in campaign_status.resource_checks "
                  "or campaign_status.costs.resource_checks in SUMMARY.json; these do not infer unmeasured peaks."])
    if errors:
        lines.extend(["", "Validity and completeness failures:", "", *["- " + str(error) for error in errors]])
    (destination / "REPORT.md").write_text("\n".join(lines) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root")
    print(json.dumps(build_report(parser.parse_args().run_root)["status"]))
