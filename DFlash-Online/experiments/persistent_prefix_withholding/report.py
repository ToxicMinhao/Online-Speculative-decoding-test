"""Offline campaign report; incomplete evidence never becomes an efficacy claim."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from online_feedback.storage import write_json
from . import analysis
from .campaign import production_expectations
from .evaluation import SCOPE
from .resources import LIMITS

TASKS = ("E7", "S7", "N_collect", "E_collect", "S_collect", "N8", "E8", "S8", "references", "final")


def optional(path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (ValueError, OSError) as error:
        return dict(status="unreadable", path=str(path), error=repr(error))


def planned_work(plan):
    missing = optional(plan["missing_references"])
    if missing is None or isinstance(missing, dict) and missing.get("status") == "unreadable":
        return None
    missing = missing["requests"] if isinstance(missing, dict) else missing
    planned = production_expectations(len(missing))
    planned["adam"] += 8
    planned["response_forward"] += 20  # Sixteen qualification plus four startup.
    planned["response_backward"] += 20
    planned["training_initialization"] += 2
    planned["qualification_generation"] += 8
    return planned


def qualification_work(root):
    """Recover pre-campaign work from durable native operation ledgers."""
    from experiments.threatened_repair_replay.work import read_ledger
    if (root/"CAMPAIGN.json").exists():
        return None  # A missing production ledger cannot be replaced by a subset.
    work = {key:{kind:0 for kind in LIMITS} for key in ("attempted","completed","failed")}
    sources = []
    mapping = dict(model_load="training_initialization", optimizer_step="adam",
                   response_forward="response_forward", response_backward="response_backward")
    for slot in ("A","B"):
        path = root/"qualification"/slot/"PHYSICAL_WORK.jsonl"
        if path.exists():
            value = read_ledger(path)
            sources.append(str(path))
            for field in work:
                for key, destination in mapping.items():
                    work[field][destination] += value[field].get(key,0)
        path = root/"serving_qualification"/slot/"calls/RESULT.json"
        value = optional(path)
        if value is not None and value.get("status") != "unreadable":
            sources.append(str(path))
            work["attempted"]["qualification_generation"] += value["attempted_generations"]
            work["completed"]["qualification_generation"] += value["returned_generations"]
    return dict(**work, pending=None, source_records=sources) if sources else None


def work_accounting(root, plan):
    """Report shared durable work without inferring missing or failed operations."""
    planned = planned_work(plan)
    work = optional(root / "WORK.json")
    source = "WORK.json" if work else None
    if work is None:
        work = qualification_work(root)
        source = "retained qualification ledgers" if work else None
    rows = []
    for kind in LIMITS:
        attempted = work.get("attempted", {}).get(kind) if work else None
        completed = work.get("completed", {}).get(kind) if work else None
        failed = work.get("failed", {}).get(kind,0) if work else None
        amount = planned[kind] if planned is not None else None
        rows.append(dict(kind=kind, planned=amount, hard_cap=LIMITS[kind],
            attempted=attempted, completed=completed,
            failed=failed, unknown=None if attempted is None or completed is None else attempted-completed-failed,
            unattempted=None if attempted is None or amount is None else max(0, amount-attempted)))
    return dict(source=source, source_records=work.get("source_records") if work else None, rows=rows,
        pending=None if work is None else work.get("pending"),
        accounting_complete=bool(work and planned is not None and not work.get("pending")
            and all(row["attempted"] == row["completed"] == row["planned"] for row in rows)),
        missing_global_ledger="Qualification receipts remain separate; absent totals are not inferred as zero." if work is None else None)


def task_records(root):
    tasks = []
    for name in TASKS:
        base = root / "tasks" / name
        status = optional(base / "STATUS.json")
        result = optional(base / "RESULT.json")
        tasks.append(dict(task=name, status=status.get("status", "unknown") if status else "unattempted",
            status_record=status, result=result, memory=optional(base / "MEMORY.json"),
            native_work=optional(base / "PHYSICAL_WORK.json"),
            retirement=optional(base / "RETIREMENT.json")))
    return tasks


def cost_records(root, tasks):
    """Keep nested costs separate: event wall time already contains training/save."""
    records = []
    for task in tasks:
        result, status = task["result"] or {}, task["status_record"] or {}
        row = dict(task=task["task"], status=task["status"], physical_gpu_uuid=status.get("physical_gpu_uuid"),
            task_wall_s=status.get("wall_s"), loading_s=result.get("loading_s"),
            startup_s=result.get("startup", {}).get("wall_s"), event_wall_s=result.get("event_wall_s"),
            publication_wait_s=result.get("publication_wait_s"),
            boundary_validation_s=result.get("boundary", {}).get("wall_s"),
            **{key: result.get("costs", {}).get(key) for key in ("training_s", "checkpoint_save_s", "export_s")})
        if task["task"] == "final":
            row.update(activation_s=result.get("activation_s"),weight_validation_s=result.get("weight_validation_s"),
                       evaluation_wall_s=result.get("evaluation_wall_s"))
        if task["task"].endswith("_collect"):
            base = root / "tasks" / task["task"]
            preparation = [optional(path) for path in sorted((base / "loop/features").glob("*.json"))
                           if not path.name.endswith(".terminal.json")]
            preparation = [value for value in preparation if value and value.get("status") == "prepared"]
            generation = [optional(path) for path in sorted((base / "records").glob("*.json"))]
            generation = [value for value in generation if value and "metadata" in value]
            row.update(prepared_receipts=len(preparation), returned_feedback_records=len(generation),
                teacher_preparation_s=sum(value["teacher_preparation_s"] for value in preparation),
                feature_persistence_s=sum(value["feature_persistence_s"] for value in preparation),
                feedback_request_s=sum(value["metadata"]["request_latency_s"] for value in generation),
                feedback_native_generate_s=sum(value["metadata"]["native_generate_s"] for value in generation))
        records.append(row)
    return records


def gate_records(plan):
    paths = {**plan.get("qualification_receipts", {}),
        **{key:plan[key] for key in ("cpu_gate", "reference_provenance", "resource_admission") if key in plan}}
    return {key: dict(path=str(path), receipt=optional(path)) for key, path in paths.items()}


def assemble_report(plan_path, *, native_trace_resolver=None):
    """Reuse the audited evaluator; otherwise publish only execution evidence."""
    plan_path = Path(plan_path)
    plan = json.loads(plan_path.read_text())
    root = Path(plan["root"])
    campaign = optional(root / "CAMPAIGN.json")
    tasks = task_records(root)
    gates = gate_records(plan)
    work = work_accounting(root, plan)
    qualification = {kind: {slot: optional(root/kind/slot/"STATUS.json") for slot in ("A", "B")}
                     for kind in ("qualification", "serving_qualification")}
    required_gates = {"A", "B", "cross_card", "serving", "cpu_gate", "reference_provenance", "resource_admission"}
    ready = bool(campaign and campaign.get("status") == "completed"
        and campaign.get("execution_commit") == plan["execution_commit"]
        and all(task["status"] == "completed" for task in tasks)
        and required_gates <= set(gates)
        and all(gates[name]["receipt"] and gates[name]["receipt"].get("status") == "passed" for name in required_gates)
        and all(gates[name]["receipt"].get("execution_commit") == plan["execution_commit"]
                for name in ("A", "B", "cross_card", "serving", "cpu_gate"))
        and work["accounting_complete"])
    efficacy = None
    if ready:
        result = analysis.write_report(root / "tasks/final/evaluation", native_trace_resolver=native_trace_resolver)
        if "decision" in result and result.get("execution_commit") == plan["execution_commit"]:
            efficacy = {key:value for key,value in result.items() if key != "paired_requests"}
    costs = cost_records(root, tasks)
    report = dict(status="COMPLETED_VALID_COMPARISON" if efficacy else "INCOMPLETE_NO_EFFICACY_CLAIM",
        execution_commit=plan["execution_commit"], scope=SCOPE, algorithm_changed=False,
        efficacy=efficacy, campaign=campaign, qualification=qualification, gates=gates,
        work=work, tasks=tasks, costs=costs, clock=optional(root/"CLOCK.json"),
        storage_plan=plan.get("storage"), production_admission=optional(root/"PRODUCTION_ADMISSION.json"),
        physical_devices=plan.get("devices"), fixed_cost_prices=plan.get("costs"),
        source_plan=str(plan_path), source_binding=plan.get("binding"),
        cost_scope="Task/event wall times include nested operations; do not add them to their component times.",
        limitations=["One training origin and seed on deliberately reused development data.",
            "No independent confirmation, long-horizon claim, or training-inclusive speedup.",
            "N7 was reused; N has one new event while E and S each have two new events.",
            "Native accepted draft tokens and stop-clipped matching counts are distinct.",
            "Partial or invalid execution provides no scientific comparison; missing totals are not zero."],
        automatic_followup_authorized=False)
    write_json(root/"SUMMARY.json", report)
    fields = sorted(set().union(*(row.keys() for row in costs)))
    with (root/"TASK_COSTS.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(costs)
    lines = ["# Persistent prefix-withholding campaign", "", SCOPE, "",
        f"Execution commit: `{plan['execution_commit']}`. Status: **{report['status']}**.", ""]
    if efficacy:
        lines.extend(["Decision: **"+efficacy["decision"]["outcome"]+"**.", "",
            "| Contrast | Request-time reduction | Native acceptance change | Requests with fewer/more/tied calls |", "|---|---:|---:|---:|"])
        for name, effect in efficacy["comparisons"].items():
            def show(value):
                if value["point"] is None:
                    return "undefined"
                bounds = value["ci95"]
                text = "undefined interval" if bounds is None else f"95% CI [{100*bounds[0]:+.3f}, {100*bounds[1]:+.3f}]%"
                return f"{100*value['point']:+.3f}% ({text})"
            lines.append(f"| {name} | {show(effect['time_reduction'])} | {show(effect['acceptance_improvement'])} | "
                f"{effect['requests_fewer_calls']}/{effect['requests_more_calls']}/{effect['request_call_ties']} |")
        lines.extend(["", "| Arm | Calls | Native accepted drafts/call | Committed progress/call | Complete request time (s) |",
                      "|---|---:|---:|---:|---:|"])
        for method, value in efficacy.get("method_totals", {}).items():
            def numeric(number):
                return "undefined" if number is None else f"{number:.6f}"
            lines.append(f"| {method} | {value['verification_calls']} | {numeric(value['native_accepted_tokens_per_call'])} | "
                         f"{numeric(value['committed_progress_per_call'])} | {value['request_latency_s']:.6f} |")
        lines.extend(["", "The fixed decision uses paired whole-request bootstrap intervals and all three repeat directions. "
            "Counts use one validated trajectory per request; times use three fresh repeat means.", "",
            "Full method totals, native/progress histograms, repeat/block totals and all three contrasts are in SUMMARY.json. "
            "Audited per-request/repeat tables and warmup records are in tasks/final/evaluation."])
    else:
        lines.extend(["The fixed scientific comparison is incomplete or invalid. No efficacy conclusion is reported.", "",
            "Completed work, failures and unavailable evidence remain in SUMMARY.json and the original task/qualification records."])
    lines.extend(["", "| Work | Planned | Completed | Failed | Unknown | Unattempted |", "|---|---:|---:|---:|---:|---:|"])
    for row in work["rows"]:
        display = lambda value: "unavailable" if value is None else str(value)
        lines.append("| " + " | ".join([row["kind"], *(display(row[key]) for key in ("planned","completed","failed","unknown","unattempted"))]) + " |")
    clock = report["clock"] or {}
    actual = campaign.get("actual_clock", {}) if campaign else {}
    lines.extend(["", f"Recorded completed occupied-card time: {clock.get('completed_card_s', 'unavailable')} seconds. "
        f"Campaign elapsed time: {actual.get('elapsed_s', 'unavailable')} seconds. "
        "Live leases and each task's wall time remain separate in the machine-readable report.",
        "Storage, source/CPU/native gates, physical cards, actual schedule and idle reasons remain recorded. "
        "TASK_COSTS.csv separates training, loading, startup, teacher preparation, persistence and publication; nested timers are not additive.", "",
        *report["limitations"], "", "No automatic retry, additional event or mechanism change is authorized."])
    (root/"REPORT.md").write_text("\n".join(lines)+"\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    args = parser.parse_args()
    assemble_report(args.plan)


if __name__ == "__main__":
    main()
