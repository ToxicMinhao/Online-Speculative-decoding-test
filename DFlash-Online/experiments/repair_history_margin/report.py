"""Publish token-free evidence for the frozen repair-history diagnostic."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .analysis import analyze_k6, build_pre_k6


def _read(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def _table(path, rows, columns):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row.get(key), ensure_ascii=False)
                             if isinstance(row.get(key), (list, dict)) else row.get(key)
                             for key in columns})


def _number(value, percent=False):
    if value is None:
        return "unavailable"
    return f"{value * 100:.4f}%" if percent else f"{value:.6f}"


def _pp(value):
    return "unavailable" if value is None else f"{100 * value:+.4f} percentage points"


def _interval(receipt):
    if receipt is None:
        return "unavailable"
    return f"[{_number(receipt['lower'], True)}, {_number(receipt['upper'], True)}]"


def _leaves(value, prefix=""):
    if isinstance(value, dict):
        return [row for key, child in value.items()
                for row in _leaves(child, prefix + ("." if prefix else "") + str(key))]
    if isinstance(value, (int, float, str, bool)) or value is None:
        return [dict(field=prefix, value=value)]
    return []


def work_accounting(plan, budget, seal):
    """Reconcile actual required work, not the budget's conservative allowances.

    Before matching, the number of required K+6 starts is genuinely unknown.
    A failed parent capture is counted from its durable attempt even when it
    produced no PARENT.json or STARTS.json row.
    """
    scope = plan.get("scope", {})
    planned = dict(parent_walk=scope.get("parent_instrumented_walks", len(plan.get("requests", []))),
                   neutral_block=scope.get("archived_k5_rounds"),
                   historical_block=scope.get("archived_k5_rounds"),
                   outcome_block=len(seal["required_k6_starts"]) if seal else None)
    rows, errors = [], []
    present = "attempted" in budget
    kinds = list(planned) + sorted(set(budget.get("attempted", {})) - set(planned))
    for kind in kinds:
        row = dict(operation=kind, planned=planned.get(kind),
                   attempted=budget.get("attempted", {}).get(kind, 0) if present else None,
                   completed=budget.get("completed", {}).get(kind, 0) if present else None,
                   failed=budget.get("failed", {}).get(kind, 0) if present else None,
                   unknown=budget.get("unknown", {}).get(kind, 0) if present else None,
                   unattempted=None)
        if present:
            if row["completed"] + row["failed"] + row["unknown"] != row["attempted"]:
                errors.append(kind + ": attempt status counts do not reconcile")
            if row["planned"] is not None:
                row["unattempted"] = row["planned"] - row["attempted"]
                if row["unattempted"] < 0:
                    errors.append(kind + ": attempted work exceeds actual required work")
        rows.append(row)
    return dict(status="passed" if present and not errors else "unavailable" if not present else "failed",
                errors=errors, rows=rows)


def report(plan, output):
    """Read durable receipts, independently recompute the joined analysis.

    Missing pre-outcome evidence leaves an explicitly incomplete report. It is
    never replaced with a retrospectively sealed, outcome-informed match list.
    """
    output = Path(output)
    plan = plan if isinstance(plan, dict) else _read(Path(plan))
    destination = output / "final_report"
    destination.mkdir(exist_ok=True)
    seal = _read(output / "PRE_K6.json")
    starts = _read(output / "STARTS.json", [])
    outcomes = _read(output / "OUTCOMES.json", [])
    status = _read(output / "RESULT.json", {})
    qa = _read(output / "MARGIN_QA.json", {})
    budget = _read(output / "BUDGET.json", {})
    seal_receipt = _read(output / "PRE_K6_SEAL.json", {})
    accounting = work_accounting(plan, budget, seal)
    parity = []
    for path in sorted((output / "requests").glob("*/PARENT_PARITY.json")):
        row = _read(path)
        parity.append(dict(request_id=path.parent.name, kind="parent", status=row["status"],
                           archived_trace_available=row.get("archived_trace_available"),
                           differences=row.get("differences", [])))
    for path in sorted((output / "requests").glob("*_PARITY.json")):
        row = _read(path)
        parity.append(dict(request_id=path.parent.name, kind=path.stem,
                           status=row["status"], differences=row.get("differences", [])))
    # Native per-start receipts are nested one request deeper.
    for path in sorted((output / "requests").glob("*/*_PARITY.json")):
        if path.name == "PARENT_PARITY.json":
            continue
        row = _read(path)
        parity.append(dict(request_id=path.parent.name, kind=path.stem,
                           status=row["status"], differences=row.get("differences", [])))
    complete = status.get("status") == "completed"
    rebuilt = build_pre_k6(starts) if seal is not None else None
    reconstruction_match = bool(seal and rebuilt["payload_sha256"] == seal["payload_sha256"])
    if seal is not None and not reconstruction_match:
        raise ValueError("Durable K4/K5 records do not reconstruct the pre-K6 seal")
    result = analyze_k6(seal, outcomes, execution_complete=complete) if seal else None
    summary = dict(
        schema="repair_history_margin.report.v1", execution_status=status.get("status", "incomplete"),
        execution_commit=plan.get("execution_commit"), panel_identity=plan.get("panel_identity"),
        checkpoints=plan.get("checkpoints"), pre_k6_seal=seal_receipt,
        pre_k6_payload_sha256=seal.get("payload_sha256") if seal else None,
        durable_matching_reconstruction_passed=reconstruction_match,
        margin_qa=qa, parity=dict(records=len(parity), passed=sum(r["status"] == "passed" for r in parity),
                                failed=sum(r["status"] != "passed" for r in parity)),
        work=budget, work_accounting=accounting, execution=status, result=result,
        interpretation="Frozen-checkpoint prognostic association, not causality or native speedup.",
    )
    _write(destination / "SUMMARY.json", summary)
    _table(destination / "PARITY.csv", parity, ("request_id", "kind", "status", "archived_trace_available", "differences"))
    _table(destination / "COSTS.csv", _leaves(budget), ("field", "value"))
    _table(destination / "WORK_ACCOUNTING.csv", accounting["rows"],
           ("operation", "planned", "attempted", "completed", "failed", "unknown", "unattempted"))
    if seal:
        _write(destination / "PRE_K6_PUBLIC.json", seal)  # stripped of token IDs by build_pre_k6
        _table(destination / "ALL_STARTS.csv", seal["labels"],
               ("source_id", "start_offset", "a4", "a5", "valid_proposal_count", "k4_first_error", "recently_repaired", "j"))
        _table(destination / "K5_MARGINS.csv", [dict(source_id=r["source_id"], start_offset=r["start_offset"], **m)
                for r in seal["labels"] for m in r["margins"]], ("source_id", "start_offset", "slot", "margin"))
        _table(destination / "REPAIR_COVERAGE.csv", seal["repairs"],
               ("source_id", "repaired_start", "j", "a4", "a5", "repaired_margin", "structural_comparators", "caliper_comparators", "match_status"))
        _table(destination / "STRUCTURAL_POOLS.csv", [dict(source_id=p["source_id"], repaired_start=p["repaired_start"],
                j=p["j"], a5=p["a5"], comparator_start=c["start_offset"], comparator_margin=c["margin"])
                for p in seal["structural_pools"] for c in p["comparators"]],
               ("source_id", "repaired_start", "j", "a5", "comparator_start", "comparator_margin"))
        _table(destination / "COMPARATOR_REUSE.csv", seal["comparator_reuse"], ("source_id", "start_offset", "uses"))
        _table(destination / "MATCHED_PAIRS.csv", result["matched_pair_outcomes"], (
            "source_id", "repaired_start", "comparator_start", "j", "a4", "a5", "repaired_margin", "comparator_margin",
            "absolute_margin_difference", "pair_weight", "valid_pair", "repaired_valid", "comparator_valid",
            "repaired_outcome", "comparator_outcome", "repaired_direct_correct", "comparator_direct_correct",
            "exact_slot_rejection_difference", "earlier_rejection_difference", "accepts_through_difference",
            "repaired_reason", "comparator_reason"))
        _table(destination / "STRUCTURAL_OUTCOMES.csv", result["structural_outcomes"], (
            "source_id", "repaired_start", "j", "comparators", "valid_comparators", "valid",
            "comparator_exact_slot_rejection_rate", "exact_slot_rejection_difference", "earlier_rejection_difference"))
        _table(destination / "PER_REQUEST.csv", result["per_request"],
               ("source_id", "pairs", "valid_pairs", "exact_slot_difference_sum"))
    lines = ["# Does recent repair history add risk beyond current confidence?", "", "## Material Passport", "",
             "- Mode: bounded experiment execution and statistical validation.",
             "- Study: frozen K+4/K+5/K+6; no training, optimizer steps, new arm or latency benchmark.",
             f"- Execution status: **{summary['execution_status']}**.",
             f"- Execution commit: `{summary['execution_commit']}`.",
             "- Scope: the prescribed GSM8K panel and fixed checkpoint transition; three aggregate A800-hour cap.", ""]
    if not seal:
        lines += ["**INCOMPLETE: no pre-K+6 comparison seal was produced.** No primary effect is available.",
                  f"Durable starts: {len(starts)}; K+6 outcomes: {len(outcomes)}.",
                  "Preserved failure receipts identify the execution blocker; no outcome-informed matching was substituted.", ""]
    else:
        p, structural, cov, balance = result["primary"], result["structural"], seal["coverage"], seal["balance"]
        lines += [f"**Decision: {result['decision']}.**", "",
                  f"Matched exact-slot risk difference: **{_pp(p['delta'])}**, "
                  f"request-cluster 95% interval {_interval(p['interval'])}. The positive direction means higher risk for recent repairs.",
                  f"Support/validity flags: {', '.join(result['interpretation_flags']) or 'none'}.", "",
                  "## Coverage and margin balance", "",
                  f"{cov['all_repairs']} recent repairs; {cov['repairs_with_structural_comparator']} have a structural comparator; "
                  f"{cov['matched_repairs']} match inside the fixed 0.25 caliper, across {cov['matched_requests']} requests.",
                  f"{cov['distinct_comparator_starts']} distinct comparator starts; maximum reuse {cov['maximum_comparator_reuse']}. "
                  f"{cov['unmatched_repairs']} repairs remain unmatched. Tables retain their margins, depths and reasons.",
                  f"Absolute SMD: {_number(balance['absolute_smd'])}; mean margins repaired/established "
                  f"{_number(balance['repaired']['mean'])}/{_number(balance['established']['mean'])}. "
                  f"Absolute paired difference median/p95/max: {_number(balance['absolute_difference']['median'])}/"
                  f"{_number(balance['absolute_difference']['q95'])}/{_number(balance['absolute_difference']['maximum'])}.",
                  "Matching uses replacement, exact-zero ties only with zero ties, and the sealed SHA256 tie breaker. "
                  "Every repair has weight one. SMD uses the two pair-weighted sample variances; identical constants give zero.", "",
                  "## K+6 outcomes", "", "| Outcome | Recent repair | Established |", "|---|---:|---:|"]
        for label, name in (("First rejection exactly at j", "exact_slot_rejection"),
                            ("First rejection before j", "earlier_rejection"), ("Accepts through j", "accepts_through")):
            lines.append(f"| {label} | {p['repaired']['counts'][name]}/{p['denominator']} ({_number(p['repaired']['rates'][name], True)}) | "
                         f"{p['established']['counts'][name]}/{p['denominator']} ({_number(p['established']['rates'][name], True)}) |")
        lines += ["", f"Valid matched pairs: {p['valid_pairs']}/{p['denominator']}. Earlier breaks remain in the denominator; "
                  "missing evidence is not assigned zero risk and disables the primary estimate.",
                  f"Earlier-break difference: {_pp(p['earlier_break_difference'])}; direct token-agreement difference "
                  f"(including positions blocked by earlier errors): {_pp(p['direct_correct_difference'])}.",
                  f"Before margin adjustment, on the **same matched repairs**, the structural-pool risk difference is "
                  f"{_pp(structural['delta'])}, interval {_interval(structural['interval'])}. "
                  "Each repair is compared with the mean of all its same-request, same-slot, same-a5 established starts.",
                  "Intervals use 10,000 whole-request draws with PCG64 seed 20261115. All pairs and reused controls in a request "
                  "move together, retaining pair weights. Requests with no matched pair are outside this matched-population bootstrap.",
                  f"Undefined primary bootstrap draws: {p['interval']['undefined_draws'] if p['interval'] else 'not run'}.", "",
                  "## Interpretation limits", "",
                  "The prespecified useful-risk threshold is +5 percentage points, not a predicted speedup. "
                  "A lower bound above it requires adequate support and balance. An upper bound below it rules out that "
                  "positive association size only on this matched population; smaller effects may remain. A negative estimate "
                  "has its own direction. Poor overlap, imbalance, missing data or an interval crossing the threshold remains unresolved.",
                  "Structural-gap attenuation after matching is consistent with current confidence explaining part of the discovery "
                  "association. It does not establish causality or identify a successful retention intervention. "
                  "Unmatchable starts, residual confounding, one seed and one checkpoint transition limit generalization.", ""]
    lines += ["## Identity, parity and work", "",
              f"Pre-K+6 payload hash: `{summary['pre_k6_payload_sha256']}`. Durable K4/K5 reconstruction: "
              f"{reconstruction_match}. File seal and checkpoint hashes are retained in SUMMARY.json.",
              f"Native parity: {summary['parity']['passed']}/{summary['parity']['records']} passed; "
              f"margin QA status: {qa.get('status', 'unavailable')}.",
              "COSTS.csv preserves the complete scalar work/time receipt, including initialization, partial work and GPU identity where recorded. "
              "The experiment does not interpret routing or replay duration as deployment latency.", "",
              f"Work accounting: {accounting['status']}. WORK_ACCOUNTING.csv separates actual required, attempted, completed, failed, "
              "unknown and unattempted operations. Failed captures remain counted even without a completed request record. "
              "Before the pre-K+6 seal exists, required and unattempted K+6 work remain unknown, rather than being replaced by an allowance.", "",
              "## Statistical interpretation audit", "",
              "11/11 categories checked: request heterogeneity (per-request table); ecological inference (pair-level risks); "
              "selection/Berkson bias (matched population and unmatched coverage); collider adjustment (associational confidence matching); "
              "base rates (all three risks and denominators); regression to the mean (history selected, no causal claim); "
              "survivorship (earlier breaks and missing work retained); look-elsewhere (one fixed primary outcome); "
              "forking paths (pre-outcome seal); correlation versus causation (prognostic only); "
              "reverse causality (K+5 inputs fixed before K+6 outcomes). These checks do not establish absence of residual bias.", "",
              "This report ends the prescribed diagnostic. No result authorizes training or an additional panel.", ""]
    (destination / "REPORT.md").write_text("\n".join(lines))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report(args.plan, args.output)


if __name__ == "__main__":
    main()
