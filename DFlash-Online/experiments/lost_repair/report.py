"""Publish the bounded lost-repair gate from compact execution receipts only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from .analysis import analyze


TABLE_FIELDS = {
    "PER_REQUEST.csv": (
        "request_id", "natural_starts", "valid_genuine_k6_rejections", "H", "M", "converse",
        "H_lost_prefix_occurrences", "missing_replays", "invalid_replays",
        "H_scheduled_distinct_anchors", "H_same_j_distinct_anchors",
        "H_comparable_M_distinct_anchors", "H_matched_distinct_anchors",
        "native_H_weight_before_matching", "projected_matched_weight",
    ),
    "ANCHOR_COVERAGE.csv": (
        "request_id", "canonical_start", "j", "a4", "a5", "a6", "previously_attained_then_lost",
        "scheduled_occurrences", "same_j_occurrences", "comparable_M_occurrences",
        "matched_occurrences", "native_H_weight", "priorities",
    ),
    "NATIVE_WEIGHTS.csv": (
        "presentation_id", "pass_index", "request_id", "canonical_start", "anchor_ordinal",
        "j", "priority", "native_priority", "successor_valid", "native_coefficient",
        "ordinary_first_error_weight", "first_error_weight", "wrong_margin", "wrong_at_tie",
        "valid_proposals", "correct_prefix", "natural_history", "same_first_error_j",
        "matched_history", "comparable_M_available", "matched_H", "matched_M", "unmatched_H_reason",
    ),
    "PRESENTATION_WEIGHTS.csv": (
        "presentation_id", "pass_index", "request_id", "expected_anchor_count",
        "sampled_anchor_occurrences", "selected_weight", "first_error_occurrences",
        "native_first_error_weight", "H_same_j_occurrences", "native_H_weight_before_matching",
        "serialized_anchor_first_error_weight", "first_error_mass_consistency_checked",
        "first_error_mass_difference",
        "matched_H_occurrences", "projected_matched_weight",
        "low_coefficient", "high_coefficient", "response_forwards", "projection_chunks",
        "backwards", "optimizer_steps",
    ),
    "MATCHED_PAIRS.csv": (
        "presentation_id", "request_id", "j", "priority", "H_start", "M_start",
        "H_margin", "M_margin", "margin_difference", "absolute_margin_difference",
        "H_weight", "M_weight", "projected_weight",
    ),
}


def _read(path, missing, errors):
    if not path.is_file():
        missing.append(str(path))
        return None
    try:
        return json.loads(path.read_text())
    except (ValueError, OSError) as error:
        errors.append(dict(path=str(path), error=repr(error)))
        return None


def _json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def _csv(path, rows):
    fields = TABLE_FIELDS[path.name]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row[key]) if isinstance(row.get(key), (dict, list))
                             else row.get(key) for key in fields})


def _number(value, *, percent=False):
    if value is None:
        return "unavailable"
    if isinstance(value, int):
        return str(value)
    return f"{100 * value:.6f}%" if percent else f"{value:.6f}"


def _numeric_leaves(value, prefix=""):
    """Cost/work publication includes scalar receipts, never private payloads."""
    rows = []
    if isinstance(value, dict):
        for key, child in value.items():
            rows.extend(_numeric_leaves(child, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        rows.append((prefix, value))
    return rows


def _identity(path):
    if not path.is_file():
        return None
    raw = path.read_bytes()
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def _partial_training(output, missing, errors):
    """Recover complete durable presentations when final training receipt is absent."""
    rows, presentations, identities = [], [], {}
    for path in sorted((output / "training/presentations").glob("*.json")):
        identities[str(path.relative_to(output))] = _identity(path)
        receipt = _read(path, missing, errors)
        if receipt is None:
            continue
        if not isinstance(receipt, dict) or not isinstance(receipt.get("training_rows"), list) or not isinstance(receipt.get("presentation"), dict):
            errors.append(dict(path=str(path), error="Incomplete durable presentation receipt"))
            continue
        rows.extend(receipt["training_rows"])
        presentations.append(receipt["presentation"])
    return dict(training_rows=rows, presentation_rows=presentations), identities


def report(plan, output):
    """Read one run root; write its public-safe ``final_report`` directory.

    The caller may pass a plan dictionary or PLAN.json path. A failed/partial
    execution still emits every requested table with explicit missing evidence.
    Raw prompts, canonical IDs, proposal arrays and tensor data are never copied.
    """
    output = Path(output)
    plan_path = None if isinstance(plan, dict) else Path(plan)
    plan = plan if isinstance(plan, dict) else json.loads(plan_path.read_text())
    ids = [str(row["request_id"]) if isinstance(row, dict) else str(row) for row in plan["requests"]]
    destination = output / "final_report"
    destination.mkdir(parents=True, exist_ok=True)
    missing, errors = [], []
    paths = {
        "census": output / "census/CENSUS.json",
        "training": output / "training/RESULT.json",
        "status": output / "STATUS.json",
        "budget": output / "BUDGET.json",
        "result": output / "RESULT.json",
    }
    receipts = {name: _read(path, missing, errors) for name, path in paths.items()}
    census = receipts["census"] or []
    training, status = receipts["training"] or {}, receipts["status"] or {}
    durable_identities = {}
    training_source = "training/RESULT.json"
    if receipts["training"] is None:
        training, durable_identities = _partial_training(output, missing, errors)
        training_source = "durable presentations; final training receipt unavailable"
    rows = training.get("training_rows", [])
    presentations = training.get("presentation_rows", [])
    accounting_available = receipts["budget"] is not None and receipts["result"] is not None
    try:
        result = analyze(
            census, rows, presentations, ids,
            census_complete=bool(status.get("census_complete", False)),
            training_complete=bool(status.get("training_complete", False) and accounting_available
                                   and receipts["training"] is not None),
            expected_natural_starts=plan.get("expected_natural_starts"),
        )
    except (KeyError, TypeError, ValueError) as error:
        errors.append(dict(path="analysis", error=repr(error)))
        result = dict(
            summary=dict(status="INCOMPLETE", gate_decision="INCOMPLETE", gate={},
                         observed_natural_starts=len(census), observed_presentations=len(presentations),
                         expected_presentations=2 * len(ids), scheduled_requests=len(ids),
                         H_observed_count=None, full_natural_denominator=None,
                         interpretation="Analysis rejected malformed or invalid evidence; no gate conclusion."),
            per_request=[], anchor_coverage_rows=[], native_weight_rows=[], training_anchor_rows=[], pairs=[],
        )
    summary = result["summary"]
    phase_result = receipts["result"] or {}
    # This field set is deliberate: private per-response payloads from the
    # execution receipts must not leak through broad dictionary serialization.
    provenance = dict(
        execution_commit=plan.get("execution_commit"),
        checkpoint_identities=plan.get("checkpoints", plan.get("checkpoint_identities", {})),
        manifest_identity=plan.get("manifest_identity"),
        source_manifest_path=plan.get("source_manifest_path"),
        plan_file_identity=_identity(plan_path) if plan_path is not None else None,
        evidence_file_identities={name: _identity(path) for name, path in paths.items()},
        durable_presentation_file_identities=durable_identities,
    )
    costs = dict(
        phase_costs=dict(_numeric_leaves(phase_result.get("phase_costs", {}))),
        work=dict(_numeric_leaves(phase_result.get("work", {}))),
        occupied_gpu_budget=dict(_numeric_leaves(receipts["budget"] or {})),
        declared_gpu_seconds_limit=plan.get("gpu_seconds_limit"),
        unavailable=not accounting_available,
    )
    summary.update(provenance=provenance, costs=costs, evidence_missing=missing,
                   evidence_errors=errors, execution_status=status.get("status", "unavailable"),
                   training_evidence_source=training_source,
                   no_weight_changes=True)
    tables = {
        "PER_REQUEST.csv": result["per_request"],
        "ANCHOR_COVERAGE.csv": result["anchor_coverage_rows"],
        "NATIVE_WEIGHTS.csv": result["training_anchor_rows"],
        "PRESENTATION_WEIGHTS.csv": result["native_weight_rows"],
        "MATCHED_PAIRS.csv": result["pairs"],
    }
    for filename, table in tables.items():
        _csv(destination / filename, table)
    _json(destination / "SUMMARY.json", summary)
    _json(destination / "GATE.json", dict(status=summary["gate_decision"],
                                          thresholds=summary["gate"],
                                          incomplete_reasons=dict(missing=missing, errors=errors)))
    (destination / "REPORT.md").write_text(render(summary))
    return dict(status=summary["status"], gate_decision=summary["gate_decision"],
                report=str(destination / "REPORT.md"), summary=str(destination / "SUMMARY.json"))


def render(s):
    lines = [
        "# K+ lost-repair gate probe", "", "## Material Passport", "",
        "ARS experiment-agent / run and validate. Verification status: ANALYZED from retained execution receipts; no independent native replication.",
        "", f"**Status: {s['status']}. Gate: {s['gate_decision']}.**",
        "Frozen-checkpoint historical replay and read-only training exposure only. Optimizer steps, candidate training and timed endpoint runs are zero by contract; recorded work is listed below.",
        "", f"Execution commit: `{s['provenance'].get('execution_commit') or 'unavailable'}`.",
        "", "## Census and token consequences", "",
        f"Scheduled requests: {s['scheduled_requests']}. Observed natural starts: {s['observed_natural_starts']}. "
        f"Validated full natural-start denominator: {_number(s.get('full_natural_denominator'))}.",
        f"Observed H: {_number(s.get('H_observed_count'))}; M: {_number(s.get('M_observed_count'))}; "
        f"converse newly passed old blocker: {_number(s.get('converse_observed_count'))}.",
        f"H / all natural starts: {_number(s.get('H_share_all_natural_starts'), percent=True)}; "
        f"H / genuine K+6 rejections: {_number(s.get('H_share_genuine_k6_rejections'), percent=True)}.",
        f"H contributing requests: {_number(s.get('H_contributing_requests'))}; "
        f"largest request share: {_number(s.get('H_largest_request_fraction'), percent=True)}.",
        f"Previously attained then lost prefix occurrences, sum(a5−a6) over H: {_number(s.get('H_lost_prefix_total'))}.",
        "", "H requires the same genuine first rejection under K+4 and K+6, with K+5 passing that slot. M requires all three to first reject there. Known seeds are excluded; native EOS/cap boundaries are retained. Other histories, earlier prefix breaks, terminal/all-correct and missing replays remain in private census records and aggregate counts.",
        "", f"History counts: `{json.dumps(s.get('history_counts', {}), sort_keys=True)}`.",
        f"H by one-based depth: `{json.dumps(s.get('H_by_depth', {}), sort_keys=True)}`.",
        f"H a5−a6 histogram: `{json.dumps(s.get('H_lost_prefix_histogram', {}), sort_keys=True)}`.",
        "", "## Scheduled training exposure", "",
        "| Stage | Scheduled occurrences | Distinct anchors | Distinct requests |",
        "|---|---:|---:|---:|",
    ]
    for name, values in s.get("exposure_funnel", {}).items():
        lines.append(f"| {name} | {values['occurrences']} | {values['distinct_anchors']} | {values['distinct_requests']} |")
    lines.extend([
        "", f"Observed response presentations: {s['observed_presentations']} / {s['expected_presentations']}.",
        f"Native H weight before matching: {_number(s.get('native_H_weight_before_matching'))}; "
        f"observed native first-error weight: {_number(s.get('native_first_error_weight'))}; "
        f"observed selected weight: {_number(s.get('selected_weight'))}.",
        f"Full-event native first-error denominator: {_number(s.get('full_native_first_error_weight'))}; "
        f"full-event selected denominator: {_number(s.get('full_selected_weight'))}. "
        f"Training evidence: {s['training_evidence_source']}.",
        f"H weight shares: {_number(s.get('H_weight_fraction_all_first_error'), percent=True)} of native first-error weight; "
        f"{_number(s.get('H_weight_fraction_all_selected'), percent=True)} of selected weight.",
        "", "Support counts each response/start once; weights count every scheduled presentation. Full-response denominators appear once in PRESENTATION_WEIGHTS.csv. NATIVE_WEIGHTS.csv records each sampled occurrence and its actual native coefficient.",
        "The native response-wide FP32 first-error mass is authoritative. The separately summed serialized anchor weights are retained and checked with unchanged native mass tolerances (rtol 2e-6, atol 2e-7); they do not silently replace a missing native denominator.",
        "High means a correct valid immediate successor, low means a wrong valid successor, and ineligible means no valid successor. Ineligible errors keep the native low coefficient and remain in denominators; matching stays within this separate class.",
        "", "## Paper-only matching gate", "",
        "H is ordered by SHA256 of compact JSON [20261114, response ID, canonical start]. Each H receives the nearest available M by wrong-minus-canonical margin within the same presentation/response/depth/priority; ties use the same hash. No M is reused within a stratum. No weight changes are applied.",
        "", "| Screening value | Observed | Threshold | Pass |", "|---|---:|---:|---|",
    ])
    for name, value in s["gate"].items():
        percent = name.startswith("projected_weight_fraction")
        passed = "undecidable (incomplete)" if value["passed"] is None else str(value["passed"])
        lines.append(f"| {name} | {_number(value['value'], percent=percent)} | {_number(value['threshold'], percent=percent)} | {passed} |")
    lines.extend([
        "", f"Projected mass Σ[0.25×min(wH,wM)]: {_number(s.get('projected_matched_weight'))}.",
        f"Unmatched H occurrences: {_number(s.get('H_unmatched_occurrences'))}; reasons: `{json.dumps(s.get('H_unmatched_reasons', {}), sort_keys=True)}`.",
        f"Margin balance: `{json.dumps(s.get('margin_balance', {}), sort_keys=True)}`.",
        "", "The 32-anchor, 16-request and 0.25% thresholds screen this scheduled batch only. A pass ends at this report and warrants a separate mechanism discussion; it does not authorize training or establish future-request recovery, acceptance gains or serving speedup.",
        "", "## Completeness, parity and costs", "",
        f"Execution status: {s['execution_status']}; missing replays: {_number(s.get('missing_replays'))}; invalid replays: {_number(s.get('invalid_replays'))}.",
        "A missing full census or training denominator makes the gate INCOMPLETE. Observed counts are measured numerators, not estimates of missing work. Null ratios are unavailable, never zero.",
        "", "| Recorded receipt | Value |", "|---|---:|",
    ])
    for section in ("phase_costs", "work", "occupied_gpu_budget"):
        for name, value in s["costs"][section].items():
            lines.append(f"| {section}.{name} | {value} |")
    if s["costs"]["unavailable"]:
        lines.append("| Complete cost receipts | unavailable |")
    lines.extend([
        "", f"Declared aggregate occupied-GPU ceiling: {s['costs']['declared_gpu_seconds_limit']} seconds. Nested phase clocks are not added. This probe contains no timed serving comparison.",
        "", "Missing evidence: " + ("; ".join(s["evidence_missing"]) or "none") + ".",
        "Analysis/read errors: " + (json.dumps(s["evidence_errors"]) if s["evidence_errors"] else "none") + ".",
        "", "## Source and checkpoint identities", "", "```json",
        json.dumps(s["provenance"], indent=2, ensure_ascii=False), "```", "",
        "Public tables contain identifiers and measurements only. Raw prompts, token arrays, reconstruction receipts and tensors remain in the private run directory. Labels and denominators require the separately recorded independent reduction before final scientific sign-off.",
        "", "Interpretation boundary: recurrence frequency and training exposure are different questions. Frequent H with little scheduled exposure identifies an exposure gap; coverage lost only at matching identifies weak support for this comparison. Historical fixed-start token differences cannot be converted to complete-request speed or assumed recoverable future gain.",
    ])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True, help="Run root containing census/training receipts")
    args = parser.parse_args()
    print(json.dumps(report(args.plan, args.output), indent=2))


if __name__ == "__main__":
    main()
