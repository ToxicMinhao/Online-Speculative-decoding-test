#!/usr/bin/env python3
"""Independent CPU scalar audit of retained event-seven evidence.

This standalone script imports no experiment implementation or reduction
helpers. It does not rerun models or bootstrap intervals. The raw-block checks
supplement, rather than replace, the campaign's source/runtime validity gates.
"""
import argparse
from collections import Counter, defaultdict
import csv
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

ARMS = ("NATIVE", "RECENT", "CONTROL")
TIMES = ("request_latency_s", "native_generate_s", "cache_maintenance_s")
COUNTS = ("verification_calls", "accepted_draft_tokens", "output_tokens", "committed_progress")


class Audit:
    def __init__(self, root, report):
        self.root, self.report = Path(root), Path(report)
        self.errors, self.missing, self.files = [], [], {}
        self.checks = 0

    def check(self, condition, label):
        self.checks += 1
        if not condition:
            self.errors.append(label)

    def load(self, path, *, lines=False, table=False, default=None):
        path = Path(path)
        if not path.exists():
            self.missing.append(str(path))
            return default
        payload = path.read_bytes()
        self.files[str(path)] = dict(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
        if table:
            return list(csv.DictReader(payload.decode().splitlines()))
        if lines:
            return [json.loads(line) for line in payload.splitlines() if line.strip()]
        return json.loads(payload)

    def same(self, actual, expected, label):
        if isinstance(expected, float):
            try:
                equal = math.isfinite(float(actual)) and math.isclose(float(actual), expected, rel_tol=1e-12, abs_tol=1e-12)
            except (ValueError, TypeError):
                equal = False
        elif expected is None:
            equal = actual is None or actual == ""
        elif isinstance(expected, bool):
            equal = actual is expected or actual == str(expected)
        elif isinstance(expected, int):
            try:
                equal = float(actual) == expected
            except (ValueError, TypeError):
                equal = False
        else:
            equal = actual == expected
        self.check(equal, label)


def validate_raw(audit, raw, saved, request, canonical, checkpoint):
    """Rebuild clipped acceptance from proposals/posteriors and committed tokens."""
    label = f"raw:{saved['request_id']}:{saved['method']}:{saved['repeat']}"
    before = len(audit.errors)
    audit.check(saved.get("valid_measurement") is True and not saved.get("errors")
        and not saved.get("structural_errors") and saved.get("validation_status") == "valid", label + ":saved_validity")
    audit.same(raw["prompt_ids"], request["prompt_ids"], label + ":prompt")
    audit.same(raw["response_ids"], canonical["response_ids"], label + ":tokens")
    audit.same(raw["finish_reason"], canonical["finish_reason"], label + ":termination")
    audit.same(raw["checkpoint_id"], checkpoint, label + ":checkpoint")
    metadata, response = raw["metadata"], raw["response_ids"]
    capture, trace = metadata["capture"], metadata["trace"]
    audit.check(not metadata["feedback_capture_in_timed_request"], label + ":capture_disabled")
    audit.same(capture["blocks"], trace, label + ":inline_native_blocks")
    audit.same(capture["prompt_ids"], raw["prompt_ids"], label + ":native_prompt")
    audit.same(capture["response_ids"], response, label + ":native_response")
    accepted, progress, native_accepted = 0, 0, 0
    for index, block in enumerate(trace):
        block_label = label + f":block{index}"
        proposed, posterior = block["draft_tokens"], block["target_argmax"]
        width, committed = block["verify_size"], block["committed_tokens"]
        advanced = block["committed_progress"]
        audit.check(type(width) is int and width == len(proposed) + 1 == len(posterior) and 1 <= width <= 16,
                    block_label + ":width")
        audit.check(type(advanced) is int and 1 <= advanced <= width, block_label + ":advance")
        rejected = next((j for j, token in enumerate(proposed) if token != posterior[j]), len(proposed))
        expected_commit = proposed[:rejected] + [posterior[rejected]]
        clipped = min(rejected, advanced)
        audit.check(advanced <= rejected + 1, block_label + ":past_rejection")
        audit.same(block["output_start"], progress, block_label + ":contiguous_start")
        audit.same(block["anchor_token"], response[progress], block_label + ":anchor")
        audit.same(committed, expected_commit[:advanced], block_label + ":posterior_commit")
        audit.same(committed, response[progress + 1:progress + 1 + advanced], block_label + ":canonical_commit")
        audit.same(block["accepted_draft_tokens"], clipped, block_label + ":clipped_accepted")
        audit.same(block["native_accepted_draft_tokens"], advanced - 1, block_label + ":native_accepted")
        audit.same(block["raw_accepted_draft_tokens_before_stop"], rejected, block_label + ":raw_accepted")
        audit.same(block["stop_clipped"], advanced < rejected + 1, block_label + ":stop_clip")
        accepted += clipped
        native_accepted += advanced - 1
        progress += advanced
    audit.same(progress, len(response) - 1, label + ":full_progress")
    if raw["finish_reason"] == "eos":
        audit.check(bool(response) and response[-1] in request["stop_token_ids"], label + ":eos")
    else:
        audit.check(raw["finish_reason"] == "length" and len(response) == request["max_new_tokens"], label + ":cap")
    result = dict(request_id=saved["request_id"], method=saved["method"], repeat=saved["repeat"],
        verification_calls=len(trace), accepted_draft_tokens=accepted, committed_progress=progress,
        output_tokens=len(response), native_accepted_draft_tokens=native_accepted, finish_reason=raw["finish_reason"],
        request_latency_s=float(metadata["request_latency_s"]), native_generate_s=float(metadata["native_generate_s"]),
        cache_maintenance_s=float(capture["cache_maintenance_s"]))
    for key in ("verification_calls", "accepted_draft_tokens", "committed_progress"):
        audit.same(metadata[key], result[key], label + ":aggregate_" + key)
    for key in TIMES:
        audit.check(math.isfinite(result[key]) and result[key] >= 0, label + ":finite_" + key)
    for key, value in result.items():
        audit.same(saved.get(key), value, label + ":measurement_" + key)
    result["independently_valid"] = len(audit.errors) == before
    return result


def serving(audit):
    folder = audit.root / "evaluation"
    requests = audit.load(folder / "REQUESTS.json", default=[])
    canonical = audit.load(folder / "CANONICAL.json", default={})
    exports = audit.load(audit.root / "EXPORTS.json", default={})
    recorded = audit.load(folder / "MEASUREMENTS.jsonl", lines=True, default=[])
    planned_ids = [r["request_id"] for r in requests]
    request_by_id = {r["request_id"]: r for r in requests}
    audit.check(len(request_by_id) == len(requests), "serving:unique_requests")
    validated, seen = [], set()
    for saved in recorded:
        key = saved["request_id"], saved["method"], saved["repeat"]
        audit.check(key not in seen, "serving:duplicate_execution:" + repr(key))
        seen.add(key)
        if key[0] not in request_by_id or key[1] not in ARMS or key[2] not in (0, 1):
            audit.check(False, "serving:unplanned_execution:" + repr(key))
            continue
        label = f"{saved['panel_ordinal']:04d}_r{key[2]}_{key[1]}"
        raw = audit.load(folder / (label + ".json"))
        if raw is None:
            continue
        try:
            validated.append(validate_raw(audit, raw, saved, request_by_id[key[0]], canonical[key[0]], exports[key[1]]))
        except (KeyError, TypeError, ValueError, IndexError) as error:
            audit.check(False, "serving:malformed_raw:" + repr(key) + ":" + repr(error))
    grouped = defaultdict(list)
    for row in validated:
        grouped[row["request_id"], row["method"]].append(row)
    paired = []
    for rid in planned_ids:
        candidate, valid = [], True
        for arm in ARMS:
            repeats = sorted(grouped[rid, arm], key=lambda r: r["repeat"])
            if len(repeats) != 2 or [r["repeat"] for r in repeats] != [0, 1]:
                valid = False
                continue
            consistency = all(repeats[0][key] == repeats[1][key]
                for key in (*COUNTS, "native_accepted_draft_tokens", "finish_reason"))
            audit.check(consistency, f"serving:repeat_consistency:{rid}:{arm}")
            valid &= consistency and all(r["independently_valid"] for r in repeats)
            candidate.append(dict(request_id=rid, method=arm, **{key: repeats[0][key] for key in (*COUNTS, "finish_reason")},
                **{key: math.fsum(r[key] for r in repeats) / 2 for key in TIMES}))
        if valid and len(candidate) == 3:
            paired.extend(candidate)
    totals = {}
    for arm in ARMS:
        selected = [r for r in paired if r["method"] == arm]
        totals[arm] = dict(requests=len(selected), **{key: sum(r[key] for r in selected) for key in COUNTS},
                          **{key: math.fsum(r[key] for r in selected) for key in TIMES})
    complete = len(recorded) == 6 * len(requests) and len(paired) == 3 * len(requests) and bool(requests)
    return dict(planned_requests=len(requests), planned_generations=6 * len(requests), recorded_generations=len(recorded),
        raw_returned=len(validated), invalid_raw=sum(not r["independently_valid"] for r in validated),
        unavailable_or_unattempted=max(0, 6 * len(requests) - len(validated)), complete=complete,
        paired_requests=len(paired) // 3, totals=totals, paired=paired, repeats=validated)


def assignments(audit):
    data = audit.load(audit.root / "ASSIGNMENTS.json", default={})
    if not data:
        return dict(status="unavailable")
    cells, rows = {}, data["arms"]
    unique = {}
    for arm in ("RECENT", "CONTROL"):
        count, weight, identifiers = Counter(), defaultdict(Fraction), set()
        for row in rows[arm]:
            key = (int(row["presentation_id"]), row["request_id"], int(row["canonical_start"]), int(row["proposal_slot"]))
            audit.check(key not in identifiers, "matching:duplicate_slot:" + arm + ":" + repr(key))
            identifiers.add(key)
            cell = key[0], key[3]
            count[cell] += 1
            weight[cell] += Fraction(float(row["weight"]))
            audit.same(row["floor"], 0.5 * min(float(row["margin"]), 1), "matching:own_parent_floor:" + arm + ":" + repr(key))
        cells[arm] = count, dict(weight)
        unique[arm] = identifiers
    audit.check(cells["RECENT"] == cells["CONTROL"], "matching:presentation_depth_count_and_weight_invariant")
    destinations, sources = set(), set()
    for pair in data.get("matches", []):
        destination, source = pair["destination_slot_id"], pair["source_slot_id"]
        audit.check(destination not in destinations, "matching:destination_reused:" + destination)
        audit.check(source not in sources, "matching:source_reused:" + source)
        destinations.add(destination)
        sources.add(source)
    raw_weight = float(sum(cells["RECENT"][1].values(), Fraction(0)))
    audit.same(data["summary"]["recent_occurrences"], len(rows["RECENT"]), "matching:summary_occurrences")
    audit.same(data["summary"]["assigned_raw_weight"], raw_weight, "matching:summary_weight")
    audit.same(data["summary"]["matched_distinct_slots"], len(sources), "matching:summary_switches")
    return dict(status="checked", occurrences={arm: len(rows[arm]) for arm in rows},
        presentation_depth_cells=len(cells["RECENT"][0]), assigned_raw_weight=raw_weight,
        switched_distinct_slots=len(sources), destination_reuse_max=1 if destinations else 0,
        rows=rows, matches=data.get("matches", []))


def compare_tables(audit, serving_result, placement, summary):
    paired_table = audit.load(audit.report / "per_request.csv", table=True, default=[])
    repeats_table = audit.load(audit.report / "all_repeats.csv", table=True, default=[])
    assignments_table = audit.load(audit.report / "assigned_slots.csv", table=True, default=[])
    matching_table = audit.load(audit.report / "matching.csv", table=True, default=[])
    def table_index(rows, fields):
        output = {tuple(str(row[key]) for key in fields): row for row in rows}
        audit.check(len(output) == len(rows), "table:unique:" + repr(fields))
        return output
    if summary.get("complete_valid_comparison"):
        index = table_index(paired_table, ("request_id", "method"))
        audit.same(len(index), len(serving_result["paired"]), "table:paired_count")
        for row in serving_result["paired"]:
            saved = index.get((row["request_id"], row["method"]), {})
            for key, value in row.items():
                audit.same(saved.get(key), value, "table:paired:" + row["request_id"] + ":" + row["method"] + ":" + key)
    else:
        audit.check(not paired_table and not summary.get("comparisons"), "table:incomplete_must_not_claim_gain")
    index = table_index(repeats_table, ("request_id", "method", "repeat"))
    for row in serving_result["repeats"]:
        saved = index.get((row["request_id"], row["method"], str(row["repeat"])), {})
        for key in (*COUNTS, *TIMES, "finish_reason", "native_accepted_draft_tokens"):
            audit.same(saved.get(key), row[key], "table:repeat:" + row["request_id"] + ":" + row["method"] + ":" + key)
    index = table_index(assignments_table, ("arm", "presentation_id", "request_id", "canonical_start", "proposal_slot"))
    expected = sum(len(rows) for rows in placement.get("rows", {}).values())
    audit.same(len(index), expected, "table:assignment_count")
    for arm, rows in placement.get("rows", {}).items():
        for row in rows:
            key = (arm, str(row["presentation_id"]), row["request_id"], str(row["canonical_start"]), str(row["proposal_slot"]))
            for field in ("weight", "margin", "floor"):
                audit.same(index.get(key, {}).get(field), float(row[field]), "table:assignment:" + repr(key) + ":" + field)
    index = table_index(matching_table, ("source_slot_id", "destination_slot_id", "presentation_id"))
    expected = sum(len(pair["presentation_ids"]) for pair in placement.get("matches", []))
    audit.same(len(index), expected, "table:matched_presentation_count")
    for pair in placement.get("matches", []):
        for ordinal, pid in enumerate(pair["presentation_ids"]):
            key = pair["source_slot_id"], pair["destination_slot_id"], str(pid)
            audit.same(index.get(key, {}).get("source_margin"), pair["source_margins"][ordinal], "table:matching_source_margin:" + repr(key))
            audit.same(index.get(key, {}).get("destination_margin"), pair["destination_margins"][ordinal], "table:matching_control_margin:" + repr(key))
    if not summary.get("complete_valid_comparison"):
        return
    for comparator in ("NATIVE", "CONTROL"):
        value = summary["comparisons"][comparator]
        a, b = serving_result["totals"]["RECENT"], serving_result["totals"][comparator]
        for prefix, totals in (("recent", a), ("reference", b)):
            for field, name in (("verification_calls", "calls"), ("accepted_draft_tokens", "accepted"), ("request_latency_s", "request_s")):
                audit.same(value[prefix + "_" + name], totals[field], "summary:" + comparator + ":" + prefix + "_" + name)
        gain = 100 * ((a["accepted_draft_tokens"] / a["verification_calls"]) /
                      (b["accepted_draft_tokens"] / b["verification_calls"]) - 1) if a["verification_calls"] and b["accepted_draft_tokens"] else None
        audit.same(value["accepted_length_relative_gain_pct"], gain, "summary:" + comparator + ":accepted_gain")
        audit.same(value["time_reduction_pct"], 100 * (1 - a["request_latency_s"] / b["request_latency_s"]),
                   "summary:" + comparator + ":time_reduction")


def run(root, report=None):
    audit = Audit(root, report or Path(root) / "final_report")
    status = audit.load(audit.root / "STATUS.json", default={})
    summary = audit.load(audit.report / "SUMMARY.json", default={})
    placement = assignments(audit)
    if status.get("status") == "completed_no_treatment":
        audit.check(not placement.get("rows", {}).get("RECENT"), "zero_treatment:eligible_rows")
        audit.check(not summary.get("complete_valid_comparison") and not summary.get("comparisons"), "zero_treatment:gain_claim")
        evidence = dict(complete=False, paired_requests=0, totals={})
        result_status = "passed_no_treatment" if not audit.errors and not audit.missing else "failed"
    else:
        evidence = serving(audit)
        compare_tables(audit, evidence, placement, summary)
        complete = status.get("status") == "completed" and evidence["complete"] and summary.get("complete_valid_comparison")
        if summary.get("complete_valid_comparison"):
            audit.check(complete, "summary:claims_complete_with_missing_or_invalid_work")
        result_status = "failed" if audit.errors else "passed" if complete and not audit.missing else "incomplete"
    return dict(status=result_status, campaign_status=status.get("status", "unavailable"), checks=audit.checks,
        errors=audit.errors, missing_files=audit.missing, input_identities=audit.files,
        serving={key: value for key, value in evidence.items() if key not in ("paired", "repeats")},
        matching={key: value for key, value in placement.items() if key not in ("rows", "matches")},
        model_work=0, optimizer_steps=0,
        numerical_comparison="Integer counts exact; independently summed binary64 means compared at 1e-12 absolute/relative report-rounding tolerance.",
        scope="Independent stdlib raw-block/scalar reduction and placement invariants; no shared experiment helpers, no new model run, no CI recomputation.",
        disclosure="Same research team; this auditor also authored preparation/matching code. Independent reduction is not independent GPU replication.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run(args.root, args.report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(result["status"], "checks", result["checks"], "errors", len(result["errors"]), "missing", len(result["missing_files"]))
    raise SystemExit(0 if result["status"] in ("passed", "passed_no_treatment") else 1)
