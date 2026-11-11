#!/usr/bin/env python3
"""Direct-loop matched-start audit, outside the scientific execution checkout.

Authorship disclosure: the same reviewer authored production matched_start.py.
This is a separately written arithmetic/record audit, not a blinded independent
authorship check or another model run. No production helper is imported.
"""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path


ARMS = ("K", "TRUE", "SHUFFLED")
STRATA = (
    "successor_high",
    "successor_low",
    "successor_ineligible",
    "all_correct",
    "seed_eos",
)
MATCHED_PREFIX = "kplus-successor-control-v1|matched-start|gsm8k|test|"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fingerprint(path):
    payload = Path(path).read_bytes()
    return {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


class Inputs:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.original_root = None
        self.files = {}

    def path(self, value):
        path = Path(value)
        if not path.is_absolute():
            return self.root / path
        if self.original_root is not None:
            try:
                return self.root / path.relative_to(self.original_root)
            except ValueError:
                pass
        return path

    def read(self, value):
        path = self.path(value)
        bound = fingerprint(path)
        if str(path) in self.files:
            require(self.files[str(path)] == bound, "Input changed between reads")
        self.files[str(path)] = bound
        return json.loads(path.read_text())

    def check_stable(self):
        for name, expected in self.files.items():
            require(fingerprint(name) == expected, "Input changed during audit: " + name)


def inspect_first_block(record, request, canonical, checkpoint, *, parent=False):
    """Walk proposal equality against canonical slots; retain native acceptance."""
    prompt = request["prompt_ids"]
    response = canonical["response_ids"]
    stops = set(request["stop_token_ids"])
    require(bool(response) and canonical["prompt_ids"] == prompt, "Invalid canonical input")
    require(record["prompt_ids"] == prompt, "Raw prompt differs")
    require(str(record["checkpoint_id"]) == str(checkpoint), "Wrong raw checkpoint")
    require(record["response_ids"] and record["response_ids"][0] == response[0], "Wrong target seed")
    if parent:
        require(record["response_ids"] == response[: len(record["response_ids"])], "Parent prefix differs")
    else:
        require(record["response_ids"] == response, "Child canonical output differs")
        require(record["finish_reason"] == canonical["finish_reason"], "Child termination differs")
    trace = record["metadata"]["trace"]
    if "capture" in record["metadata"]:
        require(trace == record["metadata"]["capture"]["blocks"], "Trace and native capture differ")
    if parent:
        require(len(trace) <= 1, "Parent exceeded one verification block")
    if not trace:
        require(len(response) == 1 and response[0] in stops, "Absent round without seed EOS")
        return dict(first_error=None, prefix=0, matches=[], stratum="seed_eos", native_prefix=0)
    first = trace[0]
    require(first["output_start"] == 0 and first["anchor_token"] == response[0], "Wrong first start")
    proposals = first["draft_tokens"]
    require(len(proposals) == 15, "Not the fixed15-proposal native block")
    require(all(type(token) is int and token >= 0 for token in proposals), "Malformed proposals")
    labels = []
    for token in response[1:16]:
        labels.append(token)
        if token in stops:
            break
    require(bool(labels), "Unexpected block after seed EOS")
    matches = []
    for index in range(len(labels)):
        matches.append(proposals[index] == labels[index])
    first_error = None
    prefix = 0
    for index, correct in enumerate(matches):
        if not correct:
            first_error = index
            break
        prefix += 1
    if first_error is None:
        stratum = "all_correct"
    elif first_error + 1 == len(labels):
        stratum = "successor_ineligible"
    elif matches[first_error + 1]:
        stratum = "successor_high"
    else:
        stratum = "successor_low"
    native_prefix = first["accepted_draft_tokens"]
    raw_prefix = first["raw_accepted_draft_tokens_before_stop"]
    require(type(native_prefix) is int and 0 <= native_prefix <= len(labels), "Invalid native prefix")
    require(type(raw_prefix) is int and native_prefix <= raw_prefix <= 15, "Invalid raw native prefix")
    return dict(first_error=first_error, prefix=prefix, matches=matches,
                stratum=stratum, native_prefix=native_prefix)


def compare_direct(parent, child):
    """Keep prespecified canonical counts separate from extra verification checks."""
    error = parent["first_error"]
    require(len(parent["matches"]) == len(child["matches"]), "Valid canonical slots differ")
    prior_correct = len(parent["matches"]) if error is None else error
    prefix_intact = True
    for position in range(prior_correct):
        if not child["matches"][position]:
            prefix_intact = False
    canonical_repair = False
    if error is not None:
        canonical_repair = prefix_intact and child["matches"][error]
    verified_repair = bool(canonical_repair and child["native_prefix"] > error)
    delta = child["prefix"] - parent["prefix"]
    native_delta = child["native_prefix"] - parent["native_prefix"]
    return dict(
        parent_first_error=error,
        valid_proposal_slots=len(parent["matches"]),
        old_error_present=error is not None,
        old_first_error_repaired_with_prefix_intact=canonical_repair,
        earlier_correct_prefix_broken=not prefix_intact,
        parent_accepted_prefix=parent["prefix"],
        child_accepted_prefix=child["prefix"],
        prefix_delta=delta,
        prefix_tokens_gained=max(delta, 0),
        prefix_tokens_lost=max(-delta, 0),
        tie=delta == 0,
        canonical_prefix_advance=delta > 0,
        native_prefix_advance=native_delta > 0,
        verified_repair=verified_repair,
        canonical_repair_without_native_verification=canonical_repair and not verified_repair,
        native_advance_without_verified_repair=native_delta > 0 and not verified_repair,
        parent_native_prefix=parent["native_prefix"],
        child_native_prefix=child["native_prefix"],
        native_prefix_delta=native_delta,
        native_prefix_tokens_gained=max(native_delta, 0),
        native_prefix_tokens_lost=max(-native_delta, 0),
        parent_canonical_native_prefix_disagreement=parent["prefix"] != parent["native_prefix"],
        child_canonical_native_prefix_disagreement=child["prefix"] != child["native_prefix"],
        prefix_preservation_denominator=prior_correct > 0,
    )


def aggregate(rows):
    output = []
    for stratum in STRATA:
        for arm in ARMS:
            selected = [row for row in rows if row["parent_stratum"] == stratum and row["method"] == arm]
            values = dict(parent_stratum=stratum, method=arm, requests=len(selected),
                          old_error_denominator=0, repair_count=0, prefix_preservation_denominator=0,
                          prefix_broken_count=0, prefix_tokens_gained=0, prefix_tokens_lost=0,
                          wins=0, losses=0, ties=0, verified_repair_count=0,
                          native_prefix_advances=0, canonical_repair_without_native_verification=0,
                          native_advance_without_verified_repair=0,
                          native_prefix_tokens_gained=0, native_prefix_tokens_lost=0,
                          parent_canonical_native_prefix_disagreements=0,
                          child_canonical_native_prefix_disagreements=0)
            for row in selected:
                values["old_error_denominator"] += row["old_error_present"]
                values["repair_count"] += row["old_first_error_repaired_with_prefix_intact"]
                values["prefix_preservation_denominator"] += row["prefix_preservation_denominator"]
                values["prefix_broken_count"] += row["earlier_correct_prefix_broken"]
                for name in ("prefix_tokens_gained", "prefix_tokens_lost"):
                    values[name] += row[name]
                values["wins"] += row["prefix_delta"] > 0
                values["losses"] += row["prefix_delta"] < 0
                values["ties"] += row["tie"]
                values["verified_repair_count"] += row["verified_repair"]
                values["native_prefix_advances"] += row["native_prefix_advance"]
                values["canonical_repair_without_native_verification"] += row["canonical_repair_without_native_verification"]
                values["native_advance_without_verified_repair"] += row["native_advance_without_verified_repair"]
                values["native_prefix_tokens_gained"] += row["native_prefix_tokens_gained"]
                values["native_prefix_tokens_lost"] += row["native_prefix_tokens_lost"]
                values["parent_canonical_native_prefix_disagreements"] += row["parent_canonical_native_prefix_disagreement"]
                values["child_canonical_native_prefix_disagreements"] += row["child_canonical_native_prefix_disagreement"]
            output.append(values)
    return output


def reconcile(reported, rows, groups, strata):
    require(reported["requests"] == 64 and reported["parent_strata"] == strata, "Reported denominators differ")
    by_key = {(row["request_id"], row["method"]): row for row in rows}
    require(len(reported["rows"]) == len(by_key) == 192, "Reported matched row coverage differs")
    checks = 0
    seen = set()
    for row in reported["rows"]:
        key = row["request_id"], row["method"]
        require(key not in seen and key in by_key, "Duplicate or foreign reported row")
        seen.add(key)
        for name, value in row.items():
            require(name in by_key[key] and by_key[key][name] == value, "Reported matched row differs: " + str(key) + ":" + name)
            checks += 1
    expected_groups = {(row["parent_stratum"], row["method"]): row for row in groups if row["requests"]}
    require(len(reported["groups"]) == len(expected_groups), "Reported group coverage differs")
    seen_groups = set()
    for row in reported["groups"]:
        key = row["parent_stratum"], row["method"]
        require(key not in seen_groups and key in expected_groups, "Duplicate or foreign reported group")
        seen_groups.add(key)
        for name, value in row.items():
            require(expected_groups[key].get(name) == value, "Reported matched group differs: " + str(key) + ":" + name)
            checks += 1
    return dict(status="passed", fields_compared=checks)


def audit(root, reported=None):
    inputs = Inputs(root)
    plan = inputs.read("sealed/PLAN.json")
    inputs.original_root = Path(plan["root"])
    panel = inputs.read(plan["panel_manifest"])
    subset = inputs.read(plan["matched_panel"])
    canonical = inputs.read(plan["references"])
    checkpoints = inputs.read("CHECKPOINTS.json")
    request_by_id = {row["request_id"]: row for row in panel["requests"]}
    require(len(request_by_id) == len(panel["requests"]), "Duplicate panel identity")
    selected = sorted(panel["requests"], key=lambda row: (
        hashlib.sha256((MATCHED_PREFIX + str(row["source_row"])).encode("utf-8")).digest(), row["source_row"]))[:64]
    require(len(subset["requests"]) == len(selected), "Changed input-only64 subset size")
    for index, (matched, original) in enumerate(zip(subset["requests"], selected)):
        expected = dict(original)
        expected["matched_start_ordinal"] = index
        expected["matched_selection_sha256"] = hashlib.sha256(
            (MATCHED_PREFIX + str(original["source_row"])).encode("utf-8")
        ).hexdigest()
        require(matched == expected, "Changed input-only64 subset")
    require(len(selected) == 64 and len({row["request_id"] for row in selected}) == 64, "Matched panel coverage differs")
    rows = []
    strata = Counter({name: 0 for name in STRATA})
    for index, request in enumerate(selected):
        rid = request["request_id"]
        require(request_by_id[rid] == request, "Changed selected input")
        ref = canonical[rid]
        require(ref["binding_digest"] == plan["canonical_binding"], "Wrong target/runtime canonical binding")
        parent_record = inputs.read(f"matched_start/parent_{index:03d}.json")
        parent_native = inputs.read(f"matched_start/parent_{index:03d}_trace.json")
        require(parent_record["record_id"] == rid, "Wrong parent identity")
        require(parent_record["metadata"]["capture"] == parent_native, "Parent native sidecar changed")
        parent = inspect_first_block(parent_record, request, ref, plan["origin_export"], parent=True)
        strata[parent["stratum"]] += 1
        for arm in ARMS:
            observed = []
            for repeat in (0, 1):
                label = f"{request['panel_ordinal']:04d}_r{repeat}_{arm}"
                raw = inputs.read("evaluation/" + label + ".json")
                require(raw["record_id"] == label, "Wrong child identity")
                native = inputs.read(raw["metadata"]["native_trace_path"])
                require(native == raw["metadata"]["capture"], "Child native sidecar changed")
                observed.append(inspect_first_block(raw, request, ref, checkpoints[arm]["checkpoint_path"]))
            require(observed[0] == observed[1], "Child first-round repeats differ")
            row = dict(request_id=rid, method=arm, parent_stratum=parent["stratum"])
            row.update(compare_direct(parent, observed[0]))
            rows.append(row)
    groups = aggregate(rows)
    parent_status = inputs.read("matched_start/STATUS.json")
    for field in ("planned_requests", "attempted_requests", "returned_requests", "validated_requests", "maximum_draft_blocks"):
        require(parent_status[field] == 64, "Parent coverage/budget differs: " + field)
    for field in ("complete_response_generations", "unattempted_requests", "unknown_requests", "returned_without_validation"):
        require(parent_status[field] == 0, "Parent incomplete or extra work: " + field)
    require(parent_status["status"] == "completed" and parent_status["draft_blocks"] == 64 - strata["seed_eos"], "Parent block accounting differs")
    comparison = dict(status="not_requested")
    if reported is not None:
        comparison = reconcile(inputs.read(reported), rows, groups, dict(strata))
    inputs.check_stable()
    return dict(status="passed", execution_source=plan["execution_source"], requests=64,
                parent_strata=dict(strata), rows=rows, groups=groups, production_reconciliation=comparison,
                files=inputs.files, model_work=0, production_imports=False,
                authorship="Same reviewer authored production matched_start.py; separate direct-loop arithmetic, not blinded independent authorship.",
                prespecified="Canonical proposal repair/prefix-breaking/gains/losses and parent true-bit denominators.",
                additional_diagnostics="Native accepted-prefix advance and verified repair are supplemental, not replacement endpoint definitions.",
                limit="Shared first conditioning prefix only; no inference about full-request speedup or causal mediation.")


def self_test():
    parent = dict(first_error=1, prefix=1, matches=[True, False, True], stratum="successor_high", native_prefix=1)
    fixed = dict(first_error=None, prefix=3, matches=[True, True, True], stratum="all_correct", native_prefix=3)
    require(compare_direct(parent, fixed)["verified_repair"], "Repair fixture failed")
    unverified = dict(fixed, native_prefix=1)
    require(compare_direct(parent, unverified)["canonical_repair_without_native_verification"], "Advance/repair separation failed")
    broken = dict(first_error=0, prefix=0, matches=[False, True, True], stratum="successor_low", native_prefix=0)
    require(compare_direct(parent, broken)["earlier_correct_prefix_broken"], "Prefix break fixture failed")
    require(not compare_direct(parent, broken)["verified_repair"], "Broken prefix counted as repair")
    tied = compare_direct(fixed, fixed)
    require(tied["tie"] and not tied["old_error_present"] and not tied["verified_repair"], "All-correct tie fixture failed")
    lost = compare_direct(fixed, parent)
    require(lost["prefix_tokens_lost"] == 2 and lost["earlier_correct_prefix_broken"], "All-correct loss fixture failed")
    acceptance_only = compare_direct(dict(parent, native_prefix=0), parent)
    require(acceptance_only["native_advance_without_verified_repair"] and acceptance_only["tie"], "Native advance mislabeled repair")
    request = dict(prompt_ids=[1], stop_token_ids=[99])
    reference = dict(prompt_ids=[1], response_ids=[10, 11, 99], finish_reason="eos")
    raw = dict(prompt_ids=[1], response_ids=[10, 11, 99], finish_reason="eos", checkpoint_id="parent",
               metadata=dict(trace=[dict(output_start=0, anchor_token=10, draft_tokens=[7, 99] + [0] * 13,
                                        accepted_draft_tokens=0, raw_accepted_draft_tokens_before_stop=0)]))
    require(inspect_first_block(raw, request, reference, "parent")["stratum"] == "successor_high", "EOS successor eligibility failed")
    raw["metadata"]["trace"][0]["draft_tokens"][:2] = [11, 7]
    require(inspect_first_block(raw, request, reference, "parent")["stratum"] == "successor_ineligible", "Post-EOS successor exclusion failed")
    reference["response_ids"] = [99]
    raw["response_ids"] = [99]
    raw["metadata"]["trace"] = []
    seed = inspect_first_block(raw, request, reference, "parent")
    require(seed["stratum"] == "seed_eos", "Seed EOS fixture failed")
    seed_comparison = compare_direct(seed, seed)
    require(seed_comparison["tie"] and not seed_comparison["prefix_preservation_denominator"], "Seed EOS denominator failed")
    row = dict(request_id="fixture", method="TRUE", parent_stratum="all_correct", **tied)
    grouped = aggregate([row])
    selected = [group for group in grouped if group["requests"]]
    require(len(grouped) == 15 and len(selected) == 1 and selected[0]["ties"] == 1
            and selected[0]["old_error_denominator"] == 0
            and selected[0]["prefix_preservation_denominator"] == 1, "Stratum denominators failed")
    return dict(status="passed", checks=12, production_imports=False, model_work=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?")
    parser.add_argument("--reported")
    parser.add_argument("--output")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), sort_keys=True))
        return
    require(args.root and args.output, "Provide finalized local root and a new output directory")
    result = audit(args.root, args.reported)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "MATCHED_START_AUDIT.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for name, records in (("MATCHED_ROWS.csv", result["rows"]), ("MATCHED_GROUPS.csv", result["groups"])):
        with (output / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    print(json.dumps({key: result[key] for key in ("status", "requests", "parent_strata", "production_reconciliation", "model_work")}))


if __name__ == "__main__":
    main()
