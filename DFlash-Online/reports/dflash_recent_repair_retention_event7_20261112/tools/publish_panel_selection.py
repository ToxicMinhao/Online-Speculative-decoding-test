#!/usr/bin/env python3
"""Publish input-only panel selection without exposing prompts or token arrays.

This post-run metadata helper is separate from the scientific implementation.
It reads the sealed public-format panel manifest, independently checks its
population and ordering, and emits an allowlisted candidate/exclusion ledger.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re


REVISION = "740312add88f781978c0658806c59bc2815b9866"
SELECTION_PREFIX = "retention-e7-v1|"
PATH_PREFIX = "retention-e7-path|"
HASH_KEYS = {"normalized_question_sha256", "prompt_ids_sha256", "rendered_prompt_sha256"}


def identity(path):
    content = Path(path).read_bytes()
    return dict(bytes=len(content), sha256=hashlib.sha256(content).hexdigest())


def stable_hash(prefix, source_id):
    return hashlib.sha256((prefix + source_id).encode("utf-8")).hexdigest()


def safe_id(value):
    """Only source/request/reason tokens, never free text or filesystem paths."""
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9_.:-]+", value) is None:
        raise ValueError("Non-identifier in a public identifier field")
    return value


def safe_matches(matches):
    result = []
    for match in matches:
        keys = []
        for key, value in match["shared_keys"]:
            if key in HASH_KEYS:
                if re.fullmatch(r"[a-f0-9]{64}", value) is None:
                    raise ValueError("Malformed public identity hash")
            elif key == "source_id":
                safe_id(value)
            else:
                continue
            keys.append([key, value])
        result.append(dict(reason=safe_id(match["reason"]),
                           request_id=safe_id(match["request_id"]), shared_keys=keys))
    return result


def reduce_panel(panel):
    audit, selected = panel["exclusion_audit"], panel["requests"]
    if (panel["source_revision"] != REVISION or panel["selection_prefix"] != SELECTION_PREFIX
            or panel["path_prefix"] != PATH_PREFIX or panel["input_only_selection"] is not True
            or any(panel[key] is not False for key in
                   ("outcomes_used", "generated_lengths_used", "reference_availability_used"))):
        raise ValueError("Sealed input-only selection policy differs")
    candidates = {safe_id(row["source_id"]): row for row in audit}
    chosen = {safe_id(row["source_id"]): row for row in selected}
    if len(candidates) != len(audit) or len(chosen) != len(selected):
        raise ValueError("Duplicate candidate or selected source identity")
    if len(audit) != panel["candidate_count"] or len(selected) != panel["count"]:
        raise ValueError("Panel population count differs")
    for source_id, row in candidates.items():
        if source_id != f"gsm8k:main:test:{row['source_row']}" or type(row["source_row"]) is not int:
            raise ValueError("Candidate source identity and row differ")
        if row["status"] not in ("eligible", "excluded", "duplicate"):
            raise ValueError("Unknown candidate eligibility status")
        if type(row["prior_endpoint_exposure"]) is not bool:
            raise ValueError("Missing explicit prior-exposure flag")
    eligible = sorted((sid for sid, row in candidates.items() if row["status"] == "eligible"),
                      key=lambda sid: (candidates[sid]["prior_endpoint_exposure"],
                                       stable_hash(SELECTION_PREFIX, sid), sid))
    expected_chosen = eligible[:panel["requested_count"]]
    if eligible != panel["eligible_order"] or list(chosen) != expected_chosen:
        raise ValueError("Candidate selection differs from the sealed tier/hash order")
    if panel["shortfall"] != max(0, panel["requested_count"] - len(eligible)):
        raise ValueError("Panel shortfall differs")
    statuses = dict(Counter(row["status"] for row in audit))
    tiers = dict(Counter(str(int(candidates[sid]["prior_endpoint_exposure"])) for sid in eligible))
    chosen_tiers = dict(Counter(str(int(candidates[sid]["prior_endpoint_exposure"])) for sid in chosen))
    if (len(eligible) != panel["eligibility_count"] or statuses != panel["exclusion_status_counts"]
            or tiers != panel["eligible_tier_counts"] or chosen_tiers != panel["selected_tier_counts"]):
        raise ValueError("Panel eligibility/exposure totals differ")
    for ordinal, (sid, row) in enumerate(chosen.items()):
        safe_id(row["request_id"])
        if (row["panel_ordinal"] != ordinal or row["selection_sha256"] != stable_hash(SELECTION_PREFIX, sid)
                or row["prior_endpoint_exposure"] != candidates[sid]["prior_endpoint_exposure"]
                or row["exposure_tier"] != int(row["prior_endpoint_exposure"])):
            raise ValueError("Selected request identity, exposure or hash differs")
    path_sources = sorted(chosen, key=lambda sid: (stable_hash(PATH_PREFIX, sid), sid))[:64]
    if panel["path_request_ids"] != [chosen[sid]["request_id"] for sid in path_sources]:
        raise ValueError("Fixed-start subset differs from its independent hash order")
    ranks = {sid: rank for rank, sid in enumerate(eligible)}
    rows, reason_rows = [], Counter()
    for sid, row in sorted(candidates.items(), key=lambda item: item[1]["source_row"]):
        matches = safe_matches(row["exclusion_matches"])
        reasons = sorted({match["reason"] for match in matches})
        reason_rows.update(reasons)
        destinations = [safe_id(value) for value in row["duplicate_of"]]
        selection_status = "selected" if sid in chosen else "eligible_unselected" if row["status"] == "eligible" else row["status"]
        rows.append(dict(source_id=sid, source_row=row["source_row"],
            request_id=chosen[sid]["request_id"] if sid in chosen else "",
            status=selection_status, eligibility_status=row["status"],
            panel_ordinal=chosen[sid]["panel_ordinal"] if sid in chosen else "",
            eligible_rank=ranks.get(sid, ""), prior_endpoint_exposure=row["prior_endpoint_exposure"],
            exposure_tier=int(row["prior_endpoint_exposure"]),
            selection_sha256=stable_hash(SELECTION_PREFIX, sid), path_sha256=stable_hash(PATH_PREFIX, sid),
            fixed_start_selected=sid in path_sources,
            exclusion_reasons=json.dumps(reasons, separators=(",", ":")),
            exclusion_matches=json.dumps(matches, sort_keys=True, separators=(",", ":")),
            duplicate_of=json.dumps(destinations, separators=(",", ":"))))
    summary = dict(schema="retention-event7-panel-publication-v1", source_revision=REVISION,
        sealed_panel_binding=panel["binding"], candidate_count=len(audit), eligible_count=len(eligible),
        selected_count=len(selected), eligible_unselected_count=len(eligible)-len(selected),
        requested_count=panel["requested_count"], shortfall=panel["shortfall"],
        fixed_start_request_count=len(path_sources), candidate_status_counts=statuses,
        published_status_counts=dict(Counter(row["status"] for row in rows)),
        eligible_tier_counts=tiers, selected_tier_counts=chosen_tiers,
        excluded_candidate_counts_by_reason=dict(reason_rows),
        reason_count_scope="A candidate can match more than one exclusion reason; reason counts need not sum to excluded candidates.",
        selection_prefix=SELECTION_PREFIX, path_prefix=PATH_PREFIX,
        selection_rule="Prefer no recorded endpoint exposure; within tiers sort SHA256(prefix + stable source ID), UTF-8.",
        fixed_start_rule="First 64 selected requests under independent path-prefix hash order, or all if fewer.",
        input_only_selection=True, outcomes_used=False, generated_lengths_used=False,
        reference_availability_used=False,
        exposure_scope="Recorded prior endpoint exposure is disclosed; absence of a record does not establish global novelty.",
        exclusion_scope="Entire pinned GSM8K test candidate population after source, normalized-question and rendered/token identity checks; no exclusions relaxed.",
        privacy="Only identifiers, offsets, hashes, flags, counts and exclusion reasons; no prompt text, token arrays or filesystem paths.",
        model_operations=0, scientific_code_modified=False,
        authorship="Same-team metadata review; helper author also contributed panel preparation. No independent GPU replication.")
    return rows, summary


def publish(source, output):
    source, output = Path(source), Path(output)
    rows, summary = reduce_panel(json.loads(source.read_text()))
    summary.update(source_manifest_identity=identity(source), publication_helper_identity=identity(__file__))
    output.mkdir(parents=True, exist_ok=True)
    ledger = output / "panel_exclusions.csv"
    with ledger.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary["panel_exclusions_identity"] = identity(ledger)
    (output / "PANEL_SELECTION.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = publish(args.manifest, args.output)
    print(json.dumps({key: result[key] for key in ("candidate_count", "eligible_count", "selected_count", "model_operations")}))
