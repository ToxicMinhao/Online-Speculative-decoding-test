"""CPU-only publication supplement; never modify original experiment receipts.

Read only whitelisted identities and measurements. In particular, this helper
does not open BUDGET_CONNECTION.json or publish prompts/proposal/token arrays.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path


START_FIELDS = (
    "request_id", "canonical_start", "round_ordinal", "valid_proposals", "a4", "a5", "a6",
    "genuine_k4_rejection", "genuine_k5_rejection", "genuine_k6_rejection", "parity_valid",
    "terminal", "stop_clipped", "history", "j", "H", "M", "converse", "converse_j",
    "previously_attained_then_lost", "converse_prefix_gain", "k6_earlier_break_vs_k4",
    "k6_earlier_break_vs_k5", "k4_earlier_break_vs_k6", "k5_earlier_break_vs_k6",
)
PARITY_FIELDS = (
    "request_id", "canonical_start", "round_ordinal", "arm", "receipt_status",
    "parity_status", "all_15_proposals_exact", "acceptance_exact", "commits_exact",
    "positions_and_seed_exact", "terminal_and_clipping_exact", "target_prefix_cache_exact",
    "neutral_draft_prefix_cache_exact", "destination_draft_rebuilt", "cap_unchanged",
    "output_matches_canonical_prefix", "reconstruction_status", "expected_sha256", "observed_sha256",
)
OPERATIONS = {
    "startup_draft", "startup_standalone_verifier", "startup_target", "startup_target_context",
    "native_proposal_plus_verification", "native_target_verify", "native_target_prefill",
    "native_draft_context_ingestion", "terminal_teacher_preparation", "readonly_training_bundle",
}


def load(path, missing, errors):
    if not path.is_file():
        missing.append(str(path))
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        errors.append(dict(path=str(path), error="unreadable or malformed JSON"))
        return None


def identity(path):
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return dict(bytes=path.stat().st_size, sha256=digest.hexdigest())


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows, fields):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)


def starts(census):
    """Expand explicit native rejection fields without importing analysis helpers."""
    rows = []
    for source in census:
        row = {key: source.get(key) for key in START_FIELDS}
        a4, a5, a6 = (row[key] for key in ("a4", "a5", "a6"))
        row.update(history="missing_replay", H=False, M=False, converse=False,
                   previously_attained_then_lost=None, converse_prefix_gain=None)
        if None in (a4, a5, a6):
            rows.append(row)
            continue
        if row["parity_valid"] is not True:
            row["history"] = "invalid_replay"
            rows.append(row)
            continue
        g4, g5, g6 = (row[key] is True for key in
                      ("genuine_k4_rejection", "genuine_k5_rejection", "genuine_k6_rejection"))
        h = g4 and g6 and a4 == a6 and a5 >= a6 + 1
        m = g4 and g5 and g6 and a4 == a5 == a6
        converse = g4 and g5 and a4 == a5 and a6 >= a4 + 1
        row.update(
            history="H" if h else "M" if m else "other_rejection_history" if g6 else "all_correct_or_terminal",
            H=h, M=m, converse=converse, j=a6 + 1 if g6 else None,
            converse_j=a4 + 1 if converse else None,
            previously_attained_then_lost=a5 - a6 if h else None,
            converse_prefix_gain=a6 - a4 if converse else 0,
            k6_earlier_break_vs_k4=a6 < a4, k6_earlier_break_vs_k5=a6 < a5,
            k4_earlier_break_vs_k6=a4 < a6, k5_earlier_break_vs_k6=a5 < a6,
        )
        rows.append(row)
    return rows


def parity_receipts(campaign, census, missing, errors):
    rows = []
    for item in census:
        folder = campaign / "census" / item["request_id"]
        stem = f"{item['round_ordinal'] - 1:04d}"
        parity = load(folder / f"{stem}_PARITY.json", missing, errors) or {}
        differences = parity.get("differences")
        for arm in ("K6", "K4", "K5"):
            receipt = load(folder / f"{stem}_{arm}.json", missing, errors) or {}
            reconstruction = receipt.get("reconstruction") or {}
            neutral = arm == "K6"
            checked = neutral and isinstance(differences, list)
            rows.append(dict(
                request_id=item["request_id"], canonical_start=item["canonical_start"],
                round_ordinal=item["round_ordinal"], arm=arm,
                receipt_status=receipt.get("status", "unavailable"),
                parity_status=parity.get("status", "unavailable") if neutral else "not_applicable",
                all_15_proposals_exact=parity.get("all_15_proposals_exact") if neutral else None,
                acceptance_exact=not any(k in differences for k in ("accepted_draft_tokens", "native_accepted_draft_tokens")) if checked else None,
                commits_exact=not any(k in differences for k in ("committed_progress", "committed_tokens")) if checked else None,
                positions_and_seed_exact=not any(k in differences for k in ("output_start", "anchor_token", "positions")) if checked else None,
                terminal_and_clipping_exact=not any(k in differences for k in ("finished", "stop_clipped")) if checked else None,
                target_prefix_cache_exact=reconstruction.get("target_prefix_cache_exact"),
                neutral_draft_prefix_cache_exact=reconstruction.get("neutral_draft_prefix_cache_exact"),
                destination_draft_rebuilt=reconstruction.get("destination_draft_rebuilt"),
                cap_unchanged=receipt.get("cap_unchanged"),
                output_matches_canonical_prefix=receipt.get("output_matches_canonical_prefix"),
                reconstruction_status=reconstruction.get("status", "unavailable"),
                expected_sha256=parity.get("expected_sha256") if neutral else None,
                observed_sha256=parity.get("observed_sha256") if neutral else None,
            ))
    summary = dict(observed_natural_starts=len(census), arms={})
    for arm in ("K6", "K4", "K5"):
        observed = [r for r in rows if r["arm"] == arm]
        summary["arms"][arm] = dict(
            expected_from_observed_census=len(census),
            receipt_status_counts=dict(Counter(r["receipt_status"] for r in observed)),
            reconstruction_status_counts=dict(Counter(r["reconstruction_status"] for r in observed)),
            checks={key: dict(passed=sum(r[key] is True for r in observed),
                             failed=sum(r[key] is False for r in observed),
                             unavailable=sum(r[key] is None for r in observed))
                    for key in PARITY_FIELDS[6:16]},
        )
    summary["neutral_parity_status_counts"] = dict(Counter(r["parity_status"] for r in rows if r["arm"] == "K6"))
    summary["full_census_denominator"] = None  # Set only from the completed run's explicit receipt.
    return rows, summary


def work_accounting(path, missing, errors):
    """Count native startup separately without exporting work metadata payloads.

    Ticket elapsed times use the ledger's occupied-GPU clock and can overlap.
    They are not kernel times or disjoint components of total execution time.
    """
    if not path.is_file():
        missing.append(str(path))
        return dict(available=False)
    pending, counts, elapsed = {}, defaultdict(Counter), defaultdict(float)
    with path.open() as stream:
        for ordinal, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except ValueError:
                errors.append(dict(path=str(path), line=ordinal, error="malformed work receipt"))
                continue
            status = row.get("status")
            if status == "attempted":
                meta = row.get("metadata", {})
                op = meta.get("operation")
                op = op if op in OPERATIONS else "unspecified"
                group = (row["kind"], op, bool(meta.get("startup", False)))
                pending[row["ticket"]] = (group, row.get("gpu_seconds"))
                counts[group]["attempted"] += 1
            elif status in ("completed", "failed"):
                previous = pending.pop(row["ticket"], None)
                if previous is None:
                    errors.append(dict(path=str(path), line=ordinal, error="completion without known attempt"))
                    continue
                group, began = previous
                counts[group][status] += 1
                end = row.get("gpu_seconds")
                if began is not None and end is not None:
                    elapsed[group] += end - began
    for group, _ in pending.values():
        counts[group]["unknown"] += 1
    rows = [dict(kind=key[0], operation=key[1], startup=key[2],
                 **{name: counter[name] for name in ("attempted", "completed", "failed", "unknown")},
                 summed_ticket_occupied_gpu_seconds=elapsed[key])
            for key, counter in sorted(counts.items())]
    return dict(available=True, by_operation=rows,
                startup_attempts_by_kind=dict(Counter({kind: sum(r["attempted"] for r in rows if r["startup"] and r["kind"] == kind)
                                                       for kind in {r["kind"] for r in rows if r["startup"]}})),
                timing_note="Ticket intervals overlap/nest and include host work; do not add them or interpret them as kernel times.")


def selected(mapping, names):
    return {key: mapping.get(key) for key in names}


def publish(campaign_root, sealed_dir, source_manifest, *, audit_receipt=None, output=None):
    campaign, sealed, source_manifest = map(Path, (campaign_root, sealed_dir, source_manifest))
    destination = Path(output) if output else campaign / "publication_supplement"
    if destination.resolve() in (campaign.resolve(), sealed.resolve(), (campaign / "final_report").resolve()):
        raise ValueError("Supplement requires a separate output directory")
    destination.mkdir(parents=True, exist_ok=True)
    missing, errors = [], []
    plan = load(campaign / "PLAN.json", missing, errors) or load(sealed / "PLAN.json", missing, errors) or {}
    status = load(campaign / "STATUS.json", missing, errors) or {}
    census = load(campaign / "census/CENSUS.json", missing, errors) or []
    result = load(campaign / "RESULT.json", missing, errors) or {}
    training = load(campaign / "training/RESULT.json", missing, errors) or {}
    inputs = load(campaign / "training/INPUTS.json", missing, errors) or {}
    budget = load(campaign / "BUDGET.json", missing, errors) or {}
    postflight = load(campaign / "POSTFLIGHT.json", missing, errors) or {}
    source = load(source_manifest, missing, errors) or {}
    expanded = starts(census)
    parity_rows, parity = parity_receipts(campaign, census, missing, errors)
    if status.get("census_complete") is True:
        parity["full_census_denominator"] = len(census)
    identities = dict(
        execution_commit=plan.get("execution_commit"), source_commit=source.get("commit"),
        source_manifest=identity(source_manifest), actual_run_plan=identity(campaign / "PLAN.json"),
        sealed_plan=identity(sealed / "PLAN.json"), seal=identity(sealed / "SEAL.json"),
        original_manifest=plan.get("manifest_identity"), original_row_range=plan.get("original_row_range"),
        batch_sha256=plan.get("original_rows_canonical_json_sha256"),
        scheduled_request_count=len(plan.get("requests", [])),
        schedule_observed=identity(campaign / "training/SCHEDULE.json"),
        schedule_reported=training.get("schedule_identity"),
        configuration=identity(sealed / "config_KPLUS.yaml"),
        recovery=dict(path=plan.get("recovery_path"),
                      declared_identity=plan.get("pinned_files", {}).get(plan.get("recovery_path")),
                      observed_input_identity=next((r for r in inputs.get("inputs", []) if r.get("path") == plan.get("recovery_path")), None)),
        checkpoints={key: selected(value, ("export", "weights_identity", "config_identity"))
                     for key, value in plan.get("checkpoints", {}).items()},
        recovery_checks=selected(inputs, ("config_matches_recovery", "accumulation_remainder", "global_step", "presentations", "horizon", "accumulation_steps", "reduction", "optimizer_creation")),
        training_integrity=selected(training, ("caller_rng_and_modes_restored", "parameters_and_gradients_unchanged")),
        postflight=selected(postflight, ("status", "inputs_and_checkpoints_unchanged", "source_unchanged")),
    )
    costs = dict(
        campaign=selected(result, ("status", "wall_s", "optimizer_steps", "backward_calls", "timed_endpoints")),
        phase_costs={key: selected(value, ("wall_s", "exit_code", "interrupted")) for key, value in result.get("phase_costs", {}).items()},
        training=selected(training, ("initialization_s", "schedule_s", "wall_s", "phase_wall_s", "peak_allocated_bytes", "peak_reserved_bytes", "response_forwards", "projection_chunks", "backwards", "optimizer_steps", "teacher_head_forwards", "startup_response_forwards", "startup_backwards", "temporary_optimizer_steps")),
        occupied_gpu=selected(budget, ("gpu_seconds", "gpu_seconds_limit", "attempted", "completed", "failed", "unknown")),
        work_ledger=work_accounting(campaign / "WORK.jsonl", missing, errors),
        timing_limitations=["Phase clocks include loading and host work, not only GPU kernels.",
                            "Census capture, teacher preparation, neutral replay and historical replay overlap with nested ledger categories; no disjoint kernel breakdown was measured.",
                            "Missing scalar receipts remain null. Startup operation counts are included in total operation counts, not additional work."],
    )
    audit = None
    if audit_receipt:
        path = Path(audit_receipt)
        record = load(path, missing, errors) or {}
        audit = dict(path=str(path), identity=identity(path), status=record.get("status"),
                     counts={key: record[key] for key in ("checks", "checks_passed", "failures_count")
                             if isinstance(record.get(key), (int, float))})
    parity.update(execution_status=status.get("status", "unavailable"),
                  census_complete=status.get("census_complete", False),
                  training_complete=status.get("training_complete", False),
                  history_counts=dict(Counter(r["history"] for r in expanded)),
                  independent_audit=audit, missing_evidence=missing, read_errors=errors)
    write_csv(destination / "ALL_STARTS.csv", expanded, START_FIELDS)
    write_csv(destination / "PARITY_RECEIPTS.csv", parity_rows, PARITY_FIELDS)
    write_json(destination / "IDENTITIES.json", identities)
    write_json(destination / "PARITY_SUMMARY.json", parity)
    write_json(destination / "COST_ACCOUNTING.json", costs)
    lines = ["## Retained-evidence publication supplement", "",
        f"Execution `{identities['execution_commit']}`. Execution status: **{status.get('status', 'unavailable')}**; census complete: {status.get('census_complete', False)}; read-only exposure complete: {status.get('training_complete', False)}.",
        "This supplement changes no checkpoint, schedule, labels in the original report, or scientific setting. It adds sanitized receipts and independently coded historical-category expansion; it is not a new native replication.", "",
        f"Observed starts: {len(expanded)}. Full census denominator: {parity['full_census_denominator'] if parity['full_census_denominator'] is not None else 'unavailable'}. Category counts: `{json.dumps(parity['history_counts'], sort_keys=True)}`.",
        f"Neutral parity: `{json.dumps(parity['neutral_parity_status_counts'], sort_keys=True)}`. See PARITY_RECEIPTS.csv for per-arm cache/output checks; null values mean unavailable or inapplicable, not success.", "",
        f"Original row range: `{identities['original_row_range']}`; batch digest: `{identities['batch_sha256']}`. Source, plan, configuration, recovery and schedule identities appear in IDENTITIES.json.",
        f"Independent audit receipt: `{audit['path']}` ({audit.get('status')})" if audit else "Independent audit receipt: not supplied; no independent final sign-off claimed.", "",
        "| Measured clock | Seconds |", "|---|---:|",
        f"| Whole campaign | {result.get('wall_s', 'unavailable')} |",
        f"| Aggregate occupied GPU | {budget.get('gpu_seconds', 'unavailable')} |",
    ]
    for name, phase in costs["phase_costs"].items():
        lines.append(f"| {name} phase | {phase.get('wall_s')} |")
    for key in ("initialization_s", "schedule_s", "wall_s"):
        lines.append(f"| Read-only training {key} | {training.get(key, 'unavailable')} |")
    lines += ["", "These clocks are nested; do not sum them. Work-ticket timings are diagnostic occupied-GPU-clock intervals and do not establish individual kernel costs. All measured operation counts, native startup counts and training peak memory are in COST_ACCOUNTING.json.",
              "Incomplete evidence retains observed counts but cannot decide the full-event gate. The original GATE.json remains authoritative; publication does not turn a partial run into a pass.",
              "Raw prompts, tokens, tensors and connection credentials are excluded from this supplement.", ""]
    (destination / "supplement.md").write_text("\n".join(lines))
    return dict(output=str(destination), observed_starts=len(expanded), missing=len(missing), errors=len(errors))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--sealed-dir", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--audit-receipt", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(json.dumps(publish(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
