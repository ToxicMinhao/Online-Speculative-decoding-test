"""Delayed native scoring of two sealed size controls with PR47 outcome reuse.

The existing native worker reconstructs each parent once. This adapter never
loads missing historical checkpoints: only live P/N/E8 sentinels and the newly
selected scalar exports are served. Calibration must be sealed before entry.
"""
from collections import Counter
import json
from pathlib import Path
import time

from online_feedback.storage import write_json
from experiments.longer_run_headroom.pairs import neutral_parity
from experiments.native_prefix_loss.probe_client import (
    ProbeBackend, check_host_admission, probe_key, read, resource_estimate,
    score_observation, sealed_probe_keys,
)


WINDOWS = tuple(range(8, 129, 8))
STATES = ("S_low", "S_high")
HISTORICAL_STATES = ("P", "N", "E")
HISTORICAL_EXECUTION = "2549f02d9b1edba8fb86f8e91a497e68c224d586"
SCORE_FIELDS = ("accepted_prefix", "valid_proposals", "first_error", "matched_eos", "censored_at_cap", "valid")


def validate_exports(rows):
    """Require exactly the two actual controls; aliases share one physical file."""
    indexed = {(row["window"], row["state"]): row for row in rows}
    expected = {(window, state) for window in WINDOWS for state in STATES}
    if len(indexed) != len(rows) or set(indexed) != expected:
        raise ValueError("Require exactly 32 S_low/S_high bindings at all sixteen windows")
    for window in WINDOWS:
        low, high = (indexed[(window, state)] for state in STATES)
        if low.get("reused_from") is not None:
            raise ValueError("S_low must be the physical control, not an alias")
        for row in (low, high):
            identity = row.get("weights_identity", {})
            if not row.get("checkpoint") or not identity.get("sha256") or not identity.get("bytes"):
                raise ValueError("Every selected control needs its exact export identity")
        if high.get("reused_from") is not None:
            if high["reused_from"] != "S_low":
                raise ValueError("Only an exact same-window S_high-to-S_low alias is allowed")
            if high["checkpoint"] != low["checkpoint"] or high["weights_identity"] != low["weights_identity"]:
                raise ValueError("Aliased controls must share the exact physical export and identity")
        elif high["checkpoint"] == low["checkpoint"]:
            raise ValueError("A shared physical export must be declared as an exact alias")
    return indexed


def validate_control_seal(path, rows):
    seal = read(path)
    if seal.get("status") != "sealed" or seal.get("exports") != rows:
        raise ValueError("Both controls at all sixteen windows must be sealed before any scoring")
    return seal


def historical_measurements(plan, keys):
    """Use authenticated old measurements without requiring their old weights."""
    receipt = read(plan["reuse_identity_receipt"])
    if receipt.get("status") != "passed":
        raise ValueError("Historical source/runtime and raw-outcome reuse has not qualified")
    if receipt.get("historical_execution_commit") != HISTORICAL_EXECUTION:
        raise ValueError("Historical outcome reuse does not bind the original PR47 execution")
    export_rows = [row for row in read(plan["historical_probe_exports"])
                   if row["state"] in HISTORICAL_STATES]
    exports = {(row["window"], row["state"]): row for row in export_rows}
    expected_exports = {(window, state) for window in WINDOWS for state in HISTORICAL_STATES}
    if len(exports) != len(export_rows) or set(exports) != expected_exports:
        raise ValueError("Missing historical P/N/E export identity bindings")
    expected = {(window, *probe_key(item), state)
                for window in WINDOWS for item in keys for state in HISTORICAL_STATES}
    result, seen = [], set()
    with Path(plan["historical_probe_outcomes"]).open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["state"] not in HISTORICAL_STATES:
                continue
            identity = (row["window"], *probe_key(row), row["state"])
            if identity in seen or identity not in expected or row.get("valid") is not True:
                raise ValueError("Historical outcome grid is duplicate, unexpected or invalid")
            export = exports[(row["window"], row["state"])]
            if row["checkpoint"] != export["checkpoint"]:
                raise ValueError("Historical outcome is bound to a different export")
            seen.add(identity)
            result.append(dict(row, measurement_origin="historical_PR47",
                historical_execution_commit=HISTORICAL_EXECUTION,
                historical_weights_identity=export["weights_identity"]))
    if seen != expected:
        raise ValueError("Historical P/N/E measurement grid is incomplete")
    return result, receipt


def sentinel_keys(keys):
    """The first eight original keys, never chosen using an observed effect."""
    return sorted(keys, key=probe_key)[:8]


def sentinel_parity(observed, historical):
    fields = ("status", "blocks", "response_ids", "output_matches_canonical_prefix",
              "native_finished", "original_cap", "original_remaining_budget", "cap_unchanged")
    differences = [name for name in fields if observed.get(name) != historical.get(name)]
    return {"status": "failed" if differences else "passed", "differing_fields": differences,
            "comparison": "exact proposals, target verdicts, commits, positions, tokens, cap and termination; clocks excluded"}


def _append_outcome(root, row, rows, receipt):
    rows.append(row)
    with (root / "PROBE_OUTCOMES.jsonl").open("a") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")
        stream.flush()
    receipt["new_logical_control_rows"] = len(rows)
    write_json(root / "STATUS.json", receipt)


def evaluate_controls(backend, exports, keys, parents, snapshots, root, budget, receipt, outcomes):
    """Score only selected controls. Preserve returned raw data before checks."""
    for window in sorted({item[0] for item in exports}):
        measured = {}
        for state in STATES:
            export = exports[(window, state)]
            alias = export.get("reused_from")
            if alias:
                rows = [dict(row, state=state, reused_from=alias) for row in measured[alias]]
                receipt["aliased_control_rows"] += len(rows)
                for row in rows:
                    _append_outcome(root, row, outcomes, receipt)
                measured[state] = rows
                continue
            rows = []
            for item in keys:
                budget.check_time()
                path = root / f"window_{window:03d}" / state / item["request_id"] / f"{item['round_ordinal']:04d}.json"
                receipt["active_probe"] = dict(item, window=window, state=state)
                receipt["control_attempts"] += 1
                write_json(root / "STATUS.json", receipt)
                observed = None
                try:
                    observed = backend.one_block(snapshots[probe_key(item)], export["checkpoint"])
                    receipt["physical_control_blocks"] += 1
                    write_json(path, observed)
                    row = score_observation(parents[item["request_id"]], item, observed,
                        window=window, state=state, checkpoint=export["checkpoint"])
                except BaseException as error:
                    status = "unknown" if observed is None else "invalid"
                    row = dict(item, window=window, state=state, checkpoint=export["checkpoint"],
                        valid=False, status=status, error=repr(error), execution_id=str(path.relative_to(root)),
                        measurement_origin="new_scalar_control")
                    receipt[status + "_control_rows"] += 1
                    _append_outcome(root, row, outcomes, receipt)
                    raise
                row.update(execution_id=str(path.relative_to(root)), measurement_origin="new_scalar_control")
                rows.append(row)
                receipt["active_probe"] = None
                _append_outcome(root, row, outcomes, receipt)
            measured[state] = rows


def qualify_reuse(backend, plan, keys, parents, snapshots, historical, root, budget, receipt):
    """At most eight neutral blocks plus the fixed 24 P/N/E8 sentinels."""
    subset = sentinel_keys(keys)
    neutral_count = plan.get("neutral_qualifier_blocks", 8)
    if type(neutral_count) is not int or not 0 <= neutral_count <= 8:
        raise ValueError("Neutral native qualification is bounded at eight endpoint blocks")
    sentinels = {(row["window"], row["state"]): row for row in plan["sentinel_exports"]}
    if len(plan["sentinel_exports"]) != 3 or set(sentinels) != {(8, state) for state in HISTORICAL_STATES}:
        raise ValueError("Sentinels require actual P/N/E exports from window 8 only")
    old = {(row["window"], *probe_key(row), row["state"]): row for row in historical}
    checkpoint = plan["checkpoints"]["K6"]["export"]
    for item in subset[:neutral_count]:
        budget.check_time()
        receipt["neutral_attempts"] += 1
        receipt["active_qualification"] = dict(item, state="K6", operation="neutral", returned=False)
        write_json(root / "STATUS.json", receipt)
        observed = backend.one_block(snapshots[probe_key(item)], checkpoint)
        receipt["neutral_blocks"] += 1
        receipt["active_qualification"]["returned"] = True
        path = root / "sentinels" / "neutral" / item["request_id"] / f"{item['round_ordinal']:04d}.json"
        write_json(path, observed)
        record = parents[item["request_id"]]
        score_observation(record, item, observed, window=0, state="K6", checkpoint=checkpoint)
        parity = neutral_parity(record["metadata"]["trace"][item["round_ordinal"]], observed["blocks"][0])
        write_json(path.with_suffix(".parity.json"), parity)
        if parity["status"] != "passed":
            raise ValueError("Neutral native reconstruction differs from original proposals/verdicts")
        receipt["active_qualification"] = None
    for state in HISTORICAL_STATES:
        export = sentinels[(8, state)]
        original = old[(8, *probe_key(subset[0]), state)]
        if export["weights_identity"] != original["historical_weights_identity"]:
            raise ValueError("Live sentinel identity differs from its historical export")
        for item in subset:
            budget.check_time()
            receipt["sentinel_attempts"] += 1
            receipt["active_qualification"] = dict(item, state=state, operation="sentinel", returned=False)
            write_json(root / "STATUS.json", receipt)
            observed = backend.one_block(snapshots[probe_key(item)], export["checkpoint"])
            receipt["sentinel_blocks"] += 1
            receipt["active_qualification"]["returned"] = True
            path = root / "sentinels" / state / item["request_id"] / f"{item['round_ordinal']:04d}.json"
            write_json(path, observed)
            score = score_observation(parents[item["request_id"]], item, observed,
                window=8, state=state, checkpoint=export["checkpoint"])
            previous = old[(8, *probe_key(item), state)]
            original_raw = read(Path(plan["historical_probe_root"]) / previous["execution_id"])
            parity = sentinel_parity(observed, original_raw)
            parity["score_differences"] = [field for field in SCORE_FIELDS if score[field] != previous[field]]
            if parity["score_differences"]:
                parity["status"] = "failed"
            write_json(path.with_suffix(".parity.json"), parity)
            receipt["sentinel_checks"].append(dict(item, state=state, **parity))
            write_json(root / "STATUS.json", receipt)
            if parity["status"] != "passed":
                raise ValueError("PR47 sentinel proposal/verdict/outcome reuse differs")
            receipt["active_qualification"] = None
    receipt["reuse_sentinel_status"] = "passed"
    write_json(root / "REUSE_QUALIFICATION.json", {
        "status": "passed", "sentinel_keys": subset, "neutral_blocks": receipt["neutral_blocks"],
        "sentinel_blocks": receipt["sentinel_blocks"], "checks": receipt["sentinel_checks"],
        "source_and_runtime_identity_receipt": plan["reuse_identity_receipt"]})


def run(plan, export_rows, output, budget_connection, *, serving_gpu="0"):
    """Reconstruct once, qualify historical reuse, then score sealed controls."""
    from experiments.verification_transition.budget import connect_budget
    from experiments.verification_transition.native import run_counted

    exports = validate_exports(export_rows)
    validate_control_seal(plan["control_seal"], export_rows)
    keys = sealed_probe_keys(plan)
    historical, reuse_receipt = historical_measurements(plan, keys)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "HISTORICAL_REUSE.json", {
        "status": "authenticated_before_scoring", "historical_rows": len(historical),
        "source_outcomes": plan["historical_probe_outcomes"],
        "source_exports": plan["historical_probe_exports"], "identity_receipt": reuse_receipt,
        "historical_execution_commit": HISTORICAL_EXECUTION,
        "missing_E_checkpoints_regenerated": False})
    with (root / "PROBE_OUTCOMES.jsonl").open("w") as stream:
        for row in historical:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    estimate = resource_estimate(plan)
    write_json(root / "RESOURCE_ESTIMATE.json", estimate)
    check_host_admission(estimate)
    budget = connect_budget(budget_connection)
    before = budget.summary()
    backend = None
    outcomes, snapshots, parents = [], {}, {}
    began = time.perf_counter()
    receipt = dict(status="running", phase="saved_parent_reconstruction", new_generations=0,
        historical_rows=len(historical), planned_control_rows=8192, requests_reconstructed=0,
        new_logical_control_rows=0, physical_control_blocks=0, aliased_control_rows=0,
        invalid_control_rows=0, unknown_control_rows=0, control_attempts=0,
        neutral_attempts=0, neutral_blocks=0, sentinel_attempts=0, sentinel_blocks=0,
        sentinel_checks=[], reuse_sentinel_status="unattempted", active_probe=None,
        active_qualification=None,
        control_seal=plan["control_seal"], budget_before=before)
    write_json(root / "STATUS.json", receipt)
    try:
        checkpoint = plan["checkpoints"]["K6"]["export"]
        backend = run_counted(budget, "model_load", lambda: ProbeBackend(plan["target_path"], checkpoint,
            worker_python=plan["worker_python"], output_root=root / "native",
            budget_connection=budget_connection, serving_gpu=serving_gpu,
            host_reserve_bytes=estimate["host_reserve_bytes"]),
            metadata=dict(operation="size_control_probe_engine_initialization"))
        for rid in sorted(Counter(row["request_id"] for row in keys)):
            record = read(plan["parent_records"][rid])
            parents[rid] = record
            subset = [row for row in keys if row["request_id"] == rid]
            reconstructed = backend.reconstruct(record, subset, checkpoint)
            write_json(root / "parents" / rid / "RECONSTRUCTION.json", reconstructed)
            if reconstructed["status"] != "passed":
                raise ValueError("Saved native parent reconstruction did not pass")
            for item in subset:
                snapshots[probe_key(item)] = reconstructed["snapshots"][str(item["round_ordinal"])]
            receipt["requests_reconstructed"] += 1
            write_json(root / "STATUS.json", receipt)
        receipt["phase"] = "historical_reuse_sentinels"
        qualify_reuse(backend, plan, keys, parents, snapshots, historical, root, budget, receipt)
        receipt["phase"] = "sealed_control_scoring"
        evaluate_controls(backend, exports, keys, parents, snapshots, root, budget, receipt, outcomes)
        receipt.update(status="completed", phase="complete", loading_s=backend.loading_s)
    except BaseException as error:
        receipt.update(status="failed", error=repr(error))
        active = receipt.get("active_qualification")
        if active is not None:
            receipt["qualification_failure"] = dict(active,
                status="invalid" if active["returned"] else "unknown", error=repr(error))
        raise
    finally:
        write_json(root / "PROBE_OUTCOMES.json", historical + outcomes)
        try:
            if backend is not None:
                backend.close()
        except BaseException as error:
            receipt.update(status="failed", cleanup_error=repr(error))
            raise
        finally:
            after = budget.summary()
            receipt["budget_after"] = after
            receipt["work_delta"] = {kind: {
                name: after[kind].get(name, 0) - before[kind].get(name, 0)
                for name in sorted(set(before[kind]) | set(after[kind]))}
                for kind in ("attempted", "completed", "failed", "unknown")}
            receipt.update(wall_s=time.perf_counter() - began,
                unattempted_logical_control_rows=8192 - len(outcomes),
                unattempted_sentinel_blocks=24 - receipt["sentinel_attempts"],
                unattempted_neutral_blocks=plan.get("neutral_qualifier_blocks", 8) - receipt["neutral_attempts"])
            write_json(root / "STATUS.json", receipt)
    return dict(receipt, rows=historical + outcomes)
