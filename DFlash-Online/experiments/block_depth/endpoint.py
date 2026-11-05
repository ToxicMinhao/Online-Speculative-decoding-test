"""Prespecified FINAL order, validity and complete-request comparisons."""

from collections import Counter
import csv
import json
import math
from pathlib import Path

import numpy as np


ARMS = ("FIVE", "THREE", "ADAPTIVE")
REQUESTS = 384
REPEATS = 2
PLANNED_GENERATIONS = REQUESTS * len(ARMS) * REPEATS


def endpoint_schedule(requests):
    if len(requests) != REQUESTS or len({r["request_id"] for r in requests}) != REQUESTS:
        raise ValueError("FINAL must contain exactly 384 distinct sealed requests")
    result = []
    for repeat in range(REPEATS):
        for block in range(24):
            shift = block % len(ARMS)
            order = ARMS[shift:] + ARMS[:shift]
            if repeat:
                order = tuple(reversed(order))
            for arm_position, arm in enumerate(order):
                for index in range(block * 16, (block + 1) * 16):
                    result.append(dict(request_id=requests[index]["request_id"], request_ordinal=index,
                                       repeat=repeat, block=block, arm_position=arm_position, arm=arm))
    return result


def validate_execution(record, request, canonical):
    """Errors accompany preserved raw records; no failed input is replaced."""
    errors = []
    for key in ("request_id", "prompt_ids", "stop_token_ids", "max_new_tokens"):
        if record.get(key) != request.get(key):
            errors.append("request_binding:" + key)
    if record.get("arm") not in ARMS or record.get("repeat") not in range(REPEATS):
        errors.append("arm_or_repeat")
    tokens, meta = record.get("response_ids", []), record.get("metadata", {})
    if not tokens or len(tokens) > request["max_new_tokens"] or any(type(t) is not int or t < 0 for t in tokens):
        errors.append("output_tokens")
    stops = set(request["stop_token_ids"])
    if any(token in stops for token in tokens[:-1]):
        errors.append("early_eos")
    end = "eos" if tokens and tokens[-1] in stops else "length"
    if record.get("finish_reason") != end or end == "length" and len(tokens) != request["max_new_tokens"]:
        errors.append("termination")
    if canonical is None:
        errors.append("canonical_unavailable")
    elif (tokens != canonical["response_ids"] or record.get("finish_reason") != canonical["finish_reason"]
          or canonical.get("prompt_ids") != request["prompt_ids"]):
        errors.append("canonical_parity")
    calls, accepted, progress = (meta.get(k) for k in ("verification_calls", "accepted_draft_tokens", "committed_progress"))
    if any(type(v) is not int or v < 0 for v in (calls, accepted, progress)):
        errors.append("native_counts")
    elif progress != len(tokens) - 1 or not 0 <= accepted <= min(progress, 15 * calls) or progress > 16 * calls:
        errors.append("native_count_consistency")
    shallow, deep = meta.get("shallow_rounds"), meta.get("deep_rounds")
    if any(type(v) is not int or v < 0 for v in (shallow, deep)) or shallow + deep != calls:
        errors.append("route_counts")
    elif (record.get("arm") == "FIVE" and shallow) or (record.get("arm") == "THREE" and deep):
        errors.append("static_route")
    for key in ("request_latency_s", "native_generate_s", "decode_round_wall_s", "cache_maintenance_s"):
        value = meta.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            errors.append("timing:" + key)
        elif key in ("request_latency_s", "native_generate_s") and value == 0:
            errors.append("timing:" + key)
        elif key == "decode_round_wall_s" and calls and value == 0:
            errors.append("timing:" + key)
    if meta.get("feedback_capture_in_timed_request") is not False or meta.get("trace"):
        errors.append("timed_capture_or_trace")
    return errors


def paired_bootstrap(mean_times):
    """One resampled question index carries every arm's two-repeat mean."""
    times = np.asarray(mean_times, dtype=np.float64)
    if times.ndim != 2 or times.shape[1] != 3 or not np.isfinite(times).all() or np.any(times <= 0):
        raise ValueError("Paired bootstrap needs finite positive three-arm times")
    draws = np.random.default_rng(20261105).integers(0, len(times), size=(10000, len(times)))
    summed = np.column_stack([times[:, arm][draws].sum(axis=1) for arm in range(3)])
    totals = times.sum(axis=0)
    comparisons = {}
    for name, arm, baseline in (("ADAPTIVE_vs_FIVE", 2, 0), ("ADAPTIVE_vs_THREE", 2, 1), ("THREE_vs_FIVE", 1, 0)):
        values = 100 * (1 - summed[:, arm] / summed[:, baseline])
        comparisons[name] = dict(point_pct=float(100 * (1 - totals[arm] / totals[baseline])),
                                 ci95_pct=np.quantile(values, [0.025, 0.975], method="linear").tolist())
    return comparisons


def analyze_endpoint(records, requests, canonical_by_id, *, unknown_executions=0):
    """Retain incomplete/invalid work; a scientific decision needs all cells."""
    schedule = endpoint_schedule(requests)
    if type(unknown_executions) is not int or unknown_executions < 0:
        raise ValueError("Unknown execution count must be a nonnegative integer")
    requests_by_id = {r["request_id"]: r for r in requests}
    expected = {(r["request_id"], r["arm"], r["repeat"]): r for r in schedule}
    seen, measurements, groups, order_errors = set(), [], {}, []
    for record in records:
        key = (record.get("request_id"), record.get("arm"), record.get("repeat"))
        request = requests_by_id.get(record.get("request_id"))
        errors = validate_execution(record, request, canonical_by_id.get(key[0])) if request else ["unsealed_request"]
        if key not in expected or key in seen:
            errors.append("unexpected_or_duplicate_cell")
        position = len(measurements)
        if position >= len(schedule) or key != tuple(schedule[position][field] for field in ("request_id", "arm", "repeat")):
            order_errors.append(position)
        seen.add(key)
        meta = record.get("metadata", {})
        row = dict(request_id=key[0], arm=key[1], repeat=key[2], valid=not errors, errors=errors,
                   output_tokens=len(record.get("response_ids", [])), termination=record.get("finish_reason"),
                   **{k: meta.get(k) for k in ("verification_calls", "accepted_draft_tokens", "committed_progress",
                     "request_latency_s", "native_generate_s", "decode_round_wall_s", "cache_maintenance_s", "shallow_rounds", "deep_rounds")})
        measurements.append(row)
        groups.setdefault((key[0], key[1]), []).append(row)
    for rows in groups.values():
        if len(rows) == REPEATS and all(r["valid"] for r in rows):
            for field in ("verification_calls", "accepted_draft_tokens", "committed_progress", "shallow_rounds", "deep_rounds"):
                if len({r[field] for r in rows}) != 1:
                    for row in rows:
                        row["valid"] = False
                        row["errors"].append("repeat_consistency:" + field)
    per_request = []
    for request in requests:
        item = dict(request_id=request["request_id"], valid=True)
        for arm in ARMS:
            rows = groups.get((request["request_id"], arm), [])
            if len(rows) != REPEATS or {r["repeat"] for r in rows} != {0, 1} or not all(r["valid"] for r in rows):
                item["valid"] = False
                continue
            rows = sorted(rows, key=lambda r: r["repeat"])
            for field in ("request_latency_s", "native_generate_s", "decode_round_wall_s", "cache_maintenance_s"):
                item[arm + "_" + field] = math.fsum(r[field] for r in rows) / REPEATS
            for field in ("verification_calls", "accepted_draft_tokens", "committed_progress", "shallow_rounds", "deep_rounds", "output_tokens"):
                item[arm + "_" + field] = rows[0][field]
        per_request.append(item)
    invalid = sum(not r["valid"] for r in measurements)
    accounting = dict(planned=PLANNED_GENERATIONS, returned=len(records), valid=len(records) - invalid,
                      invalid=invalid, unknown=unknown_executions,
                      unattempted=max(0, PLANNED_GENERATIONS - len(records) - unknown_executions),
                      excess_work=max(0, len(records) + unknown_executions - PLANNED_GENERATIONS),
                      complete_paired_requests=sum(r["valid"] for r in per_request))
    complete = (len(records) == PLANNED_GENERATIONS and not unknown_executions and not invalid and not order_errors
                and all(r["valid"] for r in per_request))
    result = dict(status="complete" if complete else "incomplete_or_invalid", accounting=accounting,
                  measurements=measurements, per_request=per_request, comparison=None,
                  execution_order_errors=order_errors,
                  interpretation="FINAL is fit-disjoint but previously exposed to the project; this is a frozen-checkpoint depth screen, not online recalibration")
    if not complete:
        return result
    means = [[r[arm + "_request_latency_s"] for arm in ARMS] for r in per_request]
    comparisons = paired_bootstrap(means)
    totals = {}
    for arm in ARMS:
        totals[arm] = {field: math.fsum(r[arm + "_" + field] for r in per_request)
                       for field in ("request_latency_s", "native_generate_s", "decode_round_wall_s", "cache_maintenance_s")}
        totals[arm].update({field: sum(r[arm + "_" + field] for r in per_request)
                           for field in ("verification_calls", "accepted_draft_tokens", "committed_progress", "output_tokens", "shallow_rounds", "deep_rounds")})
        total = totals[arm]
        calls = total["verification_calls"]
        total["accepted_per_call"] = total["accepted_draft_tokens"] / calls if calls else None
        total["progress_per_call"] = total["committed_progress"] / calls if calls else None
        total["shallow_exit_fraction"] = total["shallow_rounds"] / calls if calls else None
        total["repeat_request_latency_s"] = [math.fsum(r["request_latency_s"] for r in measurements if r["arm"] == arm and r["repeat"] == repeat) for repeat in range(2)]
    for baseline in ("FIVE", "THREE"):
        comparison = comparisons["ADAPTIVE_vs_" + baseline]
        baseline_calls = totals[baseline]["verification_calls"]
        comparison["call_reduction_pct"] = 100 * (1 - totals["ADAPTIVE"]["verification_calls"] / baseline_calls) if baseline_calls else None
        comparison["request_time_wins_losses_ties"] = dict(Counter(
            "win" if r["ADAPTIVE_request_latency_s"] < r[baseline + "_request_latency_s"] else
            "loss" if r["ADAPTIVE_request_latency_s"] > r[baseline + "_request_latency_s"] else "tie" for r in per_request))
    af, at, tf = (comparisons[k]["ci95_pct"] for k in ("ADAPTIVE_vs_FIVE", "ADAPTIVE_vs_THREE", "THREE_vs_FIVE"))
    advance = af[0] > 0 and at[0] > 0
    decision = "advance_adaptive" if advance else "prefer_FIVE" if af[1] < 0 and tf[1] < 0 else "prefer_THREE" if at[1] < 0 and tf[0] > 0 else "inconclusive"
    result.update(comparison=comparisons, arm_totals=totals, decision=decision,
                  scientific_criterion_passed=advance,
                  practical_two_pct_vs_FIVE_point_reached=comparisons["ADAPTIVE_vs_FIVE"]["point_pct"] >= 2,
                  bootstrap=dict(draws=10000, seed=20261105, unit="question with all arms and both repeats", interval="two-sided percentile 95%, linear quantiles"))
    return result


def write_analysis(directory, result):
    """Write compact analysis without substituting for raw execution records."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    (root / "ENDPOINT_ANALYSIS.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for key, filename in (("measurements", "ALL_REPEATS.csv"), ("per_request", "PER_REQUEST.csv")):
        rows = result[key]
        if not rows:
            continue
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with (root / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
