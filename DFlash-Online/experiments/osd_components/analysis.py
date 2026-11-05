"""Paired, common-cohort analysis of the four checkpoint interventions.

The runner owns input/output, historical-reference, and publication validation.
This module consumes those gates, checks table completeness/repeated calls, and
does CPU arithmetic only. It never reads model records or chooses a better repeat.
"""

import csv
import json
import math
from collections import defaultdict
from numbers import Integral, Real
from pathlib import Path

import numpy as np


ARMS = ("F", "O", "C-only", "B-only")
REPEATS = (0, 1, 2)
SIGNS = ("improves", "ties", "regresses")
COUNT_FIELDS = (
    "verification_calls", "output_tokens", "accepted_draft_tokens", "committed_progress",
)
DENOMINATOR_STATES = {
    "negative": "D_O < 0",
    "zero": "D_O = 0",
    "small_positive": "0 < D_O / N_F < 0.02",
    "positive_at_least_2pct": "D_O / N_F >= 0.02",
}
LIMITATIONS = [
    "Fixed-panel totals are exact observations conditional on passed validity gates.",
    "Bootstrap intervals are exploratory pointwise request-composition sensitivity; "
    "they are not simultaneous guarantees, training-seed uncertainty, or fresh confirmation.",
    "A partial common cohort does not correct selection bias from invalid or missing requests.",
    "Retained-gain ratios with negative O gain do not express retention of positive benefit; "
    "small positive denominators are unstable. Use absolute effects alongside harm and time.",
    "Endpoint substitution does not establish restricted trainability, necessity, "
    "training savings, or net online acceleration. Follow-up experiments require separate review.",
]


def _safe(value):
    """Keep invalid nonfinite evidence visible while permitting strict JSON output."""
    if isinstance(value, dict):
        return {key: _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _sign(delta):
    return "improves" if delta > 0 else "regresses" if delta < 0 else "ties"


def _ratio(numerator, denominator, scale=1):
    return scale * numerator / denominator if denominator else None


def _effects(totals):
    """Accept scalar totals or bootstrap vectors, preserving identical pairing."""
    f, o, c, b = (totals[arm] for arm in ARMS)
    effects = {f"D_{arm}": f - totals[arm] for arm in ARMS}
    effects.update({f"E_{arm}": totals[arm] - o for arm in ARMS})
    effects.update(M_C_given_BF=f - c, M_C_given_BO=b - o,
                   M_B_given_CF=f - b, M_B_given_CO=c - o,
                   I=c + b - f - o)
    return effects


def _denominator_status(d_o, n_f):
    if d_o < 0:
        return "negative"
    if d_o == 0:
        return "zero"
    return "small_positive" if d_o / n_f < 0.02 else "positive_at_least_2pct"


def _interval(values):
    return np.percentile(values, [2.5, 97.5], method="linear").tolist()


def _measurement(row, request, arm, repeat):
    errors = list(row.get("errors") or [])
    if row.get("valid_measurement") is not True:
        errors.append("valid_measurement_not_true")
    if row.get("panel_ordinal") != request["panel_ordinal"]:
        errors.append("panel_ordinal_mismatch")
    valid_counts = {}
    for field in COUNT_FIELDS:
        value = row.get(field)
        valid_counts[field] = not isinstance(value, bool) and isinstance(value, Integral) and value >= 0
        if not valid_counts[field]:
            errors.append("invalid_" + field)
    if valid_counts["committed_progress"] and valid_counts["output_tokens"]:
        if row["committed_progress"] != row["output_tokens"] - 1:
            errors.append("progress_length_mismatch")
    if valid_counts["accepted_draft_tokens"] and valid_counts["committed_progress"]:
        if row["accepted_draft_tokens"] > row["committed_progress"]:
            errors.append("accepted_exceeds_progress")
    latency = row.get("request_latency_s")
    if (isinstance(latency, bool) or not isinstance(latency, Real)
            or not math.isfinite(latency) or latency <= 0):
        errors.append("invalid_request_latency_s")
    result = dict(row)
    result.update(original_index=request["original_index"], method=arm, repeat=repeat,
                  errors=list(dict.fromkeys(errors)), analysis_valid=not errors,
                  status="invalid" if errors else "valid")
    return result


def _cohort(rows, requests):
    indexed = defaultdict(list)
    request_by_id = {request["request_id"]: request for request in requests}
    if len(request_by_id) != len(requests):
        raise ValueError("The sealed request manifest contains duplicate request IDs")
    ordinals = [request["panel_ordinal"] for request in requests]
    if len(set(ordinals)) != len(ordinals):
        raise ValueError("The sealed request manifest contains duplicate panel ordinals")

    unexpected = []
    for row in rows:
        request_id, arm, repeat = (row.get(key) for key in ("request_id", "method", "repeat"))
        if (request_id not in request_by_id or arm not in ARMS
                or isinstance(repeat, bool) or repeat not in REPEATS):
            unexpected.append({**row, "analysis_valid": False, "status": "unexpected",
                               "analysis_errors": ["unexpected_request_method_or_repeat"]})
        else:
            indexed[(request_id, arm, repeat)].append(row)

    paired, measurements, valid = [], [], []
    for request in requests:
        request_id = request["request_id"]
        slots = {}
        for arm in ARMS:
            for repeat in REPEATS:
                present = indexed[(request_id, arm, repeat)]
                if len(present) == 1:
                    slot = _measurement(present[0], request, arm, repeat)
                else:
                    status = "missing" if not present else "duplicate"
                    slot = {"request_id": request_id, "panel_ordinal": request["panel_ordinal"],
                            "original_index": request["original_index"], "method": arm,
                            "repeat": repeat, "status": status, "valid_measurement": False,
                            "analysis_valid": False, "errors": [status + "_measurement"]}
                    if present:
                        slot["source_rows"] = present
                slots[(arm, repeat)] = slot

        # These comparisons operate on the full declared set, never on an
        # arm-specific subset. Invalid rows retain their original failure evidence.
        lengths = [slot.get("output_tokens") for slot in slots.values()]
        if all(isinstance(length, Integral) and not isinstance(length, bool) for length in lengths):
            if len(set(lengths)) != 1:
                for slot in slots.values():
                    slot["errors"].append("output_length_mismatch_across_arms_or_repeats")
                    slot.update(analysis_valid=False, status="invalid")
        for arm in ARMS:
            arm_slots = [slots[(arm, repeat)] for repeat in REPEATS]
            counts = [slot.get("verification_calls") for slot in arm_slots]
            if all(isinstance(count, Integral) and not isinstance(count, bool) for count in counts):
                if len(set(counts)) != 1:
                    for slot in arm_slots:
                        slot["errors"].append("verification_calls_mismatch_across_repeats")
                        slot.update(analysis_valid=False, status="invalid")

        extra = [row for row in unexpected if row.get("request_id") == request_id]
        included = all(slot["analysis_valid"] for slot in slots.values()) and not extra
        failures = [{"method": arm, "repeat": repeat, "errors": slot["errors"]}
                    for (arm, repeat), slot in slots.items() if not slot["analysis_valid"]]
        if extra:
            failures.append({"method": None, "repeat": None,
                             "errors": ["unexpected_measurements_for_request"]})
        entry = {key: request[key] for key in ("request_id", "panel_ordinal", "original_index")}
        entry.update(in_common_cohort=included, failure_reasons=failures,
                     canonical_output_tokens=lengths[0] if included else None,
                     designated_counts={arm: slots[(arm, 0)].get("verification_calls") for arm in ARMS},
                     calls_saved_vs_F={}, signs={}, validity={})
        for arm in ARMS:
            entry["validity"][arm] = [{"repeat": repeat, "status": slots[(arm, repeat)]["status"],
                                       "errors": slots[(arm, repeat)]["errors"]} for repeat in REPEATS]
            delta = (slots[("F", 0)]["verification_calls"] - slots[(arm, 0)]["verification_calls"]
                     if included else None)
            entry["calls_saved_vs_F"][arm] = delta
            entry["signs"][arm] = _sign(delta) if included else None
        for slot in slots.values():
            slot["in_common_cohort"] = included
        measurements.extend(slots.values())
        paired.append(entry)
        if included:
            valid.append(slots)
    return paired, measurements, valid, unexpected


def _bootstrap(valid, samples, seed):
    n = len(valid)
    result = {
        "seed": seed, "rng": "numpy.random.default_rng/PCG64", "samples_requested": samples,
        "samples": samples if n else 0, "cohort_size": n,
        "unit": "request; identical indices across arms and all three timing repeats",
        "percentile_convention": "95% percentile [2.5, 97.5], NumPy linear interpolation",
        "pointwise": True, "simultaneous": False,
        "absolute_call_effects": {}, "pooled_time_reduction_pct": {},
        "denominator_states": {state: {"definition": definition, "count": 0, "fraction": None}
                               for state, definition in DENOMINATOR_STATES.items()},
        "retained_gain_ratios": {},
        "interpretation": LIMITATIONS[1:4],
    }
    if not n:
        result["status"] = "unavailable_empty_common_cohort"
        return result
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(samples, n))
    totals = {}
    times = {}
    for arm in ARMS:
        counts = np.asarray([slots[(arm, 0)]["verification_calls"] for slots in valid], dtype=np.int64)
        totals[arm] = counts[indices].sum(axis=1)
        latencies = np.asarray([[slots[(arm, repeat)]["request_latency_s"] for repeat in REPEATS]
                                for slots in valid], dtype=np.float64)
        times[arm] = latencies[indices].sum(axis=(1, 2))
    effects = _effects(totals)
    result["absolute_call_effects"] = {name: _interval(values) for name, values in effects.items()}
    result["pooled_time_reduction_pct"] = {
        arm: _interval(100 * (1 - times[arm] / times["F"])) for arm in ARMS}
    d_o = effects["D_O"]
    masks = {
        "negative": d_o < 0, "zero": d_o == 0,
        "small_positive": (d_o > 0) & (d_o < 0.02 * totals["F"]),
        "positive_at_least_2pct": (d_o > 0) & (d_o >= 0.02 * totals["F"]),
    }
    for state, mask in masks.items():
        count = int(mask.sum())
        result["denominator_states"][state].update(count=count, fraction=count / samples)
    zero_count = int(masks["zero"].sum())
    for arm in ("C-only", "B-only"):
        result["retained_gain_ratios"][arm] = {
            "ci95": None if zero_count else _interval(effects[f"D_{arm}"] / d_o),
            "interval_status": "undefined_resamples_preclude_finite_interval" if zero_count
                               else "all_resamples_finite_including_negative_denominators",
            "undefined_samples": zero_count,
            "finite_negative_denominator_samples": int(masks["negative"].sum()),
            "positive_small_denominator_samples": int(masks["small_positive"].sum()),
            "filtered_resamples": 0,
        }
    result["status"] = "available"
    return result


def summarize(rows, requests, *, bootstrap_samples=10000, seed=20261024):
    """Summarize numeric JSON score rows against the complete sealed request list.

    Exactly four arms and three repeats are expected per request. ``errors`` and
    ``valid_measurement`` must already include all native/reference gates. Optional
    ``cap_hit``/``finish_reason`` fields support cap statistics; omission is unknown,
    never an assumed zero. Extra columns are preserved in the measurement table.
    """
    if isinstance(bootstrap_samples, bool) or not isinstance(bootstrap_samples, Integral) or bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be a positive integer")
    requests, rows = list(requests), list(rows)
    paired, measurements, valid, unexpected = _cohort(rows, requests)
    n = len(valid)
    bootstrap = _bootstrap(valid, int(bootstrap_samples), seed)
    totals = {arm: sum(slots[(arm, 0)]["verification_calls"] for slots in valid) for arm in ARMS}
    effects = _effects(totals)
    tokens = sum(slots[("F", 0)]["output_tokens"] for slots in valid)
    latency_totals = {arm: [sum(slots[(arm, repeat)]["request_latency_s"] for slots in valid)
                            for repeat in REPEATS] for arm in ARMS}
    denominator_status = _denominator_status(effects["D_O"], totals["F"]) if n else "unavailable"
    pooled = []
    for arm in ARMS:
        deltas = [slots[("F", 0)]["verification_calls"] - slots[(arm, 0)]["verification_calls"]
                  for slots in valid]
        gross, harm = sum(max(delta, 0) for delta in deltas), sum(max(-delta, 0) for delta in deltas)
        accepted = sum(slots[(arm, 0)]["accepted_draft_tokens"] for slots in valid)
        progress = sum(slots[(arm, 0)]["committed_progress"] for slots in valid)
        caps = [slots[(arm, 0)].get("cap_hit") for slots in valid]
        caps = [slot[(arm, 0)].get("finish_reason") == "length" if cap is None
                and slot[(arm, 0)].get("finish_reason") is not None else cap
                for cap, slot in zip(caps, valid)]
        retained = _ratio(effects[f"D_{arm}"], effects["D_O"]) if n else None
        row = {
            "method": arm, "n_requests": n, "N_X": totals[arm], "output_tokens": tokens,
            "calls_per_1000_tokens": _ratio(totals[arm], tokens, 1000),
            "D_X": effects[f"D_{arm}"], "E_X": effects[f"E_{arm}"], "G_X": gross, "H_X": harm,
            "improved_requests": sum(delta > 0 for delta in deltas),
            "tied_requests": sum(delta == 0 for delta in deltas),
            "regressed_requests": sum(delta < 0 for delta in deltas),
            "R_X": retained, "retained_gain_status": denominator_status,
            "descriptive_80pct_screen": retained >= 0.8 if arm in ("C-only", "B-only")
                                          and denominator_status == "positive_at_least_2pct" else None,
            "call_reduction_pct": _ratio(effects[f"D_{arm}"], totals["F"], 100),
            "D_X_ci95": bootstrap["absolute_call_effects"].get(f"D_{arm}"),
            "E_X_ci95": bootstrap["absolute_call_effects"].get(f"E_{arm}"),
            "accepted_draft_tokens": accepted, "committed_progress": progress,
            "accepted_draft_tokens_per_call": _ratio(accepted, totals[arm]),
            "committed_progress_per_call": _ratio(progress, totals[arm]),
            "cap_hits": sum(caps) if all(isinstance(cap, bool) for cap in caps) else None,
            "cap_status_known_requests": sum(isinstance(cap, bool) for cap in caps),
            "request_latency_s_by_repeat": latency_totals[arm],
            "request_latency_s_mean_total": sum(latency_totals[arm]) / 3,
            "request_latency_s_all_repeats": sum(latency_totals[arm]),
            "time_reduction_pct": (100 * (1 - sum(latency_totals[arm]) / sum(latency_totals["F"])))
                                  if n else None,
            "time_reduction_pct_ci95": bootstrap["pooled_time_reduction_pct"].get(arm),
        }
        if not n:
            row = {key: value if key in ("method", "n_requests", "retained_gain_status") else None
                   for key, value in row.items()}
        pooled.append(row)

    interaction = [{"effect": name, "calls_saved": effects[name] if n else None,
                    "ci95": bootstrap["absolute_call_effects"].get(name)}
                   for name in ("M_C_given_BF", "M_C_given_BO", "M_B_given_CF", "M_B_given_CO", "I")]
    cross_tabs = {}
    for hybrid in ("C-only", "B-only"):
        counts = [[sum(entry["in_common_cohort"] and entry["signs"]["O"] == row_sign
                       and entry["signs"][hybrid] == column_sign for entry in paired)
                   for column_sign in SIGNS] for row_sign in SIGNS]
        cross_tabs[hybrid] = {"row_arm": "O", "column_arm": hybrid, "signs": list(SIGNS),
                              "counts": counts, "n_requests": n, "available": bool(n)}

    checks = {}
    if n:
        checks["D_plus_E_equals_D_O"] = all(row["D_X"] + row["E_X"] == effects["D_O"] for row in pooled)
        checks["gross_minus_harm_equals_net"] = all(row["G_X"] - row["H_X"] == row["D_X"] for row in pooled)
        checks["sign_counts_equal_cohort"] = all(sum(row[key] for key in
            ("improved_requests", "tied_requests", "regressed_requests")) == n for row in pooled)
        checks["cross_tab_counts_equal_cohort"] = all(sum(map(sum, tab["counts"])) == n for tab in cross_tabs.values())
        checks["interaction_identities"] = (
            effects["I"] == effects["D_O"] - effects["D_C-only"] - effects["D_B-only"]
            == effects["M_C_given_BO"] - effects["M_C_given_BF"]
            == effects["M_B_given_CO"] - effects["M_B_given_CF"])
        if not all(checks.values()):
            raise ArithmeticError("Checkpoint-component accounting identity failed")
    summary = {
        "schema_version": 1, "arms": list(ARMS), "repeats": list(REPEATS), "designated_repeat": 0,
        "cohort": {"status": "FULL" if n == len(requests) == 64 and not unexpected else "PARTIAL" if n else "UNAVAILABLE",
                   "planned_requests": len(requests), "valid_requests": n, "excluded_requests": len(requests) - n,
                   "all_planned_valid": bool(requests) and n == len(requests) and not unexpected,
                   "full_declared_panel": n == len(requests) == 64 and not unexpected,
                   "valid_request_ids": [entry["request_id"] for entry in paired if entry["in_common_cohort"]],
                   "excluded_request_ids": [entry["request_id"] for entry in paired if not entry["in_common_cohort"]],
                   "expected_measurements": len(requests) * 12, "supplied_measurements": len(rows),
                   "missing_slots": sum(row["status"] == "missing" for row in measurements),
                   "invalid_slots": sum(row["status"] == "invalid" for row in measurements),
                   "duplicate_slots": sum(row["status"] == "duplicate" for row in measurements),
                   "unexpected_measurements": len(unexpected)},
        "pooled": pooled, "interaction": interaction, "cross_tabs": cross_tabs,
        "bootstrap": bootstrap, "accounting_checks": checks,
        "per_request": paired, "measurements": measurements, "unexpected_measurements": unexpected,
        "limitations": LIMITATIONS,
    }
    return _safe(summary)


def _write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, allow_nan=False)
                             if isinstance(value, (dict, list)) else value for key, value in row.items()})


def write_report(summary, output_dir):
    """Write strict JSON, compact CSV tables, and a scoped arithmetic report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {"summary": "SUMMARY.json", "pooled": "POOLED.csv", "interaction": "INTERACTION.csv",
             "per_request": "PER_REQUEST.csv", "measurements": "MEASUREMENTS.csv",
             "unexpected_measurements": "UNEXPECTED_MEASUREMENTS.csv", "cross_tabs": "CROSS_TABS.csv",
             "report": "REPORT.md"}
    (output_dir / files["summary"]).write_text(json.dumps(summary, indent=2, ensure_ascii=False,
                                                        allow_nan=False) + "\n", encoding="utf-8")
    for key in ("pooled", "interaction", "per_request", "measurements", "unexpected_measurements"):
        _write_csv(output_dir / files[key], summary[key])
    cross_rows = [{"hybrid": hybrid, "O_vs_F": row_sign, "hybrid_vs_F": column_sign,
                   "requests": tab["counts"][i][j]}
                  for hybrid, tab in summary["cross_tabs"].items()
                  for i, row_sign in enumerate(SIGNS) for j, column_sign in enumerate(SIGNS)]
    _write_csv(output_dir / files["cross_tabs"], cross_rows)
    cohort = summary["cohort"]
    lines = ["# Checkpoint-component arithmetic report", "", "## Material Passport", "",
             "ARS experiment-agent; CPU analysis of supplied measurements. "
             "Correctness qualification belongs to the runner; no model execution or training by this module.", "",
             f"**{cohort['status']}**: {cohort['valid_requests']}/{cohort['planned_requests']} declared requests "
             "in the common four-arm/all-repeat cohort. Full-panel claims require all 64 requests.", "",
             "Calls and token statistics use designated repeat 0. Timing pools all three repeats. "
             "Every table and bootstrap uses the same common cohort.", "",
             "| Arm | Requests | Calls | Saved vs F | Extra vs O | Gross saved | Gross harm | Improved/tied/regressed | Retained ratio | Call reduction % | Time reduction % |",
             "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|"]
    def display(value):
        return "unavailable" if value is None else f"{value:.6g}" if isinstance(value, float) else str(value)
    for row in summary["pooled"]:
        signs = "/".join(display(row[key]) for key in ("improved_requests", "tied_requests", "regressed_requests"))
        values = [row[key] for key in ("method", "n_requests", "N_X", "D_X", "E_X", "G_X", "H_X")]
        values += [signs, row["R_X"], row["call_reduction_pct"], row["time_reduction_pct"]]
        lines.append("| " + " | ".join(map(display, values)) + " |")
    lines += ["", "Retained-ratio denominator status: " + summary["pooled"][0]["retained_gain_status"] + ".",
              "", "| Conditional effect | Calls saved | Pointwise 95% interval |", "|---|---:|---|"]
    for row in summary["interaction"]:
        lines.append(f"| {row['effect']} | {display(row['calls_saved'])} | {display(row['ci95'])} |")
    lines += ["", "Bootstrap denominator states (no resamples filtered):", ""]
    for state, values in summary["bootstrap"]["denominator_states"].items():
        lines.append(f"- {values['definition']}: {values['count']} ({display(values['fraction'])}).")
    lines += ["", "Limitations:", ""] + ["- " + note for note in summary["limitations"]]
    lines += ["", "The CSV/JSON artifacts retain absolute intervals, ratio interval status, cross-tabs, "
              "every timing repeat, missing/invalid slots, and mismatch evidence. No outcome automatically "
              "selects or authorizes a later training experiment.", ""]
    (output_dir / files["report"]).write_text("\n".join(lines), encoding="utf-8")
    return {key: str(output_dir / name) for key, name in files.items()}
