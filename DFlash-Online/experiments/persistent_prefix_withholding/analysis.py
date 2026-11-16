"""Prespecified ratios of sums and paired-request uncertainty for N8/E8/S8."""

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from online_feedback.storage import write_json
from .evaluation import METHODS, PANEL_SIZE, REPEATS, SCOPE, audit_raw_records

COUNTS = ("verification_calls", "native_accepted_draft_tokens", "accepted_draft_tokens",
          "committed_progress", "output_tokens")
TIMES = ("request_latency_s", "native_generate_s", "cache_maintenance_s")
CONTRASTS = (("E", "N"), ("E", "S"), ("S", "N"))
BOOTSTRAP_SEED = 1708
BOOTSTRAP_DRAWS = 10000


def reduce_requests(rows):
    """One count trajectory and three-repeat mean time per arm and request."""
    grouped, identities = defaultdict(list), {}
    for row in rows:
        if row.get("phase") == "warmup":
            continue
        if (row.get("phase") != "scored" or row.get("method") not in METHODS
                or row.get("event") != 8 or row.get("scope") != SCOPE
                or row.get("valid_measurement") is not True or row.get("errors")
                or row.get("validation_status") != "valid"):
            raise ValueError("Only complete valid event-eight scored records enter analysis")
        ordinal = row["panel_ordinal"]
        identity = row["source_id"], row["request_id"]
        if ordinal in identities and identities[ordinal] != identity:
            raise ValueError("Panel identity changed between measurements")
        identities[ordinal] = identity
        grouped[ordinal, row["method"]].append(row)
    if (sorted(identities) != list(range(PANEL_SIZE))
            or len({v[0] for v in identities.values()}) != PANEL_SIZE
            or len({v[1] for v in identities.values()}) != PANEL_SIZE):
        raise ValueError("Missing or duplicated fixed-panel requests")
    result = []
    for ordinal, (source_id, request_id) in sorted(identities.items(), key=lambda item: item[1][0]):
        paired = dict(panel_ordinal=ordinal, source_id=source_id, request_id=request_id)
        for method in METHODS:
            records = sorted(grouped[ordinal, method], key=lambda row: row["repeat"])
            if [row["repeat"] for row in records] != list(range(REPEATS)):
                raise ValueError("Every request requires three distinct repeats from every arm")
            first = records[0]
            for key in COUNTS:
                if (type(first.get(key)) is not int or first[key] < 0
                        or any(row[key] != first[key] for row in records)):
                    raise ValueError("Invalid or repeat-inconsistent native counts")
            calls, progress, native = first["verification_calls"], first["committed_progress"], first["native_accepted_draft_tokens"]
            if (native != progress - calls or progress != first["output_tokens"] - 1
                    or first["accepted_draft_tokens"] > progress
                    or (calls == 0 and progress != 0)):
                raise ValueError("Native acceptance/progress identities failed")
            if (first["finish_reason"] not in ("eos", "length")
                    or first["cap_hit"] != (first["finish_reason"] == "length")
                    or (first["cap_hit"] and first["output_tokens"] != 2048)
                    or any(row[field] != first[field] for row in records
                           for field in ("checkpoint_id", "physical_gpu_uuid", "finish_reason", "cap_hit", "native_acceptance_histogram"))):
                raise ValueError("Endpoint, device, termination or native histogram changed")
            if any(not np.isfinite(row[key]) or row[key] < 0
                   or (key != "cache_maintenance_s" and row[key] == 0)
                   for row in records for key in TIMES):
                raise ValueError("Invalid request latency")
            histogram = first["native_acceptance_histogram"]
            if (sum(histogram.values()) != calls
                    or sum(int(k) * value for k, value in histogram.items()) != native):
                raise ValueError("Native histogram differs from its counts")
            paired[method] = {key: first[key] for key in COUNTS}
            paired[method].update({key: float(np.mean([row[key] for row in records])) for key in TIMES})
            paired[method].update(checkpoint_id=first["checkpoint_id"], cap_hit=first["cap_hit"],
                finish_reason=first["finish_reason"], native_acceptance_histogram=histogram,
                repeat_request_latency_s=[row["request_latency_s"] for row in records])
        if len({(paired[m]["output_tokens"], paired[m]["finish_reason"]) for m in METHODS}) != 1:
            raise ValueError("Paired output length or termination differs")
        result.append(paired)
    if len({row["physical_gpu_uuid"] for row in rows}) != 1:
        raise ValueError("All final serving must use one physical GPU")
    return result


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator > 0 else None


def method_totals(paired, method):
    values = [row[method] for row in paired]
    totals = {key: sum(row[key] for row in values) for key in (*COUNTS, *TIMES)}
    calls = totals["verification_calls"]
    histogram = Counter()
    for row in values:
        histogram.update(row["native_acceptance_histogram"])
    positive = [row for row in values if row["verification_calls"] > 0]
    totals.update(native_accepted_tokens_per_call=ratio(totals["native_accepted_draft_tokens"], calls),
        matched_accepted_tokens_per_call=ratio(totals["accepted_draft_tokens"], calls),
        committed_progress_per_call=ratio(totals["committed_progress"], calls),
        calls_per_1000_output_tokens=ratio(1000 * calls, totals["output_tokens"]),
        equal_request_native_acceptance_mean=(float(np.mean([r["native_accepted_draft_tokens"] / r["verification_calls"] for r in positive])) if positive else None),
        equal_request_denominator=len(positive), zero_call_requests=len(values)-len(positive),
        cap_hit_requests=sum(row["cap_hit"] for row in values),
        native_acceptance_histogram=dict(sorted(histogram.items(), key=lambda item: int(item[0]))))
    return totals


def interval(point, draws):
    finite = np.isfinite(draws)
    # Undefined bootstrap draws remain visible; do not discard them to earn a pass.
    bounds = np.quantile(draws, [.025, .975], method="linear").tolist() if finite.all() else None
    return dict(point=point, ci95=bounds, undefined_draws=int((~finite).sum()))


def compare(paired, candidate, reference, indices):
    def array(method, field):
        return np.asarray([r[method][field] for r in paired], dtype=np.float64)
    ca, ra = array(candidate, "native_accepted_draft_tokens"), array(reference, "native_accepted_draft_tokens")
    cv, rv = array(candidate, "verification_calls"), array(reference, "verification_calls")
    ct, rt = array(candidate, "request_latency_s"), array(reference, "request_latency_s")
    with np.errstate(divide="ignore", invalid="ignore"):
        c_accept = ca[indices].sum(1) / cv[indices].sum(1)
        r_accept = ra[indices].sum(1) / rv[indices].sum(1)
        acceptance_draws = c_accept / r_accept - 1
        acceptance_difference = c_accept - r_accept
        time_draws = 1 - ct[indices].sum(1) / rt[indices].sum(1)
        call_draws = 1 - cv[indices].sum(1) / rv[indices].sum(1)
    c_point, r_point = ratio(ca.sum(), cv.sum()), ratio(ra.sum(), rv.sum())
    improvement = None if c_point is None or r_point is None or r_point == 0 else c_point / r_point - 1
    absolute = None if c_point is None or r_point is None else c_point - r_point
    savings = rv - cv
    repeat_reductions = []
    for repeat in range(REPEATS):
        c_sum = sum(r[candidate]["repeat_request_latency_s"][repeat] for r in paired)
        r_sum = sum(r[reference]["repeat_request_latency_s"][repeat] for r in paired)
        repeat_reductions.append(1 - c_sum / r_sum)
    return dict(candidate=candidate, reference=reference,
        time_reduction=interval(float(1 - ct.sum() / rt.sum()), time_draws),
        speedup=float(rt.sum() / ct.sum()),
        acceptance_improvement=interval(improvement, acceptance_draws),
        acceptance_difference=interval(absolute, acceptance_difference),
        call_reduction=interval(None if rv.sum() == 0 else float(1-cv.sum()/rv.sum()), call_draws),
        repeat_time_reductions=repeat_reductions,
        requests_fewer_calls=int((savings > 0).sum()), requests_more_calls=int((savings < 0).sum()),
        request_call_ties=int((savings == 0).sum()),
        gross_calls_saved=int(savings[savings > 0].sum()), gross_calls_added=int(-savings[savings < 0].sum()))


def practical_pass(effect):
    time_effect, acceptance = effect["time_reduction"], effect["acceptance_improvement"]
    return bool(time_effect["point"] >= .005 and time_effect["ci95"] is not None
        and time_effect["ci95"][0] > 0 and acceptance["ci95"] is not None
        and acceptance["ci95"][0] > 0 and all(value > 0 for value in effect["repeat_time_reductions"]))


def decision(comparisons):
    en, es = comparisons["E_vs_N"], comparisons["E_vs_S"]
    clears_s = all(es[key]["ci95"] is not None and es[key]["ci95"][0] > 0
                   for key in ("time_reduction", "acceptance_improvement"))
    inferior_s = all(es[key]["ci95"] is not None and es[key]["ci95"][1] < 0
                     for key in ("time_reduction", "acceptance_improvement"))
    if practical_pass(en):
        outcome = "FEASIBLE_E_WITH_ADDED_VALUE_OVER_FIXED_LR_CONTROL" if clears_s else "FEASIBLE_E_LR_ALTERNATIVE_UNRESOLVED"
    else:
        outcome = "NO_PRACTICAL_ADVANCEMENT_THIS_PILOT"
    time_bounds = en["time_reduction"]["ci95"]
    accept_bounds = en["acceptance_improvement"]["ci95"]
    if practical_pass(en):
        failure_kind = None
    elif time_bounds is not None and time_bounds[1] < 0:
        failure_kind = "negative_serving_time"
    elif accept_bounds is not None and accept_bounds[0] > 0 and not (time_bounds is not None and time_bounds[0] > 0):
        failure_kind = "acceptance_only_without_demonstrated_time_benefit"
    elif 0 < en["time_reduction"]["point"] < .005:
        failure_kind = "subthreshold_time_point"
    else:
        failure_kind = "unresolved_or_mixed"
    return dict(outcome=outcome, scope=SCOPE, feasibility_pass=practical_pass(en),
        nonadvancement_kind=failure_kind,
        e_vs_s="E_INFERIOR_TO_S" if inferior_s else "E_CLEARLY_BETTER_ON_BOTH" if clears_s else "UNRESOLVED_OR_METRIC_SPECIFIC",
        e_vs_s_metric_directions={key: ("favors_E" if es[key]["ci95"] is not None and es[key]["ci95"][0] > 0 else
            "favors_S" if es[key]["ci95"] is not None and es[key]["ci95"][1] < 0 else "unresolved")
            for key in ("time_reduction", "acceptance_improvement")},
        practical_checks=dict(time_point_at_least_half_percent=en["time_reduction"]["point"] >= .005,
            time_lower_bound_positive=en["time_reduction"]["ci95"] is not None and en["time_reduction"]["ci95"][0] > 0,
            acceptance_lower_bound_positive=en["acceptance_improvement"]["ci95"] is not None and en["acceptance_improvement"]["ci95"][0] > 0,
            all_three_time_directions_positive=all(v > 0 for v in en["repeat_time_reductions"])),
        s_vs_n_descriptive_practical_pass=practical_pass(comparisons["S_vs_N"]),
        automatic_followup_authorized=False)


def summarize(rows):
    paired = reduce_requests(rows)
    indices = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED)).integers(
        0, len(paired), size=(BOOTSTRAP_DRAWS, len(paired)))
    comparisons = {a + "_vs_" + b: compare(paired, a, b, indices) for a, b in CONTRASTS}
    scored = [row for row in rows if row["phase"] == "scored"]
    def view(selected):
        return {method: {key: sum(r[key] for r in selected if r["method"] == method)
                         for key in (*COUNTS, *TIMES)} for method in METHODS}
    return dict(scope=SCOPE, request_count=len(paired), count_trajectory_repeat=0,
        method_totals={method: method_totals(paired, method) for method in METHODS},
        comparisons=comparisons, decision=decision(comparisons),
        repeat_totals={str(r): view([row for row in scored if row["repeat"] == r]) for r in range(REPEATS)},
        block_views={f"repeat_{r}_block_{b}": view([row for row in scored if row["repeat"] == r and row["block_ordinal"] == b])
                     for r in range(REPEATS) for b in range(4)},
        bootstrap=dict(draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED, generator="PCG64", order="lexicographic source_id",
            unit="paired whole request, all arms and three repeat times together", percentiles=[2.5, 97.5], method="linear"),
        limitations=["One training origin and seed on deliberately reused development data.",
            "Bootstrap intervals do not correct prior adaptive dataset/method use or shared session disturbances.",
            "No independent confirmation, training-inclusive speedup, or long-horizon claim."], paired_requests=paired)


def write_report(root, *, native_trace_resolver=None):
    """Revalidate raw traces, then write compact tables and the fixed decision."""
    root = Path(root)
    audit = audit_raw_records(root, native_trace_resolver=native_trace_resolver)
    write_json(root / "RAW_AUDIT.json", {key: value for key, value in audit.items() if key != "rows"})
    if audit["status"] != "passed":
        write_json(root / "SUMMARY.json", dict(status="INVALID_OR_INCOMPLETE", scope=SCOPE, errors=audit["errors"]))
        return audit
    summary = summarize(audit["rows"])
    summary["execution_commit"] = audit["execution_commit"]
    write_json(root / "SUMMARY.json", summary)
    for phase, filename in (("scored", "ALL_REPEATS.csv"), ("warmup", "WARMUPS.csv")):
        fields = ["source_id", "request_id", "panel_ordinal", "method", "repeat", "block_ordinal",
            "position", "activation", "physical_gpu_uuid", "checkpoint_id", "finish_reason", "cap_hit", *COUNTS, *TIMES]
        with (root / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in audit["rows"]:
                if row["phase"] == phase:
                    writer.writerow({key: row[key] for key in fields})
    with (root / "PER_REQUEST.csv").open("w", newline="") as stream:
        fields = ["source_id", "request_id", "panel_ordinal", "method", *COUNTS, *TIMES]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in summary["paired_requests"]:
            for method in METHODS:
                writer.writerow({**{key: row[key] for key in fields[:3]}, "method": method,
                    **{key: row[method][key] for key in (*COUNTS, *TIMES)}})
    lines = ["# Persistent prefix-withholding event-eight comparison", "", SCOPE, "",
        "Execution commit: `" + audit["execution_commit"] + "`.", "",
        "Decision: **" + summary["decision"]["outcome"] + "**.", "",
        "| Comparison | Request-time reduction | Native acceptance improvement |", "|---|---:|---:|"]
    for name, effect in summary["comparisons"].items():
        def show(value):
            if value["point"] is None:
                return "undefined"
            bounds = value["ci95"]
            interval_text = "undefined interval" if bounds is None else f"95% CI [{100*bounds[0]:+.3f}, {100*bounds[1]:+.3f}]%"
            return f"{100*value['point']:+.3f}% ({interval_text})"
        lines.append(f"| {name} | {show(effect['time_reduction'])} | {show(effect['acceptance_improvement'])} |")
    lines.extend(["", "Native acceptance uses accepted draft tokens per verification; the stop-clipped matching counter remains separate.",
        "Counts use one validated trajectory per request. Latencies use the mean of three fresh repeats.", "",
        *summary["limitations"], "", "No subsequent experiment is authorized by this result."])
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")
    return summary
