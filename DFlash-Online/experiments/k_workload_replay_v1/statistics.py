"""Prespecified conditional request-panel effects for one fixed trajectory."""
from collections import defaultdict

import numpy as np

from experiments.full_k_gsm8k.statistics import COUNT_FIELDS, TIMINGS, comparison, method_totals
from .contract import ARMS, EVENT_DOMAINS

COMPARISONS = (("R", "U"), ("R", "K"), ("U", "K"))
DOMAINS = ("math", "coding")
BOOTSTRAP_SEED = 20261102
RESAMPLES = 10000


def reduce_final(rows):
    """Count each request once; average its three measured serving times."""
    grouped, identities, checkpoints = defaultdict(list), {}, {}
    for row in rows:
        if (row["phase"] != "final" or row["event"] != 8 or row["seed"] != 17
                or row["method"] not in ARMS or row["validation_status"] != "valid"
                or row["valid_measurement"] is not True or row["errors"] or row["capture"] is not False):
            raise ValueError("Only complete valid capture-off final measurements enter endpoint estimates")
        ordinal = row["panel_ordinal"]
        identity = (row["request_id"], row["domain"])
        if ordinal in identities and identities[ordinal] != identity:
            raise ValueError("Panel ordinal has conflicting request or domain identity")
        identities[ordinal] = identity
        if row["method"] in checkpoints and checkpoints[row["method"]] != row["checkpoint_id"]:
            raise ValueError("A final method must use one checkpoint across the entire panel")
        checkpoints[row["method"]] = row["checkpoint_id"]
        grouped[ordinal, row["method"]].append(row)
    if (sorted(identities) != list(range(256)) or len({v[0] for v in identities.values()}) != 256
            or [identities[i][1] for i in range(256)] != [DOMAINS[i % 2] for i in range(256)]):
        raise ValueError("Require all 256 unique alternating math/coding requests")
    paired = []
    for ordinal, (request_id, domain) in sorted(identities.items()):
        value = dict(panel_ordinal=ordinal, request_id=request_id, domain=domain)
        for method in ARMS:
            records = sorted(grouped[ordinal, method], key=lambda row: row["repeat"])
            if [row["repeat"] for row in records] != [0, 1, 2]:
                raise ValueError("Require three distinct repeats per request and method")
            first = records[0]
            for field in COUNT_FIELDS:
                if any(type(row[field]) is not int or row[field] < 0 or row[field] != first[field] for row in records):
                    raise ValueError("Invalid or unequal repeated count trajectory")
            if first["verification_calls"] <= 0 or first["output_tokens"] <= 0:
                raise ValueError("Call and output denominators must be positive")
            if (first["committed_progress"] != first["output_tokens"] - 1
                    or not 0 <= first["accepted_draft_tokens"] <= first["committed_progress"]
                    or first["finish_reason"] not in ("eos", "length")
                    or first["cap_hit"] is not (first["finish_reason"] == "length")):
                raise ValueError("Output, accepted-progress or termination accounting differs")
            if any(row["checkpoint_id"] != first["checkpoint_id"] or row["finish_reason"] != first["finish_reason"]
                   or row["cap_hit"] != first["cap_hit"] for row in records):
                raise ValueError("Checkpoint or termination differs across repeats")
            for row in records:
                if any(not np.isfinite(row[field]) or row[field] < 0 for field in TIMINGS):
                    raise ValueError("Invalid serving timing")
                if row["request_latency_s"] <= 0 or row["native_generate_s"] <= 0:
                    raise ValueError("Request and native timings must be positive")
            value[method] = {field: first[field] for field in COUNT_FIELDS}
            value[method].update({field: float(np.mean([row[field] for row in records])) for field in TIMINGS})
            value[method].update(checkpoint_id=first["checkpoint_id"], finish_reason=first["finish_reason"],
                                 cap_hit=first["cap_hit"], count_trajectory_repeat=0, timing_repeats=3)
        if len({(value[m]["output_tokens"], value[m]["finish_reason"]) for m in ARMS}) != 1:
            raise ValueError("Cross-arm output length or termination differs")
        paired.append(value)
    return paired


def _arrays(paired):
    return {method: {field: np.array([row[method][field] for row in paired], dtype=np.float64)
                     for field in (*COUNT_FIELDS, *TIMINGS)} for method in ARMS}


def _summary(paired, indices):
    arrays = _arrays(paired)
    totals = {method: method_totals(values) for method, values in arrays.items()}
    for value in totals.values():
        value["output_tokens_per_call"] = value["output_tokens"] / value["verification_calls"]
        value["output_tokens_per_s"] = value["output_tokens"] / value["request_latency_s"]
    effects = {}
    for candidate, reference in COMPARISONS:
        a, b = arrays[candidate], arrays[reference]
        result = comparison(a, b, indices)
        # Outputs are equal within each request. The ratio therefore equals
        # reference total time / candidate total time, including paired draws.
        draws = b["request_latency_s"][indices].sum(axis=1) / a["request_latency_s"][indices].sum(axis=1)
        result["serving_throughput_ratio"] = {
            "point": float(b["request_latency_s"].sum() / a["request_latency_s"].sum()),
            "ci95": np.quantile(draws, [0.025, 0.975], method="linear").tolist(),
            "denominator": "candidate/reference output tokens per second, using per-request three-repeat mean times",
        }
        effects[candidate + "_vs_" + reference] = result
    return dict(request_count=len(paired), method_totals=totals, comparisons=effects)


def summarize(rows):
    paired = reduce_final(rows)
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    # Within each replicate, draw 128 math indices then 128 coding indices.
    draws = rng.integers(0, 128, size=(RESAMPLES, 2, 128))
    pooled_indices = np.concatenate([2 * draws[:, 0], 2 * draws[:, 1] + 1], axis=1)
    result = _summary(paired, pooled_indices)
    result["domains"] = {
        domain: _summary([row for row in paired if row["domain"] == domain], draws[:, index])
        for index, domain in enumerate(DOMAINS)
    }
    result["repeat_totals"] = {
        str(repeat): {method: {field: sum(row[field] for row in rows
            if row["repeat"] == repeat and row["method"] == method)
            for field in (*COUNT_FIELDS, *TIMINGS)} for method in ARMS}
        for repeat in range(3)
    }
    result["bootstrap"] = dict(seed=BOOTSTRAP_SEED, generator="PCG64", resamples=RESAMPLES,
        strata={"math": 128, "coding": 128}, percentiles=[2.5, 97.5], method="linear",
        draw_order="Each replicate: 128 math indices, then 128 coding indices; all arms share sampled identities",
        unit="Distinct paired request; count once after repeat equality, average three times before resampling")
    result["primary"] = "R_vs_U"
    result["required_usefulness"] = "R_vs_K"
    result["scope"] = ("Conditional request-panel intervals for one trained trajectory/seed on a reused development panel; "
                       "not seed robustness, deployment-workload uncertainty, or uncertainty over the adaptive trajectory.")
    result["secondary_intervals"] = "Per-domain, U/K and timing intervals are descriptive and unadjusted."
    result["timing_scope"] = "Capture-off checkpoint-ready C=1 serving; excludes adaptation and publication."
    return result


def summarize_events(rows):
    """Describe adaptive pre-update trajectories; never bootstrap these rows."""
    events = []
    for event in sorted({row["event"] for row in rows}):
        if type(event) is not int or not 1 <= event <= 8:
            raise ValueError("Only the eight fixed pre-update events may be reported")
        selected = [row for row in rows if row["event"] == event]
        grouped = defaultdict(dict)
        for row in selected:
            key = row["request_id"]
            if row["method"] in grouped[key]:
                raise ValueError("Duplicate event request/method measurement")
            grouped[key][row["method"]] = row
        valid = (len(grouped) == 128 and len(selected) == 384
                 and all(set(values) == set(ARMS) for values in grouped.values())
                 and all(row["valid_measurement"] is True
                         and row["domain"] == EVENT_DOMAINS[event - 1]
                         and row["capture"] is (row["method"] == "K")
                         and all(type(row[field]) is int and row[field] >= 0 for field in COUNT_FIELDS)
                         and row["verification_calls"] > 0 for row in selected))
        if valid:
            valid = all(len({value[m]["output_tokens"] for m in ARMS}) == 1 for value in grouped.values())
        if valid and event <= 2:
            valid = all(len({value[m]["verification_calls"] for m in ARMS}) == 1 for value in grouped.values())
        item = dict(event=event, domain=EVENT_DOMAINS[event - 1], returned_rows=len(selected),
                    distinct_requests=len(grouped), valid_complete=valid)
        if valid:
            totals = {method: {field: sum(row[field] for row in selected if row["method"] == method)
                               for field in COUNT_FIELDS} for method in ARMS}
            for value in totals.values():
                value["accepted_tokens_per_call"] = value["accepted_draft_tokens"] / value["verification_calls"]
                value["committed_progress_per_call"] = value["committed_progress"] / value["verification_calls"]
            item.update(method_totals=totals, comparisons={a + "_vs_" + b: {
                "call_reduction_pct": 100 * (1 - totals[a]["verification_calls"] / totals[b]["verification_calls"])
            } for a, b in COMPARISONS})
        events.append(item)
    future = [item for item in events if 3 <= item["event"] <= 8]
    future_summary = dict(events=[item["event"] for item in future], expected_requests=768,
                          valid_complete=len(future) == 6 and all(item["valid_complete"] for item in future))
    if future_summary["valid_complete"]:
        totals = {method: {field: sum(item["method_totals"][method][field] for item in future)
                           for field in COUNT_FIELDS} for method in ARMS}
        future_summary.update(method_totals=totals, comparisons={a + "_vs_" + b: {
            "call_reduction_pct": 100 * (1 - totals[a]["verification_calls"] / totals[b]["verification_calls"])
        } for a, b in COMPARISONS})
        future_summary["domains"] = {}
        for domain in DOMAINS:
            selected = [item for item in future if item["domain"] == domain]
            domain_totals = {method: {field: sum(item["method_totals"][method][field] for item in selected)
                                     for field in COUNT_FIELDS} for method in ARMS}
            future_summary["domains"][domain] = dict(request_count=128 * len(selected), method_totals=domain_totals,
                comparisons={a + "_vs_" + b: {"call_reduction_pct": 100 * (
                    1 - domain_totals[a]["verification_calls"] / domain_totals[b]["verification_calls"])}
                    for a, b in COMPARISONS})
    return dict(events=events, future_events_3_to_8=future_summary, intervals=None,
        scope="Descriptive adaptive trajectory; no independent-request bootstrap. K capture-on and U/R capture-off times are not comparable.")
