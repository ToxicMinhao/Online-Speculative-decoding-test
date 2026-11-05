"""Matched-start route pricing and its fixed, optimistic budget decision.

Times are synchronized wall milliseconds through proposal-ready, including CPU
wait/dispatch. This module does not time kernels or infer acceptance quality.
"""

from collections import Counter, defaultdict
import math
import numpy as np


ROUTES = ("FIVE", "THREE", "ADAPTIVE3", "ADAPTIVE5")
COHORT_SIZE = 16
MAX_STARTS = 8
REPEATS = 4
MAX_MEASUREMENTS = COHORT_SIZE * MAX_STARTS * len(ROUTES) * REPEATS
BOOTSTRAP_SEED = 20261105
BOOTSTRAP_DRAWS = 10000
TARGET_PCT = 2.0


def select_round_starts(observed_rounds):
    """Evenly spaced distinct native starts, including first and last."""
    if type(observed_rounds) is not int or observed_rounds < 1:
        raise ValueError("A profiling request must have at least one native round")
    count = min(MAX_STARTS, observed_rounds)
    if count == 1:
        return [0]
    return [i * (observed_rounds - 1) // (count - 1) for i in range(count)]


def pricing_schedule(prompt_records):
    """Four balanced route orders at every preselected canonical start."""
    if len(prompt_records) != COHORT_SIZE:
        raise ValueError("Pricing uses exactly the first 16 READOUT requests")
    if len({row["request_id"] for row in prompt_records}) != COHORT_SIZE:
        raise ValueError("Pricing prompt identities must be distinct")
    schedule = []
    for prompt_ordinal, prompt in enumerate(prompt_records):
        for anchor_ordinal, start in enumerate(select_round_starts(prompt["observed_rounds"])):
            for repeat in range(REPEATS):
                rotation = (prompt_ordinal + anchor_ordinal + repeat) % len(ROUTES)
                order = ROUTES[rotation:] + ROUTES[:rotation]
                for position, route in enumerate(order):
                    schedule.append(dict(
                        request_id=prompt["request_id"], prompt_ordinal=prompt_ordinal,
                        round_index=start, anchor_ordinal=anchor_ordinal,
                        repeat=repeat, route=route, route_position=position,
                    ))
    return schedule


def _key(row):
    return row["request_id"], row["round_index"], row["repeat"], row["route"]


def analyze_pricing(measurements, prompt_records):
    """Analyze the complete fixed grid, or return visible incomplete accounting.

    Every measurement needs ``valid=True`` after the native cache/proposal gate.
    Invalid/missing measurements cannot supply an optimistic screening result.
    The caller retains raw records before calling this pure reducer.
    """
    schedule = pricing_schedule(prompt_records)
    expected = {_key(row): row for row in schedule}
    seen, valid, errors = Counter(), {}, []
    for ordinal, row in enumerate(measurements):
        key = _key(row)
        seen[key] += 1
        error = None
        if key not in expected:
            error = "outside_sealed_schedule"
        elif seen[key] != 1:
            error = "duplicate_measurement"
        elif ordinal >= len(schedule) or key != _key(schedule[ordinal]):
            error = "measurement_order_differs"
        elif row.get("valid") is not True:
            error = "native_validation_failed_or_missing"
        elif type(row.get("elapsed_ms")) not in (int, float) or not math.isfinite(row["elapsed_ms"]) or row["elapsed_ms"] <= 0:
            error = "nonpositive_or_nonfinite_wall_time"
        elif any(row.get(field, expected[key][field]) != expected[key][field]
                 for field in ("prompt_ordinal", "anchor_ordinal", "route_position")):
            error = "sealed_order_metadata_differs"
        if error:
            errors.append(dict(record_ordinal=ordinal, key=list(key), error=error))
        else:
            valid[key] = row
    missing = [list(key) for key in expected if key not in seen]
    accounting = dict(
        planned_measurements=len(schedule), returned_measurements=len(measurements),
        maximum_measurements=MAX_MEASUREMENTS, valid_unique_measurements=len(valid),
        invalid_measurements=len(errors), unattempted_measurements=len(missing),
        missing=missing, errors=errors,
    )
    result = dict(
        schema="block_depth_pricing_v1", accounting=accounting,
        timing_scope="synchronized wall through proposal ready; includes route dispatch and CPU waits",
        bootstrap=dict(seed=BOOTSTRAP_SEED, generator="PCG64", draws=BOOTSTRAP_DRAWS,
                       percentiles=[2.5, 97.5], method="linear", unit="paired prompt"),
        economic_target_pct=TARGET_PCT,
        qualification_warmup_and_graph_work_included=False,
        interpretation="Optimistic same-progress route-cost screen, not a bound when proposals improve",
    )
    if missing or errors or len(valid) != len(schedule):
        return dict(result, status="incomplete_or_invalid", decision="blocked", optimistic_savings=None)

    per_prompt, per_start = [], []
    for prompt in prompt_records:
        request_id, rounds = prompt["request_id"], prompt["observed_rounds"]
        request_s = prompt["request_latency_s"]
        if type(request_s) not in (int, float) or not math.isfinite(request_s) or request_s <= 0:
            raise ValueError("The native complete-request timing denominator must be finite and positive")
        starts = select_round_starts(rounds)
        weight = rounds / len(starts)
        route_means, savings, continuing_costs = defaultdict(list), [], []
        for start in starts:
            times = {
                route: float(np.mean([valid[request_id, start, repeat, route]["elapsed_ms"]
                                      for repeat in range(REPEATS)]))
                for route in ROUTES
            }
            saving = max(0.0, times["FIVE"] - min(times["ADAPTIVE3"], times["ADAPTIVE5"]))
            extra = times["ADAPTIVE5"] - times["ADAPTIVE3"]
            per_start.append(dict(
                request_id=request_id, round_index=start, expansion_weight=weight,
                route_mean_ms=times, optimistic_saving_ms=saving,
                paired_continuation_cost_ms=extra,
            ))
            savings.append(saving)
            continuing_costs.append(extra)
            for route, value in times.items():
                route_means[route].append(value)
        per_prompt.append(dict(
            request_id=request_id, observed_rounds=rounds, sampled_rounds=len(starts),
            expansion_weight=weight, native_request_ms=1000 * request_s,
            route_mean_ms={route: float(np.mean(values)) for route, values in route_means.items()},
            optimistic_saved_ms=weight * sum(savings),
            paired_continuation_cost_ms=float(np.mean(continuing_costs)),
        ))
    saved = np.asarray([row["optimistic_saved_ms"] for row in per_prompt], dtype=np.float64)
    complete = np.asarray([row["native_request_ms"] for row in per_prompt], dtype=np.float64)
    draws = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED)).integers(
        0, COHORT_SIZE, size=(BOOTSTRAP_DRAWS, COHORT_SIZE),
    )
    bootstrap = 100 * saved[draws].sum(axis=1) / complete[draws].sum(axis=1)
    interval = np.quantile(bootstrap, [0.025, 0.975], method="linear").tolist()
    effect = dict(
        saved_ms=float(saved.sum()), native_complete_request_ms=float(complete.sum()),
        point_pct=float(100 * saved.sum() / complete.sum()), ci95_pct=interval,
    )
    total_rounds = sum(row["observed_rounds"] for row in per_prompt)
    return dict(
        result, status="complete", per_prompt=per_prompt, per_start=per_start,
        optimistic_savings=effect,
        decision="stop_pricing_only" if interval[1] < TARGET_PCT else "continue_to_readout_fit",
        continuation_cost_ms=float(np.mean([row["paired_continuation_cost_ms"] for row in per_start])),
        continuation_cost_estimator="mean paired ADAPTIVE5 minus ADAPTIVE3 across sampled starts and four repeats",
        target_saved_ms_per_existing_round=0.02 * float(complete.sum()) / total_rounds,
        target_saved_ms_per_half_exiting_round=0.04 * float(complete.sum()) / total_rounds,
    )
