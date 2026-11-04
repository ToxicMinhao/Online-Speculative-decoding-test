"""Fixed request-level analysis and bounded decision for the five-policy screen."""

from collections import defaultdict
import numpy as np
from experiments.full_k_gsm8k.statistics import comparison, method_totals
from .evaluation import METHODS, PANEL_SIZE, REPEATS

COUNT_FIELDS = (
    "verification_calls",
    "accepted_draft_tokens",
    "committed_progress",
    "output_tokens",
)
TIMINGS = ("request_latency_s", "native_generate_s", "cache_maintenance_s")
BOOTSTRAP_SEED = 20261104
RESAMPLES = 10000
ANALYSIS_RULES = dict(
    policy="vfd-s2-v1",
    primary="PROGRESS64_vs_FULL",
    statistic="100 * (1 - sum(candidate per-request repeat means) / sum(comparator per-request repeat means))",
    bootstrap=dict(
        seed=BOOTSTRAP_SEED,
        generator="PCG64",
        draws=RESAMPLES,
        percentiles=[2.5, 97.5],
        method="linear",
        unit="paired request",
        shared_draws=True,
    ),
    acceptable_slowdown_pct=0.5,
    clear_simpler_dominance="RANDOM64 or ONCE has strictly lower updater time than PROGRESS64 and PROGRESS64-versus-arm latency interval upper bound < 0",
    efficiency_preference=["ONCE", "RANDOM64", "PROGRESS64"],
    updater_time_scope="Measured logical standalone training-pass worker wall plus scoring/selection worker wall, including packet loads and setup; shared event-1 work charged to every applicable policy; initialization, restoration, preparation and export excluded; compute timers reported separately",
    control_intervals="Descriptive and unadjusted; intervals crossing zero do not establish the added scoring mechanism",
    automatic_extension=False,
)


def reduce_final(rows, requests=None):
    """Require every valid execution; retain one count trajectory and mean time.

    Native token/trace validation belongs to ``evaluation.audit_stage``. This
    reducer repeats the scalar and identity checks so a saved validity flag
    alone cannot admit an inconsistent, incomplete or nonfinite measurement.
    """
    grouped = defaultdict(list)
    identities, checkpoint_ids = {}, {}
    for row in rows:
        if (
            row["phase"] != "final"
            or row["event"] != 4
            or row["seed"] != 17
            or row["method"] not in METHODS
            or row["validation_status"] != "valid"
            or row["valid_measurement"] is not True
            or row["errors"]
        ):
            raise ValueError("Only valid event-4 final executions enter analysis")
        ordinal, request_id, method = (
            row["panel_ordinal"],
            row["request_id"],
            row["method"],
        )
        if (
            type(ordinal) is not int
            or not isinstance(request_id, str)
            or not request_id
        ):
            raise ValueError("Invalid request identity")
        if ordinal in identities and identities[ordinal] != request_id:
            raise ValueError("A panel ordinal has multiple request identities")
        if method in checkpoint_ids and checkpoint_ids[method] != row["checkpoint_id"]:
            raise ValueError("A method must use one unchanged checkpoint throughout")
        identities[ordinal] = request_id
        checkpoint_ids[method] = row["checkpoint_id"]
        grouped[ordinal, method].append(row)
    if (
        len(rows) != PANEL_SIZE * len(METHODS) * REPEATS
        or sorted(identities) != list(range(PANEL_SIZE))
        or len(set(identities.values())) != PANEL_SIZE
    ):
        raise ValueError("Require all 200 distinct requests and all 2000 executions")
    if len(set(checkpoint_ids.values())) != len(METHODS):
        raise ValueError("Require five distinct policy checkpoint identities")
    if requests is not None and [
        (r["panel_ordinal"], r["request_id"]) for r in requests
    ] != sorted(identities.items()):
        raise ValueError("Final identities differ from the sealed request panel")
    paired = []
    for ordinal, request_id in sorted(identities.items()):
        value = dict(seed=17, event=4, panel_ordinal=ordinal, request_id=request_id)
        for method in METHODS:
            records = sorted(grouped[ordinal, method], key=lambda row: row["repeat"])
            if [row["repeat"] for row in records] != list(range(REPEATS)):
                raise ValueError("Every method/request requires two distinct repeats")
            first = records[0]
            for field in COUNT_FIELDS:
                if any(
                    type(row[field]) is not int
                    or row[field] < 0
                    or row[field] != first[field]
                    for row in records
                ):
                    raise ValueError("Invalid or inconsistent repeated counts")
            if (
                first["verification_calls"] <= 0
                or first["output_tokens"] <= 0
                or first["accepted_draft_tokens"] > first["output_tokens"]
                or first["committed_progress"] > first["output_tokens"]
            ):
                raise ValueError(
                    "Invalid count denominator or progress/acceptance accounting"
                )
            if (
                first["finish_reason"] not in ("eos", "length")
                or type(first["cap_hit"]) is not bool
                or first["cap_hit"] != (first["finish_reason"] == "length")
                or any(
                    row["finish_reason"] != first["finish_reason"]
                    or row["cap_hit"] != first["cap_hit"]
                    for row in records
                )
            ):
                raise ValueError("Invalid or inconsistent termination/cap status")
            if any(
                type(row[field]) not in (int, float)
                or not np.isfinite(row[field])
                or row[field] < 0
                for row in records
                for field in TIMINGS
            ):
                raise ValueError("Invalid latency measurement")
            if any(row["request_latency_s"] <= 0 for row in records):
                raise ValueError("Complete-request latency must be positive")
            times = {
                field: float(np.mean([row[field] for row in records]))
                for field in TIMINGS
            }
            value[method] = {field: first[field] for field in COUNT_FIELDS}
            value[method].update(
                **times,
                cap_hit=first["cap_hit"],
                finish_reason=first["finish_reason"],
                checkpoint_id=first["checkpoint_id"],
                count_trajectory_repeat=0,
                timing_repeats=REPEATS
            )
        if (
            len(
                {
                    (value[m]["output_tokens"], value[m]["finish_reason"])
                    for m in METHODS
                }
            )
            != 1
        ):
            raise ValueError("Paired output lengths or termination differ")
        paired.append(value)
    return paired


def interpret(effects, updater_costs):
    """Apply the agreed finite decision tree without treating a screen as adoption."""
    primary = effects["PROGRESS64_vs_FULL"]["request_latency_s_reduction"]
    controls = {
        arm: effects["PROGRESS64_vs_" + arm]["request_latency_s_reduction"]
        for arm in ("STATIC64", "RANDOM64", "ONCE")
    }
    if updater_costs is None:
        return dict(
            decision="pending_cost_evidence",
            automatic_followup_authorized=False,
            reason="Measured logical updater costs are required for the decision.",
        )
    if set(updater_costs) != set(METHODS) or any(
        type(v) not in (int, float) or not np.isfinite(v) or v <= 0
        for v in updater_costs.values()
    ):
        raise ValueError("Require finite positive updater time for each real policy")
    dominates = [
        arm
        for arm in ("ONCE", "RANDOM64")
        if updater_costs[arm] < updater_costs["PROGRESS64"]
        and controls[arm]["ci95_pct"][1] < 0
    ]
    advance = (
        primary["ci95_pct"][0] > 0
        and all(controls[arm]["point_pct"] > 0 for arm in ("STATIC64", "RANDOM64"))
        and not dominates
    )
    efficiency = [
        arm
        for arm in ("ONCE", "RANDOM64", "PROGRESS64")
        if updater_costs[arm] < updater_costs["FULL"]
        and effects[arm + "_vs_FULL"]["request_latency_s_reduction"]["ci95_pct"][0]
        > -0.5
    ]
    if advance:
        decision, preferred = "advance_to_longer_in_domain_run", "PROGRESS64"
        reason = "Prespecified advancement criterion passes; a longer run still requires separate authorization."
    elif "PROGRESS64" in efficiency:
        decision, preferred = "efficiency_only", efficiency[0]
        reason = (
            "Measured updater savings satisfy the declared latency tolerance; prefer the simpler qualifying policy."
            if preferred != "PROGRESS64"
            else "Measured updater savings satisfy the declared latency tolerance."
        )
    else:
        decision, preferred = "otherwise", None
        reason = (
            "Measured latency argues against the rule at this budget."
            if primary["ci95_pct"][1] < 0 or dominates
            else "The bounded screen is inconclusive at this budget."
        )
    return dict(
        decision=decision,
        preferred_policy=preferred,
        reason=reason,
        primary_pass=primary["ci95_pct"][0] > 0,
        simpler_dominating_arms=dominates,
        efficiency_qualified_arms=efficiency,
        scoring_mechanism_established_against_both_selectors=all(
            controls[a]["ci95_pct"][0] > 0 for a in ("STATIC64", "RANDOM64")
        ),
        selector_control_caution="Crossing-zero selector intervals leave added scoring value unestablished even if advancement passes.",
        updater_costs_s=updater_costs,
        updater_scope=ANALYSIS_RULES["updater_time_scope"],
        automatic_followup_authorized=False,
        limits="One training seed and four events on common feedback; no seed-level replication, broad-domain claim or training-inclusive serving speedup.",
    )


def summarize(rows, requests=None, updater_costs=None):
    paired = reduce_final(rows, requests)
    arrays = {
        method: {
            field: np.array([row[method][field] for row in paired], dtype=np.float64)
            for field in (*COUNT_FIELDS, *TIMINGS)
        }
        for method in METHODS
    }
    indices = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED)).integers(
        0, len(paired), size=(RESAMPLES, len(paired))
    )
    contrasts = [
        ("PROGRESS64", arm) for arm in ("FULL", "STATIC64", "RANDOM64", "ONCE")
    ]
    # These descriptive controls are needed to apply the predeclared efficiency preference.
    contrasts += [(arm, "FULL") for arm in ("RANDOM64", "ONCE")]
    effects = {
        a + "_vs_" + b: comparison(arrays[a], arrays[b], indices) for a, b in contrasts
    }
    totals = {method: method_totals(values) for method, values in arrays.items()}
    for method in METHODS:
        totals[method].update(
            mean_request_ms=1000 * totals[method]["request_latency_s"] / PANEL_SIZE,
            cap_hits=sum(row[method]["cap_hit"] for row in paired),
            eos_responses=sum(row[method]["finish_reason"] == "eos" for row in paired),
        )
    repeats = {
        str(repeat): {
            method: {
                field: sum(
                    row[field]
                    for row in rows
                    if row["repeat"] == repeat and row["method"] == method
                )
                for field in (*COUNT_FIELDS, *TIMINGS)
            }
            for method in METHODS
        }
        for repeat in range(REPEATS)
    }
    return dict(
        request_count=len(paired),
        execution_count=len(rows),
        analysis_rules=ANALYSIS_RULES,
        method_totals=totals,
        comparisons=effects,
        repeat_totals=repeats,
        interpretation=interpret(effects, updater_costs),
        primary_measure="complete_request_time",
        secondary_intervals="Control, call and component-time intervals are descriptive and unadjusted.",
        timing_scope="Fresh contemporaneous event4 serving; cache maintenance included; loading, switching and updater time excluded.",
        canonical_scope="Every execution checked against exact prompt/target/runtime-bound canonical IDs and termination.",
    )
