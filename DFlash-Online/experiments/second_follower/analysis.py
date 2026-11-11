"""Fixed two-arm endpoint reductions from audited records; no model work.

Calls use one repeat-consistent trajectory. Each request contributes the mean
of its two timing repeats and stays paired across arms in every bootstrap draw.
The later matched-parent-start diagnostic is a separate estimand.
"""

from collections import defaultdict
import math

import numpy as np

from .evaluation import ENDPOINT_REPEATS, EVENT, METHODS, PANEL_SIZE

COUNT_FIELDS = (
    "verification_calls",
    "accepted_draft_tokens",
    "native_accepted_draft_tokens",
    "committed_progress",
    "output_tokens",
)
TIMINGS = ("request_latency_s", "native_generate_s", "cache_maintenance_s")
BOOTSTRAP_SEED = 20261113
RESAMPLES = 10000
CONTRAST = "SECOND_vs_NATIVE"


def reduce_stage(rows, *, dataset="gsm8k", event=EVENT, panel_size=PANEL_SIZE):
    """Require the complete fixed endpoint before reducing its repeated rows."""
    if dataset != "gsm8k" or event != EVENT or type(panel_size) is not int or panel_size != PANEL_SIZE:
        raise ValueError("Require the sealed 256-request GSM8K event-five endpoint")
    if len(rows) != PANEL_SIZE * len(METHODS) * ENDPOINT_REPEATS:
        raise ValueError("Require all 1,024 timed completions; no complete-case subset")
    grouped = defaultdict(list)
    identities = {}
    checkpoints = defaultdict(set)
    for row in rows:
        if (
            row["phase"] != "final"
            or row["event"] != event
            or row["dataset"] != dataset
            or row["seed"] != 17
            or row["method"] not in METHODS
            or row["validation_status"] != "valid"
            or row["valid_measurement"] is not True
            or row["errors"]
            or row.get("structural_errors")
        ):
            raise ValueError("Only valid prescribed endpoint executions enter analysis")
        ordinal, request_id = row["panel_ordinal"], row["request_id"]
        if type(ordinal) is not int or not 0 <= ordinal < PANEL_SIZE:
            raise ValueError("Invalid panel ordinal")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("Missing request identity")
        if ordinal in identities and identities[ordinal] != request_id:
            raise ValueError("A panel ordinal has multiple request identities")
        if type(row["repeat"]) is not int or row["repeat"] not in (0, 1):
            raise ValueError("Require distinct sweep-zero and sweep-one measurements")
        identities[ordinal] = request_id
        grouped[ordinal, row["method"]].append(row)
        checkpoint = row["checkpoint_id"]
        if not isinstance(checkpoint, str) or not checkpoint:
            raise ValueError("Missing checkpoint identity")
        checkpoints[row["method"]].add(checkpoint)
    if sorted(identities) != list(range(PANEL_SIZE)) or len(set(identities.values())) != PANEL_SIZE:
        raise ValueError("Missing or duplicated paired request identities")
    if any(len(checkpoints[method]) != 1 for method in METHODS) or checkpoints["NATIVE"] == checkpoints["SECOND"]:
        raise ValueError("Require two distinct, fixed arm exports across all requests")

    paired = []
    for ordinal, request_id in sorted(identities.items()):
        value = dict(seed=17, dataset=dataset, event=event, panel_ordinal=ordinal, request_id=request_id)
        for method in METHODS:
            records = sorted(grouped[ordinal, method], key=lambda row: row["repeat"])
            if [row["repeat"] for row in records] != [0, 1]:
                raise ValueError("Require each arm's two distinct repeats for every request")
            first = records[0]
            for field in COUNT_FIELDS:
                if any(
                    type(row[field]) is not int
                    or row[field] < 0
                    or row[field] != first[field]
                    for row in records
                ):
                    raise ValueError("Invalid or inconsistent repeated counts")
            seed_eos = (
                first["verification_calls"] == 0
                and first["output_tokens"] == 1
                and first["committed_progress"] == 0
                and first["finish_reason"] == "eos"
                and all(row.get("first_round_present") is False for row in records)
            )
            if first["verification_calls"] == 0 and not seed_eos:
                raise ValueError("Zero calls require an immediate target-seed EOS")
            if (
                first["accepted_draft_tokens"] > first["committed_progress"]
                or first["native_accepted_draft_tokens"]
                != first["committed_progress"] - first["verification_calls"]
                or first["committed_progress"] != first["output_tokens"] - 1
            ):
                raise ValueError("Native acceptance or committed-progress accounting differs")
            if (
                any(
                    row["finish_reason"] != first["finish_reason"]
                    or row["cap_hit"] != first["cap_hit"]
                    or type(row["cap_hit"]) is not bool
                    for row in records
                )
                or first["finish_reason"] not in ("eos", "length")
                or first["cap_hit"] != (first["finish_reason"] == "length")
                or not 1 <= first["output_tokens"] <= 2048
                or (first["cap_hit"] and first["output_tokens"] != 2048)
            ):
                raise ValueError("Invalid or changed termination across repeats")
            for row in records:
                for field in TIMINGS:
                    if (
                        type(row[field]) not in (int, float)
                        or not math.isfinite(row[field])
                        or row[field] < 0
                        or (field != "cache_maintenance_s" and row[field] == 0)
                    ):
                        raise ValueError("Invalid latency measurement")
            value[method] = {field: first[field] for field in COUNT_FIELDS}
            value[method].update(
                {field: float(np.mean([row[field] for row in records])) for field in TIMINGS}
            )
            value[method].update(
                cap_hit=first["cap_hit"],
                finish_reason=first["finish_reason"],
                checkpoint_id=first["checkpoint_id"],
                count_trajectory_repeat=0,
                timing_repeats=ENDPOINT_REPEATS,
            )
        if len({(value[method]["output_tokens"], value[method]["finish_reason"]) for method in METHODS}) != 1:
            raise ValueError("Paired canonical output length or termination differs")
        paired.append(value)
    return paired


def reduction(candidate, reference, indices):
    """Ratio of paired sums; undefined zero denominators remain disclosed."""
    candidate_sum, reference_sum = float(candidate.sum()), float(reference.sum())
    sampled_reference = reference[indices].sum(axis=1)
    sampled_difference = (reference[indices] - candidate[indices]).sum(axis=1)
    draws = np.full(len(indices), np.nan)
    np.divide(100 * sampled_difference, sampled_reference, out=draws, where=sampled_reference > 0)
    finite = draws[np.isfinite(draws)]
    return dict(
        point_pct=100 * (reference_sum - candidate_sum) / reference_sum if reference_sum > 0 else None,
        ci95_pct=np.quantile(finite, [.025, .975], method="linear").tolist() if len(finite) else None,
        undefined_bootstrap_draws=int(len(draws) - len(finite)),
        candidate_sum=candidate_sum,
        reference_sum=reference_sum,
    )


def method_totals(values):
    total = {field: int(values[field].sum()) for field in COUNT_FIELDS}
    total.update({field: float(values[field].sum()) for field in TIMINGS})
    calls = total["verification_calls"]
    if calls <= 0:
        raise ValueError("No positive pooled call denominator")
    total.update(
        calls_per_1000_output_tokens=1000 * calls / total["output_tokens"],
        matched_accepted_tokens_per_call=total["accepted_draft_tokens"] / calls,
        native_accepted_tokens_per_call=total["native_accepted_draft_tokens"] / calls,
        committed_progress_per_call=total["committed_progress"] / calls,
    )
    return total


def _summarize(rows, paired):
    arrays = {
        method: {
            field: np.array([row[method][field] for row in paired], dtype=np.float64)
            for field in (*COUNT_FIELDS, *TIMINGS)
        }
        for method in METHODS
    }
    totals = {method: method_totals(values) for method, values in arrays.items()}
    indices = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED)).integers(
        0, PANEL_SIZE, size=(RESAMPLES, PANEL_SIZE)
    )
    native, second = arrays["NATIVE"], arrays["SECOND"]
    delta = native["verification_calls"] - second["verification_calls"]
    time_delta = native["request_latency_s"] - second["request_latency_s"]
    effect = dict(
        call_reduction=reduction(second["verification_calls"], native["verification_calls"], indices),
        request_wins=int((delta > 0).sum()),
        request_regressions=int((delta < 0).sum()),
        request_ties=int((delta == 0).sum()),
        request_denominator=PANEL_SIZE,
        gross_calls_saved=int(np.maximum(delta, 0).sum()),
        gross_calls_added=int(np.maximum(-delta, 0).sum()),
        net_calls_saved=int(delta.sum()),
        request_latency_wins=int((time_delta > 0).sum()),
        request_latency_regressions=int((time_delta < 0).sum()),
        request_latency_ties=int((time_delta == 0).sum()),
    )
    effect.update({field + "_reduction": reduction(second[field], native[field], indices) for field in TIMINGS})
    for method in METHODS:
        totals[method]["cap_hit_requests"] = sum(row[method]["cap_hit"] for row in paired)
        totals[method]["seed_eos_requests"] = sum(row[method]["verification_calls"] == 0 for row in paired)
    repeats = {
        str(repeat): {
            method: {
                field: sum(
                    row[field] for row in rows
                    if row["repeat"] == repeat and row["method"] == method
                )
                for field in (*COUNT_FIELDS, *TIMINGS)
            }
            for method in METHODS
        }
        for repeat in range(ENDPOINT_REPEATS)
    }
    sweeps = {}
    for repeat, values in repeats.items():
        sweeps[repeat] = {}
        for field in ("verification_calls", *TIMINGS):
            denominator = values["NATIVE"][field]
            sweeps[repeat][field + "_reduction_pct"] = (
                100 * (denominator - values["SECOND"][field]) / denominator
                if denominator > 0 else None
            )
    return dict(
        dataset="gsm8k",
        event=EVENT,
        request_count=PANEL_SIZE,
        timed_completions=len(rows),
        bootstrap=dict(
            resamples=RESAMPLES, seed=BOOTSTRAP_SEED, generator="PCG64",
            percentiles=[2.5, 97.5], method="linear",
            unit="Paired whole request carrying both arms; two timing repeats form one mean.",
        ),
        method_totals=totals,
        comparisons={CONTRAST: effect},
        primary_bootstrap_defined=all(
            effect[field]["undefined_bootstrap_draws"] == 0
            for field in ("call_reduction", "request_latency_s_reduction")
        ),
        repeat_totals=repeats,
        sweep_reductions=sweeps,
        primary=[CONTRAST],
        intervals="Paired-request percentile intervals conditional on this fixed panel and inherited seed.",
        timing_scope="Fresh checkpoint-ready serving; training, publication and transfers have separate clocks.",
        comparison_scope=(
            "Exact PR39 panel with known prior project exposure; "
            "reused native control weights and one new candidate event."
        ),
        mechanism_scope="Matched later-parent-start diagnostics are separate from this complete-request endpoint.",
    )


def summarize(rows, *, dataset="gsm8k", event=EVENT, panel_size=PANEL_SIZE):
    return _summarize(rows, reduce_stage(rows, dataset=dataset, event=event, panel_size=panel_size))


def request_table(paired):
    """Flat per-request count trajectories, timing means and signed differences."""
    result = []
    for row in paired:
        flat = {key: row[key] for key in ("seed", "dataset", "event", "panel_ordinal", "request_id")}
        for method in METHODS:
            flat.update({method + "_" + key: value for key, value in row[method].items()})
        flat["calls_saved"] = row["NATIVE"]["verification_calls"] - row["SECOND"]["verification_calls"]
        if flat["calls_saved"] > 0:
            flat["call_outcome"] = "improved"
        elif flat["calls_saved"] < 0:
            flat["call_outcome"] = "worse"
        else:
            flat["call_outcome"] = "tied"
        for field in TIMINGS:
            flat[field + "_saved"] = row["NATIVE"][field] - row["SECOND"][field]
        result.append(flat)
    return result


def endpoint_from_audit(audit, panel_size=PANEL_SIZE):
    """Only a complete passed raw audit authorizes endpoint inference."""
    try:
        if audit.get("status") != "passed" or audit.get("errors") or audit.get("model_work") != 0:
            raise ValueError("Require a passed read-only native raw audit")
        paired = reduce_stage(audit["rows"], panel_size=panel_size)
        summary = _summarize(audit["rows"], paired)
        if not summary["primary_bootstrap_defined"]:
            return dict(
                status="undefined_primary_bootstrap",
                inference_permitted=False,
                summary=summary,
                errors=[
                    "Primary bootstrap contains zero-denominator draws; "
                    "finite-draw intervals are descriptive only."
                ],
            ), paired
        return dict(status="passed", inference_permitted=True, summary=summary), paired
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        return dict(status="invalid_or_incomplete", inference_permitted=False, errors=[repr(error)]), []


def decision(summary):
    """Apply the fixed endpoint rule without authorizing another configuration."""
    if summary is None:
        return dict(outcome="invalid_or_incomplete", automatic_followup_authorized=False)
    effect = summary["comparisons"][CONTRAST]
    if any(
        effect[field]["undefined_bootstrap_draws"]
        for field in ("call_reduction", "request_latency_s_reduction")
    ):
        return dict(
            outcome="undefined_primary_bootstrap",
            call_benefit_supported=False,
            latency_benefit_supported=False,
            latency_regression_supported=False,
            automatic_followup_authorized=False,
            scope=(
                "Undefined primary draws prevent a supported endpoint claim; "
                "retain valid raw results and descriptive reductions."
            ),
        )
    calls = effect["call_reduction"]["ci95_pct"]
    timing = effect["request_latency_s_reduction"]["ci95_pct"]
    sweeps = [
        summary["sweep_reductions"][str(i)]["request_latency_s_reduction_pct"]
        for i in range(ENDPOINT_REPEATS)
    ]
    call_benefit = calls is not None and calls[0] > 0
    call_harm = calls is not None and calls[1] < 0
    timing_regression = timing is not None and timing[1] < 0
    if call_benefit:
        outcome = "call_benefit_supported"
    elif call_harm:
        outcome = "call_harm_supported"
    else:
        outcome = "call_difference_unresolved"
    return dict(
        outcome=outcome,
        call_benefit_supported=call_benefit,
        latency_benefit_supported=(
            timing is not None and timing[0] > 0
            and all(value is not None and value > 0 for value in sweeps)
        ),
        latency_regression_supported=timing_regression,
        sweep_reductions_pct=sweeps,
        stop_serving_promotion=bool(call_benefit and timing_regression),
        local_explanation=(
            "Assess LONG and census-weighted LONG+SHORT changes separately; "
            "report donor loss even if calls improve."
        ),
        automatic_followup_authorized=False,
        scope=(
            "One inherited seed and previously measured panel; unresolved intervals "
            "do not establish equivalence or generalization."
        ),
    )


def analyze(rows):
    """Arithmetic entry point for already audited rows; prefer endpoint_from_audit."""
    paired = reduce_stage(rows)
    summary = _summarize(rows, paired)
    return dict(summary=summary, request_rows=request_table(paired), decision=decision(summary))
