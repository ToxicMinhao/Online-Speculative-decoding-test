"""Paired KL/native diagnostics; historical/fresh times are never compared."""
from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path


POPULATIONS = ("source", "heldout")
BOUNDARY = (
    "Better native drafting means fewer verification calls for identical output tokens in this diagnostic. "
    "It does not establish a serving-time improvement. Historical frozen and fresh S8 source timings "
    "are not a latency comparison. The interpretation patterns identify where to investigate next, "
    "not the underlying cause; they do not establish that forward KL is generally unsuitable."
)


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _clean(value):
    """Keep nonfinite evidence visible in valid JSON rather than silently dropping it."""
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_clean(item) for item in value]
    return value


def _ratio(numerator, denominator, scale=1):
    return scale * numerator / denominator if numerator is not None and denominator else None


def _reduction(candidate, control):
    return 100 * (1 - candidate / control) if candidate is not None and control else None


def _validate(pair):
    # Import lazily so the CLI can call this module without a controller/report cycle.
    from .experiment import validate_pair

    errors = []
    recorded = pair.get("validation") or {}
    if recorded.get("status") != "passed" or recorded.get("errors") != []:
        errors.append("recorded_pair_validation_failed_or_missing")
        errors.extend("recorded:" + str(error) for error in recorded.get("errors", []))
    try:
        checked = validate_pair(pair)
        if isinstance(checked, dict) and checked.get("status") != "passed":
            errors.extend("recomputed:" + str(error) for error in checked.get("errors", []))
            if not checked.get("errors"):
                errors.append("recomputed_pair_validation_failed")
        elif checked is False:
            errors.append("recomputed_pair_validation_failed")
    except (ValueError, TypeError, KeyError, OverflowError, AssertionError) as error:
        errors.append("recomputed:" + str(error))
    return sorted(set(errors))


def comparison(pair, *, extra_errors=()):
    """Keep invalid raw values, but never calculate an improvement from them."""
    errors = sorted(set(_validate(pair) + list(extra_errors)))
    result = {key: pair.get(key) for key in ("request_id", "group_id", "domain", "population")}
    result.update(valid=not errors, invalid_reasons=errors)
    for checkpoint in ("F", "S8"):
        arm = pair.get(checkpoint) or {}
        native, loss = arm.get("native") or {}, arm.get("kl") or {}
        for key in ("verification_calls", "accepted_draft_tokens", "verification_committed_tokens",
                    "initial_output_tokens", "output_tokens", "cap_hit", "eos_hit", "finish_reason",
                    "input_ids_sha256", "checkpoint_identity_sha256"):
            result[f"{checkpoint}_{key}"] = native.get(key)
        for key in ("loss", "eligible_positions", "zero_supervision", "normalization_denominator",
                    "unique_supervised_positions", "anchor_identity_sha256", "fixed_anchor_identity_sha256",
                    "anchor_positions", "loss_mask_sha256", "weights_sha256", "anchor_seed",
                    "sampled_blocks", "eligible_blocks", "unique_supervised_position_ids",
                    "weighted_position_sum", "sampling_and_weights_sha256"):
            result[f"{checkpoint}_{key}"] = loss.get(key)
        calls, tokens = native.get("verification_calls"), native.get("output_tokens")
        counters_valid = all(_finite(native.get(key)) for key in
                             ("verification_calls", "output_tokens", "accepted_draft_tokens", "verification_committed_tokens"))
        for key, numerator, denominator, scale in (
            ("calls_per_1000_output_tokens", calls, tokens, 1000),
            ("accepted_draft_tokens_per_call", native.get("accepted_draft_tokens"), calls, 1),
            ("committed_progress_per_call", native.get("verification_committed_tokens"), calls, 1),
            ("emitted_tokens_per_call", tokens, calls, 1),
        ):
            result[f"{checkpoint}_{key}"] = _ratio(numerator, denominator, scale) if counters_valid else None
        result[f"{checkpoint}_native_origin"] = (
            "fresh" if pair.get("population") == "source" and checkpoint == "S8" else "retained"
        )
        result[f"{checkpoint}_kl_origin"] = "fresh" if pair.get("population") == "heldout" else "retained"
    result["zero_supervision_pair"] = any(result[f"{c}_zero_supervision"] is True for c in ("F", "S8"))
    result["kl_valid"] = (result["valid"] and not result["zero_supervision_pair"] and
                          all(_finite(result[f"{c}_loss"]) for c in ("F", "S8")))
    result["verification_calls_saved"] = (
        result["F_verification_calls"] - result["S8_verification_calls"] if result["valid"] else None
    )
    result["verification_call_reduction_percent"] = (
        _reduction(result["S8_verification_calls"], result["F_verification_calls"]) if result["valid"] else None
    )
    saved = result["verification_calls_saved"]
    result["native_outcome"] = "invalid" if saved is None else "better" if saved > 0 else "worse" if saved < 0 else "tied"
    result["kl_change_S8_minus_F"] = result["S8_loss"] - result["F_loss"] if result["kl_valid"] else None
    result["kl_reduction_percent"] = _reduction(result["S8_loss"], result["F_loss"]) if result["kl_valid"] else None
    change = result["kl_change_S8_minus_F"]
    result["kl_outcome"] = "excluded" if change is None else "better" if change < 0 else "worse" if change > 0 else "tied"
    return result


def pool(rows):
    rows = list(rows)
    valid = [row for row in rows if row["valid"]]
    losses = [row for row in rows if row["kl_valid"]]
    result = dict(requests=len(rows), valid_requests=len(valid), invalid_requests=len(rows) - len(valid),
                  kl_pairs=len(losses), zero_supervision_pairs=sum(row["zero_supervision_pair"] for row in rows))
    for outcome in ("better", "worse", "tied"):
        result[f"native_{outcome}_requests"] = sum(row["native_outcome"] == outcome for row in rows)
        result[f"kl_{outcome}_requests"] = sum(row["kl_outcome"] == outcome for row in rows)
    for checkpoint in ("F", "S8"):
        for key in ("verification_calls", "accepted_draft_tokens", "verification_committed_tokens", "output_tokens"):
            result[f"{checkpoint}_{key}"] = sum(row[f"{checkpoint}_{key}"] for row in valid) if valid else None
        # Equal weight for each already-normalized response, never token-weighted.
        result[f"{checkpoint}_mean_kl"] = sum(row[f"{checkpoint}_loss"] for row in losses) / len(losses) if losses else None
        calls, tokens = result[f"{checkpoint}_verification_calls"], result[f"{checkpoint}_output_tokens"]
        result[f"{checkpoint}_calls_per_1000_output_tokens"] = _ratio(calls, tokens, 1000)
        result[f"{checkpoint}_accepted_draft_tokens_per_call"] = _ratio(result[f"{checkpoint}_accepted_draft_tokens"], calls)
        result[f"{checkpoint}_committed_progress_per_call"] = _ratio(result[f"{checkpoint}_verification_committed_tokens"], calls)
        result[f"{checkpoint}_emitted_tokens_per_call"] = _ratio(tokens, calls)
    result["verification_calls_saved"] = (result["F_verification_calls"] - result["S8_verification_calls"]) if valid else None
    result["verification_call_reduction_percent"] = _reduction(result["S8_verification_calls"], result["F_verification_calls"])
    result["kl_change_S8_minus_F"] = result["S8_mean_kl"] - result["F_mean_kl"] if losses else None
    result["kl_reduction_percent"] = _reduction(result["S8_mean_kl"], result["F_mean_kl"])
    return result


def _patterns(source, heldout):
    patterns = []
    if not source or not heldout:
        return ["Missing paired population evidence; no pattern interpretation."]
    s_calls, h_calls = source["verification_calls_saved"], heldout["verification_calls_saved"]
    s_kl, h_kl = source["kl_change_S8_minus_F"], heldout["kl_change_S8_minus_F"]
    if s_calls is not None and s_calls > 0 and h_calls is not None and h_calls < 0 and h_kl is not None and h_kl > 0:
        patterns.append("Source drafting improves while held-out KL and drafting worsen: investigate transfer under this feedback budget.")
    if h_kl is not None and h_kl < 0 and h_calls is not None and h_calls < 0:
        patterns.append("Held-out KL improves while held-out drafting worsens: investigate the measured KL/native-drafting mismatch in this setup.")
    if s_kl is not None and s_kl < 0 and s_calls is not None and s_calls <= 0:
        patterns.append("Source KL improves without fewer source verification calls: useful native drafting improvement remains unproven even on these source requests.")
    return patterns or ["No single declared diagnostic pattern covers this combination; inspect the paired request results."]


def summarize(result):
    groups = result.get("groups", [])
    raw = [(group, population, row) for group in groups for population in POPULATIONS for row in group.get(population, [])]
    identities = Counter((row.get("group_id"), population, row.get("request_id")) for _, population, row in raw)
    comparisons = []
    for group, population, row in raw:
        errors = []
        if any(row.get(key) != group.get(key) for key in ("group_id", "domain")) or row.get("population") != population:
            errors.append("pair_group_or_population_identity_mismatch")
        if identities[(row.get("group_id"), population, row.get("request_id"))] != 1:
            errors.append("duplicate_request_pair")
        comparisons.append(comparison(row, extra_errors=errors))
    pooled = []
    scopes = [("overall", "all", comparisons)]
    scopes += [("domain", domain, [row for row in comparisons if row["domain"] == domain])
               for domain in sorted({group["domain"] for group in groups})]
    scopes += [("group", group["group_id"], [row for row in comparisons if row["group_id"] == group["group_id"]])
               for group in groups]
    for scope, scope_id, rows in scopes:
        for population in POPULATIONS:
            pooled.append(dict(scope=scope, scope_id=scope_id, population=population,
                               **pool(row for row in rows if row["population"] == population)))
    interpretations = []
    for scope, scope_id, _ in scopes:
        populations = {row["population"]: row for row in pooled if row["scope"] == scope and row["scope_id"] == scope_id}
        interpretations.append(dict(scope=scope, scope_id=scope_id,
                                    patterns=_patterns(populations.get("source"), populations.get("heldout"))))
    costs = []
    for group in groups:
        costs.append(dict(group_id=group["group_id"], domain=group["domain"], status=group.get("status"),
                          actual_group_wall_s=group.get("actual_wall_s"), components=group.get("costs", {}),
                          startup=group.get("startup"), cleanup=group.get("cleanup"),
                          accounting="Diagnostic costs only; nested timers are not summed. No historical/fresh latency comparison."))
    complete = (result.get("status") == "complete" and len(groups) == 6 and
                all(group.get("status") == "complete" for group in groups) and
                all(len(group.get(population, [])) == 4 for group in groups for population in POPULATIONS) and
                all(row["valid"] for row in comparisons))
    return dict(status="complete" if complete else "partial_or_failed" if groups else "not_executed",
                execution_status=result.get("status"), groups=[dict(group_id=g["group_id"], domain=g["domain"],
                status=g.get("status"), error=g.get("error")) for g in groups],
                planned=result.get("planned", {"new_source_generations": 24, "heldout_response_loss_evaluations": 48, "optimizer_steps": 0}),
                actual_source_generations=result.get("actual_source_generations"),
                actual_response_kl_evaluations=result.get("actual_response_kl_evaluations"),
                optimizer_steps=result.get("optimizer_steps"), explicit_warmups=result.get("explicit_warmups"),
                unexecuted_groups=result.get("unexecuted_groups", []),
                actual_wall_s=result.get("actual_wall_s"), comparisons=comparisons, pooled=pooled,
                interpretations=interpretations, costs=costs, interpretation_boundary=BOUNDARY,
                provenance=result.get("bindings", result.get("provenance", {
                    "evidence_bindings": result.get("evidence_bindings"), "source_manifest": result.get("source_manifest")})),
                evidence_location="Full paired raw records, exact tokens, trajectories and validation errors remain in RESULT.json and event artifacts.")


def _csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: json.dumps(_clean(value), sort_keys=True) if isinstance(value, (dict, list, tuple)) else _clean(value)
                         for key, value in row.items()} for row in rows)


def _fmt(value, digits=4):
    return "—" if value is None else str(value) if isinstance(value, int) else f"{value:.{digits}f}"


def write_reports(result, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    report = summarize(result)
    (out / "SUMMARY.json").write_text(json.dumps(_clean(report), indent=2, sort_keys=True, allow_nan=False) + "\n")
    for filename, rows in (("PAIRED_REQUESTS.csv", report["comparisons"]), ("POOLED.csv", report["pooled"]), ("COSTS.csv", report["costs"])):
        _csv(out / filename, rows)
    lines = ["# DFlash KL-to-native diagnostic", "", f"Status: **{report['status']}** (execution: {report['execution_status']}).", "", BOUNDARY, "",
             "The fixed scope is six existing groups, 24 new natural S8 source generations, 48 held-out response-level KL evaluations, and zero optimizer steps.", "",
             "| Population | KL evidence | Native evidence |", "|---|---|---|",
             "| Source | Retained F/S8 fixed-anchor losses | Retained F; fresh natural S8 generation |",
             "| Held-out | Fresh paired F/S8 losses on the same saved answer and anchors | Retained F/S8 generations |", "",
             "Invalid pairs are retained but excluded from gain calculations. Zero-supervision pairs remain in valid native comparisons and are excluded from KL-improvement claims. Each normalized response loss receives equal weight. Call reductions use summed paired counts; positive values mean fewer calls.", ""]
    if not report["comparisons"]:
        lines += ["No diagnostic measurements have been executed.", ""]
    for population in POPULATIONS:
        lines += [f"## {population.capitalize()} requests", "", "| Scope | Valid / recorded | KL pairs | F KL | S8 KL | F calls | S8 calls | Call reduction | Better / worse / tied |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for row in report["pooled"]:
            if row["population"] != population:
                continue
            lines.append(f"| {row['scope']}:{row['scope_id']} | {row['valid_requests']}/{row['requests']} | {row['kl_pairs']} | {_fmt(row['F_mean_kl'])} | {_fmt(row['S8_mean_kl'])} | {_fmt(row['F_verification_calls'])} | {_fmt(row['S8_verification_calls'])} | {_fmt(row['verification_call_reduction_percent'], 2)}% | {row['native_better_requests']}/{row['native_worse_requests']}/{row['native_tied_requests']} |")
        lines.append("")
    lines += ["## Observations and remaining uncertainty", "", "These pattern labels guide further diagnosis and do not identify causes. Source and held-out populations are never pooled together.", ""]
    for row in report["interpretations"]:
        lines.append(f"- **{row['scope']}:{row['scope_id']}**: " + " ".join(row["patterns"]))
    invalid = [row for row in report["comparisons"] if not row["valid"]]
    lines += ["", f"Invalid paired comparisons: {len(invalid)}. Zero-supervision pairs: {sum(r['zero_supervision_pair'] for r in report['comparisons'])}.", ""]
    for group in report["groups"]:
        if group["status"] != "complete" or group.get("error"):
            lines.append(f"- Incomplete group `{group['group_id']}`: {group['status']}; error: {group.get('error')}")
    lines += ["", "## Costs and evidence", "", f"Actual diagnostic wall time: {_fmt(report['actual_wall_s'], 2)} seconds.",
              f"Recorded new source generations: {_fmt(report['actual_source_generations'])}; held-out response KL evaluations: {_fmt(report['actual_response_kl_evaluations'])}; optimizer steps: {_fmt(report['optimizer_steps'])}; explicit warmups: {_fmt(report['explicit_warmups'])}.",
              "Teacher preparation, loss forwards/projection chunks, native source replay, checkpoint loading, cache work, startup and cleanup costs remain separate in COSTS.csv and raw receipts. Nested timers must not be added blindly. No latency-reduction or serving-speedup estimate is calculated.", "",
              "PAIRED_REQUESTS.csv retains every paired outcome, raw counters/losses, exposure and validation errors. POOLED.csv contains group, domain and population-specific overall aggregates. Full exact token streams, native trajectories, anchors, masks, weights and identity bindings remain in the private run artifacts. No optimizer state is created or loaded, and no new serving checkpoint is produced.", ""]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return report
