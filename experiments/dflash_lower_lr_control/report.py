"""Three-arm lower-LR control reports; fresh and historical times are not paired."""
from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path

ARMS = ("F", "S8", "L8")
POPULATIONS = ("source", "heldout")
CONTRASTS = (("L8", "F"), ("L8", "S8"), ("S8", "F"))
PLANNED = {"optimizer_steps": 48, "training_presentations": 96,
           "native_generations": 48, "response_kl_evaluations": 48, "explicit_warmups": 0}
BOUNDARY = (
    "Better native drafting means fewer verification calls for identical output tokens in this diagnostic; "
    "it does not establish a serving-time improvement. Historical F/S8 and fresh L8 timings are not a "
    "latency comparison. Beating S8 while remaining worse than F means reduced harm, not successful "
    "transfer. Interpretation patterns guide further investigation rather than identifying the underlying "
    "cause or establishing that forward KL is generally unsuitable. This reused diagnostic set and one "
    "training seed do not establish broad generalization; any later selected method needs fresh held-out confirmation."
)


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    return value


def _ratio(numerator, denominator, scale=1):
    return scale * numerator / denominator if numerator is not None and denominator else None


def _reduction(candidate, control):
    return 100 * (1 - candidate / control) if candidate is not None and control else None


def _validate(pair):
    from .experiment import validate_pair
    errors = []
    recorded = pair.get("validation") or {}
    if recorded.get("status") != "passed" or recorded.get("errors") != []:
        errors.append("recorded_pair_validation_failed_or_missing")
        errors.extend("recorded:" + str(item) for item in recorded.get("errors", []))
    try:
        checked = validate_pair(pair)
        if not isinstance(checked, dict) or checked.get("status") != "passed" or checked.get("errors") != []:
            errors.append("recomputed_pair_validation_failed")
            if isinstance(checked, dict):
                errors.extend("recomputed:" + str(item) for item in checked.get("errors", []))
    except (ValueError, TypeError, KeyError, OverflowError, AssertionError) as error:
        errors.append("recomputed:" + str(error))
    return sorted(set(errors))


def _contrast(result, candidate, control, *, valid, kl_valid, mean=False):
    prefix = f"{candidate}_vs_{control}_"
    new, old = result.get(f"{candidate}_verification_calls"), result.get(f"{control}_verification_calls")
    saved = old - new if valid else None
    result[prefix + "verification_calls_saved"] = saved
    result[prefix + "verification_call_reduction_percent"] = _reduction(new, old) if valid else None
    result[prefix + "native_outcome"] = "invalid" if saved is None else "better" if saved > 0 else "worse" if saved < 0 else "tied"
    field = "mean_kl" if mean else "loss"
    new, old = result.get(f"{candidate}_{field}"), result.get(f"{control}_{field}")
    change = new - old if kl_valid else None
    result[prefix + "kl_change"] = change
    result[prefix + "kl_reduction_percent"] = _reduction(new, old) if kl_valid else None
    result[prefix + "kl_outcome"] = "excluded" if change is None else "better" if change < 0 else "worse" if change > 0 else "tied"


def comparison(pair, *, extra_errors=()):
    errors = sorted(set(_validate(pair) + list(extra_errors)))
    result = {key: pair.get(key) for key in ("request_id", "group_id", "domain", "population")}
    result.update(valid=not errors, invalid_reasons=errors)
    for arm in ARMS:
        native, loss = (pair.get(arm) or {}).get("native") or {}, (pair.get(arm) or {}).get("kl") or {}
        for key in ("verification_calls", "accepted_draft_tokens", "verification_committed_tokens",
                    "initial_output_tokens", "output_tokens", "cap_hit", "eos_hit", "finish_reason",
                    "input_ids_sha256", "checkpoint_identity_sha256", "checkpoint_id"):
            result[f"{arm}_{key}"] = native.get(key)
        result[f"{arm}_context_tokens"] = len(native["input_ids"]) if isinstance(native.get("input_ids"), list) else None
        for key in ("loss", "eligible_positions", "zero_supervision", "normalization_denominator",
                    "unique_supervised_positions", "anchor_identity_sha256", "fixed_anchor_identity_sha256",
                    "anchor_positions", "loss_mask_sha256", "weights_sha256", "anchor_seed",
                    "sampled_blocks", "eligible_blocks", "unique_supervised_position_ids",
                    "weighted_position_sum", "sampling_and_weights_sha256", "diagnostic_model_mode",
                    "training_rng_preserved", "excluded_from_loss_improvement"):
            result[f"{arm}_{key}"] = loss.get(key)
        calls, tokens = native.get("verification_calls"), native.get("output_tokens")
        numeric = all(_finite(native.get(key)) for key in ("verification_calls", "output_tokens", "accepted_draft_tokens", "verification_committed_tokens"))
        for key, numerator, denominator, scale in (
            ("calls_per_1000_output_tokens", calls, tokens, 1000),
            ("accepted_draft_tokens_per_call", native.get("accepted_draft_tokens"), calls, 1),
            ("committed_progress_per_call", native.get("verification_committed_tokens"), calls, 1),
            ("emitted_tokens_per_call", tokens, calls, 1),
        ):
            result[f"{arm}_{key}"] = _ratio(numerator, denominator, scale) if numeric else None
        result[f"{arm}_native_origin"] = result[f"{arm}_kl_origin"] = "fresh" if arm == "L8" else "retained"
    result["zero_supervision_pair"] = any(result[f"{arm}_zero_supervision"] is True for arm in ARMS)
    result["kl_valid"] = (result["valid"] and not result["zero_supervision_pair"] and
                          all(_finite(result[f"{arm}_loss"]) for arm in ARMS))
    for candidate, control in CONTRASTS:
        _contrast(result, candidate, control, valid=result["valid"], kl_valid=result["kl_valid"])
    return result


def pool(rows):
    rows = list(rows)
    valid, losses = [row for row in rows if row["valid"]], [row for row in rows if row["kl_valid"]]
    result = dict(requests=len(rows), valid_requests=len(valid), invalid_requests=len(rows) - len(valid),
                  kl_pairs=len(losses), zero_supervision_pairs=sum(row["zero_supervision_pair"] for row in rows))
    for arm in ARMS:
        for key in ("verification_calls", "accepted_draft_tokens", "verification_committed_tokens", "output_tokens"):
            result[f"{arm}_{key}"] = sum(row[f"{arm}_{key}"] for row in valid) if valid else None
        # Responses, not tokens/anchors, receive equal weight after qualified normalization.
        result[f"{arm}_mean_kl"] = sum(row[f"{arm}_loss"] for row in losses) / len(losses) if losses else None
        calls, tokens = result[f"{arm}_verification_calls"], result[f"{arm}_output_tokens"]
        result[f"{arm}_calls_per_1000_output_tokens"] = _ratio(calls, tokens, 1000)
        result[f"{arm}_accepted_draft_tokens_per_call"] = _ratio(result[f"{arm}_accepted_draft_tokens"], calls)
        result[f"{arm}_committed_progress_per_call"] = _ratio(result[f"{arm}_verification_committed_tokens"], calls)
        result[f"{arm}_emitted_tokens_per_call"] = _ratio(tokens, calls)
    for candidate, control in CONTRASTS:
        _contrast(result, candidate, control, valid=bool(valid), kl_valid=bool(losses), mean=True)
        prefix = f"{candidate}_vs_{control}_"
        for metric in ("native", "kl"):
            for outcome in ("better", "worse", "tied"):
                result[prefix + metric + "_" + outcome + "_requests"] = sum(row[prefix + metric + "_outcome"] == outcome for row in rows)
    return result


def _patterns(source, heldout):
    if not source or not heldout:
        return ["Missing paired population evidence; no transfer interpretation."]
    source_kl = source.get("L8_vs_F_kl_change")
    source_calls = source.get("L8_vs_F_verification_calls_saved")
    heldout_f = heldout.get("L8_vs_F_verification_calls_saved")
    heldout_s = heldout.get("L8_vs_S8_verification_calls_saved")
    findings = []
    if source_kl is None or heldout_f is None:
        return ["Insufficient valid source-loss or held-out native evidence; no transfer interpretation."]
    if source_kl < 0 and heldout_f > 0:
        findings.append("Lower source KL and fewer held-out verification calls than F are observed at the smaller LR, supporting transfer on these requests. Judge the reported effect sizes; no success threshold is imposed.")
    elif heldout_s is not None and heldout_s > 0 and heldout_f < 0:
        findings.append("L8 improves held-out calls over S8 but remains worse than F: reduced harm, not successful transfer.")
    elif heldout_s is not None and heldout_s > 0 and heldout_f == 0:
        findings.append("L8 improves held-out calls over S8 and ties F: recovery to frozen performance, not an additional drafting benefit.")
    if source_kl < 0 and heldout_f <= 0:
        findings.append("Source KL decreases without a held-out call benefit over F; transfer remains unresolved. Feedback coverage is a possible next question, not an identified cause.")
    if source_kl >= 0:
        findings.append("No mean source-KL decrease is observed for L8; useful source learning is not established by this metric, so this control cannot establish its transfer.")
    if source_calls is not None and source_calls <= 0 and source_kl < 0:
        findings.append("Source KL decreases without fewer source calls; source fitting and native drafting remain separate measurements.")
    findings.append("Negligible source learning would make the control inconclusive. No numerical negligibility cutoff was prespecified; inspect the effect sizes rather than assigning one retrospectively.")
    return findings


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
    scopes = [("overall", "all", comparisons)]
    scopes += [("domain", domain, [row for row in comparisons if row["domain"] == domain]) for domain in sorted({g["domain"] for g in groups})]
    scopes += [("group", group["group_id"], [row for row in comparisons if row["group_id"] == group["group_id"]]) for group in groups]
    pooled = [dict(scope=scope, scope_id=scope_id, population=population,
                   **pool(row for row in rows if row["population"] == population))
              for scope, scope_id, rows in scopes for population in POPULATIONS]
    interpretations = []
    for scope, scope_id, _ in scopes:
        populations = {row["population"]: row for row in pooled if row["scope"] == scope and row["scope_id"] == scope_id}
        interpretations.append(dict(scope=scope, scope_id=scope_id, patterns=_patterns(populations.get("source"), populations.get("heldout"))))
    costs, training = [], []
    for group in groups:
        receipt = group.get("training") or {}
        costs.append(dict(group_id=group["group_id"], domain=group["domain"], status=group.get("status"),
                          actual_group_wall_s=group.get("actual_wall_s"), components=group.get("costs", {}),
                          training_components=receipt.get("costs", {}), startup=group.get("startup"), cleanup=group.get("cleanup"),
                          accounting="Teacher preparation may be nested in training, native activation in loading, and memory peaks are maxima rather than additive costs. Preserve receipts; do not sum overlapping timers. No historical/fresh latency comparison."))
        training.append(dict(group_id=group["group_id"], domain=group["domain"], status=group.get("status"),
                             checkpoint=group.get("checkpoint"), training=receipt))
    counts = {key: result.get(key) for key in ("optimizer_steps", "training_presentations", "actual_native_generations", "actual_response_kl_evaluations", "explicit_warmups")}
    expected_counts = dict(optimizer_steps=48, training_presentations=96, actual_native_generations=48,
                           actual_response_kl_evaluations=48, explicit_warmups=0)
    accounting_errors = ([key + "_does_not_match_completed_scope" for key, expected in expected_counts.items()
                          if counts[key] != expected] if result.get("status") == "complete" else [])
    complete = (not accounting_errors and result.get("status") == "complete" and len(groups) == 6 and all(g.get("status") == "complete" for g in groups)
                and all(len(g.get(population, [])) == 4 for g in groups for population in POPULATIONS)
                and all(row["valid"] for row in comparisons))
    return dict(status="complete" if complete else "partial_or_failed" if groups else "not_executed",
                execution_status=result.get("status"), groups=[dict(group_id=g["group_id"], domain=g["domain"], status=g.get("status"), error=g.get("error")) for g in groups],
                planned=result.get("planned", PLANNED), accounting_errors=accounting_errors, **counts, actual_wall_s=result.get("actual_wall_s"),
                unexecuted_groups=result.get("unexecuted_groups", []), comparisons=comparisons, pooled=pooled,
                interpretations=interpretations, costs=costs, training=training, interpretation_boundary=BOUNDARY,
                provenance=result.get("bindings", result.get("provenance", {"evidence_bindings": result.get("evidence_bindings"), "source_manifest": result.get("source_manifest")})),
                evidence_location="Full paired raw records, tokens, trajectories, training exposure and validation errors remain in private RESULT.json and event artifacts.")


def _csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: json.dumps(_clean(value), sort_keys=True) if isinstance(value, (dict, list, tuple)) else _clean(value) for key, value in row.items()} for row in rows)


def _fmt(value, digits=4):
    return "—" if value is None else str(value) if isinstance(value, int) else f"{value:.{digits}f}"


def write_reports(result, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    report = summarize(result)
    (out / "SUMMARY.json").write_text(json.dumps(_clean(report), indent=2, sort_keys=True, allow_nan=False) + "\n")
    for filename, rows in (("PAIRED_REQUESTS.csv", report["comparisons"]), ("POOLED.csv", report["pooled"]), ("COSTS.csv", report["costs"]), ("TRAINING.csv", report["training"])):
        _csv(out / filename, rows)
    lines = ["# DFlash lower-learning-rate control", "", f"Status: **{report['status']}** (execution: {report['execution_status']}).", "", BOUNDARY, "",
             "F is released DFlash. S8 and L8 both originally start from the same released weights and fresh AdamW state with seed 17. S8 is the retained eight-step 3e-4 arm; only L8 is newly trained, for eight steps at 3e-5.", "",
             "Each independent group uses its same four frozen-generated source responses in four passes (16 presentations, eight steps). Only each response receives loss; its prompt supplies context. Held-out responses never enter training.", "",
             "The fixed new-work budget is 48 optimizer steps, 96 presentations, 48 natural native generations, 48 response-level KL evaluations, and zero explicit warmups. F/S8 evidence is retained, not rerun.", "",
             "Training anchors retain their original per-presentation sampling trajectory. Source diagnostic anchors are fixed by PR9; held-out diagnostic anchors are fixed by PR10. These three contracts remain distinct.", "",
             "Invalid triplets remain visible but are excluded from gain calculations. Zero-supervision cases are excluded only from KL improvement claims. Mean KL equally weights normalized response losses. Call reductions use summed paired counts; positive values mean fewer calls.", ""]
    if not report["comparisons"]:
        lines += ["No control measurements have been executed.", ""]
    for population in POPULATIONS:
        lines += [f"## {population.capitalize()} requests", "", "| Scope | Valid / recorded | KL pairs | F KL | S8 KL | L8 KL | F calls | S8 calls | L8 calls | L8 vs F | L8 vs S8 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for row in report["pooled"]:
            if row["population"] != population:
                continue
            values = [_fmt(row[f"{arm}_mean_kl"]) for arm in ARMS] + [_fmt(row[f"{arm}_verification_calls"]) for arm in ARMS]
            values += [_fmt(row[f"L8_vs_{control}_verification_call_reduction_percent"], 2) + "%" for control in ("F", "S8")]
            lines.append(f"| {row['scope']}:{row['scope_id']} | {row['valid_requests']}/{row['requests']} | {row['kl_pairs']} | " + " | ".join(values) + " |")
        lines += ["", "| Scope | L8 vs F: better / worse / tied | L8 vs S8: better / worse / tied |", "|---|---:|---:|"]
        for row in report["pooled"]:
            if row["population"] == population:
                counts = ["/".join(str(row[f"L8_vs_{control}_native_{outcome}_requests"]) for outcome in ("better", "worse", "tied")) for control in ("F", "S8")]
                lines.append(f"| {row['scope']}:{row['scope_id']} | " + " | ".join(counts) + " |")
        lines.append("")
    lines += ["## Interpretation", "", "Source and held-out populations are never pooled together. These patterns describe this bounded control; they do not select a mechanism or establish a cause.", ""]
    for row in report["interpretations"]:
        lines.append(f"- **{row['scope']}:{row['scope_id']}**: " + " ".join(row["patterns"]))
    lines += ["", f"Invalid triplets: {sum(not row['valid'] for row in report['comparisons'])}. Zero-supervision triplets: {sum(row['zero_supervision_pair'] for row in report['comparisons'])}.", ""]
    if report["accounting_errors"]:
        lines.append("Completed-scope accounting errors: " + "; ".join(report["accounting_errors"]))
    for group in report["groups"]:
        if group["status"] != "complete" or group.get("error"):
            lines.append(f"- Incomplete group `{group['group_id']}`: {group['status']}; error: {group.get('error')}")
    lines += ["", "## Costs and retained evidence", "", f"Actual campaign wall time: {_fmt(report['actual_wall_s'], 2)} seconds.",
              f"Recorded new work: {_fmt(report['optimizer_steps'])} optimizer steps; {_fmt(report['training_presentations'])} presentations; {_fmt(report['actual_native_generations'])} native generations; {_fmt(report['actual_response_kl_evaluations'])} response-level KL evaluations; {_fmt(report['explicit_warmups'])} explicit warmups.", "",
              "COSTS.csv preserves teacher preparation, training, evaluation, export/loading, native activation/cache work and peak-memory receipts separately. Nested timers must not be added blindly, and peak memory is not additive. No latency-reduction or serving-speedup estimate is calculated.", "",
              "TRAINING.csv retains each training receipt and genuine group-specific L8 checkpoint binding, including step learning rates and original-exposure checks. PAIRED_REQUESTS.csv retains all three arm measurements and validation errors. POOLED.csv supplies group/domain/overall aggregates separated by population. Exact token IDs, native per-round offsets, masks, anchor/RNG metadata, zero-supervision records and raw receipts remain in private artifacts.", ""]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return report
