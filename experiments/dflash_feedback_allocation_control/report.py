"""Three-arm feedback-allocation reports; fresh and historical times are not paired."""
from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from .protocol import PLANNED

ARMS = ("F", "S8", "P8")
POPULATIONS = ("source", "heldout")
CONTRASTS = (("P8", "F"), ("P8", "S8"), ("S8", "F"))
BOUNDARY = (
    "Better native drafting means fewer verification calls for identical output tokens in this diagnostic; "
    "it does not establish a serving-time improvement. Historical F/S8 and fresh P8 timings are not a "
    "latency comparison. Beating S8 while remaining worse than F means reduced harm, not successful "
    "transfer. There are three P8 adaptation trajectories, not six independent replicates; original-group "
    "summaries share their dataset's P8 checkpoint. Allocation changes diversity, repetition, pairing/order "
    "and potentially supervised token work together: this is neither a pure diversity test nor an "
    "equal-total-compute comparison with six retained S8 models. Interpretation does not identify a cause. "
    "This reused diagnostic set and one "
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
    recorded = pair.get("validation")
    recorded = recorded if isinstance(recorded, dict) else {}
    if recorded.get("status") != "passed" or recorded.get("errors") != []:
        errors.append("recorded_pair_validation_failed_or_missing")
        errors.extend("recorded:" + str(item) for item in (recorded.get("errors") or []))
    try:
        checked = validate_pair(pair)
        if not isinstance(checked, dict) or checked.get("status") != "passed" or checked.get("errors") != []:
            errors.append("recomputed_pair_validation_failed")
            if isinstance(checked, dict):
                errors.extend("recomputed:" + str(item) for item in (checked.get("errors") or []))
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
    result = {key: pair.get(key) for key in ("request_id", "group_id", "domain", "population", "pool_id", "pool_request_id", "checkpoint_bindings")}
    result.update(valid=not errors, invalid_reasons=errors)
    for arm in ARMS:
        arm_value = pair.get(arm)
        arm_value = arm_value if isinstance(arm_value, dict) else {}
        native, loss = arm_value.get("native"), arm_value.get("kl")
        native = native if isinstance(native, dict) else {}
        loss = loss if isinstance(loss, dict) else {}
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
                    "training_rng_preserved", "excluded_from_loss_improvement", "block_keep_mask",
                    "label_positions", "teacher_positions", "binary_loss_mask", "position_weights",
                    "anchor_rng_after_sha256"):
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
        result[f"{arm}_native_origin"] = result[f"{arm}_kl_origin"] = "fresh" if arm == "P8" else "retained"
        # Preserve offsets/counts without duplicating prompt/answer token contents.
        rounds = native.get("rounds")
        result[f"{arm}_rounds"] = ([{key: value for key, value in event.items()
                                     if key != "committed_token_ids"} if isinstance(event, dict) else event
                                    for event in rounds] if isinstance(rounds, list) else rounds)
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
    source_kl = source.get("P8_vs_F_kl_change")
    source_calls = source.get("P8_vs_F_verification_calls_saved")
    heldout_f = heldout.get("P8_vs_F_verification_calls_saved")
    heldout_s = heldout.get("P8_vs_S8_verification_calls_saved")
    findings = []
    if source_kl is None or heldout_f is None:
        return ["Insufficient valid source-loss or held-out native evidence; no transfer interpretation."]
    if source_kl < 0 and heldout_f > 0:
        findings.append("Lower source KL and fewer held-out verification calls than F are observed under this allocation. This pattern supports useful transfer on these requests only if the source learning is non-negligible; assess its magnitude from the reported effects.")
    elif heldout_s is not None and heldout_s > 0 and heldout_f < 0:
        findings.append("P8 improves held-out calls over S8 but remains worse than F: reduced harm, not successful transfer.")
    elif heldout_s is not None and heldout_s > 0 and heldout_f == 0:
        findings.append("P8 improves held-out calls over S8 and ties F: recovery to frozen performance, not an additional drafting benefit.")
    if source_kl < 0 and heldout_f <= 0:
        findings.append("Source KL decreases without a held-out call benefit over F; transfer remains unresolved. If the source learning is non-negligible, this allocation was insufficient on these requests; it does not rule out feedback coverage as a limitation or establish it as the cause.")
    if source_kl >= 0:
        findings.append("No mean source-KL decrease is observed for P8; useful source learning is not established by this metric, so this control cannot establish its transfer.")
    if source_calls is not None and source_calls <= 0 and source_kl < 0:
        findings.append("Source KL decreases without fewer source calls; source fitting and native drafting remain separate measurements.")
    findings.append("Negligible source learning would make the control inconclusive. No numerical negligibility cutoff was prespecified; inspect the effect sizes rather than assigning one retrospectively.")
    return findings


COUNT_FIELDS = ("optimizer_steps", "training_presentations", "actual_native_generations",
                "actual_response_kl_evaluations", "explicit_warmups", "attempted_native_generations",
                "unknown_native_work_receipts", "unattempted_native_generations",
                "unattempted_response_kl_evaluations", "unknown_training_work_receipts",
                "unknown_response_kl_worker_allocations", "unattempted_optimizer_steps",
                "unknown_optimizer_steps", "unknown_response_kl_evaluations",
                "attempted_response_kl_evaluations")


def summarize(result):
    pools = result.get("pools", [])
    raw = [(owner, population, row) for owner in pools for population in POPULATIONS
           for row in owner.get(population, [])]
    # Original-group identities remain distinct even if local request IDs collide.
    identities = Counter((row.get("domain"), row.get("group_id"), population, row.get("request_id"))
                         for _, population, row in raw)
    owners = Counter(owner.get("pool_id") for owner in pools)
    group_owners = Counter((owner.get("domain"), group_id) for owner in pools
                          for group_id in owner.get("group_ids", []))
    comparisons = []
    for owner, population, row in raw:
        errors = []
        checkpoint = owner.get("checkpoint")
        checkpoint = checkpoint if isinstance(checkpoint, dict) else {}
        checkpoint_hash = checkpoint.get("checkpoint_identity_sha256")
        if (checkpoint.get("checkpoint_id") != "P8" or
                any(checkpoint.get(key) != owner.get(key) for key in ("pool_id", "domain", "group_ids")) or
                checkpoint.get("group_id") is not None or
                not isinstance(checkpoint_hash, str) or len(checkpoint_hash) != 64 or
                any(character not in "0123456789abcdef" for character in checkpoint_hash)):
            errors.append("missing_or_invalid_pool_P8_checkpoint_owner")
        if (row.get("checkpoint_bindings") or {}).get("P8") != checkpoint_hash:
            errors.append("pair_P8_checkpoint_differs_from_pool_checkpoint")
        if (row.get("pool_id") != owner.get("pool_id") or row.get("domain") != owner.get("domain") or
                row.get("group_id") not in owner.get("group_ids", []) or row.get("population") != population):
            errors.append("pair_pool_group_or_population_identity_mismatch")
        if identities[(row.get("domain"), row.get("group_id"), population, row.get("request_id"))] != 1:
            errors.append("duplicate_request_pair")
        if owners[owner.get("pool_id")] != 1 or group_owners[(row.get("domain"), row.get("group_id"))] != 1:
            errors.append("duplicate_pool_or_original_group_owner")
        comparisons.append(comparison(row, extra_errors=errors))
    scopes = [("overall", "all", None, None, comparisons)]
    domains = sorted({owner["domain"] for owner in pools})
    scopes += [("domain", domain, domain, None, [row for row in comparisons if row["domain"] == domain])
               for domain in domains]
    for owner in pools:
        rows = [row for row in comparisons if row["pool_id"] == owner["pool_id"]]
        scopes.append(("pool", owner["pool_id"], owner["domain"], owner["pool_id"], rows))
        for group_id in owner.get("group_ids", []):
            scopes.append(("group", group_id, owner["domain"], owner["pool_id"],
                           [row for row in rows if row["group_id"] == group_id]))
    pooled, interpretations = [], []
    for scope, scope_id, domain, pool_id, rows in scopes:
        populations = {}
        for population in POPULATIONS:
            values = [row for row in rows if row["population"] == population]
            summary = dict(scope=scope, scope_id=scope_id, domain=domain, pool_id=pool_id,
                           population=population,
                           recorded_adaptation_trajectories=len({row["pool_id"] for row in values}),
                           **pool(values))
            pooled.append(summary)
            populations[population] = summary
        interpretations.append(dict(scope=scope, scope_id=scope_id, domain=domain, pool_id=pool_id,
                                    patterns=_patterns(populations["source"], populations["heldout"])))
    costs, training, pool_statuses = [], [], []
    for owner in pools:
        receipt = owner.get("training") or {}
        identity = {key: owner.get(key) for key in ("pool_id", "domain", "group_ids", "status")}
        costs.append(dict(**identity, actual_pool_wall_s=owner.get("actual_wall_s"),
                          components=owner.get("costs", {}), training_components=receipt.get("costs", {}),
                          startup=owner.get("startup"), cleanup=owner.get("cleanup"),
                          accounting="Teacher/training/evaluation/export/loading receipts can overlap; memory peaks are not additive. Do not sum nested timers. No historical/fresh latency comparison."))
        training.append(dict(**identity, checkpoint=owner.get("checkpoint"), training=receipt,
                             training_validation=owner.get("training_validation"),
                             trainer_stop=owner.get("trainer_stop")))
        pool_statuses.append(dict(**identity, error=owner.get("error"), cleanup_error=owner.get("cleanup_error"),
                                  work_incomplete=owner.get("work_incomplete"),
                                  recorded_native_receipts=len(owner.get("native_raw", [])),
                                  recorded_source_pairs=len(owner.get("source", [])),
                                  recorded_heldout_pairs=len(owner.get("heldout", []))))
    counts = {key: result.get(key) for key in COUNT_FIELDS}
    expected = dict(optimizer_steps=24, training_presentations=48, actual_native_generations=48,
                    actual_response_kl_evaluations=48, explicit_warmups=0)
    accounting_errors = []
    if result.get("status") == "complete":
        accounting_errors += [key + "_does_not_match_completed_scope" for key, count in expected.items()
                              if counts[key] != count]
        if counts["attempted_native_generations"] not in (None, 48):
            accounting_errors.append("attempted_native_generations_does_not_match_completed_scope")
        if counts["attempted_response_kl_evaluations"] not in (None, 48):
            accounting_errors.append("attempted_response_kl_evaluations_does_not_match_completed_scope")
        for key in ("unknown_native_work_receipts", "unattempted_native_generations",
                    "unattempted_response_kl_evaluations", "unknown_training_work_receipts",
                    "unknown_response_kl_worker_allocations", "unattempted_optimizer_steps",
                    "unknown_optimizer_steps", "unknown_response_kl_evaluations"):
            if counts[key] not in (None, 0, []):
                accounting_errors.append(key + "_present_in_completed_scope")
        if result.get("unexecuted_pools"):
            accounting_errors.append("unexecuted_pools_present_in_completed_scope")
    complete = (result.get("status") == "complete" and not accounting_errors and len(pools) == 3 and
                len(domains) == 3 and len(owners) == 3 and len(group_owners) == 6 and
                all(owner.get("status") == "complete" and len(owner.get("group_ids", [])) == 2 and
                    len(set(owner.get("group_ids", []))) == 2 for owner in pools) and
                all(len(owner.get(population, [])) == 8 for owner in pools for population in POPULATIONS) and
                all(sum(row.get("group_id") == group_id for row in owner.get(population, [])) == 4
                    for owner in pools for group_id in owner.get("group_ids", []) for population in POPULATIONS) and
                all(row["valid"] for row in comparisons))
    # Keep executor-provided work receipts verbatim; missing work is not inferred
    # from paired rows, because failure may happen before a pair is assembled.
    accounting_receipts = {key: value for key, value in result.items()
                           if key in COUNT_FIELDS or any(part in key for part in
                               ("accounting", "unattempted", "unknown_", "invalid_", "attempted_"))}
    return dict(status="complete" if complete else "partial_or_failed" if pools else "not_executed",
                execution_status=result.get("status"), pools=pool_statuses,
                planned=result.get("planned", PLANNED), **counts, accounting_errors=accounting_errors,
                accounting_receipts=accounting_receipts, actual_wall_s=result.get("actual_wall_s"),
                first_pool_qualification=result.get("first_pool_qualification"),
                unexecuted_pools=result.get("unexecuted_pools", []),
                planned_adaptation_trajectories=3, recorded_adaptation_trajectories=len(owners),
                completed_adaptation_trajectories=sum(owner.get("status") == "complete" for owner in pools),
                original_groups=len(group_owners), invalid_pairs=sum(not row["valid"] for row in comparisons),
                comparisons=comparisons, pooled=pooled, interpretations=interpretations,
                costs=costs, training=training, interpretation_boundary=BOUNDARY,
                primary_outcome="heldout_P8_vs_F_verification_call_reduction_percent_from_summed_paired_counts",
                provenance=result.get("bindings", result.get("provenance", {
                    "evidence_bindings": result.get("evidence_bindings"),
                    "source_manifest": result.get("source_manifest")})),
                evidence_location="Full paired raw records, native_raw (including rejected/unpaired receipts), tokens, training exposure and validation failures remain in private RESULT.json and event artifacts.")


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
    for filename, rows in (("PAIRED_REQUESTS.csv", report["comparisons"]), ("POOLED.csv", report["pooled"]),
                           ("COSTS.csv", report["costs"]), ("TRAINING.csv", report["training"])):
        _csv(out / filename, rows)
    lines = ["# DFlash feedback-allocation control", "",
             f"Status: **{report['status']}** (execution: {report['execution_status']}).", "",
             "Primary outcome: held-out verification-call change for P8 versus retained F, computed from summed valid paired counts. P8 versus each request's original-group S8 is the additional comparison.", "",
             BOUNDARY, "",
             "F is released UltraChat DFlash. Three P8 models independently start from released weights and fresh AdamW state with seed 17; each uses eight steps at 3e-4. Retained S8 used four source responses in four passes (16 presentations, eight steps) per original group.", "",
             "Each P8 dataset pool combines its two original groups' eight frozen-generated source responses, ordered A0/B0/A1/B1/A2/B2/A3/B3 and repeated twice without shuffling. Each accumulation pair averages two separately normalized response losses before clipping and stepping. Only the response receives loss; the prompt supplies context. Held-out responses never enter training.", "",
             "The fixed new-work budget is three trajectories, 24 optimizer steps, 48 presentations, 48 natural native generations, 48 response-level KL evaluations, and zero explicit warmups. The first complete pool's correctness qualification is included in this budget. F/S8/L8 evidence is retained, not rerun; there are no extra evaluations, intermediate diagnostics, block probes, retries or retuning. Null and harmful results do not stop a valid run.", "",
             "P8 uses a fresh normal training RNG trajectory for the new order. Source diagnostic anchors remain those of PR9; held-out anchors remain those of PR10, with the original masks, weights, positions and denominators in eval/no_grad. Diagnostic anchors do not replace training anchors.", "",
             "Native evaluation is natural generation from each original prompt, concurrency 1, greedy, block size 10, cap 1,024 or EOS, capture disabled, with pinned runtime/tokenizer/context/cache settings. Exact output-token and termination equality to retained F is required.", "",
             "Invalid triplets remain visible but are excluded from gain calculations. Zero-supervision cases are excluded only from KL improvement claims. Mean KL equally weights normalized response losses. Call reductions use summed paired counts; positive values mean fewer calls. Source and held-out populations remain separate.", ""]
    if not report["comparisons"]:
        lines += ["No paired control measurements are recorded.", ""]
    for population in ("heldout", "source"):
        lines += [f"## {population.capitalize()} requests", "",
                  "| Scope | Valid / recorded | KL pairs | F KL | S8 KL | P8 KL | F calls | S8 calls | P8 calls | P8 vs F | P8 vs S8 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        selected = [row for row in report["pooled"] if row["population"] == population]
        for row in selected:
            values = [_fmt(row[f"{arm}_mean_kl"]) for arm in ARMS]
            values += [_fmt(row[f"{arm}_verification_calls"]) for arm in ARMS]
            values += ["—" if row[f"P8_vs_{control}_verification_call_reduction_percent"] is None else
                       _fmt(row[f"P8_vs_{control}_verification_call_reduction_percent"], 2) + "%"
                       for control in ("F", "S8")]
            lines.append(f"| {row['scope']}:{row['scope_id']} | {row['valid_requests']}/{row['requests']} | {row['kl_pairs']} | " + " | ".join(values) + " |")
        lines += ["", "| Scope | P8 vs F: better / worse / tied | P8 vs S8: better / worse / tied |",
                  "|---|---:|---:|"]
        for row in selected:
            counts = ["/".join(str(row[f"P8_vs_{control}_native_{outcome}_requests"])
                              for outcome in ("better", "worse", "tied")) for control in ("F", "S8")]
            lines.append(f"| {row['scope']}:{row['scope_id']} | " + " | ".join(counts) + " |")
        lines.append("")
    lines += ["## Interpretation", "",
              "Original-group rows are descriptive subdivisions of their shared pool trajectory, not independent adaptation replicates. These patterns do not select a mechanism or establish a cause.", ""]
    for row in report["interpretations"]:
        lines.append(f"- **{row['scope']}:{row['scope_id']}**: " + " ".join(row["patterns"]))
    lines += ["", f"Invalid triplets: {report['invalid_pairs']}. Zero-supervision triplets: {sum(row['zero_supervision_pair'] for row in report['comparisons'])}.", ""]
    if report["accounting_errors"]:
        lines.append("Completed-scope accounting errors: " + "; ".join(report["accounting_errors"]))
    for owner in report["pools"]:
        if owner["status"] != "complete" or owner.get("error"):
            lines.append(f"- Incomplete pool `{owner['pool_id']}`: {owner['status']}; error: {owner.get('error')}")
    lines += ["", "## Costs and retained evidence", "",
              f"Actual campaign wall time: {_fmt(report['actual_wall_s'], 2)} seconds.",
              f"Recorded new work: {_fmt(report['optimizer_steps'])} optimizer steps; {_fmt(report['training_presentations'])} presentations; {_fmt(report['actual_native_generations'])} native generations; {_fmt(report['actual_response_kl_evaluations'])} response-level KL evaluations; {_fmt(report['explicit_warmups'])} explicit warmups.", "",
              "Unknown and unattempted work receipts: `" + json.dumps(_clean({key: report[key] for key in COUNT_FIELDS if key.startswith(('unknown_', 'unattempted_'))}), sort_keys=True) + "`. Unexecuted pools: `" + json.dumps(report["unexecuted_pools"]) + "`.", "",
              "COSTS.csv preserves teacher preparation, training, evaluation, export/loading, native activation/cache work and peak-memory receipts separately. Nested timers must not be added blindly, and peak memory is not additive. No latency-reduction or serving-speedup estimate is calculated.", "",
              "TRAINING.csv retains complete training/export receipts, sampled anchors, exposure and effective learning rates, and each final P8 checkpoint's dataset-pool ownership. PAIRED_REQUESTS.csv retains original request/group IDs, pool identities, all arm measurements, per-round offsets and validation errors. POOLED.csv provides pool/dataset/original-group/overall summaries separated by population. Full native_raw, rejected and unpaired records, token IDs and diagnostic evidence remain in private RESULT.json and event artifacts.", ""]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return report
