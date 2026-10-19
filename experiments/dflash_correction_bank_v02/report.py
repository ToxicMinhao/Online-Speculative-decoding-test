"""Paired descriptive results; no model loading, replay or feedback generation."""
from __future__ import annotations

from collections import defaultdict
import csv
import json
import math
from pathlib import Path

from experiments.dflash_per_turn.report import totals

COMPARISONS = (("BANK", "F"), ("OSD", "F"), ("BANK", "OSD"))
SCOPE = ("Fresh capture-disabled checkpoint comparisons on the BANK arm's saved inputs. "
         "Complete request time includes prefill, BANK descriptor/routing/fusion when applicable, "
         "drafting and target verification. Adaptation and external checkpoint/cache switching "
         "are reported separately. Ratios pool sums, not percentages.")


def _number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _sum(rows, key):
    rows = list(rows)
    return sum(row[key] for row in rows) if all(_number(row.get(key)) for row in rows) else None


def _reduction(candidate, control):
    return 100*(1-candidate/control) if candidate is not None and control else None


def _validity(row, canonical):
    problems = []
    if canonical is None:
        return ["missing_bank_canonical"]
    for key in ("same_generation_valid", "input_match", "canonical_output_match", "checkpoint_identity_match"):
        if row.get(key) is not True:
            problems.append(key)
    for key in ("input_ids", "output_ids"):
        if row.get(key) != canonical.get(key):
            problems.append("exact_" + key + "_mismatch")
    if row.get("capture") is not False or row.get("feedback") is not None or row.get("capture_costs") is not None:
        problems.append("capture_not_disabled")
    if not _number(row.get("request_s")) or row.get("request_s") == 0:
        problems.append("invalid_request_time")
    if row.get("arm") == "BANK":
        read = row.get("read", {})
        for key in ("forecast_saved_before_response", "descriptor_extracted", "routing_recomputed",
                    "fusion_recomputed", "cost_in_request_s"):
            if read.get(key) is not True:
                problems.append("bank_" + key)
    return problems


def paired_turn(result, turn, candidate, control):
    canonical = next((r for r in result["canonical"] if r["arm"] == "BANK" and r["turn"] == turn), None)
    groups = [[r for r in result["scores"] if r["turn"] == turn and r["arm"] == arm]
              for arm in (candidate, control)]
    problems = []
    for arm, rows in zip((candidate, control), groups):
        if len(rows) != 3 or {r.get("repeat") for r in rows} != {0, 1, 2}:
            problems.append(arm + ":missing_or_duplicate_repeat")
        identities = {r.get("state_sha256") for r in rows}
        if len(identities) != 1 or None in identities:
            problems.append(arm + ":state_identity_changed")
        for row in rows:
            problems.extend(arm+":"+p for p in _validity(row, canonical))
    new, old = (totals(group) for group in groups)
    valid = not problems
    return dict(episode_id=result["episode_id"], source=result["source"], synthetic=result["synthetic"],
        turn=turn, candidate=candidate, control=control, paired_input_trajectory="BANK", valid=valid,
        invalid_reasons=sorted(set(problems)), candidate_metrics=new, control_metrics=old,
        request_time_reduction_percent=_reduction(new["complete_request_s"], old["complete_request_s"]) if valid else None,
        verification_call_reduction_percent=_reduction(new["verification_calls"], old["verification_calls"]) if valid else None,
        calls_per_1000_reduction_percent=_reduction(new["calls_per_1000_output_tokens"], old["calls_per_1000_output_tokens"]) if valid else None), groups


def _pooled(results, candidate, control, *, active_only=False):
    new, old, invalid = [], [], []
    valid_contexts = 0
    for result in results:
        for turn in range(1, result["planned_requests"]+1):
            canonical = next((r for r in result["canonical"] if r["arm"] == "BANK" and r["turn"] == turn), None)
            if active_only and (canonical is None or not any(canonical.get("read", {}).get("coefficients", []))):
                continue
            row, groups = paired_turn(result, turn, candidate, control)
            if row["valid"]:
                valid_contexts += 1; new.extend(groups[0]); old.extend(groups[1])
            else:
                invalid.append(dict(episode_id=result["episode_id"], turn=turn, reasons=row["invalid_reasons"]))
    a, b = totals(new), totals(old)
    return dict(candidate=candidate, control=control, active_bank_turns_only=active_only, valid_contexts=valid_contexts,
        excluded_contexts=invalid, candidate_metrics=a, control_metrics=b,
        request_time_reduction_percent=_reduction(a["complete_request_s"], b["complete_request_s"]),
        verification_call_reduction_percent=_reduction(a["verification_calls"], b["verification_calls"]),
        calls_per_1000_reduction_percent=_reduction(a["calls_per_1000_output_tokens"], b["calls_per_1000_output_tokens"]))


def cost_breakdown(result):
    by_arm = {}
    for arm in ("BANK", "OSD"):
        canonical = [r for r in result["canonical"] if r["arm"] == arm]
        updates = [r for r in result["updates"] if r["arm"] == arm]
        components = dict(canonical_serving_s=_sum(canonical, "request_s"),
            external_activation_s=_sum(canonical, "deployment_activation_s"),
            external_cache_maintenance_s=_sum(canonical, "cache_maintenance_s"),
            recurring_adaptation_s=_sum(updates, "recurring_adaptation_s"))
        complete = (len(canonical) == result["planned_requests"] and len(updates) == result["planned_requests"]-1
                    and all(value is not None for value in components.values()))
        by_arm[arm] = dict(nonoverlapping_components=components,
            canonical_input_trajectory=arm,
            complete_nonoverlapping_accounting=complete,
            serving_only_s=components["canonical_serving_s"],
            serving_plus_adaptation_episode_s=sum(components.values()) if complete else None,
            one_time_loading_receipt=result.get("startup"),
            actual_shared_worker_activation_s=_sum(canonical, "activation_s"),
            update_cost_details=[dict(turn=r["event"], costs=r.get("costs"),
                accounting_costs=r.get("accounting", {}).get("costs"),
                recurring_adaptation_s=r.get("recurring_adaptation_s"),
                one_time_loading_s=r.get("one_time_loading_s"),
                exposure=r.get("accounting")) for r in updates])
    final = [r for r in result.get("feedback_summaries", []) if r.get("scheduled_steps") == 0]
    frozen_rows, invalid_contexts = [], []
    for turn in range(1, result["planned_requests"]+1):
        pair, groups = paired_turn(result, turn, "BANK", "F")
        if pair["valid"]:
            frozen_rows.extend(groups[1])
        else:
            invalid_contexts.append(dict(turn=turn, reasons=pair["invalid_reasons"]))
    frozen_request = _sum(frozen_rows, "request_s")
    frozen_cache = _sum(frozen_rows, "cache_maintenance_s")
    frozen_complete = (not invalid_contexts and len(frozen_rows) == 3*result["planned_requests"] and
                       frozen_request is not None and frozen_cache is not None)
    frozen_reference = dict(trajectory="BANK", reconstructed_from_fresh_timing_repeats=True,
        actual_frozen_canonical_session=False, complete_valid_context_coverage=frozen_complete,
        invalid_contexts=invalid_contexts, timing_executions_used=len(frozen_rows),
        serving_only_s=frozen_request/3 if frozen_complete else None,
        cache_maintenance_s=frozen_cache/3 if frozen_complete else None,
        serving_plus_adaptation_episode_s=(frozen_request+frozen_cache)/3 if frozen_complete else None,
        adaptation_s=0., recurring_activation_s=0.,
        interpretation="Frozen reference reconstructed from the mean of three fresh paired measurements per BANK input. "
            "Includes the pinned per-request cache maintenance; excludes one-time loading and artificial method-switch "
            "publications. It is not a separately executed frozen conversation. BANK/OSD deployment costs use their "
            "own canonical trajectories, which may differ.")
    return dict(arms=by_arm, final_feedback_analysis_only=final,
        frozen_reference_on_bank_trajectory=frozen_reference,
        benchmark_request_s=_sum(result["scores"], "request_s"), benchmark_executions=len(result["scores"]),
        benchmark_excluded_from_deployment=True, actual_stream_wall_s=result.get("actual_wall_s"),
        accounting_note="Update components explain recurring_adaptation_s and are not added to it again. "
            "Capture included in an outer timer is never added twice. Final response analysis and "
            "benchmark-only loading/replays are not deployment work. Native sub-timers may overlap.")


def bank_usage(result):
    rows = [r for r in result["canonical"] if r["arm"] == "BANK"]
    counts = {0: 0, 1: 0, 2: 0}
    decisions = []
    for row in rows:
        read = row.get("read", {})
        coefficients = read.get("coefficients", [])
        active = sum(a > 0 for a in coefficients)
        counts[active] = counts.get(active, 0)+1
        feedback = next((u for u in result.get("feedback_summaries", []) if u["event"] == row["turn"]), {})
        predicted = read.get("forecast")
        if predicted is None and type(read.get("diagnostics", {}).get("objective")) in (int, float):
            predicted = -read["diagnostics"]["objective"]
        decisions.append(dict(turn=row["turn"], coefficients=coefficients, active_modules=active,
            pre_read_state_sha256=row.get("state_sha256"), predicted_gain=predicted,
            prediction_scope="Predicted local matching-prefix progress gain; not request speedup",
            realized_served_gain=feedback.get("realized_progress_gain"),
            supervised_positions=feedback.get("accounting", {}).get("eligible_positions"),
            new_memory_blocks=feedback.get("new_memory_blocks"),
            protection_failures=feedback.get("protection_failures"),
            source_fit=feedback.get("source_fit"),
            post_write_fit_gain=feedback.get("post_write_gain", feedback.get("singleton_fit_gain")),
            owner=feedback.get("owner"), scheduled_steps=feedback.get("scheduled_steps"),
            accepted_steps=feedback.get("accepted_steps"), blocked_writes=feedback.get("blocked_writes"),
            prewrite_singleton_gains=feedback.get("prewrite_singleton_gains"),
            singleton_gain_scope="Post-response diagnosis of existing heads; never an input to this request's read",
            read=read, feedback=feedback))
    owners = defaultdict(int)
    for row in decisions:
        if row["owner"] is not None:
            owners[str(row["owner"])] += 1
    return dict(requests=len(rows), zero_count=counts[0], singleton_count=counts[1], pair_count=counts[2],
        owner_event_counts=dict(owners),
        active_use_rate=(counts[1]+counts[2])/len(rows) if rows else None,
        decisions=decisions, distinction="Served gain uses a pre-response selection; post-write gain is source fit. "
            "Neither is itself a verification or latency improvement.")


def summarize(results):
    results = list(results)
    sources = sorted({r["source"] for r in results})
    per_turn = [paired_turn(r, t, *pair)[0] for r in results for t in range(1, r["planned_requests"]+1)
                for pair in COMPARISONS]
    by_source = {source: [_pooled([r for r in results if r["source"] == source], *pair) for pair in COMPARISONS]
                 for source in sources}
    return dict(schema_version=1, scope=SCOPE,
        interpretation="These six inspected streams support mechanism development, not an unseen-data generalization claim.",
        per_turn=per_turn, by_source=by_source,
        active_turns_by_source={source: [_pooled([r for r in results if r["source"] == source], *pair,
                                                active_only=True) for pair in COMPARISONS] for source in sources},
        overall=[_pooled(results, *pair) for pair in COMPARISONS],
        conversations=[dict(episode_id=r["episode_id"], source=r["source"], synthetic=r["synthetic"],
            status=r["status"], comparisons=[_pooled([r], *pair) for pair in COMPARISONS],
            active_turn_comparisons=[_pooled([r], *pair, active_only=True) for pair in COMPARISONS],
            bank=bank_usage(r), history_comparisons=r.get("history_comparisons", []),
            cost=cost_breakdown(r), implementation_checks=r.get("implementation_checks"),
            qualification_checks=r.get("qualification_checks"),
            pre_request_state_references=r.get("pre_request_states", []),
            osd_retention_receipts=[u.get("retention") for u in r["updates"] if u["arm"] == "OSD"])
            for r in results],
        invalid_timed_executions=sum(row.get("same_generation_valid") is not True for r in results for row in r["scores"]),
        result_count=len(results))


def _csv_value(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) if isinstance(value, (dict, list)) else value


def _write_csv(path, rows):
    rows = list(rows)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows({key: _csv_value(value) for key, value in row.items()} for row in rows)


def _pct(value):
    return "unavailable" if value is None else f"{value:+.2f}%"


def write_reports(results, directory):
    results = list(results); directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    report = summarize(results)
    (directory/"REPORT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    per_turn = []
    for row in report["per_turn"]:
        item = {key: value for key, value in row.items() if not key.endswith("_metrics")}
        for role in ("candidate", "control"):
            item.update({role+"_"+key: value for key, value in row[role+"_metrics"].items()})
        per_turn.append(item)
    _write_csv(directory/"per_turn.csv", per_turn)
    _write_csv(directory/"timing_repeats.csv", [row for result in results for row in result["scores"]])
    _write_csv(directory/"bank_decisions.csv", [dict(episode_id=r["episode_id"], **row)
        for r in results for row in bank_usage(r)["decisions"]])
    text = ["# Correction-bank v0.2 comparison", "", SCOPE, "",
        "Positive percentages mean reduction. Invalid paired contexts remain in the evidence and are excluded from gain claims.", "",
        "| Source | Comparison | Valid contexts | Request-time reduction | Verification-call reduction |",
        "| --- | --- | ---: | ---: | ---: |"]
    for source, rows in report["by_source"].items():
        for row in rows:
            text.append(f'| {source} | {row["candidate"]} vs {row["control"]} | {row["valid_contexts"]} | '
                        f'{_pct(row["request_time_reduction_percent"])} | {_pct(row["verification_call_reduction_percent"])} |')
    text += ["", "Complete costs, every regression, ownership/read decisions and immutable state references are in REPORT.json. "
        "Every timing execution is retained in timing_repeats.csv; per_turn.csv uses paired sums across three repeats.", "",
        "The main comparison describes checkpoints on the bank trajectory. Each learning arm was trained on its own canonical responses; "
        "history differences are reported rather than rewritten. Bank source-fit gains, served anchored loss changes, "
        "and native serving improvements are different quantities.", "",
        "Diagnosis should distinguish correction capacity, blocked preservation constraints, failed transfer selection, "
        "fusion behavior and runtime overhead. A negative serving result alone does not identify which mechanism failed.", "",
        "No extra baseline campaign or ablation was run by this report. These inspected streams do not establish generalization."]
    (directory/"CONVERSATION_REPORT.md").write_text("\n".join(text)+"\n")
    return report
