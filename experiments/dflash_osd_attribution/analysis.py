"""Offline attribution of the five completed main streams; never loads a model.

The caller resolves and verifies artifact bindings. Timing repeats describe one
request, not independent token samples. This module makes no causal attribution
from token recurrence and never converts local repairs into saved-time shares.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
import re
import unicodedata

STREAM_COUNTS = {"wildchat_00": 20, "wildchat_01": 16, "wildchat_02": 17,
                 "wildchat_19": 23, "personamem_persona42": 18}
METHODS = ("F", "OSD")
CATEGORIES = ("whitespace_punctuation", "code_syntax", "identifiers_numbers",
              "ordinary_text", "mixed_uncategorized")
CODE_LEXEMES = frozenset(("def", "class", "import", "return", "const", "let", "var",
                        "function", "elif", "None", "null", "True", "False",
                        "==", "!=", "=>", "->", "::", "&&", "||", "++", "--"))


def token_hash(ids):
    return hashlib.sha256(json.dumps(ids, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _num(x):
    return type(x) in (int, float) and math.isfinite(x) and x >= 0


def _sum(rows, key):
    rows = list(rows)
    return sum(r[key] for r in rows) if all(_num(r.get(key)) for r in rows) else None


def _pct(new, old):
    return 100 * (1 - new / old) if old else None


def category(text):
    """Mutually exclusive lexical heuristic, never a semantic/domain classifier."""
    if text is None:
        return "mixed_uncategorized"
    core = text.strip()
    if not core or all(c.isspace() for c in text):
        return "whitespace_punctuation"
    if core in CODE_LEXEMES:
        return "code_syntax"
    if all(unicodedata.category(c).startswith(("P", "S")) for c in core):
        return "whitespace_punctuation"
    if (re.fullmatch(r"[+-]?(?:\d+(?:[.,]\d+)*)", core) or
            re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", core) and
            ("_" in core or any(c.isdigit() for c in core) or re.search(r"[a-z][A-Z]", core))):
        return "identifiers_numbers"
    if core.isalpha():
        return "ordinary_text"
    return "mixed_uncategorized"


def validated_events(trace, row):
    """Return block/decision rows and structural issues for one native trajectory.

    Position zero is the first generated token. A target prediction at a rejected
    proposal history is not a label for the canonical history. Such positions
    remain recorded but cannot enter the repairs/new-errors denominator.
    """
    issues, blocks, decisions = [], [], []
    inputs, output = row["input_ids"], row["output_ids"]
    if trace.get("input_token_ids") != inputs or trace.get("final_output_token_ids") != output:
        issues.append("trace_input_or_output_ids_mismatch")
    rec = trace.get("reconciliation", {})
    if rec.get("status") != "passed":
        issues.append("trace_reconciliation_not_passed")
    for key, expected in (("input_ids_sha256", token_hash(inputs)), ("output_ids_sha256", token_hash(output))):
        if rec.get(key) != expected:
            issues.append("trace_" + key + "_mismatch")
    events = [e for e in trace.get("events", []) if e.get("kind") == "verification"]
    committed = [t for e in trace.get("events", []) for t in e.get("committed_token_ids", [])]
    if committed != output:
        issues.append("trace_committed_sequence_mismatch")
    if len(events) != row.get("verification_calls"):
        issues.append("trace_verification_count_mismatch")
    accepted = sum(e.get("committed_provenance", []).count("accepted_draft") for e in events)
    if accepted != row.get("accepted_draft_tokens"):
        issues.append("trace_accepted_count_mismatch")
    if sum(len(e.get("committed_token_ids", [])) for e in events) != row.get("verification_committed_tokens"):
        issues.append("trace_committed_count_mismatch")
    for event in events:
        start = event.get("output_offset")
        proposed = event.get("draft_proposed_token_ids", [])
        positions = event.get("draft_proposed_positions", [])
        verifier_input = event.get("verification_input_token_ids", [])
        target_positions = event.get("target_prediction_positions", [])
        target_tokens = event.get("target_predicted_token_ids", [])
        targets = dict(zip(target_positions, target_tokens))
        problems = []
        if type(start) is not int or start < 1 or start > len(output):
            issues.append("invalid_block_output_start"); continue
        if positions != list(range(len(inputs) + start, len(inputs) + start + len(proposed))):
            problems.append("proposal_positions_not_absolute_contiguous")
        if (len(target_positions) != len(target_tokens) or len(targets) != len(target_positions) or
                not set(positions).issubset(targets)):
            problems.append("target_prediction_positions_missing_or_duplicate")
        if verifier_input != [output[start-1]] + proposed:
            problems.append("verification_seed_or_proposals_mismatch")
        context = event.get("context", {})
        if context.get("committed_output_prefix_length") != start or context.get("input_ids_sha256") != token_hash(inputs):
            problems.append("block_prefix_binding_mismatch")
        if event.get("committed_token_ids") != output[start:start+len(event.get("committed_token_ids", []))]:
            problems.append("committed_canonical_segment_mismatch")
        rejection = event.get("rejection_boundary_proposal_index")
        raw_rejection = next((i for i, p in enumerate(proposed)
                              if targets.get(len(inputs)+start+i) != p), None)
        if rejection != raw_rejection:
            problems.append("first_rejection_boundary_mismatch")
        issues.extend(problems)
        accepted_prefix = raw_rejection if raw_rejection is not None else len(proposed)
        blocks.append(dict(block_output_start=start, absolute_token_start=len(inputs)+start,
            native_round_index=event.get("round_index"), proposed_count=len(proposed),
            accepted_prefix_length_before_stop=accepted_prefix,
            accepted_draft_tokens_emitted=event.get("committed_provenance", []).count("accepted_draft"),
            first_rejection_block_position=rejection,
            first_rejection_output_position=None if rejection is None else start+rejection,
            stop_clipped=event.get("stop_clipped"), valid=not problems))
        for i, proposal in enumerate(proposed):
            position = start+i
            # The target conditions on the seed and earlier proposals. Equality
            # is checked against exact IDs, not text or verification round index.
            prefix_ok = (not problems and position < len(output) and
                         proposed[:i] == output[start:position])
            target = targets.get(len(inputs)+position)
            if prefix_ok and target != output[position]:
                issues.append("canonical_prefix_target_disagrees_with_canonical_token")
                prefix_ok = False
            decisions.append(dict(absolute_output_position=position,
                absolute_token_position=len(inputs)+position, block_output_start=start,
                block_position=i, proposed_token_id=proposal, target_token_id=target,
                canonical_token_id=output[position] if position < len(output) else None,
                canonical_prefix_valid=prefix_ok, correct=proposal == target,
                unavailable_reason=None if prefix_ok else ("beyond_final_output" if position >= len(output)
                                                          else "noncanonical_proposal_history_or_invalid_event")))
    return blocks, decisions, sorted(set(issues))


def _validate_scores(result, turn, canonical):
    groups = {arm: sorted((r for r in result["scores"] if r["arm"] == arm and r["turn"] == turn),
                         key=lambda r: r.get("repeat", -1)) for arm in METHODS}
    problems = []
    for arm, rows in groups.items():
        if len(rows) != 3 or [r.get("repeat") for r in rows] != [0, 1, 2]:
            problems.append(arm + ":missing_or_duplicate_repeat")
        if len({r.get("state_sha256") for r in rows}) != 1 or any(not r.get("state_sha256") for r in rows):
            problems.append(arm + ":state_identity_changed_or_absent")
        for row in rows:
            for key in ("same_generation_valid", "input_match", "canonical_output_match", "checkpoint_identity_match"):
                if row.get(key) is not True:
                    problems.append(arm + ":" + key)
            for key in ("input_ids", "output_ids"):
                if row.get(key) != canonical.get(key):
                    problems.append(arm + ":exact_" + key + "_mismatch")
            if row.get("capture") is not False or row.get("feedback") is not None or row.get("capture_costs") is not None:
                problems.append(arm + ":capture_contaminated")
            if not _num(row.get("request_s")) or row["request_s"] == 0:
                problems.append(arm + ":invalid_request_time")
        for metric in ("verification_calls", "accepted_draft_tokens", "verification_committed_tokens"):
            if any(not _num(r.get(metric)) for r in rows) or len({r.get(metric) for r in rows}) != 1:
                problems.append(arm + ":repeat_" + metric + "_mismatch")
    return groups, sorted(set(problems))


def _trained_tokens(result, turn):
    """Only answers whose completed update precedes this scored request count."""
    tokens, events, issues = set(), [], []
    for update in result.get("updates", []):
        if update.get("arm") != "OSD" or update.get("event", turn) >= turn:
            continue
        source = next((r for r in result["canonical"] if r["arm"] == "OSD" and r["turn"] == update["event"]), None)
        responses = update.get("responses", [])
        if (update.get("status") != "passed" or update.get("mask_and_schedule_check") is not True or
                source is None or len(responses) != 1 or
                any(responses[0].get(k) != source.get(k) for k in ("input_ids", "output_ids"))):
            issues.append(dict(event=update["event"], reason="training_response_binding_unavailable_or_mismatch"))
            continue
        tokens.update(source["output_ids"]); events.append(update["event"])
    return tokens, events, issues


def _aligned_decisions(frozen, osd, output, prior_tokens, decode_token):
    """Retain canonical-position pairs without conflating drafter conditioning.

    Equal verifier prefixes are necessary but insufficient for a DFlash repair
    comparison: a token at masked slot 3 is not conditioned like slot 1 of a
    later block. Only equal block starts receive a matched-condition effect.
    """
    maps = []
    for rows in (frozen, osd):
        by_position = defaultdict(list)
        for row in rows:
            if row["canonical_prefix_valid"]:
                by_position[row["absolute_output_position"]].append(row)
        maps.append(by_position)
    aligned, unmatched = [], []
    for position in sorted(set(maps[0]) | set(maps[1])):
        a, b = maps[0].get(position, []), maps[1].get(position, [])
        if len(a) != 1 or len(b) != 1:
            unmatched.append(dict(absolute_output_position=position, frozen_decisions=len(a), osd_decisions=len(b),
                                  reason="missing_or_ambiguous_canonical_prefix_decision")); continue
        a, b = a[0], b[0]
        token = output[position]
        earlier_training = token in prior_tokens
        current_repeat = token in output[:position]
        text = decode_token(token)
        same_start = a["block_output_start"] == b["block_output_start"]
        effect = ("repair" if not a["correct"] and b["correct"] else
                  "new_error" if a["correct"] and not b["correct"] else
                  "both_correct" if a["correct"] else "both_wrong")
        aligned.append(dict(absolute_output_position=position, canonical_token_id=token,
            token_text=text, token_text_available=text is not None, category=category(text),
            frozen_correct=a["correct"], osd_correct=b["correct"],
            comparison_scope="matched_condition" if same_start else "different_start_native_observation",
            effect=effect if same_start else None,
            native_outcome={"repair": "frozen_wrong_osd_correct", "new_error": "frozen_correct_osd_wrong",
                            "both_correct": "both_correct", "both_wrong": "both_wrong"}[effect],
            frozen_block_start=a["block_output_start"], osd_block_start=b["block_output_start"],
            same_block_start=same_start,
            frozen_block_position=a["block_position"], osd_block_position=b["block_position"],
            appeared_in_prior_trained_response=earlier_training,
            appeared_earlier_in_current_response=current_repeat,
            recurrence=("earlier_training" if earlier_training else
                        "current_response_only" if current_repeat else "not_seen_in_either")))
    return aligned, unmatched


def cost_rows(result):
    """Detail timers may overlap; only outer disjoint costs enter episode totals."""
    updates = [u for u in result["updates"] if u["arm"] == "OSD"]
    canonical = [r for r in result["canonical"] if r["arm"] == "OSD"]
    serving = _sum(canonical, "request_s")
    activation = _sum(canonical, "deployment_activation_s")
    cache = _sum(canonical, "cache_maintenance_s")
    adaptation = _sum(updates, "recurring_adaptation_s")
    components = [serving, activation, cache, adaptation]
    details = [dict(episode_id=result["episode_id"], event=u["event"],
                    recurring_adaptation_s=u.get("recurring_adaptation_s"),
                    costs=u.get("costs", {}), accounting_costs=u.get("accounting", {}).get("costs", {}),
                    optimizer_steps=u.get("accounting", {}).get("optimizer_steps_total")) for u in updates]
    cost_dicts = [u.get("costs", {}) for u in updates]
    return dict(episode_id=result["episode_id"], canonical_trajectory="OSD",
        canonical_serving_s=serving, external_activation_s=activation, external_cache_s=cache,
        recurring_adaptation_s=adaptation,
        serving_plus_adaptation_episode_s=sum(components) if all(v is not None for v in components) else None,
        capture_preparation_s=_sum(cost_dicts, "capture_preparation_s"),
        teacher_preparation_s=_sum(cost_dicts, "teacher_preparation_s"),
        training_including_teacher_s=_sum(cost_dicts, "training_s"),
        draft_training_and_optimizer_s=_sum(cost_dicts, "draft_training_and_optimizer_s"),
        export_s=_sum(cost_dicts, "export_s"), optimizer_state_save_s=_sum(cost_dicts, "optimizer_state_save_s"),
        one_time_loading=result.get("startup"), actual_stream_wall_s=result.get("actual_wall_s"),
        update_details=details,
        accounting="Episode total sums serving + external activation + external cache + adaptation outer wall once. "
                   "Capture already inside serving stays there; preparation/training/export sub-timers are not added again. "
                   "Teacher preparation is included in training_s. Benchmark-only work is excluded. No kernel breakdown inferred.")


def _aggregate(rows, label):
    valid = [r for r in rows if r["valid"]]
    f_time = sum(r["frozen_request_s"] for r in valid)
    o_time = sum(r["osd_request_s"] for r in valid)
    f_calls = sum(r["frozen_verification_calls"] for r in valid)
    o_calls = sum(r["osd_verification_calls"] for r in valid)
    output = sum(r["output_tokens"] for r in valid)
    return dict(group=label, requests=len(rows), valid_requests=len(valid), excluded_requests=len(rows)-len(valid),
        frozen_request_s=f_time, osd_request_s=o_time, saved_request_s=f_time-o_time,
        request_time_reduction_percent=_pct(o_time, f_time), frozen_verification_calls=f_calls,
        osd_verification_calls=o_calls, saved_verification_calls=f_calls-o_calls,
        verification_call_reduction_percent=_pct(o_calls, f_calls), output_tokens=output,
        frozen_calls_per_1000_tokens=1000*f_calls/output if output else None,
        osd_calls_per_1000_tokens=1000*o_calls/output if output else None)


def analyze_campaign(results, load_trace, decode_token, *, require_full_campaign=True):
    """Return JSON/CSV-ready tables; caller owns validated artifact resolution.

    ``load_trace(binding)`` must verify the original binding before decoding JSON.
    ``decode_token(token_id)`` must use the campaign's pinned tokenizer, or return
    None when that tokenizer is unavailable (no lexical substitution). These
    dependencies are explicit to prevent accidental runtime/model execution.
    """
    results = list(results)
    ids = [r["episode_id"] for r in results]
    if len(ids) != len(set(ids)) or any(i not in STREAM_COUNTS for i in ids):
        raise ValueError("Only the five distinct completed main streams are allowed")
    if any(r.get("qualification") is not False or r.get("status") != "complete" for r in results):
        raise ValueError("Qualification and incomplete streams are excluded")
    if require_full_campaign and (set(ids) != set(STREAM_COUNTS) or
            any(r["planned_requests"] != STREAM_COUNTS[r["episode_id"]] for r in results)):
        raise ValueError("Expected exactly the declared 94 main-campaign requests")
    per_turn, timing, blocks, decisions, unmatched, exclusions, recurrence_issues = [], [], [], [], [], [], []
    token_coverage = []
    for result in results:
        stream = result["episode_id"]
        for turn in range(1, result["planned_requests"]+1):
            identity = dict(episode_id=stream, source=result["source"], turn=turn)
            canonical = [r for r in result["canonical"] if r["arm"] == "BANK" and r["turn"] == turn]
            if len(canonical) != 1:
                raise ValueError(f"{stream} turn {turn}: unique BANK canonical required")
            canonical = canonical[0]
            groups, problems = _validate_scores(result, turn, canonical)
            record = {**identity, "valid": not problems, "invalid_reasons": problems,
                      "output_tokens": len(canonical["output_ids"]), "context_tokens": len(canonical["input_ids"])}
            for arm, rows in groups.items():
                for row in rows:
                    timing.append({**identity, "arm": arm, "repeat": row.get("repeat"),
                        "request_s": row.get("request_s"), "state_sha256": row.get("state_sha256"),
                        "paired_request_valid": not problems, "invalid_reasons": problems,
                        "verification_calls": row.get("verification_calls"), "trajectory": row.get("trajectory")})
            if not problems:
                for arm, name in (("F", "frozen"), ("OSD", "osd")):
                    rows = groups[arm]
                    record[name+"_request_s"] = sum(r["request_s"] for r in rows)/3
                    for metric in ("verification_calls", "accepted_draft_tokens", "verification_committed_tokens"):
                        record[name+"_"+metric] = rows[0][metric]
                    record[name+"_state_sha256"] = rows[0]["state_sha256"]
                record["saved_request_s"] = record["frozen_request_s"]-record["osd_request_s"]
                record["saved_verification_calls"] = record["frozen_verification_calls"]-record["osd_verification_calls"]
                record["request_time_reduction_percent"] = _pct(record["osd_request_s"], record["frozen_request_s"])
                record["verification_call_reduction_percent"] = _pct(record["osd_verification_calls"], record["frozen_verification_calls"])
                prior, prior_events, prior_issues = _trained_tokens(result, turn)
                recurrence_issues.extend({**identity, **v} for v in prior_issues)
                method_decisions, token_problems = {}, []
                method_blocks = {}
                for arm in METHODS:
                    row = groups[arm][0]  # One trajectory: deterministic repeat zero.
                    try:
                        trace = load_trace(row["trajectory"])
                        b, d, issues = validated_events(trace, row)
                    except (OSError, ValueError, KeyError, TypeError) as error:
                        b, d, issues = [], [], ["trace_unavailable:"+str(error)]
                    token_problems.extend(arm+":"+p for p in issues)
                    method_decisions[arm], method_blocks[arm] = d, b
                    blocks.extend({**identity, "arm": arm, "selected_repeat": 0, **v} for v in b)
                if not token_problems:
                    aligned, missing = _aligned_decisions(method_decisions["F"], method_decisions["OSD"],
                                                          canonical["output_ids"], prior, decode_token)
                    if prior_issues:
                        for value in aligned:
                            value["appeared_in_prior_trained_response"] = None
                            value["recurrence"] = "prior_training_evidence_unavailable"
                    decisions.extend({**identity, "prior_trained_events": prior_events, **v} for v in aligned)
                    unmatched.extend({**identity, **v} for v in missing)
                    starts = [{v["block_output_start"] for v in method_blocks[a]} for a in METHODS]
                    token_coverage.append({**identity, "valid": True, "paired_canonical_decisions": len(aligned),
                        "matched_condition_decisions": sum(v["same_block_start"] for v in aligned),
                        "different_start_observations": sum(not v["same_block_start"] for v in aligned),
                        "unmatched_decision_positions": len(missing), "matched_block_starts": len(starts[0]&starts[1]),
                        "frozen_unmatched_block_starts": len(starts[0]-starts[1]),
                        "osd_unmatched_block_starts": len(starts[1]-starts[0]),
                        **{a.lower()+"_noncanonical_or_beyond_output_proposals": sum(not d["canonical_prefix_valid"]
                           for d in method_decisions[a]) for a in METHODS}})
                else:
                    exclusions.append({**identity, "scope": "token_analysis", "reasons": token_problems})
                    token_coverage.append({**identity, "valid": False, "reasons": token_problems})
            else:
                exclusions.append({**identity, "scope": "timing_and_token_analysis", "reasons": problems})
            per_turn.append(record)
    total = _aggregate(per_turn, "all_five_main_streams")
    conversations = [_aggregate([r for r in per_turn if r["episode_id"] == stream], stream) for stream in ids]
    for row in conversations:
        row["share_of_net_saved_time_percent"] = 100*row["saved_request_s"]/total["saved_request_s"] if total["saved_request_s"] else None
        row["share_of_net_saved_calls_percent"] = 100*row["saved_verification_calls"]/total["saved_verification_calls"] if total["saved_verification_calls"] else None
    matched = [d for d in decisions if d["comparison_scope"] == "matched_condition"]
    different = [d for d in decisions if d["comparison_scope"] == "different_start_native_observation"]
    pattern, different_pattern = [], []
    for source in sorted({r["source"] for r in results}):
        subset = [d for d in matched if d["source"] == source]
        different_subset = [d for d in different if d["source"] == source]
        for cat in CATEGORIES:
            for recurrence in ("earlier_training", "current_response_only", "not_seen_in_either", "prior_training_evidence_unavailable"):
                selected = [d for d in subset if d["category"] == cat and d["recurrence"] == recurrence]
                counts = Counter(d["effect"] for d in selected)
                pattern.append(dict(source=source, category=cat, recurrence=recurrence,
                    comparison_scope="matched_condition",
                    paired_decision_denominator=len(selected), frozen_error_denominator=counts["repair"]+counts["both_wrong"],
                    frozen_correct_denominator=counts["new_error"]+counts["both_correct"],
                    **{effect: counts[effect] for effect in ("repair", "new_error", "both_correct", "both_wrong")}))
                observations = [d for d in different_subset if d["category"] == cat and d["recurrence"] == recurrence]
                outcomes = Counter(d["native_outcome"] for d in observations)
                different_pattern.append(dict(source=source, category=cat, recurrence=recurrence,
                    comparison_scope="different_start_native_observation",
                    paired_observation_denominator=len(observations),
                    frozen_error_observation_denominator=outcomes["frozen_wrong_osd_correct"]+outcomes["both_wrong"],
                    frozen_correct_observation_denominator=outcomes["frozen_correct_osd_wrong"]+outcomes["both_correct"],
                    **{effect: outcomes[effect] for effect in
                       ("frozen_wrong_osd_correct", "frozen_correct_osd_wrong", "both_correct", "both_wrong")}))
    return dict(schema_version=2, overall=total, per_turn=per_turn, conversations=conversations,
        sources=[_aggregate([r for r in per_turn if r["source"] == source], source) for source in sorted({r["source"] for r in results})],
        timing_repeats=timing, blocks=blocks, aligned_decisions=decisions, unmatched_positions=unmatched,
        matched_condition_decisions=matched, different_start_observations=different,
        token_coverage=token_coverage, pattern_summary=pattern, costs=[cost_rows(r) for r in results],
        different_start_pattern_summary=different_pattern,
        exclusions=exclusions, recurrence_evidence_issues=recurrence_issues,
        method="Three repeat times averaged per request; one validated repeat-zero trajectory per method/request. "
               "Ratios pool summed measurements. Exact canonical prefix and absolute output position align decisions. "
               "Repairs/new errors and their recurrence statistics additionally require identical draft-block starts. "
               "Different-start native outcomes retain separate descriptive denominators, never pooled repair counts. "
               "Code syntax and other token categories are lexical heuristics, not semantic task labels. Recurrence is exact token-ID presence; "
               "not_seen_in_either does not establish semantic novelty. Associations do not prove memorization or specialization. "
               "Local repair frequency cannot be converted to a percentage of total saved time. Conversation contributions may be negative or exceed 100%.")
