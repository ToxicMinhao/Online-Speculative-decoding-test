"""Compact evidence report; no model execution and no extrapolated speed claims."""
from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path

from .bound import TOLERANCE_FORMULA


def _csv(path, rows):
    rows = list(rows)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True, ensure_ascii=False, allow_nan=False)
                             if isinstance(v, (dict, list)) else v for k, v in row.items()})


def _json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)+"\n")


def _cell(value):
    if value is None:
        return "unavailable"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _table(columns, rows):
    if not rows:
        return ["No supported observations available.", ""]
    lines = ["| " + " | ".join(label for _, label in columns) + " |",
             "| " + " | ".join("---" for _ in columns) + " |"]
    lines.extend("| " + " | ".join(_cell(row.get(key)) for key, _ in columns) + " |" for row in rows)
    return lines+[""]


def summarize_diagnostic(comparisons):
    """Denominators are anchor decisions, not independent requests or gains in time."""
    comparisons = list(comparisons)
    if len({r["anchor_id"] for r in comparisons}) != len(comparisons):
        raise ValueError("Duplicate diagnostic anchor IDs would inflate denominators")
    anchors, decisions = [], []
    for row in comparisons:
        anchors.append({k: v for k, v in row.items() if k != "decisions"})
        for original in row.get("decisions", []):
            decision = dict(original)
            # Never allow a child flag to override a failed frozen reproduction.
            decision["valid"] = row.get("valid") is True and original.get("valid") is True
            decision["repair"] = decision["valid"] and original.get("repair") is True
            decision["introduced_error"] = decision["valid"] and original.get("introduced_error") is True
            capacity = dict(original.get("capacity") or {})
            decision["capacity"] = capacity
            for key, value in capacity.items():
                decision["capacity_"+key] = value
            decisions.append(decision)
    summaries = []
    groups = [(stream, [r for r in anchors if r["stream"] == stream],
               [r for r in decisions if r["stream"] == stream]) for stream in sorted({r["stream"] for r in anchors})]
    groups.append(("all_available", anchors, decisions))
    for stream, group_anchors, group_decisions in groups:
        valid = [r for r in group_anchors if r.get("valid") is True]
        repairs = [r for r in group_decisions if r["repair"]]
        counts = Counter()
        for repair in repairs:
            capacity = repair["capacity"]
            label = capacity.get("classification") if capacity.get("eligible") is True else None
            counts[label if label in ("excluded", "borderline", "not_excluded") else "missing_or_unavailable"] += 1
        evaluated = counts["excluded"]+counts["borderline"]+counts["not_excluded"]
        summaries.append(dict(stream=stream, anchors=len(group_anchors), valid_anchors=len(valid),
            invalid_anchors=len(group_anchors)-len(valid),
            frozen_correct_prefix_sum=sum(r["frozen_correct_prefix_length"] for r in valid),
            osd_correct_prefix_sum=sum(r["osd_correct_prefix_length"] for r in valid),
            anchors_prefix_improved=sum(r["correct_prefix_gain"] > 0 for r in valid),
            anchors_prefix_worsened=sum(r["correct_prefix_gain"] < 0 for r in valid),
            anchors_prefix_unchanged=sum(r["correct_prefix_gain"] == 0 for r in valid),
            valid_position_denominator=sum(r["valid"] for r in group_decisions),
            valid_repair_denominator=len(repairs),
            valid_new_errors=sum(r["introduced_error"] for r in group_decisions),
            distinct_repaired_output_positions=len({(r["stream"], r["absolute_output_position"]) for r in repairs}),
            evaluated_repair_denominator=evaluated,
            mathematically_excluded=counts["excluded"],
            borderline_inconclusive=counts["borderline"], not_excluded=counts["not_excluded"],
            bound_missing_or_unavailable=counts["missing_or_unavailable"],
            excluded_fraction_of_all_valid_repairs=counts["excluded"]/len(repairs) if repairs else None,
            excluded_fraction_of_evaluated_repairs=counts["excluded"]/evaluated if evaluated else None))
    return dict(anchors=anchors, decisions=decisions, summary=summaries)


def _pattern_summary(rows):
    result = []
    for source in sorted({r["source"] for r in rows}):
        for recurrence in sorted({r["recurrence"] for r in rows if r["source"] == source}):
            group = [r for r in rows if r["source"] == source and r["recurrence"] == recurrence]
            result.append(dict(source=source, recurrence=recurrence,
                **{key: sum(r[key] for r in group) for key in
                   ("paired_decision_denominator", "frozen_error_denominator", "frozen_correct_denominator",
                    "repair", "new_error", "both_correct", "both_wrong")}))
    return result


def write_report(output_dir, analysis=None, diagnostic=None, inventory=None):
    """Write supplied evidence without running analysis, reconstructing or training.

    ``diagnostic`` is either a list of ``protocol.compare_anchor`` outputs or a
    dictionary with ``comparisons`` and optional run metadata. Missing sections
    are explicitly unavailable, never filled with historical or assumed values.
    """
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    if diagnostic is None:
        comparisons = []
    elif isinstance(diagnostic, dict):
        comparisons = diagnostic.get("comparisons", [])
    else:
        comparisons = diagnostic
    local = summarize_diagnostic(comparisons)
    overall_local = local["summary"][-1]
    native_patterns = _pattern_summary(analysis.get("pattern_summary", [])) if analysis is not None else []
    payload = dict(schema="osd_gain_attribution_report_v1", analysis=analysis,
                   diagnostic=diagnostic, diagnostic_summary=local["summary"], inventory=inventory)
    _json(out/"REPORT.json", payload)
    for name, rows in local.items():
        _csv(out/("diagnostic_"+name+".csv"), rows)
    if analysis is not None:
        for name in ("per_turn", "conversations", "sources", "timing_repeats", "blocks", "aligned_decisions",
                     "unmatched_positions", "token_coverage", "pattern_summary", "costs", "exclusions",
                     "recurrence_evidence_issues", "missing_streams"):
            _csv(out/(name+".csv"), analysis.get(name, []))
        _csv(out/"pattern_recurrence_summary.csv", native_patterns)
        _json(out/"native_overall.json", analysis.get("overall"))
    cases = (inventory or {}).get("cases", [])
    _csv(out/"artifact_cases.csv", cases)
    _csv(out/"artifact_trace_inventory.csv", (inventory or {}).get("trace_inventory", []))
    missing = [r for r in cases if r.get("status") != "ready"]
    lines = ["# OSD gains and correction-space attribution", "",
        "This report separates realized native serving performance from local fixed-prefix diagnostic forwards. "
        "No local repair rate is converted into a percentage of total time saved.", "",
        "## Existing native campaign", ""]
    if analysis is None:
        lines += ["Native campaign analysis is unavailable in this report.", ""]
    else:
        total = analysis["overall"]
        planned = analysis.get("planned_requests", 94)
        lines += [f"Coverage: {total['valid_requests']} valid paired requests out of {total['requests']} supplied requests; "
                  f"the fixed campaign denominator is {planned} planned requests. "
                  "Qualification and the unexecuted sixth stream are excluded. Three original timing repeats are "
                  "averaged per request; one validated repeat-zero trajectory per method/request supplies token observations.", ""]
        missing_streams = analysis.get("missing_streams", [])
        if missing_streams:
            lines += _table([("stream", "Missing source stream"), ("reason", "Reason"), ("issues", "Issues")], missing_streams)
            lines += ["Missing streams remain absent from supplied/valid denominators; no measurements are imputed "
                      "and no later checkpoint or replacement stream is used.", ""]
        lines += _table([("group", "Conversation"), ("valid_requests", "Valid requests"),
            ("saved_request_s", "Saved serving seconds"), ("request_time_reduction_percent", "Time reduction %"),
            ("saved_verification_calls", "Saved verification calls"),
            ("verification_call_reduction_percent", "Call reduction %"),
            ("share_of_net_saved_time_percent", "Share of net saved time %"),
            ("share_of_net_saved_calls_percent", "Share of net saved calls %")], analysis["conversations"])
        lines += _table([("group", "Pooled source"), ("valid_requests", "Valid requests"),
            ("saved_request_s", "Saved serving seconds"), ("request_time_reduction_percent", "Time reduction %"),
            ("saved_verification_calls", "Saved calls"), ("verification_call_reduction_percent", "Call reduction %")],
            analysis["sources"]+[total])
        lines += ["Ratios use summed measurements. Contributions retain negative values and can exceed 100% "
                  "when another conversation regresses. Paired serving measurements use the BANK arm's exact saved inputs "
                  "with available checkpoints; they do not include adaptation or checkpoint activation.", "",
                  "### Where recorded prediction changes occur", ""]
        lines += _table([("source", "Source"), ("recurrence", "Canonical-token recurrence"),
            ("paired_decision_denominator", "Paired positions"), ("frozen_error_denominator", "Frozen errors"),
            ("repair", "Repairs"), ("frozen_correct_denominator", "Frozen correct"), ("new_error", "New errors")], native_patterns)
        lines += ["Decision pairing requires exact canonical prefix token IDs and absolute output position; "
                  "verification-call number is not an alignment key. Unmatched starts/positions and rejected-history "
                  "predictions remain in the coverage/exclusion tables. Different block starts can still produce "
                  "comparable canonical-prefix decisions, but do not establish the mechanism of the change.", "",
                  "The declared categories—whitespace/punctuation, code-syntax lexemes, identifiers/numbers, ordinary "
                  "text and mixed/uncategorized—are lexical heuristics. The full category denominators are in "
                  "pattern_summary.csv. Earlier-training recurrence requires a verified prior OSD training response; "
                  "current-answer-only recurrence excludes it. Exact token-ID absence is a limited familiarity proxy, "
                  "not semantic novelty. These associations do not prove memorization or domain specialization.", "",
                  "### Cost boundary", ""]
        if analysis.get("tokenizer", {}).get("status") == "unavailable":
            lines[-2:-2] = ["Lexical categorization is unavailable because the pinned tokenizer could not be loaded: "
                + _cell(analysis["tokenizer"].get("reason")) + ". No substitute tokenizer was used. Token text remains "
                "null and token_text_available=false; mixed/uncategorized entries in this case mark missing lexical "
                "evidence rather than observed mixed text. Exact token-ID alignment and recurrence, timing and "
                "verification-count analyses remain available wherever their own validity checks pass.", ""]
        lines += _table([("episode_id", "Conversation"), ("canonical_serving_s", "OSD serving s"),
            ("capture_preparation_s", "Preparation s"), ("teacher_preparation_s", "Teacher s"),
            ("draft_training_and_optimizer_s", "Draft train/optimizer s"), ("export_s", "Export s"),
            ("external_activation_s", "Activation s"), ("external_cache_s", "Cache s"),
            ("recurring_adaptation_s", "Adaptation outer s"),
            ("serving_plus_adaptation_episode_s", "Episode s")], analysis["costs"])
        lines += ["These deployment costs use OSD's own canonical history, which may differ from BANK's paired inputs. "
                  "Episode cost adds serving, external activation, external cache maintenance and the adaptation outer "
                  "timer once. Preparation, training and export detail timers are not added to that outer timer again. "
                  "Teacher time is already included in training_s; capture inside request latency is not added twice. "
                  "One-time loading, benchmark replays and actual campaign wall time remain separate in REPORT.json. "
                  "Aggregate request latency does not identify individual kernel costs.", ""]
    lines += ["## Local matched-prefix diagnostic", ""]
    if not comparisons:
        lines += ["No local diagnostic comparisons are available; no capacity or local-improvement conclusion is made.", ""]
    else:
        lines += ["Only the final saved request of each available planned conversation is eligible. "
                  "At most 32 deterministic evenly spaced frozen block starts per request yield at most 160 pairs/320 "
                  "draft-block forwards. These are diagnostic forwards, not new native latency measurements or "
                  "independent conversation samples.", ""]
        lines += _table([("stream", "Conversation"), ("anchors", "Anchors"), ("valid_anchors", "Valid"),
            ("invalid_anchors", "Invalid"), ("frozen_correct_prefix_sum", "Frozen prefix sum"),
            ("osd_correct_prefix_sum", "OSD prefix sum"), ("anchors_prefix_improved", "Improved"),
            ("anchors_prefix_worsened", "Worsened"), ("valid_repair_denominator", "Positional repairs"),
            ("valid_new_errors", "New errors")], local["summary"])
        lines += ["Correct-prefix length stops at the first mismatch or matched EOS. Positional repairs after an "
                  "earlier mismatch remain labeled positional-only; they do not extend accepted progress. Overlapping "
                  "anchors can inspect the same absolute output position, so anchor decisions are not independent "
                  "tokens. Frozen proposal reproduction failures invalidate affected pairs and remain preserved.", "",
                  "### Mathematical correction-space screen", ""]
        lines += _table([("stream", "Conversation"), ("valid_repair_denominator", "All valid repairs"),
            ("evaluated_repair_denominator", "With bound evidence"), ("mathematically_excluded", "Excluded"),
            ("borderline_inconclusive", "Borderline"), ("not_excluded", "Not excluded"),
            ("bound_missing_or_unavailable", "Missing evidence")], local["summary"])
        lines += [f"Declared tolerance: `{TOLERANCE_FORMULA}`. Both deficit and bound use float64 with the saved fixed "
                  "artifacts and the original clipped feature map. The radius constrains the combined effective "
                  "matrix, including permitted fusion; it is not multiplied by module count.", "",
                  "A deficit exceeding the bound by more than the tolerance excludes that pairwise repair under "
                  "the mathematical correction model. Borderline cases are inconclusive. Native BF16 canonical margins "
                  "are saved separately and never determine this exclusion. The bound does not exclude every rounded "
                  "implementation. Passing the bound proves neither a shared learnable correction nor protection "
                  "feasibility, victory over every competing token or improved native verification.", ""]
    lines += ["## Evidence availability and unresolved explanations", ""]
    if inventory is None:
        lines += ["Artifact inventory was not supplied; availability is not inferred.", ""]
    elif missing:
        lines += _table([("stream", "Missing/unavailable case"), ("issues", "Reason")], missing)
        lines += ["Unavailable checkpoints, traces or projections were not recreated, substituted or replaced by extra cases.", ""]
    else:
        lines += ["No unavailable diagnostic case is listed in the supplied inventory. Individual invalid pairs and "
                  "missing capacity measurements, if present, remain separate exclusions above.", ""]
    unavailable_traces = [row for row in (inventory or {}).get("trace_inventory", [])
                          if row.get("status") != "available"]
    if unavailable_traces:
        lines += [f"The inventory records {len(unavailable_traces)} unavailable original timing trajectories. "
                  "Their identities and reasons remain in artifact_trace_inventory.csv; missing token evidence "
                  "is not filled with a different repeat or a regenerated trajectory.", ""]
    lines += ["The observations cannot establish whether OSD's benefit requires full-backbone training, which internal "
              "features caused a repair, or whether memory, selection, fusion or protection explains every bank failure. "
              "The five final requests limit the diagnostic's coverage. Local repair frequency cannot apportion "
              "total saved serving time.", "", "## Supported design implications", ""]
    if overall_local["mathematically_excluded"]:
        lines += [f"{overall_local['mathematically_excluded']} of {overall_local['valid_repair_denominator']} valid "
                  "OSD-repaired anchor decisions are excluded by the original mathematical pairwise capacity bound. "
                  "Those decisions cannot be repaired merely by choosing a different feasible combined matrix in "
                  "that unchanged mathematical space. This does not establish that full-backbone training is necessary.", ""]
    elif overall_local["evaluated_repair_denominator"]:
        lines += ["The available bound evaluations exclude none of the measured OSD repairs. This screen therefore "
                  "does not identify insufficient pairwise capacity as their explanation; learnability, joint "
                  "feasibility, protection and selection remain unresolved.", ""]
    else:
        lines += ["No evaluated valid-repair capacity evidence is available, so no correction-space conclusion is justified.", ""]
    if overall_local["bound_missing_or_unavailable"]:
        lines += [f"{overall_local['bound_missing_or_unavailable']} valid repaired decisions lack usable bound evidence; "
                  "they remain in the repair denominator and cannot be classified.", ""]
    lines += ["Preserve useful native gains and regressions when considering a replacement. These measurements "
              "support attribution only; no revised mechanism, training, tuning or broader campaign is performed.", ""]
    (out/"REPORT.md").write_text("\n".join(lines))
    return payload
