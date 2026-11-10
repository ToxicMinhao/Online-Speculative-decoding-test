"""CPU-only first-transition decomposition on the sealed request sample.

Native inputs use the retained serving record format. Crossed cells contain a
status, verification_calls, output_matches_canonical, and termination_matches.
Public rows contain counts and flags, never prompts or token arrays. An observed
no-divergence request has NOT_APPLICABLE cells and zero transition effects;
unfinished work has missing effects. Neither is silently removed from the panel.
"""
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np


DATASETS = ("gsm8k", "mathqa")
PANEL_SIZE = 64
RESAMPLES = 10_000
BOOTSTRAP_SEED = 20261110
BOUNDARY_TOKENS = 16
CELLS = ("C44", "C46", "C64", "C66")
EFFECTS = ("D", "R4", "R6", "M4", "M6")


def _count(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _walk(record):
    """Check actual canonical commits; output_start excludes the initial seed."""
    canonical = record["response_ids"]
    trace = record["metadata"]["trace"]
    if not canonical or record["finish_reason"] not in ("eos", "length"):
        raise ValueError("A native walk requires its original canonical termination")
    offset = 0
    for index, row in enumerate(trace):
        progress = _count(row["committed_progress"], "committed_progress")
        if not 1 <= progress <= 16 or row["output_start"] != offset:
            raise ValueError("Native trace has invalid progress or a discontinuous prefix")
        end = offset + progress
        if row["anchor_token"] != canonical[offset]:
            raise ValueError("Native seed disagrees with the canonical prefix")
        if row["committed_tokens"] != canonical[offset + 1:end + 1]:
            raise ValueError("Native committed tokens disagree with the canonical answer")
        if end >= len(canonical) or row["finished"] != (index == len(trace) - 1):
            raise ValueError("Native terminal boundary or progress exceeds the saved answer")
        offset = end
    if offset != len(canonical) - 1:
        raise ValueError("Native walk is incomplete")
    if record["metadata"]["verification_calls"] != len(trace):
        raise ValueError("Native verification count disagrees with its trajectory")
    return trace


def _repair_check(row4, row6, canonical):
    """Check the first *valid* KPLUS4 rejection, not a later isolated match."""
    start = row4["output_start"] + 1
    available = canonical[start:start + 15]
    proposals4 = row4["draft_tokens"]
    proposals6 = row6["draft_tokens"]
    targets4 = row4["target_argmax"]
    rejection = next((i for i, token in enumerate(proposals4[:len(available)])
                      if token != targets4[i]), None)
    if rejection is None:
        return dict(first_rejected_proposal_position=None,
                    first_rejected_absolute_output_position=None,
                    preceding_matches_preserved=None,
                    first_rejected_position_repaired=None,
                    repair_check="no_valid_first_rejection")
    preceding4 = proposals4[:rejection] == available[:rejection]
    if not preceding4 or targets4[rejection] != available[rejection]:
        raise ValueError("First native rejection lacks its canonical preceding prefix")
    preceding6 = proposals6[:rejection] == available[:rejection]
    repaired = preceding6 and proposals6[rejection] == available[rejection]
    return dict(first_rejected_proposal_position=rejection + 1,
                first_rejected_absolute_output_position=start + rejection,
                preceding_matches_preserved=preceding6,
                first_rejected_position_repaired=repaired,
                repair_check="checked")


def find_transition(native4, native6):
    """Locate first actual progress divergence; earlier proposal differences remain visible."""
    if (native4["response_ids"] != native6["response_ids"]
            or native4["finish_reason"] != native6["finish_reason"]
            or native4["prompt_ids"] != native6["prompt_ids"]
            or native4["max_new_tokens"] != native6["max_new_tokens"]):
        raise ValueError("Native walks must share exact request, output and stopping policy")
    trace4, trace6 = _walk(native4), _walk(native6)
    canonical = native4["response_ids"]
    previous_differences = []
    common = dict(total_calls4=len(trace4), total_calls6=len(trace6),
                  endpoint_call_difference=len(trace6) - len(trace4),
                  output_tokens=len(canonical), finish_reason=native4["finish_reason"],
                  prior_proposal_difference_rounds=previous_differences)
    for index, (row4, row6) in enumerate(zip(trace4, trace6)):
        if row4["output_start"] != row6["output_start"]:
            raise ValueError("Earlier progress did not reach the same canonical prefix")
        progress4, progress6 = row4["committed_progress"], row6["committed_progress"]
        if progress4 != progress6:
            end4 = row4["output_start"] + progress4 + 1
            end6 = row6["output_start"] + progress6 + 1
            remaining4, remaining6 = len(canonical) - end4, len(canonical) - end6
            return dict(common, transition_kind="advance" if progress6 > progress4 else "regression",
                        divergence_round=index + 1, common_output_start=row4["output_start"],
                        progress4=progress4, progress6=progress6,
                        successor_output_tokens4=end4, successor_output_tokens6=end6,
                        remaining_tokens4=remaining4, remaining_tokens6=remaining6,
                        terminal4=remaining4 == 0, terminal6=remaining6 == 0,
                        boundary=remaining4 <= BOUNDARY_TOKENS or remaining6 <= BOUNDARY_TOKENS,
                        interior=remaining4 > BOUNDARY_TOKENS and remaining6 > BOUNDARY_TOKENS,
                        C44=len(trace4) - index - 1, C66=len(trace6) - index - 1,
                        selected_round_proposals_differ=row4["draft_tokens"] != row6["draft_tokens"],
                        **_repair_check(row4, row6, canonical))
        if row4["draft_tokens"] != row6["draft_tokens"]:
            previous_differences.append(index + 1)
    if len(trace4) != len(trace6):
        raise ValueError("Equal progress through termination cannot leave an extra native round")
    return dict(common, transition_kind="none", divergence_round=None,
                common_output_start=None, progress4=None, progress6=None,
                successor_output_tokens4=None, successor_output_tokens6=None,
                remaining_tokens4=None, remaining_tokens6=None, terminal4=None, terminal6=None,
                boundary=None, interior=None, C44=None, C66=None,
                selected_round_proposals_differ=None, first_rejected_proposal_position=None,
                first_rejected_absolute_output_position=None, preceding_matches_preserved=None,
                first_rejected_position_repaired=None, repair_check="not_applicable")


def _crossed_cell(cell, *, terminal):
    if terminal:
        if cell is not None and cell.get("verification_calls") not in (None, 0):
            raise ValueError("A terminal successor must have zero remaining verification calls")
        return 0, "terminal_zero"
    if cell is None:
        return None, "unattempted"
    status = cell["status"]
    if status != "completed":
        return None, status
    if not (cell["output_matches_canonical"] and cell["termination_matches"]):
        return None, "invalid"
    return _count(cell["verification_calls"], "crossed verification_calls"), "completed"


def request_row(dataset, request_id, selection_rank, native4=None, native6=None, *, crossed=None):
    """Produce one public-safe row, including selected but unattempted requests."""
    if dataset not in DATASETS or not 0 <= selection_rank < PANEL_SIZE:
        raise ValueError("Unknown dataset or selection rank")
    base = dict(dataset=dataset, request_id=request_id, selection_rank=selection_rank)
    if native4 is None or native6 is None:
        return dict(base, status="incomplete", transition_kind="unknown",
                    endpoint_call_difference=None, boundary=None, interior=None,
                    **{name: None for name in (*CELLS, *EFFECTS)})
    transition = find_transition(native4, native6)
    row = dict(base, **transition)
    row["prior_proposal_difference_count"] = len(transition["prior_proposal_difference_rounds"])
    if transition["transition_kind"] == "none":
        row.update(status="completed", **{name: None for name in CELLS},
                   **{name + "_status": "not_applicable" for name in CELLS},
                   **{name: 0 for name in EFFECTS}, both_restart_positive=False)
        return row
    crossed = crossed or {}
    for name, terminal in (("C46", row["terminal6"]), ("C64", row["terminal4"])):
        row[name], row[name + "_status"] = _crossed_cell(crossed.get(name), terminal=terminal)
    row["C44_status"] = "terminal_zero" if row["terminal4"] else "authenticated_native"
    row["C66_status"] = "terminal_zero" if row["terminal6"] else "authenticated_native"
    if any(row[name] is None for name in CELLS):
        row.update(status="incomplete", **{name: None for name in EFFECTS}, both_restart_positive=None)
        return row
    c44, c46, c64, c66 = (row[name] for name in CELLS)
    row.update(status="completed", D=c66 - c44, R4=c46 - c44,
               R6=c66 - c64, M4=c64 - c44, M6=c66 - c46)
    if not (row["D"] == row["endpoint_call_difference"]
            == row["R4"] + row["M6"] == row["M4"] + row["R6"]):
        raise ValueError("Four-cell decomposition does not reconcile with endpoint counts")
    row["both_restart_positive"] = row["R4"] > 0 and row["R6"] > 0
    return row


def _spread(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return dict(n=0, mean=None, median=None, minimum=None, maximum=None,
                    q25=None, q75=None, sample_sd=None)
    return dict(n=len(values), mean=float(values.mean()), median=float(np.median(values)),
                minimum=float(values.min()), maximum=float(values.max()),
                q25=float(np.quantile(values, .25)), q75=float(np.quantile(values, .75)),
                sample_sd=float(values.std(ddof=1)) if len(values) > 1 else None,
                positive=int((values > 0).sum()), negative=int((values < 0).sum()),
                zero=int((values == 0).sum()))


def _concentration(values, selected, ids):
    contribution = np.where(selected, values, 0.)
    absolute = np.abs(contribution)
    positive = np.maximum(contribution, 0.)
    absolute_sum, positive_sum = float(absolute.sum()), float(positive.sum())
    largest = int(np.argmax(absolute))
    return dict(absolute_sum=absolute_sum, positive_sum=positive_sum,
                signed_sum=float(contribution.sum()),
                largest_absolute_request=ids[largest] if absolute_sum else None,
                largest_absolute_share=float(absolute.max() / absolute_sum) if absolute_sum else None,
                largest_positive_share=float(positive.max() / positive_sum) if positive_sum else None,
                nonzero_contributors=int(np.count_nonzero(contribution)))


def _interval(draws):
    finite = draws[np.isfinite(draws)]
    return np.quantile(finite, [.025, .975], method="linear").tolist() if len(finite) else None


def _effect_summary(values, selected, indices, ids):
    """Resample all original rows, recomputing each conditional denominator."""
    contribution = np.where(selected, values, 0.)
    sampled_count = selected[indices].sum(axis=1)
    sampled_total = contribution[indices].sum(axis=1)
    conditional = np.full(len(indices), np.nan)
    np.divide(sampled_total, sampled_count, out=conditional, where=sampled_count > 0)
    unconditional = sampled_total / len(values)
    count = int(selected.sum())
    total = float(contribution.sum())
    leave_one_out = []
    for index, request_id in enumerate(ids):
        denominator = count - int(selected[index])
        numerator = total - contribution[index]
        leave_one_out.append(dict(request_id=request_id,
                                  eligible_count=denominator,
                                  conditional_mean=float(numerator / denominator) if denominator else None,
                                  unconditional_mean=float(numerator / (len(values) - 1))))
    valid_loo = [row["conditional_mean"] for row in leave_one_out if row["conditional_mean"] is not None]
    # There is no scientifically approved minimum-n cutoff. One observation has
    # no empirical spread, so show its descriptive point, not a useful interval.
    useful_interval = count >= 2
    return dict(conditional=_spread(values[selected]), unconditional=_spread(contribution),
                conditional_ci95=_interval(conditional) if useful_interval else None,
                conditional_interval_status="reported_with_sample_size" if useful_interval else "descriptive_only_zero_or_one_eligible",
                unconditional_ci95=_interval(unconditional),
                empty_stratum_resamples=int((sampled_count == 0).sum()),
                nonempty_stratum_resamples=int((sampled_count > 0).sum()),
                bootstrap_eligible_count=_spread(sampled_count),
                concentration=_concentration(values, selected, ids),
                leave_one_out=leave_one_out,
                loo_all_conditional_means_positive=bool(valid_loo) and len(valid_loo) == len(ids) and min(valid_loo) > 0,
                loo_conditional_signs=dict(positive=sum(x > 0 for x in valid_loo),
                                           negative=sum(x < 0 for x in valid_loo),
                                           zero=sum(x == 0 for x in valid_loo),
                                           empty=len(ids) - len(valid_loo)))


def summarize(rows, *, dataset):
    """Analyze exactly64 sealed rows from one dataset; incomplete panels stay incomplete."""
    if dataset not in DATASETS or len(rows) != PANEL_SIZE:
        raise ValueError("Analysis requires all64 selected rows of one known dataset")
    rows = sorted(rows, key=lambda row: row["selection_rank"])
    if (any(row["dataset"] != dataset for row in rows)
            or [row["selection_rank"] for row in rows] != list(range(PANEL_SIZE))
            or len({row["request_id"] for row in rows}) != PANEL_SIZE):
        raise ValueError("Dataset, selection rank or request identity mismatch")
    complete = [row for row in rows if row["status"] == "completed"]
    known = [row for row in rows if row.get("endpoint_call_difference") is not None]
    result = dict(dataset=dataset, selected_requests=PANEL_SIZE,
                  complete_requests=len(complete), incomplete_requests=PANEL_SIZE - len(complete),
                  transitions=dict(Counter(row["transition_kind"] for row in rows)),
                  native_endpoint_known_requests=len(known),
                  native_endpoint_total_calls4=sum(row["total_calls4"] for row in known),
                  native_endpoint_total_calls6=sum(row["total_calls6"] for row in known),
                  endpoint_call_ties=sum(row["endpoint_call_difference"] == 0 for row in known),
                  endpoint_call_difference=_spread([row["endpoint_call_difference"] for row in known]),
                  bootstrap=dict(resamples=RESAMPLES, seed=BOOTSTRAP_SEED, generator="PCG64",
                                 method="linear_percentile", unit="full_selected_request_row",
                                 strata_recomputed_after_resampling=True),
                  sparsity_policy="No data-selected minimum-n rule. n<2 conditional intervals are descriptive-only; report all other n, empty draws and concentration without automatic support classification.",
                  automatic_followup_authorized=False)
    if len(complete) != PANEL_SIZE:
        return dict(result, status="incomplete", interpretation="TECHNICALLY_INCOMPLETE",
                    strata=None, inference="No bootstrap or go/no-go conclusion from selectively completed work.")
    for row in rows:
        if any(type(row[name]) is not int for name in EFFECTS):
            raise ValueError("Completed transition effects must be observed integer counts")
        if row["transition_kind"] == "none":
            if any(row[name] is not None for name in CELLS) or any(row[name] != 0 for name in EFFECTS):
                raise ValueError("No-divergence cells must be not applicable with zero transition effects")
        elif (any(type(row[name]) is not int or row[name] < 0 for name in CELLS)
              or row["D"] != row["C66"] - row["C44"]
              or row["R4"] != row["C46"] - row["C44"]
              or row["R6"] != row["C66"] - row["C64"]
              or row["M4"] != row["C64"] - row["C44"]
              or row["M6"] != row["C66"] - row["C46"]):
            raise ValueError("Completed four-cell row has missing or inconsistent effects")
        if row["D"] != row["endpoint_call_difference"]:
            raise ValueError("Transition and endpoint differences disagree")
    indices = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED)).integers(
        0, PANEL_SIZE, size=(RESAMPLES, PANEL_SIZE))
    ids = [row["request_id"] for row in rows]
    advance = np.array([row["transition_kind"] == "advance" for row in rows])
    interior = np.array([row["interior"] is True for row in rows])
    strata = {}
    for name, selected in (("all_advances", advance), ("interior_advances", advance & interior)):
        values = {effect: np.array([row[effect] for row in rows], dtype=float) for effect in ("R4", "R6")}
        both = selected & (values["R4"] > 0) & (values["R6"] > 0)
        count = int(selected.sum())
        stratum = dict(eligible_requests=count, both_restart_positive_count=int(both.sum()),
                       both_restart_positive_fraction=float(both.sum() / count) if count else None,
                       repair_verified_count=sum(row.get("first_rejected_position_repaired") is True for row, use in zip(rows, selected) if use),
                       repair_not_verified_count=sum(row.get("first_rejected_position_repaired") is not True for row, use in zip(rows, selected) if use))
        for effect in ("R4", "R6"):
            stratum[effect] = _effect_summary(values[effect], selected, indices, ids)
        stratum["support_facts"] = dict(
            repeated_both_positive=int(both.sum()) >= 2,
            both_conditional_ci_lower_above_zero=all(
                stratum[e]["conditional_ci95"] is not None and stratum[e]["conditional_ci95"][0] > 0 for e in ("R4", "R6")),
            both_loo_means_stay_positive=all(stratum[e]["loo_all_conditional_means_positive"] for e in ("R4", "R6")),
            descriptive_sample_size=count,
            automatic_supported_verdict=False)
        strata[name] = stratum
    return dict(result, status="completed", strata=strata,
                boundary_advances=int((advance & ~interior).sum()),
                inference="Descriptive conditional decomposition; assess sparsity and conflicting signs explicitly. No latency or unique-attribution claim.")


def write_tables(output, rows, summaries):
    """Write public-safe rows and machine-readable summaries without raw tokens."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (output / "PER_REQUEST.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})
    (output / "SUMMARY.json").write_text(json.dumps(summaries, indent=2, allow_nan=False) + "\n")


def render_report(summaries, *, execution_commit, work=None):
    """Describe support facts without converting a small stratum into an automatic go decision."""
    lines = ["# Verification-transition replay", "", "## Material Passport", "",
             "ARS experiment-agent; CPU analysis of retained native replay records.",
             f"Execution commit: `{execution_commit}`. No training is part of this diagnostic.", "",
             "Positive restart effects mean that KPLUS6's successor costs more remaining calls "
             "under the specified fixed checkpoint. An expensive farther restart requires an "
             "actual KPLUS6 advance; earlier regressions are retained separately.", ""]
    for dataset, summary in summaries.items():
        lines.extend([f"## {dataset}", "",
                      f"Selected requests: {summary['selected_requests']}; complete: "
                      f"{summary['complete_requests']}; incomplete: {summary['incomplete_requests']}.",
                      f"First transitions: {summary['transitions']}; endpoint call ties: "
                      f"{summary['endpoint_call_ties']} / {summary['native_endpoint_known_requests']} known pairs.", ""])
        if summary["status"] != "completed":
            lines.extend(["**TECHNICALLY INCOMPLETE.** Missing continuations remain missing. "
                          "No go/no-go conclusion is drawn from the selectively completed subset.", ""])
            continue
        lines.extend(["| Stratum | Advances | Both R positive | R4 mean [95% CI] | R6 mean [95% CI] |",
                      "|---|---:|---:|---|---|"])
        for name, stratum in summary["strata"].items():
            def value(effect):
                metric = stratum[effect]
                return f"{metric['conditional']['mean']} [{metric['conditional_ci95']}]"
            lines.append(f"| {name} | {stratum['eligible_requests']} | "
                         f"{stratum['both_restart_positive_count']} | {value('R4')} | {value('R6')} |")
        lines.extend(["", "Full-row paired bootstrap: 10,000 PCG64 draws, seed20261110; two-sided "
                      "linear-percentile intervals. Each draw resamples the original64 requests "
                      "and recomputes the eligible stratum. The complete tables preserve "
                      "unconditional contributions, spread, empty-stratum draws, contributor "
                      "shares and leave-one-request-out signs.", "",
                      "No automatic support label is assigned. Inspect the interior sample size, "
                      "repaired-prefix checks, repetition, both mean intervals and concentration. "
                      "Sparse, boundary-only, conflicting or unresolved evidence is inconclusive. "
                      "R4 positive but R6 nonpositive does not establish a weakness that KPLUS6 "
                      "still needs the proposed algorithm to fix.", ""])
    lines.extend(["## Interpretation limits", "",
                  "This is a conditional checkpoint-switch decomposition on reused development "
                  "requests, not a unique percentage attribution of learning gains. Restart and "
                  "checkpoint effects can interact. Untimed replay does not establish serving "
                  "speedup. Even supporting evidence would justify proposing a short controlled "
                  "training pilot, not implementing or launching it here. No automatic follow-up.", ""])
    if work is not None:
        lines.extend(["## Recorded work", "", "```json", json.dumps(work, indent=2, allow_nan=False), "```", ""])
    return "\n".join(lines)
