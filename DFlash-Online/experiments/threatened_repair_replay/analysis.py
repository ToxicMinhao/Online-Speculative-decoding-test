"""Scalar analysis for the fixed threatened-repair replay experiment.

Requests, rather than blocks or windows, are the bootstrap units. Every sealed
window stays in the primary estimate, including exact native-fallback windows.
This module has no access to model state, training or selection randomness.
"""
from collections import Counter, defaultdict
import csv
from fractions import Fraction
import json
import math
from pathlib import Path

import numpy as np


WINDOWS = tuple(range(8, 129, 8))
STATES = ("P", "N", "U", "T")
PROBE_KEY = ("request_id", "canonical_start", "round_ordinal")
TAG_KEY = ("request_id", "canonical_start", "proposal_slot")
CONTRASTS = (("T", "N"), ("T", "U"))


def key(row, fields):
    return tuple(row[name] for name in fields)


def _unique(rows, fields):
    result = {key(row, fields): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate scalar identity: {fields}")
    return result


def _prefix(row):
    length, valid = row["accepted_prefix"], row["valid_proposals"]
    if not row["valid"]:
        raise ValueError("Invalid observation cannot enter a paired estimate")
    if type(length) is not int or type(valid) is not int or not 0 <= length <= valid <= 15:
        raise ValueError("Accepted prefix or valid-proposal count is out of range")
    expected = length + 1 if length < valid else None
    if row["first_error"] != expected:
        raise ValueError("First error disagrees with clipped accepted prefix")
    return length


def tradeoff(prefixes, valid):
    """Exact integer decomposition; a full/terminal prefix has no blocker."""
    p = prefixes["P"]
    row = {"pre_prefix": p, "valid_proposals": valid, "pre_has_blocker": int(p < valid)}
    for state in ("N", "U", "T"):
        a = prefixes[state]
        row.update({f"{state}_acquired": max(a - p, 0),
                    f"{state}_lost": max(p - a, 0),
                    f"{state}_net": a - p,
                    f"{state}_earlier_error": int(a < p),
                    f"{state}_repairs_pre_blocker": int(p < valid and a > p)})
    for state in ("U", "T"):
        row[f"{state}_displaced_native_blocker_repair"] = int(
            p < valid and prefixes["N"] > p and prefixes[state] <= p)
    for candidate, control in CONTRASTS:
        name = f"{candidate}_{control}"
        row[f"{name}_acquisition_difference"] = row[f"{candidate}_acquired"] - row[f"{control}_acquired"]
        row[f"{name}_loss_difference"] = row[f"{candidate}_lost"] - row[f"{control}_lost"]
        row[f"{name}_net"] = prefixes[candidate] - prefixes[control]
        if row[f"{name}_net"] != row[f"{name}_acquisition_difference"] - row[f"{name}_loss_difference"]:
            raise ValueError("Integer acquisition/loss identity failed")
    return row


def _fraction(value):
    return {"numerator": value.numerator, "denominator": value.denominator}


def _mean(rows, field):
    return sum((Fraction(row[field]) for row in rows), Fraction()) / len(rows)


def bootstrap_requests(request_rows, *, draws=10000, seed=1707):
    """Both contrasts use the same paired whole-request resamples."""
    rows = sorted(request_rows, key=lambda row: row["request_id"])
    rng = np.random.Generator(np.random.PCG64(seed))
    indices = rng.integers(0, len(rows), size=(draws, len(rows)))
    values = np.asarray([[row["T_N"], row["T_U"]] for row in rows], dtype=np.float64)
    samples = values[indices].mean(axis=1)
    return {
        "draws": draws, "seed": seed, "bit_generator": "PCG64",
        "request_order": [row["request_id"] for row in rows],
        "method": "paired whole requests; shared draws; linear 2.5/97.5 percentiles",
        "scope": "conditional on one native trajectory, fixed windows/control and reused development requests",
        "intervals": {name: {"lower": float(np.quantile(samples[:, col], .025, method="linear")),
                               "upper": float(np.quantile(samples[:, col], .975, method="linear"))}
                      for col, name in enumerate(("T_N", "T_U"))},
    }


def reduce_probes(probe_rows, sealed_probes, *, windows=WINDOWS, draws=10000, seed=1707):
    """Require the complete sealed conceptual grid, including explicitly reused rows."""
    sealed = _unique(sealed_probes, PROBE_KEY)
    actual = _unique(probe_rows, ("window", *PROBE_KEY, "state"))
    expected = {(window, *probe, state) for window in windows for probe in sealed for state in STATES}
    if not sealed or set(actual) != expected:
        raise ValueError(f"Incomplete/different probe grid: expected {len(expected)}, observed {len(actual)}")
    requests = sorted({probe[0] for probe in sealed})
    grouped, blocks = defaultdict(list), []
    for window in windows:
        for probe in sorted(sealed):
            observations = {state: actual[(window, *probe, state)] for state in STATES}
            prefixes = {state: _prefix(row) for state, row in observations.items()}
            valid = observations["P"]["valid_proposals"]
            if any(row["valid_proposals"] != valid for row in observations.values()):
                raise ValueError("Branches use different canonical termination boundaries")
            if "valid_proposals" in sealed[probe] and sealed[probe]["valid_proposals"] != valid:
                raise ValueError("Probe boundary differs from the sealed input")
            for state, row in observations.items():
                origin = row.get("reused_from")
                if origin and (origin not in STATES or prefixes[state] != prefixes[origin]):
                    raise ValueError("Reused probe result disagrees with its source state")
            row = dict(zip(PROBE_KEY, probe), window=window, **tradeoff(prefixes, valid))
            row.update({f"a_{state}": value for state, value in prefixes.items()})
            blocks.append(row)
            grouped[(probe[0], window)].append(row)
    fields = (tuple(f"a_{state}" for state in STATES) +
              tuple(f"{state}_{metric}" for state in ("N", "U", "T")
                    for metric in ("acquired", "lost", "net", "earlier_error", "repairs_pre_blocker")) +
              tuple(f"{state}_displaced_native_blocker_repair" for state in ("U", "T")) +
              tuple(f"{a}_{b}_{metric}" for a, b in CONTRASTS
                    for metric in ("acquisition_difference", "loss_difference", "net")))
    cells = {(rid, window): {field: _mean(grouped[(rid, window)], field) for field in fields}
             for rid in requests for window in windows}
    request_rows = []
    for rid in requests:
        row = {"request_id": rid, "starts": len(grouped[(rid, windows[0])]), "windows": len(windows)}
        row.update({field: float(sum(cells[(rid, window)][field] for window in windows) / len(windows))
                    for field in fields})
        row.update(T_N=row["T_N_net"], T_U=row["T_U_net"])
        request_rows.append(row)
    window_rows = []
    for window in windows:
        row = {"window": window, "requests": len(requests), "starts": len(sealed)}
        row.update({field: float(sum(cells[(rid, window)][field] for rid in requests) / len(requests))
                    for field in fields})
        row.update(T_N=row["T_N_net"], T_U=row["T_U_net"])
        window_rows.append(row)
    means = {field: sum(cell[field] for cell in cells.values()) / len(cells) for field in fields}
    primary = {"unit": "accepted draft proposals per fixed block", "request_count": len(requests),
               "window_count": len(windows), "sealed_starts": len(sealed),
               "conceptual_observations": len(probe_rows), "paired_block_windows": len(blocks),
               "levels": {state: float(means[f"a_{state}"]) for state in STATES},
               "contrasts": {f"{a}_{b}": {"estimate": float(means[f"{a}_{b}_net"]),
                                             "exact": _fraction(means[f"{a}_{b}_net"])} for a, b in CONTRASTS},
               "decomposition": {name: float(value) for name, value in means.items() if not name.startswith("a_")},
               "gross_counts": {field: sum(row[field] for row in blocks) for field in fields if not field.startswith("a_")},
               "all_integer_decompositions_pass": True,
               "reduction": "mean starts within request/window, all fixed windows within request, equal requests"}
    boot = bootstrap_requests(request_rows, draws=draws, seed=seed)
    for name, interval in boot["intervals"].items():
        primary["contrasts"][name].update(interval)
    primary["bootstrap"] = boot
    return primary, request_rows, window_rows, blocks


def summarize_selection(rows, *, windows=WINDOWS, expected_tags=None):
    """Validate the funnel and same-position control; never choose replay examples."""
    records = _unique(rows, ("window", *TAG_KEY))
    identities = {key(row, TAG_KEY) for row in rows}
    if expected_tags is not None and identities != {key(row, TAG_KEY) for row in expected_tags}:
        raise ValueError("Selection population differs from sealed memory tags")
    if set(records) != {(w, *tag) for w in windows for tag in identities}:
        raise ValueError("Selection funnel is missing a fixed window/tag")
    selected, threats, summaries = set(), set(), []
    for window in windows:
        group = [records[(window, *tag)] for tag in sorted(identities)]
        u, t = set(), set()
        counts = Counter()
        for cohort in ("all", "excluded", "eligible", "already_wrong"):
            counts[f"{cohort}_denominator"] = 0
            for state in ("P", "N"):
                for metric in ("direct_correct", "ties", "wrong_ties"):
                    counts[f"{cohort}_{state}_{metric}"] = 0
        for row in group:
            tag, j = key(row, TAG_KEY), row["proposal_slot"]
            p, n = row["a_P"], row["a_N"]
            if not 1 <= j <= 15 or not 0 <= p <= 15 or not 0 <= n <= 15:
                raise ValueError("Selection prefix/slot out of range")
            excluded = bool(row["excluded_current"])
            eligible = not excluded and p >= j
            threatened = eligible and n == j - 1
            if bool(row["currently_accepted"]) != eligible or bool(row["threat"]) != threatened:
                raise ValueError("Selection flags contradict the finite native transition")
            counts["all_tags"] += 1
            counts["current_response_exclusions"] += excluded
            counts["currently_accepted"] += eligible
            counts["already_incorrect"] += not excluded and p < j
            counts["threats"] += threatened
            counts["earlier_breaks"] += eligible and n < j - 1
            counts["retained_eligible"] += eligible and n >= j
            cohorts = ["all", "excluded" if excluded else "eligible" if eligible else "already_wrong"]
            for cohort in cohorts:
                counts[f"{cohort}_denominator"] += 1
            for state, prefix in (("P", p), ("N", n)):
                correct, margin = bool(row[f"{state}_direct_correct"]), row[f"{state}_margin"]
                wrong_tie = bool(row[f"{state}_wrong_tie"])
                if (not math.isfinite(margin) or (margin > 0 and not correct) or
                    (margin < 0 and correct) or (prefix >= j and not correct) or
                    (prefix == j - 1 and correct) or wrong_tie != (margin == 0 and not correct)):
                    raise ValueError("Selection direct correctness, margin or tie metadata is inconsistent")
                for cohort in cohorts:
                    counts[f"{cohort}_{state}_direct_correct"] += correct
                    counts[f"{cohort}_{state}_ties"] += margin == 0
                    counts[f"{cohort}_{state}_wrong_ties"] += wrong_tie
            earlier = eligible and n < j - 1
            counts["earlier_break_j_correct"] += earlier and bool(row["N_direct_correct"])
            counts["earlier_break_j_wrong"] += earlier and not bool(row["N_direct_correct"])
            if threatened:
                threats.add(tag)
            for state, destination in (("U", u), ("T", t)):
                if row[f"selected_{state}"]:
                    if not eligible or state == "T" and not threatened:
                        raise ValueError("A selected replay row is not in its permitted population")
                    destination.add(tag)
        if len(u) != len(t) or len(t) != min(4, counts["threats"]):
            raise ValueError("Replay branch counts do not obey the fixed threat budget")
        if Counter(tag[2] for tag in u) != Counter(tag[2] for tag in t):
            raise ValueError("UNIFORM/THREAT positional weights do not match")
        if counts["earlier_break_j_correct"] + counts["earlier_break_j_wrong"] != counts["earlier_breaks"]:
            raise ValueError("Earlier-break direct-correctness partition failed")
        selected.update(t)
        concentrations = {state: Counter(tag[0] for tag in chosen) for state, chosen in (("U", u), ("T", t))}
        summaries.append(dict(window=window, **counts, selected_count=len(t),
            active=bool(t), identical_sets=u == t, overlap=len(u & t),
            selected_U_requests=len(concentrations["U"]), selected_T_requests=len(concentrations["T"]),
            selected_U_max_tags_per_request=max(concentrations["U"].values(), default=0),
            selected_T_max_tags_per_request=max(concentrations["T"].values(), default=0)))
    result = {"distinct_selected_T_tags": len(selected),
              "distinct_selected_T_requests": len({tag[0] for tag in selected}),
              "active_windows": sum(row["active"] for row in summaries),
              "nonidentical_selection_windows": sum(not row["identical_sets"] for row in summaries),
              "all_threat_distinct_tags": len(threats), "all_threat_distinct_requests": len({tag[0] for tag in threats}),
              "all_threat_occurrences": sum(row["threats"] for row in summaries),
              "selected_T_occurrences": sum(row["selected_count"] for row in summaries),
              "per_window": summaries}
    result["opportunity_placement_pass"] = (result["distinct_selected_T_tags"] >= 16 and
        result["distinct_selected_T_requests"] >= 8 and result["active_windows"] >= 4 and
        result["nonidentical_selection_windows"] >= 4)
    return result


def summarize_memory(rows, selection_rows):
    """Use one selected union row per tag/window/state, and one shared T set."""
    chosen = {(r["window"], *key(r, TAG_KEY)): r for r in selection_rows if r["selected_U"] or r["selected_T"]}
    records = _unique(rows, ("window", *TAG_KEY, "state"))
    if set(records) != {(*tag, state) for tag in chosen for state in STATES}:
        raise ValueError("Selected-memory union grid is incomplete/different")
    groups = {"selected_T": [], "union": []}
    for identity, selection in sorted(chosen.items()):
        states = {state: records[(*identity, state)] for state in STATES}
        if len({row["valid_proposals"] for row in states.values()}) != 1:
            raise ValueError("Selected-memory branches use different canonical boundaries")
        j = identity[-1]
        reduced = {"window": identity[0], **dict(zip(TAG_KEY, identity[1:]))}
        for state, row in states.items():
            a, margin, correct = _prefix(row), row["margin"], bool(row["direct_correct"])
            if j > row["valid_proposals"]:
                raise ValueError("Selected-memory tag is outside the valid suffix")
            if not math.isfinite(margin) or (margin > 0 and not correct) or (margin < 0 and correct):
                raise ValueError("Native margin/correctness is invalid")
            if (a >= j and not correct) or (a == j - 1 and correct):
                raise ValueError("Selected-memory direct correctness contradicts its prefix")
            reduced.update({f"{state}_accepts": int(a >= j), f"{state}_exact_rejection": int(a == j - 1),
                            f"{state}_earlier_break": int(a < j - 1), f"{state}_direct_correct": int(correct),
                            f"{state}_margin": margin, f"{state}_wrong_tie": int(margin == 0 and not correct)})
        for control in ("N", "U"):
            reduced[f"T_{control}_rescue"] = int(reduced["T_accepts"] and not reduced[f"{control}_accepts"])
            reduced[f"T_{control}_loss"] = int(reduced[f"{control}_accepts"] and not reduced["T_accepts"])
        if states["P"]["accepted_prefix"] != selection["a_P"] or states["N"]["accepted_prefix"] != selection["a_N"]:
            raise ValueError("Selected-memory P/N outcomes differ from selection observations")
        groups["union"].append(reduced)
        if selection["selected_T"]:
            if states["P"]["accepted_prefix"] < j or states["N"]["accepted_prefix"] != j - 1:
                raise ValueError("Threat-memory observations do not reproduce sealed selection")
            groups["selected_T"].append(reduced)
    output = {}
    for name, group in groups.items():
        stats = {"tag_window_count": len(group), "distinct_tags": len({key(r, TAG_KEY) for r in group})}
        for state in STATES:
            for metric in ("accepts", "exact_rejection", "earlier_break", "direct_correct", "wrong_tie"):
                stats[f"{state}_{metric}"] = sum(row[f"{state}_{metric}"] for row in group)
            stats[f"{state}_mean_margin"] = math.fsum(row[f"{state}_margin"] for row in group) / len(group) if group else None
        for control in ("N", "U"):
            for metric in ("rescue", "loss"):
                stats[f"T_{control}_{metric}"] = sum(row[f"T_{control}_{metric}"] for row in group)
        output[name] = stats
    output["manipulation_pass"] = output["selected_T"]["T_N_rescue"] >= 1
    return output


def decide(primary, support, memory, validity):
    if not validity.get("complete", False):
        return "INCOMPLETE"
    if not validity.get("all_required_checks_passed", False):
        return "INVALID"
    if not support["opportunity_placement_pass"]:
        return "INSUFFICIENT_OPPORTUNITY_OR_PLACEMENT"
    if not memory["manipulation_pass"]:
        return "NO_DEMONSTRATED_TARGETED_CONTROL"
    contrasts = primary["contrasts"]
    if all(contrasts[name]["lower"] > 0 for name in ("T_N", "T_U")):
        return "LOCAL_ADVANCEMENT_SCREEN_PASSED"
    if contrasts["T_N"]["upper"] < 0:
        return "INDEPENDENT_HARM"
    if contrasts["T_U"]["upper"] < 0:
        return "THREAT_SELECTION_WORSE_THAN_UNIFORM"
    if contrasts["T_N"]["lower"] > 0:
        return "REPLAY_BENEFIT_WITHOUT_DEMONSTRATED_TARGETING_ADVANTAGE"
    return "UNRESOLVED_LOCAL_TRANSFER"


def _production_population(sealed_probes, expected_tags, windows):
    requests = Counter(row["request_id"] for row in sealed_probes)
    if tuple(windows) != WINDOWS or len(requests) != 64 or set(requests.values()) != {4}:
        raise ValueError("Production plan requires 64 requests, four sealed starts each and all 16 fixed windows")
    if expected_tags is None or len(_unique(expected_tags, TAG_KEY)) != 815 or len({row["request_id"] for row in expected_tags}) != 128:
        raise ValueError("Production plan requires 815 sealed memory tags across 128 requests")


def analyze(probe_rows, selection_rows, memory_rows, *, sealed_probes, windows=WINDOWS,
            validity, work, expected_tags=None, draws=10000, seed=1707, require_full_population=True):
    if require_full_population:
        _production_population(sealed_probes, expected_tags, windows)
    primary, requests, per_window, tradeoffs = reduce_probes(
        probe_rows, sealed_probes, windows=windows, draws=draws, seed=seed)
    support = summarize_selection(selection_rows, windows=windows, expected_tags=expected_tags)
    memory = summarize_memory(memory_rows, selection_rows)
    active = {row["window"] for row in support["per_window"] if row["active"]}
    indexed = {(row["window"], *key(row, PROBE_KEY), row["state"]): row for row in probe_rows}
    for row in probe_rows:
        if row["window"] not in active and row["state"] in ("U", "T"):
            native = indexed[(row["window"], *key(row, PROBE_KEY), "N")]
            if row["accepted_prefix"] != native["accepted_prefix"]:
                raise ValueError("A no-threat window must be exact native fallback")
    secondary = {"active_window_count": len(active), "active_window_only": True,
        "contrasts": {name: math.fsum(row[name] for row in per_window if row["window"] in active) / len(active)
                      if active else None for name in ("T_N", "T_U")}}
    summary = {"status": "completed", "primary": primary, "support": support, "memory": memory,
               "active_window_secondary": secondary, "validity": validity, "work": work}
    summary["decision"] = decide(primary, support, memory, validity)
    return summary, requests, per_window, tradeoffs


PROBE_PUBLIC = ("window", *PROBE_KEY, "state", "accepted_prefix", "valid_proposals", "first_error", "valid", "reused_from", "execution_id")
SELECTION_PUBLIC = ("window", *TAG_KEY, "a_P", "a_N", "excluded_current", "currently_accepted", "threat", "selected_U", "selected_T",
                    "P_direct_correct", "N_direct_correct", "P_margin", "N_margin", "P_wrong_tie", "N_wrong_tie")
MEMORY_PUBLIC = ("window", *TAG_KEY, "state", "accepted_prefix", "valid_proposals", "first_error", "direct_correct", "margin", "valid", "reused_from", "execution_id")


def _csv(path, rows, fields):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_report(output, *, probe_rows, selection_rows, memory_rows, sealed_probes,
                 validity, work, execution_commit, windows=WINDOWS, expected_tags=None,
                 require_full_population=True):
    """Publish available scalar evidence even if execution stopped incomplete."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    _csv(output / "PROBE_OUTCOMES.csv", probe_rows, PROBE_PUBLIC)
    _csv(output / "WINDOW_SELECTION.csv", selection_rows, SELECTION_PUBLIC)
    _csv(output / "MEMORY_OUTCOMES.csv", memory_rows, MEMORY_PUBLIC)
    summary = {"status": "incomplete", "decision": "INCOMPLETE", "validity": validity,
               "work": work, "execution_commit": execution_commit,
               "observed_probe_rows": len(probe_rows), "planned_probe_rows": len(sealed_probes) * len(windows) * 4}
    if validity.get("complete") and validity.get("all_required_checks_passed"):
        try:
            summary, requests, per_window, tradeoffs = analyze(
                probe_rows, selection_rows, memory_rows, sealed_probes=sealed_probes,
                windows=windows, validity=validity, work=work, expected_tags=expected_tags,
                require_full_population=require_full_population)
        except (ValueError, KeyError) as error:
            summary.update(status="invalid", decision="INVALID", analysis_error=str(error))
        else:
            for filename, rows in (("PER_REQUEST.csv", requests), ("PER_WINDOW.csv", per_window),
                                   ("REPAIR_TRADEOFF.csv", tradeoffs)):
                _csv(output / filename, rows, tuple(rows[0]))
    elif validity.get("complete"):
        summary.update(status="invalid", decision="INVALID")
    summary["execution_commit"] = execution_commit
    (output / "PRIMARY_RESULTS.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    text = ["# Threatened-repair replay", "", "## Material Passport", "",
            f"- Execution commit: `{execution_commit}`.", f"- Status: **{summary['status']}**.",
            "- Scope: one continuing native trajectory; disposable local U/T branches; no serving-speed measurement.",
            "- Reused development probes; statistical intervals describe request variation conditional on this trajectory.",
            "", f"**Decision: {summary['decision']}.**", ""]
    if summary["status"] == "completed":
        primary = summary["primary"]
        text += ["| Contrast | Accepted proposals per fixed block | Paired 95% interval |", "|---|---:|---:|"]
        for name, contrast in primary["contrasts"].items():
            text.append(f"| {name} | {contrast['estimate']:+.6f} | [{contrast['lower']:+.6f}, {contrast['upper']:+.6f}] |")
        text += ["", "All sealed windows, including no-threat zero-contrast windows, enter the primary reduction.",
                 "Selected-memory rescues are manipulation checks, not transfer evidence. The acquisition/loss identity is checked on every integer block outcome.",
                 "Intervals spanning zero are unresolved, not equivalence. Passing both contrasts only supports considering a separately designed rollout."]
    else:
        text += ["No completed-panel efficacy estimate is reported. Available rows and physical work remain preserved; missing work is not imputed.",
                 str(summary.get("analysis_error", validity.get("errors", [])))]
    text += ["", "No further windows, tuning, seeds, generations or rollout are authorized by this report.", ""]
    (output / "REPORT.md").write_text("\n".join(text))
    return summary
