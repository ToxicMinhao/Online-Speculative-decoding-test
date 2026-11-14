"""Prespecified reductions for the native K+ correct-prefix loss screen.

The primary unit is a request, averaged over its fixed starts and all windows.
No model state, gradient, selection or training operation is performed here.
"""
from collections import Counter, defaultdict
import csv
from fractions import Fraction
import json
from pathlib import Path

import numpy as np

WINDOWS = tuple(range(8, 129, 8))
STATES = ("P", "N", "E", "C", "H")
CANDIDATES = STATES[1:]
PROBE_KEY = ("request_id", "canonical_start", "round_ordinal")
SOURCE_KEY = ("request_id", "canonical_start", "anchor_ordinal")
CONTRASTS = (("E", "N"), ("E", "H"), ("C", "H"), ("C", "N"), ("H", "P"))
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 1707


def key(row, fields):
    return tuple(row[field] for field in fields)


def _index(rows, fields):
    indexed = {key(row, fields): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"Duplicate observation identity: {fields}")
    return indexed


def _prefix(row):
    if row.get("valid") is not True:
        raise ValueError("Invalid or unattempted observation cannot enter a gain estimate")
    length, valid = row["accepted_prefix"], row["valid_proposals"]
    if type(length) is not int or type(valid) is not int or not 0 <= length <= valid <= 15:
        raise ValueError("Accepted prefix and scored suffix must be integer counts within 0..15")
    if row["first_error"] != (length + 1 if length < valid else None):
        raise ValueError("First-error offset disagrees with accepted-prefix length")
    return length


def tradeoff(prefixes, valid):
    """Count prefix positions separately from block-level useful repairs."""
    before = prefixes["P"]
    row = {"pre_prefix": before, "valid_proposals": valid,
           "pre_has_blocker": int(before < valid)}
    for state in CANDIDATES:
        after = prefixes[state]
        row.update({f"{state}_acquired": max(after - before, 0),
                    f"{state}_lost": max(before - after, 0),
                    f"{state}_net": after - before,
                    f"{state}_earlier_error": int(after < before),
                    f"{state}_repairs_pre_blocker": int(before < valid and after > before)})
    for state in ("E", "C", "H"):
        row[f"{state}_displaced_native_blocker_repair"] = int(
            before < valid and prefixes["N"] > before and prefixes[state] <= before)
    for candidate, control in CONTRASTS:
        label = f"{candidate}_{control}"
        acquisition = row.get(f"{candidate}_acquired", 0) - row.get(f"{control}_acquired", 0)
        loss = row.get(f"{candidate}_lost", 0) - row.get(f"{control}_lost", 0)
        row.update({f"{label}_acquisition_difference": acquisition,
                    f"{label}_loss_difference": loss,
                    f"{label}_net": prefixes[candidate] - prefixes[control]})
        if row[f"{label}_net"] != acquisition - loss:
            raise ValueError("Acquisition/loss identity failed")
    row.update({f"a_{state}": count for state, count in prefixes.items()})
    return row


def _paired_block(observations, identity):
    prefixes = {state: _prefix(observations[state]) for state in STATES}
    boundary = observations["P"]["valid_proposals"]
    if any(observations[state]["valid_proposals"] != boundary for state in STATES):
        raise ValueError("Compared states do not share the canonical scoring boundary")
    for state, row in observations.items():
        origin = row.get("reused_from")
        # External historical reuse is authenticated upstream; same-grid aliases
        # additionally have a directly checkable source outcome here.
        if origin in STATES and prefixes[state] != prefixes[origin]:
            raise ValueError("Aliased outcome differs from its named source state")
    return {**identity, **tradeoff(prefixes, boundary)}


def _metric_fields():
    return (tuple(f"a_{state}" for state in STATES) +
            tuple(f"{state}_{metric}" for state in CANDIDATES
                  for metric in ("acquired", "lost", "net", "earlier_error", "repairs_pre_blocker")) +
            tuple(f"{state}_displaced_native_blocker_repair" for state in ("E", "C", "H")) +
            tuple(f"{a}_{b}_{metric}" for a, b in CONTRASTS
                  for metric in ("acquisition_difference", "loss_difference", "net")))


def _mean(rows, field):
    return sum((Fraction(row[field]) for row in rows), Fraction()) / len(rows)


def bootstrap_requests(request_rows, *, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED):
    """Resample whole requests, preserving every start, state and window."""
    ordered = sorted(request_rows, key=lambda row: row["request_id"])
    names = [f"{a}_{b}" for a, b in CONTRASTS]
    values = np.asarray([[row[name] for name in names] for row in ordered], dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64(seed))
    indices = rng.integers(0, len(ordered), size=(draws, len(ordered)))
    samples = values[indices].mean(axis=1)
    return {
        "draws": draws, "seed": seed, "bit_generator": "PCG64",
        "request_order": [row["request_id"] for row in ordered],
        "method": "paired whole requests; common draws; linear 2.5/97.5 percentiles",
        "scope": "conditional on one native trajectory and reused development panel",
        "intervals": {name: {"lower": float(np.quantile(samples[:, i], .025, method="linear")),
                              "upper": float(np.quantile(samples[:, i], .975, method="linear"))}
                      for i, name in enumerate(names)},
    }


def reduce_probes(probe_rows, sealed_probes, *, windows=WINDOWS,
                  draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED,
                  require_full_population=True):
    sealed = _index(sealed_probes, PROBE_KEY)
    actual = _index(probe_rows, ("window", *PROBE_KEY, "state"))
    requests = sorted({probe[0] for probe in sealed})
    if require_full_population:
        populations = Counter(probe[0] for probe in sealed)
        if tuple(windows) != WINDOWS or len(requests) != 64 or set(populations.values()) != {4}:
            raise ValueError("Production population must contain 64 requests, four starts and all 16 windows")
        if draws != BOOTSTRAP_DRAWS or seed != BOOTSTRAP_SEED:
            raise ValueError("Production bootstrap must retain 10,000 PCG64/1707 draws")
    expected = {(window, *probe, state) for window in windows for probe in sealed for state in STATES}
    if not sealed or not windows or set(actual) != expected:
        raise ValueError(f"Incomplete/different probe grid: expected {len(expected)}, observed {len(actual)}")
    blocks, grouped = [], defaultdict(list)
    for window in windows:
        for probe in sorted(sealed):
            observations = {state: actual[(window, *probe, state)] for state in STATES}
            row = _paired_block(observations, dict(zip(PROBE_KEY, probe), window=window))
            if ("valid_proposals" in sealed[probe] and
                    sealed[probe]["valid_proposals"] != row["valid_proposals"]):
                raise ValueError("Scored boundary differs from sealed input")
            blocks.append(row)
            grouped[(probe[0], window)].append(row)
    fields = _metric_fields()
    cells = {(rid, window): {field: _mean(grouped[(rid, window)], field) for field in fields}
             for rid in requests for window in windows}
    request_rows, window_rows = [], []
    for rid in requests:
        row = {"request_id": rid, "starts": len(grouped[(rid, windows[0])]), "windows": len(windows)}
        row.update({field: float(sum(cells[(rid, w)][field] for w in windows) / len(windows)) for field in fields})
        row.update({f"{a}_{b}": row[f"{a}_{b}_net"] for a, b in CONTRASTS})
        request_rows.append(row)
    for window in windows:
        row = {"window": window, "pass_index": 0 if window <= 64 else 1,
               "requests": len(requests), "starts": len(sealed)}
        row.update({field: float(sum(cells[(rid, window)][field] for rid in requests) / len(requests)) for field in fields})
        row.update({f"{a}_{b}": row[f"{a}_{b}_net"] for a, b in CONTRASTS})
        window_rows.append(row)
    means = {field: sum(cell[field] for cell in cells.values()) / len(cells) for field in fields}
    primary = {"unit": "accepted draft proposals per fixed block", "primary_contrast": "E_N",
               "request_count": len(requests), "window_count": len(windows), "sealed_starts": len(sealed),
               "conceptual_observations": len(probe_rows), "paired_block_windows": len(blocks),
               "levels": {state: float(means[f"a_{state}"]) for state in STATES},
               "contrasts": {f"{a}_{b}": {"estimate": float(means[f"{a}_{b}_net"]),
                    "exact": {"numerator": means[f"{a}_{b}_net"].numerator,
                              "denominator": means[f"{a}_{b}_net"].denominator},
                    "primary": (a, b) == ("E", "N")} for a, b in CONTRASTS},
               "decomposition": {field: float(value) for field, value in means.items() if not field.startswith("a_")},
               "gross_counts": {field: sum(row[field] for row in blocks) for field in fields if not field.startswith("a_")},
               "blocker_denominator": sum(row["pre_has_blocker"] for row in blocks),
               "all_integer_decompositions_pass": True,
               "reduction": "four starts within request/window; sixteen windows within request; equal requests"}
    primary["bootstrap"] = bootstrap_requests(request_rows, draws=draws, seed=seed)
    for name, interval in primary["bootstrap"]["intervals"].items():
        primary["contrasts"][name].update(interval)
    passes = []
    for pass_index in (0, 1):
        subset = [row for row in window_rows if row["pass_index"] == pass_index]
        if subset:
            passes.append({"pass_index": pass_index, "windows": len(subset),
                           **{field: sum(row[field] for row in subset) / len(subset) for field in fields}})
    primary["pass_descriptions"] = passes
    primary["request_signs"] = {name: {"improved": sum(row[name] > 0 for row in request_rows),
                                        "regressed": sum(row[name] < 0 for row in request_rows),
                                        "tied": sum(row[name] == 0 for row in request_rows)}
                                 for name in (f"{a}_{b}" for a, b in CONTRASTS)}
    return primary, request_rows, window_rows, blocks


def reduce_source(source_rows, *, windows=WINDOWS):
    """Summarize fitting geometry without treating overlapping anchors as samples."""
    if not source_rows:
        return {"status": "unavailable", "reason": "No source observation rows retained"}, []
    actual = _index(source_rows, ("window", *SOURCE_KEY, "state"))
    identities = {key(row, ("window", *SOURCE_KEY)) for row in source_rows}
    if set(actual) != {(*identity, state) for identity in identities for state in STATES}:
        raise ValueError("Source grid does not pair all five states")
    if {identity[0] for identity in identities} != set(windows):
        raise ValueError("Source grid does not contain every fixed window")
    for window in windows:
        if len({identity[1] for identity in identities if identity[0] == window}) != 2:
            raise ValueError("Source scoring must contain the original two responses per window")
    blocks = []
    for identity in sorted(identities):
        observations = {state: actual[(*identity, state)] for state in STATES}
        keep = observations["P"].get("keep")
        if type(keep) is not bool or any(observations[state].get("keep") is not keep for state in STATES):
            raise ValueError("Source states must retain an identical explicit sampled-anchor keep flag")
        row = _paired_block(observations, dict(zip(("window", *SOURCE_KEY), identity), keep=keep))
        if not keep and row["valid_proposals"] != 0:
            raise ValueError("A dropped/padded source anchor cannot contribute scored proposals")
        blocks.append(row)
    fields = _metric_fields()
    counts = {field: sum(row[field] for row in blocks) for field in fields if not field.startswith("a_")}
    eligible = [row for row in blocks if row["valid_proposals"] > 0]
    denominators = {
        "sampled_anchor_slots": len(blocks),
        "kept_anchor_occurrences": sum(row["keep"] for row in blocks),
        "dropped_or_padded_anchor_slots": sum(not row["keep"] for row in blocks),
        "eligible_blocks": len(eligible),
        "zero_supervision_anchor_slots": sum(row["valid_proposals"] == 0 for row in blocks),
        "kept_but_zero_supervision_anchor_occurrences": sum(row["keep"] and row["valid_proposals"] == 0 for row in blocks),
        "scored_proposal_occurrences": sum(row["valid_proposals"] for row in blocks),
        "pre_correct_prefix_occurrences": sum(row["pre_prefix"] for row in blocks),
        "blocks_with_pre_first_error": sum(row["pre_has_blocker"] for row in blocks),
        "scope": "One count per paired anchor, not five states; seed, invalid gaps, EOS suffix and padding excluded from scored positions",
    }
    return {"status": "complete", "unit": "overlapping sampled fitting-block occurrences",
            "paired_block_count": len(blocks), "response_window_count": 2 * len(windows),
            "conceptual_state_block_count": len(source_rows), "denominators": denominators,
            "blocker_denominator": denominators["blocks_with_pre_first_error"],
            "scored_proposal_occurrences": denominators["scored_proposal_occurrences"],
            "zero_supervision_blocks": denominators["zero_supervision_anchor_slots"],
            "gross_counts": counts,
            "mean_levels_all_sampled_slots": {state: sum(row[f"a_{state}"] for row in blocks) / len(blocks) for state in STATES},
            "mean_levels_eligible_blocks": {state: sum(row[f"a_{state}"] for row in eligible) / len(eligible) if eligible else None for state in STATES},
            "note": "Descriptive fitting outcomes; padding retained in raw rows but never counted as eligible exposure or independent evidence"}, blocks


def grid_accounting(probe_rows, sealed_probes, *, windows=WINDOWS):
    expected = {(window, *key(probe, PROBE_KEY), state)
                for window in windows for probe in sealed_probes for state in STATES}
    seen, counts = set(), Counter()
    for row in probe_rows:
        identity = key(row, ("window", *PROBE_KEY, "state"))
        if identity in seen:
            counts["duplicate"] += 1
        seen.add(identity)
        if identity not in expected:
            counts["unexpected"] += 1
        if row.get("status") in ("unattempted", "unknown"):
            counts[row["status"]] += 1
        else:
            counts["valid" if row.get("valid") is True else "invalid"] += 1
    counts["unattempted"] += len(expected - seen)
    return {"planned": len(expected), "recorded": len(probe_rows),
            **{name: counts[name] for name in ("valid", "invalid", "unknown", "unattempted", "duplicate", "unexpected")}}


def decide(primary, validity):
    if not validity.get("complete", False):
        return "INCOMPLETE"
    if not validity.get("all_required_checks_passed", False):
        return "INVALID"
    comparison = primary["contrasts"]["E_N"]
    if comparison["estimate"] > 0 and comparison["lower"] > 0:
        return "LOCAL_PREFIX_WITHHOLDING_SCREEN_PASSED"
    if comparison["upper"] < 0:
        return "NATIVE_TRANSFERS_BETTER"
    return "UNRESOLVED_STOP_THIS_SCREEN"


PROBE_PUBLIC = ("window", *PROBE_KEY, "state", "accepted_prefix", "valid_proposals", "first_error",
                "valid", "status", "matched_eos", "censored_at_cap", "reused_from", "execution_id", "error")
SOURCE_PUBLIC = ("window", *SOURCE_KEY, "pass_index", "presentation_id", "keep", "state", "accepted_prefix", "valid_proposals", "first_error",
                 "valid", "reused_from", "execution_id", "error")


def _csv(path, rows, fields):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_report(output, *, probe_rows, sealed_probes, validity, work, execution_commit,
                 source_rows=(), state_rows=(), windows=WINDOWS, require_full_population=True):
    """Write raw scalar outcomes before checking completeness or computing gains."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    _csv(output / "PROBE_OUTCOMES.csv", probe_rows, PROBE_PUBLIC)
    _csv(output / "SOURCE_OUTCOMES.csv", source_rows, SOURCE_PUBLIC)
    (output / "STATE_AND_DISPLACEMENT.json").write_text(json.dumps(list(state_rows), indent=2, allow_nan=False) + "\n")
    summary = {"status": "incomplete", "decision": "INCOMPLETE", "execution_commit": execution_commit,
               "validity": validity, "work": work, "grid": grid_accounting(probe_rows, sealed_probes, windows=windows)}
    if validity.get("complete") and validity.get("all_required_checks_passed"):
        try:
            primary, requests, per_window, blocks = reduce_probes(
                probe_rows, sealed_probes, windows=windows, require_full_population=require_full_population)
            source, source_blocks = reduce_source(source_rows, windows=windows)
            if source["status"] != "complete":
                raise ValueError("Required fitting comparison is unavailable")
        except (KeyError, ValueError) as error:
            summary.update(status="invalid", decision="INVALID", analysis_error=str(error))
        else:
            summary.update(status="completed", primary=primary, source=source,
                           decision=decide(primary, validity))
            for name, rows in (("PER_REQUEST.csv", requests), ("PER_WINDOW.csv", per_window),
                               ("INDEPENDENT_TRADEOFF.csv", blocks), ("SOURCE_TRADEOFF.csv", source_blocks)):
                _csv(output / name, rows, tuple(rows[0]))
    elif validity.get("complete"):
        summary.update(status="invalid", decision="INVALID")
    (output / "SUMMARY.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    text = ["# Native K+ correct-prefix loss investigation", "", "## Material Passport", "",
            "ARS-Codex 3.24.2 experiment-agent; scalar analysis of retained execution records.",
            f"Execution commit: `{execution_commit}`. Status: **{summary['status']}**.", "",
            "Only native N advances the event. E withholds current correct-prefix gradients; C withholds current first-error gradients. "
            "P is the static pre-update model; H executes Adam with zero current gradients, so persistent moments and weight decay can move it.",
            "Displayed full losses retain their native value; equal displayed loss does not imply equal gradients or updates.", "",
            f"**Decision: {summary['decision']}.**", ""]
    if summary["status"] == "completed":
        primary = summary["primary"]
        text += ["| Comparison | Mean proposal difference per fixed block | Conditional paired 95% interval |",
                 "|---|---:|---:|"]
        for name, contrast in primary["contrasts"].items():
            text.append(f"| {name}{' (primary)' if contrast['primary'] else ' (descriptive)'} | "
                        f"{contrast['estimate']:+.6f} | [{contrast['lower']:+.6f}, {contrast['upper']:+.6f}] |")
        text += ["", "The primary estimate equally averages starts within request/window, all sixteen windows within request, and the 64 requests. "
                 "Whole-request resampling retains all correlated starts, windows and states; the 20,480 outcomes are not independent samples.",
                 "", "| State | Independent acquired positions | Independent lost positions | Independent net | Fitting net |",
                 "|---|---:|---:|---:|---:|"]
        for state in CANDIDATES:
            gross = primary["gross_counts"]
            text.append(f"| {state} | {gross[f'{state}_acquired']} | {gross[f'{state}_lost']} | "
                        f"{gross[f'{state}_net']} | {summary['source']['gross_counts'][f'{state}_net']} |")
        denominators = summary["source"]["denominators"]
        text += ["", "Acquisition/loss totals count prefix positions. Blocker repairs and earlier breaks use block/window denominators and are retained separately. "
                 "Lower losses can improve net acceptance despite fewer acquisitions; that is not evidence of stronger repair learning.",
                 "Source observations are fitting-block occurrences with overlapping anchors, not independent confirmation.",
                 f"Source denominators (one count per paired anchor, not per state): {denominators['sampled_anchor_slots']} sampled slots, "
                 f"{denominators['kept_anchor_occurrences']} kept anchors, {denominators['eligible_blocks']} eligible blocks, "
                 f"{denominators['zero_supervision_anchor_slots']} zero-supervision slots "
                 f"({denominators['kept_but_zero_supervision_anchor_occurrences']} kept but empty), and "
                 f"{denominators['scored_proposal_occurrences']} scored proposal occurrences. "
                 "Padding is preserved in raw records but excluded from eligible exposure."]
    else:
        text += ["No complete-panel efficacy estimate is reported. Invalid, missing and unattempted work remain visible; no outcome is imputed.",
                 str(summary.get("analysis_error", validity.get("errors", [])))]
    text += ["", "## Interpretation limits", "",
             "This is a reused development panel and one native trajectory. Local one-step branches cannot be summed into an alternative training trajectory or a serving speedup. "
             "Removing a role also changes gradient magnitude, clipping and interaction with Adam history; this screen does not isolate directional causation. "
             "Intervals containing zero are unresolved, not equivalence. C/H remain explanatory controls, not replacement winners.",
             "A positive primary screen permits considering a separately designed scale control and short continuation; it does not authorize them. "
             "A null or negative result ends this screen without tuning or sample expansion.", "",
             "Work and state/displacement receipts are retained separately; nested timers must not be summed as independent costs.", ""]
    (output / "REPORT.md").write_text("\n".join(text))
    return summary
