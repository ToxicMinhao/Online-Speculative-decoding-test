"""Fixed-request attribution analysis for native-LR size controls.

P/N/E are authenticated historical measurements. Only S_low and S_high are
newly scored. The unit of resampling is the whole request, not an anchor or
window. No measurement here changes a model or selects a calibration alpha.
"""
from collections import Counter, defaultdict
from fractions import Fraction
import json
from pathlib import Path

import numpy as np

from experiments.native_prefix_loss.analysis import _csv, _index, _prefix, key


WINDOWS = tuple(range(8, 129, 8))
STATES = ("P", "N", "E", "S_low", "S_high")
HISTORICAL_STATES = STATES[:3]
CONTROL_STATES = STATES[3:]
CANDIDATES = STATES[1:]
PROBE_KEY = ("request_id", "canonical_start", "round_ordinal")
CONTRASTS = (("E", "N"), ("E", "S_low"), ("E", "S_high"),
             ("S_low", "N"), ("S_high", "N"), ("S_high", "S_low"))
PRIMARY = ("E_S_low", "E_S_high")
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 1707


def tradeoff(prefixes, valid):
    """Separate accepted-position counts from block-level repair opportunities."""
    before = prefixes["P"]
    row = {"pre_prefix": before, "valid_proposals": valid,
           "pre_has_blocker": int(before < valid),
           "pre_has_prefix": int(before > 0)}
    for state in CANDIDATES:
        after = prefixes[state]
        row.update({f"{state}_acquired": max(after - before, 0),
                    f"{state}_lost": max(before - after, 0),
                    f"{state}_net": after - before,
                    f"{state}_earlier_break": int(after < before),
                    f"{state}_repairs_pre_blocker": int(before < valid and after > before)})
    for state in ("E", *CONTROL_STATES):
        row[f"{state}_displaced_native_repair"] = int(
            before < valid and prefixes["N"] > before and prefixes[state] <= before)
    for candidate, control in CONTRASTS:
        name = f"{candidate}_{control}"
        acquisition = row[f"{candidate}_acquired"] - row[f"{control}_acquired"]
        loss = row[f"{candidate}_lost"] - row[f"{control}_lost"]
        row.update({f"{name}_acquisition_difference": acquisition,
                    f"{name}_loss_difference": loss,
                    f"{name}_net": prefixes[candidate] - prefixes[control]})
        if row[f"{name}_net"] != acquisition - loss:
            raise ValueError("Accepted-prefix acquisition/loss decomposition failed")
    row.update({f"a_{state}": value for state, value in prefixes.items()})
    return row


def _paired_block(observations, identity):
    prefixes = {state: _prefix(observations[state]) for state in STATES}
    boundary = observations["P"]["valid_proposals"]
    if any(row["valid_proposals"] != boundary for row in observations.values()):
        raise ValueError("States do not share the saved canonical scoring boundary")
    for state, row in observations.items():
        alias = row.get("reused_from")
        if alias is not None:
            if state != "S_high" or alias != "S_low":
                raise ValueError("Only an explicit exact S_high/S_low alias is supported")
            if any(row.get(field) != observations[alias].get(field) for field in
                   ("accepted_prefix", "valid_proposals", "first_error", "matched_eos", "censored_at_cap")):
                raise ValueError("Aliased scoring outcome differs from its source")
    return {**identity, **tradeoff(prefixes, boundary)}


def _metric_fields():
    return (tuple(f"a_{state}" for state in STATES) +
            tuple(f"{state}_{metric}" for state in CANDIDATES
                  for metric in ("acquired", "lost", "net", "earlier_break", "repairs_pre_blocker")) +
            tuple(f"{state}_displaced_native_repair" for state in ("E", *CONTROL_STATES)) +
            tuple(f"{a}_{b}_{metric}" for a, b in CONTRASTS
                  for metric in ("acquisition_difference", "loss_difference", "net")))


def _mean(rows, field):
    return sum((Fraction(row[field]) for row in rows), Fraction()) / len(rows)


def bootstrap_requests(request_rows, *, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED):
    """Use PR47's deterministic request order and shared draws for all contrasts."""
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
        "intervals": {name: {
            "lower": float(np.quantile(samples[:, index], .025, method="linear")),
            "upper": float(np.quantile(samples[:, index], .975, method="linear"))}
            for index, name in enumerate(names)},
    }


def reduce_probes(probe_rows, sealed_probes, *, windows=WINDOWS,
                  draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED,
                  require_full_population=True):
    sealed = _index(sealed_probes, PROBE_KEY)
    actual = _index(probe_rows, ("window", *PROBE_KEY, "state"))
    requests = sorted({item[0] for item in sealed})
    if require_full_population:
        populations = Counter(item[0] for item in sealed)
        if tuple(windows) != WINDOWS or len(requests) != 64 or set(populations.values()) != {4}:
            raise ValueError("Production population requires 64 requests, four starts and all sixteen windows")
        if draws != BOOTSTRAP_DRAWS or seed != BOOTSTRAP_SEED:
            raise ValueError("Production bootstrap requires 10,000 PCG64/1707 draws")
    expected = {(window, *probe, state) for window in windows for probe in sealed for state in STATES}
    if not sealed or not windows or set(actual) != expected:
        raise ValueError(f"Incomplete/different five-state grid: expected {len(expected)}, observed {len(actual)}")

    blocks, grouped = [], defaultdict(list)
    for window in windows:
        for probe in sorted(sealed):
            observations = {state: actual[(window, *probe, state)] for state in STATES}
            row = _paired_block(observations, dict(zip(PROBE_KEY, probe), window=window))
            if "valid_proposals" in sealed[probe] and sealed[probe]["valid_proposals"] != row["valid_proposals"]:
                raise ValueError("Scored boundary differs from the sealed key")
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
        row = {"window": window, "pass_index": int(window > 64),
               "requests": len(requests), "starts": len(sealed)}
        row.update({field: float(sum(cells[(rid, window)][field] for rid in requests) / len(requests)) for field in fields})
        row.update({f"{a}_{b}": row[f"{a}_{b}_net"] for a, b in CONTRASTS})
        window_rows.append(row)

    means = {field: sum(cell[field] for cell in cells.values()) / len(cells) for field in fields}
    summary = {
        "unit": "accepted draft proposals per fixed block", "primary_contrasts": list(PRIMARY),
        "historical_contrast": "E_N", "request_count": len(requests), "window_count": len(windows),
        "sealed_starts": len(sealed), "paired_block_windows": len(blocks),
        "conceptual_observations": len(probe_rows),
        "historical_observations": len(windows) * len(sealed) * 3,
        "new_logical_control_observations": len(windows) * len(sealed) * 2,
        "levels": {state: float(means[f"a_{state}"]) for state in STATES},
        "contrasts": {f"{a}_{b}": {
            "estimate": float(means[f"{a}_{b}_net"]),
            "exact": {"numerator": means[f"{a}_{b}_net"].numerator,
                      "denominator": means[f"{a}_{b}_net"].denominator},
            "primary": f"{a}_{b}" in PRIMARY,
            "measurement_origin": "historical_PR47_recomputed" if (a, b) == ("E", "N") else "historical_and_new_or_new_controls"}
            for a, b in CONTRASTS},
        "gross_counts": {field: sum(row[field] for row in blocks) for field in fields if not field.startswith("a_")},
        "denominators": {
            "paired_block_windows": len(blocks),
            "valid_proposal_occurrences": sum(row["valid_proposals"] for row in blocks),
            "pre_first_blocker_opportunities": sum(row["pre_has_blocker"] for row in blocks),
            "pre_prefix_break_opportunities": sum(row["pre_has_prefix"] for row in blocks),
            "native_repair_displacement_opportunities": sum(row["N_repairs_pre_blocker"] for row in blocks),
            "scope": "Paired block/window occurrences; no multiplication by five states or independent-token interpretation"},
        "all_scalar_arms_exact_aliases": all(actual[(w, *probe, "S_high")].get("reused_from") == "S_low"
                                           for w in windows for probe in sealed),
        "all_integer_decompositions_pass": True,
        "reduction": "four starts within request/window; sixteen windows within request; equal requests",
    }
    summary["distinct_primary_comparisons"] = 1 if summary["all_scalar_arms_exact_aliases"] else 2
    summary["bootstrap"] = bootstrap_requests(request_rows, draws=draws, seed=seed)
    for name, interval in summary["bootstrap"]["intervals"].items():
        summary["contrasts"][name].update(interval)
    summary["pass_descriptions"] = []
    for pass_index in (0, 1):
        subset = [row for row in window_rows if row["pass_index"] == pass_index]
        if subset:
            summary["pass_descriptions"].append({"pass_index": pass_index, "windows": len(subset),
                **{field: sum(row[field] for row in subset) / len(subset) for field in fields}})
    summary["request_signs"] = {name: {
        "improved": sum(row[name] > 0 for row in request_rows),
        "regressed": sum(row[name] < 0 for row in request_rows),
        "tied": sum(row[name] == 0 for row in request_rows)}
        for name in (f"{a}_{b}" for a, b in CONTRASTS)}
    return summary, request_rows, window_rows, blocks


def grid_accounting(rows, sealed_probes, *, windows=WINDOWS):
    expected = {(window, *key(probe, PROBE_KEY), state)
                for window in windows for probe in sealed_probes for state in STATES}
    counts, seen = Counter(), set()
    for row in rows:
        identity = key(row, ("window", *PROBE_KEY, "state"))
        if identity in seen:
            counts["duplicate"] += 1
        seen.add(identity)
        if identity not in expected:
            counts["unexpected"] += 1
        if row.get("status") in ("unknown", "unattempted"):
            counts[row["status"]] += 1
        else:
            counts["valid" if row.get("valid") is True else "invalid"] += 1
    counts["unattempted"] += len(expected - seen)
    return {"planned": len(expected), "recorded": len(rows),
            **{name: counts[name] for name in ("valid", "invalid", "unknown", "unattempted", "duplicate", "unexpected")}}


def decide(summary, validity):
    if not validity.get("complete"):
        return "INCOMPLETE"
    if not validity.get("all_required_checks_passed"):
        return "INVALID_OR_UNQUALIFIED"
    primary = [summary["contrasts"][name] for name in PRIMARY]
    if all(row["estimate"] > 0 and row["lower"] > 0 for row in primary):
        return "LOCAL_ROLE_ADVANTAGE_BEYOND_BOTH_NEARBY_SIZE_CONTROLS"
    if all(row["upper"] < 0 for row in primary):
        return "BOTH_SIZE_CONTROLS_OUTPERFORM_HISTORICAL_E"
    if any(summary["contrasts"][f"{state}_N"]["lower"] > 0 for state in CONTROL_STATES):
        return "ORDINARY_UPDATE_SIZE_REMAINS_A_VIABLE_EXPLANATION"
    return "UNRESOLVED_STOP_THIS_SCREEN"


PUBLIC_FIELDS = ("window", *PROBE_KEY, "state", "accepted_prefix", "valid_proposals", "first_error",
                 "valid", "status", "matched_eos", "censored_at_cap", "reused_from", "measurement_origin",
                 "historical_execution_commit", "execution_id", "error")


def write_report(output, *, probe_rows, sealed_probes, validity, work, execution_commit,
                 calibration=(), geometry=(), windows=WINDOWS, require_full_population=True):
    """Retain incomplete records; never infer a primary effect from a subset."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    _csv(output / "PROBE_OUTCOMES.csv", probe_rows, PUBLIC_FIELDS)
    summary = {"status": "incomplete", "decision": "INCOMPLETE", "execution_commit": execution_commit,
               "validity": validity, "work": work,
               "grid": grid_accounting(probe_rows, sealed_probes, windows=windows)}
    for name, rows in (("CALIBRATION.json", calibration), ("GEOMETRY.json", geometry)):
        (output / name).write_text(json.dumps(list(rows), indent=2, allow_nan=False) + "\n")
    if validity.get("complete") and validity.get("all_required_checks_passed"):
        try:
            estimate, requests, window_rows, blocks = reduce_probes(
                probe_rows, sealed_probes, windows=windows, require_full_population=require_full_population)
        except (KeyError, ValueError) as error:
            summary.update(status="invalid", decision="INVALID_OR_UNQUALIFIED", analysis_error=str(error))
        else:
            summary.update(status="completed", comparison=estimate, decision=decide(estimate, validity))
            for name, rows in (("PER_REQUEST.csv", requests), ("PER_WINDOW.csv", window_rows),
                               ("ACQUISITION_LOSS.csv", blocks)):
                _csv(output / name, rows, tuple(rows[0]))
    elif validity.get("complete"):
        summary.update(status="invalid", decision="INVALID_OR_UNQUALIFIED")
    (output / "SUMMARY.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    text = ["# Native K+ update-size attribution", "", "## Material Passport", "",
            "ARS-Codex 3.24.2 experiment-agent; analysis of authenticated historical and new fixed-block observations.",
            f"Execution commit: `{execution_commit}`. Status: **{summary['status']}**.", "",
            "P/N/E outcomes are reused from PR47. Only the two size-calibrated native-LR controls are newly scored. "
            "Both controls at all sixteen windows are selected and sealed before scoring; historical E outcomes were already known.", "",
            f"**Decision: {summary['decision']}.**", ""]
    if summary["status"] == "completed":
        result = summary["comparison"]
        text += ["| Contrast | Mean accepted-prefix difference per fixed block | Conditional paired 95% interval |",
                 "|---|---:|---:|"]
        for name, row in result["contrasts"].items():
            label = "historical" if name == "E_N" else ("joint primary" if row["primary"] else "descriptive")
            text.append(f"| {name} ({label}) | {row['estimate']:+.6f} | [{row['lower']:+.6f}, {row['upper']:+.6f}] |")
        text += ["", "Advancement requires both E-minus-control means and both lower bounds above zero. "
                 "Exact aliases count once; unresolved intervals do not establish equivalence.", "",
                 "| State versus P | Acquired prefix occurrences | Lost prefix occurrences | Net |",
                 "|---|---:|---:|---:|"]
        for state in CANDIDATES:
            gross = result["gross_counts"]
            text.append(f"| {state} | {gross[state+'_acquired']} | {gross[state+'_lost']} | {gross[state+'_net']} |")
        den = result["denominators"]
        text += ["", f"Denominators: {den['paired_block_windows']} paired block/windows; "
                 f"{den['pre_first_blocker_opportunities']} old-first-blocker repair opportunities; "
                 f"{den['pre_prefix_break_opportunities']} old-prefix break opportunities; "
                 f"{den['native_repair_displacement_opportunities']} native repairs that another endpoint could displace. "
                 "The acquisition/loss table retains every key and all separate counts.", "",
                 "The estimate averages starts within request/window, windows within request, then requests equally. "
                 "Common 10,000 PCG64/1707 draws resample whole requests; no window is an independent training replicate."]
    else:
        text += ["No complete-panel efficacy estimate is reported. Invalid, unknown, missing and unattempted work remains visible.",
                 str(summary.get("analysis_error", validity.get("errors", [])))]
    text += ["", "## Interpretation limits", "",
             "Equal global norm does not imply equal per-layer motion, direction or functional change. Scalar endpoints retain N's updated "
             "Adam moments, while E's can differ. This compares the total local role intervention against ordinary LR reduction; it does "
             "not isolate a harmful gradient component. Beating two nearby endpoints does not rule out every LR or an unobserved "
             "nonmonotonic acceptance optimum. Per-window alpha uses E's endpoint size and is not a deployable learned LR schedule.", "",
             "E inherits earlier prefix learning and persistent optimizer history. Sixteen forks cannot be summed into an E trajectory. "
             "This reused development panel and one training seed limit inference. Fixed-block acceptance is not complete-request "
             "average acceptance or speedup. E tensor geometry exists only at windows 8/128; other E geometry is unavailable.", "",
             "No new source-fitting observations, complete responses or teacher features are produced. Report physical calibration, "
             "startup, reconstruction, loads, exports and storage separately; nested clocks are not additive. A positive local screen "
             "permits a separately designed continuation, not an automatic next experiment.", ""]
    (output / "REPORT.md").write_text("\n".join(text))
    return summary
