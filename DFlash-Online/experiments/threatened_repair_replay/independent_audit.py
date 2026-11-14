"""Independent CPU audit of public TRR scalars and sealed populations.

This intentionally imports no experiment reducers or model code. It checks
published arithmetic; it cannot establish that a native forward was correct.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

import numpy as np


WINDOWS = tuple(range(8, 129, 8))
STATES = ("P", "N", "U", "T")
PROBE = ("request_id", "canonical_start", "round_ordinal")
TAG = ("request_id", "canonical_start", "proposal_slot")
INTEGERS = {"window", "canonical_start", "round_ordinal", "proposal_slot",
            "accepted_prefix", "valid_proposals", "first_error", "a_P", "a_N"}
BOOLEANS = {"valid", "excluded_current", "currently_accepted", "threat",
            "selected_U", "selected_T", "direct_correct", "P_direct_correct",
            "N_direct_correct", "P_wrong_tie", "N_wrong_tie"}


def read_csv(path):
    """Parse published scalars without interpreting nonempty 'False' as true."""
    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for name, value in row.items():
            if name in INTEGERS:
                row[name] = int(value) if value else None
            elif name in BOOLEANS:
                if value not in ("True", "False"):
                    raise ValueError(f"Invalid boolean {name}={value!r}")
                row[name] = value == "True"
            elif name == "margin" or name.endswith("_margin"):
                row[name] = float(value)
    return rows


def identity(row, fields):
    return tuple(row[field] for field in fields)


def index(rows, fields):
    result = {identity(row, fields): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate identities in {fields}")
    return result


def require(condition, message):
    if not condition:
        raise ValueError(message)


def prefix(row):
    a, v = row["accepted_prefix"], row["valid_proposals"]
    require(row["valid"] is True and 0 <= a <= v <= 15, "Invalid prefix observation")
    require(row["first_error"] == (a + 1 if a < v else None), "First-error mismatch")
    return a


def native_decision(correct, margin, a, j):
    require(math.isfinite(margin), "Nonfinite native margin")
    require(not (margin > 0 and not correct or margin < 0 and correct), "Margin/argmax mismatch")
    require(not (a >= j and not correct or a == j - 1 and correct), "Prefix/argmax mismatch")


def decomposition(p, n, u, t, valid):
    """Integer occurrence accounting, including loss of a native blocker repair."""
    values = {"N": n, "U": u, "T": t}
    out = {}
    for arm, a in values.items():
        out[f"{arm}_acquired"] = max(a - p, 0)
        out[f"{arm}_lost"] = max(p - a, 0)
        out[f"{arm}_net"] = a - p
        out[f"{arm}_earlier_error"] = int(a < p)
        out[f"{arm}_repairs_pre_blocker"] = int(p < valid and a > p)
    for arm in ("U", "T"):
        out[f"{arm}_displaced_native_blocker_repair"] = int(p < valid and n > p and values[arm] <= p)
    for control in ("N", "U"):
        name = f"T_{control}"
        out[f"{name}_acquisition_difference"] = out["T_acquired"] - out[f"{control}_acquired"]
        out[f"{name}_loss_difference"] = out["T_lost"] - out[f"{control}_lost"]
        out[f"{name}_net"] = t - values[control]
        require(out[f"{name}_net"] == out[f"{name}_acquisition_difference"] - out[f"{name}_loss_difference"],
                "Integer acquisition/loss identity failed")
    return out


def paired_intervals(request_values):
    """Use one resampling matrix for both complete-request contrast vectors."""
    request_ids = sorted(request_values)
    values = np.array([request_values[rid] for rid in request_ids], dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64(1707))
    draws = rng.integers(0, len(values), size=(10000, len(values)))
    samples = values[draws].mean(axis=1)
    endpoints = np.quantile(samples, [.025, .975], axis=0, method="linear")
    return {name: {"lower": float(endpoints[0, col]), "upper": float(endpoints[1, col])}
            for col, name in enumerate(("T_N", "T_U"))}


def hash_order(arm, window, tag):
    rid, start, j = tag
    text = (f"trr-v1-T|{window}|{rid}|{start}|{j}" if arm == "T" else
            f"trr-v1-U|{window}|{j}|{rid}|{start}")
    return hashlib.sha256(text.encode("utf-8")).digest(), tag


def audit_selection(rows, sealed_tags):
    records = index(rows, ("window", *TAG))
    require(set(records) == {(w, *tag) for w in WINDOWS for tag in sealed_tags},
            "Selection must cover every sealed tag in all 16 windows")
    all_selected, all_threats, union = set(), set(), {}
    active = set()
    nonidentical = 0
    threat_occurrences = selected_occurrences = 0
    for window in WINDOWS:
        eligible, threats, chosen = set(), set(), {"U": set(), "T": set()}
        for tag in sealed_tags:
            row = records[(window, *tag)]
            p, n, j = row["a_P"], row["a_N"], tag[-1]
            require(1 <= j <= 15 and 0 <= p <= 15 and 0 <= n <= 15, "Selection prefix range")
            e = not row["excluded_current"] and p >= j
            h = e and n == j - 1
            require(row["currently_accepted"] == e and row["threat"] == h, "Eligibility/threat flag mismatch")
            for state, a in (("P", p), ("N", n)):
                correct, margin = row[f"{state}_direct_correct"], row[f"{state}_margin"]
                native_decision(correct, margin, a, j)
                require(row[f"{state}_wrong_tie"] == (margin == 0 and not correct), "Wrong-tie flag mismatch")
            if e:
                eligible.add(tag)
            if h:
                threats.add(tag)
            for arm in ("U", "T"):
                if row[f"selected_{arm}"]:
                    chosen[arm].add(tag)
        expected_t = set(sorted(threats, key=lambda tag: hash_order("T", window, tag))[:4])
        expected_u = set()
        for j, count in Counter(tag[-1] for tag in expected_t).items():
            stratum = [tag for tag in eligible if tag[-1] == j]
            expected_u.update(sorted(stratum, key=lambda tag: hash_order("U", window, tag))[:count])
        require(chosen["T"] == expected_t and chosen["U"] == expected_u, "Deterministic replay selection mismatch")
        all_selected.update(expected_t)
        all_threats.update(threats)
        threat_occurrences += len(threats)
        selected_occurrences += len(expected_t)
        nonidentical += expected_t != expected_u
        if expected_t:
            active.add(window)
        union.update({(window, *tag): records[(window, *tag)] for tag in expected_t | expected_u})
    support = dict(distinct_selected_T_tags=len(all_selected),
        distinct_selected_T_requests=len({tag[0] for tag in all_selected}),
        active_windows=len(active), nonidentical_selection_windows=nonidentical,
        all_threat_distinct_tags=len(all_threats), all_threat_distinct_requests=len({tag[0] for tag in all_threats}),
        all_threat_occurrences=threat_occurrences, selected_T_occurrences=selected_occurrences)
    support["opportunity_placement_pass"] = (len(all_selected) >= 16 and
        support["distinct_selected_T_requests"] >= 8 and len(active) >= 4 and nonidentical >= 4)
    return support, active, union


def audit_memory(rows, selected_union):
    records = index(rows, ("window", *TAG, "state"))
    require(set(records) == {(*key, state) for key in selected_union for state in STATES},
            "Selected-memory outcome grid is incomplete")
    groups = {"selected_T": [], "union": []}
    for key, selection in selected_union.items():
        j = key[-1]
        observations = {s: records[(*key, s)] for s in STATES}
        require(len({r["valid_proposals"] for r in observations.values()}) == 1, "Memory boundaries differ")
        a = {s: prefix(r) for s, r in observations.items()}
        require(a["P"] == selection["a_P"] and a["N"] == selection["a_N"], "Memory selection P/N mismatch")
        item = {"tag": key[1:]}
        for state, row in observations.items():
            require(j <= row["valid_proposals"], "Memory tag outside valid proposals")
            native_decision(row["direct_correct"], row["margin"], a[state], j)
            item.update({f"{state}_accepts": int(a[state] >= j),
                f"{state}_exact_rejection": int(a[state] == j - 1),
                f"{state}_earlier_break": int(a[state] < j - 1),
                f"{state}_direct_correct": int(row["direct_correct"]),
                f"{state}_wrong_tie": int(not row["direct_correct"] and row["margin"] == 0),
                f"{state}_margin": row["margin"]})
        for control in ("N", "U"):
            item[f"T_{control}_rescue"] = int(a["T"] >= j > a[control])
            item[f"T_{control}_loss"] = int(a[control] >= j > a["T"])
        groups["union"].append(item)
        if selection["selected_T"]:
            groups["selected_T"].append(item)
    result = {}
    for name, group in groups.items():
        stats = dict(tag_window_count=len(group), distinct_tags=len({r["tag"] for r in group}))
        for state in STATES:
            for metric in ("accepts", "exact_rejection", "earlier_break", "direct_correct", "wrong_tie"):
                stats[f"{state}_{metric}"] = sum(r[f"{state}_{metric}"] for r in group)
            stats[f"{state}_mean_margin"] = math.fsum(r[f"{state}_margin"] for r in group) / len(group) if group else None
        for control in ("N", "U"):
            for metric in ("rescue", "loss"):
                stats[f"T_{control}_{metric}"] = sum(r[f"T_{control}_{metric}"] for r in group)
        result[name] = stats
    result["manipulation_pass"] = result["selected_T"]["T_N_rescue"] > 0
    return result


def recompute(probe_rows, selection_rows, memory_rows, sealed_probes, sealed_tags):
    probes, tags = index(sealed_probes, PROBE), index(sealed_tags, TAG)
    request_counts = Counter(k[0] for k in probes)
    require(len(probes) == 256 and len(request_counts) == 64 and set(request_counts.values()) == {4},
            "Require 64 sealed requests with four distinct starts each")
    require(len(tags) == 815 and len({k[0] for k in tags}) == 128, "Require all 815 sealed tags across 128 requests")
    support, active, selected_union = audit_selection(selection_rows, tags)
    memory = audit_memory(memory_rows, selected_union)
    records = index(probe_rows, ("window", *PROBE, "state"))
    require(set(records) == {(w, *k, s) for w in WINDOWS for k in probes for s in STATES},
            "Probe outcomes must cover all states, starts and 16 windows")
    request_totals, levels, gross = defaultdict(Counter), Counter(), Counter()
    window_totals = defaultdict(Counter)
    for window in WINDOWS:
        for key, sealed in probes.items():
            rows = {s: records[(window, *key, s)] for s in STATES}
            a = {s: prefix(row) for s, row in rows.items()}
            v = rows["P"]["valid_proposals"]
            require(all(row["valid_proposals"] == v for row in rows.values()), "Probe boundaries differ")
            require("valid_proposals" not in sealed or sealed["valid_proposals"] == v, "Sealed boundary mismatch")
            for s, row in rows.items():
                origin = row.get("reused_from")
                require(not origin or origin in STATES and a[s] == a[origin], "Reused prefix differs")
            require(window in active or a["N"] == a["U"] == a["T"], "No-threat window is not native fallback")
            metrics = decomposition(a["P"], a["N"], a["U"], a["T"], v)
            gross.update(metrics)
            levels.update(a)
            for contrast, control in (("T_N", "N"), ("T_U", "U")):
                delta = a["T"] - a[control]
                request_totals[key[0]][contrast] += delta
                window_totals[window][contrast] += delta
    # The sealed design is balanced: four starts in each of sixteen windows.
    request_values = {rid: [counts["T_N"] / 64, counts["T_U"] / 64]
                      for rid, counts in request_totals.items()}
    intervals = paired_intervals(request_values)
    contrasts = {}
    for contrast in ("T_N", "T_U"):
        value = Fraction(sum(r[contrast] for r in request_totals.values()), 64 * 16 * 4)
        contrasts[contrast] = dict(estimate=float(value), exact=dict(numerator=value.numerator,
            denominator=value.denominator), **intervals[contrast])
    primary = dict(request_count=64, window_count=16, sealed_starts=256,
        conceptual_observations=len(probe_rows), paired_block_windows=4096,
        levels={s: levels[s] / 4096 for s in STATES}, contrasts=contrasts,
        decomposition={name: count / 4096 for name, count in gross.items()},
        gross_counts=dict(gross), all_integer_decompositions_pass=True,
        bootstrap=dict(draws=10000, seed=1707, bit_generator="PCG64",
                       request_order=sorted(request_values), intervals=intervals))
    secondary = dict(active_window_count=len(active), active_window_only=True,
        contrasts={name: sum(window_totals[w][name] for w in active) / (256 * len(active)) if active else None
                   for name in ("T_N", "T_U")})
    if not support["opportunity_placement_pass"]:
        decision = "INSUFFICIENT_OPPORTUNITY_OR_PLACEMENT"
    elif not memory["manipulation_pass"]:
        decision = "NO_DEMONSTRATED_TARGETED_CONTROL"
    elif min(contrasts[name]["lower"] for name in contrasts) > 0:
        decision = "LOCAL_ADVANCEMENT_SCREEN_PASSED"
    elif contrasts["T_N"]["upper"] < 0:
        decision = "INDEPENDENT_HARM"
    elif contrasts["T_U"]["upper"] < 0:
        decision = "THREAT_SELECTION_WORSE_THAN_UNIFORM"
    elif contrasts["T_N"]["lower"] > 0:
        decision = "REPLAY_BENEFIT_WITHOUT_DEMONSTRATED_TARGETING_ADVANTAGE"
    else:
        decision = "UNRESOLVED_LOCAL_TRANSFER"
    return dict(primary=primary, support=support, memory=memory,
                active_window_secondary=secondary, decision=decision)


def compare(expected, published, path="", errors=None):
    """Compare every independently computed field, ignoring report-only prose."""
    errors = [] if errors is None else errors
    checked = 0
    if isinstance(expected, dict):
        for key, value in expected.items():
            label = f"{path}.{key}" if path else key
            if not isinstance(published, dict) or key not in published:
                errors.append(f"Missing {label}")
            else:
                checked += compare(value, published[key], label, errors)[0]
    else:
        checked += 1
        equal = (isinstance(published, (int, float)) and math.isfinite(published) and
                 math.isclose(expected, published, rel_tol=0, abs_tol=1e-12)) if isinstance(expected, float) else expected == published
        if not equal:
            errors.append(f"{path}: recomputed={expected!r}, published={published!r}")
    return checked, errors


def audit_report(report_dir, population_dir=None):
    report_dir = Path(report_dir)
    population_dir = Path(population_dir) if population_dir else report_dir
    published = json.loads((report_dir / "PRIMARY_RESULTS.json").read_text())
    counts = {name: len(read_csv(report_dir / name)) for name in
              ("PROBE_OUTCOMES.csv", "WINDOW_SELECTION.csv", "MEMORY_OUTCOMES.csv")}
    result = dict(status="failed", scope="independent public-scalar audit; no model execution",
                  experiment_status=published.get("status"), observed_rows=counts, checks=0, errors=[])
    if published.get("status") != "completed":
        valid_status = published.get("status") in ("incomplete", "invalid")
        correct_decision = published.get("decision") == ("INCOMPLETE" if published.get("status") == "incomplete" else "INVALID")
        no_claim = all(name not in published for name in ("primary", "support", "memory", "active_window_secondary"))
        result.update(checks=3, status="passed" if valid_status and correct_decision and no_claim else "failed",
                      efficacy_audited=False)
        if result["status"] == "failed":
            result["errors"].append("Incomplete/invalid report contains a completed efficacy claim")
        return result
    try:
        require(published.get("validity", {}).get("complete") is True and
                published.get("validity", {}).get("all_required_checks_passed") is True,
                "Completed report lacks passing execution validity")
        expected = recompute(read_csv(report_dir / "PROBE_OUTCOMES.csv"),
            read_csv(report_dir / "WINDOW_SELECTION.csv"), read_csv(report_dir / "MEMORY_OUTCOMES.csv"),
            read_csv(population_dir / "PROBE_KEYS.csv"), read_csv(population_dir / "MEMORY_REFERENCE.csv"))
        result["checks"], result["errors"] = compare(expected, published)
        result.update(recomputed=expected, efficacy_audited=True)
        result["status"] = "failed" if result["errors"] else "passed"
    except (ValueError, KeyError) as error:
        result["errors"].append(str(error))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--population-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_report(args.report_dir, args.population_dir)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "checks", "errors")}))
    raise SystemExit(0 if result["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
