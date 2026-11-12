"""Prespecified, outcome-blind matching and request-cluster risk analysis.

The matching stage consumes K+4/K+5 evidence only. Its output is a complete
serializable seal: labels, margins, all structural pools and selected pairs.
K+6 outcomes may be joined only after that seal has been persisted.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math

import numpy as np

SEED = 20261115
CALIPER = 0.25
BOOTSTRAP_DRAWS = 10_000
PRACTICAL_DIFFERENCE = 0.05
OUTCOMES = ("exact_slot_rejection", "earlier_rejection", "accepts_through")


def start_key(row):
    return str(row["source_id"]), int(row["start_offset"])


def payload_sha256(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def match_tie_key(source_id, repaired_start, comparator_start, slot):
    """Exact compact-JSON tie-break contract from the handoff."""
    raw = json.dumps([SEED, str(source_id), int(repaired_start),
                      int(comparator_start), int(slot)], ensure_ascii=False,
                     separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return dict(count=0, mean=None, minimum=None, q05=None, q25=None,
                    median=None, q75=None, q95=None, maximum=None)
    quantiles = np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95])
    return dict(count=len(values), mean=float(values.mean()),
                minimum=float(values.min()), maximum=float(values.max()),
                **dict(zip(("q05", "q25", "median", "q75", "q95"),
                           map(float, quantiles))))


def margin_balance(pairs):
    """Pair-weighted margins; sample SDs define the pooled within-group SD."""
    repaired = np.asarray([p["repaired_margin"] for p in pairs], dtype=np.float64)
    established = np.asarray([p["comparator_margin"] for p in pairs], dtype=np.float64)
    smd, constant_difference = None, False
    if len(pairs):
        variance_r = float(np.var(repaired, ddof=1)) if len(pairs) > 1 else 0.0
        variance_c = float(np.var(established, ddof=1)) if len(pairs) > 1 else 0.0
        pooled_sd = math.sqrt((variance_r + variance_c) / 2)
        difference = abs(float(repaired.mean() - established.mean()))
        if pooled_sd:
            smd = difference / pooled_sd
        elif difference == 0:
            smd = 0.0
        else:
            # Infinite SMD is represented explicitly, not serialized as invalid JSON.
            constant_difference = True
    else:
        pooled_sd = None
    return dict(repaired=distribution(repaired), established=distribution(established),
                absolute_difference=distribution(np.abs(repaired - established)),
                pooled_sample_sd=pooled_sd, absolute_smd=smd,
                undefined_nonidentical_constant_groups=constant_difference,
                imbalanced=constant_difference or smd is None or smd > 0.10)


def _validate_start(row):
    source, offset = start_key(row)
    a4, a5, limit = int(row["a4"]), int(row["a5"]), int(row["valid_proposal_count"])
    if not source or offset < 0 or not 0 <= limit <= 15:
        raise ValueError("Invalid source identity, start offset or proposal boundary")
    if not 0 <= a4 <= limit or not 0 <= a5 <= limit:
        raise ValueError("Accepted prefix exceeds the valid EOS/cap boundary")
    genuine = bool(row["k4_first_error"])
    if genuine and a4 >= limit:
        raise ValueError("A genuine K+4 error needs a valid canonical position")
    margins = {int(item["slot"]): float(item["margin"]) for item in row["margins"]}
    if len(margins) != len(row["margins"]):
        raise ValueError("Duplicate K+5 margin slot")
    if set(margins) != set(range(1, a5 + 1)):
        raise ValueError("Margins must cover every and only K+5 accepted proposal")
    if any(not math.isfinite(m) or m < 0 for m in margins.values()):
        raise ValueError("An accepted native K+5 token has a negative/nonfinite margin")
    return dict(source_id=source, start_offset=offset, a4=a4, a5=a5,
                valid_proposal_count=limit, k4_first_error=genuine,
                margins=[dict(slot=j, margin=margins[j]) for j in sorted(margins)])


def build_pre_k6(starts):
    """Label and match starts without accepting any K+6 input.

    Each input has ``source_id, start_offset, a4, a5, valid_proposal_count,
    k4_first_error`` and ``margins=[{slot, margin}, ...]``. Explicitly invalid
    starts remain in missing-work accounting; valid inputs require complete
    accepted-slot margins. Slots are one-based and exclude the known seed.
    """
    rows, unavailable, seen = [], [], set()
    for supplied in starts:
        key = start_key(supplied)
        if key in seen:
            raise ValueError("Duplicate source/start identity")
        seen.add(key)
        if not supplied.get("valid", True):
            unavailable.append(dict(source_id=key[0], start_offset=key[1],
                                    reason=supplied.get("reason", "invalid_or_missing")))
            continue
        rows.append(_validate_start(supplied))
    by_request = defaultdict(list)
    for row in rows:
        by_request[row["source_id"]].append(row)
    repairs, pools, pairs = [], [], []
    for row in rows:
        j = row["a4"] + 1
        repaired = row["k4_first_error"] and row["a5"] >= j
        row["recently_repaired"] = repaired
        row["j"] = j if repaired else None
        if not repaired:
            continue
        margin = row["margins"][j - 1]["margin"]
        controls = []
        for other in by_request[row["source_id"]]:
            if (other["start_offset"] != row["start_offset"]
                    and other["a5"] == row["a5"]
                    and min(other["a4"], other["a5"], other["valid_proposal_count"]) >= j):
                controls.append(dict(start_offset=other["start_offset"],
                                     margin=other["margins"][j - 1]["margin"]))
        controls.sort(key=lambda item: item["start_offset"])
        eligible = [control for control in controls
                    if abs(control["margin"] - margin) <= CALIPER
                    and (control["margin"] == 0) == (margin == 0)]
        repair = dict(source_id=row["source_id"], repaired_start=row["start_offset"],
                      j=j, a4=row["a4"], a5=row["a5"], repaired_margin=margin,
                      structural_comparators=len(controls), caliper_comparators=len(eligible))
        repair["match_status"] = ("matched" if eligible else
                                  "outside_margin_caliper_or_zero_tie_rule" if controls else
                                  "no_structural_comparator")
        repairs.append(repair)
        pools.append({**repair, "comparators": controls})
        if eligible:
            selected = min(eligible, key=lambda item: (
                abs(item["margin"] - margin), match_tie_key(
                    row["source_id"], row["start_offset"], item["start_offset"], j)))
            pairs.append({**repair, "comparator_start": selected["start_offset"],
                          "comparator_margin": selected["margin"],
                          "absolute_margin_difference": abs(selected["margin"] - margin),
                          "pair_weight": 1})
    matched_keys = {(p["source_id"], p["repaired_start"]) for p in pairs}
    # Only pools of the matched repair population enter the structural contrast.
    required = set(matched_keys)
    for pool in pools:
        if (pool["source_id"], pool["repaired_start"]) in matched_keys:
            required.update((pool["source_id"], c["start_offset"]) for c in pool["comparators"])
    reuse = Counter((p["source_id"], p["comparator_start"]) for p in pairs)
    balance = margin_balance(pairs)
    unmatched = [r for r in repairs if r["match_status"] != "matched"]
    coverage = dict(
        total_starts=len(starts), valid_starts=len(rows), unavailable_starts=len(unavailable),
        all_repairs=len(repairs), repairs_with_structural_comparator=sum(bool(p["comparators"]) for p in pools),
        matched_repairs=len(pairs), distinct_repaired_starts=len(matched_keys),
        matched_requests=len({p["source_id"] for p in pairs}),
        distinct_comparator_starts=len(reuse), maximum_comparator_reuse=max(reuse.values(), default=0),
        unmatched_repairs=len(unmatched), unmatched_margin=distribution([r["repaired_margin"] for r in unmatched]),
        unmatched_depth_counts=dict(sorted(Counter(r["j"] for r in unmatched).items())),
        unmatched_reason_counts=dict(Counter(r["match_status"] for r in unmatched)),
        required_k6_unique_starts=len(required),
    )
    flags = []
    if len(matched_keys) < 100:
        flags.append("fewer_than_100_matched_repaired_starts")
    if coverage["matched_requests"] < 40:
        flags.append("fewer_than_40_matched_requests")
    if balance["imbalanced"]:
        flags.append("absolute_smd_above_0.10_or_undefined")
    payload = dict(schema="repair_history_margin.pre_k6.v1", seed=SEED, caliper=CALIPER,
                   labels=rows, unavailable_starts=unavailable, repairs=repairs,
                   structural_pools=pools, matched_pairs=pairs, coverage=coverage,
                   balance=balance, support_flags=flags,
                   comparator_reuse=[dict(source_id=key[0], start_offset=key[1], uses=count)
                                     for key, count in sorted(reuse.items())],
                   required_k6_starts=[dict(source_id=key[0], start_offset=key[1]) for key in sorted(required)])
    payload["payload_sha256"] = payload_sha256(payload)
    return payload


def validate_seal(seal):
    payload = {key: value for key, value in seal.items() if key != "payload_sha256"}
    if payload_sha256(payload) != seal["payload_sha256"]:
        raise ValueError("Pre-K+6 matching seal changed")


def classify_k6(row, slot):
    """Classify against the full canonical suffix, never conditioning on reach."""
    if row is None or not row.get("valid", True):
        return dict(valid=False, reason="missing" if row is None else row.get("reason", "invalid"))
    accepted, limit = int(row["a6"]), int(row["valid_proposal_count"])
    if not 0 <= accepted <= limit <= 15 or not 1 <= slot <= limit:
        return dict(valid=False, reason="invalid_acceptance_or_slot_boundary")
    if accepted < slot and not row.get("k6_first_error", False):
        return dict(valid=False, reason="no_genuine_rejection_before_slot")
    if "direct_correct" in row:
        direct_map = row["direct_correct"]
        direct = direct_map.get(str(slot), direct_map.get(slot))
    else:
        proposals, canonical = row.get("proposals", []), row.get("canonical", [])
        direct = proposals[slot - 1] == canonical[slot - 1] if min(len(proposals), len(canonical)) >= slot else None
    if not isinstance(direct, (bool, np.bool_)):
        return dict(valid=False, reason="missing_direct_token_comparison")
    outcome = ("accepts_through" if accepted >= slot else
               "exact_slot_rejection" if accepted == slot - 1 else "earlier_rejection")
    if (outcome == "accepts_through" and not direct) or (outcome == "exact_slot_rejection" and direct):
        return dict(valid=False, reason="native_acceptance_disagrees_with_direct_token")
    return dict(valid=True, outcome=outcome, direct_correct=bool(direct))


def cluster_bootstrap(rows, value_key, *, draws=BOOTSTRAP_DRAWS, seed=SEED):
    """Resample matched requests, preserving pair counts and reused controls."""
    clusters = defaultdict(list)
    for row in rows:
        clusters[str(row["source_id"])].append(float(row[value_key]))
    if not clusters:
        return dict(draws=draws, valid_draws=0, undefined_draws=draws,
                    lower=None, upper=None, requests=0, seed=seed)
    arrays = [clusters[key] for key in sorted(clusters)]
    sums = np.asarray([sum(values) for values in arrays], dtype=np.float64)
    counts = np.asarray([len(values) for values in arrays], dtype=np.int64)
    rng = np.random.Generator(np.random.PCG64(seed))
    values = []
    for first in range(0, draws, 500):
        indices = rng.integers(len(arrays), size=(min(500, draws - first), len(arrays)))
        numerators, denominators = sums[indices].sum(axis=1), counts[indices].sum(axis=1)
        values.extend((numerators[denominators > 0] / denominators[denominators > 0]).tolist())
    interval = np.quantile(values, [0.025, 0.975]) if values else [None, None]
    return dict(draws=draws, valid_draws=len(values), undefined_draws=draws - len(values),
                lower=float(interval[0]) if values else None,
                upper=float(interval[1]) if values else None, requests=len(clusters), seed=seed)


def _rates(rows, prefix):
    counts = {outcome: sum(row.get(prefix + "_outcome") == outcome for row in rows)
              for outcome in OUTCOMES}
    valid = sum(row.get(prefix + "_valid", False) for row in rows)
    total = len(rows)
    return dict(denominator=total, valid=valid, missing=total - valid, counts=counts,
                rates={name: value / total if total and valid == total else None for name, value in counts.items()},
                direct_correct=sum(row.get(prefix + "_direct_correct") is True for row in rows),
                direct_correct_rate=(sum(row.get(prefix + "_direct_correct") is True for row in rows) / total
                                     if total and valid == total else None))


def analyze_k6(seal, outcomes, *, execution_complete=True):
    """Join an immutable pre-outcome seal to unique native K+6 replay records.

    Primary estimates require complete matched pairs. Missing evidence is not
    coded as a non-event and does not create a selected complete-case estimand.
    Earlier rejection is a valid outcome, and always stays in the denominator.
    """
    validate_seal(seal)
    lookup = {start_key(row): row for row in outcomes}
    if len(lookup) != len(outcomes):
        raise ValueError("Duplicate K+6 replay identity")
    required = {start_key(row) for row in seal["required_k6_starts"]}
    if set(lookup) - required:
        raise ValueError("K+6 replay contains starts outside the sealed work list")
    pool_lookup = {(p["source_id"], p["repaired_start"]): p for p in seal["structural_pools"]}
    pairs, structural = [], []
    for supplied in seal["matched_pairs"]:
        row = dict(supplied)
        for role, offset in (("repaired", row["repaired_start"]), ("comparator", row["comparator_start"])):
            result = classify_k6(lookup.get((row["source_id"], offset)), row["j"])
            row.update({role + "_" + key: value for key, value in result.items()})
        row["valid_pair"] = row["repaired_valid"] and row["comparator_valid"]
        for outcome in OUTCOMES:
            row[outcome + "_difference"] = (
                int(row["repaired_outcome"] == outcome) - int(row["comparator_outcome"] == outcome)
            ) if row["valid_pair"] else None
        row["direct_correct_difference"] = (int(row["repaired_direct_correct"]) - int(row["comparator_direct_correct"])) if row["valid_pair"] else None
        pairs.append(row)
        pool = pool_lookup[(row["source_id"], row["repaired_start"])]
        control_rows = [classify_k6(lookup.get((row["source_id"], c["start_offset"])), row["j"])
                        for c in pool["comparators"]]
        complete = row["repaired_valid"] and bool(control_rows) and all(c["valid"] for c in control_rows)
        structural_row = dict(source_id=row["source_id"], repaired_start=row["repaired_start"],
                              j=row["j"], comparators=len(control_rows),
                              valid_comparators=sum(c["valid"] for c in control_rows), valid=complete)
        for outcome in OUTCOMES:
            control_rate = sum(c.get("outcome") == outcome for c in control_rows) / len(control_rows) if complete else None
            structural_row["comparator_" + outcome + "_rate"] = control_rate
            structural_row[outcome + "_difference"] = (int(row["repaired_outcome"] == outcome) - control_rate) if complete else None
        structural.append(structural_row)
    all_pairs = bool(pairs) and all(row["valid_pair"] for row in pairs)
    all_structural = bool(structural) and all(row["valid"] for row in structural)
    primary = dict(denominator=len(pairs), valid_pairs=sum(row["valid_pair"] for row in pairs),
                   repaired=_rates(pairs, "repaired"), established=_rates(pairs, "comparator"),
                   delta=None, interval=None, earlier_break_difference=None,
                   acceptance_through_difference=None, direct_correct_difference=None)
    if all_pairs:
        for field, metric in (("delta", "exact_slot_rejection_difference"),
                              ("earlier_break_difference", "earlier_rejection_difference"),
                              ("acceptance_through_difference", "accepts_through_difference"),
                              ("direct_correct_difference", "direct_correct_difference")):
            primary[field] = sum(row[metric] for row in pairs) / len(pairs)
        primary["interval"] = cluster_bootstrap(pairs, "exact_slot_rejection_difference")
    structural_result = dict(denominator=len(structural), valid_repairs=sum(row["valid"] for row in structural),
                             delta=None, interval=None, earlier_break_difference=None)
    if all_structural:
        structural_result.update(
            delta=sum(row["exact_slot_rejection_difference"] for row in structural) / len(structural),
            interval=cluster_bootstrap(structural, "exact_slot_rejection_difference"),
            earlier_break_difference=sum(row["earlier_rejection_difference"] for row in structural) / len(structural))
    flags = list(seal["support_flags"])
    if seal["unavailable_starts"]:
        flags.append("pre_k6_starts_unavailable")
    if not all_pairs:
        flags.append("incomplete_matched_outcomes")
    if not all_structural:
        flags.append("incomplete_structural_outcomes")
    if not execution_complete:
        flags.append("execution_incomplete")
    interval = primary["interval"]
    if flags or interval is None:
        decision = "unresolved"
    elif interval["lower"] > PRACTICAL_DIFFERENCE:
        decision = "additional_risk_above_five_percentage_points_supported"
    elif interval["upper"] < PRACTICAL_DIFFERENCE:
        decision = "prespecified_five_point_positive_association_ruled_out"
    else:
        decision = "unresolved"
    by_request = []
    for source in sorted({p["source_id"] for p in pairs}):
        group = [p for p in pairs if p["source_id"] == source]
        by_request.append(dict(source_id=source, pairs=len(group),
                               valid_pairs=sum(p["valid_pair"] for p in group),
                               exact_slot_difference_sum=sum(p["exact_slot_rejection_difference"] for p in group)
                               if all(p["valid_pair"] for p in group) else None))
    return dict(schema="repair_history_margin.result.v1", pre_k6_payload_sha256=seal["payload_sha256"],
                decision=decision, interpretation_flags=flags, primary=primary,
                structural=structural_result, coverage=seal["coverage"], balance=seal["balance"],
                matched_pair_outcomes=pairs, structural_outcomes=structural, per_request=by_request,
                required_k6_starts=len(required), recorded_k6_starts=len(outcomes),
                missing_k6_starts=len(required - set(lookup)))
