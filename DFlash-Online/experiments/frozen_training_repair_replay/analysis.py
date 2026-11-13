"""Paired, slot-weighted reduction for the sealed event-seven training tags.

The 814 double-presented slots each receive the same total weight as the one
singleton. Whole requests, including all their paired slots, are bootstrapped.
This module consumes detached scalar observations and does no model work.
"""
from collections import Counter, defaultdict
from fractions import Fraction
import math

import numpy as np


OUTCOMES = ("earlier_break", "first_mismatch_at_tag", "accepts_through")
IDENTITY = ("request_id", "canonical_start", "proposal_slot", "presentation_id")
SLOT_IDENTITY = IDENTITY[:3]


def outcome(prefix, proposal_slot):
    if not isinstance(prefix, int) or not 0 <= prefix <= 15 or not 1 <= proposal_slot <= 15:
        raise ValueError("Invalid prefix length or one-based proposal slot")
    if prefix < proposal_slot - 1:
        return OUTCOMES[0]
    return OUTCOMES[1] if prefix == proposal_slot - 1 else OUTCOMES[2]


def _key(row, fields=IDENTITY):
    return tuple(row[field] for field in fields)


def pair_occurrences(native, recent):
    """Validate pairing before exposing the deliberately small public schema."""
    maps = []
    for rows in (native, recent):
        mapping = {_key(row): row for row in rows}
        if len(mapping) != len(rows):
            raise ValueError("Duplicate occurrence identity")
        maps.append(mapping)
    if not maps[0] or maps[0].keys() != maps[1].keys():
        raise ValueError("Missing or different paired occurrence identities")
    paired = []
    for key in sorted(maps[0]):
        left, right = (mapping[key] for mapping in maps)
        for field in ("pass_index", "anchor_ordinal", "parent_margin", "floor"):
            if left[field] != right[field]:
                raise ValueError(f"Paired {field} differs")
        if left["pass_index"] not in (0, 1) or left["presentation_id"] // 128 != left["pass_index"]:
            raise ValueError("Pass does not match presentation identity")
        if not left["valid"] or not right["valid"]:
            raise ValueError("Sealed canonical label is no longer valid")
        if left["floor"] != .5 * min(left["parent_margin"], 1):
            raise ValueError("Frozen floor does not match parent margin")
        row = {field: left[field] for field in (*IDENTITY, "pass_index", "anchor_ordinal", "parent_margin", "floor")}
        for arm, original in (("native", left), ("recent", right)):
            prefix, margin = original["correct_prefix"], original["margin"]
            if not math.isfinite(margin) or not math.isfinite(original["parent_margin"]):
                raise ValueError("Nonfinite scalar observation")
            label = outcome(prefix, row["proposal_slot"])
            direct = bool(original["native_correct"])
            if margin > 0 and not direct or margin < 0 and direct:
                raise ValueError("Native argmax correctness contradicts nonzero margin")
            if label == OUTCOMES[2] and not direct:
                raise ValueError("An accepted-through tag cannot be directly incorrect")
            if label == OUTCOMES[1] and direct:
                raise ValueError("An exact first mismatch cannot be directly correct")
            row.update({f"{arm}_outcome": label, f"{arm}_correct_prefix": prefix,
                        f"{arm}_accepts_through": int(label == OUTCOMES[2]),
                        f"{arm}_direct_correct": int(direct), f"{arm}_margin": margin,
                        f"{arm}_below_floor": int(margin < row["floor"]),
                        f"{arm}_wrong_at_tie": int(margin == 0 and not direct)})
        row["rescue"] = int(row["recent_accepts_through"] and not row["native_accepts_through"])
        row["loss"] = int(row["native_accepts_through"] and not row["recent_accepts_through"])
        row["net"] = row["rescue"] - row["loss"]
        row["direct_correctness_difference"] = row["recent_direct_correct"] - row["native_direct_correct"]
        row["margin_difference"] = row["recent_margin"] - row["native_margin"]
        row["outcome_disagrees"] = int(row["native_outcome"] != row["recent_outcome"])
        paired.append(row)
    return paired


def reduce_slots(paired):
    grouped = defaultdict(list)
    for row in paired:
        grouped[_key(row, SLOT_IDENTITY)].append(row)
    slots = []
    numeric = ("accepts_through", "direct_correct", "margin", "below_floor", "wrong_at_tie", "correct_prefix")
    for identity, rows in sorted(grouped.items()):
        if len(rows) not in (1, 2) or len({r["pass_index"] for r in rows}) != len(rows):
            raise ValueError("Slot must have one or two distinct eligible passes")
        slot = dict(zip(SLOT_IDENTITY, identity), occurrences=len(rows),
                    presentation_ids=",".join(str(r["presentation_id"]) for r in rows),
                    passes=",".join(str(r["pass_index"]) for r in rows))
        for arm in ("native", "recent"):
            for name in numeric:
                slot[f"{arm}_{name}"] = sum(r[f"{arm}_{name}"] for r in rows) / len(rows)
            for label in OUTCOMES:
                slot[f"{arm}_{label}_fraction"] = sum(r[f"{arm}_outcome"] == label for r in rows) / len(rows)
        for name in ("rescue", "loss", "net", "direct_correctness_difference", "margin_difference"):
            slot[name] = sum(r[name] for r in rows) / len(rows)
        for left in OUTCOMES:
            for right in OUTCOMES:
                slot[f"cell_{left}__{right}"] = sum(
                    r["native_outcome"] == left and r["recent_outcome"] == right for r in rows
                ) / len(rows)
        slot["any_outcome_disagreement"] = int(any(r["outcome_disagrees"] for r in rows))
        slot["any_retention_disagreement"] = int(any(r["rescue"] or r["loss"] for r in rows))
        slots.append(slot)
    return slots


def _fraction(value):
    value = Fraction(value)
    return {"numerator": value.numerator, "denominator": value.denominator}


def _cohort_summary(rows, *, slot_weighted):
    count = len(rows)
    retention = {arm: sum(Fraction(r[f"{arm}_accepts_through"]) for r in rows) for arm in ("native", "recent")}
    rescue, loss = (sum(Fraction(r[name]) for r in rows) for name in ("rescue", "loss"))
    net = retention["recent"] - retention["native"]
    headroom = count - retention["native"]
    if net != rescue - loss or net > headroom:
        raise ValueError("Paired rescue/loss/headroom identity failed")
    summary = dict(denominator=count, unit="equal-weight distinct slot" if slot_weighted else "paired occurrence",
        native_retention_pct=float(100 * retention["native"] / count),
        recent_retention_pct=float(100 * retention["recent"] / count),
        native_headroom_pct=float(100 * headroom / count),
        rescues=float(rescue) if slot_weighted else int(rescue),
        losses=float(loss) if slot_weighted else int(loss),
        rescue_pct=float(100 * rescue / count), loss_pct=float(100 * loss / count),
        net_difference_pp=float(100 * net / count), exact_net_difference_pp=_fraction(100 * net / count),
        exact_native_retention_pct=_fraction(100 * retention["native"] / count),
        exact_recent_retention_pct=_fraction(100 * retention["recent"] / count),
        net_equals_rescues_minus_losses=True, net_le_native_headroom=True)
    for arm in ("native", "recent"):
        summary[f"{arm}_accepted_equivalents"] = float(retention[arm]) if slot_weighted else int(retention[arm])
        for name in ("direct_correct", "below_floor", "wrong_at_tie"):
            summary[f"{arm}_{name}_pct"] = 100 * sum(r[f"{arm}_{name}"] for r in rows) / count
        summary[f"{arm}_mean_margin"] = math.fsum(r[f"{arm}_margin"] for r in rows) / count
        summary[f"{arm}_mean_correct_prefix"] = math.fsum(r[f"{arm}_correct_prefix"] for r in rows) / count
        for label in OUTCOMES:
            mass = sum(r[f"{arm}_{label}_fraction"] for r in rows) if slot_weighted else sum(r[f"{arm}_outcome"] == label for r in rows)
            summary[f"{arm}_{label}_equivalents"] = mass
    summary["direct_correctness_difference_pp"] = 100 * math.fsum(r["direct_correctness_difference"] for r in rows) / count
    summary["mean_margin_difference"] = math.fsum(r["margin_difference"] for r in rows) / count
    return summary


def request_summary(slots, paired):
    grouped, occurrences = defaultdict(list), defaultdict(list)
    for row in slots:
        grouped[row["request_id"]].append(row)
    for row in paired:
        occurrences[row["request_id"]].append(row)
    output = []
    for rid, rows in sorted(grouped.items()):
        raw = occurrences[rid]
        record = dict(request_id=rid, slots=len(rows), occurrences=len(raw),
            native_retained_equivalents=sum(r["native_accepts_through"] for r in rows),
            recent_retained_equivalents=sum(r["recent_accepts_through"] for r in rows),
            rescue_equivalents=sum(r["rescue"] for r in rows), loss_equivalents=sum(r["loss"] for r in rows),
            net_equivalents=sum(r["net"] for r in rows),
            any_outcome_disagreement=int(any(r["outcome_disagrees"] for r in raw)),
            any_retention_disagreement=int(any(r["rescue"] or r["loss"] for r in raw)),
            any_direct_correctness_disagreement=int(any(r["direct_correctness_difference"] for r in raw)),
            any_margin_disagreement=int(any(r["margin_difference"] for r in raw)))
        output.append(record)
    return output


def bootstrap_requests(requests, *, draws=10000, seed=1707):
    """Resample all request IDs jointly; use the draw's actual slot denominator."""
    rng = np.random.Generator(np.random.PCG64(seed))
    index = rng.integers(0, len(requests), size=(draws, len(requests)))
    counts = np.asarray([r["slots"] for r in requests], dtype=np.float64)
    differences = np.asarray([r["net_equivalents"] for r in requests], dtype=np.float64)
    values = 100 * differences[index].sum(axis=1) / counts[index].sum(axis=1)
    return dict(lower_pp=float(np.quantile(values, .025, method="linear")),
                upper_pp=float(np.quantile(values, .975, method="linear")),
                draws=draws, undefined_draws=0, seed=seed, bit_generator="PCG64",
                request_order=[r["request_id"] for r in requests],
                method="whole-request paired cluster bootstrap; linear 2.5/97.5 percentiles",
                scope="sensitivity to represented fitted requests; not training-seed uncertainty")


def summarize(native_rows, recent_rows, *, require_full_population=True, draws=10000, seed=1707):
    paired = pair_occurrences(native_rows, recent_rows)
    slots = reduce_slots(paired)
    requests = request_summary(slots, paired)
    pass_counts = Counter(r["pass_index"] for r in paired)
    multiplicity = Counter(r["occurrences"] for r in slots)
    if require_full_population and (len(requests), len(slots), len(paired), dict(pass_counts), dict(multiplicity)) != (
            128, 815, 1629, {0: 814, 1: 815}, {2: 814, 1: 1}):
        raise ValueError("Sealed population counts differ from 128 requests / 815 slots / 1629 occurrences")
    table = []
    for left in OUTCOMES:
        for right in OUTCOMES:
            table.append(dict(native_outcome=left, recent_outcome=right,
                paired_occurrences=sum(r["native_outcome"] == left and r["recent_outcome"] == right for r in paired),
                slot_equivalents=sum(r[f"cell_{left}__{right}"] for r in slots)))
    result = dict(status="completed", primary=_cohort_summary(slots, slot_weighted=True),
        occurrence_summary=_cohort_summary(paired, slot_weighted=False),
        pass_summaries={str(p): _cohort_summary([r for r in paired if r["pass_index"] == p], slot_weighted=False)
                        for p in sorted(pass_counts)},
        primary_interval=bootstrap_requests(requests, draws=draws, seed=seed),
        denominator_checks=dict(requests=len(requests), slots=len(slots), paired_occurrences=len(paired),
            arm_rows=2 * len(paired), pass_occurrences=dict(sorted(pass_counts.items())),
            slot_multiplicity=dict(sorted(multiplicity.items())), full_population_required=require_full_population),
        disagreements={key: sum(r[key] for r in requests) for key in (
            "any_outcome_disagreement", "any_retention_disagreement", "any_direct_correctness_disagreement", "any_margin_disagreement")})
    interval, primary = result["primary_interval"], result["primary"]
    result["interpretation"] = ("fitted_retention_advantage" if interval["lower_pp"] > 0 else
        "fitted_retention_disadvantage" if interval["upper_pp"] < 0 else "direction_unresolved")
    result["all_retention_tied"] = primary["rescues"] == primary["losses"] == 0
    return result, paired, slots, requests, table
