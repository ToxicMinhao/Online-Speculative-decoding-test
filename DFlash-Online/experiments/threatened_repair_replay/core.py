"""Outcome-local threatened selection and immutable full-layout replay plans.

This module sees only memory observations. Held-out probe outcomes cannot enter
either selection rule. Hashes select tags; layout order controls computation.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import math


def tag_key(row):
    return str(row["request_id"]), int(row["canonical_start"]), int(row["proposal_slot"])


def computation_key(row):
    return (str(row["request_id"]), int(row["reference_presentation_id"]),
            int(row["anchor_ordinal"]), int(row["proposal_slot"]))


def _indexed(rows):
    result = {tag_key(row): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("Repeated memory tag")
    return result


def selection_hash(arm, window, row):
    rid, start, slot = tag_key(row)
    text = (f"trr-v1-T|{window}|{rid}|{start}|{slot}" if arm == "T" else
            f"trr-v1-U|{window}|{slot}|{rid}|{start}")
    if arm not in ("T", "U"):
        raise ValueError("Unknown replay arm")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _observation(row, slot):
    prefix = int(row["accepted_prefix"])
    valid = int(row["valid_proposals"])
    correct, margin = bool(row["native_correct"]), float(row["margin"])
    if (not row["valid"] or not 1 <= slot <= valid <= 15 or not 0 <= prefix <= valid
            or not math.isfinite(margin) or (margin > 0 and not correct)
            or (margin < 0 and correct) or (prefix >= slot and not correct)
            or (prefix == slot - 1 and correct)):
        raise ValueError("Invalid memory prefix, native decision or margin")
    return prefix, correct, margin


def select_replay(memory_tags, pre_rows, native_rows, *, window, current_request_ids):
    """Select at most four impending first rejections and a same-position control."""
    tags, pre, native = map(_indexed, (memory_tags, pre_rows, native_rows))
    if tags.keys() != pre.keys() or tags.keys() != native.keys():
        raise ValueError("Memory observations do not cover the sealed population")
    excluded = set(map(str, current_request_ids))
    rows, eligible, threats = [], [], []
    for identity, tag in sorted(tags.items()):
        rid, _, slot = identity
        p, pc, pm = _observation(pre[identity], slot)
        n, nc, nm = _observation(native[identity], slot)
        if (pre[identity]["canonical_token"] != tag["canonical_token"] or
                native[identity]["canonical_token"] != tag["canonical_token"] or
                pre[identity]["weight"] != tag["weight"] or
                native[identity]["weight"] != tag["weight"]):
            raise ValueError("Memory labels or positional weights changed")
        accepted = rid not in excluded and p >= slot
        threatened = accepted and n == slot - 1
        if accepted:
            eligible.append(tag)
        if threatened:
            threats.append(tag)
        rows.append(dict(window=window, request_id=rid, canonical_start=identity[1],
            proposal_slot=slot, a_P=p, a_N=n, excluded_current=rid in excluded,
            currently_accepted=accepted, threat=threatened,
            P_direct_correct=pc, N_direct_correct=nc, P_margin=pm, N_margin=nm,
            P_wrong_tie=pm == 0 and not pc, N_wrong_tie=nm == 0 and not nc,
            earlier_break=accepted and n < slot - 1,
            T_hash=selection_hash("T", window, tag), U_hash=selection_hash("U", window, tag)))
    ordering = lambda arm: lambda tag: (selection_hash(arm, window, tag), tag_key(tag))
    selected_t = sorted(threats, key=ordering("T"))[:4]
    counts = Counter(tag["proposal_slot"] for tag in selected_t)
    selected_u = []
    for slot, count in sorted(counts.items()):
        candidates = [tag for tag in eligible if tag["proposal_slot"] == slot]
        selected_u.extend(sorted(candidates, key=ordering("U"))[:count])
    u, t = {tag_key(tag) for tag in selected_u}, {tag_key(tag) for tag in selected_t}
    for row in rows:
        row.update(selected_U=tag_key(row) in u, selected_T=tag_key(row) in t)
    # Computation never follows pseudorandom selection rank.
    return dict(window=window, rows=rows, eligible=sorted(eligible, key=computation_key),
        threats=sorted(threats, key=computation_key),
        selected_U=sorted(deepcopy(selected_u), key=computation_key),
        selected_T=sorted(deepcopy(selected_t), key=computation_key),
        active=bool(t), identical_sets=u == t, overlap=len(u & t),
        counts_by_slot=dict(sorted(counts.items())))


def bind_layouts(layouts, candidate_tags):
    """Attach independent frozen tag records to their original full presentations."""
    by_layout = {row["layout_id"]: row for row in layouts}
    if len(by_layout) != len(layouts):
        raise ValueError("Repeated reference layout")
    grouped = defaultdict(list)
    for tag in _indexed(candidate_tags).values():
        grouped[tag["layout_id"]].append(tag)
    result = []
    for identity, tags in grouped.items():
        if identity not in by_layout:
            raise ValueError("A selected tag has no retained reference layout")
        layout = deepcopy(by_layout[identity])
        layout["candidate_tags"] = sorted(deepcopy(tags), key=computation_key)
        for tag in tags:
            if (str(tag["request_id"]), int(tag["reference_presentation_id"])) != (
                    str(layout["request_id"]), int(layout["reference_presentation_id"])):
                raise ValueError("Tag/reference presentation identity mismatch")
        result.append(layout)
    return sorted(result, key=lambda row: (str(row["request_id"]), int(row["reference_presentation_id"])))


def padded_layouts(layouts, selected_u, selected_t):
    """Both branches receive this exact union, including zero-weight layouts."""
    u, t = _indexed(selected_u), _indexed(selected_t)
    if Counter(key[2] for key in u) != Counter(key[2] for key in t) or len(u) > 4:
        raise ValueError("Replay control is not count/position matched")
    merged = dict(u)
    for key, tag in t.items():
        if key in merged and merged[key] != tag:
            raise ValueError("Overlapping selection changed immutable tag metadata")
        merged[key] = tag
    result = bind_layouts(layouts, list(merged.values()))
    if len(result) > 8:
        raise ValueError("Replay union exceeded eight original layouts")
    for layout in result:
        layout["selected_U"] = [tag_key(tag) for tag in layout["candidate_tags"] if tag_key(tag) in u]
        layout["selected_T"] = [tag_key(tag) for tag in layout["candidate_tags"] if tag_key(tag) in t]
    return result
