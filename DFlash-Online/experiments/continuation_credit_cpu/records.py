"""Read compact native blocks and compute parent-defined continuation evidence.

Only JSON and Python float arithmetic are used. Saved supports retain their FP32
values; all score reductions below use FP64. No model/runtime imports belong here.
"""

from collections import Counter
import json
import math
from pathlib import Path


BLOCK_SIZE = 16
FOLLOWER_CAPACITY = BLOCK_SIZE - 2
EXPECTED_COHORT = {"HIGH": 2114, "LOW": 4215, "NO_ERROR": 1863}


def block_roles(block):
    """Reconstruct roles from native argmax, including correctly predicted ties."""
    labels, proposals, valid = block["labels"], block["proposals"], block["valid"]
    m = sum(valid)
    if (len(labels) != BLOCK_SIZE or len(proposals) != BLOCK_SIZE or len(valid) != BLOCK_SIZE
            or valid != [False] + [True] * m + [False] * (BLOCK_SIZE - m - 1)):
        raise ValueError("Expected a seed followed by contiguous valid proposals")
    e = next((j for j in range(1, m + 1) if proposals[j] != labels[j]), -1)
    L = m if e < 0 else e - 1
    K = m if e < 0 else next(
        (j - 1 for j in range(e + 1, m + 1) if proposals[j] != labels[j]), m)
    return dict(L=L, e=e, r=K - e if e > 0 else 0,
                n=m - e if e > 0 else 0, K=K, m=m,
                has_error=e > 0, high=e > 0 and K > e)


def validate_block(block):
    """Check identities needed for this replay, without re-auditing provenance."""
    roles = block_roles(block)
    m, L, e, K = (roles[key] for key in ("m", "L", "e", "K"))
    if block["m"] != m or [p["position"] for p in block["positions"]] != list(range(1, m + 1)):
        raise ValueError("Compact positions do not cover every valid proposal")
    saved = block["roles"]
    expected_roles = dict(L=L, e=e, K=K, m=m, H=K - L if e > 0 else 0,
                          prefix=[0 < j <= L for j in range(BLOCK_SIZE)],
                          repair=[j == e for j in range(BLOCK_SIZE)],
                          labels=block["labels"], valid=block["valid"])
    if any(saved[key] != value for key, value in expected_roles.items()):
        raise ValueError("Saved roles disagree with native predictions")
    for p in block["positions"]:
        j = p["position"]
        correct = block["labels"][j] == block["proposals"][j]
        margin, support = p["margin"], p["support_fp32"]
        if (p["canonical"] != block["labels"][j]
                or p["prediction"] != block["proposals"][j]
                or p["correct"] != correct or p["rival"] == p["canonical"]):
            raise ValueError("Position token/argmax identities disagree")
        if not math.isfinite(margin) or not math.isfinite(support):
            raise ValueError("Nonfinite native margin/support")
        if ((correct and margin < 0) or (not correct and margin > 0)
                or p["margin_tie"] != (margin == 0)):
            raise ValueError("Native correctness or tie flag disagrees with margin")
        # This checks the saved FP32 support formula, not a model precision gate.
        if not 0 <= support <= 1 or not math.isclose(
                support, max(0.0, math.tanh(margin / 2)), rel_tol=2e-6, abs_tol=2e-7):
            raise ValueError("Saved support differs from max(0,tanh(margin/2))")
    return roles


def continuation_scores(positions, e):
    """Existing area and shared credit; the correct-token minimum never resets."""
    if e < 0:
        return None, None, []
    old_minimum = correct_minimum = 1.0
    old_area = shared_area = 0.0
    error_count = 1
    followers = []
    for p in positions:
        if p["position"] <= e:
            continue
        old_minimum = min(old_minimum, p["support_fp32"] if p["correct"] else 0.0)
        old_area += old_minimum
        if not p["correct"]:
            error_count += 1
            continue
        correct_minimum = min(correct_minimum, p["support_fp32"])
        credit = correct_minimum / error_count / FOLLOWER_CAPACITY
        shared_area += correct_minimum / error_count
        followers.append(dict(position=p["position"], w=correct_minimum,
                              error_count=error_count, credit=credit))
    old, shared = old_area / FOLLOWER_CAPACITY, shared_area / FOLLOWER_CAPACITY
    if not -1e-15 <= old <= shared + 1e-15 <= 1 + 1e-15:
        raise ValueError("Continuation score ordering failed")
    return old, shared, followers


def paired_block(parent, child, *, response_ordinal, record_id, anchor):
    """Score one exact response/anchor pair and its eligible later followers."""
    roles, child_roles = validate_block(parent), validate_block(child)
    if parent["labels"] != child["labels"] or parent["valid"] != child["valid"]:
        raise ValueError("Paired canonical labels or valid positions differ")
    positions = {p["position"]: p for p in parent["positions"]}
    L, e, K, m = (roles[key] for key in ("L", "e", "K", "m"))
    child_L = child_roles["L"]
    old, shared, followers = continuation_scores(parent["positions"], e)
    prefix_margins = [positions[j]["margin"] for j in range(1, L + 1)]
    prefix_margin = min(prefix_margins, default=0.0)
    deficit = max(0.0, -positions[e]["margin"]) if e > 0 else 0.0
    row = dict(response_ordinal=response_ordinal, record_id=record_id, anchor=anchor,
        **roles, child_L=child_L, u_old=old, u_shared=shared,
        delta_u=None if old is None else shared - old,
        later_error_count=sum(not p["correct"] for j, p in positions.items() if j > e) if e > 0 else 0,
        beta=math.exp(-(e - 1) / 7) if e > 0 else 0.0, d=deficit,
        h1=positions[e + 1]["support_fp32"] if 0 < e < m else 0.0,
        minimum_original_prefix_margin=prefix_margin, empty_prefix=L == 0,
        error_tie=e > 0 and positions[e]["margin"] == 0,
        prefix_tie=any(margin == 0 for margin in prefix_margins),
        Y_full=child_L - L, beyond_K=max(0, child_L - K),
        crossed_next=K < m and child_L > K,
        useful_repair=child_L >= e if e > 0 else None,
        lost_prefix=max(0, L - child_L), Y_old=min(child_L, K) - L)
    eligible = []
    if roles["high"]:
        for f in followers:
            j = f["position"]
            if j <= K:
                continue
            later_deficits = [max(0.0, -positions[k]["margin"])
                              for k in range(e + 1, j + 1) if not positions[k]["correct"]]
            eligible.append(dict(response_ordinal=response_ordinal, record_id=record_id,
                anchor=anchor, **f, target=int(child_L >= j), e=e, r=roles["r"], n=roles["n"],
                distance=j - e, u_old=old, first_error_deficit=deficit,
                prefix_margin=prefix_margin, empty_prefix=L == 0,
                later_mean_deficit=sum(later_deficits) / len(later_deficits) if later_deficits else 0.0,
                later_max_deficit=max(later_deficits, default=0.0)))
    return row, eligible


def read_receipt(path, binding, *, endpoint):
    receipt = json.loads(Path(path).read_text())
    if (receipt["status"] != "passed" or receipt["endpoint"] != endpoint
            or receipt["record_id"] != binding["record_id"]
            or receipt["anchors"] != binding["anchors"] or len(receipt["blocks"]) != 64):
        raise ValueError(f"Incomplete or misaligned response/anchor receipt: {path}")
    return receipt


def load_records(root, optional_tail_root=None):
    """Load the fixed PR64 cohort; optional TAIL outcomes remain descriptive."""
    root = Path(root)
    contract = json.loads((root / "CONTRACT.json").read_text())
    bindings = contract["features"]
    if (len(bindings) != 128 or {b["ordinal"] for b in bindings} != set(range(128))
            or len({b["record_id"] for b in bindings}) != 128
            or any(len(b["anchors"]) != 64 or len(set(b["anchors"])) != 64 for b in bindings)):
        raise ValueError("Require 128 distinct responses and 64 distinct anchors per response")
    blocks, followers, secondary, missing_tail = [], [], [], []
    for binding in sorted(bindings, key=lambda b: b["ordinal"]):
        ordinal = binding["ordinal"]
        directory = root / "primary" / f'gpu{binding["physical_shard"]}'
        parent = read_receipt(directory / "parent" / f"{ordinal:03d}.json", binding, endpoint="parent")
        child = read_receipt(directory / "child" / f"{ordinal:03d}.json", binding, endpoint="child")
        tail = None
        if optional_tail_root is not None:
            path = Path(optional_tail_root) / "mechanism" / "primary" / f"{ordinal:03d}.json"
            if path.exists():
                tail = read_receipt(path, binding, endpoint="TAIL")
            else:
                missing_tail.append(ordinal)
        for index, anchor in enumerate(binding["anchors"]):
            identity = dict(response_ordinal=ordinal, record_id=binding["record_id"], anchor=anchor)
            row, eligible = paired_block(parent["blocks"][index], child["blocks"][index], **identity)
            blocks.append(row)
            followers.extend(eligible)
            if tail is not None:
                tail_row, _ = paired_block(parent["blocks"][index], tail["blocks"][index], **identity)
                secondary.append(tail_row)
    counts = Counter("HIGH" if row["high"] else "LOW" if row["has_error"] else "NO_ERROR" for row in blocks)
    if len(blocks) != 8192 or dict(counts) != EXPECTED_COHORT:
        raise ValueError(f"Retained cohort differs from the prespecified whole cohort: {dict(counts)}")
    summary = dict(primary_root=str(root), optional_tail_root=str(optional_tail_root) if optional_tail_root else None,
        primary_responses=128, primary_blocks=len(blocks), parent_roles=dict(counts),
        eligible_later_followers=len(followers), tail_responses=len(secondary) // 64,
        tail_blocks=len(secondary), missing_tail_ordinals=missing_tail,
        arithmetic="FP64 reductions of retained FP32 support values; no model precision change",
        support_formula_check=dict(relative_tolerance=2e-6, absolute_tolerance=2e-7),
        alignment="response identity, original ordered anchors, canonical tokens, contiguous validity and native roles",
        secondary_scope="Same parent/responses as primary; descriptive, not independent replication")
    # Targeted preparation review inspected this exact C receipt and its writer.
    # It records summaries, not the per-token q(y) required to replay smoothing.
    summary["canonical_probability_availability"] = dict(
        status="deferred: canonical probabilities not retained",
        checked_C_receipt=str(Path(contract["endpoints"]["child"]["export"]).parent
                              / "RESPONSE_OBSERVATIONS.jsonl"),
        inspected_field="prefix_allocation",
        retained_fields=["blocks", "valid_positions", "prefix_positions", "changed_blocks",
            "changed_positions", "prefix_weight_before", "prefix_weight_after", "weight_moved",
            "mass_error_abs_sum", "mass_violations", "bounds_violations", "support_violations",
            "nonfinite_values", "prefix_length_counts", "by_prefix_length_and_slot", "delta_numerator"],
        writer="online_feedback/prefix_allocation.py:PrefixAllocation.reduce_response",
        missing="Per-token canonical probabilities q(y), including future valid positions",
        scope="Primary compact position schema and retained C receipt/writer inspection; no new model readout")
    return blocks, followers, secondary, summary
