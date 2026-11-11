"""Strict retained-state reuse and canonical/native per-start scoring."""

from __future__ import annotations

from .census import digest


def retained_same_state(parent, child, ordinal):
    """Conservative sufficient condition, stronger than shared output offset.

    Full earlier native blocks must agree, including proposals/verdicts/commits
    and positions. Under the pinned worker that also fixes target-feature
    ingestion boundaries. Child draft KV was built with its own weights.
    """
    p, c = parent["metadata"]["trace"], child["metadata"]["trace"]
    n = ordinal - 1
    if not 0 < n < len(c):
        return False
    fields = (
        "prompt_ids",
        "response_ids",
        "stop_token_ids",
        "max_new_tokens",
        "finish_reason",
    )
    if any(parent[k] != child[k] for k in fields) or p[:n] != c[:n]:
        return False
    return (
        p[n]["output_start"] == c[n]["output_start"]
        and p[n]["anchor_token"] == c[n]["anchor_token"]
        and p[n]["positions"] == c[n]["positions"]
    )


def neutral_parity(expected, observed):
    fields = (
        "output_start",
        "anchor_token",
        "draft_tokens",
        "target_argmax",
        "positions",
        "accepted_draft_tokens",
        "native_accepted_draft_tokens",
        "committed_progress",
        "committed_tokens",
        "finished",
        "stop_clipped",
    )
    differences = [k for k in fields if expected[k] != observed[k]]
    return dict(
        status="passed" if not differences else "failed",
        differences=differences,
        all_15_proposals_exact=expected["draft_tokens"] == observed["draft_tokens"],
        expected_sha256=digest(expected),
        observed_sha256=digest(observed),
    )


def score(selected, parent_block, child_block, canonical, *, provenance):
    if digest(parent_block) != selected["parent_block_sha256"]:
        raise ValueError("Parent block differs from sealed sample")
    if (
        child_block["output_start"] != parent_block["output_start"]
        or child_block["anchor_token"] != parent_block["anchor_token"]
        or child_block["positions"] != parent_block["positions"]
    ):
        raise ValueError("Child block starts from another native state")
    start = parent_block["output_start"] + 1
    committed = child_block["committed_tokens"]
    if committed != canonical[start : start + len(committed)]:
        raise ValueError("Child emitted noncanonical tokens")
    labels = canonical[start : start + selected["valid_slots"]]
    proposed = child_block["draft_tokens"][: len(labels)]
    j = selected["j"] - 1
    intact = proposed[:j] == labels[:j]
    match = proposed[j] == labels[j]
    accepted = child_block["accepted_draft_tokens"]
    if proposed[:accepted] != labels[:accepted] or accepted > len(labels):
        raise ValueError("Native accepted prefix is not canonical")
    parent_accepted = parent_block["accepted_draft_tokens"]
    repair = intact and match and accepted > parent_accepted
    both = proposed[j + 1 : j + 3] == labels[j + 1 : j + 3]
    canonical_prefix = next(
        (i for i, (a, b) in enumerate(zip(proposed, labels)) if a != b), len(labels)
    )
    delta = accepted - parent_accepted
    return dict(
        selected,
        status="valid",
        provenance=provenance,
        child_accepted=accepted,
        parent_accepted=parent_accepted,
        intact=intact,
        blocker_still_wrong=not match,
        still_blocked=intact and not match,
        both_followers_intact=both,
        blocked_with_both=intact and not match and both,
        repair=repair,
        canonical_repair=intact and match,
        repair_with_two_accepted=repair and both and accepted >= j + 3,
        earlier_prefix_break=not intact,
        native_prefix_change=delta,
        prefix_gained=max(delta, 0),
        prefix_lost=max(-delta, 0),
        prefix_tie=delta == 0,
        child_canonical_prefix=canonical_prefix,
        canonical_native_prefix_difference=canonical_prefix - accepted,
        child_block_sha256=digest(child_block),
    )
