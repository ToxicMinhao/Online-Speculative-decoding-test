"""Prespecified first-block repair counts at the same prompt plus target seed."""

from collections import defaultdict
from .data import digest
from .evaluation import METHODS


def observation(record, request, canonical, *, checkpoint=None):
    """Extract original native proposals; never use post-error target_argmax labels."""
    if (
        record["prompt_ids"] != request["prompt_ids"]
        or canonical["prompt_ids"] != request["prompt_ids"]
    ):
        raise ValueError("Matched-start prompt mismatch")
    if checkpoint is not None and str(record["checkpoint_id"]) != str(checkpoint):
        raise ValueError("Wrong matched-start export")
    seed = canonical["response_ids"][0]
    if not record["response_ids"] or record["response_ids"][0] != seed:
        raise ValueError("Target seed differs")
    trace = record["metadata"]["trace"]
    if not trace:
        if seed not in request["stop_token_ids"] or len(canonical["response_ids"]) != 1:
            raise ValueError("Missing first round without seed EOS")
        proposals = []
    else:
        block = trace[0]
        if (
            block["output_start"] != 0
            or block["anchor_token"] != seed
            or len(block["draft_tokens"]) != 15
        ):
            raise ValueError("Not the original first native block")
        proposals = block["draft_tokens"]
    return dict(
        request_id=request["request_id"],
        prompt_ids=request["prompt_ids"],
        seed_token=seed,
        start_digest=digest(request["prompt_ids"] + [seed]),
        draft_tokens=proposals,
        checkpoint_id=record.get("checkpoint_id"),
        canonical_binding_digest=canonical["binding_digest"],
    )


def _state(item, request, canonical):
    seed = canonical["response_ids"][0]
    if (
        item["request_id"] != request["request_id"]
        or item["prompt_ids"] != request["prompt_ids"]
        or item["seed_token"] != seed
        or item["start_digest"] != digest(request["prompt_ids"] + [seed])
        or item["canonical_binding_digest"] != canonical["binding_digest"]
    ):
        raise ValueError("Unmatched common start or canonical binding")
    slots = canonical["response_ids"][1:16]
    proposals = item["draft_tokens"]
    if not slots:
        if proposals or seed not in request["stop_token_ids"]:
            raise ValueError("Invalid seed-EOS first round")
        return dict(prefix=0, first_error=None, stratum="seed_eos", matches=[])
    if len(proposals) != 15 or any(type(t) is not int or t < 0 for t in proposals):
        raise ValueError("Require15 native proposals")
    # Canonical includes its terminal EOS, but never a post-terminal slot.
    if any(t in request["stop_token_ids"] for t in slots[:-1]):
        raise ValueError("Malformed canonical terminal")
    matches = [a == b for a, b in zip(proposals, slots)]
    first = next((i for i, v in enumerate(matches) if not v), None)
    prefix = len(matches) if first is None else first
    group = (
        "all_correct"
        if first is None
        else (
            "successor_ineligible"
            if first + 1 >= len(matches)
            else "successor_high" if matches[first + 1] else "successor_low"
        )
    )
    return dict(prefix=prefix, first_error=first, stratum=group, matches=matches)


def analyze(parent, children, requests, canonicals):
    """Parents/children keyed by request ID; children hold each real arm's observation."""
    ids = [r["request_id"] for r in requests]
    if (
        len(ids) != 64
        or len(set(ids)) != 64
        or set(parent) != set(ids)
        or set(children) != set(ids)
    ):
        raise ValueError(
            "Require the exact prespecified64 matched starts without filtering"
        )
    details = []
    groups = {}
    for request in requests:
        rid = request["request_id"]
        reference = canonicals[rid]
        p = _state(parent[rid], request, reference)
        if set(children[rid]) != set(METHODS):
            raise ValueError("Missing real child arm")
        for arm in METHODS:
            c = _state(children[rid][arm], request, reference)
            if len(c["matches"]) != len(p["matches"]):
                raise ValueError("Different valid canonical slots")
            f = p["first_error"]
            before = len(p["matches"]) if f is None else f
            broken = any(not v for v in c["matches"][:before])
            repaired = f is not None and all(c["matches"][: f + 1])
            delta = c["prefix"] - p["prefix"]
            row = dict(
                request_id=rid,
                method=arm,
                parent_stratum=p["stratum"],
                parent_first_error=f,
                valid_proposal_slots=len(p["matches"]),
                old_error_present=f is not None,
                old_first_error_repaired_with_prefix_intact=repaired,
                earlier_correct_prefix_broken=broken,
                parent_accepted_prefix=p["prefix"],
                child_accepted_prefix=c["prefix"],
                prefix_delta=delta,
                prefix_tokens_gained=max(delta, 0),
                prefix_tokens_lost=max(-delta, 0),
                tie=delta == 0,
            )
            details.append(row)
            key = (p["stratum"], arm)
            g = groups.setdefault(
                key,
                dict(
                    parent_stratum=key[0],
                    method=arm,
                    requests=0,
                    old_error_denominator=0,
                    repair_count=0,
                    prefix_preservation_denominator=0,
                    prefix_broken_count=0,
                    prefix_tokens_gained=0,
                    prefix_tokens_lost=0,
                    wins=0,
                    losses=0,
                    ties=0,
                ),
            )
            g["requests"] += 1
            g["old_error_denominator"] += f is not None
            g["repair_count"] += repaired
            g["prefix_preservation_denominator"] += before > 0
            g["prefix_broken_count"] += broken
            g["prefix_tokens_gained"] += max(delta, 0)
            g["prefix_tokens_lost"] += max(-delta, 0)
            g["wins"] += delta > 0
            g["losses"] += delta < 0
            g["ties"] += delta == 0
    return dict(
        status="passed",
        requests=64,
        parent_strata={
            name: sum(
                _state(parent[r["request_id"]], r, canonicals[r["request_id"]])[
                    "stratum"
                ]
                == name
                for r in requests
            )
            for name in (
                "successor_high",
                "successor_low",
                "successor_ineligible",
                "all_correct",
                "seed_eos",
            )
        },
        groups=list(groups.values()),
        rows=details,
        scope="Common request start only; not a decomposition of full-request call differences",
        successor_labels="Canonical continuation at the common start, including EOS and excluding post-terminal slots",
        model_work=0,
    )
