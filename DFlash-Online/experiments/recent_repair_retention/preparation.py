"""Prospective repair identities and their frozen, presentation-specific exposure.

Serving history identifies a decision; the real training forward determines
whether that exact decision can receive the added loss. Neither stage reads an
event-seven outcome. Public PR42 ``H``/``j`` columns are deliberately unused.
"""
from __future__ import annotations

from collections import Counter
import math


def truth(value):
    """Parse retained CSV booleans without treating the string 'False' as true."""
    if value in (True, "True", "true", 1):
        return True
    if value in (False, "False", "false", 0):
        return False
    raise ValueError(f"Expected an explicit retained boolean, got {value!r}")


def slot_key(row):
    return (str(row["request_id"]), int(row["canonical_start"]),
            int(row["proposal_slot"]))


def slot_id(key):
    return f"{key[0]}|{key[1]}|{key[2]}"


def canonical_start(sequence_anchor, prompt_length):
    """An overlapping block at another start remains a different identity."""
    start = int(sequence_anchor) - int(prompt_length)
    if start < 0:
        raise ValueError("Training anchor precedes the supervised response")
    return start


def recent_slots(all_start_rows):
    """Select only K5's genuine first rejection subsequently passed by K6."""
    result, seen = [], set()
    for row in all_start_rows:
        key = str(row["request_id"]), int(row["canonical_start"])
        if key in seen:
            raise ValueError("Duplicate native canonical block identity")
        seen.add(key)
        a5, a6, valid = map(int, (row["a5"], row["a6"], row["valid_proposals"]))
        if not 0 <= a5 <= valid <= 15 or not 0 <= a6 <= valid:
            raise ValueError("Invalid clipped native proposal counts")
        if (truth(row["parity_valid"]) and truth(row["genuine_k5_rejection"])
                and a6 > a5):
            j = a5 + 1
            if not 1 <= j <= valid:
                raise ValueError("A genuine rejection must be a valid non-seed proposal")
            result.append(dict(request_id=key[0], canonical_start=key[1],
                proposal_slot=j, j=j, a5=a5, a6=a6, valid_proposals=valid,
                slot_id=slot_id((*key, j)), kind="recent"))
    return sorted(result, key=slot_key)


def candidate_slots(all_start_rows):
    """Include established slots that could satisfy the structural match.

    Margin and two-pass eligibility matching happen only after frozen training
    observations are available. Candidate generation does not guess a margin.
    """
    rows = list(all_start_rows)
    recent = recent_slots(rows)
    recent_keys = {slot_key(r) for r in recent}
    wanted = {}
    for row in recent:
        wanted.setdefault((row["request_id"], row["a6"]), set()).add(row["proposal_slot"])
    established = []
    for row in rows:
        if not truth(row["parity_valid"]):
            continue
        request, start = str(row["request_id"]), int(row["canonical_start"])
        a5, a6 = int(row["a5"]), int(row["a6"])
        for j in sorted(wanted.get((request, a6), ())):
            key = request, start, j
            if a5 >= j and a6 >= j and key not in recent_keys:
                established.append(dict(request_id=request, canonical_start=start,
                    proposal_slot=j, j=j, a5=a5, a6=a6,
                    valid_proposals=int(row["valid_proposals"]),
                    slot_id=slot_id(key), kind="established"))
    return dict(recent=recent, established=sorted(established, key=slot_key))


def verify_pr42_tag_sanity(all_start_rows):
    rows = list(all_start_rows)
    tags = recent_slots(rows)
    example = [r for r in tags if (r["request_id"], r["canonical_start"]) == ("train_00768", 3)]
    actual = (len(rows), len(tags), len({r["request_id"] for r in tags}))
    if actual != (4877, 828, 128) or len(example) != 1 or example[0]["proposal_slot"] != 7:
        raise ValueError(f"PR42 prospective-tag sanity failed: {actual}, example={example}")
    return dict(status="passed", native_starts=4877, recent_slots=828,
                requests=128, example_slot=7)


def join_parent_occurrences(all_start_rows, native_weight_rows, parent_margin_rows):
    """Join exact block/slot identities, preserving each pass's eligibility.

    ``parent_margin_rows`` is emitted by the frozen native-layout observer and
    contains scalar fields: presentation/pass/request/start/anchor/slot, margin,
    weight, canonical_token, correct_prefix, valid, native_correct and label_match.
    The observer binds label_match to the retained private canonical response.
    Missing sampled positions are exclusions; identity conflicts are failures.
    """
    candidates = candidate_slots(all_start_rows)
    weights, presentations = {}, {}
    for row in native_weight_rows:
        pid, request = int(row["presentation_id"]), str(row["request_id"])
        binding = request, int(row["pass_index"])
        if pid in presentations and presentations[pid] != binding:
            raise ValueError("Presentation identity conflict in native weights")
        presentations[pid] = binding
        key = pid, request, int(row["canonical_start"])
        if key in weights:
            raise ValueError("Duplicate kept training block in native weights")
        weights[key] = row
    margins = {}
    for row in parent_margin_rows:
        key = int(row["presentation_id"]), *slot_key(row)
        if key in margins:
            raise ValueError("Duplicate frozen parent slot observation")
        margins[key] = row
    by_request = {}
    for pid, (request, pass_index) in presentations.items():
        by_request.setdefault(request, []).append((pid, pass_index))
    accepted = {"recent": [], "established": []}
    audit = []
    for kind, slots in candidates.items():
        for slot in slots:
            request, start, j = slot_key(slot)
            if request not in by_request:
                raise ValueError("Native schedule lacks a prospective request")
            for pid, pass_index in sorted(by_request[request]):
                status = "eligible"
                native = weights.get((pid, request, start))
                observed = margins.get((pid, request, start, j))
                if native is None:
                    status = "not_sampled_or_not_kept"
                elif int(native["correct_prefix"]) < j:
                    status = "frozen_training_prefix_disagreement"
                elif int(native["valid_proposals"]) < j:
                    status = "invalid_native_position"
                elif observed is None:
                    raise ValueError(f"Missing required frozen margin: {(pid, request, start, j)}")
                else:
                    if (int(observed["pass_index"]) != pass_index
                            or int(observed["anchor_ordinal"]) != int(native["anchor_ordinal"])
                            or int(observed["correct_prefix"]) != int(native["correct_prefix"])):
                        raise ValueError("Frozen native-layout observation does not reproduce PR42 identity/prefix")
                    if not truth(observed["label_match"]):
                        raise ValueError("Frozen training label differs from private canonical token")
                    if not truth(observed["valid"]):
                        status = "invalid_native_position"
                    elif not truth(observed["native_correct"]):
                        raise ValueError("Correct-prefix observation contains a wrong native argmax")
                    else:
                        margin, weight = float(observed["margin"]), float(observed["weight"])
                        if not math.isfinite(margin) or margin < 0 or not math.isfinite(weight) or weight <= 0:
                            raise ValueError("Invalid correct-parent margin or positional weight")
                        row = dict(observed, **slot, presentation_id=pid, pass_index=pass_index,
                                   margin=margin, weight=weight, floor=0.50 * min(margin, 1.0))
                        accepted[kind].append(row)
                audit.append(dict(slot, presentation_id=pid, pass_index=pass_index, status=status))
    for rows in accepted.values():
        rows.sort(key=lambda r: (r["presentation_id"], r["anchor_ordinal"], r["proposal_slot"]))
    return dict(**accepted, eligibility=audit,
        exclusion_counts=dict(Counter(r["status"] for r in audit if r["status"] != "eligible")))


def prepare_assignments(all_start_rows, native_weight_rows, parent_margin_rows):
    from .matching import match_control
    joined = join_parent_occurrences(all_start_rows, native_weight_rows, parent_margin_rows)
    matched = match_control(joined["recent"], joined["established"])
    return dict(matched, eligibility=joined["eligibility"],
                exclusion_counts=joined["exclusion_counts"])


def prepare_private_candidates(all_starts_path, retained_root, output_path):
    """Bind candidate labels to authenticated native K6 verifier trajectories.

    Only labels actually accepted by K6 are needed. For them the saved target
    prediction at j and the complete canonical response token must agree.
    """
    import csv
    from pathlib import Path
    from experiments.lost_repair.prepare import identity, read, write

    all_starts_path, retained_root, output_path = map(Path, (all_starts_path,retained_root,output_path))
    if output_path.exists():
        raise FileExistsError("Keep the previously sealed private candidate ledger")
    with all_starts_path.open() as handle:
        native_rows = list(csv.DictReader(handle))
    sanity = verify_pr42_tag_sanity(native_rows)
    candidates = candidate_slots(native_rows)
    inventory = read(retained_root/"delivery/COMPACT_MANIFEST.json")
    traces, pins = {}, {str(all_starts_path):identity(all_starts_path)}
    plan = read(retained_root/"sealed/PLAN.json")
    requests = {r["request_id"]:r for r in plan["requests"]}
    output = []
    for kind, slots in candidates.items():
        for slot in slots:
            request, start, j = slot_key(slot)
            if request not in traces:
                relative = f"campaign_001/census/{request}/K6_CAPTURE.json"
                path = retained_root/relative
                actual = identity(path)
                if actual != inventory[relative]:
                    raise ValueError("Retained native K6 trace identity changed")
                trace = read(path)
                if (trace["prompt_ids"] != requests[request]["prompt_ids"]
                        or trace["checkpoint_id"] != plan["checkpoints"]["K6"]["export"]
                        or trace["max_new_tokens"] != 2048
                        or trace["metadata"]["block_size"] != 16):
                    raise ValueError("Native K6 trace does not bind the prescribed input/checkpoint")
                traces[request] = (trace,{int(b["output_start"]):b for b in trace["metadata"]["trace"]})
                pins[str(path)] = actual
            trace, blocks = traces[request]
            block = blocks[start]
            if (int(block["accepted_draft_tokens"]) != slot["a6"]
                    or int(block["anchor_token"]) != trace["response_ids"][start]
                    or j > block["accepted_draft_tokens"]):
                raise ValueError("Prospective slot is not accepted by the exact native parent block")
            y = int(trace["response_ids"][start+j])
            if (int(block["target_argmax"][j-1]) != y
                    or int(block["draft_tokens"][j-1]) != y):
                raise ValueError("Canonical token differs from accepted native verifier decision")
            output.append(dict(slot,canonical_token=y))
    output.sort(key=slot_key)
    result = dict(schema="recent-repair-retention-private-candidates-v1",slots=output,
        sanity=sanity,counts={k:len(v) for k,v in candidates.items()},
        source_files=pins,labels="K6 accepted native target_argmax, checked against complete canonical tokens",
        event7_outcomes_used=False,model_work=0)
    write(output_path,result)
    return dict(sanity=sanity,counts=result["counts"],identity=identity(output_path))
