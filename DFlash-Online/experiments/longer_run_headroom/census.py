"""Parent-only census and input-bound sampling, before loading child outcomes."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

from experiments.verification_transition.data import identity, read, write

DATASETS = ("gsm8k", "mathqa")
EXPECTED_ROUNDS = {"gsm8k": 2761, "mathqa": 4330}
SOURCE_COMMIT = "d7b1311e7b2f910e935bf81b4e40282767d7540c"


def digest(value):
    return hashlib.sha256(
        json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def classify(proposals, canonical, seed, stops):
    """Score canonical followers, never verifier logits after a rejection.

    Inputs begin after the known seed. A canonical EOS is a valid label and
    ends the available suffix; a saved cap boundary merely censors it.
    """
    if seed in stops:
        return dict(group="seed_eos", j=None, v=0, R=0, valid_slots=0, prefix=0)
    labels = list(canonical[: len(proposals)])
    for i, token in enumerate(labels):
        if token in stops:
            labels = labels[: i + 1]
            break
    matches = [p == y for p, y in zip(proposals, labels)]
    error = next((i for i, ok in enumerate(matches) if not ok), None)
    if error is None:
        return dict(
            group="all_correct",
            j=None,
            v=0,
            R=0,
            valid_slots=len(labels),
            prefix=len(labels),
        )
    v = len(labels) - error - 1
    run = 0
    for ok in matches[error + 1 :]:
        if not ok:
            break
        run += 1
    group = (
        "insufficient_lookahead"
        if v < 2
        else "R0" if run == 0 else "SHORT" if run == 1 else "LONG"
    )
    return dict(
        group=group, j=error + 1, v=v, R=run, valid_slots=len(labels), prefix=error
    )


def validate_walk(record, item):
    request, canonical = item["request"], item["canonical"]
    if (
        record["record_id"] != item["request_id"]
        or record["prompt_ids"] != request["prompt_ids"]
        or record["response_ids"] != canonical["response_ids"]
        or record["finish_reason"] != canonical["finish_reason"]
        or record["max_new_tokens"] != request["max_new_tokens"]
        or record["stop_token_ids"] != request["stop_token_ids"]
    ):
        raise ValueError("Walk identity/canonical contract changed")
    response = record["response_ids"]
    output_start = 0
    trace = record["metadata"]["trace"]
    if len(trace) != record["metadata"]["verification_calls"]:
        raise ValueError("Walk call count differs from trace")
    for block in trace:
        start = block["output_start"]
        end = start + 1 + block["committed_progress"]
        if (
            start != output_start
            or block["anchor_token"] != response[start]
            or len(block["draft_tokens"]) != 15
            or block["committed_tokens"] != response[start + 1 : end]
            or block["positions"]
            != list(
                range(
                    len(request["prompt_ids"]) + start,
                    len(request["prompt_ids"]) + start + 16,
                )
            )
        ):
            raise ValueError("Native block does not align with its canonical walk")
        accepted = block["accepted_draft_tokens"]
        if (
            block["draft_tokens"][:accepted]
            != response[start + 1 : start + 1 + accepted]
        ):
            raise ValueError("Accepted proposals differ from canonical tokens")
        output_start += block["committed_progress"]
    if output_start + 1 != len(response):
        raise ValueError("Committed progress does not cover canonical output")


def parent_rows(record, item):
    validate_walk(record, item)
    response = record["response_ids"]
    rows = []
    for ordinal, block in enumerate(record["metadata"]["trace"], 1):
        start = block["output_start"]
        row = dict(
            dataset=item["dataset"],
            request_id=item["request_id"],
            selection_rank=item["selection_rank"],
            parent_round_ordinal=ordinal,
            opening=ordinal == 1,
            output_start=start,
            parent_accepted=block["accepted_draft_tokens"],
            parent_native_accepted=block["native_accepted_draft_tokens"],
            remaining_budget=record["max_new_tokens"] - start - 1,
            canonical_prefix_sha256=digest(
                record["prompt_ids"] + response[: start + 1]
            ),
            parent_block_sha256=digest(block),
            **classify(
                block["draft_tokens"],
                response[start + 1 :],
                block["anchor_token"],
                record["stop_token_ids"],
            ),
        )
        row["selection_sha256"] = digest(
            ["kplus-run-probe-v1", row["dataset"], row["request_id"], ordinal]
        )
        rows.append(row)
    return rows


def sample(rows):
    selected = []
    groups = []
    for dataset in DATASETS:
        for group in ("SHORT", "LONG"):
            eligible = [
                r
                for r in rows
                if r["dataset"] == dataset and not r["opening"] and r["group"] == group
            ]
            chosen = sorted(
                eligible,
                key=lambda r: (
                    r["selection_sha256"],
                    r["request_id"],
                    r["parent_round_ordinal"],
                ),
            )[:64]
            fraction = len(chosen) / len(eligible) if eligible else 0.0
            selected.extend(dict(r, inclusion_fraction=fraction) for r in chosen)
            groups.append(
                dict(
                    dataset=dataset,
                    group=group,
                    census=len(eligible),
                    selected=len(chosen),
                    inclusion_fraction=fraction,
                    selected_requests=len({r["request_id"] for r in chosen}),
                )
            )
    return selected, groups


def census_summary(rows):
    result = {}
    for dataset in DATASETS:
        own = [r for r in rows if r["dataset"] == dataset]
        requests = sorted({r["request_id"] for r in own})
        per_request = []
        for request in requests:
            later = [r for r in own if r["request_id"] == request and not r["opening"]]
            c = Counter(r["group"] for r in later)
            per_request.append(
                dict(
                    request_id=request,
                    later_rounds=len(later),
                    counts=dict(c),
                    both_groups=c["SHORT"] > 0 and c["LONG"] > 0,
                )
            )
        cells = Counter((r["group"], r["j"], r["v"]) for r in own if not r["opening"])
        result[dataset] = dict(
            parent_rounds=len(own),
            requests=len(requests),
            opening_counts=dict(Counter(r["group"] for r in own if r["opening"])),
            later_rounds=sum(not r["opening"] for r in own),
            later_counts=dict(Counter(r["group"] for r in own if not r["opening"])),
            per_request=per_request,
            cells=[
                dict(group=g, j=j, v=v, count=n)
                for (g, j, v), n in sorted(
                    cells.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0, kv[0][2])
                )
            ],
            requests_with_both=sum(r["both_groups"] for r in per_request),
        )
    return result


def build(evidence, output):
    started = time.perf_counter()
    evidence, output = Path(evidence), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    inventory = read(evidence / "EVIDENCE_MANIFEST.json")["files"]
    checked = []

    def authenticated(relative):
        path = evidence / relative
        if identity(path) != inventory[relative]:
            raise ValueError("PR34 compact artifact changed: " + relative)
        checked.append(dict(path=str(path), **inventory[relative]))
        return read(path)

    plan = authenticated("PLAN.json")
    if plan["execution_commit"] != SOURCE_COMMIT:
        raise ValueError("Wrong PR34 execution")
    rows = []
    for item in plan["requests"]:
        # Child files are deliberately never opened in this stage.
        relative = f'campaign/raw/{item["dataset"]}/{item["selection_rank"]:03d}/KPLUS4_native.json'
        record = authenticated(relative)
        if (
            record["checkpoint_id"]
            != plan["checkpoints"][item["dataset"]]["KPLUS4"]["export"]
        ):
            raise ValueError("Parent checkpoint identity changed")
        rows.extend(parent_rows(record, item))
    summary = census_summary(rows)
    for dataset, expected in EXPECTED_ROUNDS.items():
        if (
            summary[dataset]["parent_rounds"] != expected
            or summary[dataset]["requests"] != 64
        ):
            raise ValueError(
                "Parent census does not reproduce declared round/request counts"
            )
        if summary[dataset]["later_rounds"] != expected - 64:
            raise ValueError("Opening-round exclusion changed")
    selected, groups = sample(rows)
    write(output / "CENSUS_ROWS.json", rows)
    write(output / "CENSUS.json", summary)
    write(
        output / "SELECTED_STARTS.json",
        dict(
            status="sealed_before_child_outcome_reads",
            source_execution_commit=SOURCE_COMMIT,
            key=["kplus-run-probe-v1", "dataset", "request_id", "parent_round_ordinal"],
            per_group_limit=64,
            groups=groups,
            starts=selected,
        ),
    )
    write(
        output / "SEAL.json",
        dict(
            source_execution_commit=SOURCE_COMMIT,
            created_utc=__import__("datetime")
            .datetime.now(__import__("datetime").timezone.utc)
            .isoformat(),
            files={
                name: identity(output / name)
                for name in ["CENSUS_ROWS.json", "CENSUS.json", "SELECTED_STARTS.json"]
            },
            inspected_files=checked,
            child_outcome_files_read=0,
            cpu_wall_s=time.perf_counter() - started,
        ),
    )
    return dict(
        groups=groups,
        census={
            d: {
                k: v
                for k, v in s.items()
                if k
                in (
                    "parent_rounds",
                    "later_rounds",
                    "later_counts",
                    "opening_counts",
                    "requests_with_both",
                )
            }
            for d, s in summary.items()
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.evidence, args.output), indent=2))
