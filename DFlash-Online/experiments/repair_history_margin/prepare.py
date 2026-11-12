"""Seal the original-order PR39 panel and its input-only exclusions.

This module reads only source questions, serving inputs, canonical references,
and K+5 traces. It never opens a K+6 outcome record or loads a model.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from experiments.lost_repair.prepare import (
    MANIFEST_IDENTITY, WEIGHT_HASHES, checked_identity, identity, read, write,
)
from experiments.mathqa_transfer.data import normalize, TOKENIZER_HASHES
from experiments.successor_shuffle.data import verify_panel
from experiments.survival_confirmation.data import REVISION, SOURCE_HASHES, SUFFIX, digest, render_question

COUNT = 192


def text_digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def match_keys(row):
    """Opaque request IDs are not a substitute for source or content identity."""
    keys = set()
    for name in ("source_id", "dataset_id"):
        if row.get(name):
            keys.add(("source_id", row[name]))
    for name in ("normalized_question_sha256", "rendered_prompt_sha256", "prompt_ids_sha256"):
        if row.get(name):
            keys.add((name, row[name]))
    return keys


def select_panel(panel, exclusions, count=COUNT):
    """Take the first distinct eligible inputs, keeping every exclusion visible."""
    excluded = defaultdict(list)
    for entry in exclusions:
        for key in match_keys(entry):
            excluded[key].append(entry)
    selected, audit, seen = [], [], {}
    for original_ordinal, source in enumerate(panel["requests"]):
        row = deepcopy(source)
        keys = match_keys(row)
        hits = {(e["reason"], e["request_id"]): e
                for key in keys for e in excluded.get(key, ())}
        duplicates = sorted({seen[key] for key in keys if key in seen})
        status = "selected"
        if hits:
            status = "excluded"
        elif duplicates:
            status = "duplicate"
        elif len(selected) >= count:
            status = "eligible_beyond_limit"
        if not hits and not duplicates:
            for key in keys:
                seen[key] = row["request_id"]
        audit.append(dict(
            request_id=row["request_id"], source_id=row["source_id"],
            original_panel_ordinal=original_ordinal, status=status,
            matches=[dict(reason=e["reason"], request_id=e["request_id"],
                          shared_keys=[list(k) for k in sorted(keys & match_keys(e))])
                     for _, e in sorted(hits.items())],
            duplicate_of=duplicates,
        ))
        if status == "selected":
            row["original_panel_ordinal"] = original_ordinal
            row["selection_rank"] = len(selected)
            selected.append(row)
    return dict(
        schema="repair-history-margin-panel-v1", requests=selected,
        count=len(selected), requested_count=count, shortfall=max(0, count-len(selected)),
        source_panel_count=len(panel["requests"]), selection="first distinct eligible inputs in original PR39 order",
        source_revision=REVISION, selection_uses_outcomes=False,
        selection_uses_lengths=False, selection_uses_reference_availability=False,
        exclusion_audit=audit, exclusion_status_counts=dict(Counter(r["status"] for r in audit)),
        exclusion_reason_counts=dict(Counter(reason for r in audit
            for reason in {m["reason"] for m in r["matches"]})),
        prior_exposure="PR39 TRUE/KPLUS5 serving plus each request's retained prior-project exposure; not globally unused",
    )


def question_from_prompt(ids, tokenizer):
    text = tokenizer.decode(ids, skip_special_tokens=False)
    start, end = "<|im_start|>user\n", "<|im_end|>"
    if not text.startswith(start) or end not in text[len(start):]:
        raise ValueError("Expected one qualified user prompt, not a continuation")
    question, tail = text[len(start):].split(end, 1)
    if "<|im_start|>user" in tail:
        raise ValueError("Unexpected additional user message")
    if question.endswith(SUFFIX):
        question = question[:-len(SUFFIX)]
    return question, text


def exclusion_rows(rows, *, reason, split, questions, tokenizer):
    lookup = defaultdict(list)
    for index, question in enumerate(questions):
        lookup[normalize(question)].append(index)
    result = []
    for row in rows:
        question, text = question_from_prompt(row["prompt_ids"], tokenizer)
        indices = lookup[normalize(question)]
        if len(indices) != 1:
            raise ValueError(f"Cannot uniquely resolve {reason} request {row['request_id']} to {split} source")
        index = indices[0]
        result.append(dict(
            reason=reason, request_id=row["request_id"], source_id=f"gsm8k:main:{split}:{index}",
            dataset_id=row.get("dataset_id"), source_revision=REVISION,
            normalized_question_sha256=text_digest(normalize(question)),
            rendered_prompt_sha256=text_digest(text), prompt_ids_sha256=digest(row["prompt_ids"]),
        ))
    return result


def public_panel(panel):
    return dict(panel, requests=[{k:v for k,v in r.items() if k != "prompt_ids"} for r in panel["requests"]])


def authenticate_trace(trace, request, canonical, checkpoint):
    """Bind an existing native K+5 walk without using it to select the panel."""
    if (trace["checkpoint_id"] != checkpoint
            or trace["prompt_ids"] != request["prompt_ids"]
            or trace["response_ids"] != canonical["response_ids"]
            or trace["finish_reason"] != canonical["finish_reason"]
            or trace["max_new_tokens"] != 2048
            or trace["stop_token_ids"] != [151645, 151643]
            or trace["metadata"]["block_size"] != 16
            or trace["metadata"]["context_limit"] != 4096):
        raise ValueError("Retained K+5 native trace does not bind the selected input/reference")
    if (canonical["prompt_ids"] != request["prompt_ids"] or not canonical["verified"]):
        raise ValueError("Retained canonical reference is not verified for this exact input")
    return dict(rounds=len(trace["metadata"]["trace"]), output_tokens=len(trace["response_ids"]),
                finish_reason=trace["finish_reason"])


def prepare_inputs(*, research_root, code_root, output, tokenizer_path):
    """Create a private immutable input bundle; source/code sealing follows separately."""
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    research, code, output = map(Path, (research_root, code_root, output))
    if output.exists():
        raise FileExistsError("Preserve the existing input seal; use a new destination")
    if output.resolve().is_relative_to(code.resolve().parent):
        raise ValueError("Private inputs must remain outside the Git checkout")
    pr39 = research / "dflash_successor_shuffle_20261111/delivery/retained"
    inventory_path = pr39 / "COMPACT_DELIVERY_MANIFEST.json"
    inventory = read(inventory_path)
    pinned = {str(inventory_path): identity(inventory_path)}

    def retained(relative):
        path = pr39 / relative
        checked_identity(path, inventory["files"][relative])
        pinned[str(path)] = identity(path)
        return read(path)

    panel = retained("sealed/PANEL.json")
    verify_panel(panel)
    # Public original-order panel is the handoff's independently named source.
    public_path = code / "reports/dflash_successor_shuffle_20261111/PANEL_MANIFEST.json"
    published = read(public_path)
    if [{k:v for k,v in r.items() if k != "prompt_ids"} for r in panel["requests"]] != published["requests"]:
        raise ValueError("Private PR39 panel differs from its public source manifest")
    pinned[str(public_path)] = identity(public_path)
    source_manifest_path = research / "projected_target_feasibility_20261022/data/approved/manifest.json"
    source_manifest = read(source_manifest_path)
    pinned[str(source_manifest_path)] = identity(source_manifest_path)
    questions = {}
    for split in ("train", "test"):
        entry = source_manifest["sources"]["gsm8k_" + split]
        path = Path(entry["local_path"])
        actual = identity(path)
        if entry["revision"] != REVISION or actual["sha256"] != SOURCE_HASHES[split]:
            raise ValueError("Pinned GSM8K source changed")
        pinned[str(path)] = actual
        questions[split] = pq.read_table(path, columns=["question"])["question"].to_pylist()
    tokenizer_path = Path(tokenizer_path)
    for filename, sha in TOKENIZER_HASHES.items():
        actual = identity(tokenizer_path / filename)
        if actual["sha256"] != sha:
            raise ValueError("Pinned tokenizer changed")
        pinned[str(tokenizer_path / filename)] = actual
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    for row in panel["requests"]:
        question = questions["test"][row["source_row"]]
        ids, text = render_question(question, tokenizer)
        if (ids != row["prompt_ids"] or text_digest(text) != row["rendered_prompt_sha256"]
                or text_digest(normalize(question)) != row["normalized_question_sha256"]):
            raise ValueError("Original PR39 prompt/source reconstruction failed")

    original_path = research / "dflash_kplus_20261103/production_001/sealed/ORIGINAL_MANIFEST.json"
    checked_identity(original_path, MANIFEST_IDENTITY)
    original = read(original_path)
    pinned[str(original_path)] = identity(original_path)
    train = original["train_requests"][:768]
    if len(train) != 768 or any(r["event_id"] != f"event_{i//128+1:03d}" for i,r in enumerate(train)):
        raise ValueError("Expected exact original event1–6 training windows")
    discovery_path = research / "dflash_lost_repair_20261114/delivery/retained/sealed/PLAN.json"
    discovery = read(discovery_path)
    pinned[str(discovery_path)] = identity(discovery_path)
    if discovery["requests"] != [dict(r, stop_token_ids=[151645,151643]) for r in original["train_requests"][768:896]]:
        raise ValueError("PR42 discovery batch differs from its exact original window")
    pr34_path = research / "dflash_verification_transition_replay_20261110/native_001/completed_evidence/PLAN.json"
    pr34 = read(pr34_path)
    pinned[str(pr34_path)] = identity(pr34_path)
    discovery_test = [r["request"] for r in pr34["requests"] if r["dataset"] == "gsm8k"]
    if len(discovery_test) != 64:
        raise ValueError("Expected all64 PR34/40 GSM8K discovery requests")
    exclusions = exclusion_rows(train, reason="events_1_6_training", split="train", questions=questions["train"], tokenizer=tokenizer)
    exclusions += exclusion_rows(discovery["requests"], reason="PR42_discovery_batch", split="train", questions=questions["train"], tokenizer=tokenizer)
    exclusions += exclusion_rows(discovery_test, reason="PR34_PR40_discovery_panel", split="test", questions=questions["test"], tokenizer=tokenizer)
    selected = select_panel(panel, exclusions)
    output.mkdir(parents=True)
    # The selection is durable before any canonical or native trace is opened.
    write(output / "PANEL.json", selected)
    write(output / "PANEL_MANIFEST.json", public_panel(selected))
    write(output / "EXCLUSIONS.json", dict(rows=exclusions, counts=dict(Counter(r["reason"] for r in exclusions))))

    plan39 = retained("sealed/PLAN.json")
    checkpoint = plan39["checkpoints"]["TRUE"]
    if checkpoint["weights_identity"]["sha256"] != WEIGHT_HASHES["K5"]:
        raise ValueError("PR39 TRUE is not the specified native K+5")
    canonicals = retained("evaluation/CANONICAL.json")
    requests, trace_counts = [], Counter()
    for request in selected["requests"]:
        ordinal = request["original_panel_ordinal"]
        relative = f"evaluation/{ordinal:04d}_r0_TRUE.json"
        path = pr39 / relative
        canonical = canonicals[request["request_id"]]
        trace_receipt = None
        if path.exists():
            trace = retained(relative)
            details = authenticate_trace(trace, request, canonical, checkpoint["checkpoint_path"])
            trace_receipt = dict(local_path=str(path), path=str(Path(inventory["root"])/relative),
                                 **identity(path), **details)
            trace_counts["available"] += 1
            trace_counts["rounds"] += details["rounds"]
        else:
            trace_counts["missing"] += 1
        requests.append(dict(dataset="gsm8k", source_id=request["source_id"], request_id=request["request_id"],
                             selection_rank=request["selection_rank"], request=request, canonical=canonical,
                             native_K5=trace_receipt))
    write(output / "INPUTS.json", dict(schema="repair-history-margin-inputs-v1", requests=requests,
        panel_identity=identity(output/"PANEL.json"), checkpoints={"K5":checkpoint},
        traces=dict(trace_counts), source_files=pinned,
        excluded_inputs=len(exclusions), outcome_access="No K+6 outcomes read; K+5 traces bound after panel seal"))
    write(output / "INPUT_SEAL.json", dict(status="sealed_input_only", count=len(requests),
        files={p.name:identity(p) for p in sorted(output.iterdir()) if p.is_file()},
        scientific_code_seal="Runner must bind clean execution commit/config/analysis before model work"))
    return dict(requests=len(requests), shortfall=selected["shortfall"],
                selection_counts=selected["exclusion_status_counts"], traces=dict(trace_counts),
                input_seal=identity(output/"INPUT_SEAL.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-root", required=True)
    parser.add_argument("--code-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    print(json.dumps(prepare_inputs(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
