"""Prepare the matched GSM8K stream without generating or training anything.

The preferred input is the successful reference's response archive: only its
prompt token IDs and request identities are reused. Responses and answer labels
never enter this manifest. The alternate raw-data path verifies the pinned data
file and every reconstructed prompt against retained reference prompt digests.
Those checks happen during preparation; the runtime validator performs no hashes.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


SCHEMA = "dflash_osd_persistent_manifest_v1"
TRAIN_REQUESTS = 7473
RESPONSES_PER_EVENT = 128
EVENTS = 58
FEEDBACK_REQUESTS = RESPONSES_PER_EVENT * EVENTS
EVAL_REQUESTS = 1024
OUTPUT_CAP = 2048
DATASET_REVISION = "740312add88f781978c0658806c59bc2815b9866"
TARGET_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
TRAIN_DATA_SHA256 = "f8eb683c6e1a00b5fbe3a4fd05806ced4f7c2d10325cb3d362d68ae0209eab81"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _token_ids(value, name):
    _require(isinstance(value, list) and bool(value), f"{name}: nonempty token IDs required")
    _require(all(type(token) is int and token >= 0 for token in value),
             f"{name}: token IDs must be nonnegative integers")
    return list(value)


def _reference_requests(path, expected_count=TRAIN_REQUESTS):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    _require(len(rows) == expected_count, "Training reference request count changed")
    identities = []
    for index, row in enumerate(rows):
        _require(row.get("dataset_index") == index and row.get("source_line_1based") == index + 1,
                 "Training reference order must be the complete chronological source order")
        identity = row.get("source_record_id")
        _require(isinstance(identity, str) and bool(identity), "Missing training dataset identity")
        _require(isinstance(row.get("record_sha256"), str), "Missing original record identity")
        identities.append(identity)
    _require(len(set(identities)) == len(rows), "Duplicate training dataset identity")
    return rows


def load_training_archive(archive, request_manifest, *, expected_count=TRAIN_REQUESTS):
    """Copy original prompt IDs; reject missing, reordered, or mismatched records."""
    archive = Path(archive)
    reference = _reference_requests(request_manifest, expected_count)
    rows = []
    for index, identity in enumerate(reference):
        path = archive / f"{index:05d}.json"
        raw = json.loads(path.read_text())
        for key in ("dataset_index", "source_line_1based", "source_record_id", "record_sha256"):
            _require(raw.get(key) == identity[key], f"Training archive identity differs: {path}, {key}")
        rows.append({
            "dataset_index": index,
            "dataset_id": identity["source_record_id"],
            "prompt_ids": _token_ids(raw.get("prompt_token_ids"), str(path)),
        })
    return rows


def load_training_raw(raw_train, request_manifest, tokenizer, prompt_provenance):
    """Verified fallback when the original prompt archive is unavailable.

    The native normalized file contains answers, but only its user question is
    rendered. The retained ONLINE_ROWS.json hashes verify every reconstructed
    prompt ID sequence before it can enter the manifest.
    """
    data = Path(raw_train).read_bytes()
    _require(hashlib.sha256(data).hexdigest() == TRAIN_DATA_SHA256,
             "Raw training data differs from the pinned normalized GSM8K file")
    reference = _reference_requests(request_manifest)
    raw_rows = [json.loads(line) for line in data.decode().splitlines() if line.strip()]
    _require(len(raw_rows) == TRAIN_REQUESTS, "Raw training request count changed")
    provenance = json.loads(Path(prompt_provenance).read_text())
    original = provenance["arms"]["osd"]
    _require(len(original) == TRAIN_REQUESTS, "Prompt provenance must cover all training requests")
    rows = []
    for index, (raw, identity, evidence) in enumerate(zip(raw_rows, reference, original)):
        _require(raw.get("id") == identity["source_record_id"], "Raw training identity/order changed")
        _require(evidence.get("dataset_index") == index
                 and evidence.get("source_record_id") == identity["source_record_id"],
                 "Prompt provenance identity/order changed")
        messages = raw.get("conversations")
        _require(isinstance(messages, list) and len(messages) == 2
                 and messages[0].get("role") == "user"
                 and messages[1].get("role") == "assistant"
                 and isinstance(messages[0].get("content"), str),
                 "Expected the original single-question GSM8K normalization")
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": messages[0]["content"]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        ids = _token_ids(tokenizer.encode(text, add_special_tokens=False), f"training {index}")
        digest = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
        _require(digest == evidence.get("prompt_token_ids_sha256"),
                 f"Reconstructed prompt differs from the reference at dataset index {index}")
        rows.append({"dataset_index": index, "dataset_id": identity["source_record_id"],
                     "prompt_ids": ids})
    return rows


def load_evaluation_manifest(path, *, expected_count=EVAL_REQUESTS):
    """Keep the original timed request order and IDs; exclude the warmup."""
    source = json.loads(Path(path).read_text())
    _require(source.get("seed") == 42 and source.get("num_timed") == expected_count
             and source.get("max_samples") == expected_count
             and source.get("qualification_only") is False,
             "Expected the completed seed-42 full heldout reference manifest")
    requests = source["requests"]
    _require(len(requests) == expected_count + 1, "Heldout reference request count changed")
    warmup = requests[0]
    _require(warmup.get("kind") == "warmup" and warmup.get("request_index") == 0
             and warmup.get("max_new_tokens") == 64,
             "Expected the original first warmup request")
    rows = []
    for index, raw in enumerate(requests[1:], start=1):
        _require(raw.get("kind") == "timed" and raw.get("request_index") == index,
                 "Heldout timed request order changed")
        _require(raw.get("max_new_tokens") == OUTPUT_CAP, "Heldout output cap changed")
        identity = raw.get("dataset_id")
        _require(isinstance(identity, str) and bool(identity), "Missing heldout dataset identity")
        rows.append({"request_index": index, "dataset_id": identity,
                     "prompt_ids": _token_ids(raw.get("rendered_prompt_token_ids"), f"heldout {index}")})
    _require(len({row["dataset_id"] for row in rows}) == len(rows), "Duplicate heldout dataset identity")
    _require(warmup.get("dataset_id") == rows[-1]["dataset_id"]
             and warmup.get("rendered_prompt_token_ids") == rows[-1]["prompt_ids"],
             "Original warmup/timed wraparound relation changed")
    return rows


def generation_cap(prompt_length, *, cap_policy, train_context_limit):
    """Choose an explicit training-stream output budget without truncating IDs."""
    _require(type(prompt_length) is int and prompt_length > 0, "Positive prompt length required")
    _require(type(train_context_limit) is int and train_context_limit > 0,
             "Positive training context limit required")
    _require(cap_policy in {"remaining", "fixed2048"}, "Unknown output-cap policy")
    cap = min(OUTPUT_CAP, train_context_limit - prompt_length) if cap_policy == "remaining" else OUTPUT_CAP
    _require(cap >= 2, "Prompt leaves fewer than two response tokens; no truncation or resampling")
    _require(prompt_length + cap <= train_context_limit,
             "Fixed 2048 output cap exceeds the training context; increase its explicit limit")
    return cap


def build_manifest(train_rows, eval_rows, *, cap_policy, train_context_limit=2048,
                   serving_context_limit=None, provenance=None):
    """Bind the original request sets to the selected persistent 58-event budget."""
    _require(len(train_rows) == TRAIN_REQUESTS and len(eval_rows) == EVAL_REQUESTS,
             "Full campaign requires exactly 7473 training-stream and 1024 heldout requests")
    train, evaluation = [], []
    for index, row in enumerate(train_rows):
        _require(row.get("dataset_index") == index, "Training source order changed")
        ids = _token_ids(row.get("prompt_ids"), f"training {index}")
        feedback = index < FEEDBACK_REQUESTS
        train.append({
            "request_id": f"train_{index:05d}", "dataset_id": row["dataset_id"],
            "dataset_index": index, "request_index": index + 1, "prompt_ids": ids,
            "max_new_tokens": generation_cap(len(ids), cap_policy=cap_policy,
                                               train_context_limit=train_context_limit),
            "collect_feedback": feedback,
            "event_id": f"event_{index // RESPONSES_PER_EVENT + 1:03d}" if feedback else None,
        })
    for index, row in enumerate(eval_rows, start=1):
        _require(row.get("request_index") == index, "Heldout source order changed")
        evaluation.append({
            "request_id": f"eval_{index:05d}", "dataset_id": row["dataset_id"],
            "request_index": index, "prompt_ids": _token_ids(row.get("prompt_ids"), f"heldout {index}"),
            "max_new_tokens": OUTPUT_CAP, "collect_feedback": False,
        })
    minimum_serving_context = max(len(row["prompt_ids"]) + row["max_new_tokens"]
                                  for row in train + evaluation)
    if serving_context_limit is None:
        serving_context_limit = minimum_serving_context
    manifest = {
        "schema": SCHEMA,
        "policy": {"cap_policy": cap_policy, "train_context_limit": train_context_limit,
                   "serving_context_limit": serving_context_limit,
                   "minimum_serving_context": minimum_serving_context,
                   "requested_output_cap": OUTPUT_CAP, "no_truncation": True,
                   "training_prompts": "original raw question, Qwen no-thinking template",
                   "evaluation_prompts": "exact reference IDs including original reasoning instruction",
                   "extra_warmup_requests": 0},
        "counts": {"train_requests": TRAIN_REQUESTS, "feedback_requests": FEEDBACK_REQUESTS,
                   "serve_only_requests": TRAIN_REQUESTS - FEEDBACK_REQUESTS,
                   "events": EVENTS, "responses_per_event": RESPONSES_PER_EVENT,
                   "eval_requests": EVAL_REQUESTS},
        "provenance": provenance or {},
        "train_requests": train, "eval_requests": evaluation,
    }
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest):
    """Cheap runtime contract checks; no token rewriting or file/tensor hashing."""
    _require(manifest.get("schema") == SCHEMA, "Unknown persistent OSD manifest schema")
    policy = manifest["policy"]
    _require(policy.get("no_truncation") is True and policy.get("extra_warmup_requests") == 0
             and policy.get("requested_output_cap") == OUTPUT_CAP, "Manifest prompt/output policy changed")
    serving_context = policy["serving_context_limit"]
    _require(type(serving_context) is int and serving_context > 0, "Invalid serving context limit")
    train, evaluation = manifest["train_requests"], manifest["eval_requests"]
    expected = {"train_requests": TRAIN_REQUESTS, "feedback_requests": FEEDBACK_REQUESTS,
                "serve_only_requests": TRAIN_REQUESTS - FEEDBACK_REQUESTS,
                "events": EVENTS, "responses_per_event": RESPONSES_PER_EVENT,
                "eval_requests": EVAL_REQUESTS}
    _require(manifest.get("counts") == expected and len(train) == TRAIN_REQUESTS
             and len(evaluation) == EVAL_REQUESTS, "Persistent campaign counts changed")
    for index, row in enumerate(train):
        _require(row.get("dataset_index") == index and row.get("request_index") == index + 1
                 and row.get("request_id") == f"train_{index:05d}", "Training request order changed")
        ids = _token_ids(row.get("prompt_ids"), f"training {index}")
        expected_cap = generation_cap(len(ids), cap_policy=policy["cap_policy"],
                                      train_context_limit=policy["train_context_limit"])
        _require(row.get("max_new_tokens") == expected_cap, "Training request cap differs from policy")
        feedback = index < FEEDBACK_REQUESTS
        _require(row.get("collect_feedback") is feedback, "Training feedback membership changed")
        event = f"event_{index // RESPONSES_PER_EVENT + 1:03d}" if feedback else None
        _require(row.get("event_id") == event, "Training event membership changed")
    for index, row in enumerate(evaluation, start=1):
        _require(row.get("request_index") == index and row.get("request_id") == f"eval_{index:05d}",
                 "Heldout request order changed")
        _token_ids(row.get("prompt_ids"), f"heldout {index}")
        _require(row.get("max_new_tokens") == OUTPUT_CAP and row.get("collect_feedback") is False,
                 "Heldout requests must retain the original cap and supply no feedback")
    for rows, name in ((train, "training"), (evaluation, "heldout")):
        identities = [row.get("dataset_id") for row in rows]
        _require(all(isinstance(identity, str) and identity for identity in identities)
                 and len(set(identities)) == len(rows), f"Invalid or duplicate {name} dataset identity")
    _require(not ({row["dataset_id"] for row in train} & {row["dataset_id"] for row in evaluation}),
             "Training and heldout dataset identities overlap")
    minimum_context = max(len(row["prompt_ids"]) + row["max_new_tokens"] for row in train + evaluation)
    _require(policy.get("minimum_serving_context") == minimum_context and serving_context >= minimum_context,
             "Serving context does not admit every complete prompt and reserved output")
    _require(Counter(row["event_id"] for row in train if row["collect_feedback"])
             == {f"event_{i:03d}": RESPONSES_PER_EVENT for i in range(1, EVENTS + 1)},
             "Each event must contain exactly 128 original chronological responses")
    return {"counts": dict(expected), "cap_policy": policy["cap_policy"],
            "train_context_limit": policy["train_context_limit"],
            "serving_context_limit": serving_context,
            "train_output_cap_min": min(row["max_new_tokens"] for row in train),
            "train_output_cap_max": max(row["max_new_tokens"] for row in train),
            "train_prompt_tokens_max": max(len(row["prompt_ids"]) for row in train),
            "eval_prompt_tokens_max": max(len(row["prompt_ids"]) for row in evaluation)}


def prepare_manifest(train_archive, train_request_manifest, eval_manifest, *, cap_policy,
                     train_context_limit=2048, serving_context_limit=None, raw_train=None,
                     tokenizer_path=None, prompt_provenance=None):
    """Prepare from archived IDs, or explicitly request the verified raw fallback."""
    _require((train_archive is None) != (raw_train is None),
             "Choose exactly one training source: archive or verified raw file")
    provenance = {"reference_train_manifest": str(Path(train_request_manifest).resolve()),
                  "reference_eval_manifest": str(Path(eval_manifest).resolve()),
                  "dataset_revision": DATASET_REVISION, "target_revision": TARGET_REVISION,
                  "ground_truth_answers_used": False, "warmup_excluded": True}
    if train_archive is not None:
        _require(tokenizer_path is None and prompt_provenance is None,
                 "Archive reuse needs no tokenizer or reconstruction provenance")
        train = load_training_archive(train_archive, train_request_manifest)
        provenance.update(training_source="original_reference_prompt_ids",
                          train_archive=str(Path(train_archive).resolve()))
    else:
        _require(tokenizer_path is not None and prompt_provenance is not None,
                 "Raw fallback requires a local pinned tokenizer and original prompt provenance")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        train = load_training_raw(raw_train, train_request_manifest, tokenizer, prompt_provenance)
        provenance.update(training_source="verified_raw_question_reconstruction",
                          raw_train=str(Path(raw_train).resolve()), tokenizer_path=str(Path(tokenizer_path).resolve()),
                          prompt_provenance=str(Path(prompt_provenance).resolve()),
                          raw_data_sha256=TRAIN_DATA_SHA256, all_reference_prompt_ids_verified=True)
    evaluation = load_evaluation_manifest(eval_manifest)
    return build_manifest(train, evaluation, cap_policy=cap_policy,
                          train_context_limit=train_context_limit,
                          serving_context_limit=serving_context_limit, provenance=provenance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--train-archive")
    source.add_argument("--raw-train")
    parser.add_argument("--train-request-manifest", required=True)
    parser.add_argument("--eval-manifest", required=True)
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--prompt-provenance", help="Original ONLINE_ROWS.json, required for raw fallback")
    parser.add_argument("--cap-policy", choices=("remaining", "fixed2048"), required=True)
    parser.add_argument("--train-context-limit", type=int, default=2048)
    parser.add_argument("--serving-context-limit", type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    _require(not output.exists(), "Refusing to replace an existing request manifest")
    manifest = prepare_manifest(
        args.train_archive, args.train_request_manifest, args.eval_manifest,
        cap_policy=args.cap_policy, train_context_limit=args.train_context_limit,
        serving_context_limit=args.serving_context_limit, raw_train=args.raw_train,
        tokenizer_path=args.tokenizer_path, prompt_provenance=args.prompt_provenance,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"output": str(output.resolve()), **validate_manifest(manifest)}, indent=2))


if __name__ == "__main__":
    main()
