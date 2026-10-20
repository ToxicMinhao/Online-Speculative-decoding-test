"""Prepare the pinned independent-request screen, without model execution.

Only local, hash-verified upstream files are read. Run with ``python -m
experiments.dflash_baseline_learning_curve.data --bindings local_paths.json
--tokenizer /qualified/tokenizer --out /private/prepared``. This command never
downloads corpora, loads model weights, or generates answers. Dataset outputs
and sampled prompt text belong outside the public checkout.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import re
import unicodedata

from experiments.dflash_per_turn import framing

SEED = 20261020
DOMAINS = ("gsm8k", "codealpaca", "sharegpt")
ROLES = ("feedback", "evaluation")
CODEALPACA_EVAL_ROWS = 1024
SHAREGPT_FEEDBACK_ROWS = 30000
SHAREGPT_EVAL_ROWS = 1024
CATALOG_PATH = Path(__file__).with_name("dataset_sources.json")
# The pinned split file uses <conversation-id>_<source-message-offset>.
# Fail closed for missing/ambiguous offsets (including nested chunk suffixes).
SHAREGPT_CHUNK_ID = re.compile(r"(?P<root>[^_\s]+)_(?P<start>0|[1-9][0-9]*)")
INITIAL_CHUNK_RULE = "zero_source_offset_and_user_first_v1"
IMAGE_MARKER = re.compile(r"<\s*(?:image|img)\b|!\[[^\]]*\]\(", re.I)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_digest(value):
    return digest(json.dumps(value, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":"), allow_nan=False).encode())


def prompt_identity(text):
    """Dedup-only normalization; never use the normalized value for serving."""
    return digest(" ".join(unicodedata.normalize("NFC", text).split()).encode())


def sharegpt_chunk_identity(source_id):
    """Parse the pinned split convention, never infer start from row order/role."""
    if not isinstance(source_id, str):
        return None
    match = SHAREGPT_CHUNK_ID.fullmatch(source_id)
    if match is None:
        return None
    return {"conversation_root_id": match["root"], "source_start_index": int(match["start"])}


def catalog():
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


def qualified_tokenizer_binding():
    return {"model": framing.MODEL_ID, "revision": framing.TOKENIZER_REVISION,
            "files_sha256": framing.TOKENIZER_SHA256,
            "chat_template_sha256": framing.CHAT_TEMPLATE_SHA256,
            "framing": "dflash_per_turn.framing lossless manual user framing",
            "system_text": framing.COMMON_SYSTEM_TEXT}


def _load_source_files(source_bindings, source_catalog):
    sources, receipts = {}, {}
    require(set(source_bindings) == set(source_catalog["sources"]),
            "Exactly the four pinned source-file bindings are required")
    for name, spec in source_catalog["sources"].items():
        bound = source_bindings[name]
        path = Path(bound["path"] if isinstance(bound, dict) else bound).resolve()
        require(path.is_file(), "Missing pinned dataset artifact: " + name)
        require(path.stat().st_size == spec["bytes"], "Dataset file size differs: " + name)
        observed = file_digest(path)
        require(observed == spec["sha256"], "Dataset file SHA256 differs: " + name)
        if spec["format"] == "parquet":
            import pyarrow.parquet as pq
            # The raw answers remain in the immutable source file; they are not
            # loaded into the preparation pipeline at all for these datasets.
            columns = ["question"] if spec["domain"] == "gsm8k" else ["instruction", "input"]
            rows = pq.read_table(path, columns=columns).to_pylist()
        elif spec["format"] == "json":
            with path.open(encoding="utf-8") as handle:
                rows = json.load(handle)
        else:
            raise ValueError("Unsupported pinned source format: " + name)
        require(isinstance(rows, list) and rows, "Nonempty source rows required: " + name)
        sources[name] = rows
        receipts[name] = {**spec, "local_path": str(path), "observed_sha256": observed,
                          "rows": len(rows)}
    return sources, receipts


def _english_detector():
    require(importlib.metadata.version("langdetect") == "1.0.9",
            "English selection requires pinned langdetect==1.0.9")
    from langdetect.detector_factory import DetectorFactory, PROFILES_DIRECTORY
    factory = DetectorFactory()
    factory.load_profile(PROFILES_DIRECTORY)
    factory.seed = SEED

    def detect(text):
        try:
            detector = factory.create()
            detector.append(text)
            probabilities = detector.get_probabilities()
            en = next((float(x.prob) for x in probabilities if x.lang == "en"), 0.0)
            return {"en_probability": en, "top_language": probabilities[0].lang,
                    "eligible": probabilities[0].lang == "en" and en >= 0.9}
        except Exception as error:
            # Preserve a named exclusion, never fall back to 'English by default'.
            return {"eligible": False, "error_type": type(error).__name__}
    return detect


def _partition_indices(sources, domain, role):
    if domain == "gsm8k":
        key = "gsm8k_train" if role == "feedback" else "gsm8k_test"
        return key, list(range(len(sources[key])))
    if domain == "codealpaca":
        key = "codealpaca_train"
        boundary = max(0, len(sources[key]) - CODEALPACA_EVAL_ROWS)
        return key, list(range(0, boundary) if role == "feedback" else
                         range(boundary, len(sources[key])))
    key = "sharegpt_train"
    n = len(sources[key])
    return key, list(range(min(n, SHAREGPT_FEEDBACK_ROWS)) if role == "feedback" else
                     range(min(n, SHAREGPT_FEEDBACK_ROWS),
                           min(n, SHAREGPT_FEEDBACK_ROWS + SHAREGPT_EVAL_ROWS)))


def _extract_prompt(row, domain, english_detector):
    """Return prompt/provenance or one auditable exclusion; never return answers."""
    if not isinstance(row, dict):
        return None, "malformed_record"
    if domain == "gsm8k":
        text, fields = row.get("question"), {"question": row.get("question")}
        metadata = {"source_prompt_fields": fields}
    elif domain == "codealpaca":
        instruction, extra = row.get("instruction"), row.get("input", "")
        if not isinstance(instruction, str) or not isinstance(extra, str):
            return None, "nontext_prompt_field"
        text = instruction + ("\n\n" + extra if extra else "")
        metadata = {"source_prompt_fields": {"instruction": instruction, "input": extra}}
    else:
        messages = row.get("conversations")
        source_id = row.get("id")
        if not isinstance(source_id, (str, int)) or not str(source_id):
            return None, "missing_conversation_id"
        chunk = sharegpt_chunk_identity(source_id)
        if chunk is None:
            return None, "sharegpt_unverified_chunk_origin"
        if chunk["source_start_index"] != 0:
            return chunk, "sharegpt_continuation_chunk"
        if not isinstance(messages, list) or not messages:
            return chunk, "missing_conversation"
        first = messages[0]
        if not isinstance(first, dict):
            return chunk, "nontext_conversation"
        first_role = first.get("from", first.get("role"))
        if first_role not in ("human", "user"):
            return {**chunk, "first_message_role": first_role}, "sharegpt_initial_chunk_not_user_first"
        users = []
        for message in messages:
            if not isinstance(message, dict):
                return None, "nontext_conversation"
            value = message.get("value", message.get("content"))
            if not isinstance(value, str) or IMAGE_MARKER.search(value):
                return None, "nontext_or_image_conversation"
            role = message.get("from", message.get("role"))
            if role in ("human", "user"):
                users.append(value)
            elif role not in ("gpt", "assistant", "chatgpt", "bing", "bard", "system"):
                return None, "unsupported_conversation_role"
        if not users:
            return None, "no_user_request"
        # Never search past an assistant/system prefix for an isolated follow-up.
        text = first.get("value", first.get("content"))
        first_language = english_detector(text)
        conversation_language = english_detector("\n\n".join(users))
        metadata = {"conversation_id": str(source_id),
                    **chunk, "source_message_index": 0,
                    "first_message_role": first_role, "initial_chunk_rule": INITIAL_CHUNK_RULE,
                    "source_prompt_fields": {"first_user": text},
                    "language": {"first_user": first_language, "all_source_users": conversation_language}}
        if not first_language["eligible"] or not conversation_language["eligible"]:
            return metadata, "english_filter_rejected"
    if not isinstance(text, str) or not text.strip():
        return None, "empty_or_nontext_prompt"
    metadata["text"] = text
    return metadata, None


def encode_request(tokenizer, text):
    """One independent user request; no dataset answer or prior request."""
    prefix = list(tokenizer.encode(framing.BOS + framing.SYSTEM_HEADER +
                                  framing.COMMON_SYSTEM_TEXT + framing.EOT,
                                  add_special_tokens=False))
    return prefix + framing.user_segment_ids(tokenizer, text)


def _select_manifest(sources, receipts, tokenizer, *, tokenizer_binding,
                     source_catalog, english_detector, encode=encode_request,
                     context_length=32768, max_input_tokens=31744,
                     output_cap=1024, block_size=10):
    """Pure selection core; injected fakes are for synthetic CPU tests only."""
    require((context_length, max_input_tokens, output_cap, block_size) ==
            (32768, 31744, 1024, 10), "Qualified serving budget cannot be silently changed")
    rng, seen_prompts, seen_sources, seen_conversations = random.Random(SEED), {}, set(), set()
    groups = [{"id": f"{domain}_{index:02d}", "group_id": f"{domain}_{index:02d}",
               "domain": domain, "feedback": [], "evaluation": []}
              for domain in DOMAINS for index in range(2)]
    exclusions, pools, shortfalls = [], [], []
    for domain_index, domain in enumerate(DOMAINS):
        for role in ROLES:
            key, indices = _partition_indices(sources, domain, role)
            rng.shuffle(indices)
            accepted, inspected = [], 0
            for rank, source_index in enumerate(indices):
                inspected += 1
                row = sources[key][source_index]
                source_id = str(row.get("id")) if domain == "sharegpt" and isinstance(row, dict) else f"{key}:{source_index}"
                identity = (domain, source_id)
                item, reason = _extract_prompt(row, domain, english_detector)
                audit = {"domain": domain, "role": role, "source_key": key,
                         "source_index": source_index, "source_id": source_id,
                         "candidate_rank": rank}
                if reason:
                    exclusions.append({**audit, "reason": reason,
                                       **({k: item[k] for k in ("language", "conversation_root_id",
                                                              "source_start_index", "first_message_role")
                                           if k in item} if item else {})})
                    continue
                text = item["text"]
                content_id = prompt_identity(text)
                audit["prompt_dedup_sha256"] = content_id
                if content_id in seen_prompts:
                    exclusions.append({**audit, "reason": "duplicate_prompt",
                                       "duplicate_of": seen_prompts[content_id]})
                    continue
                if identity in seen_sources:
                    exclusions.append({**audit, "reason": "duplicate_source_id"})
                    continue
                root = item.get("conversation_root_id")
                if root is not None and root in seen_conversations:
                    exclusions.append({**audit, "reason": "duplicate_conversation_root"})
                    continue
                input_ids = encode(tokenizer, text)
                require(isinstance(input_ids, list) and input_ids and
                        all(type(token) is int and 0 <= token < framing.VOCAB_SIZE for token in input_ids),
                        "Tokenizer returned invalid input token IDs")
                total = len(input_ids) + output_cap + block_size - 1
                if len(input_ids) > max_input_tokens or total > context_length:
                    exclusions.append({**audit, "reason": "context_budget_exceeded",
                                       "input_tokens": len(input_ids), "reserved_total_tokens": total})
                    continue
                group_index, within_role = divmod(len(accepted), 4)
                group = groups[domain_index * 2 + group_index]
                request_id = f"{group['id']}_{role}_{within_role:02d}"
                spec = source_catalog["sources"][key]
                record = {**item, **audit, "request_id": request_id,
                          "role": role, "order": within_role,
                          "dataset": spec["dataset"], "source_revision": spec["revision"],
                          "source_split": spec["split"], "source_repository_path": spec["repository_path"],
                          "prompt_sha256": digest(text.encode()), "input_ids": input_ids,
                          "input_ids_sha256": json_digest(input_ids),
                          "token_budget": {"input_tokens": len(input_ids), "output_cap": output_cap,
                              "final_proposal_allowance": block_size - 1,
                              "reserved_total_tokens": total, "context_limit": context_length,
                              "configured_input_limit": max_input_tokens,
                              "effective_input_limit": min(max_input_tokens, context_length-output_cap-block_size+1)}}
                group[role].append(record)
                accepted.append(record)
                seen_prompts[content_id] = request_id
                seen_sources.add(identity)
                if root is not None:
                    seen_conversations.add(root)
                if len(accepted) == 8:
                    break
            pools.append({"domain": domain, "role": role, "source_key": key,
                          "partition_rows": len(indices), "inspected_rows": inspected,
                          "accepted_rows": len(accepted), "uninspected_rows": len(indices)-inspected})
            if len(accepted) != 8:
                shortfalls.append({"domain": domain, "role": role, "required": 8,
                                   "selected": len(accepted)})
    manifest = {"schema_version": 1, "status": "shortfall" if shortfalls else "ready",
                "selection_seed": SEED, "training_seed": 17, "groups": groups,
                "sources": receipts, "source_catalog": source_catalog,
                "tokenizer": tokenizer_binding, "selection": source_catalog["selection"],
                "exclusions": exclusions, "candidate_pools": pools, "shortfalls": shortfalls,
                "counts": {"groups": len(groups), "requests": len(seen_prompts),
                           "feedback": sum(len(g["feedback"]) for g in groups),
                           "evaluation": sum(len(g["evaluation"]) for g in groups)},
                "contract": {"independent_requests": True, "dataset_answers_used": False,
                             "generation_started": False, "preprocessing_pinned_before_generation": True},
                "code_sha256": {"data.py": file_digest(Path(__file__)),
                                "framing.py": file_digest(Path(framing.__file__))}}
    manifest["manifest_payload_sha256"] = json_digest(manifest)
    return manifest


def prepare_manifest(source_bindings, tokenizer, *, tokenizer_binding=None,
                     context_length=32768, max_input_tokens=31744,
                     output_cap=1024, block_size=10):
    framing.validate_tokenizer(tokenizer)
    binding = qualified_tokenizer_binding()
    require(tokenizer_binding is None or tokenizer_binding == binding,
            "Tokenizer binding differs from the qualified tokenizer")
    source_catalog = catalog()
    sources, receipts = _load_source_files(source_bindings, source_catalog)
    return _select_manifest(sources, receipts, tokenizer, tokenizer_binding=binding,
        source_catalog=source_catalog, english_detector=_english_detector(),
        context_length=context_length, max_input_tokens=max_input_tokens,
        output_cap=output_cap, block_size=block_size)


def validate_manifest(manifest):
    """Reject altered, incomplete or overlapping data before any generation."""
    require(manifest.get("status") == "ready" and not manifest.get("shortfalls"),
            "Dataset preparation has a shortfall; do not substitute requests")
    expected = manifest.get("manifest_payload_sha256")
    require(expected == json_digest({k: v for k, v in manifest.items() if k != "manifest_payload_sha256"}),
            "Prepared manifest integrity check failed")
    require(manifest.get("selection_seed") == SEED, "Dataset selection seed changed")
    require(manifest.get("tokenizer") == qualified_tokenizer_binding(), "Tokenizer pin changed")
    pinned_catalog = catalog()
    require(manifest.get("source_catalog") == pinned_catalog and
            manifest.get("selection") == pinned_catalog["selection"],
            "Dataset source/preprocessing pins changed")
    require(set(manifest.get("sources", {})) == set(pinned_catalog["sources"]),
            "Prepared source bindings are incomplete")
    for key, spec in pinned_catalog["sources"].items():
        receipt = manifest["sources"][key]
        require(all(receipt.get(k) == v for k, v in spec.items()) and
                receipt.get("observed_sha256") == spec["sha256"] and
                type(receipt.get("rows")) is int and receipt["rows"] > 0,
                "Prepared upstream file binding changed: " + key)
    require(manifest.get("contract", {}).get("dataset_answers_used") is False,
            "Dataset answers cannot supply serving input")
    groups = manifest.get("groups", [])
    expected_groups = [(f"{d}_{i:02d}", d) for d in DOMAINS for i in range(2)]
    require([(g.get("id"), g.get("domain")) for g in groups] == expected_groups,
            "Exactly six ordered groups are required")
    ids, prompts, source_ids, conversations = set(), set(), set(), set()
    for group in groups:
        require(group.get("group_id") == group["id"], "Group identity aliases disagree")
        for role in ROLES:
            require(len(group.get(role, [])) == 4, "Each role requires exactly four requests")
            for order, row in enumerate(group[role]):
                require(row["request_id"] == f"{group['id']}_{role}_{order:02d}" and
                        row["request_id"] not in ids and row["role"] == role and row["order"] == order,
                        "Duplicate request ID or altered role/order")
                ids.add(row["request_id"])
                text = row["text"]
                require(digest(text.encode()) == row["prompt_sha256"], "Exact prompt text changed")
                dedup = prompt_identity(text)
                require(dedup == row["prompt_dedup_sha256"] and dedup not in prompts,
                        "Prompt duplicate or changed dedup identity")
                prompts.add(dedup)
                require(json_digest(row["input_ids"]) == row["input_ids_sha256"], "Input token IDs changed")
                framing.check_input(row["input_ids"])
                require(row["token_budget"] == {
                    "input_tokens": len(row["input_ids"]), "output_cap": 1024,
                    "final_proposal_allowance": 9, "reserved_total_tokens": len(row["input_ids"])+1033,
                    "context_limit": 32768, "configured_input_limit": 31744,
                    "effective_input_limit": 31735},
                        "Input token budget changed")
                spec = pinned_catalog["sources"][row["source_key"]]
                require(row["source_revision"] == spec["revision"] and row["source_split"] == spec["split"] and
                        row["dataset"] == spec["dataset"] and row["domain"] == spec["domain"] == group["domain"] and
                        row["source_repository_path"] == spec["repository_path"],
                        "Request source binding changed")
                n = manifest["sources"][row["source_key"]]["rows"]
                index = row["source_index"]
                require(type(index) is int and 0 <= index < n, "Source row index changed")
                identity = (group["domain"], row["source_id"])
                require(identity not in source_ids, "Duplicate source identity")
                source_ids.add(identity)
                if group["domain"] == "gsm8k":
                    require(row["source_split"] == ("train" if role == "feedback" else "test"),
                            "GSM8K must use official train/test partition")
                    require(row["text"] == row["source_prompt_fields"]["question"],
                            "GSM8K question field changed")
                elif group["domain"] == "codealpaca":
                    boundary = max(0, n-CODEALPACA_EVAL_ROWS)
                    require((index < boundary) == (role == "feedback"), "CodeAlpaca partitions overlap")
                    fields = row["source_prompt_fields"]
                    require(text == fields["instruction"] + ("\n\n"+fields["input"] if fields["input"] else ""),
                            "CodeAlpaca source fields were changed or omitted")
                if group["domain"] == "sharegpt":
                    require((0 <= index < min(n, SHAREGPT_FEEDBACK_ROWS)) if role == "feedback" else
                            (SHAREGPT_FEEDBACK_ROWS <= index < min(n, SHAREGPT_FEEDBACK_ROWS+SHAREGPT_EVAL_ROWS)),
                            "ShareGPT conversation partitions overlap")
                    require(text == row["source_prompt_fields"]["first_user"] and
                            all(row["language"][k].get("eligible") is True and
                                row["language"][k].get("top_language") == "en" and
                                row["language"][k].get("en_probability", 0) >= 0.9
                                for k in ("first_user", "all_source_users")),
                            "ShareGPT exact first-user or language evidence changed")
                    chunk = sharegpt_chunk_identity(row.get("conversation_id"))
                    require(row.get("source_id") == row.get("conversation_id") and chunk is not None and
                            chunk["source_start_index"] == row.get("source_start_index") == 0 and
                            row.get("source_message_index") == 0 and
                            row.get("first_message_role") in ("human", "user") and
                            row.get("initial_chunk_rule") == INITIAL_CHUNK_RULE,
                            "ShareGPT request is not a verified user-first initial chunk")
                    root = chunk["conversation_root_id"]
                    require(root == row["conversation_root_id"] and root not in conversations,
                            "ShareGPT conversations overlap")
                    conversations.add(root)
    require(len(ids) == 48 and manifest["counts"] ==
            {"groups": 6, "requests": 48, "feedback": 24, "evaluation": 24},
            "Prepared request counts changed")
    return manifest


def validate_reencoded_inputs(manifest, tokenizer):
    validate_manifest(manifest)
    framing.validate_tokenizer(tokenizer)
    for group in manifest["groups"]:
        for role in ROLES:
            for row in group[role]:
                framing.require_identical_input(row["input_ids"], encode_request(tokenizer, row["text"]))
    return True


def load_manifest(path):
    return validate_manifest(json.loads(Path(path).read_text(encoding="utf-8")))


def _write(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    require(not args.out.exists(), "Use a new preparation directory; preserve prior manifests")
    require(not args.out.resolve().is_relative_to(Path(__file__).resolve().parents[2]),
            "Prepared prompts and runtime artifacts must remain outside public Git")
    tokenizer = framing.load_pinned_tokenizer(args.tokenizer)
    bindings = json.loads(args.bindings.read_text(encoding="utf-8"))
    manifest = prepare_manifest(bindings, tokenizer)
    args.out.mkdir(parents=True)
    _write(args.out / "manifest.json", manifest)
    _write(args.out / "exclusions.json", manifest["exclusions"])
    _write(args.out / "source_bindings.json", manifest["sources"])
    print(json.dumps({"status": manifest["status"], "counts": manifest["counts"],
                      "manifest_sha256": file_digest(args.out / "manifest.json"),
                      "generation_started": False}))
    return 0 if manifest["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
