"""Validate and load the approved, locally retained six-stream dataset.

Only provenance hashes/IDs live in code. Raw user text and source assistant
records remain in the separate local dataset package, never this module.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

EXPECTED_STREAM_COUNTS = {
    "wildchat_00": 20, "wildchat_01": 16, "wildchat_02": 17,
    "wildchat_19": 23, "personamem_persona42": 18, "personamem_persona404": 22,
}
STREAM_IDS = tuple(EXPECTED_STREAM_COUNTS)
TOTAL_REQUESTS = 116
UPDATE_EVENTS = 110
OPTIMIZER_STEPS = 220
SAMPLING_SEED = 20261015
MANIFEST_SHA256 = "7332c8e81422617ab99f86a0fe391a07f1328aa75620c8718fa1348757a001de"
PACKAGE_SHA256 = "7f242e37dd1e694e4d7c27789bf2a36e5b62c2d14341bab07a742c3a056517fc"
SOURCE_PINS = {
    "wildchat": ("allenai/WildChat-4.8M", "c827c6df8fcf008219ffaffa4d1dd77491099367"),
    "personamem": ("bowen-upenn/PersonaMem-v2", "ed956dea41521fc4499acbc63f966e0fd3c053ba"),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def _path(root, relative):
    relative = Path(relative)
    require(not relative.is_absolute() and ".." not in relative.parts,
            "Dataset path escapes package")
    path = root / relative
    require(path.is_file() and path.resolve().is_relative_to(root), "Dataset file missing or escaping package")
    return path


def _groups(messages):
    groups, pending = [], []
    for index, message in enumerate(messages):
        role = message["role"]
        if role == "user":
            require(isinstance(message["content"], str) and message["content"], "Nontext/empty source user")
            pending.append((index, message))
        elif role == "assistant":
            if pending:
                groups.append(pending)
                pending = []
        elif role == "system":
            require(not groups and not pending, "Interior source system message")
        else:
            raise ValueError("Unsupported source role: " + role)
    if pending:
        groups.append(pending)
    return groups


def _validate_stream(entry, document, original, budget):
    """Check chronology/grouping against source; never import source assistants."""
    stream_id = entry["stream_id"]
    count = EXPECTED_STREAM_COUNTS[stream_id]
    source = entry["source"]
    require((entry["source_dataset"], entry["source_revision"]) == SOURCE_PINS[source], "Source revision changed")
    require(entry["synthetic"] is (source == "personamem"), "Synthetic source flag changed")
    require(entry["selected_grouped_request_range_inclusive"] == [1, count], "Selected chronology changed")
    require(entry["request_count"] == count and len(document["requests"]) == count, "Request count changed")
    require(not entry["original_assistants_used_for_serving"] and
            not entry["source_system_messages_used_for_serving"], "Original assistant/system injection")
    require(entry["join_separator"] == "\n\n", "Grouping separator changed")
    for key in ("stream_id", "source", "source_dataset", "source_revision", "source_record_id", "synthetic"):
        require(document[key] == entry[key], "Request stream/manifest disagreement: " + key)
    messages = original["conversation" if source == "wildchat" else "chat_history"]
    groups = _groups(messages)
    require(len(groups) == entry["source_grouped_request_count"], "Original group count changed")
    if source == "personamem":
        require(original["metadata"]["persona_id"] == entry["persona_id"], "Persona ID mismatch")
    result, source_indices = [], []
    for turn, (request, group) in enumerate(zip(document["requests"], groups[:count]), 1):
        require(request["request_index"] == turn and request["request_id"] == f"{stream_id}_r{turn:02d}",
                "Request order/identity changed")
        users = request["user_messages"]
        require(len(users) == len(group) and len(request["source_message_spans"]) == len(group),
                "Source message grouping changed")
        exact = "\n\n".join(message["content"] for _, message in group)
        require(request["serving_text"] == exact, "Selected user text differs from original, including whitespace")
        cursor = 0
        for offset, ((index, original_user), user, span) in enumerate(zip(group, users, request["source_message_spans"])):
            if offset:
                cursor += 2
            content = original_user["content"]
            source_id = f'{entry["source_record_id"]}#message/{index}'
            require(user["role"] == "user" and user["content"] == content, "User text changed")
            require(user["source_message_index_zero_based"] == index and user["source_message_id"] == source_id,
                    "Source message identity changed")
            require(user["content_sha256"] == sha(content.encode()), "User content hash differs")
            require(user["native_turn_identifier"] == original_user.get("turn_identifier") and
                    user["native_openai_id"] == original_user.get("openai_id"), "Native source ID changed")
            require(span == {"source_message_id": source_id, "start_char": cursor,
                             "end_char": cursor + len(content)}, "Original source boundaries changed")
            cursor += len(content)
            source_indices.append(index)
        result.append({"turn": turn, "request_id": request["request_id"], "prompt": exact,
                       "source_message_ids": [user["source_message_id"] for user in users],
                       "source_message_indices_zero_based": [index for index, _ in group],
                       "source_message_spans": request["source_message_spans"],
                       "prompt_sha256": sha(exact.encode()), "annotation": request["annotation"]})
    require(source_indices == entry["original_source_message_indices_zero_based"], "Source chronology changed")
    require(len(source_indices) == entry["user_message_count"], "User message count changed")
    require(budget["request_count"] == count and budget["all_fit"] and not budget["source_assistant_content_used"],
            "Unqualified prepared context budget")
    require(budget["max_input_tokens_upper_bound"] <= 31744 and budget["max_total_tokens_upper_bound"] <= 32768,
            "Prepared context budget exceeds qualified limits")
    return {**entry, "requests": result}


def load_streams(dataset_dir):
    """Validate the immutable prepared package, returning only its six streams.

    Each request has ``turn``, exact ``prompt``, source IDs/boundaries and
    audit-only annotations. Serving MUST use ``prompt`` alone, not annotations.
    Original assistant messages/profiles are validated locally but never returned.
    """
    root = Path(dataset_dir).resolve()
    manifest_bytes = _path(root, "manifest.json").read_bytes()
    package_bytes = _path(root, "PACKAGE_SHA256.json").read_bytes()
    require(sha(manifest_bytes) == MANIFEST_SHA256, "Approved dataset manifest changed")
    require(sha(package_bytes) == PACKAGE_SHA256, "Approved package inventory changed")
    for item in json.loads(package_bytes)["files"]:
        payload = _path(root, item["path"]).read_bytes()
        require(len(payload) == item["bytes"] and sha(payload) == item["sha256"],
                "Approved package artifact changed: " + item["path"])
    manifest = json.loads(manifest_bytes)
    require(manifest["sampling_seed"] == SAMPLING_SEED, "Sampling seed changed")
    require([s["stream_id"] for s in manifest["streams"]] == list(STREAM_IDS), "Approved stream order changed")
    result = []
    for entry in manifest["streams"]:
        original_bytes = _path(root, entry["original_record"]).read_bytes()
        require(sha(original_bytes) == entry["original_sha256"], "Original source record changed")
        request_path = entry["requests_file"]
        payload = _path(root, request_path).read_bytes()
        require(sha(payload) == manifest["output_sha256"][request_path], "Prepared request file changed")
        result.append(_validate_stream(entry, json.loads(payload), json.loads(original_bytes),
            json.loads(_path(root, entry["token_budget_file"]).read_text())))
    require(sum(len(s["requests"]) for s in result) == TOTAL_REQUESTS, "Pilot request count differs")
    personas = [s["persona_id"] for s in result if s["source"] == "personamem"]
    require(personas == [42, 404] and len(set(personas)) == 2, "Persona streams are not independent IDs")
    return result
