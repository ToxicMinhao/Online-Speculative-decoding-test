"""Bind existing canonical requests without generating or rewriting any text."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from experiments.dflash_per_turn.experiment import EXPECTED_STREAMS, digest, require, valid_ids, validate_stream


SCHEMA = "osd_agreement_filter_canonical_v1"


def text_sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_binding(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _canonical(path, stream_id):
    raw = Path(path).read_bytes()
    if Path(path).suffix == ".jsonl":
        events = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
        rows = [event["data"] for event in events if event.get("event") == "canonical"]
    else:
        result = json.loads(raw)
        require(result.get("episode_id") == stream_id and result.get("status") == "complete",
                "Reference must be the complete original stream result")
        rows = result.get("canonical", [])
    selected = [row for row in rows if row.get("turn") in (1, 2, 3)]
    require([row.get("turn") for row in selected] == [1, 2, 3],
            "Reference must contain exactly one original canonical request in turns 1-3 order")
    require(all(row.get("episode_id") == stream_id and row.get("canonical") is True for row in selected),
            "Reference includes a replay or another conversation")
    return selected


def prepare_references(streams, source_files):
    """Read complete RESULT.json/EVENTS.jsonl files and produce a portable bound bundle.

    Source file hashes describe the entire historical artifacts. The portable bundle
    retains exact token arrays, user wording and source bindings; it never reloads
    a tokenizer or regenerates histories.
    """
    require([s["stream_id"] for s in streams] == list(EXPECTED_STREAMS), "Six-stream order changed")
    require(set(source_files) == set(EXPECTED_STREAMS), "Exactly six historical source artifacts required")
    bundle = {"schema": SCHEMA, "origin": "existing_per_turn_pilot_canonical_inputs",
              "benchmark_turns": [2, 3], "streams": []}
    for stream in streams:
        rows = validate_stream(stream)
        path = Path(source_files[stream["stream_id"]])
        historical = _canonical(path, stream["stream_id"])
        requests, contexts = [], []
        for row, old in zip(rows[:3], historical):
            text = row["prompt"]
            require(old.get("source_user_text") == text and
                    old.get("source_user_text_sha256") == text_sha256(text),
                    "Historical user whitespace, wording or order differs from approved dataset")
            require(old.get("source_message_ids") == row.get("source_message_ids"), "Source message IDs differ")
            for key in ("input_ids", "output_ids"):
                valid_ids(old.get(key), output=key == "output_ids")
            require(old.get("input_ids_sha256") == digest(old["input_ids"]),
                    "Historical native input observation hash differs")
            requests.append({"turn": row["turn"], "source_user_text": text,
                             "source_user_text_sha256": text_sha256(text),
                             "source_message_ids": copy.deepcopy(row.get("source_message_ids")),
                             "request_id": row.get("request_id")})
            if row["turn"] in (2, 3):
                context = {"episode_id": stream["stream_id"], "turn": row["turn"],
                           "question_id": old.get("question_id"), "source": stream["source"],
                           "synthetic": stream["synthetic"], "source_user_text": text,
                           "source_user_text_sha256": text_sha256(text),
                           "input_ids": list(old["input_ids"]), "output_ids": list(old["output_ids"]),
                           "input_ids_sha256": digest(old["input_ids"]),
                           "output_ids_sha256": digest(old["output_ids"]),
                           "historical_checkpoint_identity_sha256": old.get("checkpoint_identity_sha256"),
                           "historical_record_sha256": digest(old)}
                context["context_sha256"] = digest(context)
                contexts.append(context)
        entry = {"stream_id": stream["stream_id"], "source": stream["source"],
                 "synthetic": stream["synthetic"], "source_binding": file_binding(path),
                 "stream_binding": copy.deepcopy(stream.get("binding")),
                 "requests": requests, "contexts": contexts}
        entry["entry_sha256"] = digest(entry)
        bundle["streams"].append(entry)
    bundle["bundle_sha256"] = digest(bundle)
    validate_references(bundle, streams)
    return bundle


def validate_references(bundle, streams):
    require(bundle.get("schema") == SCHEMA and bundle.get("benchmark_turns") == [2, 3],
            "Wrong canonical bundle schema or benchmark turns")
    require(bundle.get("bundle_sha256") == digest({k: v for k, v in bundle.items() if k != "bundle_sha256"}),
            "Canonical bundle changed")
    require([s["stream_id"] for s in streams] == list(EXPECTED_STREAMS), "Approved stream order changed")
    entries = bundle.get("streams", [])
    require([e.get("stream_id") for e in entries] == list(EXPECTED_STREAMS), "Canonical stream order changed")
    result = {}
    for stream, entry in zip(streams, entries):
        rows = validate_stream(stream)
        require(entry.get("entry_sha256") == digest({k: v for k, v in entry.items() if k != "entry_sha256"}),
                "Canonical stream binding changed")
        require(entry.get("source") == stream["source"] and entry.get("synthetic") is stream["synthetic"],
                "Source/synthetic identity differs")
        binding = entry.get("source_binding", {})
        require(type(binding.get("bytes")) is int and binding["bytes"] > 0 and
                isinstance(binding.get("sha256"), str) and len(binding["sha256"]) == 64,
                "Missing complete historical source-file binding")
        requests = entry.get("requests", [])
        require(len(requests) == 3, "Three source request identities required")
        for expected, saved in zip(rows[:3], requests):
            require(saved.get("turn") == expected["turn"] and saved.get("source_user_text") == expected["prompt"] and
                    saved.get("source_user_text_sha256") == text_sha256(expected["prompt"]) and
                    saved.get("source_message_ids") == expected.get("source_message_ids") and
                    saved.get("request_id") == expected.get("request_id"),
                    "Canonical source text, whitespace, IDs or order changed")
        contexts = entry.get("contexts", [])
        require([r.get("turn") for r in contexts] == [2, 3], "Exactly two immutable contexts required")
        for context in contexts:
            require(context.get("context_sha256") == digest({k: v for k, v in context.items() if k != "context_sha256"}),
                    "Canonical context changed")
            require(context.get("episode_id") == stream["stream_id"] and
                    context.get("source") == stream["source"] and
                    context.get("synthetic") is stream["synthetic"], "Canonical provenance differs")
            expected = rows[context["turn"] - 1]
            require(context.get("source_user_text") == expected["prompt"] and
                    context.get("source_user_text_sha256") == text_sha256(expected["prompt"]),
                    "Canonical user text changed")
            for key in ("input_ids", "output_ids"):
                valid_ids(context.get(key), output=key == "output_ids")
                require(context.get(key + "_sha256") == digest(context[key]), "Canonical token IDs changed")
        result[stream["stream_id"]] = {row["turn"]: copy.deepcopy(row) for row in contexts}
    return result


def load_references(path, streams):
    bundle = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_references(bundle, streams)
