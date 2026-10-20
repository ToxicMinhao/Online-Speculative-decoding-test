"""Synthetic-only preparation tests; no corpus, model or network access."""
from copy import deepcopy
import json

import pytest

from experiments.dflash_baseline_learning_curve import data


def english(text):
    is_english = "NON_ENGLISH" not in text
    return {"eligible": is_english, "top_language": "en" if is_english else "fr",
            "en_probability": 0.99 if is_english else 0.0}


def encode(_tokenizer, text):
    # Synthetic IDs preserve whitespace and stay inside the real vocab.
    return [ord(c) for c in text]


@pytest.fixture
def sources(monkeypatch):
    # Smaller synthetic row partitions; production's declared partition
    # constants are tested independently below.
    monkeypatch.setattr(data, "CODEALPACA_EVAL_ROWS", 16)
    monkeypatch.setattr(data, "SHAREGPT_FEEDBACK_ROWS", 32)
    monkeypatch.setattr(data, "SHAREGPT_EVAL_ROWS", 16)
    return {
        "gsm8k_train": [{"question": f"How many training apples are in box {i}?",
                         "answer": "ORIGINAL_ANSWER_NEVER_INCLUDED"} for i in range(24)],
        "gsm8k_test": [{"question": f"How many evaluation apples are in box {i}?",
                        "answer": "ORIGINAL_ANSWER_NEVER_INCLUDED"} for i in range(24)],
        "codealpaca_train": [{"instruction": f"Write a function with index {i}.",
                              "input": f"  def existing_{i}():\n    pass\n",
                              "output": "ORIGINAL_ANSWER_NEVER_INCLUDED"} for i in range(48)],
        "sharegpt_train": [{"id": f"conversation-{i}_0", "conversations": [
            {"from": "human", "value": f"Explain the English example topic number {i}."},
            {"from": "gpt", "value": "ORIGINAL_ANSWER_NEVER_INCLUDED"},
            {"from": "human", "value": "Please explain more about this English topic."}]}
            for i in range(48)]}


def prepared(sources):
    return data._select_manifest(sources,
        {key: {**spec, "rows": len(sources[key]), "local_path": "/synthetic/" + key,
               "observed_sha256": spec["sha256"]}
         for key, spec in data.catalog()["sources"].items()}, None,
        tokenizer_binding=data.qualified_tokenizer_binding(), source_catalog=data.catalog(),
        english_detector=english, encode=encode)


def all_rows(manifest):
    return [r for g in manifest["groups"] for role in data.ROLES for r in g[role]]


def test_complete_selection_reproducible_distinct_and_answers_unused(sources):
    original = deepcopy(sources)
    first, second = prepared(sources), prepared(sources)
    assert sources == original
    assert first == second
    assert first["counts"] == {"groups": 6, "requests": 48, "feedback": 24, "evaluation": 24}
    assert len({r["request_id"] for r in all_rows(first)}) == 48
    assert len({r["prompt_dedup_sha256"] for r in all_rows(first)}) == 48
    assert "ORIGINAL_ANSWER_NEVER_INCLUDED" not in json.dumps(first)
    assert data.validate_manifest(first) is first
    for group in first["groups"]:
        for role in data.ROLES:
            assert len(group[role]) == 4
            for r in group[role]:
                if group["domain"] == "gsm8k":
                    assert r["source_split"] == ("train" if role == "feedback" else "test")
                elif group["domain"] == "codealpaca":
                    assert (r["source_index"] < 32) == (role == "feedback")
                else:
                    assert (r["source_index"] < 32) == (role == "feedback")


def test_whitespace_and_code_input_are_not_stripped_or_dropped(sources):
    for i, row in enumerate(sources["gsm8k_train"]):
        row["question"] = f"\n  Math request {i}  \n\tindented\n"
    manifest = prepared(sources)
    for row in all_rows(manifest):
        assert row["input_ids"] == [ord(c) for c in row["text"]]
        if row["domain"] == "gsm8k" and row["role"] == "feedback":
            assert row["text"].startswith("\n  ") and row["text"].endswith("\n")
        if row["domain"] == "codealpaca":
            original = sources["codealpaca_train"][row["source_index"]]
            assert row["text"] == original["instruction"] + "\n\n" + original["input"]


def test_global_prompt_dedup_cross_domain_and_role(sources):
    # Force the complete CodeAlpaca feedback pool to duplicate already selected
    # math prompts, including whitespace differences; never fill with substitutes.
    initial = prepared(sources)
    math_prompts = [r["text"] for r in all_rows(initial) if r["domain"] == "gsm8k"]
    for i in range(32):
        sources["codealpaca_train"][i] = {"instruction": " \n" + math_prompts[i % len(math_prompts)],
                                         "input": "", "output": "unused"}
    manifest = prepared(sources)
    assert manifest["status"] == "shortfall"
    assert sum(x["reason"] == "duplicate_prompt" for x in manifest["exclusions"]) == 32
    assert any(x["domain"] == "codealpaca" and x["role"] == "feedback" for x in manifest["shortfalls"])
    with pytest.raises(ValueError, match="shortfall"):
        data.validate_manifest(manifest)


def test_sharegpt_partitions_exclude_all_continuation_offsets(sources):
    # Only offset zero can supply a first request, regardless of row partition.
    for i, row in enumerate(sources["sharegpt_train"]):
        row["id"] = f"same-root_{i}"
    manifest = prepared(sources)
    share = [r for r in all_rows(manifest) if r["domain"] == "sharegpt"]
    assert len(share) == 1
    excluded = [x for x in manifest["exclusions"] if x["reason"] == "sharegpt_continuation_chunk"]
    assert len(excluded) == 47
    assert {x["source_start_index"] for x in excluded} == set(range(1, 48))
    assert all(x["conversation_root_id"] == "same-root" for x in excluded)
    assert manifest["status"] == "shortfall"


def test_duplicate_initial_chunk_cannot_cross_role_partitions(sources):
    for row in sources["sharegpt_train"]:
        row["id"] = "same-root_0"
    manifest = prepared(sources)
    share = [r for r in all_rows(manifest) if r["domain"] == "sharegpt"]
    assert len(share) == 1 and share[0]["role"] == "feedback"
    excluded = [x for x in manifest["exclusions"] if x["reason"] == "duplicate_source_id"]
    assert len(excluded) == 47
    assert manifest["status"] == "shortfall"


def test_language_and_nontext_exclusions_are_explicit(sources):
    for row in sources["sharegpt_train"][:32]:
        row["conversations"][0]["value"] = "NON_ENGLISH question"
    for row in sources["sharegpt_train"][32:]:
        row["conversations"][0]["value"] = [{"type": "image"}]
    manifest = prepared(sources)
    reasons = [x["reason"] for x in manifest["exclusions"]]
    assert reasons.count("english_filter_rejected") == 32
    assert reasons.count("nontext_or_image_conversation") == 16
    assert manifest["status"] == "shortfall"


def test_oversize_exclusion_uses_final_proposal_allowance(sources):
    # 31,736 tokens fits nominal input limit but not 1024+9 reserved output.
    for row in sources["gsm8k_train"]:
        row["question"] = "x" * 31736
    manifest = prepared(sources)
    oversized = [x for x in manifest["exclusions"] if x["reason"] == "context_budget_exceeded"]
    assert len(oversized) == 24
    assert all(x["reserved_total_tokens"] == 32769 for x in oversized)
    assert manifest["shortfalls"][0]["selected"] == 0
    assert sources["gsm8k_train"][0]["question"] == "x" * 31736


def test_manifest_tampering_and_reencoding_detected(sources, monkeypatch):
    manifest = prepared(sources)
    broken = deepcopy(manifest)
    broken["groups"][0]["feedback"][0]["text"] += " "
    with pytest.raises(ValueError, match="integrity"):
        data.validate_manifest(broken)
    monkeypatch.setattr(data.framing, "validate_tokenizer", lambda _: None)
    monkeypatch.setattr(data, "encode_request", encode)
    assert data.validate_reencoded_inputs(manifest, None)
    monkeypatch.setattr(data, "encode_request", lambda _t, text: encode(None, text) + [1])
    with pytest.raises(ValueError, match="differ"):
        data.validate_reencoded_inputs(manifest, None)


def test_rehashed_manifest_still_rejects_partition_or_budget_changes(sources):
    manifest = prepared(sources)
    for kind in ("partition", "budget", "source_pin"):
        broken = deepcopy(manifest)
        row = broken["groups"][2]["feedback"][0]
        if kind == "partition":
            row["source_index"] = 47
            expected = "partitions overlap"
        elif kind == "budget":
            row["token_budget"]["output_cap"] = 256
            expected = "budget changed"
        else:
            broken["sources"]["codealpaca_train"]["observed_sha256"] = "0" * 64
            expected = "binding changed"
        del broken["manifest_payload_sha256"]
        broken["manifest_payload_sha256"] = data.json_digest(broken)
        with pytest.raises(ValueError, match=expected):
            data.validate_manifest(broken)


def test_pinned_artifact_hash_required_before_reading(tmp_path):
    file = tmp_path / "source.json"
    file.write_text('[{"question":"synthetic","answer":"unused"}]')
    source_catalog = {"sources": {"x": {"format": "json", "bytes": file.stat().st_size,
                                      "sha256": "0" * 64}}}
    with pytest.raises(ValueError, match="SHA256"):
        data._load_source_files({"x": str(file)}, source_catalog)
    source_catalog["sources"]["x"]["sha256"] = data.file_digest(file)
    rows, receipts = data._load_source_files({"x": str(file)}, source_catalog)
    assert rows["x"][0]["question"] == "synthetic"
    assert receipts["x"]["observed_sha256"] == data.file_digest(file)


def test_public_cli_never_accepts_output_inside_repository(tmp_path):
    with pytest.raises(ValueError, match="outside public Git"):
        data.main(["--bindings", str(tmp_path / "missing.json"), "--tokenizer", "/missing",
                   "--out", str(data.Path(__file__).parent / "forbidden_synthetic_output")])


def test_declared_production_partitions_and_revision_pins():
    assert (data.CODEALPACA_EVAL_ROWS, data.SHAREGPT_FEEDBACK_ROWS, data.SHAREGPT_EVAL_ROWS) == (1024, 30000, 1024)
    sources = {"codealpaca_train": [None] * 20022, "sharegpt_train": [None] * 40000}
    assert data._partition_indices(sources, "codealpaca", "evaluation")[1] == list(range(18998, 20022))
    assert data._partition_indices(sources, "sharegpt", "feedback")[1] == list(range(30000))
    assert data._partition_indices(sources, "sharegpt", "evaluation")[1] == list(range(30000, 31024))
    for spec in data.catalog()["sources"].values():
        assert len(spec["revision"]) == 40 and len(spec["sha256"]) == 64


def test_assistant_answers_do_not_change_language_inputs():
    seen = []
    def detector(text):
        seen.append(text)
        return english(text)
    row = {"id": "a_0", "conversations": [
        {"from": "human", "value": "  Explain this task.\n"},
        {"from": "gpt", "value": "NON_ENGLISH answer never used"},
        {"from": "human", "value": "Another English follow up."}]}
    item, reason = data._extract_prompt(row, "sharegpt", detector)
    assert reason is None and item["text"] == "  Explain this task.\n"
    assert seen == ["  Explain this task.\n", "  Explain this task.\n\n\nAnother English follow up."]
    assert "NON_ENGLISH" not in json.dumps(item)


@pytest.mark.parametrize("speaker", ["gpt", "human"])
def test_noninitial_chunk_is_excluded_even_if_it_starts_with_user(speaker):
    row = {"id": "abc1234_12", "conversations": [
        {"from": speaker, "value": "Earlier context or follow-up."},
        {"from": "human", "value": "Please explain why that approach uses this value."}]}
    calls = []
    item, reason = data._extract_prompt(row, "sharegpt", lambda text: calls.append(text))
    assert reason == "sharegpt_continuation_chunk"
    assert item == {"conversation_root_id": "abc1234", "source_start_index": 12}
    assert calls == []  # Eligibility is resolved before language checks/tokenization.


@pytest.mark.parametrize("speaker", ["gpt", "assistant", "system"])
def test_offset_zero_does_not_allow_scanning_past_nonuser_first_message(speaker):
    row = {"id": "abc1234_0", "conversations": [
        {"from": speaker, "value": "An explanation with omitted context."},
        {"from": "human", "value": "Why does that use this value?"}]}
    item, reason = data._extract_prompt(row, "sharegpt", english)
    assert reason == "sharegpt_initial_chunk_not_user_first"
    assert item["source_start_index"] == 0 and item["first_message_role"] == speaker
    assert "text" not in item


@pytest.mark.parametrize("identity", ["abc1234", "abc1234_00", "abc1234_bad", "abc1234_2_0", 1234])
def test_unverified_chunk_origin_is_never_assumed_initial(identity):
    row = {"id": identity, "conversations": [{"from": "human", "value": "A standalone looking request."}]}
    _, reason = data._extract_prompt(row, "sharegpt", english)
    assert reason == "sharegpt_unverified_chunk_origin"


def test_initial_chunk_keeps_exact_first_message_and_origin_receipt():
    text = "\n  Explain this code:\n    value = 3\n"
    row = {"id": "abc1234_0", "conversations": [
        {"from": "human", "value": text},
        {"from": "gpt", "value": "Unused original answer."},
        {"from": "human", "value": "Why this value?"}]}
    item, reason = data._extract_prompt(row, "sharegpt", english)
    assert reason is None and item["text"] == text
    assert item["source_start_index"] == item["source_message_index"] == 0
    assert item["initial_chunk_rule"] == data.INITIAL_CHUNK_RULE
    assert item["first_message_role"] == "human"
    assert item["conversation_root_id"] == "abc1234"


def test_manifest_logs_assistant_first_and_unknown_origins_without_replacement(sources):
    for row in sources["sharegpt_train"][:32]:
        row["conversations"].insert(0, {"from": "gpt", "value": "An earlier explanation."})
    for i, row in enumerate(sources["sharegpt_train"][32:]):
        row["id"] = f"unknown{i}"
    manifest = prepared(sources)
    assert manifest["status"] == "shortfall"
    reasons = [x["reason"] for x in manifest["exclusions"]]
    assert reasons.count("sharegpt_initial_chunk_not_user_first") == 32
    assert reasons.count("sharegpt_unverified_chunk_origin") == 16
    assert all(x["first_message_role"] == "gpt" for x in manifest["exclusions"]
               if x["reason"] == "sharegpt_initial_chunk_not_user_first")
    assert not [r for r in all_rows(manifest) if r["domain"] == "sharegpt"]


@pytest.mark.parametrize("change", ["continuation", "assistant_first", "source_message", "missing_evidence"])
def test_rehashed_manifest_cannot_readmit_invalid_chunk_origins(sources, change):
    manifest = prepared(sources)
    row = manifest["groups"][4]["feedback"][0]
    if change == "continuation":
        row["conversation_id"] = row["conversation_root_id"] + "_2"
        row["source_id"] = row["conversation_id"]
        row["source_start_index"] = 2
    elif change == "assistant_first":
        row["first_message_role"] = "gpt"
    elif change == "source_message":
        row["source_message_index"] = 1
    else:
        del row["initial_chunk_rule"]
    del manifest["manifest_payload_sha256"]
    manifest["manifest_payload_sha256"] = data.json_digest(manifest)
    with pytest.raises(ValueError, match="verified user-first initial chunk"):
        data.validate_manifest(manifest)
