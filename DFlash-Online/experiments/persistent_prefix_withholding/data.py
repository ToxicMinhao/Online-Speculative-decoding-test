"""Seal the v2 reused-development panel using inputs and exposure records only.

Reference provenance is a separate gate. This module never chooses a request by
reference availability, source answers, generated length, or model performance.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import unicodedata

import numpy as np

REVISION = "740312add88f781978c0658806c59bc2815b9866"
DATASET = f"openai/gsm8k:main:test@{REVISION}"
LABEL = "DEVELOPMENT_FEASIBILITY_REUSED_DATA"
TOKENIZER_HASHES = {
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "tokenizer_config.json": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
}
IDENTITY_KEYS = ("source_id", "normalized_question_sha256",
                 "rendered_prompt_sha256", "prompt_ids_sha256")


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def token_digest(tokens):
    return digest(json.dumps(tokens, separators=(",", ":")))


def normalize(text):
    """Match the retained audit's NFKC and whitespace normalization."""
    return " ".join(unicodedata.normalize("NFKC", text).split())


def selection_key(source_id):
    return digest(f"kplus-persistent-prefix-v1|eval|{DATASET}|{source_id}")


def identity_index(rows):
    return {key: {row[key] for row in rows} for key in IDENTITY_KEYS}


def overlap_reasons(row, index):
    return [key for key in IDENTITY_KEYS if row[key] in index[key]]


def choose_warmups(development_rows):
    """Fix short/long inputs before panel ranking, using prompt length only."""
    ordered = sorted(development_rows,
                     key=lambda row: (len(row["prompt_ids"]), row["source_id"]))
    if len(ordered) < 2 or ordered[0]["source_id"] == ordered[-1]["source_id"]:
        raise ValueError("Two distinct inherited development inputs are required")
    result = [deepcopy(ordered[0]), deepcopy(ordered[-1])]
    for ordinal, row in enumerate(result):
        row.update(panel_ordinal=ordinal, warmup_role=("short_input", "long_input")[ordinal])
    return result


def select_panel(candidates, adaptation_rows, warmups, direct_sources, uncertain_ids=()):
    """Apply hard exclusions, then tier preference and the fixed input hash."""
    if len({row["source_id"] for row in candidates}) != len(candidates):
        raise ValueError("Candidate canonical source IDs must be unique")
    adaptation = identity_index(adaptation_rows)
    excluded_warmups = identity_index(warmups)
    seen = {key: set() for key in IDENTITY_KEYS}
    tiers = {"A": [], "B": []}
    audit = []
    uncertain_ids = set(uncertain_ids)
    for candidate in sorted(candidates, key=lambda row: row["original_index"]):
        row = deepcopy(candidate)
        reasons = []
        reasons.extend("adaptation_overlap:" + key for key in overlap_reasons(row, adaptation))
        reasons.extend("warmup_overlap:" + key for key in overlap_reasons(row, excluded_warmups))
        reasons.extend("duplicate_prior_source_row:" + key for key in overlap_reasons(row, seen))
        if not row["prompt_ids"] or len(row["prompt_ids"]) + 2048 > 4096:
            reasons.append("input_context_inadmissible")
        for key in IDENTITY_KEYS:
            seen[key].add(row[key])
        exposure = sorted(direct_sources.get(row["source_id"], []))
        uncertain = row["source_id"] in uncertain_ids
        tier = "B" if exposure or uncertain else "A"
        row.update(exposure_tier=tier, direct_exposure_sources=exposure,
                   direct_exposure_uncertain=uncertain,
                   selection_key=selection_key(row["source_id"]))
        audit.append({key: row[key] for key in IDENTITY_KEYS} | {
            "original_index": row["original_index"], "exposure_tier": tier,
            "direct_exposure_sources": exposure, "direct_exposure_uncertain": uncertain,
            "status": "excluded" if reasons else "eligible", "exclusion_reasons": reasons,
        })
        if not reasons:
            tiers[tier].append(row)
    rank = lambda row: (row["selection_key"], row["source_id"])
    for rows in tiers.values():
        rows.sort(key=rank)
    selected = (tiers["A"] + tiers["B"])[:128]
    if len(selected) != 128:
        raise ValueError(f"Hard-eligible panel shortfall: {len(selected)} / 128")
    selected.sort(key=rank)
    for ordinal, row in enumerate(selected):
        row.update(panel_ordinal=ordinal, block_ordinal=ordinal // 32,
                   request_id=f"persistent-v2-gsm8k-{row['original_index']:04d}",
                   collect_feedback=False, max_new_tokens=2048)
    return selected, audit, {
        "eligible_tier_counts": {tier: len(rows) for tier, rows in tiers.items()},
        "selected_tier_counts": dict(Counter(row["exposure_tier"] for row in selected)),
    }


def block_permutations():
    rng = np.random.Generator(np.random.PCG64(1708))
    return [[(rng.permutation(32) + block * 32).tolist()
             for block in range(4)] for _ in range(3)]


def render_request(question, source_row, split, tokenizer, suffix, stop_token_ids):
    source_id = f"gsm8k:main:{split}:{source_row}"
    message = question + (suffix if split == "test" else "")
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": message}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )
    tokens = tokenizer.encode(rendered, add_special_tokens=False)
    return dict(source_id=source_id, dataset_id=source_id, dataset="gsm8k",
                original_index=source_row, question=question, rendered_prompt=rendered,
                prompt_ids=tokens, prompt_ids_sha256=token_digest(tokens),
                rendered_prompt_sha256=digest(rendered),
                normalized_question_sha256=digest(normalize(question)),
                max_new_tokens=2048, stop_token_ids=list(stop_token_ids),
                collect_feedback=split == "train")


def prepare(workspace, tokenizer_path, output):
    """Resolve retained identities once, and write private sealed input artifacts."""
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    root = Path(workspace) / "research"
    task = root / "dflash_persistent_prefix_withholding_20261115"
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    files = {}

    def record_file(path):
        path = Path(path)
        payload = path.read_bytes()
        files[str(path)] = {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
        return payload

    def read(path):
        return json.loads(record_file(path))

    def write(name, value):
        (output / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")

    audit_path = task / "data_audit/FRESHNESS_AUDIT.json"
    prior = read(audit_path)
    for name, expected in TOKENIZER_HASHES.items():
        record_file(Path(tokenizer_path) / name)
        if files[str(Path(tokenizer_path) / name)]["sha256"] != expected:
            raise ValueError("Native tokenizer identity mismatch: " + name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    questions, aliases = {}, {}
    for split in ("train", "test"):
        source = prior["sources"][split]["source"]
        path = Path(source["local_path"])
        record_file(path)
        if source["revision"] != REVISION or files[str(path)]["sha256"] != source["sha256"]:
            raise ValueError("Pinned source identity mismatch")
        questions[split] = [row["question"] for row in pq.read_table(path, columns=["question"]).to_pylist()]
        lookup = {normalize(text): index for index, text in enumerate(questions[split])}
        native_path = root / f"dflash_paid_learning_20261004/data_audit_sources/gsm8k_{split}.jsonl"
        native_rows = [json.loads(line) for line in record_file(native_path).splitlines() if line]
        aliases[split] = {}
        for row in native_rows:
            user = [m["content"] for m in row["conversations"] if m["role"] == "user"]
            if len(user) != 1:
                raise ValueError("Opaque source alias must resolve to one question")
            aliases[split][row["id"]] = lookup[normalize(user[0])]
        if len(aliases[split]) != len(questions[split]) or len(lookup) != len(questions[split]):
            raise ValueError("Source/alias mapping is incomplete or duplicated")

    native_root = root / "dflash_full_k_gsm8k_20261031/production_002/sealed"
    original = read(native_root / "ORIGINAL_MANIFEST.json")
    native_plan = read(native_root / "PLAN.json")
    if original["provenance"]["dataset_revision"] != REVISION:
        raise ValueError("Native ancestry source revision changed")
    stops = native_plan["evaluation_requests"][0]["stop_token_ids"]
    first = original["eval_requests"][0]
    first_question = questions["test"][aliases["test"][first["dataset_id"]]]
    decoded = tokenizer.decode(first["prompt_ids"], skip_special_tokens=False)
    user = decoded.split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0]
    if not user.startswith(first_question):
        raise ValueError("Native evaluation question and wrapper do not align")
    suffix = user[len(first_question):]
    candidates = [render_request(question, index, "test", tokenizer, suffix, stops)
                  for index, question in enumerate(questions["test"])]
    development = []
    for old in original["eval_requests"]:
        row = deepcopy(candidates[aliases["test"][old["dataset_id"]]])
        if row["prompt_ids"] != old["prompt_ids"]:
            raise ValueError("Current rendering differs from an inherited native evaluation input")
        row.update(request_id=old["request_id"], native_source_id=old["dataset_id"])
        development.append(row)
    if len(development) != 1024:
        raise ValueError("Expected the complete inherited 1024-request development inventory")
    warmups = choose_warmups(development)

    # The live recovery ancestry is checked by the campaign. Bind its complete
    # expected input stream here; never treat an unresolved alias as eligible.
    training = []
    for ordinal, old in enumerate(original["train_requests"][:1024]):
        source_row = aliases["train"][old["dataset_id"]]
        row = render_request(questions["train"][source_row], source_row, "train", tokenizer, suffix, stops)
        if row["prompt_ids"] != old["prompt_ids"] or old["event_id"] != f"event_{ordinal // 128 + 1:03d}":
            raise ValueError("Current training identity differs from the inherited stream")
        row.update(deepcopy(old), native_source_id=old["dataset_id"], native_stream_index=ordinal)
        training.append(row)
    if len(training) != 1024:
        raise ValueError("Incomplete expected adaptation ancestry through event 8")
    event7_ids = {row["request_id"] for row in training[768:896]}
    feature_path = root / "dflash_lost_repair_20261114/delivery/retained/campaign_001/census/FEATURES.json"
    schedule_path = root / "dflash_lost_repair_20261114/delivery/retained/campaign_001/training/SCHEDULE.json"
    features, schedule = read(feature_path), read(schedule_path)
    current = read(root / "dflash_recent_repair_retention_event7_20261112/delivery/final_001/retained/TRAINING_PLAN.json")
    if (files[str(feature_path)]["sha256"] != current["input_identities"][current["feature_manifest"]]["sha256"]
            or files[str(schedule_path)]["sha256"] != current["input_identities"][current["schedule"]]["sha256"]):
        raise ValueError("Event-7 native features/schedule binding changed")
    if ({Path(path).parent.name for path in features["feature_paths"]} != event7_ids
            or Counter(row["record_id"] for row in schedule["records"]) != Counter({key: 2 for key in event7_ids})):
        raise ValueError("Event-7 input identities or two-pass schedule are incomplete")
    before8 = identity_index(training[:896])
    if any(overlap_reasons(row, before8) for row in training[896:1024]):
        raise ValueError("Event-8 inputs overlap the preceding adaptation ancestry")

    direct_sources = defaultdict(list)
    checked_direct = []
    for label, relative, key in (
        ("PR44", "dflash_recent_repair_retention_event7_20261112/execution_package_001/sealed/PANEL.json", "requests"),
        ("PR46", "dflash_threatened_repair_replay_20261113/delivery/execution_002_stream/retained/sealed/PLAN.json", "probe_requests"),
        ("PR47", "dflash_native_prefix_loss_20261114/delivery/execution_002_stream/retained/sealed/PLAN.json", "probe_requests"),
        ("PR48", "dflash_native_update_size_20261115/delivery/execution_003/raw/sealed/PLAN.json", "probe_requests"),
    ):
        path = root / relative
        carrier = read(path)
        observed = []
        for row in carrier[key]:
            source_id = row.get("source_id", row["dataset_id"])
            if not source_id.startswith("gsm8k:main:test:"):
                raise ValueError("Unresolved direct-development source identity")
            expected = candidates[int(source_id.rsplit(":", 1)[1])]
            if row["prompt_ids"] != expected["prompt_ids"]:
                raise ValueError("Direct-probe source ID and rendered prompt disagree")
            direct_sources[source_id].append(label)
            observed.append(source_id)
        checked_direct.append(dict(pr=label, path=str(path), field=key, request_count=len(observed)))
    pr45_path = root / "dflash_event7_frozen_training_replay_20261113/delivery/completed_001/NATIVE7/PLAN.json"
    pr45 = read(pr45_path)
    if {Path(path).parent.name for path in pr45["feature_paths"]} != event7_ids:
        raise ValueError("PR45 frozen-training probe identities are unresolved")
    checked_direct.append(dict(pr="PR45", path=str(pr45_path), field="feature_paths",
                               request_count=128, split="train", test_requests=0))

    selected, exclusions, tier_counts = select_panel(candidates, training, warmups, direct_sources)
    known_exposure = defaultdict(list)
    for source in prior["completed_use_evidence"]:
        for index in source["source_rows"]:
            known_exposure[f"gsm8k:main:test:{index}"].append(source["source"])
    exposure = [dict(source_id=row["source_id"], exposure_tier=row["exposure_tier"],
                     direct_exposure_sources=row["direct_exposure_sources"],
                     known_prior_exposure_sources=known_exposure[row["source_id"]]) for row in selected]
    permutations = block_permutations()
    summary = dict(status="PANEL_SEALED_REFERENCE_AND_LIVE_ANCESTRY_GATES_PENDING", label=LABEL,
        source_revision=REVISION, source_test_rows=len(candidates), selected_count=len(selected),
        **tier_counts, hard_excluded_count=sum(row["status"] == "excluded" for row in exclusions),
        known_exposed_selected=sum(bool(row["known_prior_exposure_sources"]) for row in exposure),
        known_project_exposure_lower_bound=prior["proven_completed_unique_test_requests"],
        exposure_limits="Reuses the bounded PR49 lower-bound inventory; not an exhaustive negative.",
        direct_classification_limits="Explicit PR44 panel, PR45 training-only plan, PR46/47/48 probe carriers; no outcome fields used.",
        checked_direct_manifests=checked_direct, inherited_evaluation_token_arrays_matched=1024,
        expected_adaptation_inputs_through_event8=1024, event8_disjoint=True,
        event7_feature_and_schedule_binding=True, suffix=suffix,
        warmup_rule="minimum/maximum prompt-token length among inherited1024; canonical source-ID tie break",
        warmups=[{key: row[key] for key in ("source_id", "request_id", "warmup_role")} | {"prompt_tokens":len(row["prompt_ids"])} for row in warmups],
        reference_availability_used_for_selection=False, answers_or_model_outcomes_used=False,
        no_new_model_work=True, files=files)
    write("PANEL.json", dict(schema="persistent_prefix_panel_v2", label=LABEL, dataset=DATASET,
                             requests=selected, within_block_permutations=permutations, **tier_counts))
    write("WARMUPS.json", dict(requests=warmups, selection_rule=summary["warmup_rule"]))
    write("EVENT8_INPUTS.json", dict(requests=training[896:1024], own_feedback_required=True))
    write("EXPECTED_ANCESTRY.json", dict(requests=training, live_state_validation_required=True))
    write("EXCLUSIONS.json", exclusions)
    write("EXPOSURE.json", dict(label=LABEL, checked_direct_manifests=checked_direct,
                                lower_bound_only=True, selected_requests=exposure))
    write("DATA_RECEIPT.json", summary)
    write("SEAL.json", {path.name: {"bytes": path.stat().st_size,
          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
          for path in sorted(output.glob("*.json"))})
    return {key: summary[key] for key in ("status", "selected_count", "eligible_tier_counts",
                                         "selected_tier_counts", "hard_excluded_count", "known_exposed_selected")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.workspace, args.tokenizer, args.output), indent=2))


if __name__ == "__main__":
    main()
