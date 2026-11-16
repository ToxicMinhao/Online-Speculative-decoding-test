#!/usr/bin/env python3
"""Independently bound final-panel availability using two executed histories.

CPU-only local source inspection; no model/runtime imports, network or raw
prompt output. Run with the existing dflash-specforge Python (PyArrow 26.0.0).
The uncovered set is an upper bound, not a certified eligible panel.
"""
from __future__ import annotations

import hashlib
import json
import sys
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pyarrow
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
TASK = OUT.parent
NATIVE = ROOT / "research/dflash_full_k_gsm8k_20261031/production_002/sealed/ORIGINAL_MANIFEST.json"
PIN = ROOT / "research/projected_target_feasibility_20261022/data/approved/manifest.json"
IDMAP = ROOT / "research/dflash_paid_learning_20261004/data_audit_sources/gsm8k_test.jsonl"
A = ROOT / "research/adaflash_reproduction_20260930/runs/serving_c1_reference_001/gpu0_ar"
B = ROOT / "research/dflash_objective_check_20261005/retrieved_001/run_001/evaluation/code_s0_buffered_gsm8k"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def text_sha(text):
    return sha(text.encode("utf-8"))


def load(path):
    return json.loads(path.read_text())


def rows(path):
    # Only LF separates JSONL records; do not split valid embedded U+2028.
    return [json.loads(line) for line in path.read_text().split("\n") if line.strip()]


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).split())


def identity(path):
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": sha(path.read_bytes())}


def main():
    checks = []

    def require(ok, name):
        if not ok:
            raise AssertionError(name)
        checks.append(name)

    native, pin = load(NATIVE), load(PIN)
    source = pin["sources"]["gsm8k_test"]
    parquet = Path(source["local_path"])
    require(source["dataset"] == "openai/gsm8k" and source["config"] == "main"
            and source["split"] == "test", "official native-domain test source")
    require(source["revision"] == native["provenance"]["dataset_revision"],
            "parquet revision equals native original-manifest revision")
    require(parquet.stat().st_size == source["bytes"], "parquet byte-size pin")
    require(sha(parquet.read_bytes()) == source["sha256"] == source["observed_sha256"],
            "parquet SHA256 pins")
    questions = pq.read_table(parquet, columns=["question"])["question"].to_pylist()
    require(len(questions) == source["rows"], "source row count")
    require(len(set(questions)) == len(questions), "exact question uniqueness")
    require(len(set(map(normalized, questions))) == len(questions),
            "NFKC whitespace-normalized question uniqueness")
    index = {question: i for i, question in enumerate(questions)}

    opaque = rows(IDMAP)
    require(len(opaque) == len(questions), "opaque-map row count")
    require(len({row["id"] for row in opaque}) == len(opaque), "opaque-ID uniqueness")
    mapped = {}
    for row in opaque:
        user = [turn["content"] for turn in row["conversations"] if turn["role"] == "user"]
        require(len(user) == 1 and user[0] in index,
                f"opaque ID maps to one exact source question: {row['id']}")
        mapped[row["id"]] = user[0]
    require(set(mapped.values()) == set(questions), "full exact-question bijection")

    am, ar, az = load(A / "REQUEST_MANIFEST.json"), rows(A / "REQUESTS.jsonl"), load(A / "RESULT.json")
    bm, br, bz = load(B / "REQUEST_MANIFEST.json"), rows(B / "EVALUATION.jsonl"), load(B / "RESULT.json")
    require(az["status"] == az["benchmark"]["status"] == "passed", "AR completed terminal status")
    require(bz["status"] == "passed" and bz["mode"] == "evaluate", "objective-check completed evaluation status")
    require([(x["request_index"], x["dataset_id"], x["kind"], x["prompt_sha256"]) for x in ar]
            == [(x["request_index"], x["dataset_id"], x["kind"], x["prompt_sha256"]) for x in am["requests"]],
            "AR executed rows equal planned identities/order/prompt hashes")
    kinds = Counter(x["kind"] for x in ar)
    require(kinds["timed"] == az["benchmark"]["timed_requests"] == am["num_timed"],
            "AR actual timed count equals terminal and manifest")
    require(kinds["warmup"] == az["benchmark"]["warmup_requests"], "AR actual warmup count equals terminal")
    require(all(x["actual_output_token_ids_available"] and x["actual_output_token_ids"] for x in ar),
            "AR actual completion token evidence present")
    require([x["dataset_id"] for x in br] == bm["ids"] == bz["evaluated_ids"],
            "objective-check actual IDs equal manifest and terminal order")
    require(len(br) == bm["count"] == bz["count"] == bz["completed"],
            "objective-check actual count equals manifest and terminal")
    require(sha((B / "EVALUATION.jsonl").read_bytes()) == bz["output_sha256"],
            "objective-check output file matches terminal SHA256")
    require(sha(IDMAP.read_bytes()) == bm["sha256"] == bz["dataset_sha256"] == bz["active_dataset_sha256"],
            "objective-check consumed exact opaque-ID dataset bytes")
    require(all(x["output_ids"] for x in br), "objective-check actual completion token evidence present")

    prefixes, suffixes, reasoning = set(), set(), set()
    for row in am["requests"]:
        question = mapped[row["dataset_id"]]
        before, separator, after = row["prompt"].partition(question)
        require(separator == question, f"AR rendered prompt contains exact source question: {row['request_index']}")
        prefixes.add(before)
        suffixes.add(after)
        reasoning.add(after.split("<|im_end|>", 1)[0])
        require(text_sha(row["prompt"]) == row["prompt_sha256"],
                f"AR prompt-text hash: {row['request_index']}")
        require(text_sha(json.dumps(row["rendered_prompt_token_ids"])) == row["rendered_prompt_token_ids_sha256"],
                f"AR prompt-token hash: {row['request_index']}")
    require(len(prefixes) == len(suffixes) == len(reasoning) == 1, "one exact AR wrapper and reasoning suffix")
    before, after, instruction = next(iter(prefixes)), next(iter(suffixes)), next(iter(reasoning))
    for row, manifest_row in zip(br, bm["prompts"]):
        question = mapped[row["dataset_id"]]
        for key in ("dataset_id", "user_content_sha256", "prompt_text_sha256", "prompt_token_ids_sha256"):
            require(row[key] == manifest_row[key], f"objective-check manifest binding {key}: {row['request_index']}")
        require(text_sha(question + instruction) == row["user_content_sha256"],
                f"objective-check exact source-question content hash: {row['request_index']}")
        require(text_sha(before + question + after) == row["prompt_text_sha256"],
                f"objective-check exact rendered-prompt hash: {row['request_index']}")
        require(text_sha(json.dumps(row["prompt_token_ids"], separators=(",", ":"))) == row["prompt_token_ids_sha256"],
                f"objective-check prompt-token hash: {row['request_index']}")

    native_eval = native["eval_requests"]
    timed = {x["dataset_id"]: x for x in am["requests"] if x["kind"] == "timed"}
    require(len(native_eval) == native["counts"]["eval_requests"] == len(timed), "native evaluation count")
    require({x["dataset_id"] for x in native_eval} == set(timed), "native eval IDs exactly equal executed AR timed IDs")
    require(all(x["prompt_ids"] == timed[x["dataset_id"]]["rendered_prompt_token_ids"] for x in native_eval),
            "all native eval prompt-token arrays equal executed AR prompt arrays")
    a_ids, b_ids, universe = {x["dataset_id"] for x in ar}, {x["dataset_id"] for x in br}, set(mapped)
    require(a_ids | b_ids <= universe, "all actually used IDs belong to authenticated source test universe")
    excluded, uncovered = a_ids | b_ids, universe - (a_ids | b_ids)
    require(len({normalized(mapped[x]) for x in excluded}) == len(excluded),
            "excluded IDs count distinct normalized source questions")
    required = 128
    require(len(uncovered) < required, "decisive upper bound is below required panel size")

    sources = [TASK / "HANDOFF.txt", NATIVE, PIN, parquet, IDMAP]
    sources += [A / name for name in ("REQUEST_MANIFEST.json", "REQUESTS.jsonl", "RESULT.json")]
    sources += [B / name for name in ("REQUEST_MANIFEST.json", "EVALUATION.jsonl", "RESULT.json")]
    summary = {
        "schema": "persistent_prefix_freshness_independent_crosscheck_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "material_passport": {"origin_skill": "ARS-Codex 3.24.2", "origin_mode": "experiment-agent validate",
                              "verification_status": "ANALYZED", "version_label": "data_crosscheck_v1"},
        "status": "CPU_DATA_GATE_FAIL_INSUFFICIENT_UNUSED_SAME_SPLIT_REQUESTS",
        "readiness": "STOP_BEFORE_MODEL_WORK", "required_requests": required,
        "dataset": {k: source[k] for k in ("dataset", "revision", "config", "split", "rows", "sha256")},
        "actual_execution_counts": {"adaflash_ar_timed_rows": kinds["timed"], "adaflash_ar_warmup_rows": kinds["warmup"],
                                    "adaflash_ar_unique_questions": len(a_ids), "objective_check_rows": len(br),
                                    "objective_check_unique_questions": len(b_ids), "intersection": len(a_ids & b_ids),
                                    "used_union_lower_bound": len(excluded), "eligible_upper_bound": len(uncovered),
                                    "panel_shortfall_lower_bound": required - len(uncovered)},
        "scope": "Only two executed histories are needed for this upper bound; no exhaustive project exclusion union claimed.",
        "uncovered_status": "Not certified eligible; other earlier evaluation/probe/development use, adaptation and admissibility exclusions can only reduce this set.",
        "matching": {"exact_source_question_bijection": len(mapped), "normalized_distinct_questions": len(questions),
                     "native_eval_token_arrays_exact_matches": len(native_eval),
                     "objective_check_exact_content_hash_matches": len(br), "objective_check_exact_rendered_hash_matches": len(br)},
        "uncovered_source_indices": sorted(index[mapped[x]] for x in uncovered),
        "assertions_passed": len(checks),
        "new_work": {"gpu_model_work": 0, "model_forward": 0, "adam": 0, "complete_generations": 0,
                     "ssh_calls": 0, "network_calls": 0, "experiment_code_changes": 0},
        "environment": {"python": sys.version.split()[0], "executable": sys.executable, "pyarrow": pyarrow.__version__},
        "script": identity(Path(__file__)), "sources": [identity(path) for path in sources],
        "independence": "Recomputed directly from pinned source and raw history; no other agent's audit/report/results read.",
        "limitations": ["This is a local historical evidence audit, not a GPU reproducibility rerun.",
                        "No current tokenizer/model environment qualification is needed to prove the upper bound.",
                        "No efficacy or method-failure conclusion follows from a data-readiness stop."]}
    (OUT / "CROSSCHECK.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in ("status", "required_requests", "actual_execution_counts", "assertions_passed")}, indent=2))


if __name__ == "__main__":
    main()
