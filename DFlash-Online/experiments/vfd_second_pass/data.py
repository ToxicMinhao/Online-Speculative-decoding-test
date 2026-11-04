"""Input-only GSM8K preparation; immutable source rows and prior-use provenance."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

from experiments.mathqa_transfer.data import (
    file_identity,
    history_candidates,
    normalize,
    token_digest,
    TOKENIZER_HASHES,
    TOKENIZER_REVISION,
    IGNORED_HISTORY_FIELDS,
    PROMPT_FIELDS,
    TOKEN_FIELDS,
    TOKEN_HASH_FIELDS,
)

POLICY = "vfd-s2-v1"
DATASET_REVISION = "740312add88f781978c0658806c59bc2815b9866"
TEST_SHA256 = "ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59"
SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
PANEL_SIZE = 200
TRAIN_SIZE = 512
ORIGINAL_MANIFEST_SHA256 = (
    "764fa339938a240259949040134581a2231de542b7c799af5a7eb85c5ea1dbd4"
)


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def render_source(path, tokenizer):
    import pyarrow.parquet as pq

    identity = file_identity(path)
    if identity["sha256"] != TEST_SHA256:
        raise ValueError("GSM8K test parquet differs from the pinned source")
    questions = pq.read_table(path, columns=["question"]).to_pydict()["question"]
    if len(questions) != 1319:
        raise ValueError("Pinned GSM8K test must contain 1319 source rows")
    rows = []
    for index, question in enumerate(questions):
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Malformed source question")
        message = question + SUFFIX
        rendered = tokenizer.apply_chat_template(
            [dict(role="user", content=message)],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        ids = tokenizer.encode(rendered, add_special_tokens=False)
        stable_id = f"gsm8k:test:{index}"
        rows.append(
            dict(
                source_row=index,
                source_id=stable_id,
                stable_id=stable_id,
                question=question,
                user_message=message,
                rendered_prompt=rendered,
                prompt_ids=ids,
                prompt_ids_sha256=token_digest(ids),
                prompt_tokens=len(ids),
                selection_key=hashlib.sha256(
                    f"{POLICY}|eval|{stable_id}".encode()
                ).hexdigest(),
            )
        )
    return rows, dict(
        **identity,
        revision=DATASET_REVISION,
        split="test",
        rows=1319,
        source_answers_used=False,
        read_columns=["question"],
    )


def prior_use(paths, source_rows, *, progress=None):
    """Match only selected input carriers, never answers or unused catalogs."""
    texts, tokens = defaultdict(set), defaultdict(set)
    for row in source_rows:
        for field in ("question", "user_message", "rendered_prompt"):
            texts[normalize(row[field])].add(row["source_row"])
        tokens[row["prompt_ids_sha256"]].add(row["source_row"])
    seen, receipts, duplicates, failures = {}, [], [], []
    overlap = defaultdict(list)
    for path in paths:
        path = Path(path)
        if not path.is_file():
            failures.append(dict(path=str(path), error="unavailable"))
            continue
        raw = path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        receipt = dict(path=str(path), bytes=len(raw), sha256=sha)
        if sha in seen:
            duplicates.append(dict(**receipt, identical_to=seen[sha]))
            continue
        seen[sha] = str(path)
        try:
            content = (
                [json.loads(line) for line in raw.splitlines() if line.strip()]
                if path.suffix == ".jsonl"
                else json.loads(raw)
            )
        except (ValueError, UnicodeDecodeError) as error:
            failures.append(dict(**receipt, error=str(error)))
            continue
        found = set()

        def scan(item, field=""):
            field = field.lower()
            if field in IGNORED_HISTORY_FIELDS:
                return
            if isinstance(item, dict):
                if item.get("role", item.get("from")) in (
                    "assistant",
                    "gpt",
                    "chatgpt",
                    "bing",
                    "bard",
                ):
                    return
                if item.get("source_key") in ("gsm8k_test", "gsm8k:test"):
                    index = item.get("source_index")
                    if type(index) is int and 0 <= index < len(source_rows):
                        found.add(index)
                for key, value in item.items():
                    scan(value, key)
            elif isinstance(item, list):
                if field in TOKEN_FIELDS and item and all(type(v) is int for v in item):
                    found.update(tokens.get(token_digest(item), ()))
                else:
                    for value in item:
                        if not isinstance(value, (int, float, bool)):
                            scan(value, field)
            elif isinstance(item, str):
                if field in PROMPT_FIELDS:
                    for text in (
                        item,
                        *re.findall(
                            r"<\|im_start\|>user\n(.*?)<\|im_end\|>", item, re.S
                        ),
                    ):
                        found.update(texts.get(normalize(text), ()))
                        if text.endswith(SUFFIX):
                            found.update(texts.get(normalize(text[: -len(SUFFIX)]), ()))
                if field in TOKEN_HASH_FIELDS:
                    found.update(tokens.get(item, ()))
                if field in ("source_id", "stable_id", "dataset_id", "id"):
                    match = re.fullmatch(r"(?:gsm8k_test|gsm8k:test):(\d+)", item)
                    if match and 0 <= int(match[1]) < len(source_rows):
                        found.add(int(match[1]))

        scan(content)
        receipts.append(dict(**receipt, matched_source_rows=sorted(found)))
        for index in found:
            overlap[index].append(receipt)
        if progress and len(receipts) % 10000 == 0:
            progress(
                f"Inspected {len(receipts)} distinct input inventories; {len(overlap)} used test rows"
            )
    return dict(overlap), dict(
        schema="vfd_prior_use_v1",
        scan_receipts=receipts,
        duplicates=duplicates,
        unreadable_or_unavailable=failures,
        distinct_scanned_files=len(receipts),
        used_source_rows=sorted(overlap),
        coverage="Available retained input inventories only; reservations count as prior use. No inaccessible-history or pretraining claim.",
    )


def select_panel(rows, training, overlap, *, history_complete=False, count=PANEL_SIZE):
    adaptation_ids = {row["dataset_id"] for row in training}
    adaptation_texts = {normalize(row["question"]) for row in training}
    adaptation_tokens = {token_digest(row["prompt_ids"]) for row in training}
    seen, eligible, exclusions = set(), [], []
    for row in rows:
        text = normalize(row["question"])
        reason = None
        if (
            row["stable_id"] in adaptation_ids
            or text in adaptation_texts
            or row["prompt_ids_sha256"] in adaptation_tokens
        ):
            reason = "adaptation_overlap"
        elif text in seen:
            reason = "duplicate_normalized_question"
        elif len(row["prompt_ids"]) + 2048 > 4096:
            reason = "serving_context_overflow"
        seen.add(text)
        if reason:
            exclusions.append(dict(source_id=row["stable_id"], reason=reason))
        else:
            eligible.append(dict(row, prior_project_use=row["source_row"] in overlap))
    ordered = sorted(
        eligible, key=lambda row: (row["prior_project_use"], row["selection_key"])
    )
    selected = [
        dict(
            row,
            panel_ordinal=i,
            original_index=row["source_row"],
            request_index=i,
            request_id=f"vfd-s2-eval-{i:04d}",
            dataset_id=row["stable_id"],
            max_new_tokens=2048,
            collect_feedback=False,
            stop_token_ids=[151645, 151643],
        )
        for i, row in enumerate(ordered[:count])
    ]
    if len(selected) != count:
        raise ValueError("Insufficient distinct train-disjoint test questions")
    return dict(
        schema="vfd_endpoint_panel_v1",
        status="prepared",
        requests=selected,
        source_count=len(rows),
        eligible_count=len(eligible),
        exclusions=exclusions,
        prior_use_count=sum(row["prior_project_use"] for row in selected),
        unused_in_available_inventories_count=sum(
            not row["prior_project_use"] for row in selected
        ),
        freshness_claim=(
            "train-disjoint"
            if not history_complete
            else "historical usage established within declared coverage"
        ),
        selection=dict(
            policy=POLICY,
            stable_id="gsm8k:test:<zero-based source row>",
            ordering="unused first, then ascending SHA-256 of vfd-s2-v1|eval|stable_id",
            normalized_prompt="NFKC + casefold + collapsed whitespace",
            outcomes_used=False,
        ),
    )


def prepare(args):
    from transformers import AutoTokenizer

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    tokenizer_files = [
        file_identity(Path(args.tokenizer) / name) for name in TOKENIZER_HASHES
    ]
    if any(
        item["sha256"] != TOKENIZER_HASHES[Path(item["path"]).name]
        for item in tokenizer_files
    ):
        raise ValueError("Tokenizer differs from PR25")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    if file_identity(args.manifest)["sha256"] != ORIGINAL_MANIFEST_SHA256:
        raise ValueError("Original ordered PR25 training/evaluation manifest changed")
    manifest = json.loads(Path(args.manifest).read_text())
    original = manifest["train_requests"][:TRAIN_SIZE]
    if len(original) != TRAIN_SIZE:
        raise ValueError("Require original first 512 requests")
    training = []
    for index, request in enumerate(original):
        if (
            request["dataset_index"] != index
            or request["event_id"] != f"event_{index // 128 + 1:03d}"
        ):
            raise ValueError("Original training stream order changed")
        decoded = tokenizer.decode(request["prompt_ids"], skip_special_tokens=False)
        match = re.fullmatch(
            r"<\|im_start\|>user\n(.*?)<\|im_end\|>\n<\|im_start\|>assistant\n<think>\n\n</think>\n\n",
            decoded,
            re.S,
        )
        if not match:
            raise ValueError("Unrecognized original training prompt wrapper")
        training.append(dict(request, question=match[1], rendered_prompt=decoded))
    rows, source = render_source(args.source, tokenizer)
    overlap, history = prior_use(
        history_candidates(args.workspace, out.parent),
        rows,
        progress=lambda text: print(text, flush=True),
    )
    panel = select_panel(rows, training, overlap, history_complete=False)
    panel.update(
        source=source,
        tokenizer=dict(revision=TOKENIZER_REVISION, files=tokenizer_files),
        wrapper=dict(
            reasoning_suffix=SUFFIX, enable_thinking=False, add_generation_prompt=True
        ),
        training_manifest=file_identity(args.manifest),
        training_original_rows_digest=digest(original),
        historical_coverage_complete=False,
    )
    old = manifest["eval_requests"]
    old_tokens = {token_digest(row["prompt_ids"]) for row in old}
    panel["original_1024_panel_overlap"] = sum(
        row["prompt_ids_sha256"] in old_tokens for row in panel["requests"]
    )
    # Reference verification is deliberately separate from input selection. The
    # seal may admit retained references only after exact prompt/runtime/source checks.
    values = {
        "PANEL.json": panel,
        "TRAINING_REQUESTS.json": training,
        "HISTORY_INVENTORY.json": history,
        "CANONICALS.json": {},
        "REFERENCE_MISSING.json": dict(
            request_ids=[row["request_id"] for row in panel["requests"]],
            count=PANEL_SIZE,
            maximum_new_generations=PANEL_SIZE,
            reason="Pending exact retained reference source and runtime verification",
        ),
    }
    reference_root, reference_inventory = getattr(
        args, "reference_root", None
    ), getattr(args, "reference_inventory", None)
    if bool(reference_root) != bool(reference_inventory):
        raise ValueError(
            "Reference reuse requires both original root and frozen archive inventory"
        )
    if reference_root:
        references, missing, audit = reuse_pr25_references(
            panel["requests"], reference_root, reference_inventory
        )
        values.update(
            {
                "CANONICALS.json": references,
                "REFERENCE_MISSING.json": missing,
                "REFERENCE_REUSE_AUDIT.json": audit,
            }
        )
    for name, value in values.items():
        (out / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )
    return dict(
        status="prepared",
        requests=len(panel["requests"]),
        training=len(training),
        prior_use_count=panel["prior_use_count"],
        old_panel_overlap=panel["original_1024_panel_overlap"],
        history_files=history["distinct_scanned_files"],
        history_failures=len(history["unreadable_or_unavailable"]),
    )


def reuse_pr25_references(requests, root, inventory_path):
    """Revalidate PR25 tokens and native sidecars at exact rendered prompts.

    Missing prompts remain missing. No old scalar count becomes a reference and
    no prompt-only match is admitted without the authenticated target/runtime pin.
    """
    from experiments.o_half_matched_lr.archive import validate_record

    root = Path(root)
    inventory = json.loads(Path(inventory_path).read_text())["files"]
    checked = {}

    def read(relative):
        path = root / relative
        identity = file_identity(path)
        expected = inventory[str(relative)]
        if (
            identity["bytes"] != expected["bytes"]
            or identity["sha256"] != expected["sha256"]
        ):
            raise ValueError(
                "Retained PR25 evidence differs from archive: " + str(relative)
            )
        checked[str(relative)] = identity
        return json.loads(path.read_text())

    plan = read("sealed/PLAN.json")
    models = read("sealed/MODEL_PIN.json")
    runtime = read("sealed/RUNTIME_PIN.json")
    canonical = read("sealed/CANONICALS.json")
    if plan["execution_source"]["commit"] != "e0a9d3c235e1efd15404c549282d21c99243a21c":
        raise ValueError("PR25 scientific execution changed")
    if (
        models["target"]["files"]["tokenizer.json"]["sha256"]
        != TOKENIZER_HASHES["tokenizer.json"]
    ):
        raise ValueError("PR25 target tokenizer binding changed")
    reference_binding = dict(
        target_files=models["target"]["files"],
        native_runtime_sha256=runtime["native_runtime_sha256"],
        python_version=runtime["python_version"],
        torch=runtime["torch"],
        transformers=runtime["transformers"],
        serving=plan["serving"],
    )
    by_prompt = {token_digest(r["prompt_ids"]): r for r in plan["evaluation_requests"]}
    references, missing = {}, []
    for request in requests:
        old = by_prompt.get(token_digest(request["prompt_ids"]))
        if old is None:
            missing.append(request["request_id"])
            continue
        if (
            old["prompt_ids"] != request["prompt_ids"]
            or old["stop_token_ids"] != request["stop_token_ids"]
            or old["max_new_tokens"] != request["max_new_tokens"]
        ):
            raise ValueError("Canonical generation settings differ")
        label = f"{old['panel_ordinal']:04d}_r0_KPLUS"
        path = f"campaign_001/evaluation/final/{label}.json"
        raw = read(path)
        expected = canonical[old["request_id"]]
        errors = validate_record(
            raw,
            old,
            raw["checkpoint_id"],
            expected,
            label=label,
            session="kplus_full_gsm8k_evaluation",
        )
        native = raw["metadata"]["native_trace_path"]
        relative = native.split("/production_001/", 1)[1]
        if read(relative) != raw["metadata"]["capture"]:
            errors.append("native_sidecar_mismatch")
        if errors:
            raise ValueError("Retained canonical invalid: " + repr(errors))
        references[request["request_id"]] = dict(
            prompt_ids=request["prompt_ids"],
            response_ids=raw["response_ids"],
            finish_reason=raw["finish_reason"],
            verified=True,
            binding_digest=digest(reference_binding),
            source="audited_PR25_exact_prompt",
            source_request_id=old["request_id"],
            source_record=checked[path],
            source_native=checked[relative],
            source_original_record_path=native.split("/production_001/", 1)[0]
            + "/production_001/"
            + path,
            source_original_native_path=native,
        )
    return (
        references,
        dict(
            request_ids=missing,
            count=len(missing),
            maximum_new_generations=len(missing),
        ),
        dict(
            status="passed",
            reused_count=len(references),
            missing_count=len(missing),
            binding=reference_binding,
            binding_digest=digest(reference_binding),
            source_inventory=file_identity(inventory_path),
            checked_files=checked,
            model_work=0,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "tokenizer", "manifest", "workspace", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--reference-root")
    parser.add_argument("--reference-inventory")
    print(json.dumps(prepare(parser.parse_args()), indent=2))
