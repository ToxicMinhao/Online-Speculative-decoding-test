"""Deterministic, input-only MathQA selection for the frozen-checkpoint test.

Source catalogs are not evidence that a question was used. Historical selection
carriers are inspected for input identities/text only; outcomes never select the
new panel. The immutable source archive remains outside the source repository.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import subprocess
import unicodedata
import zipfile


SOURCE_REVISION = "4f958013c164a40745a5b771dfb005df308deb4a"
SOURCE_BLOB = "193fb150706468110459ddcdc52ad48e3f0b7491"
SOURCE_ROWS = 2985
PANEL_SIZE = 1024
PROMPT_CAP = 2048
SELECTION_PREFIX = "mathqa-transfer-v1:"
TOKENIZER_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
TOKENIZER_HASHES = {
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "tokenizer_config.json": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
}
HISTORY_MARKERS = ("manifest", "panel", "select", "request", "prompt", "probe", "reserv", "window", "continuation", "plan")
NONALLOCATION_MARKERS = ("exclusion", "source", "audit", "retrieval", "retirement", "archive", "export")
INPUT_DIRECTORIES = {"inputs", "prompts", "evaluation_inputs", "support"}
PRIOR_ALLOCATION_AUDIT = "research/projected_target_feasibility_20261022/data/approved/prior_exclusion_audit.json"
IGNORED_HISTORY_FIELDS = {
    "output", "answer", "response", "completion", "reference_answer", "generated_text",
    "output_text", "response_text", "assistant", "metrics", "exclusions",
    "rejections", "candidate_pools", "sources", "source_catalog", "history_files",
    "rationale", "correct", "annotated_formula", "linear_formula", "category",
}
PROMPT_FIELDS = {"text", "prompt", "content", "value", "user", "user_text", "question", "problem",
                 "user_message", "rendered_prompt", "prompt_text", "instruction"}
TOKEN_FIELDS = {"prompt_ids", "input_ids", "prompt_token_ids", "rendered_prompt_token_ids"}
TOKEN_HASH_FIELDS = {"prompt_ids_sha256", "prompt_token_ids_sha256", "rendered_token_sha256"}


def normalize(text):
    """The handoff's NFKC, case, and whitespace normalization."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def token_digest(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def file_identity(path):
    path = Path(path)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


def read_source(path):
    """Authenticate the pinned ZIP, then expose only the two serving fields."""
    data = Path(path).read_bytes()
    blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
    if blob != SOURCE_BLOB:
        raise ValueError("MathQA archive differs from the required Git blob")
    with zipfile.ZipFile(path) as archive:
        raw = archive.read("test.json")
    rows = json.loads(raw)
    if not isinstance(rows, list) or len(rows) != SOURCE_ROWS:
        raise ValueError("MathQA test.json must contain exactly 2,985 rows")
    inputs = [{"Problem": row.get("Problem"), "options": row.get("options")}
              if isinstance(row, dict) else {} for row in rows]
    source = {**file_identity(path), "repository": "https://github.com/math-QA/math-QA",
              "revision": SOURCE_REVISION, "repository_path": "data/MathQA.zip", "git_blob": blob,
              "member": "test.json", "member_sha256": hashlib.sha256(raw).hexdigest(),
              "member_bytes": len(raw), "rows": len(rows), "serving_fields": ["Problem", "options"],
              "source_answers_used": False}
    return inputs, source


def assemble(row):
    question, options = row.get("Problem"), row.get("options")
    if not isinstance(question, str) or not question.strip():
        return None
    if not isinstance(options, str) or not options.strip():
        return None
    return question.strip() + "\n\nOptions: " + options.strip()


def render_inputs(source_rows, tokenizer):
    """Use PR25's single-user chat wrapper and reasoning-off tokenization."""
    rows = []
    for index, source in enumerate(source_rows):
        message = assemble(source)
        row = {"source_row": index, "source_id": f"mathqa:test:{index}"}
        if message is None:
            row["invalid_reason"] = "malformed_or_empty_input"
        else:
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": message}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False,
            )
            ids = tokenizer.encode(rendered, add_special_tokens=False)
            row.update(question=source["Problem"].strip(), user_message=message,
                       rendered_prompt=rendered, prompt_ids=ids, prompt_tokens=len(ids),
                       prompt_ids_sha256=token_digest(ids),
                       selection_key=hashlib.sha256((SELECTION_PREFIX + str(index)).encode()).hexdigest())
        rows.append(row)
    return rows


def history_candidates(workspace, exclude_root):
    """Discover retained input-selection carriers, including symlinked archives."""
    root = Path(workspace)
    paths = subprocess.check_output(["rg", "--files", "-L", "research", "-g", "*.json", "-g", "*.jsonl"],
                                    cwd=root, text=True).splitlines()
    ignored = {"github_checkout", "provenance", "source_cache", "source_artifacts", "preparation_dependencies",
               "source_snapshot", "SOURCE_SNAPSHOT", "source", "sources", "node_modules", "site-packages"}
    excluded = Path(exclude_root).resolve()
    selected = set()
    for name in sorted(paths):
        path = root / name
        # Catalog names apply inside the workspace, not to a server's parent
        # directory such as A800's /data/sources/minhao/.
        relative_parts = Path(name).parts
        if ignored.intersection(relative_parts) or path.resolve().is_relative_to(excluded):
            continue
        filename = path.name.lower()
        selected_carrier = (any(s in filename for s in HISTORY_MARKERS)
                            or (path.suffix == ".jsonl" and INPUT_DIRECTORIES.intersection(relative_parts)))
        if selected_carrier and not any(s in filename for s in NONALLOCATION_MARKERS):
            selected.add(str(path))
            yield path
    # This retained audit also identifies two allocated training files outside
    # the research tree. Its lookup-only source catalogs are deliberately unused.
    audit = root / PRIOR_ALLOCATION_AUDIT
    if audit.exists():
        for record in json.loads(audit.read_text())["history_files"]:
            path = Path(record["path"])
            if str(path) not in selected:
                selected.add(str(path))
                yield path


def history_index(rendered_rows):
    """Build the source lookup once for the historical-file scan."""
    stems, prompts, tokens = defaultdict(set), defaultdict(set), defaultdict(set)
    for row in rendered_rows:
        if "invalid_reason" in row:
            continue
        stems[normalize(row["question"])].add(row["source_row"])
        for field in ("user_message", "rendered_prompt"):
            prompts[normalize(row[field])].add(row["source_row"])
        tokens[row["prompt_ids_sha256"]].add(row["source_row"])
    return stems, prompts, tokens


def history_matches(value, rendered_rows, *, lookup=None):
    """Return input overlaps; ignore answers, outputs, and unused source catalogs."""
    stems, prompts, tokens = lookup if lookup is not None else history_index(rendered_rows)
    found = defaultdict(set)
    counts = Counter()

    def match_text(text):
        variants = [text, *re.findall(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", text, re.S)]
        for variant in variants:
            normalized = normalize(variant)
            for index in stems.get(normalized, ()):
                found[index].add("normalized_stem")
            for index in prompts.get(normalized, ()):
                found[index].add("normalized_prompt")
            # Options are a wrapper, not part of the question stem. Earlier
            # MathQA/AQuA prompts may carry different answer-option formatting.
            stem = re.split(r"\n\s*options\s*:", variant, maxsplit=1, flags=re.I)[0]
            for index in stems.get(normalize(stem), ()):
                found[index].add("normalized_stem_without_options")

    def scan(item, field=""):
        lower = field.lower()
        if lower in IGNORED_HISTORY_FIELDS:
            return
        if isinstance(item, dict):
            if item.get("role", item.get("from")) in ("assistant", "gpt", "chatgpt", "bing", "bard"):
                return
            if item.get("source_key") in ("mathqa_test", "mathqa:test") and type(item.get("source_index")) is int:
                found[item["source_index"]].add("source_id")
            for key, content in item.items():
                scan(content, key)
        elif isinstance(item, list):
            if lower in TOKEN_FIELDS and item and all(type(x) is int for x in item):
                counts["token_input_records"] += 1
                for index in tokens.get(token_digest(item), ()):
                    found[index].add("exact_prompt_token_ids")
            else:
                for content in item:
                    if not isinstance(content, (int, float, bool)):
                        scan(content, field)
        elif isinstance(item, str):
            if lower in PROMPT_FIELDS:
                counts["text_input_fields"] += 1
                match_text(item)
            if lower in TOKEN_HASH_FIELDS:
                for index in tokens.get(item, ()):
                    found[index].add("exact_prompt_token_hash")
            if lower in ("source_id", "dataset_id", "request_id", "id"):
                match = re.fullmatch(r"(?:mathqa_test|mathqa:test):(\d+)", item, re.I)
                if match:
                    found[int(match[1])].add("source_id")
                if re.search(r"math.?qa|aqua", item, re.I):
                    counts["mathqa_or_aqua_identity_fields"] += 1

    scan(value)
    return dict(found), dict(counts)


def inventory_history(paths, rendered_rows, progress=None):
    """Retain identities for all scanned files and every matching input."""
    seen, receipts, duplicates, failures, overlaps = {}, [], [], [], defaultdict(list)
    unavailable = []
    totals = Counter()
    lookup = history_index(rendered_rows)
    for path in paths:
        if not Path(path).is_file():
            unavailable.append(str(path))
            continue
        raw = Path(path).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        receipt = {"path": str(path), "bytes": len(raw), "sha256": digest}
        if digest in seen:
            duplicates.append({**receipt, "identical_to": seen[digest]})
            continue
        seen[digest] = str(path)
        try:
            value = ([json.loads(line) for line in raw.splitlines() if line.strip()]
                     if Path(path).suffix == ".jsonl" else json.loads(raw))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            failures.append({**receipt, "error": str(error)})
            continue
        found, counts = history_matches(value, rendered_rows, lookup=lookup)
        totals.update(counts)
        receipts.append({**receipt, "matched_source_rows": sorted(found), **counts})
        for index, reasons in found.items():
            if not 0 <= index < len(rendered_rows):
                raise ValueError("Retained MathQA source ID is outside test.json: " + str(index))
            overlaps[index].append({**receipt, "reasons": sorted(reasons)})
        if progress and len(receipts) % 1000 == 0:
            progress(f"Scanned {len(receipts)} distinct inventories; {len(overlaps)} source rows overlap")
    return dict(overlaps), {
        "schema": "mathqa_history_inventory_v1", "distinct_scanned_files": len(receipts),
        "duplicate_files": len(duplicates), "scan_receipts": receipts, "duplicates": duplicates,
        "unreadable_files": failures, "input_fields_checked": dict(totals),
        "unavailable_referenced_inventories": unavailable,
        "overlapping_source_rows": sorted(overlaps),
        "allocation_rule": "Retained selected/allocated/reserved input carriers, including unexecuted reservations; raw source catalogs are not usage.",
        "filename_markers": list(HISTORY_MARKERS), "excluded_filename_markers": list(NONALLOCATION_MARKERS),
        "allocated_jsonl_directories": sorted(INPUT_DIRECTORIES), "explicit_prior_audit": PRIOR_ALLOCATION_AUDIT,
        "ignored_fields": sorted(IGNORED_HISTORY_FIELDS),
        "limits": "Available local retained inventories only. Token-only inputs are matched by exact IDs/hashes; different-tokenizer records require retained text. No claim about inaccessible history, pretraining, or semantic near-duplicates.",
    }


def select_panel(rows, overlaps, count=PANEL_SIZE):
    """Filter in source order, then take the fixed hash-ranked panel."""
    eligible, excluded = [], []
    stems, prompts, tokens = {}, {}, {}
    for row in rows:
        index = row["source_row"]
        reason = row.get("invalid_reason")
        duplicate_of = None
        if not reason:
            stem, prompt, token = normalize(row["question"]), normalize(row["rendered_prompt"]), row["prompt_ids_sha256"]
            if row["prompt_tokens"] > PROMPT_CAP:
                reason = "rendered_prompt_over_2048_tokens"
            elif stem in stems:
                reason, duplicate_of = "duplicate_normalized_question_stem", stems[stem]
            elif prompt in prompts or token in tokens:
                reason = "duplicate_rendered_prompt"
                duplicate_of = prompts.get(prompt, tokens.get(token))
            else:
                stems[stem], prompts[prompt], tokens[token] = index, index, index
                if index in overlaps:
                    reason = "available_project_inventory_overlap"
        if reason:
            excluded.append({"source_row": index, "source_id": row["source_id"], "reason": reason,
                             "duplicate_of": duplicate_of, "evidence": overlaps.get(index, [])})
        else:
            eligible.append(row)
    ranked = sorted(eligible, key=lambda row: row["selection_key"])
    selected = [dict(row, ordinal=i, request_id=f"mathqa-transfer-{i:04d}") for i, row in enumerate(ranked[:count])]
    return {"status": "prepared" if len(selected) == count else "selection_shortfall",
            "source_rows": len(rows), "eligible_count": len(eligible), "required_count": count,
            "requests": selected, "exclusions": excluded,
            "exclusion_counts": dict(Counter(row["reason"] for row in excluded)),
            "eligible_order": [r["source_row"] for r in ranked],
            "selection": {"source_order_filter": True, "hash": "SHA-256", "prefix": SELECTION_PREFIX,
                          "row_number": "zero-based decimal", "ordering": "ascending digest", "outcomes_used": False}}


def prepare(args):
    from transformers import AutoTokenizer

    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=False)
    source_rows, provenance = read_source(args.archive)
    tokenizer_files = [file_identity(Path(args.tokenizer) / name) for name in TOKENIZER_HASHES]
    for identity in tokenizer_files:
        if identity["sha256"] != TOKENIZER_HASHES[Path(identity["path"]).name]:
            raise ValueError("Tokenizer bytes differ from the retained PR25 pin: " + identity["path"])
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    rendered = render_inputs(source_rows, tokenizer)
    paths = history_candidates(Path(args.workspace), Path(args.exclude_root))
    overlap, inventory = inventory_history(paths, rendered, progress=lambda text: print(text, flush=True))
    panel = select_panel(rendered, overlap)
    panel.update(source=provenance, tokenizer={"path": args.tokenizer, "revision": TOKENIZER_REVISION,
                 "files": tokenizer_files},
                 wrapper={"messages": "single user", "add_generation_prompt": True, "enable_thinking": False,
                          "encode_add_special_tokens": False},
                 stage="prepared input panel; campaign source/runtime/checkpoint seal remains separate")
    for name, content in (("PANEL.json", panel), ("HISTORY_INVENTORY.json", inventory)):
        (destination / name).write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    if inventory["unreadable_files"]:
        raise ValueError("Historical input inventories contain unreadable files; inspect preserved evidence before sealing")
    if panel["status"] != "prepared":
        raise ValueError(f"Only {panel['eligible_count']} eligible MathQA requests; 1,024 required")
    return {"status": "prepared", "requests": len(panel["requests"]), "eligible_count": panel["eligible_count"],
            "exclusions": panel["exclusion_counts"], "distinct_inventory_files": inventory["distinct_scanned_files"],
            "prompt_token_range": [min(r["prompt_tokens"] for r in panel["requests"]),
                                   max(r["prompt_tokens"] for r in panel["requests"])]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "tokenizer", "workspace", "exclude-root", "output"):
        parser.add_argument("--" + name, required=True)
    print(json.dumps(prepare(parser.parse_args()), indent=2))
