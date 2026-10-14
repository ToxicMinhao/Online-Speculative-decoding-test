#!/usr/bin/env python3
"""Build and validate the fixed STABLE/SHORT request streams; Python stdlib only."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys


SEED = 20261014
DATASETS = {
    "code": "allenai/tulu-3-sft-personas-code",
    "math": "allenai/tulu-3-sft-personas-math",
}
STREAM_BLOCKS = {
    "STABLE_CODE": ("C1", "C2", "C3", "C4"),
    "STABLE_MATH": ("M1", "M2", "M3", "M4"),
    "SHORT_CODE_FIRST": ("C1", "M2", "C3", "M4"),
    "SHORT_MATH_FIRST": ("M1", "C2", "M3", "C4"),
}
PHASES = ("initial_context", "adaptation_support", "evaluation", "continuation")
SELECTION_FIELDS = frozenset({
    "source_dataset", "source_revision", "source_split", "source_row_index",
    "source_id", "label", "prompt", "question_id", "prompt_sha256", "replica",
    "block", "position_in_block", "sampling_rank", "seed",
})
PLACEMENT_FIELDS = frozenset({"role", "split", "stream_id", "stream_type", "turn", "phase"})


class ValidationError(ValueError):
    """An input, stream, or saved artifact fails a required invariant."""


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def normalized_prompt(prompt):
    """Collapse Unicode whitespace for duplicate detection only; never edit text."""
    return " ".join(prompt.split())


def strict_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON object key: " + key)
        result[key] = value
    return result


def read_selection(path):
    raw = Path(path).read_bytes()
    rows = []
    # JSONL delimiters are LF, not Unicode line-separator characters that may
    # legitimately occur unescaped inside an original JSON string.
    lines = raw.decode("utf-8").split("\n")
    if lines[-1] == "":
        lines.pop()
    for number, line in enumerate(lines, 1):
        require(bool(line.strip()), f"selection line {number}: blank line")
        try:
            row = json.loads(line, object_pairs_hook=strict_object)
        except (ValueError, TypeError) as exc:
            raise ValidationError(f"selection line {number}: {exc}") from exc
        rows.append(row)
    validate_selection(rows)
    return rows, raw


def valid_int(value, minimum, maximum=None):
    return type(value) is int and value >= minimum and (maximum is None or value <= maximum)


def validate_selection(rows):
    require(len(rows) == 256, "selection must have exactly 256 questions")
    seen = {key: set() for key in ("question_id", "source_id", "source_row", "prompt", "normalized", "slot", "rank")}
    counts = Counter()
    revisions = defaultdict(set)
    for index, row in enumerate(rows, 1):
        where = f"selection row {index}"
        require(isinstance(row, dict), f"{where}: expected an object")
        require(set(row) == SELECTION_FIELDS, f"{where}: unexpected or missing fields: {sorted(set(row) ^ SELECTION_FIELDS)}")
        label = row["label"]
        require(isinstance(label, str) and label in DATASETS, f"{where}: invalid label")
        require(row["source_dataset"] == DATASETS[label], f"{where}: dataset/label mismatch")
        require(row["source_split"] == "train", f"{where}: source split must be train")
        require(isinstance(row["source_revision"], str) and re.fullmatch(r"[0-9a-f]{40}", row["source_revision"]) is not None,
                f"{where}: source_revision must be a pinned 40-character lowercase commit SHA")
        revisions[label].add(row["source_revision"])
        require(valid_int(row["source_row_index"], 0), f"{where}: invalid source_row_index")
        source_id = row["source_id"]
        require((isinstance(source_id, str) and bool(source_id.strip())) or valid_int(source_id, 0),
                f"{where}: source_id must be a nonempty string or nonnegative integer")
        for key in ("question_id", "prompt"):
            require(isinstance(row[key], str) and bool(row[key].strip()), f"{where}: {key} must be nonempty text")
        require("\x00" not in row["prompt"], f"{where}: NUL in prompt")
        require(row["prompt_sha256"] == sha256(row["prompt"].encode("utf-8")), f"{where}: prompt SHA256 mismatch")
        require(valid_int(row["replica"], 1, 4), f"{where}: invalid replica")
        require(valid_int(row["position_in_block"], 1, 8), f"{where}: invalid position_in_block")
        require(valid_int(row["sampling_rank"], 1), f"{where}: sampling_rank must be positive and 1-based")
        require(type(row["seed"]) is int and row["seed"] == SEED, f"{where}: seed mismatch")
        prefix = "C" if label == "code" else "M"
        require(isinstance(row["block"], str) and row["block"] in {prefix + str(i) for i in range(1, 5)},
                f"{where}: block/label mismatch")
        keys = {
            "question_id": row["question_id"],
            "source_id": (label, type(source_id).__name__, source_id),
            "source_row": (label, row["source_row_index"]),
            "prompt": row["prompt"],
            "normalized": normalized_prompt(row["prompt"]),
            "slot": (row["replica"], row["block"], row["position_in_block"]),
            "rank": (label, row["sampling_rank"]),
        }
        for key, value in keys.items():
            require(value not in seen[key], f"{where}: duplicate {key}")
            seen[key].add(value)
        counts[(row["replica"], label)] += 1
    for replica in range(1, 5):
        for label in DATASETS:
            require(counts[(replica, label)] == 32, f"replica {replica}: expected 32 {label} questions")
    require(all(len(value) == 1 for value in revisions.values()), "each source must have exactly one revision")
    expected_slots = {(r, p + str(b), i) for r in range(1, 5) for p in ("C", "M") for b in range(1, 5) for i in range(1, 9)}
    require(seen["slot"] == expected_slots, "missing or unexpected block slots")


def split_for(replica):
    return "development" if replica <= 2 else "confirmation"


def build_streams(selection):
    validate_selection(selection)
    slots = {(r["replica"], r["block"], r["position_in_block"]): r for r in selection}
    streams = []
    for replica in range(1, 5):
        for stream_type, blocks in STREAM_BLOCKS.items():
            for block_index, block in enumerate(blocks):
                for position in range(1, 9):
                    row = dict(slots[(replica, block, position)])
                    row.update(role="user", split=split_for(replica), stream_id=f"R{replica}_{stream_type}",
                               stream_type=stream_type, turn=8 * block_index + position, phase=PHASES[block_index])
                    streams.append(row)
    validate_streams(selection, streams)
    return streams


def validate_streams(selection, streams):
    validate_selection(selection)
    require(len(streams) == 512, "streams must contain exactly 512 placements")
    source = {row["question_id"]: row for row in selection}
    groups = defaultdict(list)
    placements = Counter()
    phases = defaultdict(set)
    block_copies = defaultdict(list)
    for number, row in enumerate(streams, 1):
        where = f"placement {number}"
        require(isinstance(row, dict) and set(row) == SELECTION_FIELDS | PLACEMENT_FIELDS,
                f"{where}: unexpected or missing placement fields")
        qid = row["question_id"]
        require(isinstance(qid, str) and qid in source, f"{where}: unknown question_id")
        require(json_bytes({key: row[key] for key in SELECTION_FIELDS}) == json_bytes(source[qid]),
                f"{where}: selected question metadata/text changed")
        replica, stream_type, turn = row["replica"], row["stream_type"], row["turn"]
        require(isinstance(stream_type, str) and stream_type in STREAM_BLOCKS, f"{where}: unknown stream type")
        require(valid_int(turn, 1, 32), f"{where}: invalid turn")
        phase_index = (turn - 1) // 8
        require(row["phase"] == PHASES[phase_index], f"{where}: phase mismatch (support/evaluation leakage)")
        require(row["role"] == "user", f"{where}: assistant or non-user placement forbidden")
        require(row["split"] == split_for(replica), f"{where}: development/confirmation split mismatch")
        require(row["stream_id"] == f"R{replica}_{stream_type}", f"{where}: stream_id mismatch")
        require(row["block"] == STREAM_BLOCKS[stream_type][phase_index], f"{where}: incorrect block order")
        require(row["position_in_block"] == (turn - 1) % 8 + 1, f"{where}: incorrect within-block order")
        groups[(replica, stream_type)].append(row)
        placements[qid] += 1
        phases[qid].add(row["phase"])
    require(len(groups) == 16, "expected exactly 16 streams")
    for (replica, stream_type), rows in groups.items():
        require([row["turn"] for row in rows] == list(range(1, 33)), f"R{replica}_{stream_type}: turns not exactly 1..32 in order")
        require(len({row["question_id"] for row in rows}) == 32, f"R{replica}_{stream_type}: question repeats within stream")
        for i in range(4):
            block_rows = rows[i * 8:(i + 1) * 8]
            signature = [(r["question_id"], r["prompt"], r["prompt_sha256"]) for r in block_rows]
            block_copies[(replica, block_rows[0]["block"])].append(signature)
    require(set(placements) == set(source) and set(placements.values()) == {2}, "every question must appear exactly twice")
    require(all(len(value) == 1 for value in phases.values()), "question has multiple phases (support/evaluation leakage)")
    require(len(block_copies) == 32 and all(len(copies) == 2 and copies[0] == copies[1] for copies in block_copies.values()),
            "shared blocks must occur twice with identical IDs, text, and order")
    expected_order = [(r, kind, turn) for r in range(1, 5) for kind in STREAM_BLOCKS for turn in range(1, 33)]
    require([(r["replica"], r["stream_type"], r["turn"]) for r in streams] == expected_order,
            "placements must use canonical replica/stream/turn order")


def json_bytes(value, pretty=False):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2 if pretty else None,
                       separators=None if pretty else (",", ":"), allow_nan=False) + "\n").encode("utf-8")


def length_stats(rows):
    result = {}
    for key, values in (("characters", [len(r["prompt"]) for r in rows]),
                        ("whitespace_words", [len(r["prompt"].split()) for r in rows])):
        result[key] = {"count": len(values), "min": min(values), "max": max(values),
                       "mean": statistics.mean(values), "median": statistics.median(values), "total": sum(values)}
    return result


def preview_bytes(streams, kind):
    rows = [r for r in streams if r["replica"] == 1 and r["stream_type"] == kind]
    parts = [f"# Replica 1 — {kind}\n\nAll 32 user prompts follow in order. Metadata is outside the original prompt text.\n"]
    for row in rows:
        # Choose a fence longer than any run present in the unmodified prompt.
        longest = max((len(match.group()) for match in re.finditer(r"`+", row["prompt"])), default=0)
        fence = "`" * max(3, longest + 1)
        metadata = json.dumps({key: row[key] for key in ("question_id", "source_id", "block", "label", "phase")}, ensure_ascii=False, sort_keys=True)
        parts.append(f"\n## Turn {row['turn']:02d}\n\nMetadata: {metadata}\n\n{fence}text\n{row['prompt']}\n{fence}\n")
    return "".join(parts).encode("utf-8")


def make_artifacts(selection, selection_raw):
    streams = build_streams(selection)
    stream_bytes = b"".join(json_bytes(row) for row in streams)
    source_revisions = {label: next(r["source_revision"] for r in selection if r["label"] == label) for label in DATASETS}
    report = {
        "schema_version": 1, "status": "PASS", "seed": SEED,
        "selection_sha256": sha256(selection_raw), "streams_sha256": sha256(stream_bytes),
        "sources": {label: {"dataset": dataset, "revision": source_revisions[label], "split": "train"} for label, dataset in DATASETS.items()},
        "counts": {"unique_questions": 256, "code_questions": 128, "math_questions": 128,
                   "replicas": 4, "streams": 16, "requests_per_stream": 32, "placements": 512,
                   "questions_per_replica": 64, "questions_per_label_per_replica": 32,
                   "placements_per_question": 2, "unique_questions_per_phase": {phase: 64 for phase in PHASES},
                   "placements_per_phase": {phase: 128 for phase in PHASES}},
        "splits": {"development": [1, 2], "confirmation": [3, 4]},
        "block_order": STREAM_BLOCKS,
        "validation": {key: True for key in (
            "counts_and_labels", "pinned_source_revisions", "unique_source_ids_and_rows", "unique_question_ids",
            "unique_exact_prompts", "unique_whitespace_normalized_prompts", "disjoint_replicas",
            "complete_block_slots", "exact_shared_block_reuse", "exact_prompt_sha256", "no_repeated_question_within_stream",
            "no_cross_phase_question_reuse", "support_evaluation_disjoint", "each_question_placed_twice",
            "development_confirmation_disjoint", "canonical_placement_order", "user_only_schema")},
        "length_statistics_unique_questions": {"all": length_stats(selection), **{label: length_stats([r for r in selection if r["label"] == label]) for label in DATASETS}},
        "length_statistics_placements_by_stream_type": {kind: length_stats([r for r in streams if r["stream_type"] == kind]) for kind in STREAM_BLOCKS},
        "length_units": "Unicode code points and whitespace-delimited words; these are not model-token counts.",
        "validation_scope": "Structural integrity of the saved selection and derived streams. Source extraction, English/self-contained quality review, and sampling provenance require the accompanying selection records; no model-performance screening is performed here.",
    }
    return {"streams.jsonl": stream_bytes, "validation.json": json_bytes(report, pretty=True),
            "preview_R1_STABLE_CODE.md": preview_bytes(streams, "STABLE_CODE"),
            "preview_R1_SHORT_CODE_FIRST.md": preview_bytes(streams, "SHORT_CODE_FIRST")}


def write_or_check(artifacts, output_dir, check=False):
    directory = Path(output_dir)
    # Preflight every file before writing any, so a mismatch leaves all existing files intact.
    for name, data in artifacts.items():
        path = directory / name
        require(not path.is_symlink(), f"refusing symlink output: {name}")
        if path.exists():
            require(path.is_file() and path.read_bytes() == data, f"existing artifact differs: {name}; choose a new output directory")
        else:
            require(not check, f"missing artifact in --check mode: {name}")
    if check:
        return
    directory.mkdir(parents=True, exist_ok=True)
    for name, data in artifacts.items():
        path = directory / name
        if not path.exists():
            with path.open("xb") as handle:
                handle.write(data)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True, help="Saved 256-question selection JSONL")
    parser.add_argument("--output-dir", type=Path, required=True, help="Separate artifact directory")
    parser.add_argument("--check", action="store_true", help="Validate existing artifacts by exact rebuild; write nothing")
    args = parser.parse_args(argv)
    try:
        selection, raw = read_selection(args.selection)
        artifacts = make_artifacts(selection, raw)
        write_or_check(artifacts, args.output_dir, check=args.check)
    except (ValidationError, OSError, UnicodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"status": "PASS", "mode": "check" if args.check else "build",
                      "questions": 256, "streams": 16, "placements": 512,
                      "artifacts": list(artifacts)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
