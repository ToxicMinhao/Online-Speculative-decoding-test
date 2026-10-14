#!/usr/bin/env python3
"""Extend the saved STABLE/SHORT selection for two updates; Python stdlib only."""

import argparse
from collections import Counter, defaultdict
import importlib.util
import json
from pathlib import Path
import random
import re
import sys


LEGACY_PATH = Path(__file__).resolve().parents[1] / "stable_short" / "build_streams.py"
_spec = importlib.util.spec_from_file_location("_two_update_stable_short", LEGACY_PATH)
legacy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(legacy)

SEED = legacy.SEED
DATASETS = legacy.DATASETS
STREAM_BLOCKS = legacy.STREAM_BLOCKS
STREAM_TYPES = (*STREAM_BLOCKS, "RANDOM_MIX")
PHASES = ("context", "support", "evaluation_then_support", "final_evaluation")
SELECTION_FIELDS = legacy.SELECTION_FIELDS
PLACEMENT_FIELDS = legacy.PLACEMENT_FIELDS
ValidationError = legacy.ValidationError
require = legacy.require
sha256 = legacy.sha256
json_bytes = legacy.json_bytes
read_selection = legacy.read_selection
validate_selection = legacy.validate_selection
split_for = legacy.split_for


def random_mix_questions(selection, replica):
    """One private RNG shuffles labels, then samples code followed by math per phase."""
    rng = random.Random(SEED + replica)
    labels = ["code"] * 16 + ["math"] * 16
    rng.shuffle(labels)
    output = []
    for phase_index in range(4):
        phase_labels = labels[phase_index * 8:(phase_index + 1) * 8]
        sampled = {}
        for label, prefix in (("code", "C"), ("math", "M")):
            pool = sorted((row for row in selection if row["replica"] == replica
                           and row["block"] == prefix + str(phase_index + 1)),
                          key=lambda row: row["position_in_block"])
            sampled[label] = iter(rng.sample(pool, phase_labels.count(label)))
        output.extend(next(sampled[label]) for label in phase_labels)
    return output


def build_streams(selection: list) -> list:
    """Build 640 user placements without altering source text, IDs, or selection."""
    original = legacy.build_streams(selection)
    streams = []
    for replica in range(1, 5):
        for source in original:
            if source["replica"] == replica:
                row = dict(source)
                row["phase"] = PHASES[(row["turn"] - 1) // 8]
                streams.append(row)
        for turn, source in enumerate(random_mix_questions(selection, replica), 1):
            row = dict(source)
            row.update(role="user", split=split_for(replica),
                       stream_id=f"R{replica}_RANDOM_MIX", stream_type="RANDOM_MIX",
                       turn=turn, phase=PHASES[(turn - 1) // 8])
            streams.append(row)
    validate_streams(selection, streams)
    return streams


def validate_streams(selection, streams):
    """Validate exact reuse and seeded mixing; execution must enforce support timing."""
    validate_selection(selection)
    require(len(streams) == 640, "streams must contain exactly 640 placements")
    sources = {row["question_id"]: row for row in selection}
    groups = defaultdict(list)
    placements = Counter()
    question_phases = defaultdict(set)
    originals = []
    for number, row in enumerate(streams, 1):
        where = f"placement {number}"
        require(isinstance(row, dict) and set(row) == SELECTION_FIELDS | PLACEMENT_FIELDS,
                f"{where}: unexpected or missing placement fields")
        qid = row["question_id"]
        require(isinstance(qid, str) and qid in sources, f"{where}: unknown question_id")
        require(json_bytes({key: row[key] for key in SELECTION_FIELDS}) == json_bytes(sources[qid]),
                f"{where}: selected question metadata/text changed")
        replica, kind, turn = row["replica"], row["stream_type"], row["turn"]
        require(isinstance(kind, str) and kind in STREAM_TYPES, f"{where}: unknown stream type")
        require(legacy.valid_int(turn, 1, 32), f"{where}: invalid turn")
        phase_index = (turn - 1) // 8
        require(row["phase"] == PHASES[phase_index], f"{where}: chronological phase mismatch")
        require(row["role"] == "user", f"{where}: non-user placement forbidden")
        require(row["split"] == split_for(replica), f"{where}: development/confirmation split mismatch")
        require(row["stream_id"] == f"R{replica}_{kind}", f"{where}: stream_id mismatch")
        require(row["block"][1:] == str(phase_index + 1), f"{where}: question from wrong phase pool")
        groups[(replica, kind)].append(row)
        placements[qid] += 1
        question_phases[qid].add(row["phase"])
        if kind != "RANDOM_MIX":
            old_row = dict(row)
            old_row["phase"] = legacy.PHASES[phase_index]
            originals.append(old_row)
    require(len(groups) == 20, "expected exactly 20 streams")
    for (replica, kind), rows in groups.items():
        where = f"R{replica}_{kind}"
        require([row["turn"] for row in rows] == list(range(1, 33)),
                f"{where}: turns not exactly 1..32 in order")
        require(len({row["question_id"] for row in rows}) == 32,
                f"{where}: question repeats within stream")
        if kind == "RANDOM_MIX":
            require(Counter(row["label"] for row in rows) == {"code": 16, "math": 16},
                    f"{where}: RANDOM_MIX must be globally balanced 16 code / 16 math")
            require([row["question_id"] for row in rows] ==
                    [row["question_id"] for row in random_mix_questions(selection, replica)],
                    f"{where}: seeded RANDOM_MIX IDs/order mismatch")
    # Restore only the phase names before applying the unchanged old contract.
    legacy.validate_streams(selection, originals)
    require(set(placements) == set(sources) and Counter(placements.values()) == {2: 128, 3: 128},
            "every question must appear twice or three times (128 questions each)")
    require(all(len(phases) == 1 for phases in question_phases.values()),
            "question crosses chronological phase pools")
    expected_order = [(replica, kind, turn) for replica in range(1, 5)
                      for kind in STREAM_TYPES for turn in range(1, 33)]
    require([(row["replica"], row["stream_type"], row["turn"]) for row in streams] == expected_order,
            "placements must use canonical replica/stream/turn order")


def preview_bytes(streams, replica, kind):
    rows = [row for row in streams if row["replica"] == replica and row["stream_type"] == kind]
    parts = [f"# Replica {replica} — {kind}\n\nAll 32 original user prompts in order. "
             "Metadata is outside prompt text. Turns 17–24 become support only after their "
             "evaluations are recorded. No responses or reference answers are included.\n"]
    for row in rows:
        longest = max((len(match.group()) for match in re.finditer(r"`+", row["prompt"])), default=0)
        fence = "`" * max(3, longest + 1)
        metadata = json.dumps({key: row[key] for key in
                               ("question_id", "source_id", "block", "label", "phase")},
                              ensure_ascii=False, sort_keys=True)
        parts.append(f"\n## Turn {row['turn']:02d}\n\nMetadata: {metadata}\n\n"
                     f"{fence}text\n{row['prompt']}\n{fence}\n")
    return "".join(parts).encode("utf-8")


def make_artifacts(selection, selection_raw):
    streams = build_streams(selection)
    stream_bytes = b"".join(json_bytes(row) for row in streams)
    original_bytes = b"".join(json_bytes(row) for row in legacy.build_streams(selection))
    sources = {label: {"dataset": dataset, "split": "train",
                       "revision": next(row["source_revision"] for row in selection if row["label"] == label)}
               for label, dataset in DATASETS.items()}
    manifest = {
        "schema_version": 1, "dataset": "dflash_two_update_20261014", "selection_seed": SEED,
        "selection_sha256": sha256(selection_raw), "streams_sha256": sha256(stream_bytes),
        "legacy_streams_sha256_before_phase_rename": sha256(original_bytes), "sources": sources,
        "builder_sha256": sha256(Path(__file__).read_bytes()),
        "legacy_builder_sha256": sha256(LEGACY_PATH.read_bytes()),
        "splits": {"development": [1, 2], "confirmation": [3, 4]},
        "stream_types": STREAM_TYPES, "original_block_order": STREAM_BLOCKS,
        "phase_schedule": [{"turns": [1 + 8 * i, 8 + 8 * i], "phase": phase}
                           for i, phase in enumerate(PHASES)],
        "support_chronology": [
            {"update": 1, "support_turns": [9, 16], "activate_after_turn": 16},
            {"update": 2, "support_turns": [17, 24], "activate_after_turn": 24,
             "requires": "Record evaluations for turns 17–24 before using their responses as support."}],
        "random_mix": {
            "seeds": {str(replica): SEED + replica for replica in range(1, 5)},
            "algorithm": "Python random.Random(seed): shuffle 16 code then 16 math labels; "
                         "for each phase, sample code then math without replacement from pools "
                         "sorted by position_in_block; consume samples in label order.",
            "balance_scope": "whole 32-turn stream only; no per-phase balance requirement"},
        "streams": [{"stream_id": f"R{replica}_{kind}", "replica": replica,
                     "stream_type": kind, "split": split_for(replica),
                     "question_ids": [row["question_id"] for row in streams
                                      if row["replica"] == replica and row["stream_type"] == kind],
                     "label_schedule": [row["label"] for row in streams
                                        if row["replica"] == replica and row["stream_type"] == kind]}
                    for replica in range(1, 5) for kind in STREAM_TYPES],
    }
    report = {
        "schema_version": 1, "status": "PASS", "streams_sha256": sha256(stream_bytes),
        "counts": {"unique_questions": 256, "replicas": 4, "streams": 20,
                   "requests_per_stream": 32, "placements": 640,
                   "placement_multiplicity": {"2": 128, "3": 128},
                   "placements_per_phase": {phase: 160 for phase in PHASES},
                   "unique_questions_per_phase": {phase: 64 for phase in PHASES}},
        "validation": {key: True for key in (
            "saved_selection_contract", "pinned_source_revisions", "original_prompts_and_ids_unchanged",
            "original_four_streams_identical_except_phase_names", "exact_shared_blocks",
            "globally_balanced_random_mix", "deterministic_random_mix_ids_and_order",
            "sampling_without_replacement", "same_phase_question_reuse_only", "disjoint_replicas",
            "no_within_stream_duplicates", "two_or_three_placements", "canonical_placement_order",
            "user_only_schema", "development_confirmation_disjoint")},
        "random_mix_phase_label_counts": {
            f"R{replica}": {phase: dict(Counter(row["label"] for row in streams
                                              if row["replica"] == replica
                                              and row["stream_type"] == "RANDOM_MIX"
                                              and row["phase"] == phase)) for phase in PHASES}
            for replica in range(1, 5)},
        "validation_scope": "Dataset structure and deterministic construction only. The runner must "
                            "enforce evaluation-before-support and response-only supervision. "
                            "No model execution, output preservation, training, or efficacy is tested here.",
    }
    artifacts = {"streams.jsonl": stream_bytes, "manifest.json": json_bytes(manifest, pretty=True),
                 "validation.json": json_bytes(report, pretty=True)}
    artifacts.update({f"preview_R{replica}_{kind}.md": preview_bytes(streams, replica, kind)
                      for replica in range(1, 5) for kind in STREAM_TYPES})
    return artifacts


def write_or_check(artifacts, output_dir, check=False):
    repository = Path(__file__).resolve().parents[2]
    output = Path(output_dir).resolve()
    require(output != repository and repository not in output.parents,
            "dataset output must be outside the public code checkout")
    legacy.write_or_check(artifacts, output_dir, check=check)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True, help="Existing 256-question selection JSONL")
    parser.add_argument("--output-dir", type=Path, required=True, help="New dataset directory outside this checkout")
    parser.add_argument("--check", action="store_true", help="Rebuild and compare exact bytes; write nothing")
    args = parser.parse_args(argv)
    try:
        selection, raw = read_selection(args.selection)
        artifacts = make_artifacts(selection, raw)
        write_or_check(artifacts, args.output_dir, check=args.check)
    except (ValidationError, OSError, UnicodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"status": "PASS", "mode": "check" if args.check else "build",
                      "questions": 256, "streams": 20, "placements": 640,
                      "artifacts": list(artifacts)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
