"""Authenticate cached inputs and seal the three depth-screen manifests.

This command is CPU-only and offline. It decodes only source questions and
historical input inventories. It never loads canonical answers or model weights.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

from experiments.mathqa_transfer.data import file_identity, token_digest, TOKENIZER_HASHES
from experiments.vfd_second_pass.data import TEST_SHA256, ORIGINAL_MANIFEST_SHA256
from .data import build_manifests, render_rows, DATASET_REVISION, TOKENIZER_REVISION


TRAIN_SHA256 = "ea82612ea9582142387730c793eb67d3b12849002bc0b7fa6f8efafa7351419d"
SOURCE_ROWS = dict(train=7473, test=1319)


def historical_overlap(train_rows, test_rows, original, inventory, newer_panel):
    """Bind history by exact old train tokens and audited official test indices.

    Keep one original receipt per matching test ID plus its full match count.
    The large source inventory remains separately identified, not recopied.
    """
    overlap = defaultdict(list)
    old_train = defaultdict(list)
    for ordinal, row in enumerate(original["train_requests"]):
        old_train[token_digest(row["prompt_ids"])].append((ordinal, row))
    matched_train = 0
    for row in train_rows:
        candidates = old_train.get(row["prompt_ids_sha256"], [])
        matches = [(ordinal, old) for ordinal, old in candidates if old["prompt_ids"] == row["prompt_ids"]]
        if matches:
            matched_train += 1
            overlap[row["source_id"]].append(dict(
                kind="retained_PR25_training_prompt_match", original_manifest_indices=[i for i, _ in matches],
                original_dataset_ids=[old["dataset_id"] for _, old in matches],
                token_ids_equal=True,
            ))
    receipt_matches = defaultdict(list)
    for index, receipt in enumerate(inventory["scan_receipts"]):
        for source_row in receipt["matched_source_rows"]:
            receipt_matches[source_row].append(index)
    if set(receipt_matches) != set(inventory["used_source_rows"]):
        raise ValueError("Historical used-source summary differs from original scan receipts")
    by_index = {row["source_row"]: row for row in test_rows}
    for source_row, indices in receipt_matches.items():
        if source_row not in by_index:
            raise ValueError("Historical GSM8K test source row is outside the pinned catalog")
        first = inventory["scan_receipts"][indices[0]]
        overlap[by_index[source_row]["source_id"]].append(dict(
            kind="retained_input_inventory_match", inventory_receipt_index=indices[0],
            matching_inventory_receipts=len(indices),
            first_receipt={k: first[k] for k in ("path", "bytes", "sha256")},
        ))
    newer_count = 0
    for ordinal, old in enumerate(newer_panel["requests"]):
        row = by_index.get(old["source_row"])
        if row is None or old["source_id"] != row["source_id"] or old["prompt_ids"] != row["prompt_ids"]:
            raise ValueError("Newer VFD panel does not match pinned test prompt tokens")
        newer_count += 1
        overlap[row["source_id"]].append(dict(
            kind="completed_VFD_second_pass_panel", panel_ordinal=ordinal,
            source_id=row["source_id"], token_ids_equal=True,
        ))
    return dict(overlap), dict(
        original_training_requests=len(original["train_requests"]),
        current_train_questions_matching_original_prompts=matched_train,
        prior_test_source_rows=len(receipt_matches), newer_vfd_panel_requests=newer_count,
        original_inventory_distinct_files=inventory["distinct_scanned_files"],
        original_inventory_unavailable_count=len(inventory["unreadable_or_unavailable"]),
        coverage="Retained prior input-inventory audit, PR25 training prompts, and subsequent VFD second-pass panel; not a global unseen-data claim",
        historical_exposure_affects_selection=False,
    )


def prepare(args):
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    started = time.monotonic()
    paths = dict(
        train=Path(args.source) / "train-00000-of-00001.parquet",
        test=Path(args.source) / "test-00000-of-00001.parquet",
        original_manifest=Path(args.original_manifest), model_pin=Path(args.model_pin),
        history_inventory=Path(args.history_inventory), newer_panel=Path(args.newer_panel),
    )
    pins = {name: file_identity(path) for name, path in paths.items()}
    for name, expected in dict(train=TRAIN_SHA256, test=TEST_SHA256,
                               original_manifest=ORIGINAL_MANIFEST_SHA256).items():
        if pins[name]["sha256"] != expected:
            raise ValueError(f"Pinned {name} input differs")
    model = json.loads(paths["model_pin"].read_text())
    tokenizer_files = {}
    for name, expected in TOKENIZER_HASHES.items():
        identity = file_identity(Path(args.tokenizer) / name)
        retained = model["target"]["files"][name]
        if identity["sha256"] != expected or identity["sha256"] != retained["sha256"] or identity["bytes"] != retained["bytes"]:
            raise ValueError(f"Tokenizer differs from both PR25 pins: {name}")
        tokenizer_files[name] = identity
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    rendered = {}
    for split in ("train", "test"):
        questions = pq.read_table(paths[split], columns=["question"]).to_pydict()["question"]
        if len(questions) != SOURCE_ROWS[split]:
            raise ValueError(f"Pinned {split} split size differs")
        rendered[split] = render_rows(questions, split, tokenizer)
    original = json.loads(paths["original_manifest"].read_text())
    inventory = json.loads(paths["history_inventory"].read_text())
    newer = json.loads(paths["newer_panel"].read_text())
    overlap, history_summary = historical_overlap(rendered["train"], rendered["test"], original, inventory, newer)
    provenance = dict(
        inputs=pins, dataset_revision=DATASET_REVISION,
        tokenizer=dict(revision=TOKENIZER_REVISION, files=tokenizer_files),
        decoded_parquet_columns=["question"], source_solutions_decoded=False,
        canonical_outputs_loaded=False, all_model_work=0, history=history_summary,
    )
    manifests = build_manifests(rendered["train"], rendered["test"],
                               historical_overlap=overlap, source_provenance=provenance)
    selected_ids = {row["source_id"] for part in manifests["manifests"].values() for row in part["requests"]}
    compact_history = dict(summary=history_summary, input_identities=pins,
                           selected_request_evidence={key: value for key, value in overlap.items() if key in selected_ids})
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    values = dict(MANIFESTS=manifests, SOURCE_PINS=provenance, HISTORY_OVERLAP=compact_history)
    values.update(manifests["manifests"])
    for name, value in values.items():
        (out / f"{name}.json").write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    summary = dict(
        status="sealed_inputs", output=str(out), elapsed_s=time.monotonic() - started,
        counts={role: part["count"] for role, part in manifests["manifests"].items()},
        historical_exposure={role: part["prior_project_use_count"] for role, part in manifests["manifests"].items()},
        exclusions=manifests["exclusions"], eligible_counts=manifests["eligible_counts"],
        source_answers_used=False, canonical_outputs_loaded=False, model_work=0,
        historical_coverage=history_summary,
    )
    (out / "PREPARATION.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    notes = ["# K+ block-depth input preparation", "", "CPU-only input preparation; no model weights, dataset solutions, or FINAL outputs loaded.", "",
             "The three manifests are disjoint by canonical source ID, normalized question/rendered prompt, and prompt token IDs.",
             "Historical exposure is disclosed without affecting selection. FINAL is auxiliary-fit-disjoint, not globally fresh.", "",
             "| Manifest | Requests | Recorded prior exposure |", "|---|---:|---:|"]
    notes.extend(f"| {role} | {part['count']} | {part['prior_project_use_count']} |" for role, part in manifests["manifests"].items())
    notes.extend(["", f"Excluded input records: {len(manifests['exclusions'])}; all reasons remain in MANIFESTS.json.",
                  "PR25's train prompts use raw questions; test prompts retain its reasoning suffix. Source wording and rendered IDs remain exact.",
                  "Train eligibility reserves the full 2,048-token cap within 2,272; test eligibility reserves it within 4,096. Valid future cap hits remain included.",
                  "", "The source history inventory is retained at its authenticated original path. This package copies compact selected-ID evidence rather than the entire inventory.",
                  "No generation, fitting, qualification or final evaluation occurred during this command."])
    (out / "PREPARATION_REPORT.md").write_text("\n".join(notes) + "\n")
    identities = {path.name: file_identity(path) for path in sorted(out.iterdir())}
    code = {name: file_identity(Path(__file__).with_name(name)) for name in ("data.py", "prepare.py")}
    seal = dict(schema="block_depth_input_seal_v1", status="sealed", files=identities,
                preparation_code=code, scientific_execution_commit="pending publication and runner seal",
                no_final_outcomes_inspected=True, model_work=0)
    (out / "SEAL.json").write_text(json.dumps(seal, indent=2, allow_nan=False) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "tokenizer", "model-pin", "original-manifest", "history-inventory", "newer-panel", "output"):
        parser.add_argument("--" + name, required=True)
    print(json.dumps(prepare(parser.parse_args()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
