"""Authenticate retained inputs and seal outcome-independent TRR populations.

No model is imported or evaluated here. Memory layouts are original presentation
bindings; probe selection accepts only origin identities, never predictions.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import csv
import hashlib
import json
import math
from pathlib import Path

from experiments.frozen_training_repair_replay.prepare import (
    identity, join_occurrences, read, write,
)
from experiments.recent_repair_retention.panel import verify_panel
from experiments.repair_history_margin.prepare import match_keys
from experiments.survival_confirmation.data import digest

WINDOWS = tuple(range(8, 129, 8))
EXPECTED = {
    "origin_state": "15ac28e1235851a15158e789af1480fd2d82535903b1dec5d992ec843c9ae3a1",
    "K6": "8c257620e44226c5c0469e5886224ed5d2dcf9fa2a94eddbdd10d7e36c697560",
    "K5": "9dcb0d637eabef6bd4e34c552da0448faa9510e6bcaf5c753a6bd1b43674b795",
    "schedule": "6ac18ca2c16ded078f831ab90b72b58333403e5bece9553943de491e5193a1d0",
    "feedback_rows": "90d84cdfa142e8be0eda2ef03778c190e51e74e899aac277e2e80fce32c6fda3",
    "NATIVE7": "30123d8de0bab8462a7afb77669079a65f62e771437232dcb0cba21413bed2cb",
    "NATIVE7_state": "be5cccf3a7d4cee1ca65d0df390d5f4de3d198643a97fc5125afc68442e1fa45",
}
CONSTANTS = dict(
    windows=list(WINDOWS), replay_coefficient=0.10, replay_epsilon=1e-6,
    maximum_tags=4, gamma=7, head_chunk_blocks=8, block_size=16,
    source_step=768, destination_step=896, native_steps=128,
    bootstrap_draws=10000, bootstrap_seed=1707, bootstrap_rng="PCG64",
    bootstrap_quantile="linear", card_seconds_limit=10800,
    maximum_qualification_steps=6, duplicate_native_checks=2,
    primary_unit="accepted draft proposals per fixed block",
    primary_reduction="mean starts within request, then all 16 windows, then equal requests",
)


def checked_identity(path, expected):
    """Compare bytes with a prior pin; hashing new bytes is not authentication."""
    if expected is None:
        raise ValueError(f"Missing retained identity pin: {path}")
    actual = identity(path)
    wanted = expected if isinstance(expected, str) else expected["sha256"]
    if actual["sha256"] != wanted or (isinstance(expected, dict)
            and actual["bytes"] != expected["bytes"]):
        raise ValueError(f"Retained artifact changed: {path}")
    return actual


def attempt_accounting(prior=None):
    """Bind the approved six-case retry to its failed attempt's physical work.

    The new process retains the original 168-step ceiling. Only the cumulative
    ceiling changes to 171; occupied GPU time is carried over without rounding.
    """
    accounting = dict(prior_adam_steps=0, prior_gpu_seconds=0.0,
        current_attempt_adam_limit=168, cumulative_adam_limit=168,
        cumulative_gpu_seconds_limit=CONSTANTS["card_seconds_limit"],
        current_attempt_gpu_seconds_limit=float(CONSTANTS["card_seconds_limit"]))
    if prior is None:
        return accounting
    if prior["approved_cumulative_adam_limit"] != 171:
        raise ValueError("This retry requires the approved cumulative 171-Adam limit")
    # Authenticate both files before parsing either; newly hashing them is not
    # sufficient evidence for the quantity deducted from the budget.
    checked_identity(prior["status_path"], prior["status_identity"])
    checked_identity(prior["training_work_path"], prior["training_work_identity"])
    status = read(prior["status_path"])
    if (status["status"] != "failed"
            or status["phase"] != "training_qualification_and_native_trajectory"
            or not status["execution_commit"]):
        raise ValueError("Prior attempt must have failed during training qualification")
    from .work import read_ledger
    work = read_ledger(prior["training_work_path"])
    records = [json.loads(line) for line in Path(prior["training_work_path"]).read_text().splitlines()]
    starts = {row["serial"]:row for row in records if row["phase"] == "started"}
    for row in records:
        if row["phase"] == "ledger_closed":
            continue
        if (row["phase"] not in ("started", "completed", "failed")
                or row.get("category") not in ("startup", "qualification")):
            raise ValueError("Prior attempt contains scientific or malformed work")
        start = starts[row["serial"]]
        if (row["kind"], row["category"]) != (start["kind"], start["category"]):
            raise ValueError("Prior work completion differs from its recorded start")
        if row["kind"] == "optimizer_step" and row["category"] != "qualification":
            raise ValueError("Prior Adam applications must be disposable qualification steps")
    if (not records or records[-1]["phase"] != "ledger_closed"
            or sum(row["phase"] == "ledger_closed" for row in records) != 1 or work["unknown"]
            or work["attempted"].get("optimizer_step", 0) != 3
            or work["completed"].get("optimizer_step", 0) != 3
            or work["failed"].get("optimizer_step", 0)):
        raise ValueError("Prior attempt must contain exactly three completed, known qualification Adam steps")
    seconds = status["budget"]["gpu_seconds"]
    if (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
            or not math.isfinite(seconds) or not 0 < seconds < CONSTANTS["card_seconds_limit"]
            or status["budget"]["gpu_seconds_limit"] != CONSTANTS["card_seconds_limit"]):
        raise ValueError("Prior occupied GPU time must be finite and below the original cap")
    accounting.update(prior_adam_steps=3, prior_gpu_seconds=seconds,
        cumulative_adam_limit=171,
        current_attempt_gpu_seconds_limit=CONSTANTS["card_seconds_limit"]-seconds,
        prior_execution_commit=status["execution_commit"], prior_training_work=work,
        prior_native_budget=status["budget"])
    return accounting


class RetainedInputs:
    """A caller-pinned inventory anchors compact historical receipts."""
    def __init__(self, inventory_path, inventory_identity):
        self.inventory_identity = checked_identity(inventory_path, inventory_identity)
        supplied = read(inventory_path)
        self.pins = supplied.get("files", supplied)
        self.authenticated = {str(Path(inventory_path)): self.inventory_identity}

    def verify(self, path, expected=None):
        path = Path(path)
        # Explicit nested pins are admitted only after the calling code has
        # authenticated their enclosing seal through this inventory.
        pin = self.pins.get(str(path)) if expected is None else expected
        actual = checked_identity(path, pin)
        self.authenticated[str(path)] = actual
        return actual


def authenticate_source(manifest_path, expected_identity, execution_commit):
    """Check the sealed commit and the actual staged files and symlinks."""
    manifest_identity = checked_identity(manifest_path, expected_identity)
    source = read(manifest_path)
    commit = source.get("execution_commit", source.get("commit"))
    if commit != execution_commit or source.get("dirty") is True:
        raise ValueError("Source manifest differs from the clean execution commit")
    root_value = source.get("code_path", source.get("source_path"))
    if not root_value or not source.get("files"):
        raise ValueError("Source manifest must bind the staged code path and files")
    root = Path(root_value)
    for name, expected in source["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Source entry escapes its declared checkout")
        path = root / relative
        if path.is_symlink():
            raise ValueError("A regular source-file entry became a symlink")
        checked_identity(path, expected)
    for name, target in source.get("symlinks", {}).items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Source symlink entry escapes its declared checkout")
        path = root / relative
        if not path.is_symlink() or str(path.readlink()) != target:
            raise ValueError("Staged source symlink changed")
    return dict(status="passed", execution_commit=commit, source_path=str(root),
        regular_files=len(source["files"]), symlinks=len(source.get("symlinks", {})),
        source_manifest_identity=manifest_identity,
        commit_basis="presealed clean-source manifest plus exact staged file identities")


def slot_key(row):
    return str(row["request_id"]), int(row["canonical_start"]), int(row["proposal_slot"])


def deduplicate_memory(occurrences, schedule, *, require_population=True):
    """Keep the earliest eligible *whole presentation*, including the singleton."""
    grouped = defaultdict(list)
    seen = set()
    for supplied in occurrences:
        row = deepcopy(supplied)
        key = slot_key(row)
        occurrence = (int(row["presentation_id"]), *key)
        if occurrence in seen:
            raise ValueError("Duplicate eligible memory occurrence")
        seen.add(occurrence)
        grouped[key].append(row)
    tags, layouts = [], {}
    for key, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: int(row["presentation_id"]))
        row = rows[0]
        pid = int(row["presentation_id"])
        binding = schedule["records"][pid]
        if (binding["presentation_id"], binding["record_id"], binding["pass_index"]) != (
                pid, key[0], int(row["pass_index"])):
            raise ValueError("Memory reference differs from its original presentation")
        anchors = binding["sampled_anchors"][0]
        ordinal = int(row["anchor_ordinal"])
        if not anchors["keep_mask"][0][ordinal] or not 1 <= key[2] <= 15:
            raise ValueError("Memory tag is not a valid original proposal")
        if any((other["canonical_token"], other["weight"]) !=
               (row["canonical_token"], row["weight"]) for other in rows):
            raise ValueError("Repeated memory tag changed its token or position weight")
        layout_id = f"{key[0]}:{pid}"
        tags.append(dict(row, reference_presentation_id=pid, layout_id=layout_id,
            eligible_presentation_ids=[int(item["presentation_id"]) for item in rows],
            eligibility_count=len(rows),
            sequence_start=int(anchors["positions"][0][ordinal])))
        layouts[layout_id] = dict(layout_id=layout_id, request_id=key[0],
            reference_presentation_id=pid, binding=deepcopy(binding))
    layouts = sorted(layouts.values(), key=lambda row: (
        row["request_id"], row["reference_presentation_id"]))
    census = dict(tags=len(tags), requests=len({row["request_id"] for row in tags}),
        layouts=len(layouts), eligible_occurrences=len(occurrences),
        eligibility_counts=dict(sorted(Counter(row["eligibility_count"] for row in tags).items())))
    if require_population and census != dict(tags=815, requests=128, layouts=129,
            eligible_occurrences=1629, eligibility_counts={1: 1, 2: 814}):
        raise ValueError(f"Original memory population differs: {census}")
    return tags, layouts, census


def select_probe_keys(origins, *, require_population=True):
    """Read only identity fields, so endpoint outcomes cannot enter ranking."""
    grouped, seen = defaultdict(list), set()
    for supplied in origins:
        rid, start, ordinal = (str(supplied["request_id"]),
            int(supplied["canonical_start"]), int(supplied["round_ordinal"]))
        key = (rid, start, ordinal)
        if key in seen or start < 0 or ordinal < 0:
            raise ValueError("Duplicate or invalid probe origin")
        seen.add(key)
        message = f"trr-v1-probe|{rid}|{start}|{ordinal}"
        grouped[rid].append(dict(request_id=rid, canonical_start=start,
            round_ordinal=ordinal, selection_sha256=hashlib.sha256(message.encode("utf-8")).hexdigest()))
    if require_population and (len(origins), len(grouped), min(map(len, grouped.values()))) != (2775, 64, 19):
        raise ValueError("Authenticated fixed-start source population differs; disclose before execution")
    selected = []
    for rid, rows in sorted(grouped.items()):
        chosen = sorted(rows, key=lambda row: (row["selection_sha256"],
            row["request_id"], row["canonical_start"], row["round_ordinal"]))[:4]
        selected.extend(chosen)
    if require_population and len(selected) != 256:
        raise ValueError("Expected four fixed starts per request")
    return selected


def validate_disjoint(panel_requests, original_manifest, exclusions):
    """Revalidate source/content identity against all event-one-to-seven inputs."""
    training = [row for row in original_manifest["train_requests"]
                if row["collect_feedback"] and int(row["event_id"].split("_")[-1]) <= 7]
    if len(training) != 896 or len({row["request_id"] for row in training}) != 896:
        raise ValueError("Require all 896 event-one-to-seven training requests")
    excluded = [row for row in exclusions["rows"] if row["reason"] in (
        "events_1_6_training", "PR42_discovery_batch")]
    if Counter(row["reason"] for row in excluded) != {
            "events_1_6_training": 768, "PR42_discovery_batch": 128}:
        raise ValueError("Normalized exclusion inventory is incomplete")
    by_id = {row["request_id"]: row for row in excluded}
    train_keys = set()
    for row in training:
        old = by_id[row["request_id"]]
        if (old["dataset_id"] != row["dataset_id"]
                or old["prompt_ids_sha256"] != digest(row["prompt_ids"])):
            raise ValueError("Normalized exclusion record differs from original training manifest")
        train_keys |= match_keys(old)
    seen = set()
    for row in panel_requests:
        keys = match_keys(row)
        if keys & train_keys or keys & seen:
            raise ValueError("Probe request overlaps training or another probe input")
        seen |= keys
    return dict(status="passed", training_requests=896,
        probe_requests=len(panel_requests), source_and_content_checks=True)


def write_csv(path, rows):
    path = Path(path)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def prepare(spec):
    """Seal private bindings at actual artifact paths before model execution.

    Required spec keys: previous_root (PR44), frozen_root (PR45), public_dir
    (PR44 public report), exclusions_path and exclusions_identity (the original
    normalized inventory), output, execution_commit, source_manifest and its
    source_manifest_identity, retained_identities and retained_identities_identity.
    The retained inventory must come from verified historical delivery receipts;
    it must pin the compact plan/seal/result/origins/parent files before parsing.
    An authorized retry additionally supplies prior_attempt with status_path,
    status_identity, training_work_path, training_work_identity and the approved
    cumulative Adam limit. Its authenticated work is deducted, not restarted.
    """
    previous, frozen, public = map(Path, (spec["previous_root"], spec["frozen_root"], spec["public_dir"]))
    output = Path(spec["output"])
    if output.exists():
        raise FileExistsError("Preparation requires a new output directory")
    inputs = RetainedInputs(spec["retained_identities"], spec["retained_identities_identity"])
    authenticated, verify = inputs.authenticated, inputs.verify
    source_verification = authenticate_source(spec["source_manifest"],
        spec["source_manifest_identity"], spec["execution_commit"])
    authenticated[str(Path(spec["source_manifest"]))] = source_verification["source_manifest_identity"]
    prior = spec.get("prior_attempt")
    budget_accounting = attempt_accounting(prior)
    if prior is not None:
        for stem in ("status", "training_work"):
            authenticated[str(Path(prior[stem+"_path"]))] = prior[stem+"_identity"]
        if Path(prior["status_path"]).parent.resolve() == output.parent.resolve():
            raise ValueError("Retry must preserve the prior attempt in a separate destination")
    verify(previous / "sealed/PLAN.json")
    verify(previous / "PRETRAINING_SEAL.json")
    verify(previous / "TRAINING_PLAN.json")

    original = read(previous / "sealed/PLAN.json")
    seal = read(previous / "PRETRAINING_SEAL.json")
    for path, expected in seal["inputs"].items():
        verify(path, expected)
    verify(previous / "ASSIGNMENTS.json", seal["assignments"])
    parents = []
    for path, expected in seal["parent_observations"].items():
        verify(path, expected)
        parents.extend(read(path)["rows"])
    verify(previous / "training/NATIVE/RESULT.json")
    native_result = read(previous / "training/NATIVE/RESULT.json")
    if (native_result["status"], native_result["end_step"]) != ("completed", 896):
        raise ValueError("Historical native endpoint is incomplete")
    endpoint = str(Path(native_result["export_path"]) / "model.safetensors")
    final_state = native_result["full_state_path"]
    verify(endpoint, EXPECTED["NATIVE7"])
    verify(final_state, EXPECTED["NATIVE7_state"])
    verify(original["origin_state"], EXPECTED["origin_state"])
    verify(original["schedule"], EXPECTED["schedule"])
    for name in ("K5", "K6"):
        verify(Path(original["checkpoints"][name]["export"]) / "model.safetensors", EXPECTED[name])
    old = original["original_plan"]
    # Models/runtime pins are inherited rather than silently replaced.
    for path, expected in old["pinned_files"].items():
        verify(path, expected)
    original_manifest_path = next(path for path in old["pinned_files"]
                                  if Path(path).name == "ORIGINAL_MANIFEST.json")
    original_manifest = read(original_manifest_path)
    feedback_rows = original_manifest["train_requests"][768:896]
    if (digest(feedback_rows) != EXPECTED["feedback_rows"]
            or old["requests"] != [dict(row, stop_token_ids=[151645, 151643]) for row in feedback_rows]):
        raise ValueError("Original event-seven request rows changed")
    schedule = read(original["schedule"])
    candidates = read(original["candidate_slots"])["slots"]
    verify(public / "assigned_slots.csv")
    with (public / "assigned_slots.csv").open() as stream:
        assigned_public = list(csv.DictReader(stream))
    occurrences = join_occurrences(read(previous / "ASSIGNMENTS.json")["arms"]["RECENT"],
        parents, candidates, assigned_public, schedule)
    verify(frozen / "MANIFEST.json")
    frozen_manifest = read(frozen / "MANIFEST.json")
    verify(frozen / "sealed_occurrences.json", frozen_manifest["occurrence_identity"])
    if occurrences != read(frozen / "sealed_occurrences.json"):
        raise ValueError("Reconstructed tag join differs from audited PR45")
    tags, layouts, census = deduplicate_memory(occurrences, schedule)
    features = {row["record_id"]: row["path"] for row in native_result["features"]}
    if len(features) != 128 or any(path not in authenticated for path in features.values()):
        raise ValueError("Teacher-feature pool is incomplete or unauthenticated")
    for layout in layouts:
        layout["feature_path"] = features[layout["request_id"]]

    origins_path = previous / "final_report/fixed_start_origins.csv"
    # Some historical deployments store the compact report beside a continuation;
    # permit only an explicitly supplied, authenticated equivalent path.
    if "origins_path" in spec:
        origins_path = Path(spec["origins_path"])
    verify(origins_path)
    with origins_path.open() as stream:
        keys = select_probe_keys(list(csv.DictReader(stream)))
    panel = read(original["panel_manifest"])
    verify_panel(panel)
    panel_by_id = {row["request_id"]: row for row in panel["requests"]}
    requests = [panel_by_id[rid] for rid in sorted({row["request_id"] for row in keys})]
    if set(row["request_id"] for row in requests) != set(original["path_request_ids"]):
        raise ValueError("Probe requests differ from PR44's sealed fixed-start panel")
    verify(spec["exclusions_path"], spec["exclusions_identity"])
    disjoint = validate_disjoint(requests, original_manifest, read(spec["exclusions_path"]))
    captures = {}
    parent_root = Path(spec.get("parent_records_root", previous))
    for row in requests:
        rid = row["request_id"]
        path = parent_root / "fixed_starts/requests" / rid / "PARENT_K6.json"
        verify(path)
        record = read(path)
        if (record["prompt_ids"] != row["prompt_ids"] or record["stop_token_ids"] != row["stop_token_ids"]
                or record["max_new_tokens"] != 2048):
            raise ValueError("Retained parent request differs from sealed probe input")
        for key in (key for key in keys if key["request_id"] == rid):
            block = record["metadata"]["trace"][key["round_ordinal"]]
            if block["output_start"] != key["canonical_start"]:
                raise ValueError("Selected probe ordinal differs from retained parent trace")
        captures[rid] = str(path)

    output.mkdir(parents=True)
    write(output / "MEMORY_PRIVATE.json", tags)
    write(output / "LAYOUTS.json", layouts)
    write(output / "PROBE_KEYS.json", keys)
    public_memory = [{key: row[key] for key in (
        "request_id", "canonical_start", "proposal_slot", "reference_presentation_id",
        "layout_id", "pass_index", "anchor_ordinal", "sequence_start", "weight", "eligibility_count")}
        for row in tags]
    write_csv(output / "MEMORY_REFERENCE.csv", public_memory)
    write_csv(output / "PROBE_KEYS.csv", keys)
    plan = dict(schema="threatened-repair-replay-v1", status="sealed_before_model_work",
        execution_commit=spec["execution_commit"], constants=CONSTANTS,
        prior_attempt=deepcopy(prior), budget_accounting=budget_accounting,
        native_plan_path=str(previous / "TRAINING_PLAN.json"),
        root=str(output.parent), code_path=source_verification["source_path"],
        storage=dict(maximum_additional_bytes=240 * 1024**3, reserve_bytes=8 * 1024**3),
        host_reserve_bytes=32 * 1024**3,
        serving_gpu=0, serving_gpu_identity="GPU-9986735a-3e68-d1bd-5998-9634ef24341e",
        origin_state=original["origin_state"], origin_export=original["origin_export"],
        origin_config=original["origin_config"], schedule=original["schedule"],
        feature_paths=[features[row["request_id"]] for row in old["requests"]],
        historical_native_export=native_result["export_path"], historical_native_state=final_state,
        target_path=original["target_path"], worker_python=original["worker_python"],
        training_python=original["training_python"], checkpoints=original["checkpoints"],
        serving=original["serving"], runtime_contract=old["runtime_contract"],
        memory_tags=str(output / "MEMORY_PRIVATE.json"), memory_layouts=str(output / "LAYOUTS.json"),
        probe_keys=str(output / "PROBE_KEYS.json"), probe_requests=requests,
        parent_records=captures, population=census, disjointness=disjoint,
        input_identities=authenticated, source_manifest=str(spec["source_manifest"]),
        source_verification=source_verification, retained_identities=str(spec["retained_identities"]),
        retained_identities_identity=inputs.inventory_identity,
        immutable_files={p.name: identity(p) for p in output.iterdir() if p.is_file()},
        historical_outcomes_used_for_probe_selection=False,
        feedback_batch_identity_kind="canonical_json_of_original_request_rows",
        feedback_rows_sha256=EXPECTED["feedback_rows"])
    write(output / "INPUT_IDENTITIES.json", authenticated)
    write(output / "PLAN.json", plan)
    return plan


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    args = parser.parse_args()
    result = prepare(read(args.spec))
    print(json.dumps(dict(status=result["status"], population=result["population"])))
