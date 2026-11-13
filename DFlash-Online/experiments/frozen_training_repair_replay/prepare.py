"""Bind the existing event-seven population before either frozen endpoint scores it."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

ENDPOINTS = {
    "NATIVE7": "30123d8de0bab8462a7afb77669079a65f62e771437232dcb0cba21413bed2cb",
    "RECENT7": "d873ae4455f355d8bb11bfbe771071519044155b42db051feb0d28d585eaacc9",
}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def occurrence_key(row):
    return (int(row["presentation_id"]), str(row["request_id"]),
            int(row["canonical_start"]), int(row["proposal_slot"]))


def join_occurrences(assignments, parent_rows, candidates, public_rows, schedule):
    """Exact historical join, never an endpoint-dependent eligibility decision."""
    def indexed(rows):
        result = {occurrence_key(row): row for row in rows}
        if len(result) != len(rows):
            raise ValueError("Duplicate historical occurrence identity")
        return result
    parent = indexed(parent_rows)
    public = indexed([row for row in public_rows if row["arm"] == "RECENT"])
    labels = {(str(row["request_id"]), int(row["canonical_start"]), int(row["proposal_slot"])):
              int(row["canonical_token"]) for row in candidates}
    assigned = indexed(assignments)
    if set(public) != set(assigned):
        raise ValueError("Public RECENT assignments differ from private sealed assignments")
    output = []
    for key, row in sorted(assigned.items()):
        old = parent[key]
        pid, request, start, slot = key
        binding = schedule["records"][pid]
        if (binding["record_id"], binding["pass_index"]) != (request, int(row["pass_index"])):
            raise ValueError("Assignment presentation/pass differs from the retained schedule")
        for field in ("pass_index", "anchor_ordinal", "proposal_slot", "canonical_start"):
            if int(row[field]) != int(old[field]) or int(row[field]) != int(public[key][field]):
                raise ValueError(f"Historical assignment identity differs: {field}")
        for field in ("margin", "weight"):
            if float(row[field]) != float(old[field]) or float(row[field]) != float(public[key][field]):
                raise ValueError(f"Historical assignment scalar differs: {field}")
        floor = 0.5 * min(float(old["margin"]), 1.0)
        if float(row["floor"]) != floor or float(public[key]["floor"]) != floor:
            raise ValueError("Original parent floor differs")
        y = labels[(request, start, slot)]
        if not (int(row["canonical_token"]) == int(old["canonical_token"]) == y
                and old["label_match"] and old["valid"] and old["native_correct"]
                and int(old["correct_prefix"]) >= slot):
            raise ValueError("Private canonical label/parent prefix differs")
        anchor = binding["sampled_anchors"][0]
        ordinal = int(row["anchor_ordinal"])
        if not anchor["keep_mask"][0][ordinal] or int(anchor["positions"][0][ordinal]) != int(old["sequence_start"]):
            raise ValueError("Assigned anchor is absent from the original retained layout")
        output.append(dict(presentation_id=pid, pass_index=int(row["pass_index"]),
            request_id=request, canonical_start=start, proposal_slot=slot,
            anchor_ordinal=ordinal, canonical_token=y, parent_margin=float(old["margin"]),
            floor=floor, weight=float(old["weight"])))
    counts = Counter((r["request_id"],r["canonical_start"],r["proposal_slot"]) for r in output)
    passes = Counter(r["pass_index"] for r in output)
    if (len(output), len(counts), len({r["request_id"] for r in output}), dict(passes), dict(Counter(counts.values()))) != (
            1629, 815, 128, {0:814, 1:815}, {1:1, 2:814}):
        raise ValueError("Frozen population must be 128 requests / 815 slots / 1629 occurrences with the singleton preserved")
    sanity = [r for r in output if (r["request_id"],r["canonical_start"]) == ("train_00768",3)]
    if len(sanity) != 2 or any((r["proposal_slot"],r["parent_margin"],r["floor"]) != (7,.75,.375) for r in sanity):
        raise ValueError("Prespecified train_00768 start-3 sanity check differs")
    return output


def prepare(previous_root, public_dir, output, execution_commit, source_manifest, prior_pins):
    previous_root, public_dir, output = map(Path, (previous_root, public_dir, output))
    if output.exists():
        raise FileExistsError("A fresh diagnostic destination is required")
    output.mkdir(parents=True)
    started = time.perf_counter()
    pins, authenticated = read(prior_pins), {}
    def verify(path, expected):
        path = Path(path); actual = identity(path)
        if (actual["sha256"] != expected if isinstance(expected,str) else actual != expected):
            raise ValueError(f"Historical input identity differs: {path}")
        authenticated[str(path)] = actual
        return actual
    for name, expected in pins.items():
        verify(previous_root / name, expected)
    original = read(previous_root / "sealed/PLAN.json")
    seal = read(previous_root / "PRETRAINING_SEAL.json")
    for path, expected in seal["inputs"].items():
        verify(path, expected)
    verify(previous_root / "ASSIGNMENTS.json", seal["assignments"])
    parents = []
    for path, expected in seal["parent_observations"].items():
        verify(path,expected); parents.extend(read(path)["rows"])
    for name in ("assigned_slots.csv", "parent_and_endpoint_identities.csv", "training_response_order.csv", "training_anchor_bindings.csv"):
        authenticated[str(public_dir/name)] = identity(public_dir/name)
    assignments = read(previous_root / "ASSIGNMENTS.json")["arms"]["RECENT"]
    with (public_dir / "assigned_slots.csv").open() as handle:
        public_rows = list(csv.DictReader(handle))
    schedule = read(original["schedule"])
    candidates = read(original["candidate_slots"])["slots"]
    occurrences = join_occurrences(assignments, parents, candidates, public_rows, schedule)
    write(output / "sealed_occurrences.json", occurrences)
    manifest = read(original["feature_manifest"])
    features = manifest if isinstance(manifest,list) else manifest.get("feature_paths")
    if features is None:
        features = [r.get("feature_path", r.get("feature",{}).get("path")) for r in manifest["records"]]
    if len(features) != 128 or any(str(p) not in authenticated for p in features):
        raise ValueError("Teacher-feature pool is not exactly the authenticated 128 inputs")
    expected_chunks = sum((len(r["sampled_anchors"][0]["positions"][0])+7)//8 for r in schedule["records"])
    if len(schedule["records"]) != 256 or expected_chunks != 8170:
        raise ValueError("Original forward geometry differs from the fixed work budget")
    endpoints = {}
    with (public_dir / "parent_and_endpoint_identities.csv").open() as handle:
        published = {(r["checkpoint"],r["artifact"]):r for r in csv.DictReader(handle)}
    for arm, expected in ENDPOINTS.items():
        old_arm = arm.removesuffix("7")
        result = read(previous_root / "training" / old_arm / "RESULT.json")
        if result["end_step"] != 896 or result["status"] != "completed":
            raise ValueError("Endpoint is not the completed step-896 draft")
        path = Path(result["export_path"]) / "model.safetensors"
        actual = verify(path, expected)
        if actual["sha256"] != published[(old_arm,"weights_identity")]["sha256"] or actual != result["weights_identity"]:
            raise ValueError("Endpoint differs from published PR44 identity")
        endpoints[arm] = dict(path=str(path), **actual, global_step=896)
        plan = dict(arm=arm, execution_commit=execution_commit,
            origin_config=original["origin_config"],origin_state=original["origin_state"],
            schedule=original["schedule"], feature_paths=features,
            sealed_occurrences=str(output/"sealed_occurrences.json"),
            endpoint_path=str(path),endpoint_sha256=expected,
            sealed_occurrences_identity=identity(output/"sealed_occurrences.json"),
            input_identities={p: authenticated[p] for p in
                [original["origin_config"], original["origin_state"], original["schedule"], *features]})
        write(output/f"PLAN_{arm}.json",plan)
    free = shutil.disk_usage(output).free
    # No checkpoints/teacher packets are written. Includes staged source, output,
    # compiler cache growth, retrieval overlap, then a separate 8-GiB reserve.
    required = (2+8)*1024**3
    if free < required:
        raise RuntimeError("Diagnostic disk admission failed (2 GiB writes + 8 GiB reserve)")
    gpu = subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu","--format=csv"],text=True)
    processes = subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,gpu_uuid,used_gpu_memory","--format=csv"],text=True)
    if len(processes.strip().splitlines()) != 1:
        raise RuntimeError("GPU workload present; do not launch the diagnostic concurrently")
    source = read(source_manifest)
    if source["execution_commit"] != execution_commit:
        raise ValueError("Staged source revision differs from the intended execution")
    code_root = Path(source["code_path"])
    for name, expected in source["files"].items():
        if identity(code_root/name) != expected:
            raise ValueError(f"Staged source differs: {name}")
    manifest = dict(schema="event7-frozen-training-replay-v1", status="sealed",execution_commit=execution_commit,
        previous_execution_commit=original["execution_commit"], input_identities=authenticated,
        source_manifest=identity(source_manifest), endpoints=endpoints,
        occurrence_identity=identity(output/"sealed_occurrences.json"),
        row_keys=[[r[k] for k in ("presentation_id","request_id","canonical_start","proposal_slot")] for r in occurrences],
        population=dict(requests=128,slots=815,occurrences=1629,pass_counts=[814,815],singleton_slots=1),
        work=dict(endpoints=2,response_forwards=512,projection_chunks=16340,backward_calls=0,optimizer_steps=0,generations=0),
        reduction=dict(unit="equal slots after mean over each slot's paired eligible presentations",bootstrap_unit="whole request",draws=10000,seed=1707,rng="PCG64",quantiles="linear",primary="RECENT7 minus NATIVE7 accepts-through percentage points"),
        device=dict(physical_gpu=0,gpu_inventory=gpu,active_processes=processes),
        storage=dict(free_bytes=free,maximum_additional_bytes=2*1024**3,reserve_bytes=8*1024**3),
        preparation_wall_s=time.perf_counter()-started, prior_results_used_for_selection=False)
    write(output/"MANIFEST.json",manifest)
    return manifest


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("previous-root","public-dir","output","execution-commit","source-manifest","prior-pins"):
        p.add_argument("--"+name,required=True)
    args=p.parse_args()
    result=prepare(**vars(args))
    print(json.dumps({k:result[k] for k in ("status","execution_commit","population","work","preparation_wall_s")},indent=2))
