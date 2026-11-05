"""Read-only admission for native depth qualification and route pricing.

This milestone loads the retained K+58 export, never its old optimizer or
feedback tensors. Hashing and runtime inspection are CPU work; no model is
constructed and no scientific forward, fitting step or generation occurs here.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import time

from online_feedback.storage import write_json
from experiments.osd_components.prepare import verify_files, verify_native_runtime
from experiments.kw_v1.preflight import hardware_admission


GIB = 2 ** 30
PROJECT_ROOT = Path("/data/sources/minhao")
SOURCE_ROOT = PROJECT_ROOT / "DFlash-Online"
ORIGINAL_ROOT = PROJECT_ROOT / "runs/dflash_kplus_20261103/production_001"
EXPORT = ORIGINAL_ROOT / "campaign_001/arms/KPLUS/trainer/events/event_058/export"
BOUNDARY = ORIGINAL_ROOT / "campaign_001/arms/KPLUS/boundaries/event_058.json"
TARGET = PROJECT_ROOT / "models/qwen3_dflash_foundation_20261022/target"
WORKER_PYTHON = "/data/conda_envs/minhao/adaflash_author/bin/python"
EXPORT_FILES = {
    "model.safetensors": {
        "bytes": 2097259104,
        "sha256": "4bd8e872e4b3976b0da257f0b181f0f27c8b369e2faf386d7d833de9a1c87d67",
    },
    "config.json": {
        "bytes": 1211,
        "sha256": "5749d96966f9ce7e6a9662c9f482b8aa0d461845fb6b9ce09ac2c08311fab2aa",
    },
}
BUDGET = dict(
    readout_generations=16,
    qualification_generations_max=16,
    timed_route_measurements_max=2048,
    readout_optimizer_steps=0,
    backbone_optimizer_steps=0,
    gate_fit_rows=0,
    final_generations=0,
)
SERVING = dict(
    block_size=16, max_new_tokens=2048, context_limit=4096,
    concurrency=1, precision="bfloat16", temperature=0,
    attention_backend="fa3", capture=False,
    cache_policy="idle_full_flush_before_every_request_and_publication",
)


def identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return dict(bytes=path.stat().st_size, sha256=digest.hexdigest())


def capacity_plan():
    """Conservative additional-write ceiling; no feature pool or checkpoint copy."""
    return dict(
        source_manifests_receipts_bytes=GIB,
        initialized_sidecar_and_compact_evidence_bytes=GIB,
        graph_compiler_cache_bytes=2 * GIB,
        retrieval_and_temporary_staging_bytes=2 * GIB,
        peak_additional_bytes=6 * GIB,
        reserve_bytes=8 * GIB,
        peak_required_bytes=14 * GIB,
        host_reserve_bytes=32 * GIB,
        new_backbone_checkpoint_bytes=0,
        teacher_feature_pool_bytes=0,
        offload_assumed=False,
    )


def owned_path(path):
    """Reject unresolved escapes before reading inputs or creating receipts."""
    path = Path(path)
    if not path.is_absolute() or not path.resolve().is_relative_to(PROJECT_ROOT):
        raise ValueError("Artifact path escapes the authorized project root: " + str(path))
    return path


def validate_plan(plan):
    fixed = dict(
        schema="block_depth_qualification_v1",
        stage="native_qualification_and_pricing",
        checkpoint_path=str(EXPORT), target_path=str(TARGET),
        worker_python=WORKER_PYTHON, budget=BUDGET, serving=SERVING,
        storage=capacity_plan(),
    )
    for key, expected in fixed.items():
        if plan.get(key) != expected:
            raise ValueError("Qualification plan differs from the fixed contract: " + key)
    if not re.fullmatch(r"[0-9a-f]{40}", plan.get("code_commit", "")):
        raise ValueError("Require the exact published execution commit")
    root = owned_path(plan["output_root"])
    if not root.is_relative_to(PROJECT_ROOT / "runs/dflash_block_depth_20261105"):
        raise ValueError("Require a separate block-depth output destination")
    if Path(plan["source_root"]) != SOURCE_ROOT:
        raise ValueError("Require the qualified DFlash-Online source root")
    owned_path(plan["source_root"])
    for key in ("model_pin", "runtime_pin", "manifest_file"):
        path = owned_path(plan[key])
        if str(path) not in plan["input_identities"]:
            raise ValueError("An input file has no sealed identity: " + key)
    for name in plan["source_files"]:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Source identity escapes its checkout: " + name)
    minimum = {
        "experiments/block_depth/native_routes.py",
        "experiments/block_depth/preflight.py",
        "online_feedback/sglang_hooks.py",
        "online_feedback/native.py",
    }
    if not minimum <= set(plan["source_files"]):
        raise ValueError("Source pin omits native qualification dependencies")
    return dict(status="passed", budget=BUDGET, stage=plan["stage"])


def check_capacity(*, free_bytes, host_available_bytes, initial=True):
    for value in (free_bytes, host_available_bytes):
        if type(value) is not int or value < 0:
            raise ValueError("Capacity measurements must be nonnegative integer bytes")
    budget = capacity_plan()
    required = budget["peak_required_bytes"] if initial else budget["reserve_bytes"]
    result = dict(
        status="passed", free_bytes=free_bytes, required_bytes=required,
        host_available_bytes=host_available_bytes,
        host_required_bytes=budget["host_reserve_bytes"],
        initial_admission=initial, capacity=budget,
    )
    if free_bytes < required or host_available_bytes < budget["host_reserve_bytes"]:
        raise RuntimeError("Qualification storage/host reserve unavailable: " + repr(result))
    return result


def verify_export(path, boundary):
    """Authenticate retained weights and the completed event, without loading state."""
    if str(path) != str(EXPORT):
        raise ValueError("Require the retained event-58 K+ export, not another checkpoint")
    scientific = boundary["scientific_check"]
    if (
        boundary.get("status") != "passed" or boundary.get("arm") != "KPLUS"
        or boundary.get("event") != 58
        or boundary.get("weights") != str(EXPORT / "model.safetensors")
        or boundary.get("weights_identity") != EXPORT_FILES["model.safetensors"]
        or scientific.get("status") != "passed"
        or scientific.get("completed_event") != 58
        or scientific.get("global_step") != 7424
        or scientific.get("presentations") != 14848
        or scientific.get("export_exact") is not True
    ):
        raise ValueError("Retained K+58 boundary does not authenticate this export")
    verify_files(path, EXPORT_FILES, exact_files=True)
    config = json.loads((Path(path) / "config.json").read_text())
    if config["num_hidden_layers"] != 5 or config["block_size"] != 16:
        raise ValueError("Retained export is not the specified five-layer DFlash-b16")
    return dict(status="passed", path=str(path), files=EXPORT_FILES,
                original_event=58, original_steps=7424, optimizer_state_loaded=False)


def verify_manifests(payload):
    if payload.get("schema") != "block_depth_manifests_v1" or payload.get("study_id") != "vfd_depth35_gsm8k_v1":
        raise ValueError("Wrong depth-screen manifest")
    parts = payload["manifests"]
    if set(parts) != {"READOUT", "GATE", "FINAL"}:
        raise ValueError("Require exactly the three predeclared input partitions")
    all_ids, all_tokens = set(), set()
    from experiments.mathqa_transfer.data import normalize
    all_text = set()
    for role, count in (("READOUT", 384), ("GATE", 128), ("FINAL", 384)):
        rows = parts[role]["requests"]
        if parts[role]["count"] != count or len(rows) != count:
            raise ValueError("Manifest partition count differs: " + role)
        for ordinal, row in enumerate(rows):
            ids = tuple(row["prompt_ids"])
            text = normalize(row["question"])
            if row["panel_ordinal"] != ordinal or row["role"] != role:
                raise ValueError("Manifest order/role differs")
            if row["source_id"] in all_ids or ids in all_tokens or text in all_text:
                raise ValueError("Input partitions overlap by source ID or normalized prompt")
            all_ids.add(row["source_id"])
            all_tokens.add(ids)
            all_text.add(text)
            if (not ids or row["max_new_tokens"] != 2048
                    or len(ids) + 2048 > (4096 if role == "FINAL" else 2272)
                    or row["stop_token_ids"] != [151645, 151643]):
                raise ValueError("Manifest input/context/termination contract differs")
    return dict(status="passed", selected_requests=896, readout_generation_prefix=16,
                final_outputs_read=False, final_generations=0)


def preflight(plan, outfile, *, postflight=False):
    """Verify all admission evidence and persist failures before propagating them."""
    if not isinstance(plan, dict):
        plan = json.loads(Path(plan).read_text())
    outfile = owned_path(outfile)
    if outfile.exists():
        raise FileExistsError("Preserve the existing preflight receipt: " + str(outfile))
    result = dict(status="running", postflight=postflight, model_work=0,
                  optimizer_steps=0, code_commit=plan.get("code_commit"))
    started = time.perf_counter()
    try:
        result["plan"] = validate_plan(plan)
        if not outfile.resolve().is_relative_to(Path(plan["output_root"]).resolve()):
            raise ValueError("Admission receipt must belong to this output directory")
        for path, expected in plan["input_identities"].items():
            owned_path(path)
            if identity(path) != expected:
                raise ValueError("Sealed input changed: " + path)
        verify_files(plan["source_root"], plan["source_files"])
        result["source_files_checked"] = len(plan["source_files"])
        result["manifests"] = verify_manifests(json.loads(Path(plan["manifest_file"]).read_text()))
        result["export"] = verify_export(plan["checkpoint_path"], json.loads(BOUNDARY.read_text()))
        model_pin = json.loads(Path(plan["model_pin"]).read_text())
        verify_files(plan["target_path"], model_pin["target"]["files"], exact_files=True)
        result["target"] = dict(path=plan["target_path"], files=model_pin["target"]["files"])
        runtime_pin = json.loads(Path(plan["runtime_pin"]).read_text())
        result["runtime"] = verify_native_runtime(plan["worker_python"], runtime_pin)
        if not postflight:
            result["hardware"] = hardware_admission()
        memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        result["capacity"] = check_capacity(
            free_bytes=shutil.disk_usage(plan["output_root"]).free,
            host_available_bytes=int(memory["MemAvailable"].split()[0]) * 1024,
            initial=not postflight,
        )
        result["status"] = "passed"
        return result
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        result["wall_s"] = time.perf_counter() - started
        write_json(outfile, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("outfile", type=Path)
    parser.add_argument("--postflight", action="store_true")
    args = parser.parse_args()
    receipt = preflight(args.plan, args.outfile, postflight=args.postflight)
    print(json.dumps(dict(status=receipt["status"], wall_s=receipt["wall_s"])))
