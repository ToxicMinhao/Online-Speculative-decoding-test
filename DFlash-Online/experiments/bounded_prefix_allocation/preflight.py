"""Authenticate the staged package and complete CPU recoveries before GPU work."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

from online_feedback.storage import write_json
from .prepare import identity, load_inputs, reference_readiness, storage_plan

# Source-backed reservations, not measured peaks of this new candidate.
HOST_POLICY = dict(historical_training_reservation_bytes=47_283_793_920,
    historical_serving_reservation_bytes=13_175_410_688,
    training_reservation_bytes=60_000_000_000, serving_reservation_bytes=14_000_000_000,
    host_headroom_bytes=16 * 2**30, training_vram_bytes=40_000_000_000,
    serving_vram_bytes=72_000_000_000,
    historical_source="PR49 serving_continuation_001/RESOURCE_ADMISSION.json",
    historical_policy="Sampled process-tree peak plus12GiB trainer or4GiB serving",
    qualification_policy="At most one trainer; concurrent one serving worker only when admitted")


def host_admission(available_bytes, policy=HOST_POLICY):
    training = policy["training_reservation_bytes"]
    serving = policy["serving_reservation_bytes"]
    headroom = policy["host_headroom_bytes"]
    required = max(training, serving) + headroom
    return dict(status="passed" if available_bytes >= required else "blocked",
        available_host_bytes=available_bytes, sequential_required_bytes=required,
        concurrent_training_serving_required_bytes=training+serving+headroom,
        parallel_qualification_admitted=available_bytes >= training+serving+headroom,
        two_trainers_admitted=False, policy=dict(policy))


def validate_allocation_roots(plan):
    """Charge this campaign's prior administrative staging without adopting old assets."""
    root, source = Path(plan["root"]), Path(plan["source_root"]).parent
    owner = source.parent
    run_parent = owner / "runs/dflash_bounded_prefix_allocation_20261116"
    run_match = re.fullmatch(r"execution_([0-9]{3})", root.name)
    source_match = re.fullmatch(r"bounded_prefix_source_20261116_([0-9]{3})", source.name)
    if root.parent != run_parent or run_match is None or source_match is None:
        raise ValueError("Current allocation paths differ from this campaign's run/source names")
    prior = plan.get("prior_staging_roots", [])
    if not isinstance(prior, list) or not all(isinstance(value, str) for value in prior):
        raise ValueError("Prior staging roots must be an explicit list of paths")
    expected = [str(root), str(source), *prior]
    if len(set(expected)) != len(expected) or plan["allocation_roots"] != expected:
        raise ValueError("Charged allocation roots differ from current plus declared prior staging")
    for value in expected:
        path = Path(value)
        if not path.is_absolute() or path.resolve() != path or not path.is_dir():
            raise ValueError("Charged allocation root is missing or linked: " + value)
    for value in prior:
        path = Path(value)
        earlier_run = re.fullmatch(r"execution_([0-9]{3})", path.name)
        earlier_source = re.fullmatch(r"bounded_prefix_source_20261116_([0-9]{3})", path.name)
        if not ((path.parent == run_parent and earlier_run is not None
                 and int(earlier_run[1]) < int(run_match[1]))
                or (path.parent == owner and earlier_source is not None
                    and int(earlier_source[1]) < int(source_match[1]))):
            raise ValueError("Prior staging root is not an earlier directory of this campaign: " + value)
    return dict(status="passed", allocation_roots=expected, prior_staging_roots=prior,
                historical_project_assets_adopted=False)


def verify_source(plan):
    root = Path(plan["source_root"])
    manifest = json.loads(Path(plan["source_manifest"]).read_text())
    if manifest["commit"] != plan["execution_commit"] or manifest["dirty"]:
        raise ValueError("Require the clean published execution source")
    for name, expected in manifest["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Source manifest escapes its declared root")
        path = root / relative
        if expected.get("kind") == "symlink":
            if (not path.is_symlink() or str(path.readlink()) != expected["target"]
                    or not path.exists() or not path.resolve().is_relative_to(root.resolve())):
                raise ValueError("Staged internal source link differs: " + name)
        elif path.is_symlink() or identity(path) != expected:
            raise ValueError("Staged source differs: " + name)
    return dict(status="passed", execution_commit=manifest["commit"],
        manifest_identity=identity(plan["source_manifest"]), checked_files=len(manifest["files"]))


def runtime_inventory(plan):
    """Inspect installed versions/sources and physical inventory without CUDA init."""
    from experiments.persistent_prefix_withholding.preflight import check_environment
    from experiments.persistent_prefix_withholding.launch import check_compilers

    compilers = check_compilers()
    if plan["qualified_compilers"] != compilers:
        raise ValueError("Sealed compiler paths differ from the qualified toolchain")
    for name, value in compilers.items():
        if os.environ.get(name) != value:
            raise ValueError("Required compiler environment is not set: " + name)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    script = ("import importlib.metadata as m,json,platform,torch,transformers; "
        "print(json.dumps(dict(versions=dict(python=platform.python_version(),torch=torch.__version__,"
        "transformers=transformers.__version__),package_metadata=dict(torch=m.version('torch'),"
        "transformers=m.version('transformers')),cuda_version=torch.version.cuda,cuda_initialized=torch.cuda.is_initialized())))")
    versions = {}
    for name, key in (("training", "training_python"), ("serving", "worker_python")):
        observed = json.loads(subprocess.check_output([plan[key], "-c", script], env=environment, text=True))
        expected = dict(plan["runtime_contract"][name], cuda=plan["runtime_contract"]["serving"]["cuda"])
        check_environment(observed, expected, name)
        versions[name] = observed
    runtime = Path(plan["worker_python"]).parent.parent / "lib/python3.12/site-packages/sglang/srt"
    sources = json.loads(Path(plan["installed_runtime_identities"]).read_text())
    pinned = plan["runtime_contract"]["native_runtime_sha256"]
    actual_sources = {}
    for name in sorted(set(sources) | set(pinned)):
        actual = identity(runtime / name)
        if (name in sources and actual != sources[name]) or (name in pinned and actual["sha256"] != pinned[name]):
            raise ValueError("Installed qualified native source differs: " + name)
        actual_sources[name] = actual
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.free",
                                   "--format=csv,noheader,nounits"], text=True)
    cards = []
    for line in raw.splitlines():
        index, uuid, used, free = [x.strip() for x in line.split(",")]
        cards.append(dict(index=int(index), uuid=uuid, used_mib=int(used), free_mib=int(free)))
    if set(plan["devices"].values()) != {x["uuid"] for x in cards}:
        raise ValueError("Physical GPU identities differ")
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory",
                                        "--format=csv,noheader,nounits"], text=True)
    if processes.strip():
        raise RuntimeError("Both cards must be idle before bounded qualification")
    policy = plan["host_resource_policy"]
    if policy != HOST_POLICY:
        raise ValueError("Changed qualification resource reservation")
    if min(row["free_mib"] * 2**20 for row in cards) < policy["serving_vram_bytes"]:
        raise RuntimeError("Qualified native worker's VRAM reservation does not fit")
    available = next(int(line.split()[1])*1024 for line in Path("/proc/meminfo").read_text().splitlines()
                     if line.startswith("MemAvailable:"))
    host = host_admission(available, policy)
    if host["status"] != "passed":
        raise RuntimeError("One qualified worker plus host reserve does not fit: " + repr(host))
    return dict(versions=versions, installed_sources=actual_sources, qualified_compilers=compilers,
                physical_cards=cards, gpu_processes=processes, host=host)


def inspect_states(plan):
    """Load each retained recovery once, read-only mmap, and verify its export."""
    import torch
    import yaml
    from experiments.ok_coding.training import _export_equal
    from experiments.ok_continuation.resume import differences
    from experiments.persistent_prefix_withholding.state import (
        CONTRACT_KEY, validate_legacy_origin, validate_snapshot as validate_control)
    from .training import fork_state, validate_snapshot

    if torch.cuda.is_initialized():
        raise RuntimeError("CPU recovery check must not initialize CUDA")
    original_config = Path(plan["origin_config"]).read_text()
    expected = json.loads(Path(plan["expected_ancestry"]).read_text())["requests"]
    result, history_six = {}, None
    endpoints = {"K6": dict(state=plan["origin_state"], export=plan["origin_export"]), **plan["controls"]}
    for name, endpoint in endpoints.items():
        state = torch.load(endpoint["state"], map_location="cpu", weights_only=False, mmap=True)
        if name == "K6":
            receipt = validate_legacy_origin(state, original_config, 6, expected_cuda_rng_states=1)
            history_six = state["events"]
            config = Path(plan["configs"]["C"]).read_text()
            migrated = fork_state(state, binding=plan["binding"], config_text=config,
                                  run_id=yaml.safe_load(config)["run_id"])
            validate_snapshot(migrated, binding=plan["binding"], config_text=config,
                original_config_text=original_config, completed_event=6,
                expected_contract=migrated["algorithm_contract"],
                expected_recipe=migrated["backend"]["optimizer"]["recipe"], expected_cuda_rng_states=1)
            for key in ("draft_state_dict", "backend", "rng", "events", "global_step", "presentations"):
                if differences(state[key], migrated[key]):
                    raise ValueError("Candidate fork changed original scientific state: " + key)
            del migrated
            number = 6
        else:
            number = 8
            contract = state["algorithm_contract"]
            receipt = validate_control(state, arm=name, binding=contract[CONTRACT_KEY]["binding"],
                config_text=state["config_text"], original_config_text=original_config,
                completed_event=8, expected_contract=contract,
                expected_recipe=state["backend"]["optimizer"]["recipe"], expected_cuda_rng_states=1)
            if differences(history_six, state["events"][:6]):
                raise ValueError("Retained control changed inherited K6 history: " + name)
        actual_ids = [f["record_id"] for event in state["events"] for f in event["features"]]
        if actual_ids != [r["request_id"] for r in expected[:number*128]]:
            raise ValueError("Retained recovery ancestry differs: " + name)
        _export_equal(state, endpoint["export"])
        result[name] = dict(receipt, export_exact=True, ancestry_records=len(actual_ids),
            mmap=True, map_location="cpu", read_only_use=True, optimizer_steps=0)
        del state
        gc.collect()
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU recovery validation unexpectedly initialized CUDA")
    return result


def inspect(plan_path):
    plan = json.loads(Path(plan_path).read_text())
    root = Path(plan["root"])
    began = time.perf_counter()
    result = dict(status="running", execution_commit=plan["execution_commit"], model_work=0, inputs={})
    try:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise ValueError("CPU preflight requires CUDA_VISIBLE_DEVICES empty")
        # This is the same real loader used by collection/final consumers.
        result["input_readiness"] = reference_readiness(plan)
        from .runtime import LIMITS
        if plan["limits"] != LIMITS or plan.get("new_reference_generations") != 0:
            raise ValueError("Changed bounded work or reference-generation contract")
        for path, expected in plan["input_identities"].items():
            actual = identity(path)
            if actual != expected:
                raise ValueError("Required immutable input changed: " + path)
            result["inputs"][path] = actual
        result["source"] = verify_source(plan)
        result["runtime"] = runtime_inventory(plan)
        result["states"] = inspect_states(plan)
        inputs = load_inputs(plan)["event8_inputs"]["requests"]
        config = json.loads((Path(plan["origin_export"]) / "config.json").read_text())
        measured = storage_plan([len(r["prompt_ids"]) for r in inputs],
            weight_bytes=Path(plan["origin_export"] + "/model.safetensors").stat().st_size,
            recovery_bytes=max(Path(plan["origin_state"]).stat().st_size,
                               *(Path(v["state"]).stat().st_size for v in plan["controls"].values())),
            hidden_size=config["hidden_size"], captured_layers=len(config["dflash_config"]["target_layer_ids"]))
        if measured != plan["storage"]:
            raise ValueError("Measured storage geometry differs from the sealed plan")
        free = shutil.disk_usage(root).free
        result["storage"] = dict(measured, actual_free_bytes=free)
        if measured["status"] != "passed" or free < measured["required_initial_free_bytes"]:
            raise RuntimeError("Complete peak allocation and free reserve do not fit")
        baseline = plan["compiler_baseline_allocated_bytes"]
        if (type(baseline) is not int or baseline < 0
                or plan["compiler_growth_reserved_bytes"] != measured["compiler_bytes"]):
            raise ValueError("Compiler baseline or reserved growth differs")
        result["charged_allocation"] = validate_allocation_roots(plan)
        result.update(status="passed", cuda_initialized=False,
                      parallel_qualification_admitted=result["runtime"]["host"]["parallel_qualification_admitted"])
    except BaseException as error:
        result.update(status="failed", error=repr(error), error_path=getattr(error, "filename", None))
        raise
    finally:
        result["wall_s"] = time.perf_counter() - began
        write_json(root / "CPU_PREFLIGHT.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    args = parser.parse_args()
    print(json.dumps(inspect(args.plan)))


if __name__ == "__main__":
    main()
