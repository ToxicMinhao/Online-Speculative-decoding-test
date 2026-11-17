"""CPU-only assembled-package, complete-origin/control and peak-resource admission."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from online_feedback.storage import write_json
from experiments.bounded_prefix_allocation.preflight import verify_source, host_admission
from experiments.bounded_prefix_allocation.preflight import HOST_POLICY
from experiments.persistent_prefix_withholding.preflight import check_environment
from .prepare import identity, load_inputs, reference_readiness, storage_plan, plan_binding, DEVICES
from .runtime import LIMITS, ALLOCATION_BYTES, RESERVE_BYTES


def validate_allocation_roots(plan):
    """Charge only this new run and its source tree, never historical assets."""
    root, source = Path(plan["root"]), Path(plan["source_root"])
    expected = [str(root), str(source)]
    if (plan["allocation_roots"] != expected or root == source
            or root.is_relative_to(source) or source.is_relative_to(root)):
        raise ValueError("Allocation roots must be disjoint new execution/source directories")
    for path in (root, source):
        if not path.is_absolute() or path.resolve() != path or not path.is_dir():
            raise ValueError("Allocation path is absent or linked: " + str(path))
    if (not root.name.startswith("execution_")
            or root.parent.name != "dflash_unchanged_c_mathqa_transfer_20261117"
            or not source.name.startswith("DFlash-Online-unchanged-c-mathqa-transfer-20261117-")):
        raise ValueError("Allocation root belongs to another experiment")
    return dict(status="passed", allocation_roots=expected, old_assets_charged=False)


def runtime_inventory(plan):
    """Inspect installed imports and native source bytes without initializing CUDA."""
    from experiments.persistent_prefix_withholding.launch import check_compilers
    compilers = check_compilers()
    if compilers != plan["qualified_compilers"] or any(os.environ.get(k) != v for k, v in compilers.items()):
        raise ValueError("Qualified compiler environment differs")
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    code = ("import importlib.metadata as m,json,platform,torch,transformers; "
        "print(json.dumps(dict(versions=dict(python=platform.python_version(),torch=torch.__version__,"
        "transformers=transformers.__version__),package_metadata=dict(torch=m.version('torch'),"
        "transformers=m.version('transformers')),cuda_version=torch.version.cuda,cuda_initialized=torch.cuda.is_initialized())))")
    versions = {}
    for name, key in (("training", "training_python"), ("serving", "worker_python")):
        observed = json.loads(subprocess.check_output([plan[key], "-c", code], env=environment, text=True))
        expected = dict(plan["runtime_contract"][name], cuda=plan["runtime_contract"]["serving"]["cuda"])
        check_environment(observed, expected, name)
        versions[name] = observed
    runtime = Path(plan["worker_python"]).parent.parent / "lib/python3.12/site-packages/sglang/srt"
    sources = {}
    for name, expected in plan["runtime_contract"]["native_runtime_sha256"].items():
        actual = identity(runtime / name)
        if actual["sha256"] != expected:
            raise ValueError("Historical native verifier source differs: " + name)
        sources[name] = actual
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.free",
                                   "--format=csv,noheader,nounits"], text=True)
    cards = []
    for line in raw.splitlines():
        index, uuid, used, free = [value.strip() for value in line.split(",")]
        cards.append(dict(index=int(index), uuid=uuid, used_mib=int(used), free_mib=int(free)))
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory",
                                        "--format=csv,noheader,nounits"], text=True)
    if plan["devices"] != DEVICES or set(plan["devices"].values()) != {r["uuid"] for r in cards}:
        raise ValueError("Physical A800 identities differ")
    if processes.strip():
        raise RuntimeError("Both A800s must be idle before admission")
    if plan["host_resource_policy"] != HOST_POLICY:
        raise ValueError("Host resource reservation changed")
    if min(r["free_mib"] * 2**20 for r in cards) < HOST_POLICY["serving_vram_bytes"]:
        raise RuntimeError("Native serving VRAM reservation does not fit")
    available = next(int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines()
                     if line.startswith("MemAvailable:"))
    host = host_admission(available)
    if host["status"] != "passed":
        raise RuntimeError("Single-worker host memory admission failed")
    return dict(versions=versions, installed_sources=sources, qualified_compilers=compilers,
                physical_cards=cards, gpu_processes=processes, host=host)


def inspect_states(plan, *, expected_cuda_rng_states=1):
    """Authenticate the origin, C metadata-only fork and retained controls, read-only."""
    import torch
    import yaml
    from experiments.ok_coding.training import _export_equal
    from experiments.ok_continuation.resume import differences
    from .state import validate_origin, fork_state, validate_snapshot
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU recovery validation must not initialize CUDA")
    state = torch.load(plan["origin_state"], map_location="cpu", weights_only=False, mmap=True)
    original_text = Path(plan["origin_config"]).read_text()
    receipt = validate_origin(state, original_text, expected_cuda_rng_states=expected_cuda_rng_states)
    status = json.loads(Path(plan["origin_status"]).read_text())
    if (status["status"] != "completed" or status["dataset"] != "mathqa" or status["arm"] != "KPLUS"
            or status["binding"] != plan["origin_binding"] or status["boundary"]["event"] != 4
            or status["boundary"]["state_identity"] != plan["origin_state_identity"]
            or status["boundary"]["weights_identity"] != plan["origin_weights_identity"]):
        raise ValueError("Completed MathQA KPLUS4 receipt differs from its authenticated origin")
    expected = json.loads(Path(plan["expected_ancestry"]).read_text())["train_requests"]
    actual = [r["record_id"] for event in state["events"] for r in event["features"]]
    if len(expected) != 512 or actual != [r["request_id"] for r in expected]:
        raise ValueError("Original 512-response adaptation ancestry differs")
    _export_equal(state, plan["origin_export"])
    forks = {}
    for arm in ("C",):
        config = Path(plan["configs"][arm]).read_text()
        own, migration = fork_state(state, arm=arm, binding=plan["binding"], config_text=config,
            run_id=yaml.safe_load(config)["run_id"], original_config_text=original_text,
            expected_cuda_rng_states=expected_cuda_rng_states)
        forks[arm] = validate_snapshot(own, origin=state, arm=arm, binding=plan["binding"],
            config_text=config, original_config_text=original_text, completed_event=4,
            expected_recipe=own["backend"]["optimizer"]["recipe"], expected_cuda_rng_states=expected_cuda_rng_states)
        for key in ("draft_state_dict", "rng", "events", "global_step", "presentations"):
            if differences(state[key], own[key]):
                raise ValueError("Metadata-only fork changed scientific state: " + key)
        forks[arm]["migration"] = migration
        del own
    validate_origin(state, original_text, expected_cuda_rng_states=expected_cuda_rng_states)
    # Each historical control is checked with its original PR51 contract. Do
    # not rewrite its config, event history or source binding into the new run.
    from experiments.mathqa_fixed_lr_transfer.state import validate_snapshot as validate_control
    controls = {}
    for label, saved in plan['retained_endpoints'].items():
        end = saved['endpoint']
        for path, expected_id in ((end['state'], end['state_identity']),
                (Path(end['export'])/'model.safetensors', end['weights_identity'])):
            if identity(path) != expected_id:
                raise ValueError('Retained control identity changed: '+str(path))
        current = torch.load(end['state'], map_location='cpu', weights_only=False, mmap=True)
        controls[label] = validate_control(current, origin=state, arm=label[0],
            binding=saved['source_binding'], config_text=current['config_text'],
            original_config_text=original_text, completed_event=int(label[1:]),
            expected_recipe=current['backend']['optimizer']['recipe'],
            expected_cuda_rng_states=expected_cuda_rng_states)
        _export_equal(current, end['export'])
        event = current['events'][-1]
        pool = plan['common_e5'] if label.endswith('5') else plan['retained_pools'][label[0]+'_collect6']
        if [row['path'] for row in event['features']] != pool['feature_paths']:
            raise ValueError('Retained control trained on a different feedback pool')
        controls[label]['producer_pool_verified'] = True
        del current
        gc.collect()
    del state
    gc.collect()
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU recovery validation unexpectedly initialized CUDA")
    return dict(status="passed", origin=receipt, forks=forks, export_exact=True,
                ancestry_records=512, controls=controls, mmap=True, map_location="cpu", optimizer_steps=0)


def authenticate_retained_assets(plan, *, stage="historical"):
    """Require original content evidence before reusing prepared teacher tensors.

    A new hash of a current file pins future use but cannot supply a missing
    historical reference. The preparation step preserves that distinction.
    """
    if stage in ('prerequisite','main'):
        from .replay_gate import pinned,validate_prerequisite,validate_main_receipt,require_current_features
        authentication=pinned(plan['asset_authentication'])
        admission=validate_prerequisite(plan,authentication)
        if stage=='main':
            admission=validate_main_receipt(plan,json.loads(Path(plan['prerequisite']['qualified_receipt']).read_text()))
        current=require_current_features(plan)
    else:
        authentication=read_authentication(plan)
        admission=current=None
    checked = {}
    for path, expected in authentication['files'].items():
        actual = identity(path)
        if actual != expected:
            raise ValueError('Reused artifact differs from its authenticated identity: '+path)
        checked[path] = actual
    return dict(status='passed', historical_feature_identity_verified=stage=='historical', files=checked, admission=admission, current_features=current)


def read_authentication(plan):
    artifact = plan['asset_authentication']
    if identity(artifact['path']) != artifact['identity']:
        raise ValueError('Retained-artifact authentication receipt changed')
    receipt = json.loads(Path(artifact['path']).read_text())
    if (receipt.get('status') != 'passed' or
            receipt.get('historical_feature_identity_verified') is not True):
        raise ValueError('Historical teacher-feature content identity is not established')
    return receipt


def inspect(plan_path, *, stage="prerequisite"):
    plan = json.loads(Path(plan_path).read_text())
    root = Path(plan["root"])
    started = time.perf_counter()
    result = dict(stage=stage,status="running", execution_commit=plan["execution_commit"], model_work=0, inputs={})
    try:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise ValueError("CPU preflight requires CUDA_VISIBLE_DEVICES empty")
        seal = json.loads((root / "sealed/SEAL.json").read_text())
        if (plan["binding"] != plan_binding(plan) or seal != dict(binding=plan["binding"],
                execution_commit=plan["execution_commit"], input_identities=plan["input_identities"])):
            raise ValueError("Scientific plan or input seal changed")
        if (plan["limits"] != LIMITS or plan["new_reference_generations"] != 258
                or plan["native_control_reuse"] is not True or plan["reference_reuse"]):
            raise ValueError("Changed bounded work or unqualified historical reuse")
        result["input_readiness"] = reference_readiness(plan)
        for path, expected in plan["input_identities"].items():
            actual = identity(path)
            if actual != expected:
                raise ValueError("Immutable input changed: " + path)
            result["inputs"][path] = actual
        result["source"] = verify_source(plan)
        result["runtime"] = runtime_inventory(plan)
        result["retained_assets"] = authenticate_retained_assets(plan,stage=stage)
        result["states"] = inspect_states(plan)
        inputs = load_inputs(plan)
        geometry = json.loads((Path(plan["origin_export"]) / "config.json").read_text())
        measured = storage_plan([len(r["prompt_ids"]) for r in inputs['event6_inputs']['requests']],
            weight_bytes=Path(plan["origin_export"], "model.safetensors").stat().st_size,
            recovery_bytes=Path(plan["origin_state"]).stat().st_size, hidden_size=geometry["hidden_size"],
            captured_layers=len(geometry["dflash_config"]["target_layer_ids"]))
        if measured != plan["storage"] or measured["allocation_cap_bytes"] != ALLOCATION_BYTES or measured["reserve_bytes"] != RESERVE_BYTES:
            raise ValueError("Measured storage geometry differs from admission")
        free = shutil.disk_usage(root).free
        result["storage"] = dict(measured, actual_free_bytes=free)
        if measured["status"] != "passed" or free < measured["required_initial_free_bytes"]:
            raise RuntimeError("Full peak plus server free-space reserve does not fit")
        if (type(plan["compiler_baseline_allocated_bytes"]) is not int or plan["compiler_baseline_allocated_bytes"] < 0
                or plan["compiler_growth_reserved_bytes"] != measured["compiler_bytes"]):
            raise ValueError("Compiler baseline or reserve differs")
        result["charged_allocation"] = validate_allocation_roots(plan)
        result.update(status="passed", cuda_initialized=False,
            parallel_qualification_admitted=result["runtime"]["host"]["parallel_qualification_admitted"])
    except BaseException as error:
        result.update(status="failed", error=repr(error), error_path=getattr(error, "filename", None))
        raise
    finally:
        result["wall_s"] = time.perf_counter() - started
        write_json(root / ("CPU_PREFLIGHT_"+stage.upper()+".json"), result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--stage",choices=("prerequisite","main"),required=True)
    args = parser.parse_args()
    print(json.dumps(inspect(args.plan,stage=args.stage)))


if __name__ == "__main__":
    main()
