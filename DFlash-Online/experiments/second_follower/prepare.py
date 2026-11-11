"""Seal existing fork, inputs and analysis before candidate outcomes; no model work."""

import argparse
from copy import deepcopy
from pathlib import Path
import shutil
import subprocess
import yaml

from online_feedback.storage import write_json
from experiments.ok_coding.storage import identity
from experiments.survival_head.data import read, digest
from experiments.successor_shuffle.campaign import shared_inputs, authenticate_true
from experiments.successor_shuffle.seal import checked_file
from .contract import SCHEMA, ANALYSIS, SERVING, TRUE_EXPORT_SHA256


def source_manifest(code):
    repo = code.parent
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repo, text=True
    ).strip()
    if dirty:
        raise ValueError("Execution checkout must be clean: " + dirty)
    paths = (
        subprocess.check_output(["git", "ls-files", "-z", "DFlash-Online"], cwd=repo)
        .decode()
        .split("\0")
    )
    files = {}
    symlinks = {}
    for name in paths:
        if not name:
            continue
        path = repo / name
        relative = str(path.relative_to(code))
        if path.is_symlink():
            symlinks[relative] = str(path.readlink())
        elif path.is_file():
            files[relative] = identity(path)
    return dict(
        status="verified",
        commit=commit,
        dirty=False,
        source_path=str(code),
        files=files,
        symlinks=symlinks,
    )


def seal(root, prior, matched_inputs, previous_exposure_attempt=None):
    from .evaluation import evaluation_schedule

    root = Path(root).resolve()
    prior = Path(prior).resolve()
    code = Path(__file__).resolve().parents[2]
    if root.exists():
        raise FileExistsError("Preserve existing run")
    old = read(prior / "sealed/PLAN.json")
    oldseal = read(prior / "sealed/SEAL.json")
    checked_file(
        prior / "sealed/PLAN.json", oldseal["files"][str(prior / "sealed/PLAN.json")]
    )
    authenticate_true(old)
    pool, baseline = shared_inputs(old)
    for key in ("origin_state", "origin_weights"):
        checked_file(old[key], old[key + "_identity"])
    panel = read(old["panel_manifest"])
    refs = read(old["references"])
    if len(panel["requests"]) != 256 or set(refs) != {
        r["request_id"] for r in panel["requests"]
    }:
        raise ValueError("Require exact completed PR39 panel and canonical inventory")
    # Original canonical file is directly bound by completed raw audit/report.
    oldmanifest = read(prior / "final_report/INPUT_MANIFEST.json")["files"]
    for path in (old["panel_manifest"], old["references"]):
        checked_file(path, oldmanifest[path])
    matched = read(matched_inputs)
    from .matched_start import verify_panel_disjoint

    disjoint = verify_panel_disjoint(matched, panel["requests"])
    root.mkdir(parents=True)
    sealed = root / "sealed"
    sealed.mkdir()
    source = source_manifest(code)
    write_json(root / "SOURCE.json", source)
    for name in (
        "ORIGINAL_CONFIG.yaml",
        "MODEL_PIN.json",
        "RUNTIME_PIN.json",
        "TRAINING_PIN.json",
        "BOUND_RUNTIME_PIN.json",
    ):
        src = prior / "sealed" / name
        checked_file(src, oldseal["files"][str(src)])
        shutil.copyfile(src, sealed / name)
    for name, src in [
        ("PANEL.json", old["panel_manifest"]),
        ("CANONICAL.json", old["references"]),
    ]:
        shutil.copyfile(src, sealed / name)
    write_json(sealed / "MATCHED_INPUTS.json", matched)
    write_json(sealed / "PANEL_DISJOINT.json", disjoint)
    checked_file(old["configs"]["TRUE"], oldseal["files"][old["configs"]["TRUE"]])
    configuration = yaml.safe_load(Path(old["configs"]["TRUE"]).read_text())
    configs = {}
    for arm in ("NATIVE", "SECOND"):
        config = deepcopy(configuration)
        config["run_id"] = "second_follower_" + arm
        config["output_dir"] = str(root / "training" / arm)
        path = sealed / f"config_{arm}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        configs[arm] = str(path)
    plan = {
        k: deepcopy(old[k])
        for k in (
            "target",
            "draft",
            "training_python",
            "worker_python",
            "origin_state",
            "origin_export",
            "origin_weights",
            "origin_state_identity",
            "origin_weights_identity",
            "pool",
            "pool_identity",
            "original_schedule",
            "original_schedule_identity",
            "baseline_result",
            "baseline_result_identity",
        )
    }
    control = deepcopy(old["checkpoints"]["TRUE"])
    if control["weights_identity"]["sha256"] != TRUE_EXPORT_SHA256:
        raise ValueError("Wrong step640 control")
    plan.update(
        schema=SCHEMA,
        root=str(root),
        dataset="gsm8k",
        event=5,
        parent_plan=str(prior / "sealed/PLAN.json"),
        parent_plan_identity=identity(prior / "sealed/PLAN.json"),
        original_config=str(sealed / "ORIGINAL_CONFIG.yaml"),
        configs=configs,
        panel_manifest=str(sealed / "PANEL.json"),
        references=str(sealed / "CANONICAL.json"),
        matched_start=str(sealed / "MATCHED_INPUTS.json"),
        matched_start_identity=identity(sealed / "MATCHED_INPUTS.json"),
        checkpoints={"NATIVE": control},
        serving=SERVING,
        analysis=ANALYSIS,
        evaluation_schedule=evaluation_schedule(panel["requests"], sealed_n=256),
        panel_size=256,
        source_manifest=str(root / "SOURCE.json"),
        execution_source={k: source[k] for k in ("commit", "source_path", "dirty")},
        training_gpu=0,
        serving_gpu=0,
        training_gpu_identity=old["serving_gpu_identity"],
        serving_gpu_identity=old["serving_gpu_identity"],
        storage=dict(
            maximum_additional_bytes=80 * 2**30,
            reserve_bytes=8 * 2**30,
            basis="3 disposable gradient/Adam endpoints, one final recovery/export, atomic overlap, compiler and raw receipts; no deletion credit",
        ),
        budget=dict(
            main_optimizer_steps=128,
            main_presentations=256,
            timed_completions=1024,
            matched_starts=128,
            matched_child_blocks=256,
            gpu_seconds=4500,
            early_exposure_forwards=16,
            disposable_optimizer_steps=3,
            deferred_native_twin_optimizer_steps_max=1,
            deferred_native_twin_forwards_max=2,
            deferred_native_twin_backwards_max=2,
            disposable_presentations=6,
            discarded_startup_forwards=10,
            discarded_startup_backwards=10,
        ),
        automatic_retry=False,
    )
    plan["previous_attempts"] = []
    if previous_exposure_attempt is not None:
        previous = Path(previous_exposure_attempt).resolve()
        previous_plan = read(previous / "sealed/PLAN.json")
        status = read(previous / "STATUS.json")
        exposure = read(previous / "qualification/exposure/RESULT.json")
        check = read(previous / "qualification/exposure/PARENT_WEIGHTS.json")
        if (
            status["status"] != "failed"
            or status["phase"] != "qualification_exposure"
            or exposure["response_forwards"] != 16
            or exposure["optimizer_steps"] != 0
            or exposure["response_backwards"] != 0
            or check["compared_tensors"] != 0
            or [r["reason"] for r in check["metadata_differences"]] != ["type"]
            or previous_plan.get("previous_attempts")
            or previous_plan["origin_state_identity"] != plan["origin_state_identity"]
            or identity(previous_plan["panel_manifest"])
            != identity(plan["panel_manifest"])
        ):
            raise ValueError(
                "Previous attempt is not the preserved mapping-checker failure"
            )
        names = [
            "sealed/PLAN.json",
            "STATUS.json",
            "RESOURCES.json",
            "qualification/exposure/RESULT.json",
            "qualification/exposure/PARENT_WEIGHTS.json",
            "qualification/exposure/startup/RESULT.json",
        ]
        plan["previous_attempts"] = [
            dict(
                root=str(previous),
                execution_commit=previous_plan["execution_source"]["commit"],
                gpu_seconds=status["gpu_clock_elapsed_s"],
                response_forwards=18,
                response_backwards=2,
                optimizer_steps=0,
                source_exposure_forwards=16,
                discarded_startup_forwards=2,
                reason="OrderedDict-versus-dict type check stopped before tensor comparisons",
                files={str(previous / n): identity(previous / n) for n in names},
            )
        ]
    plan["prior_gpu_seconds"] = sum(r["gpu_seconds"] for r in plan["previous_attempts"])
    if not 0 <= plan["prior_gpu_seconds"] < 4500:
        raise ValueError("No remaining GPU budget")
    plan["binding"] = digest(plan)
    write_json(sealed / "PLAN.json", plan)
    files = {str(p): identity(p) for p in sealed.iterdir() if p.is_file()}
    files[str(root / "SOURCE.json")] = identity(root / "SOURCE.json")
    write_json(sealed / "SEAL.json", dict(binding=plan["binding"], files=files))
    return plan


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("--prior", required=True)
    p.add_argument("--matched-inputs", required=True)
    p.add_argument("--previous-exposure-attempt")
    a = p.parse_args()
    seal(a.root, a.prior, a.matched_inputs, a.previous_exposure_attempt)
