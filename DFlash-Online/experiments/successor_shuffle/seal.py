"""Bind actual retained fork, feedback, panel and source before new outcomes."""

import argparse
from copy import deepcopy
from pathlib import Path
import shutil
import yaml

from online_feedback.storage import write_json
from experiments.ok_coding.storage import identity
from experiments.survival_head.data import read, digest, verify_shared_packets
from experiments.survival_head.seal import authenticate_source
from experiments.kplus_abt.seal import validate_config_text
from experiments.kplus_abt.campaign import schedule_bindings
from experiments.survival_head.contract import SERVING
from .evaluation import evaluation_schedule

TRUE_HASH = "9dcb0d637eabef6bd4e34c552da0448faa9510e6bcaf5c753a6bd1b43674b795"


def checked_file(path, expected):
    observed = identity(path)
    if observed != expected:
        raise ValueError("Retained artifact changed: " + str(path))
    return observed


def seal(root, prepared, pilot, confirmation, source_manifest):
    from .data import verify_panel
    from .report import ANALYSIS

    root, prepared, pilot, confirmation = map(
        lambda p: Path(p).resolve(), (root, prepared, pilot, confirmation)
    )
    source = read(source_manifest)
    code = Path(__file__).resolve().parents[2]
    authenticate_source(source, code)
    required = {
        str(p.relative_to(code))
        for p in (code / "experiments/successor_shuffle").glob("*.py")
    }
    required.add("online_feedback/successor_shuffle.py")
    if not required.issubset(source["files"]):
        raise ValueError("Execution source omits successor modules")
    old = read(pilot / "sealed/PLAN.json")
    old_seal = read(pilot / "sealed/SEAL.json")
    completed = read(pilot / "final_report/INPUT_MANIFEST.json")
    if (
        old_seal["binding"] != old["binding"]
        or old_seal["execution_source"] != old["execution_source"]
    ):
        raise ValueError("Original PR37 seal differs from authenticated plan")
    if (pilot / "EXIT_CODE").read_text().strip() != "0" or (
        confirmation / "EXIT_CODE"
    ).read_text().strip() != "0":
        raise ValueError("Require completed PR37 and PR38")
    checked_file(
        pilot / "sealed/PLAN.json", completed["files"][str(pilot / "sealed/PLAN.json")]
    )
    for name in (
        "ORIGINAL_CONFIG.yaml",
        "MODEL_PIN.json",
        "RUNTIME_PIN.json",
        "TRAINING_PIN.json",
    ):
        checked_file(
            pilot / "sealed" / name, old_seal["files"][str(pilot / "sealed" / name)]
        )
    checked_file(old["origin_state"], old["origin_state_identity"])
    checked_file(old["origin_weights"], old["origin_weights_identity"])
    pool_path = pilot / "feedback/POOL.json"
    pool = read(pool_path)
    verify_shared_packets(pool)
    schedule_path = pilot / "feedback/PRESENTATION_SCHEDULE.json"
    schedule = read(schedule_path)
    result_path = pilot / "campaign_001/gsm8k/event_005/KPLUS/RESULT.json"
    true = read(result_path)
    checked_file(result_path, completed["files"][str(result_path)])
    # The original report binds the pool through qualification admission.
    admission_path = pilot / "qualification/ADMISSION.json"
    checked_file(admission_path, completed["files"][str(admission_path)])
    admission = read(admission_path)
    case = next(row for row in admission["cases"] if row["case"] == "KPLUS")
    checked_file(case["result"], case["result_identity"])
    checked_file(pool_path, read(case["result"])["inputs"]["pool_identity"])
    from experiments.survival_head.prepare_feedback import rebound_schedule

    checked_file(old["original_schedule"], old["original_schedule_identity"])
    expected_schedule = rebound_schedule(
        read(old["original_schedule"]),
        [read(row["canonical"]["path"]) for row in pool["records"]],
        [row["feature"]["path"] for row in pool["records"]],
    )
    if schedule != expected_schedule:
        raise ValueError(
            "PR37 schedule differs from its authenticated original and packet paths"
        )
    if (
        true["status"] != "completed"
        or true["method"] != "KPLUS"
        or true["start_step"] != 512
        or true["end_step"] != 640
        or true["optimizer_steps"] != 128
        or true["boundary"]["status"] != "passed"
        or not true["boundary"]["export_exact"]
        or schedule_bindings(true) != schedule_bindings(schedule)
    ):
        raise ValueError("TRUE origin, schedule or completed boundary differs")
    export = Path(true["checkpoint_path"])
    weights = identity(export / "model.safetensors")
    pin38 = read(confirmation / "sealed/PLAN.json")["checkpoints"]["KPLUS"]
    if (
        weights != true["weights_identity"]
        or weights != pin38["weights_identity"]
        or weights["sha256"] != TRUE_HASH
    ):
        raise ValueError("TRUE does not match PR37 result and PR38 frozen pin")
    panel = read(prepared / "PANEL.json")
    verify_panel(panel)
    n = len(panel["requests"])
    if n > 256 or n < 8 or n % 8:
        raise ValueError("Panel must contain complete eight-request blocks")
    refs = read(prepared / "REUSED_CANONICAL.json")
    reuse = read(prepared / "CANONICAL_REUSE_AUDIT.json")
    root.mkdir(parents=True, exist_ok=True)
    sealed = root / "sealed"
    sealed.mkdir(exist_ok=False)
    for path in prepared.glob("*.json"):
        shutil.copy2(path, sealed / path.name)
    for name in (
        "ORIGINAL_CONFIG.yaml",
        "MODEL_PIN.json",
        "RUNTIME_PIN.json",
        "TRAINING_PIN.json",
    ):
        shutil.copy2(pilot / "sealed" / name, sealed / name)
    # PR38 expanded the pin to include all files used by reused canonicals.
    prior_runtime = confirmation / "sealed/BOUND_RUNTIME_PIN.json"
    confirmation_seal = read(confirmation / "sealed/SEAL.json")
    checked_file(prior_runtime, confirmation_seal["files"][str(prior_runtime)])
    shutil.copy2(prior_runtime, sealed / "BOUND_RUNTIME_PIN.json")
    pinned = reuse["target_runtime_binding"]
    model = read(sealed / "MODEL_PIN.json")
    runtime = read(sealed / "BOUND_RUNTIME_PIN.json")
    if (
        pinned["target_files"] != model["target"]["files"]
        or any(
            pinned[k] != runtime[k] for k in ("python_version", "torch", "transformers")
        )
        or any(
            runtime["native_runtime_sha256"].get(k) != v
            for k, v in pinned["native_runtime_sha256"].items()
        )
    ):
        raise ValueError("Canonical target/runtime differs from qualified serving")
    missing = [
        r["request_id"] for r in panel["requests"] if r["request_id"] not in refs
    ]
    from experiments.adaflash_c1.storage import RAW_BYTES_PER_GENERATION

    # Two final states and exports, qualification tensors and one atomic save
    # overlap. No credit for future deletion or shared source-data retirement.
    raw_bytes = (6 * n + len(missing) + 64) * RAW_BYTES_PER_GENERATION
    storage = dict(
        reserve_bytes=8 * 2**30,
        maximum_additional_bytes=80 * 2**30 + raw_bytes,
        details=dict(
            training_states_qualification_and_overlap=64 * 2**30,
            source_cache_and_staging=8 * 2**30,
            report_and_working_allowance=8 * 2**30,
            raw_records_native_sidecars=raw_bytes,
        ),
        no_future_deletion_credit=True,
    )
    plan = dict(
        schema="kplus_successor_shuffle_v1",
        root=str(root),
        panel_size=n,
        execution_source={
            k: source[k] for k in ("commit", "dirty", "repository", "source_path")
        },
        source_manifest=str(Path(source_manifest).resolve()),
        automatic_retry=False,
        pilot_root=str(pilot),
        confirmation_root=str(confirmation),
        origin_state=old["origin_state"],
        origin_export=old["origin_export"],
        origin_weights=old["origin_weights"],
        origin_state_identity=old["origin_state_identity"],
        origin_weights_identity=old["origin_weights_identity"],
        original_config=str(sealed / "ORIGINAL_CONFIG.yaml"),
        pool=str(pool_path),
        pool_identity=identity(pool_path),
        original_schedule=str(schedule_path),
        original_schedule_identity=identity(schedule_path),
        baseline_result=str(result_path),
        baseline_result_identity=identity(result_path),
        panel_manifest=str(sealed / "PANEL.json"),
        matched_panel=str(sealed / "MATCHED_START_PANEL.json"),
        reused_references=str(sealed / "REUSED_CANONICAL.json"),
        references=str(root / "references/CANONICAL.json"),
        canonical_binding=reuse["binding_digest"],
        missing_reference_ids=missing,
        target=old["target"],
        draft=old["draft"],
        worker_python=old["worker_python"],
        training_python=old["training_python"],
        controller_python=old["training_python"],
        training_gpu=1,
        serving_gpu=0,
        training_gpu_identity=old["training_gpu_identity"],
        serving_gpu_identity=old["serving_gpu_identity"],
        serving=deepcopy(SERVING),
        analysis=deepcopy(ANALYSIS),
        storage=storage,
        true_reuse=dict(
            enabled=True,
            checkpoint_path=str(export),
            weights_identity=weights,
            result_path=str(result_path),
            result_identity=identity(result_path),
            authentication_path=str(sealed / "TRUE_AUTHENTICATION.json"),
        ),
        checkpoints={
            "TRUE": dict(
                checkpoint_path=str(export),
                weights_identity=weights,
                export_files={
                    p.name: identity(p) for p in export.iterdir() if p.is_file()
                },
            )
        },
        budget=dict(
            optimizer_steps=256,
            training_presentations=512,
            feedback_generations=0,
            timed_completions=6 * n,
            reference_generations=len(missing),
            parent_draft_blocks=64,
            qualification_optimizer_steps=3,
            qualification_presentations=6,
            extra_generation_warmups=0,
        ),
        evaluation_schedule=evaluation_schedule(panel["requests"], sealed_n=n),
        configs={},
    )
    original = (sealed / "ORIGINAL_CONFIG.yaml").read_text()
    for arm in ("K", "TRUE", "SHUFFLED"):
        config = yaml.safe_load(original)
        config.update(
            run_id="successor_shuffle_" + arm,
            output_dir=str(root / "campaign_001/gsm8k/event_005" / arm),
        )
        text = yaml.safe_dump(config, sort_keys=False)
        validate_config_text(text, original)
        path = sealed / f"config_{arm}.yaml"
        path.write_text(text)
        plan["configs"][arm] = str(path)
    from .contract import budgets

    plan["budgets"] = budgets(
        n, reuse_true=True, missing_references=len(missing), parent_blocks=min(64, n)
    )
    plan["binding"] = digest(plan)
    write_json(
        sealed / "TRUE_AUTHENTICATION.json",
        dict(
            status="passed",
            binding=plan["binding"],
            execution_commit=source["commit"],
            origin=True,
            packets=True,
            schedule=True,
            optimizer=True,
            runtime=True,
            step640=True,
            export=True,
            checks=dict(
                origin_state=old["origin_state_identity"],
                origin_weights=old["origin_weights_identity"],
                pool=identity(pool_path),
                schedule=identity(schedule_path),
                result=identity(result_path),
                inherited_config=identity(sealed / "ORIGINAL_CONFIG.yaml"),
                boundary=true["boundary"],
                pr38_weights=pin38["weights_identity"],
                weights=weights,
            ),
        ),
    )
    write_json(sealed / "PLAN.json", plan)
    files = {str(p): identity(p) for p in sealed.iterdir() if p.is_file()}
    files[str(Path(source_manifest).resolve())] = identity(source_manifest)
    write_json(sealed / "SEAL.json", dict(binding=plan["binding"], files=files))
    return plan


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("root", "prepared", "pilot", "confirmation", "source-manifest"):
        p.add_argument("--" + key, required=True)
    a = p.parse_args()
    print(
        seal(a.root, a.prepared, a.pilot, a.confirmation, a.source_manifest)["budget"]
    )
