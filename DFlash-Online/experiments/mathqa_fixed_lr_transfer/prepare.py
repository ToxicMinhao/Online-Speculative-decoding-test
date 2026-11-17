"""Seal the historical MathQA origin, new inputs and fixed N/S continuation.

Preparation copies compact metadata only. Models and recovery tensors stay at
their authenticated server paths; the 258 correctness references are new work.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import tarfile

from experiments.bounded_prefix_allocation.prepare import identity, _digest, _write
from .data import validate_manifests
from .evaluation import evaluation_schedule
from .runtime import ALLOCATION_BYTES, RESERVE_BYTES, LIMITS

TASK_NAME = "dflash_mathqa_fixed_lr_transfer_20261116"
DEVICES = dict(A="GPU-9986735a-3e68-d1bd-5998-9634ef24341e",
               B="GPU-c6943f33-bcd0-a724-0b35-f77af6b0a3f1")
INPUTS = dict(data_panel="PANEL.json", warmups="WARMUPS.json", adaptation="TRAINING.json",
    event5_inputs="EVENT5_INPUTS.json", event6_inputs="EVENT6_INPUTS.json",
    evaluation_schedule="EVALUATION_SCHEDULE.json", data_seal="DATA_SEAL.json",
    source_provenance="SOURCE_PROVENANCE.json")
MUTABLE_PLAN_FIELDS = {"binding", "compiler_baseline_allocated_bytes", "estimated_seconds"}


def plan_binding(plan):
    """Only resource observations may change after scientific inputs are sealed."""
    return _digest({k: v for k, v in plan.items() if k not in MUTABLE_PLAN_FIELDS})


def storage_plan(event5_lengths, event6_lengths, *, weight_bytes, recovery_bytes,
                 hidden_size, captured_layers, retire_verified_duplicates=True):
    """Price four retained boundaries, two disposable states and all three pools.

    Only exact, already-prepared capture duplicates may be retired. The bound
    includes both workers' temporary capture/terminal/prepared overlap and one
    serialized atomic checkpoint publication; it assumes no old-file deletion.
    """
    first, second = list(event5_lengths), list(event6_lengths)
    if (len(first) != 128 or len(second) != 128
            or any(type(n) is not int or not 0 < n <= 2048 for n in first + second)
            or any(type(n) is not int or n <= 0 for n in
                   (weight_bytes, recovery_bytes, hidden_size, captured_layers))):
        raise ValueError("Require measured tensor geometry and all 256 continuation prompts")
    row_bytes = hidden_size * (captured_layers + 1) * 2 + 12
    features = sum(n + 2048 for n in first + second + second) * row_bytes
    # Allow growth of compact event/RNG metadata at the two new boundaries.
    state_bytes = recovery_bytes + 16 * 2**20
    boundary_bytes = state_bytes + weight_bytes
    parts = dict(prepared_feature_bytes=features,
        retained_duplicate_capture_bytes=0 if retire_verified_duplicates else features,
        temporary_feature_bytes=4 * (max(first + second) + 2049) * row_bytes,
        checkpoint_bytes=4 * boundary_bytes,
        qualification_payload_bytes=2 * boundary_bytes,
        atomic_publication_bytes=boundary_bytes,
        compiler_bytes=4 * 2**30, compact_bytes=2_000_000_000,
        source_and_workspace_bytes=2 * 2**30)
    parts["serialization_margin_bytes"] = math.ceil(sum(parts.values()) * .05)
    peak = sum(parts.values())
    return dict(status="passed" if peak <= ALLOCATION_BYTES else "blocked",
        peak_new_bytes=peak, allocation_cap_bytes=ALLOCATION_BYTES,
        reserve_bytes=RESERVE_BYTES, required_initial_free_bytes=peak + RESERVE_BYTES,
        row_bytes=row_bytes, publication_concurrency=1, retained_boundaries=["N5", "N6", "S5", "S6"],
        feature_pools=3, common_event5_copies=1, existing_dependencies_copied=False,
        verified_duplicate_retirement=retire_verified_duplicates,
        retained_unique_prepared_features=True, **parts)


def load_inputs(plan, *, package_root=None):
    """Use the new panel schema before any collection/reference worker starts."""
    values = {}
    for key in INPUTS:
        remote = Path(plan[key])
        path = remote if package_root is None else Path(package_root) / remote.relative_to(plan["root"])
        if identity(path) != plan["input_identities"].get(str(remote)):
            raise ValueError("Unbound or changed packaged input: " + key)
        values[key] = json.loads(path.read_text())
    validate_manifests(values["data_panel"], values["warmups"], values["adaptation"])
    provenance = values["source_provenance"]
    if (provenance["status"] != "passed" or provenance["ancestry_count"] != 768
            or provenance["selected_ancestry_content_overlap"] != 0):
        raise ValueError("Complete origin/continuation exposure provenance is required")
    seal = values["data_seal"]
    if (seal["status"] != "passed" or seal["first_model_work_before_seal"] is not False
            or not seal["input_only_eligibility"]):
        raise ValueError("Exposure audit or input-only data seal is incomplete")
    for name, expected in seal["files"].items():
        if plan["input_identities"].get(plan["root"] + "/sealed/" + name) != expected:
            raise ValueError("Original data seal identity differs: " + name)
    train = values["adaptation"]["train_requests"]
    for event, offset in ((5, 0), (6, 128)):
        part = values[f"event{event}_inputs"]
        if (part["event"] != event or part["start_ordinal"] != 512 + offset
                or part["stop_ordinal"] != 640 + offset or part["requests"] != train[offset:offset+128]):
            raise ValueError("Event slice no longer matches the historical ordered stream")
    schedule = values["evaluation_schedule"]
    if (schedule["activations"] != evaluation_schedule(values["data_panel"]["requests"])
            or schedule["scored_executions"] != 1536 or schedule["warmup_executions"] != 48):
        raise ValueError("Balanced final evaluation schedule changed")
    return values


def input_readiness(plan, *, package_root=None):
    load_inputs(plan, package_root=package_root)
    return dict(status="passed", scored_inputs=256, warmup_inputs=2, adaptation_inputs=256,
        new_canonical_generations=258, canonical_references_not_yet_generated=True,
        before_worker_construction=True, model_work=0)


def reference_readiness(plan, *, package_root=None):
    """Staging gate: references must be generated later by the fixed origin."""
    for key in ("references", "warmup_references"):
        path = Path(plan[key])
        if package_root is not None:
            path = Path(package_root) / path.relative_to(plan["root"])
        if path.exists():
            raise ValueError("New-panel canonical generation has already been attempted")
    return input_readiness(plan, package_root=package_root)


def build_archive(plan, package_root, archive_path):
    checked = input_readiness(plan, package_root=package_root)
    if json.loads((Path(package_root) / "sealed/PLAN.json").read_text()) != plan:
        raise ValueError("Archive plan differs from the inspected plan")
    with tarfile.open(archive_path, "x:gz") as archive:
        for path in sorted(Path(package_root).rglob("*")):
            if path.is_symlink():
                raise ValueError("Input package must contain regular files only")
            if path.is_file():
                archive.add(path, arcname=str(path.relative_to(package_root)), recursive=False)
    return dict(status="passed", archive_identity=identity(archive_path), readiness=checked)


def unpack_archive(archive_path, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive_path, "r:gz") as archive:
        if any(not (m.isfile() or m.isdir()) for m in archive.getmembers()):
            raise ValueError("Input package contains a link or special file")
        archive.extractall(destination, filter="data")
    plan = json.loads((destination / "sealed/PLAN.json").read_text())
    return dict(plan=plan, readiness=input_readiness(plan, package_root=destination))


def build(workspace, destination, *, remote_root, source_root, execution_commit):
    import yaml
    from experiments.persistent_prefix_withholding.launch import QUALIFIED_COMPILERS
    from .state import arm_lr, validate_config_text
    from .preflight import HOST_POLICY

    workspace, destination = Path(workspace), Path(destination)
    task = workspace / "research" / TASK_NAME
    historical = workspace / "research/dflash_kplus_abt_20261110/execution_002_preparation/sealed_package/sealed"
    old = json.loads((historical / "PLAN.json").read_text())
    old_seal = json.loads((historical / "SEAL.json").read_text())
    for name in ("PLAN.json", "ORIGINAL_CONFIG.yaml", "MODEL_PIN.json", "RUNTIME_PIN.json", "TRAINING_PIN.json", "mathqa/TRAINING.json"):
        remote = str(Path(old["root"]) / "sealed" / name)
        if identity(historical / name) != old_seal["files"][remote]:
            raise ValueError("Historical sealed metadata changed: " + name)
    inherited = workspace / "research/dflash_functional_step_geometry_20261108/pilot_r1_001/sealed_package/sealed"
    inherited_plan = json.loads((inherited / "PLAN.json").read_text())
    inherited_seal = json.loads((inherited / "SEAL.json").read_text())
    inherited_files = {}
    for name in ("PLAN.json", "ORIGINAL_CONFIG.yaml", "MODEL_PIN.json", "RUNTIME_PIN.json", "TRAINING_PIN.json", "mathqa/TRAINING.json"):
        remote = str(Path(inherited_plan["root"]) / "sealed" / name)
        observed = identity(inherited / name)
        if observed != inherited_seal["files"][remote]:
            raise ValueError("Original inherited FSG seal changed: " + name)
        if name != "PLAN.json" and observed != identity(historical / name):
            raise ValueError("PR36 changed inherited MathQA input/model/runtime pins: " + name)
        inherited_files[remote] = observed
    for key in ("origin_state", "origin_export", "origin_state_identity", "origin_weights_identity", "origin_binding"):
        if inherited_plan["datasets"]["mathqa"][key] != old["datasets"]["mathqa"][key]:
            raise ValueError("PR36 inherited a different MathQA origin: " + key)
    data = task / "data_readiness/sealed"
    data_seal = json.loads((data / "DATA_SEAL.json").read_text())
    for name, expected in data_seal["files"].items():
        if identity(data / name) != expected:
            raise ValueError("Selected input changed after its data seal: " + name)
    destination.mkdir(parents=True, exist_ok=False)
    sealed = destination / "sealed"
    sealed.mkdir()
    _write(sealed / "INHERITED_FSG_PROVENANCE.json", dict(status="passed",
        seal_identity=identity(inherited / "SEAL.json"), files=inherited_files,
        exact_PR36_input_model_runtime_pins=True, exact_MathQA_KPLUS4_origin=True))
    origin = old["datasets"]["mathqa"]
    serving = json.loads((historical / "RUNTIME_PIN.json").read_text())
    training = json.loads((historical / "TRAINING_PIN.json").read_text())
    model = json.loads((historical / "MODEL_PIN.json").read_text())
    plan = dict(schema="mathqa_fixed_lr_transfer_v1", root=remote_root, source_root=source_root,
        execution_commit=execution_commit, source_manifest=remote_root + "/SOURCE.json",
        target=old["target"], draft=old["draft"], training_python=old["training_python"], worker_python=old["worker_python"],
        origin_state=origin["origin_state"], origin_export=origin["origin_export"],
        origin_state_identity=origin["origin_state_identity"], origin_weights_identity=origin["origin_weights_identity"],
        origin_binding=origin["origin_binding"], devices=dict(DEVICES), methods=["N", "S"], events=[5, 6],
        limits=dict(LIMITS), new_reference_generations=258, automatic_retry=False,
        configs={}, input_identities={}, qualified_compilers=dict(QUALIFIED_COMPILERS), host_resource_policy=dict(HOST_POLICY),
        runtime_contract=dict(training=training["environment"],
            serving=dict(python=serving["python_version"], torch=serving["torch"], transformers=serving["transformers"], cuda=serving["cuda"]),
            native_runtime_sha256=serving["native_runtime_sha256"]),
        compiler_root="/data/sources/minhao/.cache", compiler_growth_reserved_bytes=4 * 2**30,
        compiler_baseline_allocated_bytes=0, allocation_roots=[remote_root, source_root],
        reference_reuse=False, native_control_reuse=False,
        native_reuse_reason="Historical event-five state/export and exact prepared feedback pools were retired; fresh N/S fallback.",
        expected_work=dict(adam=522, response_forward=1056, response_backward=1056,
            training_initialization=6, feedback_generation=384, teacher_preparation=384,
            canonical_generation=258, qualification_generation=8, warmup_generation=48, scored_generation=1536),
        expected_training_initializations=dict(qualification=2, production=4),
        checkpoint_boundary_allowance_s=60., final_activation_allowance_s=180.,
        estimated_seconds=dict(common5=250., references=350., training_A=180., training_B=180.,
            serving_A=45., serving_B=45., N5=600., S5=600., N_collect6=250., S_collect6=250.,
            N6=600., S6=600., final=1800.),
        estimate_basis="Conservative preparation estimates; measured qualification replaces lower estimates before production.")
    def copy(source, name, key=None):
        shutil.copyfile(source, sealed / name)
        remote = remote_root + "/sealed/" + name
        if key:
            plan[key] = remote
        return remote
    for key, name in INPUTS.items():
        copy(data / name, name, key)
    for name, key in (("ORIGINAL_CONFIG.yaml", "origin_config"), ("MODEL_PIN.json", "model_pin"),
                      ("RUNTIME_PIN.json", "serving_pin"), ("TRAINING_PIN.json", "training_pin")):
        copy(historical / name, name, key)
    copy(historical / "PLAN.json", "HISTORICAL_PLAN.json", "historical_plan")
    copy(historical / "SEAL.json", "HISTORICAL_SEAL.json", "historical_seal")
    origin_root = workspace / "research/dflash_role_loss_pilot_20261107/result_review_001/retained"
    status_path = origin_root / "campaign_001/trajectories/mathqa/KPLUS/STATUS.json"
    config_path = origin_root / "campaign_001/trajectories/mathqa/KPLUS/trainer/events/event_004/export/config.json"
    copy(status_path, "ORIGIN_STATUS.json", "origin_status")
    copy(config_path, "ORIGIN_DRAFT_CONFIG.json", "origin_draft_config")
    ancestry = origin_root / "sealed/mathqa/TRAINING.json"
    copy(ancestry, "ORIGIN_TRAINING.json", "expected_ancestry")
    inventory_path = origin_root.parent / "REMOTE_INVENTORY.json"
    inventory = json.loads(inventory_path.read_text())
    entries = {row["path"]: {k: row[k] for k in ("bytes", "sha256")} for row in inventory["files"]}
    origin_metadata = {}
    for path in (status_path, config_path, ancestry):
        relative = str(path.relative_to(origin_root))
        if identity(path) != entries[relative]:
            raise ValueError("Retained origin metadata changed: " + relative)
        origin_metadata[relative] = entries[relative]
    _write(sealed / "ORIGIN_METADATA_PROVENANCE.json", dict(status="passed",
        inventory=str(inventory_path), inventory_identity=identity(inventory_path), files=origin_metadata))
    copy(task / "ORIGIN_REMOTE_CHECK.json", "ORIGIN_REMOTE_CHECK.json", "origin_remote_check")
    original_text = (historical / "ORIGINAL_CONFIG.yaml").read_text()
    for arm in ("N", "S"):
        config = yaml.safe_load(original_text)
        config.update(run_id="mathqa-fixed-lr-" + arm, output_dir=remote_root + "/training/" + arm)
        config["training"]["learning_rate"] = arm_lr(arm)
        text = yaml.safe_dump(config, sort_keys=False)
        validate_config_text(text, arm, original_text)
        (sealed / f"config_{arm}.yaml").write_text(text)
        plan["configs"][arm] = remote_root + f"/sealed/config_{arm}.yaml"
    plan["references"] = remote_root + "/references/CANONICALS.json"
    plan["warmup_references"] = remote_root + "/references/WARMUP_CANONICALS.json"
    for family in ("target", "draft"):
        for name, value in model[family]["files"].items():
            if name.endswith((".safetensors", ".json")) or name in ("merges.txt", "vocab.json"):
                plan["input_identities"][plan[family] + "/" + name] = value
    plan["input_identities"][plan["origin_state"]] = origin["origin_state_identity"]
    plan["input_identities"][plan["origin_export"] + "/model.safetensors"] = origin["origin_weights_identity"]
    plan["input_identities"][plan["origin_export"] + "/config.json"] = identity(config_path)
    geometry = json.loads(config_path.read_text())
    lengths = [[len(r["prompt_ids"]) for r in json.loads((sealed / f"EVENT{i}_INPUTS.json").read_text())["requests"]] for i in (5, 6)]
    plan["storage"] = storage_plan(*lengths, weight_bytes=origin["origin_weights_identity"]["bytes"],
        recovery_bytes=origin["origin_state_identity"]["bytes"], hidden_size=geometry["hidden_size"],
        captured_layers=len(geometry["dflash_config"]["target_layer_ids"]))
    for path in sorted(sealed.iterdir()):
        plan["input_identities"][remote_root + "/sealed/" + path.name] = identity(path)
    plan["binding"] = plan_binding(plan)
    _write(sealed / "PLAN.json", plan)
    _write(sealed / "SEAL.json", dict(binding=plan["binding"], execution_commit=execution_commit,
                                      input_identities=plan["input_identities"]))
    _write(destination / "INPUT_READINESS.json", input_readiness(plan, package_root=destination))
    _write(destination / "READINESS.json", dict(status="CPU_PREPARED_REMOTE_PREFLIGHT_PENDING",
        execution_commit=execution_commit, binding=plan["binding"], origin_event=4,
        native_control_reuse=False, scored_requests=256, warmup_requests=2, canonical_generations=258,
        storage=plan["storage"], model_work=0, gpu_qualified=False))
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace")
    parser.add_argument("destination")
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--execution-commit", required=True)
    args = parser.parse_args()
    plan = build(args.workspace, args.destination, remote_root=args.remote_root,
                 source_root=args.source_root, execution_commit=args.execution_commit)
    print(json.dumps(dict(status="prepared", binding=plan["binding"], storage=plan["storage"])))


if __name__ == "__main__":
    main()
