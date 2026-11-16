"""Prepare the bounded candidate from authenticated, completed PR49 artifacts.

Only compact metadata is copied. Existing model states and feature packets stay
at their retained paths; complete references are staged before any worker starts.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import shutil
import tarfile

ALLOCATION_BYTES = 80_000_000_000
FREE_RESERVE_BYTES = 400_000_000_000
REFERENCE_NAME = Path("references/CANONICALS.json")
METHODS = ("N", "C", "S")


def identity(path):
    digest = hashlib.sha256()
    size = 0
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return dict(bytes=size, sha256=digest.hexdigest())


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def authenticated_delivery_file(delivery, relative):
    """Authenticate one compact file against its retained delivery manifest."""
    delivery = Path(delivery)
    manifest_path = delivery / "DELIVERY_MANIFEST.json"
    receipt = delivery / "DELIVERY.json"
    if receipt.exists():
        expected = json.loads(receipt.read_text())["manifest_identity"]
        if identity(manifest_path) != expected:
            raise ValueError("Retained delivery manifest changed")
    manifest = json.loads(manifest_path.read_text())
    path = delivery / relative
    if identity(path) != manifest[relative]:
        raise ValueError("Retained delivery file changed: " + str(path))
    return path


def endpoint_from_result(result):
    if (result.get("status") != "completed" or result.get("event") != 8
            or result.get("arm") not in ("N", "S")
            or result["endpoint"]["global_step"] != 1024
            or result["endpoint"]["presentations"] != 2048
            or result["boundary"].get("status") != "passed"
            or result["boundary"].get("export_exact") is not True
            or result["boundary"]["live"].get("exact_resume") is not True):
        raise ValueError("Control must be a completed, checked event-eight endpoint")
    return deepcopy(result["endpoint"])


def storage_plan(prompt_lengths, *, weight_bytes, recovery_bytes, hidden_size,
                 captured_layers, qualification_payload_bytes=None):
    """One own-feedback pool, both C boundaries and one atomic publication.

    The bounded qualification workspace is reusable after its compact receipts
    are sealed. Its reservation is explicit, not sixteen full saved endpoints.
    """
    lengths = list(prompt_lengths)
    dimensions = (weight_bytes, recovery_bytes, hidden_size, captured_layers)
    if (len(lengths) != 128 or any(type(n) is not int or not 0 < n <= 2048 for n in lengths)
            or any(type(n) is not int or n <= 0 for n in dimensions)):
        raise ValueError("Require all 128 prompts and measured positive tensor sizes")
    row_bytes = hidden_size * (captured_layers + 1) * 2 + 12
    qualification = (recovery_bytes + weight_bytes if qualification_payload_bytes is None
                     else qualification_payload_bytes)
    if type(qualification) is not int or qualification < 0:
        raise ValueError("Qualification workspace reservation must be nonnegative")
    components = dict(
        prepared_feature_bytes=sum(n + 2048 for n in lengths) * row_bytes,
        temporary_feature_bytes=2 * (max(lengths) + 2049) * row_bytes,
        checkpoint_bytes=2 * (recovery_bytes + weight_bytes),
        atomic_publication_bytes=recovery_bytes + weight_bytes,
        qualification_payload_bytes=qualification,
        compiler_bytes=4 * 2**30,
        compact_bytes=2_000_000_000,
        source_and_workspace_bytes=2 * 2**30,
    )
    components["serialization_margin_bytes"] = math.ceil(sum(components.values()) * .05)
    peak = sum(components.values())
    return dict(status="passed" if peak <= ALLOCATION_BYTES else "blocked",
        peak_new_bytes=peak, allocation_cap_bytes=ALLOCATION_BYTES,
        reserve_bytes=FREE_RESERVE_BYTES, required_initial_free_bytes=peak + FREE_RESERVE_BYTES,
        row_bytes=row_bytes, publication_concurrency=1, retained_candidate_boundaries=[7, 8],
        existing_dependencies_copied=False, **components)


def _path(plan, key, package_root):
    path = Path(plan[key])
    if package_root is None:
        return path
    return Path(package_root) / path.relative_to(plan["root"])


def load_inputs(plan, *, package_root=None):
    """The actual final/collection input loader, called before backend creation."""
    from experiments.persistent_prefix_withholding.evaluation import validate_references, evaluation_schedule

    values = {}
    for key in ("data_panel", "references", "warmups", "warmup_references", "event8_inputs"):
        path = _path(plan, key, package_root)
        if identity(path) != plan["input_identities"].get(plan[key]):
            raise ValueError("Packaged input changed or is unbound: " + key)
        values[key] = json.loads(path.read_text())
    panel = values["data_panel"]
    requests, warmups = panel["requests"], values["warmups"]["requests"]
    if len(requests) != 128 or len(warmups) != 2 or len(values["event8_inputs"]["requests"]) != 128:
        raise ValueError("Require unchanged 128 panel, two warmups and 128 feedback inputs")
    evaluation_schedule(requests, panel["within_block_permutations"], methods=METHODS)
    for rows, references in ((requests, values["references"]), (warmups, values["warmup_references"])):
        validate_references(rows, references)
        if len({row["source_id"] for row in rows}) != len(rows):
            raise ValueError("Duplicate source identity")
        for ordinal, row in enumerate(rows):
            canonical = references[row["request_id"]]
            if (row["panel_ordinal"] != ordinal or row["collect_feedback"] is not False
                    or row["stop_token_ids"] != [151645, 151643]
                    or row["prompt_ids_sha256"] != _digest(row["prompt_ids"])
                    or canonical["source_id"] != row["source_id"]
                    or canonical["binding_digest"] != _digest({k: v for k, v in canonical.items() if k != "binding_digest"})):
                raise ValueError("Input/reference identity, token binding or termination settings differ")
    panel_sources = {r["source_id"] for r in requests}
    feedback = values["event8_inputs"]["requests"]
    if (panel_sources & {r["source_id"] for r in warmups + feedback}
            or len({r["source_id"] for r in feedback}) != 128):
        raise ValueError("Feedback or warmup identity overlaps the fixed evaluation panel")
    for ordinal, row in enumerate(feedback):
        if (row["request_index"] != row["native_stream_index"] + 1
                or row["native_stream_index"] != 896 + ordinal
                or row["collect_feedback"] is not True or row["max_new_tokens"] != 2048
                or row["stop_token_ids"] != [151645, 151643]
                or row["prompt_ids_sha256"] != _digest(row["prompt_ids"])
                or not row["prompt_ids"] or len(row["prompt_ids"]) + 2048 > 4096):
            raise ValueError("Own-feedback input binding or serving geometry differs")
    return values


def reference_readiness(plan, *, package_root=None):
    if Path(plan["references"]) != Path(plan["root"]) / REFERENCE_NAME:
        raise ValueError("Complete canonical references must be physically packaged in the new root")
    inputs = load_inputs(plan, package_root=package_root)
    return dict(status="passed", reference_path=plan["references"], complete_references=len(inputs["references"]),
        new_reference_generations=0, before_worker_construction=True, model_work=0)


def stage_complete_references(plan, source, package_root):
    """Copy the immutable complete reference payload once; never replace a partial file."""
    source, package_root = Path(source), Path(package_root)
    expected = plan["input_identities"][plan["references"]]
    if identity(source) != expected:
        raise ValueError("Complete retained reference source changed")
    if Path(plan["references"]) != Path(plan["root"]) / REFERENCE_NAME:
        raise ValueError("Reference destination is not the declared complete payload")
    destination = package_root / REFERENCE_NAME
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        stream.write(source.read_bytes())
    return reference_readiness(plan, package_root=package_root)


def build_archive(plan, package_root, archive_path):
    receipt = reference_readiness(plan, package_root=package_root)
    if json.loads((Path(package_root) / "sealed/PLAN.json").read_text()) != plan:
        raise ValueError("Packaged plan differs from checked plan")
    with tarfile.open(archive_path, "x:gz") as archive:
        for path in sorted(Path(package_root).rglob("*")):
            if path.is_symlink():
                raise ValueError("Execution payload cannot contain symlinks")
            if path.is_file():
                archive.add(path, arcname=str(path.relative_to(package_root)), recursive=False)
    return dict(status="passed", archive_identity=identity(archive_path), readiness=receipt)


def unpack_archive(archive_path, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive_path, "r:gz") as archive:
        if any(not (m.isfile() or m.isdir()) for m in archive.getmembers()):
            raise ValueError("Execution payload must contain regular files only")
        archive.extractall(destination, filter="data")
    plan = json.loads((destination / "sealed/PLAN.json").read_text())
    return dict(plan=plan, readiness=reference_readiness(plan, package_root=destination))


def build(workspace, destination, *, remote_root, source_root, execution_commit):
    import yaml
    from .runtime import LIMITS
    from .preflight import HOST_POLICY
    from experiments.persistent_prefix_withholding.launch import QUALIFIED_COMPILERS

    workspace, destination = Path(workspace), Path(destination)
    old_task = workspace / "research/dflash_persistent_prefix_withholding_20261115"
    old = old_task / "delivery/production_continuation_001"
    restart = old_task / "delivery/production_restart_001"
    prior = json.loads(authenticated_delivery_file(restart, "sealed/PLAN.json").read_text())
    destination.mkdir(parents=True, exist_ok=False)
    sealed = destination / "sealed"
    sealed.mkdir()
    plan = {key: deepcopy(prior[key]) for key in (
        "training_python", "worker_python", "target", "origin_state", "origin_export", "origin_config",
        "event7_feature_paths", "event7_schedule", "runtime_contract", "devices")}
    plan.update(root=remote_root, source_root=source_root, execution_commit=execution_commit,
        base_scientific_commit="9b2d8d8752c5b48e34a3537743ac349ab159b284",
        source_manifest=remote_root + "/SOURCE.json", scope="DEVELOPMENT_FEASIBILITY_REUSED_DATA",
        automatic_retry=False, methods=list(METHODS), limits=dict(LIMITS), new_reference_generations=0,
        qualified_compilers=dict(QUALIFIED_COMPILERS), host_resource_policy=dict(HOST_POLICY),
        configs={"C": remote_root + "/sealed/config_C.yaml"},
        compiler_root="/data/sources/minhao/.cache", compiler_growth_reserved_bytes=4*2**30,
        allocation_roots=[remote_root, str(Path(source_root).parent)], prior_allocation_bytes=0,
        controls={}, input_identities={}, reused_source_inputs={},
        candidate=dict(name="bounded_prefix_allocation_v1", smoothing=0.5, cap=0.25,
                       original_denominator=True, native_base="KPLUS"))
    kept = {prior[k] for k in ("origin_state", "origin_config", "event7_schedule")}
    kept.update(prior["event7_feature_paths"])
    draft = "/data/sources/minhao/models/qwen3_dflash_foundation_20261022/draft/"
    for path, value in prior["input_identities"].items():
        if path in kept or path.startswith((prior["origin_export"] + "/", prior["target"] + "/", draft)):
            plan["input_identities"][path] = value
    for arm, delivery in (("N", old), ("S", restart)):
        relative = "tasks/" + arm + "8/RESULT.json"
        result_path = authenticated_delivery_file(delivery, relative)
        result = json.loads(result_path.read_text())
        endpoint = endpoint_from_result(result)
        config_path = authenticated_delivery_file(delivery, "tasks/" + arm + "8/trainer/events/event_008/export/config.json")
        plan["controls"][arm] = endpoint
        plan["input_identities"][endpoint["state"]] = endpoint["state_identity"]
        plan["input_identities"][endpoint["export"] + "/model.safetensors"] = endpoint["weights_identity"]
        plan["input_identities"][endpoint["export"] + "/config.json"] = identity(config_path)
        name = arm + "8_RESULT.json"
        shutil.copyfile(result_path, sealed / name)
        plan["controls"][arm]["source_receipt"] = remote_root + "/sealed/" + name
    source_files = dict(data_panel="PANEL.json", warmups="WARMUPS.json", warmup_references="WARMUP_REFERENCES.json",
        event8_inputs="EVENT8_INPUTS.json", expected_ancestry="EXPECTED_ANCESTRY.json",
        installed_runtime_identities="INSTALLED_SOURCE_IDENTITIES.json")
    remote_old = Path("/data/sources/minhao/runs/dflash_persistent_prefix_withholding_20261115")
    for key, name in source_files.items():
        original = prior[key]
        local = old_task / "delivery" / Path(original).relative_to(remote_old)
        if identity(local) != prior["input_identities"][original]:
            raise ValueError("Immutable prior input changed: " + key)
        shutil.copyfile(local, sealed / name)
        plan[key] = remote_root + "/sealed/" + name
        plan["reused_source_inputs"][key] = dict(path=original, identity=identity(local))
    complete = authenticated_delivery_file(restart, "references/CANONICALS.json")
    plan["references"] = remote_root + "/" + str(REFERENCE_NAME)
    plan["reused_source_inputs"]["references"] = dict(path=prior["references"], identity=identity(complete))
    original_config_path = old_task / "v2_reference/ORIGINAL_CONFIG.yaml"
    if identity(original_config_path) != plan["input_identities"][plan["origin_config"]]:
        raise ValueError("Original training config identity differs")
    config = yaml.safe_load(original_config_path.read_text())
    config.update(run_id="bounded-prefix-allocation-C", output_dir=remote_root + "/training/C")
    (sealed / "config_C.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    inspection = json.loads((old_task / "v2_reference/DEPENDENCY_INSPECTION.json").read_text())
    draft_config = inspection["dependencies"]["origin_export"]["config.json"]["content"]
    if identity(old_task / "v2_reference/ORIGINAL_CONFIG.yaml") != plan["input_identities"][plan["origin_config"]]:
        raise ValueError("Original config changed")
    plan["storage"] = storage_plan(
        [len(r["prompt_ids"]) for r in json.loads((sealed / "EVENT8_INPUTS.json").read_text())["requests"]],
        weight_bytes=plan["input_identities"][plan["origin_export"] + "/model.safetensors"]["bytes"],
        recovery_bytes=max(plan["input_identities"][plan["origin_state"]]["bytes"], *(v["state_identity"]["bytes"] for v in plan["controls"].values())),
        hidden_size=draft_config["hidden_size"], captured_layers=len(draft_config["dflash_config"]["target_layer_ids"]))
    for path in sorted(sealed.iterdir()):
        plan["input_identities"][remote_root + "/sealed/" + path.name] = identity(path)
    plan["input_identities"][plan["references"]] = identity(complete)
    plan["binding"] = _digest({k: plan[k] for k in ("candidate", "controls", "input_identities", "devices")})
    _write(sealed / "PLAN.json", plan)
    _write(destination / "INPUT_READINESS.json", stage_complete_references(plan, complete, destination))
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
