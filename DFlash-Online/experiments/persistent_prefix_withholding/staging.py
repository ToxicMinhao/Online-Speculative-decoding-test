"""Assemble and check the actual reference payload before any worker starts.

These CPU helpers are shared by local package construction, remote unpacking,
and production-entry validation. No model or backend is created here.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tarfile

from .references import file_identity, load_missing_requests, load_reference_inputs

REFERENCE_NAME = Path("references/CANONICALS.json")


def _destination(plan, package_root):
    root = Path(plan["root"])
    destination = Path(plan["references"])
    if destination != root / REFERENCE_NAME:
        raise ValueError("Reference destination must be the new run's references/CANONICALS.json")
    actual_root = root if package_root is None else Path(package_root)
    actual_destination = actual_root / REFERENCE_NAME
    if not actual_destination.resolve().is_relative_to(actual_root.resolve()):
        raise ValueError("Reference destination escapes its run directory")
    return actual_root, actual_destination


def _local_reference_plan(plan, package_root, input_resolver):
    local = deepcopy(plan)
    _, destination = _destination(plan, package_root)
    local["references"] = str(destination)
    for key in ("data_panel", "missing_references", "reused_references"):
        local[key] = str(input_resolver(plan[key]))
        local["input_identities"][local[key]] = plan["input_identities"][plan[key]]
    return local


def reference_readiness(plan, *, package_root=None, input_resolver=Path):
    """Validate the real staged seed and full consumer input before scheduling."""
    local = _local_reference_plan(plan, package_root, input_resolver)
    requests, reused = load_missing_requests(local)
    expected = plan["input_identities"][plan["reused_references"]]
    actual = file_identity(local["references"])
    if actual != expected:
        raise ValueError("Staged canonical seed must preserve the exact immutable reused-reference bytes")
    return dict(status="passed", reference_path=plan["references"], seed_identity=actual,
        immutable_source=plan["reused_references"], reused_references=len(reused),
        missing_references=len(requests), panel_requests=len(reused)+len(requests),
        model_work=0, before_worker_construction=True)


def seed_reference_destination(plan, *, package_root=None, input_resolver=Path):
    """Create the seed once from authenticated old references; never overwrite."""
    load_reference_inputs(plan, input_resolver=input_resolver)
    root, destination = _destination(plan, package_root)
    source = Path(input_resolver(plan["reused_references"]))
    payload = source.read_bytes()
    if file_identity(source) != plan["input_identities"][plan["reused_references"]]:
        raise ValueError("Immutable reference source changed while preparing its seed")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        stream.write(payload)
    receipt = reference_readiness(plan, package_root=root, input_resolver=input_resolver)
    with (root / "REFERENCE_SEED.json").open("x") as stream:
        stream.write(json.dumps(receipt, indent=2)+"\n")
    return receipt


def build_package_archive(plan, package_root, archive_path, *, input_resolver=Path,
                          source_delta=None):
    """Pack exactly the checked payload; source-delta paths are explicit."""
    package_root, archive_path = Path(package_root), Path(archive_path)
    receipt = reference_readiness(plan, package_root=package_root, input_resolver=input_resolver)
    if json.loads((package_root / "sealed/PLAN.json").read_text()) != plan:
        raise ValueError("Packaged PLAN differs from the validated reference destination")
    with tarfile.open(archive_path, "x:gz") as archive:
        for path in sorted(package_root.rglob("*")):
            if path.is_file() or path.is_symlink():
                archive.add(path, arcname=str(path.relative_to(package_root)), recursive=False)
        for name, path in sorted((source_delta or {}).items()):
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Source delta must remain within its staging directory")
            archive.add(path, arcname="source_delta/"+str(relative), recursive=False)
    return dict(status="passed", archive_identity=file_identity(archive_path), reference_seed=receipt)


def unpack_package_archive(archive_path, destination, *, input_resolver=Path):
    """Unpack without replacing prior evidence, then validate complete readiness."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            target = destination / member.name
            if not member.isdir() and (target.exists() or target.is_symlink()):
                raise FileExistsError("Package extraction would replace existing evidence: "+str(target))
        archive.extractall(destination, filter="data")
    plan = json.loads((destination / "sealed/PLAN.json").read_text())
    receipt = reference_readiness(plan, package_root=destination, input_resolver=input_resolver)
    return dict(status="passed", plan=plan, reference_seed=receipt)
