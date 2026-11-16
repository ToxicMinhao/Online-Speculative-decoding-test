"""Exact published source plus metadata staging; no model import or execution.

The archive has two namespaces: source/ for the full tracked DFlash-Online tree,
and package/ for the sealed run metadata, including all canonical references.
This standalone entry point verifies bytes before importing the staged reader.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile


def identity(path):
    with Path(path).open("rb") as stream:
        return dict(bytes=Path(path).stat().st_size,
                    sha256=hashlib.file_digest(stream, "sha256").hexdigest())


def relative(name):
    path = Path(name)
    if not path.parts or path.is_absolute() or ".." in path.parts:
        raise ValueError("Archive paths must be nonempty contained relative names")
    return path


def verify_source(root, manifest):
    root = Path(root)
    if manifest.get("dirty") is not False or len(manifest["commit"]) != 40:
        raise ValueError("Source manifest must identify a clean published commit")
    for name, expected in manifest["files"].items():
        path = root / relative(name)
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Source path escapes the staged tree: " + name)
        if expected.get("kind") == "symlink":
            if not path.is_symlink() or str(path.readlink()) != expected["target"] or not path.exists():
                raise ValueError("Source internal link differs: " + name)
        elif path.is_symlink() or identity(path) != expected:
            raise ValueError("Source bytes differ: " + name)
    return dict(status="passed", files_checked=len(manifest["files"]), commit=manifest["commit"])


def build_bundle(checkout, package_root, archive_path):
    """Bundle only a clean complete tracked source and already sealed metadata."""
    checkout, package_root, archive_path = map(Path, (checkout, package_root, archive_path))
    manifest = json.loads((package_root / "SOURCE.json").read_text())
    plan = json.loads((package_root / "sealed/PLAN.json").read_text())
    commit = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=all"], text=True)
    tracked = subprocess.check_output(["git", "-C", str(checkout), "ls-files", "-z", "--", "."]).decode().split("\0")
    if (dirty or commit != manifest["commit"] or commit != plan["execution_commit"]
            or set(filter(None, tracked)) != set(manifest["files"])):
        raise ValueError("Bundle requires the exact clean published tracked source")
    source = verify_source(checkout, manifest)
    from .prepare import reference_readiness
    ready = reference_readiness(plan, package_root=package_root)
    paths = sorted(p for p in package_root.rglob("*") if not p.is_dir())
    if any(path.is_symlink() or not path.is_file() for path in paths):
        raise ValueError("Run package must contain regular files only")
    with tarfile.open(archive_path, "x:gz", dereference=False) as archive:
        for name in sorted(manifest["files"]):
            archive.add(checkout / name, arcname="source/" + name, recursive=False)
        for path in paths:
            archive.add(path, arcname="package/" + str(path.relative_to(package_root)), recursive=False)
    return dict(status="passed", archive_identity=identity(archive_path), source=source, readiness=ready,
                package_files=len(paths), model_work=0)


def _members(stream):
    rows = stream.getmembers()
    names = [row.name for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate archive member")
    links = {row.name for row in rows if row.issym()}
    for row in rows:
        path = relative(row.name)
        if len(path.parts) < 2 or path.parts[0] not in ("source", "package"):
            raise ValueError("Archive member has an unknown namespace")
        if not row.isfile() and not (row.issym() and path.parts[0] == "source"):
            raise ValueError("Only regular files and internal source links are permitted")
        if any(str(parent) in links for parent in path.parents):
            raise ValueError("Archive cannot write through a symlink parent")
        if row.issym():
            target = Path(row.linkname)
            # Normalize lexically; a source link must remain below source/.
            normalized = Path(os.path.normpath(str(path.parent / target)))
            if target.is_absolute() or not normalized.parts or normalized.parts[0] != "source":
                raise ValueError("Archive source link escapes its namespace")
    return rows


def unpack_bundle(archive_path, root, source_root, *, expected_identity, require_plan_destinations=False):
    """Exclusive extraction followed by the actual staged input/reference reader.

    Local CPU fixtures may map destinations to temporary directories. The CLI
    additionally requires exact approved destinations and the A800 helper.
    """
    archive_path, root, source_root = map(Path, (archive_path, root, source_root))
    if identity(archive_path) != expected_identity:
        raise ValueError("Transferred archive identity differs")
    if source_root.exists() or source_root.is_symlink():
        raise FileExistsError("Existing source tree must never be overwritten")
    if (root / "STAGING.json").exists():
        raise FileExistsError("Existing staging attempt must remain unchanged")
    if root.resolve().is_relative_to(source_root.resolve()) or source_root.resolve().is_relative_to(root.resolve()):
        raise ValueError("Run and source destinations must be disjoint")
    with tarfile.open(archive_path, "r:gz") as stream:
        rows = _members(stream)
        manifest = json.load(stream.extractfile("package/SOURCE.json"))
        plan = json.load(stream.extractfile("package/sealed/PLAN.json"))
        source_names = {str(Path(row.name).relative_to("source")) for row in rows if row.name.startswith("source/")}
        if source_names != set(manifest["files"]) or plan["execution_commit"] != manifest["commit"]:
            raise ValueError("Archive source membership or execution identity differs")
        if Path(plan["source_manifest"]) != Path(plan["root"]) / "SOURCE.json":
            raise ValueError("Sealed source manifest must be the physically packaged file")
        if require_plan_destinations and (str(root) != plan["root"] or str(source_root) != plan["source_root"]):
            raise ValueError("Requested destinations differ from the sealed plan")
        destinations = {"source": source_root, "package": root}
        for row in rows:
            path = Path(row.name)
            destination = destinations[path.parts[0]] / Path(*path.parts[1:])
            if destination.exists() or destination.is_symlink():
                raise FileExistsError("Refuse to overwrite staging evidence: " + str(destination))
            if not destination.resolve().is_relative_to(destinations[path.parts[0]].resolve()):
                raise ValueError("Existing destination ancestor escapes its root")
        source_root.mkdir(parents=True, exist_ok=False)
        root.mkdir(parents=True, exist_ok=True)
        for row in rows:
            path = Path(row.name)
            destination = destinations[path.parts[0]] / Path(*path.parts[1:])
            destination.parent.mkdir(parents=True, exist_ok=True)
            if row.issym():
                destination.symlink_to(row.linkname)
            else:
                with stream.extractfile(row) as incoming, destination.open("xb") as outgoing:
                    shutil.copyfileobj(incoming, outgoing, length=1024*1024)
                destination.chmod(row.mode & 0o777)
    source = verify_source(source_root, manifest)
    checked_inputs = 0
    for original, expected in plan["input_identities"].items():
        if Path(original).is_relative_to(plan["root"]):
            if identity(root / Path(original).relative_to(plan["root"])) != expected:
                raise ValueError("Packaged input differs: " + original)
            checked_inputs += 1
    sys.path[:0] = [str(source_root), str(source_root / "training"), str(source_root / "inference")]
    from experiments.bounded_prefix_allocation.prepare import reference_readiness
    ready = reference_readiness(plan, package_root=root)
    receipt = dict(status="passed", archive_identity=expected_identity, source=source,
        readiness=ready, packaged_inputs_checked=checked_inputs,
        source_root=str(source_root), root=str(root), model_work=0,
        torch_imported="torch" in sys.modules, automatic_launch=False)
    with (root / "STAGING.json").open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--bytes", required=True, type=int)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    args = parser.parse_args()
    approved = Path("/data/sources/minhao")
    for path in (args.archive, args.root, args.source_root):
        if (not path.is_absolute() or path == approved or path.resolve() != path
                or not path.resolve().is_relative_to(approved)):
            raise ValueError("Staging requires exact approved A800 paths")
    if (os.environ.get("A800_STORAGE_ROOT") != str(approved)
            or os.environ.get("HF_ENDPOINT") != "https://hf-mirror.com"):
        raise RuntimeError("Source the approved A800 environment helper before staging")
    result = unpack_bundle(args.archive, args.root, args.source_root,
        expected_identity=dict(bytes=args.bytes, sha256=args.sha256), require_plan_destinations=True)
    if result["torch_imported"]:
        raise RuntimeError("Administrative staging unexpectedly imported Torch")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
