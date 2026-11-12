"""Seal the fixed event-7 lost-repair probe without loading models or runtimes.

The output is a private staging package, separate from the Git checkout. Its
paths already name the A800 destination; remote preflight authenticates the
large checkpoints instead of copying them into this package.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


MANIFEST_IDENTITY = dict(bytes=10868762,
    sha256="764fa339938a240259949040134581a2231de542b7c799af5a7eb85c5ea1dbd4")
ROWS_SHA256 = "90d84cdfa142e8be0eda2ef03778c190e51e74e899aac277e2e80fce32c6fda3"
WEIGHT_HASHES = {
    "K4": "4f3010e961bd7dcf35fa100486c355ffbc7629d816dda6e2b8fa001878255701",
    "K5": "9dcb0d637eabef6bd4e34c552da0448faa9510e6bcaf5c753a6bd1b43674b795",
    "K6": "8c257620e44226c5c0469e5886224ed5d2dcf9fa2a94eddbdd10d7e36c697560",
}
EXPORT_CONFIG_IDENTITY = dict(bytes=1211,
    sha256="5749d96966f9ce7e6a9662c9f482b8aa0d461845fb6b9ce09ac2c08311fab2aa")
K5_EXPORT = (
    "/data/sources/minhao/runs/dflash_survival_head_20261110/execution_001/"
    "campaign_001/gsm8k/event_005/KPLUS/trainer/events/event_005/export"
)
RECOVERY_IDENTITY = dict(bytes=6298193179,
    sha256="15ac28e1235851a15158e789af1480fd2d82535903b1dec5d992ec843c9ae3a1")
GPU_SECONDS_LIMIT = 14400
GIB = 1024 ** 3


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return dict(bytes=path.stat().st_size, sha256=digest.hexdigest())


def checked_identity(path, expected):
    actual = identity(path)
    if actual != {key: expected[key] for key in ("bytes", "sha256")}:
        raise ValueError("Pinned local file differs: " + str(path))
    return actual


def select_requests(manifest):
    """Take the scheduled contiguous window, never select by model outcomes."""
    rows = deepcopy(manifest["train_requests"][768:896])
    if len(rows) != 128:
        raise ValueError("Original manifest lacks the complete 128-request event 7")
    for ordinal, row in enumerate(rows, 768):
        if (row["request_id"] != f"train_{ordinal:05d}"
                or row["request_index"] != ordinal + 1
                or row["dataset_index"] != ordinal
                or row["event_id"] != "event_007"
                or not row["collect_feedback"]):
            raise ValueError("Scheduled event-7 row identity or chronology differs")
        tokens = row["prompt_ids"]
        if (not tokens or any(type(token) is not int or token < 0 for token in tokens)
                or row["max_new_tokens"] != 2048 or len(tokens) + 2048 > 4096):
            raise ValueError("Scheduled request violates the fixed full-context contract")
    for label, values in (
        ("request IDs", [r["request_id"] for r in rows]),
        ("source IDs", [r["dataset_id"] for r in rows]),
        ("token prompts", [tuple(r["prompt_ids"]) for r in rows]),
    ):
        if len(set(values)) != 128:
            raise ValueError("Duplicate " + label + " in the fixed event-7 window")
    return rows


def source_manifest(code):
    """Authenticate tracked DFlash-Online files from a clean committed checkout."""
    code = Path(code).resolve()
    repository = Path(subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], cwd=code, text=True).strip())
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repository, text=True).strip()
    if dirty:
        raise ValueError("Execution checkout must be clean before sealing: " + dirty)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
    prefix = code.relative_to(repository)
    paths = subprocess.check_output(
        ["git", "ls-files", "-z", "--", str(prefix)], cwd=repository).decode().split("\0")
    files, symlinks = {}, {}
    for name in filter(None, paths):
        path = repository / name
        relative = path.relative_to(code).as_posix()
        if path.is_symlink():
            symlinks[relative] = str(path.readlink())
        else:
            files[relative] = identity(path)
    if not files:
        raise ValueError("No tracked DFlash-Online source files found")
    return dict(status="verified", commit=commit, dirty=False,
                source_path=str(code), files=files, symlinks=symlinks)


def checkpoints(runtime_plan, inventory):
    prior = runtime_plan["checkpoints"]["gsm8k"]
    exports = {"K4": prior["KPLUS4"]["export"], "K5": K5_EXPORT,
               "K6": prior["KPLUS6"]["export"]}
    if exports["K6"] != inventory["k6_export"]:
        raise ValueError("Runtime plan and inventory identify different K+6 exports")
    result = {}
    for arm, export in exports.items():
        expected = dict(bytes=2097259104, sha256=WEIGHT_HASHES[arm])
        if arm != "K5" and prior["KPLUS" + arm[-1]]["weights_identity"] != expected:
            raise ValueError("Retained checkpoint identity differs: " + arm)
        result[arm] = dict(export=export, weights_identity=expected,
                           config_identity=dict(EXPORT_CONFIG_IDENTITY))
    return result


def seal(manifest_path, inventory_path, runtime_plan_path, output, remote_root, *, code=None):
    """Create a fresh private package; no training or native execution occurs."""
    code = Path(code or Path(__file__).resolve().parents[2]).resolve()
    output, remote_root = Path(output).resolve(), Path(remote_root)
    if output.exists():
        raise FileExistsError("Preserve an existing sealed package; choose a fresh destination")
    if output.is_relative_to(code.parent):
        raise ValueError("Private request IDs/tokens must remain outside the Git checkout")
    if not remote_root.is_absolute() or not remote_root.is_relative_to("/data/sources/minhao/runs"):
        raise ValueError("Remote run root is outside approved artifact storage")
    manifest_id = checked_identity(manifest_path, MANIFEST_IDENTITY)
    inventory, runtime_plan = read(inventory_path), read(runtime_plan_path)
    if {k: inventory["manifest"][k] for k in ("bytes", "sha256")} != manifest_id:
        raise ValueError("Inventory and original manifest identities disagree")
    rows = select_requests(read(manifest_path))
    rows_hash = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if rows_hash != ROWS_SHA256 or rows_hash != inventory["manifest"]["original_rows_canonical_json_sha256"]:
        raise ValueError("Original event-7 row binding differs")
    if inventory["stop_token_ids"] != [151645, 151643]:
        raise ValueError("Qualified stop-token identities changed")
    source = source_manifest(code)
    checkpoint_rows = checkpoints(runtime_plan, inventory)
    shared = runtime_plan["shared"]
    if inventory["target_path"] != shared["target_path"]:
        raise ValueError("Target model paths differ between retained inventories")
    pin_documents = {}
    for name, entry in inventory["pin_files"].items():
        actual = identity(entry["local"])
        if actual["sha256"] != entry["sha256"]:
            raise ValueError("Native pin file changed: " + name)
        pin_documents[name] = read(entry["local"])
    model_pin = pin_documents["MODEL_PIN.json"]
    if model_pin["target"]["files"] != shared["target_files"]:
        raise ValueError("Target model file inventory differs from native runtime plan")
    runtime_pin = pin_documents["RUNTIME_PIN.json"]
    if runtime_pin["native_runtime_sha256"] != shared["runtime"]["native_runtime_sha256"]:
        raise ValueError("Native runtime source pins differ")
    for relative, expected in runtime_pin["native_runtime_sha256"].items():
        if identity(code / "external_runtime/sglang/srt" / relative)["sha256"] != expected:
            raise ValueError("Bundled native runtime source differs: " + relative)

    # Copy the configuration byte-for-byte. The read-only probe restores K+6
    # from recovery; its inherited model/training configuration is not rewritten.
    output.mkdir(parents=True)
    sealed = output / "sealed"
    sealed.mkdir()
    write(output / "SOURCE.json", source)
    pinned = {str(remote_root / "SOURCE.json"): identity(output / "SOURCE.json")}
    copies = {"ORIGINAL_MANIFEST.json": manifest_path,
              "config_KPLUS.yaml": inventory["config_path"]}
    copies.update({name: entry["local"] for name, entry in inventory["pin_files"].items()})
    for name, original in copies.items():
        shutil.copyfile(original, sealed / name)
        pinned[str(remote_root / "sealed" / name)] = identity(sealed / name)
    for family, base_path in (("target", inventory["target_path"]),
                              ("draft", inventory["draft_foundation_path"])):
        pinned.update({str(Path(base_path) / name): value
                       for name, value in model_pin[family]["files"].items()})
    recovery = inventory["k6_recovery_path_reported_by_parent_remote_audit"]
    if Path(recovery) != Path(checkpoint_rows["K6"]["export"]).parents[2] / "training_state.pt":
        raise ValueError("Require the actual K+6 trainer-boundary recovery artifact")
    pinned[recovery] = dict(RECOVERY_IDENTITY)
    for row in checkpoint_rows.values():
        pinned[str(Path(row["export"]) / "config.json")] = row["config_identity"]

    requests = [dict(row, stop_token_ids=list(inventory["stop_token_ids"])) for row in rows]
    plan = dict(
        schema="dflash-lost-repair-probe-v1", execution_commit=source["commit"],
        source_manifest_path=str(remote_root / "SOURCE.json"),
        requests=requests, original_row_range=[768, 896],
        manifest_identity=manifest_id, original_rows_canonical_json_sha256=rows_hash,
        checkpoints=checkpoint_rows, checkpoint_identities=checkpoint_rows,
        pinned_files=pinned, target_path=inventory["target_path"],
        worker_python=inventory["worker_python"], training_python=inventory["training_python"],
        config_path=str(remote_root / "sealed/config_KPLUS.yaml"), recovery_path=recovery,
        runtime_contract=dict(
            serving={k: runtime_pin[k] for k in ("python_version", "torch", "transformers", "cuda")},
            training=dict(pin_documents["TRAINING_PIN.json"]["environment"]),
            native_runtime_sha256=runtime_pin["native_runtime_sha256"],
        ),
        versions=dict(
            serving=dict(python=runtime_pin["python_version"], torch=runtime_pin["torch"],
                         transformers=runtime_pin["transformers"]),
            training=dict(pin_documents["TRAINING_PIN.json"]["environment"]),
        ),
        serving=inventory["serving"], gpu_seconds_limit=GPU_SECONDS_LIMIT,
        storage=dict(artifact_allowance_bytes=40 * GIB, reserve_bytes=8 * GIB,
                     required_free_bytes=48 * GIB),
        output_root=str(remote_root / "campaign_001"),
        scope=dict(feedback_requests=128, readonly_presentations=256,
                   optimizer_steps=0, backward_calls=0, timed_endpoints=0,
                   natural_start_selection="complete_census_including_opening",
                   full_valid_census_required=True),
        provenance=dict(inventory=identity(inventory_path), runtime_plan=identity(runtime_plan_path),
                        checkpoint_config_identity_basis="executor remote audit before sealing",
                        historical_capture_reused=False),
    )
    write(sealed / "PLAN.json", plan)
    package_files = {str(remote_root / path.relative_to(output)): identity(path)
                     for path in sorted(output.rglob("*")) if path.is_file()}
    write(sealed / "SEAL.json", dict(status="sealed", execution_commit=source["commit"],
                                      requests=128, files=package_files))
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "inventory", "runtime-plan", "output", "remote-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    plan = seal(args.manifest, args.inventory, args.runtime_plan, args.output, args.remote_root)
    print(json.dumps(dict(status="sealed", requests=len(plan["requests"]),
                          execution_commit=plan["execution_commit"],
                          plan=str(args.output / "sealed/PLAN.json"))))


if __name__ == "__main__":
    main()
