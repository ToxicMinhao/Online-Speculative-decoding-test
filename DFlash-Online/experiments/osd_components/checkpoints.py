"""Exact endpoint substitution for the fixed five-layer, 58-tensor diagnostic.

This module only constructs checkpoint files on the CPU. Native fused-weight,
cache and generation qualification remains the experiment runner's responsibility.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import tempfile
import time

from online_feedback.sglang_worker import checkpoint_layout, semantic_config
from online_feedback.storage import write_json


CONDITIONING_KEYS = frozenset({"fc.weight", "hidden_norm.weight"})
LAYER_SUFFIXES = (
    "input_layernorm.weight", "self_attn.q_proj.weight",
    "self_attn.k_proj.weight", "self_attn.v_proj.weight",
    "self_attn.o_proj.weight", "self_attn.q_norm.weight",
    "self_attn.k_norm.weight", "post_attention_layernorm.weight",
    "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
)
BODY_KEYS = frozenset({"norm.weight"} | {
    f"layers.{layer}.{suffix}" for layer in range(5) for suffix in LAYER_SUFFIXES
})
EXPECTED_KEYS = CONDITIONING_KEYS | BODY_KEYS
RECEIPTS = ("STORAGE_ADMISSION.json", "PARAMETER_INVENTORY.json",
            "CHECKPOINT_QUALIFICATION.json")


def partition_keys(keys) -> dict:
    """Reject any addition/omission; never classify arbitrary ``layers.*`` keys."""
    actual = set(keys)
    if actual != EXPECTED_KEYS:
        raise ValueError(
            "Expected exactly the fixed 58 draft tensors; "
            f"missing={sorted(EXPECTED_KEYS - actual)}, "
            f"unexpected={sorted(actual - EXPECTED_KEYS)}")
    if CONDITIONING_KEYS & BODY_KEYS or len(EXPECTED_KEYS) != 58:
        raise AssertionError("The fixed conditioning/body partition is not disjoint")
    return {"C": sorted(CONDITIONING_KEYS), "B": sorted(BODY_KEYS)}


def storage_budget(export_bytes: int, free_bytes: int) -> dict:
    """Pure admission calculation; GB is decimal, the 8 GiB reserve is binary."""
    if (type(export_bytes) is not int or export_bytes <= 0
            or type(free_bytes) is not int or free_bytes < 0):
        raise ValueError("Storage sizes must be integer bytes; export must be positive")
    budget = {
        "export_bytes": export_bytes,
        "export_slots": 3,
        "export_and_atomic_overlap_bytes": 3 * export_bytes,
        "records_and_work_bytes": 2_000_000_000,
        "reserve_bytes": 8 * 2**30,
        "minimum_free_bytes": 18_000_000_000,
        "free_bytes": free_bytes,
    }
    budget["required_bytes"] = max(
        budget["minimum_free_bytes"],
        budget["export_and_atomic_overlap_bytes"]
        + budget["records_and_work_bytes"] + budget["reserve_bytes"],
    )
    budget["admitted"] = free_bytes >= budget["required_bytes"]
    return budget


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_files(path):
    return {
        item.name: {"bytes": item.stat().st_size, "sha256": _file_hash(item)}
        for item in [path / "config.json", *sorted(path.glob("*.safetensors"))]
    }


def _tensor_hash(value):
    import torch

    # BF16 NumPy conversion is unsupported; hash its exact underlying bytes.
    return hashlib.sha256(value.contiguous().view(torch.uint8).numpy()).hexdigest()


def _equal_bits(left, right):
    import torch

    return (left.shape == right.shape and left.dtype == right.dtype
            and torch.equal(left.view(torch.int16), right.view(torch.int16)))


def _load(path):
    from safetensors import safe_open

    state = {}
    for filename in sorted(path.glob("*.safetensors")):
        with safe_open(filename, framework="pt", device="cpu") as source:
            for name in source.keys():
                if name in state:
                    raise ValueError("Duplicate checkpoint tensor: " + name)
                state[name] = source.get_tensor(name)
    partition_keys(state)
    return state


def _compose(conditioning, body):
    """Select only explicit approved members; arithmetic is intentionally absent."""
    state = {key: conditioning[key] for key in sorted(CONDITIONING_KEYS)}
    state.update({key: body[key] for key in sorted(BODY_KEYS)})
    partition_keys(state)
    return state


def _materialize(destination, expected, origins, frozen, layout, config):
    from safetensors import safe_open
    from safetensors.torch import save_file

    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.pending-",
                                      dir=destination.parent))
    try:
        shutil.copyfile(frozen / "config.json", temporary / "config.json")
        save_file(expected, temporary / "model.safetensors", metadata={"format": "pt"})
        if checkpoint_layout(temporary, check_finite=True) != layout:
            raise ValueError("Reloaded hybrid tensor keys, shapes or dtypes differ")
        if semantic_config(temporary) != config:
            raise ValueError("Reloaded hybrid changes semantic draft configuration")
        if (temporary / "config.json").read_bytes() != (frozen / "config.json").read_bytes():
            raise ValueError("Hybrid configuration differs from released bytes")
        checks = {}
        with safe_open(temporary / "model.safetensors", framework="pt", device="cpu") as saved:
            for key in sorted(EXPECTED_KEYS):
                value = saved.get_tensor(key)
                if not _equal_bits(value, expected[key]):
                    raise ValueError("Reloaded hybrid tensor has wrong exact origin: " + key)
                checks[key] = {"origin": origins[key], "exact_bytes_equal": True,
                               "sha256": _tensor_hash(value)}
        files = _source_files(temporary)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError("Refusing to replace checkpoint work: " + str(destination))
        temporary.rename(destination)
        return {"status": "passed", "path": str(destination), "files": files,
                "per_key": checks, "released_config_bytes_equal": True,
                "layout_equal": True, "finite": True}
    finally:
        # Only our uncommitted temporary export is disposable after a failure.
        if temporary.exists():
            shutil.rmtree(temporary)


def build_hybrids(frozen: Path, trained: Path, output: Path) -> dict:
    """Validate F/O, then atomically save and qualify C-only and B-only exports.

    F and O are returned as untouched original directories. ``output`` may
    contain unrelated runner records, but all three receipt names and both arm
    directories must be fresh. No retry, replacement or source mutation occurs.
    """
    import torch

    started = time.perf_counter()
    frozen, trained = Path(frozen).resolve(strict=True), Path(trained).resolve(strict=True)
    output = Path(output).resolve()
    for source in (frozen, trained):
        if not source.is_dir() or not (source / "config.json").is_file():
            raise ValueError("Checkpoint directory and config.json are required")
        if output == source or source in output.parents:
            raise ValueError("Hybrid output must not be inside either source checkpoint")
    for name in (*RECEIPTS, "C-only", "B-only"):
        if (output / name).exists() or (output / name).is_symlink():
            raise FileExistsError("Refusing to replace checkpoint work: " + str(output / name))

    layouts = {"F": checkpoint_layout(frozen, check_finite=True),
               "O": checkpoint_layout(trained, check_finite=True)}
    groups = partition_keys(layouts["F"])
    partition_keys(layouts["O"])
    if layouts["F"] != layouts["O"]:
        raise ValueError("Endpoint tensor keys, shapes or dtypes differ")
    if any(dtype != "BF16" for _, dtype in layouts["F"].values()):
        raise ValueError("Every endpoint tensor must have BF16 dtype")
    config = semantic_config(frozen)
    if semantic_config(trained) != config:
        raise ValueError("Endpoints differ in semantic draft configuration")
    if config.get("num_hidden_layers") != 5:
        raise ValueError("The fixed checkpoint partition requires five draft layers")

    source_files = {"F": _source_files(frozen), "O": _source_files(trained)}
    export_bytes = max(sum(row["bytes"] for row in files.values())
                       for files in source_files.values())
    # A small header allowance also covers a consolidated safetensors header.
    export_bytes += 64 * 2**10
    output.mkdir(parents=True, exist_ok=True)
    admission = storage_budget(export_bytes, shutil.disk_usage(output).free)
    write_json(output / RECEIPTS[0], admission)
    if not admission["admitted"]:
        raise ValueError("Insufficient storage for two hybrids, atomic overlap and reserve")

    states = {"F": _load(frozen), "O": _load(trained)}
    rows = []
    for key in sorted(EXPECTED_KEYS):
        left, right = states["F"][key], states["O"][key]
        changed = int(torch.count_nonzero(left.view(torch.int16) != right.view(torch.int16)))
        group = "C" if key in CONDITIONING_KEYS else "B"
        rows.append({"key": key, "group": group, "shape": list(left.shape),
                     "dtype": "BF16", "numel": left.numel(),
                     "bytes": left.numel() * left.element_size(),
                     "changed_elements": changed,
                     "frozen_sha256": _tensor_hash(left),
                     "trained_sha256": _tensor_hash(right),
                     "origins": {"F": "F", "O": "O",
                                 "C-only": "O" if group == "C" else "F",
                                 "B-only": "F" if group == "C" else "O"}})
    inventory = {"tensor_count": len(rows), "partition": groups, "tensors": rows,
                 "changed_element_convention": "unequal BF16 bit patterns",
                 "sources": {"F": {"path": str(frozen), "files": source_files["F"]},
                             "O": {"path": str(trained), "files": source_files["O"]}}}
    for group in ("C", "B", "all"):
        selected = [row for row in rows if group == "all" or row["group"] == group]
        inventory[group] = {"tensor_count": len(selected), **{
            field: sum(row[field] for row in selected)
            for field in ("numel", "bytes", "changed_elements")}}
    write_json(output / RECEIPTS[1], inventory)

    reconstructed = {}
    for arm in ("F", "O"):
        combined = _compose(states[arm], states[arm])
        reconstructed[arm] = {
            key: _equal_bits(combined[key], states[arm][key]) for key in sorted(EXPECTED_KEYS)}
        if not all(reconstructed[arm].values()):
            raise ValueError("Endpoint reconstruction through the partition failed: " + arm)

    qualified = {}
    arms = {"F": str(frozen), "O": str(trained)}
    for arm, c_source, b_source in (("C-only", "O", "F"), ("B-only", "F", "O")):
        expected = _compose(states[c_source], states[b_source])
        origins = {key: c_source if key in CONDITIONING_KEYS else b_source
                   for key in EXPECTED_KEYS}
        destination = output / arm
        qualified[arm] = _materialize(destination, expected, origins,
                                      frozen, layouts["F"], config)
        arms[arm] = str(destination)
    unchanged = {"F": _source_files(frozen) == source_files["F"],
                 "O": _source_files(trained) == source_files["O"]}
    if not all(unchanged.values()):
        raise ValueError("Source checkpoint changed during hybrid construction")
    qualification = {
        "status": "passed", "scope": "CPU construction only; native publication pending",
        "partition_complete_and_disjoint": True,
        "endpoint_reconstruction_exact": reconstructed,
        "original_sources_unchanged": unchanged,
        "semantic_config_equal": True, "hybrids": qualified,
        "construction_seconds": time.perf_counter() - started,
    }
    write_json(output / RECEIPTS[2], qualification)
    return {"arms": arms, "inventory": inventory, "qualification": qualification,
            "storage_admission": admission}
