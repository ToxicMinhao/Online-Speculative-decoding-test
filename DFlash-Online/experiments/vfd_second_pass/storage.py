"""Peak admission for five rolling states; retirement only after all users finish."""

import json
from pathlib import Path
import shutil

from experiments.kplus_full_gsm8k.storage import (
    GIB,
    allocated_bytes,
    compiler_growth,
    _retire,
    retire_redundant_features,
)
from experiments.ok_coding.storage import identity
from online_feedback.storage import write_json
from .contract import ARMS


def capacity_plan(weight_bytes):
    # Five own complete states, atomic replacement, and the shared pass1 fork:
    # seven * (weights + two BF16 moments). Five final exports remain available.
    feature_peak = (129 * 2272 + 1) * (4096 * 6 * 2 + 12)
    tensor_peak = 26 * weight_bytes + feature_peak
    padded = (tensor_peak * 105 + 99) // 100
    allowances = dict(
        compact_bytes=16 * GIB,
        workspace_bytes=2 * GIB,
        compiler_bytes=4 * GIB,
        reserve_bytes=8 * GIB,
    )
    return dict(
        weight_bytes=weight_bytes,
        state_weight_equivalents=26,
        feature_peak_bytes=feature_peak,
        tensor_peak_bytes=tensor_peak,
        tensor_peak_with_margin_bytes=padded,
        **allowances,
        required_initial_free_bytes=padded + sum(allowances.values()),
        lifecycle="incremental preparation; all-policy event barrier; five rolling own states; export before atomic replacement",
        offload_assumed=False
    )


def admit(root, plan, *, receipt_path=None):
    root = Path(root).resolve(strict=True)
    tensors, compact = [], [root]
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("Run storage cannot follow symlinks: " + str(path))
        (
            tensors
            if path.is_file() and path.suffix in {".pt", ".safetensors"}
            else compact
        ).append(path)
    tensor_bytes, compact_bytes = allocated_bytes(tensors), allocated_bytes(compact)
    if not (root / "COMPILER_BASELINE.json").exists() and any(
        (root / "campaign_001").rglob("*WORK.jsonl")
    ):
        raise ValueError("Compiler baseline is missing after scientific work started")
    compiler = compiler_growth(root)
    required = (
        max(0, plan["tensor_peak_with_margin_bytes"] - tensor_bytes)
        + max(0, plan["compact_bytes"] - compact_bytes)
        + max(0, plan["compiler_bytes"] - compiler["compiler_growth_bytes"])
        + plan["workspace_bytes"]
        + plan["reserve_bytes"]
    )
    free = shutil.disk_usage(root).free
    result = dict(
        status=(
            "passed"
            if (
                free >= required
                and tensor_bytes <= plan["tensor_peak_with_margin_bytes"]
                and compact_bytes <= plan["compact_bytes"]
                and compiler["compiler_growth_bytes"] <= plan["compiler_bytes"]
            )
            else "failed"
        ),
        free_bytes=free,
        required_bytes=required,
        tensor_bytes=tensor_bytes,
        compact_bytes=compact_bytes,
        plan=plan,
        **compiler,
        accounting="unique inode st_blocks; existing own bytes only; no future offload credit"
    )
    if receipt_path is not None:
        write_json(receipt_path, result)
    if result["status"] != "passed":
        raise ValueError("Five-policy peak storage admission failed")
    return result


def retire_pool(root, event_root, pool, results):
    """Every policy's validated successor must exist before shared features retire."""
    if set(results) != set(ARMS) or any(
        r["status"] != "completed" for r in results.values()
    ):
        raise ValueError("Do not retire shared feedback before every policy completes")
    for arm, result in results.items():
        if (
            result["state"]["status"] != "passed"
            or not Path(result["state"]["state_path"]).is_file()
        ):
            raise ValueError("Missing own successor state: " + arm)
        if identity(result["state"]["state_path"]) != result["state"]["identity"]:
            raise ValueError(
                "Validated successor changed before shared-pool retirement: " + arm
            )
    for path, expected in pool["identities"].items():
        if identity(path) != expected:
            raise ValueError("Shared feature identity changed before retirement")
    evidence = [event_root / "POOL.json", event_root / "COSTS.json"]
    evidence += [event_root / arm / "RESULT.json" for arm in ARMS]
    evidence += [Path(path).with_suffix(".json") for path in pool["features"]]
    return _retire(
        root,
        event_root / "POOL_RETIREMENT.json",
        [(Path(path), "consumed_shared_teacher_features") for path in pool["features"]],
        evidence,
        reason="All five own event states validated; exact token, anchor, score and cost evidence retained.",
        retained={
            arm: result["state"]["state_path"] for arm, result in results.items()
        },
    )


def retire_previous_export(root, event_root, previous, successor):
    if previous is None:
        return None
    previous, successor = Path(previous), Path(successor)
    if previous == successor or not (successor / "model.safetensors").is_file():
        raise ValueError("Cannot retire the active FULL export")
    return _retire(
        root,
        event_root / "PREVIOUS_FULL_EXPORT_RETIREMENT.json",
        [(previous / "model.safetensors", "obsolete_FULL_export")],
        [event_root / "FULL/RESULT.json", event_root / "PUBLICATION.json"],
        reason="Exact new FULL export published and checked; previous feedback collection is complete.",
        retained={"successor": str(successor)},
    )
