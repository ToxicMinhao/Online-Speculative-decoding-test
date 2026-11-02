"""Phase-aware storage admission for one durable reservoir and three arms.

The lifecycle never relies on historical deletion. Superseded files are retired
only after their own successor and remaining serving dependencies are verified.
"""

from pathlib import Path
import os
import shutil
import time

import numpy as np

from online_feedback.storage import write_json

GIB = 2**30
ROW_BYTES = (5 * 4096 + 4096) * 2 + 8 + 4


def capacity_plan(requests, weight_bytes, *, compiler_baseline):
    if len(requests) != 1024:
        raise ValueError("Capacity must cover the complete sealed stream")
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([20261102, 0, 2])))
    slots = []
    phases = []
    for event in range(1, 9):
        fresh = requests[(event - 1) * 128:event * 128]
        memory_rows = sum(len(requests[index]["prompt_ids"]) + 2048 for index in slots)
        fresh_rows = sum(len(row["prompt_ids"]) + 2048 for row in fresh)
        packet_overlap = max(len(row["prompt_ids"]) + 2048 for row in fresh)
        # Capture: initial W + three current 3W states + three W exports.
        # Training: initial W + three current 3W states + one pending 3W state.
        # A new prepared packet can coexist with its native/terminal features.
        phases.append({
            "event": event, "past_memory_rows": memory_rows,
            "fresh_rows": fresh_rows, "incremental_raw_terminal_rows": packet_overlap,
            "capture_tensor_bytes": 13 * weight_bytes + (memory_rows + fresh_rows + packet_overlap) * ROW_BYTES,
            "training_tensor_bytes": 13 * weight_bytes + (memory_rows + fresh_rows) * ROW_BYTES,
        })
        for index in range((event - 1) * 128, event * 128):
            if index < 128:
                slots.append(index)
            else:
                destination = int(rng.integers(0, index + 1))
                if destination < 128:
                    slots[destination] = index
        remaining_rows = sum(len(requests[index]["prompt_ids"]) + 2048 for index in slots)
        # Export after eviction: initial W + states 9W + exports 3W + workspace W.
        phases[-1]["export_tensor_bytes"] = 14 * weight_bytes + remaining_rows * ROW_BYTES
        phases[-1]["post_eviction_memory_rows"] = remaining_rows
    peak = max(row[key] for row in phases for key in (
        "capture_tensor_bytes", "training_tensor_bytes", "export_tensor_bytes"))
    padded = (peak * 105 + 99) // 100
    allowances = {"compiler_growth_bytes": 4 * GIB, "compact_bytes": 3 * GIB,
                  "workspace_bytes": 2 * GIB, "reserve_bytes": 8 * GIB}
    return {
        "weight_bytes": weight_bytes, "row_bytes": ROW_BYTES, "phases": phases,
        "tensor_peak_bytes": peak, "tensor_peak_with_margin_bytes": padded,
        "tensor_headroom_percent": 5, **allowances,
        "required_initial_free_bytes": padded + sum(allowances.values()),
        "compiler_baseline": compiler_baseline,
        "compiler_allowance_basis": "4 GiB additional growth exceeds the measured existing qualified caches; same kernels and cached runtime are reused. Enforced during this run.",
        "lifecycle": "One shared initial; retire preceding serving exports after all serving checks; defer exports until successor state validation and reservoir eviction.",
    }


def allocated_bytes(paths):
    """Count allocated filesystem blocks once per inode, including hardlinks."""
    seen = set()
    total = 0
    for root in map(Path, paths):
        entries = root.rglob("*") if root.is_dir() else [root]
        for path in entries:
            if not path.is_file() or path.is_symlink():
                continue
            stat = path.stat()
            inode = (stat.st_dev, stat.st_ino)
            if inode not in seen:
                seen.add(inode)
                total += stat.st_blocks * 512
    return total


def admit(root, plan, receipt_path):
    free = shutil.disk_usage(root).free
    receipt = {"status": "passed" if free >= plan["required_initial_free_bytes"] else "failed",
               "free_bytes": free, "required_bytes": plan["required_initial_free_bytes"],
               "headroom_bytes": free - plan["required_initial_free_bytes"], "plan": plan}
    write_json(receipt_path, receipt)
    if receipt["status"] != "passed":
        raise RuntimeError(f"Storage admission failed: missing {-receipt['headroom_bytes']} bytes")
    return receipt


def check_growth(root, plan, *, pending_write_bytes=0):
    """Resource gate, not permission to reduce reserves or scientific work."""
    started = time.perf_counter()
    root = Path(root)
    caches = plan["compiler_baseline"]
    compiler_growth = sum(max(0, allocated_bytes([row["path"]]) - row["allocated_bytes"]) for row in caches)
    compact = allocated_bytes([path for path in root.rglob("*")
                               if path.is_file() and path.suffix not in (".pt", ".safetensors")])
    free = shutil.disk_usage(root).free
    receipt = {"free_bytes": free, "compiler_growth_bytes": compiler_growth,
               "compact_allocated_bytes": compact, "pending_write_bytes": pending_write_bytes,
               "resource_check_wall_s": time.perf_counter() - started}
    if (compiler_growth > plan["compiler_growth_bytes"] or compact > plan["compact_bytes"]
            or free < plan["reserve_bytes"] + pending_write_bytes):
        raise RuntimeError(f"Bounded storage allowance exceeded: {receipt}")
    return receipt


def require_write_space(root, plan, pending_write_bytes):
    free = shutil.disk_usage(root).free
    if free < plan["reserve_bytes"] + pending_write_bytes:
        raise RuntimeError("Insufficient free space for next declared write while preserving reserve")


def retire_owned(path, *, root, receipt_path, reason, retained_evidence):
    """Delete exactly one verified disposable new-run file, with a durable receipt."""
    path, root = Path(path).absolute(), Path(root).absolute()
    if not path.is_relative_to(root) or path.resolve() != path or not path.is_file():
        raise ValueError("Retirement is limited to named existing files in this new run")
    for evidence in retained_evidence:
        if not Path(evidence).is_file():
            raise ValueError("Retained successor/evidence must exist before retiring a dependency")
    stat = path.stat()
    record = {"status": "prepared", "path": str(path), "bytes": stat.st_size,
              "allocated_bytes": stat.st_blocks * 512, "inode": stat.st_ino,
              "hardlinks": stat.st_nlink, "reason": reason,
              "retained_evidence": [str(value) for value in retained_evidence],
              "archived": False}
    write_json(receipt_path, record)
    path.unlink()
    write_json(receipt_path, {**record, "status": "retired"})
    return record
