"""Read-only remaining-capacity calculation after complete-boundary admission.

The caller validates source, identities and chronological completion before this
module runs. Existing allocations cancel against current free space. Existing
retirement receives zero credit because immediate physical release is unproven.
"""
import json
import os
from pathlib import Path
import stat

from .storage import (
    ARMS, SEEDS, COMPILER_CACHE_ALLOWANCE, FEATURE_ROW_BYTES, GIB, WEIGHTS,
    full_campaign_budget, required_bytes,
)

FIXED_ALLOWANCE = 12 * GIB
NO_EXISTING_RELEASE_PROOF = (
    "Physical release not established; open handles/snapshots may retain allocations"
)


def allocation_envelope(new_live_bytes, reclaimed_existing_bytes):
    """Preserve fresh slack arithmetic, then subtract proven physical release."""
    if new_live_bytes < 0 or reclaimed_existing_bytes < 0:
        raise ValueError("Storage accounting cannot contain negative allocations")
    return max(0, required_bytes(new_live_bytes) - FIXED_ALLOWANCE
               - reclaimed_existing_bytes)


def _signature(info):
    return (info.st_dev, info.st_ino, info.st_nlink, info.st_size,
            info.st_blocks, info.st_mtime_ns, info.st_ctime_ns)


def inventory_allocations(campaign_root):
    """Inventory regular files without following symlinks or counting aliases twice."""
    root = Path(campaign_root).resolve()
    device = root.stat().st_dev
    inodes, excluded = {}, []
    for folder, directories, files in os.walk(root, followlinks=False):
        # os.walk does not follow directory symlinks; make that visible in evidence.
        for name in list(directories):
            path = Path(folder) / name
            if path.is_symlink():
                excluded.append(str(path))
                directories.remove(name)
        for name in sorted(files):
            path = Path(folder) / name
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode):
                excluded.append(str(path))
                continue
            key = (info.st_dev, info.st_ino)
            row = inodes.setdefault(key, {
                "device": info.st_dev, "inode": info.st_ino,
                "allocated_bytes": info.st_blocks * 512, "logical_bytes": info.st_size,
                "link_count": info.st_nlink, "aliases": [],
                "same_filesystem": info.st_dev == device,
                "stat_signature": list(_signature(info)),
            })
            if row["stat_signature"] != list(_signature(info)):
                raise ValueError("Campaign allocation changed during inventory: " + str(path))
            row["aliases"].append(str(path))
    rows = sorted(inodes.values(), key=lambda row: (row["device"], row["inode"]))
    for row in rows:
        row["aliases"].sort()
        row["all_links_inventoried"] = len(row["aliases"]) == row["link_count"]
    return {"filesystem_device": device, "inodes": rows,
            "unique_allocated_bytes": sum(row["allocated_bytes"] for row in rows
                                          if row["same_filesystem"]),
            "excluded_nonregular_paths": sorted(excluded),
            "allocation_basis": "Regular-file st_blocks * 512, unique (st_dev, st_ino); directory metadata uses fixed allowance"}


def _physical_reclamation(path, expected):
    """No existing-file release is established by this admission mechanism.

    Even unshared blocks with every hard link scheduled for unlink may remain
    allocated through an open descriptor, mapping or filesystem snapshot. Do
    not speculate about that capacity; all existing retirement credit is zero.
    """
    return {"bytes": 0, "reason": NO_EXISTING_RELEASE_PROOF}


def _retirement_credits(root, inventory, completed_jobs, completed_retention):
    """Authorize only rolling event-1/3 paths and their replaced latest alias."""
    unlinks, latest = {}, {}
    for seed in SEEDS:
        for event in (1, 3):
            if (seed, event + 1) in completed_retention:
                continue
            retirement = (seed, event + 1)
            for arm in ARMS:
                if (seed, event, arm) not in completed_jobs:
                    continue
                arm_root = root / f"seed_{seed}" / arm
                old = arm_root / "events" / f"event_{event:03d}"
                bindings_path = root / f"seed_{seed}" / f"resume_{event:03d}_{arm}" / "STATE_IDENTITIES.json"
                bindings = json.loads(bindings_path.read_text()) if bindings_path.exists() else {}
                candidates = [old / "training_state.pt", *(old / "export").glob("*")]
                for path in candidates:
                    # The recovery validator checked these identities once. Do
                    # not give credit to extra files outside its sealed binding.
                    if str(path) in bindings and path.is_file() and not path.is_symlink():
                        unlinks[str(path)] = retirement
                # An existing latest alias disappears on the successor's atomic save.
                # If that save already completed, a stale alias cannot be credited.
                if (seed, event + 1, arm) not in completed_jobs:
                    latest[str(arm_root / "training_state.pt")] = (
                        str(old / "training_state.pt"), retirement)
    credits = []
    for row in inventory["inodes"]:
        aliases = row["aliases"]
        candidate_phases = [unlinks[path] for path in aliases if path in unlinks]
        reason = "No authorized existing rolling artifact"
        proof = {"bytes": 0, "reason": reason}
        if candidate_phases:
            if not row["same_filesystem"]:
                reason = "Other filesystem cannot supply current free capacity"
            elif not row["all_links_inventoried"]:
                reason = "Outside or unaccounted hard-link alias remains"
            else:
                remaining = []
                for path in aliases:
                    if path in unlinks:
                        continue
                    replacement = latest.get(path)
                    if replacement is None or replacement[0] not in aliases:
                        remaining.append(path)
                if remaining:
                    reason = "Inventoried alias is not scheduled for final unlink"
                else:
                    proof = _physical_reclamation(aliases[0], row)
                    reason = proof["reason"]
        row["retirement_credit_bytes"] = proof["bytes"]
        row["retirement_credit_reason"] = reason
    return credits


def remaining_campaign_budget(plan, campaign_root, boundary):
    """Simulate future writes at an already-validated complete recovery boundary.

    Boundary fields list completed (seed,event), (seed,event,arm), evaluation and
    retention identities. Partial scientific actions are rejected by the caller.
    No file is deleted, rewritten, transferred or assumed to become shorter.
    """
    root = Path(campaign_root).resolve()
    completed_pools = {tuple(row) for row in boundary["completed_pools"]}
    completed_jobs = {tuple(row) for row in boundary["completed_jobs"]}
    completed_evaluations = {tuple(row) for row in boundary["completed_evaluations"]}
    completed_retention = {tuple(row) for row in boundary["completed_retention"]}
    fresh = full_campaign_budget(plan)
    inventory = inventory_allocations(root)
    baseline = inventory["unique_allocated_bytes"]
    credits = _retirement_credits(root, inventory, completed_jobs, completed_retention)
    occupied = COMPILER_CACHE_ALLOWANCE  # Smaller remaining cache use is not proven.
    reclaimed = 0
    phases, new_jobs = [], set()

    def record(name, temporary=0):
        payload = occupied + temporary
        envelope = allocation_envelope(payload, reclaimed)
        phases.append({"phase": name, "new_live_payload_bytes": payload,
            "temporary_bytes": temporary,
            "slack_adjusted_new_bytes": required_bytes(payload) - FIXED_ALLOWANCE,
            "existing_live_allocated_bytes": baseline - reclaimed,
            "existing_reclaimed_bytes": reclaimed, "incremental_envelope_bytes": envelope,
            "required_bytes": FIXED_ALLOWANCE + envelope})

    record("resume_current")
    for seed in SEEDS:
        for arm in ARMS:
            if (seed, 1, arm) not in completed_jobs:
                record(f"seed_{seed}_{arm}_initial_pending", WEIGHTS)
                occupied += WEIGHTS
                record(f"seed_{seed}_{arm}_initial_saved")
        for window in plan["windows"]:
            event = window["event_index"]
            if (seed, event) not in completed_pools:
                if any(request["max_new_tokens"] != 2048 for request in window["requests"]):
                    raise ValueError("Resume must preserve the full 2048-token response cap")
                rows = sum(len(request["prompt_ids"]) + 2048 for request in window["requests"])
                pool_bytes = rows * FEATURE_ROW_BYTES
                occupied += pool_bytes
                record(f"seed_{seed}_event_{event}_capture")
                occupied += pool_bytes
                record(f"seed_{seed}_event_{event}_prepared")
            for arm in ARMS:
                job = (seed, event, arm)
                if job in completed_jobs:
                    continue
                record(f"seed_{seed}_event_{event}_{arm}_pending", 5 * WEIGHTS)
                occupied += 4 * WEIGHTS
                new_jobs.add(job)
                record(f"seed_{seed}_event_{event}_{arm}_saved")
            if event in (2, 4) and (seed, event) not in completed_retention:
                # Even when evaluation was completed before interruption, credit
                # waits until the still-pending authorized retirement phase.
                evaluation = "evaluated_before_retirement" if (seed, event) not in completed_evaluations else "prior_evaluation_before_retirement"
                record(f"seed_{seed}_event_{event}_{evaluation}")
                for arm in ARMS:
                    if (seed, event - 1, arm) in new_jobs:
                        occupied -= 4 * WEIGHTS
                reclaimed += sum(row["credit_bytes"] for row in credits
                                 if (row["seed"], row["after_event"]) == (seed, event))
                record(f"seed_{seed}_event_{event}_rolling_retired")
    peak = max(phases, key=lambda row: row["required_bytes"])
    return {"mode": "resume_remaining", "offload_assumed": False,
        "peak_required_bytes": peak["required_bytes"], "peak_phase": peak["phase"],
        "full_fresh_bound_bytes": fresh["peak_required_bytes"],
        "validated_boundary": boundary, "next_phase": boundary["next_phase"],
        "inventory": inventory, "current_baseline_allocated_bytes": baseline,
        "admitted_retirement_credits": credits,
        "existing_retirement_credit_policy": NO_EXISTING_RELEASE_PROOF,
        "final_new_live_payload_bytes": occupied,
        "compiler_cache_allowance_bytes": COMPILER_CACHE_ALLOWANCE,
        "compact_and_workspace_bytes": 4 * GIB, "reserve_bytes": 8 * GIB,
        "fixed_allowance_bytes": FIXED_ALLOWANCE, "slack_fraction": 0.10,
        "slack_order": "10% allowance on new live bytes BEFORE verified existing reclamation",
        "capture_and_prepared_copies": True, "max_new_tokens": 2048,
        "phases": phases}
