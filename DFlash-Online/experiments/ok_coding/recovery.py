"""Validate an exact completed-work boundary before any recovery admission."""

import json
from pathlib import Path

from online_feedback.storage import write_json
from .attempts import reserve_receipt
from .contract import ARMS, SEEDS, validate_plan
from .storage import identity


def read_json(path):
    return json.loads(Path(path).read_text())


def check_retirement(seed_root, seed, event):
    folder = seed_root / "retention"
    stem = f"event_{event:03d}_"
    final = folder / (stem + "RETENTION.json")
    intent = folder / (stem + "RETIREMENT_INTENT.json")
    progress = folder / (stem + "RETIREMENT_PROGRESS.json")
    if (intent.exists() or progress.exists()) and not final.exists():
        raise ValueError("Incomplete retirement blocks recovery: " + str(final))
    if not final.exists():
        return False
    receipt = read_json(final)
    if (receipt["status"] != "completed" or receipt["seed"] != seed
            or receipt["event"] != event or not receipt["all_feedback_retained"]):
        raise ValueError("Invalid completed retention receipt: " + str(final))
    if intent.exists() and read_json(intent)["candidates"] != receipt["retired"]:
        raise ValueError("Retirement intent and completion differ: " + str(final))
    if progress.exists() and read_json(progress)["retired"] != receipt["retired"]:
        raise ValueError("Retirement progress and completion differ: " + str(final))
    return True


def check_endpoint_files(seed_root, arm, event, result):
    audit = seed_root / f"resume_{event:03d}_{arm}"
    bindings = read_json(audit / "STATE_IDENTITIES.json")
    state = seed_root / arm / "events" / f"event_{event:03d}" / "training_state.pt"
    export = state.parent / "export"
    if result["full_state_path"] != str(state) or result["export_path"] != str(export):
        raise ValueError("Recovered endpoint belongs to a different trajectory")
    if str(state) not in bindings or not any(Path(name).parent == export for name in bindings):
        raise ValueError("State/export identities are incomplete")
    existing_export = {str(path) for path in export.iterdir()} if export.exists() else set()
    if existing_export - bindings.keys():
        raise ValueError("Unbound file in recovered export: " + str(export))
    for name, expected in bindings.items():
        path = Path(name)
        if path != state and path.parent != export:
            raise ValueError("Unexpected endpoint identity path: " + name)
        if path.is_symlink():
            raise ValueError("Endpoint identity cannot be a symbolic link: " + name)
        if path.exists():
            if identity(path) != expected:
                raise ValueError("Recovered endpoint changed: " + name)
        elif event in (2, 4):
            raise ValueError("Required endpoint is missing: " + name)
        else:
            retired = read_json(seed_root / "retention" /
                                f"event_{event + 1:03d}_RETENTION.json")["retired"]
            if not any(row["path"] == name and row["identity"] == expected for row in retired):
                raise ValueError("Missing boundary lacks verified successor retirement: " + name)
    if event == 1:
        initial = read_json(audit / "INITIAL.json")
        path = seed_root / arm / "initial_state.pt"
        if initial["path"] != str(path) or identity(path) != {
                key: initial[key] for key in ("bytes", "sha256")}:
            raise ValueError("Required initial state changed: " + str(path))


def inspect_boundary(root, plan):
    """Read artifacts and action receipts, never infer completion from counters."""
    from .campaign import completed_job, read_work, require_complete_work, verify_pool
    from .report import audit_cell

    root = Path(root).resolve()
    output = root / "campaign_001"
    previous = read_json(output / "STATUS.json")
    if previous["status"] == "completed":
        raise ValueError("Campaign already completed; no resume is needed")
    if read_json(output / "PLAN_CHECK.json") != validate_plan(plan):
        raise ValueError("Recovery plan differs from the admitted original plan")
    boundary = {"status": "passed", "previous_status": previous,
                "completed_pools": [], "completed_jobs": [],
                "completed_evaluations": [], "completed_retention": [],
                "next_phase": None, "reused_complete_artifacts": []}

    def stage(name, complete):
        if not complete and boundary["next_phase"] is None:
            boundary["next_phase"] = name
        elif complete and boundary["next_phase"] is not None:
            raise ValueError("Completed work follows an unstarted phase: " + name)

    for seed in SEEDS:
        seed_root = output / f"seed_{seed}"
        # Check interrupted retirement even before any old file was unlinked.
        retained = {event: check_retirement(seed_root, seed, event) for event in range(1, 5)}
        previous_export = plan["draft"]
        latest_results = {}
        for window in plan["windows"]:
            event = window["event_index"]
            stem = f"seed_{seed}_event_{event}"
            event_root = seed_root / window["event_id"]
            work_path = seed_root / f"event_{event:03d}_WORK.jsonl"
            pool_done = event_root.exists() or work_path.exists()
            if pool_done:
                limits = {"feedback": 128, "terminal_teacher_forwards": 128}
                require_complete_work(read_work(work_path, limits), limits)
                pool = read_json(event_root / "POOL.json")
                verify_pool(pool)
                if (pool["checkpoint"] != previous_export
                        or pool["record_ids"] != [r["request_id"] for r in window["requests"]]
                        or pool["teacher_terminal_forwards"] != 128):
                    raise ValueError("Recovered pool differs from sealed feedback trajectory")
                sources = pool.get("retained_source_identities")
                if not sources:
                    raise ValueError("Native capture/terminal identities missing")
                for name, expected in sources.items():
                    if identity(name) != expected:
                        raise ValueError("Retained native feedback changed: " + name)
                read_json(event_root / "COSTS.json")
                boundary["completed_pools"].append([seed, event])
                boundary["reused_complete_artifacts"].append(str(event_root / "POOL.json"))
            stage(stem + "_capture", pool_done)
            for arm in ARMS:
                audit = seed_root / f"resume_{event:03d}_{arm}"
                done = audit.exists()
                endpoint = seed_root / arm / "events" / window["event_id"]
                if not done and endpoint.exists():
                    raise ValueError("Endpoint without completed job evidence: " + str(endpoint))
                if done:
                    result, _ = completed_job(seed_root, arm, event)
                    check_endpoint_files(seed_root, arm, event, result)
                    latest_results[arm] = result
                    boundary["completed_jobs"].append([seed, event, arm])
                    boundary["reused_complete_artifacts"].append(str(audit / "WORKER.json"))
                    if arm == "O":
                        previous_export = result["export_path"]
                stage(stem + "_" + arm + "_pending", done)
            if event in (2, 4):
                evaluation = event_root / "evaluation"
                if evaluation.exists():
                    _, errors = audit_cell(root, plan, seed, event)
                    if errors:
                        raise ValueError("Incomplete/invalid evaluation cannot be retried: " + str(errors))
                    boundary["completed_evaluations"].append([seed, event])
                    boundary["reused_complete_artifacts"].append(str(evaluation / "STATUS.json"))
                stage(stem + "_evaluation", evaluation.exists())
            stage(stem + "_retention", retained[event])
            if retained[event]:
                boundary["completed_retention"].append([seed, event])
        for arm, result in latest_results.items():
            latest = seed_root / arm / "training_state.pt"
            state = Path(result["full_state_path"])
            if result["latest_state_path"] != str(latest) or latest.is_symlink():
                raise ValueError("Latest state path differs from its own trajectory")
            if not latest.samefile(state) and identity(latest) != identity(state):
                raise ValueError("Latest state differs from its completed endpoint")
    if boundary["next_phase"] is None:
        boundary["next_phase"] = "finalize"
    return boundary


def record_admission(root, boundary):
    path = reserve_receipt(Path(root) / "campaign_001" / "recovery", prefix="admission_")
    receipt = {**boundary, "automatic_retry": False, "additional_scientific_work": 0,
               "receipt_path": str(path)}
    write_json(path, receipt)
    return receipt
