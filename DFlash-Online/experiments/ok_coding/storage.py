"""Admit the complete two-seed campaign, then check each actual write phase.

The budget retains both copies of every feedback pool and all required endpoints.
Only this campaign's rolling event-1/3 boundaries may be retired here, after
verified successors and their evaluations. Historical artifacts are untouched.
"""
import json
from pathlib import Path
import shutil
import time

from online_feedback.storage import write_json
from experiments.ok_events_5_8.storage import (
    GIB, WEIGHTS, FEATURE_ROW_BYTES, identity, required_bytes,
)

SEEDS = (17, 29)
ARMS = ("O", "K")
COMPILER_CACHE_ALLOWANCE = 16 * GIB


def full_campaign_budget(plan):
    """Maximum-length admission with atomic-save overlap; no transfer assumption."""
    occupied = COMPILER_CACHE_ALLOWANCE
    phases = []

    def record(name, temporary=0):
        payload = occupied + temporary
        phases.append({"phase": name, "payload_bytes": payload,
                       "required_bytes": required_bytes(payload)})

    # Four true event-zero states contain weights but no Adam moments. Retain
    # them throughout. Initial snapshots need an additional atomic-save slot.
    for seed in SEEDS:
        for arm in ARMS:
            record(f"seed_{seed}_{arm}_initial_pending", WEIGHTS)
            occupied += WEIGHTS
            record(f"seed_{seed}_{arm}_initial_saved")
        for window in plan["windows"]:
            event = window["event_index"]
            rows = sum(len(r["prompt_ids"]) + r["max_new_tokens"]
                       for r in window["requests"])
            pool_bytes = rows * FEATURE_ROW_BYTES
            occupied += pool_bytes
            record(f"seed_{seed}_event_{event}_capture")
            occupied += pool_bytes
            record(f"seed_{seed}_event_{event}_prepared")
            for arm in ARMS:
                # 3W full Adam state, W export, W transient export workspace.
                record(f"seed_{seed}_event_{event}_{arm}_pending", 5 * WEIGHTS)
                occupied += 4 * WEIGHTS
                record(f"seed_{seed}_event_{event}_{arm}_saved")
            if event in (2, 4):
                record(f"seed_{seed}_event_{event}_evaluated_before_retirement")
                occupied -= len(ARMS) * 4 * WEIGHTS
                record(f"seed_{seed}_event_{event}_rolling_retired")
    peak = max(phases, key=lambda row: row["required_bytes"])
    return {
        "mode": "retain_local", "offload_assumed": False,
        "peak_required_bytes": peak["required_bytes"], "peak_phase": peak["phase"],
        "final_incremental_payload_bytes": occupied,
        "capture_and_prepared_copies": True, "feedback_pools": 8,
        "retained_endpoint_pairs": 8, "retained_initial_states": 4,
        "compiler_cache_allowance_bytes": COMPILER_CACHE_ALLOWANCE,
        "compact_and_workspace_bytes": 4 * GIB, "reserve_bytes": 8 * GIB,
        "slack_fraction": 0.10, "phases": phases,
    }


class StorageLifecycle:
    def __init__(self, plan, seed_root, *, seed):
        self.plan = plan
        self.output = Path(seed_root).resolve()
        self.seed = seed
        self.root = self.output / "retention"
        self.root.mkdir(exist_ok=True)
        settings = plan["storage"]
        if (seed not in SEEDS or settings["mode"] != "retain_local"
                or settings["offload_assumed"] is not False
                or settings["reserve_bytes"] != 8 * GIB):
            raise ValueError("Require declared local retention and unchanged reserve")

    def admit(self, phase, payload, basis):
        free = shutil.disk_usage(self.output).free
        row = {"phase": phase, "seed": self.seed, "payload_bytes": payload,
               "required_bytes": required_bytes(payload), "free_bytes": free,
               "reserve_bytes": 8 * GIB, "basis": basis,
               "admitted": free >= required_bytes(payload)}
        write_json(self.root / (phase + "_STORAGE.json"), row)
        if not row["admitted"]:
            raise RuntimeError("Insufficient peak storage: " + phase)
        return row

    def admit_collection(self, window):
        rows = sum(len(r["prompt_ids"]) + r["max_new_tokens"]
                   for r in window["requests"])
        return self.admit(f"event_{window['event_index']:03d}_collection",
                          rows * FEATURE_ROW_BYTES,
                          "All 128 cap-length captures; previous pools stay retained")

    def admit_preparation(self, window, records):
        rows = sum(len(r["prompt_ids"]) + len(r["response_ids"]) for r in records)
        return self.admit(f"event_{window['event_index']:03d}_preparation",
                          rows * FEATURE_ROW_BYTES,
                          "Prepared features coexist with original captures")

    def admit_training(self, arm, event, pool, parents):
        extra_initial = WEIGHTS if event == 1 else 0
        return self.admit(f"event_{event:03d}_{arm}_training",
                          5 * WEIGHTS + extra_initial,
                          "Pending complete state, export workspace and initial snapshot")

    def after_event(self, event, parents, pool, *, evaluated):
        if evaluated != (event in (2, 4)):
            raise ValueError("Retention must follow the fixed endpoint evaluation")
        started = time.perf_counter()
        before = shutil.disk_usage(self.output).free
        for arm in ARMS:
            audit = self.output / f"resume_{event:03d}_{arm}"
            endpoint = json.loads((audit / "SAVED_ENDPOINT.json").read_text())
            live = json.loads((audit / "LIVE_BOUNDARY.json").read_text())
            if (endpoint["status"] != "passed" or not endpoint["export_exact"]
                    or not live["exact_boundary"]
                    or endpoint["completed_event"] != event
                    or endpoint["global_step"] != event * 128):
                raise ValueError("Own successor validation missing; retain rolling state")
            bindings = json.loads((audit / "STATE_IDENTITIES.json").read_text())
            for name, expected in bindings.items():
                if identity(name) != expected:
                    raise ValueError("Successor identity changed: " + name)
        retired = []
        if event in (2, 4):
            previous = event - 1
            candidates = []
            for arm in ARMS:
                old = self.output / arm / "events" / f"event_{previous:03d}"
                bindings = json.loads((self.output / f"resume_{previous:03d}_{arm}"
                                       / "STATE_IDENTITIES.json").read_text())
                for path in [old / "training_state.pt", *(old / "export").glob("*")]:
                    if not path.is_file():
                        continue
                    if (path.is_symlink() or not path.resolve().is_relative_to(self.output)
                            or identity(path) != bindings.get(str(path))):
                        raise ValueError("Rolling state identity differs: " + str(path))
                    candidates.append({"path": str(path), "identity": bindings[str(path)],
                        "reason": "Verified own successor and required evaluation completed",
                        "retained_successor": parents[arm]["state"],
                        "lost_capability": f"Exact event-{previous} optimizer resume/serving"})
            write_json(self.root / f"event_{event:03d}_RETIREMENT_INTENT.json",
                       {"before_free_bytes": before, "candidates": candidates})
            for row in candidates:
                Path(row["path"]).unlink()
                retired.append(row)
                write_json(self.root / f"event_{event:03d}_RETIREMENT_PROGRESS.json",
                           {"retired": retired})
        receipt = {"status": "completed", "event": event, "seed": self.seed,
                   "retired": retired, "before_free_bytes": before,
                   "free_bytes": shutil.disk_usage(self.output).free,
                   "all_feedback_retained": True, "wall_s": time.perf_counter() - started}
        write_json(self.root / f"event_{event:03d}_RETENTION.json", receipt)
        return receipt

    def finalize(self, parents):
        from experiments.ok_events_5_8.campaign import verify_pool

        retained = {}
        for event in (2, 4):
            for arm in ARMS:
                path = self.output / f"resume_{event:03d}_{arm}" / "STATE_IDENTITIES.json"
                for name, expected in json.loads(path.read_text()).items():
                    if identity(name) != expected:
                        raise ValueError("Required retained endpoint changed: " + name)
                    retained[name] = expected
        for arm in ARMS:
            initial = self.output / f"resume_001_{arm}" / "INITIAL.json"
            receipt = json.loads(initial.read_text())
            expected = {key: receipt[key] for key in ("bytes", "sha256")}
            if identity(receipt["path"]) != expected:
                raise ValueError("Required true initial state changed: " + arm)
            retained[receipt["path"]] = expected
        pool_bindings = {}
        for event in range(1, 5):
            pool = json.loads((self.output / f"event_{event:03d}" / "POOL.json").read_text())
            verify_pool(pool)
            sources = pool.get("retained_source_identities")
            if not sources:
                raise ValueError("Native capture/terminal source identities missing")
            for name, expected in sources.items():
                if identity(name) != expected:
                    raise ValueError("Retained native feedback changed: " + name)
            pool_bindings[str(event)] = {"features": pool["identities"], "native_sources": sources}
        write_json(self.root / "FINAL.json", {
            "status": "passed", "seed": self.seed, "required_events": [2, 4],
            "retained_files": retained, "latest_states": parents,
            "pool_bindings": pool_bindings, "offload_assumed": False})
