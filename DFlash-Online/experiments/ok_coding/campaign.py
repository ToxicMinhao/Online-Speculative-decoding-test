"""Fixed two-seed coding campaign; each ordinary training job is a fresh exec."""

import argparse
import gc
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from online_feedback.storage import write_json
from experiments.ok_events_5_8.campaign import collect_pool as native_collect_pool, identity, verify_pool
from experiments.kw_v1.evaluate import append_json
from .contract import ARMS, CELLS, NEW_BUDGET, SEEDS, training_contract, validate_plan

CACHE_VARIABLES = (
    "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "CUDA_CACHE_PATH",
    "TORCH_EXTENSIONS_DIR", "TVM_FFI_CACHE_DIR", "PYTORCH_KERNEL_CACHE_PATH",
)
TRAINING_LIMITS = {"forwards": 256, "backwards": 256, "optimizer_steps": 128}


def collect_pool(backend, window, checkpoint, output, progress, save_progress, storage):
    pool, costs = native_collect_pool(backend, window, checkpoint, output, progress, save_progress, storage)
    started = time.perf_counter()
    sources = {}
    for path in sorted((output / "responses").glob("*.json")):
        record = json.loads(path.read_text())
        capture = Path(record["metadata"]["capture"]["feature_path"])
        sources[str(path)] = identity(path)
        sources[str(capture)] = identity(capture)
    for path in sorted((output / "features").glob("*.terminal.json")):
        record = json.loads(path.read_text())
        tensor = Path(record["tensor_path"])
        sources[str(path)] = identity(path)
        sources[str(tensor)] = identity(tensor)
    pool["retained_source_identities"] = sources
    costs["source_retention_hash_s"] = time.perf_counter() - started
    write_json(output / "POOL.json", pool)
    write_json(output / "COSTS.json", costs)
    return pool, costs


class WorkLedger:
    """Durable starts retain unknown partial work; no action is retried."""

    def __init__(self, path, limits):
        self.path = Path(path)
        self.limits = dict(limits)
        if self.path.exists():
            raise FileExistsError("Existing work cannot be silently retried: " + str(path))
        self.attempts = {name: 0 for name in limits}
        self.completed = dict(self.attempts)

    def call(self, kind, action, *args, **kwargs):
        if self.attempts[kind] >= self.limits[kind]:
            raise RuntimeError("Authorized work ceiling reached: " + kind)
        ordinal = self.attempts[kind] + 1
        append_json(self.path, {"kind": kind, "phase": "started", "ordinal": ordinal})
        self.attempts[kind] = ordinal
        result = action(*args, **kwargs)
        append_json(self.path, {"kind": kind, "phase": "completed", "ordinal": ordinal})
        self.completed[kind] += 1
        return result

    def wrap(self, owner, name, kind):
        action = getattr(owner, name)

        def counted(*args, **kwargs):
            return self.call(kind, action, *args, **kwargs)

        setattr(owner, name, counted)


def read_work(path, limits):
    starts = {name: set() for name in limits}
    completed = {name: set() for name in limits}
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            row = json.loads(line)
            kind, ordinal = row["kind"], row["ordinal"]
            if kind not in starts or type(ordinal) is not int or not 1 <= ordinal <= limits[kind]:
                raise ValueError("Work ledger exceeds its sealed action budget")
            if row["phase"] == "started":
                if ordinal in starts[kind] or ordinal != len(starts[kind]) + 1:
                    raise ValueError("Duplicate or out-of-order action start")
                starts[kind].add(ordinal)
            elif row["phase"] == "completed":
                if ordinal not in starts[kind] or ordinal in completed[kind]:
                    raise ValueError("Work completion lacks exactly one preceding start")
                completed[kind].add(ordinal)
            else:
                raise ValueError("Unknown work phase")
    return {
        "attempts": {name: len(values) for name, values in starts.items()},
        "completed": {name: len(values) for name, values in completed.items()},
        "unknown": {name: len(starts[name] - completed[name]) for name in starts},
    }


def require_complete_work(work, expected):
    if (work["attempts"] != expected or work["completed"] != expected
            or work["unknown"] != {name: 0 for name in expected}):
        raise ValueError("Incomplete or unknown work cannot be declared completed")


def fresh_worker_environment(cache_root, environment=None):
    """Model caches remain pinned; all training compiler/kernel caches start empty."""
    environment = dict(os.environ if environment is None else environment)
    visible = environment.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible:
        raise ValueError("Each training exec requires exactly one CUDA-visible updater GPU")
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=False)
    for name in CACHE_VARIABLES:
        path = cache_root / name.lower()
        path.mkdir()
        environment[name] = str(path.resolve())
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def startup_metadata():
    import torch
    import torch._dynamo.config as dynamo_config

    if torch.cuda.is_initialized():
        raise ValueError("Training worker must enter with a fresh CUDA runtime")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible:
        raise ValueError("Exactly one CUDA-visible device is required")
    caches = {}
    for name in CACHE_VARIABLES:
        path = Path(os.environ[name])
        if not path.is_dir() or any(path.iterdir()):
            raise ValueError("Training cache did not start empty: " + str(path))
        caches[name] = {"path": str(path), "initially_empty": True}
    return {"pid": os.getpid(), "parent_pid": os.getppid(), "python": sys.executable,
            "cuda_visible_devices": visible, "cuda_initialized_on_entry": False,
            "caches": caches, "torch_version": torch.__version__,
            "torch_git_version": torch.version.git_version,
            "assume_static_by_default": dynamo_config.assume_static_by_default,
            "automatic_dynamic_shapes": dynamo_config.automatic_dynamic_shapes,
            "TORCH_LOGS": os.environ.get("TORCH_LOGS"), "warmup_model_work": 0}


def train_worker(root, seed, arm, event):
    """One ordinary event from its own complete boundary; no serving in this exec."""
    import torch
    from .training import (CodingTrainer, compare_initial_states, load_snapshot,
                           validate_event_receipt, verify_live_resume, verify_saved_boundary)

    root = Path(root)
    plan = json.loads((root / "sealed" / "PLAN.json").read_text())
    validate_plan(plan)
    if seed not in SEEDS or arm not in ARMS or event not in (1, 2, 3, 4):
        raise ValueError("Training job is outside the fixed campaign")
    seed_root = root / "campaign_001" / f"seed_{seed}"
    audit = seed_root / f"resume_{event:03d}_{arm}"
    job = json.loads((audit / "JOB.json").read_text())
    pool = json.loads((seed_root / f"event_{event:03d}" / "POOL.json").read_text())
    verify_pool(pool)
    contract = training_contract(plan, seed)
    config = plan["configs"][str(seed)]
    trainer = None
    started = time.perf_counter()
    receipt = {"status": "running", "seed": seed, "arm": arm, "event": event,
               "automatic_retry": False, "qualification_model_work": 0}
    primary_error = None
    ledger = WorkLedger(audit / "WORK.jsonl", TRAINING_LIMITS)
    try:
        receipt["startup"] = startup_metadata()
        write_json(audit / "WORKER.json", receipt)
        state = None
        if event > 1:
            parent = job["parent"]
            state, saved_check = load_snapshot(
                parent["state"], parent["export"], arm=arm, seed=seed,
                campaign_contract=contract, config_text=Path(config).read_text(), completed_event=event - 1)
            saved_check["state_identity"] = identity(parent["state"])
            write_json(audit / "SAVED.json", saved_check)
            used = {feature["record_id"] for old in state["events"] for feature in old["features"]}
            if used & set(pool["record_ids"]):
                raise ValueError("New pool overlaps this trajectory's own prior history")
        trainer = CodingTrainer(config, seed_root / arm, arm=arm, seed=seed,
            campaign_contract=contract, resume_state=job["parent"]["state"] if event > 1 else None,
            completed_event=event - 1)
        if event == 1:
            initial = trainer.save_initial_state(seed_root / arm / "initial_state.pt")
            write_json(audit / "INITIAL.json", initial)
            if arm == "K":
                parity = compare_initial_states(seed_root / "O" / "initial_state.pt", initial["path"],
                                                seed=seed, campaign_contract=contract)
                write_json(audit / "INITIAL_PARITY.json", parity)
        else:
            verify_live_resume(trainer, state, receipt_path=audit / "LIVE.json")
        del state
        loading_s = time.perf_counter() - started
        receipt["startup"]["device_name"] = torch.cuda.get_device_name(0)
        receipt["startup"]["device_uuid"] = str(torch.cuda.get_device_properties(0).uuid)
        loading_peak = {"allocated": torch.cuda.max_memory_allocated(),
                        "reserved": torch.cuda.max_memory_reserved()}
        torch.cuda.reset_peak_memory_stats()
        # Install observers only after the exact ready-to-train boundary checks.
        # Native forward, backward and Adam arithmetic are called exactly once.
        ledger.wrap(trainer.core.strategy, "forward_loss", "forwards")
        ledger.wrap(trainer.backend, "backward", "backwards")
        ledger.wrap(trainer.backend.optimizer.optimizer, "step", "optimizer_steps")
        ordinary = json.loads(Path(job["ordinary"]).read_text()) if arm == "K" else None
        event_started = time.perf_counter()
        result = trainer.train_event(f"event_{event:03d}", pool["features"], passes=2,
                                     presentation_schedule=ordinary)
        event_wall_s = time.perf_counter() - event_started
        require_complete_work(read_work(audit / "WORK.jsonl", TRAINING_LIMITS), TRAINING_LIMITS)
        check = validate_event_receipt(result, event_number=event,
            objective="osd" if arm == "O" else "osd_first_error",
            reference_schedule=ordinary, trainer_status=trainer.status())
        write_json(audit / "EVENT_CHECK.json", check)
        retained = trainer.events_root / f"event_{event:03d}" / "training_state.pt"
        os.link(trainer.latest, retained)
        result.update(latest_state_path=str(trainer.latest), full_state_path=str(retained),
                      state_retention="Immutable event hard link; latest is atomically replaced")
        write_json(retained.parent / "RESULT.json", result)
        validation_started = time.perf_counter()
        saved, saved_check = load_snapshot(retained, result["export_path"], arm=arm, seed=seed,
            campaign_contract=contract, config_text=Path(config).read_text(), completed_event=event)
        write_json(audit / "SAVED_ENDPOINT.json", saved_check)
        verify_saved_boundary(trainer, saved, audit / "LIVE_BOUNDARY.json")
        del saved
        bindings = {str(retained): identity(retained)}
        bindings.update({str(path): identity(path) for path in sorted(Path(result["export_path"]).iterdir())
                         if path.is_file()})
        write_json(audit / "STATE_IDENTITIES.json", bindings)
        costs = {"loading_initialization_resume_validation_s": loading_s,
                 "event_wall_s": event_wall_s, **result["costs"],
                 "objective_work": result["objective_work"],
                 "saved_state_validation_and_hash_s": time.perf_counter() - validation_started,
                 "loading_peak_allocated_bytes": loading_peak["allocated"],
                 "loading_peak_reserved_bytes": loading_peak["reserved"],
                 "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                 "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                 "peak_scope": "After loading: ordinary training, persistence and boundary validation",
                 "compilation_scope": "First ordinary work includes cold compilation; no separate warmup",
                 "checkpoint_recomputation": "Native backward work included in ordinary update timing"}
        write_json(audit / "COSTS.json", costs)
        receipt.update(status="completed", result_path=str(retained.parent / "RESULT.json"))
    except BaseException as error:
        primary_error = error
        receipt.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        cleanup_error = None
        if torch.cuda.is_initialized():
            receipt["observed_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            receipt["observed_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        if trainer is not None:
            receipt["trainer_status"] = trainer.status()
            try:
                trainer.close()
            except BaseException as error:
                receipt.update(status="failed", cleanup_error=repr(error))
                cleanup_error = error
            write_json(audit / "CLOSED.json", trainer.status())
        receipt["work"] = read_work(audit / "WORK.jsonl", TRAINING_LIMITS)
        receipt["worker_wall_s"] = time.perf_counter() - started
        write_json(audit / "WORKER.json", receipt)
        gc.collect()
        if cleanup_error is not None and primary_error is None:
            raise cleanup_error
    return receipt


def dispatch_training(root, seed, arm, event, parent, ordinary):
    seed_root = Path(root) / "campaign_001" / f"seed_{seed}"
    audit = seed_root / f"resume_{event:03d}_{arm}"
    audit.mkdir(exist_ok=False)
    environment = fresh_worker_environment(audit / "caches")
    write_json(audit / "JOB.json", {"seed": seed, "arm": arm, "event": event,
                                   "parent": parent, "ordinary": ordinary})
    command = [sys.executable, "-u", "-m", "experiments.ok_coding.campaign", str(root),
               "--train-worker", "--seed", str(seed), "--arm", arm, "--event", str(event)]
    started = time.perf_counter()
    with (audit / "WORKER.log").open("w") as stream:
        process = subprocess.run(command, env=environment, stdout=stream, stderr=subprocess.STDOUT)
    write_json(audit / "EXEC.json", {"command": command, "returncode": process.returncode,
                                    "wall_s": time.perf_counter() - started, "automatic_retry": False})
    if process.returncode != 0:
        raise RuntimeError("Fresh training worker failed; no retry: " + str(audit))
    worker = json.loads((audit / "WORKER.json").read_text())
    if worker["status"] != "completed":
        raise ValueError("Successful process exit lacks complete scientific boundary")
    require_complete_work(read_work(audit / "WORK.jsonl", TRAINING_LIMITS), TRAINING_LIMITS)
    result = json.loads(Path(worker["result_path"]).read_text())
    return result, json.loads((audit / "COSTS.json").read_text())


class FeedbackWork:
    """Observe qualified collection/preparation without changing its native path."""

    def __init__(self, backend, ledger):
        self.backend = backend
        self.ledger = ledger

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def generate(self, *args, **kwargs):
        return self.ledger.call("feedback", self.backend.generate, *args, **kwargs)

    def prepare_feedback(self, *args, **kwargs):
        return self.ledger.call("terminal_teacher_forwards", self.backend.prepare_feedback, *args, **kwargs)


def reconcile_work(output, progress):
    limits = {**TRAINING_LIMITS, "feedback": 128, "terminal_teacher_forwards": 128}
    total = {phase: {kind: 0 for kind in limits} for phase in ("attempts", "completed", "unknown")}
    arm_counters, arm_presentations, evaluations = {}, {}, {}
    for seed in SEEDS:
        for arm in ARMS:
            key = f"{seed}/{arm}"
            arm_counters[key], arm_presentations[key] = 0, 0
        for event in range(1, 5):
            seed_root = Path(output) / f"seed_{seed}"
            pool_work = read_work(seed_root / f"event_{event:03d}_WORK.jsonl",
                                  {"feedback": 128, "terminal_teacher_forwards": 128})
            for phase, values in pool_work.items():
                for kind, count in values.items():
                    total[phase][kind] += count
            for arm in ARMS:
                audit = seed_root / f"resume_{event:03d}_{arm}"
                work = read_work(audit / "WORK.jsonl", TRAINING_LIMITS)
                for phase, values in work.items():
                    for kind, count in values.items():
                        total[phase][kind] += count
                closed = audit / "CLOSED.json"
                if closed.exists():
                    value = json.loads(closed.read_text())
                    arm_counters[f"{seed}/{arm}"] = value["global_step"]
                    arm_presentations[f"{seed}/{arm}"] = value["presentations"]
            path = seed_root / f"event_{event:03d}" / "evaluation" / "STATUS.json"
            if path.exists():
                evaluations[f"{seed}/{event}"] = json.loads(path.read_text())
    progress["work"] = total
    progress["arm_counters"] = arm_counters
    progress["arm_presentations"] = arm_presentations
    progress["optimizer_steps"] = total["completed"]["optimizer_steps"]
    progress["presentations"] = sum(arm_presentations.values())
    progress["feedback_attempted"] = total["attempts"]["feedback"]
    progress["feedback_returned"] = total["completed"]["feedback"]
    progress["terminal_teacher_forwards"] = total["completed"]["terminal_teacher_forwards"]
    progress["evaluations"] = evaluations
    for suffix, key in (("attempted", "attempted_generations"), ("returned", "returned_generations"),
                        ("valid", "completed_valid_generations"), ("unknown", "unknown_generation_attempts"),
                        ("invalid", "invalid_generations"), ("pending", "pending_canonical_generations")):
        progress["evaluation_" + suffix] = sum(row.get(key, 0) for row in evaluations.values())


def require_complete_campaign(progress):
    require_complete_work(progress["work"], {
        "forwards": 4096, "backwards": 4096, "optimizer_steps": 2048,
        "feedback": 1024, "terminal_teacher_forwards": 1024})
    expected_steps = {f"{seed}/{arm}": 512 for seed in SEEDS for arm in ARMS}
    expected_presentations = {key: 1024 for key in expected_steps}
    if progress["arm_counters"] != expected_steps or progress["arm_presentations"] != expected_presentations:
        raise ValueError("All four own trajectories must reach step512/presentation1024")
    if any(progress["evaluation_" + key] != expected for key, expected in {
            "attempted": 4608, "returned": 4608, "valid": 4608,
            "unknown": 0, "invalid": 0, "pending": 0}.items()):
        raise ValueError("All4608 evaluations must complete and pass")
    if list(progress["evaluations"]) != [f"{seed}/{event}" for seed, event in CELLS]:
        raise ValueError("The four fixed cells did not complete in order")


def completed_job(seed_root, arm, event):
    """Reuse one fully closed job; never infer completion from a saved export alone."""
    from .training import validate_event_receipt

    audit = seed_root / f"resume_{event:03d}_{arm}"
    worker = json.loads((audit / "WORKER.json").read_text())
    if worker["status"] != "completed":
        raise ValueError("Incomplete training job cannot be retried: " + str(audit))
    require_complete_work(read_work(audit / "WORK.jsonl", TRAINING_LIMITS), TRAINING_LIMITS)
    result = json.loads(Path(worker["result_path"]).read_text())
    ordinary = None
    if arm == "K":
        ordinary = json.loads((seed_root / "O" / "events" / f"event_{event:03d}" / "RESULT.json").read_text())
    validate_event_receipt(result, event_number=event, objective="osd" if arm == "O" else "osd_first_error",
                           reference_schedule=ordinary)
    saved = json.loads((audit / "SAVED_ENDPOINT.json").read_text())
    live = json.loads((audit / "LIVE_BOUNDARY.json").read_text())
    closed = json.loads((audit / "CLOSED.json").read_text())
    execution = json.loads((audit / "EXEC.json").read_text())
    if (saved["status"] != "passed" or not saved["export_exact"] or not live["exact_boundary"]
            or saved["completed_event"] != event or saved["global_step"] != event * 128
            or closed["global_step"] != event * 128 or closed["presentations"] != event * 256
            or execution["returncode"] != 0):
        raise ValueError("Completed job lacks exact saved/live/closed boundary evidence")
    return result, json.loads((audit / "COSTS.json").read_text())


def admit_recovery(root, plan, *, write_receipt=True):
    from .recovery import inspect_boundary, record_admission

    boundary = inspect_boundary(root, plan)
    return record_admission(root, boundary) if write_receipt else boundary


def run(root, *, resume=False):
    from online_feedback.sglang_backend import SGLangDFlashBackend
    from .evaluation import evaluate
    from .storage import StorageLifecycle

    root = Path(root).resolve()
    plan = json.loads((root / "sealed" / "PLAN.json").read_text())
    checked_plan = validate_plan(plan)
    output = root / "campaign_001"
    from .preflight import verify
    from .attempts import reserve_directory
    admission = verify(root, resume=resume)
    output.mkdir(exist_ok=resume)
    if not resume:
        write_json(output / "PLAN_CHECK.json", checked_plan)
    started = time.perf_counter()
    backend = None
    costs = (json.loads((output / "COSTS.json").read_text()) if resume
             else {"execution_source": plan["execution_source"]})
    progress = {"status": "running", "phase": "loading_native", "events_completed": [],
                "feedback_attempted": 0, "feedback_returned": 0, "active_feedback": None,
                "automatic_retry": False, "extra_generations": 0, "optimizer_previews": 0,
                "qualification_model_work": 0, "extra_optimizer_steps": 0,
                "explicit_boundary_recovery": bool(resume)}

    def save():
        write_json(output / "STATUS.json", progress)

    # Reconstruct counters before recording the new attempt; do not reset science.
    reconcile_work(output, progress)
    save()
    attempt = reserve_directory(output / "native_attempts")
    native_path = attempt / "native"
    attempt_status = {
        "attempt_id": attempt.name, "native_path": str(native_path),
        "mode": "resume" if resume else "fresh",
        "execution_source": plan["execution_source"],
        "admission_receipts": [admission["receipt_path"]],
        "startup": {"status": "starting"}, "closure": {"status": "pending"},
    }
    if resume:
        attempt_status["admission_receipts"].append(admission["recovery_boundary"]["receipt_path"])
    write_json(attempt / "STATUS.json", attempt_status)
    try:
        backend = SGLangDFlashBackend(plan["target"], plan["draft"], worker_python=plan["worker_python"],
            output_root=native_path, serving_gpu="0", context_limit=4096, block_size=16)
        attempt_status["startup"] = {"status": "passed", "loading_s": backend.loading_s}
        write_json(attempt / "STATUS.json", attempt_status)
        costs.setdefault("backend_loading_attempts_s", []).append(backend.loading_s)
        costs["backend_loading_s"] = sum(costs["backend_loading_attempts_s"])
        for seed in SEEDS:
            seed_root = output / f"seed_{seed}"
            seed_root.mkdir(exist_ok=resume)
            storage = StorageLifecycle(plan, seed_root, seed=seed)
            parents = {arm: {"state": None, "export": plan["draft"], "receipt": None} for arm in ARMS}
            for window in plan["windows"]:
                event = window["event_index"]
                progress.update(seed=seed, event=event, phase="collecting_feedback")
                save()
                pool_path = seed_root / window["event_id"] / "POOL.json"
                if resume and pool_path.exists():
                    pool = json.loads(pool_path.read_text())
                    pool_costs = json.loads((pool_path.parent / "COSTS.json").read_text())
                    if (pool["checkpoint"] != parents["O"]["export"]
                            or pool["record_ids"] != [request["request_id"] for request in window["requests"]]):
                        raise ValueError("Recovered feedback differs from the sealed source trajectory")
                else:
                    ledger = WorkLedger(seed_root / f"event_{event:03d}_WORK.jsonl",
                                        {"feedback": 128, "terminal_teacher_forwards": 128})
                    pool, pool_costs = collect_pool(FeedbackWork(backend, ledger), window, parents["O"]["export"],
                        seed_root / window["event_id"], progress, save, storage)
                    require_complete_work(read_work(ledger.path, ledger.limits), ledger.limits)
                costs[f"seed_{seed}_event_{event:03d}_feedback"] = pool_costs
                ordinary = None
                for arm in ARMS:
                    progress["phase"] = "training_" + arm
                    save()
                    verify_pool(pool)
                    audit = seed_root / f"resume_{event:03d}_{arm}"
                    if resume and audit.exists():
                        result, training_costs = completed_job(seed_root, arm, event)
                    else:
                        storage.admit_training(arm, event, pool, parents)
                        result, training_costs = dispatch_training(root, seed, arm, event, parents[arm], ordinary)
                    verify_pool(pool)
                    result_path = seed_root / arm / "events" / f"event_{event:03d}" / "RESULT.json"
                    if arm == "O":
                        ordinary = str(result_path)
                    parents[arm] = {"state": result["full_state_path"], "export": result["export_path"],
                                    "receipt": str(result_path)}
                    costs[f"seed_{seed}_event_{event:03d}_{arm}"] = training_costs
                    reconcile_work(output, progress)
                    write_json(output / "COSTS.json", costs)
                    save()
                progress["events_completed"].append([seed, event])
                if (seed, event) in CELLS:
                    progress["phase"] = "evaluation"
                    save()
                    evaluation_root = seed_root / window["event_id"] / "evaluation"
                    if resume and evaluation_root.exists():
                        status = json.loads((evaluation_root / "STATUS.json").read_text())
                    else:
                        status = evaluate(backend, plan["evaluation_requests"],
                            {"F": plan["draft"], **{arm: parents[arm]["export"] for arm in ARMS}},
                            evaluation_root, seed=seed, event=event,
                            canonical_path=output / "EVALUATION_CANONICAL.json")
                    costs[f"seed_{seed}_event_{event:03d}_evaluation"] = status
                progress["phase"] = "retention"
                save()
                retained = seed_root / "retention" / f"event_{event:03d}_RETENTION.json"
                if not (resume and retained.exists()):
                    storage.after_event(event, parents, pool, evaluated=event in (2, 4))
                write_json(seed_root / "CURRENT_PARENTS.json", parents)
            storage.finalize(parents)
        reconcile_work(output, progress)
        require_complete_campaign(progress)
        progress.update(status="completed", phase="finished")
    except BaseException as error:
        progress.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        attempt_status["failure"] = repr(error)
        if backend is None:
            attempt_status["startup"] = {"status": "failed", "error": repr(error)}
        write_json(attempt / "STATUS.json", attempt_status)
        raise
    finally:
        cleanup_error = None
        attempt_status["closure"] = {"status": "no_backend" if backend is None else "closed"}
        if backend is not None:
            try:
                backend.close()
            except BaseException as error:
                attempt_status["closure"] = {"status": "failed", "error": repr(error)}
                progress["cleanup_error"] = repr(error)
                if progress["status"] != "failed":
                    progress.update(status="failed", error=repr(error))
                    cleanup_error = error
        write_json(attempt / "STATUS.json", attempt_status)
        reconcile_work(output, progress)
        costs["campaign_wall_s"] = costs.get("campaign_wall_s", 0.0) + time.perf_counter() - started
        costs["accounting"] = (
            "Feedback and terminal preparation are shared within each seed/event. Ordinary training runs in16 "
            "fresh interpreters with empty compiler caches. Cold compilation and checkpoint recomputation "
            "are included in ordinary update time. Nested capture/preparation/persistence timers are not additive. "
            "Serving is sequential with training and retention; no training-inclusive deployment speedup is measured.")
        write_json(output / "COSTS.json", costs)
        attempt_status["campaign_status"] = progress["status"]
        write_json(attempt / "STATUS.json", attempt_status)
        save()
        if cleanup_error is not None:
            raise cleanup_error
    return progress


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("--train-worker", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--event", type=int, choices=(1, 2, 3, 4))
    args = parser.parse_args()

    def interrupted(signum, frame):
        raise InterruptedError("Supervisor signal " + str(signum))

    signal.signal(signal.SIGTERM, interrupted)
    if args.train_worker:
        train_worker(args.root, args.seed, args.arm, args.event)
    else:
        run(args.root, resume=args.resume)
