"""Four shared-feedback events followed by exactly one five-policy endpoint.

The parent runs no CUDA model. It owns one native serving process on GPU0 and
executes one isolated trainer at a time on GPU1. Every worker exits before any
endpoint timing. There is no automatic restart or efficacy-based early stop.
"""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

from online_feedback.storage import write_json
from experiments.ok_coding.campaign import WorkLedger, read_work, require_complete_work
from experiments.ok_coding.storage import identity
from experiments.ok_events_5_8.campaign import publish_checked, verify_pool
from experiments.full_k_gsm8k.campaign import online_record_check
from experiments.kplus_full_gsm8k.preflight import timing_barrier
from experiments.kw_v1.evaluate import append_json
from .contract import ARMS, BUDGET, PASS2, physical_training, validate_plan
from .storage import (
    admit,
    retire_redundant_features,
    retire_pool,
    retire_previous_export,
)
from .evaluation import evaluate, prepare_canonicals, audit_canonicals

COLLECTION_LIMITS = dict(feedback_generations=512, teacher_preparations=512)


def read(path):
    return json.loads(Path(path).read_text())


def resource_check(root, plan, phase, backend=None, timed=False):
    receipt = dict(phase=phase, storage=admit(root, plan["storage"]))
    if timed:
        receipt["quiescence"] = timing_barrier(backend.worker.pid)
    append_json(root / "campaign_001/RESOURCE_CHECKS.jsonl", receipt)
    return receipt


def collect(backend, root, event, requests, checkpoint, ledger, save_progress):
    """Prepare and verify each captured response once before retiring raw copies."""
    folder = root / f"campaign_001/events/event_{event:03d}"
    folder.mkdir(parents=True, exist_ok=False)
    records, preparations, paths = [], [], []
    costs = dict(
        collection_wall_s=0.0,
        teacher_preparation_s=0.0,
        feature_persistence_s=0.0,
        preparation_wall_s=0.0,
        retention_s=0.0,
    )
    started = time.perf_counter()
    for request in requests:
        began = time.perf_counter()
        record = ledger.call(
            "feedback_generations",
            backend.generate,
            request["prompt_ids"],
            request_id=request["request_id"],
            session_id=f"vfd_event_{event:03d}",
            request_index=request["request_index"],
            max_new_tokens=2048,
            purpose="serving",
            save_to=folder / "responses" / (request["request_id"] + ".json"),
        )
        costs["collection_wall_s"] += time.perf_counter() - began
        online_record_check(record, request, checkpoint, capture=True)
        if (
            record["record_id"] != request["request_id"]
            or len(record["prompt_ids"]) + len(record["response_ids"]) > 2272
        ):
            raise ValueError("Feedback identity or complete training context differs")
        records.append(record)
        began = time.perf_counter()
        prepared = ledger.call(
            "teacher_preparations",
            backend.prepare_feedback,
            record,
            folder / "loop/features" / (record["record_id"] + ".pt"),
        )
        costs["preparation_wall_s"] += time.perf_counter() - began
        preparations.append(prepared)
        write_json(folder / "PREPARATIONS.json", preparations)
        if prepared["status"] != "prepared":
            raise ValueError(
                "Ineligible feedback cannot be substituted or silently skipped"
            )
        costs["teacher_preparation_s"] += prepared["teacher_preparation_s"]
        costs["feature_persistence_s"] += prepared["feature_persistence_s"]
        paths.append(prepared["feature_path"])
        began = time.perf_counter()
        retire_redundant_features(record, prepared, root, folder)
        costs["retention_s"] += time.perf_counter() - began
        save_progress()
    pool = dict(
        event_id=f"event_{event:03d}",
        checkpoint=checkpoint,
        record_ids=[r["record_id"] for r in records],
        features=paths,
        identities={path: identity(path) for path in paths},
        immutable=True,
        teacher_terminal_forwards=128,
        response_generations=128,
    )
    costs.update(
        wall_s=time.perf_counter() - started,
        request_latency_s=sum(r["metadata"]["request_latency_s"] for r in records),
        capture_host_s=sum(r["metadata"]["capture"]["capture_host_s"] for r in records),
        cache_maintenance_s=sum(
            r["metadata"]["capture"]["cache_maintenance_s"] for r in records
        ),
    )
    write_json(folder / "POOL.json", pool)
    write_json(folder / "COSTS.json", costs)
    verify_pool(pool)
    return pool, costs


def execute_worker(root, plan, event, arm):
    folder = root / "campaign_001/worker_logs"
    folder.mkdir(exist_ok=True)
    label = f"event_{event:03d}_{arm}"
    command = [
        plan["training_python"],
        "-u",
        "-m",
        "experiments.vfd_second_pass.worker",
        str(root),
        str(event),
        arm,
    ]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=plan["training_gpu"])
    started = time.perf_counter()
    receipt = dict(command=command, status="starting", automatic_retry=False)
    with (folder / (label + ".log")).open("x") as output:
        write_json(folder / (label + ".json"), receipt)
        process = None
        try:
            process = subprocess.Popen(
                command,
                stdout=output,
                stderr=subprocess.STDOUT,
                env=environment,
                start_new_session=True,
            )
            receipt.update(pid=process.pid, status="running")
            write_json(folder / (label + ".json"), receipt)
            code = process.wait()
        except BaseException:
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            raise
        finally:
            code = process.returncode if process is not None else None
            receipt.update(
                exit_code=code,
                wall_s=time.perf_counter() - started,
                status="completed" if code == 0 else "failed",
            )
            write_json(folder / (label + ".json"), receipt)
    result_path = (
        root / "campaign_001/qualification/RESULT.json"
        if arm == "QUALIFICATION"
        else root / f"campaign_001/events/event_{event:03d}/{arm}/RESULT.json"
    )
    if code:
        raise RuntimeError(
            f'{label} stopped with exit {code}; retained log {folder/(label+".log")}'
        )
    result = read(result_path)
    if result["status"] != ("passed" if arm == "QUALIFICATION" else "completed"):
        raise ValueError("Worker exit and durable status disagree")
    if any(result["work"]["unknown"].values()):
        raise ValueError("Worker left unknown scientific work")
    return result


def accounting(root, plan):
    """Partial statuses never turn unknown work into completed work."""
    work = {
        key: 0
        for key in (
            "feedback_generations",
            "teacher_preparations",
            "optimizer_steps",
            "training_presentations",
            "scoring_forwards",
            "timed_generations",
            "canonical_generations",
        )
    }
    collection = read_work(
        root / "campaign_001/COLLECTION_WORK.jsonl", COLLECTION_LIMITS
    )
    work.update(collection["completed"])
    jobs, job_work = [], []
    # Launch receipts survive a worker killed before it can write RESULT.json.
    for launch_path in sorted((root / "campaign_001/worker_logs").glob("event_*.json")):
        launch = read(launch_path)
        event, arm = int(launch["command"][-2]), launch["command"][-1]
        if arm == "QUALIFICATION":
            continue
        presentations = (
            128 if arm == "SHARED" else (PASS2[arm] if event == 1 else 128 + PASS2[arm])
        )
        scores = (
            256
            if arm == "SHARED"
            else (0 if event == 1 else {"STATIC64": 128, "PROGRESS64": 256}.get(arm, 0))
        )
        limits = dict(
            training_presentations=presentations,
            backwards=presentations,
            optimizer_steps=presentations // 2,
            scoring_forwards=scores,
        )
        folder = root / f"campaign_001/events/event_{event:03d}/{arm}"
        measured = read_work(folder / "WORK.jsonl", limits)
        result_path = folder / "RESULT.json"
        job_work.append(
            dict(
                arm=arm,
                event=event,
                work=measured,
                launch=launch,
                result_available=result_path.exists(),
            )
        )
        for key in ("optimizer_steps", "training_presentations", "scoring_forwards"):
            work[key] += measured["completed"][key]
        if result_path.exists():
            result = read(result_path)
            if result["work"] != measured:
                raise ValueError("Worker result disagrees with original work ledger")
            jobs.append(result)
    for subdir, key, field in [
        ("evaluation/final", "timed_generations", "returned_generations"),
        ("canonicals", "canonical_generations", "returned_generations"),
    ]:
        path = root / "campaign_001" / subdir / "STATUS.json"
        if path.exists():
            work[key] = read(path)[field]
    qualification_path = root / "campaign_001/qualification/RESULT.json"
    qualification_limits = dict(
        training_presentations=4, backwards=4, optimizer_steps=2, scoring_forwards=8
    )
    qualification = read_work(
        qualification_path.parent / "WORK.jsonl", qualification_limits
    )
    logical = {
        arm: sum(row.get("logical_updater_s", 0.0) for row in jobs if row["arm"] == arm)
        for arm in ARMS
    }
    costs = dict(
        preparation_s=0.0,
        training_s=0.0,
        scoring_s=0.0,
        selection_s=0.0,
        training_compute_s=0.0,
        persistence_s=0.0,
        export_s=0.0,
        activation_s=0.0,
    )
    for path in (root / "campaign_001/events").glob("event_*/COSTS.json"):
        costs["preparation_s"] += read(path)["preparation_wall_s"]
    # Training/scoring below count actual execution, not five copies of SHARED.
    for result in jobs:
        event, arm = result["event"], result["arm"]
        passes = (
            [result.get("pass1")]
            if arm == "SHARED"
            else ([result.get("pass1")] if event > 1 else []) + [result.get("pass2")]
        )
        costs["training_s"] += sum(p["pass_wall_s"] for p in passes if p)
        costs["training_compute_s"] += sum(p["training_s"] for p in passes if p)
        costs["selection_s"] += result.get("physical_selection_s", 0.0)
        costs["scoring_s"] += result.get("physical_scoring_wall_s", 0.0)
        costs["persistence_s"] += result.get("state", {}).get("persistence_s", 0.0)
        costs["export_s"] += result.get("publication", {}).get("export_s", 0.0)
    return dict(
        work=work,
        qualification_work=(qualification or {}).get("completed", {}),
        physical_work_detail=dict(
            collection=collection, jobs=job_work, qualification=qualification
        ),
        logical_updater_seconds=logical,
        costs=costs,
        final_checkpoints=plan["final_checkpoints"],
    )


def run(root):
    from online_feedback.sglang_backend import SGLangDFlashBackend

    root = Path(root)
    plan = read(root / "sealed/PLAN.json")
    validate_plan(plan)
    base = root / "campaign_001"
    base.mkdir(exist_ok=False)
    ledger = WorkLedger(base / "COLLECTION_WORK.jsonl", COLLECTION_LIMITS)
    status = dict(
        status="running",
        phase="initialization",
        events_completed=0,
        automatic_retry=False,
        execution_source=plan["execution_source"],
    )
    started = time.perf_counter()
    backend = None
    activation_s = 0.0

    def save():
        status.update(accounting(root, plan))
        status["costs"]["activation_s"] = activation_s
        status["campaign_wall_s"] = time.perf_counter() - started
        write_json(base / "STATUS.json", status)

    try:
        resource_check(root, plan, "before_initialization")
        backend = SGLangDFlashBackend(
            plan["target"],
            plan["draft"],
            worker_python=plan["worker_python"],
            output_root=base / "native",
            serving_gpu=plan["native_gpu"],
            context_limit=4096,
            block_size=16,
        )
        status["backend_loading_s"] = backend.loading_s
        current_full = plan["draft"]
        previous_export = None
        save()
        for event in range(1, 5):
            status.update(current_event=event, phase="feedback_collection")
            save()
            resource_check(root, plan, f"before_event_{event}")
            pool, costs = collect(
                backend,
                root,
                event,
                plan["training_requests"][(event - 1) * 128 : event * 128],
                current_full,
                ledger,
                save,
            )
            folder = base / f"events/event_{event:03d}"
            if event == 1:
                status["phase"] = "bounded_qualification"
                save()
                qualification = execute_worker(root, plan, 1, "QUALIFICATION")
                require_complete_work(
                    qualification["work"],
                    dict(
                        training_presentations=4,
                        backwards=4,
                        optimizer_steps=2,
                        scoring_forwards=8,
                    ),
                )
                status["phase"] = "shared_first_pass"
                save()
                execute_worker(root, plan, 1, "SHARED")
            results = {}
            for arm in ARMS:
                status.update(phase="training", active_arm=arm)
                save()
                resource_check(root, plan, f"event_{event}_{arm}")
                results[arm] = execute_worker(root, plan, event, arm)
                save()
                if arm == "FULL":
                    # FULL is trained first. Its old export is no longer needed:
                    # feedback is immutable and all other policies use own states.
                    # Publish/retire now so only five final exports overlap.
                    current_full = results["FULL"]["publication"]["checkpoint_path"]
                    publication = publish_checked(
                        backend, current_full, folder / "PUBLICATION.json"
                    )
                    activation_s += publication["publication_and_validation_s"]
                    retire_previous_export(root, folder, previous_export, current_full)
                    previous_export = current_full
            retire_pool(root, folder, pool, results)
            status.update(
                events_completed=event, active_arm=None, phase="event_boundary"
            )
            save()
            expected = physical_training(event)
            if any(status["work"][key] != value for key, value in expected.items()):
                raise ValueError("Cumulative physical training budget differs")
            print(
                json.dumps(dict(event=event, status="completed", work=status["work"])),
                flush=True,
            )
        require_complete_work(
            read_work(base / "COLLECTION_WORK.jsonl", COLLECTION_LIMITS),
            COLLECTION_LIMITS,
        )
        if any(
            status["work"][key] != BUDGET[key]
            for key in ("optimizer_steps", "training_presentations", "scoring_forwards")
        ):
            raise ValueError(
                "Completed main training/scoring differs from fixed budget"
            )
        status["phase"] = "canonical_references"
        save()
        resource_check(root, plan, "before_references", backend, timed=True)
        # Use released weights for missing references; they are not extra arms.
        publication = publish_checked(
            backend, plan["draft"], base / "REFERENCE_PUBLICATION.json"
        )
        activation_s += publication["publication_and_validation_s"]
        audit = read(root / "sealed/REFERENCE_REUSE_AUDIT.json")
        canonicals = prepare_canonicals(
            backend,
            plan["evaluation_requests"],
            base / "canonicals",
            canonicals=read(root / "sealed/CANONICALS.json"),
            binding_digest=audit["binding_digest"],
        )
        checked = audit_canonicals(
            plan["evaluation_requests"],
            canonicals,
            binding_digest=audit["binding_digest"],
            evidence_root=root,
        )
        write_json(base / "CANONICAL_VALIDATION.json", checked)
        if checked["status"] != "passed":
            raise ValueError("Canonical source audit failed")
        status["phase"] = "final_timing"
        save()
        before = {
            arm: identity(Path(path) / "model.safetensors")
            for arm, path in plan["final_checkpoints"].items()
        }
        write_json(base / "FINAL_EXPORT_IDENTITIES.json", before)
        endpoint = evaluate(
            backend,
            plan["evaluation_requests"],
            plan["final_checkpoints"],
            base / "evaluation/final",
            canonicals=canonicals,
            boundary_check=lambda: resource_check(
                root, plan, "timed_block", backend, timed=True
            ),
        )
        activation_s += endpoint["activation_s"]
        status["endpoint_weight_validation_s"] = endpoint["weight_validation_s"]
        for arm, path in plan["final_checkpoints"].items():
            if identity(Path(path) / "model.safetensors") != before[arm]:
                raise ValueError("Endpoint checkpoint changed: " + arm)
        status.update(status="completed", phase="completed", evaluation=endpoint)
    except BaseException as error:
        status.update(
            status="failed", error=repr(error), traceback=traceback.format_exc()
        )
        raise
    finally:
        if backend is not None:
            try:
                backend.close()
            except BaseException as error:
                status.update(status="failed", cleanup_error=repr(error))
                save()
                raise
        save()
    return status


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    args = parser.parse_args()
    run(args.root)
