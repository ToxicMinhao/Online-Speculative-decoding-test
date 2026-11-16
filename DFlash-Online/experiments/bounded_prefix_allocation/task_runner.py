"""One candidate-chain task, with the shared native training and serving paths.

C7, its own feedback collection, C8 and final evaluation run serially. Completed
C7/C8 recoveries and exports are retained. This entry point never retries work,
trains a control, creates a reference, or changes the fixed objective.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import resource
import signal
import time
import traceback

from experiments.persistent_prefix_withholding.task_runner import create_backend, close_backend
from experiments.persistent_prefix_withholding.qualification_entry import HostMemorySample
from online_feedback.storage import write_json
from .runtime import WorkLedger, boundary, exclusive, occupied_card, read

TASKS = ("C7", "C_collect", "C8", "final")
METHODS = ("N", "C", "S")
SESSION = "bounded_prefix_allocation_event8"


def endpoint(plan, event):
    """Read a complete candidate boundary, never substitute a control state."""
    if event not in (7, 8):
        raise ValueError("Candidate endpoint must be C7 or C8")
    result = read(Path(plan["root"]) / "tasks" / f"C{event}" / "RESULT.json")
    if (result["status"] != "completed" or result["arm"] != "C" or result["event"] != event
            or result["binding"] != plan["binding"] or result["execution_commit"] != plan["execution_commit"]):
        raise ValueError("Missing or foreign complete candidate endpoint")
    saved = result["endpoint"]
    expected_state = Path(plan["root"]) / "tasks" / f"C{event}" / "trainer/training_state.pt"
    if (Path(saved["state"]) != expected_state or saved["global_step"] != event*128
            or saved["presentations"] != event*256):
        raise ValueError("Candidate endpoint path or cumulative counters differ")
    return saved


def own_features(plan, origin):
    """Require all own C7 packets before constructing the C8 trainer."""
    base = Path(plan["root"]) / "tasks/C_collect"
    receipt = read(base / "FEATURES.json")
    requests = read(plan["event8_inputs"])["requests"]
    expected = [str(base / "loop/features" / (request["request_id"] + ".pt")) for request in requests]
    if (receipt["status"] != "completed" or receipt["arm"] != "C"
            or receipt["binding"] != plan["binding"]
            or receipt["own_checkpoint"] != origin["export"]
            or receipt["own_weights_identity"] != origin["weights_identity"]
            or receipt["generated_responses"] != 128 or receipt["teacher_preparations"] != 128
            or len(expected) != 128 or receipt["feature_paths"] != expected):
        raise ValueError("C8 requires its complete immutable C7 feedback pool")
    return expected


def train_task(plan, task, base, ledger):
    from .training import CandidateTrainer
    from experiments.prefix_guard.qualify import pin_retained_graph_runtime
    from experiments.functional_step_geometry.native_pilot import discarded_startup
    from experiments.ok_coding.storage import identity
    from online_feedback.rng import preserve_rng

    event = int(task[1:])
    if task not in ("C7", "C8"):
        raise ValueError("Only the two candidate training events are authorized")
    origin = dict(state=plan["origin_state"], export=plan["origin_export"]) if event == 7 else endpoint(plan, 7)
    features = plan["event7_feature_paths"] if event == 7 else own_features(plan, origin)
    if len(features) != 128 or len(set(features)) != 128:
        raise ValueError("Require exactly 128 distinct prepared response packets")
    trainer = None
    publication = exclusive(Path(plan["root"]) / "PUBLICATION.lock")
    publication_held = False
    result = dict(task=task, arm="C", event=event, status="running", binding=plan["binding"],
        execution_commit=plan["execution_commit"], origin=origin, candidate=plan["candidate"])
    began = time.perf_counter()
    try:
        result["runtime_compatibility"] = pin_retained_graph_runtime(base)
        trainer = ledger.call({"training_initialization": 1}, "trainer_initialization", CandidateTrainer,
            plan["configs"]["C"], base / "trainer", binding=plan["binding"],
            origin_state=origin["state"], origin_export=origin["export"],
            original_config_text=Path(plan["origin_config"]).read_text(), completed_event=event-1,
            schedule=read(plan["event7_schedule"]) if event == 7 else None)
        result["loading_s"] = time.perf_counter() - began
        # Install counters before disposable startup too: a failed partial
        # startup preserves its actual attempted calls rather than assuming two.
        ledger.wrap(trainer.strategy, "forward_loss", "response_forward")
        ledger.wrap(trainer.backend, "backward", "response_backward")
        ledger.wrap(trainer.backend.optimizer.optimizer, "step", "adam")
        result["startup"] = discarded_startup(trainer, features, base / "startup")
        ordinary_passes = trainer._train_passes

        def passes_then_publication(*args, **kwargs):
            nonlocal publication_held
            rows = ordinary_passes(*args, **kwargs)
            wait_started = time.perf_counter()
            publication.__enter__()
            publication_held = True
            result["publication_wait_s"] = time.perf_counter()-wait_started
            boundary(plan, upcoming_bytes=plan["storage"]["atomic_publication_bytes"])
            return rows

        trainer._train_passes = passes_then_publication
        started = time.perf_counter()
        receipt = trainer.train_event(f"event_{event:03d}", features, 2)
        result["event_wall_s"] = time.perf_counter()-started
        result["boundary"] = trainer.verify_boundary(receipt["checkpoint_path"], base / "BOUNDARY.json")
        if (trainer.global_step != event*128 or trainer.presentations != event*256
                or trainer.core.accumulation_remainder):
            raise ValueError("Candidate endpoint lost its complete own counters")
        with preserve_rng():
            state_identity = identity(trainer.latest)
            weights_identity = identity(Path(receipt["checkpoint_path"]) / "model.safetensors")
        result.update(status="completed", endpoint=dict(state=str(trainer.latest),
            export=receipt["checkpoint_path"], state_identity=state_identity, weights_identity=weights_identity,
            global_step=trainer.global_step, presentations=trainer.presentations),
            costs=receipt["costs"], physical_work=dict(trainer.work),
            retained_complete_candidate_boundaries=list(range(7, event+1)))
        write_json(base / "RESULT.json", result)
        return result
    except BaseException as error:
        result.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        write_json(base / "FAILED.json", result)
        if trainer is not None:
            trainer.failed = True
        raise
    finally:
        if publication_held:
            publication.__exit__(None, None, None)
        if trainer is not None:
            import torch
            write_json(base / "MEMORY.json", dict(
                peak_gpu_allocated_bytes=int(torch.cuda.max_memory_allocated()),
                peak_gpu_reserved_bytes=int(torch.cuda.max_memory_reserved()),
                host_maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                scope="Task including model load, startup, training and publication"))
            trainer.close()


def collect_task(plan, task, base, ledger, uuid):
    from experiments.full_k_gsm8k.campaign import online_record_check
    from experiments.kplus_full_gsm8k.storage import retire_redundant_features

    if task != "C_collect":
        raise ValueError("Only C7 own-feedback collection is authorized")
    origin = endpoint(plan, 7)
    requests = read(plan["event8_inputs"])["requests"]
    if len(requests) != 128 or len({row["request_id"] for row in requests}) != 128:
        raise ValueError("Require the complete sealed 128-request event-eight order")
    backend, packets = None, []
    result = dict(status="preparing", arm="C", binding=plan["binding"],
        execution_commit=plan["execution_commit"], own_checkpoint=origin["export"],
        own_weights_identity=origin["weights_identity"], generated_responses=0,
        teacher_preparations=0, feature_paths=packets, generation_s=0., preparation_s=0.,
        redundant_capture_retirement_s=0.)
    (base / "loop/features").mkdir(parents=True)
    (base / "records").mkdir()
    write_json(base / "FEATURES.json", result)
    try:
        backend = create_backend(plan, base, origin["export"], uuid)
        result["loading_s"] = backend.loading_s
        for request in requests:
            boundary(plan, upcoming_bytes=plan["storage"]["temporary_feature_bytes"])
            started = time.perf_counter()
            record = ledger.call({"feedback_generation": 1}, request["request_id"], backend.generate,
                request["prompt_ids"], max_new_tokens=2048, request_id=request["request_id"],
                session_id="bounded_prefix_allocation_C_event8", request_index=request["request_index"],
                purpose="serving", save_to=base / "records" / (request["request_id"] + ".json"))
            result["generation_s"] += time.perf_counter()-started
            result["generated_responses"] += 1
            # The native backend saves the raw record before this validation.
            write_json(base / "FEATURES.json", result)
            online_record_check(record, request, origin["export"], capture=True)
            packet = base / "loop/features" / (request["request_id"] + ".pt")
            started = time.perf_counter()
            prepared = ledger.call({"teacher_preparation": 1}, request["request_id"], backend.prepare_feedback, record, packet)
            result["preparation_s"] += time.perf_counter()-started
            result["teacher_preparations"] += 1
            write_json(base / "FEATURES.json", result)
            if prepared.get("status") != "prepared":
                raise ValueError("Own response has no eligible prepared packet; no replacement allowed")
            started = time.perf_counter()
            # Deletes only decoded-byte-equivalent new capture/terminal copies;
            # compact provenance and the durable own training packet remain.
            retire_redundant_features(record, prepared, plan["root"], base)
            result["redundant_capture_retirement_s"] += time.perf_counter()-started
            packets.append(str(packet))
            write_json(base / "FEATURES.json", result)
        result["status"] = "completed"
        write_json(base / "FEATURES.json", result)
        return result
    except BaseException as error:
        result.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        write_json(base / "FEATURES.json", result)
        raise
    finally:
        close_backend(backend, base)


def final_task(plan, base, ledger, uuid):
    from experiments.persistent_prefix_withholding.evaluation import evaluate_final, evaluation_schedule
    from experiments.kplus_full_gsm8k.preflight import timing_barrier
    from .analysis import write_report

    panel = read(plan["data_panel"])
    requests, permutations = panel["requests"], panel["within_block_permutations"]
    checkpoints = {"C": endpoint(plan, 8)["export"],
                   **{arm: plan["controls"][arm]["export"] for arm in ("N", "S")}}
    backend = None
    try:
        backend = create_backend(plan, base, checkpoints["N"], uuid)
        generate = backend.generate

        def counted(*args, **kwargs):
            kind = "warmup_generation" if kwargs["request_id"].startswith("warmup_") else "scored_generation"
            return ledger.call({kind: 1}, kwargs["request_id"], generate, *args, **kwargs)

        backend.generate = counted

        def quiet_boundary():
            timing_barrier(backend.worker.pid)
            boundary(plan)
            return dict(status="passed", physical_gpu_uuid=uuid, both_cards_checked=True,
                other_gpu_idle=True, host_quiet=True, storage_admitted=True, time_admitted=True)

        evaluation_root = base / "evaluation"
        result = evaluate_final(backend, requests, checkpoints, evaluation_root,
            canonicals=read(plan["references"]), warmup_requests=read(plan["warmups"])["requests"],
            warmup_canonicals=read(plan["warmup_references"]),
            schedule=evaluation_schedule(requests, permutations, methods=METHODS),
            within_block_permutations=permutations, physical_gpu_uuid=uuid,
            boundary_check=quiet_boundary, execution_commit=plan["execution_commit"],
            methods=METHODS, session=SESSION)
        summary = write_report(evaluation_root)
        if summary["status"] != "completed":
            raise ValueError("Final raw-record audit is invalid or incomplete")
        return dict(status="completed", evaluation=result, summary=str(evaluation_root / "SUMMARY.json"),
                    loading_s=backend.loading_s, checkpoints=checkpoints)
    finally:
        close_backend(backend, base)


def run_task(plan_path, task, uuid):
    plan = read(plan_path)
    if task not in TASKS or uuid not in plan["devices"].values() or os.environ.get("CUDA_VISIBLE_DEVICES") != uuid:
        raise ValueError("Require one declared task on exactly its sealed physical GPU")
    if task == "final" and uuid != plan["devices"]["A"]:
        raise ValueError("Final timing must remain on the fixed physical card A")
    base = Path(plan["root"]) / "tasks" / task
    base.mkdir(parents=True, exist_ok=False)
    result = dict(status="running", task=task, physical_gpu_uuid=uuid,
        execution_commit=plan["execution_commit"], binding=plan["binding"],
        started_epoch=time.time(), automatic_retry=False)
    write_json(base / "STATUS.json", result)
    ledger = WorkLedger(plan["root"], task)
    monitor = HostMemorySample()

    def interrupt(signum, frame):
        raise InterruptedError("Bounded candidate task interrupted; evidence preserved")

    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, interrupt)
    try:
        result["initial_resource_boundary"] = boundary(plan, compiler=True)
        with occupied_card(plan["root"], task, uuid), monitor:
            if task in ("C7", "C8"):
                receipt = train_task(plan, task, base, ledger)
            elif task == "C_collect":
                receipt = collect_task(plan, task, base, ledger, uuid)
            else:
                receipt = final_task(plan, base, ledger, uuid)
            result["final_resource_boundary"] = boundary(plan, compiler=True)
        write_json(base / "RESULT.json", receipt)
        result.update(status="completed", result=str(base / "RESULT.json"))
    except BaseException as error:
        result.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        result.update(wall_s=time.time()-result["started_epoch"],
            host_maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            sampled_process_tree_peak_rss_bytes=monitor.maximum_bytes)
        write_json(base / "STATUS.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--gpu-uuid", required=True)
    args = parser.parse_args()
    run_task(args.plan, args.task, args.gpu_uuid)


if __name__ == "__main__":
    main()
