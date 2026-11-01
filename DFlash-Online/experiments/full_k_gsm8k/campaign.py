"""Run two complete, synchronous own-feedback streams, then the fixed final panel.

The existing controller owns serve/capture/prepare/train/publish ordering. This
wrapper adds explicit work accounting, initial/recovery boundaries, intermediate
evaluation and the requested two-arm horizon. Numerical training stays native.
"""
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
from experiments.ok_coding.campaign import WorkLedger, read_work
from experiments.ok_events_5_8.storage import identity
from experiments.osd_persistent.run import check_event, check_first_updated_request
from experiments.osd_persistent.retention import checked_retire
from experiments.o_half_matched_lr.archive import full_trace_errors
from .contract import ARMS, EVENTS, INTERMEDIATE_EVENTS, validate_plan
from .storage import admit, boundary_files


ARM_LIMITS = {
    "online_generations": 7473, "correctness_generations": 4,
    "intermediate_generations": 384, "terminal_preparation_forwards": 7424,
    "response_forwards": 14848, "response_backwards": 14848, "optimizer_steps": 7424,
}


def read(path):
    return json.loads(Path(path).read_text())


class ArmLedger(WorkLedger):
    """A restart may extend only a previously validated complete boundary."""

    def __init__(self, path, *, resume=False):
        if not resume:
            super().__init__(path, ARM_LIMITS)
            return
        self.path, self.limits = Path(path), dict(ARM_LIMITS)
        previous = read_work(path, ARM_LIMITS)
        if any(previous["unknown"].values()):
            raise ValueError("Unknown scientific work cannot be repeated on resume")
        self.attempts = previous["attempts"]
        self.completed = previous["completed"]


def expected_work(completed_events, tail=0):
    return {
        "online_generations": 128 * completed_events + tail,
        "correctness_generations": 4 if completed_events else 0,
        "intermediate_generations": 128 * sum(e <= completed_events for e in INTERMEDIATE_EVENTS),
        "terminal_preparation_forwards": 128 * completed_events,
        "response_forwards": 256 * completed_events,
        "response_backwards": 256 * completed_events,
        "optimizer_steps": 128 * completed_events,
    }


def validate_restart(root, plan, arm):
    """Validate chronology and exact saved state before capacity or model loading."""
    import torch
    from experiments.ok_coding.training import _export_equal
    from .training import validate_snapshot

    base = root / "campaign_001/arms" / arm
    progress = read(base / "STATUS.json")
    event = progress["events_completed"]
    if progress["phase"] not in {"idle_boundary", "tail", "completed"} or not 1 <= event <= EVENTS:
        raise ValueError("Resume requires a fully recorded event boundary; partial science remains stopped")
    if progress["arm"] != arm or progress["binding"] != plan["binding"]:
        raise ValueError("Resume arm or scientific binding differs")
    actual = read_work(base / "WORK.jsonl", ARM_LIMITS)
    expected = expected_work(event, progress["tail_completed"])
    if actual["completed"] != expected or actual["attempts"] != expected or any(actual["unknown"].values()):
        raise ValueError("Resume would repeat, omit or hide scientific work")
    initial = torch.load(base / "trainer/initial_state.pt", map_location="cpu", weights_only=False)
    boundary = read(base / "boundaries" / f"event_{event:03d}.json")
    state = torch.load(boundary["state"], map_location="cpu", weights_only=False)
    checked = validate_snapshot(state, arm=arm, binding=plan["binding"],
        config_text=Path(plan["configs"][arm]).read_text(),
        original_config_text=Path(plan["original_config"]).read_text(), completed_event=event,
        expected_contract=initial["algorithm_contract"], expected_recipe=initial["backend"]["optimizer"]["recipe"])
    if identity(boundary["state"]) != boundary["state_identity"]:
        raise ValueError("Own retained resume container changed")
    _export_equal(state, Path(boundary["weights"]).parent)
    if not os.path.samefile(base / "trainer/training_state.pt", boundary["state"]):
        raise ValueError("Latest state is not the validated own chronological boundary")
    for number, event_record in enumerate(state["events"], 1):
        result = read(base / "trainer/events" / f"event_{number:03d}/RESULT.json")
        if any(result.get(key) != value for key, value in event_record.items()):
            raise ValueError("Saved event history differs from its immutable result")
        marker = read(base / "boundaries" / f"event_{number:03d}.json")
        if marker["status"] != "passed" or marker["event"] != number:
            raise ValueError("Incomplete chronological boundary history")
    del initial, state
    gc.collect()
    return progress, {**checked, "work": actual, "state": boundary["state"], "export": str(Path(boundary["weights"]).parent)}


def save_boundary(root, arm, trainer, receipt):
    """Retain the actual committed inode before a later atomic replacement."""
    event = len(trainer.events)
    export = Path(receipt["training"]["checkpoint_path"])
    base = root / "campaign_001/arms" / arm
    marker = base / "boundaries" / f"event_{event:03d}.json"
    check = trainer.verify_boundary(export_path=export,
        receipt_path=base / "boundaries" / f"event_{event:03d}_validation.json")
    snapshot = export.parent / "training_state.pt"
    os.link(trainer.latest, snapshot)
    data = dict(status="passed", arm=arm, event=event, state=str(snapshot),
        weights=str(export / "model.safetensors"), state_identity=identity(snapshot),
        weights_identity=identity(export / "model.safetensors"), scientific_check=check)
    write_json(marker, data)
    return data


def retire_previous_state(root, arm, event):
    """Only a new run's superseded state is retired, after successor validation."""
    if event == 1:
        return
    base = root / "campaign_001/arms" / arm
    previous = read(base / "boundaries" / f"event_{event-1:03d}.json")
    successor = read(base / "boundaries" / f"event_{event:03d}.json")
    path = Path(previous["state"])
    if (successor["status"] != "passed" or path.resolve() != path
            or not path.is_relative_to(base) or identity(path) != previous["state_identity"]
            or path.stat().st_nlink != 1):
        raise ValueError("Superseded state is not an exclusive verified own predecessor")
    destination = base / "boundaries" / f"event_{event-1:03d}_retirement.json"
    row = dict(status="prepared", retired_state=str(path), identity=previous["state_identity"],
        successor=successor["state"], reason="Verified own successor and compact history retained",
        lost_capability="Exact prior-event optimizer resume; retired, not archived")
    write_json(destination, row)
    path.unlink()
    write_json(destination, {**row, "status": "completed"})


def counted_backend(backend, ledger, progress):
    generate, prepare = backend.generate, backend.prepare_feedback

    def generate_once(*args, **kwargs):
        kind = {"online": "online_generations", "correctness": "correctness_generations",
                "intermediate": "intermediate_generations"}[progress["generation_phase"]]
        return ledger.call(kind, generate, *args, **kwargs)

    def prepare_once(*args, **kwargs):
        return ledger.call("terminal_preparation_forwards", prepare, *args, **kwargs)

    backend.generate, backend.prepare_feedback = generate_once, prepare_once


def online_record_check(record, request, checkpoint, *, capture):
    if (record["prompt_ids"] != request["prompt_ids"] or record["checkpoint_id"] != str(checkpoint)
            or record["max_new_tokens"] != 2048
            or record["metadata"]["feedback_capture_in_timed_request"] is not capture
            or record["metadata"]["cache_policy"] != "idle_full_flush_before_every_request_and_publication"
            or full_trace_errors(record)):
        raise ValueError("Own-feedback native identity, capture or trace validation failed")


def run_arm(root, arm, *, resume=False):
    from online_feedback.controller import OnlineController
    from online_feedback.sglang_backend import SGLangDFlashBackend
    from .evaluation import evaluate_intermediate
    from .training import FullKTrainer, preserve_training_rng
    from .costs import begin_event_profile, finish_event_profile

    root = Path(root)
    plan = read(root / "sealed/PLAN.json")
    validate_plan(plan)
    base = root / "campaign_001/arms" / arm
    started = time.perf_counter()
    backend = trainer = None
    if resume:
        progress, recovery = validate_restart(root, plan, arm)
    else:
        base.mkdir(parents=True, exist_ok=False)
        recovery = None
        progress = dict(status="running", phase="initializing", arm=arm, binding=plan["binding"],
            events_completed=0, tail_completed=0, generation_phase="online", costs={}, event_costs=[],
            segments=[], automatic_retry=False, invalid_online_records=[])
    # Recovery validation deliberately precedes remaining-write admission.
    admit(root, plan["storage"], validated_tensor_paths=boundary_files(root / "campaign_001"),
          receipt_path=base / f"ADMISSION_{len(progress['segments']):03d}.json")
    write_json(base / "STATUS.json", progress)
    ledger = ArmLedger(base / "WORK.jsonl", resume=resume)
    manifest = read(plan["manifest"])
    canonicals = read(plan["canonicals"])
    requests = manifest["train_requests"]
    try:
        # Exclusive native child directories retain every initialization attempt.
        native_parent = base / f"native_attempt_{len(progress['segments'])+1:03d}"
        native_parent.mkdir(exist_ok=False)
        backend = SGLangDFlashBackend(plan["target"], plan["draft"], worker_python=plan["worker_python"],
            output_root=native_parent / "native", serving_gpu="0", context_limit=4096, block_size=16)
        counted_backend(backend, ledger, progress)
        load_started = time.perf_counter()
        trainer = FullKTrainer(plan["configs"][arm], base / "trainer", arm=arm, binding=plan["binding"],
            original_config_text=Path(plan["original_config"]).read_text(),
            resume_state=recovery["state"] if recovery else None,
            completed_event=progress["events_completed"])
        progress["segments"].append(dict(backend_loading_s=backend.loading_s,
            trainer_loading_s=time.perf_counter()-load_started, resume=resume,
            native_root=str(native_parent / "native"), pid=os.getpid()))
        if recovery is None:
            initial_started = time.perf_counter()
            trainer.save_initial(comparator_path=(root / "campaign_001/arms/K/trainer/initial_state.pt")
                                 if arm == "K_HALF" else None)
            progress["segments"][-1]["initial_validation_s"] = time.perf_counter()-initial_started
        else:
            backend.publish(recovery["export"])
        ledger.wrap(trainer.core.strategy, "forward_loss", "response_forwards")
        ledger.wrap(trainer.backend, "backward", "response_backwards")
        ledger.wrap(trainer.backend.optimizer.optimizer, "step", "optimizer_steps")
        previous_export = Path(recovery["export"]) if recovery else None
        for event in range(progress["events_completed"]+1, EVENTS+1):
            event_id = f"event_{event:03d}"
            event_root = base / "events" / event_id
            event_root.mkdir(parents=True, exist_ok=False)
            admit(root, plan["storage"], validated_tensor_paths=boundary_files(root / "campaign_001"),
                  receipt_path=event_root / "STORAGE_ADMISSION.json")
            progress.update(phase="collecting", generation_phase="online", current_event=event)
            write_json(base / "STATUS.json", progress)
            controller = OnlineController(backend, trainer, event_root / "loop")
            group = requests[(event-1)*128:event*128]
            records = []
            for request in group:
                record = controller.serve(request["prompt_ids"], request_id=request["request_id"],
                    session_id="gsm8k_train", request_index=request["request_index"], max_new_tokens=2048)
                online_record_check(record, request, previous_export or plan["draft"], capture=True)
                records.append(record)
                progress["online_generations"] = ledger.completed["online_generations"]
                write_json(base / "STATUS.json", progress)
            if event == 1:
                progress["generation_phase"] = "correctness"
                probe = records[0]
                replay = controller.evaluate(probe["prompt_ids"], canonical_response_ids=probe["response_ids"],
                    label="first_capture_disabled", max_new_tokens=2048)
                if (replay["record"]["finish_reason"] != probe["finish_reason"]
                        or replay["record"]["metadata"]["verification_calls"] != probe["metadata"]["verification_calls"]):
                    raise ValueError("Inherited capture-disabled correctness check differs")
            progress.update(phase="training", generation_phase="online")
            write_json(base / "STATUS.json", progress)
            compiler_before = begin_event_profile()
            try:
                receipt = controller.update(event_id)
            finally:
                write_json(event_root / "RESOURCE_COSTS.json", finish_event_profile(compiler_before))
            validation_started = time.perf_counter()
            check_event(receipt, trainer, backend, event_root, event, 128)
            boundary = save_boundary(root, arm, trainer, receipt)
            validation_s = time.perf_counter()-validation_started
            export = Path(receipt["training"]["checkpoint_path"])
            if event == 1:
                progress["generation_phase"] = "correctness"
                correctness = check_first_updated_request(backend, requests[128], plan["draft"], str(export), event_root)
                checked = [read(event_root / f"qualification_{name}.json") for name in ("updated", "frozen", "reloaded")]
                if len({record["finish_reason"] for record in checked}) != 1 or any(full_trace_errors(r) for r in checked):
                    raise ValueError("Inherited updated/frozen/reloaded trace or termination mismatch")
                write_json(event_root / "INHERITED_CORRECTNESS.json", {**correctness, "generations": 4,
                    "first_capture_disabled_in_loop": True, "disposable_optimizer_steps": 0})
            if event in INTERMEDIATE_EVENTS:
                progress.update(phase="intermediate", generation_phase="intermediate")
                write_json(base / "STATUS.json", progress)
                with preserve_training_rng(trainer):
                    evaluate_intermediate(backend, plan["evaluation_requests"], str(export),
                        base / "evaluation" / event_id, arm=arm, event=event, canonicals=canonicals)
                trainer.verify_boundary(export_path=export,
                    receipt_path=base / "evaluation" / event_id / "TRAINING_STATE_UNCHANGED.json")
            retention_started = time.perf_counter()
            checked_retire(event_root, records, receipt, previous_export, base)
            retire_previous_state(root, arm, event)
            costs = dict(read(event_root / f"loop/events/{event_id}.costs.json"),
                **receipt["training"]["costs"],
                teacher_preparation_s=sum(r["teacher_preparation_s"] for r in receipt["preparation"]),
                feature_persistence_s=sum(r["feature_persistence_s"] for r in receipt["preparation"]),
                activation=receipt["activation"], publication_validation_s=validation_s,
                retention_s=time.perf_counter()-retention_started)
            progress["event_costs"].append({"event": event, "costs": costs})
            progress.update(events_completed=event, phase="idle_boundary", generation_phase="online",
                final_checkpoint=str(export), boundary=boundary, actual_work=read_work(base / "WORK.jsonl", ARM_LIMITS))
            if progress["actual_work"]["completed"] != expected_work(event):
                raise ValueError("Complete event work accounting differs from its fixed budget")
            write_json(base / "STATUS.json", progress)
            previous_export = export
            print(json.dumps({"arm": arm, "event": event, "steps": trainer.global_step,
                              "online": ledger.completed["online_generations"]}), flush=True)
        progress.update(phase="tail", generation_phase="online")
        for request in requests[7424+progress["tail_completed"]:]:
            record = backend.generate(request["prompt_ids"], request_id=request["request_id"],
                session_id="gsm8k_tail", request_index=request["request_index"], max_new_tokens=2048,
                purpose="evaluation", save_to=base / "tail" / (request["request_id"]+".json"))
            online_record_check(record, request, previous_export, capture=False)
            progress["tail_completed"] += 1
            write_json(base / "STATUS.json", progress)
        final_check = trainer.verify_boundary(export_path=previous_export, receipt_path=base / "FINAL_STATE_CHECK.json")
        progress.update(status="completed", phase="completed", final_state_check=final_check,
            actual_work=read_work(base / "WORK.jsonl", ARM_LIMITS))
        if progress["actual_work"]["completed"] != ARM_LIMITS or progress["actual_work"]["attempts"] != ARM_LIMITS:
            raise ValueError("Full arm accounting does not match the declared stream budget")
    except BaseException as error:
        progress.update(status="failed", error=repr(error), traceback=traceback.format_exc(),
                        actual_work=read_work(base / "WORK.jsonl", ARM_LIMITS))
        raise
    finally:
        if trainer is not None:
            trainer.close()
        if backend is not None:
            backend.close()
        progress.setdefault("worker_segments_s", []).append(time.perf_counter()-started)
        write_json(base / "STATUS.json", progress)


def run(root, *, resume=False):
    """Arm workers exit before final timed evaluation; no background training."""
    root = Path(root)
    plan = read(root / "sealed/PLAN.json")
    validate_plan(plan)
    campaign = root / "campaign_001"
    campaign.mkdir(exist_ok=resume)
    started = time.perf_counter()
    status = dict(status="running", execution_source=plan["execution_source"], completed_arms=[], automatic_retry=False)
    try:
        for arm in ARMS:
            base = campaign / "arms" / arm
            prior = read(base / "STATUS.json") if (base / "STATUS.json").exists() else None
            if prior is not None:
                if not resume:
                    raise ValueError("Existing scientific work requires explicit complete-boundary resume")
                validate_restart(root, plan, arm)
                if prior["status"] == "completed":
                    status["completed_arms"].append(arm)
                    continue
            admit(root, plan["storage"], validated_tensor_paths=boundary_files(campaign),
                  receipt_path=campaign / f"ADMISSION_{arm}.json")
            command = [plan["training_python"], "-u", "-m", "experiments.full_k_gsm8k.campaign",
                       str(root), "--arm", arm]
            if prior is not None:
                command.append("--resume")
            log_path = campaign / f"{arm}_{time.time_ns()}.log"
            with log_path.open("w") as log:
                completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            receipt = dict(arm=arm, command=command, log=str(log_path), returncode=completed.returncode)
            write_json(campaign / f"EXEC_{arm}.json", receipt)
            if completed.returncode:
                raise RuntimeError("Own-feedback arm failed; see " + str(log_path))
            status["completed_arms"].append(arm)
            write_json(campaign / "STATUS.json", status)
        from .audit import audit_training
        audit_training(root)
        final = campaign / "evaluation/final"
        if final.exists():
            if not resume or read(final / "STATUS.json")["status"] != "completed":
                raise ValueError("Partial final evaluation is preserved and cannot be silently repeated")
        else:
            from online_feedback.sglang_backend import SGLangDFlashBackend
            from .evaluation import evaluate_final
            admit(root, plan["storage"], validated_tensor_paths=boundary_files(campaign),
                  receipt_path=campaign / "ADMISSION_FINAL.json")
            backend = SGLangDFlashBackend(plan["target"], plan["draft"], worker_python=plan["worker_python"],
                output_root=campaign / "final_native_attempt_001/native", serving_gpu="0", context_limit=4096, block_size=16)
            try:
                write_json(campaign / "FINAL_LOADING.json", {"native_loading_s": backend.loading_s})
                evaluate_final(backend, plan["evaluation_requests"], plan["checkpoints"], final,
                               canonicals=read(plan["canonicals"]))
            finally:
                backend.close()
        status["status"] = "completed"
    except BaseException as error:
        status.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        status["campaign_wall_s"] = time.perf_counter()-started
        write_json(campaign / "STATUS.json", status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    # The enclosing deadline must unwind the worker so its owned native
    # subprocess is closed and partial work is recorded before process exit.
    def terminate(_signum, _frame):
        raise SystemExit(128 + signal.SIGTERM)

    signal.signal(signal.SIGTERM, terminate)
    if args.arm:
        run_arm(args.root, args.arm, resume=args.resume)
    else:
        run(args.root, resume=args.resume)


if __name__ == "__main__":
    main()
