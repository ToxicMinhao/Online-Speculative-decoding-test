"""One bounded, independent single-card production task.

The supervisor chooses dependencies and a physical card. This process owns only
that task's runtime; every scientific operation is reserved in the shared ledger
before execution. No failed task is automatically restarted.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import time
import traceback

from online_feedback.storage import write_json
from .resources import ALLOCATION_BYTES, CARD_SECONDS, ELAPSED_SECONDS, FREE_RESERVE_BYTES, LIMITS, effective_limits, effective_elapsed_limit


def read(path):
    return json.loads(Path(path).read_text())


@contextmanager
def exclusive(path):
    """Cross-process serialization for small ledgers or heavyweight publication."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class SharedWork:
    def __init__(self, root, task):
        self.root, self.task = Path(root), task

    def reserve(self, counts, label):
        with exclusive(self.root / "WORK.lock"):
            state = read(self.root / "WORK.json")
            limits = dict(LIMITS)
            if state.get("operational_amendment") is not None:
                plan = read(self.root / "sealed/PLAN.json")
                limits = effective_limits(plan)
                if (state["operational_amendment"] != plan["operational_amendment"]
                        or state.get("limits") != limits
                        or state.get("scientific_binding") != plan["binding"]):
                    raise ValueError("Shared ledger differs from authenticated operational limits")
            elif (self.root / "sealed/PLAN.json").exists():
                plan = read(self.root / "sealed/PLAN.json")
                if plan.get("operational_amendment") is not None:
                    raise ValueError("Amended production ledger is missing its limit binding")
            for kind, amount in counts.items():
                if kind not in limits or type(amount) is not int or amount < 0:
                    raise ValueError("Unknown or negative scientific work reservation")
                if state["attempted"][kind] + amount > limits[kind]:
                    raise RuntimeError("Shared scientific work cap exceeded: " + kind)
            for kind, amount in counts.items():
                state["attempted"][kind] += amount
            number = state["next_operation"]
            state["next_operation"] += 1
            state["pending"][str(number)] = dict(task=self.task, label=label, counts=counts)
            write_json(self.root / "WORK.json", state)
        return number

    def complete(self, number):
        with exclusive(self.root / "WORK.lock"):
            state = read(self.root / "WORK.json")
            entry = state["pending"].pop(str(number))
            for kind, amount in entry["counts"].items():
                state["completed"][kind] += amount
            write_json(self.root / "WORK.json", state)

    def call(self, counts, label, function, *args, **kwargs):
        number = self.reserve(counts, label)
        result = function(*args, **kwargs)
        self.complete(number)
        return result

    def wrap(self, owner, name, kind):
        original = getattr(owner, name)
        def counted(*args, **kwargs):
            return self.call({kind: 1}, name, original, *args, **kwargs)
        setattr(owner, name, counted)


def clock_status(root, now=None):
    state = read(Path(root) / "CLOCK.json")
    now = time.time() if now is None else now
    active = sum(now - value["started_epoch"] for value in state["active"].values())
    return dict(occupied_s=state["completed_card_s"] + active,
                elapsed_s=now - state["first_model_load_epoch"], active=state["active"])


@contextmanager
def occupied_card(root, task, uuid):
    root = Path(root)
    with exclusive(root / "CLOCK.lock"):
        state = read(root / "CLOCK.json")
        if task in state["active"] or any(v["physical_gpu_uuid"] == uuid for v in state["active"].values()):
            raise RuntimeError("A scientific worker already owns this physical card")
        state["active"][task] = dict(physical_gpu_uuid=uuid, started_epoch=time.time(), pid=os.getpid())
        write_json(root / "CLOCK.json", state)
    try:
        yield
    finally:
        with exclusive(root / "CLOCK.lock"):
            state = read(root / "CLOCK.json")
            lease = state["active"].pop(task)
            elapsed = time.time() - lease["started_epoch"]
            state["completed_card_s"] += elapsed
            state["finished"].append(dict(task=task, occupied_card_s=elapsed, **lease))
            write_json(root / "CLOCK.json", state)


def compiler_cache_growth(plan):
    """Charge only cache growth since the pre-GPU baseline, once per task."""
    root = Path(plan["compiler_root"])
    before = int(plan["compiler_baseline_allocated_bytes"])
    reserve = int(plan["compiler_growth_reserved_bytes"])
    if before < 0 or reserve < 0:
        raise ValueError("Compiler baseline and growth allowance must be nonnegative")
    allocated = int(subprocess.check_output(["du", "-s", "--block-size=1", str(root)], text=True).split()[0]) if root.exists() else 0
    growth = max(0, allocated-before)
    receipt = dict(root=str(root), baseline_allocated_bytes=before, allocated_bytes=allocated,
                   growth_bytes=growth, reserved_growth_bytes=reserve)
    if growth > reserve:
        raise RuntimeError("Compiler-cache growth exceeded its fixed reservation: " + repr(receipt))
    return receipt


def _authenticated_source_links(plan):
    """Read exact authored links from current and explicitly pinned prior sources."""
    current = dict(source_root=plan["source_root"],
                   source_manifest=plan["source_manifest"],
                   execution_commit=plan["execution_commit"])
    links = {}
    for entry in [current, *plan.get("prior_source_manifests", [])]:
        source = Path(entry["source_root"]).absolute()
        payload = Path(entry["source_manifest"]).read_bytes()
        if entry is not current:
            actual = dict(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
            if actual != entry["manifest_identity"]:
                raise ValueError("Prior source manifest identity differs")
        manifest = json.loads(payload)
        if manifest["dirty"] or manifest["commit"] != entry["execution_commit"]:
            raise ValueError("Source manifest does not identify its clean published checkout")
        for name, item in manifest["files"].items():
            if item.get("kind") != "symlink":
                continue
            relative = Path(name)
            if not relative.parts or relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Source manifest link escapes its source root")
            links[source / relative] = (source, item["target"])
    return links


def storage_boundary(plan, *, upcoming_bytes=0, check_compiler=True):
    """Charge all declared new roots once per inode, including old staging."""
    seen, allocation = set(), 0
    source_links = None
    for value in plan["allocation_roots"]:
        root = Path(value)
        if not root.exists():
            continue
        for path in [root, *root.rglob("*")]:
            if path.is_symlink():
                # Include prior staging in the same allocation budget without
                # following links or admitting any mutable run-state link.
                if source_links is None:
                    source_links = _authenticated_source_links(plan)
                declared = source_links.get(path.absolute())
                if (declared is None or os.readlink(path) != declared[1]
                        or not path.resolve().is_relative_to(declared[0].resolve()) or not path.exists()):
                    raise ValueError("Only authenticated internal source symlinks are admissible")
            try:
                stat = path.lstat()
            except FileNotFoundError:
                # Atomic metadata replacement or validated recovery retirement
                # may finish while the other worker samples allocated storage.
                continue
            key = stat.st_dev, stat.st_ino
            if key not in seen:
                seen.add(key)
                allocation += stat.st_blocks * 512
    physical_allocation = allocation
    compiler = compiler_cache_growth(plan) if check_compiler else None
    cache_growth = compiler["growth_bytes"] if compiler else 0
    allocation += int(plan.get("compiler_growth_reserved_bytes", 0))
    free = shutil.disk_usage(plan["root"]).free
    # Reserving a cache allowance is not the same as already writing it: free
    # capacity must still cover its unconsumed portion.
    required = max(0, plan["storage"]["peak_new_bytes"] - physical_allocation - cache_growth)
    passed = (allocation + upcoming_bytes <= ALLOCATION_BYTES
              and free - max(required, upcoming_bytes) >= FREE_RESERVE_BYTES)
    receipt = dict(status="passed" if passed else "blocked", allocated_new_bytes=allocation,
        free_bytes=free, projected_remaining_peak_bytes=required, upcoming_bytes=upcoming_bytes)
    if compiler is not None:
        receipt["compiler"] = compiler
    if not passed:
        raise RuntimeError("Persistent campaign storage admission failed: " + repr(receipt))
    return receipt


def work_boundary(plan, *, upcoming_bytes=0, check_compiler=False):
    clock = clock_status(plan["root"])
    if clock["occupied_s"] >= CARD_SECONDS or clock["elapsed_s"] >= effective_elapsed_limit(plan):
        raise RuntimeError("Persistent campaign shared time cap reached")
    return dict(storage=storage_boundary(plan, upcoming_bytes=upcoming_bytes, check_compiler=check_compiler), clock=clock)


def endpoint(plan, arm, event):
    if arm == "N" and event == 7:
        return dict(state=plan["native_n7_state"], export=plan["native_n7_export"], historical=True)
    receipt = read(Path(plan["root"]) / "tasks" / f"{arm}{event}" / "RESULT.json")
    if receipt["status"] != "completed" or receipt["event"] != event or receipt["arm"] != arm:
        raise ValueError("Missing complete own endpoint")
    return receipt["endpoint"]


def create_backend(plan, base, checkpoint, uuid):
    from online_feedback.sglang_backend import SGLangDFlashBackend
    return SGLangDFlashBackend(plan["target"], checkpoint,
        worker_python=plan["worker_python"], output_root=base / "native",
        serving_gpu=uuid, context_limit=4096, block_size=16,
        worker_module="experiments.persistent_prefix_withholding.worker")


def close_backend(backend, base):
    """Preserve initialization accounting before terminating the owned engine."""
    if backend is None:
        return
    try:
        if not backend.failed:
            write_json(base / "PHYSICAL_WORK.json", backend._call("work_receipt"))
    finally:
        backend.close()


def train_task(plan, task, base, ledger):
    from .training import PersistentPrefixTrainer
    from experiments.prefix_guard.qualify import pin_retained_graph_runtime
    from experiments.functional_step_geometry.native_pilot import discarded_startup
    from experiments.ok_coding.storage import identity
    from online_feedback.rng import preserve_rng

    arm, event = task[0], int(task[1:])
    origin = dict(state=plan["origin_state"], export=plan["origin_export"]) if event == 7 else endpoint(plan, arm, 7)
    features = plan["event7_feature_paths"] if event == 7 else read(Path(plan["root"]) / "tasks" / f"{arm}_collect" / "FEATURES.json")["feature_paths"]
    if len(features) != 128:
        raise ValueError("A production event requires its complete 128 own responses")
    trainer = None
    publication = exclusive(Path(plan["root"]) / "PUBLICATION.lock")
    publication_held = False
    result = dict(task=task, arm=arm, event=event, status="running")
    start = time.perf_counter()
    try:
        result["runtime_compatibility"] = pin_retained_graph_runtime(base)
        trainer = ledger.call({"training_initialization": 1}, "trainer_initialization", PersistentPrefixTrainer,
            plan["configs"][arm], base / "trainer", arm=arm, binding=plan["binding"],
            origin_state=origin["state"], origin_export=origin["export"],
            original_config_text=Path(plan["origin_config"]).read_text(), completed_event=event-1,
            schedule=read(plan["event7_schedule"]) if event == 7 else None)
        result["loading_s"] = time.perf_counter() - start
        result["startup"] = ledger.call({"response_forward": 2, "response_backward": 2},
            "disposable_startup", discarded_startup, trainer, features, base / "startup")
        ledger.wrap(trainer.strategy, "forward_loss", "response_forward")
        ledger.wrap(trainer.backend, "backward", "response_backward")
        ledger.wrap(trainer.backend.optimizer.optimizer, "step", "adam")
        ordinary_passes = trainer._train_passes
        def passes_then_publication_lock(*args, **kwargs):
            nonlocal publication_held
            rows = ordinary_passes(*args, **kwargs)
            wait_started = time.perf_counter()
            publication.__enter__()
            publication_held = True
            result["publication_wait_s"] = time.perf_counter() - wait_started
            work_boundary(plan, upcoming_bytes=plan["storage"]["atomic_publication_bytes"])
            return rows
        trainer._train_passes = passes_then_publication_lock
        began = time.perf_counter()
        receipt = trainer.train_event(f"event_{event:03d}", features, 2)
        result["event_wall_s"] = time.perf_counter() - began
        boundary = trainer.verify_boundary(receipt["checkpoint_path"], base / "BOUNDARY.json")
        if (trainer.global_step != 128 * event or trainer.presentations != 256 * event
                or trainer.core.accumulation_remainder != 0):
            raise ValueError("Own-state production counters differ")
        with preserve_rng():
            state_id = identity(trainer.latest)
            weights_id = identity(Path(receipt["checkpoint_path"]) / "model.safetensors")
        result.update(status="completed", endpoint=dict(state=str(trainer.latest),
            export=receipt["checkpoint_path"], state_identity=state_id, weights_identity=weights_id,
            global_step=trainer.global_step, presentations=trainer.presentations),
            boundary=boundary, costs=receipt["costs"], physical_work=dict(trainer.work))
        write_json(base / "RESULT.json", result)
        if event == 8 and arm in ("E", "S") and not plan.get("production_restart"):
            old = endpoint(plan, arm, 7)
            old_path = Path(old["state"])
            allowed = Path(plan["root"]) / "tasks" / f"{arm}7" / "trainer" / "training_state.pt"
            if old_path.resolve() != allowed.resolve() or identity(old_path) != old["state_identity"]:
                raise ValueError("Superseded recovery identity differs; refusing retirement")
            retirement = dict(status="validated", path=str(old_path), reason="Own event-eight complete recovery/export passed",
                lost_capability="Direct event-seven optimizer resume; serving export retained", retained=result["endpoint"])
            write_json(base / "RETIREMENT.json", retirement)
            old_path.unlink()
            retirement["status"] = "completed"
            write_json(base / "RETIREMENT.json", retirement)
        return result
    except BaseException:
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
                scope="Entire production task including load, startup, training and publication"))
            trainer.close()


def collect_task(plan, task, base, ledger, uuid):
    from experiments.full_k_gsm8k.campaign import online_record_check
    from experiments.kplus_full_gsm8k.storage import retire_redundant_features

    arm = task[0]
    origin = endpoint(plan, arm, 7)
    requests = read(plan["event8_inputs"])["requests"]
    if len(requests) != 128:
        raise ValueError("Require all128 sealed event-eight requests")
    backend, packets = None, []
    (base / "loop/features").mkdir(parents=True)
    (base / "records").mkdir()
    try:
        backend = create_backend(plan, base, origin["export"], uuid)
        for request in requests:
            work_boundary(plan, upcoming_bytes=plan["storage"]["temporary_feature_bytes"])
            record = ledger.call({"feedback_generation": 1}, request["request_id"], backend.generate,
                request["prompt_ids"], max_new_tokens=2048, request_id=request["request_id"],
                session_id=f"persistent_{arm}_event8", request_index=request["request_index"],
                purpose="serving", save_to=base / "records" / (request["request_id"] + ".json"))
            online_record_check(record, request, origin["export"], capture=True)
            packet_path = base / "loop/features" / (request["request_id"] + ".pt")
            prepared = ledger.call({"teacher_preparation": 1}, request["request_id"], backend.prepare_feedback, record, packet_path)
            if prepared.get("status") != "prepared":
                raise ValueError("An own-feedback response has no eligible native prepared packet")
            retire_redundant_features(record, prepared, plan["root"], base)
            packets.append(str(packet_path))
            write_json(base / "FEATURES.json", dict(status="preparing", arm=arm, feature_paths=packets,
                generated_responses=len(packets), own_checkpoint=origin["export"]))
        result = dict(status="completed", arm=arm, generated_responses=128,
            teacher_preparations=128, feature_paths=packets, own_checkpoint=origin["export"], loading_s=backend.loading_s)
        write_json(base / "FEATURES.json", result)
        return result
    finally:
        close_backend(backend, base)


def reference_task(plan, base, ledger, uuid):
    from .references import canonical_record, file_identity, load_missing_requests, record_errors
    requests, merged = load_missing_requests(plan)
    write_json(base / "BOUND_REFERENCE_REQUESTS.json", dict(
        status="validated_before_backend", requests=requests,
        reused_count=len(merged), missing_count=len(requests),
        input_identities={key: file_identity(plan[key]) for key in
                          ("data_panel", "missing_references", "reused_references")},
        checkpoint=plan["origin_export"], collect_features=False, teacher_preparations=0))
    backend = None
    try:
        backend = create_backend(plan, base, plan["origin_export"], uuid)
        for request in requests:
            work_boundary(plan)
            path = base / (request["request_id"] + ".json")
            record = ledger.call({"reference_generation": 1}, request["request_id"], backend.generate,
                request["prompt_ids"], max_new_tokens=2048, request_id=request["request_id"],
                session_id="persistent_fixed_K6_references", request_index=request["panel_ordinal"],
                purpose="evaluation", save_to=path)
            errors = record_errors(record, request)
            write_json(path.with_name(path.stem + "_VALIDATION.json"), dict(errors=errors, status="invalid" if errors else "passed"))
            if errors or record["checkpoint_id"] != plan["origin_export"]:
                raise ValueError("Fixed K6 reference failed authenticated native validation")
            merged[request["request_id"]] = canonical_record(request, record, path, file_identity(path),
                dict(kind="fixed_K6_native_verifier", execution_commit=plan["execution_commit"],
                     source_provenance=plan["reference_provenance"]))
            write_json(plan["references"], merged)
        return dict(status="completed", new_reference_generations=len(requests), loading_s=backend.loading_s)
    finally:
        close_backend(backend, base)


def final_task(plan, base, ledger, uuid):
    from .evaluation import evaluate_final, evaluation_schedule
    from experiments.kplus_full_gsm8k.preflight import timing_barrier
    panel = read(plan["data_panel"])
    requests, permutations = panel["requests"], panel["within_block_permutations"]
    checkpoints = {arm: endpoint(plan, arm, 8)["export"] for arm in ("N", "E", "S")}
    backend = None
    try:
        backend = create_backend(plan, base, checkpoints["N"], uuid)
        original = backend.generate
        def counted(*args, **kwargs):
            kind = "warmup_generation" if kwargs["request_id"].startswith("warmup_") else "scored_generation"
            return ledger.call({kind: 1}, kwargs["request_id"], original, *args, **kwargs)
        backend.generate = counted
        def quiet_boundary():
            timing_barrier(backend.worker.pid)
            work_boundary(plan)
            return dict(status="passed", physical_gpu_uuid=uuid, both_cards_checked=True,
                other_gpu_idle=True, host_quiet=True, storage_admitted=True, time_admitted=True)
        return evaluate_final(backend, requests, checkpoints, base / "evaluation",
            canonicals=read(plan["references"]), warmup_requests=read(plan["warmups"])["requests"],
            warmup_canonicals=read(plan["warmup_references"]), schedule=evaluation_schedule(requests, permutations),
            within_block_permutations=permutations, physical_gpu_uuid=uuid, boundary_check=quiet_boundary,
            execution_commit=plan["execution_commit"])
    finally:
        close_backend(backend, base)


def run_task(plan_path, task, uuid):
    plan = read(plan_path)
    if uuid not in plan["devices"].values() or os.environ.get("CUDA_VISIBLE_DEVICES") != uuid:
        raise ValueError("Worker must be bound to exactly one sealed physical GPU")
    base = Path(plan["root"]) / "tasks" / task
    base.mkdir(parents=True, exist_ok=False)
    result = dict(status="running", task=task, physical_gpu_uuid=uuid,
        execution_commit=plan["execution_commit"], started_epoch=time.time(), automatic_retry=False)
    write_json(base / "STATUS.json", result)
    ledger = SharedWork(plan["root"], task)
    def interrupt(signum, frame):
        raise InterruptedError("Task interrupted; no automatic retry")
    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, interrupt)
    try:
        result["initial_resource_boundary"] = work_boundary(plan, check_compiler=True)
        with occupied_card(plan["root"], task, uuid):
            if task in ("E7", "S7", "N8", "E8", "S8"):
                receipt = train_task(plan, task, base, ledger)
            elif task in ("N_collect", "E_collect", "S_collect"):
                receipt = collect_task(plan, task, base, ledger, uuid)
            elif task == "references":
                receipt = reference_task(plan, base, ledger, uuid)
            elif task == "final":
                receipt = final_task(plan, base, ledger, uuid)
            else:
                raise ValueError("Unknown or unapproved scientific task")
            result["final_resource_boundary"] = work_boundary(plan, check_compiler=True)
        write_json(base / "RESULT.json", receipt)
        result.update(status="completed", result=str(base / "RESULT.json"))
    except BaseException as error:
        result.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        result.update(wall_s=time.time()-result["started_epoch"],
                      host_maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        write_json(base / "STATUS.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    args = parser.parse_args()
    run_task(args.plan, args.task, args.gpu_uuid)


if __name__ == "__main__":
    main()
