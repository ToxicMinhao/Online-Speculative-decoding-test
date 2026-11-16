"""Dependency-aware two-card supervisor, followed by one quiet final sweep."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

from online_feedback.storage import write_json
from .resources import CARD_SECONDS, ELAPSED_SECONDS, HOST_HEADROOM_BYTES, LIMITS, effective_limits, effective_elapsed_limit, time_admission
from .task_runner import clock_status, read, storage_boundary

REQUIRED_COSTS = frozenset(("training_event", "training_startup", "training_load", "publication",
    "feedback_response", "teacher_response", "reference_response", "serving_load",
    "scored_request", "warmup_request", "activation", "resident_wait"))


@dataclass(frozen=True)
class Task:
    name: str
    dependencies: tuple[str, ...]
    kind: str
    components: dict

    @property
    def estimated_card_s(self):
        return sum(self.components.values())


def task_graph(costs, missing_reference_count):
    """All costs are sealed before outcomes; no efficacy-driven scheduling."""
    import math
    if set(costs) != REQUIRED_COSTS or not all(math.isfinite(v) and v >= 0 for v in costs.values()):
        raise ValueError("Every fixed compute/loading/publication cost category must be priced")
    if type(missing_reference_count) is not int or not 0 <= missing_reference_count <= 128:
        raise ValueError("Invalid missing-reference generation budget")
    def training():
        return {key: costs[key] for key in ("training_event", "training_startup", "training_load", "publication", "resident_wait")}
    tasks = [Task(arm + "7", (), "training", training()) for arm in ("E", "S")]
    for arm in ("N", "E", "S"):
        dependencies = () if arm == "N" else (arm + "7",)
        tasks.append(Task(arm + "_collect", dependencies, "collection", dict(
            feedback=128 * costs["feedback_response"], teacher=128 * costs["teacher_response"],
            loading=costs["serving_load"], resident_wait=costs["resident_wait"])))
        tasks.append(Task(arm + "8", (arm + "_collect",), "training", training()))
    tasks.append(Task("references", (), "reference", dict(
        references=missing_reference_count * costs["reference_response"],
        loading=costs["serving_load"] if missing_reference_count else 0,
        resident_wait=costs["resident_wait"] if missing_reference_count else 0)))
    tasks.append(Task("final", ("N8", "E8", "S8", "references"), "final", dict(
        scored=1152 * costs["scored_request"], warmups=72 * costs["warmup_request"],
        activations=36 * costs["activation"], loading=costs["serving_load"],
        resident_wait=costs["resident_wait"])))
    return tasks


def ordered_ready(tasks, completed, active):
    """Longest remaining dependency path first; ties use stable task names."""
    by_name = {task.name: task for task in tasks}
    def path_cost(name):
        children = [row.name for row in tasks if name in row.dependencies]
        return by_name[name].estimated_card_s + max((path_cost(child) for child in children), default=0.)
    ready = [task for task in tasks if task.name not in completed | active
             and set(task.dependencies) <= completed]
    return sorted(ready, key=lambda task: (-path_cost(task.name), task.name))


def remaining_schedule(tasks, *, completed, active, devices, task_memory_bytes,
                       available_host_bytes):
    """Price the same feasible two-card queue at entry and later boundaries.

    ``active`` contains fixed UUIDs, conservative remaining seconds and observed
    resident RAM. MemAvailable already excludes those residents, so add them back
    once before reserving each running worker's full peak. Pending tasks retain
    their full sealed estimate. A final sweep exclusively occupies card A while
    both cards are otherwise idle. Card-seconds and elapsed seconds stay distinct.
    """
    by_name = {task.name: task for task in tasks}
    completed = set(completed)
    if (len(by_name) != len(tasks) or set(devices) != {"A", "B"}
            or len(set(devices.values())) != 2 or completed & set(active)
            or not (completed | set(active)) <= by_name.keys()):
        raise ValueError("Invalid task or physical-card scheduling state")
    if any(not set(task.dependencies) <= by_name.keys() for task in tasks):
        raise ValueError("Task dependency is absent from the sealed queue")
    if any(not set(by_name[name].dependencies) <= completed for name in completed | set(active)):
        raise ValueError("Completed/active task has an unfinished dependency")
    bounds = {task.name: task_memory_bytes[task.kind] for task in tasks}
    values = [available_host_bytes, *bounds.values(), *(task.estimated_card_s for task in tasks)]
    if not all(math.isfinite(value) and value >= 0 for value in values):
        raise ValueError("Task duration and host reservations must be finite and nonnegative")
    running, trace, components = {}, [], {}
    memory_capacity = available_host_bytes-HOST_HEADROOM_BYTES
    for name, row in active.items():
        remaining = row['remaining_s']
        resident = row['resident_bytes']
        if (row['uuid'] not in devices.values() or not math.isfinite(remaining)
                or remaining < 0 or not math.isfinite(resident) or resident < 0):
            raise ValueError("Invalid active worker reservation")
        memory_capacity += resident
        running[name] = dict(finish=remaining, uuid=row['uuid'],
                             reserved_bytes=max(bounds[name], resident))
        components[name] = remaining
        trace.append(dict(task=name, uuid=row['uuid'], start_s=0., finish_s=remaining,
                          active_at_boundary=True, reserved_host_bytes=max(bounds[name], resident)))
    if len({row['uuid'] for row in running.values()}) != len(running):
        raise ValueError("Two active workers cannot share a physical card")
    if any(by_name[name].kind == 'final' for name in running) and len(running) != 1:
        raise ValueError("Final evaluation must have both cards otherwise idle")
    if any(by_name[name].kind == 'final' for name in running) and len(completed) != len(tasks)-1:
        raise ValueError("Final evaluation cannot overlap unfinished production tasks")
    if any(by_name[name].kind == 'final' and row['uuid'] != devices['A'] for name, row in running.items()):
        raise ValueError("Final evaluation must use its fixed physical card A")
    for task in tasks:
        if task.name not in completed and task.name not in active:
            components[task.name] = task.estimated_card_s
    if running and sum(row['reserved_bytes'] for row in running.values()) > memory_capacity:
        return dict(status='blocked', reason='Active worker reservations exceed available host capacity',
            remaining_components=components, remaining_card_s=sum(components.values()),
            remaining_elapsed_s=None, schedule=trace, host_capacity_bytes=memory_capacity)
    now = 0.
    while len(completed) < len(tasks):
        idle = [uuid for uuid in devices.values() if uuid not in {row['uuid'] for row in running.values()}]
        for task in ordered_ready(tasks, completed, set(running)):
            if not idle or any(by_name[name].kind == 'final' for name in running):
                break
            if task.kind == 'final' and (running or len(completed) != len(tasks)-1):
                continue
            reserved = sum(row['reserved_bytes'] for row in running.values())
            if reserved+bounds[task.name] > memory_capacity:
                continue
            uuid = devices['A'] if task.kind == 'final' else idle[0]
            running[task.name] = dict(finish=now+task.estimated_card_s, uuid=uuid,
                                      reserved_bytes=bounds[task.name])
            trace.append(dict(task=task.name, uuid=uuid, start_s=now,
                              finish_s=now+task.estimated_card_s, active_at_boundary=False,
                              reserved_host_bytes=bounds[task.name]))
            idle.remove(uuid)
        if not running:
            return dict(status='blocked', reason='No feasible ready task under dependencies and host reserve',
                remaining_components=components, remaining_card_s=sum(components.values()),
                remaining_elapsed_s=None, schedule=trace, host_capacity_bytes=memory_capacity)
        now = min(row['finish'] for row in running.values())
        done = {name for name, row in running.items() if row['finish'] <= now}
        completed.update(done)
        for name in done:
            del running[name]
    return dict(status='passed', remaining_components=components,
        remaining_card_s=sum(components.values()), remaining_elapsed_s=now,
        schedule=trace, host_capacity_bytes=memory_capacity,
        active_remaining_policy='Full sealed remaining reservation unless an explicit smaller bound is supplied')


def planned_elapsed(tasks):
    """Legacy pure dry-run helper; production uses measured host reservations."""
    result = remaining_schedule(tasks, completed=set(), active={}, devices=dict(A='A', B='B'),
        task_memory_bytes={task.kind: 0 for task in tasks}, available_host_bytes=HOST_HEADROOM_BYTES)
    if result['status'] != 'passed':
        raise ValueError(result['reason'])
    return result['remaining_elapsed_s']


def schedule_admission(plan, tasks, *, completed, active, clock, available_host_bytes=None):
    """One accounting path for initial admission and all production boundaries."""
    residents = {name: dict(uuid=row['uuid'], remaining_s=row['task'].estimated_card_s,
        resident_bytes=process_tree_rss(row['process'].pid)) for name, row in active.items()}
    schedule = remaining_schedule(tasks, completed=completed, active=residents,
        devices=plan['devices'], task_memory_bytes=plan['task_memory_bytes'],
        available_host_bytes=current_memory() if available_host_bytes is None else available_host_bytes)
    if schedule['status'] != 'passed':
        return dict(status='blocked', remaining_schedule=schedule)
    admission = time_admission(occupied_spent_s=clock['occupied_s'],
        remaining_components=schedule['remaining_components'], elapsed_spent_s=clock['elapsed_s'],
        remaining_schedule_s=schedule['remaining_elapsed_s'], elapsed_cap_s=effective_elapsed_limit(plan))
    return dict(admission, remaining_schedule=schedule)


def mandatory_gates(plan):
    """No production entry until source, native and resource gates are complete."""
    for label in ("A", "B", "cross_card", "serving"):
        receipt = read(plan["qualification_receipts"][label])
        compatible = receipt.get("execution_commit") == plan["execution_commit"]
        if plan.get('production_restart'):
            from .production_restart import qualified_receipt_matches
            compatible = qualified_receipt_matches(plan, label, receipt)
        elif plan.get('production_continuation'):
            from .production_continue import qualified_receipt_matches
            compatible = qualified_receipt_matches(plan, label, receipt)
        if receipt.get("status") != "passed" or not compatible:
            raise ValueError("Unpassed qualification gate: " + label)
    for label in ("cpu_gate", "reference_provenance", "resource_admission"):
        receipt = read(plan[label])
        if receipt.get("status") != "passed":
            raise ValueError("Unpassed production gate: " + label)
    if len(set(plan["devices"].values())) != 2 or set(plan["devices"]) != {"A", "B"}:
        raise ValueError("Require exactly two distinct sealed physical cards")
    if read(plan["cpu_gate"])["execution_commit"] != plan["execution_commit"]:
        raise ValueError("CPU gate did not qualify this execution checkout")


def production_expectations(missing_reference_count):
    expected = {key: 0 for key in LIMITS}
    expected.update(adam=640, response_forward=1290, response_backward=1290,
        training_initialization=5, feedback_generation=384, teacher_preparation=384,
        reference_generation=missing_reference_count, warmup_generation=72, scored_generation=1152)
    return expected


def current_memory():
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        values[key] = int(value.split()[0]) * 1024
    return values["MemAvailable"]


def process_tree_rss(pid):
    """Include native runtime children when reserving active-worker RAM growth."""
    pairs = {}
    for line in subprocess.check_output(["ps", "-eo", "pid=,ppid=,rss="], text=True).splitlines():
        process, parent, rss = map(int, line.split())
        pairs[process] = parent, rss * 1024
    members = {pid}
    while True:
        expanded = members | {process for process, (parent, _) in pairs.items() if parent in members}
        if expanded == members:
            break
        members = expanded
    return sum(pairs.get(process, (0, 0))[1] for process in members)


def memory_admission(plan, task, active):
    """Reserve future active-worker peaks as well as the new worker's peak."""
    bounds = plan["task_memory_bytes"]
    if task.kind not in bounds or bounds[task.kind] <= 0:
        raise ValueError("No measured/source-bounded task working set")
    future = sum(max(0, bounds[row["task"].kind] - process_tree_rss(row["process"].pid)) for row in active.values())
    available = current_memory()
    required = bounds[task.kind] + future + HOST_HEADROOM_BYTES
    return dict(status="passed" if available >= required else "serialize",
                available_host_bytes=available, required_host_bytes=required,
                active_future_growth_bytes=future, new_worker_peak_bytes=bounds[task.kind])


def gpu_admission(plan, task, uuid):
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,memory.free", "--format=csv,noheader,nounits"], text=True)
    free = {line.split(",")[0].strip(): int(line.split(",")[1].strip()) * 2**20 for line in raw.splitlines()}
    needed = plan["task_vram_bytes"][task.kind]
    return dict(status="passed" if free[uuid] >= needed else "blocked", physical_gpu_uuid=uuid,
                free_bytes=free[uuid], required_bytes=needed)


def start_task(plan_path, plan, task, uuid):
    from .launch import QUALIFIED_COMPILERS, check_compilers
    check_compilers()
    root = Path(plan["root"])
    environment = dict(os.environ)
    environment.update(QUALIFIED_COMPILERS)
    environment["CUDA_VISIBLE_DEVICES"] = uuid
    environment["PYTHONPATH"] = os.pathsep.join([plan["source_root"], str(Path(plan["source_root"])/"training"),
        str(Path(plan["source_root"])/"inference"), environment.get("PYTHONPATH", "")])
    # Each independent single-card trainer obtains its own local rendezvous.
    for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        environment.pop(key, None)
    log = (root / "logs" / (task.name + ".log")).open("w")
    process = subprocess.Popen([plan["training_python"], "-u", "-m",
        "experiments.persistent_prefix_withholding.task_runner", "--plan", str(plan_path),
        "--task", task.name, "--gpu-uuid", uuid], env=environment, cwd=plan["source_root"],
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    return dict(process=process, log=log, task=task, uuid=uuid, started_epoch=time.time())


def stop_workers(active):
    """Stop only owned workers; their handlers close their own native engines."""
    for row in active.values():
        if row["process"].poll() is None:
            os.killpg(row["process"].pid, signal.SIGTERM)
    for row in active.values():
        try:
            row["process"].wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(row["process"].pid, signal.SIGKILL)
            row["process"].wait()
        row["log"].close()


def run_campaign(plan_path):
    plan = read(plan_path)
    mandatory_gates(plan)
    root = Path(plan["root"])
    if (root / "CAMPAIGN.json").exists():
        raise FileExistsError("No automatic scientific campaign retry or resume")
    (root / "logs").mkdir(exist_ok=True)
    missing = read(plan["missing_references"])
    missing = missing["requests"] if isinstance(missing, dict) else missing
    tasks = task_graph(plan["costs"], len(missing))
    restart = None
    if plan.get("production_restart"):
        from .production_continue import bound_json
        restart = bound_json(plan["restart_import"])
        if (restart["status"] != "passed" or restart["execution_commit"] != plan["execution_commit"]
                or restart["binding"] != plan["binding"]):
            raise ValueError("Restart import does not identify the current execution")
    initial_completed = set(restart["reused_tasks"]) if restart else set()
    clock = clock_status(root)
    priced = schedule_admission(plan, tasks, completed=initial_completed, active={}, clock=clock)
    write_json(root / "PRODUCTION_ADMISSION.json", priced)
    if priced["status"] != "passed":
        raise RuntimeError("The complete fixed comparison does not fit the declared time cap")
    storage_boundary(plan)
    initial_work = read(root / "WORK.json")
    if restart:
        if initial_work != restart["initial_work"]:
            raise ValueError("Restart work differs from the authenticated imported ledger")
    elif initial_work["pending"] or initial_work["attempted"] != initial_work["completed"]:
        raise ValueError("Unknown qualification work prevents production entry")
    remaining_work = restart["remaining"] if restart else production_expectations(len(missing))
    limits = effective_limits(plan)
    if any(initial_work['attempted'][kind]+amount > limits[kind]
           for kind, amount in remaining_work.items()):
        raise ValueError("Complete production work exceeds the effective operational limits")
    progress = dict(status="running", execution_commit=plan["execution_commit"], completed=sorted(initial_completed),
        schedule=[], idle_reasons=[], automatic_retry=False, production_admission=priced,
        initial_work=initial_work, production_expected=remaining_work, effective_limits=limits)
    write_json(root / "CAMPAIGN.json", progress)
    active, completed = {}, set(initial_completed)
    try:
        while len(completed) < len(tasks):
            clock = clock_status(root)
            if clock["occupied_s"] >= CARD_SECONDS or clock["elapsed_s"] >= effective_elapsed_limit(plan):
                raise RuntimeError("The shared occupied-card or elapsed time cap was reached")
            for name, worker in list(active.items()):
                code = worker["process"].poll()
                if code is None:
                    continue
                worker["log"].close()
                task_status = read(root / "tasks" / name / "STATUS.json")
                progress["schedule"].append(dict(task=name, physical_gpu_uuid=worker["uuid"],
                    started_epoch=worker["started_epoch"], finished_epoch=time.time(), exit_code=code,
                    status=task_status["status"]))
                del active[name]
                if code or task_status["status"] != "completed":
                    raise RuntimeError("Scientific task stopped: " + name)
                completed.add(name)
                progress["completed"] = sorted(completed)
            remaining = [task for task in tasks if task.name not in completed]
            if not remaining:
                break
            # Fixed task boundaries reprice cost only, never losses or efficacy.
            # Charge a running task's full remaining estimate conservatively.
            clock = clock_status(root)
            admission = schedule_admission(plan, tasks, completed=completed, active=active, clock=clock)
            progress["latest_admission"] = admission
            idle_devices = [uuid for uuid in plan["devices"].values() if uuid not in {w["uuid"] for w in active.values()}]
            for task in ordered_ready(tasks, completed, set(active)):
                if not idle_devices or (task.kind == "final" and active):
                    break
                if admission["status"] != "passed":
                    if not active:
                        raise RuntimeError("Remaining fixed work no longer fits time admission")
                    progress["idle_reasons"].append(dict(epoch=time.time(), reason="time_margin_pending_active_completion", task=task.name))
                    break
                uuid = plan["devices"]["A"] if task.kind == "final" else idle_devices[0]
                memory = memory_admission(plan, task, active)
                gpu = gpu_admission(plan, task, uuid)
                if memory["status"] != "passed" or gpu["status"] != "passed":
                    if not active:
                        raise RuntimeError("No safe single-task RAM/VRAM admission: " + repr((memory, gpu)))
                    progress["idle_reasons"].append(dict(epoch=time.time(), task=task.name, memory=memory, gpu=gpu))
                    continue
                storage_boundary(plan)
                active[task.name] = start_task(plan_path, plan, task, uuid)
                idle_devices.remove(uuid)
                if task.kind == "final":
                    break
            write_json(root / "CAMPAIGN.json", progress)
            time.sleep(1)
        progress.update(status="completed", actual_clock=clock_status(root), work=read(root / "WORK.json"))
        if progress["work"]["pending"]:
            raise RuntimeError("Unknown scientific work remains after nominal completion")
        for kind, amount in progress["production_expected"].items():
            expected = initial_work["completed"][kind] + amount
            attempted = initial_work["attempted"][kind] + amount
            if (progress["work"]["completed"][kind] != expected
                    or progress["work"]["attempted"][kind] != attempted):
                raise RuntimeError("Final production work differs: " + kind)
    except BaseException as error:
        progress.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        stop_workers(active)
        progress["actual_clock"] = clock_status(root)
        progress["work"] = read(root / "WORK.json")
        write_json(root / "CAMPAIGN.json", progress)
    return progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    args = parser.parse_args()
    run_campaign(args.plan)


if __name__ == "__main__":
    main()
