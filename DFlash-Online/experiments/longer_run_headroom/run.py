"""Bounded frozen replay; original walks, neutral checks and one child block."""

from __future__ import annotations

import argparse
import importlib.metadata
import platform
from pathlib import Path
import os
import shutil
import subprocess
import sys
import time
import traceback

from online_feedback.storage import write_json
from experiments.verification_transition.data import read, identity
from experiments.verification_transition.budget import (
    WorkBudget,
    BudgetService,
    connect_budget,
)
from experiments.verification_transition.run import (
    active_owned_processes,
    stop_owned_groups,
)
from .pairs import neutral_parity, score

LIMITS = dict(
    original_completion=128, neutral_block=256, child_block=256, draft_block=8000
)
GPU_SECONDS = 2700


def preflight(plan, output, source):
    if plan["limits"] != LIMITS or plan["gpu_seconds"] != GPU_SECONDS:
        raise ValueError("Frozen diagnostic budget changed")
    checks = []
    for dataset, arms in plan["checkpoints"].items():
        for arm, item in arms.items():
            actual = identity(item["weights"])
            if actual != item["weights_identity"]:
                raise ValueError(
                    "Pinned checkpoint missing or changed: " + item["weights"]
                )
            checks.append(dict(path=item["weights"], **actual))
    shared = plan["shared"]
    for name, expected in shared["target_files"].items():
        path = Path(shared["target_path"]) / name
        actual = identity(path)
        if actual != expected:
            raise ValueError("Target/tokenizer changed: " + str(path))
        checks.append(dict(path=str(path), **actual))
    runtime = shared["runtime"]
    actual_versions = dict(
        python_version=platform.python_version(),
        torch=importlib.metadata.version("torch"),
        transformers=importlib.metadata.version("transformers"),
    )
    for key, value in actual_versions.items():
        if key in runtime and value != runtime[key]:
            raise ValueError("Pinned environment differs: " + key)
    version = ".".join(runtime["python_version"].split(".")[:2])
    package = (
        Path(runtime["python"]).parent.parent
        / "lib"
        / f"python{version}"
        / "site-packages/sglang/srt"
    )
    for name, expected in runtime["native_runtime_sha256"].items():
        path = package / name
        actual = identity(path)
        if actual["sha256"] != expected:
            raise ValueError("Native runtime changed: " + str(path))
        checks.append(dict(path=str(path), **actual))
    for item in plan["raw_records"]:
        if identity(item["path"]) != {k: item[k] for k in ("bytes", "sha256")}:
            raise ValueError("Retained source walk changed: " + item["path"])
        checks.append(item)
    if not output.resolve().is_relative_to("/data/sources/minhao/runs"):
        raise ValueError("Output is outside the approved experiment root")
    free = shutil.disk_usage(output).free
    if free < 10 * 1024**3:
        raise RuntimeError("Require 2 GiB artifacts plus 8 GiB reserve")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=source, text=True
    )
    if dirty:
        raise ValueError("Execution checkout must be clean and published")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source, text=True
    ).strip()
    jobs = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True,
    )
    if jobs.strip():
        raise RuntimeError("Native replay requires a quiet host")
    sources = {
        str(p.relative_to(source)): identity(p)
        for p in sorted(source.rglob("*.py"))
        if "__pycache__" not in p.parts
    }
    return dict(
        status="passed",
        execution_commit=commit,
        artifact_checks=checks,
        source_files=sources,
        environment=actual_versions,
        free_bytes=free,
        reserve_bytes=8 * 1024**3,
        artifact_allowance_bytes=2 * 1024**3,
    )


def counted(budget, kind, function, **metadata):
    ticket = budget.attempt(kind, **metadata)
    try:
        result = function()
    except BaseException as error:
        budget.fail(ticket, repr(error))
        raise
    budget.complete(ticket)
    return result


def worker(plan_path, output, connection):
    from .native_worker import HeadroomBackend

    plan = read(plan_path)
    budget = connect_budget(read(connection))
    rows = plan["initial_rows"]
    backend = None
    active_keys = []

    def persist(status, **extra):
        write_json(output / "ROWS.json", rows)
        write_json(
            output / "STATUS.json",
            dict(
                status=status,
                valid=sum(r["status"] == "valid" for r in rows),
                pending=sum(r["status"] == "pending" for r in rows),
                invalid=sum(r["status"] == "invalid" for r in rows),
                planned=len(rows),
                work=budget.summary(),
                **extra,
            ),
        )

    persist("loading")
    started = time.perf_counter()
    load = budget.attempt("model_load")
    try:
        backend = HeadroomBackend(
            plan["shared"]["target_path"],
            plan["checkpoints"]["gsm8k"]["KPLUS4"]["export"],
            worker_python=plan["shared"]["runtime"]["python"],
            output_root=output / "native",
            budget_connection=read(connection),
        )
        budget.complete(load, loading_s=backend.loading_s)
        for number, job in enumerate(plan["jobs"]):
            item = job["item"]
            dataset = item["dataset"]
            request_id = item["request_id"]
            folder = output / "replays" / dataset / f'{item["selection_rank"]:03d}'
            folder.mkdir(parents=True, exist_ok=False)
            parent = read(job["parent_path"])
            active_keys = [
                (dataset, request_id, s["parent_round_ordinal"]) for s in job["starts"]
            ]
            checkpoint4 = plan["checkpoints"][dataset]["KPLUS4"]["export"]
            checkpoint6 = plan["checkpoints"][dataset]["KPLUS6"]["export"]
            boundaries = sorted({s["parent_round_ordinal"] - 1 for s in job["starts"]})
            result = counted(
                budget,
                "original_completion",
                lambda: backend.native_walk(
                    dict(item["request"], canonical=item["canonical"]),
                    checkpoint4,
                    boundaries,
                ),
                dataset=dataset,
                request_id=request_id,
            )
            write_json(folder / "PARENT.json", result["record"])
            a, b = parent["metadata"]["trace"], result["record"]["metadata"]["trace"]
            parity = [neutral_parity(x, y) for x, y in zip(a, b)]
            valid = (
                len(a) == len(b)
                and result["record"]["response_ids"] == parent["response_ids"]
                and result["record"]["finish_reason"] == parent["finish_reason"]
                and all(r["status"] == "passed" for r in parity)
            )
            write_json(
                folder / "PARENT_PARITY.json",
                dict(status="passed" if valid else "failed", blocks=parity),
            )
            if not valid:
                raise ValueError("Fresh parent walk failed exact archived parity")
            # All neutral replays use parent weights before one switch to child.
            for s in job["starts"]:
                ordinal = s["parent_round_ordinal"]
                handle = result["successors"][str(ordinal - 1)]
                active_keys = [(dataset, request_id, ordinal)]
                observed = counted(
                    budget,
                    "neutral_block",
                    lambda: backend.one_block(handle, checkpoint4),
                    dataset=dataset,
                    request_id=request_id,
                    ordinal=ordinal,
                )
                write_json(folder / f"{ordinal:04d}_NEUTRAL.json", observed)
                check = neutral_parity(a[ordinal - 1], observed["blocks"][0])
                write_json(folder / f"{ordinal:04d}_PARITY.json", check)
                if check["status"] != "passed":
                    raise ValueError("Neutral one-block reconstruction mismatch")
            for s in job["starts"]:
                ordinal = s["parent_round_ordinal"]
                handle = result["successors"][str(ordinal - 1)]
                active_keys = [(dataset, request_id, ordinal)]
                observed = counted(
                    budget,
                    "child_block",
                    lambda: backend.one_block(handle, checkpoint6),
                    dataset=dataset,
                    request_id=request_id,
                    ordinal=ordinal,
                )
                write_json(folder / f"{ordinal:04d}_CHILD.json", observed)
                paired = score(
                    s,
                    a[ordinal - 1],
                    observed["blocks"][0],
                    item["canonical"]["response_ids"],
                    provenance="native_parent_reconstruction_with_neutral_parity",
                )
                index = next(
                    i
                    for i, r in enumerate(rows)
                    if (r["dataset"], r["request_id"], r["parent_round_ordinal"])
                    == (dataset, request_id, ordinal)
                )
                rows[index] = paired
                persist(
                    "running",
                    completed_parent_requests=number,
                    active_request=request_id,
                )
            backend.release_request()
            persist("running", completed_parent_requests=number + 1)
        persist("completed", worker_wall_s=time.perf_counter() - started)
        return 0
    except BaseException as error:
        if backend is None:
            budget.fail(load, repr(error))
        for row in rows:
            key = (row["dataset"], row["request_id"], row["parent_round_ordinal"])
            if row["status"] == "pending":
                row["reason"] = (
                    repr(error)
                    if key in active_keys
                    else "unattempted after execution stopped"
                )
                # Integrity failures invalidate the affected start; resource stops do not.
                if key in active_keys and isinstance(error, ValueError):
                    row["status"] = "invalid"
        persist(
            "incomplete",
            reason=repr(error),
            traceback=traceback.format_exc(),
            worker_wall_s=time.perf_counter() - started,
        )
        return 2
    finally:
        if backend is not None:
            backend.close()


def execute(inputs, output):
    output = Path(output).absolute()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    plan = read(inputs)
    if identity(inputs) != read(Path(inputs).parent / "SEAL.json")["plan"]:
        raise ValueError("Bound plan changed after sealing")
    source = Path(__file__).resolve().parents[2]
    admission = preflight(plan, output, source)
    write_json(output / "PREFLIGHT.json", admission)
    plan["execution_commit"] = admission["execution_commit"]
    write_json(output / "PLAN.json", plan)
    budget = WorkBudget(output / "WORK.jsonl", limits=LIMITS, gpu_seconds=GPU_SECONDS)
    service = BudgetService(budget)
    connection = output / "budget_connection.json"
    write_json(connection, service.connection)
    connection.chmod(0o600)
    proxy = connect_budget(service.connection)
    child = None
    timeout = False
    try:
        proxy.start_gpu(1)
        with (output / "worker.log").open("w") as log:
            child = subprocess.Popen(
                [
                    plan["shared"]["runtime"]["python"],
                    "-u",
                    "-m",
                    "experiments.longer_run_headroom.run",
                    "--worker",
                    "--inputs",
                    str(output / "PLAN.json"),
                    "--output",
                    str(output),
                    "--connection",
                    str(connection),
                ],
                cwd=source,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            write_json(
                output / "PROCESS.json",
                dict(pid=child.pid, occupied_gpus=1, gpu_seconds_limit=GPU_SECONDS),
            )
            try:
                code = child.wait(
                    timeout=max(0.01, GPU_SECONDS - 30 - proxy.summary()["gpu_seconds"])
                )
            except subprocess.TimeoutExpired:
                timeout = True
                write_json(
                    output / "TIMEOUT_CLEANUP.json", stop_owned_groups(proxy, child)
                )
                code = 2
        remaining = active_owned_processes({r["pgid"] for r in proxy.processes()})
        if remaining:
            write_json(output / "ORPHAN_CLEANUP.json", stop_owned_groups(proxy, child))
            code = 2
        proxy.end_gpu()
        if timeout or not (output / "STATUS.json").exists():
            rows = (
                read(output / "ROWS.json")
                if (output / "ROWS.json").exists()
                else plan["initial_rows"]
            )
            write_json(output / "ROWS.json", rows)
            write_json(
                output / "STATUS.json",
                dict(
                    status="incomplete",
                    reason="time cap" if timeout else "worker failed",
                    valid=sum(r["status"] == "valid" for r in rows),
                    pending=sum(r["status"] == "pending" for r in rows),
                ),
            )
        postflight = []
        for arms in plan["checkpoints"].values():
            for item in arms.values():
                actual = identity(item["weights"])
                postflight.append(
                    dict(
                        path=item["weights"],
                        unchanged=actual == item["weights_identity"],
                    )
                )
        source_unchanged = all(
            identity(source / name) == expected
            for name, expected in admission["source_files"].items()
        )
        records_unchanged = all(
            identity(item["path"]) == {k: item[k] for k in ("bytes", "sha256")}
            for item in plan["raw_records"]
        )
        write_json(
            output / "POSTFLIGHT.json",
            dict(
                checkpoints=postflight,
                source_unchanged=source_unchanged,
                records_unchanged=records_unchanged,
            ),
        )
        if (
            not all(r["unchanged"] for r in postflight)
            or not source_unchanged
            or not records_unchanged
        ):
            code = 2
        write_json(
            output / "RESULT.json",
            dict(
                exit_code=code,
                timed_out=timeout,
                work=budget.summary(),
                wall_s=time.perf_counter() - started,
                execution_commit=plan["execution_commit"],
            ),
        )
        return code
    finally:
        groups = {r["pgid"] for r in proxy.processes()}
        if (child is not None and child.poll() is None) or active_owned_processes(
            groups
        ):
            write_json(output / "FINAL_CLEANUP.json", stop_owned_groups(proxy, child))
        proxy.end_gpu()
        service.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--worker", action="store_true")
    p.add_argument("--connection", type=Path)
    a = p.parse_args()
    try:
        code = (
            worker(a.inputs, a.output, a.connection)
            if a.worker
            else execute(a.inputs, a.output)
        )
    except BaseException as error:
        a.output.mkdir(parents=True, exist_ok=True)
        write_json(
            a.output / "EXECUTION_FAILURE.json",
            dict(error=repr(error), traceback=traceback.format_exc()),
        )
        raise
    raise SystemExit(code)
