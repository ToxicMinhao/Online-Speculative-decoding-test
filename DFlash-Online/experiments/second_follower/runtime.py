"""One fixed event, fresh paired endpoint and bounded matched-start explanation."""

import argparse
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

from online_feedback.storage import write_json
from experiments.survival_head.data import read
from experiments.ok_coding.storage import identity
from experiments.survival_confirmation.runtime import (
    capacity,
    physical_gpu,
    finish_worker,
)
from experiments.kplus_full_gsm8k.preflight import timing_barrier
from .contract import SCHEMA, ANALYSIS, SERVING, GPU_SECONDS_LIMIT


def verify_exports(plan, checkpoints):
    from experiments.osd_components.prepare import verify_files

    checked = {}
    for arm, item in checkpoints.items():
        verify_files(item["checkpoint_path"], item["export_files"], exact_files=True)
        if (
            identity(Path(item["checkpoint_path"]) / "model.safetensors")
            != item["weights_identity"]
        ):
            raise ValueError("Checkpoint weights changed: " + arm)
        checked[arm] = item["export_files"]
    return checked


def preflight(plan, *, postflight=False):
    from experiments.successor_shuffle.campaign import shared_inputs, authenticate_true
    from experiments.survival_head.seal import authenticate_source
    from experiments.osd_components.prepare import verify_files, verify_native_runtime

    root = Path(plan["root"])
    code = Path(__file__).resolve().parents[2]
    record = dict(status="running")
    started = time.perf_counter()
    try:
        if (
            plan["schema"] != SCHEMA
            or plan["analysis"] != ANALYSIS
            or plan["serving"] != SERVING
            or plan["budget"]["main_optimizer_steps"] != 128
            or plan["budget"]["timed_completions"] != 1024
        ):
            raise ValueError("Fixed experiment contract differs")
        seal = read(root / "sealed/SEAL.json")
        for path, expected in seal["files"].items():
            if identity(path) != expected:
                raise ValueError("Sealed input changed: " + path)
        source = read(plan["source_manifest"])
        authenticate_source(source, code)
        if (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=code, text=True
            ).strip()
            != source["commit"]
        ):
            raise ValueError("Execution Git commit changed")
        if subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=code, text=True
        ).strip():
            raise ValueError("Execution checkout not clean")
        if identity(plan["parent_plan"]) != plan["parent_plan_identity"]:
            raise ValueError("Parent plan changed")
        authenticate_true(read(plan["parent_plan"]))
        verify_exports(plan, plan["checkpoints"])
        shared_inputs(plan)
        for key in ("origin_state", "origin_weights"):
            if identity(plan[key]) != plan[key + "_identity"]:
                raise ValueError("Parent state changed")
        pins = read(root / "sealed/MODEL_PIN.json")
        for name in ("target", "draft"):
            verify_files(plan[name], pins[name]["files"], exact_files=True)
        import platform, torch, transformers

        versions = dict(
            python=platform.python_version(),
            torch=torch.__version__,
            transformers=transformers.__version__,
        )
        if (
            versions != read(root / "sealed/TRAINING_PIN.json")["environment"]
            or torch.cuda.is_initialized()
        ):
            raise ValueError("Training runtime changed or preflight initialized CUDA")
        record["training_runtime"] = versions
        record["runtime"] = verify_native_runtime(
            plan["worker_python"], read(root / "sealed/BOUND_RUNTIME_PIN.json")
        )
        record["capacity"] = capacity(plan)
        record["physical_gpu"] = physical_gpu(plan)
        record["quiet"] = timing_barrier()
        record["status"] = "passed"
    except BaseException as error:
        record.update(status="failed", error=repr(error))
        raise
    finally:
        record["wall_s"] = time.perf_counter() - started
        write_json(
            root / ("POSTFLIGHT.json" if postflight else "PREFLIGHT.json"), record
        )
    return record


def owned_processes(pid):
    """Snapshot this phase's descendants, including workers in detached sessions."""
    rows = [
        tuple(map(int, line.split()))
        for line in subprocess.check_output(
            ["ps", "-eo", "pid=,ppid=,pgid="], text=True
        ).splitlines()
    ]
    selected = {pid}
    while True:
        expanded = selected | {p for p, parent, _ in rows if parent in selected}
        if expanded == selected:
            break
        selected = expanded
    result = []
    for p, _, group in rows:
        if p in selected:
            try:
                start = (
                    Path(f"/proc/{p}/stat").read_text().rsplit(")", 1)[1].split()[19]
                )
                result.append((p, group, start))
            except FileNotFoundError:
                pass
    return result


def stop_owned_groups(snapshot):
    """Kill only still-live groups authenticated to the original phase tree."""
    groups = set()
    for pid, group, start in snapshot:
        try:
            current = (
                Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
            )
            if current == start and os.getpgid(pid) == group:
                groups.add(group)
        except ProcessLookupError:
            pass
        except FileNotFoundError:
            pass
    for group in groups:
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return sorted(groups)


def launch(plan, module, args, path, deadline, *, allow_failure=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    remaining = deadline - time.time()
    if remaining <= 75:
        raise TimeoutError("Declared GPU budget exhausted before phase")
    command = [plan["training_python"], "-u", "-m", module, plan["root"], *args]
    record = dict(status="running", command=command)
    start = time.perf_counter()
    with path.with_suffix(".log").open("x") as log:
        process = subprocess.Popen(
            command,
            cwd=plan["execution_source"]["source_path"],
            env=dict(os.environ, CUDA_VISIBLE_DEVICES="0"),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        record["pid"] = process.pid
        write_json(path, record)
        try:
            code = process.wait(timeout=max(1, remaining - 75))
        except BaseException:
            snapshot = owned_processes(process.pid)
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                record["forced_owned_groups"] = stop_owned_groups(snapshot)
                process.wait()
            record["remaining_owned_groups"] = stop_owned_groups(snapshot)
            raise
        finally:
            record.update(
                returncode=process.returncode,
                wall_s=time.perf_counter() - start,
                status="completed" if process.returncode == 0 else "failed",
            )
            write_json(path, record)
    if code and not allow_failure:
        raise RuntimeError("Native phase failed; evidence retained: " + str(path))
    return record


def evaluate(root, deadline):
    from online_feedback.sglang_backend import SGLangDFlashBackend
    from .evaluation import evaluate_stage, audit_stage

    root = Path(root)
    plan = read(root / "sealed/PLAN.json")
    checkpoints = read(root / "CHECKPOINTS.json")
    paths = {a: c["checkpoint_path"] for a, c in checkpoints.items()}
    requests = read(plan["panel_manifest"])["requests"]
    refs = read(plan["references"])
    out = root / "evaluation"
    backend = None
    receipt = dict(status="running")
    failure = None
    started = time.perf_counter()
    try:
        capacity(plan)
        timing_barrier()
        verify_exports(plan, checkpoints)
        backend = SGLangDFlashBackend(
            plan["target"],
            paths["NATIVE"],
            worker_python=plan["worker_python"],
            output_root=root / "evaluation_native",
            serving_gpu="0",
            context_limit=4096,
            block_size=16,
        )
        receipt["loading_s"] = backend.loading_s

        def boundary():
            if time.time() >= deadline - 75:
                raise TimeoutError("Declared GPU time cap")
            return dict(
                storage=capacity(plan),
                quiescence=timing_barrier(backend.worker.pid),
                physical_gpu=physical_gpu(plan),
            )

        kwargs = dict(
            canonicals=refs,
            sealed_n=256,
            schedule=plan["evaluation_schedule"],
            physical_gpu=plan["serving_gpu_identity"],
        )
        evaluate_stage(backend, requests, paths, out, boundary_check=boundary, **kwargs)
        receipt["exports"] = verify_exports(plan, checkpoints)
        audit = audit_stage(out, requests, paths, evidence_root=root, **kwargs)
        write_json(out / "AUDIT.json", audit)
        if audit["status"] != "passed":
            raise ValueError("Native raw audit failed")
        receipt["status"] = "completed"
    except BaseException as error:
        failure = error
        receipt.update(status="failed", error=repr(error))
        raise
    finally:
        finish_worker(backend, receipt, out / "WORKER.json", started, failure)
    return receipt


def run(root):
    root = Path(root)
    plan = read(root / "sealed/PLAN.json")
    if (root / "STATUS.json").exists():
        raise FileExistsError("Preserve existing execution")
    status = dict(
        status="running",
        phase="preflight",
        execution_commit=plan["execution_source"]["commit"],
    )
    costs = {}
    start = time.perf_counter()
    deadline = None

    def phase(name):
        status["phase"] = name
        write_json(root / "STATUS.json", status)

    try:
        costs["preflight"] = preflight(plan)
        deadline = time.time() + GPU_SECONDS_LIMIT
        write_json(
            root / "GPU_CLOCK.json",
            dict(
                start_epoch=time.time(),
                deadline_epoch=deadline,
                limit_s=GPU_SECONDS_LIMIT,
                policy="conservative wall since first GPU phase including subprocess gaps",
            ),
        )
        phase("qualification_exposure")
        costs["exposure"] = launch(
            plan,
            "experiments.second_follower.qualification",
            ["--case", "exposure"],
            root / "qualification/EXPOSURE_EXEC.json",
            deadline,
        )
        exposure = read(root / "qualification/exposure/RESULT.json")
        phase("qualification_updates")
        for case in ("NATIVE", "ZERO", "ACTIVE"):
            costs[case] = launch(
                plan,
                "experiments.second_follower.qualification",
                ["--case", case],
                root / f"qualification/{case}_EXEC.json",
                deadline,
            )
        cases = {
            k: read(root / f"qualification/{k}/RESULT.json")
            for k in ("NATIVE", "ZERO", "ACTIVE")
        }
        if any(
            r["status"] != "completed" or r["finite"]["status"] != "passed"
            for r in cases.values()
        ):
            raise ValueError("Numerical qualification incomplete")
        if (
            not cases["ZERO"]["exact"]["exact_equal"]
            or not cases["ACTIVE"]["exact"]["exact_equal"]
        ):
            raise ValueError("Native identity or sampling changed")
        admission = dict(
            status="passed",
            cases={k: identity(root / f"qualification/{k}/RESULT.json") for k in cases},
            first_active_window=exposure["first_active_window"],
            zero_transfer_exact=True,
            active_changed_gradients=cases["ACTIVE"]["gradient_changes"]["changed"],
            active_changed_weights=cases["ACTIVE"]["weight_changes"]["changed"],
            positive_transfer_checked=cases["ACTIVE"]["transferred_weight"] > 0,
        )
        write_json(root / "qualification/ADMISSION.json", admission)
        phase("training")
        costs["training"] = launch(
            plan,
            "experiments.second_follower.training",
            [],
            root / "training/TRAIN_EXEC.json",
            deadline,
        )
        result = read(root / "training/SECOND/RESULT.json")
        checkpoints = dict(
            plan["checkpoints"],
            SECOND=dict(
                checkpoint_path=result["checkpoint_path"],
                weights_identity=result["weights_identity"],
                export_files={
                    p.name: identity(p)
                    for p in Path(result["checkpoint_path"]).iterdir()
                    if p.is_file()
                },
                cumulative_adam_steps=640,
            ),
        )
        checkpoints["NATIVE"]["cumulative_adam_steps"] = 640
        write_json(root / "CHECKPOINTS.json", checkpoints)
        transferred = sum(
            r["response_observation"]["sum_metrics"][
                "second_follower_transferred_weight"
            ]
            for r in result["records"]
        )
        if (
            transferred == 0
            or checkpoints["NATIVE"]["weights_identity"]
            == checkpoints["SECOND"]["weights_identity"]
        ):
            status.update(
                status="completed_inactive",
                phase="stopped_before_duplicate_serving",
                transferred_mass=transferred,
            )
            costs["postflight"] = preflight(plan, postflight=True)
            return
        phase("timed_evaluation")
        costs["evaluation"] = launch(
            plan,
            "experiments.second_follower.runtime",
            ["--evaluate", "--deadline", str(deadline)],
            root / "EVALUATE_EXEC.json",
            deadline,
        )
        phase("matched_start")
        remaining = deadline - time.time()
        if remaining > 120:
            costs["matched_start"] = launch(
                plan,
                "experiments.second_follower.matched_start",
                ["--remaining-s", str(remaining - 75)],
                root / "MATCHED_EXEC.json",
                deadline,
                allow_failure=True,
            )
        else:
            write_json(
                root / "matched_start/STATUS.json",
                dict(status="pending_resource_bound", unattempted_starts=128),
            )
        phase("postflight")
        costs["postflight"] = preflight(plan, postflight=True)
        local = root / "matched_start/RESULT.json"
        local_complete = local.exists() and read(local).get("status") == "completed"
        status.update(
            status=(
                "completed" if local_complete else "completed_endpoint_local_incomplete"
            ),
            phase="completed",
            endpoint_complete=True,
            matched_start_complete=local_complete,
        )
    except BaseException as error:
        status.update(
            status="failed", error=repr(error), traceback=traceback.format_exc()
        )
        raise
    finally:
        status["wall_s"] = time.perf_counter() - start
        if deadline:
            status["gpu_clock_elapsed_s"] = GPU_SECONDS_LIMIT - (deadline - time.time())
        write_json(root / "RESOURCES.json", costs)
        write_json(root / "STATUS.json", status)


if __name__ == "__main__":
    from experiments.role_loss_pilot.campaign import install_signal_handlers

    install_signal_handlers()
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("--evaluate", action="store_true")
    p.add_argument("--deadline", type=float)
    a = p.parse_args()
    evaluate(a.root, a.deadline) if a.evaluate else run(a.root)
