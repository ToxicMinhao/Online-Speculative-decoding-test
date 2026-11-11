"""Bounded successor control: qualify, train, compare, then stop."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

from online_feedback.storage import write_json
from experiments.ok_coding.storage import identity
from experiments.survival_head.data import read, verify_shared_packets
from experiments.survival_confirmation.runtime import (
    capacity,
    physical_gpu,
    finish_worker,
)
from experiments.kplus_full_gsm8k.preflight import timing_barrier
from experiments.o_half_matched_lr.archive import validate_record
from .evaluation import METHODS, SESSION


def preflight(plan, *, postflight=False):
    root = Path(plan["root"])
    code = Path(__file__).resolve().parents[2]
    record = dict(status="running")
    start = time.perf_counter()
    from .contract import SERVING
    from .report import ANALYSIS

    if (
        plan["schema"] != "kplus_successor_shuffle_v1"
        or plan["serving"] != SERVING
        or plan["analysis"] != ANALYSIS
        or plan["budget"]["timed_completions"] != 6 * plan["panel_size"]
        or plan["budget"]["optimizer_steps"] != 256
        or not plan["true_reuse"]["enabled"]
        or plan["budget"]["parent_draft_blocks"] != 64
    ):
        raise ValueError("Sealed successor experiment contract changed")
    try:
        seal = read(root / "sealed/SEAL.json")
        for path, expected in seal["files"].items():
            if identity(path) != expected:
                raise ValueError("Sealed input changed: " + path)
        source = read(plan["source_manifest"])
        if (
            source["commit"] != plan["execution_source"]["commit"]
            or source["dirty"]
            or source["source_path"] != str(code)
        ):
            raise ValueError("Require pinned clean execution source")
        for relative, expected in source["files"].items():
            if identity(code / relative) != expected:
                raise ValueError("Source changed: " + relative)
        for relative, target in source.get("symlinks", {}).items():
            if (
                not (code / relative).is_symlink()
                or str((code / relative).readlink()) != target
            ):
                raise ValueError("Source symlink changed")
        if (
            identity(plan["origin_state"]) != plan["origin_state_identity"]
            or identity(plan["origin_weights"]) != plan["origin_weights_identity"]
        ):
            raise ValueError("Shared parent state changed")
        verify_shared_packets(read(plan["pool"]))
        for key in ("pool", "original_schedule", "baseline_result"):
            if identity(plan[key]) != plan[key + "_identity"]:
                raise ValueError("Shared training binding changed")
        from experiments.osd_components.prepare import (
            verify_files,
            verify_native_runtime,
        )

        model = read(root / "sealed/MODEL_PIN.json")
        for name in ("target", "draft"):
            verify_files(plan[name], model[name]["files"], exact_files=True)
        import platform, torch, transformers

        actual = dict(
            python=platform.python_version(),
            torch=torch.__version__,
            transformers=transformers.__version__,
        )
        if (
            actual != read(root / "sealed/TRAINING_PIN.json")["environment"]
            or torch.cuda.is_initialized()
        ):
            raise ValueError(
                "Qualified training environment changed or CPU preflight initialized CUDA"
            )
        record["training_runtime"] = actual
        from experiments.survival_head.training import validate_origin
        from experiments.ok_coding.training import _export_equal

        state = torch.load(
            plan["origin_state"], map_location="cpu", weights_only=False, mmap=True
        )
        record["complete_origin"] = validate_origin(
            state,
            dataset="gsm8k",
            original_config_text=Path(plan["original_config"]).read_text(),
        )
        _export_equal(state, plan["origin_export"])
        del state
        record["runtime"] = verify_native_runtime(
            plan["worker_python"], read(root / "sealed/BOUND_RUNTIME_PIN.json")
        )
        record["capacity"] = capacity(plan)
        record["physical_gpu"] = physical_gpu(plan)
        record["quiet"] = timing_barrier()
        record["source_files"] = len(source["files"])
        record["status"] = "passed"
    except BaseException as error:
        record.update(status="failed", error=repr(error))
        raise
    finally:
        record["wall_s"] = time.perf_counter() - start
        write_json(
            root / ("POSTFLIGHT.json" if postflight else "PREFLIGHT.json"), record
        )
    return record


def launch(plan, module, arguments, receipt_path):
    path = Path(receipt_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [plan["training_python"], "-u", "-m", module, plan["root"], *arguments]
    start = time.perf_counter()
    value = dict(command=command, status="running")
    with path.with_suffix(".log").open("x") as log:
        process = subprocess.Popen(
            command,
            cwd=plan["execution_source"]["source_path"],
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(plan["training_gpu"])),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        value["pid"] = process.pid
        write_json(path, value)
        try:
            code = process.wait()
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait()
            value.update(
                returncode=process.returncode,
                wall_s=time.perf_counter() - start,
                status="completed" if process.returncode == 0 else "failed",
            )
            write_json(path, value)
    if code:
        raise RuntimeError(
            "Subprocess failed; preserve evidence, no retry: " + str(path)
        )
    return value


def verify_exports(plan, checkpoints):
    results = {}
    for arm, item in checkpoints.items():
        export = Path(item["checkpoint_path"])
        if identity(export / "model.safetensors") != item["weights_identity"]:
            raise ValueError("Export changed: " + arm)
        results[arm] = item["weights_identity"]
    return results


def prepare_references(plan, checkpoints):
    from online_feedback.sglang_backend import SGLangDFlashBackend

    root = Path(plan["root"]) / "references"
    root.mkdir(exist_ok=False)
    requests = read(plan["panel_manifest"])["requests"]
    refs = read(plan["reused_references"])
    missing = [r for r in requests if r["request_id"] not in refs]
    if [r["request_id"] for r in missing] != plan["missing_reference_ids"]:
        raise ValueError("Reference inventory changed")
    receipt = dict(
        status="running",
        planned_generations=len(missing),
        attempted_generations=0,
        returned_generations=0,
        valid_generations=0,
        invalid_generations=0,
        reused=len(refs),
        loading_s=0.0,
        generation_wall_s=0.0,
    )
    backend = None
    failure = None
    start = time.perf_counter()
    try:
        if missing:
            timing_barrier()
            capacity(plan)
            backend = SGLangDFlashBackend(
                plan["target"],
                checkpoints["TRUE"]["checkpoint_path"],
                worker_python=plan["worker_python"],
                output_root=root / "native",
                serving_gpu=str(plan["serving_gpu"]),
                context_limit=4096,
                block_size=16,
            )
            receipt["loading_s"] = backend.loading_s
        for request in missing:
            timing_barrier(backend.worker.pid)
            label = "canonical_" + str(request["panel_ordinal"]).zfill(4)
            path = root / (label + ".json")
            receipt["attempted_generations"] += 1
            write_json(root / "STATUS.json", receipt)
            timer = time.perf_counter()
            try:
                raw = backend.generate(
                    request["prompt_ids"],
                    max_new_tokens=2048,
                    request_id=label,
                    session_id=SESSION,
                    request_index=request["panel_ordinal"],
                    purpose="evaluation",
                    save_to=path,
                )
            finally:
                receipt["generation_wall_s"] += time.perf_counter() - timer
            receipt["returned_generations"] += 1
            ref = dict(
                prompt_ids=raw["prompt_ids"],
                response_ids=raw["response_ids"],
                finish_reason=raw["finish_reason"],
                verified=True,
                binding_digest=plan["canonical_binding"],
                source="fresh_pinned_native_verifier_before_timing",
                source_record_path=str(path),
                source_record_identity=identity(path),
            )
            errors = validate_record(
                raw, request, raw["checkpoint_id"], ref, label=label, session=SESSION
            )
            if read(raw["metadata"]["native_trace_path"]) != raw["metadata"]["capture"]:
                errors.append("native_sidecar_mismatch")
            write_json(
                root / (label + "_VALIDATION.json"),
                dict(status="invalid" if errors else "valid", errors=errors),
            )
            if errors:
                receipt["invalid_generations"] += 1
                raise ValueError("Canonical reference invalid: " + repr(errors))
            receipt["valid_generations"] += 1
            refs[request["request_id"]] = ref
            write_json(root / "PARTIAL_CANONICAL.json", refs)
        write_json(root / "CANONICAL.json", refs)
        receipt.update(
            status="completed", canonical_identity=identity(root / "CANONICAL.json")
        )
    except BaseException as error:
        failure = error
        receipt.update(status="failed", error=repr(error))
        raise
    finally:
        receipt.update(
            unattempted_generations=len(missing) - receipt["attempted_generations"],
            unknown_generations=receipt["attempted_generations"]
            - receipt["returned_generations"],
            returned_without_validation=receipt["returned_generations"]
            - receipt["valid_generations"]
            - receipt["invalid_generations"],
        )
        finish_worker(backend, receipt, root / "STATUS.json", start, failure)
    return receipt


def parent_probes(plan):
    from online_feedback.sglang_backend import SGLangDFlashBackend
    from .matched_start import observation

    root = Path(plan["root"]) / "matched_start"
    root.mkdir(exist_ok=False)
    panel = read(plan["panel_manifest"])["requests"]
    subset = read(plan["matched_panel"])
    ids = [row["request_id"] for row in subset["requests"]]
    requests = {r["request_id"]: r for r in panel}
    refs = read(plan["references"])
    receipt = dict(
        status="running",
        planned_requests=len(ids),
        attempted_requests=0,
        returned_requests=0,
        validated_requests=0,
        draft_blocks=0,
        maximum_draft_blocks=64,
        complete_response_generations=0,
        untimed=True,
    )
    backend = None
    failure = None
    start = time.perf_counter()
    rows = {}
    try:
        if len(ids) != 64 or len(set(ids)) != 64:
            raise ValueError("Require sealed64 unique parent starts")
        timing_barrier()
        capacity(plan)
        backend = SGLangDFlashBackend(
            plan["target"],
            plan["origin_export"],
            worker_python=plan["worker_python"],
            output_root=root / "native",
            serving_gpu=str(plan["serving_gpu"]),
            context_limit=4096,
            block_size=16,
            worker_module="experiments.successor_shuffle.parent_worker",
        )
        receipt["loading_s"] = backend.loading_s
        for index, rid in enumerate(ids):
            request = requests[rid]
            ref = refs[rid]
            trace = root / f"parent_{index:03d}_trace.json"
            receipt["attempted_requests"] += 1
            write_json(root / "STATUS.json", receipt)
            result = backend._call(
                "generate",
                prompt_ids=request["prompt_ids"],
                max_new_tokens=2048,
                capture=False,
                trace_path=str(trace),
                stop_token_ids=backend.stop_tokens,
            )
            receipt["returned_requests"] += 1
            record = dict(
                record_id=rid,
                checkpoint_id=plan["origin_export"],
                prompt_ids=request["prompt_ids"],
                response_ids=result["output_ids"],
                metadata=dict(trace=result["trace"]["blocks"], capture=result["trace"]),
                diagnostic_stop="after_first_native_round",
                max_new_tokens=2048,
                stop_token_ids=backend.stop_tokens,
            )
            write_json(root / f"parent_{index:03d}.json", record)
            blocks = record["metadata"]["trace"]
            receipt["draft_blocks"] += len(blocks)
            if len(blocks) > 1 or receipt["draft_blocks"] > 64:
                raise ValueError("Parent one-block budget exceeded")
            if (
                record["response_ids"]
                != ref["response_ids"][: len(record["response_ids"])]
            ):
                raise ValueError("Parent canonical prefix mismatch")
            if not blocks and record["response_ids"][-1] not in backend.stop_tokens:
                raise ValueError("Missing parent draft round")
            rows[rid] = observation(
                record, request, ref, checkpoint=plan["origin_export"]
            )
            receipt["validated_requests"] += 1
            write_json(root / "PARTIAL_PARENT.json", rows)
        write_json(root / "PARENT.json", rows)
        receipt["status"] = "completed"
    except BaseException as error:
        failure = error
        receipt.update(status="failed", error=repr(error))
        raise
    finally:
        receipt.update(
            unattempted_requests=len(ids) - receipt["attempted_requests"],
            unknown_requests=receipt["attempted_requests"]
            - receipt["returned_requests"],
            returned_without_validation=receipt["returned_requests"]
            - receipt["validated_requests"],
        )
        finish_worker(backend, receipt, root / "STATUS.json", start, failure)
    return receipt


def evaluate(plan, checkpoints):
    from online_feedback.sglang_backend import SGLangDFlashBackend
    from .evaluation import evaluate_stage, audit_stage

    root = Path(plan["root"])
    out = root / "evaluation"
    paths = {a: checkpoints[a]["checkpoint_path"] for a in METHODS}
    requests = read(plan["panel_manifest"])["requests"]
    refs = read(plan["references"])
    receipt = dict(status="running")
    backend = None
    failure = None
    start = time.perf_counter()
    try:
        timing_barrier()
        capacity(plan)
        verify_exports(plan, checkpoints)
        backend = SGLangDFlashBackend(
            plan["target"],
            paths["TRUE"],
            worker_python=plan["worker_python"],
            output_root=root / "evaluation_native",
            serving_gpu=str(plan["serving_gpu"]),
            context_limit=4096,
            block_size=16,
        )
        receipt["loading_s"] = backend.loading_s

        def boundary():
            return dict(
                storage=capacity(plan),
                quiescence=timing_barrier(backend.worker.pid),
                physical_gpu=physical_gpu(plan),
            )

        args = dict(
            canonicals=refs,
            sealed_n=plan["panel_size"],
            schedule=plan["evaluation_schedule"],
            physical_gpu=plan["serving_gpu_identity"],
        )
        evaluate_stage(backend, requests, paths, out, boundary_check=boundary, **args)
        receipt["exports"] = verify_exports(plan, checkpoints)
        audit = audit_stage(out, requests, paths, evidence_root=root, **args)
        write_json(out / "AUDIT.json", audit)
        if audit["status"] != "passed":
            raise ValueError("Raw serving audit failed")
        receipt["status"] = "completed"
    except BaseException as error:
        failure = error
        receipt.update(status="failed", error=repr(error))
        raise
    finally:
        finish_worker(backend, receipt, out / "WORKER.json", start, failure)
    return receipt


def run(root):
    from .qualification import summarize

    root = Path(root).resolve()
    plan = read(root / "sealed/PLAN.json")
    if (root / "STATUS.json").exists():
        raise FileExistsError("Preserve existing execution; no automatic retry")
    started = time.perf_counter()
    status = dict(
        status="running",
        phase="preflight",
        execution_commit=plan["execution_source"]["commit"],
    )
    costs = {}

    def phase(name):
        status["phase"] = name
        write_json(root / "STATUS.json", status)

    try:
        costs["preflight"] = preflight(plan)
        phase("qualification")
        costs["qualification"] = {}
        for case in ("TRUE", "IDENTITY", "SHUFFLED"):
            costs["qualification"][case] = launch(
                plan,
                "experiments.successor_shuffle.qualification",
                ["--case", case],
                root / "qualification" / f"{case}_EXEC.json",
            )
        admission = summarize(root)
        write_json(root / "qualification/ADMISSION.json", admission)
        if admission["status"] != "passed":
            raise ValueError("Native qualification failed")
        phase("training")
        costs["training"] = {}
        checkpoints = dict(plan["checkpoints"])
        results = {}
        for arm in ("K", "SHUFFLED"):
            costs["training"][arm] = launch(
                plan,
                "experiments.successor_shuffle.training",
                ["--train", arm],
                root / "campaign_001" / f"{arm}_EXEC.json",
            )
            result = read(root / "campaign_001/gsm8k/event_005" / arm / "RESULT.json")
            results[arm] = result
            checkpoints[arm] = dict(
                checkpoint_path=result["checkpoint_path"],
                weights_identity=result["weights_identity"],
                result_path=str(
                    root / "campaign_001/gsm8k/event_005" / arm / "RESULT.json"
                ),
            )
        write_json(root / "CHECKPOINTS.json", checkpoints)
        from .campaign import manipulation_summary

        manipulation = manipulation_summary(
            plan, results["SHUFFLED"], checkpoints["TRUE"]["weights_identity"]
        )
        if (
            not manipulation["labels_changed"]
            or not manipulation["final_export_changed"]
        ):
            status.update(
                status="completed_no_actuation",
                phase="stopped_before_redundant_timing",
                reason="no_changed_assignments_or_identical_TRUE_SHUFFLED_exports",
            )
            return
        phase("canonical_references")
        costs["references"] = prepare_references(plan, checkpoints)
        phase("matched_parent_starts")
        costs["matched_start"] = parent_probes(plan)
        phase("timed_evaluation")
        costs["evaluation"] = evaluate(plan, checkpoints)
        phase("postflight")
        costs["postflight"] = preflight(plan, postflight=True)
        status.update(status="completed", phase="completed")
    except BaseException as error:
        status.update(
            status="failed", error=repr(error), traceback=traceback.format_exc()
        )
        raise
    finally:
        status["wall_s"] = time.perf_counter() - started
        write_json(root / "RESOURCES.json", costs)
        write_json(root / "STATUS.json", status)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root")
    a = p.parse_args()

    def stop(signum, frame):
        raise KeyboardInterrupt("Supervisor signal " + str(signum))

    signal.signal(signal.SIGTERM, stop)
    run(a.root)
