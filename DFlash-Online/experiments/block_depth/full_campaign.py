"""Frozen K+58: fit one readout, calibrate one gate, execute the sealed FINAL.

The parent owns no CUDA model. Native serving stays on GPU0; each readout group
uses an isolated GPU1 process which exits before calibration or FINAL timing.
Completed work is never repeated automatically after an interrupted operation.
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
from online_feedback.sglang_backend import SGLangDFlashBackend
from experiments.ok_coding.campaign import WorkLedger, read_work, require_complete_work
from experiments.kw_v1.evaluate import append_json
from experiments.kplus_full_gsm8k.preflight import timing_barrier
from experiments.kplus_full_gsm8k.storage import retire_redundant_features, _retire
from .preflight import identity
from .qualify import require_weights
from .endpoint import endpoint_schedule, validate_execution, analyze_endpoint, write_analysis
from .gate import fit_gate


def read(path):
    return json.loads(Path(path).read_text())


def rpc(backend, name, output, **arguments):
    return backend._call("depth_rpc", name=name,
                         arguments=dict(outputfile=str(output), **arguments))["result"]


def lean_record(backend, request, arm, repeat, output, *, gate=False):
    """Preserve the native result before validating it; no timed token tracing."""
    started = time.perf_counter()
    native = backend._call("generate_lean", prompt_ids=request["prompt_ids"],
        max_new_tokens=2048, stop_token_ids=request["stop_token_ids"], mode=arm,
        trace_mode="gate" if gate else "none", trace_path=str(output.with_suffix(".native.json")))
    elapsed = time.perf_counter() - started
    tokens, meta = native["output_ids"], dict(native["trace"])
    meta.update(request_latency_s=elapsed, native_generate_s=native["native_generate_s"],
        worker_request_latency_s=native["request_latency_s"],
        feedback_capture_in_timed_request=False, behavior=arm,
        accepted_count_definition="native committed progress minus one per verification")
    record = dict(request_id=request["request_id"], record_id=request["request_id"],
        request_index=request["panel_ordinal"], arm=arm, repeat=repeat,
        prompt_ids=request["prompt_ids"], response_ids=tokens,
        stop_token_ids=request["stop_token_ids"], max_new_tokens=2048,
        finish_reason="eos" if tokens and tokens[-1] in request["stop_token_ids"] else "length",
        metadata=meta, native_meta_info=native["meta_info"])
    write_json(output, record)
    return record


def train_group(root, plan, group, pool, previous):
    folder = root / f"readout/group_{group:03d}/trainer"
    folder.mkdir(parents=True, exist_ok=False)
    command = [plan["training_python"], "-u", "-m", "experiments.block_depth.readout_training",
        "--config", plan["training_config"], "--output", str(folder),
        "--pool", str(pool), "--group", str(group)]
    if previous:
        command += ["--resume-state", str(previous)]
    receipt = dict(status="starting", command=command, automatic_retry=False)
    started, process, validated = time.perf_counter(), None, False
    try:
        with (folder / "worker.log").open("x") as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES="1"), start_new_session=True)
            receipt.update(status="running", pid=process.pid)
            write_json(folder / "LAUNCH.json", receipt)
            code = process.wait()
        if code:
            raise RuntimeError(f"Readout group {group} stopped with exit {code}: {folder/'worker.log'}")
        result = read(folder / "RESULT.json")
        if result["optimizer_steps"] != 128 or result["response_presentations"] != 256:
            raise ValueError("Readout worker did not complete its fixed group")
        if not result["frozen_parameter_versions_unchanged"] or result["boundary_check"]["status"] != "passed":
            raise ValueError("Readout fitting changed a frozen parameter")
        validated = True
        return folder, result
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=45)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        receipt.update(exit_code=process.returncode if process else None,
            status="completed" if validated else "failed",
            wall_s=time.perf_counter() - started)
        write_json(folder / "LAUNCH.json", receipt)


def retire_consumed_group(root, folder, pool, trainer):
    """Keep compact tokens/anchors/exposure and validated own continuation state."""
    result = read(trainer / "RESULT.json")
    state = trainer / "training_state.pt"
    export = trainer / "events" / folder.name / "export/readout.safetensors"
    if (result["optimizer_steps"] != 128 or not state.is_file() or not export.is_file()
            or not result["frozen_parameter_versions_unchanged"] or result["boundary_check"]["status"] != "passed"):
        raise ValueError("Group successor evidence is incomplete; retain its feature pool")
    for path, expected in pool["identities"].items():
        if identity(path) != expected:
            raise ValueError("Consumed feature changed before retirement")
    return _retire(root, folder / "POOL_RETIREMENT.json",
        [(Path(path), "consumed_readout_teacher_features") for path in pool["features"]],
        [folder / "POOL.json", trainer / "RESULT.json"] +
        [Path(path).with_suffix(".json") for path in pool["features"]],
        reason="This completed group is qualified; compact feedback, anchors and own readout state remain. Teacher tensors are retired, not archived.",
        retained={"state": str(state), "state_identity": identity(state), "export": str(export),
                  "export_identity": identity(export)})


def run(plan):
    from .full_plan import preflight, resource_check

    root = Path(plan["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    if (root / "STATUS.json").exists():
        raise FileExistsError("Do not retry or overwrite a prior full campaign")
    started = time.perf_counter()
    preflight(plan, root / "PREFLIGHT.json")
    parts = read(plan["manifest_file"])["manifests"]
    canonicals = read(plan["canonical_file"])
    limits = dict(readout_generations=368, readout_reconstructions=16, teacher_preparations=384,
                  gate_generations=128, calibration_requests=128,
                  canonical_generations=384-len(canonicals), final_generations=2304)
    ledger = WorkLedger(root / "WORK.jsonl", limits)
    backend, active, records = None, None, []
    costs, trained = {}, []

    def status(phase, **extra):
        write_json(root / "STATUS.json", dict(status=phase, active=active,
            work=read_work(root / "WORK.jsonl", limits), trained_groups=len(trained),
            final_returned=len(records), wall_s=time.perf_counter()-started, **extra))

    try:
        status("loading")
        backend = SGLangDFlashBackend(plan["target_path"], plan["checkpoint_path"],
            worker_python=plan["worker_python"], output_root=root / "native", serving_gpu="0",
            context_limit=4096, block_size=16, worker_module="experiments.block_depth.worker")
        costs["native_loading_s"] = backend.loading_s
        require_weights(backend, plan["checkpoint_path"], root / "WEIGHTS_BEFORE.json")
        rpc(backend, "depth_mode", root / "READOUT_MODE.json", mode="FIVE", audit=False)
        previous = None
        for group in range(1, 4):
            resource_check(root, plan, f"READOUT_{group}")
            folder = root / f"readout/group_{group:03d}"
            folder.mkdir(parents=True)
            paths, preparations = [], []
            group_started = time.perf_counter()
            for request in parts["READOUT"]["requests"][(group-1)*128:group*128]:
                ordinal = request["panel_ordinal"]
                active = dict(phase="READOUT", ordinal=ordinal)
                status("collecting_readout")
                outfile = folder / "responses" / (request["request_id"] + ".json")
                if ordinal < 16:
                    source = read(Path(plan["qualification_root"]) / f"{ordinal:02d}_READOUT_FIVE.json")
                    if source["prompt_ids"] != request["prompt_ids"]:
                        raise ValueError("Retained READOUT input differs from sealed manifest")
                    replay = ledger.call("readout_reconstructions", rpc, backend, "depth_reconstruct",
                        folder / f"reconstruction_{ordinal:02d}.json", record=source)
                    record = replay["record"]
                    write_json(outfile, record)
                else:
                    record = ledger.call("readout_generations", backend.generate, request["prompt_ids"],
                        request_id=request["request_id"], session_id="depth_readout", request_index=ordinal,
                        max_new_tokens=2048, purpose="serving", save_to=outfile)
                if record["record_id"] != request["request_id"] or len(record["prompt_ids"])+len(record["response_ids"]) > 2272:
                    raise ValueError("READOUT identity/context changed")
                prepared = ledger.call("teacher_preparations", backend.prepare_feedback, record,
                    folder / "loop/features" / (request["request_id"] + ".pt"))
                if prepared["status"] != "prepared":
                    raise ValueError("No eligible readout feedback; preserve without replacing the request")
                preparations.append(prepared)
                paths.append(prepared["feature_path"])
                write_json(folder / "PREPARATIONS.json", preparations)
                retire_redundant_features(record, prepared, root, folder)
                active = None
            pool = dict(features=paths, identities={path: identity(path) for path in paths}, immutable=True)
            write_json(folder / "POOL.json", pool)
            costs[f"readout_group_{group}_collection_s"] = time.perf_counter()-group_started
            status("fitting_readout", group=group)
            trainer, result = train_group(root, plan, group, folder / "POOL.json", previous)
            trained.append(result)
            previous = trainer / "training_state.pt"
            retire_consumed_group(root, folder, pool, trainer)
            status("readout_group_completed", group=group)
        sidecar = trainer / "events/group_003/export/readout.safetensors"
        require_weights(backend, plan["checkpoint_path"], root / "WEIGHTS_AFTER_READOUT.json")
        rpc(backend, "depth_load", root / "READOUT_LOAD.json", sidecar_path=str(sidecar), gate_path=None)
        resource_check(root, plan, "GATE")
        gate_rows, behaviors = [], []
        stage_started = time.perf_counter()
        for ordinal, request in enumerate(parts["GATE"]["requests"]):
            active = dict(phase="GATE", ordinal=ordinal)
            status("calibrating_gate")
            arm = "FIVE" if ordinal % 2 == 0 else "THREE"
            record = ledger.call("gate_generations", lean_record, backend, request, arm, 0,
                root / "gate/responses" / (request["request_id"] + ".json"), gate=True)
            # Native GATE traces are checked at every sampled matched start before fitting.
            candidate = dict(record, metadata={**record["metadata"], "trace": []})
            errors = validate_execution(candidate, request, record)
            write_json(root / "gate/validity" / (request["request_id"] + ".json"), errors)
            if errors:
                raise ValueError("Invalid GATE behavior: " + repr(errors))
            calibration = ledger.call("calibration_requests", rpc, backend, "depth_calibrate",
                root / "gate/calibration" / (request["request_id"] + ".json"), record=record)
            if calibration["status"] != "passed" or calibration["restoration_status"] != "passed":
                raise ValueError("GATE matched-prefix reconstruction/cache restoration failed")
            gate_rows.extend(calibration["rows"])
            behaviors.append(dict(request_id=request["request_id"], **record["metadata"]))
            active = None
        write_json(root / "gate/ROWS.json", gate_rows)
        write_json(root / "gate/BEHAVIORS.json", behaviors)
        gate = fit_gate(gate_rows, behaviors, plan["continuation_cost_ms"])
        write_json(root / "gate/GATE.json", gate)
        rpc(backend, "depth_load", root / "GATE_LOAD.json", sidecar_path=str(sidecar), gate_path=str(root / "gate/GATE.json"))
        costs["gate_collection_reconstruction_fit_s"] = time.perf_counter()-stage_started
        # Missing references are untimed native target-verified greedy completions.
        # They are never feedback and do not enter fitting or calibration.
        stage_started = time.perf_counter()
        for request in parts["FINAL"]["requests"]:
            if request["request_id"] in canonicals:
                continue
            active = dict(phase="canonical", request_id=request["request_id"])
            status("preparing_missing_canonicals")
            record = ledger.call("canonical_generations", lean_record, backend, request, "FIVE", 0,
                root / "canonicals" / (request["request_id"] + ".json"))
            errors = validate_execution(record, request, record)
            if errors:
                raise ValueError("Invalid native canonical reference: " + repr(errors))
            canonicals[request["request_id"]] = record
            active = None
        write_json(root / "CANONICALS.json", canonicals)
        costs["missing_canonical_s"] = time.perf_counter()-stage_started
        routes = rpc(backend, "depth_status", root / "FROZEN_ROUTES.json")
        if {row["graph"] for row in routes["graph_setup"]} != {"static_three", "prefix", "shallow", "deep"}:
            raise ValueError("All four route graphs must be built before FINAL")
        write_json(root / "FINAL_FREEZE.json", dict(sidecar=identity(sidecar), gate=identity(root / "gate/GATE.json"),
            canonicals=identity(root / "CANONICALS.json"), routes=routes, no_more_fitting=True,
            code_commit=plan["code_commit"], continuation_cost_ms=plan["continuation_cost_ms"]))
        resource_check(root, plan, "FINAL")
        write_json(root / "TIMING_BARRIER.json", timing_barrier(backend.worker.pid))
        stage_started = time.perf_counter()
        requests = parts["FINAL"]["requests"]
        schedule = endpoint_schedule(requests)
        write_json(root / "FINAL_SCHEDULE.json", schedule)
        repeat_counts = {}
        for serial, item in enumerate(schedule):
            active = dict(phase="FINAL", serial=serial, **item)
            status("evaluating_final")
            request = requests[item["request_ordinal"]]
            record = ledger.call("final_generations", lean_record, backend, request, item["arm"], item["repeat"],
                root / "final/raw" / f"{serial:04d}.json")
            records.append(record)
            errors = validate_execution(record, request, canonicals[request["request_id"]])
            key = (request["request_id"], item["arm"])
            counts = tuple(record["metadata"][k] for k in ("verification_calls", "accepted_draft_tokens", "shallow_rounds", "deep_rounds"))
            if key in repeat_counts and repeat_counts[key] != counts:
                errors.append("repeat_count_mismatch")
            repeat_counts[key] = counts
            append_json(root / "final/VALIDITY.jsonl", dict(serial=serial, valid=not errors, errors=errors))
            active = None
            if errors:
                raise ValueError("FINAL correctness failure: " + repr(errors))
        costs["final_endpoint_wall_s"] = time.perf_counter()-stage_started
        require_complete_work(read_work(root / "WORK.jsonl", limits), limits)
        require_weights(backend, plan["checkpoint_path"], root / "WEIGHTS_AFTER.json")
        if identity(sidecar) != read(root / "FINAL_FREEZE.json")["sidecar"]:
            raise ValueError("Frozen readout artifact changed")
        if identity(root / "gate/GATE.json") != read(root / "FINAL_FREEZE.json")["gate"]:
            raise ValueError("Frozen gate artifact changed")
        result = analyze_endpoint(records, requests, canonicals)
        write_analysis(root / "report", result)
        status("completed", decision=result["decision"])
    except BaseException as error:
        write_json(root / "FAILURE.json", dict(error=repr(error), traceback=traceback.format_exc(), active=active))
        status("failed", error=repr(error))
        if records:
            work = read_work(root / "WORK.jsonl", limits)
            write_analysis(root / "report", analyze_endpoint(records, parts["FINAL"]["requests"], canonicals,
                unknown_executions=work["unknown"]["final_generations"]))
        raise
    finally:
        costs["campaign_wall_s"] = time.perf_counter()-started
        write_json(root / "COSTS.json", costs)
        write_json(root / "READOUT_RESULTS.json", trained)
        if backend is not None:
            backend.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    run(read(parser.parse_args().plan))
