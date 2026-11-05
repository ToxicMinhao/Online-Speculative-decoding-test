"""First milestone only: native parity, real cache switching, and route pricing.

Sixteen original FIVE completions are the first READOUT responses. Sixteen
additional fit-only completions exercise forced five, static three and mixed
depths. Matched-start profiling runs at idle while each original request's KV
bytes still exist. There is no fitting or FINAL-panel execution here.
"""

import argparse
import json
import math
from pathlib import Path
import time
import traceback

from online_feedback.storage import write_json
from online_feedback.sglang_backend import SGLangDFlashBackend
from online_feedback.native import validate_generation
from .pricing import analyze_pricing, select_round_starts, ROUTES


MODES = ("FORCED5",) * 4 + ("THREE",) * 4 + ("ALTERNATE",) * 8


def generation_errors(request, record):
    """Validate source binding, native trajectory and costs for either route."""
    errors = []
    for field in ("prompt_ids", "stop_token_ids", "max_new_tokens"):
        if record.get(field) != request[field]:
            errors.append(field + "_differs_from_manifest")
    metadata, response = record["metadata"], record["response_ids"]
    trace = metadata["trace"]
    try:
        _, finish = validate_generation(
            request["prompt_ids"], record["prompt_ids"] + response, trace,
            max_new_tokens=request["max_new_tokens"], stop_tokens=request["stop_token_ids"],
        )
        if finish != record["finish_reason"]:
            errors.append("termination_does_not_match_native_tokens")
    except (ValueError, RuntimeError) as error:
        errors.append("native_trajectory: " + str(error))
    if type(metadata["verification_calls"]) is not int or metadata["verification_calls"] != len(trace):
        errors.append("verification_count_differs_from_trace")
    accepted = metadata["accepted_draft_tokens"]
    if (type(accepted) is not int or accepted < 0 or accepted > len(response)
            or accepted != sum(block["accepted_draft_tokens"] for block in trace)):
        errors.append("invalid_accepted_token_count")
    if metadata["committed_progress"] != len(response) - 1:
        errors.append("invalid_committed_progress")
    for field in ("request_latency_s", "native_generate_s"):
        value = metadata[field]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            errors.append("invalid_" + field)
    cache = metadata["capture"]["cache_maintenance_s"]
    if type(cache) not in (int, float) or not math.isfinite(cache) or cache < 0:
        errors.append("invalid_cache_maintenance_s")
    if metadata.get("feedback_capture_in_timed_request") is not False:
        errors.append("feedback_capture_must_be_disabled")
    return errors


def parity_errors(reference, candidate, *, forced_five):
    errors = []
    for field in ("prompt_ids", "response_ids", "finish_reason", "stop_token_ids", "max_new_tokens"):
        if reference.get(field) != candidate.get(field):
            errors.append(field + "_mismatch")
    if forced_five:
        old, new = reference["metadata"]["trace"], candidate["metadata"]["trace"]
        if (len(old) != len(new) or reference["metadata"]["verification_calls"]
                != candidate["metadata"]["verification_calls"]):
            errors.append("verification_count_mismatch")
        for index, (left, right) in enumerate(zip(old, new)):
            for field in ("positions", "anchor_token", "draft_tokens", "committed_tokens", "committed_progress"):
                if left[field] != right[field]:
                    errors.append(f"round_{index}_{field}_mismatch")
    return errors


def profile_errors(profile, prompt, ordinal):
    """Reject partial/invalid pricing immediately, before another generation."""
    errors = []
    expected = []
    for anchor_ordinal, start in enumerate(select_round_starts(prompt["observed_rounds"])):
        for repeat in range(4):
            rotation = (ordinal + anchor_ordinal + repeat) % 4
            for route in ROUTES[rotation:] + ROUTES[:rotation]:
                expected.append((prompt["request_id"], start, repeat, route))
    actual = [(r["request_id"], r["round_index"], r["repeat"], r["route"])
              for r in profile["measurements"]]
    if profile["status"] != "passed" or actual != expected:
        errors.append("profile_status_or_schedule_differs")
    checks = profile.get("restoration_checks", {})
    restoration = ("request_row_ids", "request_mapping", "allocator", "native_graph_inputs",
                   "attention_metadata", "cache_bytes")
    if (profile.get("restoration_status") != "passed"
            or any(checks.get(name) is not True for name in restoration)):
        errors.append("native_cache_restoration_failed_or_unverified")
    for row in profile["measurements"]:
        value = row["elapsed_ms"]
        if (row.get("valid") is not True or type(value) not in (int, float)
                or not math.isfinite(value) or value <= 0):
            errors.append("invalid_route_measurement")
    return errors


def require_weights(backend, checkpoint, path):
    receipt = backend.check_weights(checkpoint)
    write_json(path, receipt)
    if receipt.get("status") != "passed" or receipt.get("all_equal") is not True:
        raise RuntimeError("Retained K+58 native weight comparison failed")
    return receipt


def work_accounting(records, active=None):
    returned = len(records)
    return dict(planned_generations=32, returned_generations=returned,
                valid_generations=sum(r["valid"] for r in records),
                invalid_generations=sum(not r["valid"] for r in records),
                unknown_generations=int(active is not None),
                unattempted_generations=32 - returned - int(active is not None),
                readout_generations=sum(r["phase"] == "READOUT" for r in records),
                qualification_generations=sum(r["phase"] == "qualification" for r in records),
                optimizer_steps=0, fitting_presentations=0, final_generations=0)


def run(plan, root):
    from .preflight import preflight

    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "PLAN.json", plan)
    preflight(plan, root / "PREFLIGHT.json")
    manifests = json.loads(Path(plan["manifest_file"]).read_text())
    requests = manifests["manifests"]["READOUT"]["requests"][:16]
    rows, measurements, prompts, profiles = [], [], [], []
    backend, active, active_profile = None, None, None
    started = time.perf_counter()

    def status(state, **extra):
        write_json(root / "STATUS.json", dict(
            status=state, **work_accounting(rows, active),
            returned_route_measurements=len(measurements),
            active_profile=active_profile,
            wall_s=time.perf_counter() - started, **extra,
        ))

    def rpc(name, filename, **arguments):
        return backend._call("depth_rpc", name=name,
                             arguments=dict(outputfile=str(root / filename), **arguments))["result"]

    try:
        status("loading")
        backend = SGLangDFlashBackend(
            plan["target_path"], plan["checkpoint_path"],
            worker_python=plan["worker_python"], output_root=root / "native",
            serving_gpu="0", context_limit=4096, block_size=16,
            worker_module="experiments.block_depth.worker",
        )
        require_weights(backend, plan["checkpoint_path"], root / "WEIGHTS_BEFORE.json")
        for ordinal, request in enumerate(requests):
            references = {}
            for phase, mode in (("READOUT", "FIVE"), ("qualification", MODES[ordinal])):
                route = rpc("depth_mode", f"{ordinal:02d}_{phase}_mode.json", mode=mode,
                            audit=phase == "qualification")
                if route.get("status") != "passed":
                    raise RuntimeError("Native route activation/audit failed")
                label = f"{ordinal:02d}_{phase}_{mode}"
                active = dict(label=label, phase=phase, ordinal=ordinal, mode=mode)
                status("generating")
                record = backend.generate(
                    request["prompt_ids"], max_new_tokens=2048, request_id=request["request_id"],
                    session_id="block_depth_qualification", request_index=ordinal,
                    purpose="evaluation", save_to=root / f"{label}.json",
                )
                active = None
                write_json(root / f"{label}.json", record)
                errors = generation_errors(request, record)
                if phase == "qualification":
                    errors.extend(parity_errors(references["FIVE"], record, forced_five=mode == "FORCED5"))
                row = dict(label=label, phase=phase, mode=mode, valid=not errors,
                           errors=errors, request_id=request["request_id"],
                           verification_calls=record["metadata"]["verification_calls"],
                           output_tokens=len(record["response_ids"]),
                           request_latency_s=record["metadata"]["request_latency_s"])
                rows.append(row)
                write_json(root / "GENERATIONS.json", rows)
                write_json(root / f"{label}_PARITY.json", row)
                status("validating")
                if errors:
                    raise RuntimeError("Native output/proposal parity failed: " + repr(errors))
                if phase == "READOUT":
                    references["FIVE"] = record
                    prompt = dict(request_id=request["request_id"],
                                  observed_rounds=record["metadata"]["verification_calls"],
                                  request_latency_s=record["metadata"]["request_latency_s"],
                                  native_generate_s=record["metadata"]["native_generate_s"],
                                  cache_maintenance_s=record["metadata"]["capture"]["cache_maintenance_s"])
                    prompts.append(prompt)
                    write_json(root / "PRICING_PROMPTS.json", prompts)
                    active_profile = dict(request_id=request["request_id"], ordinal=ordinal,
                                          path=str(root / f"{ordinal:02d}_PROFILE.json"),
                                          planned_measurements=16 * len(select_round_starts(prompt["observed_rounds"])))
                    status("profiling")
                    profile = rpc("depth_profile", f"{ordinal:02d}_PROFILE.json",
                                  record=record, prompt_ordinal=ordinal)
                    profiles.append(profile)
                    measurements.extend(profile["measurements"])
                    write_json(root / "PRICING_MEASUREMENTS.json", measurements)
                    active_profile = None
                    errors = profile_errors(profile, prompt, ordinal)
                    if errors:
                        raise RuntimeError("Native matched-start reconstruction failed: " + repr(errors))
                else:
                    observed = rpc("depth_status", f"{ordinal:02d}_ROUTE_STATUS.json")
                    if observed.get("status") != "passed" or observed.get("errors"):
                        raise RuntimeError("Layer/cache route audit failed: " + repr(observed))
        result = analyze_pricing(measurements, prompts)
        write_json(root / "PRICING.json", result)
        require_weights(backend, plan["checkpoint_path"], root / "WEIGHTS_AFTER.json")
        write_json(root / "COSTS.json", dict(
            loading_s=backend.loading_s, wall_s=time.perf_counter() - started,
            profiling=[{k: v for k, v in p.items() if k not in ("measurements", "parity")} for p in profiles],
            optimizer_steps=0, fitting_started=False,
        ))
        if result["status"] != "complete":
            raise RuntimeError("Pricing accounting is incomplete or invalid")
        status("completed", economic_decision=result["decision"], stopped_before_fitting=True)
    except BaseException as error:
        if active_profile is not None:
            partial_path = Path(active_profile["path"])
            if partial_path.exists():
                partial = json.loads(partial_path.read_text())
                profiles.append(partial)
                measurements.extend(partial.get("measurements", []))
                write_json(root / "PRICING_MEASUREMENTS.json", measurements)
                active_profile["preserved_partial_receipt"] = str(partial_path)
                active_profile["partial_returned_measurements"] = len(partial.get("measurements", []))
                active_profile["partial_work_accounting"] = partial.get("accounting")
                if partial.get("accounting") is None:
                    active_profile["unknown_measurement_count_bounds"] = [
                        0, active_profile["planned_measurements"] - len(partial.get("measurements", []))]
            else:
                active_profile["unknown_measurement_count_bounds"] = [0, active_profile["planned_measurements"]]
        write_json(root / "FAILURE.json", dict(type=type(error).__name__, message=str(error),
                                               traceback=traceback.format_exc(), active=active,
                                               active_profile=active_profile))
        status("blocked", failure=str(error), stopped_before_fitting=True)
        raise
    finally:
        if backend is not None:
            backend.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.plan.read_text()), args.output)


if __name__ == "__main__":
    main()
