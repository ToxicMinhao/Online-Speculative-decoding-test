"""Finish only the eight unexecuted serving checks after the compiler repair."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import subprocess
import time
import traceback

from online_feedback.storage import write_json
from .launch import child_environment, run_qualification_child
from .references import file_identity
from .requalify import accounting
from .resources import CARD_SECONDS, ELAPSED_SECONDS, LIMITS
from .task_runner import clock_status, read, storage_boundary

PRIOR_FILES = (
    "CLOCK.json", "REQUALIFICATION.json", "CUMULATIVE_WORK.json", "sealed/PLAN.json",
    "qualification/A/STATUS.json", "qualification/B/STATUS.json", "qualification/CROSS_CARD.json",
    "serving_qualification/A/STATUS.json", "serving_qualification/A/runtime/worker.log",
    "serving_qualification/A/runtime/PHYSICAL_WORK.json",
)
SERVING_BUDGET = {key: 8 if key == "qualification_generation" else 0 for key in LIMITS}


def authenticate(plan):
    """Reuse passed training, resolving the known pre-request serving failure."""
    binding = plan["serving_continuation"]
    prior_root, root = Path(binding["prior_root"]), Path(plan["root"])
    if root.resolve().is_relative_to(prior_root.resolve()):
        raise ValueError("Serving continuation requires a separate new directory")
    for name in PRIOR_FILES:
        if file_identity(prior_root/name) != binding["files"][name]:
            raise ValueError("Prior qualification evidence changed: " + name)
    authorization = binding["authorization"]
    if file_identity(authorization["path"]) != authorization["identity"]:
        raise ValueError("Serving continuation authorization changed")
    authority = read(authorization["path"])
    expected = dict(scope="serving_qualification_only", prior_root=str(prior_root), new_root=str(root),
        new_generations=8, new_adam_steps=0, new_response_forwards=0, new_response_backwards=0,
        new_training_initializations=0, preserve_original_clock=True, main_campaign_authorized=False)
    if any(authority.get(key) != value for key, value in expected.items()):
        raise ValueError("Authority must permit only the eight remaining serving checks")
    prior_plan = read(prior_root/"sealed/PLAN.json")
    for key in ("compiler_root", "compiler_baseline_allocated_bytes", "compiler_growth_reserved_bytes",
                "target", "origin_export", "training_python", "worker_python", "devices", "runtime_contract"):
        if plan[key] != prior_plan[key]:
            raise ValueError("Continuation changed a qualified setting: " + key)
    for key in ("warmups", "warmup_references"):
        if file_identity(plan[key]) != file_identity(prior_plan[key]):
            raise ValueError("Continuation changed the fixed serving qualification inputs")
    previous = read(prior_root/"REQUALIFICATION.json")
    training = {slot: read(prior_root/"qualification"/slot/"STATUS.json") for slot in ("A", "B")}
    cross = read(prior_root/"qualification/CROSS_CARD.json")
    serving = read(prior_root/"serving_qualification/A/STATUS.json")
    if any(row["execution_commit"] != binding["execution_commit"]
           for row in (prior_plan, previous, cross, serving, *training.values())):
        raise ValueError("Prior qualification source identity differs")
    if (previous["status"] != "stopped" or previous["production_launched"]
            or previous["completed_phases"] != ["cpu_preflight", "training_A", "training_B"]
            or cross["status"] != "passed" or cross["attempted_optimizer_applications"] != 8):
        raise ValueError("Require the passed eight-case training qualification")
    for slot, steps in (("A", 5), ("B", 3)):
        report = training[slot]
        expected_work = dict(model_load=1, optimizer_step=steps,
                             response_forward=2+2*steps, response_backward=2+2*steps)
        work = report["physical_work"]
        if (report["status"] != "passed" or work["attempted"] != expected_work
                or work["completed"] != expected_work or work["failed"]
                or any(work.get("unknown", {}).values())):
            raise ValueError("Retained training work did not complete exactly")
    # The authenticated traceback locates this failure in the constructor,
    # before evaluate() creates its request ledger. Missing files alone are not
    # evidence of zero attempted generations.
    if (serving["status"] != "failed" or "SGLang startup failed" not in serving["error"]
            or "SGLangDFlashBackend(" not in serving["traceback"]
            or "__init__" not in serving["traceback"] or "qualification" in serving
            or "loading_s" in serving or (prior_root/"serving_qualification/A/calls").exists()
            or "c++: not found" not in (prior_root/"serving_qualification/A/runtime/worker.log").read_text()):
        raise ValueError("The prior failure is not proven to precede request execution")
    clock = read(prior_root/"CLOCK.json")
    if (clock["active"] or not all(math.isfinite(clock[k]) and clock[k] >= 0
            for k in ("first_model_load_epoch", "completed_card_s"))
            or not math.isclose(clock["completed_card_s"],
                sum(row["occupied_card_s"] for row in clock["finished"]), abs_tol=1e-9)):
        raise ValueError("Prior shared clock has active or unaccounted work")
    cumulative = read(prior_root/"CUMULATIVE_WORK.json")["cumulative"]
    expected_work = {key: 0 for key in LIMITS}
    expected_work.update(adam=10, response_forward=26, response_backward=26, training_initialization=3)
    if (any(cumulative[field] != expected_work for field in ("attempted", "completed"))
            or any(cumulative["failed"].values()) or any(cumulative["unknown"].values())):
        raise ValueError("Cumulative native training work is incomplete or differs")
    return dict(root=str(prior_root), execution_commit=binding["execution_commit"],
        files=binding["files"], authorization=authorization, training=training,
        clock=clock, work=expected_work, prior_complete_generation_attempts=0,
        generation_resolution="Authenticated constructor failure before evaluate; original generic lower-bound receipt retained",
        prior_graph_work=read(prior_root/"serving_qualification/A/runtime/PHYSICAL_WORK.json"))


def run(plan_path):
    plan_path = Path(plan_path)
    plan = read(plan_path)
    root = Path(plan["root"])
    if any((root/name).exists() for name in ("SERVING_CONTINUATION.json", "CLOCK.json", "LAUNCH.json")):
        raise FileExistsError("No automatic retry or reused serving directory")
    (root/"logs").mkdir(exist_ok=True)
    result = dict(status="cpu_preflight", execution_commit=plan["execution_commit"],
        scope="serving_qualification_only", automatic_retry=False, production_launched=False,
        started_phases=[], completed_phases=[], began_epoch=time.time())
    write_json(root/"SERVING_CONTINUATION.json", result)
    prior = None
    try:
        prior = authenticate(plan)
        write_json(root/"PRIOR_QUALIFICATION.json", prior)
        write_json(root/"CLOCK.json", prior["clock"])
        with (root/"logs/CPU_PREFLIGHT.log").open("x") as log:
            subprocess.run([plan["training_python"], "-u", "-m",
                "experiments.persistent_prefix_withholding.preflight", str(root)],
                cwd=plan["source_root"], env=child_environment(plan),
                stdout=log, stderr=subprocess.STDOUT, check=True)
        cpu = read(root/"CPU_PREFLIGHT.json")
        if cpu["status"] != "passed" or cpu["execution_commit"] != plan["execution_commit"]:
            raise ValueError("CPU preflight did not qualify the current source")
        result["completed_phases"].append("cpu_preflight")
        clock = clock_status(root)
        if clock["elapsed_s"] >= ELAPSED_SECONDS or clock["occupied_s"] >= CARD_SECONDS:
            raise TimeoutError("Original qualification clock exhausted")
        plan["gpu_deadline_monotonic"] = time.monotonic()+ELAPSED_SECONDS-clock["elapsed_s"]
        write_json(plan_path, plan)
        serving = {}
        for slot in ("A", "B"):
            result["storage"] = storage_boundary(plan)
            result["status"] = "serving_"+slot
            result["started_phases"].append("serving_"+slot)
            write_json(root/"SERVING_CONTINUATION.json", result)
            serving[slot] = run_qualification_child(plan_path, plan, "serving", slot)
            result["completed_phases"].append("serving_"+slot)
        from .serving_qualification import compare_serving_cards
        cross = compare_serving_cards(serving["A"], serving["B"])
        write_json(root/"serving_qualification/CROSS_CARD.json",
                   dict(cross, execution_commit=plan["execution_commit"]))
        from .admission import qualify_resources
        admission = qualify_resources(plan, prior["training"], serving)
        write_json(root/"RESOURCE_ADMISSION.json", admission)
        result.update(status="qualified" if admission["status"] == "passed" else "qualified_resource_blocked",
                      resource_admission=admission["status"])
    except BaseException as error:
        result.update(status="stopped", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        result["wall_s"] = time.time()-result["began_epoch"]
        if (root/"CLOCK.json").exists():
            result["clock"] = clock_status(root)
        write_json(root/"SERVING_CONTINUATION.json", result)
        try:
            work = accounting(root, prior, result["started_phases"])
            work["planned_retry"] = SERVING_BUDGET
            work["unattempted"] = {key: None if work["unattempted"][key] is None else
                SERVING_BUDGET[key]-work["attempt"]["attempted"][key] for key in LIMITS}
            write_json(root/"CUMULATIVE_WORK.json", work)
            if "cpu_preflight" in result["completed_phases"]:
                result["final_storage"] = storage_boundary(plan)
        except Exception as error:
            result.update(status="stopped", final_accounting_or_storage_error=repr(error))
        write_json(root/"SERVING_CONTINUATION.json", result)
    if result["status"] == "stopped":
        raise RuntimeError("Serving continuation failed final accounting or storage checks")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    run(parser.parse_args().plan)


if __name__ == "__main__":
    main()
