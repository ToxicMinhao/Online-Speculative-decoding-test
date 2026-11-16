"""One explicitly authorized qualification retry; never start production work."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import subprocess
import time
import traceback

from online_feedback.storage import write_json
from experiments.threatened_repair_replay.work import read_ledger
from .launch import child_environment, run_qualification_child, seed_production_work
from .references import file_identity
from .resources import CARD_SECONDS, ELAPSED_SECONDS, LIMITS
from .task_runner import clock_status, read, storage_boundary

RETRY_WORK = dict(adam=8, response_forward=20, response_backward=20,
                  training_initialization=2, qualification_generation=8)
PRIOR_WORK = dict(adam=2, response_forward=6, response_backward=6,
                  training_initialization=1, qualification_generation=0)
NATIVE_NAMES = dict(optimizer_step="adam", response_forward="response_forward",
                    response_backward="response_backward", model_load="training_initialization")
PRIOR_FILES = ("CLOCK.json", "LAUNCH.json", "qualification/A/STATUS.json",
               "qualification/A/PHYSICAL_WORK.jsonl", "sealed/PLAN.json")


def authenticate_prior(plan):
    """Bind the failed attempt, the narrowly authorized retry and the old clocks."""
    prior = plan["prior_qualification_attempt"]
    old_root, root = Path(prior["root"]), Path(plan["root"])
    if root.resolve().is_relative_to(old_root.resolve()):
        raise ValueError("Retry must use a separate directory outside the failed attempt")
    for name in PRIOR_FILES:
        if file_identity(old_root/name) != prior["files"][name]:
            raise ValueError("Prior qualification evidence changed: " + name)
    authorization = plan["qualification_retry_authorization"]
    if file_identity(authorization["path"]) != authorization["identity"]:
        raise ValueError("Qualification retry authorization changed")
    expected = dict(scope="qualification_only", prior_root=str(old_root), new_root=str(root),
        new_adam_steps=8, new_response_forwards=20, new_response_backwards=20,
        new_initializations=2, new_generations=8, preserve_original_clock=True,
        main_campaign_authorized=False)
    receipt = read(authorization["path"])
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("Retry authorization does not match this bounded attempt")
    old_plan, launch, status, clock = (read(old_root/name) for name in
        ("sealed/PLAN.json", "LAUNCH.json", "qualification/A/STATUS.json", "CLOCK.json"))
    if (launch["status"] != "stopped" or status["status"] != "failed"
            or launch["completed_phases"] != ["cpu_preflight"]
            or not status["no_production_training"] or not status["no_export"]
            or status["complete_generations"] != 0
            or any(row["execution_commit"] != prior["execution_commit"]
                   for row in (old_plan, launch, status))):
        raise ValueError("Prior attempt is not the preserved pre-production failure")
    if clock["active"] or (old_root/"CAMPAIGN.json").exists():
        raise ValueError("Prior execution has active leases or production work")
    if (not all(math.isfinite(clock[key]) and clock[key] >= 0 for key in
                ("first_model_load_epoch", "completed_card_s"))
            or not math.isclose(clock["completed_card_s"],
                sum(row["occupied_card_s"] for row in clock["finished"]), abs_tol=1e-9)):
        raise ValueError("Prior occupied-card accounting is incomplete")
    for key in ("compiler_root", "compiler_baseline_allocated_bytes", "compiler_growth_reserved_bytes"):
        if old_plan[key] != plan[key]:
            raise ValueError("Retry cannot reset the compiler storage baseline: " + key)
    native = read_ledger(old_root/"qualification/A/PHYSICAL_WORK.jsonl")
    expected_native = {old: PRIOR_WORK[new] for old, new in NATIVE_NAMES.items()}
    if (native["attempted"] != expected_native or native["completed"] != expected_native
            or native["failed"] or any(native["unknown"].values())
            or any(status["physical_work"][key] != native[key]
                   for key in ("attempted", "completed", "failed"))):
        raise ValueError("Prior physical work differs or contains unknown work")
    return dict(root=str(old_root), execution_commit=prior["execution_commit"],
                files=prior["files"], authorization=authorization, clock=clock,
                work={key: PRIOR_WORK.get(key, 0) for key in LIMITS})


def accounting(root, prior, started_phases):
    """Recover known work even when a child fails before its final status write."""
    work = {field: {key: 0 for key in LIMITS}
            for field in ("attempted", "completed", "failed", "unknown")}
    missing, unresolved_kinds = [], set()
    for phase in started_phases:
        kind, slot = phase.split("_")
        if kind == "training":
            path = root/"qualification"/slot/"PHYSICAL_WORK.jsonl"
            if not path.exists():
                missing.append(str(path))
                unresolved_kinds.update(NATIVE_NAMES.values())
                continue
            native = read_ledger(path)
            for field in work:
                for old, new in NATIVE_NAMES.items():
                    work[field][new] += native[field].get(old, 0)
        else:
            path = root/"serving_qualification"/slot/"calls/RESULT.json"
            if not path.exists():
                missing.append(str(path))
                unresolved_kinds.add("qualification_generation")
                continue
            calls = read(path)
            work["attempted"]["qualification_generation"] += calls["attempted_generations"]
            work["completed"]["qualification_generation"] += calls["returned_generations"]
            work["unknown"]["qualification_generation"] += calls["unknown_generations"]
    cumulative = {field: dict(counts) for field, counts in work.items()}
    if prior is not None:
        for field in ("attempted", "completed"):
            for key, value in prior["work"].items():
                cumulative[field][key] += value
    return dict(attempt=work, cumulative=cumulative if prior is not None else None,
        unattempted={key: None if key in unresolved_kinds else
                     RETRY_WORK.get(key, 0)-work["attempted"][key] for key in LIMITS},
        unresolved_sources=missing, counts_are_lower_bounds=bool(missing),
        planned_retry=RETRY_WORK, original_campaign_limits_unchanged=dict(LIMITS))


def run(plan_path):
    plan_path = Path(plan_path)
    plan = read(plan_path)
    root = Path(plan["root"])
    if any((root/name).exists() for name in ("LAUNCH.json", "REQUALIFICATION.json", "CLOCK.json")):
        raise FileExistsError("Use a new qualification directory; no automatic retry")
    (root/"logs").mkdir(exist_ok=True)
    receipt = dict(status="cpu_preflight", execution_commit=plan["execution_commit"],
        scope="qualification_only", began_epoch=time.time(), automatic_retry=False,
        production_launched=False, completed_phases=[], started_phases=[], storage_boundaries=[])
    write_json(root/"REQUALIFICATION.json", receipt)
    prior = None
    try:
        prior = authenticate_prior(plan)
        write_json(root/"PRIOR_QUALIFICATION.json", prior)
        # Copy the existing clock exactly. CPU preparation also consumes elapsed
        # time; neither the compiler baseline nor the failure's cost is reset.
        write_json(root/"CLOCK.json", prior["clock"])
        with (root/"logs/CPU_PREFLIGHT.log").open("x") as log:
            subprocess.run([plan["training_python"], "-u", "-m",
                "experiments.persistent_prefix_withholding.preflight", str(root)],
                cwd=plan["source_root"], env=child_environment(plan),
                stdout=log, stderr=subprocess.STDOUT, check=True)
        cpu = read(root/"CPU_PREFLIGHT.json")
        if cpu["status"] != "passed" or cpu["execution_commit"] != plan["execution_commit"]:
            raise ValueError("CPU preflight did not pass for the execution source")
        receipt["completed_phases"].append("cpu_preflight")
        clock = clock_status(root)
        if clock["occupied_s"] >= CARD_SECONDS or clock["elapsed_s"] >= ELAPSED_SECONDS:
            raise TimeoutError("Original qualification clock has exhausted the shared cap")
        plan["gpu_deadline_monotonic"] = time.monotonic()+ELAPSED_SECONDS-clock["elapsed_s"]
        write_json(plan_path, plan)
        from .qualification import compare_cards
        from .serving_qualification import compare_serving_cards
        reports = {}
        for kind, compare in (("training", compare_cards), ("serving", compare_serving_cards)):
            reports[kind] = {}
            for slot in ("A", "B"):
                phase = kind+"_"+slot
                receipt["storage_boundaries"].append(dict(before=phase, **storage_boundary(plan)))
                receipt.update(status=phase)
                receipt["started_phases"].append(phase)
                write_json(root/"REQUALIFICATION.json", receipt)
                reports[kind][slot] = run_qualification_child(plan_path, plan, kind, slot)
                receipt["completed_phases"].append(phase)
            left, right = reports[kind]["A"], reports[kind]["B"]
            cross = (compare(left["qualification"], right["qualification"])
                     if kind == "training" else compare(left, right))
            directory = "qualification" if kind == "training" else "serving_qualification"
            write_json(root/directory/"CROSS_CARD.json", dict(cross, execution_commit=plan["execution_commit"]))
        seed_production_work(root, reports["training"], reports["serving"])
        # Retain the helper's exact per-attempt receipt; the separate cumulative
        # receipt includes the failed attempt and cannot enable production.
        (root/"WORK.json").rename(root/"REQUALIFICATION_WORK.json")
        from .admission import qualify_resources
        admission = qualify_resources(plan, reports["training"], reports["serving"])
        write_json(root/"RESOURCE_ADMISSION.json", admission)
        receipt.update(status="qualified" if admission["status"] == "passed"
                       else "qualified_resource_blocked", resource_admission=admission["status"])
    except BaseException as error:
        receipt.update(status="stopped", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        receipt["wall_s"] = time.time()-receipt["began_epoch"]
        if (root/"CLOCK.json").exists():
            receipt["clock"] = clock_status(root)
        if "cpu_preflight" in receipt["completed_phases"]:
            try:
                receipt["final_storage"] = storage_boundary(plan)
            except Exception as error:
                receipt.update(status="stopped", final_storage_error=repr(error))
        write_json(root/"REQUALIFICATION.json", receipt)
        try:
            work = accounting(root, prior, receipt["started_phases"])
            write_json(root/"CUMULATIVE_WORK.json", work)
        except Exception as error:
            write_json(root/"CUMULATIVE_WORK.json", dict(status="unknown", error=repr(error),
                original_records_preserved=True, prior=prior, started_phases=receipt["started_phases"]))
            receipt.update(status="stopped", accounting_error=repr(error))
            write_json(root/"REQUALIFICATION.json", receipt)
    if receipt["status"] == "stopped":
        raise RuntimeError("Qualification stopped during final storage or accounting checks")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    run(parser.parse_args().plan)


if __name__ == "__main__":
    main()
