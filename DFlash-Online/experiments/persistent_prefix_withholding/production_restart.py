"""Authenticate and import the explicitly authorized interrupted-production retry.

Only S8, the missing canonical references and final evaluation remain executable.
Completed tasks are copied as compact evidence, never rerun. The old interrupted
forward remains charged and unresolved; importing a run cannot repair its ledger.
This module performs CPU file operations only and never constructs a model.
"""
from __future__ import annotations

from copy import deepcopy
import argparse
import math
from pathlib import Path
import subprocess
import time
import traceback

from online_feedback.storage import write_json
from . import production_continue as continuation
from .references import file_identity
from .resources import LIMITS, effective_limits
from .staging import reference_readiness
from .task_runner import read

SCOPE = "restart_S8_references_final_only"
FAILED_COMMIT = "db6e1221d8d1b234d9d07e246025d1afd5dc30d5"
REUSED_TASKS = ("E7", "S7", "N_collect", "E_collect", "S_collect", "E8", "N8")
REMAINING_TASKS = ("S8", "references", "final")
COMPACT_SUFFIXES = frozenset((".json", ".jsonl", ".yaml", ".yml"))


def restart_expected_work():
    """Return fresh work and distinct cumulative attempted/completed endpoints."""
    prior_completed = dict.fromkeys(LIMITS, 0)
    prior_completed.update(adam=545, response_forward=1106, response_backward=1106,
        training_initialization=8, qualification_generation=8,
        feedback_generation=384, teacher_preparation=384)
    prior_attempted = dict(prior_completed, response_forward=1107)
    remaining = dict.fromkeys(LIMITS, 0)
    remaining.update(adam=128, response_forward=258, response_backward=258,
        training_initialization=1, reference_generation=24,
        warmup_generation=72, scored_generation=1152)
    return dict(prior_attempted=prior_attempted, prior_completed=prior_completed,
        remaining=remaining,
        expected_attempted={k: prior_attempted[k]+remaining[k] for k in LIMITS},
        expected_completed={k: prior_completed[k]+remaining[k] for k in LIMITS})


def compact_files(root):
    """Exact compact tree for copying; bulky tensors remain at retained paths."""
    root = Path(root)
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Completed task evidence must not contain symlinks")
        if path.is_file() and path.suffix in COMPACT_SUFFIXES:
            rows.append(dict(relative_path=str(path.relative_to(root)), identity=file_identity(path)))
    return rows


def _source_check(plan, manifest, failed_plan):
    previous = continuation.bound_json(manifest["failed_source"])
    current = read(plan["source_manifest"])
    difference = continuation.bound_json(manifest["source_differences"])
    if (previous["commit"] != FAILED_COMMIT or previous["dirty"]
            or current["commit"] != plan["execution_commit"] or current["dirty"]
            or file_identity(manifest["failed_source"]["path"]) != file_identity(failed_plan["source_manifest"])):
        raise ValueError("Restart source must extend the exact clean failed production source")
    changes = {name: dict(before=previous["files"].get(name), after=current["files"].get(name))
               for name in sorted(set(previous["files"]) | set(current["files"]))
               if previous["files"].get(name) != current["files"].get(name)}
    if (difference.get("status") != "reviewed"
            or difference.get("execution_commit") != plan["execution_commit"]
            or difference.get("failed_execution_commit") != FAILED_COMMIT
            or difference.get("changes") != changes):
        raise ValueError("Restart source changes differ from the reviewed exact diff")
    allowed = continuation.ORCHESTRATION_FILES | {
        "experiments/persistent_prefix_withholding/production_restart.py",
        "experiments/persistent_prefix_withholding/staging.py"}
    for name in changes:
        if not (name in allowed or name.startswith("tests/test_persistent_prefix_") and name.endswith(".py")
                or name.startswith("reports/dflash_persistent_prefix_withholding_20261115/")):
            raise ValueError("Restart changes an unqualified scientific component: "+name)


def _scientific_check(plan, failed):
    if plan["binding"] != failed["binding"]:
        raise ValueError("Restart changed the scientific binding")
    for key in continuation.SCIENTIFIC_VALUES:
        if plan[key] != failed[key]:
            raise ValueError("Restart changed a scientific setting: "+key)
    for key in continuation.SCIENTIFIC_FILES:
        if file_identity(plan[key]) != file_identity(failed[key]):
            raise ValueError("Restart changed a sealed scientific input: "+key)
    for arm in ("N", "E", "S"):
        if file_identity(plan["configs"][arm]) != file_identity(failed["configs"][arm]):
            raise ValueError("Restart changed an arm configuration")
    if any(plan["input_identities"].get(path) != identity for path, identity in failed["input_identities"].items()):
        raise ValueError("Restart dropped or changed an immutable input identity")
    if not set(failed["allocation_roots"]) | {failed["root"]} <= set(plan["allocation_roots"]):
        raise ValueError("Failed-run storage must remain charged")
    before = {row["source_root"]: row for row in failed["prior_source_manifests"]}
    after = {row["source_root"]: row for row in plan["prior_source_manifests"]}
    if any(after.get(key) != value for key, value in before.items()):
        raise ValueError("Prior source identities cannot be dropped")


def authenticate_restart(plan):
    """Validate durable evidence before copying anything or scheduling work."""
    manifest = continuation.bound_json(plan["production_restart"])
    if (manifest.get("scope") != SCOPE or manifest.get("execution_commit") != plan["execution_commit"]
            or manifest.get("binding") != plan["binding"]):
        raise ValueError("Restart identity or authorized scope differs")
    failed = continuation.bound_json(manifest["failed_plan"])
    if failed["execution_commit"] != FAILED_COMMIT or Path(plan["root"]).resolve() == Path(failed["root"]).resolve():
        raise ValueError("Restart must preserve the failed run in a separate destination")
    _scientific_check(plan, failed)
    _source_check(plan, manifest, failed)
    original_proof, receipts = continuation.authenticate_continuation(failed)
    retained_proof = continuation.bound_json(manifest["failed_qualification_reuse"])
    if original_proof != retained_proof:
        raise ValueError("Retained full qualification proof differs from authenticated evidence")
    campaign = continuation.bound_json(manifest["failed_campaign"])
    work = continuation.bound_json(manifest["failed_work"])
    clock = continuation.bound_json(manifest["failed_clock"])
    expected = restart_expected_work()
    pending = {"3453": dict(task="S8", label="forward_loss", counts=dict(response_forward=1))}
    if (campaign["status"] != "failed" or set(campaign["completed"]) != set(REUSED_TASKS)
            or work["attempted"] != expected["prior_attempted"]
            or work["completed"] != expected["prior_completed"] or work["pending"] != pending
            or work["next_operation"] != 3454 or work["scientific_binding"] != plan["binding"]):
        raise ValueError("Failed production work or completed-task accounting differs")
    if (clock["active"] or clock["first_model_load_epoch"] != original_proof["clock"]["first_model_load_epoch"]
            or clock["finished"][:len(original_proof["clock"]["finished"])] != original_proof["clock"]["finished"]
            or not math.isclose(clock["completed_card_s"], 2849.2045271396637, abs_tol=1e-9)
            or not math.isclose(clock["completed_card_s"], sum(x["occupied_card_s"] for x in clock["finished"]), abs_tol=1e-9)):
        raise ValueError("Restart must preserve every failed/completed clock lease and the original epoch")
    amendment = continuation.bound_json(manifest["amendment"])
    if amendment != plan["operational_amendment"]:
        raise ValueError("Restart work amendment differs from its explicit authorization")
    elapsed = continuation.bound_json(manifest["elapsed_amendment"])
    authority = continuation.bound_json(manifest["authorization"])
    from .resources import effective_elapsed_limit
    if (elapsed != plan["elapsed_time_amendment"] or effective_elapsed_limit(plan) != 21600
            or authority.get("scope") != "one_explicit_packaging_repair_restart"
            or authority.get("failed_execution_commit") != FAILED_COMMIT
            or authority.get("first_model_load_epoch") != clock["first_model_load_epoch"]
            or authority.get("elapsed_cap_s") != 21600 or authority.get("occupied_card_cap_s") != 14400
            or authority.get("retry_tasks") != list(REMAINING_TASKS)
            or authority.get("reused_tasks") != list(REUSED_TASKS)
            or authority.get("failed_forward_remains_unknown") != 1
            or authority.get("qualification_repeated") is not False
            or authority.get("scientific_changes") is not False
            or authority.get("additional_retry_authorized") is not False):
        raise ValueError("Restart elapsed/work scope differs from the explicit user decision")
    limits = effective_limits(plan)
    if any(expected["expected_attempted"][key] > limits[key] for key in LIMITS):
        raise ValueError("The complete restart does not fit its operational work limits")
    tasks = manifest["completed_tasks"]
    if set(tasks) != set(REUSED_TASKS):
        raise ValueError("Only the seven completed tasks may be reused")
    required_artifacts = set()
    for name, node in tasks.items():
        root = Path(node["root"])
        if root.resolve() != (Path(failed["root"])/"tasks"/name).resolve() or compact_files(root) != node["files"]:
            raise ValueError("Completed task compact evidence changed: "+name)
        status, result = read(root/"STATUS.json"), read(root/"RESULT.json")
        if status["status"] != "completed" or result["status"] != "completed" or status["execution_commit"] != FAILED_COMMIT:
            raise ValueError("Attempted or foreign task cannot be imported as complete")
        if name in ("S7", "E8", "N8"):
            endpoint = result["endpoint"]
            required_artifacts.update((endpoint["state"], str(Path(endpoint["export"])/"model.safetensors")))
        if name == "S_collect":
            if len(result["feature_paths"]) != 128 or result["feature_paths"] != read(root/"FEATURES.json")["feature_paths"]:
                raise ValueError("S8 retry requires its exact complete own-feedback pool")
            required_artifacts.update(result["feature_paths"])
    artifacts = manifest["retained_artifacts"]
    if not required_artifacts <= artifacts.keys():
        raise ValueError("Required own-state, endpoint or S-feedback tensors are unbound")
    for path, identity in artifacts.items():
        if file_identity(path) != identity:
            raise ValueError("Retained production tensor changed: "+path)
    readiness = reference_readiness(plan)
    initial_work = deepcopy(work)
    initial_work.update(pending={}, inherited_pending=deepcopy(pending),
        inherited_pending_source=plan["production_restart"], next_operation=3454,
        limits=limits, operational_amendment=plan["operational_amendment"],
        origin="Failed production retained exactly; explicit S8 restart from complete S7")
    initial_work["unknown"] = dict.fromkeys(LIMITS, 0)
    initial_work["unknown"]["response_forward"] = 1
    return dict(status="passed", scope=SCOPE, execution_commit=plan["execution_commit"],
        binding=plan["binding"], manifest=plan["production_restart"], failed_execution_commit=FAILED_COMMIT,
        failed_root=failed["root"], failed_work=work, inherited_pending=pending, failed_pending=pending,
        initial_work=initial_work, clock=clock, source_manifest=plan["production_restart"],
        remaining=expected["remaining"], expected_attempted=expected["expected_attempted"],
        expected_completed=expected["expected_completed"],
        reused_tasks=list(REUSED_TASKS), remaining_tasks=list(REMAINING_TASKS),
        completed_tasks=tasks, retained_artifacts=artifacts, work=expected, effective_limits=limits,
        qualification=original_proof, receipts=original_proof["receipts"],
        reference_readiness=readiness, elapsed_time_amendment=elapsed, authorization=manifest["authorization"],
        qualification_repeated=False, feedback_recollected=False,
        no_step919_resume_claim=True), receipts


def import_restart(plan):
    """Import once; interruption requires review, never implicit replay/recovery."""
    root = Path(plan["root"])
    markers = ("RESTART_IMPORT.json", "RESTART_IMPORT_PENDING.json", "WORK.json", "CLOCK.json", "CAMPAIGN.json", "tasks")
    if any((root/name).exists() for name in markers):
        raise FileExistsError("Restart destination already contains an attempted import or execution")
    proof, receipts = authenticate_restart(plan)
    write_json(root/"RESTART_IMPORT_PENDING.json", dict(status="copying", execution_commit=plan["execution_commit"]))
    for name, node in proof["completed_tasks"].items():
        for row in node["files"]:
            source = Path(node["root"])/row["relative_path"]
            payload = source.read_bytes()
            if file_identity(source) != row["identity"]:
                raise ValueError("Completed evidence changed during import")
            destination = root/"tasks"/name/row["relative_path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as stream:
                stream.write(payload)
            if file_identity(destination) != row["identity"]:
                raise ValueError("Copied completed-task evidence differs")
    work = proof["initial_work"]
    write_json(root/"INITIAL_WORK.json", work)
    write_json(root/"WORK.json", work)
    write_json(root/"CLOCK.json", proof["clock"])
    write_json(root/"RESTART_IMPORT.json", proof)
    plan["restart_import"] = dict(path=str(root/"RESTART_IMPORT.json"), identity=file_identity(root/"RESTART_IMPORT.json"))
    (root/"RESTART_IMPORT_PENDING.json").unlink()
    return proof


def qualified_receipt_matches(plan, label, receipt):
    proof = continuation.bound_json(plan["restart_import"])
    if (proof["status"] != "passed" or proof["execution_commit"] != plan["execution_commit"]
            or proof["binding"] != plan["binding"] or proof["manifest"] != plan["production_restart"]):
        return False
    node = proof["receipts"][label]
    return (node["path"] == plan["qualification_receipts"][label]
            and receipt == continuation.bound_json(node) and receipt["execution_commit"] == node["execution_commit"])


def run(plan_path):
    """The tracked launch path: authenticate, preflight, admit, then remaining work."""
    from .admission import qualify_resources
    from .campaign import run_campaign
    from .launch import child_environment
    from .resources import effective_elapsed_limit
    from .task_runner import storage_boundary

    plan_path = Path(plan_path)
    plan = read(plan_path)
    root = Path(plan["root"])
    if any((root/name).exists() for name in
           ("RESTART_STATUS.json", "RESTART_IMPORT.json", "RESTART_IMPORT_PENDING.json", "WORK.json", "CLOCK.json", "CAMPAIGN.json", "tasks")):
        raise FileExistsError("Restart destination has already been attempted; no automatic retry")
    (root/"logs").mkdir(exist_ok=True)
    status = dict(status="checking_staged_payload", execution_commit=plan["execution_commit"],
        started_epoch=time.time(), qualification_repeated=False, automatic_retry=False)
    write_json(root/"RESTART_STATUS.json", status)
    try:
        status["reference_readiness"] = reference_readiness(plan)
        proof = import_restart(plan)
        plan["gpu_deadline_monotonic"] = time.monotonic()+effective_elapsed_limit(plan)-(time.time()-proof["clock"]["first_model_load_epoch"])
        write_json(plan_path, plan)
        with (root/"logs/CPU_PREFLIGHT.log").open("x") as log:
            subprocess.run([plan["training_python"], "-u", "-m", "experiments.persistent_prefix_withholding.preflight", str(root)],
                cwd=plan["source_root"], env=child_environment(plan), stdout=log, stderr=subprocess.STDOUT, check=True)
        cpu = read(plan["cpu_gate"])
        if cpu["status"] != "passed" or cpu["execution_commit"] != plan["execution_commit"]:
            raise ValueError("Current CPU input/runtime preflight failed")
        status["storage"] = storage_boundary(plan)
        receipts = {key: continuation.bound_json(node) for key, node in proof["receipts"].items()}
        admission = qualify_resources(plan, {s: receipts[s] for s in ("A", "B")},
            {s: receipts["serving_"+s] for s in ("A", "B")})
        write_json(plan["resource_admission"], admission)
        if admission["status"] != "passed":
            raise RuntimeError("Remaining fixed work fails resource admission; original time retained")
        plan.update(costs=admission["costs"], task_memory_bytes=admission["task_memory_bytes"],
                    task_vram_bytes=admission["task_vram_bytes"])
        write_json(plan_path, plan)
        status["status"] = "production"
        write_json(root/"RESTART_STATUS.json", status)
        status["campaign"] = run_campaign(plan_path)
        status["status"] = "completed"
    except BaseException as error:
        status.update(status="stopped", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        status["wall_s"] = time.time()-status["started_epoch"]
        write_json(root/"RESTART_STATUS.json", status)
        from .report import assemble_report
        try:
            assemble_report(plan_path)
        except Exception as error:
            write_json(root/"REPORT_FAILURE.json", dict(error=repr(error), raw_records_preserved=True))
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    run(parser.parse_args().plan)


if __name__ == "__main__":
    main()
