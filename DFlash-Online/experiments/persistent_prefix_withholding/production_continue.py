"""Authenticated production-only continuation; qualification is never repeated."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import subprocess
import time
import traceback

from online_feedback.storage import write_json
from .launch import child_environment
from .references import file_identity
from .resources import LIMITS, ELAPSED_SECONDS
from .task_runner import read, storage_boundary

ORCHESTRATION_FILES = frozenset(
    "experiments/persistent_prefix_withholding/"+name for name in (
        "campaign.py", "resources.py", "launch.py", "preflight.py", "prepare.py",
        "task_runner.py", "references.py", "production_continue.py", "continue_serving.py",
        "requalify.py", "admission.py", "report.py"))
SCIENTIFIC_VALUES = ("target", "origin_state", "origin_export", "origin_config",
    "native_n7_state", "native_n7_export", "training_python", "worker_python", "devices",
    "runtime_contract", "event7_feature_paths", "compiler_root", "compiler_baseline_allocated_bytes",
    "compiler_growth_reserved_bytes")
SCIENTIFIC_FILES = ("event7_schedule", "event8_inputs", "data_panel", "missing_references",
    "reused_references", "warmups", "warmup_references", "expected_ancestry",
    "installed_runtime_identities", "n7_parity_receipt")


def bound_json(node):
    path = node["path"] if "path" in node else node["manifest"]
    if file_identity(path) != node["identity"]:
        raise ValueError("Continuation evidence changed: " + path)
    return read(path)


def verify_source_differences(plan, manifest):
    """Check the reviewed exact diff, retaining unchanged numerical components."""
    difference = bound_json(manifest["source_differences"])
    current = read(plan["source_manifest"])
    if (difference["status"] != "reviewed" or difference["execution_commit"] != plan["execution_commit"]
            or current["dirty"] or current["commit"] != plan["execution_commit"]):
        raise ValueError("Require reviewed source differences and the clean execution source")
    for role in ("training", "serving"):
        source = difference["sources"][role]
        previous = bound_json(source["manifest"])
        if previous["dirty"] or previous["commit"] != source["execution_commit"]:
            raise ValueError("Historical source manifest is not clean or bound")
        expected_commit = manifest["receipts"]["A" if role == "training" else "serving"]["execution_commit"]
        if source["execution_commit"] != expected_commit:
            raise ValueError("Source comparison does not use the actual qualification commit")
        before, after = previous["files"], current["files"]
        changes = {name: dict(before=before.get(name), after=after.get(name))
                   for name in sorted(set(before)|set(after)) if before.get(name) != after.get(name)}
        if changes != difference["changes"][role]:
            raise ValueError("Actual source changes differ from the reviewed diff")
        for name in changes:
            if not (name in ORCHESTRATION_FILES
                    or name.startswith("tests/test_persistent_prefix_") and name.endswith(".py")
                    or name.startswith("reports/dflash_persistent_prefix_withholding_20261115/")):
                raise ValueError("Unqualified scientific component changed: " + name)
    return difference


def authenticate_continuation(plan):
    manifest = bound_json(plan["production_continuation"])
    if (manifest["scope"] != "production_only" or manifest["execution_commit"] != plan["execution_commit"]
            or manifest["binding"] != plan["binding"]):
        raise ValueError("Production continuation identity differs")
    raw = manifest["retained_evidence"]
    paths = {node["path"] for node in raw}
    required = set()
    for slot in ("A", "B"):
        base = Path(manifest["receipts"][slot]["path"]).parent
        required.update(str(base/name) for name in ("PHYSICAL_WORK.jsonl", "cases/RESULT.json", "startup/EXACT_RESTORATION.json"))
        base = Path(manifest["receipts"]["serving_"+slot]["path"]).parent/"calls"
        required.add(str(base/"RESULT.json"))
        required.update(str(base/f"q{slot}_r{repeat}_w{request}.json") for repeat in range(2) for request in range(2))
    if len(paths) != len(raw) or not required <= paths:
        raise ValueError("Required raw qualification receipts are missing or duplicated")
    for node in raw:
        if file_identity(node["path"]) != node["identity"]:
            raise ValueError("Retained raw qualification evidence changed: " + node["path"])
    original, serving_plan = (bound_json(manifest[key]) for key in ("original_plan", "serving_plan"))
    if original["binding"] != plan["binding"]:
        raise ValueError("Operational amendment must not recompute the scientific binding")
    for historical in (original, serving_plan):
        if any(plan[key] != historical[key] for key in SCIENTIFIC_VALUES):
            raise ValueError("Qualified numerical/model/runtime settings changed")
        if any(file_identity(plan[key]) != file_identity(historical[key]) for key in SCIENTIFIC_FILES):
            raise ValueError("A sealed scientific input changed")
        if any(plan["input_identities"].get(path) != identity
               for path, identity in historical["input_identities"].items()):
            raise ValueError("Required immutable input identity was dropped or changed")
        gate, old_gate = read(plan["reference_provenance"]), read(historical["reference_provenance"])
        for value, owner in ((gate, plan), (old_gate, historical)):
            path = owner["reference_provenance"]
            if (file_identity(path) != owner["input_identities"][path]
                    or value["status"] != "passed"
                    or value["new_reference_route"] != "fixed_original_K6_native_greedy_target_verifier"
                    or value["no_target_only_AR_claim"] is not True):
                raise ValueError("Reference route provenance is not authenticated")
        for key in ("reference_reuse_receipt", "reference_provenance"):
            if file_identity(gate[key]) != file_identity(old_gate[key]):
                raise ValueError("Retained reference authentication evidence changed")
        if not set(historical["allocation_roots"]) <= set(plan["allocation_roots"]):
            raise ValueError("Earlier allocated roots cannot be removed from accounting")
        prior_sources = {row["source_root"]: row for row in plan["prior_source_manifests"]}
        for row in historical["prior_source_manifests"]:
            if prior_sources.get(row["source_root"]) != row:
                raise ValueError("Prior allocated source identity was dropped or changed")
    for arm in ("N", "E", "S"):
        if file_identity(plan["configs"][arm]) != file_identity(original["configs"][arm]):
            raise ValueError("Qualified serialized algorithm contract changed")
    receipts = {}
    for label in ("A", "B", "cross_card", "serving", "serving_A", "serving_B"):
        node = manifest["receipts"][label]
        receipt = bound_json(node)
        if receipt["status"] != "passed" or receipt["execution_commit"] != node["execution_commit"]:
            raise ValueError("Unpassed or misidentified qualification receipt: " + label)
        if label in plan["qualification_receipts"] and plan["qualification_receipts"][label] != node["path"]:
            raise ValueError("Production gate points at another qualification receipt")
        receipts[label] = receipt
    if (original["execution_commit"] != manifest["receipts"]["A"]["execution_commit"]
            or serving_plan["execution_commit"] != manifest["receipts"]["serving"]["execution_commit"]):
        raise ValueError("Qualification plans and receipts identify different executions")
    verify_source_differences(plan, manifest)
    amendment = bound_json(manifest["amendment"])
    if amendment != plan["operational_amendment"]:
        raise ValueError("Operational amendment differs from the authorized bound document")
    from .resources import effective_limits
    limits = effective_limits(plan)
    work_record = bound_json(manifest["work"])
    cumulative = work_record["cumulative"]
    expected = {key: 0 for key in LIMITS}
    expected.update(adam=10, response_forward=26, response_backward=26,
                    training_initialization=3, qualification_generation=8)
    if (any(cumulative[key] != expected for key in ("attempted", "completed"))
            or any(cumulative["failed"].values()) or any(cumulative["unknown"].values())
            or work_record.get("unresolved_sources") or work_record.get("counts_are_lower_bounds")
            or work_record.get("pending")):
        raise ValueError("Unknown, failed or different cumulative work cannot be discarded")
    clock = bound_json(manifest["clock"])
    if (clock["active"] or clock["first_model_load_epoch"] != 1794802603.911516
            or not math.isclose(clock["completed_card_s"], 435.09486651420593, abs_tol=1e-9)
            or not math.isclose(clock["completed_card_s"],
                sum(row["occupied_card_s"] for row in clock["finished"]), abs_tol=1e-9)):
        raise ValueError("Original shared clock is incomplete or was reset")
    from .campaign import production_expectations
    missing = read(plan["missing_references"])
    missing = missing["requests"] if isinstance(missing, dict) else missing
    remaining = production_expectations(len(missing))
    if len(missing) != 24 or any(expected[key]+remaining[key] > limits[key] for key in limits):
        raise ValueError("Complete unchanged production work exceeds effective limits")
    return dict(status="passed", execution_commit=plan["execution_commit"], binding=plan["binding"],
        manifest=plan["production_continuation"], receipts=manifest["receipts"],
        original_serving_binding=serving_plan["binding"], clock=clock, cumulative=cumulative,
        remaining=remaining, effective_limits=limits, qualification_repeated=False,
        source_differences=manifest["source_differences"]), receipts


def qualified_receipt_matches(plan, label, receipt):
    """The ordinary gate permits old commits only through this authenticated seam."""
    proof = bound_json(plan["qualification_reuse"])
    if (proof["status"] != "passed" or proof["execution_commit"] != plan["execution_commit"]
            or proof["binding"] != plan["binding"] or proof["manifest"] != plan["production_continuation"]):
        return False
    manifest = bound_json(proof["manifest"])
    node = manifest["receipts"][label]
    return (node == proof["receipts"][label] and node["path"] == plan["qualification_receipts"][label]
            and receipt == bound_json(node) and receipt["execution_commit"] == node["execution_commit"])


def run(plan_path):
    plan_path = Path(plan_path)
    plan = read(plan_path)
    root = Path(plan["root"])
    if any((root/name).exists() for name in ("PRODUCTION_CONTINUATION.json", "CAMPAIGN.json", "WORK.json", "CLOCK.json")):
        raise FileExistsError("Production continuation already started; no automatic retry")
    (root/"logs").mkdir(exist_ok=True)
    status = dict(status="authenticating", execution_commit=plan["execution_commit"],
                  started_epoch=time.time(), qualification_repeated=False, automatic_retry=False)
    write_json(root/"PRODUCTION_CONTINUATION.json", status)
    try:
        proof, receipts = authenticate_continuation(plan)
        write_json(root/"QUALIFICATION_REUSE.json", proof)
        plan["qualification_reuse"] = dict(path=str(root/"QUALIFICATION_REUSE.json"),
                                          identity=file_identity(root/"QUALIFICATION_REUSE.json"))
        write_json(root/"CLOCK.json", proof["clock"])
        work = dict(proof["cumulative"], limits=proof["effective_limits"], scientific_binding=plan["binding"],
                    operational_amendment=plan["operational_amendment"], pending={}, next_operation=0,
                    origin="Authenticated complete cumulative qualification including failed original attempt")
        write_json(root/"INITIAL_WORK.json", work)
        write_json(root/"WORK.json", work)
        plan["gpu_deadline_monotonic"] = time.monotonic()+ELAPSED_SECONDS-(time.time()-proof["clock"]["first_model_load_epoch"])
        write_json(plan_path, plan)
        with (root/"logs/CPU_PREFLIGHT.log").open("x") as log:
            subprocess.run([plan["training_python"], "-u", "-m", "experiments.persistent_prefix_withholding.preflight", str(root)],
                cwd=plan["source_root"], env=child_environment(plan), stdout=log, stderr=subprocess.STDOUT, check=True)
        cpu = read(plan["cpu_gate"])
        if cpu["status"] != "passed" or cpu["execution_commit"] != plan["execution_commit"]:
            raise ValueError("Current CPU input/runtime preflight is not passed")
        status["storage"] = storage_boundary(plan)
        from .admission import qualify_resources
        admission = qualify_resources(plan, {slot:receipts[slot] for slot in ("A", "B")},
            {slot:receipts["serving_"+slot] for slot in ("A", "B")})
        write_json(plan["resource_admission"], admission)
        if admission["status"] != "passed":
            raise RuntimeError("Fresh full-work resource admission failed; original clocks retained")
        plan.update(costs=admission["costs"], task_memory_bytes=admission["task_memory_bytes"], task_vram_bytes=admission["task_vram_bytes"])
        write_json(plan_path, plan)
        status["status"] = "production"
        write_json(root/"PRODUCTION_CONTINUATION.json", status)
        from .campaign import run_campaign
        status["campaign"] = run_campaign(plan_path)
        status["status"] = "completed"
    except BaseException as error:
        status.update(status="stopped", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        status["wall_s"] = time.time()-status["started_epoch"]
        write_json(root/"PRODUCTION_CONTINUATION.json", status)
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
