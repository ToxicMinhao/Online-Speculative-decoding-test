"""Explicit evaluation-only continuation after the publication-budget failure.

Preparation is CPU-only. Execution requires a separate command; neither path
calls training, reference preparation, qualification or canonical generation.
The failed execution and its seals are read-only inputs, never rewritten.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import signal
import time
import traceback

from online_feedback.storage import write_json
from experiments.verification_transition.data import identity, read
from . import runtime

SCHEMA = "retention-e7-evaluation-continuation-v1"
REPAIR_FILES = frozenset({
    "experiments/recent_repair_retention/fixed_starts.py",
    "experiments/recent_repair_retention/resume_evaluation.py",
    "experiments/recent_repair_retention/report.py",
    "experiments/recent_repair_retention/README.md",
    "tests/test_recent_repair_retention_runtime.py",
    "tests/test_recent_repair_retention_resume.py",
    "tests/test_recent_repair_retention_report.py",
})
CHANGED_PLAN_FIELDS = frozenset({"root", "code_path", "source_manifest", "execution_commit",
    "fixed_start_seconds_limit", "campaign_timeout_s"})
LINKED_EVIDENCE = ("training", "qualification", "reference", "canonicals", "ASSIGNMENTS.json",
    "PRETRAINING_SEAL.json", "EXPORTS.json", "ENDPOINT_DIFFERENCES.json")


def authenticate_source(manifest_path, source):
    manifest = read(manifest_path)
    if manifest.get("dirty"):
        raise ValueError("Continuation source must be a clean published checkout")
    for name, expected in manifest["files"].items():
        if identity(Path(source) / name) != expected:
            raise ValueError("Published source changed: " + name)
    for name, expected in manifest["symlinks"].items():
        if str((Path(source) / name).readlink()) != expected:
            raise ValueError("Published source symlink changed: " + name)
    return manifest


def source_changes(original, patched):
    changed = sorted(name for name in original["files"].keys() | patched["files"].keys()
        if original["files"].get(name) != patched["files"].get(name))
    forbidden = [name for name in changed if name not in REPAIR_FILES and not
        name.startswith("reports/dflash_recent_repair_retention_event7_20261112/")]
    if forbidden or original["symlinks"] != patched["symlinks"]:
        raise ValueError("Change outside the evaluation-accounting repair: " + repr(forbidden))
    return changed


def prior_failure(root, plan):
    """Prove the failed wrapper stopped before transition_begin/generate."""
    status, fixed = read(root / "STATUS.json"), read(root / "fixed_starts/STATUS.json")
    if (status["status"] != "failed" or status["phase"] != "fixed_start_diagnostic"
            or fixed["status"] != "failed" or fixed["phase"] != "pre_endpoint"
            or fixed["parent_requests"] != 0 or fixed["endpoint_blocks"] != 0
            or "model_load allowance exhausted" not in fixed.get("error", "")):
        raise ValueError("This entrypoint only resumes the identified pre-generation budget failure")
    if read(root / "fixed_starts/STARTS.json") or read(root / "fixed_starts/OUTCOMES.json"):
        raise ValueError("Existing fixed-start outcomes require a different recovery plan")
    if any((root / "fixed_starts/requests").rglob("*.json")) or (root / "evaluation").exists():
        raise ValueError("Existing parent generation or timed evaluation must not be repeated")
    import json
    ledger = [json.loads(line) for line in (root / "fixed_starts/WORK.jsonl").read_text().splitlines()]
    pending, starts, completed, failed = {}, Counter(), Counter(), Counter()
    for row in ledger:
        kind, state = row["kind"], row.get("status")
        if state == "attempted":
            pending[row["ticket"]] = kind
            starts[kind] += 1
            if kind in ("draft_block", "target_call", "context_forward") and not row["metadata"].get("startup"):
                raise ValueError("Non-startup model work exists; parent absence is not established")
        elif state in ("completed", "failed"):
            pending.pop(row["ticket"])
            (completed if state == "completed" else failed)[kind] += 1
    if (pending or starts["parent_walk"] != 1 or failed["parent_walk"] != 1
            or completed["parent_walk"] or any(starts[k] for k in ("neutral_block", "historical_block", "outcome_block"))):
        raise ValueError("Failed-attempt work is not the known pre-generation case")
    return dict(prior_campaign_wall_s=status["wall_s"], prior_fixed_start_gpu_s=fixed["work"]["gpu_seconds"],
        failed_startup_work=dict(attempted=dict(starts), completed=dict(completed), failed=dict(failed), unknown={}),
        actual_parent_generations_before=0, pending_parent_generations=len(plan["path_request_ids"]),
        pending_timed_completions=6*plan["panel_size"], new_optimizer_steps=0)


def authenticate_completed(plan):
    """Read retained receipts and immutable outputs; do not load any model."""
    root = Path(plan["root"])
    runtime.verify_execution_plan(plan)
    runtime.verify_pretraining(plan)
    files = [root / "TRAINING_PLAN.json", root / "STATUS.json", root / "PRETRAINING_SEAL.json",
        root / "fixed_starts/STATUS.json", root / "fixed_starts/WORK.jsonl", root / "EXPORTS.json",
        root / "ENDPOINT_DIFFERENCES.json", root / "canonicals/STATUS.json", root / "canonicals/CANONICAL.json"]
    for case in ("NATIVE", "DISABLED", "ZERO", "ACTIVE"):
        path = Path(plan["qualification_root"]) / case / "RESULT.json"
        value = read(path)
        if value["status"] != "passed":
            raise ValueError("Prior qualification is incomplete: " + case)
        for key in (("exact_native", "exact_selections") if case in ("DISABLED", "ZERO") else
                    ("active_restored", "long_restored") if case == "ACTIVE" else ()):
            if not value[key]["exact_equal"]:
                raise ValueError("Prior exact qualification failed: " + case + "/" + key)
        files.append(path)
    exports = read(root / "EXPORTS.json")
    if set(exports) != set(runtime.METHODS):
        raise ValueError("Require all three retained endpoint exports")
    for arm in runtime.METHODS:
        folder = root / "training" / arm
        status, result = read(folder / "STATUS.json"), read(folder / "RESULT.json")
        if (status["status"] != "completed" or result["status"] != "completed"
                or result["start_step"] != 768 or result["end_step"] != 896
                or result["optimizer_steps"] != 128 or len(result["records"]) != 256
                or status["global_step"] != 896 or status["presentations"] != 1792
                or result["checkpoint_path"] != exports[arm] or not result["boundary"]["export_exact"]):
            raise ValueError("Retained scientific endpoint is incomplete: " + arm)
        first = result["first_firing"]
        if first["status"] != "not_observed" and not (first["status"] == "passed" and
                first["active_exact"]["exact_equal"] and first["final_exact_restoration"]["exact_equal"]):
            raise ValueError("Retained first-firing check did not pass: " + arm)
        for path, expected in ((Path(exports[arm]) / "model.safetensors", result["weights_identity"]),
                               (Path(result["state_path"]), result["state_identity"])):
            if identity(path) != expected:
                raise ValueError("Retained weights/recovery changed: " + arm)
            files.append(path)
        files.extend([folder / "STATUS.json", folder / "RESULT.json"])
    canonical_status = read(root / "canonicals/STATUS.json")
    references = read(root / "canonicals/CANONICAL.json")
    requests = read(plan["panel_manifest"])["requests"]
    if canonical_status["status"] != "completed" or set(references) != {r["request_id"] for r in requests}:
        raise ValueError("Canonical preparation is incomplete")
    reused = read(plan["reused_references"])
    for request in requests:
        rid = request["request_id"]
        reference = references[rid]
        if not reference["verified"] or reference["prompt_ids"] != request["prompt_ids"]:
            raise ValueError("Canonical reference is not bound to its sealed prompt")
        if rid in reused:
            if reference != reused[rid]:
                raise ValueError("Reused canonical reference changed")
        else:
            label = f"reference_{request['panel_ordinal']:04d}"
            path = root / "canonicals" / (label + ".json")
            errors = runtime.validate_record(read(path), request, plan["checkpoints"]["K6"]["export"],
                reference, label=label, session="retention_canonical")
            if errors:
                raise ValueError("Prepared canonical reference failed revalidation: " + repr(errors))
            files.append(path)
    return {str(path): identity(path) for path in files}


def prepare(previous_plan, output, code_path, source_manifest):
    """Seal a concrete evaluation continuation without native model execution."""
    output, previous_plan = Path(output).resolve(), Path(previous_plan).resolve()
    if output.exists():
        raise FileExistsError("Use a new output directory; preserve previous attempts")
    old = read(previous_plan)
    root = Path(old["root"])
    evidence = authenticate_completed(old)
    failure = prior_failure(root, old)
    original_source = authenticate_source(old["source_manifest"], old["code_path"])
    patched_source = authenticate_source(source_manifest, code_path)
    if original_source["commit"] != old["execution_commit"]:
        raise ValueError("Retained source manifest differs from its execution commit")
    changed = source_changes(original_source, patched_source)
    remaining_fixed = old["fixed_start_seconds_limit"] - failure["prior_fixed_start_gpu_s"]
    remaining_campaign = old["campaign_timeout_s"] - failure["prior_campaign_wall_s"]
    if min(remaining_fixed, remaining_campaign) <= 0:
        raise ValueError("No original execution time budget remains")
    current = dict(old, root=str(output), code_path=str(Path(code_path).resolve()),
        source_manifest=str(Path(source_manifest).resolve()), execution_commit=patched_source["commit"],
        fixed_start_seconds_limit=remaining_fixed, campaign_timeout_s=remaining_campaign)
    provenance = dict(schema=SCHEMA, status="prepared_not_executed", previous_execution_root=str(root),
        previous_plan=str(previous_plan), training_execution_commit=old["execution_commit"],
        evaluation_execution_commit=patched_source["commit"], original_source_manifest=old["source_manifest"],
        patched_source_manifest=current["source_manifest"], changed_source_files=changed,
        fixed_start_remaining_s=remaining_fixed, campaign_remaining_s=remaining_campaign,
        receipt_identities=evidence, **failure)
    output.mkdir(parents=True)
    for name in LINKED_EVIDENCE:
        (output / name).symlink_to(root / name, target_is_directory=(root / name).is_dir())
    write_json(output / "TRAINING_PLAN.json", current)
    write_json(output / "CONTINUATION.json", provenance)
    write_json(output / "CONTINUATION_PLAN.json", dict(schema=SCHEMA,
        previous_plan=str(previous_plan), previous_plan_identity=identity(previous_plan),
        active_plan=str(output / "TRAINING_PLAN.json"), active_plan_identity=identity(output / "TRAINING_PLAN.json"),
        provenance=str(output / "CONTINUATION.json"), provenance_identity=identity(output / "CONTINUATION.json")))
    return provenance


def verify(plan_path):
    wrapper = read(plan_path)
    if wrapper["schema"] != SCHEMA:
        raise ValueError("Unknown continuation contract")
    for field in ("previous_plan", "active_plan", "provenance"):
        if identity(wrapper[field]) != wrapper[field + "_identity"]:
            raise ValueError("Sealed continuation input changed: " + field)
    old, plan, provenance = (read(wrapper[key]) for key in ("previous_plan", "active_plan", "provenance"))
    if set(plan) != set(old) or any(plan[key] != old[key] for key in old if key not in CHANGED_PLAN_FIELDS):
        raise ValueError("Continuation changed scientific configuration")
    for name in LINKED_EVIDENCE:
        link, expected = Path(plan["root"]) / name, Path(old["root"]) / name
        if (not link.is_symlink() or link.readlink() != expected
                or link.resolve(strict=True) != expected.resolve(strict=True)):
            raise ValueError("Continuation evidence link changed: " + name)
    if authenticate_completed(old) != provenance["receipt_identities"]:
        raise ValueError("Retained completion evidence changed after continuation sealing")
    failed = prior_failure(Path(old["root"]), old)
    if failed != {key: provenance[key] for key in failed}:
        raise ValueError("Retained failed-attempt accounting changed")
    original = authenticate_source(old["source_manifest"], old["code_path"])
    patched = authenticate_source(plan["source_manifest"], plan["code_path"])
    if source_changes(original, patched) != provenance["changed_source_files"]:
        raise ValueError("Repair-source difference changed")
    if (plan["execution_commit"] != provenance["evaluation_execution_commit"]
            or plan["execution_commit"] != patched["commit"]
            or old["execution_commit"] != provenance["training_execution_commit"]
            or old["execution_commit"] != original["commit"]):
        raise ValueError("Continuation source identities differ")
    if (plan["fixed_start_seconds_limit"] != old["fixed_start_seconds_limit"] - failed["prior_fixed_start_gpu_s"]
            or plan["campaign_timeout_s"] != old["campaign_timeout_s"] - failed["prior_campaign_wall_s"]):
        raise ValueError("Continuation reset an original execution time allowance")
    return old, plan, provenance


@contextmanager
def deadline(seconds):
    def expired(signum, frame):
        raise TimeoutError("Remaining original campaign timeout exhausted")
    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def preflight(plan):
    """Reuse runtime/model admission, without reinterpreting the old sealed plan."""
    from experiments.lost_repair.run import preflight as original
    if Path(__file__).resolve().parents[2] != Path(plan["code_path"]).resolve():
        raise ValueError("Execute the sealed patched checkout")
    legacy = deepcopy(plan["original_plan"])
    legacy.update(source_manifest_path=plan["source_manifest"], storage=dict(required_free_bytes=
        plan["storage"]["reserve_bytes"] + plan["storage"]["maximum_additional_bytes"]))
    k4 = legacy["checkpoints"]["K4"]["export"]
    legacy["checkpoints"] = {key:value for key,value in legacy["checkpoints"].items() if key in ("K5", "K6")}
    legacy["pinned_files"] = {key:value for key,value in legacy["pinned_files"].items() if not key.startswith(k4 + "/")}
    receipt = original(legacy, Path(plan["root"]) / "campaign_001")
    receipt.update(physical_gpu=runtime.physical_gpu(plan), capacity=runtime.capacity(plan))
    available = next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
                     if line.startswith('MemAvailable:'))
    if available < plan["host_memory_required_bytes"]:
        raise RuntimeError("Insufficient host memory for retained fixed-start states")
    receipt["host_available_bytes"] = available
    return receipt


def run(plan_path):
    began = time.perf_counter()
    old, plan, provenance = verify(plan_path)
    root = Path(plan["root"])
    if (root / "STATUS.json").exists():
        raise FileExistsError("No automatic retry of the explicit continuation")
    status = dict(status="running", phase="preflight", execution_commit=plan["execution_commit"],
        training_execution_commit=old["execution_commit"], new_optimizer_steps=0,
        prior_campaign_wall_s=provenance["prior_campaign_wall_s"])
    try:
        write_json(root / "STATUS.json", status)
        remaining = plan["campaign_timeout_s"] - (time.perf_counter() - began)
        if remaining <= 0:
            raise TimeoutError("Original campaign timeout exhausted during verification")
        with deadline(remaining):
            write_json(root / "PREFLIGHT.json", preflight(plan))
            exports, references = read(root / "EXPORTS.json"), read(root / "canonicals/CANONICAL.json")
            status["phase"] = "fixed_start_diagnostic"
            write_json(root / "STATUS.json", status)
            runtime.launch(plan, "experiments.recent_repair_retention.fixed_starts", ["--plan", root / "TRAINING_PLAN.json",
                "--exports", root / "EXPORTS.json", "--references", root / "canonicals/CANONICAL.json",
                "--output", root / "fixed_starts"], "fixed_starts")
            status["phase"] = "timed_serving"
            write_json(root / "STATUS.json", status)
            runtime.serve(plan, exports, references)
            verify(plan_path)
            write_json(root / "POSTFLIGHT.json", preflight(plan))
            status.update(status="completed", phase="complete")
    except BaseException as error:
        status.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        status["wall_s"] = time.perf_counter() - began
        status["combined_campaign_wall_s"] = provenance["prior_campaign_wall_s"] + status["wall_s"]
        write_json(root / "STATUS.json", status)
    return status


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare")
    preparation.add_argument("--previous-plan", required=True)
    preparation.add_argument("--output", required=True)
    preparation.add_argument("--code-path", required=True)
    preparation.add_argument("--source-manifest", required=True)
    execution = commands.add_parser("run")
    execution.add_argument("--plan", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.previous_plan, args.output, args.code_path, args.source_manifest)
    else:
        run(args.plan)
