"""Sealed scope and admission for readout fitting, calibration and FINAL."""

from copy import deepcopy
import json
import math
from pathlib import Path
import re
import shutil
import time

from online_feedback.storage import write_json
from experiments.osd_components.prepare import verify_files, verify_native_runtime
from experiments.kw_v1.preflight import hardware_admission
from .endpoint import endpoint_schedule
from .preflight import (GIB, PROJECT_ROOT, SOURCE_ROOT, EXPORT, TARGET, BOUNDARY,
                        WORKER_PYTHON, SERVING, identity, owned_path,
                        verify_export, verify_manifests)


TRAINING_PYTHON = "/data/conda_envs/minhao/dflash_specforge_20261022/bin/python"
OUTPUT_ROOT = PROJECT_ROOT / "runs/dflash_block_depth_20261105/full_production_001"
QUALIFICATION_COMMIT = "9150cc3b20714153b8e4fb3d70e8a32861c5921f"
ANALYSIS = dict(bootstrap_draws=10000, bootstrap_seed=20261105, interval="two-sided paired percentile 95%",
               unit="question; all arms and both repeats together",
               scientific_criterion="exact parity and both ADAPTIVE/static lower confidence limits > 0",
               practical_target_pct_vs_FIVE=2.0)
NUMERICAL = dict(ridge="float64 solve", ridge_penalty=1e-3, feature_scale_floor=1e-6,
                 gate_inference="folded float32 coefficients", readout="qualified BF16")
READOUT_GROUPS = [dict(group=i + 1, ordinal_start=i * 128, ordinal_stop=(i + 1) * 128,
                      presentations=256, steps=128) for i in range(3)]


def work_budget(reused_canonicals):
    if type(reused_canonicals) is not int or not 0 <= reused_canonicals <= 384:
        raise ValueError("Canonical reuse count is outside the fixed FINAL panel")
    missing = 384 - reused_canonicals
    return dict(readout_responses=384, retained_readout_responses=16, new_readout_generations=368,
                gate_behavior_generations=128, total_behavior_generations=512,
                readout_teacher_preparations=384, readout_groups=3,
                readout_presentations=768, readout_optimizer_steps=384, backbone_optimizer_steps=0,
                gate_requests=128, gate_calibration_starts_max=8192, gate_proposal_replays_max=16384,
                final_timed_generations=2304, reused_canonicals=reused_canonicals,
                untimed_reference_generations_max=missing,
                new_complete_generations_max=368 + 128 + missing + 2304,
                extra_qualification_generations=0, extra_pricing_measurements=0)


def capacity_plan():
    """One complete feature group plus atomic one-response write overlap.

    Each token stores six BF16 4096-vectors. Only one group is retained at a
    time. The ten-percent allowance covers serialization and small tensors;
    compact evidence/compiler/staging/sidecar allowances are additional.
    """
    response = 2272 * 6 * 4096 * 2
    features = math.ceil((128 + 1) * response * 1.10)
    additional = features + (4 + 4 + 2 + 1) * GIB
    return dict(feature_tokens_per_response_max=2272, feature_vectors_per_token=6,
                hidden_dimension=4096, bytes_per_feature_value=2,
                feature_response_bytes=response, retained_group_responses=128,
                duplicate_write_responses=1, feature_serialization_allowance_fraction=.10,
                teacher_feature_pool_peak_bytes=features,
                compact_evidence_bytes=4 * GIB, compiler_cache_bytes=4 * GIB,
                staging_bytes=2 * GIB, sidecar_states_and_exports_bytes=GIB,
                peak_additional_bytes=additional, reserve_bytes=8 * GIB,
                peak_required_bytes=additional + 8 * GIB, host_reserve_bytes=32 * GIB,
                new_backbone_checkpoint_bytes=0, offload_assumed=False,
                retention="Retire only completed group feature packets after compact evidence and sidecar state persist; no second group overlaps")


def check_capacity(*, free_bytes, host_available_bytes, initial=True):
    if any(type(value) is not int or value < 0 for value in (free_bytes, host_available_bytes)):
        raise ValueError("Capacity measurements require integer nonnegative bytes")
    budget = capacity_plan()
    needed = budget["peak_required_bytes"] if initial else budget["reserve_bytes"]
    receipt = dict(status="passed", free_bytes=free_bytes, required_bytes=needed,
                   host_available_bytes=host_available_bytes, host_required_bytes=budget["host_reserve_bytes"],
                   initial_admission=initial, capacity=budget)
    if free_bytes < needed or host_available_bytes < budget["host_reserve_bytes"]:
        raise RuntimeError("Full depth-screen capacity unavailable: " + repr(receipt))
    return receipt


def make_plan(qualification_plan, manifests, canonical_audit, pricing, *,
              code_commit, source_files, input_identities, paths):
    """Pure plan construction; callers seal identities at their actual paths."""
    verify_manifests(manifests)
    if qualification_plan["code_commit"] != QUALIFICATION_COMMIT:
        raise ValueError("Reuse the successful native milestone, not failed attempt001")
    if (canonical_audit.get("status") != "passed" or canonical_audit["requested"] != 384
            or canonical_audit["reused"] + canonical_audit["missing_count"] != 384):
        raise ValueError("Canonical availability audit is incomplete")
    if pricing.get("status") != "complete" or pricing.get("decision") != "continue_to_readout_fit":
        raise ValueError("The existing economic gate does not authorize fitting")
    required = {"output_root", "canonical_file", "canonical_audit_file", "pricing_file",
                "qualification_status_file", "training_config"}
    if set(paths) != required:
        raise ValueError("Full plan requires exactly the named artifact paths")
    result = deepcopy(qualification_plan)
    result["qualification_provenance"] = {key: result.pop(key) for key in
        ("qualification_modes", "pricing", "stop_after", "prior_attempt") if key in result}
    result.update(paths)
    result["config_file"] = paths["training_config"]
    result.update(schema="block_depth_full_v1", stage="readout_gate_final", code_commit=code_commit,
                  qualification_code_commit=QUALIFICATION_COMMIT,
                  qualification_root=qualification_plan["output_root"],
                  source_files=source_files, input_identities=input_identities,
                  training_python=TRAINING_PYTHON, budget=work_budget(canonical_audit["reused"]),
                  storage=capacity_plan(), continuation_cost_ms=pricing["continuation_cost_ms"],
                  canonical_binding_digest=canonical_audit["binding_digest"],
                  final_schedule=endpoint_schedule(manifests["manifests"]["FINAL"]["requests"]),
                  readout_groups=deepcopy(READOUT_GROUPS), analysis=deepcopy(ANALYSIS), numerical=deepcopy(NUMERICAL),
                  no_test_calibration=True, automatic_retry=False, stop_after="full_FINAL_report")
    validate_plan(result)
    return result


def validate_plan(plan):
    fixed = dict(schema="block_depth_full_v1", stage="readout_gate_final", checkpoint_path=str(EXPORT),
                 target_path=str(TARGET), worker_python=WORKER_PYTHON, training_python=TRAINING_PYTHON,
                 source_root=str(SOURCE_ROOT), serving=SERVING,
                 storage=capacity_plan(), qualification_code_commit=QUALIFICATION_COMMIT,
                 no_test_calibration=True, automatic_retry=False, stop_after="full_FINAL_report",
                 readout_groups=READOUT_GROUPS, analysis=ANALYSIS, numerical=NUMERICAL)
    for key, expected in fixed.items():
        if plan.get(key) != expected:
            raise ValueError("Full experiment fixed contract changed: " + key)
    output = Path(plan["output_root"])
    if output.parent != OUTPUT_ROOT.parent or not re.fullmatch(r"full_production_\d{3}", output.name):
        raise ValueError("Require a separate numbered full-run destination")
    if not re.fullmatch(r"[0-9a-f]{40}", plan.get("code_commit", "")):
        raise ValueError("Pin a complete published execution commit")
    if plan["budget"] != work_budget(plan["budget"]["reused_canonicals"]):
        raise ValueError("Full experiment work budget changed")
    if plan.get("config_file") != plan["training_config"]:
        raise ValueError("Training configuration aliases disagree")
    if not math.isfinite(plan["continuation_cost_ms"]):
        raise ValueError("Continuation cost is nonfinite")
    for key in ("manifest_file", "model_pin", "runtime_pin", "canonical_file", "canonical_audit_file",
                "pricing_file", "qualification_status_file", "training_config"):
        path = owned_path(plan[key])
        if str(path) not in plan["input_identities"]:
            raise ValueError("Unsealed full-run input: " + key)
    owned_path(plan["output_root"])
    owned_path(plan["qualification_root"])
    minimum = {"experiments/block_depth/" + name + ".py" for name in
               ("gate", "endpoint", "full_plan", "readout_training", "native_routes", "worker", "reconstruct", "full_campaign", "full_report")}
    if not minimum <= set(plan["source_files"]):
        raise ValueError("Source pin omits a full-run dependency")
    for relative in plan["source_files"]:
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Source identity escapes the checkout")
    return dict(status="passed", stage=plan["stage"], budget=plan["budget"])


def verify_inputs(plan):
    """Rebind immutable inputs before model execution; no endpoint outcomes."""
    for path, expected in plan["input_identities"].items():
        owned_path(path)
        if identity(path) != expected:
            raise ValueError("Sealed full-run input changed: " + path)
    manifests = json.loads(Path(plan["manifest_file"]).read_text())
    verify_manifests(manifests)
    if endpoint_schedule(manifests["manifests"]["FINAL"]["requests"]) != plan["final_schedule"]:
        raise ValueError("FINAL balanced order changed")
    references = json.loads(Path(plan["canonical_file"]).read_text())
    audit = json.loads(Path(plan["canonical_audit_file"]).read_text())
    if len(references) != plan["budget"]["reused_canonicals"] or audit["binding_digest"] != plan["canonical_binding_digest"]:
        raise ValueError("Retained canonical source binding changed")
    requests = {r["request_id"]: r for r in manifests["manifests"]["FINAL"]["requests"]}
    for key, value in references.items():
        if (key not in requests or value.get("verified") is not True
                or value["prompt_ids"] != requests[key]["prompt_ids"]
                or value["binding_digest"] != plan["canonical_binding_digest"]):
            raise ValueError("Canonical reference differs from its sealed input")
    missing = [key for key in requests if key not in references]
    if audit["missing_request_ids"] != missing or len(missing) != plan["budget"]["untimed_reference_generations_max"]:
        raise ValueError("Missing-reference work changed")
    pricing = json.loads(Path(plan["pricing_file"]).read_text())
    status = json.loads(Path(plan["qualification_status_file"]).read_text())
    if (pricing.get("status") != "complete" or pricing.get("decision") != "continue_to_readout_fit"
            or pricing["continuation_cost_ms"] != plan["continuation_cost_ms"]):
        raise ValueError("The priced continuation cost changed")
    if (status.get("status") != "completed" or status.get("valid_generations") != 32
            or status.get("returned_route_measurements") != 2048
            or any(status.get(k) != 0 for k in ("invalid_generations", "unknown_generations", "unattempted_generations", "optimizer_steps", "fitting_presentations", "final_generations"))):
        raise ValueError("Native qualification/pricing is incomplete")
    import yaml
    config = yaml.safe_load(Path(plan["training_config"]).read_text())
    required = {
        "model": dict(target_model_path=str(TARGET), draft_checkpoint_path=str(EXPORT), torch_dtype="bfloat16"),
        "data": dict(max_length=2272),
        "training": dict(max_steps=384, batch_size=1, accumulation_steps=2, learning_rate=3e-4,
                         lr_scheduler="constant", warmup_ratio=0., max_grad_norm=1., seed=17,
                         attention_backend="flex_attention", num_anchors=64, objective_chunk_blocks=8, loss_decay_gamma=7.),
        "online_feedback": dict(responses_per_event=128, passes_per_event=2, max_events=3,
                                optimizer="bf16_adamw", weight_decay=.01, shuffle_responses=True, kl_precision="bfloat16"),
    }
    for section, fields in required.items():
        if any(config[section].get(key) != value for key, value in fields.items()):
            raise ValueError("Readout training configuration changed: " + section)
    return dict(status="passed", reused_canonicals=len(references), missing_canonicals=len(missing),
                final_outputs_analyzed=False, model_work=0)


def resource_check(root, plan, phase):
    """Recheck the declared remaining peak before another artifact stage."""
    root = owned_path(root)
    if root != Path(plan["output_root"]) or not re.fullmatch(r"[A-Z_0-9]+", phase):
        raise ValueError("Resource receipt is outside the sealed stage")
    memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    receipt = check_capacity(free_bytes=shutil.disk_usage(root).free,
        host_available_bytes=int(memory["MemAvailable"].split()[0]) * 1024, initial=True)
    receipt["phase"] = phase
    path = root / "resources" / (phase + ".json")
    if path.exists():
        raise FileExistsError("Preserve the existing resource receipt")
    write_json(path, receipt)
    return receipt


def preflight(plan, outfile, *, postflight=False):
    """Admission retains the previous model/runtime checks with full capacity."""
    outfile = owned_path(outfile)
    if outfile.exists():
        raise FileExistsError("Preserve the previous admission receipt")
    result, started = dict(status="running", code_commit=plan.get("code_commit"), model_work=0), time.perf_counter()
    try:
        result["plan"] = validate_plan(plan)
        result["inputs"] = verify_inputs(plan)
        verify_files(plan["source_root"], plan["source_files"])
        result["source_files_checked"] = len(plan["source_files"])
        result["export"] = verify_export(plan["checkpoint_path"], json.loads(BOUNDARY.read_text()))
        model = json.loads(Path(plan["model_pin"]).read_text())
        verify_files(plan["target_path"], model["target"]["files"], exact_files=True)
        result["runtime"] = verify_native_runtime(plan["worker_python"], json.loads(Path(plan["runtime_pin"]).read_text()))
        if not postflight:
            result["hardware"] = hardware_admission()
        memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        result["capacity"] = check_capacity(free_bytes=shutil.disk_usage(plan["output_root"]).free,
            host_available_bytes=int(memory["MemAvailable"].split()[0]) * 1024, initial=not postflight)
        result["status"] = "passed"
        return result
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        result["wall_s"] = time.perf_counter() - started
        write_json(outfile, result)
