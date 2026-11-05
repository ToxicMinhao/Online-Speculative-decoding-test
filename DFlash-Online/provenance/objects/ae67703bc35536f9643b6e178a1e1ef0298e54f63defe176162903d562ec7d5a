"""Build the review package from retained successful-run evidence; no model imports.

This prepares data/configuration for review. It neither installs experiment code
nor admits storage, contacts A800, or launches a training/generation process.
"""
from pathlib import Path
import copy
import hashlib
import json
import random

import yaml


ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT.parent / "dflash_online_osd_rerun_20261023"
DESTINATION = "/data/sources/minhao/runs/osd_transfer_budget_20261024/execution_001"


def write_json(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def differences(before, after, prefix=""):
    result = []
    for key in sorted(before.keys() | after.keys()):
        path = f"{prefix}.{key}" if prefix else key
        old, new = before.get(key), after.get(key)
        if isinstance(old, dict) and isinstance(new, dict):
            result.extend(differences(old, new, path))
        elif old != new:
            result.append({"field": path, "reference": old, "planned": new})
    return result


def main():
    seal = json.loads((REFERENCE / "sealed/SEAL.json").read_text())
    config_path, manifest_path = (REFERENCE / "sealed/config.yaml",
                                  REFERENCE / "sealed/MANIFEST.json")
    source_check = {
        relative: digest(REFERENCE / "worktree" / relative) == expected
        for relative, expected in seal["source_sha256"].items()
    }
    assert all(source_check.values()), "Retained source changed from the executed seal"
    assert digest(config_path) == seal["config_sha256"]
    assert digest(manifest_path) == seal["manifest_sha256"]

    original = json.loads(manifest_path.read_text())
    selected = copy.deepcopy(original["train_requests"][:640])
    assert len({row["dataset_id"] for row in selected}) == 640
    assert len({tuple(row["prompt_ids"]) for row in selected}) == 640
    panel_rng, diagnostic_rng = random.Random(17), random.Random(20261024)
    blocks = []
    for number in range(1, 6):
        rows = selected[(number - 1) * 128:number * 128]
        for row in rows:
            row["collect_feedback"] = number < 5
            row["event_id"] = f"event_{number:03d}" if number < 5 else None
        panel = sorted(panel_rng.sample(range(128), 32)) if number > 1 else []
        probes = sorted(diagnostic_rng.sample(range(128), 8)) if number < 5 else []
        blocks.append({
            "block": f"B{number}", "canonical_checkpoint": "F" if number == 1 else f"S{number-1}",
            "feedback_for_S": number < 5, "requests": rows,
            "panel_local_indices": panel,
            "panel_request_ids": [rows[index]["request_id"] for index in panel],
            "source_probe_request_ids": [rows[index]["request_id"] for index in probes],
        })
    request_manifest = {
        "schema": "osd_transfer_budget_review_manifest_v1", "status": "review_prepared_not_execution_sealed",
        "reference_manifest": str(manifest_path), "reference_manifest_sha256": seal["manifest_sha256"],
        "provenance": original["provenance"],
        "selection": "First 640 training-stream requests in retained source order; no outcome-based selection",
        "prior_project_exposure": "All questions were used in earlier runs; this run starts from released weights and generates fresh answers",
        "prompt_policy": "Exact serving-stream token IDs; do not use the differently instructed historical evaluation prompts",
        "training_seed": 17, "panel_selection_seed": 17, "source_probe_selection_seed": 20261024,
        "counts": {"canonical_generations": 640, "fresh_feedback_responses": 512,
                   "distinct_panel_requests": 128, "primary_fresh_vs_repeat_requests": 96,
                   "benchmark_generations": 1536, "all_generations": 2176,
                   "physical_optimizer_steps": 896, "physical_response_presentations": 1792,
                   "source_diagnostic_response_forwards": 112},
        "blocks": blocks,
    }
    write_json("REQUEST_MANIFEST.json", request_manifest)

    previous_config = yaml.safe_load(config_path.read_text())
    planned_config = copy.deepcopy(previous_config)
    planned_config["training"]["max_steps"] = 512
    planned_config["online_feedback"]["max_events"] = 4
    planned_config["run_id"] = "osd-transfer-budget-20261024"
    planned_config["output_dir"] = DESTINATION + "/trainer_runtime"
    (ROOT / "shared_training.yaml").write_text(yaml.safe_dump(planned_config, sort_keys=False))
    delta = differences(previous_config, planned_config)
    assert {row["field"] for row in delta} == {
        "training.max_steps", "online_feedback.max_events", "run_id", "output_dir"}
    write_json("CONFIG_DIFF.json", {
        "status": "planning_only", "changes": delta,
        "learning_parameter_changes": [],
        "reason": "Shorter declared horizon and new run paths only; direct BF16 AdamW uses constant LR",
        "branch_configuration": "Both branches use the same byte-identical shared_training.yaml; output_root distinguishes branches",
    })
    runtime = json.loads((ROOT.parent / "dflash_online_sglang_20261023/evidence/runtime_004.json").read_text())
    write_json("PARAMETER_LOCK.json", {
        "status": "review_lock_not_live_verified", "source_root": "/data/sources/minhao/DFlash-Online",
        "reference_seal": str(REFERENCE / "sealed/SEAL.json"),
        "reference_config_sha256": seal["config_sha256"],
        "unchanged_source_sha256": seal["source_sha256"],
        "native_runtime_sha256": runtime["native_runtime_sha256"],
        "target_revision": "b968826d9c46dd6066d109eabc6255188de91218",
        "draft_revision": "9b41424b7109f9c5413454f481b09a82b85333f4",
        "dflash_commit": "07ebd93db9f472af339b644bb70221ad8428328a",
        "specforge_commit": "1839aae30003e3573d00979eb1edd57347b3578c",
        "objective": "osd", "reduction": "response_mean", "temperature": 1.0, "hard_label_mix": 0.0,
        "adam_betas": [0.9, 0.999], "adam_epsilon": 1e-8,
        "parameter_and_moment_precision": "bfloat16", "weighted_reduction_precision": "float32",
        "serving": {"block_size": 16, "context_limit": 4096, "native_capacity": 4112,
                    "output_cap": 2048, "eos_enabled": True, "concurrency": 1,
                    "greedy": True, "attention_backend": "fa3", "chunked_prefill_size": 8192,
                    "cache_policy": "idle_full_flush_before_every_request_and_publication",
                    "capture_during_benchmarks": False, "timing_repeats": 3,
                    "mem_fraction_static": 0.65, "cuda_graph_max_bs": 8,
                    "skip_server_warmup": True, "SGLANG_ENABLE_SPEC_V2": "0"},
        "source_diagnostic": {"responses_per_buffer": 8, "anchor_cap": 512,
                              "layout_seed": 20261024, "layout_lifetime": "fixed per source response across all events",
                              "mode": "eval/no_grad", "training_rng_preserved": True},
        "runtime": {"trainer_python": seal["trainer_python"], "trainer_versions": seal["trainer_versions"],
                    "worker_python": runtime["python"], "worker_torch": runtime["torch"],
                    "worker_transformers": runtime["transformers"], "training_gpu": 1, "serving_gpu": 0},
    })
    W = 2097259104
    rows = 128 * 2272
    prepared = rows * 4096 * 6 * 2 + rows * 12
    native = rows * 4096 * 6 * 2
    components = {"B1_retained_prepared_bytes": prepared,
                  "current_buffer_native_and_prepared_bytes": native + prepared,
                  "four_state_bounds_bytes": 4 * 3 * W,
                  "five_export_bounds_bytes": 5 * W}
    tensor_bytes = sum(components.values())
    allowance = {"tensor_serialization_margin_bytes": (tensor_bytes + 19)//20,
                 "compact_allowance_bytes": 8 * 2**30, "work_allowance_bytes": 2 * 2**30,
                 "free_reserve_bytes": 8 * 2**30}
    write_json("STORAGE_DESIGN.json", {
        "status": "not_admitted", "live_A800_capacity": "unavailable: SSH authentication rejected",
        "basis": "Conservative all-local peak; actual sizes/liveness and any offload require execution admission",
        "released_export_reference_bytes": W, "components": components, "allowance": allowance,
        "gross_new_run_capacity_bytes_including_reserve": tensor_bytes + sum(allowance.values()),
        "counts_are_run_files_not_existing_installed_models": True,
        "offload": "Allowed to Fenrir after verifying destination capacity, exact transfer and dependencies; never assume a move freed space",
    })
    write_json("DESIGN_VALIDATION.json", {
        "status": "static_preparation_passed_not_runtime_qualified", "source_files_checked": len(source_check),
        "source_files_matching_seal": sum(source_check.values()), "reference_config_matches_seal": True,
        "reference_manifest_matches_seal": True, "request_id_count": 640, "unique_prompt_count": 640,
        "prompt_tokens_min": min(len(row["prompt_ids"]) for row in selected),
        "prompt_tokens_max": max(len(row["prompt_ids"]) for row in selected),
        "full_output_cap_fits_training": all(len(row["prompt_ids"])+2048 <= 2272 for row in selected),
        "config_diff_allowlist_passed": True, "counts": request_manifest["counts"],
        "live_source_verified": False, "storage_admitted": False, "model_execution_started": False,
    })
    print(json.dumps({"prepared": str(ROOT), "source_files_verified": len(source_check),
                      "counts": request_manifest["counts"], "model_execution_started": False}))


if __name__ == "__main__":
    main()
