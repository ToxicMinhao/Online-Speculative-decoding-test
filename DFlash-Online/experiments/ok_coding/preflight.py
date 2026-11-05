"""CPU-only source, recipe and full-capacity admission for the coding campaign."""
import json
from pathlib import Path
import platform
import shutil
import sys
import time

from online_feedback.storage import write_json
from experiments.kw_v1.preflight import hardware_admission
from experiments.osd_components.prepare import verify_files, verify_native_runtime
from .storage import GIB, identity, full_campaign_budget


def verify_seal(root):
    seal = json.loads((root / "sealed" / "SEAL.json").read_text())
    for filename, expected in seal["files"].items():
        if identity(filename) != expected:
            raise ValueError("Sealed source or input changed: " + filename)
    for folder, names in seal.get("exact_file_sets", {}).items():
        actual = sorted(path.name for path in Path(folder).iterdir() if path.is_file())
        if actual != names:
            raise ValueError("Sealed file set differs: " + folder)
    return {"status": "passed", "files_checked": len(seal["files"]),
            "execution_source": seal["execution_source"]}


def check_configs(plan):
    import yaml
    from online_feedback.recipe import load_event_config

    normalized = []
    result = {}
    for seed in (17, 29):
        path = Path(plan["configs"][str(seed)])
        config, recipe = load_event_config(path)
        raw = yaml.safe_load(path.read_text())
        expected_training = {
            "seed": seed, "max_steps": 7424, "batch_size": 1,
            "accumulation_steps": 2, "learning_rate": 3e-4,
            "lr_scheduler": "constant", "warmup_ratio": 0.0,
            "max_grad_norm": 1.0, "attention_backend": "flex_attention",
            "num_anchors": 512, "objective_chunk_blocks": 8,
            "loss_decay_gamma": 7.0, "optimizer_cpu_offload": False,
        }
        if any(raw["training"].get(k) != v for k, v in expected_training.items()):
            raise ValueError("Qualified training recipe differs for seed " + str(seed))
        expected_online = dict(responses_per_event=128, passes_per_event=2,
            max_events=58, optimizer="bf16_adamw", weight_decay=0.01,
            shuffle_responses=True, kl_precision="bfloat16")
        if raw["online_feedback"] != expected_online:
            raise ValueError("Online recipe/horizon differs")
        if (raw["data"]["max_length"] != 2272
                or raw["model"]["torch_dtype"] != "bfloat16"
                or raw["model"]["draft_checkpoint_path"] != plan["draft"]
                or raw["model"]["target_model_path"] != plan["target"]):
            raise ValueError("Released initialization/model/length binding differs")
        raw["training"]["seed"] = "SEED"
        raw["run_id"] = "RUN_ID"
        raw["output_dir"] = "OUTPUT"
        normalized.append(raw)
        result[str(seed)] = {"config": str(path), "identity": identity(path),
                            "native_horizon": 7424, "native_max_events": 58,
                            "campaign_events": 4, "seed": seed}
    if normalized[0] != normalized[1]:
        raise ValueError("Seed configs differ beyond declared seed/output identity")
    return result


def storage_admission(plan, free_bytes, available_host_bytes, *, campaign_root=None, boundary=None):
    fresh_capacity = full_campaign_budget(plan)
    capacity = fresh_capacity
    if boundary is not None:
        from .resume_storage import remaining_campaign_budget
        capacity = remaining_campaign_budget(plan, campaign_root, boundary)
    settings = plan["storage"]
    if (settings["reserve_bytes"] != 8 * GIB
            or settings["mode"] != "retain_local"
            or settings["offload_assumed"] is not False
            or settings["campaign_capacity"] != fresh_capacity):
        raise ValueError("Storage contract differs from full two-seed calculation")
    required = capacity["peak_required_bytes"]
    host_required = 80 * GIB
    return {"admitted": free_bytes >= required and available_host_bytes >= host_required,
            "free_bytes": free_bytes, "required_bytes": required,
            "host_available_bytes": available_host_bytes,
            "host_required_bytes": host_required, "capacity": capacity,
            "memory_scope": "Host admission for decoded state comparisons; GPU capacity checked separately"}


def verify(root, *, postflight=False, resume=False):
    root = Path(root).resolve()
    started = time.perf_counter()
    from .attempts import reserve_receipt
    receipt_path = (reserve_receipt(root / "resume_admission") if resume
                    else root / ("POSTFLIGHT.json" if postflight else "PREFLIGHT.json"))
    result = {"receipt_path": str(receipt_path), "status": "running", "model_forwards": 0, "backwards": 0,
              "optimizer_steps": 0, "generations": 0, "postflight": postflight,
              "explicit_resume": resume}
    try:
        result["source"] = verify_seal(root)
        plan = json.loads((root / "sealed" / "PLAN.json").read_text())
        from .contract import validate_plan
        result["plan"] = validate_plan(plan)
        result["configs"] = check_configs(plan)
        boundary = None
        if resume:
            from .campaign import admit_recovery
            boundary = admit_recovery(root, plan, write_receipt=False)
            result["recovery_boundary"] = boundary
        if not postflight:
            if (root / "campaign_001").exists() and not resume:
                raise ValueError("Fresh execution only; explicit boundary recovery required")
            result["hardware"] = hardware_admission()
            mem = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
            result["storage"] = storage_admission(plan, shutil.disk_usage(root).free,
                int(mem["MemAvailable"].split()[0]) * 1024,
                campaign_root=root / "campaign_001", boundary=boundary)
            if not resume:
                write_json(root / "STORAGE_ADMISSION.json", result["storage"])
            if not result["storage"]["admitted"]:
                raise RuntimeError("Remaining-campaign disk/host storage admission failed" if resume
                                   else "Full-campaign disk/host storage admission failed")
        pins = json.loads((root / "sealed" / "MODEL_PIN.json").read_text())
        for model in ("target", "draft"):
            verify_files(plan[model], pins[model]["files"], exact_files=True)
        result["native_runtime"] = verify_native_runtime(plan["worker_python"],
            json.loads((root / "sealed" / "RUNTIME_PIN.json").read_text()))
        import torch
        import transformers
        actual = {"python": platform.python_version(), "torch": torch.__version__,
                  "transformers": transformers.__version__}
        pinned = json.loads((root / "sealed" / "TRAINING_PIN.json").read_text())
        if actual != pinned["environment"] or torch.cuda.is_initialized():
            raise ValueError("Training runtime changed or CPU preflight initialized CUDA")
        result["training_runtime"] = actual
        result["memory_feasibility"] = {
            "configuration": "Qualified O/K tensor shapes: <=2272 tokens, <=512 anchors, projection chunks8",
            "devices": "One otherwise idle80GiB A800 per native process; same frozen target/draft pins",
            "limit": "Metadata/capacity admission is not a new coding-specific peak measurement. Actual ordinary jobs record peaks and stop on OOM; no extra qualification forwards."}
        if resume:
            from .recovery import record_admission
            result["recovery_boundary"] = record_admission(root, boundary)
        result["status"] = "passed"
        return result
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        result["wall_s"] = time.perf_counter() - started
        write_json(receipt_path, result)


if __name__ == "__main__":
    print(json.dumps(verify(sys.argv[1], postflight="--postflight" in sys.argv[2:],
                            resume="--resume" in sys.argv[2:])))
