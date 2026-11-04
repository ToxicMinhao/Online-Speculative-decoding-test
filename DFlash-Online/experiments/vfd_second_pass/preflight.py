"""CPU-only immutable-input authentication and five-policy peak admission."""

import argparse
import json
from pathlib import Path
import platform
import sys
import time

from online_feedback.storage import write_json
from experiments.ok_coding.preflight import verify_seal
from experiments.ok_coding.storage import identity
from experiments.osd_components.prepare import verify_files, verify_native_runtime
from experiments.kw_v1.preflight import hardware_admission
from experiments.kplus_full_gsm8k.preflight import timing_barrier
from .contract import ORIGINAL_EXECUTION, digest, validate_plan
from .storage import admit, capacity_plan


def read(path):
    return json.loads(Path(path).read_text())


def same_file(path, expected):
    actual = identity(path)
    if actual != {key: expected[key] for key in ("bytes", "sha256")}:
        raise ValueError("Authenticated input changed: " + str(path))
    return actual


def verify_original(plan, audit):
    """Bind the original K+ event zero, never a trained endpoint or old K alias."""
    state = audit["initial_state"]
    fixed = dict(
        global_step=0,
        presentations=0,
        events=[],
        adam_state_entries=0,
        released_weights_exact=True,
        rng_equal_original_K=True,
        parameter_tensors=58,
        cuda_rng_states=1,
        horizon=7424,
        accumulation_steps=2,
        reduction="response_mean",
    )
    if (
        audit["status"] != "passed"
        or audit["errors"]
        or audit["model_forwards"] != 0
        or audit["optimizer_steps"] != 0
        or audit["cuda_initialized"]
        or any(state.get(key) != value for key, value in fixed.items())
    ):
        raise ValueError("Event-zero CPU authentication is incomplete")
    original = Path(plan["original_root"])
    expected_path = original / "campaign_001/arms/KPLUS/trainer/initial_state.pt"
    if Path(plan["original_initial_state"]) != expected_path or state["identity"][
        "path"
    ] != str(expected_path):
        raise ValueError("Require the authenticated PR25 K+ event-zero state")
    checked = {str(expected_path): same_file(expected_path, state["identity"])}
    for name, expected in audit["seals"].items():
        path = original / "sealed" / name
        if str(path) != expected["path"]:
            raise ValueError("Historical seal root changed")
        checked[str(path)] = same_file(path, expected)
    old_plan = read(original / "sealed/PLAN.json")
    if old_plan["execution_source"]["commit"] != ORIGINAL_EXECUTION:
        raise ValueError("Historical scientific execution changed")
    same_file(plan["original_config"], audit["seals"]["config_KPLUS.yaml"])
    original_rows = read(original / "sealed/ORIGINAL_MANIFEST.json")["train_requests"][
        :512
    ]
    current = plan["training_requests"]
    if len(original_rows) != 512 or len(current) != 512:
        raise ValueError("Require exactly the original first 512 training requests")
    for index, (old, new) in enumerate(zip(original_rows, current)):
        if any(new.get(key) != value for key, value in old.items()):
            raise ValueError("Original training identity/order changed: " + str(index))
    if digest(original_rows) != audit["stream"]["ordered_512_sha256"]:
        raise ValueError("Ordered training stream differs from CPU authentication")
    return dict(
        status="passed",
        initial_identity=checked[str(expected_path)],
        original_binding=old_plan["binding"],
        files_checked=len(checked),
        ordered_training_digest=digest(original_rows),
    )


def verify_references(plan, audit, models, runtime, references, missing):
    """A retained answer is reusable only with identical inputs and runtime."""
    binding = audit["binding"]
    expected = dict(
        target_files=models["target"]["files"],
        native_runtime_sha256=runtime["native_runtime_sha256"],
        python_version=runtime["python_version"],
        torch=runtime["torch"],
        transformers=runtime["transformers"],
        serving=read(Path(plan["original_root"]) / "sealed/PLAN.json")["serving"],
    )
    if (
        audit["status"] != "passed"
        or audit["model_work"] != 0
        or binding != expected
        or audit["binding_digest"] != digest(binding)
    ):
        raise ValueError("Retained canonical target/runtime binding changed")
    for key, value in plan["serving"].items():
        if key != "capture" and binding["serving"].get(key) != value:
            raise ValueError("Canonical serving setting changed: " + key)
    requests = {row["request_id"]: row for row in plan["evaluation_requests"]}
    missing_ids = missing["request_ids"]
    if (
        len(requests) != 200
        or len(missing_ids) != len(set(missing_ids))
        or set(references) & set(missing_ids)
        or set(references) | set(missing_ids) != set(requests)
        or audit["reused_count"] != len(references)
        or audit["missing_count"] != len(missing_ids)
        or missing["count"] != len(missing_ids)
        or missing["maximum_new_generations"] != len(missing_ids)
    ):
        raise ValueError("Reference reuse/missing partition changed")
    for request_id, reference in references.items():
        if (
            reference["verified"] is not True
            or reference["binding_digest"] != audit["binding_digest"]
            or reference["prompt_ids"] != requests[request_id]["prompt_ids"]
            or not reference["response_ids"]
        ):
            raise ValueError("Retained canonical identity changed: " + request_id)
    for relative, expected_file in audit["checked_files"].items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Reference source escapes original root")
        same_file(Path(plan["original_root"]) / path, expected_file)
    from .evaluation import audit_canonicals

    raw_check = audit_canonicals(
        [row for row in plan["evaluation_requests"] if row["request_id"] in references],
        references,
        binding_digest=audit["binding_digest"],
    )
    if raw_check["status"] != "passed":
        raise ValueError("Retained canonical raw validation failed: " + repr(raw_check))
    return dict(
        status="passed",
        binding_digest=audit["binding_digest"],
        reused=len(references),
        missing=len(missing_ids),
        files_checked=len(audit["checked_files"]),
        raw_validation=raw_check,
    )


def verify(root, *, postflight=False):
    root = Path(root)
    started = time.perf_counter()
    result = dict(
        status="running",
        postflight=postflight,
        model_forwards=0,
        optimizer_steps=0,
        generations=0,
    )
    try:
        result["seal"] = verify_seal(root)
        plan = read(root / "sealed/PLAN.json")
        result["plan"] = validate_plan(plan)
        if result["seal"]["execution_source"] != plan["execution_source"]:
            raise ValueError("Plan and source seal identify different implementations")
        if read(root / "sealed/TRAINING_REQUESTS.json") != plan["training_requests"]:
            raise ValueError("Sealed training requests differ from plan")
        if read(root / "sealed/PANEL.json")["requests"] != plan["evaluation_requests"]:
            raise ValueError("Sealed endpoint differs from plan")
        audit = read(root / "sealed/INITIAL_STATE_AND_PINS_AUDIT.json")
        result["original"] = verify_original(plan, audit)
        for name in ("MODEL_PIN.json", "RUNTIME_PIN.json", "TRAINING_PIN.json"):
            same_file(root / "sealed" / name, audit["seals"][name])
        from .training import validate_config_text

        validate_config_text(
            Path(plan["config"]).read_text(),
            "K",
            Path(plan["original_config"]).read_text(),
        )
        result["config"] = dict(status="passed", identity=identity(plan["config"]))
        models, runtime = read(root / "sealed/MODEL_PIN.json"), read(
            root / "sealed/RUNTIME_PIN.json"
        )
        for model in ("target", "draft"):
            verify_files(plan[model], models[model]["files"], exact_files=True)
        result["models"] = dict(
            status="passed",
            files_checked=sum(len(models[k]["files"]) for k in ("target", "draft")),
        )
        result["native_runtime"] = verify_native_runtime(plan["worker_python"], runtime)
        import torch
        import transformers

        actual = dict(
            python=platform.python_version(),
            torch=torch.__version__,
            transformers=transformers.__version__,
        )
        if (
            actual != read(root / "sealed/TRAINING_PIN.json")["environment"]
            or Path(sys.executable).resolve() != Path(plan["training_python"]).resolve()
            or torch.cuda.is_initialized()
        ):
            raise ValueError(
                "Training runtime changed or CPU preflight initialized CUDA"
            )
        result["training_runtime"] = actual
        result["references"] = verify_references(
            plan,
            read(root / "sealed/REFERENCE_REUSE_AUDIT.json"),
            models,
            runtime,
            read(root / "sealed/CANONICALS.json"),
            read(root / "sealed/REFERENCE_MISSING.json"),
        )
        if plan["storage"] != capacity_plan(
            (Path(plan["draft"]) / "model.safetensors").stat().st_size
        ):
            raise ValueError("Full-cap peak storage calculation changed")
        if not postflight:
            result["hardware"] = hardware_admission()
            result["quiescence"] = timing_barrier()
            memory = dict(
                line.split(":", 1)
                for line in Path("/proc/meminfo").read_text().splitlines()
            )
            result["host_available_bytes"] = (
                int(memory["MemAvailable"].split()[0]) * 1024
            )
            if result["host_available_bytes"] < 80 * 2**30:
                raise ValueError(
                    "Require 80 GiB available host memory for state validation"
                )
            from experiments.kplus_full_gsm8k.training import validate_snapshot
            from experiments.ok_coding.training import _export_equal

            state = torch.load(
                plan["original_initial_state"], map_location="cpu", weights_only=False
            )
            result["initial_snapshot"] = validate_snapshot(
                state,
                binding=result["original"]["original_binding"],
                config_text=state["config_text"],
                original_config_text=Path(plan["original_config"]).read_text(),
                completed_event=0,
                expected_contract=state["algorithm_contract"],
                expected_recipe=state["backend"]["optimizer"]["recipe"],
                expected_cuda_rng_states=1,
            )
            if state["config_text"] != audit["initial_state"]["config_text"]:
                raise ValueError("Authenticated event-zero configuration changed")
            _export_equal(state, Path(plan["draft"]))
            result["released_weight_equality"] = True
            del state
        result["storage"] = admit(
            root,
            plan["storage"],
            receipt_path=root
            / ("POSTFLIGHT_STORAGE.json" if postflight else "STORAGE_ADMISSION.json"),
        )
        result["seal_after"] = verify_seal(root)
        if torch.cuda.is_initialized():
            raise ValueError("CPU preflight unexpectedly initialized CUDA")
        result["status"] = "passed"
        return result
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        result["wall_s"] = time.perf_counter() - started
        write_json(
            root / ("POSTFLIGHT.json" if postflight else "PREFLIGHT.json"), result
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("--postflight", action="store_true")
    args = parser.parse_args()
    result = verify(args.root, postflight=args.postflight)
    print(json.dumps(dict(status=result["status"], wall_s=result["wall_s"])))


if __name__ == "__main__":
    main()
