"""Bind a completed CPU input seal to published code and actual A800 paths."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

import yaml

from online_feedback.storage import write_json
from .contract import NEW_BUDGET, data_binding, validate_plan
from .storage import GIB, identity, full_campaign_budget


def seal(args):
    source = Path(args.code).resolve()
    prepared = Path(args.prepared).resolve()
    output = Path(args.output).resolve()
    remote = Path(args.remote_root)
    remote_code = Path(args.remote_code)
    if output.exists():
        raise FileExistsError("Preserve existing input seals: " + str(output))
    version = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source,
                                     text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=source,
                                   text=True)
    if dirty:
        raise ValueError("Publish the exact implementation before sealing execution")
    input_seal = json.loads((prepared / "SEAL.json").read_text())
    for name, expected in input_seal["files"].items():
        content_identity = {key: expected[key] for key in ("bytes", "sha256")}
        if identity(prepared / name) != content_identity:
            raise ValueError("CPU preparation changed: " + name)
    output.mkdir(parents=True)
    sealed = output / "sealed"
    sealed.mkdir()
    shutil.copytree(prepared, output / "data_preparation")
    previous = Path(args.pins).resolve()
    for name in ("MODEL_PIN.json", "RUNTIME_PIN.json", "TRAINING_PIN.json"):
        shutil.copy2(previous / name, sealed / name)
    model_pin = json.loads((sealed / "MODEL_PIN.json").read_text())
    tokenizer = json.loads((prepared / "TOKENIZER.json").read_text())
    for item in tokenizer["files"]:
        name = Path(item["path"]).name
        expected = {key: item[key] for key in ("bytes", "sha256")}
        if model_pin["target"]["files"].get(name) != expected:
            raise ValueError("Prepared tokenizer differs from qualified target artifact: " + name)
    raw = yaml.safe_load((previous / "config.yaml").read_text())
    configs = {}
    for seed in (17, 29):
        config = json.loads(json.dumps(raw))
        config["training"]["seed"] = seed
        config["run_id"] = f"ok-coding-20261030-seed-{seed}"
        config["output_dir"] = str(remote / "campaign_001" / f"seed_{seed}")
        name = f"config_seed_{seed}.yaml"
        (sealed / name).write_text(yaml.safe_dump(config, sort_keys=False))
        configs[str(seed)] = str(remote / "sealed" / name)
    training = json.loads((prepared / "TRAINING_REQUESTS.json").read_text())
    evaluation = json.loads((prepared / "EVALUATION_REQUESTS.json").read_text())
    windows = json.loads((prepared / "WINDOWS.json").read_text())
    contract = json.loads((prepared / "DATA_CONTRACT.json").read_text())
    binding = data_binding(training, evaluation)
    if binding != contract["data_binding"]:
        raise ValueError("Data binding differs between preparation and execution")
    execution = {"commit": version, "dirty": False,
                 "repository": "https://github.com/ToxicMinhao/Online-Speculative-decoding-test",
                 "source_path": str(remote_code)}
    plan = {
        "configs": configs, "windows": windows, "evaluation_requests": evaluation,
        "data_binding": binding,
        "evaluation_schedule": json.loads((prepared / "EVALUATION_SCHEDULE.json").read_text()),
        "target": raw["model"]["target_model_path"],
        "draft": raw["model"]["draft_checkpoint_path"],
        "worker_python": "/data/conda_envs/minhao/adaflash_author/bin/python",
        "training_python": "/data/conda_envs/minhao/dflash_specforge_20261022/bin/python",
        "execution_source": execution, "training_seeds": [17, 29],
        "stop_step": 512, "inherited_horizon": 7424,
        "new_budget": NEW_BUDGET, "no_automatic_retry": True,
        "serving": {"block_size": 16, "max_new_tokens": 2048, "context_limit": 4096,
                    "concurrency": 1, "temperature": 0, "capture": False,
                    "precision": "bfloat16", "attention_backend": "fa3"},
        "storage": {"mode": "retain_local", "offload_assumed": False,
                    "reserve_bytes": 8 * GIB},
    }
    plan["storage"]["campaign_capacity"] = full_campaign_budget(plan)
    check = validate_plan(plan)
    write_json(sealed / "PLAN.json", plan)
    write_json(sealed / "PLAN_CHECK.json", check)
    files = {}
    tracked = subprocess.check_output(["git", "ls-files", "-z", "."], cwd=source).split(b"\0")
    for encoded in tracked:
        if not encoded:
            continue
        name = encoded.decode()
        path = source / name
        if path.is_file():
            files[str(remote_code / name)] = identity(path)
    for path in sorted(output.rglob("*")):
        if path.is_file():
            files[str(remote / path.relative_to(output))] = identity(path)
    exact = {str(remote_code / "experiments/ok_coding"):
             sorted(p.name for p in (source / "experiments/ok_coding").iterdir() if p.is_file())}
    write_json(sealed / "SEAL.json", {"execution_source": execution, "files": files,
                                     "exact_file_sets": exact})
    return {"status": "sealed_before_model_work", "execution_source": execution,
            "data_binding": binding, "files": len(files),
            "storage": plan["storage"]["campaign_capacity"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("code", "prepared", "pins", "output", "remote-root", "remote-code"):
        parser.add_argument("--" + name, required=True)
    result = seal(parser.parse_args())
    print(json.dumps({key: value for key, value in result.items() if key != "storage"}))
