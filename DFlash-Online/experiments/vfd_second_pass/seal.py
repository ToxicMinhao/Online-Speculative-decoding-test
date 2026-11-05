"""Seal the fixed input-only panel, original training stream and published code."""

import argparse
import json
from pathlib import Path
import shutil

from online_feedback.storage import write_json
from experiments.ok_coding.storage import identity
from .contract import (
    ARMS,
    EVENTS,
    SCHEMA,
    BUDGET,
    SERVING,
    ORIGINAL_EXECUTION,
    validate_plan,
)
from .data import digest, render_source, select_panel
from .statistics import ANALYSIS_RULES
from .storage import capacity_plan


def read(path):
    return json.loads(Path(path).read_text())


def build(root, inputs, source_manifest):
    from transformers import AutoTokenizer
    import yaml

    root, inputs = Path(root), Path(inputs)
    source = read(source_manifest)
    if source["dirty"] or len(source["commit"]) != 40:
        raise ValueError("Publish clean code before sealing")
    for path, expected in source["files"].items():
        if identity(path) != expected:
            raise ValueError("Deployed code differs from publication: " + path)
    audit = read(inputs / "INITIAL_STATE_AND_PINS_AUDIT.json")
    if (
        audit["status"] != "passed"
        or audit["model_forwards"]
        or audit["optimizer_steps"]
    ):
        raise ValueError("Require authenticated CPU-only event-zero audit")
    original_root = Path(audit["seals"]["PLAN.json"]["path"]).parent.parent
    for entry in audit["seals"].values():
        if identity(entry["path"]) != {key: entry[key] for key in ("bytes", "sha256")}:
            raise ValueError("Original source binding changed")
    original = read(original_root / "sealed/PLAN.json")
    manifest = read(original_root / "sealed/ORIGINAL_MANIFEST.json")
    if original["execution_source"]["commit"] != ORIGINAL_EXECUTION:
        raise ValueError("Wrong scientific initialization provenance")
    panel, history, training = [
        read(inputs / name)
        for name in ("PANEL.json", "HISTORY_INVENTORY.json", "TRAINING_REQUESTS.json")
    ]
    first = manifest["train_requests"][:512]
    if digest(first) != panel["training_original_rows_digest"] or len(training) != 512:
        raise ValueError("Original ordered training stream changed")
    for current, prior in zip(training, first):
        if any(current.get(key) != value for key, value in prior.items()):
            raise ValueError("Prepared training request differs from original")
    if history["unreadable_or_unavailable"]:
        raise ValueError("Available history inventory contains unreadable inputs")
    tokenizer = AutoTokenizer.from_pretrained(original["target"], local_files_only=True)
    rows, source_data = render_source(inputs / "gsm8k_test.parquet", tokenizer)
    rebuilt = select_panel(rows, training, {i: [] for i in history["used_source_rows"]})
    if (
        rebuilt["requests"] != panel["requests"]
        or source_data["sha256"] != panel["source"]["sha256"]
    ):
        raise ValueError("Pinned tokenizer/hash selection differs from prepared panel")
    reused, missing = read(inputs / "CANONICALS.json"), read(
        inputs / "REFERENCE_MISSING.json"
    )
    selected = {row["request_id"] for row in panel["requests"]}
    absent = missing["request_ids"]
    if (
        set(reused) & set(absent)
        or set(reused) | set(absent) != selected
        or len(set(absent)) != len(absent)
        or missing["count"] != len(absent)
    ):
        raise ValueError(
            "Canonical reuse/missing identities do not partition the sealed panel"
        )
    root.mkdir(parents=True, exist_ok=False)
    sealed = root / "sealed"
    sealed.mkdir()
    for name in (
        "HANDOFF.md",
        "PANEL.json",
        "TRAINING_REQUESTS.json",
        "HISTORY_INVENTORY.json",
        "CANONICALS.json",
        "REFERENCE_MISSING.json",
        "REFERENCE_REUSE_AUDIT.json",
        "INITIAL_STATE_AND_PINS_AUDIT.json",
        "gsm8k_test.parquet",
    ):
        shutil.copy2(inputs / name, sealed / name)
    for name in ("MODEL_PIN.json", "RUNTIME_PIN.json", "TRAINING_PIN.json"):
        shutil.copy2(original_root / "sealed" / name, sealed / name)
    original_config = original_root / "sealed/config_KPLUS.yaml"
    shutil.copy2(original_config, sealed / "ORIGINAL_CONFIG.yaml")
    cfg = yaml.safe_load(original_config.read_text())
    cfg.update(
        run_id="vfd-second-pass-four-event", output_dir=str(root / "campaign_001")
    )
    (sealed / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    pins = read(sealed / "MODEL_PIN.json")
    plan = dict(
        schema=SCHEMA,
        arms=list(ARMS),
        events=EVENTS,
        seed=17,
        budget=BUDGET,
        serving=SERVING,
        original_execution=ORIGINAL_EXECUTION,
        automatic_retry=False,
        inherited_optimizer_horizon=7424,
        execution_source={
            key: source[key] for key in ("commit", "dirty", "repository", "branch")
        },
        target=original["target"],
        draft=original["draft"],
        worker_python=original["worker_python"],
        training_python="/data/conda_envs/minhao/dflash_specforge_20261022/bin/python",
        original_root=str(original_root),
        original_initial_state=audit["initial_state"]["identity"]["path"],
        original_config=str(sealed / "ORIGINAL_CONFIG.yaml"),
        config=str(sealed / "config.yaml"),
        training_requests=training,
        evaluation_requests=panel["requests"],
        analysis=ANALYSIS_RULES,
        policy_version="vfd-s2-v1",
        event_hash_identity="integer events 1..4 rendered as decimal strings",
        storage=capacity_plan(pins["draft"]["files"]["model.safetensors"]["bytes"]),
        final_checkpoints={
            arm: str(root / f"campaign_001/exports/{arm}_event_004") for arm in ARMS
        },
        canonical_path=str(root / "campaign_001/canonicals/CANONICALS.json"),
        output_root=str(root),
        native_gpu="0",
        training_gpu="1",
        source_answers_used=False,
        qualification=dict(
            score_forwards=8, training_presentations=4, optimizer_steps=2
        ),
        reference_missing=read(sealed / "REFERENCE_MISSING.json")["count"],
    )
    check = validate_plan(plan)
    write_json(sealed / "PLAN.json", plan)
    write_json(sealed / "SOURCE.json", source)
    write_json(sealed / "CONTRACT_CHECK.json", check)
    write_json(
        sealed / "PRECOMMITTED_ANALYSIS.json",
        dict(
            panel_digest=digest(panel),
            policy_version=plan["policy_version"],
            analysis=ANALYSIS_RULES,
            analysis_digest=digest(ANALYSIS_RULES),
        ),
    )
    files = {
        **source["files"],
        **{str(path): identity(path) for path in sorted(sealed.iterdir())},
    }
    write_json(
        sealed / "SEAL.json",
        dict(execution_source=plan["execution_source"], files=files),
    )
    return dict(
        status="sealed",
        root=str(root),
        commit=source["commit"],
        requests=200,
        reference_generations=plan["reference_missing"],
        storage=plan["storage"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--source-manifest", required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.root, args.inputs, args.source_manifest)))
