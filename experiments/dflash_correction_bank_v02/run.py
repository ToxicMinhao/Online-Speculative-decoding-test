#!/usr/bin/env python3
"""Prepare a CPU plan, or explicitly execute the reviewed qualification/campaign."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.dflash_per_turn.dataset import load_streams, MANIFEST_SHA256, PACKAGE_SHA256
from experiments.dflash_per_turn.framing import load_pinned_tokenizer
from experiments.dflash_per_turn.run import qualify_text, write_json
from experiments.dflash_correction_bank_v02.experiment import plan, run_stream, require, digest
from experiments.dflash_correction_bank_v02.report import write_reports

HERE = Path(__file__).resolve().parent


def source_inventory():
    paths = []
    for name in ("dflash_correction_bank_v02", "dflash_correction_bank", "dflash_per_turn", "dflash_two_update", "dflash_two_update_followup"):
        paths.extend((HERE.parent/name).glob("*.py"))
        paths.extend((HERE.parent/name).glob("*.json"))
    return {str(p.relative_to(HERE.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths) if not p.name.startswith("test_")}


def run_campaign(streams, backend_factory, out, *, config, emit_factory=None):
    plan(streams)  # Check exact manifest order and all six counts even for a restarted campaign.
    out = Path(out)
    started = time.monotonic()
    campaign = dict(status="running", qualification=None, results=[], actual_wall_s=0.)
    completed = []

    def execute(stream, index, qualification):
        destination = out/("qualification" if qualification else "campaign")/stream["stream_id"]
        destination.mkdir(parents=True, exist_ok=False)
        def emit(kind, data):
            with (destination/"EVENTS.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(dict(event=kind, data=data), ensure_ascii=False, allow_nan=False)+"\n")
                handle.flush(); os.fsync(handle.fileno())
        if emit_factory is not None:
            emit = emit_factory(destination)
        backend = backend_factory({**config, "output_root": str(destination/"native"),
            "dataset_source": stream["source"], "qualification": qualification})
        result = run_stream(stream, backend, stream_index=index, qualification=qualification, emit=emit,
            snapshot=lambda value: write_json(destination/"PROGRESS.json", value, replace=True))
        write_json(destination/"RESULT.json", result)
        write_reports([result], destination/"report")
        if result["status"] == "complete" and result["implementation_checks"]["status"] == "passed":
            receipt_retention = backend.finalize_retention(result_path=destination/"RESULT.json")
            write_json(destination/"FINAL_RETENTION.json", receipt_retention)
        receipt = dict(episode_id=stream["stream_id"], status=result["status"],
            result_path=str(destination/"RESULT.json"),
            result_sha256=hashlib.sha256((destination/"RESULT.json").read_bytes()).hexdigest(),
            implementation_checks=result["implementation_checks"], qualification_checks=result.get("qualification_checks"))
        return result, receipt

    result, receipt = execute(streams[0], 0, True)
    campaign["qualification"] = receipt
    if result["status"] != "complete" or result["implementation_checks"]["status"] != "passed":
        campaign.update(status="stopped", reason="Correctness qualification failed; no performance threshold or automatic retuning.")
    else:
        # New backend, empty bank/memory, released OSD and seed17, including the first stream again.
        config = {**config, "qualified_cost_profile": result["qualification_checks"]["cost_profile_binding"]}
        for index, stream in enumerate(streams):
            result, receipt = execute(stream, index, False)
            completed.append(result); campaign["results"].append(receipt)
            campaign["actual_wall_s"] = time.monotonic()-started
            write_json(out/"CAMPAIGN_STATE.json", campaign, replace=True)
            if result["status"] != "complete" or result["implementation_checks"]["status"] != "passed":
                campaign.update(status="stopped", reason="Implementation/output-preservation failure; preserve raw evidence and investigate.")
                break
        else:
            campaign["status"] = "complete"
    campaign["actual_wall_s"] = time.monotonic()-started
    campaign["qualification_excluded_from_main_results"] = True
    write_json(out/"CAMPAIGN_STATE.json", campaign, replace=True)
    write_reports(completed, out/"report")
    return campaign


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "qualify-text", "run"))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--runtime-config", type=Path)
    parser.add_argument("--reviewed-code-sha256")
    parser.add_argument("--allow-gpu-execution", action="store_true")
    args = parser.parse_args(argv)
    require(not args.out.exists(), "Use a fresh output directory; do not overwrite previous attempts")
    require(not args.out.resolve().is_relative_to(HERE.parents[1]), "Runtime data belongs outside the public checkout")
    streams = load_streams(args.dataset)
    protocol = plan(streams)
    inventory = source_inventory(); reviewed = digest(inventory)
    if args.action == "run":
        require(args.allow_gpu_execution and args.runtime_config and args.tokenizer,
                "GPU execution needs reviewed code, explicit run flag, tokenizer and runtime configuration")
        require(args.reviewed_code_sha256 == reviewed, "Source differs from reviewed code")
    else:
        require(not (args.allow_gpu_execution or args.runtime_config or args.reviewed_code_sha256),
                "CPU metadata/text checks cannot start a GPU campaign")
    args.out.mkdir(parents=True)
    write_json(args.out/"PLAN.json", protocol)
    write_json(args.out/"SOURCE_MANIFEST.json", dict(code_sha256=inventory, reviewed_code_sha256=reviewed,
        dataset_manifest_sha256=MANIFEST_SHA256, dataset_package_sha256=PACKAGE_SHA256))
    if args.action == "plan":
        print(json.dumps(dict(status="review_only", gpu_started=False, reviewed_code_sha256=reviewed,
                              request_contexts=116, bank_update_events=110, osd_update_events=110, timed_replays=1044)))
        return 0
    require(args.tokenizer is not None, "Pinned tokenizer path required")
    tokenizer = load_pinned_tokenizer(args.tokenizer)
    write_json(args.out/"TEXT_QUALIFICATION.json", qualify_text(streams, tokenizer))
    del tokenizer
    if args.action == "qualify-text":
        print(json.dumps(dict(status="passed", gpu_started=False))); return 0
    config = json.loads(args.runtime_config.read_text())
    require(config.get("output_cap") == 1024 and config.get("block_size") == 10,
            "Pinned output cap or draft block size changed")
    write_json(args.out/"RUNTIME_CONFIG.json", config)
    # All model/runtime imports and execution are beyond the explicit reviewed-run gate.
    from experiments.dflash_correction_bank_v02.native import NativeBackend, admit_campaign_storage
    admission_config = {**config, "output_root": str(args.out/"qualification")}
    try:
        admission = admit_campaign_storage(admission_config)
    except ValueError as error:
        write_json(args.out/"STORAGE_ADMISSION.json", dict(status="blocked", gpu_started=False, error=str(error)))
        raise
    write_json(args.out/"STORAGE_ADMISSION.json", admission)
    campaign = run_campaign(streams, NativeBackend, args.out, config=config)
    print(json.dumps(campaign))
    return 0 if campaign["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
