#!/usr/bin/env python3
"""Prepare a metadata-only plan, or explicitly run the reviewed six-stream pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

if __package__:
    from .dataset import load_streams, MANIFEST_SHA256, PACKAGE_SHA256
    from .experiment import plan, run_stream, require, digest
    from .framing import load_pinned_tokenizer, conservative_budget
    from .report import write_reports
else:
    from dataset import load_streams, MANIFEST_SHA256, PACKAGE_SHA256
    from experiment import plan, run_stream, require, digest
    from framing import load_pinned_tokenizer, conservative_budget
    from report import write_reports

HERE = Path(__file__).resolve().parent


def source_inventory():
    """Bind this protocol and reused historical scientific implementations."""
    paths = list(HERE.glob("*.py"))
    for name in ("dflash_two_update", "dflash_two_update_followup"):
        paths.extend((HERE.parent / name).glob("*.py"))
        paths.extend((HERE.parent / name).glob("*.json"))
    return {str(p.relative_to(HERE.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths) if not p.name.startswith("test_")}


def write_json(path, value, *, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not replace:
        require(not path.exists(), "Preserve existing artifact: " + str(path))
    temporary = path.with_name(path.name + ".pending")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def qualify_text(streams, tokenizer):
    receipts = []
    for stream in streams:
        bounds = conservative_budget(tokenizer, [row["prompt"] for row in stream["requests"]])
        receipts.append(dict(stream_id=stream["stream_id"], request_count=len(bounds),
            maximum_input_bound=max(r["input_tokens_upper_bound"] for r in bounds),
            maximum_total_bound=max(r["including_current_answer_and_speculation_upper_bound"] for r in bounds),
            source_text_sha256=[hashlib.sha256(r["prompt"].encode()).hexdigest() for r in stream["requests"]],
            status="passed"))
    return dict(status="passed", streams=receipts, model_loaded=False,
                scope="Pinned source equality, lossless tokenizer round trips and conservative bounds; not GPU qualification.")


def run_campaign(streams, backend_factory, out, *, config, emit_factory=None):
    """Serial timed work; first complete stream is the automatic admission gate."""
    results, case_budget = [], {"harmful": 0, "beneficial": 0}
    started = time.monotonic()
    campaign = dict(status="running", results=[], case_budget=case_budget, first_stream_gate=None)
    for index, stream in enumerate(streams):
        stream_out = Path(out) / stream["stream_id"]
        stream_out.mkdir(exist_ok=False)
        def emit(kind, data):
            with (stream_out / "EVENTS.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(dict(event=kind, data=data), ensure_ascii=False, allow_nan=False) + "\n")
                handle.flush(); os.fsync(handle.fileno())
        if emit_factory is not None:
            emit = emit_factory(stream_out)
        backend = backend_factory({**config, "output_root": str(stream_out / "native"),
            "remaining_diagnostic_cases": 3 - sum(case_budget.values()), "dataset_source": stream["source"]})
        result = run_stream(stream, backend, stream_index=index, case_budget=case_budget, emit=emit,
            snapshot=lambda value: write_json(stream_out / "PROGRESS.json", value, replace=True))
        results.append(result)
        write_json(stream_out / "RESULT.json", result)
        write_reports([result], stream_out / "report")
        campaign["results"].append({"episode_id": stream["stream_id"], "status": result["status"],
            "result_path": str(stream_out / "RESULT.json"),
            "result_sha256": hashlib.sha256((stream_out / "RESULT.json").read_bytes()).hexdigest(),
            "implementation_checks": result["implementation_checks"]})
        if index == 0:
            campaign["first_stream_gate"] = result["implementation_checks"]
        campaign["actual_wall_s"] = time.monotonic() - started
        write_json(Path(out) / "CAMPAIGN_STATE.json", campaign, replace=True)
        if result["status"] != "complete" or result["implementation_checks"]["status"] != "passed":
            campaign.update(status="stopped", reason="Implementation/output-preservation gate failed; no automatic retry.")
            break
        # Native retirement is optional; it must verify the already-durable
        # complete result before removing any now-unneeded rolling state.
        if hasattr(backend, "finalize_retention"):
            receipt = backend.finalize_retention(result_path=stream_out / "RESULT.json")
            write_json(stream_out / "RETENTION.json", receipt)
    else:
        campaign["status"] = "complete"
    campaign["actual_wall_s"] = time.monotonic() - started
    write_json(Path(out) / "CAMPAIGN_STATE.json", campaign, replace=True)
    write_reports(results, Path(out) / "report")
    return campaign


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "qualify-text", "run"))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--runtime-config", type=Path)
    parser.add_argument("--allow-gpu-execution", action="store_true")
    parser.add_argument("--reviewed-code-sha256")
    args = parser.parse_args(argv)
    require(not args.out.exists(), "Use a fresh output directory; preserve prior attempts")
    require(not args.out.resolve().is_relative_to(HERE.parents[1]), "Runtime artifacts belong outside public checkout")
    streams = load_streams(args.dataset)
    protocol = plan(streams)
    sources = source_inventory()
    code_hash = digest(sources)
    if args.action != "run":
        require(not (args.runtime_config or args.allow_gpu_execution or args.reviewed_code_sha256),
                "Metadata/tokenizer actions cannot accept GPU execution arguments")
    else:
        require(args.allow_gpu_execution and args.runtime_config and args.tokenizer,
                "Execution requires reviewed code, explicit GPU flag, tokenizer and runtime config")
        require(args.reviewed_code_sha256 == code_hash, "Code differs from the reviewed source receipt")
    if args.action in ("qualify-text", "run"):
        require(args.tokenizer is not None, "Pinned tokenizer path required")
    args.out.mkdir(parents=True)
    write_json(args.out / "PLAN.json", protocol)
    write_json(args.out / "SOURCE_MANIFEST.json", dict(code_sha256=sources, reviewed_code_sha256=code_hash,
        dataset_manifest_sha256=MANIFEST_SHA256, dataset_package_sha256=PACKAGE_SHA256))
    if args.action == "plan":
        print(json.dumps(dict(status="review_only", gpu_started=False, reviewed_code_sha256=code_hash,
                              requests=116, updates=110, optimizer_steps=220, timed_replays=990)))
        return 0
    tokenizer = load_pinned_tokenizer(args.tokenizer)
    write_json(args.out / "TEXT_QUALIFICATION.json", qualify_text(streams, tokenizer))
    del tokenizer
    if args.action == "qualify-text":
        print(json.dumps(dict(status="passed", gpu_started=False, requests=116)))
        return 0
    config = json.loads(args.runtime_config.read_text())
    require(config.get("output_cap") == 1024 and config.get("block_size") == 10,
            "Per-turn cap/block geometry changed")
    # Source/environment/model/storage checks and all GPU imports occur only
    # after the user's later execution approval, expressed by explicit run flags.
    if __package__:
        from .native import NativeBackend
    else:
        from native import NativeBackend
    write_json(args.out / "RUNTIME_CONFIG.json", config)
    result = run_campaign(streams, NativeBackend, args.out, config=config)
    print(json.dumps(result))
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
