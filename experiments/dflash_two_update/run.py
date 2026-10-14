#!/usr/bin/env python3
"""Plan without importing a GPU runtime; run one episode only when explicit."""
import argparse
import hashlib
import json
from pathlib import Path
import os
import sys

from experiment import load_dataset, plan, run_episode, require
from report import summarize


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "run"))
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--streams", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episode")
    parser.add_argument("--runtime-config", type=Path)
    parser.add_argument("--allow-gpu-execution", action="store_true")
    args = parser.parse_args()
    require(not args.out.exists(), "Use a fresh output directory; previous attempts must remain")
    repository = Path(__file__).resolve().parents[2]
    require(not args.out.resolve().is_relative_to(repository), "Runtime/data artifacts belong outside public code checkout")
    streams = load_dataset(args.selection, args.streams)
    protocol = plan(streams)
    args.out.mkdir(parents=True)
    code = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}
    write(args.out / "PLAN.json", protocol)
    write(args.out / "SOURCE_MANIFEST.json", {"code_sha256": code, "selection_sha256": hashlib.sha256(args.selection.read_bytes()).hexdigest(),
          "streams_file_sha256": hashlib.sha256(args.streams.read_bytes()).hexdigest()})
    if args.mode == "plan":
        require(not args.allow_gpu_execution and not args.runtime_config and not args.episode,
                "Plan mode does not accept execution settings")
        print(json.dumps({"status": "plan_only", "episodes": 10, "adaptation_events": 20}))
        return 0
    require(args.allow_gpu_execution and args.runtime_config is not None and args.episode,
            "Run needs an explicit episode, runtime config, and --allow-gpu-execution")
    require(args.episode in {p["episode_id"] for p in protocol["episodes"]}, "Only development episodes are executable")
    config = json.loads(args.runtime_config.read_text())
    config["output_root"] = str(args.out.resolve() / "native")
    write(args.out / "RUNTIME_CONFIG.json", config)
    # Lazy import: constructing the plan and CPU tests never initializes CUDA.
    from native import NativeBackend
    backend = NativeBackend(config)
    event_path = args.out / "EVENTS.jsonl"
    def emit(kind, data):
        with event_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"event": kind, "data": data}, ensure_ascii=False, allow_nan=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
    result = run_episode([r for r in streams if r["stream_id"] == args.episode], backend, emit=emit)
    write(args.out / "RESULT.json", result)
    write(args.out / "REPORT.json", summarize([result]))
    print(json.dumps({"status": result["status"], "episode": args.episode, "report": str(args.out / "REPORT.json")}))
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
