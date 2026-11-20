"""Run the bounded CPU checks and save compact, source-bound receipts."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import unittest


SOURCE_FILES = (
    "online_feedback/first_error.py",
    "online_feedback/objective.py",
    "online_feedback/kplus.py",
    "online_feedback/prefix_allocation.py",
    "training/specforge/algorithms/common/dflash_family_model.py",
)
EXECUTION_COMMIT = "b4459820b3822c3d760bb3fd880e4d139fa2bbc2"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source = args.source_root.resolve()
    here = Path(__file__).resolve().parent
    if args.output is None:
        args.output = source / "reports/dflash_fresh_mathqa_c_arepo_osd_20261119"
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(here))
    import torch
    from direct_support_score import operation_contract

    torch.set_num_threads(1)
    source_hashes = {name: sha256(source / name) for name in SOURCE_FILES}
    source_commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    source_prefix = subprocess.check_output(["git", "-C", str(source), "rev-parse", "--show-prefix"], text=True).strip()
    execution_hashes = {
        name: hashlib.sha256(subprocess.check_output(
            ["git", "-C", str(source), "show", f"{EXECUTION_COMMIT}:{source_prefix}{name}"])).hexdigest()
        for name in SOURCE_FILES
    }
    matches_execution = source_hashes == execution_hashes
    started = datetime.now(timezone.utc).isoformat()
    clock = time.perf_counter()
    suite = unittest.defaultTestLoader.loadTestsFromName("test_direct_support_score")
    with (args.output / "CPU_TEST_LOG.txt").open("w") as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    elapsed = time.perf_counter() - clock
    unchanged = source_hashes == {name: sha256(source / name) for name in SOURCE_FILES}
    receipt = {
        "status": "PASS" if result.wasSuccessful() and unchanged and matches_execution and not torch.cuda.is_initialized() else "FAIL",
        "scope": "CPU synthetic adapted-reference qualification; no official reproduction or efficacy claim",
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed,
        "tests_run": result.testsRun,
        "failures": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped),
        "randomized_oracle_cases": 100,
        "python": platform.python_version(), "python_executable_name": Path(sys.executable).name,
        "torch": torch.__version__, "torch_threads": torch.get_num_threads(),
        "cuda_initialized": torch.cuda.is_initialized(),
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "optimizer_steps": 0, "model_generations": 0, "gpu_training": 0,
        "source_checkout_commit": source_commit,
        "source_git_prefix": source_prefix,
        "scientific_execution_commit": EXECUTION_COMMIT,
        "source_matches_execution_commit": matches_execution,
        "source_sha256": source_hashes, "source_files_unchanged": unchanged,
        "reference_sha256": {p.name: sha256(p) for p in sorted(here.glob("*.py"))},
        "test_log_sha256": sha256(args.output / "CPU_TEST_LOG.txt"),
        "verification_independence": "Separate scalar implementation by the same implementation agent; no blind review claimed",
    }
    (args.output / "CPU_RESULTS.json").write_text(json.dumps(receipt, indent=2) + "\n")
    (args.output / "OPERATION_CONTRACT.json").write_text(json.dumps(operation_contract(), indent=2) + "\n")
    print(json.dumps({k: receipt[k] for k in ("status", "tests_run", "failures", "errors", "elapsed_seconds", "cuda_initialized")}))
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
