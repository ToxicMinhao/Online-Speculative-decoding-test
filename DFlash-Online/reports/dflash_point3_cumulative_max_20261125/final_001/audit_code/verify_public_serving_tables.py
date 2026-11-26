"""Recheck every published Point3 contrast using only the public ordinal tables."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory
    audit = json.loads((root / "INDEPENDENT_SERVING_AUDIT.json").read_text())
    records = list(csv.DictReader((root / "PER_REQUEST.csv").open()))
    by_id = {int(row["request_ordinal"]): row for row in records}
    order = audit["bootstrap"]["public_request_order"]
    assert len(order) == len(set(order)) == len(records) == 128
    assert set(order) == set(by_id)
    draws = np.random.Generator(np.random.PCG64(20261125)).integers(0, 128, (10000, 128))
    assert hashlib.sha256(draws.astype("<i8").tobytes()).hexdigest() == audit["bootstrap"]["draw_sha256"]
    checked = 0
    for comparison in csv.DictReader((root / "CONTRASTS.csv").open()):
        for metric in ("request_latency_s", "native_generate_s", "cache_maintenance_s", "verification_calls"):
            candidate = np.array([float(by_id[key][comparison["candidate"] + "_" + metric]) for key in order])
            reference = np.array([float(by_id[key][comparison["reference"] + "_" + metric]) for key in order])
            name = ("verification_call_reduction_pct" if metric == "verification_calls" else metric + "_reduction_pct")
            point = 100 * (1 - candidate.sum() / reference.sum())
            samples = 100 * (1 - candidate[draws].sum(1) / reference[draws].sum(1))
            low, high = np.percentile(samples, [2.5, 97.5], method="linear")
            for suffix, value in (("", point), ("_ci95_low", low), ("_ci95_high", high)):
                assert math.isclose(float(comparison[name + suffix]), value, rel_tol=1e-12, abs_tol=1e-10)
                checked += 1
    repeats = list(csv.DictReader((root / "ALL_REPEATS.csv").open()))
    assert len(repeats) == 1152
    for request in records:
        for arm in ("C", "F_C_SUM", "F_C_MAX"):
            rows = [row for row in repeats if row["request_ordinal"] == request["request_ordinal"] and row["method"] == arm]
            assert sorted(int(row["repeat"]) for row in rows) == [0, 1, 2]
            for metric in ("request_latency_s", "native_generate_s", "cache_maintenance_s"):
                mean = sum(float(row[metric]) for row in rows) / 3
                assert math.isclose(mean, float(request[arm + "_" + metric]), rel_tol=1e-12, abs_tol=1e-10)
                checked += 1
            for metric in ("output_tokens", "verification_calls", "accepted_draft_tokens", "native_accepted_draft_tokens", "committed_progress", "finish_reason"):
                assert {row[metric] for row in rows} == {request[arm + "_" + metric]}
                checked += 1
    receipt = dict(status="passed", checked_scalar_fields=checked, inputs="Public ordinal tables only",
                   requests=128, scored_repeats=1152, bootstrap_draws=10000,
                   no_private_identity_required=True, gpu_work=0,
                   auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (root / "PUBLIC_TABLE_RECHECK.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
