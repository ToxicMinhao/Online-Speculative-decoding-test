"""Compare a primary reduction with already saved independent raw-reader results."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def compare(independent: Path, primary: Path, output: Path):
    checked, errors, consumed = 0, [], {}

    def read(path, csv_file=False):
        content = path.read_bytes()
        consumed[str(path)] = {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
        return list(csv.DictReader(content.decode().splitlines())) if csv_file else json.loads(content)

    def check(condition, location):
        nonlocal checked
        checked += 1
        if not condition:
            errors.append(location)

    def equal(actual, expected, location):
        if isinstance(expected, float):
            check(isinstance(actual, (float, int)) and math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12), location)
        else:
            check(actual == expected, location)

    oracle = read(independent / "INDEPENDENT_RESULTS.json")
    result = read(primary / "SUMMARY.json")
    check(oracle["status"] == "passed", "independent_status")
    check(result["status"] == "completed", "primary_status")
    own_rows = read(independent / "INDEPENDENT_ROWS.csv", True)
    main_rows = read(primary / "PER_REQUEST.csv", True)
    own_map = {(r["source_id"], r["request_id"], r["arm"]): r for r in own_rows}
    main_map = {(r["source_id"], r["request_id"], r["arm"]): r for r in main_rows}
    check(len(own_rows) == len(own_map) == len(main_rows) == len(main_map) == 384, "unique384rows")
    check(set(own_map) == set(main_map), "exact_source_request_arm_set")
    for key in sorted(set(own_map) & set(main_map)):
        for field in ("V", "P", "A", "M", "output_tokens", "request_index", "native_source_id", "finish_reason", "cap_hit", "valid"):
            equal(main_map[key][field], own_map[key][field], f"row:{key}:{field}")
        equal(main_map[key]["zero_calls"], own_map[key]["zero_call"], f"row:{key}:zero_call")
        equal(main_map[key]["stop_clipped_blocks"], own_map[key]["terminal_clipped_blocks"], f"row:{key}:clipped_blocks")
    own_pairs = read(independent / "INDEPENDENT_PAIRED.csv", True)
    main_pairs = read(primary / "PAIRED_REQUESTS.csv", True)
    om = {(r["source_id"], r["request_id"]): r for r in own_pairs}
    pm = {(r["source_id"], r["request_id"]): r for r in main_pairs}
    check(len(own_pairs) == len(om) == len(main_pairs) == len(pm) == 128, "unique128pairs")
    check(set(om) == set(pm), "exact_paired_population")
    for key in sorted(set(om) & set(pm)):
        check(pm[key]["valid"] == "True", f"pair:{key}:valid")
        for arm in ("N", "E", "S"):
            for field in ("V", "P", "A", "M", "output_tokens", "cap_hit"):
                equal(pm[key][f"{arm}_{field}"], om[key][f"{arm}_{field}"], f"pair:{key}:{arm}:{field}")
        for contrast in ("E_vs_N", "S_vs_N", "E_vs_S"):
            equal(pm[key][f"{contrast}_calls_saved"], om[key][f"{contrast}_calls_saved"], f"pair:{key}:{contrast}")
    saved = oracle["summary"]
    for arm in ("N", "E", "S"):
        for field in ("V", "P", "A", "M", "output_tokens", "requests", "eos", "progress_per_call"):
            equal(result["totals"][arm][field], saved["totals"][arm][field], f"total:{arm}:{field}")
        for field, old in (("native_acceptance", "pooled_native_acceptance"), ("cap_hits", "cap"), ("zero_calls", "zero_call")):
            equal(result["totals"][arm][field], saved["totals"][arm][old], f"total:{arm}:{field}")
    for contrast in ("E_vs_N", "S_vs_N", "E_vs_S"):
        main, own = result["comparisons"][contrast], saved["contrasts"][contrast]
        for field in ("net_calls_saved", "gross_saved", "gross_added"):
            equal(main[field], own[field], f"comparison:{contrast}:{field}")
        for field, old in (("fewer_calls", "fewer"), ("more_calls", "more"), ("tied_calls", "tied")):
            equal(main[field], own[old], f"comparison:{contrast}:{field}")
        for field, old in (("absolute_acceptance_change", "absolute_acceptance"), ("relative_acceptance_change_pct", "relative_acceptance_pct"), ("call_reduction_pct", "call_reduction_pct")):
            equal(main[field], own[old]["point"], f"comparison:{contrast}:{field}:point")
            equal(main["intervals"][field]["undefined_draws"], own[old]["undefined_draws"], f"comparison:{contrast}:{field}:undefined")
            for i in range(2):
                equal(main["intervals"][field]["interval"][i], own[old]["ci95"][i], f"comparison:{contrast}:{field}:ci{i}")
    for field in ("seed", "draws", "numpy_version", "percentile_method", "ordered_source_list_sha256", "draw_matrix_sha256"):
        equal(result["bootstrap"][field], saved["bootstrap"][field], f"bootstrap:{field}")
    receipt = {"status": "passed" if not errors else "failed", "checks": checked, "errors": errors,
               "independent_results_existed_before_primary_source_review": True,
               "independent_count_checks": oracle["checks"], "independent_count_errors": oracle["error_count"],
               "count_and_identity_requirement": "exact", "floating_absolute_tolerance": 1e-12,
               "new_model_work": 0, "scientific_replication": False, "consumed_inputs": consumed}
    output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--independent", type=Path, required=True)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = compare(args.independent, args.primary, args.output)
    print(json.dumps({"status": receipt["status"], "checks": receipt["checks"], "errors": receipt["errors"]}))
    raise SystemExit(0 if receipt["status"] == "passed" else 1)
