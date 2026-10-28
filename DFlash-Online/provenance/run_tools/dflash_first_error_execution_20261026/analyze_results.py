"""Local descriptive recalculation; reads retained results, never runs a model."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
RUN = ROOT / "raw/campaign_001"
REPORT = RUN / "evaluation/report"
ARMS = ("O", "C", "K", "A")
PAIRS = (("K", "O"), ("A", "C"), ("C", "O"), ("A", "K"))


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def close(a, b):
    assert np.isclose(a, b, rtol=1e-12, atol=1e-10), (a, b)


def reduction(candidate, reference):
    return 100.0 * (reference - candidate) / reference


def main():
    summary = json.loads((REPORT / "SUMMARY.json").read_text())
    per_request = read_csv(REPORT / "PER_REQUEST.csv")
    repeats = read_csv(REPORT / "ALL_REPEATS.csv")
    assert len(per_request) == 64 and len(repeats) == 768
    ids = [row["request_id"] for row in per_request]
    assert len(set(ids)) == 64
    assert all(row["valid"] == row["complete"] == "True" and not row["errors"] for row in per_request)
    slots = {(row["request_id"], row["method"], int(row["repeat"])): row for row in repeats}
    assert len(slots) == 768
    assert all(row["report_valid"] == row["valid_measurement"] == "True" for row in repeats)
    assert all(row["errors"] in ("", "[]") and row["report_errors"] in ("", "[]") for row in repeats)

    calls = np.empty((64, 4), dtype=np.int64)
    times = np.empty((64, 4, 3), dtype=np.float64)
    totals = {}
    for j, arm in enumerate(ARMS):
        for i, rid in enumerate(ids):
            rows = [slots[rid, arm, repeat] for repeat in range(3)]
            for name in ("verification_calls", "accepted_draft_tokens", "committed_progress", "output_tokens", "finish_reason", "cap_hit"):
                assert len({row[name] for row in rows}) == 1, (rid, arm, name)
            calls[i, j] = int(rows[0]["verification_calls"])
            times[i, j, :] = [float(row["request_latency_s"]) for row in rows]
            close(calls[i, j], int(per_request[i][f"{arm}_verification_calls"]))
            close(times[i, j].mean(), float(per_request[i][f"{arm}_mean_request_latency_s"]))
        totals[arm] = {
            "verification_calls_one_trajectory": int(calls[:, j].sum()),
            "summed_mean_request_time_s": float(times[:, j, :].mean(axis=1).sum()),
            "repeat_total_request_time_s": times[:, j, :].sum(axis=0).tolist(),
        }
        close(totals[arm]["verification_calls_one_trajectory"], summary[arm]["verification_calls"])
        close(totals[arm]["summed_mean_request_time_s"], summary[arm]["summed_mean_request_latency_s"])

    mean_times = times.mean(axis=2)
    rng = np.random.default_rng(20261027)
    indices = rng.integers(0, 64, size=(10_000, 64))
    contrasts = {}
    for candidate, reference in PAIRS:
        j, k = ARMS.index(candidate), ARMS.index(reference)
        call_delta = calls[:, k] - calls[:, j]
        time_delta = mean_times[:, k] - mean_times[:, j]
        call_dist = reduction(calls[indices, j].sum(axis=1), calls[indices, k].sum(axis=1))
        time_dist = reduction(mean_times[indices, j].sum(axis=1), mean_times[indices, k].sum(axis=1))
        call_loo = reduction(calls[:, j].sum() - calls[:, j], calls[:, k].sum() - calls[:, k])
        time_loo = reduction(mean_times[:, j].sum() - mean_times[:, j], mean_times[:, k].sum() - mean_times[:, k])
        name = f"{candidate}_vs_{reference}"
        contrasts[name] = {
            "call_reduction_pct": float(reduction(calls[:, j].sum(), calls[:, k].sum())),
            "request_time_reduction_pct": float(reduction(mean_times[:, j].sum(), mean_times[:, k].sum())),
            "net_calls_saved": int(call_delta.sum()),
            "gross_calls_saved": int(call_delta.clip(min=0).sum()),
            "gross_extra_calls": int((-call_delta).clip(min=0).sum()),
            "request_call_improved_regressed_tied": [int((call_delta > 0).sum()), int((call_delta < 0).sum()), int((call_delta == 0).sum())],
            "net_request_time_saved_s": float(time_delta.sum()),
            "request_time_improved_regressed_tied": [int((time_delta > 0).sum()), int((time_delta < 0).sum()), int((time_delta == 0).sum())],
            "repeat_request_time_reduction_pct": reduction(times[:, j, :].sum(axis=0), times[:, k, :].sum(axis=0)).tolist(),
            "request_mix_bootstrap_95pct_call_reduction_range": np.quantile(call_dist, [0.025, 0.975]).tolist(),
            "request_mix_bootstrap_95pct_time_reduction_range": np.quantile(time_dist, [0.025, 0.975]).tolist(),
            "leave_one_request_out_call_reduction_range": [float(call_loo.min()), float(call_loo.max())],
            "leave_one_request_out_time_reduction_range": [float(time_loo.min()), float(time_loo.max())],
        }
        close(contrasts[name]["call_reduction_pct"], summary["comparisons"][name]["call_reduction_pct"])
        close(contrasts[name]["request_time_reduction_pct"], summary["comparisons"][name]["request_time_reduction_pct"])

    exposure = {}
    costs = json.loads((RUN / "COSTS.json").read_text())
    for arm in ("C", "K", "A"):
        result = json.loads((RUN / arm / "events/event_001/RESULT.json").read_text())
        records = result["records"]
        boundary = [row for row in records if row["optimizer_stepped"]]
        assert len(records) == 256 and len(boundary) == 128
        assert all(row["metrics_scope"] == "accumulation_window" for row in boundary)
        computed = {key: sum(row["metrics"][key] for row in boundary) for key in result["selection_totals"]}
        for key, value in computed.items():
            close(value, result["selection_totals"][key])
            close(value, costs[arm]["selection_totals"][key])
        total_blocks = computed["empty_blocks"] + computed["no_error_blocks"] + sum(computed[f"first_error_position_{i}"] for i in range(1, 16))
        exposure[arm] = {
            "totals_recalculated_once_per_optimizer_boundary": computed,
            "blocks": int(total_blocks),
            "selected_occurrence_pct": 100 * computed["selected_position_occurrences"] / computed["available_position_occurrences"],
            "selected_weight_pct": 100 * computed["selected_position_weight"] / computed["available_position_weight"],
            "no_error_block_pct": 100 * computed["no_error_blocks"] / total_blocks,
            "passes": [],
        }
        for p in range(2):
            rows = [row for row in boundary if row["pass"] == p]
            partial = {key: sum(row["metrics"][key] for row in rows) for key in computed}
            exposure[arm]["passes"].append({
                "pass": p + 1,
                "available_occurrences": partial["available_position_occurrences"],
                "selected_occurrences": partial["selected_position_occurrences"],
                "selected_occurrence_pct": 100 * partial["selected_position_occurrences"] / partial["available_position_occurrences"],
                "selected_weight_pct": 100 * partial["selected_position_weight"] / partial["available_position_weight"],
                "no_error_blocks": partial["no_error_blocks"],
            })

    files = [REPORT / name for name in ("PER_REQUEST.csv", "ALL_REPEATS.csv", "REPEAT_TOTALS.csv", "SUMMARY.json")]
    files += [RUN / "COSTS.json"] + [RUN / arm / "events/event_001/RESULT.json" for arm in "CKA"]
    output = {
        "scope": "Local descriptive reanalysis; no model/GPU rerun; source reports and boundary metrics verified numerically",
        "analysis_unit": "64 paired development requests; 3 timing repeats do not constitute extra requests or training seeds",
        "training_seed": 17,
        "bootstrap": {
            "resamples": 10000, "seed": 20261027, "numpy_version": np.__version__,
            "method": "paired request resampling with replacement, same 64-index draws for all contrasts; ratio of sums; call count once/request; mean of 3 timings/request; percentile 2.5/97.5",
            "interpretation": "Exploratory request-mix sensitivity conditional on one trained seed and the reused 64-request panel; not population/seed confidence, no hypothesis tests or significance claims",
        },
        "checks": {"64_complete_unique_requests": True, "768_unique_valid_measurements": True, "repeat_count_and_termination_fields_identical": True, "summary_recomputed": True, "boundary_exposure_and_cost_totals_recomputed": True},
        "arms": totals, "contrasts": contrasts, "training_exposure": exposure,
        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
    }
    (ROOT / "RESULT_INTERPRETATION.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"arms": totals, "contrasts": contrasts, "exposure_rates": {arm: {k: v for k, v in data.items() if k != "totals_recalculated_once_per_optimizer_boundary"} for arm, data in exposure.items()}}, indent=2))


if __name__ == "__main__":
    main()
