#!/usr/bin/env python3
"""CPU-only scalar audit. Reads a sealed, one-time inventory; never searches for data.

Usage: python analyze_metrics.py \
  --inventory <study>/reviews/COMPACT_INVENTORY_ONCE.json --private-output <study>/reviews
Only standard-library JSON/CSV/scalar arithmetic is used. No tensor or model imports.
"""
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

EPS = 1e-6
REPORTS = {"PR23": "dflash_full_k_gsm8k_20261031", "PR25": "dflash_kplus_20261103", "PR32": "dflash_role_loss_pilot_20261107"}
COMMITS = {"PR23": "0698eeb2ddd97b9e2ac6953989dc82acbc0a1e34", "PR25": "e0a9d3c235e1efd15404c549282d21c99243a21c", "PR32": "cdb640543d8fe3a43c8e63af4ccaaa002e09a599"}
INPUTS, ROWS, CHECKS, MISSING = {}, [], [], []


def read(path, expected=None):
    path = Path(path)
    assert path.suffix in {".json", ".jsonl", ".csv"}, path
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    INPUTS[str(path.resolve())] = {"bytes": len(data), "sha256": digest}
    if expected:
        assert len(data) == expected["bytes"] and digest == expected["sha256"], path
    return data.decode()


def quantile(values, probability):
    values = sorted(values)
    place = (len(values) - 1) * probability
    lo, hi = math.floor(place), math.ceil(place)
    return values[lo] + (values[hi] - values[lo]) * (place - lo)


def emit(pop, metric, value, aggregation, count, source, note="", event="all", pass_index="all"):
    pr, dataset, arm = pop
    ROWS.append(dict(experiment=pr, dataset=dataset, arm=arm, event=event, pass_index=pass_index,
                     metric=metric, value=value, aggregation_level=aggregation, population_count=count,
                     source=source, notes=note))


def distribution(pop, metric, values, level, source, note=""):
    assert values and all(math.isfinite(x) for x in values)
    for label, value in [("mean", statistics.fmean(values)), ("min", min(values)),
                         ("p05", quantile(values, .05)), ("p50", quantile(values, .5)),
                         ("p95", quantile(values, .95)), ("max", max(values))]:
        emit(pop, metric + "_" + label, value, level, len(values), source, note)


def summed(records):
    result = collections.defaultdict(float)
    for r in records:
        for key, value in r.items():
            if isinstance(value, (int, float)):
                result[key] += value
    return dict(result)


def pooled(pop, totals, count, source, level, event="all", pass_index="all"):
    def put(metric, value, note=""):
        emit(pop, metric, value, level, count, source, note, event, pass_index)
    for name in ["available_position_occurrences", "selected_position_occurrences", "available_position_weight",
                 "selected_position_weight", "empty_blocks", "no_error_blocks"]:
        put(name, totals[name], "Sum across original response presentations; overlapping anchors count as occurrences.")
    a, s = totals["available_position_weight"], totals["selected_position_weight"]
    assert a >= s > 0
    put("pooled_selected_weight_fraction", s / a, "Ratio of sums; excludes epsilon; not mean response fraction.")
    put("pooled_available_selected_weight_ratio", a / s, "Ratio of sums; excludes epsilon; not mean D_r/Z_r.")
    put("pooled_D_over_Z", (a + count * EPS) / (s + count * EPS), "Ideal scalar ratio of sums with one epsilon per response; not native BF16/FP32 replay.")
    put("pooled_objective_numerator_denominator", totals["main_loss_numerator"] / totals["loss_denominator"], "Descriptive ratio of summed logs; not the equal-response objective.")
    if pop[0] == "PR23":
        # Historical counts and the pinned gamma=7 recover ideal decay-weighted mass.
        # They do not recover native FP32 exponential/reduction operation order.
        counts = [totals[f"first_error_position_{k}"] for k in range(1, 16)]
        assert all(n >= 0 and n.is_integer() for n in counts)
        error = math.fsum(n * math.exp(-(k - 1) / 7) for k, n in enumerate(counts, 1))
        note = "Derived FP64 sum_k count(first_error_position_k)*exp(-(k-1)/7), gamma=7; not native FP32 weight/reduction reconstruction."
        put("ideal_histogram_first_error_weight", error, note)
        put("ideal_histogram_first_error_fraction_selected_weight", error / s,
            note + " Denominator is the pooled recorded FP32 selected-weight sum; not a mean response fraction.")
    if "kplus_W1" in totals:
        error = totals["kplus_W1"] + totals["kplus_W2"]
        moved = totals["kplus_error_weight_moved"]
        n = totals["kplus_response_count"]
        assert n == count
        for name, value in [("original_first_error_weight", error), ("redistributed_error_weight", moved),
                            ("original_error_fraction_selected_weight", error / s),
                            ("pooled_moved_fraction_error_weight", moved / error),
                            ("pooled_moved_fraction_selected_weight", moved / s),
                            ("mixed_priority_response_frequency", totals["kplus_mixed_response"] / n)]:
            put(name, value, "Scalar weight only; does not identify gradient shares or efficacy.")
        for name in ["kplus_coefficient1", "kplus_coefficient2", "kplus_error_weight_moved_share"]:
            put("mean_logged_" + name, totals[name] / n, "Sum of logged response values divided by actual response count; not formula evaluated at pooled W1/W2.")
    if "role_loss_full_to_selected_ratio" in totals:
        assert totals["role_loss_response_count"] == count
        put("mean_logged_full_to_selected_ratio", totals["role_loss_full_to_selected_ratio"] / count,
            "Recorded native FP32 per-response ratios summed then divided by response count; distinct from pooled D/Z.")


def originals(pop, entries, public_rows, count_per_event):
    """Use original JSONL once; RESULT receipt copies never contribute observations."""
    response_files = sorted([x for x in entries if Path(x["local_candidate"]).name == "RESPONSE_OBSERVATIONS.jsonl"], key=lambda x: x["local_candidate"])
    window_files = {str(Path(x["local_candidate"]).parent): x for x in entries if Path(x["local_candidate"]).name == "WINDOW_OBSERVATIONS.jsonl"}
    records, windows, duplicates, seen = [], [], 0, {}
    pair_ratios = []
    for entry in response_files:
        path = Path(entry["local_candidate"])
        event_rows = []
        for line in read(path, entry["expected"]).splitlines():
            r = json.loads(line)
            assert r["scope"] == "one_original_response_forward"
            key = (r["event_id"], r["pass_index"], r["record_id"])
            original_identity = json.dumps(r, sort_keys=True, separators=(",", ":"))
            if key in seen:
                assert original_identity == seen[key], key
                duplicates += 1
                continue
            seen[key] = original_identity
            event_rows.append(r)
        event_rows.sort(key=lambda x: x["presentation_ordinal"])
        assert len(event_rows) == count_per_event
        assert [r["presentation_ordinal"] for r in event_rows] == list(range(count_per_event))
        assert collections.Counter(r["pass_index"] for r in event_rows) == {0: 128, 1: 128}
        assert all(r["sum_metrics"]["kplus_response_count"] == 1 for r in event_rows)
        for r in event_rows:
            m = r["sum_metrics"]
            a, s = m["available_position_weight"], m["selected_position_weight"]
            assert a >= s >= 0
            r["derived_ratio"] = (a + EPS) / (s + EPS)
        we = window_files[str(path.parent)]
        ew = [json.loads(line) for line in read(we["local_candidate"], we["expected"]).splitlines()]
        assert len(ew) == count_per_event // 2
        for j, w in enumerate(ew):
            pair = event_rows[j * 2:j * 2 + 2]
            assert w["scope"] == "optimizer_window" and w["event_id"] == pair[0]["event_id"]
            assert w["record_ids"] == [r["record_id"] for r in pair]
            assert w["window"] == j + 1
            assert w["mean_response_loss"] == sum(r["loss"] for r in pair) / 2
            factors = [r["derived_ratio"] for r in pair]
            pair_ratios.append(max(factors) / min(factors))
        records.extend(event_rows)
        windows.extend(ew)
    expected = len(public_rows) * 128
    assert len(records) == expected and len(windows) == expected // 2
    assert len({w["global_step"] for w in windows}) == len(windows)
    assert sorted(w["global_step"] for w in windows) == list(range(1, len(windows) + 1))
    assert all(w["learning_rate"] == .0003 for w in windows)
    sources = f"private original JSONLs: {pop[0]}/{pop[1]}/{pop[2]}; exact paths/hashes in metrics provenance"
    emit(pop, "original_response_presentations", len(records), "original_response", len(records), sources,
         "Two presentations of each feedback response retained; repeated presentations are not independent samples.")
    emit(pop, "duplicate_original_rows_excluded", duplicates, "original_response", len(records), sources)
    emit(pop, "optimizer_windows", len(windows), "optimizer_window", len(windows), sources)
    # Check every public pass total against only original response rows, never receipt copies.
    public_checked = 0
    for pub in public_rows:
        event, pass_index = int(pub["event"]), int(pub["pass_index"])
        rr = [r for r in records if r["event_id"] == f"event_{event:03d}" and r["pass_index"] == pass_index]
        totals = summed(r["sum_metrics"] for r in rr)
        expected_totals = json.loads(pub["sum_metrics"]) if "sum_metrics" in pub else pub
        for key, total in totals.items():
            if key in expected_totals:
                assert math.isclose(total, float(expected_totals[key]), rel_tol=1e-12, abs_tol=1e-9), (pop, event, pass_index, key, total, expected_totals[key])
                public_checked += 1
        pooled(pop, totals, len(rr), sources, "pass_total_from_original_responses", event, pass_index)
        emit(pop, "actual_mean_response_loss", statistics.fmean(r["loss"] for r in rr), "pass_response_mean", len(rr), sources, "Equal-response mean of logged native losses.", event, pass_index)
        if "actual_mean_response_loss" in pub:
            assert math.isclose(statistics.fmean(r["loss"] for r in rr), float(pub["actual_mean_response_loss"]), rel_tol=1e-12)
    for eid in sorted({r["event_id"] for r in records}):
        rr = [r for r in records if r["event_id"] == eid]
        event = int(eid.removeprefix("event_"))
        pooled(pop, summed(r["sum_metrics"] for r in rr), len(rr), sources, "event_total_from_original_responses", event)
        emit(pop, "actual_mean_response_loss", statistics.fmean(r["loss"] for r in rr), "event_response_mean", len(rr), sources, "Equal-response mean of logged native losses.", event)
    totals = summed(r["sum_metrics"] for r in records)
    pooled(pop, totals, len(records), sources, "trajectory_pooled_original_responses")
    distribution(pop, "ideal_D_over_Z_from_logged_weights", [r["derived_ratio"] for r in records], "original_response_distribution", sources,
                 "Double-precision arithmetic on logged FP32 sums plus epsilon; not bitwise native execution or causal effect.")
    distribution(pop, "within_window_max_min_D_over_Z", pair_ratios, "optimizer_window_distribution", sources,
                 "Derived FP64 from logged FP32 sums plus epsilon: max(D1/Z1,D2/Z2)/min(D1/Z1,D2/Z2), paired by original order and verified window record IDs; not native bitwise gradient multipliers or gradient conflict.")
    for name in ["kplus_coefficient1", "kplus_coefficient2", "kplus_error_weight_moved_share"]:
        distribution(pop, "logged_" + name, [r["sum_metrics"][name] for r in records], "original_response_distribution", sources)
    for metric, numerator in [("response_original_error_fraction_selected_weight", lambda m: m["kplus_W1"] + m["kplus_W2"]),
                              ("response_moved_fraction_selected_weight", lambda m: m["kplus_error_weight_moved"])]:
        values = [numerator(r["sum_metrics"]) / r["sum_metrics"]["selected_position_weight"] for r in records if r["sum_metrics"]["selected_position_weight"] > 0]
        distribution(pop, metric, values, "original_response_distribution", sources,
                     "Ratio for each positive-selected-weight response; scalar weight only; zero-weight responses counted separately.")
    if "role_loss_full_to_selected_ratio" in totals:
        distribution(pop, "logged_native_full_to_selected_ratio", [r["sum_metrics"]["role_loss_full_to_selected_ratio"] for r in records], "original_response_distribution", sources)
    distribution(pop, "logged_response_loss", [r["loss"] for r in records], "original_response_distribution", sources)
    distribution(pop, "pre_clip_gradient_norm", [w["gradient_norm"] for w in windows], "optimizer_window_distribution", sources,
                 "Logged return from clip_grad_norm_, after equal-response gradient normalization and before clipping.")
    emit(pop, "actual_mean_response_loss", statistics.fmean(r["loss"] for r in records), "original_response_mean", len(records), sources)
    emit(pop, "clipping_frequency_norm_gt_1", sum(w["gradient_norm"] > 1 for w in windows) / len(windows), "optimizer_window", len(windows), sources,
         "Count of logged pre-clip gradient norms strictly greater than configured max_grad_norm=1.")
    emit(pop, "clipping_count_norm_gt_1", sum(w["gradient_norm"] > 1 for w in windows), "optimizer_window", len(windows), sources)
    for metric, condition in [("zero_available_weight_responses", lambda m: m["available_position_weight"] == 0),
                              ("zero_selected_weight_responses", lambda m: m["selected_position_weight"] == 0),
                              ("zero_first_error_weight_responses", lambda m: m["kplus_W1"] + m["kplus_W2"] == 0),
                              ("single_priority_responses", lambda m: (m["kplus_W1"] == 0) != (m["kplus_W2"] == 0))]:
        emit(pop, metric, sum(condition(r["sum_metrics"]) for r in records), "original_response", len(records), sources)
    CHECKS.append({"population": pop, "responses": len(records), "windows": len(windows), "duplicates_excluded": duplicates,
                   "public_scalar_total_checks": public_checked, "pairing": "all exact IDs/order and equal-response window loss", "source_identity": "all original JSONLs match retained manifest hashes"})
    CHECKS[-1]["windows_with_norm_gt_1"] = [w for w in windows if w["gradient_norm"] > 1]


def main():
    started = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", type=Path, required=True)
    ap.add_argument("--private-output", type=Path, required=True)
    args = ap.parse_args()
    output = Path(__file__).resolve().parent
    reports = output.parent
    inv = json.loads(read(args.inventory))
    entries = inv["entries"]
    for manifest in sorted({x["manifest"] for x in entries}):
        # Manifest identities are retained in provenance; only declared original paths are opened.
        read(manifest)
    for pr, folder in REPORTS.items():
        name = "EVENT_EXPOSURE.csv" if pr == "PR23" else "TRAINING_EXPOSURE.csv"
        path = reports / folder / name
        public = list(csv.DictReader(read(path).splitlines()))
        assert len(public) == {"PR23": 116, "PR25": 116, "PR32": 48}[pr]
        if pr == "PR23":
            for arm in ["K", "K_HALF"]:
                rr = [r for r in public if r["arm"] == arm]
                assert len(rr) == 58 and {int(r["event"]) for r in rr} == set(range(1, 59))
                pop = (pr, "gsm8k", arm)
                scalar = [{k: float(v) for k, v in r.items() if k not in {"arm", "event"}} for r in rr]
                for r, s in zip(rr, scalar):
                    assert s["original_response_observations"] == s["response_presentations"] == 256
                    pooled(pop, s, 256, str(path.relative_to(reports)), "public_event_aggregate", int(r["event"]))
                totals = summed(scalar)
                pooled(pop, totals, int(totals["response_presentations"]), str(path.relative_to(reports)), "trajectory_from_public_event_aggregates")
                for metric in ["per_response_D_over_Z_distribution", "within_window_normalization_imbalance", "actual_mean_response_loss", "pre_clip_gradient_norm", "clipping_frequency", "zero_selected_response_count", "native_recorded_first_error_scalar_weight"]:
                    emit(pop, metric, "unavailable at this aggregation level", "unavailable", int(totals["response_presentations"]), str(path.relative_to(reports)), "All 58 local response/window/RESULT candidates absent for this arm; no recovery search.")
                    MISSING.append({"population": pop, "field": metric, "reason": "only public event aggregates locally available"})
        else:
            populations = [("gsm8k", "KPLUS")] if pr == "PR25" else [("gsm8k", "KPLUS"), ("mathqa", "KPLUS")]
            for dataset, arm in populations:
                relevant = [x for x in entries if x["pr"] == pr and x["present"] and (pr == "PR25" or f"/trajectories/{dataset}/{arm}/" in x["local_candidate"])]
                rr = public if pr == "PR25" else [r for r in public if r["dataset"] == dataset and r["arm"] == arm]
                originals((pr, dataset, arm), relevant, rr, 256)
    # The public PR25 training summary is an independent container, not extra observations.
    training_path = reports / REPORTS["PR25"] / "TRAINING.json"
    training = json.loads(read(training_path))
    assert training["actual_work"]["completed"]["response_forwards"] == 14848
    assert training["actual_work"]["completed"]["optimizer_steps"] == 7424
    assert len(training["pass_rows"]) == 116
    args.private_output.mkdir(parents=True, exist_ok=True)
    with (output / "AUDIT_METRICS.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, list(ROWS[0]))
        writer.writeheader()
        writer.writerows(ROWS)
    provenance = dict(analysis="scalar retained-record audit only", command=sys.argv,
                      script={"path": str(Path(__file__).resolve()), "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
                      reference_snapshot="d47ab1ce644c0702ade657e857f7f00af4185fbb", scientific_execution_commits=COMMITS,
                      started_unix=started, elapsed_seconds=time.time() - started, consumed_files=INPUTS,
                      validation_checks=CHECKS, missing_fields=MISSING,
                      unavailable_paths=[x["local_candidate"] for x in entries if not x["present"]],
                      excluded="RESULT.json receipt copies never counted; PR32 MIX/ROLE originals inventoried but not included in K/K+ metric populations; all model/feature/logit tensors excluded",
                      numerical_scope="Descriptive Python scalar summaries of logged values; no FP64 retrofit to native algorithm; quantiles use linear interpolation.",
                      counts_are="response presentations and windows, not independent experimental samples",
                      model_work=0, optimizer_work=0, remote_operations=0)
    (args.private_output / "METRICS_PROVENANCE.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"metric_rows": len(ROWS), "inputs": len(INPUTS), "checks": CHECKS, "elapsed_seconds": provenance["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
