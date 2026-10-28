"""Summarize retained checkpoint diagnostics; never load or execute a model."""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


ARMS = ("F", "C", "P")
PARTITIONS = ("all", "reference_correct", "reference_wrong")
METRICS = ("soft_ce", "target_entropy", "kl_to_r",
           "target_token_logit_margin_vs_best_other", "projected_probability_margin")
COUNTS = ("exposure", "correct", "repairs", "damage", "reference_correct", "reference_wrong",
          "unchanged_decisions", "changed_decisions", "remaining_wrong_changed_decisions")


def rows(path):
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            yield {key: (None if value == "" else value) for key, value in row.items()}


def flag(value):
    if value not in ("True", "False"):
        raise ValueError(f"Expected boolean, got {value!r}")
    return value == "True"


def layout_key(row):
    return row["group"], row["record_id"], int(row["pass_index"])


def mean(values):
    values = list(values)
    return math.fsum(values) / len(values) if values else None


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def win_counts(differences, higher_better=True):
    differences = list(differences)
    direction = 1 if higher_better else -1
    return {"improved": sum(direction * x > 0 for x in differences),
            "worse": sum(direction * x < 0 for x in differences),
            "tie": sum(x == 0 for x in differences), "denominator": len(differences)}


def summarize(root):
    layouts = {}
    for row in rows(root / "layouts.csv"):
        key = (*layout_key(row), row["arm"], row["partition"])
        assert key not in layouts, f"Duplicate layout partition: {key}"
        layouts[key] = row
    identities = sorted({key[:3] for key in layouts})
    assert len(identities) == 32
    assert {key[0] for key in identities} == {"gsm8k", "codealpaca"}
    responses = sorted({key[:2] for key in identities})
    assert len(responses) == 16
    for group in ("gsm8k", "codealpaca"):
        assert sum(key[0] == group for key in responses) == 8
    assert set(identities) == {(*key, p) for key in responses for p in (0, 1)}
    assert set(layouts) == {(*key, arm, part) for key in identities for arm in ARMS for part in PARTITIONS}

    prefixes = {}
    for row in rows(root / "prefixes.csv"):
        key = (*layout_key(row), row["arm"], int(row["block"]))
        assert key not in prefixes, f"Duplicate block: {key}"
        assert not flag(row["eligibility_gap"])
        prefixes[key] = row
    assert set(prefixes) == {(*key, arm, block) for key in identities for arm in ARMS for block in range(64)}

    reference_positions, seen = {}, set()
    counts = defaultdict(Counter)
    position_sums = defaultdict(lambda: defaultdict(list))
    max_ce_entropy_kl_residual = 0.0
    for row in rows(root / "positions.csv"):
        base = (*layout_key(row), int(row["block"]), int(row["slot"]))
        arm, pred, label = row["arm"], int(row["predicted_id"]), int(row["label"])
        assert arm in ARMS and 1 <= base[-1] <= 15
        assert (base, arm) not in seen, f"Duplicate position: {base}, {arm}"
        seen.add((base, arm))
        weight = float(row["weight"])
        assert weight > 0
        correct, initial_correct = flag(row["correct"]), flag(row["reference_correct"])
        assert correct == (pred == label)
        if arm == "F":
            assert initial_correct == correct
            reference_positions[base] = (pred, label, weight, initial_correct)
        initial_pred, initial_label, initial_weight, initial_membership = reference_positions[base]
        assert (label, weight, initial_correct) == (initial_label, initial_weight, initial_membership)
        repaired, damaged = not initial_correct and correct, initial_correct and not correct
        assert flag(row["repaired"]) == repaired and flag(row["damaged"]) == damaged
        key = (*layout_key(row), arm)
        counts[key].update(exposure=1, correct=int(correct), repairs=int(repaired), damage=int(damaged),
            reference_correct=int(initial_correct), reference_wrong=int(not initial_correct),
            unchanged_decisions=int(pred == initial_pred), changed_decisions=int(pred != initial_pred),
            remaining_wrong_changed_decisions=int(not initial_correct and not correct and pred != initial_pred))
        for metric in METRICS:
            value = float(row[metric])
            assert math.isfinite(value), (base, arm, metric)
            position_sums[key][metric].append(weight * value)
        residual = abs(float(row["soft_ce"]) - float(row["target_entropy"]) - float(row["kl_to_r"]))
        max_ce_entropy_kl_residual = max(max_ce_entropy_kl_residual, residual)
    assert len(seen) == 3 * len(reference_positions)
    for identity in identities:
        fdenom = float(layouts[(*identity, "F", "all")]["original_denominator"])
        for arm in ARMS:
            key = (*identity, arm)
            total = layouts[(*key, "all")]
            assert float(total["original_denominator"]) == fdenom
            for field in ("exposure", "correct", "repairs", "damage"):
                assert int(total[field]) == counts[key][field], (key, field)
            for partition in PARTITIONS[1:]:
                assert int(layouts[(*key, partition)]["exposure"]) == counts[key][partition]
            assert sum(int(prefixes[(*key, block)]["eligible_count"]) for block in range(64)) == counts[key]["exposure"]
            for metric in METRICS:
                assert math.isclose(math.fsum(position_sums[key][metric]), float(total[metric + "_sum"]),
                    rel_tol=1e-5, abs_tol=2e-4), (key, metric)

    by_response = {}
    for identity in responses:
        for arm in ARMS:
            ls = [layouts[(*identity, p, arm, "all")] for p in (0, 1)]
            ps = [prefixes[(*identity, p, arm, b)] for p in (0, 1) for b in range(64)]
            by_response[(*identity, arm)] = {
                **dict(group=identity[0], record_id=identity[1], arm=arm),
                **{metric: mean(float(row[metric + "_contribution"]) for row in ls) for metric in METRICS},
                "accuracy": mean(int(row["correct"]) / int(row["exposure"]) for row in ls),
                "matching_prefix_mean": mean(int(row["matching_prefix_length"]) for row in ps)}

    aggregates, pairs, comparison = [], [], []
    for group in ("all", "gsm8k", "codealpaca"):
        ids = [key for key in identities if group == "all" or key[0] == group]
        rs = [key for key in responses if group == "all" or key[0] == group]
        for arm in ARMS:
            ls = [layouts[(*key, arm, "all")] for key in ids]
            ps = [prefixes[(*key, arm, b)] for key in ids for b in range(64)]
            summed = sum((counts[(*key, arm)] for key in ids), Counter())
            partitions = {}
            for partition in PARTITIONS:
                part = [layouts[(*key, arm, partition)] for key in ids]
                weight = math.fsum(float(row["weight"]) for row in part)
                partitions[partition] = {
                    "weight": weight, "exposure": sum(int(row["exposure"]) for row in part),
                    "pooled_weighted_numerators": {metric: math.fsum(float(row[metric + "_sum"]) for row in part) for metric in METRICS},
                    "pooled_weighted_means": {metric: ratio(math.fsum(float(row[metric + "_sum"]) for row in part), weight) for metric in METRICS},
                    "original_objective_equal_layout_contributions": {metric: mean(float(row[metric + "_contribution"]) for row in part) for metric in METRICS}}
            item = dict(group=group, arm=arm, responses=len(rs), layouts=len(ids), blocks=len(ps),
                objective_equal_layout_means={metric: mean(float(row[metric + "_contribution"]) for row in ls) for metric in METRICS},
                accuracy_equal_layout_mean=mean(int(row["correct"]) / int(row["exposure"]) for row in ls),
                counts={field: summed[field] for field in COUNTS}, repair_rate=ratio(summed["repairs"], summed["reference_wrong"]),
                damage_rate=ratio(summed["damage"], summed["reference_correct"]),
                matching_prefix_mean=mean(int(row["matching_prefix_length"]) for row in ps),
                matching_non_eos_prefix_mean=mean(int(row["matching_non_eos_prefix_length"]) for row in ps),
                matched_eos_blocks=sum(flag(row["matched_eos"]) for row in ps),
                cap_boundary_censored_blocks=sum(flag(row["cap_boundary_censored"]) for row in ps),
                partitions=partitions)
            aggregates.append(item)
            comparison.append({"group": group, "arm": arm, "responses": len(rs), "layouts": len(ids),
                **item["objective_equal_layout_means"], "accuracy_equal_layout_mean": item["accuracy_equal_layout_mean"],
                "exposure": summed["exposure"], "repairs": summed["repairs"], "repair_denominator": summed["reference_wrong"],
                "damage": summed["damage"], "damage_denominator": summed["reference_correct"],
                "unchanged_decisions": summed["unchanged_decisions"], "matching_prefix_mean": item["matching_prefix_mean"],
                "cap_boundary_censored_blocks": item["cap_boundary_censored_blocks"]})
        for arm, baseline in (("C", "F"), ("P", "F"), ("P", "C")):
            block_deltas = [int(prefixes[(*key, arm, b)]["matching_prefix_length"]) -
                            int(prefixes[(*key, baseline, b)]["matching_prefix_length"]) for key in ids for b in range(64)]
            pairs.append(dict(group=group, arm=arm, baseline=baseline,
                block_prefix=win_counts(block_deltas), block_prefix_delta_mean=mean(block_deltas),
                response_wins={metric: win_counts(
                    (by_response[(*key, arm)][metric] - by_response[(*key, baseline)][metric] for key in rs),
                    higher_better=metric != "kl_to_r") for metric in ("kl_to_r", "matching_prefix_mean", "accuracy")}))
    result = dict(schema="fixed_checkpoint_fit_summary_v1", validation="complete CSV identities/counts/reductions cross-checked",
        definitions={"objective": "Equal mean of original weighted response-layout means; both passes retained.",
            "partition_means": "Pooled weighted numerator / pooled subset weight; distinct from original objective.",
            "partition_contributions": "Equal layout mean of subset numerator / original layout denominator.",
            "counts": "Repeated sampled token exposures, not independent examples.",
            "response_wins": "Each response averages its two layouts; exact sign of the paired difference, no inference.",
            "block_prefix": "Includes matched EOS. Saved cap-censor flags retained. No native generation claim."},
        source_sha256={name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in ("layouts.csv", "prefixes.csv", "positions.csv")},
        source_counts=dict(responses=16, layouts=32, checkpoint_layouts=96, block_evaluations=len(prefixes),
            positions=len(seen), reference_position_exposures=len(reference_positions)),
        max_position_ce_minus_entropy_minus_kl_residual=max_ce_entropy_kl_residual,
        aggregates=aggregates, paired_comparisons=pairs, responses=list(by_response.values()))
    return result, comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    summary, comparison = summarize(args.results)
    (args.results / "SUMMARY.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    with (args.results / "COMPARISON.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    print(json.dumps(summary["source_counts"]))


if __name__ == "__main__":
    main()
