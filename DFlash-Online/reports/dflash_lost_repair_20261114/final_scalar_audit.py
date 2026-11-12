"""Independent stdlib reduction of retained census/exposure and public tables.

This uses no production analysis helper. Authorship overlaps reporting code;
the separate RAW_AUDIT remains the independently authored raw-native check.
"""
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("task_root", type=Path, help="Private task root containing delivery/retained and reviews")
ROOT = parser.parse_args().task_root
C = ROOT / "delivery/retained/campaign_001"
S = json.loads((C / "final_report/SUMMARY.json").read_text())
T = json.loads((C / "training/RESULT.json").read_text())
N = json.loads((C / "census/CENSUS.json").read_text())
CHECKS = 0


def same(actual, expected):
    global CHECKS
    CHECKS += 1
    if isinstance(expected, float):
        assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-10), (actual, expected)
    else:
        assert actual == expected, (actual, expected)


def key(row):
    return row["request_id"], row["canonical_start"]


def digest(row):
    return hashlib.sha256(json.dumps([20261114, *key(row)], separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


labels, counts, H = {}, Counter(), []
for row in N:
    same(row["parity_valid"], True)
    a, b, c = row["a4"], row["a5"], row["a6"]
    genuine = row["genuine_k6_rejection"]
    h = genuine and row["genuine_k4_rejection"] and a == c and b >= c + 1
    m = genuine and row["genuine_k4_rejection"] and row["genuine_k5_rejection"] and a == b == c
    converse = row["genuine_k4_rejection"] and row["genuine_k5_rejection"] and a == b and c >= a + 1
    label = "H" if h else "M" if m else "other_rejection_history" if genuine else "all_correct_or_terminal"
    labels[key(row)] = (label, c + 1 if genuine else None)
    counts[label] += 1
    counts["converse"] += converse
    if h:
        H.append(row)
same(len(labels), len(N))
same(S["history_counts"], {name: counts[name] for name in S["history_counts"]})
same(S["H_observed_count"], len(H))
same(S["M_observed_count"], counts["M"])
same(S["converse_observed_count"], counts["converse"])
same(S["observed_genuine_k6_rejections"], sum(r["genuine_k6_rejection"] for r in N))
same(S["H_contributing_requests"], len({r["request_id"] for r in H}))
same(S["H_lost_prefix_total"], sum(r["a5"] - r["a6"] for r in H))
same(S["H_largest_request_count"], max(Counter(r["request_id"] for r in H).values()))
same(S["H_by_depth"], {str(k): v for k, v in Counter(r["a6"] + 1 for r in H).items()})
same(S["H_lost_prefix_histogram"], {str(k): v for k, v in Counter(r["a5"] - r["a6"] for r in H).items()})
same(S["H_share_all_natural_starts"], len(H) / len(N))
same(S["H_share_genuine_k6_rejections"], len(H) / S["observed_genuine_k6_rejections"])

scheduled, agreed, comparable, matched, groups = [], [], [], [], defaultdict(lambda: [[], []])
observed = {}
for row in T["training_rows"]:
    k = row["presentation_id"], *key(row)
    assert k not in observed
    observed[k] = row
    label, slot = labels.get(key(row), (None, None))
    if label == "H":
        scheduled.append(row)
    if row["j"] is not None and row["j"] == slot and label in ("H", "M"):
        group = row["presentation_id"], row["request_id"], row["j"], row["priority"]
        groups[group][0 if label == "H" else 1].append(row)
        if label == "H":
            agreed.append(row)
pairs = []
for group, (hs, ms) in groups.items():
    if ms:
        comparable.extend(hs)
    available = list(ms)
    for h in sorted(hs, key=digest):
        if not available:
            break
        m = min(available, key=lambda r: (abs(h["wrong_margin"] - r["wrong_margin"]), digest(r)))
        available.remove(m)
        matched.append(h)
        pairs.append(dict(presentation_id=group[0], request_id=group[1], j=group[2], priority=group[3],
                          H_start=h["canonical_start"], M_start=m["canonical_start"],
                          H_margin=h["wrong_margin"], M_margin=m["wrong_margin"],
                          H_weight=h["first_error_weight"], M_weight=m["first_error_weight"],
                          projected_weight=.25 * min(h["first_error_weight"], m["first_error_weight"])))
for name, rows in (("native_replay", H), ("scheduled", scheduled), ("same_j", agreed), ("comparable_M", comparable), ("matched", matched)):
    same(S["exposure_funnel"][name], dict(occurrences=len(rows), distinct_anchors=len({key(r) for r in rows}), distinct_requests=len({r["request_id"] for r in rows})))

pp = T["presentation_rows"]
same(len(pp), 256)
same(Counter(p["request_id"] for p in pp), Counter({r["request_id"]: 2 for r in N}))
mass = sum(p["first_error_weight"] for p in pp)
selected = sum(p["selected_weight"] for p in pp)
h_mass = sum(r["first_error_weight"] for r in agreed)
projected = sum(r["projected_weight"] for r in pairs)
same(S["native_first_error_weight"], mass)
same(S["selected_weight"], selected)
same(S["native_H_weight_before_matching"], h_mass)
same(S["projected_matched_weight"], projected)
same(S["H_weight_fraction_all_first_error"], h_mass / mass)
same(S["H_weight_fraction_all_selected"], h_mass / selected)
same(S["projected_weight_fraction_all_first_error"], projected / mass)
same(S["gate_decision"], "FAIL")
same(S["gate"]["distinct_matched_H_anchors"]["passed"], True)
same(S["gate"]["distinct_matched_H_requests"]["passed"], True)
same(S["gate"]["projected_weight_fraction_all_native_first_error"]["passed"], False)
same(S["margin_balance"]["pairs"], len(pairs))
same(S["margin_balance"]["mean_H_margin"], sum(r["H_margin"] for r in pairs) / len(pairs))
same(S["margin_balance"]["mean_M_margin"], sum(r["M_margin"] for r in pairs) / len(pairs))
same(S["margin_balance"]["mean_absolute_difference"], sum(abs(r["H_margin"] - r["M_margin"]) for r in pairs) / len(pairs))

for p in pp:
    rows = [r for r in T["training_rows"] if r["presentation_id"] == p["presentation_id"]]
    same(len(rows), p["expected_anchor_count"])
    total = sum(r["first_error_weight"] for r in rows)
    same(abs(total - p["first_error_weight"]) <= 2e-7 + 2e-6 * abs(p["first_error_weight"]), True)

def csv_rows(name):
    with (C / "final_report" / name).open() as stream:
        return list(csv.DictReader(stream))

csv_pairs = csv_rows("MATCHED_PAIRS.csv")
same(len(csv_pairs), len(pairs))
index = {(r["presentation_id"], r["request_id"], r["H_start"]): r for r in pairs}
for row in csv_pairs:
    original = index[(int(row["presentation_id"]), row["request_id"], int(row["H_start"]))]
    for field, expected in original.items():
        actual = row[field]
        if isinstance(expected, (float, int)):
            actual = type(expected)(actual)
        same(actual, expected)

csv_weights = csv_rows("NATIVE_WEIGHTS.csv")
same(len(csv_weights), len(observed))
for row in csv_weights:
    original = observed[(int(row["presentation_id"]), row["request_id"], int(row["canonical_start"]))]
    same(float(row["first_error_weight"]), original["first_error_weight"])
    label, slot = labels.get(key(original), ("not_a_natural_start", None))
    same(row["natural_history"], label)
    same(row["same_first_error_j"] == "True", slot is not None and original["j"] == slot)

per_request = csv_rows("PER_REQUEST.csv")
same(len(per_request), 128)
for row in per_request:
    rid = row["request_id"]
    native = [r for r in N if r["request_id"] == rid]
    same(int(row["natural_starts"]), len(native))
    same(int(row["H"]), sum(labels[key(r)][0] == "H" for r in native))
    same(int(row["M"]), sum(labels[key(r)][0] == "M" for r in native))
    same(float(row["native_H_weight_before_matching"]), sum(r["first_error_weight"] for r in agreed if r["request_id"] == rid))

receipt = dict(status="PASSED", scalar_and_table_checks=CHECKS,
               scope="Independent stdlib census/exposure/matching/table reduction; reporting authorship overlap; no model rerun.",
               natural_starts=len(N), H=len(H), matched_anchors=len({key(r) for r in matched}),
               matched_occurrences=len(pairs), native_first_error_mass=mass, projected_mass=projected,
               projected_fraction=projected / mass, gate="FAIL")
(ROOT / "reviews/FINAL_SCALAR_AUDIT.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt, indent=2))
