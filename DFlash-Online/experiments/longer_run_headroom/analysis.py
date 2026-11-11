"""Request-cluster estimates for the fixed, parent-selected diagnostic sample."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import time

import numpy as np

SEED = 20261112
DRAWS = 10000
FIELDS = (
    "still_blocked",
    "blocked_with_both",
    "repair",
    "repair_with_two_accepted",
    "earlier_prefix_break",
    "native_prefix_change",
    "prefix_gained",
    "prefix_lost",
    "prefix_tie",
)


def ratio(a, b):
    a, b = np.broadcast_arrays(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    return np.divide(a, b, out=np.full(a.shape, np.nan), where=b != 0)


def interval(values):
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    return dict(
        lower=float(np.quantile(finite, 0.025)) if len(finite) else None,
        upper=float(np.quantile(finite, 0.975)) if len(finite) else None,
        defined_draws=len(finite),
        undefined_draws=len(values) - len(finite),
        degenerate=bool(len(finite) and finite.min() == finite.max()),
    )


def cluster_total(rows, ids, value):
    lookup = {key: i for i, key in enumerate(ids)}
    result = np.zeros(len(ids))
    for row in rows:
        result[lookup[row["request_id"]]] += value(row)
    return result


def group_summary(rows, ids, draws):
    count = cluster_total(rows, ids, lambda r: 1)
    summary = dict(
        starts=len(rows), distinct_requests=len({r["request_id"] for r in rows})
    )
    for field in FIELDS:
        values = cluster_total(rows, ids, lambda r: r[field])
        n = int(values.sum())
        summary[field] = dict(
            total=n,
            per_start=n / len(rows) if rows else None,
            contributing_requests=int(np.count_nonzero(values)),
            interval=interval(ratio(draws @ values, draws @ count)),
        )
    summary["gain_starts"] = sum(r["native_prefix_change"] > 0 for r in rows)
    summary["loss_starts"] = sum(r["native_prefix_change"] < 0 for r in rows)
    return summary


def standardized(census, paired, ids, draws):
    """Fixed observed common cells; missing resampled cells are undefined."""
    parent = Counter((r["group"], r["j"], r["v"]) for r in census)
    observed = Counter((r["group"], r["j"], r["v"]) for r in paired)
    census_common = sorted(
        {(j, v) for g, j, v in parent if g == "SHORT" and parent["LONG", j, v]}
    )
    common = [
        c for c in census_common if observed["SHORT", *c] and observed["LONG", *c]
    ]
    rows = []
    weights = []
    boot_weights = []
    matrices = {g: {k: [] for k in FIELDS} for g in ("SHORT", "LONG")}
    for j, v in common:
        cell = dict(
            j=j, v=v, weight=min(parent["SHORT", j, v], parent["LONG", j, v]), groups={}
        )
        weights.append(cell["weight"])
        count_by_group = {}
        for group in ("SHORT", "LONG"):
            own = [r for r in paired if (r["group"], r["j"], r["v"]) == (group, j, v)]
            prior = [r for r in census if (r["group"], r["j"], r["v"]) == (group, j, v)]
            count_by_group[group] = draws @ cluster_total(prior, ids, lambda r: 1)
            denominator = draws @ cluster_total(own, ids, lambda r: 1)
            cell["groups"][group] = dict(
                census=len(prior),
                starts=len(own),
                requests=len({r["request_id"] for r in own}),
                means={key: sum(r[key] for r in own) / len(own) for key in FIELDS},
            )
            for field in FIELDS:
                matrices[group][field].append(
                    ratio(
                        draws @ cluster_total(own, ids, lambda r: r[field]), denominator
                    )
                )
        boot_weights.append(np.minimum(count_by_group["SHORT"], count_by_group["LONG"]))
        rows.append(cell)
    result = dict(
        census_common_cells=len(census_common),
        observed_common_cells=len(common),
        cells=rows,
        support={
            g: dict(
                total_census=sum(n for (k, j, v), n in parent.items() if k == g),
                census_in_common=sum(parent[g, *c] for c in census_common),
                census_in_observed_common=sum(parent[g, *c] for c in common),
                valid_sample=sum(r["group"] == g for r in paired),
                sample_in_observed_common=sum(observed[g, *c] for c in common),
            )
            for g in ("SHORT", "LONG")
        },
        estimates={},
        interval_scope="Conditional on all fixed observed cells being represented in the draw. High undefined fractions and sparse cells preclude a demonstrated standardized difference.",
    )
    if not common:
        return result
    w = np.asarray(weights)
    bw = np.asarray(boot_weights).T
    for field in FIELDS:
        metric = {}
        boot = {}
        for group in ("SHORT", "LONG"):
            metric[group] = float(
                np.dot(w, [r["groups"][group]["means"][field] for r in rows]) / w.sum()
            )
            # Any missing sampled cell invalidates that draw; do not change support.
            means = np.asarray(matrices[group][field]).T
            boot[group] = ratio(np.sum(bw * means, axis=1), bw.sum(axis=1))
        metric["LONG_minus_SHORT"] = metric["LONG"] - metric["SHORT"]
        metric["intervals"] = {g: interval(boot[g]) for g in boot}
        metric["intervals"]["LONG_minus_SHORT"] = interval(boot["LONG"] - boot["SHORT"])
        result["estimates"][field] = metric
    return result


def summarize(census, paired, *, draws=DRAWS, seed=SEED):
    result = {}
    for dataset in sorted({r["dataset"] for r in census}):
        all_parent = [r for r in census if r["dataset"] == dataset]
        parent = [r for r in all_parent if not r["opening"]]
        ids = sorted({r["request_id"] for r in all_parent})
        selected = [r for r in paired if r["dataset"] == dataset]
        valid = [r for r in selected if r["status"] == "valid"]
        rng = np.random.Generator(np.random.PCG64(seed))
        weights = rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=draws)
        longs = [r for r in valid if r["group"] == "LONG"]
        census_longs = [r for r in parent if r["group"] == "LONG"]
        n = cluster_total(longs, ids, lambda r: 1)
        blocked = cluster_total(longs, ids, lambda r: r["blocked_with_both"])
        repair = cluster_total(longs, ids, lambda r: r["repair"])
        realized = cluster_total(longs, ids, lambda r: r["repair_with_two_accepted"])
        cl = cluster_total(census_longs, ids, lambda r: 1)
        cp = cluster_total(parent, ids, lambda r: 1)
        a = (
            len(census_longs) / len(parent) * blocked.sum() / len(longs)
            if longs
            else None
        )
        b = realized.sum() / repair.sum() if repair.sum() else None
        aboot = ratio(weights @ cl, weights @ cp) * ratio(
            weights @ blocked, weights @ n
        )
        bboot = ratio(weights @ realized, weights @ repair)
        result[dataset] = dict(
            requests=len(ids),
            parent_rounds=len(all_parent),
            later_rounds=len(parent),
            census_long=len(census_longs),
            census_long_share=len(census_longs) / len(parent),
            validity=dict(Counter(r["status"] for r in selected)),
            groups={
                g: group_summary([r for r in valid if r["group"] == g], ids, weights)
                for g in ("SHORT", "LONG")
            },
            A=dict(
                estimate=float(a) if a is not None else None,
                numerator=int(blocked.sum()),
                denominator=len(longs),
                contributing_requests=int(np.count_nonzero(blocked)),
                threshold=0.01,
                interval=interval(aboot),
            ),
            B=dict(
                estimate=float(b) if b is not None else None,
                numerator=int(realized.sum()),
                denominator=int(repair.sum()),
                repair_requests=int(np.count_nonzero(repair)),
                contributing_requests=int(np.count_nonzero(realized)),
                threshold=0.5,
                interval=interval(bboot),
            ),
            unconditional_realized=dict(
                numerator=int(realized.sum()),
                denominator=len(longs),
                fraction=float(realized.sum() / len(longs)) if longs else None,
                interval=interval(ratio(weights @ realized, weights @ n)),
            ),
            standardized=standardized(parent, valid, ids, weights),
        )
    return dict(
        seed=seed,
        draws=draws,
        resampling="PCG64 multinomial request-cluster weights; independent reset per dataset",
        datasets=result,
    )


def save_csv(path, rows):
    keys = sorted({k for r in rows for k in r})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--census", type=Path, required=True)
    p.add_argument("--rows", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    started = time.perf_counter()
    census = json.loads(a.census.read_text())
    paired = json.loads(a.rows.read_text())
    report = summarize(census, paired)
    report["cpu_wall_s"] = time.perf_counter() - started
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / "SUMMARY.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    save_csv(a.output / "PER_START.csv", paired)
    per_request = []
    for dataset in sorted(report["datasets"]):
        for rid in sorted({r["request_id"] for r in census if r["dataset"] == dataset}):
            for group in ("SHORT", "LONG"):
                own = [
                    r
                    for r in paired
                    if (r["dataset"], r["request_id"], r["group"])
                    == (dataset, rid, group)
                ]
                valid = [r for r in own if r["status"] == "valid"]
                per_request.append(
                    dict(
                        dataset=dataset,
                        request_id=rid,
                        group=group,
                        selected=len(own),
                        valid=len(valid),
                        **{k: sum(r[k] for r in valid) for k in FIELDS},
                    )
                )
    save_csv(a.output / "PER_REQUEST.csv", per_request)


if __name__ == "__main__":
    main()
