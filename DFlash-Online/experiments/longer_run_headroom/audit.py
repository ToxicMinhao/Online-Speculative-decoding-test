"""Separate direct reduction of raw parent/child blocks, without production helpers."""

from pathlib import Path
import argparse, collections, csv, hashlib, json, sys, time
import numpy as np

start = time.perf_counter()
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=Path, required=True)
parser.add_argument("--evidence", type=Path, required=True)
args = parser.parse_args()
task = args.task
source = args.evidence
load = lambda p: json.loads(Path(p).read_text())
plan = load(task / "preparation/bound/PLAN.json")
selected = load(task / "preparation/census/SELECTED_STARTS.json")["starts"]
actual = load(task / "delivery/retained/ROWS.json")
summary = load(task / "analysis/results/SUMMARY.json")["datasets"]
raw = []
census = []
assertions = 0
for item in plan["original_requests"]:
    d = item["dataset"]
    rank = item["selection_rank"]
    rid = item["request_id"]
    p = load(source / f"campaign/raw/{d}/{rank:03d}/KPLUS4_native.json")
    canonical = item["canonical"]["response_ids"]
    stops = set(p["stop_token_ids"])
    for ordinal, block in enumerate(p["metadata"]["trace"], 1):
        proposed = block["draft_tokens"]
        offset = block["output_start"] + 1
        labels = canonical[offset : offset + 15]
        for i, t in enumerate(labels):
            if t in stops:
                labels = labels[: i + 1]
                break
        correctness = [x == y for x, y in zip(proposed, labels)]
        first = next((i for i, x in enumerate(correctness) if not x), None)
        nrun = 0
        if first is not None:
            for x in correctness[first + 1 :]:
                if not x:
                    break
                nrun += 1
        followers = 0 if first is None else len(labels) - first - 1
        group = (
            "all_correct"
            if first is None
            else (
                "insufficient_lookahead"
                if followers < 2
                else "R0" if nrun == 0 else "SHORT" if nrun == 1 else "LONG"
            )
        )
        census.append(dict(dataset=d, request_id=rid, ordinal=ordinal, group=group))
    own = [x for x in selected if x["dataset"] == d and x["request_id"] == rid]
    for choice in own:
        ordinal = choice["parent_round_ordinal"]
        pb = p["metadata"]["trace"][ordinal - 1]
        ar = next(
            x
            for x in actual
            if x["dataset"] == d
            and x["request_id"] == rid
            and x["parent_round_ordinal"] == ordinal
        )
        if ar["status"] != "valid":
            continue
        if ar["provenance"] == "retained_exact_prior_native_walk":
            child = load(source / f"campaign/raw/{d}/{rank:03d}/KPLUS6_native.json")
            assert (
                p["metadata"]["trace"][: ordinal - 1]
                == child["metadata"]["trace"][: ordinal - 1]
            )
            cb = child["metadata"]["trace"][ordinal - 1]
        else:
            folder = task / f"delivery/retained/replays/{d}/{rank:03d}"
            neutral = load(folder / f"{ordinal:04d}_NEUTRAL.json")
            cb = load(folder / f"{ordinal:04d}_CHILD.json")["blocks"][0]
            for k in [
                "draft_tokens",
                "accepted_draft_tokens",
                "committed_tokens",
                "positions",
                "target_argmax",
                "anchor_token",
                "output_start",
                "finished",
                "stop_clipped",
            ]:
                assert neutral["blocks"][0][k] == pb[k], (d, rid, ordinal, k)
                assertions += 1
        j = choice["j"] - 1
        labels = canonical[
            pb["output_start"] + 1 : pb["output_start"] + 1 + choice["valid_slots"]
        ]
        matches = [a == b for a, b in zip(cb["draft_tokens"], labels)]
        intact = all(matches[:j])
        repair = (
            intact
            and matches[j]
            and cb["accepted_draft_tokens"] > pb["accepted_draft_tokens"]
        )
        both = all(matches[j + 1 : j + 3])
        delta = cb["accepted_draft_tokens"] - pb["accepted_draft_tokens"]
        computed = dict(
            still_blocked=intact and not matches[j],
            blocked_with_both=intact and not matches[j] and both,
            repair=repair,
            repair_with_two_accepted=repair
            and both
            and cb["accepted_draft_tokens"] >= j + 3,
            earlier_prefix_break=not intact,
            native_prefix_change=delta,
            prefix_gained=max(delta, 0),
            prefix_lost=max(-delta, 0),
            prefix_tie=delta == 0,
        )
        for k, v in computed.items():
            assert ar[k] == v, (d, rid, ordinal, k, v, ar[k])
            assertions += 1
        raw.append(dict(dataset=d, request_id=rid, group=choice["group"], **computed))
results = {}
for d, s in summary.items():
    rows = [r for r in raw if r["dataset"] == d]
    parents = [x for x in census if x["dataset"] == d]
    ids = sorted(set(x["request_id"] for x in parents))
    later = [x for x in parents if x["ordinal"] > 1]
    lc = sum(x["group"] == "LONG" for x in later)
    census_report = load(task / "preparation/census/CENSUS.json")[d]
    assert (
        dict(collections.Counter(x["group"] for x in later))
        == census_report["later_counts"]
    )
    assert (
        dict(collections.Counter(x["group"] for x in parents if x["ordinal"] == 1))
        == census_report["opening_counts"]
    )
    for group in ["SHORT", "LONG"]:
        eligible = [x for x in later if x["group"] == group]

        def order(x):
            key = ["kplus-run-probe-v1", d, x["request_id"], x["ordinal"]]
            return hashlib.sha256(
                json.dumps(key, separators=(",", ":"), ensure_ascii=False).encode()
            ).hexdigest()

        chosen = sorted(eligible, key=order)[:64]
        expected = {(x["request_id"], x["ordinal"]) for x in chosen}
        observed = {
            (x["request_id"], x["parent_round_ordinal"])
            for x in selected
            if x["dataset"] == d and x["group"] == group
        }
        assert expected == observed
        assertions += 1
    assertions += 2
    assert len(later) == s["later_rounds"] and lc == s["census_long"]
    assertions += 2
    groups = {}
    for g in ["SHORT", "LONG"]:
        own = [r for r in rows if r["group"] == g]
        reference = s["groups"][g]
        totals = {k: sum(r[k] for r in own) for k in computed}
        for k, n in totals.items():
            assert n == reference[k]["total"], (d, g, k)
            assertions += 1
        assert (
            totals["native_prefix_change"]
            == totals["prefix_gained"] - totals["prefix_lost"]
        )
        assertions += 1
        groups[g] = dict(n=len(own), **totals)
    long = groups["LONG"]
    a = lc / len(later) * long["blocked_with_both"] / long["n"]
    b = long["repair_with_two_accepted"] / long["repair"] if long["repair"] else None
    assert a == s["A"]["estimate"] and b == s["B"]["estimate"]
    assertions += 2
    # Independently loop over request clusters, with the declared random draw weights.
    boot = np.random.Generator(np.random.PCG64(20261112)).multinomial(
        len(ids), np.full(len(ids), 1 / len(ids)), size=10000
    )
    matrix = []
    for rid in ids:
        cp = [r for r in later if r["request_id"] == rid]
        cr = [r for r in rows if r["request_id"] == rid and r["group"] == "LONG"]
        matrix.append(
            [
                len(cp),
                sum(r["group"] == "LONG" for r in cp),
                len(cr),
                sum(r["blocked_with_both"] for r in cr),
                sum(r["repair"] for r in cr),
                sum(r["repair_with_two_accepted"] for r in cr),
            ]
        )
    totals = boot @ np.asarray(matrix)
    aa = []
    bb = []
    for pp, ll, nn, blocked, rep, real in totals:
        aa.append(ll / pp * blocked / nn if nn and pp else np.nan)
        bb.append(real / rep if rep else np.nan)
    for name, vals in [("A", aa), ("B", bb)]:
        finite = np.asarray(vals)[np.isfinite(vals)]
        ref = s[name]["interval"]
        assert len(vals) - len(finite) == ref["undefined_draws"]
        assertions += 1
        if len(finite):
            assert np.allclose(
                np.quantile(finite, [0.025, 0.975]),
                [ref["lower"], ref["upper"]],
                rtol=0,
                atol=1e-15,
            )
            assertions += 2
    results[d] = dict(groups=groups, A=a, B=b)
out = dict(
    status="passed",
    assertions=assertions,
    validated_pairs=len(raw),
    counts=results,
    cpu_wall_s=time.perf_counter() - start,
    provenance="Separately coded arithmetic and raw-trace audit by the same execution author; no independent-agent review available due to platform thread limit. Zero model work.",
)
(task / "analysis/ARITHMETIC_AUDIT.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
