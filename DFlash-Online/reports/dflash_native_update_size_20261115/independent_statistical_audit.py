"""Independent CPU reduction of retained PR48 observations; no production imports."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np

TASK = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
ROOT = TASK.parent
RAW = TASK / "delivery/execution_003/raw"
OLD = ROOT / "dflash_native_prefix_loss_20261114/delivery/execution_002_stream/retained"
PARENTS = ROOT / "dflash_recent_repair_retention_event7_20261112/delivery/final_001/retained/fixed_starts/requests"
STATES = ["P", "N", "E", "S_low", "S_high"]
WINDOWS = list(range(8, 129, 8))
CONTRASTS = [("E", "N"), ("E", "S_low"), ("E", "S_high"),
             ("S_low", "N"), ("S_high", "N"), ("S_high", "S_low")]
checks = 0


def require(condition, description):
    global checks
    checks += 1
    if not condition:
        raise AssertionError(description)


def load(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run():
    began = time.monotonic()
    rows = load(RAW / "probes/PROBE_OUTCOMES.json")
    sealed = load(OLD / "sealed/PROBE_KEYS.json")
    report = load(RAW / "final_report/SUMMARY.json")
    requests = sorted({row["request_id"] for row in sealed})
    keys = sorted((row["request_id"], row["canonical_start"], row["round_ordinal"]) for row in sealed)
    require(len(keys) == len(set(keys)) == 256, "sealed start identity")
    require(len(requests) == 64 and set(Counter(k[0] for k in keys).values()) == {4}, "balanced requests")
    expected = {(w, *k, state) for w in WINDOWS for k in keys for state in STATES}
    lookup = {(row["window"], row["request_id"], row["canonical_start"], row["round_ordinal"], row["state"]): row for row in rows}
    require(len(rows) == len(lookup) == 20480 and set(lookup) == expected, "complete five-state grid")
    require(all(row["valid"] is True for row in rows), "all observations valid")
    parents = {rid: load(PARENTS / rid / "PARENT_K6.json") for rid in requests}
    raw_counts = Counter()
    for row in rows:
        historical = row["state"] in STATES[:3]
        require(row["measurement_origin"] == ("historical_PR47" if historical else "new_scalar_control"), "origin")
        path = (OLD if historical else RAW) / "probes" / row["execution_id"]
        record = load(path)
        parent = parents[row["request_id"]]
        require(record["status"] == "completed" and len(record["blocks"]) == 1, "native completed block")
        block = record["blocks"][0]
        start = row["canonical_start"] + 1
        original = parent["metadata"]["trace"][row["round_ordinal"]]
        require(all(block[field] == original[field] for field in ["output_start", "anchor_token", "positions"]), "exact conditioned start")
        require(block["output_start"] + 1 == start, "absolute position")
        labels = parent["response_ids"][start:start + len(block["draft_tokens"])]
        stops = set(parent["stop_token_ids"])
        if block["anchor_token"] in stops:
            labels = []
        else:
            for index, token in enumerate(labels):
                if token in stops:
                    labels = labels[:index + 1]
                    break
        prefix = 0
        for proposed, canonical in zip(block["draft_tokens"], labels):
            if proposed != canonical:
                break
            prefix += 1
        require(prefix == row["accepted_prefix"] == block["accepted_draft_tokens"], "independent canonical prefix")
        require(len(labels) == row["valid_proposals"], "EOS/cap scoring boundary")
        require(row["first_error"] == (prefix + 1 if prefix < len(labels) else None), "first mismatch")
        require(row["matched_eos"] == any(token in stops for token in labels[:prefix]), "EOS inclusion")
        censor = prefix == len(labels) and parent["finish_reason"] == "length" and start + len(labels) == len(parent["response_ids"])
        require(row["censored_at_cap"] == censor, "cap censoring")
        end = start + block["committed_progress"]
        require(block["committed_tokens"] == parent["response_ids"][start:end], "canonical commit")
        require(record["response_ids"] == parent["response_ids"][:end], "exact response prefix")
        require(record["native_finished"] == block["finished"] == (end == len(parent["response_ids"])), "termination")
        require(record["output_matches_canonical_prefix"] and record["cap_unchanged"] and record["original_cap"] == parent["max_new_tokens"], "native contract")
        require(record["reconstruction"]["status"] == "passed", "reconstruction receipt")
        raw_counts["historical" if historical else "new"] += 1
        raw_counts["matched_eos"] += int(row["matched_eos"])
        raw_counts["censored_at_cap"] += int(censor)
    values = np.empty((64, 16, 4, 5), dtype=np.int64)
    gross = Counter()
    pair_rows = {}
    valid_total = 0
    opportunity_blocker = opportunity_prefix = opportunity_native_repair = 0
    for ir, rid in enumerate(requests):
        request_keys = [key for key in keys if key[0] == rid]
        for iw, window in enumerate(WINDOWS):
            for ik, key in enumerate(request_keys):
                observations = {state: lookup[(window, *key, state)] for state in STATES}
                valid = observations["P"]["valid_proposals"]
                require(all(o["valid_proposals"] == valid for o in observations.values()), "matched scoring boundaries")
                a = {state: observations[state]["accepted_prefix"] for state in STATES}
                values[ir, iw, ik] = [a[state] for state in STATES]
                before = a["P"]
                valid_total += valid
                opportunity_blocker += before < valid
                opportunity_prefix += before > 0
                native_repair = before < valid and a["N"] > before
                opportunity_native_repair += native_repair
                pair = {f"a_{state}": a[state] for state in STATES}
                pair.update(pre_prefix=before, valid_proposals=valid,
                            pre_has_blocker=int(before < valid), pre_has_prefix=int(before > 0))
                for state in STATES[1:]:
                    delta = a[state] - before
                    data = {"acquired": max(delta, 0), "lost": max(-delta, 0), "net": delta,
                            "earlier_break": int(delta < 0), "repairs_pre_blocker": int(before < valid and delta > 0)}
                    for metric, number in data.items():
                        name = f"{state}_{metric}"
                        pair[name] = number
                        gross[name] += number
                for state in ["E", "S_low", "S_high"]:
                    name = f"{state}_displaced_native_repair"
                    pair[name] = int(native_repair and a[state] <= before)
                    gross[name] += pair[name]
                for a_state, b_state in CONTRASTS:
                    name = f"{a_state}_{b_state}"
                    pair[name + "_acquisition_difference"] = pair[a_state + "_acquired"] - pair[b_state + "_acquired"]
                    pair[name + "_loss_difference"] = pair[a_state + "_lost"] - pair[b_state + "_lost"]
                    pair[name + "_net"] = a[a_state] - a[b_state]
                    require(pair[name + "_net"] == pair[name + "_acquisition_difference"] - pair[name + "_loss_difference"], "integer decomposition")
                    for metric in ["acquisition_difference", "loss_difference", "net"]:
                        gross[name + "_" + metric] += pair[name + "_" + metric]
                pair_rows[(rid, key[1], key[2], window)] = pair
    with (RAW / "final_report/ACQUISITION_LOSS.csv").open() as stream:
        generated = list(csv.DictReader(stream))
    require(len(generated) == len(pair_rows) == 4096, "decomposition CSV cardinality")
    for row in generated:
        expected_pair = pair_rows[(row["request_id"], int(row["canonical_start"]), int(row["round_ordinal"]), int(row["window"]))]
        for field, expected_value in expected_pair.items():
            require(float(row[field]) == expected_value, f"CSV metric {field}")
    for filename, grouping, cardinality in [("PER_REQUEST.csv", "request_id", 64), ("PER_WINDOW.csv", "window", 16)]:
        with (RAW / "final_report" / filename).open() as stream:
            generated_means = list(csv.DictReader(stream))
        require(len(generated_means) == cardinality, filename + " cardinality")
        require(len({row[grouping] for row in generated_means}) == cardinality, filename + " uniqueness")
        for row in generated_means:
            grouped = [value for key, value in pair_rows.items()
                       if (key[0] == row[grouping] if grouping == "request_id" else key[3] == int(row[grouping]))]
            require(len(grouped) == (64 if grouping == "request_id" else 256), filename + " averaging count")
            for field, recorded in row.items():
                if field in {"request_id", "window", "starts", "windows", "requests", "pass_index"}:
                    continue
                source_field = field if field in grouped[0] else field + "_net"
                expected_value = sum(value[source_field] for value in grouped) / len(grouped)
                require(float(recorded) == expected_value, filename + ":" + field)
    comparison = report["comparison"]
    for field, number in gross.items():
        require(comparison["gross_counts"][field] == number, f"gross {field}")
    denominators = {"paired_block_windows": 4096, "valid_proposal_occurrences": valid_total,
                    "pre_first_blocker_opportunities": opportunity_blocker,
                    "pre_prefix_break_opportunities": opportunity_prefix,
                    "native_repair_displacement_opportunities": opportunity_native_repair}
    for field, number in denominators.items():
        require(comparison["denominators"][field] == number, f"denominator {field}")
    request_means = values.mean(axis=2).mean(axis=1)
    require(requests == comparison["bootstrap"]["request_order"], "deterministic bootstrap order")
    for i, state in enumerate(STATES):
        require(request_means[:, i].mean() == comparison["levels"][state], f"state mean {state}")
    draws = np.random.Generator(np.random.PCG64(1707)).integers(0, 64, size=(10000, 64))
    contrasts = {}
    for a, b in CONTRASTS:
        name = f"{a}_{b}"
        delta = request_means[:, STATES.index(a)] - request_means[:, STATES.index(b)]
        lower, upper = np.quantile(delta[draws].mean(axis=1), [0.025, 0.975], method="linear")
        result = dict(estimate=float(delta.mean()), lower=float(lower), upper=float(upper),
                      request_positive=int((delta > 0).sum()), request_negative=int((delta < 0).sum()), request_tie=int((delta == 0).sum()))
        for field in ["estimate", "lower", "upper"]:
            require(result[field] == comparison["contrasts"][name][field], f"contrast {name}:{field}")
        contrasts[name] = result
    pass_view = {}
    for index, slice_ in enumerate([slice(0, 8), slice(8, 16)]):
        pass_view[str(index)] = {f"{a}_{b}": float((values[:, slice_, :, STATES.index(a)] - values[:, slice_, :, STATES.index(b)]).mean()) for a,b in CONTRASTS}
    success = all(contrasts[name]["estimate"] > 0 and contrasts[name]["lower"] > 0 for name in ["E_S_low", "E_S_high"])
    require(not success and report["decision"] == "UNRESOLVED_STOP_THIS_SCREEN", "joint primary decision")
    return {"status": "passed", "scope": "Independent CPU raw-token rescoring and statistical reduction; no model work",
            "execution_commit": report["execution_commit"], "checks": checks,
            "raw_rows": len(rows), "raw_native_records_checked": dict(raw_counts), "request_count": 64,
            "paired_block_windows": 4096, "contrasts": contrasts, "gross_counts": dict(gross),
            "denominators": denominators, "pass_views": pass_view,
            "primary_joint_advancement_passed": success, "decision": report["decision"],
            "bootstrap": {"draws": 10000, "seed": 1707, "generator": "PCG64", "unit": "whole request", "percentiles": "linear"},
            "seconds": time.monotonic()-began, "numpy_version": np.__version__,
            "source_hashes": {str(path): digest(path) for path in [Path(__file__), RAW/"probes/PROBE_OUTCOMES.json", RAW/"final_report/SUMMARY.json", OLD/"sealed/PROBE_KEYS.json"]},
            "independence": "Separate same-model agent; production analysis inspected for contract, never imported or called. Native execution itself not independently rerun."}


if __name__ == "__main__":
    result = run()
    (OUT / "AUDIT.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ["status", "checks", "raw_rows", "contrasts", "seconds"]}, indent=2))
