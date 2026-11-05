"""Standalone CPU reconstruction of the K+ depth qualification/pricing records.

No experiment modules, model libraries, remote calls, or new executions are used.
This is a separate calculation by a same-team author who also implemented parts
of the experiment; it is not an independent GPU replication or blinded review.
"""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


COMMIT = "9150cc3b20714153b8e4fb3d70e8a32861c5921f"
ROUTES = ("FIVE", "THREE", "ADAPTIVE3", "ADAPTIVE5")
MODES = ["FORCED5"] * 4 + ["THREE"] * 4 + ["ALTERNATE"] * 8
BINS = (0, 256, 512, 1024, 2048, 4096, math.inf)
RESTORATION = ("request_row_ids", "request_mapping", "allocator", "native_graph_inputs", "attention_metadata", "cache_bytes")
GRAPH_LAYERS = {"static_three": [1, 1, 1, 0, 0], "prefix": [1, 1, 1, 0, 0],
                "shallow": [0, 0, 0, 0, 0], "deep": [0, 0, 0, 1, 1]}


def starts(count):
    n = min(8, count)
    return [0] if n == 1 else [i * (count - 1) // (n - 1) for i in range(n)]


def finite_positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def raw_generation_issues(record, request, reference=None, mode="FIVE"):
    issues = []
    for key in ("prompt_ids", "stop_token_ids", "max_new_tokens"):
        if record[key] != request[key]:
            issues.append(key + "_binding")
    tokens, meta = record["response_ids"], record["metadata"]
    if not tokens or len(tokens) > 2048 or any(type(t) is not int or t < 0 for t in tokens):
        issues.append("invalid_output_tokens")
    eos = request["stop_token_ids"]
    if any(t in eos for t in tokens[:-1]):
        issues.append("early_eos")
    expected_end = "eos" if tokens and tokens[-1] in eos else "length"
    if record["finish_reason"] != expected_end or expected_end == "length" and len(tokens) != 2048:
        issues.append("termination")
    offset, accepted = 0, 0
    for index, block in enumerate(meta["trace"]):
        progress = block["committed_progress"]
        if (block["output_start"] != offset or offset >= len(tokens)
                or block["anchor_token"] != tokens[offset]
                or block["committed_tokens"] != tokens[offset + 1:offset + 1 + progress]
                or type(progress) is not int or progress <= 0):
            issues.append(f"round_{index}_commit")
        anchor = len(request["prompt_ids"]) + offset
        if block["positions"] != list(range(anchor, anchor + 16)) or len(block["draft_tokens"]) != 15:
            issues.append(f"round_{index}_geometry")
        if not 0 <= block["accepted_draft_tokens"] <= progress:
            issues.append(f"round_{index}_accepted")
        offset += progress
        accepted += block["accepted_draft_tokens"]
    if (offset != len(tokens) - 1 or meta["committed_progress"] != offset
            or meta["verification_calls"] != len(meta["trace"])
            or meta["accepted_draft_tokens"] != accepted):
        issues.append("native_count_totals")
    if any(not finite_positive(meta[key]) for key in ("request_latency_s", "native_generate_s")):
        issues.append("timing")
    if meta.get("feedback_capture_in_timed_request") is not False:
        issues.append("feature_capture_enabled")
    if reference is not None:
        if tokens != reference["response_ids"] or record["finish_reason"] != reference["finish_reason"]:
            issues.append("canonical_parity")
        if mode == "FORCED5":
            other = reference["metadata"]
            if meta["verification_calls"] != other["verification_calls"]:
                issues.append("forced_five_round_count")
            for i, (old, new) in enumerate(zip(other["trace"], meta["trace"])):
                for key in ("positions", "anchor_token", "draft_tokens", "committed_tokens", "committed_progress"):
                    if old[key] != new[key]:
                        issues.append(f"forced_five_round_{i}_{key}")
    return issues


def summarize_routes(measurements, native_prompts, *, economic_eligible):
    """Independent scalar calculation; only complete four-route starts enter."""
    grouped = defaultdict(list)
    for row in measurements:
        grouped[row["request_id"], row["round_index"]].append(row)
    per_start, per_prompt = [], []
    for (request_id, index), rows in sorted(grouped.items()):
        keys = {(r["route"], r["repeat"]) for r in rows}
        if (len(rows) != 16 or keys != {(r, i) for r in ROUTES for i in range(4)}
                or any(r.get("valid") is not True or not finite_positive(r["elapsed_ms"]) for r in rows)):
            continue
        means = {route: math.fsum(r["elapsed_ms"] for r in rows if r["route"] == route) / 4
                 for route in ROUTES}
        nrounds = native_prompts[request_id]["rounds"]
        row = dict(request_id=request_id, round_index=index, prefix_length=rows[0]["prefix_length"],
                   expansion_weight=nrounds / len(starts(nrounds)), **{r + "_ms": v for r, v in means.items()})
        row["optimistic_saving_ms"] = max(0.0, means["FIVE"] - min(means["ADAPTIVE3"], means["ADAPTIVE5"]))
        row["continuation_delta_ms"] = means["ADAPTIVE5"] - means["ADAPTIVE3"]
        per_start.append(row)
    for request_id, native in native_prompts.items():
        anchors = [r for r in per_start if r["request_id"] == request_id]
        if len(anchors) != len(starts(native["rounds"])):
            continue
        per_prompt.append(dict(
            request_id=request_id, observed_rounds=native["rounds"], sampled_rounds=len(anchors),
            request_ms=1000 * native["request_s"],
            optimistic_saved_ms=math.fsum(r["expansion_weight"] * r["optimistic_saving_ms"] for r in anchors),
            continuation_delta_ms=float(np.mean([r["continuation_delta_ms"] for r in anchors])),
            **{route + "_ms": float(np.mean([r[route + "_ms"] for r in anchors])) for route in ROUTES},
        ))
    bins = []
    for lower, upper in zip(BINS, BINS[1:]):
        anchors = [r for r in per_start if lower <= r["prefix_length"] < upper]
        if not anchors:
            continue
        weight = math.fsum(r["expansion_weight"] for r in anchors)
        bins.append(dict(
            prefix_interval=f"[{lower},{upper})", sampled_starts=len(anchors),
            requests=len({r["request_id"] for r in anchors}), expanded_round_weight=weight,
            **{route + "_sample_mean_ms": float(np.mean([r[route + "_ms"] for r in anchors])) for route in ROUTES},
            **{route + "_weighted_mean_ms": math.fsum(r["expansion_weight"] * r[route + "_ms"] for r in anchors) / weight for route in ROUTES},
            paired_continuation_delta_ms=float(np.mean([r["continuation_delta_ms"] for r in anchors])),
        ))
    result = dict(per_start=per_start, per_prompt=per_prompt, context_bins=bins, economic=None,
                  complete_prompt_costs=len(per_prompt), complete_start_costs=len(per_start))
    if economic_eligible and len(per_prompt) == 16:
        values = np.asarray([[r["optimistic_saved_ms"], r["request_ms"]] for r in per_prompt])
        sampled = np.random.default_rng(20261105).integers(0, 16, size=(10000, 16))
        totals = values[sampled].sum(axis=1)
        interval = np.quantile(100 * totals[:, 0] / totals[:, 1], [0.025, 0.975], method="linear")
        result["economic"] = dict(
            point_pct=float(100 * values[:, 0].sum() / values[:, 1].sum()), ci95_pct=interval.tolist(),
            optimistic_saved_ms=float(values[:, 0].sum()), native_request_ms=float(values[:, 1].sum()),
            continuation_delta_ms=float(np.mean([r["continuation_delta_ms"] for r in per_start])),
            decision="stop_pricing_only" if interval[1] < 2 else "continue_to_readout_fit",
        )
    return result


class Audit:
    def __init__(self, root):
        self.root, self.checks, self.errors, self.sources = Path(root), 0, [], {}

    def check(self, condition, label):
        self.checks += 1
        if not condition:
            self.errors.append(label)

    def load(self, path, optional=False):
        path = Path(path)
        if not path.is_absolute():
            path = self.root / path
        if optional and not path.exists():
            return None
        raw = path.read_bytes()
        self.sources[str(path)] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        return json.loads(raw)


def inspect_run(root, manifest_path, expected_commit=COMMIT):
    audit = Audit(root)
    manifest = audit.load(Path(manifest_path).absolute())
    requests = manifest["manifests"]["READOUT"]["requests"][:16]
    plan, status = audit.load("PLAN.json"), audit.load("STATUS.json", optional=True)
    preflight = audit.load("PREFLIGHT.json", optional=True)
    audit.check(plan["code_commit"] == expected_commit, "execution_commit")
    bound = plan["input_identities"][plan["manifest_file"]]
    actual = audit.sources[str(Path(manifest_path).absolute())]
    audit.check(bound["bytes"] == actual["bytes"] and bound["sha256"] == actual["sha256"], "sealed_manifest_identity")
    generation_rows = audit.load("GENERATIONS.json", optional=True) or []
    declared = {row["label"]: row for row in generation_rows}
    records, generation_findings, profiles, measurements, native_prompts = {}, [], [], [], {}
    route_status_count, graph_evidence = 0, []
    for ordinal, request in enumerate(requests):
        for phase, mode in (("READOUT", "FIVE"), ("qualification", MODES[ordinal])):
            label = f"{ordinal:02d}_{phase}_{mode}"
            record = audit.load(label + ".json", optional=True)
            if record is None:
                continue
            reference = records.get(f"{ordinal:02d}_READOUT_FIVE") if phase == "qualification" else None
            issues = raw_generation_issues(record, request, reference, mode)
            audit.check(phase == "READOUT" or reference is not None, label + ":reference_present")
            records[label] = record
            generation_findings.append(dict(label=label, phase=phase, mode=mode, valid=not issues, issues=issues))
            if label in declared:
                audit.check(declared[label]["valid"] == (not issues), label + ":reported_validity")
                audit.check(declared[label]["verification_calls"] == len(record["metadata"]["trace"]), label + ":reported_calls")
            if phase == "READOUT" and not issues:
                native_prompts[request["request_id"]] = dict(rounds=len(record["metadata"]["trace"]), request_s=record["metadata"]["request_latency_s"])
            if phase == "qualification":
                route = audit.load(f"{ordinal:02d}_ROUTE_STATUS.json", optional=True)
                if route:
                    route_status_count += 1
                    audit.check(route.get("status") == "passed", label + ":native_route_status")
                    rounds = len(record["metadata"]["trace"])
                    desired = [5] * rounds if mode == "FORCED5" else [3] * rounds if mode == "THREE" else [(3, 5)[i % 2] for i in range(rounds)]
                    audit.check(route["routes"] == desired, label + ":whole_block_depths")
                    audit.check(route.get("live_rounds") == rounds, label + ":live_rounds")
                    audit.check(route["head_invocations"] == 3 * rounds, label + ":one_live_head_plus_two_audit_heads")
                    audit.check(route.get("shadow_proposal_forwards") == 2 * rounds,
                                label + ":two_shadow_proposals_per_live_round")
                    audit.check(route.get("shadow_vocab_projections") == 2 * rounds,
                                label + ":two_shadow_heads_per_live_round")
                    audit.check(len(route["audit_records"]) == rounds, label + ":same_start_audit_count")
                    if mode == "ALTERNATE":
                        audit.check(rounds >= 3 and route["routes"][:3] == [3, 5, 3],
                                    label + ":actual_three_five_three_switch")
                    setups = route.get("graph_setup", [])
                    audit.check({r["graph"] for r in setups} == set(GRAPH_LAYERS),
                                label + ":four_captured_graphs")
                    for setup in setups:
                        name = setup["graph"]
                        audit.check(setup["captured_layer_calls"] == GRAPH_LAYERS.get(name),
                                    label + ":captured_layers_" + name)
                        audit.check(setup.get("warmup_executions") == 2 and setup.get("capture_executions") == 1,
                                    label + ":graph_setup_accounting_" + name)
                        graph_evidence.append(dict(label=label, **setup))
                    for i, pair in enumerate(route["audit_records"]):
                        equality = len(pair["original_five_proposals"]) == 15 and pair["original_five_proposals"] == pair["forced_five_proposals"]
                        audit.check(pair["all_15_equal"] == equality, label + f":shadow_{i}_reported_parity")
                        audit.check(equality, label + f":shadow_{i}_actual_parity")
                        audit.check(pair["canonical_start"] == record["metadata"]["trace"][i]["positions"][0], label + f":shadow_{i}_canonical_start")
        profile = audit.load(f"{ordinal:02d}_PROFILE.json", optional=True)
        if profile is None:
            continue
        profiles.append(profile)
        rows = profile["measurements"]
        measurements.extend(rows)
        baseline = records[f"{ordinal:02d}_READOUT_FIVE"]
        trace = baseline["metadata"]["trace"]
        indices = starts(len(trace))
        audit.check(profile["sampled_round_indices"] == indices, f"profile_{ordinal}:sampling")
        expected = []
        for a, index in enumerate(indices):
            for repeat in range(4):
                rotation = (ordinal + a + repeat) % 4
                for position, route in enumerate(ROUTES[rotation:] + ROUTES[:rotation]):
                    expected.append((index, a, repeat, route, position))
        actual = [(r["round_index"], r["anchor_ordinal"], r["repeat"], r["route"], r["route_position"]) for r in rows]
        audit.check(actual == expected[:len(actual)], f"profile_{ordinal}:balanced_order")
        for i, row in enumerate(rows):
            original = trace[row["round_index"]]
            valid = row["route"] not in ("FIVE", "ADAPTIVE5") or row["proposal_ids"] == original["draft_tokens"]
            audit.check(row["valid"] == valid, f"profile_{ordinal}:row_{i}_native_parity")
            audit.check(len(row["proposal_ids"]) == 15 and row["prefix_length"] == original["positions"][0], f"profile_{ordinal}:row_{i}_geometry")
            audit.check(row["depth"] == (5 if row["route"] in ("FIVE", "ADAPTIVE5") else 3), f"profile_{ordinal}:row_{i}_depth")
            audit.check(finite_positive(row["elapsed_ms"]), f"profile_{ordinal}:row_{i}_time")
        for pair in profile.get("parity", []):
            original = trace[pair["block_index"]]["draft_tokens"]
            audit.check(pair["expected"] == original and pair["all_15_equal"] == (pair["observed"] == original), f"profile_{ordinal}:warmup_parity")
        accounting = profile.get("accounting")
        if accounting:
            audit.check(accounting["planned_timed_measurements"] == len(expected), f"profile_{ordinal}:planned_work")
            audit.check(accounting["returned_timed_measurements"] == len(rows), f"profile_{ordinal}:returned_work")
            audit.check(sum(accounting[k] for k in ("returned_timed_measurements", "unknown_timed_measurements", "unattempted_timed_measurements")) == len(expected), f"profile_{ordinal}:partial_work_partition")
        if profile["status"] == "passed":
            audit.check(len(rows) == len(expected) and all(r["valid"] for r in rows), f"profile_{ordinal}:passed_grid")
            audit.check(profile.get("restoration_status") == "passed" and all(profile.get("restoration_checks", {}).get(k) is True for k in RESTORATION), f"profile_{ordinal}:exact_restoration")
    original_measurements = audit.load("PRICING_MEASUREMENTS.json", optional=True)
    if original_measurements is not None:
        audit.check(original_measurements == measurements, "flattened_profile_measurements")
    weights = [audit.load(name, optional=True) for name in ("WEIGHTS_BEFORE.json", "WEIGHTS_AFTER.json")]
    weights_pass = all(row and row.get("status") == "passed" and row.get("all_equal") is True for row in weights)
    unknown_generation = int(bool(status and status.get("unknown_generations")))
    generation_counts = dict(planned=32, returned=len(records), valid=sum(row["valid"] for row in generation_findings),
                             invalid=sum(not row["valid"] for row in generation_findings), unknown=unknown_generation,
                             unattempted=32 - len(records) - unknown_generation)
    if status:
        for key in ("returned", "valid", "invalid", "unknown", "unattempted"):
            audit.check(status[key + "_generations"] == generation_counts[key], "status_generation_" + key)
        audit.check(all(status[key] == 0 for key in ("optimizer_steps", "fitting_presentations", "final_generations")), "no_fitting_or_final_work")
    complete = len(records) == 32 and generation_counts["invalid"] == 0 and len(profiles) == 16 and route_status_count == 16 and all(p["status"] == "passed" for p in profiles) and weights_pass and not audit.errors
    complete = bool(complete and preflight and preflight.get("status") == "passed"
                    and preflight.get("code_commit") == expected_commit)
    stats = summarize_routes(measurements, native_prompts, economic_eligible=complete)
    production = audit.load("PRICING.json", optional=True)
    if production and stats["economic"]:
        expected = stats["economic"]
        for key in ("point_pct", "ci95_pct"):
            audit.check(np.allclose(production["optimistic_savings"][key], expected[key], rtol=1e-12, atol=1e-12), "reconstructed_economic_" + key)
        audit.check(math.isclose(production["continuation_cost_ms"], expected["continuation_delta_ms"], rel_tol=1e-12, abs_tol=1e-12), "reconstructed_continuation_cost")
        audit.check(production["decision"] == expected["decision"], "reconstructed_economic_decision")
    if status and status["status"] == "completed":
        audit.check(complete, "claimed_complete_qualification")
    unknown_profiles = [dict(request_id=p["request_id"], active=p.get("active_operation"), accounting=p.get("accounting"))
                        for p in profiles if p["status"] != "passed"]
    active_profile = status.get("active_profile") if status else None
    return dict(
        schema="independent_block_depth_audit_v1", execution_commit=plan["code_commit"],
        audit_checks=audit.checks, audit_errors=audit.errors, qualified_complete=complete and not audit.errors,
        generation_accounting=generation_counts, generation_findings=generation_findings,
        route_accounting=dict(maximum=2048, returned=len(measurements), profiles_with_receipts=len(profiles),
                              completed_profiles=sum(p["status"] == "passed" for p in profiles),
                              incomplete_profiles=unknown_profiles, active_profile=active_profile,
                              total_unknown="See per-profile receipts/bounds; absent interrupted receipts do not imply zero work"),
        weights_before_after_pass=weights_pass, statistics=stats, source_identities=audit.sources,
        native_route_evidence=dict(status_receipts=route_status_count, graph_capture_receipts=graph_evidence),
        independence="Separate CPU calculations without experiment imports; same-team author also implemented data/pricing/qualification, not a blinded review or independent GPU replication",
        interpretation="No economic conclusion without complete native qualification. A pricing stop is a same-progress skip-cost budget decision, not proof that better proposals cannot compensate. No final endpoint or fitting is authorized here.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-commit", default=COMMIT)
    args = parser.parse_args()
    result = inspect_run(args.root, args.manifest, args.expected_commit)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "AUDIT.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for key in ("per_start", "per_prompt", "context_bins"):
        rows = result["statistics"][key]
        if rows:
            with (args.output / (key.upper() + ".csv")).open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    lines = ["# K+ depth qualification/pricing evidence audit", "",
             f"Execution commit: `{result['execution_commit']}`.",
             f"Checks: {result['audit_checks']}; audit discrepancies: {len(result['audit_errors'])}.",
             f"Complete native qualification supported: {result['qualified_complete']}.", "",
             "Generation accounting: " + json.dumps(result["generation_accounting"]),
             "Returned timed route measurements: " + str(result["route_accounting"]["returned"]), "",
             "Economic result: " + json.dumps(result["statistics"]["economic"]), "",
             result["independence"], "", result["interpretation"]]
    (args.output / "AUDIT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({key: result[key] for key in ("audit_checks", "audit_errors", "qualified_complete", "generation_accounting")}, indent=2))
    if result["audit_errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
