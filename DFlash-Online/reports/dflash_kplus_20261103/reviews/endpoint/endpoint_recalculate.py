#!/usr/bin/env python3
"""Private, separate CPU reconstruction of the fixed K+ endpoint.

Imports only the standard library and NumPy. It runs no model, imports no
experiment implementation, and never emits private prompt/response tokens.
"""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import numpy as np


EXECUTION = "e0a9d3c235e1efd15404c549282d21c99243a21c"
METHODS = ("K", "KPLUS")
COUNTS = ("verification_calls", "accepted_draft_tokens", "committed_progress", "output_tokens")
TIMES = ("request_latency_s", "native_generate_s", "cache_maintenance_s")
WORK = dict(online_generations=7473, correctness_generations=4,
            terminal_preparation_forwards=7424, response_forwards=14848,
            response_backwards=14848, optimizer_steps=7424)
BUDGET = dict(training_events=58, optimizer_steps=7424, response_presentations=14848,
              response_forwards=14848, response_backwards=14848, online_generations=7473,
              terminal_preparation_forwards=7424, intermediate_generations=0,
              final_generations=6144, correctness_generations=4, total_generations=13621,
              disposable_optimizer_steps=0, baseline_optimizer_steps=0)
FAILURE_FILES = ("campaign_001/STATUS.json", "campaign_001/FINAL_AUDIT.json",
                 "EXIT_CODE", "CAMPAIGN_EXIT_CODE", "REPORT_EXIT_CODES", "POSTFLIGHT.json")


def require(condition, label):
    if not condition:
        raise ValueError(label)


def tokens(values):
    return isinstance(values, list) and all(type(value) is int and value >= 0 for value in values)


def schedule(requests):
    require(len(requests) == 1024, "endpoint requires 1024 sealed requests")
    require([row["panel_ordinal"] for row in requests] == list(range(1024)), "sealed panel order")
    require(len({row["request_id"] for row in requests}) == 1024, "sealed request identities")
    slots = []
    for block in range(16):
        for repeat in range(3):
            order = METHODS if (block + repeat) % 2 == 0 else tuple(reversed(METHODS))
            for position, method in enumerate(order):
                for ordinal in range(block * 64, (block + 1) * 64):
                    slots.append(dict(seed=17, event=58, phase="final", block_ordinal=block,
                        panel_ordinal=ordinal, request_id=requests[ordinal]["request_id"],
                        repeat=repeat, method=method, position=position))
    return slots


def native_record(raw, request, canonical, checkpoint, label):
    """Rebuild output and every verification decision before using scalars."""
    meta, capture = raw["metadata"], raw["metadata"]["capture"]
    response, prompt = raw["response_ids"], raw["prompt_ids"]
    require(raw["record_id"] == label and raw["request_index"] == request["panel_ordinal"]
            and raw["session_id"] == "kplus_full_gsm8k_evaluation", "raw request identity")
    require(raw["checkpoint_id"] == checkpoint and raw["purpose"] == "evaluation", "raw checkpoint/purpose")
    require(tokens(prompt) and prompt == request["prompt_ids"] and prompt, "exact prompt IDs")
    require(tokens(response) and 0 < len(response) <= 2048, "complete response IDs")
    require(raw["stop_token_ids"] == request["stop_token_ids"] and raw["max_new_tokens"] == 2048,
            "stop tokens/output cap")
    require(response == canonical["response_ids"] and raw["finish_reason"] == canonical["finish_reason"],
            "canonical output/stop equality")
    stops = [index for index, token in enumerate(response) if token in request["stop_token_ids"]]
    require((stops == [len(response) - 1] and raw["finish_reason"] == "eos")
            or (not stops and len(response) == 2048 and raw["finish_reason"] == "length"), "termination semantics")
    require(meta["block_size"] == 16 and meta["context_limit"] == 4096
            and meta["runtime"] == "pinned_SGLang_FA3"
            and meta["cache_policy"] == "idle_full_flush_before_every_request_and_publication", "runtime geometry/cache")
    require(meta["feedback_capture_in_timed_request"] is False and capture["collect_features"] is False
            and capture["feature_bytes"] == 0 and capture["native_capture_validated"] is True, "capture-off contract")
    require(capture["prompt_ids"] == prompt and capture["response_ids"] == response
            and capture["blocks"] == meta["trace"], "native trace binding")
    require(capture["status"] == "passed" and capture["target_argmax_ids"] == response
            and capture["covered_token_ids"] == prompt + response[:-1]
            and capture["covered_rows"] == len(prompt) + len(response) - 1
            and capture["block_size"] == 16 and type(capture["prefill_chunks"]) is int
            and capture["prefill_chunks"] > 0, "native committed coverage")
    require(capture["label_binding_scope"] == "native_prefill_and_committed_verification"
            and capture["final_feature_logits_used_for_labels"] is False
            and capture["final_emitted_token_feature_present"] is False
            and meta["trace_instrumentation_in_timed_request"] is True, "native label and timing contract")
    rebuilt, accepted = [response[0]], 0
    for index, block in enumerate(meta["trace"]):
        proposed, posterior = block["draft_tokens"], block["target_argmax"]
        width, progress = block["verify_size"], block["committed_progress"]
        require(tokens(proposed) and tokens(posterior) and type(width) is int
                and width == len(proposed) + 1 == len(posterior) == 16,
                "native verification geometry")
        mismatch = next((index for index in range(width - 1) if proposed[index] != posterior[index]), width - 1)
        require(type(progress) is int and 1 <= progress <= mismatch + 1, "native committed progress")
        committed = (proposed[:mismatch] + [posterior[mismatch]])[:progress]
        require(block["committed_tokens"] == committed == posterior[:progress], "native posterior prefix")
        require(block["anchor_token"] == rebuilt[-1] and block["output_start"] == len(rebuilt) - 1,
                "native causal anchor")
        start = len(prompt) + len(rebuilt) - 1
        require(block["positions"] == list(range(start, start + width)), "native causal positions")
        require(block["accepted_draft_tokens"] == min(mismatch, progress)
                and block["native_accepted_draft_tokens"] == progress - 1
                and block["raw_accepted_draft_tokens_before_stop"] == mismatch
                and block["stop_clipped"] == (progress < mismatch + 1), "native acceptance accounting")
        require(block["finished"] is (index == len(meta["trace"]) - 1), "native finish boundary")
        require(not block["stop_clipped"] or block["finished"], "clipped progress only at termination")
        rebuilt.extend(committed)
        accepted += min(mismatch, progress)
    require(rebuilt == response, "complete reconstructed output")
    counts = dict(verification_calls=len(meta["trace"]), accepted_draft_tokens=accepted,
                  committed_progress=len(response) - 1, output_tokens=len(response))
    for name in COUNTS[:-1]:
        require(type(meta[name]) is int and meta[name] == counts[name], "native aggregate " + name)
    require(counts["verification_calls"] > 0 and 0 <= accepted <= counts["committed_progress"], "native count bounds")
    values = dict(counts)
    for name in TIMES:
        value = capture[name] if name == "cache_maintenance_s" else meta[name]
        require(type(value) in (int, float) and math.isfinite(value)
                and (value >= 0 if name == "cache_maintenance_s" else value > 0), "finite clock " + name)
        values[name] = float(value)
    values.update(prompt_tokens=len(prompt), cap_hit=raw["finish_reason"] == "length",
                  finish_reason=raw["finish_reason"], checkpoint_id=checkpoint)
    return values


def paired_rows(rows, requests):
    """Require every original slot, then count requests once and average times."""
    expected = schedule(requests)
    require(len(rows) == 6144, "6144 final measurements required")
    grouped = defaultdict(list)
    for row, slot in zip(rows, expected):
        require(all(row.get(key) == value for key, value in slot.items()), "complete chronological final schedule")
        require(row["valid_measurement"] is True and row["validation_status"] == "valid"
                and not row["errors"] and not row["structural_errors"], "invalid final measurement")
        require(all(type(row[key]) is int and row[key] >= 0 for key in COUNTS), "integer final counts")
        require(row["verification_calls"] > 0 and row["committed_progress"] == row["output_tokens"] - 1
                and 0 <= row["accepted_draft_tokens"] <= row["committed_progress"], "final count structure")
        for name in TIMES:
            require(type(row[name]) in (int, float) and math.isfinite(row[name])
                    and (row[name] >= 0 if name == "cache_maintenance_s" else row[name] > 0), "final clock " + name)
        grouped[slot["panel_ordinal"], slot["method"]].append(row)
    paired = []
    for ordinal, request in enumerate(requests):
        result = dict(panel_ordinal=ordinal, request_id=request["request_id"])
        for method in METHODS:
            repeats = sorted(grouped[ordinal, method], key=lambda row: row["repeat"])
            require([row["repeat"] for row in repeats] == [0, 1, 2], "all three repetitions")
            require(all(all(row[key] == repeats[0][key] for key in COUNTS) for row in repeats), "repeat count stability")
            result[method] = {key: repeats[0][key] for key in COUNTS}
            result[method].update({key: sum(row[key] for row in repeats) / 3 for key in TIMES})
        require(result["K"]["output_tokens"] == result["KPLUS"]["output_tokens"], "paired output token count")
        paired.append(result)
    return paired


def reconstruct_statistics(paired, rows):
    """Independent ratio-of-sums and paired request bootstrap, without strata."""
    require(len(paired) == 1024, "complete paired panel for statistics")
    arrays = {method: {key: np.array([row[method][key] for row in paired], dtype=np.float64)
                       for key in COUNTS + TIMES} for method in METHODS}
    totals = {}
    for method, values in arrays.items():
        total = {key: int(values[key].sum()) for key in COUNTS}
        total.update({key: float(values[key].sum()) for key in TIMES})
        calls = total["verification_calls"]
        total.update(accepted_tokens_per_call=total["accepted_draft_tokens"] / calls,
                     committed_progress_per_call=total["committed_progress"] / calls,
                     output_tokens_per_call=total["output_tokens"] / calls,
                     output_tokens_per_s=total["output_tokens"] / total["request_latency_s"])
        totals[method] = total
    generator = np.random.Generator(np.random.PCG64(20261102))
    draws = {key: np.empty(10000) for key in ("verification_calls", "request_latency_s")}
    throughput = np.empty(10000)
    for replicate in range(10000):
        indices = generator.integers(0, 1024, size=1024)
        for key in draws:
            reference = arrays["K"][key][indices].sum()
            candidate = arrays["KPLUS"][key][indices].sum()
            draws[key][replicate] = 100 * (1 - candidate / reference)
            if key == "request_latency_s":
                throughput[replicate] = reference / candidate
    comparisons = {}
    for name, key in (("call_reduction", "verification_calls"), ("request_latency_s_reduction", "request_latency_s")):
        comparisons[name] = dict(
            point_pct=100 * (1 - totals["KPLUS"][key] / totals["K"][key]),
            ci95_pct=np.percentile(draws[key], [2.5, 97.5], method="linear").tolist(),
            reference_sum=totals["K"][key], candidate_sum=totals["KPLUS"][key])
    difference = arrays["K"]["verification_calls"] - arrays["KPLUS"]["verification_calls"]
    comparisons.update(request_wins=int(np.count_nonzero(difference > 0)),
        request_regressions=int(np.count_nonzero(difference < 0)),
        request_ties=int(np.count_nonzero(difference == 0)), request_denominator=1024,
        gross_calls_saved=int(difference[difference > 0].sum()),
        gross_calls_added=int(-difference[difference < 0].sum()),
        serving_throughput_ratio=dict(point=totals["K"]["request_latency_s"] / totals["KPLUS"]["request_latency_s"],
            ci95=np.percentile(throughput, [2.5, 97.5], method="linear").tolist()))
    repeats = []
    for repeat in range(3):
        values = {method: {key: sum(row[key] for row in rows if row["repeat"] == repeat and row["method"] == method)
                           for key in COUNTS + TIMES} for method in METHODS}
        repeats.append(dict(repeat=repeat, method_totals=values,
            request_time_reduction_pct=100 * (1 - values["KPLUS"]["request_latency_s"] / values["K"]["request_latency_s"])))
    return dict(request_count=1024, method_totals=totals, comparisons={"KPLUS_vs_K": comparisons},
        repeat_totals=repeats, bootstrap=dict(draws=10000, seed=20261102, generator="PCG64",
            sampling="1024 paired request indices with replacement; same indices for both endpoints/methods",
            percentiles="2.5/97.5 linear", count_repeats="verified identical, counted once",
            timing_repeats="mean of three per method/request before resampling"))


class Reader:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.origin = None
        self.files = {}

    def path(self, value):
        path = Path(value)
        if path.is_absolute() and not path.is_relative_to(self.root):
            require(self.origin is not None and path.is_relative_to(self.origin), "unmapped external evidence path")
            path = self.root / path.relative_to(self.origin)
        elif not path.is_absolute():
            path = self.root / path
        require(path.resolve().is_relative_to(self.root), "evidence path escapes retrieved root")
        return path

    def read(self, value, lines=False, plain=False):
        path = self.path(value)
        payload = path.read_bytes()
        self.files[str(path.relative_to(self.root))] = dict(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
        if plain:
            return payload.decode()
        return [json.loads(line) for line in payload.splitlines() if line] if lines else json.loads(payload)


def reporting_recovery(reader, receipt):
    """Authorize analysis of one preserved operational failure, never raw work."""
    require(receipt["schema"] == "kplus_cpu_reporting_recovery_v1"
            and receipt["status"] == "completed_saved_evidence_reporting_recovery"
            and receipt["execution_commit"] == EXECUTION and receipt["model_work"] == 0,
            "explicit completed CPU reporting-recovery receipt")
    revision = receipt["analysis_commit"]
    require(isinstance(revision, str) and len(revision) == 40
            and all(value in "0123456789abcdef" for value in revision) and revision != EXECUTION,
            "separate CPU reporting analysis commit")
    require(set(receipt["preserved_evidence"]) == set(FAILURE_FILES), "exact six original failure bindings")
    originals = {}
    for name in FAILURE_FILES:
        originals[name] = reader.read(name, plain=not name.endswith(".json"))
        require(reader.files[name]["sha256"] == receipt["preserved_evidence"][name],
                "unchanged original failure evidence " + name)
    campaign = originals["campaign_001/STATUS.json"]
    audit = originals["campaign_001/FINAL_AUDIT.json"]
    require(campaign["status"] == "failed" and campaign["automatic_retry"] is False
            and not campaign.get("cleanup_errors")
            and campaign["error"] == "ValueError('Complete final endpoint failed raw-evidence validation')",
            "only original completed-work reporting failure is recoverable")
    require(audit["status"] == "invalid_or_incomplete" and audit["rows"] == []
            and audit["errors"] == ["missing_or_malformed_evidence: FileNotFoundError(2, 'No such file or directory')"],
            "original missing-path aggregate audit with zero validated rows")
    expected_exits = {"EXIT_CODE": "1", "CAMPAIGN_EXIT_CODE": "1", "REPORT_EXIT_CODES": "report=1\npostflight=0"}
    require(all(originals[name].strip() == expected for name, expected in expected_exits.items()),
            "original wrapper/campaign/report/postflight exit state")
    require(originals["POSTFLIGHT.json"]["status"] == "passed", "passed original postflight")
    for name in ("training_quiescence", "post_evaluation_boundary"):
        boundary = campaign[name]
        require(boundary["resource"]["status"] == boundary["timing"]["status"] == "passed",
                "passed original pre/post evaluation boundaries")


def completed_work(reader, analysis_recovery=None):
    starts, ends = {key: set() for key in WORK}, {key: set() for key in WORK}
    for row in reader.read("campaign_001/arms/KPLUS/WORK.jsonl", lines=True):
        kind, ordinal = row["kind"], row["ordinal"]
        require(kind in WORK and type(ordinal) is int and 1 <= ordinal <= WORK[kind], "declared work kind/ordinal")
        if row["phase"] == "started":
            require(ordinal == len(starts[kind]) + 1 and ordinal not in starts[kind], "unique sequential work start")
            starts[kind].add(ordinal)
        else:
            require(row["phase"] == "completed" and ordinal in starts[kind] and ordinal not in ends[kind], "work completion")
            ends[kind].add(ordinal)
    observed = {key: len(value) for key, value in ends.items()}
    require(observed == WORK and starts == ends, "complete work budget with zero unknown operations")
    status = reader.read("campaign_001/arms/KPLUS/STATUS.json")
    require(status["status"] == "completed" and status["events_completed"] == 58 and status["tail_completed"] == 49,
            "completed one-arm stream and untrained tail")
    require(status["actual_work"]["attempts"] == status["actual_work"]["completed"] == WORK
            and not any(status["actual_work"]["unknown"].values()), "saved work matches ledger")
    if analysis_recovery is None:
        require(reader.read("campaign_001/STATUS.json")["status"] == "completed", "campaign completion")
    else:
        reporting_recovery(reader, analysis_recovery)
    return observed


def serving_receipts(reader, folder, planned, checkpoints):
    """Authenticate the checkpoint and quiescence evidence outside request clocks."""
    transitions, current = [], None
    batches = []
    for slot in planned:
        batch = [slot[key] for key in ("block_ordinal", "repeat", "position")]
        if not batches or batches[-1] != batch:
            batches.append(batch)
        checkpoint = checkpoints[slot["method"]]
        if checkpoint != current:
            transitions.append(slot)
            current = checkpoint
    receipts = reader.read(folder + "ACTIVATIONS.jsonl", lines=True)
    require(len(receipts) == len(transitions), "complete publication inventory")
    helpers = ("helper_matches_current_layers", "helper_norm_matches_current_layers", "fused_enabled")
    for receipt, slot in zip(receipts, transitions):
        checkpoint = checkpoints[slot["method"]]
        state = receipt["native_status"]
        published = state["publication"]
        require(all(receipt.get(key) == value for key, value in slot.items())
                and not receipt["errors"] and receipt["checkpoint_structure_valid"] is True
                and receipt["checkpoint_id"] == checkpoint, "published slot and checkpoint")
        require(state["status"] == published["status"] == "passed" and state["poisoned"] is None
                and state["capture_pending"] is False and state["cache_tokens"] == published["cache_after_tokens"] == 0
                and state["attention_backend"] == "fa3" and state["block_size"] == 16
                and published["checkpoint_path"] == checkpoint
                and all(published[key] is True for key in helpers), "native publication contract")
    for method, checkpoint in checkpoints.items():
        receipt = reader.read(folder + "WEIGHTS_" + method + ".json")
        weights = receipt["native_reply"]
        require(not receipt["errors"] and weights["status"] == "passed" and weights["all_equal"] is True
                and weights["tensors"] == weights["source_tensor_count"] == 58
                and weights["native_tensor_count"] == 43 and weights["checkpoint_path"] == checkpoint
                and all(weights[key] is True for key in helpers), "loaded native weights " + method)

    def boundary(receipt):
        resource, timing = receipt["resource"], receipt["timing"]
        require(timing["status"] == "passed" and not timing["transfers"] and not timing["training"]
                and not timing["foreign_gpu_processes"], "timed serving quiescence")
        require(resource["status"] == "passed" and resource["compiler_growth_bytes"] <= 4 * 1024 ** 3
                and resource["free_bytes"] >= resource["required_additional_and_reserve_bytes"], "serving resource admission")

    boundaries = reader.read(folder + "BOUNDARY_CHECKS.jsonl", lines=True)
    require([row["batch"] for row in boundaries] == batches and len(batches) == 96,
            "one quiescence/resource receipt before each 64-request batch")
    for receipt in boundaries:
        boundary(receipt["check"])
    campaign = reader.read("campaign_001/STATUS.json")
    boundary(campaign["training_quiescence"])
    boundary(campaign["post_evaluation_boundary"])
    return dict(publications=len(receipts), loaded_weight_checks=2,
                batch_boundaries=96, before_backend_and_after_final_boundaries=2,
                scope="Saved boundary receipts; no claim of continuous external-process observation")


def historical_control(controls, baseline, identity):
    """Bind the separate count map to its sealed file and provenance receipt."""
    require(controls["status"] == "passed" and controls["model_work"] == 0
            and controls["baseline_counts_identity"] == identity, "authenticated retained K58 controls")
    provenance = controls["baseline_counts"]
    semantic_bytes = json.dumps(baseline, sort_keys=True, separators=(",", ":")).encode()
    totals = {key: sum(row[key] for row in baseline.values()) for key in COUNTS}
    require(provenance["requests"] == len(baseline) == 1024
            and provenance["repeats_per_request"] == 3
            and provenance["count_fields"] == list(COUNTS)
            and provenance["count_map_semantic_sha256"] == hashlib.sha256(semantic_bytes).hexdigest()
            and provenance["totals_per_repeat"] == totals
            and provenance["all_published_rows_match_raw"] is True, "retained K58 count-map provenance")


def endpoint_evidence(reader, analysis_recovery=None):
    plan = reader.read("sealed/PLAN.json")
    reader.origin = Path(plan["root"])
    require(plan["execution_source"]["commit"] == EXECUTION and plan["execution_source"]["dirty"] is False,
            "launched execution source")
    require(plan["budget"] == BUDGET and plan["arms"] == ["KPLUS"] and plan["seed"] == 17
            and plan["events"] == 58 and plan["intermediate_events"] == []
            and plan["automatic_retry"] is False, "fixed one-arm scientific budget")
    seal = reader.read("sealed/SEAL.json")
    for name in ("PLAN.json", "CANONICALS.json", "BASELINE_COUNTS.json", "CONTROL_AUDIT.json"):
        reader.read("sealed/" + name)
        require(reader.files["sealed/" + name] == seal["files"][str(reader.origin / "sealed" / name)], "sealed input hash " + name)
    requests = plan["evaluation_requests"]
    planned = schedule(requests)
    require(plan["evaluation_schedule"] == planned, "presealed complete endpoint schedule")
    canonicals, baseline = reader.read(plan["canonicals"]), reader.read(plan["baseline_counts"])
    ids = {row["request_id"] for row in requests}
    require(set(canonicals) == set(baseline) == ids, "complete canonical and historical count inventories")
    for request in requests:
        require(tokens(request["prompt_ids"]) and request["prompt_ids"] and len(request["prompt_ids"]) <= 2048
                and tokens(request["stop_token_ids"]) and request["stop_token_ids"]
                and request["max_new_tokens"] == 2048, "sealed request token contract")
        old = baseline[request["request_id"]]
        require(set(old) == set(COUNTS) and all(type(value) is int and value >= 0 for value in old.values()), "retained count fields")
        require(old["verification_calls"] > 0 and old["output_tokens"] == len(canonicals[request["request_id"]]["response_ids"]),
                "historical output identity")
    require(sum(row["verification_calls"] for row in baseline.values()) == 43387
            and sum(row["output_tokens"] for row in baseline.values()) == 314847, "historical K58 totals")
    controls = reader.read("sealed/CONTROL_AUDIT.json")
    historical_control(controls, baseline, reader.files["sealed/BASELINE_COUNTS.json"])
    work = completed_work(reader, analysis_recovery)
    folder = "campaign_001/evaluation/final/"
    protocol, status = reader.read(folder + "PROTOCOL.json"), reader.read(folder + "STATUS.json")
    checkpoints = dict(K=plan["references"]["K"], KPLUS=str(reader.origin / "campaign_001/arms/KPLUS/trainer/events/event_058/export"))
    require(protocol["schedule"] == planned and protocol["checkpoints"] == checkpoints
            and protocol["capture"] is False and protocol["extra_warmup_generations"] == 0
            and protocol["historical_call_gate"] is True, "final protocol")
    for name, expected in (("REQUESTS.json", requests), ("CANONICAL_INPUT.json", canonicals), ("BASELINE_COUNTS.json", baseline)):
        require(reader.read(folder + name) == expected, "final immutable input copy " + name)
    expected_status = dict(status="completed", planned_generations=6144, attempted_generations=6144,
        returned_generations=6144, completed_valid_generations=6144, invalid_generations=0,
        unknown_generation_attempts=0, unattempted_generations=0, pending_canonical_generations=0,
        extra_generations=0, automatic_retry=False, optimizer_steps=0, feedback_collections=0)
    require(all(status.get(key) == value for key, value in expected_status.items()), "final budget and validity status")
    serving = serving_receipts(reader, folder, planned, checkpoints)
    rows = reader.read(folder + "MEASUREMENTS.jsonl", lines=True)
    require(len(rows) == 6144, "complete raw-measurement inventory")
    blocks, failures = 0, []
    for saved, slot in zip(rows, planned):
        ordinal, method = slot["panel_ordinal"], slot["method"]
        label = f"{ordinal:04d}_r{slot['repeat']}_{method}"
        try:
            require(all(saved.get(key) == value for key, value in slot.items()), "chronological slot")
            raw_path = folder + label + ".json"
            require(reader.path(saved["record_path"]) == reader.path(raw_path), "saved raw path")
            raw = reader.read(raw_path)
            value = native_record(raw, requests[ordinal], canonicals[slot["request_id"]], checkpoints[method], label)
            require(reader.read(raw["metadata"]["native_trace_path"]) == raw["metadata"]["capture"], "original native sidecar")
            require(all(saved.get(key) == result for key, result in value.items()), "raw versus measured scalars")
            if method == "K":
                require(all(value[key] == baseline[slot["request_id"]][key] for key in COUNTS), "historical K58 per-request counts")
            blocks += value["verification_calls"]
        except (KeyError, ValueError, TypeError, IndexError, OSError) as error:
            failures.append(label + ": " + type(error).__name__ + ": " + str(error))
    require(not failures, "raw endpoint failures: " + "; ".join(failures[:20]))
    paired = paired_rows(rows, requests)
    return paired, rows, dict(raw_executions=6144, native_blocks=blocks, training_work=work, serving=serving,
        reporting_recovery=analysis_recovery,
        total_complete_generations=WORK["online_generations"] + WORK["correctness_generations"] + len(rows),
        execution_source=plan["execution_source"], historical_control_scope="Authenticated retained count map, not new decoding of original K state tensors")


def run(root, output, *, analysis_recovery=None):
    started, reader = time.perf_counter(), Reader(root)
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    result = dict(status="invalid_or_incomplete", efficacy_interpretation_allowed=False, errors=[], endpoint=None,
        material_passport=dict(mode="validate", verification_status="ANALYZED", execution_commit=EXECUTION,
            scope="Separate CPU endpoint implementation; same-team objective authorship overlap; not independent GPU reproduction"))
    try:
        recovery = None
        if analysis_recovery is not None:
            payload = Path(analysis_recovery).read_bytes()
            recovery = json.loads(payload)
            result["analysis_recovery_receipt"] = dict(path=str(Path(analysis_recovery).resolve()),
                bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
        paired, rows, evidence = endpoint_evidence(reader, recovery)
        endpoint = reconstruct_statistics(paired, rows)
        contrast = endpoint["comparisons"]["KPLUS_vs_K"]
        calls_supported = contrast["call_reduction"]["ci95_pct"][0] > 0
        time_supported = contrast["request_latency_s_reduction"]["ci95_pct"][0] > 0
        result.update(status="passed", efficacy_interpretation_allowed=True, endpoint=endpoint, evidence=evidence,
            conditional_claims=dict(call_improvement_supported_on_this_panel=calls_supported,
                                    serving_time_gain_supported_on_this_panel=calls_supported and time_supported))
        (destination / "PAIRED_RECONSTRUCTION.json").write_text(json.dumps(paired, indent=2, allow_nan=False) + "\n")
    except (KeyError, ValueError, TypeError, IndexError, OSError, ZeroDivisionError) as error:
        result["errors"].append(type(error).__name__ + ": " + str(error))
    result.update(generated_utc=datetime.now(timezone.utc).isoformat(), wall_s=time.perf_counter() - started,
        runtime=dict(python=platform.python_version(), numpy=np.__version__),
        scope="One fitted seed, reused panel, conditional request uncertainty; no async/training-inclusive speedup claim.")
    (destination / "ENDPOINT_RECALCULATION.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (destination / "INPUT_IDENTITIES.json").write_text(json.dumps(dict(root=str(reader.root), files=reader.files,
        auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root")
    parser.add_argument("--output", required=True)
    parser.add_argument("--analysis-recovery", help="Explicit hash-bound receipt for the preserved completed-work reporting failure")
    options = parser.parse_args()
    result = run(options.run_root, options.output, analysis_recovery=options.analysis_recovery)
    print(json.dumps({key: result[key] for key in ("status", "efficacy_interpretation_allowed", "errors")}))
    raise SystemExit(0 if result["status"] == "passed" else 1)
