"""Authenticate and reduce PR49's fixed event-seven collection population.

This module reads JSON and writes compact count tables. It never loads a model,
feature tensor, optimizer or serving backend. Collection latency is not analyzed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np

ARMS = ("N", "E", "S")
CONTRASTS = (("E", "N"), ("S", "N"), ("E", "S"))
SOURCE_COMMIT = "db6e1221d8d1b234d9d07e246025d1afd5dc30d5"


def identity(data):
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


class Checks:
    """Accumulate discrepancies, without publishing private token values."""
    def __init__(self):
        self.count = 0
        self.errors = []

    def require(self, condition, label):
        self.count += 1
        if not condition:
            self.errors.append(label)
        return bool(condition)


class Evidence:
    """Resolve only declared mirrors and authenticate the small JSON actually read."""
    def __init__(self, roots, checks):
        self.checks = checks
        self.roots = {}
        self.consumed = {}
        for name, root in roots.items():
            root = Path(root).resolve()
            manifest_path = root / "DELIVERY_MANIFEST.json"
            try:
                raw = manifest_path.read_bytes()
                manifest = json.loads(raw)
                self.consumed[str(manifest_path)] = dict(identity(raw), role="delivery_manifest")
            except (OSError, ValueError) as error:
                checks.require(False, f"manifest_unavailable:{name}:{type(error).__name__}")
                manifest = {}
            self.roots[name] = (root, manifest)

    def read(self, root_name, relative):
        root, manifest = self.roots[root_name]
        relative = str(relative)
        path = root / relative
        self.checks.require(relative in manifest, f"manifest_missing:{root_name}:{relative}")
        try:
            payload = path.read_bytes()
        except OSError as error:
            self.checks.require(False, f"unavailable:{root_name}:{relative}:{type(error).__name__}")
            return None
        actual = identity(payload)
        self.consumed[str(path)] = dict(actual, root=root_name, relative_path=relative)
        self.checks.require(actual == manifest.get(relative), f"identity_mismatch:{root_name}:{relative}")
        try:
            return json.loads(payload)
        except (ValueError, UnicodeError):
            self.checks.require(False, f"invalid_json:{root_name}:{relative}")
            return None


def valid_tokens(value, *, nonempty=True):
    return (isinstance(value, list) and (bool(value) or not nonempty)
            and all(type(token) is int and token >= 0 for token in value))


def validate_record(record, request, arm, checkpoint, persisted, checks):
    """Validate native evidence and reconstruct V/P/A/M from every block."""
    tag = f"{arm}:{request['request_id']}"
    before = len(checks.errors)
    check = lambda condition, field: checks.require(condition, f"{tag}:{field}")
    prompt, response = record["prompt_ids"], record["response_ids"]
    if not check(valid_tokens(prompt) and valid_tokens(response), "integer_token_arrays"):
        return None
    meta, capture = record["metadata"], record["metadata"]["capture"]
    trace = meta["trace"]
    stop = request["stop_token_ids"]
    check(valid_tokens(stop), "stop_integer_array")
    check(prompt == request["prompt_ids"], "exact_sealed_prompt")
    check(record["record_id"] == request["request_id"], "request_id")
    check(type(record["request_index"]) is int and record["request_index"] == request["request_index"], "request_index")
    check(record["session_id"] == f"persistent_{arm}_event8", "session")
    check(record["checkpoint_id"] == checkpoint, "event7_checkpoint")
    check(record["purpose"] == "serving", "purpose")
    check(record["stop_token_ids"] == stop, "stop_tokens")
    check(type(record["max_new_tokens"]) is int and record["max_new_tokens"] == 2048, "output_cap")
    check(type(meta["block_size"]) is int and meta["block_size"] == 16, "block_size")
    check(type(meta["context_limit"]) is int and meta["context_limit"] == 4096, "context_limit")
    check(len(response) <= 2048 and len(prompt) + 2048 <= 4096, "length_budget")
    check(not any(t in stop for t in response[:-1]), "no_early_stop")
    eos = response[-1] in stop
    check(record["finish_reason"] == ("eos" if eos else "length"), "termination")
    check(eos or len(response) == 2048, "length_finish_at_cap")
    if "cap_hit" in record:
        check(record["cap_hit"] is (not eos), "cap_flag")
    check(meta["feedback_capture_in_timed_request"] is True, "capture_mode")
    check(meta["trace_instrumentation_in_timed_request"] is True, "trace_mode")
    check(meta["runtime"] == "pinned_SGLang_FA3", "runtime")
    check(meta["cache_policy"] == "idle_full_flush_before_every_request_and_publication", "cache_policy")
    check(capture["collect_features"] is True, "feature_capture")
    check(capture["native_capture_validated"] is True and capture["status"] == "passed", "capture_validity")
    check(capture["label_binding_scope"] == "native_prefill_and_committed_verification", "native_labels")
    check(capture["final_feature_logits_used_for_labels"] is False and capture["final_emitted_token_feature_present"] is False, "terminal_feature_convention")
    check(capture["prompt_ids"] == prompt and capture["response_ids"] == response, "capture_token_binding")
    check(capture["target_argmax_ids"] == response, "target_argmax_labels")
    check(capture["covered_token_ids"] == (prompt + response)[:-1], "coverage")
    check(type(capture["covered_rows"]) is int and capture["covered_rows"] == len(prompt) + len(response) - 1, "coverage_rows")
    check(type(capture["prefill_chunks"]) is int and capture["prefill_chunks"] >= 1, "prefill_chunks")
    check(capture["block_size"] == 16, "capture_block_size")
    check(capture["blocks"] == trace, "embedded_trace_binding")
    check(persisted == capture, "independent_json_capture_equality")
    if not check(isinstance(trace, list), "trace_list"):
        return None

    progress = native = matching = clipped = 0
    for number, block in enumerate(trace):
        label = lambda value, field: check(value, f"block_{number}:{field}")
        draft, target = block["draft_tokens"], block["target_argmax"]
        width, committed = block["verify_size"], block["committed_progress"]
        fields = ("output_start", "anchor_token", "raw_accepted_draft_tokens_before_stop",
                  "accepted_draft_tokens", "native_accepted_draft_tokens")
        shape_ok = label(type(width) is int and type(committed) is int
                         and all(type(block[k]) is int for k in fields)
                         and valid_tokens(draft, nonempty=False) and valid_tokens(target)
                         and valid_tokens(block["committed_tokens"])
                         and valid_tokens(block["positions"])
                         and 1 <= width <= 16 and width == len(draft) + 1 == len(target)
                         and 1 <= committed <= width, "integer_geometry")
        if not shape_ok:
            continue
        accepted = next((i for i, value in enumerate(draft) if value != target[i]), len(draft))
        label(committed <= accepted + 1, "prefix_commit_bound")
        label(block["committed_tokens"] == (draft[:accepted] + [target[accepted]])[:committed]
              == target[:committed], "posterior_commit")
        label(block["output_start"] == progress, "absolute_output_start")
        label(progress < len(response) and block["anchor_token"] == response[progress], "seed")
        label(block["committed_tokens"] == response[progress + 1:progress + committed + 1], "canonical_commit")
        label(block["positions"] == list(range(len(prompt) + progress, len(prompt) + progress + width)), "absolute_positions")
        label(block["raw_accepted_draft_tokens_before_stop"] == accepted, "raw_acceptance")
        label(block["accepted_draft_tokens"] == min(accepted, committed), "matching_acceptance")
        label(block["native_accepted_draft_tokens"] == committed - 1, "native_acceptance")
        label(block["stop_clipped"] is (committed < accepted + 1), "stop_clipping")
        label(block["finished"] is (progress + committed == len(response) - 1), "finished")
        progress += committed
        native += block["native_accepted_draft_tokens"]
        matching += block["accepted_draft_tokens"]
        clipped += block["stop_clipped"] is True
    calls = len(trace)
    check(progress == len(response) - 1, "whole_response_progress")
    check(native == progress - calls, "native_identity")
    for field, value in [("verification_calls", calls), ("accepted_draft_tokens", matching), ("committed_progress", progress)]:
        check(type(meta[field]) is int and meta[field] == value, "stored_" + field)
    if "native_accepted_draft_tokens" in meta:
        check(type(meta["native_accepted_draft_tokens"]) is int and meta["native_accepted_draft_tokens"] == native, "stored_native_acceptance")
    check(meta["committed_progress_per_call"] == (progress / calls if calls else None), "stored_progress_ratio")
    check(meta["calls_per_1000_output_tokens"] == 1000 * calls / len(response), "stored_call_ratio")
    return {"source_id": request["source_id"], "request_id": request["request_id"],
            "native_source_id": request.get("native_source_id"), "request_index": request["request_index"],
            "arm": arm, "checkpoint_id": checkpoint, "valid": len(checks.errors) == before,
            "output_tokens": len(response), "finish_reason": record["finish_reason"],
            "cap_hit": not eos, "zero_calls": calls == 0, "V": calls, "P": progress,
            "A": native, "M": matching, "stop_clipped_blocks": clipped,
            "native_acceptance": native / calls if calls else None,
            "progress_per_call": progress / calls if calls else None}


def population_errors(requests, record_ids, expected_count=128):
    ids = [r["request_id"] for r in requests]
    sources = [r["source_id"] for r in requests]
    errors = []
    if len(ids) != expected_count or len(set(ids)) != expected_count:
        errors.append("sealed_request_population")
    if len(set(sources)) != expected_count:
        errors.append("sealed_source_duplicates")
    prompts = [tuple(r["prompt_ids"]) for r in requests if "prompt_ids" in r]
    if prompts and len(set(prompts)) != expected_count:
        errors.append("sealed_prompt_duplicates")
    for arm in ARMS:
        found = record_ids[arm]
        if len(found) != expected_count or len(set(found)) != expected_count or set(found) != set(ids):
            errors.append(f"{arm}:missing_duplicate_or_extra_records")
    return errors


def pairing_errors(records):
    reference = records["N"]
    return [f"{arm}:paired_{key}" for arm in ("E", "S")
            for key in ("prompt_ids", "response_ids", "finish_reason")
            if records[arm][key] != reference[key]]


def interval(values):
    undefined = int(np.count_nonzero(~np.isfinite(values)))
    if undefined:
        return {"interval": None, "undefined_draws": undefined,
                "reason": "A required pooled denominator is zero; no draws were removed or replaced."}
    return {"interval": np.percentile(values, [2.5, 97.5], method="linear").tolist(), "undefined_draws": 0}


def pooled_statistics(rows, draws_count=10000, seed=1708):
    by_arm = {a: sorted([r for r in rows if r["arm"] == a], key=lambda r: (r["source_id"], r["request_id"])) for a in ARMS}
    ordered = [[r["source_id"], r["request_id"]] for r in by_arm["N"]]
    assert all([[r["source_id"], r["request_id"]] for r in by_arm[a]] == ordered for a in ARMS)
    n = len(ordered)
    draws = np.random.Generator(np.random.PCG64(seed)).integers(0, n, size=(draws_count, n), dtype=np.int64)
    sums, totals = {}, {}
    for arm in ARMS:
        current = by_arm[arm]
        counts = np.array([[r[k] for k in ("V", "A", "P", "M")] for r in current], dtype=np.int64)
        sums[arm] = counts[draws].sum(axis=1)
        v, a, p, m = map(int, counts.sum(axis=0))
        totals[arm] = {"requests": n, "output_tokens": sum(r["output_tokens"] for r in current),
                       "eos": sum(r["finish_reason"] == "eos" for r in current),
                       "cap_hits": sum(r["cap_hit"] for r in current), "zero_calls": sum(r["zero_calls"] for r in current),
                       "V": v, "A": a, "P": p, "M": m,
                       "native_acceptance": a / v if v else None, "progress_per_call": p / v if v else None}
    comparisons = {}
    for x, y in CONTRASTS:
        tx, ty = totals[x], totals[y]
        lx, ly = tx["native_acceptance"], ty["native_acceptance"]
        differences = np.array([ry["V"] - rx["V"] for rx, ry in zip(by_arm[x], by_arm[y])])
        with np.errstate(divide="ignore", invalid="ignore"):
            bx = sums[x][:, 1] / sums[x][:, 0]
            by = sums[y][:, 1] / sums[y][:, 0]
            boot = {"absolute_acceptance_change": bx - by,
                    "relative_acceptance_change_pct": 100 * (bx / by - 1),
                    "call_reduction_pct": 100 * (sums[y][:, 0] - sums[x][:, 0]) / sums[y][:, 0]}
        comparisons[f"{x}_vs_{y}"] = {
            "absolute_acceptance_change": lx - ly if lx is not None and ly is not None else None,
            "relative_acceptance_change_pct": 100 * (lx / ly - 1) if lx is not None and ly else None,
            "call_reduction_pct": 100 * (ty["V"] - tx["V"]) / ty["V"] if ty["V"] else None,
            "net_calls_saved": int(differences.sum()), "gross_saved": int(differences.clip(min=0).sum()),
            "gross_added": int((-differences).clip(min=0).sum()),
            "fewer_calls": int((differences > 0).sum()), "more_calls": int((differences < 0).sum()),
            "tied_calls": int((differences == 0).sum()), "intervals": {key: interval(value) for key, value in boot.items()}}
    ordered_bytes = json.dumps(ordered, separators=(",", ":"), ensure_ascii=False).encode()
    return {"totals": totals, "comparisons": comparisons,
            "bootstrap": {"seed": seed, "generator": "NumPy Generator(PCG64)", "draws": draws_count,
                          "requests_per_draw": n, "ordered_source_list_sha256": hashlib.sha256(ordered_bytes).hexdigest(),
                          "source_list_encoding": "UTF-8 compact JSON [[source_id,request_id],...] sorted by those strings",
                          "draw_matrix_sha256": hashlib.sha256(draws.astype("<i8", copy=False).tobytes()).hexdigest(),
                          "draw_matrix_encoding": "row-major little-endian int64", "numpy_version": np.__version__,
                          "percentile_method": "linear", "percentiles": [2.5, 97.5], "shared_draws": True}}


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def reduce(args):
    started = time.perf_counter()
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    checks = Checks()
    checks.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CUDA_visibility_must_be_empty")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        checks.require(os.environ.get(key) == "1", f"{key}_must_be_1")
    evidence = Evidence({"production": args.delivery, "qualification": args.qualification_delivery}, checks)
    plan = evidence.read("production", "sealed/PLAN.json") or {}
    checks.require(plan.get("execution_commit") == SOURCE_COMMIT, "collection_execution_commit")
    requests = []
    if "event8_inputs" in plan:
        request_file = Path(plan["event8_inputs"]).relative_to(Path(plan["event8_inputs"]).parents[1])
        population = evidence.read("qualification", request_file) or {}
        requests = population.get("requests", [])
    else:
        checks.require(False, "sealed_population_unavailable")
    source_order = {r["request_id"]: i for i, r in enumerate(requests)}
    roots = Path(args.delivery)
    record_ids = {a: [f.stem for f in (roots / f"tasks/{a}_collect/records").glob("*.json")] for a in ARMS}
    for error in population_errors(requests, record_ids):
        checks.require(False, error)
    checkpoints = {}
    for arm in ARMS:
        result = evidence.read("production", f"tasks/{arm}_collect/RESULT.json") or {}
        checkpoints[arm] = result.get("own_checkpoint")
        checks.require(result.get("status") == "completed" and result.get("generated_responses") == 128, f"{arm}:collection_complete")
    rows, paired = [], []
    for request in requests:
        saved, current = {}, {}
        for arm in ARMS:
            relative = f"tasks/{arm}_collect/records/{request['request_id']}.json"
            record_errors_before = len(checks.errors)
            record = evidence.read("production", relative)
            if record is None:
                rows.append(dict(source_id=request["source_id"], request_id=request["request_id"], arm=arm, valid=False, unavailable=True))
                continue
            try:
                native_relative = Path(record["metadata"]["native_trace_path"]).relative_to(plan["root"])
                checks.require(native_relative == Path(f"tasks/{arm}_collect/native/request_{source_order[request['request_id']] + 1:06d}.json"), f"{arm}:{request['request_id']}:sealed_collection_order")
                persisted = evidence.read("production", native_relative)
                count = validate_record(record, request, arm, checkpoints[arm], persisted, checks)
                if count is not None:
                    count["valid"] = count["valid"] and len(checks.errors) == record_errors_before
                    rows.append(count)
                    current[arm] = count
                else:
                    rows.append(dict(source_id=request["source_id"], request_id=request["request_id"], arm=arm, valid=False))
                saved[arm] = record
            except (KeyError, TypeError, ValueError, IndexError) as error:
                checks.require(False, f"{arm}:{request['request_id']}:malformed:{type(error).__name__}:{str(error)[:100]}")
                if arm not in current:
                    rows.append(dict(source_id=request["source_id"], request_id=request["request_id"], arm=arm, valid=False))
        pair = {"source_id": request["source_id"], "request_id": request["request_id"], "request_index": request["request_index"]}
        pair_errors = pairing_errors(saved) if set(saved) == set(ARMS) else ["incomplete_three_arm_pair"]
        for error in pair_errors:
            checks.require(False, f"{request['request_id']}:{error}")
        pair["valid"] = not pair_errors and len(current) == 3 and all(r["valid"] for r in current.values())
        for arm, row in current.items():
            for key in ("V", "A", "P", "M", "output_tokens", "cap_hit", "zero_calls"):
                pair[f"{arm}_{key}"] = row[key]
        for x, y in CONTRASTS:
            pair[f"{x}_vs_{y}_calls_saved"] = current[y]["V"] - current[x]["V"] if x in current and y in current else None
        paired.append(pair)
    # Preserve identifiable extra/unbound files without inventing sealed identities.
    for arm in ARMS:
        for request_id in sorted(set(record_ids[arm]) - set(source_order)):
            unbound = evidence.read("production", f"tasks/{arm}_collect/records/{request_id}.json") or {}
            rows.append(dict(source_id=None, request_id=unbound.get("record_id", request_id),
                             arm=arm, valid=False, unbound=True))
    # A missing provenance input is an incomplete analysis, not a lost failure log.
    provenance_identity = None
    try:
        provenance_bytes = Path(args.provenance).read_bytes()
        provenance_identity = dict(identity(provenance_bytes), path=str(Path(args.provenance).resolve()))
        provenance = json.loads(provenance_bytes)
    except (OSError, ValueError) as error:
        checks.require(False, f"provenance_unavailable:{type(error).__name__}")
        provenance = {}
    checks.require(provenance.get("status") == "passed", "independent_provenance")
    checks.require(provenance.get("source_commit") == SOURCE_COMMIT, "provenance_source_commit")
    for arm in ARMS:
        expected_export = provenance.get("checkpoints", {}).get(arm, {}).get("export")
        checks.require(expected_export is not None and expected_export == checkpoints[arm], f"{arm}:provenance_export_binding")
    for item in provenance.get("consumed_inputs", []):
        try:
            payload = Path(item["local_path"]).read_bytes()
            checks.require(identity(payload) == item["identity"], "provenance_input:" + item["local_path"])
        except (OSError, KeyError) as error:
            checks.require(False, f"provenance_input_unavailable:{item.get('local_path')}:{type(error).__name__}")
    checks.require(len(rows) == 384 and len(paired) == 128, "complete_count_tables")
    checks.require(all(row["valid"] for row in paired), "all_pairs_valid")
    valid = not checks.errors
    summary = {"status": "completed" if valid else "incomplete", "scope": "RETROSPECTIVE_CAPTURE_ENABLED_COLLECTION_COUNTS",
               "collection_execution_commit": SOURCE_COMMIT, "analysis_commit": args.analysis_commit,
               "new_model_work": 0, "latency_excluded": True, "request_count": len(requests),
               "contrasts_in_declared_order": [f"{a}_vs_{b}" for a,b in CONTRASTS]}
    if valid:
        summary.update(pooled_statistics(rows))
    else:
        summary["reason"] = "Evidence incomplete or invalid; no selected-subset primary comparison."
    write_csv(destination / "PER_REQUEST.csv", rows)
    write_csv(destination / "PAIRED_REQUESTS.csv", paired)
    write_json(destination / "VALIDATION.json", {"status": "passed" if valid else "incomplete", "checks": checks.count,
               "expected_records": 384, "observed_record_files": {a:len(record_ids[a]) for a in ARMS},
               "identifiable_rows": len(rows), "complete_valid_pairs": sum(p["valid"] for p in paired), "errors": checks.errors})
    write_json(destination / "SOURCE_MANIFEST.json", {"collection_commit": SOURCE_COMMIT, "analysis_commit": args.analysis_commit,
               "mirrors": {k:str(v[0]) for k,v in evidence.roots.items()}, "recorded_root": plan.get("root"),
               "consumed_inputs": evidence.consumed, "provenance_receipt": provenance_identity,
               "checkpoint_identity_policy": "Inherited authenticated receipts; no model tensor reads"})
    write_json(destination / "SUMMARY.json", summary)
    usage = {"wall_s": time.perf_counter() - started, "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
             "threads_per_process": 1, "observed_threads": int(next(line.split()[1] for line in Path("/proc/self/status").read_text().splitlines() if line.startswith("Threads:"))),
             "forbidden_model_modules_loaded": [name for name in sys.modules if name == "torch" or name.startswith("torch.") or name == "transformers"],
             "process_cpu_s": time.process_time(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
             "new_gpu_work": 0, "model_loads": 0, "tensor_deserializations": 0, "training_steps": 0, "generations": 0,
             "data_bytes_written_before_resource_receipt": sum(f.stat().st_size for f in destination.rglob("*") if f.is_file()),
             "bytes_limit": 1000000000}
    write_json(destination / "RESOURCE_USAGE.json", usage)
    return summary, usage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delivery", required=True)
    parser.add_argument("--qualification-delivery", required=True)
    parser.add_argument("--provenance", required=True)
    parser.add_argument("--analysis-commit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summary, usage = reduce(args)
    print(json.dumps({"status":summary["status"], "wall_s":usage["wall_s"], "new_gpu_work":0}))
    return 0 if summary["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
