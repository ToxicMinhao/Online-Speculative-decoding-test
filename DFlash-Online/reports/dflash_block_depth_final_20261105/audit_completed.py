"""Separate CPU reconstruction of the frozen-depth campaign evidence.

Only stdlib and NumPy are used. No experiment imports, model execution, remote
access or result-selected changes are made. Authorship overlaps implementation;
this is an independent calculation, not a blinded review/GPU replication.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import struct

import numpy as np


COMMIT = "0f704167b268672d980fabe704898abf9b134c39"
ARMS = ("FIVE", "THREE", "ADAPTIVE")
RESTORE = ("request_row_ids", "request_mapping", "allocator", "native_graph_inputs", "attention_metadata", "cache_bytes")
GRAPH_LAYERS = {"static_three": [1, 1, 1, 0, 0], "prefix": [1, 1, 1, 0, 0],
                "shallow": [0, 0, 0, 0, 0], "deep": [0, 0, 0, 1, 1]}


def file_identity(path):
    payload = Path(path).read_bytes()
    return dict(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())


def finite(value, positive=False):
    return type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else value >= 0)


def schedule(requests):
    result = []
    for repeat in range(2):
        for block in range(24):
            rotation = block % 3
            arms = ARMS[rotation:] + ARMS[:rotation]
            if repeat:
                arms = arms[::-1]
            for position, arm in enumerate(arms):
                for ordinal in range(block * 16, block * 16 + 16):
                    result.append(dict(request_id=requests[ordinal]["request_id"], request_ordinal=ordinal,
                                       repeat=repeat, block=block, arm_position=position, arm=arm))
    return result


def progress(proposals, suffix, eos):
    accepted = 0
    for proposal, token in zip(proposals, suffix):
        if token != proposal:
            break
        accepted += 1
        if token in eos:
            return accepted
    return min(accepted + 1, len(suffix))


def selected_starts(request_id, count):
    return sorted(sorted(range(count), key=lambda i: hashlib.sha256(
        f"vfd_depth35_gsm8k_v1|GATE|{request_id}|{i}".encode()).digest())[:64])


def execution_files(root):
    """Native sidecars have their own schema and are not extra executions."""
    return sorted((Path(root) / "final/raw").glob("[0-9][0-9][0-9][0-9].json"))


def comparisons(means):
    times = np.asarray(means, dtype=np.float64)
    draws = np.random.default_rng(20261105).integers(0, len(times), size=(10000, len(times)))
    totals = np.array([math.fsum(times[:, i]) for i in range(3)])
    samples = np.stack([times[:, i][draws].sum(1) for i in range(3)], axis=1)
    result = {}
    for key, arm, baseline in (("ADAPTIVE_vs_FIVE", 2, 0), ("ADAPTIVE_vs_THREE", 2, 1), ("THREE_vs_FIVE", 1, 0)):
        reductions = (1 - samples[:, arm] / samples[:, baseline]) * 100
        result[key] = dict(point_pct=float((1 - totals[arm] / totals[baseline]) * 100),
                           ci95_pct=np.quantile(reductions, [.025, .975], method="linear").tolist())
    return result


class Audit:
    def __init__(self, root):
        self.root = Path(root)
        self.checks, self.errors, self.sources = 0, [], {}

    def check(self, condition, label):
        self.checks += 1
        if not condition:
            self.errors.append(label)

    def load(self, name, optional=False, lines=False):
        path = Path(name)
        path = path if path.is_absolute() else self.root / path
        if optional and not path.exists():
            return None
        payload = path.read_bytes()
        self.sources[str(path)] = file_identity(path)
        return [json.loads(line) for line in payload.splitlines() if line] if lines else json.loads(payload)

    def match_file(self, name, expected, label):
        path = self.root / name
        self.check(path.is_file(), label + ":exists")
        if path.is_file():
            actual = file_identity(path)
            self.sources[str(path)] = actual
            self.check(actual == expected, label + ":identity")


def audit_ledger(audit, status, plan):
    limits = dict(readout_generations=368, readout_reconstructions=16, teacher_preparations=384,
                  gate_generations=128, calibration_requests=128,
                  canonical_generations=plan["budget"]["untimed_reference_generations_max"], final_generations=2304)
    starts, done = {key: set() for key in limits}, {key: set() for key in limits}
    for i, row in enumerate(audit.load("WORK.jsonl", optional=True, lines=True) or []):
        key, index, phase = row["kind"], row["ordinal"], row["phase"]
        audit.check(key in limits and type(index) is int and 1 <= index <= limits.get(key, 0), f"ledger_{i}:budget")
        if key not in limits:
            continue
        if phase == "started":
            audit.check(index == len(starts[key]) + 1 and index not in starts[key], f"ledger_{i}:unique_start")
            starts[key].add(index)
        elif phase == "completed":
            audit.check(index in starts[key] and index not in done[key], f"ledger_{i}:unique_completion")
            done[key].add(index)
        else:
            audit.check(False, f"ledger_{i}:phase")
    work = dict(attempts={k: len(v) for k, v in starts.items()}, completed={k: len(v) for k, v in done.items()},
                unknown={k: len(starts[k] - done[k]) for k in limits})
    audit.check(status.get("work") == work, "reported_work_ledger")
    if status["status"] == "completed":
        audit.check(work["attempts"] == limits and work["completed"] == limits and not any(work["unknown"].values()), "complete_fixed_budget")
    return dict(**work, planned=limits, unattempted={k: limits[k] - len(starts[k]) for k in limits})


def audit_training(audit, parts, complete):
    groups, seen = [], set()
    for group in range(1, 4):
        prefix = f"readout/group_{group:03d}/trainer"
        result = audit.load(prefix + "/RESULT.json", optional=True)
        if result is None:
            audit.check(not complete, f"group_{group}:missing_completed_group")
            continue
        label = f"group_{group}"
        audit.check(result["status"] == "completed" and result["optimizer_steps"] == 128 and result["response_presentations"] == 256, label + ":fixed_work")
        audit.check(result["start_step"] == (group - 1) * 128 and result["end_step"] == group * 128, label + ":continuous_steps")
        audit.check(result["objective"] == "osd_first_error" and result["reduction"] == "response_mean" and result["passes"] == 2, label + ":objective")
        ids = {r["request_id"] for r in parts["READOUT"]["requests"][(group - 1) * 128:group * 128]}
        audit.check(not seen.intersection(ids), label + ":disjoint_group")
        seen |= ids
        records = result["records"]
        audit.check(len(records) == 256, label + ":presentation_count")
        for pass_index in range(2):
            actual = [r["record_id"] for r in records if r["pass"] == pass_index]
            audit.check(len(actual) == 128 and set(actual) == ids, label + f":pass_{pass_index}_membership")
        for i, row in enumerate(records):
            prefix = label + f":presentation_{i}"
            audit.check(row["optimizer_stepped"] == (i % 2 == 1), prefix + ":accumulation")
            audit.check(row["optimizer_step_after"] == (group - 1) * 128 + (i + 1) // 2, prefix + ":counter")
            audit.check(row["learning_rate"] == (3e-4 if i % 2 else None), prefix + ":lr")
            observation = row["response_observation"]
            audit.check(observation["record_id"] == row["record_id"] and observation["sampled_anchors"] == row["sampled_anchors"], prefix + ":anchor_binding")
            audit.check(math.isfinite(observation["loss"]) and all(math.isfinite(v) for v in observation["sum_metrics"].values()), prefix + ":finite_loss")
            for draw in row["sampled_anchors"]:
                audit.check(len(draw["positions"]) == len(draw["keep_mask"]) == 1, prefix + ":batch_one")
                audit.check(len(draw["positions"][0]) <= 64 and len(draw["positions"][0]) == len(draw["keep_mask"][0]), prefix + ":anchor_limit")
        steps = result["readout_steps"]
        audit.check([r["global_step"] for r in steps] == list(range((group - 1) * 128 + 1, group * 128 + 1)), label + ":step_observations")
        audit.check(all(finite(r["gradient_norm"]) for r in steps), label + ":finite_gradients")
        audit.check(any(sum(r["changed_coordinates"].values()) for r in steps), label + ":nonzero_updates")
        module_work = result["actual_module_work"]
        for name in ("frozen_prefix_forwards", "sidecar_forwards"):
            audit.check(module_work.get(name) == 256, label + ":" + name)
        for layer in range(1, 6):
            audit.check(module_work.get(f"backbone_layer_{layer}_forwards", 0) == (256 if layer <= 3 else 0), label + f":layer_{layer}")
        audit.check(result["frozen_parameter_versions_unchanged"] is True, label + ":frozen_backbone")
        boundary = audit.load(f"readout/group_{group:03d}/trainer/BOUNDARY_CHECK.json")
        audit.check(boundary["status"] == "passed" and not any(boundary["differences"].values()), label + ":decoded_boundary")
        audit.check(boundary["global_step"] == group * 128 and boundary["presentations"] == group * 256, label + ":boundary_counters")
        if group > 1:
            resume = audit.load(f"readout/group_{group:03d}/trainer/RESUME_CHECK.json")
            audit.check(resume["status"] == "passed" and all(resume["readout_exact"].values()) and resume["optimizer_exact"] and resume["rng_exact"], label + ":own_state_resume")
            audit.check(resume["global_step"] == (group - 1) * 128 and resume["presentations"] == (group - 1) * 256, label + ":resume_counters")
        audit.check(result["startup_work"]["extra_optimizer_steps"] == 0 and result["startup_work"]["extra_response_forwards"] == 0, label + ":no_extra_startup")
        groups.append(dict(group=group, steps=128, presentations=256, training_s=result["costs"]["training_s"], actual_module_work=module_work))
    frozen = audit.load("readout/group_003/trainer/FROZEN_READOUT.json", optional=True)
    if frozen:
        audit.check(frozen["status"] == "passed" and frozen["global_step"] == 384 and frozen["presentations"] == 768 and frozen["backbone_optimizer_steps"] == 0 and frozen["readout_frozen_before_gate"], "readout_frozen_before_gate")
    elif complete:
        audit.check(False, "missing_readout_freeze")
    return groups


def audit_gate(audit, parts, plan, complete):
    rows, behaviors = [], []
    for ordinal, request in enumerate(parts["GATE"]["requests"]):
        request_id = request["request_id"]
        receipt = audit.load(f"gate/calibration/{request_id}.json", optional=True)
        if receipt is None:
            audit.check(not complete, request_id + ":missing_calibration")
            continue
        behavior = audit.load(f"gate/responses/{request_id}.json")
        meta = behavior["metadata"]
        route = "FIVE" if ordinal % 2 == 0 else "THREE"
        audit.check(behavior["arm"] == route and behavior["prompt_ids"] == request["prompt_ids"], request_id + ":behavior_binding")
        audit.check(receipt["status"] == "passed" and receipt["restoration_status"] == "passed", request_id + ":replay_validity")
        trace, tokens = meta["trace"], behavior["response_ids"]
        audit.check(meta["observed_rounds"] == meta["verification_calls"] == len(trace), request_id + ":behavior_round_count")
        audit.check(meta["committed_progress"] == len(tokens) - 1 == sum(b["committed_progress"] for b in trace), request_id + ":prefill_excluded_progress")
        offset = 0
        for block in trace:
            audit.check(block["output_start"] == offset and block["positions"] == list(range(len(request["prompt_ids"]) + offset, len(request["prompt_ids"]) + offset + 16)), request_id + ":canonical_start")
            offset += block["committed_progress"]
        starts = selected_starts(request_id, len(trace))
        audit.check(receipt["sampled_round_indices"] == starts, request_id + ":sampled_starts")
        audit.check([r["round_index"] for r in receipt["rows"]] == starts, request_id + ":row_order")
        if trace:
            audit.check(all(receipt["restoration_checks"].get(k) is True for k in RESTORE), request_id + ":cache_restoration")
        for row in receipt["rows"]:
            block = trace[row["round_index"]]
            suffix = tokens[block["output_start"] + 1:]
            p3, p5 = (progress(row["proposals"][arm], suffix, request["stop_token_ids"]) for arm in ("THREE", "FIVE"))
            audit.check(row["p3"] == p3 and row["p5"] == p5, request_id + f":progress_{row['round_index']}")
            audit.check(row["valid"] and row["proposals"][route] == block["draft_tokens"] and (p5 if route == "FIVE" else p3) == block["committed_progress"], request_id + ":behavior_reconstruction")
            audit.check(row["observed_rounds"] == len(trace) and row["sampled_rounds"] == len(starts), request_id + ":expansion_weight")
            audit.check(len(row["features"]) == 97 and all(math.isfinite(v) for v in row["features"]), request_id + ":features")
        accounting = receipt["accounting"]
        audit.check(accounting["returned_draft_forwards"] == 2 * len(starts) and accounting["unknown_draft_forwards"] == accounting["unattempted_draft_forwards"] == 0, request_id + ":paired_work")
        rows.extend(receipt["rows"])
        behaviors.append(dict(request_id=request_id, **meta))
    audit.check(len(rows) <= 8192, "calibration_start_budget")
    gate = audit.load("gate/GATE.json", optional=True)
    if gate is not None:
        audit.check(len(behaviors) == 128, "gate_requires_all_behaviors")
        audit.check(audit.load("gate/ROWS.json") == rows, "flattened_calibration_rows")
        audit.check(audit.load("gate/BEHAVIORS.json") == behaviors, "flattened_behavior_rows")
        x = np.asarray([r["features"] for r in rows], dtype=np.float64)
        y = np.asarray([r["p5"] - r["p3"] for r in rows], dtype=np.float64)
        w = np.asarray([r["observed_rounds"] / r["sampled_rounds"] for r in rows])
        w /= w.sum()
        center = w @ x
        scale = np.maximum(np.sqrt(w @ ((x - center) ** 2)), 1e-6)
        z = (x - center) / scale
        intercept = float(w @ y)
        beta = np.linalg.solve(z.T @ (w[:, None] * z) + .001 * np.eye(97), z.T @ (w * (y - intercept)))
        for name, value in (("feature_mean", center), ("feature_scale", scale), ("standardized_weight", beta),
                            ("folded_weight", beta / scale), ("folded_intercept", intercept - center @ (beta / scale))):
            audit.check(np.allclose(gate[name], value, rtol=1e-8, atol=1e-8), "independent_gate_fit:" + name)
        five = [r for r in behaviors if r["behavior"] == "FIVE"]
        kappa = 1000 * math.fsum(r["decode_round_wall_s"] for r in five) / sum(r["committed_progress"] for r in five)
        audit.check(math.isclose(gate["kappa_ms_per_token"], kappa, rel_tol=1e-12, abs_tol=1e-12), "independent_kappa")
        audit.check(gate["continuation_cost_ms"] == plan["continuation_cost_ms"], "fixed_pricing_threshold")
    return dict(behaviors=len(behaviors), paired_starts=len(rows), paired_proposal_forwards=2 * len(rows),
                negative_targets=sum(r["p5"] < r["p3"] for r in rows), fit_reconstructed=gate is not None)


def audit_endpoint(audit, parts, status):
    requests = parts["FINAL"]["requests"]
    planned = schedule(requests)
    saved = audit.load("FINAL_SCHEDULE.json", optional=True)
    if saved is not None:
        audit.check(saved == planned, "final_balanced_schedule")
    canonical = audit.load("CANONICALS.json", optional=True) or {}
    raw_paths = execution_files(audit.root)
    expected_names = [f"{i:04d}.json" for i in range(len(raw_paths))]
    audit.check([p.name for p in raw_paths] == expected_names, "contiguous_raw_serials")
    validity = audit.load("final/VALIDITY.jsonl", optional=True, lines=True) or []
    validity_by_serial = {r["serial"]: r for r in validity}
    audit.check(len(validity_by_serial) == len(validity), "unique_validity_serials")
    cells, measured = {}, []
    for serial, path in enumerate(raw_paths):
        slot = planned[serial]
        request = requests[slot["request_ordinal"]]
        raw = audit.load(path)
        label = f"final_{serial}"
        meta, output = raw["metadata"], raw["response_ids"]
        key = (raw["request_id"], raw["arm"], raw["repeat"])
        audit.check(key == tuple(slot[k] for k in ("request_id", "arm", "repeat")) and key not in cells, label + ":identity_order_unique")
        cells[key] = raw
        audit.check(raw["prompt_ids"] == request["prompt_ids"] and raw["stop_token_ids"] == request["stop_token_ids"] and raw["max_new_tokens"] == 2048, label + ":input_binding")
        reference = canonical[request["request_id"]]
        audit.check(reference["prompt_ids"] == request["prompt_ids"] and output == reference["response_ids"] and raw["finish_reason"] == reference["finish_reason"], label + ":canonical_parity")
        eos = set(request["stop_token_ids"])
        audit.check(bool(output) and len(output) <= 2048 and all(type(v) is int and v >= 0 for v in output) and not any(v in eos for v in output[:-1]), label + ":valid_tokens")
        audit.check(raw["finish_reason"] == ("eos" if output[-1] in eos else "length") and (output[-1] in eos or len(output) == 2048), label + ":termination")
        counts = [meta[k] for k in ("verification_calls", "accepted_draft_tokens", "committed_progress", "shallow_rounds", "deep_rounds")]
        calls, accepted, progress_count, shallow, deep = counts
        audit.check(all(type(v) is int and v >= 0 for v in counts) and progress_count == len(output) - 1 and 0 <= accepted <= min(progress_count, 15 * calls) and progress_count <= 16 * calls and shallow + deep == calls, label + ":counts")
        audit.check(accepted == progress_count - calls, label + ":native_accepted_convention")
        audit.check((raw["arm"] != "FIVE" or shallow == 0) and (raw["arm"] != "THREE" or deep == 0), label + ":static_route")
        for field in ("request_latency_s", "native_generate_s", "decode_round_wall_s", "cache_maintenance_s"):
            audit.check(finite(meta[field], field in ("request_latency_s", "native_generate_s")), label + ":" + field)
        audit.check(meta.get("feedback_capture_in_timed_request") is False and not meta.get("trace"), label + ":no_timed_capture")
        native = audit.load(path.with_suffix(".native.json"))
        audit.check(native["status"] == "passed" and native["output_ids"] == output and native["output_length"] == len(output), label + ":native_sidecar_outputs")
        audit.check(native["feedback_capture"] is False and native["trace_instrumentation"] is False, label + ":native_capture_disabled")
        for field in ("verification_calls", "accepted_draft_tokens", "committed_progress", "shallow_rounds", "deep_rounds", "decode_round_wall_s", "cache_maintenance_s"):
            audit.check(native[field] == meta[field], label + ":native_sidecar_" + field)
        audit.check(serial in validity_by_serial and validity_by_serial[serial]["valid"] and not validity_by_serial[serial]["errors"], label + ":saved_validity")
        measured.append(dict(serial=serial, **slot, output_tokens=len(output), **{k: meta[k] for k in ("request_latency_s", "native_generate_s", "decode_round_wall_s", "cache_maintenance_s", "verification_calls", "accepted_draft_tokens", "committed_progress", "shallow_rounds", "deep_rounds")}))
    per_request = []
    for request in requests:
        request_id = request["request_id"]
        if not all((request_id, arm, repeat) in cells for arm in ARMS for repeat in range(2)):
            continue
        item = dict(request_id=request_id)
        for arm in ARMS:
            pair = [cells[(request_id, arm, repeat)]["metadata"] for repeat in range(2)]
            for field in ("verification_calls", "accepted_draft_tokens", "committed_progress", "shallow_rounds", "deep_rounds"):
                audit.check(pair[0][field] == pair[1][field], request_id + ":" + arm + ":repeat_" + field)
                item[arm + "_" + field] = pair[0][field]
            for field in ("request_latency_s", "native_generate_s", "decode_round_wall_s", "cache_maintenance_s"):
                item[arm + "_" + field] = math.fsum(r[field] for r in pair) / 2
        per_request.append(item)
    result = dict(returned=len(raw_paths), planned=2304, paired_requests=len(per_request),
                  unknown=status["work"]["unknown"]["final_generations"],
                  unattempted=2304-status["work"]["attempts"]["final_generations"],
                  per_request=per_request, measurements=measured, comparison=None)
    if len(raw_paths) == 2304 and len(per_request) == 384 and not audit.errors:
        result["comparison"] = comparisons([[r[arm + "_request_latency_s"] for arm in ARMS] for r in per_request])
        result["totals"] = {arm: {field: math.fsum(r[arm + "_" + field] for r in per_request)
            for field in ("request_latency_s", "native_generate_s", "decode_round_wall_s", "cache_maintenance_s", "verification_calls", "accepted_draft_tokens", "committed_progress", "shallow_rounds", "deep_rounds")} for arm in ARMS}
        production = audit.load("report/ENDPOINT_ANALYSIS.json")
        audit.check(production["status"] == "complete", "production_endpoint_complete")
        for key, values in result["comparison"].items():
            audit.check(np.allclose(values["point_pct"], production["comparison"][key]["point_pct"], rtol=1e-11, atol=1e-11), key + ":point_reconstruction")
            audit.check(np.allclose(values["ci95_pct"], production["comparison"][key]["ci95_pct"], rtol=1e-11, atol=1e-11), key + ":interval_reconstruction")
        result["scientific_criterion_passed"] = all(result["comparison"][key]["ci95_pct"][0] > 0 for key in ("ADAPTIVE_vs_FIVE", "ADAPTIVE_vs_THREE"))
        result["practical_two_pct_vs_FIVE_point_reached"] = result["comparison"]["ADAPTIVE_vs_FIVE"]["point_pct"] >= 2
        audit.check(result["scientific_criterion_passed"] == production["scientific_criterion_passed"], "scientific_decision")
        audit.check(result["practical_two_pct_vs_FIVE_point_reached"] == production["practical_two_pct_vs_FIVE_point_reached"], "separate_practical_target")
    return result


def inspect(root, manifest_path):
    audit = Audit(root)
    plan, status = audit.load("PLAN.json"), audit.load("STATUS.json")
    manifest = audit.load(Path(manifest_path).resolve())
    audit.check(plan["code_commit"] == COMMIT, "execution_commit")
    audit.check(file_identity(manifest_path) == plan["input_identities"][plan["manifest_file"]], "sealed_manifest")
    complete = status["status"] == "completed"
    work = audit_ledger(audit, status, plan)
    training = audit_training(audit, manifest["manifests"], complete)
    gate = audit_gate(audit, manifest["manifests"], plan, complete)
    freeze = audit.load("FINAL_FREEZE.json", optional=True)
    if freeze:
        for key, path in (("sidecar", "readout/group_003/trainer/events/group_003/export/readout.safetensors"), ("gate", "gate/GATE.json"), ("canonicals", "CANONICALS.json")):
            audit.match_file(path, freeze[key], "frozen_" + key)
        sidecar_path = audit.root / "readout/group_003/trainer/events/group_003/export/readout.safetensors"
        with sidecar_path.open("rb") as stream:
            header = json.loads(stream.read(struct.unpack("<Q", stream.read(8))[0]))
        expected_shapes = {"norm.weight": [4096], "A": [32, 4096], "B": [4096, 32]}
        audit.check(set(header) - {"__metadata__"} == set(expected_shapes), "only_three_sidecar_tensors")
        for name, shape in expected_shapes.items():
            audit.check(header[name]["shape"] == shape and header[name]["dtype"] == "BF16", "sidecar_shape_dtype_" + name)
        audit.check(freeze["no_more_fitting"] and freeze["code_commit"] == COMMIT, "final_freeze_contract")
        graphs = freeze["routes"]["graph_setup"]
        audit.check({r["graph"] for r in graphs} == set(GRAPH_LAYERS), "graphs_built_before_final")
        for row in graphs:
            audit.check(row["captured_layer_calls"] == GRAPH_LAYERS[row["graph"]], "graph_layers_" + row["graph"])
    elif complete:
        audit.check(False, "missing_final_freeze")
    for name in ("READOUT_LOAD.json", "GATE_LOAD.json"):
        loaded = audit.load(name, optional=True)
        audit.check(bool(loaded) or not complete, name + ":required")
        if loaded:
            audit.check(loaded["status"] == "passed" and loaded["sidecar_frozen"] and loaded["graph_parameter_storage_preserved"], name + ":immutable_graph_storage")
    for name in ("WEIGHTS_BEFORE.json", "WEIGHTS_AFTER_READOUT.json", "WEIGHTS_AFTER.json"):
        receipt = audit.load(name, optional=True)
        audit.check(bool(receipt) or not complete, name + ":required")
        if receipt:
            audit.check(receipt["status"] == "passed" and receipt["all_equal"] is True, name + ":frozen_weights")
    for name in ("PREFLIGHT.json", "POSTFLIGHT.json"):
        receipt = audit.load(name, optional=True)
        audit.check(bool(receipt) or not complete, name + ":required")
        if receipt:
            audit.check(receipt["status"] == "passed" and receipt["code_commit"] == COMMIT, name + ":source_model_runtime_admission")
    endpoint = audit_endpoint(audit, manifest["manifests"], status)
    reconstruction = [audit.load(f"readout/group_001/reconstruction_{i:02d}.json", optional=True) for i in range(16)]
    reconstruction = [r for r in reconstruction if r]
    for row in reconstruction:
        audit.check(row["status"] == "passed" and row["new_generations"] == 0 and row["active_forward"] is None, "retained_feedback_reconstruction")
        audit.check(row["posterior_positions_checked"] == 16 * row["verification_forwards"], "reconstruction_all_posterior_positions")
    return dict(verification_status="ANALYZED", execution_commit=COMMIT, source_root=str(audit.root),
        checks=audit.checks, errors=audit.errors, claimed_complete=complete,
        complete_supported=complete and not audit.errors and endpoint["returned"] == 2304,
        work=work, training=training, gate=gate, endpoint=endpoint,
        reconstruction=dict(responses=len(reconstruction), prefill_forwards=sum(r["prefill_forwards"] for r in reconstruction),
            verification_forwards=sum(r["verification_forwards"] for r in reconstruction),
            posterior_positions_checked=sum(r["posterior_positions_checked"] for r in reconstruction)),
        source_identities=audit.sources,
        disclosure="Standalone NumPy/stdlib reconstruction; author also implemented gate/endpoint/data/plan. Not blinded and not an independent GPU repetition. Retired feature tensors and full GPU state are not re-executed.",
        interpretation="Complete-request timing is primary. Native decode subcounter is secondary. FINAL is fit-disjoint but historically exposed. Fixed checkpoint, one gate, one dataset; no online recalibration or training-inclusive speedup inference.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = inspect(args.root, args.manifest)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "AUDIT.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for key in ("per_request", "measurements"):
        rows = result["endpoint"][key]
        if rows:
            with (args.output / (key.upper() + ".csv")).open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    text = ["# Frozen-depth completed-record audit", "", "Verification Status: ANALYZED", "",
            f"Execution `{COMMIT}`; checks {result['checks']}; errors {len(result['errors'])}; complete supported {result['complete_supported']}.", "",
            json.dumps(result["endpoint"]["comparison"], indent=2), "", result["disclosure"], "", result["interpretation"]]
    (args.output / "AUDIT.md").write_text("\n".join(text) + "\n")
    print(json.dumps({k: result[k] for k in ("checks", "errors", "complete_supported")}, indent=2))
    if result["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
