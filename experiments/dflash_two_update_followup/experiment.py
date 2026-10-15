"""CPU-only protocol for the two requested, source-bound follow-up comparisons.

Imports load no model and launch nothing. Original saved contexts are immutable
benchmark inputs; the author-path arm builds its own generated history. Neither
timing copies nor earlier conditioning tokens become new support responses.
"""
from __future__ import annotations

import copy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import time


_spec = importlib.util.spec_from_file_location(
    "_followup_two_update_contract", Path(__file__).resolve().parents[1] / "dflash_two_update/experiment.py")
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
SCENARIOS, PHASES, REPEATS = _base.SCENARIOS, _base.PHASES, _base.REPEATS
INPUT_LIMIT, CONTEXT_LIMIT = _base.INPUT_LIMIT, _base.CONTEXT_LIMIT
OUTPUT_CAP, BLOCK_SIZE = _base.OUTPUT_CAP, _base.BLOCK_SIZE
digest, require, load_dataset = _base.digest, _base.require, _base.load_dataset
AUTHOR_ARM = "AdaFlash OSD training path under our session protocol"
AUTHOR_COMMIT = "507b7aba5fbccaa99a400f161e4f2794ba6a388c"
MODES = ("checkpoints-only", "author-path", "both")


def validate_episode(rows, *, allow_reserved=False):
    require(isinstance(rows, list) and len(rows) == 32, "Expected 32 ordered turns")
    require([r.get("turn") for r in rows] == list(range(1, 33)), "Expected 32 ordered turns")
    first = rows[0]
    replica, scenario = first.get("replica"), first.get("stream_type")
    admitted = (1, 2, 3, 4) if allow_reserved else (1, 2)
    require(type(replica) is int and replica in admitted and scenario in SCENARIOS,
            "Only R1/R2 development episodes are admitted for execution")
    episode = f"R{replica}_{scenario}"
    split = "development" if replica in (1, 2) else "confirmation"
    for row in rows:
        require(type(row["turn"]) is int and row.get("stream_id") == episode and
                row.get("replica") == replica and row.get("stream_type") == scenario and
                row.get("split") == split and row.get("phase") == PHASES[(row["turn"] - 1) // 8] and
                isinstance(row.get("question_id"), str) and bool(row["question_id"]) and
                isinstance(row.get("prompt"), str) and bool(row["prompt"]), "Episode metadata differs")
    return (replica - 1) * len(SCENARIOS) + SCENARIOS.index(scenario)


def plan(streams, mode="both"):
    """Describe work without dispatching it; R3/R4 source rows stay reserved."""
    require(mode in MODES, "Unknown follow-up mode")
    groups = {}
    for row in streams:
        groups.setdefault(row["stream_id"], []).append(row)
    for rows in groups.values():
        validate_episode(rows, allow_reserved=True)
    expected = [f"R{replica}_{scenario}" for replica in (1, 2) for scenario in SCENARIOS]
    require(all(name in groups for name in expected), "Exactly ten development episodes required")
    episodes = []
    for name in expected:
        rows = groups[name]
        episodes.append({"episode_id": name, "scenario": rows[0]["stream_type"],
                         "replica": rows[0]["replica"], "request_ids": [r["question_id"] for r in rows],
                         "checkpoint_comparison": {"turns": list(range(25, 33)), "methods": ["F", "C1", "C2"]},
                         "author_path": {"updates_after_turns": [16, 24],
                                         "support_turns": [list(range(9, 17)), list(range(17, 25))],
                                         "early_methods": ["F", "C1", "A1"],
                                         "final_methods": ["F", "C1", "C2", "A2"]}})
    author = mode in ("author-path", "both")
    comparison = mode == "checkpoints-only"
    return {"schema_version": 1, "status": "review_only", "gpu_launch_authorized": False,
            "mode": mode, "streams_sha256": digest(streams), "episodes": episodes,
            "author_arm": AUTHOR_ARM, "author_commit": AUTHOR_COMMIT,
            "paired_timing_repeats": REPEATS, "adaptation_events": 20 if author else 0,
            "optimizer_steps_per_event": 8 if author else 0, "optimizer_steps_total": 160 if author else 0,
            "response_presentations_total": 320 if author else 0,
            "canonical_request_executions": 320 if author else 0,
            "checkpoint_comparison_executions": 720 if comparison else 0,
            "author_path_score_executions": 1680 if author else 0,
            "total_score_executions": 1680 if author else 720,
            "notes": ["Metadata-only plan; no native backend, remote command or GPU launch.",
                      "Author-path final scoring includes the requested F/C1/C2 comparison contemporaneously.",
                      "Both mode reuses the author-path final F/C1/C2 matrix; it does not execute duplicate comparisons.",
                      "Timing repeats are measurements, never training data or independent prompts.",
                      "R3/R4 remain reserved. All scores replay original saved canonical inputs.",
                      "All serving disables proposed-block tensor capture; support stores only committed token metadata."]}


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def load_reference(path, expected_sha256):
    """Load JSON or gzip JSON only after checking the caller's file-byte binding."""
    require(_sha(expected_sha256), "Expected reference file SHA256 is required")
    path = Path(path)
    data = path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == expected_sha256, "Reference artifact SHA256 differs")
    payload = gzip.decompress(data) if data[:2] == b"\x1f\x8b" else data
    reference = json.loads(payload)
    require(isinstance(reference, dict) and "source_binding" not in reference,
            "Reference must be an original result object without an injected source binding")
    reference["source_binding"] = {"path": str(path.resolve()), "file_sha256": expected_sha256,
                                   "uncompressed_sha256": hashlib.sha256(payload).hexdigest(),
                                   "result_sha256": digest(reference)}
    return reference


def checkpoint_identities(reference):
    return {"F": reference["canonical"][0]["checkpoint_identity_sha256"],
            **{f"C{event}": update["checkpoint"]["checkpoint_identity_sha256"]
               for event, update in enumerate(reference["updates"], 1)}}


def _tokens(ids, name, *, maximum):
    require(isinstance(ids, list) and ids and len(ids) <= maximum and
            all(type(value) is int and value >= 0 for value in ids), "Invalid " + name + " tokens")


def validate_reference(reference, rows):
    """Check compact evidence only; never reopen retired tensors or remote weights."""
    validate_episode(rows)
    require(isinstance(reference, dict) and reference.get("status") == "complete", "Reference is incomplete")
    require(reference.get("episode_id") == rows[0]["stream_id"] and
            reference.get("scenario") == rows[0]["stream_type"] and
            reference.get("replica") == rows[0]["replica"], "Reference episode differs")
    binding = reference.get("source_binding")
    require(isinstance(binding, dict) and _sha(binding.get("file_sha256")) and
            _sha(binding.get("uncompressed_sha256")) and
            binding.get("result_sha256") == digest({k: v for k, v in reference.items() if k != "source_binding"}),
            "Reference source binding is absent or changed")
    canonical, updates, scores = (reference.get(key) for key in ("canonical", "updates", "scores"))
    require(isinstance(canonical, list) and len(canonical) == 32, "Reference needs 32 canonical turns")
    require(isinstance(updates, list) and len(updates) == 2, "Reference needs both completed updates")
    for event, update in enumerate(updates, 1):
        checkpoint = update.get("checkpoint", {})
        require(update.get("checkpoint_id") == f"U{event}" and update.get("optimizer_steps") == 8 * event and
                update.get("status") == "passed" and isinstance(update.get("checkpoint_path"), str) and
                bool(update["checkpoint_path"]) and _sha(checkpoint.get("checkpoint_identity_sha256")),
                "Reference checkpoint completion/binding differs")
        files = checkpoint.get("files", {})
        require(all(isinstance(files.get(name), dict) and _sha(files[name].get("sha256")) and
                    type(files[name].get("bytes")) is int and files[name]["bytes"] > 0
                    for name in ("config.json", "model.safetensors", "version.txt")),
                "Reference checkpoint file bindings absent")
    identities = checkpoint_identities(reference)
    require(all(_sha(value) for value in identities.values()), "Reference checkpoint identity absent")
    for row, item in zip(rows, canonical):
        turn = row["turn"]
        expected_checkpoint = "F" if turn <= 16 else ("U1" if turn <= 24 else "U2")
        require(item.get("turn") == turn and item.get("episode_id") == row["stream_id"] and
                item.get("question_id") == row["question_id"] and item.get("phase") == row["phase"] and
                item.get("checkpoint_id") == expected_checkpoint, "Reference canonical chronology differs")
        _tokens(item.get("input_ids"), "reference input", maximum=INPUT_LIMIT)
        _tokens(item.get("output_ids"), "reference output", maximum=OUTPUT_CAP)
        require(item.get("input_ids_sha256") == digest(item["input_ids"]), "Reference input hash differs")
        label = expected_checkpoint.replace("U", "C")
        require(item.get("checkpoint_identity_sha256") == identities[label], "Reference weights changed within window")
    require(isinstance(scores, list) and len(scores) == 144, "Reference needs complete original scored windows")
    expected = {(turn, repeat, method) for turn in range(17, 33) for repeat in range(REPEATS)
                for method in ("F", "U1" if turn <= 24 else "U2", "AR")}
    observed = [(s.get("turn"), s.get("repeat"), s.get("checkpoint_id")) for s in scores]
    require(len(set(observed)) == 144 and set(observed) == expected, "Reference scores incomplete/duplicated")
    for score in scores:
        item = canonical[score["turn"] - 1]
        require(score.get("input_ids_sha256") == digest(item["input_ids"]), "Reference score input differs")
        if score["checkpoint_id"] != "AR":
            require(score.get("output_ids") == item["output_ids"] and
                    score.get("canonical_output_match") is True, "Reference SD preservation failed")
    return reference


def method_order(methods, repeat, episode_index):
    require(len(methods) >= 2 and len(set(methods)) == len(methods), "Distinct methods required")
    require(type(repeat) is int and 0 <= repeat < REPEATS, "Invalid repeat")
    offset = (repeat + episode_index) % len(methods)
    return list(methods[offset:]) + list(methods[:offset])


def _capture_off(result):
    require(result.get("feedback") is None and result.get("capture_costs") is None,
            "Proposed-block tensor capture is forbidden in follow-up serving")


def _input(ids):
    _tokens(ids, "input", maximum=INPUT_LIMIT)
    require(len(ids) + OUTPUT_CAP + BLOCK_SIZE - 1 <= CONTEXT_LIMIT,
            "Full accumulated history exceeds qualified context; truncation prohibited")


def _canonical_call(backend, ids, checkpoint, turn, identities):
    _input(ids)
    output = backend.serve(list(ids), checkpoint, capture=False, turn_index=turn, repeat=None)
    _base.valid_result(output, ids, checkpoint, False)
    _capture_off(output)
    require(output.get("checkpoint_identity_sha256") == identities[checkpoint], "Canonical checkpoint identity changed")
    return output


def score_window(backend, saved, methods, *, episode_index, identities, emit, destination):
    for repeat in range(REPEATS):
        order = method_order(methods, repeat, episode_index)
        emit("method_order", {"turns": [row["turn"] for row in saved], "repeat": repeat, "methods": order})
        for method in order:
            for item in saved:
                ids = item["input_ids"]
                _input(ids)
                output = backend.serve(list(ids), method, capture=False, turn_index=item["turn"], repeat=repeat)
                require(isinstance(output, dict), "Backend result must be an object")
                input_match = output.get("input_ids_sha256") == digest(ids)
                output_match = output.get("output_ids") == item["output_ids"]
                identity_match = output.get("checkpoint_identity_sha256") == identities[method]
                scored = {**output, "episode_id": item["episode_id"], "turn": item["turn"],
                          "question_id": item["question_id"], "repeat": repeat, "checkpoint_id": method,
                          "benchmark_source": "original_saved_canonical", "training_data": False,
                          "reference_input_ids_sha256": digest(ids),
                          "reference_output_ids_sha256": digest(item["output_ids"]),
                          "input_match": input_match, "canonical_output_match": output_match,
                          "checkpoint_identity_match": identity_match,
                          "same_generation_valid": input_match and output_match and identity_match}
                # Preserve every discrepant common-input measurement before failing.
                destination.append(scored)
                emit("score", scored)
                _base.valid_result(output, ids, method, False)
                _capture_off(output)
                require(input_match and output_match, "Native SD output/input mismatch; investigate common-input preservation")
                require(identity_match, "Score checkpoint identity changed; fixed weights required")


def _result(rows, reference, mode):
    old_updates = []
    for update in reference["updates"]:
        accounting = update.get("accounting", {})
        exposure = {key: value for key, value in accounting.items() if key == "response_audits" or
                    any(term in key for term in ("eligible", "presentation", "support_turn", "optimizer_steps"))}
        old_updates.append({"checkpoint_id": update["checkpoint_id"], "optimizer_steps": update["optimizer_steps"],
                            "checkpoint_identity_sha256": update["checkpoint"]["checkpoint_identity_sha256"],
                            "recipe": copy.deepcopy(accounting.get("recipe")), "actual_exposure": copy.deepcopy(exposure),
                            "observed_update_call_s": update.get("observed_update_call_s"),
                            "training_s": update.get("training_s"), "export_s": update.get("export_s"),
                            "optimizer_state_save_s": update.get("optimizer_state_save_s"),
                            "trainer_startup_s": update.get("trainer_startup_s"),
                            "checkpoint_transfer": copy.deepcopy(update.get("checkpoint_transfer")),
                            "costs": copy.deepcopy(update.get("costs")),
                            "accounting_costs": copy.deepcopy(accounting.get("costs"))})
    return {"schema_version": 1, "episode_id": rows[0]["stream_id"], "scenario": rows[0]["stream_type"],
            "replica": rows[0]["replica"], "mode": mode, "status": "running", "canonical": [],
            "scores": [], "updates": [], "support": [], "trajectory_comparisons": [],
            "reference_source_binding": copy.deepcopy(reference["source_binding"]),
            "original_update_costs": old_updates,
            "original_capture_costs": [{"turn": row["turn"], "costs": copy.deepcopy(row["capture_costs"])}
                                       for row in reference["canonical"] if row.get("capture_costs") is not None],
            "original_cost_scope": ("Historical source-bound measurements from the completed original campaign; "
                                    "not fresh paired timing. Original capture is nested in its canonical request timer; "
                                    "captured-feature teacher-head reconstruction differs from new full teacher preparation. "
                                    "Original training/export/state-save/startup costs are nested in the original update wait. "
                                    "Recipe losses use different reductions and are not compared."),
            "author_arm": AUTHOR_ARM, "author_commit": AUTHOR_COMMIT}


def _finish(result, backend, started, emit):
    try:
        result["cleanup"] = backend.finish_episode()
        if isinstance(result["cleanup"], dict) and result["cleanup"].get("status") in ("failed", "error"):
            raise ValueError("Backend cleanup failed")
    except Exception as error:
        result.update(status="failed", cleanup_error=repr(error))
        emit("failure", {"cleanup_error": repr(error)})
    result["experiment_wall_s"] = time.monotonic() - started
    result["timing_note"] = ("Experiment wall includes diagnostic score replays and lifecycle; it is not production session time. "
                             "Canonical request time includes token collection if performed. Support preparation is separate; "
                             "teacher replay, training and checkpoint export are nested inside observed_update_call_s. "
                             "Canonical publication is activation_s outside request_s; do not add nested timers twice.")


def run_checkpoint_comparison(rows, reference, backend, *, emit=lambda kind, data: None):
    """F/current-U1/current-U2 on original turns25–32; no generation or learning."""
    episode_index = validate_episode(rows)
    validate_reference(reference, rows)
    result, started = _result(rows, reference, "checkpoints-only"), time.monotonic()
    try:
        backend.start_episode(result["episode_id"], reference=reference)
        score_window(backend, reference["canonical"][24:32], ["F", "C1", "C2"],
                     episode_index=episode_index, identities=checkpoint_identities(reference),
                     emit=emit, destination=result["scores"])
        result["status"] = "complete"
    except Exception as error:
        result.update(status="failed", error=repr(error))
        emit("failure", {"error": repr(error), "completed_scores": len(result["scores"])})
    finally:
        _finish(result, backend, started, emit)
    return result


def _prepare(backend, canonical, emit, destination):
    support = []
    for item in canonical:
        started = time.monotonic()
        prepared = backend.prepare_support(copy.deepcopy(item))
        require(isinstance(prepared, dict), "Token-only support metadata required")
        require(prepared.get("support_turn") == item["turn"] and
                prepared.get("capture_checkpoint") == item["checkpoint_id"] and
                prepared.get("input_ids") == item["input_ids"] and prepared.get("output_ids") == item["output_ids"],
                "Support must contain exactly its own canonical response and conditioning history")
        require(not any(prepared.get(key) is not None for key in
                        ("feedback", "capture_costs", "tensors", "tensor_path", "feature_path", "proposed_blocks")),
                "Support must be token-only, without proposed-block capture")
        # Reject tensor objects; serialization also proves support metadata can be preserved.
        digest(prepared)
        prepared = {**prepared, "observed_support_prepare_s": time.monotonic() - started}
        destination.append(prepared)
        emit("support", prepared)
        support.append(prepared)
    return support


def run_author_episode(rows, reference, backend, *, emit=lambda kind, data: None):
    """Own-history A1/A2 learning; fixed original-context F/C1/C2/A1/A2 scoring."""
    episode_index = validate_episode(rows)
    validate_reference(reference, rows)
    result, started = _result(rows, reference, "author-path"), time.monotonic()
    identities, messages = checkpoint_identities(reference), []
    try:
        backend.start_episode(result["episode_id"], reference=reference)
        for row in rows:
            turn = row["turn"]
            checkpoint = "F" if turn <= 16 else ("A1" if turn <= 24 else "A2")
            messages.append({"role": "user", "content": row["prompt"]})
            ids = backend.encode(messages)
            output = _canonical_call(backend, ids, checkpoint, turn, identities)
            item = {**output, "episode_id": result["episode_id"], "turn": turn,
                    "question_id": row["question_id"], "phase": row["phase"],
                    "checkpoint_id": checkpoint, "input_ids": list(ids)}
            result["canonical"].append(item)
            emit("canonical", item)
            original = reference["canonical"][turn - 1]
            comparison = {"episode_id": result["episode_id"], "turn": turn,
                          "question_id": row["question_id"], "checkpoint_id": checkpoint,
                          "reference_checkpoint_id": original["checkpoint_id"],
                          "input_match": ids == original["input_ids"],
                          "output_match": output["output_ids"] == original["output_ids"],
                          "own_input_ids_sha256": digest(ids), "reference_input_ids_sha256": digest(original["input_ids"]),
                          "own_output_ids_sha256": digest(output["output_ids"]),
                          "reference_output_ids_sha256": digest(original["output_ids"]),
                          "mismatch_policy": "retain_trajectory_evidence"}
            result["trajectory_comparisons"].append(comparison)
            emit("trajectory_comparison", comparison)
            messages.append({"role": "assistant", "content": output["text"], "output_ids": list(output["output_ids"])})
            if turn in (24, 32):
                methods = ["F", "C1", "A1"] if turn == 24 else ["F", "C1", "C2", "A2"]
                saved = reference["canonical"][16:24] if turn == 24 else reference["canonical"][24:32]
                score_window(backend, saved, methods, episode_index=episode_index, identities=identities,
                             emit=emit, destination=result["scores"])
                emit("evaluation_window_complete", {"turns": [x["turn"] for x in saved], "methods": methods})
            if turn in (16, 24):
                event = 1 if turn == 16 else 2
                # At24 this runs only after every first-window score has been emitted.
                support = _prepare(backend, result["canonical"][turn - 8:turn], emit, result["support"])
                started_update = time.monotonic()
                update = backend.train_event(event, copy.deepcopy(support))
                require(isinstance(update, dict), "Training update metadata required")
                update = {**update, "observed_update_call_s": time.monotonic() - started_update}
                result["updates"].append(update)
                emit("update", update)
                require(update.get("checkpoint_id") == f"A{event}" and update.get("optimizer_steps") == 8 * event,
                        "Cumulative optimizer state/publication differs")
                identity = update.get("checkpoint_identity_sha256", update.get("checkpoint", {}).get("checkpoint_identity_sha256"))
                require(_sha(identity), "Published author checkpoint identity absent")
                identities[f"A{event}"] = identity
        result["status"] = "complete"
    except Exception as error:
        result.update(status="failed", error=repr(error))
        emit("failure", {"error": repr(error), "completed_turns": len(result["canonical"]),
                         "completed_scores": len(result["scores"])})
    finally:
        _finish(result, backend, started, emit)
    return result
