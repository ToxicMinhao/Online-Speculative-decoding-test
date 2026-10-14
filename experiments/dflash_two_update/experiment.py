"""Two-update chronology and matched-input evaluation, independent of GPU runtime.

No model is loaded on import. The native backend is selected explicitly by CLI.
Canonical answers always come from the currently serving adaptive checkpoint.
Timing replays never append answers to history or become new training examples.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import time


SCENARIOS = ("STABLE_CODE", "STABLE_MATH", "SHORT_CODE_FIRST",
             "SHORT_MATH_FIRST", "RANDOM_MIX")
PHASES = ("context", "support", "evaluation_then_support", "final_evaluation")
REPEATS = 3
INPUT_LIMIT, CONTEXT_LIMIT, OUTPUT_CAP, BLOCK_SIZE = 31744, 32768, 256, 10


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def dataset_api():
    path = Path(__file__).resolve().parents[2] / "datasets/two_update/build_streams.py"
    spec = importlib.util.spec_from_file_location("two_update_dataset_builder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_dataset(selection_path, streams_path):
    api = dataset_api()
    selection, _ = api.read_selection(Path(selection_path))
    streams = [json.loads(line) for line in Path(streams_path).read_text().split("\n") if line]
    api.validate_streams(selection, streams)
    return streams


def plan(streams):
    groups = {}
    for row in streams:
        if row["replica"] in (1, 2):
            groups.setdefault(row["stream_id"], []).append(row)
    require(len(groups) == 10, "Exactly ten development episodes required")
    episodes = []
    for name, rows in groups.items():
        require([r["turn"] for r in rows] == list(range(1, 33)), "Incomplete chronological episode")
        require(all(r["split"] == "development" for r in rows), "Confirmation episode forbidden")
        episodes.append({"episode_id": name, "scenario": rows[0]["stream_type"],
                         "replica": rows[0]["replica"], "request_ids": [r["question_id"] for r in rows],
                         "updates_after_turns": [16, 24], "support_turns": [list(range(9, 17)), list(range(17, 25))],
                         "scored_turns": [list(range(17, 25)), list(range(25, 33))]})
    return {"schema_version": 1, "status": "plan_only", "baseline": "full_weight_forward_KL_DFlash_frozen_target",
            "streams_sha256": digest(streams), "episodes": episodes, "adaptation_events": 20,
            "optimizer_steps_per_event": 8, "optimizer_steps_total": 160,
            "response_presentations_total": 320, "paired_timing_repeats": REPEATS,
            "distinct_scored_request_contexts": 160,
            "scored_SD_executions": 960, "AR_reference_executions": 480,
            "canonical_request_executions": 320,
            "notes": ["Timing replays are repeated measurements, not independent questions or extra training.",
                      "Additional capture-off frozen timings on turns 9–16 account for a no-training baseline.",
                      "Replicas 3–4 remain reserved; this plan does not authorize a GPU launch."]}


def valid_result(result, input_ids, checkpoint, capture):
    require(isinstance(result, dict), "Backend result must be an object")
    ids = result.get("output_ids")
    require(isinstance(ids, list) and ids and all(type(x) is int and x >= 0 for x in ids),
            "Native output token IDs required")
    require(len(ids) <= OUTPUT_CAP and isinstance(result.get("text"), str), "Output cap/text contract differs")
    require(type(result.get("request_s")) in (float, int) and
            math.isfinite(result["request_s"]) and result["request_s"] >= 0, "Invalid request timer")
    for key in ("accepted_draft_tokens", "verification_calls"):
        require(type(result.get(key)) is int and result[key] >= 0, "Missing native metric: " + key)
    require(result["accepted_draft_tokens"] <= len(ids), "Accepted tokens exceed committed output")
    require((checkpoint == "AR" and result["verification_calls"] == 0) or
            (checkpoint != "AR" and result["verification_calls"] > 0), "Verification count differs")
    require(not capture or isinstance(result.get("feedback"), dict), "Captured current-response feedback absent")
    require(result.get("input_ids_sha256", digest(input_ids)) == digest(input_ids), "Backend input changed")
    for key in ("cache_maintenance_s", "activation_s", "native_activation_s", "startup_s"):
        value = result.get(key, 0.)
        require(type(value) in (float, int) and math.isfinite(value) and value >= 0, "Invalid cost: " + key)


def call(backend, input_ids, checkpoint, *, capture, turn, repeat=None):
    require(type(turn) is int and 1 <= turn <= 32, "Invalid turn")
    require(input_ids and all(type(x) is int and x >= 0 for x in input_ids), "Invalid input tokens")
    # These are the qualified independent limits, not a history-truncation policy.
    require(len(input_ids) <= INPUT_LIMIT and len(input_ids) + OUTPUT_CAP + BLOCK_SIZE - 1 <= CONTEXT_LIMIT,
            "Full accumulated history exceeds qualified context; truncation prohibited")
    result = backend.serve(list(input_ids), checkpoint, capture=capture, turn_index=turn, repeat=repeat)
    valid_result(result, input_ids, checkpoint, capture)
    return result


def score_window(backend, saved, checkpoint, *, episode_index, emit, destination=None):
    """Three alternating SD pairs, then grouped AR to avoid per-request restarts."""
    scored = [] if destination is None else destination
    for repeat in range(REPEATS):
        order = ["F", checkpoint] if (repeat + episode_index) % 2 == 0 else [checkpoint, "F"]
        for method in order:
            for item in saved:
                result = call(backend, item["input_ids"], method, capture=False,
                              turn=item["turn"], repeat=repeat)
                match = result["output_ids"] == item["output_ids"]
                row = {"turn": item["turn"], "question_id": item["question_id"], "repeat": repeat,
                       "checkpoint_id": method, "input_ids_sha256": digest(item["input_ids"]),
                       "canonical_output_match": match, "same_generation_valid": match, **result}
                emit("score", row)
                scored.append(row)
                # Preserve the mismatch before stopping; never drop/repair the affected row.
                require(match, "Native SD output mismatch; affected comparison requires investigation")
    for repeat in range(REPEATS):
        for item in saved:
            result = call(backend, item["input_ids"], "AR", capture=False, turn=item["turn"], repeat=repeat)
            match = result["output_ids"] == item["output_ids"]
            row = {"turn": item["turn"], "question_id": item["question_id"], "repeat": repeat,
                   "checkpoint_id": "AR", "input_ids_sha256": digest(item["input_ids"]),
                   "canonical_output_match": match, "same_generation_valid": match, **result}
            emit("score", row)
            scored.append(row)
    return scored


def run_episode(rows, backend, *, emit=lambda kind, data: None):
    """Run one development episode. No automatic resume, retuning, or confirmation."""
    require(len(rows) == 32 and [r["turn"] for r in rows] == list(range(1, 33)), "Expected32orderedturns")
    first = rows[0]
    episode = first["stream_id"]
    require(first["replica"] in (1, 2) and first["stream_type"] in SCENARIOS, "Only development is admitted")
    require(all(r["stream_id"] == episode and r["replica"] == first["replica"] and
                r["stream_type"] == first["stream_type"] and r["split"] == "development" and
                r["phase"] == PHASES[(r["turn"] - 1) // 8] for r in rows), "Episode metadata differs")
    episode_index = (first["replica"] - 1) * 5 + SCENARIOS.index(first["stream_type"])
    start = time.monotonic()
    result = {"schema_version": 1, "episode_id": episode, "scenario": first["stream_type"],
              "replica": first["replica"], "status": "running", "canonical": [], "scores": [],
              "updates": [], "baseline_early": [], "baseline": "full_weight_forward_KL_DFlash_frozen_target"}
    messages, support, saved = [], [], []
    try:
        backend.start_episode(episode)
        for row in rows:
            turn = row["turn"]
            checkpoint = "F" if turn <= 16 else ("U1" if turn <= 24 else "U2")
            messages.append({"role": "user", "content": row["prompt"]})
            ids = backend.encode(messages)
            capture = 9 <= turn <= 24
            output = call(backend, ids, checkpoint, capture=capture, turn=turn)
            item = {"episode_id": episode, "turn": turn, "question_id": row["question_id"],
                    "phase": row["phase"], "checkpoint_id": checkpoint, "input_ids": ids, **output}
            emit("canonical", item)
            result["canonical"].append(item)
            messages.append({"role": "assistant", "content": output["text"], "output_ids": output["output_ids"]})
            if capture:
                support.append(output["feedback"])
            if turn <= 8:
                result["baseline_early"].append(item)
            elif turn <= 16:
                # Frozen serving has no reason to retain learning tensors: charge capture
                # only to the adaptive session, with the same complete request boundary.
                base = call(backend, ids, "F", capture=False, turn=turn, repeat=0)
                matched = base["output_ids"] == output["output_ids"]
                emit("baseline_early", {"turn": turn, "canonical_output_match": matched, **base})
                require(matched, "Capture-on/off changed native output; investigate before training")
                result["baseline_early"].append({"turn": turn, **base})
            else:
                saved.append(item)
            if turn in (24, 32):
                score_window(backend, saved, checkpoint, episode_index=episode_index,
                             emit=emit, destination=result["scores"])
                emit("evaluation_window_complete", {"turns": [x["turn"] for x in saved], "checkpoint_id": checkpoint})
                saved = []
            if turn in (16, 24):
                event = 1 if turn == 16 else 2
                require(len(support) == 8, "Training must contain exactly eight new responses")
                update_started = time.monotonic()
                update = backend.train_event(event, list(support))
                update["observed_update_call_s"] = time.monotonic() - update_started
                require(update["checkpoint_id"] == "U" + str(event) and update["optimizer_steps"] == 8 * event,
                        "Cumulative optimizer state/publication differs")
                emit("update", update)
                result["updates"].append(update)
                support = []
        result["status"] = "complete"
    except Exception as error:
        result.update(status="failed", error=repr(error))
        emit("failure", {"error": repr(error), "completed_turns": len(result["canonical"])})
    finally:
        try:
            result["cleanup"] = backend.finish_episode()
        except Exception as error:
            result.update(status="failed", cleanup_error=repr(error))
        result["experiment_wall_s"] = time.monotonic() - start
    result["timing_note"] = ("Experiment wall time includes repeated diagnostic evaluations and engine lifecycle. "
                             "It is not production episode time. Capture is included in canonical request_s; "
                             "nested capture timers must not be added twice.")
    return result
