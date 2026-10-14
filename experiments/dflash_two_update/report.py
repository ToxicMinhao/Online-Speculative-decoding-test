"""Descriptive paired report. Training loss cannot substitute for native serving."""
from collections import defaultdict


INFERENCE_SCOPE = ("Inference-only performance with an available checkpoint: complete request time includes "
                   "prefill, drafting and target verification, and excludes training and activation. "
                   "Paired scoring has capture disabled; canonical request time includes capture when enabled.")


def totals(rows):
    tokens = sum(len(r["output_ids"]) for r in rows)
    calls = sum(r["verification_calls"] for r in rows)
    accepted = sum(r["accepted_draft_tokens"] for r in rows)
    seconds = sum(r["request_s"] for r in rows)
    verified = sum(r["verification_committed_tokens"] for r in rows) if rows and all("verification_committed_tokens" in r for r in rows) else None
    return {"measurement_scope": INFERENCE_SCOPE,
            "requests": len(rows), "output_tokens": tokens, "verification_calls": calls,
            "accepted_draft_tokens": accepted, "complete_request_s": seconds,
            "calls_per_1000_output_tokens": 1000 * calls / tokens if tokens else None,
            "accepted_draft_tokens_per_verification": accepted / calls if calls else None,
            "committed_output_tokens_per_verification": tokens / calls if calls else None,
            "verification_committed_tokens_per_call": verified / calls if calls and verified is not None else None,
            "output_tokens_per_second": tokens / seconds if seconds else None}


def timer_sums(records):
    """Diagnostic aggregates only: these timers can nest and must not be added."""
    result = defaultdict(float)
    for record in records:
        for key, value in record.items():
            if key.endswith("_s") and type(value) in (float, int):
                result[key] += value
    return dict(result)


def cost_breakdown(episode):
    canonical, updates = episode["canonical"], episode["updates"]
    capture = [dict(turn=r["turn"], costs=r["capture_costs"]) for r in canonical if r.get("capture_costs")]
    presentations = [p for u in updates for b in u.get("accounting", {}).get("batches", [])
                     for p in b["presentations"]]
    phases = timer_sums(p.get("phase_seconds", {}) for p in presentations)
    replay = timer_sums(p.get("forward_nested_phase_seconds", {}) for p in presentations)
    return {
        "capture": {"per_request": capture, "timer_sums": timer_sums(r["costs"] for r in capture),
                    "scope": "Inside canonical request_s. Transfer overlaps capture timers; do not add again."},
        "replay": {"nested_phase_seconds": replay,
                   "scope": "Inside forward_and_heads_s and training_s. Draft replay includes feature transfer; "
                            "teacher_head_reconstruction_s reconstructs labels from captured features, "
                            "not a target-transformer rerun. Missing timers are unmeasured, not zero."},
        "training": {"event_timer_sums": timer_sums(u.get("costs", {}) for u in updates),
                     "response_phase_seconds": phases,
                     "scope": "Includes replay. Export/state-save appear in event timers separately. All are "
                              "inside observed_synchronous_update_wait_s; do not add to that wait."},
        "per_update": [{"checkpoint_id": u.get("checkpoint_id"),
                        "training_s": u.get("training_s"), "export_s": u.get("export_s"),
                        "optimizer_state_save_s": u.get("optimizer_state_save_s"),
                        "checkpoint_transfer": u.get("checkpoint_transfer"),
                        "observed_update_call_s": u["observed_update_call_s"]} for u in updates],
        "activation_s": sum(r.get("activation_s", 0.) for r in canonical),
        "native_activation_s": (sum(r["native_activation_s"] for r in canonical)
                                if canonical and all("native_activation_s" in r for r in canonical) else None),
        "activation_scope": "Canonical full publication RPC, including checkpoint validation. "
                            "native_activation_s is a nested worker timer, never an additional charge. "
                            "Benchmark-only switches remain in scored rows and actual experiment wall time.",
        "transfer_scope": "Capture device-to-host transfer is in capture.timer_sums.transfer_s. "
                          "Replay feature transfer is inside draft replay; export/publication device copies "
                          "are inside those timers. Per-update checkpoint_transfer states whether a separate "
                          "checkpoint transfer occurred; null means unrecorded, not free. This native runner "
                          "shares checkpoint paths between workers and performs no separate transfer."}


def summarize(episodes):
    results = []
    for episode in episodes:
        groups = defaultdict(list)
        for row in episode["scores"]:
            block = "17–24" if row["turn"] <= 24 else "25–32"
            groups[(block, row["repeat"], row["checkpoint_id"])].append(row)
        windows = []
        for block in ("17–24", "25–32"):
            adaptive = "U1" if block == "17–24" else "U2"
            expected_turns = list(range(17, 25)) if adaptive == "U1" else list(range(25, 33))
            for repeat in range(3):
                entries = {arm: groups[(block, repeat, arm)] for arm in ("F", adaptive, "AR")}
                metrics = {arm: totals(rows) for arm, rows in entries.items()}
                paired = (all(sorted(r["turn"] for r in entries[arm]) == expected_turns for arm in ("F", adaptive)) and
                          all(r.get("same_generation_valid") is True for arm in ("F", adaptive) for r in entries[arm]) and
                          {(r["turn"], r["question_id"], r["input_ids_sha256"]) for r in entries["F"]} ==
                          {(r["turn"], r["question_id"], r["input_ids_sha256"]) for r in entries[adaptive]} and
                          {r["turn"]: r["output_ids"] for r in entries["F"]} ==
                          {r["turn"]: r["output_ids"] for r in entries[adaptive]})
                base_s = metrics["F"]["complete_request_s"]
                windows.append({"turns": block, "repeat": repeat, "metrics": metrics,
                    "paired_same_generation_valid": paired,
                    "latency_reduction_percent": 100 * (1 - metrics[adaptive]["complete_request_s"] / base_s)
                        if paired and base_s else None,
                    "AR_same_generation_valid": paired and sorted(r["turn"] for r in entries["AR"]) == expected_turns and
                        all(r.get("same_generation_valid") is True for r in entries["AR"]) and
                        {r["turn"]: (r["input_ids_sha256"], r["output_ids"]) for r in entries["AR"]} ==
                        {r["turn"]: (r["input_ids_sha256"], r["output_ids"]) for r in entries["F"]}})
        canonical = episode["canonical"]
        serving_s = sum(r["request_s"] for r in canonical)
        cache_s = sum(r.get("cache_maintenance_s", 0.) for r in canonical)
        publication_s = sum(r.get("activation_s", 0.) for r in canonical)
        update_wait_s = sum(r["observed_update_call_s"] for r in episode["updates"])
        complete = (episode["status"] == "complete" and [r["turn"] for r in canonical] == list(range(1, 33)) and
                    [u.get("checkpoint_id") for u in episode["updates"]] == ["U1", "U2"] and
                    [u.get("optimizer_steps") for u in episode["updates"]] == [8, 16] and
                    all(w["paired_same_generation_valid"] for w in windows))
        results.append({"episode_id": episode["episode_id"], "scenario": episode["scenario"],
            "replica": episode["replica"], "status": episode["status"], "error": episode.get("error"),
            "windows": windows, "canonical_serving_s": serving_s,
            "canonical_cache_maintenance_s": cache_s, "canonical_publication_s": publication_s,
            "observed_synchronous_update_wait_s": update_wait_s,
            "accounted_adaptive_episode_s": serving_s + cache_s + publication_s + update_wait_s if complete else None,
            "experiment_wall_s": episode["experiment_wall_s"],
            "initial_and_diagnostic_loading": episode.get("cleanup"),
            "cost_breakdown": cost_breakdown(episode),
            "updates": episode["updates"],
            "frozen_episode_reference_s_by_repeat": [
                sum(r["request_s"] + r.get("cache_maintenance_s", 0.) for r in episode["baseline_early"]) +
                sum(r["request_s"] + r.get("cache_maintenance_s", 0.) for r in episode["scores"]
                    if r["checkpoint_id"] == "F" and r["repeat"] == repeat)
                if complete and [r["turn"] for r in episode["baseline_early"]] == list(range(1, 17)) and
                    sum(r["checkpoint_id"] == "F" and r["repeat"] == repeat for r in episode["scores"]) == 16 else None
                for repeat in range(3)]})
    return {"schema_version": 2, "episodes": results,
            "interpretation_limits": [
                INFERENCE_SCOPE,
                "Keep every scenario, episode, failure, mismatch and timing repeat visible; no winner filtering.",
                "Accounted episode cost sums nonoverlapping serving/cache/publication/update waits. Nested capture/replay/training timers explain costs and are not added again.",
                "Accounted episode cost is reconstructed from serialized measurements, not a measured uninterrupted production-session wall clock; diagnostic restore costs must be identified in backend receipts.",
                "Initial loading, diagnostic validation, repeated timing, AR engine switching and benchmark-only checkpoint switches are experiment overhead. Checkpoint validation needed for canonical publication belongs to activation, not excluded diagnostics.",
                "Frozen episode references reuse the initial 16 measurements; they are not three independent full-session timings.",
                "Three timing repeats reuse the same contexts and checkpoints; they are not three training seeds.",
                "committed_output_tokens_per_verification is all output tokens divided by calls, including the initial target token; use trace-derived verification_committed_tokens_per_call for verification-only progress when available.",
                "U1 and U2 are evaluated on different questions/histories; their raw difference does not isolate the incremental effect of update2.",
                "AR mismatches invalidate an identical-output acceleration claim for that comparison, not the retained reference measurements.",
                "No positive outcome, forgetting mechanism, or confirmation activation is assumed."]}
