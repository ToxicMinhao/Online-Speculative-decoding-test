#!/usr/bin/env python3
"""Read-only CPU audit of retrieved Track 2 compact evidence.

No experiment implementation, model framework, network or GPU API is imported.
Private prompts/responses are inspected in memory and are never emitted.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import resource
import time

import numpy as np
from audit_math import ARMS, PAIRS, COUNT_KEYS, TIME_KEYS, selected_records, admit_fresh, endpoint_statistics

EXECUTION = "8a4cbcc8b78adfee845a0a1409db4a0d29c54058"
DOMAINS = ("math", "math", "coding", "coding", "math", "math", "coding", "coding")
LIMITS = dict(pre_update_generations=3072, capture_parity_generations=4, terminal_preparation_forwards=1024)
WORKER_LIMITS = dict(response_forwards=256, response_backwards=256, optimizer_steps=128)


class Audit:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.files, self.errors, self.sections = {}, [], {}
        self.checks = 0
        self.plan = self.inputs = self.canonicals = None
        self.selection_plans = {}

    def check(self, condition, message):
        self.checks += 1
        if not condition:
            self.errors.append(message)
        return bool(condition)

    def path(self, value):
        path = Path(value)
        if path.is_absolute() and not path.is_relative_to(self.root):
            for marker in ("/campaign_001/", "/sealed/", "/final_report/"):
                if marker in str(path):
                    path = self.root / marker.strip("/") / str(path).split(marker, 1)[1]
                    break
        if not path.is_absolute():
            path = self.root / path
        if not path.resolve().is_relative_to(self.root):
            raise ValueError("Evidence path escapes the retrieved run root")
        return path

    def read(self, value, lines=False):
        path = self.path(value)
        data = path.read_bytes()
        self.files[str(path.relative_to(self.root))] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        return [json.loads(line) for line in data.splitlines() if line] if lines else json.loads(data)

    def same(self, expected, observed, label):
        """Compare rebuilt fields; emit only field locators, never private data."""
        if isinstance(expected, dict):
            if not self.check(isinstance(observed, dict), label + ": mapping type"):
                return
            for key, value in expected.items():
                if self.check(key in observed, label + ": missing " + key):
                    self.same(value, observed[key], label + "/" + key)
        elif isinstance(expected, (list, tuple)):
            if self.check(isinstance(observed, (list, tuple)) and len(expected) == len(observed), label + ": list length"):
                for index, (left, right) in enumerate(zip(expected, observed)):
                    self.same(left, right, label + "/" + str(index))
        elif isinstance(expected, float):
            self.check(type(observed) in (float, int) and np.isclose(expected, observed, rtol=1e-12, atol=1e-12), label + ": number")
        else:
            self.check(expected == observed, label + ": value")

    def section(self, name, action):
        before, started = len(self.errors), time.perf_counter()
        try:
            result = action()
        except (OSError, KeyError, TypeError, ValueError, IndexError, ZeroDivisionError, AssertionError) as error:
            self.errors.append(name + ": unavailable/malformed evidence: " + repr(error))
            result = {}
        self.sections[name] = dict(status="passed" if len(self.errors) == before else "failed_or_incomplete",
                                   errors=len(self.errors) - before, wall_s=time.perf_counter() - started, details=result)

    def inputs_and_source(self):
        self.plan = self.read("sealed/PLAN.json")
        self.inputs = self.read("sealed/REQUESTS.json")
        self.canonicals = self.read("sealed/CANONICALS.json")
        self.check(self.plan["execution_source"]["commit"] == EXECUTION, "Execution source differs from launched PR24")
        self.check(self.plan["execution_source"]["dirty"] is False, "Execution source was dirty")
        for filename, key in (("REQUESTS.json", "requests"), ("CANONICALS.json", "canonicals")):
            self.same(self.plan["binding_material"][key], self.files["sealed/" + filename], "sealed/" + filename)
        training, panel = self.inputs["training"], self.inputs["evaluation"]
        self.check(len(training) == 1024 and len(panel) == 256, "Fixed complete input cardinality")
        self.check([r["global_ordinal"] for r in training] == list(range(1024)), "Fresh chronology 0..1023")
        self.check([r["event"] for r in training] == [i // 128 + 1 for i in range(1024)], "Fresh event mapping")
        self.check([r["domain"] for r in training] == [DOMAINS[i // 128] for i in range(1024)], "Fixed domain recurrence")
        self.check([r["panel_ordinal"] for r in panel] == list(range(256)), "Panel chronology")
        self.check([r["domain"] for r in panel] == ["math" if i % 2 == 0 else "coding" for i in range(256)], "Balanced panel")
        self.check(set(self.canonicals) == {r["request_id"] for r in panel}, "Canonical inventory")
        all_requests = training + panel
        self.check(len({r["request_id"] for r in all_requests}) == 1280, "Unique train/evaluation identities")
        self.check(len({tuple(r["prompt_ids"]) for r in all_requests}) == 1280, "Exact prompt disjointness")
        for row in all_requests:
            self.check(row["max_new_tokens"] == 2048 and 0 < len(row["prompt_ids"]) <= 224, "Full prompt/cap: " + row["request_id"])
        return dict(execution_commit=EXECUTION, training_requests=1024, evaluation_requests=256,
                    scope="Pinned input receipts; no new remote environment or weight-tensor inspection.")

    def native_record(self, raw, request, *, capture, canonical=None, label="record"):
        """Reconstruct full output, posterior decisions and aggregate counts."""
        metadata, response, prompt = raw["metadata"], raw["response_ids"], raw["prompt_ids"]
        self.check(prompt == request["prompt_ids"], label + ": input tokens")
        self.check(raw["stop_token_ids"] == request["stop_token_ids"] and raw["max_new_tokens"] == 2048, label + ": stop/cap")
        self.check(raw["purpose"] == ("serving" if capture else "evaluation"), label + ": purpose")
        self.check(metadata["feedback_capture_in_timed_request"] is capture, label + ": timed capture")
        self.check(metadata["capture"]["collect_features"] is capture, label + ": feature capture")
        self.check(metadata["capture"]["native_capture_validated"] is True, label + ": capture validation")
        size = metadata["capture"]["feature_bytes"]
        self.check(size > 0 if capture else size == 0, label + ": capture bytes")
        self.check(metadata["block_size"] == 16 and metadata["context_limit"] == 4096, label + ": geometry")
        self.check(metadata["runtime"] == "pinned_SGLang_FA3", label + ": native runtime")
        self.check(metadata["cache_policy"] == "idle_full_flush_before_every_request_and_publication", label + ": cache policy")
        self.check(0 < len(response) <= 2048 and all(type(token) is int and token >= 0 for token in response), label + ": response")
        stops = [i for i, token in enumerate(response) if token in raw["stop_token_ids"]]
        self.check((not stops or stops == [len(response) - 1]) and (bool(stops) or len(response) == 2048), label + ": termination")
        self.check(raw["finish_reason"] == ("eos" if stops else "length"), label + ": stop reason")
        if canonical is not None:
            self.check(response == canonical["response_ids"] and raw["finish_reason"] == canonical["finish_reason"], label + ": canonical output")
        reconstructed, accepted_total = [response[0]], 0
        for index, block in enumerate(metadata["trace"]):
            name = label + "/block" + str(index)
            proposed, posterior = block["draft_tokens"], block["target_argmax"]
            width, progress = block["verify_size"], block["committed_progress"]
            self.check(width == len(proposed) + 1 == len(posterior) and 1 <= width <= 16, name + ": verification geometry")
            first_error = next((i for i in range(width - 1) if proposed[i] != posterior[i]), width - 1)
            self.check(1 <= progress <= first_error + 1, name + ": accepted-prefix progress")
            committed = (proposed[:first_error] + [posterior[first_error]])[:progress]
            self.check(block["committed_tokens"] == committed == posterior[:progress], name + ": posterior commit")
            self.check(block["anchor_token"] == reconstructed[-1] and block["output_start"] == len(reconstructed) - 1, name + ": causal anchor")
            start = len(prompt) + len(reconstructed) - 1
            self.check(block["positions"] == list(range(start, start + width)), name + ": positions")
            accepted = min(first_error, progress)
            self.check(block["accepted_draft_tokens"] == accepted, name + ": accepted count")
            self.check(block["native_accepted_draft_tokens"] == progress - 1, name + ": native accepted count")
            self.check(block["raw_accepted_draft_tokens_before_stop"] == first_error, name + ": raw accepted count")
            self.check(block["stop_clipped"] == (progress < first_error + 1), name + ": clipping")
            reconstructed.extend(committed)
            accepted_total += accepted
        self.check(reconstructed == response, label + ": full trace reconstruction")
        counts = dict(verification_calls=len(metadata["trace"]), accepted_draft_tokens=accepted_total,
                      committed_progress=len(response) - 1, output_tokens=len(response))
        for key in COUNT_KEYS[:-1]:
            self.check(metadata[key] == counts[key], label + ": aggregate " + key)
        for key in TIME_KEYS[:-1]:
            self.check(type(metadata[key]) in (int, float) and np.isfinite(metadata[key]) and metadata[key] > 0, label + ": timing " + key)
        observed = metadata["capture"]
        self.check(observed["prompt_ids"] == prompt and observed["response_ids"] == response
                   and observed["blocks"] == metadata["trace"], label + ": native capture binding")
        self.check(self.read(metadata["native_trace_path"]) == observed, label + ": native sidecar equality")
        counts.update({key: metadata[key] for key in TIME_KEYS[:-1]})
        counts["cache_maintenance_s"] = observed["cache_maintenance_s"]
        self.check(np.isfinite(counts["cache_maintenance_s"]) and counts["cache_maintenance_s"] >= 0, label + ": cache time")
        return counts

    def work_ledger(self, path, expected):
        starts, ends = {key: set() for key in expected}, {key: set() for key in expected}
        for row in self.read(path, lines=True):
            kind, number = row["kind"], row["ordinal"]
            self.check(kind in expected and type(number) is int and 1 <= number <= expected[kind], str(path) + ": operation")
            if row["phase"] == "started":
                self.check(number == len(starts[kind]) + 1 and number not in starts[kind], str(path) + ": unique start")
                starts[kind].add(number)
            elif row["phase"] == "completed":
                self.check(number in starts[kind] and number not in ends[kind], str(path) + ": unique completion")
                ends[kind].add(number)
            else:
                self.check(False, str(path) + ": unknown phase")
        observed = dict(attempts={k: len(v) for k, v in starts.items()}, completed={k: len(v) for k, v in ends.items()},
                        unknown={k: len(starts[k] - ends[k]) for k in starts})
        self.check(observed["attempts"] == observed["completed"] == expected, str(path) + ": complete budget")
        self.check(not any(observed["unknown"].values()), str(path) + ": no unknown operation")
        return observed

    def publication(self, activation, checkpoint, label, weights=None):
        native = activation["native_status"]
        publication = native["publication"]
        self.check(activation["checkpoint_id"] == checkpoint and activation["checkpoint_structure_valid"] is True, label + ": exported identity")
        self.check(native["status"] == "passed" and native["poisoned"] is None and native["capture_pending"] is False, label + ": healthy publication")
        self.check(native["cache_tokens"] == publication["cache_after_tokens"] == 0, label + ": empty publication cache")
        self.check(native["attention_backend"] == "fa3" and native["block_size"] == 16, label + ": publication runtime")
        self.check(publication["status"] == "passed" and publication["checkpoint_path"] == checkpoint, label + ": published path")
        helpers = ("helper_matches_current_layers", "helper_norm_matches_current_layers", "fused_enabled")
        self.check(all(publication[key] is True for key in helpers), label + ": refreshed native helpers")
        if weights is not None:
            self.check(weights["status"] == "passed" and weights["all_equal"] is True and weights["checkpoint_path"] == checkpoint,
                       label + ": decoded-weight equality receipt")
            self.check(weights["tensors"] == weights["source_tensor_count"] == 58 and weights["native_tensor_count"] == 43,
                       label + ": complete weight inventory")
            self.check(all(weights[key] is True for key in helpers), label + ": helper equality receipt")

    def selection_and_memory(self):
        descriptors = self.read("campaign_001/ALL_DESCRIPTORS.json")
        training = self.inputs["training"]
        self.check(set(descriptors) == {r["request_id"] for r in training}, "All1024 original descriptors")
        for request in training:
            record = descriptors[request["request_id"]]
            self.check(record["record_id"] == request["request_id"] and record["global_ordinal"] == request["global_ordinal"]
                       and record["event_index"] == request["event"], "Descriptor chronology: " + request["request_id"])
            vector = np.array(record["descriptor"], dtype=np.float64)
            self.check(vector.shape == (4096,) and np.isfinite(vector).all() and abs(np.linalg.norm(vector) - 1) <= 1e-12,
                       "Full-width finite unit descriptor: " + request["request_id"])
            packet = self.read(f"campaign_001/events/event_{request['event']:03d}/packets/{request['request_id']}.json")
            self.check(record == packet["descriptor"], "Packet descriptor equality: " + request["request_id"])
            checks = packet["descriptor_checks"]
            self.check(checks["row_index"] == len(request["prompt_ids"]) - 1 and checks["hidden_dimensions"] == 4096
                       and np.isfinite(checks["original_norm"]) and checks["original_norm"] > 0
                       and abs(checks["normalized_norm"] - 1) <= 1e-12 and checks["all_finite"] is True,
                       "Descriptor row/norm: " + request["request_id"])
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([20261102, 0, 2])))
        slots, seen, seen_ids, summaries = [], 0, [], []
        for event in range(1, 9):
            folder = f"campaign_001/events/event_{event:03d}"
            fresh = [descriptors[r["request_id"]] for r in training[(event - 1) * 128:event * 128]]
            before = self.read(folder + "/MEMORY_BEFORE.json")
            self.check(slots == before["slots"], folder + ": prior common slots")
            self.same(rng.bit_generator.state, before["rng_state"], folder + ": prior RNG")
            self.check(before["seen_count"] == seen and before["seen_record_ids"] == seen_ids and before["completed_event"] == event - 1,
                       folder + ": prior memory counter/identity")
            plan = selected_records(event, fresh, slots)
            saved = self.read(folder + "/SELECTION.json")
            self.same(plan["passes"], saved["pass_record_ids"], folder + ": fixed pass plans")
            for key in ("uniform", "coverage"):
                self.same(plan[key], saved[key], folder + ": " + key)
            self.selection_plans[event] = plan["passes"]
            slots, seen, steps = admit_fresh(slots, fresh, rng, seen)
            seen_ids.extend(row["record_id"] for row in fresh)
            after = self.read(folder + "/MEMORY_AFTER.json")
            self.check(slots == after["slots"], folder + ": posterior Algorithm R slots")
            self.same(rng.bit_generator.state, after["rng_state"], folder + ": posterior RNG")
            self.check(after["seen_count"] == seen and after["seen_record_ids"] == seen_ids and after["completed_event"] == event,
                       folder + ": posterior memory counter/identity")
            insertion = self.read(folder + "/MEMORY_INSERTION.json")
            self.same(steps, insertion["insertions"], folder + ": every admission/eviction draw")
            self.check(set(insertion["verified_arms"]) == set(ARMS), folder + ": commit-before-insertion")
            complete = self.read(folder + "/COMPLETE.json")
            self.check(complete["status"] == "passed" and set(complete["verified_arms"]) == set(ARMS), folder + ": completed commits")
            summaries.append(dict(event=event, memory_age_events=dict(Counter(event + 1 - row["event_index"] for row in slots)),
                memory_domains=dict(Counter(DOMAINS[row["global_ordinal"] // 128] for row in slots)),
                uniform_coverage=plan["uniform"]["coverage"] if event > 1 else None,
                selected_coverage=plan["coverage"]["coverage"] if event > 1 else None))
        self.check(self.read("campaign_001/events/event_008/MEMORY_AFTER.json") == self.read("campaign_001/MEMORY.json"), "Final reservoir state")
        return dict(events=summaries, unique_admissions=seen, final_capacity=len(slots),
                    limits="Extraction checked against compact norm/binding receipts; retired source tensors not re-decoded.")

    def training_and_work(self):
        serving = self.work_ledger("campaign_001/WORK.jsonl", LIMITS)
        totals, workers, originals = Counter(), [], {}
        for arm in ARMS:
            for event in range(1, 9):
                base = f"campaign_001/arms/{arm}"
                workers.append(dict(arm=arm, event=event, work=self.work_ledger(f"{base}/events/event_{event:03d}/WORK.jsonl", WORKER_LIMITS)))
                result = self.read(f"{base}/trainer/events/event_{event:03d}/RESULT.json")
                self.check(result["status"] == "completed" and result["objective"] == "osd_first_error", f"{arm}/{event}: unchanged objective")
                self.check(result["response_presentations"] == 256 and result["optimizer_steps"] == 128
                           and result["start_step"] == (event - 1) * 128 and result["end_step"] == event * 128, f"{arm}/{event}: boundaries")
                plans, expected = result["pass_plan"], self.selection_plans[event][arm]
                self.same(expected, [[r["record_id"] for r in rows] for rows in plans], f"{arm}/{event}: original pass inputs")
                current = {r["record_id"] for r in plans[0]}
                for row in plans[0]:
                    binding = {key: row[key] for key in ("record_id", "global_ordinal", "path", "sha256")}
                    if row["record_id"] in originals:
                        self.same(originals[row["record_id"]], binding, f"{arm}/{event}: shared fresh binding")
                    else:
                        originals[row["record_id"]] = binding
                for row in plans[1]:
                    self.same(originals[row["record_id"]], {k: row[k] for k in originals[row["record_id"]]}, f"{arm}/{event}: replay binding")
                records = result["records"]
                self.check(len(records) == 256, f"{arm}/{event}: presentations")
                for number, row in enumerate(records):
                    prefix = f"{arm}/{event}/presentation{number}"
                    self.check(row["pass"] == number // 128 and row["optimizer_stepped"] is (number % 2 == 1), prefix + ": accumulation")
                    self.check(row["optimizer_step_after"] == (event - 1) * 128 + (number + 1) // 2, prefix + ": optimizer continuity")
                    self.check(row["learning_rate"] == (3e-4 if number % 2 else None), prefix + ": fixed LR")
                    self.check(row["replay"] is (row["record_id"] not in current), prefix + ": replay label")
                    draws = row["sampled_anchors"]
                    self.check(len(draws) == 1 and 0 < len(draws[0]["positions"][0]) <= 512, prefix + ": anchor draw")
                    self.check(row["active_anchors"] == sum(draws[0]["keep_mask"][0]), prefix + ": active anchors")
                    self.check(all(np.isfinite(float(value)) for value in row["metrics"].values()), prefix + ": finite metrics")
                    totals.update(presentations=1, optimizer_steps=int(row["optimizer_stepped"]), replay_presentations=int(row["replay"]),
                                  prompt_tokens=row["prompt_tokens"], response_tokens=row["response_tokens"], active_anchors=row["active_anchors"])
                for index in (0, 1):
                    self.check(sorted(r["record_id"] for r in records[index * 128:(index + 1) * 128]) == sorted(expected[index]),
                               f"{arm}/{event}: exact exposure pass{index}")
                boundary = self.read(f"{base}/boundaries/event_{event:03d}.json")
                self.check(boundary["status"] == "passed" and boundary["arm"] == arm and boundary["event"] == event, f"{arm}/{event}: boundary")
        self.check(totals["presentations"] == 6144 and totals["optimizer_steps"] == 3072 and totals["replay_presentations"] == 448, "Training/replay budget")
        neutral = self.read("campaign_001/events/event_001/NEUTRAL_STATE.json")
        self.check(neutral["status"] == "passed" and set(neutral["comparisons"]) == {"U", "R"}, "Neutral event1 state comparison")
        return dict(serving_work=serving, training_workers=workers, exposure=dict(totals),
                    limits="Compact presentations/work/boundary receipts; model and Adam tensors are not re-decoded.")

    def future_serving(self):
        events, all_calls = [], {arm: 0 for arm in ARMS}
        for event in range(1, 9):
            folder = f"campaign_001/events/event_{event:03d}"
            rows = self.read(folder + "/MEASUREMENTS.jsonl", lines=True)
            requests = self.inputs["training"][(event - 1) * 128:event * 128]
            lookup = {r["request_id"]: r for r in requests}
            order = ARMS[(event - 1) % 3:] + ARMS[:(event - 1) % 3]
            checkpoints = ({arm: self.plan["draft"] for arm in ARMS} if event == 1 else
                           self.read(f"campaign_001/events/event_{event-1:03d}/COMPLETE.json")["checkpoints"])
            for arm in ARMS:
                publication = self.read(folder + "/PUBLICATION_" + arm + ".json")
                self.check(not publication["errors"], folder + ": publication " + arm)
                self.publication(publication["activation"], checkpoints[arm], folder + "/" + arm, publication["equality"])
            self.check([(r["method"], r["request_id"]) for r in rows] == [(arm, r["request_id"]) for arm in order for r in requests],
                       folder + ": complete rotated schedule")
            grouped, totals = defaultdict(dict), {arm: Counter() for arm in ARMS}
            for row in rows:
                arm, rid = row["method"], row["request_id"]
                raw = self.read(row["record_path"])
                self.check(raw["checkpoint_id"] == checkpoints[arm], folder + ": preceding arm checkpoint")
                canonical = next(iter(grouped[rid].values()), None)
                self.check(row["valid_measurement"] is True and not row["errors"], folder + ": valid row")
                counts = self.native_record(raw, lookup[rid], capture=arm == "K", canonical=canonical, label=f"event{event}/{arm}/{rid}")
                self.check(raw["record_id"] == rid and raw["request_index"] == lookup[rid]["request_index"]
                           and raw["session_id"] == f"event_{event:03d}", folder + ": request identity")
                self.same({key: counts[key] for key in COUNT_KEYS}, row, folder + ": counts")
                grouped[rid][arm] = raw
                totals[arm].update({key: counts[key] for key in COUNT_KEYS})
            if event <= 2:
                for rid, methods in grouped.items():
                    self.check(len({m["metadata"]["verification_calls"] for m in methods.values()}) == 1, folder + ": neutral counts " + rid)
            if event in (1, 3):
                parity = self.read(folder + "/PARITY.json")
                self.check([r["request_id"] for r in parity] == [r["request_id"] for r in requests[:2]], folder + ": fixed parity controls")
                for row in parity:
                    rid = row["request_id"]
                    raw = self.read(row["capture_off_path"])
                    counts = self.native_record(raw, lookup[rid], capture=False, canonical=grouped[rid]["K"], label=folder + ": parity")
                    self.check(not row["errors"] and counts["verification_calls"] == grouped[rid]["K"]["metadata"]["verification_calls"], folder + ": capture parity")
            events.append(dict(event=event, domain=DOMAINS[event - 1], method_totals={arm: dict(totals[arm]) for arm in ARMS}))
            if event >= 3:
                for arm in ARMS:
                    all_calls[arm] += totals[arm]["verification_calls"]
        return dict(events=events, future_events_3_to_8_calls=all_calls,
                    future_call_reduction_pct={a + "_vs_" + b: 100 * (1 - all_calls[a] / all_calls[b]) for a, b in PAIRS},
                    uncertainty="Descriptive only; no adaptive-trajectory bootstrap.", timing="No capture-on/off wall-time comparison.")

    def final_endpoint(self):
        folder = "campaign_001/final_evaluation"
        protocol, status = self.read(folder + "/PROTOCOL.json"), self.read(folder + "/STATUS.json")
        rows, planned = self.read(folder + "/MEASUREMENTS.jsonl", lines=True), []
        for block in range(4):
            base = ARMS if block % 2 == 0 else ("K", "R", "U")
            for repeat in range(3):
                shift = (block + repeat) % 3
                for position, arm in enumerate(base[shift:] + base[:shift]):
                    for ordinal in range(block * 64, (block + 1) * 64):
                        request = self.inputs["evaluation"][ordinal]
                        planned.append(dict(method=arm, repeat=repeat, position=position, block_ordinal=block,
                                            panel_ordinal=ordinal, request_id=request["request_id"], domain=request["domain"]))
        self.same(planned, protocol["schedule"], "Final balanced full schedule")
        self.check(len(rows) == 2304, "All2304 final rows retained")
        for key in ("attempted_generations", "returned_generations", "completed_valid_generations"):
            self.check(status[key] == 2304, "Final complete " + key)
        for key in ("invalid_generations", "unknown_generation_attempts", "unattempted_generations", "extra_generations"):
            self.check(status[key] == 0, "Final no dropped/unknown/extra " + key)
        self.check(status["status"] == "completed" and protocol["capture"] is False and protocol["extra_warmup_generations"] == 0, "Final protocol")
        self.check(protocol["boundary_check_enabled"] is True and len(status["boundary_checks"]) == 37, "All37 final resource/timing gates")
        self.same(self.read("campaign_001/events/event_008/COMPLETE.json")["checkpoints"], protocol["checkpoints"], "Final event8 exports")
        batches = list(dict.fromkeys((s["block_ordinal"], s["repeat"], s["position"]) for s in planned))
        observed = [(b["block_ordinal"], b["repeat"], b["position"]) for b in status["boundary_checks"] if b["boundary"] == "before_batch"]
        self.check(observed == batches and status["boundary_checks"][-1]["boundary"] == "after_final_batch", "Final timing-boundary chronology")
        for boundary in status["boundary_checks"]:
            timing, storage = boundary["receipt"]["timing"], boundary["receipt"]["resource"]
            self.check(timing["status"] == "passed" and not timing["other_gpu_pids"] and not timing["artifact_transfer_pids"], "Final transfer/updater quiescence")
            self.check(storage["free_bytes"] >= self.plan["storage"]["reserve_bytes"]
                       and storage["compiler_growth_bytes"] <= self.plan["storage"]["compiler_growth_bytes"]
                       and storage["compact_allocated_bytes"] <= self.plan["storage"]["compact_bytes"], "Final resource limits")
        activations = self.read(folder + "/ACTIVATIONS.jsonl", lines=True)
        expected_publications, current = [], None
        for slot in planned[::64]:
            checkpoint = protocol["checkpoints"][slot["method"]]
            if checkpoint != current:
                expected_publications.append(slot)
                current = checkpoint
        self.check(len(activations) == len(expected_publications), "Final publications match actual arm transitions")
        for activation, slot in zip(activations, expected_publications):
            self.same(slot, activation, "Final publication chronology")
            self.check(not activation["errors"], "Final publication validity")
            self.publication(activation, protocol["checkpoints"][activation["method"]], "Final activation")
        for arm in ARMS:
            weights = self.read(folder + "/WEIGHTS_" + arm + ".json")
            self.check(not weights["errors"], "Final loaded-weight receipt " + arm)
            activation = next(row for row in activations if row["method"] == arm)
            self.publication(activation, protocol["checkpoints"][arm], "Final weights " + arm, weights["native_reply"])
        grouped = defaultdict(list)
        for saved, slot in zip(rows, planned):
            self.same(slot, saved, "Final chronological slot")
            rid, arm = slot["request_id"], slot["method"]
            request = self.inputs["evaluation"][slot["panel_ordinal"]]
            raw = self.read(saved["record_path"])
            self.check(raw["checkpoint_id"] == protocol["checkpoints"][arm], "Final fixed arm checkpoint")
            label = f"{slot['panel_ordinal']:04d}_r{slot['repeat']}_{arm}"
            self.check(raw["record_id"] == label and raw["request_index"] == slot["panel_ordinal"]
                       and raw["session_id"] == "k_workload_replay_v1_final", label + ": native identity")
            counts = self.native_record(raw, request, capture=False, canonical=self.canonicals[rid], label=label)
            self.same(counts, saved, label + ": rebuilt scalars")
            self.check(saved["valid_measurement"] is True and not saved["errors"] and saved["validation_status"] == "valid", label + ": validity")
            grouped[slot["panel_ordinal"], arm].append((slot["repeat"], counts))
        paired = []
        for ordinal, request in enumerate(self.inputs["evaluation"]):
            value = dict(panel_ordinal=ordinal, request_id=request["request_id"], domain=request["domain"])
            for arm in ARMS:
                records = sorted(grouped[ordinal, arm])
                self.check([number for number, _ in records] == [0, 1, 2], f"{ordinal}/{arm}: three repeats")
                first = records[0][1]
                self.check(all(all(row[key] == first[key] for key in COUNT_KEYS) for _, row in records), f"{ordinal}/{arm}: repeat counts")
                value[arm] = {key: first[key] for key in COUNT_KEYS}
                value[arm].update({key: float(np.mean([row[key] for _, row in records])) for key in TIME_KEYS})
            paired.append(value)
        recomputed = endpoint_statistics(paired)
        self.same(recomputed, self.read("final_report/SUMMARY.json")["endpoint"], "Independent endpoint statistics")
        return recomputed

    def campaign_and_retention(self):
        status = self.read("campaign_001/STATUS.json")
        self.check(status["status"] == "completed" and status["events_completed"] == 8, "Campaign completed8events")
        self.check(status["automatic_retry"] is False, "No automatic retry")
        totals = Counter()
        for path in sorted((self.root / "campaign_001/RETIREMENTS").glob("*.json")):
            row = self.read(path)
            self.check(row["status"] == "retired" and row["archived"] is False, "Retired not archived: " + path.name)
            self.check(bool(row["retained_evidence"]) and bool(row["reason"]), "Retirement evidence/reason: " + path.name)
            totals.update(paths=1, logical_bytes=row["bytes"], recorded_allocated_bytes=row["allocated_bytes"])
        return dict(status=status, retirements=dict(totals), limits="No remote tensor/existence audit; no deleted tensors reconstructed.")


def fallacies(valid):
    scope = "Assessed against reconstructed records" if valid else "Empirical assessment pending valid complete evidence"
    notes = [
        ("Simpson's paradox", "Pooled/domain ratios and gross harms remain visible; inspect direction disagreements."),
        ("Ecological fallacy", "Pooled ratio describes this panel, not every request; wins/regressions/ties remain explicit."),
        ("Berkson's paradox", "Fixed reused development inventories are selected populations, not deployment traffic."),
        ("Collider bias", "Selection reads past prompt descriptors, never labels/loss/outcomes; invalid records cannot be silently excluded."),
        ("Base rate neglect", "Final strata128/128 and future math256/coding512 are design choices, not population prevalence."),
        ("Regression to the mean", "Contemporaneous K/U controls and no outcome-selected samples limit regression-to-mean explanations."),
        ("Survivorship bias", "Complete schedules and attempted/unknown work are checked; incomplete campaigns get no efficacy verdict."),
        ("Look-elsewhere effect", "R/U is primary and R/K required; domain, U/K and time intervals remain descriptive/unadjusted."),
        ("Garden of forking paths", "Seed/descriptor/dose/memory/schedule/bootstrap fixed; reused panel is not confirmatory."),
        ("Correlation versus causation", "Shared K supervision and one trajectory do not establish universal causality or deployment/net speedup."),
        ("Reverse causality", "Replay records are strictly past; future serving first changes at event3 after the event2 update."),
    ]
    return [dict(name=name, severity="CAUTION", assessment=text, evidence_status=scope) for name, text in notes]


def run(root, output):
    started, audit = time.perf_counter(), Audit(root)
    for name, action in (("source_inputs", audit.inputs_and_source), ("selection_memory", audit.selection_and_memory),
                         ("training_work", audit.training_and_work), ("future_serving", audit.future_serving),
                         ("final_endpoint", audit.final_endpoint), ("campaign_retention", audit.campaign_and_retention)):
        audit.section(name, action)
    valid = not audit.errors
    result = dict(status="passed" if valid else "invalid_or_incomplete", efficacy_interpretation_allowed=valid,
        execution_commit=EXECUTION, checks=audit.checks, errors=audit.errors, sections=audit.sections,
        material_passport=dict(origin_skill="ARS experiment-agent", mode="validate", verification_status="ANALYZED",
            scope="Independent code reconstruction of compact evidence; same-team author overlap, not an independent GPU reproduction."),
        fallacy_coverage="11/11 categories; empirical conclusions conditional on complete valid evidence",
        fallacies=fallacies(valid), runtime=dict(python=platform.python_version(), numpy=np.__version__,
            wall_s=time.perf_counter() - started, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        generated_utc=datetime.now(timezone.utc).isoformat(), private_data_policy="No private prompt or response tokens emitted.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "FINAL_EVIDENCE_AUDIT.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    sources = dict(root=str(audit.root), files=audit.files,
        auditor_files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")})
    (output / "FINAL_AUDIT_INPUTS.json").write_text(json.dumps(sources, indent=2) + "\n")
    lines = ["# Track2 separate compact-evidence audit", "", "## Material Passport", "",
        "ARS experiment-agent / validate; verification status ANALYZED.", "",
        result["material_passport"]["scope"], "", "Execution source: " + EXECUTION + ".", "",
        "Status: **" + result["status"] + "**. Checks: " + str(audit.checks) + ".", "",
        "No model execution was rerun; retired feature/Adam tensors were not reconstructed. Input hashes are retained in "
        "FINAL_AUDIT_INPUTS.json. Private prompts/responses remain outside this report.", "",
        "| Section | Status | Errors |", "|---|---|---:|"]
    lines.extend(f"| {name} | {value['status']} | {value['errors']} |" for name, value in audit.sections.items())
    lines.extend(["", "Final intervals use10,000PCG64 stratified128math+128coding paired resamples. Counts appear once per request; "
        "times average three repeats. These are conditional panel intervals for one trained trajectory on a reused development panel. "
        "Future events3–8 remain descriptive; capture-on/off times are not compared.", "",
        "Fallacy scan:11/11 categories; detailed assessments and recomputed numbers are in FINAL_EVIDENCE_AUDIT.json."])
    if not valid:
        lines.extend(["", "No efficacy conclusion is permitted from this incomplete or invalid evidence.", "",
                      *["- " + error for error in audit.errors]])
    (output / "FINAL_EVIDENCE_AUDIT.md").write_text("\n".join(lines) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(args.run_root, args.output)
    print(json.dumps({key: result[key] for key in ("status", "checks", "efficacy_interpretation_allowed")}))
    raise SystemExit(0 if result["status"] == "passed" else 1)
