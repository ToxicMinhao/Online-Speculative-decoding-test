#!/usr/bin/env python3
"""Independent, standard-library reduction of private lost-repair evidence.

No production analysis/matching code is imported. No model is loaded. Missing
raw observations remain missing: an incomplete audit never decides the gate.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path


class Audit:
    def __init__(self):
        self.checks = 0
        self.errors = []
        self.missing = []
        self.files = {}
        self.boundary_rows = []
        self.boundary_sources = {}

    def check(self, condition, label, location=""):
        self.checks += 1
        if not condition:
            self.errors.append(dict(check=label, location=str(location)))
        return bool(condition)

    def close(self, actual, expected, label, location=""):
        ok = (actual is not None and expected is not None and math.isfinite(actual)
              and math.isfinite(expected) and abs(actual - expected) <= 2e-7 + 2e-6 * abs(expected))
        return self.check(ok, label, location)

    def read(self, path, *, optional=False):
        path = Path(path)
        if not path.is_file():
            if not optional:
                self.missing.append(str(path))
            return None
        try:
            content = path.read_bytes()
            self.files[str(path)] = dict(bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
            return json.loads(content)
        except (ValueError, OSError) as error:
            self.errors.append(dict(check="unreadable JSON", location=str(path), error=repr(error)))
            return None

    def table(self, path):
        path = Path(path)
        if not path.is_file():
            self.missing.append(str(path))
            return None
        try:
            content = path.read_bytes()
            self.files[str(path)] = dict(bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
            return list(csv.DictReader(content.decode().splitlines()))
        except (ValueError, OSError) as error:
            self.errors.append(dict(check="unreadable CSV", location=str(path), error=repr(error)))
            return None


def prefix_measure(block, response, stops):
    """Native acceptance is primary; canonical tokens establish real rejection."""
    start = block["output_start"]
    labels = response[start + 1:start + 16]
    for index, token in enumerate(labels):
        if token in stops:
            labels = labels[:index + 1]
            break
    proposals = block["draft_tokens"]
    mismatch = next((index for index, (got, want) in enumerate(zip(proposals, labels))
                     if got != want), None)
    accepted = block["accepted_draft_tokens"]
    valid = (0 <= accepted <= len(labels) and len(proposals) == 15
             and proposals[:accepted] == labels[:accepted])
    genuine = mismatch is not None and mismatch == accepted
    return dict(accepted=accepted, valid_slots=len(labels), valid=valid,
                genuine=genuine, j=mismatch + 1 if genuine else None)


def historical_label(measurements, parity_ok):
    result = dict(history="missing_replay", H=False, M=False, converse=False,
                  j=None, lost=0, valid=False)
    if set(measurements) != {"K4", "K5", "K6"}:
        return result
    if not parity_ok or not all(x["valid"] for x in measurements.values()):
        result["history"] = "invalid_replay"
        return result
    a, b, c = (measurements[name] for name in ("K4", "K5", "K6"))
    h = (a["genuine"] and c["genuine"] and a["accepted"] == c["accepted"]
         and b["accepted"] > c["accepted"])
    m = (a["genuine"] and b["genuine"] and c["genuine"]
         and a["accepted"] == b["accepted"] == c["accepted"])
    converse = (a["genuine"] and b["genuine"] and a["accepted"] == b["accepted"]
                and c["accepted"] > a["accepted"])
    return dict(history="H" if h else "M" if m else
                "other_rejection_history" if c["genuine"] else "all_correct_or_terminal",
                H=h, M=m, converse=converse, j=c["j"],
                lost=b["accepted"] - c["accepted"] if h else 0, valid=True,
                a4=a["accepted"], a5=b["accepted"], a6=c["accepted"],
                genuine_k6_rejection=c["genuine"], valid_proposals=c["valid_slots"])


PARITY_FIELDS = ("output_start", "anchor_token", "draft_tokens", "target_argmax",
                 "positions", "accepted_draft_tokens", "native_accepted_draft_tokens",
                 "committed_progress", "committed_tokens", "finished", "stop_clipped")


def raw_census(audit, root, requests, k6_export=None):
    derived, captures, completed = {}, {}, 0
    for request in requests:
        rid = request["request_id"]
        folder = root / "census" / rid
        capture = audit.read(folder / "K6_CAPTURE.json")
        if capture is None:
            continue
        try:
            before_errors = len(audit.errors)
            response = capture["response_ids"]
            trace = capture["metadata"]["trace"]
            captures[rid] = capture
            audit.check(capture["record_id"] == rid, "capture request identity", rid)
            if k6_export is not None:
                audit.check(capture["checkpoint_id"] == k6_export, "exact K+6 capture lineage", rid)
            audit.check(capture["prompt_ids"] == request["prompt_ids"], "capture exact input tokens", rid)
            audit.check(capture["stop_token_ids"] == request["stop_token_ids"], "capture stop identities", rid)
            audit.check(capture["max_new_tokens"] == request["max_new_tokens"], "capture cap", rid)
            audit.check(len(trace) == capture["metadata"]["verification_calls"], "capture call accounting", rid)
            stops = capture["stop_token_ids"]
            prompt_tokens = len(request["prompt_ids"])
            audit.check(prompt_tokens < 8192, "single native prompt-prefill chunk", rid)
            captured_prefill = capture.get("metadata", {}).get("capture", {}).get("prefill_chunks")
            if captured_prefill is not None:
                audit.check(captured_prefill == 1, "recorded single prompt ingestion", rid)
            ends = [prompt_tokens] + [prompt_tokens + b["output_start"] + b["committed_progress"]
                                      for b in trace]
            self_contained_boundaries = [[0, ends[0]]] + [[before, after]
                                                         for before, after in zip(ends, ends[1:])]
            audit.boundary_sources[rid] = dict(
                origin="prompt followed by each original native committed block",
                target_feature_ingestion_ranges=self_contained_boundaries,
                final_range_is_after_last_verification_not_a_new_start=True,
            )
            audit.check(0 < len(response) <= request["max_new_tokens"], "output length", rid)
            audit.check(not any(token in stops for token in response[:-1]), "no output after EOS", rid)
            eos = bool(response and response[-1] in stops)
            audit.check(capture["finish_reason"] == ("eos" if eos else "length"), "termination reason", rid)
            audit.check(eos or len(response) == request["max_new_tokens"], "cap or EOS reached", rid)
            output_start = 0
            for ordinal, original in enumerate(trace):
                where = f"{rid}/{ordinal}"
                start, progress = original["output_start"], original["committed_progress"]
                audit.check(start == output_start, "contiguous natural starts", where)
                audit.check(original["anchor_token"] == response[start], "known seed", where)
                audit.check(original["committed_tokens"] == response[start + 1:start + 1 + progress],
                            "canonical committed output", where)
                audit.check(original["positions"] == list(range(len(request["prompt_ids"]) + start,
                                                                 len(request["prompt_ids"]) + start + 16)),
                            "absolute native positions", where)
                audit.check(bool(original["finished"]) == (start + 1 + progress == len(response)),
                            "terminal round boundary", where)
                output_start += progress
                parent = prefix_measure(original, response, stops)
                audit.check(parent["valid"], "native accepted prefix", where)
                measurements = {}
                parity_ok = True
                boundary_checks = {}
                for arm in ("K6", "K4", "K5"):
                    receipt = audit.read(folder / f"{ordinal:04d}_{arm}.json")
                    if receipt is None:
                        continue
                    good = audit.check(receipt["status"] == "completed" and len(receipt["blocks"]) == 1,
                                       "single completed native replay", where + arm)
                    if not good:
                        parity_ok = False
                        continue
                    block = receipt["blocks"][0]
                    for field in ("output_start", "anchor_token", "positions"):
                        parity_ok &= audit.check(block[field] == original[field], "matched start " + field, where + arm)
                    parity_ok &= audit.check(receipt["response_ids"] == response[:len(receipt["response_ids"])],
                                             "replay canonical committed prefix", where + arm)
                    parity_ok &= audit.check(receipt["cap_unchanged"] and receipt["original_cap"] == request["max_new_tokens"],
                                             "unchanged replay output cap", where + arm)
                    parity_ok &= audit.check(receipt["native_finished"] == (len(receipt["response_ids"]) == len(response)),
                                             "replay native termination boundary", where + arm)
                    reconstruction = receipt["reconstruction"]
                    parity_ok &= audit.check(reconstruction["status"] == "passed"
                                             and reconstruction["target_prefix_cache_exact"]
                                             and reconstruction["destination_draft_rebuilt"]
                                             and not reconstruction["source_draft_cache_transplanted"],
                                             "genuine native context reconstruction", where + arm)
                    boundary_checks[arm] = audit.check(
                        reconstruction["ingestion_segments"] == ordinal + 1
                        and reconstruction["ingestion_tokens"] == prompt_tokens + start
                        and reconstruction["remaining_budget"] == request["max_new_tokens"] - start - 1,
                        "original ingestion count/coverage and remaining budget", where + arm)
                    parity_ok &= boundary_checks[arm]
                    measurements[arm] = prefix_measure(block, response, stops)
                    if arm == "K6":
                        parity_ok &= audit.check(all(original[k] == block[k] for k in PARITY_FIELDS),
                                                 "all-proposal neutral parity", where)
                        parity_ok &= audit.check(reconstruction["neutral_draft_prefix_cache_exact"],
                                                 "neutral draft KV identity", where)
                saved_parity = audit.read(folder / f"{ordinal:04d}_PARITY.json")
                if saved_parity is not None:
                    parity_ok &= audit.check(saved_parity["status"] == "passed" and not saved_parity["differences"],
                                             "retained parity receipt", where)
                else:
                    parity_ok = False
                key = (rid, start)
                audit.boundary_rows.append(dict(
                    request_id=rid, round_ordinal=ordinal + 1, canonical_start=start,
                    prompt_tokens=prompt_tokens, expected_ingestion_segments=ordinal + 1,
                    expected_ingestion_tokens=prompt_tokens + start,
                    K6_receipt_matches=boundary_checks.get("K6"),
                    K4_receipt_matches=boundary_checks.get("K4"),
                    K5_receipt_matches=boundary_checks.get("K5"),
                ))
                audit.check(key not in derived, "unique natural start", where)
                derived[key] = dict(request_id=rid, canonical_start=start,
                                    **historical_label(measurements, parity_ok))
            audit.check(output_start + 1 == len(response), "full canonical walk coverage", rid)
            audit.check(capture["metadata"]["accepted_draft_tokens"]
                        == sum(b["accepted_draft_tokens"] for b in trace), "accepted-token accounting", rid)
            if len(audit.errors) == before_errors and all(derived[(rid, b["output_start"])]["valid"] for b in trace):
                completed += 1
        except (KeyError, TypeError, IndexError, ValueError) as error:
            audit.errors.append(dict(check="malformed raw capture/replay", location=rid, error=repr(error)))
    stored = audit.read(root / "census/CENSUS.json")
    if stored is not None:
        audit.check(len(stored) == len(derived), "census natural-start denominator")
        for row in stored:
            key = (row["request_id"], row["canonical_start"])
            if not audit.check(key in derived, "census start has raw capture", key):
                continue
            ours = derived[key]
            if ours["valid"]:
                for field in ("a4", "a5", "a6", "genuine_k6_rejection", "valid_proposals"):
                    audit.check(row[field] == ours[field], "census field " + field, key)
    return derived, captures, completed


def tie_key(rid, start):
    payload = json.dumps([20261114, str(rid), int(start)], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def support(rows):
    return dict(occurrences=len(rows), distinct_anchors=len({(r["request_id"], r["canonical_start"]) for r in rows}),
                distinct_requests=len({r["request_id"] for r in rows}))


def training_reduction(audit, root, requests, histories, captures):
    final = audit.read(root / "training/RESULT.json")
    schedule = audit.read(root / "training/SCHEDULE.json")
    schedule_rows = {r["presentation_id"]: r for r in schedule.get("records", [])} if schedule else {}
    if schedule is not None:
        audit.check(schedule["passes"] == 2 and schedule["response_presentations"] == 256
                    and schedule["sampler_draws"] == 256 and len(schedule_rows) == 256,
                    "complete two-pass disposable schedule")
        audit.check(schedule["model_forwards"] == schedule["optimizer_steps"] == 0,
                    "sampling does not execute a model/update")
    raw_rows, presentations = [], []
    if final is not None:
        raw_rows, presentations = final.get("training_rows", []), final.get("presentation_rows", [])
    else:
        for path in sorted((root / "training/presentations").glob("*.json")):
            item = audit.read(path)
            if item is not None:
                raw_rows.extend(item.get("training_rows", []))
                if item.get("presentation") is not None:
                    presentations.append(item["presentation"])
    by_presentation = defaultdict(list)
    for row in raw_rows:
        by_presentation[row["presentation_id"]].append(row)
    pids, seen, joined = set(), set(), []
    counts = Counter()
    passes = defaultdict(set)
    native_denominator = selected_denominator = serialized_mass = 0.0
    for presentation in presentations:
        try:
            pid, rid = presentation["presentation_id"], presentation["request_id"]
            audit.check(pid not in pids, "unique presentation", pid)
            pids.add(pid)
            counts[rid] += 1
            passes[rid].add(presentation["pass_index"])
            audit.check(presentation["pass_index"] == pid // 128, "native pass boundary", pid)
            rows = by_presentation[pid]
            audit.check(len(rows) == presentation["expected_anchor_count"], "anchor accounting", pid)
            audit.check(presentation["backwards"] == presentation["optimizer_steps"] == 0,
                        "read-only presentation", pid)
            binding = schedule_rows.get(pid)
            if audit.check(binding is not None, "retained scheduled presentation", pid):
                audit.check(binding["record_id"] == rid and binding["pass_index"] == presentation["pass_index"]
                            and binding["sampled_anchors"] == presentation["sampled_anchors"],
                            "exact presentation order and anchor bindings", pid)
            sampled = presentation["sampled_anchors"][0]
            positions, kept = sampled["positions"][0], sampled["keep_mask"][0]
            audit.check(len(positions) == len(kept) == presentation["sampled_anchor_slots"],
                        "sampler slot geometry", pid)
            audit.check(sum(kept) == len(rows), "kept sampled anchors", pid)
            native_denominator += presentation["first_error_weight"]
            selected_denominator += presentation["selected_weight"]
            response_mass = math.fsum(r["first_error_weight"] for r in rows)
            serialized_mass += response_mass
            audit.close(response_mass, presentation["first_error_weight"], "native first-error mass", pid)
            w1, w2 = presentation["ordinary_low_weight"], presentation["ordinary_high_weight"]
            c1, c2 = ((1 - .5 * w2 / (w1 + 2 * w2), 1 + .5 * w1 / (w1 + 2 * w2))
                      if w1 > 0 and w2 > 0 else (1., 1.))
            audit.close(presentation["low_coefficient"], c1, "native low coefficient", pid)
            audit.close(presentation["high_coefficient"], c2, "native high coefficient", pid)
            ordinary_classes = defaultdict(float)
            expected_selected = 0.0
            for row in rows:
                key = (rid, row["canonical_start"])
                occurrence = (pid, *key)
                audit.check(occurrence not in seen, "unique sampled occurrence", occurrence)
                seen.add(occurrence)
                audit.check(row["request_id"] == rid, "presentation response identity", occurrence)
                block = row["anchor_ordinal"]
                audit.check(kept[block] and row["kept"] and row["sequence_start"] == positions[block],
                            "retained anchor occurrence binding", occurrence)
                if rid in captures:
                    audit.check(row["canonical_start"] == row["sequence_start"] - len(captures[rid]["prompt_ids"]),
                                "absolute-to-canonical anchor mapping", occurrence)
                    response = captures[rid]["response_ids"]
                    stops = captures[rid]["stop_token_ids"]
                    available = response[row["canonical_start"] + 1:row["canonical_start"] + 16]
                    for index, token in enumerate(available):
                        if token in stops:
                            available = available[:index + 1]
                            break
                    audit.check(row["valid_proposals"] == len(available), "training EOS/cap validity", occurrence)
                j = row["j"]
                audit.check(j is None or 1 <= j <= row["valid_proposals"] <= 15,
                            "valid first-error slot", occurrence)
                selected_count = j if j is not None else row["valid_proposals"]
                expected_selected += math.fsum(math.exp(-(i - 1) / 7) for i in range(1, selected_count + 1))
                if j is not None:
                    audit.check(row["correct_prefix"] == j - 1, "correct-prefix before error", occurrence)
                    audit.check(math.isfinite(row["wrong_margin"]) and row["wrong_margin"] >= 0,
                                "finite nonnegative wrong margin", occurrence)
                    audit.check(row["wrong_at_tie"] == (row["wrong_margin"] == 0),
                                "native wrong-tie flag", occurrence)
                    audit.close(row["ordinary_first_error_weight"], math.exp(-(j - 1) / 7),
                                "native decay weight", occurrence)
                    native_class = row["native_priority"]
                    ordinary_classes[native_class] += row["ordinary_first_error_weight"]
                    coefficient = c2 if native_class == "high" else c1
                    audit.close(row["native_coefficient"], coefficient, "per-anchor native coefficient", occurrence)
                    audit.close(row["first_error_weight"], row["ordinary_first_error_weight"] * coefficient,
                                "per-anchor weighted native mass", occurrence)
                    audit.check(row["priority"] == (native_class if row["successor_valid"] else "ineligible"),
                                "successor eligibility class", occurrence)
                    if rid in captures:
                        audit.check(row["canonical_token"] == captures[rid]["response_ids"][row["canonical_start"] + j]
                                    and row["wrong_token"] != row["canonical_token"],
                                    "training mismatch canonical binding", occurrence)
                else:
                    audit.check(row["first_error_weight"] == 0, "zero no-error mass", occurrence)
                history = histories.get(key)
                label = history["history"] if history and history["valid"] else None
                agreed = label if label in ("H", "M") and j == history["j"] else None
                joined.append(dict(row, audit_history=label, agreed=agreed))
            audit.close(ordinary_classes["low"], w1, "ordinary low response weight", pid)
            audit.close(ordinary_classes["high"], w2, "ordinary high response weight", pid)
            audit.close(expected_selected, presentation["selected_weight"], "selected-position denominator", pid)
        except (KeyError, TypeError, IndexError, ValueError) as error:
            audit.errors.append(dict(check="malformed training receipt", location=str(presentation.get("presentation_id")), error=repr(error)))
    strata = defaultdict(lambda: dict(H=[], M=[]))
    for row in joined:
        if row["agreed"]:
            strata[(row["presentation_id"], row["request_id"], row["j"], row["priority"])][row["agreed"]].append(row)
    pairs, comparable = [], []
    for stratum, members in strata.items():
        unused = list(members["M"])
        if unused:
            comparable.extend(members["H"])
        for h in sorted(members["H"], key=lambda row: tie_key(row["request_id"], row["canonical_start"])):
            if not unused:
                break
            distances = [(abs(m["wrong_margin"] - h["wrong_margin"]), tie_key(m["request_id"], m["canonical_start"]), index)
                         for index, m in enumerate(unused)]
            m = unused.pop(min(distances)[2])
            pairs.append(dict(presentation_id=stratum[0], request_id=stratum[1], j=stratum[2], priority=stratum[3],
                              H_start=h["canonical_start"], M_start=m["canonical_start"],
                              H_margin=h["wrong_margin"], M_margin=m["wrong_margin"],
                              H_weight=h["first_error_weight"], M_weight=m["first_error_weight"],
                              projected_weight=.25 * min(h["first_error_weight"], m["first_error_weight"])))
    h_native = [r for r in histories.values() if r["H"]]
    h_scheduled = [r for r in joined if r["audit_history"] == "H"]
    h_same = [r for r in joined if r["agreed"] == "H"]
    matched_ids = {(p["presentation_id"], p["request_id"], p["H_start"]) for p in pairs}
    h_matched = [r for r in h_same if (r["presentation_id"], r["request_id"], r["canonical_start"]) in matched_ids]
    complete = (final is not None and final.get("status") == "completed" and len(presentations) == 256
                and all(counts[r["request_id"]] == 2 and passes[r["request_id"]] == {0, 1} for r in requests)
                and pids == set(range(256)))
    return dict(complete=complete, observed_presentations=len(presentations), native_first_error_weight=native_denominator,
                selected_weight=selected_denominator, serialized_anchor_first_error_weight=serialized_mass,
                native_H_weight_before_matching=math.fsum(r["first_error_weight"] for r in h_same),
                projected_matched_weight=math.fsum(p["projected_weight"] for p in pairs),
                H_unmatched_occurrences=len(h_same) - len(pairs),
                exposure_funnel=dict(native_replay=support(h_native), scheduled=support(h_scheduled),
                                     same_j=support(h_same), comparable_M=support(comparable), matched=support(h_matched)),
                pairs=pairs, per_presentation=presentations)


def check_public_privacy(audit, root):
    """Inspect only intended publication artifacts, without copying raw content."""
    forbidden = {"prompt_ids", "response_ids", "draft_tokens", "target_argmax", "canonical_token",
                 "wrong_token", "input_ids", "covered_token_ids", "hidden_states",
                 "target_last_hidden_states", "messages", "question"}
    def walk(value, location):
        if isinstance(value, dict):
            for key, child in value.items():
                audit.check(key not in forbidden, "no private payload field in public JSON", location + "/" + key)
                walk(child, location + "/" + key)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, location + "/" + str(index))
    for path in sorted((root / "final_report").glob("*")):
        if path.suffix == ".json":
            value = audit.read(path)
            if value is not None:
                walk(value, path.name)
        elif path.suffix == ".csv":
            rows = audit.table(path)
            if rows:
                audit.check(not (set(rows[0]) & forbidden), "no raw token/text columns in public CSV", path.name)
        elif path.suffix == ".md":
            text = path.read_text()
            audit.check(not any('"' + key + '":' in text for key in forbidden),
                        "no serialized private payloads in report prose", path.name)
    return dict(scope="public report JSON/CSV schemas and serialized-payload scan",
                full_semantic_text_redaction_audit=False)


def execute(campaign_root, plan_path):
    audit = Audit()
    root = Path(campaign_root)
    plan = audit.read(plan_path)
    if plan is None:
        return dict(status="INCOMPLETE", gate_decision="INCOMPLETE", missing=audit.missing, errors=audit.errors)
    requests = plan["requests"]
    histories, captures, complete_requests = raw_census(
        audit, root, requests, plan.get("checkpoints", {}).get("K6", {}).get("export"))
    training = training_reduction(audit, root, requests, histories, captures)
    status = audit.read(root / "STATUS.json") or {}
    budget = audit.read(root / "BUDGET.json")
    execution = audit.read(root / "RESULT.json")
    if budget is not None:
        audit.check(budget["gpu_seconds_limit"] == 14400, "four-hour aggregate budget")
        audit.check(all(budget["attempted"].get(name, 0) == 0 for name in
                        ("optimizer_step", "backward", "timing_sweep")), "zero forbidden work")
        if status.get("status") == "completed":
            audit.check(not budget["failed"] and not budget["unknown"], "complete work accounting")
            audit.check(budget["completed"].get("feedback_capture") == 128
                        and budget["completed"].get("training_forward") == 256,
                        "full capture and read-only presentation budget")
    if execution is not None:
        audit.check(execution["optimizer_steps"] == execution["backward_calls"] == execution["timed_endpoints"] == 0,
                    "zero optimizer/backward/timed endpoint result")
    summary = audit.read(root / "final_report/SUMMARY.json")
    h = [r for r in histories.values() if r["H"]]
    m = [r for r in histories.values() if r["M"]]
    rejections = [r for r in histories.values() if r["valid"] and r["genuine_k6_rejection"]]
    measured = dict(observed_natural_starts=len(histories), H_observed_count=len(h), M_observed_count=len(m),
                    converse_observed_count=sum(r["converse"] for r in histories.values()),
                    H_contributing_requests=len({r["request_id"] for r in h}),
                    H_lost_prefix_total=sum(r["lost"] for r in h), observed_genuine_k6_rejections=len(rejections),
                    **{k: training[k] for k in ("observed_presentations", "native_first_error_weight", "selected_weight",
                                              "serialized_anchor_first_error_weight", "native_H_weight_before_matching",
                                              "projected_matched_weight", "H_unmatched_occurrences", "exposure_funnel")})
    if summary is not None:
        for key, value in measured.items():
            if isinstance(value, dict) or isinstance(value, int):
                audit.check(summary.get(key) == value, "report summary " + key)
            elif training["observed_presentations"]:
                audit.close(summary.get(key), value, "report summary " + key)
        audit.check(summary.get("provenance", {}).get("execution_commit") == plan["execution_commit"],
                    "reported scientific execution commit")
    table = audit.table(root / "final_report/MATCHED_PAIRS.csv")
    if table is not None:
        mapping = {(int(r["presentation_id"]), r["request_id"], int(r["H_start"])): r for r in table}
        audit.check(len(table) == len(mapping) == len(training["pairs"]), "matched-pair count")
        for pair in training["pairs"]:
            key = (pair["presentation_id"], pair["request_id"], pair["H_start"])
            row = mapping.get(key)
            if audit.check(row is not None, "matched H occurrence", key):
                audit.check(int(row["M_start"]) == pair["M_start"] and int(row["j"]) == pair["j"]
                            and row["priority"] == pair["priority"], "independent nearest-margin matching", key)
                audit.close(float(row["projected_weight"]), pair["projected_weight"], "paper-only pair weight", key)
    per_request = audit.table(root / "final_report/PER_REQUEST.csv")
    if per_request is not None:
        audit.check(len(per_request) == len(requests), "per-request row count")
        for row in per_request:
            rid = row["request_id"]
            starts = [r for r in histories.values() if r["request_id"] == rid]
            expected = dict(natural_starts=len(starts), H=sum(r["H"] for r in starts), M=sum(r["M"] for r in starts),
                            converse=sum(r["converse"] for r in starts), H_lost_prefix_occurrences=sum(r["lost"] for r in starts))
            for key, value in expected.items():
                audit.check(int(row[key]) == value, "per-request " + key, rid)
    complete = (complete_requests == 128 and len(captures) == 128 and training["complete"]
                and status.get("census_complete") and status.get("training_complete")
                and not audit.errors and not audit.missing)
    native_mass = training["native_first_error_weight"]
    fraction = training["projected_matched_weight"] / native_mass if native_mass else None
    values = [training["exposure_funnel"]["matched"]["distinct_anchors"],
              training["exposure_funnel"]["matched"]["distinct_requests"], fraction]
    gate = [dict(value=value, threshold=threshold, observed_threshold_met=value is not None and value >= threshold,
                 passed=(value is not None and value >= threshold) if complete else None)
            for value, threshold in zip(values, [32, 16, .0025])]
    decision = "PASS" if complete and all(r["passed"] for r in gate) else "FAIL" if complete else "INCOMPLETE"
    if summary is not None and complete:
        audit.check(summary["gate_decision"] == decision, "gate decision")
    privacy = check_public_privacy(audit, root)
    if audit.errors or audit.missing:
        complete, decision = False, "INCOMPLETE"
        for item in gate:
            item["passed"] = None
    return dict(status="VERIFIED" if complete and not audit.errors else "INCOMPLETE",
                gate_decision=decision if not audit.errors else "INCOMPLETE", gate=gate,
                checks=audit.checks, errors=audit.errors, missing=audit.missing,
                capture_requests=len(captures), valid_complete_replay_requests=complete_requests,
                unknown_natural_denominator_requests=128 - len(captures),
                full_natural_denominator=len(histories) if complete_requests == 128 else None,
                measured=measured, input_files=audit.files,
                ingestion_boundary_checks=audit.boundary_rows,
                original_ingestion_boundaries=audit.boundary_sources,
                cost_receipts=dict(budget=budget, phases=execution.get("phase_costs") if execution else None),
                public_privacy=privacy,
                limitations=["Independent arithmetic and token-record audit; no independent model execution.",
                             "Training argmax/priority observations originate from retained native forwards; this audit checks their bindings, decay/coefficient arithmetic, sums and joins without loading logits/tensors.",
                             "Original ingestion ranges are independently derived from prompt/commit records. Reconstruction receipts validate segment count and total coverage; individual reconstructed segment ranges are not serialized, so their exact execution is supported by constructor assertions rather than independent raw range observations."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = execute(args.campaign_root, args.plan)
    except (KeyError, TypeError, ValueError, OSError) as error:
        result = dict(status="INCOMPLETE", gate_decision="INCOMPLETE",
                      errors=[dict(check="unsupported or malformed retained evidence", error=repr(error))])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    boundary_rows = result.get("ingestion_boundary_checks", [])
    if boundary_rows:
        with args.output.with_name(args.output.stem + "_INGESTION_BOUNDARIES.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(boundary_rows[0]))
            writer.writeheader()
            writer.writerows(boundary_rows)
    print(json.dumps({key: result.get(key) for key in ("status", "gate_decision", "checks", "capture_requests")}
                     | dict(errors=len(result.get("errors", [])), missing=len(result.get("missing", [])))))
    return 0 if result["status"] == "VERIFIED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
