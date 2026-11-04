"""Strict raw-record report for the bounded four-event VFD second-pass screen."""

from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path

from online_feedback.storage import write_json
from .contract import ARMS, PASS2, BUDGET, logical_steps
from .evaluation import audit_stage, audit_canonicals
from .statistics import reduce_final, summarize


def load(path):
    return json.loads(Path(path).read_text())


def csv_write(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(
            {
                key: (
                    json.dumps(value, separators=(",", ":"))
                    if isinstance(value, (dict, list))
                    else value
                )
                for key, value in row.items()
            }
            for row in rows
        )


def score_evidence(event_root, arm, event, selection, order):
    """Recalculate every selector scalar from the retained compact snapshots."""
    from online_feedback.vfd_scoring import score_response_pair, select_responses

    rows = []
    if arm in ("STATIC64", "PROGRESS64"):
        saved = load(event_root / arm / "SCORING.json")
        source = event_root / ("SHARED" if event == 1 else arm)
        before = load(source / "SCORES_BEFORE.json")
        after = load(source / "SCORES_AFTER.json") if arm == "PROGRESS64" else {}
        if set(before) != set(order) or (after and set(after) != set(order)):
            raise ValueError(
                "Score snapshots do not contain all current-event responses"
            )
        rows = [score_response_pair(before[key], after.get(key)) for key in before]
        if rows != saved:
            raise ValueError(
                "Saved selector scores differ from compact before/after evidence"
            )
    expected = select_responses(rows, order, event_id=event, policy=arm)
    for key, value in expected.items():
        if key != "selected_response_lengths" and selection.get(key) != value:
            raise ValueError(
                "Saved selector differs from the fixed hash/score rule: " + key
            )
    return rows


def worker_work(folder, result, presentations, scoring):
    from experiments.ok_coding.campaign import read_work, require_complete_work

    limits = dict(
        training_presentations=presentations,
        backwards=presentations,
        optimizer_steps=presentations // 2,
        scoring_forwards=scoring,
    )
    work = read_work(folder / "WORK.jsonl", limits)
    require_complete_work(work, limits)
    if result["status"] != "completed" or work != result["work"]:
        raise ValueError("Worker completion differs from its durable action ledger")
    return work


def audit_training(root, plan, status):
    """Reconstruct physical and logical work from every committed pass receipt."""
    root = Path(root)
    errors, pass_rows, selections, scores, blocks, events = [], [], [], [], [], []
    physical_steps = physical_presentations = logical_total = 0
    costs = {arm: 0.0 for arm in ARMS}
    try:
        shared = load(root / "events/event_001/SHARED/pass_0.json")
        physical_steps += shared["optimizer_steps"]
        physical_presentations += shared["response_presentations"]
        shared_result = load(root / "events/event_001/SHARED/RESULT.json")
        worker_work(root / "events/event_001/SHARED", shared_result, 128, 256)
        if shared_result["pass1"] != shared:
            raise ValueError(
                "Shared pass receipt differs from its completed worker result"
            )
        scoring_forwards = 256
        prior_steps = {arm: 0 for arm in ARMS}
        for event in range(1, 5):
            event_root = root / f"events/event_{event:03d}"
            results = {arm: load(event_root / arm / "RESULT.json") for arm in ARMS}
            full_first = results["FULL"]["pass1"]
            full_second = results["FULL"]["pass2"]
            order1, order2 = full_first["executed_order"], full_second["executed_order"]
            if (
                len(set(order1)) != 128
                or len(order1) != 128
                or sorted(order1) != sorted(order2)
            ):
                raise ValueError(
                    f"event {event}: FULL permutations differ from the response inventory"
                )
            event_selections = {}
            for arm, result in results.items():
                first, second = result["pass1"], result["pass2"]
                initial = result["initial_state"]
                if (
                    initial["status"] != "passed"
                    or initial["exact_resume"] is not True
                    or initial["global_step"]
                    != (64 if event == 1 else prior_steps[arm])
                ):
                    raise ValueError("Policy continuation/fork restoration failed")
                count = PASS2[arm] if event == 1 else 128 + PASS2[arm]
                score_count = (
                    0
                    if event == 1
                    else {"STATIC64": 128, "PROGRESS64": 256}.get(arm, 0)
                )
                worker_work(event_root / arm, result, count, score_count)
                scoring_forwards += score_count
                if first["executed_order"] != order1 or (
                    event == 1 and first != shared
                ):
                    raise ValueError(
                        f"event {event}/{arm}: first-pass shared order or fork changed"
                    )
                if arm == "ONCE" and second is not None:
                    raise ValueError("ONCE has a second pass")
                selected = result["selection"]["selected_ids"]
                induced = [
                    record_id for record_id in order2 if record_id in set(selected)
                ]
                if (
                    len(selected) != PASS2[arm]
                    or len(set(selected)) != len(selected)
                    or not set(selected).issubset(order2)
                    or result["selection"]["training_order"] != induced
                    or (second is not None and second["executed_order"] != induced)
                ):
                    raise ValueError(
                        f"event {event}/{arm}: selection count/order invalid"
                    )
                event_selections[arm] = set(selected)
                for pass_index, receipt in enumerate((first, second)):
                    if receipt is None:
                        continue
                    count = 128 if pass_index == 0 else PASS2[arm]
                    if (
                        receipt["response_presentations"] != count
                        or receipt["optimizer_steps"] != count // 2
                        or receipt["start_step"] != prior_steps[arm]
                        or receipt["end_step"] != prior_steps[arm] + count // 2
                        or len(receipt["records"]) != count
                        or [r["record_id"] for r in receipt["records"]]
                        != receipt["executed_order"]
                        or sum(bool(r["optimizer_stepped"]) for r in receipt["records"])
                        != count // 2
                        or not math.isfinite(receipt["training_s"])
                        or receipt["training_s"] < 0
                    ):
                        raise ValueError(
                            f"event {event}/{arm}/pass{pass_index+1}: committed accounting invalid"
                        )
                    prior_steps[arm] = receipt["end_step"]
                    logical_total += receipt["optimizer_steps"]
                    physical = not (event == 1 and pass_index == 0)
                    if physical:
                        physical_steps += receipt["optimizer_steps"]
                        physical_presentations += receipt["response_presentations"]
                    pass_rows.append(
                        dict(
                            event=event,
                            arm=arm,
                            pass_number=pass_index + 1,
                            optimizer_steps=receipt["optimizer_steps"],
                            response_presentations=count,
                            start_step=receipt["start_step"],
                            end_step=receipt["end_step"],
                            training_s=receipt["training_s"],
                            pass_wall_s=receipt["pass_wall_s"],
                            physical_receipt=(
                                "shared_event1_first_pass"
                                if not physical
                                else "policy_pass"
                            ),
                        )
                    )
                if (
                    result["state"]["status"] != "passed"
                    or result["state"]["global_step"] != prior_steps[arm]
                    or result["state"]["presentations"] != 2 * prior_steps[arm]
                ):
                    raise ValueError(
                        "Persisted policy state counters differ from committed passes"
                    )
                scored = score_evidence(
                    event_root, arm, event, result["selection"], order2
                )
                worker_costs = result["costs"]
                if any(
                    type(value) not in (int, float)
                    or not math.isfinite(value)
                    or value < 0
                    for value in worker_costs.values()
                ):
                    raise ValueError("Invalid separately measured worker cost")
                expected_training = first["pass_wall_s"] + (
                    second["pass_wall_s"] if second else 0.0
                )
                expected_compute = first["training_s"] + (
                    second["training_s"] if second else 0.0
                )
                expected_scoring = sum(
                    worker_costs[key]
                    for key in (
                        "scoring_before_s",
                        "scoring_after_s",
                        "scoring_reduction_s",
                    )
                )
                checks = (
                    (worker_costs["training_s"], expected_training),
                    (worker_costs["training_compute_s"], expected_compute),
                    (worker_costs["scoring_s"], expected_scoring),
                    (
                        result["logical_updater_s"],
                        expected_training
                        + expected_scoring
                        + worker_costs["selection_s"],
                    ),
                )
                if any(
                    not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-6)
                    for actual, expected in checks
                ):
                    raise ValueError(
                        "Logical updater worker timing does not reconcile with its components"
                    )
                costs[arm] += result["logical_updater_s"]
                selections.append(
                    dict(
                        event=event,
                        arm=arm,
                        **result["selection"],
                        logical_updater_s=result["logical_updater_s"],
                        costs=result["costs"],
                    )
                )
                for row in scored:
                    scores.append(
                        dict(
                            event=event,
                            arm=arm,
                            **{
                                k: row[k]
                                for k in (
                                    "response_id",
                                    "response_length",
                                    "context_length",
                                    "eligible_anchors",
                                    "sampled_anchors",
                                    "static_score",
                                    "progress_score",
                                    "has_after_snapshot",
                                )
                            },
                            **row["counts"],
                        )
                    )
                    blocks.extend(
                        dict(
                            event=event,
                            arm=arm,
                            response_id=row["response_id"],
                            **block,
                        )
                        for block in row["blocks"]
                    )
            events.append(
                dict(
                    event=event,
                    static_progress_selection_overlap=len(
                        event_selections["STATIC64"] & event_selections["PROGRESS64"]
                    ),
                    comparable_state=event == 1,
                )
            )
        if prior_steps != {arm: logical_steps(arm, 4) for arm in ARMS}:
            raise ValueError("Final per-policy logical optimizer counters differ")
        if (physical_steps, physical_presentations, logical_total) != (
            1664,
            3328,
            1920,
        ):
            raise ValueError("Physical/shared versus logical work does not reconcile")
        if scoring_forwards != status["work"]["scoring_forwards"]:
            raise ValueError("Scoring forward total differs from worker action ledgers")
        for arm in ARMS:
            if not math.isclose(
                costs[arm],
                status["logical_updater_seconds"][arm],
                rel_tol=1e-10,
                abs_tol=1e-6,
            ):
                raise ValueError("Logical updater costs do not reconcile: " + arm)
    except (OSError, KeyError, TypeError, ValueError, IndexError, ImportError) as error:
        errors.append(repr(error))
    return dict(
        status="passed" if not errors else "invalid_or_incomplete",
        errors=errors,
        physical_optimizer_steps=physical_steps,
        physical_training_presentations=physical_presentations,
        logical_optimizer_steps=logical_total,
        logical_training_presentations=2 * logical_total,
        logical_updater_seconds=costs,
        passes=pass_rows,
        selections=selections,
        scores=scores,
        blocks=blocks,
        event_selection_overlap=events,
        model_work=0,
    )


def audit_qualification(folder, status):
    from experiments.ok_coding.campaign import read_work
    from online_feedback.vfd_scoring import score_response_pair

    limits = dict(
        training_presentations=4, backwards=4, optimizer_steps=2, scoring_forwards=8
    )
    result = load(folder / "RESULT.json")
    work = read_work(folder / "WORK.jsonl", limits)
    if (
        result["status"] != "passed"
        or result["fork_equality"] is not True
        or result["restoration_after_discard"]["status"] != "passed"
        or work != result["work"]
        or work["attempts"] != work["completed"]
        or any(work["unknown"].values())
        or result["extra_generations"] != 0
    ):
        raise ValueError("Disposable qualification integrity or work accounting failed")
    if any(
        work["completed"][key] != value
        for key, value in status["qualification_work"].items()
    ):
        raise ValueError(
            "Qualification work was combined with scientific work or miscounted"
        )
    if not result["before"] or set(result["before"]) != set(result["after"]):
        raise ValueError("Missing qualification before/after scoring example")
    examples = [
        score_response_pair(result["before"][key], result["after"][key])
        for key in result["before"]
    ]
    if examples != result["examples"]:
        raise ValueError("Qualification examples differ from before/after evidence")
    return dict(status="passed", work=work, examples=len(examples), model_work=0)


def budget_errors(status, missing_count):
    errors = []
    for key in (
        "feedback_generations",
        "teacher_preparations",
        "optimizer_steps",
        "training_presentations",
        "timed_generations",
    ):
        if status["work"].get(key) != BUDGET[key]:
            errors.append("work_" + key)
    for key, cap in (
        ("scoring_forwards", 1408),
        ("canonical_generations", missing_count),
    ):
        value = status["work"].get(key)
        if type(value) is not int or not 0 <= value <= cap:
            errors.append("work_" + key)
    if status["work"].get("canonical_generations") != missing_count:
        errors.append("missing_canonical_generation_count")
    for key, cap in (
        ("scoring_forwards", 8),
        ("training_presentations", 4),
        ("optimizer_steps", 2),
    ):
        value = status["qualification_work"].get(key)
        if type(value) is not int or not 0 <= value <= cap:
            errors.append("qualification_" + key)
    return errors


def report(root, output):
    """Write an incomplete receipt on failure; never analyze a surviving subset."""
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    receipt = dict(status="invalid_or_incomplete", errors=[], model_work=0)
    try:
        plan = load(root / "sealed/PLAN.json")
        status = load(root / "campaign_001/STATUS.json")
        requests = plan["evaluation_requests"]
        checkpoints = plan["final_checkpoints"]
        missing = load(root / "sealed/REFERENCE_MISSING.json")["count"]
        receipt["errors"].extend(budget_errors(status, missing))
        if status["status"] != "completed":
            receipt["errors"].append("campaign_not_completed")
        qualification = audit_qualification(root / "campaign_001/qualification", status)
        write_json(output / "QUALIFICATION_AUDIT.json", qualification)
        training = audit_training(root / "campaign_001", plan, status)
        write_json(
            output / "TRAINING_AUDIT.json",
            {
                k: v
                for k, v in training.items()
                if k not in ("passes", "selections", "scores", "blocks")
            },
        )
        if training["status"] != "passed":
            receipt["errors"].extend(training["errors"])
        canonical = load(root / "campaign_001/canonicals/CANONICALS.json")
        binding = load(root / "sealed/REFERENCE_REUSE_AUDIT.json")["binding_digest"]
        references = audit_canonicals(
            requests, canonical, binding_digest=binding, evidence_root=root
        )
        write_json(output / "REFERENCE_AUDIT.json", references)
        if references["status"] != "passed":
            receipt["errors"].append("canonical_reference_audit_failed")
        endpoint = audit_stage(
            root / "campaign_001/evaluation/final",
            requests,
            checkpoints,
            canonicals=canonical,
            evidence_root=root,
        )
        write_json(
            output / "ENDPOINT_AUDIT.json",
            {k: v for k, v in endpoint.items() if k != "rows"},
        )
        if endpoint["status"] != "passed":
            receipt["errors"].extend(endpoint["errors"])
        if receipt["errors"]:
            raise ValueError("Required raw validity/accounting checks failed")
        summary = summarize(
            endpoint["rows"], requests, status["logical_updater_seconds"]
        )
        summary.update(
            work=status["work"],
            qualification_work=status["qualification_work"],
            costs=status["costs"],
            event_selection_overlap=training["event_selection_overlap"],
            physical_optimizer_steps=1664,
            logical_optimizer_steps=1920,
        )
        write_json(output / "SUMMARY.json", summary)
        paired = reduce_final(endpoint["rows"], requests)
        csv_write(
            output / "PAIRED_REQUESTS.csv",
            [
                dict(
                    request_id=row["request_id"],
                    panel_ordinal=row["panel_ordinal"],
                    **{
                        arm + "_" + key: value
                        for arm in ARMS
                        for key, value in row[arm].items()
                    },
                )
                for row in paired
            ],
        )
        columns = (
            "request_id",
            "panel_ordinal",
            "method",
            "repeat",
            "block_ordinal",
            "position",
            "verification_calls",
            "accepted_draft_tokens",
            "committed_progress",
            "output_tokens",
            "request_latency_s",
            "native_generate_s",
            "cache_maintenance_s",
            "finish_reason",
            "cap_hit",
            "checkpoint_id",
            "validation_status",
        )
        csv_write(
            output / "ALL_REPEATS.csv",
            [{key: row[key] for key in columns} for row in endpoint["rows"]],
        )
        csv_write(output / "TRAINING_PASSES.csv", training["passes"])
        csv_write(output / "SELECTIONS.csv", training["selections"])
        csv_write(output / "SCORES.csv", training["scores"])
        csv_write(output / "SCORE_BLOCKS.csv", training["blocks"])
        groups = defaultdict(list)
        for row in training["scores"]:
            groups[row["event"], row["arm"]].append(row)
        score_summary = []
        for (event, arm), rows in groups.items():
            for field in ("static_score", "progress_score"):
                values = [
                    row[field]
                    for row in rows
                    if field in row
                    and (field != "progress_score" or row["has_after_snapshot"])
                ]
                counts = Counter(values)
                if values:
                    score_summary.append(
                        dict(
                            event=event,
                            arm=arm,
                            field=field,
                            responses=len(values),
                            zero_count=sum(value == 0 for value in values),
                            zero_fraction=sum(value == 0 for value in values)
                            / len(values),
                            tied_response_count=sum(
                                count for count in counts.values() if count > 1
                            ),
                            distinct_scores=len(counts),
                        )
                    )
        csv_write(output / "SCORE_SUMMARY.csv", score_summary)
        lines = [
            "# Four-event VFD second-pass screen",
            "",
            "## Material Passport",
            "",
            "ARS experiment-agent run/validate. Status: complete and ANALYZED from retained native evidence. One training seed; no independent GPU replication.",
            "",
            "The five policies learned from the same 512 FULL-generated feedback responses. Only second-pass response selection differed. All 2,000 timed executions passed canonical token, termination, repeated-count and native-trace checks.",
            "",
            "| PROGRESS64 versus | Complete-request reduction | 95% paired interval |",
            "|---|---:|---:|",
        ]
        for arm in ("FULL", "STATIC64", "RANDOM64", "ONCE"):
            effect = summary["comparisons"]["PROGRESS64_vs_" + arm][
                "request_latency_s_reduction"
            ]
            lines.append(
                f"| {arm} | {effect['point_pct']:.4f}% | [{effect['ci95_pct'][0]:.4f}%, {effect['ci95_pct'][1]:.4f}%] |"
            )
        lines += [
            "",
            "| Policy | Verification calls | Accepted draft tokens / call | Logical updater seconds |",
            "|---|---:|---:|---:|",
        ]
        for arm in ARMS:
            total = summary["method_totals"][arm]
            lines.append(
                f"| {arm} | {total['verification_calls']} | {total['accepted_tokens_per_call']:.4f} | {status['logical_updater_seconds'][arm]:.3f} |"
            )
        decision = summary["interpretation"]
        lines += [
            "",
            f"Decision: **{decision['decision']}**. {decision['reason']}",
            f"Preferred policy: {decision['preferred_policy'] or 'none established'}.",
            "",
            decision["selector_control_caution"],
            "",
            f"Event-1 STATIC64/PROGRESS64 selection overlap: {training['event_selection_overlap'][0]['static_progress_selection_overlap']}/64. Later overlaps are descriptive because selectors rank under different policy states.",
            "",
            "Two timing repeats were averaged per request before 10,000 paired request-bootstrap draws (PCG64 seed 20261104). Repeats do not increase sample size. Timing includes native cache maintenance and excludes loading, switching and updater work. All capped responses remain included.",
            "",
            f"Physical training: 1,664 optimizer steps / 3,328 presentations. Logical standalone totals: 1,920 / 3,840. Shared preparation: 512 feedback generations and 512 teacher preparations. Missing canonical references: {missing} generations, excluded from endpoint timing. Disposable qualification costs are reported separately in SUMMARY.json.",
            "",
            "Preparation, scoring, training, persistence, export and activation costs are separate in SUMMARY.json. Logical updater comparisons charge shared event-1 work to every applicable policy and include scoring, selection and training worker wall, including packet loads and pass setup. Initialization/restoration and export are excluded; native compute timers remain separate.",
            "",
            "The panel is train-disjoint, with historical overlap retained in the sealed data manifest. One seed and four events are a screen, not seed-level replication. No training-inclusive serving-speedup or causal mechanism claim follows from cheaper updates. No automatic extension or tuning is authorized.",
        ]
        (output / "REPORT.md").write_text("\n".join(lines) + "\n")
        receipt.update(status="passed", decision=decision["decision"])
    except (
        OSError,
        KeyError,
        ValueError,
        TypeError,
        IndexError,
        RuntimeError,
        ImportError,
    ) as error:
        receipt["errors"].append(repr(error))
        (output / "REPORT.md").write_text(
            "# Four-event VFD second-pass screen\n\nStatus: **invalid or incomplete**. No efficacy estimate was calculated from a subset. See REPORT_STATUS.json for preserved failure details.\n"
        )
    write_json(output / "REPORT_STATUS.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = report(args.root, args.output)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "passed" else 1)
