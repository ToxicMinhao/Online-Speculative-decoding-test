#!/usr/bin/env python3
"""Summarize a successful separate audit without changing run artifacts."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def main(root, output):
    root, output = Path(root), Path(output)
    audit = json.loads((output / "FINAL_EVIDENCE_AUDIT.json").read_text())
    assert audit["status"] == "passed" and not audit["errors"]
    sections = {k: v["details"] for k, v in audit["sections"].items()}
    endpoint, future = sections["final_endpoint"], sections["future_serving"]
    inputs = json.loads((output / "FINAL_AUDIT_INPUTS.json").read_text())
    additional_inputs = {}

    def read(relative, lines=False):
        raw = (root / relative).read_bytes()
        additional_inputs[relative] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        return [json.loads(line) for line in raw.splitlines()] if lines else json.loads(raw)

    rows = read("campaign_001/final_evaluation/MEASUREMENTS.jsonl", True)
    repeat_totals = defaultdict(Counter)
    for row in rows:
        for key in ("verification_calls", "request_latency_s", "native_generate_s", "cache_maintenance_s"):
            repeat_totals[row["repeat"], row["method"]][key] += row[key]
    pairs = (("R", "U"), ("R", "K"), ("U", "K"))
    repeats = [dict(repeat=repeat, methods={arm: dict(repeat_totals[repeat, arm]) for arm in "KUR"},
                    request_time_reduction_pct={a + "_vs_" + b: 100 * (1 - repeat_totals[repeat, a]["request_latency_s"] /
                                                                         repeat_totals[repeat, b]["request_latency_s"])
                                                for a, b in pairs}) for repeat in range(3)]
    exposure = {}
    for arm in "KUR":
        totals = Counter()
        for event in range(1, 9):
            result = read(f"campaign_001/arms/{arm}/trainer/events/event_{event:03d}/RESULT.json")
            for row in result["records"]:
                totals.update(response_presentations=1, optimizer_steps=int(row["optimizer_stepped"]),
                              replay_presentations=int(row["replay"]), prompt_tokens=row["prompt_tokens"],
                              response_tokens=row["response_tokens"], active_anchors=row["active_anchors"])
        exposure[arm] = dict(totals)
    future_by_domain = {}
    for domain in ("math", "coding"):
        events = [row for row in future["events"] if row["event"] >= 3 and row["domain"] == domain]
        calls = {arm: sum(row["method_totals"][arm]["verification_calls"] for row in events) for arm in "KUR"}
        future_by_domain[domain] = dict(requests=128 * len(events), calls=calls,
            reduction_pct={a + "_vs_" + b: 100 * (1 - calls[a] / calls[b]) for a, b in pairs})
    parity_calls = 0
    for event in (1, 3):
        for row in read(f"campaign_001/events/event_{event:03d}/PARITY.json"):
            relative = "campaign_001/" + row["capture_off_path"].split("/campaign_001/", 1)[1]
            parity_calls += read(relative)["metadata"]["verification_calls"]
    blocks = dict(pre_update=sum(row["method_totals"][arm]["verification_calls"] for row in future["events"] for arm in "KUR"),
                  final=sum(row["verification_calls"] for row in rows), capture_parity=parity_calls)
    blocks["total"] = sum(blocks.values())
    fallacies = [
        ("Simpson's paradox", "CAUTION", "No classic all-strata reversal. R/U is +0.0920% pooled, -1.3259% math and +1.1534% coding. Opposing domain effects nearly cancel; the pooled number must not hide math harm."),
        ("Ecological fallacy", "CAUTION", "The ratio of call totals is not an average individual benefit. R/U has 101 wins, 107 regressions and 48 ties, with 241 calls saved and 229 added. R/K has 101/112/43 and 241/262."),
        ("Berkson's paradox", "CAUTION", "The fixed, filtered, previously used math/code panel is a selected population. Neither 256 panel requests nor the designed recurrence stream estimates natural conversation traffic."),
        ("Collider bias", "NOTE", "Every planned final slot and pre-update row is present and valid, so no success-conditioned subset was used. The selector uses current/past prompt descriptors; final outcomes and labels do not enter the reconstructed choices."),
        ("Base rate neglect", "CAUTION", "Final sampling is 128 math plus 128 coding requests, while future events 3-8 contain 256 math and 512 coding. Ratios weight by reference call totals; neither mix is an estimate of deployment prevalence."),
        ("Regression to the mean", "NOTE", "Contemporaneous fixed K/U controls and all prespecified events are retained. Samples were not chosen after an extreme efficacy result. Three timing repeats concern the same trained endpoints, not three training replications."),
        ("Survivorship bias", "NOTE", "All 2304 final slots, 3072 pre-update generations, four parity controls and 24 worker ledgers reconcile. No invalid, unknown or unattempted final generation is dropped; no unknown worker/root operation remains."),
        ("Look-elsewhere effect", "CAUTION", "Pooled R/U is primary; R/K is required. Neither interval excludes zero. Domain, U/K and timing intervals are descriptive and unadjusted; coding R/U alone cannot turn the overall endpoint into a positive result."),
        ("Garden of forking paths", "CAUTION", "The presealed recipe, fixed seeds, eight events, 32-record allocation, exact selection and bootstrap agree. No outcome-based rerun appears in this evidence. The reused panel and prior research choices still preclude a fresh confirmatory claim."),
        ("Correlation versus causation", "CAUTION", "The controlled R/U contrast compares selectors at the same allocation under shared K-generated supervision. It does not identify why domain effects differ, establish seed robustness, or show asynchronous or training-inclusive speedup. Coverage improved at all seven replay events without a demonstrated overall serving gain."),
        ("Reverse causality", "NOTE", "All selected old records precede their training event; admission follows all three commits. Future-serving counts first differ at event 3 after the event-2 update. Temporal order is correct, but it does not prove a general mechanism explanation."),
    ]
    result = dict(material_passport=dict(mode="run→validate", verification_status="ANALYZED",
        scope="Separate CPU implementation; same-team author overlap; no independent GPU reproduction",
        execution_commit=audit["execution_commit"], source_handoff="TRACK2_K_REPLAY_EXECUTOR_INSTRUCTIONS_20261102.txt"),
        audit_status="passed", checks=audit["checks"], discrepancies=0, source_files_read=len(inputs["files"]),
        native_trace_blocks=blocks, final=endpoint, final_repeats=repeats, future_by_domain=future_by_domain,
        future_pooled=future["future_call_reduction_pct"], training_by_arm=exposure,
        fallacy_coverage="11/11", fallacies=[dict(name=n, severity=s, assessment=t) for n, s, t in fallacies],
        interpretation="No demonstrated pooled superiority over U or K; the fixed coverage allocation is not validated as a K upgrade. Park this candidate in its present form; no additional experiment is authorized by this review.",
        supplementary_input_hashes=additional_inputs, review_generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        generated_utc=datetime.now(timezone.utc).isoformat())
    (output / "FINAL_REVIEW_DETAIL.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    lines = ["# Track2 final evidence interpretation", "", "## Material Passport", "",
        "Mode: run→validate. Verification status: **ANALYZED**. Execution: `" + audit["execution_commit"] + "`.", "",
        "Source handoff: `TRACK2_K_REPLAY_EXECUTOR_INSTRUCTIONS_20261102.txt` (2026-11-02). "
        "This is a separate CPU reconstruction by the same team, with authorship overlap in the original selection, "
        "evaluation and reporting implementation. The auditor imports no experiment code. It is not an independent GPU reproduction.", "",
        "## Finding", "",
        "The retained evidence passes **2,917,883 checks with zero discrepancies**. All six audit sections passed on the first run; "
        "no production file was changed and no audit repair was needed. The measured result does not establish a useful overall "
        "improvement from this replay allocation. R saves 12 calls against U but uses 21 more than K; both pooled intervals span harm "
        "and benefit. R's coding advantage over U is accompanied by a math regression. Coverage rises at every replay event, "
        "without a demonstrated overall serving benefit.", "",
        "The complete result supports parking this candidate in its current form. The evidence does not show the consistent "
        "across-domain/future-request benefit needed to motivate an unchanged seed replication under the handoff. "
        "Changing the replay dose or descriptor is not authorized by this review. This is an inconclusive/negative utility result "
        "from a valid run, not an implementation failure and not proof of equivalence.", "",
        "## Reconstructed endpoint", "",
        "Positive reduction means fewer calls or less time for the candidate. Counts use 256 unique requests once, after exact "
        "repeat equality; time uses each request/method's mean of three repeats. The independent 10,000-draw bootstrap uses "
        "PCG64 seed 20261102, jointly paired indices, 128 math and 128 coding samples per draw, and linear percentiles. "
        "Every reconstructed numerical field agrees with the saved endpoint using relative and absolute tolerances of 1e-12.", "",
        "| Panel | Contrast | Call reduction % [95% interval] | Wins / regressions / ties | Gross saved / added | Time reduction % |",
        "|---|---|---:|---:|---:|---:|"]
    for name, panel in [("pooled", endpoint)] + list(endpoint["domains"].items()):
        for a, b in pairs:
            row = panel["comparisons"][a + "_vs_" + b]
            value = row["call_reduction"]
            lines.append(f"| {name} | {a}/{b} | {value['point_pct']:.4f} [{value['ci95_pct'][0]:.4f}, {value['ci95_pct'][1]:.4f}] | "
                         f"{row['request_wins']} / {row['request_regressions']} / {row['request_ties']} | "
                         f"{row['gross_calls_saved']} / {row['gross_calls_added']} | {row['request_latency_s_reduction']['point_pct']:.4f} |")
    lines += ["", "These are conditional request-panel intervals for one seed and one fitted trajectory on a reused development "
        "panel. They do not include training-seed or adaptive-trajectory uncertainty. Secondary/domain/time intervals are "
        "unadjusted and descriptive; no p-values or multiplicity-adjusted confirmatory findings are asserted.", "",
        "| Arm | Calls | Accepted draft tokens | Accepted / call | Committed progress / call | Output / call | Mean-repeat total request seconds |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for arm, row in endpoint["method_totals"].items():
        lines.append(f"| {arm} | {row['verification_calls']} | {row['accepted_draft_tokens']} | {row['accepted_tokens_per_call']:.6f} | "
                     f"{row['committed_progress_per_call']:.6f} | {row['output_tokens_per_call']:.6f} | {row['request_latency_s']:.6f} |")
    lines += ["", "Each arm commits 75,650 post-anchor tokens and returns the same 75,906 complete output tokens. Accepted draft tokens "
        "exclude target corrections; committed progress and full output are distinct quantities. Counts are identical in all three repeats.", "",
        "| Repeat | K request seconds | U request seconds | R request seconds | R/U time reduction % | R/K time reduction % |",
        "|---|---:|---:|---:|---:|---:|"]
    for row in repeats:
        times = [row["methods"][arm]["request_latency_s"] for arm in "KUR"]
        lines.append(f"| {row['repeat']} | {times[0]:.6f} | {times[1]:.6f} | {times[2]:.6f} | "
                     f"{row['request_time_reduction_pct']['R_vs_U']:.4f} | {row['request_time_reduction_pct']['R_vs_K']:.4f} |")
    lines += ["", "R is slower than U and K in each complete final repeat. Mean-time R/U is -0.6691% [-1.7590, 0.3440]%; "
        "R/K is -0.6441% [-1.7911, 0.4262]%. Neither establishes a serving speedup. Publication and loading are outside request timing; "
        "the 36 batch boundaries plus the final boundary all carry passing transfer/updater quiescence and resource receipts.", "",
        "## Future-serving stream", "",
        "All 3,072 pre-update records are valid. Events 1 and 2 have exact neutral arm counts; replay can first affect event 3. "
        "Events 3-8 are descriptive observations of this adaptive trajectory, without a bootstrap. K has capture on; U/R have capture off. "
        "Their pre-update wall times are not speedup comparators.", "",
        "| Event | Domain | K calls | U calls | R calls | R/U reduction % | R/K reduction % |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for row in future["events"]:
        calls = {arm: row["method_totals"][arm]["verification_calls"] for arm in "KUR"}
        lines.append(f"| {row['event']} | {row['domain']} | {calls['K']} | {calls['U']} | {calls['R']} | "
                     f"{100*(1-calls['R']/calls['U']):.4f} | {100*(1-calls['R']/calls['K']):.4f} |")
    lines += ["", "Events 3-8 total K/U/R = 40,372 / 40,180 / 40,248 calls: R/U -0.1692%, R/K +0.3071%, U/K +0.4756%. "
        "R loses to U on five of these six events. Future math totals are 10,636 / 10,546 / 10,594 and coding totals "
        "29,736 / 29,634 / 29,654; R uses more calls than U in both domain aggregates. This does not support treating the small final "
        "pooled R/U point estimate as a consistent selector benefit.", "",
        "## Selection, memory and training evidence", "",
        "All 1,024 4,096-dimensional descriptors are finite and unit-normalized in FP64, and match their packet receipts. "
        "Those receipts identify the last prompt row and a finite positive original norm. Every selected identity, sorted pass plan, "
        "coverage score and marginal trajectory is independently reconstructed: clipped nonnegative cosine, 32 distinct greedy choices, "
        "exact score ties by original ordinal, common fresh96 seed [20261102,e,0], and uniform old32 seed [20261102,e,1]. "
        "For events 2-8, R coverage exceeds U respectively by 0.006149, 0.014710, 0.019605, 0.010082, 0.006201, 0.017103 and 0.030149.", "",
        "The shared capacity128 reservoir processes all 1,024 unique fresh records in chronological order. All 896 post-fill Algorithm R draws, "
        "counter values, slot identities, eviction receipts and before/after PCG64 state match exactly under the separate seed "
        "[20261102,0,2]. No current/future record enters old replay; insertion receipts follow all three arm commitments. "
        "Final memory contains 63 math and 65 coding records. Domains are joined afterward for reporting, not supplied to selection.", "",
        "All 24 training workers complete 256 response forwards, 256 backwards and 128 Adam steps. Their 6,144 presentation records "
        "match the exact two-pass selections and original path/hash bindings. Every two-microstep accumulation boundary, optimizer "
        "counter, fixed 3e-4 LR receipt, replay flag, anchor count and finite metric is checked. Event-1 neutral decoded-state receipts "
        "and all 24 state-boundary receipts pass. The audit does not newly decode model, Adam or RNG tensors.", "",
        "| Arm | Presentations | Adam steps | Replay presentations | Prompt tokens | Response tokens | Active anchors |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for arm, row in exposure.items():
        lines.append("| " + arm + " | " + " | ".join(str(row[key]) for key in ("response_presentations", "optimizer_steps", "replay_presentations", "prompt_tokens", "response_tokens", "active_anchors")) + " |")
    lines += ["", "Totals are 6,144 presentations, 3,072 Adam steps, 448 replay presentations, 346,925 prompt tokens, "
        "1,761,381 response tokens and 1,631,357 active anchors. Root work reconciles 3,072 pre-update generations, four capture parity "
        "generations and 1,024 preparation forwards, each with zero unknown operations. Final evaluation adds 2,304 generations: "
        "5,380 complete generations overall. Equal response counts do not imply equal token work.", "",
        "## Native validity, costs and retained capability", "",
        f"The audit reconstructs all {blocks['total']:,} native verification blocks: {blocks['pre_update']:,} pre-update, "
        f"{blocks['final']:,} final and {blocks['capture_parity']} parity. It verifies first-error decisions, committed prefixes, causal "
        "anchors/positions, complete output/termination, accepted/progress/call metrics and exact native sidecar equality. "
        "All final outputs match the retained canonical tokens and stop semantics. All four fixed capture controls match K. "
        "Publication helper/cache receipts and complete native weight inventories pass; no new GPU inference or weight comparison was run.", "",
        "The parent retrieval verified all 19,639 compact files. This auditor reads and hashes 14,932 evidence files; "
        "`FINAL_AUDIT_INPUTS.json` records their identities and the auditor source hashes. Original bulk tensors are outside this retrieval. "
        "Descriptor extraction and scientific-state equality therefore rely on their retained execution receipts; they are not "
        "fresh tensor-level replications. The audit's first execution took 16.77 seconds with approximately 608 MiB peak RSS.", "",
        "Recorded campaign costs remain separate and may nest: campaign 17,525.49 s; adaptation-ready 15,327.05 s; training workers "
        "10,016.63 s; final evaluation 2,191.94 s; export 434.00 s; selection 0.683 s; descriptors 0.453 s. These do not sum into an "
        "independently measured total or prove training-inclusive/asynchronous speedup. Detailed costs remain in the original status/report.", "",
        "There are 2,986 retirement receipts, recording 208,569,153,843 logical bytes and 208,574,205,952 allocated bytes cumulatively. "
        "They state retired, not archived, with reasons and retained-evidence references. This audit does not verify current remote existence "
        "or reconstruct retired features/intermediate states. Intermediate tensor replay/resume is lost; compact receipts remain. "
        "No raw prompt or response text is included here, and no production file was modified.", "",
        "## Statistical fallacy scan: 11/11", "",
        "| Category | Assessment | Evidence and limit |", "|---|---|---|"]
    lines.extend(f"| {name} | {severity} | {note} |" for name, severity, note in fallacies)
    lines += ["", "Machine-readable details: `FINAL_REVIEW_DETAIL.json`. Reconstruction evidence: `FINAL_EVIDENCE_AUDIT.json`; "
        "input provenance: `FINAL_AUDIT_INPUTS.json`. No stage or project direction is advanced by this review."]
    (output / "FINAL_INTERPRETATION.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(dict(status="written", native_blocks=blocks, outputs=["FINAL_INTERPRETATION.md", "FINAL_REVIEW_DETAIL.json"])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    main(args.run_root, args.output)
