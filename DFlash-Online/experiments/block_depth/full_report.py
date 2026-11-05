"""Compact outcome and cost report from preserved full-run records."""

import argparse
import json
from pathlib import Path

from online_feedback.storage import write_json


def read(path):
    return json.loads(Path(path).read_text())


def report(root):
    root = Path(root)
    status = read(root / "STATUS.json")
    plan = read(root / "PLAN.json")
    costs = read(root / "COSTS.json") if (root / "COSTS.json").exists() else {}
    result_path = root / "report/ENDPOINT_ANALYSIS.json"
    endpoint = read(result_path) if result_path.exists() else None
    destination = root / "report"
    destination.mkdir(exist_ok=True)
    rows = ["# Frozen K+58 block-depth experiment", "",
        f"Status: **{status['status']}**. Execution commit: `{plan['code_commit']}`.", "",
        "This fixed-checkpoint screen compares original FIVE, a fitted rank-32 THREE readout, "
        "and a frozen ridge-gated ADAPTIVE route. The target and all existing K+58 parameters "
        "remain frozen. FINAL contains384 fit-disjoint GSM8K test requests; prior project exposure "
        "is disclosed in the sealed manifests. It is not an online-recalibration experiment.", "",
        "## Work and validity", "",
        "| Operation | Completed | Unknown | Planned |", "|---|---:|---:|---:|"]
    work = status["work"]
    for key, completed in work["completed"].items():
        planned = {"readout_generations":368, "readout_reconstructions":16, "teacher_preparations":384,
                   "gate_generations":128, "calibration_requests":128, "final_generations":2304}.get(key)
        if key == "canonical_generations":
            planned = plan["budget"]["untimed_reference_generations_max"]
        rows.append(f"| {key} | {completed} | {work['unknown'][key]} | {planned} |")
    rows += ["", "The16 retained READOUT generations and earlier qualification/pricing remain separately recorded. "
        "Reconstruction replays their exact saved targets and commits, without regenerating answers. "
        "Missing canonical references are separately counted native target-verified greedy executions; "
        "they are never training feedback. No outcome-based request substitution is allowed.", ""]
    if endpoint and endpoint["comparison"]:
        rows += ["## FINAL comparison", "",
            "Positive reduction favors ADAPTIVE. Each question's two repeats are averaged before pooling. "
            "Intervals use10,000 paired question bootstrap draws, seed20261105.", "",
            "| Comparison | Request-time reduction | 95% interval | Call reduction |",
            "|---|---:|---:|---:|"]
        for name in ("ADAPTIVE_vs_FIVE", "ADAPTIVE_vs_THREE"):
            item = endpoint["comparison"][name]
            rows.append(f"| {name} | {item['point_pct']:+.4f}% | "
                f"[{item['ci95_pct'][0]:+.4f}%, {item['ci95_pct'][1]:+.4f}%] | {item['call_reduction_pct']:+.4f}% |")
        rows += ["", "| Path | Calls | Mean-summed request seconds | Progress/call | Shallow fraction |",
                 "|---|---:|---:|---:|---:|"]
        for name, item in endpoint["arm_totals"].items():
            rows.append(f"| {name} | {item['verification_calls']} | {item['request_latency_s']:.4f} | "
                f"{item['progress_per_call']:.4f} | {item['shallow_exit_fraction']:.4f} |")
        rows += ["", f"Prespecified decision: **{endpoint['decision']}**. "
            f"Both paired95% lower bounds above zero: **{endpoint['scientific_criterion_passed']}**. "
            f"Practical2% over FIVE reached by point estimate: **{endpoint['practical_two_pct_vs_FIVE_point_reached']}**.", ""]
    else:
        rows += ["FINAL is incomplete or invalid; no scientific superiority claim is available.", ""]
    rows += ["## Costs and implementation scope", "", "| Stage | Wall seconds |", "|---|---:|"]
    for name, value in costs.items():
        rows.append(f"| {name} | {value:.4f} |")
    training_path = root / "READOUT_RESULTS.json"
    if training_path.exists():
        training = read(training_path)
        rows += ["", f"Readout groups completed: {len(training)}. "
            f"Auxiliary optimizer steps: {sum(r['optimizer_steps'] for r in training)}; "
            f"response presentations: {sum(r['response_presentations'] for r in training)}. "
            "Backbone optimizer steps:0. Per-group results retain native objective exposure, anchors, "
            "finite-gradient/update checks, own-state restoration and peak memory.", ""]
    gate_file = root / "gate/GATE.json"
    if gate_file.exists():
        gate = read(gate_file)
        rows += [f"Gate: {gate['rows']} paired starts; κ={gate['kappa_ms_per_token']:.9f} ms/token; "
            f"continuation cost={gate['continuation_cost_ms']:.9f} ms. "
            "Fit and threshold were frozen before FINAL. GATEκ uses synchronized native decode-round wall time, "
            "excluding prefill/trace-copy/cleanup. No gate fit uses FINAL outcomes.", ""]
    rows += ["Complete request latency includes native generation, cache maintenance and required request/RPC "
        "overhead. FINAL has no capture, detailed trajectory trace, shadow proposals or concurrent updater. "
        "Its lightweight host decode-round subcounter adds no synchronization and may omit trailing GPU "
        "completion; complete-request timing remains primary. Accepted-token counts follow the native "
        "committed-progress-minus-one convention; committed progress is reported separately.", "",
        "Consumed READOUT feature tensors are retired only after their group successor and compact evidence "
        "validate; retirement receipts identify exact paths and lost tensor replay capability. They are not "
        "described as archived. Code, masks/anchors, token records, calibration rows, frozen sidecar/gate "
        "and raw endpoint measurements remain retained.", "",
        "The earlier2% optimistic pricing screen was a budget decision. Final scientific advancement instead "
        "requires exact validity and superiority to both fixed paths. A null or harmful result receives the "
        "same endpoint and is not retuned.", ""]
    if (root / "FAILURE.json").exists():
        rows += ["## Preserved execution failure", "", read(root / "FAILURE.json")["error"], ""]
    (destination / "REPORT.md").write_text("\n".join(rows))
    write_json(destination / "PROVENANCE.json", dict(code_commit=plan["code_commit"], plan=plan,
        status=status, costs=costs, original_failures_preserved=True))
    return destination / "REPORT.md"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    print(report(parser.parse_args().root))
