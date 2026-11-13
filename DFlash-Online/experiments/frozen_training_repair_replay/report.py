"""Publish scalar-only fitted-retention results; raw logits/tokens stay private."""
import argparse
import csv
import json
from pathlib import Path

from .analysis import OUTCOMES, summarize


def _read(path):
    return json.loads(Path(path).read_text())


def load_arm(root, arm):
    root = Path(root)
    status = _read(root / "STATUS.json")
    expected = dict(status="completed", arm=arm, response_forwards=256, projection_chunks=8170,
                    backward_calls=0, optimizer_steps=0, generations=0)
    for key, value in expected.items():
        if status.get(key) != value:
            raise ValueError(f"Cannot publish incomplete/invalid arm: {key}={status.get(key)!r}")
    rows = []
    files = sorted((root / "presentations").glob("*.json"))
    if len(files) != 256:
        raise ValueError("Expected all 256 presentation receipts")
    for index, path in enumerate(files):
        value = _read(path)
        if value["presentation"]["presentation_id"] != index:
            raise ValueError("Presentation receipt order differs")
        rows.extend(value["rows"])
    return rows, status


def _csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def narrative(summary, *, execution_commit, reproduction_command):
    p = summary["primary"]
    raw = summary["occurrence_summary"]
    ci = summary["primary_interval"]
    direction = {
        "fitted_retention_advantage": "RECENT has a demonstrated fitted retention advantage on these fixed training inputs.",
        "fitted_retention_disadvantage": "RECENT has a demonstrated fitted retention disadvantage on these fixed training inputs.",
        "direction_unresolved": "The direction of the fitted retention difference remains unresolved."
    }[summary["interpretation"]]
    text = ["# Frozen event-seven training-repair replay", "", "## Material Passport", "",
        "- Status: completed frozen-checkpoint replay; statistical interpretation is conditional on these endpoints.",
        f"- Executed source: `{execution_commit}`.",
        "- Population: 128 fitted requests, 815 original slots and 1,629 paired eligible presentations.",
        "- No optimizer steps, backward calls, generations, coefficient changes or new training.", "",
        f"**{direction}** RECENT minus NATIVE acceptance through the tagged position is "
        f"**{p['net_difference_pp']:+.6f} percentage points**, with request-cluster 95% interval "
        f"**[{ci['lower_pp']:+.6f}, {ci['upper_pp']:+.6f}] pp**. "
        "PR #44's measured serving result remains unchanged.", "",
        "## Primary equal-slot result", "",
        "Each slot first averages its one or two eligible presentations. All 815 slots then receive equal weight. "
        "The singleton is retained at full slot weight; neither pass nor reachable/firing cases are selected after seeing outcomes.", "",
        "| Measure | NATIVE7 | RECENT7 |", "|---|---:|---:|",
        f"| Accepted-through slot equivalents / 815 | {p['native_accepted_equivalents']:g} | {p['recent_accepted_equivalents']:g} |",
        f"| Acceptance through tag | {p['native_retention_pct']:.6f}% | {p['recent_retention_pct']:.6f}% |",
        f"| Earlier-break equivalents | {p['native_earlier_break_equivalents']:g} | {p['recent_earlier_break_equivalents']:g} |",
        f"| First-mismatch-at-tag equivalents | {p['native_first_mismatch_at_tag_equivalents']:g} | {p['recent_first_mismatch_at_tag_equivalents']:g} |",
        f"| Direct correctness at tag | {p['native_direct_correct_pct']:.6f}% | {p['recent_direct_correct_pct']:.6f}% |",
        f"| Mean canonical-minus-rival margin | {p['native_mean_margin']:.6f} | {p['recent_mean_margin']:.6f} |",
        f"| Below original K+6 floor | {p['native_below_floor_pct']:.6f}% | {p['recent_below_floor_pct']:.6f}% |",
        f"| Wrong-at-tie | {p['native_wrong_at_tie_pct']:.6f}% | {p['recent_wrong_at_tie_pct']:.6f}% |", "",
        f"RECENT rescues **{p['rescues']:g} slot equivalents ({p['rescue_pct']:.6f}%)** and loses "
        f"**{p['losses']:g} ({p['loss_pct']:.6f}%)**. Net = rescues − losses passes exactly. "
        f"Native headroom is **{p['native_headroom_pct']:.6f} percentage points**, the absolute maximum "
        "net increase available on these tags. The observed net does not exceed it.", "",
        "## Occurrences, disagreements and pass checks", "",
        f"Raw counts on 1,629 paired occurrences: NATIVE accepts through {raw['native_accepted_equivalents']}; "
        f"RECENT {raw['recent_accepted_equivalents']}; {raw['rescues']} rescues and {raw['losses']} losses. "
        "These integer counts are separate from the primary equal-slot result.", "",
        f"Requests with a retention disagreement: {summary['disagreements']['any_retention_disagreement']}/128; "
        f"with any three-category outcome disagreement: {summary['disagreements']['any_outcome_disagreement']}/128; "
        f"with a direct-correctness disagreement: {summary['disagreements']['any_direct_correctness_disagreement']}/128.", "",
        "| Pass | Paired occurrences | Native accepts through | Recent accepts through | Rescues | Losses | Net pp |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for pass_id, row in summary["pass_summaries"].items():
        text.append(f"| {pass_id} | {row['denominator']} | {row['native_accepted_equivalents']} | "
                    f"{row['recent_accepted_equivalents']} | {row['rescues']} | {row['losses']} | {row['net_difference_pp']:+.6f} |")
    text.extend(["", "The complete 3×3 outcomes are in [outcome_table.csv](outcome_table.csv), with separate integer occurrence "
        "and equal-slot-equivalent cells. [paired_occurrences.csv](paired_occurrences.csv) retains every failure, including "
        "earlier breaks; [slots.csv](slots.csv) and [per_request.csv](per_request.csv) expose every denominator. "
        "Direct correctness and margins are scored even after an earlier failure. A correct later token is not usable "
        "prefix progress if an earlier token blocks it. Margin zero does not determine the native argmax winner.", "",
        "## Uncertainty and interpretation", "",
        "The primary interval uses 10,000 paired whole-request resamples with PCG64 seed 1707 and linear percentile "
        "limits. Each draw divides the summed slot-level differences by its resampled slot count. It measures "
        "sensitivity to which fitted requests are represented, not variation across training seeds. Exact rational "
        "cohort results and bootstrap ordering are retained in [SUMMARY.json](SUMMARY.json). Only this primary contrast "
        "has an inferential interval; pass, margin and outcome summaries are descriptive.", "",
        "The held-out PR #44 repair comparison remains **8 rescues versus 11 losses** (RECENT accepts through 275/431 "
        "versus NATIVE 278/431). This fitted-input diagnostic is not another serving test. It cannot convert local "
        "retention into time saved, locate the update that first caused damage, establish that earlier intervention "
        "would help, justify a coefficient change, or identify the cause of any fitted-versus-held-out difference. "
        "The +0.50% serving target is not a threshold for this percentage-point diagnostic.", ""])
    if summary["all_retention_tied"]:
        text.append("All retention decisions tie: zero paired retention disagreements. This does not establish equivalence on future data.\n")
    text.extend(["## Work and reproduction", "",
        "The two endpoints run sequentially on one A800 using fresh scoring models. K+6 supplies the retained sampler "
        "RNG and validated configuration; each endpoint export is strict-loaded before any draft forward. "
        "The full original frozen training geometry is used: zero dropout, gradients disabled, BF16, Flex Attention, "
        "16-token blocks including seed, and eight-block vocabulary projections with original partial chunks. "
        "Natural order, anchors and keep masks must match every retained binding. Installed endpoint tensors are "
        "checked exactly before and after replay.", "",
        "| Work | NATIVE7 | RECENT7 | Total |", "|---|---:|---:|---:|",
        "| Response forwards | 256 | 256 | 512 |", "| Vocabulary projection chunks | 8,170 | 8,170 | 16,340 |",
        "| Backward calls / optimizer steps / generations | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |", "",
        "Exact endpoint/input identities and measured device/worker costs are in the run receipts. No prompt text, "
        "token IDs, model tensors or private raw paths appear in the scalar tables.", ""])
    if "work" in summary:
        text.extend(["| Measured interval (seconds) | NATIVE7 | RECENT7 |", "|---|---:|---:|"])
        for key, label in (("synchronized_forward_wall_s", "Synchronized scoring forwards"),
                           ("model_loading_and_integrity_s", "Model loading and initial tensor checks"),
                           ("post_scoring_integrity_s", "Final tensor integrity check"), ("wall_s", "Worker total")):
            values = [summary["work"][arm].get(key) for arm in ("NATIVE7", "RECENT7")]
            rendered = [f"{value:.6f}" if value is not None else "unavailable" for value in values]
            text.append(f"| {label} | {rendered[0]} | {rendered[1]} |")
        text.extend(["", "Synchronized scoring time waits for the selected device around each response and includes "
            "host dispatch and scalar extraction. Worker totals additionally include initialization, integrity checks "
            "and persistence. These are nested wall-clock intervals, not additive costs or measured GPU-kernel busy time.", ""])
    text.extend(["Reproduction command:", "", "```bash", reproduction_command, "```", "",
        "## Decision boundary", "",
        f"Native headroom is **{p['native_headroom_pct']:.6f} pp**; RECENT rescues **{p['rescues']:g}** "
        f"slot equivalents and loses **{p['losses']:g}**, for **{p['net_difference_pp']:+.6f} pp**. " + direction])
    if summary["interpretation"] == "fitted_retention_advantage":
        text.append("Together with the unchanged held-out result, this supports fitted retention without a demonstrated "
                    "transfer benefit. It argues against spending more compute merely strengthening this instance-specific rule; "
                    "these panels alone do not identify the cause of the gap.")
    elif summary["interpretation"] == "fitted_retention_disadvantage":
        text.append("The unfavorable fitted result weakens this preservation mechanism even on its own examples.")
    else:
        text.append(f"The interval remains compatible with changes from {ci['lower_pp']:+.6f} to {ci['upper_pp']:+.6f} pp. "
                    "No fitted retention advantage is established; a flat pass-through result with a higher margin "
                    "would support only a surrogate change.")
    text.append("Stop after this report. Any future training is a separate research decision.\n")
    return "\n".join(text)


def execute(native_path, recent_path, output, manifest_path, *, reproduction_command):
    native, native_status = load_arm(native_path, "NATIVE7")
    recent, recent_status = load_arm(recent_path, "RECENT7")
    manifest = _read(manifest_path)
    commits = {s["execution_commit"] for s in (native_status, recent_status)}
    if commits != {manifest["execution_commit"]}:
        raise ValueError("Arm execution revisions do not match the manifest")
    summary, paired, slots, requests, table = summarize(native, recent)
    summary["execution_commit"] = manifest["execution_commit"]
    summary["work"] = {arm: {key: state.get(key) for key in (
        "response_forwards", "projection_chunks", "backward_calls", "optimizer_steps",
        "generations", "wall_s", "synchronized_forward_wall_s", "model_loading_and_integrity_s",
        "post_scoring_integrity_s", "peak_allocated_bytes", "peak_reserved_bytes")}
        for arm, state in (("NATIVE7", native_status), ("RECENT7", recent_status))}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in (("paired_occurrences", paired), ("slots", slots), ("per_request", requests), ("outcome_table", table)):
        _csv(output / f"{name}.csv", rows)
    (output / "SUMMARY.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (output / "REPORT.md").write_text(narrative(summary, execution_commit=manifest["execution_commit"],
        reproduction_command=reproduction_command))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", required=True)
    parser.add_argument("--recent", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--reproduction-command", required=True)
    args = parser.parse_args()
    execute(args.native, args.recent, args.output, args.manifest,
            reproduction_command=args.reproduction_command)
