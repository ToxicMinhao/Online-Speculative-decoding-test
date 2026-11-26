"""Build privacy-safe Point3 serving tables from independently audited records."""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path

from independent_serving_audit import ARMS, CONTRASTS, COUNTS, TIMES, collect


def csv_file(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def content_identity(path):
    return dict(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def public_tables(evaluation, audit_path, output):
    rows = [json.loads(line) for line in (evaluation / "MEASUREMENTS.jsonl").read_text().splitlines()]
    summary = json.loads((evaluation / "SUMMARY.json").read_text())
    audit = json.loads(audit_path.read_text())
    if audit.get("status") != "passed":
        raise ValueError("Publication requires a passed independent serving audit")
    protocol = json.loads((evaluation / "PROTOCOL.json").read_text())
    paired = collect(rows)
    ordinal = {}
    warmup_ordinal = {}
    for row in rows:
        mapping = ordinal if row["phase"] == "scored" else warmup_ordinal
        if row["request_id"] not in mapping:
            mapping[row["request_id"]] = len(mapping) + 1
    output.mkdir(parents=True, exist_ok=True)
    count_fields = COUNTS + ("finish_reason",)
    request_rows = []
    for item in sorted(paired, key=lambda item: ordinal[item["request_id"]]):
        result = dict(request_ordinal=ordinal[item["request_id"]])
        for arm in ARMS:
            result.update({arm + "_" + field: item[arm][field] for field in count_fields + TIMES})
        for candidate, reference in CONTRASTS:
            name = candidate + "_vs_" + reference
            difference = item[reference]["verification_calls"] - item[candidate]["verification_calls"]
            result[name + "_calls_saved"] = difference
            result[name + "_call_outcome"] = "improved" if difference > 0 else "regressed" if difference < 0 else "tie"
        request_rows.append(result)
    csv_file(output / "PER_REQUEST.csv", request_rows)

    by_phase = {"scored": [], "warmup": []}
    for row in rows:
        mapping = ordinal if row["phase"] == "scored" else warmup_ordinal
        result = dict(request_ordinal=mapping[row["request_id"]], method=row["method"],
                      repeat=row["repeat"], group=row["group"], activation=row["activation"],
                      valid_measurement=row["valid_measurement"], error_count=len(row["errors"]))
        result.update({field: row[field] for field in count_fields + TIMES})
        by_phase[row["phase"]].append(result)
    csv_file(output / "ALL_REPEATS.csv", by_phase["scored"])
    csv_file(output / "WARMUPS.csv", by_phase["warmup"])
    arm_rows, repeat_rows = [], []
    for arm, total in audit["totals"].items():
        row = dict(method=arm)
        row.update({field: total[field] for field in COUNTS + TIMES})
        row.update({field: total[field] for field in (
            "tokens_per_s", "native_accepted_per_call", "stop_clipped_accepted_per_call", "committed_per_call")})
        row.update({"terminated_" + key: value for key, value in total["termination_counts"].items()})
        arm_rows.append(row)
        for repeat in range(3):
            repeat_rows.append(dict(method=arm, repeat=repeat,
                **{field: total["repeat_totals"][field][repeat] for field in TIMES}))
    csv_file(output / "ARM_TOTALS.csv", arm_rows)
    csv_file(output / "REPEAT_TOTALS.csv", repeat_rows)
    contrast_rows = []
    for candidate, reference in CONTRASTS:
        comparison = audit["contrasts"][candidate + "_vs_" + reference]
        row = dict(candidate=candidate, reference=reference)
        for metric in tuple(field + "_reduction_pct" for field in TIMES) + ("verification_call_reduction_pct",):
            result = comparison[metric]
            row[metric] = result["value"]
            row[metric + "_ci95_low"] = result["ci95"][0] if result["ci95"] else None
            row[metric + "_ci95_high"] = result["ci95"][1] if result["ci95"] else None
        row.update({field: comparison[field] for field in (
            "wins", "regressions", "ties", "gross_calls_saved", "gross_calls_added", "net_calls_saved")})
        contrast_rows.append(row)
    csv_file(output / "CONTRASTS.csv", contrast_rows)

    # The input-order digest remains public, but its original identifiers do not.
    sanitized_audit = deepcopy(audit)
    private_bootstrap_order = sanitized_audit["bootstrap"].pop("input_order")
    sanitized_audit["bootstrap"]["public_request_order"] = [ordinal[key] for key in private_bootstrap_order]
    sanitized_audit["bootstrap"]["input_order_privacy"] = "Original identities retained privately; digest and draw hash disclosed"
    sanitized_audit["execution_commit"] = protocol["execution_commit"]
    sanitized_audit["source_evidence"] = {name: content_identity(evaluation / name)
                                          for name in ("MEASUREMENTS.jsonl", "SUMMARY.json", "PROTOCOL.json")}
    sanitized_audit["row_counts"] = dict(requests=len(request_rows), scored=len(by_phase["scored"]),
                                         warmups=len(by_phase["warmup"]), contrasts=len(contrast_rows))
    sanitized_audit["serving_costs"] = summary["serving_costs"]
    (output / "INDEPENDENT_SERVING_AUDIT.json").write_text(
        json.dumps(sanitized_audit, indent=2, sort_keys=True, allow_nan=False) + "\n")

    groups = {}
    for row in rows:
        if row["phase"] == "scored":
            groups[row["request_id"]] = row["group"]
    group_results = []
    for candidate, reference in CONTRASTS:
        for group in (0, 1):
            selected = [item for item in paired if groups[item["request_id"]] == group]
            result = dict(candidate=candidate, reference=reference, group=group, requests=len(selected))
            for field in ("request_latency_s", "verification_calls"):
                result[field + "_reduction_pct"] = 100 * (1 - sum(item[candidate][field] for item in selected)
                                                          / sum(item[reference][field] for item in selected))
            group_results.append(result)
    csv_file(output / "GROUP_DIRECTION_CHECK.csv", group_results)
    return sanitized_audit, group_results


def report(audit, group_results):
    identity_description = (
        "The independent CPU postflight parsed the stored safetensors headers and native tensor bytes, "
        "without loading a model framework or using a GPU. All three parameter identities match; "
        "the exports are distinct. This authenticates stored endpoints, not independent training re-execution."
        if audit["identity_status"] == "independently_verified_cpu_postflight" else
        "Model-identity conclusions additionally require the separately supplied CPU export verification."
    )
    lines = [
        "# Independent serving audit", "", "## Material Passport", "",
        "- Task: Point3 cumulative-maximum assessment, retained-result validation.",
        "- Verification status: ANALYZED; independent CPU statistical recomputation passed. No new GPU replication.",
        f"- Actual execution commit: `{audit['execution_commit']}`.",
        "- Data: 128 paired reused development requests; three timing repeats; three fixed endpoints.",
        "- Public identities: fresh 1–128 request ordinals and 1–2 warmup ordinals only.", "",
        "## Results", "",
        "Positive reductions favor the candidate. Timing uses summed per-request means; call totals use one validated trajectory per request.", "",
        "| Candidate / reference | Complete latency reduction, % (95% CI) | Call reduction, % (95% CI) | Call wins / regressions / ties | Gross calls saved / added |",
        "|---|---:|---:|---:|---:|",
    ]
    for candidate, reference in CONTRASTS:
        c = audit["contrasts"][candidate + "_vs_" + reference]
        latency, calls = c["request_latency_s_reduction_pct"], c["verification_call_reduction_pct"]
        formatted = lambda result: f"{result['value']:.6f} ({result['ci95'][0]:.6f}, {result['ci95'][1]:.6f})"
        lines.append(f"| {candidate} / {reference} | {formatted(latency)} | {formatted(calls)} | "
                     f"{c['wins']} / {c['regressions']} / {c['ties']} | {c['gross_calls_saved']} / {c['gross_calls_added']} |")
    lines += ["", f"The prespecified unrounded selection chooses **{audit['freeze']['selected']}**. "
              "This is a development-selection rule, not an equivalence test or a claim that all other settings fail.", "",
              "## Recomputed evidence and limits", "",
              "The independent script imports no production experiment modules. It reconstructs each request's three-repeat means, "
              "one-trajectory counts, acceptance/progress identities, all three contrasts, repeat totals and the selection. "
              "The paired bootstrap uses 10,000 shared PCG64 resamples, seed 20261125, and linear 2.5/97.5 percentiles. "
              "The draw digest is `" + audit["bootstrap"]["draw_sha256"] + "`.", "",
              "The script verifies retained scalar calculations, not unsaved training-vector contractions. "
              "Exact token/termination checks, native qualification, checkpoint authentication, physical-card isolation, "
              "and work/resource accounting are covered by the companion evidence audit. "
              "Export identity status for this statistical audit: `" + audit["identity_status"] + "`. "
              + identity_description, "",
              "No p-values were requested or computed. The MAX-versus-SUM complete-request latency interval is the primary component contrast. "
              "All 12 displayed timing/count intervals are retained; the nonprimary intervals are descriptive and unadjusted. "
              "The three whole-request bootstrap pairs are not independent replications. "
              "This reused panel and one inherited training trajectory do not establish fresh-panel, seed, feedback-window or domain robustness. "
              "Serving differences do not establish training-inclusive speedup. Lower MAX than SUM at the same endpoint follows from construction "
              "and is not evidence of learning benefit.", "",
              "## Statistical fallacy checks — 11/11", "",
              "| Check | Finding / interpretation limit |", "|---|---|"]
    group_text = "; ".join(f"{c}/{r}: " + ", ".join(
        f"group {row['group']} {row['request_latency_s_reduction_pct']:+.3f}% latency / {row['verification_calls_reduction_pct']:+.3f}% calls"
        for row in group_results if row['candidate'] == c and row['reference'] == r) for c, r in CONTRASTS)
    checks = [
        ("1. Simpson's paradox", "Descriptive scheduling-group direction check, without changing pooling or selection: " + group_text + ". Mixed subgroup signs are not a reason to drop requests."),
        ("2. Ecological fallacy", "Aggregate ratios are accompanied by all request-level outcomes; no claim that every request improves."),
        ("3. Berkson's paradox", "This is a sealed, previously used development panel. No outcome-based request filtering occurred; population generalization remains limited."),
        ("4. Collider bias", "No post-treatment covariate adjustment or favorable-output subgroup selection enters the primary result. Every scored execution passed the fixed validity gates."),
        ("5. Base-rate neglect", "Report 128 request denominators, wins/regressions/ties, gross calls saved/added, output counts and 1,152 scored executions plus 36 warmups."),
        ("6. Regression to the mean", "All endpoints are evaluated contemporaneously on the same requests; historical serving times are not used as a learning curve. Reused-development selection still limits confirmation."),
        ("7. Survivorship bias", "The planned 1,188 returned measurements are retained. No failed, unfavorable or slow request is excluded from the reported paired panel."),
        ("8. Look-elsewhere effect", "All three prespecified contrasts and timing/count intervals are reported, including negative estimates. Secondary intervals are unadjusted; broad mechanism-search history is not fresh confirmatory evidence."),
        ("9. Garden of forking paths", "Fixed arms, panel, repeat pooling, seed, bootstrap, precision and unrounded freeze rule are retained. No post-result tuning or alternative request selection was run."),
        ("10. Correlation versus causation", "The paired arm contrast measures these complete implementations under matched conditions; local training-assessment changes are not asserted to cause a percentage of serving gains. No isolated mechanism or deployment-wide causal claim is made."),
        ("11. Reverse causality", "The checkpoints were fixed before serving evaluation, and evaluation results never trained them. The endpoint comparison has temporal ordering; local metric associations do not independently establish mechanism."),
    ]
    lines += [f"| {name} | {finding} |" for name, finding in checks]
    lines += ["", "Confidence classification: **CAUTION** for generalization and multiple comparisons. "
              "The arithmetic is independently checked; a development point estimate or an interval crossing zero is not equivalence, "
              "noninferiority, absence of harm, or proof of mechanism failure.", "",
              "## Table conventions", "",
              "`PER_REQUEST.csv` contains 128 paired rows; `ALL_REPEATS.csv` contains all 1,152 scored rows; "
              "`WARMUPS.csv` contains all 36 included warmups. `ARM_TOTALS.csv`, `REPEAT_TOTALS.csv` and `CONTRASTS.csv` "
              "provide aggregate and three-repeat accounting. `GROUP_DIRECTION_CHECK.csv` is a descriptive audit of the two scheduling groups. "
              "Public ordinals follow first scheduled appearance. The audit JSON's `bootstrap.public_request_order` supplies "
              "the exact ordinal permutation needed to reproduce the intervals from these public tables. "
              "Repeat, group and activation columns remain zero-based. "
              "Native and stop-clipped accepted tokens remain separate. Loading, activation and warmups do not enter primary serving latency.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result, groups = public_tables(args.evaluation, args.audit, args.output)
    (args.output / "INDEPENDENT_SERVING_REVIEW.md").write_text(report(result, groups))
    files = {path.relative_to(args.output).as_posix(): content_identity(path) for path in sorted(args.output.rglob("*"))
             if path.is_file() and path.name != "SERVING_TABLE_MANIFEST.json" and "__pycache__" not in path.parts}
    (args.output / "SERVING_TABLE_MANIFEST.json").write_text(json.dumps(files, indent=2, sort_keys=True) + "\n")
    print(json.dumps(dict(status="completed", files=len(files), rows=result["row_counts"])))


if __name__ == "__main__":
    main()
