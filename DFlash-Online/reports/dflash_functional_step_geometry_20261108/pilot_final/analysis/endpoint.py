"""Reproduce the completed dataset endpoint from immutable native records.

This program reads JSON and runs the prespecified NumPy bootstrap. It never
constructs a model, optimizer, serving backend, or replacement measurement.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

CODE = EVIDENCE = OUTPUT = None
DATASET = None
COMMIT = "6440e8ff5af1a845f830d65c884f6711d7e4a687"


def read(path):
    return json.loads(Path(path).read_text())


def write(name, value):
    (OUTPUT / name).write_text(json.dumps(value, indent=2) + "\n")


def csv_table(name, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with (OUTPUT / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def flattened(value, prefix=""):
    result = {}
    for name, item in value.items():
        key = prefix + name
        if isinstance(item, dict):
            result.update(flattened(item, key + "."))
        elif isinstance(item, (tuple, list)):
            result[key] = json.dumps(item)
        else:
            result[key] = item
    return result


def main():
    global CODE, EVIDENCE, OUTPUT, DATASET
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code", required=True, type=Path, help="Pinned DFlash-Online checkout")
    parser.add_argument("--evidence", required=True, type=Path, help="Extracted private pilot snapshot")
    parser.add_argument("--output", required=True, type=Path, help="New analysis output directory")
    parser.add_argument("--dataset", choices=("gsm8k", "mathqa"), required=True)
    args = parser.parse_args()
    DATASET = args.dataset
    CODE, EVIDENCE, OUTPUT = args.code.resolve(), args.evidence.resolve(), args.output.resolve()
    started = time.perf_counter()
    OUTPUT.mkdir(exist_ok=False)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for path in (CODE / "inference", CODE / "training", CODE):
        sys.path.insert(0, str(path))
    from experiments.functional_step_geometry.evaluation import audit_stage
    from experiments.functional_step_geometry.statistics import reduce_stage, summarize

    plan = read(EVIDENCE / "sealed/PLAN.json")
    seal = read(EVIDENCE / "sealed/SEAL.json")
    assert plan["execution_source"]["commit"] == COMMIT
    source_origin = Path(plan["execution_source"]["source_path"])
    source_files = []
    for name, expected in seal["files"].items():
        remote = Path(name)
        if remote.is_relative_to(source_origin) and remote.suffix == ".py":
            local = CODE / remote.relative_to(source_origin)
            payload = local.read_bytes()
            actual = dict(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
            if actual != expected:
                raise ValueError("Analysis source differs from execution: " + str(local))
            source_files.append(dict(path=str(remote.relative_to(source_origin)), **actual))
    if not source_files:
        raise ValueError("No execution Python files authenticated")
    write("SOURCE_CHECK.json", dict(execution_commit=COMMIT, files=source_files, status="passed"))

    def local(recorded):
        return EVIDENCE / Path(recorded).relative_to(plan["root"])

    dataset = plan["datasets"][DATASET]
    requests = read(local(dataset["panel_manifest"]))["requests"]
    canonical = read(local(dataset["references"]))
    checkpoint_records = {
        arm: read(EVIDENCE / f"campaign_001/{DATASET}/event_006/{arm}/RESULT.json")
        for arm in ("KPLUS", "FSG", "SCALAR")
    }
    checkpoints = {arm: row["checkpoint_path"] for arm, row in checkpoint_records.items()}
    folder = EVIDENCE / f"campaign_001/evaluation/{DATASET}"
    audit = audit_stage(folder, requests, checkpoints, canonicals=canonical,
                        dataset=DATASET, event=6,
                        schedule=dataset["evaluation_schedule"], evidence_root=EVIDENCE)
    write("RAW_AUDIT.json", audit)
    if audit["status"] != "passed":
        raise ValueError("Raw endpoint evidence invalid: " + repr(audit["errors"]))

    summary = summarize(audit["rows"], dataset=DATASET, event=6)
    if summary != read(folder / "SUMMARY.json"):
        write("RECOMPUTED_SUMMARY.json", summary)
        raise ValueError("Recomputed statistics differ from the retained endpoint summary")
    write("SUMMARY.json", summary)
    paired = reduce_stage(audit["rows"], dataset=DATASET, event=6)
    write("PAIRED.json", paired)
    csv_table("requests.csv", [flattened(row) for row in paired])
    csv_table("repeats.csv", [{k: v for k, v in row.items() if not isinstance(v, (dict, list))}
                              for row in audit["rows"]])

    methods = []
    for method, values in summary["method_totals"].items():
        row = dict(method=method, requests=len(paired), **values)
        row["mean_complete_request_time_s"] = values["request_latency_s"] / len(paired)
        methods.append(row)
    csv_table("methods.csv", methods)
    comparisons = []
    for contrast, values in summary["comparisons"].items():
        row = dict(comparison=contrast, **flattened(values))
        for measure in ("call_reduction", "request_latency_s_reduction"):
            reduction = values[measure]
            row[measure + ".increase_pct"] = -reduction["point_pct"]
            row[measure + ".increase_ci95_low_pct"] = -reduction["ci95_pct"][1]
            row[measure + ".increase_ci95_high_pct"] = -reduction["ci95_pct"][0]
        comparisons.append(row)
    csv_table("comparisons.csv", comparisons)
    csv_table("sweeps.csv", [dict(sweep=repeat, method=method, **values)
                             for repeat, methods in summary["repeat_totals"].items()
                             for method, values in methods.items()])
    write("VALIDITY.json", dict(status="passed", requests=len(paired), executions=len(audit["rows"]),
        errors=audit["errors"], native_progress=audit["progress"],
        retained_summary_exact=True, analysis_source=COMMIT,
        checkpoint_identities={arm: {key: row[key] for key in
            ("checkpoint_path", "weights_identity", "state_path", "state_identity")}
            for arm, row in checkpoint_records.items()},
        boundary="Complete request includes native generation and recorded cache maintenance; loading/activation excluded",
        new_model_work=0, bootstrap_unit="paired request, timing repeats averaged within request",
        wall_s=time.perf_counter()-started, peak_host_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    print(json.dumps(dict(status="passed", requests=len(paired), executions=len(audit["rows"]),
                          wall_s=time.perf_counter()-started, new_model_work=0)))


if __name__ == "__main__":
    main()
