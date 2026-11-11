"""Bind authenticated child records only after the parent sample is sealed."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import time

from experiments.verification_transition.data import identity, read, write
from .census import validate_walk, SOURCE_COMMIT
from .pairs import retained_same_state, score


def bind(evidence, census, output):
    started = time.perf_counter()
    evidence, census, output = map(Path, (evidence, census, output))
    output.mkdir(parents=True, exist_ok=False)
    seal = read(census / "SEAL.json")
    for name, expected in seal["files"].items():
        if identity(census / name) != expected:
            raise ValueError("Parent census/sample changed after sealing")
    selected = read(census / "SELECTED_STARTS.json")
    plan = read(evidence / "PLAN.json")
    if plan["execution_commit"] != SOURCE_COMMIT:
        raise ValueError("Unexpected retained execution")
    inventory = read(evidence / "EVIDENCE_MANIFEST.json")["files"]
    if identity(evidence / "PLAN.json") != inventory["PLAN.json"]:
        raise ValueError("PR34 source plan changed after parent selection")
    initial_rows, jobs, checked = [], [], []
    remote = "/data/sources/minhao/runs/dflash_verification_transition_replay_20261110/diagnostic_001"
    for item in plan["requests"]:
        own = [
            s
            for s in selected["starts"]
            if (s["dataset"], s["request_id"]) == (item["dataset"], item["request_id"])
        ]
        if not own:
            continue
        records = {}
        for arm in ("KPLUS4", "KPLUS6"):
            relative = f'campaign/raw/{item["dataset"]}/{item["selection_rank"]:03d}/{arm}_native.json'
            path = evidence / relative
            if identity(path) != inventory[relative]:
                raise ValueError("Retained walk changed: " + relative)
            records[arm] = read(path)
            validate_walk(records[arm], item)
            if (
                records[arm]["checkpoint_id"]
                != plan["checkpoints"][item["dataset"]][arm]["export"]
            ):
                raise ValueError("Child checkpoint identity changed")
            checked.append(dict(path=remote + "/" + relative, **inventory[relative]))
        p, c = records["KPLUS4"], records["KPLUS6"]
        reconstruct = []
        for s in own:
            index = s["parent_round_ordinal"] - 1
            if retained_same_state(p, c, s["parent_round_ordinal"]):
                initial_rows.append(
                    score(
                        s,
                        p["metadata"]["trace"][index],
                        c["metadata"]["trace"][index],
                        item["canonical"]["response_ids"],
                        provenance="retained_exact_prior_native_walk",
                    )
                )
            else:
                initial_rows.append(
                    dict(
                        s,
                        status="pending",
                        reason="native state requires reconstruction",
                    )
                )
                reconstruct.append(s)
        if reconstruct:
            jobs.append(
                dict(
                    item=item,
                    starts=sorted(reconstruct, key=lambda s: s["parent_round_ordinal"]),
                    parent_path=f'{remote}/campaign/raw/{item["dataset"]}/{item["selection_rank"]:03d}/KPLUS4_native.json',
                    expected_parent_rounds=len(p["metadata"]["trace"]),
                )
            )
    write(output / "INITIAL_ROWS.json", initial_rows)
    bound = dict(
        source_execution_commit=SOURCE_COMMIT,
        shared=plan["shared"],
        checkpoints=plan["checkpoints"],
        original_requests=plan["requests"],
        parent_seal=seal,
        source_plan_identity=identity(evidence / "PLAN.json"),
        source_evidence_manifest=identity(evidence / "EVIDENCE_MANIFEST.json"),
        raw_records=checked,
        jobs=jobs,
        initial_rows=initial_rows,
        numerical_policy="unchanged native BF16/FA3; greedy b16; context4096; cap2048 or EOS; C1",
        limits=dict(
            original_completion=128,
            neutral_block=256,
            child_block=256,
            draft_block=8000,
        ),
        gpu_seconds=2700,
        bootstrap_seed=20261112,
        bootstrap_draws=10000,
        thresholds=dict(A=0.01, B=0.5),
        work_plan=dict(
            parent_walks=len(jobs),
            parent_rounds=sum(j["expected_parent_rounds"] for j in jobs),
            neutral_blocks=sum(len(j["starts"]) for j in jobs),
            child_blocks=sum(len(j["starts"]) for j in jobs),
            retained_pairs=sum(r["status"] == "valid" for r in initial_rows),
            cpu_preparation_s=time.perf_counter() - started,
        ),
        source_state_reuse_rule="All earlier full native blocks match, plus prefix/seed/positions/cap. Pinned native ingestion boundaries follow those identical full blocks; otherwise reconstruct.",
    )
    write(output / "PLAN.json", bound)
    write(
        output / "SEAL.json",
        dict(
            plan=identity(output / "PLAN.json"),
            parent_selection=identity(census / "SELECTED_STARTS.json"),
            source_execution_commit=SOURCE_COMMIT,
            child_read_after_parent_seal=True,
        ),
    )
    return bound["work_plan"]


if __name__ == "__main__":
    import json

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evidence", required=True)
    p.add_argument("--census", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    print(json.dumps(bind(a.evidence, a.census, a.output), indent=2))
