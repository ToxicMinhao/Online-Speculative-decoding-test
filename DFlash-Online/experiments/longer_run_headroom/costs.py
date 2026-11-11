from pathlib import Path
import argparse, json, collections, time

parser = argparse.ArgumentParser(
    description="Audit bounded replay cost and KV receipts"
)
parser.add_argument("--task", type=Path, required=True)
root = parser.parse_args().task
run = root / "delivery/retained"
read = lambda p: json.loads(p.read_text())
entries = [json.loads(line) for line in (run / "WORK.jsonl").read_text().splitlines()]
attempts = {x["ticket"]: x for x in entries if x.get("status") == "attempted"}
completed = [x for x in entries if x.get("status") == "completed"]
startup = collections.Counter()
durations = collections.defaultdict(float)
operation_counts = collections.Counter()
for row in completed:
    a = attempts[row["ticket"]]
    kind = a["kind"]
    meta = a.get("metadata", {})
    durations[kind] += row["gpu_seconds"] - a["gpu_seconds"]
    operation_counts[(kind, meta.get("operation", ""))] += 1
    if meta.get("startup"):
        startup[kind] += 1
result = read(run / "RESULT.json")
status = read(run / "STATUS.json")
post = read(run / "POSTFLIGHT.json")
neutral = list((run / "replays").rglob("*_NEUTRAL.json"))
child = list((run / "replays").rglob("*_CHILD.json"))
receipt_rows = []
for p in neutral + child:
    r = read(p)
    c = r["reconstruction"]
    assert r["cap_unchanged"] and r["output_matches_canonical_prefix"]
    assert (
        c["status"] == "passed"
        and c["target_prefix_cache_exact"]
        and c["destination_draft_rebuilt"]
        and not c["source_draft_cache_transplanted"]
    )
    if p in neutral:
        assert c["neutral_draft_prefix_cache_exact"]
    receipt_rows.append(
        dict(
            path=str(p.relative_to(run)),
            role="neutral" if p in neutral else "child",
            wall_s=r["wall_s"],
            **c
        )
    )
counts = result["work"]["completed"]
assert counts == result["work"]["attempted"]
assert not result["work"]["failed"] and not result["work"]["unknown"]
assert (
    status["status"] == "completed"
    and status["valid"] == 256
    and status["pending"] == status["invalid"] == 0
)
assert (
    post["source_unchanged"]
    and post["records_unchanged"]
    and all(x["unchanged"] for x in post["checkpoints"])
)
raw_parent_rounds = sum(
    len(read(p)["metadata"]["trace"]) for p in (run / "replays").rglob("PARENT.json")
)
assert (
    raw_parent_rounds + len(neutral) + len(child) + startup["draft_block"]
    == counts["draft_block"]
)
out = dict(
    execution_commit=result["execution_commit"],
    status="passed",
    work=result["work"],
    wrapper_wall_s=result["wall_s"],
    native_rounds=raw_parent_rounds + len(neutral) + len(child),
    parent_rounds=raw_parent_rounds,
    neutral_blocks=len(neutral),
    child_blocks=len(child),
    retained_pairs=1,
    startup=dict(startup),
    nested_operation_wall_s=dict(durations),
    operations=[
        dict(kind=k, operation=o, count=n) for (k, o), n in operation_counts.items()
    ],
    reconstruction_receipts=len(receipt_rows),
    neutral_draft_KV_checks=len(neutral),
    target_KV_checks=len(receipt_rows),
    replay_receipt_wall_s=sum(x["wall_s"] for x in receipt_rows),
    cost_note="Enclosing operation times overlap. Publication is inside walk/child host calls; reconstruction and block execution share each replay receipt. These are asynchronous host-operation clocks, not GPU kernel times or serving latency.",
    no_training=True,
    no_benchmarks=True,
)
(root / "analysis/REPLAY_COSTS.json").write_text(json.dumps(out, indent=2) + "\n")
(root / "analysis/KV_PARITY_RECEIPTS.json").write_text(
    json.dumps(receipt_rows, indent=2) + "\n"
)
print(
    json.dumps(
        {
            k: out[k]
            for k in [
                "wrapper_wall_s",
                "native_rounds",
                "parent_rounds",
                "neutral_blocks",
                "child_blocks",
                "startup",
                "reconstruction_receipts",
            ]
        },
        indent=2,
    )
)
