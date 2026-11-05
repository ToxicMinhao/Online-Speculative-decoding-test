"""Prespecified workload, policy identities and accounting."""

import hashlib
import json

ARMS = ("FULL", "ONCE", "RANDOM64", "STATIC64", "PROGRESS64")
EVENTS = 4
RESPONSES = 128
PASS2 = dict(FULL=128, ONCE=0, RANDOM64=64, STATIC64=64, PROGRESS64=64)
SCHEMA = "vfd_second_pass_v1"
ORIGINAL_EXECUTION = "e0a9d3c235e1efd15404c549282d21c99243a21c"
BUDGET = dict(
    feedback_generations=512,
    teacher_preparations=512,
    optimizer_steps=1664,
    training_presentations=3328,
    scoring_forwards=1408,
    timed_generations=2000,
    canonical_generations=200,
    qualification_scoring_forwards=8,
    qualification_training_presentations=4,
    qualification_optimizer_steps=2,
)
SERVING = dict(
    block_size=16,
    max_new_tokens=2048,
    context_limit=4096,
    concurrency=1,
    precision="bfloat16",
    temperature=0,
    attention_backend="fa3",
    capture=False,
)


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def logical_steps(arm, events):
    return events * (RESPONSES + PASS2[arm]) // 2


def physical_training(events):
    if not 0 <= events <= EVENTS:
        raise ValueError("The screen contains exactly four events")
    # Four extra copies of event-one pass one are logical, not physical work.
    steps = sum(logical_steps(arm, events) for arm in ARMS) - (256 if events else 0)
    return dict(optimizer_steps=steps, training_presentations=2 * steps)


def validate_order(order, identities):
    if len(order) != len(identities) or sorted(order) != sorted(identities):
        raise ValueError("Shared pass order must contain each response exactly once")


def validate_subset(order, selected, arm):
    if len(selected) != PASS2[arm] or len(set(selected)) != len(selected):
        raise ValueError("Second-pass selection has the wrong distinct response count")
    if not set(selected).issubset(order):
        raise ValueError("Selected feedback is outside the current event")
    return [record_id for record_id in order if record_id in set(selected)]


def validate_plan(plan):
    fixed = dict(
        schema=SCHEMA,
        arms=list(ARMS),
        events=EVENTS,
        seed=17,
        budget=BUDGET,
        serving=SERVING,
        original_execution=ORIGINAL_EXECUTION,
        automatic_retry=False,
        inherited_optimizer_horizon=7424,
    )
    if any(plan.get(key) != value for key, value in fixed.items()):
        raise ValueError("The sealed experiment differs from the fixed handoff")
    if len(plan["training_requests"]) != 512 or len(plan["evaluation_requests"]) != 200:
        raise ValueError(
            "Require the original 512 feedback requests and sealed 200-question endpoint"
        )
    if (
        plan["execution_source"]["dirty"]
        or len(plan["execution_source"]["commit"]) != 40
    ):
        raise ValueError(
            "Publish and seal the exact clean implementation before GPU work"
        )
    return dict(status="passed", budget=BUDGET)
