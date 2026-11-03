"""Fixed scientific scope of the eight-event workload replay probe."""

ARMS = ("K", "U", "R")
EVENTS = 8
RESPONSES = 128
PASSES = 2
REPLAY_COUNT = 32
MEMORY_CAPACITY = 128
TRAINING_SEED = 17
SELECTION_SEED = 20261102
BASE_COMMIT = "4d94f30c42701d0f86ac04e2829c834593c16d8e"
CODING_BINDING = "6def4832d12698fc2733e91c427db77558f3fe50e7f51063e9c2a11f9bc961af"
EVENT_DOMAINS = ("math", "math", "coding", "coding", "math", "math", "coding", "coding")
BUDGET = {
    "pre_update_generations": 3072,
    "shared_feedback_generations": 1024,  # Included in pre-update generations.
    "final_generations": 2304,
    "capture_parity_generations": 4,
    "complete_generations": 5380,
    "terminal_preparation_forwards": 1024,
    "optimizer_steps": 3072,
    "response_forwards": 6144,
    "response_backwards": 6144,
    "replay_presentations": 448,  # Included in response presentations.
    "scoring_model_passes": 0,
    "disposable_optimizer_steps": 0,
}


def serving_order(event):
    """Rotate preceding-checkpoint serving; all serving finishes before learning."""
    if not 1 <= event <= EVENTS:
        raise ValueError("The experiment has exactly eight events")
    shift = (event - 1) % len(ARMS)
    return ARMS[shift:] + ARMS[:shift]


def final_order(block, repeat):
    base = ARMS if block % 2 == 0 else ("K", "R", "U")
    shift = (block + repeat) % 3
    return base[shift:] + base[:shift]


def final_schedule(requests):
    if len(requests) != 256:
        raise ValueError("The fixed endpoint contains 128 math and 128 coding requests")
    slots = []
    for block in range(4):
        for repeat in range(3):
            for position, arm in enumerate(final_order(block, repeat)):
                for ordinal in range(block * 64, (block + 1) * 64):
                    request = requests[ordinal]
                    slots.append({
                        "method": arm, "repeat": repeat, "position": position,
                        "block_ordinal": block, "panel_ordinal": ordinal,
                        "request_id": request["request_id"], "domain": request["domain"],
                    })
    return slots
