"""Fixed successor-assignment control at the original GSM8K event-five fork."""

from copy import deepcopy

SCHEMA = "kplus_successor_shuffle_v1"
ARMS = METHODS = ("K", "TRUE", "SHUFFLED")
DATASET = "gsm8k"
DATASETS = (DATASET,)
EVENT = 5
EVENTS = (EVENT,)
START_EVENT, FINAL_EVENT = 4, 5
START_STEP, END_STEP = 512, 640
SEED, SHUFFLE_SEED = 17, 20261111
PANEL_SIZE, ENDPOINT_REPEATS, BLOCK_SIZE = 256, 2, 8
QUALIFICATION_CASES = ("TRUE", "IDENTITY", "SHUFFLED")
TRUE_EXPORT_SHA256 = "9dcb0d637eabef6bd4e34c552da0448faa9510e6bcaf5c753a6bd1b43674b795"
ANALYSIS = dict(
    resamples=10000,
    seed=20261111,
    generator="PCG64",
    percentile_method="linear",
    primary=[["TRUE", "SHUFFLED"]],
    controls=[["TRUE", "K"], ["SHUFFLED", "K"]],
    intervals_adjusted=False,
    uncertainty_unit="paired_request",
    repeats_reduction="within_request_mean",
    effect="percent_reduction_of_paired_totals",
    first_round="shared_start_only",
)
SERVING = dict(
    block_size=16,
    max_new_tokens=2048,
    context_limit=4096,
    concurrency=1,
    greedy=True,
    capture=False,
    extra_warmup_generations=0,
    cache_policy="idle_full_flush_before_every_request_and_publication",
    same_physical_card=True,
)


def training_arms(reuse_true=True):
    if type(reuse_true) is not bool:
        raise ValueError("Seal an explicit TRUE reuse decision")
    return ("K", "SHUFFLED") if reuse_true else ARMS


def expected_work(arm, completed_event=EVENT):
    if (
        arm not in ARMS
        or type(completed_event) is not int
        or completed_event not in (4, 5)
    ):
        raise ValueError("Require an original or event-five boundary")
    steps = 128 * (completed_event - 4)
    return dict(
        optimizer_steps=steps,
        ordinary_presentations=2 * steps,
        response_forwards=2 * steps,
        response_backwards=2 * steps,
    )


def budgets(
    panel_size=PANEL_SIZE, *, reuse_true=True, missing_references=0, parent_blocks=64
):
    if (
        type(panel_size) is not int
        or not 8 <= panel_size <= PANEL_SIZE
        or panel_size % 8
        or type(missing_references) is not int
        or not 0 <= missing_references <= panel_size
        or type(parent_blocks) is not int
        or not 0 <= parent_blocks <= min(64, panel_size)
    ):
        raise ValueError(
            "Seal complete eight-request blocks and bounded extra reference/parent work"
        )
    count = len(training_arms(reuse_true))
    return dict(
        main=dict(
            training_events=count,
            optimizer_steps=128 * count,
            ordinary_presentations=256 * count,
            response_forwards=256 * count,
            response_backwards=256 * count,
            extra_detached_projection_calls_max=256 * 64,
            endpoint_generations=6 * panel_size,
            unique_feedback_responses=128,
            unique_teacher_packets=128,
            feedback_generations=0,
            teacher_preparations=0,
            canonical_generations=missing_references,
            parent_one_block_evaluations=parent_blocks,
        ),
        qualification=dict(
            cases=3,
            optimizer_steps=3,
            ordinary_presentations=6,
            case_forwards=6,
            case_backwards=6,
            startup_forwards=6,
            startup_backwards=6,
            response_forwards=12,
            response_backwards=12,
            extra_detached_projection_calls_max=4 * 64,
            next_anchor_draws=3,
            serving_generations=0,
            automatic_retry=False,
        ),
        startup=dict(
            runtimes=count,
            response_forwards=2 * count,
            response_backwards=2 * count,
            optimizer_steps=0,
        ),
        serving=deepcopy(SERVING),
    )


def experiment_contract(binding, arm, dataset=DATASET, *, origin_binding):
    if arm not in ARMS or dataset != DATASET:
        raise ValueError("Require a prescribed successor-control arm")
    for value in (binding, origin_binding):
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)
        ):
            raise ValueError("Require sealed study and original PR32 bindings")
    return dict(
        schema=SCHEMA,
        binding=binding,
        arm=arm,
        dataset=dataset,
        origin_binding=origin_binding,
        origin_method="KPLUS",
        origin_event=START_EVENT,
        start_adam_step=START_STEP,
        final_adam_step=END_STEP,
        events=list(EVENTS),
        initialization="exact_complete_PR32_GSM8K_KPLUS4",
        seed_reset=False,
        native_sampling=True,
        inherited_horizon=7424,
        feedback="same_128_PR37_saved_responses_packets_order_and_native_anchor_draws",
        loss="native_K" if arm == "K" else "native_KPLUS",
        priority_assignment=(
            "hash_permutation_within_response_and_first_error_position"
            if arm == "SHUFFLED"
            else "native"
        ),
        shuffle_seed=SHUFFLE_SEED if arm == "SHUFFLED" else None,
        eligible_successor="valid_immediate_canonical_including_EOS",
        ineligible_successor="low_priority",
        additional_backbone_forwards=0,
        serving=deepcopy(SERVING),
    )
