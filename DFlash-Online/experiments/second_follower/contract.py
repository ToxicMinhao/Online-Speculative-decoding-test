"""Fixed second-follower treatment; no new serving policy or tuned coefficient."""

from copy import deepcopy
from experiments.successor_shuffle.contract import SERVING, TRUE_EXPORT_SHA256

SCHEMA = "kplus_second_follower_v1"
ARMS = ("NATIVE", "SECOND")
START_STEP, END_STEP = 512, 640
GPU_SECONDS_LIMIT = 4500
ANALYSIS = dict(
    resamples=10000,
    seed=20261113,
    generator="PCG64",
    percentile_method="linear",
    uncertainty_unit="paired_whole_request",
)


def expected_work(arm):
    if arm not in ARMS:
        raise ValueError("Undeclared arm")
    return dict(
        optimizer_steps=128,
        ordinary_presentations=256,
        response_forwards=256,
        response_backwards=256,
    )


def experiment_contract(binding, arm, dataset="gsm8k", *, origin_binding):
    if arm not in ARMS or dataset != "gsm8k":
        raise ValueError("Require fixed GSM8K event-five comparison")
    return dict(
        schema=SCHEMA,
        binding=binding,
        arm=arm,
        dataset=dataset,
        origin_binding=origin_binding,
        origin_method="KPLUS",
        origin_event=4,
        start_adam_step=512,
        final_adam_step=640,
        inherited_horizon=7424,
        seed_reset=False,
        native_sampling=True,
        relative_transfer_cap=0.125,
        rule="min(c*B_SHORT/8,c*B_LONG/8,(c-1)*B_SHORT)",
        pooling="response_and_first_error_depth",
        denominator="unchanged_native_selected_weight_plus_epsilon",
        serving=deepcopy(SERVING),
    )
