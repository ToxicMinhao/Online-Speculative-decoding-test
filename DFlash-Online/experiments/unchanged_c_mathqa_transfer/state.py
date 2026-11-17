"""Complete MathQA C recovery: immutable native ancestry plus fixed allocation.

Only experiment identity changes when KPLUS4 is forked. Native weights, Adam,
RNG, learning rate and horizon remain intact. C5/C6 preserve their own allocation
contract; a metadata-only native view lets the historical validators check every
ordinary training field without weakening their original no-controller guards.
"""
from copy import deepcopy
import math

from experiments.full_k_gsm8k import training as native
from experiments.mathqa_fixed_lr_transfer import state as baseline
from experiments.ok_continuation.resume import differences
from experiments.role_loss_pilot import training as historical
from online_feedback.kplus import kplus_contract
from online_feedback.prefix_allocation import prefix_allocation_contract

CONTRACT_KEY = "unchanged_c_mathqa_transfer_v1"
NATIVE_LR = 3e-4
RELATIVE_CAP = .25
ARMS = ("C",)


def arm_lr(arm):
    if arm != "C":
        raise ValueError("Only the unchanged C arm may train in this campaign")
    return NATIVE_LR


def experiment_contract(arm, binding):
    native.experiment_contract("K", binding)
    return dict(schema=CONTRACT_KEY, arm=arm, dataset="mathqa", binding=binding,
        learning_rate=arm_lr(arm), learning_rate_hex=arm_lr(arm).hex(),
        first_event=5, final_event=6, inherited_horizon=7424,
        initialization="complete_MathQA_KPLUS4_then_own_complete_C5",
        feedback="retained_PR51_common_event5_then_own_C5_event6",
        native_loss=kplus_contract(), allocation=prefix_allocation_contract(RELATIVE_CAP),
        reference=None, seed_reset=False)


def native_base_contract(contract):
    result = baseline.native_base_contract(contract)
    result.pop(CONTRACT_KEY, None)
    return result


def continuation_contract(origin, *, arm, binding):
    return dict(deepcopy(origin["algorithm_contract"]),
                **{CONTRACT_KEY: experiment_contract(arm, binding)})


def validate_config_text(config_text, arm, original_config_text):
    arm_lr(arm)
    baseline.validate_config_text(config_text, "N", original_config_text)


def require_optimizer_lr(optimizer, arm):
    arm_lr(arm)
    baseline.require_optimizer_lr(optimizer, "N")


def validate_origin(state, original_config_text, *, expected_cuda_rng_states=1):
    if CONTRACT_KEY in state["algorithm_contract"]:
        raise ValueError("C cannot replace the original MathQA KPLUS4 origin")
    return baseline.validate_origin(state, original_config_text,
                                   expected_cuda_rng_states=expected_cuda_rng_states)


def fork_state(origin, *, arm, binding, config_text, run_id, original_config_text,
               expected_cuda_rng_states=1):
    validate_origin(origin, original_config_text, expected_cuda_rng_states=expected_cuda_rng_states)
    validate_config_text(config_text, arm, original_config_text)
    require_optimizer_lr(origin["backend"]["optimizer"], arm)
    state = dict(origin, config_text=config_text, run_id=run_id,
                 algorithm_contract=continuation_contract(origin, arm=arm, binding=binding))
    receipt = dict(status="passed", arm=arm, source_event=4, source_global_step=512,
        source_presentations=1024, learning_rate_before=NATIVE_LR, learning_rate_after=NATIVE_LR,
        learning_rate_hex=NATIVE_LR.hex(), allocation=prefix_allocation_contract(RELATIVE_CAP),
        allowed_changes=["run/output identity", "additive C allocation contract"],
        weights_moments_rng_history_unchanged=True, optimizer_reset=False, seed_reset=False,
        optimizer_payload_isolated_on_live_load=True)
    return state, receipt


def validate_allocation_observation(observation):
    """Validate the immutable per-response reduction, not recomputation hooks."""
    allocation = observation["prefix_allocation"]
    required = ("changed_positions", "changed_blocks", "weight_moved", "prefix_weight_before",
                "prefix_weight_after", "mass_error_abs_sum", "delta_numerator", "mass_violations",
                "bounds_violations", "support_violations", "nonfinite_values")
    if not all(math.isfinite(float(allocation[key])) for key in required):
        raise ValueError("Nonfinite C allocation observation")
    if any(allocation[key] != 0 for key in ("mass_violations", "bounds_violations", "support_violations", "nonfinite_values")):
        raise ValueError("C coefficient invariants failed")
    return allocation


def validate_event_receipt(event, *, arm, binding, event_number,
                           require_completed=True, trainer_status=None):
    if (event_number not in (5, 6) or event.get("method") != "C"
            or event.get("dataset") != "mathqa" or event.get("kplus_loss") != kplus_contract()
            or event.get(CONTRACT_KEY) != experiment_contract(arm, binding)):
        raise ValueError("Event lost the unchanged MathQA C identity")
    # PR51's native validator independently checks original support, first-error
    # mass, response averaging, samples, native LR and every accounting field.
    view = dict(event, method="N", **{baseline.CONTRACT_KEY: baseline.experiment_contract("N", binding)})
    checked = baseline.validate_event_receipt(view, arm="N", binding=binding,
        event_number=event_number, require_completed=require_completed, trainer_status=trainer_status)
    for row in event["records"]:
        validate_allocation_observation(row["response_observation"])
    return dict(checked, arm="C", allocation_contract_preserved=True)


def validate_snapshot(state, *, origin, arm, binding, config_text, original_config_text,
                      completed_event, expected_recipe, expected_cuda_rng_states=1):
    if completed_event not in (4, 5, 6):
        raise ValueError("C permits only complete MathQA event4/5/6 boundaries")
    validate_config_text(config_text, arm, original_config_text)
    expected = continuation_contract(origin, arm=arm, binding=binding)
    if (differences(state["algorithm_contract"], expected) or state["config_text"] != config_text
            or differences(state["events"][:4], origin["events"])):
        raise ValueError("Foreign C contract, config or changed historical receipts")
    optimizer = state["backend"]["optimizer"]
    require_optimizer_lr(optimizer, arm)
    if differences(optimizer["recipe"], expected_recipe):
        raise ValueError("Saved C optimizer recipe differs")
    for number, event in enumerate(state["events"], 1):
        if number <= 4:
            historical.validate_event_receipt(event, arm="KPLUS", dataset="mathqa",
                                              event_number=number, require_completed=False)
        else:
            validate_event_receipt(event, arm=arm, binding=binding,
                                   event_number=number, require_completed=False)
    old_binding = origin["algorithm_contract"][historical.CONTRACT_KEY]["binding"]
    contract = historical.native_contract_view(origin["algorithm_contract"], old_binding, "KPLUS", "mathqa")
    for number, event in enumerate(state["events"], 1):
        baseline.event_native_view(event, "N", number)
    view = dict(state, algorithm_contract=contract)
    checked = native.validate_snapshot(view, arm="K", binding=old_binding,
        config_text=config_text, original_config_text=original_config_text,
        completed_event=completed_event, expected_contract=contract,
        expected_recipe=optimizer["recipe"], expected_cuda_rng_states=expected_cuda_rng_states)
    return dict(checked, arm="C", actual_learning_rate=NATIVE_LR,
        allocation_contract_preserved=True, historical_receipts_unchanged=True, seed_reset=False)
