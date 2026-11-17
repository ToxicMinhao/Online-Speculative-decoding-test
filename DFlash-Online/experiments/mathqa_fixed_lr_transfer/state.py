"""Complete MathQA N/S state validation; only the declared LR metadata changes.

Historical receipts stay immutable. A metadata-only native view lets the shared
validator check all weights, moments, counters and RNG without weakening its
original K/K_HALF rate and event guards.
"""
from copy import deepcopy
import math

from experiments.full_k_gsm8k import training as native
from experiments.ok_continuation.resume import differences
from experiments.role_loss_pilot import training as historical
from online_feedback.kplus import kplus_contract

CONTRACT_KEY = "mathqa_fixed_lr_transfer_v1"
ARMS = ("N", "S")
NATIVE_LR = 3e-4
SCALAR_LR = 2.88e-4  # Pin the handoff's decimal, not a later floating multiplication.


def arm_lr(arm):
    if arm not in ARMS:
        raise ValueError("MathQA fixed-LR arm must be N or S")
    return SCALAR_LR if arm == "S" else NATIVE_LR


def experiment_contract(arm, binding):
    native.experiment_contract("K", binding)
    return dict(schema=CONTRACT_KEY, arm=arm, dataset="mathqa", binding=binding,
        learning_rate=arm_lr(arm), learning_rate_hex=arm_lr(arm).hex(),
        learning_rate_multiplier=.96 if arm == "S" else 1.,
        first_event=5, final_event=6, inherited_horizon=7424,
        initialization="complete_MathQA_KPLUS4_then_own_complete_event5",
        feedback="common_KPLUS4_event5_then_own_event6", native_loss=kplus_contract(),
        reference=None, seed_reset=False)


def native_base_contract(contract):
    result = dict(contract)
    result.pop(historical.CONTRACT_KEY, None)
    result.pop(CONTRACT_KEY, None)
    return result


def continuation_contract(origin, *, arm, binding):
    return dict(deepcopy(origin["algorithm_contract"]),
                **{CONTRACT_KEY: experiment_contract(arm, binding)})


def _config_with_native_lr(config_text):
    import yaml
    config = yaml.safe_load(config_text)
    config["training"]["learning_rate"] = NATIVE_LR
    return yaml.safe_dump(config)


def validate_config_text(config_text, arm, original_config_text):
    import yaml
    if yaml.safe_load(config_text)["training"]["learning_rate"] != arm_lr(arm):
        raise ValueError("Actual MathQA optimizer LR differs from the sealed arm")
    native.validate_config_text(_config_with_native_lr(config_text), "K", original_config_text)


def optimizer_lr_view(optimizer, learning_rate):
    """Copy mutable metadata only; tensor payloads remain read-only until loading."""
    recipe = deepcopy(optimizer["recipe"])
    recipe["optimizer_options"]["lr"] = learning_rate
    saved = optimizer["optimizer_state_dict"]
    return dict(recipe=recipe, optimizer_state_dict=dict(saved,
        param_groups=[dict(group, lr=learning_rate) for group in saved["param_groups"]]))


def require_optimizer_lr(optimizer, arm):
    expected = arm_lr(arm)
    groups = optimizer["optimizer_state_dict"]["param_groups"]
    if (optimizer["recipe"]["optimizer_options"]["lr"] != expected
            or len(groups) != 1 or any(group["lr"] != expected for group in groups)):
        raise ValueError("Optimizer recipe and actual group LR must agree with the arm")


def validate_origin(state, original_config_text, *, expected_cuda_rng_states=1):
    old = state["algorithm_contract"].get(historical.CONTRACT_KEY, {})
    if old.get("arm") != "KPLUS" or old.get("dataset") != "mathqa" or CONTRACT_KEY in state["algorithm_contract"]:
        raise ValueError("Require the original complete MathQA KPLUS4 origin")
    return historical.validate_snapshot(state, arm="KPLUS", dataset="mathqa",
        binding=old["binding"], config_text=state["config_text"],
        original_config_text=original_config_text, completed_event=4,
        expected_contract=state["algorithm_contract"],
        expected_recipe=state["backend"]["optimizer"]["recipe"],
        expected_cuda_rng_states=expected_cuda_rng_states)


def fork_state(origin, *, arm, binding, config_text, run_id, original_config_text,
               expected_cuda_rng_states=1):
    """Authenticate first, then add arm identity and change only S's LR metadata."""
    validate_origin(origin, original_config_text, expected_cuda_rng_states=expected_cuda_rng_states)
    validate_config_text(config_text, arm, original_config_text)
    optimizer = origin["backend"]["optimizer"]
    require_optimizer_lr(optimizer, "N")
    state = dict(origin, config_text=config_text, run_id=run_id,
        algorithm_contract=continuation_contract(origin, arm=arm, binding=binding),
        backend={"optimizer": optimizer_lr_view(optimizer, arm_lr(arm))})
    receipt = dict(status="passed", arm=arm, source_event=4, source_global_step=512,
        source_presentations=1024, learning_rate_before=NATIVE_LR,
        learning_rate_after=arm_lr(arm), learning_rate_hex=arm_lr(arm).hex(),
        allowed_changes=["run/output identity", "additive N/S contract", "S LR in config, recipe and group"],
        weights_moments_rng_history_unchanged=True, optimizer_reset=False, seed_reset=False,
        optimizer_payload_isolated_on_live_load=True)
    return state, receipt


def event_native_view(event, arm, number):
    expected = NATIVE_LR if number <= 4 else arm_lr(arm)
    for row in event["records"]:
        if row["learning_rate"] != (expected if row["optimizer_stepped"] else None):
            raise ValueError("Event records differ from the actual historical/new learning rate")
    if expected == NATIVE_LR:
        return event
    return dict(event, records=[dict(row,
        learning_rate=NATIVE_LR if row["optimizer_stepped"] else None) for row in event["records"]])


def validate_event_receipt(event, *, arm, binding, event_number,
                           require_completed=True, trainer_status=None):
    if (event_number not in (5, 6) or event.get("method") != arm
            or event.get("dataset") != "mathqa" or event.get("kplus_loss") != kplus_contract()
            or event.get(CONTRACT_KEY) != experiment_contract(arm, binding)):
        raise ValueError("Event lost its immutable MathQA N/S identity")
    checked = native.validate_event_receipt(event_native_view(event, arm, event_number),
        arm="K", event_number=event_number, require_completed=require_completed,
        trainer_status=trainer_status)
    for row in event["records"]:
        observation = row["response_observation"]
        if (observation["record_id"] != row["record_id"]
                or observation["sampled_anchors"] != row["sampled_anchors"]
                or observation["pass_index"] != row["pass"]
                or observation["sum_metrics"]["kplus_response_count"] != 1
                or not all(math.isfinite(float(value)) for value in observation["sum_metrics"].values())):
            raise ValueError("Native K+ observation is not bound to its original forward")
        sums = observation["sum_metrics"]
        total = sums["kplus_W1"] + sums["kplus_W2"]
        tolerance = kplus_contract()["mass_conservation_atol"] + kplus_contract()["mass_conservation_rtol"] * total
        if abs(sums["kplus_first_error_weight_after"] - total) > tolerance:
            raise ValueError("Native K+ first-error mass conservation failed")
    return dict(checked, arm=arm, actual_learning_rate=arm_lr(arm))


def validate_snapshot(state, *, origin, arm, binding, config_text, original_config_text,
                      completed_event, expected_recipe, expected_cuda_rng_states=1):
    if completed_event not in (4, 5, 6):
        raise ValueError("MathQA continuation permits only complete event4/5/6 states")
    validate_config_text(config_text, arm, original_config_text)
    expected = continuation_contract(origin, arm=arm, binding=binding)
    if (differences(state["algorithm_contract"], expected) or state["config_text"] != config_text
            or differences(state["events"][:4], origin["events"])):
        raise ValueError("Foreign own-state contract, config or changed original event history")
    optimizer = state["backend"]["optimizer"]
    require_optimizer_lr(optimizer, arm)
    if differences(optimizer["recipe"], expected_recipe):
        raise ValueError("Saved native optimizer recipe differs")
    for number, event in enumerate(state["events"], 1):
        if number <= 4:
            historical.validate_event_receipt(event, arm="KPLUS", dataset="mathqa",
                                              event_number=number, require_completed=False)
        else:
            validate_event_receipt(event, arm=arm, binding=binding,
                                   event_number=number, require_completed=False)
    old_binding = origin["algorithm_contract"][historical.CONTRACT_KEY]["binding"]
    contract = historical.native_contract_view(origin["algorithm_contract"], old_binding, "KPLUS", "mathqa")
    config = _config_with_native_lr(config_text)
    projected_optimizer = optimizer_lr_view(optimizer, NATIVE_LR)
    view = dict(state, config_text=config, algorithm_contract=contract,
        backend={"optimizer": projected_optimizer},
        events=[event_native_view(row, arm, i) for i, row in enumerate(state["events"], 1)])
    checked = native.validate_snapshot(view, arm="K", binding=old_binding,
        config_text=config, original_config_text=original_config_text,
        completed_event=completed_event, expected_contract=contract,
        expected_recipe=projected_optimizer["recipe"], expected_cuda_rng_states=expected_cuda_rng_states)
    return dict(checked, arm=arm, actual_learning_rate=arm_lr(arm),
                historical_receipts_unchanged=True, seed_reset=False)
