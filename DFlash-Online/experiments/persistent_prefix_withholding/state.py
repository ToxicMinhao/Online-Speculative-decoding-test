"""Own-state boundaries and the one explicitly permitted S learning-rate fork.

Validation projects metadata into the historical K validator only after checking
our actual arm contract and actual learning rates. Model tensors, Adam moments,
RNG and historical receipts are never normalized or reconstructed.
"""
from copy import deepcopy

from experiments.full_k_gsm8k import training as native
from experiments.functional_step_geometry import training as geometry
from experiments.ok_continuation.resume import differences
from experiments.recent_repair_retention import training as retention
from experiments.role_loss_pilot import training as historical
from online_feedback.kplus import kplus_contract

CONTRACT_KEY = "persistent_prefix_withholding_v2"
ARMS = ("N", "E", "S")
NATIVE_LR = 3e-4
SCALAR_MULTIPLIER = 0.96
SCALAR_LR = float(NATIVE_LR) * SCALAR_MULTIPLIER


def arm_lr(arm):
    if arm not in ARMS:
        raise ValueError("Persistent arm must be N, E or S")
    return SCALAR_LR if arm == "S" else NATIVE_LR


def experiment_contract(arm, binding):
    native.experiment_contract("K", binding)
    return dict(schema=CONTRACT_KEY, arm=arm, binding=binding,
                learning_rate=arm_lr(arm), learning_rate_hex=arm_lr(arm).hex(),
                first_new_event=7 if arm != "N" else 8, final_event=8,
                prefix_gradient="stopped" if arm == "E" else "native",
                current_roles=True, inherited_horizon=7424,
                learning_rate_multiplier=SCALAR_MULTIPLIER if arm == "S" else 1.,
                native_loss=kplus_contract(), reference=None, seed_reset=False)


def native_base_contract(contract):
    """Remove only authenticated experiment identities, retaining objective pins."""
    result = dict(contract)
    for key in (historical.CONTRACT_KEY, geometry.CONTRACT_KEY, retention.SCHEMA, CONTRACT_KEY):
        result.pop(key, None)
    return result


def native_view(contract):
    old = contract[historical.CONTRACT_KEY]
    base = native_base_contract(contract)
    base[native.CONTRACT_KEY] = native.experiment_contract("K", old["binding"])
    return base


def _config_with_native_lr(config_text):
    import yaml
    config = yaml.safe_load(config_text)
    config["training"]["learning_rate"] = NATIVE_LR
    return yaml.safe_dump(config)


def validate_config_text(config_text, arm, original_config_text):
    import yaml
    if yaml.safe_load(config_text)["training"]["learning_rate"] != arm_lr(arm):
        raise ValueError("Actual persistent arm learning rate differs")
    native.validate_config_text(_config_with_native_lr(config_text), "K", original_config_text)


def _optimizer_metadata_view(optimizer, learning_rate):
    recipe = deepcopy(optimizer["recipe"])
    recipe["optimizer_options"]["lr"] = learning_rate
    saved = optimizer["optimizer_state_dict"]
    groups = [dict(group, lr=learning_rate) for group in saved["param_groups"]]
    return dict(recipe=recipe, optimizer_state_dict=dict(saved, param_groups=groups))


def _event_native_view(event, arm, number):
    """Validate S's actual LR before adapting its receipt for the old K checker."""
    expected = arm_lr(arm) if number >= 7 else NATIVE_LR
    records = event["records"]
    for row in records:
        if row["learning_rate"] != (expected if row["optimizer_stepped"] else None):
            raise ValueError("Actual event learning rate differs from its historical/new contract")
    if expected == NATIVE_LR:
        return event
    return dict(event, records=[dict(row, learning_rate=NATIVE_LR if row["optimizer_stepped"] else None)
                                for row in records])


def validate_event_receipt(event, *, arm, binding, event_number, require_completed=True, trainer_status=None):
    if event_number not in (7, 8) or (arm == "N" and event_number != 8):
        raise ValueError("Persistent new-event identity differs")
    if (event.get("method") != arm or event.get(CONTRACT_KEY) != experiment_contract(arm, binding)
            or event.get("kplus_loss") != kplus_contract()):
        raise ValueError("Persistent event lost its immutable arm/objective binding")
    result = native.validate_event_receipt(_event_native_view(event, arm, event_number), arm="K",
        event_number=event_number, require_completed=require_completed, trainer_status=trainer_status)
    for record in event["records"]:
        observation = record["response_observation"]
        if (observation["record_id"] != record["record_id"]
                or observation["sampled_anchors"] != record["sampled_anchors"]
                or observation["sum_metrics"]["kplus_response_count"] != 1):
            raise ValueError("Response observations do not identify their original forward")
        roles = record["prefix_roles"]
        if roles["role"] != ("E" if arm == "E" else "N"):
            raise ValueError("A persistent presentation used another arm's gradient role")
    return dict(result, arm=arm)


def validate_legacy_origin(state, original_config_text, completed_event, expected_cuda_rng_states=1):
    """Authenticate existing K6 or native N7 without rewriting the historical gate."""
    if completed_event == 6:
        return retention.validate_origin(state, original_config_text,
                                         expected_cuda_rng_states=expected_cuda_rng_states)
    if completed_event != 7 or CONTRACT_KEY in state["algorithm_contract"]:
        raise ValueError("Only original K6 or authenticated native N7 may seed this pilot")
    contract = state["algorithm_contract"]
    if contract.get(retention.SCHEMA, {}).get("arm") != "NATIVE":
        raise ValueError("N8 must continue original native N7, not a retention candidate")
    origin_contract = dict(contract)
    origin_contract.pop(retention.SCHEMA)
    view = retention.native_view(dict(state, algorithm_contract=origin_contract))
    result = native.validate_snapshot(dict(state, algorithm_contract=view), arm="K",
        binding=view[native.CONTRACT_KEY]["binding"], config_text=state["config_text"],
        original_config_text=original_config_text, completed_event=7,
        expected_contract=view, expected_recipe=state["backend"]["optimizer"]["recipe"],
        expected_cuda_rng_states=expected_cuda_rng_states)
    _validate_history(state["events"], arm="N", binding=None)
    return dict(result, native_N7_reused=True)


def _validate_history(events, arm, binding):
    for number, event in enumerate(events, 1):
        if number <= 4:
            historical.validate_event_receipt(event, arm="KPLUS", dataset="gsm8k",
                                              event_number=number, require_completed=False)
        elif number <= 6:
            geometry.validate_event_receipt(event, arm="KPLUS", dataset="gsm8k",
                                            event_number=number, require_completed=False)
        elif number == 7 and arm == "N":
            retention.validate_event(event, "NATIVE", complete=False)
        else:
            validate_event_receipt(event, arm=arm, binding=binding, event_number=number,
                                   require_completed=False)


def validate_snapshot(state, *, arm, binding, config_text, original_config_text, completed_event,
                      expected_contract, expected_recipe, expected_cuda_rng_states=1, qualification_only=False):
    """Validate a complete own trajectory; only validation metadata is projected."""
    if completed_event not in (6, 7, 8) or (arm == "N" and completed_event < 7 and not qualification_only):
        raise ValueError("Persistent state is outside its authorized event boundaries")
    validate_config_text(config_text, arm, original_config_text)
    contract = state["algorithm_contract"]
    if (contract.get(CONTRACT_KEY) != experiment_contract(arm, binding)
            or differences(contract, expected_contract) or state["config_text"] != config_text):
        raise ValueError("Foreign persistent state or changed algorithm/configuration")
    optimizer = state["backend"]["optimizer"]
    if differences(optimizer["recipe"], expected_recipe):
        raise ValueError("Persistent optimizer recipe differs")
    if (optimizer["recipe"]["optimizer_options"]["lr"] != arm_lr(arm)
            or any(group["lr"] != arm_lr(arm) for group in optimizer["optimizer_state_dict"]["param_groups"])):
        raise ValueError("Saved optimizer recipe/group LR mismatch")
    _validate_history(state["events"], arm, binding)
    view = native_view(contract)
    projected_optimizer = _optimizer_metadata_view(optimizer, NATIVE_LR)
    projected_config = _config_with_native_lr(config_text)
    projected = dict(state, algorithm_contract=view, config_text=projected_config,
        backend={"optimizer": projected_optimizer},
        events=[_event_native_view(event, arm, number) for number, event in enumerate(state["events"], 1)])
    result = native.validate_snapshot(projected, arm="K", binding=view[native.CONTRACT_KEY]["binding"],
        config_text=projected_config, original_config_text=original_config_text,
        completed_event=completed_event, expected_contract=view,
        expected_recipe=projected_optimizer["recipe"], expected_cuda_rng_states=expected_cuda_rng_states)
    return dict(result, arm=arm, actual_learning_rate=arm_lr(arm), historical_receipts_unchanged=True)


def fork_state(state, *, arm, binding, config_text, run_id, qualification_only=False):
    """Metadata-only fork after origin validation; no model-sized duplicate.

    The caller validates the original K6/N7 before invoking this function. All
    scientific tensor/RNG/history objects remain shared read-only until the live
    runtime copies them to the selected device.
    """
    completed = len(state["events"])
    expected = 7 if arm == "N" and not qualification_only else 6
    if completed != expected or CONTRACT_KEY in state["algorithm_contract"]:
        raise ValueError("A fork is permitted once, from original K6 or native N7")
    optimizer = state["backend"]["optimizer"]
    if (optimizer["recipe"]["optimizer_options"]["lr"] != NATIVE_LR
            or len(optimizer["optimizer_state_dict"]["param_groups"]) != 1
            or optimizer["optimizer_state_dict"]["param_groups"][0]["lr"] != NATIVE_LR):
        raise ValueError("Original optimizer must have the qualified native learning rate")
    contract = dict(state["algorithm_contract"], **{CONTRACT_KEY: experiment_contract(arm, binding)})
    result = dict(state, algorithm_contract=contract, config_text=config_text, run_id=run_id,
                  backend={"optimizer": _optimizer_metadata_view(optimizer, arm_lr(arm))})
    receipt = dict(status="passed", arm=arm, source_event=completed,
        source_global_step=state["global_step"], learning_rate_before=NATIVE_LR,
        learning_rate_after=arm_lr(arm), learning_rate_repr=repr(arm_lr(arm)),
        learning_rate_hex=arm_lr(arm).hex(), weights_moments_rng_history_shared_unchanged=True,
        allowed_changes=["run_id", "config_text run/output identity and S LR", "additive arm contract",
                         "optimizer recipe/group LR" if arm == "S" else "optimizer metadata copied unchanged"],
        full_origin_copy=False, optimizer_reset=False, rng_reset=False)
    return result, receipt
