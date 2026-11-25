"""Authenticate PR59 PAIR1, then import its scientific state without rewriting it.

This is a fork adapter, not a production trainer. New event/checkpoint validation
must validate the immutable PR59 history separately from the new event contract.
No function here installs the frontier allocator, samples, steps, or saves state.
"""

from copy import deepcopy
import hashlib
import json
import re

import yaml


PARENT_EXECUTION_COMMIT = '416cb7931028621c09b8fa983e53fc8a7dcebaab'
PARENT_PUBLICATION_COMMIT = 'a780c94fbac2c4e5a2d31b22cb99d0c380ec85b2'
PARENT_BINDING = 'b120c76b0036b4786eb6b9d11b1594ec1dc6e57e3d20c9ed931924262b1c072a'
PARENT_IDENTITY = {
    'bytes': 6296473051,
    'sha256': '728770a7abf0ba3ef96805ef760997976ad7bedbbee1d8debe165e785a4d98a2',
}
OLD_CONTRACT_KEY = 'repair_conditional_kl_pilot_v1'
FORK_CONTRACT_KEY = 'frontier_preservation_v1'
ARMS = ('CONTROL', 'FRONTIER_P')
SCIENTIFIC_FIELDS = (
    'draft_state_dict', 'backend', 'rng', 'global_step', 'presentations',
    'events', 'strategy', 'world_size', 'reduction', 'accumulation_steps', 'horizon',
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _native():
    # Lazy import keeps metadata/contract inspection independent of Torch setup.
    from experiments.repair_conditional_kl_pilot import training
    return training


def _differences(first, second):
    from experiments.ok_continuation.resume import differences
    return differences(first, second)


def _digest(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(',', ':'), allow_nan=False,
    ).encode()).hexdigest()


def scientific_state(state):
    """Return a borrowed, read-only comparison view; do not copy multi-GB tensors.

    Callers must not mutate this view. Import below uses the existing isolated
    native loader, which copies parameters and avoids optimizer-counter aliases.
    Run/configuration/algorithm bindings are intentionally compared separately.
    """
    _require(all(key in state for key in SCIENTIFIC_FIELDS), 'Incomplete scientific state')
    return {key: state[key] for key in SCIENTIFIC_FIELDS}


def _parent_boundary(state):
    _require(
        state.get('global_step') == 128 and state.get('presentations') == 256
        and len(state.get('events', [])) == 1
        and state['events'][0].get('event_id') == 'event_001',
        'Fork requires the complete original PR59 event-one boundary',
    )


def validate_parent_snapshot(
    state, *, parent_identity, parent_config_text, original_config_text,
    expected_contract, expected_recipe, expected_cuda_rng_states=1,
):
    """Run the ORIGINAL validator with the ORIGINAL PAIR/plan/config bindings.

    ``parent_identity`` must come from the caller's actual file hash/readback,
    not from this constant. This function validates the loaded state; it does
    not open the recovery file or claim to have authenticated bytes itself.
    """
    _require(parent_identity == PARENT_IDENTITY, 'Parent recovery identity differs')
    _parent_boundary(state)
    native = _native()
    old = native.experiment_contract('PAIR', PARENT_BINDING, maximum_events=1)
    _require(
        expected_contract.get(OLD_CONTRACT_KEY) == old
        and state.get('algorithm_contract', {}).get(OLD_CONTRACT_KEY) == old,
        'Validate the original PR59 PAIR contract; do not rebind its history',
    )
    _require(state.get('config_text') == parent_config_text, 'Original parent config differs')
    result = native.validate_snapshot(
        state, arm='PAIR', binding=PARENT_BINDING,
        config_text=parent_config_text, original_config_text=original_config_text,
        completed_event=1, maximum_events=1, expected_contract=expected_contract,
        expected_recipe=expected_recipe, expected_cuda_rng_states=expected_cuda_rng_states,
    )
    _require(result.get('status') == 'passed', 'Original native parent validation failed')
    return dict(
        status='passed', schema='frontier_parent_validation_v1',
        parent_execution_commit=PARENT_EXECUTION_COMMIT,
        parent_publication_commit=PARENT_PUBLICATION_COMMIT,
        parent_plan_binding=PARENT_BINDING, parent_identity=deepcopy(parent_identity),
        original_config_sha256=hashlib.sha256(parent_config_text.encode()).hexdigest(),
        parent_history_digest=_digest(state['events']),
        global_step=128, presentations=256, completed_event=1,
        native_validation=deepcopy(result), model_work=0,
    )


def fork_contract(arm, *, source_binding):
    """Declare the new arm explicitly; CONTROL never means native arm BASE."""
    _require(arm in ARMS, 'Unknown frontier fork arm')
    _require(
        isinstance(source_binding, str) and re.fullmatch('[0-9a-f]{64}', source_binding),
        'Require the sealed new experiment binding',
    )
    return dict(
        name=FORK_CONTRACT_KEY, version=1, arm=arm, source_binding=source_binding,
        parent_execution_commit=PARENT_EXECUTION_COMMIT,
        parent_publication_commit=PARENT_PUBLICATION_COMMIT,
        parent_plan_binding=PARENT_BINDING, parent_identity=deepcopy(PARENT_IDENTITY),
        native_runtime_arm='PAIR', pair_enabled=True,
        preservation_allocator=(
            'native_C_alpha_0.5_cap_0.25' if arm == 'CONTROL'
            else 'frontier_preservation_v1_signed_window_sensitivity'
        ),
        prefix_relative_cap=0.25,
        scoring_reference='beta_prefix_with_native_Kplus_PAIR_repair',
        sensitivity='mixed_precision_FP32_optimizer_shadow_native_BF16_model',
        parent_step=128, parent_presentations=256, parent_completed_event=1,
        new_events=['event_002'], additional_optimizer_steps=128,
        additional_presentations=256, final_step=256, final_presentations=512,
        final_completed_event=2, inherited_horizon=7424,
        feedback='one_authenticated_common_parent_buffer',
        reset_optimizer=False, reset_seed=False, rewrite_parent_history=False,
    )


def validate_fork_contract(contract):
    _require(isinstance(contract, dict), 'Malformed fork contract')
    expected = fork_contract(contract.get('arm'), source_binding=contract.get('source_binding'))
    _require(contract == expected, 'Fork contract differs from the authorized one-event scope')
    return expected


def bind_fork_metadata(parent_validation, *, contract, parent_config_text, new_config_text):
    """Return NEW metadata, leaving parent config, contract and history untouched.

    This receipt is not a checkpoint and is not proof that FRONTIER_P is installed.
    Production integration must install/check the declared allocator and validate
    event_002 under the new binding rather than rewriting old event receipts.
    """
    validate_fork_contract(contract)
    _require(
        parent_validation.get('status') == 'passed'
        and parent_validation.get('schema') == 'frontier_parent_validation_v1'
        and parent_validation.get('parent_plan_binding') == PARENT_BINDING
        and parent_validation.get('parent_identity') == PARENT_IDENTITY
        and parent_validation.get('original_config_sha256')
        == hashlib.sha256(parent_config_text.encode()).hexdigest(),
        'A successful original-parent validation is required before new metadata binding',
    )
    before, after = yaml.safe_load(parent_config_text), yaml.safe_load(new_config_text)
    _require(isinstance(before, dict) and isinstance(after, dict), 'Malformed native configuration')
    comparable = deepcopy(after)
    for key in ('run_id', 'output_dir'):
        if key in before:
            comparable[key] = before[key]
        else:
            comparable.pop(key, None)
    _require(comparable == before, 'Fork configuration changed beyond run/output metadata')
    _require(isinstance(after.get('run_id'), str) and bool(after['run_id']), 'Missing new run identity')
    return dict(
        schema='frontier_fork_binding_v1', status='bound_not_runtime_qualified',
        contract=deepcopy(contract), parent_validation=deepcopy(parent_validation),
        run_id=after['run_id'], config_sha256=hashlib.sha256(new_config_text.encode()).hexdigest(),
        parent_history_policy='retain exact event_001; separately validate new event_002',
        allocator_runtime_verified=False, model_work=0,
    )


def import_parent_state(trainer, state, *, contract, **parent_validation_arguments):
    """Authenticate, restore exact warm science, and verify without fresh-origin paths.

    The compatible native trainer is constructed as arm='PAIR' for BOTH forks.
    Its constructor may seed/build temporary modules, but this import restores
    every parent RNG stream after construction, before any training draw.
    It intentionally does not change ``trainer.contract`` or install an allocator.
    """
    validate_fork_contract(contract)
    _require(getattr(trainer, 'arm', None) == 'PAIR', 'Both fork runtimes must retain PAIR')
    _require(
        getattr(trainer.bundle.model, 'repair_conditional_kl', None) is True,
        'PAIR must be installed before importing either fork',
    )
    _require(not trainer.closed and not trainer.failed, 'Cannot import into a closed/failed trainer')
    _require(
        not trainer.core.accumulation_remainder
        and not trainer.core._ratio_totals and not trainer.core._sum_totals,
        'Import requires a clean native accumulation boundary',
    )
    _require(
        all(parameter.grad is None for parameter in trainer.bundle.draft_model.parameters()),
        'Import requires cleared live gradients',
    )
    receipt = validate_parent_snapshot(state, **parent_validation_arguments)
    native = _native()
    native.PilotTrainer._load_state(trainer, state)
    actual = native.native.live_state(trainer)
    mismatch = _differences(scientific_state(actual), scientific_state(state))
    _require(not mismatch, 'Imported scientific state differs: ' + repr(mismatch[:8]))
    _require(trainer.core._micro == 256, 'Native accumulation counter was not restored')
    _require(_digest(state['events']) == receipt['parent_history_digest'], 'Parent history mutated')
    return dict(
        status='passed', fork_arm=contract['arm'], native_runtime_arm='PAIR',
        exact_weights_optimizer_rng_counters_history=True,
        parent_validation=receipt, model_work=0,
        allocator_runtime_verified=False,
        next_required='Bind new metadata and install/validate the declared preservation allocator',
    )
