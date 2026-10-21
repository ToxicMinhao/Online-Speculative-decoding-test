"""Frozen P8 diagnostics with exact original PR9/PR10 response bindings.

Only this entry point admits P8. The unchanged L8 evaluator's low-level sampler,
loss, state restoration and reference checks are reused; no optimizer or trainer
is accepted or created. Pool ownership is metadata and never changes a seed.
"""
from __future__ import annotations

import time
from experiments.dflash_lower_lr_control import loss as qualified
from .evidence import composite_identity, validate_pool

author, require, digest = qualified.author, qualified.require, qualified.digest
SAMPLING_FIELDS = qualified.SAMPLING_FIELDS
audit_record = qualified.audit_record
source_binding = qualified.source_binding
source_anchor_seed = qualified.source_anchor_seed
heldout_anchor_seed = qualified.heldout_anchor_seed
validate_reference_input = qualified.validate_reference_input
validate_reference_sampling = qualified.validate_reference_sampling
_rng_state, _restore_rng = qualified._rng_state, qualified._restore_rng
_state_hash, _seed_global = qualified._state_hash, qualified._seed_global
_evaluate_one, source_rng_digest = qualified._evaluate_one, qualified.source_rng_digest


def validate_original_bindings(records, references, original_pool, population):
    """Admit exactly one complete original population without replacing its IDs."""
    validate_pool(original_pool)
    require(population in ('source', 'heldout'), 'Unknown evaluation population')
    pairs = original_pool['reference_pairs'][population]
    require(len(records) == len(references) == len(pairs) == 8,
        'Exactly eight original P8 diagnostic responses required')
    originals = original_pool['support'] if population == 'source' else [p['F']['native'] for p in pairs]
    owners = []
    for record, reference, original, pair in zip(records, references, originals, pairs):
        require(composite_identity(record) == composite_identity(original) == composite_identity(pair),
            'P8 diagnostic original request/role/order differs')
        for key in ('input_ids', 'output_ids', 'record_id', 'episode_id', 'turn', 'target_id',
                    'target_identity_sha256', 'capture_checkpoint_sha256'):
            if key in original:
                require(record.get(key) == original[key], 'P8 original diagnostic field differs: '+key)
        require(reference == pair['F']['kl'], 'P8 diagnostic reference differs from sealed original F')
        audit = audit_record(record)
        validate_reference_input(record, audit, reference, population)
        require(all(key in reference for key in SAMPLING_FIELDS) and
            author._sha256(reference.get('anchor_rng_after_sha256')),
            'Missing original diagnostic sampling/RNG receipt')
        if population == 'source':
            require(source_binding(record, audit) == source_binding(original, audit_record(original)),
                'P8 diagnostic source binding differs')
        owners.append(dict(pool_id=original_pool['pool_id'], pool_request_id=pair['pool_request_id'],
            domain=original_pool['domain'], original_group_id=original['group_id'],
            **{key: original[key] for key in ('episode_id', 'turn', 'target_id',
                'target_identity_sha256', 'capture_checkpoint_sha256') if key in original}))
    return owners


def validate_pool_diagnostics(pool):
    """Pure metadata admission before creating a P8 optimizer or doing training."""
    for population in ('source', 'heldout'):
        pairs = pool['reference_pairs'][population]
        records = pool['support'] if population == 'source' else [p['F']['native'] for p in pairs]
        validate_original_bindings(records, [p['F']['kl'] for p in pairs], pool, population)
    return dict(status='passed', pool_id=pool['pool_id'], source_bindings=8, heldout_bindings=8)


def evaluate_responses(draft, embed_tokens, lm_head, teacher_features, records, references, *, original_pool, population, checkpoint_id='P8',
                       checkpoint_identity_sha256, mask_token_id=128002,
                       attention_backend='flex_attention', projection_chunk=64):
    """Evaluate held-out responses, preserving state even if evaluation raises.

    A successful result may contain explicitly excluded zero-supervision rows.
    Those rows must not enter a loss-improvement claim. Numerical/identity errors
    raise and are never converted into valid measurements. On failure the raised
    exception carries ``diagnostic_partial`` for durable caller-side recording.
    """
    import torch
    require(checkpoint_id == 'P8', 'Only the completed P8 checkpoint is evaluated')
    require(population in ('source', 'heldout'), 'Unknown evaluation population')
    require(author._sha256(checkpoint_identity_sha256), 'Invalid checkpoint identity')
    require(type(mask_token_id) is int and mask_token_id >= 0, 'Invalid mask token')
    require(type(projection_chunk) is int and projection_chunk > 0, 'Invalid projection chunk')
    require(attention_backend in ('sdpa', 'flex_attention'), 'Unsupported attention backend')
    records = list(records)
    require(bool(records), 'No held-out evaluation records supplied')
    references = list(references)
    ownership = validate_original_bindings(records, references, original_pool, population)
    audits = [audit_record(record) for record in records]
    require(len(references) == len(records), 'Missing retained diagnostic references')
    expected_role = 'feedback' if population == 'source' else 'evaluation'
    require(all(r['role'] == expected_role for r in records), 'Response role differs from population')
    seeds = [validate_reference_input(r, a, ref, population)
             for r, a, ref in zip(records, audits, references)]
    require(len({(a['group_id'], a['request_id']) for a in audits}) == len(audits),
            'Duplicate held-out request identity')
    require(not author._has_adaptive_head(draft), 'Adaptive draft heads must remain disabled')
    roots = (draft, embed_tokens, lm_head)
    modules = list(dict.fromkeys(module for root in roots for module in root.modules()))
    parameters = list(dict.fromkeys(parameter for root in roots for parameter in root.parameters()))
    require(bool(parameters), 'Model has no parameters')
    device = parameters[0].device
    require(all(p.device == device and p.grad is None for p in parameters),
            'Read-only evaluation requires colocated weights without pending gradients')
    require(device.type != 'cuda' or all(p.dtype == torch.bfloat16 for p in parameters),
            'Native evaluation must preserve BF16 parameters')
    tensors = parameters + list(dict.fromkeys(buffer for root in roots for buffer in root.buffers()))
    versions = [tensor._version for tensor in tensors]
    parameter_ids = [id(p) for root in roots for p in root.parameters()]
    modes = [(module, module.training) for module in modules]
    grad_flags = [(parameter, parameter.requires_grad) for parameter in parameters]
    global_before = _rng_state()
    started = time.perf_counter()
    values, failure = [], None
    attempted_response_calls = 0
    try:
        for root in roots:
            root.eval()
        for parameter in parameters:
            parameter.requires_grad_(False)
        with torch.no_grad():
            for record, audit, reference, seed, owner in zip(records, audits, references, seeds, ownership):
                row_before = _rng_state()
                try:
                    _seed_global()
                    attempted_response_calls += 1
                    value = _evaluate_one(draft, embed_tokens, lm_head, teacher_features, record, audit,
                        device=device, mask_token_id=mask_token_id, attention_backend=attention_backend,
                        projection_chunk=projection_chunk, seed=seed,
                        rng_digest=source_rng_digest if population == 'source' else _state_hash)
                    value.update(owner, checkpoint_id=checkpoint_id,
                        checkpoint_identity_sha256=checkpoint_identity_sha256,
                        diagnostic_model_mode='eval_no_grad')
                    # Preserve the failed row before checking paired anchor identity.
                    values.append(value)
                    validate_reference_sampling(value, reference)
                finally:
                    _restore_rng(row_before)
                value.update(checkpoint_id=checkpoint_id, checkpoint_identity_sha256=checkpoint_identity_sha256,
                    diagnostic_model_mode='eval_no_grad', training_rng_preserved=True,
                    global_rng_before_sha256=_state_hash(row_before),
                    global_rng_after_sha256=_state_hash(_rng_state()))
                require(value['global_rng_before_sha256'] == value['global_rng_after_sha256'],
                        'Evaluation perturbed global RNG')
        require(parameter_ids == [id(p) for root in roots for p in root.parameters()] and
                versions == [tensor._version for tensor in tensors] and all(p.grad is None for p in parameters),
                'Read-only evaluation mutated model state or gradients')
    except BaseException as error:
        failure = error
        raise
    finally:
        for parameter, flag in grad_flags:
            parameter.requires_grad_(flag)
        for module, mode in modes:
            module.training = mode
        _restore_rng(global_before)
        if failure is not None:
            failure.diagnostic_partial = dict(status='failed', checkpoint_id=checkpoint_id,
                pool_id=original_pool['pool_id'], population=population,
                checkpoint_identity_sha256=checkpoint_identity_sha256, response_losses=values,
                attempted_response_calls=attempted_response_calls, returned_response_rows=len(values),
                unknown_response_calls=attempted_response_calls-len(values),
                unattempted_response_calls=len(records)-attempted_response_calls,
                error_type=type(failure).__name__, error=str(failure),
                global_rng_restored=_state_hash(global_before) == _state_hash(_rng_state()),
                optimizer_steps=0, diagnostic_wall_s=time.perf_counter()-started)
    require(_state_hash(global_before) == _state_hash(_rng_state()), 'Global RNG restoration failed')
    eligible = [v['loss'] for v in values if v['loss_improvement_eligible']]
    wall_s = time.perf_counter()-started
    costs = dict(diagnostic_wall_s=wall_s,
        teacher_preparation_s=sum(v['phase_seconds']['teacher_preparation_s'] for v in values),
        forward_and_loss_s=sum(v['phase_seconds']['forward_and_loss_s'] for v in values),
        accounting='Teacher and forward costs are nested within diagnostic_wall_s; no training/export included')
    return dict(status='passed', checkpoint_id=checkpoint_id, pool_id=original_pool['pool_id'],
        pool_payload_sha256=original_pool['pool_payload_sha256'],
        checkpoint_identity_sha256=checkpoint_identity_sha256, response_losses=values,
        eligible_response_count=len(eligible), excluded_response_count=len(values)-len(eligible),
        attempted_response_calls=attempted_response_calls, returned_response_rows=len(values),
        unknown_response_calls=0, unattempted_response_calls=0,
        loss_mean=sum(eligible)/len(eligible) if eligible else None,
        reduction='equal mean of individually normalized nonzero-supervision response losses',
        population=population, optimizer_steps=0, optimizer_created=False, evaluation_mode=True, gradients_enabled=False,
        global_rng_before_sha256=_state_hash(global_before), global_rng_after_sha256=_state_hash(_rng_state()),
        model_modes_restored=True, parameter_grad_flags_restored=True,
        diagnostic_wall_s=wall_s, costs=costs)
