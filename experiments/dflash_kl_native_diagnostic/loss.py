"""Read-only held-out KL using the qualified AdaFlash OSD loss unchanged.

This module never constructs a trainer or optimizer. Evaluation records remain
held-out records. The only sampling generator is request-derived and checkpoint
independent; global randomness, model modes and requires-grad flags are restored.
Torch is imported only when evaluation is requested.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import math
import random
import time

_spec = importlib.util.spec_from_file_location('_kl_diagnostic_author_loss',
    Path(__file__).resolve().parents[1] / 'dflash_two_update_followup/author_training.py')
author = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(author)

require, digest = author.require, author.digest
BASE_SEED = 17
INPUT_LIMIT, CONTEXT_LIMIT, OUTPUT_CAP = 31744, 32768, 1024


def audit_record(record):
    """Validate a held-out record without using or weakening training guards."""
    require(record.get('role') == 'evaluation', 'KL diagnostic accepts evaluation records only')
    for name in ('group_id', 'request_id'):
        require(isinstance(record.get(name), str) and record[name], 'Missing ' + name)
    ids, output = record.get('input_ids'), record.get('output_ids')
    require(isinstance(ids, list) and ids and isinstance(output, list) and
            all(type(v) is int and v >= 0 for v in ids + output), 'Invalid saved token IDs')
    require(len(ids) <= INPUT_LIMIT and len(output) <= OUTPUT_CAP and
            len(ids) + len(output) + author.BLOCK_SIZE - 1 <= CONTEXT_LIMIT,
            'Full context exceeds the qualified envelope; truncation is forbidden')
    for name, values in (('input_ids_sha256', ids), ('output_ids_sha256', output)):
        require(name not in record or record[name] == digest(values), 'Saved token hash differs: ' + name)
    require('input_token_count' not in record or record['input_token_count'] == len(ids),
            'Saved input boundary differs')
    require('output_cap' not in record or record['output_cap'] == OUTPUT_CAP, 'Saved output cap differs')
    record_id = record.get('record_id', record['request_id'])
    require(isinstance(record_id, str) and record_id, 'Missing record_id')
    mask = [0] * len(ids) + [1] * len(output) + [0] * (author.BLOCK_SIZE - 1)
    return dict(group_id=record['group_id'], request_id=record['request_id'], record_id=record_id,
        role='evaluation', input_ids_sha256=digest(ids), output_ids_sha256=digest(output),
        sequence_sha256=digest(ids + output), loss_mask_sha256=digest(mask),
        input_token_count=len(ids), output_token_count=len(output), response_start=len(ids),
        response_end_exclusive=len(ids) + len(output), real_sequence_length=len(ids) + len(output),
        padded_sequence_length=len(mask), right_padding_positions=author.BLOCK_SIZE - 1,
        response_loss_mask=mask)


def anchor_seed(audit):
    """No checkpoint, timing, response order, or mutable state enters the seed."""
    identity = {key: audit[key] for key in ('group_id', 'request_id', 'record_id',
                'input_ids_sha256', 'output_ids_sha256', 'sequence_sha256', 'loss_mask_sha256')}
    return int(digest(dict(seed=BASE_SEED, purpose='held_out_kl_diagnostic_v1',
                           request=identity))[:16], 16) % (2**63)


def _rng_state():
    import torch
    state = dict(python=random.getstate(), torch=torch.get_rng_state().clone())
    try:
        import numpy as np
        state['numpy'] = np.random.get_state()
    except ImportError:
        pass
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        state['cuda'] = [v.clone() for v in torch.cuda.get_rng_state_all()]
    return state


def _restore_rng(state):
    import torch
    random.setstate(state['python'])
    torch.set_rng_state(state['torch'].cpu())
    if 'numpy' in state:
        import numpy as np
        np.random.set_state(state['numpy'])
    if 'cuda' in state:
        torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda']])


def _state_hash(value):
    import torch
    result = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            result.update(str((str(tensor.dtype), tuple(tensor.shape))).encode())
            result.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item):
                visit(key)
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            result.update(type(item).__name__.encode())
            for child in item:
                visit(child)
        elif hasattr(item, 'dtype') and hasattr(item, 'tobytes'):
            result.update(str((str(item.dtype), tuple(item.shape))).encode())
            result.update(item.tobytes())
        else:
            result.update(repr(item).encode())
        result.update(b'\0')
    visit(value)
    return result.hexdigest()


def _seed_global():
    import torch
    random.seed(BASE_SEED)
    torch.manual_seed(BASE_SEED)
    try:
        import numpy as np
        np.random.seed(BASE_SEED)
    except ImportError:
        pass


def _empty_sampling():
    # The author sampler has no defined sample with fewer than two anchors.
    # No synthetic anchor, denominator, teacher reconstruction or resample.
    result = dict(anchor_positions=[], block_keep_mask=[], label_positions=[], teacher_positions=[],
        binary_loss_mask=[], position_weights=[], sampled_blocks=0, eligible_blocks=0,
        eligible_positions=0, unique_supervised_positions=0, unique_supervised_position_ids=[],
        weighted_position_sum=0.0, normalization_denominator=None, excluded_anchor_positions=0,
        zero_supervision=True, backbone_forward_calls=0, head_projection_chunks=0)
    result['sampling_and_weights_sha256'] = digest(result)
    return result


def _evaluate_one(draft, embed_tokens, lm_head, teacher_features, record, audit, *, device,
                  mask_token_id, attention_backend, projection_chunk):
    import torch
    synchronize = (lambda: torch.cuda.synchronize(device)) if device.type == 'cuda' else lambda: None
    seed = anchor_seed(audit)
    generator = torch.Generator(device=device).manual_seed(seed)
    generator_before = _state_hash(generator.get_state())
    value = dict(audit, anchor_seed=seed, anchor_rng_before_sha256=generator_before)
    if audit['output_token_count'] < 2:
        value.update(_empty_sampling(), status='excluded', loss=None, loss_improvement_eligible=False,
            excluded_from_loss_improvement=True,
            excluded_reason='author_sampler_requires_at_least_two_response_anchors',
            phase_seconds=dict(teacher_preparation_s=0.0, forward_and_loss_s=0.0))
    else:
        synchronize()
        started = time.perf_counter()
        ids = torch.tensor([record['input_ids'] + record['output_ids']], dtype=torch.long, device=device)
        features = teacher_features(ids)
        require(isinstance(features, (tuple, list)) and len(features) == 2,
                'Teacher callback must return two complete causal feature tensors')
        hidden, teacher = features
        for feature in features:
            require(feature.ndim == 3 and feature.shape[:2] == ids.shape and feature.device == device and
                    not feature.requires_grad and bool(torch.isfinite(feature).all()),
                    'Missing/nonfinite/unfrozen complete teacher features')
        synchronize()
        teacher_s = time.perf_counter() - started
        started = time.perf_counter()
        pad = author.BLOCK_SIZE - 1
        ids = torch.cat((ids, ids.new_full((1, pad), mask_token_id)), dim=1)
        hidden = torch.cat((hidden, hidden.new_zeros((1, pad, hidden.shape[-1]))), dim=1)
        teacher = torch.cat((teacher, teacher.new_zeros((1, pad, teacher.shape[-1]))), dim=1)
        mask = torch.tensor([audit['response_loss_mask']], dtype=torch.float32, device=device)
        loss, sampling = author.author_response_loss(draft, embed_tokens, lm_head, ids, hidden, teacher,
            mask, mask_token_id=mask_token_id, generator=generator, attention_backend=attention_backend,
            projection_chunk=projection_chunk, checkpoint_projection=False)
        synchronize()
        loss_value = float(loss)
        require(math.isfinite(loss_value), 'Nonfinite held-out KL loss')
        excluded = sampling['zero_supervision']
        value.update(sampling, status='excluded' if excluded else 'passed', loss=loss_value,
            loss_improvement_eligible=not excluded, excluded_from_loss_improvement=excluded,
            excluded_reason='sample_has_zero_supervision' if excluded else None,
            phase_seconds=dict(teacher_preparation_s=teacher_s, forward_and_loss_s=time.perf_counter()-started))
    value['anchor_rng_after_sha256'] = _state_hash(generator.get_state())
    value['anchor_sampling_attempts'] = int(audit['output_token_count'] >= 2)
    return value


def evaluate_responses(draft, embed_tokens, lm_head, teacher_features, records, *, checkpoint_id,
                       checkpoint_identity_sha256, mask_token_id=128002,
                       attention_backend='flex_attention', projection_chunk=64):
    """Evaluate held-out responses, preserving state even if evaluation raises.

    A successful result may contain explicitly excluded zero-supervision rows.
    Those rows must not enter a loss-improvement claim. Numerical/identity errors
    raise and are never converted into valid measurements. On failure the raised
    exception carries ``diagnostic_partial`` for durable caller-side recording.
    """
    import torch
    require(checkpoint_id in ('F', 'S8'), 'Only F and S8 are in this diagnostic')
    require(author._sha256(checkpoint_identity_sha256), 'Invalid checkpoint identity')
    require(type(mask_token_id) is int and mask_token_id >= 0, 'Invalid mask token')
    require(type(projection_chunk) is int and projection_chunk > 0, 'Invalid projection chunk')
    require(attention_backend in ('sdpa', 'flex_attention'), 'Unsupported attention backend')
    records = list(records)
    require(bool(records), 'No held-out evaluation records supplied')
    audits = [audit_record(record) for record in records]
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
    try:
        for root in roots:
            root.eval()
        for parameter in parameters:
            parameter.requires_grad_(False)
        with torch.no_grad():
            for record, audit in zip(records, audits):
                row_before = _rng_state()
                try:
                    _seed_global()
                    value = _evaluate_one(draft, embed_tokens, lm_head, teacher_features, record, audit,
                        device=device, mask_token_id=mask_token_id, attention_backend=attention_backend,
                        projection_chunk=projection_chunk)
                finally:
                    _restore_rng(row_before)
                value.update(checkpoint_id=checkpoint_id, checkpoint_identity_sha256=checkpoint_identity_sha256,
                    diagnostic_model_mode='eval_no_grad', training_rng_preserved=True,
                    global_rng_before_sha256=_state_hash(row_before),
                    global_rng_after_sha256=_state_hash(_rng_state()))
                require(value['global_rng_before_sha256'] == value['global_rng_after_sha256'],
                        'Evaluation perturbed global RNG')
                values.append(value)
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
                checkpoint_identity_sha256=checkpoint_identity_sha256, response_losses=values,
                error_type=type(failure).__name__, error=str(failure),
                global_rng_restored=_state_hash(global_before) == _state_hash(_rng_state()),
                optimizer_steps=0, diagnostic_wall_s=time.perf_counter()-started)
    require(_state_hash(global_before) == _state_hash(_rng_state()), 'Global RNG restoration failed')
    eligible = [v['loss'] for v in values if v['loss_improvement_eligible']]
    return dict(status='passed', checkpoint_id=checkpoint_id,
        checkpoint_identity_sha256=checkpoint_identity_sha256, response_losses=values,
        eligible_response_count=len(eligible), excluded_response_count=len(values)-len(eligible),
        loss_mean=sum(eligible)/len(eligible) if eligible else None,
        reduction='equal mean of individually normalized nonzero-supervision response losses',
        optimizer_steps=0, optimizer_created=False, evaluation_mode=True, gradients_enabled=False,
        global_rng_before_sha256=_state_hash(global_before), global_rng_after_sha256=_state_hash(_rng_state()),
        model_modes_restored=True, parameter_grad_flags_restored=True,
        diagnostic_wall_s=time.perf_counter()-started)
