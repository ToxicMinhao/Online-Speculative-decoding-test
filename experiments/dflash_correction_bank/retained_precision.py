"""Exact-anchor, read-only precision replay for an already learned correction.

This never samples, writes bank state, runs an optimizer, or generates a response.
The controller owns the durable once-per-case reconstruction limit. Features are
reconstructed once from the retained BANK response, then compared with its saved
compressed objective before any native/reference precision result is accepted.
"""
from __future__ import annotations

import time

import numpy as np

from .feedback import FrozenResponseObjective, author


# Replay is expected to reproduce the same frozen objective. These tolerances
# admit floating-point reduction noise, not replacement anchors or supervision.
STAT_ATOL = 1e-9
STAT_RTOL = 1e-7
WEIGHT_ATOL = 1e-8
WEIGHT_RTOL = 2e-7


def retained_sample_plan(feedback):
    """Validate saved anchors, masks and exact normalized weights without Torch.

    No RNG restoration/resampling is needed: full anchor and position metadata
    were retained by the original BANK feature collector. RNG arrays are kept as
    provenance only. Any absent/inconsistent record fails before teacher work.
    """
    response = feedback['response']
    summary = feedback['summary']
    metadata = summary['metadata']
    audit = author.audit_response(response)
    require = author.require
    require(summary['response_id'] == response['record_id'], 'Retained feedback response identity differs')
    require(metadata.get('sampling_and_weights_sha256') == author.digest(
        {key: value for key, value in metadata.items() if key != 'sampling_and_weights_sha256'}),
        'Retained sampling/weight metadata hash differs')
    for key, expected in audit.items():
        require(metadata.get(key) == expected, 'Retained response audit differs: ' + key)
    count, block = audit['sampled_anchor_count'], author.BLOCK_SIZE
    anchors = np.asarray(metadata['anchor_positions'])
    keep = np.asarray(metadata['block_keep_mask'])
    require(anchors.shape == (count,) and anchors.dtype.kind in 'iu', 'Invalid retained anchor positions')
    require(keep.shape == (count,) and keep.dtype.kind == 'b', 'Invalid retained anchor keep mask')
    # This pilot has one response per sample; each retained sampled anchor is a
    # valid current-response position. The original sampler sorts its anchors.
    require(bool(keep.all()) and bool(np.all(np.diff(anchors) > 0)), 'Invalid retained anchor order/keep mask')
    begin, end, length = audit['response_start'], audit['response_end_exclusive'], audit['padded_sequence_length']
    require(bool(np.all((anchors >= begin) & (anchors < end))), 'Retained anchor is outside newest response')
    offsets = np.arange(block)
    labels = anchors[:, None] + offsets[None, :]
    safe = np.minimum(labels, length - 1)
    teacher_positions = np.maximum(safe - 1, 0)
    binary = (keep[:, None] & (labels < length) & (offsets[None, :] > 0) &
              (safe >= begin) & (safe < end)).astype(np.float32)
    for key, expected in (('label_positions', labels), ('teacher_positions', teacher_positions),
                          ('binary_loss_mask', binary)):
        saved = np.asarray(metadata[key])
        require(saved.shape == expected.shape and np.array_equal(saved, expected),
                'Retained position/mask differs: ' + key)
    weights = np.asarray(metadata['position_weights'], dtype=np.float64)
    decay = np.exp(-np.maximum(offsets - 1, 0).astype(np.float32) / author.RECIPE['loss_decay_gamma'])
    expected_weights = binary * decay[None, :]
    require(weights.shape == (count, block) and np.isfinite(weights).all() and
            np.allclose(weights, expected_weights, atol=WEIGHT_ATOL, rtol=WEIGHT_RTOL),
            'Retained author position weights differ')
    require(np.array_equal(weights > 0, binary > 0), 'Retained positive-weight mask differs')
    selected = np.flatnonzero(weights.reshape(-1) > 0)
    raw_weights = weights.reshape(-1)[selected]
    omega = np.asarray(summary['omega'], dtype=np.float64)
    expected_omega = raw_weights / raw_weights.sum() if raw_weights.size else raw_weights
    require(omega.shape == expected_omega.shape and np.array_equal(omega, expected_omega),
            'Retained normalized supervision weights differ')
    supervised = labels.reshape(-1)[selected].tolist()
    require(metadata['eligible_positions'] == len(supervised) and
            metadata['unique_supervised_position_ids'] == sorted(set(supervised)) and
            metadata['unique_supervised_positions'] == len(set(supervised)), 'Retained exposure differs')
    require(metadata['weighted_position_sum'] == float(raw_weights.sum()), 'Retained position-weight sum differs')
    require(metadata['zero_supervision'] == (not supervised), 'Retained empty-supervision flag differs')
    require(metadata['eligible_blocks'] == int((binary.sum(-1) > 0).sum()), 'Retained eligible-block count differs')
    for key in ('rng_before', 'rng_after'):
        value = np.asarray(feedback[key])
        require(value.ndim == 1 and value.size > 0 and value.dtype == np.uint8, 'Missing original RNG evidence: ' + key)
    return dict(response=response, metadata=metadata, audit=audit, anchors=anchors.copy(), keep=keep.copy(),
                labels=labels, teacher_positions=teacher_positions, selected=selected,
                omega=omega.copy(), weights=weights, binary=binary,
                method='explicit_original_anchor_and_weight_replay_no_sampler_no_rng_restoration',
                sampling_and_weights_sha256=metadata['sampling_and_weights_sha256'])


def reconstruct_retained_features(draft, embed_tokens, teacher_features, feedback, *,
                                 synchronize=lambda: None, mask_token_id=128002,
                                 attention_backend='flex_attention'):
    """One full teacher replay and one frozen draft forward, using saved anchors."""
    import torch

    plan = retained_sample_plan(feedback)
    for module in (draft, embed_tokens):
        if module.training or any(p.requires_grad or p.grad is not None for p in module.parameters()):
            raise ValueError('Retained reconstruction requires frozen eval modules')
    response = plan['response']
    device = next(draft.parameters()).device
    synchronize(); started = time.perf_counter()
    ids = torch.tensor([response['input_ids'] + response['output_ids']], device=device, dtype=torch.long)
    with torch.no_grad():
        hidden, teacher = teacher_features(ids)
    if hidden.shape[:2] != ids.shape or teacher.shape[:2] != ids.shape:
        raise ValueError('Retained reconstruction requires full exact causal teacher history')
    synchronize(); teacher_s = time.perf_counter() - started; started = time.perf_counter()
    pad = author.BLOCK_SIZE - 1
    ids = torch.cat((ids, ids.new_full((1, pad), mask_token_id)), dim=1)
    hidden = torch.cat((hidden, hidden.new_zeros((1, pad, hidden.shape[-1]))), dim=1)
    teacher = torch.cat((teacher, teacher.new_zeros((1, pad, teacher.shape[-1]))), dim=1)
    anchors = torch.tensor(plan['anchors'][None, :], device=device, dtype=torch.long)
    keep = torch.tensor(plan['keep'][None, :], device=device, dtype=torch.bool)
    labels = torch.tensor(plan['labels'][None, :, :], device=device, dtype=torch.long)
    count = len(plan['anchors'])
    noise = torch.full((1, count, author.BLOCK_SIZE), mask_token_id, device=device, dtype=torch.long)
    noise[:, :, 0] = torch.where(keep, ids.gather(1, anchors), mask_token_id)
    positions = torch.cat((torch.arange(ids.shape[1], device=device).unsqueeze(0), labels.reshape(1, -1)), dim=1)
    mask = author.create_attention_mask(anchors, keep, ids.shape[1], backend=attention_backend)
    with torch.no_grad():
        output = draft(position_ids=positions, noise_embedding=embed_tokens(noise.reshape(1, -1)),
                       target_hidden=hidden, attention_mask=mask)
    if isinstance(output, tuple):
        if len(output) != 2 or output[1] is not None:
            raise ValueError('Adaptive head is outside retained bank replay')
        output = output[0]
    if output.shape[:2] != (1, count * author.BLOCK_SIZE):
        raise ValueError('Retained reconstruction draft feature shape differs')
    selected = torch.tensor(plan['selected'], device=device, dtype=torch.long)
    teacher_positions = torch.tensor(plan['teacher_positions'].reshape(-1)[plan['selected']],
                                     device=device, dtype=torch.long)
    h = output.reshape(-1, output.shape[-1])[selected].detach().cpu()
    teacher_h = teacher[0, teacher_positions].detach().cpu()
    synchronize()
    return dict(h=h, teacher_h=teacher_h, omega=plan['omega'], metadata=plan['metadata'],
                audit={**plan['audit'], 'sampling_and_weights_sha256':plan['sampling_and_weights_sha256'],
                       'reconstruction_method':plan['method'], 'sampler_invocations':0,
                       'teacher_reconstructions':1, 'draft_feature_forwards':1,
                       'source_answer_regenerated':False, 'original_normalized_weights_exact':True},
                costs={'teacher_feature_reconstruction_s':teacher_s,
                       'frozen_draft_feature_reconstruction_s':time.perf_counter()-started})


def compare_retained_objective(objective, feedback):
    """Check the full stored objective statistics, not merely its fitted loss."""
    rows = {}
    for key in ('omega', 'phi', 'G', 'S', 'T', 'M'):
        expected = np.asarray(feedback['M'] if key == 'M' else feedback['summary'][key])
        actual = np.asarray(getattr(objective, key))
        shape_matches = actual.shape == expected.shape
        finite = bool(np.isfinite(actual).all() and np.isfinite(expected).all())
        delta = np.abs(actual - expected) if shape_matches else None
        passed = bool(shape_matches and finite and (np.array_equal(actual, expected) if key == 'omega' else
                      np.allclose(actual, expected, atol=STAT_ATOL, rtol=STAT_RTOL)))
        rows[key] = dict(passed=passed, shape_matches=shape_matches, finite=finite,
                         maximum_absolute_difference=(float(delta.max()) if delta.size else 0.) if delta is not None and finite else None,
                         maximum_retained_magnitude=(float(np.abs(expected).max()) if expected.size else 0.) if np.isfinite(expected).all() else None,
                         exact=bool(shape_matches and np.array_equal(actual, expected)))
    return dict(passed=all(row['passed'] for row in rows.values()), statistics=rows,
                absolute_tolerance=STAT_ATOL, relative_tolerance=STAT_RTOL,
                normalized_weights_require_exact_equality=True,
                scope='Original sampled frozen-reference compressed objective; original native hidden tensors were not retained')


def reconstruct_retained_precision(draft, embed_tokens, teacher_features, lm_head_weight,
                                   projections, feedback, selected_matrix, *, synchronize=lambda: None,
                                   mask_token_id=128002, attention_backend='flex_attention'):
    """Return available precision evidence or explicit evidence mismatch, no retry.

    Invalid retained metadata raises before reconstruction. A reconstructed
    objective mismatch returns unavailable; the caller must not resample or
    repeat feature extraction in response. This diagnostic is independent of
    native drafting effectiveness, which requires the separate ten generations.
    """
    matrix = np.array(selected_matrix, dtype=np.float64, copy=True)
    rank = projections.U.shape[1]
    if matrix.shape != (rank, projections.V_in.shape[1]) or not np.isfinite(matrix).all():
        raise ValueError('Invalid retained singleton matrix')
    features = reconstruct_retained_features(draft, embed_tokens, teacher_features, feedback,
        synchronize=synchronize, mask_token_id=mask_token_id, attention_backend=attention_backend)
    objective = FrozenResponseObjective(features['h'], features['teacher_h'], features['omega'], lm_head_weight,
        projections.U, projections.V_in, projection_chunk=32, synchronize=synchronize)
    comparison = compare_retained_objective(objective, feedback)
    result = dict(status='measured' if comparison['passed'] else 'unavailable',
                  reconstruction=features['audit'], retained_objective_check=comparison,
                  costs={**features['costs'], 'objective_reconstruction_s':objective.compression_s},
                  precision=None,
                  interpretation='BF16 correction survival does not establish improved acceptance or fewer verification calls.')
    if not comparison['passed']:
        result['reason'] = 'Reconstructed frozen objective does not match retained source feedback; no precision inference or resampling.'
        return result
    from .precision import qualify_native_precision
    result['precision'] = qualify_native_precision(objective, {'zero':np.zeros_like(matrix), 'selected':matrix})
    result['costs']['native_precision_s'] = result['precision']['wall_s']
    return result
