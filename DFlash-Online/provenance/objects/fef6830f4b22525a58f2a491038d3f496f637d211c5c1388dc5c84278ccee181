"""Replay a saved native target path to collect causally aligned features."""
from .storage import write_json


def replay_frozen_target(target, layer_ids, record, device, failure_path):
    """Return features, final hidden states, labels and compact replay evidence.

    The saved draft proposals determine each target forward's input and shape.
    Every posterior row must match, including rows discarded by verification.
    Only canonical input rows are retained as feedback features.
    """
    import torch
    from dflash.model import _make_cache, _crop_to, extract_context_feature

    prompt, response = record['prompt_ids'], record['response_ids']
    prompt_length = len(prompt)
    diagnostics = {
        'label_binding_scope': 'exact_saved_native_prefill_and_verification_blocks',
        'prefill_forward_count': 0,
        'verification_forward_count': 0,
        'final_feature_forward_count': 0,
        'full_posterior_rows_checked': 0,
        'canonical_feature_rows_before_final': 0,
    }

    def fail(reason, **details):
        write_json(failure_path, {
            'record_id': record['record_id'], 'reason': reason,
            'prompt_ids': prompt, 'response_ids': response,
            'diagnostics': diagnostics, **details,
        })
        raise RuntimeError(f'native target replay failed: {reason}; evidence saved')

    def ids(tokens):
        return torch.tensor([tokens], dtype=torch.long, device=device)

    def positions(start, length):
        return torch.arange(start, start + length, device=device).unsqueeze(0)

    if not prompt or not response:
        fail('empty_prompt_or_response')
    trace = record.get('metadata', {}).get('trace')
    if not isinstance(trace, list):
        fail('missing_native_verification_trace')
    cache = _make_cache(target.config)
    output = target(
        ids(prompt), position_ids=positions(0, prompt_length),
        past_key_values=cache, use_cache=True, logits_to_keep=1,
        output_hidden_states=True,
    )
    diagnostics['prefill_forward_count'] += 1
    first = output.logits.argmax(-1)[0].tolist()
    if first != response[:1]:
        fail('prefill_argmax_mismatch', expected=response[:1], actual=first)
    teacher_ids = first
    feature_parts = [extract_context_feature(output.hidden_states, layer_ids)]
    # Pinned Transformers 5.12.1 exposes the normalized last hidden state here.
    hidden_parts = [output.hidden_states[-1]]
    _crop_to(cache, prompt_length)
    del output

    offset = 0
    for index, block in enumerate(trace):
        required = {'verify_size', 'committed_progress', 'anchor_token', 'draft_tokens',
                    'target_argmax', 'output_start', 'committed_tokens'}
        if not isinstance(block, dict) or not required.issubset(block):
            fail('incomplete_verification_block', block_index=index, block=block)
        size, progress = block['verify_size'], block['committed_progress']
        tokens = [block['anchor_token']] + block['draft_tokens']
        saved_posterior = block['target_argmax']
        if (type(size) is not int or type(progress) is not int
                or not 0 < progress <= size
                or len(tokens) != size or len(saved_posterior) != size
                or block['output_start'] != offset
                or offset + progress >= len(response)
                or tokens[0] != response[offset]):
            fail('invalid_block_geometry', block_index=index, block=block)
        canonical_inputs = response[offset:offset + progress]
        canonical_labels = response[offset + 1:offset + progress + 1]
        if (tokens[:progress] != canonical_inputs
                or block['committed_tokens'] != canonical_labels
                or saved_posterior[:progress] != canonical_labels):
            fail('noncanonical_committed_prefix', block_index=index, block=block)

        output = target(
            ids(tokens), position_ids=positions(prompt_length + offset, size),
            past_key_values=cache, use_cache=True, output_hidden_states=True,
        )
        diagnostics['verification_forward_count'] += 1
        posterior = output.logits.argmax(-1)[0].tolist()
        if posterior != saved_posterior:
            fail('full_verification_posterior_mismatch', block_index=index,
                 expected=saved_posterior, actual=posterior, block=block)
        diagnostics['full_posterior_rows_checked'] += size
        teacher_ids.extend(posterior[:progress])
        feature_parts.append(extract_context_feature(output.hidden_states, layer_ids)[:, :progress])
        hidden_parts.append(output.hidden_states[-1][:, :progress])
        offset += progress
        _crop_to(cache, prompt_length + offset)
        del output

    rows = sum(part.shape[1] for part in feature_parts)
    diagnostics['canonical_feature_rows_before_final'] = rows
    if offset != len(response) - 1 or teacher_ids != response or rows != prompt_length + len(response) - 1:
        fail('incomplete_canonical_replay', replayed_response_positions=offset,
             target_argmax_ids=teacher_ids, feature_rows=rows)

    # Verification leaves the last committed token as the next unprocessed
    # anchor. Its feature is needed by the unchanged full-length training payload.
    output = target(
        ids(response[-1:]), position_ids=positions(prompt_length + offset, 1),
        past_key_values=cache, use_cache=True, logits_to_keep=1,
        output_hidden_states=True,
    )
    diagnostics['final_feature_forward_count'] += 1
    feature_parts.append(extract_context_feature(output.hidden_states, layer_ids))
    hidden_parts.append(output.hidden_states[-1])
    features, last_hidden = torch.cat(feature_parts, dim=1), torch.cat(hidden_parts, dim=1)
    if (features.shape[1] != prompt_length + len(response)
            or last_hidden.shape[1] != features.shape[1]
            or not torch.isfinite(features).all() or not torch.isfinite(last_hidden).all()):
        fail('invalid_collected_features', feature_shape=list(features.shape),
             last_hidden_shape=list(last_hidden.shape))
    diagnostics['target_forward_count'] = sum(diagnostics[key] for key in (
        'prefill_forward_count', 'verification_forward_count', 'final_feature_forward_count'))
    diagnostics['full_posterior_match'] = True
    diagnostics['final_feature_logits_used_for_labels'] = False
    return features, last_hidden, teacher_ids, diagnostics
