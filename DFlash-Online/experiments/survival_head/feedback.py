"""Exact saved-path feedback reconstruction; never generate replacement answers.

The runtime adapter repeats the original SGLang prefill and rectangular target
verification inputs. The ordinary final-row teacher operation remains separate.
CPU validation is necessary but does not establish native numerical parity: the
two retained packets must pass exact tensor comparison before pool admission.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

from online_feedback.native import validate_generation
from online_feedback.storage import write_json


BLOCK_FIELDS = (
    'output_start', 'anchor_token', 'draft_tokens', 'target_argmax', 'verify_size',
    'accepted_draft_tokens', 'native_accepted_draft_tokens',
    'raw_accepted_draft_tokens_before_stop', 'committed_progress',
    'committed_tokens', 'positions', 'stop_clipped', 'finished',
)


def _require(value, message):
    if not value:
        raise ValueError(message)


def _tokens(value, name):
    _require(isinstance(value, list) and bool(value)
             and all(type(token) is int and token >= 0 for token in value),
             'Missing exact token list: ' + name)
    return value


def validate_saved_record(record):
    """Reject incomplete, inconsistent, or differently shaped source evidence."""
    prompt = _tokens(record['prompt_ids'], 'prompt_ids')
    response = _tokens(record['response_ids'], 'response_ids')
    stops = _tokens(record['stop_token_ids'], 'stop_token_ids')
    metadata, capture = record['metadata'], record['metadata']['capture']
    blocks = metadata['trace']
    _require(isinstance(record['record_id'], str) and bool(record['record_id'])
             and record['purpose'] == 'serving', 'Source must be identified serving feedback')
    _require(type(record['max_new_tokens']) is int and record['max_new_tokens'] > 0,
             'Invalid original output cap')
    _require(isinstance(blocks, list) and metadata['verification_calls'] == len(blocks),
             'Saved verification count differs')
    _require(metadata['runtime'] == 'pinned_SGLang_FA3'
             and metadata['block_size'] == 16
             and metadata['cache_policy'] == 'idle_full_flush_before_every_request_and_publication',
             'Saved native runtime/cache contract differs')
    _require(capture['status'] == 'passed'
             and capture['native_capture_validated'] is True
             and capture['collect_features'] is True
             and capture['label_binding_scope'] == 'native_prefill_and_committed_verification'
             and capture['final_feature_logits_used_for_labels'] is False
             and capture['final_emitted_token_feature_present'] is False,
             'Original native capture evidence is incomplete')
    # The old compact capture stores the number, not the individual boundaries.
    # Refuse multi-chunk inputs instead of inventing their unretained shapes.
    _require(type(capture['prefill_chunks']) is int and capture['prefill_chunks'] == 1
             and 0 < len(prompt) <= 8192, 'Exact prefill chunk geometry is unavailable')
    _require(capture['block_size'] == 16 and capture['prompt_ids'] == prompt
             and capture['response_ids'] == capture['target_argmax_ids'] == response
             and capture['covered_token_ids'] == (prompt + response)[:-1]
             and capture['covered_rows'] == len(prompt) + len(response) - 1
             and capture['blocks'] == blocks, 'Saved capture and record disagree')
    _require(len(response) >= 2, 'No eligible two-token feedback response')
    _, finish = validate_generation(prompt, prompt + response, blocks,
        max_new_tokens=record['max_new_tokens'], stop_tokens=stops)
    _require(finish == record['finish_reason'], 'Saved termination differs')
    offset = 0
    for index, block in enumerate(blocks):
        _require(isinstance(block, dict) and set(BLOCK_FIELDS).issubset(block),
                 'Incomplete native block evidence')
        candidates = [block['anchor_token']] + _tokens(block['draft_tokens'], 'draft_tokens')
        posterior = _tokens(block['target_argmax'], 'target_argmax')
        progress = block['committed_progress']
        _require(type(progress) is int and 1 <= progress <= 16
                 and type(block['verify_size']) is int and block['verify_size'] == 16
                 and len(candidates) == len(posterior) == 16
                 and block['output_start'] == offset
                 and block['positions'] == list(range(len(prompt) + offset, len(prompt) + offset + 16)),
                 'Saved native block geometry differs')
        accepted_end = next((slot for slot in range(1, 16)
                             if candidates[slot] != posterior[slot - 1]), 16)
        committed = response[offset + 1:offset + progress + 1]
        _require(progress <= accepted_end and len(committed) == progress
                 and candidates[:progress] == response[offset:offset + progress]
                 and committed == posterior[:progress] == block['committed_tokens']
                 and committed == (candidates[1:accepted_end] + [posterior[accepted_end - 1]])[:progress],
                 'Saved proposal/posterior does not explain canonical commits')
        _require(block['raw_accepted_draft_tokens_before_stop'] == accepted_end - 1
                 and block['accepted_draft_tokens'] == min(accepted_end - 1, progress)
                 and block['native_accepted_draft_tokens'] == progress - 1
                 and type(block['stop_clipped']) is bool
                 and block['stop_clipped'] == (progress < accepted_end)
                 and type(block['finished']) is bool
                 and block['finished'] == (index == len(blocks) - 1),
                 'Saved acceptance or terminal clipping differs')
        _require(not block['stop_clipped'] or block['finished'],
                 'A nonterminal native block cannot be stop-clipped')
        offset += progress
    _require(offset == len(response) - 1, 'Saved trajectory is incomplete')
    return prompt, response, blocks


def source_contract(record):
    prompt, response, blocks = validate_saved_record(record)
    return dict(record_id=record['record_id'], checkpoint_id=record['checkpoint_id'],
        prompt_ids=prompt, response_ids=response, finish_reason=record['finish_reason'],
        max_new_tokens=record['max_new_tokens'], stop_token_ids=record['stop_token_ids'],
        prefill_chunks=1, runtime=record['metadata']['runtime'],
        blocks=[{key: row[key] for key in BLOCK_FIELDS} for row in blocks])


def source_contract_sha256(record):
    payload = json.dumps(source_contract(record), sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def expected_reconstruction_work(record):
    prompt, response, blocks = validate_saved_record(record)
    return dict(new_generations=0, draft_forwards=0, prefill_forwards=1,
        verification_forwards=len(blocks), terminal_forwards=1,
        target_forwards=2 + len(blocks),
        prefill_input_rows=len(prompt), verification_input_rows=16 * len(blocks),
        terminal_input_rows=len(prompt) + len(response),
        retained_native_rows=len(prompt) + len(response) - 1, retained_terminal_rows=1)


def compare_prepared_packets(expected_path, actual_path):
    """Bitwise full-packet qualification, including the separate terminal row."""
    import torch
    from experiments.ok_coding.storage import identity
    from online_feedback.training import validate_target_binding

    expected = torch.load(expected_path, map_location='cpu', weights_only=False)
    actual = torch.load(actual_path, map_location='cpu', weights_only=False)
    for packet in (expected, actual):
        validate_target_binding(packet)
        _require(packet['hidden_states'].dtype == packet['target_last_hidden_states'].dtype == torch.bfloat16,
                 'Retained native teacher features must remain BF16')
    for key in ('record_id', 'prompt_length'):
        _require(expected[key] == actual[key], 'Retained packet identity differs: ' + key)
    for key in ('target_layer_ids', 'target_argmax_ids', 'input_ids',
                'label_binding_scope', 'final_feature_logits_used_for_labels'):
        _require(expected['label_binding'][key] == actual['label_binding'][key],
                 'Retained label binding differs: ' + key)
    fields = {}
    for key in ('input_ids', 'loss_mask', 'hidden_states', 'target_last_hidden_states'):
        left, right = expected[key], actual[key]
        same_layout = left.shape == right.shape and left.dtype == right.dtype
        finite = bool(torch.isfinite(left).all() and torch.isfinite(right).all())
        equal = same_layout and torch.equal(left.contiguous().view(torch.uint8),
                                            right.contiguous().view(torch.uint8))
        fields[key] = dict(shape=list(right.shape), dtype=str(right.dtype),
                           exact=bool(equal), finite=finite)
        if key in ('hidden_states', 'target_last_hidden_states') and same_layout:
            fields[key].update(
                native_rows_exact=bool(torch.equal(left[:, :-1].contiguous().view(torch.uint8),
                                                   right[:, :-1].contiguous().view(torch.uint8))),
                terminal_row_exact=bool(torch.equal(left[:, -1:].contiguous().view(torch.uint8),
                                                    right[:, -1:].contiguous().view(torch.uint8))))
    passed = all(row['exact'] and row['finite'] for row in fields.values())
    return dict(status='passed' if passed else 'failed', record_id=actual['record_id'],
        expected_identity=identity(expected_path), actual_identity=identity(actual_path),
        fields=fields, full_packet_bitwise_equal=passed,
        historical_kv_byte_equality='unavailable_original_KV_not_retained')


def require_reconstruction_qualification(receipts, expected_record_ids):
    """Exactly the two retained source packets qualify the later missing pool."""
    _require(len(receipts) == len(expected_record_ids) == len(set(expected_record_ids)) == 2,
             'Require exactly two distinct retained qualification packets')
    _require([r['record_id'] for r in receipts] == list(expected_record_ids),
             'Reconstruction qualification request IDs differ')
    for receipt in receipts:
        _require(receipt['status'] == 'passed'
                 and receipt['packet_comparison']['status'] == 'passed'
                 and receipt['packet_comparison']['full_packet_bitwise_equal'] is True
                 and receipt['replay']['status'] == 'passed'
                 and receipt['replay']['full_posterior_match'] is True
                 and receipt['replay']['all_commits_exact'] is True,
                 'Retained packet reconstruction has not passed exactly')
    return dict(status='passed', record_ids=list(expected_record_ids),
                exact_packet_comparisons=2, automatic_retry=False)


def reconstruct_feedback(backend, record, feature_path, *, retained_packet=None):
    """Prepare one pilot-owned packet through the additive isolated worker.

    Pool callers must first require_reconstruction_qualification on the two
    preserved packets. This function itself also serves that qualification.
    No ordinary generate() operation exists in the reconstruction worker.
    """
    from online_feedback.rng import preserve_rng

    validate_saved_record(record)
    path = Path(feature_path).absolute()
    receipt_path = path.with_suffix('.reconstruction.json')
    replay_path = path.with_suffix('.replay.json')
    _require(all(not p.exists() for p in (path, path.with_suffix('.json'), receipt_path, replay_path)),
             'Reconstruction outputs are immutable')
    receipt = dict(status='running', record_id=record['record_id'],
        source_contract_sha256=source_contract_sha256(record),
        expected_work=expected_reconstruction_work(record), automatic_retry=False,
        packet_comparison=None, terminal_attempts=0, terminal_forwards=0)
    started = time.perf_counter()
    write_json(receipt_path, receipt)
    try:
        with preserve_rng():
            replay = backend._call('reconstruct', record=deepcopy(record), outputfile=str(replay_path))
            receipt['replay'] = replay
            _require(replay['status'] == 'passed'
                     and replay['source_contract_sha256'] == receipt['source_contract_sha256']
                     and replay['full_posterior_match'] is True
                     and replay['all_commits_exact'] is True,
                     'Native saved-path replay did not pass')
            receipt['terminal_attempts'] += 1
            write_json(receipt_path, receipt)
            prepared = backend.prepare_feedback(replay['record'], path)
            _require(prepared['status'] == 'prepared'
                     and prepared['terminal_feature']['target_forward_count'] == 1,
                     'Ordinary terminal-row preparation did not pass')
            receipt['terminal_forwards'] += 1
            receipt['preparation'] = prepared
        if retained_packet is not None:
            comparison = compare_prepared_packets(retained_packet, path)
            receipt['packet_comparison'] = comparison
            _require(comparison['status'] == 'passed', 'Retained packet differs bitwise; no pool admission')
        receipt.update(status='passed', feature_path=str(path),
            native_tensor_parity_qualified=retained_packet is not None,
            target_forwards=replay['prefill_forwards'] + replay['verification_forwards'] + 1)
        return receipt
    except BaseException as error:
        receipt.update(status='failed', error=repr(error))
        if replay_path.exists():
            receipt['replay'] = json.loads(replay_path.read_text())
        raise
    finally:
        terminal_work = path.with_suffix('.terminal.work.json')
        if terminal_work.exists():
            observed = json.loads(terminal_work.read_text())
            receipt['terminal_work'] = observed
            receipt['terminal_forwards'] = observed['target_forward_completions']
            # The attempt field above includes entry into prepare_feedback;
            # this field records the actual target-call admission separately.
            receipt['terminal_target_attempts'] = observed['target_forward_attempts']
        receipt['wall_s'] = time.perf_counter() - started
        write_json(receipt_path, receipt)
