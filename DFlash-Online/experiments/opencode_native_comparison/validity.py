"""Token/termination checks shared by the CODE evaluator, with independent AR checks."""
from __future__ import annotations

import math


def validate_ar_output(prompt_ids, output_ids, meta_info, *, max_new_tokens,
                       stop_token_ids, context_limit=4096):
    """Validate native AR output without constructing speculative trace blocks.

    The pinned scheduler checks the length cap before EOS. An EOS at exactly
    the cap may therefore carry a native ``length`` reason; retain that raw
    reason while using the established EOS-first semantic reporting convention.
    """
    if (not prompt_ids or max_new_tokens < 1
            or len(prompt_ids) + max_new_tokens > context_limit):
        raise ValueError('AR request exceeds the logical context budget')
    if (not isinstance(output_ids, list) or not output_ids
            or any(type(token) is not int or token < 0 for token in output_ids)
            or len(output_ids) > max_new_tokens):
        raise ValueError('Invalid AR output tokens or length')
    if any(token in stop_token_ids for token in output_ids[:-1]):
        raise ValueError('AR continued after an EOS token')
    eos = output_ids[-1] in stop_token_ids
    if not eos and len(output_ids) != max_new_tokens:
        raise ValueError('AR ended before EOS or the requested cap')
    if meta_info['prompt_tokens'] != len(prompt_ids):
        raise ValueError('AR native prompt count differs from the saved input')
    if meta_info['completion_tokens'] != len(output_ids):
        raise ValueError('AR native completion count differs from returned tokens')
    if meta_info['cached_tokens'] != 0:
        raise ValueError('AR reused cached prompt tokens after a cold-cache flush')
    reason = meta_info['finish_reason']
    if reason['type'] == 'length':
        if len(output_ids) != max_new_tokens or reason['length'] != max_new_tokens:
            raise ValueError('AR native length termination contradicts the cap')
    elif reason['type'] == 'stop':
        matched = reason['matched']
        matched = matched if isinstance(matched, list) else [matched]
        if not eos or output_ids[-1] not in matched:
            raise ValueError('AR native stop termination contradicts its tokens')
    else:
        raise ValueError('Unexpected AR native termination: ' + str(reason))
    return output_ids, 'eos' if eos else 'length'


def validate_canonical(record, canonical, *, expected_record_id=None):
    """Bind the execution slot separately from its shared canonical token record."""
    if expected_record_id is not None and record['record_id'] != expected_record_id:
        raise ValueError('Canonical comparison has the wrong execution record_id')
    for key in ('prompt_ids', 'response_ids', 'finish_reason',
                'max_new_tokens', 'stop_token_ids'):
        if record[key] != canonical[key]:
            raise ValueError('Canonical mismatch: ' + key)


def canonical_for_method(bundle, method):
    """AR has its own reference; every speculative endpoint uses frozen DFlash."""
    if set(bundle) != {'AR', 'F'}:
        raise ValueError('Require separate AR and frozen DFlash canonical records')
    return bundle['AR' if method == 'AR' else 'F']


def output_comparison(record, reference):
    """Describe cross-path outputs without relaxing the shared input contract.

    Token and termination differences are observations, not within-path validity
    failures. Call ``validate_canonical`` for repetitions and speculative arms.
    """
    for key in ('prompt_ids', 'max_new_tokens', 'stop_token_ids'):
        if record[key] != reference[key]:
            raise ValueError('Output comparison input mismatch: ' + key)
    left, right = record['response_ids'], reference['response_ids']
    first = next((i for i, (a, b) in enumerate(zip(left, right)) if a != b), None)
    if first is None and len(left) != len(right):
        first = min(len(left), len(right))
    same_tokens = left == right
    same_finish = record['finish_reason'] == reference['finish_reason']
    return dict(token_ids_equal=same_tokens, termination_equal=same_finish,
        output_equal=same_tokens and same_finish, first_difference=first,
        record_output_tokens=len(left), reference_output_tokens=len(right),
        record_finish_reason=record['finish_reason'],
        reference_finish_reason=reference['finish_reason'])


def validate_measurement(record, *, arm):
    """Keep legitimate zero-call speculative rows and N/A AR rows distinct."""
    meta = record['metadata']
    for key in ('request_latency_s', 'native_generate_s'):
        value = meta[key]
        if not math.isfinite(value) or value < 0:
            raise ValueError('Invalid serving measurement: ' + key)
    if meta['native_generate_s'] > meta['request_latency_s']:
        raise ValueError('Native generation exceeds the complete request timer')
    if arm == 'AR':
        for key in ('verification_calls', 'accepted_draft_tokens',
                    'committed_progress', 'committed_progress_per_call',
                    'calls_per_1000_output_tokens'):
            if meta[key] is not None:
                raise ValueError('AR speculative metric must be N/A: ' + key)
        return
    calls, accepted = meta['verification_calls'], meta['accepted_draft_tokens']
    if type(calls) is not int or calls < 0 or type(accepted) is not int or accepted < 0:
        raise ValueError('Invalid speculative counts')
    blocks = meta['trace']
    if calls != len(blocks) or accepted != sum(b['accepted_draft_tokens'] for b in blocks):
        raise ValueError('Speculative counts disagree with their retained trace')
    # This field is the inherited stop-clipped accepted count. The separately
    # named native/raw proposal counters are not reconstructed from it.
    progress = len(record['response_ids']) - 1
    if accepted > progress or meta['committed_progress'] != progress:
        raise ValueError('Stop-clipped acceptance or committed progress is inconsistent')
    if calls == 0 and (len(record['response_ids']) != 1 or accepted != 0):
        raise ValueError('Zero-call request must consist only of its prefill token')
