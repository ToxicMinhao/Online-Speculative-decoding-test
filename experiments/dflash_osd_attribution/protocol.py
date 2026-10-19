"""Fixed scope and pairing rules; importing this module starts no model work."""
from __future__ import annotations

import hashlib
import json
from typing import Any

STREAMS = {'wildchat_00': 20, 'wildchat_01': 16, 'wildchat_02': 17,
           'wildchat_19': 23, 'personamem_persona42': 18}
MAX_ANCHORS_PER_REQUEST = 32
MAX_PAIRS = 160
MAX_BLOCK_EVALUATIONS = 320
BLOCK_SIZE = 10
PROPOSAL_COUNT = 9
OUTPUT_CAP = 1024
TIMING_REPEATS = (0, 1, 2)
SCHEMA = 'osd_gain_attribution_v1'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value: Any):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def spaced_indices(n: int):
    require(type(n) is int and n >= 0, 'Nonnegative number of distinct starts required')
    k = min(n, MAX_ANCHORS_PER_REQUEST)
    if k < 2:
        return list(range(k))
    # Endpoint-inclusive integer floor rule; unique for n>=k. No random seed,
    # token category, margin or observed OSD improvement enters selection.
    return [i*(n-1)//(k-1) for i in range(k)]


def select_anchors(stream, turn, input_ids, output_ids, frozen_trace, eos_ids):
    require(stream in STREAMS and turn == STREAMS[stream], 'Only the five final saved requests are eligible')
    require(frozen_trace['input_token_ids'] == input_ids and
            frozen_trace['final_output_token_ids'] == output_ids, 'Trace and saved canonical tokens differ')
    events = [e for e in frozen_trace['events'] if e['kind'] == 'verification']
    starts = sorted({e['output_offset'] for e in events})
    require(len(starts) == len(events), 'Duplicate frozen block starts are not silently deduplicated')
    by_start = {e['output_offset']: e for e in events}
    anchors = []
    for index in spaced_indices(len(starts)):
        offset = starts[index]; e = by_start[offset]
        require(type(offset) is int and 1 <= offset < len(output_ids), 'A draft block requires a canonical seed and following token')
        proposals = e['draft_proposed_token_ids']
        require(len(proposals) == PROPOSAL_COUNT, 'Pinned block10 must contain nine proposed positions')
        positions = list(range(len(input_ids)+offset, len(input_ids)+offset+PROPOSAL_COUNT))
        require(e['draft_proposed_positions'] == positions, 'Original absolute proposal positions differ')
        require(e['verification_input_token_ids'] == [output_ids[offset-1]]+proposals,
                'Native verification seed differs from canonical history')
        require(e['context']['committed_output_prefix_length'] == offset, 'Committed prefix length differs')
        prefix = input_ids+output_ids[:offset]
        anchors.append(dict(anchor_id=f'{stream}:t{turn}:p{offset}', stream=stream, turn=turn,
            block_output_start=offset, input_ids=input_ids, output_ids=output_ids,
            canonical_prefix_ids=prefix, canonical_prefix_sha256=digest(prefix),
            seed_token_id=output_ids[offset-1], expected_frozen_proposal_ids=proposals,
            canonical_next_ids=output_ids[offset:offset+PROPOSAL_COUNT],
            expected_absolute_positions=positions, eos_ids=sorted(set(eos_ids)),
            selection_index=index, total_distinct_frozen_starts=len(starts),
            selection_rule='endpoint-inclusive floor(i*(n-1)/(min(n,32)-1)); no outcome selection'))
    return anchors


def correct_prefix(proposals, canonical, eos_ids):
    length = 0
    for proposal, token in zip(proposals, canonical):
        if proposal != token:
            return length, 'mismatch'
        if token in eos_ids:
            return length, 'matched_eos'
        length += 1
    return length, 'available_canonical_end' if len(canonical) < len(proposals) else 'block_end'


def compare_anchor(anchor, frozen, osd):
    """Local matched-prefix evidence; never convert its repair rate to time saved."""
    reasons = []
    for arm, row in (('F', frozen), ('OSD', osd)):
        if row.get('status') != 'passed':
            reasons.append(arm+': native diagnostic failed')
        if row.get('anchor_id') != anchor['anchor_id'] or row.get('arm') != arm:
            reasons.append(arm+': identity mismatch')
        if row.get('canonical_prefix_ids') != anchor['canonical_prefix_ids']:
            reasons.append(arm+': exact prefix mismatch')
        if row.get('draft_block_evaluations') != 1:
            reasons.append(arm+': not exactly one draft block')
        for flag in ('valid_reconstruction', 'target_seed_matches', 'cache_valid',
                     'no_generated_history', 'no_training', 'native_hook_passed'):
            if row.get(flag) is not True:
                reasons.append(arm+': failed/missing '+flag)
        for flag in ('target_verifications', 'regenerated_response_tokens', 'training_steps'):
            if row.get(flag) != 0:
                reasons.append(arm+': nonzero/missing '+flag)
        if len(row.get('proposed_token_ids', [])) != PROPOSAL_COUNT:
            reasons.append(arm+': proposal geometry mismatch')
        expected = anchor['checkpoints'][arm]['checkpoint_identity_sha256']
        if row.get('checkpoint_identity_sha256') != expected:
            reasons.append(arm+': checkpoint mismatch')
    if frozen.get('proposed_token_ids') != anchor['expected_frozen_proposal_ids']:
        reasons.append('F: reconstructed proposals differ from retained native evidence')
    valid = not reasons
    lengths = {}; decisions = []
    for arm, row in (('F', frozen), ('OSD', osd)):
        lengths[arm] = correct_prefix(row.get('proposed_token_ids', []), anchor['canonical_next_ids'], anchor['eos_ids'])
    for i, token in enumerate(anchor['canonical_next_ids']):
        f = frozen.get('proposed_token_ids', []); o = osd.get('proposed_token_ids', [])
        if i >= len(f) or i >= len(o):
            break
        repaired, introduced = f[i] != token and o[i] == token, f[i] == token and o[i] != token
        details = frozen.get('positions', [])
        detail = details[i] if i < len(details) else {}
        capacity = dict(detail.get('capacity') or {})
        if valid and repaired and all(key in capacity for key in ('deficit', 'bound', 'tolerance')):
            gap, tol = capacity['deficit']-capacity['bound'], capacity['tolerance']
            capacity.update(eligible=True, ineligibility_reasons=[],
                classification='excluded' if gap > tol else ('borderline' if abs(gap) <= tol else 'not_excluded'))
        elif valid and repaired:
            capacity.update(eligible=False, classification='unavailable', reason='Missing mathematical capacity evidence')
        else:
            capacity.update(eligible=False, classification='not_evaluated',
                            reason='Bound denominator includes only valid OSD-repaired frozen errors')
        decisions.append(dict(anchor_id=anchor['anchor_id'], stream=anchor['stream'],
            turn=anchor['turn'], block_output_start=anchor['block_output_start'], block_position=i, native_block_slot=i+1,
            absolute_output_position=anchor['block_output_start']+i,
            absolute_token_position=len(anchor['input_ids'])+anchor['block_output_start']+i,
            canonical_token_id=token, frozen_proposal=f[i], osd_proposal=o[i], valid=valid,
            repair=bool(valid and repaired), introduced_error=bool(valid and introduced),
            within_frozen_correct_prefix=i < lengths['F'][0], within_osd_correct_prefix=i < lengths['OSD'][0],
            positional_only_after_earlier_mismatch=i > min(lengths['F'][0], lengths['OSD'][0]),
            frozen_native_margin=detail.get('native_bf16_canonical_margin'),
            osd_native_margin=(osd.get('positions', [])[i].get('native_bf16_canonical_margin')
                               if i < len(osd.get('positions', [])) else None),
            capacity=capacity))
        if token in anchor['eos_ids']:
            break
    return dict(anchor_id=anchor['anchor_id'], stream=anchor['stream'], turn=anchor['turn'],
        valid=valid, invalid_reasons=reasons,
        frozen_correct_prefix_length=lengths['F'][0], osd_correct_prefix_length=lengths['OSD'][0],
        correct_prefix_gain=lengths['OSD'][0]-lengths['F'][0] if valid else None,
        stop_reasons={arm: pair[1] for arm, pair in lengths.items()}, decisions=decisions,
        interpretation='Local fixed-start prediction changes; not a fraction of total native time saved.')
