import copy
import pytest

from .protocol import compare_anchor, correct_prefix, select_anchors, spaced_indices
from .test_native_probe import anchor


def paired():
    a = anchor()
    rows = []
    for arm in ('F', 'OSD'):
        rows.append(dict(status='passed', anchor_id=a['anchor_id'], arm=arm,
            canonical_prefix_ids=a['canonical_prefix_ids'], draft_block_evaluations=1,
            checkpoint_identity_sha256=a['checkpoints'][arm]['checkpoint_identity_sha256'],
            proposed_token_ids=a['expected_frozen_proposal_ids'][:], valid_reconstruction=True,
            target_seed_matches=True, cache_valid=True, no_generated_history=True, no_training=True,
            native_hook_passed=True, target_verifications=0, regenerated_response_tokens=0, training_steps=0,
            positions=[dict(native_bf16_canonical_margin=1000, capacity=dict(deficit=3., bound=1.,
                tolerance=1e-7, classification='not_evaluated', eligible=False)) for _ in range(4)]))
    a['expected_frozen_proposal_ids'][0] = rows[0]['proposed_token_ids'][0] = 90
    return a, rows[0], rows[1]


@pytest.mark.parametrize('n', [0, 1, 2, 31, 32, 33, 128, 1024])
def test_even_spacing_and_budget(n):
    indices = spaced_indices(n)
    assert len(indices) == min(n, 32)
    assert len(set(indices)) == len(indices)
    if n:
        assert indices[0] == 0 and indices[-1] == n-1


def test_eos_excluded_from_prefix():
    assert correct_prefix([1, 2, 3], [1, 2, 3], [2]) == (1, 'matched_eos')
    assert correct_prefix([1, 8, 3], [1, 2, 3], [2]) == (1, 'mismatch')


def test_capacity_classified_after_pairing_and_native_margin_not_substituted():
    a, f, o = paired()
    result = compare_anchor(a, f, o)
    assert result['valid']
    decision = result['decisions'][0]
    assert decision['repair']
    assert decision['capacity']['classification'] == 'excluded'
    assert decision['capacity']['eligible']
    assert decision['frozen_native_margin'] == 1000
    assert not result['decisions'][0]['positional_only_after_earlier_mismatch']
    assert result['decisions'][1]['positional_only_after_earlier_mismatch']


@pytest.mark.parametrize('field,value', [
    ('valid_reconstruction', False), ('target_seed_matches', False), ('cache_valid', False),
    ('native_hook_passed', False), ('status', 'failed'), ('canonical_prefix_ids', [999]),
    ('checkpoint_identity_sha256', 'later-checkpoint'), ('draft_block_evaluations', 2),
    ('training_steps', 1), ('target_verifications', 1), ('no_generated_history', False)])
def test_invalid_pairs_stay_visible_without_repairs_or_capacity_claims(field, value):
    a, f, o = paired(); o[field] = value
    value = compare_anchor(a, f, o)
    assert not value['valid'] and value['invalid_reasons']
    assert not any(d['repair'] or d['capacity']['eligible'] for d in value['decisions'])


def test_nonrepair_and_missing_evidence():
    a, f, o = paired()
    f['positions'][0].pop('capacity')
    result = compare_anchor(a, f, o)
    assert result['decisions'][0]['capacity']['classification'] == 'unavailable'
    assert result['decisions'][1]['capacity']['classification'] == 'not_evaluated'


def test_selected_final_anchors_include_exact_seed_and_positions():
    a = anchor(); ids, outputs = a['input_ids'], a['output_ids']
    event = dict(kind='verification', output_offset=1, draft_proposed_token_ids=a['expected_frozen_proposal_ids'],
        draft_proposed_positions=list(range(3,12)), verification_input_token_ids=[20]+a['expected_frozen_proposal_ids'],
        context=dict(committed_output_prefix_length=1))
    trace = dict(input_token_ids=ids, final_output_token_ids=outputs, events=[event])
    result = select_anchors('wildchat_00', 20, ids, outputs, trace, [24])
    assert result[0]['canonical_prefix_ids'] == [10, 11, 20]
    assert result[0]['canonical_next_ids'] == [21,22,23,24]
    with pytest.raises(ValueError, match='final'):
        select_anchors('wildchat_00', 19, ids, outputs, trace, [24])
    trace['events'].append(copy.deepcopy(event))
    with pytest.raises(ValueError, match='Duplicate'):
        select_anchors('wildchat_00', 20, ids, outputs, trace, [24])
