"""Exact original PR9/PR10 diagnostics on small CPU tensors only."""
import copy
from unittest.mock import patch

import pytest

torch = pytest.importorskip('torch')
from . import evidence, loss, training
from .test_training import numeric_groups, new
from experiments.dflash_baseline_learning_curve.test_training import make_trainer
from experiments.dflash_kl_native_diagnostic import loss as heldout_loss


@pytest.fixture(autouse=True)
def one_thread():
    torch.set_num_threads(1)


def diagnostic_pool(dtype=torch.float32, heldout_length=None):
    groups = numeric_groups()
    for group in groups:
        base, _ = make_trainer(dtype=dtype)
        source = training.curve.FixedFeedbackTrainer(base.draft, base.embed_tokens, base.lm_head,
            base.teacher_features, episode_id=group['group_id'], target_id=group['support'][0]['target_id'],
            target_identity_sha256=group['support'][0]['target_identity_sha256'],
            released_checkpoint_sha256=group['checkpoints']['F']['checkpoint_identity_sha256'],
            mask_token_id=19, attention_backend='sdpa', projection_chunk=7)
        source_refs = source.source_losses(group['support'])['response_losses']
        for support, pair, reference in zip(group['support'], group['reference_pairs']['source'], source_refs):
            reference.update(request_id=support['record_id'], input_ids_sha256=loss.digest(support['input_ids']),
                output_ids_sha256=loss.digest(support['output_ids']))
            for label in ('F', 'S8'):
                pair[label]['kl'] = dict(reference, checkpoint_id=label,
                    checkpoint_identity_sha256=group['checkpoints'][label]['checkpoint_identity_sha256'])
        if heldout_length is not None:
            for pair in group['reference_pairs']['heldout']:
                for label in ('F', 'S8'):
                    row = pair[label]['native']
                    output = row['output_ids'][:heldout_length]
                    n = len(output)
                    row.update(output_ids=output, output_tokens=n, initial_output_tokens=1,
                        verification_calls=int(n > 1), accepted_draft_tokens=max(0, n-2),
                        committed_progress_per_call=n-1 if n > 1 else None,
                        emitted_tokens_per_verification_call=n if n > 1 else None,
                        verification_committed_tokens=n-1, rounds=[] if n == 1 else [dict(round_index=0,
                            output_offset=1, accepted_prefix_length=n-2, accepted_draft_tokens=n-2,
                            proposed_tokens=n-1, committed_tokens=n-1, committed_token_ids=output[1:],
                            first_rejection_offset=n-2, stop_clipped=False)])
        heldout = [pair['F']['native'] for pair in group['reference_pairs']['heldout']]
        refs = heldout_loss.evaluate_responses(base.draft, base.embed_tokens, base.lm_head, base.teacher_features,
            heldout, checkpoint_id='F', checkpoint_identity_sha256=group['checkpoints']['F']['checkpoint_identity_sha256'],
            mask_token_id=19, attention_backend='sdpa', projection_chunk=7)['response_losses']
        for pair, reference in zip(group['reference_pairs']['heldout'], refs):
            for label in ('F', 'S8'):
                pair[label]['kl'] = dict(reference, checkpoint_id=label,
                    checkpoint_identity_sha256=group['checkpoints'][label]['checkpoint_identity_sha256'])
        group['scores'] = [copy.deepcopy(p[label]['native']) for p in group['reference_pairs']['heldout']
            for label in ('F', 'S8')]
    return evidence.build_pools(groups)[0]


def inputs(pool, population):
    records = pool['support'] if population == 'source' else [p['F']['native'] for p in pool['reference_pairs'][population]]
    return records, [p['F']['kl'] for p in pool['reference_pairs'][population]]


def evaluate(trainer, pool, population):
    records, references = inputs(pool, population)
    return loss.evaluate_responses(trainer.draft, trainer.embed_tokens, trainer.lm_head, trainer.teacher_features,
        records, references, original_pool=pool, population=population, checkpoint_identity_sha256='e'*64,
        mask_token_id=19, attention_backend='sdpa', projection_chunk=7)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('population', ['source', 'heldout'])
def test_p8_exact_original_seeds_masks_positions_weights_denominators_without_optimizer(dtype, population):
    pool = diagnostic_pool(dtype)
    trainer, _ = new(pool, dtype=dtype, dropout=.7)
    trainer.draft.train()
    trainer.draft.dropout.eval()
    roots = (trainer.draft, trainer.embed_tokens, trainer.lm_head)
    modes = [m.training for root in roots for m in root.modules()]
    flags = [p.requires_grad for root in roots for p in root.parameters()]
    sampler = trainer.generator.get_state().clone()
    before = loss._state_hash(loss._rng_state())
    forward = trainer.draft.forward
    def frozen_forward(*args, **kwargs):
        assert not torch.is_grad_enabled() and not trainer.draft.training
        assert all(not p.requires_grad for root in roots for p in root.parameters())
        return forward(*args, **kwargs)
    with patch.object(torch.optim, 'AdamW', side_effect=AssertionError('optimizer forbidden')), \
         patch.object(torch.Tensor, 'backward', side_effect=AssertionError('backward forbidden')), \
         patch.object(trainer.draft, 'forward', side_effect=frozen_forward):
        result = evaluate(trainer, pool, population)
    records, refs = inputs(pool, population)
    for value, record, ref, pair in zip(result['response_losses'], records, refs, pool['reference_pairs'][population]):
        assert value['loss'] == ref['loss']
        assert value['checkpoint_id'] == 'P8' and value['pool_id'] == pool['pool_id']
        assert value['pool_request_id'] == pair['pool_request_id'] and value['group_id'] == record['group_id']
        assert value['record_id'] == ref['record_id'] and value['role'] == record['role']
        assert value['anchor_seed'] == ref['anchor_seed']
        assert value['anchor_rng_after_sha256'] == ref['anchor_rng_after_sha256']
        assert all(value[k] == ref[k] for k in loss.SAMPLING_FIELDS)
        assert value['response_loss_mask'] == [0]*len(record['input_ids'])+[1]*len(record['output_ids'])+[0]*9
    assert not result['optimizer_created'] and result['optimizer_steps'] == 0
    assert result['attempted_response_calls'] == result['returned_response_rows'] == 8
    assert result['unknown_response_calls'] == result['unattempted_response_calls'] == 0
    assert result['eligible_response_count']+result['excluded_response_count'] == 8
    assert before == loss._state_hash(loss._rng_state()) and torch.equal(sampler, trainer.generator.get_state())
    assert modes == [m.training for root in roots for m in root.modules()]
    assert flags == [p.requires_grad for root in roots for p in root.parameters()]
    assert all(p.grad is None for root in roots for p in root.parameters())


@pytest.mark.parametrize('change', ['source_episode', 'source_turn', 'request', 'order', 'tokens', 'reference'])
def test_changed_original_binding_fails_before_teacher(change):
    pool = diagnostic_pool()
    trainer, calls = new(pool)
    records, references = copy.deepcopy(inputs(pool, 'source'))
    if change == 'source_episode': records[0]['episode_id'] = pool['pool_id']
    if change == 'source_turn': records[0]['turn'] += 1
    if change == 'request': records[0]['request_id'] = 'other'
    if change == 'order': records.reverse()
    if change == 'tokens': records[0]['output_ids'][0] += 1
    if change == 'reference': references[0]['anchor_seed'] += 1
    with pytest.raises(ValueError):
        loss.evaluate_responses(trainer.draft, trainer.embed_tokens, trainer.lm_head, trainer.teacher_features,
            records, references, original_pool=pool, population='source', checkpoint_identity_sha256='e'*64,
            mask_token_id=19, attention_backend='sdpa', projection_chunk=7)
    assert not calls


def test_sampling_failure_preserves_failed_row_and_restores_rng_modes():
    pool = diagnostic_pool()
    trainer, _ = new(pool, dropout=.7)
    before = loss._state_hash(loss._rng_state())
    modes = [m.training for m in trainer.draft.modules()]
    original = loss._evaluate_one
    def changed(*args, **kwargs):
        value = original(*args, **kwargs)
        value['position_weights'][0][1] += 1
        return value
    with patch.object(loss, '_evaluate_one', side_effect=changed), pytest.raises(ValueError) as error:
        evaluate(trainer, pool, 'heldout')
    partial = error.value.diagnostic_partial
    assert len(partial['response_losses']) == 1 and partial['global_rng_restored']
    assert partial['attempted_response_calls'] == partial['returned_response_rows'] == 1
    assert partial['unknown_response_calls'] == 0 and partial['unattempted_response_calls'] == 7
    assert partial['response_losses'][0]['pool_id'] == pool['pool_id']
    assert before == loss._state_hash(loss._rng_state())
    assert modes == [m.training for m in trainer.draft.modules()]


def test_teacher_failure_preserves_unknown_attempt_and_unattempted_rows():
    pool = diagnostic_pool()
    trainer, _ = new(pool)
    teacher = trainer.teacher_features
    count = 0
    def fail_second(ids):
        nonlocal count
        count += 1
        if count == 2:
            raise RuntimeError('Teacher failure in second request')
        return teacher(ids)
    with patch.object(trainer, 'teacher_features', side_effect=fail_second), pytest.raises(RuntimeError) as error:
        evaluate(trainer, pool, 'heldout')
    partial = error.value.diagnostic_partial
    assert partial['population'] == 'heldout'
    assert partial['attempted_response_calls'] == 2 and partial['returned_response_rows'] == 1
    assert partial['unknown_response_calls'] == 1 and partial['unattempted_response_calls'] == 6
    assert len(partial['response_losses']) == 1 and partial['global_rng_restored']


@pytest.mark.parametrize('length', [1, 2])
def test_zero_or_short_supervision_is_retained_without_resampling(length):
    pool = diagnostic_pool(heldout_length=length)
    trainer, calls = new(pool)
    result = evaluate(trainer, pool, 'heldout')
    for row, ref in zip(result['response_losses'], inputs(pool, 'heldout')[1]):
        assert row['anchor_sampling_attempts'] == int(length >= 2)
        assert row['sampling_and_weights_sha256'] == ref['sampling_and_weights_sha256']
        assert row['zero_supervision'] == ref['zero_supervision']
        if row['zero_supervision']:
            assert not row['loss_improvement_eligible'] and row['excluded_from_loss_improvement']
    if length == 1:
        assert result['eligible_response_count'] == 0 and result['loss_mean'] is None and not calls


def test_original_l8_evaluator_still_rejects_p8():
    pool = diagnostic_pool()
    trainer, calls = new(pool)
    rows, refs = inputs(pool, 'source')
    with pytest.raises(ValueError, match='Only the new L8'):
        loss.qualified.evaluate_responses(trainer.draft, trainer.embed_tokens, trainer.lm_head, trainer.teacher_features,
            rows, refs, population='source', checkpoint_id='P8', checkpoint_identity_sha256='e'*64)
    assert not calls
