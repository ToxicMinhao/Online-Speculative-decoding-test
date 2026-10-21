"""Small synthetic CPU training checks; not native-model qualification."""
import copy
import json
from unittest.mock import patch

import pytest

torch = pytest.importorskip('torch')
from . import evidence, training
from .test_evidence import groups_fixture
from experiments.dflash_baseline_learning_curve.test_training import make_trainer, loss_at, pool as historical_pool
from experiments.dflash_kl_native_diagnostic.test_experiment import loss_row


@pytest.fixture(autouse=True)
def one_thread():
    torch.set_num_threads(1)


def numeric_groups():
    """Full original-evidence fixtures remapped into a tiny synthetic vocabulary."""
    groups = groups_fixture(colliding_request_ids=True)
    for gi, group in enumerate(groups):
        for population, pairs in group['reference_pairs'].items():
            for index, pair in enumerate(pairs):
                prompt = [gi+1, index+(8 if population == 'heldout' else 0)]
                output = [(index+i+gi) % 19 for i in range(3 if gi % 2 == 0 else 9)]
                for label in ('F', 'S8'):
                    row = pair[label]['native']
                    count = len(output)-1
                    row.update(input_ids=prompt, output_ids=output, input_ids_sha256=training.digest(prompt),
                        output_tokens=len(output), eos_hit=False, finish_reason={'type': 'synthetic_stop'},
                        accepted_draft_tokens=count-1, verification_committed_tokens=count,
                        committed_progress_per_call=float(count), emitted_tokens_per_verification_call=float(len(output)),
                        rounds=[dict(round_index=0, output_offset=1, accepted_prefix_length=count-1,
                            accepted_draft_tokens=count-1, proposed_tokens=count, committed_tokens=count,
                            committed_token_ids=output[1:], first_rejection_offset=count-1, stop_clipped=False)])
                    pair[label]['kl'] = loss_row(row)
        group['feedback'] = [copy.deepcopy(p['F']['native']) for p in group['reference_pairs']['source']]
        group['scores'] = [copy.deepcopy(p[label]['native']) for p in group['reference_pairs']['heldout']
            for label in ('F', 'S8')]
        for support, native in zip(group['support'], group['feedback']):
            support.update(input_ids=native['input_ids'], output_ids=native['output_ids'],
                input_token_count=len(native['input_ids']), input_ids_sha256=training.digest(native['input_ids']),
                output_ids_sha256=training.digest(native['output_ids']))
        for request, pair in zip(group['evaluation_requests'], group['reference_pairs']['heldout']):
            request['input_ids'] = pair['F']['native']['input_ids']
        from . import loss
        for population, pairs in group['reference_pairs'].items():
            for index, pair in enumerate(pairs):
                record = pair['F']['native']
                if population == 'source':
                    record = dict(group['support'][index], group_id=group['group_id'],
                        request_id=pair['request_id'], role='feedback')
                audit = loss.audit_record(record)
                seed = loss.source_anchor_seed(record, audit) if population == 'source' else loss.heldout_anchor_seed(audit)
                for label in ('F', 'S8'):
                    pair[label]['kl'].update(anchor_seed=seed, anchor_rng_after_sha256='b'*64)
    return groups


def numeric_pool():
    return evidence.build_pools(numeric_groups())[0]


def new(pool=None, *, dtype=torch.float32, dropout=0., artifact_dir=None):
    pool = numeric_pool() if pool is None else pool
    base, calls = make_trainer(dtype=dtype, dropout=dropout)
    support = pool['support'][0]
    trainer = training.PooledFeedbackTrainer(base.draft, base.embed_tokens, base.lm_head, base.teacher_features,
        original_pool=pool, target_id=support['target_id'], target_identity_sha256=support['target_identity_sha256'],
        released_checkpoint_sha256=pool['checkpoints']['F']['checkpoint_identity_sha256'], mask_token_id=19,
        attention_backend='sdpa', projection_chunk=7, artifact_dir=artifact_dir)
    return trainer, calls


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_two_passes_match_manual_separate_normalization_clip_once_and_fresh_adam(dtype, tmp_path):
    pool = numeric_pool()
    trainer, calls = new(pool, dtype=dtype, artifact_dir=tmp_path)
    control, _ = new(pool, dtype=dtype)
    rows = pool['support']
    assert trainer.optimizer.state == {} and trainer.optimizer.param_groups[0]['lr'] == 3e-4
    control.draft.train()
    different_from_pooled = False
    for step in range(8):
        control.optimizer.zero_grad(set_to_none=True)
        values = [loss_at(control, row, control.generator) for row in rows[(step % 4)*2:(step % 4)*2+2]]
        if step == 0:
            mean = sum(v[0] for v in values)/2
            weighted = sum(v[0]*v[1]['normalization_denominator'] for v in values)/sum(
                v[1]['normalization_denominator'] for v in values)
            a = torch.autograd.grad(mean, tuple(control.parameters.values()), retain_graph=True)
            b = torch.autograd.grad(weighted, tuple(control.parameters.values()), retain_graph=True)
            different_from_pooled = any(not torch.allclose(x.float(), y.float(), atol=1e-5, rtol=1e-4)
                for x, y in zip(a, b))
        for value, _ in values:
            (value/2).backward()
        torch.nn.utils.clip_grad_norm_(list(control.parameters.values()), 1., error_if_nonfinite=True)
        control.optimizer.step()
        control.optimizer.zero_grad(set_to_none=True)
        control.optimizer_steps += 1
    assert different_from_pooled
    with patch.object(torch.nn.utils, 'clip_grad_norm_', wraps=torch.nn.utils.clip_grad_norm_) as clip:
        account = trainer.train_eight(rows)
    assert clip.call_count == 8
    for name, parameter in trainer.parameters.items():
        assert torch.equal(parameter, control.parameters[name])
    assert training.curve.state_digest(trainer.optimizer.state_dict()) == training.curve.state_digest(control.optimizer.state_dict())
    assert account['response_order'] == [r['pool_request_id'] for r in rows]*2
    assert len(set(account['response_order'])) == 8
    assert [b['pass_index'] for b in account['batches']] == [0]*4+[1]*4
    assert len(calls) == 16 and all(tuple(c[0].tolist()) in {tuple(r['input_ids']+r['output_ids']) for r in rows} for c in calls)
    assert training.validate_training_accounting(json.loads(json.dumps(account)), pool)['status'] == 'passed'
    assert len((tmp_path/'TRAIN_STEPS.jsonl').read_text().splitlines()) == 8
    assert not (tmp_path/'SOURCE_LOSSES.jsonl').exists()
    assert trainer.source_anchor_specs is None and account['source_diagnostic_steps'] == []
    assert all(p.grad is None for p in trainer.parameters.values())


def test_colliding_turns_and_request_ids_preserve_source_ownership():
    pool = numeric_pool()
    trainer, _ = new(pool)
    account = trainer.train_eight(pool['support'])
    assert len({r['turn'] for r in pool['support']}) == len({r['record_id'] for r in pool['support']}) == 4
    values = [p for batch in account['batches'] for p in batch['presentations']]
    for index, value in enumerate(values):
        original = pool['support'][index % 8]
        assert all(value[k] == original[k] for k in (*training.ORIGINAL_FIELDS, 'pool_id', 'pool_request_id'))
        assert value['episode_id'] != pool['pool_id']
    assert len(trainer.seen_record_ids) == 8


@pytest.mark.parametrize('change', ['role', 'tokens', 'order', 'turn', 'episode', 'pool_request'])
def test_changed_support_is_rejected_before_any_teacher_or_optimizer_work(change):
    pool = numeric_pool()
    trainer, calls = new(pool)
    rows = copy.deepcopy(pool['support'])
    if change == 'role': rows[0]['role'] = 'evaluation'
    if change == 'tokens': rows[0]['output_ids'][0] += 1
    if change == 'order': rows.reverse()
    if change == 'turn': rows[0]['turn'] += 1
    if change == 'episode': rows[0]['episode_id'] = pool['pool_id']
    if change == 'pool_request': rows[0]['pool_request_id'] = rows[1]['pool_request_id']
    with pytest.raises(ValueError): trainer.train_eight(rows)
    assert trainer.optimizer_steps == 0 and not trainer.optimizer.state and not calls


@pytest.mark.parametrize('population', ['source', 'heldout'])
def test_missing_original_diagnostic_evidence_rejected_before_new_optimizer(population):
    groups = numeric_groups()
    for label in ('F', 'S8'):
        groups[0]['reference_pairs'][population][0][label]['kl']['anchor_seed'] += 1
    pool = evidence.build_pools(groups)[0]
    base, calls = make_trainer()
    with patch.object(torch.optim, 'AdamW', side_effect=AssertionError('No optimizer before admission')), \
         pytest.raises(ValueError, match='seed differs'):
        training.PooledFeedbackTrainer(base.draft, base.embed_tokens, base.lm_head, base.teacher_features,
            original_pool=pool, target_id=pool['support'][0]['target_id'],
            target_identity_sha256=pool['support'][0]['target_identity_sha256'],
            released_checkpoint_sha256=pool['checkpoints']['F']['checkpoint_identity_sha256'], mask_token_id=19)
    assert not calls


def test_training_sampler_must_remain_fresh_until_first_presentation():
    pool = numeric_pool()
    trainer, calls = new(pool)
    torch.rand(1, generator=trainer.generator)
    with pytest.raises(ValueError, match='seed-17'):
        trainer.train_eight(pool['support'])
    assert not calls and not trainer.optimizer.state


@pytest.mark.parametrize('change', ['scale', 'lr', 'owner', 'mask', 'denominator', 'rng', 'order', 'steps'])
def test_accounting_rejects_changed_recipe_identity_or_observed_exposure(change):
    pool = numeric_pool()
    trainer, _ = new(pool)
    account = trainer.train_eight(pool['support'])
    batch = account['batches'][0]
    value = batch['presentations'][0]
    if change == 'scale': batch['backward_response_scale'] = 1.
    if change == 'lr': batch['effective_learning_rate'] = 3e-5
    if change == 'owner': value['episode_id'] = pool['pool_id']
    if change == 'mask': value['unique_supervised_position_ids'] = [0]
    if change == 'denominator': value['normalization_denominator'] += 1
    if change == 'rng': value['anchor_rng_before_sha256'] = 'a'*64
    if change == 'order': account['response_order'].reverse()
    if change == 'steps': account['optimizer_steps_total'] = 7
    with pytest.raises(ValueError): training.validate_training_accounting(account, pool)


def test_partial_backward_and_unknown_optimizer_failure_are_preserved(tmp_path):
    pool = numeric_pool()
    trainer, _ = new(pool, artifact_dir=tmp_path)
    real = trainer._train_response
    count = 0
    def fail_second(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 2: raise RuntimeError('second response failed')
        return real(*args, **kwargs)
    with patch.object(trainer, '_train_response', side_effect=fail_second), pytest.raises(RuntimeError) as error:
        trainer.train_eight(pool['support'])
    partial = error.value.training_partial
    assert partial['attempted_response_calls'] == 2
    assert partial['completed_backward_presentations'] == partial['uncommitted_backward_presentations'] == 1
    assert partial['unknown_backward_response_calls'] == 1 and partial['unattempted_response_calls'] == 14
    assert partial['pending_response_identity']['pool_request_id'] == pool['support'][1]['pool_request_id']
    assert partial['optimizer_steps'] == partial['response_presentations'] == partial['optimizer_steps_unknown'] == 0
    assert (tmp_path/'FAILURES.jsonl').exists() and trainer.failed
    other, _ = new(pool)
    with patch.object(other.optimizer, 'step', side_effect=RuntimeError('uncertain optimizer mutation')), pytest.raises(RuntimeError) as error:
        other.train_eight(pool['support'])
    assert error.value.training_partial['optimizer_steps_unknown'] == 1
    assert error.value.training_partial['optimizer_steps'] == 0


def test_p8_is_one_fresh_run_final_binding_only_and_original_guards_unchanged():
    pool = numeric_pool()
    trainer, _ = new(pool)
    for method in ('restore', 'snapshot', 'source_losses', 'train_to_step', 'bind_checkpoint', 'train_event'):
        with pytest.raises(ValueError): getattr(trainer, method)({})
    with pytest.raises(ValueError): trainer.bind_p8_checkpoint('e'*64)
    trainer.train_eight(pool['support'])
    trainer.bind_p8_checkpoint('e'*64)
    assert trainer.active_checkpoint_sha256 == 'e'*64 and not trainer.awaiting_checkpoint
    with pytest.raises(ValueError, match='once'): trainer.bind_p8_checkpoint('e'*64)
    with pytest.raises(ValueError, match='once'): trainer.train_eight(pool['support'])
    old, _ = make_trainer()
    with pytest.raises(ValueError, match='progression'): old.train_to_step(8, historical_pool())
    with pytest.raises(ValueError, match='Measure fixed-anchor'): old.train_to_step(2, historical_pool())
    from experiments.dflash_lower_lr_control import training as lower
    assert lower.RECIPE['learning_rate'] == 3e-5 and lower.RECIPE['passes'] == 4
    assert training.curve.RECIPE['learning_rate'] == 3e-4 and training.curve.RECIPE['passes'] == 4
    assert training.RECIPE['learning_rate'] == 3e-4 and training.RECIPE['passes'] == 2
