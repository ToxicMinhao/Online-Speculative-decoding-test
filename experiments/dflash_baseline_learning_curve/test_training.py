"""CPU contract/numerical tests; not native model or GPU qualification."""
import copy
import importlib.util
import json
from pathlib import Path
import random
from unittest.mock import patch

import pytest

torch = pytest.importorskip('torch')
from torch import nn
from torch.nn import functional as F
from . import training

RELEASED, TARGET, S2, S8 = 'a'*64, 'c'*64, 'b'*64, 'd'*64


def response(turn, length=None):
    prompt = [(turn+i)%19 for i in range(7+turn)]
    return dict(episode_id='group', target_id='target', target_identity_sha256=TARGET,
        capture_checkpoint_sha256=RELEASED, record_id=f'group:r{turn}', turn=turn,
        input_ids=prompt, input_token_count=len(prompt), output_ids=[(turn+2*i)%19 for i in range(length or 8+turn)],
        output_cap=1024, role='feedback')


def pool():
    return [response(i) for i in range(1,5)]


@pytest.fixture(autouse=True)
def single_thread():
    torch.set_num_threads(1)


def make_trainer(*, artifact_dir=None, dropout=0.0, dtype=torch.float32):
    torch.manual_seed(41)
    class Draft(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(8,4)
            self.mix = nn.Linear(4,4)
            self.dropout = nn.Dropout(dropout)
        def forward(self, *, position_ids, noise_embedding, target_hidden, attention_mask):
            kv = torch.cat((self.fc(target_hidden),noise_embedding),dim=1).unsqueeze(1)
            hidden = F.scaled_dot_product_attention(noise_embedding.unsqueeze(1),kv,kv,
                attn_mask=attention_mask).squeeze(1)
            return self.mix(self.dropout(hidden+noise_embedding)),None
    draft=Draft().to(dtype)
    embed=nn.Embedding(20,4).to(dtype).requires_grad_(False).eval()
    head=nn.Linear(4,20,bias=False).to(dtype).requires_grad_(False).eval()
    calls=[]
    def teacher(ids):
        calls.append(ids.detach().cpu().clone())
        base=embed(ids)
        causal=base.cumsum(1)/torch.arange(1,ids.shape[1]+1,device=ids.device,dtype=dtype).view(1,-1,1)
        return torch.cat((base,causal),dim=-1),causal
    result=training.FixedFeedbackTrainer(draft,embed,head,teacher,episode_id='group',target_id='target',
        target_identity_sha256=TARGET,released_checkpoint_sha256=RELEASED,mask_token_id=19,
        attention_backend='sdpa',projection_chunk=7,artifact_dir=artifact_dir)
    return result,calls


def loss_at(trainer,response,generator):
    audit=training.audit_response(response)
    ids=torch.tensor([response['input_ids']+response['output_ids']])
    with torch.no_grad():
        h,t=trainer.teacher_features(ids)
    ids=torch.cat((ids,ids.new_full((1,9),19)),dim=1)
    h=torch.cat((h,h.new_zeros((1,9,h.shape[-1]))),dim=1)
    t=torch.cat((t,t.new_zeros((1,9,t.shape[-1]))),dim=1)
    mask=torch.zeros_like(ids,dtype=torch.float32)
    mask[:,audit['response_start']:audit['response_end_exclusive']]=1
    return training.author.author_response_loss(trainer.draft,trainer.embed_tokens,trainer.lm_head,
        ids,h,t,mask,mask_token_id=19,generator=generator,attention_backend='sdpa',projection_chunk=7,
        checkpoint_projection=False)


def test_historical_recipe_is_not_mutated_and_only_fixed_schedule_changes():
    path=Path(training.__file__).parents[1]/'dflash_two_update_followup/author_training.py'
    spec=importlib.util.spec_from_file_location('_baseline_test_historical',path)
    historical=importlib.util.module_from_spec(spec);spec.loader.exec_module(historical)
    assert historical.OUTPUT_CAP==256
    assert historical.RECIPE['passes']==2 and historical.RECIPE['responses_per_step']==2
    assert training.author.OUTPUT_CAP==1024
    for name in historical.RECIPE:
        if name not in ('passes','optimizer_steps_per_event'):
            assert training.RECIPE[name]==historical.RECIPE[name]
    assert training.RECIPE['passes']==4
    assert training.RECIPE['optimizer_steps_per_trajectory']==8
    assert 'optimizer_steps_per_event' not in training.RECIPE
    assert [r['record_id'] for _,_,pair in training.step_schedule(0,2,pool()) for r in pair]==[r['record_id'] for r in pool()]
    assert [r['record_id'] for _,_,pair in training.step_schedule(2,8,pool()) for r in pair]==[r['record_id'] for r in pool()]*3
    for before,after in [(0,8),(2,2),(8,8),(8,10),(0,1)]:
        with pytest.raises(ValueError,match='progression'):
            training.step_schedule(before,after,pool())


def test_full_prompt_response_only_masks_and_context_envelope():
    r=response(1,length=1024);r['input_ids']=list(range(27000));r['input_token_count']=27000
    audit=training.audit_response(r)
    assert audit['real_sequence_length']==28024 and audit['sampled_anchor_count']==512
    assert audit['loss_mask_sha256']==training.digest([0]*27000+[1]*1024+[0]*9)
    r['input_ids']=list(range(31744));r['input_token_count']=31744
    with pytest.raises(ValueError,match='context envelope'):
        training.audit_response(r)


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_pair_mean_matches_independent_normalization_with_one_clip_step(dtype):
    trainer,_=make_trainer(dtype=dtype);rows=pool();rows[0]=response(1,length=5);rows[1]=response(2,length=19)
    trainer.source_losses(rows)
    initial=trainer.snapshot()
    control,_=make_trainer(dtype=dtype);control.restore(initial)
    expected_grads=[];pooled_different=False
    control.draft.train()
    for offset in [0,2]:
        control.optimizer.zero_grad(set_to_none=True)
        losses=[loss_at(control,r,control.generator) for r in rows[offset:offset+2]]
        independent=(losses[0][0]+losses[1][0])/2
        denoms=[v[1]['normalization_denominator'] for v in losses]
        pooled=(losses[0][0]*denoms[0]+losses[1][0]*denoms[1])/sum(denoms)
        if offset==0:
            independent_grads=torch.autograd.grad(independent,tuple(control.parameters.values()),retain_graph=True)
            pooled_grads=torch.autograd.grad(pooled,tuple(control.parameters.values()),retain_graph=True)
            pooled_different=any(not torch.allclose(a.float(),b.float(),rtol=1e-4,atol=1e-5)
                                 for a,b in zip(independent_grads,pooled_grads))
        independent.backward()
        torch.nn.utils.clip_grad_norm_(list(control.parameters.values()),1.,error_if_nonfinite=True)
        expected_grads.append({k:p.grad.clone() for k,p in control.parameters.items()})
        control.optimizer.step();control.optimizer.zero_grad(set_to_none=True);control.optimizer_steps+=1
    captured=[];real_step=trainer.optimizer.step
    def step(*args,**kwargs):
        captured.append({k:p.grad.clone() for k,p in trainer.parameters.items()})
        return real_step(*args,**kwargs)
    with patch.object(trainer.optimizer,'step',side_effect=step),patch.object(
            torch.nn.utils,'clip_grad_norm_',wraps=torch.nn.utils.clip_grad_norm_) as clip:
        result=trainer.train_to_step(2,rows)
    assert len(captured)==clip.call_count==2 and pooled_different
    for actual,expected in zip(captured,expected_grads):
        for name in actual:
            torch.testing.assert_close(actual[name],expected[name],rtol=0.02 if dtype==torch.bfloat16 else 3e-6,
                                       atol=0.0005 if dtype==torch.bfloat16 else 3e-7)
    for p,q in zip(trainer.parameters.values(),control.parameters.values()):
        torch.testing.assert_close(p,q,rtol=0.01 if dtype==torch.bfloat16 else 1e-6,
                                  atol=0.001 if dtype==torch.bfloat16 else 1e-7)
    assert all(b['backward_response_scale']==0.5 for b in result['batches'])


def test_four_passes_keep_optimizer_sampler_and_saved_order(tmp_path):
    trainer,calls=make_trainer(artifact_dir=tmp_path);rows=pool();owner=id(trainer.optimizer)
    source0=trainer.source_losses(rows)
    first=trainer.train_to_step(2,rows);trainer.bind_checkpoint(1,S2)
    source2=trainer.source_losses(rows)
    second=trainer.train_to_step(8,rows);trainer.bind_checkpoint(2,S8)
    source8=trainer.source_losses(rows)
    assert id(trainer.optimizer)==owner and trainer.optimizer_steps==8
    assert trainer.response_presentations==16 and trainer.completed_events==2
    assert {int(s['step']) for s in trainer.optimizer.state.values()}=={8}
    assert first['optimizer_steps_this_event']==2 and second['optimizer_steps_this_event']==6
    assert first['response_order']+second['response_order']==[r['record_id'] for r in rows]*4
    assert [b['pass_index'] for b in first['batches']+second['batches']]==[0,0,1,1,2,2,3,3]
    assert first['anchor_rng_after_sha256']==second['anchor_rng_before_sha256']
    assert source0['fixed_anchor_set_sha256']==source2['fixed_anchor_set_sha256']==source8['fixed_anchor_set_sha256']
    # Teacher sees each independent prompt+response, never a concatenated pool.
    allowed={tuple(r['input_ids']+r['output_ids']) for r in rows}
    assert all(tuple(v[0].tolist()) in allowed for v in calls)
    for event in [first,second]:
        for batch in event['batches']:
            for v in batch['presentations']:
                assert all(v['response_start']<p<v['response_end_exclusive'] for p in v['unique_supervised_position_ids'])
                for labels,ptrs,mask in zip(v['label_positions'],v['teacher_positions'],v['binary_loss_mask']):
                    assert mask[0]==0
                    assert all(ptr==pos-1 for pos,ptr,active in zip(labels,ptrs,mask) if active)
    assert len((tmp_path/'TRAIN_STEPS.jsonl').read_text().splitlines())==8
    assert len((tmp_path/'SOURCE_LOSSES.jsonl').read_text().splitlines())==3
    assert all(p.grad is None for p in trainer.embed_tokens.parameters())
    assert all(p.grad is None for p in trainer.lm_head.parameters())


def test_fixed_source_diagnostics_do_not_change_training_randomness_or_modes():
    trainer,_=make_trainer(dropout=.25);rows=pool();trainer.source_losses(rows)
    initial=trainer.snapshot();before_global=training.state_digest(training.rng_state())
    before_sampler=trainer.generator.get_state().clone();before_weights=copy.deepcopy(trainer.draft.state_dict())
    trainer.draft.train();trainer.draft.dropout.eval()
    modes=[m.training for m in trainer.draft.modules()]
    repeated=trainer.source_losses(rows)
    assert before_global==training.state_digest(training.rng_state())
    assert torch.equal(before_sampler,trainer.generator.get_state())
    assert modes==[m.training for m in trainer.draft.modules()]
    assert all(torch.equal(v,before_weights[k]) for k,v in trainer.draft.state_dict().items())
    assert repeated['training_rng_preserved']
    with_diag=trainer.train_to_step(2,rows);trainer.bind_checkpoint(1,S2)
    expected=trainer.snapshot()
    other,_=make_trainer(dropout=.25);other.restore(initial)
    without_diag=other.train_to_step(2,rows);other.bind_checkpoint(1,S2)
    assert with_diag['anchor_exposure_sha256']==without_diag['anchor_exposure_sha256']
    assert all(torch.equal(v,expected['draft'][k]) for k,v in other.draft.state_dict().items())
    assert training.state_digest(other.optimizer.state_dict())==training.state_digest(expected['optimizer'])


def test_source_failure_restores_rng_and_modes_and_poison_trainer(tmp_path):
    trainer,_=make_trainer(dropout=.25,artifact_dir=tmp_path)
    before=training.rng_state();sampler=trainer.generator.get_state().clone();modes=[m.training for m in trainer.draft.modules()]
    with patch.object(trainer,'teacher_features',side_effect=RuntimeError('controlled teacher failure')):
        with pytest.raises(RuntimeError,match='controlled teacher failure'):
            trainer.source_losses(pool())
    assert training.state_digest(before)==training.state_digest(training.rng_state())
    assert torch.equal(sampler,trainer.generator.get_state())
    assert modes==[m.training for m in trainer.draft.modules()] and trainer.failed
    failure=json.loads((tmp_path/'FAILURES.jsonl').read_text().strip())
    assert failure['global_rng_restored'] and failure['training_anchor_rng_restored']
    assert failure['operation']=='source_losses'


def test_immutable_frozen_feedback_and_checkpoint_boundaries():
    trainer,_=make_trainer();rows=pool()
    with pytest.raises(ValueError,match='Measure fixed-anchor'):
        trainer.train_to_step(2,rows)
    trainer.source_losses(rows);trainer.train_to_step(2,rows)
    with pytest.raises(ValueError,match='pending'):
        trainer.train_to_step(8,rows)
    trainer.bind_checkpoint(1,S2)
    with pytest.raises(ValueError,match='step 2'):
        trainer.train_to_step(8,rows)
    trainer.source_losses(rows)
    for mutation in ['order','tokens','checkpoint','role']:
        altered=copy.deepcopy(rows)
        if mutation=='order':altered=altered[::-1]
        if mutation=='tokens':altered[0]['output_ids'][0]+=1
        if mutation=='checkpoint':altered[0]['capture_checkpoint_sha256']=S2
        if mutation=='role':altered[0]['role']='evaluation'
        with pytest.raises(ValueError):trainer.train_to_step(8,altered)
        assert trainer.optimizer_steps==2 and not trainer.failed
    assert all(r['capture_checkpoint_sha256']==RELEASED for r in rows)


def test_saved_recovery_preserves_moments_sampler_fixed_anchors_and_continuation(tmp_path):
    trainer,_=make_trainer();rows=pool();trainer.source_losses(rows)
    trainer.train_to_step(2,rows);trainer.bind_checkpoint(1,S2);trainer.source_losses(rows)
    receipt=trainer.save_checkpoint_state(tmp_path/'step2.pt',include_draft=True)
    saved=torch.load(receipt['path'],weights_only=False)
    assert receipt['optimizer_steps']==2 and receipt['checkpoint_sha256']==S2 and receipt['checkpoint_persistence_s']>=0
    with pytest.raises(ValueError,match='fresh path'):trainer.save_checkpoint_state(tmp_path/'step2.pt')
    result=trainer.train_to_step(8,rows);trainer.bind_checkpoint(2,S8);trainer.source_losses(rows)
    expected=trainer.snapshot()
    other,_=make_trainer();other.restore(saved)
    actual=other.train_to_step(8,rows);other.bind_checkpoint(2,S8);other.source_losses(rows)
    assert actual['anchor_exposure_sha256']==result['anchor_exposure_sha256']
    assert training.state_digest(other.snapshot())==training.state_digest(expected)
    altered=copy.deepcopy(saved);altered['recipe']['learning_rate']=1e-4
    with pytest.raises(ValueError,match='protocol'):other.restore(altered)


def test_partial_failure_keeps_completed_step_receipt_and_blocks_retry(tmp_path):
    trainer,_=make_trainer(artifact_dir=tmp_path);rows=pool();trainer.source_losses(rows)
    original=trainer._train_response;count=[0]
    def fail_third(*args,**kwargs):
        count[0]+=1
        if count[0]==3:raise RuntimeError('controlled partial failure')
        return original(*args,**kwargs)
    with patch.object(trainer,'_train_response',side_effect=fail_third):
        with pytest.raises(RuntimeError,match='controlled partial failure'):trainer.train_to_step(2,rows)
    assert trainer.optimizer_steps==1 and trainer.failed
    steps=[json.loads(v) for v in (tmp_path/'TRAIN_STEPS.jsonl').read_text().splitlines()]
    failure=json.loads((tmp_path/'FAILURES.jsonl').read_text().strip())
    assert len(steps)==1 and steps[0]['optimizer_steps_after']==1 and failure['optimizer_steps']==1
    with pytest.raises(ValueError,match='failed'):trainer.train_to_step(2,rows)
