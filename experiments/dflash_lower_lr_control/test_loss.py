"""Small CPU tensors verify PR9/PR10 sampling contracts independently."""
import copy
from unittest.mock import patch
import pytest

torch=pytest.importorskip('torch')
from . import loss
from .test_training import original,new
from experiments.dflash_baseline_learning_curve.test_training import pool
from experiments.dflash_kl_native_diagnostic.test_loss import models,record
from experiments.dflash_kl_native_diagnostic import loss as old_loss

@pytest.fixture(autouse=True)
def one_thread():torch.set_num_threads(1)


def source_inputs(retained):
    rows=[];refs=[]
    for record,binding,ref in zip(pool(),retained['updates'][0]['accounting']['support_bindings'],
                                  retained['source_losses'][0]['response_losses']):
        rows.append(dict(record,group_id=record['episode_id'],request_id=record['record_id']))
        refs.append(dict(ref,request_id=record['record_id'],input_ids_sha256=binding['input_ids_sha256'],
                         output_ids_sha256=binding['output_ids_sha256']))
    return rows,refs


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_source_uses_exact_pr9_seeds_and_loss(dtype):
    retained=original(dtype);trainer=new(retained,dtype=dtype);rows,refs=source_inputs(retained)
    before=loss._state_hash(loss._rng_state());sampler=trainer.generator.get_state().clone()
    result=loss.evaluate_responses(trainer.draft,trainer.embed_tokens,trainer.lm_head,trainer.teacher_features,
        rows,refs,population='source',checkpoint_identity_sha256='e'*64,mask_token_id=19,
        attention_backend='sdpa',projection_chunk=7)
    for actual,reference in zip(result['response_losses'],refs):
        assert actual['loss']==reference['loss']
        assert actual['role']=='feedback' and actual['checkpoint_id']=='L8'
        assert actual['anchor_seed']==reference['anchor_seed']
        assert actual['anchor_rng_after_sha256']==reference['anchor_rng_after_sha256']
        assert actual['response_loss_mask']==[0]*actual['input_token_count']+[1]*actual['output_token_count']+[0]*9
    assert before==loss._state_hash(loss._rng_state()) and torch.equal(sampler,trainer.generator.get_state())


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_heldout_matches_pr10_without_role_changes_or_optimizer(dtype):
    parts=models(dtype);rows=[record()]
    reference=old_loss.evaluate_responses(*parts[:4],rows,checkpoint_id='F',checkpoint_identity_sha256='a'*64,
        mask_token_id=19,attention_backend='sdpa',projection_chunk=7)['response_losses']
    with patch.object(torch.optim,'AdamW',side_effect=AssertionError('no optimizer')), \
         patch.object(torch.Tensor,'backward',side_effect=AssertionError('no backward')):
        result=loss.evaluate_responses(*parts[:4],rows,reference,population='heldout',
            checkpoint_identity_sha256='e'*64,mask_token_id=19,attention_backend='sdpa',projection_chunk=7)
    actual=result['response_losses'][0]
    assert actual['loss']==reference[0]['loss'] and actual['role']=='evaluation'
    assert actual['anchor_seed']==reference[0]['anchor_seed']
    assert actual['anchor_rng_after_sha256']==reference[0]['anchor_rng_after_sha256']
    assert not result['optimizer_created'] and result['optimizer_steps']==0


def test_reference_sampling_mismatch_preserves_failed_row_rng_and_modes():
    parts=models();rows=[record()]
    refs=old_loss.evaluate_responses(*parts[:4],rows,checkpoint_id='F',checkpoint_identity_sha256='a'*64,
        mask_token_id=19,attention_backend='sdpa',projection_chunk=7)['response_losses']
    refs[0]['position_weights'][0][1]+=1
    before=loss._state_hash(loss._rng_state());modes=[m.training for root in parts[:3] for m in root.modules()]
    with pytest.raises(ValueError,match='position_weights') as err:
        loss.evaluate_responses(*parts[:4],rows,refs,population='heldout',checkpoint_identity_sha256='e'*64,
            mask_token_id=19,attention_backend='sdpa',projection_chunk=7)
    assert len(err.value.diagnostic_partial['response_losses'])==1
    assert err.value.diagnostic_partial['global_rng_restored']
    assert before==loss._state_hash(loss._rng_state())
    assert modes==[m.training for root in parts[:3] for m in root.modules()]


def test_missing_original_anchor_seed_rejected_before_teacher():
    parts=models();rows=[record()]
    refs=old_loss.evaluate_responses(*parts[:4],rows,checkpoint_id='F',checkpoint_identity_sha256='a'*64,
        mask_token_id=19,attention_backend='sdpa',projection_chunk=7)['response_losses']
    refs[0]['anchor_seed']+=1
    teacher_calls=len(parts[4])
    with pytest.raises(ValueError,match='seed differs'):
        loss.evaluate_responses(*parts[:4],rows,refs,population='heldout',checkpoint_identity_sha256='e'*64,
            mask_token_id=19,attention_backend='sdpa',projection_chunk=7)
    assert len(parts[4])==teacher_calls


@pytest.mark.parametrize('length',[0,1,2])
def test_short_or_zero_supervision_retained_without_resampling(length):
    parts=models();rows=[record(length=length)]
    refs=old_loss.evaluate_responses(*parts[:4],rows,checkpoint_id='F',checkpoint_identity_sha256='a'*64,
        mask_token_id=19,attention_backend='sdpa',projection_chunk=7)['response_losses']
    result=loss.evaluate_responses(*parts[:4],rows,refs,population='heldout',checkpoint_identity_sha256='e'*64,
        mask_token_id=19,attention_backend='sdpa',projection_chunk=7)
    assert result['response_losses'][0]['sampling_and_weights_sha256']==refs[0]['sampling_and_weights_sha256']
    assert result['response_losses'][0]['anchor_sampling_attempts']==int(length>=2)
