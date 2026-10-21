"""Synthetic CPU checks; no native/model campaign execution."""
import copy
import json
from unittest.mock import patch
import pytest

torch=pytest.importorskip('torch')
from . import training
from experiments.dflash_baseline_learning_curve.test_training import make_trainer,pool,RELEASED,TARGET,S2,S8,loss_at

@pytest.fixture(autouse=True)
def one_thread(): torch.set_num_threads(1)


def original(dtype=torch.float32,dropout=0.):
    old,_=make_trainer(dtype=dtype,dropout=dropout)
    rows=pool();s0=old.source_losses(rows)
    u1=old.train_to_step(2,rows);old.bind_checkpoint(1,S2);s2=old.source_losses(rows)
    u2=old.train_to_step(8,rows);old.bind_checkpoint(2,S8);s8=old.source_losses(rows)
    return dict(status='passed',updates=[dict(accounting=u1),dict(accounting=u2)],source_losses=[s0,s2,s8])


def new(retained,artifact_dir=None,dtype=torch.float32,dropout=0.):
    base,_=make_trainer(dtype=dtype,dropout=dropout)
    return training.LowerLRTrainer(base.draft,base.embed_tokens,base.lm_head,base.teacher_features,
        episode_id='group',target_id='target',target_identity_sha256=TARGET,
        released_checkpoint_sha256=RELEASED,mask_token_id=19,attention_backend='sdpa',projection_chunk=7,
        retained_training=retained,artifact_dir=artifact_dir)


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_eight_steps_match_manual_qualified_pair_loss_and_adam(dtype,tmp_path):
    retained=original(dtype);trainer=new(retained,tmp_path,dtype);control=new(retained,dtype=dtype)
    assert not trainer.optimizer.state and trainer.optimizer.param_groups[0]['lr']==3e-5
    control.draft.train()
    rows=pool()
    for step in range(8):
        control.optimizer.zero_grad(set_to_none=True)
        for row in rows[(step%2)*2:(step%2)*2+2]:
            loss,_=loss_at(control,row,control.generator)
            (loss/2).backward()
        torch.nn.utils.clip_grad_norm_(list(control.parameters.values()),1.,error_if_nonfinite=True)
        control.optimizer.step();control.optimizer.zero_grad(set_to_none=True)
        control.optimizer_steps+=1
    actual=trainer.train_to_step8(rows)
    for name,p in trainer.parameters.items():
        assert torch.equal(p,control.parameters[name])
    assert training.curve.state_digest(trainer.optimizer.state_dict())==training.curve.state_digest(control.optimizer.state_dict())
    assert actual['optimizer_steps_total']==8 and actual['response_presentations_total']==16
    assert actual['response_order']==[r['record_id'] for r in rows]*4
    assert all(b['effective_learning_rate']==3e-5 and b['original_exposure_match'] for b in actual['batches'])
    assert actual['source_diagnostic_steps']==[] and trainer.source_anchor_specs is None
    assert len((tmp_path/'TRAIN_STEPS.jsonl').read_text().splitlines())==8
    assert len((tmp_path/'ANCHOR_CHECKS.jsonl').read_text().splitlines())==8
    assert all(p.grad is None for p in trainer.parameters.values())
    trainer.bind_l8_checkpoint('e'*64)
    assert trainer.active_checkpoint_sha256=='e'*64 and not trainer.awaiting_checkpoint
    with pytest.raises(ValueError,match='bound once'):trainer.bind_l8_checkpoint('e'*64)
    with pytest.raises(ValueError,match='start once'):trainer.train_to_step8(rows)


def test_mismatch_is_durable_before_optimizer_step(tmp_path):
    retained=original();trainer=new(retained,tmp_path);real=trainer._train_response
    def altered(*a,**k):
        row=real(*a,**k);row['eligible_positions']+=1;return row
    with patch.object(trainer,'_train_response',side_effect=altered):
        with pytest.raises(ValueError,match='anchors/exposure') as failure:trainer.train_to_step8(pool())
    assert trainer.failed and trainer.optimizer_steps==0 and not trainer.optimizer.state
    receipt=json.loads((tmp_path/'ANCHOR_CHECKS.jsonl').read_text())
    assert receipt['status']=='failed' and any(e['field']=='eligible_positions' for e in receipt['errors'])
    assert failure.value.training_partial['optimizer_steps']==0
    assert (tmp_path/'FAILURES.jsonl').exists()


def test_heldout_and_changed_feedback_are_rejected_before_learning():
    retained=original()
    for change in ('role','tokens','order'):
        trainer=new(retained);rows=pool()
        if change=='role':rows[0]['role']='evaluation'
        if change=='tokens':rows[0]['output_ids'][0]+=1
        if change=='order':rows.reverse()
        with pytest.raises(ValueError):trainer.train_to_step8(rows)
        assert trainer.optimizer_steps==0 and not trainer.optimizer.state


def test_historical_guards_and_recipe_untouched():
    retained=original();new(retained)
    old,_=make_trainer()
    assert old.optimizer.param_groups[0]['lr']==3e-4
    with pytest.raises(ValueError,match='progression'):old.train_to_step(8,pool())
    with pytest.raises(ValueError,match='Measure fixed-anchor'):old.train_to_step(2,pool())
    for key,value in training.curve.RECIPE.items():
        assert training.RECIPE[key]==(3e-5 if key=='learning_rate' else value)


def test_reference_diagnostic_binding_must_precede_training():
    retained=original();retained['source_losses'][1]['source_anchor_specs'][0]['seed']+=1
    with pytest.raises(ValueError,match='diagnostic anchors'):new(retained)


def test_no_optimizer_resume_or_intermediate_export():
    trainer=new(original())
    for method in ('restore','snapshot','source_losses','train_to_step','bind_checkpoint'):
        with pytest.raises(ValueError):getattr(trainer,method)({})


def test_partial_step_distinguishes_backwards_from_optimizer_exposure(tmp_path):
    trainer=new(original(),tmp_path);real=trainer._train_response;count=0
    def fail_second(*a,**k):
        nonlocal count
        count+=1
        if count==2:raise RuntimeError('second response fails before completed backward')
        return real(*a,**k)
    with patch.object(trainer,'_train_response',side_effect=fail_second):
        with pytest.raises(RuntimeError) as err:trainer.train_to_step8(pool())
    partial=err.value.training_partial
    assert partial['attempted_response_calls']==2
    assert partial['completed_backward_presentations']==partial['uncommitted_backward_presentations']==1
    assert partial['optimizer_steps']==partial['response_presentations']==0
    assert len(partial['pending_presentations'])==1


def test_actual_training_loss_receipts_pass_controller_validation():
    from . import loss,experiment
    from experiments.dflash_kl_native_diagnostic import loss as historical_loss
    retained=original();trainer=new(retained);rows=pool();account=trainer.train_to_step8(rows)
    files={'model.safetensors':dict(bytes=2,sha256='e'*64)}
    identity=training.digest(files)
    trainer.bind_l8_checkpoint(identity)
    source=[];source_refs=[];heldout=[];heldout_refs=[]
    for row,binding,ref in zip(rows,retained['updates'][0]['accounting']['support_bindings'],
                                retained['source_losses'][0]['response_losses']):
        source.append(dict(row,group_id='group',request_id=row['record_id']))
        source_refs.append(dict(ref,request_id=row['record_id'],input_ids_sha256=binding['input_ids_sha256'],
                               output_ids_sha256=binding['output_ids_sha256']))
        heldout.append(dict(row,group_id='group',request_id=row['record_id']+'_heldout',
                            record_id=row['record_id']+'_heldout',role='evaluation'))
    args=(trainer.draft,trainer.embed_tokens,trainer.lm_head,trainer.teacher_features)
    heldout_refs=historical_loss.evaluate_responses(*args,heldout,checkpoint_id='F',
        checkpoint_identity_sha256=RELEASED,mask_token_id=19,attention_backend='sdpa',projection_chunk=7)['response_losses']
    values={}
    for population,records,refs in [('source',source,source_refs),('heldout',heldout,heldout_refs)]:
        values[population]=loss.evaluate_responses(*args,records,refs,population=population,
            checkpoint_identity_sha256=identity,mask_token_id=19,attention_backend='sdpa',projection_chunk=7)['response_losses']
    group=dict(group_id='group',training=retained,support=rows,
        checkpoints={'F':dict(checkpoint_identity_sha256=RELEASED)},reference_pairs={
            'source':[{'F':dict(native=record,kl=ref)} for record,ref in zip(source,source_refs)],
            'heldout':[{'F':dict(native=record,kl=ref)} for record,ref in zip(heldout,heldout_refs)]})
    receipt=dict(status='passed',checkpoint=dict(checkpoint_id='L8',group_id='group',optimizer_steps=8,
        learning_rate=3e-5,checkpoint_identity_sha256=identity,files=files),accounting=account,losses=values)
    # Actual worker RPC is JSON, which converts recipe betas tuples to arrays.
    group,receipt=json.loads(json.dumps([group,receipt]))
    assert experiment.validate_training(receipt,group)['status']=='passed'
    receipt['accounting']['batches'][0]['effective_learning_rate']=3e-4
    with pytest.raises(ValueError,match='LR'):experiment.validate_training(receipt,group)
