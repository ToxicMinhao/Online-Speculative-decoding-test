"""Retained-anchor integrity tests; native reconstruction tests use optional Torch."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from .feedback import author
from .retained_precision import (retained_sample_plan, reconstruct_retained_features,
                                reconstruct_retained_precision, compare_retained_objective)


def _rehash(metadata):
    metadata['sampling_and_weights_sha256'] = author.digest(
        {k:v for k,v in metadata.items() if k != 'sampling_and_weights_sha256'})


def _feedback():
    response = dict(record_id='wildchat_00:BANK:t1', episode_id='wildchat_00', turn=1,
                    target_id='T0', target_identity_sha256='a'*64,
                    capture_checkpoint_sha256='b'*64, input_ids=[2,3,4], output_ids=[5,6,7,8],
                    input_token_count=3, output_cap=1024)
    audit = author.audit_response(response)
    anchors = np.array([3,4,5]); labels = anchors[:, None] + np.arange(10)[None, :]
    binary = ((np.arange(10)[None, :] > 0) & (labels < 7)).astype(np.float32)
    weights = binary * np.exp(-np.maximum(np.arange(10)-1,0).astype(np.float32)/7)
    chosen = weights.reshape(-1) > 0; raw = weights.reshape(-1)[chosen].astype(np.float64)
    supervised = labels.reshape(-1)[chosen].tolist()
    metadata = dict(audit, anchor_positions=anchors.tolist(), block_keep_mask=[True]*3,
        label_positions=labels.tolist(), teacher_positions=(labels-1).tolist(),
        binary_loss_mask=binary.tolist(), position_weights=weights.tolist(),
        eligible_positions=len(supervised), unique_supervised_position_ids=sorted(set(supervised)),
        unique_supervised_positions=len(set(supervised)), zero_supervision=False,
        eligible_blocks=3, weighted_position_sum=float(raw.sum()))
    _rehash(metadata)
    return dict(response=response, summary=dict(response_id=response['record_id'], metadata=metadata,
                omega=raw/raw.sum()), rng_before=np.array([17],dtype=np.uint8),
                rng_after=np.array([18],dtype=np.uint8))


def test_original_anchor_positions_and_weights_used_without_sampler(monkeypatch):
    monkeypatch.setattr(author, 'sample_anchor_positions', lambda *a,**k:pytest.fail('must never sample'))
    feedback = _feedback(); plan = retained_sample_plan(feedback)
    np.testing.assert_array_equal(plan['anchors'],[3,4,5])
    np.testing.assert_array_equal(plan['omega'],feedback['summary']['omega'])
    assert all(3 <= p < 7 for p in plan['labels'].reshape(-1)[plan['selected']])
    assert plan['method'].endswith('no_sampler_no_rng_restoration')


def test_hash_mismatch_rejected_before_reconstruction():
    feedback = _feedback(); feedback['summary']['metadata']['position_weights'][0][1] = .8
    with pytest.raises(ValueError,match='metadata hash'):
        retained_sample_plan(feedback)


@pytest.mark.parametrize('field', ['input_ids','output_ids'])
def test_saved_token_change_cannot_silently_change_conditioning(field):
    feedback = _feedback(); feedback['response'][field][0] += 1
    with pytest.raises(ValueError,match='response audit'):
        retained_sample_plan(feedback)


def test_changed_history_mask_rejected_even_with_rehashed_metadata():
    feedback = _feedback(); metadata = feedback['summary']['metadata']
    metadata['binary_loss_mask'][0][0]=1; _rehash(metadata)
    with pytest.raises(ValueError,match='position/mask'):
        retained_sample_plan(feedback)


def test_new_anchor_cannot_be_substituted():
    feedback = _feedback(); metadata = feedback['summary']['metadata']
    metadata['anchor_positions'][0]=2; _rehash(metadata)
    with pytest.raises(ValueError,match='outside newest response'):
        retained_sample_plan(feedback)


def test_normalized_weights_must_be_exact_original():
    feedback = _feedback(); feedback['summary']['omega'][0] += 1e-12
    with pytest.raises(ValueError,match='normalized supervision'):
        retained_sample_plan(feedback)


def test_teacher_prediction_position_off_by_one_rejected():
    feedback = _feedback(); metadata = feedback['summary']['metadata']
    metadata['teacher_positions'][0][1] += 1; _rehash(metadata)
    with pytest.raises(ValueError,match='teacher_positions'):
        retained_sample_plan(feedback)


def test_full_compressed_statistic_check_detects_wrong_source_features():
    values = dict(omega=np.array([1.]),phi=np.ones((1,2)),G=np.ones((2,2)),S=np.eye(2),
                  T=np.eye(2)[None,:,:],M=np.ones((2,2)))
    feedback = dict(summary={k:v.copy() for k,v in values.items() if k!='M'},M=values['M'].copy())
    objective = SimpleNamespace(**values)
    assert compare_retained_objective(objective,feedback)['passed']
    objective.T = objective.T + .001
    check=compare_retained_objective(objective,feedback)
    assert not check['passed'] and not check['statistics']['T']['passed']
    assert check['statistics']['G']['passed']


def _native_fixture():
    torch = pytest.importorskip('torch')
    from .feedback import sampled_features, FrozenResponseObjective
    from .projections import build_projections
    torch.manual_seed(2)
    class Draft(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.fc=torch.nn.Linear(8,4,bias=False).double(); self.calls=0
        def forward(self,*,position_ids,noise_embedding,target_hidden,attention_mask):
            self.calls+=1; self.last_mask=attention_mask.clone()
            # An actual deterministic nonzero feature dependence, without a
            # large architecture: attention structure is checked independently.
            return noise_embedding+self.fc(target_hidden[:,:1]).expand_as(noise_embedding),None
    draft=Draft().eval().requires_grad_(False)
    embed=torch.nn.Embedding(32,4).double().eval().requires_grad_(False)
    response=dict(record_id='wildchat_00:BANK:t2',episode_id='wildchat_00',turn=2,target_id='T0',
        target_identity_sha256='a'*64,capture_checkpoint_sha256='b'*64,
        input_ids=[3]*25,output_ids=list(range(3,21)),input_token_count=25,output_cap=1024)
    counter={'calls':0}
    def teacher(ids):
        counter['calls']+=1; x=embed(ids); return torch.cat((x,x),-1),x
    original=sampled_features(draft,embed,teacher,response,generator=torch.Generator().manual_seed(17),
        mask_token_id=31,attention_backend='sdpa')
    E=torch.randn(32,4,dtype=torch.float64)
    projections=build_projections(E,4,rank=2)
    obj=FrozenResponseObjective(original['h'],original['teacher_h'],original['omega'],E,
                                projections.U,projections.V_in)
    feedback=dict(response=response,summary=dict(response_id=response['record_id'],metadata=original['metadata'],
        **{key:getattr(obj,key) for key in ('omega','phi','G','S','T')}),M=obj.M,
        rng_before=original['rng_before'],rng_after=original['rng_after'])
    return torch,draft,embed,teacher,counter,original,E,projections,feedback


def test_reconstructs_original_features_once_no_sampler_no_rng_or_parameter_change(monkeypatch):
    torch,draft,embed,teacher,counter,original,E,p,feedback=_native_fixture()
    monkeypatch.setattr(author,'sample_anchor_positions',lambda *a,**k:pytest.fail('resampling forbidden'))
    before_rng=torch.get_rng_state().clone()
    before_params=[p.detach().clone() for p in draft.parameters()]
    before_draft,before_teacher=draft.calls,counter['calls']
    actual=reconstruct_retained_features(draft,embed,teacher,feedback,mask_token_id=31,attention_backend='sdpa')
    assert draft.calls-before_draft==counter['calls']-before_teacher==1
    assert torch.equal(actual['h'],original['h']) and torch.equal(actual['teacher_h'],original['teacher_h'])
    assert torch.equal(before_rng,torch.get_rng_state())
    assert all(torch.equal(old,new) for old,new in zip(before_params,draft.parameters()))
    length=25+18+9
    for block,anchor in enumerate(feedback['summary']['metadata']['anchor_positions']):
        row=draft.last_mask[0,0,block*10]
        assert row[:anchor].all() and not row[anchor:length].any()
        assert row[length+block*10:length+(block+1)*10].all()


def test_precision_only_for_matching_retained_objective(monkeypatch):
    torch,draft,embed,teacher,counter,original,E,p,feedback=_native_fixture()
    monkeypatch.setattr(author,'sample_anchor_positions',lambda *a,**k:pytest.fail('resampling forbidden'))
    result=reconstruct_retained_precision(draft,embed,teacher,E,p,feedback,np.eye(2)*.01,
                                         mask_token_id=31,attention_backend='sdpa')
    assert result['status']=='measured' and result['retained_objective_check']['passed']
    assert {row['label'] for row in result['precision']['cases']}=={'zero','selected'}
    assert result['reconstruction']['sampler_invocations']==0
    changed=deepcopy(feedback);changed['summary']['G']=changed['summary']['G']+.01
    result=reconstruct_retained_precision(draft,embed,teacher,E,p,changed,np.eye(2)*.01,
                                         mask_token_id=31,attention_backend='sdpa')
    assert result['status']=='unavailable' and result['precision'] is None
    assert not result['retained_objective_check']['statistics']['G']['passed']
