"""CPU mathematical and causal-mask tests; these are not GPU qualification."""
import copy
import numpy as np
import pytest
import torch

from .artifacts import save_snapshot,load_snapshot
from .feedback import FrozenResponseObjective,sampled_features,author
from .projections import build_projections
from .core import BankConfig,CorrectionBank,FeedbackSummary
from .training import check_direct_protection

torch.set_num_threads(1)


def objective_fixture(seed=6):
    rng=np.random.default_rng(seed)
    E=torch.tensor(rng.normal(size=(19,7)),dtype=torch.float64)
    projections=build_projections(E,7,rank=3)
    h=torch.tensor(rng.normal(size=(8,7)),dtype=torch.float64)
    teacher=torch.tensor(rng.normal(size=(8,7)),dtype=torch.float64)
    omega=rng.uniform(size=8);omega/=omega.sum()
    obj=FrozenResponseObjective(h,teacher,omega,E,projections.U,projections.V_in,projection_chunk=3)
    return obj,projections


def test_compressed_teacher_loss_gradient_and_fisher_are_exact():
    obj,projections=objective_fixture()
    rng=np.random.default_rng(71)
    for C in (np.zeros((3,3)),rng.normal(size=(3,3))*.05):
        loss,gradient=obj.loss_gradient(C);direct,dg=obj.direct_loss_gradient(C)
        assert abs(loss-direct)<1e-12
        np.testing.assert_allclose(gradient,dg,atol=1e-12)
        step=rng.normal(size=(3,3))
        fd=(obj.loss_gradient(C+1e-6*step)[0]-obj.loss_gradient(C-1e-6*step)[0])/2e-6
        assert abs(fd-np.sum(gradient*step))<1e-8
    np.testing.assert_allclose(obj.G,obj.loss_gradient(np.zeros((3,3)))[1],atol=1e-12)
    C=rng.normal(size=(3,3));q=0.
    for w,phi,T in zip(obj.omega,obj.phi,obj.T):q+=w*(C@phi)@T@(C@phi)
    tiny=1e-4
    fd=(obj.loss_gradient(tiny*C)[0]+obj.loss_gradient(-tiny*C)[0])/tiny**2
    assert abs(fd-q)<1e-5


def test_direct_history_check_detects_actual_loss_failure_independent_of_claimed_slack():
    obj,p=objective_fixture()
    bank=CorrectionBank(BankConfig(kappa=p.kappa,rank=3))
    read=bank.read(np.array([1.,0.,0.]))
    summary=FeedbackSummary(read.descriptor,obj.G,obj.S,obj.omega,obj.phi,obj.T,
                            read.forecast,obj.G-read.forecast,'response1')
    bank.observe_and_write(summary,obj,read)
    checks=check_direct_protection(bank,{'response1':obj})
    assert checks['passed'] and checks['records']
    bank.matrices[0]*=-100
    assert not check_direct_protection(bank,{'response1':obj})['passed']


def test_sampled_features_use_author_sampler_and_only_new_response():
    torch.manual_seed(2)
    class Draft(torch.nn.Module):
        def __init__(self):
            super().__init__();self.fc=torch.nn.Linear(8,4,bias=False).double()
        def forward(self,*,position_ids,noise_embedding,target_hidden,attention_mask):
            # Retain the observed author attention mask for an independent check.
            self.last_mask=attention_mask.detach().clone()
            self.last_output=noise_embedding+self.fc(target_hidden[:, :noise_embedding.shape[1]])*0
            return self.last_output,None
    draft=Draft().eval().requires_grad_(False)
    embed=torch.nn.Embedding(32,4).double().eval().requires_grad_(False)
    # Enough full history for the fake forward's shape, without supervising it.
    response=dict(record_id='test:BANK:t2',episode_id='test',turn=2,target_id='T0',
                  target_identity_sha256='a'*64,capture_checkpoint_sha256='b'*64,
                  input_ids=[3]*200,output_ids=list(range(3,21)),input_token_count=200,output_cap=1024)
    def teacher(ids):
        x=embed(ids);return torch.cat((x,x),-1),x
    generator=torch.Generator().manual_seed(17)
    data=sampled_features(draft,embed,teacher,response,generator=generator,attention_backend='sdpa',mask_token_id=31)
    metadata=data['metadata']
    assert all(200<=p<218 for p in metadata['unique_supervised_position_ids'])
    assert data['omega'].sum()==pytest.approx(1.)
    n=len(metadata['anchor_positions']);length=227
    for block,anchor in enumerate(metadata['anchor_positions']):
        row=draft.last_mask[0,0,block*10]
        assert bool(row[:anchor].all())
        assert not bool(row[anchor:length].any())
        assert bool(row[length+block*10:length+(block+1)*10].all())
    independent=torch.Generator().manual_seed(17)
    mask=torch.tensor([[0]*200+[1]*18+[0]*9],dtype=torch.float32)
    anchors,keep=author.sample_anchor_positions(length,mask,generator=independent)
    assert anchors[0].tolist()==metadata['anchor_positions']
    assert np.array_equal(data['rng_after'],independent.get_state().numpy())
    assert all(tp==lp-1 for tps,lps in zip(metadata['teacher_positions'],metadata['label_positions']) for tp,lp in zip(tps,lps))


def test_snapshots_deduplicate_immutable_arrays_and_reject_tampering(tmp_path):
    x=np.arange(12,dtype=np.float64).reshape(3,4)
    first=save_snapshot({'fisher':x,'turn':1},tmp_path/'one.json')
    second=save_snapshot({'fisher':x,'turn':2},tmp_path/'two.json')
    assert len(list((tmp_path/'arrays').glob('*.npy')))==1
    x[0,0]=999
    loaded=load_snapshot(first)
    assert loaded['fisher'][0,0]==0 and not loaded['fisher'].flags.writeable
    assert load_snapshot(second)['turn']==2
    path=next((tmp_path/'arrays').glob('*.npy'))
    with path.open('ab') as f:f.write(b'bad')
    with pytest.raises(ValueError,match='changed'):load_snapshot(first)


def test_zero_exposure_is_constant_loss_without_resampling():
    E=torch.eye(4,dtype=torch.float64)
    obj=FrozenResponseObjective(torch.empty((0,4)),torch.empty((0,4)),np.array([]),E,
                                np.eye(4)[:,:2],np.eye(4)[:,:2])
    assert obj.loss_gradient(np.zeros((2,2)))[0]==0
    assert np.count_nonzero(obj.G)==0


def test_empty_factor_shapes_remain_serializable(tmp_path):
    data={'phi':np.empty((0,64)), 'T':np.empty((0,64,64)), 'omega':np.empty(0)}
    restored=load_snapshot(save_snapshot(data,tmp_path/'empty.json'))
    for key in data:
        assert restored[key].shape==data[key].shape


def test_teacher_uses_native_head_logits_with_explicit_probability_precision():
    torch.manual_seed(7)
    E=torch.randn(19,6).bfloat16()
    h=torch.randn(4,6).bfloat16();teacher=torch.randn(4,6).bfloat16()
    p=build_projections(E,descriptor_dimension=6,rank=3)
    objective=FrozenResponseObjective(h,teacher,np.full(4,.25),E,p.U,p.V_in)
    expected=torch.nn.functional.linear(teacher,E)
    assert torch.equal(objective._teacher_logits(teacher.double()),expected)
    probabilities=expected.double().softmax(-1)
    P=torch.tensor(p.P)
    phi=torch.tensor(objective.phi)
    M=((probabilities@P)*.25).T@phi
    np.testing.assert_allclose(objective.M,M.numpy(),atol=1e-12)
    C=np.ones((3,3))*.01
    direct,dg=objective.direct_loss_gradient(C);compressed,cg=objective.loss_gradient(C)
    assert abs(direct-compressed)<1e-12
    np.testing.assert_allclose(dg,cg,atol=1e-12)
    assert objective.teacher_precision['logits_dtype']=='torch.bfloat16'
    assert objective.teacher_precision['native_probability_max_difference']>0


def test_feedback_service_round_trip_keeps_direct_history_and_final_write_is_skipped(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from . import training
    from .training import BankTrainingService,plain
    obj,p=objective_fixture()
    service=BankTrainingService.__new__(BankTrainingService)
    service.config={'episode_id':'toy','target':{'identity_sha256':'a'*64}}
    service.components=SimpleNamespace(lm_head=SimpleNamespace(weight=obj.E),embed_tokens=None)
    service.projections=p;service.bank=CorrectionBank(BankConfig(kappa=p.kappa,rank=3))
    service.bank_directory=tmp_path;service.arrays=tmp_path/'arrays'
    service.bank_draft=None;service.teacher_features=None;service.bank_generator=None
    service._check_bank_frozen=lambda:None;service._synchronize=lambda:None
    service.last_turn=0;service.qualification_objectives={};service.qualification_replay={}
    service.snapshot_reference=save_snapshot(service.bank.snapshot(),tmp_path/'initial.json',array_root=service.arrays)
    features={'h':obj.h,'teacher_h':obj.teacher_h,'omega':obj.omega,'metadata':{'eligible_positions':8},
              'rng_before':np.array([17],dtype=np.uint8),'rng_after':np.array([18],dtype=np.uint8),
              'costs':{'teacher_preparation_s':0.,'frozen_feature_s':0.}}
    monkeypatch.setattr(training,'sampled_features',lambda *a,**kw:features)
    for turn in (1,2,3):
        read=service.bank.read(np.array([1.,0.,0.]))
        snapshot=service.snapshot_reference
        saved=plain({'alpha':read.alpha,'effective':read.effective,'forecast':read.forecast,
                     'descriptor':read.descriptor,'diagnostics':read.diagnostics,'snapshot_sha256':snapshot['sha256']})
        response={'record_id':f'toy:BANK:t{turn}','episode_id':'toy','turn':turn,
                  'target_identity_sha256':'a'*64,'input_ids':[1,2],'output_ids':[3,4]}
        before=service.bank.matrices.copy()
        result=service.feedback({'turn':turn,'final':turn==3,'qualification':True,
            'before_snapshot':snapshot,'response':response,'served_read':saved})
        assert result['scheduled_steps']==(0 if turn==3 else 2)
        checks=result['qualification_checks']
        assert checks['protection_records_immutable'] and checks['served_record_precedes_trials']
        assert checks['compressed_direct']['passed'] and checks['direct_historical']['passed']
        assert checks['compiled_fusion']['passed']
        precision=checks['native_precision']
        assert precision['status']=='measured' and not precision['performance_gate']
        assert {c['label'] for c in precision['cases']}==({'zero','served','controlled_nonzero'} if turn==3 else
                                                      {'zero','served','controlled_nonzero','fitted'})
        assert result['qualification_replay'] and load_snapshot(result['feedback'])['teacher_precision']
        assert result['costs']['qualification_only_s']>=result['costs']['diagnostic_reconstruction_s']
        if turn==3:np.testing.assert_array_equal(before,service.bank.matrices)
    assert len(service.qualification_replay)==3


def test_one_token_final_feedback_has_no_anchors_calls_or_rng_consumption():
    draft=torch.nn.Linear(7,7).eval().requires_grad_(False)
    embed=torch.nn.Embedding(19,7).eval().requires_grad_(False)
    generator=torch.Generator().manual_seed(17);before=generator.get_state().clone()
    response=dict(record_id='short:BANK:t20',episode_id='short',turn=20,target_id='T0',
        target_identity_sha256='a'*64,capture_checkpoint_sha256='b'*64,
        input_ids=[3]*20,output_ids=[128009],input_token_count=20,output_cap=1024)
    def forbidden(*args,**kwargs):raise AssertionError('No teacher/backbone work for empty final analysis')
    draft.forward=forbidden
    features=sampled_features(draft,embed,forbidden,response,generator=generator,final_analysis=True)
    assert features['h'].shape==(0,7) and features['omega'].size==0
    assert features['metadata']['analysis_only'] and features['metadata']['zero_supervision']
    assert features['metadata']['sampled_anchor_count']==0
    assert features['metadata']['eligible_positions']==0 and features['costs']['teacher_preparation_s']==0
    assert np.array_equal(features['rng_before'],features['rng_after'])
    assert torch.equal(before,generator.get_state())
    # Only final analysis is relaxed; ordinary training retains the author rule.
    with pytest.raises(ValueError,match='requires >=2'):
        sampled_features(draft,embed,forbidden,response,generator=generator)
    with pytest.raises(ValueError,match='requires >=2'):author.audit_response(response)
    with pytest.raises(ValueError,match='Token hash differs'):
        sampled_features(draft,embed,forbidden,{**response,'output_ids_sha256':'c'*64},
                         generator=generator,final_analysis=True)
    for malformed in ([],[128256],[-1],[True]):
        with pytest.raises(ValueError,match='Invalid empty-final token IDs'):
            sampled_features(draft,embed,forbidden,{**response,'output_ids':malformed},
                             generator=generator,final_analysis=True)
    with pytest.raises(ValueError,match='Invalid empty-final token IDs'):
        sampled_features(draft,embed,forbidden,{**response,'input_ids':[128256],'input_token_count':1},
                         generator=generator,final_analysis=True)


def test_one_token_final_analysis_completes_training_service_without_a_write(tmp_path):
    from types import SimpleNamespace
    from .training import BankTrainingService,plain
    obj,p=objective_fixture()
    service=BankTrainingService.__new__(BankTrainingService)
    service.config={'episode_id':'toy','target':{'identity_sha256':'a'*64}}
    embed=torch.nn.Embedding(19,7).double().eval().requires_grad_(False)
    service.components=SimpleNamespace(lm_head=SimpleNamespace(weight=obj.E),embed_tokens=embed)
    service.projections=p;service.bank=CorrectionBank(BankConfig(kappa=p.kappa,rank=3))
    service.bank_directory=tmp_path;service.arrays=tmp_path/'arrays'
    service.bank_draft=torch.nn.Linear(7,7).eval().requires_grad_(False)
    def forbidden(*args,**kwargs):raise AssertionError('No teacher replay for empty final analysis')
    service.teacher_features=forbidden;service.bank_generator=torch.Generator().manual_seed(17)
    service._check_bank_frozen=lambda:None;service._synchronize=lambda:None
    service.last_turn=19;service.qualification_objectives={};service.qualification_replay={}
    service.snapshot_reference=save_snapshot(service.bank.snapshot(),tmp_path/'initial.json',array_root=service.arrays)
    before=service.bank.matrices.copy();read=service.bank.read(np.array([1.,0.,0.]))
    snapshot=service.snapshot_reference
    saved=plain({'alpha':read.alpha,'effective':read.effective,'forecast':read.forecast,
        'descriptor':read.descriptor,'diagnostics':read.diagnostics,'snapshot_sha256':snapshot['sha256']})
    response=dict(record_id='toy:BANK:t20',episode_id='toy',turn=20,target_id='T0',
        target_identity_sha256='a'*64,capture_checkpoint_sha256='b'*64,
        input_ids=[1,2],output_ids=[128009],input_token_count=2,output_cap=1024)
    result=service.feedback({'turn':20,'final':True,'qualification':True,
        'before_snapshot':snapshot,'response':response,'served_read':saved})
    assert result['status']=='passed' and result['scheduled_steps']==result['accepted_steps']==0
    assert result['analysis_no_op'] and result['accounting']['eligible_positions']==0
    assert result['served_gain'] is None and result['owner'] is None
    np.testing.assert_array_equal(before,service.bank.matrices)
    assert not service.bank.protection and service.last_turn==20
    assert load_snapshot(result['feedback'])['summary']['phi'].shape==(0,3)
    assert result['qualification_checks']['native_precision']['sampled_positions']==0
    assert all(c['no_eligible_anchors'] for c in result['qualification_checks']['native_precision']['cases'])
