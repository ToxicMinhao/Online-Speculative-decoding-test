"""Bounded controller tests with real triplet/native validation, no model execution."""
import copy

import pytest

from . import experiment as ex
from experiments.dflash_kl_native_diagnostic.test_experiment import native_row, loss_row


def groups_fixture():
    groups=[]
    for number in range(6):
        group=dict(group_id=f'g{number}',domain='test',reference_pairs={})
        for population,role in [('source','feedback'),('heldout','evaluation')]:
            group['reference_pairs'][population]=[]
            for index in range(4):
                native={label:native_row(group['group_id'],role,index,label) for label in ('F','S8')}
                value=dict(group_id=group['group_id'],domain='test',population=population,request_id=native['F']['request_id'],
                    checkpoint_bindings={'F':'f'*64,'S8':'8'*64},validation={'status':'passed','errors':[]})
                for label in ('F','S8'):
                    kl=loss_row(native[label]);kl.update(anchor_rng_after_sha256='a'*64,role=role)
                    value[label]=dict(native=native[label],kl=kl)
                group['reference_pairs'][population].append(value)
        groups.append(group)
    return groups


class FakeBackend:
    def __init__(self, corrupt=None, harmful=True):
        self.corrupt,self.harmful=corrupt,harmful
        self.calls=[];self.stopped=False;self.closed=False
    def start_group(self,group):
        self.group=group;self.calls.append('start');return {'status':'passed'}
    def train_group(self):
        self.calls.append('train')
        if self.corrupt=='partial_training':
            error=ValueError('training failed before complete accounting')
            error.diagnostic_partial=dict(status='failed',accounting=None,losses={'source':[],'heldout':[]})
            raise error
        losses={population:[] for population in ('source','heldout')}
        for population in losses:
            for reference in self.group['reference_pairs'][population]:
                loss=copy.deepcopy(reference['F']['kl'])
                loss.update(checkpoint_id='L8',checkpoint_identity_sha256='c'*64)
                losses[population].append(loss)
        if self.corrupt=='anchor':losses['source'][0]['anchor_seed']+=1
        if self.corrupt=='nonfinite':losses['source'][0]['loss']=float('nan')
        return dict(status='passed',checkpoint={'checkpoint_id':'L8','checkpoint_identity_sha256':'c'*64},
            accounting={'optimizer_steps_total':8,'response_presentations_total':16},losses=losses,
            costs={'training_s':2.,'teacher_preparation_s':.5})
    def stop_training(self):
        self.calls.append('stop');self.stopped=True
        return dict(status='failed' if self.corrupt=='quiescence' else 'passed',training_quiescent=self.corrupt!='quiescence')
    def serve_request(self,record,population):
        assert self.stopped,'Training must terminate before native evaluation'
        self.calls.append(('serve',population,'L8',record['request_id']))
        value=copy.deepcopy(record)
        value.update(checkpoint_id='L8',checkpoint_identity_sha256='c'*64)
        if self.harmful:
            value.update(verification_calls=2,accepted_draft_tokens=0,committed_progress_per_call=1.,
                emitted_tokens_per_verification_call=1.5,rounds=[dict(round_index=i,output_offset=i+1,
                    accepted_prefix_length=0,accepted_draft_tokens=0,proposed_tokens=2,committed_tokens=1,
                    committed_token_ids=[token],first_rejection_offset=0,stop_clipped=False)
                    for i,token in enumerate(record['output_ids'][1:])])
        if self.corrupt=='output':value['output_ids'][0]+=1
        if self.corrupt=='termination':value['finish_reason']={'type':'length'}
        return value
    def finish_group(self):
        self.calls.append('finish');self.closed=True
        return dict(status='passed',startup_s=.2,shutdown_s=.1,activation_s=.3,native_activation_s=.2)


@pytest.fixture(autouse=True)
def orchestration_training_gate(monkeypatch):
    # Trainer mathematics and exposure checks have separate qualified toy tests.
    # Keep real native/triplet validation for the controller's sequencing tests.
    monkeypatch.setattr(ex,'validate_training',lambda receipt,group:{'status':'passed','errors':[]})


def test_all_six_continue_even_when_every_new_native_result_is_harmful():
    backends=[]
    def factory(_):
        value=FakeBackend();backends.append(value);return value
    result=ex.run_campaign(groups_fixture(),factory)
    assert result['status']=='complete' and result['unexecuted_groups']==[]
    assert result['first_group_qualification']['status']=='passed'
    assert (result['optimizer_steps'],result['training_presentations'],result['actual_native_generations'],
            result['actual_response_kl_evaluations'],result['explicit_warmups'])==(48,96,48,48,0)
    assert all(value.closed for value in backends)
    for group in result['groups']:
        for row in group['source']+group['heldout']:
            assert row['validation']['status']=='passed'
            assert row['L8']['native']['verification_calls']>row['F']['native']['verification_calls']


def test_training_terminates_before_serving_and_controls_are_never_regenerated():
    backends=[]
    def factory(_):
        value=FakeBackend(harmful=False);backends.append(value);return value
    result=ex.run_campaign(groups_fixture(),factory)
    assert result['status']=='complete'
    for value in backends:
        assert value.calls[:3]==['start','train','stop']
        serves=[row for row in value.calls if isinstance(row,tuple)]
        assert len(serves)==8 and all(row[2]=='L8' for row in serves)
        assert [row[1] for row in serves]==['source']*4+['heldout']*4
    assert all(not hasattr(backend,'serve_frozen') for backend in backends)


@pytest.mark.parametrize('corrupt',['output','termination','anchor','nonfinite'])
def test_invalid_first_group_preserves_raw_snapshots_and_stops(corrupt):
    backend=FakeBackend(corrupt);snapshots=[];events=[]
    result=ex.run_campaign(groups_fixture(),lambda _:backend,
        snapshot=lambda value:snapshots.append(copy.deepcopy(value)),emit=lambda *value:events.append(value))
    assert result['status']=='failed' and len(result['unexecuted_groups'])==5
    group=result['groups'][0]
    assert group['status']=='failed' and group['native_raw'] and group['source'] and backend.closed
    assert group['source'][0]['validation']['status']=='failed'
    assert result['actual_native_generations']==1
    assert snapshots and events and any(row[0]=='native_raw' for row in events)
    if corrupt=='nonfinite':
        assert group['training']['losses']['source'][0]['loss']['nonfinite_float']=='nan'


def test_unconfirmed_quiescence_prevents_all_native_generation():
    backend=FakeBackend('quiescence')
    result=ex.run_campaign(groups_fixture(),lambda _:backend)
    assert result['status']=='failed' and result['actual_native_generations']==0
    assert not any(isinstance(row,tuple) for row in backend.calls)
    assert backend.closed


def test_partial_none_accounting_does_not_obscure_original_failure():
    backend=FakeBackend('partial_training');snapshots=[]
    result=ex.run_campaign(groups_fixture(),lambda _:backend,snapshot=lambda value:snapshots.append(copy.deepcopy(value)))
    assert result['status']=='failed' and result['groups'][0]['work_incomplete'] is True
    assert result['groups'][0]['training']['accounting'] is None
    assert 'training failed before complete accounting' in result['groups'][0]['error']['message']
    assert result['actual_native_generations']==0 and result['actual_response_kl_evaluations']==0
    assert len(result['unexecuted_groups'])==5 and snapshots and backend.closed


def test_cost_receipts_and_reporter_round_trip():
    from .report import summarize
    result=ex.run_campaign(groups_fixture(),lambda _:FakeBackend())
    report=summarize(result)
    assert report['status']=='complete' and len(report['comparisons'])==48
    assert report['accounting_errors']==[]
    assert report['costs'][0]['training_components']=={'training_s':2.,'teacher_preparation_s':.5}
    assert report['costs'][0]['components']['activation_s']==.3
    assert report['costs'][0]['components']['native_activation_s']==.2
    pooled=next(row for row in report['pooled'] if row['scope']=='overall' and row['population']=='heldout')
    assert pooled['L8_vs_F_native_worse_requests']==24
    assert pooled['L8_vs_F_verification_call_reduction_percent']==-100.
