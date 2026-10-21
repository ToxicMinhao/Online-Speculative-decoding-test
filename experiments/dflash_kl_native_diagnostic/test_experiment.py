"""Small synthetic evidence, never real prompts/answers, exercises the protocol."""
import copy
import math

import pytest

from . import experiment as ex


def native_row(group='g0', role='feedback', index=0, label='F'):
    rid = f'{group}_{role}_{index}'
    inp, out = [10, 11], [20, 21, 128009]
    return dict(status='passed', group_id=group, domain='test', request_id=rid, role=role,
        checkpoint_id=label, checkpoint_identity_sha256=('f' if label == 'F' else '8') * 64,
        input_ids=inp, output_ids=out, input_ids_sha256=ex.digest(inp), output_tokens=3,
        cap_hit=False, eos_hit=True, finish_reason={'type':'stop', 'matched':128009},
        capture=False, feedback=None, capture_costs=None, request_s=.1,
        verification_calls=1, accepted_draft_tokens=1, verification_committed_tokens=2,
        initial_output_tokens=1, committed_progress_per_call=2., emitted_tokens_per_verification_call=3.,
        rounds=[dict(round_index=0, output_offset=1, accepted_prefix_length=1,
            accepted_draft_tokens=1, proposed_tokens=2, committed_tokens=2,
            committed_token_ids=[21, 128009], first_rejection_offset=1, stop_clipped=False)])


def loss_row(native, loss=1.):
    weights = [0., 1., math.exp(-1/7)] + [0.]*7
    meta = dict(anchor_positions=[2], block_keep_mask=[True], label_positions=[list(range(2,12))],
        teacher_positions=[list(range(1,11))], binary_loss_mask=[[0.,1.,1.]+[0.]*7], position_weights=[weights],
        sampled_blocks=1, eligible_blocks=1, eligible_positions=2, unique_supervised_positions=2,
        unique_supervised_position_ids=[3,4], weighted_position_sum=sum(weights),
        normalization_denominator=sum(weights)+1e-6, excluded_anchor_positions=1, zero_supervision=False,
        backbone_forward_calls=1, head_projection_chunks=1)
    return dict(**meta, sampling_and_weights_sha256=ex.digest(meta), anchor_seed=17,
        loss=loss, record_id=native['request_id'], request_id=native['request_id'],
        checkpoint_id=native['checkpoint_id'], checkpoint_identity_sha256=native['checkpoint_identity_sha256'],
        input_ids_sha256=ex.digest(native['input_ids']), output_ids_sha256=ex.digest(native['output_ids']),
        diagnostic_model_mode='eval_no_grad', training_rng_preserved=True)


def pair_fixture(population='source'):
    role = 'feedback' if population == 'source' else 'evaluation'
    f, s = native_row(role=role), native_row(role=role, label='S8')
    return dict(group_id='g0', domain='test', population=population, request_id=f['request_id'],
        checkpoint_bindings={'F':'f'*64, 'S8':'8'*64}, F=dict(native=f, kl=loss_row(f)),
        S8=dict(native=s, kl=loss_row(s,.2)), validation={'status':'passed','errors':[]})


def groups_fixture():
    groups=[]
    for i in range(6):
        name=f'g{i}'
        feedback=[native_row(name,index=j) for j in range(4)]
        scores=[native_row(name,'evaluation',j,label) for j in range(4) for label in ('F','S8')]
        sources=[]
        for label in ('F','S8'):
            rows=[loss_row({**r,'checkpoint_id':label,'checkpoint_identity_sha256':('f' if label=='F' else '8')*64}) for r in feedback]
            sources.append(dict(checkpoint_id=label, checkpoint_sha256=('f' if label=='F' else '8')*64,
                training_rng_preserved=True, diagnostic_model_mode='eval_no_grad', response_losses=rows))
        groups.append(dict(group_id=name,domain='test',feedback=feedback,scores=scores,
            evaluation_requests=[{'request_id':r['request_id']} for r in scores if r['checkpoint_id']=='F'],
            checkpoints={k:{'checkpoint_identity_sha256':v*64} for k,v in [('F','f'),('S8','8')]},
            training={'source_losses':sources}))
    return groups


class FakeBackend:
    def __init__(self, corrupt=None): self.corrupt=corrupt; self.closed=False
    def start_group(self, group): self.group=group;return {'status':'passed'}
    def serve_source(self, record):
        row={**record,'checkpoint_id':'S8','checkpoint_identity_sha256':'8'*64}
        if self.corrupt=='output':row['output_ids']=[20,22,128009]
        if self.corrupt=='termination':row['finish_reason']={'type':'length'}
        return row
    def evaluate_heldout(self, records, label):
        rows=[loss_row({**r,'checkpoint_id':label,'checkpoint_identity_sha256':('f' if label=='F' else '8')*64}) for r in records]
        if self.corrupt=='anchor' and label=='S8':rows[0]['anchor_seed']=18
        if self.corrupt=='nonfinite':rows[0]['loss']=float('nan')
        return dict(status='passed',checkpoint_id=label,checkpoint_identity_sha256=('f' if label=='F' else '8')*64,response_losses=rows)
    def finish_group(self):self.closed=True;return dict(status='passed',costs={'diagnostic_s':1.})


def test_all_groups_continue_without_performance_gate():
    r=ex.run_campaign(groups_fixture(),lambda _:FakeBackend())
    assert r['status']=='complete'
    assert r['actual_source_generations']==24 and r['actual_response_kl_evaluations']==48
    assert r['optimizer_steps']==0 and r['explicit_warmups']==0
    assert r['first_group_qualification']['status']=='passed'


@pytest.mark.parametrize('corrupt', ['output','termination','anchor','nonfinite'])
def test_failure_preserves_raw_and_stops(corrupt):
    snapshots=[];backends=[]
    def factory(_):
        b=FakeBackend(corrupt);backends.append(b);return b
    result=ex.run_campaign(groups_fixture(),factory,snapshot=lambda r:snapshots.append(copy.deepcopy(r)))
    assert result['status']=='failed' and len(result['unexecuted_groups'])==5
    assert result['groups'][0]['source_raw'] and backends[0].closed
    assert snapshots and 'error' in result['groups'][0]
    if corrupt=='nonfinite':
        assert result['groups'][0]['heldout_loss_raw'][0]['response_losses'][0]['loss']['nonfinite_float']=='nan'


@pytest.mark.parametrize('population',['source','heldout'])
def test_pair_accepts_exact_control(population):
    assert ex.validate_pair(pair_fixture(population))['status']=='passed'


@pytest.mark.parametrize('change', ['checkpoint','denominator','weight','mask','native_counter','foreign_loss','rng'])
def test_pair_rejects_bad_evidence(change):
    p=pair_fixture('heldout');loss=p['S8']['kl'];native=p['S8']['native']
    if change=='checkpoint':loss['checkpoint_identity_sha256']='x'*64
    if change=='denominator':loss['normalization_denominator']+=1
    if change=='weight':loss['position_weights'][0][1]=.5
    if change=='mask':loss['binary_loss_mask'][0][0]=1
    if change=='native_counter':native['accepted_draft_tokens']=9
    if change=='foreign_loss':loss['unique_supervised_position_ids']=[0,1]
    if change=='rng':loss['training_rng_preserved']=False
    assert ex.validate_pair(p)['status']=='failed'


def test_no_s2_and_no_update_api():
    assert not hasattr(FakeBackend(),'train')
    r=ex.run_campaign(groups_fixture(),lambda _:FakeBackend())
    assert all(set(p['checkpoint_bindings'])=={'F','S8'} for g in r['groups'] for p in g['source']+g['heldout'])


def test_group_costs_preserve_nested_receipts_and_wall():
    class CostBackend(FakeBackend):
        def finish_group(self):
            return dict(status='passed',startup_s=2.,shutdown_s=1.,activation_s=3.,native_activation_s=1.)
        def evaluate_heldout(self, records, label):
            v=super().evaluate_heldout(records,label)
            v['costs']={'checkpoint_loading_s':.1}
            v['diagnostic_wall_s']=.5
            for row in v['response_losses']:row['phase_seconds']={'teacher_preparation_s':.03,'forward_and_loss_s':.04}
            return v
    r=ex.run_campaign(groups_fixture(),lambda _:CostBackend())
    cost=r['groups'][0]['costs']
    assert cost['activation_s']==3 and cost['native_activation_s']==1
    assert cost['source_replay_request_s']==.4 and cost['actual_group_wall_s']>0
    assert len(cost['heldout_loss_receipts'])==2
    assert cost['heldout_loss_receipts'][0]['response_phase_seconds'][0]['teacher_preparation_s']==.03


def test_partial_failed_kl_work_is_preserved_and_counted():
    class PartialBackend(FakeBackend):
        def evaluate_heldout(self, records, label):
            v=super().evaluate_heldout(records,label)
            error=ValueError('third response failed')
            error.diagnostic_partial={**v,'status':'failed','response_losses':v['response_losses'][:2]}
            raise error
    r=ex.run_campaign(groups_fixture(),lambda _:PartialBackend())
    assert r['status']=='failed' and r['actual_response_kl_evaluations']==2
    assert len(r['groups'][0]['failed_loss_partial']['response_losses'])==2
    assert r['groups'][0]['loss_work_incomplete'] is True
