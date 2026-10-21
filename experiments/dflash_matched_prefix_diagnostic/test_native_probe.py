"""CPU wrappers and BF16 toy heads; no real model, server or saved-case execution."""
import copy
import json
import sys
import types

import pytest

from . import native_probe as p
from . import protocol


def anchor():
    a = dict(anchor_id='gsm8k_00:r:1',group_id='gsm8k_00',domain='gsm8k',population='heldout',request_id='r',
        input_ids=[10,11],output_ids=[20,21,22,23,24],block_output_start=1,
        canonical_prefix_ids=[10,11,20],seed_token_id=20,canonical_next_ids=[21,22,23,24],
        expected_absolute_positions=list(range(3,12)), expected_frozen_proposal_ids=[21,22,23,24,1,2,3,4,5],
        eos_ids=[128001,128008,128009],response_boundary='cap',reaches_response_boundary=True,
        checkpoints={name:dict(path=name,checkpoint_identity_sha256=character*64)
                     for name,character in [('F','f'),('S8','a'),('L8','b')]})
    a['canonical_prefix_sha256'] = p.digest(a['canonical_prefix_ids'])
    return a


def fake_runtime(monkeypatch, *, mismatch=False, repeat_forward=False, bad_flush=False,
                 mutate=None, verify=False, frozen_tail=False, head_failure=False):
    torch = pytest.importorskip('torch')
    flags = dict(prefills=0,blocks=0,head_calls=0,flushes=0,context=[],prefix_indices=[])
    class Params:
        def __init__(self,**kwargs): self.kwargs=kwargs
        def normalize(self,tokenizer): pass
    class Req:
        def __init__(self,**kwargs):
            self.__dict__.update(kwargs); self.req_pool_idx=0
            self.prefix_indices=torch.tensor([],dtype=torch.int64)
        def set_extend_input_len(self,n): self.extend_input_len=n
    class Batch:
        @classmethod
        def init_new(cls,reqs,*args,**kwargs):
            obj=cls();obj.reqs=reqs;return obj
        def prepare_for_extend(self):
            req=self.reqs[0];flags['prefix_indices'].append(req.prefix_indices.clone())
            self.seq_lens=torch.tensor([len(req.fill_ids)])
    def target_forward(*args,**kwargs): flags['target_calls']=flags.get('target_calls',0)+1
    class Worker:
        block_size=10;tp_size=1;dynamic_verify_len=False;use_thresh_head_two_model=False;use_compact_draft_cache=False
        def __init__(self):
            self.draft_model=torch.nn.Linear(2,2,bias=False).to(torch.bfloat16)
            self.target_worker=types.SimpleNamespace(model_runner=types.SimpleNamespace(
                model=torch.nn.Linear(2,2,bias=False).to(torch.bfloat16)),forward_batch_generation=target_forward)
            self.draft_model_runner=types.SimpleNamespace(forward=self.forward)
        def forward(self,*_):
            flags['blocks']+=1
            if mutate:
                model=self.draft_model if mutate=='draft' else self.target_worker.model_runner.model
                with torch.no_grad(): model.weight.add_(1.)
        def forward_batch_generation(self,batch):
            flags['prefills']+=1;flags['context'].append(list(batch.reqs[0].fill_ids))
            self.target_worker.forward_batch_generation(batch)
            batch.spec_info=types.SimpleNamespace(verified_id=torch.tensor([99 if mismatch else 20]))
        def _greedy_sample_from_vocab_parallel_head(self,**kwargs):
            flags['head_calls']+=1
            return torch.tensor([21,22,23,24,1,2,3,4,6 if frozen_tail else 5])
        def _prepare_for_speculative_decoding(self,batch,draft_input):
            flags['seed']=int(draft_input.verified_id[0])
            self._draft_block_positions_buf=torch.arange(int(batch.seq_lens[0]),int(batch.seq_lens[0])+10).view(1,-1)
            self._draft_block_ids_buf=torch.tensor([[flags['seed']]+[128002]*9])
            self.draft_model_runner.forward(batch)
            if repeat_forward:self.draft_model_runner.forward(batch)
            if verify:self.target_worker.forward_batch_generation(batch)
            self._greedy_sample_from_vocab_parallel_head(hidden_states=torch.zeros((9,2),dtype=torch.bfloat16),lm_head=None)
            raise AssertionError('Verification must never be reached')
    worker=Worker()
    def flush():
        flags['flushes']+=1
        return not bad_flush or flags['flushes']<2
    scheduler=types.SimpleNamespace(draft_worker=worker,flush_cache=flush,tokenizer=None,
        model_config=types.SimpleNamespace(vocab_size=128256),req_to_token_pool=types.SimpleNamespace(
            req_to_token=torch.arange(33000,dtype=torch.int32).view(1,-1)),
        token_to_kv_pool_allocator=None,tree_cache=None,enable_overlap=False,spec_algorithm=None)
    for name,module in {
        'sglang.srt.managers.schedule_batch':types.SimpleNamespace(Req=Req,ScheduleBatch=Batch),
        'sglang.srt.sampling.sampling_params':types.SimpleNamespace(SamplingParams=Params),
        'sglang.srt.model_executor.forward_batch_info':types.SimpleNamespace(ForwardMode=types.SimpleNamespace(DECODE='decode'))}.items():
        monkeypatch.setitem(sys.modules,name,module)
    monkeypatch.setattr(p.base,'_worker_config',lambda:{})
    monkeypatch.setattr(p.base,'_serving_entry',lambda _:types.SimpleNamespace(
        inherited=types.SimpleNamespace(strict_idle=lambda _:None)))
    def inspect(hidden,head,proposals,a,arm,out):
        if head_failure:
            error=ValueError('Nonfinite native BF16 logits')
            error.diagnostic_partial=dict(native_logits_finite=False,nonfinite_logit_count=3)
            raise error
        ids=proposals.tolist()
        return dict(proposal_ids=ids,proposed_token_ids=ids,native_argmax_ids=ids,native_logits_finite=True,
            native_feature_dtype='torch.bfloat16',native_logits_dtype='torch.bfloat16',
            inspection_reduction_dtype='torch.float32',nonfinite_logit_count=0,
            positions=[dict(block_position=i,native_block_slot=i+1,absolute_output_position=a['block_output_start']+i,
                absolute_position=a['expected_absolute_positions'][i],canonical_token_id=y,proposal_token_id=ids[i],
                native_argmax_id=ids[i],native_best_other_id=0,native_canonical_logit=2.,native_best_other_logit=1.,
                native_canonical_log_probability=-.5,native_bf16_canonical_margin=1.,native_argmax_tie_count=1,
                canonical_tied_for_max=False) for i,y in enumerate(a['canonical_next_ids'])])
    monkeypatch.setattr(p,'inspect_head',inspect)
    return scheduler,flags


def test_exact_prefix_seed_exclusion_and_geometry():
    a=anchor();assert p.validate_anchor(a)==[10,11]
    a.update(block_output_start=3,canonical_prefix_ids=[10,11,20,21,22],seed_token_id=22,
        canonical_next_ids=[23,24],expected_absolute_positions=list(range(5,14)))
    assert p.validate_anchor(a)==[10,11,20,21]


@pytest.mark.parametrize('field,value',[('canonical_prefix_ids',[10,11]),('seed_token_id',99),
    ('canonical_next_ids',[22]),('expected_frozen_proposal_ids',[21]),('block_output_start',0),
    ('expected_absolute_positions',list(range(2,11))),('eos_ids',[24])])
def test_bad_geometry_rejected(field,value):
    a=anchor();a[field]=value
    with pytest.raises(ValueError):p.validate_anchor(a)


def test_one_native_block_is_valid_shared_protocol_and_no_verification(monkeypatch,tmp_path):
    scheduler,flags=fake_runtime(monkeypatch);a=anchor();saved=copy.deepcopy(a)
    row=p.run_native_block(scheduler,a,'F',tmp_path)
    assert protocol.validate_row(a,'F',row)==dict(status='passed',errors=[])
    assert flags['blocks']==flags['head_calls']==flags['target_calls']==1
    assert flags['context']==[[10,11]] and flags['seed']==20 and flags['flushes']==2
    assert row['native_materialized_block_ids']==[20]+[128002]*9 and a==saved
    assert json.loads(next(tmp_path.glob('*.json')).read_text())==row


def test_long_prefix_int32_pool_uses_owned_int64_copies(monkeypatch,tmp_path):
    torch=pytest.importorskip('torch');scheduler,flags=fake_runtime(monkeypatch);a=anchor()
    a['input_ids']=[10]*8431;a['canonical_prefix_ids']=a['input_ids']+[20]
    a['canonical_prefix_sha256']=p.digest(a['canonical_prefix_ids'])
    a['expected_absolute_positions']=list(range(8432,8441))
    row=p.run_native_block(scheduler,a,'F',tmp_path)
    assert row['target_prefill_chunks']==[8192,239] and flags['blocks']==1
    assert scheduler.req_to_token_pool.req_to_token.dtype==torch.int32
    assert all(x.dtype==torch.int64 for x in flags['prefix_indices'])
    copied=flags['prefix_indices'][1]
    assert torch.equal(copied,torch.arange(8192,dtype=torch.int64))
    with torch.inference_mode():copied[0]=999
    assert scheduler.req_to_token_pool.req_to_token[0,0]==0
    assert protocol.validate_row(a,'F',row)['status']=='passed'


def test_seed_mismatch_stops_BEFORE_saved_seed_and_zero_draft_work(monkeypatch,tmp_path):
    scheduler,flags=fake_runtime(monkeypatch,mismatch=True)
    with pytest.raises(ValueError,match='Reconstructed target seed') as error:
        p.run_native_block(scheduler,anchor(),'F',tmp_path)
    row=error.value.diagnostic_partial
    assert flags['blocks']==flags['head_calls']==0 and 'seed' not in flags
    assert row['reconstructed_target_seed_id']==99 and row['native_draft_block_evaluations']==0
    assert row['target_prefill_chunks']==[2] and row['status']=='failed'
    assert json.loads(next(tmp_path.glob('*.json')).read_text())==row


@pytest.mark.parametrize('option,match',[('repeat_forward','More than one'),('bad_flush','after diagnostic'),
    ('verify','verification/output'),('frozen_tail','all nine'),('head_failure','Nonfinite')])
def test_failures_are_durable_restore_wrappers_and_no_retry(monkeypatch,tmp_path,option,match):
    scheduler,flags=fake_runtime(monkeypatch,**{option:True})
    original=scheduler.draft_worker._greedy_sample_from_vocab_parallel_head
    with pytest.raises(ValueError,match=match) as error:p.run_native_block(scheduler,anchor(),'F',tmp_path)
    assert error.value.diagnostic_partial['status']=='failed'
    assert scheduler.draft_worker._greedy_sample_from_vocab_parallel_head==original
    assert flags['flushes']==2 and flags['blocks']==1
    assert json.loads(next(tmp_path.glob('*.json')).read_text())['native_draft_block_evaluations']==1
    with pytest.raises(ValueError,match='overwrite'):p.run_native_block(scheduler,anchor(),'F',tmp_path)
    assert flags['blocks']==1


@pytest.mark.parametrize('model',['target','draft'])
def test_mutation_invalidates_either_model(monkeypatch,tmp_path,model):
    scheduler,_=fake_runtime(monkeypatch,mutate=model)
    with pytest.raises(ValueError,match='changed '+model) as error:p.run_native_block(scheduler,anchor(),'F',tmp_path)
    assert not error.value.diagnostic_partial[model+'_weights_unchanged']


def test_bf16_head_fp32_logprob_margin_and_ties():
    torch=pytest.importorskip('torch')
    E=torch.tensor([[1.,0.],[1.,0.],[0.,1.]],dtype=torch.bfloat16)
    head=types.SimpleNamespace(weight=E,shard_indices=types.SimpleNamespace(
        num_added_elements=0,org_vocab_start_index=0,num_org_elements=3))
    hidden=torch.tensor([[1.,0.]]*9,dtype=torch.bfloat16);a=anchor();a['canonical_next_ids']=[0,1,2]
    row=p.inspect_head(hidden,head,torch.tensor([0]*9),a,'F')
    first,second,third=row['positions']
    assert first['native_bf16_canonical_margin']==second['native_bf16_canonical_margin']==0.
    assert first['canonical_tied_for_max'] and second['canonical_tied_for_max']
    assert first['native_argmax_tie_count']==2 and first['correct'] and not second['correct']
    assert third['native_bf16_canonical_margin']==-1. and not third['canonical_tied_for_max']
    expected=float(torch.log_softmax(torch.tensor([1.,1.,0.]),dim=0)[0])
    assert first['native_canonical_log_probability']==pytest.approx(expected)
    assert row['native_logits_dtype']=='torch.bfloat16' and row['inspection_reduction_dtype']=='torch.float32'
    with pytest.raises(ValueError,match='argmax'):p.inspect_head(hidden,head,torch.tensor([1]*9),a,'F')


def test_nonfinite_head_retains_counts_and_native_ids():
    torch=pytest.importorskip('torch')
    head=types.SimpleNamespace(weight=torch.tensor([[float('nan'),0.],[1.,0.]],dtype=torch.bfloat16),
        shard_indices=types.SimpleNamespace(num_added_elements=0,org_vocab_start_index=0,num_org_elements=2))
    a=anchor();a['canonical_next_ids']=[1]
    with pytest.raises(ValueError,match='Nonfinite') as error:
        p.inspect_head(torch.ones((9,2),dtype=torch.bfloat16),head,torch.tensor([0]*9),a,'F')
    assert error.value.diagnostic_partial['nonfinite_logit_count']==9
    assert error.value.diagnostic_partial['proposal_ids']==[0]*9


def test_plan_uses_current_groups_no_historical_stream_or_projection_constraints():
    a=anchor();cfg=dict(episode_id=a['group_id'],matched_prefix_probe_plan=[a],
        matched_prefix_checkpoints=a['checkpoints'],released=a['checkpoints']['F'])
    assert p.validate_plan(cfg)=={a['anchor_id']:a}
    a['population']='source';a['source_control']=True
    assert p.validate_plan(cfg)=={a['anchor_id']:a}
    with pytest.raises(ValueError,match='Duplicate'):p.validate_plan({**cfg,'matched_prefix_probe_plan':[a,a]})
    a['group_id']='different'
    with pytest.raises(ValueError,match='group/checkpoint'):p.validate_plan(cfg)


def test_service_claim_before_publication_and_duplicate_refused(tmp_path):
    a=anchor();s=p.ProbeService.__new__(p.ProbeService)
    s.plan={a['anchor_id']:a};s.used=set();s.claims=tmp_path/'claims';s.claims.mkdir();s.out=tmp_path
    calls=[]
    def publish(cp):
        assert list(s.claims.glob('*.json'));calls.append('publish');return dict(activation_s=.2)
    def rpc(name,anchor_id,arm):
        calls.append('block');path=tmp_path/'blocks';path.mkdir()
        p.write_json(path/(p.digest([anchor_id,arm])+'.json'),dict(status='passed',costs={}))
    s._publish_exact=publish;s.engine=types.SimpleNamespace(collective_rpc=rpc)
    body=dict(anchor_id=a['anchor_id'],arm='F',checkpoint=a['checkpoints']['F'])
    assert s.probe(body)['publication']['activation_s']==.2
    assert calls==['publish','block']
    with pytest.raises(ValueError,match='repeated'):s.probe(body)
    with pytest.raises(ValueError,match='forbidden'):s.generate({})


def test_checkpoint_activation_refresh_even_initial_f_and_stale_helpers_refused(monkeypatch,tmp_path):
    a=anchor();s=p.ProbeService.__new__(p.ProbeService);s.config={};s.out=tmp_path
    s.active=a['checkpoints']['F'];s.refreshed=set();calls=[]
    receipt=dict(status='passed',success=True,helper_matches_current_layers=True,
        helper_norm_matches_current_layers=True,cache_after={'tree_total_size':0})
    s.bridge=types.SimpleNamespace(_two_update_admitted=set(),publish=lambda body:(calls.append(body) or receipt),
        state=lambda:{'draft_model_path':s.active['path']})
    monkeypatch.setattr(p.native,'admit_checkpoint',lambda *_:None)
    assert s._publish_exact(s.active)['helper_refresh_valid'] and len(calls)==1
    assert s._publish_exact(s.active)['already_active'] and len(calls)==1
    s.refreshed.clear();receipt['helper_norm_matches_current_layers']=False
    with pytest.raises(ValueError,match='refresh'):s._publish_exact(s.active)


@pytest.mark.parametrize('stage',['activation','scheduler'])
def test_missing_scheduler_receipt_keeps_work_unknown_and_activation_separate(tmp_path,stage):
    a=anchor();s=p.ProbeService.__new__(p.ProbeService);s.out=tmp_path;s.plan={a['anchor_id']:a};s.used=set()
    s.claims=tmp_path/'claims';s.claims.mkdir()
    def fail(*args,**kwargs):raise RuntimeError('interrupted')
    s._publish_exact=fail if stage=='activation' else lambda _:dict(activation_s=.2)
    s.engine=types.SimpleNamespace(collective_rpc=fail)
    with pytest.raises(RuntimeError) as error:
        s.probe(dict(anchor_id=a['anchor_id'],arm='F',checkpoint=a['checkpoints']['F']))
    raw=error.value.diagnostic_partial
    assert raw['native_draft_block_evaluations']==(None if stage=='scheduler' else 0)
    assert raw['counters_known']==(stage=='activation')
    assert raw['failure_stage']==('scheduler_transport' if stage=='scheduler' else 'activation')
    assert ('unresolved_scheduler_transport_s' in raw['costs'])==(stage=='scheduler')
    assert json.loads(next((tmp_path/'blocks').glob('*.json')).read_text())==raw


def test_nonfinite_fp32_inspection_is_durable_json_not_lost_to_serialization(monkeypatch,tmp_path):
    scheduler,_=fake_runtime(monkeypatch)
    def fail(*args,**kwargs):
        error=ValueError('Nonfinite FP32 inspection')
        error.diagnostic_partial=dict(positions=[dict(native_bf16_canonical_margin=float('-inf'))])
        raise error
    monkeypatch.setattr(p,'inspect_head',fail)
    with pytest.raises(ValueError) as error:p.run_native_block(scheduler,anchor(),'F',tmp_path)
    assert error.value.diagnostic_partial['positions'][0]['native_bf16_canonical_margin']=={'nonfinite_float':'-inf'}
    assert json.loads(next(tmp_path.glob('*.json')).read_text())==error.value.diagnostic_partial
