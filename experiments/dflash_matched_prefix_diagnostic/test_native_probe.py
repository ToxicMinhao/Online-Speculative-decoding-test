"""CPU wrappers and BF16 toy heads; no real model, server or saved-case execution."""
import copy
import json
import sys
import types

import pytest

from . import native_probe as p
from . import protocol, evidence


def anchor():
    a = dict(anchor_id='gsm8k_00:r:1',group_id='gsm8k_00',domain='gsm8k',population='heldout',request_id='r',
        input_ids=[10,11],output_ids=[20,21,22,23,24],block_output_start=1,
        canonical_prefix_ids=[10,11,20],seed_token_id=20,canonical_next_ids=[21,22,23,24],
        expected_absolute_positions=list(range(3,12)), expected_frozen_proposal_ids=[21,22,23,24,1,2,3,4,5],
        eos_ids=[128001,128008,128009],response_boundary='cap',reaches_response_boundary=True,
        checkpoints={name:dict(path=name,checkpoint_identity_sha256=character*64)
                     for name,character in [('F','f'),('S8','a'),('L8','b')]})
    bind_reconstruction(a, [])
    return a


def bind_reconstruction(a, rounds):
    a['canonical_prefix_sha256'] = p.digest(a['canonical_prefix_ids'])
    a['retained_round_index'] = len(rounds)
    state=dict(mode=evidence.RECONSTRUCTION_MODE, prefill=dict(
        input_ids_sha256=p.digest(a['input_ids']),prompt_tokens=len(a['input_ids']),
        seed_token_id=a['output_ids'][0],event_sha256=p.digest('fixture-prefill')),
        rounds=rounds,target_verification_count=len(rounds))
    state['schedule_sha256']=evidence.reconstruction_digest(state)
    a['reconstruction']=state


def replay_anchor():
    """Two retained target verifications commit 3 then 2 tokens, no draft replay."""
    a=anchor();a['output_ids']=list(range(20,40));a['block_output_start']=6
    a['canonical_prefix_ids']=a['input_ids']+a['output_ids'][:6]
    a['seed_token_id']=25;a['canonical_next_ids']=a['output_ids'][6:15]
    a['expected_absolute_positions']=list(range(8,17));a['reaches_response_boundary']=False
    rounds=[]
    for i,(start,proposals,predictions) in enumerate([
        (1,[21,22,80,81,82,83,84,85,86],[21,22,23,90,91,92,93,94,95,96]),
        (4,[24,80,81,82,83,84,85,86,87],[24,25,90,91,92,93,94,95,96,97])]):
        accepted=2 if i==0 else 1
        committed=proposals[:accepted]+[predictions[accepted]]
        rounds.append(dict(round_index=i,output_offset=start,
            verification_input_token_ids=[a['output_ids'][start-1]]+proposals,
            verification_input_positions=list(range(len(a['input_ids'])+start-1,len(a['input_ids'])+start+9)),
            draft_proposed_token_ids=proposals,target_predicted_token_ids=predictions,
            target_prediction_positions=list(range(len(a['input_ids'])+start,len(a['input_ids'])+start+10)),
            committed_token_ids=committed,native_acceptance_length_before_stop=accepted,
            stop_clipped=False,finished=False,commit_length=len(committed),
            next_seed_token_id=committed[-1],raw_event_sha256=p.digest(['fixture-round',i])))
    bind_reconstruction(a,rounds)
    return a


def fake_runtime(monkeypatch, *, mismatch=False, repeat_forward=False, bad_flush=False,
                 mutate=None, verify=False, frozen_tail=False, head_failure=False,
                 replay_rounds=None, replay_error=None):
    torch = pytest.importorskip('torch')
    flags = dict(prefills=0,blocks=0,head_calls=0,flushes=0,context=[],prefix_indices=[],
        target_verify_calls=0,prepare_verify_calls=0,native_verify_calls=0,appends=[],
        replay_inputs=[],replay_positions=[],commits=[])
    replay_rounds = copy.deepcopy(replay_rounds or [])
    class Mode(str):
        def is_extend(self): return self == 'extend'
        def is_idle(self): return self == 'idle'
        def is_target_verify(self): return self == 'target_verify'
    modes = types.SimpleNamespace(DECODE=Mode('decode'),EXTEND=Mode('extend'),
        TARGET_VERIFY=Mode('target_verify'),IDLE=Mode('idle'))
    class Params:
        def __init__(self,**kwargs): self.kwargs=kwargs
        def normalize(self,tokenizer): pass
    class Req:
        def __init__(self,**kwargs):
            self.__dict__.update(kwargs); self.req_pool_idx=0
            self.prefix_indices=torch.tensor([],dtype=torch.int64)
            self.output_ids=[];self.kv_committed_len=0;self.kv_allocated_len=0
            self.spec_verify_ct=0;self.spec_accepted_tokens=0;self.spec_proposed_tokens=0
            self.grammar=None;self.finished_reason=None
        def set_extend_input_len(self,n): self.extend_input_len=n
        def check_finished(self): pass
        def finished(self): return self.finished_reason is not None
    class Batch:
        @classmethod
        def init_new(cls,reqs,*args,**kwargs):
            obj=cls();obj.reqs=reqs;obj.device=torch.device('cpu')
            obj.has_grammar=False;obj.sampling_info=types.SimpleNamespace(is_all_greedy=True)
            return obj
        def prepare_for_extend(self):
            req=self.reqs[0];flags['prefix_indices'].append(req.prefix_indices.clone())
            self.seq_lens=torch.tensor([len(req.fill_ids)],dtype=torch.int64)
            self.seq_lens_cpu=self.seq_lens.clone();self.seq_lens_sum=int(self.seq_lens[0])
            self.forward_mode=modes.EXTEND;self.input_ids=torch.tensor(req.fill_ids,dtype=torch.int64)
            self.extend_prefix_lens=[len(req.prefix_indices)]
            self.extend_seq_lens=[req.extend_input_len]
            req.kv_committed_len=req.kv_allocated_len=len(req.fill_ids)
        def get_model_worker_batch(self): return self
        def batch_size(self): return len(self.reqs)
    class VerifyInput:
        def __init__(self,**kwargs): self.__dict__.update(kwargs)
        def prepare_for_verify(self,batch,page_size,*,build_custom_mask=True):
            flags['prepare_verify_calls']+=1
            assert page_size==1 and build_custom_mask
            assert self.draft_token.dtype==self.positions.dtype==torch.int64
            assert self.draft_token_num==10 and self.candidate_lens is None
            batch.input_ids=self.draft_token
            flags['replay_inputs'].append(self.draft_token.tolist())
            flags['replay_positions'].append(self.positions.tolist())
        def verify(self,*,batch,logits_output,page_size):
            flags['native_verify_calls']+=1
            candidates=self.draft_token.tolist(); predicted=logits_output.next_token_logits.argmax(-1).tolist()
            accepted=0
            while accepted<9 and candidates[accepted+1]==predicted[accepted]: accepted+=1
            committed=candidates[1:accepted+1]+[predicted[accepted]]
            if replay_error=='commit': committed[-1]+=1
            req=batch.reqs[0];req.output_ids.extend(committed)
            if replay_error=='finished':req.finished_reason='unexpected_stop'
            advance=len(committed)+(1 if replay_error=='cache' else 0)
            batch.seq_lens+=advance;batch.seq_lens_cpu+=advance;batch.seq_lens_sum+=advance
            req.kv_committed_len+=advance;req.kv_allocated_len=req.kv_committed_len
            req.spec_verify_ct+=1;req.spec_accepted_tokens+=accepted;req.spec_proposed_tokens+=10
            flags['commits'].append(committed)
            return (torch.tensor([committed[-1]],dtype=torch.int64),
                torch.tensor([len(committed)],dtype=torch.int32),
                logits_output.hidden_states[:len(committed)], [accepted])
    def target_forward(batch,**kwargs):
        flags['target_calls']=flags.get('target_calls',0)+1
        if kwargs.get('is_verify'):
            index=flags['target_verify_calls'];flags['target_verify_calls']+=1
            assert batch.forward_mode.is_target_verify()
            record=replay_rounds[index]
            predicted=list(record['target_predicted_token_ids'])
            if replay_error=='target':predicted[-1]+=1
            logits=torch.zeros((10,128),dtype=torch.bfloat16)
            logits[torch.arange(10),torch.tensor(predicted)]=1
            return types.SimpleNamespace(logits_output=types.SimpleNamespace(
                next_token_logits=logits,hidden_states=torch.zeros((10,2),dtype=torch.bfloat16)),
                can_run_cuda_graph=False)
        return types.SimpleNamespace(logits_output=None,can_run_cuda_graph=False)
    class Worker:
        block_size=10;tp_size=1;dynamic_verify_len=False;use_thresh_head_two_model=False;use_compact_draft_cache=False
        device=torch.device('cpu');page_size=1
        def __init__(self):
            self.draft_model=torch.nn.Linear(2,2,bias=False).to(torch.bfloat16)
            self.target_worker=types.SimpleNamespace(model_runner=types.SimpleNamespace(
                model=torch.nn.Linear(2,2,bias=False).to(torch.bfloat16)),forward_batch_generation=target_forward)
            self.draft_model_runner=types.SimpleNamespace(forward=self.forward)
            self.model_runner=types.SimpleNamespace(attn_backend=object())
        def forward(self,*_):
            flags['blocks']+=1
            if mutate:
                model=self.draft_model if mutate=='draft' else self.target_worker.model_runner.model
                with torch.no_grad(): model.weight.add_(1.)
        def _append_target_hidden_to_draft_kv(self,batch,draft_input):
            n=int(draft_input.ctx_lens[0]) if draft_input.ctx_lens is not None else 0
            if n:
                flags['appends'].append(n)
                draft_input.draft_seq_lens=batch.seq_lens.to(torch.int32).clone()
                draft_input.ctx_lens=None;draft_input.target_hidden=None
        def forward_batch_generation(self,batch):
            if batch.forward_mode.is_extend():
                flags['prefills']+=1;flags['context'].append(list(batch.reqs[0].fill_ids))
                self.target_worker.forward_batch_generation(batch)
                batch.spec_info=types.SimpleNamespace(verified_id=torch.tensor([99 if mismatch else 20]),
                    draft_seq_lens=torch.tensor(batch.extend_prefix_lens,dtype=torch.int32),
                    ctx_lens=torch.tensor(batch.extend_seq_lens,dtype=torch.int32),target_hidden=None)
                self._append_target_hidden_to_draft_kv(batch,batch.spec_info)
                return types.SimpleNamespace(next_token_ids=batch.spec_info.verified_id,num_accepted_tokens=0)
            draft_input=batch.spec_info
            self._prepare_for_speculative_decoding(batch,draft_input)
            verify_input=batch.spec_info
            result=self.target_worker.forward_batch_generation(batch,is_verify=True)
            ids,commit_lens,hidden,accepted=verify_input.verify(batch=batch,
                logits_output=result.logits_output,page_size=self.page_size)
            draft_input.verified_id=ids;draft_input.ctx_lens=commit_lens;draft_input.target_hidden=hidden
            self._append_target_hidden_to_draft_kv(batch,draft_input)
            batch.spec_info=draft_input;batch.forward_mode=modes.DECODE
            return types.SimpleNamespace(next_token_ids=ids,num_accepted_tokens=sum(accepted))
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
        'sglang.srt.model_executor.forward_batch_info':types.SimpleNamespace(ForwardMode=modes),
        'sglang.srt.speculative.dflash_info':types.SimpleNamespace(DFlashVerifyInput=VerifyInput),
        'sglang.srt.speculative.dflash_utils':types.SimpleNamespace(
            resolve_dflash_verify_mask_policy=lambda backend:('fixture',True))}.items():
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
                native_argmax_id=ids[i],native_best_other_id=0 if ids[i]==y else ids[i],
                native_canonical_logit=2. if ids[i]==y else 0.,native_best_other_logit=1.,
                native_canonical_log_probability=-.5,native_bf16_canonical_margin=1. if ids[i]==y else -1.,native_argmax_tie_count=1,
                canonical_tied_for_max=False) for i,y in enumerate(a['canonical_next_ids'])])
    monkeypatch.setattr(p,'inspect_head',inspect)
    return scheduler,flags


def test_exact_prefix_seed_exclusion_and_geometry():
    a=anchor();assert p.validate_anchor(a)==[10,11]
    a=replay_anchor()
    assert p.validate_anchor(a)==[10,11,20,21,22,23,24]


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
    bind_reconstruction(a,[])
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
    assert row['reconstructed_prefill_seed_id']==99 and row['native_draft_block_evaluations']==0
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
    for field in ('native_draft_block_evaluations','draft_block_evaluations','native_head_calls',
                  'reconstruction_target_verifications','reconstruction_target_verify_attempts',
                  'target_verifications','scored_target_verifications'):
        assert raw[field]==(None if stage=='scheduler' else 0), field
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


@pytest.mark.parametrize('arm',['F','S8','L8'])
def test_saved_verification_schedule_rebuilds_prefix_without_unselected_draft_work(monkeypatch,tmp_path,arm):
    a=replay_anchor();saved=copy.deepcopy(a)
    scheduler,flags=fake_runtime(monkeypatch,replay_rounds=a['reconstruction']['rounds'])
    worker=scheduler.draft_worker
    original_prepare=worker._prepare_for_speculative_decoding
    original_target=worker.target_worker.forward_batch_generation
    row=p.run_native_block(scheduler,a,arm,tmp_path)
    assert protocol.validate_row(a,arm,row)==dict(status='passed',errors=[])
    assert flags['context']==[[10,11]]  # Never cold-prefill the completed canonical prefix.
    assert flags['appends']==[2,3,2]
    assert flags['commits']==[[21,22,23],[24,25]]
    assert flags['replay_inputs']==[r['verification_input_token_ids'] for r in a['reconstruction']['rounds']]
    assert flags['replay_positions']==[r['verification_input_positions'] for r in a['reconstruction']['rounds']]
    assert flags['prepare_verify_calls']==flags['native_verify_calls']==flags['target_verify_calls']==2
    assert flags['blocks']==flags['head_calls']==1 and flags['target_calls']==3
    assert flags['seed']==25 and row['reconstructed_target_seed_id']==25
    assert row['target_verifications']==2
    assert row['regenerated_response_tokens']==row['training_steps']==0
    assert row['native_draft_block_evaluations']==1
    assert worker._prepare_for_speculative_decoding==original_prepare
    assert worker.target_worker.forward_batch_generation==original_target
    assert a==saved and json.loads(next(tmp_path.glob('*.json')).read_text())==row


@pytest.mark.parametrize('fault',['target','commit','cache','finished'])
def test_replay_discrepancies_stop_before_measured_block_and_restore_hooks(monkeypatch,tmp_path,fault):
    a=replay_anchor()
    scheduler,flags=fake_runtime(monkeypatch,replay_rounds=a['reconstruction']['rounds'],replay_error=fault)
    worker=scheduler.draft_worker;prepare=worker._prepare_for_speculative_decoding
    target=worker.target_worker.forward_batch_generation
    with pytest.raises(ValueError) as error:p.run_native_block(scheduler,a,'F',tmp_path)
    row=error.value.diagnostic_partial
    assert row['status']=='failed' and flags['flushes']==2
    assert row['native_draft_block_evaluations']==flags['blocks']==flags['head_calls']==0
    assert row['target_verifications']==flags['target_verify_calls']==1
    assert flags['native_verify_calls']==(0 if fault=='target' else 1)
    assert worker._prepare_for_speculative_decoding==prepare
    assert worker.target_worker.forward_batch_generation==target
    assert json.loads(next(tmp_path.glob('*.json')).read_text())==row
    assert protocol.validate_row(a,'F',row)['status']=='failed'


def test_saved_reconstruction_schema_is_required_and_hash_bound(monkeypatch,tmp_path):
    scheduler,flags=fake_runtime(monkeypatch);a=anchor()
    del a['reconstruction']
    with pytest.raises((ValueError,KeyError)):p.run_native_block(scheduler,a,'F',tmp_path)
    assert flags.get('target_calls',0)==flags['blocks']==0
    a=replay_anchor();a['reconstruction']['rounds'][0]['verification_input_token_ids'][2]=100
    with pytest.raises(ValueError):p.run_native_block(scheduler,a,'F',tmp_path)
    assert flags['blocks']==0 and flags['prefills']==0


def test_original_prompt_chunking_then_saved_verifications_preserves_int64_indices(monkeypatch,tmp_path):
    torch=pytest.importorskip('torch');a=replay_anchor();old_length=len(a['input_ids'])
    a['input_ids']=[10]*8431
    a['canonical_prefix_ids']=a['input_ids']+a['output_ids'][:a['block_output_start']]
    delta=len(a['input_ids'])-old_length
    a['expected_absolute_positions']=[x+delta for x in a['expected_absolute_positions']]
    rounds=a['reconstruction']['rounds']
    for record in rounds:
        for field in ('verification_input_positions','target_prediction_positions'):
            record[field]=[x+delta for x in record[field]]
    bind_reconstruction(a,rounds)
    scheduler,flags=fake_runtime(monkeypatch,replay_rounds=rounds)
    row=p.run_native_block(scheduler,a,'F',tmp_path)
    assert row['target_prefill_chunks']==[8192,239]
    assert [len(x) for x in flags['context']]==[8192,8431]
    assert flags['appends']==[8192,239,3,2]
    assert flags['target_calls']==4 and flags['blocks']==1 and row['target_verifications']==2
    assert all(x.dtype==torch.int64 for x in flags['prefix_indices'])
    copied=flags['prefix_indices'][1]
    assert torch.equal(copied,torch.arange(8192,dtype=torch.int64))
    assert copied.data_ptr()!=scheduler.req_to_token_pool.req_to_token.data_ptr()
    assert protocol.validate_row(a,'F',row)['status']=='passed'


def test_failed_native_case_shape_replays_87_plus_7_1_1_5_not_cold_101(monkeypatch,tmp_path):
    a=anchor();a['input_ids']=[10]*87;a['output_ids']=list(range(20,60))
    a['block_output_start']=15;a['canonical_prefix_ids']=a['input_ids']+a['output_ids'][:15]
    a['seed_token_id']=a['output_ids'][14];a['canonical_next_ids']=a['output_ids'][15:24]
    a['expected_absolute_positions']=list(range(102,111));a['reaches_response_boundary']=False
    rounds=[]
    for i,(offset,commit_length) in enumerate([(1,7),(8,1),(9,1),(10,5)]):
        accepted=commit_length-1;committed=a['output_ids'][offset:offset+commit_length]
        proposals=committed[:accepted]+[80]*(9-accepted)
        predictions=committed+[100]*(10-commit_length)
        rounds.append(dict(round_index=i,output_offset=offset,
            verification_input_token_ids=[a['output_ids'][offset-1]]+proposals,
            verification_input_positions=list(range(87+offset-1,87+offset+9)),
            draft_proposed_token_ids=proposals,target_predicted_token_ids=predictions,
            target_prediction_positions=list(range(87+offset,87+offset+10)),
            committed_token_ids=committed,native_acceptance_length_before_stop=accepted,
            stop_clipped=False,finished=False,commit_length=commit_length,
            next_seed_token_id=committed[-1],raw_event_sha256=p.digest(['shape-round',i])))
    bind_reconstruction(a,rounds)
    scheduler,flags=fake_runtime(monkeypatch,replay_rounds=rounds)
    row=p.run_native_block(scheduler,a,'F',tmp_path)
    assert flags['appends']==[87,7,1,1,5] and flags['context']==[[10]*87]
    assert flags['target_calls']==5 and flags['blocks']==flags['head_calls']==1
    assert row['target_verifications']==row['reconstruction_target_verifications']==4
    assert row['reconstruction_target_verify_attempts']==4 and row['scored_target_verifications']==0
    assert [(r['seq_lens_before'],r['seq_lens_after']) for r in row['reconstruction_rounds']]==[
        (87,94),(94,95),(95,96),(96,101)]
    assert all(r['status']=='passed' and r['target_predictions_match'] and r['committed_tokens_match']
        and r['cache_advance_matches'] for r in row['reconstruction_rounds'])
    assert row['native_materialized_block_ids']==[34]+[128002]*9
    assert row['native_absolute_proposal_positions']==list(range(102,111))
    assert protocol.validate_row(a,'F',row)['status']=='passed'


@pytest.mark.parametrize('fault',['seed','cache','grammar','nongreedy'])
def test_recorded_verification_helper_refuses_changed_conditioning_before_prepare(monkeypatch,fault):
    torch=pytest.importorskip('torch');a=replay_anchor();record=a['reconstruction']['rounds'][0]
    scheduler,flags=fake_runtime(monkeypatch,replay_rounds=a['reconstruction']['rounds'])
    batch=types.SimpleNamespace(batch_size=lambda:1,has_grammar=fault=='grammar',
        sampling_info=types.SimpleNamespace(is_all_greedy=fault!='nongreedy'),
        seq_lens=torch.tensor([3 if fault=='cache' else 2]),
        forward_mode=types.SimpleNamespace(is_idle=lambda:False))
    draft_input=types.SimpleNamespace(verified_id=torch.tensor([99 if fault=='seed' else 20]))
    with pytest.raises(ValueError):p.prepare_recorded_verification(scheduler.draft_worker,batch,draft_input,record)
    assert flags['prepare_verify_calls']==flags['native_verify_calls']==flags['target_verify_calls']==0
    assert flags['blocks']==0 and flags['appends']==[]


def observer_entry_fixture(monkeypatch,tmp_path,*,installed=False,active=None,guard_failure=False):
    events=[]
    class Scheduler:
        _d197_trace_installed=installed
    def source_guard():
        events.append('source_guard')
        if guard_failure:raise ValueError('pinned source changed')
        return dict(source_sha256='a'*64)
    def write(path,payload):
        events.append('receipt')
        p.write_json(path,payload)
    def d181(*args,**kwargs):
        events.append(('D181',args,kwargs))
        assert Scheduler.matched_prefix_probe is p._scheduler_rpc
        assert (tmp_path/'REPLAY_OBSERVER_COMPATIBILITY.json').exists()
        return 'native scheduler result'
    def d202(*args,**kwargs):
        events.append('D202')
        raise AssertionError('Whole-response observer installer must not run')
    entry=types.SimpleNamespace(source_guard=source_guard,_active_trace=active,
        _original_scheduler=d181,serving_scheduler=d202,
        inherited=types.SimpleNamespace(write=write,outpath=lambda name:tmp_path/name))
    monkeypatch.setattr(p.base,'_worker_config',lambda:{'fixture':'replay'})
    monkeypatch.setattr(p.base,'_serving_entry',lambda config:entry)
    monkeypatch.setitem(sys.modules,'sglang.srt.managers.scheduler',types.SimpleNamespace(Scheduler=Scheduler))
    return events,Scheduler


def test_replay_scheduler_checks_sources_and_records_observer_policy_before_d181(monkeypatch,tmp_path):
    events,scheduler=observer_entry_fixture(monkeypatch,tmp_path)
    assert p._probe_scheduler('fixture argument',setting=7)=='native scheduler result'
    assert events==['source_guard','receipt',('D181',('fixture argument',),dict(setting=7))]
    receipt=json.loads((tmp_path/'REPLAY_OBSERVER_COMPATIBILITY.json').read_text())
    assert receipt['status']=='passed' and receipt['sources']==dict(source_sha256='a'*64)
    assert receipt['whole_response_observers_installed'] is False
    assert receipt['native_verify_unchanged'] is True and receipt['native_acceptance_unchanged'] is True
    assert scheduler.matched_prefix_probe is p._scheduler_rpc


@pytest.mark.parametrize('condition',['installed','active_trace','source_guard_failure'])
def test_replay_scheduler_refuses_observers_or_source_failure_before_native_start(monkeypatch,tmp_path,condition):
    events,scheduler=observer_entry_fixture(monkeypatch,tmp_path,
        installed=condition=='installed',active={} if condition=='active_trace' else None,
        guard_failure=condition=='source_guard_failure')
    with pytest.raises(ValueError):p._probe_scheduler('must not run')
    assert events==['source_guard']
    assert not hasattr(scheduler,'matched_prefix_probe')
    assert not (tmp_path/'REPLAY_OBSERVER_COMPATIBILITY.json').exists()
