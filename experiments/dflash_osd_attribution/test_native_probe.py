"""CPU-only qualification; no native server/model or recorded-case execution."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest

from experiments.dflash_osd_attribution import native_probe as p


def anchor():
    return dict(anchor_id="wildchat_00:20:5", stream="wildchat_00", turn=20,
        input_ids=[10, 11], output_ids=[20, 21, 22, 23, 24], block_output_start=1,
        canonical_prefix_ids=[10, 11, 20], seed_token_id=20,
        canonical_next_ids=[21, 22, 23, 24], expected_frozen_proposal_ids=[21, 22, 23, 24, 1, 2, 3, 4, 5],
        eos_ids=[24], projections={"path":"unused","sha256":"unused"},
        checkpoints={"F": {"checkpoint_identity_sha256":"frozen", "path":"F"},
                     "OSD": {"checkpoint_identity_sha256":"updated", "path":"D19"}})


def test_native_conditioning_excludes_exactly_known_seed():
    a = anchor()
    assert p.validate_anchor(a) == [10, 11]
    a.update(block_output_start=3, canonical_prefix_ids=[10, 11, 20, 21, 22], seed_token_id=22,
             canonical_next_ids=[23, 24])
    assert p.validate_anchor(a) == [10, 11, 20, 21]


@pytest.mark.parametrize("field,value", [("canonical_prefix_ids",[10,11]), ("seed_token_id",99),
    ("canonical_next_ids",[22]), ("expected_frozen_proposal_ids",[21]), ("block_output_start",0)])
def test_rejects_mismatched_geometry(field, value):
    a = anchor(); a[field] = value
    with pytest.raises((ValueError, RuntimeError)):
        p.validate_anchor(a)


def test_prefix_stops_at_eos_and_first_mismatch():
    assert p.prefix_score([5,7,8],[5,7,8], [7]) == dict(correct_proposed_prefix_length=1, boundary="eos", matched_eos=True, prefix_including_eos=2, compared_positions=3)
    assert p.prefix_score([4,7,8],[5,7,8], [7])["correct_proposed_prefix_length"] == 0
    assert p.prefix_score([5,8,7],[5,7,8], [7])["correct_proposed_prefix_length"] == 1


def test_plan_limits_and_checkpoint_binding(monkeypatch):
    monkeypatch.setattr(p.base, "verify_checkpoint", lambda _: None)
    monkeypatch.setattr(p, "checked", lambda _: None)
    a = anchor(); config = dict(attribution_probe_plan=[a], released=a["checkpoints"]["F"])
    assert list(p.validate_plan(config)) == [a["anchor_id"]]
    output = list(range(20, 70))
    config["attribution_probe_plan"] = [dict(a, anchor_id=f"x{i}", output_ids=output,
        block_output_start=i+1, seed_token_id=output[i], canonical_next_ids=output[i+1:i+10],
        canonical_prefix_ids=a["input_ids"]+output[:i+1]) for i in range(33)]
    with pytest.raises((ValueError, RuntimeError), match="32"):
        p.validate_plan(config)
    config["attribution_probe_plan"] = [a, a]
    with pytest.raises((ValueError, RuntimeError), match="Duplicate"):
        p.validate_plan(config)


def fake_runtime(monkeypatch, *, mismatch=False, repeat_forward=False, bad_flush=False):
    torch = pytest.importorskip("torch")
    flags = dict(prefills=0, blocks=0, head_calls=0, flushes=0, context=[], claimed=[], prefix_indices=[])
    class Params:
        def __init__(self, **kwargs): self.kwargs = kwargs
        def normalize(self, tokenizer): pass
    class Req:
        def __init__(self, **kw):
            self.__dict__.update(kw); self.req_pool_idx=0
            self.prefix_indices=torch.tensor([],dtype=torch.int64)
        def set_extend_input_len(self,n): self.extend_input_len=n
    class Batch:
        @classmethod
        def init_new(cls, reqs, *args, **kwargs):
            obj=cls(); obj.reqs=reqs; return obj
        def prepare_for_extend(self):
            req=self.reqs[0]
            flags['prefix_indices'].append(req.prefix_indices.clone())
            self.seq_lens=torch.tensor([len(req.fill_ids)])
    class Model:
        def named_parameters(self): return []
    class Worker:
        block_size=10;tp_size=1;dynamic_verify_len=False;use_thresh_head_two_model=False;use_compact_draft_cache=False
        draft_model=Model()
        def __init__(self):
            self.draft_model_runner=types.SimpleNamespace(forward=self.forward)
        def forward(self,*_): flags["blocks"]+=1
        def forward_batch_generation(self,batch):
            flags["prefills"]+=1
            flags["context"].append(list(batch.reqs[0].fill_ids))
            batch.spec_info=types.SimpleNamespace(verified_id=torch.tensor([20 if not mismatch else 99]))
        def _greedy_sample_from_vocab_parallel_head(self,**kwargs):
            flags["head_calls"]+=1
            return torch.tensor([21,22,23,24,1,2,3,4,5])
        def _prepare_for_speculative_decoding(self,batch,draft_input):
            flags["seed"]=int(draft_input.verified_id[0])
            self._draft_block_positions_buf=torch.arange(int(batch.seq_lens[0]), int(batch.seq_lens[0])+10).view(1,-1)
            self.draft_model_runner.forward(batch)
            if repeat_forward: self.draft_model_runner.forward(batch)
            self._greedy_sample_from_vocab_parallel_head(hidden_states=torch.zeros((9,4),dtype=torch.bfloat16),lm_head=None)
            raise AssertionError("Target verification must never be reached")
    worker=Worker()
    def flush():
        flags["flushes"]+=1
        return not bad_flush or flags["flushes"]<2
    scheduler=types.SimpleNamespace(draft_worker=worker,flush_cache=flush,tokenizer=None,
        model_config=types.SimpleNamespace(vocab_size=100),req_to_token_pool=types.SimpleNamespace(req_to_token=torch.arange(20000,dtype=torch.int32).view(1,-1)),
        token_to_kv_pool_allocator=None,tree_cache=None,enable_overlap=False,spec_algorithm=None)
    modules={"sglang.srt.managers.schedule_batch":types.SimpleNamespace(Req=Req,ScheduleBatch=Batch),
             "sglang.srt.sampling.sampling_params":types.SimpleNamespace(SamplingParams=Params),
             "sglang.srt.model_executor.forward_batch_info":types.SimpleNamespace(ForwardMode=types.SimpleNamespace(DECODE="decode"))}
    for name,module in modules.items(): monkeypatch.setitem(sys.modules,name,module)
    monkeypatch.setattr(p.base,"_worker_config",lambda: {})
    monkeypatch.setattr(p.base,"_serving_entry",lambda _:types.SimpleNamespace(inherited=types.SimpleNamespace(strict_idle=lambda _:None)))
    def inspect(hidden, head, proposals,a,arm,out):
        return dict(proposal_ids=proposals.tolist(),proposed_token_ids=proposals.tolist(),positions=[],**p.prefix_score(proposals.tolist(),a["canonical_next_ids"],a["eos_ids"]))
    monkeypatch.setattr(p,"inspect_head",inspect)
    return scheduler,flags


def test_one_native_block_no_verification_or_generated_history(monkeypatch,tmp_path):
    scheduler,flags=fake_runtime(monkeypatch)
    a=anchor(); original=copy.deepcopy(a)
    result=p.run_native_block(scheduler,a,"F",tmp_path)
    assert flags["blocks"]==flags["head_calls"]==1
    assert flags["prefills"]==1 and flags["context"]==[[10,11]]
    assert flags["seed"]==20 and flags["flushes"]==2
    assert result["draft_block_evaluations"]==1 and result["valid_reconstruction"]
    assert result["target_verifications"]==result["training_steps"]==result["regenerated_response_tokens"]==0
    assert a==original


def test_full_context_prefill_chunking_without_extra_draft_blocks(monkeypatch,tmp_path):
    torch=pytest.importorskip('torch')
    scheduler,flags=fake_runtime(monkeypatch)
    a=anchor();a["input_ids"]=[10]*8431;a["canonical_prefix_ids"]=a["input_ids"]+[20]
    result=p.run_native_block(scheduler,a,"F",tmp_path)
    assert result["target_prefill_chunks"]==[8192,239]
    assert flags["prefills"]==2 and flags["blocks"]==1
    assert flags["context"][-1]==a["input_ids"]
    assert scheduler.req_to_token_pool.req_to_token.dtype == torch.int32
    assert all(indices.dtype == torch.int64 for indices in flags['prefix_indices'])
    assert torch.equal(flags['prefix_indices'][1], torch.arange(8192, dtype=torch.int64))


def test_seed_mismatch_retained_invalid_without_retry(monkeypatch,tmp_path):
    scheduler,flags=fake_runtime(monkeypatch,mismatch=True)
    result=p.run_native_block(scheduler,anchor(),"F",tmp_path)
    assert not result["valid_reconstruction"] and result["reconstructed_target_seed_id"]==99
    assert flags["blocks"]==1 and flags["seed"]==20


def test_extra_forward_refused_and_cache_restored(monkeypatch,tmp_path):
    scheduler,flags=fake_runtime(monkeypatch,repeat_forward=True)
    original=scheduler.draft_worker._greedy_sample_from_vocab_parallel_head
    with pytest.raises((ValueError, RuntimeError),match="More than one"):
        p.run_native_block(scheduler,anchor(),"F",tmp_path)
    assert flags["flushes"]==2
    assert scheduler.draft_worker._greedy_sample_from_vocab_parallel_head==original


def test_failed_final_cache_flush_invalidates_probe(monkeypatch,tmp_path):
    scheduler,_=fake_runtime(monkeypatch,bad_flush=True)
    with pytest.raises((ValueError, RuntimeError),match="after diagnostic"):
        p.run_native_block(scheduler,anchor(),"F",tmp_path)


def test_native_head_capture_keeps_bf16_and_math_separate(monkeypatch,tmp_path):
    torch=pytest.importorskip("torch")
    from experiments.dflash_correction_bank.projections import FixedProjections
    from experiments.dflash_correction_bank.artifacts import binding
    U=np.array([[1.],[0.],[0.],[0.]])
    proj=FixedProjections(U,U,U,np.array([[0.],[1.],[2.]]),4.)
    path=tmp_path/"projections.npz";proj.save(path)
    E=torch.tensor([[0.,1.,0.,0.],[1.,0.,0.,0.],[2.,0.,0.,0.]],dtype=torch.bfloat16)
    hidden=torch.tensor([[1.,0.,0.,0.]]*9,dtype=torch.bfloat16)
    lm=types.SimpleNamespace(weight=E,shard_indices=types.SimpleNamespace(num_added_elements=0,org_vocab_start_index=0,num_org_elements=3))
    a=anchor();a.update(canonical_next_ids=[1],projections=binding(path))
    result=p.inspect_head(hidden,lm,torch.tensor([2]*9),a,"F",tmp_path)
    row=result["positions"][0]
    assert row["native_bf16_canonical_margin"]==-1.
    assert row["capacity"]["deficit"]==1. and row["capacity"]["bound"]==.25
    assert row["capacity"]["classification"]=="not_evaluated"
    assert row["capacity"]["effective_radius"]==.25
    evidence=np.load(result["feature_artifact"]["path"])
    assert evidence["h"].dtype==np.float32 and evidence["E_j"][0,0]==2.


def test_service_no_generation():
    with pytest.raises((ValueError, RuntimeError),match="forbidden"):
        p.ProbeService.generate(None,{})


def test_service_claims_once_before_any_model_work(tmp_path):
    a=anchor()
    service=p.ProbeService.__new__(p.ProbeService)
    service.plan={a["anchor_id"]:a};service.used=set();service.claims=tmp_path/"claims";service.claims.mkdir()
    service.out=tmp_path
    calls=[]
    def publish(checkpoint):
        assert list(service.claims.glob("*.json")), "Durable claim precedes publication"
        calls.append("publish");return {"status":"passed"}
    service._publish_exact=publish
    service.flush=lambda _:calls.append("flush")
    def rpc(name,anchor_id,arm):
        calls.append("block")
        block=service.out/"blocks";block.mkdir(exist_ok=True)
        p.atomic_json(block/(p.digest([anchor_id,arm])+".json"),dict(status="passed",draft_block_evaluations=1))
    service.engine=types.SimpleNamespace(collective_rpc=rpc)
    body=dict(anchor_id=a["anchor_id"],arm="F",checkpoint=a["checkpoints"]["F"])
    assert service.probe(body)["draft_block_evaluations"]==1
    assert calls==["publish","flush","block","flush"]
    with pytest.raises((ValueError,RuntimeError),match="repeated"):
        service.probe(body)
    assert calls.count("block")==1


def test_service_refuses_mismatched_checkpoint_before_claim(tmp_path):
    a=anchor();service=p.ProbeService.__new__(p.ProbeService)
    service.plan={a["anchor_id"]:a};service.used=set();service.claims=tmp_path
    with pytest.raises((ValueError,RuntimeError),match="Checkpoint identity"):
        service.probe(dict(anchor_id=a["anchor_id"],arm="OSD",checkpoint=a["checkpoints"]["F"]))
    assert not list(tmp_path.iterdir())


def test_final_request_selection_is_not_configurable():
    a=anchor();a["turn"]=19
    with pytest.raises((ValueError,RuntimeError),match="actual final saved"):
        p.validate_anchor(a)
