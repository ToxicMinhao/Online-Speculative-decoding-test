import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from experiments.verification_transition.budget import WorkBudget, BudgetExceeded
from experiments.verification_transition.native import (
    Ingestion, NativeSuccessors, acceptance_totals, assert_neutral_suffix, run_counted,
)


class Mode:
    def __init__(self, extend=False):
        self.extend = extend

    def is_extend(self):
        return self.extend

    def is_idle(self):
        return False


class Req:
    def __init__(self):
        self.origin_input_ids = [10, 11, 12]
        self.output_ids = [20, 21, 22]
        self.sampling_params = SimpleNamespace(max_new_tokens=8)
        self.grammar = None

    def finished(self):
        return len(self.output_ids) >= self.sampling_params.max_new_tokens


class Cache:
    def __init__(self, count):
        self.rows = [(torch.arange(40, dtype=torch.float32).reshape(20, 2)+i,
                      torch.arange(40, dtype=torch.float32).reshape(20, 2)-i)
                     for i in range(count)]

    def get_kv_buffer(self, layer):
        return self.rows[layer]


class Allocator:
    def __init__(self):
        self.state = (torch.arange(6, 20),)

    def backup_state(self):
        return self.state

    def restore_state(self, state):
        self.state = state


def fixture():
    pool = SimpleNamespace(req_to_token=torch.arange(20, dtype=torch.int32).reshape(1, 20), free_slots=[])
    allocator = Allocator()
    layers = [SimpleNamespace(self_attn=SimpleNamespace(attn=SimpleNamespace(layer_id=i))) for i in range(5)]
    target = SimpleNamespace(model=SimpleNamespace(layers=layers[:2]), req_to_token_pool=pool,
        token_to_kv_pool=Cache(2), token_to_kv_pool_allocator=allocator)
    draft = SimpleNamespace(model=SimpleNamespace(layers=layers), req_to_token_pool=pool,
        token_to_kv_pool=Cache(5), token_to_kv_pool_allocator=allocator)
    worker = SimpleNamespace(page_size=1, use_compact_draft_cache=False, block_size=16,
        draft_model=draft.model, model_runner=target, draft_model_runner=draft, device='cpu',
        coefficient=2., ingestion_calls=[])

    def ingest(batch, info):
        end = int(batch.seq_lens[0])
        start = end-len(info.target_hidden)
        positions = pool.req_to_token[0, start:end].long()
        worker.ingestion_calls.append((start, end, batch.forward_mode.is_extend()))
        for key, value in draft.token_to_kv_pool.rows:
            key[positions] = info.target_hidden*worker.coefficient
            value[positions] = info.target_hidden*worker.coefficient+1
        info.draft_seq_lens = batch.seq_lens.clone()
        info.ctx_lens.zero_()
        info.target_hidden = info.target_hidden[:0]

    worker._append_target_hidden_to_draft_kv = ingest
    worker.forward_batch_generation = lambda batch, **kwargs: None
    batch = SimpleNamespace(reqs=[Req()], req_pool_indices=torch.tensor([0]),
        seq_lens=torch.tensor([5], dtype=torch.int32), seq_lens_cpu=torch.tensor([5], dtype=torch.int32),
        spec_info=SimpleNamespace(draft_seq_lens=torch.tensor([5],dtype=torch.int32),
            ctx_lens=torch.tensor([0]),target_hidden=torch.empty((0,2)),verified_id=torch.tensor([22])),
        sampling_info=SimpleNamespace(is_all_greedy=True), forward_mode=Mode(),is_extend_in_batch=False,
        req_to_token_pool=pool)
    segments = [Ingestion(0,3,torch.arange(6,dtype=torch.float32).reshape(3,2),Mode(True),True,0),
                Ingestion(3,5,torch.arange(6,10,dtype=torch.float32).reshape(2,2),Mode(),False,3)]
    for segment in segments:
        ingest(SimpleNamespace(seq_lens=torch.tensor([segment.end]),forward_mode=segment.forward_mode),
            SimpleNamespace(target_hidden=segment.hidden.clone(),ctx_lens=torch.tensor([segment.end-segment.start])))
    worker.ingestion_calls.clear()
    return SimpleNamespace(draft_worker=worker), batch, segments


class NativeSuccessorTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.budget = WorkBudget(Path(self.directory.name)/'work.jsonl')
        self.scheduler, self.batch, segments = fixture()
        request = dict(request_id='request',prompt_ids=[10,11,12],max_new_tokens=8)
        self.capture = NativeSuccessors(self.scheduler,request,'KPLUS4',[1],self.budget)
        self.capture.ingestions = segments
        self.capture.round_index = 1
        self.capture.capture(self.batch)
        self.state = self.capture.successors[1]
        self.capture.install()
        self.addCleanup(self.capture.close)

    def test_neutral_constructor_really_rebuilds_cache_with_original_chunks(self):
        worker = self.scheduler.draft_worker
        for cache in [worker.model_runner.token_to_kv_pool,worker.draft_model_runner.token_to_kv_pool]:
            for key,value in cache.rows:
                key.fill_(999);value.fill_(999)
        rebuilt = self.capture.rebuild(self.state,'KPLUS4')
        self.assertEqual(worker.ingestion_calls,[(0,3,True),(3,5,False)])
        self.assertTrue(self.capture.reconstruction['neutral_draft_prefix_cache_exact'])
        self.assertTrue(self.capture.reconstruction['target_prefix_cache_exact'])
        self.assertEqual(rebuilt.reqs[0].sampling_params.max_new_tokens,8)
        self.assertEqual(self.capture.reconstruction['remaining_budget'],5)
        self.assertEqual(self.budget.summary()['completed']['context_forward'],2)

    def test_crossed_cache_uses_destination_weights_and_preserves_source_target(self):
        worker=self.scheduler.draft_worker
        worker.coefficient=3.
        self.capture.rebuild(self.state,'KPLUS6')
        self.assertTrue(self.capture.reconstruction['target_prefix_cache_exact'])
        self.assertIsNone(self.capture.reconstruction['neutral_draft_prefix_cache_exact'])
        self.assertFalse(self.capture.reconstruction['source_draft_cache_transplanted'])
        expected=torch.arange(10,dtype=torch.float32).reshape(5,2)*3
        self.assertTrue(torch.equal(worker.draft_model_runner.token_to_kv_pool.rows[0][0][:5],expected))
        self.assertFalse(torch.equal(self.state.draft_reference_cache[0][0],expected))

    def test_snapshot_owns_mutable_request_and_allocator_values(self):
        self.batch.reqs[0].output_ids.append(23)
        self.batch.spec_info.verified_id.fill_(99)
        self.scheduler.draft_worker.model_runner.token_to_kv_pool_allocator.state[0].zero_()
        self.assertEqual(self.state.output_ids,[20,21,22])
        self.assertEqual(self.state.batch.spec_info.verified_id.item(),22)
        self.assertEqual(self.state.allocator[0][0].item(),6)

    def test_neutral_wrong_ingestion_fails_bitwise_cache_gate(self):
        self.scheduler.draft_worker.coefficient=3.
        with self.assertRaisesRegex(ValueError,'bitwise'):
            self.capture.rebuild(self.state,'KPLUS4')

    def test_ingestion_gap_is_rejected_before_native_context_call(self):
        self.capture.ingestions=[]
        self.capture.round_index=0
        info=SimpleNamespace(target_hidden=torch.ones(2,2),ctx_lens=torch.tensor([2]),draft_seq_lens=torch.tensor([0]))
        with self.assertRaisesRegex(ValueError,'gap'):
            self.scheduler.draft_worker._append_target_hidden_to_draft_kv(self.batch,info)
        self.assertEqual(self.scheduler.draft_worker.ingestion_calls,[])

    def test_cap_change_is_not_accepted_as_successor(self):
        self.batch.reqs[0].sampling_params.max_new_tokens=2048
        with self.assertRaisesRegex(ValueError,'cap'):
            self.capture.capture(self.batch)


class NativeParityAndBudgetTests(unittest.TestCase):
    def test_terminal_clipping_preserves_ordinary_acceptance_convention(self):
        blocks = [dict(accepted_draft_tokens=3, native_accepted_draft_tokens=3,
                       committed_progress=4, finished=False),
                  dict(accepted_draft_tokens=2, native_accepted_draft_tokens=1,
                       committed_progress=2, finished=True)]
        self.assertEqual(acceptance_totals(blocks),
                         dict(accepted_draft_tokens=5, native_accepted_draft_tokens=4))

    def test_neutral_gate_requires_proposals_not_only_committed_tokens(self):
        block=dict(draft_tokens=[1]*15,committed_progress=1,committed_tokens=[2],
            positions=list(range(16)),target_argmax=[2]*16,finished=True,output_start=2)
        original=dict(response_ids=[1,1,1,2],finish_reason='eos',metadata=dict(trace=[block,block]))
        replay=dict(response_ids=original['response_ids'],finish_reason='eos',metadata=dict(trace=[dict(block)]))
        self.assertEqual(assert_neutral_suffix(original,dict(round_index=1),replay)['status'],'passed')
        replay['metadata']['trace'][0]['draft_tokens']=[3]*15
        with self.assertRaisesRegex(ValueError,'suffix differs'):
            assert_neutral_suffix(original,dict(round_index=1),replay)

    def test_failed_model_call_stays_charged_and_no_overrun_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            budget=WorkBudget(Path(directory)/'work.jsonl',limits={'draft_block':1})
            def fail():
                raise RuntimeError('native failure')
            with self.assertRaises(RuntimeError):run_counted(budget,'draft_block',fail)
            launched=[]
            with self.assertRaises(BudgetExceeded):run_counted(budget,'draft_block',lambda:launched.append(True))
            self.assertEqual(launched,[])
            self.assertEqual(budget.summary()['failed']['draft_block'],1)
            self.assertFalse(budget.summary()['unknown'])

    def test_startup_batch_admission_counts_each_proposal_block(self):
        from experiments.verification_transition import native_worker
        with tempfile.TemporaryDirectory() as directory:
            budget=WorkBudget(Path(directory)/'work.jsonl',limits={'draft_block':4})
            previous=native_worker._BUDGET
            native_worker._BUDGET=budget
            self.addCleanup(setattr,native_worker,'_BUDGET',previous)
            batch=SimpleNamespace(batch_size=4,forward_mode=SimpleNamespace(is_target_verify=lambda:True))
            result=native_worker._startup_model_call(lambda model,batch:7,None,(batch,),{},draft=True)
            self.assertEqual(result,7)
            self.assertEqual(budget.summary()['completed']['draft_block'],4)
            self.assertNotIn('target_call',budget.summary()['attempted'])

    def test_startup_admission_failure_does_not_launch_partial_batch(self):
        from experiments.verification_transition import native_worker
        with tempfile.TemporaryDirectory() as directory:
            budget=WorkBudget(Path(directory)/'work.jsonl',limits={'draft_block':2})
            previous=native_worker._BUDGET
            native_worker._BUDGET=budget
            self.addCleanup(setattr,native_worker,'_BUDGET',previous)
            batch=SimpleNamespace(batch_size=4,forward_mode=SimpleNamespace(is_target_verify=lambda:True))
            called=[]
            with self.assertRaises(BudgetExceeded):
                native_worker._startup_model_call(lambda model,batch:called.append(True),None,(batch,),{},draft=True)
            self.assertEqual(called,[])
            self.assertEqual(budget.summary()['failed']['draft_block'],2)
            self.assertFalse(budget.summary()['unknown'])


if __name__=='__main__':
    unittest.main()
