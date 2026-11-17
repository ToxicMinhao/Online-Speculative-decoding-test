"""A fake native runtime tests the actual replay control path without a GPU.

The fake models canonical cache truncation, rejected-history features, EOS
clipping and the verifier's destructive hidden-state return convention. Native
numerical equality still requires the separately admitted two-packet check.
"""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from experiments.survival_head.feedback_replay import check_cache_prefix, reconstruct
from online_feedback import sglang_hooks as hooks
from test_feedback import source_record


class Mode:
    def __init__(self, value):
        self.value = value

    def is_target_verify(self):
        return self.value == 'verify'


class FakeSampling(SimpleNamespace):
    def normalize(self, tokenizer):
        self.stop_strs = []
        self.stop_regex_strs = []


class FakeReq(SimpleNamespace):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.output_ids = []
        self.ended = False
        self.kv_committed_len = self.kv_allocated_len = 0

    def check_finished(self):
        self.ended = bool(self.output_ids and (
            self.output_ids[-1] in self.eos_token_ids
            or len(self.output_ids) == self.sampling_params.max_new_tokens))

    def finished(self):
        return self.ended


class FakeBatch(SimpleNamespace):
    @classmethod
    def init_new(cls, **kwargs):
        value = cls(**kwargs)
        value.req_pool_indices = torch.tensor([0])
        return value

    def prepare_for_extend(self):
        req = self.reqs[0]
        self.seq_lens = torch.tensor([len(req.origin_input_ids)], dtype=torch.int32)
        self.seq_lens_cpu = self.seq_lens.clone()
        self.seq_lens_sum = len(req.origin_input_ids)
        self.input_ids = torch.tensor(req.origin_input_ids)
        self.extend_prefix_lens = [0]
        self.extend_seq_lens = [len(req.origin_input_ids)]
        self.forward_mode = Mode('extend')
        self.req_to_token_pool.req_to_token[0, :self.seq_lens_sum] = torch.arange(self.seq_lens_sum)
        req.kv_committed_len = req.kv_allocated_len = self.seq_lens_sum

    def get_model_worker_batch(self):
        return self


class FakeVerify(SimpleNamespace):
    def prepare_for_verify(self, batch, page_size, *, build_custom_mask):
        assert page_size == 1 and not build_custom_mask
        batch.input_ids = self.draft_token
        start = int(batch.seq_lens[0])
        batch.out_cache_loc = torch.arange(start, start + 16)
        batch.req_to_token_pool.req_to_token[0, start:start + 16] = batch.out_cache_loc

    def verify(self, *, batch, logits_output, page_size):
        posterior = logits_output.next_token_logits.argmax(-1).tolist()
        candidates = self.draft_token.tolist()
        accepted_end = next((i for i in range(1, 16) if candidates[i] != posterior[i - 1]), 16)
        req = batch.reqs[0]
        committed = 0
        for token in candidates[1:accepted_end] + [posterior[accepted_end - 1]]:
            req.output_ids.append(token)
            committed += 1
            req.check_finished()
            if req.finished():
                break
        next_aux = logits_output.hidden_states[:committed]
        logits_output.hidden_states = None
        req.kv_committed_len += committed
        req.kv_allocated_len = req.kv_committed_len
        batch.seq_lens += committed
        batch.seq_lens_cpu += committed
        batch.seq_lens_sum += committed
        batch.out_cache_loc = batch.out_cache_loc[:committed]
        return torch.tensor(req.output_ids[-1:]), torch.tensor([committed]), next_aux, [committed - 1]


class FakeTarget:
    def __init__(self, record, *, posterior_mismatch=False, cache_corruption=False):
        self.record = record
        self.calls = []
        self.posterior_mismatch = posterior_mismatch
        self.cache_corruption = cache_corruption
        self.pool = SimpleNamespace(req_to_token=torch.zeros((1, 4096), dtype=torch.int64))
        self.model_runner = SimpleNamespace(
            device='cpu', req_to_token_pool=self.pool,
            model_config=SimpleNamespace(vocab_size=256, context_len=4096),
            token_to_kv_pool_allocator=object(),
            server_args=SimpleNamespace(page_size=1), attn_backend=object())

    def forward_batch_generation(self, batch, *, is_verify=False):
        ids = batch.input_ids.tolist()
        if not is_verify:
            assert len(self.calls) == 0
            prefix, start = [], 0
            posterior = self.record['response_ids'][:1]
        else:
            index = len(self.calls) - 1
            block = self.record['metadata']['trace'][index]
            start = len(self.record['prompt_ids']) + block['output_start']
            assert ids == [block['anchor_token']] + block['draft_tokens']
            assert batch.spec_info.positions.tolist() == list(range(start, start + 16))
            assert int(batch.seq_lens[0]) == start
            prefix = self.record['prompt_ids'] + self.record['response_ids'][:block['output_start']]
            posterior = list(block['target_argmax'])
            if self.posterior_mismatch and index == 0:
                posterior[-1] += 1  # discarded posterior must still match exactly
            if self.cache_corruption and index == 0:
                batch.req_to_token_pool.req_to_token[0, 0] = 999
        self.calls.append(dict(tokens=ids, start=start, verify=is_verify))
        rows = []
        for offset, token in enumerate(ids):
            prefix.append(token)
            rows.append([start + offset, sum(prefix)])
        hidden = torch.tensor(rows, dtype=torch.bfloat16)
        logits = torch.zeros((len(posterior), 256))
        logits[torch.arange(len(posterior)), torch.tensor(posterior)] = 1
        return SimpleNamespace(next_token_ids=torch.tensor(posterior[:1]),
            logits_output=SimpleNamespace(hidden_states=torch.cat([hidden, hidden + 100], dim=-1),
                teacher_hidden_states=hidden, next_token_logits=logits))


def modules():
    return {
        'sglang.srt.managers.schedule_batch': SimpleNamespace(Req=FakeReq, ScheduleBatch=FakeBatch),
        'sglang.srt.mem_cache.cache_init_params': SimpleNamespace(CacheInitParams=SimpleNamespace),
        'sglang.srt.mem_cache.radix_cache': SimpleNamespace(RadixCache=lambda args: args),
        'sglang.srt.model_executor.forward_batch_info': SimpleNamespace(
            CaptureHiddenMode=SimpleNamespace(FULL='full'),
            ForwardMode=SimpleNamespace(TARGET_VERIFY=Mode('verify'), DECODE=Mode('decode'))),
        'sglang.srt.sampling.sampling_params': SimpleNamespace(SamplingParams=FakeSampling),
        'sglang.srt.speculative.spec_info': SimpleNamespace(SpeculativeAlgorithm=SimpleNamespace(DFLASH='DFLASH')),
        'sglang.srt.speculative.dflash_info': SimpleNamespace(DFlashVerifyInput=FakeVerify),
        'sglang.srt.speculative.dflash_utils': SimpleNamespace(resolve_dflash_verify_mask_policy=lambda backend: (None, False)),
        'sglang.srt.utils': SimpleNamespace(require_mlp_sync=lambda args: False),
    }


class NativeReplayTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / 'replay.json'

    def run_replay(self, *, posterior_mismatch=False, cache_corruption=False, cleanup_failure=False):
        self.record = source_record()
        self.target = FakeTarget(self.record, posterior_mismatch=posterior_mismatch,
                                 cache_corruption=cache_corruption)
        scheduler = SimpleNamespace(tp_worker=self.target,
            server_args=SimpleNamespace(tp_size=1, pp_size=1, page_size=1),
            draft_worker=SimpleNamespace(block_size=16, use_compact_draft_cache=False))
        cleanup = [0.01, RuntimeError('cleanup failed')] if cleanup_failure else [0.01, 0.01]
        with patch.dict('sys.modules', modules()), patch.object(hooks, '_ACTIVE', None), \
             patch.object(hooks, 'flush_idle', side_effect=cleanup), patch.object(torch.cuda, 'synchronize'):
            return reconstruct(scheduler, self.record, str(self.path))

    def test_exact_saved_walk_retains_only_correctly_conditioned_rows(self):
        result = self.run_replay()
        self.assertEqual(result['status'], 'passed')
        self.assertEqual([r['start'] for r in self.target.calls], [0, 2, 5])
        self.assertEqual(result['prefill_forwards'], 1)
        self.assertEqual(result['verification_forwards'], 2)
        self.assertEqual(result['terminal_forwards'], 0)
        self.assertEqual(result['draft_forwards'], 0)
        self.assertEqual(result['new_generations'], 0)
        self.assertEqual(result['posterior_positions_checked'], 32)
        self.assertEqual(result['cache_prefix_checks'], 5)
        capture = result['record']['metadata']['capture']
        self.assertEqual(capture['blocks'], self.record['metadata']['trace'])
        packet = torch.load(capture['feature_path'], weights_only=False)
        tokens = (self.record['prompt_ids'] + self.record['response_ids'])[:-1]
        expected = torch.tensor([[i, sum(tokens[:i + 1])] for i in range(len(tokens))], dtype=torch.bfloat16)
        self.assertTrue(torch.equal(packet['target_last_hidden_states'], expected))
        self.assertTrue(torch.equal(packet['hidden_states'], torch.cat([expected, expected + 100], -1)))
        self.assertEqual(self.record['metadata']['capture']['feature_path'], '/retired/original/native_capture.pt')

    def test_rejected_suffix_posterior_mismatch_stops_before_further_target_work(self):
        with self.assertRaisesRegex(RuntimeError, 'full verification posterior differs'):
            self.run_replay(posterior_mismatch=True)
        result = json.loads(self.path.read_text())
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(len(self.target.calls), 2)
        self.assertEqual(result['verification_attempts'], 1)
        self.assertEqual(result['verification_forwards'], 1)
        self.assertEqual(result['emitted_prefix'], [10])
        self.assertFalse(self.path.with_suffix('.capture.pt').exists())

    def test_committed_cache_mapping_mutation_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, 'previously committed cache mapping'):
            self.run_replay(cache_corruption=True)
        result = json.loads(self.path.read_text())
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(len(self.target.calls), 2)

    def test_cleanup_failure_changes_receipt_to_failed(self):
        with self.assertRaisesRegex(RuntimeError, 'cleanup failed'):
            self.run_replay(cleanup_failure=True)
        result = json.loads(self.path.read_text())
        self.assertEqual(result['status'], 'failed')
        self.assertIn('cleanup failed', result['cleanup_error'])

    def test_failure_during_cleanup_preserves_original_posterior_error(self):
        with self.assertRaisesRegex(RuntimeError, 'full verification posterior differs'):
            self.run_replay(posterior_mismatch=True, cleanup_failure=True)
        result = json.loads(self.path.read_text())
        self.assertIn('posterior differs', result['error'])
        self.assertIn('cleanup failed', result['cleanup_error'])

    def test_existing_artifact_is_preserved(self):
        self.path.write_text('original')
        with self.assertRaises(FileExistsError):
            self.run_replay()
        self.assertEqual(self.path.read_text(), 'original')
        self.assertEqual(self.target.calls, [])


if __name__ == '__main__':
    unittest.main()
