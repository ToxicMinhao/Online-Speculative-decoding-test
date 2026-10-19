"""Native-wrapper CPU regressions, not GPU integration qualification."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from . import native_hooks as hooks


def factors(d=4, rank=0):
    return SimpleNamespace(B=np.zeros((d, rank), np.float32), A=np.zeros((rank, d), np.float32))


def context(*, effective=None):
    seen = []
    bank = SimpleNamespace(version=0)
    bank.identity = lambda: bank.version
    def read(descriptor, *, context_length):
        seen.append((descriptor.copy(), context_length))
        return SimpleNamespace(alpha=np.array([1., 0., 0., 0.]), effective=effective or factors(),
                               forecast=.25, diagnostics={'test': True})
    bank.read = read
    projection = np.zeros((4, 64), np.float32)
    projection[:2, :2] = np.eye(2)
    correction = hooks.RequestCorrection({'arm': 'BANK', 'state_sha256': 'a' * 64},
        {'V_descriptor': projection}, active_bank={'bank': bank, 'state_sha256': 'a' * 64})
    return correction, seen


def test_zero_returns_exact_original_tensor():
    h = torch.randn(9, 4, dtype=torch.bfloat16)
    assert hooks.corrected_features(h, torch.empty(4, 0), torch.empty(0, 4)) is h
    assert hooks.corrected_features(h, torch.zeros(4, 2), torch.randn(2, 4)) is h


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_fp32_residual_then_native_cast_and_pair_concatenation(dtype):
    generator = torch.Generator().manual_seed(17)
    h = torch.randn(9, 8, generator=generator).to(dtype)
    b1, b2 = torch.randn(8, 3, generator=generator), torch.randn(8, 3, generator=generator)
    a1, a2 = torch.randn(3, 8, generator=generator), torch.randn(3, 8, generator=generator)
    B, A = torch.cat([.5 * b1, .5 * b2], dim=1), torch.cat([a1, a2], dim=0)
    expected = (h.float() + (h.float() @ A.T) @ B.T).to(dtype)
    actual = hooks.corrected_features(h, B, A)
    assert torch.equal(actual, expected)
    explicit = (h.float() + .5 * (h.float() @ a1.T) @ b1.T + .5 * (h.float() @ a2.T) @ b2.T).to(dtype)
    torch.testing.assert_close(actual, explicit, atol=2e-5 if dtype == torch.float32 else .125,
                               rtol=2e-5 if dtype == torch.float32 else .015625)


def test_rank64_pair_not_truncated_and_no_feature_clipping():
    h = torch.eye(64) * 100
    actual = hooks.corrected_features(h, torch.eye(64), torch.eye(64))
    assert torch.equal(actual, h * 2)


def test_prefill_routes_once_from_last_prompt_and_keeps_factors_fixed(tmp_path):
    f = factors(rank=1); f.B[0, 0] = 1.; f.A[0, 0] = .1
    correction, seen = context(effective=f)
    path = tmp_path / 'forecast.json'
    correction.prefill(torch.ones(8192, 4), start=0, prompt_length=8431, forecast_path=path)
    assert not path.exists() and not seen
    with pytest.raises(RuntimeError, match='before a causal prompt read'):
        correction.apply(torch.ones(9, 4))
    tail = torch.ones(239, 4); tail[-1] = torch.tensor([3., 4., 9., 8.])
    correction.prefill(tail, start=8192, prompt_length=8431, forecast_path=path)
    np.testing.assert_allclose(seen[0][0][:2], [.6, .8], atol=1e-7)
    assert seen[0][1] == 8431 and len(seen) == 1
    h = torch.ones(9, 4)
    expected = correction.apply(h)
    f.B[:] = 99  # Native compilation owns its factors, no shared-state view.
    torch.testing.assert_close(correction.apply(h), expected)
    assert json.loads(path.read_text())['forecast_saved_before_response']
    with pytest.raises(RuntimeError, match='after read was fixed'):
        correction.prefill(torch.ones(1, 4), start=8431, prompt_length=8432, forecast_path=path)
    assert correction.finish()['coefficients'] == [1., 0., 0., 0.]
    correction.bank.version = 1
    with pytest.raises(RuntimeError, match='Serving mutated'):
        correction.finish()


@pytest.mark.parametrize('start,prompt,message', [(1, 2, 'not contiguous'), (0, 1, 'generated response')])
def test_rejects_prefix_gaps_and_response_descriptor_leakage(tmp_path, start, prompt, message):
    correction, _ = context()
    with pytest.raises(RuntimeError, match=message):
        correction.prefill(torch.ones(2, 4), start=start, prompt_length=prompt, forecast_path=tmp_path / 'f.json')


def install_stub(tmp_path):
    class Worker:
        tp_size = 1
        use_compact_draft_cache = False
        def __init__(self):
            self.forward_calls, self.head_calls = [], []
        def forward_batch_generation(self, batch, **kwargs):
            self.forward_calls.append((batch.spec_info, kwargs))
            features = batch.features
            batch.spec_info = SimpleNamespace(ctx_lens=torch.zeros(1, dtype=torch.int32))
            batch.extend_lens[0] = 0; batch.prefix_lens[0] = -1
            return SimpleNamespace(logits_output=SimpleNamespace(teacher_hidden_states=features), marker='native')
        def _greedy_sample_from_vocab_parallel_head(self, *, hidden_states, lm_head, chunk_size=256):
            self.head_calls.append((hidden_states, lm_head, chunk_size))
            return torch.argmax(hidden_states @ lm_head.T, dim=-1)
    class Scheduler:
        pass
    names = ('sglang', 'sglang.srt', 'sglang.srt.speculative', 'sglang.srt.managers',
             'sglang.srt.speculative.dflash_worker', 'sglang.srt.managers.scheduler')
    modules = {name: ModuleType(name) for name in names}
    modules[names[-2]].DFlashWorker = Worker
    modules[names[-1]].Scheduler = Scheduler
    entry = SimpleNamespace(RequestTrace=type('Trace', (), {}), _active_trace=None,
                            inherited=SimpleNamespace(strict_idle=lambda _: None))
    with patch.dict(sys.modules, modules):
        hooks.install(entry, {'worker_out': str(tmp_path)}, projections={}, bank_factory=lambda _: None)
    return entry, Worker(), Scheduler


def batch(start, length, *, rows=None):
    return SimpleNamespace(forward_mode=SimpleNamespace(is_extend=lambda: True), is_extend_in_batch=False,
        batch_size=lambda: 1, extend_lens=[length], prefix_lens=[start], spec_info=None,
        features=torch.ones(length if rows is None else rows, 4))


def test_installed_wrapper_copies_geometry_before_runtime_mutation(tmp_path):
    entry, worker, _ = install_stub(tmp_path)
    correction, seen = context()
    entry._active_trace = SimpleNamespace(bank_correction=correction, input_ids=[1] * 8431,
                                         bank_forecast_path=tmp_path / 'f.json')
    first = batch(0, 8192)
    assert worker.forward_batch_generation(first, marker='preserved').marker == 'native'
    assert worker.forward_calls[0] == (None, {'marker': 'preserved'})
    assert first.spec_info.ctx_lens.tolist() == [0] and not seen
    second = batch(8192, 239); second.features[-1] = torch.tensor([3., 4., 9., 8.])
    worker.forward_batch_generation(second)
    assert len(seen) == 1 and correction.prompt_seen == 8431
    h, E = torch.ones(9, 4), torch.eye(4)
    worker._greedy_sample_from_vocab_parallel_head(hidden_states=h, lm_head=E, chunk_size=9)
    assert worker.head_calls[0][0] is h and worker.head_calls[0][1] is E
    assert worker.head_calls[0][2] == 9


def test_installed_wrapper_rejects_misreported_chunk_rows(tmp_path):
    entry, worker, _ = install_stub(tmp_path)
    correction, seen = context()
    entry._active_trace = SimpleNamespace(bank_correction=correction, input_ids=[1] * 8431,
                                         bank_forecast_path=tmp_path / 'f.json')
    with pytest.raises(RuntimeError, match='actual native prefill length'):
        worker.forward_batch_generation(batch(0, 8192, rows=8191))
    assert not seen


def test_other_methods_do_not_require_descriptor_geometry_or_change_head(tmp_path):
    entry, worker, _ = install_stub(tmp_path)
    entry._active_trace = SimpleNamespace(bank_correction=hooks.RequestCorrection({'arm': 'OSD'}, None))
    assert worker.forward_batch_generation(batch(0, 2)).marker == 'native'
    h, E = torch.ones(9, 4), torch.eye(4)
    worker._greedy_sample_from_vocab_parallel_head(hidden_states=h, lm_head=E)
    assert worker.head_calls[0][0] is h


def test_canonical_cycle_context_uses_actual_request_prefix_before_forward(tmp_path):
    entry, worker, _ = install_stub(tmp_path)
    trace = SimpleNamespace(bank_correction=hooks.RequestCorrection({'arm': 'F'}, None),
        input_ids=[10, 11], bank_head_capture=hooks.FrozenHeadCapture(enabled=True),
        bank_cycle_observations=[], committed=[999])  # Stale observer length must be irrelevant.
    entry._active_trace = trace
    value = batch(0, 2)
    value.forward_mode = SimpleNamespace(is_extend=lambda: False)
    value.reqs = [SimpleNamespace(output_ids=[20, 21, 22])]
    with patch.object(torch.cuda, 'is_available', return_value=False):
        worker.forward_batch_generation(value)
    assert trace.bank_current_context_length == 5
    assert trace.bank_cycle_observations[0]['context_length'] == 5


def trajectory():
    return dict(input_token_ids=[10, 11], final_output_token_ids=[20, 21, 22], events=[dict(
        kind='verification', context={'verification_input_start_position': 2}, output_offset=1,
        draft_proposed_token_ids=list(range(9)), draft_proposed_positions=list(range(3, 12)),
        native_acceptance_length_before_stop=1, committed_token_ids=[21, 22], finished=True, stop_clipped=False)])


def test_frozen_head_capture_matches_exact_native_blocks_and_owns_features(tmp_path):
    recorder = hooks.FrozenHeadCapture(enabled=True)
    h = torch.ones(9, 4, dtype=torch.bfloat16)
    recorder.append(h, torch.arange(9)); h[:] = 9
    result = recorder.finish(trajectory(), tmp_path / 'heads.npz')
    document = json.loads(Path(result['path']).read_text())
    block = document['blocks'][0]
    assert block['block_start'] == 0 and block['absolute_block_start'] == 2
    assert block['output_positions'] == list(range(1, 10))
    assert block['eligible_mask'] == [True, True] + [False] * 7
    assert block['canonical_token_ids'] == [21, 22] + [None] * 7
    with np.load(document['tensors']['path']) as arrays:
        np.testing.assert_array_equal(arrays['frozen_h'], np.ones((1, 9, 4), np.float32))
    assert not recorder.hidden and not recorder.proposals


def test_capture_disabled_replay_does_no_tensor_work_or_artifact_write(tmp_path):
    recorder = hooks.FrozenHeadCapture(enabled=False)
    recorder.append(None, None)
    assert recorder.finish(None, tmp_path / 'unused') is None
    assert recorder.bytes == 0 and not list(tmp_path.iterdir())


def test_capture_rejects_proposal_mismatch_and_preserves_existing_evidence(tmp_path):
    recorder = hooks.FrozenHeadCapture(enabled=True)
    recorder.append(torch.ones(9, 4, dtype=torch.bfloat16), torch.arange(9) + 1)
    with pytest.raises(RuntimeError, match='proposals differ'):
        recorder.finish(trajectory(), tmp_path / 'heads.npz')
    assert not (tmp_path / 'heads.npz').exists()


def test_single_token_answer_has_explicit_empty_head_capture(tmp_path):
    result = hooks.FrozenHeadCapture(enabled=True).finish(
        dict(input_token_ids=[10], final_output_token_ids=[20], events=[]), tmp_path / 'empty.npz')
    doc = json.loads(Path(result['path']).read_text())
    assert doc['blocks'] == []


def test_profile_cannot_run_in_main_campaign(tmp_path):
    _, _, Scheduler = install_stub(tmp_path)
    with pytest.raises(RuntimeError, match='qualification-only'):
        Scheduler.correction_bank_profile(Scheduler())
