"""Focused CPU checks for native correction placement, causality and accounting."""
from __future__ import annotations

from pathlib import Path
import json
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from . import native, native_hooks as hooks

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'Torch CPU is required for feature hook checks')
class CorrectionFeatures(unittest.TestCase):
    def test_zero_keeps_original_tensor_and_values(self):
        h = torch.randn(9, 8, dtype=torch.bfloat16)
        result = hooks.corrected_features(h, torch.zeros(2, 2), torch.randn(8, 2), torch.randn(8, 2))
        self.assertIs(result, h)
        self.assertTrue(torch.equal(result, h))

    def test_compiled_fusion_matches_explicit_same_boundary(self):
        generator = torch.Generator().manual_seed(17)
        h = torch.randn(9, 8, generator=generator)
        U = torch.randn(8, 2, generator=generator)
        V = torch.randn(8, 2, generator=generator)
        matrices = torch.randn(4, 2, 2, generator=generator)
        alpha = torch.tensor([.7, 0., .8, 0.])  # Sum may exceed one.
        compiled = hooks.corrected_features(h, torch.einsum('k,kij->ij', alpha, matrices), U, V)
        raw = h@V
        phi = raw/raw.norm(dim=-1, keepdim=True).clamp_min(1.)
        explicit = h+sum(alpha[i]*(phi@matrices[i].T)@U.T for i in range(4))
        self.assertTrue(torch.allclose(compiled, explicit, atol=2e-6, rtol=2e-6))

    def context(self):
        seen = []
        bank = SimpleNamespace(matrices=np.zeros((4, 2, 2)), allocated=np.zeros(4, dtype=bool),
                               feedback=[], protection=[], write_log=[])
        def read(descriptor):
            seen.append(descriptor.copy())
            return SimpleNamespace(alpha=np.zeros(4), effective=np.zeros((2, 2)),
                forecast=np.zeros((2, 2)), diagnostics={'cold_start': True})
        bank.read = read
        context = hooks.RequestCorrection({'arm': 'BANK', 'state_sha256': 'a'*64},
            {'U': np.eye(4, 2), 'V_in': np.eye(4, 2), 'V_descriptor': np.eye(4, 2)}, None,
            active_bank={'bank': bank, 'state_sha256': 'a'*64})
        return context, seen

    def test_descriptor_after_complete_chunks_before_any_draft(self):
        context, seen = self.context()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'forecast.json'
            context.prefill(torch.ones(8192, 4), start=0, prompt_length=8431, forecast_path=path)
            self.assertEqual(seen, [])
            self.assertFalse(path.exists())
            with self.assertRaisesRegex(RuntimeError, 'before a causal prompt read'):
                context.apply(torch.ones(9, 4))
            tail = torch.ones(239, 4)
            tail[-1] = torch.tensor([3., 4., 8., 9.])
            context.prefill(tail, start=8192, prompt_length=8431, forecast_path=path)
            self.assertTrue(path.is_file())
            np.testing.assert_allclose(seen[0], [.6, .8], atol=1e-7)
            h = torch.ones(9, 4)
            self.assertIs(context.apply(h), h)
            self.assertIs(context.finish(), context.forecast)
            self.assertTrue(context.forecast['cost_in_request_s'])
            with self.assertRaisesRegex(RuntimeError, 'after read was fixed'):
                context.prefill(tail, start=8431, prompt_length=8431, forecast_path=path)

    def test_gap_or_generated_positions_rejected(self):
        context, _ = self.context()
        with self.assertRaisesRegex(RuntimeError, 'not contiguous'):
            context.prefill(torch.ones(2, 4), start=1, prompt_length=2, forecast_path='unused')
        context, _ = self.context()
        with self.assertRaisesRegex(RuntimeError, 'generated response'):
            context.prefill(torch.ones(3, 4), start=0, prompt_length=2, forecast_path='unused')

    def test_serving_never_accepts_mutated_bank(self):
        context, _ = self.context()
        with tempfile.TemporaryDirectory() as directory:
            context.prefill(torch.ones(2, 4), start=0, prompt_length=2,
                            forecast_path=Path(directory)/'forecast.json')
            context.bank.matrices[0, 0, 0] = 1.
            with self.assertRaisesRegex(RuntimeError, 'mutated'):
                context.finish()


@unittest.skipIf(torch is None, 'Torch CPU is required for installed hook checks')
class InstalledForwardWrapper(unittest.TestCase):
    """Exercise install() itself against the native prefill consumption contract."""
    def install_stub(self, directory):
        class StubDFlashWorker:
            tp_size = 1
            use_compact_draft_cache = False

            def __init__(self):
                self.forward_calls = []
                self.head_calls = []

            def forward_batch_generation(self, batch, **kwargs):
                # The first real prefill may have no speculative input yet.
                self.forward_calls.append(dict(spec_info_before=batch.spec_info, kwargs=kwargs))
                features = batch.features
                # Pinned native forward creates a new DFlashDraftInput, appends
                # its context to KV, and exposes that *consumed* state on batch.
                draft_input = SimpleNamespace(ctx_lens=torch.tensor(batch.extend_lens), target_hidden=features)
                draft_input.ctx_lens.zero_()
                draft_input.target_hidden = draft_input.target_hidden[:0]
                batch.spec_info = draft_input
                # Deliberately invalidate the mutable geometry too: the wrapper
                # must hold scalar values captured before native execution.
                batch.extend_lens[0] = 0
                batch.prefix_lens[0] = -1
                return SimpleNamespace(logits_output=SimpleNamespace(teacher_hidden_states=features),
                                       native_marker='unchanged')

            def _greedy_sample_from_vocab_parallel_head(self, *, hidden_states, lm_head, chunk_size=256):
                self.head_calls.append((hidden_states, lm_head, chunk_size))
                return torch.argmax(hidden_states @ lm_head.T, dim=-1)

        class StubScheduler:
            pass

        names = ('sglang', 'sglang.srt', 'sglang.srt.speculative', 'sglang.srt.managers',
                 'sglang.srt.speculative.dflash_worker', 'sglang.srt.managers.scheduler')
        modules = {name: ModuleType(name) for name in names}
        modules['sglang.srt.speculative.dflash_worker'].DFlashWorker = StubDFlashWorker
        modules['sglang.srt.managers.scheduler'].Scheduler = StubScheduler
        patcher = patch.dict(sys.modules, modules)
        patcher.start(); self.addCleanup(patcher.stop)
        entry = SimpleNamespace(RequestTrace=type('UncalledTrace', (), {}), _active_trace=None,
                                inherited=SimpleNamespace(strict_idle=lambda _: None))
        hooks.install(entry, {'worker_out': str(directory)}, projections={}, bank_factory=lambda _: None)
        return entry, StubDFlashWorker()

    def context(self):
        observed = []
        bank = SimpleNamespace(matrices=np.zeros((4, 2, 2)), allocated=np.zeros(4, dtype=bool),
                               feedback=[], protection=[], write_log=[])
        def read(descriptor):
            observed.append(descriptor.copy())
            return SimpleNamespace(alpha=np.zeros(4), effective=np.zeros((2, 2)),
                                   forecast=np.zeros((2, 2)), diagnostics={'cold_start': True})
        bank.read = read
        context = hooks.RequestCorrection({'arm': 'BANK', 'state_sha256': 'a'*64},
            {'U': np.eye(4, 2), 'V_in': np.eye(4, 2), 'V_descriptor': np.eye(4, 2)}, None,
            active_bank={'bank': bank, 'state_sha256': 'a'*64})
        return context, observed

    def batch(self, start, length, *, feature_rows=None):
        return SimpleNamespace(forward_mode=SimpleNamespace(is_extend=lambda: True),
            is_extend_in_batch=False, batch_size=lambda: 1, extend_lens=[length], prefix_lens=[start],
            spec_info=None, features=torch.ones(length if feature_rows is None else feature_rows, 4))

    def test_installed_wrapper_survives_consumed_state_and_routes_only_final_chunk(self):
        with tempfile.TemporaryDirectory() as directory:
            entry, worker = self.install_stub(directory)
            context, observed = self.context()
            forecast = Path(directory)/'forecast.json'
            entry._active_trace = SimpleNamespace(bank_correction=context, input_ids=[1]*8431,
                                                   bank_forecast_path=forecast)
            first = self.batch(0, 8192)
            result = worker.forward_batch_generation(first, test_argument='preserved')
            self.assertEqual(result.native_marker, 'unchanged')
            self.assertIsNone(worker.forward_calls[0]['spec_info_before'])
            self.assertEqual(worker.forward_calls[0]['kwargs'], {'test_argument': 'preserved'})
            self.assertEqual(first.spec_info.ctx_lens.tolist(), [0])
            self.assertEqual(context.prompt_seen, 8192)
            self.assertEqual(observed, [])
            self.assertFalse(forecast.exists())
            second = self.batch(8192, 239)
            second.spec_info = first.spec_info
            second.features[-1] = torch.tensor([3., 4., 8., 9.])
            worker.forward_batch_generation(second)
            self.assertEqual(second.spec_info.ctx_lens.tolist(), [0])
            self.assertEqual(context.prompt_seen, 8431)
            self.assertEqual(len(observed), 1)
            np.testing.assert_allclose(observed[0], [.6, .8], atol=1e-7)
            self.assertTrue(forecast.is_file())
            hidden = torch.arange(36, dtype=torch.float32).reshape(9, 4)
            E = torch.eye(4)
            predicted = worker._greedy_sample_from_vocab_parallel_head(hidden_states=hidden, lm_head=E, chunk_size=9)
            self.assertIs(worker.head_calls[0][0], hidden)
            self.assertIs(worker.head_calls[0][1], E)
            self.assertEqual(worker.head_calls[0][2], 9)
            self.assertTrue(torch.equal(predicted, torch.argmax(hidden @ E.T, dim=-1)))
            self.assertEqual(len(observed), 1)

    def test_installed_wrapper_rejects_returned_feature_length_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            entry, worker = self.install_stub(directory)
            context, observed = self.context()
            forecast = Path(directory)/'forecast.json'
            entry._active_trace = SimpleNamespace(bank_correction=context, input_ids=[1]*8431,
                                                   bank_forecast_path=forecast)
            with self.assertRaisesRegex(RuntimeError, 'actual native prefill length'):
                worker.forward_batch_generation(self.batch(0, 8192, feature_rows=8191))
            self.assertEqual(observed, [])
            self.assertEqual(context.prompt_seen, 0)
            self.assertFalse(forecast.exists())

    def test_installed_wrapper_rejects_prefix_gap_before_routing(self):
        with tempfile.TemporaryDirectory() as directory:
            entry, worker = self.install_stub(directory)
            context, observed = self.context()
            forecast = Path(directory)/'forecast.json'
            entry._active_trace = SimpleNamespace(bank_correction=context, input_ids=[1]*8431,
                                                   bank_forecast_path=forecast)
            worker.forward_batch_generation(self.batch(0, 8192))
            with self.assertRaisesRegex(RuntimeError, 'not contiguous'):
                worker.forward_batch_generation(self.batch(8191, 239))
            self.assertEqual(observed, [])
            self.assertFalse(forecast.exists())

    def test_other_arms_keep_native_forward_and_head_without_bank_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            entry, worker = self.install_stub(directory)
            context = hooks.RequestCorrection({'arm': 'OSD'}, {}, None)
            entry._active_trace = SimpleNamespace(bank_correction=context)
            batch = self.batch(0, 2)
            result = worker.forward_batch_generation(batch)
            self.assertEqual(result.native_marker, 'unchanged')
            hidden, E = torch.ones(9, 4), torch.eye(4)
            predicted = worker._greedy_sample_from_vocab_parallel_head(hidden_states=hidden, lm_head=E)
            self.assertIs(worker.head_calls[0][0], hidden)
            self.assertTrue(torch.equal(predicted, torch.argmax(hidden @ E.T, dim=-1)))


class ControllerAccounting(unittest.TestCase):
    def test_qualification_requires_measurements_not_positive_native_gains(self):
        backend=native.NativeBackend({})
        evidence=dict(sampled_positions=0,compiled_fusion={'passed':True},
            current_response_only=True,fixed_sampled_objective=True,compressed_direct={'passed':True},
            direct_historical={'passed':True},source_fit={'gain':None},native_precision_reported=True,
            native_precision={'status':'measured','sampled_positions':0,'performance_gate':False})
        result=dict(canonical=[dict(arm='BANK',turn=t,read={'coefficients':[0.]*4}) for t in (1,2)],
            scores=[dict(arm=a,turn=t,repeat=0,read={'coefficients':[0.]*4},same_generation_valid=True)
                    for t in (1,2) for a in ('BANK','F')])
        with tempfile.TemporaryDirectory() as directory:
            backend.out=Path(directory);backend.qualification_evidence=[evidence]
            empty=backend.qualify(result)
            self.assertFalse(empty['compiled_fusion'])
            self.assertFalse(empty['compressed_feedback'])
            self.assertFalse(empty['native_precision_reported'])
            evidence['sampled_positions']=1
            evidence['source_fit']={'gain':0.,'performance_gate':False}
            evidence['native_precision']=dict(status='measured',sampled_positions=1,performance_gate=False,
                cases=[dict(label='fitted',correction_survived_feature_rounding=False,
                            native_loss_change=0.,reference_loss_change=-.01)])
            measured=backend.qualify(result)
            self.assertTrue(measured['compiled_fusion'])
            self.assertTrue(measured['native_precision_reported'])
            self.assertEqual(measured['nonempty_feedback_responses'],1)
            self.assertFalse(measured['performance_win_required'])

    def test_bank_read_in_request_but_activation_and_cache_outside(self):
        backend = native.NativeBackend({})
        backend.projections = {'path': 'p', 'sha256': 'p'*64}
        backend._ensure_worker = lambda kind: 0.
        backend.publish = lambda label: dict(activation_s=7., native_activation_s=2.)
        calls = []
        def call(action, body):
            calls.append(action)
            if action == 'bank_activate':
                return {'loading_s': 1.}
            if action == 'flush':
                return {'seconds': 3.}
            return dict(input_ids=[1, 2], input_ids_sha256=native.digest([1, 2]), output_ids=[3],
                        read={'routing_recomputed': True}, status='passed')
        backend.worker = SimpleNamespace(call=call)
        state = dict(arm='BANK', turn=1, state_sha256='a'*64)
        with patch.object(native.time, 'perf_counter', side_effect=[10., 12., 20., 25.]):
            result = backend.serve([1, 2], 'BANK', state, turn_index=1, repeat=0)
        self.assertEqual(calls, ['bank_activate', 'flush', 'generate'])
        self.assertEqual(result['request_s'], 5.)
        self.assertEqual(result['activation_s'], 9.)
        self.assertEqual(result['native_activation_s'], 2.)
        self.assertEqual(result['cache_maintenance_s'], 3.)
        self.assertEqual(result['state_sha256'], 'a'*64)
        self.assertEqual(result['deployment_activation_s'], 0.)
        with patch.object(native.time, 'perf_counter', side_effect=[30., 32., 40., 45.]):
            canonical = backend.serve([1, 2], 'BANK', state, turn_index=1, canonical=True)
        self.assertEqual(canonical['activation_s'], 9.)
        self.assertEqual(canonical['deployment_activation_s'], 2.)
        self.assertEqual(canonical['benchmark_method_restore_s'], 7.)

    def test_nonbank_serving_never_reads_bank(self):
        backend = native.NativeBackend({}); backend.projections = {}
        backend._ensure_worker = lambda kind: 0.
        backend.publish = lambda label: dict(activation_s=0., native_activation_s=0.)
        calls = []
        def call(action, body):
            calls.append(action)
            if action == 'flush':
                return {'seconds': 0.}
            return dict(input_ids=[1], input_ids_sha256=native.digest([1]), output_ids=[2])
        backend.worker = SimpleNamespace(call=call)
        state = dict(arm='OSD', turn=2, state_sha256='d'*64, checkpoint={'checkpoint_id': 'D1'})
        backend.serve([1], 'OSD', state, turn_index=2, repeat=0)
        self.assertEqual(calls, ['flush', 'generate'])

    def test_benchmark_or_wrong_state_is_not_feedback(self):
        backend = native.NativeBackend({})
        with self.assertRaisesRegex(ValueError, 'Only actual canonical'):
            backend.learn('BANK', {'canonical': False}, {}, turn_index=1)

    def test_import_did_not_launch_worker_or_cuda(self):
        backend = native.NativeBackend({})
        self.assertIsNone(backend.worker)
        self.assertIsNone(backend.trainer)
        self.assertIsNone(backend.episode_id)


class StorageAndPins(unittest.TestCase):
    def test_cumulative_budget_counts_all_retention_and_peak_overlap(self):
        M = 2*2**30
        budget = native.campaign_storage_bound(M)
        self.assertEqual(budget['episodes'], 7)
        self.assertEqual(budget['retained_osd_bundles_bytes'], 7*(3*M+64*2**20))
        self.assertEqual(budget['retained_bank_and_qualification_bytes'], 60*2**30)
        self.assertEqual(budget['transient_rolling_bytes'], 8*M+512*2**20)
        self.assertEqual(budget['required_free_bytes'], budget['planned_writes_bytes']+8*2**30)
        self.assertFalse(budget['assumptions']['offload_assumed'])
        self.assertFalse(budget['assumptions']['final_osd_retirement_assumed'])

    def test_cannot_lower_qualification_replay_budget(self):
        with self.assertRaisesRegex(ValueError, 'Qualification bank/replay'):
            native.campaign_storage_bound(2*2**30, qualification_bank_bytes=8*2**30)

    def test_campaign_guard_before_gpu_requires_complete_retention_capacity(self):
        with tempfile.TemporaryDirectory() as directory:
            models = Path(directory)/'models.json'
            models.write_text(json.dumps({'released_draft': {'checkpoint_files': {
                'model.safetensors': {'bytes': 2*2**30}, 'config.json': {'bytes': 10}}}}))
            config = {'model_bindings': native.base.binding(models), 'output_root': str(Path(directory)/'outputs')}
            required = native.campaign_storage_bound(2*2**30)['required_free_bytes']
            with patch.object(native.shutil, 'disk_usage', return_value=SimpleNamespace(free=required)):
                receipt = native.admit_campaign_storage(config)
                self.assertEqual(receipt['status'], 'passed')
            with patch.object(native.shutil, 'disk_usage', return_value=SimpleNamespace(free=required-1)):
                with self.assertRaisesRegex(ValueError, 'No GPU started'):
                    native.admit_campaign_storage(config)
            with self.assertRaisesRegex(ValueError, 'cumulative campaign storage'):
                native.admit_campaign_storage({**config, 'campaign_storage_budget_bytes': 1})

    def test_numerical_versions_fail_closed_before_model_loading(self):
        import scipy
        with patch.object(scipy, '__version__', '0.0.unreviewed'):
            with self.assertRaisesRegex(ValueError, 'numerical environment differs'):
                native.check_numerical_environment()


if __name__ == '__main__':
    unittest.main()
