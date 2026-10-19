"""Focused CPU checks for native correction placement, causality and accounting."""
from __future__ import annotations

from pathlib import Path
import json
import tempfile
from types import SimpleNamespace
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


class ControllerAccounting(unittest.TestCase):
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
