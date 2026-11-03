"""CPU checks for admission and new-run-only tensor retirement."""
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from experiments.kplus_full_gsm8k import storage
from experiments.kplus_full_gsm8k.preflight import timing_barrier
from online_feedback.storage import write_json


class StorageTests(unittest.TestCase):
    def setUp(self):
        cache = tempfile.TemporaryDirectory()
        self.addCleanup(cache.cleanup)
        self.cache = Path(cache.name)
        variables = {name: str(self.cache / name) for name in storage.CACHE_ENVIRONMENT}
        variables['FLASHINFER_WORKSPACE_BASE'] = str(self.cache / 'flashinfer_base')
        environment = patch.dict(os.environ, variables)
        environment.start()
        self.addCleanup(environment.stop)

    def test_budget_contains_atomic_overlap_full_features_and_reserve(self):
        plan = storage.capacity_plan(1000)
        self.assertEqual(plan['state_weight_equivalents'], 9)
        self.assertEqual(plan['reserve_bytes'], 8 * 2**30)
        self.assertGreater(plan['feature_peak_bytes'], 128 * 2272 * 4096 * 6 * 2)
        self.assertEqual(plan['required_initial_free_bytes'],
            plan['tensor_peak_with_margin_bytes'] + sum(plan[k] for k in
                ('compact_bytes', 'compiler_bytes', 'workspace_bytes', 'reserve_bytes')))

    def test_failed_admission_is_durable_and_future_removal_gets_no_credit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = storage.capacity_plan(1000)
            receipt = root / 'ADMISSION.json'
            with patch.object(storage.shutil, 'disk_usage', return_value=types.SimpleNamespace(free=1)):
                with self.assertRaisesRegex(ValueError, 'admission'):
                    storage.admit(root, plan, receipt_path=receipt)
            saved = json.loads(receipt.read_text())
            self.assertEqual(saved['status'], 'failed')
            self.assertEqual(saved['validated_existing_tensor_bytes'], 0)
            self.assertTrue(saved['no_future_deletion_credit'])

    def test_boundary_inventory_ignores_scientific_validation_sidecars(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / 'campaign_001/arms/KPLUS'
            state = base / 'trainer/events/event_001/training_state.pt'
            state.parent.mkdir(parents=True)
            state.write_bytes(b'state')
            weights = state.parent / 'model.safetensors'
            weights.write_bytes(b'weights')
            write_json(base / 'boundaries/event_001.json', {
                'status': 'passed', 'state': str(state), 'weights': str(weights),
            })
            write_json(base / 'boundaries/event_001_validation.json', {
                'status': 'passed', 'completed_event': 1,
            })
            self.assertEqual(storage.boundary_files(root), [state, weights])
            with patch.object(storage.shutil, 'disk_usage', return_value=types.SimpleNamespace(free=10**15)):
                result = storage.admit(root, storage.capacity_plan(1000))
            self.assertEqual(result['status'], 'passed')

    def test_sparse_tensors_receive_only_allocated_credit_and_hardlinks_count_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / 'campaign_001/arms/KPLUS'
            initial = base / 'trainer/initial_state.pt'
            initial.parent.mkdir(parents=True)
            with initial.open('wb') as stream:
                stream.write(b'x')
                stream.truncate(2**20)
            snapshot = initial.parent / 'snapshot.pt'
            snapshot.hardlink_to(initial)
            weights = initial.parent / 'model.safetensors'
            weights.write_bytes(b'weights')
            write_json(base / 'boundaries/event_001.json', {
                'status': 'passed', 'state': str(snapshot), 'weights': str(weights),
            })
            with patch.object(storage.shutil, 'disk_usage', return_value=types.SimpleNamespace(free=10**15)):
                result = storage.admit(root, storage.capacity_plan(1000))
            expected = 512 * (initial.stat().st_blocks + weights.stat().st_blocks)
            self.assertEqual(result['validated_existing_tensor_bytes'], expected)
            self.assertLess(expected, initial.stat().st_size)

    def test_compact_allocation_deduplicates_links_and_rejects_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / 'first.json'
            first.write_text('{}')
            second = root / 'second.json'
            second.hardlink_to(first)
            self.assertEqual(storage.allocated_bytes([first, second]), first.stat().st_blocks * 512)
            (root / 'redirect.json').symlink_to(first)
            with patch.object(storage.shutil, 'disk_usage', return_value=types.SimpleNamespace(free=10**15)):
                with self.assertRaisesRegex(ValueError, 'symlink'):
                    storage.admit(root, storage.capacity_plan(1000))

    def test_compiler_growth_is_measured_and_over_limit_fails_with_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = storage.capacity_plan(1000)
            plan['compiler_bytes'] = 4096
            with patch.object(storage.shutil, 'disk_usage', return_value=types.SimpleNamespace(free=10**15)):
                initial = storage.admit(root, plan)
                baseline = (root / 'COMPILER_BASELINE.json').read_bytes()
                cache = self.cache / 'TORCHINDUCTOR_CACHE_DIR'
                cache.mkdir()
                (cache / 'compiled.so').write_bytes(b'x' * 5000)
                receipt = root / 'FAILED_GROWTH.json'
                with self.assertRaisesRegex(ValueError, 'admission'):
                    storage.admit(root, plan, receipt_path=receipt)
            failed = json.loads(receipt.read_text())
            self.assertEqual(initial['compiler_growth_bytes'], 0)
            self.assertGreater(failed['compiler_growth_bytes'], 4096)
            self.assertEqual(failed['status'], 'failed')
            self.assertEqual((root / 'COMPILER_BASELINE.json').read_bytes(), baseline)

    def test_cache_inventory_cannot_change_or_be_rebased_after_work(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = storage.capacity_plan(1000)
            with patch.object(storage.shutil, 'disk_usage', return_value=types.SimpleNamespace(free=10**15)):
                storage.admit(root, plan)
                with patch.dict(os.environ, {'TRITON_CACHE_DIR': str(self.cache / 'other')}):
                    with self.assertRaisesRegex(ValueError, 'roots changed'):
                        storage.check_growth(root, plan)
                (root / 'COMPILER_BASELINE.json').unlink()
                work = root / 'campaign_001/arms/KPLUS/WORK.jsonl'
                work.parent.mkdir(parents=True)
                work.write_text('')
                with self.assertRaisesRegex(ValueError, 'baseline is missing'):
                    storage.check_growth(root, plan)

    def test_retirement_rejects_foreign_file_before_deletion(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / 'run'; root.mkdir()
            foreign = base / 'old.pt'; foreign.write_bytes(b'original')
            with self.assertRaises(ValueError):
                storage._retire(root, root / 'DELETE.json', [(foreign, 'foreign')], [], reason='test', retained={})
            self.assertEqual(foreign.read_bytes(), b'original')

    def test_retirement_requires_single_obsolete_link(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = root / 'old.pt'; value.write_bytes(b'original')
            (root / 'live.pt').hardlink_to(value)
            with self.assertRaises(ValueError):
                storage._retire(root, root / 'DELETE.json', [(value, 'old')], [], reason='test', retained={})
            self.assertTrue(value.exists())

    def prepared_fixture(self, root):
        event = root / 'events/event_001'
        event.mkdir(parents=True)
        feature = event / 'loop/features/r.pt'
        feature.parent.mkdir(parents=True)
        capture = event / 'request_r.pt'
        terminal = feature.with_suffix('.terminal.pt')
        record = dict(record_id='r', prompt_ids=[1], response_ids=[2, 3],
                      metadata={'capture': {'feature_path': str(capture)}})
        prepared = dict(status='prepared', record_id='r', input_ids=[1, 2, 3], response_only_loss=True,
                        feature_path=str(feature), terminal_feature={'tensor_path': str(terminal)})
        native_rows = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)
        terminal_row = torch.arange(8, 12, dtype=torch.bfloat16).reshape(1, 4)
        torch.save(dict(input_ids=[1, 2], target_argmax_ids=[2, 3],
                        hidden_states=native_rows, target_last_hidden_states=native_rows + 2), capture)
        torch.save(dict(input_ids=[1, 2, 3], positions=[2], final_feature_logits_used_for_labels=False,
                        hidden_states=terminal_row, target_last_hidden_states=terminal_row + 2), terminal)
        rows = torch.cat((native_rows, terminal_row)).unsqueeze(0)
        torch.save(dict(record_id='r', input_ids=torch.tensor([[1, 2, 3]]), prompt_length=1,
                        loss_mask=torch.tensor([[0., 1., 1.]]), hidden_states=rows,
                        target_last_hidden_states=rows + 2), feature)
        write_json(capture.with_suffix('.json'), record)
        write_json(feature.with_suffix('.json'), prepared)
        write_json(terminal.with_suffix('.json'), prepared['terminal_feature'])
        return event, record, prepared, capture, terminal, feature

    def test_redundant_feature_retirement_keeps_packet_and_compact_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            event, record, prepared, capture, terminal, feature = self.prepared_fixture(root)
            result = storage.retire_redundant_features(record, prepared, root, event)
            self.assertEqual(result['status'], 'completed')
            self.assertFalse(capture.exists())
            self.assertFalse(terminal.exists())
            self.assertTrue(feature.exists())
            self.assertTrue(capture.with_suffix('.json').exists())
            equality = json.loads((event / 'preparation_retirement/r.equivalence.json').read_text())
            self.assertTrue(equality['exact_decoded_rows'])
            self.assertEqual(equality['additional_model_forwards'], 0)
            with self.assertRaises(ValueError):
                storage.retire_redundant_features(record, prepared, root, event)

    def test_one_changed_feature_value_prevents_both_source_deletions(self):
        for key, offset in (('hidden_states', 0), ('target_last_hidden_states', -1)):
            with self.subTest(key=key, row=offset), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                event, record, prepared, capture, terminal, feature = self.prepared_fixture(root)
                packet = torch.load(feature, weights_only=False)
                packet[key][0, offset, 0] += 1
                torch.save(packet, feature)
                with self.assertRaisesRegex(ValueError, 'exactly preserve'):
                    storage.retire_redundant_features(record, prepared, root, event)
                self.assertTrue(capture.exists())
                self.assertTrue(terminal.exists())
                self.assertTrue(feature.exists())


class QuiescenceTests(unittest.TestCase):
    def check_output(self, command, **kwargs):
        if command[:2] == ['nvidia-smi', '--query-compute-apps=pid']:
            return self.gpus
        if command[-1] == 'pid=,ppid=':
            return '100 1\n101 100\n'
        return self.processes

    def test_empty_gpu_and_no_transfers_pass(self):
        self.gpus=''; self.processes='9 python report.py\n'
        with patch('subprocess.check_output', side_effect=self.check_output):
            self.assertEqual(timing_barrier()['status'], 'passed')

    def test_native_worker_child_is_allowed(self):
        self.gpus='101\n'; self.processes='100 python native_worker.py\n'
        with patch('subprocess.check_output', side_effect=self.check_output):
            self.assertEqual(timing_barrier(100)['allowed_worker_pids'], [100, 101])

    def test_foreign_gpu_transfer_or_training_stops(self):
        for gpu, processes in [('200\n',''), ('','10 scp artifact\n'), ('','10 python runner --train\n')]:
            self.gpus, self.processes = gpu, processes
            with patch('subprocess.check_output', side_effect=self.check_output), self.assertRaises(ValueError):
                timing_barrier(100)


if __name__ == '__main__':
    unittest.main()
