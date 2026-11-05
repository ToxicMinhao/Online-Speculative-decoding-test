import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.o_half_matched_lr import campaign, storage
from experiments.ok_events_5_8.storage import identity


class OHalfOrchestrationTests(unittest.TestCase):
    def test_peak_includes_atomic_overlap_and_unchanged_reserve(self):
        capacity = storage.future_capacity()
        self.assertEqual(capacity['future_tensor_peak_bytes'], 23 * storage.WEIGHTS)
        self.assertEqual(capacity['reserve_bytes'], 8 * storage.GIB)
        self.assertEqual(capacity['preexisting_deletion_credit_bytes'], 0)
        self.assertEqual(capacity['compiler_bytes'], 8 * storage.GIB)
        self.assertEqual(capacity['peak_required_bytes'], 75_394_485_270)

    def test_resume_does_not_credit_existing_odd_tensors(self):
        capacity = storage.future_capacity(1, 1)
        self.assertEqual(capacity['compiler_bytes'], 7 * storage.GIB)
        # The completed event1 occupies disk already. Its later removal earns
        # no speculative free-space credit in this additional-write calculation.
        self.assertEqual(capacity['preexisting_deletion_credit_bytes'], 0)
        self.assertGreater(capacity['future_tensor_peak_bytes'], 0)
        with self.assertRaises(ValueError):
            storage.future_capacity(5, 1)

    def test_retirement_keeps_even_successor_and_receipts(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            audit, odd = campaign.job_paths(root, 17, 1)
            successor_audit, even = campaign.job_paths(root, 17, 2)
            for endpoint in (odd, even):
                (endpoint / 'export').mkdir(parents=True)
                (endpoint / 'training_state.pt').write_bytes(b'state' + str(endpoint).encode())
                (endpoint / 'export/model.safetensors').write_bytes(b'weights')
                (endpoint / 'export/config.json').write_text('{}')
            for directory in (audit, successor_audit):
                directory.mkdir(parents=True)
            paths = [odd / 'training_state.pt', odd / 'export/model.safetensors', odd / 'export/config.json']
            expected = {str(p): identity(p) for p in paths}
            (audit / 'STATE_IDENTITIES.json').write_text(json.dumps(expected))
            (successor_audit / 'SAVED_ENDPOINT.json').write_text(json.dumps({'status': 'passed'}))
            latest = even.parent.parent / 'training_state.pt'
            os.link(even / 'training_state.pt', latest)
            receipt = storage.retire_odd_boundary(root, 17, 2)
            self.assertEqual(receipt['status'], 'completed')
            self.assertTrue((even / 'training_state.pt').exists())
            self.assertTrue((odd / 'export/config.json').exists())
            for p in paths[:2]:
                self.assertFalse(p.exists())
                storage.validate_retired(root, 17, 1, p, expected[str(p)])
            self.assertEqual(storage.retire_odd_boundary(root, 17, 2), receipt)

    def test_retirement_refuses_unverified_successor(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises((OSError, ValueError)):
                storage.retire_odd_boundary(folder, 17, 2)

    def test_training_before_bridge_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            campaign.job_paths(folder, 17, 1)[0].mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, 'before complete bridge'):
                campaign.inspect_boundary(folder, {})

    def test_gap_before_later_job_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            for seed in (17, 29):
                campaign.evaluation_path(folder, seed, 'bridge').mkdir(parents=True)
            campaign.job_paths(folder, 17, 2)[0].mkdir(parents=True)
            with patch.object(campaign, 'validate_evaluation', return_value=[]):
                with self.assertRaisesRegex(ValueError, 'after a gap'):
                    campaign.inspect_boundary(folder, {})

    def test_empty_work_is_explicit_and_not_complete(self):
        with tempfile.TemporaryDirectory() as folder:
            counts = campaign.reconcile(folder)
            self.assertEqual(counts['optimizer_steps'], 0)
            self.assertEqual(counts['evaluation_attempted'], 0)
            with self.assertRaises(ValueError):
                campaign.require_complete(counts)


if __name__ == '__main__':
    unittest.main()
