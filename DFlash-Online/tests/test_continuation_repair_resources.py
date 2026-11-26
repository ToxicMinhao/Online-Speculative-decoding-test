"""Prospective resource ceilings retain failed work and all-location charges."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.continuation_repair_pilot.launch import AdamLedger, boundary


class ResourceTests(unittest.TestCase):
    def test_qualification_failed_application_still_consumes_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'ADAM_LEDGER.json'
            ledger = AdamLedger(path, phase='qualification')
            first = ledger.reserve('BASE')  # Failure before completion is unknown work.
            self.assertEqual(first, 0)
            for arm in ('TAIL_DISABLED', 'TAIL', 'BASE', 'TAIL_DISABLED', 'TAIL'):
                resumed = AdamLedger(path, phase='qualification')
                resumed.complete(resumed.reserve(arm))
            with self.assertRaises(ValueError):
                AdamLedger(path, phase='qualification').reserve('TAIL')
            receipt = json.loads(path.read_text())
            self.assertEqual(len(receipt['calls']), 6)
            self.assertEqual(receipt['calls'][0]['status'], 'attempted')

    def test_training_has_own_fixed_budget_and_arm(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'ADAM_LEDGER.json'
            qualification = AdamLedger(path, phase='qualification')
            qualification.complete(qualification.reserve('BASE'))
            main = AdamLedger(path, phase='training', arm='TAIL')
            with self.assertRaises(ValueError):
                main.reserve('BASE')
            for _ in range(128):
                main.complete(main.reserve('TAIL'))
            with self.assertRaises(ValueError):
                main.reserve('TAIL')
            records = json.loads(path.read_text())['calls']
            self.assertEqual(sum(r['phase'] == 'training' for r in records), 128)
            self.assertEqual(sum(r['phase'] == 'qualification' for r in records), 1)

    def test_admission_charges_pending_writes_and_local_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = dict(root=directory, local_charged_bytes=1_500_000_000,
                        limits=dict(elapsed_s=14400, occupied_card_s=14400,
                                    reserve_bytes=400_000_000_000, allocated_bytes=24_000_000_000))
            usage = dict(charged_bytes=500_000_000, free_bytes=422_000_000_000)
            with patch('experiments.continuation_repair_pilot.launch.storage_usage', return_value=usage):
                boundary(plan, remaining_write_bytes=22_000_000_000)
                with self.assertRaises(RuntimeError):
                    boundary(plan, remaining_write_bytes=22_000_000_001)
            usage['free_bytes'] = 500_000_000_000
            with patch('experiments.continuation_repair_pilot.launch.storage_usage', return_value=usage):
                with self.assertRaises(RuntimeError):
                    boundary(plan, remaining_write_bytes=22_000_000_001)
            receipt = json.loads((Path(directory) / 'STORAGE.json').read_text())
            self.assertEqual(receipt['all_locations_charged_bytes'], 2_000_000_000)

    def test_original_clock_and_consumed_card_time_are_not_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = dict(root=directory, local_charged_bytes=0,
                        limits=dict(elapsed_s=14400, occupied_card_s=14400,
                                    reserve_bytes=400_000_000_000, allocated_bytes=24_000_000_000))
            usage = dict(charged_bytes=0, free_bytes=450_000_000_000)
            clock_path = Path(directory) / 'CLOCK.json'
            for first, occupied in ((0., 1.), (14399., 14400.)):
                saved = dict(first_gpu_action_epoch=first, occupied_card_s=occupied)
                clock_path.write_text(json.dumps(saved))
                with patch('experiments.continuation_repair_pilot.launch.storage_usage', return_value=usage), \
                     patch('experiments.continuation_repair_pilot.launch.time.time', return_value=14400.):
                    with self.assertRaises(RuntimeError):
                        boundary(plan)
                self.assertEqual(json.loads(clock_path.read_text()), saved)


if __name__ == '__main__':
    unittest.main()
