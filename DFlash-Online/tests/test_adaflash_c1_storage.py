"""Admission covers every capped execution without crediting another arm."""
import json
from pathlib import Path
import tempfile
import unittest

from experiments.adaflash_c1.storage import capacity_plan, RAW_BYTES_PER_GENERATION
from experiments.kplus_full_gsm8k.storage import boundary_files


class StorageTests(unittest.TestCase):
    def test_full_cap_budget_includes_qualification_and_overlaps(self):
        plan = capacity_plan(2097259104)
        self.assertEqual(plan['raw_budget_generations'], 7473 + 9216 + 4)
        self.assertGreaterEqual(plan['compact_bytes'], 16693 * RAW_BYTES_PER_GENERATION)
        self.assertEqual(plan['required_initial_free_bytes'], 143200462131)
        self.assertEqual(plan['state_weight_equivalents'], 9)
        self.assertFalse(plan['offload_assumed'])

    def test_a_credit_cannot_include_kplus_state(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for arm in ('A', 'KPLUS'):
                trainer = root / 'campaign_001/arms' / arm / 'trainer'
                trainer.mkdir(parents=True)
                (trainer / 'initial_state.pt').write_bytes(arm.encode())
            self.assertEqual(boundary_files(root, arm='A'),
                [root / 'campaign_001/arms/A/trainer/initial_state.pt'])
            self.assertEqual(boundary_files(root),
                [root / 'campaign_001/arms/KPLUS/trainer/initial_state.pt'])

    def test_unvalidated_boundary_is_not_storage_credit(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            boundary = root / 'campaign_001/arms/A/boundaries'
            boundary.mkdir(parents=True)
            (boundary / 'event_001.json').write_text(json.dumps({'status': 'failed'}))
            with self.assertRaisesRegex(ValueError, 'unvalidated'):
                boundary_files(root, arm='A')


if __name__ == '__main__':
    unittest.main()
