"""Repairs retain the original clock, consumed work, and immutable bindings."""
import tempfile
from pathlib import Path
import unittest

from .qualify import remaining_budget, verify_original_records
from experiments.ok_coding.storage import identity


class RepairBudgetTests(unittest.TestCase):
    def test_repair_preserves_elapsed_clock(self):
        clock = dict(first_gpu_action_epoch=100, elapsed_limit_s=1000, occupied_card_limit_s=2000)
        self.assertEqual(remaining_budget(clock, 40, 1000), 100)

    def test_repair_counts_prior_card_usage(self):
        clock = dict(first_gpu_action_epoch=100, elapsed_limit_s=1000, occupied_card_limit_s=2000)
        self.assertEqual(remaining_budget(clock, 1950, 200), 50)
        self.assertEqual(remaining_budget(clock, 2001, 200), -1)

    def test_binding_requires_clock_process_result_and_window(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / name for name in ('CLOCK.json', 'PROCESS.json', 'RESULT.json', 'WINDOW.json')]
            for path in paths:
                path.write_text('{}')
            plan = dict(original_clock=str(paths[0]), original_process=str(paths[1]),
                        original_probe=str(root), prior_record_identities={str(p): identity(p) for p in paths})
            verify_original_records(plan)
            del plan['prior_record_identities'][str(paths[3])]
            with self.assertRaisesRegex(ValueError, 'omits'):
                verify_original_records(plan)
            plan['prior_record_identities'][str(paths[3])] = identity(paths[3])
            paths[3].write_text('{"changed": true}')
            with self.assertRaisesRegex(ValueError, 'changed'):
                verify_original_records(plan)


if __name__ == '__main__':
    unittest.main()
