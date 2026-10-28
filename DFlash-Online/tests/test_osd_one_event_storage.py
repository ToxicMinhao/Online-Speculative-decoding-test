import unittest
from experiments.osd_one_event.storage import budget, RESERVE_BYTES, WORK_BYTES, COMPACT_BYTES


class StorageTests(unittest.TestCase):
    def test_reserve_is_in_additional_write_requirement(self):
        tensors = 1000
        required = 1050 + RESERVE_BYTES + WORK_BYTES + COMPACT_BYTES
        self.assertTrue(budget(tensors, required)['admitted'])
        self.assertFalse(budget(tensors, required - 1)['admitted'])

    def test_existing_compact_preserves_reserve(self):
        result = budget(1000, 100 * 2**30, compact_present=1234)
        self.assertEqual(result['remaining_compact_allowance_bytes'], COMPACT_BYTES - 1234)
        self.assertEqual(result['reserve_bytes'], 8 * 2**30)
        self.assertFalse(budget(1000, 100 * 2**30, compact_present=COMPACT_BYTES + 1)['admitted'])
