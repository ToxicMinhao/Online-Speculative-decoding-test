"""CPU-only observer tests: importing the reviewed entry constructs no model."""
import json
from pathlib import Path
import random
import tempfile
import unittest

from experiments.full_c_gsm8k import initialization_trace as trace


class TraceTests(unittest.TestCase):
    def test_observer_keeps_global_generator_progress_and_records_actual_calls(self):
        before = random.getstate()
        self.addCleanup(lambda: random.setstate(before))
        random.seed(17)
        expected = random.Random(17)
        expected_values = [expected.getrandbits(24), expected.random()]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'calls.json'
            with trace.PythonRandomTrace(path) as observer:
                observer.stage = 'fixture'
                actual = [random.getrandbits(24), random.random()]
            receipt = json.loads(path.read_text())
        self.assertEqual(actual, expected_values)
        self.assertEqual(random.getstate(), expected.getstate())
        self.assertEqual([row['function'] for row in receipt['global_random_calls']], ['getrandbits','random'])
        self.assertTrue(all(row['changed'] for row in receipt['global_random_calls']))
        self.assertEqual(receipt['dropped_call_records'], 0)
        self.assertEqual(receipt['observer_draws'], 0)
        self.assertFalse(trace.torch.cuda.is_initialized())

    def test_nested_seed_calls_have_distinct_pairs_and_preserve_rng(self):
        before = random.getstate()
        self.addCleanup(lambda: random.setstate(before))
        expected = random.Random(17)
        expected_value = expected.getrandbits(24)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'calls.json'
            with trace.PythonRandomTrace(path):
                random.seed(17)
                actual = random.getrandbits(24)
            receipt = json.loads(path.read_text())
        self.assertEqual(actual, expected_value)
        self.assertEqual(random.getstate(), expected.getstate())
        rows = receipt['global_random_calls']
        seeds = [row for row in rows if row['function'] == 'seed']
        self.assertEqual({row['call_type'] for row in seeds}, {'python', 'c'})
        self.assertEqual(len(seeds), 2)
        self.assertTrue(all('before' in row and 'after' in row for row in rows))
        self.assertEqual(receipt['dropped_call_records'], 0)

    def test_private_generator_is_ignored_and_exception_restores_profiler(self):
        before = random.getstate()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'calls.json'
            with self.assertRaisesRegex(ValueError, 'test'):
                with trace.PythonRandomTrace(path):
                    random.Random(17).getrandbits(24)
                    raise ValueError('test')
            receipt = json.loads(path.read_text())
        self.assertEqual(receipt['global_random_calls'], [])
        self.assertEqual(random.getstate(), before)
        self.assertIsNone(trace.sys.getprofile())


if __name__ == '__main__':
    unittest.main()
