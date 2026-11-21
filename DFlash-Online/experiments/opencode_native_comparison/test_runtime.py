import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from . import runtime


class WorkTests(unittest.TestCase):
    def test_failure_is_not_a_completed_forward_or_hidden_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime.initialize(directory)
            ledger = runtime.WorkLedger(directory, 'test', 'qualification')
            def fail():
                raise RuntimeError('retained failure')
            with self.assertRaisesRegex(RuntimeError, 'retained failure'):
                ledger.call({'response_forward':1}, 'broken_forward', fail)
            state = runtime.read(Path(directory)/'WORK.json')
            self.assertEqual(state['attempted']['qualification.response_forward'], 1)
            self.assertEqual(state['completed']['qualification.response_forward'], 0)
            self.assertEqual(len(state['pending']), 1)
            self.assertEqual(len(state['failures']), 1)
            with self.assertRaises(FileExistsError):
                runtime.initialize(directory)

    def test_short_response_has_no_teacher_work(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime.initialize(directory)
            with self.assertRaisesRegex(ValueError, 'ineligible_short_response'):
                runtime.require_eligible_response({'response_ids':[3]})
            state = runtime.read(Path(directory)/'WORK.json')
            self.assertEqual(sum(state['attempted'].values()), 0)

    def test_forecast_uses_fixed_contingency_and_independent_card_total(self):
        checked = dict(clock=dict(elapsed_s=100.,occupied_s=200.),storage={})
        with patch.object(runtime, 'boundary', return_value=checked):
            result = runtime.forecast({}, 1000, 2000)
            self.assertEqual(result['contingency_elapsed_total_s'], 4950)
            self.assertEqual(result['projected_card_total_s'], 2200)
            with self.assertRaises(RuntimeError):
                runtime.forecast({}, 48*3600, 2000)
            with self.assertRaises(ValueError):
                runtime.forecast({}, float('nan'), 2000)


if __name__ == '__main__':
    unittest.main()
