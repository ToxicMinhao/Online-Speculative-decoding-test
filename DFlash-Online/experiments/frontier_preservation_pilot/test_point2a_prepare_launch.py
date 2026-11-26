"""Point2A identity, admission and cumulative work-bound regressions."""
import json
from pathlib import Path
import tempfile
import unittest

from online_feedback.storage import write_json
from experiments.ok_coding.storage import identity
from .point2a_prepare import CASES, historical_controls, resource_admission
from .point2a_launch import Point2AStepLedger, completed_cases, completed_oracle, phase_clock, remaining_time


class LedgerTests(unittest.TestCase):
    def test_new_cases_and_explicit_identical_candidate_skip(self):
        rows = [dict(attempt=1, case=c, status='completed') for c in CASES]
        self.assertTrue(completed_cases({'calls': rows}, 1))
        self.assertFalse(completed_cases({'calls': rows[:-1]}, 1))
        self.assertTrue(completed_cases({'calls': rows[:-1]}, 1, True))
        rows[-1]['status'] = 'reserved'
        self.assertFalse(completed_cases({'calls': rows}, 1))

    def test_success_requires_all_28_oracle_backwards(self):
        rows = [dict(attempt=1, status='completed') for _ in range(28)]
        self.assertTrue(completed_oracle({'token_backwards': rows}, 1))
        self.assertFalse(completed_oracle({'token_backwards': rows[:-1]}, 1))
        rows[-1]['status'] = 'reserved'
        self.assertFalse(completed_oracle({'token_backwards': rows}, 1))

    def test_reserve_before_completion_and_six_total(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'steps.json'
            ledger = Point2AStepLedger(path, {'contract': 'original'})
            ledger.start_attempt(1)
            with self.assertRaises(ValueError):
                ledger.reserve('BETA')
            for case in CASES:
                n = ledger.reserve(case)
                self.assertEqual(json.loads(path.read_text())['calls'][-1]['status'], 'reserved')
                ledger.complete(n)
            ledger.finish_attempt(1, False)
            with self.assertRaises(ValueError):
                ledger.start_attempt(2)
            ledger.start_attempt(2, 'identified fix')
            for case in CASES:
                ledger.complete(ledger.reserve(case))
            with self.assertRaises(RuntimeError):
                ledger.reserve('C')
            self.assertEqual(len(ledger.value['calls']), 6)
            ledger.finish_attempt(2, True)
            with self.assertRaises(ValueError):
                ledger.start_attempt(3, 'extra')

    def test_token_budget_separate_and_pending_is_charged(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'steps.json'
            ledger = Point2AStepLedger(p, {'original': True})
            ledger.start_attempt(1)
            for i in range(30):
                n = ledger.reserve_token({'token_ordinal': i})
                self.assertEqual(json.loads(p.read_text())['token_backwards'][-1]['status'], 'reserved')
                if i < 29:
                    ledger.complete_token(n)
            self.assertEqual(ledger.value['calls'], [])
            with self.assertRaises(RuntimeError):
                ledger.reserve_token('excess')
            ledger.finish_attempt(1, False)
            second = Point2AStepLedger(p, {'original': True})
            second.start_attempt(2, 'identified fix')
            for i in range(30):
                second.complete_token(second.reserve_token(i))
            with self.assertRaises(RuntimeError):
                second.reserve_token('excess')
            self.assertEqual(len(second.value['token_backwards']), 60)
            self.assertEqual(second.value['token_backwards'][29]['status'], 'reserved')
            with self.assertRaises(ValueError):
                second.complete_token(1)

    def test_original_clock_and_history_survive_retry(self):
        with tempfile.TemporaryDirectory() as d:
            clock = phase_clock(d, {'contract': 1}, 1, 1000)
            self.assertEqual(clock['historical_occupied_card_s'], 5510.8580226639515)
            clock['completed_occupied_card_s'] = 300
            write_json(Path(d)/'CLOCK.json', clock)
            second = phase_clock(d, {'contract': 1}, 2, 1600)
            self.assertEqual(second['first_gpu_action_epoch'], 1000)
            self.assertEqual(remaining_time(second, 1600, 100), 3000)
            with self.assertRaises(FileExistsError):
                phase_clock(d, {'contract': 1}, 1, 1800)


class ControlTests(unittest.TestCase):
    def test_full_endpoint_identity_is_required(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            published = dict(cases={}, raw_input_receipts={})
            for name in ('C', 'F'):
                row = dict(status='completed', gradient_identity={'sha256': 'gradient'},
                    endpoint=dict(parameters={'sha256': 'weights'}, exp_avg={'sha256': 'moment'}, global_step=129))
                path = root/f'attempt_001/native/{name}.json'
                write_json(path, row)
                published['cases'][name] = row
                published['raw_input_receipts'][str(path.relative_to(root))] = identity(path)
            self.assertEqual(set(historical_controls(root, published)), {'C', 'F_BETA'})
            published['cases']['C']['endpoint']['exp_avg']['sha256'] = 'different'
            with self.assertRaisesRegex(ValueError, 'scientific identity'):
                historical_controls(root, published)

    def test_all_storage_and_memory_reserves_apply(self):
        values = dict(free_bytes=407_000_000_000, charged_bytes=850_000_000,
            remaining_write_bytes=1_000_000_000, host_available_bytes=104_000_000_000,
            gpu_free_bytes=85_000_000_000, gpu_busy=False)
        self.assertTrue(resource_admission(**values)['admitted'])
        for key, val in [('free_bytes', 400_500_000_000), ('charged_bytes', 1_100_000_000),
                         ('host_available_bytes', 86_000_000_000), ('gpu_free_bytes', 59_000_000_000), ('gpu_busy', True)]:
            self.assertFalse(resource_admission(**dict(values, **{key: val}))['admitted'])


if __name__ == '__main__':
    unittest.main()
