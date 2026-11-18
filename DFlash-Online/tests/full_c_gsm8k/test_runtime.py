"""Failure accounting and the original, non-resetting campaign clock."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from experiments.full_c_gsm8k import runtime
from experiments.full_c_gsm8k.contract import PRODUCTION,QUALIFICATION


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        runtime.initialize_work(self.root)

    def test_failed_operation_stays_charged_and_unknown(self):
        ledger=runtime.WorkLedger(self.root,'qualification')
        def fails():raise RuntimeError('lost worker')
        with self.assertRaisesRegex(RuntimeError,'lost worker'):
            ledger.call({'adam':1},'temporary step',fails)
        state=json.loads((self.root/'WORK.json').read_text())
        self.assertEqual(state['attempted']['adam'],1)
        self.assertEqual(state['completed']['adam'],0)
        self.assertEqual(len(state['pending']),1)
        with self.assertRaises(FileExistsError):runtime.initialize_work(self.root)

    def test_counted_native_returns_before_completion(self):
        ledger=runtime.WorkLedger(self.root,'serving')
        self.assertEqual(ledger.call({'scored_generation':1},'request',lambda:7),7)
        state=json.loads((self.root/'WORK.json').read_text())
        self.assertEqual(state['attempted'],state['completed'])
        self.assertEqual(state['pending'],{})

    def test_qualification_and_production_counts_remain_distinct(self):
        self.assertEqual(PRODUCTION['adam'],7424)
        self.assertEqual(QUALIFICATION['adam'],6)
        self.assertEqual(PRODUCTION['response_forward'],14848)
        self.assertEqual(QUALIFICATION['response_forward'],14)
        self.assertEqual(sum(PRODUCTION[k] for k in PRODUCTION if k.endswith('generation')),26485)
        for key in PRODUCTION:
            self.assertLessEqual(PRODUCTION[key]+QUALIFICATION[key],runtime.LIMITS[key])

    def test_time_admission_keeps_original_consumed_clock_and_contingency(self):
        plan={'root':str(self.root)}
        with patch.object(runtime,'clock_status',return_value=dict(elapsed_s=1000,occupied_s=2000)):
            receipt=runtime.require_time_for(plan,60000,90000)
            self.assertEqual(receipt['conservative_elapsed_s'],79800)
            self.assertEqual(receipt['clock']['elapsed_s'],1000)
            with self.assertRaisesRegex(RuntimeError,'fixed campaign'):
                runtime.require_time_for(plan,65000,90000)
        with patch.object(runtime,'clock_status',return_value=dict(elapsed_s=1,occupied_s=144000)):
            with self.assertRaises(TimeoutError):runtime.enforce_clock(plan)


if __name__=='__main__':unittest.main()
