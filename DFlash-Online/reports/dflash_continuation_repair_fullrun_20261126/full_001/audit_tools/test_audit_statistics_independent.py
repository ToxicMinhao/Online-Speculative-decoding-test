import sys
import unittest
from pathlib import Path
study=Path(__file__).resolve().parent
code=study/'implementation_checkout/DFlash-Online'
sys.path[:0]=[str(code),str(code/'tests')]
from audit_statistics_independent import reduce_independently,compare
from test_continuation_repair_fullrun_serving import rows_fixture
from experiments.continuation_repair_fullrun.analysis import summarize,selection_decision

class IndependentArithmeticTests(unittest.TestCase):
    def test_same_draw_stream_and_results(self):
        rows=rows_fixture();actual=summarize(rows,expected_requests=4)
        actual['decision']=selection_decision(actual,complete=True)
        expected=reduce_independently(rows,n=4)
        self.assertEqual(expected['draw_sha256'],actual['bootstrap']['draw_sha256'])
        self.assertEqual(compare({k:expected[k] for k in ('requests','totals','contrasts','decision')},actual),127)
    def test_corrupt_scalar_and_missing_repeat_rejected(self):
        with self.assertRaises(ValueError):compare({'value':.1},{'value':.2})
        with self.assertRaises(ValueError):reduce_independently(rows_fixture()[:-1],n=4)

if __name__=='__main__':unittest.main()
