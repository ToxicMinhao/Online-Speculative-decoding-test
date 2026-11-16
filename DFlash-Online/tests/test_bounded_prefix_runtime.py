import json
import tempfile
import unittest
from pathlib import Path
from experiments.bounded_prefix_allocation import runtime, analysis
from experiments.persistent_prefix_withholding import evaluation

class BoundedRuntimeTests(unittest.TestCase):
    def test_failure_stays_charged_and_cannot_reset(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);runtime.initialize_work(root);ledger=runtime.WorkLedger(root,'fixture')
            def fail():raise RuntimeError('fixture')
            with self.assertRaises(RuntimeError):ledger.call({'adam':1},'fail',fail)
            state=json.loads((root/'WORK.json').read_text())
            self.assertEqual(state['attempted']['adam'],1);self.assertEqual(state['completed']['adam'],0)
            self.assertEqual(len(state['pending']),1)
            with self.assertRaises(FileExistsError):runtime.initialize_work(root)

    def test_counted_calls_and_bounds(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);runtime.initialize_work(root);ledger=runtime.WorkLedger(root,'fixture')
            self.assertEqual(ledger.call({'adam':1,'response_forward':2},'pair',lambda:7),7)
            with self.assertRaises(RuntimeError):ledger.call({'adam':272},'too_many',lambda:None)
            state=json.loads((root/'WORK.json').read_text());self.assertEqual(state['attempted'],state['completed'])
            self.assertEqual(state['completed']['adam'],1)
            with self.assertRaises(ValueError):ledger.call({'reference_generation':1},'forbidden',lambda:None)

    def test_c_order_preserves_fixed_schedule(self):
        requests=[{'panel_ordinal':i} for i in range(128)]
        orders=[[list(range(32*b,32*(b+1))) for b in range(4)] for _ in range(3)]
        old=evaluation.evaluation_schedule(requests,orders)
        new=evaluation.evaluation_schedule(requests,orders,methods=('N','C','S'))
        self.assertEqual(new,[{**r,'method':('C' if r['method']=='E' else r['method'])} for r in old])
        balance=evaluation.schedule_balance(new,methods=('N','C','S'))
        self.assertEqual((balance['scored_generations'],balance['warmups']),(1152,72))
        self.assertTrue(all(x==[4,4,4] for x in balance['positions'].values()))

    def test_decision_needs_control_points_and_each_repeat(self):
        def effect(point,ci):return dict(point=point,ci95=ci)
        cn=dict(time_reduction=effect(.006,[.001,.01]),acceptance_improvement=effect(.01,[.001,.02]),repeat_time_reductions=[.004,.006,.008])
        cs=dict(time_reduction=effect(.001,[-.004,.006]),acceptance_improvement=effect(.002,[-.005,.01]))
        both={'C_vs_N':cn,'C_vs_S':cs}
        self.assertEqual(analysis.decision(both)['outcome'],'ELIGIBLE_FOR_SEPARATE_CONFIRMATION')
        self.assertFalse(analysis.decision(both)['cs_intervals_resolved'])
        cs['time_reduction']['point']=0
        self.assertEqual(analysis.decision(both)['outcome'],'CLOSE_FIXED_CANDIDATE_AT_THIS_SCREEN')
        cs['time_reduction']['point']=.001;cn['repeat_time_reductions'][1]=-.001
        self.assertFalse(analysis.decision(both)['checks']['all_repeats_faster'])

if __name__=='__main__':unittest.main()
