import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from experiments.verification_transition.contract import (
    select_requests, interleaved_requests, neutral_boundaries,
)
from experiments.verification_transition.budget import (
    BudgetExceeded, BudgetService, WorkBudget, connect_budget,
)


class SelectionTests(unittest.TestCase):
    def test_outcome_independent_exact_selection_and_interleaving(self):
        panel = [{'request_id': str(i), 'length': 999-i, 'improvement': i} for i in range(256)]
        chosen = {dataset: select_requests(dataset, panel) for dataset in ('gsm8k', 'mathqa')}
        expected = sorted((hashlib.sha256(('kplus_transition_replay_v1\ngsm8k\n'+str(i)).encode()).hexdigest(),str(i)) for i in range(256))[:64]
        self.assertEqual([r['request_id'] for r in chosen['gsm8k']], [v for _,v in expected])
        changed = list(reversed([{'request_id':str(i),'length':0,'improvement':-i} for i in range(256)]))
        self.assertEqual(chosen['gsm8k'], select_requests('gsm8k', changed))
        combined = interleaved_requests(chosen)
        self.assertEqual([(r['selection_rank'],r['dataset']) for r in combined[:4]],
                         [(0,'gsm8k'),(0,'mathqa'),(1,'gsm8k'),(1,'mathqa')])
        self.assertEqual(len(combined),128)

    def test_duplicate_and_incomplete_panels_fail(self):
        for panel in ([{'request_id':'x'}]*256, [{'request_id':str(i)} for i in range(255)]):
            with self.assertRaises(ValueError):select_requests('gsm8k',panel)

    def test_neutral_boundary_budget_and_terminal_rules(self):
        self.assertEqual(neutral_boundaries([]),[])
        self.assertEqual(neutral_boundaries([{'terminal':True}]),[])
        self.assertEqual(neutral_boundaries([{'terminal':False},{'terminal':True}]),[1])
        self.assertEqual(neutral_boundaries([{'terminal':False}]*4+[{'terminal':True}]),[1,3])


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.clock=[0.0]
        self.budget=WorkBudget(Path(self.tmp.name)/'WORK.jsonl', limits={'draft_block':2},
                               gpu_seconds=10,clock=lambda:self.clock[0])

    def test_failed_unknown_and_completed_all_consume_allowance(self):
        first=self.budget.attempt('draft_block',operation='proposal_only')
        self.budget.fail(first,'native failure')
        self.budget.attempt('draft_block',operation='proposal_and_verifier')
        with self.assertRaises(BudgetExceeded):self.budget.attempt('draft_block')
        summary=self.budget.summary()
        self.assertEqual(summary['attempted'],{'draft_block':2})
        self.assertEqual(summary['failed'],{'draft_block':1})
        self.assertEqual(summary['unknown'],{'draft_block':1})
        self.assertEqual(summary['remaining']['draft_block'],0)
        rows=[json.loads(x) for x in self.budget.path.read_text().splitlines()]
        self.assertEqual(rows[-1]['kind'],'budget_stop')

    def test_gpu_occupancy_is_summed_and_loading_included(self):
        self.budget.start_gpu(2);self.clock[0]=4.5
        self.assertEqual(self.budget.summary()['gpu_seconds'],9)
        self.clock[0]=5
        with self.assertRaises(BudgetExceeded):self.budget.attempt('model_load')
        self.budget.end_gpu();self.clock[0]=100
        self.assertEqual(self.budget.summary()['gpu_seconds'],10)

    def test_target_call_is_separate_from_block(self):
        a=self.budget.attempt('draft_block');b=self.budget.attempt('target_call')
        self.budget.complete(b);self.budget.complete(a)
        self.assertEqual(self.budget.summary()['completed'],{'target_call':1,'draft_block':1})
        with self.assertRaises(ValueError):self.budget.attempt('backward')

    def test_cross_process_clients_share_one_limit(self):
        service=BudgetService(self.budget);self.addCleanup(service.close)
        first=connect_budget(service.connection);second=connect_budget(service.connection)
        a=first.attempt('draft_block');second.complete(a)
        second.attempt('draft_block')
        with self.assertRaises(BudgetExceeded):first.attempt('draft_block')
        self.assertEqual(self.budget.summary()['attempted'],{'draft_block':2})


if __name__=='__main__':unittest.main()
