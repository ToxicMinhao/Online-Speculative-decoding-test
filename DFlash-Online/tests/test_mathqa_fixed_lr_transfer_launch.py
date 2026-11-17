"""CPU tests for real campaign ordering and bounded work accounting."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.mathqa_fixed_lr_transfer.launch import production_groups, remaining_cost, require_cross_card_identity
from experiments.mathqa_fixed_lr_transfer.runtime import WorkLedger, initialize_work, LIMITS, require_time_for
from experiments.bounded_prefix_allocation.staging import check_inputs


class LaunchTests(unittest.TestCase):
    def test_all_resource_schedules_preserve_own_feedback_dependencies(self):
        plan=dict(devices=dict(A='gpuA',B='gpuB'))
        for same,two,mixed in ((True,False,False),(False,True,True),(False,False,True),(False,False,False)):
            groups=production_groups(plan,same_card=same,two_trainers=two,mixed_workers=mixed)
            done=set()
            for jobs in groups:
                self.assertEqual(len({j[1] for j in jobs}),len(jobs))
                for name,slot,module,args in jobs:
                    if name.endswith('_collect6'):self.assertIn(name[0]+'5',done)
                    elif name in ('N6','S6'):self.assertIn(name[0]+'_collect6',done)
                    elif name=='final':
                        self.assertTrue({'N6','S6'}<=done)
                        self.assertEqual((len(jobs),slot),(1,'A'))
                    if same:self.assertEqual(slot,'A')
                done.update(j[0] for j in jobs)
            self.assertEqual(done,{'N5','S5','N_collect6','S_collect6','N6','S6','final'})

    def test_parallel_work_charges_both_cards(self):
        jobs=production_groups(dict(devices=dict(A='a',B='b')),same_card=False,two_trainers=True,mixed_workers=True)
        costs={j[0]:10 for group in jobs for j in group}
        self.assertEqual(remaining_cost(jobs,costs),(40,70))

    def test_failed_work_is_preserved_and_not_counted_as_completed(self):
        with tempfile.TemporaryDirectory() as root:
            initialize_work(root)
            ledger=WorkLedger(root,'test')
            def fail():raise RuntimeError('native failure')
            with self.assertRaisesRegex(RuntimeError,'native failure'):
                ledger.call({'adam':1},'first',fail)
            work=json.loads((Path(root)/'WORK.json').read_text())
            self.assertEqual(work['attempted']['adam'],1)
            self.assertEqual(work['completed']['adam'],0)
            self.assertEqual(work['pending']['0']['label'],'first')
            with self.assertRaises(FileExistsError):initialize_work(root)

    def test_budget_refusal_does_not_execute_callable(self):
        with tempfile.TemporaryDirectory() as root:
            initialize_work(root)
            called=[]
            with self.assertRaisesRegex(RuntimeError,'cap exceeded'):
                WorkLedger(root,'test').call({'adam':LIMITS['adam']+1},'too many',lambda:called.append(1))
            self.assertEqual(called,[])

    def test_completed_work_reconciles(self):
        with tempfile.TemporaryDirectory() as root:
            initialize_work(root)
            self.assertEqual(WorkLedger(root,'test').call({'response_forward':1},'forward',lambda:7),7)
            work=json.loads((Path(root)/'WORK.json').read_text())
            self.assertFalse(work['pending'])
            self.assertEqual(work['attempted'],work['completed'])

    def test_occupied_limit_is_independent_of_elapsed_limit(self):
        with patch('experiments.mathqa_fixed_lr_transfer.runtime.clock_status',
                   return_value=dict(elapsed_s=100,occupied_s=13000)):
            with self.assertRaises(RuntimeError):require_time_for({'root':'x'},100,1000)
        with patch('experiments.mathqa_fixed_lr_transfer.runtime.clock_status',
                   return_value=dict(elapsed_s=13000,occupied_s=100)):
            with self.assertRaises(RuntimeError):require_time_for({'root':'x'},1000,100)

    def test_cross_card_scientific_difference_allows_same_card_fallback(self):
        for error in ("native: cross-card preclip", "scaled: cross-card endpoint",
                      "serving: cross-card native counts"):
            self.assertFalse(require_cross_card_identity(dict(status="failed",
                same_card_checks_passed=True,errors=[error])))
        self.assertTrue(require_cross_card_identity(dict(status="passed",
            same_card_checks_passed=True,errors=[])))

    def test_cross_card_provenance_difference_must_stop(self):
        for error in ("Cross-card binding", "Cross-card execution_commit",
                      "Cross-card origin_before", "Cross-card origin_after",
                      "A: incomplete same-card qualification", "unknown discrepancy"):
            with self.assertRaisesRegex(ValueError,"provenance"):
                require_cross_card_identity(dict(status="failed",
                    same_card_checks_passed=True,errors=[error]))
        with self.assertRaisesRegex(ValueError,"same-card"):
            require_cross_card_identity(dict(status="failed",
                same_card_checks_passed=False,errors=["native: cross-card forward"]))

    def test_staging_reader_does_not_accept_arbitrary_module(self):
        with self.assertRaisesRegex(ValueError,'Unknown staged'):
            check_inputs({},Path('.'),'arbitrary.module')


if __name__=='__main__':unittest.main()
