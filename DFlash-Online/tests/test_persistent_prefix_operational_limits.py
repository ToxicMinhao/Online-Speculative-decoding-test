import copy
import json
from pathlib import Path
import tempfile
import unittest

from experiments.persistent_prefix_withholding.resources import LIMITS, PRODUCTION_AMENDMENT, effective_limits
from experiments.persistent_prefix_withholding.task_runner import SharedWork
from online_feedback.storage import write_json


def amended_plan():
    amendment = dict(authority=PRODUCTION_AMENDMENT, scientific_binding='original',
        original_limits=dict(LIMITS), effective_limits=dict(LIMITS, adam=650,
        response_forward=1316, response_backward=1316, training_initialization=8))
    return dict(binding='original', limits=dict(LIMITS), operational_amendment=amendment)


class OperationalLimitsTest(unittest.TestCase):
    def test_narrow_amendment_leaves_science_and_original_limits_unchanged(self):
        plan = amended_plan()
        before = copy.deepcopy(plan)
        self.assertEqual(effective_limits({}), LIMITS)
        self.assertEqual(effective_limits(plan)['adam'],650)
        self.assertEqual(plan,before)
        self.assertEqual(LIMITS['adam'],648)
        for field in ('binding','limits'):
            altered=copy.deepcopy(plan); altered[field]='changed'
            with self.assertRaises(ValueError): effective_limits(altered)
        for key in LIMITS:
            altered=copy.deepcopy(plan); altered['operational_amendment']['effective_limits'][key]+=1
            with self.assertRaises(ValueError): effective_limits(altered)

    def test_shared_ledger_uses_effective_cap_and_carries_prior_work(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); plan=amended_plan()
            write_json(root/'sealed/PLAN.json',plan)
            counts=dict.fromkeys(LIMITS,0);counts['adam']=649
            ledger=dict(attempted=dict(counts),completed=dict(counts),pending={},next_operation=0,
                limits=effective_limits(plan),scientific_binding=plan['binding'],
                operational_amendment=plan['operational_amendment'])
            write_json(root/'WORK.json',ledger)
            work=SharedWork(root,'N8'); operation=work.reserve({'adam':1},'last');work.complete(operation)
            self.assertEqual(json.loads((root/'WORK.json').read_text())['completed']['adam'],650)
            with self.assertRaises(RuntimeError):work.reserve({'adam':1},'unauthorized')
            changed=json.loads((root/'WORK.json').read_text());changed['limits']['adam']=651
            write_json(root/'WORK.json',changed)
            with self.assertRaises(ValueError):work.reserve({'adam':0},'changed-cap')

    def test_amended_ledger_cannot_omit_amendment(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);write_json(root/'sealed/PLAN.json',amended_plan())
            write_json(root/'WORK.json',dict(attempted=dict.fromkeys(LIMITS,0)))
            with self.assertRaises(ValueError):SharedWork(root,'N8').reserve({'adam':1},'missing')

if __name__=='__main__':unittest.main()
