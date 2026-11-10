"""Reviewed same-layout qualification retains historical work and strict guards."""
import json
from pathlib import Path
import tempfile
import unittest

from experiments.functional_step_geometry.resume_qualification import (
    POLICY, PRIOR_COUNTS, admit_resume, correction_activity)
from experiments.functional_step_geometry.seal_revised_qualification import remaining_work_plan
from experiments.functional_step_geometry.work import LIMITS, QualificationWork


class RevisedQualificationTests(unittest.TestCase):
    def test_remaining_work_fits_original_cumulative_limits(self):
        remaining=remaining_work_plan()
        for key in ('adam','ordinary_forward','ordinary_backward','additional_backward','serving_generation'):
            self.assertEqual(PRIOR_COUNTS.get(key+'_attempted',0)+remaining[key],LIMITS[key])
        self.assertLessEqual(PRIOR_COUNTS['additional_forward_attempted']+
                             remaining['additional_forward_maximum'],LIMITS['additional_forward'])
        self.assertEqual(sum(remaining['additional_backward_parts'].values()),20)
        self.assertEqual(remaining['completed_adam_steps_repeated'],0)

    def test_policy_keeps_tolerances_and_guards(self):
        self.assertEqual((POLICY['atol'],POLICY['rtol'],POLICY['maximum_relative_l2']),
                         (1e-5,.016,.016))
        self.assertEqual(POLICY['production_head_chunk_blocks'],8)
        self.assertEqual(POLICY['native_fixture_anchor_count'],9)
        self.assertEqual(POLICY['norm_guard'],'unchanged')
        self.assertIn('failed',POLICY['historical_cross_layout'])

    def _prior(self,root):
        previous=root/'previous'; (previous/'qualification').mkdir(parents=True)
        ledger=QualificationWork(previous/'QUALIFICATION_WORK.jsonl')
        for kind in ['adam','ordinary_forward','ordinary_backward','additional_forward','additional_backward']:
            for _ in range(PRIOR_COUNTS[kind+'_attempted']):
                ledger.record(kind,'attempted');ledger.record(kind,'completed')
        (previous/'qualification/QUALIFICATION.json').write_text(json.dumps(dict(
            status='failed',work=PRIOR_COUNTS,error='Native chunked/unchunked oracle failed')))
        return previous,dict(revised_qualification=dict(previous_root=str(previous),policy=POLICY))

    def test_admission_copies_old_attempts_without_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);previous,plan=self._prior(root)
            new=root/'new';new.mkdir()
            _,work=admit_resume(plan,new)
            self.assertEqual(work.summary(),PRIOR_COUNTS)
            self.assertEqual((new/'QUALIFICATION_WORK.jsonl').read_bytes(),
                             (previous/'QUALIFICATION_WORK.jsonl').read_bytes())
            with self.assertRaises(FileExistsError):admit_resume(plan,new)

    def test_changed_historical_accounting_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);previous,plan=self._prior(root)
            with (previous/'QUALIFICATION_WORK.jsonl').open('a') as f:
                f.write(json.dumps(dict(kind='adam',status='attempted'))+'\n')
            new=root/'new';new.mkdir()
            with self.assertRaisesRegex(ValueError,'accounting'):admit_resume(plan,new)
            self.assertFalse((new/'QUALIFICATION_WORK.jsonl').exists())

    def test_all_fallback_is_correctness_valid_but_not_pilot_activity(self):
        row=dict(decision=dict(accepted=False,reason='rounded_norm_exceeds_native'),
                 committed=dict(identical_native=True),raw_fsg=dict(norm=1.0001,native_norm=1.))
        data={'gsm8k':dict(cases=[dict(case='FSG',reused_step=row)]),
              'mathqa':dict(cases=[dict(case='FSG',hook=dict(steps=[row]))])}
        result=correction_activity(data)
        self.assertTrue(result['all_fsg_fallback'])
        self.assertEqual(result['nontrivial_accepted_fsg_cases'],0)
        row['decision']=dict(accepted=True,reason='accepted')
        row['committed']=dict(identical_native=False)
        self.assertEqual(correction_activity(data)['nontrivial_accepted_fsg_cases'],2)


if __name__=='__main__':unittest.main()
