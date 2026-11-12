"""Native acceptance boundaries and explicit work accounting."""
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from experiments.lost_repair.run import block_observation, counted
from experiments.verification_transition.budget import WorkBudget

class LostRepairRunTests(unittest.TestCase):
    def test_rejection_requires_actual_mismatch_after_accepted_prefix(self):
        block=dict(output_start=0,draft_tokens=[2,9,4],anchor_token=1,
                   accepted_draft_tokens=1,finished=False,stop_clipped=False)
        row=block_observation(block,dict(response_ids=[1,2,3,4],stop_token_ids=[0]))
        self.assertTrue(row['genuine_rejection'])
        self.assertEqual(row['first_canonical_mismatch'],2)

    def test_capped_matching_suffix_is_not_rejection(self):
        block=dict(output_start=0,draft_tokens=[2,3,8],anchor_token=1,
                   accepted_draft_tokens=2,finished=True,stop_clipped=True)
        row=block_observation(block,dict(response_ids=[1,2,3],stop_token_ids=[0]))
        self.assertFalse(row['genuine_rejection'])
        self.assertEqual(row['valid_proposals'],2)

    def test_accepted_wrong_token_stops(self):
        block=dict(output_start=0,draft_tokens=[9],anchor_token=1,
                   accepted_draft_tokens=1,finished=False,stop_clipped=False)
        with self.assertRaises(ValueError):
            block_observation(block,dict(response_ids=[1,2],stop_token_ids=[]))

    def test_failure_is_charged_and_optimizer_denied(self):
        with TemporaryDirectory() as root:
            budget=WorkBudget(Path(root)/'work.jsonl',limits={'feedback_capture':128})
            with self.assertRaises(ZeroDivisionError):
                counted(budget,'feedback_capture',lambda:1/0)
            self.assertEqual(budget.summary()['failed'],{'feedback_capture':1})
            for kind in ('optimizer_step','backward','timing_sweep'):
                with self.assertRaises(ValueError):budget.attempt(kind)

if __name__ == '__main__':unittest.main()
