"""Pre-outcome sealing and archived-native parity are correctness gates."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from experiments.repair_history_margin.run import seal_pre_k6, check_seal, validate_parent, validate_commits, GPU_SECONDS
from experiments.verification_transition.budget import WorkBudget
from experiments.lost_repair.run import counted


def record():
    block = dict(output_start=0, anchor_token=1, draft_tokens=[2,3], target_argmax=[2,3,4],
                 positions=[0,1,2], accepted_draft_tokens=2,native_accepted_draft_tokens=2,
                 committed_progress=3, committed_tokens=[2,3,4],finished=True,stop_clipped=False)
    return dict(prompt_ids=[8], response_ids=[1,2,3,4], finish_reason='eos',
                max_new_tokens=2048,stop_token_ids=[4],metadata={'trace':[block]})


class RunnerTests(unittest.TestCase):
    def test_seal_is_durable_immutable_and_checked(self):
        with TemporaryDirectory() as root:
            seal = seal_pre_k6({'labels':[], 'matched_pairs':[]}, root)
            self.assertFalse(seal['k6_loaded'])
            check_seal(root, seal)
            with self.assertRaises(FileExistsError):
                seal_pre_k6({}, root)
            (Path(root)/'PRE_K6.json').write_text('{}')
            with self.assertRaises(ValueError):check_seal(root, seal)

    def test_native_trace_must_match_all_proposals_even_if_outputs_match(self):
        expected, observed = record(), record()
        self.assertEqual(validate_parent(observed, expected, expected)['status'],'passed')
        observed['metadata']['trace'][0]['draft_tokens'][-1] = 99
        result = validate_parent(observed, expected, expected)
        self.assertEqual(result['status'],'failed')
        self.assertIn('archive:native_blocks',result['differences'])

    def test_missing_archive_disclosed_and_canonical_still_required(self):
        result=validate_parent(record(),record(),None)
        self.assertFalse(result['archived_trace_available'])
        changed=record();changed['finish_reason']='length'
        self.assertEqual(validate_parent(changed,record(),None)['status'],'failed')

    def test_committed_target_repair_must_remain_canonical(self):
        saved=record(); block=saved['metadata']['trace'][0]
        validate_commits(block,saved)
        block['committed_tokens'][-1]=99
        with self.assertRaises(ValueError):validate_commits(block,saved)

    def test_three_hour_aggregate_clock_and_no_training(self):
        self.assertEqual(GPU_SECONDS,10800)
        clock=[0.0]
        with TemporaryDirectory() as root:
            budget=WorkBudget(Path(root)/'work.jsonl',gpu_seconds=GPU_SECONDS,
                              limits={'parent_walk':1},clock=lambda:clock[0])
            budget.start_gpu(1)
            for kind in ('optimizer_step','backward','feedback_generation','timing_sweep'):
                with self.assertRaises(ValueError):budget.attempt(kind)
            with self.assertRaises(ZeroDivisionError):counted(budget,'parent_walk',lambda:1/0)
            self.assertEqual(budget.summary()['failed'],{'parent_walk':1})
            clock[0]=10800
            with self.assertRaises(RuntimeError):budget.check_time()


if __name__=='__main__':unittest.main()
