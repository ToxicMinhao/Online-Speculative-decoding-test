"""CPU checks for sealed same-pass reference identities and anchor bindings."""
from copy import deepcopy
import unittest

from online_feedback.reference_schedule import build_reference_schedule, validate_reference_schedule


def original_receipt():
    records = []
    for pass_index in range(2):
        order = list(range(128)) if pass_index == 0 else list(reversed(range(128)))
        for offset, identity in enumerate(order):
            records.append(dict(record_id=f'r{identity}', **{'pass': pass_index},
                sampled_anchors=[dict(positions=[[identity+pass_index*1000, identity+pass_index*1000+1]],
                                      keep_mask=[[True, offset % 3 != 0]])],
                optimizer_stepped=bool(offset%2), optimizer_step_after=pass_index*64+(offset+1)//2))
    return dict(status='completed', objective='osd', passes=2, optimizer_steps=128,
                response_presentations=256, start_step=0, end_step=128,
                features=[dict(record_id=f'r{i}',path=f'feature-{i}.pt') for i in range(128)],
                records=records)


class ReferenceScheduleTests(unittest.TestCase):
    def setUp(self):
        self.receipt=original_receipt()

    def test_all_windows_are_next_disjoint_pair_within_the_same_pass(self):
        result=build_reference_schedule(self.receipt,evaluation_ids=['eval0'])
        self.assertEqual(len(result['windows']),128)
        for row in result['windows']:
            p,b=row['pass_index'],row['batch_index']
            self.assertEqual(row['reference_batch_index'],(b+1)%64)
            self.assertFalse(set(row['learning_record_ids']) & set(row['reference_record_ids']))
            expected=self.receipt['records'][p*128+2*((b+1)%64):p*128+2*((b+1)%64)+2]
            self.assertEqual(row['reference'],[{key:r[key] for key in ('record_id','pass','sampled_anchors')}
                                                for r in expected])
        self.assertEqual(result['windows'][63]['reference_record_ids'],['r0','r1'])
        self.assertEqual(result['windows'][64]['learning_record_ids'],['r127','r126'])
        self.assertEqual(result['windows'][127]['reference_record_ids'],['r127','r126'])
        self.assertTrue(validate_reference_schedule(result,self.receipt,evaluation_ids=['eval0'])['passed'])

    def test_reference_uses_its_own_pass_anchors_and_detached_json_copy(self):
        result=build_reference_schedule(self.receipt)
        source=self.receipt['records'][130]['sampled_anchors']
        self.assertEqual(result['windows'][64]['reference'][0]['sampled_anchors'],source)
        self.assertNotEqual(result['windows'][64]['reference'][0]['sampled_anchors'],
                            result['windows'][0]['reference'][0]['sampled_anchors'])
        source[0]['positions'][0][0]=-1
        self.assertNotEqual(result['windows'][64]['reference'][0]['sampled_anchors'],source)
        with self.assertRaisesRegex(ValueError,'differs'):
            validate_reference_schedule(result,self.receipt)

    def test_editing_reference_to_learning_or_other_pass_is_rejected(self):
        result=build_reference_schedule(self.receipt)
        for replacement in (result['windows'][0]['learning'],result['windows'][64]['reference']):
            changed=deepcopy(result);changed['windows'][0]['reference']=replacement
            with self.assertRaisesRegex(ValueError,'differs'):
                validate_reference_schedule(changed,self.receipt)

    def test_evaluation_overlap_and_incomplete_or_duplicate_pass_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'Evaluation'):
            build_reference_schedule(self.receipt,evaluation_ids=['r3'])
        for mutate in (lambda r:r['records'].pop(),
                       lambda r:r['records'][1].update(record_id='r0'),
                       lambda r:r['records'][128].update(**{'pass':0}),
                       lambda r:r['records'][1].update(optimizer_stepped=False)):
            bad=deepcopy(self.receipt);mutate(bad)
            with self.assertRaises(ValueError):build_reference_schedule(bad)


if __name__=='__main__':unittest.main()
