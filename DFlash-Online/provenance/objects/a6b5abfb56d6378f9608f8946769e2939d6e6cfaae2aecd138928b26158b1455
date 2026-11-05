"""Small independent accounting tests; no model or native runtime imports."""
import unittest
from audit_results import expected_slots, totals_and_comparison, leave_one_out


class AuditMathTests(unittest.TestCase):
    def test_balanced_384_slots_and_included_gate(self):
        requests=[dict(request_id=f'r{i}',panel_ordinal=i) for i in range(64)]
        slots=expected_slots(requests)
        self.assertEqual(len(slots),384)
        self.assertEqual(len(set(slots)),384)
        self.assertEqual(sum(rid in {f'r{i}' for i in range(8)} for rid,_,_ in slots),48)
        for repeat in range(3):
            first=[arm for index,(_,rep,arm) in enumerate(slots) if rep==repeat and index%2==0]
            self.assertEqual(first.count('K1'),32)
            self.assertEqual(first.count('KRC'),32)

    def test_pooled_primary_and_loo_preserve_all_requests(self):
        requests=[dict(request_id=f'r{i}',panel_ordinal=i) for i in range(3)]
        calls={'K1':[5,10,15],'KRC':[3,11,15]}
        rows=[]
        for i,request in enumerate(requests):
            for arm in calls:
                for repeat in range(3):
                    rows.append(dict(request_id=request['request_id'],method=arm,repeat=repeat,
                        verification_calls=calls[arm][i],accepted_draft_tokens=50-calls[arm][i],
                        committed_progress=50,output_tokens=51,request_latency_s=calls[arm][i]*(repeat+1)))
        totals,comparison,values=totals_and_comparison(rows,requests)
        self.assertEqual(totals['K1']['verification_calls'],30)
        self.assertEqual(totals['KRC']['verification_calls'],29)
        self.assertEqual(comparison['call_request_directions'],dict(improved=1,regressed=1,tied=1))
        self.assertEqual((comparison['call_gross_saved'],comparison['call_gross_extra']),(2,1))
        self.assertAlmostEqual(comparison['call_reduction_pct'],100/30)
        omissions=leave_one_out(values,requests)
        self.assertEqual([row['net_calls_saved'] for row in omissions],[-1,2,1])
        self.assertEqual([row['remaining_requests'] for row in omissions],[2,2,2])
        self.assertEqual(totals['K1']['verification_calls'],30)
        self.assertEqual(totals['K1']['summed_mean_request_latency_s'],60)

    def test_missing_repeat_cannot_be_a_complete_sample(self):
        with self.assertRaisesRegex(ValueError,'Incomplete'):
            totals_and_comparison([], [dict(request_id='q',panel_ordinal=0)])


if __name__=='__main__':
    unittest.main()
