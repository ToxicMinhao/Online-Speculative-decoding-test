"""Small synthetic tests; no experimental observation is generated."""
import copy
import unittest
import diagnostics as analyze


def response(role='A'):
    row = {key:0 for key in analyze.COUNTS}
    row.update(role=role, record_id='synthetic', valid_occurrences=6, protected_occurrences=3,
        blocks=2, old_matching_prefix_tokens=3, new_matching_prefix_tokens=4,
        net_matching_prefix_change=1, old_first_errors=1, original_first_errors=1,
        first_errors_repaired_with_prefix_intact=1, K_useful_repairs=1, C_useful_repairs=1,
        useful_repairs_retained=1, gained_prefix_occurrences=1, native_gained_prefix_occurrences=1,
        native_prefix_gains_retained=1, canonical_margin_ties=6, risk_unaveraged=.2,
        teacher_kl={'prefix':dict(numerator=2.,denominator=2.,mean=1.,occurrences=3),
                    'first_error':dict(numerator=3.,denominator=1.,mean=3.,occurrences=1)})
    return row


class RecordedDiagnosticTests(unittest.TestCase):
    def test_counts_and_teacher_support_must_match(self):
        row=response(); before=copy.deepcopy(row); native=copy.deepcopy(row)
        analyze.verify_response(row,before,native)
        for field in ('useful_repairs_lost','net_matching_prefix_change','native_prefix_gains_lost'):
            bad=copy.deepcopy(row);bad[field]+=1
            with self.assertRaises(ValueError):
                analyze.verify_response(bad,before,native)
        bad=copy.deepcopy(row);bad['teacher_kl']['prefix']['denominator']=4
        with self.assertRaises(ValueError):
            analyze.verify_response(bad,before,native)

    def test_pair_uses_teacher_ratio_of_sums_but_equal_response_draft_risk(self):
        a=response();b=response()
        b['teacher_kl']['prefix'].update(numerator=8.,denominator=4.,mean=2.)
        b['risk_unaveraged']=.4
        pooled=analyze.aggregate_pair([a,b],{'raw':.5*(.2+.4),'value':.5*(.2+.4),'eta':0},
            {},event=5,window=1,role='A',candidate='native')
        self.assertAlmostEqual(pooled['teacher_prefix_mean'],10/6)
        self.assertNotEqual(pooled['teacher_prefix_mean'],1.5)
        self.assertAlmostEqual(pooled['protected_old_draft_kl'],.3)
        self.assertEqual(pooled['response_denominator'],2)
        self.assertEqual(pooled['C_useful_repairs'],2)

    def test_role_and_event_summaries_stay_separate(self):
        rows=[]
        for event in (5,6):
            for role in ('A','D'):
                row=analyze.aggregate_pair([response(role),response(role)],{'raw':.2,'value':.2,'eta':0},
                    {},event=event,window=1,role=role,candidate='native')
                rows.append(row)
        summary=analyze.summarize_events(rows)
        self.assertEqual(len(summary),4)
        self.assertEqual({(r['event'],r['pair']) for r in summary},{(5,'A'),(5,'D'),(6,'A'),(6,'D')})
        self.assertTrue(all(r['window_denominator']==1 for r in summary))

    def test_disjoint_32_pair_offset_wraps_inside_pass(self):
        records=[{'record_id':f'p{p}_r{i}'} for p in range(2) for i in range(128)]
        first=analyze.expected_binding(records,5,1)
        self.assertEqual(first['fitting_ids'],['p0_r0','p0_r1'])
        self.assertEqual(first['off_fit_ids'],['p0_r64','p0_r65'])
        last=analyze.expected_binding(records,6,128)
        self.assertEqual(last['fitting_ids'],['p1_r126','p1_r127'])
        self.assertEqual(last['off_fit_ids'],['p1_r62','p1_r63'])
        self.assertEqual(last['diagnostic_rng_key'],[20261108,0,6,128])

    def test_scalar_mismatch_retained_and_committed_is_not_counterfactual(self):
        production={'native':{'native_norm':2.,'norm':2.},'committed':{'native_norm':2.,'norm':1.},
                    'decision':{'accepted':True,'reason':'actual_risk_decreased'}}
        receipt={'candidates':{'scalar':{'candidate':{'norm':1.01,'native_norm':2.,'target_norm':1.,
                    'relative_norm_mismatch':.01},'decision':{'accepted':False,'reason':'numerical_ambiguity'}}}}
        scalar=analyze.candidate_control('scalar',receipt,production)
        committed=analyze.candidate_control('committed',receipt,production)
        self.assertEqual(scalar['scalar_relative_norm_mismatch'],.01)
        self.assertFalse(scalar['candidate_guard_accepted'])
        self.assertTrue(scalar['diagnostic_only'])
        self.assertFalse(committed['diagnostic_only'])
        self.assertTrue(committed['candidate_guard_accepted'])

    def test_contrast_summary_counts_negative_and_missing_size_without_significance(self):
        rows=[]
        for delta in (-2, 0, 3):
            row=dict(event=5,pair='D',comparison='feasible_fsg_minus_native',
                protected_old_draft_kl_delta=-.1,norm_relative_difference=None)
            for key in ('C_useful_repairs','net_matching_prefix_change','old_prefix_tokens_lost','new_earlier_errors'):
                row[key+'_delta']=delta
            rows.append(row)
        result=analyze.contrast_summary(rows)[0]
        self.assertEqual(result['windows'],3)
        self.assertEqual(result['C_useful_repairs_delta'],1)
        self.assertEqual(result['C_useful_repairs_negative_windows'],1)
        self.assertEqual(result['C_useful_repairs_tied_windows'],1)
        self.assertEqual(result['lower_KL_fewer_repairs_lower_net_windows'],1)
        self.assertIsNone(result['max_absolute_relative_norm_difference'])


if __name__=='__main__':
    unittest.main()
