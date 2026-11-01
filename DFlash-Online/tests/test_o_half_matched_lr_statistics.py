"""Small count fixtures verify the fixed paired analysis and decision map."""
import copy
import unittest

import numpy as np

from experiments.o_half_matched_lr import statistics as S


def measurement(calls):
    return dict(verification_calls=calls, accepted_draft_tokens=40,
                committed_progress=60, output_tokens=60)


def rows(count=4):
    return [dict(seed=seed, event=4, request_id=f'r{i}', panel_ordinal=i,
                 **{method: measurement(calls+i) for method,calls in
                    [('F',22),('O',21),('K',20),('O_HALF',19),('K_HALF',18)]})
            for seed in S.SEEDS for i in range(count)]


def effect(point, low, high):
    return {'call_reduction': {'point_pct':point, 'ci95_pct':[low,high]}}


def decision_input(point=.7, low=.1, high=1.3, seeds=(.6,.8)):
    combined={left+'_vs_'+right:effect(point,low,high) for left,right in S.COMPARISONS}
    per_seed={str(seed):copy.deepcopy(combined) for seed in S.SEEDS}
    for seed,value in zip(S.SEEDS,seeds):
        per_seed[str(seed)]['K_HALF_vs_O_HALF']=effect(value,low,high)
    return combined,per_seed


class MatchedLRStatisticsTests(unittest.TestCase):
    def test_shared_request_bootstrap_and_no_timing_effects(self):
        records=rows()
        for row in records:
            row['O_HALF']['verification_calls']=20
            row['K_HALF']['verification_calls']=10 if row['seed']==17 else 30
            row['K_HALF']['request_latency_s']=float('nan')
        summary=S.summarize(records)
        self.assertEqual(summary['combined']['K_HALF_vs_O_HALF']['call_reduction']['ci95_pct'],[0,0])
        self.assertEqual(summary['request_count'],4)
        self.assertEqual(summary['request_seed_count'],8)
        self.assertEqual(summary['combined']['K_HALF_vs_O_HALF']['request_seed_wins'],4)
        self.assertEqual(summary['interpretation']['branch'],'D')
        self.assertFalse(any('latency' in key for key in summary['combined']['K_HALF_vs_O_HALF']))
        expected=np.random.Generator(np.random.PCG64(20261102)).integers(0,4,size=(10000,4))
        np.testing.assert_array_equal(S.bootstrap_indices(4),expected)

    def test_ratio_of_sums_and_reverse_denominator(self):
        indices=np.array([[0,1],[0,1]])
        direct=S.comparison(np.array([[1,99]]),np.array([[2,100]]),indices)
        self.assertAlmostEqual(direct['call_reduction']['point_pct'],100*(1-100/102))
        self.assertNotAlmostEqual(direct['call_reduction']['point_pct'],25.5)
        result=S.summarize(rows())
        forward=result['combined']['K_HALF_vs_O_HALF']['call_reduction']['point_pct']
        reverse=result['reverse_primary']['combined']['call_reduction']['point_pct']
        self.assertNotAlmostEqual(reverse,-forward)

    def test_interaction_uses_common_F_denominator(self):
        calls={m:np.array([[v,v]]) for m,v in [('F',20),('O',20),('K',18),('O_HALF',15),('K_HALF',14)]}
        result=S.objective_lr_interaction(calls,np.array([[0,1],[0,1]]))
        self.assertEqual(result['point_pct'],-5.)
        self.assertEqual(result['ci95_pct'],[-5.,-5.])
        self.assertEqual(result['numerator_call_difference'],-2)
        self.assertNotAlmostEqual(result['point_pct'],100*(1-14/15)-100*(1-18/20))

    def test_gain_harm_before_seed_pooling_and_zero_gain(self):
        f=np.array([[10,10],[10,10]])
        b=np.array([[5,15],[15,5]])
        a=np.array([[6,11],[13,4]])
        result=S.gain_harm(f,b,a,S.bootstrap_indices(2))
        self.assertEqual(result['comparator_gross_gain']['point'],10)
        self.assertEqual(result['retained_gain']['point'],9)
        self.assertEqual(result['lost_gain']['point'],1)
        self.assertEqual(result['added_gain']['point'],1)
        self.assertEqual(result['prevented_harm']['point'],6)
        self.assertEqual(result['net_calls_saved_vs_comparator'],6)
        self.assertTrue(result['identity_verified_at_each_seed_request'])
        empty=S.gain_harm(f,f,a,S.bootstrap_indices(2))
        self.assertIsNone(empty['retained_gain_fraction']['point'])
        self.assertIsNone(empty['retained_gain_fraction']['ci95'])

    def test_A_requires_threshold_interval_and_both_seed_directions(self):
        c,p=decision_input();self.assertEqual(S.interpret(c,p)['branch'],'A')
        self.assertFalse(S.interpret(c,p)['true_effect_above_priority_target_supported'])
        c,p=decision_input(.49,.1,.8);self.assertEqual(S.interpret(c,p)['branch'],'D')
        c,p=decision_input(.7,-.1,1.3);self.assertEqual(S.interpret(c,p)['branch'],'D')
        c,p=decision_input(.7,.1,1.3,(-.1,1.5));self.assertEqual(S.interpret(c,p)['branch'],'D')

    def test_B_requires_entire_interval_and_consistent_seed_points(self):
        c,p=decision_input(.1,-.5,.5,(-.2,.4))
        self.assertEqual(S.interpret(c,p)['branch'],'B')
        c,p=decision_input(.1,-.3,.4,(-.6,.8))
        result=S.interpret(c,p)
        self.assertEqual(result['branch'],'D')
        self.assertTrue(result['pooled_practical_equivalence'])
        self.assertFalse(result['equivalence_seed_point_consistency'])
        c,p=decision_input(.1,-.6,.6,(0,.2))
        self.assertFalse(S.interpret(c,p)['pooled_practical_equivalence'])

    def test_C_and_E(self):
        c,p=decision_input(-.7,-1.2,-.1,(-.6,-.8))
        self.assertEqual(S.interpret(c,p)['branch'],'C')
        c,p=decision_input(-.7,-1.2,-.1,(.1,-1.5))
        self.assertEqual(S.interpret(c,p)['branch'],'D')
        result=S.interpret({}, {}, valid=False)
        self.assertEqual(result['branch'],'E')
        self.assertFalse(result['automatic_followup_authorized'])

    def test_pair_identity_integer_count_and_all_secondary_contracts(self):
        records=rows()
        with self.assertRaises(ValueError):S.summarize(records[:-1])
        with self.assertRaises(ValueError):S.summarize(records+[records[0]])
        records[0]['O']['verification_calls']=1.5
        with self.assertRaisesRegex(ValueError,'Fractional'):S.summarize(records)
        summary=S.summarize(rows())
        self.assertEqual(len(summary['combined']),8)
        self.assertEqual(set(summary['gain_harm']['combined']),
                         {'K_HALF_vs_O_HALF','O_HALF_vs_O','K_HALF_vs_K'})
        self.assertEqual(set(summary['method_totals']['combined']),set(S.METHODS))


if __name__=='__main__':
    unittest.main()
