"""The routing cost proxy is causal and separate from measured request gains."""
import copy
import unittest
from .costs import profile_from_native, observe

class CostTests(unittest.TestCase):
    def profile(self):
        raw={'paths':[{'modules':k,'repeat_s':[.001*(k+1)]*20,'mean_s':.001*(k+1)} for k in range(3)],
             'last_canonical_context_length':100}
        observations=[{'context_length':100,'host_forward_s':.01,'device_elapsed_s':.009}]
        return profile_from_native(raw,observations)

    def test_profile_and_context_extrapolation_declared(self):
        result=self.profile()
        self.assertEqual(set(result['bins']),{'8192','16384','32768'})
        self.assertAlmostEqual(result['bins']['8192']['cycle_s']['2'],.012)
        self.assertFalse(result['bins']['8192']['extrapolated'])
        self.assertTrue(result['bins']['32768']['extrapolated'])

    def test_observed_path_increment_removed_once_before_update(self):
        result=self.profile();untouched=copy.deepcopy(result['bins']['8192'])
        observe(result,[{'context_length':9000,'host_forward_s':.02,'device_elapsed_s':.022}],2)
        self.assertAlmostEqual(result['bins']['16384']['cycle_s']['0'],.020)
        self.assertAlmostEqual(result['bins']['16384']['cycle_s']['2'],.022)
        self.assertEqual(result['bins']['8192'],untouched)

    def test_no_observation_does_not_mutate_profile(self):
        result=self.profile();before=copy.deepcopy(result)
        observe(result,[],0)
        self.assertEqual(result,before)

    def test_incomplete_profile_rejected(self):
        with self.assertRaises(ValueError):profile_from_native({'paths':[]},[])
