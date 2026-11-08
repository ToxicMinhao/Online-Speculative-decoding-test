"""CPU oracles for consequence reporting; these introduce no acceptance gate."""
import json
import math
import unittest

import torch

from online_feedback import functional_geometry_math as geometry
from online_feedback.fsg_consequence_diagnostic import (
    compare_endpoints, compare_sketches, decision_summary, prefix_consequences,
)


def fixture(size=13):
    generator = torch.Generator().manual_seed(17)
    origin = (torch.randn(size, generator=generator).bfloat16(),)
    native = (origin[0]+torch.randn(size, generator=generator).bfloat16()*.02,)
    native[0][::3] = origin[0][::3]
    gradients = [(torch.randn(size, generator=generator).bfloat16(),) for _ in range(2)]
    left = geometry.make_sketch(gradients, [.5, .9], origin, native)
    changed = [(gradients[0][0]+torch.linspace(-.02, .02, size).bfloat16(),), gradients[1]]
    right = geometry.make_sketch(changed, [.51, .89], origin, native)
    return origin, native, left, right


def dense_proposal(sketch, origin, native):
    delta = native[0].float()-origin[0].float()
    for gradients, scale, coefficient in zip(sketch.gradients, sketch.normalizers, sketch.coefficients):
        delta = delta-gradients[0].float()*float(scale*coefficient)
    return delta


class ConsequenceTests(unittest.TestCase):
    def test_continuous_comparison_matches_dense_and_existing_writer(self):
        origin, native, left, right = fixture()
        report = compare_sketches(left, right, origin, native, chunk_size=4)
        a, b = [dense_proposal(s, origin, native).double() for s in (left, right)]
        self.assertAlmostEqual(report['difference_norm'], float((a-b).norm()), places=14)
        self.assertAlmostEqual(report['direction_cosine'], float((a@b)/(a.norm()*b.norm())), places=14)
        destination = (torch.nn.Parameter(origin[0].clone()),)
        written = geometry.write_candidate(destination, origin, native, left, chunk_size=4)
        self.assertAlmostEqual(report['left_norm'], written['continuous_norm'], places=14)
        self.assertTrue(torch.equal(destination[0], (origin[0].float()+a.float()).bfloat16()))

    def test_rounded_endpoints_and_zero_native_coordinates_are_explicit(self):
        origin = (torch.tensor([.05, .05, .05], dtype=torch.bfloat16),)
        native = (origin[0].clone(),); native[0][0] += .001
        left = (native[0].clone(),); left[0][1] += .001
        right = (native[0].clone(),); right[0][1] += .002
        report = compare_endpoints(left, right, origin, native, chunk_size=2)
        self.assertEqual(report['differing_coordinates'], 1)
        self.assertEqual(report['differing_zero_native_coordinates'], 1)
        self.assertEqual(report['left_changed_zero_native_coordinates'], 1)
        self.assertGreater(report['difference_on_zero_native_norm'], 0.)
        self.assertFalse(report['identical'])
        self.assertFalse(report['left_identical_native'])

    def test_gate_disagreement_and_both_fallback_keep_raw_differences(self):
        origin, native, left, right = fixture()
        a, b = [(native[0].clone(),) for _ in range(2)]
        b[0][0] += .01
        raw = compare_endpoints(a, b, origin, native)
        self.assertFalse(raw['identical'])
        same = compare_endpoints(native, native, origin, native)
        risk = lambda value: geometry.Risk(value, value, 1e-12, 0.)
        accept = geometry.gate(risk(.5), risk(1.), 1., 1.)
        reject = geometry.gate(risk(2.), risk(1.), 1., 1.)
        self.assertTrue(decision_summary(accept, reject, raw, same)['acceptance_disagrees'])
        report = decision_summary(reject, reject, raw, same)
        self.assertTrue(report['both_fallback_with_distinct_raw_endpoints'])
        self.assertTrue(report['committed_endpoints_identical'])
        self.assertFalse(report['raw_endpoints_identical'])

    def test_larger_vector_chunking_and_no_input_mutation(self):
        origin, native, left, right = fixture(1003)
        before = [v.clone() for v in (*origin, *native, *left.gradients[0], *right.gradients[0])]
        a = compare_sketches(left, right, origin, native, chunk_size=7)
        b = compare_sketches(left, right, origin, native, chunk_size=2000)
        for key in a:
            if isinstance(a[key], float):
                self.assertAlmostEqual(a[key], b[key], places=12)
            else:
                self.assertEqual(a[key], b[key])
        for actual, expected in zip((*origin, *native, *left.gradients[0], *right.gradients[0]), before):
            self.assertTrue(torch.equal(actual, expected))
        json.dumps(a, allow_nan=False)

    def test_endpoint_identity_does_not_use_cancellation_prone_displacements(self):
        origin = (torch.tensor([1e20], dtype=torch.bfloat16),)
        left = (torch.tensor([1.], dtype=torch.bfloat16),)
        right = (torch.tensor([2.], dtype=torch.bfloat16),)
        report = compare_endpoints(left, right, origin, origin)
        self.assertFalse(report['identical'])
        self.assertEqual(report['differing_coordinates'], 1)
        self.assertEqual(report['difference_norm'], 1.)

    def test_continuous_difference_can_disappear_after_bf16_rounding(self):
        origin, native, left, right = fixture()
        continuous = compare_sketches(left, right, origin, native)
        self.assertGreater(continuous['difference_norm'], 0.)
        a, b = [(torch.nn.Parameter(origin[0].clone()),) for _ in range(2)]
        geometry.write_candidate(a, origin, native, left)
        geometry.write_candidate(b, origin, native, right)
        self.assertTrue(compare_endpoints(a, b, origin, native)['identical'])

    def test_empty_zero_native_and_inactive_proposals(self):
        origin = (torch.ones(3, dtype=torch.bfloat16),)
        empty = geometry.make_sketch([], [], origin, origin)
        result = compare_sketches(empty, empty, origin, origin)
        self.assertTrue(result['identical'])
        self.assertIsNone(result['relative_difference_to_native_norm'])
        self.assertIsNone(result['direction_cosine'])
        empty_result = compare_endpoints((), (), (), ())
        self.assertEqual(empty_result['coordinates'], 0)
        self.assertTrue(empty_result['identical'])

    def test_nonfinite_statistics_stop(self):
        old = (torch.zeros(2, dtype=torch.bfloat16),)
        bad = (torch.tensor([float('nan'), 0.], dtype=torch.bfloat16),)
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            compare_endpoints(bad, old, old, old)

    def test_prefix_repairs_respect_first_error_eos_seed_and_padding(self):
        labels = torch.tensor([[[0, 1, 2, 9, 4], [9, 1, 2, 3, 4], [0, 1, 2, 3, 4]]])
        validity = torch.ones_like(labels).float(); validity[0, 2, 3] = 0
        before = labels.clone(); before[0, 0, 2] = 8
        native = labels.clone()
        left = native.clone(); left[0, 0, 1] = 8
        right = native.clone(); right[..., 0] = 42; right[0, 0, 4] = 8; right[0, 2, 4] = 8
        report = prefix_consequences(before, native, left, right, labels, validity, (9,))
        self.assertEqual(report['valid_proposal_occurrences'], 5)
        self.assertEqual(report['differing_valid_proposals'], 1)
        self.assertEqual(report['left_from_pre']['old_prefix_tokens_lost'], 1)
        self.assertEqual(report['right_from_pre']['net_matching_prefix_change'], 2)
        self.assertEqual(report['left_repair_identities_vs_native']['useful_repairs_lost'], 1)
        self.assertEqual(report['right_repair_identities_vs_native']['useful_repairs_retained'], 1)


if __name__ == '__main__':
    unittest.main()
