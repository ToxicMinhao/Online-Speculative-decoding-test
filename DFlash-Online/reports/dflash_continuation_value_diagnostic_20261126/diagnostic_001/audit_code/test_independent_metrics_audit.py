"""Small synthetic fixtures for the independent metrics checker."""
import math
import unittest

import numpy as np

from independent_metrics_audit import Audit, audit_compact, expected_metrics, roles


def block(matches, supports=None):
    labels = list(range(100, 116))
    proposals = labels.copy()
    for index, correct in enumerate(matches, 1):
        if not correct:
            proposals[index] = labels[index] - 1
    mask = [False] + [True] * len(matches) + [False] * (15 - len(matches))
    result = dict(labels=labels, valid=mask, proposals=proposals, m=len(matches),
                  support_source='synthetic_FP32')
    result['roles'] = roles(result)
    positions = []
    for index, correct in enumerate(matches, 1):
        support = (supports or {}).get(index, .5 if correct else 0.)
        margin = float(np.float32(2 * math.atanh(support))) if correct else -.5
        native = float(np.float32(max(0., math.tanh(margin / 2))))
        positions.append(dict(position=index, canonical=labels[index], prediction=proposals[index],
                              rival=labels[index]+1 if correct else proposals[index],
                              margin=margin, margin_tie=margin == 0., wrong_rival_tie=False,
                              support_fp32=native, correct=correct))
    result['positions'] = positions
    return result


class MetricsAuditTests(unittest.TestCase):
    def test_area_normalization_and_later_bottleneck(self):
        parent = block([True, True, False, True, True, True, True, False],
                       {4: .9, 5: .2, 6: .9, 7: .9})
        result = expected_metrics(parent, parent)
        self.assertEqual((result['e'], result['r'], result['K'], result['n']), (3, 4, 7, 5))
        self.assertAlmostEqual(result['A'], 1.5, places=6)
        self.assertAlmostEqual(result['u'], 1.5 / 14, places=7)
        self.assertEqual(result['run_bottleneck_indices'], [0, 1, 1, 1])
        self.assertEqual(result['run_immediate_controlled_terms'], 1)
        self.assertEqual(result['Y'], 0)

    def test_raw_repair_is_not_useful_after_earlier_damage(self):
        parent = block([True, True, False, True, True, True, True, False])
        child = block([False, True, True, True, True, True, True, False])
        result = expected_metrics(parent, child)
        self.assertTrue(result['raw_repair'])
        self.assertFalse(result['useful_repair'])
        self.assertEqual(result['Y'], -2)
        self.assertEqual(result['lost_prefix'], 2)
        self.assertEqual(result['damaged_prefix_positions'], 1)
        self.assertEqual(result['surviving_origin_followers'], 4)
        self.assertEqual(result['reachable_origin_followers'], 0)

    def test_progress_capped_at_frozen_parent_K(self):
        parent = block([True, False, True, False, True, True])
        child = block([True] * 6)
        result = expected_metrics(parent, child)
        self.assertEqual(result['Y'], 2)
        self.assertEqual(result['uncapped_progress'], 5)
        self.assertEqual(result['reachable_origin_followers'], 1)

    def test_native_ties_can_be_correct_or_incorrect(self):
        current = block([True, False])
        for position in current['positions']:
            position.update(margin=0., support_fp32=0., margin_tie=True)
        audit = Audit()
        audit_compact(current, audit, 'ties')
        self.assertEqual(audit.failures, [])

    def test_gaps_rejected_and_empty_block_has_no_error(self):
        empty = block([])
        self.assertEqual(roles(empty)['e'], -1)
        self.assertEqual(expected_metrics(empty, empty)['A'], 0)
        invalid = block([True, True, True])
        invalid['valid'][2] = False
        with self.assertRaises(ValueError):
            roles(invalid)

    def test_large_support_discrepancy_is_not_silent(self):
        current = block([True])
        current['positions'][0]['support_fp32'] = float(np.float32(.1))
        audit = Audit()
        audit_compact(current, audit, 'wrong-support')
        self.assertTrue(any('ULPs' in row['label'] for row in audit.failures))


if __name__ == '__main__':
    unittest.main()
