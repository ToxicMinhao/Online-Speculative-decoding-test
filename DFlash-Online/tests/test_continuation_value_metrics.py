"""CPU contracts for the fixed continuation score and frozen-parent outcomes."""
from copy import deepcopy
import math
import unittest
import torch
from experiments.continuation_value_diagnostic.metrics import compact_blocks, continuation_area, coverage_table, legacy_pairs, paired_block, parent_predictors, roles_from_predictions

def block(matches, margins=None):
    """Synthetic compact evidence with the same shape as one native readout."""
    labels = [1] * 16
    predictions = [3] + [1 if match else 2 for match in matches] + [0] * (15 - len(matches))
    valid = [False] + [True] * len(matches) + [False] * (15 - len(matches))
    margins = margins if margins is not None else [2.0 if match else -1.0 for match in matches]
    supports = torch.tanh(torch.tensor(margins, dtype=torch.float32) / 2).clamp_min(0)
    positions = [dict(position=j, canonical=1, prediction=predictions[j], rival=2, correct=predictions[j] == 1, margin=float(margin), support_fp32=float(support), margin_tie=margin == 0, wrong_rival_tie=False) for j, (margin, support) in enumerate(zip(margins, supports), 1)]
    roles = roles_from_predictions(labels, valid, predictions)
    return dict(labels=labels, valid=valid, proposals=predictions, positions=positions, roles=roles, m=len(matches))

def pair(parent, child, anchor=11, record_id='response-A'):
    return paired_block(parent, child, record_id=record_id, anchor=anchor)

def old_block(origin, current):
    row = pair(origin, current)
    return dict(origin=origin['roles'], current=current['roles'], proposals=current['proposals'], positions=[dict(p) for p in current['positions'] if p['position'] <= row['K']], metrics=dict(raw_repair=int(row['raw_repair']), useful_repair=int(row['useful_repair']), damaged_prefix_positions=row['damaged_prefix_positions'], damaged_prefix_blocks=int(row['prefix_damaged']), surviving_followers=row['surviving_origin_followers'], reachable_followers=row['reachable_origin_followers'], matching_prefix_length=row['child_L'], fixed_K_shortfall=max(0, row['K'] - row['child_L'])))

class ContinuationValueMetricsTests(unittest.TestCase):

    def test_external_parent_records_cannot_hide_nonfinite_or_duplicate_positions(self):
        parent = block([True, False, True])
        parent['positions'][0]['margin'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            parent_predictors(parent)
        parent = block([True, False, True])
        parent['positions'].append(dict(parent['positions'][0]))
        with self.assertRaisesRegex(ValueError, 'duplicated'):
            parent_predictors(parent)

    def test_coverage_never_duplicates_response_or_anchor_units(self):
        parent = block([False, True])
        row = pair(parent, parent)
        with self.assertRaisesRegex(ValueError, 'double-count'):
            coverage_table([row, row], ['response-A'])
        with self.assertRaisesRegex(ValueError, 'unique'):
            coverage_table([row], ['response-A', 'response-A'])

    def test_prespecified_area_examples(self):
        for supports, area in [([0.9, 0.9, 0.9, 0.9], 3.6), ([0.9, 0.2, 0.9, 0.9], 1.5), ([0.8, 0.8], 1.6), ([0.9, 0, 0.9], 0.9), ([], 0)]:
            with self.subTest(supports=supports, area=area):
                result = continuation_area(supports)
                self.assertAlmostEqual(result['A'], area)
                self.assertAlmostEqual(result['u'], area / 14)

    def test_area_earliest_tie_and_deterministic_bottleneck(self):
        result = continuation_area([0.9, 0.9, 0.2, 0.2, 0.7])
        assert result['bottleneck_indices'] == [0, 0, 2, 2, 2]
        assert result['tied_terms'] == [False, True, False, True, True]
        with self.assertRaisesRegex(ValueError, 'finite'):
            continuation_area([float('nan')])

    def test_native_seed_eos_padding_and_gap_boundaries(self):
        labels = torch.ones((1, 4, 16), dtype=torch.long)
        weights = torch.ones_like(labels, dtype=torch.float32)
        labels[0, 0, 0] = 7
        labels[0, 1, 3] = 7
        weights[0, 2, 4] = 0
        weights[0, 3, 10:] = 0
        logits = torch.zeros((1, 4, 16, 8), dtype=torch.bfloat16)
        logits.scatter_(-1, labels.unsqueeze(-1), 3)
        logits[:, :, 0, 4] = 5
        result = compact_blocks(logits, labels, weights, eos_token_ids=[7])
        assert [value['m'] for value in result] == [0, 3, 3, 9]
        assert [value['roles']['L'] for value in result] == [0, 3, 3, 9]
        assert all((not value['valid'][0] for value in result))
        assert result[1]['positions'][-1]['canonical'] == 7

    def test_native_argmax_ties_and_strongest_noncanonical_rival(self):
        labels = torch.ones((1, 1, 16), dtype=torch.long)
        weights = torch.zeros_like(labels, dtype=torch.float32)
        weights[..., 1:5] = 1
        logits = torch.full((1, 1, 16, 5), -2, dtype=torch.bfloat16)
        logits[0, 0, 1, 1:3] = 4
        logits[0, 0, 2, :2] = 4
        logits[0, 0, 3] = torch.tensor([-1, 5, 4, 4, 1])
        logits[0, 0, 4] = torch.tensor([-1, 1, 2, 4, 3])
        result = compact_blocks(logits, labels, weights)[0]
        a, b, c, d = result['positions']
        assert (a['prediction'], a['rival'], a['margin'], a['correct']) == (1, 2, 0, True)
        assert (b['prediction'], b['rival'], b['margin'], b['correct']) == (0, 0, 0, False)
        assert (c['prediction'], c['rival'], c['margin'], c['wrong_rival_tie']) == (1, 2, 1, True)
        assert (d['prediction'], d['rival'], d['margin']) == (3, 3, -3)
        assert a['support_fp32'] == b['support_fp32'] == d['support_fp32'] == 0
        assert result['roles']['e'] == 2
        assert result['roles']['K'] == 3

    def test_native_shapes_precision_and_nonfinite_reject(self):
        logits = torch.zeros((1, 1, 16, 4), dtype=torch.bfloat16)
        labels = torch.zeros((1, 1, 16), dtype=torch.long)
        weights = torch.ones_like(labels, dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, 'BF16'):
            compact_blocks(logits.float(), labels, weights)
        logits[0, 0, 1, 0] = torch.nan
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            compact_blocks(logits, labels, weights)

    def test_parent_roles_area_deficit_and_full_suffix_zero_tail(self):
        parent = block([True, False, True, True, True, False, True], [2, -1, 4, 1, 3, -2, 4])
        row = parent_predictors(parent)
        assert (row['L'], row['e'], row['r'], row['n'], row['K']) == (1, 2, 3, 5, 5)
        assert row['high'] and row['s']
        assert row['beta'] == math.exp(-1 / 7)
        assert row['running_minima_fp64'][-2:] == [0, 0]
        assert row['A'] == math.fsum(row['running_minima_fp64'])
        assert row['D'] > 0
        self.assertAlmostEqual(row['D'], row['r'] * row['h1'] / 14 - row['u'])
        assert row['run_immediate_controlled_terms'] == 1
        assert row['run_later_controlled_terms'] == 2
        assert row['run_immediate_controlled_fraction'] == 1 / 3

    def test_high_correct_tie_remains_eligible_and_single_run_has_zero_deficit(self):
        parent = block([False, True, False, True], [0, 0, -2, 3])
        row = parent_predictors(parent)
        assert row['high'] and row['r'] == 1
        assert row['empty_prefix'] and row['error_tie']
        assert row['minimum_original_prefix_margin'] == 0
        assert row['h1'] == row['u'] == row['D'] == 0
        assert row['n'] == 3 and row['K'] == 2

    def test_no_error_low_last_error_and_empty_population_stay_visible(self):
        rows = [pair(b, b, anchor=i) for i, b in enumerate([block([True, True]), block([False, False]), block([True, False]), block([])])]
        coverage = coverage_table(rows, ['response-A', 'empty-response'])
        assert coverage[0]['no_error_blocks'] == 1
        assert coverage[0]['low_blocks'] == 2
        assert coverage[0]['empty_blocks'] == 1
        assert coverage[0]['high_blocks'] == 0
        assert coverage[1]['blocks'] == 0
        assert rows[2]['r'] == rows[2]['n'] == rows[2]['A'] == rows[2]['D'] == 0
        assert rows[0]['e'] == -1 and rows[0]['K'] == rows[0]['L'] == 2

    def test_frozen_outcomes(self):
        for matches, signed, raw, lost, reachable, uncapped in [([False, True, True, True, True, True], -2, True, 2, 0, -2), ([True, True, True, False, True, False], 1, True, 0, 0, 1), ([True, True, True, True, False, False], 2, True, 0, 1, 2), ([True, True, True, True, True, False], 3, True, 0, 2, 3), ([True, True, False, True, True, False], 0, False, 0, 0, 0), ([True, True, True, True, True, True], 3, True, 0, 2, 4)]:
            with self.subTest(matches=matches, signed=signed, raw=raw, lost=lost, reachable=reachable, uncapped=uncapped):
                parent = block([True, True, False, True, True, False])
                row = pair(parent, block(matches))
                assert row['K'] == 5
                assert row['Y'] == signed and row['raw_repair'] == raw
                assert row['useful_repair'] == (signed > 0)
                assert row['prefix_damaged'] == (signed < 0)
                assert row['lost_prefix'] == lost
                assert row['reachable_origin_followers'] == reachable
                assert row['uncapped_progress'] == uncapped
                assert -row['L'] <= signed <= row['r'] + 1

    def test_position_damage_is_distinct_from_lost_consecutive_prefix(self):
        row = pair(block([True, True, True, False, True]), block([False, True, True, True, True]))
        assert row['damaged_prefix_positions'] == 1
        assert row['lost_prefix'] == 3
        assert row['raw_repair'] and (not row['useful_repair'])
        assert row['surviving_origin_followers'] == 1

    def test_child_does_not_redefine_parent_roles_and_canonical_binding(self):
        parent = block([True, False, True, False])
        child = block([True, True, True, True])
        row = pair(parent, child)
        assert row['K'] == 3 and row['e'] == 2 and (row['r'] == 1)
        assert row['child_L'] == 4 and row['uncapped_progress'] == 3 and (row['Y'] == 2)
        child['labels'][1] = 3
        with self.assertRaisesRegex(ValueError, 'canonical'):
            pair(parent, child)

    def test_legacy_offline_area_and_integer_parity_without_full_logits(self):
        parent = block([True, False, True, True, False, True])
        child = block([True, True, True, True, True, True])
        old_parent = dict(sets={'new_buffer': [dict(record_id='a', anchors=[9], blocks=[old_block(parent, parent)])]})
        old_child = dict(sets={'new_buffer': [dict(record_id='a', anchors=[9], blocks=[old_block(parent, child)])]})
        row = legacy_pairs(old_parent, old_child, 'new_buffer')[0]
        fresh = pair(parent, child)
        for field in ('Y', 'A', 'u', 'D', 'r', 'n', 'K', 'surviving_origin_followers'):
            assert row[field] == fresh[field]
        assert row['inferred_zero_area_terms'] == 2
        assert row['follower_supports_fp32'][-2:] == [None, None]
        assert row['running_minima_fp64'][-2:] == [0, 0]
        assert row['cohort'] == 'legacy_new_buffer'
        broken = deepcopy(old_child)
        broken['sets']['new_buffer'][0]['blocks'][0]['metrics']['useful_repair'] = 0
        with self.assertRaisesRegex(ValueError, 'integer outcome'):
            legacy_pairs(old_parent, broken, 'new_buffer')

    def test_coverage_keeps_high_pool_requirements_separate(self):
        strong = block([False, True], [-1, 3])
        weak = block([False, True], [-1, 0.1])
        low = block([False, False])
        rows = [pair(b, b, anchor=i) for i, b in enumerate((strong, weak, low))]
        result = coverage_table(rows, ['response-A'])[0]
        assert result['mixed_high_low_pool']
        assert result['at_least_two_high']
        assert result['high_score_variation']

    def test_empty_prefix_native_margin_and_beta_are_parent_only(self):
        origin = block([False, True, True], [-2.5, 0.5, 1])
        row = pair(origin, block([True, True, True]))
        assert row['beta'] == 1
        assert row['d'] == 2.5 and row['error_margin'] == -2.5
        assert row['minimum_original_prefix_margin'] == 0 and row['empty_prefix']
        assert row['D'] == 0

    def test_noncontiguous_external_validity_rejected(self):
        raw = block([True, False])
        raw['valid'][4] = True
        with self.assertRaisesRegex(ValueError, 'contiguous'):
            parent_predictors(raw)
if __name__ == '__main__':
    unittest.main()
