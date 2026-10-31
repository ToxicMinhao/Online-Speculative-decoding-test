import csv
import copy

import numpy as np
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from experiments.k_prefix_protection_v1 import statistics as stats


def measurements(calls):
    return {'verification_calls': calls, 'accepted_draft_tokens': 60,
            'committed_progress': 100, 'output_tokens': 100,
            'request_latency_s': calls * 0.2, 'native_generate_s': calls * 0.19,
            'cache_maintenance_s': calls * 0.01}


def paired_rows(count=4):
    return [{'seed': seed, 'request_id': f'id_{i}', 'panel_ordinal': i,
             **{method: measurements(calls + i) for method, calls in
                [('F', 20), ('K', 19), ('K_HALF', 18), ('KP', 17)]}}
            for seed in (17, 29) for i in range(count)]












def comparison(point, low, high):
    return {'call_reduction': {'point_pct': point, 'ci95_pct': [low, high]}}


def interpretation_inputs():
    combined = {'KP_vs_K': comparison(0.7, 0.1, 1.2),
                'KP_vs_F': comparison(1.0, 0.2, 1.8),
                'KP_vs_K_HALF': comparison(0.1, -0.2, 0.4)}
    per_seed = {str(seed): copy.deepcopy(combined) for seed in (17, 29)}
    return combined, per_seed

class TestStatistics(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)

    def test_bootstrap_matches_explicit_pcg64_and_joint_unit(self):
        expected = np.random.Generator(np.random.PCG64(20261102)).integers(0, 4, size=(10000, 4))
        np.testing.assert_array_equal(stats.bootstrap_indices(4), expected)
        rows = paired_rows()
        for row in rows:
            row['KP']['verification_calls'] = 10 if row['seed'] == 17 else 30
            row['K']['verification_calls'] = 20
        summary = stats.summarize(rows)
        # Both fitted-seed effects cancel within every request, hence every joint draw.
        assert summary['combined']['KP_vs_K']['call_reduction']['ci95_pct'] == [0, 0]
        assert summary['combined']['KP_vs_K']['request_seed_wins'] == 4
        assert summary['combined']['KP_vs_K']['request_seed_regressions'] == 4
        assert summary['interpretation']['endpoint_label'] == 'mixed fitted-seed directions'

    def test_ratio_of_sums_not_average_request_reductions(self):
        result = stats.ratio_reduction([1, 99], [2, 100], np.array([[0, 1], [0, 1]]))
        self.assertAlmostEqual(result['point_pct'], 100 * (1 - 100 / 102))
        self.assertNotAlmostEqual(result['point_pct'], (50 + 1) / 2)

    def test_gain_harm_positive_parts_before_seed_pooling_and_identity(self):
        frozen = np.array([[10, 10], [10, 10]])
        incumbent = np.array([[5, 15], [15, 5]])
        candidate = np.array([[6, 11], [13, 4]])
        result = stats.gain_harm(frozen, incumbent, candidate, stats.bootstrap_indices(2))
        assert result['K_gross_gain']['point'] == 10
        assert result['K_gross_harm']['point'] == 10
        assert result['retained_gain']['point'] == 9
        assert result['lost_gain']['point'] == 1
        assert result['added_gain']['point'] == 1
        assert result['prevented_harm']['point'] == 6
        assert result['added_harm']['point'] == 0
        assert result['net_calls_saved_vs_K'] == 6
        assert result['retained_gain_fraction']['point'] == 0.9
        assert result['identity_verified_at_each_seed_request']

    def test_zero_incumbent_gain_is_undefined_not_zero_retention(self):
        same = np.array([[10, 10], [10, 10]])
        result = stats.gain_harm(same, same, same - 1, stats.bootstrap_indices(2))
        assert result['retained_gain_fraction']['point'] is None
        assert result['retained_gain_fraction']['ci95'] is None
        assert result['retained_gain_fraction']['undefined_bootstrap_draws'] == 10000

    def test_statistics_reject_nonpaired_duplicate_and_nonfinite_rows(self):
        rows = paired_rows()
        with self.assertRaisesRegex(ValueError, 'identities'):
            stats.summarize(rows[:-1])
        with self.assertRaisesRegex(ValueError, 'duplicated'):
            stats.summarize(rows + [rows[0]])
        rows[0]['KP']['request_latency_s'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            stats.summarize(rows)

    def test_interpretation_requires_all_primary_and_frozen_criteria(self):
        combined, per_seed = interpretation_inputs()
        decision = stats.interpret(combined, per_seed)
        assert decision['supported_useful_improvement']
        assert decision['KP_K_HALF_practical_equivalence_supported']
        assert not decision['superiority_over_tested_K_HALF_supported']
        combined['KP_vs_K']['call_reduction']['ci95_pct'][0] = -0.1
        assert stats.interpret(combined, per_seed)['endpoint_label'] == 'promising but unresolved'
        combined, per_seed = interpretation_inputs()
        per_seed['29']['KP_vs_F']['call_reduction']['point_pct'] = -0.1
        assert not stats.interpret(combined, per_seed)['supported_useful_improvement']

    def test_nonsignificant_secondary_is_not_equivalence(self):
        combined, per_seed = interpretation_inputs()
        combined['KP_vs_K_HALF'] = comparison(0.1, -2, 2)
        decision = stats.interpret(combined, per_seed)
        assert not decision['superiority_over_tested_K_HALF_supported']
        assert not decision['KP_K_HALF_practical_equivalence_supported']

    def test_precision_note_reproduces_original_seed_and_pairing(self):
        tmp_path = self.directory
        source = tmp_path / 'paired.csv'
        fields = ['seed', 'event', 'request_id', 'panel_ordinal', 'F_verification_calls',
                  'O_verification_calls', 'K_verification_calls']
        with source.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for event in (2, 4):
                for seed in (17, 29):
                    for i in range(128):
                        writer.writerow(dict(zip(fields, [seed, event, f'id_{i}', i, 100, 110, 105])))
        note = stats.write_precision_note(source, tmp_path)
        assert note['bootstrap']['seed'] == 20261030
        result = note['combined_historical_intervals']['4']['K_vs_O']
        self.assertAlmostEqual(result['point_pct'], 100 * (1 - 105 / 110))
        np.testing.assert_allclose(result['ci95_pct'], [result['point_pct']] * 2)
        assert (tmp_path / 'PRECISION_NOTE.md').exists()
        assert note['priority_target_pct'] == 0.5


if __name__ == "__main__":
    unittest.main()
