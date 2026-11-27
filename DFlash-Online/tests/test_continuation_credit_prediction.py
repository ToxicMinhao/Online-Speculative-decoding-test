"""CPU checks for prespecified grouped prediction and fixed-OOF resampling."""

import copy
import json
import unittest

import numpy as np

from experiments.continuation_credit_cpu.prediction import (
    BOOTSTRAP_DRAWS, PENALTY, SEED, _bootstrap_comparison, _features,
    _logistic_objective, _sample_weights, run_prediction,
)


def fixture():
    rows = []
    for response in range(12):
        for block in range(2):
            for follower in range(1 + block):
                severity = (response + 2 * block + follower) % 5
                rows.append({
                    "response_ordinal": response, "record_id": f"private-{response:02d}",
                    "anchor": 20 * block, "position": 6 + follower,
                    "target": bool((response + block + follower) % 3 == 0),
                    "e": 1 + response % 3, "r": 1 + block, "n": 12 - response % 3,
                    "distance": 5 + follower, "error_count": 2 + block,
                    "u_old": (1 + block) * 0.6 / 14, "w": 0.3 + 0.1 * follower,
                    "first_error_deficit": 0.5 + response % 2,
                    "prefix_margin": 0.0 if response % 3 == 0 else 1.5,
                    "empty_prefix": response % 3 == 0,
                    "later_mean_deficit": severity / 2, "later_max_deficit": severity,
                })
    return rows


class PredictionTests(unittest.TestCase):
    def test_hierarchical_weights_preserve_response_and_block_mass(self):
        rows = fixture()
        weights = _sample_weights(rows)
        self.assertAlmostEqual(weights.sum(), 1)
        for response in range(12):
            response_rows = np.array([row['response_ordinal'] == response for row in rows])
            self.assertAlmostEqual(weights[response_rows].sum(), 1 / 12)
            for anchor in (0, 20):
                block = np.array([row['response_ordinal'] == response and row['anchor'] == anchor for row in rows])
                self.assertAlmostEqual(weights[block].sum(), 1 / 24)
                self.assertTrue(np.equal(weights[block], weights[block][0]).all())

    def test_logistic_gradient_and_unpenalized_intercept(self):
        design = np.array([[0.5, -1], [2, 0.3], [-0.5, 1.5]])
        target = np.array([0.0, 1.0, 1.0])
        weights = np.array([0.25, 0.25, 0.5])
        parameters = np.array([0.3, -0.8, 0.4])
        value, gradient = _logistic_objective(parameters, design, target, weights)
        logits = parameters[0] + design @ parameters[1:]
        probability = 1 / (1 + np.exp(-logits))
        expected = np.dot(weights, -target * np.log(probability) - (1 - target) * np.log1p(-probability))
        expected += PENALTY * np.dot(parameters[1:], parameters[1:]) / 2
        self.assertAlmostEqual(value, expected, places=14)
        self.assertAlmostEqual(gradient[0], np.dot(weights, probability - target), places=14)
        for index in range(parameters.size):
            shift = np.zeros_like(parameters)
            shift[index] = 1e-6
            finite_difference = (_logistic_objective(parameters + shift, design, target, weights)[0]
                                 - _logistic_objective(parameters - shift, design, target, weights)[0]) / 2e-6
            self.assertAlmostEqual(gradient[index], finite_difference, places=8)
        extreme, extreme_gradient = _logistic_objective(np.array([1000.0, 1000.0, -1000.0]), design, target, weights)
        self.assertTrue(np.isfinite(extreme))
        self.assertTrue(np.isfinite(extreme_gradient).all())

    def test_folds_predictions_and_bootstrap_are_deterministic_and_public(self):
        rows = fixture()
        result = run_prediction(rows)
        reversed_result = run_prediction(list(reversed(rows)))
        self.assertEqual(result, reversed_result)
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(len(result['models']), 12)
        expected_order = np.random.Generator(np.random.PCG64(SEED)).permutation(sorted({r['record_id'] for r in rows}))
        fold_lookup = {int(record_id.rsplit('-', 1)[1]): index % 4 for index, record_id in enumerate(expected_order)}
        for row in result['predictions']:
            self.assertEqual(row['fold'], fold_lookup[row['response_ordinal']])
            self.assertNotIn('record_id', row)
            for name in ('CONST', 'STRUCT', 'SEVERITY'):
                self.assertLessEqual(0, row[f'rho_{name}'])
                self.assertLessEqual(row[f'rho_{name}'], 1)
        self.assertNotIn('private-', json.dumps(result))
        for comparison in result['bootstrap'].values():
            self.assertEqual(len(comparison['log_loss']['draws']), BOOTSTRAP_DRAWS)
            self.assertEqual(comparison['log_loss']['valid_draws'], BOOTSTRAP_DRAWS)
            self.assertEqual(comparison['log_loss']['unidentifiable_draw_indices'], [])

    def test_standardization_and_fit_do_not_see_held_out_features(self):
        rows = fixture()
        original = run_prediction(rows)
        fold = original['folds'][0]
        modified = copy.deepcopy(rows)
        held_out = set(fold['test_response_ordinals'])
        for row in modified:
            if row['response_ordinal'] in held_out:
                row['first_error_deficit'] = 1e9
                row['later_mean_deficit'] = 1e9
                row['later_max_deficit'] = 1e9
        changed = run_prediction(modified)
        original_models = [model for model in original['models'] if model['fold'] == 0]
        changed_models = [model for model in changed['models'] if model['fold'] == 0]
        self.assertEqual(original_models, changed_models)
        training_rows = [row for row in rows if row['response_ordinal'] not in held_out]
        weights = _sample_weights(training_rows)
        for model in original_models:
            if model['model'] == 'CONST':
                expected = weights @ np.array([row['target'] for row in training_rows])
                self.assertAlmostEqual(model['probability'], expected)
            else:
                design = _features(training_rows, severity=model['model'] == 'SEVERITY')
                expected_mean = weights @ design
                np.testing.assert_allclose(model['mean'], expected_mean, rtol=1e-14, atol=1e-14)

    def test_whole_response_resampling_uses_response_means_and_keeps_bad_draws(self):
        rows = fixture()
        weights = _sample_weights(rows)
        differences = np.array([row['response_ordinal'] + 0.5 * (row['anchor'] == 20) for row in rows])
        draws = np.array([[0] * 12, [11] * 12, list(range(12))])
        result = _bootstrap_comparison(differences, rows, weights, draws)
        np.testing.assert_allclose(result['draws'], [0.25, 11.25, 5.75])
        differences[0] = np.nan
        failed = _bootstrap_comparison(differences, rows, weights, draws)
        self.assertEqual(failed['draws'], [None, 11.25, None])
        self.assertEqual(failed['unidentifiable_draw_indices'], [0, 2])
        self.assertEqual(failed['valid_draws'], 1)

    def test_single_class_folds_remain_unavailable_without_changing_membership(self):
        rows = fixture()
        ordinary = run_prediction(rows)
        for row in rows:
            row['target'] = False
        result = run_prediction(rows)
        self.assertEqual(result['folds'], ordinary['folds'])
        self.assertEqual(result['status'], 'incomplete_folds')
        for model in result['models']:
            if model['model'] == 'CONST':
                self.assertEqual(model['probability'], 0)
            else:
                self.assertEqual(model['status'], 'unavailable')
                self.assertIn('one target class', model['reason'])
        for row in result['predictions']:
            self.assertIsNone(row['rho_STRUCT'])
            self.assertIsNone(row['rho_SEVERITY'])
        for comparison in result['bootstrap'].values():
            self.assertEqual(comparison['log_loss']['valid_draws'], 0)
            self.assertEqual(len(comparison['log_loss']['unidentifiable_draw_indices']), BOOTSTRAP_DRAWS)

    def test_empty_and_duplicate_rows_have_explicit_outcomes(self):
        empty = run_prediction([])
        self.assertEqual(empty['status'], 'unavailable')
        self.assertEqual(empty['predictions'], [])
        rows = fixture()
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            run_prediction(rows + rows[:1])


if __name__ == '__main__':
    unittest.main()
