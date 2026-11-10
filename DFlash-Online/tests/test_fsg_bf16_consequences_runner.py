"""CPU identity, reproduction and work contracts for the BF16 consequence test."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from experiments.functional_step_geometry import diagnose_bf16_consequences as diagnostic
from experiments.functional_step_geometry.diagnose_gradients import DiagnosticWork
from experiments.ok_continuation.resume import differences
from online_feedback.rng import capture_rng


class WindowIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.feature_paths = []
        self.records = []
        self.raw = []
        for index, positions in enumerate(([8, 2, 6, 3], [7, 3, 5])):
            record_id = f'retained-{index}'
            raw = dict(record_id=record_id, prompt_length=2,
                input_ids=(torch.arange(14) + index * 20).reshape(1, 14),
                loss_mask=torch.tensor([[0., 0.] + [1.] * 12]),
                hidden_states=torch.ones(1, 14, 3) * (index + 1))
            path = self.root / f'feature_{index}.pt'
            torch.save(raw, path)
            self.raw.append(raw)
            self.feature_paths.append(path)
            self.records.append(dict(record_id=record_id, sampled_anchors=[dict(
                positions=[list(positions)], keep_mask=[[True] * len(positions)])]))
        self.presentations_path = self.root / 'PRESENTATIONS.json'
        self.write_records()

        def labels_and_weights(ids, mask, anchors, keep):
            labels = torch.zeros(1, anchors.shape[1], 16, dtype=torch.int64)
            weights = torch.zeros_like(labels, dtype=torch.float32)
            return labels, weights, None

        self.model = SimpleNamespace(
            _block_labels_and_weights=mock.Mock(side_effect=labels_and_weights),
            _sample_anchor_positions=mock.Mock(side_effect=AssertionError('Sampling is forbidden')))
        self.trainer = SimpleNamespace(max_length=2272,
            bundle=SimpleNamespace(model=self.model),
            _batch=mock.Mock(side_effect=lambda raw: SimpleNamespace(tensors={
                key: raw[key] for key in ('input_ids', 'loss_mask', 'hidden_states')})))

    def write_records(self):
        self.presentations_path.write_text(json.dumps(self.records))

    def test_binds_both_original_responses_and_every_original_anchor(self):
        before_rng = capture_rng()
        responses, metadata = diagnostic.load_window(
            self.trainer, self.feature_paths, self.presentations_path)
        self.assertEqual(len(responses), 2)
        for index, response in enumerate(responses):
            self.assertEqual(response['record_id'], self.records[index]['record_id'])
            self.assertEqual(response['anchors'].tolist(),
                             self.records[index]['sampled_anchors'][0]['positions'])
            self.assertEqual(response['keep'].tolist(),
                             self.records[index]['sampled_anchors'][0]['keep_mask'])
            self.assertEqual(response['anchors'].dtype, torch.int64)
            self.assertTrue(torch.equal(response['input_ids'], self.raw[index]['input_ids']))
            self.assertTrue(torch.equal(response['hidden_states'], self.raw[index]['hidden_states']))
        self.model._sample_anchor_positions.assert_not_called()
        self.assertFalse(differences(before_rng, capture_rng()))

    def test_swapped_features_cannot_silently_rebind_to_other_presentations(self):
        with self.assertRaises(ValueError):
            diagnostic.load_window(self.trainer, list(reversed(self.feature_paths)),
                                   self.presentations_path)
        self.model._sample_anchor_positions.assert_not_called()

    def test_missing_response_or_multiple_draws_stop_without_resampling(self):
        with self.assertRaises(ValueError):
            diagnostic.load_window(self.trainer, self.feature_paths[:1], self.presentations_path)
        self.records[0]['sampled_anchors'].append(deepcopy(self.records[0]['sampled_anchors'][0]))
        self.write_records()
        with self.assertRaises(ValueError):
            diagnostic.load_window(self.trainer, self.feature_paths, self.presentations_path)
        self.model._sample_anchor_positions.assert_not_called()

    def test_original_keep_mask_is_preserved_without_filtering_anchors(self):
        self.records[0]['sampled_anchors'][0]['keep_mask'][0][1] = False
        self.write_records()
        responses, metadata = diagnostic.load_window(
            self.trainer, self.feature_paths, self.presentations_path)
        self.assertEqual(responses[0]['anchors'].tolist(), [[8, 2, 6, 3]])
        self.assertEqual(responses[0]['keep'].tolist(), [[True, False, True, True]])
        self.model._sample_anchor_positions.assert_not_called()


class ReproductionAndWorkTests(unittest.TestCase):
    @staticmethod
    def retained_step():
        return dict(global_step=513, arm='FSG', strength=1.,
            sketch=dict(sigma=1., gram=[[2., .25], [.25, 3.]], rhs=[.1, .2]),
            native_risk=dict(value=.072, eta=1e-9),
            raw_fsg=dict(norm=1.4584, native_norm=1.4583, continuous_norm=1.4559,
                         changed_coordinates=125, identical_native=False),
            decision=dict(accepted=False, reason='rounded_norm_exceeds_native'),
            native=dict(norm=1.4583, native_norm=1.4583),
            responses=[dict(anchors=4, protected_occurrences=2),
                       dict(anchors=3, protected_occurrences=1)],
            status='failed', error='historical oracle failure', hook_total_s=29.)

    def test_exact_scientific_receipt_ignores_historical_runtime_failure(self):
        retained = self.retained_step()
        observed = deepcopy(retained)
        observed.update(status='completed', error=None, hook_total_s=7.)
        receipt = diagnostic.reproduction_receipt(observed, retained)
        self.assertEqual(receipt['status'], 'passed')
        self.assertEqual(receipt['differences'], [])
        with tempfile.TemporaryDirectory() as directory:
            diagnostic.require_reproduction(observed, retained, Path(directory))
            saved = json.loads((Path(directory) / 'NATIVE_REPRODUCTION.json').read_text())
            self.assertEqual(saved['status'], 'passed')

    def test_smallest_float_discrepancy_is_saved_and_still_stops(self):
        retained = self.retained_step()
        observed = deepcopy(retained)
        observed['raw_fsg']['norm'] = float.fromhex(retained['raw_fsg']['norm'].hex())
        observed['raw_fsg']['norm'] += 2 ** -51
        self.assertNotEqual(observed['raw_fsg']['norm'], retained['raw_fsg']['norm'])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                diagnostic.require_reproduction(observed, retained, Path(directory))
            saved = json.loads((Path(directory) / 'NATIVE_REPRODUCTION.json').read_text())
            self.assertEqual(saved['status'], 'failed')
            self.assertTrue(saved['differences'])
            self.assertNotEqual(saved['observed']['raw_fsg']['norm'],
                                saved['expected']['raw_fsg']['norm'])

    def test_native_endpoint_and_response_exposure_are_part_of_reproduction(self):
        retained = self.retained_step()
        for change in (
            lambda row: row['native'].update(norm=9.),
            lambda row: row['responses'][0].update(anchors=1),
            lambda row: row['responses'][1].update(protected_occurrences=0),
        ):
            observed = deepcopy(retained)
            change(observed)
            self.assertEqual(diagnostic.reproduction_receipt(observed, retained)['status'], 'failed')

    def test_no_update_budget_and_durable_failure_before_cap(self):
        expected = dict(response_forward=10, draft_backward=4, optimizer_step=0,
            feedback_generation=0, teacher_preparation=0, serving_generation=0)
        self.assertEqual(diagnostic.BUDGET, expected)
        with tempfile.TemporaryDirectory() as directory:
            work = DiagnosticWork(directory, diagnostic.BUDGET)
            with self.assertRaisesRegex(RuntimeError, 'response failed'):
                with work.operation('response_forward'):
                    raise RuntimeError('response failed')
            for _ in range(9):
                with work.operation('response_forward'):
                    pass
            self.assertEqual(work.counts['response_forward_attempted'], 10)
            self.assertEqual(work.counts['response_forward_completed'], 9)
            self.assertEqual(work.counts['response_forward_failed'], 1)
            for kind in ('response_forward', 'optimizer_step', 'feedback_generation',
                         'teacher_preparation', 'serving_generation'):
                before = work.path.read_bytes()
                with self.assertRaisesRegex(ValueError, 'work limit exceeded'):
                    with work.operation(kind):
                        self.fail('Out-of-budget work executed')
                self.assertEqual(work.path.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
