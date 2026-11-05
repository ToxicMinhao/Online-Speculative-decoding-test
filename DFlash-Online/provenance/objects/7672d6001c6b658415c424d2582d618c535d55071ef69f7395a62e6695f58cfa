import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import test_osd_recipe as legacy
from online_feedback import training as T

import torch

from runner import compare_states, tensor_difference, run_trials


class DifferenceTests(unittest.TestCase):
    def test_exact_and_chunked_known_difference(self):
        left = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.bfloat16)
        right = torch.tensor([[1., 4.], [3., 1.]], dtype=torch.bfloat16)
        result = tensor_difference(left, right, chunk_elements=1, sample_limit=1)
        self.assertEqual(result['unequal_count'], 2)
        self.assertEqual(result['unequal_fraction'], .5)
        self.assertEqual(result['finite_max_abs_error'], 3.)
        self.assertAlmostEqual(result['finite_discrepancy_l2'], 13 ** .5)
        self.assertEqual(result['samples'], [dict(flat_index=1, left=2., right=4., error=-2.)])
        self.assertTrue(tensor_difference(left, left)['exact_equal'])

    def test_large_integer_inequality_is_not_hidden_by_fp64_conversion(self):
        result = tensor_difference(torch.tensor([2 ** 60], dtype=torch.int64),
                                   torch.tensor([2 ** 60 + 1], dtype=torch.int64))
        self.assertFalse(result['exact_equal'])
        self.assertEqual(result['unequal_count'], 1)

    def test_nonfinite_is_preserved_without_invalid_json(self):
        result = tensor_difference(torch.tensor([float('nan'), float('inf'), 2.]),
                                   torch.tensor([1., float('inf'), 3.]))
        self.assertEqual(result['nonfinite_left'], 2)
        self.assertEqual(result['nonfinite_right'], 1)
        self.assertEqual(result['finite_pairs'], 1)
        self.assertEqual(result['unequal_count'], 2)
        self.assertEqual(result['samples'][0]['left'], 'nan')
        json.dumps(result, allow_nan=False)

    def test_all_state_components_are_reported_without_fail_fast(self):
        left = dict(draft={'q_proj': torch.tensor([1., 2.])}, optimizer={'step': torch.tensor(1)},
                    rng=torch.tensor([1, 2], dtype=torch.uint8), global_step=1)
        right = dict(draft={'q_proj': torch.tensor([1., 3.])}, optimizer={'step': torch.tensor(2)},
                     rng=torch.tensor([1, 3], dtype=torch.uint8), global_step=2)
        result = compare_states(left, right)
        self.assertFalse(result['exact_equal'])
        self.assertEqual(result['totals']['unequal_tensors'], 3)
        self.assertEqual(result['metadata_differences'][0]['name'], 'global_step')
        self.assertEqual(result['totals']['unequal_coordinates'], 3)

    def test_shape_dtype_and_missing_fields_remain_explicit(self):
        self.assertFalse(tensor_difference(torch.ones(2), torch.ones(3))['comparable'])
        result = compare_states({'x': torch.ones(2)}, {'y': torch.ones(2)})
        self.assertEqual(result['missing_left'], ['y'])
        self.assertEqual(result['missing_right'], ['x'])
        self.assertFalse(result['exact_equal'])


class RunnerIntegrationTests(unittest.TestCase):
    setUpClass = classmethod(legacy.OSDRecipeTests.setUpClass.__func__)
    tearDownClass = classmethod(legacy.OSDRecipeTests.tearDownClass.__func__)
    trainer = legacy.OSDRecipeTests.trainer
    saved = legacy.OSDRecipeTests.saved

    def setUp(self):
        legacy.OSDRecipeTests.setUp(self)
        patch = mock.patch.object(T, 'ranking_eos_token_ids', return_value=(28,))
        patch.start()
        self.addCleanup(patch.stop)

    def test_three_trials_retain_all_states_and_export_before_comparing(self):
        for index in range(8, 128):
            raw = torch.load(self.features[index % 2], weights_only=True)
            raw['record_id'] = f'response_{index}'
            raw['label_binding']['record_id'] = raw['record_id']
            path = self.root / f'feedback_{index}.pt'
            torch.save(raw, path)
            self.features.append(path)
        ordinary = self.trainer('ordinary', objective='osd')
        ordinary.train_event('e1', self.features)
        receipt = ordinary.events_root / 'e1/RESULT.json'
        ordinary.close()
        destination = self.root / 'diagnostic'
        with mock.patch('torch.cuda.reset_peak_memory_stats'), \
             mock.patch('torch.cuda.max_memory_allocated', return_value=0), \
             mock.patch('torch.cuda.max_memory_reserved', return_value=0):
            result = run_trials(self.config, receipt, destination)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['work_counts']['backward_completed'], 6)
        self.assertEqual(result['work_counts']['temporary_optimizer_step_completed'], 3)
        for label in ('O1', 'O2', 'U0'):
            for stage in ('initial', 'after_backward_0', 'after_backward_1', 'pre_optimizer', 'final'):
                self.assertTrue((destination / label / f'{stage}.pt').exists())
            self.assertTrue((destination / label / 'trainer/training_state.pt').exists())
            self.assertTrue((destination / label / 'export/model.safetensors').exists())
        for comparison in result['comparisons'].values():
            self.assertTrue(all(stage['exact_equal'] for stage in comparison.values()))


if __name__ == '__main__':
    unittest.main()
