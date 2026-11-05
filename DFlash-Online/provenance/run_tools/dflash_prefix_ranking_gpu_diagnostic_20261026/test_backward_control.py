"""The bounded control uses real tiny backwards but no optimizer step/export."""
import unittest
from unittest import mock

import torch

import test_osd_recipe as legacy
from online_feedback import training as T
from backward_control import run_backward_control


class BackwardControlTests(unittest.TestCase):
    setUpClass = classmethod(legacy.OSDRecipeTests.setUpClass.__func__)
    tearDownClass = classmethod(legacy.OSDRecipeTests.tearDownClass.__func__)
    trainer = legacy.OSDRecipeTests.trainer
    saved = legacy.OSDRecipeTests.saved

    def setUp(self):
        legacy.OSDRecipeTests.setUp(self)

    def test_three_fresh_backward_graphs_restore_state_without_optimizer_or_export(self):
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
        destination = self.root / 'backward_control'
        with mock.patch('torch.cuda.reset_peak_memory_stats'), \
             mock.patch('torch.cuda.max_memory_allocated', return_value=0), \
             mock.patch('torch.cuda.max_memory_reserved', return_value=0), \
             mock.patch.object(T.SpecForgeEventTrainer, '_export', side_effect=AssertionError('No export')):
            result = run_backward_control(self.config, receipt, destination)
        self.assertEqual(result['status'], 'completed')
        self.assertTrue(result['final_state_restored_exactly'])
        self.assertEqual(result['work_counts']['backward_completed'], 3)
        self.assertEqual(result['work_counts']['optimizer_steps_attempted'], 0)
        self.assertEqual(len(result['repeats']), 3)
        self.assertTrue(all(row['exact_equal'] for row in result['comparisons'].values()))
        self.assertEqual(list(destination.rglob('training_state.pt')), [])
        self.assertEqual(list(destination.rglob('export')), [])


if __name__ == '__main__':
    unittest.main()
