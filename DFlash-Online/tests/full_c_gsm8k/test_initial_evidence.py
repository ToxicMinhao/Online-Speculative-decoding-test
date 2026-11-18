"""Initialization failures retain RNG evidence before the unchanged gate raises."""
from copy import deepcopy
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from experiments.full_c_gsm8k import training as T
from experiments.full_k_gsm8k import training as native
from online_feedback.rng import capture_rng, restore_rng
from test_full_k_training import own_state
from full_c_gsm8k.test_training import candidate_state


class InitialEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.global_rng = capture_rng()
        self.addCleanup(lambda: restore_rng(self.global_rng))
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_failure_retains_all_rng_components_and_complete_mismatch_paths(self):
        original, _ = own_state()
        original['rng']['python'] = random.Random(17).getstate()
        current = deepcopy(original)
        current['rng']['python'] = random.Random(18).getstate()
        current['rng']['torch'][0] ^= 1
        before = capture_rng()
        with self.assertRaisesRegex(ValueError, 'Independent arm initialization differs'):
            T.initial_comparison_evidence(original, current, self.root/'comparison.json')
        result = json.loads((self.root/'comparison.json').read_text())
        self.assertEqual(result['status'], 'failed')
        self.assertFalse(result['rng_component_equal']['python'])
        self.assertFalse(result['rng_component_equal']['torch'])
        self.assertTrue(result['rng_component_equal']['numpy'])
        self.assertTrue(result['rng_component_equal']['cuda'])
        self.assertGreater(result['mismatch_count'], 8)
        self.assertIn('state.rng.torch', result['component_mismatches']['rng'])
        self.assertEqual(result['mismatch_count'], len(result['mismatches']))
        for key in ('weights', 'optimizer', 'contract', 'global_step', 'presentations', 'events'):
            self.assertEqual(result['component_mismatches'][key], [])
        saved = result['current_rng']['python']['state']
        self.assertEqual(saved['type'], 'tuple')
        self.assertEqual(saved['values'][1]['values'], list(current['rng']['python'][1]))
        self.assertEqual(result['current_rng']['torch']['state']['values'], current['rng']['torch'].tolist())
        for side in ('original_rng', 'current_rng'):
            self.assertEqual(set(result[side]), set(original['rng']))
            for component in result[side].values():
                self.assertEqual(len(component['state_sha256']), 64)
        self.assertFalse(result['full_model_tensor_payload_saved'])
        self.assertFalse(result['rng_restored_to_match'])
        self.assertFalse(result['tolerance_changed'])
        self.assertEqual(native.differences(before, capture_rng()), [])

    def test_matching_case_preserves_exact_native_result_and_all_state(self):
        original, _ = own_state()
        current = deepcopy(original)
        before = deepcopy(current)
        rng_before = capture_rng()
        expected = native.compare_initial_states(original, current)
        actual = T.initial_comparison_evidence(original, current, self.root/'comparison.json')
        result = json.loads((self.root/'comparison.json').read_text())
        self.assertEqual(actual, expected)
        self.assertEqual(result['status'], 'passed')
        self.assertEqual(result['comparison_result'], expected)
        self.assertEqual(result['mismatch_count'], 0)
        self.assertTrue(all(result['rng_component_equal'].values()))
        self.assertEqual(native.differences(before, current), [])
        self.assertEqual(native.differences(rng_before, capture_rng()), [])

    def test_checking_evidence_exists_before_native_comparison_is_called(self):
        original, _ = own_state()
        path = self.root/'comparison.json'
        real_compare = native.compare_initial_states
        def compare(first, second):
            receipt = json.loads(path.read_text())
            self.assertEqual(receipt['status'], 'checking')
            self.assertEqual(set(receipt['current_rng']), set(second['rng']))
            return real_compare(first, second)
        with patch.object(native, 'compare_initial_states', side_effect=compare):
            T.initial_comparison_evidence(original, deepcopy(original), path)

    def test_save_initial_failure_does_not_publish_full_state_or_success_receipt(self):
        original, _ = own_state()
        current, args = candidate_state()
        original['rng']['python'] = random.Random(17).getstate()
        current['rng']['python'] = random.Random(18).getstate()
        trainer = object.__new__(T.FullCTrainer)
        trainer.output_root = self.root
        trainer.global_step = trainer.presentations = 0
        trainer.events = []
        trainer.binding = args['binding']
        trainer.cfg = SimpleNamespace(model=SimpleNamespace(draft_checkpoint_path='released'))
        trainer.original_config_text = args['original_config_text']
        trainer.expected_cuda_rng_states = 0
        trainer.validation_arguments = lambda: {}
        with patch.object(native, 'live_state', return_value=current), \
                patch.object(T, 'validate_snapshot', return_value={'status':'passed'}), \
                patch.object(T, '_export_equal'), \
                patch.object(torch, 'load', return_value=original), \
                patch.object(native, 'validate_snapshot', return_value={'status':'passed'}), \
                patch.object(T, 'save_torch') as save:
            with self.assertRaisesRegex(ValueError, 'Independent arm initialization differs'):
                trainer.save_initial('original_witness.pt')
        save.assert_not_called()
        self.assertEqual(json.loads((self.root/'INITIAL_COMPARISON.json').read_text())['status'], 'failed')
        self.assertFalse((self.root/'initial_state.pt').exists())
        self.assertFalse((self.root/'INITIAL_CHECK.json').exists())


if __name__ == '__main__':
    unittest.main()
