"""Persistence and RNG contracts shared by the serving and training adapters."""

import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from online_feedback.rng import capture_rng, preserve_rng, restore_rng
from online_feedback.storage import save_torch, write_json


class StorageTests(unittest.TestCase):
    def test_json_keeps_exact_text_and_tokens(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'nested/record.json'
            record = {'prompt': 'def f():\n    return 1\n', 'ids': [4, 8, 15]}
            write_json(path, record)
            self.assertEqual(json.loads(path.read_text()), record)

    def test_interrupted_tensor_save_preserves_committed_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'state.pt'
            save_torch(path, {'weight': torch.tensor([1.0])})
            with patch('torch.save', side_effect=OSError('disk full')):
                with self.assertRaises(OSError):
                    save_torch(path, {'weight': torch.tensor([2.0])})
            self.assertEqual(torch.load(path, weights_only=True)['weight'].item(), 1.0)
            self.assertFalse(path.with_name('state.pt.tmp').exists())

    def test_invalid_json_leaves_previous_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'record.json'
            write_json(path, {'finished': 1})
            with self.assertRaises(ValueError):
                write_json(path, {'metric': float('nan')})
            self.assertEqual(json.loads(path.read_text()), {'finished': 1})


class RngTests(unittest.TestCase):
    def draws(self):
        return random.random(), np.random.rand(), torch.rand(3)

    def test_replay_and_export_do_not_change_future_training_draws(self):
        initial = capture_rng()
        expected = self.draws()
        restore_rng(initial)
        with preserve_rng():
            self.draws()
            self.draws()
        actual = self.draws()
        self.assertEqual(actual[:2], expected[:2])
        self.assertTrue(torch.equal(actual[2], expected[2]))

    def test_exception_still_restores_rng(self):
        initial = capture_rng()
        expected = self.draws()
        restore_rng(initial)
        with self.assertRaises(RuntimeError):
            with preserve_rng():
                self.draws()
                raise RuntimeError('interrupted replay')
        actual = self.draws()
        self.assertEqual(actual[:2], expected[:2])
        self.assertTrue(torch.equal(actual[2], expected[2]))


if __name__ == '__main__':
    unittest.main()
