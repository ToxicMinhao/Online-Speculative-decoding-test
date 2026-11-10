"""CPU fixture/cleanup checks; native maximum-shape admission is separate."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from experiments.kplus_abt import resource
from experiments.ok_continuation.resume import differences
from online_feedback.rng import capture_rng
from specforge.runtime.contracts import TrainBatch
from test_adversarial_block_runtime import build


class AdversarialBlockResourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def test_full_shape_right_padding_retains_values_dtype_and_native_anchor_pairs(self):
        _, batch, _, _, _ = build()
        original = deepcopy(batch.tensors)
        draw = dict(positions=[[4, 6, 8]], keep_mask=[[True, False, True]])
        padded, anchors, keep = resource.allocation_fixture(batch, draw)
        self.assertEqual(anchors.shape, (1, 512))
        self.assertEqual(anchors[0, :7].tolist(), [4, 6, 8, 4, 6, 8, 4])
        self.assertEqual(keep[0, :7].tolist(), [True, False, True, True, False, True, True])
        for name, values in original.items():
            self.assertEqual(padded.tensors[name].shape[1], 2272)
            self.assertEqual(padded.tensors[name].dtype, values.dtype)
            self.assertTrue(torch.equal(padded.tensors[name][:, :12], values))
            self.assertEqual(padded.tensors[name][:, 12:].count_nonzero().item(), 0)
            self.assertTrue(torch.equal(batch.tensors[name], values))
        self.assertFalse(padded.metadata['resource_fixture']['scientific_feedback'])
        self.assertFalse(padded.metadata['resource_fixture']['loss_interpretation'])

    def test_fixture_rejects_truncation_and_malformed_retained_anchors(self):
        _, batch, _, _, _ = build()
        with self.assertRaisesRegex(ValueError, 'truncation'):
            resource.allocation_fixture(batch, dict(positions=[[4]], keep_mask=[[True]]), sequence_length=4)
        for draw in (dict(positions=[[]], keep_mask=[[]]),
                     dict(positions=[[100]], keep_mask=[[True]]),
                     dict(positions=[[4]], keep_mask=[[False]])):
            with self.assertRaises(ValueError):
                resource.allocation_fixture(batch, draw)

    def fixture_trainer(self, directory):
        model, batch, author, backend, native = build()
        records, paths = [], []
        for ordinal in range(2):
            identifier = f'fixture{ordinal}'
            path = directory/f'{identifier}.pt'
            torch.save(dict(batch.tensors, record_id=identifier), path)
            paths.append(path)
            records.append(dict(record_id=identifier, sampled_anchors=[
                dict(positions=[[4, 6, 8, 9, 10]], keep_mask=[[True]*5])]))
        native._micro = 1026
        trainer = SimpleNamespace(bundle=SimpleNamespace(model=model, draft_model=model.draft_model),
            backend=backend, core=native, dataset='fixture', global_step=513, presentations=1026,
            horizon=7424, response_records=records, window_records=[dict(window=1)],
            _anchor_observations=deepcopy(records[-1]['sampled_anchors']),
            _native_forward=author.forward_loss, work=dict(draft_forwards=17, head_calls=41))
        trainer._batch = lambda raw: TrainBatch(sample_ids=[raw['record_id']], strategy='dflash',
            tensors={key:value for key, value in raw.items() if isinstance(value, torch.Tensor)})
        def count(name):
            def callback(module, args):
                trainer.work[name] += 1
            return callback
        model.draft_model.register_forward_pre_hook(count('draft_forwards'))
        model.lm_head.register_forward_pre_hook(count('head_calls'))
        return trainer, paths

    def test_four_views_zero_steps_restore_endpoint_rng_metadata_and_gradients(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trainer, paths = self.fixture_trainer(root)
            before_state = deepcopy(resource.cpu_copy(resource._state(trainer)))
            before_rng, before_work, before_costs = capture_rng(), deepcopy(trainer.work), deepcopy(trainer.bundle.model.costs)
            with patch.object(resource, 'SEQUENCE_LENGTH', 24), patch.object(resource, 'ANCHOR_BLOCKS', 7), \
                 patch.object(trainer.backend, 'step', side_effect=AssertionError('forbidden Adam')), \
                 patch.object(trainer.backend.optimizer, 'step', side_effect=AssertionError('forbidden Adam')), \
                 patch.object(trainer.bundle.model, '_sample_anchor_positions',
                              side_effect=AssertionError('forbidden resampling')):
                receipt = resource.run(trainer, paths, root/'resource')
            self.assertEqual(receipt['status'], 'passed')
            self.assertEqual(receipt['work']['completed'], resource.RESOURCE_BUDGET)
            self.assertEqual(receipt['work']['unknown'], {key:0 for key in resource.RESOURCE_BUDGET})
            self.assertEqual(receipt['physical_work']['draft_forwards'], 4)
            self.assertGreater(receipt['physical_work']['head_calls'], 0)
            self.assertEqual(trainer.backend.backward_boundaries, [False, False, False, True])
            self.assertEqual(trainer.backend.steps, 0)
            self.assertFalse(differences(before_state, resource._state(trainer)))
            self.assertFalse(differences(before_rng, capture_rng()))
            self.assertEqual(trainer.work, before_work)
            self.assertEqual(trainer.bundle.model.costs, before_costs)
            self.assertTrue(all(parameter.grad is None for parameter in trainer.bundle.model.parameters()))
            self.assertFalse(hasattr(trainer.bundle.model, 'adversarial_block'))
            self.assertEqual(len(receipt['rows']), 2)
            self.assertEqual(len(list((root/'resource').glob('RESPONSE_*.pt'))), 2)

    def test_failed_forward_keeps_partial_work_and_restores_state_without_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trainer, paths = self.fixture_trainer(root)
            original = trainer._native_forward
            calls = []
            def fail_third(batch, ctx):
                calls.append(1)
                if len(calls) == 3:
                    raise RuntimeError('resource fixture failure')
                return original(batch, ctx)
            trainer._native_forward = fail_third
            before_state, before_rng = resource.cpu_copy(resource._state(trainer)), capture_rng()
            with patch.object(resource, 'SEQUENCE_LENGTH', 24), patch.object(resource, 'ANCHOR_BLOCKS', 7):
                with self.assertRaisesRegex(RuntimeError, 'resource fixture failure'):
                    resource.run(trainer, paths, root/'resource')
            receipt = json.loads((root/'resource/RESULT.json').read_text())
            self.assertEqual(receipt['status'], 'failed')
            self.assertEqual(receipt['work']['attempts']['response_forwards'], 3)
            self.assertEqual(receipt['work']['completed']['response_forwards'], 2)
            self.assertEqual(receipt['work']['unknown']['response_forwards'], 1)
            self.assertEqual(receipt['work']['completed']['response_backwards'], 2)
            self.assertEqual(trainer.backend.steps, 0)
            self.assertFalse(differences(before_state, resource._state(trainer)))
            self.assertFalse(differences(before_rng, capture_rng()))
            self.assertTrue(receipt['gradients_cleared'])


if __name__ == '__main__':
    unittest.main()
