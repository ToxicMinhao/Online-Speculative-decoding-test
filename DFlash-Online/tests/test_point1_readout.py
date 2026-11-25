"""CPU coverage for frozen Point1 endpoints and native assessment reduction."""
from copy import deepcopy
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import torch
from torch import nn
import torch.nn.functional as F

from experiments.frontier_preservation_pilot.math import current_roles
from experiments.frontier_preservation_pilot.native import FrozenResponse, rng_digest
from experiments.frontier_preservation_pilot.point1_readout import (
    _chunk_assessment, compare_readouts, evaluate_responses, validate_disjoint, witness_responses,
)
from experiments.frontier_preservation_pilot.allocation_witness import block_readouts
from online_feedback.rng import capture_rng


class NativeFixture(nn.Module):
    def __init__(self, trainer):
        super().__init__()
        self.draft_model = nn.Linear(32, 32, bias=False, dtype=torch.bfloat16)
        with torch.no_grad():
            self.draft_model.weight.copy_(torch.eye(32))
        self.lm_head = nn.Identity()
        self.block_size, self.objective_chunk_blocks = 16, 8
        self.loss_dtype, self.eos_token_ids = torch.bfloat16, (29,)
        self.costs, self.trainer, self.head_shapes = {'old': 3}, trainer, []
        self._sample_anchor_positions = lambda: (_ for _ in ()).throw(AssertionError('resampled'))
        self.fail = False
        self.register_buffer('scratch', torch.tensor([7]))
        self._cached_seq_len = 11
        self.lm_head.register_forward_pre_hook(lambda module, args: self.head_shapes.append(tuple(args[0].shape)))

    def _forward_draft_blocks(self, input_ids, hidden_states, loss_mask):
        torch.rand(1)
        self.trainer.work['draft'] = self.trainer.work.get('draft', 0) + 1
        self.costs['temporary'] = 1
        self.scratch.add_(1)
        self._cached_seq_len = 99
        self.trainer._anchor_observations.append({'temporary': 1})
        if self.fail:
            raise RuntimeError('native fixture failure')
        anchors, keep = self._sample_anchor_positions()
        return anchors, keep, self.draft_model(hidden_states)

    def _block_labels_and_weights(self, input_ids, loss_mask, anchors, keep):
        return (input_ids.reshape(1, anchors.shape[1], 16),
                loss_mask.reshape(1, anchors.shape[1], 16), input_ids)


def fixture(blocks=9, record_id='first'):
    trainer = SimpleNamespace(global_step=128, presentations=256, events=[{'event': 1}],
                              core=SimpleNamespace(accumulation_remainder=0), work={'old': 7},
                              _anchor_observations=[{'old': 4}])
    model = NativeFixture(trainer)
    trainer.bundle = SimpleNamespace(model=model)
    trainer._batch = lambda raw: SimpleNamespace(tensors=raw['tensors'])
    labels = torch.arange(16).repeat(blocks).reshape(1, -1)
    weights = torch.ones_like(labels, dtype=torch.float32)
    weights[:, ::16] = 0
    proposals = labels.clone()
    proposals[:, 4::16] = proposals[:, 7::16] = 31
    hidden = torch.full((1, blocks * 16, 32), -2.0, dtype=torch.bfloat16)
    hidden.scatter_(-1, proposals.unsqueeze(-1), 2.0)
    state = capture_rng()
    anchors = torch.arange(blocks).reshape(1, -1) * 16
    response = FrozenResponse(record_id, dict(input_ids=labels, hidden_states=hidden, loss_mask=weights),
                              anchors, torch.ones_like(anchors, dtype=torch.bool), state, state)
    return trainer, response


class Point1ReadoutTests(unittest.TestCase):
    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def test_native_chunk_shapes_reduction_and_state_isolation(self):
        trainer, response = fixture()
        model = trainer.bundle.model
        model.train()
        model.lm_head.eval()
        parameter = next(model.draft_model.parameters())
        parameter.grad = torch.ones_like(parameter)
        gradient = parameter.grad.clone()
        old_rng = rng_digest(capture_rng())
        result = evaluate_responses(trainer, [response], endpoint='parent')
        self.assertEqual(model.head_shapes, [(1, 8, 16, 32), (1, 1, 16, 32)])
        self.assertEqual((result['head_calls'], result['draft_forwards'], result['backwards']), (2, 1, 0))
        self.assertEqual(trainer.work, {'old': 7})
        self.assertEqual(model.costs, {'old': 3})
        self.assertEqual(model.scratch.item(), 7)
        self.assertEqual(model._cached_seq_len, 11)
        self.assertEqual(trainer._anchor_observations, [{'old': 4}])
        self.assertTrue(model.training)
        self.assertFalse(model.lm_head.training)
        self.assertTrue(torch.equal(parameter.grad, gradient))
        self.assertEqual(rng_digest(capture_rng()), old_rng)
        logits = response.tensors['hidden_states'].reshape(1, 9, 16, 32)
        labels = response.tensors['input_ids'].reshape(1, 9, 16)
        roles = current_roles(logits.argmax(-1), labels, response.tensors['loss_mask'].reshape(1, 9, 16))
        numerator = None
        for start in (0, 8):
            part = slice(start, start + 8)
            mask = roles['assessment_mask'][:, part]
            selected, y = logits[:, part][mask], labels[:, part][mask]
            wrong = selected.clone().scatter(-1, y.unsqueeze(-1), -torch.inf)
            margin = (selected.gather(-1, y.unsqueeze(-1)).float()
                      - wrong.max(-1, keepdim=True).values.float()).squeeze(-1)
            term = (roles['assessment_weight'][:, part][mask] * F.softplus(-margin) / math.log(2)).sum()
            numerator = term if numerator is None else numerator + term
        self.assertEqual(result['mean_response_assessment_fp32'], float(numerator / 9))

    def test_functional_BF16_endpoints_leave_native_weights_and_counters(self):
        trainer, response = fixture(1)
        native = trainer.bundle.model.draft_model.weight.detach().clone()
        parent = evaluate_responses(trainer, [response], endpoint='parent')
        values = native.float().clone()
        values[4, 31] = 3
        result = evaluate_responses(trainer, [response], endpoint='Q_F', origins=parent,
                                    parameters={'model.draft_model.weight': values})
        self.assertTrue(result['functional_endpoint'])
        self.assertEqual(result['live_counters'], [128, 256, 1])
        self.assertTrue(torch.equal(trainer.bundle.model.draft_model.weight, native))
        self.assertEqual(result['responses'][0]['blocks'][0]['metrics']['useful_repair'], 1)
        contrast = compare_readouts(parent, result)
        self.assertGreater(contrast['assessment_benefit_fp64'], 0)
        self.assertEqual(contrast['block_metrics']['useful_repair']['gains'], 1)
        self.assertAlmostEqual(contrast['assessment_benefit_fp64'],
                               contrast['positive_contribution_sum_fp64'] + contrast['negative_contribution_sum_fp64'])
        trainer.global_step, trainer.presentations = 129, 258
        evaluate_responses(trainer, [response], endpoint='N_C', origins=parent)
        trainer.global_step, trainer.presentations = 256, 512
        with self.assertRaisesRegex(ValueError, 'boundary'):
            evaluate_responses(trainer, [response], endpoint='N_C', origins=parent)

    def test_failure_restores_mode_RNG_costs_and_observations(self):
        trainer, response = fixture(1)
        model = trainer.bundle.model
        model.train()
        model.fail = True
        before = rng_digest(capture_rng())
        with self.assertRaisesRegex(RuntimeError, 'native fixture'):
            evaluate_responses(trainer, [response], endpoint='parent')
        self.assertTrue(model.training)
        self.assertEqual(rng_digest(capture_rng()), before)
        self.assertEqual(trainer.work, {'old': 7})
        self.assertEqual(model.costs, {'old': 3})
        self.assertEqual(model.scratch.item(), 7)
        self.assertEqual(model._cached_seq_len, 11)
        self.assertEqual(trainer._anchor_observations, [{'old': 4}])

    def test_fixed_origins_keep_EOS_gap_and_recompute_wrong_rival(self):
        trainer, response = fixture(3)
        labels = response.tensors['input_ids'].reshape(1, 3, 16)
        weights = response.tensors['loss_mask'].reshape(1, 3, 16)
        logits = response.tensors['hidden_states'].reshape(1, 3, 16, 32)
        weights[0, 0, 3] = 0
        labels[0, 1, 5] = 29
        weights[0, 2] = 0
        origin = block_readouts(logits, labels, weights, eos_token_ids=(29,))
        logits[0, 1, 4, 31] = -3
        logits[0, 1, 4, 30] = 4
        rows = block_readouts(logits, labels, weights, eos_token_ids=(29,), origin=origin)
        self.assertEqual([row['origin']['m'] for row in rows], [2, 5, 0])
        self.assertEqual(rows[1]['positions'][3]['rival'], 30)
        self.assertTrue(rows[1]['positions'][3]['rival_changed'])
        self.assertTrue(torch.isfinite(_chunk_assessment(rows, torch.device('cpu'))))
        self.assertEqual(rows[2]['assessment_sum_fp64'], 0)
        self.assertEqual(rows[2]['positions'], [])

    def test_empty_response_preserves_fixed_mean_denominator(self):
        trainer, response = fixture(1)
        _, empty = fixture(1, 'empty')
        empty.tensors['loss_mask'].zero_()
        result = evaluate_responses(trainer, [response, empty], endpoint='parent')
        first, second = result['responses']
        self.assertEqual(second['nonempty_blocks'], 0)
        self.assertEqual(second['response_assessment_fp32'], 0)
        self.assertEqual(result['mean_response_assessment'], first['response_assessment_fp32'] / 2)
        contrast = compare_readouts(result, result)
        self.assertEqual(contrast['assessment_benefit_fp64'], 0)
        self.assertEqual(contrast['positive_contribution_sum_fp64'], 0)
        self.assertEqual(contrast['negative_contribution_sum_fp64'], 0)

    def test_disjointness_checks_identifiers_and_paths(self):
        first = [dict(record_id='a', feature_path='/one')]
        second = [dict(record_id='b', feature_path='/two')]
        self.assertTrue(validate_disjoint(first, second)['feature_paths_disjoint'])
        with self.assertRaisesRegex(ValueError, 'identities overlap'):
            validate_disjoint(first, [dict(record_id='a', feature_path='/two')])
        with self.assertRaisesRegex(ValueError, 'paths overlap'):
            validate_disjoint(first, [dict(record_id='b', feature_path='/one')])

    def test_witness_loader_replays_anchors_without_sampling(self):
        trainer, response = fixture(1)
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'packet.pt'
            torch.save(dict(record_id=response.record_id, tensors=response.tensors), path)
            before = rng_digest(capture_rng())
            rows = witness_responses(trainer, [dict(record_id=response.record_id, feature_path=str(path), anchors=[0])])
            self.assertEqual(rows[0].anchors.tolist(), [[0]])
            self.assertEqual(rng_digest(capture_rng()), before)
            with self.assertRaisesRegex(ValueError, 'ID changed'):
                witness_responses(trainer, [dict(record_id='different', feature_path=str(path), anchors=[0])])

    def test_paired_readout_rejects_support_or_identity_change(self):
        trainer, response = fixture(1)
        first = evaluate_responses(trainer, [response], endpoint='parent')
        second = deepcopy(first)
        second['responses'][0]['blocks'][0]['origin']['K'] -= 1
        with self.assertRaisesRegex(ValueError, 'origins differ'):
            compare_readouts(first, second)
        second = deepcopy(first)
        second['responses'][0]['record_id'] = 'different'
        with self.assertRaisesRegex(ValueError, 'identities'):
            compare_readouts(first, second)


if __name__ == '__main__':
    unittest.main()
