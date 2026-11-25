"""Frozen origin margins, matched progress and state isolation on CPU only."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch

from online_feedback.rng import capture_rng
from online_feedback.storage import save_torch
from experiments.ok_coding.storage import identity
from experiments.ok_continuation.resume import differences
from .contract import digest
from .witness import (
    SCHEMA,
    MARGIN_SCHEMA,
    block_summary,
    compare_matched_witnesses,
    evaluate_witness,
    origin_margin_observations,
)


def logits_for(proposals, vocab=8):
    logits = torch.full((*proposals.shape, vocab), -2.0, dtype=torch.bfloat16)
    logits.scatter_(-1, proposals.unsqueeze(-1), 2.0)
    return logits


def readout(logits, labels, weights, *, arm='BASE', origin=None, eos=()):
    observations = origin_margin_observations(
        logits,
        labels,
        weights,
        eos,
        origin=None if origin is None else origin['rows'][0]['margin_observations'],
    )
    return dict(
        schema=SCHEMA,
        margin_schema=MARGIN_SCHEMA,
        status='passed',
        arm=arm,
        global_step=0 if origin is None else 128,
        witness_identity={'sha256': 'fixture'},
        origin_readout_digest=None if origin is None else digest(origin),
        rows=[
            dict(
                record_id='response0',
                anchors=list(range(labels.shape[1])),
                blocks=block_summary(logits.argmax(-1), labels, weights, eos),
                margin_observations=observations,
            )
        ],
    )


class MarginTests(unittest.TestCase):
    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def test_original_rival_stays_fixed_when_another_wrong_token_wins(self):
        labels = torch.tensor([[[0, 1, 2, 3]]])
        weights = torch.tensor([[[0.0, 1.0, 1.0, 1.0]]])
        source = logits_for(torch.tensor([[[0, 1, 5, 3]]]))
        origin = readout(source, labels, weights)
        endpoint = source.clone()
        endpoint[0, 0, 2, 2] = 1.0
        endpoint[0, 0, 2, 5] = 0.0
        endpoint[0, 0, 2, 7] = 3.0
        before = endpoint.clone()
        result = readout(endpoint.requires_grad_(), labels, weights, origin=origin)
        margin = result['rows'][0]['margin_observations'][0]
        self.assertEqual(margin['position'], 2)
        self.assertEqual(margin['original_rival_token_id'], 5)
        self.assertEqual(margin['current_wrong_token_id'], 7)
        self.assertEqual((margin['M_original'], margin['M_current']), (1.0, -2.0))
        self.assertTrue(margin['original_rival_addressed'])
        self.assertTrue(margin['rival_changed'])
        self.assertFalse(margin['correct'])
        self.assertTrue(torch.equal(endpoint, before))
        self.assertIsNone(endpoint.grad)
        result['arm'] = 'PAIR'
        matched = compare_matched_witnesses(
            origin, readout(source, labels, weights, origin=origin), result
        )['summary']
        self.assertEqual(matched['PAIR']['old_rival_addressed_but_still_wrong'], 1)
        self.assertEqual(matched['PAIR']['addressed_still_wrong_with_rival_change'], 1)
        self.assertEqual(matched['PAIR']['positive_original_margin_but_still_wrong'], 1)
        self.assertEqual(matched['PAIR_minus_BASE_M_original_mean'], 5.0)
        self.assertEqual(matched['PAIR_minus_BASE_M_current_mean'], 2.0)

    def test_native_ties_define_correctness_and_gathers_subtract_in_fp32(self):
        labels = torch.tensor([[[0, 1], [0, 5]]])
        weights = torch.tensor([[[0.0, 1.0], [0.0, 1.0]]])
        origin = readout(logits_for(torch.tensor([[[0, 4], [0, 2]]])), labels, weights)
        logits = torch.full((1, 2, 2, 8), -2.0, dtype=torch.bfloat16)
        logits[0, 0, 1, 1] = logits[0, 0, 1, 4] = 2.0
        logits[0, 1, 1, 5] = logits[0, 1, 1, 2] = 2.0
        result = readout(logits, labels, weights, origin=origin)
        first, second = result['rows'][0]['margin_observations']
        self.assertEqual(first['M_current'], 0.0)
        self.assertEqual(second['M_current'], 0.0)
        self.assertTrue(first['correct'])
        self.assertFalse(second['correct'])
        self.assertTrue(first['original_rival_addressed'])
        self.assertFalse(second['original_rival_addressed'])
        self.assertFalse(first['original_margin_positive'])
        logits[0, 0, 1, 1], logits[0, 0, 1, 4] = 256.0, -1.0
        values = readout(logits, labels, weights, origin=origin)['rows'][0][
            'margin_observations'
        ][0]
        self.assertEqual(values['M_original'], 257.0)
        self.assertNotEqual(
            values['M_original'], float(logits[0, 0, 1, 1] - logits[0, 0, 1, 4])
        )

    def test_no_error_anchor_gap_valid_eos_and_post_eos_support(self):
        labels = torch.tensor([[[0, 1, 2, 3, 4]] * 5])
        proposals = labels.clone()
        weights = torch.ones_like(labels, dtype=torch.float32)
        weights[..., 0] = 0
        proposals[0, 0, 0] = 7
        weights[0, 1, 2] = 0
        proposals[0, 1, 3] = 7
        labels[0, 2, 2] = 6
        proposals[0, 2, 2:] = 7
        labels[0, 3, 2] = proposals[0, 3, 2] = 6
        proposals[0, 3, 3:] = 7
        labels[0, 4, 0] = 6
        proposals[0, 4, 1:] = 7
        origin = readout(logits_for(proposals), labels, weights, eos=(6,))
        margins = origin['rows'][0]['margin_observations']
        self.assertEqual(
            [None if row is None else row['position'] for row in margins],
            [None, None, 2, None, None],
        )
        endpoint = readout(
            logits_for(torch.full_like(proposals, 7)), labels, weights, origin=origin, eos=(6,)
        )
        self.assertIsNone(endpoint['rows'][0]['margin_observations'][0])
        self.assertEqual(origin['rows'][0]['blocks']['prefix_length'][0], [4, 1, 1, 2, 0])

    def test_matched_useful_repair_prefix_damage_and_no_error_block_progress(self):
        labels = torch.tensor([[[0, 1, 2, 3, 4]] * 5])
        weights = torch.ones_like(labels, dtype=torch.float32)
        weights[..., 0] = 0
        initial = labels.clone()
        initial[0, :4, 3] = 7
        origin = readout(logits_for(initial), labels, weights)
        base = initial.clone()
        base[0, 0, 3] = base[0, 2, 3] = 3
        base[0, 1, 3] = 3
        base[0, 1, 1] = 7  # Boundary fixed but earlier prefix damaged.
        base[0, 4, 2] = 7  # No-error origin block still contributes damage.
        pair = initial.clone()
        pair[0, 1, 3] = pair[0, 2, 3] = 3
        result = compare_matched_witnesses(
            origin,
            readout(logits_for(base), labels, weights, origin=origin),
            readout(logits_for(pair), labels, weights, origin=origin, arm='PAIR'),
        )
        summary = result['summary']
        for name in ('PAIR_only', 'BASE_only', 'both', 'neither'):
            self.assertEqual(summary['useful_repair_' + name], 1)
        self.assertEqual(summary['origin_repair_occurrences'], 4)
        self.assertEqual(summary['block_occurrences'], 5)
        self.assertEqual(summary['BASE']['boundaries_repaired'], 3)
        self.assertEqual(summary['BASE']['useful_repairs'], 2)
        self.assertEqual(summary['BASE']['damaged_prefix_positions'], 2)
        self.assertEqual(summary['PAIR']['damaged_prefix_positions'], 0)
        self.assertEqual(summary['PAIR_minus_BASE_prefix_length'], 5)
        self.assertEqual(
            (
                summary['PAIR_improved_blocks'],
                summary['PAIR_regressed_blocks'],
                summary['PAIR_tied_blocks'],
            ),
            (2, 1, 2),
        )
        self.assertIsNone(result['per_block'][-1]['origin_margin'])
        self.assertNotIn('PAIR_minus_BASE_M_original', result['per_block'][-1])

    def test_margins_remain_when_endpoint_error_moves_earlier_and_binding_cannot_drift(self):
        labels = torch.tensor([[[0, 1, 2, 3]]])
        weights = torch.tensor([[[0.0, 1.0, 1.0, 1.0]]])
        origin = readout(logits_for(torch.tensor([[[0, 1, 7, 3]]])), labels, weights)
        endpoint = logits_for(torch.tensor([[[0, 7, 2, 3]]]))
        base = readout(endpoint, labels, weights, origin=origin)
        pair = readout(endpoint, labels, weights, arm='PAIR', origin=origin)
        result = compare_matched_witnesses(origin, base, pair)
        self.assertTrue(result['per_block'][0]['PAIR']['boundary_repaired'])
        self.assertFalse(result['per_block'][0]['PAIR']['useful_repair'])
        self.assertEqual(result['per_block'][0]['PAIR']['margin']['position'], 2)
        for mutate in (
            lambda row: row.update(origin_readout_digest='changed'),
            lambda row: row.update(global_step=0),
            lambda row: row['rows'][0].update(anchors=[99]),
            lambda row: row['rows'][0]['margin_observations'][0].update(
                original_rival_token_id=6
            ),
        ):
            bad = deepcopy(pair)
            mutate(bad)
            with self.assertRaises(ValueError):
                compare_matched_witnesses(origin, base, bad)

    def test_all_correct_origin_has_no_fabricated_margin_average(self):
        labels = torch.tensor([[[0, 1, 2, 3]]])
        weights = torch.tensor([[[0.0, 1.0, 1.0, 1.0]]])
        logits = logits_for(labels)
        origin = readout(logits, labels, weights)
        result = compare_matched_witnesses(
            origin,
            readout(logits, labels, weights, origin=origin),
            readout(logits, labels, weights, origin=origin, arm='PAIR'),
        )
        self.assertEqual(result['summary']['origin_repair_occurrences'], 0)
        self.assertIsNone(result['summary']['PAIR_minus_BASE_M_current_mean'])
        self.assertEqual(result['summary']['original_prefix_positions'], 3)


class FakeModel(torch.nn.Module):
    def __init__(self, trainer):
        super().__init__()
        self.draft_model = torch.nn.Linear(1, 1, bias=False)
        self.lm_head = torch.nn.Identity()
        self.block_size, self.objective_chunk_blocks = 4, 1
        self.eos_token_ids, self.loss_dtype = (), torch.bfloat16
        self.costs = {'existing': 3.0}
        self.trainer = trainer
        self._sample_anchor_positions = lambda *args, **kwargs: None

    def _forward_draft_blocks(self, input_ids, hidden_states, loss_mask):
        torch.rand(1)  # Verify observation preserves even a forward's RNG use.
        self.trainer.work['draft_forwards'] += 1
        anchors, keep = self._sample_anchor_positions()
        proposals = torch.tensor([[[0, 1, 7, 3], [0, 1, 2, 3]]])[:, : anchors.shape[1]]
        return anchors, keep, logits_for(proposals).reshape(1, -1, 8)

    def _block_labels_and_weights(self, input_ids, loss_mask, anchors, keep):
        labels = torch.tensor([[[0, 1, 2, 3]] * anchors.shape[1]])
        weights = torch.tensor([[[0.0, 1.0, 1.0, 1.0]] * anchors.shape[1]])
        return labels, weights, labels


class WitnessIsolationTests(unittest.TestCase):
    def test_observation_uses_only_existing_passes_and_preserves_state_even_on_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            trainer = SimpleNamespace(
                arm='BASE',
                global_step=0,
                presentations=0,
                events=[],
                work={'draft_forwards': 10},
                _anchor_observations=[{'saved': True}],
                optimizer={'moments': torch.tensor([0.5])},
            )
            model = FakeModel(trainer)
            model.train()
            model.lm_head.eval()
            next(model.draft_model.parameters()).grad = torch.tensor([[0.75]])
            trainer.bundle = SimpleNamespace(model=model)
            trainer._batch = lambda raw: SimpleNamespace(tensors=raw)
            rows = []
            for index in range(8):
                path = root / f'packet{index}.pt'
                save_torch(
                    path,
                    dict(
                        input_ids=torch.tensor([[0, 1, 2, 3]]),
                        hidden_states=torch.zeros(1),
                        loss_mask=torch.ones(1, 4),
                    ),
                )
                rows.append(
                    dict(
                        record_id=f'r{index}',
                        feature_path=str(path),
                        feature_identity=identity(path),
                        anchors=[0, 1] if index < 7 else [],
                        keep=[True, True] if index < 7 else [],
                    )
                )
            witness = root / 'bindings.pt'
            save_torch(witness, dict(schema=SCHEMA, rows=rows))
            before = dict(
                parameters=deepcopy(model.state_dict()),
                gradient=next(model.parameters()).grad.clone(),
                modes=[m.training for m in model.modules()],
                optimizer=deepcopy(trainer.optimizer),
                rng=capture_rng(),
                work=deepcopy(trainer.work),
                costs=deepcopy(model.costs),
                anchors=deepcopy(trainer._anchor_observations),
            )
            result = evaluate_witness(trainer, witness, root / 'origin.json', endpoint='origin')
            self.assertEqual(result['diagnostic_draft_forwards'], 7)
            self.assertEqual(result['diagnostic_head_calls'], 14)
            self.assertEqual(len(result['rows'][-1]['margin_observations']), 0)
            for key, actual in dict(
                parameters=model.state_dict(),
                gradient=next(model.parameters()).grad,
                modes=[m.training for m in model.modules()],
                optimizer=trainer.optimizer,
                rng=capture_rng(),
                work=trainer.work,
                costs=model.costs,
                anchors=trainer._anchor_observations,
            ).items():
                self.assertFalse(differences(actual, before[key]), key)
            trainer.global_step, trainer.presentations = 128, 256
            trainer.events = ['event_001']
            endpoint = evaluate_witness(
                trainer,
                witness,
                root / 'BASE1.json',
                origin_path=root / 'origin.json',
                endpoint='BASE1',
            )
            self.assertEqual(endpoint['origin_readout_digest'], digest(result))
            with patch.object(
                model.lm_head, 'forward', side_effect=RuntimeError('fixture failed head')
            ):
                with self.assertRaisesRegex(RuntimeError, 'fixture failed head'):
                    evaluate_witness(
                        trainer, witness, root / 'failed.json', origin_path=root / 'origin.json'
                    )
            failed = json.loads((root / 'failed.json').read_text())
            self.assertEqual(failed['status'], 'failed')
            self.assertEqual(failed['diagnostic_draft_forwards'], 1)
            self.assertEqual(failed['diagnostic_head_calls_attempted'], 1)
            self.assertEqual(failed['diagnostic_head_calls'], 0)
            self.assertEqual(trainer.work, before['work'])
            self.assertFalse(differences(capture_rng(), before['rng']))
            self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main()
