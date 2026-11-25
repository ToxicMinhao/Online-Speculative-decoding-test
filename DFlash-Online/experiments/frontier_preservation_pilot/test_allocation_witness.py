"""Fixed-support witness semantics and private native-path isolation on CPU."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from online_feedback.rng import capture_rng
from .allocation_witness import (
    block_readouts,
    evaluate_sets,
    freeze_sets,
    summarize_sets,
)
from .math import assessment, current_roles
from .native import rng_digest


def logits_for(proposals):
    value = torch.full((*proposals.shape, 32), -2.0, dtype=torch.bfloat16)
    return value.scatter_(-1, proposals.unsqueeze(-1), 2.0)


def example(blocks=1):
    labels = torch.arange(16).reshape(1, 1, 16).expand(1, blocks, 16).clone()
    weights = torch.ones_like(labels, dtype=torch.float32)
    weights[..., 0] = 0
    q = labels.clone()
    q[..., 4] = q[..., 7] = 31
    return labels, weights, q


class FakeModel(nn.Module):
    def __init__(self, trainer):
        super().__init__()
        self.draft_model = nn.Linear(1, 1, bias=False)
        self.lm_head = nn.Identity()
        self.block_size, self.objective_chunk_blocks = 16, 8
        self.eos_token_ids, self.loss_dtype = (), torch.bfloat16
        self.costs = {'sentinel': 9}
        self.trainer = trainer
        self.bad = False
        self._sample_anchor_positions = self.sample
        self.widths = []

    def sample(self, length, mask, device, max_valid_anchors=None):
        self.widths.append(max_valid_anchors)
        # Same native candidate validity/ranking; deterministic under sealed RNG.
        valid = (mask[:, :-1] > 0.5) & (mask[:, 1:] > 0.5)
        draws = torch.rand(valid.shape)
        draws[~valid] = 2
        anchors = draws.argsort(-1)[:, :max_valid_anchors].sort(-1).values
        return anchors, torch.ones_like(anchors, dtype=torch.bool)

    def _forward_draft_blocks(self, input_ids, hidden_states, loss_mask):
        torch.rand(1)
        self.costs['sentinel'] += 1
        self.trainer.work['draft'] += 1
        anchors, keep = self._sample_anchor_positions()
        if self.bad:
            raise RuntimeError('fixture native failure')
        _, _, q = example(anchors.shape[1])
        if self.trainer.global_step == 256:
            q[..., 4] = 4
            if self.trainer.fork_arm == 'CONTROL':
                q[..., 2] = 31
        return anchors, keep, logits_for(q).reshape(1, -1, 32)

    def _block_labels_and_weights(self, input_ids, loss_mask, anchors, keep):
        labels, weights, _ = example(anchors.shape[1])
        return labels, weights, labels


def fixture(root, *, empty_first=False):
    trainer = SimpleNamespace(
        arm='PAIR',
        fork_arm='CONTROL',
        global_step=128,
        presentations=256,
        events=[{'event_id': 'event_001'}],
        core=SimpleNamespace(accumulation_remainder=0),
        work={'draft': 10},
        _anchor_observations=[{'old': True}],
    )
    model = FakeModel(trainer)
    trainer.bundle = SimpleNamespace(model=model)
    trainer._original_sampler = model.sample
    trainer._batch = lambda raw: SimpleNamespace(tensors=raw['tensors'])
    paths = []
    for index in range(16):
        path = root / f'packet{index}.pt'
        mask = torch.ones(1, 25)
        if empty_first and index == 0:
            mask.zero_()
        torch.save(
            dict(
                record_id=f'record{index}',
                tensors=dict(
                    input_ids=torch.zeros(1, 25, dtype=torch.long),
                    hidden_states=torch.ones(1, 25, 1),
                    loss_mask=mask,
                ),
            ),
            path,
        )
        paths.append(path)
    return trainer, paths[:8], paths[8:]


class AllocationWitnessTests(unittest.TestCase):
    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def test_useful_repair_is_conditional_and_fixed_K_survives_role_change(self):
        labels, weights, q = example(2)
        origin = block_readouts(logits_for(q), labels, weights)
        self.assertEqual(
            [origin[0]['origin'][k] for k in ('L', 'e', 'H', 'K')], [3, 4, 3, 6]
        )
        q[..., 4] = 4
        q[0, 0, 2] = 31
        q[0, 1, 6] = 31
        endpoint = block_readouts(logits_for(q), labels, weights, origin=origin)
        first, second = [b['metrics'] for b in endpoint]
        self.assertEqual((first['raw_repair'], first['useful_repair']), (1, 0))
        self.assertEqual(
            (first['damaged_prefix_positions'], first['fixed_K_shortfall']), (1, 5)
        )
        self.assertEqual(
            (first['surviving_followers'], first['reachable_followers']), (2, 0)
        )
        self.assertEqual((second['raw_repair'], second['useful_repair']), (1, 1))
        self.assertEqual(
            (second['surviving_followers'], second['reachable_followers']), (1, 1)
        )
        self.assertEqual(endpoint[0]['origin'], origin[0]['origin'])
        self.assertEqual(len(endpoint[0]['positions']), 6)

    def test_assessment_matches_core_FP32_native_gather_and_fresh_rival(self):
        labels, weights, q = example()
        initial_logits = logits_for(q)
        origin = block_readouts(initial_logits, labels, weights)
        logits = initial_logits.clone()
        logits[0, 0, 4, 4] = 256
        logits[0, 0, 4, 31] = -1
        logits[0, 0, 4, 30] = 257  # BF16 rounds to256; native lowest-ID tie wins y.
        before = logits.clone()
        result = block_readouts(logits, labels, weights, origin=origin)
        frozen = current_roles(q, labels, weights)
        value, _ = assessment(logits, frozen)
        self.assertEqual(result[0]['metrics']['assessment'], float(value))
        position = result[0]['positions'][3]
        self.assertEqual(position['origin_rival_margin'], 257.0)
        self.assertTrue(position['rival_changed'])
        self.assertTrue(position['margin_tie'])
        self.assertTrue(position['correct'])
        self.assertTrue(torch.equal(logits, before))

    def test_gap_EOS_empty_and_no_error_keep_origin_support(self):
        labels, weights, q = example(4)
        q.copy_(labels)
        weights[0, 0, 3] = 0
        labels[0, 1, 5] = q[0, 1, 5] = 29
        weights[0, 2] = 0
        origin = block_readouts(logits_for(q), labels, weights, eos_token_ids=(29,))
        self.assertEqual([b['origin']['m'] for b in origin], [2, 5, 0, 15])
        self.assertEqual([b['origin']['e'] for b in origin], [-1] * 4)
        q[..., 2] = 31
        result = block_readouts(
            logits_for(q), labels, weights, eos_token_ids=(29,), origin=origin
        )
        self.assertEqual([b['metrics']['raw_repair'] for b in result], [0] * 4)
        self.assertEqual(result[2]['metrics']['assessment'], 0)
        self.assertEqual(len(result[3]['positions']), 15)
        bad = labels.clone()
        bad[0, 0, 1] = 20
        with self.assertRaisesRegex(ValueError, 'labels'):
            block_readouts(
                logits_for(q), bad, weights, eos_token_ids=(29,), origin=origin
            )

    def test_freeze_caps_native_draws_preserves_RNG_and_rejects_overlap(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            trainer, new, parent = fixture(root, empty_first=True)
            before = rng_digest(capture_rng())
            result = freeze_sets(trainer, new, parent, root / 'witness')
            self.assertEqual(result['live_rng_after'], before)
            self.assertEqual(result['sets']['new_buffer'][0]['anchors'], [])
            self.assertEqual(trainer.bundle.model.widths, [8] * 15)
            self.assertEqual(trainer.work, {'draft': 10})
            with self.assertRaisesRegex(ValueError, 'disjoint'):
                freeze_sets(trainer, new, new, root / 'overlap')
            self.assertEqual(rng_digest(capture_rng()), before)
            self.assertFalse((root / 'overlap/BINDINGS.json').exists())

    def test_all_endpoints_match_and_diagnostics_restore_production_state(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            trainer, new, parent = fixture(root, empty_first=True)
            witness = root / 'witness'
            freeze_sets(trainer, new, parent, witness)
            model = trainer.bundle.model
            model.train()
            model.lm_head.eval()
            before = rng_digest(capture_rng())
            original = evaluate_sets(trainer, witness, 'parent')
            self.assertEqual(original['draft_forwards_completed'], 15)
            self.assertEqual(original['head_calls_completed'], 15)
            self.assertEqual(original['physical_work'], {'draft': 15})
            self.assertEqual(trainer.work, {'draft': 10})
            self.assertEqual(model.costs, {'sentinel': 9})
            self.assertTrue(model.training)
            self.assertFalse(model.lm_head.training)
            self.assertEqual(rng_digest(capture_rng()), before)
            trainer.global_step, trainer.presentations = 256, 512
            trainer.events.append({'event_id': 'event_002'})
            evaluate_sets(trainer, witness, 'CONTROL')
            trainer.fork_arm = 'FRONTIER_P'
            evaluate_sets(trainer, witness, 'FRONTIER_P')
            result = summarize_sets(witness)
            group = result['sets']['new_buffer']
            self.assertEqual(group['block_occurrences'], 56)
            paired = group['CONTROL_to_FRONTIER_P']
            self.assertEqual(paired['raw_repair']['ties'], 56)
            self.assertEqual(paired['useful_repair']['gains'], 56)
            self.assertEqual(paired['fixed_K_shortfall']['gains'], 56)
            self.assertTrue(result['diagnostic_only'])
            self.assertEqual(trainer._anchor_observations, [{'old': True}])
            with self.assertRaises(FileExistsError):
                evaluate_sets(trainer, witness, 'FRONTIER_P')

    def test_failure_is_durable_and_does_not_leak_rng_modes_or_work(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            trainer, new, parent = fixture(root)
            witness = root / 'witness'
            freeze_sets(trainer, new, parent, witness)
            model = trainer.bundle.model
            model.bad = True
            before = rng_digest(capture_rng())
            with self.assertRaisesRegex(RuntimeError, 'fixture native failure'):
                evaluate_sets(trainer, witness, 'parent')
            receipt = json.loads((witness / 'parent.json').read_text())
            self.assertEqual(receipt['status'], 'failed')
            self.assertEqual(receipt['draft_forwards_attempted'], 1)
            self.assertEqual(receipt['draft_forwards_completed'], 0)
            self.assertEqual(receipt['physical_work'], {'draft': 1})
            self.assertEqual(trainer.work, {'draft': 10})
            self.assertEqual(model.costs, {'sentinel': 9})
            self.assertEqual(rng_digest(capture_rng()), before)
            self.assertTrue(model.training)

    def test_endpoint_counter_and_packet_changes_fail_before_model_work(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            trainer, new, parent = fixture(root)
            witness = root / 'witness'
            freeze_sets(trainer, new, parent, witness)
            with self.assertRaisesRegex(ValueError, 'endpoint'):
                evaluate_sets(trainer, witness, 'CONTROL')
            new[0].write_bytes(b'changed packet')
            with self.assertRaisesRegex(ValueError, 'packet changed'):
                evaluate_sets(trainer, witness, 'parent')
            receipt = json.loads((witness / 'parent.json').read_text())
            self.assertEqual(receipt['draft_forwards_attempted'], 0)


if __name__ == '__main__':
    unittest.main()
