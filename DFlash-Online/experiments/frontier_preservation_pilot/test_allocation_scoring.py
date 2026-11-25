"""CPU composition controls using the actual native FKL/K+/PAIR reducers."""

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from torch import nn

from online_feedback.kplus import KPlusRedistribution
from online_feedback.objective import TargetObjectiveModel
from online_feedback.prefix_allocation import PrefixAllocation
from online_feedback.ranking import fixed_anchors
from online_feedback.rng import capture_rng
from .allocation_scoring import NativeAllocationScorer
from .native import FrozenResponse, WorkReceipt, rng_digest
from .math import adamw_shadow, assessment, optimizer_vjp
from online_feedback.kplus import MASS_ATOL, MASS_RTOL


class RecordingHead(nn.Linear):
    def __init__(self):
        super().__init__(4, 4, bias=False, dtype=torch.bfloat16)
        self.weight.data.copy_(torch.eye(4))
        self.requires_grad_(False)
        self.shapes = []

    def forward(self, hidden):
        self.shapes.append(tuple(hidden.shape))
        return super().forward(hidden)


class TinyNativeModel(TargetObjectiveModel):
    """Shared trainable backbone; inherited native objective is left unchanged."""

    def __init__(self):
        nn.Module.__init__(self)
        self.draft_model = nn.Linear(4, 4, dtype=torch.bfloat16)
        with torch.no_grad():
            self.draft_model.weight.copy_(torch.eye(4) + torch.arange(16).reshape(4, 4) / 1000)
            self.draft_model.bias.copy_(torch.tensor([0.05, -0.1, 0.08, 0.02]))
        self.lm_head = RecordingHead()
        self.block_size = 16
        self.objective_chunk_blocks = 8
        self.loss_decay_gamma = 7.0
        self.objective = 'osd_first_error'
        self.loss_dtype = torch.bfloat16
        self.is_osd = self.uses_teacher = True
        self.ranking_enabled = False
        self.reference = None
        self.eos_token_ids = (3,)
        self.first_error_redistribution = KPlusRedistribution()
        self.prefix_allocation = PrefixAllocation(relative_cap=0)
        self.repair_conditional_kl = True
        self.costs = {'sentinel': 19.0}
        self.forward_calls = 0

    def _sample_anchor_positions(self, length, mask, device, max_valid_anchors=None):
        anchors = torch.arange(0, length, 16, device=device).unsqueeze(0)
        return anchors, torch.ones_like(anchors, dtype=torch.bool)

    def _forward_draft_blocks(
        self, input_ids, hidden_states, loss_mask, max_valid_anchors=None
    ):
        self.forward_calls += 1
        anchors, keep = self._sample_anchor_positions(
            input_ids.shape[1], loss_mask, input_ids.device
        )
        indices = (anchors.unsqueeze(-1) + torch.arange(16)).clamp(max=input_ids.shape[1] - 1)
        selected = hidden_states[:, indices.reshape(-1)]
        return anchors, keep, self.draft_model(selected)


def response(blocks, offset=0):
    size = blocks * 16
    labels = torch.zeros(1, size, dtype=torch.long)
    hidden = torch.zeros(1, size, 4, dtype=torch.bfloat16)
    hidden[..., 0] = 2
    for block in range(blocks):
        error = 3 + (block + offset) % 4
        for position in (error, error + (1 if block % 2 else 3)):
            hidden[0, block * 16 + position, 0] = 0
            hidden[0, block * 16 + position, 1] = 2
    teacher = torch.zeros_like(hidden)
    teacher[..., 0] = 2.5
    state = capture_rng()
    anchors = torch.arange(0, size, 16).unsqueeze(0)
    return FrozenResponse(
        'private-fixture',
        dict(
            input_ids=labels,
            hidden_states=hidden,
            loss_mask=torch.ones_like(labels),
            target_last_hidden_states=teacher,
        ),
        anchors,
        torch.ones_like(anchors, dtype=torch.bool),
        state,
        state,
    )


def fixture(work=None):
    model = TinyNativeModel()
    optimizer = torch.optim.AdamW(
        model.draft_model.parameters(),
        lr=0.003,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )
    for parameter in model.draft_model.parameters():
        optimizer.state[parameter] = dict(
            step=torch.tensor(128.0),
            exp_avg=torch.full_like(parameter, 0.002),
            exp_avg_sq=torch.full_like(parameter, 0.001),
        )
    trainer = SimpleNamespace(
        bundle=SimpleNamespace(model=model),
        backend=SimpleNamespace(
            optimizer=SimpleNamespace(optimizer=optimizer, max_grad_norm=1.0)
        ),
        core=SimpleNamespace(accumulation_remainder=0),
    )
    return NativeAllocationScorer(trainer, [response(10), response(3, 1)], work)


class AllocationScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def test_freeze_full_roles_native_chunks_and_teacher_alignment(self):
        scorer = fixture()
        snapshots = scorer.freeze_current()
        self.assertEqual(
            [s['roles']['labels'].shape for s in snapshots], [(1, 10, 16), (1, 3, 16)]
        )
        self.assertEqual(
            scorer.model.lm_head.shapes, [(1, 8, 16, 4), (1, 2, 16, 4), (1, 3, 16, 4)]
        )
        for snapshot in snapshots:
            roles = snapshot['roles']
            selected = roles['prefix'] | roles['repair']
            chunks = [
                (roles['beta'][:, i : i + 8] * selected[:, i : i + 8]).sum()
                for i in range(0, selected.shape[1], 8)
            ]
            expected = chunks[0]
            for value in chunks[1:]:
                expected = expected + value
            self.assertTrue(torch.equal(expected + 1e-6, roles['denominator']))
            self.assertTrue(torch.equal(snapshot['direct_loss_mask'], selected))
            self.assertTrue((snapshot['repair_coefficients'][~roles['repair']] == 0).all())
            self.assertTrue(snapshot['priority1'].any())
            self.assertTrue(snapshot['priority2'].any())
            self.assertFalse(snapshot['teacher_hidden'].requires_grad)
        self.assertNotEqual(
            float(snapshots[0]['roles']['denominator']),
            float(snapshots[1]['roles']['denominator']),
        )
        self.assertEqual(scorer.model.costs, {'sentinel': 19.0})

    def test_reference_is_exact_native_two_response_bf16_backward_accumulation(self):
        scorer = fixture()
        scorer.freeze_current()
        actual = scorer.reference_gradient()
        original_costs = scorer.model.costs
        for response_item in scorer.responses:
            with fixed_anchors(scorer.model, response_item.anchors, response_item.keep):
                loss, _, _ = scorer.model(**response_item.tensors)
            (loss / 2).backward()
        expected = [p.grad.detach().clone() for p in scorer.parameters]
        for parameter in scorer.parameters:
            parameter.grad = None
        self.assertIs(scorer.model.costs, original_costs)
        for observed, native in zip(actual, expected):
            self.assertEqual(observed.dtype, torch.bfloat16)
            self.assertTrue(torch.equal(observed, native), (observed, native))

    def test_sparse_old_C_observation_does_not_change_scoring_or_forward_count(self):
        ordinary = fixture()
        observed = fixture()
        observed.model.load_state_dict(ordinary.model.state_dict())
        observed.collect_diagnostics = True
        left, right = ordinary.freeze_current(), observed.freeze_current()
        self.assertEqual(ordinary.model.forward_calls, observed.model.forward_calls)
        self.assertEqual(ordinary.model.lm_head.shapes, observed.model.lm_head.shapes)
        for before, after in zip(left, right):
            for key in ('labels', 'beta', 'prefix', 'repair', 'denominator'):
                self.assertTrue(torch.equal(before['roles'][key], after['roles'][key]))
            self.assertNotIn('old_C_coefficients', before)
            self.assertEqual(after['old_C_coefficients'].shape, after['roles']['beta'].shape)
            self.assertFalse(after['old_C_coefficients'].requires_grad)
        for a, b in zip(ordinary.reference_gradient(), observed.reference_gradient()):
            self.assertTrue(torch.equal(a, b))

    def test_token_oracle_retains_original_eight_block_head_boundary_and_raw_loss(self):
        scorer = fixture()
        scorer.freeze_current()
        scorer.model.lm_head.shapes.clear()
        before = scorer.model.forward_calls
        loss = scorer.token_fkl(0, 9, 1)
        self.assertEqual(loss.dtype, torch.bfloat16)
        self.assertEqual(scorer.model.forward_calls, before + 1)
        self.assertEqual(scorer.model.lm_head.shapes, [(1, 2, 16, 4)] * 2)
        gradients = torch.autograd.grad(loss, scorer.parameters)
        self.assertTrue(all(g.isfinite().all() for g in gradients))
        self.assertTrue(all(p.grad is None for p in scorer.parameters))
        with self.assertRaisesRegex(ValueError, 'sealed current prefix'):
            scorer.token_fkl(0, 0, 8)

    def test_full_composition_freezes_state_and_repeats_fixed_represented_direction(self):
        scorer = fixture()
        parameter_state = [p.detach().clone() for p in scorer.parameters]
        optimizer_state = deepcopy(scorer.trainer.backend.optimizer.optimizer.state_dict())
        rng = rng_digest(capture_rng())
        prefix = scorer.model.prefix_allocation
        costs = scorer.model.costs
        scorer.freeze_current()
        gradient = scorer.reference_gradient()
        preview = scorer.preview_and_direction(gradient)
        self.assertEqual(len(preview['diagnostics']['response_assessments']), 2)
        self.assertTrue(
            all(w.dtype == torch.float32 and w.isfinite().all() for w in preview['w'])
        )
        first = scorer.fast_scores(preview['w'])
        second = scorer.fast_scores(preview['w'])
        for original, represented, repeated in zip(
            preview['w'], first['effective_direction'], second['effective_direction']
        ):
            self.assertTrue(torch.equal(original.to(torch.bfloat16), represented))
            self.assertTrue(torch.equal(represented, repeated))
        for a, b in zip(first['scores'], second['scores']):
            self.assertTrue(torch.equal(a, b))
            self.assertEqual(a.dtype, torch.float32)
        for a, b in zip(first['allocations'], second['allocations']):
            self.assertTrue(torch.equal(a, b))
        scorer.reference_gradient(first['allocations'])
        self.assertEqual(rng_digest(capture_rng()), rng)
        self.assertIs(scorer.model.prefix_allocation, prefix)
        self.assertIs(scorer.model.costs, costs)
        self.assertEqual(costs, {'sentinel': 19.0})
        self.assertIsNone(prefix.last_summary)
        for old, current in zip(parameter_state, scorer.parameters):
            self.assertTrue(torch.equal(old, current))
            self.assertIsNone(current.grad)
        after = scorer.trainer.backend.optimizer.optimizer.state_dict()
        self.assertEqual(optimizer_state['param_groups'], after['param_groups'])
        for key, values in optimizer_state['state'].items():
            for name, value in values.items():
                self.assertTrue(torch.equal(value, after['state'][key][name]))
        scorer.close()

    def test_preview_matches_direct_joint_two_response_assessment_gradient(self):
        scorer = fixture()
        scorer.freeze_current()
        gradient = scorer.reference_gradient()
        wrapper = scorer.trainer.backend.optimizer
        optimizer = wrapper.optimizer
        states = [optimizer.state[p] for p in scorer.parameters]
        preview, context = adamw_shadow(
            scorer.parameters,
            gradient,
            [s['exp_avg'] for s in states],
            [s['exp_avg_sq'] for s in states],
            [s['step'] for s in states],
            optimizer.param_groups[0],
            max_grad_norm=wrapper.max_grad_norm,
        )
        values = []
        for index, item in enumerate(scorer.responses):
            functional = {
                name: p.to(torch.bfloat16) for name, p in zip(scorer.parameter_names, preview)
            }
            hidden = torch.func.functional_call(scorer.hidden, functional, (item,)).reshape(
                1, -1, 16, 4
            )
            # Tiny fixture only: collect full logits from original chunk shapes
            # and reduce the independent assessment formula across the response.
            logits = torch.cat(
                [
                    scorer.model.lm_head(hidden[:, start : start + 8])
                    for start in range(0, hidden.shape[1], 8)
                ],
                dim=1,
            )
            value, _ = assessment(logits, scorer.snapshots[index]['roles'])
            values.append(value)
        cotangent = torch.autograd.grad((values[0] + values[1]) / 2, preview)
        expected = optimizer_vjp(context, cotangent)
        actual = scorer.preview_and_direction(gradient)
        for observed, direct in zip(actual['w'], expected):
            self.assertTrue(torch.allclose(observed, direct, rtol=MASS_RTOL, atol=MASS_ATOL))

    def test_fast_fkl_rows_are_promoted_then_normalized_once_per_response(self):
        scorer = fixture()
        scorer.freeze_current()
        direction = tuple(
            torch.full_like(p, 0.17, dtype=torch.float32) for p in scorer.parameters
        )

        def known_rows(hidden, teacher):
            primal, _ = torch.autograd.forward_ad.unpack_dual(hidden)
            rows = primal[..., 0]
            raw_direction = torch.arange(16, dtype=torch.bfloat16).expand_as(rows)
            return torch.autograd.forward_ad.make_dual(rows, raw_direction)

        with mock.patch.object(scorer, '_token_losses', side_effect=known_rows):
            fast = scorer.fast_scores(direction)
        for index in range(2):
            roles = scorer.snapshots[index]['roles']
            raw = torch.arange(16, dtype=torch.bfloat16).expand_as(roles['beta'])
            expected = -torch.where(roles['prefix'], raw.float(), 0.0) / (
                2 * roles['denominator']
            )
            self.assertTrue(torch.equal(fast['scores'][index], expected))
            self.assertTrue(
                torch.equal(fast['response_denominators'][index], roles['denominator'])
            )
        self.assertEqual(fast['accumulation_factor'], 0.5)
        self.assertEqual(fast['direction_scale'], 1.0)

    def test_checkpoint_work_accounting_and_no_pending_successful_operations(self):
        with TemporaryDirectory() as directory:
            work = WorkReceipt(Path(directory) / 'WORK.json')
            scorer = fixture(work)
            scorer.freeze_current()
            gradient = scorer.reference_gradient()
            preview = scorer.preview_and_direction(gradient)
            scorer.fast_scores(preview['w'])
            self.assertFalse(work.value['pending'])
            self.assertEqual(work.value['attempted'], work.value['completed'])
            self.assertEqual(work.value['native_adam_steps'], 0)

    def test_failure_restores_rng_cost_hook_and_leaves_attempt_visible(self):
        with TemporaryDirectory() as directory:
            work = WorkReceipt(Path(directory) / 'WORK.json')
            scorer = fixture(work)
            scorer.freeze_current()
            costs, prefix = scorer.model.costs, scorer.model.prefix_allocation
            state = rng_digest(capture_rng())
            with mock.patch.object(
                scorer.model, '_chunk_loss', side_effect=RuntimeError('fixture failure')
            ):
                with self.assertRaisesRegex(RuntimeError, 'fixture failure'):
                    scorer.reference_gradient()
            self.assertIs(scorer.model.costs, costs)
            self.assertIs(scorer.model.prefix_allocation, prefix)
            self.assertEqual(state, rng_digest(capture_rng()))
            self.assertTrue(work.value['pending'])
            self.assertTrue(all(p.grad is None for p in scorer.parameters))

    def test_empty_response_retains_window_half_and_native_no_error_branch(self):
        scorer = fixture()
        scorer.responses[1].tensors['loss_mask'].zero_()
        scorer.freeze_current()
        roles = scorer.snapshots[1]['roles']
        self.assertEqual(int(roles['valid'].sum()), 0)
        self.assertEqual(float(roles['denominator']), float(torch.tensor(1e-6)))
        gradient = scorer.reference_gradient()
        result = scorer.preview_and_direction(gradient)
        values = result['diagnostics']['response_assessments']
        self.assertEqual(values[1], 0)
        self.assertEqual(result['diagnostics']['window_assessment'], values[0] / 2)
        fast = scorer.fast_scores(result['w'])
        self.assertTrue((fast['scores'][1] == 0).all())
        self.assertTrue(torch.equal(fast['allocations'][1], roles['beta']))


if __name__ == '__main__':
    unittest.main()
