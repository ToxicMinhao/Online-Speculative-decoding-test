"""KR scheduling and real tiny-CPU K/Adam integration; no CUDA execution."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest import mock

import torch

from online_feedback import training as T
from online_feedback.reference_replay_event import ReferenceReplayEventTrainer, expand_reference_schedule
from online_feedback.rng import capture_rng
from .protocol import build_replay_schedule, validate_replay_schedule, check_training
from test_training_contract import trees_equal
import test_osd_recipe as legacy


def small_reference(receipt):
    count = len(receipt['features'])
    windows = []
    for pass_index in range(receipt['passes']):
        records = receipt['records'][pass_index*count:(pass_index+1)*count]
        pairs = [records[index:index+2] for index in range(0, count, 2)]
        for index, pair in enumerate(pairs):
            def binding(rows):
                return [{key: deepcopy(row[key]) for key in ('record_id', 'pass', 'sampled_anchors')}
                        for row in rows]
            windows.append(dict(step=len(windows)+1, pass_index=pass_index, batch_index=index,
                                reference_batch_index=(index+1) % len(pairs),
                                learning=binding(pair), reference=binding(pairs[(index+1) % len(pairs)])))
    return dict(features=deepcopy(receipt['features']), passes=receipt['passes'], windows=windows)


def full_receipt():
    features = [dict(record_id=f'r{i}', path=f'/{i}.pt') for i in range(128)]
    rows = []
    for pass_index in range(2):
        order = list(range(128)) if pass_index == 0 else list(reversed(range(128)))
        for offset, record in enumerate(order):
            rows.append(dict(record_id=f'r{record}', optimizer_stepped=bool(offset % 2),
                             optimizer_step_after=64*pass_index+(offset+1)//2,
                             sampled_anchors=[dict(positions=[[record+pass_index]], keep_mask=[[True]])]))
            rows[-1]['pass'] = pass_index
    return dict(features=features, records=rows)


class ReplayScheduleTests(unittest.TestCase):
    def test_full_schedule_counts_cyclic_wrap_and_same_pass_anchors(self):
        original = full_receipt()
        schedule = build_replay_schedule(original)
        self.assertTrue(validate_replay_schedule(schedule, original)['passed'])
        self.assertEqual((schedule['outer_windows'], schedule['optimizer_steps'],
                          schedule['response_presentations']), (128, 256, 512))
        self.assertEqual([row['role'] for row in schedule['updates']], ['A', 'R'] * 128)
        exposure = Counter((row['role'], item['record_id'], item['pass'])
                           for row in schedule['updates'] for item in row['records'])
        self.assertTrue(all(value == 1 for value in exposure.values()))
        self.assertEqual(len(exposure), 512)
        for final, first in ((127, 0), (255, 128)):
            self.assertEqual(schedule['updates'][final]['records'], schedule['updates'][first]['records'])
        self.assertNotEqual(schedule['updates'][127]['records'], schedule['updates'][128]['records'])
        for index in range(0, 256, 2):
            a, r = schedule['updates'][index:index+2]
            self.assertFalse({row['record_id'] for row in a['records']}
                             & {row['record_id'] for row in r['records']})

    def test_mutation_wrong_pass_or_evaluation_overlap_is_rejected(self):
        original = full_receipt()
        schedule = build_replay_schedule(original)
        schedule['updates'][1]['records'][0]['pass'] = 1
        with self.assertRaisesRegex(ValueError, 'binding'):
            validate_replay_schedule(schedule, original)
        with self.assertRaisesRegex(ValueError, 'Evaluation'):
            build_replay_schedule(original, evaluation_ids=['r0'])
        wrong = deepcopy(original)
        wrong['records'][0]['optimizer_stepped'] = True
        with self.assertRaisesRegex(ValueError, 'boundaries'):
            build_replay_schedule(wrong)

    def test_old_K_receipt_cannot_be_relabelled_as_KR_accounting(self):
        original = full_receipt()
        schedule = build_replay_schedule(original)
        receipt = dict(original, status='completed', training_variant='KR', objective='osd_first_error',
                       passes=2, response_presentations=256, optimizer_steps=128,
                       start_step=0, end_step=128)
        with self.assertRaisesRegex(ValueError, '512'):
            check_training(receipt, schedule)


class ReplayTrainingTests(unittest.TestCase):
    setUpClass = classmethod(legacy.OSDRecipeTests.setUpClass.__func__)
    tearDownClass = classmethod(legacy.OSDRecipeTests.tearDownClass.__func__)
    saved = legacy.OSDRecipeTests.saved

    def setUp(self):
        legacy.OSDRecipeTests.setUp(self)
        patch = mock.patch.object(T, 'ranking_eos_token_ids', return_value=(28,))
        patch.start()
        self.addCleanup(patch.stop)

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def trainer(self, name, replay=False):
        kind = ReferenceReplayEventTrainer if replay else T.SpecForgeEventTrainer
        trainer = kind(self.config, self.root / name, objective='osd_first_error')
        self.addCleanup(trainer.close)
        return trainer

    def ordinary(self, capture_first=None):
        trainer = self.trainer('K')
        if capture_first is not None:
            original = trainer.backend.optimizer.optimizer.step
            def step(*args, **kwargs):
                result = original(*args, **kwargs)
                if not capture_first:
                    capture_first.update(weights=deepcopy(trainer.bundle.model.state_dict()),
                                         optimizer=deepcopy(trainer.backend.optimizer.state_dict()),
                                         rng=capture_rng(),
                                         gradients={name: None if param.grad is None else param.grad.clone()
                                                    for name, param in trainer.bundle.draft_model.named_parameters()})
                return result
            trainer.backend.optimizer.optimizer.step = step
        receipt = trainer.train_event('event', self.features[:4], passes=2)
        return receipt, expand_reference_schedule(small_reference(receipt))

    def test_two_pass_replay_keeps_first_A_exact_and_one_continuous_optimizer(self):
        first = {}
        ordinary, schedule = self.ordinary(first)
        trainer = self.trainer('KR', replay=True)
        checked = []
        native = trainer.backend.optimizer.optimizer
        native_step = native.step
        first_state = {}
        before_steps = []
        def observed_step(*args, **kwargs):
            before_steps.append([float(state['step']) for state in native.state.values()])
            result = native_step(*args, **kwargs)
            if not first_state:
                first_state.update(weights=deepcopy(trainer.bundle.model.state_dict()),
                                   optimizer=deepcopy(trainer.backend.optimizer.state_dict()), rng=capture_rng(),
                                   gradients={name: None if param.grad is None else param.grad.clone()
                                              for name, param in trainer.bundle.draft_model.named_parameters()})
            return result
        native.step = observed_step
        def first_gate(rows):
            keys = ('record_id', 'pass', 'sampled_anchors', 'metrics', 'optimizer_stepped',
                    'optimizer_step_after', 'learning_rate')
            self.assertEqual([{key: row[key] for key in keys} for row in rows],
                             [{key: row[key] for key in keys} for row in ordinary['records'][:2]])
            self.assertEqual(trainer.global_step, 1)
            checked.append(rows)
        receipt = trainer.train_replay_event('event', self.features[:4], schedule, first_update_check=first_gate)
        self.assertTrue(trees_equal(first_state, first))
        self.assertEqual(len(checked), 1)
        gate = json.loads((trainer.events_root / 'event/FIRST_AR_WINDOW.json').read_text())
        self.assertEqual(gate['status'], 'passed')
        self.assertTrue(gate['first_A_record_gate_passed'])
        self.assertEqual(gate['native_Adam_steps'], [1, 2])
        self.assertEqual(gate['additional_forwards'], 0)
        self.assertIs(trainer.backend.optimizer.optimizer, native)
        self.assertEqual((receipt['optimizer_steps'], receipt['response_presentations']), (8, 16))
        self.assertEqual(receipt['training_variant'], 'KR')
        self.assertEqual([row['role'] for row in receipt['records']], ['A','A','R','R'] * 4)
        for step, counters in enumerate(before_steps):
            self.assertTrue(all(value == step for value in counters))
        self.assertEqual(len(before_steps), 8)
        work = receipt['reference_replay']['counts']
        self.assertEqual(work['response_forwards_completed'], 16)
        self.assertEqual(work['backward_completed'], 16)
        self.assertEqual(work['Adam_completed'], 8)
        self.assertEqual(work['anchor_rng_draws'], 16)
        self.assertEqual(work['shuffle_rng_draws'], 2)
        self.assertEqual(work['outer_windows_completed'], 4)
        self.assertNotIn('prefix_guard', trainer.bundle.model.__dict__)
        self.assertIsNone(trainer._active_binding)
        saved = self.saved(trainer)
        self.assertEqual(saved['events'][0]['training_variant'], 'KR')
        self.assertEqual(saved['algorithm_contract']['reference_replay_event']['arm'], 'KR')
        self.assertEqual((saved['global_step'], saved['presentations']), (8, 16))
        boundaries = [row for row in receipt['records'] if row['optimizer_stepped']]
        self.assertEqual(receipt['selection_totals']['response_presentations'], 16)
        self.assertEqual(receipt['selection_totals']['selected_position_occurrences'],
                         sum(row['metrics']['selected_position_occurrences'] for row in boundaries))

    def test_repeated_run_reproduces_weights_moments_rng_and_all_bindings(self):
        ordinary, schedule = self.ordinary()
        receipts, states = [], []
        for name in ('KR1', 'KR2'):
            trainer = self.trainer(name, replay=True)
            receipts.append(trainer.train_replay_event('event', self.features[:4], schedule))
            states.append(self.saved(trainer))
        for key in ('draft_state_dict', 'backend', 'rng', 'global_step', 'presentations'):
            self.assertTrue(trees_equal(states[0][key], states[1][key]), key)
        for receipt in receipts:
            for update in schedule['updates']:
                actual = receipt['records'][2*(update['optimizer_step']-1):2*update['optimizer_step']]
                self.assertEqual([row['sampled_anchors'] for row in actual],
                                 [row['sampled_anchors'] for row in update['records']])

    def test_each_update_averages_two_separately_normalized_response_losses(self):
        ordinary, schedule = self.ordinary()
        trainer = self.trainer('normalization', replay=True)
        forward = trainer.strategy.forward_loss
        terms = []
        def observe(*args, **kwargs):
            output = forward(*args, **kwargs)
            numerator, denominator = output.loss_terms
            terms.append((float(numerator.detach()), float(denominator.detach())))
            return output
        trainer.strategy.forward_loss = observe
        receipt = trainer.train_replay_event('event', self.features[:4], schedule)
        self.assertEqual(len(terms), 16)
        self.assertTrue(all(denominator == 1 for _, denominator in terms))
        for index in range(8):
            expected = (terms[2*index][0] + terms[2*index+1][0]) / 2
            actual = receipt['records'][2*index+1]['metrics']['loss']
            self.assertAlmostEqual(actual, expected, places=7)

    def test_gate_failure_stops_before_first_R_and_preserves_first_A_work(self):
        ordinary, schedule = self.ordinary()
        trainer = self.trainer('failed', replay=True)
        def stop(rows):
            raise ValueError('First A mismatch')
        with self.assertRaisesRegex(ValueError, 'First A mismatch'):
            trainer.train_replay_event('event', self.features[:4], schedule, first_update_check=stop)
        report = json.loads((trainer.events_root / 'event/REPLAY_FAILED.json').read_text())
        self.assertEqual(report['counts']['Adam_completed'], 1)
        self.assertEqual(report['counts']['backward_completed'], 2)
        self.assertEqual(report['counts']['presentations_completed'], 2)
        self.assertNotIn('R_backward_attempted', report['counts'])
        self.assertEqual(report['active_update']['role'], 'A')
        self.assertTrue(trainer.failed)
        self.assertFalse(trainer.latest.exists())
        self.assertEqual(len((trainer.events_root / 'event/REPLAY_PRESENTATIONS.jsonl').read_text().splitlines()), 2)

    def test_selected_masks_are_recomputed_by_current_K_forward(self):
        ordinary, schedule = self.ordinary()
        trainer = self.trainer('masks', replay=True)
        from online_feedback import objective
        original = objective.first_error_mask
        seen = []
        def observe(proposals, labels, weights, eos_token_ids):
            mask = original(proposals, labels, weights, eos_token_ids)
            seen.append((trainer.global_step, mask.clone(), proposals.clone()))
            return mask
        with mock.patch.object(objective, 'first_error_mask', side_effect=observe):
            receipt = trainer.train_replay_event('event', self.features[:4], schedule)
        self.assertTrue(seen)
        self.assertEqual({step for step, _, _ in seen}, set(range(8)))
        self.assertTrue(all(not mask.requires_grad for _, mask, _ in seen))
        self.assertEqual(receipt['reference_replay']['counts']['backward_completed'], 16)
