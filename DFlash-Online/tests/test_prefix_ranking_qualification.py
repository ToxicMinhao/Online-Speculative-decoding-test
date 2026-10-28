"""CPU qualification-control tests; no full models, native runtime or CUDA."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from experiments.prefix_ranking.qualify import (
    PLAN, RankingObserver, WorkLog, check_rival_gradient, compare_detached_blocks,
    prefix_oracle, run_window, run_qualification,
)
from online_feedback.objective import TargetObjectiveModel
from online_feedback.ranking import matching_prefix_mask
from online_feedback.rng import capture_rng
from online_feedback.qualify import assert_state_equal
from test_objective_integration import tiny_author
from test_osd_objective import osd_batch
import test_osd_recipe as legacy
from online_feedback import training as T


class QualificationHelpersTests(unittest.TestCase):
    def test_prefix_oracle_handles_mismatch_eos_gaps_seed_and_boundary(self):
        labels = torch.tensor([[[0, 1, 2, 9, 4], [9, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]])
        proposals = labels.clone()
        proposals[0, 2, 2] = 7
        weights = torch.tensor([[[0., 1., 1., 1., 1.], [0., 1., 1., 1., 1.],
                                 [0., 1., 1., 1., 1.], [0., 1., 0., 1., 0.]]])
        expected = torch.tensor([[[0, 1, 1, 1, 0], [0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 1, 0, 0, 0]]], dtype=torch.bool)
        self.assertTrue(torch.equal(prefix_oracle(proposals, labels, weights, (9,)), expected))
        self.assertTrue(torch.equal(matching_prefix_mask(proposals, labels, weights, (9,)), expected))

    def test_gradient_check_uses_all_vocabulary_and_detached_sample(self):
        source = torch.tensor([[9., 8., 7., 6., 12.], [2., 8., 8., 7., 6.]], dtype=torch.bfloat16,
                              requires_grad=True)
        result = check_rival_gradient(source, torch.tensor([0, 1]))
        self.assertEqual(result['vocabulary'], 5)
        self.assertEqual(result['projected_dtype'], 'torch.bfloat16')
        self.assertIsNone(source.grad)

    def test_failing_work_stays_durable_and_is_not_completed(self):
        with tempfile.TemporaryDirectory() as directory:
            log = WorkLog(directory)
            with self.assertRaisesRegex(AssertionError, 'original failure'):
                with log.work('temporary_optimizer_step'):
                    raise AssertionError('original failure')
            rows = [json.loads(line) for line in (Path(directory) / 'WORK.jsonl').read_text().splitlines()]
            self.assertEqual([r['status'] for r in rows], ['started', 'failed'])
            self.assertEqual(log.counts['temporary_optimizer_step_attempted'], 1)
            self.assertEqual(log.counts['temporary_optimizer_step_completed'], 0)

    def test_declared_budget_is_18_forwards_5_steps_no_generations(self):
        self.assertEqual(sum(PLAN[key] for key in ('current_training_forwards', 'reference_training_forwards',
                                                  'detached_comparison_forwards')), 18)
        self.assertEqual((PLAN['temporary_optimizer_steps'], PLAN['generations'], PLAN['teacher_model_forwards']),
                         (5, 0, 0))

    def test_observer_checks_existing_reference_forward_without_extra_forward(self):
        with tempfile.TemporaryDirectory() as directory:
            model = TargetObjectiveModel(tiny_author(), 'osd_uniform')
            model.begin_event()
            log = WorkLog(directory)
            log.presentation = 0
            observer = RankingObserver(model, log)
            with observer.installed():
                loss, _, _ = model(**osd_batch(model))
                loss.backward()
                observer.check_saved_gradients()
            self.assertEqual(log.counts['current_training_forward_completed'], 1)
            self.assertEqual(log.counts['reference_training_forward_completed'], 1)
            self.assertEqual(len(observer.prefix_checks), 1)
            self.assertEqual(len(observer.gradient_checks), 1)
            self.assertFalse(observer.gradient_samples)
            model.end_event()


class QualificationWindowTests(unittest.TestCase):
    setUpClass = classmethod(legacy.OSDRecipeTests.setUpClass.__func__)
    tearDownClass = classmethod(legacy.OSDRecipeTests.tearDownClass.__func__)
    trainer = legacy.OSDRecipeTests.trainer
    saved = legacy.OSDRecipeTests.saved

    def setUp(self):
        legacy.OSDRecipeTests.setUp(self)
        patch = mock.patch.object(T, 'ranking_eos_token_ids', return_value=(28,))
        patch.start()
        self.addCleanup(patch.stop)

    def test_disposable_window_consumes_full_pool_order_and_stops_after_two_rows(self):
        full = self.trainer('ordinary', objective='osd').train_event('e1', self.features)
        trainer = self.trainer('window', objective='osd')
        trainer.bundle.model.begin_event()
        log = WorkLog(self.root / 'log')
        with mock.patch('torch.randperm', wraps=torch.randperm) as draw:
            rows = list(run_window(trainer, full, log))
        self.assertEqual(draw.call_args_list[0].args, (8,))
        self.assertEqual([r['record_id'] for r in rows], [r['record_id'] for r in full['records'][:2]])
        self.assertEqual([r['sampled_anchors'] for r in rows], [r['sampled_anchors'] for r in full['records'][:2]])
        self.assertEqual([r['metrics'] for r in rows], [r['metrics'] for r in full['records'][:2]])
        self.assertEqual((trainer.global_step, trainer.presentations), (1, 2))
        self.assertFalse(trainer.latest.exists())
        self.assertEqual(list(trainer.events_root.iterdir()), [])
        trainer.bundle.model.end_event()

    def test_complete_five_trial_control_with_cpu_fixtures_and_no_export(self):
        for index in range(8, 128):
            raw = torch.load(self.features[index % 2], weights_only=True)
            raw['record_id'] = f'response_{index}'
            raw['label_binding']['record_id'] = raw['record_id']
            path = self.root / f'feedback_{index}.pt'
            torch.save(raw, path)
            self.features.append(path)
        ordinary = self.trainer('ordinary_full', objective='osd')
        result = ordinary.train_event('e1', self.features)
        receipt = ordinary.events_root / 'e1/RESULT.json'
        ordinary.close()
        destination = self.root / 'qualification'
        with mock.patch.object(T.SpecForgeEventTrainer, '_export', side_effect=AssertionError('no export')), \
             mock.patch('torch.cuda.reset_peak_memory_stats'), \
             mock.patch('torch.cuda.max_memory_allocated', return_value=0), \
             mock.patch('torch.cuda.max_memory_reserved', return_value=0):
            summary = run_qualification(self.config, receipt, destination)
        self.assertEqual(summary['status'], 'passed')
        self.assertEqual(len(summary['trials']), 5)
        self.assertEqual(summary['work_counts']['temporary_optimizer_step_completed'], 5)
        self.assertEqual(summary['work_counts']['reference_training_forward_completed'], 4)
        self.assertTrue(summary['temporary_states_discarded'])
        self.assertFalse((destination / 'disposable_O_state.pt').exists())
        self.assertEqual(list(destination.rglob('training_state.pt')), [])
        self.assertEqual(json.loads((destination / 'RESULT.json').read_text())['status'], 'passed')

        # A correctness failure must not consume the remaining trials or lose
        # its report, and must never become a serving checkpoint.
        failed = self.root / 'failed_qualification'
        with mock.patch('experiments.prefix_ranking.qualify.check_rival_gradient',
                        side_effect=AssertionError('injected gradient failure')), \
             mock.patch('torch.cuda.reset_peak_memory_stats'), \
             mock.patch('torch.cuda.max_memory_allocated', return_value=0), \
             mock.patch('torch.cuda.max_memory_reserved', return_value=0):
            with self.assertRaisesRegex(AssertionError, 'injected gradient failure'):
                run_qualification(self.config, receipt, failed)
        failure = json.loads((failed / 'RESULT.json').read_text())
        self.assertEqual(failure['status'], 'failed')
        self.assertEqual(failure['trials'][-1]['label'], 'U1')
        self.assertEqual(failure['trials'][-1]['status'], 'failed')
        self.assertEqual(failure['work_counts']['temporary_optimizer_step_completed'], 3)
        self.assertTrue((failed / 'disposable_O_state.pt').exists())
        self.assertEqual(list(failed.rglob('training_state.pt')), [])

    def test_detached_checks_preserve_rng_modes_and_original_anchors(self):
        full = self.trainer('ordinary', objective='osd').train_event('e1', self.features)
        trainer = self.trainer('current', objective='osd_uniform')
        trainer.bundle.model.begin_event()
        model = trainer.bundle.model
        modes = {name: module.training for name, module in model.named_modules()}
        rng = capture_rng()
        by_id = {f['record_id']: f for f in full['features']}
        log = WorkLog(self.root / 'checks')
        compare_detached_blocks(trainer, [by_id[r['record_id']] for r in full['records'][:2]], full['records'][:2], log)
        self.assertEqual(log.counts['detached_comparison_forward_completed'], 4)
        assert_state_equal(capture_rng(), rng)
        self.assertEqual(modes, {name: module.training for name, module in model.named_modules()})
        self.assertEqual((trainer.global_step, trainer.presentations), (0, 0))
        trainer.bundle.model.end_event()


if __name__ == '__main__':
    unittest.main()
