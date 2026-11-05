"""CPU checks of separate learning/reference pairs and exact replay isolation."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import torch

from online_feedback import training as T
from online_feedback.reference_correction_event import ReferenceCorrectionEvent, ReferencePrefixCorrection
from online_feedback.rng import capture_rng
from test_prefix_correction_event import WindowTrainer
from test_training_contract import trees_equal
import test_osd_recipe as legacy


def small_schedule(records, pairs_per_pass):
    windows = []
    for offset in range(0, len(records), pairs_per_pass * 2):
        pairs = [records[offset + 2*index:offset + 2*index+2] for index in range(pairs_per_pass)]
        for index, learning in enumerate(pairs):
            reference = pairs[(index + 1) % pairs_per_pass]
            windows.append({
                'step': len(windows) + 1, 'pass_index': learning[0]['pass'],
                'batch_index': index, 'reference_batch_index': (index + 1) % pairs_per_pass,
                'learning': deepcopy(learning), 'reference': deepcopy(reference),
                'learning_record_ids': [row['record_id'] for row in learning],
                'reference_record_ids': [row['record_id'] for row in reference],
            })
    return {'windows': windows}


class ReferenceWindowTrainer(WindowTrainer):
    """Four fixed records: A labels 1 and R labels 0 on the same small features."""

    def __init__(self, root):
        super().__init__()
        self.feature_paths = {}
        records = []
        for index in range(4):
            raw = dict(record_id=f'response_{index}',
                       input_ids=torch.full((1, 3), 1 if index < 2 else 0, dtype=torch.long),
                       hidden_states=torch.tensor([[[1., 0.], [1., 0.], [1., 0.]]], dtype=torch.bfloat16),
                       loss_mask=torch.ones(1, 3))
            path = Path(root) / f'{index}.pt'
            torch.save(raw, path)
            self.feature_paths[raw['record_id']] = path
            records.append(dict(record_id=raw['record_id'], pass_index=0,
                                sampled_anchors=[dict(positions=[[0]], keep_mask=[[True]])]))
            records[-1]['pass'] = 0
        self.schedule = small_schedule(records, pairs_per_pass=2)
        self.bundle.model._block_labels_and_weights = lambda inputs, mask, anchors, keep: (
            inputs.unsqueeze(1), torch.tensor([[[0., 1., 1.]]]), torch.tensor([[[0, 1, 2]]]))

    def _batch(self, raw):
        return SimpleNamespace(tensors=raw)

    def present(self, ordinal):
        raw = torch.load(self.feature_paths[f'response_{ordinal}'], weights_only=True)
        batch = SimpleNamespace(sample_ids=[raw['record_id']], inputs=raw['input_ids'],
                                labels=raw['input_ids'].unsqueeze(1), features=raw['hidden_states'],
                                loss_scale=1. if ordinal < 2 else 0.)
        result = self.core.train_step(batch)
        self.presentations += 1
        self.global_step += int(result.optimizer_stepped)
        return result


class ReferenceWindowTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.trainer = ReferenceWindowTrainer(self.directory.name)

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def test_reference_denominator_not_learning_and_consecutive_mask_refresh(self):
        gated = []
        with ReferenceCorrectionEvent(self.trainer, schedule=self.trainer.schedule,
                                      feature_paths=self.trainer.feature_paths,
                                      probe_steps=(1, 2), first_window_check=gated.append) as session:
            for index in range(4):
                self.trainer.present(index)
                if index % 2:
                    self.assertIsNone(session.active)
                    self.assertFalse(hasattr(self.trainer.bundle.model, 'prefix_guard'))
                    self.assertFalse(self.trainer.bundle.model.lm_head._forward_pre_hooks)
        report = session.report()
        self.assertEqual([row['correction']['status'] for row in report['steps']],
                         ['applied', 'empty_protection'])
        self.assertEqual(report['probes'][0]['reference']['post_k']['denominator'], 6.)
        self.assertEqual(report['probes'][0]['learning']['post_k']['denominator'], 0.)
        self.assertEqual(report['probes'][1]['reference']['post_k']['denominator'], 0.)
        self.assertEqual([row['record_id'] for row in gated[0]['responses']], ['response_0', 'response_1'])
        self.assertEqual(len(gated), 1)
        self.assertEqual(report['counts']['Adam_completed'], 2)
        self.assertEqual(report['counts']['K_backward_completed'], 4)
        for name in ('original_response_forwards', 'pre_reference_response_forwards',
                     'post_k_reference_response_forwards', 'post_c_reference_response_forwards',
                     'post_k_learning_response_forwards', 'post_c_learning_response_forwards'):
            self.assertEqual(report['work'][name], 4, name)
        self.assertEqual(report['work']['violation_backward_completed'], 2)
        self.assertEqual(sum(count for name, count in report['forward_calls'].items()
                             if name.endswith('/draft')), 24)
        for probe in report['probes']:
            self.assertEqual(probe['learning']['work']['violation_backward_completed'], 0)
            self.assertEqual(probe['learning']['work']['direct_corrections'], 0)
            self.assertNotIn('_replay_rng', str(probe['reference']['responses']))
        self.assertEqual(report['arm'], 'KRC')
        self.assertEqual((self.trainer.global_step, self.trainer.presentations), (2, 4))
        for state in self.trainer.backend.optimizer.optimizer.state.values():
            self.assertEqual(float(state['step']), 2.)
        self.assertIsNone(session.trainer)
        session.close()

    def test_nonprobe_window_has_no_learning_diagnostic_or_corrected_forward(self):
        with ReferenceCorrectionEvent(self.trainer, schedule=self.trainer.schedule,
                                      feature_paths=self.trainer.feature_paths, probe_steps=()) as session:
            for index in range(4):
                self.trainer.present(index)
        work = session.report()['work']
        self.assertEqual(work['original_response_forwards'], 4)
        self.assertEqual(work['pre_reference_response_forwards'], 4)
        self.assertEqual(work['post_k_reference_response_forwards'], 4)
        self.assertEqual(work['post_c_reference_response_forwards'], 0)
        self.assertEqual(work['post_k_learning_response_forwards'], 0)
        self.assertEqual(work['post_c_learning_response_forwards'], 0)
        self.assertEqual(session.report()['probes'], [])

    def test_reference_replays_same_random_draws_and_restores_current_rng(self):
        model = self.trainer.bundle.model
        original_forward = model._forward_draft_blocks
        def stochastic(*args):
            anchors, keep, hidden = original_forward(*args)
            return anchors, keep, hidden + torch.randn_like(hidden) * 3
        model._forward_draft_blocks = stochastic
        binding = self.trainer.schedule['windows'][0]['reference']
        recorder = ReferencePrefixCorrection(self.trainer, record_ids=('response_2', 'response_3'))
        before = capture_rng()
        recorder.capture_reference(binding, self.trainer.feature_paths)
        self.assertTrue(trees_equal(before, capture_rng()))
        original = [row['original_proposals'].clone() for row in recorder.responses]
        torch.rand(21)  # Stand in for progressed ordinary K sampling.
        progressed = capture_rng()
        post = recorder.measure_post_k()
        for expected, row in zip(original, post['responses']):
            self.assertEqual(expected.tolist(), row['proposals'])
        self.assertEqual(post['V'], 0.)
        self.assertTrue(trees_equal(progressed, capture_rng()))
        recorder.apply_once()
        corrected = recorder.measure_post_c()
        for expected, row in zip(original, corrected['responses']):
            self.assertEqual(expected.tolist(), row['proposals'])
        self.assertTrue(trees_equal(progressed, capture_rng()))
        recorder.close()
        self.assertFalse(any('_replay_rng' in row for row in recorder.responses))

    def test_learning_identity_failure_prevents_optimizer_and_preserves_evidence(self):
        session = ReferenceCorrectionEvent(self.trainer, schedule=self.trainer.schedule,
                                           feature_paths=self.trainer.feature_paths, probe_steps=())
        with self.assertRaisesRegex(ValueError, 'Learning identities'):
            with session:
                self.trainer.present(1)
        self.assertEqual(session.counts['Adam_completed'], 0)
        self.assertEqual(session.work['pre_reference_response_forwards'], 2)
        self.assertEqual(session.work['original_response_forwards'], 0)
        self.assertEqual(session.status, 'failed')

    def test_first_learning_gate_stops_before_reference_correction(self):
        def mismatch(report):
            raise ValueError('A reproduction mismatch')
        session = ReferenceCorrectionEvent(self.trainer, schedule=self.trainer.schedule,
                                           feature_paths=self.trainer.feature_paths, probe_steps=(1,),
                                           first_window_check=mismatch)
        with self.assertRaisesRegex(ValueError, 'A reproduction mismatch'):
            with session:
                self.trainer.present(0)
                self.trainer.present(1)
        self.assertEqual(session.counts['Adam_completed'], 1)
        self.assertEqual(session.work['direct_corrections'], 0)
        self.assertEqual(session.work['post_c_reference_response_forwards'], 0)
        self.assertFalse(hasattr(self.trainer.bundle.model, 'prefix_guard'))


class NativeReferenceIntegrationTests(unittest.TestCase):
    setUpClass = classmethod(legacy.OSDRecipeTests.setUpClass.__func__)
    tearDownClass = classmethod(legacy.OSDRecipeTests.tearDownClass.__func__)
    trainer = legacy.OSDRecipeTests.trainer
    saved = legacy.OSDRecipeTests.saved

    def setUp(self):
        legacy.OSDRecipeTests.setUp(self)
        patch = mock.patch.object(T, 'ranking_eos_token_ids', return_value=(28,))
        patch.start()
        self.addCleanup(patch.stop)

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def test_real_two_pass_event_keeps_sampling_anchors_and_first_native_adam_exact(self):
        first_states = {}
        def capture_first(trainer, name):
            original = trainer.backend.optimizer.optimizer.step
            def step(*args, **kwargs):
                result = original(*args, **kwargs)
                if name not in first_states:
                    first_states[name] = deepcopy((trainer.bundle.model.state_dict(),
                                                  trainer.backend.optimizer.state_dict(), capture_rng()))
                return result
            trainer.backend.optimizer.optimizer.step = step
        ordinary = self.trainer('K', objective='osd_first_error')
        capture_first(ordinary, 'K')
        receipt = ordinary.train_event('e1', self.features[:4], passes=2)
        schedule = small_schedule(receipt['records'], pairs_per_pass=2)
        paths = {row['record_id']: Path(row['path']) for row in receipt['features']}
        observed = self.trainer('KRC', objective='osd_first_error')
        capture_first(observed, 'KRC')
        with ReferenceCorrectionEvent(observed, schedule=schedule, feature_paths=paths,
                                      probe_steps=(1, 4)) as session:
            actual = observed.train_event('e1', self.features[:4], passes=2,
                                          presentation_schedule=receipt)
        self.assertTrue(trees_equal(first_states['K'], first_states['KRC']))
        self.assertEqual((actual['optimizer_steps'], actual['response_presentations']), (4, 8))
        for key in ('record_id', 'pass', 'sampled_anchors'):
            self.assertEqual([row[key] for row in receipt['records']], [row[key] for row in actual['records']])
        self.assertEqual(session.counts['Adam_completed'], 4)
        self.assertEqual(session.work['original_response_forwards'], 8)
        self.assertEqual(session.work['pre_reference_response_forwards'], 8)
        self.assertEqual(session.work['post_k_reference_response_forwards'], 8)
        self.assertEqual(session.work['post_c_reference_response_forwards'], 4)
        self.assertEqual(session.work['post_k_learning_response_forwards'], 4)
        self.assertEqual(session.work['post_c_learning_response_forwards'], 4)
        state = self.saved(observed)
        self.assertEqual(state['algorithm_contract']['reference_correction_event']['arm'], 'KRC')
        self.assertNotIn('prefix_correction_event', state['algorithm_contract'])
        self.assertEqual((state['global_step'], state['presentations']), (4, 8))


if __name__ == '__main__':
    unittest.main()
