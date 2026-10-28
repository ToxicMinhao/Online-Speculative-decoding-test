"""Real tiny-BF16 trainer tests for correction-disabled continuation.

The saved fixture is produced by the original KRC controller and serializer.
Production event limits are reduced to four steps only inside these CPU tests.
"""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest import mock

import torch

from experiments.ok_continuation.resume import verify_live_resume
from online_feedback import krc_continuation as C
from online_feedback.krc_off import KRCOffTrainer, ZERO_REFERENCE_WORK, off_contract
from online_feedback.reference_correction_event import ReferenceCorrectionEvent
from online_feedback.rng import capture_rng
from online_feedback import training as T
from test_reference_correction_event import small_schedule
from test_training_contract import trees_equal
import test_osd_recipe as legacy


class OffContinuationTests(unittest.TestCase):
    setUpClass = classmethod(legacy.OSDRecipeTests.setUpClass.__func__)
    tearDownClass = classmethod(legacy.OSDRecipeTests.tearDownClass.__func__)
    trainer = legacy.OSDRecipeTests.trainer
    saved = legacy.OSDRecipeTests.saved

    def setUp(self):
        legacy.OSDRecipeTests.setUp(self)
        for patch in (mock.patch.object(T, 'ranking_eos_token_ids', return_value=(28,)),
                      mock.patch.object(C, 'RESUME_STEP', 4),
                      mock.patch.object(C, 'EVENT_WINDOWS', 4)):
            patch.start()
            self.addCleanup(patch.stop)

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def prepare(self):
        ordinary = self.trainer('ordinary', objective='osd_first_error')
        first = ordinary.train_event('event_001', self.features[:4])
        second = ordinary.train_event('event_002', self.features[4:])
        source = self.trainer('KRC1', objective='osd_first_error')
        schedule = small_schedule(first['records'], 2)
        paths = {row['record_id']: Path(row['path']) for row in first['features']}
        with ReferenceCorrectionEvent(source, schedule=schedule, feature_paths=paths,
                                      probe_steps=(1, 64, 128)):
            source.train_event('event_001', self.features[:4], presentation_schedule=first)
        source_path = self.root / 'KRC1_read_only.pt'
        source_path.write_bytes(source.latest.read_bytes())
        source_state = self.saved(source)
        self.assertNotIn('status', source_state['events'][0])
        # Continue the actual saved KRC1 with the unmodified plain K trainer.
        # This is an uninterrupted CPU reference, not another production arm.
        expected = source.train_event('event_002', self.features[4:], presentation_schedule=second)
        expected_state = self.saved(source)
        return source_path, source_state, second, expected, expected_state

    def resume(self, path, name='OFF'):
        resumed = KRCOffTrainer(self.config, self.root / name, resume_state=path)
        self.addCleanup(resumed.close)
        return resumed

    def enable(self, trainer, state, reference, name='off_checks'):
        receipt = verify_live_resume(trainer, state)
        trainer.enable_off_continuation(resume_check=receipt,
                                       first_A_records=reference['records'][:2],
                                       output_dir=self.root / name)
        return receipt

    def test_off_is_exact_plain_K_from_same_saved_state_with_zero_reference_work(self):
        path, initial, schedule, expected, expected_state = self.prepare()
        resumed = self.resume(path)
        before_rng = capture_rng()
        before_contract = deepcopy(resumed.contract)
        receipt = self.enable(resumed, initial, expected)
        self.assertTrue(receipt['exact_resume'])
        self.assertTrue(trees_equal(before_rng, capture_rng()))
        contract = deepcopy(before_contract)
        contract['reference_correction_event'] = off_contract()
        self.assertEqual(resumed.contract, contract)
        optimizer = resumed.backend.optimizer.optimizer
        with mock.patch.object(ReferenceCorrectionEvent, '__init__',
                               side_effect=AssertionError('OFF cannot create a reference controller')):
            actual = resumed.train_event('event_002', self.features[4:], presentation_schedule=schedule)
        actual_state = self.saved(resumed)
        for field in ('draft_state_dict', 'backend', 'rng', 'global_step', 'presentations'):
            self.assertTrue(trees_equal(expected_state[field], actual_state[field]), field)
        for field in ('record_id', 'pass', 'sampled_anchors', 'optimizer_step_after', 'metrics'):
            self.assertEqual([row[field] for row in actual['records']],
                             [row[field] for row in expected['records']], field)
        report = resumed.off_report()
        self.assertEqual(report['status'], 'completed')
        self.assertEqual(report['first_A']['status'], 'passed')
        self.assertEqual(report['zero_reference_work'], ZERO_REFERENCE_WORK)
        self.assertEqual(report['work'], {
            'response_forwards': 8, 'K_backwards': 8, 'Adam_steps': 4, 'presentations': 8,
            'reference_forwards': 0, 'reference_backwards': 0, 'protected_prefix_captures': 0,
            'direct_corrections': 0, 'probe_forwards': 0,
        })
        self.assertEqual(report['first_A']['extra_forwards'], 0)
        self.assertEqual(report['first_A']['extra_optimizer_steps'], 0)
        self.assertIs(resumed.backend.optimizer.optimizer, optimizer)
        self.assertEqual([float(value['step']) for value in optimizer.state.values()],
                         [8.] * len(optimizer.state))
        self.assertEqual(actual['training_variant'], 'KRC2_OFF')
        self.assertEqual(actual_state['events'][0], initial['events'][0])
        self.assertFalse(hasattr(resumed.bundle.model, 'prefix_guard'))
        self.assertIsNone(resumed.bundle.model.reference)
        self.assertFalse(resumed.bundle.model._forward_hooks)
        self.assertFalse(resumed.bundle.model.lm_head._forward_pre_hooks)
        self.assertTrue(trees_equal(initial, torch.load(path, weights_only=False)))

    def test_migration_requires_successful_exact_restore_and_can_only_happen_once(self):
        path, state, _, reference, _ = self.prepare()
        resumed = self.resume(path)
        original = deepcopy(resumed.contract)
        with self.assertRaisesRegex(ValueError, 'Exact live KRC1 restoration'):
            resumed.enable_off_continuation(resume_check={'status': 'failed'},
                                           first_A_records=reference['records'][:2],
                                           output_dir=self.root / 'bad')
        self.assertEqual(resumed.contract, original)
        with self.assertRaisesRegex(ValueError, 'Install the KRC'):
            resumed.train_event('event_002', self.features[4:])
        receipt = self.enable(resumed, state, reference)
        with self.assertRaisesRegex(ValueError, 'unchanged restored KRC1 boundary'):
            resumed.enable_off_continuation(resume_check=receipt,
                                           first_A_records=reference['records'][:2],
                                           output_dir=self.root / 'again')

    def test_wrong_first_A_metric_fails_after_only_actual_first_step_and_preserves_evidence(self):
        path, state, schedule, reference, _ = self.prepare()
        reference = deepcopy(reference)
        reference['records'][0]['metrics']['loss'] += 1
        resumed = self.resume(path)
        self.enable(resumed, state, reference)
        with self.assertRaisesRegex(ValueError, 'first-A check failed'):
            resumed.train_event('event_002', self.features[4:], presentation_schedule=schedule)
        report = resumed.off_report()
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['work']['Adam_steps'], 1)
        self.assertEqual(report['work']['response_forwards'], 2)
        gate = json.loads((self.root / 'off_checks/FIRST_A.json').read_text())
        self.assertEqual(gate['status'], 'failed')
        self.assertEqual(gate['differences'], ['presentation_1.metrics'])
        self.assertFalse(resumed.latest.exists())
        self.assertFalse(resumed.bundle.model._forward_hooks)
        self.assertFalse(resumed.bundle.model.lm_head._forward_pre_hooks)

    def test_wrong_anchor_draw_fails_without_forcing_draw_or_first_update(self):
        path, state, schedule, reference, _ = self.prepare()
        schedule = deepcopy(schedule)
        schedule['records'][0]['sampled_anchors'][0]['positions'][0][0] += 1
        resumed = self.resume(path)
        self.enable(resumed, state, reference)
        with self.assertRaisesRegex(ValueError, 'Sampled anchors differ'):
            resumed.train_event('event_002', self.features[4:], presentation_schedule=schedule)
        report = resumed.off_report()
        self.assertEqual(report['work']['Adam_steps'], 0)
        self.assertEqual(report['work']['K_backwards'], 0)
        self.assertEqual(resumed.global_step, 4)
        self.assertTrue(trees_equal(state['draft_state_dict'], dict(resumed.bundle.draft_model.state_dict())))

    def test_nonfinite_first_A_evidence_is_preserved_without_masking_failure(self):
        path, state, schedule, reference, _ = self.prepare()
        reference = deepcopy(reference)
        reference['records'][0]['metrics']['loss'] = float('nan')
        resumed = self.resume(path)
        self.enable(resumed, state, reference)
        with self.assertRaisesRegex(ValueError, 'first-A check failed'):
            resumed.train_event('event_002', self.features[4:], presentation_schedule=schedule)
        gate = json.loads((self.root / 'off_checks/FIRST_A.json').read_text())
        self.assertEqual(gate['retained_records'][0]['metrics']['loss'], 'nan')
        self.assertEqual(gate['status'], 'failed')
        work = json.loads((self.root / 'off_checks/WORK.json').read_text())
        self.assertEqual(work['status'], 'failed')
        self.assertEqual(work['work']['Adam_steps'], 1)

    def test_wrong_saved_contract_missing_state_and_unbound_schedule_are_rejected(self):
        path, state, _, reference, _ = self.prepare()
        changed = deepcopy(state)
        changed['algorithm_contract']['reference_correction_event'] = off_contract()
        changed_path = self.root / 'already_OFF.pt'
        torch.save(changed, changed_path)
        with self.assertRaisesRegex(ValueError, 'Resume algorithm'):
            self.resume(changed_path, 'bad')
        with self.assertRaisesRegex(ValueError, 'complete recovery state'):
            self.resume(None, 'missing')
        resumed = self.resume(path)
        self.enable(resumed, state, reference)
        with self.assertRaisesRegex(ValueError, 'retained event-2 order'):
            resumed.train_event('event_002', self.features[4:])
        self.assertEqual(resumed.off_report()['work']['Adam_steps'], 0)


if __name__ == '__main__':
    unittest.main()
