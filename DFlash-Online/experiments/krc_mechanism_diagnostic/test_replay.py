"""Focused CPU checks for independent restore, endpoint and retention gates."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from experiments.ok_continuation.test_resume import event, fixture, runtime
from experiments.krc_events_3_4.test_campaign import correction_report
from experiments.krc_events_3_4.campaign import artifact_identity
from online_feedback.krc_late_continuation import KRCLateContinuationTrainer
from online_feedback.krc_mechanism_replay import KRCMechanismEvent4Trainer, KRCMechanismEvent4Base
from online_feedback.training import SpecForgeEventTrainer
from . import protocol as P
from .state import compare_endpoints, verify_live_endpoint, retire_verified_replay
from .preflight import storage_budget
from .campaign import work_ledger


class ProtocolTests(unittest.TestCase):
    def test_all_fixed_windows_bind_own_anchors_and_cyclic_disjoint_pairs(self):
        for number in (2, 4):
            ordinary = event(number)
            for index, row in enumerate(ordinary['records']):
                row['sampled_anchors'][0]['positions'] = [[1 + index, 3 + index]]
            before = deepcopy(ordinary)
            windows = P.build_window_plan(ordinary, number, evaluation_ids=['evaluation'])
            self.assertEqual(ordinary, before)
            self.assertEqual([row['step'] for row in windows], list(P.WINDOWS))
            for row in windows:
                pair, pass_index = row['batch_index'], row['pass_index']
                for role, offset in (('A', 0), ('R', 1), ('D', 32)):
                    start = pass_index * 128 + ((pair + offset) % 64) * 2
                    self.assertEqual(row[role], [{key: r[key] for key in
                        ('record_id', 'pass', 'sampled_anchors')} for r in ordinary['records'][start:start + 2]])
                self.assertEqual(len({r['record_id'] for role in ('A', 'R', 'D') for r in row[role]}), 6)
            self.assertEqual(P.validate_window_plan(windows, ordinary, number)['candidate_slots'], 40)
            changed = deepcopy(windows)
            changed[-1]['D'][0]['sampled_anchors'][0]['positions'][0][0] += 1
            with self.assertRaisesRegex(ValueError, 'fixed same-pass'):
                P.validate_window_plan(changed, ordinary, number)
            with self.assertRaisesRegex(ValueError, 'Evaluation'):
                P.build_window_plan(ordinary, number, evaluation_ids=['r000'])

    def test_no_event3_or_favorable_window_substitution(self):
        with self.assertRaisesRegex(ValueError, 'Only independent'):
            P.build_window_plan(event(3), 3)
        windows = P.build_window_plan(event(2), 2)
        windows[1] = deepcopy(windows[0])
        with self.assertRaises(ValueError):
            P.validate_window_plan(windows, event(2), 2)

    def test_exact_work_and_isolation_requirements_include_noops(self):
        report = correction_report(2)
        report['mechanism'] = dict(work=dict(pre_capture_response_forwards=40,
            candidate_response_forwards=240, d_gradient_response_forwards=20, d_gradient_backwards=0),
            rows=[dict(step=step, candidate_order=list(P.CANDIDATES),
                restoration_passed=True, k_only_witness_passed=True,
                candidates={name: {'status': 'valid'} for name in P.CANDIDATES}) for step in P.WINDOWS])
        self.assertEqual(P.validate_diagnostic_report(report)['additional_response_forwards'], 300)
        self.assertEqual(P.validate_original_report(report, 2)['core_response_forwards'], 768)
        for mutate in (
            lambda r: r['mechanism']['work'].update(d_gradient_backwards=21),
            lambda r: r['mechanism']['rows'][0].update(k_only_witness_passed=False),
            lambda r: r['mechanism']['rows'].pop(),
        ):
            changed = deepcopy(report)
            mutate(changed)
            with self.assertRaises(ValueError):
                P.validate_diagnostic_report(changed)
        ledger = work_ledger({2: report, 4: report}, {'episodes': {'event_002': {'startup': dict(
            response_evaluations=2, backwards=2, optimizer_steps=0)}, 'event_004': {'startup': dict(
            response_evaluations=2, backwards=2, optimizer_steps=0)}}})
        self.assertEqual((ledger['core_response_forwards'], ledger['ordinary_Adam_steps']), (2136, 256))
        self.assertEqual(ledger['startup_response_forwards'], 4)
        self.assertTrue(ledger['within_budget'])
        invalid = deepcopy(report)
        invalid['mechanism']['rows'][0]['candidates']['R_HALF'] = dict(status='invalid',
            error='Nonfinite dose candidate', restoration_after_failure={'parameters': True, 'optimizer': True})
        invalid['mechanism']['work']['candidate_response_forwards'] -= 6
        check = P.validate_diagnostic_report(invalid)
        self.assertEqual(check['additional_response_forwards'], 294)
        self.assertEqual(len(check['invalid_candidates']), 1)
        invalid['mechanism']['rows'][0]['candidates']['R_HALF']['restoration_after_failure']['optimizer'] = False
        with self.assertRaisesRegex(ValueError, 'exact restoration'):
            P.validate_diagnostic_report(invalid)


class BoundaryTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def initialize(self, number, *, history=None):
        def initialize(trainer, *args, **kwargs):
            trainer.global_step, trainer.presentations = number * 128, number * 256
            trainer.accumulation = 2
            trainer.core = SimpleNamespace(accumulation_remainder=0)
            trainer.events = history if history is not None else [event(i, 'osd_first_error') for i in range(1, number + 1)]
        return patch.object(SpecForgeEventTrainer, '__init__', initialize)

    def test_event4_restores_only384_768_and_old_constructor_stays_krc2_only(self):
        with self.initialize(3):
            trainer = KRCMechanismEvent4Trainer('config', self.root / 'new', resume_state='KRC3')
            self.assertEqual(trainer.global_step, 384)
            with self.assertRaisesRegex(ValueError, 'event-2 boundary'):
                KRCLateContinuationTrainer('config', self.root / 'old', resume_state='KRC3')
        for number in (1, 2, 4):
            with self.initialize(number), self.assertRaisesRegex(ValueError, 'KRC3 boundary'):
                KRCMechanismEvent4Trainer('config', self.root / str(number), resume_state='other')
        with self.initialize(3, history=[event(3, 'osd_first_error')]), self.assertRaisesRegex(ValueError, 'history'):
            KRCMechanismEvent4Trainer('config', self.root / 'history', resume_state='bad')
        with self.assertRaisesRegex(ValueError, 'complete own KRC3'):
            KRCMechanismEvent4Trainer('config', self.root / 'none', resume_state=None)

    def test_event4_cannot_train_event3_or_continue_after4(self):
        with self.initialize(3):
            trainer = KRCMechanismEvent4Trainer('config', self.root / 'guard', resume_state='KRC3')
        for event_id, step in (('event_003', 384), ('event_004', 512), ('event_005', 384)):
            trainer.global_step = step
            with self.assertRaisesRegex(ValueError, 'only event 4'):
                trainer._prepare_event(event_id, ['feature'] * 128, 2)
        with self.assertRaisesRegex(ValueError, 'KRC3 at384|KRC3 at 384'):
            KRCMechanismEvent4Base(SimpleNamespace(global_step=256, presentations=512),
                schedule={}, feature_paths={})

    def test_endpoint_metadata_may_differ_but_scientific_tensors_metrics_rng_may_not(self):
        original = fixture('osd_first_error', number=4)
        original['events'][-1]['records'][0].update(metrics={'loss': 1.25}, training_s=1.0)
        changed = deepcopy(original)
        changed['run_id'] = 'diagnostic'
        changed['events'][-1]['records'][0]['training_s'] = 2.0
        changed['events'][-1]['objective_work'] = {'diagnostic_chunks': 123}
        result = compare_endpoints(changed, original, event_number=4,
            config_text=original['config_text'], expected_cuda_rng_states=0)
        self.assertTrue(result['exact_decoded_scientific_state'])
        self.assertGreater(result['metadata_difference_count'], 0)
        for name, mutate in (
            ('weight', lambda s: s['draft_state_dict'][next(iter(s['draft_state_dict']))].add_(1)),
            ('rng', lambda s: s['rng']['torch'].__setitem__(0, s['rng']['torch'][0] ^ 1)),
            ('metric', lambda s: s['events'][-1]['records'][0]['metrics'].update(loss=1.5)),
            ('anchor', lambda s: s['events'][-1]['records'][0]['sampled_anchors'][0]['positions'][0].__setitem__(0, 2)),
        ):
            bad = deepcopy(changed)
            mutate(bad)
            with self.subTest(name=name), self.assertRaises(ValueError):
                compare_endpoints(bad, original, event_number=4, config_text=original['config_text'],
                    expected_cuda_rng_states=0, receipt_path=self.root / (name + '.json'))

    def test_live_endpoint_rejects_gradient_or_controller_without_reseeding(self):
        state = fixture('osd_first_error', number=2)
        path = self.root / 'state.pt'
        torch.save(state, path)
        trainer = runtime(state)
        trainer._restore(path)
        self.assertTrue(verify_live_endpoint(trainer, state, self.root / 'live.json')['exact_boundary'])
        trainer._krc_continuation_active = True
        with self.assertRaisesRegex(ValueError, 'Live endpoint'):
            verify_live_endpoint(trainer, state, self.root / 'controller.json')
        trainer._krc_continuation_active = False
        next(trainer.bundle.draft_model.parameters()).grad = torch.ones((2, 2), dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, 'Live endpoint'):
            verify_live_endpoint(trainer, state, self.root / 'grad.json')


class RetentionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.event_root = self.root / 'campaign_001/event_002'
        self.new_state = self.event_root / 'replay/training_state.pt'
        self.new_export = self.event_root / 'replay/events/event_002/export'
        self.original_state = self.root / 'original/training_state.pt'
        self.original_export = self.root / 'original/export'
        for path in (self.new_export, self.original_export):
            path.mkdir(parents=True)
            (path / 'model.safetensors').write_bytes(b'the same qualified weight tensor')
            (path / 'config.json').write_text('{}')
        self.new_state.write_bytes(b'new serialization')
        self.original_state.write_bytes(b'original serialization')
        self.item = dict(event_number=2, endpoint_state=str(self.original_state),
                         endpoint_checkpoint=str(self.original_export))
        self.receipt = dict(full_state_path=str(self.new_state), checkpoint_path=str(self.new_export))
        self.sealed = {str(path): {key: artifact_identity(path)[key] for key in ('bytes', 'sha256')}
            for path in (self.original_state, *self.original_export.iterdir())}
        for name in ('ENDPOINT_EQUALITY.json', 'SAVED_STATE_CHECK.json', 'ORIGINAL_ENDPOINT_CHECK.json',
            'LIVE_ENDPOINT.json', 'TRAINING_CHECK.json', 'CORRECTION_EVIDENCE.json', 'REPLAY_STATE_IDENTITIES.json'):
            (self.event_root / name).write_text('{}')

    def test_only_verified_regenerated_tensors_removed_with_exact_original_mapping(self):
        state = fixture('osd_first_error', number=2)
        result = retire_verified_replay(self.event_root, self.receipt, self.item, state,
            dict(status='passed', exact_decoded_scientific_state=True), self.sealed)
        self.assertEqual(result['status'], 'completed')
        self.assertFalse(self.new_state.exists())
        self.assertFalse((self.new_export / 'model.safetensors').exists())
        self.assertTrue((self.new_export / 'config.json').exists())
        self.assertTrue(self.original_state.exists())
        self.assertEqual(len(result['removed_paths']), 2)
        self.assertTrue((self.event_root / 'DECODED_REPLAY_METADATA.json').is_file())

    def test_missing_evidence_or_changed_original_prevents_retirement(self):
        (self.original_export / 'model.safetensors').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Original endpoint changed'):
            retire_verified_replay(self.event_root, self.receipt, self.item,
                fixture('osd_first_error', number=2),
                dict(status='passed', exact_decoded_scientific_state=True), self.sealed)
        self.assertTrue(self.new_state.exists())

    def test_peak_admission_includes_temporary_export_and_fixed_reserve(self):
        with patch('shutil.disk_usage', return_value=SimpleNamespace(free=31_590_000_000)):
            budget = storage_budget(self.root, 2_097_259_104, available_host_bytes=139_000_000_000)
        self.assertTrue(budget['admitted'])
        self.assertEqual(budget['tensor_peak_bytes'], int(5 * 2_097_259_104 * 1.10))
        self.assertEqual(budget['reserve_bytes'], 8 * 2**30)
        with patch('shutil.disk_usage', return_value=SimpleNamespace(free=budget['required_bytes'] - 1)):
            self.assertFalse(storage_budget(self.root, 2_097_259_104,
                available_host_bytes=139_000_000_000)['admitted'])


if __name__ == '__main__':
    unittest.main()
