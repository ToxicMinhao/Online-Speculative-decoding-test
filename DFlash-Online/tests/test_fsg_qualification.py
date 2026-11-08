"""Focused CPU checks for disposable FSG qualification and failure accounting."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from experiments.functional_step_geometry import prepare_feedback, qualify
from experiments.functional_step_geometry.work import LIMITS, QualificationWork
from test_functional_geometry_runtime import TinyBlocks


class QualificationWorkTests(unittest.TestCase):
    def test_failed_attempt_remains_in_budget_after_ledger_reload(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'work.jsonl'
            work = QualificationWork(path)
            with self.assertRaisesRegex(ValueError, 'fixture failure'):
                with work.operation('additional_forward'):
                    raise ValueError('fixture failure')
            restored = QualificationWork(path)
            self.assertEqual(restored.summary()['additional_forward_attempted'], 1)
            self.assertEqual(restored.summary()['additional_forward_failed'], 1)
            self.assertNotIn('additional_forward_completed', restored.summary())

    def test_hook_callback_maps_one_native_step_once_and_enforces_limit(self):
        with tempfile.TemporaryDirectory() as temporary:
            work = QualificationWork(Path(temporary) / 'work.jsonl')
            for _ in range(LIMITS['adam']):
                work.callback('optimizer_steps', 'attempted')
                work.callback('optimizer_steps', 'completed')
            with self.assertRaisesRegex(RuntimeError, 'ceiling'):
                work.callback('optimizer_steps', 'attempted')
            self.assertEqual(work.summary()['adam_attempted'], 8)
            self.assertEqual(work.summary()['adam_completed'], 8)
            work.callback('additional_response_forwards', 'attempted')
            work.callback('additional_response_forwards', 'completed')
            work.callback('additional_backwards', 'attempted')
            work.callback('additional_backwards', 'completed')
            self.assertEqual(work.summary()['additional_forward_completed'], 1)
            self.assertEqual(work.summary()['additional_backward_completed'], 1)
            self.assertNotIn('optimizer_steps_attempted', work.summary())

    def test_four_cases_per_dataset_respect_declared_adam_budget(self):
        self.assertEqual(qualify.CASES,
                         (('KPLUS', None), ('FSG_ZERO', 0.), ('FSG', 1.), ('SCALAR', 1.)))
        self.assertEqual(2 * len(qualify.CASES), LIMITS['adam'])
        self.assertEqual(2 * 2 * len(qualify.CASES), LIMITS['ordinary_forward'])


class RuntimeRestorationTests(unittest.TestCase):
    def test_restoration_recovers_modes_values_and_removes_new_runtime_fields(self):
        model = TinyBlocks()
        original_buffer = model.replay_count
        snapshot = qualify.runtime_state(model)
        with torch.no_grad():
            model.replay_count.add_(3)
        model.register_buffer('unexpected_buffer', torch.ones(2), persistent=False)
        model.max_seq_len_cached = 900
        model._cached_seq_len = 100
        model.eval()
        qualify.restore_runtime(model, snapshot)
        self.assertEqual(int(model.replay_count), 0)
        self.assertIs(model.replay_count, original_buffer)
        self.assertFalse(hasattr(model, 'unexpected_buffer'))
        self.assertFalse(hasattr(model, 'max_seq_len_cached'))
        self.assertIsNone(model._cached_seq_len)
        self.assertTrue(model.training)

    def test_export_uses_explicit_disposable_payload_without_pending_state(self):
        from safetensors.torch import save_file
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / 'config'
            config.mkdir()
            checkpoint = root / 'disposable.pt'
            weights = {'weight': torch.ones(2, dtype=torch.bfloat16)}
            payload = dict(weights=weights, draft_state_dict=weights)
            torch.save(payload, checkpoint)
            trainer = SimpleNamespace(cfg=SimpleNamespace(model=SimpleNamespace(
                draft_model_config=str(config), target_model_path='target',
                embedding_key='model.embed_tokens.weight')))
            def write_export(_checkpoint, _config, destination, **kwargs):
                Path(destination).mkdir()
                save_file(weights, str(Path(destination) / 'model.safetensors'))
            with mock.patch('specforge.export.to_hf.export_to_hf', side_effect=write_export) as export:
                receipt = qualify.export_disposable_state(trainer, checkpoint, root / 'export', payload)
            export.assert_called_once_with(str(checkpoint), str(config / 'config.json'),
                str(root / 'export'), embedding_source='target', embedding_key='model.embed_tokens.weight')
            decoded = torch.load(checkpoint, weights_only=False)
            self.assertIs(decoded['weights'], decoded['draft_state_dict'])
            self.assertTrue(receipt['exact_tensor_values_and_membership'])
            self.assertIn('model.safetensors', receipt['files'])
            self.assertGreaterEqual(receipt['export_s'], 0)
            self.assertGreaterEqual(receipt['verification_s'], 0)

    def test_disposable_export_mismatch_fails_and_keeps_decoded_checkpoint(self):
        from safetensors.torch import save_file
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / 'disposable.pt'
            payload = dict(draft_state_dict={'weight': torch.ones(2, dtype=torch.bfloat16)})
            torch.save(payload, checkpoint)
            trainer = SimpleNamespace(cfg=SimpleNamespace(model=SimpleNamespace(
                draft_model_config='config.json', target_model_path='target',
                embedding_key='model.embed_tokens.weight')))
            def write_wrong_export(_checkpoint, _config, destination, **kwargs):
                Path(destination).mkdir()
                save_file({'weight': torch.zeros(2, dtype=torch.bfloat16)},
                          str(Path(destination) / 'model.safetensors'))
            with mock.patch('specforge.export.to_hf.export_to_hf', side_effect=write_wrong_export):
                with self.assertRaisesRegex(ValueError, 'differs from exact export'):
                    qualify.export_disposable_state(trainer, checkpoint, root / 'export', payload)
            self.assertTrue(checkpoint.is_file())
            self.assertFalse((root / 'EXPORT_EXACT.json').exists())


def failing_case_trainer():
    optimizer = SimpleNamespace(step=lambda: torch.tensor(1.))
    return SimpleNamespace(dataset='gsm8k',
        backend=SimpleNamespace(optimizer=optimizer, backward=lambda *_args, **_kwargs: None),
        author_strategy=SimpleNamespace(forward_loss=lambda *_args, **_kwargs: None),
        bundle=SimpleNamespace(model=SimpleNamespace(begin_event=lambda: None)),
        core=SimpleNamespace(train_step=mock.Mock(side_effect=ValueError('primary model failure'))),
        _anchor_observations=[], _batch=lambda raw: raw,
        global_step=512, presentations=1024, horizon=7424)


class FailurePreservationTests(unittest.TestCase):
    def test_completed_forward_stays_completed_when_following_training_work_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            feature = root / 'feature.pt'
            torch.save({'record_id': 'fixture'}, feature)
            trainer = failing_case_trainer()
            def after_forward(batch, context):
                trainer.author_strategy.forward_loss(batch, context)
                raise ValueError('failure after forward')
            trainer.core.train_step = after_forward
            work = QualificationWork(root / 'work.jsonl')
            with mock.patch.object(qualify, 'restore_origin'), \
                 mock.patch.object(qualify, 'set_observation_directory'):
                with self.assertRaisesRegex(ValueError, 'failure after forward'):
                    qualify.run_case(trainer, [feature], 'KPLUS', None,
                        work, {}, root / 'case', None)
            self.assertEqual(work.summary()['ordinary_forward_attempted'], 1)
            self.assertEqual(work.summary()['ordinary_forward_completed'], 1)
            self.assertNotIn('ordinary_forward_failed', work.summary())

    def test_case_report_failure_does_not_replace_original_model_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            feature = root / 'feature.pt'
            torch.save({'record_id': 'fixture'}, feature)
            trainer = failing_case_trainer()
            with mock.patch.object(qualify, 'restore_origin'), \
                 mock.patch.object(qualify, 'set_observation_directory'), \
                 mock.patch.object(qualify, 'write_json', side_effect=OSError('report disk failure')):
                with self.assertRaisesRegex(ValueError, 'primary model failure'):
                    qualify.run_case(trainer, [feature], 'KPLUS', None,
                        QualificationWork(root / 'work.jsonl'), {}, root / 'case', None)
            self.assertEqual(trainer.core.train_step.call_count, 1)

    def test_case_hook_cleanup_failure_does_not_replace_original_model_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            feature = root / 'feature.pt'
            torch.save({'record_id': 'fixture'}, feature)
            trainer = failing_case_trainer()
            hook = SimpleNamespace(report=lambda: {}, close=mock.Mock(side_effect=OSError('cleanup failed')))
            trainer.functional_geometry = hook
            with mock.patch.object(qualify, 'restore_origin'), \
                 mock.patch.object(qualify, 'set_observation_directory'), \
                 mock.patch.object(qualify, 'install_functional_geometry', return_value=hook):
                with self.assertRaisesRegex(ValueError, 'primary model failure'):
                    qualify.run_case(trainer, [feature], 'FSG_ZERO', 0.,
                        QualificationWork(root / 'work.jsonl'), {}, root / 'case', None)
            hook.close.assert_called_once()

    def test_feedback_close_failure_preserves_generation_failure_and_attempt_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'sealed').mkdir()
            requests = root / 'requests.json'
            requests.write_text(json.dumps({'train_requests': [
                {'request_id': 'new_512', 'request_index': 512, 'prompt_ids': [1, 2]},
                {'request_id': 'new_513', 'request_index': 513, 'prompt_ids': [3, 4]},
            ]}))
            plan = dict(target='target', worker_python='python', execution_source={'commit': 'fixture'},
                datasets={'gsm8k': dict(origin_export='origin', training_manifest=str(requests))})
            (root / 'sealed/PLAN.json').write_text(json.dumps(plan))
            backend = SimpleNamespace(generate=mock.Mock(side_effect=ValueError('primary generation failure')),
                close=mock.Mock(side_effect=OSError('secondary close failure')))
            with mock.patch.object(prepare_feedback, 'SGLangDFlashBackend', return_value=backend):
                with self.assertRaisesRegex(ValueError, 'primary generation failure'):
                    prepare_feedback.prepare(root)
            report = json.loads((root / 'qualification_feedback/PREPARATION.json').read_text())
            self.assertEqual(report['shared_feedback_generations'], 1)
            self.assertEqual(report['teacher_preparations'], 0)
            self.assertIn('primary generation failure', report['error'])
            backend.generate.assert_called_once()


if __name__ == '__main__':
    unittest.main()
