"""Focused CPU checks for disposable FSG qualification and failure accounting."""
import json
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from experiments.functional_step_geometry import prepare_feedback, qualify
from experiments.functional_step_geometry.work import LIMITS, QualificationWork
from test_functional_geometry_runtime import TinyBlocks
from online_feedback.optimizer import DirectBF16AdamW
from online_feedback.rng import capture_rng, restore_rng
from specforge.training.controller import TrainerCore
from specforge.training.strategies.base import StepContext, StepOutput


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


def replay_fixture(root):
    """Real BF16 Adam and TrainerCore, with a small deterministic CPU forward."""
    model = torch.nn.Module()
    model.draft_model = torch.nn.Linear(2, 2, bias=False).bfloat16()
    model.begin_event = lambda: None
    optimizer = DirectBF16AdamW(model.draft_model)
    for parameter in optimizer.model_params:
        optimizer.optimizer.state[parameter] = dict(step=torch.tensor(512.),
            exp_avg=torch.zeros_like(parameter), exp_avg_sq=torch.zeros_like(parameter))
    trainer = SimpleNamespace(dataset='gsm8k',
        bundle=SimpleNamespace(model=model, draft_model=model.draft_model),
        backend=SimpleNamespace(optimizer=optimizer, backward=lambda loss, **kwargs: loss.backward(),
                                step=lambda: optimizer.step()),
        global_step=512, presentations=1024, horizon=7424,
        response_records=[], window_records=[], work={}, _anchor_observations=[],
        _batch=lambda raw: raw, events_root=root / 'events')
    trainer.events_root.mkdir()

    def forward(raw, _context):
        anchors = torch.randperm(4).tolist()
        trainer._anchor_observations.append({'anchors': anchors})
        loss = model.draft_model(raw['x']).float().square().sum()
        trainer.work['vocabulary_head_calls'] = trainer.work.get('vocabulary_head_calls', 0) + 1
        trainer.response_records.append(dict(record_id=raw['record_id'],
            event_id=trainer._active_event, loss=float(loss.detach()),
            sampled_anchors=deepcopy(trainer._anchor_observations)))
        return StepOutput(loss=loss, metrics={}, ratio_metrics={},
                          sum_metrics={'response_presentations': torch.tensor(1.)})

    trainer.author_strategy = SimpleNamespace(forward_loss=forward)
    trainer.strategy = trainer.author_strategy
    trainer.core = TrainerCore(trainer.strategy, trainer.backend, accumulation_steps=2)
    trainer.core._micro = 1024
    paths = []
    for index in range(2):
        path = root / f'feature_{index}.pt'
        torch.save(dict(record_id=f'response_{index}', x=torch.tensor([[index + 1., 2.]]).bfloat16()), path)
        paths.append(path)
    return trainer, paths


class OrdinaryRecoveryTests(unittest.TestCase):
    def test_native_observer_metadata_view_does_not_change_saved_endpoint(self):
        final = dict(global_step=513, presentations=1026, micro=1026, accumulation=0,
            gradients={'p': None}, optimizer={'optimizer_state_dict': {'state': {0: {'step': torch.tensor(513.)}}}})
        native = qualify.retained_native_view(final)
        self.assertEqual((native['global_step'], native['presentations']), (512, 1025))
        self.assertEqual((final['global_step'], final['presentations']), (513, 1026))
        self.assertIs(native['optimizer'], final['optimizer'])

    def test_recovery_recaptures_exact_gradients_without_repeating_adam(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            torch.manual_seed(17)
            trainer, paths = replay_fixture(root)
            origin = qualify.scientific_state(trainer)
            baseline_runtime = qualify.runtime_state(trainer.bundle.model)
            qualify.set_observation_directory(trainer, 'disposable_kplus')
            rows, original_gradients = [], {}
            ordinary_step = trainer.backend.optimizer.step

            def original_step():
                original_gradients.update({name: p.grad.detach().cpu().clone()
                    for name, p in trainer.bundle.draft_model.named_parameters()})
                return ordinary_step()

            with mock.patch.object(trainer.backend.optimizer, 'step', side_effect=original_step):
                for path in paths:
                    raw = torch.load(path, weights_only=True)
                    trainer._anchor_observations.clear()
                    result = trainer.core.train_step(raw, StepContext(global_step=512, total_steps=7424))
                    rows.append(dict(record_id=raw['record_id'],
                        sampled_anchors=deepcopy(trainer._anchor_observations),
                        metrics=result.materialize_metrics(), optimizer_stepped=result.optimizer_stepped))
                    trainer.presentations += 1
                    trainer.global_step += int(result.optimizer_stepped)
            endpoint = qualify.scientific_state(trainer)
            payload = root / 'retained.pt'
            torch.save(dict(endpoint, draft_state_dict=endpoint['weights']), payload)
            observations = root / 'observations.jsonl'
            observations.write_text(''.join(json.dumps(row) + '\n' for row in trainer.response_records))
            retained = dict(payload=payload, observations=observations, rows=rows, case=root / 'old_case')

            def restore(_trainer, _runtime, _root):
                trainer.bundle.draft_model.load_state_dict(origin['weights'])
                trainer.backend.optimizer.load_state_dict(deepcopy(origin['optimizer']))
                trainer.backend.optimizer.zero_grad()
                trainer.global_step, trainer.presentations = 512, 1024
                trainer.core._micro = 1024
                trainer.core._ratio_totals.clear()
                trainer.core._sum_totals.clear()
                trainer.response_records.clear()
                trainer.window_records.clear()
                trainer.work.clear()
                qualify.restore_runtime(trainer.bundle.model, baseline_runtime)
                restore_rng(origin['rng'])

            work = QualificationWork(root / 'work.jsonl')
            with mock.patch.object(qualify, 'restore_origin', side_effect=restore), \
                 mock.patch.object(qualify, 'export_disposable_state', return_value={'status': 'passed'}), \
                 mock.patch.object(trainer.backend.optimizer.optimizer, 'step', side_effect=AssertionError('Adam forbidden')):
                reference, receipt = qualify.recover_ordinary_reference(trainer, paths, retained, work,
                    baseline_runtime, root / 'recovered')
            self.assertEqual(receipt['status'], 'passed')
            self.assertEqual(receipt['optimizer_steps'], 0)
            self.assertEqual(receipt['reused_optimizer_steps'], 1)
            self.assertEqual(work.summary()['additional_forward_completed'], 2)
            self.assertEqual(work.summary()['additional_backward_completed'], 2)
            self.assertNotIn('adam_attempted', work.summary())
            self.assertFalse(qualify.differences(reference['gradients'], original_gradients))
            self.assertFalse(qualify.differences(reference['final'], endpoint))
            self.assertEqual(receipt['physical_work']['counts']['vocabulary_head_calls'], 2)


def failed_attempt_fixture(root):
    previous, destination = root / 'old', root / 'new'
    previous.mkdir()
    destination.mkdir()
    case = previous / 'qualification/gsm8k/KPLUS'
    case.mkdir(parents=True)
    observations = previous / 'qualification/gsm8k/trainer/events/disposable_kplus/RESPONSE_OBSERVATIONS.jsonl'
    observations.parent.mkdir(parents=True)
    observations.write_text('{}\n{}\n')
    startup = previous / 'qualification/gsm8k/STARTUP_EXACT_RESTORATION.json'
    qualify.write_json(startup, {'exact_equal': True})
    qualify.write_json(case / 'DECODED_STATE.json', {'exact_equal': True})
    qualify.write_json(case / 'RESULT.json', dict(status='failed', case='KPLUS',
        error='Native Adam counter differs: weight',
        traceback='restore_origin(trainer, baseline_runtime, root) then migrate_origin'))
    qualify.write_json(case / 'PRESENTATIONS.json', [
        dict(record_id='a', optimizer_stepped=False), dict(record_id='b', optimizer_stepped=True)])
    (case / 'disposable_state.pt').write_bytes(b'bound temporary payload')
    original_plan = dict(binding='fixed', geometry={}, serving={}, numerical_recipe={}, qualification_budget={},
        execution_source={'commit': 'original'}, datasets={'gsm8k': dict(origin_state='state',
            origin_state_identity={'bytes': 1}, origin_export='export', origin_weights_identity={'bytes': 2})})
    qualify.write_json(previous / 'sealed/PLAN.json', original_plan)
    preparation = dict(status='completed', datasets={'gsm8k': [{'request_id': 'a'}, {'request_id': 'b'}]})
    preparation_path = previous / 'qualification_feedback/PREPARATION.json'
    qualify.write_json(preparation_path, preparation)
    work = QualificationWork(previous / 'QUALIFICATION_WORK.jsonl')
    for kind, count in (('adam', 1), ('ordinary_forward', 2), ('ordinary_backward', 2),
                        ('additional_forward', 2), ('additional_backward', 2)):
        for _ in range(count):
            with work.operation(kind):
                pass
    def entry(path):
        return dict(path=str(path), identity=qualify.identity(path))
    plan = deepcopy(original_plan)
    plan['execution_source'] = {'commit': 'repair'}
    plan['repair'] = dict(previous_root=str(previous),
        prior_work_ledger=entry(work.path), shared_preparation=entry(preparation_path),
        retained_reference=dict(disposable_state=entry(case / 'disposable_state.pt'),
            presentations=entry(case / 'PRESENTATIONS.json'), decoded_state=entry(case / 'DECODED_STATE.json'),
            failed_result=entry(case / 'RESULT.json'), startup_restoration=entry(startup),
            response_observations=entry(observations)))
    return previous, destination, plan, preparation


class RecoveryAdmissionTests(unittest.TestCase):
    def test_admission_copies_exact_work_prefix_without_mutating_original(self):
        with tempfile.TemporaryDirectory() as temporary:
            previous, destination, plan, preparation = failed_attempt_fixture(Path(temporary))
            original = (previous / 'QUALIFICATION_WORK.jsonl').read_bytes()
            admitted = qualify.authenticate_failed_ordinary(previous, destination, plan, preparation)
            self.assertEqual((destination / 'QUALIFICATION_WORK.jsonl').read_bytes(), original)
            self.assertEqual(admitted['receipt']['remaining_adam_steps'], 7)
            work = QualificationWork(destination / 'QUALIFICATION_WORK.jsonl')
            with work.operation('additional_forward', purpose='replay'):
                pass
            self.assertEqual(work.summary()['adam_completed'], 1)
            self.assertEqual((previous / 'QUALIFICATION_WORK.jsonl').read_bytes(), original)

    def test_changed_retained_payload_stops_before_new_work_ledger(self):
        with tempfile.TemporaryDirectory() as temporary:
            previous, destination, plan, preparation = failed_attempt_fixture(Path(temporary))
            (previous / 'qualification/gsm8k/KPLUS/disposable_state.pt').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'Sealed recovery artifact changed'):
                qualify.authenticate_failed_ordinary(previous, destination, plan, preparation)
            self.assertFalse((destination / 'QUALIFICATION_WORK.jsonl').exists())

    def test_different_failure_stage_is_not_an_admitted_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            previous, destination, plan, preparation = failed_attempt_fixture(Path(temporary))
            path = previous / 'qualification/gsm8k/KPLUS/RESULT.json'
            row = json.loads(path.read_text())
            row['traceback'] = 'unknown stage'
            qualify.write_json(path, row)
            plan['repair']['retained_reference']['failed_result']['identity'] = qualify.identity(path)
            with self.assertRaisesRegex(ValueError, 'authenticated post-Adam'):
                qualify.authenticate_failed_ordinary(previous, destination, plan, preparation)
            self.assertFalse((destination / 'QUALIFICATION_WORK.jsonl').exists())


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
