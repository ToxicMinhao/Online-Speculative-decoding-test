"""CPU-only migration, ordinary-loss, schedule and continuation boundaries."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
import yaml

from experiments.o_half_matched_lr import training as T
from experiments.o_half_matched_lr.contract import EVENT_WORK, NEW_BUDGET
from experiments.ok_coding import training as old
from experiments.ok_continuation.test_resume import runtime
from online_feedback.objective import TargetObjectiveModel
from online_feedback.optimizer import DirectBF16AdamW
from online_feedback.rng import capture_rng, restore_rng
from online_feedback.storage import save_torch
from test_first_error_objective import logit_fixture, probability_oracle
from test_objective_integration import tiny_author
from test_ok_coding_training import snapshot, contract


BINDING = dict(upstream_data_binding='retained-original-O', execution_commit='a' * 40)


def half_config(source):
    raw = yaml.safe_load(source['config_text'])
    raw['training']['learning_rate'] = 1.5e-4
    raw['run_id'] = 'own-o-half'
    raw['output_dir'] = '/new-output'
    return yaml.safe_dump(raw)


def arguments(source, seed=17):
    return dict(seed=seed, binding=BINDING, original_contract=contract(seed),
                original_algorithm_contract=deepcopy(source['algorithm_contract']),
                original_optimizer_recipe=json.loads(json.dumps(source['backend']['optimizer']['recipe'])),
                original_config_text=source['config_text'], config_text=half_config(source),
                expected_cuda_rng_states=0)


def migrated(seed=17):
    source = snapshot(seed, 'O')
    kwargs = arguments(source, seed)
    expected = deepcopy(source['algorithm_contract'])
    expected[T.CONTRACT_KEY] = T.experiment_contract(seed, BINDING)
    candidate, receipt = T.migrate_source(source, **kwargs, new_contract=expected)
    return source, candidate, receipt, kwargs


class OHalfScientificStateTests(unittest.TestCase):
    def setUp(self):
        rng = capture_rng()
        self.addCleanup(lambda: restore_rng(rng))

    def test_migration_preserves_original_O_and_changes_only_allowed_fields(self):
        for seed in (17, 29):
            source, candidate, receipt, kwargs = migrated(seed)
            self.assertEqual(receipt['original_arm'], 'O')
            self.assertTrue(receipt['source_weights_rng_moments_counters_history_exact'])
            self.assertEqual(candidate['backend']['optimizer']['optimizer_state_dict']['state'], {})
            for key in ('draft_state_dict', 'rng', 'global_step', 'presentations', 'events'):
                self.assertFalse(old.inherited.differences(source[key], candidate[key]), key)
            self.assertEqual(source['backend']['optimizer']['recipe']['optimizer_options']['lr'], 3e-4)
            self.assertEqual(candidate['backend']['optimizer']['recipe']['optimizer_options']['lr'], 1.5e-4)
            self.assertEqual(candidate['backend']['optimizer']['optimizer_state_dict']['param_groups'][0]['lr'], 1.5e-4)
            self.assertEqual(candidate['algorithm_contract']['online_target_objective']['name'], 'osd')
            T.validate_snapshot(candidate, **kwargs, completed_event=0)

    def test_original_snapshot_and_recipe_guards_remain_strict(self):
        _, candidate, _, _ = migrated()
        with self.assertRaisesRegex(ValueError, 'learning_rate'):
            old.validate_snapshot(candidate, arm='O', seed=17, campaign_contract=contract(17),
                config_text=candidate['config_text'], completed_event=0, expected_cuda_rng_states=0)

    def test_only_learning_rate_and_run_output_identity_may_change(self):
        source, _, _, kwargs = migrated()
        for section, key, value in [('training', 'seed', 29), ('training', 'num_anchors', 64),
                                    ('model', 'draft_checkpoint_path', '/different'),
                                    ('online_feedback', 'weight_decay', .005),
                                    ('data', 'hidden_states_path', '/different')]:
            raw = yaml.safe_load(kwargs['config_text'])
            raw[section][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                T.validate_config_text(yaml.safe_dump(raw), 17, source['config_text'])

    def test_own_completed_event_states_preserve_adam_continuity(self):
        for number in range(1, 5):
            _, own, _, kwargs = migrated()
            continued = snapshot(17, 'O', number)
            own.update(draft_state_dict=continued['draft_state_dict'],
                       global_step=number * 128, presentations=number * 256,
                       events=continued['events'],
                       backend={'optimizer': T._optimizer_view(continued['backend']['optimizer'], 1.5e-4)})
            checked = T.validate_snapshot(own, **kwargs, completed_event=number)
            self.assertEqual(checked['global_step'], number * 128)
            self.assertEqual(checked['adam_state_entries'], 58)

    def test_wrong_objective_seed_mechanism_config_and_incomplete_boundary_rejected(self):
        _, own, _, kwargs = migrated()
        mutations = [
            lambda s: s['algorithm_contract']['online_target_objective'].update(name='osd_first_error'),
            lambda s: s['algorithm_contract'][T.CONTRACT_KEY].update(seed=29),
            lambda s: s['algorithm_contract'].update(prefix_guard={}),
            lambda s: s['algorithm_contract'].update(qualification_only={}),
            lambda s: s['backend'].update(controller={}),
            lambda s: s.update(qualification_steps=1),
            lambda s: s['rng'].pop('numpy'),
            lambda s: s.update(presentations=1),
            lambda s: s['backend']['optimizer']['optimizer_state_dict']['param_groups'][0].update(lr=3e-4),
        ]
        for mutation in mutations:
            broken = deepcopy(own)
            mutation(broken)
            with self.assertRaises((ValueError, KeyError)):
                T.validate_snapshot(broken, **kwargs, completed_event=0)

    def test_parameter_order_and_every_optimizer_option_are_pinned(self):
        _, own, _, kwargs = migrated()
        for mutation in ('order', 'foreach'):
            bad = deepcopy(own)
            recipe = bad['backend']['optimizer']['recipe']
            if mutation == 'order':
                recipe['parameters'] = tuple(reversed(recipe['parameters']))
            else:
                recipe['optimizer_options']['foreach'] = True
                bad['backend']['optimizer']['optimizer_state_dict']['param_groups'][0]['foreach'] = True
            with self.assertRaisesRegex(ValueError, 'recipe or parameter ordering'):
                T.validate_snapshot(bad, **kwargs, completed_event=0)

    def test_K_or_trained_O_source_cannot_initialize_new_arm(self):
        for arm, event in (('K', 0), ('O', 1)):
            source = snapshot(17, arm, event)
            kwargs = arguments(source)
            expected = deepcopy(source['algorithm_contract'])
            expected[T.CONTRACT_KEY] = T.experiment_contract(17, BINDING)
            with self.assertRaises(ValueError):
                T.migrate_source(source, **kwargs, new_contract=expected)

    def test_atomic_save_and_restore_leave_complete_rng_and_optimizer_exact(self):
        _, own, _, _ = migrated()
        trainer = runtime(own)
        trainer.__class__ = T.OHalfTrainer
        trainer.backend.optimizer = DirectBF16AdamW(trainer.bundle.draft_model, learning_rate=1.5e-4)
        trainer.backend.load_state_dict = lambda state: trainer.backend.optimizer.load_state_dict(state['optimizer'])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'state.pt'
            save_torch(path, own)
            decoded = torch.load(path, map_location='cpu', weights_only=False)
            trainer._load_decoded(decoded)
            self.assertFalse(old.inherited.differences(own['rng'], capture_rng()))
            self.assertFalse(old.inherited.differences(own['backend']['optimizer'], trainer.backend.optimizer.state_dict()))
            self.assertFalse(old.inherited.differences(own['draft_state_dict'], dict(trainer.bundle.draft_model.state_dict())))
            self.assertEqual(trainer.core._micro, 0)

    def test_wrapper_requires_checked_natural_schedule_and_never_forces_anchors(self):
        trainer = object.__new__(T.OHalfTrainer)
        trainer.completed_event = 0
        with self.assertRaisesRegex(ValueError, 'natural schedule'):
            trainer.train_event('event_001', [])
        from experiments.ok_continuation.test_resume import event
        receipt = event(objective='osd')
        with mock.patch.object(T.SpecForgeEventTrainer, 'train_event', return_value='native') as native:
            self.assertEqual(trainer.train_event('event_001', [], presentation_schedule=receipt), 'native')
            self.assertIs(native.call_args.kwargs['presentation_schedule'], receipt)
        wrong = deepcopy(receipt)
        wrong['objective'] = 'osd_first_error'
        with self.assertRaises(ValueError):
            trainer.train_event('event_001', [], presentation_schedule=wrong)

    def test_native_O_keeps_loss_and_gradient_after_first_error(self):
        logits, teacher, labels, weights = logit_fixture()
        results = {}
        for objective in ('osd', 'osd_first_error'):
            model = TargetObjectiveModel(tiny_author(), objective, loss_precision='bfloat16')
            model.lm_head = torch.nn.Identity()
            current = logits.clone().requires_grad_()
            terms = model._chunk_loss(current, None, labels, weights, teacher)
            loss = terms[0] / (terms[1] + 1e-6)
            loss.backward()
            results[objective] = (loss, current.grad, terms[1])
        expected_logits = logits.clone().requires_grad_()
        oracle, _, _, _ = probability_oracle(expected_logits, teacher, labels, weights, 'osd')
        oracle.backward()
        torch.testing.assert_close(results['osd'][0], oracle, rtol=0, atol=0)
        torch.testing.assert_close(results['osd'][1], expected_logits.grad, rtol=0, atol=0)
        self.assertGreater(float(results['osd'][1][0, 0, 2:].abs().sum()), 0)
        self.assertEqual(float(results['osd_first_error'][1][0, 0, 2:].abs().sum()), 0)
        self.assertGreater(float(results['osd'][2]), float(results['osd_first_error'][2]))

    def test_budget_contains_no_reference_or_disposable_optimizer_work(self):
        self.assertEqual(EVENT_WORK, dict(forwards=256, backwards=256, optimizer_steps=128))
        self.assertEqual(NEW_BUDGET['optimizer_steps'], 1024)
        self.assertEqual(NEW_BUDGET['presentations'], 2048)
        self.assertEqual(NEW_BUDGET['reference_forwards'], 0)
        self.assertEqual(NEW_BUDGET['qualification_optimizer_steps'], 0)
        self.assertEqual(NEW_BUDGET['evaluation'], 1072)


class OHalfTinyIntegrationTests(unittest.TestCase):
    """Real tiny DFlash/O/Adam loop; eager CPU attention replaces GPU Flex."""

    @classmethod
    def setUpClass(cls):
        import test_ok_coding_training as fixtures
        fixtures.TinyCodingIntegrationTests.setUpClass.__func__(cls)

    @classmethod
    def tearDownClass(cls):
        import test_ok_coding_training as fixtures
        fixtures.TinyCodingIntegrationTests.tearDownClass.__func__(cls)

    def setUp(self):
        import test_ok_coding_training as fixtures
        import specforge.training.assembly as assembly
        from online_feedback.objective import _model_options
        from online_feedback.rng import preserve_rng
        from specforge.algorithms.common.dflash_family_model import OnlineDFlashModel
        fixtures.TinyCodingIntegrationTests.setUp(self)
        tiny_loader = assembly.build_model_bundle.side_effect

        def five_layer_tiny(cfg, **kwargs):
            bundle = tiny_loader(cfg, **kwargs)
            config = deepcopy(bundle.draft_model.config)
            config.num_hidden_layers = 5
            config.layer_types = ['full_attention'] * 5
            config.dflash_config['target_layer_ids'] = list(bundle.draft_model.target_layer_ids)
            with preserve_rng():
                torch.manual_seed(41)
                draft = type(bundle.draft_model)(config).to(dtype=torch.bfloat16)
            bundle.model = OnlineDFlashModel(draft_model=draft, **_model_options(bundle.model))
            bundle.draft_model = draft
            return bundle

        assembly.build_model_bundle.side_effect = five_layer_tiny
        for index in range(len(self.features), 128):
            raw = torch.load(self.features[index % 2], weights_only=True)
            raw['record_id'] = f'response_{index}'
            raw['label_binding']['record_id'] = raw['record_id']
            path = self.root/f'feedback_{index}.pt'
            torch.save(raw, path)
            self.features.append(path)

    def test_native_wrapper_instrumentation_and_own_resume(self):
        ordinary = old.CodingTrainer(self.config, self.root/'source', arm='O', seed=17,
                                      campaign_contract=contract(17))
        self.addCleanup(ordinary.close)
        source = deepcopy(ordinary.initial_state())
        self.assertFalse(torch.cuda.is_initialized(), 'Tiny fixture must remain CPU-only')
        self.assertEqual(source['rng']['cuda'], [])
        schedule = ordinary.train_event('event_001', self.features)
        ordinary.close()
        kwargs = arguments(source)
        self.config.write_text(kwargs['config_text'])
        kwargs.pop('config_text')
        checked = T.OHalfTrainer(self.config, self.root/'half', **kwargs, source_state=source)
        self.addCleanup(checked.close)
        self.assertEqual(checked.objective, 'osd')
        self.assertFalse(checked.bundle.model.ranking_enabled)
        receipt = checked.train_event('event_001', self.features, presentation_schedule=schedule)
        state = torch.load(checked.latest, map_location='cpu', weights_only=False)
        self.assertTrue(T.verify_live(checked, state)['exact_resume'])
        self.assertEqual(len(receipt['records']), 256)
        self.assertEqual(receipt['objective_work']['work']['response_forwards'], 256)
        self.assertEqual(receipt['objective_work']['work']['current_draft_forwards'], 256)
        self.assertEqual(receipt['objective_work']['reference_operations'], 0)
        self.assertTrue(all('prefix_protection' not in r['response_observation'] for r in receipt['records']))
        own_path = checked.latest
        checked.close()
        restored = T.OHalfTrainer(self.config, self.root/'restored', **kwargs,
                                  resume_state=own_path, completed_event=1)
        self.addCleanup(restored.close)
        self.assertTrue(T.verify_live(restored, state)['exact_resume'])
        restored.close()
        # The wrapper's observation hooks must leave native O weights, moments,
        # RNG and per-step gradients exactly as the unchanged half-LR O loop.
        plain = T.SpecForgeEventTrainer(self.config, self.root/'plain',
                                       objective='osd', reduction='response_mean', ranking_lambda=0.)
        self.addCleanup(plain.close)
        plain.bundle.draft_model.load_state_dict(source['draft_state_dict'])
        plain.backend.load_state_dict({'optimizer': T._optimizer_view(source['backend']['optimizer'], 1.5e-4)})
        restore_rng(source['rng'])
        result = plain.train_event('event_001', self.features, presentation_schedule=schedule)
        plain_state = torch.load(plain.latest, map_location='cpu', weights_only=False)
        for key in ('draft_state_dict', 'backend', 'rng', 'global_step', 'presentations'):
            self.assertFalse(old.inherited.differences(state[key], plain_state[key]), key)
        self.assertEqual([r['metrics'] for r in receipt['records']], [r['metrics'] for r in result['records']])
        self.assertEqual([r['sampled_anchors'] for r in receipt['records']], [r['sampled_anchors'] for r in result['records']])
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main()
