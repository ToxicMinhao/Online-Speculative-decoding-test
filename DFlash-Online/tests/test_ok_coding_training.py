"""CPU checks for genuine event-zero states, seeds, restores and natural draws."""
from copy import deepcopy
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch
import yaml
from safetensors.torch import save_file

from experiments.ok_coding import training as C
from experiments.ok_continuation.test_resume import draft, event, fixture, runtime
from online_feedback.optimizer import DirectBF16AdamW
from online_feedback.presentation_schedule import PresentationSchedule
from online_feedback.rng import capture_rng, restore_rng


def contract(seed):
    return dict(schema='ok_coding_v1', training_seed=seed, data_binding='sealed-inputs',
                execution_commit='a'*40, events=[1, 2, 3, 4],
                responses_per_event=128, passes_per_event=2)


def configuration(seed, released='/released'):
    raw = yaml.safe_load((Path(__file__).parents[1]/'configs/osd.yaml').read_text())
    raw['training']['seed'] = seed
    raw['data']['max_length'] = 2272
    raw['model']['draft_checkpoint_path'] = str(released)
    return yaml.safe_dump(raw)


def snapshot(seed=17, arm='O', number=0, released='/released'):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if number:
        state = fixture(C.OBJECTIVES[arm], number)
    else:
        model = draft()
        optimizer = DirectBF16AdamW(model)
        state = dict(event_schema=C.SCHEMA, draft_state_dict=dict(deepcopy(model.state_dict())),
            backend={'optimizer': deepcopy(optimizer.state_dict())}, rng=capture_rng(),
            global_step=0, presentations=0, events=[], strategy='dflash', run_id='fixture',
            world_size=1, reduction='response_mean', accumulation_steps=2, horizon=7424,
            algorithm_contract={'online_target_objective': {'name': C.OBJECTIVES[arm]}})
    state['config_text'] = configuration(seed, released)
    state['algorithm_contract'][C.CONTRACT_KEY] = C.coding_contract(seed, contract(seed))
    return state


class CodingStateTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        saved_rng = capture_rng()
        self.addCleanup(lambda: restore_rng(saved_rng))

    def validate(self, state, seed=17, arm='O', number=0):
        return C.validate_snapshot(state, arm=arm, seed=seed, campaign_contract=contract(seed),
            config_text=state['config_text'], completed_event=number, expected_cuda_rng_states=0)

    def test_both_seed_initial_pairs_match_with_empty_native_adam_and_distinct_rng(self):
        states = []
        for seed in (17, 29):
            ordinary, candidate = snapshot(seed), snapshot(seed, 'K')
            before = capture_rng()
            checked = C.compare_initial_states(ordinary, candidate, seed=seed,
                campaign_contract=contract(seed), expected_cuda_rng_states=0)
            self.assertTrue(checked['exact_initial_state'])
            self.assertFalse(C.inherited.differences(before, capture_rng()))
            self.assertEqual(ordinary['backend']['optimizer']['optimizer_state_dict']['state'], {})
            states.append(ordinary)
        self.assertFalse(C.inherited.differences(states[0]['draft_state_dict'], states[1]['draft_state_dict']))
        for component in ('python', 'numpy', 'torch'):
            self.assertTrue(C.inherited.differences(states[0]['rng'][component], states[1]['rng'][component]))

    def test_init_serializer_has_no_dummy_event_and_decoded_released_check(self):
        released = self.root/'released'
        released.mkdir()
        state = snapshot(29, released=released)
        save_file(state['draft_state_dict'], released/'model.safetensors')
        trainer = runtime(state)
        trainer.__class__ = C.CodingTrainer
        trainer.arm, trainer.seed, trainer.completed_event = 'O', 29, 0
        trainer.campaign_contract = contract(29)
        trainer.cfg = SimpleNamespace(run_id='fixture', model=SimpleNamespace(draft_checkpoint_path=str(released)))
        trainer.strategy = SimpleNamespace(checkpoint_state_filter=lambda values: dict(values))
        trainer.backend.state_dict = lambda: dict(model=trainer.bundle.draft_model.state_dict(),
                                                  optimizer=trainer.backend.optimizer.state_dict())
        restore_rng(state['rng'])
        with mock.patch.object(trainer.backend.optimizer, 'step', side_effect=AssertionError('no warmup')):
            receipt = trainer.save_initial_state(self.root/'initial.pt', expected_cuda_rng_states=0)
        saved = torch.load(receipt['path'], weights_only=False)
        self.assertEqual(saved['events'], [])
        self.assertEqual(saved['backend']['optimizer']['optimizer_state_dict']['state'], {})
        self.assertTrue(receipt['released_weights_exact'])
        self.assertFalse(C.inherited.differences(state['rng'], capture_rng()))
        self.assertFalse(C.inherited.differences(state['draft_state_dict'], saved['draft_state_dict']))
        with self.assertRaises(FileExistsError):
            trainer.save_initial_state(self.root/'initial.pt', expected_cuda_rng_states=0)
        next(trainer.bundle.draft_model.parameters()).detach().add_(1)
        with self.assertRaisesRegex(ValueError, 'released weights'):
            trainer.save_initial_state(self.root/'changed.pt', expected_cuda_rng_states=0)
        self.assertFalse((self.root/'changed.pt').exists())

    def test_zero_and_each_complete_event_restore_exact_own_state(self):
        for seed in (17, 29):
            for arm in ('O', 'K'):
                for number in range(5):
                    with self.subTest(seed=seed, arm=arm, event=number):
                        state = snapshot(seed, arm, number)
                        path = self.root/f'{seed}_{arm}_{number}.pt'
                        torch.save(state, path)
                        export = self.root/path.stem
                        export.mkdir()
                        save_file(state['draft_state_dict'], export/'model.safetensors')
                        loaded, receipt = C.load_snapshot(path, export, arm=arm, seed=seed,
                            campaign_contract=contract(seed), config_text=state['config_text'],
                            completed_event=number, expected_cuda_rng_states=0)
                        self.assertTrue(receipt['export_exact'])
                        trainer = runtime(state)
                        trainer._restore(path)
                        self.assertTrue(C.verify_live_resume(trainer, loaded)['exact_resume'])
                        self.assertFalse(C.inherited.differences(state['rng'], capture_rng()))

    def test_zero_state_rejects_warmup_history_rng_options_and_wrong_seed(self):
        original = snapshot()
        mutations = {
            'fake_history': lambda s: s['events'].append(event()),
            'one_step': lambda s: s.update(global_step=1),
            'populated_adam': lambda s: s['backend']['optimizer']['optimizer_state_dict']['state'].update({0:{}}),
            'group_options': lambda s: s['backend']['optimizer']['optimizer_state_dict']['param_groups'][0].update(lr=1.),
            'missing_cuda_rng': lambda s: s['rng'].pop('cuda'),
            'wrong_seed': lambda s: s['algorithm_contract'][C.CONTRACT_KEY].update(training_seed=29),
            'controller': lambda s: s['algorithm_contract'].update(response_weight_controller={}),
        }
        for label, mutate in mutations.items():
            state = deepcopy(original)
            mutate(state)
            with self.subTest(label=label), self.assertRaises(ValueError):
                self.validate(state)
        with self.assertRaisesRegex(ValueError, 'different training seed'):
            C.coding_contract(29, contract(17))

    def test_pair_gate_rejects_changed_weights_rng_or_trainable_order(self):
        ordinary, candidate = snapshot(), snapshot(arm='K')
        for label in ('weight', 'rng', 'order'):
            changed = deepcopy(candidate)
            if label == 'weight':
                next(iter(changed['draft_state_dict'].values())).add_(1)
            elif label == 'rng':
                changed['rng']['torch'] = torch.Generator().manual_seed(29).get_state()
            else:
                recipe = changed['backend']['optimizer']['recipe']
                recipe['parameters'] = tuple(reversed(recipe['parameters']))
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, 'initialization differs'):
                C.compare_initial_states(ordinary, changed, seed=17, campaign_contract=contract(17),
                                         expected_cuda_rng_states=0)

    def test_event_one_checks_natural_schedule_and_keeps_old_guard(self):
        ordinary, candidate = event(), event(objective='osd_first_error')
        result = C.validate_event_receipt(candidate, event_number=1,
            objective='osd_first_error', reference_schedule=ordinary)
        self.assertTrue(result['schedule_matched'])
        schedule = PresentationSchedule(ordinary, [r['record_id'] for r in ordinary['features']],
                                        2, features=ordinary['features'])
        for row in candidate['records']:
            schedule.begin_presentation(row['pass'], row['record_id'])
            schedule.check_anchors(row['sampled_anchors'][0])
            schedule.finish_presentation()
        self.assertEqual(schedule.position, 256)
        candidate['records'][0]['sampled_anchors'][0]['positions'][0][0] += 1
        with self.assertRaisesRegex(ValueError, 'natural response order or anchors'):
            C.validate_event_receipt(candidate, event_number=1,
                objective='osd_first_error', reference_schedule=ordinary)
        with self.assertRaisesRegex(ValueError, 'replay event 1'):
            C.inherited.validate_event_receipt(event(objective='osd_first_error'), event_number=1,
                objective='osd_first_error', reference_schedule=ordinary)

    def test_seed_horizon_native_schedule_and_weights_path_are_pinned(self):
        text = configuration(29)
        C.validate_config_text(text, 29)
        for section, key, value in [('training', 'seed', 17), ('training', 'max_steps', 512),
                                   ('online_feedback', 'max_events', 4),
                                   ('model', 'draft_checkpoint_path', None)]:
            raw = yaml.safe_load(text)
            raw[section][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                C.validate_config_text(yaml.safe_dump(raw), 29)


class TinyCodingIntegrationTests(unittest.TestCase):
    """Actual tiny DFlash, native BF16 Adam and sampling; no CUDA model work."""

    @classmethod
    def setUpClass(cls):
        from test_osd_recipe import OSDRecipeTests
        cls.legacy = OSDRecipeTests
        cls.legacy.setUpClass.__func__(cls)

    @classmethod
    def tearDownClass(cls):
        cls.legacy.tearDownClass.__func__(cls)

    def setUp(self):
        import specforge.training.assembly as assembly
        from online_feedback import training
        self.legacy.setUp(self)
        raw = yaml.safe_load(self.config.read_text())
        raw['data']['max_length'] = 2272
        self.config.write_text(yaml.safe_dump(raw))
        patch = mock.patch.object(training, 'ranking_eos_token_ids', return_value=(28,))
        patch.start()
        self.addCleanup(patch.stop)
        tiny_loader = assembly.build_model_bundle.side_effect

        def released_tiny(cfg, **kwargs):
            # The existing tiny factory uses its own seed to make fixed weights.
            # Isolate that fixture detail so the requested training seed survives.
            with C.preserve_rng():
                bundle = tiny_loader(cfg, **kwargs)
            bundle.model.block_size = bundle.draft_model.config.block_size = 16
            bundle.model.loss_decay_gamma = 7.
            bundle.model.objective_chunk_blocks = 8
            return bundle

        assembly.build_model_bundle.side_effect = released_tiny

    def build(self, seed, arm, name, **kwargs):
        raw = yaml.safe_load(self.config.read_text())
        raw['training']['seed'] = seed
        self.config.write_text(yaml.safe_dump(raw))
        trainer = C.CodingTrainer(self.config, self.root/name, arm=arm, seed=seed,
                                 campaign_contract=contract(seed), **kwargs)
        self.addCleanup(trainer.close)
        return trainer

    def test_real_construction_preserves_pair_parity_and_seed_distinction(self):
        states = {}
        for seed in (17, 29):
            for arm in ('O', 'K'):
                trainer = self.build(seed, arm, f'{seed}_{arm}')
                states[seed, arm] = deepcopy(trainer.initial_state())
                self.assertEqual(states[seed, arm]['events'], [])
                self.assertEqual(states[seed, arm]['backend']['optimizer']['optimizer_state_dict']['state'], {})
                trainer.close()
            for field in ('draft_state_dict', 'backend', 'rng', 'global_step', 'presentations', 'events'):
                self.assertFalse(C.inherited.differences(states[seed, 'O'][field], states[seed, 'K'][field]), field)
        for field in ('python', 'numpy', 'torch'):
            self.assertTrue(C.inherited.differences(states[17, 'O']['rng'][field], states[29, 'O']['rng'][field]))
        self.assertFalse(torch.cuda.is_initialized())

    def test_actual_seed29_event_one_natural_schedule_and_complete_restore(self):
        for index in range(len(self.features), 128):
            raw = torch.load(self.features[index % 2], weights_only=True)
            raw['record_id'] = f'response_{index}'
            raw['label_binding']['record_id'] = raw['record_id']
            path = self.root/f'feedback_{index}.pt'
            torch.save(raw, path)
            self.features.append(path)
        ordinary = self.build(29, 'O', 'ordinary')
        receipt = ordinary.train_event('event_001', self.features)
        ordinary.close()
        candidate = self.build(29, 'K', 'candidate')
        checked = candidate.train_event('event_001', self.features, presentation_schedule=receipt)
        gate = C.validate_event_receipt(checked, event_number=1, objective='osd_first_error',
                                       reference_schedule=receipt, trainer_status=candidate.status())
        self.assertTrue(gate['schedule_matched'])
        state = torch.load(candidate.latest, weights_only=False)
        source = candidate.latest
        candidate.close()
        restored = self.build(29, 'K', 'restored', resume_state=source, completed_event=1)
        self.assertTrue(C.verify_live_resume(restored, state)['exact_resume'])
        self.assertEqual((restored.global_step, restored.presentations), (128, 256))
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main()
