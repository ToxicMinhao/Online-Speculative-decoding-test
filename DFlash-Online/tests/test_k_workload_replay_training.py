"""CPU validation and exact tiny-native continuity for additive Track2 training."""
from copy import deepcopy
from pathlib import Path
import json
import unittest
from unittest import mock

import torch
import yaml
from safetensors.torch import save_file

from experiments.k_workload_replay_v1 import training as T
from online_feedback.rng import capture_rng, restore_rng
import test_full_k_training as full_k_fixtures

own_state = full_k_fixtures.own_state

BINDING = 'a' * 64


def packets(event_number):
    return [dict(record_id=f'original_{index}', global_ordinal=index,
                 path=f'/prepared/original_{index}.pt', sha256=f'{index:064x}')
            for index in range(128 * (event_number - 1), 128 * event_number)]


def plan(arm, event_number):
    fresh = packets(event_number)
    second = fresh if arm == 'K' or event_number == 1 else packets(1)[:32] + fresh[:96]
    return [deepcopy(fresh), deepcopy(second)]


def replay_state(arm='K', event=0):
    state, arguments = own_state('K', event)
    state['algorithm_contract'].pop(T.full_k.CONTRACT_KEY)
    state['algorithm_contract'][T.CONTRACT_KEY] = T.experiment_contract(arm, BINDING)
    for number, entry in enumerate(state['events'], 1):
        inputs = plan(arm, number)
        for rows in inputs:
            for row in rows:
                row.update(prompt_length=2, response_tokens=4, length=6)
        entry['features'] = deepcopy(inputs[0])
        entry['pass_plan'] = deepcopy(inputs)
        for index, row in enumerate(entry['records']):
            packet = inputs[index // 128][index % 128]
            row.update(record_id=packet['record_id'], global_ordinal=packet['global_ordinal'],
                       replay=packet['global_ordinal'] < (number - 1) * 128,
                       prompt_tokens=2, response_tokens=4,
                       active_anchors=sum(row['sampled_anchors'][0]['keep_mask'][0]))
        entry['pass_exposure'] = T.pass_exposure(entry['records'])
    arguments.update(arm=arm, expected_contract=deepcopy(state['algorithm_contract']))
    return state, arguments


class PassPlanTests(unittest.TestCase):
    def test_explicit_passes_are_sorted_before_the_native_shuffle(self):
        before = capture_rng()
        source = plan('R', 2)
        expected = deepcopy(source)
        source[0].reverse()
        source[1].reverse()
        result = T.validate_pass_plan(source, arm='R', event_number=2, previous_records=packets(1))
        self.assertEqual(result, expected)
        self.assertFalse(T.differences(before, capture_rng()))
        self.assertNotEqual(source, result)

    def test_only_replay_arms_admit_the_exact_mixed_allocation(self):
        for arm in T.ARMS:
            for number in (1, 2, 8):
                previous = [row for n in range(1, number) for row in packets(n)]
                T.validate_pass_plan(plan(arm, number), arm=arm, event_number=number,
                                     previous_records=previous)
        with self.assertRaisesRegex(ValueError, 'two identical'):
            T.validate_pass_plan(plan('R', 2), arm='K', event_number=2)
        with self.assertRaisesRegex(ValueError, 'exactly 96'):
            T.validate_pass_plan(plan('K', 2), arm='U', event_number=2)
        for number in (0, 9):
            with self.assertRaises(ValueError):
                T.validate_pass_plan(plan('K', 1), arm='K', event_number=number)

    def test_duplicate_future_renamed_or_mutated_old_packets_fail(self):
        source = plan('U', 2)
        mutations = [lambda rows: rows[1].__setitem__(1, deepcopy(rows[1][0])),
                     lambda rows: rows[1][0].update(global_ordinal=300),
                     lambda rows: rows[1][0].update(record_id='renamed'),
                     lambda rows: rows[1][0].update(sha256='b' * 64),
                     lambda rows: rows[1][0].update(path='/replacement.pt'),
                     lambda rows: rows[1][-1].update(sha256='b' * 64)]
        for mutate in mutations:
            broken = deepcopy(source)
            mutate(broken)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                T.validate_pass_plan(broken, arm='U', event_number=2, previous_records=packets(1))

    def test_128_fresh_and_128_input_limits_and_no_history_relabeling(self):
        for count in (0, 127, 129):
            with self.assertRaises(ValueError):
                T.validate_pass_plan([packets(1), (packets(1) + packets(2))[:count]],
                                     arm='K', event_number=1)
        prior = packets(1)
        current = plan('K', 2)
        current[0][0]['record_id'] = prior[0]['record_id']
        current[1] = deepcopy(current[0])
        with self.assertRaisesRegex(ValueError, 'reuse'):
            T.validate_pass_plan(current, arm='K', event_number=2, previous_records=prior)


class ReplaySnapshotTests(unittest.TestCase):
    def setUp(self):
        before = capture_rng()
        self.addCleanup(lambda: restore_rng(before))

    def test_every_arm_accepts_own_complete_boundaries_without_rng_changes(self):
        for arm in T.ARMS:
            for number in (0, 1, 2, 8):
                state, arguments = replay_state(arm, number)
                before = capture_rng()
                receipt = T.validate_snapshot(state, **arguments)
                self.assertEqual(receipt['global_step'], number * 128)
                self.assertFalse(T.differences(before, capture_rng()))

    def test_preserved_horizon_and_recipe_reject_shortening_or_numerical_change(self):
        state, arguments = replay_state()
        for section, key, value in [('training', 'max_steps', 1024),
                                    ('online_feedback', 'max_events', 8),
                                    ('training', 'learning_rate', 1.5e-4),
                                    ('data', 'max_length', 2048)]:
            changed = yaml.safe_load(state['config_text'])
            changed[section][key] = value
            with self.assertRaises(ValueError):
                T.validate_config_text(yaml.safe_dump(changed), 'R', arguments['original_config_text'])

    def test_state_requires_exact_arm_adam_rng_counters_and_history(self):
        state, arguments = replay_state('R', 2)
        mutations = [lambda s: s['algorithm_contract'][T.CONTRACT_KEY].update(arm='U'),
                     lambda s: s.update(global_step=255),
                     lambda s: s['rng'].pop('numpy'),
                     lambda s: s['events'][1]['pass_plan'][1][0].update(sha256='f' * 64),
                     lambda s: s['backend']['optimizer']['optimizer_state_dict']['state'][0].pop('exp_avg'),
                     lambda s: s['backend']['optimizer']['optimizer_state_dict']['param_groups'][0].update(lr=.01)]
        for mutate in mutations:
            broken = deepcopy(state)
            mutate(broken)
            with self.assertRaises(ValueError):
                T.validate_snapshot(broken, **arguments)

    def test_neutral_equality_checks_decoded_state_and_native_anchors(self):
        first, _ = replay_state('K', 1)
        second = deepcopy(first)
        second['algorithm_contract'][T.CONTRACT_KEY] = T.experiment_contract('R', BINDING)
        second['run_id'] = 'R'
        second['events'][0]['features'][0]['path'] = '/another/path.pt'
        T.compare_neutral_states(first, second)
        second['events'][0]['records'][0]['sampled_anchors'][0]['positions'][0][0] += 1
        with self.assertRaisesRegex(ValueError, 'anchors differ'):
            T.compare_neutral_states(first, second)
        initial, _ = replay_state('K')
        other = deepcopy(initial)
        other['algorithm_contract'][T.CONTRACT_KEY] = T.experiment_contract('U', BINDING)
        self.assertTrue(T.compare_initial_states(initial, other)['exact_released_weights_empty_adam_rng'])
        other['rng']['torch'][0] ^= 1
        with self.assertRaises(ValueError):
            T.compare_initial_states(initial, other)

    def test_historical_guard_still_rejects_replay(self):
        state, arguments = own_state('K', 2)
        state['events'][1]['features'] = deepcopy(state['events'][0]['features'])
        for row, old in zip(state['events'][1]['records'], state['events'][0]['records']):
            row['record_id'] = old['record_id']
        with self.assertRaisesRegex(ValueError, 'reuses'):
            T.full_k.validate_snapshot(state, **arguments)


class ReplayTinyIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        full_k_fixtures.FullKTinyIntegrationTests.setUpClass.__func__(cls)

    @classmethod
    def tearDownClass(cls):
        full_k_fixtures.FullKTinyIntegrationTests.tearDownClass.__func__(cls)

    def setUp(self):
        full_k_fixtures.FullKTinyIntegrationTests.setUp(self)

    def build(self, name, arm='K', **kwargs):
        config = self.root / (arm + '.yaml')
        config.write_text(self.original_config_text)
        trainer = T.WorkloadReplayTrainer(config, self.root / name, arm=arm, binding=BINDING,
            original_config_text=self.original_config_text, expected_cuda_rng_states=0, **kwargs)
        self.addCleanup(trainer.close)
        return trainer

    def bindings(self, number=1):
        bound = []
        for index, source in enumerate(self.features):
            path = source
            if number != 1:
                raw = torch.load(source, weights_only=True)
                raw['record_id'] = f'fresh{number}_{index}'
                raw['label_binding']['record_id'] = raw['record_id']
                path = self.root / f'fresh{number}_{index}.pt'
                torch.save(raw, path)
            record_id = torch.load(path, weights_only=True)['record_id']
            bound.append(dict(record_id=record_id, global_ordinal=128 * (number - 1) + index,
                              path=str(path), sha256=T.file_sha256(path)))
        return bound

    def initialize_k(self, trainer):
        released = self.root / 'released'
        released.mkdir()
        save_file(dict(trainer.bundle.draft_model.state_dict()), released / 'model.safetensors')
        return trainer.save_initial()['state_path']

    def test_native_neutral_event_exact_three_arm_equality_and_deferred_export(self):
        fresh = self.bindings()
        states = []
        initial = None
        for arm in T.ARMS:
            trainer = self.build(arm, arm)
            if arm == 'K':
                initial = self.initialize_k(trainer)
            else:
                receipt = trainer.save_initial(comparator_path=initial, shared_snapshot=True)
                self.assertEqual(Path(receipt['state_path']).suffix, '.json')
                self.assertFalse((trainer.output_root / 'initial_state.pt').exists())
            result = trainer.train_event('event_001', pass_plan=[fresh, fresh])
            self.assertEqual(result['export_status'], 'deferred')
            self.assertIsNone(result['checkpoint_path'])
            self.assertFalse(Path(result['export_path']).exists())
            trainer.verify_boundary()
            states.append(torch.load(trainer.latest, weights_only=False))
            exported = trainer.materialize_export()
            self.assertTrue(exported['export_exact'])
            trainer.verify_boundary(exported['export_path'])
            self.assertEqual([row['replay_presentations'] for row in result['pass_exposure']], [0, 0])
            self.assertEqual(result['objective_work']['work']['response_forwards'], 256)
            trainer.close()
            self.assertFalse(hasattr(trainer, '_native_forward'))
        for state in states[1:]:
            T.compare_neutral_states(states[0], state)
        plain = T.SpecForgeEventTrainer(self.config, self.root / 'plain', objective=T.OBJECTIVE, ranking_lambda=0.)
        self.addCleanup(plain.close)
        ordinary = plain.train_event('event_001', self.features)
        reference = torch.load(plain.latest, weights_only=False)
        for key in ('draft_state_dict', 'backend', 'rng', 'global_step', 'presentations'):
            self.assertFalse(T.differences(states[0][key], reference[key]), key)
        self.assertEqual([row['sampled_anchors'] for row in states[0]['events'][0]['records']],
                         [row['sampled_anchors'] for row in ordinary['records']])
        self.assertFalse(torch.cuda.is_initialized())

    def test_mixed_event_resume_shared_initial_reference_and_packet_immutability(self):
        first = self.bindings()
        k = self.build('initial_K')
        initial = self.initialize_k(k)
        k.close()
        trainer = self.build('initial_U', 'U')
        shared = trainer.save_initial(comparator_path=initial, shared_snapshot=True)['state_path']
        trainer.close()
        trainer = self.build('U1', 'U', resume_state=shared, completed_event=0)
        trainer.train_event('event_001', pass_plan=[first, first])
        saved = self.root / 'event1.pt'
        saved.write_bytes(trainer.latest.read_bytes())
        second = self.bindings(2)
        mixed = [second, first[:32] + second[:96]]
        result = trainer.train_event('event_002', pass_plan=mixed)
        expected = torch.load(trainer.latest, weights_only=False)
        trainer.verify_boundary()
        self.assertEqual([row['replay_presentations'] for row in result['pass_exposure']], [0, 32])
        self.assertEqual(sum(row['optimizer_steps'] for row in result['pass_exposure']), 128)
        trainer.close()
        restored = self.build('restored_U', 'U', resume_state=saved, completed_event=1)
        repeated = restored.train_event('event_002', pass_plan=mixed)
        actual = torch.load(restored.latest, weights_only=False)
        for key in ('draft_state_dict', 'backend', 'rng', 'global_step', 'presentations'):
            self.assertFalse(T.differences(expected[key], actual[key]), key)
        self.assertEqual([row['sampled_anchors'] for row in result['records']],
                         [row['sampled_anchors'] for row in repeated['records']])
        restored.close()
        rejected = self.build('bad_U', 'U', resume_state=saved, completed_event=1)
        changed = Path(first[0]['path'])
        raw = torch.load(changed, weights_only=True)
        raw['hidden_states'].add_(1)
        torch.save(raw, changed)
        with self.assertRaisesRegex(ValueError, 'immutable content'):
            rejected.train_event('event_002', pass_plan=mixed)
        self.assertEqual(rejected.global_step, 128)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main()
