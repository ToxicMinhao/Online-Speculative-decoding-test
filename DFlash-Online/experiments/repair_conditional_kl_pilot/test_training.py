"""CPU boundary tests reuse the qualified native 58-tensor state fixture."""
from copy import deepcopy
from pathlib import Path
import sys
import os
import subprocess

import unittest
import torch
from safetensors.torch import save_file

from experiments.repair_conditional_kl_pilot import configuration as C, training as T
from experiments.repair_conditional_kl_pilot.witness import block_summary, compare_blocks
from online_feedback.rng import capture_rng

TESTS = Path(__file__).resolve().parents[2] / 'tests'
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))
import test_full_k_training as fixtures

BINDING = 'e' * 64


def state_fixture(arm='BASE', event=0, maximum_events=2):
    state, old = fixtures.own_state('K', event)
    state['config_text'] = C.build_config(old['original_config_text'], arm=arm,
                                         output_dir='/pilot/' + arm, maximum_events=maximum_events)
    state['algorithm_contract'].pop(T.native.CONTRACT_KEY)
    state['algorithm_contract'][T.CONTRACT_KEY] = T.experiment_contract(arm, BINDING, maximum_events)
    for entry in state['events']:
        entry.update(method=arm, dataset='OpenCodeInstruct',
                     **{T.CONTRACT_KEY: T.experiment_contract(arm, BINDING, maximum_events)})
        for row in entry['records']:
            observation = dict(record_id=row['record_id'], pass_index=row['pass'],
                sampled_anchors=deepcopy(row['sampled_anchors']), loss=1., sum_metrics={},
                ratio_metrics={'repair_pair_auxiliary': [1., 2.]} if arm == 'PAIR' else {},
                prefix_allocation=dict.fromkeys(('changed_positions', 'changed_blocks', 'weight_moved',
                    'prefix_weight_before', 'prefix_weight_after', 'mass_error_abs_sum', 'delta_numerator',
                    'mass_violations', 'bounds_violations', 'support_violations', 'nonfinite_values'), 0.))
            row['response_observation'] = observation
    return state, dict(arm=arm, binding=BINDING, maximum_events=maximum_events,
        config_text=state['config_text'], original_config_text=old['original_config_text'],
        completed_event=event, expected_contract=deepcopy(state['algorithm_contract']),
        expected_recipe=deepcopy(state['backend']['optimizer']['recipe']), expected_cuda_rng_states=0)


class PilotTrainingTests(unittest.TestCase):
    def test_native_complete_state_and_moments_are_preserved(self):
        for arm in T.ARMS:
            for event in (0, 1, 2):
                state, args = state_fixture(arm, event)
                before = deepcopy(state)
                checked = T.validate_snapshot(state, **args)
                assert checked['global_step'] == event * 128
                assert checked['adam_state_entries'] == (58 if event else 0)
                assert not T.differences(before, state)


    def test_sealed_one_event_does_not_accept_second_event_or_foreign_state(self):
        state, args = state_fixture('PAIR', 2, maximum_events=1)
        with self.assertRaisesRegex(ValueError, 'endpoint'):
            T.validate_snapshot(state, **args)
        state, args = state_fixture('PAIR', 1, maximum_events=1)
        for mutate in (
            lambda s: s['algorithm_contract'][T.CONTRACT_KEY].update(arm='BASE'),
            lambda s: s['events'][0]['records'][0]['response_observation']['ratio_metrics'].clear(),
            lambda s: s['events'][0]['records'][0]['response_observation']['prefix_allocation'].update(mass_violations=1),
            lambda s: s.update(global_step=129),
            lambda s: s['backend']['optimizer']['optimizer_state_dict']['param_groups'][0].update(lr=.001),
        ):
            changed = deepcopy(state)
            mutate(changed)
            with self.assertRaises(ValueError):
                T.validate_snapshot(changed, **args)


    def test_initial_comparison_allows_only_declared_arm_and_output_identity(self):
        base, _ = state_fixture('BASE')
        pair = deepcopy(base)
        pair['algorithm_contract'][T.CONTRACT_KEY] = T.experiment_contract('PAIR', BINDING)
        first, second = T.initial_witness(base), T.initial_witness(pair)
        assert T.compare_initial_witnesses(first, second)['exact_weights_empty_Adam_RNG']
        second['rng']['torch'][0] ^= 1
        with self.assertRaisesRegex(ValueError, 'initialization differs'):
            T.compare_initial_witnesses(first, second)


    def test_disposable_guard_forbids_second_adam_without_any_forward(self):
        trainer = object.__new__(T.PilotTrainer)
        trainer.qualification_only = True
        trainer.global_step, trainer.presentations = 1, 2
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            trainer.run_pair([])


    def test_witness_counts_repair_and_earlier_prefix_damage_together(self):
        labels = torch.tensor([[[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]])
        weights = torch.tensor([[[0., 1, 1, 1, 1], [0., 1, 1, 1, 1]]])
        before = torch.tensor([[[0, 1, 2, 9, 4], [0, 1, 9, 3, 4]]])
        after = torch.tensor([[[0, 9, 2, 3, 4], [0, 1, 2, 3, 4]]])
        result = compare_blocks(block_summary(before, labels, weights), block_summary(after, labels, weights))
        assert result['repaired_boundaries'] == 2
        assert result['damaged_prefix_positions'] == 1
        assert result['repaired_without_prefix_damage'] == 1
        assert result['net_consecutive_match_length_change'] == 1
        assert result['regressed_blocks'] == result['improved_blocks'] == 1


    def test_witness_respects_eos_and_refuses_changed_context(self):
        labels = torch.tensor([[[0, 1, 3, 4]]])
        weights = torch.tensor([[[0., 1, 1, 1]]])
        result = block_summary(labels, labels, weights, (3,))
        assert result['prefix_length'] == [[2]]
        assert result['valid'] == [[[False, True, True, False]]]
        changed = deepcopy(result)
        changed['labels'][0][0][1] = 2
        with self.assertRaisesRegex(ValueError, 'changed'):
            compare_blocks(result, changed)


class TinyPilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get('REPAIR_PAIR_TINY_ARM'):
            fixtures.FullKTinyIntegrationTests.setUpClass.__func__(cls)

    @classmethod
    def tearDownClass(cls):
        if os.environ.get('REPAIR_PAIR_TINY_ARM'):
            fixtures.FullKTinyIntegrationTests.tearDownClass.__func__(cls)

    def setUp(self):
        if os.environ.get('REPAIR_PAIR_TINY_ARM'):
            fixtures.FullKTinyIntegrationTests.setUp(self)

    def test_real_native_event_witness_and_endpoint_guard(self):
        arm = os.environ.get('REPAIR_PAIR_TINY_ARM')
        if arm is None:
            root = Path(__file__).resolve().parents[2]
            for current in T.ARMS:
                environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', REPAIR_PAIR_TINY_ARM=current)
                environment['PYTHONPATH'] = os.pathsep.join(str(root / path)
                    for path in ('.', 'tests', 'training', 'inference'))
                result = subprocess.run([sys.executable, '-m', 'unittest', '-v', __name__ +
                    '.TinyPilotTests.test_real_native_event_witness_and_endpoint_guard'],
                    cwd=root, env=environment, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, current + '\n' + result.stdout + result.stderr)
            return
        config = self.root / 'pilot.yaml'
        config.write_text(C.build_config(self.original_config_text, arm=arm,
            output_dir=self.root / 'common', maximum_events=1))
        trainer = T.PilotTrainer(config, self.root / 'pilot', arm=arm, binding=BINDING,
            original_config_text=self.original_config_text, maximum_events=1, expected_cuda_rng_states=0)
        self.addCleanup(trainer.close)
        released = self.root / 'released'
        released.mkdir()
        save_file(dict(trainer.bundle.draft_model.state_dict()), released / 'model.safetensors')
        trainer.save_initial()
        if arm == 'BASE':
            from experiments.repair_conditional_kl_pilot.witness import freeze_witness, evaluate_witness
            before = capture_rng()
            witness = self.root / 'witness.pt'
            freeze_witness(trainer, self.features, witness)
            evaluate_witness(trainer, witness, self.root / 'origin.json', endpoint='origin')
            self.assertFalse(T.differences(before, capture_rng()))
            import json
            from unittest.mock import patch
            before_work = deepcopy(trainer.work)
            failure_path = self.root / 'witness_failure.json'
            with patch.object(trainer.bundle.model.lm_head, 'forward',
                              side_effect=RuntimeError('injected diagnostic head failure')):
                with self.assertRaisesRegex(RuntimeError, 'injected diagnostic'):
                    evaluate_witness(trainer, witness, failure_path, endpoint='failed_origin')
            failure = json.loads(failure_path.read_text())
            self.assertEqual(failure['status'], 'failed')
            self.assertEqual(failure['diagnostic_draft_forwards'], 1)
            self.assertEqual(failure['diagnostic_head_calls_attempted'], 1)
            self.assertEqual(failure['diagnostic_head_calls'], 0)
            self.assertGreater(failure['physical_work']['draft_forwards'], 0)
            self.assertEqual(trainer.work, before_work)
            self.assertFalse(T.differences(before, capture_rng()))
        receipt = trainer.train_event('event_001', self.features)
        self.assertEqual(receipt['optimizer_steps'], 128)
        trainer.verify_boundary(receipt['export_path'])
        metrics = receipt['records'][0]['response_observation']['ratio_metrics']
        self.assertEqual('repair_pair_auxiliary' in metrics, arm == 'PAIR')
        if arm == 'BASE':
            evaluate_witness(trainer, witness, self.root / 'endpoint.json',
                             origin_path=self.root / 'origin.json', endpoint='BASE1')
        with self.assertRaisesRegex(ValueError, 'endpoint'):
            trainer.train_event('event_002', self.features)
        trainer.close()
        from experiments.repair_conditional_kl_pilot.qualification import run_training_qualification
        original = self.root / 'original.yaml'
        original.write_text(self.original_config_text)
        qualified = run_training_qualification(dict(root=str(self.root), binding=BINDING,
            execution_commit='cpu_fixture', configs={arm: str(config)}, original_config=str(original),
            maximum_events=1, expected_cuda_rng_states=0), arm, feature_paths=self.features[:2])
        self.assertTrue(qualified['exact_export_and_state_reload'])
        self.assertEqual(qualified['counts']['adam_completed'], 1)
        self.assertEqual(qualified['startup_counts']['adam_completed'], 0)
        self.assertFalse(torch.cuda.is_initialized())
