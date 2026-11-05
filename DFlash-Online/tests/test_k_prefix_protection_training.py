"""CPU scientific-state and qualification checks, independent of GPU execution."""
from copy import deepcopy
from pathlib import Path
import random
import json
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch
import yaml

from experiments.ok_coding import training as old
from experiments.ok_continuation.test_resume import draft, event, fixture, runtime
from experiments.k_prefix_protection_v1 import training as T
from experiments.k_prefix_protection_v1 import qualification as Q
from online_feedback.optimizer import DirectBF16AdamW
from online_feedback.rng import capture_rng, restore_rng
from online_feedback.storage import save_torch

BINDING = dict(upstream_data_binding='sealed-original-coding', execution_commit='a'*40)


def old_contract(seed):
    return dict(training_seed=seed, data_binding='sealed-original-coding', events=[1, 2, 3, 4])


def config(seed, arm='KP'):
    raw = yaml.safe_load((Path(__file__).parents[1]/'configs/osd.yaml').read_text())
    raw['training']['seed'] = seed
    raw['training']['learning_rate'] = T.LEARNING_RATES[arm]
    raw['data']['max_length'] = 2272
    raw['model']['draft_checkpoint_path'] = '/released'
    return yaml.safe_dump(raw)


def source(seed=17, completed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if completed:
        value = fixture('osd_first_error', completed)
    else:
        model = draft()
        optimizer = DirectBF16AdamW(model)
        value = dict(event_schema=T.SCHEMA, draft_state_dict=dict(deepcopy(model.state_dict())),
            backend={'optimizer': deepcopy(optimizer.state_dict())}, rng=capture_rng(),
            global_step=0, presentations=0, events=[], strategy='dflash', run_id='fixture',
            world_size=1, reduction='response_mean', accumulation_steps=2, horizon=7424,
            algorithm_contract={'online_target_objective': {'name': 'osd_first_error'}})
    value['config_text'] = config(seed)
    value['algorithm_contract'][old.CONTRACT_KEY] = old.coding_contract(seed, old_contract(seed))
    return value


def migrated(seed=17, arm='KP', completed=0, qualification=None):
    original = source(seed, completed)
    contract = deepcopy(original['algorithm_contract'])
    contract[T.CONTRACT_KEY] = T.experiment_contract(arm, seed, BINDING, qualification=qualification)
    candidate, receipt = T.migrate_source(original, arm=arm, seed=seed, binding=BINDING,
        original_contract=old_contract(seed), config_text=config(seed, arm), new_contract=contract,
        completed_event=completed, qualification=qualification, expected_cuda_rng_states=0)
    return original, candidate, receipt


def validate(state, seed=17, arm='KP', completed=0, qualification=None, steps=0):
    return T.validate_snapshot(state, arm=arm, seed=seed, binding=BINDING,
        original_contract=old_contract(seed), config_text=config(seed, arm), completed_event=completed,
        qualification=qualification, qualification_steps=steps, expected_cuda_rng_states=0)


class NewStateContracts(unittest.TestCase):
    def setUp(self):
        state = capture_rng()
        self.addCleanup(lambda: restore_rng(state))

    def test_initial_migration_changes_only_declared_lr_and_metadata(self):
        for seed in (17, 29):
            for arm in T.ARMS:
                original, value, receipt = migrated(seed, arm)
                self.assertTrue(receipt['source_weights_rng_moments_counters_history_exact'])
                self.assertEqual(value['backend']['optimizer']['optimizer_state_dict']['state'], {})
                self.assertFalse(old.inherited.differences(original['draft_state_dict'], value['draft_state_dict']))
                self.assertFalse(old.inherited.differences(original['rng'], value['rng']))
                self.assertEqual(value['global_step'], 0)
                self.assertEqual(original['backend']['optimizer']['recipe']['optimizer_options']['lr'], 3e-4)
                self.assertEqual(value['backend']['optimizer']['recipe']['optimizer_options']['lr'], T.LEARNING_RATES[arm])
                validate(value, seed, arm)

    def test_old_config_and_snapshot_guards_still_reject_half_lr(self):
        _, value, _ = migrated(arm='K_HALF')
        with self.assertRaisesRegex(ValueError, 'learning_rate'):
            old.validate_config_text(value['config_text'], 17)
        with self.assertRaises(ValueError):
            old.validate_snapshot(value, arm='K', seed=17, campaign_contract=old_contract(17),
                config_text=value['config_text'], completed_event=0, expected_cuda_rng_states=0)

    def test_own_completed_events_and_own_optimizer_history_validate(self):
        for arm in T.ARMS:
            for number in range(1, 5):
                _, value, _ = migrated(arm=arm)
                historical = source(completed=number)
                value.update(draft_state_dict=historical['draft_state_dict'],
                    global_step=number*128, presentations=number*256, events=historical['events'],
                    backend=deepcopy(historical['backend']))
                optimizer = value['backend']['optimizer']
                optimizer['recipe']['optimizer_options']['lr'] = T.LEARNING_RATES[arm]
                optimizer['optimizer_state_dict']['param_groups'][0]['lr'] = T.LEARNING_RATES[arm]
                checked = validate(value, arm=arm, completed=number)
                self.assertEqual(checked['global_step'], number*128)
                self.assertEqual(checked['adam_state_entries'], 58)

    def test_no_weights_only_reset_wrong_arm_or_stale_reference_contract(self):
        _, value, _ = migrated()
        mutations = [lambda v: v['backend'].clear(), lambda v: v['rng'].pop('numpy'),
            lambda v: v['algorithm_contract'][T.CONTRACT_KEY].update(reference='event_current'),
            lambda v: v['algorithm_contract'][T.CONTRACT_KEY].update(arm='K_HALF'),
            lambda v: v.update(presentations=1), lambda v: v['events'].append(event())]
        for mutate in mutations:
            broken = deepcopy(value)
            mutate(broken)
            with self.assertRaises((ValueError, KeyError)):
                validate(broken)

    def test_disposable_counter_is_513_not_fake_event5(self):
        qual = dict(case='Q1', source='original_K17_event4')
        _, value, _ = migrated(completed=4, qualification=qual)
        value['global_step'] += 1
        value['presentations'] += 2
        for moments in value['backend']['optimizer']['optimizer_state_dict']['state'].values():
            moments['step'] += 1
        checked = validate(value, completed=4, qualification=qual, steps=1)
        self.assertEqual(checked['global_step'], 513)
        self.assertEqual(len(value['events']), 4)
        with self.assertRaises(ValueError):
            validate(value, completed=4, qualification=qual)
        with self.assertRaises(ValueError):
            validate(value, completed=5, qualification=qual, steps=1)

    def test_production_cannot_migrate_trained_original_k(self):
        original = source(completed=4)
        contract = deepcopy(original['algorithm_contract'])
        contract[T.CONTRACT_KEY] = T.experiment_contract('KP', 17, BINDING)
        with self.assertRaisesRegex(ValueError, 'step-zero'):
            T.migrate_source(original, arm='KP', seed=17, binding=BINDING,
                original_contract=old_contract(17), config_text=config(17), new_contract=contract,
                completed_event=4, expected_cuda_rng_states=0)

    def test_atomic_serialization_and_decoded_restore_preserve_rng(self):
        for arm in T.ARMS:
            _, value, _ = migrated(arm=arm)
            trainer = runtime(value)
            trainer.__class__ = T.PrefixProtectionTrainer
            trainer.backend.optimizer = DirectBF16AdamW(trainer.bundle.draft_model,
                                                        learning_rate=T.LEARNING_RATES[arm])
            trainer.backend.load_state_dict = lambda state: trainer.backend.optimizer.load_state_dict(state['optimizer'])
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder)/'state.pt'
                save_torch(path, value)
                decoded = torch.load(path, map_location='cpu', weights_only=False)
                trainer._load_decoded(decoded)
                self.assertFalse(old.inherited.differences(value['rng'], capture_rng()))
                self.assertFalse(old.inherited.differences(value['draft_state_dict'], dict(trainer.bundle.draft_model.state_dict())))
                self.assertFalse(old.inherited.differences(value['backend']['optimizer'], trainer.backend.optimizer.state_dict()))
                self.assertEqual(trainer.core._micro, 0)
                self.assertFalse(path.with_name(path.name+'.tmp').exists())


class QualificationChecks(unittest.TestCase):
    def test_exact_four_cases_and_work_accounting(self):
        self.assertEqual([row[0] for row in Q.CASES], ['Q1', 'Q2', 'Q3', 'Q4'])
        self.assertEqual(sum(row[2] for row in Q.CASES), 6)
        self.assertEqual(Q.LIMITS, dict(forwards=8, backwards=8, reference_forwards=6, optimizer_steps=4))

    def test_qualification_uses_four_cold_processes_and_aggregates_real_work(self):
        from experiments.ok_coding.campaign import WorkLedger
        from online_feedback.storage import write_json
        seen = []
        row = dict(loss=1., ratio_metrics={}, sum_metrics={})
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            def execute(command, **kwargs):
                name = command[-1]
                directory = root/'qualification_001'/name
                seen.append((name, kwargs['env']['TORCHINDUCTOR_CACHE_DIR']))
                self.assertTrue(Path(seen[-1][1]).is_dir())
                self.assertEqual(list(Path(seen[-1][1]).iterdir()), [])
                references = next(n for case, _, n in Q.CASES if case == name)
                ledger = WorkLedger(directory/'WORK.jsonl', Q.case_limits(references))
                for kind, count in ledger.limits.items():
                    for _ in range(count):
                        ledger.call(kind, lambda: None)
                state = source(completed=4)
                gradients = {'a': torch.ones(2, dtype=torch.bfloat16)}
                if name == 'Q4':
                    gradients['a'][0] += 1
                    values = state['backend']['optimizer']['optimizer_state_dict']['state']
                    next(iter(values.values()))['exp_avg'][0, 0] += 1
                save_torch(directory/'POST_STATE.pt', state)
                save_torch(directory/'NORMALIZED_PRECLIP_GRADIENTS.pt', gradients)
                write_json(directory/'RESPONSES.json', dict(responses=[row, row], anchors=[[[1]], [[3]]]))
                write_json(directory/'CASE.json', dict(status='completed', startup={'device_uuid': 'GPU-fixture'}))
                return SimpleNamespace(returncode=0)
            with mock.patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': '0'}), mock.patch.object(Q.subprocess, 'run', side_effect=execute):
                result = Q.run(root)
            self.assertEqual(result['status'], 'passed')
            self.assertEqual([name for name, _ in seen], ['Q1', 'Q2', 'Q3', 'Q4'])
            self.assertEqual(len({cache for _, cache in seen}), 4)
            self.assertEqual(result['work']['completed'], Q.LIMITS)
            self.assertFalse(torch.cuda.is_initialized())
            self.assertEqual(json.loads((root/'qualification_001/RESULT.json').read_text())['status'], 'passed')

    def test_preclip_observer_rejects_all_none_and_preserves_native_dtype(self):
        model = draft()
        trainer = SimpleNamespace(bundle=SimpleNamespace(draft_model=model))
        with self.assertRaisesRegex(ValueError, 'all-None'):
            Q.normalized_gradients(trainer)
        first = next(model.parameters())
        first.grad = torch.ones_like(first)
        gradients = Q.normalized_gradients(trainer)
        self.assertEqual(next(value.dtype for value in gradients.values() if value is not None), torch.bfloat16)
        self.assertEqual(sum(value is None for value in gradients.values()), 57)
        first.grad.fill_(float('nan'))
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            Q.normalized_gradients(trainer)

    def test_neutral_gate_detects_gradient_loss_counter_and_rng_difference(self):
        original = source(completed=4)
        row = dict(loss=1., ratio_metrics={'loss': [1., 1.]}, sum_metrics={'positions': 2.})
        baseline = dict(state=original, gradients={'a': torch.ones(2, dtype=torch.bfloat16)},
                        responses=[row, row], anchors=[[[1]], [[3]]])
        for kind in ('gradient', 'loss', 'counter', 'rng', 'none'):
            other = deepcopy(baseline)
            if kind == 'gradient': other['gradients']['a'][0] += 1
            elif kind == 'loss': other['responses'][0]['loss'] = 2.
            elif kind == 'counter': other['state']['global_step'] += 1
            elif kind == 'rng': other['state']['rng']['torch'][0] ^= 1
            else: other['gradients']['a'] = None
            comparison = Q.compare_cases(baseline, other)
            with self.assertRaisesRegex(ValueError, 'neutral wrapper differs'):
                Q.require_neutral_equal(comparison)
        Q.require_neutral_equal(Q.compare_cases(baseline, deepcopy(baseline)))

    def test_q4_requires_gradients_and_moments_not_parameter_change(self):
        original = source(completed=4)
        row = dict(loss=1., ratio_metrics={}, sum_metrics={})
        baseline = dict(state=original, gradients={'a': torch.ones(2, dtype=torch.bfloat16)},
                        responses=[row, row], anchors=[[[1]], [[3]]])
        other = deepcopy(baseline)
        other['gradients']['a'][0] += 1
        with self.assertRaisesRegex(ValueError, 'both accumulated'):
            Q.require_forced_active(baseline, other, Q.compare_cases(baseline, other))
        moments = other['state']['backend']['optimizer']['optimizer_state_dict']['state']
        next(iter(moments.values()))['exp_avg'][0, 0] += 1
        result = Q.require_forced_active(baseline, other, Q.compare_cases(baseline, other))
        self.assertTrue(result['updated_bf16_parameters']['exact'])
        self.assertFalse(result['moments']['exact'])


if __name__ == '__main__':
    unittest.main()
