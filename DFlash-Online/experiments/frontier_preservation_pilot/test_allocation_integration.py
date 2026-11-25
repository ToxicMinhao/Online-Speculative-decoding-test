"""Actual CPU native TrainerCore/AdamW replay and disposable restoration tests."""

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import MethodType, SimpleNamespace
import unittest
from unittest import mock

import torch

from experiments.repair_conditional_kl_pilot.training import PilotTrainer
from online_feedback.objective import OSDTrainStrategy
from online_feedback.optimizer import DirectBF16AdamW
from online_feedback.prefix_allocation import PrefixAllocation
from online_feedback.rng import capture_rng, restore_rng
from online_feedback.training import ReductionAdapter
from specforge.runtime.contracts import TrainBatch
from specforge.training.controller import TrainerCore
from .allocation_integration import (
    DisposableStepLedger,
    qualify_integration,
    train_window,
    qualification_cached_contract,
)
from .allocation_scoring import NativeAllocationScorer
from .native import WorkReceipt, freeze_two_responses, rng_digest
from .test_allocation_scoring import TinyNativeModel, response


class CPUBackend:
    def __init__(self, model):
        self.optimizer = DirectBF16AdamW(model.draft_model, learning_rate=0.003)
        self.parallel_config = None
        for parameter in model.draft_model.parameters():
            self.optimizer.optimizer.state[parameter] = dict(
                step=torch.tensor(128.0),
                exp_avg=torch.full_like(parameter, 0.002),
                exp_avg_sq=torch.full_like(parameter, 0.001),
            )

    def backward(self, loss, is_boundary=False):
        loss.backward()

    def scale_gradients(self, scale):
        torch._foreach_mul_([p.grad for p in self.optimizer.model_params], scale)

    def step(self):
        return self.optimizer.step()

    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state['optimizer'])


def fixture(root):
    root = Path(root)
    model = TinyNativeModel()
    model.prefix_allocation = PrefixAllocation()
    model.draft_model.config = SimpleNamespace(attention_dropout=0.0)
    backend = CPUBackend(model)
    trainer = SimpleNamespace(
        bundle=SimpleNamespace(model=model, draft_model=model.draft_model),
        backend=backend,
        global_step=128,
        presentations=256,
        events=[],
        horizon=7424,
        arm='PAIR',
        qualification_only=True,
        response_records=[],
        window_records=[],
        work={},
        _anchor_observations=[],
        _active_event=None,
        _presentation_schedule=None,
        events_root=root / 'events',
    )
    trainer.events_root.mkdir()
    trainer.check_installed_arm = MethodType(PilotTrainer.check_installed_arm, trainer)
    original_sampler = model._sample_anchor_positions
    trainer.sampler_calls = 0

    def observed_sampler(*args, **kwargs):
        trainer.sampler_calls += 1
        torch.rand(())
        anchors, keep = original_sampler(*args, **kwargs)
        trainer._anchor_observations.append(
            dict(positions=anchors.tolist(), keep_mask=keep.tolist())
        )
        return anchors, keep

    model._sample_anchor_positions = observed_sampler
    author = OSDTrainStrategy(model)
    original_forward = author.forward_loss

    def checked_forward(batch, ctx):
        trainer.check_installed_arm()
        result = original_forward(batch, ctx)
        trainer.response_records.append(
            dict(loss=float(result.loss.detach()), record_id=batch.sample_ids[0])
        )
        return result

    author.forward_loss = checked_forward
    trainer.strategy = ReductionAdapter(author, 'response_mean')
    trainer.core = TrainerCore(trainer.strategy, backend, accumulation_steps=2)
    trainer.core._micro = 256
    trainer._batch = lambda raw: TrainBatch(
        sample_ids=[raw['record_id']],
        strategy='dflash',
        tensors={name: value for name, value in raw.items() if isinstance(value, torch.Tensor)},
    )
    features = []
    for index, blocks in enumerate((10, 3)):
        item = response(blocks, index)
        size = item.tensors['input_ids'].shape[1]
        positions = torch.arange(size).remainder(16).to(torch.bfloat16)
        item.tensors['hidden_states'][..., 2] = positions / 64
        item.tensors['hidden_states'][..., 0] += positions / 32
        raw = dict(item.tensors, record_id=f'private_fixture_{index}')
        path = root / f'feature_{index}.pt'
        torch.save(raw, path)
        features.append(dict(path=str(path), record_id=raw['record_id']))
    state = dict(
        draft_state_dict=deepcopy(model.draft_model.state_dict()),
        backend=dict(optimizer=deepcopy(backend.optimizer.state_dict())),
        global_step=128,
        presentations=256,
        events=[],
        rng=capture_rng(),
    )
    responses = freeze_two_responses(trainer, [f['path'] for f in features])
    scorer = NativeAllocationScorer(trainer, responses)
    snapshots = scorer.freeze_current()
    gradient = scorer.reference_gradient()
    preview = scorer.preview_and_direction(gradient)
    fast = scorer.fast_scores(preview['w'])
    return trainer, state, features, responses, snapshots, gradient, fast['allocations']


def verify_parent(trainer, state, path):
    for name, value in trainer.bundle.draft_model.state_dict().items():
        if not torch.equal(value, state['draft_state_dict'][name]):
            raise AssertionError('Parent parameters changed')
    actual = trainer.backend.optimizer.state_dict()['optimizer_state_dict']
    expected = state['backend']['optimizer']['optimizer_state_dict']
    for key, values in expected['state'].items():
        for name, value in values.items():
            if not torch.equal(value, actual['state'][key][name]):
                raise AssertionError('Parent Adam state changed')
    assert trainer.global_step == state['global_step']
    assert trainer.presentations == state['presentations']
    assert trainer.core._micro == state['presentations']
    assert not trainer.core._ratio_totals and not trainer.core._sum_totals
    assert rng_digest(capture_rng()) == rng_digest(state['rng'])
    assert all(p.grad is None for p in trainer.bundle.draft_model.parameters())
    return dict(passed=True, parent_restored=True)


class AllocationIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def test_four_native_windows_exact_disabled_g0_parity_and_active_bf16_update(self):
        with TemporaryDirectory() as directory:
            trainer, state, features, responses, snapshots, gradient, allocations = fixture(
                directory
            )
            source = deepcopy(state)
            work = WorkReceipt(Path(directory) / 'WORK.json')
            result = qualify_integration(
                trainer,
                state,
                features,
                responses,
                snapshots,
                allocations,
                gradient,
                work,
                Path(directory) / 'qualification',
                verify=verify_parent,
            )
            self.assertTrue(result['passed'], result)
            self.assertTrue(result['control_disabled_exact'])
            self.assertTrue(result['beta_g0_exact'])
            self.assertTrue(result['activity']['bf16_parameters_changed'])
            self.assertEqual(work.value['native_adam_steps'], 4)
            self.assertFalse(work.value['pending'])
            self.assertEqual(trainer.sampler_calls, 6)  # freeze2 + CONTROL2 + DISABLED2.
            verify_parent(trainer, state, None)
            for key, values in source['backend']['optimizer']['optimizer_state_dict'][
                'state'
            ].items():
                for name, value in values.items():
                    self.assertTrue(
                        torch.equal(
                            value,
                            state['backend']['optimizer']['optimizer_state_dict']['state'][key][
                                name
                            ],
                        )
                    )

    def test_disposable_ledger_counts_unknown_attempts_and_enforces_global_limits(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'STEP_LEDGER.json'
            first = DisposableStepLedger(
                path, {'sha256': 'fixed-window'}, Path(directory) / 'first'
            )
            for index in range(4):
                row = first.reserve(str(index))
                if index:
                    first.complete(row)
            with self.assertRaisesRegex(ValueError, 'already attempted'):
                first.reserve('0')
            retry = DisposableStepLedger(
                path, {'sha256': 'fixed-window'}, Path(directory) / 'retry'
            )
            for index in range(4):
                retry.complete(retry.reserve(str(index)))
            self.assertEqual(retry.value['steps'][0]['status'], 'attempted')
            with self.assertRaisesRegex(ValueError, 'Eight-step'):
                retry.reserve('fifth')
            with self.assertRaisesRegex(ValueError, 'four unused'):
                DisposableStepLedger(
                    path, {'sha256': 'fixed-window'}, Path(directory) / 'third'
                )
            other = Path(directory) / 'WINDOW_LEDGER.json'
            DisposableStepLedger(other, {'sha256': 'one'}, 'one')
            DisposableStepLedger(other, {'sha256': 'two'}, 'two')
            with self.assertRaisesRegex(ValueError, 'Two-window'):
                DisposableStepLedger(other, {'sha256': 'three'}, 'three')

    def test_interrupted_native_step_spends_allowance_and_restores_parent(self):
        with TemporaryDirectory() as directory:
            trainer, state, features, responses, snapshots, gradient, allocations = fixture(
                directory
            )
            original = trainer.backend.optimizer.step
            work = WorkReceipt(Path(directory) / 'WORK.json')
            ledger = DisposableStepLedger(
                Path(directory) / 'STEPS.json', {'sha256': 'fixed'}, directory
            )

            def interrupted_step():
                original()
                raise RuntimeError('fixture interruption after Adam')

            with mock.patch.object(
                trainer.backend.optimizer, 'step', side_effect=interrupted_step
            ):
                with self.assertRaisesRegex(RuntimeError, 'fixture interruption'):
                    qualify_integration(
                        trainer,
                        state,
                        features,
                        responses,
                        snapshots,
                        allocations,
                        gradient,
                        work,
                        Path(directory) / 'qualification',
                        verify=verify_parent,
                        step_ledger=ledger,
                    )
            verify_parent(trainer, state, None)
            self.assertEqual(len(ledger.value['steps']), 1)
            self.assertEqual(ledger.value['steps'][0]['status'], 'attempted')
            self.assertIn('disposable_native_adam_step', work.value['pending'])

    def test_cached_window_samples_no_more_and_uses_saved_observer_receipt(self):
        with TemporaryDirectory() as directory:
            trainer, state, features, responses, snapshots, gradient, allocations = fixture(
                directory
            )
            before = trainer.sampler_calls
            original = trainer.bundle.model.prefix_allocation
            with qualification_cached_contract(trainer):
                result = train_window(
                    trainer,
                    features,
                    enabled=True,
                    responses=responses,
                    snapshots=snapshots,
                    coefficients=allocations,
                )
            self.assertEqual(trainer.sampler_calls, before)
            self.assertIs(trainer.bundle.model.prefix_allocation, original)
            self.assertEqual(
                result['window']['end_rng_sha256'], rng_digest(responses[-1].rng_after_sampling)
            )
            for row, item in zip(result['rows'], responses):
                self.assertEqual(
                    row['sampled_anchors'],
                    [dict(positions=item.anchors.tolist(), keep_mask=item.keep.tolist())],
                )
            self.assertEqual((trainer.global_step, trainer.presentations), (129, 258))

    def test_failure_midwindow_restores_parent_and_original_hooks(self):
        with TemporaryDirectory() as directory:
            trainer, state, features, responses, snapshots, gradient, allocations = fixture(
                directory
            )
            work = WorkReceipt(Path(directory) / 'WORK.json')
            original = trainer.core.train_step
            calls = []

            def fail_second(*args, **kwargs):
                calls.append(1)
                if len(calls) == 2:
                    raise RuntimeError('fixture midwindow failure')
                return original(*args, **kwargs)

            with mock.patch.object(trainer.core, 'train_step', side_effect=fail_second):
                with self.assertRaisesRegex(RuntimeError, 'fixture midwindow failure'):
                    qualify_integration(
                        trainer,
                        state,
                        features,
                        responses,
                        snapshots,
                        allocations,
                        gradient,
                        work,
                        Path(directory) / 'qualification',
                        verify=verify_parent,
                    )
            verify_parent(trainer, state, None)
            self.assertEqual(work.value['native_adam_steps'], 0)

    def test_disabled_path_rejects_cache_and_cached_scope_rejects_foreign_extensions(self):
        with TemporaryDirectory() as directory:
            trainer, state, features, responses, snapshots, gradient, allocations = fixture(
                directory
            )
            with self.assertRaisesRegex(ValueError, 'Disabled training'):
                train_window(trainer, features, responses=responses)
            original = trainer.check_installed_arm
            with self.assertRaisesRegex(ValueError, 'other objective hooks'):
                with qualification_cached_contract(trainer):
                    from .allocation_replay import replay_allocation

                    with replay_allocation(trainer.bundle.model, snapshots[0], allocations[0]):
                        trainer.bundle.model.block_observer = object()
                        trainer.check_installed_arm()
            trainer.bundle.model.block_observer = None
            self.assertEqual(trainer.check_installed_arm, original)


if __name__ == '__main__':
    unittest.main()
