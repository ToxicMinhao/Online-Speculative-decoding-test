"""CPU regressions for nested endpoint checks and post-step failure evidence."""
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from experiments.kplus_abt import qualification
from experiments.ok_coding.storage import identity
from experiments.ok_continuation.resume import differences
from online_feedback.rng import capture_rng
from online_feedback.storage import save_torch, write_json


def nested_endpoint():
    """Use the native wrapper shape, including Adam tensors and scalar options."""
    weights = {
        'weight': torch.tensor([[1., 2.], [3., 4.]], dtype=torch.bfloat16),
        'unused': torch.tensor([5., 6.], dtype=torch.bfloat16),
    }
    moments = {
        index: dict(step=torch.tensor(513.), exp_avg=torch.full_like(value, .125),
                    exp_avg_sq=torch.full_like(value, .25))
        for index, value in enumerate(weights.values())
    }
    options = dict(lr=.0003, betas=(.9, .999), eps=1e-8, weight_decay=.01,
                   amsgrad=False, foreach=None, maximize=False, capturable=False,
                   differentiable=False, fused=None)
    optimizer = dict(
        recipe=dict(type='direct_bf16_adamw', version=1, max_grad_norm=1.,
                    parameters=[(name, list(value.shape), str(value.dtype))
                                for name, value in weights.items()],
                    optimizer_options=options),
        optimizer_state_dict=dict(state=moments, param_groups=[dict(options, params=[0, 1])]),
    )
    return dict(weights=weights, optimizer=optimizer,
                gradients=dict(weight=torch.full_like(weights['weight'], .5), unused=None))


class EndpointFiniteTests(unittest.TestCase):
    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def test_native_nested_state_passes_without_mutation_or_rng_change(self):
        saved = nested_endpoint()
        before, rng = deepcopy(saved), capture_rng()
        result = qualification.finite_endpoint(saved)
        self.assertEqual(result, dict(status='passed', tensor_leaves_checked=dict(
            weights=2, optimizer=6, gradients=1)))
        self.assertEqual(differences(saved, before), [])
        self.assertEqual(differences(capture_rng(), rng), [])
        self.assertIsNone(saved['gradients']['unused'])

    def test_tensor_leaves_in_lists_and_tuples_are_checked(self):
        saved = nested_endpoint()
        saved['optimizer']['optimizer_state_dict']['param_groups'][0]['lr'] = torch.tensor(.0003)
        saved['optimizer']['recipe']['optimizer_options']['betas'] = (
            torch.tensor(.9), torch.tensor(float('inf')))
        before = deepcopy(saved)
        with self.assertRaisesRegex(ValueError, r'Nonfinite optimizer\.recipe\.optimizer_options\.betas\.1'):
            qualification.finite_endpoint(saved)
        self.assertEqual(differences(saved, before), [])

    def test_nonfinite_leaf_identifies_each_scientific_field_without_mutation(self):
        cases = (
            ('weights.weight', ('weights', 'weight'), float('nan')),
            ('optimizer.optimizer_state_dict.state.0.exp_avg',
             ('optimizer', 'optimizer_state_dict', 'state', 0, 'exp_avg'), float('inf')),
            ('gradients.weight', ('gradients', 'weight'), -float('inf')),
        )
        for name, keys, value in cases:
            with self.subTest(field=name):
                saved = nested_endpoint()
                leaf = saved
                for key in keys:
                    leaf = leaf[key]
                leaf.flatten()[0] = value
                before = deepcopy(saved)
                with self.assertRaises(ValueError) as raised:
                    qualification.finite_endpoint(saved)
                self.assertEqual(str(raised.exception), 'Nonfinite ' + name)
                self.assertEqual(differences(saved, before), [])

    def test_unknown_leaf_is_not_silently_ignored(self):
        saved = nested_endpoint()
        saved['optimizer']['unexpected'] = object()
        with self.assertRaisesRegex(TypeError, 'optimizer.unexpected'):
            qualification.finite_endpoint(saved)


class CPUTrainer:
    """Exercise run_case's real save/check/failure order without model execution."""
    def __init__(self, output, saved, *, bad_counter=False):
        self.output_root = output
        self.events_root = output/'events'
        self.events_root.mkdir(parents=True)
        write_json(output/'RESUME_CHECK.json', dict(status='passed'))
        draft = torch.nn.Module()
        for name, value in saved['weights'].items():
            parameter = torch.nn.Parameter(value.clone())
            parameter.grad = saved['gradients'][name]
            draft.register_parameter(name, parameter)
        self.bundle = SimpleNamespace(draft_model=draft, model=SimpleNamespace(begin_event=lambda: None))
        self.native_steps = 0
        optimizer = SimpleNamespace(step=self.step)
        self.backend = SimpleNamespace(
            optimizer=SimpleNamespace(optimizer=optimizer, state_dict=lambda: deepcopy(saved['optimizer'])),
            backward=lambda: None)
        self.core = SimpleNamespace(strategy=SimpleNamespace(forward_loss=lambda: None),
                                    train_step=self.train_step, _micro=1024, accumulation_remainder=0)
        self._anchor_observations = []
        self.response_records, self.window_records, self.view_records = [], [], []
        self.work = {}
        self.presentations, self.global_step = 1024, 511 if bad_counter else 512
        self.horizon = 7424
        self.failed, self.closed = False, False

    def step(self):
        self.native_steps += 1

    def install_two_views(self, **kwargs):
        return False

    def _batch(self, raw):
        return raw

    def train_step(self, batch, context):
        self.core.strategy.forward_loss()
        self.backend.backward()
        self.core._micro += 1
        self.response_records.append(dict(sampled_anchors=[dict(positions=[[1]], keep_mask=[[True]])], loss=.25))
        stepped = len(self.response_records) == 2
        if stepped:
            self.backend.optimizer.optimizer.step()
            self.window_records.append(dict(global_step=513))
        return SimpleNamespace(optimizer_stepped=stepped, materialize_metrics=lambda: None)

    def close(self):
        self.closed = True


class EndpointFailurePersistenceTests(unittest.TestCase):
    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def run_failure(self, failure):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            features = [root/f'feature_{index}.pt' for index in range(2)]
            for feature in features:
                save_torch(feature, dict(tokens=torch.tensor([1, 2])))
            preparation = root/'PREPARED.json'
            write_json(preparation, dict(datasets=dict(gsm8k=[dict(feature_path=str(path)) for path in features])))
            config = root/'config.yaml'
            config.write_text('fixture: true\n')
            write_json(root/'sealed/PLAN.json', dict(
                shared_feedback_preparation=str(preparation), original_config=str(config),
                execution_source=dict(commit='fixture'), binding='fixture',
                datasets=dict(gsm8k=dict(configs=dict(KPLUS=str(config), ADV=str(config)),
                                        origin_state='unused', origin_export='unused'))))
            saved = nested_endpoint()
            if failure == 'finite':
                saved['optimizer']['optimizer_state_dict']['state'][0]['exp_avg'][0, 0] = float('nan')
            case = 'DISABLED' if failure == 'comparison' else 'KPLUS'
            if case == 'DISABLED':
                save_torch(root/'qualification/gsm8k/KPLUS/ENDPOINT.pt', saved)
            output = root/'qualification/gsm8k'/case
            trainers = []

            def make_trainer(config_path, trainer_output, **kwargs):
                trainer = CPUTrainer(trainer_output, saved, bad_counter=failure == 'counter')
                trainers.append(trainer)
                return trainer

            with ExitStack() as stack:
                stack.enter_context(patch.object(qualification, 'AdversarialBlockTrainer', side_effect=make_trainer))
                stack.enter_context(patch.object(qualification, 'pin_retained_graph_runtime'))
                stack.enter_context(patch.object(qualification, 'discarded_startup', return_value=dict(status='fixture')))
                stack.enter_context(patch.object(torch.cuda, 'reset_peak_memory_stats'))
                if failure == 'work':
                    stack.enter_context(patch.object(qualification, 'require_complete_work',
                                                     side_effect=ValueError('forced work gate')))
                if failure == 'comparison':
                    stack.enter_context(patch.object(qualification, 'persist_comparison',
                                                     side_effect=ValueError('forced comparison gate')))
                with self.assertRaises(ValueError):
                    qualification.run_case(root, 'gsm8k', case)

            receipt = qualification.read(output/'RESULT.json')
            retained = torch.load(output/'ENDPOINT.pt', map_location='cpu', weights_only=False)
            self.assertEqual(receipt['status'], 'failed')
            self.assertEqual(receipt['endpoint_identity'], identity(output/'ENDPOINT.pt'))
            expected_status = {'finite': 'failed', 'comparison': 'passed'}.get(failure, 'unchecked')
            self.assertEqual(receipt['endpoint_finite']['status'], expected_status)
            self.assertEqual(differences({field: retained[field] for field in saved}, saved), [])
            self.assertEqual(receipt['work']['completed'], dict(response_forwards=2, response_backwards=2,
                                                               optimizer_steps=1))
            self.assertEqual(receipt['work']['unknown'], dict(response_forwards=0, response_backwards=0,
                                                             optimizer_steps=0))
            self.assertEqual(len(receipt['observations']), 2)
            self.assertEqual(len(receipt['windows']), 1)
            self.assertEqual(trainers[0].native_steps, 1)
            self.assertTrue(trainers[0].failed)
            self.assertTrue(trainers[0].closed)
            self.assertFalse(receipt['automatic_retry'])
            if failure == 'finite':
                self.assertIn('optimizer.optimizer_state_dict.state.0.exp_avg', receipt['error'])

    def test_nonfinite_failure_retains_endpoint_and_exact_error_path(self):
        self.run_failure('finite')

    def test_work_gate_failure_retains_unchecked_endpoint(self):
        self.run_failure('work')

    def test_state_counter_failure_retains_unchecked_endpoint(self):
        self.run_failure('counter')

    def test_comparison_failure_retains_validated_endpoint(self):
        self.run_failure('comparison')


if __name__ == '__main__':
    unittest.main()
