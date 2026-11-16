"""Bounded qualification identities and entry-point failure accounting."""
from copy import deepcopy
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from experiments.persistent_prefix_withholding import qualification as Q
from experiments.persistent_prefix_withholding import qualification_entry as E


class QualificationTests(unittest.TestCase):
    @staticmethod
    def optimizer_fixture():
        generator = torch.Generator().manual_seed(17)
        initial = (torch.randn(4096, generator=generator) * .05).bfloat16()
        gradient = (torch.randn(4096, generator=generator) * .2).bfloat16()
        parameter = torch.nn.Parameter(initial.clone())
        optimizer = torch.optim.AdamW([parameter], lr=3e-4, weight_decay=.01)
        state = optimizer.state_dict()
        state['state'][0] = dict(step=torch.tensor(768.), exp_avg=torch.zeros_like(parameter),
                                exp_avg_sq=torch.zeros_like(parameter))
        backend = dict(optimizer=dict(recipe={'fixed': True}, optimizer_state_dict=state))
        return parameter, optimizer, initial, gradient, backend

    def test_non_capturable_loader_alias_explains_changed_adam_counter(self):
        parameter, optimizer, initial, gradient, backend = self.optimizer_fixture()
        source = backend['optimizer']['optimizer_state_dict']
        endpoints = []
        # Reproduce CUDA's transfer ownership on CPU: moments get separate
        # storage, while the real PyTorch loader retains the CPU step tensor.
        for expected_step in (769., 770.):
            copied = deepcopy(source)
            copied['state'][0]['step'] = source['state'][0]['step']
            optimizer.load_state_dict(copied)
            self.assertIs(optimizer.state[parameter]['step'], source['state'][0]['step'])
            with torch.no_grad():
                parameter.copy_(initial)
            parameter.grad = gradient.clone()
            optimizer.step()
            self.assertEqual(source['state'][0]['step'].item(), expected_step)
            endpoints.append(parameter.detach().clone())
        self.assertGreater(torch.count_nonzero(endpoints[0] != endpoints[1]).item(), 0)

    def test_cuda_restore_clones_host_steps_and_repeated_updates_match(self):
        parameter, optimizer, initial, gradient, backend = self.optimizer_fixture()
        untouched = deepcopy(backend)
        endpoints, adam_states = [], []
        for _ in range(2):
            isolated = Q.isolated_backend_for_load(backend, torch.device('cuda'))
            source_state = backend['optimizer']['optimizer_state_dict']['state'][0]
            load_state = isolated['optimizer']['optimizer_state_dict']['state'][0]
            self.assertIsNot(load_state['step'], source_state['step'])
            self.assertIs(load_state['exp_avg'], source_state['exp_avg'])
            self.assertIs(load_state['exp_avg_sq'], source_state['exp_avg_sq'])
            # No GPU is allocated: emulate only the moment copies that happen
            # on actual CUDA load, leaving the helper's step ownership intact.
            load_state['exp_avg'] = load_state['exp_avg'].clone()
            load_state['exp_avg_sq'] = load_state['exp_avg_sq'].clone()
            optimizer.load_state_dict(isolated['optimizer']['optimizer_state_dict'])
            with torch.no_grad():
                parameter.copy_(initial)
            parameter.grad = gradient.clone()
            optimizer.step()
            endpoints.append(parameter.detach().clone())
            adam_states.append(Q.tree_identity(optimizer.state_dict()))
            self.assertEqual(source_state['step'].item(), 768.)
            self.assertEqual(optimizer.state[parameter]['step'].item(), 769.)
        self.assertTrue(torch.equal(endpoints[0], endpoints[1]))
        self.assertEqual(adam_states[0], adam_states[1])
        self.assertEqual(Q.tree_identity(backend), Q.tree_identity(untouched))

    def test_cpu_restore_owns_moments_as_well_as_counters(self):
        _, _, _, _, backend = self.optimizer_fixture()
        isolated = Q.isolated_backend_for_load(backend, torch.device('cpu'))
        source_state = backend['optimizer']['optimizer_state_dict']['state'][0]
        loaded_state = isolated['optimizer']['optimizer_state_dict']['state'][0]
        for field in ('step', 'exp_avg', 'exp_avg_sq'):
            self.assertIsNot(loaded_state[field], source_state[field])
            loaded_state[field].add_(1)
            self.assertFalse(torch.equal(loaded_state[field], source_state[field]))

    def test_termination_signals_raise_so_existing_finally_runs(self):
        with patch.object(E.signal, 'signal') as install:
            E.install_interrupt_handlers()
        self.assertEqual([call.args[0] for call in install.call_args_list],
                         [E.signal.SIGTERM, E.signal.SIGINT])
        for call in install.call_args_list:
            signal_number, handler = call.args
            with self.assertRaisesRegex(InterruptedError, 'no automatic retry'):
                handler(signal_number, None)

    def test_both_cli_entries_install_handlers_before_starting_work(self):
        from experiments.persistent_prefix_withholding import serving_qualification as serving
        with TemporaryDirectory() as directory:
            path = Path(directory)/'plan.json'
            path.write_text('{}')
            for module in (E, serving):
                events = []
                with patch('sys.argv', ['qualification', '--plan', str(path), '--slot', 'A']), \
                     patch.object(module, 'install_interrupt_handlers', side_effect=lambda: events.append('handlers')), \
                     patch.object(module, 'run', side_effect=lambda *args: events.append('run')):
                    module.main()
                self.assertEqual(events, ['handlers', 'run'])

    def test_digest_compares_every_represented_bit_and_none_membership(self):
        value = torch.tensor([0., -.0, 1.], dtype=torch.bfloat16)
        first = Q.tensor_identity(value)
        changed = value.clone()
        changed[1] = 0.
        self.assertEqual(first['elements'], 3)
        self.assertNotEqual(first['sha256'], Q.tensor_identity(changed)['sha256'])
        self.assertIsNone(Q.tensor_identity(None))
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            Q.tensor_identity(torch.tensor([float('nan')]))

    def test_cross_card_requires_all_scientific_fields_and_both_gates(self):
        row = dict(fixed_pair=['r0', 'r1'], forward=[dict(loss=1.)], preclip=dict(p=dict(sha256='abc')),
                   endpoint=dict(weights={}, optimizer={}, rng={}, global_step=769))
        first = dict(status='passed', cases=[dict(row, case=case) for case in ('ALL', 'E', 'S')])
        second = dict(status='passed', cases=[dict(row, case=case) for case in ('N', 'E', 'S')])
        self.assertEqual(Q.compare_cards(first, second)['attempted_optimizer_applications'], 8)
        for field in ('fixed_pair', 'forward', 'preclip', 'endpoint'):
            bad = deepcopy(second)
            bad['cases'][1][field] = 'changed'
            with self.assertRaises(ValueError):
                Q.compare_cards(first, bad)
        with self.assertRaises(ValueError):
            Q.compare_cards(dict(first, status='failed'), second)

    def test_visible_device_gate_precedes_model_work(self):
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES='other'), patch.object(torch.cuda, 'device_count', return_value=1):
            with self.assertRaisesRegex(ValueError, 'sealed physical GPU'):
                E.run(dict(devices={'A':'GPU-fixed'}), 'A')

    def test_loading_failure_has_durable_zero_training_accounting(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = root/'config.yaml'
            config.write_text('pinned')
            schedule = root/'schedule.json'
            schedule.write_text('{}')
            plan = dict(root=str(root), devices=dict(A='GPU-fixed'), execution_commit='a'*40,
                configs=dict(N=str(config)), binding='b'*64, origin_state='retained.pt',
                origin_export='retained_export', origin_config=str(config), event7_schedule=str(schedule))
            with patch.dict(os.environ, CUDA_VISIBLE_DEVICES='GPU-fixed'), \
                 patch.object(torch.cuda, 'device_count', return_value=1), \
                 patch('experiments.prefix_guard.qualify.pin_retained_graph_runtime', autospec=True) as runtime_pin, \
                 patch.object(E, 'PersistentPrefixTrainer', side_effect=ValueError('source identity')):
                with self.assertRaisesRegex(ValueError, 'source identity'):
                    E.run(plan, 'A')
            runtime_pin.assert_called_once_with(root/'qualification/A')
            receipt = json.loads((root/'qualification/A/STATUS.json').read_text())
            self.assertEqual(receipt['status'], 'failed')
            self.assertEqual(receipt['physical_work']['attempted'], dict(model_load=1))
            self.assertEqual(receipt['physical_work']['failed'], dict(model_load=1))
            self.assertEqual(receipt['optimizer_steps_startup'], 0)
            self.assertFalse(receipt['automatic_retry'])


if __name__ == '__main__':
    unittest.main()
