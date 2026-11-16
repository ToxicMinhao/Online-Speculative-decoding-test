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
