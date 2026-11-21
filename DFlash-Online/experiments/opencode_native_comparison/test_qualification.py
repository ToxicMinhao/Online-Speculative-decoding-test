"""Exact qualification slots, synthetic provenance and disposable isolation."""
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from . import qualification as Q
from . import configuration as C
from .test_training import fixtures, BINDING
from online_feedback.rng import capture_rng
from online_feedback.training import validate_target_binding
from experiments.ok_continuation.resume import differences


class MemoryLedger:
    def __init__(self):
        self.attempted, self.completed = Counter(), Counter()

    def call(self, counts, label, function, *args, **kwargs):
        self.attempted.update(counts)
        result = function(*args, **kwargs)
        self.completed.update(counts)
        return result


class QualificationContractTests(unittest.TestCase):
    def test_native_costs_exclude_qualification_observers_without_new_forwards(self):
        rows=[dict(training_s=3.),dict(training_s=5.)]
        result=Q.response_costs(rows,preclip_observer_s=1.,block_observer_s=.5)
        self.assertEqual(result['response_training_s'],[3.,5.])
        self.assertEqual(result['native_response_training_s'],8.)
        self.assertEqual(result['observer_adjusted_training_s'],6.5)

    def test_exact_slots_are_within_the_declared_ceiling(self):
        plan = Q.qualification_slots()
        totals = Counter()
        for row in plan['cases'] + plan['startup']:
            totals.update({key: row[key] for key in ('adam', 'response_forward', 'response_backward',
                                                  'teacher_preparation', 'generation')})
        totals['training_initialization'] = plan['training_initialization']
        self.assertEqual(dict(totals), plan['totals'])
        self.assertEqual(totals['adam'], 17)
        self.assertEqual(totals['response_forward'], 40)
        self.assertEqual(totals['response_backward'], 40)
        self.assertEqual(totals['training_initialization'], 3)
        self.assertEqual(len(plan['cases']), 17)

    def test_synthetic_fixture_preserves_rng_and_cannot_enter_native_feedback(self):
        with tempfile.TemporaryDirectory() as folder:
            before = capture_rng()
            paths = Q.create_synthetic_packets(folder, hidden_width=6, target_width=2)
            self.assertFalse(differences(before, capture_rng()))
            trainer = SimpleNamespace(hidden_width=6, vocab_size=16,
                                      bundle=SimpleNamespace(target_hidden_size=2))
            rows = Q.synthetic_metadata(trainer, paths)
            self.assertEqual([row['length'] for row in rows], [4096, 4096])
            self.assertEqual([row['response_tokens'] for row in rows], [2048, 2048])
            for path in paths:
                packet = torch.load(path, weights_only=True)
                with self.assertRaisesRegex(ValueError, 'verified canonical'):
                    validate_target_binding(packet)
            self.assertEqual(paths, Q.create_synthetic_packets(folder, hidden_width=6, target_width=2))
            paths[0].write_bytes(paths[0].read_bytes() + b'changed')
            with self.assertRaisesRegex(ValueError, 'Retained synthetic'):
                Q.create_synthetic_packets(folder, hidden_width=6, target_width=2)

    def test_incomplete_synthetic_fixture_is_preserved_not_regenerated(self):
        with tempfile.TemporaryDirectory() as folder:
            first = Path(folder) / 'synthetic4096_0.pt'
            first.write_bytes(b'interrupted')
            with self.assertRaisesRegex(ValueError, 'Incomplete prior'):
                Q.create_synthetic_packets(folder, hidden_width=6, target_width=2)
            self.assertEqual(first.read_bytes(), b'interrupted')

    def test_failed_operation_remains_attempted_but_not_completed(self):
        ledger = MemoryLedger()
        with self.assertRaises(ValueError):
            ledger.call({'adam': 1}, 'failed', lambda: (_ for _ in ()).throw(ValueError('stop')))
        self.assertEqual(ledger.attempted['adam'], 1)
        self.assertEqual(ledger.completed['adam'], 0)

    def test_constructor_failure_is_saved_before_any_cleanup(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            original = root / 'original.yaml'
            original.write_text('fixture: true\n')
            plan = dict(root=str(root/'run'), configs={'O_CODE58': 'unused.yaml'},
                original_config=str(original), binding=BINDING, execution_commit='0'*40,
                devices={'B': 'CPU-fixture'})
            ledger = MemoryLedger()
            with patch('experiments.prefix_guard.qualify.pin_retained_graph_runtime', return_value={}), \
                    patch('experiments.opencode_native_comparison.training.CodeTrainer',
                          side_effect=RuntimeError('retained constructor failure')):
                with self.assertRaisesRegex(RuntimeError, 'retained constructor failure'):
                    Q.run_training_qualification(plan, 'O_CODE58',
                        feature_paths=['a.pt', 'b.pt', 'c.pt', 'd.pt'], ledger=ledger)
            report = json.loads((root/'run/qualification/O_CODE58/DETAILS.json').read_text())
            self.assertEqual(report['status'], 'failed')
            self.assertIn('retained constructor failure', report['error'])
            self.assertEqual(ledger.attempted['training_initialization'], 1)
            self.assertEqual(ledger.completed['training_initialization'], 0)


class TinyQualificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get('OPENCODE_TINY_QUAL'):
            fixtures.FullKTinyIntegrationTests.setUpClass.__func__(cls)

    @classmethod
    def tearDownClass(cls):
        if os.environ.get('OPENCODE_TINY_QUAL'):
            fixtures.FullKTinyIntegrationTests.tearDownClass.__func__(cls)

    def setUp(self):
        if os.environ.get('OPENCODE_TINY_QUAL'):
            fixtures.FullKTinyIntegrationTests.setUp(self)

    def test_bounded_C_controls_on_real_tiny_CPU_runtime(self):
        if not os.environ.get('OPENCODE_TINY_QUAL'):
            root = Path(__file__).resolve().parents[2]
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OPENCODE_TINY_QUAL='1')
            environment['PYTHONPATH'] = os.pathsep.join(str(root/p) for p in ('.', 'tests', 'training', 'inference'))
            run = subprocess.run([sys.executable, '-m', 'unittest', '-v',
                __name__ + '.TinyQualificationTests.test_bounded_C_controls_on_real_tiny_CPU_runtime'],
                cwd=root, env=environment, text=True, capture_output=True)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            return
        from .training import CodeTrainer
        arm = 'C_CODE58'
        config = self.root/'code.yaml'
        config.write_text(C.build_config(self.original_config_text, arm=arm, output_dir=self.root/'common'))
        # Author fixture supplies deterministic released weights independently.
        trainer = CodeTrainer(config, self.root/'fixture_builder', arm=arm, binding=BINDING,
            original_config_text=self.original_config_text, expected_cuda_rng_states=0)
        released = self.root/'released'
        released.mkdir()
        save_file(dict(trainer.bundle.draft_model.state_dict()), released/'model.safetensors')
        trainer.close()
        original = self.root/'original.yaml'
        original.write_text(self.original_config_text)
        plan = dict(root=str(self.root/'qualification_run'), configs={arm: str(config)},
            original_config=str(original), binding=BINDING, execution_commit='0'*40,
            devices={'B': 'CPU-fixture'}, expected_cuda_rng_states=0)
        ledger = MemoryLedger()
        with patch('experiments.prefix_guard.qualify.pin_retained_graph_runtime', return_value={'CPU_fixture': True}):
            report = Q.run_training_qualification(plan, arm, feature_paths=self.features[:4], ledger=ledger)
        self.assertEqual(report['status'], 'passed')
        self.assertEqual([row['case'] for row in report['cases']], list(Q.CASES[arm]))
        self.assertEqual(ledger.completed, Counter(adam=7, response_forward=16,
                                                 response_backward=16, training_initialization=1))
        self.assertEqual(ledger.attempted, ledger.completed)
        self.assertTrue(report['origin_restored'])
        self.assertTrue(Path(report['active_C_export']).is_dir())
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main()
