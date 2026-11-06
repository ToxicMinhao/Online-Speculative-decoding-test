"""Bounded clipping runner: state isolation, native Adam and compact evidence."""
from copy import deepcopy
import csv
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

from experiments.adaflash_clipping import run as R
from online_feedback.rng import capture_rng, restore_rng
from online_feedback.storage import save_torch


class RunnerUnitTests(unittest.TestCase):
    def test_pair_statistics_are_global_and_zero_norm_is_explicit(self):
        stats = R.paired_tensor_statistics(
            dict(a=torch.tensor([3., 0.]), b=torch.tensor([4.])),
            dict(a=torch.tensor([0., 3.]), b=torch.tensor([4.])))
        row = stats['global_statistics']
        self.assertEqual(row['left_norm'], 5.)
        self.assertEqual(row['right_norm'], 5.)
        self.assertAlmostEqual(row['cosine'], 16/25)
        self.assertEqual(row['different'], 2)
        self.assertEqual(row['left_nonzero'], 2)
        self.assertAlmostEqual(row['different_fraction'], 2/3)
        zero = R.paired_tensor_statistics({'p':torch.zeros(2)}, {'p':torch.zeros(2)})
        self.assertIsNone(zero['global_statistics']['cosine'])
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            R.paired_tensor_statistics({'p':torch.tensor([float('nan')])}, {'p':torch.ones(1)})

    def test_original_forward_capture_does_not_count_checkpoint_recomputation(self):
        from torch.utils.checkpoint import checkpoint
        observed = R.ObservedLoss('repo_position', chunk_blocks=1)
        logits = torch.randn(1, 2, 4, 7, dtype=torch.bfloat16, requires_grad=True)
        teacher = torch.randn_like(logits)
        labels = torch.tensor([[[0,1,2,3], [0,4,5,6]]])
        weights = torch.tensor([[[0.,1.,.8,.6], [0.,1.,.8,.6]]])
        observed.prepare_response(weights)
        parts = []
        for i in range(2):
            parts.append(checkpoint(observed.chunk_terms, logits[:,i:i+1],
                teacher[:,i:i+1], labels[:,i:i+1], weights[:,i:i+1], use_reentrant=False))
        terms = tuple(sum(items) for items in zip(*parts))
        observed.response_metrics(terms, weights.sum()+1e-6)
        terms[0].backward()
        self.assertEqual(len(observed.responses), 1)
        self.assertEqual(tuple(observed.responses[0]['predictions'].shape), (1,2,4))
        self.assertEqual(observed.chunks, [])
        self.assertIsNotNone(logits.grad)

    def test_bound_anchors_preserve_rng_and_restore_sampler(self):
        sampler = lambda *args: (_ for _ in ()).throw(AssertionError('resampled'))
        model = SimpleNamespace(draft_model=torch.nn.Linear(2,2), _sample_anchor_positions=sampler)
        record = dict(sampled_anchors=[dict(positions=[[2,4]], keep_mask=[[True,False]])])
        before = capture_rng()
        with R.retained_anchors(model, record):
            anchors, keep = model._sample_anchor_positions(100)
            self.assertEqual(anchors.tolist(), [[2,4]])
            self.assertEqual(keep.tolist(), [[True,False]])
        self.assertIs(model._sample_anchor_positions, sampler)
        self.assertFalse(R.differences(before, capture_rng()))

    def test_failure_persists_scope_and_uncompleted_work(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'run'
            with self.assertRaisesRegex(ValueError, 'six fixed'):
                R.run('unused', 'unused', dict(feature_paths=[], records=[]), output)
            failed = json.loads((output/'RESULT.json').read_text())
            self.assertEqual(failed['status'], 'failed')
            self.assertEqual(failed['uncompleted_branches'], 6)
            self.assertFalse(failed['automatic_retry'])
            self.assertEqual(failed['work'], {})
            self.assertEqual(failed['plan']['generations'], 0)
        self.assertEqual(R.serializable(float('nan')), 'nan')


class RunnerTinyIntegrationTests(unittest.TestCase):
    """Real tiny SpecForge/KL/BF16 Adam path; only hardware assembly is mocked."""
    @classmethod
    def setUpClass(cls):
        if os.environ.get('CLIPPING_RUNNER_CPU_CHILD') == '1':
            from test_full_k_training import FullKTinyIntegrationTests
            FullKTinyIntegrationTests.setUpClass.__func__(cls)

    @classmethod
    def tearDownClass(cls):
        if os.environ.get('CLIPPING_RUNNER_CPU_CHILD') == '1':
            from test_full_k_training import FullKTinyIntegrationTests
            FullKTinyIntegrationTests.tearDownClass.__func__(cls)

    def setUp(self):
        if os.environ.get('CLIPPING_RUNNER_CPU_CHILD') == '1':
            from test_full_k_training import FullKTinyIntegrationTests
            FullKTinyIntegrationTests.setUp(self)

    def test_six_independent_native_steps_and_exact_budget(self):
        if os.environ.get('CLIPPING_RUNNER_CPU_CHILD') != '1':
            root = Path(__file__).resolve().parents[1]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='', CLIPPING_RUNNER_CPU_CHILD='1')
            env['PYTHONPATH'] = os.pathsep.join(str(root / p) for p in ('.','tests','training','inference'))
            result = subprocess.run([sys.executable, '-m', 'unittest', '-v',
                'test_adaflash_clipping_run.RunnerTinyIntegrationTests.test_six_independent_native_steps_and_exact_budget'],
                cwd=root, env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            return
        from experiments.prefix_ranking.qualify import cpu_copy
        original = R.SpecForgeEventTrainer(self.config, self.root/'initial_source', objective='osd')
        initial = dict(draft_state_dict=cpu_copy(dict(original.bundle.draft_model.state_dict())),
            backend=cpu_copy(dict(optimizer=original.backend.optimizer.state_dict())),
            rng=capture_rng(), global_step=0, presentations=0, events=[], config_text=self.config.read_text())
        original.close()
        initial_path = self.root/'initial.pt'
        save_torch(initial_path, initial)
        prepared = dict(feature_paths=[str(p) for p in self.features[:6]], records=[])
        for path in prepared['feature_paths']:
            raw = torch.load(path, weights_only=True)
            indices = (raw['loss_mask'][0]>0).nonzero().flatten().tolist()
            prepared['records'].append(dict(record_id=raw['record_id'], **{'pass':0},
                sampled_anchors=[dict(positions=[indices[:2]], keep_mask=[[True,True]])]))
        snapshots = []
        native_preclip_norms = []
        native_factory = R.SpecForgeEventTrainer
        from online_feedback.optimizer import DirectBF16AdamW
        native_adam_step = DirectBF16AdamW.step
        def observe_native_step(optimizer):
            native_preclip_norms.append(sum(float(p.grad.double().square().sum())
                for p in optimizer.model_params if p.grad is not None) ** .5)
            return native_adam_step(optimizer)
        def factory(*args, **kwargs):
            instance = native_factory(*args, **kwargs)
            snapshots.append(instance)
            return instance
        with patch.object(R, 'ranking_eos_token_ids', return_value=(28,)), \
                patch.object(R, 'SpecForgeEventTrainer', side_effect=factory), \
                patch.object(DirectBF16AdamW, 'step', observe_native_step):
            result = R.run(self.config, initial_path, prepared, self.root/'diagnostic')
        self.assertEqual(result['status'], 'passed')
        self.assertEqual(len(snapshots), 7)  # disposable startup plus six fresh runtimes
        self.assertTrue(all(trainer.closed for trainer in snapshots))
        self.assertEqual(result['work']['adam_step_completed'], 6)
        self.assertEqual(result['work']['core_backward_completed'], 12)
        self.assertEqual(result['work']['startup_backward_completed'], 2)
        self.assertEqual(result['work']['response_model_forwards'], 26)
        self.assertEqual(result['work']['post_response_forward_completed'], 12)
        self.assertEqual(len(result['comparisons']), 3)
        self.assertTrue(all(row['identical_preupdate_blocks'] for row in result['comparisons']))
        self.assertTrue(all(row['identical_final_training_rng'] for row in result['comparisons']))
        self.assertEqual(len(native_preclip_norms), 6)
        for index, comparison in enumerate(result['comparisons']):
            self.assertAlmostEqual(comparison['gradients']['global_statistics']['left_norm'],
                                   native_preclip_norms[2*index])
            self.assertAlmostEqual(comparison['gradients']['global_statistics']['right_norm'],
                                   native_preclip_norms[2*index+1])
        self.assertEqual(list((self.root/'diagnostic').rglob('*.pt')), [])
        self.assertFalse(R.differences(initial, torch.load(initial_path, weights_only=False)))
        for number in range(1,4):
            for mode in R.MODES:
                receipt = json.loads((self.root/'diagnostic'/f'batch_{number:02d}_{mode}'/'RESULT.json').read_text())
                self.assertEqual(receipt['adam_steps'], 1)
                self.assertEqual(receipt['presentations'], 2)
                self.assertTrue(receipt['restoration']['fresh_adam'])
                self.assertEqual(len(receipt['responses_before']), 2)
                self.assertEqual(len(receipt['responses_after']), 2)
        self.assertFalse(torch.cuda.is_initialized())
        # Build the final tables from the actual tiny-run receipt schema.
        from experiments.adaflash_clipping.report import build
        (self.root/'STATUS.json').write_text(json.dumps(dict(status='passed', execution_commit='fixture', wall_s=1.)))
        (self.root/'preparation').mkdir()
        (self.root/'preparation'/'PREPARED.json').write_text(json.dumps(dict(work={}, costs={'wall_s':0.})))
        build(self.root, self.root/'report')
        for name, count in [('LOSSES.csv',24), ('SOURCE_PREFIXES.csv',12), ('PAIRED_BATCHES.csv',3)]:
            with (self.root/'report'/name).open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), count)


if __name__ == '__main__':
    unittest.main()
