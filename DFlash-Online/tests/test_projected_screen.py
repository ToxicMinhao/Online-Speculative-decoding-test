"""Focused CPU checks of the bounded campaign's protocol gates."""
import copy
import math
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import torch
import yaml

from experiments.projected_target import run as runner
from specforge.optimizer import BF16Optimizer


class RunnerTests(unittest.TestCase):
    def test_actual_upstream_eight_step_schedule_is_pinned(self):
        model = torch.nn.Linear(1, 1, bias=False).to(torch.bfloat16)
        optimizer = BF16Optimizer(model, lr=6e-4, total_steps=8, warmup_ratio=.04,
                                  lr_scheduler='cosine', offload_master=True, max_grad_norm=1.)
        used = []
        for _ in range(8):
            used.append(optimizer.optimizer.param_groups[0]['lr'])
            model.weight.grad = torch.ones_like(model.weight)
            optimizer.step()
        for actual, expected in zip(used, runner.EXPECTED_LRS):
            self.assertAlmostEqual(actual, expected, places=14)
        self.assertEqual(optimizer.optimizer.param_groups[0]['lr'], 0)

    def test_pair_rejects_output_and_termination_discrepancies(self):
        original = {'record_id': 'r', 'prompt_ids': [1, 2], 'response_ids': [3, 4],
                    'finish_reason': 'eos', 'stop_token_ids': [4], 'max_new_tokens': 512}
        runner.check_pair(original, original)
        for key, replacement in {'prompt_ids': [2, 1], 'response_ids': [3, 5],
                                 'finish_reason': 'length', 'stop_token_ids': [5],
                                 'max_new_tokens': 256}.items():
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.check_pair({**original, key: replacement}, original)

    def test_exposure_counts_overlaps_once_per_sample_not_once_per_output(self):
        record = {'record_id': 'r', 'pass': 0, 'sampled_anchors': [
            {'positions': [[3, 8, 0]], 'keep_mask': [[True, True, False]]}]}
        canonical = {'r': {'prompt_ids': [1] * 3, 'response_ids': [2] * 10}}
        result = runner.exposure([record], canonical)
        self.assertEqual(result['total_supervised_positions'], 13)  # 9+4; overlaps count as exposure.
        expected = sum(math.exp(-(k-1)/7) for k in range(1,10)) + sum(math.exp(-(k-1)/7) for k in range(1,5))
        self.assertAlmostEqual(result['total_weight'], expected)

    def test_invalid_completed_generation_still_consumes_budget(self):
        original = {'record_id': 'r', 'prompt_ids': [1], 'response_ids': [3, 4],
                    'finish_reason': 'eos', 'stop_token_ids': [4], 'max_new_tokens': 512}
        backend = SimpleNamespace(render=lambda _: [1],
                                  generate=lambda *args, **kwargs: {**original, 'response_ids': [2, 4]})
        request = {'messages': [], 'prompt_ids': [1], 'request_id': 'r', 'index': 1, 'role': 'A'}
        completed = []
        with self.assertRaises(ValueError):
            runner.generate(backend, request, Path('.'), arm='C', group='gsm8k', canonical=original,
                            completed_generation=lambda: completed.append('executed'))
        self.assertEqual(completed, ['executed'])

    def test_training_order_and_counts_are_enforced(self):
        ids = [f'a{i}' for i in range(8)]
        rows = [{'record_id': name, 'pass': p, 'optimizer_stepped': i % 2 == 1}
                for p in range(2) for i, name in enumerate(ids)]
        receipt = {'records': rows, 'optimizer_steps': 8, 'counters': {'presentations': 16}}
        runner.validate_learning(receipt, ids, runner.EXPECTED_LRS)
        changed = copy.deepcopy(receipt)
        changed['records'][8]['pass'] = 0
        with self.assertRaises(AssertionError):
            runner.validate_learning(changed, ids, runner.EXPECTED_LRS)
        with self.assertRaises(AssertionError):
            runner.validate_learning(receipt, ids, [.0006] * 8)

    def test_provisional_manifest_or_recipe_change_cannot_reach_gpu(self):
        config = yaml.safe_load((Path(__file__).resolve().parents[1] /
            'experiments/projected_target/config.yaml').read_text())
        manifest = {'status': 'sealed_before_generation', 'selection_seed': 20261022, 'groups': []}
        for gi, group in enumerate(['gsm8k', 'codealpaca', 'sharegpt']):
            requests = [{'request_id': f'{group}_{i}', 'role': 'A' if i < 8 else 'B',
                         'prompt_ids': [gi * 16 + i + 1]} for i in range(16)]
            manifest['groups'].append({'name': group, 'requests': requests})
        target, draft = config['model']['target_model_path'], config['model']['draft_checkpoint_path']
        runner.validate_plan(manifest, config, target, draft)
        with self.assertRaises(AssertionError):
            runner.validate_plan({**manifest, 'status': 'conditional_proposal_not_approved'}, config, target, draft)
        changed = copy.deepcopy(config)
        changed['training']['loss_decay_gamma'] = 5
        with self.assertRaises(AssertionError):
            runner.validate_plan(manifest, changed, target, draft)

    def test_different_anchors_or_exposure_are_rejected(self):
        left = {'records': [{'sampled_anchors': [{'positions': [[1]], 'keep_mask': [[True]]}]}],
                'exposure': {'total_supervised_positions': 15}, 'actual_learning_rates': runner.EXPECTED_LRS}
        runner.compare_learning(left, copy.deepcopy(left))
        right = copy.deepcopy(left)
        right['records'][0]['sampled_anchors'][0]['positions'][0][0] = 2
        with self.assertRaises(AssertionError):
            runner.compare_learning(left, right)



if __name__ == '__main__':
    unittest.main()
