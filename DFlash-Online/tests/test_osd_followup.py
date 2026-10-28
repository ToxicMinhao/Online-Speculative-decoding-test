"""CPU protocol checks for the new OSD arm; no model generations or training."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
import yaml

from experiments.projected_target import osd_followup as run
from online_feedback.storage import write_json


class FollowupTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def canonical(self):
        return {'record_id': 'a1', 'prompt_ids': [1, 2], 'response_ids': [3, 4],
                'finish_reason': 'eos', 'stop_token_ids': [4], 'max_new_tokens': 512,
                'metadata': {'verification_calls': 1}}

    def test_frozen_reproduction_requires_tokens_termination_and_counts(self):
        original = self.canonical()
        run.check_frozen_reproduction(original, original)
        for key, value in (('response_ids', [3, 5]), ('finish_reason', 'length'),
                           ('metadata', {'verification_calls': 2})):
            with self.subTest(key=key), self.assertRaises(ValueError):
                run.check_frozen_reproduction({**original, key: value}, original)

    def test_feature_check_preserves_exact_saved_response_and_mask(self):
        record = self.canonical()
        feature = {'record_id': 'a1', 'input_ids': torch.tensor([[1, 2, 3, 4]]),
                   'loss_mask': torch.tensor([[0., 0., 1., 1.]]),
                   'target_last_hidden_states': torch.ones(1, 4, 3)}
        path = self.root / 'feature.pt'
        torch.save(feature, path)
        run.check_feature(path, record)
        for field, changed in (('input_ids', torch.tensor([[1, 2, 3, 5]])),
                               ('loss_mask', torch.tensor([[0., 1., 1., 1.]])),
                               ('target_last_hidden_states', torch.ones(1, 3, 3))):
            with self.subTest(field=field):
                torch.save({**feature, field: changed}, path)
                with self.assertRaises(AssertionError):
                    run.check_feature(path, record)

    def test_training_match_checks_both_rng_endpoints_and_sampling(self):
        source, output = self.root / 'old', self.root / 'new/O'
        (source / 'C').mkdir(parents=True)
        output.mkdir(parents=True)
        receipt = {'records': [{'sampled_anchors': [{'positions': [[2]], 'keep_mask': [[True]]}]}],
                   'exposure': {'total_supervised_positions': 1}, 'actual_learning_rates': [0.001]}
        for name in ('initial_rng.pt', 'final_rng.pt'):
            torch.save({'rng': torch.tensor([1, 2])}, source / 'C' / name)
            torch.save({'rng': torch.tensor([1, 2])}, output / name)
        run.match_training(receipt, source, receipt, output)
        saved = json.loads((output.parent / 'TRAINING_MATCH.json').read_text())
        self.assertTrue(saved['matched_initial_rng'] and saved['matched_final_rng'])
        torch.save({'rng': torch.tensor([1, 3])}, output / 'final_rng.pt')
        with self.assertRaises(AssertionError):
            run.match_training(receipt, source, receipt, output)
        changed = copy.deepcopy(receipt)
        changed['records'][0]['sampled_anchors'][0]['positions'] = [[3]]
        with self.assertRaises(AssertionError):
            run.match_training(receipt, source, changed, output)

    def test_storage_budgets_new_exports_and_rolling_state_not_teacher_rebuild(self):
        expected = run.FULL_STATE + 3 * run.EXPORT + run.FEATURE_ALLOWANCE + run.RESERVE
        with mock.patch.object(run.shutil, 'disk_usage', return_value=SimpleNamespace(free=expected)):
            run.admit_followup_storage(self.root)
        receipt = json.loads((self.root / 'storage/initial.json').read_text())
        self.assertEqual(receipt['required_additional_bytes'], expected)
        self.assertTrue(receipt['source_features_reused'])
        with mock.patch.object(run.shutil, 'disk_usage', return_value=SimpleNamespace(free=expected - 1)):
            with self.assertRaises(ValueError):
                run.admit_followup_storage(self.root)

    def test_failed_training_keeps_partial_step_and_presentation_counts(self):
        groups = [{'name': name} for name in ('gsm8k', 'codealpaca', 'sharegpt')]
        write_json(self.root / 'gsm8k/O/trainer/events/update/RESULT.json',
                   {'counters': {'global_step': 8, 'presentations': 16}})
        write_json(self.root / 'codealpaca/O/trainer/events/update/FAILED.json',
                   {'global_step': 3, 'presentations': 7})
        self.assertEqual(run.training_counts(self.root, groups), (11, 23))

    def test_output_cannot_be_inside_historical_campaign(self):
        args = SimpleNamespace(source=str(self.root), output=str(self.root / 'nested'))
        with self.assertRaisesRegex(ValueError, 'immutable source'):
            run.run(args)

    def test_first_group_correctness_continues_all_groups_despite_worse_calls(self):
        config_path = Path(run.__file__).with_name('config.yaml')
        config = yaml.safe_load(config_path.read_text())
        source, output = self.root / 'source', self.root / 'output'
        source.mkdir()
        manifest = {'status': 'sealed_before_generation', 'selection_seed': 20261022, 'groups': []}
        retained = {}
        for group_index, name in enumerate(('gsm8k', 'codealpaca', 'sharegpt')):
            requests, canonical = [], {}
            for index in range(16):
                request_id, prompt = f'{name}_{index}', [16 * group_index + index + 10]
                requests.append({'request_id': request_id, 'prompt_ids': prompt, 'messages': prompt,
                                 'role': 'A' if index < 8 else 'B'})
                canonical[request_id] = {
                    **self.canonical(), 'record_id': request_id, 'prompt_ids': prompt,
                    'response_ids': [2, 3, 4], 'metadata': {'verification_calls': 1},
                }
            manifest['groups'].append({'name': name, 'requests': requests})
            retained[name] = (canonical, [source / name / f'a{i}.pt' for i in range(8)], {})
        write_json(source / 'manifest.json', manifest)
        args = SimpleNamespace(source=str(source), output=str(output), manifest=str(source / 'manifest.json'),
                               config=str(config_path), target=config['model']['target_model_path'],
                               draft=config['model']['draft_checkpoint_path'])
        backend = SimpleNamespace(loading_s=0., checkpoint_id=args.draft,
                                  target=torch.nn.Linear(1, 1), draft=torch.nn.Linear(1, 1))
        for parameter in backend.target.parameters():
            parameter.requires_grad_(False)
        stages = []

        def fake_generate(backend, request, destination, *, arm, group, completed_generation, canonical=None):
            record = copy.deepcopy(retained[group][0][request['request_id']])
            record['metadata']['verification_calls'] = 1 if arm == 'F' else 2
            write_json(destination / 'records' / f'{request["request_id"]}.json', record)
            stages.append((group, arm))
            completed_generation()
            return record

        def fake_run_arm(backend, released, group, arm, features, canonical, root, config,
                         completed_training, completed_generation, *, training_check):
            stages.append((group['name'], 'training'))
            receipt = {'counters': {'global_step': 8, 'presentations': 16}}
            directory = root / group['name'] / 'O'
            write_json(directory / 'trainer/events/update/RESULT.json', receipt)
            completed_training()
            training_check(receipt, directory)
            for request in group['requests']:
                fake_generate(backend, request, directory, arm=arm, group=group['name'],
                              completed_generation=completed_generation, canonical=canonical[request['request_id']])
            return receipt, {'outputs_equal': True}

        inventory = {name: (value[0], value[1], {'exposure': {'total_supervised_positions': 10}})
                     for name, value in retained.items()}
        with mock.patch.object(run, 'admit_followup_storage'), \
                mock.patch.object(run, 'load_source_group', side_effect=lambda source, group: inventory[group['name']]), \
                mock.patch.object(run, 'NativeDFlashBackend', return_value=backend), \
                mock.patch.object(run, 'generate', side_effect=fake_generate), \
                mock.patch.object(run, 'run_arm', side_effect=fake_run_arm), \
                mock.patch.object(run, 'match_training') as training_match, \
                mock.patch('experiments.projected_target.osd_report.build_report'):
            run.run(args)
        self.assertEqual(training_match.call_count, 3)
        status = json.loads((output / 'STATUS.json').read_text())
        self.assertEqual(status['status'], 'completed')
        self.assertEqual(status['generations_completed'], 96)
        self.assertEqual(status['optimizer_steps_completed'], 24)
        self.assertEqual(status['response_presentations_completed'], 48)
        for group_index, name in enumerate(('gsm8k', 'codealpaca', 'sharegpt')):
            expected = [(name, 'F')] * 16 + [(name, 'training')] + [(name, 'O')] * 16
            self.assertEqual(stages[group_index * 33:(group_index + 1) * 33], expected)
            self.assertEqual(json.loads((output / name / 'QUALIFICATION.json').read_text())['status'], 'passed')


if __name__ == '__main__':
    unittest.main()
