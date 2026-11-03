"""CPU checks of fixed input identity, bounded storage and pre-model sealing."""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import yaml

from experiments.full_k_gsm8k.contract import SERVING, stable_digest
from experiments.k_workload_replay_v1 import data as D, seal as S, storage as C
from experiments.k_workload_replay_v1.contract import ARMS, BUDGET, CODING_BINDING, EVENT_DOMAINS
from experiments.ok_events_5_8.storage import identity
from online_feedback.storage import write_json


def source_fixture():
    def rows(prefix, count, start, token_start):
        return [dict(request_id=f'{prefix}_{n:05d}', request_index=n,
                     source_id=f'{prefix}:source:{n}', prompt_ids=[token_start + n],
                     max_new_tokens=2048, stop_token_ids=[151645, 151643])
                for n in range(start, start + count)]
    math = dict(train_requests=rows('train', 512, 0, 1),
                eval_requests=rows('eval', 128, 1, 2000))
    training = rows('coding_training', 512, 0, 1000)
    evaluation = rows('coding_evaluation', 128, 0, 3000)
    def references(values):
        return {row['request_id']: dict(response_ids=[6, 151645], finish_reason='eos',
                source_record_path='/old/' + row['request_id'] + '.json', prompt_ids=row['prompt_ids'])
                for row in values}
    kwargs = dict(math_canonicals=references(math['eval_requests']),
                  coding_canonicals=references(evaluation),
                  decode=lambda ids: f'<|im_start|>user\nQuestion {ids[0]}<|im_end|>',
                  stop_ids=[151645, 151643])
    return math, training, evaluation, kwargs


class InputTests(unittest.TestCase):
    def test_exact_source_ids_ordinals_domains_and_balanced_panel(self):
        math, train, evaluation, kwargs = source_fixture()
        result = D.build_inputs(math, train, evaluation, **kwargs)
        self.assertEqual([row['domain'] for row in result['events']], list(EVENT_DOMAINS))
        self.assertEqual([row['global_ordinal'] for row in result['training']], list(range(1024)))
        self.assertEqual(result['training'][256]['request_id'], 'coding_training_00000')
        self.assertEqual(result['training'][512]['request_id'], 'train_00256')
        self.assertEqual([row['request_id'] for row in result['evaluation'][:4]],
                         ['eval_00001', 'coding_evaluation_00000', 'eval_00002', 'coding_evaluation_00001'])
        for row in result['training'] + result['evaluation']:
            self.assertEqual(row['request_id'], row['source_request_id'])
            self.assertTrue(row['original_source_id'])
            self.assertEqual(row['max_new_tokens'], 2048)
        for offset in range(0, 256, 64):
            self.assertEqual(sum(row['domain'] == 'math' for row in result['evaluation'][offset:offset+64]), 32)

    def test_changed_order_cap_stop_or_canonical_binding_fails(self):
        for failure in ('order', 'cap', 'stop', 'canonical'):
            math, train, evaluation, kwargs = source_fixture()
            if failure == 'order':
                train.reverse()
            elif failure == 'cap':
                train[0]['max_new_tokens'] = 512
            elif failure == 'stop':
                evaluation[0]['stop_token_ids'] = [151645]
            else:
                kwargs['math_canonicals']['eval_00001']['prompt_ids'] = [999]
            with self.subTest(failure=failure), self.assertRaises(ValueError):
                D.build_inputs(math, train, evaluation, **kwargs)

    def test_cross_domain_and_train_eval_normalized_user_duplicates_fail(self):
        math, train, evaluation, kwargs = source_fixture()
        # Different tokens and system prefixes must not hide equal user content.
        kwargs['decode'] = lambda ids: f'system{ids[0]}<|im_start|>user\n repeated   question <|im_end|>'
        with self.assertRaisesRegex(ValueError, r'duplicate \(user\)'):
            D.build_inputs(math, train, evaluation, **kwargs)
        math, train, evaluation, kwargs = source_fixture()
        evaluation[0]['prompt_ids'] = list(math['train_requests'][0]['prompt_ids'])
        kwargs['coding_canonicals'][evaluation[0]['request_id']]['prompt_ids'] = evaluation[0]['prompt_ids']
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            D.build_inputs(math, train, evaluation, **kwargs)


class StorageTests(unittest.TestCase):
    def test_reservoir_capacity_and_peak_include_states_exports_and_reserve(self):
        math, train, evaluation, kwargs = source_fixture()
        inputs = D.build_inputs(math, train, evaluation, **kwargs)
        before = np.random.get_state()
        plan = C.capacity_plan(inputs['training'], 2_097_252_864, compiler_baseline=[])
        after = np.random.get_state()
        self.assertTrue(np.array_equal(before[1], after[1]))
        self.assertEqual(plan['row_bytes'], 49164)
        self.assertEqual(plan['reserve_bytes'], 8 * C.GIB)
        self.assertEqual(len(plan['phases']), 8)
        self.assertEqual(plan['phases'][0]['past_memory_rows'], 0)
        self.assertTrue(all(row['past_memory_rows'] == 128 * 2049 for row in plan['phases'][1:]))
        self.assertGreater(plan['required_initial_free_bytes'], plan['tensor_peak_bytes'] + 8 * C.GIB)

    def test_inode_accounting_deduplicates_hardlinks_and_excludes_symlinks(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'source.pt'
            source.write_bytes(b'x' * 8192)
            os.link(source, root / 'same.pt')
            (root / 'link.pt').symlink_to(source)
            self.assertEqual(C.allocated_bytes([root, source]), source.stat().st_blocks * 512)

    def test_retirement_is_owned_and_requires_retained_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            owned = root / 'new'
            owned.mkdir()
            disposable = owned / 'old.pt'
            disposable.write_text('old')
            successor = owned / 'new.pt'
            successor.write_text('successor')
            with self.assertRaises(ValueError):
                C.retire_owned(disposable, root=owned, receipt_path=owned/'receipt.json',
                               reason='verified successor', retained_evidence=[owned/'missing'])
            outside = root / 'historical.pt'
            outside.write_text('history')
            with self.assertRaises(ValueError):
                C.retire_owned(outside, root=owned, receipt_path=owned/'receipt.json',
                               reason='outside', retained_evidence=[successor])
            C.retire_owned(disposable, root=owned, receipt_path=owned/'receipt.json',
                           reason='verified successor', retained_evidence=[successor])
            self.assertFalse(disposable.exists())
            receipt = json.loads((owned/'receipt.json').read_text())
            self.assertEqual(receipt['status'], 'retired')
            self.assertFalse(receipt['archived'])
            self.assertTrue(outside.exists())

    def test_reserve_and_allowance_exhaustion_stop_before_next_write(self):
        with tempfile.TemporaryDirectory() as folder:
            plan = dict(reserve_bytes=800, compiler_growth_bytes=400, compact_bytes=300,
                        compiler_baseline=[])
            with mock.patch.object(C.shutil, 'disk_usage', return_value=SimpleNamespace(free=900)):
                C.require_write_space(folder, plan, 100)
                with self.assertRaises(RuntimeError):
                    C.require_write_space(folder, plan, 101)
                with self.assertRaises(RuntimeError):
                    C.check_growth(folder, plan, pending_write_bytes=101)


class SealTests(unittest.TestCase):
    def test_dirty_source_is_rejected_before_any_seal_write(self):
        with mock.patch.object(S.subprocess, 'check_output', side_effect=['a'*40, ' M altered.py\n']):
            with self.assertRaisesRegex(ValueError, 'clean implementation'):
                S.execution_source(Path('/unused'), Path('/remote/code'))

    def test_preflight_hash_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            artifact = root / 'INPUT_SOURCE_CHECK.json'
            write_json(artifact, {'status': 'passed'})
            index = root / 'PREFLIGHT_RECEIPT_INDEX.json'
            write_json(index, dict(status='completed_read_only_preflight',
                files=[dict(path=str(artifact), **identity(artifact))]))
            artifact.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'receipt changed'):
                S.verify_preflight(index)

    def test_seal_writes_fixed_plan_exact_sources_and_three_unchanged_configs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            code = root / 'code'
            (code/'experiments/k_workload_replay_v1').mkdir(parents=True)
            (code/'experiments/k_workload_replay_v1/seal.py').write_text('# sealed fixture\n')
            original = root / 'original'
            original.mkdir()
            raw = yaml.safe_load((Path(__file__).parents[1]/'configs/osd.yaml').read_text())
            raw['data']['max_length'] = 2272
            (original/'ORIGINAL_CONFIG.yaml').write_text(yaml.safe_dump(raw))
            for name in S.PIN_NAMES:
                write_json(original/name, {'fixture': name})
            preflight = root / 'preflight'
            preflight.mkdir()
            index = preflight/'PREFLIGHT_RECEIPT_INDEX.json'
            write_json(index, dict(status='completed_read_only_preflight', files=[]))
            math, train, evaluation, kwargs = source_fixture()
            inputs = D.build_inputs(math, train, evaluation, **kwargs)
            weights = 2_097_252_864
            storage = C.capacity_plan(inputs['training'], weights + 1000, compiler_baseline=[])
            receipts = {
                'REMOTE_INVENTORY.json': dict(models={'draft': {'weights': [dict(tensor_bytes=weights,
                    file_bytes=weights+1000)]}}, compiler_caches=[]),
                'STORAGE_ADMISSION_PLAN.json': dict(required_initial_free_bytes=storage['required_initial_free_bytes'],
                    peak={'tensor_bytes': storage['tensor_peak_bytes']}, prepared_feature_row_bytes=49164, phases=[]),
                'INPUT_SOURCE_CHECK.json': {'checks': []},
                'CANONICAL_SOURCE_CHECK.json': {'records': [], 'coding_provenance_limit': 'Disclosed fixture scope'},
            }
            source = dict(commit='a'*40, dirty=False, source_path='/remote/code', scientific_base=S.BASE_COMMIT)
            old_plan = dict(target=raw['model']['target_model_path'], draft=raw['model']['draft_checkpoint_path'],
                            training_python='/pinned/training/python', worker_python='/pinned/serving/python')
            args = SimpleNamespace(code=str(code), preflight=str(index), output=str(root/'out'),
                                   remote_root='/remote/run', remote_code='/remote/code')
            with mock.patch.object(S, 'execution_source', return_value=source), \
                 mock.patch.object(S, 'verify_preflight', return_value=receipts), \
                 mock.patch.object(S, 'prepare_inputs', return_value=deepcopy(inputs)), \
                 mock.patch.object(S, 'verify_original_recipe', return_value=(original,old_plan,
                     {kind:{'files':{}} for kind in ('draft','target')})), \
                 mock.patch.object(S.subprocess, 'check_output', return_value=b'experiments/k_workload_replay_v1/seal.py\0'):
                result = S.seal(args)
            self.assertEqual(result['model_work'], 0)
            plan = S.read(root/'out/sealed/PLAN.json')
            S.validate_plan(plan)
            self.assertEqual(plan['budget'], BUDGET)
            self.assertEqual(plan['arms'], list(ARMS))
            self.assertEqual(plan['storage']['required_initial_free_bytes'], storage['required_initial_free_bytes'])
            for arm in ARMS:
                text = (root/f'out/sealed/config_{arm}.yaml').read_text()
                S.validate_config_text(text, arm, (original/'ORIGINAL_CONFIG.yaml').read_text())
            bound = S.read(root/'out/sealed/SEAL.json')['files']
            self.assertEqual(bound['/remote/code/experiments/k_workload_replay_v1/seal.py'],
                             identity(code/'experiments/k_workload_replay_v1/seal.py'))
            self.assertIn('/remote/run/sealed/PLAN.json', bound)
            broken = deepcopy(plan)
            broken['budget']['optimizer_steps'] += 1
            with self.assertRaises(ValueError):
                S.validate_plan(broken)


if __name__ == '__main__':
    unittest.main()
