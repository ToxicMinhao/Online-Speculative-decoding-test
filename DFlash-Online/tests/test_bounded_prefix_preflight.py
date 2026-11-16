"""CPU-only package/source/resource admission contracts, without model construction."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.bounded_prefix_allocation.prepare import identity
from experiments.bounded_prefix_allocation.preflight import HOST_POLICY, host_admission, verify_source, inspect, inspect_states
from test_bounded_prefix_prepare import fixture


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name)

    def test_current_host_fits_one_trainer_and_serving_not_two_trainers(self):
        receipt=host_admission(104_154_714_112)
        self.assertEqual(receipt['status'],'passed')
        self.assertTrue(receipt['parallel_qualification_admitted'])
        self.assertFalse(receipt['two_trainers_admitted'])
        self.assertEqual(receipt['policy']['historical_training_reservation_bytes'],47_283_793_920)
        self.assertLess(receipt['sequential_required_bytes'],112*2**30)

    def test_less_memory_selects_sequential_or_blocks(self):
        minimum=HOST_POLICY['training_reservation_bytes']+HOST_POLICY['host_headroom_bytes']
        receipt=host_admission(minimum)
        self.assertEqual(receipt['status'],'passed')
        self.assertFalse(receipt['parallel_qualification_admitted'])
        self.assertEqual(host_admission(minimum-1)['status'],'blocked')

    def source_fixture(self):
        root=self.base/'source';root.mkdir();(root/'code.py').write_text('constant = 1\n')
        manifest=dict(commit='a'*40,dirty=False,files={'code.py':identity(root/'code.py')})
        path=self.base/'SOURCE.json';path.write_text(json.dumps(manifest))
        return dict(source_root=str(root),source_manifest=str(path),execution_commit='a'*40),manifest

    def test_published_source_exact_identity(self):
        plan,_=self.source_fixture()
        self.assertEqual(verify_source(plan)['checked_files'],1)
        (Path(plan['source_root'])/'code.py').write_text('constant = 2\n')
        with self.assertRaises(ValueError):verify_source(plan)

    def test_dirty_or_wrong_commit_rejected(self):
        plan,manifest=self.source_fixture();manifest['dirty']=True
        Path(plan['source_manifest']).write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):verify_source(plan)
        manifest['dirty']=False;manifest['commit']='b'*40
        Path(plan['source_manifest']).write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):verify_source(plan)

    def test_internal_source_link_allowed_but_escape_rejected(self):
        plan,manifest=self.source_fixture();root=Path(plan['source_root'])
        (root/'alias.py').symlink_to('code.py')
        manifest['files']['alias.py']=dict(kind='symlink',target='code.py')
        Path(plan['source_manifest']).write_text(json.dumps(manifest))
        self.assertEqual(verify_source(plan)['checked_files'],2)
        (root/'alias.py').unlink();(self.base/'outside.py').write_text('value=1\n')
        (root/'alias.py').symlink_to('../outside.py')
        manifest['files']['alias.py']['target']='../outside.py'
        Path(plan['source_manifest']).write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):verify_source(plan)

    def test_recovery_loader_uses_read_only_cpu_mmap_for_all_three_states(self):
        def state(events):
            return dict(events=[dict(features=[dict(record_id=f"r{i}") for i in range(n*128,(n+1)*128)]) for n in range(events)],
                algorithm_contract={"persistent_prefix_withholding_v2":{"binding":"a"*64}},
                config_text="run_id: test\n",draft_state_dict={},rng={},global_step=events*128,
                presentations=events*256,backend={"optimizer":{"recipe":{}}})
        k6,n8,s8=state(6),state(8),state(8)
        config=self.base/'config.yaml';config.write_text('run_id: test\n')
        ancestry=self.base/'ancestry.json';ancestry.write_text(json.dumps(dict(requests=[dict(request_id=f'r{i}') for i in range(1024)])))
        plan=dict(origin_state='K6.pt',origin_export='K6/export',origin_config=str(config),
            expected_ancestry=str(ancestry),configs={'C':str(config)},binding='a'*64,
            controls={'N':dict(state='N8.pt',export='N8/export'),'S':dict(state='S8.pt',export='S8/export')})
        with patch('torch.cuda.is_initialized',return_value=False), \
             patch('torch.load',side_effect=[k6,n8,s8]) as load, \
             patch('experiments.ok_coding.training._export_equal') as export, \
             patch('experiments.persistent_prefix_withholding.state.validate_legacy_origin',return_value={}), \
             patch('experiments.persistent_prefix_withholding.state.validate_snapshot',return_value={}), \
             patch('experiments.bounded_prefix_allocation.training.fork_state',return_value=k6), \
             patch('experiments.bounded_prefix_allocation.training.validate_snapshot',return_value={}):
            receipt=inspect_states(plan)
        self.assertEqual(set(receipt),{'K6','N','S'})
        self.assertEqual(export.call_count,3)
        self.assertEqual([call.args[0] for call in load.call_args_list],['K6.pt','N8.pt','S8.pt'])
        for call in load.call_args_list:
            self.assertEqual(call.kwargs,dict(map_location='cpu',weights_only=False,mmap=True))
        self.assertEqual(receipt['K6']['ancestry_records'],768)
        self.assertEqual(receipt['N']['ancestry_records'],1024)

    def test_missing_actual_references_fail_before_runtime_or_recovery(self):
        root=self.base/'package';plan=fixture(root)
        old_root=plan['root'];plan['root']=str(root);plan['execution_commit']='a'*40
        for key in ('references','data_panel','warmups','warmup_references','event8_inputs'):
            old=plan[key];new=str(root/Path(old).relative_to(old_root));plan[key]=new
            plan['input_identities'][new]=plan['input_identities'].pop(old)
        path=root/'sealed/PLAN.json';path.write_text(json.dumps(plan))
        Path(plan['references']).unlink()
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':''}), \
             patch('experiments.bounded_prefix_allocation.preflight.runtime_inventory') as runtime, \
             patch('experiments.bounded_prefix_allocation.preflight.inspect_states') as states:
            with self.assertRaises(FileNotFoundError):inspect(path)
            runtime.assert_not_called();states.assert_not_called()
        receipt=json.loads((root/'CPU_PREFLIGHT.json').read_text())
        self.assertEqual(receipt['status'],'failed');self.assertEqual(receipt['model_work'],0)
        self.assertIn('CANONICALS',receipt['error_path'])


if __name__=='__main__':unittest.main()
