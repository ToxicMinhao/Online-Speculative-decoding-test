"""Small CPU resource-accounting fixtures; no model downloads or GPU work."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from experiments.native_update_size import resources
from experiments.native_update_size.prepare import artifact_row


class InventoryTests(unittest.TestCase):
    def test_artifact_owner_without_container_account_retains_numeric_uid(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'retained';path.write_text('retained evidence')
            with patch('experiments.native_update_size.prepare.pwd.getpwuid',side_effect=KeyError):
                row=artifact_row(path,'KEEP_INPUT','identity check')
            self.assertEqual(row['owner'],f'uid:{path.stat().st_uid}')
            self.assertEqual(row['owner_uid'],path.stat().st_uid)
            self.assertEqual(row['classification'],'KEEP_INPUT')
            self.assertEqual(row['status'],'present')
            self.assertEqual(row['logical_bytes'],path.stat().st_size)

    def test_physical_hardlinks_deduplicate_and_external_symlinks_are_not_followed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'owned';root.mkdir();(root/'child').mkdir()
            one=root/'child'/'weights';one.write_bytes(b'x'*16384)
            os.link(one,root/'copy')
            outside=Path(tmp)/'outside';outside.mkdir();(outside/'not_owned').write_bytes(b'x'*65536)
            (root/'outside').symlink_to(outside,target_is_directory=True)
            (root/'file_alias').symlink_to(outside/'not_owned')
            result=resources.inventory([root])
            self.assertEqual(result['status'],'passed')
            self.assertEqual(result['regular_files'],1)
            self.assertEqual(result['logical_bytes'],16384)
            self.assertEqual(len(result['symlinks']),2)
            self.assertEqual(result['top_files'][0]['links'],2)
            expected=sum(p.lstat().st_blocks*512 for p in [root,root/'child',one,root/'outside',root/'file_alias'])
            self.assertEqual(result['allocated_bytes'],expected)
            self.assertTrue(result['directory_rows_overlap'])
            self.assertFalse(any('not_owned'in r['path']for r in result['top_files']))

    def test_overlapping_roots_are_not_counted_twice(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);child=root/'child';child.mkdir();(child/'x').write_text('x')
            result=resources.inventory([root,child]);only=resources.inventory([root])
            self.assertEqual(result['allocated_bytes'],only['allocated_bytes'])
            self.assertEqual(result['roots'][1]['allocated_bytes'],0)

    def test_symlink_root_is_review_and_missing_root_is_not_zero_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'actual').mkdir();(root/'alias').symlink_to(root/'actual',target_is_directory=True)
            result=resources.inventory([root/'alias',root/'absent'])
            self.assertEqual(result['roots'][0]['status'],'REVIEW_symlink_root_not_followed')
            self.assertEqual(result['status'],'incomplete')
            self.assertTrue(result['errors'])

    def test_memory_values_are_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'meminfo';path.write_text('MemTotal: 200 kB\nMemAvailable: 125 kB\n')
            self.assertEqual(resources.memory_available(path),{'MemTotal':204800,'MemAvailable':128000})


class ImportProbeTests(unittest.TestCase):
    def fake_packages(self,root):
        for name in ['torch','transformers','safetensors','specforge','sglang']:
            path=root/name;path.mkdir()
            content="__version__='fixture'\n"
            if name=='torch':
                content+="from types import SimpleNamespace\ncuda=SimpleNamespace(is_initialized=lambda: False)\nversion=SimpleNamespace(cuda='test-cuda')\n"
            (path/'__init__.py').write_text(content)
        pins={}
        for relative in resources.RUNTIME_FILES:
            path=root/'sglang'/'srt'/relative;path.parent.mkdir(parents=True,exist_ok=True)
            data=('# '+relative+'\n').encode();path.write_bytes(data);pins[relative]=hashlib.sha256(data).hexdigest()
        return pins

    def test_cpu_subprocess_records_real_origins_versions_and_all_seven_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pins=self.fake_packages(root)
            expected={'python_version':sys.version.split()[0],'torch':'fixture','cuda':'test-cuda','native_runtime_sha256':pins}
            result=resources.import_snapshot(sys.executable,root,'serving',expected)
            self.assertEqual(result['status'],'passed',result)
            receipt=result['receipt']
            self.assertFalse(receipt['cuda_initialized'])
            self.assertEqual(receipt['cuda_visible_devices'],'')
            self.assertEqual(receipt['modules']['torch']['realpath'],str(root/'torch/__init__.py'))
            self.assertEqual(len(receipt['runtime_files']),7)
            self.assertIn('distribution_versions',receipt)
            self.assertIn('editable_and_pth',receipt)

    def test_version_and_runtime_hash_failures_remain_visible(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pins=self.fake_packages(root);pins['models/dflash.py']='wrong'
            result=resources.import_snapshot(sys.executable,root,'serving',{
                'python_version':'not-current','native_runtime_sha256':pins})
            self.assertEqual(result['status'],'failed')
            self.assertEqual(len(result['mismatches']),2)
            self.assertTrue(any(r.get('runtime_file')=='models/dflash.py'for r in result['mismatches']))

    def test_training_import_from_an_old_checkout_is_rejected(self):
        data={'modules':{'specforge':{'realpath':'/old/checkout/specforge/__init__.py'}},'errors':[],
              'cuda_initialized':False,'runtime_files':{}}
        reply={'status':'passed','stdout':'RESOURCE_IMPORT_PROBE='+json.dumps(data),'stderr':''}
        with patch.object(resources,'command',return_value=reply):
            result=resources.import_snapshot('/x/python','/new/code','training',{})
        self.assertEqual(result['status'],'failed')
        self.assertEqual(result['mismatches'][0]['component'],'specforge_import_origin')

    def test_cuda_initialization_is_a_failure_even_when_versions_match(self):
        data={'modules':{},'errors':[],'cuda_initialized':True,'runtime_files':{}}
        reply={'status':'passed','stdout':'RESOURCE_IMPORT_PROBE='+json.dumps(data),'stderr':''}
        with patch.object(resources,'command',return_value=reply):
            result=resources.import_snapshot('/x/python','/x/code','training',{})
        self.assertEqual(result['status'],'failed')
        self.assertTrue(any('initialized CUDA' in row.get('error','')for row in result['mismatches']))


class ResourceContractTests(unittest.TestCase):
    def test_gpu_output_is_readonly_and_structured(self):
        responses=[{'status':'passed','stdout':'0, GPU-abc, A800, 81920, 100, 0\n'},
                   {'status':'passed','stdout':''}]
        with patch.object(resources,'command',side_effect=responses) as call:
            result=resources.gpu_snapshot()
        self.assertEqual(result['devices']['rows'][0]['uuid'],'GPU-abc')
        self.assertEqual(result['compute_jobs']['rows'],[])
        self.assertFalse(result['cuda_context_created'])
        self.assertTrue(all(c.args[0][0]=='nvidia-smi'for c in call.call_args_list))

    def test_light_check_never_imports_or_walks_storage(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(resources,'gpu_snapshot',return_value={}),patch.object(resources,'job_snapshot',return_value={}),patch.object(resources,'inventory')as walk,patch.object(resources,'import_snapshot')as imports:
                result=resources.resource_snapshot({'root':tmp},full_inventory=False)
            walk.assert_not_called();imports.assert_not_called()
            self.assertEqual(result['optimizer_steps'],0)
            self.assertGreater(result['disk']['free_bytes'],0)

    def test_missing_command_is_not_an_empty_success(self):
        with patch.object(resources.subprocess,'run',side_effect=FileNotFoundError('fixture')):
            result=resources.command(['missing'])
        self.assertEqual(result['status'],'unavailable')
        self.assertIn('error',result)


if __name__=='__main__':
    unittest.main()
