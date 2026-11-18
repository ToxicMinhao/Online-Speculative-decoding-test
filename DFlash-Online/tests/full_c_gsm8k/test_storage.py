"""Storage limits, exact predecessor evidence and generic C retirement wiring."""
from copy import deepcopy
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file
from experiments.full_c_gsm8k import storage as S
from online_feedback.rng import capture_rng
import test_kplus_storage as inherited


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)/'run';self.root.mkdir()
        self.cache=Path(self.temp.name)/'cache';self.cache.mkdir()
        variables={name:str(self.cache/name) for name in S.native.CACHE_ENVIRONMENT}
        variables['FLASHINFER_WORKSPACE_BASE']=str(self.cache/'flashinfer')
        env=patch.dict(os.environ,variables);env.start();self.addCleanup(env.stop)

    def test_peak_prices_all_complete_state_and_atomic_publication_overlap(self):
        plan=S.capacity_plan(2_097_259_104,complete_state_bytes=6_302_000_000)
        self.assertEqual(plan['allocation_limit_bytes'],100_000_000_000)
        self.assertEqual(plan['reserve_bytes'],400_000_000_000)
        self.assertEqual(plan['rolling_complete_roles'],['latest','predecessor','atomic_temporary'])
        self.assertTrue(plan['initial_retained_separately'])
        self.assertEqual(plan['publication_export_copies'],2)
        self.assertFalse(plan['compact_is_subcap'])
        self.assertLess(plan['projected_peak_bytes'],100_000_000_000)
        self.assertGreater(plan['feature_peak_bytes'],128*2272*4096*6*2)

    def test_compact_over_allowance_consumes_overall_budget_without_separate_rejection(self):
        plan=S.capacity_plan(1000)
        current=dict(tensor_bytes=0,compact_bytes=21_000_000_000,external_bytes=0,total_bytes=21_000_000_000)
        with patch.object(S,'allocation_snapshot',return_value=current),patch.object(S.shutil,'disk_usage',return_value=SimpleNamespace(free=500_000_000_000)):
            receipt=S.admit(self.root,plan)
        self.assertEqual(receipt['status'],'passed')
        self.assertEqual(receipt['compact_allowance_excess_bytes'],1_000_000_000)
        self.assertEqual(receipt['remaining_components']['compact'],0)

    def test_overall_or_shared_reserve_failure_is_durable(self):
        plan=S.capacity_plan(1000)
        for free,current in ((399_000_000_000,None),(10**15,dict(tensor_bytes=0,compact_bytes=101_000_000_000,external_bytes=0,total_bytes=101_000_000_000))):
            destination=self.root/('FAIL_'+str(free)+'.json')
            with patch.object(S.shutil,'disk_usage',return_value=SimpleNamespace(free=free)):
                if current is None:
                    with self.assertRaises(ValueError):S.admit(self.root,plan,receipt_path=destination)
                else:
                    with patch.object(S,'allocation_snapshot',return_value=current),self.assertRaises(ValueError):S.admit(self.root,plan,receipt_path=destination)
            self.assertEqual(json.loads(destination.read_text())['status'],'failed')

    def test_external_and_cache_growth_are_charged_and_hardlinks_count_once(self):
        external=Path(self.temp.name)/'source';external.mkdir();(external/'code.py').write_bytes(b'a'*8192)
        tensor=self.root/'weights.pt';tensor.write_bytes(b'x'*4096);(self.root/'link.pt').hardlink_to(tensor)
        plan=S.capacity_plan(1000,external_roots=[external],compiler_bytes=4096)
        with patch.object(S.shutil,'disk_usage',return_value=SimpleNamespace(free=10**15)):
            before=S.admit(self.root,plan)
            compiled=self.cache/'TORCHINDUCTOR_CACHE_DIR';compiled.mkdir();(compiled/'large.so').write_bytes(b'a'*8192)
            after=S.admit(self.root,plan)
        self.assertEqual(before['current']['tensor_bytes'],tensor.stat().st_blocks*512)
        self.assertGreater(after['compiler_growth_bytes'],4096)
        self.assertEqual(after['status'],'passed')
        self.assertGreater(before['current']['external_bytes'],0)
        (self.root/'symlink').symlink_to(tensor)
        with self.assertRaisesRegex(ValueError,'symlink'):S.allocation_snapshot(self.root)

    def test_only_manifest_authenticated_internal_source_links_are_charged(self):
        external=Path(self.temp.name)/'source';external.mkdir()
        target=external/'actual.py';target.write_bytes(b'authored source')
        link=external/'alias.py';link.symlink_to('actual.py')
        manifest=self.root/'SOURCE.json'
        manifest.write_text(json.dumps(dict(commit='a'*40,dirty=False,files={
            'alias.py':dict(kind='symlink',target='actual.py'),'actual.py':S.identity(target)})))
        policy=dict(source_root=str(external),source_manifest=str(manifest),execution_commit='a'*40)
        result=S.allocation_snapshot(self.root,[external],source_policy=policy)
        expected=sum(path.lstat().st_blocks*512 for path in (external,target,link))
        self.assertEqual(result['external_bytes'],expected)
        link.unlink();link.symlink_to('../run/SOURCE.json')
        with self.assertRaisesRegex(ValueError,'authenticated internal'):
            S.allocation_snapshot(self.root,[external],source_policy=policy)

    def checkpoint(self):
        value=torch.tensor([1.,2.],dtype=torch.bfloat16)
        state=dict(draft_state_dict={'weight':value},backend={'optimizer':{'state':{'step':torch.tensor(128.),'exp_avg':value.clone(),'exp_avg_sq':value.clone()}}},
            rng=capture_rng(),events=[{'event_id':'event_001'}],global_step=128,presentations=256,
            horizon=7424,accumulation_steps=2,algorithm_contract={'method':'CPU fixture'},config_text='fixture',run_id='fixture')
        path=self.root/'previous.pt';torch.save(state,path)
        export=self.root/'previous_export';export.mkdir();save_file(state['draft_state_dict'],export/'model.safetensors')
        return path,export,state

    def test_every_model_optimizer_and_rng_tensor_is_fingerprinted_before_retirement(self):
        path,export,state=self.checkpoint();destination=self.root/'FINGERPRINTS.json'
        result=S.checkpoint_fingerprints(path,expected_identity=S.identity(path),export_path=export,receipt_path=destination)
        self.assertTrue(result['exact_export_tensors'])
        self.assertEqual(result['tensors']['draft_state_dict']['weight']['shape'],[2])
        self.assertIn('sha256',result['tensors']['backend']['optimizer']['state']['exp_avg'])
        self.assertIn('sha256',result['tensors']['rng']['torch'])
        self.assertEqual(result['event_history']['event_ids'],['event_001'])
        self.assertTrue(path.exists());self.assertTrue((export/'model.safetensors').exists())
        save_file({'weight':state['draft_state_dict']['weight']+1},export/'model.safetensors')
        with self.assertRaisesRegex(ValueError,'exact export'):S.checkpoint_fingerprints(path,expected_identity=S.identity(path),export_path=export,receipt_path=self.root/'BAD.json')
        self.assertFalse((self.root/'BAD.json').exists())

    def test_retirement_adapter_preserves_generic_native_c_arm_and_evidence_order(self):
        path,export,state=self.checkpoint();event=self.root/'event_002';event.mkdir()
        feature=event/'feature.pt';torch.save({'record_id':'r','input_ids':torch.tensor([[1,2]]),'hidden_states':torch.ones(1,2,3,dtype=torch.bfloat16)},feature)
        previous=dict(state=str(path),state_identity=S.identity(path),weights_identity=S.identity(export/'model.safetensors'))
        receipt={'preparation':[{'feature_path':str(feature)}]}
        def retire(root,event_root,records,received,previous_export,previous_boundary,current_boundary,*,arm):
            self.assertEqual(arm,'KPLUS_C')
            self.assertTrue((event/'PREDECESSOR_TENSOR_FINGERPRINTS.json').is_file())
            self.assertTrue((event/'FEATURE_RETIREMENT_IDENTITIES.json').is_file())
            return {'status':'completed'}
        with patch.object(S.native,'retire_event',side_effect=retire) as method:
            self.assertEqual(S.retire_event(self.root,event,[],receipt,export,previous,{'state':str(self.root/'current.pt')}),{'status':'completed'})
        self.assertEqual(method.call_count,1)
        self.assertTrue(path.exists())  # CPU test does not delete real experiment payloads.

    def test_only_passed_disposable_qualification_payloads_are_retired(self):
        from experiments.persistent_prefix_withholding.qualification import tree_identity
        from online_feedback.storage import write_json
        _,_,active_state=self.checkpoint()
        base=self.root/'qualification/training_B';(base/'trainer').mkdir(parents=True)
        initial=base/'trainer/initial_state.pt';active=base/'active_state.pt'
        initial_state=deepcopy(active_state)
        initial_state.update(global_step=0,presentations=0,events=[])
        initial_state['backend']={'optimizer':{'state':{}}}
        torch.save(initial_state,initial);torch.save(active_state,active)
        released=self.root/'released';released.mkdir();save_file(initial_state['draft_state_dict'],released/'model.safetensors')
        export=base/'active_export';reconstructed=base/'cpu_reconstructed_export'
        for folder in (export,reconstructed):
            folder.mkdir();save_file(active_state['draft_state_dict'],folder/'model.safetensors')
            write_json(folder/'config.json',{'fixture':True})
        payloads=[initial,active,*export.iterdir(),*reconstructed.iterdir()]
        detail=dict(status='passed',cpu_reconstruction_verified=True,
            initialization={'state_path':str(initial)},snapshot_path=str(active),export=str(export),
            disposable_payload_paths=list(map(str,[initial,active,export,reconstructed])),
            disposable_payload_files=list(map(str,payloads)),
            cpu_export=dict(status='passed',exact_weights_file_identity=True,
                weights_files={'model.safetensors':S.identity(export/'model.safetensors')}),
            origin_before=dict(weights=tree_identity(initial_state['draft_state_dict']),
                optimizer=tree_identity(initial_state['backend']),rng=tree_identity(initial_state['rng'])),
            cases=[dict(case='active',endpoint=dict(weights=tree_identity(active_state['draft_state_dict']),
                optimizer=tree_identity(active_state['backend']['optimizer']),rng=tree_identity(active_state['rng']),
                global_step=128,presentations=256))])
        qualification=dict(status='passed',training={'status':'completed','result':detail},serving={'status':'completed'})
        for path,value in ((self.root/'QUALIFICATION.json',qualification),(base/'DETAILS.json',detail),
            (base/'CPU_EXPORT_QUALIFICATION.json',detail['cpu_export']),
            (self.root/'phases/qualification_training/STATUS.json',qualification['training']),
            (self.root/'phases/qualification_serving/STATUS.json',qualification['serving'])):
            write_json(path,value)
        production=self.root/'campaign_001/arms/KPLUS_C/trainer/initial_state.pt'
        production.parent.mkdir(parents=True);production.write_bytes(b'protected production fixture')
        plan={'root':str(self.root),'draft':str(released),'initial_witness':str(released/'model.safetensors')}
        bad=deepcopy(qualification);bad['serving']['status']='failed'
        with self.assertRaisesRegex(ValueError,'passed training'):S.retire_qualification(plan,bad)
        self.assertTrue(all(p.exists() for p in payloads))
        receipt=S.retire_qualification(plan,qualification)
        self.assertEqual(receipt['status'],'completed')
        self.assertTrue(all(not p.exists() for p in payloads))
        self.assertEqual(production.read_bytes(),b'protected production fixture')
        self.assertTrue((base/'ACTIVE_RETIREMENT_FINGERPRINTS.json').is_file())
        self.assertTrue((released/'model.safetensors').is_file())

    def test_exact_raw_to_prepared_check_is_reused_without_weakening(self):
        event,record,prepared,capture,terminal,feature=inherited.StorageTests.prepared_fixture(self,self.root)
        changed=torch.load(feature,weights_only=False);changed['hidden_states'][0,0,0]+=1;torch.save(changed,feature)
        with self.assertRaisesRegex(ValueError,'exactly preserve'):
            S.retire_redundant_features(record,prepared,self.root,event)
        self.assertTrue(capture.exists());self.assertTrue(terminal.exists())


if __name__=='__main__':unittest.main()
