"""Typed replay prerequisite/main transition without model or optimizer work."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.unchanged_c_mathqa_transfer import replay_gate as G,prepare as P


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')
    return {'path':str(path),'identity':P.identity(path)}


def fixture(root):
    rows=[]
    for i in range(128):
        path=root/'packets'/f'{i}.pt';path.parent.mkdir(exist_ok=True);path.write_bytes(f'CPU fixture {i}'.encode())
        rows.append(dict(path=str(path),identity=P.identity(path),metadata={'record_id':str(i)},
            tensors={name:dict(shape=[1,2],dtype='torch.bfloat16',sha256='a'*64) for name in G.TENSOR_FIELDS},
            matches_original_token_mask_shape_metadata=True,old_content_hash_available=False))
    common=dict(manifest='/old/FEATURES.json',manifest_identity={'bytes':20,'sha256':'b'*64},
        own_checkpoint='/old/origin/export',own_weights_identity={'bytes':30,'sha256':'c'*64},
        feature_paths=[r['path'] for r in rows],feature_identities={r['path']:r['identity'] for r in rows})
    state=root/'original.pt';state.write_bytes(b'original state')
    original={str(state):P.identity(state)}
    plan=dict(root=str(root/'execution'),binding='fixed-scientific-plan',execution_commit='published-source',
        source_root='/source',runtime_contract={'torch':'fixed'},common_e5=common,
        retained_asset_identities=original,origin_state_identity=P.identity(state),
        origin_weights_identity=common['own_weights_identity'],pr51_binding='original-binding',
        retained_endpoints={a+'5':{'endpoint':{'state':'/old/'+a+'5.pt','state_identity':{'bytes':5,'sha256':a*64}}} for a in ('N','S')})
    pool=dict(kind=G.CURRENT_FEATURES,status='current_pool_sealed',historical_feature_identity_verified=False,
        model_work=0,manifest_path=common['manifest'],manifest_identity=common['manifest_identity'],
        producer=common['own_checkpoint'],producer_identity=common['own_weights_identity'],features=rows)
    manifest=write(root/'current.json',pool)
    plan['prerequisite']=dict(kind=G.PREREQUISITE,authorized_arms=['N','S'],event=5,
        historical_feature_identity_verified=False,feature_manifest=manifest,
        qualified_receipt=plan['root']+'/replay/FUNCTIONAL_E5_REPLAY_QUALIFIED.json')
    auth=dict(kind=G.CURRENT_AUTHENTICATION,status='passed',historical_feature_identity_verified=False,
        authorized_work=['N5_REPLAY','S5_REPLAY'],feature_manifest_identity=manifest['identity'],
        files={**original,**common['feature_identities']})
    plan['asset_authentication']=write(root/'current_auth.json',auth)
    for arm in ('N','S'):
        folder=Path(plan['root'])/'prerequisite'/('replay_'+arm+'5')
        prereq=G.validate_prerequisite(plan,auth)
        prereq['current_features']={'feature_manifest_identity':manifest['identity']}
        result=dict(schema='pr52_e5_control_replay_v1',status='passed',arm=arm,event=5,
            binding=plan['binding'],source_binding=plan['pr51_binding'],execution_commit=plan['execution_commit'],
            historical_endpoint=plan['retained_endpoints'][arm+'5']['endpoint'],
            counts=dict(optimizer_steps=128,response_presentations=256,start_step=512,end_step=640,end_presentations=1280),
            historical_feature_identity_verified=False,checkpoint_files_written=0,export_files_written=0,
            prerequisite=prereq)
        comparison=dict(schema='mathqa_e5_exact_replay_comparison_v1',status='passed',arm=arm,
            responses_compared=256,windows_compared=128,presentations_compared=256,event_compared=True,endpoint_compared=True,
            categories={name:'exact' for name in ('draft_tensors','optimizer_moments_counters_options','rng',
                'scientific_configuration','original_event_ancestry','sampler_accumulation_boundary','event_scientific_receipts')},
            first_divergence=None,historical_feature_identity_verified=False,exclusions=deepcopy(G.COMPARISON_EXCLUSIONS))
        write(folder/'RESULT.json',result);write(folder/'comparison/COMPARISON.json',comparison)
    return plan


class ReplayGateTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.plan=fixture(self.root)

    def qualified(self):
        base=Path(self.plan['root'])/'prerequisite'
        return G.make_functional_receipt(self.plan,base/'replay_N5/RESULT.json',base/'replay_S5/RESULT.json')

    def test_current_manifest_is_not_main_authority(self):
        current=G.read(self.plan['asset_authentication']['path'])
        self.assertEqual(G.validate_prerequisite(self.plan,current)['authorized_work'],['N5_REPLAY','S5_REPLAY'])
        with self.assertRaisesRegex(ValueError,'Both complete'):
            G.validate_main_receipt(self.plan,current)
        with self.assertRaisesRegex(ValueError,'only original N5'):
            G.validate_replay_prerequisite(self.plan,'C')

    def test_both_exact_full_event_receipts_qualify_without_historical_claim(self):
        receipt=self.qualified();result=G.validate_main_receipt(self.plan,receipt)
        self.assertFalse(result['historical_feature_identity_verified'])
        self.assertTrue(result['functional_e5_replay_verified'])
        self.assertEqual(receipt['context']['clock'],self.plan['root']+'/CLOCK.json')

    def test_missing_scaled_receipt_cannot_pass(self):
        receipt=self.qualified();del receipt['comparisons']['S']
        with self.assertRaisesRegex(ValueError,'Both complete'):
            G.validate_main_receipt(self.plan,receipt)

    def test_modified_current_feature_invalidates_qualified_receipt(self):
        receipt=self.qualified();Path(self.plan['common_e5']['feature_paths'][0]).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'feature changed after replay'):
            G.validate_main_receipt(self.plan,receipt)

    def test_endpoint_only_or_missing_scientific_receipt_fields_fail(self):
        base=Path(self.plan['root'])/'prerequisite/replay_S5/comparison/COMPARISON.json'
        value=G.read(base);value['responses_compared']=255;write(base,value)
        with self.assertRaisesRegex(ValueError,'exact scientific'):
            self.qualified()

    def test_missing_optimizer_category_is_not_a_match(self):
        base=Path(self.plan['root'])/'prerequisite/replay_N5/comparison/COMPARISON.json'
        value=G.read(base);del value['categories']['optimizer_moments_counters_options'];write(base,value)
        with self.assertRaisesRegex(ValueError,'exact scientific'):
            self.qualified()

    def test_changed_clock_root_source_runtime_or_pool_binding_invalidates_receipt(self):
        receipt=self.qualified()
        for field,value in (('root','/new-clock'),('binding','other'),('execution_commit','other'),
                            ('runtime_contract',{'torch':'different'})):
            changed=deepcopy(self.plan);changed[field]=value
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'Both complete'):
                G.validate_main_receipt(changed,receipt)

    def test_changed_comparison_bytes_preserve_failure_instead_of_accepting(self):
        receipt=self.qualified();path=Path(receipt['comparisons']['N']['comparison']['path'])
        value=G.read(path);value['status']='failed';write(path,value)
        with self.assertRaisesRegex(ValueError,'Changed replay evidence'):
            G.validate_main_receipt(self.plan,receipt)

    def test_false_historical_claim_is_required_in_both_stages(self):
        receipt=self.qualified();receipt['historical_feature_identity_verified']=True
        with self.assertRaisesRegex(ValueError,'Both complete'):
            G.validate_main_receipt(self.plan,receipt)
        auth=G.read(self.plan['asset_authentication']['path']);auth['historical_feature_identity_verified']=True
        with self.assertRaisesRegex(ValueError,'cannot authorize'):
            G.validate_prerequisite(self.plan,auth)

    def test_current_manifest_builder_uses_original_producer_and_order(self):
        manifest=G.read(self.plan['prerequisite']['feature_manifest']['path'])
        pool={k:manifest[k] for k in ('manifest_path','manifest_identity','producer','producer_identity','features')}
        audit=dict(status='current_files_validated_historical_content_unproven',historical_feature_identity_verified=False,
                   model_work=0,cuda_initialized=False,pools={'common5':pool})
        self.assertEqual(G.make_current_manifest({'common_e5':self.plan['common_e5']},audit),manifest)
        pool['features'].reverse()
        with self.assertRaisesRegex(ValueError,'producer or order'):
            G.make_current_manifest({'common_e5':self.plan['common_e5']},audit)

    def test_current_authentication_does_not_enable_original_historical_route(self):
        auth=G.read(self.plan['asset_authentication']['path'])
        assets=dict(expected_existing_files=self.plan['retained_asset_identities'],common_e5=self.plan['common_e5'])
        P.require_current_authentication(assets,auth,self.plan['prerequisite']['feature_manifest']['identity'])
        with self.assertRaisesRegex(ValueError,'Historical common-E5'):
            P.require_asset_authentication(assets,auth)

    def test_prerequisite_gate_checks_tensors_before_returning_pool(self):
        with patch.object(G,'require_current_features',side_effect=ValueError('tensor changed')) as checked:
            with self.assertRaisesRegex(ValueError,'tensor changed'):
                G.validate_replay_prerequisite(self.plan,'N')
            checked.assert_called_once_with(self.plan)

    def test_real_cpu_tensor_bytes_detect_a_manifest_disagreement(self):
        import hashlib
        import torch
        from online_feedback import training
        manifest=G.read(self.plan['prerequisite']['feature_manifest']['path'])
        packet={name:torch.tensor([[1.,2.]],dtype=torch.bfloat16) for name in G.TENSOR_FIELDS}
        for row in manifest['features']:
            torch.save(packet,row['path'])
            row['identity']=P.identity(row['path'])
            row['metadata']={'fixture':True}
            row['tensors']={name:dict(shape=list(v.shape),dtype=str(v.dtype),
                sha256=hashlib.sha256(v.view(torch.uint8).numpy().tobytes()).hexdigest()) for name,v in packet.items()}
            self.plan['common_e5']['feature_identities'][row['path']]=row['identity']
        self.plan['prerequisite']['feature_manifest']=write(self.root/'current.json',manifest)
        with patch.object(training,'validate_feature',return_value={'fixture':True}),patch.object(training,'validate_target_binding'):
            self.assertEqual(G.require_current_features(self.plan)['features'],128)
            manifest['features'][0]['tensors']['hidden_states']['sha256']='0'*64
            self.plan['prerequisite']['feature_manifest']=write(self.root/'current.json',manifest)
            with self.assertRaisesRegex(ValueError,'tensor changed'):
                G.require_current_features(self.plan)
        self.assertFalse(torch.cuda.is_initialized())

    def test_full_prerequisite_builder_seals_both_stages_before_archive(self):
        from unchanged_c_mathqa_transfer.test_data_prepare import assembled_fixture
        from experiments.unchanged_c_mathqa_transfer import state
        data_root=self.root/'research'/P.TASK_NAME/'data_readiness'
        assembled_fixture(data_root)
        delivery=self.root/'old_delivery';sealed=delivery/'sealed';sealed.mkdir(parents=True)
        original_text='run_id: historical\noutput_dir: /historical\ntraining:\n  learning_rate: 0.0003\n'
        (sealed/'ORIGINAL_CONFIG.yaml').write_text(original_text)
        names=['MODEL_PIN.json','RUNTIME_PIN.json','TRAINING_PIN.json','ORIGIN_STATUS.json',
               'ORIGIN_TRAINING.json','ORIGIN_REMOTE_CHECK.json','HISTORICAL_PLAN.json','HISTORICAL_SEAL.json']
        for name in names:write(sealed/name,{})
        write(sealed/'ORIGIN_DRAFT_CONFIG.json',{'hidden_size':4096,'dflash_config':{'target_layer_ids':[1,2,3,4,5]}})
        for arm,lr in [('N','0.0003'),('S','0.000288')]:
            (sealed/('config_'+arm+'.yaml')).write_text('training:\n  learning_rate: '+lr+'\n')
        old=dict(target='/target',draft='/draft',training_python='/train/python',worker_python='/serve/python',
            origin_state='/origin/state',origin_export='/origin/export',origin_state_identity={'bytes':6000000000,'sha256':'e'*64},
            origin_weights_identity={'bytes':2000000000,'sha256':'f'*64},origin_binding='old-binding',
            devices={'A':'a','B':'b'},qualified_compilers={'CC':'cc','CXX':'c++'},host_resource_policy={},
            runtime_contract={},compiler_root='/compiler',compiler_growth_reserved_bytes=4*2**30,
            checkpoint_boundary_allowance_s=1,final_activation_allowance_s=1,binding='oldbinding')
        write(sealed/'PLAN.json',old)
        write(delivery/'RETRIEVAL_MANIFEST.json',{'sealed/'+p.name:P.identity(p) for p in sealed.iterdir()})
        assets=dict(plan=old,delivery=str(delivery),common_e5=self.plan['common_e5'],
            retained_endpoints=self.plan['retained_endpoints'],retained_pools={},
            expected_existing_files=self.plan['retained_asset_identities'])
        output=self.root/'package'
        with patch.object(P,'authenticated_retained_assets',return_value=assets),patch.object(state,'validate_config_text') as validate:
            plan=P.build_prerequisite(self.root,output,remote_root='/remote/run',source_root='/remote/source',
                execution_commit='sealed-commit',asset_authentication=self.plan['asset_authentication']['path'],
                current_feature_manifest=self.plan['prerequisite']['feature_manifest']['path'])
            validate.assert_called_once()
        self.assertEqual(plan['prerequisite']['kind'],G.PREREQUISITE)
        self.assertEqual(set(plan['replay_configs']),{'N','S'})
        self.assertEqual(plan['expected_work']['adam'],524)
        self.assertEqual(plan['expected_work']['response_forward'],1060)
        self.assertEqual(plan['storage']['replay_checkpoint_bytes'],0)
        self.assertEqual(plan['storage']['replay_publication_bytes'],0)
        self.assertFalse(Path(plan['prerequisite']['qualified_receipt']).exists())
        archive=self.root/'prepared.tar.gz';P.build_archive(plan,output,archive)
        loaded=P.unpack_archive(archive,self.root/'unpacked')['plan']
        self.assertEqual(loaded,plan)
        self.assertEqual((self.root/'unpacked/sealed/replay_config_S.yaml').read_bytes(),(sealed/'config_S.yaml').read_bytes())
        self.assertEqual(P.plan_binding(loaded),loaded['binding'])


if __name__=='__main__':unittest.main()
