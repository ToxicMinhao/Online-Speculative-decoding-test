"""CPU-only package consumption, retained-identity and peak-storage checks."""
from copy import deepcopy
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from experiments.unchanged_c_mathqa_transfer import data as D, prepare as P


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def requests():
    rows=[]
    for i in range(258):
        question=f"Fixture {i}?"; options="a) 1, b) 2"
        prompt=f"user\n{question}\nOptions: {options}\nassistant"
        row=dict(source_id=f"mathqa:test:{i}",source_row=i,question=question,options=options,
                 user_message=question+"\n\nOptions: "+options,rendered_prompt=prompt,
                 prompt_ids=[1,i+10],prompt_tokens=2)
        row.update(D.fingerprints(question,options,prompt,row['prompt_ids']))
        rows.append(row)
    return D.seal_requests(rows)


def assembled_fixture(root):
    """Assemble real sealed files consumed by the production loader, without GPUs."""
    roles=requests()
    train=dict(start_ordinal=512,stop_ordinal=768,train_requests=[
        dict(source_id=f"mathqa:train:{i}",request_id=f"train_{i:05d}",request_index=i+1,
             prompt_ids=[2,i+1000],prompt_ids_sha256=D.token_digest([2,i+1000]),
             max_new_tokens=2048,stop_token_ids=[151645,151643],collect_feedback=True)
        for i in range(512,768)])
    values=dict(data_panel={'requests':roles['scored']},warmups={'requests':roles['warmup']},adaptation=train,
        source_provenance=dict(status='passed',ancestry_count=768,selected_ancestry_content_overlap=0,
                               selected_PR51_heldout_overlap=0),
        exposure_inventory=dict(status='passed',missing_dependencies=[]),
        evaluation_schedule=dict(activations=D.evaluation_schedule(roles['scored']),
                                 scored_executions=2304,warmup_executions=72))
    for event,offset in ((5,0),(6,128)):
        values[f'event{event}_inputs']=dict(event=event,start_ordinal=512+offset,
            stop_ordinal=640+offset,requests=train['train_requests'][offset:offset+128])
    plan=dict(root='/fixture/run',input_identities={},references='/fixture/run/references/CANONICALS.json',
              warmup_references='/fixture/run/references/WARMUP_CANONICALS.json')
    for key,value in values.items():
        name=P.INPUTS[key];path=root/'sealed'/name;write(path,value)
        plan[key]=plan['root']+'/sealed/'+name
        plan['input_identities'][plan[key]]=P.identity(path)
    seal=dict(status='passed',first_model_work_before_seal=False,input_only_eligibility=True,
              files={Path(p).name:v for p,v in plan['input_identities'].items()})
    path=root/'sealed/DATA_SEAL.json';write(path,seal)
    plan['data_seal']=plan['root']+'/sealed/DATA_SEAL.json'
    plan['input_identities'][plan['data_seal']]=P.identity(path)
    plan['binding']=P.plan_binding(plan)
    write(root/'sealed/PLAN.json',plan)
    return plan


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def assets(self):
        return dict(expected_existing_files={'/retained/state.pt':dict(bytes=9,sha256='a'*64)},
            common_e5={'feature_paths':['/retained/feature.pt']})

    def authentication(self):
        return dict(status='passed',historical_feature_identity_verified=True,files={
            '/retained/state.pt':dict(bytes=9,sha256='a'*64),
            '/retained/feature.pt':dict(bytes=5,sha256='b'*64)})

    def test_fresh_hashes_cannot_claim_historical_tensor_equality(self):
        receipt=self.authentication();receipt['historical_feature_identity_verified']=False
        with self.assertRaisesRegex(ValueError,'Historical common-E5'):
            P.require_asset_authentication(self.assets(),receipt)

    def test_missing_live_tensor_identity_fails(self):
        receipt=self.authentication();del receipt['files']['/retained/feature.pt']
        with self.assertRaisesRegex(ValueError,'packet'):
            P.require_asset_authentication(self.assets(),receipt)

    def test_changed_retained_identity_fails(self):
        receipt=self.authentication();receipt['files']['/retained/state.pt']['bytes']=10
        with self.assertRaisesRegex(ValueError,'retained asset'):
            P.require_asset_authentication(self.assets(),receipt)

    def test_matching_authenticated_fixture_preserves_all_original_pins(self):
        receipt=self.authentication()
        self.assertEqual(P.require_asset_authentication(self.assets(),receipt),receipt['files'])

    def test_missing_authentication_writes_blocker_before_any_package(self):
        destination=self.root/'blocked'
        with patch.object(P,'authenticated_retained_assets',return_value=self.assets()):
            with self.assertRaises(P.ReadinessBlocked):
                P.build(self.root,destination,remote_root='/future/run',source_root='/future/source',
                        execution_commit='fixture')
        receipt=P.read(destination/'READINESS.json')
        self.assertEqual(receipt['status'],'BLOCKED_RETAINED_ASSET_AUTHENTICATION')
        self.assertEqual(receipt['model_work'],0)
        self.assertFalse((destination/'sealed/PLAN.json').exists())
        self.assertEqual(list(destination.iterdir()),[destination/'READINESS.json'])

    def test_fresh_only_authentication_writes_blocker_and_stays_unqualified(self):
        auth=self.root/'fresh.json';value=self.authentication();value['historical_feature_identity_verified']=False;write(auth,value)
        destination=self.root/'blocked'
        with patch.object(P,'authenticated_retained_assets',return_value=self.assets()):
            with self.assertRaises(P.ReadinessBlocked):
                P.build(self.root,destination,remote_root='/future/run',source_root='/future/source',
                        execution_commit='fixture',asset_authentication=auth)
        self.assertFalse(P.read(destination/'READINESS.json')['gpu_qualified'])

    def test_actual_archive_unpacks_and_real_consumer_reads_all_seals(self):
        package=self.root/'package';plan=assembled_fixture(package)
        archive=self.root/'package.tar.gz'
        P.build_archive(plan,package,archive)
        unpacked=P.unpack_archive(archive,self.root/'unpacked')
        self.assertEqual(unpacked['readiness'],dict(status='passed',scored_inputs=256,warmup_inputs=2,
            adaptation_inputs=256,new_canonical_generations=258,before_worker_construction=True,model_work=0))
        self.assertEqual(unpacked['plan'],plan)
        self.assertFalse((self.root/'unpacked/references/CANONICALS.json').exists())

    def test_changed_input_fails_without_creating_archive(self):
        package=self.root/'package';plan=assembled_fixture(package)
        panel=P.read(package/'sealed/PANEL.json');panel['requests'][0]['prompt_ids'].append(999)
        write(package/'sealed/PANEL.json',panel)
        archive=self.root/'package.tar.gz'
        with self.assertRaisesRegex(ValueError,'changed packaged input'):
            P.build_archive(plan,package,archive)
        self.assertFalse(archive.exists())

    def test_new_panel_cannot_reuse_preexisting_canonicals(self):
        package=self.root/'package';plan=assembled_fixture(package)
        write(package/'references/CANONICALS.json',{})
        with self.assertRaisesRegex(ValueError,'already been attempted'):
            P.build_archive(plan,package,self.root/'package.tar.gz')

    def test_unpacked_archive_rejects_links_before_loader(self):
        archive=self.root/'link.tar.gz'
        with tarfile.open(archive,'w:gz') as handle:
            member=tarfile.TarInfo('sealed/PANEL.json');member.type=tarfile.SYMTYPE;member.linkname='../outside'
            handle.addfile(member)
        with self.assertRaisesRegex(ValueError,'link or special'):
            P.unpack_archive(archive,self.root/'unpacked')

    def test_three_arm_schedule_is_checked_not_just_file_counts(self):
        package=self.root/'package';plan=assembled_fixture(package)
        value=P.read(package/'sealed/EVALUATION_SCHEDULE.json');value['activations'][0]['method']='S'
        write(package/'sealed/EVALUATION_SCHEDULE.json',value)
        path=plan['evaluation_schedule'];plan['input_identities'][path]=P.identity(package/'sealed/EVALUATION_SCHEDULE.json')
        # Rebinding metadata cannot make an incorrect scientific schedule valid.
        seal=P.read(package/'sealed/DATA_SEAL.json');seal['files']['EVALUATION_SCHEDULE.json']=plan['input_identities'][path]
        write(package/'sealed/DATA_SEAL.json',seal);plan['input_identities'][plan['data_seal']]=P.identity(package/'sealed/DATA_SEAL.json')
        with self.assertRaisesRegex(ValueError,'three-arm schedule'):
            P.input_readiness(plan,package_root=package)

    def test_plan_identity_ignores_only_declared_administrative_fields(self):
        plan={'root':'/fixture','compiler_baseline_allocated_bytes':0,'estimated_seconds':{'final':10}}
        baseline=P.plan_binding(plan)
        changed=deepcopy(plan);changed['compiler_baseline_allocated_bytes']=10;changed['estimated_seconds']={'final':20}
        self.assertEqual(P.plan_binding(changed),baseline)
        changed['root']='/changed'
        self.assertNotEqual(P.plan_binding(changed),baseline)

    def test_storage_prices_full_capped_pool_without_duplicate_retirement(self):
        result=P.storage_plan([100]*128,weight_bytes=2_000_000_000,recovery_bytes=6_000_000_000,
                              hidden_size=4096,captured_layers=5)
        boundary=8_000_000_000+16*2**20
        self.assertEqual(result['checkpoint_bytes'],2*boundary)
        self.assertEqual(result['qualification_payload_bytes'],2*boundary)
        self.assertEqual(result['atomic_publication_bytes'],boundary)
        self.assertGreater(result['retained_duplicate_capture_bytes'],result['prepared_feature_bytes'])
        self.assertFalse(result['duplicate_retirement_required_for_admission'])
        self.assertEqual(result['feature_pools'],1)
        self.assertEqual(result['common_event5_copies'],0)
        self.assertEqual(result['required_initial_free_bytes'],result['peak_new_bytes']+400_000_000_000)

    def test_storage_over_budget_is_blocked_and_incomplete_geometry_rejected(self):
        result=P.storage_plan([2048]*128,weight_bytes=20_000_000_000,recovery_bytes=60_000_000_000,
                              hidden_size=4096,captured_layers=5)
        self.assertEqual(result['status'],'blocked')
        with self.assertRaisesRegex(ValueError,'128 E6 prompts'):
            P.storage_plan([100]*127,weight_bytes=1,recovery_bytes=1,hidden_size=1,captured_layers=1)


if __name__=='__main__':unittest.main()
