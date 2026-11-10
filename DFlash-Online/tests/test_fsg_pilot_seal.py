"""Conditional pilot admission is evidence-derived and changes no scientific inputs."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


from experiments.functional_step_geometry import seal_pilot as S
from experiments.functional_step_geometry.contract import ARMS, DATASETS, MAIN_BUDGET, DIAGNOSTIC_BUDGET
from online_feedback.storage import write_json
from test_ok_coding_training import configuration


def receipts():
    plan = dict(scope='bf16_realization_r1_qualification', realization_revision=S.REALIZATION_R1,
        geometry=S.geometry_contract(S.REALIZATION_R1),future_pilot_budget=deepcopy(MAIN_BUDGET),
        future_diagnostic_budget=deepcopy(DIAGNOSTIC_BUDGET),execution_source=dict(commit='a'*40))
    datasets = {}
    for dataset in DATASETS:
        cases = []
        for arm in ('FSG','SCALAR'):
            step = dict(decision=dict(accepted=arm=='FSG',reason='fixture'),
                committed=dict(identical_native=arm=='SCALAR',identical_prestate=False),
                final_candidate_check='actual_bf16_replay',scalar_matching_exercised=arm=='SCALAR')
            cases.append(dict(status='passed',case=arm,optimizer_steps=1,presentations=2,
                              hook=dict(steps=[step])))
        datasets[dataset] = dict(cases=cases,serving=dict(status='passed',generations=2,
                                                       exact_output_and_termination=True))
    result = dict(status='passed',revision=S.REALIZATION_R1,execution_commit='a'*40,
        main_pilot_started=False,limits=deepcopy(S.LIMITS),datasets=datasets,
        work={name+'_'+phase:value for name,value in S.LIMITS.items() for phase in ('attempted','completed')})
    result['activity'] = S.activity(datasets)
    formula = dict(status='passed',numerical_protocol='independent_same_native_layout',
        tolerances=dict(atol=1e-5,rtol=.016,maximum_relative_l2=.016),native_vjp=dict(status='passed'))
    previous = dict(status='passed',datasets=dict(
        gsm8k=dict(oracles=dict(formula=deepcopy(formula),resource=dict(status='passed'))),
        mathqa=dict(cases=[dict(case='FSG',oracles=deepcopy(formula))])))
    postflight = dict(status='passed',postflight=True,execution_commit='a'*40)
    return plan,result,postflight,previous


def fixture(root):
    qualification, prior, code = root/'qualification',root/'prior',root/'code'
    package, retained = qualification/'sealed_package',qualification/'retained'
    remote = Path('/remote/qualification'); original = configuration(17)
    write_json(code/'source.json',{'source':'fixture'})
    sealed = package/'sealed'; sealed.mkdir(parents=True)
    (sealed/'ORIGINAL_CONFIG.yaml').write_text(original)
    plan,result,postflight,previous = receipts()
    plan.update(root=str(remote),binding='b'*64,arms=list(ARMS),datasets={},
        original_config=str(remote/'sealed/ORIGINAL_CONFIG.yaml'),
        supplementary_reuse=dict(prior_root='/remote/previous'))
    preparation = dict(status='completed',datasets={})
    for dataset in DATASETS:
        directory = sealed/dataset
        train = [dict(request_id=f'{dataset}_{i}',event_id=f'event_{i//128+5:03d}') for i in range(256)]
        panel = [dict(request_id=f'{dataset}_eval_{i}',panel_ordinal=i,prompt_ids=[i+5],
            stop_token_ids=[2],original_index=i,max_new_tokens=2048) for i in range(256)]
        canonicals = {row['request_id']:dict(prompt_ids=row['prompt_ids'],response_ids=[7,2],
            verified=True,finish_reason='eos',binding_digest='reference') for row in panel}
        write_json(directory/'TRAINING.json',dict(train_requests=train))
        write_json(directory/'PANEL.json',dict(requests=panel))
        write_json(directory/'CANONICALS.json',canonicals)
        item = dict(origin_state=f'/origin/{dataset}/state.pt',origin_export=f'/origin/{dataset}/export',
            origin_weights=f'/origin/{dataset}/export/model.safetensors',origin_binding='c'*64,
            origin_state_identity=dict(bytes=6297245019,sha256='d'*64),
            origin_weights_identity=dict(bytes=2097259104,sha256='e'*64),configs={})
        for key,name in (('training_manifest','TRAINING.json'),('panel_manifest','PANEL.json'),('references','CANONICALS.json')):
            item[key] = str(remote/'sealed'/dataset/name)
        for arm in ARMS:
            (directory/f'{arm}.yaml').write_text(original)
            item['configs'][arm] = str(remote/'sealed'/dataset/f'{arm}.yaml')
        plan['datasets'][dataset] = item
        preparation['datasets'][dataset] = [dict(request_id=train[i]['request_id'],status='completed',
            record_path=f'/packets/{dataset}/{i}.json',feature_path=f'/packets/{dataset}/{i}.pt',
            record_identity=dict(bytes=4,sha256='0'*64),feature_identity=dict(bytes=8,sha256='1'*64)) for i in (0,1)]
    write_json(prior/'retained/qualification/QUALIFICATION.json',previous)
    write_json(sealed/'PLAN.json',plan)
    files = {str(remote/p.relative_to(package)):S.identity(p) for p in sealed.rglob('*') if p.is_file()}
    files['/remote/previous/qualification/QUALIFICATION.json'] = S.identity(prior/'retained/qualification/QUALIFICATION.json')
    write_json(sealed/'SEAL.json',dict(files=files))
    write_json(retained/'qualification/QUALIFICATION.json',result)
    write_json(retained/'POSTFLIGHT.json',postflight)
    write_json(retained/'qualification_feedback/PREPARATION.json',preparation)
    (retained/'EXIT_CODE').write_text('0\n')
    entries = {str(p.relative_to(retained)):S.identity(p) for p in retained.rglob('*') if p.is_file()}
    write_json(retained/'COMPACT_RETRIEVAL_MANIFEST.json',dict(root=str(remote),files=entries))
    return qualification,prior,code


class PilotSealTests(unittest.TestCase):
    def test_admission_derives_active_fsg_and_exercised_but_rejected_scalar(self):
        plan,result,post,prior = receipts()
        actual = S.qualification_admission(plan,result,post,'0\n',prior)
        self.assertEqual(actual['nontrivial_accepted_fsg_cases'],2)
        self.assertEqual(actual['scalar_matching_exercised_cases'],2)
        self.assertFalse(actual['storage_admitted'])
        self.assertTrue(all(actual['checks'].values()))

    def test_failures_and_forged_activity_cannot_admit_pilot(self):
        for change in (
            lambda p,q,r,h:q.update(status='failed'),
            lambda p,q,r,h:r.update(status='failed'),
            lambda p,q,r,h:q['work'].update(additional_backward_failed=1),
            lambda p,q,r,h:q['work'].update(adam_completed=3),
            lambda p,q,r,h:q['datasets']['gsm8k']['serving'].update(exact_output_and_termination=False),
            lambda p,q,r,h:q['datasets']['mathqa']['cases'][0].update(status='failed'),
            lambda p,q,r,h:q['activity'].update(nontrivial_fsg=False),
            lambda p,q,r,h:h['datasets']['gsm8k']['oracles']['resource'].update(status='failed'),
            lambda p,q,r,h:h['datasets']['gsm8k']['oracles']['formula']['tolerances'].update(atol=.1),
        ):
            args=receipts(); change(*args)
            with self.assertRaises(ValueError): S.qualification_admission(*args[:3],'0',args[3])
        args=receipts()
        with self.assertRaises(ValueError): S.qualification_admission(*args[:3],'1',args[3])

    def test_inactive_or_unexercised_controls_fail_even_with_consistent_summary(self):
        for arm in ('FSG','SCALAR'):
            plan,result,post,prior=receipts()
            for data in result['datasets'].values():
                step=next(c for c in data['cases'] if c['case']==arm)['hook']['steps'][0]
                if arm=='FSG': step['committed']['identical_native']=True
                else: step['scalar_matching_exercised']=False
            result['activity']=S.activity(result['datasets'])
            with self.assertRaises(ValueError): S.qualification_admission(plan,result,post,'0',prior)

    def test_scope_geometry_budget_and_revision_are_not_mutable(self):
        for mutation in (
            lambda p:p.update(scope='main_pilot'),
            lambda p:p['geometry'].update(strength=2),
            lambda p:p['future_pilot_budget'].update(optimizer_steps=1537),
            lambda p:p['future_diagnostic_budget'].update(windows=17),
            lambda p:p.update(realization_revision='legacy_raw_norm_v0'),
        ):
            p,q,r,h=receipts();mutation(p)
            with self.assertRaises(ValueError):S.qualification_admission(p,q,r,'0',h)

    def test_full_seal_preserves_origins_schedule_and_reuses_four_packets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); qualification,prior,code=fixture(root)
            original=S.read(qualification/'sealed_package/sealed/PLAN.json')
            def git(args,**kwargs):
                return 'refs/remotes/origin/fixture\n' if args[1]=='for-each-ref' else b'source.json\0'
            def prepared(package,old,binding,proof):
                return {d:dict(canonicals=S.read(package/'sealed'/d/'CANONICALS.json'),
                    schedule=S.evaluation_schedule(S.read(package/'sealed'/d/'PANEL.json')['requests'],d),
                    proof=dict(files=[]),native=dict(records=[])) for d in DATASETS}
            with patch.object(S,'prepare_canonical_inputs',side_effect=prepared), \
                 patch.object(S,'published_source',return_value=dict(commit='f'*40,dirty=False)), \
                 patch.object(S.subprocess,'check_output',side_effect=git):
                receipt=S.seal(code,'f'*40,root/'out',qualification=qualification,prior=prior)
            plan=S.read(root/'out/sealed/PLAN.json'); seal=S.read(root/'out/sealed/SEAL.json')
            self.assertEqual(plan['scope'],S.SCOPE)
            self.assertEqual(plan['operational_timeout'],S.OPERATIONAL_TIMEOUT)
            self.assertEqual(plan['events'],[5,6]);self.assertEqual(plan['main_budget'],MAIN_BUDGET)
            self.assertEqual(plan['main_storage']['required_initial_free_bytes'],188444277887)
            self.assertEqual(receipt['feedback_accounting']['new_feedback'],508)
            self.assertEqual(receipt['feedback_accounting']['reused_feedback'],4)
            self.assertFalse(receipt['actual_storage_admitted'])
            for dataset,item in plan['datasets'].items():
                self.assertEqual(item['origin_state'],original['datasets'][dataset]['origin_state'])
                self.assertEqual(item['origin_export'],original['datasets'][dataset]['origin_export'])
                self.assertEqual(len(item['evaluation_schedule']),1536)
                self.assertEqual(item['evaluation_schedule'],S.evaluation_schedule(
                    S.read(root/'out/sealed'/dataset/'PANEL.json')['requests'],dataset))
            phantom='/remote/qualification/COMPACT_RETRIEVAL_MANIFEST.json'
            self.assertNotIn(phantom,seal['files'])
            copied=str(S.REMOTE_ROOT/'sealed/qualification_evidence/r1/COMPACT_RETRIEVAL_MANIFEST.json')
            self.assertIn(copied,seal['files'])
            self.assertEqual(plan['binding'],S.read(root/'out/sealed/PILOT_ADMISSION.json')['binding'])

    def test_changed_packet_ids_or_panel_tokens_fail_before_execution(self):
        with tempfile.TemporaryDirectory() as temporary:
            qualification,prior,code=fixture(Path(temporary))
            plan=S.read(qualification/'sealed_package/sealed/PLAN.json')
            prep=S.read(qualification/'retained/qualification_feedback/PREPARATION.json')
            loader=lambda name:S.read(qualification/'sealed_package'/Path(name).relative_to('/remote/qualification'))
            prep['datasets']['gsm8k'][0]['request_id']='different'
            with self.assertRaises(ValueError):S.validate_feedback(prep,plan['datasets'],loader)
            panel=loader(plan['datasets']['gsm8k']['panel_manifest'])['requests']
            canonicals=loader(plan['datasets']['gsm8k']['references'])
            schedule=S.evaluation_schedule(panel,'gsm8k')
            panel[0]['prompt_ids']=[42]
            with self.assertRaises(ValueError):S.validate_evaluation_inputs(panel,{a:a for a in ARMS},
                canonicals,dataset='gsm8k',event=6,schedule=schedule)

    def test_missing_or_modified_compact_evidence_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            qualification,_,_=fixture(Path(temporary))
            retained=qualification/'retained'
            S.verify_retrieval(retained,Path('/remote/qualification'))
            (retained/'EXIT_CODE').write_text('1\n')
            with self.assertRaises(ValueError):S.verify_retrieval(retained,Path('/remote/qualification'))

    def test_copied_evidence_records_actual_destination_and_detects_changed_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'original.json'
            write_json(source,dict(tokens=[3,2]))
            expected=S.identity(source)
            row=dict(recorded_path='/historical/original.json',local_path=str(source),identity=expected)
            prepared={'gsm8k':dict(proof=dict(files=[deepcopy(row)]),
                native=dict(records=[dict(row,request_id='gsm8k:1')]))}
            copied=S.copy_canonical_evidence(prepared,root/'out')
            self.assertEqual(copied,2*expected['bytes'])
            receipt=S.read(root/'out/sealed/reference_evidence/gsm8k/BINDING.json')
            entry=receipt['native']['records'][0]
            self.assertEqual(entry['recorded_path'],'/historical/original.json')
            self.assertEqual(entry['retained_path'],str(S.REMOTE_ROOT/'sealed/reference_evidence/gsm8k/native/0000.json'))
            self.assertEqual(S.identity(root/'out/sealed/reference_evidence/gsm8k/native/0000.json'),expected)
            source.write_text('changed')
            with self.assertRaises(ValueError):S.copy_canonical_evidence(prepared,root/'changed')

    def test_launcher_has_no_qualification_retry_or_extra_model_work(self):
        script=Path(S.__file__).with_name('run_pilot.sh').read_text()
        self.assertIn('set -eu',script)
        self.assertIn('test ! -e "$run_root/START_UTC"',script)
        self.assertEqual(script.count('-m experiments.functional_step_geometry.native_pilot'),1)
        self.assertNotIn('-m experiments.functional_step_geometry.qualify',script)
        self.assertIn('--postflight',script)
        self.assertIn('timeout --signal=TERM --kill-after=120s 48h',script)


if __name__=='__main__':unittest.main()
