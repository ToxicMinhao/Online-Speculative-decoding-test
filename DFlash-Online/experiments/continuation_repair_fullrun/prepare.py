"""Bind the new panel and fixed fresh trajectories to authenticated native inputs."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from experiments.ok_coding.storage import identity
from experiments.kplus_full_gsm8k.storage import compiler_snapshot
from online_feedback.storage import write_json
from .configuration import build_config
from .contract import (ARMS, METHODS, SCHEMA, SERVING, RESOURCE_LIMITS,
    PRODUCTION_WORK, QUALIFICATION_CEILINGS, OPTIONAL_WORK_CEILINGS, FORMULA_FILES, INPUT_IDENTITY,
    digest, validate_inputs)
from .data import original_inputs, select_witness

PR65_PLAN_IDENTITY = dict(bytes=203176,
    sha256='2d4e24408fc64343406525a9a1ee9eb61ca553ffca69f650d278867012b207d4')
PR56_PLAN_IDENTITY = dict(bytes=20373,
    sha256='99aa38b6307ec5ecf2e2c6ba2db8144897352df7e53aabffc496c7b3287e81b0')
# Equal to the retained qualified native export configuration, including AREPO58.
TRAINED_CONFIG_IDENTITY = dict(bytes=1211,
    sha256='5749d96966f9ce7e6a9662c9f482b8aa0d461845fb6b9ce09ac2c08311fab2aa')
ENDPOINT_IDENTITIES = {
    'AREPO':dict(bytes=2097259104,sha256='25540287dcb37baa2386fb7fe57c5b823f2976a84269bf8a018c5779a62b7c30'),
    'F':dict(bytes=2097259104,sha256='c702878094f38ad5843e6ac40b327720b3147ffbf669a5d9ca1974865ca6c080'),
}


def read(path):
    return json.loads(Path(path).read_text())


def verify_panel(panel_root, original):
    panel_root=Path(panel_root)
    seal=read(panel_root/'DATA_SEAL.json')
    if (seal['status']!='sealed_before_model_work' or seal['original_inputs_identity']!=INPUT_IDENTITY
            or seal['original_training_order_digest']!=digest(original['train_requests'])):
        raise ValueError('Panel seal changed the original training stream')
    for name,expected in seal['files'].items():
        if identity(panel_root/name)!=expected:
            raise ValueError('Sealed panel artifact changed: '+name)
    panel=read(panel_root/'EVALUATION_INPUTS.json')
    validate_inputs(original,panel)
    witness=read(panel_root/'WITNESS_SELECTION.json')
    if witness!=select_witness(original):
        raise ValueError('Witness selection differs from the fixed input-only rank')
    exposure=read(panel_root/'EXPOSURE.json')
    if exposure['supplement']['coverage']['gaps'] or exposure['coverage']['through_pr']!=65:
        raise ValueError('Post-PR56 exposure inventory has unresolved coverage gaps')
    return panel,seal


def validate_warmups(original, references, provenance, released_checkpoint):
    from experiments.opencode_native_comparison.evaluation import check_reference
    requests=original['warmup_requests']
    if set(references)!={row['request_id'] for row in requests}:
        raise ValueError('Retained warmup references do not cover the original two requests')
    for request in requests:
        reference=references[request['request_id']]
        check_reference(request,reference,provenance,method='F')
        if reference['source_checkpoint']!=released_checkpoint:
            raise ValueError('Warmup reference was not produced by released frozen DFlash')
    return dict(status='passed',reused=2,new_generations=0,
                reference_kind='native_frozen_DFlash',original_PR56_warmups=True)


def prepare(*, root, source_root, source_manifest, commit, pr56_plan, pr65_plan,
            panel_root, storage_path, arepo_status):
    root,source_root,panel_root=Path(root).resolve(),Path(source_root).resolve(),Path(panel_root).resolve()
    destination=root/'sealed/PLAN.json'
    if destination.exists() or (root/'CLOCK.json').exists():
        raise FileExistsError('Never replace the sealed or started full-run plan')
    source=read(source_manifest)
    if source['commit']!=commit or not (source.get('clean') is True or source.get('dirty') is False):
        raise ValueError('Require the clean published execution source')
    if identity(pr56_plan)!=PR56_PLAN_IDENTITY:
        raise ValueError('Original PR56 plan differs from its retained execution seal')
    previous=read(pr56_plan)
    original=original_inputs(previous['manifest'])
    panel,seal=verify_panel(panel_root,original)
    if identity(pr65_plan)!=PR65_PLAN_IDENTITY:
        raise ValueError('Qualification parent contract is not the authenticated original PR65 seal')
    pilot=read(pr65_plan)
    storage=read(storage_path)
    if storage['status']!='admitted':
        raise ValueError('Whole-campaign resource forecast must be admitted before sealing')
    bindings={}
    def bound(path, expected=None):
        path=str(path);actual=identity(path)
        if expected is not None and actual!=expected:
            raise ValueError('Retained dependency changed: '+path)
        bindings[path]=actual
        return actual
    bound(pr56_plan,PR56_PLAN_IDENTITY);bound(pr65_plan,PR65_PLAN_IDENTITY)
    bound(previous['manifest'],INPUT_IDENTITY)
    for name in ('original_config','runtime_pin','training_pin','model_pin'):
        bound(previous[name],previous['input_identities'][previous[name]])
    # Released weights/tokenizer pins are shared by both fresh initializations.
    model_pin=read(previous['model_pin'])
    authenticated=[]
    for role in ('target','draft'):
        for name,expected in model_pin[role]['files'].items():
            path=Path(previous[role])/name
            bound(path,expected)
            authenticated.append(dict(path=str(path),role='released_'+role,expected_identity=expected))
    arepo=read(arepo_status)
    if (arepo['status']!='completed' or arepo['arm']!='AREPO_CODE58'
            or arepo['events_completed']!=58 or arepo['boundary']['status']!='passed'
            or arepo['boundary']['scientific_check']['global_step']!=7424
            or arepo['boundary']['scientific_check']['presentations']!=14848):
        raise ValueError('Retained AREPO is not the completed 58-event endpoint')
    bound(previous['configs']['AREPO_CODE58'],previous['input_identities'][previous['configs']['AREPO_CODE58']])
    from .configuration import validate_config_text
    validate_config_text(Path(previous['configs']['AREPO_CODE58']).read_text(),
                         Path(previous['original_config']).read_text())
    checkpoints=dict(F=previous['draft'],AREPO=arepo['final_checkpoint'])
    retained_endpoints={}
    bound(arepo_status)
    for name,path in checkpoints.items():
        directory=Path(path)
        bound(directory/'model.safetensors',ENDPOINT_IDENTITIES[name])
        expected_companions=({'config.json':TRAINED_CONFIG_IDENTITY} if name=='AREPO'
            else {key:value for key,value in model_pin['draft']['files'].items() if key!='model.safetensors'})
        companions={str(directory/key):bound(directory/key,value) for key,value in expected_companions.items()}
        if str(directory/'config.json') not in companions:
            raise ValueError('Serving endpoint lacks its companion configuration')
        retained_endpoints[name]=dict(checkpoint=str(directory),
            checkpoint_identity=ENDPOINT_IDENTITIES[name],companion_files=companions)
    # PR65 state/data are disposable qualification fixtures only, never production initializers.
    qualification_files=[pilot['parent_plan'],pilot['parent_recovery'],
        str(Path(pilot['parent_export'])/'model.safetensors'),pilot['retained_event'],
        pilot['feature_manifest'],pilot['warmup_canonicals'],*pilot['configs'].values()]
    for path in qualification_files:bound(path,pilot['sealed_files'][path])
    qualification_parent=read(pilot['parent_plan'])
    for path in (qualification_parent['original_config'],qualification_parent['configs']['PAIR']):
        bound(path,pilot['sealed_files'][path])
    for row in read(pilot['feature_manifest']):bound(row['path'],row['identity'])
    warmup_validation=validate_warmups(original,read(pilot['warmup_canonicals']),
        previous['canonical_provenance'],previous['draft'])
    for path in [panel_root/'DATA_SEAL.json',*(panel_root/name for name in seal['files'])]:bound(path)
    configs={}
    for arm in ARMS:
        path=root/'configs'/f'{arm}.yaml';path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(build_config(Path(previous['original_config']).read_text(),arm=arm,
            output_dir=root/'campaign_001/arms'/arm/'trainer'))
        configs[arm]=str(path);bound(path)
    inherited=('original_config','runtime_pin','training_pin','model_pin','target','draft',
        'training_python','worker_python','zstd','devices','canonical_provenance')
    plan={name:deepcopy(previous[name]) for name in inherited}
    plan.update(schema=SCHEMA,root=str(root),source_root=str(source_root),
        source_manifest=str(source_manifest),source_identity=identity(source_manifest),
        execution_commit=commit,execution_source=dict(commit=commit,dirty=False),
        arms=list(ARMS),methods=list(METHODS),events=58,seed=17,configs=configs,
        manifest=previous['manifest'],original_inputs=previous['manifest'],original_inputs_identity=INPUT_IDENTITY,
        original_training_order_digest=seal['original_training_order_digest'],
        evaluation_inputs=str(panel_root/'EVALUATION_INPUTS.json'),data_seal=str(panel_root/'DATA_SEAL.json'),
        exposure_inventory=str(panel_root/'EXPOSURE.json'),witness_selection=str(panel_root/'WITNESS_SELECTION.json'),
        serving=SERVING,limits=RESOURCE_LIMITS,production_work=PRODUCTION_WORK,
        optional_work_ceilings=OPTIONAL_WORK_CEILINGS,
        qualification_work=QUALIFICATION_CEILINGS,qualification_plan=str(pr65_plan),
        qualification_parent_plan=pilot['parent_plan'],qualification_parent_recovery=pilot['parent_recovery'],
        qualification_parent_export=pilot['parent_export'],qualification_parent_export_identity=pilot['parent_export_identity'],
        qualification_configs=pilot['configs'],qualification_retained_event=pilot['retained_event'],
        qualification_feature_manifest=pilot['feature_manifest'],
        qualification_pair_ids=previous['qualification_pair_ids'],
        warmup_canonicals=pilot['warmup_canonicals'],warmup_validation=warmup_validation,
        checkpoints=checkpoints,checkpoint_identities=ENDPOINT_IDENTITIES,retained_endpoints=retained_endpoints,
        retained_arepo_status=str(arepo_status),retained_arepo_status_identity=bindings[str(arepo_status)],
        formula_code_identities={name:identity(source_root/name) for name in FORMULA_FILES},
        authenticated_files=authenticated,input_identities=bindings,sealed_files=bindings,
        allocation_roots=[str(root),str(source_root.parent)],compiler_baseline=compiler_snapshot(),
        storage=storage,forecast=deepcopy(storage['forecast']),production_admitted=True,bootstrap_seed=20261126,
        session='continuation_repair_fullrun_20261126',automatic_retry='infrastructure_only_within_fixed_caps',
        checkpoint_selection=False,initialization='released_fresh_independently_per_arm')
    plan['binding']=digest(plan)
    destination.parent.mkdir(parents=True,exist_ok=True)
    write_json(destination,plan)
    write_json(root/'KEEP_MANIFEST.json',dict(historical_retirement_authorized=False,
        historical_dependencies=bindings,keep_new=['sealed inputs/source','raw feedback tokens/termination',
            'event schedules/anchors','scalar training/activity/costs','failures','witness features/readouts','all evaluation records'],
        final_recoveries=[str(root/'campaign_001/arms'/arm/'trainer/events'/f'event_{event:03d}/training_state.pt')
                          for arm in ARMS for event in (57,58)],
        final_exports=[str(root/'campaign_001/arms'/arm/'trainer/events/event_058/export') for arm in ARMS],
        rolling_retention='retire only after two newer complete boundaries validate; always keep final57/58'))
    return plan


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','source-root','source-manifest','commit','pr56-plan','pr65-plan','panel-root','storage-path','arepo-status'):
        parser.add_argument('--'+name,required=True)
    result=prepare(**vars(parser.parse_args()))
    print(json.dumps(dict(status='sealed',binding=result['binding'],root=result['root'])))
