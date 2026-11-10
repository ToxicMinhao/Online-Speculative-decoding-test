"""Bind R1 to the passed proofs, original states, and four new live cases."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess

import yaml
from experiments.ok_coding.storage import identity
from experiments.role_loss_pilot.contract import stable_digest
from experiments.role_loss_pilot.training import validate_config_text
from online_feedback.storage import write_json
from .contract import geometry_contract
from .qualify_realization import REVISION, LIMITS
from .seal import published_source

STUDY=Path('/home/minhao2/EAGLE_SPEC/research/dflash_functional_step_geometry_20261108')
REMOTE_ROOT=Path('/data/sources/minhao/runs/dflash_functional_step_geometry_20261108/bf16_realization_r1_001')
REMOTE_CODE=Path('/data/sources/minhao/fsg_source_20261109_realization_r1/DFlash-Online')


def seal(code,commit,output):
    code,output=Path(code),Path(output)
    execution=published_source(code,commit);execution['source_path']=str(REMOTE_CODE)
    prior=STUDY/'qualification_resume_001';package=prior/'sealed_package'
    retained=prior/'retained';old=json.loads((package/'sealed/PLAN.json').read_text())
    old_root=Path(old['root']);old_seal=json.loads((package/'sealed/SEAL.json').read_text())
    previous=json.loads((retained/'qualification/QUALIFICATION.json').read_text())
    if previous['status']!='passed':raise ValueError('Require passed unchanged qualification evidence')
    for p in (package/'sealed').rglob('*'):
        if p.is_file() and p.name!='SEAL.json':
            if identity(p)!=old_seal['files'][str(old_root/p.relative_to(package))]:
                raise ValueError('Original sealed configuration changed: '+str(p))
    output.mkdir(parents=True,exist_ok=False)
    shutil.copytree(package/'sealed',output/'sealed',ignore=shutil.ignore_patterns('PLAN.json','SEAL.json'))
    plan=deepcopy(old);geometry=geometry_contract(REVISION)
    handoff=STUDY/'bf16_realization_r1_001/HANDOFF.txt'
    binding=stable_digest(dict(previous_binding=old['binding'],geometry=geometry,handoff=identity(handoff)))
    plan.update(root=str(REMOTE_ROOT),scope='bf16_realization_r1_qualification',
        execution_source=execution,binding=binding,realization_revision=REVISION,
        geometry=geometry,original_config=str(REMOTE_ROOT/'sealed/ORIGINAL_CONFIG.yaml'),
        supplementary_limits=LIMITS,main_pilot_authorized='conditional_after_revised_checks_activity_native_bindings_and_separate_storage',
        main_pilot_condition='All four cases complete; FSG nontrivial acceptance and SCALAR matching/check; original pilot only',
        historical_qualification_work=previous['work'],supplementary_references={})
    plan['qualification_budget']=dict(optimizer_steps=4,ordinary_presentations=8,
        additional_response_forwards=32,additional_backwards=16,serving_generations=4)
    plan['numerical_recipe']['geometry']=geometry
    plan['numerical_recipe']['qualification_work_limits']=LIMITS
    plan['numerical_recipe']['native_oracle_budget']=dict(additional_response_forwards=0,
        additional_backwards=0,optimizer_steps=0,serving_generations=0,
        status='reuse_passed_unchanged_proofs_without_new_model_work')
    plan['numerical_recipe']['realization_revision']=REVISION
    plan['supplementary_reuse']=dict(prior_root=str(old_root),
        formula='same-layout independent oracle and high-precision CPU proofs retained',
        startup='separately discarded then exactly restored before warmed reference capture',
        max_resource='reuse qualified two-by-512-anchor four-VJP fixture',
        cross_layout='original failure remains failed sensitivity diagnostic',
        original_origins=True,retired_disposable_optimizer_required=False)
    plan.pop('repair',None);plan.pop('startup_repair',None)
    for dataset,item in plan['datasets'].items():
        for key in ('training_manifest','panel_manifest','references'):
            item[key]=str(REMOTE_ROOT/Path(item[key]).relative_to(old_root))
        for arm,path in item['configs'].items():
            relative=Path(path).relative_to(old_root);p=output/relative
            cfg=yaml.safe_load(p.read_text());cfg['output_dir']=str(REMOTE_ROOT/'qualification'/dataset/'trainer')
            text=yaml.safe_dump(cfg,sort_keys=False)
            validate_config_text(text,(output/'sealed/ORIGINAL_CONFIG.yaml').read_text())
            p.write_text(text);item['configs'][arm]=str(REMOTE_ROOT/relative)
        ordinary=retained/'qualification'/dataset/'KPLUS'
        rows='BOUNDARY_REPLAY_PRESENTATIONS.json' if dataset=='gsm8k' else 'PRESENTATIONS.json'
        export=json.loads((ordinary/'EXPORT_EXACT.json').read_text())
        if export['status']!='passed' or not export['exact_tensor_values_and_membership']:
            raise ValueError('Ordinary export has not passed exact comparison')
        plan['supplementary_references'][dataset]=dict(
            export=str(old_root/'qualification'/dataset/'KPLUS/export'),
            export_files=export['files'],
            presentations=str(old_root/'qualification'/dataset/'KPLUS'/rows),
            presentations_identity=identity(ordinary/rows))
    (output/'qualification_feedback').mkdir()
    shutil.copy2(retained/'qualification_feedback/PREPARATION.json',output/'qualification_feedback/PREPARATION.json')
    shutil.copy2(handoff,output/'sealed/REALIZATION_R1_HANDOFF.txt')
    shutil.copy2(STUDY/'bf16_realization_r1_001/AUTHORIZATION.json',output/'sealed/REALIZATION_AUTHORIZATION.json')
    write_json(output/'sealed/NUMERICAL_RECIPE.json',plan['numerical_recipe'])
    write_json(output/'sealed/PLAN.json',plan)
    files=dict(old_seal['files'])
    for p in sorted(retained.rglob('*')):
        if p.is_file():files[str(old_root/p.relative_to(retained))]=identity(p)
    for item in plan['supplementary_references'].values():
        for relative,digest in item['export_files'].items():files[str(Path(item['export'])/relative)]=digest
    for encoded in subprocess.check_output(['git','ls-files','-z','.'],cwd=code).split(b'\0'):
        if encoded:
            p=code/encoded.decode()
            if p.is_file():files[str(REMOTE_CODE/p.relative_to(code))]=identity(p)
    for p in sorted(output.rglob('*')):
        if p.is_file():files[str(REMOTE_ROOT/p.relative_to(output))]=identity(p)
    write_json(output/'sealed/SEAL.json',dict(execution_source=execution,files=files,
        scope=plan['scope'],new_limits=LIMITS,historical_work_separate=True))
    return dict(status='sealed',root=str(REMOTE_ROOT),source=execution,binding=binding,
                files=len(files),limits=LIMITS,pilot_storage_admitted=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--code',default=str(Path(__file__).resolve().parents[2]))
    p.add_argument('--commit',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();print(json.dumps(seal(args.code,args.commit,args.output),indent=2))
