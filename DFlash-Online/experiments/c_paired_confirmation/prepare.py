"""Package immutable input seals and the unchanged scientific dependencies."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

from experiments.bounded_prefix_allocation.prepare import identity, _digest, _write
from .data import validate_manifests, evaluation_schedule
from .resources import storage_plan, production_groups
from .runtime import LIMITS

TASK_NAME = 'dflash_c_paired_confirmation_20261117'
MUTABLE_PLAN_FIELDS = {'binding', 'compiler_baseline_allocated_bytes', 'estimated_seconds'}


def read(path):
    return json.loads(Path(path).read_text())


def plan_binding(plan):
    return _digest({key:value for key,value in plan.items() if key not in MUTABLE_PLAN_FIELDS})


def load_inputs(plan, *, package_root=None):
    """Use the same consumer paths before staging and on A800."""
    result = {}
    for dataset, sub in plan['datasets'].items():
        def path(key):
            remote = Path(sub[key])
            return remote if package_root is None else Path(package_root)/remote.relative_to(plan['root'])
        values = {key:read(path(key)) for key in ('data_panel','warmups','adaptation','first_inputs','next_inputs','evaluation_schedule','data_seal')}
        validate_manifests(dataset, values['data_panel'], values['warmups'], values['adaptation'])
        for role, offset in (('first_inputs',0),('next_inputs',128)):
            if values[role]['requests'] != values['adaptation']['train_requests'][offset:offset+128]:
                raise ValueError('Original continuation input ordering differs: '+dataset+'/'+role)
        if values['evaluation_schedule']['activations'] != evaluation_schedule(values['data_panel']['requests']):
            raise ValueError('Sealed N/C activation order differs')
        seal = values['data_seal']
        if (seal['status'] != 'sealed' or seal['validation']['status'] != 'passed'
                or seal['no_gpu_or_remote_work'] is not True):
            raise ValueError('Both input panels must be sealed before any model work')
        for name, expected in seal['files'].items():
            actual_path = path('data_seal').parent/name
            if identity(actual_path) != expected:
                raise ValueError('Sealed input artifact changed: '+str(actual_path))
        result[dataset] = values
    if set(result) != {'gsm8k','mathqa'}:
        raise ValueError('Incomplete two-dataset campaign')
    return result


def reference_readiness(plan, *, package_root=None):
    """Staging verifies sealed inputs; new canonical references follow qualification."""
    inputs=load_inputs(plan,package_root=package_root)
    return dict(status='passed',model_work=0,canonical_generation_pending=True,
        datasets={dataset:dict(scored=len(value['data_panel']['requests']),
            warmups=len(value['warmups']['requests']),
            adaptation=len(value['adaptation']['train_requests']))
            for dataset,value in inputs.items()})


def build(spec, *, data_directory, destination, remote_root, source_root, execution_commit):
    """spec contains authenticated ancestry paths, never replacement data."""
    import yaml
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    sealed=destination/'sealed';sealed.mkdir()
    plan=deepcopy(spec)
    plan.update(schema='c_paired_confirmation_v1',root=remote_root,source_root=source_root,
        execution_commit=execution_commit,source_manifest=remote_root+'/SOURCE.json',
        limits=dict(LIMITS),methods=['N','C'],automatic_retry=False,
        reference_reuse=False,native_control_reuse=False,continuation_rng_seed=29,
        allocation_roots=[remote_root,source_root],compiler_baseline_allocated_bytes=0,
        compiler_growth_reserved_bytes=4*2**30,production_groups=production_groups())
    plan.setdefault('input_identities',{})
    geometry={}
    for dataset, sub in plan['datasets'].items():
        dest=sealed/dataset;dest.mkdir()
        data=Path(data_directory)/dataset
        def copy(src,name):
            shutil.copyfile(src,dest/name)
            remote=remote_root+'/sealed/'+dataset+'/'+name
            plan['input_identities'][remote]=identity(dest/name)
            return remote
        files=dict(data_panel='PANEL.json',warmups='WARMUPS.json',adaptation='TRAINING.json',
            first_inputs=f"EVENT{sub['first_event']}_INPUTS.json",next_inputs=f"EVENT{sub['second_event']}_INPUTS.json",
            evaluation_schedule='EVALUATION_SCHEDULE.json',data_seal='DATA_SEAL.json')
        data_seal=read(data/'DATA_SEAL.json')
        for name, expected in data_seal['files'].items():
            if identity(data/name)!=expected:raise ValueError('Changed input seal '+name)
            copy(data/name,name)
        for key,name in files.items():
            if not (dest/name).exists():copy(data/name,name)
            sub[key]=remote_root+'/sealed/'+dataset+'/'+name
        sub['expected_ancestry']=copy(sub.pop('local_expected_ancestry'),'ORIGIN_ANCESTRY.json')
        sub.pop('local_historical_plan', None)
        original_text=Path(sub.pop('local_origin_config')).read_text()
        (dest/'ORIGINAL_CONFIG.yaml').write_text(original_text)
        sub['origin_config']=remote_root+'/sealed/'+dataset+'/ORIGINAL_CONFIG.yaml'
        plan['input_identities'][sub['origin_config']]=identity(dest/'ORIGINAL_CONFIG.yaml')
        sub['configs']={}
        for arm in ('N','C'):
            config=yaml.safe_load(original_text)
            config.update(run_id=dataset+'-'+arm+'29-paired-confirmation',output_dir=remote_root+'/training/'+dataset+'/'+arm)
            config['training']['learning_rate']=.000300
            text=yaml.safe_dump(config,sort_keys=False)
            from experiments.full_k_gsm8k.training import validate_config_text
            validate_config_text(text,'K',original_text)
            name='config_'+arm+'.yaml';(dest/name).write_text(text)
            sub['configs'][arm]=remote_root+'/sealed/'+dataset+'/'+name
            plan['input_identities'][sub['configs'][arm]]=identity(dest/name)
        sub['references']=remote_root+'/references/'+dataset+'/CANONICALS.json'
        sub['warmup_references']=remote_root+'/references/'+dataset+'/WARMUP_CANONICALS.json'
        geometry[dataset]=dict(sub['geometry'],next_prompt_lengths=[len(r['prompt_ids']) for r in read(dest/files['next_inputs'])['requests']],
            recovery_bytes=sub['origin_state_identity']['bytes'],export_bytes=sub['origin_weights_identity']['bytes'])
    retain=storage_plan(geometry,retain_final_exports=True)
    plan['storage']=retain if retain['status']=='passed' else storage_plan(geometry,retain_final_exports=False)
    plan['storage_policy_comparison']=dict(all_final_exports=retain,
        selected='retain_final_exports' if plan['storage']['retain_final_exports'] else 'qualified_cpu_reconstruction')
    if plan['storage']['status']!='passed':
        _write(destination/'READINESS.json',dict(status='blocked',storage=plan['storage'],model_work=0))
        raise RuntimeError('Priced complete trajectory exceeds the storage cap')
    plan['binding']=plan_binding(plan)
    _write(sealed/'PLAN.json',plan)
    _write(sealed/'SEAL.json',dict(binding=plan['binding'],execution_commit=execution_commit,input_identities=plan['input_identities']))
    load_inputs(plan,package_root=destination)
    return plan
