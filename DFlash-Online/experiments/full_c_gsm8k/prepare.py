"""Package the authenticated original stream and two unchanged evaluation panels."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil

from online_feedback.storage import write_json
from experiments.ok_coding.storage import identity
from experiments.full_k_gsm8k.contract import SERVING, stable_digest
from .contract import ARM, METHODS, PRODUCTION, QUALIFICATION, SERVING_QUALIFICATION, SCHEMA, plan_binding
from .storage import capacity_plan


def build(data, original_plan, original_manifest, destination, *, remote_root, source_root, execution_commit):
    import yaml
    from .evaluation import evaluation_schedule
    from experiments.kplus_full_gsm8k.training import validate_config_text
    data, destination = Path(data), Path(destination)
    destination.mkdir(parents=True,exist_ok=False)
    sealed=destination/'sealed';sealed.mkdir()
    old=json.loads(Path(original_plan).read_text())
    inputs=json.loads((data/'INPUT_IDENTITIES.json').read_text())
    if inputs['errors'] or inputs['model_work'] != 0:
        raise ValueError('Local original-input review has not passed')
    dependencies=json.loads((data/'PROTECTED_DEPENDENCIES.json').read_text())
    plan=dict(schema=SCHEMA,root=remote_root,source_root=source_root,
        source_manifest=remote_root+'/SOURCE.json',execution_commit=execution_commit,
        execution_source=dict(commit=execution_commit,dirty=False),
        arms=[ARM],methods=list(METHODS),seed=17,events=58,serving=SERVING,
        learning_rates={ARM:3e-4},intermediate_events=[],automatic_retry=False,
        production_work=PRODUCTION,qualification_work=QUALIFICATION,
        serving_qualification_schedule=SERVING_QUALIFICATION,
        forecast=dict(factor=1.3,fixed_margin_s=1800),
        devices=dict(A='GPU-9986735a-3e68-d1bd-5998-9634ef24341e',B='GPU-c6943f33-bcd0-a724-0b35-f77af6b0a3f1'),
        protected_dependencies=dependencies, input_identities={},datasets={},
        estimated_seconds=dict(train=35500.,final_gsm8k=8500.,final_mathqa=14000.,audit=1800.))
    plan.update({k:old[k] for k in ('target','draft','training_python','worker_python')})
    def save(name,value=None,source=None):
        path=sealed/name
        path.parent.mkdir(parents=True,exist_ok=True)
        if source is not None:shutil.copyfile(source,path)
        else:write_json(path,value)
        remote=remote_root+'/sealed/'+name
        plan['input_identities'][remote]=identity(path)
        return remote
    manifest=json.loads(Path(original_manifest).read_text())
    stream=json.loads((data/'TRAINING_STREAM.json').read_text())
    if manifest['train_requests'] != stream['requests']:
        raise ValueError('Repackaging changed the original 7473-request stream')
    plan['manifest']=save('ORIGINAL_MANIFEST.json',source=original_manifest)
    plan['manifest_binding']=stable_digest(manifest)
    for key,name in (('original_config','ORIGINAL_CONFIG.yaml'),('runtime_pin','RUNTIME_PIN.json'),
                     ('training_pin','TRAINING_PIN.json'),('model_pin','MODEL_PIN.json')):
        plan[key]=save(name,source=data/'original_pins'/name)
    original=(data/'original_pins/ORIGINAL_CONFIG.yaml').read_text()
    cfg=yaml.safe_load(original)
    cfg.update(run_id='full-c-gsm8k',output_dir=remote_root+'/campaign_001/arms/'+ARM)
    text=yaml.safe_dump(cfg,sort_keys=False)
    validate_config_text(text,original)
    (sealed/'config_C.yaml').write_text(text)
    plan['config_path']=remote_root+'/sealed/config_C.yaml'
    plan['configs']={ARM:plan['config_path']}
    plan['input_identities'][plan['config_path']]=identity(sealed/'config_C.yaml')
    required=dependencies['rows']
    plan['initial_witness']=next(r['path'] for r in required if r['role']=='initial_state_witness')
    plan['original_initial_state']=plan['initial_witness']
    plan['references']={label:str(Path(next(r['path'] for r in required
        if r['role']==label+'_selected_endpoint' and r['path'].endswith('model.safetensors'))).parent)
        for label in ('O58','KPLUS58')}
    qualification=json.loads((data/'QUALIFICATION_FEATURES.json').read_text())
    plan['qualification_feature_paths']=[r['path'] for r in qualification['records']]
    plan['qualification_feature_bindings']=qualification
    save('QUALIFICATION_FEATURES.json',qualification)
    warmups=json.loads((data/'WARMUPS.json').read_text())
    warmup_refs=json.loads((data/'WARMUP_CANONICALS.json').read_text())
    for dataset in ('gsm8k','mathqa'):
        panel=json.loads((data/(dataset.upper()+'_PANEL.json')).read_text())
        refs=json.loads((data/(dataset.upper()+'_CANONICALS.json')).read_text())
        sub={}
        for key,name,value in (
            ('requests','PANEL.json',panel),('canonicals','CANONICALS.json',refs),
            ('warmups','WARMUPS.json',dict(requests=warmups[dataset])),
            ('warmup_canonicals','WARMUP_CANONICALS.json',warmup_refs[dataset]),
            ('schedule','SCHEDULE.json',evaluation_schedule(panel['requests'])),
            ('historical_counts','HISTORICAL_COUNTS.json',json.loads((data/(dataset.upper()+'_BASELINE_COUNTS.json')).read_text()))):
            sub[key]=save(dataset+'/'+name,value)
        sub['reference_checkpoints']=dict(scored=next(iter(refs.values()))['reference_checkpoint'],
            warmup=next(iter(warmup_refs[dataset].values()))['reference_checkpoint'])
        sub['warmup_ids']=[r['request_id'] for r in warmups[dataset]]
        sub['bootstrap_seed']=20261118 if dataset=='gsm8k' else 20261119
        plan['datasets'][dataset]=sub
    for name in ('INPUT_IDENTITIES.json','OSD_RECIPE_AUDIT.json','WARMUP_SELECTION.json'):
        save(name,source=data/name)
    weight=next(r['expected_identity']['bytes'] for r in required
        if r['role']=='KPLUS58_selected_endpoint' and r['path'].endswith('model.safetensors'))
    plan['storage']=capacity_plan(weight,external_roots=[str(Path(source_root).parent)])
    plan['storage'].update({key:plan[key] for key in ('source_root','source_manifest','execution_commit')})
    plan['binding']=plan_binding(plan)
    write_json(sealed/'PLAN.json',plan)
    return plan


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ('data','original-plan','original-manifest','destination','remote-root','source-root','execution-commit'):
        parser.add_argument('--'+arg,required=True)
    a=parser.parse_args()
    build(a.data,a.original_plan,a.original_manifest,a.destination,
        remote_root=a.remote_root,source_root=a.source_root,execution_commit=a.execution_commit)
