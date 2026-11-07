"""Seal reviewed inputs, authenticated canonical traces and six fresh recipes."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess

import yaml

from online_feedback.storage import write_json
from experiments.ok_coding.storage import identity
from experiments.o_half_matched_lr.archive import validate_record
from .contract import (ARMS, DATASETS, SCHEMA, EVENTS, NEW_BUDGET, ANALYSIS,
                       SERVING, stable_digest, loss_contract, validate_plan)
from .diagnostics import diagnostic_manifest
from .evaluation import evaluation_schedule, schedule_balance, validate_evaluation_inputs
from .storage import capacity_plan
from .training import validate_config_text


def authenticate_references(requests, references, mirror, origin, binding):
    """Check complete retained traces before attaching current input identities.

    Native sidecars are checked again before fixed-context reconstruction. This
    stage authenticates their embedded capture/verification evidence in the
    retained per-request records, without generating a replacement reference.
    """
    result, files = {}, {}
    for request in requests:
        key = request['request_id']
        original = references[key]
        source = Path(original['source_record_path'])
        local = Path(mirror) / source.relative_to(origin)
        raw = json.loads(local.read_text())
        check_request = dict(request, panel_ordinal=raw['request_index'])
        errors = validate_record(raw, check_request, raw['checkpoint_id'], original,
                                 label=raw['record_id'], session=raw['session_id'])
        if errors:
            raise ValueError(f'Invalid retained canonical {key}: {errors}')
        source_identity = identity(local)
        files[str(source)] = source_identity
        result[key] = dict(deepcopy(original), prompt_ids=request['prompt_ids'],
            verified=True, binding_digest=binding,
            source_record=dict(path=str(source), **source_identity))
    return result, files


def seal(args):
    code, prepared, prior, output = map(Path, (args.code, args.prepared, args.prior_sealed, args.output))
    remote, remote_code = Path(args.remote_root), Path(args.remote_code)
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=code,text=True).strip()
    if subprocess.check_output(['git','status','--porcelain'],cwd=code,text=True):
        raise ValueError('Publish a clean source commit before sealing')
    old = json.loads((prior/'PLAN.json').read_text())
    execution = dict(commit=commit, dirty=False, source_path=str(remote_code),
        repository='https://github.com/ToxicMinhao/Online-Speculative-decoding-test')
    source_inputs = {str(p):identity(p) for p in prepared.rglob('*.json')}
    binding = stable_digest(dict(prepared=source_inputs, source=execution,
        loss={arm:loss_contract(arm) for arm in ARMS}, analysis=ANALYSIS))
    output.mkdir(parents=True,exist_ok=False)
    sealed = output/'sealed'; sealed.mkdir()
    for name in ('ORIGINAL_CONFIG.yaml','MODEL_PIN.json','RUNTIME_PIN.json','TRAINING_PIN.json'):
        shutil.copy2(prior/name,sealed/name)
    initial_audit=json.loads((prior/'CONTROL_AUDIT.json').read_text())
    if (initial_audit['status']!='passed' or initial_audit['model_work']!=0
            or initial_audit['original_initial_state']!=old['original_initial_state']):
        raise ValueError('Initial K state lacks its retained authentication')
    qualification_pins=json.loads(Path(args.qualification_pins).read_text())
    preparation_pin=qualification_pins[args.qualification_prepared]
    feature_pins={name:value for name,value in qualification_pins.items()
                  if '/preparation/features/' in name}
    if len(feature_pins)!=6:
        raise ValueError('Require all six previously authenticated feature files')
    dependency_files={old['original_initial_state']:initial_audit['original_initial_identity'],
        args.qualification_config:identity(args.qualification_config_copy),
        args.qualification_prepared:preparation_pin,**feature_pins}
    write_json(sealed/'DEPENDENCY_PINS.json',dependency_files)
    shutil.copy2(args.handoff,sealed/'HANDOFF.md')
    for name in ('PREPARATION.json','MATHQA_EXCLUSIONS.json'):
        shutil.copy2(prepared/name,sealed/name)
    original_config = (sealed/'ORIGINAL_CONFIG.yaml').read_text()
    pins = {name:identity(sealed/name) for name in ('MODEL_PIN.json','RUNTIME_PIN.json')}
    datasets, reference_files, balances = {}, {}, {}
    for dataset in DATASETS:
        folder=sealed/dataset; folder.mkdir()
        for name in ('TRAINING.json','PANEL.json'):
            shutil.copy2(prepared/dataset/name,folder/name)
        requests=json.loads((folder/'PANEL.json').read_text())['requests']
        reference_binding=stable_digest(dict(pins=pins,inputs=[r['prompt_ids'] for r in requests]))
        references, evidence=authenticate_references(requests,
            json.loads((prepared/dataset/'CANONICALS.json').read_text()),
            getattr(args,dataset+'_mirror'),getattr(args,dataset+'_origin'),reference_binding)
        reference_files.update(evidence)
        write_json(folder/'CANONICALS.json',references)
        write_json(folder/'DIAGNOSTIC.json',diagnostic_manifest(requests,references,dataset))
        configs={}
        for arm in ARMS:
            raw=yaml.safe_load(original_config)
            raw['run_id']=f'roleloss-{dataset}-{arm.lower()}'
            raw['output_dir']=str(remote/'campaign_001/trajectories'/dataset/arm/'trainer')
            text=yaml.safe_dump(raw,sort_keys=False)
            validate_config_text(text,original_config)
            name=f'config_{arm}.yaml'; (folder/name).write_text(text)
            configs[arm]=str(remote/'sealed'/dataset/name)
        schedules={str(event):evaluation_schedule(requests,dataset,event) for event in (2,4)}
        for event in (2,4):
            checkpoints={arm:str(remote/'campaign_001/trajectories'/dataset/arm/f'event_{event}') for arm in ARMS}
            validate_evaluation_inputs(requests,checkpoints,references,dataset=dataset,event=event,
                                       schedule=schedules[str(event)])
        balances[dataset]={event:schedule_balance(slots) for event,slots in schedules.items()}
        datasets[dataset]=dict(training_manifest=str(remote/'sealed'/dataset/'TRAINING.json'),
            panel_manifest=str(remote/'sealed'/dataset/'PANEL.json'),
            references=str(remote/'sealed'/dataset/'CANONICALS.json'),
            diagnostic_manifest=str(remote/'sealed'/dataset/'DIAGNOSTIC.json'),
            configs=configs,evaluation_schedules=schedules)
    weight_bytes=json.loads((sealed/'MODEL_PIN.json').read_text())['draft']['files']['model.safetensors']['bytes']
    plan=dict(schema=SCHEMA,root=str(remote),arms=list(ARMS),seed=17,events=EVENTS,
        budget=NEW_BUDGET,analysis=ANALYSIS,serving=SERVING,automatic_retry=False,
        execution_source=execution,binding=binding,datasets=datasets,
        loss_contracts={arm:loss_contract(arm) for arm in ARMS},
        original_config=str(remote/'sealed/ORIGINAL_CONFIG.yaml'),
        original_initial_state=old['original_initial_state'],
        target=old['target'],draft=old['draft'],training_python=old['training_python'],
        worker_python=old['worker_python'],serving_gpu=0,training_gpu=1,
        storage=capacity_plan(weight_bytes),
        qualification=dict(config=args.qualification_config,initial_state=old['original_initial_state'],
            prepared=args.qualification_prepared,maximum_optimizer_steps=6,timeout_s=1800))
    local=deepcopy(plan)
    for item in local['datasets'].values():
        for key in ('training_manifest','panel_manifest','references','diagnostic_manifest'):
            item[key]=str(output/Path(item[key]).relative_to(remote))
    write_json(sealed/'PLAN_CHECK.json',validate_plan(local))
    write_json(sealed/'ORDER_BALANCE.json',balances)
    write_json(sealed/'REFERENCE_AUDIT.json',dict(status='passed',records=512,extra_generations=0,
        evidence_scope='complete retained native per-request records; sidecars rechecked before reconstruction',
        files=reference_files))
    write_json(sealed/'PLAN.json',plan)
    files=dict(reference_files,**dependency_files)
    for encoded in subprocess.check_output(['git','ls-files','-z','.'],cwd=code).split(b'\0'):
        if encoded:
            name=encoded.decode()
            if (code/name).is_file(): files[str(remote_code/name)]=identity(code/name)
    for path in sorted(output.rglob('*')):
        if path.is_file(): files[str(remote/path.relative_to(output))]=identity(path)
    folder=code/'experiments/role_loss_pilot'
    write_json(sealed/'SEAL.json',dict(execution_source=execution,files=files,
        exact_file_sets={str(remote_code/'experiments/role_loss_pilot'):
            sorted(p.name for p in folder.iterdir() if p.is_file())}))
    return dict(status='sealed_before_model_work',execution_source=execution,binding=binding,
                files=len(files),budget=NEW_BUDGET,storage=plan['storage'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('code','prepared','prior-sealed','output','remote-root','remote-code','handoff',
                 'qualification-config','qualification-prepared','gsm8k-mirror','gsm8k-origin',
                 'mathqa-mirror','mathqa-origin','qualification-pins','qualification-config-copy'):
        parser.add_argument('--'+name,required=True)
    print(json.dumps(seal(parser.parse_args()),indent=2))


if __name__=='__main__': main()
