"""Metadata-only preparation of a source-bound ABT campaign; no model loads."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess

import yaml

from experiments.ok_coding.storage import identity
from experiments.role_loss_pilot.contract import stable_digest
from online_feedback.storage import write_json
from .contract import ARMS, DATASETS, EVENTS, RHO, ANALYSIS, SCHEMA, budgets
from .data import read, validate_inputs
from .evaluation import evaluation_schedule, validate_evaluation_inputs
from .storage import capacity_plan

PRIOR = Path('/home/minhao2/EAGLE_SPEC/research/dflash_functional_step_geometry_20261108/pilot_r1_001/sealed_package')


def validate_config_text(text, original_text):
    """Metadata-only gate; native trainer repeats the full inherited recipe gate."""
    source, current = yaml.safe_load(original_text), yaml.safe_load(text)
    checked = deepcopy(current)
    for key in ('run_id', 'output_dir'):
        if key in source:
            checked[key] = source[key]
        else:
            checked.pop(key, None)
    if checked != source:
        raise ValueError('ABT recipe changed beyond run/output metadata')


def _authenticate_prior(prior):
    old = read(prior/'sealed/PLAN.json')
    seal = read(prior/'sealed/SEAL.json')
    rows = {}
    def verify(path):
        relative = path.relative_to(prior)
        recorded = str(Path(old['root'])/relative)
        expected = seal['files'].get(recorded)
        if expected is None or identity(path) != expected:
            raise ValueError('Retained sealed input changed: '+str(path))
        rows[str(relative)] = expected
    for name in ('ORIGINAL_CONFIG.yaml','MODEL_PIN.json','RUNTIME_PIN.json','TRAINING_PIN.json'):
        verify(prior/'sealed'/name)
    for dataset in DATASETS:
        for name in ('TRAINING.json','PANEL.json','CANONICALS.json'):
            verify(prior/'sealed'/dataset/name)
        validate_inputs(*[read(prior/'sealed'/dataset/name)
                          for name in ('TRAINING.json','PANEL.json','CANONICALS.json')], dataset=dataset)
        for path in sorted((prior/'sealed/reference_evidence'/dataset).rglob('*')):
            if path.is_file():
                verify(path)
        evidence = read(prior/'sealed/reference_evidence'/dataset/'BINDING.json')
        if (evidence['proof']['status'] != 'passed'
                or evidence['proof']['current_model_runtime_exact'] is not True
                or evidence['native']['status'] != 'passed'
                or evidence['native']['count'] != 256
                or evidence['native']['new_generations'] != 0):
            raise ValueError('Canonical native evidence is incomplete')
        for row in evidence['proof']['files']+evidence['native']['records']:
            path = prior/Path(row['retained_path']).relative_to(old['root'])
            if identity(path) != row['identity']:
                raise ValueError('Canonical proof bytes changed')
    return old, rows


def execution_source(code, commit, remote_code):
    head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=code,text=True).strip()
    dirty = subprocess.check_output(['git','status','--porcelain','--untracked-files=all'],cwd=code,text=True)
    if head != commit or dirty.strip():
        raise ValueError('Seal only the exact clean committed execution checkout')
    remote = subprocess.check_output(['git','config','--get','remote.origin.url'],cwd=code,text=True).strip()
    return dict(commit=commit, dirty=False, repository=remote, source_path=str(remote_code))


def seal(code, commit, output, *, remote_root, remote_code, prior=PRIOR,
         retained_qualification_bytes=2**30, retrieval_staging=False):
    code, output, prior = Path(code), Path(output), Path(prior)
    remote_root, remote_code = Path(remote_root), Path(remote_code)
    if not remote_root.is_absolute() or not remote_code.is_absolute():
        raise ValueError('Actual absolute run/source destinations are required')
    source = execution_source(code,commit,remote_code)
    old, authenticated = _authenticate_prior(prior)
    old_root = Path(old['root'])
    binding = stable_digest(dict(schema=SCHEMA, source_commit=commit, rho=RHO,
        inputs=authenticated, budgets=budgets(), analysis=ANALYSIS,
        origins={d:{k:old['datasets'][d][k] for k in
            ('origin_state','origin_state_identity','origin_weights','origin_weights_identity')}
            for d in DATASETS}))
    output.mkdir(parents=True, exist_ok=False)
    (output/'sealed').mkdir()
    for name in ('ORIGINAL_CONFIG.yaml','MODEL_PIN.json','RUNTIME_PIN.json','TRAINING_PIN.json'):
        shutil.copy2(prior/'sealed'/name,output/'sealed'/name)
    shutil.copytree(prior/'sealed/reference_evidence',output/'sealed/reference_evidence')
    for dataset in DATASETS:
        file = output/'sealed/reference_evidence'/dataset/'BINDING.json'
        evidence = read(file)
        for row in evidence['proof']['files']+evidence['native']['records']:
            row['retained_path'] = str(remote_root/Path(row['retained_path']).relative_to(old_root))
        write_json(file,evidence)
    plan = dict(schema=SCHEMA, root=str(remote_root), binding=binding,
        execution_source=source, arms=list(ARMS), datasets={}, events=list(EVENTS), rho=RHO,
        budgets=budgets(), analysis=deepcopy(ANALYSIS), automatic_retry=False,
        operational_timeout=dict(seconds=24*3600, signal='TERM', kill_after_seconds=120, automatic_retry=False),
        target=old['target'], draft=old['draft'],
        worker_python=old['worker_python'], training_python=old['training_python'],
        training_gpu=old['training_gpu'], original_config=str(remote_root/'sealed/ORIGINAL_CONFIG.yaml'),
        qualification_admission=str(remote_root/'qualification/ADMISSION.json'),
        shared_feedback_preparation=str(remote_root/'qualification_feedback/PREPARATION.json'),
        feedback_accounting=dict(unique_feedback=512, qualification_new=4,
            main_new=508, main_reused_from_qualification=4, canonical_generations=0),
        qualification_storage=dict(required_initial_free_bytes=40*2**30,
            reserve_bytes=8*2**30, retained_qualification_bytes=retained_qualification_bytes,
            policy='One baseline and one current disposable candidate, graded/served then retired; failed tensors retained'),
        author_experiment_exception='User scoped D24 exception for this fixed ABT pilot; not FreeLB reproduction',
        live_origin_and_storage_admitted=False, native_qualification_passed=False,
        main_storage=capacity_plan(max(v['origin_weights_identity']['bytes'] for v in old['datasets'].values()),
            max(v['origin_state_identity']['bytes'] for v in old['datasets'].values()),
            retained_qualification_bytes=retained_qualification_bytes,retrieval_staging=retrieval_staging))
    config_text = (output/'sealed/ORIGINAL_CONFIG.yaml').read_text()
    for dataset in DATASETS:
        destination = output/'sealed'/dataset
        destination.mkdir()
        for name in ('TRAINING.json','PANEL.json'):
            shutil.copy2(prior/'sealed'/dataset/name,destination/name)
        canonicals = read(prior/'sealed'/dataset/'CANONICALS.json')
        for row in canonicals.values():
            row['binding_digest'] = binding
        write_json(destination/'CANONICALS.json',canonicals)
        item = {k:deepcopy(v) for k,v in old['datasets'][dataset].items() if k.startswith('origin_')}
        item.update(training_manifest=str(remote_root/'sealed'/dataset/'TRAINING.json'),
            panel_manifest=str(remote_root/'sealed'/dataset/'PANEL.json'),
            references=str(remote_root/'sealed'/dataset/'CANONICALS.json'),
            canonical_binding_evidence=str(remote_root/'sealed/reference_evidence'/dataset/'BINDING.json'),
            configs={})
        panel = read(destination/'PANEL.json')['requests']
        item['evaluation_schedule'] = evaluation_schedule(panel,dataset)
        validate_evaluation_inputs(panel,{arm:'/future/'+dataset+'/'+arm for arm in ARMS},canonicals,
                                   dataset=dataset,event=6,schedule=item['evaluation_schedule'])
        for arm in ARMS:
            cfg = yaml.safe_load(config_text)
            cfg['run_id'] = f'kplus_abt_{dataset}_{arm}'
            cfg['output_dir'] = str(remote_root/'campaign_001'/dataset/arm)
            text = yaml.safe_dump(cfg,sort_keys=False)
            validate_config_text(text,config_text)
            path = destination/f'config_{arm}.yaml';path.write_text(text)
            item['configs'][arm] = str(remote_root/'sealed'/dataset/path.name)
        plan['datasets'][dataset] = item
    write_json(output/'sealed/PLAN.json',plan)
    files = {}
    for path in sorted((output/'sealed').rglob('*')):
        if path.is_file():
            files[str(remote_root/path.relative_to(output))] = identity(path)
    for item in plan['datasets'].values():
        files[item['origin_state']] = item['origin_state_identity']
        files[item['origin_weights']] = item['origin_weights_identity']
    for encoded in subprocess.check_output(['git','ls-files','-z','.'],cwd=code).split(b'\0'):
        if encoded:
            path = code/encoded.decode()
            if path.is_file():
                files[str(remote_code/path.relative_to(code))] = identity(path)
    write_json(output/'sealed/SEAL.json',dict(schema=SCHEMA,binding=binding,
        execution_source=source,files=files,model_work=0,native_qualification_passed=False))
    return dict(status='sealed', binding=binding, files=len(files), model_work=0,
                native_qualification_passed=False,live_storage_admitted=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code',default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument('--commit',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--remote-root',required=True)
    parser.add_argument('--remote-code',required=True)
    parser.add_argument('--prior',default=str(PRIOR))
    parser.add_argument('--retained-qualification-bytes',type=int,default=2**30)
    parser.add_argument('--retrieval-staging',action='store_true')
    args = parser.parse_args()
    print(json.dumps(seal(args.code,args.commit,args.output,remote_root=args.remote_root,
        remote_code=args.remote_code,prior=args.prior,
        retained_qualification_bytes=args.retained_qualification_bytes,
        retrieval_staging=args.retrieval_staging),indent=2))
