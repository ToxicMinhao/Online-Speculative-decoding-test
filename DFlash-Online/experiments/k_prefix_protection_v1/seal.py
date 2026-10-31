"""Bind CPU preparation, reused pools and published source at actual execution paths."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import yaml

from online_feedback.storage import write_json
from experiments.ok_events_5_8.storage import identity
from .contract import NEW_BUDGET, QUALIFICATION_BUDGET, ORIGINAL_COMMIT, validate_plan
from .storage import future_capacity


def seal(args):
    source, prepared = Path(args.code).resolve(), Path(args.prepared).resolve()
    original = Path(args.original).resolve()
    output, remote = Path(args.output).resolve(), Path(args.remote_root)
    remote_code=Path(args.remote_code)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
    if subprocess.check_output(['git','status','--porcelain'],cwd=source,text=True):
        raise ValueError('Publish clean exact code before execution sealing')
    preparation=json.loads((prepared/'SEAL.json').read_text())
    for name,expected in preparation['files'].items():
        if identity(prepared/name) != {k:expected[k] for k in ('bytes','sha256')}:
            raise ValueError('Prepared file changed: '+name)
    audit=json.loads(Path(args.inventory).read_text())
    if audit['status'] != 'passed' or audit['model_work'] != 0:
        raise ValueError('Require the completed CPU-only original evidence audit')
    original_plan=json.loads((original/'sealed/PLAN.json').read_text())
    data=json.loads((prepared/'DATA_CONTRACT.json').read_text())
    output.mkdir(parents=True,exist_ok=False)
    sealed=output/'sealed'
    sealed.mkdir()
    shutil.copytree(prepared,output/'data_preparation')
    shutil.copy2(args.inventory,sealed/'ORIGINAL_ARTIFACT_AUDIT.json')
    for name in ('MODEL_PIN.json','RUNTIME_PIN.json','TRAINING_PIN.json'):
        shutil.copy2(original/'sealed'/name,sealed/name)
    configs={}
    for seed in (17,29):
        configs[str(seed)]={}
        # Local compact retrieval preserves the exact original config bytes.
        for arm in ('K_HALF','KP'):
            raw=yaml.safe_load((original/'sealed'/f'config_seed_{seed}.yaml').read_text())
            raw['training']['learning_rate']=1.5e-4 if arm=='K_HALF' else 3e-4
            raw['run_id']=f'k-prefix-protection-v1-{seed}-{arm}'
            raw['output_dir']=str(remote/'campaign_001'/f'seed_{seed}'/arm)
            name=f'config_seed_{seed}_{arm}.yaml'
            (sealed/name).write_text(yaml.safe_dump(raw,sort_keys=False))
            configs[str(seed)][arm]=str(remote/'sealed'/name)
    execution={'commit':commit,'dirty':False,'source_path':str(remote_code),
               'repository':'https://github.com/ToxicMinhao/Online-Speculative-decoding-test'}
    binding={'upstream_data_binding':audit['original_data_binding'],
             'execution_commit':commit,'original_execution_commit':ORIGINAL_COMMIT,
             'evaluation_binding':data['evaluation_binding']}
    plan={name:audit[name] for name in ('original_run','original_configs','original_campaign_contracts',
        'original_initial_states','original_k4','pools','original_schedules','qualification','target','draft')}
    plan.update(configs=configs,training_binding=binding,execution_source=execution,
        original_execution_commit=ORIGINAL_COMMIT,training_seeds=[17,29],stop_step=512,
        inherited_horizon=7424,no_automatic_retry=True,new_budget=NEW_BUDGET,
        qualification_budget=QUALIFICATION_BUDGET,
        evaluation_requests=json.loads((prepared/'EVALUATION_REQUESTS.json').read_text()),
        evaluation_schedule=json.loads((prepared/'EVALUATION_SCHEDULE.json').read_text()),
        worker_python=original_plan['worker_python'],training_python=original_plan['training_python'],
        serving=original_plan['serving'],storage=future_capacity())
    write_json(sealed/'PLAN_CHECK.json',validate_plan(plan))
    write_json(sealed/'PLAN.json',plan)
    files={}
    tracked=subprocess.check_output(['git','ls-files','-z','.'],cwd=source).split(b'\0')
    for encoded in tracked:
        if encoded:
            name=encoded.decode()
            if (source/name).is_file(): files[str(remote_code/name)]=identity(source/name)
    for path in sorted(output.rglob('*')):
        if path.is_file(): files[str(remote/path.relative_to(output))]=identity(path)
    exact={str(remote_code/'experiments/k_prefix_protection_v1'):
           sorted(p.name for p in (source/'experiments/k_prefix_protection_v1').iterdir() if p.is_file())}
    write_json(sealed/'SEAL.json',dict(execution_source=execution,files=files,exact_file_sets=exact))
    return {'status':'sealed_before_model_work','execution_source':execution,'files':len(files),
            'storage':plan['storage']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('code','prepared','original','inventory','output','remote-root','remote-code'):
        parser.add_argument('--'+name,required=True)
    print(json.dumps(seal(parser.parse_args())))

if __name__=='__main__':
    main()
