"""Seal the retained PR29 setup, three controls and literal-loss qualification inputs."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

import yaml
from online_feedback.storage import write_json
from online_feedback.adaflash_repo import adaflash_repo_contract
from experiments.ok_coding.storage import identity
from .contract import (SCHEMA, ARMS, METHODS, EVENTS, LEARNING_RATES, NEW_BUDGET,
                       ANALYSIS, SERVING, stable_digest, validate_plan,
                       final_schedule, schedule_balance)
from .storage import capacity_plan
from .training import validate_config_text


def seal(args):
    code, prior, output = map(Path, (args.code, args.prior_sealed, args.output))
    remote, remote_code = Path(args.remote_root), Path(args.remote_code)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=code, text=True).strip()
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=code, text=True):
        raise ValueError('Publish a clean execution commit before sealing')
    old = json.loads((prior / 'PLAN.json').read_text())
    audit = json.loads((prior / 'CONTROL_AUDIT.json').read_text())
    export_audit = json.loads(Path(args.export_audit).read_text())
    if (old['schema'] != 'adaflash_c1_v1' or audit['status'] != 'passed'
            or export_audit['status'] != 'passed'
            or audit['model_work'] != 0 or export_audit['model_work'] != 0):
        raise ValueError('Require authenticated PR29 inputs and retained export evidence')
    names = ('ORIGINAL_MANIFEST.json', 'ORIGINAL_CONFIG.yaml', 'CANONICALS.json',
             'MODEL_PIN.json', 'RUNTIME_PIN.json', 'TRAINING_PIN.json')
    for name in names:
        original = str(Path(audit['original_root']) / 'sealed' / name)
        if identity(prior / name) != audit['files'][original]:
            raise ValueError('Retained PR29 input differs: ' + name)
    output.mkdir(parents=True, exist_ok=False)
    sealed = output / 'sealed'
    sealed.mkdir()
    for name in names:
        shutil.copy2(prior / name, sealed / name)
    shutil.copy2(prior / 'CONTROL_AUDIT.json', sealed / 'CONTROL_AUDIT.json')
    shutil.copy2(args.export_audit, sealed / 'EXPORT_AUDIT.json')
    shutil.copy2(args.handoff, sealed / 'HANDOFF.md')
    write_json(sealed / 'LOSS_CONTRACT.json', adaflash_repo_contract())
    raw = yaml.safe_load((sealed / 'ORIGINAL_CONFIG.yaml').read_text())
    raw['run_id'] = 'adaflash-released-loss-c1'
    raw['output_dir'] = str(remote / 'campaign_001/arms/AREPO')
    config = yaml.safe_dump(raw, sort_keys=False)
    validate_config_text(config, (sealed / 'ORIGINAL_CONFIG.yaml').read_text())
    (sealed / 'config_AREPO.yaml').write_text(config)
    execution = dict(commit=commit, dirty=False, source_path=str(remote_code),
        repository='https://github.com/ToxicMinhao/Online-Speculative-decoding-test')
    references = {arm: export_audit['checkpoints'][arm] for arm in ('O', 'KPLUS', 'A')}
    weight_bytes = json.loads((sealed / 'MODEL_PIN.json').read_text())['draft']['files']['model.safetensors']['bytes']
    plan = dict(schema=SCHEMA, root=str(remote), arms=list(ARMS), methods=list(METHODS),
        seed=17, events=EVENTS, budget=NEW_BUDGET, analysis=ANALYSIS, serving=SERVING,
        learning_rates=LEARNING_RATES, intermediate_events=[], automatic_retry=False,
        execution_source=execution, loss=adaflash_repo_contract(),
        source_parent=old['execution_source']['commit'],
        manifest=str(remote / 'sealed/ORIGINAL_MANIFEST.json'), manifest_binding=old['manifest_binding'],
        original_config=str(remote / 'sealed/ORIGINAL_CONFIG.yaml'),
        configs={'AREPO': str(remote / 'sealed/config_AREPO.yaml')},
        canonicals=str(remote / 'sealed/CANONICALS.json'), references=references,
        original_initial_state=old['original_initial_state'],
        original_event_results=old['original_event_results'],
        original_full_k_plan=old['original_full_k_plan'],
        evaluation_requests=old['evaluation_requests'],
        evaluation_schedule=final_schedule(old['evaluation_requests']),
        target=old['target'], draft=old['draft'], training_python=old['training_python'],
        worker_python=old['worker_python'], serving_gpu=0, training_gpu=1,
        binding=stable_digest(dict(manifest=old['manifest_binding'], source=execution,
            config=raw, references=references, loss=adaflash_repo_contract())),
        storage=capacity_plan(weight_bytes),
        qualification=dict(directory=str(remote/'qualification'), maximum_optimizer_steps=6, timeout_s=1800,
            config=args.qualification_config, initial_state=args.qualification_initial_state,
            prepared=args.qualification_prepared))
    local = {**plan, **{key: str(sealed / Path(plan[key]).name)
                       for key in ('manifest', 'canonicals')}}
    write_json(sealed / 'PLAN_CHECK.json', validate_plan(local))
    write_json(sealed / 'ORDER_BALANCE.json', schedule_balance(plan['evaluation_schedule']))
    write_json(sealed / 'PLAN.json', plan)
    files = {}
    for encoded in subprocess.check_output(['git', 'ls-files', '-z', '.'], cwd=code).split(b'\0'):
        if encoded:
            name = encoded.decode()
            if (code / name).is_file():
                files[str(remote_code / name)] = identity(code / name)
    for path in sorted(output.rglob('*')):
        if path.is_file():
            files[str(remote / path.relative_to(output))] = identity(path)
    folder = code / 'experiments/adaflash_arepo'
    write_json(sealed / 'SEAL.json', dict(execution_source=execution, files=files,
        exact_file_sets={str(remote_code / 'experiments/adaflash_arepo'):
                        sorted(p.name for p in folder.iterdir() if p.is_file())}))
    return dict(status='sealed_before_model_work', execution_source=execution,
                files=len(files), budget=NEW_BUDGET, storage=plan['storage'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('code', 'prior-sealed', 'export-audit', 'handoff',
                 'output', 'remote-root', 'remote-code', 'qualification-config',
                 'qualification-initial-state', 'qualification-prepared'):
        parser.add_argument('--' + name, required=True)
    print(json.dumps(seal(parser.parse_args()), indent=2))


if __name__ == '__main__':
    main()
