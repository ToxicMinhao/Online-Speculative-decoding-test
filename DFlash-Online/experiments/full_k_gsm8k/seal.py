"""Bind the complete original request stream to the published two-arm runner."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import yaml

from online_feedback.storage import write_json
from experiments.ok_events_5_8.storage import identity
from .contract import (ARMS, EVENTS, INTERMEDIATE_EVENTS, LEARNING_RATES, NEW_BUDGET,
                       SERVING, stable_digest, validate_plan)
from .evaluation import final_schedule, intermediate_schedule, schedule_balance
from .storage import capacity_plan
from .training import validate_config_text


def seal(args):
    source, inventory, pins, output = map(Path, (args.code, args.inventory, args.pins, args.output))
    remote, remote_code = Path(args.remote_root), Path(args.remote_code)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip()
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=source, text=True):
        raise ValueError('Publish exact clean source before sealing')
    audit = json.loads(inventory.read_text())
    if audit['status'] != 'passed' or audit['model_work'] != 0 or audit['raw_generations'] != 6144:
        raise ValueError('The complete authoritative original-evidence audit must pass')
    prepared = inventory.parent
    if identity(prepared / 'CANONICALS.json') != audit['canonical_identity']:
        raise ValueError('Retrieved canonical evidence differs from the authoritative audit')
    original_path = prepared / 'ORIGINAL_CONFIG.yaml'
    if identity(original_path) != audit['files'][audit['original_config']]:
        raise ValueError('Original recipe bytes differ from the audited source')
    if identity(prepared / 'ORIGINAL_MANIFEST.json') != audit['files'][audit['original_manifest']]:
        raise ValueError('Original request manifest bytes differ from the audited source')
    output.mkdir(parents=True, exist_ok=False)
    sealed = output / 'sealed'
    sealed.mkdir()
    for name in ('REUSE_AUDIT.json', 'ORIGINAL_MANIFEST.json', 'ORIGINAL_CONFIG.yaml', 'CANONICALS.json'):
        shutil.copy2(prepared / name, sealed / name)
    for name in ('MODEL_PIN.json', 'RUNTIME_PIN.json', 'TRAINING_PIN.json'):
        shutil.copy2(pins / name, sealed / name)
    configs, config_contracts = {}, {}
    for arm in ARMS:
        raw = yaml.safe_load(original_path.read_text())
        raw['training']['learning_rate'] = LEARNING_RATES[arm]
        raw['run_id'] = 'full-k-gsm8k-' + arm.lower()
        raw['output_dir'] = str(remote / 'campaign_001/arms' / arm)
        text = yaml.safe_dump(raw, sort_keys=False)
        validate_config_text(text, arm, original_path.read_text())
        name = f'config_{arm}.yaml'
        (sealed / name).write_text(text)
        configs[arm] = str(remote / 'sealed' / name)
        config_contracts[arm] = raw
    source_binding = dict(commit=commit, dirty=False, source_path=str(remote_code),
        repository='https://github.com/ToxicMinhao/Online-Speculative-decoding-test')
    checkpoints = dict(audit['checkpoints'])
    checkpoints.update({arm: str(remote / 'campaign_001/arms' / arm / 'trainer/events/event_058/export')
                        for arm in ARMS})
    requests = audit['requests']
    plan = dict(schema='full_k_gsm8k_v1', arms=list(ARMS), seed=17, events=EVENTS,
        budget=NEW_BUDGET, serving=SERVING, intermediate_events=list(INTERMEDIATE_EVENTS),
        learning_rates=LEARNING_RATES, automatic_retry=False, execution_source=source_binding,
        manifest=str(remote / 'sealed/ORIGINAL_MANIFEST.json'), manifest_binding=audit['manifest_binding'],
        original_config=str(remote / 'sealed/ORIGINAL_CONFIG.yaml'), configs=configs,
        canonicals=str(remote / 'sealed/CANONICALS.json'), evaluation_requests=requests,
        checkpoints=checkpoints, original_run=audit['original_run'], target=audit['target'], draft=audit['draft'],
        training_python=audit['training_python'], worker_python=audit['worker_python'],
        serving_gpu=0, training_gpu=1,
        binding=stable_digest(dict(manifest=audit['manifest_binding'], source=source_binding, configs=config_contracts)),
        evaluation_schedule=final_schedule(requests),
        intermediate_schedules={f'{arm}/{event}': intermediate_schedule(requests, arm, event)
            for arm in ARMS for event in INTERMEDIATE_EVENTS},
        storage=capacity_plan(audit['files'][str(Path(audit['draft']) / 'model.safetensors')]['bytes']))
    write_json(sealed / 'PLAN_CHECK.json', validate_plan({**plan, 'manifest': str(sealed / 'ORIGINAL_MANIFEST.json')}))
    write_json(sealed / 'ORDER_BALANCE.json', schedule_balance(plan['evaluation_schedule']))
    write_json(sealed / 'PLAN.json', plan)
    files = {}
    for encoded in subprocess.check_output(['git', 'ls-files', '-z', '.'], cwd=source).split(b'\0'):
        if encoded:
            name = encoded.decode()
            if (source / name).is_file():
                files[str(remote_code / name)] = identity(source / name)
    for path in sorted(output.rglob('*')):
        if path.is_file():
            files[str(remote / path.relative_to(output))] = identity(path)
    folder = source / 'experiments/full_k_gsm8k'
    exact = {str(remote_code / 'experiments/full_k_gsm8k'):
             sorted(path.name for path in folder.iterdir() if path.is_file())}
    write_json(sealed / 'SEAL.json', dict(execution_source=source_binding, files=files, exact_file_sets=exact))
    return dict(status='sealed_before_model_work', execution_source=source_binding,
                files=len(files), budget=NEW_BUDGET, storage=plan['storage'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('code', 'inventory', 'pins', 'output', 'remote-root', 'remote-code'):
        parser.add_argument('--' + name, required=True)
    print(json.dumps(seal(parser.parse_args()), indent=2))


if __name__ == '__main__':
    main()
