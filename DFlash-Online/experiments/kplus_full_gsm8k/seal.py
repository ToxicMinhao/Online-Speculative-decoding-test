"""Bind the unchanged GSM8K stream and K58 control before model work."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

import yaml
from online_feedback.storage import write_json
from experiments.ok_coding.storage import identity
from .contract import (SCHEMA, ARMS, EVENTS, LEARNING_RATES, NEW_BUDGET,
                       SERVING, stable_digest, validate_plan)
from .evaluation import final_schedule, schedule_balance
from .storage import capacity_plan
from .training import validate_config_text


def seal(args):
    code, prior, output = map(Path, (args.code, args.prior_sealed, args.output))
    remote, remote_code = Path(args.remote_root), Path(args.remote_code)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=code, text=True).strip()
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=code, text=True):
        raise ValueError('Publish a clean execution commit before sealing')
    old = json.loads((prior / 'PLAN.json').read_text())
    audit = json.loads(Path(args.control_audit).read_text())
    if (audit['status'] != 'passed' or audit['model_work'] != 0
            or old['execution_source']['commit'] != '0698eeb2ddd97b9e2ac6953989dc82acbc0a1e34'):
        raise ValueError('Require authenticated original full-K evidence before sealing')
    for name in ('PLAN.json', 'ORIGINAL_MANIFEST.json', 'ORIGINAL_CONFIG.yaml', 'CANONICALS.json',
                 'MODEL_PIN.json', 'RUNTIME_PIN.json', 'TRAINING_PIN.json'):
        original = str(Path(audit['original_root']) / 'sealed' / name)
        if identity(prior / name) != audit['files'][original]:
            raise ValueError('Local retained input differs from authenticated original: ' + name)
    output.mkdir(parents=True, exist_ok=False)
    sealed = output / 'sealed'
    sealed.mkdir()
    for name in ('ORIGINAL_MANIFEST.json', 'ORIGINAL_CONFIG.yaml', 'CANONICALS.json',
                 'MODEL_PIN.json', 'RUNTIME_PIN.json', 'TRAINING_PIN.json'):
        shutil.copy2(prior / name, sealed / name)
    shutil.copy2(args.control_audit, sealed / 'CONTROL_AUDIT.json')
    shutil.copy2(args.baseline_counts, sealed / 'BASELINE_COUNTS.json')
    if identity(sealed / 'BASELINE_COUNTS.json') != audit['baseline_counts_identity']:
        raise ValueError('Baseline rows differ from the authenticated retained trajectories')
    raw = yaml.safe_load((sealed / 'ORIGINAL_CONFIG.yaml').read_text())
    raw['training']['learning_rate'] = 3e-4
    raw['run_id'] = 'kplus-full-gsm8k'
    raw['output_dir'] = str(remote / 'campaign_001/arms/KPLUS')
    config = yaml.safe_dump(raw, sort_keys=False)
    validate_config_text(config, (sealed / 'ORIGINAL_CONFIG.yaml').read_text())
    (sealed / 'config_KPLUS.yaml').write_text(config)
    execution = dict(commit=commit, dirty=False, source_path=str(remote_code),
        repository='https://github.com/ToxicMinhao/Online-Speculative-decoding-test')
    weight_bytes = json.loads((sealed / 'MODEL_PIN.json').read_text())['draft']['files']['model.safetensors']['bytes']
    plan = dict(schema=SCHEMA, root=str(remote), arms=list(ARMS), seed=17, events=EVENTS,
        budget=NEW_BUDGET, serving=SERVING, learning_rates=LEARNING_RATES,
        intermediate_events=[], automatic_retry=False, execution_source=execution,
        source_parent='4d94f30c42701d0f86ac04e2829c834593c16d8e',
        manifest=str(remote / 'sealed/ORIGINAL_MANIFEST.json'), manifest_binding=old['manifest_binding'],
        original_config=str(remote / 'sealed/ORIGINAL_CONFIG.yaml'),
        configs={'KPLUS': str(remote / 'sealed/config_KPLUS.yaml')},
        canonicals=str(remote / 'sealed/CANONICALS.json'),
        baseline_counts=str(remote / 'sealed/BASELINE_COUNTS.json'),
        references={'K': old['checkpoints']['K']},
        original_initial_state=audit['original_initial_state'],
        original_event_results=audit['original_event_results'],
        original_full_k_plan=audit['original_plan'],
        evaluation_requests=old['evaluation_requests'], evaluation_schedule=final_schedule(old['evaluation_requests']),
        target=old['target'], draft=old['draft'], training_python=old['training_python'],
        worker_python=old['worker_python'], serving_gpu=0, training_gpu=1,
        binding=stable_digest(dict(manifest=old['manifest_binding'], source=execution, config=raw,
                                   retained_control=audit['control_identity'])),
        storage=capacity_plan(weight_bytes))
    local = {**plan, **{key: str(sealed / Path(plan[key]).name)
                       for key in ('manifest', 'canonicals', 'baseline_counts')}}
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
    folder = code / 'experiments/kplus_full_gsm8k'
    write_json(sealed / 'SEAL.json', dict(execution_source=execution, files=files,
        exact_file_sets={str(remote_code / 'experiments/kplus_full_gsm8k'):
                        sorted(p.name for p in folder.iterdir() if p.is_file())}))
    return dict(status='sealed_before_model_work', execution_source=execution,
                files=len(files), budget=NEW_BUDGET, storage=plan['storage'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('code', 'prior-sealed', 'control-audit', 'baseline-counts',
                 'output', 'remote-root', 'remote-code'):
        parser.add_argument('--' + name, required=True)
    print(json.dumps(seal(parser.parse_args()), indent=2))


if __name__ == '__main__':
    main()
