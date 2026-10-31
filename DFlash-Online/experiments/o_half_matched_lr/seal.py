"""Seal the unchanged panel, exact reuse audit and published new harness."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import yaml

from online_feedback.storage import write_json
from experiments.ok_events_5_8.storage import identity
from .contract import NEW_BUDGET, ORIGINAL_COMMIT, PANEL_BINDING, validate_plan
from .evaluation import bridge_schedule, final_schedule
from .storage import future_capacity


def seal(args):
    source, inventory, pins, output = map(Path, (args.code, args.inventory, args.pins, args.output))
    remote, remote_code = Path(args.remote_root), Path(args.remote_code)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip()
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=source, text=True):
        raise ValueError('Publish exact clean source before GPU sealing')
    audit = json.loads(inventory.read_text())
    archive = json.loads(Path(args.archive).read_text())
    original = json.loads((pins / 'PLAN.json').read_text())
    if audit['status'] != 'passed' or audit['model_work'] != 0 or archive['status'] != 'passed':
        raise ValueError('Reuse audit must pass on authoritative retained evidence')
    if identity(args.archive) != audit['archive']['identity']:
        raise ValueError('Retrieved immutable archive differs from authoritative audit')
    output.mkdir(parents=True, exist_ok=False)
    sealed = output / 'sealed'
    sealed.mkdir()
    shutil.copy2(inventory, sealed / 'REUSE_AUDIT.json')
    shutil.copy2(args.archive, sealed / 'ARCHIVE.json')
    for name in ('MODEL_PIN.json', 'RUNTIME_PIN.json', 'TRAINING_PIN.json'):
        shutil.copy2(pins / name, sealed / name)
    configs = {}
    for seed in (17, 29):
        raw = yaml.safe_load((pins / f'config_seed_{seed}.yaml').read_text())
        raw['training']['learning_rate'] = 1.5e-4
        raw['run_id'] = f'o-half-matched-lr-{seed}'
        raw['output_dir'] = str(remote / 'campaign_001' / f'seed_{seed}/O_HALF')
        name = f'config_seed_{seed}_O_HALF.yaml'
        (sealed / name).write_text(yaml.safe_dump(raw, sort_keys=False))
        configs[str(seed)] = {'O_HALF': str(remote / 'sealed' / name)}
    source_binding = dict(commit=commit, dirty=False, source_path=str(remote_code),
                          repository='https://github.com/ToxicMinhao/Online-Speculative-decoding-test')
    keys = ('original_run', 'pr21_run', 'original_configs', 'original_campaign_contracts',
            'original_initial_states', 'original_endpoints', 'original_algorithm_contracts',
            'original_optimizer_recipes', 'pools', 'original_schedules', 'target', 'draft')
    plan = {key: audit[key] for key in keys}
    requests = archive['requests']
    plan.update(configs=configs, execution_source=source_binding,
                archive=dict(path=str(remote / 'sealed/ARCHIVE.json'), identity=identity(sealed / 'ARCHIVE.json')),
                training_binding=dict(upstream_data_binding=audit['original_data_binding'],
                    evaluation_binding=PANEL_BINDING, execution_commit=commit,
                    original_execution_commit=ORIGINAL_COMMIT),
                original_execution_commit=ORIGINAL_COMMIT, training_seeds=[17, 29],
                stop_step=512, inherited_horizon=7424, no_automatic_retry=True,
                new_budget=NEW_BUDGET, evaluation_requests=requests,
                bridge_schedule=sum((bridge_schedule(requests, seed) for seed in (17, 29)), []),
                evaluation_schedule=sum((final_schedule(requests, seed) for seed in (17, 29)), []),
                worker_python=original['worker_python'], training_python=original['training_python'],
                serving=original['serving'], storage=future_capacity())
    write_json(sealed / 'PLAN_CHECK.json', validate_plan(plan))
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
    folder = source / 'experiments/o_half_matched_lr'
    exact = {str(remote_code / 'experiments/o_half_matched_lr'): sorted(p.name for p in folder.iterdir() if p.is_file())}
    write_json(sealed / 'SEAL.json', dict(execution_source=source_binding, files=files, exact_file_sets=exact))
    return dict(status='sealed_before_model_work', execution_source=source_binding,
                files=len(files), budget=NEW_BUDGET, storage=plan['storage'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('code', 'inventory', 'archive', 'pins', 'output', 'remote-root', 'remote-code'):
        parser.add_argument('--' + name, required=True)
    print(json.dumps(seal(parser.parse_args()), indent=2))


if __name__ == '__main__':
    main()
