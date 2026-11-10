"""Seal the remaining qualification work after the immutable-origin repair.

The first ordinary K+ update is reused from its decoded artifact. Its already
consumed work remains in the cumulative ledger; no response is regenerated and
the main pilot remains outside this launch.
"""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess

import yaml

from experiments.ok_coding.storage import identity
from experiments.role_loss_pilot.training import validate_config_text
from online_feedback.storage import write_json
from . import seal as original


FAILED_COMMIT = 'f834dbd0da553d6231e39a4b5ff7a3b10a5dd623'
REMOTE_CODE = Path('/data/sources/minhao/fsg_source_20261108_repair/DFlash-Online')
REMOTE_ROOT = Path('/data/sources/minhao/runs/dflash_functional_step_geometry_20261108/qualification_002')
OLD_ROOT = original.REMOTE_ROOT
REFERENCE_FILES = {
    'presentations': 'qualification/gsm8k/KPLUS/PRESENTATIONS.json',
    'decoded_state': 'qualification/gsm8k/KPLUS/DECODED_STATE.json',
    'response_observations': 'qualification/gsm8k/trainer/events/disposable_kplus/RESPONSE_OBSERVATIONS.jsonl',
    'window_observations': 'qualification/gsm8k/trainer/events/disposable_kplus/WINDOW_OBSERVATIONS.jsonl',
    'startup_restoration': 'qualification/gsm8k/STARTUP_EXACT_RESTORATION.json',
    'failed_result': 'qualification/gsm8k/KPLUS/RESULT.json',
    'failed_qualification_summary': 'qualification/QUALIFICATION.json',
    'prior_preflight': 'PREFLIGHT.json',
}


def validate_failed_boundary(failure, previous):
    """Accept only the documented post-Adam restore failure with exact accounting."""
    failure = Path(failure)
    summary = json.loads((failure / 'qualification/QUALIFICATION.json').read_text())
    if (previous['execution_source']['commit'] != FAILED_COMMIT
            or summary['execution_commit'] != FAILED_COMMIT or summary['status'] != 'failed'
            or summary['main_pilot_started']
            or 'Native Adam counter differs:' not in summary['error']):
        raise ValueError('Repair requires the preserved first ordinary-update restore failure')
    counts = Counter()
    for line in (failure / 'QUALIFICATION_WORK.jsonl').read_text().splitlines():
        row = json.loads(line)
        counts[(row['kind'], row['status'])] += 1
    expected = {}
    for kind, amount in (('adam', 1), ('ordinary_forward', 2), ('ordinary_backward', 2),
                         ('additional_forward', 2), ('additional_backward', 2)):
        for status in ('attempted', 'completed'):
            expected[(kind, status)] = amount
    if counts != Counter(expected):
        raise ValueError('Prior qualification work differs; do not reset or hide attempted work')
    for relative in ('qualification/gsm8k/KPLUS/DECODED_STATE.json',
                     'qualification/gsm8k/STARTUP_EXACT_RESTORATION.json'):
        if json.loads((failure / relative).read_text())['exact_equal'] is not True:
            raise ValueError('Retained ordinary state or startup equality failed')
    presentations = json.loads((failure / REFERENCE_FILES['presentations']).read_text())
    if len(presentations) != 2 or [r['optimizer_stepped'] for r in presentations] != [False, True]:
        raise ValueError('Retained ordinary pair is incomplete')
    return {f'{kind}_{status}': count for (kind, status), count in sorted(counts.items())}


def retained_dependencies(failure, previous, payload_identity, old_sealed):
    """Authenticate compact local records; actual remote tensor bytes gate preflight."""
    failure = Path(failure)
    preparation_path = failure / 'qualification_feedback/PREPARATION.json'
    preparation = json.loads(preparation_path.read_text())
    if (preparation['status'] != 'completed' or preparation['shared_feedback_generations'] != 4
            or preparation['teacher_preparations'] != 4
            or preparation['extra_qualification_generations'] != 0 or preparation['main_updates'] != 0):
        raise ValueError('Require the four existing shared packets without recollection')
    dependencies = {}
    for dataset, rows in preparation['datasets'].items():
        if dataset not in ('gsm8k', 'mathqa') or len(rows) != 2:
            raise ValueError('Require exactly two shared packets per original dataset')
        old_manifest = Path(previous['datasets'][dataset]['training_manifest']).relative_to(OLD_ROOT / 'sealed')
        selected = json.loads((Path(old_sealed) / old_manifest).read_text())['train_requests'][:2]
        if [r['request_id'] for r in rows] != [r['request_id'] for r in selected]:
            raise ValueError('Shared packets are not the first prescribed event-5 pair')
        for row, request in zip(rows, selected):
            if row['status'] != 'completed':
                raise ValueError('A shared feedback packet is incomplete')
            record = Path(row['record_path'])
            local = failure / record.relative_to(OLD_ROOT)
            if identity(local) != row['record_identity']:
                raise ValueError('Retained canonical record differs')
            raw = json.loads(local.read_text())
            if (raw['record_id'] != request['request_id']
                    or raw['prompt_ids'] != request['prompt_ids']
                    or raw['checkpoint_id'] != previous['datasets'][dataset]['origin_export']):
                raise ValueError('Retained feedback differs from its exact input or KPLUS4 producer')
            dependencies[str(record)] = row['record_identity']
            feature = Path(row['feature_path'])
            feature.relative_to(OLD_ROOT / 'qualification_feedback' / dataset / 'features')
            dependencies[str(feature)] = row['feature_identity']
    if set(preparation['datasets']) != {'gsm8k', 'mathqa'}:
        raise ValueError('Both dataset feedback pools must remain available')
    expected_path = OLD_ROOT / 'qualification/gsm8k/KPLUS/disposable_state.pt'
    if (Path(payload_identity['path']) != expected_path
            or set(payload_identity['identity']) != {'bytes', 'sha256'}
            or payload_identity['identity']['bytes'] <= 0
            or len(payload_identity['identity']['sha256']) != 64
            or any(c not in '0123456789abcdef' for c in payload_identity['identity']['sha256'])):
        raise ValueError('Require the actual retained disposable-state identity')
    references = {'disposable_state': deepcopy(payload_identity)}
    dependencies[str(expected_path)] = payload_identity['identity']
    for key, relative in REFERENCE_FILES.items():
        item = dict(path=str(OLD_ROOT / relative), identity=identity(failure / relative))
        references[key] = item
        dependencies[item['path']] = item['identity']
    return preparation, references, dependencies


def seal(args):
    code, old_sealed, failure, output = map(Path,
        (args.code, args.previous_sealed, args.failure, args.output))
    execution = original.published_source(code, args.commit)
    execution['source_path'] = str(REMOTE_CODE)
    previous = json.loads((old_sealed / 'PLAN.json').read_text())
    previous_seal = json.loads((old_sealed / 'SEAL.json').read_text())
    # Authenticate the original sealed metadata locally before copying it.
    for name, expected in previous_seal['files'].items():
        path = Path(name)
        if path.is_relative_to(OLD_ROOT / 'sealed'):
            local = old_sealed / path.relative_to(OLD_ROOT / 'sealed')
            if identity(local) != expected:
                raise ValueError('Original sealed qualification metadata changed: ' + str(local))
    consumed = validate_failed_boundary(failure, previous)
    payload_identity = json.loads(Path(args.disposable_identity).read_text())
    preparation, reference, files = retained_dependencies(failure, previous, payload_identity, old_sealed)
    if previous['numerical_recipe'] != original.numerical_recipe():
        raise ValueError('Recovery may not change the fixed numerical recipe')
    output.mkdir(parents=True, exist_ok=False)
    sealed = output / 'sealed'
    shutil.copytree(old_sealed, sealed, ignore=shutil.ignore_patterns('PLAN.json', 'SEAL.json'))
    feedback = output / 'qualification_feedback'
    feedback.mkdir()
    shutil.copy2(failure / 'qualification_feedback/PREPARATION.json', feedback / 'PREPARATION.json')
    shutil.copy2(failure / 'QUALIFICATION_WORK.jsonl', sealed / 'PRIOR_QUALIFICATION_WORK.jsonl')
    shutil.copy2(args.disposable_identity, sealed / 'DISPOSABLE_IDENTITY.json')
    shutil.copy2(old_sealed / 'PLAN.json', sealed / 'PREVIOUS_PLAN.json')
    shutil.copy2(old_sealed / 'SEAL.json', sealed / 'PREVIOUS_SEAL.json')
    plan = deepcopy(previous)
    plan.update(root=str(REMOTE_ROOT), execution_source=execution,
        scope='bounded_qualification_repair_only', main_pilot_authorized=False,
        automatic_retry=False, original_config=str(REMOTE_ROOT / 'sealed/ORIGINAL_CONFIG.yaml'))
    # The scientific binding remains the original experiment's binding. The new
    # code identity and narrow repair authorization are recorded independently.
    plan['repair'] = dict(previous_root=str(OLD_ROOT), previous_execution_commit=FAILED_COMMIT,
        prior_work_ledger=dict(path=str(OLD_ROOT / 'QUALIFICATION_WORK.jsonl'),
            identity=identity(failure / 'QUALIFICATION_WORK.jsonl')),
        shared_preparation=dict(path=str(OLD_ROOT / 'qualification_feedback/PREPARATION.json'),
            identity=identity(failure / 'qualification_feedback/PREPARATION.json')),
        retained_reference=reference, prior_work=consumed, remaining_adam_steps=7,
        new_feedback_generations=0, new_teacher_preparations=0,
        reason='optimizer checkpoint tensor ownership repair; no algorithm or tolerance change',
        cumulative_work_limits=previous['numerical_recipe']['qualification_work_limits'],
        original_failure_preserved=True, initialized_work_ledger='copy authenticated prior bytes at runner startup')
    for dataset, item in plan['datasets'].items():
        for key in ('training_manifest', 'panel_manifest', 'references'):
            item[key] = str(REMOTE_ROOT / Path(item[key]).relative_to(OLD_ROOT))
        for arm, old_path in item['configs'].items():
            relative = Path(old_path).relative_to(OLD_ROOT)
            local = output / relative
            raw = yaml.safe_load(local.read_text())
            raw['output_dir'] = str(REMOTE_ROOT / 'qualification' / dataset / 'trainer')
            text = yaml.safe_dump(raw, sort_keys=False)
            validate_config_text(text, (sealed / 'ORIGINAL_CONFIG.yaml').read_text())
            local.write_text(text)
            item['configs'][arm] = str(REMOTE_ROOT / relative)
    write_json(sealed / 'PLAN.json', plan)
    for key in ('prior_work_ledger', 'shared_preparation'):
        item = plan['repair'][key]
        files[item['path']] = item['identity']
    for name in ('PLAN.json', 'SEAL.json'):
        files[str(OLD_ROOT / 'sealed' / name)] = identity(old_sealed / name)
    for encoded in subprocess.check_output(['git', 'ls-files', '-z', '.'], cwd=code).split(b'\0'):
        if encoded:
            name = encoded.decode()
            if (code / name).is_file():
                files[str(REMOTE_CODE / name)] = identity(code / name)
    for path in sorted(output.rglob('*')):
        if path.is_file():
            files[str(REMOTE_ROOT / path.relative_to(output))] = identity(path)
    write_json(sealed / 'SEAL.json', dict(execution_source=execution, files=files,
        scope=plan['scope'], previous_execution_commit=FAILED_COMMIT,
        cumulative_work_limits=plan['repair']['cumulative_work_limits']))
    return dict(status='sealed_remaining_qualification_before_model_work',
        execution_source=execution, binding=plan['binding'], files=len(files),
        previous_adam_steps=1, remaining_adam_steps=7, new_feedback_generations=0,
        new_teacher_preparations=0, main_pilot_authorized=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--code', default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument('--previous-sealed', default=str(original.STUDY / 'qualification_001/sealed'))
    parser.add_argument('--failure', default=str(original.STUDY / 'failure_001'))
    parser.add_argument('--disposable-identity', default=str(original.STUDY / 'DISPOSABLE_IDENTITY.json'))
    parser.add_argument('--output', default=str(original.STUDY / 'qualification_002'))
    print(json.dumps(seal(parser.parse_args()), indent=2))


if __name__ == '__main__':
    main()
