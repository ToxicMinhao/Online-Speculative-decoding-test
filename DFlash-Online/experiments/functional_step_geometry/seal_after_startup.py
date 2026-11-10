"""Seal the remaining qualification, preserving failed work and passed checks."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess

import yaml

from experiments.ok_coding.storage import identity
from experiments.role_loss_pilot.training import validate_config_text
from online_feedback.storage import write_json
from .seal import published_source
from .qualify import REFERENCE_STARTUP
from .resume_after_startup import PRIOR_COUNTS, CUMULATIVE_LIMITS, remaining_work_plan

STUDY = Path('/home/minhao2/EAGLE_SPEC/research/dflash_functional_step_geometry_20261108')
REMOTE_ROOT = Path('/data/sources/minhao/runs/dflash_functional_step_geometry_20261108/qualification_004')
REMOTE_CODE = Path('/data/sources/minhao/fsg_source_20261109_startup_repaired/DFlash-Online')


def seal(code, commit, output):
    code, output = Path(code), Path(output)
    execution = published_source(code, commit)
    execution['source_path'] = str(REMOTE_CODE)
    retained = STUDY/'revised_qualification_001/retained'
    previous = json.loads((retained/'sealed/PLAN.json').read_text())
    old_root = Path(previous['root'])
    previous_result = json.loads((retained/'qualification/QUALIFICATION.json').read_text())
    if previous_result['status'] != 'failed' or previous_result['work'] != PRIOR_COUNTS:
        raise ValueError('Require the latest failed qualification, including its consumed work')
    previous_sealed = STUDY/'qualification_003/sealed'
    old_seal = json.loads((retained/'sealed/SEAL.json').read_text())
    for local in previous_sealed.rglob('*'):
        if local.is_file() and local.name != 'SEAL.json':
            expected = old_seal['files'][str(old_root/'sealed'/local.relative_to(previous_sealed))]
            if identity(local) != expected:
                raise ValueError(f'Previous sealed input differs: {local}')
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(previous_sealed, output/'sealed',
                    ignore=shutil.ignore_patterns('PLAN.json', 'SEAL.json'))
    plan = deepcopy(previous)
    plan.update(root=str(REMOTE_ROOT), execution_source=execution,
        original_config=str(REMOTE_ROOT/'sealed/ORIGINAL_CONFIG.yaml'),
        scope='remaining_qualification_after_startup_repair', main_pilot_authorized=False,
        main_pilot_condition='not launched by this qualification-only entry point')
    plan['original_qualification_budget'] = deepcopy(previous['qualification_budget'])
    plan['qualification_budget'] = dict(optimizer_steps=CUMULATIVE_LIMITS['adam'],
        ordinary_presentations=CUMULATIVE_LIMITS['ordinary_forward'],
        additional_response_forwards=CUMULATIVE_LIMITS['additional_forward'],
        additional_backwards=CUMULATIVE_LIMITS['additional_backward'],
        serving_generations=CUMULATIVE_LIMITS['serving_generation'])
    plan['revised_qualification']['reference_startup'] = deepcopy(REFERENCE_STARTUP)
    plan['startup_repair'] = dict(previous_root=str(old_root), prior_counts=PRIOR_COUNTS,
        previous_ledger_identity=identity(retained/'QUALIFICATION_WORK.jsonl'),
        cumulative_limits=CUMULATIVE_LIMITS, remaining_work=remaining_work_plan(),
        reference_startup=REFERENCE_STARTUP,
        retained_oracles=str(old_root/'qualification/gsm8k/revised_oracles'),
        completed_checks_reused=True, completed_adam_steps_repeated=0,
        algorithm_changed=False, tolerance_changed=False)
    for name in ('ordinary_gradient_debug_001', 'ordinary_gradient_debug_002'):
        plan['revised_qualification']['separate_diagnostics'].append(dict(name=name,
            report=str(old_root.parent/name), included_in_original_qualification=False))
    for dataset, item in plan['datasets'].items():
        for key in ('training_manifest', 'panel_manifest', 'references'):
            item[key] = str(REMOTE_ROOT/Path(item[key]).relative_to(old_root))
        for arm, old_path in item['configs'].items():
            relative = Path(old_path).relative_to(old_root)
            local = output/relative
            cfg = yaml.safe_load(local.read_text())
            # Only the disposable output directory changes in each config.
            cfg['output_dir'] = str(REMOTE_ROOT/'qualification'/dataset/'trainer')
            text = yaml.safe_dump(cfg, sort_keys=False)
            validate_config_text(text, (output/'sealed/ORIGINAL_CONFIG.yaml').read_text())
            local.write_text(text)
            item['configs'][arm] = str(REMOTE_ROOT/relative)
    (output/'qualification_feedback').mkdir()
    shutil.copy2(retained/'qualification_feedback/PREPARATION.json',
                 output/'qualification_feedback/PREPARATION.json')
    shutil.copy2(retained/'QUALIFICATION_WORK.jsonl', output/'sealed/PRIOR_STARTUP_WORK.jsonl')
    shutil.copy2(STUDY/'qualification_resume_001/AUTHORIZATION.json', output/'sealed/AUTHORIZATION.json')
    write_json(output/'sealed/PLAN.json', plan)

    old_seal = json.loads((retained/'sealed/SEAL.json').read_text())
    files = dict(old_seal['files'])
    for local in sorted(retained.rglob('*')):
        if local.is_file():
            files[str(old_root/local.relative_to(retained))] = identity(local)
    debug = STUDY/'ordinary_gradient_debug_002/retained'
    for name in ('ordinary_gradient_debug_001', 'ordinary_gradient_debug_002'):
        for local in sorted((debug/name).rglob('*')):
            if local.is_file():
                files[str(old_root.parent/name/local.relative_to(debug/name))] = identity(local)
    for encoded in subprocess.check_output(['git', 'ls-files', '-z', '.'], cwd=code).split(b'\0'):
        if encoded:
            name = encoded.decode()
            if (code/name).is_file():
                files[str(REMOTE_CODE/name)] = identity(code/name)
    for local in sorted(output.rglob('*')):
        if local.is_file():
            files[str(REMOTE_ROOT/local.relative_to(output))] = identity(local)
    write_json(output/'sealed/SEAL.json', dict(execution_source=execution, files=files,
        scope=plan['scope'], cumulative_work_limits=CUMULATIVE_LIMITS,
        original_work_limits=plan['original_qualification_budget']))
    return dict(status='sealed', source=execution, root=str(REMOTE_ROOT), files=len(files),
                remaining_work=remaining_work_plan(), main_pilot_authorized=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code', default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument('--commit', required=True)
    parser.add_argument('--output', default=str(STUDY/'qualification_resume_001/sealed_package'))
    args = parser.parse_args()
    print(json.dumps(seal(args.code, args.commit, args.output), indent=2))
