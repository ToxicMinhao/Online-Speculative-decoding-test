"""Seal the reviewed numerical-test revision without changing the algorithm."""
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
from . import seal as original
from .resume_qualification import POLICY, PRIOR_COUNTS

REMOTE_ROOT = Path('/data/sources/minhao/runs/dflash_functional_step_geometry_20261108/qualification_003')
REMOTE_CODE = Path('/data/sources/minhao/fsg_source_20261109_revised/DFlash-Online')
STUDY = Path('/home/minhao2/EAGLE_SPEC/research/dflash_functional_step_geometry_20261108')


def remaining_work_plan():
    return dict(adam=5, ordinary_forward=10, ordinary_backward=10,
        additional_backward=20, additional_forward_maximum=60, serving_generation=8,
        additional_backward_parts=dict(gsm_ordinary_replay_and_startup=2, math_startup=2,
            remaining_filter_hooks=8, independent_native_oracles=4, expanded_resource=4),
        resource_cases=1, resource_anchor_counts=[512,512], resource_sequence_length=2272,
        independent_fixture_anchors=9, completed_adam_steps_repeated=0)


def seal(code, commit, output):
    code, output = Path(code), Path(output)
    execution = original.published_source(code, commit)
    execution['source_path'] = str(REMOTE_CODE)
    previous_dir = STUDY/'qualification_002/sealed'
    previous = json.loads((previous_dir/'PLAN.json').read_text())
    old_root = Path(previous['root'])
    failure = STUDY/'failure_002'
    if json.loads((failure/'qualification/QUALIFICATION.json').read_text())['work'] != PRIOR_COUNTS:
        raise ValueError('The retained qualification work differs')
    output.mkdir(parents=True, exist_ok=False)
    sealed = output/'sealed'
    shutil.copytree(previous_dir, sealed, ignore=shutil.ignore_patterns('PLAN.json','SEAL.json'))
    plan = deepcopy(previous)
    plan.update(root=str(REMOTE_ROOT), execution_source=execution,
        original_config=str(REMOTE_ROOT/'sealed/ORIGINAL_CONFIG.yaml'),
        scope='resume_revised_qualification', main_pilot_authorized=True,
        main_pilot_condition='all checks pass, representative nontrivial accepted correction, storage and integration')
    retained_step = old_root/'qualification/gsm8k/FSG/geometry/step_000513.json'
    consequence_root = old_root.parent/'bf16_consequences_001'
    separate = [dict(name=name, report=str(old_root.parent/subdir), included_in_original_qualification=False)
        for name,subdir in [('native_head_diagnosis','diagnostic_001'),
            ('auxiliary_fp32_head_diagnosis','auxiliary_head_precision_001'),
            ('bf16_consequence_diagnosis','bf16_consequences_001')]]
    plan['revised_qualification'] = dict(policy=POLICY, previous_root=str(old_root),
        prior_work=PRIOR_COUNTS, remaining=remaining_work_plan(),
        retained_fsg_step=str(retained_step), consequence_root=str(consequence_root),
        separate_diagnostics=separate, production_precision_changed=False,
        algorithm_changed=False, tolerance_changed=False, scientific_test_comparison_changed=True)
    for dataset,item in plan['datasets'].items():
        for key in ('training_manifest','panel_manifest','references'):
            item[key] = str(REMOTE_ROOT/Path(item[key]).relative_to(old_root))
        for arm,old_path in item['configs'].items():
            relative = Path(old_path).relative_to(old_root)
            local = output/relative
            config = yaml.safe_load(local.read_text())
            config['output_dir'] = str(REMOTE_ROOT/'qualification'/dataset/'trainer')
            text = yaml.safe_dump(config,sort_keys=False)
            validate_config_text(text,(sealed/'ORIGINAL_CONFIG.yaml').read_text())
            local.write_text(text)
            item['configs'][arm] = str(REMOTE_ROOT/relative)
    feedback = output/'qualification_feedback'; feedback.mkdir()
    shutil.copy2(failure/'qualification_feedback/PREPARATION.json',feedback/'PREPARATION.json')
    shutil.copy2(failure/'QUALIFICATION_WORK.jsonl',sealed/'PRIOR_REVISED_WORK.jsonl')
    shutil.copy2(STUDY/'revised_qualification_001/AUTHORIZATION.json',sealed/'AUTHORIZATION.json')
    write_json(sealed/'PLAN.json',plan)
    files = {}
    old_seal = json.loads((previous_dir/'SEAL.json').read_text())
    # Keep immutable model, origin, runtime and previous-repair evidence. Old
    # source mirrors are provenance, while execution binds only the new mirror.
    old_code = Path(previous['execution_source']['source_path'])
    for name,expected in old_seal['files'].items():
        if not Path(name).is_relative_to(old_root) and not Path(name).is_relative_to(old_code):
            files[name] = expected
    for local in sorted(failure.rglob('*')):
        if local.is_file():
            files[str(old_root/local.relative_to(failure))] = identity(local)
    neutral = json.loads((failure/'qualification/gsm8k/FSG_ZERO/RESULT.json').read_text())
    for name, expected in neutral['export_verification']['files'].items():
        files[str(Path(neutral['export'])/name)] = expected
    for local in sorted((STUDY/'bf16_consequences_001/retained/bf16_consequences_001').rglob('*')):
        if local.is_file():
            files[str(consequence_root/local.relative_to(STUDY/'bf16_consequences_001/retained/bf16_consequences_001'))] = identity(local)
    for encoded in subprocess.check_output(['git','ls-files','-z','.'],cwd=code).split(b'\0'):
        if encoded:
            name=encoded.decode()
            if (code/name).is_file():files[str(REMOTE_CODE/name)]=identity(code/name)
    for local in sorted(output.rglob('*')):
        if local.is_file():files[str(REMOTE_ROOT/local.relative_to(output))]=identity(local)
    write_json(sealed/'SEAL.json',dict(execution_source=execution,files=files,
        scope=plan['scope'],policy=POLICY,cumulative_work_limits=plan['qualification_budget']))
    return dict(status='sealed',execution_source=execution,files=len(files),
                remaining=remaining_work_plan(),conditional_pilot_separately_admitted=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--commit',required=True)
    p.add_argument('--code',default=str(Path(__file__).resolve().parents[2]))
    p.add_argument('--output',default=str(STUDY/'qualification_003'))
    a=p.parse_args();print(json.dumps(seal(a.code,a.commit,a.output),indent=2))
