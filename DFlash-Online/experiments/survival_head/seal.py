"""Seal actual paths and input-only panel before any pilot model execution."""
import argparse
from copy import deepcopy
from pathlib import Path
import shutil
import yaml

from online_feedback.storage import write_json
from online_feedback.presentation_schedule import PresentationSchedule
from experiments.ok_coding.storage import identity
from experiments.kplus_abt.seal import validate_config_text
from .contract import ARMS, ANALYSIS, SCHEMA, budgets
from .data import read, digest, select_panel, feedback_requests
from .evaluation import evaluation_schedule, validate_evaluation_inputs
from .storage import capacity_plan


def authenticate_source(source, code, *, verify_bytes=True):
    """Bind all pilot modules and its native training dependencies, not a subset."""
    code = Path(code).resolve()
    commit = source.get('commit', '')
    if (source.get('status') != 'verified' or source.get('dirty') is not False
            or len(commit) != 40 or any(value not in '0123456789abcdef' for value in commit)
            or Path(source['source_path']).resolve() != code or not source.get('files')):
        raise ValueError('Require authenticated clean commit-derived source')
    required = {str(path.relative_to(code)) for path in (code/'experiments/survival_head').glob('*.py')}
    required.update(('online_feedback/survival_head.py', 'online_feedback/survival_head_runtime.py',
        'online_feedback/survival_feedback_worker.py', 'online_feedback/objective.py',
        'online_feedback/training.py', 'online_feedback/optimizer.py',
        'training/specforge/training/controller.py', 'training/specforge/core/chunking.py'))
    if not required.issubset(source['files']):
        raise ValueError('Source receipt omits executing pilot or native training modules')
    for relative, expected in source['files'].items():
        path = Path(relative)
        if path.is_absolute() or '..' in path.parts or str(path) != relative:
            raise ValueError('Source receipt paths must be normalized relative paths')
        if verify_bytes and identity(code/path) != expected:
            raise ValueError('Staged source differs: '+relative)


def seal(root, prior_root, source_receipt):
    root, prior = Path(root).resolve(), Path(prior_root).resolve()
    source = read(source_receipt)
    code = Path(source['source_path']).resolve()
    if code != Path(__file__).resolve().parents[2]:
        raise ValueError('Execute the staged source being sealed')
    authenticate_source(source, code)
    old = read(prior/'sealed/PLAN.json'); old_seal = read(prior/'sealed/SEAL.json')
    recorded_root = Path(old['root'])
    recorded_plan = str(recorded_root/'sealed/PLAN.json')
    if (old_seal.get('binding') != old['binding']
            or old_seal.get('execution_source') != old['execution_source']
            or identity(prior/'sealed/PLAN.json') != old_seal['files'].get(recorded_plan)):
        raise ValueError('Retained prior plan/origin/runtime metadata changed')
    original = old['datasets']['gsm8k']
    selected_inputs = ['ORIGINAL_CONFIG.yaml', 'MODEL_PIN.json', 'RUNTIME_PIN.json', 'TRAINING_PIN.json',
                       'gsm8k/TRAINING.json', 'gsm8k/PANEL.json', 'gsm8k/CANONICALS.json']
    inputs = {}
    for relative in selected_inputs:
        path = prior/'sealed'/relative
        expected = old_seal['files'][str(recorded_root/'sealed'/relative)]
        if identity(path) != expected: raise ValueError('Retained scientific input changed')
        inputs[str(path)] = expected
    pool_path = prior/'campaign_001/gsm8k/event_005/shared/POOL.json'
    schedule_path = prior/'campaign_001/gsm8k/event_005/KPLUS/RESULT.json'
    pool = read(pool_path); schedule = read(schedule_path)
    requests = feedback_requests(old)
    if (schedule['status'] != 'completed' or schedule['method'] != 'KPLUS'
            or schedule.get('dataset') != 'gsm8k' or schedule.get('event_id') != 'event_005'
            or schedule['start_step'] != 512 or schedule['end_step'] != 640
            or schedule.get('optimizer_steps') != 128
            or pool.get('dataset') != 'gsm8k' or pool.get('event') != 5
            or pool.get('producer') != 'KPLUS4'
            or [r['record_id'] for r in pool['records']] != [r['request_id'] for r in requests]):
        raise ValueError('Require original complete GSM8K event-five pool and schedule')
    PresentationSchedule(schedule, [row['request_id'] for row in requests], 2)
    for row in pool['records']:
        canonical = row['canonical']
        if identity(canonical['path']) != {key:canonical[key] for key in ('bytes','sha256')}:
            raise ValueError('Original canonical packet bytes changed before sealing')
        record = read(canonical['path'])
        if record['record_id'] != row['record_id'] or len(record['prompt_ids']) + len(record['response_ids']) > 2272:
            raise ValueError('Original packet identity or measured storage length bound differs')
    feature_sizes = [row['feature']['bytes'] for row in pool['records']]
    if any(type(value) is not int or value <= 0 for value in feature_sizes):
        raise ValueError('Original pool must retain every exact positive packet size')
    panel = select_panel(read(prior/'sealed/gsm8k/PANEL.json'), requests)
    binding = digest(dict(schema=SCHEMA, source_commit=source['commit'],
        inputs=inputs, panel=panel['binding'], prior_pool=identity(pool_path),
        original_schedule=identity(schedule_path), origin={k:v for k,v in original.items() if k.startswith('origin_')},
        budgets=budgets(), analysis=ANALYSIS))
    sealed = root/'sealed'; sealed.mkdir(parents=True, exist_ok=False)
    for name in selected_inputs[:4]: shutil.copy2(prior/'sealed'/name, sealed/name)
    shutil.copy2(prior/'sealed/gsm8k/TRAINING.json', sealed/'TRAINING.json')
    shutil.copy2(prior/'sealed/gsm8k/PANEL.json', sealed/'SOURCE_PANEL.json')
    write_json(sealed/'PANEL.json', panel)
    canonical = read(prior/'sealed/gsm8k/CANONICALS.json')
    canonical = {r['request_id']:deepcopy(canonical[r['request_id']]) for r in panel['requests']}
    for row in canonical.values(): row['binding_digest'] = binding
    write_json(sealed/'CANONICAL.json', canonical)
    execution = {k:source[k] for k in ('commit', 'dirty', 'repository', 'source_path')}
    plan = dict(schema=SCHEMA, root=str(root), binding=binding, execution_source=execution,
        arms=list(ARMS), dataset='gsm8k', event=5, budgets=budgets(), analysis=deepcopy(ANALYSIS),
        automatic_retry=False, priority_root=str(prior), prior_pool=str(pool_path),
        prior_pool_identity=identity(pool_path), original_schedule=str(schedule_path),
        original_schedule_identity=identity(schedule_path),
        retained_packet_paths=[r['feature']['path'] for r in pool['records'][:2]],
        original_config=str(sealed/'ORIGINAL_CONFIG.yaml'), training_manifest=str(sealed/'TRAINING.json'),
        panel_manifest=str(sealed/'PANEL.json'), references=str(sealed/'CANONICAL.json'),
        evaluation_schedule=evaluation_schedule(panel['requests']), configs={},
        training_gpu=1, serving_gpu=0,
        training_gpu_identity='GPU-c6943f33-bcd0-a724-0b35-f77af6b0a3f1',
        serving_gpu_identity='GPU-9986735a-3e68-d1bd-5998-9634ef24341e',
        main_storage=capacity_plan(original['origin_weights_identity']['bytes'], original['origin_state_identity']['bytes'],
            feature_bytes=sum(feature_sizes), max_feature_bytes=max(feature_sizes)),
        author_experiment_exception='User explicitly permits this survival-head pilot after VAT source review; no VAT reproduction claim',
        qualification_admission=str(root/'qualification/ADMISSION.json'))
    plan.update({k:old[k] for k in ('target','draft','worker_python','training_python')})
    plan.update({k:deepcopy(v) for k,v in original.items() if k.startswith('origin_')})
    original_text = (sealed/'ORIGINAL_CONFIG.yaml').read_text()
    for arm in ARMS:
        config = yaml.safe_load(original_text)
        config.update(run_id='survival_head_'+arm, output_dir=str(root/'campaign_001/gsm8k/event_005'/arm))
        text = yaml.safe_dump(config, sort_keys=False); validate_config_text(text, original_text)
        path = sealed/f'config_{arm}.yaml'; path.write_text(text); plan['configs'][arm] = str(path)
    plan['datasets'] = {'gsm8k':{k:plan[k] for k in ('origin_state','origin_export','origin_weights',
        'origin_state_identity','origin_weights_identity','training_manifest','panel_manifest','references','configs')}}
    validate_evaluation_inputs(panel['requests'], {a:'/future/'+a for a in ARMS}, canonical,
        dataset='gsm8k', event=5, schedule=plan['evaluation_schedule'])
    write_json(sealed/'PLAN.json', plan)
    bound_files = {str(code/p):v for p,v in source['files'].items()}
    bound_files.update({str(p):identity(p) for p in sealed.rglob('*') if p.is_file()})
    bound_files.update({str(pool_path):identity(pool_path), str(schedule_path):identity(schedule_path)})
    write_json(sealed/'SEAL.json', dict(binding=binding, execution_source=execution, files=bound_files,
        inherited_inputs=inputs, selection_before_model_execution=True))
    return plan


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root'); parser.add_argument('--prior', required=True)
    parser.add_argument('--source-receipt', required=True)
    args = parser.parse_args(); seal(args.root, args.prior, args.source_receipt)
