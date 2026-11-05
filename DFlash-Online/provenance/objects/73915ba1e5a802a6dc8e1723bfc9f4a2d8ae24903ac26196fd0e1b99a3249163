"""Install additive KRC files and seal actual retained-response reference bindings."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

SOURCE_ROOT = Path('/data/sources/minhao/DFlash-Online')


def identity(path):
    path = Path(path)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(bytes=path.stat().st_size, sha256=digest)


def prepare(root):
    root = Path(root)
    prior = root.parent / 'dflash_kc_event_20261027'
    seal = json.loads((prior / 'sealed/SEAL.json').read_text())
    for path, expected in seal['files'].items():
        if identity(path) != expected:
            raise ValueError('Qualified source or retained input changed: ' + path)
    if (prior / 'EXIT_CODE').read_text().strip() != '0':
        raise ValueError('The prior KC event did not complete successfully')

    additions = json.loads((root / 'ADDITIONS.json').read_text())
    # Check the complete transfer and every destination before making any copy.
    for name, expected in additions.items():
        source, destination = root / 'code' / name, SOURCE_ROOT / name
        if identity(source) != expected or destination.exists():
            raise ValueError('Changed transfer or non-additive install: ' + name)
    for name in additions:
        destination = SOURCE_ROOT / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / 'code' / name, destination)

    sys.path.insert(0, str(SOURCE_ROOT))
    from online_feedback.reference_schedule import build_reference_schedule, validate_reference_schedule

    sealed = root / 'sealed'
    sealed.mkdir(exist_ok=False)
    for name in ('config.yaml', 'MODEL_PIN.json', 'RUNTIME_PIN.json', 'TRAINING_PIN.json'):
        shutil.copy2(prior / 'sealed' / name, sealed / name)
    old = json.loads((prior / 'sealed/PLAN.json').read_text())
    ordinary = json.loads(Path(old['ordinary_receipt']).read_text())
    manifest = json.loads(Path(old['manifest']).read_text())
    evaluation_ids = [row['request_id'] for row in manifest['eval_requests']]
    schedule = build_reference_schedule(ordinary, evaluation_ids=evaluation_ids)
    schedule_check = validate_reference_schedule(schedule, ordinary, evaluation_ids=evaluation_ids)
    schedule_path = sealed / 'REFERENCE_SCHEDULE.json'
    schedule_path.write_text(json.dumps(schedule, indent=2) + '\n')
    (sealed / 'REFERENCE_SCHEDULE_CHECK.json').write_text(json.dumps(schedule_check, indent=2) + '\n')
    plan = {key: old[key] for key in ('ordinary_receipt', 'pinned_initial', 'target', 'draft',
                                     'worker_python', 'manifest', 'k_receipt', 'reference_diagnostic')}
    plan.update(config=str(sealed / 'config.yaml'), reference_schedule=str(schedule_path),
        scope='one KRC event with next-pair reference correction and fresh K1/KRC development evaluation',
        main=dict(optimizer_steps=128, presentations=256, maximum_direct_corrections=128,
                  response_forwards=786, maximum_backwards=512),
        references=dict(selection='next_disjoint_pair_cyclic_within_pass', windows=128,
                        response_presentations=256, exposure='additional_training_exposure'),
        probes=dict(steps=[1,64,128], diagnostic_response_forwards=18,
                    learning_post_K=6, learning_post_correction=6, reference_post_correction=6),
        startup=dict(response_evaluations=2, backwards=2, optimizer_steps=0, restoration='exact'),
        evaluation=dict(generations=384, arms=['K','KRC'], requests=64, repeats=3),
        qualification_optimizer_steps=0, final_checkpoint_count=1,
        no_automatic_retry=True, numerics=json.loads((root/'NUMERICS.json').read_text()))
    (sealed / 'PLAN.json').write_text(json.dumps(plan, indent=2) + '\n')
    paths = list(sealed.iterdir()) + [Path(plan['reference_diagnostic'])]
    paths += [SOURCE_ROOT / name for name in additions]
    paths += [p for p in root.iterdir() if p.is_file() and p.suffix in ('.py','.sh','.json','.md')]
    paths += [prior/'sealed/SEAL.json', prior/'POSTFLIGHT.json', prior/'EXIT_CODE']
    for path in paths:
        seal['files'][str(path)] = identity(path)
    seal['parent_seal'] = str(prior / 'sealed/SEAL.json')
    seal['scope'] = plan['scope']
    (sealed / 'SEAL.json').write_text(json.dumps(seal, indent=2) + '\n')
    print(json.dumps(dict(status='sealed', source_changes=0, new_files=list(additions),
                         files=len(seal['files']), reference_schedule_check=schedule_check)))


if __name__ == '__main__':
    prepare(sys.argv[1])
