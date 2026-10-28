"""Install the KC event integration against the tested correction and K foundation."""
import hashlib
import json
from pathlib import Path
import shutil
import sys


def identity(path):
    path = Path(path)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(bytes=path.stat().st_size, sha256=digest)


def prepare(root):
    root = Path(root)
    prior = root.parent / 'dflash_prefix_correction_step_20261027'
    event_prior = root.parent / 'dflash_prefix_preservation_guard_20261027/main_003'
    seal = json.loads((prior / 'sealed/SEAL.json').read_text())
    for path, expected in seal['files'].items():
        if identity(path) != expected:
            raise ValueError('Qualified source or original input changed: ' + path)
    if (prior / 'EXIT_CODE').read_text().strip() != '0':
        raise ValueError('The prior reference run did not complete successfully')
    source_root = Path('/data/sources/minhao/DFlash-Online')
    additions = json.loads((root / 'ADDITIONS.json').read_text())
    for name, expected in additions.items():
        source, destination = root / 'code' / name, source_root / name
        if identity(source) != expected or destination.exists():
            raise ValueError('Changed transfer or non-additive install: ' + name)
    for name in additions:
        destination = source_root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / 'code' / name, destination)
    sealed = root / 'sealed'
    sealed.mkdir(exist_ok=False)
    for name in ('config.yaml', 'MODEL_PIN.json', 'RUNTIME_PIN.json', 'TRAINING_PIN.json'):
        shutil.copy2(prior / 'sealed' / name, sealed / name)
    old = json.loads((prior / 'sealed/PLAN.json').read_text())
    event_old = json.loads((event_prior / 'sealed/PLAN.json').read_text())
    plan = {key: old[key] for key in ('ordinary_receipt','pinned_initial','target','draft','worker_python')}
    plan.update(config=str(sealed / 'config.yaml'),
        manifest=event_old['manifest'], k_receipt=event_old['k_receipt'],
        reference_diagnostic=str(prior/'execution_001/RESULT.json'),
        scope='one KC event with tested correction and fresh K1/KC paired development evaluation',
        main=dict(optimizer_steps=128, presentations=256, maximum_direct_corrections=128,
                  response_forwards=518, maximum_backwards=512),
        probes=dict(steps=[1,64,128],corrected_response_forwards=6),
        startup=dict(response_evaluations=2, backwards=2, optimizer_steps=0, restoration='exact'),
        evaluation=dict(generations=384,arms=['K','KC'],requests=64,repeats=3),
        qualification_optimizer_steps=0, final_checkpoint_count=1,
        no_automatic_retry=True, numerics=json.loads((root/'NUMERICS.json').read_text()))
    (sealed / 'PLAN.json').write_text(json.dumps(plan, indent=2) + '\n')
    paths = list(sealed.iterdir()) + [Path(plan['reference_diagnostic'])]
    paths += [source_root / name for name in additions]
    paths += [p for p in root.iterdir() if p.is_file() and p.suffix in ('.py','.sh','.json','.md')]
    paths += [prior/'sealed/SEAL.json', prior/'POSTFLIGHT.json', prior/'EXIT_CODE']
    for path in paths:
        seal['files'][str(path)] = identity(path)
    seal['parent_seal'] = str(prior / 'sealed/SEAL.json')
    seal['scope'] = plan['scope']
    (sealed / 'SEAL.json').write_text(json.dumps(seal, indent=2) + '\n')
    print(json.dumps(dict(status='sealed', source_changes=0, new_files=list(additions), files=len(seal['files']))))


if __name__ == '__main__':
    prepare(sys.argv[1])
