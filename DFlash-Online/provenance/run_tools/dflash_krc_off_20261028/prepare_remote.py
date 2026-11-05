"""Bind the correction-removal control to unchanged KRC1 and event-2 evidence."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

SOURCE = Path('/data/sources/minhao/DFlash-Online')
RUNS = Path('/data/sources/minhao/runs')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def identity(path):
    path = Path(path)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(bytes=path.stat().st_size, sha256=digest)


def retained_counts(path, method, requests):
    records = [json.loads(line) for line in Path(path).read_text().splitlines()]
    result = {}
    for request in requests:
        rid = request['request_id']
        rows = [row for row in records if row['request_id'] == rid and row['method'] == method]
        if (len(rows) != 3 or {row['repeat'] for row in rows} != {0, 1, 2}
                or not all(row['valid_measurement'] for row in rows)
                or len({row['verification_calls'] for row in rows}) != 1):
            raise ValueError(f'Invalid retained {method} records for {rid}')
        result[rid] = rows[0]['verification_calls']
    return result


def prepare(root):
    root = Path(root)
    parent = RUNS / 'dflash_krc_continuation_20261028'
    if (parent / 'EXIT_CODE').read_text().strip() != '0':
        raise ValueError('The retained KRC2 campaign must be complete')
    old = read(parent / 'sealed/PLAN.json')
    parent_seal = read(parent / 'sealed/SEAL.json')
    paths = set(map(Path, parent_seal['files']))
    for path, expected in parent_seal['files'].items():
        if identity(path) != expected:
            raise ValueError('Retained KRC2 dependency changed: ' + path)
    paths.update([parent / 'sealed/SEAL.json', parent / 'POSTFLIGHT.json', parent / 'EXIT_CODE'])

    additions = read(root / 'ADDITIONS.json')
    for name, expected in additions.items():
        if identity(root / 'code' / name) != expected or (SOURCE / name).exists():
            raise ValueError('Require exact additive installation: ' + name)
    for name in additions:
        destination = SOURCE / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / 'code' / name, destination)
        paths.add(destination)
    sys.path.insert(0, str(SOURCE))
    from experiments.krc_off.evaluation_protocol import validate_plan

    krc2_path = parent / 'campaign_001/KRC2/events/event_002/RESULT.json'
    krc2 = read(krc2_path)
    ordinary = read(old['event2_ordinary_receipt'])
    if krc2['features'] != ordinary['features'] or len(krc2['features']) != 128:
        raise ValueError('Retained KRC2 and ordinary event-2 feature pools differ')
    cap = next(row for row in ordinary['features'] if row['record_id'] == 'train_00187')
    if cap['response_tokens'] != 2048 or cap['length'] != 2133:
        raise ValueError('The capped event-2 response must remain unchanged')

    panels = []
    for panel in old['panels']:
        measurements = parent / 'campaign_001/evaluation' / panel['name'] / 'MEASUREMENTS.jsonl'
        requests = panel['requests']
        panels.append(dict(name=panel['name'], requests=requests,
            expected_control_counts=retained_counts(measurements, 'KRC2', requests),
            historical_counts=dict(KRC1=panel['historical_counts']['KRC1'],
                                   K2=retained_counts(measurements, 'K2', requests)),
            provenance=dict(unchanged_panel=str(parent / 'sealed/PLAN.json'),
                canonical=panel['provenance']['canonical'],
                control_count_gate='retained per-request KRC2 counts',
                control_measurements=str(measurements))))
        paths.add(measurements)
    sealed = root / 'sealed'
    sealed.mkdir(exist_ok=False)
    for name in ('config.yaml', 'MODEL_PIN.json', 'RUNTIME_PIN.json', 'TRAINING_PIN.json'):
        shutil.copy2(parent / 'sealed' / name, sealed / name)
    # This is only the existing A presentation/anchor receipt. No reference
    # schedule or protected-prefix record is created for the OFF arm.
    shutil.copy2(old['event2_ordinary_receipt'], sealed / 'PRESENTATION_SCHEDULE.json')
    plan = {key: old[key] for key in ('startup_ordinary_receipt', 'pinned_initial',
        'krc1_state', 'krc1_receipt', 'krc1_checkpoint', 'event2_ordinary_receipt',
        'event2_k_receipt', 'target', 'draft', 'worker_python', 'retained_pool')}
    plan.update(config=str(sealed / 'config.yaml'), krc2_receipt=str(krc2_path),
        retained_krc2_checkpoint=krc2['checkpoint_path'],
        presentation_schedule=str(sealed / 'PRESENTATION_SCHEDULE.json'), panels=panels,
        historical_KRC2_costs=str(parent / 'campaign_001/COSTS.json'),
        scope='exact own KRC1 resume, correction removed during event 2 only',
        startup=dict(forwards=2, backwards=2, optimizer_steps=0),
        main=dict(start_step=128, end_step=256, windows=128, presentations=256,
                  optimizer_steps=128, core_forwards=256, core_backwards=256,
                  reference_operations=0, direct_corrections=0, diagnostic_forwards=0),
        evaluation=dict(generations=768, initial_gate_generations=48, repeats=3),
        capped_response_preserved=cap, no_new_feedback=True, no_automatic_retry=True)
    validate_plan(plan)
    write(sealed / 'PLAN.json', plan)
    paths.update([krc2_path, Path(plan['historical_KRC2_costs'])])
    paths.update(sealed.iterdir())
    paths.update(p for p in root.iterdir() if p.is_file() and p.suffix in {'.py', '.sh', '.json', '.md'})
    exact = dict(parent_seal['exact_file_sets'])
    checkpoint = Path(plan['retained_krc2_checkpoint'])
    files = [p for p in checkpoint.iterdir() if p.is_file()]
    exact[str(checkpoint)] = sorted(p.name for p in files)
    paths.update(files)
    write(sealed / 'SEAL.json', dict(schema='krc_off_seal_v1', sealed_before_training=True,
        sealed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        files={str(p): identity(p) for p in sorted(paths)}, exact_file_sets=exact,
        parent_seals=[str(parent / 'sealed/SEAL.json')]))
    print(json.dumps(dict(status='sealed', files=len(paths), retained_responses=128,
        retained_KRC2_calls={panel['name']: sum(panel['expected_control_counts'].values()) for panel in panels},
        reference_operations=0, capped_response_preserved=True)))


if __name__ == '__main__':
    prepare(sys.argv[1])
