"""Seal the retained event-2 pool, own KRC recovery and two unchanged panels."""
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


def counts(path, method, requests):
    records = [json.loads(line) for line in Path(path).read_text().splitlines()]
    result = {}
    for request in requests:
        rid = request['request_id']
        rows = [r for r in records if r['request_id'] == rid and r['method'] == method]
        if (len(rows) != 3 or {r['repeat'] for r in rows} != {0, 1, 2}
                or not all(r['valid_measurement'] for r in rows)
                or len({r['verification_calls'] for r in rows}) != 1):
            raise ValueError(f'Invalid retained {method} measurements for {rid}')
        result[rid] = rows[0]['verification_calls']
    return result


def prepare(root):
    root = Path(root)
    prior = RUNS/'dflash_krc_event_20261027'
    continuation = RUNS/'dflash_ok_continuation_20261027'
    confirmation = RUNS/'dflash_krc_confirmation_20261028'
    replay = RUNS/'dflash_reference_replay_20261028'
    paths = set()
    parents = [prior, continuation, confirmation, replay]
    current_files = read(prior/'sealed/SEAL.json')['files']
    historical_source_bindings = []
    historical_paths = {'online_feedback/objective.py', 'online_feedback/training.py'}
    for parent in parents:
        if (parent/'EXIT_CODE').read_text().strip() != '0':
            raise ValueError('Retained run incomplete: ' + str(parent))
        seal_path = parent/'sealed/SEAL.json'
        for name, expected in read(seal_path)['files'].items():
            actual = identity(name)
            if (actual != expected and parent == continuation
                    and Path(name).is_relative_to(SOURCE)
                    and str(Path(name).relative_to(SOURCE)) in historical_paths):
                archived = root/'historical_ok_code'/Path(name).relative_to(SOURCE)
                if identity(archived) != expected or actual != current_files.get(name):
                    raise ValueError('Historical/current source mapping differs: ' + name)
                historical_source_bindings.append(dict(original_path=name,
                    original_identity=expected, retained_original=str(archived),
                    execution_identity=actual, execution_pin=str(prior/'sealed/SEAL.json'),
                    scope='Existing optional prefix-guard hooks/reporting; current KRC1 source retained unchanged'))
                paths.add(archived)
            elif actual != expected:
                raise ValueError('Retained dependency changed: ' + name)
            paths.add(Path(name))
        paths.update([seal_path, parent/'POSTFLIGHT.json', parent/'EXIT_CODE'])
    write(root/'HISTORICAL_SOURCE_BINDINGS.json', historical_source_bindings)
    additions = read(root/'ADDITIONS.json')
    for name, expected in additions.items():
        if identity(root/'code'/name) != expected or (SOURCE/name).exists():
            raise ValueError('Require an exact additive installation: ' + name)
    for name in additions:
        target = SOURCE/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root/'code'/name, target)
        paths.add(target)
    sys.path.insert(0, str(SOURCE))
    from experiments.krc_continuation.protocol import (
        build_continuation_schedule, validate_continuation_schedule)
    from experiments.krc_continuation.evaluation_protocol import validate_plan

    old = read(prior/'sealed/PLAN.json')
    old_replay = read(replay/'sealed/PLAN.json')
    ordinary_path = continuation/'campaign_001/O/events/event_002/RESULT.json'
    k_path = continuation/'campaign_001/K/events/event_002/RESULT.json'
    krc_path = prior/'campaign_001/KRC/events/event_001/RESULT.json'
    ordinary, retained_k, krc = map(read, (ordinary_path, k_path, krc_path))
    pool_path = continuation/'campaign_001/event_002/POOL.json'
    pool = read(pool_path)
    if ordinary['features'] != retained_k['features'] or len(ordinary['features']) != 128:
        raise ValueError('Retained event-2 feedback differs between O and K')
    if set(pool['record_ids']) != {r['record_id'] for r in ordinary['features']}:
        raise ValueError('Immutable pool and presentation inputs differ')
    for name, expected in pool['identities'].items():
        if identity(name) != expected:
            raise ValueError('Retained teacher feature changed: ' + name)
        paths.add(Path(name))
    panels = []
    for p in old_replay['panels']:
        name, requests = p['name'], p['requests']
        if name == 'development':
            baseline = continuation/'campaign_001/evaluation_002/MEASUREMENTS.jsonl'
            historical = prior/'campaign_001/evaluation/MEASUREMENTS.jsonl'
            k1_method = 'K'
            expected = counts(baseline, 'K', requests)
            paths.add(baseline)
        else:
            historical = confirmation/'campaign_001/evaluation/MEASUREMENTS.jsonl'
            k1_method, expected = 'K1', None
        panels.append(dict(name=name, requests=requests, expected_control_counts=expected,
            historical_counts=dict(K1=counts(historical, k1_method, requests),
                                   KRC1=counts(historical, 'KRC', requests)),
            provenance=dict(unchanged_panel=str(replay/'sealed/PLAN.json'),
                canonical=p['provenance']['canonical'], historical_counts=str(historical),
                control_count_gate='retained K2 development only' if expected else 'none')))
        paths.add(historical)
    evaluation_ids = [r['request_id'] for p in panels for r in p['requests']]
    evaluation_prompts = {tuple(r['prompt_ids']) for p in panels for r in p['requests']}
    response_paths = sorted((continuation/'campaign_001/event_002/responses').glob('*.json'))
    if len(response_paths) != 128:
        raise ValueError('All 128 saved event-2 responses are required')
    for path in response_paths:
        row = read(path)
        if tuple(row['prompt_ids']) in evaluation_prompts:
            raise ValueError('Evaluation input overlaps training: ' + str(path))
        paths.add(path)
    cap = next(row for row in ordinary['features'] if row['record_id'] == 'train_00187')
    if cap['response_tokens'] != 2048 or cap['length'] != 2133:
        raise ValueError('Retained capped response changed')
    sealed = root/'sealed'
    sealed.mkdir(exist_ok=False)
    for name in ('config.yaml', 'MODEL_PIN.json', 'RUNTIME_PIN.json', 'TRAINING_PIN.json'):
        shutil.copy2(prior/'sealed'/name, sealed/name)
    schedule = build_continuation_schedule(ordinary, evaluation_ids=evaluation_ids)
    check = validate_continuation_schedule(schedule, ordinary, evaluation_ids=evaluation_ids)
    write(sealed/'REFERENCE_SCHEDULE.json', schedule)
    write(sealed/'REFERENCE_SCHEDULE_CHECK.json', check)
    plan = dict(config=str(sealed/'config.yaml'),
        startup_ordinary_receipt=old['ordinary_receipt'], pinned_initial=old['pinned_initial'],
        krc1_state=str(prior/'campaign_001/KRC/training_state.pt'), krc1_receipt=str(krc_path),
        krc1_checkpoint=krc['checkpoint_path'], event2_ordinary_receipt=str(ordinary_path),
        event2_k_receipt=str(k_path), retained_k2_checkpoint=retained_k['checkpoint_path'],
        reference_schedule=str(sealed/'REFERENCE_SCHEDULE.json'), panels=panels,
        target=old['target'], draft=old['draft'], worker_python=old['worker_python'],
        retained_pool=str(pool_path), historical_K2_costs=str(continuation/'campaign_001/event_002/COSTS.json'),
        scope='exact own-state KRC continuation from event 1 to 2; no later events',
        startup=dict(forwards=2, backwards=2, optimizer_steps=0),
        main=dict(start_step=128, end_step=256, windows=128, presentations=256,
            optimizer_steps=128, core_forwards=768, maximum_backwards=512,
            maximum_corrections=128, diagnostic_forwards=0),
        evaluation=dict(generations=768, initial_gate_generations=48, repeats=3),
        capped_response_preserved=cap, no_new_feedback=True, no_automatic_retry=True)
    validate_plan(plan)
    write(sealed/'PLAN.json', plan)
    paths.update([ordinary_path, k_path, krc_path, pool_path, Path(plan['krc1_state']),
                  Path(plan['historical_K2_costs'])])
    paths.update(sealed.iterdir())
    paths.update(p for p in root.iterdir() if p.is_file() and p.suffix in {'.py','.sh','.json','.md'})
    exact = {}
    for folder in map(Path, [plan['target'], plan['draft'], plan['krc1_checkpoint'], plan['retained_k2_checkpoint']]):
        files = [p for p in folder.iterdir() if p.is_file()]
        exact[str(folder)] = sorted(p.name for p in files)
        paths.update(files)
    write(sealed/'SEAL.json', dict(schema='krc_continuation_seal_v1', sealed_before_training=True,
        sealed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        files={str(p):identity(p) for p in sorted(paths)}, exact_file_sets=exact,
        parent_seals=[str(p/'sealed/SEAL.json') for p in parents]))
    print(json.dumps(dict(status='sealed', files=len(paths), retained_responses=128,
        K2_development_calls=sum(panels[0]['expected_control_counts'].values()),
        additional_K2_historical_gate=False, capped_response_preserved=True)))


if __name__ == '__main__':
    prepare(sys.argv[1])
