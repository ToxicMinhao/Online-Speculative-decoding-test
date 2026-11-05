"""Seal retained KRC2, events 3/4 and three-arm evaluation before execution."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import time

SOURCE = Path('/data/sources/minhao/DFlash-Online')
RUNS = Path('/data/sources/minhao/runs')
PUBLICATION_DOCUMENTS = {'AGENTS.md', 'README.md', 'SOURCES.json'}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


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
                or any(type(row['repeat']) is not int for row in rows)
                or not all(row['valid_measurement'] is True for row in rows)
                or any(type(row['verification_calls']) is not int or row['verification_calls'] <= 0 for row in rows)
                or len({row['verification_calls'] for row in rows}) != 1):
            raise ValueError(f'Invalid retained {method} records for {rid}')
        result[rid] = rows[0]['verification_calls']
    return result


def verify_parent(parent, documentation_bindings):
    if ((parent / 'EXIT_CODE').read_text().strip() != '0'
            or read(parent / 'POSTFLIGHT.json')['status'] != 'passed'
            or read(parent / 'campaign_001/STATUS.json')['status'] != 'completed'):
        raise ValueError('The retained KRC2 campaign must be complete and audited')
    seal = read(parent / 'sealed/SEAL.json')
    verified, used = {}, []
    for name, expected in seal['files'].items():
        actual = identity(name)  # A missing inherited pin is always fatal.
        if actual != expected:
            binding = documentation_bindings.get(name)
            permitted = SOURCE / Path(name).name
            if (Path(name) != permitted or permitted.name not in PUBLICATION_DOCUMENTS
                    or not binding or binding['original_identity'] != expected
                    or binding['execution_identity'] != actual
                    or identity(binding['retained_original']) != expected):
                raise ValueError('Retained KRC2 dependency changed: ' + name)
            used.append(dict(original_path=name, **binding))
        verified[name] = actual
    for folder, expected in seal['exact_file_sets'].items():
        actual = sorted(path.name for path in Path(folder).iterdir() if path.is_file())
        if actual != expected:
            raise ValueError('Retained export/model file set changed: ' + folder)
    return seal, verified, used


def validate_installation(root, execution_source):
    additions = read(root / 'ADDITIONS.json')
    if (not re.fullmatch('[0-9a-f]{40}', execution_source.get('execution_git_commit', ''))
            or execution_source.get('dirty_files') != []
            or execution_source.get('working_source') != str(SOURCE)
            or execution_source.get('source_subdirectory') != 'DFlash-Online'
            or not execution_source.get('repository') or not execution_source.get('branch')
            or execution_source.get('additions') != additions):
        raise ValueError('Require committed execution source and exact addition identities')
    staged = {str(path.relative_to(root / 'code')) for path in (root / 'code').rglob('*') if path.is_file()}
    if not additions or staged != set(additions):
        raise ValueError('Staged source file list differs from ADDITIONS.json')
    for name, expected in additions.items():
        relative = Path(name)
        target = SOURCE / relative
        if (relative.is_absolute() or '..' in relative.parts
                or not target.resolve().is_relative_to(SOURCE.resolve())
                or target.exists() or target.is_symlink()
                or (root / 'code' / name).is_symlink()
                or identity(root / 'code' / name) != expected):
            raise ValueError('Require exact additive installation: ' + name)
    return additions


def validate_pool(pool, ordinary, retained_k, response_paths, panels, event_number):
    features = ordinary['features']
    ids = [row['record_id'] for row in features]
    feature_paths = [row['path'] for row in features]
    if (features != retained_k['features'] or len(ids) != 128 or len(set(ids)) != 128
            or pool['event_id'] != f'event_{event_number:03d}' or pool['immutable'] is not True
            or pool['response_generations'] != 128 or pool['record_ids'] != ids
            or pool['features'] != feature_paths or set(pool['identities']) != set(feature_paths)
            or len(response_paths) != 128):
        raise ValueError('Retained O/K features, immutable pool or response count differs')
    evaluation_ids = {row['request_id'] for panel in panels for row in panel['requests']}
    evaluation_prompts = {tuple(row['prompt_ids']) for panel in panels for row in panel['requests']}
    if evaluation_ids & set(ids):
        raise ValueError('Evaluation IDs overlap training')
    raw = {}
    for path in response_paths:
        row = read(path)
        rid = row['record_id']
        if rid in raw or rid not in ids or tuple(row['prompt_ids']) in evaluation_prompts:
            raise ValueError('Duplicate/foreign response or evaluation prompt overlaps training: ' + str(path))
        feature = features[ids.index(rid)]
        if (row['checkpoint_id'] != pool['checkpoint'] or row['session_id'] != pool['event_id']
                or len(row['prompt_ids']) != feature['prompt_length']
                or len(row['response_ids']) != feature['response_tokens']
                or feature['length'] != feature['prompt_length'] + feature['response_tokens']):
            raise ValueError('Retained response/feature identity differs: ' + rid)
        raw[rid] = row
    if set(raw) != set(ids):
        raise ValueError('Retain every immutable feedback response')
    capped = [row for row in features if row['response_tokens'] == 2048]
    if event_number == 4:
        cap = next((row for row in capped if row['record_id'] == 'train_00399'), None)
        if (cap is None or cap['length'] != 2231 or cap['prompt_length'] != 183
                or raw['train_00399']['finish_reason'] != 'length'):
            raise ValueError('Retained capped train_00399 must remain unchanged')
    return feature_paths, capped


def build_panels(old, parent, continuation, paths):
    ok_plan_path = continuation / 'sealed/PLAN.json'
    ok_plan = read(ok_plan_path)
    # O1 is the exact event-1 O evidence used to initialize the retained O/K run.
    initial_receipt = Path(ok_plan['initial']['O']['receipt'])
    o1_measurements = initial_receipt.parents[3] / 'evaluation/MEASUREMENTS.jsonl'
    o2_measurements = continuation / 'campaign_001/evaluation_002/MEASUREMENTS.jsonl'
    event4_measurements = continuation / 'campaign_001/evaluation_004/MEASUREMENTS.jsonl'
    panels = []
    for panel in old['panels']:
        name, requests = panel['name'], panel['requests']
        measurements = parent / 'campaign_001/evaluation' / name / 'MEASUREMENTS.jsonl'
        historical = dict(K1=panel['historical_counts']['K1'], KRC1=panel['historical_counts']['KRC1'],
            K2=retained_counts(measurements, 'K2', requests),
            KRC2=retained_counts(measurements, 'KRC2', requests))
        expected = None
        if name == 'development':
            historical.update(O1=retained_counts(o1_measurements, 'O', requests),
                              O2=retained_counts(o2_measurements, 'O', requests))
            expected = {arm + '4': retained_counts(event4_measurements, arm, requests) for arm in ('O', 'K')}
            if {arm: sum(values.values()) for arm, values in expected.items()} != {'O4': 2994, 'K4': 2942}:
                raise ValueError('Retained development O4/K4 totals differ from the reviewed evidence')
            paths.update([initial_receipt, o1_measurements, o2_measurements, event4_measurements])
        panels.append(dict(name=name, requests=requests, expected_control_counts=expected,
            historical_counts=historical, provenance=dict(unchanged_panel=str(parent / 'sealed/PLAN.json'),
                canonical=panel['provenance']['canonical'], event2_counts=str(measurements),
                event1_K_counts=panel['provenance']['historical_counts'],
                event1_O_counts=str(o1_measurements) if name == 'development' else None,
                event2_O_counts=str(o2_measurements) if name == 'development' else None,
                event4_control_counts=str(event4_measurements) if name == 'development' else None)))
        paths.add(measurements)
    paths.add(ok_plan_path)
    return panels


def prepare(root):
    root = Path(root).resolve()
    parent = RUNS / 'dflash_krc_continuation_20261028'
    continuation = RUNS / 'dflash_ok_continuation_20261027'
    old = read(parent / 'sealed/PLAN.json')
    bindings_path = root / 'DOCUMENTATION_BINDINGS.json'
    bindings = read(bindings_path) if bindings_path.exists() else {}
    parent_seal, verified, document_changes = verify_parent(parent, bindings)
    paths = set(map(Path, parent_seal['files']))
    paths.update([parent / 'sealed/PLAN.json', parent / 'sealed/SEAL.json', parent / 'POSTFLIGHT.json',
                  parent / 'EXIT_CODE', parent / 'campaign_001/STATUS.json'])
    for binding in document_changes:
        paths.add(Path(binding['retained_original']))
    execution_source = read(root / 'EXECUTION_SOURCE.json')
    additions = validate_installation(root, execution_source)
    panels = build_panels(old, parent, continuation, paths)
    krc2_path = parent / 'campaign_001/KRC2/events/event_002/RESULT.json'
    krc2 = read(krc2_path)
    for key in ('full_state_path', 'checkpoint_path'):
        if not isinstance(krc2.get(key), str) or not Path(krc2[key]).exists():
            raise ValueError('Retained KRC2 receipt must resolve existing ' + key)
    paths.update([krc2_path, Path(krc2['full_state_path'])])
    retained_events, events, capped = [], [], {}
    prior_o_checkpoint = read(old['event2_ordinary_receipt'])['checkpoint_path']
    for number in (3, 4):
        event_id = f'event_{number:03d}'
        ordinary_path = continuation / f'campaign_001/O/events/{event_id}/RESULT.json'
        k_path = continuation / f'campaign_001/K/events/{event_id}/RESULT.json'
        pool_path = continuation / f'campaign_001/{event_id}/POOL.json'
        ordinary, retained_k, pool = map(read, (ordinary_path, k_path, pool_path))
        if pool['checkpoint'] != prior_o_checkpoint:
            raise ValueError('Feedback must come from the retained preceding O checkpoint')
        responses = sorted((pool_path.parent / 'responses').glob('*.json'))
        feature_paths, capped[event_id] = validate_pool(pool, ordinary, retained_k, responses, panels, number)
        for path, expected in pool['identities'].items():
            if identity(path) != expected:
                raise ValueError('Retained teacher feature changed: ' + path)
            verified[path] = expected
            paths.add(Path(path))
        paths.update([ordinary_path, k_path, pool_path, pool_path.parent / 'COSTS.json', *responses])
        retained_events.append((number, ordinary, retained_k))
        events.append(dict(event_number=number, ordinary_receipt=str(ordinary_path), k_receipt=str(k_path),
                           pool=str(pool_path), feature_paths=feature_paths))
        prior_o_checkpoint = ordinary['checkpoint_path']
    # Check every inherited/new artifact before mutating the additive source tree.
    for path in paths:
        if not path.is_file():
            raise ValueError('Required artifact missing: ' + str(path))
    for name in additions:
        target = SOURCE / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / 'code' / name, target)
        paths.update([target, root / 'code' / name])
        verified[str(target)] = identity(target)
        if verified[str(target)] != additions[name]:
            raise ValueError('Installed source differs: ' + name)
        verified[str(root / 'code' / name)] = additions[name]
    sys.path.insert(0, str(SOURCE))
    from experiments.krc_events_3_4.protocol import build_continuation_schedule, validate_retained
    from experiments.krc_events_3_4.evaluation_protocol import validate_plan
    retained_check = validate_retained(krc2, retained_events)
    sealed = root / 'sealed'
    sealed.mkdir(exist_ok=False)
    for name in ('config.yaml', 'MODEL_PIN.json', 'RUNTIME_PIN.json', 'TRAINING_PIN.json'):
        shutil.copy2(parent / 'sealed' / name, sealed / name)
    evaluation_ids = [row['request_id'] for panel in panels for row in panel['requests']]
    for event, (_, ordinary, _) in zip(events, retained_events):
        path = sealed / f"REFERENCE_SCHEDULE_{event['event_number']:03d}.json"
        write(path, build_continuation_schedule(ordinary, event['event_number'], evaluation_ids=evaluation_ids))
        event['reference_schedule'] = str(path)
    plan = {key: old[key] for key in ('startup_ordinary_receipt', 'pinned_initial', 'target', 'draft', 'worker_python')}
    plan.update(config=str(sealed / 'config.yaml'), krc2_state=krc2['full_state_path'],
        krc2_receipt=str(krc2_path), krc2_checkpoint=krc2['checkpoint_path'], events=events, panels=panels,
        retained_o4_checkpoint=retained_events[-1][1]['checkpoint_path'],
        retained_k4_checkpoint=retained_events[-1][2]['checkpoint_path'], execution_source=execution_source,
        scope='KRC2 own-state continuation through events 3 and 4, then O4/K4/KRC4 evaluation; stop at event 4',
        startup=dict(forwards=2, backwards=2, optimizer_steps=0),
        main=dict(start_step=256, end_step=512, presentations=512, optimizer_steps=256,
                  core_forwards=1536, maximum_backwards=1024, maximum_corrections=256, diagnostic_forwards=0),
        evaluation=dict(generations=1152, initial_gate_generations=72, repeats=3),
        capped_responses_preserved=capped, no_new_feedback=True, no_automatic_retry=True,
        historical_training_costs={event['event_number']: str(Path(event['pool']).parent / 'COSTS.json') for event in events})
    write(sealed / 'PLAN_CHECK.json', validate_plan(plan))
    write(sealed / 'RETAINED_CHECK.json', retained_check)
    write(sealed / 'PLAN.json', plan)
    write(root / 'DOCUMENTATION_CHANGE_CHECK.json', document_changes)
    paths.update(sealed.iterdir())
    paths.update(path for path in root.iterdir() if path.is_file() and path.suffix in {'.py', '.sh', '.json', '.md', '.txt'})
    exact = dict(parent_seal['exact_file_sets'])
    for key in ('target', 'draft', 'krc2_checkpoint', 'retained_o4_checkpoint', 'retained_k4_checkpoint'):
        folder = Path(plan[key])
        files = [path for path in folder.iterdir() if path.is_file()]
        if not files:
            raise ValueError('Required export/model is empty: ' + str(folder))
        exact[str(folder)] = sorted(path.name for path in files)
        paths.update(files)
    # Inherited files were verified once above and are never mutated by preparation.
    files = {str(path): verified[str(path)] if str(path) in verified else identity(path) for path in sorted(paths)}
    write(sealed / 'SEAL.json', dict(schema='krc_events_3_4_seal_v1', sealed_before_training=True,
        sealed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), files=files,
        exact_file_sets=exact, parent_seals=[str(parent / 'sealed/SEAL.json')], execution_source=execution_source))
    return dict(status='sealed', files=len(files), retained_responses=256,
        development_control_calls={arm: sum(values.values()) for arm, values in panels[0]['expected_control_counts'].items()},
        additional_historical_control_gate=False, complete_state_source='retained receipt full_state_path',
        execution_git_commit=execution_source['execution_git_commit'], capped_train_00399_preserved=True)


if __name__ == '__main__':
    print(json.dumps(prepare(sys.argv[1])))
