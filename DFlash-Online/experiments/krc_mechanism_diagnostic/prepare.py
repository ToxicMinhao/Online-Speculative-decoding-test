"""A800 preparation: verify retained evidence, install additions, seal inputs.

Run this script only after the workspace A800 rules, live jobs and storage have
been checked. It performs no model construction, forward, training or launch.
"""
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import time

SOURCE = Path('/data/sources/minhao/DFlash-Online')
RUNS = Path('/data/sources/minhao/runs')
KRC3_SHA256 = '66033b346717c66df8a78c8206beb8f5278b507153ad5adb3e3d0f234c453808'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def identity(path):
    path = Path(path)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(bytes=path.stat().st_size, sha256=digest)


def require_identity(path, expected):
    actual = identity(path)
    if actual != expected:
        raise ValueError('Required retained artifact changed: ' + str(path))
    return actual


def completed(root):
    if ((root / 'EXIT_CODE').read_text().strip() != '0'
            or read(root / 'POSTFLIGHT.json')['status'] != 'passed'
            or read(root / 'campaign_001/STATUS.json')['status'] != 'completed'):
        raise ValueError('Required original campaign is not complete/audited: ' + str(root))


def validate_installation(root, execution_source):
    additions = read(root / 'ADDITIONS.json')
    if (not re.fullmatch('[0-9a-f]{40}', execution_source.get('execution_git_commit', ''))
            or execution_source.get('dirty_files') != []
            or execution_source.get('working_source') != str(SOURCE)
            or execution_source.get('source_subdirectory') != 'DFlash-Online'
            or not execution_source.get('repository') or not execution_source.get('branch')
            or execution_source.get('additions') != additions):
        raise ValueError('Require committed clean execution source and exact additive identities')
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
            raise ValueError('Require exact additive source installation: ' + name)
    return additions


def validate_pool(pool, ordinary, responses, panels, event_number):
    ids = [row['record_id'] for row in ordinary['features']]
    feature_paths = [row['path'] for row in ordinary['features']]
    if (pool['event_id'] != f'event_{event_number:03d}' or pool['immutable'] is not True
            or pool['response_generations'] != 128 or pool['record_ids'] != ids
            or pool['features'] != feature_paths or set(pool['identities']) != set(feature_paths)
            or len(responses) != 128):
        raise ValueError('Retained immutable O-generated feedback pool differs')
    evaluation_ids = {row['request_id'] for panel in panels for row in panel['requests']}
    evaluation_prompts = {tuple(row['prompt_ids']) for panel in panels for row in panel['requests']}
    if set(ids) & evaluation_ids:
        raise ValueError('Evaluation IDs overlap training')
    raw = {}
    features = {row['record_id']: row for row in ordinary['features']}
    for path in responses:
        response = read(path)
        rid = response['record_id']
        if rid not in features or rid in raw or tuple(response['prompt_ids']) in evaluation_prompts:
            raise ValueError('Duplicate/foreign response or evaluation prompt overlap: ' + str(path))
        feature = features[rid]
        if (response['checkpoint_id'] != pool['checkpoint'] or response['session_id'] != pool['event_id']
                or len(response['prompt_ids']) != feature['prompt_length']
                or len(response['response_ids']) != feature['response_tokens']
                or feature['length'] != feature['prompt_length'] + feature['response_tokens']):
            raise ValueError('Response and prepared feature bindings differ: ' + rid)
        raw[rid] = response
    if set(raw) != set(ids):
        raise ValueError('Every retained response must remain present')
    cap_id, cap_length = ('train_00187', 2133) if event_number == 2 else ('train_00399', 2231)
    cap = features[cap_id]
    if (cap['response_tokens'] != 2048 or cap['length'] != cap_length
            or raw[cap_id]['finish_reason'] != 'length'):
        raise ValueError('The original capped response changed: ' + cap_id)
    return dict(capped_response=cap, evaluation_ids_excluded=True, evaluation_prompts_excluded=True,
                retained_responses=len(raw), newly_collected_responses=0)


def prepare(root):
    root = Path(root).resolve()
    early = RUNS / 'dflash_krc_continuation_20261028'
    late = RUNS / 'dflash_krc_events_3_4_20261028'
    original_o = RUNS / 'dflash_ok_continuation_20261027'
    for parent in (early, late):
        completed(parent)
    old, latest = read(early / 'sealed/PLAN.json'), read(late / 'sealed/PLAN.json')
    parent_seal = read(late / 'sealed/SEAL.json')
    # The latest pre-run seal already contains the verified historical source
    # mappings and original documentation copies. Do not replace old pins by
    # current Git IDs or silently ignore an unavailable historical file.
    verified = {name: require_identity(name, expected) for name, expected in parent_seal['files'].items()}
    exact = dict(parent_seal['exact_file_sets'])
    for folder, names in exact.items():
        if sorted(path.name for path in Path(folder).iterdir() if path.is_file()) != names:
            raise ValueError('Retained exact file set changed: ' + folder)
    paths = set(map(Path, verified))
    for parent in (early, late):
        paths.update(parent / name for name in ('sealed/PLAN.json', 'sealed/SEAL.json',
            'POSTFLIGHT.json', 'EXIT_CODE', 'campaign_001/STATUS.json'))
    execution_source = read(root / 'EXECUTION_SOURCE.json')
    additions = validate_installation(root, execution_source)
    panels = latest['panels']
    if panels != old['panels']:
        # Later panels add historical counts but retain the exact request data.
        if [(row['name'], row['requests']) for row in panels] != [(row['name'], row['requests']) for row in old['panels']]:
            raise ValueError('Evaluation exclusion request identities changed')
    early_seal = read(early / 'sealed/SEAL.json')
    krc1 = Path(old['krc1_state'])
    expected_krc1 = early_seal['files'][str(krc1)]
    require_identity(krc1, expected_krc1)
    episode_specs = []
    retained_checks = []
    for number in (2, 4):
        event_id = f'event_{number:03d}'
        ordinary_path = original_o / f'campaign_001/O/events/{event_id}/RESULT.json'
        pool_path = original_o / f'campaign_001/{event_id}/POOL.json'
        if number == 2:
            parent_state, parent_receipt = str(krc1), old['krc1_receipt']
            endpoint_receipt = str(early / 'campaign_001/KRC2/events/event_002/RESULT.json')
        else:
            parent_receipt = str(late / 'campaign_001/KRC/events/event_003/RESULT.json')
            parent_state = read(parent_receipt)['full_state_path']
            endpoint_receipt = str(late / 'campaign_001/KRC/events/event_004/RESULT.json')
            state_identities = read(late / 'campaign_001/events/event_003/STATE_IDENTITIES.json')
            expected = next(row for row in state_identities if row['path'] == parent_state)
            if expected['sha256'] != KRC3_SHA256:
                raise ValueError('KRC3 identity differs from the explicit authorized handoff')
            require_identity(parent_state, {key: expected[key] for key in ('bytes', 'sha256')})
            paths.add(late / 'campaign_001/events/event_003/STATE_IDENTITIES.json')
        parent, endpoint, ordinary, pool = map(read, (parent_receipt, endpoint_receipt, ordinary_path, pool_path))
        endpoint_state = endpoint['full_state_path']
        if number == 2:
            require_identity(endpoint_state, parent_seal['files'][endpoint_state])
        else:
            state_identities = read(late / 'campaign_001/events/event_004/STATE_IDENTITIES.json')
            expected = next(row for row in state_identities if row['path'] == endpoint_state)
            require_identity(endpoint_state, {key: expected[key] for key in ('bytes', 'sha256')})
            paths.add(late / 'campaign_001/events/event_004/STATE_IDENTITIES.json')
        prior_o = read(original_o / f'campaign_001/O/events/event_{number-1:03d}/RESULT.json') if number == 4 else read(old['startup_ordinary_receipt'])
        if pool['checkpoint'] != prior_o['checkpoint_path']:
            raise ValueError('Feedback source is not the preceding retained O checkpoint')
        responses = sorted((pool_path.parent / 'responses').glob('*.json'))
        retained_checks.append(dict(event_number=number,
            **validate_pool(pool, ordinary, responses, panels, number)))
        for name, expected in pool['identities'].items():
            verified[name] = require_identity(name, expected)
            paths.add(Path(name))
        paths.update(map(Path, (parent_receipt, parent_state, endpoint_receipt, endpoint_state,
            ordinary_path, pool_path)))
        paths.update(responses)
        episode_specs.append(dict(event_number=number, parent_event=number - 1,
            start_step=(number - 1) * 128, end_step=number * 128,
            parent_state=parent_state, parent_checkpoint=parent['checkpoint_path'], parent_receipt=parent_receipt,
            endpoint_state=endpoint_state, endpoint_checkpoint=endpoint['checkpoint_path'],
            endpoint_receipt=endpoint_receipt, ordinary_receipt=str(ordinary_path), pool=str(pool_path)))
    for path in paths:
        if not path.is_file():
            raise ValueError('Required input unavailable: ' + str(path))
    # First mutation: every inherited input and every staged source is already
    # verified. An interrupted installation is retained as a failed preparation,
    # not overwritten by an automatic retry.
    for name, expected in additions.items():
        target = SOURCE / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / 'code' / name, target)
        require_identity(target, expected)
        paths.update((target, root / 'code' / name))
        verified[str(target)] = verified[str(root / 'code' / name)] = expected
    sys.path.insert(0, str(SOURCE))
    from experiments.ok_continuation.resume import validate_event_receipt
    from experiments.krc_mechanism_diagnostic.protocol import (
        BUDGET, reference_schedule, build_window_plan, validate_plan)
    sealed = root / 'sealed'
    sealed.mkdir(exist_ok=False)
    for name in ('config.yaml', 'MODEL_PIN.json', 'RUNTIME_PIN.json', 'TRAINING_PIN.json'):
        if identity(early / 'sealed' / name) != identity(late / 'sealed' / name):
            raise ValueError('Historical event2/event4 configuration or runtime pin differs: ' + name)
        shutil.copy2(late / 'sealed' / name, sealed / name)
    evaluation_ids = [row['request_id'] for panel in panels for row in panel['requests']]
    all_windows = []
    for item in episode_specs:
        number = item['event_number']
        ordinary = read(item['ordinary_receipt'])
        validate_event_receipt(read(item['parent_receipt']), event_number=number - 1, objective='osd_first_error')
        validate_event_receipt(read(item['endpoint_receipt']), event_number=number,
                               objective='osd_first_error', reference_schedule=ordinary)
        schedule_path, windows_path = sealed / f'REFERENCE_SCHEDULE_{number:03d}.json', sealed / f'WINDOW_PLAN_{number:03d}.json'
        write(schedule_path, reference_schedule(ordinary, number, evaluation_ids=evaluation_ids))
        windows = build_window_plan(ordinary, number, evaluation_ids=evaluation_ids)
        write(windows_path, windows)
        all_windows.extend(windows)
        item.update(reference_schedule=str(schedule_path), window_plan=str(windows_path))
    write(sealed / 'WINDOW_PLAN.json', all_windows)
    plan = {key: old[key] for key in ('startup_ordinary_receipt', 'pinned_initial', 'target', 'draft', 'worker_python')}
    plan.update(config=str(sealed / 'config.yaml'), episodes=episode_specs, evaluation_ids=evaluation_ids,
        evaluation_exclusions=[dict(name=panel['name'], requests=panel['requests']) for panel in panels],
        execution_source=execution_source, budget=BUDGET, no_new_feedback=True, no_automatic_retry=True,
        original_scientific_sources_unchanged=True,
        scope='Independent original event2/4 replay with fixed temporary diagnostic candidates only',
        stop='After diagnostic evidence/report; no replacement mechanism or serving evaluation')
    write(sealed / 'PLAN_CHECK.json', validate_plan(plan))
    write(sealed / 'RETAINED_INPUT_CHECK.json', retained_checks)
    write(sealed / 'PLAN.json', plan)
    paths.update(sealed.iterdir())
    paths.update(path for path in root.iterdir() if path.is_file() and path.suffix in {'.json', '.txt', '.md', '.py', '.sh'})
    for item in episode_specs:
        for name in ('parent_checkpoint', 'endpoint_checkpoint'):
            folder = Path(item[name])
            files = [path for path in folder.iterdir() if path.is_file()]
            if not files:
                raise ValueError('Required original export is empty: ' + str(folder))
            exact[str(folder)] = sorted(path.name for path in files)
            paths.update(files)
    files = {str(path): verified.get(str(path)) or identity(path) for path in sorted(paths)}
    write(sealed / 'SEAL.json', dict(schema='krc_mechanism_diagnostic_seal_v1', sealed_before_training=True,
        sealed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), files=files,
        exact_file_sets=exact, parent_seals=[str(late / 'sealed/SEAL.json'), str(early / 'sealed/SEAL.json')],
        execution_source=execution_source))
    return dict(status='sealed', files=len(files), independent_events=[2, 4], windows=20,
        candidate_slots=80, execution_git_commit=execution_source['execution_git_commit'])


if __name__ == '__main__':
    print(json.dumps(prepare(sys.argv[1])))
