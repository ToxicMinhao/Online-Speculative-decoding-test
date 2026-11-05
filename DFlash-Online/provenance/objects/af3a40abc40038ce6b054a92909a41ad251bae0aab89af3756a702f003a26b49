"""Bind selected inputs, fixed checkpoint bytes and unchanged serving code before use."""

import hashlib
import json
from pathlib import Path
import shutil
import time


def identity(path):
    with path.open('rb') as stream:
        value = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(bytes=path.stat().st_size, sha256=value)


def seal(root):
    root = Path(root)
    output = root / 'sealed'
    output.mkdir(exist_ok=False)
    code = Path('/data/sources/minhao/DFlash-Online')
    ordinary = Path('/data/sources/minhao/runs/dflash_osd_one_event_20261025')
    previous = Path('/data/sources/minhao/runs/dflash_first_error_execution_20261026')
    requests = json.loads((root / 'inventory_audit/NEW_PANEL.json').read_text())
    selection = json.loads((root / 'inventory_audit/SELECTION_AUDIT.json').read_text())
    assert len(requests) == 64 and selection['selection_seed'] == 20261027
    assert selection['selection_complete'] is True and selection['selected_count'] == 64
    assert selection['global_freshness'] is False
    assert selection['freshness_scope'].startswith('New to the current one-event O/K study only;')
    assert identity(root / 'inventory_audit/NEW_PANEL.json')['sha256'] == selection['panel_sha256']
    assert [row['original_index'] for row in requests] == selection['selected_source_rows_in_order']
    receipts = {'O': ordinary / 'campaign_001/trainer/events/event_001/RESULT.json',
                'K': previous / 'campaign_001/K/events/event_001/RESULT.json'}
    checkpoints = {}
    for arm, path in receipts.items():
        result = json.loads(path.read_text())
        assert result['status'] == 'completed' and result['optimizer_steps'] == 128
        assert result['objective'] == {'O': 'osd', 'K': 'osd_first_error'}[arm]
        checkpoints[arm] = result['checkpoint_path']
    models = Path('/data/sources/minhao/models/qwen3_dflash_foundation_20261022')
    plan = dict(selection_seed=20261027, freshness_scope='current_one_event_O_K_study',
                older_project_exposure_disclosed=True, planned_generations=384,
                checkpoints=checkpoints, checkpoint_receipts={k: str(v) for k, v in receipts.items()},
                target=str(models / 'target'), released_draft=str(models / 'draft'),
                worker_python='/data/conda_envs/minhao/adaflash_author/bin/python', requests=requests,
                qualification_requests=8, qualification_generations_included=48,
                order='(zero_based_request_ordinal + zero_based_repeat) % 2',
                input_reference='first O execution per request; native verifier agreement, not target-only',
                serving=dict(greedy=True, block_size=16, output_cap=2048, context_limit=4096,
                             concurrency=1, capture=False, extra_generation_warmups=0))
    from campaign import validate_plan
    validate_plan(plan)
    (output / 'PLAN.json').write_text(json.dumps(plan, indent=2) + '\n')
    shutil.copy2(ordinary / 'sealed/RUNTIME_PIN.json', output / 'RUNTIME_PIN.json')
    paths = set(receipts.values())
    old = json.loads((previous / 'SEAL.json').read_text())
    for name, expected in old['source_sha256'].items():
        path = code / name
        assert identity(path)['sha256'] == expected, str(path)
        paths.add(path)
    # Training-only startup states are not dependencies of this checkpoint replay.
    from experiments.osd_components.prepare import verify_files
    model_pin = json.loads((ordinary / 'sealed/MODEL_PIN.json').read_text())
    for name in ('target', 'draft'):
        verify_files(models / name, model_pin[name]['files'], exact_files=True)
    retained = json.loads((previous / 'RETAINED_ARTIFACTS.json').read_text())
    ordinary_files = {name: expected for name, expected in retained.items()
                      if str(name).startswith(checkpoints['O'] + '/')}
    assert ordinary_files, 'Historical O export identity is required'
    assert set(ordinary_files) == {str(p) for p in Path(checkpoints['O']).iterdir() if p.is_file()}
    for name, expected in ordinary_files.items():
        assert identity(Path(name)) == {key: expected[key] for key in ('bytes', 'sha256')}, name
    directories = [*map(Path, checkpoints.values()), models / 'target', models / 'draft']
    exact_file_sets = {}
    for folder in directories:
        paths.update(p for p in folder.iterdir() if p.is_file())
        exact_file_sets[str(folder)] = sorted(p.name for p in folder.iterdir() if p.is_file())
    for folder in [root / 'inventory_audit', code / 'experiments/fresh_confirmation', output]:
        paths.update(p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    paths.update(root / name for name in ('campaign.py', 'preflight.py', 'seal_remote.py', 'test_campaign.py',
                 'run_campaign.sh', 'supervise.sh', 'AUTHORIZATION.md', 'FRESHNESS_AMENDMENT.md'))
    files = {str(p): identity(p) for p in sorted(paths)}
    storage = dict(reserve_bytes=8 * 1024**3, peak_new_bytes=2 * 1024**3,
                   components='compact native traces/repeats, logs, bounded retrieval staging; no checkpoint writes',
                   free_bytes=shutil.disk_usage(root).free)
    assert storage['free_bytes'] >= storage['reserve_bytes'] + storage['peak_new_bytes']
    result = dict(schema='fresh_confirmation_seal_v1', sealed_before_generation=True,
                  sealed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                  files=files, exact_file_sets=exact_file_sets, storage=storage,
                  checkpoint_identity_provenance={
                      'O': 'Matches prior retained export digests and exact file membership.',
                      'K': 'New digest of the fixed completed K export at its original receipt path; '
                           'no prior export digest is claimed. Native loaded-tensor equality is checked before use.'},
                  prior_source_files_unchanged=len(old['source_sha256']))
    (output / 'SEAL.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(status='sealed', files=len(files), requests=len(requests),
                          checkpoints=checkpoints, storage=storage)))


if __name__ == '__main__':
    import sys
    seal(sys.argv[1])
