"""Seal the additional panel and existing K1/KRC exports before generation."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time


def identity(path):
    path = Path(path)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(bytes=path.stat().st_size, sha256=digest)


def seal(root):
    root = Path(root)
    output = root / 'sealed'
    output.mkdir(exist_ok=False)
    code = Path('/data/sources/minhao/DFlash-Online')
    previous = Path('/data/sources/minhao/runs/dflash_krc_event_20261027')
    original = Path('/data/sources/minhao/runs/dflash_osd_one_event_20261025')
    requests = json.loads((root / 'inventory_audit/NEW_PANEL.json').read_text())
    selection = json.loads((root / 'inventory_audit/SELECTION_AUDIT.json').read_text())
    if len(requests) != 64 or selection['selection_seed'] != 20261028:
        raise ValueError('Require the sealed seed-20261028 64-request panel')
    if not selection['selection_complete'] or selection['missing_evidence'] or selection['selected_count'] != 64:
        raise ValueError('Panel preparation or exclusion evidence is incomplete')
    if [row['original_index'] for row in requests] != selection['selected_source_rows_in_order']:
        raise ValueError('Panel order differs from the deterministic selection')
    if identity(root / 'inventory_audit/NEW_PANEL.json')['sha256'] != selection['panel_sha256']:
        raise ValueError('Selected panel changed after its selection audit')
    receipts = {
        'K1': Path('/data/sources/minhao/runs/dflash_first_error_execution_20261026/campaign_001/K/events/event_001/RESULT.json'),
        'KRC': previous / 'campaign_001/KRC/events/event_001/RESULT.json',
    }
    checkpoints = {}
    for arm, path in receipts.items():
        receipt = json.loads(path.read_text())
        if (receipt['status'], receipt['optimizer_steps'], receipt['objective']) != ('completed', 128, 'osd_first_error'):
            raise ValueError('Wrong retained one-event checkpoint: ' + arm)
        checkpoints[arm] = receipt['checkpoint_path']
    old_seal = json.loads((previous / 'sealed/SEAL.json').read_text())
    paths = set(receipts.values())
    unchanged = 0
    for name, expected in old_seal['files'].items():
        if name.startswith(str(code) + '/') or name.startswith(checkpoints['K1'] + '/'):
            if identity(name) != expected:
                raise ValueError('Qualified code or K1 changed: ' + name)
            paths.add(Path(name))
            unchanged += 1
    # KRC is the completed export at the exact original result receipt path.
    # Its loaded tensor equality was verified in the completed development run.
    previous_weights = previous / 'campaign_001/evaluation/WEIGHTS_KRC.json'
    native_identity = json.loads(previous_weights.read_text())
    if (native_identity['errors'] or not native_identity['native_reply']['all_equal']
            or native_identity['native_reply']['checkpoint_path'] != checkpoints['KRC']):
        raise ValueError('Retained KRC publication evidence does not identify this export')
    paths.add(previous_weights)
    models = Path('/data/sources/minhao/models/qwen3_dflash_foundation_20261022')
    plan = dict(selection_seed=20261028, freshness_scope='current_K_KRC_study',
        older_project_exposure_disclosed=True, planned_generations=384,
        checkpoints=checkpoints, checkpoint_receipts={k: str(v) for k,v in receipts.items()},
        target=str(models/'target'), released_draft=str(models/'draft'),
        worker_python='/data/conda_envs/minhao/adaflash_author/bin/python', requests=requests,
        qualification_requests=8, qualification_generations_included=48,
        order='(zero_based_request_ordinal + zero_based_repeat) % 2',
        input_reference='first fresh K1 execution per request; native verifier agreement, not target-only',
        serving=dict(greedy=True, dtype='bfloat16', block_size=16, output_cap=2048,
                     context_limit=4096, concurrency=1, capture=False, extra_generation_warmups=0))
    from campaign import validate_plan
    validate_plan(plan)
    (output/'PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
    for name in ('RUNTIME_PIN.json','MODEL_PIN.json'):
        shutil.copy2(original/'sealed'/name, output/name)
    from experiments.osd_components.prepare import verify_files
    model_pin = json.loads((output/'MODEL_PIN.json').read_text())
    for name in ('target','draft'):
        verify_files(models/name, model_pin[name]['files'], exact_files=True)
    exact_file_sets = {}
    for folder in [*map(Path,checkpoints.values()), models/'target', models/'draft']:
        files = [p for p in folder.iterdir() if p.is_file()]
        paths.update(files)
        exact_file_sets[str(folder)] = sorted(p.name for p in files)
    for folder in (root/'inventory_audit', code/'experiments/krc_confirmation', output):
        paths.update(p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    paths.update(root/name for name in ('campaign.py','test_campaign.py','preflight.py',
        'seal_remote.py','run_campaign.sh','supervise.sh','AUTHORIZATION.json','HANDOFF.md'))
    files = {str(p): identity(p) for p in sorted(paths)}
    storage = dict(reserve_bytes=8*2**30, peak_new_bytes=2*2**30,
        components='compact traces/repeats, logs, archive/retrieval staging; no weights/optimizer writes',
        free_bytes=shutil.disk_usage(root).free)
    if storage['free_bytes'] < storage['reserve_bytes'] + storage['peak_new_bytes']:
        raise RuntimeError('Insufficient peak storage plus unchanged reserve')
    result = dict(schema='krc_confirmation_seal_v1',sealed_before_generation=True,
        sealed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),files=files,
        exact_file_sets=exact_file_sets,storage=storage,prior_source_and_K1_files_unchanged=unchanged,
        checkpoint_identity_provenance={
            'K1':'Matches retained KRC campaign seal and original completed K1 receipt.',
            'KRC':'Exact completed KRC export path and prior native loaded-tensor equality; file bytes pinned now and rechecked after evaluation.'})
    (output/'SEAL.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status='sealed',files=len(files),requests=len(requests),checkpoints=checkpoints,storage=storage)))


if __name__=='__main__':
    seal(sys.argv[1])
