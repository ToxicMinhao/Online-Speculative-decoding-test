"""Bind G's code, unchanged configuration, original feedback and retained K1."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


def identity(path):
    with Path(path).open('rb') as stream:
        value = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(bytes=Path(path).stat().st_size, sha256=value)


def seal(root):
    root = Path(root)
    runs = root.parent
    code = Path('/data/sources/minhao/DFlash-Online')
    original = runs / 'dflash_osd_one_event_20261025'
    prior = runs / 'dflash_first_error_execution_20261026'
    sealed = root / 'sealed'
    sealed.mkdir(exist_ok=False)
    ordinary_receipt = original / 'campaign_001/trainer/events/event_001/RESULT.json'
    k_receipt = prior / 'campaign_001/K/events/event_001/RESULT.json'
    ordinary = json.loads(ordinary_receipt.read_text())
    k = json.loads(k_receipt.read_text())
    manifest = original / 'sealed/MANIFEST.json'
    from campaign import validate_scope
    validate_scope(json.loads(manifest.read_text()), ordinary, k)
    retained = json.loads((prior / 'RETAINED_ARTIFACTS.json').read_text())
    paths = {ordinary_receipt, k_receipt, manifest}
    for feature in ordinary['features']:
        path = Path(feature['path'])
        expected = retained[str(path)]
        if identity(path) != {key: expected[key] for key in ('bytes', 'sha256')}:
            raise ValueError('Original feedback changed: ' + str(path))
        paths.add(path)
    for name in ('config.yaml', 'RUNTIME_PIN.json', 'MODEL_PIN.json'):
        shutil.copy2(original / 'sealed' / name, sealed / name)
    pinned_initial = runs / 'dflash_prefix_ranking_gpu_diagnostic_20261026/trials/O2/initial.pt'
    old_qualification_seal = json.loads((prior / 'SEAL.json').read_text())
    if identity(pinned_initial)['sha256'] != old_qualification_seal['dependency_sha256'][str(pinned_initial)]:
        raise ValueError('Historical initial-state pin changed')
    paths.add(pinned_initial)
    author = code / 'training'
    git = ['git', '-c', 'safe.directory=' + str(author), '-C', str(author)]
    commit = subprocess.check_output(git + ['rev-parse', 'HEAD'], text=True).strip()
    changes = subprocess.check_output(git + ['diff', '--name-only'], text=True).splitlines()
    if commit != '1839aae30003e3573d00979eb1edd57347b3578c' or set(changes) - {'scripts/prepare_hidden_states.py'}:
        raise ValueError('Qualified author training sources changed')
    author_files = subprocess.check_output(git + ['ls-files', 'specforge'], text=True).splitlines()
    paths.update(author / name for name in author_files if (author / name).is_file())
    environment = json.loads((code / 'SOURCES.json').read_text())['environment']
    training_pin = {key: environment[key] for key in ('python', 'torch', 'transformers')}
    (sealed / 'TRAINING_PIN.json').write_text(json.dumps(dict(environment=training_pin,
        author_commit=commit, compatibility_changes=changes), indent=2) + '\n')
    k_export = Path(k['checkpoint_path'])
    previous = json.loads((runs / 'dflash_ok_continuation_20261027/sealed/SEAL.json').read_text())
    for path in k_export.iterdir():
        if path.is_file():
            if identity(path) != previous['files'][str(path)]:
                raise ValueError('Retained K1 changed: ' + str(path))
            paths.add(path)
    models = Path('/data/sources/minhao/models/qwen3_dflash_foundation_20261022')
    plan = dict(config=str(sealed / 'config.yaml'), manifest=str(manifest),
        ordinary_receipt=str(ordinary_receipt), k_receipt=str(k_receipt),
        pinned_initial=str(pinned_initial), target=str(models / 'target'), draft=str(models / 'draft'),
        worker_python='/data/conda_envs/minhao/adaflash_author/bin/python',
        main=dict(optimizer_steps=128, presentations=256, passes=2, seed=17),
        qualification=dict(temporary_optimizer_steps=3, response_presentations=6,
                           startup_draft_response_evaluations=2, startup_optimizer_steps=0),
        probes=dict(steps=[1, 64, 128], candidates=3, responses_per_step=2, response_evaluations=18),
        evaluation=dict(arms=['K', 'G'], requests=64, repeats=3, generations=384,
                        canonical='retained F', fresh_panel=False),
        serving=dict(block=16, output_cap=2048, context=4096, greedy=True, concurrency=1,
                     capture=False, concurrent_training=False),
        new_feedback_generations=0, no_automatic_retry=True,
        projection=dict(h_gradient_accumulation='float32', model_and_adam='bfloat16',
                        guarantee='first-order ideal only; rounded derivative reported',
                        norm_probe_target='pre-round FP32 projected displacement norm'))
    (sealed / 'PLAN.json').write_text(json.dumps(plan, indent=2) + '\n')
    for base in (code / 'online_feedback', code / 'experiments', code / 'tests'):
        paths.update(p for p in base.rglob('*.py') if '__pycache__' not in p.parts)
    paths.update(p for p in sealed.iterdir() if p.is_file())
    paths.update(root / name for name in ('campaign.py', 'seal_remote.py', 'preflight.py',
                 'test_campaign.py', 'run_campaign.sh', 'supervise.sh', 'AUTHORIZATION.md'))
    result = dict(sealed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                  files={str(p): identity(p) for p in sorted(paths)},
                  k_exact_file_set=sorted(p.name for p in k_export.iterdir() if p.is_file()),
                  k_export=str(k_export))
    (sealed / 'SEAL.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(status='sealed', files=len(paths))))


if __name__ == '__main__':
    seal(sys.argv[1])
