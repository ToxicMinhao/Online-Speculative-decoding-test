"""Validate existing inputs, models, retained controls and peak storage before R."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

from experiments.osd_components.prepare import verify_files, verify_native_runtime
from experiments.osd_one_event.protocol import validate_manifest
from online_feedback.qualify_sglang import check_recipe
from online_feedback.storage import write_json


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def run(root):
    root = Path(root)
    code = Path('/data/sources/minhao/DFlash-Online')
    runs = Path('/data/sources/minhao/runs')
    ordinary_root = runs / 'dflash_osd_one_event_20261025'
    prefix_root = runs / 'dflash_prefix_ranking_execution_20261026'
    started = time.perf_counter()
    seal = json.loads((root / 'SEAL.json').read_text())
    for category, base in (('source_sha256', code), ('execution_sha256', root),
                           ('dependency_sha256', Path('/'))):
        for name, expected in seal[category].items():
            if digest(base / name) != expected:
                raise ValueError('Pinned source/dependency changed: ' + name)
    old = json.loads((ordinary_root / 'sealed/SEAL.json').read_text())
    for name, expected in old['input_sha256'].items():
        if digest(ordinary_root / 'sealed' / name) != expected:
            raise ValueError('Original sealed input changed: ' + name)
    manifest = json.loads((ordinary_root / 'sealed/MANIFEST.json').read_text())
    validate_manifest(manifest)
    recipe = check_recipe(ordinary_root / 'sealed/config.yaml')
    models = Path('/data/sources/minhao/models/qwen3_dflash_foundation_20261022')
    model_pin = json.loads((ordinary_root / 'sealed/MODEL_PIN.json').read_text())
    for name in ('target', 'draft'):
        verify_files(models / name, model_pin[name]['files'], exact_files=True)
    runtime = verify_native_runtime('/data/conda_envs/minhao/adaflash_author/bin/python',
        json.loads((ordinary_root / 'sealed/RUNTIME_PIN.json').read_text()))
    old_stats = json.loads((runs / 'dflash_prefix_ranking_implementation_20261025/INSTALL.json').read_text())[
        'retained_ordinary_artifact_stats']
    retained = {}
    for name, expected in old_stats.items():
        path = Path(name)
        stat = path.stat()
        if [stat.st_size, stat.st_mtime_ns] != expected:
            raise ValueError('Retained ordinary artifact changed: ' + name)
        retained[name] = {'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns, 'sha256': digest(path)}
    ordinary = json.loads((ordinary_root / 'campaign_001/trainer/events/event_001/RESULT.json').read_text())
    receipts = {'O': ordinary}
    for arm in ('U', 'P'):
        state_root = prefix_root / 'campaign_001' / arm
        receipt_path = state_root / 'events/event_001/RESULT.json'
        receipt = json.loads(receipt_path.read_text())
        receipts[arm] = receipt
        if receipt['features'] != ordinary['features']:
            raise ValueError('Retained control feedback differs: ' + arm)
        for left, right in zip(receipt['records'], ordinary['records'], strict=True):
            for key in ('record_id', 'pass', 'sampled_anchors', 'optimizer_stepped'):
                if left[key] != right[key]:
                    raise ValueError('Retained control schedule differs: ' + arm + '.' + key)
        for path in [receipt_path, state_root / 'training_state.pt', *Path(receipt['checkpoint_path']).iterdir()]:
            if path.is_file():
                stat = path.stat()
                retained[str(path)] = {'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                                       'sha256': digest(path)}
    for arm, receipt in receipts.items():
        if (receipt['status'], receipt['optimizer_steps'], receipt['response_presentations'],
                len(receipt['features'])) != ('completed', 128, 256, 128):
            raise ValueError('Retained control event incomplete: ' + arm)
    write_json(root / 'RETAINED_ARTIFACTS.json', retained)
    weight_bytes = sum(path.stat().st_size for path in (models / 'draft').glob('*.safetensors'))
    budget = {
        'draft_bytes': weight_bytes,
        'qualification_and_main_state_overlap_bytes': int(42 * weight_bytes * 1.10),
        'compiler_logs_staging_and_work_bytes': 10 * 2**30,
        'reserve_bytes': 8 * 2**30,
        'free_bytes': shutil.disk_usage(root).free,
        'reason': 'Retain all four qualification trials, failure/atomic overlap and R state/export/initial; no reclamation assumed.',
    }
    budget['required_bytes'] = sum(budget[key] for key in (
        'qualification_and_main_state_overlap_bytes', 'compiler_logs_staging_and_work_bytes', 'reserve_bytes'))
    budget['admitted'] = weight_bytes > 0 and budget['free_bytes'] >= budget['required_bytes']
    write_json(root / 'STORAGE_ADMISSION.json', budget)
    if not budget['admitted']:
        raise RuntimeError('R peak storage budget plus reserve does not fit')
    write_json(root / 'PREFLIGHT.json', {
        'status': 'passed', 'reviewed_source_files': len(seal['source_sha256']),
        'retained_artifacts': len(retained), 'recipe': recipe, 'runtime': runtime,
        'original_inputs_unchanged': True, 'storage': budget, 'wall_s': time.perf_counter() - started})
    print('R source/model/runtime/feedback/control/storage preflight passed.', flush=True)


if __name__ == '__main__':
    run(sys.argv[1])
