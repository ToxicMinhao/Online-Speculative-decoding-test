"""Install only the reviewed U/P source/test files; never launch a model."""
import hashlib
import json
from pathlib import Path
import shutil

root = Path('/data/sources/minhao/runs/dflash_prefix_ranking_implementation_20261025')
code = Path('/data/sources/minhao/DFlash-Online')
changes = json.loads((root / 'SOURCE_CHANGES.json').read_text())
reserve = 8 * 1024**3
allowance = 256 * 1024**2
free_before = shutil.disk_usage(code).free
assert free_before >= reserve + allowance, 'CPU-only source/test storage admission failed'
for item in changes:
    existing = code / item['path']
    digest = hashlib.sha256(existing.read_bytes()).hexdigest() if existing.exists() else None
    assert digest == item['before_sha256'], f"Source changed during implementation: {existing}"
    incoming = root / 'source' / item['path']
    assert hashlib.sha256(incoming.read_bytes()).hexdigest() == item['after_sha256']

ordinary = Path('/data/sources/minhao/runs/dflash_osd_one_event_20261025/campaign_001')
receipt_path = ordinary / 'trainer/events/event_001/RESULT.json'
receipt = json.loads(receipt_path.read_text())
assert receipt['optimizer_steps'] == 128 and receipt['response_presentations'] == 256
retained = [Path(feature['path']) for feature in receipt['features']]
retained += [receipt_path, ordinary / 'trainer/training_state.pt']
retained += list((ordinary / 'trainer/events/event_001/export').glob('*'))
retained = [path for path in retained if path.is_file()]
before = {str(path): (path.stat().st_size, path.stat().st_mtime_ns) for path in retained}
assert len(receipt['features']) == 128 and all(Path(f['path']).is_file() for f in receipt['features'])

for item in changes:
    destination = code / item['path']
    if destination.exists():
        backup = root / 'before' / item['path']
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(destination, backup)
    shutil.copy2(root / 'source' / item['path'], destination)
    assert hashlib.sha256(destination.read_bytes()).hexdigest() == item['after_sha256']

after = {str(path): (path.stat().st_size, path.stat().st_mtime_ns) for path in retained}
assert before == after, 'Completed ordinary O artifacts changed'
result = {
    'status': 'source_installed_cpu_tests_only', 'files': changes,
    'free_bytes_before': free_before, 'free_bytes_after': shutil.disk_usage(code).free,
    'reserve_bytes': reserve, 'cpu_work_allowance_bytes': allowance,
    'retained_ordinary_artifact_stats': before,
    'ordinary_artifacts_unchanged': True, 'gpu_launch': False,
    'training_launch': False, 'generation_launch': False,
}
(root / 'INSTALL.json').write_text(json.dumps(result, indent=2) + '\n')
print('Installed eight source/test files; ordinary O artifacts preserved. No GPU launch.')
