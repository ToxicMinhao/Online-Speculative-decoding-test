"""Install only the reviewed R files; preserve the previous source beside receipts."""
from pathlib import Path
import hashlib
import json
import shutil

root = Path('/data/sources/minhao/runs/dflash_reference_margin_implementation_20261026')
live = Path('/data/sources/minhao/DFlash-Online')
changes = json.loads((root / 'SOURCE_CHANGES.json').read_text())
free_before = shutil.disk_usage(root).free
assert free_before >= (8 * 1024**3 + 512 * 1024**2)
for item in changes:
    path = live / item['path']
    actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
    assert actual == item['before_sha256'], (str(path), 'live source changed')
    staged = root / 'review_source' / item['path']
    assert hashlib.sha256(staged.read_bytes()).hexdigest() == item['after_sha256']
for item in changes:
    path = live / item['path']
    if path.exists():
        backup = root / 'source_before' / item['path']
        backup.parent.mkdir(parents=True, exist_ok=True)
        assert not backup.exists()
        shutil.copy2(path, backup)
    shutil.copy2(root / 'review_source' / item['path'], path)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == item['after_sha256']
receipt = {'files': changes, 'free_bytes_before': free_before,
           'free_bytes_after': shutil.disk_usage(root).free,
           'planned_additional_bytes': 512 * 1024**2, 'reserve_bytes': 8 * 1024**3,
           'removed_paths': [], 'gpu_experiment_started': False}
(root / 'SOURCE_DEPLOYMENT.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2))
