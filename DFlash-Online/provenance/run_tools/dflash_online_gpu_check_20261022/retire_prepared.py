"""Retire exactly the three previously reviewed and prepared restart payloads."""
from pathlib import Path
import json
import os
import shutil

root = Path('/data/sources/minhao/runs/dflash_online_gpu_check_20261022')
receipt = json.loads((root / 'PREPARED.json').read_text())
assert receipt['status'] == 'prepared_no_deletion'
assert len(receipt['prepared']) == 3
receipt['retired'] = []
receipt['free_before_retirement'] = shutil.disk_usage(root).free
for row in receipt['prepared']:
    path = Path(row['path'])
    stat = path.stat()
    assert (stat.st_ino, stat.st_size, stat.st_mtime_ns) == (row['inode'], row['bytes'], row['mtime_ns'])
    model = row['retained_model']
    kept = Path(model['path']).stat()
    assert (kept.st_ino, kept.st_size, kept.st_mtime_ns) == (model['inode'], model['bytes'], model['mtime_ns'])
    assert Path(row['retained_metadata']).is_file()
    # Final live-use check does not print other users' commands or environments.
    for process in Path('/proc').iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        try:
            assert str(path) not in (process / 'maps').read_text(), f'Live mapping: {process.name}'
            for descriptor in (process / 'fd').iterdir():
                assert descriptor.resolve() != path, f'Live descriptor: {process.name}'
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    path.unlink()
    receipt['retired'].append({'path': str(path), 'bytes': row['bytes'],
                               'retained_model': model['path'], 'retained_metadata': row['retained_metadata']})
    receipt['free_after_retirement'] = shutil.disk_usage(root).free
    receipt['status'] = 'retired'
    (root / 'STORAGE_RETIREMENT.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({'retired_bytes': sum(r['bytes'] for r in receipt['retired']),
                  'free_bytes': receipt['free_after_retirement']}))
