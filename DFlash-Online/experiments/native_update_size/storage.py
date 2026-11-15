"""One allocation budget spanning CPU reconciliation and the entire campaign."""
import json
import os
from pathlib import Path
import shutil
import time

CAP_BYTES = 110_000_000_000
RESERVE_BYTES = 400_000_000_000


def allocated_files(roots):
    """Count shared inodes once, without following directory symlinks."""
    rows = {}
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for parent, directories, files in os.walk(root, followlinks=False):
            directories[:] = [name for name in directories if not Path(parent, name).is_symlink()]
            for name in files:
                path = Path(parent, name)
                try:
                    stat = path.lstat()
                except FileNotFoundError:
                    continue  # A live cache may release its own temporary file.
                key = f'{stat.st_dev}:{stat.st_ino}'
                rows[key] = max(rows.get(key, 0), stat.st_blocks * 512)
    return rows


def initialize(plan):
    path = Path(plan['allocation_baseline'])
    if path.exists():
        return json.loads(path.read_text())
    baseline = dict(roots=plan['allocation_roots'], files=allocated_files(plan['allocation_roots']),
                    free_bytes=shutil.disk_usage(plan['root']).free, created_at=time.time())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(baseline, indent=2) + '\n')
    return baseline


def check_write(plan, requested_bytes, phase):
    """Admit a projected allocation before writing; preserve every measurement."""
    baseline = initialize(plan)
    current = allocated_files(plan['allocation_roots'])
    growth = sum(max(0, size - baseline['files'].get(key, 0)) for key, size in current.items())
    free = shutil.disk_usage(plan['root']).free
    receipt = dict(phase=phase, at=time.time(), new_allocated_bytes=growth,
                   requested_bytes=int(requested_bytes), projected_new_bytes=growth+int(requested_bytes),
                   free_bytes=free, remaining_free_bytes=free-int(requested_bytes),
                   cap_bytes=CAP_BYTES, reserve_bytes=RESERVE_BYTES)
    receipt['status'] = 'passed' if receipt['projected_new_bytes'] <= CAP_BYTES and receipt['remaining_free_bytes'] >= RESERVE_BYTES else 'failed'
    with Path(plan['root'], 'ALLOCATION.jsonl').open('a') as stream:
        stream.write(json.dumps(receipt) + '\n'); stream.flush(); os.fsync(stream.fileno())
    if receipt['status'] != 'passed':
        raise RuntimeError('Campaign allocation or shared free-space reserve would be exceeded')
    return receipt
