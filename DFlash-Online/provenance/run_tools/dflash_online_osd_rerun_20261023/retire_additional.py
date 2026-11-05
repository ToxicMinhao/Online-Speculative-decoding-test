"""Retire the three audited completed-run Adam states, preserving other metadata."""
import gc
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch


def digest(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def equal(left, right):
    if isinstance(left, torch.Tensor):
        return left.dtype == right.dtype and torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return left.dtype == right.dtype and np.array_equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal(value, right[key]) for key, value in left.items())
    if isinstance(left, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right))
    return left == right


root = Path('/data/sources/minhao/runs/dflash_online_osd_rerun_20261023/storage')
plan = json.loads((root / 'ADDITIONAL_PLAN.json').read_text())
receipt_path = root / 'ADDITIONAL_RETIREMENT.json'
assert not receipt_path.exists()
metadata_root = root / 'retained_metadata'
metadata_root.mkdir()
receipt = {'status': 'preparing', 'started_unix': time.time(), 'removed': [],
           'free_before': shutil.disk_usage(root).free, 'metadata': [],
           'lost_capability': plan['lost_capability']}
for evidence in plan['compact_evidence']:
    assert digest(evidence['remote_path']) == evidence['sha256']
for row in plan['candidates']:
    path = Path(row['path'])
    stat = path.stat()
    assert not path.is_symlink() and path.resolve() == path
    assert (stat.st_size, stat.st_ino, stat.st_mtime_ns, stat.st_nlink) == (
        row['bytes'], row['inode'], row['mtime_ns'], row['nlink'])
    assert digest(path) == row['state_sha256_from_original_receipt']
    assert digest(row['retained_export_path']) == row['expected_export_sha256']
    state = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    assert state['includes_draft'] is False and 'optimizer' in state
    metadata = {key: value for key, value in state.items() if key != 'optimizer'}
    output = metadata_root / (row['checkpoint_id'] + '_without_optimizer.pt')
    torch.save(metadata, output)
    loaded = torch.load(output, map_location='cpu', weights_only=False)
    assert equal(metadata, loaded)
    receipt['metadata'].append({'source': str(path), 'retained': str(output),
                                'sha256': digest(output), 'all_nonoptimizer_fields_equal': True,
                                'retained_keys': list(metadata),
                                'retained_export': row['retained_export_path']})
    del state, metadata, loaded
    gc.collect()
receipt.update(status='retiring', verified_compact_files=len(plan['compact_evidence']))
receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
try:
    for row in plan['candidates']:
        Path(row['path']).unlink()
        receipt['removed'].append(row['path'])
        receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    receipt['status'] = 'completed'
except BaseException as error:
    receipt.update(status='partial_failure', error=repr(error))
    raise
finally:
    receipt.update(free_after=shutil.disk_usage(root).free, finished_unix=time.time())
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({key: receipt[key] for key in ('status', 'free_before', 'free_after')}, indent=2))
