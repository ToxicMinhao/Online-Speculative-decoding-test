"""Execute only the reviewed list after compact evidence and weights verify."""
import gc
import hashlib
import json
from pathlib import Path
import shutil
import time

import torch
from safetensors.torch import load_file

root = Path('/data/sources/minhao/runs/dflash_online_osd_rerun_20261023/storage')
plan = json.loads((root / 'PLAN.json').read_text())
verified = json.loads((root / 'FENRIR_VERIFIED.json').read_text())
assert verified['status'] == 'verified'
assert hashlib.sha256((root / 'compact_evidence.tar.gz').read_bytes()).hexdigest() == verified['archive_sha256']
assert not (root / 'RETIREMENT.json').exists()

weight_checks = []
states = [row for row in plan['retire'] if row['kind'] == 'optimizer_state']
for row, model_path in zip(states, plan['retained_model_exports']):
    state = torch.load(row['path'], map_location='cpu', weights_only=False, mmap=True)
    saved = state['draft_state_dict']
    exported = load_file(model_path, device='cpu')
    assert set(saved) == set(exported), (set(saved) - set(exported), set(exported) - set(saved))
    for name, value in exported.items():
        assert saved[name].dtype == value.dtype and torch.equal(saved[name], value), name
    weight_checks.append({'state': row['path'], 'retained_export': model_path,
                          'tensors': len(saved), 'all_equal': True})
    del state, saved, exported
    gc.collect()

unique_inodes = {}
for row in plan['retire']:
    path = Path(row['path'])
    stat = path.stat()
    assert not path.is_symlink() and path.resolve() == path
    assert (stat.st_size, stat.st_ino, stat.st_mtime_ns) == (row['bytes'], row['inode'], row['mtime_ns'])
    identity = (stat.st_dev, stat.st_ino)
    entry = unique_inodes.setdefault(identity, {'bytes': stat.st_size, 'links': stat.st_nlink, 'selected_links': 0})
    entry['selected_links'] += 1
for row in plan['evidence']:
    path = Path(row['path'])
    assert path.is_file() and path.stat().st_size == row['bytes']
receipt = {**plan, 'status': 'retiring', 'weight_checks': weight_checks,
           'fenrir_verification': verified, 'actual_free_before': shutil.disk_usage(root).free,
           'selected_path_bytes': plan['total_bytes'],
           'unique_file_bytes_unlinked': sum(row['bytes'] for row in unique_inodes.values()
                                           if row['links'] == row['selected_links']),
           'removed': [], 'started_unix': time.time()}
receipt_path = root / 'RETIREMENT.json'
receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
try:
    for row in plan['retire']:
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
print(json.dumps({key: receipt[key] for key in ('status', 'actual_free_before', 'free_after', 'total_bytes')}, indent=2))
