"""Preserve selected superseded states' weights and metadata; no deletion here."""
from pathlib import Path
import gc
import hashlib
import json
import os
import shutil

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path('/data/sources/minhao/runs/dflash_online_gpu_check_20261022')
OLD = Path('/data/sources/minhao/runs/dflash_online_foundation_20261022/qualification_003/trainer')
PER_TURN = Path('/data/sources/minhao/EAGLE_SPEC/research/dflash_per_turn_20261015/campaign_001/attempt_001/wildchat_00/native/wildchat_00')
ROOT.mkdir(exist_ok=True)
META = ROOT / 'metadata'
META.mkdir(exist_ok=True)


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def save_metadata(name, data):
    path = META / name
    with path.open('xb') as stream:
        torch.save(data, stream)
        stream.flush()
        os.fsync(stream.fileno())
    loaded = torch.load(path, map_location='cpu', weights_only=False)
    assert equal(data, loaded), name
    return str(path)


def strip_moments(optimizer):
    removed = []
    for parameter, state in optimizer['state'].items():
        for key in ('exp_avg', 'exp_avg_sq'):
            tensor = state.pop(key)
            removed.append({'parameter': str(parameter), 'field': key,
                            'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
                            'bytes': tensor.numel() * tensor.element_size()})
    return removed


def describe(path):
    stat = path.stat()
    return {'path': str(path), 'bytes': stat.st_size, 'allocated_bytes': stat.st_blocks * 512,
            'inode': stat.st_ino, 'mtime_ns': stat.st_mtime_ns}


receipt = {'free_before': shutil.disk_usage(ROOT).free, 'prepared': [],
           'reason': 'Fresh refactored online-loop GPU qualification; old state schemas superseded',
           'lost': 'Exact Adam/FP32-master continuation at the two old loop states and exact Adam continuation at completed WildChat00 D1. Weights/results remain; removed tensors are not archived.'}

for name, label in [('training_state.pt', 'old_committed'), ('training_state.pending.pt', 'old_uncommitted')]:
    path = OLD / name
    row = describe(path)
    data = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    assert data['event_schema'] == 'specforge_event_trainer_v1'
    weights = data.pop('draft_state_dict')
    if label == 'old_committed':
        model_path = OLD / 'events/event_1/export/model.safetensors'
    else:
        weight_root = ROOT / 'retained_old_uncommitted_weights'
        weight_root.mkdir(exist_ok=True)
        model_path = weight_root / 'model.safetensors'
        save_file(weights, str(model_path))
        shutil.copyfile(OLD / 'events/event_1/export/config.json', weight_root / 'config.json')
    with safe_open(model_path, framework='pt', device='cpu') as model:
        assert set(model.keys()) == set(weights)
        assert all(torch.equal(model.get_tensor(key), value) for key, value in weights.items())
    optimizer = data['backend']['optimizer']
    row['removed_moments'] = strip_moments(optimizer['optimizer_state_dict'])
    masters = optimizer.pop('fp32_params')
    row['removed_master_tensor_count'] = len(masters)
    row['retained_metadata'] = save_metadata(label + '.pt', data)
    row['retained_model'] = describe(model_path)
    row['all_weight_tensors_equal'] = len(weights)
    row['global_step'] = data['global_step']
    receipt['prepared'].append(row)
    print(label + ': weights and metadata verified', flush=True)
    del data, weights, optimizer, masters
    gc.collect()

path = PER_TURN / 'training_worker/states/D1.pt'
row = describe(path)
update = json.loads((PER_TURN / 'training_worker/UPDATE_1.json').read_text())
assert update['status'] == 'passed'
data = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
assert data['includes_draft'] is False
assert data['active_checkpoint_sha256'] == update['checkpoint']['checkpoint_identity_sha256']
model_path = PER_TURN / 'checkpoints/D1/model.safetensors'
# One-time historical-artifact verification before retirement, not runtime hashing.
with model_path.open('rb') as stream:
    assert hashlib.file_digest(stream, 'sha256').hexdigest() == update['checkpoint']['files']['model.safetensors']['sha256']
row['removed_moments'] = strip_moments(data['optimizer'])
row['retained_metadata'] = save_metadata('completed_wildchat00_D1.pt', data)
row['retained_model'] = describe(model_path)
row['retained_result'] = str(PER_TURN / 'training_worker/UPDATE_1.json')
receipt['prepared'].append(row)
receipt['free_after_prepare'] = shutil.disk_usage(ROOT).free
receipt['potential_free_after_retirement'] = receipt['free_after_prepare'] + sum(r['allocated_bytes'] for r in receipt['prepared'])
receipt['status'] = 'prepared_no_deletion'
(ROOT / 'PREPARED.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({key: receipt[key] for key in ('status', 'free_after_prepare', 'potential_free_after_retirement')}), flush=True)
