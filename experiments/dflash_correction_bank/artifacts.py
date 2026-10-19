"""Immutable snapshots with content-addressed arrays, avoiding repeated Fisher copies."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def binding(path):
    path = Path(path).resolve()
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def checked(value):
    path = Path(value['path']).resolve()
    if path.stat().st_size != value['bytes'] or sha(path) != value['sha256']:
        raise ValueError('Artifact content changed: ' + str(path))
    return path


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError('Immutable artifact already exists: ' + str(path))
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(value, f, indent=2, sort_keys=True, allow_nan=False); f.write('\n')
            f.flush(); os.fsync(f.fileno())
        # Hard link is atomic and does not replace an existing destination.
        os.link(temporary, path)
    finally:
        os.unlink(temporary)
    return binding(path)


def save_snapshot(value, path, *, array_root=None):
    """Save arrays once under a shared per-episode directory; no lossy casting.

    Hashes include dtype/shape and are checked again on load. Snapshot roots are
    self-describing; transfer the JSON and its shared arrays together. The array
    content is immutable even when the caller mutates its original ndarray later.
    """
    path = Path(path).resolve()
    root = Path(array_root).resolve() if array_root else path.parent / 'arrays'
    root.mkdir(parents=True, exist_ok=True)

    def encode(x):
        if dataclasses.is_dataclass(x):
            return encode({f.name: getattr(x, f.name) for f in dataclasses.fields(x)})
        if isinstance(x, np.ndarray):
            a = np.ascontiguousarray(x)
            if a.dtype.hasobject:
                raise ValueError('Object arrays are not an evidence format')
            h = hashlib.sha256(str((a.dtype.str, a.shape)).encode())
            h.update(memoryview(a.reshape(-1)).cast('B')); identity = h.hexdigest()
            destination = root / (identity + '.npy')
            if not destination.exists():
                fd, temporary = tempfile.mkstemp(prefix='.array_', dir=root)
                try:
                    with os.fdopen(fd, 'wb') as f:
                        np.save(f, a, allow_pickle=False); f.flush(); os.fsync(f.fileno())
                    try:
                        os.link(temporary, destination)
                    except FileExistsError:
                        pass
                finally:
                    os.unlink(temporary)
            else:
                previous = np.load(destination, allow_pickle=False, mmap_mode='r')
                if previous.dtype != a.dtype or previous.shape != a.shape or not np.array_equal(previous, a):
                    raise ValueError('Existing content-addressed array differs')
            return {'__array__': binding(destination), 'dtype': a.dtype.str, 'shape': list(a.shape)}
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, dict):
            return {str(k): encode(v) for k, v in x.items()}
        if isinstance(x, (tuple, list)):
            return [encode(v) for v in x]
        if isinstance(x, Path):
            return str(x)
        return x

    return atomic_json(path, {'schema': 'correction_bank_snapshot_v1', 'value': encode(value)})


def load_snapshot(reference):
    document = json.loads(checked(reference).read_text())
    if document.get('schema') != 'correction_bank_snapshot_v1':
        raise ValueError('Unknown bank snapshot schema')

    def decode(x):
        if isinstance(x, dict) and '__array__' in x:
            a = np.load(checked(x['__array__']), allow_pickle=False, mmap_mode='r')
            if a.dtype.str != x['dtype'] or list(a.shape) != x['shape']:
                raise ValueError('Array schema changed')
            return a
        if isinstance(x, dict):
            return {k: decode(v) for k, v in x.items()}
        if isinstance(x, list):
            return [decode(v) for v in x]
        return x
    return decode(document['value'])
