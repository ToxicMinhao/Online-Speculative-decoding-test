"""Verify the seal, pinned runtime and storage before the first generation."""

import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

from online_feedback.storage import write_json
from experiments.osd_components.prepare import verify_native_runtime


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify(root, *, postflight=False):
    root = Path(root)
    started = time.perf_counter()
    seal = json.loads((root / 'sealed/SEAL.json').read_text())
    errors = []
    for directory, expected in seal['exact_file_sets'].items():
        actual = sorted(p.name for p in Path(directory).iterdir() if p.is_file())
        if actual != expected:
            errors.append(directory + ': exact file membership')
    for path, identity in seal['files'].items():
        p = Path(path)
        if not p.is_file() or p.stat().st_size != identity['bytes'] or digest(p) != identity['sha256']:
            errors.append(path)
    if errors:
        write_json(root / ('POSTFLIGHT.json' if postflight else 'PREFLIGHT.json'),
                   dict(status='failed', changed_files=errors))
        raise ValueError('Sealed files changed: ' + str(errors))
    free = shutil.disk_usage(root).free
    required = seal['storage']['reserve_bytes'] + (0 if postflight else seal['storage']['peak_new_bytes'])
    if free < required:
        raise RuntimeError(f'Storage admission failed: {free} < {required}')
    runtime = None
    if not postflight:
        plan = json.loads((root / 'sealed/PLAN.json').read_text())
        pin = json.loads((root / 'sealed/RUNTIME_PIN.json').read_text())
        runtime = verify_native_runtime(plan['worker_python'], pin)
    receipt = dict(status='passed', files_checked=len(seal['files']), free_bytes=free,
                   required_bytes=required, wall_s=time.perf_counter() - started, runtime=runtime)
    write_json(root / ('POSTFLIGHT.json' if postflight else 'PREFLIGHT.json'), receipt)
    return receipt


if __name__ == '__main__':
    print(json.dumps(verify(sys.argv[1], postflight='--postflight' in sys.argv[2:])))
