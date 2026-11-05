"""Read-only provenance and completion audit, run after serving has stopped."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def run(root):
    root = Path(root)
    if not (root / 'EXIT_CODE').is_file():
        raise RuntimeError('Wait for the supervisor to finish before artifact auditing')
    seal = json.loads((root / 'SEAL.json').read_text())
    retained = json.loads((root / 'RETAINED_ARTIFACTS.json').read_text())
    checked, errors = {}, []
    for category, base in (
        ('source_sha256', Path('/data/sources/minhao/DFlash-Online')),
        ('execution_sha256', root), ('dependency_sha256', Path('/')),
    ):
        checked[category] = 0
        for name, expected in seal[category].items():
            path = base / name
            if not path.is_file() or digest(path) != expected:
                errors.append({'category': category, 'path': str(path)})
            checked[category] += 1
    checked['retained_artifacts'] = 0
    for name, expected in retained.items():
        path = Path(name)
        if not path.is_file():
            errors.append({'category': 'retained_missing', 'path': name})
        else:
            stat = path.stat()
            if (stat.st_size != expected['bytes'] or stat.st_mtime_ns != expected['mtime_ns']
                    or digest(path) != expected['sha256']):
                errors.append({'category': 'retained_changed', 'path': name})
        checked['retained_artifacts'] += 1
    result = dict(status='passed' if not errors else 'failed', checked=checked,
                  errors=errors, free_bytes=shutil.disk_usage(root).free,
                  supervisor_exit_code=int((root / 'EXIT_CODE').read_text()),
                  checked_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    (root / 'POSTFLIGHT.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
    return int(bool(errors))


if __name__ == '__main__':
    sys.exit(run(sys.argv[1]))
