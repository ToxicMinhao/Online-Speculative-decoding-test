"""After the job exits, preserve compact evidence and audit retained artifacts."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile
from datetime import datetime, timezone

root = Path(sys.argv[1]).resolve()
if not (root / 'EXIT_CODE').is_file():
    raise RuntimeError('Do not audit or package artifacts while the campaign runs')
code = Path('/data/sources/minhao/DFlash-Online')
seal = json.loads((root / 'SEAL.json').read_text())
retained = json.loads((root / 'RETAINED_ARTIFACTS.json').read_text())
def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()
changes = []
for name, expected in seal['source_sha256'].items():
    if not (code / name).is_file() or digest(code / name) != expected:
        changes.append({'kind': 'source', 'path': name})
for name, expected in retained.items():
    path = Path(name)
    if not path.is_file() or path.stat().st_size != expected['bytes'] or digest(path) != expected['sha256']:
        changes.append({'kind': 'retained_O', 'path': name})
for name, expected in seal['dependency_sha256'].items():
    if not Path(name).is_file() or digest(Path(name)) != expected:
        changes.append({'kind': 'qualification_dependency', 'path': name})
audit = dict(status='passed' if not changes else 'failed', source_files=len(seal['source_sha256']),
             retained_artifacts=len(retained), qualification_dependencies=len(seal['dependency_sha256']),
             changes=changes, free_bytes=shutil.disk_usage(root).free,
             finished_check_utc=datetime.now(timezone.utc).isoformat(),
             retirement_performed=False, exit_code=int((root / 'EXIT_CODE').read_text()))
(root / 'FINAL_RETENTION_AUDIT.json').write_text(json.dumps(audit, indent=2) + '\n')
allowed = {'.json', '.jsonl', '.yaml', '.txt', '.md', '.log', '.utc', '.csv', '.py', '.sh', '.patch'}
files = sorted(p for p in root.rglob('*') if p.is_file() and not p.is_symlink()
               and (p.suffix in allowed or p.name == 'EXIT_CODE'))
with tarfile.open(root / 'COMPACT_EVIDENCE.tar.gz', 'w:gz') as archive:
    for path in files:
        archive.add(path, arcname=str(path.relative_to(root)), recursive=False)
print(json.dumps({**audit, 'compact_files':len(files),
                  'archive_bytes':(root/'COMPACT_EVIDENCE.tar.gz').stat().st_size}))
