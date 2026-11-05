"""Create a compact evidence archive only after the supervisor has finished."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import sys

root = Path(sys.argv[1]).resolve()
if not (root / 'EXIT_CODE').is_file():
    raise RuntimeError('Do not transfer or inventory artifacts during scientific execution')
allowed = {'.json', '.jsonl', '.csv', '.md', '.txt', '.log', '.yaml', '.yml', '.sh', '.py', '.utc'}
paths = sorted(p for p in root.rglob('*') if p.is_file()
               and (p.suffix.lower() in allowed or p.name in {'EXIT_CODE', 'SUPERVISOR_PID'})
               and '__pycache__' not in p.parts and p.name != 'RETRIEVAL_MANIFEST.json')
size = sum(p.stat().st_size for p in paths)
if shutil.disk_usage(root).free < 8 * 2**30 + 2 * size:
    raise RuntimeError('Insufficient compact staging allowance plus original reserve')
manifest = {}
for p in paths:
    with p.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    manifest[str(p.relative_to(root))] = {'bytes': p.stat().st_size, 'sha256': digest}
receipt = root / 'RETRIEVAL_MANIFEST.json'
receipt.write_text(json.dumps({'files': manifest, 'total_bytes': size}, indent=2) + '\n')
archive = root / 'compact_evidence.tar.gz'
with tarfile.open(archive, 'w:gz') as tar:
    for p in paths + [receipt]:
        tar.add(p, arcname=p.relative_to(root), recursive=False)
print(json.dumps({'files': len(paths), 'raw_bytes': size, 'archive_bytes': archive.stat().st_size,
                  'archive': str(archive)}))
