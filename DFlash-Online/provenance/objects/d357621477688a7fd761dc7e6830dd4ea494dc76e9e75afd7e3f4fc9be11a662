"""Package completed compact evidence, without exporting model/optimizer tensors."""
import hashlib
import json
from pathlib import Path
import sys
import tarfile

root = Path(sys.argv[1])
if not (root/'EXIT_CODE').exists():
    raise RuntimeError('Wait for the diagnostic and postflight to finish')
files = [p for p in root.rglob('*') if p.is_file()
         and 'code' not in p.relative_to(root).parts
         and '__pycache__' not in p.parts
         and p.name not in ('upload.tar.gz','evidence.tar.gz','RETRIEVAL_MANIFEST.json')]
if any(p.suffix in ('.pt','.safetensors') for p in files):
    raise RuntimeError('Unexpected exported tensor artifact; inspect without transferring silently')
manifest = {}
for path in files:
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream,'sha256').hexdigest()
    manifest[str(path.relative_to(root))] = dict(bytes=path.stat().st_size,sha256=digest)
(root/'RETRIEVAL_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
with tarfile.open(root/'evidence.tar.gz','w:gz') as archive:
    for path in files+[root/'RETRIEVAL_MANIFEST.json']:
        archive.add(path,arcname=str(path.relative_to(root)))
print(json.dumps(dict(files=len(files),bytes=(root/'evidence.tar.gz').stat().st_size)))
