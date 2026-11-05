"""Package completed-run text evidence without model/optimizer tensors."""
import hashlib
import json
from pathlib import Path
import sys
import tarfile

root=Path(sys.argv[1])
if not (root/'EXIT_CODE').is_file():
    raise RuntimeError('Wait for supervisor completion before retrieval')
allowed={'.json','.jsonl','.csv','.log','.txt','.md','.yaml','.py','.sh','.utc'}
files=[p for p in root.rglob('*') if p.is_file() and not p.is_symlink()
       and (p.suffix in allowed or p.name in {'PID','EXIT_CODE'})
       and p.name!='RETRIEVAL_MANIFEST.json']
manifest={}
for p in sorted(files):
    with p.open('rb') as stream:
        digest=hashlib.file_digest(stream,'sha256').hexdigest()
    manifest[str(p.relative_to(root))]={'bytes':p.stat().st_size,'sha256':digest}
receipt=root/'RETRIEVAL_MANIFEST.json'
receipt.write_text(json.dumps(manifest,indent=2)+'\n')
archive=root/'compact_evidence.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
    for p in sorted(files)+[receipt]:
        tar.add(p,arcname=str(p.relative_to(root)),recursive=False)
print(json.dumps({'files':len(files),'bytes':sum(x['bytes'] for x in manifest.values()),'archive_bytes':archive.stat().st_size}))
