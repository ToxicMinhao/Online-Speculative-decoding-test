"""Install only reviewed source changes after checking the existing source seal."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

root=Path(__file__).parent
source=Path('/data/sources/minhao/DFlash-Online')
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*2**20), b''): h.update(block)
    return h.hexdigest()
changes=json.loads((root/'CHANGED_FILES.json').read_text())
seal=json.loads((root/'PREVIOUS_SEAL.json').read_text())['source_sha256']
free=shutil.disk_usage(root).free
assert free >= 8*2**30 + 128*2**20, 'Source-stage budget exceeds free space'
for name, expected in seal.items():
    assert digest(source/name)==expected, ('Source changed since reviewed R run', name)
for row in changes:
    p=source/row['path']
    assert (digest(p) if p.exists() else None)==row['before_sha256'], ('Unexpected source',row['path'])
staged=root/'staged'
staged.mkdir()
with tarfile.open(root/'SOURCE_UPDATE.tar.gz') as archive:
    archive.extractall(staged,filter='data')
for row in changes:
    assert digest(staged/row['path'])==row['after_sha256']
backup=root/'previous_source'
for row in changes:
    p=source/row['path']
    if p.exists():
        saved=backup/row['path']; saved.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,saved)
    p.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(staged/row['path'],p)
    assert digest(p)==row['after_sha256']
changed={row['path'] for row in changes}
for name, expected in seal.items():
    if name not in changed: assert digest(source/name)==expected
receipt=dict(status='passed',changed_files=changes,prior_source_files_checked=len(seal),
             unchanged_prior_source_files=sum(name not in changed for name in seal),
             backup=str(backup),free_before=free,free_after=shutil.disk_usage(root).free,
             gpu_work=0, scientific_training=0, generations=0, removed_paths=[])
(root/'INSTALL.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k!='changed_files'},indent=2))
