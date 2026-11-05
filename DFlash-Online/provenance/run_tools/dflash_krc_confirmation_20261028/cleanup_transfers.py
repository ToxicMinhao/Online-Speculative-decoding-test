"""Retire only verified redundant transfer containers after completion."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
root=Path(sys.argv[1])
if (root/'EXIT_CODE').read_text().strip()!='0':
    raise RuntimeError('Require successful completed supervision')
manifest=json.loads((root/'RETRIEVAL_MANIFEST.json').read_text())
for relative,expected in manifest.items():
    path=root/relative
    with path.open('rb') as stream:
        digest=hashlib.file_digest(stream,'sha256').hexdigest()
    if digest!=expected['sha256'] or path.stat().st_size!=expected['bytes']:
        raise ValueError('Original compact evidence changed: '+relative)
before=shutil.disk_usage(root).free
removed=[]
for name in ('new_source.tar.gz','preparation_bundle.tar.gz','compact_evidence.tar.gz'):
    path=root/name
    removed.append(dict(path=str(path),bytes=path.stat().st_size,
        reason='Redundant transfer container; extracted source/inputs/raw originals and verified Fenrir copy retained. No model or restart state retired.'))
    path.unlink()
receipt=dict(before_free_bytes=before,after_free_bytes=shutil.disk_usage(root).free,
    verified_original_files=len(manifest),removed=removed,lost_replay_or_resume_capability='none',
    retained_remote_root=str(root),retained_fenrir_root='/home/minhao2/EAGLE_SPEC/research/dflash_krc_confirmation_20261028/raw',
    retained_source='/data/sources/minhao/DFlash-Online/experiments/krc_confirmation')
(root/'TRANSFER_CLEANUP.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
