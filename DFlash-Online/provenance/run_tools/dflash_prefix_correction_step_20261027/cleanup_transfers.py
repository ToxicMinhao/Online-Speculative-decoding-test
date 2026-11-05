import hashlib,json,shutil
from pathlib import Path
root=Path('/data/sources/minhao/runs/dflash_prefix_correction_step_20261027')
expected={'upload.tar.gz': '1dcb8cd12672629bad177cf8fee54590b63d28ad46107675cbc9c72c7e4962b2', 'evidence.tar.gz': 'a648a52262737387189b87cf15732cb79ff0ab98da5164bcb253e856eae519ed'}
before=shutil.disk_usage(root).free
rows=[]
for name,digest in expected.items():
 p=root/name
 assert p.is_file() and not p.is_symlink() and hashlib.sha256(p.read_bytes()).hexdigest()==digest,name
 rows.append(dict(path=str(p),bytes=p.stat().st_size,sha256=digest,reason='Verified redundant transfer container; extracted source/compact evidence retained on both hosts',lost_capability='none'))
for row in rows:Path(row['path']).unlink()
record=dict(removed=rows,free_before_bytes=before,free_after_bytes=shutil.disk_usage(root).free)
(root/'STORAGE_MAINTENANCE.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
