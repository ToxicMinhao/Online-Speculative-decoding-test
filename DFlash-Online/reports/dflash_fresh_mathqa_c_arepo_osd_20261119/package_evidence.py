"""Package completed, audited private evidence; never execute a model or delete data."""
from pathlib import Path
import argparse, hashlib, json, os, shutil, stat, tarfile, time

def read(p): return json.loads(p.read_text())
def ident(p):
    with p.open('rb') as s: h=hashlib.file_digest(s,'sha256').hexdigest()
    return {'bytes':p.stat().st_size,'sha256':h}
def save(p,x): p.write_text(json.dumps(x,indent=2)+'\n')
p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('destination');a=p.parse_args()
r=Path(a.root).resolve();d=Path(a.destination).resolve()
assert not d.exists(), 'Never overwrite an existing delivery'
assert r not in d.parents, 'Archive must be outside the source tree'
assert read(r/'LAUNCH.json')['status']=='completed'
assert (r/'EXIT_CODE').read_text().strip()=='0'
assert read(r/'CLOCK.json')['active']=={}
assert read(r/'final_audit/FINAL_EVIDENCE_AUDIT.json')['status']=='passed_independent_completed_evidence_audit'
assert read(r/'final_audit/AUDIT_EXECUTION.json')['exit_code']==0
before=shutil.disk_usage(r).free
assert before>400_000_000_000+6_000_000_000
began=time.time();d.mkdir()
files={}
for folder,dirs,names in os.walk(r):
    for name in names:
        f=Path(folder)/name;s=f.lstat()
        if not stat.S_ISREG(s.st_mode):raise ValueError('Nonregular evidence: '+str(f))
        files[f.relative_to(r).as_posix()]={**ident(f),'allocated_bytes':s.st_blocks*512}
save(d/'MEMBER_IDENTITIES.json',{'execution_commit':read(r/'PLAN.json')['execution_commit'],'source_root':str(r),'members':files,'regular_files':len(files),'logical_bytes':sum(x['bytes'] for x in files.values()),'allocated_bytes':sum(x['allocated_bytes'] for x in files.values())})
archive=d/'FRESH_MATHQA_COMPACT.tar.gz'
with tarfile.open(archive,'w:gz',compresslevel=3,format=tarfile.PAX_FORMAT,dereference=True) as t:
    for rel in sorted(files):t.add(r/rel,arcname=rel,recursive=False)
changed=[rel for rel,row in files.items() if ident(r/rel)!={k:row[k] for k in ('bytes','sha256')}]
if changed:raise ValueError('Source changed while packaging: '+repr(changed))
receipt={'status':'packaged_source_stable','execution_commit':read(r/'PLAN.json')['execution_commit'],'source_root':str(r),'archive':str(archive),'archive_identity':ident(archive),'manifest_identity':ident(d/'MEMBER_IDENTITIES.json'),'regular_files':len(files),'logical_bytes':sum(x['bytes'] for x in files.values()),'source_allocated_bytes':sum(x['allocated_bytes'] for x in files.values()),'started_epoch':began,'wall_s':time.time()-began,'free_before_bytes':before,'free_after_bytes':shutil.disk_usage(r).free,'deleted_paths':[],'lost_capabilities':[],'delivery_verification_pending':True}
save(d/'ARCHIVE.json',receipt);print(json.dumps(receipt),flush=True)
