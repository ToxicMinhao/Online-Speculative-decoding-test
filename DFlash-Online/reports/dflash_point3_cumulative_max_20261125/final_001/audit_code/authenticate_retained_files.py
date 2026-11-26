"""Read-only CPU authentication and resource inventory; no experiment imports."""
import hashlib,json,os,shutil,struct,subprocess,time
from pathlib import Path
start=time.time()
import argparse
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True,type=Path);root=parser.parse_args().root
plan=json.loads((root/'CONTRACT.json').read_text()); summary=json.loads((root/'evaluation/SUMMARY.json').read_text())
def identity(path):
 p=Path(path);h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(8<<20):h.update(b)
 return dict(bytes=p.stat().st_size,sha256=h.hexdigest())
def allocated(paths):
 seen=set();total=0
 for base in map(Path,paths):
  for p in (base,*base.rglob('*')):
   if p.is_symlink():continue
   s=p.stat();key=(s.st_dev,s.st_ino)
   if key not in seen:total+=s.st_blocks*512;seen.add(key)
 return total
def tensors(path):
 h=hashlib.sha256();elements=count=0
 with Path(path).open('rb') as f:
  n=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(n));base=8+n
  for name in sorted(set(header)-{'__metadata__'}):
   item=header[name];assert item['dtype']=='BF16'
   h.update(json.dumps(dict(name=name,shape=item['shape'],dtype='torch.bfloat16'),sort_keys=True).encode())
   a,b=item['data_offsets'];f.seek(base+a);remaining=b-a
   while remaining:
    data=f.read(min(8<<20,remaining));assert data;h.update(data);remaining-=len(data)
   count+=1;elements+=(b-a)//2
 return dict(sha256=h.hexdigest(),tensors=count,elements=elements)
checks={};exports={}
for arm in ('F_C_SUM','F_C_MAX'):
 s=json.loads((root/'arms'/arm/'STATUS.json').read_text())
 for key in ('checkpoint','recovery'):
  p=Path(s[key]);p=p/'model.safetensors' if p.is_dir() else p
  actual=identity(p);expected=s[key+'_identity'];assert actual==expected,(arm,key)
  checks[arm+'/'+key]=dict(path=str(p),identity=actual,matched_record=True)
  if key=='checkpoint':exports[arm]=p
control=json.loads(Path(plan['retained_C_status']).read_text())
exports['C']=Path(plan['retained_C_checkpoint'])/'model.safetensors'
for label,path,expected in [('parent_recovery',plan['parent_recovery'],plan['parent_recovery_identity']),('parent_export',plan['parent_export'],plan['parent_export_identity']),('control_recovery',control['recovery'],control['recovery_identity']),('control_export',exports['C'],control['checkpoint_identity'])]:
 actual=identity(path);assert actual==expected,label;checks[label]=dict(path=str(path),identity=actual,matched_record=True)
model_identities={}
for arm,path in exports.items():
 actual=tensors(path);expected=summary['parameter_identities'][arm]
 assert all(actual[k]==expected[k] for k in actual),arm
 model_identities[arm]=actual
source=Path(plan['source_manifest']);manifest=json.loads(source.read_text());assert manifest['commit']==plan['execution_commit'];assert identity(source)==plan['source_identity']
source_root=Path(plan['source_root'])
for rel,expected in manifest['files'].items():
 p=source_root/rel
 if expected.get('kind')=='symlink':assert p.is_symlink() and os.readlink(p)==expected['target'],rel
 else:assert not p.is_symlink() and identity(p)==expected,rel
compiler=allocated(plan['compiler_baseline']['directories'].values());growth=max(0,compiler-plan['compiler_baseline']['allocated_bytes'])
artifacts=allocated([root,source.parent]);clock=json.loads((root/'CLOCK.json').read_text())
assert clock['status']=='completed'
result=dict(status='passed',epoch=time.time(),cpu_wall_s=time.time()-start,gpu_work=0,experimental_modules_imported=False,weights_loaded_into_framework=False,
 source=dict(commit=manifest['commit'],files_verified=len(manifest['files']),manifest_identity=identity(source)),protected_files=checks,
 tensor_content_identities=model_identities,three_distinct_serving_models=len({x['sha256'] for x in model_identities.values()})==3,
 resource=dict(remote_artifact_allocated_bytes=artifacts,compiler_baseline_allocated_bytes=plan['compiler_baseline']['allocated_bytes'],compiler_current_allocated_bytes=compiler,compiler_growth_bytes=growth,charged_remote_bytes=artifacts+growth,free_bytes=shutil.disk_usage(root).free,reserve_bytes=400_000_000_000),
 gpu_inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader'],text=True),gpu_processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True),
 limitations=['Postflight CPU time is separate from the completed GPU clock.','Current storage and prior boundary samples are not a continuous disk trace.','Feedback tensor byte equality uses authenticated launch/runtime manifests; this postflight does not reread all feedback tensors.','Recovery file hashes authenticated in place; exact internal state/export equality is additionally supported by runtime receipts, not a fresh optimizer load.'])
assert result['resource']['free_bytes']>=400_000_000_000
print(json.dumps(result,indent=2))
