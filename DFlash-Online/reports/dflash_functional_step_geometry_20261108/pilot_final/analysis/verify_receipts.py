"""Authenticate consumed final evidence and check the completed supervisor receipt."""
from pathlib import Path
from collections import Counter
from datetime import datetime
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
EVIDENCE=ROOT/'evidence/pilot'
def read(p):return json.loads(p.read_text())
def identity(p):
 b=p.read_bytes();return {'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
manifest=read(ROOT/'evidence/SNAPSHOT_MANIFEST.json')
recorded={(EVIDENCE/r['path']).resolve():{'bytes':r['bytes'],'sha256':r['sha256']} for r in manifest['files']}
inputs=read(ROOT/'endpoint_review/INPUT_IDENTITIES.json')
checked=0
for path,saved in inputs.items():
 p=Path(path).resolve()
 if p.is_relative_to(EVIDENCE.resolve()):
  assert p in recorded,p
  assert identity(p)==saved==recorded[p],p
  checked+=1
assert checked>=1550,checked
files=['EXIT_CODE','END_UTC','POSTFLIGHT.json','campaign_001/STATUS.json']
for name in files:
 p=EVIDENCE/name;assert identity(p)==recorded[p.resolve()],p
assert (EVIDENCE/'EXIT_CODE').read_text().strip()=='0'
post=read(EVIDENCE/'POSTFLIGHT.json');assert post['status']=='passed' and post['postflight'] is True
assert post['execution_commit']=='6440e8ff5af1a845f830d65c884f6711d7e4a687' and post['sealed_files_checked']==15122
status=read(EVIDENCE/'campaign_001/STATUS.json');assert status['status']=='completed'
assert status['attempted_task'] is None and status['automatic_retry'] is False
assert status['execution_source']['commit']==post['execution_commit'] and status['execution_source']['dirty'] is False
expected=[]
for ds in ['gsm8k','mathqa']:
 for ev in [5,6]:
  expected.append({'kind':'collect_shared','dataset':ds,'event':ev,'producer':'KPLUS','producer_event':ev-1})
  for arm in ['KPLUS','FSG','SCALAR']:expected.append({'kind':'train_arm','dataset':ds,'event':ev,'arm':arm})
 expected.append({'kind':'evaluate_dataset','dataset':ds,'event':6})
assert status['completed_tasks']==expected
result={'status':'passed','authenticated_consumed_snapshot_files':checked,'supervisor_exit':0,'completed_tasks':len(expected),'task_counts':dict(Counter(x['kind'] for x in expected)),'execution_commit':post['execution_commit'],'source_postflight_files':post['sealed_files_checked'],'postflight_passed':True,'end_utc':(EVIDENCE/'END_UTC').read_text().strip(),'automatic_retry':False,'live_attempt':None,'new_model_work':0,'scope':'Control-plane and consumed-byte checks; native execution is not rerun.'}
(ROOT/'endpoint_review/COMPLETION_AND_AUTHENTICITY.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
