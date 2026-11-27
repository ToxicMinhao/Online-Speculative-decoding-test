"""Complete verified CPU packaging only, preserving the failed first attempt."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from experiments.continuation_repair_fullrun.compact import resume_verified_retirement
from experiments.continuation_repair_fullrun.contract import digest
from experiments.frontier_preservation_pilot.qualify import verify_source
from experiments.ok_coding.storage import identity

root=Path('/data/sources/minhao/runs/dflash_continuation_repair_fullrun_20261126/full_001')
plan_path=root/'sealed/PLAN.json'
plan=json.loads(plan_path.read_text())
if plan['binding']!=digest({k:v for k,v in plan.items() if k!='binding'}):
    raise ValueError('Original scientific plan identity differs')
verify_source(plan)
processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True).strip()
if processes:raise RuntimeError('Expected idle cards during CPU recovery')
if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU recovery must hide CUDA devices')
output=root/'infrastructure_repairs/packaging_001'
output.mkdir(parents=True,exist_ok=False)
started=time.time()
base=root/'campaign_001/arms/BASE'
paths=[root/'STATUS.json',root/'CLOCK.json',root/'WORK.json',root/'PROCESS_PID',root/'PROCESS_EXIT_CODE',base/'STATUS.json']
before={str(path):identity(path) for path in paths}
for n,path in enumerate(paths):shutil.copy2(path,output/f'original_{n:02d}_{path.name}')
status=json.loads((base/'STATUS.json').read_text())
if (status['events_completed'],status['global_step'],status['presentations'],status['phase'])!=(1,128,256,'idle_boundary'):
    raise ValueError('Only the completed first BASE boundary may resume')
marker=json.loads((base/'boundaries/event_001.json').read_text())
if marker['status']!='passed':raise ValueError('Missing native boundary validation')
for kind in ('state','weights'):
    if identity(marker[kind])!=marker[kind+'_identity']:raise ValueError('Retained endpoint changed: '+kind)
work=json.loads((root/'WORK.json').read_text())
if work['pending'] or work['completed']['production.adam']!=128 or work['completed']['production.response_forward']!=256:
    raise ValueError('Unreconciled physical work at the original boundary')
os.environ['PATH']=str(Path(plan['zstd']).parent)+os.pathsep+os.environ['PATH']
archive=root/'compact/BASE/event_001.tar.zst'
archive_before=identity(archive)
result=resume_verified_retirement(root,'BASE',1)
if result['status']!='completed' or identity(archive)!=archive_before:
    raise ValueError('Archive completion did not preserve verified archive bytes')
for path,pin in before.items():
    if identity(path)!=pin:raise ValueError('CPU recovery altered unrelated status or scientific work: '+path)
report=dict(status='passed',new_model_work=0,plan_binding=plan['binding'],execution_commit=plan['execution_commit'],
    actual_execution_commit=json.loads((root/'infrastructure_sources/repair_001/SOURCE.json').read_text())['commit'],
    state=marker['state_identity'],weights=marker['weights_identity'],global_step=128,presentations=256,
    next_event=2,original_receipts=before,archive_unchanged=archive_before,recovery=result['recovery'],
    unchanged_work_ledger=True,unchanged_original_clock=True,wall_s=time.time()-started)
with (output/'CPU_RECOVERY.json').open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
print(json.dumps({k:report[k] for k in ('status','new_model_work','global_step','presentations','next_event','wall_s')}))
