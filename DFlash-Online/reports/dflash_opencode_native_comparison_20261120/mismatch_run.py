from pathlib import Path
import os,subprocess,json,time,signal
root=Path('/data/sources/minhao/runs/dflash_opencode_native_comparison_20261120')
base=root/'diagnostic_001'
source=Path('/data/sources/minhao/opencode_native_source_20261120_003/DFlash-Online')
import sys
sys.path.insert(0,str(source))
from online_feedback.storage import write_json
from experiments.bounded_prefix_allocation.launch import stop_owned
from experiments.opencode_native_comparison.runtime import boundary
plan=json.loads((root/'sealed/PLAN_RETRY_001.json').read_text())
# Include the additional source allocation while preserving the original plan/clock.
plan['allocation_roots'].append(str(source.parent))
write_json(base/'DIAGNOSTIC_RUNTIME_PLAN.json',plan)
boundary(plan)
serving=Path(plan['worker_python']).parent.parent
env=dict(os.environ)
env.update(CUDA_VISIBLE_DEVICES=plan['devices']['B'],SGLANG_ENABLE_SPEC_V2='0',
    CONDA_PREFIX=str(serving),PATH=str(serving/'bin')+os.pathsep+env['PATH'],
    LD_LIBRARY_PATH=':'.join([str(serving/'lib/adaflash_cuda12'),str(serving/'lib'),env.get('LD_LIBRARY_PATH','')]),
    PYTHONPATH=str(source)+':'+str(source/'training')+':'+str(source/'inference'),
    CUDAHOSTCXX=env['CXX'],NVCC_CCBIN=env['CXX'],
    DFLASH_DIAGNOSTIC_COMMIT=json.loads((base/'SOURCE.json').read_text())['commit'])
start=time.time();receipts=[]
for arm in ('AR','F'):
    boundary(plan)
    cmd=[plan['worker_python'],'-u','-m','experiments.opencode_native_comparison.mismatch_diagnostic',
        '--plan',str(base/'DIAGNOSTIC_RUNTIME_PLAN.json'),'--arm',arm,'--output',str(base/arm)]
    with (base/(arm+'.log')).open('x') as log:
        process=subprocess.Popen(cmd,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        row=dict(arm=arm,pid=process.pid,started_epoch=time.time(),command=cmd);receipts.append(row)
        write_json(base/'PROCESSES.json',receipts)
        try:
            while process.poll() is None:
                if time.time()-start>1200:raise TimeoutError('Bounded diagnostic timeout')
                time.sleep(1)
        finally:
            stop_owned(process)
            row.update(exit_code=process.poll(),finished_epoch=time.time())
            write_json(base/'PROCESSES.json',receipts)
    if process.returncode:raise RuntimeError('Diagnostic stopped; no automatic retry: '+arm)
write_json(base/'COMPLETE.json',dict(status='completed',generations=2,training_steps=0,wall_s=time.time()-start,production_still_stopped=True))
