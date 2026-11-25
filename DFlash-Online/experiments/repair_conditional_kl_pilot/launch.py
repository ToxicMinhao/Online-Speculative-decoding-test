"""Run the resource-admitted bounded qualification; never auto-retry a phase."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

from online_feedback.storage import write_json
from experiments.kplus_full_gsm8k.preflight import timing_barrier
from .resources import read, initialize, boundary, clock_status


def stop_owned(process):
    import psutil
    if process.poll() is not None:
        return
    descendants=psutil.Process(process.pid).children(recursive=True)
    for child in reversed(descendants):
        try: child.terminate()
        except psutil.NoSuchProcess: pass
    process.terminate()
    _,alive=psutil.wait_procs(descendants,timeout=10)
    for child in alive:
        try: child.kill()
        except psutil.NoSuchProcess: pass
    try: process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill();process.wait()


def launch(plan_path):
    from .resources import verify_inputs
    plan=read(plan_path);root=Path(plan['root'])
    verify_inputs(plan)
    if plan['execution_mode']!='qualification_only' or plan['production_admitted']:
        raise ValueError('This launch is only the resource-admitted bounded qualification')
    if read(root/'CPU_PREFLIGHT.json')['status']!='passed':
        raise ValueError('CPU/source/input/resource preflight required')
    timing_barrier()
    initialize(plan)
    status=dict(status='running',execution_commit=plan['execution_commit'],phases=[],automatic_retry=False)
    path=root/'LAUNCH.json';write_json(path,status)
    try:
        for arm,card in (('BASE','B'),('PAIR','A')):
            boundary(plan);timing_barrier()
            environment=dict(os.environ,CUDA_VISIBLE_DEVICES=plan['devices'][card])
            logfile=root/(arm+'_qualification.log')
            with logfile.open('w') as log:
                process=subprocess.Popen([plan['training_python'],'-u','-m',
                    'experiments.repair_conditional_kl_pilot.entry','--plan',str(plan_path),
                    '--phase','qualify','--arm',arm],cwd=plan['source_root'],env=environment,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                slot=dict(arm=arm,pid=process.pid,started_epoch=time.time(),log=str(logfile))
                status['phases'].append(slot);write_json(path,status)
                try:
                    while process.poll() is None:
                        boundary(plan)
                        slot['elapsed_s']=time.time()-slot['started_epoch']
                        write_json(path,status)
                        time.sleep(5)
                finally:
                    if process.poll() is None: stop_owned(process)
                slot.update(exit_code=process.returncode,elapsed_s=time.time()-slot['started_epoch'])
                write_json(path,status)
            if process.returncode:
                raise RuntimeError(arm+' qualification failed; preserve evidence, no retry')
            if read(root/'qualification_inputs'/arm/'STATUS.json')['status']!='passed':
                raise ValueError('Returned phase did not pass its native checks')
        status['status']='completed'
        write_json(root/'QUALIFICATION.json',dict(status='passed',arms=['BASE','PAIR'],
            optimizer_steps=2,startup_optimizer_steps=0,production_started=False,
            production_blocker=plan['production_blocker'],execution_commit=plan['execution_commit']))
    except BaseException as error:
        status.update(status='failed',error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        status['clock']=clock_status(root)
        write_json(path,status)
    return status


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',required=True)
    launch(parser.parse_args().plan)
