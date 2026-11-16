"""Qualify once, then execute the fixed serial C7 → feedback → C8 → final chain."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

from online_feedback.storage import write_json
from experiments.persistent_prefix_withholding.launch import child_environment, check_compilers
from .runtime import initialize_work, read, clock_status, boundary, require_time_for, TIME_CAP_S

TASKS = ('C7','C_collect','C8','final')


def stop_owned(process):
    """Stop only this launcher's process group and its own worker descendants."""
    if process.poll() is not None:
        return
    import psutil
    try:
        descendants = psutil.Process(process.pid).children(recursive=True)
    except psutil.NoSuchProcess:
        descendants = []
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        for child in reversed(descendants):
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def run_children(plan_path, plan, jobs):
    """Independent qualification may overlap; every final timing task is alone."""
    processes, logs = {}, []
    root = Path(plan['root'])
    try:
        for name, slot, module, arguments in jobs:
            log = (root/'logs'/(name+'.log')).open('x'); logs.append(log)
            command = [plan['training_python'],'-u','-m',module,'--plan',str(plan_path),*arguments]
            process = subprocess.Popen(command, cwd=plan['source_root'], env=child_environment(plan,slot),
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            processes[name] = process
            write_json(root/'processes'/(name+'.json'),dict(command=command,pid=process.pid,
                physical_gpu_uuid=plan['devices'][slot],execution_commit=plan['execution_commit'],started_epoch=time.time()))
        while True:
            clock = clock_status(root)
            if max(clock['occupied_s'],clock['elapsed_s']) >= TIME_CAP_S:
                raise TimeoutError('Original four-hour elapsed/card cap reached')
            failed = [name for name,p in processes.items() if p.poll() not in (None,0)]
            if failed:
                raise RuntimeError('Worker failed; completed evidence retained: '+','.join(failed))
            if all(p.poll()==0 for p in processes.values()):
                break
            time.sleep(1)
    finally:
        for name,p in processes.items():
            stop_owned(p)
            path=root/'processes'/(name+'.json')
            receipt=read(path);receipt.update(exit_code=p.returncode,finished_epoch=time.time())
            write_json(path,receipt)
        for log in logs:
            log.close()


def qualification_job(kind, slot):
    return (kind+'_'+slot,slot,'experiments.bounded_prefix_allocation.qualification_entry',
            ['--kind',kind,'--slot',slot])


def qualify_resources(plan):
    """Use actual qualification costs without changing any scientific setting."""
    root=Path(plan['root'])
    training=read(root/'qualification/training_A/STATUS.json')
    serving=[read(root/'qualification'/('serving_'+s)/'STATUS.json') for s in ('A','B')]
    if any(r['status']!='passed' for r in [training,*serving]):
        raise ValueError('All declared qualification cases must pass')
    t=training['result']
    estimates=dict(plan['estimated_seconds'])
    pair=max(row.get('pair_wall_excluding_observers_s',
                     row['pair_wall_excluding_preclip_observer_s']) for row in t['cases'])
    for name in ('C7','C8'):
        # Publication/identity allowance is based on the retained measured tasks.
        measured=t['loading_s']+t['startup_s']+128*pair+plan['checkpoint_boundary_allowance_s']
        estimates[name]=max(estimates[name],measured)
    rows=[row for s in serving for row in s['result']['qualification']['rows']]
    seconds_per_token=max(r['request_latency_s']/max(1,r['output_tokens']) for r in rows)
    panel=read(plan['references'])
    total_tokens=sum(len(row['response_ids']) for row in panel.values())
    # Nine executions of each panel request plus the unchanged warmup schedule.
    estimates['final']=max(estimates['final'],9*total_tokens*seconds_per_token+
        72*max(r['request_latency_s'] for r in rows)+plan['final_activation_allowance_s'])
    training_ram=training['sampled_process_tree_peak_rss_bytes']+12*2**30
    serving_ram=max(s['sampled_process_tree_peak_rss_bytes'] for s in serving)+4*2**30
    import psutil
    available=psutil.virtual_memory().available
    if max(training_ram,serving_ram)+16*2**30 > available:
        raise RuntimeError('Measured single-worker RAM plus reserve no longer fits')
    plan['estimated_seconds']=estimates
    timing=require_time_for(plan,sum(estimates[x] for x in TASKS))
    return dict(status='passed',execution_commit=plan['execution_commit'],estimates=estimates,
        timing=timing,measured_pair_compute_s=pair,measured_s_per_output_token=seconds_per_token,
        training_host_reservation_bytes=training_ram,serving_host_reservation_bytes=serving_ram,
        free_host_bytes=available,storage=boundary(plan,compiler=True))


def run(plan_path):
    plan_path=Path(plan_path);plan=read(plan_path);root=Path(plan['root'])
    if (root/'LAUNCH.json').exists():
        raise FileExistsError('Prior launch must be preserved; no implicit retry')
    (root/'logs').mkdir(exist_ok=True)
    receipt=dict(status='cpu_preflight',execution_commit=plan['execution_commit'],started_epoch=time.time(),completed=[])
    write_json(root/'LAUNCH.json',receipt)
    try:
        check_compilers()
        plan['compiler_baseline_allocated_bytes']=int(subprocess.check_output(
            ['du','-s','-B1',plan['compiler_root']],text=True).split()[0])
        write_json(plan_path,plan)
        with (root/'logs/CPU_PREFLIGHT.log').open('x') as log:
            subprocess.run([plan['training_python'],'-u','-m','experiments.bounded_prefix_allocation.preflight',
                '--plan',str(plan_path)],cwd=plan['source_root'],env=child_environment(plan),
                stdout=log,stderr=subprocess.STDOUT,check=True)
        preflight=read(root/'CPU_PREFLIGHT.json')
        if preflight['status']!='passed':
            raise RuntimeError('Assembled CPU preflight did not pass')
        receipt['completed'].append('cpu_preflight')
        initialize_work(root)  # Immediately before the first model worker.
        receipt.update(status='qualification',initial_time_admission=require_time_for(plan,sum(plan['estimated_seconds'].values())))
        write_json(root/'LAUNCH.json',receipt)
        jobs=[qualification_job('training','A'),qualification_job('serving','B')]
        if preflight['parallel_qualification_admitted']:
            run_children(plan_path,plan,jobs)
        else:
            for job in jobs:
                run_children(plan_path,plan,[job])
        run_children(plan_path,plan,[qualification_job('serving','A')])
        admission=qualify_resources(plan)
        write_json(root/'RESOURCE_ADMISSION.json',admission);write_json(plan_path,plan)
        receipt['completed'].append('qualification')
        receipt.update(status='production',expected_remaining=admission['timing'])
        write_json(root/'LAUNCH.json',receipt)
        for i,task in enumerate(TASKS):
            boundary(plan,compiler=True)
            require_time_for(plan,sum(plan['estimated_seconds'][t] for t in TASKS[i:]))
            receipt['active_task']=task;write_json(root/'LAUNCH.json',receipt)
            run_children(plan_path,plan,[(task,'A','experiments.bounded_prefix_allocation.task_runner',
                ['--task',task,'--gpu-uuid',plan['devices']['A']])])
            result=read(root/'tasks'/task/'STATUS.json')
            if result['status']!='completed':
                raise RuntimeError('Task exited without completed receipt: '+task)
            receipt['completed'].append(task);write_json(root/'LAUNCH.json',receipt)
        work=read(root/'WORK.json')
        if work['pending'] or work['attempted']!=work['completed']:
            raise RuntimeError('Completed campaign has unresolved work')
        receipt.update(status='completed',work=work,postflight=boundary(plan,compiler=True))
    except BaseException as error:
        receipt.update(status='stopped',error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        receipt['wall_s']=time.time()-receipt['started_epoch']
        if (root/'CLOCK.json').exists():receipt['clock']=clock_status(root)
        write_json(root/'LAUNCH.json',receipt)
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True)
    args=parser.parse_args();run(args.plan)


if __name__=='__main__':
    main()
