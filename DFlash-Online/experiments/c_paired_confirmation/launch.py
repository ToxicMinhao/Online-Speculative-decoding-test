"""One fixed paired continuation campaign with a single cumulative clock."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import time
import traceback

from online_feedback.storage import write_json
from experiments.bounded_prefix_allocation.launch import stop_owned
from experiments.persistent_prefix_withholding.launch import child_environment, check_compilers
from .runtime import (initialize_work, read, clock_status, boundary, require_time_for,
                      TIME_CAP_S, CARD_CAP_S)
from .resources import production_groups, price_groups

MODULE='experiments.c_paired_confirmation'


def task_job(dataset,task,slot,plan):
    return (dataset+'_'+task,slot,MODULE+'.task_runner',
        ['--dataset',dataset,'--task',task,'--gpu-uuid',plan['devices'][slot]])


def qualification_job(dataset,kind,slot):
    return (dataset+'_'+kind+'_'+slot,slot,MODULE+'.qualification_entry',
        ['--dataset',dataset,'--kind',kind,'--slot',slot])


def run_children(plan_path,plan,jobs):
    root=Path(plan['root']);processes={};logs=[]
    try:
        for name,slot,module,args in jobs:
            log=(root/'logs'/(name+'.log')).open('x');logs.append(log)
            command=[plan['training_python'],'-u','-m',module,'--plan',str(plan_path),*args]
            p=subprocess.Popen(command,cwd=plan['source_root'],env=child_environment(plan,slot),
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            processes[name]=p
            write_json(root/'processes'/(name+'.json'),dict(command=command,pid=p.pid,
                physical_gpu_uuid=plan['devices'][slot],execution_commit=plan['execution_commit'],started_epoch=time.time()))
        while True:
            clock=clock_status(root)
            if clock['elapsed_s']>=TIME_CAP_S or clock['occupied_s']>=CARD_CAP_S:
                raise TimeoutError('Original four-hour elapsed/five-card-hour cap reached')
            failed=[name for name,p in processes.items() if p.poll() not in (None,0)]
            if failed:raise RuntimeError('Worker failed; completed evidence preserved: '+','.join(failed))
            if all(p.poll()==0 for p in processes.values()):break
            time.sleep(1)
    finally:
        for name,p in processes.items():
            stop_owned(p);path=root/'processes'/(name+'.json');receipt=read(path)
            receipt.update(exit_code=p.returncode,finished_epoch=time.time());write_json(path,receipt)
        for log in logs:log.close()


def qualification_gate(plan):
    root=Path(plan['root']);result=dict(status='passed',binding=plan['binding'],
        execution_commit=plan['execution_commit'],datasets={})
    for dataset in ('gsm8k','mathqa'):
        trained=read(root/'qualification'/f'{dataset}_training_A'/'STATUS.json')
        serving={s:read(root/'qualification'/f'{dataset}_serving_{s}'/'STATUS.json') for s in ('A','B')}
        if trained['status']!='passed' or any(v['status']!='passed' for v in serving.values()):
            raise ValueError('Incomplete dataset qualification')
        signatures=[serving[s]['result']['qualification']['signatures'] for s in ('A','B')]
        identities=[serving[s]['result']['checkpoint_identity'] for s in ('A','B')]
        if signatures[0]!=signatures[1] or identities[0]!=identities[1]:
            raise ValueError('Same-checkpoint cross-card count/output compatibility failed')
        if trained['result']['cpu_export']['status']!='passed':
            raise ValueError('CPU serving export reconstruction failed')
        result['datasets'][dataset]=dict(training='passed',serving_A='passed',serving_B='passed',
            exact_cross_card_counts_and_outputs=True,scope='two fixed warmup requests, two repeats per card',
            training_result=str(root/'qualification'/f'{dataset}_training_A'/'DETAILS.json'))
    return result


def retire_qualification(plan,dataset):
    from .retention import unlink_manifest
    root=Path(plan['root']);trained=read(root/'qualification'/f'{dataset}_training_A'/'STATUS.json')
    for slot in ('A','B'):
        if read(root/'qualification'/f'{dataset}_serving_{slot}'/'STATUS.json')['status']!='passed':
            raise ValueError('Disposable weights still have a qualification consumer')
    payloads=trained['result']['disposable_payload_files']
    if trained['result']['cpu_export']['status']!='passed':raise ValueError('Re-export remains unqualified')
    return unlink_manifest(plan,payloads,receipt_path=root/'retirement'/f'{dataset}_QUALIFICATION.json',
        reason='Disposable qualification payloads; complete compact comparisons and failures retained',
        retained=[str(root/'qualification'/f'{dataset}_training_A'/'DETAILS.json')],
        verification=dict(training='passed',serving_A='passed',serving_B='passed',cpu_export='passed'))


def admission(plan,groups):
    """Conservative measured pair/serving bounds, with all remaining work priced."""
    import psutil
    root=Path(plan['root']);estimates=plan['estimated_seconds']
    train_reserve=serve_reserve=0
    for dataset in ('gsm8k','mathqa'):
        trained=read(root/'qualification'/f'{dataset}_training_A'/'STATUS.json')
        serves=[read(root/'qualification'/f'{dataset}_serving_{s}'/'STATUS.json') for s in ('A','B')]
        result=trained['result']
        pair=max(r.get('pair_wall_excluding_observers_s',r.get('pair_wall_excluding_preclip_observer_s',0)) for r in result['cases'])
        if pair<=0:raise ValueError('Qualification did not provide a measured update cost')
        startup=result.get('startup_s',result['startup'].get('wall_s',0))
        for task in ('N_first','C_first','N_second','C_second'):
            measured=result.get('loading_s',0)+startup+128*pair+plan['checkpoint_boundary_allowance_s']
            estimates[dataset][task]=max(estimates[dataset][task],measured)
        train_reserve=max(train_reserve,trained['sampled_process_tree_peak_rss_bytes']+12*2**30)
        serve_reserve=max(serve_reserve,max(s['sampled_process_tree_peak_rss_bytes'] for s in serves)+4*2**30)
    memory=psutil.virtual_memory().available;reserve=16*2**30
    if memory<max(train_reserve,serve_reserve)+reserve:
        raise RuntimeError('Measured single worker no longer fits host memory')
    parallel=memory>=train_reserve+serve_reserve+reserve
    actual_groups=groups if parallel else [[job] for group in groups for job in group]
    costs=price_groups(actual_groups,estimates)
    # CPU exports are not occupied GPU time but remain on the elapsed clock.
    costs['elapsed_seconds']+=plan['cpu_export_reconstruction_allowance_s']
    return dict(status='passed',binding=plan['binding'],execution_commit=plan['execution_commit'],
        groups=actual_groups,training_reservation_bytes=train_reserve,serving_reservation_bytes=serve_reserve,
        host_available_bytes=memory,host_reserve_bytes=reserve,parallel_training_collection=parallel,
        estimates=estimates,time=require_time_for(plan,costs['elapsed_seconds'],costs['occupied_card_seconds']),
        storage=boundary(plan,compiler=True))


def run(plan_path):
    plan_path=Path(plan_path);plan=read(plan_path);root=Path(plan['root'])
    if (root/'LAUNCH.json').exists():raise FileExistsError('Preserve the original launch; no implicit retry')
    (root/'logs').mkdir(exist_ok=True)
    receipt=dict(status='cpu_preflight',binding=plan['binding'],execution_commit=plan['execution_commit'],
        started_epoch=time.time(),completed=[],automatic_retry=False)
    write_json(root/'LAUNCH.json',receipt)
    try:
        check_compilers()
        plan['compiler_baseline_allocated_bytes']=int(subprocess.check_output(['du','-s','-B1',plan['compiler_root']],text=True).split()[0])
        write_json(plan_path,plan)
        with (root/'logs/CPU_PREFLIGHT.log').open('x') as log:
            subprocess.run([plan['training_python'],'-u','-m',MODULE+'.preflight','--plan',str(plan_path)],
                cwd=plan['source_root'],env=child_environment(plan),stdout=log,stderr=subprocess.STDOUT,check=True)
        preflight=read(root/'CPU_PREFLIGHT.json')
        if preflight['status']!='passed':raise ValueError('Assembled CPU preflight did not pass')
        receipt['completed'].append('cpu_preflight')
        initialize_work(root)
        original=price_groups(production_groups(),plan['estimated_seconds'])
        receipt['initial_admission']=require_time_for(plan,
            original['elapsed_seconds']+plan['qualification_elapsed_allowance_s']+plan['cpu_export_reconstruction_allowance_s'],
            original['occupied_card_seconds']+plan['qualification_card_allowance_s'])
        receipt['status']='qualification';write_json(root/'LAUNCH.json',receipt)
        for dataset in ('gsm8k','mathqa'):
            jobs=[qualification_job(dataset,'training','A'),task_job(dataset,'warmup_references','B',plan)]
            if preflight['parallel_qualification_admitted']:run_children(plan_path,plan,jobs)
            else:
                for job in jobs:run_children(plan_path,plan,[job])
            for slot in ('A','B'):run_children(plan_path,plan,[qualification_job(dataset,'serving',slot)])
            retire_qualification(plan,dataset)
            receipt['completed'].append(dataset+'_qualification');write_json(root/'LAUNCH.json',receipt)
        gate=qualification_gate(plan);write_json(root/'QUALIFICATION.json',gate)
        groups=production_groups();resources=admission(plan,groups)
        write_json(root/'RESOURCE_ADMISSION.json',resources);write_json(plan_path,plan)
        receipt['status']='production';receipt['resources']=resources;write_json(root/'LAUNCH.json',receipt)
        from .retention import retire_endpoint_export,reconstruct_final_export
        actual_groups=resources['groups']
        cpu_reconstruction_pending=not plan['storage']['retain_final_exports']
        for index,group in enumerate(actual_groups):
            remaining=price_groups(actual_groups[index:],plan['estimated_seconds'])
            if cpu_reconstruction_pending:
                remaining['elapsed_seconds']+=plan['cpu_export_reconstruction_allowance_s']
            require_time_for(plan,remaining['elapsed_seconds'],remaining['occupied_card_seconds'])
            boundary(plan,compiler=True)
            if group[0][1]=='final' and group[0][0]=='gsm8k' and not plan['storage']['retain_final_exports']:
                for task in ('N_second','C_second'):reconstruct_final_export(plan,'gsm8k',task)
                cpu_reconstruction_pending=False
                boundary(plan,compiler=True)
            receipt['active_tasks']=[d+'_'+t for d,t,_ in group];write_json(root/'LAUNCH.json',receipt)
            run_children(plan_path,plan,[task_job(d,t,s,plan) for d,t,s in group])
            for dataset,task,_ in group:
                status=read(root/'tasks'/f'{dataset}_{task}'/'STATUS.json')
                if status['status']!='completed':raise ValueError('Missing completed task receipt')
                receipt['completed'].append(dataset+'_'+task)
                if task in ('N_second','C_second'):
                    retire_endpoint_export(plan,dataset,task[0]+'_first')
                if task=='C_second' and dataset=='gsm8k' and not plan['storage']['retain_final_exports']:
                    for arm in ('N','C'):retire_endpoint_export(plan,dataset,arm+'_second')
            write_json(root/'LAUNCH.json',receipt)
        work=read(root/'WORK.json')
        if work['pending'] or work['attempted']!=work['completed']:
            raise ValueError('Unknown or uncompleted work remains')
        for key,count in plan['expected_work'].items():
            if work['completed'][key]!=count:raise ValueError('Completed work differs: '+key)
        receipt.update(status='completed',work=work,postflight=boundary(plan,compiler=True))
    except BaseException as error:
        receipt.update(status='stopped',error=repr(error),traceback=traceback.format_exc());raise
    finally:
        receipt['wall_s']=time.time()-receipt['started_epoch']
        if (root/'CLOCK.json').exists():receipt['clock']=clock_status(root)
        write_json(root/'LAUNCH.json',receipt)
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',required=True)
    run(parser.parse_args().plan)


if __name__=='__main__':main()
