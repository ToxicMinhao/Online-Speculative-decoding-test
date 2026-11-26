"""Supervise the sealed two-card forward budget, then stop for CPU analysis."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

from experiments.frontier_preservation_pilot.qualify import verify_source, storage_usage
from experiments.frontier_preservation_pilot.point1_prepare import gpu_inventory
from experiments.ok_coding.storage import identity
from online_feedback.storage import write_json


def stop_process(process):
    if process.poll() is None:
        try:
            os.killpg(process.pid,signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def check_storage(contract):
    usage=storage_usage(contract)
    usage['all_locations_charged_bytes']=usage['charged_bytes']+contract['local_charged_bytes']
    usage['remaining_write_reserve_bytes']=300_000_000
    if (usage['free_bytes']-usage['remaining_write_reserve_bytes'] < contract['limits']['reserve_bytes'] or
        usage['all_locations_charged_bytes']+usage['remaining_write_reserve_bytes'] > contract['limits']['allocated_bytes']):
        raise RuntimeError('Current storage cannot admit remaining readouts/cache growth')
    write_json(Path(contract['root'])/'STORAGE.json',usage)
    return usage


def execute(contract):
    root=Path(contract['root'])
    if (root/'CLOCK.json').exists() or (root/'STATUS.json').exists():
        raise FileExistsError('Existing GPU attempt must remain preserved; no automatic retry')
    verify_source(contract)
    check_storage(contract)
    if gpu_inventory()['processes']:
        raise RuntimeError('A GPU became occupied before launch')
    clock=dict(first_gpu_action_epoch=time.time(),occupied_card_s=0.,processes=[],
               elapsed_limit_s=contract['limits']['elapsed_s'],occupied_card_limit_s=3600)
    write_json(root/'CLOCK.json',clock)
    status=dict(status='running',execution_commit=contract['execution_commit'],phases=[],
                forbidden_work=dict(backward=0,adam=0,optimizer_creation=0,teacher=0,generation=0))
    try:
        for phase in ('qualification','primary'):
            check_storage(contract)
            processes=[]
            logs=[]
            start=time.time()
            try:
                for shard,device in enumerate(contract['devices']):
                    log=open(root/f'{phase}_gpu{shard}.log','w'); logs.append(log)
                    environment=dict(os.environ,CUDA_VISIBLE_DEVICES=device)
                    command=[contract['training_python'],'-u','-m','experiments.continuation_value_diagnostic.worker',
                             '--contract',str(root/'CONTRACT.json'),'--phase',phase,'--shard',str(shard)]
                    began=time.time()
                    process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=environment,
                                             cwd=contract['source_root'],start_new_session=True)
                    record=dict(phase=phase,shard=shard,physical_gpu=device,pid=process.pid,start_epoch=began,
                                end_epoch=None,exit_code=None)
                    clock['processes'].append(record); processes.append((process,record))
                while any(record['end_epoch'] is None for _,record in processes):
                    now=time.time()
                    for process,record in processes:
                        if record['end_epoch'] is None and process.poll() is not None:
                            record.update(end_epoch=now,exit_code=process.returncode)
                    occupied=sum((r['end_epoch'] or now)-r['start_epoch'] for r in clock['processes'])
                    clock.update(occupied_card_s=occupied,elapsed_s=now-clock['first_gpu_action_epoch'])
                    write_json(root/'CLOCK.json',clock)
                    usage=storage_usage(contract)
                    usage['all_locations_charged_bytes']=usage['charged_bytes']+contract['local_charged_bytes']
                    write_json(root/'STORAGE.json',usage)
                    if (occupied>=3540 or clock['elapsed_s']>=5340 or
                        usage['free_bytes']<contract['limits']['reserve_bytes'] or
                        usage['all_locations_charged_bytes']>contract['limits']['allocated_bytes']):
                        raise RuntimeError('Time or storage cap reached; no automatic expansion')
                    failed=[r for _,r in processes if r['exit_code'] not in (None,0)]
                    if failed:
                        raise RuntimeError(f'{phase} worker failed: {failed}')
                    if any(r['end_epoch'] is None for _,r in processes):
                        time.sleep(1)
            finally:
                for process,record in processes:
                    stop_process(process)
                    if record['end_epoch'] is None:
                        record.update(end_epoch=time.time(),exit_code=process.returncode)
                for log in logs: log.close()
                clock['occupied_card_s']=sum(r['end_epoch']-r['start_epoch'] for r in clock['processes'])
                clock['elapsed_s']=time.time()-clock['first_gpu_action_epoch']
                write_json(root/'CLOCK.json',clock)
            if phase=='qualification':
                from .native import compare_repeated
                cross_card_comparisons=[]
                for endpoint in contract['endpoints']:
                    # Qualification filenames are fixed by the explicit native adapter.
                    records=[json.loads((root/phase/f'gpu{i}'/endpoint/'primary_1.json').read_text())
                             for i in range(len(contract['devices']))]
                    for other in records[1:]:
                        comparison=compare_repeated(records[0],other)
                        cross_card_comparisons.append(dict(endpoint=endpoint,**comparison))
                        if comparison['status']!='passed':
                            write_json(root/'CROSS_CARD.json',comparison)
                            raise ValueError('Physical GPU readout parity failed')
                write_json(root/'CROSS_CARD.json',dict(status='passed',endpoints=list(contract['endpoints']),
                    first_primary_exact=True,physical_gpus=contract['devices'],comparisons=cross_card_comparisons))
            status['phases'].append(dict(phase=phase,status='passed',wall_s=time.time()-start))
            write_json(root/'STATUS.json',status)
        # Endpoint weights were immutable in memory; authenticate retained file identities again.
        checked={}
        for endpoint,pin in contract['endpoints'].items():
            path=str(Path(pin['export'])/'model.safetensors')
            checked[endpoint]=identity(path)
            if checked[endpoint]!=pin['identity']:
                raise ValueError('Retained checkpoint file changed during read-only execution')
        status.update(status='completed',checkpoint_identities_after=checked)
    except BaseException as error:
        status.update(status='blocked',error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        clock['elapsed_s']=time.time()-clock['first_gpu_action_epoch']
        write_json(root/'CLOCK.json',clock)
        status['clock']=clock
        write_json(root/'STATUS.json',status)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract',required=True)
    args=parser.parse_args()
    execute(json.loads(Path(args.contract).read_text()))


if __name__=='__main__':
    main()
