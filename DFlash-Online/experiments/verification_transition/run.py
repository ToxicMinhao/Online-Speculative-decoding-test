"""Bounded native entry point, invoked only after implementation review.

The parent owns the durable shared budget and kills only its isolated worker
process group at the occupied-GPU time cap. No retries or warmup generations.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from online_feedback.storage import write_json
from .budget import BudgetService, WorkBudget, connect_budget
from .contract import DATASETS, GPU_SECONDS
from .data import read, identity, verify_seal
from .preflight import verify


def worker(plan_path, output, connection_path):
    from .campaign import run_campaign
    from .native_worker import NativeTransitionBackend

    plan = read(plan_path)
    budget = connect_budget(read(connection_path))
    backend = None
    loading = budget.attempt('model_load')
    try:
        backend = NativeTransitionBackend(
            plan['shared']['target_path'], plan['checkpoints']['gsm8k']['KPLUS4']['export'],
            worker_python=plan['shared']['runtime']['python'], output_root=output/'native',
            budget_connection=read(connection_path), serving_gpu='0', context_limit=4096, block_size=16)
        budget.complete(loading, loading_s=backend.loading_s)
        status = run_campaign(plan,backend,budget,output/'campaign')
        return 0 if status['status'] == 'completed' else 2
    except BaseException as error:
        # The loading ticket is unresolved only when loading itself failed.
        if backend is None:
            budget.fail(loading,error)
        write_json(output/'WORKER_FAILURE.json',dict(status='technically_incomplete',
                   error=str(error),traceback=traceback.format_exc()))
        return 2
    finally:
        if backend is not None:
            backend.close()


def record_interruption(plan, output, budget, reason):
    """Keep every selected ID, including work not reached before interruption."""
    from .analysis import request_row, summarize, write_tables, render_report

    campaign=output/'campaign'
    rows_path=campaign/'ROWS.json'
    rows=read(rows_path) if rows_path.exists() else [
        request_row(r['dataset'],r['request_id'],r['selection_rank']) for r in plan['requests']]
    summaries={d:summarize([r for r in rows if r['dataset']==d],dataset=d) for d in DATASETS}
    for value in summaries.values():
        value['status']='incomplete'
        value.pop('strata',None)
        value['reason']=reason
    write_tables(campaign/'report',rows,summaries)
    (campaign/'report/REPORT.md').write_text(render_report(
        summaries,execution_commit=plan['execution_commit'],work=budget.summary()))
    write_json(campaign/'STATUS.json',dict(status='incomplete',reason=reason,work=budget.summary()))


def active_owned_processes(groups):
    """Inspect only registered groups; zombies no longer occupy a GPU."""
    text = subprocess.check_output(['ps','-eo','pid=,pgid=,stat='],text=True)
    rows = []
    for line in text.splitlines():
        pid, pgid, state = line.split()
        if int(pgid) in groups and not state.startswith('Z'):
            rows.append(dict(pid=int(pid),pgid=int(pgid),state=state))
    return rows


def stop_owned_groups(proxy, child):
    """The nested SGLang worker owns a separate session; kill it as well."""
    groups = {row['pgid'] for row in proxy.processes()}
    if child is not None:
        groups.add(child.pid)
    if os.getpgrp() in groups:
        raise RuntimeError('Refusing to terminate the supervisor process group')
    before = active_owned_processes(groups)
    for pgid in groups:
        try:
            os.killpg(pgid,signal.SIGKILL)
        except ProcessLookupError:
            pass
    if child is not None:
        child.wait()
    # Poll briefly for the driver's process cleanup, keeping GPU accounting on.
    deadline = time.monotonic()+10
    remaining = active_owned_processes(groups)
    while remaining and time.monotonic() < deadline:
        time.sleep(0.1)
        remaining = active_owned_processes(groups)
    if remaining:
        raise RuntimeError('Owned native processes survived termination: '+str(remaining))
    return dict(status='stopped', registered_groups=sorted(groups), before=before, remaining=[])


def execute(inputs_path, output):
    output=Path(output).absolute()
    output.mkdir(parents=True,exist_ok=False)
    plan=verify_seal(inputs_path)
    code_root=Path(__file__).resolve().parents[2]
    started=time.monotonic()
    admission=verify(plan,output,code_root)
    write_json(output/'PREFLIGHT.json',admission)
    plan['execution_commit']=admission['execution_commit']
    write_json(output/'PLAN.json',plan)
    write_json(output/'INPUT_IDENTITY.json',identity(inputs_path))
    # Check quiescence before allocating one GPU. It is a hard execution
    # prerequisite; this script does not terminate any pre-existing process.
    jobs=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
                                   '--format=csv,noheader,nounits'],text=True).strip()
    if jobs:
        raise RuntimeError('GPU work already active; native diagnostic not started')
    transfers=subprocess.run(['pgrep','-a','-u',str(os.getuid()),'rsync|scp'],capture_output=True,text=True)
    if transfers.stdout.strip():
        raise RuntimeError('Artifact transfer active; native diagnostic not started')
    budget=WorkBudget(output/'WORK.jsonl')
    service=BudgetService(budget)
    connection=output/'budget_connection.json'
    write_json(connection,service.connection)
    connection.chmod(0o600)
    proxy=connect_budget(service.connection)
    child=None
    timeout=False
    try:
        proxy.start_gpu(1)
        with (output/'worker.log').open('w') as log:
            child=subprocess.Popen([plan['shared']['runtime']['python'],'-u','-m',
                'experiments.verification_transition.run','--worker','--inputs',str(output/'PLAN.json'),
                '--output',str(output),'--budget-connection',str(connection)],
                cwd=code_root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            write_json(output/'PROCESS.json',dict(pid=child.pid,occupied_gpus=1,cap_gpu_seconds=GPU_SECONDS))
            try:
                code=child.wait(timeout=max(0.01,GPU_SECONDS-30-proxy.summary()['gpu_seconds']))
            except subprocess.TimeoutExpired:
                timeout=True
                write_json(output/'FORCED_CLEANUP.json',stop_owned_groups(proxy,child))
                code=child.returncode
        remaining=active_owned_processes({row['pgid'] for row in proxy.processes()})
        if remaining:
            write_json(output/'ORPHAN_CLEANUP.json',stop_owned_groups(proxy,child))
            code=2
        proxy.end_gpu()
        state_path=output/'campaign/STATUS.json'
        if (timeout or not state_path.exists()
                or (code != 0 and read(state_path).get('status') == 'completed')):
            record_interruption(plan,output,budget,
                'GPU time cap (30 seconds reserved for cleanup)' if timeout else 'worker did not finish cleanly')
        postflight=[]
        for dataset in DATASETS:
            for arm,item in plan['checkpoints'][dataset].items():
                actual=identity(item['weights'])
                postflight.append(dict(dataset=dataset,arm=arm,unchanged=actual==item['weights_identity']))
        if not all(r['unchanged'] for r in postflight):
            record_interruption(plan,output,budget,'Checkpoint changed during diagnostic')
            code=2
        write_json(output/'POSTFLIGHT.json',dict(checkpoints=postflight))
        write_json(output/'RESULT.json',dict(exit_code=code,timed_out=timeout,work=budget.summary(),
            wall_s=time.monotonic()-started,implementation_and_cpu_analysis_excluded_from_gpu_work=True))
        return code
    finally:
        owned_groups={row['pgid'] for row in proxy.processes()}
        if (child is not None and child.poll() is None) or active_owned_processes(owned_groups):
            write_json(output/'FINAL_CLEANUP.json',stop_owned_groups(proxy,child))
        proxy.end_gpu()
        service.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--budget-connection',type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args()
    result=worker(args.inputs,args.output,args.budget_connection) if args.worker else execute(args.inputs,args.output)
    raise SystemExit(result)


if __name__=='__main__':main()
