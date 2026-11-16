"""One native event, delayed scalar scoring, and explicit incomplete accounting."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

from online_feedback.storage import write_json
from experiments.native_prefix_loss.campaign import CampaignBudget,stop_owned_processes,training_environment,rows
from experiments.verification_transition.budget import BudgetService
from experiments.threatened_repair_replay.prepare import authenticate_source,read
from experiments.threatened_repair_replay.work import read_ledger
from .identity_gate import check_identity
from .resources import lightweight_snapshot
from .storage import check_write


def preflight(plan):
    root=Path(plan['root'])
    gate=read(plan['cpu_gate0'])
    if gate['status']!='passed' or gate['cuda_initialized'] or gate['optimizer_steps']:
        raise ValueError('CPU Gate0 has not passed with zero GPU/optimizer work')
    source=authenticate_source(plan['source_manifest'],plan['source_identity'],plan['execution_commit'])
    if (Path(__file__).resolve().parents[2]!=Path(plan['code_path']).resolve()
            or Path(source['source_path']).resolve()!=Path(plan['code_path']).resolve()):
        raise ValueError('Running controller is not the sealed source')
    reuse=read(plan['reuse_identity_receipt'])
    if reuse['status']!='passed' or reuse['matched_rows']!=12288:
        raise ValueError('Historical outcomes are unqualified for reuse')
    for path,expected in plan['live_input_identities'].items():check_identity(path,expected)
    runtime=read(root/'preparation/RESOURCE_PREFLIGHT.json')
    if runtime['status']!='passed':raise ValueError('CPU runtime/import/resource preflight is incomplete')
    resources=lightweight_snapshot(plan)
    devices = resources['gpu']['devices']
    selected = [row for row in devices['rows'] if str(row['index']) == str(plan['serving_gpu'])]
    if (devices['status'] != 'passed' or len(selected) != 1
            or selected[0]['uuid'] != plan['serving_gpu_identity']):
        raise ValueError('Selected physical GPU differs from the sealed index/UUID binding')
    from .probe_client import resource_estimate
    estimate=resource_estimate(plan)
    required=max(plan['training_host_required_bytes'],estimate['conservative_precompaction_peak_bytes']+plan['host_reserve_bytes'])
    if resources['host']['MemAvailable']<required:
        raise MemoryError('Host RAM cannot fit the complete bounded comparison')
    # No model work is admitted while any card has a pre-existing compute job.
    jobs=resources['gpu']['compute_jobs']
    if jobs['status']!='passed':raise ValueError('GPU process inventory unavailable')
    lines=[line for line in jobs['stdout'].splitlines() if line.strip()]
    if lines:raise RuntimeError('GPU compute job already active; preserve priority and quiet serving')
    allocation=check_write(plan,plan['peak_new_bytes'],'fresh_gpu_admission')
    result=dict(status='passed',source=source,gate0=gate['status'],reuse=reuse['status'],
                resources=resources,host_required_bytes=required,allocation=allocation,probe_estimate=estimate)
    write_json(root/'PREFLIGHT.json',result)
    return result


def launch_training(plan,deadline):
    root=Path(plan['root']);command=[plan['training_python'],'-u','-m','experiments.native_update_size.runner',
        '--plan',str(root/'sealed/PLAN.json'),'--output',str(root/'training'),'--deadline',str(deadline)]
    receipt=dict(status='running',command=command,execution_commit=plan['execution_commit'])
    started=time.monotonic()
    with (root/'TRAINING.log').open('x') as log:
        process=subprocess.Popen(command,cwd=plan['code_path'],env=training_environment(plan),
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        receipt['pid']=process.pid;write_json(root/'TRAINING_PROCESS.json',receipt)
        try:process.wait(timeout=max(.001,deadline-time.monotonic()))
        except BaseException:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            raise
        finally:
            receipt.update(status='completed' if process.returncode==0 else 'failed',returncode=process.returncode,wall_s=time.monotonic()-started)
            write_json(root/'TRAINING_PROCESS.json',receipt)
    if process.returncode:raise RuntimeError('Native training/qualification stopped; no automatic retry')
    result=read(root/'training/RESULT.json')
    if result['status']!='completed':raise ValueError('Native trajectory or calibration did not complete')
    return result


def limits(plan):
    reconstruction=sum(len(read(path)['metadata']['trace']) for path in plan['parent_records'].values())
    # Startup graph capture is separate from the 8,192+32 scored endpoints.
    return dict(draft_block=8192+32+reconstruction+128,model_load=40)


def write_report(plan,receipt,budget):
    from .analysis import write_report as scientific_report
    root=Path(plan['root']);training_path=root/'training/PHYSICAL_WORK.jsonl'
    training_work=read_ledger(training_path) if training_path.exists() else {}
    training=read(root/'training/RESULT.json') if (root/'training/RESULT.json').exists() else {}
    scoring=read(root/'probes/STATUS.json') if (root/'probes/STATUS.json').exists() else {}
    gates=dict(cpu_gate0=read(plan['cpu_gate0'])['status']=='passed',
        historical_source_and_reuse=read(plan['reuse_identity_receipt'])['status']=='passed',
        native_six_case_qualification=training.get('qualification',{}).get('status')=='passed',
        all_sixteen_windows=len(training.get('windows',[]))==16 and all(r['status']=='completed'for r in training.get('windows',[])),
        final_native_parity=training.get('final_native_parity',{}).get('status')=='passed',
        exact_reuse_sentinels=scoring.get('reuse_sentinel_status')=='passed',
        all_scalar_outcomes=scoring.get('new_logical_control_rows')==8192 and scoring.get('invalid_control_rows')==0 and scoring.get('unknown_control_rows')==0)
    complete=receipt['status']=='completed' and all(gates.values())
    work=dict(native_budget=budget.summary(),training=training_work,phases=receipt['phases'],
              campaign_wall_s=receipt['wall_s'],cpu_gate0_s=read(plan['cpu_gate0'])['cpu_wall_s'],
              scope='one native trajectory; no E regeneration, source-fitting or complete responses',
              clocks='Phase times include nested operations; never sum nested timers.')
    write_json(root/'WORK_AND_COSTS.json',work)
    validity=dict(complete=complete,all_required_checks_passed=complete,gates=gates,
                  errors=[] if complete else [receipt.get('error','incomplete required checks')])
    # Preserve interrupted scoring from the append-only ledger as well.
    probe_path=root/'probes/PROBE_OUTCOMES.json'
    outcomes=read(probe_path) if probe_path.exists() else rows(root/'probes','PROBE_OUTCOMES')
    result=scientific_report(root/'final_report',probe_rows=outcomes,sealed_probes=read(plan['probe_keys']),
        validity=validity,work=work,execution_commit=plan['execution_commit'],
        calibration=training.get('calibration_trials',[]),geometry=training.get('geometry',[]))
    if receipt['status']=='completed' and (not complete or result.get('status')!='completed'):
        raise ValueError('Execution lacks a required scientific gate or complete valid outcome grid')
    return result


def run(plan_path):
    from . import probe_client
    plan=read(plan_path);root=Path(plan['root'])
    if (root/'STATUS.json').exists():raise FileExistsError('Preserve old execution; no automatic retry')
    receipt=dict(status='preflight',phase='preflight',execution_commit=plan['execution_commit'],phases={})
    write_json(root/'STATUS.json',receipt);began=time.monotonic()
    budget=CampaignBudget(root/'NATIVE_WORK.jsonl',gpu_seconds=10800,limits=limits(plan));service=BudgetService(budget)
    previous_alarm=signal.getsignal(signal.SIGALRM);failure=None
    def alarm(_signal,_frame):stop_owned_processes(budget);raise TimeoutError('180 occupied-card minute cap')
    def phase(name,operation):
        receipt['phase']=name;write_json(root/'STATUS.json',receipt);start=time.monotonic()
        try:value=operation()
        except BaseException as error:
            receipt['phases'][name]=dict(status='failed',wall_s=time.monotonic()-start,error=repr(error));raise
        receipt['phases'][name]=dict(status='completed',wall_s=time.monotonic()-start);write_json(root/'STATUS.json',receipt);return value
    try:
        phase('read_only_preflight',lambda:preflight(plan))
        budget.start_gpu(1);deadline=time.monotonic()+10800
        signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,10800)
        training=phase('qualification_and_native_calibration',lambda:launch_training(plan,deadline))
        for row in training['probe_exports']:check_identity(Path(row['checkpoint'])/'model.safetensors',row['weights_identity'])
        phase('fixed_native_scoring',lambda:probe_client.run(plan,training['probe_exports'],root/'probes',service.connection,serving_gpu=str(plan['serving_gpu'])))
        receipt.update(status='completed',phase='report')
    except BaseException as error:
        failure=error;receipt.update(status='failed',error=repr(error),traceback=traceback.format_exc());write_json(root/'FAILURE.json',receipt);raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous_alarm)
        if budget.gpu_seconds>=10800:stop_owned_processes(budget)
        budget.end_gpu();receipt.update(wall_s=time.monotonic()-began,budget=budget.summary())
        write_json(root/'STATUS.json',receipt)
        try:
            write_report(plan,receipt,budget)
            if receipt['status']=='completed':receipt['phase']='complete';write_json(root/'STATUS.json',receipt)
        except BaseException as error:
            receipt.update(status='failed',phase='report_validation',report_error=repr(error))
            write_json(root/'REPORT_FAILURE.json',dict(error=repr(error),traceback=traceback.format_exc()))
            write_json(root/'STATUS.json',receipt)
            if failure is None:raise
        finally:service.close()
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);run(parser.parse_args().plan)
