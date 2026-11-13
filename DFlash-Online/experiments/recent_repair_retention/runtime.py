"""Authenticate, prepare immutable placements, qualify, train, evaluate and stop."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

from online_feedback.storage import write_json
from experiments.verification_transition.data import identity, read
from experiments.kplus_full_gsm8k.preflight import timing_barrier
from experiments.survival_confirmation.runtime import physical_gpu, finish_worker
from experiments.o_half_matched_lr.archive import validate_record
from .analysis import ANALYSIS
from .evaluation import METHODS, evaluation_schedule, evaluate_stage, audit_stage


def capacity(plan):
    free=shutil.disk_usage(plan['root']).free
    required=plan['storage']['reserve_bytes']+plan['storage']['maximum_additional_bytes']
    receipt=dict(status='passed' if free>=required else 'failed',free_bytes=free,required_bytes=required)
    if receipt['status']!='passed':
        raise RuntimeError('Peak artifact writes plus reserve unavailable: '+repr(receipt))
    return receipt


def verify_execution_plan(plan):
    """Authenticate the original sealed plan and its staged input package.

    Frozen-reference preparation adds only the assignment path and its binding
    to TRAINING_PLAN. All original runtime, model and budget fields stay fixed.
    """
    sealed = Path(plan['root']) / 'sealed'
    receipt = read(sealed / 'SEAL.json')
    if str(sealed / 'PLAN.json') not in receipt['files']:
        raise ValueError('Execution seal does not authenticate PLAN.json')
    for path, expected in receipt['files'].items():
        if identity(path) != expected:
            raise ValueError('Sealed execution package changed: ' + path)
    original = read(sealed / 'PLAN.json')
    if any(plan.get(key) != value for key, value in original.items()):
        raise ValueError('Active execution plan differs from its sealed configuration')
    if set(plan) - set(original) - {'assignments', 'binding'}:
        raise ValueError('Unexpected changes outside the sealed execution plan')
    if receipt['execution_commit'] != plan['execution_commit']:
        raise ValueError('Sealed execution commit differs')
    return dict(status='passed', files=len(receipt['files']))


def preflight(plan, *, postflight=False):
    from experiments.lost_repair.run import preflight as original
    if plan['analysis']!=ANALYSIS or plan['production_budget']!=dict(
            trajectories=3,adam_steps=384,response_presentations=768):
        raise ValueError('Prespecified retention design changed')
    execution_seal = verify_execution_plan(plan)
    # Reuse exact source/model/runtime authentication, with this run's larger
    # storage allowance and original128 training requests. No model is loaded.
    legacy=dict(plan['original_plan'],source_manifest_path=plan['source_manifest'],
        storage=dict(required_free_bytes=plan['storage']['reserve_bytes']+
                     plan['storage']['maximum_additional_bytes']))
    old_k4=legacy['checkpoints']['K4']['export']
    legacy['checkpoints']={key:value for key,value in legacy['checkpoints'].items() if key in ('K5','K6')}
    legacy['pinned_files']={path:value for path,value in legacy['pinned_files'].items()
                            if not path.startswith(old_k4+'/')}
    result=original(legacy,Path(plan['root'])/'campaign_001')
    result['execution_seal'] = execution_seal
    for path,expected in plan['input_identities'].items():
        if identity(path)!=expected:
            raise ValueError('Sealed input changed: '+path)
    for name in ('recovery_path',):
        if plan['original_plan'][name]!=plan['origin_state']:
            raise ValueError('Origin state path differs from authenticated parent')
    result['physical_gpu']=physical_gpu(plan)
    result['capacity']=capacity(plan)
    available=next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
                   if line.startswith('MemAvailable:'))
    result['host_available_bytes']=available
    if not postflight and available<plan['host_memory_required_bytes']:
        raise RuntimeError('Insufficient host-memory headroom for retained native paths')
    result['postflight']=postflight
    write_json(Path(plan['root'])/('POSTFLIGHT.json' if postflight else 'PREFLIGHT.json'),result)
    return result


def launch(plan, module, arguments, name, *, serving_python=False):
    root=Path(plan['root'])/'processes'
    root.mkdir(exist_ok=True)
    command=[plan['worker_python'] if serving_python else plan['training_python'],
             '-u','-m',module,*map(str,arguments)]
    started=time.perf_counter()
    receipt=dict(status='running',command=command,execution_commit=plan['execution_commit'])
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0')
    env['CONDA_PREFIX']=str(Path(command[0]).parent.parent)
    env['PATH']=str(Path(command[0]).parent)+os.pathsep+env['PATH']
    with (root/(name+'.log')).open('x') as stream:
        process=subprocess.Popen(command,cwd=plan['code_path'],env=env,stdout=stream,stderr=subprocess.STDOUT)
        receipt['pid']=process.pid
        write_json(root/(name+'.json'),receipt)
        try:
            code=process.wait(timeout=plan['phase_timeout_s'])
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=60)
            receipt.update(status='completed' if process.returncode==0 else 'failed',
                           returncode=process.returncode,wall_s=time.perf_counter()-started)
            write_json(root/(name+'.json'),receipt)
    if code:
        raise RuntimeError('Preserve failed phase; no automatic retry: '+name)
    return receipt


def prepare_assignments(plan):
    from .preparation import prepare_assignments as build
    root=Path(plan['root'])
    with open(plan['all_starts'],newline='') as stream: starts=list(csv.DictReader(stream))
    with open(plan['native_weights'],newline='') as stream: weights=list(csv.DictReader(stream))
    files=sorted((root/'reference/presentations').glob('*.json'))
    if len(files)!=256 or read(root/'reference/STATUS.json')['status']!='completed':
        raise ValueError('Frozen native reference preparation incomplete')
    observations=[read(path) for path in files]
    margins=[row for data in observations for row in data['rows']]
    assigned=build(starts,weights,margins)
    assigned['frozen_selected_weight']=sum(data['presentation']['selected_weight'] for data in observations)
    weight=assigned['summary']['assigned_raw_weight']
    assigned['frozen_assigned_weight_fraction']=weight/assigned['frozen_selected_weight'] if assigned['frozen_selected_weight'] else None
    write_json(root/'ASSIGNMENTS.json',assigned)
    assignments_id=identity(root/'ASSIGNMENTS.json')
    seal=dict(schema='retention-event7-pretraining-seal-v1',execution_commit=plan['execution_commit'],
        inputs=plan['input_identities'],panel=identity(plan['panel_manifest']),
        assignments=assignments_id,parent_observations={str(p):identity(p) for p in files},
        analysis=ANALYSIS,scientific_optimizer_steps=0)
    write_json(root/'PRETRAINING_SEAL.json',seal)
    trained_plan=dict(plan,assignments=str(root/'ASSIGNMENTS.json'),
                     binding=identity(root/'PRETRAINING_SEAL.json')['sha256'])
    write_json(root/'TRAINING_PLAN.json',trained_plan)
    return trained_plan,assigned


def verify_pretraining(plan):
    root=Path(plan['root']);seal=read(root/'PRETRAINING_SEAL.json')
    if identity(root/'PRETRAINING_SEAL.json')['sha256']!=plan['binding']:
        raise ValueError('Pre-training binding changed')
    for path,expected in {**seal['inputs'],**seal['parent_observations'],
                          plan['assignments']:seal['assignments'],plan['panel_manifest']:seal['panel']}.items():
        if identity(path)!=expected:raise ValueError('Pre-training sealed content changed: '+path)


def prepare_canonicals(plan):
    """Reuse authenticated references where available; prepare missing ones once."""
    from online_feedback.sglang_backend import SGLangDFlashBackend
    root=Path(plan['root'])/'canonicals';root.mkdir(exist_ok=False)
    requests=read(plan['panel_manifest'])['requests']
    references=read(plan['reused_references'])
    missing=[r for r in requests if r['request_id'] not in references]
    if [r['request_id'] for r in missing]!=plan['missing_reference_ids']:
        raise ValueError('Sealed canonical preparation inventory changed')
    receipt=dict(status='running',planned_generations=len(missing),attempted=0,returned=0,
                 valid=0,reused=len(references),loading_s=0.,generation_s=0.,optimizer_steps=0)
    backend=None;failure=None;start=time.perf_counter()
    checkpoint=plan['checkpoints']['K6']['export']
    try:
        if missing:
            timing_barrier();capacity(plan)
            backend=SGLangDFlashBackend(plan['target_path'],checkpoint,worker_python=plan['worker_python'],
                output_root=root/'native',serving_gpu='0',context_limit=4096,block_size=16)
            receipt['loading_s']=backend.loading_s
        for request in missing:
            label=f"reference_{request['panel_ordinal']:04d}"
            receipt['attempted']+=1
            write_json(root/'STATUS.json',receipt)
            begin=time.perf_counter()
            record=backend.generate(request['prompt_ids'],max_new_tokens=2048,request_id=label,
                session_id='retention_canonical',request_index=request['panel_ordinal'],
                purpose='evaluation',save_to=root/(label+'.json'))
            receipt['returned']+=1;receipt['generation_s']+=time.perf_counter()-begin
            write_json(root/(label+'.json'),record)
            canonical=dict(prompt_ids=record['prompt_ids'],response_ids=record['response_ids'],
                finish_reason=record['finish_reason'],verified=True,binding_digest=plan['canonical_binding'])
            errors=validate_record(record,request,checkpoint,canonical,label=label,session='retention_canonical')
            write_json(root/(label+'_VALIDITY.json'),dict(errors=errors,status='failed' if errors else 'passed'))
            if errors:raise ValueError('Canonical preparation invalid: '+repr(errors))
            references[request['request_id']]=canonical;receipt['valid']+=1
            write_json(root/'CANONICAL.json',references)
        write_json(root/'CANONICAL.json',references)
        receipt['status']='completed'
    except BaseException as error:
        failure=error;receipt.update(status='failed',error=repr(error));raise
    finally:
        finish_worker(backend,receipt,root/'STATUS.json',start,failure)
    return references


def serve(plan, exports, references):
    from online_feedback.sglang_backend import SGLangDFlashBackend
    root=Path(plan['root']);requests=read(plan['panel_manifest'])['requests']
    assignment=read(plan['assignments'])['summary']
    if len(set(exports.values()))!=3 and not (
        assignment['control_identical'] and exports['CONTROL']==exports['RECENT']
        and exports['NATIVE']!=exports['RECENT']):
        raise ValueError('Only predeclared identical CONTROL may reuse RECENT export')
    timing_barrier();capacity(plan)
    backend=None;failure=None;receipt=dict(status='running');start=time.perf_counter()
    before={arm:identity(Path(path)/'model.safetensors') for arm,path in exports.items()}
    try:
        backend=SGLangDFlashBackend(plan['target_path'],exports['NATIVE'],worker_python=plan['worker_python'],
            output_root=root/'serving_native',serving_gpu='0',context_limit=4096,block_size=16)
        receipt['loading_s']=backend.loading_s
        schedule=read(plan['timing_schedule'])
        evaluate_stage(backend,requests,exports,root/'evaluation',canonicals=references,
            dataset='gsm8k',event=7,schedule=schedule,boundary_check=lambda:dict(
                storage=capacity(plan),quiescence=timing_barrier(backend.worker.pid),
                physical_gpu=physical_gpu(plan)))
        receipt['exports_unchanged']={arm:identity(Path(path)/'model.safetensors')==before[arm]
                                      for arm,path in exports.items()}
        if not all(receipt['exports_unchanged'].values()):raise ValueError('Timed export weights changed')
        receipt['status']='completed'
    except BaseException as error:
        failure=error;receipt.update(status='failed',error=repr(error));raise
    finally:
        finish_worker(backend,receipt,root/'SERVING_RESULT.json',start,failure)
    audit=audit_stage(root/'evaluation',requests,exports,canonicals=references,dataset='gsm8k',event=7,
                      schedule=schedule)
    write_json(root/'RAW_SERVING_AUDIT.json',{key:value for key,value in audit.items() if key!='rows'})
    if audit['status']!='passed':raise ValueError('Raw serving reduction failed: '+repr(audit['errors']))


def run(plan_path):
    plan=read(plan_path);root=Path(plan['root'])
    if (root/'STATUS.json').exists():raise FileExistsError('No automatic resume/retry of this bounded campaign')
    status=dict(status='running',phase='preflight',execution_commit=plan['execution_commit'],
                intended_production_steps=384,intended_timed_completions=6*plan['panel_size'])
    start=time.perf_counter()
    try:
        write_json(root/'STATUS.json',status)
        preflight(plan)
        launch(plan,'experiments.recent_repair_retention.reference',
               ['--plan',plan_path,'--output',root/'reference'],'reference')
        plan,assignments=prepare_assignments(plan)
        training_plan=root/'TRAINING_PLAN.json'
        if not assignments['summary']['training_feasible']:
            status.update(status='completed_no_treatment',phase='feasibility_stop',scientific_steps=0,timed_generations=0)
            return status
        status['phase']='qualification';write_json(root/'STATUS.json',status)
        for case in ('NATIVE','DISABLED','ZERO','ACTIVE'):
            launch(plan,'experiments.recent_repair_retention.qualification',
                ['--plan',training_plan,'--case',case,'--output',Path(plan['qualification_root'])/case],
                'qualification_'+case)
            if read(Path(plan['qualification_root'])/case/'RESULT.json')['status']!='passed':
                raise ValueError('Native qualification did not pass: '+case)
        exports={}
        for arm in METHODS:
            verify_pretraining(plan)
            if arm=='CONTROL' and assignments['summary']['control_identical']:
                exports[arm]=exports['RECENT']
                write_json(root/'CONTROL_REUSE.json',dict(reason='pretraining assignments identical',
                           scientific_optimizer_steps=0,placement_identifiable=False,export=exports[arm]))
                continue
            status.update(phase='training',arm=arm);write_json(root/'STATUS.json',status)
            launch(plan,'experiments.recent_repair_retention.training',
                ['--plan',training_plan,'--arm',arm,'--output',root/'training'/arm],'train_'+arm)
            result=read(root/'training'/arm/'RESULT.json')
            exports[arm]=result['checkpoint_path']
        write_json(root/'EXPORTS.json',exports)
        from .endpoint_difference import compare_weights
        write_json(root/'ENDPOINT_DIFFERENCES.json',{
            arm:compare_weights(exports['NATIVE'],exports[arm]) for arm in ('RECENT','CONTROL')})
        status['phase']='canonical_preparation';write_json(root/'STATUS.json',status)
        refs=prepare_canonicals(plan)
        status['phase']='fixed_start_diagnostic';write_json(root/'STATUS.json',status)
        launch(plan,'experiments.recent_repair_retention.fixed_starts',
            ['--plan',training_plan,'--exports',root/'EXPORTS.json','--references',root/'canonicals/CANONICAL.json',
             '--output',root/'fixed_starts'],'fixed_starts')
        status['phase']='timed_serving';write_json(root/'STATUS.json',status)
        serve(plan,exports,refs)
        preflight(plan,postflight=True)
        status.update(status='completed',phase='complete')
    except BaseException as error:
        status.update(status='failed',error=repr(error),traceback=traceback.format_exc());raise
    finally:
        status['wall_s']=time.perf_counter()-start
        write_json(root/'STATUS.json',status)
    return status


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True)
    args=parser.parse_args()
    run(args.plan)
