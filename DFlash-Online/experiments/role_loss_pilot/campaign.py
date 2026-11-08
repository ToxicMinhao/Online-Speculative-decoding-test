"""Six fresh independent own-feedback trajectories, followed by fixed evaluation."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

from online_feedback.storage import write_json
from experiments.adaflash_c1.campaign import read, close_owned
from experiments.ok_coding.campaign import WorkLedger, read_work, require_complete_work
from experiments.full_k_gsm8k.campaign import online_record_check
from experiments.full_k_gsm8k.costs import begin_event_profile, finish_event_profile
from experiments.osd_persistent.run import check_event, check_first_updated_request
from experiments.o_half_matched_lr.archive import full_trace_errors
from experiments.ok_coding.storage import identity
from .contract import ARMS, DATASETS, EVENTS, arm_limits, expected_work, trajectory_root, read_requests, validate_plan
from .storage import admit, check_growth, retire_event, retire_redundant_features


def install_signal_handlers():
    """Make the operational timeout enter owned worker cleanup and receipts."""
    def interrupted(signum,_frame):
        raise InterruptedError('Operational stop signal '+str(signum)+'; no automatic retry')
    for number in (signal.SIGTERM,signal.SIGINT):
        signal.signal(number,interrupted)


def resource_check(root,storage,destination,phase,event=None):
    from experiments.kw_v1.evaluate import append_json
    receipt=dict(phase=phase,event=event,**check_growth(root,storage))
    append_json(destination,receipt)
    return receipt


def evaluation_boundary(root,plan,backend=None,*,phase='timed_batch'):
    from experiments.kplus_full_gsm8k.preflight import timing_barrier
    owner=backend.worker.pid if backend is not None else None
    storage=resource_check(root,plan['storage'],root/'campaign_001/RESOURCE_CHECKS.jsonl',phase)
    return dict(storage=dict(storage,required_bytes=storage['required_additional_and_reserve_bytes']),
                quiescence=timing_barrier(owner))


def qualification_gate(root,plan):
    from .qualify import BUDGET
    receipt=read(Path(root)/'qualification/QUALIFICATION.json')
    if (receipt.get('status')!='passed' or receipt['execution_commit']!=plan['execution_source']['commit']
        or receipt['loss_contracts']!=plan['loss_contracts'] or receipt['budget']!=BUDGET
        or receipt.get('input_states_unchanged') is not True or len(receipt['cases'])!=6):
        raise ValueError('Native bounded qualification has not passed for this sealed source')
    for kind,n in (('adam_step',6),('training_response_forward',12),('core_backward',12)):
        if any(receipt['work'].get(kind+'_'+suffix)!=n for suffix in ('attempted','completed')):
            raise ValueError('Qualification work differs from six updates/twelve presentations')
    q=plan['qualification'];prepared=read(q['prepared'])
    paths=[q[k] for k in ('config','initial_state','prepared')]+prepared['feature_paths']
    if len(paths)!=9 or len(set(paths))!=9 or set(paths)!=set(receipt['input_identities']):
        raise ValueError('Qualification input set differs from sealed plan')
    for path in paths:
        if identity(path)!=receipt['input_identities'][path]:
            raise ValueError('Qualified retained input changed: '+path)
    return dict(status='passed',execution_commit=receipt['execution_commit'],
        receipt_path=str(Path(root)/'qualification/QUALIFICATION.json'),disposable_adam_steps=6)


def counted_backend(backend,ledger,progress,root):
    generate,prepare=backend.generate,backend.prepare_feedback
    def generate_once(*args,**kwargs):
        kind='correctness_generations' if progress['generation_phase']=='correctness' else 'online_generations'
        return ledger.call(kind,generate,*args,**kwargs)
    def prepare_once(record,path):
        prepared=ledger.call('terminal_preparation_forwards',prepare,record,path)
        if prepared.get('eligible') is not False and prepared.get('status')=='prepared':
            retire_redundant_features(record,prepared,root,Path(progress['event_root']))
        return prepared
    backend.generate,backend.prepare_feedback=generate_once,prepare_once


def save_boundary(base,trainer,receipt,dataset,arm):
    export=Path(receipt['training']['checkpoint_path']);event=len(trainer.events)
    result=trainer.verify_boundary(export_path=export,
        receipt_path=base/'boundaries'/f'event_{event:03d}_validation.json')
    snapshot=export.parent/'training_state.pt'
    os.link(trainer.latest,snapshot)
    row=dict(status='passed',dataset=dataset,arm=arm,event=event,state=str(snapshot),
        weights=str(export/'model.safetensors'),state_identity=identity(snapshot),
        weights_identity=identity(export/'model.safetensors'),scientific_check=result)
    write_json(base/'boundaries'/f'event_{event:03d}.json',row)
    return row


def _diagnostic_inputs(root,plan,dataset,backend,arm):
    from .diagnostics import prepare_pool
    item=plan['datasets'][dataset];directory=root/'campaign_001/diagnostics'/dataset/'pool'
    if arm=='KPLUS':
        return prepare_pool(backend,read_requests(item['panel_manifest'],'test'),read(item['references']),directory,
            dataset=dataset,diagnostic_manifest=item['diagnostic_manifest'])
    pool=read(directory/'POOL.json')
    if pool['status']!='completed' or pool['completed_preparations']!=16:
        raise ValueError('Shared fixed diagnostic pool is incomplete')
    return pool


def run_trajectory(root,dataset,arm):
    from online_feedback.controller import OnlineController
    from online_feedback.sglang_backend import SGLangDFlashBackend
    from .training import RoleLossTrainer
    from .diagnostics import snapshot_panel
    root=Path(root);plan=read(root/'sealed/PLAN.json');validate_plan(plan)
    base=trajectory_root(root,dataset,arm);base.mkdir(parents=True,exist_ok=False)
    progress=dict(status='running',phase='initializing',dataset=dataset,arm=arm,binding=plan['binding'],
        events_completed=0,event_exports={},generation_phase='online',event_costs=[],automatic_retry=False)
    limits=arm_limits(arm);ledger=WorkLedger(base/'WORK.jsonl',limits)
    backend=trainer=None;started=time.perf_counter()
    item=plan['datasets'][dataset];requests=read_requests(item['training_manifest'],'train')
    try:
        admit(root,plan['storage'],receipt_path=base/'STORAGE_ADMISSION.json')
        progress['qualification']=qualification_gate(root,plan)
        backend=SGLangDFlashBackend(plan['target'],plan['draft'],worker_python=plan['worker_python'],
            output_root=base/'native_attempt_001/native',serving_gpu='0',context_limit=4096,block_size=16,
            worker_module='experiments.role_loss_pilot.worker')
        # Preparations precede counted own-feedback hooks. They have their own
        # <=32 ledger, and no generation or training response is borrowed here.
        pool=_diagnostic_inputs(root,plan,dataset,backend,arm)
        counted_backend(backend,ledger,progress,root)
        loading=time.perf_counter()
        trainer=RoleLossTrainer(item['configs'][arm],base/'trainer',binding=plan['binding'],arm=arm,dataset=dataset,
            original_config_text=Path(plan['original_config']).read_text())
        progress.update(backend_loading_s=backend.loading_s,trainer_loading_s=time.perf_counter()-loading)
        trainer.save_initial(plan['original_initial_state'])
        diagnostic_root=root/'campaign_001/diagnostics'/dataset
        if arm=='KPLUS':
            snapshot_panel(trainer,pool,diagnostic_root/'INITIAL/event_000',dataset=dataset,arm='INITIAL',event=0)
        ledger.wrap(trainer.core.strategy,'forward_loss','response_forwards')
        ledger.wrap(trainer.backend,'backward','response_backwards')
        ledger.wrap(trainer.backend.optimizer.optimizer,'step','optimizer_steps')
        previous_export=previous_boundary=None
        for event in range(1,EVENTS+1):
            event_id=f'event_{event:03d}';event_root=base/'events'/event_id;event_root.mkdir(parents=True)
            progress.update(phase='collecting',generation_phase='online',current_event=event,event_root=str(event_root))
            admit(root,plan['storage'],receipt_path=event_root/'STORAGE_ADMISSION.json')
            write_json(base/'STATUS.json',progress)
            controller=OnlineController(backend,trainer,event_root/'loop');records=[]
            for request in requests[(event-1)*128:event*128]:
                record=controller.serve(request['prompt_ids'],request_id=request['request_id'],
                    session_id=dataset+'_train',request_index=request['request_index'],max_new_tokens=2048)
                online_record_check(record,request,previous_export or plan['draft'],capture=True)
                records.append(record)
            if event==1 and arm=='KPLUS':
                progress['generation_phase']='correctness';probe=records[0]
                replay=controller.evaluate(probe['prompt_ids'],canonical_response_ids=probe['response_ids'],
                    label='first_capture_disabled',max_new_tokens=2048)
                if (replay['record']['finish_reason']!=probe['finish_reason']
                    or replay['record']['metadata']['verification_calls']!=probe['metadata']['verification_calls']):
                    raise ValueError('Inherited capture-disabled correctness check differs')
            progress.update(phase='training',generation_phase='online');write_json(base/'STATUS.json',progress)
            profile=begin_event_profile()
            try: receipt=controller.update(event_id)
            finally: write_json(event_root/'RESOURCE_COSTS.json',finish_event_profile(profile))
            validation_started=time.perf_counter()
            check_event(receipt,trainer,backend,event_root,event,128)
            boundary=save_boundary(base,trainer,receipt,dataset,arm)
            validation_s=time.perf_counter()-validation_started
            export=Path(receipt['training']['checkpoint_path'])
            if event==1 and arm=='KPLUS':
                progress['generation_phase']='correctness'
                correct=check_first_updated_request(backend,requests[128],plan['draft'],str(export),event_root)
                checks=[read(event_root/f'qualification_{n}.json') for n in ('updated','frozen','reloaded')]
                if len({r['finish_reason'] for r in checks})!=1 or any(full_trace_errors(r) for r in checks):
                    raise ValueError('Inherited updated/frozen/reloaded correctness differs')
                write_json(event_root/'INHERITED_CORRECTNESS.json',dict(correct,generations=4))
            if event in (2,4):
                progress['event_exports'][str(event)]=str(export)
                snapshot_panel(trainer,pool,diagnostic_root/arm/f'event_{event:03d}',
                    dataset=dataset,arm=arm,event=event)
                trainer.verify_boundary(export_path=export,receipt_path=event_root/'POST_DIAGNOSTIC_STATE.json')
            retirement_started=time.perf_counter()
            retire_event(root=root,event_root=event_root,records=records,receipt=receipt,
                previous_export=previous_export,previous_boundary=previous_boundary,current_boundary=boundary,
                dataset=dataset,arm=arm)
            costs=dict(read(event_root/f'loop/events/{event_id}.costs.json'),**receipt['training']['costs'],
                teacher_preparation_s=sum(r['teacher_preparation_s'] for r in receipt['preparation']),
                feature_persistence_s=sum(r['feature_persistence_s'] for r in receipt['preparation']),
                activation=receipt['activation'],publication_validation_s=validation_s,
                retention_s=time.perf_counter()-retirement_started)
            progress['event_costs'].append(dict(event=event,costs=costs))
            actual=read_work(base/'WORK.jsonl',limits);require_complete_work(actual,expected_work(event,arm))
            progress.update(events_completed=event,phase='idle_boundary',generation_phase='online',
                final_checkpoint=str(export),final_state=boundary['state'],boundary=boundary,actual_work=actual)
            resource_check(root,plan['storage'],base/'RESOURCE_CHECKS.jsonl','after_retention',event)
            write_json(base/'STATUS.json',progress);previous_export,previous_boundary=export,boundary
            print(json.dumps(dict(dataset=dataset,arm=arm,event=event,steps=trainer.global_step)),flush=True)
        progress['final_state_check']=trainer.verify_boundary(export_path=previous_export,
            receipt_path=base/'FINAL_STATE_CHECK.json')
        require_complete_work(read_work(base/'WORK.jsonl',limits),limits)
        progress.update(status='completed',phase='completed')
    except BaseException as error:
        if trainer is not None: trainer.failed=True
        progress.update(status='failed',error=repr(error),traceback=traceback.format_exc());raise
    finally:
        errors=close_owned(progress,backend,trainer)
        progress.update(worker_wall_s=time.perf_counter()-started,actual_work=read_work(base/'WORK.jsonl',limits))
        write_json(base/'STATUS.json',progress)
        if errors: raise errors[0]
    return progress


def execute_training(root,plan,dataset,arm):
    command=[plan['training_python'],'-u','-m','experiments.role_loss_pilot.campaign',str(root),
             '--train',dataset,arm]
    campaign=root/'campaign_001';name=dataset+'_'+arm;started=time.perf_counter()
    with (campaign/(name+'.log')).open('x') as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        receipt=dict(command=command,pid=process.pid,automatic_retry=False)
        write_json(campaign/('EXEC_'+name+'.json'),receipt)
        try: code=process.wait()
        except BaseException:
            os.killpg(process.pid,signal.SIGTERM)
            try: process.wait(timeout=90)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait()
            raise
        finally:
            receipt.update(returncode=process.poll(),wall_s=time.perf_counter()-started)
            write_json(campaign/('EXEC_'+name+'.json'),receipt)
    if code or read(trajectory_root(root,dataset,arm)/'STATUS.json')['status']!='completed':
        raise RuntimeError('Trajectory failed; preserve evidence without retry')


def selected_checkpoints(root,dataset,event):
    """Bind evaluated exports to validated immutable training boundary receipts."""
    if event not in (2,4):
        raise ValueError('Only prespecified event2/4 exports may be evaluated')
    result={}
    for arm in ARMS:
        base=trajectory_root(root,dataset,arm)
        status=read(base/'STATUS.json')
        marker=read(base/'boundaries'/f'event_{event:03d}.json')
        export=Path(status['event_exports'][str(event)])
        if (status['status']!='completed' or status['events_completed']!=4
            or marker['status']!='passed' or marker['dataset']!=dataset
            or marker['arm']!=arm or marker['event']!=event
            or Path(marker['weights'])!=export/'model.safetensors'
            or identity(marker['weights'])!=marker['weights_identity']):
            raise ValueError('Selected checkpoint differs from its authenticated own boundary')
        result[arm]=str(export)
    return result


def run(root):
    from .evaluation import evaluate_stage,audit_stage
    from online_feedback.sglang_backend import SGLangDFlashBackend
    root=Path(root).absolute();plan=read(root/'sealed/PLAN.json');validate_plan(plan)
    campaign=root/'campaign_001';campaign.mkdir(exist_ok=False)
    progress=dict(status='running',execution_source=plan['execution_source'],automatic_retry=False,
        completed_trajectories=[],evaluations=[])
    backend=None;started=time.perf_counter()
    try:
        admit(root,plan['storage'],receipt_path=campaign/'STORAGE_ADMISSION.json')
        progress['qualification']=qualification_gate(root,plan)
        for dataset in DATASETS:
            for arm in ARMS:
                execute_training(root,plan,dataset,arm)
                progress['completed_trajectories'].append(dict(dataset=dataset,arm=arm))
                write_json(campaign/'STATUS.json',progress)
        progress['training_quiescence']=evaluation_boundary(root,plan,phase='before_evaluation_backend')
        backend=SGLangDFlashBackend(plan['target'],plan['draft'],worker_python=plan['worker_python'],
            output_root=campaign/'evaluation_native_attempt_001/native',serving_gpu='0',context_limit=4096,block_size=16)
        for dataset in DATASETS:
            item=plan['datasets'][dataset];requests=read_requests(item['panel_manifest'],'test')
            canonicals=read(item['references'])
            for event in (2,4):
                checkpoints=selected_checkpoints(root,dataset,event)
                output=campaign/'evaluation'/dataset/f'event_{event:03d}'
                schedule=item['evaluation_schedules'][str(event)]
                result=evaluate_stage(backend,requests,checkpoints,output,dataset=dataset,event=event,
                    canonicals=canonicals,schedule=schedule,boundary_check=lambda:evaluation_boundary(root,plan,backend))
                audit=audit_stage(output,requests,checkpoints,dataset=dataset,event=event,
                    canonicals=canonicals,schedule=schedule,evidence_root=root)
                if selected_checkpoints(root,dataset,event)!=checkpoints:
                    raise ValueError('Selected exports changed during evaluation')
                write_json(output/'AUDIT.json',audit)
                if audit['status']!='passed': raise ValueError('Serving raw-evidence audit failed')
                progress['evaluations'].append(dict(dataset=dataset,event=event,result=result,audit=audit))
                write_json(campaign/'STATUS.json',progress)
        progress['status']='completed'
    except BaseException as error:
        progress.update(status='failed',error=repr(error),traceback=traceback.format_exc());raise
    finally:
        errors=close_owned(progress,backend);progress['campaign_wall_s']=time.perf_counter()-started
        write_json(campaign/'STATUS.json',progress)
        if errors: raise errors[0]
    return progress


def main():
    install_signal_handlers()
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    parser.add_argument('--train',nargs=2,metavar=('DATASET','ARM'));args=parser.parse_args()
    if args.train: run_trajectory(args.root,*args.train)
    else: run(args.root)


if __name__=='__main__':main()
