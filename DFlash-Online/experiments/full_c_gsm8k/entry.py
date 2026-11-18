"""One explicit model-owning phase; failures preserve its original work ledger."""
import argparse
from contextlib import ExitStack
import os
from pathlib import Path
import time
import traceback

from online_feedback.storage import write_json
from experiments.persistent_prefix_withholding.qualification_entry import HostMemorySample, install_interrupt_handlers
from experiments.persistent_prefix_withholding.task_runner import create_backend, close_backend
from .runtime import WorkLedger, boundary, occupied_card, read


def run(plan, phase, *, resume=False):
    root = Path(plan['root'])
    slot = 'B' if phase in ('qualification_training','train') else 'A'
    if os.environ.get('CUDA_VISIBLE_DEVICES') != plan['devices'][slot]:
        raise ValueError('Phase does not own its pinned physical GPU')
    base = root/'phases'/phase
    phase_base=base
    if resume:
        previous=read(base/'STATUS.json')
        if previous['status']=='completed' or phase.startswith('qualification'):
            raise ValueError('Completed phases and qualification are never repeated')
        attempt=len(list((base/'attempts').glob('*')))+1
        base=base/'attempts'/f'{attempt:03d}'
        base.mkdir(parents=True,exist_ok=False)
        write_json(base/'PREVIOUS_STATUS.json',previous)
    else:
        base.mkdir(parents=True,exist_ok=False)
    report = dict(status='running',phase=phase,started_epoch=time.time(),
        execution_commit=plan['execution_commit'],binding=plan['binding'])
    monitor = HostMemorySample()
    backend = None
    ledger = WorkLedger(root,phase)
    try:
        with ExitStack() as stack:
            stack.enter_context(monitor)
            stack.enter_context(occupied_card(root,phase+'_'+slot,plan['devices'][slot]))
            if phase == 'train':
                # The serving engine remains allocated while the synchronous trainer
                # runs, so charge both physical cards for the complete worker lifetime.
                stack.enter_context(occupied_card(root,phase+'_A',plan['devices']['A']))
            report['admission'] = boundary(plan)
            if phase == 'qualification_training':
                from .qualification import run_training_qualification
                report['result'] = run_training_qualification(plan,'B',ledger=ledger)
            elif phase == 'train':
                from .campaign import run_arm
                report['result'] = run_arm(root,resume=resume,plan=plan)
            else:
                from .evaluation import evaluate_qualification, evaluate_final
                from experiments.kplus_full_gsm8k.preflight import timing_barrier
                backend = create_backend(plan,base,plan['references']['O58'],plan['devices']['A'])
                report['loading_s'] = backend.loading_s
                if phase == 'qualification_serving':
                    trained = read(root/'phases/qualification_training/STATUS.json')
                    if trained['status'] != 'completed':
                        raise ValueError('Training/export qualification must pass first')
                    checkpoints = dict(plan['references'],KPLUS_C58=trained['result']['export'])
                    ledger.wrap(backend,'generate','qualification_generation')
                    report['result'] = evaluate_qualification(backend,plan,checkpoints,base/'checks')
                else:
                    dataset = phase.removeprefix('final_')
                    sub = plan['datasets'][dataset]
                    trained = read(root/'campaign_001/arms/KPLUS_C/STATUS.json')
                    if trained['status'] != 'completed':
                        raise ValueError('Only the fixed C58 endpoint may be evaluated')
                    checkpoints = dict(plan['references'],KPLUS_C58=trained['final_checkpoint'])
                    from experiments.ok_coding.storage import identity
                    checkpoint_identities={method:identity(Path(path)/'model.safetensors')
                                           for method,path in checkpoints.items()}
                    write_json(base/'ENDPOINT_IDENTITIES.json',checkpoint_identities)
                    from .evaluation import execution_slots, label_for
                    phase_by_label={label_for(slot):slot['phase'] for slot,request in execution_slots(
                        read(sub['schedule']),read(sub['requests'])['requests'],
                        read(sub['warmups'])['requests'],dataset=dataset)}
                    original = backend.generate
                    def generate(*args,**kwargs):
                        # The evaluator passes a label with its sealed phase. The
                        # count belongs to the logical activation, including reused loads.
                        request_id = kwargs.get('request_id','')
                        kind = phase_by_label[request_id]+'_generation'
                        return ledger.call({kind:1},kind,original,*args,**kwargs)
                    backend.generate = generate
                    def quiet_boundary():
                        resources=boundary(plan)
                        quiet=timing_barrier(backend.worker.pid)
                        return dict(status='passed',physical_gpu_uuid=plan['devices']['A'],
                            both_cards_checked=True,other_gpu_idle=True,host_quiet=True,
                            storage_admitted=True,time_admitted=True,resources=resources,quiet=quiet)
                    report['result'] = evaluate_final(backend,read(sub['requests'])['requests'],checkpoints,
                        root/'evaluation'/dataset,dataset=dataset,canonicals=read(sub['canonicals']),
                        warmup_requests=read(sub['warmups'])['requests'],
                        warmup_canonicals=read(sub['warmup_canonicals']),schedule=read(sub['schedule']),
                        reference_checkpoints=sub['reference_checkpoints'],historical_counts=read(sub['historical_counts']),
                        checkpoint_identities=checkpoint_identities,resume=resume,
                        physical_gpu_uuid=plan['devices']['A'],boundary_check=quiet_boundary,
                        execution_commit=plan['execution_commit'])
            report['postflight'] = boundary(plan)
            if backend is not None:
                close_backend(backend,base)
                backend = None
            report['status'] = 'completed'
    except BaseException as error:
        report.update(status='failed',error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        report.update(wall_s=time.time()-report['started_epoch'],sampled_process_tree_peak_rss_bytes=monitor.maximum_bytes)
        write_json(base/'STATUS.json',report)
        if resume:
            write_json(phase_base/'STATUS.json',dict(report,attempt_receipt=str(base/'STATUS.json')))
        if backend is not None:
            try:
                close_backend(backend,base)
            except BaseException as error:
                report.update(status='failed',cleanup_error=repr(error))
                write_json(base/'STATUS.json',report)
                if resume:
                    write_json(phase_base/'STATUS.json',dict(report,attempt_receipt=str(base/'STATUS.json')))
                raise
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True)
    parser.add_argument('--phase',required=True,choices=('qualification_training','qualification_serving','train','final_gsm8k','final_mathqa'))
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    install_interrupt_handlers()
    run(read(args.plan),args.phase,resume=args.resume)
