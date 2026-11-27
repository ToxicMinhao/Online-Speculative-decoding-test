"""One model-owning phase of the fixed full run."""
import argparse
from pathlib import Path
import time
import traceback

from .runtime import WorkLedger, read, write_json, boundary


def run(plan, phase, *, arm=None, resume=False, attempt=1):
    root=Path(plan['root'])
    name=phase+('_'+arm if arm else '')+f'_{attempt:03d}'
    output=root/'phases'/name
    output.mkdir(parents=True,exist_ok=False)
    report=dict(status='running',phase=phase,arm=arm,attempt=attempt,
        execution_commit=plan['execution_commit'],started_epoch=time.time())
    write_json(output/'STATUS.json',report)
    ledger=WorkLedger(root,name,'qualification' if phase.startswith('qualification') else 'production')
    try:
        report['admission']=boundary(plan)
        if phase=='qualification_training':
            from .qualification import run_training_qualification
            report['result']=run_training_qualification(plan,root/'qualification'/f'training_{attempt:03d}',ledger)
            write_json(root/'qualification/TRAINING.json',report['result'])
        elif phase=='qualification_serving':
            from experiments.repair_conditional_kl_pilot.campaign import create_backend
            from .serving import qualify_serving
            from experiments.repair_conditional_kl_pilot.production import quiet_evaluation
            training=read(root/'qualification/TRAINING.json')
            if training['status']!='passed':raise ValueError('Training integration is not qualified')
            checkpoints={**{arm:item['path'] for arm,item in training['exports'].items()},
                **{k:v['checkpoint'] for k,v in plan['retained_endpoints'].items()}}
            backend=create_backend(plan,output/'native',card='A')
            try:
                def activate(method,checkpoint,slot):
                    return backend,dict(publication=backend.publish(checkpoint),loading_s=0.)
                def guarded(slot):
                    return dict(resources=boundary(plan),quiet=quiet_evaluation(plan,backend))
                request=read(plan['manifest'])['qualification_requests'][0]
                report['result']=qualify_serving(plan,root/'qualification'/f'serving_{attempt:03d}',request,
                    checkpoints,activate=activate,boundary=guarded,ledger=ledger)
                write_json(output/'PHYSICAL_WORK.json',backend._call('work_receipt'))
            finally:
                backend.close()
            write_json(root/'qualification/SERVING.json',report['result'])
        elif phase=='train':
            from .campaign import run_arm
            from experiments.prefix_guard.qualify import pin_retained_graph_runtime
            report['compiler_policy']=pin_retained_graph_runtime(output)
            report['result']=run_arm(plan,arm,resume=resume)
        elif phase=='evaluate':
            from .serving import run as evaluate
            report['result']=evaluate(plan,ledger=ledger,boundary=lambda:boundary(plan))
        else:
            raise ValueError('Unknown model-owning full-run phase')
        report['status']='completed'
    except BaseException as error:
        report.update(status='failed',error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        report['finished_epoch']=time.time()
        report['wall_s']=report['finished_epoch']-report['started_epoch']
        write_json(output/'STATUS.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True)
    parser.add_argument('--phase',required=True,choices=['qualification_training','qualification_serving','train','evaluate'])
    parser.add_argument('--arm',choices=['BASE','TAIL'])
    parser.add_argument('--attempt',type=int,default=1)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    run(read(args.plan),args.phase,arm=args.arm,resume=args.resume,attempt=args.attempt)
