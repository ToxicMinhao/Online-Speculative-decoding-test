"""One model-owning phase, with physical GPU leases and durable failure records."""
import argparse
from contextlib import ExitStack
import os
from pathlib import Path
import time
import traceback

from online_feedback.storage import write_json
from experiments.kw_v1.evaluate import append_json
from .runtime import WorkLedger, boundary, occupied_card, read


class Engines:
    """AR and DFlash never coexist on a card; every lifecycle is charged."""
    def __init__(self,plan,phase):
        self.plan,self.phase=plan,phase
        self.owned={}
        self.serial=0
        self.active=None
        self.active_kind=None

    def create(self,method,device,output):
        from online_feedback.sglang_backend import SGLangDFlashBackend
        from .ar_backend import SGLangARBackend
        self.serial+=1
        output=Path(output)
        output.mkdir(parents=True,exist_ok=True)
        lease=occupied_card(self.plan['root'],f'{self.phase}_engine_{self.serial}',device)
        lease.__enter__()
        started=time.perf_counter()
        try:
            if method=='AR':
                backend=SGLangARBackend(self.plan['target'],worker_python=self.plan['worker_python'],
                    output_root=output/'native',serving_gpu=device)
            else:
                backend=SGLangDFlashBackend(self.plan['target'],self.plan['draft'],
                    worker_python=self.plan['worker_python'],output_root=output/'native',
                    serving_gpu=device,context_limit=4096,block_size=16,
                    worker_module='experiments.persistent_prefix_withholding.worker')
        except BaseException:
            lease.__exit__(*__import__('sys').exc_info())
            raise
        self.owned[id(backend)]=(backend,lease,output)
        write_json(output/'LIFECYCLE.json',dict(status='loaded',method=method,device=device,
            loading_s=time.perf_counter()-started,worker_pid=backend.worker.pid))
        return backend

    def close(self,backend,output=None):
        _,lease,actual=self.owned.pop(id(backend))
        started=time.perf_counter()
        failure=None
        try:
            if not backend.failed:
                write_json(actual/'PHYSICAL_WORK.json',backend._call('work_receipt'))
        except BaseException as error:
            failure=error
        finally:
            try:
                backend.close()
            finally:
                lease.__exit__(None,None,None)
        write_json(actual/'CLOSE.json',dict(status='failed' if failure else 'closed',
            teardown_s=time.perf_counter()-started,error=repr(failure) if failure else None))
        if failure:
            raise failure

    def activate(self,method,checkpoint,slot):
        kind='AR' if method=='AR' else 'DFlash'
        started=time.perf_counter()
        reloaded=False
        if self.active is not None and self.active_kind!=kind:
            previous,self.active=self.active,None
            self.close(previous)
        if self.active is None:
            self.active=self.create(method,self.plan['devices']['A'],
                Path(self.plan['root'])/'evaluation/engines'/f"activation_{slot['activation']:03d}")
            self.active_kind=kind
            reloaded=True
        publication=None
        if method!='AR' and self.active.checkpoint_id!=checkpoint:
            publication=self.active.publish(checkpoint)
        return self.active,dict(publication=publication,engine_reinitialized=reloaded,
                               activation_s=time.perf_counter()-started)

    def close_all(self):
        self.active=None
        errors=[]
        for backend,_,_ in list(self.owned.values()):
            try:
                self.close(backend)
            except BaseException as error:
                errors.append(repr(error))
        if errors:
            raise RuntimeError('Owned engine cleanup failed: '+repr(errors))


def run(plan,phase,slot='A',arm=None):
    from experiments.persistent_prefix_withholding.qualification_entry import HostMemorySample
    root=Path(plan['root'])
    name=phase+'_'+(arm or slot)
    base=root/'phases'/name
    base.mkdir(parents=True,exist_ok=False)
    inputs=read(plan['manifest'])
    report=dict(status='running',phase=phase,slot=slot,arm=arm,
                execution_commit=plan['execution_commit'],started_epoch=time.time())
    write_json(base/'STATUS.json',report)
    engines=Engines(plan,name)
    qualification=phase.startswith('qualification')
    ledger=WorkLedger(root,name,'qualification' if qualification else 'production')
    try:
        with HostMemorySample() as monitor:
            report['admission']=boundary(plan)
            if phase=='qualification_serving':
                from .serving_gate import qualify_card
                report['result']=qualify_card(plan,inputs,slot,root/'qualification'/slot,
                    ledger=ledger,backend_factory=engines.create,close_backend=engines.close,
                    boundary=lambda:boundary(plan))
            elif phase=='qualification_training':
                from .qualification import run_training_qualification
                by_id=read(root/'qualification/A/FEATURES.json')
                paths=[by_id[key]['receipt']['feature_path'] for key in plan['qualification_pair_ids']]
                with occupied_card(root,name,plan['devices']['B']):
                    report['result']=run_training_qualification(plan,arm,feature_paths=paths,ledger=ledger)
            elif phase=='qualification_publication':
                from .serving_gate import publication_gate
                trained=read(root/'qualification/C_CODE58/DETAILS.json')
                if trained['status']!='passed':
                    raise ValueError('Disposable C qualification has not passed')
                report['result']=publication_gate(plan,inputs,root/'qualification/A',
                    trained['active_C_export'],root/'qualification/publication',ledger=ledger,
                    backend_factory=engines.create,close_backend=engines.close,boundary=lambda:boundary(plan))
            elif phase=='precheck':
                from .serving_gate import panel_precheck
                report['result']=panel_precheck(plan,inputs,slot,root/'precheck'/slot,shard_count=2,
                    ledger=ledger,backend_factory=engines.create,close_backend=engines.close,
                    boundary=lambda:boundary(plan))
            elif phase=='train':
                from .campaign import run_arm
                with ExitStack() as stack:
                    for card in ('A','B'):
                        stack.enter_context(occupied_card(root,name+'_'+card,plan['devices'][card]))
                    report['result']=run_arm(plan,arm)
            elif phase=='final':
                from .evaluation import execute
                from experiments.kplus_full_gsm8k.preflight import timing_barrier
                from experiments.ok_coding.storage import identity
                from .contract import ARMS
                checkpoints={'F':plan['draft'],'AR':plan['target']}
                for candidate in ARMS:
                    status=read(root/'campaign_001/arms'/candidate/'STATUS.json')
                    if status['status']!='completed' or status['events_completed']!=58:
                        raise ValueError('All three full event58 endpoints must be complete')
                    checkpoints[candidate]=status['final_checkpoint']
                before={method:identity(Path(path)/'model.safetensors') for method,path in checkpoints.items() if method!='AR'}
                write_json(base/'EXPORTS_BEFORE.json',before)
                def quiet(entry):
                    resources=boundary(plan)
                    owner=engines.active.worker.pid if engines.active is not None else None
                    return dict(activation=entry['activation'],resources=resources,timing=timing_barrier(owner),
                        physical_gpu_uuid=plan['devices']['A'],both_cards_checked=True)
                report['result']=execute(plan,root/'evaluation/final',inputs['evaluation_requests'],
                    inputs['warmup_requests'],read(root/'precheck/CANONICALS.json'),
                    read(root/'qualification/WARMUP_CANONICALS.json'),checkpoints,
                    activate=engines.activate,boundary=quiet,ledger=ledger)
                engines.close_all()
                after={method:identity(Path(path)/'model.safetensors') for method,path in checkpoints.items() if method!='AR'}
                write_json(base/'EXPORTS_AFTER.json',dict(identities=after,unchanged=before==after))
                if before!=after:
                    raise ValueError('A frozen endpoint changed during final timing')
            else:
                raise ValueError('Unknown bounded CODE phase')
            report['host_memory']=dict(sampled_process_tree_peak_rss_bytes=monitor.maximum_bytes,
                                       sampling_interval_s=.25)
            report['status']='completed'
    except BaseException as error:
        report.update(status='failed',error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        write_json(base/'STATUS.json',report)
        try:
            engines.close_all()
        except BaseException as error:
            report.update(status='failed',cleanup_error=repr(error))
            raise
        finally:
            report['wall_s']=time.time()-report['started_epoch']
            write_json(base/'STATUS.json',report)
    return report


if __name__=='__main__':
    from experiments.persistent_prefix_withholding.qualification_entry import install_interrupt_handlers
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True)
    parser.add_argument('--phase',required=True)
    parser.add_argument('--slot',default='A',choices=['A','B'])
    parser.add_argument('--arm')
    arguments=parser.parse_args()
    install_interrupt_handlers()
    run(read(arguments.plan),arguments.phase,arguments.slot,arguments.arm)
