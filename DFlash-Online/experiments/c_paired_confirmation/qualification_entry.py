"""Bounded per-dataset training and same-checkpoint two-card serving checks."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import time
import traceback

from online_feedback.storage import write_json
from experiments.fresh_confirmation.protocol import weight_check_errors
from experiments.persistent_prefix_withholding.qualification_entry import HostMemorySample, install_interrupt_handlers
from experiments.persistent_prefix_withholding.serving_qualification import evaluate, gpu_memory
from experiments.persistent_prefix_withholding.task_runner import create_backend, close_backend
from .runtime import WorkLedger, boundary, occupied_card, read
from .prepare import identity


def serving(plan, dataset, slot, base):
    sub=plan['datasets'][dataset]
    trained=read(Path(plan['root'])/'qualification'/f'{dataset}_training_A'/'STATUS.json')
    if trained['status']!='passed':
        raise ValueError('Require the completed dataset training qualification')
    checkpoint=trained['result']['export']
    warmup_task=read(Path(plan['root'])/'tasks'/f'{dataset}_warmup_references'/'RESULT.json')
    if (warmup_task['status']!='completed' or warmup_task['binding']!=plan['binding']
            or warmup_task['execution_commit']!=plan['execution_commit']):
        raise ValueError('Require this dataset’s sealed origin warmup references')
    references=read(sub['warmup_references']);requests=read(sub['warmups'])['requests']
    from .evaluation import validate_references
    validate_references(requests,references,origin_checkpoint=sub['origin_export'])
    uuid=plan['devices'][slot];ledger=WorkLedger(plan['root'],dataset+'_serving_'+slot)
    result=dict(status='running',dataset=dataset,checkpoint=checkpoint,
        checkpoint_identity=identity(Path(checkpoint)/'model.safetensors'),
        binding=plan['binding'],execution_commit=plan['execution_commit'])
    backend=None
    try:
        backend=create_backend(plan,base,checkpoint,uuid)
        result['loading_s']=backend.loading_s
        ledger.wrap(backend,'generate','qualification_generation')
        for phase in ('before','after'):
            check=backend.check_weights(checkpoint);errors=weight_check_errors(check,checkpoint)
            write_json(base/('WEIGHTS_'+phase.upper()+'.json'),dict(check=check,errors=errors))
            if errors:raise ValueError('Disposable checkpoint does not match serving consumer')
            if phase=='before':
                result['qualification']=evaluate(backend,requests,references,checkpoint,base/'calls',slot=slot,uuid=uuid)
                result['qualification']['actual_role']='same_disposable_active_C_checkpoint'
        result['gpu_memory']=gpu_memory(uuid)
        result['status']='passed'
        return result
    except BaseException as error:
        result.update(status='failed',error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        write_json(base/'RESULT.json',result)
        close_backend(backend,base)


def run(plan, dataset, kind, slot):
    if dataset not in ('gsm8k','mathqa') or (kind=='training' and slot!='A'):
        raise ValueError('Both datasets train on the admitted common physical card A')
    uuid=plan['devices'][slot]
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=uuid:
        raise ValueError('Wrong assigned physical card')
    base=Path(plan['root'])/'qualification'/f'{dataset}_{kind}_{slot}'
    base.mkdir(parents=True,exist_ok=False)
    receipt=dict(status='running',dataset=dataset,physical_gpu_uuid=uuid,
        execution_commit=plan['execution_commit'],binding=plan['binding'],started_epoch=time.time())
    monitor=HostMemorySample();write_json(base/'STATUS.json',receipt)
    try:
        with occupied_card(plan['root'],f'{dataset}_{kind}_{slot}',uuid),monitor:
            boundary(plan)
            if kind=='training':
                from .qualification import run_training_qualification
                ledger=WorkLedger(plan['root'],f'{dataset}_training_{slot}')
                receipt['result']=run_training_qualification(plan,dataset,slot,ledger=ledger)
            else:
                receipt['result']=serving(plan,dataset,slot,base)
            receipt['postflight']=boundary(plan,compiler=True)
            receipt['status']='passed'
    except BaseException as error:
        receipt.update(status='failed',error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        receipt.update(wall_s=time.time()-receipt['started_epoch'],
            sampled_process_tree_peak_rss_bytes=monitor.maximum_bytes)
        write_json(base/'STATUS.json',receipt)
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True);parser.add_argument('--dataset',required=True,choices=('gsm8k','mathqa'))
    parser.add_argument('--kind',required=True,choices=('training','serving'));parser.add_argument('--slot',required=True,choices=('A','B'))
    args=parser.parse_args();install_interrupt_handlers();run(read(args.plan),args.dataset,args.kind,args.slot)


if __name__=='__main__':main()
