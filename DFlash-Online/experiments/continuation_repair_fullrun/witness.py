"""Eight fixed BASE event-one training responses, shared by both trajectories."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import time

import torch

from experiments.continuation_value_diagnostic.native import ReadOnlyEndpoint
from experiments.continuation_value_diagnostic.metrics import paired_block
from experiments.ok_coding.storage import identity
from online_feedback.rng import capture_rng, preserve_rng
from online_feedback.storage import write_json
from .contract import WITNESS_EVENTS, SELECTION_SEED

SCHEMA='continuation_repair_fullrun_witness_v1'


def sealed_selection_ids(plan):
    rows=plan['witness_selection']
    if isinstance(rows,(str,Path)):
        rows=json.loads(Path(rows).read_text())['responses']
    return [row['request_id'] if isinstance(row,dict) else row for row in rows]


def seal_witness(trainer,plan,feature_paths):
    """Persist only selected packets and independent native anchor draws pre-update."""
    from specforge.training.strategies.base import _cpu_max_valid_anchors
    if trainer.arm!='BASE' or (trainer.global_step,trainer.presentations,len(trainer.events))!=(0,0,0):
        raise ValueError('Shared witness must be sealed at released BASE before event1')
    root=Path(plan['root'])/'witness';destination=root/'BINDINGS.json'
    if destination.exists():raise FileExistsError(destination)
    wanted=sealed_selection_ids(plan)
    if len(wanted)!=8 or len(set(wanted))!=8:raise ValueError('Require eight input-sealed witness requests')
    sources={}
    for path in feature_paths:
        raw=torch.load(path,map_location='cpu',weights_only=True)
        key=str(raw['record_id'])
        if key in wanted:sources[key]=(path,raw)
    if set(sources)!=set(wanted):raise ValueError('Selected requests missing from BASE event1 feedback')
    device=next(trainer.bundle.draft_model.parameters()).device
    root.mkdir(parents=True,exist_ok=True);rows=[]
    with preserve_rng(),torch.no_grad():
        torch.manual_seed(SELECTION_SEED)
        if device.type=='cuda':torch.cuda.manual_seed(SELECTION_SEED)
        for ordinal,key in enumerate(wanted):
            source,raw=sources[key]
            packet=root/'features'/f'{ordinal:03d}.pt';packet.parent.mkdir(exist_ok=True)
            shutil.copyfile(source,packet)
            expected=identity(source)
            if identity(packet)!=expected:raise ValueError('Fixed witness packet copy changed')
            batch=trainer._batch(raw);mask=batch.tensors['loss_mask'];available=_cpu_max_valid_anchors(mask)
            chosen=[]
            if available:
                anchors,keep=trainer._original_sampler(batch.tensors['input_ids'].shape[-1],mask.to(device),device,
                    max_valid_anchors=min(8,available))
                chosen=anchors[0,keep[0]].detach().cpu().tolist()
            if chosen!=sorted(set(chosen)) or len(chosen)!=min(8,available):
                raise ValueError('Witness sampler changed its bounded native anchor geometry')
            rows.append(dict(ordinal=ordinal,record_id=key,feature_path=str(packet),feature_identity=expected,
                             anchors=chosen,available_anchors=available))
    result=dict(schema=SCHEMA,status='sealed',features=rows,selection_seed=SELECTION_SEED,
        origin='released_model',feedback='BASE event1 TRAIN; TAIL may have different own feedback',
        maximum_response_forwards=72,maximum_blocks_per_response=8)
    write_json(destination,result)
    return result


def state_stamp(trainer):
    """Object identities and versions catch mutations without model-sized copies."""
    model=trainer.bundle.model
    parameters=[(name,id(p),p._version,None if p.grad is None else (id(p.grad),p.grad._version),
                 p.requires_grad) for name,p in model.named_parameters()]
    optimizer=trainer.backend.optimizer.optimizer
    moments=[(id(p),key,id(value),value._version) for p,state in optimizer.state.items()
             for key,value in state.items() if torch.is_tensor(value)]
    return dict(parameters=parameters,moments=moments,optimizer_groups=deepcopy([
        {k:v for k,v in group.items() if k!='params'} for group in optimizer.param_groups]),
        counters=(trainer.global_step,trainer.presentations,len(trainer.events),trainer.core._micro),
        rng=capture_rng())


def read_witness(trainer,plan,*,endpoint=None):
    """One zero-backward read at a fixed released/event boundary, <=8 forwards."""
    from experiments.ok_continuation.resume import differences
    from .runtime import WorkLedger
    event=len(trainer.events)
    endpoint=endpoint or ('released' if event==0 else f'{trainer.arm}_{event:03d}')
    if event==0:
        if trainer.arm!='BASE' or endpoint!='released':raise ValueError('Released shared witness is read once by BASE')
    elif event not in WITNESS_EVENTS or endpoint!=f'{trainer.arm}_{event:03d}':
        raise ValueError('Witness is restricted to the four fixed event endpoints')
    if (trainer.global_step,trainer.presentations)!=(event*128,event*256) or trainer.core.accumulation_remainder:
        raise ValueError('Witness requires a complete actual event boundary')
    root=Path(plan['root'])/'witness';output=root/endpoint
    if output.exists():raise FileExistsError(output)
    bindings=json.loads((root/'BINDINGS.json').read_text())
    for row in bindings['features']:
        if identity(row['feature_path'])!=row['feature_identity']:raise ValueError('Witness feature identity changed')
    output.mkdir();before=state_stamp(trainer)
    old_work=deepcopy(trainer.work);old_anchors=deepcopy(trainer._anchor_observations)
    model=trainer.bundle.model
    summaries={name:deepcopy(getattr(getattr(model,name,None),'last_summary',None))
               for name in ('prefix_allocation','continuation_repair')}
    adapter=ReadOnlyEndpoint(model,endpoint=endpoint,maximum_length=trainer.cfg.data.max_length,
        allowed_endpoints=(endpoint,))
    receipt=dict(status='running',endpoint=endpoint,event=event,responses=[],paired_blocks=[])
    started=time.perf_counter()
    ledger=WorkLedger(plan['root'],'witness:'+endpoint,phase='production')
    try:
        for row in bindings['features']:
            observed=ledger.call({'witness_response_forward':int(bool(row['anchors']))},
                'fixed_training_witness',adapter.read_response,row,output/f'{row["ordinal"]:03d}.json')
            receipt['responses'].append(observed)
            if event:
                origin=json.loads((root/'released'/f'{row["ordinal"]:03d}.json').read_text())
                for anchor,parent,child in zip(row['anchors'],origin['blocks'],observed['blocks']):
                    paired=paired_block(parent,child,record_id=row['record_id'],anchor=anchor,cohort='BASE_training_witness')
                    if paired['Y']!=paired['useful_repair']+paired['reachable_origin_followers']-paired['lost_prefix']:
                        raise ValueError('Witness signed-progress decomposition failed')
                    receipt['paired_blocks'].append(paired)
        receipt['status']='completed'
    except BaseException as error:
        receipt.update(status='failed',error=repr(error));raise
    finally:
        receipt['work']=dict(adapter.work)
        try:
            try:receipt['state']=adapter.close()
            except BaseException as error:
                already_failed=receipt['status']=='failed'
                receipt.update(status='failed',close_error=repr(error))
                if not already_failed:raise
        finally:
            receipt['observed_training_hook_work']={k:trainer.work.get(k,0)-old_work.get(k,0) for k in trainer.work}
            trainer.work.clear();trainer.work.update(old_work);trainer._anchor_observations[:]=old_anchors
            for name,summary in summaries.items():
                extension=getattr(model,name,None)
                if extension is not None:extension.last_summary=summary
            after=state_stamp(trainer)
            receipt['training_state_unchanged']=not differences(before,after)
            state_passed=receipt.get('state',{}).get('status')=='passed'
            if not receipt['training_state_unchanged'] or not state_passed:
                receipt['status']='failed'
            receipt['wall_s']=time.perf_counter()-started
            write_json(output/'STATUS.json',receipt)
        if not receipt['training_state_unchanged'] or not state_passed:
            raise ValueError('Shared witness changed live training state')
    return receipt
