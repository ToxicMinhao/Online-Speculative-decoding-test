"""Six disposable native Adam updates, exactly twelve response presentations.

Each arm independently restores the authenticated released state for two fixed
response pairs. There are no startup response forwards or extra scientific
updates. CPU expression-oracle fixtures are a separate qualification gate.
"""
import argparse
import json
import shutil
from pathlib import Path
import time
import traceback

import torch

from experiments.adaflash_arepo.qualify import finite_gradient_check
from experiments.adaflash_clipping.run import initial_check, observed_backward, retained_anchors, serializable
from experiments.prefix_guard.qualify import observe_forwards, pin_retained_graph_runtime
from experiments.prefix_ranking.qualify import WorkLog, cpu_copy, override
from experiments.ok_continuation.resume import differences
from experiments.ok_coding.storage import identity
from experiments.ok_coding.training import _export_equal
from online_feedback.kplus import install_kplus
from online_feedback.rng import capture_rng, preserve_rng, restore_rng
from online_feedback.storage import save_torch, write_json
from online_feedback.training import SpecForgeEventTrainer
from .contract import ARMS, loss_contract

BUDGET=dict(maximum_optimizer_steps=6,training_presentations=12,
    core_response_forwards=12,core_backwards=12,startup_response_forwards=0,
    startup_backwards=0,startup_optimizer_steps=0,complete_generations=0,exports=0)


class BoundedLog(WorkLog):
    """Reject an attempted seventh update or thirteenth response before execution."""
    LIMITS={'adam_step':6,'training_response_forward':12,'core_backward':12}

    def work(self,kind):
        if kind in self.LIMITS and self.counts[kind+'_attempted']>=self.LIMITS[kind]:
            raise RuntimeError('Qualification work ceiling reached: '+kind)
        return super().work(kind)


def validate_initial(initial):
    if (initial['global_step'] or initial['presentations'] or initial['events']
        or initial['backend']['optimizer']['optimizer_state_dict']['state']):
        raise ValueError('Qualification requires released weights and fresh empty Adam')


def check_updated(trainer,initial):
    """Check real BF16 movement, all moments and decoded payloads without forwards."""
    state=dict(optimizer=cpu_copy(trainer.backend.optimizer.state_dict()))
    weights=trainer.bundle.draft_model.state_dict()
    changed=0
    for name,weight in weights.items():
        if weight.dtype!=torch.bfloat16 or not bool(torch.isfinite(weight).all()):
            raise ValueError('Nonfinite or non-BF16 updated draft tensor: '+name)
        changed+=int((weight.detach().cpu()!=initial['draft_state_dict'][name]).sum())
    optimizer=state['optimizer']['optimizer_state_dict']
    if not changed or len(optimizer['state'])!=58:
        raise ValueError('Disposable update did not change native draft or initialize all moments')
    for value in optimizer['state'].values():
        if value['step'].item()!=1:
            raise ValueError('Disposable Adam counter differs')
        for key in ('exp_avg','exp_avg_sq'):
            if value[key].dtype!=torch.bfloat16 or not bool(torch.isfinite(value[key]).all()):
                raise ValueError('Disposable Adam moments are not finite native BF16')
    return dict(status='passed',changed_bf16_coordinates=changed,adam_entries=58,adam_step=1)


def run_case(trainer,initial,pair,arm,pair_index,log,root):
    from specforge.training.strategies.base import StepContext
    directory=root/f'{arm}_pair_{pair_index:02d}'; directory.mkdir()
    log.trial=directory.name
    restoration=initial_check(trainer,initial,directory/'INITIAL_STATE.json',load=True)
    if arm=='KPLUS':
        install_kplus(trainer.bundle.model)
    else:
        from online_feedback.role_loss import install_role_loss
        install_role_loss(trainer.bundle.model,arm)
    model=trainer.bundle.model
    model.begin_event()
    named={n:p for n,p in trainer.bundle.draft_model.named_parameters() if p.requires_grad}
    optimized={id(p) for p in trainer.backend.optimizer.model_params}
    if optimized!={id(p) for p in named.values()}:
        raise ValueError('Optimizer is not restricted to the complete trainable draft')
    frozen=[p for p in model.parameters() if id(p) not in optimized]
    if any(p.requires_grad or p.grad is not None for p in frozen):
        raise ValueError('Frozen target-derived parameters are not isolated')
    ordinary_step=trainer.backend.optimizer.step
    def step():
        finite_gradient_check(named,directory,log.presentation)
        with log.work('adam_step'):
            return ordinary_step()
    ordinary_forward=trainer.author_strategy.forward_loss
    def finite_forward(*args,**kwargs):
        result=ordinary_forward(*args,**kwargs)
        if not bool(torch.isfinite(result.loss).all()):
            write_json(directory/'NONFINITE_LOSS.json',dict(status='failed',optimizer_called=False))
            raise ValueError('Nonfinite loss; stopped before backward and Adam')
        return result
    rows=[]; started=time.perf_counter()
    with observe_forwards(trainer,log), observed_backward(trainer,log,'core_backward'), \
         override(trainer.backend.optimizer,'step',step), \
         override(trainer.author_strategy,'forward_loss',finite_forward):
        for ordinal,(path,record) in enumerate(pair):
            log.presentation=ordinal
            raw=torch.load(path,map_location='cpu',weights_only=True)
            with retained_anchors(model,record),log.work('training_response_forward'):
                result=trainer.core.train_step(trainer._batch(raw),StepContext(
                    global_step=trainer.global_step,total_steps=trainer.horizon,collect_detailed_metrics=True))
            metrics=result.materialize_metrics()
            if not all(torch.isfinite(torch.tensor(float(v))) for v in metrics.values()):
                raise ValueError('Nonfinite representative metric')
            if not result.optimizer_stepped:
                finite_gradient_check(named,directory,ordinal)
            rows.append(dict(record_id=record['record_id'],metrics=metrics,
                optimizer_stepped=result.optimizer_stepped))
            trainer.presentations+=1; trainer.global_step+=int(result.optimizer_stepped)
            write_json(directory/'TRAINING.json',serializable(rows))
            del raw,result
    if trainer.global_step!=1 or trainer.presentations!=2 or trainer.core.accumulation_remainder:
        raise ValueError('Disposable case must contain exactly one step/two presentations')
    updated=check_updated(trainer,initial)
    if any(p.requires_grad or p.grad is not None for p in frozen):
        raise ValueError('Target-derived parameters acquired gradients during qualification')
    updated['draft_only_optimizer_and_frozen_target_isolation']=True
    # Decode an actual temporary checkpoint, compare complete scientific tensors,
    # moments, counters and RNG, then restore the authenticated fresh state.
    payload=dict(draft_state_dict=cpu_copy(trainer.bundle.draft_model.state_dict()),
        backend=dict(optimizer=cpu_copy(trainer.backend.optimizer.state_dict())),rng=capture_rng(),
        global_step=trainer.global_step,presentations=trainer.presentations,core_micro=trainer.core._micro)
    path=directory/'disposable_decode.pt'
    with preserve_rng():
        save_torch(path,payload)
        decoded=torch.load(path,map_location='cpu',weights_only=False)
        mismatch=differences(payload,decoded)
        if mismatch:
            raise ValueError('Temporary decoded scientific state differs: '+repr(mismatch[:8]))
        initial_check(trainer,initial,directory/'RESET_BEFORE_DECODE_RESTORE.json',load=True)
        trainer.bundle.draft_model.load_state_dict(decoded['draft_state_dict'],strict=True)
        trainer.backend.load_state_dict(decoded['backend'])
        restore_rng(decoded['rng'])
        trainer.global_step,trainer.presentations=decoded['global_step'],decoded['presentations']
        trainer.core._micro=decoded['core_micro']
        decoded_live=dict(draft_state_dict=cpu_copy(trainer.bundle.draft_model.state_dict()),
            backend=dict(optimizer=cpu_copy(trainer.backend.optimizer.state_dict())),rng=capture_rng(),
            global_step=trainer.global_step,presentations=trainer.presentations,core_micro=trainer.core._micro)
        if differences(decoded,decoded_live):
            raise ValueError('Decoded disposable state did not restore exactly')
        decoded_identity=identity(path)
    restored=initial_check(trainer,initial,directory/'RESTORED_INITIAL_STATE.json',load=True)
    receipt=dict(status='passed',arm=arm,pair=pair_index,restoration=restoration,
        update=updated,decoded_state_exact=True,decoded_state_identity=decoded_identity,
        decoded_path=str(path),restored=restored,rows=rows,optimizer_steps=1,presentations=2,
        no_extra_response_forwards=True,wall_s=time.perf_counter()-started,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0)
    write_json(directory/'RESULT.json',serializable(receipt))
    free_before=shutil.disk_usage(root).free
    path.unlink()
    write_json(directory/'RETIREMENT.json',dict(status='completed',path=str(path),
        identity=decoded_identity,free_before_bytes=free_before,free_after_bytes=shutil.disk_usage(root).free,
        reason='Disposable exact decoded-state check complete; fresh restoration passed',
        lost_capability='Disposable update tensors retired, not archived; never used by production'))
    return receipt


def run(config,initial_path,prepared_path,output,*,execution_commit=None):
    root=Path(output); root.mkdir(parents=True,exist_ok=False)
    log=BoundedLog(root)
    summary=dict(status='running',budget=BUDGET,execution_commit=execution_commit,
        loss_contracts={a:loss_contract(a) for a in ARMS},cases=[],automatic_retry=False)
    write_json(root/'PLAN.json',summary)
    trainer=None; started=time.perf_counter()
    try:
        prepared=json.loads(Path(prepared_path).read_text()); paths=prepared['feature_paths']; records=prepared['records']
        if (prepared['status']!='completed' or len(paths)!=6 or len(records)!=6
            or len(set(paths))!=6 or len({r['record_id'] for r in records})!=6):
            raise ValueError('Require authenticated six-record PR31 reconstruction inventory')
        for path,record in zip(paths,records):
            raw=torch.load(path,map_location='cpu',weights_only=True)
            if raw['record_id']!=record['record_id'] or record['pass']!=0:
                raise ValueError('Retained feature/presentation binding differs')
            del raw
        initial=torch.load(initial_path,map_location='cpu',weights_only=False,mmap=True)
        validate_initial(initial)
        import yaml
        config_text=Path(config).read_text();config_data=yaml.safe_load(config_text)
        if config_text!=initial['config_text'] or config_data['training']['accumulation_steps']!=2:
            raise ValueError('Qualification config differs from original fresh state before runtime construction')
        _export_equal(initial,config_data['model']['draft_checkpoint_path'])
        summary['input_identities']={str(p):identity(p) for p in (config,initial_path,prepared_path,*paths)}
        write_json(root/'INPUT_IDENTITIES.json',summary['input_identities'])
        compatibility=pin_retained_graph_runtime(root)
        compatibility.update(applied_before='all six fresh trainer constructions; no startup response forwards',
            scope='two isolated response pairs each for KPLUS, MIX and ROLE')
        write_json(root/'RUNTIME_COMPATIBILITY.json',compatibility)
        # Same two pairs for all arms; the six-file source inventory stays
        # authenticated, but its last two records authorize no additional pass.
        pairs=[list(zip(paths[start:start+2],records[start:start+2])) for start in (0,2)]
        summary['used_feature_paths']=paths[:4]
        for arm in ARMS:
            for number,pair in enumerate(pairs,1):
                trainer=SpecForgeEventTrainer(config,root/f'{arm}_{number:02d}_runtime',
                    objective='osd_first_error',reduction='response_mean')
                torch.cuda.reset_peak_memory_stats()
                if trainer.config_text!=initial['config_text'] or trainer.accumulation!=2:
                    raise ValueError('Qualification changed native config or accumulation')
                result=run_case(trainer,initial,pair,arm,number,log,root)
                summary['cases'].append(result); write_json(root/'PROGRESS.json',serializable(summary))
                trainer.close(); trainer=None
        for kind,number in (('adam_step',6),('training_response_forward',12),('core_backward',12)):
            if log.counts[kind+'_attempted']!=number or log.counts[kind+'_completed']!=number:
                raise ValueError('Qualification budget mismatch')
        after={path:identity(path) for path in summary['input_identities']}
        if after!=summary['input_identities']:
            raise ValueError('Qualification modified source tensors/configuration')
        summary.update(status='passed',input_states_unchanged=True)
    except BaseException as error:
        if trainer is not None: trainer.failed=True
        summary.update(status='failed',error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        retained=[str(p) for p in sorted(root.rglob('disposable_decode.pt'))]
        summary.update(work=dict(log.counts),wall_s=time.perf_counter()-started,
            scientific_states_retained=bool(retained),retained_disposable_state_paths=retained,
            production_state_reuse=False,uncompleted_cases=6-len(summary['cases']))
        write_json(root/'QUALIFICATION.json',serializable(summary))
        try:
            if trainer is not None: trainer.close()
        except BaseException as error:
            summary.update(status='failed',cleanup_error=repr(error)); raise
        finally:
            write_json(root/'QUALIFICATION.json',serializable(summary))
    return summary


def main():
    from .campaign import install_signal_handlers
    install_signal_handlers()
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('root',type=Path)
    args=parser.parse_args(); plan=json.loads((args.root/'sealed/PLAN.json').read_text()); q=plan['qualification']
    run(q['config'],q['initial_state'],q['prepared'],args.root/'qualification',
        execution_commit=plan['execution_source']['commit'])


if __name__=='__main__': main()
