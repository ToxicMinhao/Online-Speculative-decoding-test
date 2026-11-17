"""One owned MathQA task; model work is charged before it is attempted."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import re
import stat
import signal
import time
import traceback

from online_feedback.storage import write_json
from experiments.persistent_prefix_withholding.task_runner import create_backend, close_backend
from experiments.persistent_prefix_withholding.qualification_entry import HostMemorySample
from .runtime import WorkLedger, read, boundary, exclusive, occupied_card

TASKS = ('references', 'C5', 'C_collect6', 'C6', 'final')


def verify_reference_seal(plan):
    """All 258 origin outputs are immutable before any qualification replay."""
    from experiments.bounded_prefix_allocation.prepare import identity
    seal=read(Path(plan['root'])/'references/SEAL.json')
    expected=dict(status='passed',binding=plan['binding'],execution_commit=plan['execution_commit'],
        checkpoint=plan['origin_export'],checkpoint_identity=plan['origin_weights_identity'],
        reference_count=258,files={plan[key]:identity(plan[key]) for key in ('references','warmup_references')})
    if seal!=expected:raise ValueError('Origin canonical-reference files changed after sealing')
    return seal


def endpoint(plan, arm, event):
    """New C boundaries and authenticated retained N/S endpoints stay distinct."""
    from .prepare import identity
    if arm in ('N', 'S'):
        saved = plan['retained_endpoints'][f'{arm}{event}']['endpoint']
        if (identity(saved['state']) != saved['state_identity'] or
                identity(Path(saved['export'])/'model.safetensors') != saved['weights_identity']):
            raise ValueError('Retained control boundary changed')
    elif arm == 'C' and event in (5, 6):
        result = read(Path(plan['root'])/'tasks'/f'C{event}'/'RESULT.json')
        if (result['status'] != 'completed' or result['arm'] != arm or result['event'] != event
                or result['binding'] != plan['binding'] or result['execution_commit'] != plan['execution_commit']):
            raise ValueError('Require this campaign’s complete C endpoint')
        saved = result['endpoint']
    else:
        raise ValueError('Unknown arm or event boundary')
    if saved['global_step'] != event*128 or saved['presentations'] != event*256:
        raise ValueError('Endpoint cumulative counters differ')
    return saved


def feature_pool(plan, arm, event):
    """Borrow the authenticated common E5 pool; only C owns new E6 feedback."""
    from .prepare import identity
    if arm != 'C' or event not in (5, 6):
        raise ValueError('Only C events 5 and 6 are authorized')
    if event == 5:
        common = plan['common_e5']
        if identity(common['manifest']) != common['manifest_identity']:
            raise ValueError('Original common E5 manifest changed')
        pool = read(common['manifest'])
        expected_binding = common['binding']
        expected_commit = common['execution_commit']
        paths = common['feature_paths']
        origin = dict(export=plan['origin_export'], weights_identity=plan['origin_weights_identity'])
    else:
        base = Path(plan['root'])/'tasks/C_collect6'
        pool = read(base/'FEATURES.json')
        expected_binding, expected_commit = plan['binding'], plan['execution_commit']
        paths = [str(base/'loop/features'/(r['request_id']+'.pt'))
                 for r in read(plan['event6_inputs'])['requests']]
        origin = endpoint(plan, 'C', 5)
    if (pool['status'] != 'completed' or pool['binding'] != expected_binding
            or pool['execution_commit'] != expected_commit or pool['event'] != event
            or pool['arm'] != ('COMMON' if event == 5 else 'C')
            or pool['own_checkpoint'] != origin['export']
            or pool['own_weights_identity'] != origin['weights_identity']
            or pool['feature_paths'] != paths or len(paths) != 128
            or pool['generated_responses'] != 128 or pool['teacher_preparations'] != 128):
        raise ValueError('Require exact original E5 or C5-owned E6 feedback')
    return paths


def check_retirement_live_use(record, prepared, root, base, backend_pid):
    """Check this worker's exact new duplicates before their existing byte check.

    Collection is synchronous and these paths belong exclusively to this task.
    Its parent process and descendants include the native backend and engine;
    no other task receives these temporary paths. Inspecting unrelated users'
    processes is neither necessary nor sufficient for this ownership contract.
    """
    import psutil

    record_id = record["record_id"]
    if Path(record_id).name != record_id:
        raise ValueError("Unsafe feedback record identity")
    receipt_path = Path(base) / "preparation_retirement" / (record_id + ".LIVE_USE.json")
    receipt = dict(status="checking", record_id=record_id, parent_pid=os.getpid(),
        backend_pid=backend_pid, scope="current collection process and all descendants",
        targets=[], inspected_processes=[], matches=[], additional_model_forwards=0)
    started = time.perf_counter()
    try:
        base, root = Path(base).absolute(), Path(root).absolute()
        if base.parent != root / "tasks" or base.name != "C_collect6" or base.resolve() != base:
            raise ValueError("Retirement must stay inside the owned collection task")
        feature = base / "loop/features" / (record_id + ".pt")
        trace = Path(record["metadata"]["native_trace_path"])
        capture = Path(record["metadata"]["capture"]["feature_path"])
        terminal = Path(prepared["terminal_feature"]["tensor_path"])
        if (Path(prepared["feature_path"]) != feature or trace.parent != base / "native"
                or re.fullmatch(r"request_[0-9]{6}\.json", trace.name) is None
                or capture != trace.with_suffix(".pt")
                or terminal != feature.with_suffix(".terminal.pt")):
            raise ValueError("Retirement paths differ from this response's exact native paths")
        targets = {str(capture), str(terminal)}
        for path in (capture, terminal):
            info = path.lstat()
            if (path.resolve() != path or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1):
                raise ValueError("Retirement target is linked or is not an exclusive regular file")
            receipt["targets"].append(dict(path=str(path), device=info.st_dev, inode=info.st_ino,
                                           bytes=info.st_size, links=info.st_nlink))
        parent = psutil.Process(os.getpid())
        processes = [parent, *parent.children(recursive=True)]
        if backend_pid not in {process.pid for process in processes}:
            raise ValueError("Native backend is absent from the owned process tree")
        for process in processes:
            row = dict(pid=process.pid, open_file_count=0, mapped_file_count=0)
            receipt["inspected_processes"].append(row)
            for kind, entries in (("open", process.open_files()),
                                  ("mapped", process.memory_maps(grouped=False))):
                row["open_file_count" if kind == "open" else "mapped_file_count"] = len(entries)
                for entry in entries:
                    # /proc can append this suffix to a still-live mapping.
                    path = entry.path.removesuffix(" (deleted)")
                    if path.startswith("/") and os.path.realpath(path) in targets:
                        receipt["matches"].append(dict(pid=process.pid, kind=kind, path=entry.path))
        if receipt["matches"]:
            raise ValueError("Disposable feature file is still open or memory-mapped")
        receipt["status"] = "passed"
    except BaseException as error:
        receipt.update(status="failed", error=repr(error))
        raise
    finally:
        receipt["inspection_s"] = time.perf_counter() - started
        write_json(receipt_path, receipt)
    return receipt


def train_task(plan, task, base, ledger):
    from .training import MathQACandidateTrainer
    from experiments.prefix_guard.qualify import pin_retained_graph_runtime
    from experiments.functional_step_geometry.native_pilot import discarded_startup
    from experiments.ok_coding.storage import identity
    from online_feedback.rng import preserve_rng

    arm, event = task[0], int(task[1:])
    features = feature_pool(plan,arm,event)
    previous = endpoint(plan,arm,5) if event == 6 else None
    result = dict(status='running',arm=arm,event=event,binding=plan['binding'],
        execution_commit=plan['execution_commit'],parent=previous or plan['origin_state'])
    trainer, locked = None, False
    publication = exclusive(Path(plan['root'])/'PUBLICATION.lock')
    try:
        result['runtime_compatibility'] = pin_retained_graph_runtime(base)
        began = time.perf_counter()
        trainer = ledger.call({'training_initialization':1},'trainer_initialization',MathQACandidateTrainer,
            plan['configs'][arm],base/'trainer',arm=arm,binding=plan['binding'],
            origin_state=plan['origin_state'],origin_export=plan['origin_export'],
            original_config_text=Path(plan['origin_config']).read_text(),completed_event=event-1,
            resume_state=previous['state'] if previous else None,
            resume_export=previous['export'] if previous else None)
        result['loading_s'] = time.perf_counter()-began
        ledger.wrap(trainer.strategy,'forward_loss','response_forward')
        ledger.wrap(trainer.backend,'backward','response_backward')
        ledger.wrap(trainer.backend.optimizer.optimizer,'step','adam')
        result['startup'] = discarded_startup(trainer,features,base/'startup')
        ordinary_passes = trainer._train_passes
        def passes_then_publication(*args,**kwargs):
            nonlocal locked
            rows = ordinary_passes(*args,**kwargs)
            began = time.perf_counter();publication.__enter__();locked=True
            result['publication_wait_s'] = time.perf_counter()-began
            boundary(plan,upcoming_bytes=plan['storage']['atomic_publication_bytes'])
            return rows
        trainer._train_passes = passes_then_publication
        began = time.perf_counter()
        receipt = trainer.train_event(f'event_{event:03d}',features,2)
        result['event_wall_s'] = time.perf_counter()-began
        result['boundary'] = trainer.verify_boundary(receipt['checkpoint_path'],base/'BOUNDARY.json')
        if (trainer.global_step,trainer.presentations,trainer.core.accumulation_remainder) != (event*128,event*256,0):
            raise ValueError('Training did not finish at the declared complete boundary')
        with preserve_rng():
            state_id = identity(trainer.latest)
            weights_id = identity(Path(receipt['checkpoint_path'])/'model.safetensors')
        result.update(status='completed',costs=receipt['costs'],physical_work=dict(trainer.work),
            endpoint=dict(state=str(trainer.latest),export=receipt['checkpoint_path'],
                state_identity=state_id,weights_identity=weights_id,
                global_step=trainer.global_step,presentations=trainer.presentations))
        return result
    except BaseException as error:
        result.update(status='failed',error=repr(error),traceback=traceback.format_exc())
        write_json(base/'FAILED.json',result)
        if trainer is not None:trainer.failed=True
        raise
    finally:
        if locked:publication.__exit__(None,None,None)
        if trainer is not None:
            import torch
            write_json(base/'MEMORY.json',dict(peak_gpu_allocated_bytes=int(torch.cuda.max_memory_allocated()),
                peak_gpu_reserved_bytes=int(torch.cuda.max_memory_reserved()),
                host_maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))
            trainer.close()


def collect_task(plan, task, base, ledger, uuid):
    import psutil
    from experiments.full_k_gsm8k.campaign import online_record_check
    from experiments.kplus_full_gsm8k.storage import retire_redundant_features
    event, arm = 6, 'C'
    origin = endpoint(plan,arm,5)
    requests = read(plan[f'event{event}_inputs'])['requests']
    if len(requests) != 128 or len({r['request_id'] for r in requests}) != 128:
        raise ValueError('Collection requires its complete distinct sealed 128-request window')
    result = dict(status='preparing',arm=arm,event=event,binding=plan['binding'],
        execution_commit=plan['execution_commit'],own_checkpoint=origin['export'],
        own_weights_identity=origin['weights_identity'],generated_responses=0,
        teacher_preparations=0,feature_paths=[],generation_s=0.,preparation_s=0.,
        redundant_capture_retirement_s=0.,
        duplicate_retirement='Only byte-verified new capture/terminal duplicates; unique prepared packets and compact provenance retained')
    (base/'loop/features').mkdir(parents=True)
    (base/'records').mkdir()
    backend = None
    try:
        backend = create_backend(plan,base,origin['export'],uuid)
        result['loading_s']=backend.loading_s
        for request in requests:
            boundary(plan,upcoming_bytes=plan['storage']['temporary_feature_bytes'])
            began=time.perf_counter()
            record=ledger.call({'feedback_generation':1},request['request_id'],backend.generate,
                request['prompt_ids'],max_new_tokens=2048,request_id=request['request_id'],
                session_id=f'mathqa_unchanged_c_event{event}',request_index=request['request_index'],
                purpose='serving',save_to=base/'records'/(request['request_id']+'.json'))
            result['generation_s']+=time.perf_counter()-began;result['generated_responses']+=1
            write_json(base/'FEATURES.json',result)
            online_record_check(record,request,origin['export'],capture=True)
            path=base/'loop/features'/(request['request_id']+'.pt');began=time.perf_counter()
            prepared=ledger.call({'teacher_preparation':1},request['request_id'],backend.prepare_feedback,record,path)
            result['preparation_s']+=time.perf_counter()-began;result['teacher_preparations']+=1
            write_json(base/'FEATURES.json',result)
            if prepared.get('status')!='prepared':raise ValueError('No eligible packet; replacement is not allowed')
            began=time.perf_counter()
            try:
                check_retirement_live_use(record,prepared,plan['root'],base,backend.worker.pid)
            except (OSError, psutil.Error) as error:
                # A failed live-use scan never authorizes retirement. Peak admission
                # prices these copies as retained, so collection may continue.
                write_json(base/'preparation_retirement'/(request['request_id']+'.DEFERRED.json'),
                    dict(status='retained',reason=repr(error),additional_model_work=0))
            else:
                retire_redundant_features(record,prepared,plan['root'],base)
            result['redundant_capture_retirement_s']+=time.perf_counter()-began
            result['feature_paths'].append(str(path));write_json(base/'FEATURES.json',result)
        result['status']='completed';write_json(base/'FEATURES.json',result)
        return result
    except BaseException as error:
        result.update(status='failed',error=repr(error),traceback=traceback.format_exc())
        write_json(base/'FEATURES.json',result);raise
    finally:
        close_backend(backend,base)


def references_task(plan,base,ledger,uuid):
    from experiments.full_k_gsm8k.campaign import online_record_check
    from experiments.fresh_confirmation.protocol import weight_check_errors
    from .evaluation import validate_references
    from experiments.bounded_prefix_allocation.prepare import identity
    requests=read(plan['data_panel'])['requests'];warmups=read(plan['warmups'])['requests']
    checkpoint=plan['origin_export'];backend=None
    refs={};warmrefs={};began=time.perf_counter()
    try:
        backend=create_backend(plan,base,checkpoint,uuid)
        (base/'records').mkdir()
        check=backend.check_weights(checkpoint);errors=weight_check_errors(check,checkpoint)
        write_json(base/'WEIGHTS_BEFORE.json',dict(check=check,errors=errors))
        if errors:raise ValueError('Canonical producer differs from authenticated KPLUS4')
        for group,destination in ((requests,refs),(warmups,warmrefs)):
            for request in group:
                boundary(plan)
                path=base/'records'/(request['request_id']+'.json')
                record=ledger.call({'canonical_generation':1},request['request_id'],backend.generate,
                    request['prompt_ids'],max_new_tokens=2048,request_id=request['request_id'],
                    session_id='mathqa_unchanged_c_origin_references',request_index=request['request_index'],
                    purpose='evaluation',save_to=path)
                online_record_check(record,request,checkpoint,capture=False)
                reference=dict(prompt_ids=record['prompt_ids'],response_ids=record['response_ids'],
                    finish_reason=record['finish_reason'],verified=True,reference_checkpoint=checkpoint,
                    reference_kind='native_greedy_target_verifier',source_record_path=str(path),
                    source_id=request['source_id'],source_record_identity=identity(path))
                reference['binding_digest']=hashlib.sha256(json.dumps(reference,sort_keys=True,separators=(',',':')).encode()).hexdigest()
                destination[request['request_id']]=reference
                write_json(base/'CANONICALS.partial.json',refs);write_json(base/'WARMUP_CANONICALS.partial.json',warmrefs)
        validate_references(requests,refs,origin_checkpoint=checkpoint)
        validate_references(warmups,warmrefs,origin_checkpoint=checkpoint)
        write_json(plan['references'],refs);write_json(plan['warmup_references'],warmrefs)
        check=backend.check_weights(checkpoint);errors=weight_check_errors(check,checkpoint)
        write_json(base/'WEIGHTS_AFTER.json',dict(check=check,errors=errors))
        if errors:raise ValueError('Origin changed during canonical generation')
        write_json(Path(plan['root'])/'references/SEAL.json',dict(status='passed',binding=plan['binding'],
            execution_commit=plan['execution_commit'],checkpoint=checkpoint,
            checkpoint_identity=plan['origin_weights_identity'],reference_count=258,
            files={plan[key]:identity(plan[key]) for key in ('references','warmup_references')}))
        verify_reference_seal(plan)
        return dict(status='completed',reference_count=len(refs)+len(warmrefs),
            checkpoint=checkpoint,loading_s=backend.loading_s,wall_s=time.perf_counter()-began,
            new_teacher_features=0,new_training=0)
    finally:
        close_backend(backend,base)


def final_task(plan,base,ledger,uuid):
    from .evaluation import evaluate_final,evaluation_schedule
    from .analysis import write_report
    from experiments.kplus_full_gsm8k.preflight import timing_barrier
    verify_reference_seal(plan)
    requests=read(plan['data_panel'])['requests']
    checkpoints={arm:endpoint(plan,arm,6)['export'] for arm in ('N','C','S')}
    backend=None
    try:
        backend=create_backend(plan,base,checkpoints['N'],uuid)
        generate=backend.generate
        def counted(*args,**kwargs):
            kind='warmup_generation' if kwargs['request_id'].startswith('warmup_') else 'scored_generation'
            return ledger.call({kind:1},kwargs['request_id'],generate,*args,**kwargs)
        backend.generate=counted
        def quiet():
            timing_barrier(backend.worker.pid);boundary(plan)
            return dict(status='passed',physical_gpu_uuid=uuid,both_cards_checked=True,
                other_gpu_idle=True,host_quiet=True,storage_admitted=True,time_admitted=True)
        evaluation=base/'evaluation'
        status=evaluate_final(backend,requests,checkpoints,evaluation,
            canonicals=read(plan['references']),warmup_requests=read(plan['warmups'])['requests'],
            warmup_canonicals=read(plan['warmup_references']),schedule=evaluation_schedule(requests),
            physical_gpu_uuid=uuid,boundary_check=quiet,execution_commit=plan['execution_commit'],
            origin_checkpoint=plan['origin_export'])
        summary=write_report(evaluation)
        if summary['status']!='completed':raise ValueError('Final comparison invalid or incomplete')
        return dict(status='completed',evaluation=status,summary=str(evaluation/'SUMMARY.json'),
            checkpoints=checkpoints,loading_s=backend.loading_s)
    finally:
        close_backend(backend,base)


def run_task(plan_path,task,uuid):
    plan=read(plan_path)
    if task not in TASKS or uuid not in plan['devices'].values() or os.environ.get('CUDA_VISIBLE_DEVICES')!=uuid:
        raise ValueError('Task must run on exactly its assigned physical card')
    if task=='final' and uuid!=plan['devices']['A']:raise ValueError('Fixed timing card is A')
    base=Path(plan['root'])/'tasks'/task;base.mkdir(parents=True,exist_ok=False)
    receipt=dict(status='running',task=task,physical_gpu_uuid=uuid,execution_commit=plan['execution_commit'],
        binding=plan['binding'],started_epoch=time.time(),automatic_retry=False)
    write_json(base/'STATUS.json',receipt)
    ledger=WorkLedger(plan['root'],task);monitor=HostMemorySample()
    def interrupt(signum,frame):raise InterruptedError('Owned task interrupted; evidence preserved')
    for number in (signal.SIGTERM,signal.SIGINT):signal.signal(number,interrupt)
    try:
        receipt['initial_resource_boundary']=boundary(plan,compiler=True)
        with occupied_card(plan['root'],task,uuid),monitor:
            if task in ('C5','C6'):result=train_task(plan,task,base,ledger)
            elif task == 'C_collect6':result=collect_task(plan,task,base,ledger,uuid)
            elif task=='references':result=references_task(plan,base,ledger,uuid)
            else:result=final_task(plan,base,ledger,uuid)
            receipt['final_resource_boundary']=boundary(plan,compiler=True)
        write_json(base/'RESULT.json',result);receipt.update(status='completed',result=str(base/'RESULT.json'))
    except BaseException as error:
        receipt.update(status='failed',error=repr(error),traceback=traceback.format_exc());raise
    finally:
        receipt.update(wall_s=time.time()-receipt['started_epoch'],
            sampled_process_tree_peak_rss_bytes=monitor.maximum_bytes)
        write_json(base/'STATUS.json',receipt)
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True);parser.add_argument('--task',choices=TASKS,required=True)
    parser.add_argument('--gpu-uuid',required=True);args=parser.parse_args()
    run_task(args.plan,args.task,args.gpu_uuid)


if __name__=='__main__':main()
