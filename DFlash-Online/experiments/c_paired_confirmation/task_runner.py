"""Single dataset task with shared clocks, native consumers and owned feedback."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import re
import signal
import stat
import time
import traceback

from online_feedback.storage import write_json
from experiments.persistent_prefix_withholding.task_runner import create_backend, close_backend
from experiments.persistent_prefix_withholding.qualification_entry import HostMemorySample
from experiments.ok_coding.storage import identity
from .runtime import WorkLedger, read, boundary, exclusive, occupied_card

DATASETS = ('gsm8k', 'mathqa')
TASKS = ('warmup_references', 'references', 'N_first', 'C_first', 'N_collect',
         'C_collect', 'N_second', 'C_second', 'final')


def task_path(plan, dataset, task):
    return Path(plan['root']) / 'tasks' / (dataset + '_' + task)


def dataset_plan(plan, dataset):
    if dataset not in DATASETS:
        raise ValueError('Only the two sealed datasets are authorized')
    return plan['datasets'][dataset]


def require_production(plan, dataset, task):
    """Direct task entry cannot bypass the qualification and resource gate."""
    root = Path(plan['root'])
    if task == 'warmup_references':
        launch = read(root/'LAUNCH.json')
        if launch['status'] != 'qualification' or (root/'PRODUCTION_STARTED.json').exists():
            raise ValueError('Warmup references belong only to the initial qualification phase')
        return
    for name in ('QUALIFICATION.json', 'RESOURCE_ADMISSION.json'):
        path = plan.get('adopted_qualification_path', root/name) if name == 'QUALIFICATION.json' else root/name
        receipt = read(path)
        if (receipt['status'] != 'passed' or receipt['binding'] != plan['binding']
                or receipt['execution_commit'] != plan['execution_commit']):
            raise ValueError('Production requires the current complete gate: '+name)
    if task == 'final':
        for name in DATASETS:
            for prerequisite in ('N_first','C_first','N_collect','C_collect','N_second','C_second','references'):
                if read(task_path(plan,name,prerequisite)/'STATUS.json')['status'] != 'completed':
                    raise ValueError('All training and preparation must end before final timing')
        if dataset == 'mathqa' and read(task_path(plan,'gsm8k','final')/'STATUS.json')['status'] != 'completed':
            raise ValueError('The fixed final order is GSM8K then MathQA')


def endpoint(plan, dataset, arm, event):
    sub = dataset_plan(plan,dataset)
    if arm not in ('N','C') or event not in (sub['first_event'],sub['second_event']):
        raise ValueError('Unknown paired event boundary')
    suffix = 'first' if event == sub['first_event'] else 'second'
    result = read(task_path(plan,dataset,arm+'_'+suffix)/'RESULT.json')
    if (result['status'] != 'completed' or result['dataset'] != dataset or result['arm'] != arm
            or result['event'] != event or result['binding'] != plan['binding']
            or result['execution_commit'] != plan['execution_commit']):
        raise ValueError('Require this trajectory’s complete own endpoint')
    saved = result['endpoint']
    if saved['global_step'] != event*128 or saved['presentations'] != event*256:
        raise ValueError('Endpoint cumulative counters differ')
    if (identity(saved['state']) != saved['state_identity'] or
            identity(Path(saved['export'])/'model.safetensors') != saved['weights_identity']):
        raise ValueError('Required own recovery or serving export changed')
    return saved


def feature_pool(plan, dataset, arm, event):
    """Use the shared first pool, then only this arm's newly produced packets."""
    sub = dataset_plan(plan,dataset)
    if arm not in ('N','C') or event not in (sub['first_event'],sub['second_event']):
        raise ValueError('Unknown feedback consumer')
    if event == sub['first_event']:
        paths = sub['common_feature_paths']
        expected = sub['common_feature_identities']
    else:
        base = task_path(plan,dataset,arm+'_collect')
        pool = read(base/'FEATURES.json')
        paths = [str(base/'loop/features'/(row['request_id']+'.pt'))
                 for row in read(sub['next_inputs'])['requests']]
        producer = endpoint(plan,dataset,arm,sub['first_event'])
        if (pool['status'] != 'completed' or pool['dataset'] != dataset or pool['arm'] != arm
                or pool['event'] != event or pool['binding'] != plan['binding']
                or pool['execution_commit'] != plan['execution_commit']
                or pool['own_checkpoint'] != producer['export']
                or pool['own_weights_identity'] != producer['weights_identity']
                or pool['feature_paths'] != paths or pool['generated_responses'] != 128
                or pool['teacher_preparations'] != 128):
            raise ValueError('Require complete own-arm feedback with exact producer binding')
        expected = {path:pool['feature_bindings'][path]['file_identity'] for path in paths}
    if len(paths) != 128 or len(set(paths)) != 128 or set(expected) != set(paths):
        raise ValueError('Require exactly 128 distinct authenticated feature packets')
    for path in paths:
        if identity(path) != expected[path]:
            raise ValueError('Feature file changed: '+path)
    return paths


def packet_binding(path, record, producer):
    """Bind all tensor contents, canonical tokens and the actual collecting export."""
    import torch
    from experiments.persistent_prefix_withholding.qualification import tensor_identity
    packet = torch.load(path,map_location='cpu',weights_only=False)
    combined = record['prompt_ids'] + record['response_ids']
    expected_mask = torch.zeros((1,len(combined)),dtype=torch.float32)
    expected_mask[:,len(record['prompt_ids']):] = 1
    if (packet['record_id'] != record['record_id'] or packet['input_ids'].tolist() != [combined]
            or packet['prompt_length'] != len(record['prompt_ids'])
            or not torch.equal(packet['loss_mask'],expected_mask)
            or record['checkpoint_id'] != producer['export']):
        raise ValueError('Prepared packet changed response-only conditioning or producer')
    return dict(file_identity=identity(path),record_id=record['record_id'],
        producer_checkpoint=producer['export'],producer_weights_identity=producer['weights_identity'],
        prompt_ids=record['prompt_ids'],response_ids=record['response_ids'],
        tensors={key:tensor_identity(value) for key,value in packet.items() if torch.is_tensor(value)})


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
        if base.parent != root / "tasks" or re.fullmatch(r"(gsm8k|mathqa)_[NC]_collect", base.name) is None or base.resolve() != base:
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



def train_task(plan, dataset, task, base, ledger, uuid):
    from .training import PairedContinuationTrainer
    from experiments.prefix_guard.qualify import pin_retained_graph_runtime
    from experiments.functional_step_geometry.native_pilot import discarded_startup
    from online_feedback.rng import preserve_rng

    sub = dataset_plan(plan,dataset)
    arm = task[0]
    event = sub['first_event'] if task.endswith('_first') else sub['second_event']
    features = feature_pool(plan,dataset,arm,event)
    previous = endpoint(plan,dataset,arm,sub['first_event']) if task.endswith('_second') else None
    result = dict(status='running',dataset=dataset,arm=arm,event=event,binding=plan['binding'],
        execution_commit=plan['execution_commit'],parent=previous or sub['origin_state'])
    trainer, locked = None, False
    publication = exclusive(Path(plan['root'])/'PUBLICATION.lock')
    try:
        result['runtime_compatibility'] = pin_retained_graph_runtime(base)
        began = time.perf_counter()
        trainer = ledger.call({'training_initialization':1},'trainer_initialization',PairedContinuationTrainer,
            sub['configs'][arm],base/'trainer',dataset=dataset,arm=arm,binding=plan['binding'],
            origin_state=sub['origin_state'],origin_export=sub['origin_export'],
            original_config_text=Path(sub['origin_config']).read_text(),completed_event=event-1,
            origin_identity=sub['origin_state_identity'],physical_gpu_uuid=uuid,
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
            began = time.perf_counter()
            publication.__enter__()
            locked = True
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
        if trainer is not None:
            trainer.failed = True
        raise
    finally:
        if locked:
            publication.__exit__(None,None,None)
        if trainer is not None:
            import torch
            write_json(base/'MEMORY.json',dict(peak_gpu_allocated_bytes=int(torch.cuda.max_memory_allocated()),
                peak_gpu_reserved_bytes=int(torch.cuda.max_memory_reserved()),
                host_maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))
            trainer.close()


def collect_task(plan, dataset, task, base, ledger, uuid):
    from experiments.full_k_gsm8k.campaign import online_record_check
    from experiments.kplus_full_gsm8k.storage import retire_redundant_features
    sub = dataset_plan(plan,dataset)
    event, arm = sub['second_event'], task[0]
    origin = endpoint(plan,dataset,arm,sub['first_event'])
    requests = read(sub['next_inputs'])['requests']
    if len(requests) != 128 or len({r['request_id'] for r in requests}) != 128:
        raise ValueError('Collection requires its complete distinct sealed 128-request window')
    result = dict(status='preparing',dataset=dataset,arm=arm,event=event,binding=plan['binding'],
        execution_commit=plan['execution_commit'],own_checkpoint=origin['export'],
        own_weights_identity=origin['weights_identity'],generated_responses=0,
        teacher_preparations=0,feature_paths=[],feature_bindings={},generation_s=0.,
        preparation_s=0.,feature_binding_s=0.,redundant_capture_retirement_s=0.)
    (base/'loop/features').mkdir(parents=True)
    (base/'records').mkdir()
    backend = None
    try:
        backend = create_backend(plan,base,origin['export'],uuid)
        result['loading_s'] = backend.loading_s
        for request in requests:
            boundary(plan,upcoming_bytes=plan['storage']['temporary_feature_bytes'])
            began = time.perf_counter()
            record = ledger.call({'feedback_generation':1},request['request_id'],backend.generate,
                request['prompt_ids'],max_new_tokens=2048,request_id=request['request_id'],
                session_id=f'{dataset}_paired_confirmation_event{event}',request_index=request['request_index'],
                purpose='serving',save_to=base/'records'/(request['request_id']+'.json'))
            result['generation_s'] += time.perf_counter()-began
            result['generated_responses'] += 1
            write_json(base/'FEATURES.json',result)
            online_record_check(record,request,origin['export'],capture=True)
            path = base/'loop/features'/(request['request_id']+'.pt')
            began = time.perf_counter()
            prepared = ledger.call({'teacher_preparation':1},request['request_id'],backend.prepare_feedback,record,path)
            result['preparation_s'] += time.perf_counter()-began
            result['teacher_preparations'] += 1
            write_json(base/'FEATURES.json',result)
            if prepared.get('status') != 'prepared':
                raise ValueError('No eligible packet; replacement is not allowed')
            began = time.perf_counter()
            result['feature_bindings'][str(path)] = packet_binding(path,record,origin)
            result['feature_binding_s'] += time.perf_counter()-began
            # Admission prices one request of overlap. Failure stops here and
            # retains the payloads; it never licenses silent deletion or growth.
            began = time.perf_counter()
            check_retirement_live_use(record,prepared,plan['root'],base,backend.worker.pid)
            retire_redundant_features(record,prepared,plan['root'],base)
            result['redundant_capture_retirement_s'] += time.perf_counter()-began
            result['feature_paths'].append(str(path))
            write_json(base/'FEATURES.json',result)
        result['status'] = 'completed'
        write_json(base/'FEATURES.json',result)
        return result
    except BaseException as error:
        result.update(status='failed',error=repr(error),traceback=traceback.format_exc())
        write_json(base/'FEATURES.json',result)
        raise
    finally:
        close_backend(backend,base)


def reference_seal(plan, dataset, *, warmups_only=False):
    sub = dataset_plan(plan,dataset)
    keys = ('warmup_references',) if warmups_only else ('references','warmup_references')
    # The two warmup references were completed before the elapsed-only resume.
    # Preserve their original producer identity; new canonical work uses the
    # actual resumed execution commit.
    commit = plan.get('original_execution_commit', plan['execution_commit']) if warmups_only else plan['execution_commit']
    return dict(status='passed',dataset=dataset,binding=plan['binding'],
        execution_commit=commit,checkpoint=sub['origin_export'],
        checkpoint_identity=sub['origin_weights_identity'],reference_count=2 if warmups_only else 258,
        files={sub[key]:identity(sub[key]) for key in keys})


def verify_reference_seal(plan, dataset, *, warmups_only=False):
    sub = dataset_plan(plan,dataset)
    path = Path(sub['references']).parent/('WARMUP_SEAL.json' if warmups_only else 'SEAL.json')
    seal = read(path)
    if seal != reference_seal(plan,dataset,warmups_only=warmups_only):
        raise ValueError('Origin canonical-reference files changed after sealing')
    return seal


def references_task(plan,dataset,task,base,ledger,uuid):
    from experiments.full_k_gsm8k.campaign import online_record_check
    from experiments.fresh_confirmation.protocol import weight_check_errors
    from .evaluation import validate_references
    sub = dataset_plan(plan,dataset)
    warmups_only = task == 'warmup_references'
    requests = read(sub['warmups' if warmups_only else 'data_panel'])['requests']
    if len(requests) != (2 if warmups_only else 256):
        raise ValueError('Canonical subset has the wrong fixed size')
    if not warmups_only:
        verify_reference_seal(plan,dataset,warmups_only=True)
        validate_references(read(sub['warmups'])['requests'],read(sub['warmup_references']),
                            origin_checkpoint=sub['origin_export'])
    checkpoint, backend = sub['origin_export'], None
    refs = {}
    began = time.perf_counter()
    try:
        backend = create_backend(plan,base,checkpoint,uuid)
        (base/'records').mkdir()
        check = backend.check_weights(checkpoint)
        errors = weight_check_errors(check,checkpoint)
        write_json(base/'WEIGHTS_BEFORE.json',dict(check=check,errors=errors))
        if errors:
            raise ValueError('Canonical producer differs from authenticated origin')
        for request in requests:
            boundary(plan)
            path = base/'records'/(request['request_id']+'.json')
            record = ledger.call({'canonical_generation':1},request['request_id'],backend.generate,
                request['prompt_ids'],max_new_tokens=2048,request_id=request['request_id'],
                session_id=dataset+'_paired_confirmation_origin_references',request_index=request['request_index'],
                purpose='evaluation',save_to=path)
            online_record_check(record,request,checkpoint,capture=False)
            reference = dict(prompt_ids=record['prompt_ids'],response_ids=record['response_ids'],
                finish_reason=record['finish_reason'],verified=True,reference_checkpoint=checkpoint,
                reference_kind='native_greedy_target_verifier',source_record_path=str(path),
                source_id=request['source_id'],source_record_identity=identity(path))
            reference['binding_digest'] = hashlib.sha256(json.dumps(reference,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            refs[request['request_id']] = reference
            write_json(base/'CANONICALS.partial.json',refs)
        validate_references(requests,refs,origin_checkpoint=checkpoint)
        check = backend.check_weights(checkpoint)
        errors = weight_check_errors(check,checkpoint)
        write_json(base/'WEIGHTS_AFTER.json',dict(check=check,errors=errors))
        if errors:
            raise ValueError('Origin changed during canonical generation')
        destination = Path(sub['warmup_references' if warmups_only else 'references'])
        if destination.exists():
            raise FileExistsError('Do not overwrite any canonical reference subset')
        write_json(destination,refs)
        seal_path = Path(sub['references']).parent/('WARMUP_SEAL.json' if warmups_only else 'SEAL.json')
        write_json(seal_path,reference_seal(plan,dataset,warmups_only=warmups_only))
        verify_reference_seal(plan,dataset,warmups_only=warmups_only)
        return dict(status='completed',dataset=dataset,binding=plan['binding'],
            execution_commit=plan['execution_commit'],reference_count=len(refs),
            total_sealed_reference_count=2 if warmups_only else 258,
            checkpoint=checkpoint,loading_s=backend.loading_s,wall_s=time.perf_counter()-began,
            new_teacher_features=0,new_training=0)
    finally:
        close_backend(backend,base)


def final_task(plan,dataset,base,ledger,uuid):
    from .evaluation import evaluate_final, evaluation_schedule
    from .analysis import write_report
    from experiments.kplus_full_gsm8k.preflight import timing_barrier
    sub = dataset_plan(plan,dataset)
    verify_reference_seal(plan,dataset)
    requests = read(sub['data_panel'])['requests']
    schedule = read(sub['evaluation_schedule'])['activations']
    if schedule != evaluation_schedule(requests):
        raise ValueError('Actual final route differs from its pre-generation seal')
    checkpoints = {arm:endpoint(plan,dataset,arm,sub['second_event'])['export'] for arm in ('N','C')}
    backend = None
    try:
        backend = create_backend(plan,base,checkpoints['N'],uuid)
        generate = backend.generate
        def counted(*args,**kwargs):
            kind = 'warmup_generation' if kwargs['request_id'].startswith('warmup_') else 'scored_generation'
            return ledger.call({kind:1},kwargs['request_id'],generate,*args,**kwargs)
        backend.generate = counted
        def quiet():
            timing_barrier(backend.worker.pid)
            boundary(plan)
            return dict(status='passed',physical_gpu_uuid=uuid,both_cards_checked=True,
                other_gpu_idle=True,host_quiet=True,storage_admitted=True,time_admitted=True)
        evaluation = base/'evaluation'
        status = evaluate_final(backend,requests,checkpoints,evaluation,dataset=dataset,
            canonicals=read(sub['references']),warmup_requests=read(sub['warmups'])['requests'],
            warmup_canonicals=read(sub['warmup_references']),schedule=schedule,
            physical_gpu_uuid=uuid,boundary_check=quiet,execution_commit=plan['execution_commit'],
            origin_checkpoint=sub['origin_export'])
        summary = write_report(evaluation)
        if summary['status'] != 'completed':
            raise ValueError('Final comparison invalid or incomplete')
        return dict(status='completed',dataset=dataset,evaluation=status,
            summary=str(evaluation/'SUMMARY.json'),checkpoints=checkpoints,loading_s=backend.loading_s)
    finally:
        close_backend(backend,base)


def run_task(plan_path,dataset,task,uuid):
    plan = read(plan_path)
    dataset_plan(plan,dataset)
    if (task not in TASKS or uuid not in plan['devices'].values()
            or os.environ.get('CUDA_VISIBLE_DEVICES') != uuid):
        raise ValueError('Task must run on exactly its assigned physical card')
    if task == 'final' and uuid != plan['devices']['A']:
        raise ValueError('Fixed timing card is A')
    require_production(plan,dataset,task)
    base = task_path(plan,dataset,task)
    base.mkdir(parents=True,exist_ok=False)
    task_id = dataset+'_'+task
    receipt = dict(status='running',task=task,dataset=dataset,physical_gpu_uuid=uuid,
        execution_commit=plan['execution_commit'],binding=plan['binding'],
        started_epoch=time.time(),automatic_retry=False)
    write_json(base/'STATUS.json',receipt)
    ledger = WorkLedger(plan['root'],task_id)
    monitor = HostMemorySample()
    def interrupt(signum,frame):
        raise InterruptedError('Owned task interrupted; evidence preserved')
    for number in (signal.SIGTERM,signal.SIGINT):
        signal.signal(number,interrupt)
    try:
        receipt['initial_resource_boundary'] = boundary(plan,compiler=True)
        with occupied_card(plan['root'],task_id,uuid),monitor:
            if task.endswith(('_first','_second')):
                result = train_task(plan,dataset,task,base,ledger,uuid)
            elif task.endswith('_collect'):
                result = collect_task(plan,dataset,task,base,ledger,uuid)
            elif task in ('references','warmup_references'):
                result = references_task(plan,dataset,task,base,ledger,uuid)
            else:
                result = final_task(plan,dataset,base,ledger,uuid)
            receipt['final_resource_boundary'] = boundary(plan,compiler=True)
        write_json(base/'RESULT.json',result)
        receipt.update(status='completed',result=str(base/'RESULT.json'))
    except BaseException as error:
        receipt.update(status='failed',error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        receipt.update(wall_s=time.time()-receipt['started_epoch'],
            sampled_process_tree_peak_rss_bytes=monitor.maximum_bytes)
        write_json(base/'STATUS.json',receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True)
    parser.add_argument('--dataset',choices=DATASETS,required=True)
    parser.add_argument('--task',choices=TASKS,required=True)
    parser.add_argument('--gpu-uuid',required=True)
    args = parser.parse_args()
    run_task(args.plan,args.dataset,args.task,args.gpu_uuid)


if __name__ == '__main__':
    main()
