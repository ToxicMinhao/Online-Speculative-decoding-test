"""One continuous own-feedback CODE arm on the qualified two-card controller."""
from pathlib import Path
import time
import traceback

from online_feedback.storage import write_json
from experiments.ok_coding.campaign import WorkLedger as ArmLedger, read_work
from experiments.full_k_gsm8k.campaign import online_record_check, save_boundary
from experiments.full_k_gsm8k.costs import begin_event_profile, finish_event_profile
from experiments.osd_persistent.run import check_event, check_first_updated_request
from experiments.o_half_matched_lr.archive import full_trace_errors
from .contract import ARMS, EVENTS, event_requests
from .runtime import WorkLedger, read, boundary, require_eligible_response
from .storage import retire_redundant_features, retire_window, retire_penultimate_export
from .compact import archive_completed_event

ARM_LIMITS = dict(online_generations=7424,correctness_generations=4,
    terminal_preparation_forwards=7424,response_forwards=14848,
    response_backwards=14848,optimizer_steps=7424)


def expected_work(events):
    return dict(online_generations=events*128,correctness_generations=4 if events else 0,
        terminal_preparation_forwards=events*128,response_forwards=events*256,
        response_backwards=events*256,optimizer_steps=events*128)


def run_arm(plan, arm):
    from online_feedback.controller import OnlineController
    from online_feedback.sglang_backend import SGLangDFlashBackend
    from .training import CodeTrainer

    if arm not in ARMS:
        raise ValueError('Only the three prescribed CODE arms can train')
    root = Path(plan['root'])
    # AR/F compatibility must be settled before spending a long trajectory.
    for path in ('qualification/QUALIFICATION.json','precheck/AR_F_PRECHECK.json'):
        if read(root/path)['status']!='passed':
            raise ValueError('Native qualification and full-panel parity are prerequisites')
    for preceding in ARMS[:ARMS.index(arm)]:
        if read(root/'campaign_001/arms'/preceding/'STATUS.json')['status']!='completed':
            raise ValueError('Preserve the declared O, AREPO, C execution order')
    base = root/'campaign_001/arms'/arm
    base.mkdir(parents=True,exist_ok=False)
    ledger = ArmLedger(base/'WORK.jsonl',ARM_LIMITS)
    shared = WorkLedger(root,arm,'production')
    inputs = read(plan['manifest'])
    progress = dict(status='running',phase='initializing',arm=arm,binding=plan['binding'],
        events_completed=0,event_costs=[],generation_phase='online',automatic_retry=False)
    backend = trainer = None
    started = time.perf_counter()
    previous_export = None
    try:
        write_json(base/'ADMISSION.json',boundary(plan))
        backend = SGLangDFlashBackend(plan['target'],plan['draft'],
            worker_python=plan['worker_python'],output_root=base/'native',
            serving_gpu=plan['devices']['A'],context_limit=4096,block_size=16,
            worker_module='experiments.persistent_prefix_withholding.worker')
        generate, prepare = backend.generate, backend.prepare_feedback
        def generate_once(*args,**kwargs):
            correctness = progress['generation_phase']=='correctness'
            kind = 'correctness_generations' if correctness else 'online_generations'
            global_kind = 'correctness_generation' if correctness else 'feedback_generation'
            return ledger.call(kind,shared.call,{global_kind:1},kind,generate,*args,**kwargs)
        def prepare_once(record,path):
            require_eligible_response(record)
            receipt = ledger.call('terminal_preparation_forwards',shared.call,
                {'teacher_preparation':1},'terminal_teacher',prepare,record,path)
            retire_redundant_features(record,receipt,root,Path(progress['event_root']))
            return receipt
        backend.generate,backend.prepare_feedback = generate_once,prepare_once
        trainer = shared.call({'training_initialization':1},'fresh_constructor',CodeTrainer,
            plan['configs'][arm],base/'trainer',arm=arm,binding=plan['binding'],
            original_config_text=Path(plan['original_config']).read_text())
        progress.update(backend_loading_s=backend.loading_s)
        initial_started = time.perf_counter()
        progress['initial'] = trainer.save_initial(plan['initial_witness'])
        if arm!=ARMS[0]:
            import torch
            from .training import compare_initial_witnesses
            first = torch.load(root/'campaign_001/arms'/ARMS[0]/'trainer/INITIAL_WITNESS.pt',map_location='cpu',weights_only=False)
            own = torch.load(base/'trainer/INITIAL_WITNESS.pt',map_location='cpu',weights_only=False)
            progress['initial_cross_arm'] = compare_initial_witnesses(first,own)
            del first,own
        progress['initial_validation_s'] = time.perf_counter()-initial_started
        for owner,name,local,global_kind in (
                (trainer.core.strategy,'forward_loss','response_forwards','response_forward'),
                (trainer.backend,'backward','response_backwards','response_backward'),
                (trainer.backend.optimizer.optimizer,'step','optimizer_steps','adam')):
            ledger.wrap(owner,name,local)
            shared.wrap(owner,name,global_kind)
        for event in range(1,EVENTS+1):
            event_id = f'event_{event:03d}'
            event_root = base/'events'/event_id
            event_root.mkdir(parents=True,exist_ok=False)
            write_json(event_root/'ADMISSION.json',boundary(plan))
            progress.update(phase='collecting',generation_phase='online',current_event=event,event_root=str(event_root))
            write_json(base/'STATUS.json',progress)
            controller = OnlineController(backend,trainer,event_root/'loop')
            records=[]
            for request in event_requests(inputs,event):
                record=controller.serve(request['prompt_ids'],request_id=request['request_id'],
                    session_id='opencode_train_'+arm,request_index=request['request_index'],max_new_tokens=2048)
                online_record_check(record,request,previous_export or plan['draft'],capture=True)
                require_eligible_response(record)
                records.append(record)
            if event==1:
                progress['generation_phase']='correctness'
                probe=records[0]
                replay=controller.evaluate(probe['prompt_ids'],canonical_response_ids=probe['response_ids'],
                    label='first_capture_disabled',max_new_tokens=2048)
                if (replay['record']['finish_reason']!=probe['finish_reason']
                        or replay['record']['metadata']['verification_calls']!=probe['metadata']['verification_calls']):
                    raise ValueError('Capture-disabled guard differs')
            progress.update(phase='training',generation_phase='online')
            write_json(base/'STATUS.json',progress)
            before=begin_event_profile()
            try:
                receipt=controller.update(event_id)
            finally:
                write_json(event_root/'RESOURCE_COSTS.json',finish_event_profile(before))
            check_event(receipt,trainer,backend,event_root,event,128)
            complete=save_boundary(root,arm,trainer,receipt)
            export=Path(receipt['training']['checkpoint_path'])
            if event==1:
                progress['generation_phase']='correctness'
                checks=check_first_updated_request(backend,inputs['train_requests'][128],plan['draft'],str(export),event_root)
                returned=[read(event_root/f'qualification_{name}.json') for name in ('updated','frozen','reloaded')]
                if len({row['finish_reason'] for row in returned})!=1 or any(full_trace_errors(row) for row in returned):
                    raise ValueError('Updated/frozen/reloaded native guard failed')
                write_json(event_root/'INHERITED_CORRECTNESS.json',dict(checks,generations=4,
                    first_capture_disabled_in_loop=True,additional_optimizer_steps=0))
            costs=dict(read(event_root/f'loop/events/{event_id}.costs.json'),
                **receipt['training']['costs'],activation=receipt['activation'],
                teacher_preparation_s=sum(row['teacher_preparation_s'] for row in receipt['preparation']),
                feature_persistence_s=sum(row['feature_persistence_s'] for row in receipt['preparation']))
            retirement_started=time.perf_counter()
            retire_window(root,arm,event_root,records,receipt,complete)
            costs['retirement_s']=time.perf_counter()-retirement_started
            progress['event_costs'].append(dict(event=event,costs=costs))
            actual=read_work(base/'WORK.jsonl',ARM_LIMITS)
            if (actual['attempts']!=expected_work(event) or actual['completed']!=expected_work(event)
                    or any(actual['unknown'].values())):
                raise ValueError('Completed CODE event work does not reconcile')
            progress.update(events_completed=event,phase='idle_boundary',generation_phase='online',
                final_checkpoint=str(export),boundary=complete,actual_work=actual)
            write_json(base/'STATUS.json',progress)
            compact_started=time.perf_counter()
            archive_completed_event(root,arm,event,records,zstd=plan['zstd'])
            costs['compact_persistence_s']=time.perf_counter()-compact_started
            write_json(base/'STATUS.json',progress)
            previous_export=export
            if event%8==0:
                from .forecast import at_training_boundary
                at_training_boundary(plan,arm,event,time.perf_counter()-started)
            print(f'{arm} event {event}/58 complete; {trainer.global_step} Adam steps',flush=True)
        trainer.verify_boundary(export_path=previous_export,receipt_path=base/'FINAL_STATE_CHECK.json')
        retire_penultimate_export(root,arm)
        progress.update(status='completed',phase='completed')
    except BaseException as error:
        progress.update(status='failed',error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        # Preserve the original scientific failure before any optional worker RPC.
        write_json(base/'STATUS.json',progress)
        errors=[]
        if backend is not None and not backend.failed:
            try:
                write_json(base/'PHYSICAL_WORK.json',backend._call('work_receipt'))
            except BaseException as error:
                errors.append(repr(error))
        for owner in (trainer,backend):
            if owner is not None:
                try:
                    owner.close()
                except BaseException as error:
                    errors.append(repr(error))
        progress.update(worker_wall_s=time.perf_counter()-started,
            actual_work=read_work(base/'WORK.jsonl',ARM_LIMITS),cleanup_errors=errors)
        if errors:
            progress['status']='failed'
        write_json(base/'STATUS.json',progress)
        if errors:
            raise RuntimeError('Owned worker cleanup failed: '+repr(errors))
    return progress
