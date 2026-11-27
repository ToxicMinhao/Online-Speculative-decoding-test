"""BASE then TAIL: fresh own-feedback trajectories with authenticated boundaries."""
from pathlib import Path
import time
import traceback

from experiments.ok_coding.campaign import WorkLedger as ArmLedger, read_work
from experiments.full_k_gsm8k.campaign import online_record_check, save_boundary
from experiments.full_k_gsm8k.costs import begin_event_profile, finish_event_profile
from experiments.osd_persistent.run import check_event, check_first_updated_request
from experiments.o_half_matched_lr.archive import full_trace_errors
from .contract import ARMS, EVENTS
from .runtime import WorkLedger, read, write_json, boundary, forecast
from .storage import retire_redundant_features, retire_window
from .compact import archive_completed_event

ARM_LIMITS = dict(online_generations=7424, correctness_generations=4,
    terminal_preparation_forwards=7424, response_forwards=14848,
    response_backwards=14848, optimizer_steps=7424)


def expected_work(events):
    return dict(online_generations=events*128,correctness_generations=4 if events else 0,
        terminal_preparation_forwards=events*128,response_forwards=events*256,
        response_backwards=events*256,optimizer_steps=events*128)


class ResumeLedger(ArmLedger):
    def __init__(self, path, *, resume=False):
        if not resume:
            super().__init__(path, ARM_LIMITS)
            return
        self.path, self.limits = Path(path), dict(ARM_LIMITS)
        saved = read_work(path, ARM_LIMITS)
        if any(saved['unknown'].values()):
            raise ValueError('Resolve interrupted physical work before resuming')
        self.attempts, self.completed = saved['attempts'], saved['completed']


def refresh_forecast(plan, arm, event, worker_wall_s):
    """Predict the remaining fixed work; outcomes never enter admission."""
    baseline = plan['forecast']
    elapsed_per_event = worker_wall_s/event
    historical_per_event = baseline['full_arm_wall_s']/58
    rate = max(elapsed_per_event, historical_per_event)
    remaining_events = 58-event + (58 if arm==ARMS[0] else 0)
    training = rate*remaining_events
    evaluation = baseline['final_evaluation_wall_s']
    return forecast(plan, remaining_wall_s=training+evaluation+baseline['audit_wall_s'],
        remaining_card_s=2*training+evaluation,
        destination=Path(plan['root'])/'forecasts'/arm/f'event_{event:03d}.json')


def run_arm(plan, arm, *, resume=False):
    from online_feedback.controller import OnlineController
    from online_feedback.sglang_backend import SGLangDFlashBackend
    from .training import FullContinuationTrainer
    from .witness import seal_witness, read_witness

    if arm not in ARMS:
        raise ValueError('Only the two prescribed fresh trajectories can train')
    root = Path(plan['root'])
    if read(root/'qualification/QUALIFICATION.json')['status']!='passed':
        raise ValueError('Full-run native qualification has not passed')
    for preceding in ARMS[:ARMS.index(arm)]:
        if read(root/'campaign_001/arms'/preceding/'STATUS.json')['status']!='completed':
            raise ValueError('Run BASE completely before TAIL')
    base = root/'campaign_001/arms'/arm
    base.mkdir(parents=True, exist_ok=resume)
    prior = read(base/'STATUS.json') if resume else None
    completed = prior['events_completed'] if prior else 0
    if prior and (prior['phase'] not in ('idle_boundary','completed') or prior['binding']!=plan['binding']):
        raise ValueError('Resume requires an authenticated complete own boundary; reconcile partial phases explicitly')
    if prior and prior['status']=='completed':
        return prior
    ledger = ResumeLedger(base/'WORK.jsonl',resume=resume)
    shared = WorkLedger(root,arm,'production')
    inputs = read(plan['manifest'])
    progress = prior or dict(status='running',phase='initializing',arm=arm,binding=plan['binding'],
        events_completed=0,event_costs=[],generation_phase='online')
    progress.update(status='running',execution_commit=plan['execution_commit'])
    backend = trainer = None
    started = time.perf_counter()
    old_wall = progress.get('worker_wall_s',0.)
    previous_export = progress.get('checkpoint')
    try:
        write_json(base/'ADMISSION.json',boundary(plan))
        backend = SGLangDFlashBackend(plan['target'],plan['draft'],
            worker_python=plan['worker_python'],output_root=base/'native'/f'load_{completed:03d}',
            serving_gpu=plan['devices']['A'],context_limit=4096,block_size=16,
            worker_module='experiments.persistent_prefix_withholding.worker')
        generate, prepare = backend.generate, backend.prepare_feedback
        def generate_once(*args,**kwargs):
            correctness = progress['generation_phase']=='correctness'
            kind = 'correctness_generations' if correctness else 'online_generations'
            global_kind = 'correctness_generation' if correctness else 'feedback_generation'
            return ledger.call(kind,shared.call,{global_kind:1},kind,generate,*args,**kwargs)
        def prepare_once(record,path):
            from experiments.opencode_native_comparison.runtime import require_eligible_response
            require_eligible_response(record)
            receipt = ledger.call('terminal_preparation_forwards',shared.call,
                {'teacher_preparation':1},'terminal_teacher',prepare,record,path)
            retire_redundant_features(record,receipt,root,Path(progress['event_root']))
            return receipt
        backend.generate,backend.prepare_feedback = generate_once,prepare_once
        state = read(base/'boundaries'/f'event_{completed:03d}.json')['state'] if completed else None
        trainer = shared.call({'training_initialization':1},'fresh_or_restored_trainer',
            FullContinuationTrainer,plan['configs'][arm],base/'trainer',arm=arm,
            binding=plan['binding'],original_config_text=Path(plan['original_config']).read_text(),
            resume_state=state,completed_event=completed)
        progress['backend_loading_s'] = progress.get('backend_loading_s',0.)+backend.loading_s
        if not completed:
            comparator = root/'campaign_001/arms/BASE/trainer/INITIAL_WITNESS.pt' if arm=='TAIL' else None
            progress['initial'] = trainer.save_initial(comparator)
        else:
            if previous_export is None:
                previous_export = str(Path(read(base/'boundaries'/f'event_{completed:03d}.json')['weights']).parent)
            progress['resume_activation'] = backend.publish(previous_export)
            progress['resume_validation'] = trainer.verify_boundary(export_path=previous_export,
                receipt_path=base/f'RESUME_{completed:03d}.json')
        original_train = trainer.train_event
        needs_startup = True
        def train_with_witness(event_id,paths,*,passes=2):
            nonlocal needs_startup
            if needs_startup:
                from experiments.repair_conditional_kl_pilot.training import discarded_startup
                startup = WorkLedger(root,arm,'startup')
                progress['startup'] = startup.call({'response_forward':2,'response_backward':2},
                    'discarded_compile_startup',discarded_startup,trainer,paths,
                    base/f'startup_{completed:03d}')
                # Counting wrappers begin only after the discarded compile work.
                for owner,name,local,global_kind in (
                    (trainer.core.strategy,'forward_loss','response_forwards','response_forward'),
                    (trainer.backend,'backward','response_backwards','response_backward'),
                    (trainer.backend.optimizer.optimizer,'step','optimizer_steps','adam')):
                    ledger.wrap(owner,name,local)
                    shared.wrap(owner,name,global_kind)
                needs_startup = False
            if arm=='BASE' and event_id=='event_001':
                seal_witness(trainer,plan,paths)
                read_witness(trainer,plan,endpoint='released')
            return original_train(event_id,paths,passes=passes)
        trainer.train_event = train_with_witness
        for event in range(completed+1,EVENTS+1):
            event_id = f'event_{event:03d}'
            event_root = base/'events'/event_id
            event_root.mkdir(parents=True,exist_ok=False)
            write_json(event_root/'ADMISSION.json',boundary(plan))
            progress.update(phase='collecting',generation_phase='online',current_event=event,event_root=str(event_root))
            write_json(base/'STATUS.json',progress)
            controller = OnlineController(backend,trainer,event_root/'loop')
            records=[]
            for request in inputs['train_requests'][(event-1)*128:event*128]:
                record=controller.serve(request['prompt_ids'],request_id=request['request_id'],
                    session_id='opencode_train_'+arm,request_index=request['request_index'],max_new_tokens=2048)
                online_record_check(record,request,previous_export or plan['draft'],capture=True)
                records.append(record)
            if event==1:
                progress['generation_phase']='correctness'
                probe=records[0]
                replay=controller.evaluate(probe['prompt_ids'],canonical_response_ids=probe['response_ids'],
                    label='first_capture_disabled',max_new_tokens=2048)
                if (replay['record']['finish_reason']!=probe['finish_reason'] or
                    replay['record']['metadata']['verification_calls']!=probe['metadata']['verification_calls']):
                    raise ValueError('Capture-disabled guard differs')
            progress.update(phase='training',generation_phase='online')
            write_json(base/'STATUS.json',progress)
            profile=begin_event_profile()
            try:
                receipt=controller.update(event_id)
            finally:
                write_json(event_root/'RESOURCE_COSTS.json',finish_event_profile(profile))
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
            if event in (1,16,32,58):
                read_witness(trainer,plan,endpoint=f'{arm}_{event:03d}')
            costs=dict(read(event_root/f'loop/events/{event_id}.costs.json'),
                **receipt['training']['costs'],activation=receipt['activation'],
                teacher_preparation_s=sum(row['teacher_preparation_s'] for row in receipt['preparation']),
                feature_persistence_s=sum(row['feature_persistence_s'] for row in receipt['preparation']))
            retired=time.perf_counter()
            retire_window(root,arm,event_root,records,receipt,complete)
            costs['retirement_s']=time.perf_counter()-retired
            progress['event_costs'].append(dict(event=event,costs=costs))
            actual=read_work(base/'WORK.jsonl',ARM_LIMITS)
            if (actual['attempts']!=expected_work(event) or actual['completed']!=expected_work(event)
                    or any(actual['unknown'].values())):
                raise ValueError('Completed full-run event work does not reconcile')
            progress.update(events_completed=event,phase='idle_boundary',generation_phase='online',
                checkpoint=str(export),final_checkpoint=str(export),checkpoint_identity=complete['weights_identity'],
                global_step=trainer.global_step,presentations=trainer.presentations,
                boundary=complete,actual_work=actual,worker_wall_s=old_wall+time.perf_counter()-started)
            write_json(base/'STATUS.json',progress)
            compact_started=time.perf_counter()
            archive_completed_event(root,arm,event,records,zstd=plan['zstd'])
            costs['compact_persistence_s']=time.perf_counter()-compact_started
            write_json(base/'STATUS.json',progress)
            previous_export=str(export)
            refresh_forecast(plan,arm,event,progress['worker_wall_s'])
            print(f'{arm} event {event}/58 complete; {trainer.global_step} Adam steps',flush=True)
        trainer.verify_boundary(export_path=previous_export,receipt_path=base/'FINAL_STATE_CHECK.json')
        progress.update(status='completed',phase='completed')
    except BaseException as error:
        progress.update(status='failed',error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        write_json(base/'STATUS.json',progress)
        errors=[]
        if backend is not None and not backend.failed:
            try:
                write_json(base/f'PHYSICAL_WORK_{completed:03d}.json',backend._call('work_receipt'))
            except BaseException as error:
                errors.append(repr(error))
        for owner in (trainer,backend):
            if owner is not None:
                try:
                    owner.close()
                except BaseException as error:
                    errors.append(repr(error))
        progress.update(worker_wall_s=old_wall+time.perf_counter()-started,
            actual_work=read_work(base/'WORK.jsonl',ARM_LIMITS),cleanup_errors=errors)
        if errors:
            progress['status']='failed'
        write_json(base/'STATUS.json',progress)
        if errors:
            raise RuntimeError('Owned worker cleanup failed: '+repr(errors))
    return progress
