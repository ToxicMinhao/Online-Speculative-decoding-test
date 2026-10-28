"""Resume KRC1 exactly, remove event-2 correction, then compare OFF with retained KRC2."""
import json
import math
from pathlib import Path
import signal
import sys
import time
import traceback

from online_feedback.storage import write_json
from .evaluation_protocol import validate_plan


def diagnostic_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: diagnostic_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [diagnostic_json(item) for item in value]
    return value


def unattempted_panel():
    return dict(status='unattempted', planned_generations=384, attempted_generations=0,
                returned_generations=0, completed_valid_generations=0, invalid_generations=0,
                unknown_generation_attempts=0, unattempted_generations=384)


def evaluation_totals(panels):
    keys = ('planned_generations', 'attempted_generations', 'returned_generations',
            'completed_valid_generations', 'invalid_generations',
            'unknown_generation_attempts', 'unattempted_generations')
    return {key: sum(panel.get(key, 0) for panel in panels.values()) for key in keys}


def evaluate_panels(backend, panels, checkpoints, root, progress):
    """Only development's first eight requests form the included native gate."""
    from .evaluate import evaluate
    root = Path(root)
    for index, panel in enumerate(panels):
        name = panel['name']
        destination = root / 'evaluation' / name
        try:
            progress['panels'][name] = evaluate(
                backend, panel['requests'], checkpoints, destination,
                expected_control_counts=panel['expected_control_counts'],
                historical_counts=panel['historical_counts'],
                qualification_requests=8 if index == 0 else 0)
        finally:
            status_path = destination / 'STATUS.json'
            if status_path.exists():
                progress['panels'][name] = json.loads(status_path.read_text())
            progress['evaluation_totals'] = evaluation_totals(progress['panels'])
            write_json(root / 'STATUS.json', progress)



def run(root):
    import torch
    from online_feedback.krc_off import KRCOffTrainer
    from online_feedback.sglang_backend import SGLangDFlashBackend
    from experiments.ok_continuation.resume import load_snapshot, verify_live_resume
    from experiments.prefix_guard.qualify import (
        WorkLog, pin_retained_graph_runtime, require_runtime_pin, run_startup)
    from .protocol import validate_retained, validate_schedule, check_event, check_off_report

    root=Path(root)
    plan=json.loads((root/'sealed/PLAN.json').read_text())
    plan_check=validate_plan(plan)
    krc1=json.loads(Path(plan['krc1_receipt']).read_text())
    ordinary2=json.loads(Path(plan['event2_ordinary_receipt']).read_text())
    k2=json.loads(Path(plan['event2_k_receipt']).read_text())
    krc2=json.loads(Path(plan['krc2_receipt']).read_text())
    retained_check=validate_retained(krc1,ordinary2,k2,krc2)
    if krc2['checkpoint_path'] != plan['retained_krc2_checkpoint']:
        raise ValueError('KRC2 evaluator checkpoint differs from retained event-2 receipt')
    schedule=json.loads(Path(plan['presentation_schedule']).read_text())
    schedule_check=validate_schedule(schedule,ordinary2,
        evaluation_ids=[r['request_id'] for panel in plan['panels'] for r in panel['requests']])
    output=root/'campaign_001'; output.mkdir(exist_ok=False)
    started=time.perf_counter()
    progress=dict(status='running',phase='resume_preflight',global_optimizer_step=128,
        new_optimizer_steps=0,new_presentations=0,event_local_windows_completed=0,
        new_feedback_generations=0,new_qualification_optimizer_steps=0,
        new_qualification_presentations=0,automatic_retry=False,
        panels={panel['name']:unattempted_panel() for panel in plan['panels']})
    progress['evaluation_totals']=evaluation_totals(progress['panels'])
    costs=dict(new_qualification_optimizer_steps=0,new_qualification_presentations=0)
    trainer=backend=startup_log=None
    failure=None
    write_json(output/'STATUS.json',progress)
    write_json(output/'PLAN_CHECK.json',plan_check)
    write_json(output/'RETAINED_CHECK.json',retained_check)
    write_json(output/'PRESENTATION_SCHEDULE_CHECK.json',schedule_check)
    try:
        costs['runtime_compatibility']=pin_retained_graph_runtime(output)
        state,snapshot_check=load_snapshot(plan['krc1_state'],krc1['checkpoint_path'],
            objective='osd_first_error',config_text=Path(plan['config']).read_text(),completed_event=1)
        write_json(output/'RESUME_SNAPSHOT.json',snapshot_check)
        costs['snapshot_validation_s']=snapshot_check['wall_s']
        progress['phase']='startup'; write_json(output/'STATUS.json',progress)
        startup_log=WorkLog(output/'startup'); startup_log.trial='continuation_process_startup'
        startup_ordinary=json.loads(Path(plan['startup_ordinary_receipt']).read_text())
        pinned=torch.load(plan['pinned_initial'],map_location='cpu',weights_only=False,mmap=True)
        require_runtime_pin()
        costs['startup']=run_startup(plan['config'],startup_ordinary,pinned,startup_log,output/'startup')
        del pinned
        expected=dict(status='passed',response_evaluations=2,backwards=2,optimizer_steps=0,
                      exact_initial_state_restored=True,separate_disposable_runtime=True)
        if any(costs['startup'][key]!=value for key,value in expected.items()):
            raise ValueError('Startup requires two disposable forward/backwards and zero Adam steps')
        costs['startup_work_counts']=dict(startup_log.counts)
        progress['phase']='resuming_KRC1'; write_json(output/'STATUS.json',progress)
        torch.cuda.reset_peak_memory_stats()
        loading_started=time.perf_counter()
        require_runtime_pin()
        trainer=KRCOffTrainer(plan['config'],output/'KRC2_OFF',resume_state=plan['krc1_state'],
            objective='osd_first_error',reduction='response_mean')
        resume=verify_live_resume(trainer,state,receipt_path=output/'LIVE_RESUME.json')
        if not resume['exact_resume'] or trainer.global_step!=128 or trainer.presentations!=256:
            raise ValueError('Continuation must resume exact KRC1 state at step 128')
        del state
        costs['loading_and_live_resume_s']=time.perf_counter()-loading_started
        require_runtime_pin()
        trainer.enable_off_continuation(resume_check=resume, first_A_records=krc2['records'][:2],
                                        output_dir=output/'off')
        progress['phase']='training_KRC2_OFF'; write_json(output/'STATUS.json',progress)
        training_started=time.perf_counter()
        receipt=trainer.train_event('event_002',[Path(row['path']) for row in ordinary2['features']],
                                    passes=2,presentation_schedule=ordinary2)
        costs['event_wall_s']=time.perf_counter()-training_started
        write_json(output/'TRAINING_CHECK.json',check_event(receipt,ordinary2,trainer.status()))
        report=trainer.off_report()
        write_json(output/'OFF_CHECK.json',check_off_report(report))
        write_json(output/'OFF_EVIDENCE.json',report)
        costs['off']=report
        costs['event']=receipt['costs']; costs['objective_work']=receipt['objective_work']
        costs['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
        costs['peak_reserved_bytes']=torch.cuda.max_memory_reserved()
        progress.update(global_optimizer_step=trainer.global_step,new_optimizer_steps=trainer.global_step-128,
                        new_presentations=trainer.presentations-256,event_local_windows_completed=128)
        checkpoint=receipt['checkpoint_path']
        require_runtime_pin()
        trainer.close(); write_json(output/'TRAINER_CLOSED.json',trainer.status())
        trainer=None
        import gc
        gc.collect(); torch.cuda.empty_cache()
        progress['phase']='loading_native'; write_json(output/'STATUS.json',progress)
        backend=SGLangDFlashBackend(plan['target'],plan['draft'],worker_python=plan['worker_python'],
            output_root=output/'native',serving_gpu='0',context_limit=4096,block_size=16)
        costs['backend_loading_s']=backend.loading_s
        progress['phase']='evaluation'; write_json(output/'STATUS.json',progress)
        evaluate_panels(backend,plan['panels'],
            {'KRC2':plan['retained_krc2_checkpoint'],'KRC2_OFF':checkpoint},output,progress)
        progress.update(status='completed',phase='finished')
    except BaseException as error:
        failure=error
        progress.update(status='failed',error=repr(error),traceback=traceback.format_exc())
        if trainer is not None:
            progress['trainer_at_failure']=trainer.status()
            progress.update(global_optimizer_step=trainer.global_step,new_optimizer_steps=trainer.global_step-128,
                            new_presentations=trainer.presentations-256)
            trainer.failed=True
        raise
    finally:
        def cleanup(action,label):
            try: action()
            except BaseException as error:
                progress.setdefault('cleanup_errors',[]).append(dict(operation=label,error=repr(error)))
                if failure is not None: failure.add_note(label+': '+repr(error))
                else: progress.update(status='failed',phase='cleanup',error=repr(error))
        if startup_log is not None:
            costs['startup_work_counts']=dict(startup_log.counts)
        if trainer is not None:
            costs['off']=trainer.off_report()
            write_json(output/'OFF_EVIDENCE.json',diagnostic_json(costs['off']))
            cleanup(trainer.close,'close_trainer')
            write_json(output/'TRAINER_CLOSED.json',trainer.status())
        if 'off' in costs:
            work=costs['off']['work']
            progress.update(actual_new_Adam_steps_completed=work['Adam_steps'],
                            actual_global_Adam_step=128+work['Adam_steps'],
                            actual_K_presentations_completed=work['presentations'])
        if backend is not None: cleanup(backend.close,'close_backend')
        costs['actual_campaign_wall_s']=time.perf_counter()-started
        costs['accounting']=(
            'KRC2_OFF resumes exact KRC1 weights, Adam and own RNG before disabling correction. '
            'The ordinary K path and natural sampling remain unchanged. '
            'Separate disposable startup uses two forward/backwards and zero Adam. '
            'Main work is 128 new Adam steps and 256 ordinary response forwards/backwards; '
            'there are no reference forwards, gradients, prefix captures, corrections or probes. '
            'Feedback/features are reused; retained preparation costs are historical. '
            'Serving starts only after training closes. Native/cache timers are nested in request time. '
            'Historical KRC2 training costs are not a fresh matched control; historical serving times '
            'are not a latency learning curve.')
        progress['evaluation_totals']=evaluation_totals(progress['panels'])
        write_json(output/'COSTS.json',diagnostic_json(costs)); write_json(output/'STATUS.json',progress)
        if failure is None and progress.get('cleanup_errors'):
            raise RuntimeError('Continuation cleanup failed; see STATUS.json')


if __name__=='__main__':
    def interrupted(signum,frame):
        raise InterruptedError('Supervisor signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted)
    run(sys.argv[1])
