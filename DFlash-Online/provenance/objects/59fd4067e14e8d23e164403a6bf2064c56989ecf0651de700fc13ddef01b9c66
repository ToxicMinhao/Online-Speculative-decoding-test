"""Resume O/K for three shared-feedback events; evaluate after events two and four."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import signal
import time
import traceback

from online_feedback.storage import write_json
from experiments.osd_components.protocol import publication_errors, weight_check_errors
from experiments.ok_continuation.evaluate import evaluate
from experiments.ok_continuation.resume import load_snapshot, verify_live_resume, validate_event_receipt

OBJECTIVES = {'O': 'osd', 'K': 'osd_first_error'}
GIB = 2**30
ROW_BYTES = 4096 * 6 * 2


def identity(path):
    path = Path(path)
    with path.open('rb') as stream:
        value = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(bytes=path.stat().st_size, sha256=value)


def validate_plan(plan):
    if plan['new_budget'] != dict(feedback=384, evaluation=768, generations=1152,
                                  presentations=1536, optimizer_steps=768):
        raise ValueError('Continuation budget changed')
    if [w['event_index'] for w in plan['windows']] != [2, 3, 4]:
        raise ValueError('Only events two through four are authorized')
    if len(plan['eval_requests']) != 64 or set(plan['initial']) != set(OBJECTIVES):
        raise ValueError('Require original development panel and own initial O/K states')
    ids, prompts = set(), set()
    for window in plan['windows']:
        if len(window['requests']) != 128:
            raise ValueError('Each feedback window needs128 requests')
        for r in window['requests']:
            if r['request_id'] in ids or tuple(r['prompt_ids']) in prompts:
                raise ValueError('Repeated feedback input')
            ids.add(r['request_id']); prompts.add(tuple(r['prompt_ids']))
            if r['max_new_tokens'] != 2048 or len(r['prompt_ids']) + 2048 > 2272:
                raise ValueError('Keep whole training sequences within2272')
    if plan['stop_step'] != 512 or plan['inherited_horizon'] != 7424:
        raise ValueError('Stop orchestration at512; retain7424 learning horizon')


def admit_phase(root, phase, *, rows=0, new_states=0, exports=0, overlap_states=0,
                compact_present=0):
    """Admit the next phase's worst-case writes, including atomic/failure overlap."""
    weights = 2097259104
    tensors = rows * (ROW_BYTES + 12) + (3 * (new_states + overlap_states) + exports) * weights
    required = int(tensors * 1.10) + 2 * GIB + 2 * GIB + 8 * GIB
    receipt = dict(phase=phase, rows=rows, new_states=new_states, exports=exports,
                   overlap_states=overlap_states, tensor_bytes=tensors,
                   tensor_margin=0.10, compact_allowance_bytes=2*GIB, work_bytes=2*GIB,
                   reserve_bytes=8*GIB, required_bytes=required,
                   free_bytes=shutil.disk_usage(root).free,
                   scope='Only this next phase is admitted; later phases recheck actual occupancy.')
    receipt['admitted'] = receipt['free_bytes'] >= required
    write_json(root / (phase + '_STORAGE.json'), receipt)
    if not receipt['admitted']:
        raise RuntimeError('Insufficient peak storage for phase: ' + phase)
    return receipt


def publish_checked(backend, checkpoint, destination):
    started = time.perf_counter()
    activation = backend.publish(checkpoint)
    errors = publication_errors(activation, checkpoint)
    checked = time.perf_counter()
    equality = backend.check_weights(checkpoint)
    errors += weight_check_errors(equality, checkpoint)
    result = dict(activation=activation, equality=equality, errors=errors,
                  weight_validation_s=time.perf_counter()-checked,
                  publication_and_validation_s=time.perf_counter()-started)
    write_json(destination, result)
    if errors:
        raise RuntimeError('Native checkpoint publication failed: ' + str(errors))
    return result


def collect_pool(backend, window, checkpoint, output, progress, save_progress):
    """Capture one O trajectory; prepare immutable feedback only after collection."""
    output.mkdir()
    (output / 'responses').mkdir()
    (output / 'features').mkdir()
    rows = sum(len(r['prompt_ids'])+r['max_new_tokens'] for r in window['requests'])
    admit_phase(output, 'collection', rows=rows)
    costs = {'publication': publish_checked(backend, checkpoint, output/'PUBLICATION.json')}
    records = []
    started = time.perf_counter()
    for request in window['requests']:
        progress['feedback_attempted'] += 1
        progress['active_feedback'] = request['request_id']; save_progress()
        record = backend.generate(request['prompt_ids'], max_new_tokens=2048,
            request_id=request['request_id'], session_id=window['event_id'],
            request_index=request['request_index'], purpose='serving',
            save_to=output/'responses'/(request['request_id']+'.json'))
        progress['feedback_returned'] += 1
        progress['active_feedback'] = None; save_progress()
        if record['checkpoint_id'] != checkpoint or record['prompt_ids'] != request['prompt_ids']:
            raise ValueError('Feedback checkpoint or exact input mismatch')
        if len(record['prompt_ids'])+len(record['response_ids']) > 2272:
            raise ValueError('Feedback cannot be silently truncated for training')
        records.append(record)
    costs['collection_wall_s'] = time.perf_counter()-started
    costs['request_latency_s'] = sum(r['metadata']['request_latency_s'] for r in records)
    costs['capture_nested'] = {k:sum(r['metadata']['capture'][k] for r in records)
        for k in ('capture_host_s','feature_persistence_s','cache_maintenance_s')}
    # Native captures are already on disk. Budget prepared copies plus both arms,
    # an atomic old/new state overlap and one bounded failure snapshot.
    actual_rows = sum(len(r['prompt_ids'])+len(r['response_ids']) for r in records)
    first = window['event_index'] == 2
    admit_phase(output, 'preparation_training', rows=actual_rows, new_states=2 if first else 0,
                exports=2, overlap_states=1 if first else 2)
    started = time.perf_counter()
    preparations = []
    for record in records:
        result = backend.prepare_feedback(record, output/'features'/(record['record_id']+'.pt'))
        preparations.append(result)
        write_json(output/'PREPARATIONS.json', preparations)
        if result['status'] != 'prepared':
            raise ValueError('Feedback ineligible; no substitution/resampling is authorized')
    costs['preparation_wall_s'] = time.perf_counter()-started
    costs['teacher_preparation_s'] = sum(p['teacher_preparation_s'] for p in preparations)
    costs['feature_persistence_s'] = sum(p['feature_persistence_s'] for p in preparations)
    paths = [p['feature_path'] for p in preparations]
    pool = dict(event_id=window['event_id'], checkpoint=checkpoint,
                record_ids=[r['record_id'] for r in records], features=paths,
                identities={p:identity(p) for p in paths}, immutable=True,
                teacher_terminal_forwards=len(preparations), response_generations=len(records))
    write_json(output/'POOL.json', pool); write_json(output/'COSTS.json', costs)
    return pool, costs


def train_event(plan, arm, event_index, parent, pool, root, ordinary=None):
    """Restore this arm's complete state; add exactly128 steps; no warmup."""
    import torch
    from online_feedback.training import SpecForgeEventTrainer

    output = root/arm
    audit_root = root/f'resume_{event_index:03d}_{arm}'; audit_root.mkdir()
    started = time.perf_counter()
    if torch.cuda.is_initialized():
        torch.cuda.reset_peak_memory_stats()
    state, receipt = load_snapshot(parent['state'], parent['export'], objective=OBJECTIVES[arm],
        config_text=Path(plan['config']).read_text(), completed_event=event_index-1)
    receipt['state_identity'] = identity(parent['state'])
    write_json(audit_root/'SAVED.json', receipt)
    trainer = None
    try:
        trainer = SpecForgeEventTrainer(plan['config'], output, resume_state=parent['state'],
                                        objective=OBJECTIVES[arm], reduction='response_mean', ranking_lambda=0)
        live = verify_live_resume(trainer, state, receipt_path=audit_root/'LIVE.json')
        del state
        loading_s = time.perf_counter()-started
        loading_peak_allocated_bytes = torch.cuda.max_memory_allocated()
        loading_peak_reserved_bytes = torch.cuda.max_memory_reserved()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        result = trainer.train_event(f'event_{event_index:03d}', pool['features'], passes=2,
                                     presentation_schedule=ordinary)
        event_wall_s = time.perf_counter()-started
        check = validate_event_receipt(result, event_number=event_index, objective=OBJECTIVES[arm],
                                       reference_schedule=ordinary, trainer_status=trainer.status())
        write_json(audit_root/'EVENT_CHECK.json', check)
        cost = dict(loading_and_resume_validation_s=loading_s, event_wall_s=event_wall_s,
                    **result['costs'], objective_work=result['objective_work'],
                    loading_peak_allocated_bytes=loading_peak_allocated_bytes,
                    loading_peak_reserved_bytes=loading_peak_reserved_bytes,
                    peak_scope='Training-only peak after loading; loading peaks are separate.',
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved())
        write_json(audit_root/'COSTS.json', cost)
        return result, cost
    except BaseException as error:
        failure = dict(error=repr(error), automatic_retry=False)
        if trainer is not None:
            failure['status'] = trainer.status()
        write_json(audit_root/'FAILED.json', failure)
        raise
    finally:
        if trainer is not None:
            original_failure = (audit_root/'FAILED.json').exists()
            try:
                trainer.close()
            except BaseException as error:
                write_json(audit_root/'CLEANUP_ERROR.json', dict(error=repr(error)))
                if not original_failure:
                    raise
            finally:
                write_json(audit_root/'CLOSED.json', trainer.status())


def verify_pool(pool):
    for path, expected in pool['identities'].items():
        if identity(path) != expected:
            raise ValueError('Shared feedback pool changed: ' + path)


def reconcile_work(output, progress):
    """Keep partial work visible without counting completed events twice."""
    latest = dict(O=128, K=128)
    presentations = dict(O=256, K=256)
    for n in (2, 3, 4):
        for arm in OBJECTIVES:
            folder = output/f'resume_{n:03d}_{arm}'
            closed, failed = folder/'CLOSED.json', folder/'FAILED.json'
            state = json.loads(closed.read_text()) if closed.exists() else None
            if state is None and failed.exists():
                state = json.loads(failed.read_text()).get('status')
            if state is not None:
                latest[arm] = state['global_step']; presentations[arm] = state['presentations']
    progress['arm_counters'] = latest
    progress['optimizer_steps'] = sum(latest.values())-256
    progress['presentations'] = sum(presentations.values())-512
    evaluations={}
    for n in (2,4):
        path=output/f'evaluation_{n:03d}/STATUS.json'
        if path.exists(): evaluations[str(n)]=json.loads(path.read_text())
    progress['evaluations']=evaluations
    progress['evaluation_returned']=sum(s['returned_generations'] for s in evaluations.values())
    progress['evaluation_attempted']=sum(s['attempted_generations'] for s in evaluations.values())
    progress['evaluation_unknown']=sum(s.get('unknown_generation_attempts',0) for s in evaluations.values())
    progress['feedback_unknown']=int(progress.get('active_feedback') is not None)
    progress['evaluation_unattempted']=768-progress['evaluation_attempted']
    progress['feedback_unattempted']=384-progress['feedback_attempted']
    progress['training_interrupted']=progress['status']=='failed' and progress['phase'].startswith('training_')
    progress['training_counter_scope']='Completed optimizer calls reported by trainer; an interrupted in-flight call is not assumed absent.'


def run(root):
    from online_feedback.sglang_backend import SGLangDFlashBackend
    from online_feedback.qualify_sglang import check_recipe

    root=Path(root); plan=json.loads((root/'sealed/PLAN.json').read_text()); validate_plan(plan)
    check_recipe(plan['config'])
    output=root/'campaign_001'; output.mkdir(exist_ok=False)
    started=time.perf_counter(); backend=None; costs={}
    progress=dict(status='running',phase='loading_native',feedback_attempted=0,feedback_returned=0,
                  optimizer_steps=0,presentations=0,evaluation_returned=0,
                  active_feedback=None,arm_counters={'O':128,'K':128},events_completed=[],
                  automatic_retry=False,extra_optimizer_steps=0,extra_generations=0)
    def save(): write_json(output/'STATUS.json',progress)
    save(); parents=plan['initial'].copy()
    try:
        backend=SGLangDFlashBackend(plan['target'],plan['draft'],worker_python=plan['worker_python'],
                    output_root=output/'native',serving_gpu='0',context_limit=4096,block_size=16)
        costs['backend_loading_s']=backend.loading_s
        for window in plan['windows']:
            n=window['event_index']; progress['event']=n; progress['phase']='collecting_feedback'; save()
            pool,pool_cost=collect_pool(backend,window,parents['O']['export'],output/window['event_id'],progress,save)
            costs[f'event_{n:03d}_feedback']=pool_cost
            ordinary=None
            for arm in OBJECTIVES:
                progress['phase']='training_'+arm; save(); verify_pool(pool)
                result,cost=train_event(plan,arm,n,parents[arm],pool,output,ordinary=ordinary)
                if arm=='O': ordinary=result
                verify_pool(pool)
                parents[arm]=dict(state=result['full_state_path'],export=result['export_path'])
                progress['optimizer_steps']+=128; progress['presentations']+=256
                progress['arm_counters'][arm]=result['end_step']; save()
                costs[f'event_{n:03d}_{arm}']=cost
                write_json(output/'COSTS.json',costs)
            progress['events_completed'].append(n); save()
            if n in (2,4):
                progress['phase']='evaluation'; save()
                status=evaluate(backend,plan['eval_requests'],{a:parents[a]['export'] for a in OBJECTIVES},
                                output/f'evaluation_{n:03d}',frozen_counts=plan['frozen_counts'])
                progress['evaluation_returned']+=status['returned_generations']; save()
        if (progress['optimizer_steps'],progress['presentations'],progress['feedback_returned'],
            progress['evaluation_returned'])!=(768,1536,384,768):
            raise ValueError('Completed work differs from the declared budget')
        progress.update(status='completed',phase='finished')
    except BaseException as error:
        progress.update(status='failed',error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        cleanup_error=None
        if backend is not None:
            try: backend.close()
            except BaseException as error:
                progress['cleanup_error']=repr(error)
                if progress['status']!='failed':
                    progress.update(status='failed',error=repr(error)); cleanup_error=error
        reconcile_work(output,progress)
        costs['campaign_wall_s']=time.perf_counter()-started
        costs['accounting']='Nested capture/preparation/training/persistence timers are not added twice. Evaluation repeats and switches are benchmark overhead. Shared-feedback objectives, not independent deployment costs.'
        write_json(output/'COSTS.json',costs); save()
        if cleanup_error: raise cleanup_error


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('root'); args=parser.parse_args()
    def interrupted(signum,frame): raise RuntimeError(f'Interrupted by signal{signum}')
    signal.signal(signal.SIGTERM,interrupted)
    run(args.root)
