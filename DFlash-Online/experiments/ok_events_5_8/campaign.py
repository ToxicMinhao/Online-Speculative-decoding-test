"""Bounded ordinary O/K continuation: own event4 -> shared feedback events5..8.

Reuse the qualified feedback, training and serving APIs. New orchestration only;
no correction, diagnostic forwards, warmup Adam steps or automatic next event.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import traceback

from online_feedback.storage import write_json
from experiments.osd_components.protocol import publication_errors, weight_check_errors
from .resume import (OBJECTIVES, OKContinuationTrainer, load_snapshot, verify_live_resume,
                     validate_event_receipt, verify_saved_boundary)

NEW_BUDGET = dict(feedback=512, evaluation=1920, generations=2432,
                  presentations=2048, optimizer_steps=1024)


def validate_plan(plan):
    from .evaluation_protocol import validate_panels
    if plan.get('new_budget') != NEW_BUDGET:
        raise ValueError('Continuation work budget differs from the authorized scope')
    if [row['event_index'] for row in plan['windows']] != [5, 6, 7, 8]:
        raise ValueError('Only ordered events 5 through 8 are authorized')
    if set(plan['initial']) != set(OBJECTIVES):
        raise ValueError('Require own O4 and K4 complete states')
    for arm, source in plan['initial'].items():
        if not all(source.get(key) for key in ('state', 'export', 'receipt')):
            raise ValueError('Missing own state/export/receipt binding: ' + arm)
    if (plan.get('stop_step') != 1024 or plan.get('inherited_horizon') != 7424
            or plan.get('no_automatic_retry') is not True):
        raise ValueError('Stop at 1024 own steps without retry; retain the 7424-step horizon')
    panel_check = validate_panels(plan['panels'])
    evaluation_ids = {row['request_id'] for panel in plan['panels'] for row in panel['requests']}
    evaluation_prompts = {tuple(row['prompt_ids']) for panel in plan['panels'] for row in panel['requests']}
    ids, prompts = set(), set()
    for window in plan['windows']:
        if window['event_id'] != f"event_{window['event_index']:03d}" or len(window['requests']) != 128:
            raise ValueError('Each fixed event requires its next 128 eligible training requests')
        for row in window['requests']:
            rid, prompt = row['request_id'], tuple(row['prompt_ids'])
            if rid in ids | evaluation_ids or prompt in prompts | evaluation_prompts:
                raise ValueError('Training inputs duplicate or overlap an evaluation identity/content')
            if (row['max_new_tokens'] != 2048 or not prompt or len(prompt) + 2048 > 2272
                    or any(type(token) is not int or token < 0 for token in prompt)):
                raise ValueError('Preserve complete training sequences within the qualified 2272 limit')
            ids.add(rid)
            prompts.add(prompt)
    return dict(status='passed', windows=[5, 6, 7, 8], new_budget=NEW_BUDGET,
                own_start_steps=512, own_end_steps=1024, panels=panel_check)


def identity(path):
    path = Path(path)
    with path.open('rb') as stream:
        value = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(bytes=path.stat().st_size, sha256=value)


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


def collect_pool(backend, window, checkpoint, output, progress, save_progress, storage):
    """Capture one O trajectory; prepare immutable feedback only after collection."""
    output.mkdir()
    (output / 'responses').mkdir()
    (output / 'features').mkdir()
    storage.admit_collection(window)
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
        if (record['checkpoint_id'] != checkpoint or record['prompt_ids'] != request['prompt_ids']
                or record['record_id'] != request['request_id']):
            raise ValueError('Feedback checkpoint, request identity or exact input mismatch')
        if len(record['prompt_ids'])+len(record['response_ids']) > 2272:
            raise ValueError('Feedback cannot be silently truncated for training')
        records.append(record)
    costs['collection_wall_s'] = time.perf_counter()-started
    costs['request_latency_s'] = sum(r['metadata']['request_latency_s'] for r in records)
    costs['capture_nested'] = {k:sum(r['metadata']['capture'][k] for r in records)
        for k in ('capture_host_s','feature_persistence_s','cache_maintenance_s')}
    # Prepared feature copies are admitted separately from captured native
    # feature tensors; no pool or feedback response is silently shortened.
    storage.admit_preparation(window, records)
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


def verify_pool(pool):
    if (len(pool['features']) != 128 or len(set(pool['features'])) != 128
            or len(pool['record_ids']) != 128 or len(set(pool['record_ids'])) != 128
            or pool['response_generations'] != 128 or pool['immutable'] is not True):
        raise ValueError('Require the complete immutable shared 128-response feedback pool')
    if set(pool['features']) != set(pool['identities']):
        raise ValueError('Feature identities do not cover the whole shared pool')
    for path, expected in pool['identities'].items():
        if identity(path) != expected:
            raise ValueError('Shared feedback pool changed: ' + path)


def train_event(plan, arm, event_index, parent, pool, root, ordinary=None):
    """Exactly restore this arm, add its next128 steps and close before serving."""
    import torch
    output = root / arm
    audit_root = root / f'resume_{event_index:03d}_{arm}'
    audit_root.mkdir()
    started = time.perf_counter()
    if torch.cuda.is_initialized():
        torch.cuda.reset_peak_memory_stats()
    state, receipt = load_snapshot(parent['state'], parent['export'], objective=OBJECTIVES[arm],
        config_text=Path(plan['config']).read_text(), completed_event=event_index - 1)
    receipt['state_identity'] = identity(parent['state'])
    write_json(audit_root / 'SAVED.json', receipt)
    # The initial parent must not already contain any newly selected request.
    used = {feature['record_id'] for event in state['events'] for feature in event['features']}
    if used & set(pool['record_ids']):
        raise ValueError('New feedback reuses a request from this own arm history')
    trainer = None
    try:
        trainer = OKContinuationTrainer(plan['config'], output, resume_state=parent['state'],
                                         arm=arm, completed_event=event_index - 1)
        verify_live_resume(trainer, state, receipt_path=audit_root / 'LIVE.json')
        del state
        loading_s = time.perf_counter() - started
        loading_peak_allocated = torch.cuda.max_memory_allocated()
        loading_peak_reserved = torch.cuda.max_memory_reserved()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        result = trainer.train_event(f'event_{event_index:03d}', pool['features'], passes=2,
                                     presentation_schedule=ordinary)
        event_wall_s = time.perf_counter() - started
        check = validate_event_receipt(result, event_number=event_index, objective=OBJECTIVES[arm],
            reference_schedule=ordinary, trainer_status=trainer.status())
        write_json(audit_root / 'EVENT_CHECK.json', check)
        # Latest is replaced atomically. Hard links at6/8 preserve complete
        # selected endpoints without a second tensor allocation.
        if event_index in (6, 8):
            retained = trainer.events_root / f'event_{event_index:03d}' / 'training_state.pt'
            os.link(trainer.latest, retained)
            result['latest_state_path'] = str(trainer.latest)
            result['full_state_path'] = str(retained)
            result['state_retention'] = 'immutable hard link; latest is atomically replaced'
            write_json(retained.parent / 'RESULT.json', result)
        saved, saved_check = load_snapshot(result['full_state_path'], result['export_path'],
            objective=OBJECTIVES[arm], config_text=Path(plan['config']).read_text(), completed_event=event_index)
        write_json(audit_root / 'SAVED_ENDPOINT.json', saved_check)
        verify_saved_boundary(trainer, saved, audit_root / 'LIVE_BOUNDARY.json')
        del saved
        retained_identities = {result['full_state_path']: identity(result['full_state_path'])}
        retained_identities.update({str(path): identity(path) for path in sorted(Path(result['export_path']).iterdir())
                                   if path.is_file()})
        write_json(audit_root / 'STATE_IDENTITIES.json', retained_identities)
        cost = dict(loading_and_resume_validation_s=loading_s, event_wall_s=event_wall_s,
            **result['costs'], objective_work=result['objective_work'],
            loading_peak_allocated_bytes=loading_peak_allocated,
            loading_peak_reserved_bytes=loading_peak_reserved,
            peak_scope='Training-only peak after loading; loading peaks separate; persistence checks add no model passes.',
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            saved_state_validation_s=saved_check.get('wall_s'))
        write_json(audit_root / 'COSTS.json', cost)
        return result, cost
    except BaseException as error:
        failure = dict(error=repr(error), automatic_retry=False)
        if trainer is not None:
            failure['status'] = trainer.status()
        write_json(audit_root / 'FAILED.json', failure)
        raise
    finally:
        if trainer is not None:
            original_failure = (audit_root / 'FAILED.json').exists()
            try:
                trainer.close()
            except BaseException as error:
                write_json(audit_root / 'CLEANUP_ERROR.json', dict(error=repr(error)))
                if not original_failure:
                    raise
            finally:
                write_json(audit_root / 'CLOSED.json', trainer.status())
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()


def reconcile_work(output, progress):
    latest, presentations = dict(O=512, K=512), dict(O=1024, K=1024)
    for number in (5, 6, 7, 8):
        for arm in OBJECTIVES:
            folder = output / f'resume_{number:03d}_{arm}'
            closed, failed = folder / 'CLOSED.json', folder / 'FAILED.json'
            state = json.loads(closed.read_text()) if closed.exists() else None
            if state is None and failed.exists():
                state = json.loads(failed.read_text()).get('status')
            if state is not None:
                latest[arm], presentations[arm] = state['global_step'], state['presentations']
    progress['arm_counters'] = latest
    progress['arm_presentations'] = presentations
    progress['optimizer_steps'] = sum(latest.values()) - 1024
    progress['presentations'] = sum(presentations.values()) - 2048
    evaluations = {}
    for number in (6, 8):
        for panel in ('development', 'additional', 'fresh') if number == 8 else ('development', 'additional'):
            path = output / f'event_{number:03d}' / 'evaluation' / panel / 'STATUS.json'
            if path.exists():
                evaluations[f'{number}/{panel}'] = json.loads(path.read_text())
    progress['evaluations'] = evaluations
    for suffix, key in (('returned', 'returned_generations'), ('attempted', 'attempted_generations'),
                        ('unknown', 'unknown_generation_attempts'), ('valid', 'completed_valid_generations'),
                        ('invalid', 'invalid_generations'), ('pending', 'pending_canonical_generations')):
        progress['evaluation_' + suffix] = sum(row.get(key, 0) for row in evaluations.values())
    progress['feedback_unknown'] = int(progress.get('active_feedback') is not None)
    progress['evaluation_unattempted'] = 1920 - progress['evaluation_attempted']
    progress['feedback_unattempted'] = 512 - progress['feedback_attempted']
    progress['training_interrupted'] = progress['status'] == 'failed' and progress['phase'].startswith('training_')
    progress['training_counter_scope'] = 'Completed optimizer calls; an interrupted in-flight call is not assumed absent.'


def run(root):
    from online_feedback.sglang_backend import SGLangDFlashBackend
    from online_feedback.qualify_sglang import check_recipe
    from .evaluate import evaluate
    from .storage import StorageLifecycle
    root = Path(root)
    plan = json.loads((root / 'sealed/PLAN.json').read_text())
    plan_check = validate_plan(plan)
    check_recipe(plan['config'])
    output = root / 'campaign_001'
    output.mkdir(exist_ok=False)
    started = time.perf_counter()
    backend = None
    costs = {'execution_source': plan.get('execution_source')}
    progress = dict(status='running', phase='loading_native', feedback_attempted=0, feedback_returned=0,
        optimizer_steps=0, presentations=0, evaluation_returned=0, active_feedback=None,
        arm_counters={'O': 512, 'K': 512}, events_completed=[], automatic_retry=False,
        extra_optimizer_steps=0, extra_generations=0)
    def save():
        write_json(output / 'STATUS.json', progress)
    save()
    write_json(output / 'PLAN_CHECK.json', plan_check)
    parents = {arm: dict(source) for arm, source in plan['initial'].items()}
    try:
        storage = StorageLifecycle(plan, output)
        backend = SGLangDFlashBackend(plan['target'], plan['draft'], worker_python=plan['worker_python'],
            output_root=output / 'native', serving_gpu='0', context_limit=4096, block_size=16)
        costs['backend_loading_s'] = backend.loading_s
        for window in plan['windows']:
            number = window['event_index']
            progress.update(event=number, phase='collecting_feedback')
            save()
            pool, pool_cost = collect_pool(backend, window, parents['O']['export'],
                output / window['event_id'], progress, save, storage)
            costs[f'event_{number:03d}_feedback'] = pool_cost
            ordinary = None
            for arm in OBJECTIVES:
                progress['phase'] = 'training_' + arm
                save()
                verify_pool(pool)
                storage.admit_training(arm, number, pool, parents)
                result, cost = train_event(plan, arm, number, parents[arm], pool, output, ordinary=ordinary)
                if arm == 'O':
                    ordinary = result
                verify_pool(pool)
                parents[arm] = dict(state=result['full_state_path'], export=result['export_path'],
                    receipt=str(output / arm / 'events' / f'event_{number:03d}' / 'RESULT.json'))
                progress['optimizer_steps'] += 128
                progress['presentations'] += 256
                progress['arm_counters'][arm] = result['end_step']
                save()
                costs[f'event_{number:03d}_{arm}'] = cost
                write_json(output / 'COSTS.json', costs)
            progress['events_completed'].append(number)
            save()
            if number in (6, 8):
                for panel in plan['panels']:
                    if panel['name'] == 'fresh' and number != 8:
                        continue
                    progress['phase'] = f"evaluation_{number:03d}_{panel['name']}"
                    save()
                    status = evaluate(backend, panel['requests'], {arm: parents[arm]['export'] for arm in OBJECTIVES},
                        output / f'event_{number:03d}' / 'evaluation' / panel['name'], event_number=number,
                        panel_name=panel['name'], canonical_mode=panel['canonical_mode'],
                        qualification_requests=8 if number == 6 and panel['name'] == 'development' else 0)
                    progress['evaluation_returned'] += status['returned_generations']
                    save()
            # Storage/offload can occur only after all training and this event's
            # timed evaluation close; never overlap transfers with measured serving.
            progress['phase'] = f'retention_{number:03d}'
            save()
            storage.after_event(number, parents, pool, evaluated=number in (6, 8))
            write_json(output / 'CURRENT_PARENTS.json', parents)
        storage.finalize(parents)
        reconcile_work(output, progress)
        if (progress['optimizer_steps'], progress['presentations'], progress['feedback_returned'],
                progress['evaluation_returned'], progress['evaluation_valid']) != (1024, 2048, 512, 1920, 1920):
            raise ValueError('Completed work differs from the declared continuation budget')
        if progress['arm_counters'] != {'O': 1024, 'K': 1024}:
            raise ValueError('Both arms must finish their own complete event8 boundary')
        progress.update(status='completed', phase='finished')
    except BaseException as error:
        progress.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        cleanup_error = None
        if backend is not None:
            try:
                backend.close()
            except BaseException as error:
                progress['cleanup_error'] = repr(error)
                if progress['status'] != 'failed':
                    progress.update(status='failed', error=repr(error))
                    cleanup_error = error
        reconcile_work(output, progress)
        costs['campaign_wall_s'] = time.perf_counter() - started
        costs['accounting'] = ('Shared O feedback collected once; 1024 ordinary updates across O/K, no correction/probes. '
            'Capture/preparation/training/persistence timers nest; evaluation repeats and checkpoint switches are benchmark '
            'overhead. Trainers close before timed serving; storage/offload runs between evaluated phases. No reseeding or '
            'scientific warmup step is introduced. Shared-feedback objectives are not independent online deployments.')
        write_json(output / 'COSTS.json', costs)
        save()
        if cleanup_error:
            raise cleanup_error


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root')
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise InterruptedError('Supervisor signal ' + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    run(args.root)
