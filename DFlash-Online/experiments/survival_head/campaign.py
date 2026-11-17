"""One event, three independent complete continuations, one quiet evaluation."""
from pathlib import Path
import time
import traceback

from experiments.kplus_abt.campaign import schedule_bindings
from online_feedback.storage import write_json
from .contract import ARMS, budgets, expected_work
from .data import read, verify_shared_packets, feedback_requests
from .storage import require_capacity


def validate_update(result, arm, baseline):
    expected = dict(status='completed', method=arm, dataset='gsm8k', event_id='event_005',
        start_step=512, end_step=640, optimizer_steps=128, response_presentations=256, passes=2)
    if arm not in ARMS or any(result.get(k) != v for k,v in expected.items()):
        raise ValueError('One-event native accounting differs')
    if schedule_bindings(result) != schedule_bindings(baseline):
        raise ValueError('Original response order or sampled anchors changed')
    if result['survival_work'] != expected_work(arm):
        raise ValueError('Head extension added unplanned native forward/update work')
    if result['head_optimizer_steps'] != (0 if arm == 'KPLUS' else 128):
        raise ValueError('Separate head optimizer accounting differs')
    return dict(status='passed', original_schedule_exact=True)


def require_admission(plan):
    from .preflight import verify_seal, priority_finished
    from .run_qualification import summarize

    current, _ = verify_seal(plan['root'])
    if current != plan:
        raise ValueError('Campaign plan changed since its caller read it')
    priority_finished(plan)
    receipt = read(Path(plan['root'])/'qualification/ADMISSION.json')
    if (plan['budgets'] != budgets() or plan['automatic_retry'] is not False
            or receipt['status'] != 'passed' or receipt['binding'] != plan['binding']
            or receipt['execution_commit'] != plan['execution_source']['commit']):
        raise ValueError('Require same-source bounded qualification before main training')
    if receipt != summarize(plan['root']):
        raise ValueError('Qualification admission no longer matches its retained raw evidence')
    return receipt


def shared_inputs(plan):
    """Bind the prepared pool and path-only schedule change to original inputs."""
    from .prepare_feedback import rebound_schedule

    root = Path(plan['root'])
    pool = read(root/'feedback/POOL.json')
    verify_shared_packets(pool)
    original_pool = read(plan['prior_pool'])
    requests = feedback_requests(plan)
    if [row['record_id'] for row in pool['records']] != [row['request_id'] for row in requests]:
        raise ValueError('Prepared pool differs from the original128 feedback requests')
    records, features = [], []
    for original, current in zip(original_pool['records'], pool['records']):
        if (original['record_id'] != current['record_id'] or any(
                original['canonical'][key] != current['canonical'][key] for key in ('bytes','sha256'))):
            raise ValueError('Prepared canonical feedback differs from the original bytes')
        records.append(read(current['canonical']['path']))
        features.append(current['feature']['path'])
    baseline = read(root/'feedback/PRESENTATION_SCHEDULE.json')
    if baseline != rebound_schedule(read(plan['original_schedule']), records, features):
        raise ValueError('Prepared schedule changed more than feature paths')
    return pool, baseline


def run(root, *, train_arm, evaluate):
    root = Path(root); plan = read(root/'sealed/PLAN.json')
    output = root/'campaign_001'; output.mkdir(exist_ok=False)
    progress = dict(status='running', completed_tasks=[], attempted_task=None,
                    execution_source=plan['execution_source'], automatic_retry=False)
    started = time.perf_counter(); final = {}
    try:
        progress['admission'] = require_admission(plan)
        require_capacity(plan, output/'STORAGE_ADMISSION.json')
        pool, baseline = shared_inputs(plan)
        for arm in ARMS:
            progress['attempted_task'] = dict(kind='train_arm', dataset='gsm8k', event=5, arm=arm)
            write_json(output/'STATUS.json', progress)
            final[arm] = train_arm(plan, arm)
            validate_update(final[arm], arm, baseline)
            if (final[arm].get('features') != baseline['features']
                    or final[arm].get('boundary', {}).get('status') != 'passed'
                    or final[arm]['boundary'].get('export_exact') is not True):
                raise ValueError('Training result lacks its exact shared input/export boundary')
            progress['completed_tasks'].append(progress['attempted_task'])
        progress['attempted_task'] = dict(kind='evaluate_dataset', dataset='gsm8k', event=5)
        write_json(output/'STATUS.json', progress)
        audited = evaluate(plan, final)
        if audited.get('status') != 'passed' or audited.get('errors') or len(audited.get('rows', [])) != 768:
            raise ValueError('Require all768 raw-audited paired endpoint executions')
        progress['completed_tasks'].append(progress['attempted_task'])
        progress.update(status='completed', attempted_task=None)
    except BaseException as error:
        progress.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        progress['wall_s'] = time.perf_counter()-started
        write_json(output/'STATUS.json', progress)
    return progress
