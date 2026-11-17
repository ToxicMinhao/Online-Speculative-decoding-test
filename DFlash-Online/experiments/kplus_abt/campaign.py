"""Fixed shared-feedback events 5–6 controller; never constructs a model."""
from pathlib import Path
import time

from online_feedback.storage import write_json
from .contract import ARMS, DATASETS, EVENTS, DIAGNOSTIC_WINDOWS, budgets
from .data import read, verify_shared_packets


def tasks():
    result = []
    for dataset in DATASETS:
        for event in EVENTS:
            result.append(dict(kind='collect_shared', dataset=dataset, event=event,
                               producer='KPLUS', producer_event=event-1))
            result.extend(dict(kind='train_arm', dataset=dataset, event=event, arm=arm) for arm in ARMS)
        result.append(dict(kind='evaluate_dataset', dataset=dataset, event=6))
    return result


def schedule_bindings(result):
    records = result['records']
    if len(records) != 256:
        raise ValueError('Every event requires 256 ordinary presentations')
    for pass_index in (0, 1):
        rows = records[pass_index*128:(pass_index+1)*128]
        if any(row['pass'] != pass_index for row in rows) or len({r['record_id'] for r in rows}) != 128:
            raise ValueError('Each ordinary pass must contain its 128 distinct shared responses')
    if {r['record_id'] for r in records[:128]} != {r['record_id'] for r in records[128:]}:
        raise ValueError('The two passes differ in response membership')
    return [{key: row[key] for key in ('record_id', 'pass', 'sampled_anchors')} for row in records]


def validate_update(result, *, dataset, arm, event, baseline=None):
    if dataset not in DATASETS or arm not in ARMS or event not in EVENTS:
        raise ValueError('Unknown ABT event')
    expected = dict(status='completed', method=arm, dataset=dataset,
        event_id=f'event_{event:03d}', start_step=(event-1)*128,
        end_step=event*128, optimizer_steps=128, response_presentations=256, passes=2)
    if any(result.get(key) != value for key, value in expected.items()):
        raise ValueError('Ordinary continuous event accounting changed')
    actual = schedule_bindings(result)
    if baseline is not None and actual != schedule_bindings(baseline):
        raise ValueError('Shared feedback response order or native anchors differ across arms')
    work = result['abt_work']
    views = 256 if arm == 'KPLUS' else 512
    if any(work.get(key) != value for key, value in dict(response_forwards=views,
            response_backwards=views, optimizer_steps=128).items()):
        raise ValueError('Main view/optimizer counts changed')
    if any(work.get(key, 0) != 0 for key in ('failed', 'unknown', 'extra_optimizer_steps')):
        raise ValueError('Failed or unknown native work')
    diagnostics = result['diagnostic_windows']
    if [row.get('window') for row in diagnostics] != list(DIAGNOSTIC_WINDOWS):
        raise ValueError('Require all four fixed diagnostic windows in every arm-event')
    for row in diagnostics:
        if (row.get('status') != 'completed' or row.get('response_forwards') != 2
                or row.get('response_backwards') != 0 or row.get('optimizer_steps') != 0
                or row.get('rng_restored_exact') is not True):
            raise ValueError('Post-update diagnostic work/restoration changed')
    return actual


def require_admission(plan):
    from .storage import check_current_capacity
    if plan.get('budgets') != budgets() or plan.get('automatic_retry') is not False:
        raise ValueError('Sealed ABT budget or retry rule changed')
    receipt = read(plan['qualification_admission'])
    required = ('native_neutral_exact', 'two_view_execution', 'mask_scope',
                'perturbation_gradient', 'bf16_realization', 'same_clean_selection',
                'one_adam_per_window', 'serving', 'resource', 'complete_state')
    if (receipt.get('status') != 'passed' or receipt.get('binding') != plan['binding']
            or receipt.get('execution_commit') != plan['execution_source']['commit']
            or any(receipt.get('checks', {}).get(k) is not True for k in required)):
        raise ValueError('Every native ABT qualification gate is required')
    storage = check_current_capacity(plan['root'], plan['main_storage'])
    if storage['status'] != 'passed':
        raise ValueError('Main ABT storage has not been admitted')
    return dict(qualification=receipt, storage=storage)


def run(root, *, collect_shared, train_arm, evaluate_dataset, retire_shared=None):
    root = Path(root)
    plan = read(root/'sealed/PLAN.json')
    output = root/'campaign_001'
    output.mkdir(exist_ok=False)
    progress = dict(status='running', completed_tasks=[], attempted_task=None,
                    automatic_retry=False, execution_source=plan['execution_source'])
    started = time.perf_counter()
    pools, updates = {}, {}
    try:
        progress['admission'] = require_admission(plan)
        for task in tasks():
            progress['attempted_task'] = task
            write_json(output/'STATUS.json', progress)
            dataset, event = task['dataset'], task['event']
            if task['kind'] == 'collect_shared':
                producer = updates.get((dataset, event-1, 'KPLUS'))
                pool = collect_shared(plan, dataset, event, producer)
                if (pool.get('dataset') != dataset or pool.get('event') != event
                        or pool.get('producer') != f'KPLUS{event-1}' or len(pool['records']) != 128):
                    raise ValueError('Feedback producer is not the exact current KPLUS boundary')
                verify_shared_packets(pool)
                pools[dataset, event] = pool
            elif task['kind'] == 'train_arm':
                arm = task['arm']; pool = pools[dataset, event]
                verify_shared_packets(pool)
                result = train_arm(plan, dataset, event, arm, pool, updates.get((dataset, event-1, arm)))
                validate_update(result, dataset=dataset, arm=arm, event=event,
                    baseline=updates.get((dataset, event, 'KPLUS')) if arm != 'KPLUS' else None)
                verify_shared_packets(pool)
                updates[dataset, event, arm] = result
                if arm == ARMS[-1] and retire_shared is not None:
                    retire_shared(plan, dataset, event, pool,
                        {name: updates[dataset, event, name] for name in ARMS})
            else:
                measured = evaluate_dataset(plan, dataset, {arm: updates[dataset, 6, arm] for arm in ARMS})
                if measured.get('status') != 'passed' or len(measured.get('rows', [])) != 2048:
                    raise ValueError('Final panel requires all 2048 valid raw-audited executions')
            progress['completed_tasks'].append(task)
            progress['attempted_task'] = None
            write_json(output/'STATUS.json', progress)
        progress['status'] = 'completed'
    except BaseException as error:
        progress.update(status='failed', error=repr(error))
        raise
    finally:
        progress.update(wall_s=time.perf_counter()-started,
            unattempted_tasks=len(tasks())-len(progress['completed_tasks'])-int(progress['attempted_task'] is not None))
        write_json(output/'STATUS.json', progress)
    return progress
