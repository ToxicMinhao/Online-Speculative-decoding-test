"""Fixed events-five/six orchestration; model work is supplied by native workers.

This controller is intentionally separate from qualification. It cannot skip the
sixteen diagnostic windows or launch from a passing numerical subcheck alone.
The native worker adapter must provide the three concrete operations below;
this file never creates a model, optimizer, or replacement feedback trajectory.
"""
import importlib
import json
from pathlib import Path
import shutil
import time

from online_feedback.storage import write_json
from .contract import ARMS, DATASETS, EVENTS, MAIN_BUDGET, DIAGNOSTIC_BUDGET, REALIZATION_R1
from .data import verify_shared_packets

REQUIRED_GATES = ('native_formula_both_datasets', 'native_state', 'serving', 'resource')
DIAGNOSTIC_WINDOWS = (1, 32, 65, 128)


def read(path):
    return json.loads(Path(path).read_text())


def tasks():
    """All arms finish one shared window before KPLUS collects the next one."""
    result = []
    for dataset in DATASETS:
        for event in EVENTS:
            result.append(dict(kind='collect_shared', dataset=dataset, event=event,
                               producer='KPLUS', producer_event=event-1))
            result.extend(dict(kind='train_arm', dataset=dataset, event=event, arm=arm)
                          for arm in ARMS)
        result.append(dict(kind='evaluate_dataset', dataset=dataset, event=6))
    return result


def diagnostic_pairs(presentations, *, dataset, event):
    """Bind off-fit pair D exactly 32 pairs ahead within each original pass."""
    if dataset not in DATASETS or event not in EVENTS or len(presentations) != 256:
        raise ValueError('Require one complete declared 256-presentation event')
    for pass_index in (0, 1):
        rows = presentations[pass_index*128:(pass_index+1)*128]
        if (any(row['pass'] != pass_index for row in rows)
                or len({row['record_id'] for row in rows}) != 128):
            raise ValueError('Each native pass must contain its 128 distinct responses')
    if {row['record_id'] for row in presentations[:128]} != {row['record_id'] for row in presentations[128:]}:
        raise ValueError('The two native passes must use the same feedback pool')
    result = []
    for window in DIAGNOSTIC_WINDOWS:
        zero = window-1
        pass_index, pair_index = divmod(zero, 64)
        offset = pass_index*128
        a = presentations[offset+2*pair_index:offset+2*pair_index+2]
        d_index = (pair_index+32) % 64
        d = presentations[offset+2*d_index:offset+2*d_index+2]
        a_ids = [row['record_id'] for row in a]
        d_ids = [row['record_id'] for row in d]
        if len(set(a_ids+d_ids)) != 4:
            raise ValueError('Fitting and off-fit diagnostic requests must be disjoint')
        result.append(dict(dataset=dataset, event=event, window=window,
            pass_index=pass_index, fitting_ids=a_ids, off_fit_ids=d_ids,
            diagnostic_rng_key=[20261108, DATASETS.index(dataset), event, window],
            scope='off-fit buffered training data, not held-out evaluation'))
    return result


def storage_estimate(weight_bytes, state_bytes):
    """Conservative no-retirement ceiling, never credits an intended deletion.

    Exact rolling retention can be sealed later, but must include validated
    successors and diagnostic completion before any feature/state is retired.
    This conservative alternative retains all twelve new boundaries and pools.
    """
    if any(type(value) is not int or value <= 0 for value in (weight_bytes, state_bytes)):
        raise ValueError('Use measured retained export and complete-state sizes')
    from experiments.adaflash_c1.storage import RAW_BYTES_PER_GENERATION
    gib = 2**30
    feature_token_bytes = 4096*6*2+12
    features = (512*2272+2272+1)*feature_token_bytes
    states = 12*(weight_bytes+state_bytes)+state_bytes+weight_bytes
    tensor_peak = states+features
    compact = (512+3072+512)*RAW_BYTES_PER_GENERATION+4*gib
    return dict(policy='retain_all_new_boundaries_and_shared_features',
        weight_bytes=weight_bytes, state_bytes=state_bytes,
        tensor_peak_bytes=tensor_peak, tensor_peak_with_margin_bytes=(tensor_peak*105+99)//100,
        compact_bytes=compact, compiler_bytes=4*gib, workspace_bytes=2*gib,
        reserve_bytes=8*gib, no_future_deletion_credit=True,
        required_initial_free_bytes=(tensor_peak*105+99)//100+compact+14*gib,
        shared_feature_tokens=512*2272, final_checkpoints=6, retained_new_boundaries=12)


def require_pilot_admission(plan, root):
    """The replacement gate is explicit; old failures never turn into passes."""
    receipt = read(plan['pilot_admission'])
    if (receipt.get('status') != 'passed' or receipt.get('pilot_permitted') is not True
            or receipt.get('execution_commit') != plan['execution_source']['commit']
            or receipt.get('binding') != plan['binding']
            or any(receipt.get('checks', {}).get(name) is not True for name in REQUIRED_GATES)
            or type(receipt.get('nontrivial_accepted_fsg_cases')) is not int
            or receipt['nontrivial_accepted_fsg_cases'] < 1
            or receipt.get('scalar_matching_exercised_cases', 0) < 1
            or receipt.get('realization_revision') != REALIZATION_R1
            or plan.get('realization_revision') != REALIZATION_R1):
        raise ValueError('Pilot requires every replacement/outstanding check and a nontrivial accepted FSG case')
    if plan.get('startup_budget') != dict(response_forwards=24, backwards=24, optimizer_steps=0):
        raise ValueError('Declare twelve discarded two-response native startups separately')
    if plan.get('main_budget') != MAIN_BUDGET or plan.get('diagnostic_budget') != DIAGNOSTIC_BUDGET:
        raise ValueError('Main or sixteen-window diagnostic budget changed')
    factory = plan.get('diagnostic_factory')
    if not factory or ':' not in factory:
        raise ValueError('The sixteen-window native diagnostic implementation is not yet bound')
    module, name = factory.split(':', 1)
    if not callable(getattr(importlib.import_module(module), name)):
        raise ValueError('A real diagnostic implementation is required before main work')
    storage = plan['main_storage']
    required = storage['required_initial_free_bytes']
    free = shutil.disk_usage(root).free
    admission = dict(status='passed' if free >= required else 'failed', free_bytes=free,
                     required_bytes=required, plan=storage)
    write_json(Path(root)/'PILOT_STORAGE_ADMISSION.json', admission)
    if not storage.get('no_future_deletion_credit') or free < required:
        raise ValueError('Full-pilot storage has not been admitted')
    return dict(qualification=receipt, storage=admission)


def schedule_bindings(result):
    """Compare normal native draws; never force a sampler to match afterwards."""
    if len(result['records']) != 256:
        raise ValueError('Event must contain 256 original response presentations')
    return [{key: row[key] for key in ('record_id', 'pass', 'sampled_anchors')}
            for row in result['records']]


def validate_update(result, *, dataset, arm, event, baseline=None):
    if (result.get('status') != 'completed' or result.get('method') != arm
            or result.get('dataset') != dataset or result.get('event_id') != f'event_{event:03d}'
            or result.get('start_step') != (event-1)*128 or result.get('end_step') != event*128
            or result.get('optimizer_steps') != 128 or result.get('response_presentations') != 256
            or result.get('passes') != 2):
        raise ValueError('Ordinary continuous event accounting changed')
    binding = schedule_bindings(result)
    if baseline is not None and binding != schedule_bindings(baseline):
        raise ValueError('Shared feedback native order or anchors differ across arms')
    if arm == 'FSG':
        diagnostic = result.get('diagnostic_windows', [])
        if [row.get('window') for row in diagnostic] != list(DIAGNOSTIC_WINDOWS):
            raise ValueError('All four prespecified diagnostic windows are required per FSG event')
        for row in diagnostic:
            if (row.get('status') != 'completed' or row.get('event') != event
                    or row.get('dataset') != dataset or row.get('restoration_exact') is not True
                    or type(row.get('response_forwards')) is not int
                    or type(row.get('backwards')) is not int
                    or not 0 <= row['response_forwards'] <= 40
                    or not 0 <= row['backwards'] <= 4):
                raise ValueError('Counterfactual accounting/restoration is incomplete')
    return binding


def run(root, *, collect_shared, train_arm, evaluate_dataset, retire_shared=None):
    """Run only the sealed causal sequence; worker failures are never retried.

    Operations must return completed immutable packet inventories, complete
    ordinary training receipts, and raw-audited endpoint results respectively.
    All model construction/closure stays in those qualified native workers.
    """
    root = Path(root)
    plan = read(root/'sealed/PLAN.json')
    output = root/'campaign_001'
    output.mkdir(exist_ok=False)
    progress = dict(status='running', execution_source=plan['execution_source'],
                    completed_tasks=[], attempted_task=None, automatic_retry=False)
    started = time.perf_counter()
    pools, updates = {}, {}
    try:
        progress['admission'] = require_pilot_admission(plan, root)
        for task in tasks():
            progress['attempted_task'] = task
            write_json(output/'STATUS.json', progress)
            dataset, event = task['dataset'], task['event']
            if task['kind'] == 'collect_shared':
                producer = updates.get((dataset, event-1, 'KPLUS'))
                pool = collect_shared(plan, dataset, event, producer)
                if (pool.get('dataset') != dataset or pool.get('event') != event
                        or pool.get('producer') != f'KPLUS{event-1}' or len(pool['records']) != 128):
                    raise ValueError('Feedback pool does not belong to the exact pre-update KPLUS')
                verify_shared_packets(pool)
                pools[dataset, event] = pool
            elif task['kind'] == 'train_arm':
                arm = task['arm']; pool = pools[dataset, event]
                verify_shared_packets(pool)
                previous = updates.get((dataset, event-1, arm))
                result = train_arm(plan, dataset, event, arm, pool, previous)
                validate_update(result, dataset=dataset, arm=arm, event=event,
                                baseline=updates.get((dataset, event, 'KPLUS')) if arm != 'KPLUS' else None)
                verify_shared_packets(pool)
                updates[dataset, event, arm] = result
                if arm == 'SCALAR' and retire_shared is not None:
                    retire_shared(plan, dataset, event, pool,
                        {name: updates[dataset, event, name] for name in ARMS})
            else:
                final = {arm: updates[dataset, 6, arm] for arm in ARMS}
                measured = evaluate_dataset(plan, dataset, final)
                if measured.get('status') != 'passed' or len(measured.get('rows', [])) != 1536:
                    raise ValueError('Final panel requires all 1536 valid, raw-audited executions')
            progress['completed_tasks'].append(task)
            progress['attempted_task'] = None
            write_json(output/'STATUS.json', progress)
        progress['status'] = 'completed'
    except BaseException as error:
        progress.update(status='failed', error=repr(error))
        raise
    finally:
        progress['wall_s'] = time.perf_counter()-started
        progress['unattempted_tasks'] = len(tasks())-len(progress['completed_tasks'])-int(progress['attempted_task'] is not None)
        write_json(output/'STATUS.json', progress)
    return progress
