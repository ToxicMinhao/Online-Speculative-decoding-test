"""Cumulative work, time and storage admission for one fixed CODE study."""
from pathlib import Path
import math
import shutil
import time

from online_feedback.storage import write_json
from experiments.persistent_prefix_withholding.task_runner import (
    exclusive, read, clock_status, occupied_card,
)
from experiments.bounded_prefix_allocation.runtime import allocated_bytes
from experiments.kplus_full_gsm8k.storage import compiler_snapshot
from .contract import ARMS, PRODUCTION_WORK, QUALIFICATION_CEILINGS, RESOURCE_LIMITS

LIMITS = {**{'production.'+k:v for k,v in PRODUCTION_WORK.items()},
          **{'qualification.'+k:v for k,v in QUALIFICATION_CEILINGS.items()}}


def initialize(root):
    root = Path(root)
    if (root/'WORK.json').exists() or (root/'CLOCK.json').exists():
        raise FileExistsError('An execution clock or work ledger already exists')
    write_json(root/'WORK.json', dict(limits=LIMITS, next_operation=0,
        attempted={k:0 for k in LIMITS}, completed={k:0 for k in LIMITS},
        pending={}, failures=[]))
    write_json(root/'CLOCK.json', dict(first_model_load_epoch=time.time(),
        completed_card_s=0., active={}, finished=[]))


class WorkLedger:
    """Reserve before execution; record exceptions without claiming unknown work finished."""
    def __init__(self, root, task, phase):
        if phase not in ('qualification', 'production'):
            raise ValueError('Work must belong to qualification or production')
        self.root, self.task, self.phase = Path(root), task, phase

    def call(self, counts, label, function, *args, **kwargs):
        from .policy_resume import effective_work_limits
        limits = effective_work_limits(self.root)
        charged = {self.phase+'.'+key:value for key,value in counts.items()}
        with exclusive(self.root/'WORK.lock'):
            state = read(self.root/'WORK.json')
            if state['limits'] != limits:
                raise ValueError('The sealed work ceilings changed')
            for key, value in charged.items():
                if key not in limits or type(value) is not int or value < 0:
                    raise ValueError('Unknown or negative work reservation')
                if state['attempted'][key]+value > limits[key]:
                    raise RuntimeError('Work ceiling reached: '+key)
            number = str(state['next_operation'])
            state['next_operation'] += 1
            for key,value in charged.items():
                state['attempted'][key] += value
            state['pending'][number] = dict(task=self.task, label=label, counts=charged)
            write_json(self.root/'WORK.json', state)
        try:
            result = function(*args, **kwargs)
        except BaseException as error:
            with exclusive(self.root/'WORK.lock'):
                state = read(self.root/'WORK.json')
                state['failures'].append(dict(operation=number, error=repr(error),
                    reconciliation='Actual work remains unknown until evidence resolves it'))
                write_json(self.root/'WORK.json', state)
            raise
        with exclusive(self.root/'WORK.lock'):
            state = read(self.root/'WORK.json')
            entry = state['pending'].pop(number)
            for key,value in entry['counts'].items():
                state['completed'][key] += value
            write_json(self.root/'WORK.json', state)
        return result

    def wrap(self, owner, name, kind):
        original = getattr(owner, name)
        def counted(*args, **kwargs):
            return self.call({kind:1}, name, original, *args, **kwargs)
        setattr(owner, name, counted)
        return original


def storage(plan, *, complete_forecast=True):
    peak = plan['storage']['projected_peak_bytes']
    if type(peak) is not int or not 0 < peak <= RESOURCE_LIMITS['new_allocated_bytes']:
        raise ValueError('The complete peak must fit the authorized 90GB limit')
    current = allocated_bytes(plan['allocation_roots'])
    compiler = compiler_snapshot()
    baseline = plan['compiler_baseline']
    if compiler['directories'] != baseline['directories']:
        raise ValueError('Compiler cache accounting roots changed')
    growth = max(0, compiler['allocated_bytes']-baseline['allocated_bytes'])
    charged = current+growth
    # One component's unexpected growth must not consume space reserved for
    # future atomic recoveries in another component.
    seen, tensor_bytes = set(), 0
    for root in map(Path,plan['allocation_roots']):
        for path in root.rglob('*'):
            if path.suffix not in ('.pt','.safetensors'):
                continue
            try:
                stat=path.lstat()
            except FileNotFoundError:
                continue
            inode=stat.st_dev,stat.st_ino
            if inode not in seen:
                tensor_bytes+=stat.st_blocks*512
                seen.add(inode)
    components=plan['storage']
    phase='training'
    transition=Path(plan['root'])/'STORAGE_PHASE.json'
    if transition.exists():
        marker=read(transition)
        if marker!=dict(phase='evaluation',training_complete=True):
            raise ValueError('Unknown storage lifecycle transition')
        for arm in ARMS:
            completed=read(Path(plan['root'])/'campaign_001/arms'/arm/'STATUS.json')
            if completed['status']!='completed' or completed['events_completed']!=58:
                raise ValueError('Lower final tensor reservation requires all58-event arms complete')
        phase='evaluation'
    if 'phases' in components:
        components=components['phases'][phase]
    compact_budget=sum(components[key] for key in ('compact_bytes','source_bytes','workspace_bytes'))
    remaining_parts=dict(
        tensors=max(0,components['tensor_peak_with_five_percent_margin_bytes']-tensor_bytes),
        compact=max(0,compact_budget-(current-tensor_bytes)),
        compiler=max(0,components['compiler_bytes']-growth))
    remaining = sum(remaining_parts.values()) if complete_forecast else 0
    free = shutil.disk_usage(plan['root']).free
    result = dict(allocated_bytes=current, compiler_growth_bytes=growth,
        charged_bytes=charged, free_bytes=free, projected_peak_bytes=peak,storage_phase=phase,
        remaining_peak_bytes=remaining,remaining_components=remaining_parts,
        required_free_bytes=RESOURCE_LIMITS['free_reserve_bytes']+remaining)
    if charged+remaining > RESOURCE_LIMITS['new_allocated_bytes'] or free < result['required_free_bytes']:
        raise RuntimeError('CODE storage admission failed: '+repr(result))
    return result


def boundary(plan):
    clock = clock_status(plan['root'])
    if (clock['elapsed_s'] >= RESOURCE_LIMITS['elapsed_s']
            or clock['occupied_s'] >= RESOURCE_LIMITS['occupied_card_s']):
        raise TimeoutError('CODE elapsed or occupied-card budget reached')
    return dict(clock=clock, storage=storage(plan))


def forecast(plan, elapsed_remaining_s, card_remaining_s):
    values = (elapsed_remaining_s, card_remaining_s)
    if any(type(v) not in (int,float) or not math.isfinite(v) or v < 0 for v in values):
        raise ValueError('Remaining forecasts must be finite and nonnegative')
    checked = boundary(plan)
    elapsed = checked['clock']['elapsed_s'] + 1.25*elapsed_remaining_s + 3600
    cards = checked['clock']['occupied_s'] + card_remaining_s
    result = dict(**checked, raw_remaining_elapsed_s=elapsed_remaining_s,
        estimated_remaining_occupied_card_s=card_remaining_s,
        contingency_elapsed_total_s=elapsed, projected_card_total_s=cards,
        elapsed_contingency=dict(factor=1.25, reserve_s=3600),
        forecast_is_guaranteed_upper_bound=False)
    if elapsed > RESOURCE_LIMITS['elapsed_s'] or cards > RESOURCE_LIMITS['occupied_card_s']:
        raise RuntimeError('The complete unchanged CODE study no longer fits: '+repr(result))
    return result


def require_eligible_response(record):
    """Stop before charging a teacher forward that cannot actually execute."""
    if len(record['response_ids']) < 2:
        raise ValueError('ineligible_short_response: preserve this response and stop its window')
