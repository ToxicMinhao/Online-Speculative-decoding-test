"""Durable work receipts and resource admission for the fixed two-arm run."""
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time

from experiments.persistent_prefix_withholding.task_runner import exclusive, clock_status, occupied_card
from experiments.bounded_prefix_allocation.runtime import allocated_bytes
from experiments.kplus_full_gsm8k.storage import compiler_snapshot

LIMITS = dict(elapsed_s=72*3600, occupied_card_s=120*3600,
              new_allocated_bytes=90_000_000_000, free_reserve_bytes=400_000_000_000)
QUALIFICATION_LIMITS = dict(adam=8, generation=16)
PRODUCTION_TARGETS = dict(feedback_generation=14848, teacher_preparation=14848,
    response_forward=29696, response_backward=29696, adam=14848,
    correctness_generation=8, witness_response_forward=72,
    canonical_generation=1024, scored_generation=12288, warmup_generation=96, activation=48)


def read(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    """Each writer owns its temporary file; publication is one atomic rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent,
                                    prefix=path.name+'.', suffix='.tmp', delete=False) as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    temporary.replace(path)


def initialize(root):
    root = Path(root)
    if (root/'CLOCK.json').exists() or (root/'WORK.json').exists():
        raise FileExistsError('Preserve the original full-run clock and work ledger')
    write_json(root/'CLOCK.json', dict(first_model_load_epoch=time.time(),
        completed_card_s=0., active={}, finished=[]))
    write_json(root/'WORK.json', dict(next_operation=0, attempted={}, completed={},
        pending={}, failures=[], qualification_limits=QUALIFICATION_LIMITS,
        production_targets=PRODUCTION_TARGETS))


class WorkLedger:
    """Reserve before a physical operation; an interrupted call remains unknown."""
    def __init__(self, root, task, phase='production'):
        self.root, self.task, self.phase = Path(root), task, phase
        if phase not in ('qualification', 'production', 'startup', 'retry'):
            raise ValueError('Unknown work accounting scope')

    def reserve_counts(self, counts, label):
        charged = {self.phase+'.'+key:value for key,value in counts.items()}
        with exclusive(self.root/'WORK.lock'):
            state = read(self.root/'WORK.json')
            for key,value in charged.items():
                if type(value) is not int or value < 0:
                    raise ValueError('Work counts must be nonnegative integers')
                kind = key.split('.', 1)[1]
                limit = QUALIFICATION_LIMITS.get(kind) if self.phase == 'qualification' else None
                if limit is not None and state['attempted'].get(key,0)+value > limit:
                    raise RuntimeError('Qualification work ceiling reached: '+key)
            number = str(state['next_operation'])
            state['next_operation'] += 1
            for key,value in charged.items():
                state['attempted'][key] = state['attempted'].get(key,0)+value
            state['pending'][number] = dict(task=self.task, label=label, counts=charged,
                                             started_epoch=time.time())
            write_json(self.root/'WORK.json', state)
        return number

    def reserve(self, arm):
        return self.reserve_counts({'adam':1}, 'native_adam_'+arm)

    def complete(self, number):
        with exclusive(self.root/'WORK.lock'):
            state = read(self.root/'WORK.json')
            pending = state['pending'].pop(str(number))
            for key,value in pending['counts'].items():
                state['completed'][key] = state['completed'].get(key,0)+value
            write_json(self.root/'WORK.json', state)

    def call(self, counts, label, function, *args, **kwargs):
        number = self.reserve_counts(counts, label)
        try:
            result = function(*args, **kwargs)
        except BaseException as error:
            with exclusive(self.root/'WORK.lock'):
                state = read(self.root/'WORK.json')
                state['failures'].append(dict(operation=number, error=repr(error),
                    reconciliation='Physical outcome remains unknown until checked against retained evidence'))
                write_json(self.root/'WORK.json', state)
            raise
        self.complete(number)
        return result

    def wrap(self, owner, name, kind):
        original = getattr(owner, name)
        def counted(*args, **kwargs):
            return self.call({kind:1}, name, original, *args, **kwargs)
        setattr(owner, name, counted)
        return original


def storage(plan, *, reserve_remaining=True):
    """Keep future atomic tensor space reserved even if compact evidence grows."""
    actual = allocated_bytes(plan['allocation_roots'])
    seen, tensor_bytes = set(), 0
    for folder in map(Path,plan['allocation_roots']):
        for path in folder.rglob('*'):
            if path.suffix not in ('.pt','.safetensors'):
                continue
            try:
                stat = path.lstat()
            except FileNotFoundError:
                continue
            inode = (stat.st_dev,stat.st_ino)
            if inode not in seen:
                tensor_bytes += stat.st_blocks*512
                seen.add(inode)
    compiler = compiler_snapshot()
    baseline = plan['compiler_baseline']
    if compiler['directories'] != baseline['directories']:
        raise ValueError('Compiler accounting paths changed')
    growth = max(0,compiler['allocated_bytes']-baseline['allocated_bytes'])
    phase = 'training'
    statuses = [Path(plan['root'])/'campaign_001/arms'/arm/'STATUS.json' for arm in ('BASE','TAIL')]
    if all(path.exists() and read(path).get('status')=='completed' and
           read(path).get('events_completed')==58 for path in statuses):
        phase = 'evaluation'
    components = plan['storage']['phases'][phase]
    compact_budget = sum(components[key] for key in ('compact_bytes','source_bytes','workspace_bytes'))
    remaining_components = dict(
        tensors=max(0,components['tensor_peak_with_five_percent_margin_bytes']-tensor_bytes),
        compact=max(0,compact_budget-(actual-tensor_bytes)),
        compiler=max(0,components['compiler_bytes']-growth))
    local = plan['storage']['external_charge_bytes']
    charged = actual+growth+local
    remaining = sum(remaining_components.values()) if reserve_remaining else 0
    peak = plan['storage']['projected_peak_bytes']
    free = shutil.disk_usage(plan['root']).free
    result = dict(remote_allocated_bytes=actual, tensor_allocated_bytes=tensor_bytes,
        compiler_growth_bytes=growth, external_charge_bytes=local, charged_bytes=charged,
        free_bytes=free, projected_peak_bytes=peak, storage_phase=phase,
        remaining_peak_bytes=remaining,remaining_components=remaining_components,
        required_free_bytes=LIMITS['free_reserve_bytes']+remaining)
    if (peak>LIMITS['new_allocated_bytes'] or charged+remaining>LIMITS['new_allocated_bytes']
            or free<result['required_free_bytes']):
        raise RuntimeError('Full-run storage admission failed: '+repr(result))
    return result


def begin_card(root, task, uuid):
    root=Path(root)
    with exclusive(root/'CLOCK.lock'):
        state=read(root/'CLOCK.json')
        if task in state['active'] or any(v['physical_gpu_uuid']==uuid for v in state['active'].values()):
            raise RuntimeError('Physical card already has an unresolved owner')
        state['active'][task]=dict(physical_gpu_uuid=uuid,started_epoch=time.time(),pid=os.getpid())
        write_json(root/'CLOCK.json',state)


def end_card(root, task):
    root=Path(root)
    with exclusive(root/'CLOCK.lock'):
        state=read(root/'CLOCK.json');lease=state['active'].pop(task)
        seconds=time.time()-lease['started_epoch']
        state['completed_card_s']+=seconds
        state['finished'].append(dict(task=task,**lease,finished_epoch=time.time(),occupied_s=seconds))
        write_json(root/'CLOCK.json',state)


def boundary(plan):
    clock = clock_status(plan['root'])
    if (clock['elapsed_s'] >= LIMITS['elapsed_s']
            or clock['occupied_s'] >= LIMITS['occupied_card_s']):
        raise TimeoutError('Original full-run time ceiling reached')
    return dict(clock=clock, storage=storage(plan))


def forecast(plan, *, remaining_wall_s, remaining_card_s, destination):
    for value in (remaining_wall_s, remaining_card_s):
        if not math.isfinite(value) or value < 0:
            raise ValueError('Resource estimates must be finite and nonnegative')
    checked = boundary(plan)
    wall = checked['clock']['elapsed_s']+1.25*remaining_wall_s+3600
    cards = checked['clock']['occupied_s']+1.25*remaining_card_s+3600
    result = dict(**checked, raw_remaining_wall_s=remaining_wall_s,
        raw_remaining_card_s=remaining_card_s, contingency_factor=1.25,
        fixed_allowance_s=3600, forecast_wall_s=wall, forecast_card_s=cards,
        guaranteed_upper_bound=False,
        status='admitted' if wall<=LIMITS['elapsed_s'] and cards<=LIMITS['occupied_card_s'] else 'blocked')
    write_json(destination, result)
    if result['status'] != 'admitted':
        raise RuntimeError('Remaining unchanged campaign no longer fits time ceilings')
    return result
