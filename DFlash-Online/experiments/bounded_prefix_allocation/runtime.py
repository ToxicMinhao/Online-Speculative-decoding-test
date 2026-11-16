"""Shared cumulative work, clocks and storage for this fixed bounded campaign."""
from __future__ import annotations

from pathlib import Path
import os
import shutil
import time

from online_feedback.storage import write_json
from experiments.persistent_prefix_withholding.task_runner import (
    exclusive, read, clock_status, occupied_card, compiler_cache_growth,
)

LIMITS = dict(adam=272, response_forward=544, response_backward=544,
    training_initialization=4, feedback_generation=128, teacher_preparation=128,
    qualification_generation=8, warmup_generation=72, scored_generation=1152)
TIME_CAP_S = 4 * 3600
ALLOCATION_BYTES = 80_000_000_000
RESERVE_BYTES = 400_000_000_000


class WorkLedger:
    """Reserve before a call; failures remain pending and charged, never reset."""
    def __init__(self, root, task):
        self.root, self.task = Path(root), task

    def call(self, counts, label, function, *args, **kwargs):
        with exclusive(self.root / 'WORK.lock'):
            state = read(self.root / 'WORK.json')
            for key, amount in counts.items():
                if key not in LIMITS or type(amount) is not int or amount < 0:
                    raise ValueError('Unknown or negative work reservation')
                if state['attempted'][key] + amount > LIMITS[key]:
                    raise RuntimeError('Cumulative work cap exceeded: ' + key)
            number = str(state['next_operation'])
            state['next_operation'] += 1
            for key, amount in counts.items():
                state['attempted'][key] += amount
            state['pending'][number] = dict(task=self.task, label=label, counts=counts)
            write_json(self.root / 'WORK.json', state)
        result = function(*args, **kwargs)
        with exclusive(self.root / 'WORK.lock'):
            state = read(self.root / 'WORK.json')
            entry = state['pending'].pop(number)
            for key, amount in entry['counts'].items():
                state['completed'][key] += amount
            write_json(self.root / 'WORK.json', state)
        return result

    def wrap(self, owner, name, kind):
        original = getattr(owner, name)
        def counted(*args, **kwargs):
            return self.call({kind: 1}, name, original, *args, **kwargs)
        setattr(owner, name, counted)
        return original


def initialize_work(root):
    root = Path(root)
    if (root/'WORK.json').exists() or (root/'CLOCK.json').exists():
        raise FileExistsError('Existing cumulative work must never be reset')
    write_json(root/'WORK.json', dict(limits=LIMITS, attempted={k: 0 for k in LIMITS},
        completed={k: 0 for k in LIMITS}, pending={}, next_operation=0))
    write_json(root/'CLOCK.json', dict(first_model_load_epoch=time.time(),
        completed_card_s=0., active={}, finished=[]))


def allocated_bytes(paths):
    seen, total = set(), 0
    for root in map(Path, paths):
        if not root.exists():
            continue
        for path in [root, *root.rglob('*')]:
            try:
                stat = path.lstat()
            except FileNotFoundError:  # A metadata file may be atomically replaced.
                continue
            key = stat.st_dev, stat.st_ino
            if key not in seen:
                seen.add(key)
                total += stat.st_blocks * 512
    return total


def boundary(plan, *, upcoming_bytes=0, compiler=False):
    """Admission includes all prior staging and atomic-checkpoint overlap."""
    clock = clock_status(plan['root'])
    if clock['occupied_s'] >= TIME_CAP_S or clock['elapsed_s'] >= TIME_CAP_S:
        raise TimeoutError('Original four-hour elapsed or occupied-card cap reached')
    allocated = allocated_bytes(plan['allocation_roots'])
    cache = compiler_cache_growth(plan) if compiler else None
    cache_growth = cache['growth_bytes'] if cache else 0
    cache_reserve = plan['compiler_growth_reserved_bytes']
    charged = allocated + cache_reserve
    remaining = max(0, plan['storage']['peak_new_bytes'] - allocated - cache_growth)
    free = shutil.disk_usage(plan['root']).free
    result = dict(clock=clock, allocated_bytes=allocated, charged_bytes=charged,
        peak_new_bytes=plan['storage']['peak_new_bytes'], free_bytes=free,
        remaining_peak_bytes=remaining, compiler=cache)
    if charged + upcoming_bytes > ALLOCATION_BYTES or free - max(remaining, upcoming_bytes) < RESERVE_BYTES:
        raise RuntimeError('Peak storage/reserve admission failed: ' + repr(result))
    return result


def require_time_for(plan, remaining_s):
    """Price the complete remaining serial chain, with 1.5x plus ten minutes."""
    clock = clock_status(plan['root'])
    conservative = 1.5 * remaining_s + 600
    receipt = dict(raw_remaining_s=remaining_s, conservative_remaining_s=conservative,
        clock=clock, elapsed_cap_s=TIME_CAP_S, occupied_cap_s=TIME_CAP_S)
    if max(clock['elapsed_s'], clock['occupied_s']) + conservative > TIME_CAP_S:
        raise RuntimeError('Whole remaining campaign cannot fit: ' + repr(receipt))
    return receipt
