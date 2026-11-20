"""Bound the new frozen-endpoint study without changing historical campaigns."""
from pathlib import Path
import math
import shutil
import time

from online_feedback.storage import write_json
from experiments.persistent_prefix_withholding.task_runner import (
    read, exclusive, occupied_card, clock_status,
)
from experiments.bounded_prefix_allocation.runtime import allocated_bytes
from experiments.kplus_full_gsm8k.storage import compiler_snapshot

LIMITS = dict(qualification_generation=24, canonical_generation=1024,
              warmup_generation=288, scored_generation=9216)
ELAPSED_S, CARD_S = 8*3600, 12*3600
PEAK_BYTES, RESERVE_BYTES = 30_000_000_000, 400_000_000_000


class WorkLedger:
    """Reserve each attempt before calling the backend; failures remain pending."""
    def __init__(self, root, task):
        self.root, self.task = Path(root), task

    def call(self, kind, label, function, *args, **kwargs):
        with exclusive(self.root/'WORK.lock'):
            state = read(self.root/'WORK.json')
            if kind not in LIMITS or state['attempted'][kind] >= LIMITS[kind]:
                raise RuntimeError('Generation budget exhausted or unknown category: '+kind)
            key = str(state['next_operation'])
            state['next_operation'] += 1
            state['attempted'][kind] += 1
            state['pending'][key] = dict(task=self.task, kind=kind, label=label)
            write_json(self.root/'WORK.json', state)
        result = function(*args, **kwargs)
        with exclusive(self.root/'WORK.lock'):
            state = read(self.root/'WORK.json')
            state['pending'].pop(key)
            state['completed'][kind] += 1
            write_json(self.root/'WORK.json', state)
        return result


def initialize(root):
    root = Path(root)
    if (root/'WORK.json').exists() or (root/'CLOCK.json').exists():
        raise FileExistsError('Never reset an existing execution clock or work ledger')
    write_json(root/'WORK.json', dict(limits=LIMITS,
        attempted={k:0 for k in LIMITS}, completed={k:0 for k in LIMITS},
        pending={}, next_operation=0, optimizer_steps=0, trained_checkpoints=0))
    write_json(root/'CLOCK.json', dict(first_model_load_epoch=time.time(),
        completed_card_s=0., active={}, finished=[]))


def enforce_clock(plan):
    result = clock_status(plan['root'])
    if result['elapsed_s'] >= ELAPSED_S or result['occupied_s'] >= CARD_S:
        raise TimeoutError('Original eight-hour elapsed / twelve-card-hour budget reached')
    return result


def storage(plan, *, complete_forecast=False):
    """Count each study's declared roots plus shared compiler-cache growth."""
    peak = plan['projected_peak_bytes']
    if type(peak) is not int or not 0 < peak <= PEAK_BYTES:
        raise ValueError('Projected peak must fit the fixed positive 30GB allocation')
    current = allocated_bytes(plan['allocation_roots'])
    compiler = compiler_snapshot()
    if compiler['directories'] != plan['compiler_baseline']['directories']:
        raise ValueError('Qualified compiler-cache roots changed')
    growth = max(0, compiler['allocated_bytes']-plan['compiler_baseline']['allocated_bytes'])
    charged = current+growth
    free = shutil.disk_usage(plan['root']).free
    required = RESERVE_BYTES + (max(0, peak-charged)
                               if complete_forecast else 0)
    if charged > PEAK_BYTES or free < required:
        raise RuntimeError(f'Storage admission failed: charged={charged}, free={free}, required={required}')
    return dict(allocated_bytes=current, compiler_growth_bytes=growth,
                charged_bytes=charged, free_bytes=free, required_free_bytes=required)


def boundary(plan):
    return dict(clock=enforce_clock(plan), storage=storage(plan, complete_forecast=True))


def forecast(plan, elapsed_remaining_s, card_remaining_s):
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0
           for v in (elapsed_remaining_s, card_remaining_s)):
        raise ValueError('Remaining forecasts must be finite and nonnegative')
    clock = enforce_clock(plan)
    # Declared engineering contingency; no scientific endpoint is shortened to fit.
    elapsed = 1.3*elapsed_remaining_s + 900
    cards = 1.3*card_remaining_s + 900
    result = dict(clock=clock, raw_remaining_elapsed_s=elapsed_remaining_s,
        raw_remaining_card_s=card_remaining_s, conservative_elapsed_s=elapsed,
        conservative_card_s=cards, contingency=dict(multiplier=1.3, fixed_s=900),
        guaranteed_upper_bound=False)
    if clock['elapsed_s']+elapsed > ELAPSED_S or clock['occupied_s']+cards > CARD_S:
        raise RuntimeError('The complete planned study no longer fits its original time caps: '+repr(result))
    return result


def schedule_rows(path):
    value = read(path)
    rows = value.get('activations') if isinstance(value, dict) else value
    if not isinstance(rows, list):
        raise ValueError('Require the sealed activation schedule')
    return rows
