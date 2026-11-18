"""Cumulative charged work and clocks for the single full C58 trajectory."""
from pathlib import Path
import shutil
import time

from online_feedback.storage import write_json
from experiments.persistent_prefix_withholding.task_runner import (
    exclusive, read, clock_status, occupied_card, compiler_cache_growth,
)
from experiments.bounded_prefix_allocation.runtime import allocated_bytes

LIMITS = dict(adam=7436, response_forward=14876, response_backward=14876,
    training_initialization=5, feedback_generation=7473, teacher_preparation=7424,
    correctness_generation=4, canonical_generation=4, qualification_generation=24,
    warmup_generation=576, scored_generation=18432)
TIME_CAP_S = 86400
CARD_CAP_S = 144000


def enforce_clock(plan, clock=None):
    clock = clock_status(plan['root']) if clock is None else clock
    if clock['occupied_s'] >= CARD_CAP_S or clock['elapsed_s'] >= TIME_CAP_S:
        raise TimeoutError('Fixed 24-hour campaign or 40-card-hour cap reached')
    return clock


class WorkLedger:
    """Charge before execution; failed calls stay visible as pending work."""
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
        raise FileExistsError('Never reset a prior campaign clock or ledger')
    write_json(root/'WORK.json', dict(limits=LIMITS,
        attempted={k: 0 for k in LIMITS}, completed={k: 0 for k in LIMITS},
        pending={}, next_operation=0))
    write_json(root/'CLOCK.json', dict(first_model_load_epoch=time.time(),
        completed_card_s=0., active={}, finished=[]))


def boundary(plan, *, upcoming_bytes=0, compiler=False):
    from .storage import admit
    clock = enforce_clock(plan)
    result = admit(Path(plan['root']), plan['storage'])
    return dict(clock=clock, storage=result)


def require_time_for(plan, elapsed_remaining_s, card_remaining_s=None):
    clock = enforce_clock(plan)
    cards = elapsed_remaining_s if card_remaining_s is None else card_remaining_s
    result = dict(clock=clock, raw_remaining_elapsed_s=elapsed_remaining_s,
        raw_remaining_card_s=cards, conservative_elapsed_s=1.3*elapsed_remaining_s+1800,
        conservative_card_s=1.3*cards+1800, elapsed_cap_s=TIME_CAP_S,
        card_cap_s=CARD_CAP_S, forecast_policy=dict(factor=1.3,fixed_margin_s=1800),
        forecast_is_guaranteed_upper_bound=False)
    if (clock['elapsed_s']+result['conservative_elapsed_s'] > TIME_CAP_S
            or clock['occupied_s']+result['conservative_card_s'] > CARD_CAP_S):
        raise RuntimeError('Complete fixed campaign no longer fits: '+repr(result))
    return result
