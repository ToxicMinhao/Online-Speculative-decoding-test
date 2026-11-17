"""Cumulative resource accounting for the unchanged-C MathQA transfer comparison."""
from pathlib import Path
import shutil
import time

from online_feedback.storage import write_json
from experiments.persistent_prefix_withholding.task_runner import (
    exclusive, read, clock_status, occupied_card, compiler_cache_growth,
)
from experiments.bounded_prefix_allocation.runtime import allocated_bytes

LIMITS = dict(adam=528, response_forward=1088, response_backward=1088,
    training_initialization=8, feedback_generation=128, teacher_preparation=128,
    canonical_generation=258, qualification_generation=8,
    warmup_generation=72, scored_generation=2304)
TIME_CAP_S = 14_400
ALLOCATION_BYTES = 90_000_000_000
RESERVE_BYTES = 400_000_000_000


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
    clock = clock_status(plan['root'])
    if max(clock['occupied_s'], clock['elapsed_s']) >= TIME_CAP_S:
        raise TimeoutError('Original four-hour elapsed or occupied-card cap reached')
    allocated = allocated_bytes(plan['allocation_roots'])
    cache = compiler_cache_growth(plan) if compiler else None
    growth = cache['growth_bytes'] if cache else 0
    charged = allocated + plan['compiler_growth_reserved_bytes']
    remaining = max(0, plan['storage']['peak_new_bytes'] - allocated - growth)
    free = shutil.disk_usage(plan['root']).free
    receipt = dict(clock=clock, allocated_bytes=allocated, charged_bytes=charged,
        peak_new_bytes=plan['storage']['peak_new_bytes'], free_bytes=free,
        remaining_peak_bytes=remaining, compiler=cache)
    if charged + upcoming_bytes > ALLOCATION_BYTES or free-max(remaining, upcoming_bytes) < RESERVE_BYTES:
        raise RuntimeError('Peak storage or reserve admission failed: ' + repr(receipt))
    return receipt


def require_time_for(plan, elapsed_remaining_s, card_remaining_s=None):
    clock = clock_status(plan['root'])
    card_remaining_s = elapsed_remaining_s if card_remaining_s is None else card_remaining_s
    elapsed = 1.5 * elapsed_remaining_s + 600
    cards = 1.5 * card_remaining_s + 600
    result = dict(clock=clock, raw_remaining_elapsed_s=elapsed_remaining_s,
        raw_remaining_card_s=card_remaining_s, conservative_elapsed_s=elapsed,
        conservative_card_s=cards, elapsed_cap_s=TIME_CAP_S, card_cap_s=TIME_CAP_S)
    if clock['elapsed_s']+elapsed > TIME_CAP_S or clock['occupied_s']+cards > TIME_CAP_S:
        raise RuntimeError('Remaining fixed campaign does not fit original clocks: ' + repr(result))
    return result


def campaign_identity(plan):
    return dict(binding=plan['binding'], execution_commit=plan['execution_commit'],
        feature_manifest=plan['prerequisite']['feature_manifest'], root=plan['root'])


def start_campaign(plan):
    """Start once, immediately before the first replay model load."""
    initialize_work(plan['root'])
    for name in ('WORK.json', 'CLOCK.json'):
        path=Path(plan['root'])/name;receipt=read(path)
        receipt['campaign_identity']=campaign_identity(plan)
        write_json(path,receipt)


def validate_continuation(plan):
    """Both stages share the original clock, charged work and sealed identity."""
    root=Path(plan['root']);work=read(root/'WORK.json');clock=read(root/'CLOCK.json')
    expected={key:0 for key in LIMITS}
    expected.update(adam=256,response_forward=516,response_backward=516,training_initialization=2)
    if (work.get('campaign_identity') != campaign_identity(plan)
            or clock.get('campaign_identity') != campaign_identity(plan)
            or work['limits'] != LIMITS or work['pending'] or clock['active']
            or work['attempted'] != expected or work['completed'] != expected):
        raise ValueError('Main continuation must retain both exact replay charges and original clock')
    return dict(status='passed',campaign_identity=campaign_identity(plan),work=work,
                clock=clock_status(root),clock_reset=False,work_reset=False)
