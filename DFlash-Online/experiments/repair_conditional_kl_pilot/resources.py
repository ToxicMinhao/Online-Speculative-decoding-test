"""Small cumulative clock and write admission, including first-arm offload."""
from pathlib import Path
import shutil
import time

from online_feedback.storage import write_json
from experiments.persistent_prefix_withholding.task_runner import exclusive, read, clock_status, occupied_card
from experiments.bounded_prefix_allocation.runtime import allocated_bytes
from experiments.kplus_full_gsm8k.storage import compiler_snapshot
from .contract import LIMITS


def capacity(requests):
    row_bytes = 49164
    feature = sum(len(r['prompt_ids']) + 2048 for r in requests) * row_bytes
    witness = sum(len(r['prompt_ids']) + 2048 for r in requests[:8]) * row_bytes
    overlap = (max(len(r['prompt_ids']) for r in requests) + 2049) * row_bytes
    weights, recovery = 2_097_259_104, 6_791_777_312
    # Only one arm collects/trains at a time. BASE's prepared pool is offloaded
    # before PAIR starts. Its eight diagnostic packets remain locally readable.
    tensors = feature + witness + overlap + 2 * recovery + 2 * weights
    peak = (tensors * 105 + 99)//100 + 4_000_000_000
    return dict(projected_peak_bytes=peak, active_feature_bound_bytes=feature,
        retained_witness_bound_bytes=witness, replacement_overlap_bytes=overlap,
        recovery_bound_bytes=recovery, export_bytes=weights,
        other_source_logs_compiler_allowance_bytes=4_000_000_000,
        tensor_margin_fraction=.05, full_output_cap_used=2048,
        prerequisite='Verified BASE feature offload before PAIR; disposable qualification tensors retired before production')


def initialize(plan):
    root = Path(plan['root'])
    if (root/'CLOCK.json').exists():
        raise FileExistsError('Pilot GPU clock already exists; no automatic retry')
    write_json(root/'CLOCK.json', dict(first_model_load_epoch=time.time(),
        completed_card_s=0., active={}, finished=[]))


def boundary(plan):
    root = Path(plan['root'])
    clock = clock_status(root) if (root/'CLOCK.json').exists() else dict(elapsed_s=0., occupied_s=0.)
    charged = allocated_bytes(plan['allocation_roots'])
    growth = max(0, compiler_snapshot()['allocated_bytes'] - plan['compiler_baseline']['allocated_bytes'])
    free = shutil.disk_usage(root).free
    result = dict(clock=clock, free_bytes=free, allocated_bytes=charged,
        compiler_growth_bytes=growth, limit=LIMITS,
        remaining_peak_bytes=max(0, plan['storage']['projected_peak_bytes']-charged-growth))
    if free-result['remaining_peak_bytes'] < LIMITS['free_reserve_bytes']:
        raise RuntimeError('Pilot projected peak would cross400GBreserve: '+repr(result))
    if charged+growth > min(LIMITS['new_allocated_bytes'],plan['storage']['projected_peak_bytes']):
        raise RuntimeError('Pilot allocation exceeded its admitted peak')
    if clock['elapsed_s'] >= LIMITS['elapsed_s'] or clock['occupied_s'] >= LIMITS['occupied_card_s']:
        raise TimeoutError('Pilot cumulative time limit reached')
    return result


class Ledger:
    """Persist a pending operation before execution; failed work stays unknown."""
    def __init__(self, path):
        self.path = Path(path)
        if not self.path.exists():
            write_json(self.path, dict(attempted={}, completed={}, pending={}, operations=0))

    def call(self, counts, label, function, *args, **kwargs):
        with exclusive(self.path.with_suffix('.lock')):
            state=read(self.path); key=str(state['operations']); state['operations']+=1
            for name,count in counts.items():
                state['attempted'][name]=state['attempted'].get(name,0)+count
            state['pending'][key]=dict(label=label,counts=counts)
            write_json(self.path,state)
        result=function(*args,**kwargs)
        with exclusive(self.path.with_suffix('.lock')):
            state=read(self.path); state['pending'].pop(key)
            for name,count in counts.items():
                state['completed'][name]=state['completed'].get(name,0)+count
            write_json(self.path,state)
        return result

    def wrap(self, owner, name, kind):
        original=getattr(owner,name)
        def wrapped(*args,**kwargs):
            return self.call({kind:1},name,original,*args,**kwargs)
        setattr(owner,name,wrapped)
        return original


def verify_inputs(plan):
    from experiments.ok_coding.storage import identity
    from .contract import digest
    if plan['binding'] != digest({k:v for k,v in plan.items() if k!='binding'}):
        raise ValueError('Sealed plan changed')
    for path,expected in plan['sealed_files'].items():
        if identity(path)!=expected:
            raise ValueError('Sealed input changed: '+path)
    source=read(plan['source_manifest'])
    if source['commit']!=plan['execution_commit']:
        raise ValueError('Staged source commit changed')
    for relative,expected in source['files'].items():
        if identity(Path(plan['source_root'])/relative)!=expected:
            raise ValueError('Staged source changed: '+relative)
    return dict(status='passed',sealed_files=len(plan['sealed_files']),source_files=len(source['files']))
