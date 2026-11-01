"""Peak admission and retirement of this run's superseded odd-event tensors."""
import json
import os
from pathlib import Path
import shutil

from experiments.ok_events_5_8.storage import GIB, WEIGHTS, identity, required_bytes
from online_feedback.storage import write_json

JOBS = tuple((seed, event) for seed in (17, 29) for event in range(1, 5))
# The completed identical-shape coding workers used <=61 MB/job. Eight GiB
# covers all eight fresh worker caches plus native initialization, without reuse.
COMPILER_BYTES = 8 * GIB


def future_capacity(completed_jobs=0, completed_initials=0):
    if not 0 <= completed_jobs <= 8 or not 0 <= completed_initials <= 2:
        raise ValueError('Invalid complete-boundary storage counters')
    if completed_initials != len({seed for seed, _ in JOBS[:completed_jobs]}):
        raise ValueError('Initial-state count disagrees with complete chronology')
    written = set()
    initialized = {seed for seed, _ in JOBS[:completed_jobs]}
    live = peak = 0
    for seed, event in JOBS[completed_jobs:]:
        if seed not in initialized:
            live += WEIGHTS
            initialized.add(seed)
        # 3W complete Adam state + W export, with another W export workspace.
        live += 4 * WEIGHTS
        peak = max(peak, live + WEIGHTS)
        written.add((seed, event))
        previous = (seed, event - 1)
        if event in (2, 4) and previous in written:
            # Credit only a future odd-event artifact created by this plan.
            # Already occupied files never reduce this additional-write bound.
            live -= 4 * WEIGHTS
            written.remove(previous)
    # Completed workers' occupied cache bytes already reduce measured free space.
    # Charge one GiB per remaining cold worker, retaining at least one GiB for
    # remaining serving initialization. This never credits deleting old caches.
    compiler = max(1, len(JOBS) - completed_jobs) * GIB
    payload = peak + compiler
    return dict(payload_bytes=payload, peak_required_bytes=required_bytes(payload),
                future_tensor_peak_bytes=peak, compiler_bytes=compiler,
                reserve_bytes=8 * GIB, compact_workspace_bytes=4 * GIB,
                slack_fraction=0.10, completed_jobs=completed_jobs,
                completed_initials=completed_initials,
                preexisting_deletion_credit_bytes=0, new_feedback_bytes=0,
                retention='Keep initial/event2/event4; retire new event1/3 tensors only after a verified successor')


def admit(root, *, completed_jobs=0, completed_initials=0, receipt='STORAGE_ADMISSION.json'):
    root = Path(root)
    capacity = future_capacity(completed_jobs, completed_initials)
    memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    row = dict(capacity=capacity, free_bytes=shutil.disk_usage(root).free,
               host_available_bytes=int(memory['MemAvailable'].split()[0]) * 1024,
               host_required_bytes=80 * GIB)
    row['admitted'] = (row['free_bytes'] >= capacity['peak_required_bytes']
                       and row['host_available_bytes'] >= row['host_required_bytes'])
    write_json(root / receipt, row)
    if not row['admitted']:
        raise RuntimeError('Remaining O_HALF capacity is not admitted')
    return row


def retirement_path(root, seed, event):
    return Path(root) / 'campaign_001' / 'retention' / f'{seed}_event_{event:03d}.json'


def retire_odd_boundary(root, seed, successor_event):
    """Call only after completed_job has validated both scientific boundaries."""
    if successor_event not in (2, 4):
        return None
    root = Path(root).resolve()
    event = successor_event - 1
    receipt_path = retirement_path(root, seed, event)
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt['status'] != 'completed':
            raise ValueError('Interrupted retirement requires explicit investigation')
        return receipt
    base = root / 'campaign_001' / f'seed_{seed}'
    prior = base / 'O_HALF' / 'events' / f'event_{event:03d}'
    successor = base / 'O_HALF' / 'events' / f'event_{successor_event:03d}'
    latest = base / 'O_HALF' / 'training_state.pt'
    if not os.path.samefile(latest, successor / 'training_state.pt'):
        raise ValueError('Latest is not the validated successor')
    saved = json.loads((base / f'resume_{event:03d}_O_HALF' / 'STATE_IDENTITIES.json').read_text())
    candidates = [prior / 'training_state.pt', prior / 'export' / 'model.safetensors']
    files = {}
    for path in candidates:
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError('Retirement escaped this new run')
        if str(path) not in saved or identity(path) != saved[str(path)]:
            raise ValueError('Superseded tensor identity differs')
        files[str(path)] = saved[str(path)]
    proof = base / f'resume_{successor_event:03d}_O_HALF' / 'SAVED_ENDPOINT.json'
    if json.loads(proof.read_text())['status'] != 'passed':
        raise ValueError('Successor is not verified')
    receipt = dict(status='prepared', seed=seed, retired_event=event,
                   successor_event=successor_event, files=files,
                   successor_proof={str(proof): identity(proof)},
                   before_free_bytes=shutil.disk_usage(root).free,
                   reason='Verified own successor; sealed rolling odd-event retention',
                   lost_capability='Exact odd-event optimizer resume and serving export; not archived',
                   retained='All receipts, RNG/anchor bindings, initial/event2/event4 full states and exports')
    write_json(receipt_path, receipt)
    for path in candidates:
        path.unlink()
    receipt.update(status='completed', after_free_bytes=shutil.disk_usage(root).free)
    write_json(receipt_path, receipt)
    return receipt


def validate_retired(root, seed, event, path, expected):
    receipt = json.loads(retirement_path(root, seed, event).read_text())
    if (event not in (1, 3) or receipt['status'] != 'completed'
            or receipt['seed'] != seed or receipt['retired_event'] != event
            or receipt['successor_event'] != event + 1
            or receipt['files'].get(str(path)) != expected or Path(path).exists()):
        raise ValueError('Missing tensor has no valid declared retirement')
    for proof, value in receipt['successor_proof'].items():
        if identity(proof) != value:
            raise ValueError('Retirement successor receipt changed')
    return receipt
