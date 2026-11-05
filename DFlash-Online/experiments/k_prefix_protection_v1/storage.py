"""Conservative additional-write admission; reused pools receive no new-write charge."""
from pathlib import Path
import shutil
from experiments.ok_events_5_8.storage import GIB, WEIGHTS, required_bytes, identity
from online_feedback.storage import write_json

QUALIFICATION_DISK = 40 * GIB
COMPILER_DISK = 24 * GIB


def future_capacity(*, completed_jobs=0, completed_initials=0, qualification_complete=False):
    """Retain every new boundary, including rolling1/3; no deletion credit assumed."""
    if not 0 <= completed_jobs <= 16 or not 0 <= completed_initials <= 4:
        raise ValueError('Invalid completed-work accounting')
    payload = COMPILER_DISK + (0 if qualification_complete else QUALIFICATION_DISK)
    payload += (4-completed_initials) * WEIGHTS
    payload += (16-completed_jobs) * 4 * WEIGHTS
    # Current export workspace overlaps full checkpoint+export. Atomic latest
    # shares an inode with the preceding retained boundary after publication.
    if completed_jobs < 16:
        payload += WEIGHTS
    return {'payload_bytes':payload, 'peak_required_bytes':required_bytes(payload),
            'qualification_bytes':0 if qualification_complete else QUALIFICATION_DISK,
            'compiler_bytes':COMPILER_DISK, 'reserve_bytes':8*GIB,
            'compact_workspace_bytes':4*GIB, 'slack_fraction':.10,
            'retention':'All16 new state/export boundaries and four initial states retained',
            'new_feedback_bytes':0, 'preexisting_deletion_credit_bytes':0,
            'completed_jobs':completed_jobs, 'completed_initials':completed_initials}


def admit(root, *, completed_jobs=0, completed_initials=0, qualification_complete=False,
          receipt='STORAGE_ADMISSION.json'):
    capacity = future_capacity(completed_jobs=completed_jobs, completed_initials=completed_initials,
                               qualification_complete=qualification_complete)
    row = {'capacity':capacity, 'free_bytes':shutil.disk_usage(root).free}
    mem = dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
    row['host_available_bytes'] = int(mem['MemAvailable'].split()[0])*1024
    row['host_required_bytes'] = 80*GIB
    row['admitted'] = (row['free_bytes'] >= capacity['peak_required_bytes']
                       and row['host_available_bytes'] >= row['host_required_bytes'])
    write_json(Path(root)/receipt, row)
    if not row['admitted']:
        raise RuntimeError('Full remaining-work disk/host capacity not admitted')
    return row
