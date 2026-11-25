"""Phase-specific production admission on measured, independent filesystems."""

import json
import os
from pathlib import Path
import shutil
import stat
import subprocess

from .contract import LIMITS
from .resources import capacity


def existing_parent(path):
    path = Path(path).resolve()
    while not path.exists():
        path = path.parent
    return path


def filesystem(path):
    """Describe actual backing storage; directory names do not create capacity."""
    parent = existing_parent(path)
    row = json.loads(
        subprocess.check_output(
            [
                'findmnt',
                '--json',
                '--target',
                str(parent),
                '--output',
                'SOURCE,TARGET,FSTYPE,OPTIONS,MAJ:MIN',
            ],
            text=True,
        )
    )['filesystems'][0]
    usage = shutil.disk_usage(parent)
    return dict(
        device=os.stat(parent).st_dev,
        capacity_key=row['maj:min'],
        source=row['source'],
        mount=row['target'],
        fstype=row['fstype'],
        readonly='ro' in row['options'].split(','),
        durable=row['fstype'] not in ('tmpfs', 'ramfs', 'overlay'),
        total_bytes=usage.total,
        free_bytes=usage.free,
    )


def allocated_by_device(roots):
    """Count actual blocks once, including nested roots and hard-linked files."""
    seen, totals = set(), {}
    pending = [Path(root) for root in roots if Path(root).exists()]
    while pending:
        path = pending.pop()
        try:
            entry = path.lstat()
        except FileNotFoundError:
            # Atomic publication and verified retirement may remove a queued
            # path while the watchdog scans; its blocks are no longer charged.
            continue
        key = (entry.st_dev, entry.st_ino)
        if key in seen:
            continue
        seen.add(key)
        totals[entry.st_dev] = totals.get(entry.st_dev, 0) + entry.st_blocks * 512
        if stat.S_ISDIR(entry.st_mode):
            try:
                pending.extend(path.iterdir())
            except FileNotFoundError:
                continue
    return totals


def phase_peaks(requests, strategy):
    bound = capacity(requests)
    feature = bound['active_feature_bound_bytes']
    witness = bound['retained_witness_bound_bytes']
    overlap = bound['replacement_overlap_bytes']
    recovery, weights = bound['recovery_bound_bytes'], bound['export_bytes']
    main = [
        0,
        0,
        0,
        recovery,
        recovery + weights,
        recovery + weights,
        recovery + weights,
        2 * recovery + weights,
        2 * recovery + 2 * weights,
    ]
    pools = [
        feature,
        feature + overlap,
        feature,
        feature,
        feature,
        2 * feature,
        2 * feature + overlap,
        2 * feature,
        2 * feature,
    ]
    names = [
        'BASE_capture',
        'BASE_prepare',
        'witness',
        'BASE_state',
        'BASE_export',
        'PAIR_capture',
        'PAIR_prepare',
        'PAIR_state',
        'PAIR_export',
    ]
    if strategy == 'local_single':
        rows = [
            dict(phase=name, main_bytes=a + b) for name, a, b in zip(names, main, pools)
        ]
    elif strategy == 'local_split':
        rows = [
            dict(
                phase=name,
                main_bytes=a + (witness if index >= 2 else 0),
                features_bytes=b,
            )
            for index, (name, a, b) in enumerate(zip(names, main, pools))
        ]
    else:
        raise ValueError('Only actual local_single/local_split storage is implemented')
    peaks = {
        'main': (max(row['main_bytes'] for row in rows) * 105 + 99) // 100
        + 4_000_000_000
    }
    if strategy == 'local_split':
        peaks['features'] = (
            max(row['features_bytes'] for row in rows) * 105 + 99
        ) // 100
    return dict(
        phases=rows,
        peaks=peaks,
        tensor_margin_fraction=0.05,
        source_logs_compiler_allowance_bytes=4_000_000_000,
        recovery_bound_bytes=recovery,
        export_bytes=weights,
        feature_pool_bound_bytes=feature,
        witness_copy_bound_bytes=witness,
        preparation_overlap_bytes=overlap,
    )


def prepare_storage(
    requests,
    *,
    root,
    source_root,
    feature_roots,
    strategy='local_single',
    extra_reserve_bytes=2_000_000_000,
):
    """Return an evidence-bearing blocked plan instead of launching on deficit."""
    components = phase_peaks(requests, strategy)
    main_roots = [str(Path(root).resolve()), str(Path(source_root).parent.resolve())]
    features = [str(Path(feature_roots[arm]).resolve()) for arm in ('BASE', 'PAIR')]
    groups = [('main', main_roots, LIMITS['free_reserve_bytes'])]
    if strategy == 'local_single':
        groups[0][1].extend(features)
    else:
        groups.append(('features', features, extra_reserve_bytes))
    blockers, measured, capacity_keys = [], [], set()
    all_blocks = allocated_by_device(main_roots + features)
    for name, roots, reserve in groups:
        identities = [filesystem(path) for path in roots]
        first = identities[0]
        if any(
            row['capacity_key'] != first['capacity_key']
            or row['device'] != first['device']
            for row in identities
        ):
            blockers.append(name + ': allocation roots span unbudgeted filesystems')
        if first['capacity_key'] in capacity_keys:
            blockers.append(name + ': aliases an already budgeted physical filesystem')
        capacity_keys.add(first['capacity_key'])
        if any(row['readonly'] or not row['durable'] for row in identities):
            blockers.append(name + ': requires writable durable backing storage')
        allocated = all_blocks.get(first['device'], 0)
        peak = components['peaks'][name]
        remaining = max(0, peak - allocated)
        free = min(row['free_bytes'] for row in identities)
        deficit = max(0, reserve + remaining - free)
        if deficit:
            blockers.append(f'{name}: projected peak misses reserve by {deficit} bytes')
        measured.append(
            dict(
                name=name,
                allocation_roots=roots,
                filesystem=first,
                projected_peak_bytes=peak,
                reserve_bytes=reserve,
                free_bytes=free,
                allocated_bytes=allocated,
                remaining_peak_bytes=remaining,
                deficit_bytes=deficit,
            )
        )
    total = sum(group['projected_peak_bytes'] for group in measured)
    if total > LIMITS['new_allocated_bytes']:
        blockers.append('Combined filesystem peak exceeds 90 GB')
    return dict(
        admitted=not blockers,
        strategy=strategy,
        groups=measured,
        projected_peak_bytes=total,
        components=components,
        blockers=blockers,
    )


def check_storage(plan, compiler_growth):
    """Recheck every sealed capacity group before writes and during execution."""
    storage = plan['storage']
    if not storage['admitted']:
        raise RuntimeError(
            'Production storage was not admitted: ' + repr(storage['blockers'])
        )
    main = next(group for group in storage['groups'] if group['name'] == 'main')
    for path in plan['compiler_baseline']['directories'].values():
        if filesystem(path)['capacity_key'] != main['filesystem']['capacity_key']:
            raise RuntimeError(
                'Compiler cache escaped its sealed main-filesystem budget'
            )
    roots = [path for group in storage['groups'] for path in group['allocation_roots']]
    allocated = allocated_by_device(roots)
    groups, charged = [], 0
    for group in storage['groups']:
        original = group['filesystem']
        if any(str(Path(path).resolve()) != path for path in group['allocation_roots']):
            raise RuntimeError('Sealed allocation root was redirected')
        current = [filesystem(path) for path in group['allocation_roots']]
        if any(
            any(
                row[key] != original[key]
                for key in (
                    'device',
                    'capacity_key',
                    'source',
                    'fstype',
                    'readonly',
                    'durable',
                )
            )
            for row in current
        ):
            raise RuntimeError('Sealed filesystem changed: ' + group['name'])
        blocks = allocated.pop(original['device'], 0)
        # The approved compiler caches are on the original /data filesystem.
        growth = compiler_growth if group['name'] == 'main' else 0
        amount = blocks + growth
        remaining = max(0, group['projected_peak_bytes'] - amount)
        free = min(row['free_bytes'] for row in current)
        row = dict(
            name=group['name'],
            allocated_bytes=blocks,
            compiler_growth_bytes=growth,
            free_bytes=free,
            remaining_peak_bytes=remaining,
            reserve_bytes=group['reserve_bytes'],
        )
        groups.append(row)
        charged += amount
        if amount > group['projected_peak_bytes']:
            raise RuntimeError('Filesystem exceeded admitted peak: ' + repr(row))
        if free - remaining < group['reserve_bytes']:
            raise RuntimeError(
                'Filesystem projected peak crosses reserve: ' + repr(row)
            )
    if allocated:
        raise RuntimeError('New allocation appeared on an unbudgeted filesystem')
    if charged > LIMITS['new_allocated_bytes']:
        raise RuntimeError('Combined allocation exceeds 90 GB')
    return dict(
        groups=groups, allocated_bytes=charged, compiler_growth_bytes=compiler_growth
    )
