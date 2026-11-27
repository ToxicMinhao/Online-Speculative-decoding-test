"""Lossless completed-event storage; verify a replacement before retiring copies.

Only new CODE event JSON/JSONL and explicitly bound native request records are
eligible. Weights, recovery states, export metadata, shared sources and caches
remain outside this path. An archive/index preserves the exact original member
bytes; historical paths are resolved explicitly through ``read_member``.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time

from online_feedback.storage import write_json
from .contract import ARMS, digest
from . import protected


def _hash(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def _inside(path, root):
    path = Path(path).absolute()
    if path != path.resolve() or not path.is_relative_to(root) or not path.is_file():
        raise ValueError('Compact member must be a regular, unsymlinked study file: ' + str(path))
    if path.suffix not in ('.json', '.jsonl'):
        raise ValueError('Only compact JSON/JSONL evidence may be archived here')
    return path


def _identity(path):
    stat = path.stat()
    if stat.st_nlink != 1:
        raise ValueError('Retire only an exclusive redundant loose copy')
    return dict(bytes=stat.st_size, sha256=_hash(path), allocated_bytes=stat.st_blocks * 512,
                device=stat.st_dev, inode=stat.st_ino, mtime_ns=stat.st_mtime_ns)


def _same_file(actual, expected):
    # Delayed allocation/preallocation can change st_blocks with identical
    # contents. Allocation is cost evidence, not immutable file identity.
    return all(actual[key] == expected[key]
               for key in ('bytes', 'sha256', 'device', 'inode', 'mtime_ns'))


def _index_path(archive):
    return Path(str(archive) + '.index.json')


def _scan(archive, index, requested=None):
    if index['binding_digest'] != digest({k:v for k,v in index.items() if k != 'binding_digest'}):
        raise ValueError('Compact index binding changed')
    if Path(archive).stat().st_size != index['archive_bytes'] or _hash(archive) != index['archive_sha256']:
        raise ValueError('Compact archive identity changed')
    expected, seen, selected = index['members'], set(), None
    process = subprocess.Popen(['zstd', '-q', '-d', '-c', str(archive)],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        # ignore_zeros continues to EOF, so an appended second tar stream cannot
        # hide behind the first archive's end markers.
        with tarfile.open(fileobj=process.stdout, mode='r|', ignore_zeros=True) as stream:
            for member in stream:
                name = member.name
                if not member.isfile() or name not in expected or name in seen:
                    raise ValueError('Unexpected, duplicate or nonregular compact member: ' + name)
                if member.size != expected[name]['bytes']:
                    raise ValueError('Compact member size changed: ' + name)
                seen.add(name)
                checksum, pieces = hashlib.sha256(), []
                with stream.extractfile(member) as payload:
                    for chunk in iter(lambda: payload.read(1024 * 1024), b''):
                        checksum.update(chunk)
                        if name == requested:
                            pieces.append(chunk)
                if checksum.hexdigest() != expected[name]['sha256']:
                    raise ValueError('Compact member bytes changed: ' + name)
                if name == requested:
                    selected = b''.join(pieces)
        process.stdout.close()
        error = process.stderr.read().decode('utf-8', 'replace')
        if process.wait() != 0:
            raise RuntimeError('Compact decompression failed: ' + error)
    finally:
        process.stdout.close()
        process.stderr.close()
        if process.poll() is None:
            process.terminate()
            process.wait()
    if seen != set(expected):
        raise ValueError('Compact archive is missing expected members')
    if requested is not None and selected is None:
        raise KeyError('Original path is not in this compact archive: ' + requested)
    return selected


def verify_archive(archive, index_path=None):
    archive = Path(archive)
    index = json.loads(Path(index_path or _index_path(archive)).read_text())
    _scan(archive, index)
    return dict(status='passed', regular_members=len(index['members']),
                logical_bytes=sum(row['bytes'] for row in index['members'].values()),
                archive_bytes=index['archive_bytes'], archive_sha256=index['archive_sha256'])


def read_member(archive, relative_path, index_path=None):
    """Read exact original bytes while verifying the full replacement archive."""
    index = json.loads(Path(index_path or _index_path(archive)).read_text())
    return _scan(Path(archive), index, requested=str(relative_path))


def archive_completed_event(study_root, arm, event, records, *, zstd='zstd'):
    """Package one published event and retire only its verified redundant JSONs.

    The caller invokes this after publication, tensor retirement and boundary
    receipts have finished. No retry overwrites an earlier archive or plan.
    """
    root = Path(study_root).resolve()
    if arm not in ARMS or type(event) is not int or not 1 <= event <= 58:
        raise ValueError('Only a prescribed completed CODE event may be compacted')
    base = root / 'campaign_001' / 'arms' / arm
    boundary = json.loads((base / 'boundaries' / f'event_{event:03d}.json').read_text())
    status = json.loads((base / 'STATUS.json').read_text())
    if (boundary['status'] != 'passed' or boundary['event'] != event
            or status['events_completed'] < event or status['phase'] != 'idle_boundary'):
        raise ValueError('Compact packaging requires a completed, published idle boundary')
    event_root = base / 'events' / f'event_{event:03d}'
    trainer_root = base / 'trainer' / 'events' / f'event_{event:03d}'
    if len(records) != 128 or len({r['record_id'] for r in records}) != 128:
        raise ValueError('Bind exactly the completed event\'s 128 feedback records')
    saved_buffer = json.loads((event_root / 'loop' / 'feedback_buffer' / 'state.json').read_text())
    if saved_buffer['records'] != records:
        raise ValueError('Native trace selection differs from the committed event records')
    candidates = set()
    for folder in (event_root, trainer_root):
        if folder.resolve() != folder or not folder.is_dir():
            raise ValueError('Completed event folder must be an owned real directory')
        for path in folder.rglob('*'):
            if path.suffix in ('.json', '.jsonl') and 'export' not in path.relative_to(folder).parts:
                candidates.add(_inside(path, root))
    for record in records:
        path = _inside(record['metadata']['native_trace_path'], root)
        if not path.is_relative_to(base / 'native'):
            raise ValueError('Native request evidence must belong to the same CODE arm')
        candidates.add(path)
    paths = sorted(candidates)
    protected.require_retirement_paths(root,paths)
    members = {str(path.relative_to(root)): _identity(path) for path in paths}
    destination = root / 'compact' / arm
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / f'event_{event:03d}.tar.zst'
    plan_path = Path(str(archive) + '.plan.json')
    index_path = _index_path(archive)
    receipt_path = Path(str(archive) + '.retirement.json')
    temporary = Path(str(archive) + '.tmp')
    if any(path.exists() for path in (archive, temporary, plan_path, index_path, receipt_path)):
        raise FileExistsError('Compact packaging was already attempted; preserve its evidence')
    plan = dict(study_root=str(root), arm=arm, event=event, members=members,
                retained_tensor_capability='Recovery states and export metadata remain loose',
                original_member_bytes_preserved=True, codec='zstd level 3 tar stream')
    write_json(plan_path, plan)
    result = dict(status='started', archive_path=str(archive), index_path=str(index_path),
                  removed_paths=[], started_epoch=time.time(), removed_allocated_bytes=0,
                  lost_scientific_capabilities=[], loose_read_requires_archive_resolver=True)
    write_json(receipt_path, result)
    try:
        with temporary.open('xb') as output:
            process = subprocess.Popen([zstd, '-q', '-3', '-c'], stdin=subprocess.PIPE,
                                       stdout=output, stderr=subprocess.PIPE)
            try:
                with tarfile.open(fileobj=process.stdin, mode='w|', format=tarfile.PAX_FORMAT) as stream:
                    for path in paths:
                        name = str(path.relative_to(root))
                        stream.add(path, arcname=name, recursive=False)
                process.stdin.close()
                error = process.stderr.read().decode('utf-8', 'replace')
                if process.wait() != 0:
                    raise RuntimeError('Compact compression failed: ' + error)
            finally:
                process.stdin.close()
                process.stderr.close()
                if process.poll() is None:
                    process.terminate()
                    process.wait()
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(archive)
        index = dict(plan, archive_bytes=archive.stat().st_size, archive_sha256=_hash(archive))
        index['binding_digest'] = digest(index)
        write_json(index_path, index)
        result['verification'] = verify_archive(archive, index_path)
        result['status'] = 'verified_before_retirement'
        write_json(receipt_path, result)
        for path in paths:
            name = str(path.relative_to(root))
            actual = _identity(_inside(path, root))
            if not _same_file(actual, members[name]):
                raise ValueError('Loose member changed after archive verification: ' + name)
            actual['allocated_bytes'] = path.stat().st_blocks * 512
            path.unlink()
            result['removed_paths'].append(name)
            result['removed_allocated_bytes'] += actual['allocated_bytes']
            if actual['allocated_bytes'] != members[name]['allocated_bytes']:
                result.setdefault('allocation_changes', {})[name] = dict(
                    planned_bytes=members[name]['allocated_bytes'],
                    before_unlink_bytes=actual['allocated_bytes'])
        result['status'] = 'completed'
    except BaseException as error:
        result.update(status='failed', error=repr(error))
        raise
    finally:
        try:
            result['protected_dependencies']=protected.check(root,purpose='compact_retirement:'+str(archive))
        except BaseException as error:
            result.update(status='failed',protected_dependency_error=repr(error))
            raise
        finally:
            result['wall_s'] = time.time() - result['started_epoch']
            write_json(receipt_path, result)
    return result


def resume_verified_retirement(study_root, arm, event):
    """Finish one failed, verified archive retirement without model work.

    Original failure bytes survive in an exclusive backup. This repair has its
    own journal and cannot be automatically repeated after an interruption.
    """
    root = Path(study_root).resolve()
    if arm not in ARMS or type(event) is not int or not 1 <= event <= 58:
        raise ValueError('Require a prescribed completed event')
    archive = root / 'compact' / arm / f'event_{event:03d}.tar.zst'
    receipt_path = Path(str(archive) + '.retirement.json')
    failed_path = Path(str(archive) + '.retirement.failed_001.json')
    recovery_path = Path(str(archive) + '.retirement.recovery_001.json')
    if failed_path.exists() or recovery_path.exists():
        raise FileExistsError('Retirement recovery already attempted; preserve its evidence')
    failed_bytes = receipt_path.read_bytes()
    previous = json.loads(failed_bytes)
    if (previous['status'] != 'failed' or previous.get('verification', {}).get('status') != 'passed'
            or 'Loose member changed after archive verification:' not in previous.get('error', '')):
        raise ValueError('Only the identified verified partial-retirement failure may resume')
    with failed_path.open('xb') as stream:
        stream.write(failed_bytes); stream.flush(); os.fsync(stream.fileno())
    recovery = dict(status='started', started_epoch=time.time(), new_removed_paths=[],
                    new_removed_allocated_bytes=0, allocation_changes={}, new_model_work=0,
                    failed_receipt=dict(path=str(failed_path), bytes=len(failed_bytes),
                        sha256=hashlib.sha256(failed_bytes).hexdigest()))
    write_json(recovery_path, recovery)
    try:
        plan = json.loads(Path(str(archive) + '.plan.json').read_text())
        index = json.loads(_index_path(archive).read_text())
        if (plan['study_root'] != str(root) or plan['arm'] != arm or plan['event'] != event
                or any(index.get(key) != value for key, value in plan.items())
                or previous['archive_path'] != str(archive)
                or previous['index_path'] != str(_index_path(archive))):
            raise ValueError('Original archive/plan/retirement binding changed')
        recovery['verification'] = verify_archive(archive)
        if recovery['verification'] != previous['verification']:
            raise ValueError('Original verified archive receipt changed')
        base = root / 'campaign_001' / 'arms' / arm
        boundary = json.loads((base/'boundaries'/f'event_{event:03d}.json').read_text())
        status = json.loads((base/'STATUS.json').read_text())
        if (boundary['status'] != 'passed' or boundary['event'] != event
                or status['events_completed'] != event or status['phase'] != 'idle_boundary'):
            raise ValueError('Recovery requires this exact completed idle boundary')
        members, removed = index['members'], previous['removed_paths']
        if len(removed) != len(set(removed)) or not set(removed) <= set(members):
            raise ValueError('Invalid previous retirement path accounting')
        remaining = []
        for name, expected in members.items():
            path = root / name
            if not path.is_relative_to(base) or '..' in Path(name).parts:
                raise ValueError('Archive member escaped the completed arm')
            if name in removed:
                if path.exists() or path.is_symlink():
                    raise ValueError('Previously retired path unexpectedly exists: ' + name)
            else:
                if not path.exists():
                    raise ValueError('Unrecorded missing loose member: ' + name)
                path = _inside(path, root)
                if not _same_file(_identity(path), expected):
                    raise ValueError('Remaining loose member differs: ' + name)
                remaining.append((name, path, expected))
        protected.require_retirement_paths(root, [path for _, path, _ in remaining])
        recovery['protected_before'] = protected.check(root, purpose='resume_compact_retirement')
        recovery['previous_removed_count'] = len(removed)
        recovery['remaining_count'] = len(remaining)
        write_json(recovery_path, recovery)
        for name, path, expected in remaining:
            actual = _identity(_inside(path, root))
            if not _same_file(actual, expected):
                raise ValueError('Remaining member changed during recovery: ' + name)
            actual['allocated_bytes'] = path.stat().st_blocks * 512
            path.unlink()
            recovery['new_removed_paths'].append(name)
            recovery['new_removed_allocated_bytes'] += actual['allocated_bytes']
            if actual['allocated_bytes'] != expected['allocated_bytes']:
                recovery['allocation_changes'][name] = dict(planned_bytes=expected['allocated_bytes'],
                    before_unlink_bytes=actual['allocated_bytes'])
            write_json(recovery_path, recovery)
        recovery['protected_after'] = protected.check(root, purpose='resumed_compact_retirement')
        recovery['status'] = 'completed'
    except BaseException as error:
        recovery.update(status='failed', error=repr(error))
        raise
    finally:
        recovery['wall_s'] = time.time() - recovery['started_epoch']
        write_json(recovery_path, recovery)
    result = dict(previous)
    result.pop('error', None)
    result.update(status='completed', removed_paths=removed+recovery['new_removed_paths'],
        removed_allocated_bytes=previous['removed_allocated_bytes']+recovery['new_removed_allocated_bytes'],
        wall_s=previous['wall_s']+recovery['wall_s'], recovery=recovery,
        protected_dependencies=recovery['protected_after'])
    write_json(receipt_path, result)
    return result
