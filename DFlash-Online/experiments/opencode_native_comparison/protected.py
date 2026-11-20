"""Recheck read-only dependencies after retiring this study's redundant files.

The CPU preflight authenticates complete bytes. Routine retirement checks compare
resolved path and stat identity; the final check hashes complete bytes again.
Stat checks are cheap corruption/deletion guards, not full byte authentication.
"""
from pathlib import Path
import hashlib
import json
import time

from online_feedback.storage import write_json
from experiments.kw_v1.evaluate import append_json
from experiments.ok_coding.storage import identity
from .contract import digest

NAME='PROTECTED_DEPENDENCIES.json'


def _stat(path):
    path=Path(path)
    resolved=path.resolve(strict=True)
    stat=resolved.stat()
    if not resolved.is_file():
        raise ValueError('A protected dependency must remain a regular file: '+str(path))
    return dict(resolved_path=str(resolved),device=stat.st_dev,inode=stat.st_ino,
                bytes=stat.st_size,mtime_ns=stat.st_mtime_ns)


def capture_authenticated(plan):
    """Snapshot dependencies only after preflight verified their expected hashes."""
    root=Path(plan['root']).resolve()
    destination=root/NAME
    if destination.exists():
        raise FileExistsError('Protected identities are immutable after initial authentication')
    expected=dict(plan['input_identities'])
    for entry in plan['authenticated_files']:
        path,pin=entry['path'],entry['expected_identity']
        if path in expected and expected[path]!=pin:
            raise ValueError('Conflicting protected dependency identities')
        expected[path]=pin
    source=str(Path(plan['source_manifest']))
    expected[source]=identity(source)
    files={}
    for path,pin in sorted(expected.items()):
        stat=_stat(path)
        if stat['bytes']!=pin['bytes'] or len(pin['sha256'])!=64:
            raise ValueError('Protected dependency differs from its preflight identity: '+path)
        files[path]=dict(stat,expected_sha256=pin['sha256'])
    if not files:
        raise ValueError('Preflight must identify the protected dependencies')
    receipt=dict(study_root=str(root),execution_commit=plan['execution_commit'],
        plan_binding=plan['binding'],files=files,
        initial_hash_authentication='complete CPU preflight; source manifest also authenticated here',
        routine_check='resolved path,device,inode,size,mtime; not a full byte rehash',
        final_check='complete expected SHA256 hashes')
    receipt['binding_digest']=digest(receipt)
    write_json(destination,receipt)
    return dict(path=str(destination),files=len(files),status='captured_after_preflight')


def _load(root):
    root=Path(root).resolve()
    receipt=json.loads((root/NAME).read_text())
    if (receipt['study_root']!=str(root)
            or receipt['binding_digest']!=digest({k:v for k,v in receipt.items() if k!='binding_digest'})):
        raise ValueError('Protected dependency receipt changed or belongs to a different study')
    return root,receipt


def check(root, *, full_hash=False, purpose='retirement'):
    """A missing receipt is an execution failure, never permission to skip checks."""
    root,receipt=_load(root)
    started=time.perf_counter()
    result=dict(status='running',purpose=purpose,files=len(receipt['files']),
        mode='full_content_hash_and_stat' if full_hash else 'stat_identity_only',
        epoch=time.time())
    try:
        for path,expected in receipt['files'].items():
            current=_stat(path)
            if current!={key:expected[key] for key in current}:
                raise ValueError('Protected dependency stat identity changed: '+path)
            if full_hash:
                checksum=hashlib.sha256()
                with Path(path).open('rb') as stream:
                    for block in iter(lambda:stream.read(1024*1024),b''):
                        checksum.update(block)
                if checksum.hexdigest()!=expected['expected_sha256'] or _stat(path)!=current:
                    raise ValueError('Protected dependency content changed: '+path)
        result['status']='passed'
    except BaseException as error:
        result.update(status='failed',error=repr(error))
        raise
    finally:
        result['wall_s']=time.perf_counter()-started
        append_json(root/'PROTECTED_RECHECKS.jsonl',result)
    return result


def require_retirement_paths(root, paths):
    """Deletion candidates must be new, regular study files and never keepers."""
    root,receipt=_load(root)
    protected={Path(row['resolved_path']) for row in receipt['files'].values()}
    protected_inodes={(row['device'],row['inode']) for row in receipt['files'].values()}
    protected.add(root/NAME)
    for value in paths:
        path=Path(value).absolute()
        resolved=path.resolve(strict=True)
        stat=resolved.stat()
        if (path!=resolved or not resolved.is_relative_to(root) or not resolved.is_file()
                or resolved in protected or (stat.st_dev,stat.st_ino) in protected_inodes):
            raise ValueError('Retirement would leave its new study scope or remove a protected dependency: '+str(path))
    return dict(status='passed',study_root=str(root),protected_files=len(protected))
