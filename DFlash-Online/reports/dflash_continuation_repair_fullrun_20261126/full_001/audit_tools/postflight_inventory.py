#!/usr/bin/env python3
"""Read-only PR66 postflight. No tensor loading, model import, SSH or deletion.

Run on the execution host only after the campaign has stopped. Hashes are
streamed; archives are inspected without extraction. Peer-location allocation
must be measured separately and supplied, never guessed from remote free space.
Receipts record a point-in-time check, not continuous monitoring or a new
scientific/serving validation. Files with paths remain private.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import socket
import stat
import subprocess
import tarfile
import time

ARMS = ('BASE', 'TAIL')
AMENDMENT_FILES = [
    'experiments/continuation_repair_fullrun/'+name
    for name in ('compact.py','entry.py','launch.py','source_amendment.py')]

LIMITS = dict(elapsed_s=259200, occupied_card_s=432000,
              new_allocated_bytes=90_000_000_000, free_reserve_bytes=400_000_000_000)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


class Incomplete(Exception):
    """Missing evidence or exceeded read-only budget; never an inferred pass."""


def require(condition, message):
    if not condition:
        raise ValueError(message)


def compare_sources(original, repaired):
    changed = sorted(name for name in set(original['files']) | set(repaired['files'])
                     if original['files'].get(name)!=repaired['files'].get(name))
    def documentary(name):
        path = Path(name)
        return path.parts[0] in ('tests','reports') or path.suffix.lower()=='.md'
    production = [name for name in changed if not documentary(name)]
    require(set(production)<=set(AMENDMENT_FILES) and AMENDMENT_FILES[0] in production
            and AMENDMENT_FILES[-1] in repaired['files'], 'Unapproved production source change')
    return dict(changed_production_paths=production,
                changed_diagnostic_or_documentation_paths=[name for name in changed if documentary(name)],
                unchanged_production_python_files=sum(name.endswith('.py') and not documentary(name)
                    and name not in production for name in original['files']))


class Budget:
    def __init__(self, *, max_hash_bytes=1_000_000_000_000, max_entries=500_000,
                 max_seconds=7200):
        self.max_hash_bytes, self.max_entries, self.max_seconds = max_hash_bytes, max_entries, max_seconds
        self.hash_bytes = self.entries = 0
        self.started = time.monotonic()

    def check(self):
        if (self.hash_bytes > self.max_hash_bytes or self.entries > self.max_entries
                or time.monotonic()-self.started > self.max_seconds):
            raise Incomplete('Read-only audit budget exhausted; partial evidence preserved')

    def hash(self, path):
        path = Path(path)
        before = path.stat()
        require(path.is_file(), 'Expected file: '+str(path))
        stamp = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        self.check()
        if self.hash_bytes+before.st_size > self.max_hash_bytes:
            raise Incomplete('Required file would exceed streamed-hash budget')
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(4*1024*1024), b''):
                h.update(chunk)
                self.hash_bytes += len(chunk)
                self.check()
        after = path.stat()
        require(stamp == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
                'File changed during hashing: '+str(path))
        return dict(bytes=after.st_size, sha256=h.hexdigest())


def allocation(roots, budget):
    """st_blocks once per inode; symlink payload counted, never followed."""
    seen, total, entries = set(), 0, 0
    for value in roots:
        root = Path(value)
        if not root.exists():
            raise Incomplete('Allocation root absent: '+str(root))
        require(not root.is_symlink(), 'Allocation root cannot be a symlink')
        pending = [root]
        while pending:
            path = pending.pop()
            info = path.lstat()
            key = (info.st_dev, info.st_ino)
            if key in seen:
                continue
            seen.add(key)
            total += info.st_blocks*512
            entries += 1
            budget.entries += 1
            budget.check()
            if stat.S_ISDIR(info.st_mode):
                pending.extend(path.iterdir())
    return dict(allocated_bytes=total, entries=entries, roots=list(map(str, roots)),
                policy='st_blocks*512 once per device/inode; no symlink traversal')


class Postflight:
    def __init__(self, plan, *, budget=None):
        self.plan = plan
        self.root = Path(plan['root'])
        self.budget = budget or Budget()
        self.inputs, self.present, self.absent = {}, {}, set()
        self.private, self.sections, self.errors = {}, {}, []
        self.member_identities = {}
        self.archive_metadata = {}

    def json(self, path):
        path = Path(path)
        self.inputs[str(path)] = self.budget.hash(path)
        return read(path)

    def section(self, name, function):
        try:
            self.sections[name] = dict(status='passed', **function())
        except (FileNotFoundError, Incomplete) as error:
            self.sections[name] = dict(status='incomplete')
            self.errors.append(dict(section=name, status='INCOMPLETE', error=repr(error)))
        except Exception as error:
            self.sections[name] = dict(status='failed')
            self.errors.append(dict(section=name, status='FAILED', error=repr(error)))

    def present_file(self, path, expected, *, role):
        path = Path(path)
        if role in ('execution_source','frozen_objective','final_complete_recovery','final_serving_export','current_recovery_alias','shared_witness_feature'):
            require(not path.is_symlink(), 'Owned/source artifact unexpectedly became a symlink')
        actual = self.budget.hash(path)
        require(actual == {k: expected[k] for k in ('bytes', 'sha256')}, 'Identity mismatch: '+str(path))
        require(str(path) not in self.absent, 'Path is both retained and retired')
        self.present[str(path)] = dict(path=str(path), exists=True, **actual, role=role)

    def absent_file(self, path):
        path = Path(path)
        require(path.is_absolute() and path.is_relative_to(self.root/'campaign_001'),
                'Retirement is outside this campaign')
        require(str(path) not in self.present, 'Retained file listed as retired')
        require(not os.path.lexists(path), 'Retired artifact still exists: '+str(path))
        self.absent.add(str(path))

    def archive(self, arm, event):
        """Authenticate each member once; retain only small retirement metadata."""
        key = arm, event
        if key in self.archive_metadata:
            return self.archive_metadata[key]
        archive = self.root/f'compact/{arm}/event_{event:03d}.tar.zst'
        index = self.json(str(archive)+'.index.json')
        require(index['binding_digest'] == digest({k:v for k,v in index.items() if k!='binding_digest'}),
                'Archive index binding mismatch')
        require(self.budget.hash(archive) == dict(bytes=index['archive_bytes'], sha256=index['archive_sha256']),
                'Archive identity mismatch')
        expected, seen, kept = index['members'], set(), {}
        process = subprocess.Popen([self.plan['zstd'], '-q', '-d', '-c', str(archive)],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            with tarfile.open(fileobj=process.stdout, mode='r|') as stream:
                for member in stream:
                    require(member.isfile() and member.name in expected and member.name not in seen,
                            'Unexpected archive member')
                    require(not Path(member.name).is_absolute() and '..' not in Path(member.name).parts,
                            'Unsafe archive member path')
                    seen.add(member.name)
                    self.budget.entries += 1
                    wanted = ('/preparation_retirement/' in member.name
                              or member.name.endswith(('/RETIREMENT.json', '/OLDER_BOUNDARY_RETIREMENT.json',
                                                       '/FEATURE_RETIREMENT_IDENTITIES.json')))
                    if wanted and member.size > 32*1024*1024:
                        raise Incomplete('Retirement metadata exceeds bounded reader size')
                    h, pieces = hashlib.sha256(), []
                    with stream.extractfile(member) as payload:
                        for chunk in iter(lambda: payload.read(1024*1024), b''):
                            h.update(chunk)
                            self.budget.hash_bytes += len(chunk)
                            self.budget.check()
                            if wanted:
                                pieces.append(chunk)
                    actual = dict(bytes=member.size, sha256=h.hexdigest())
                    require(actual == {k:expected[member.name][k] for k in ('bytes', 'sha256')},
                            'Archive member identity mismatch')
                    self.member_identities[str(self.root/member.name)] = actual
                    if wanted:
                        kept[member.name] = json.loads(b''.join(pieces))
            process.stdout.close()
            error = process.stderr.read().decode()
            require(process.wait(timeout=30) == 0, 'Decompression failed: '+error)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            process.stdout.close()
            process.stderr.close()
        require(seen == set(expected), 'Archive is missing indexed members')
        retirement = self.json(str(archive)+'.retirement.json')
        require(retirement['status']=='completed' and set(retirement['removed_paths'])==seen,
                'Compact archive retirement mismatch')
        if 'recovery' in retirement:
            self.archive_recovery(archive, retirement)
        self.archive_metadata[key] = kept
        return kept

    def archive_recovery(self, archive, final):
        """A repaired receipt never erases or relabels its failed predecessor."""
        recovery = self.json(str(archive)+'.retirement.recovery_001.json')
        require(final['recovery']==recovery and recovery['status']=='completed'
                and recovery['new_model_work']==0, 'Archive recovery proof differs')
        prior_pin = recovery['failed_receipt']
        failed_path = Path(str(archive)+'.retirement.failed_001.json')
        require(Path(prior_pin['path'])==failed_path and self.budget.hash(failed_path)==
                {key:prior_pin[key] for key in ('bytes','sha256')}, 'Original failed receipt bytes changed')
        previous = self.json(failed_path)
        require(previous['status']=='failed' and previous['verification']==recovery['verification']
                and previous['verification']['status']=='passed'
                and 'Loose member changed after archive verification:' in previous.get('error',''),
                'Recovery does not bind the identified packaging failure')
        old, new = previous['removed_paths'], recovery['new_removed_paths']
        require(len(set(old))==len(old) and len(set(new))==len(new) and not set(old)&set(new)
                and final['removed_paths']==old+new
                and recovery['previous_removed_count']==len(old) and recovery['remaining_count']==len(new),
                'Recovery repeats or omits compact retirement')
        require(final['removed_allocated_bytes']==previous['removed_allocated_bytes']+
                recovery['new_removed_allocated_bytes'], 'Recovery allocation accounting differs')
        require(abs(final['wall_s']-previous['wall_s']-recovery['wall_s'])<1e-6,
                'Failed/recovered packaging time was dropped')
        self.private.setdefault('archive_recoveries',[]).append(dict(
            archive=str(archive),failed_receipt_identity=self.budget.hash(failed_path),
            original_removed=len(old),recovered_removed=len(new),new_model_work=0))

    def compact_identity(self, path):
        path = Path(path)
        if path.is_file():
            return self.budget.hash(path)
        if str(path) in self.member_identities:
            return self.member_identities[str(path)]
        raise Incomplete('Compact retirement proof unavailable: '+str(path))

    def retirement(self, receipt):
        require(receipt['status']=='completed' and receipt['files']==receipt['removed'],
                'Tensor retirement is not complete')
        require(len({row['path'] for row in receipt['removed']}) == len(receipt['removed']),
                'Duplicate retired artifact')
        require(bool(receipt['compact_evidence']), 'Retirement lacks compact evidence')
        for path, expected in receipt['compact_evidence'].items():
            require(self.compact_identity(path)==expected, 'Retained retirement evidence differs')
        for row in receipt['removed']:
            self.absent_file(row['path'])
        return Counter(row['kind'] for row in receipt['removed'])

    def verify_source_tree(self, root, source):
        root = Path(root)
        for name, expected in source['files'].items():
            path = root/name
            require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'Unsafe source path')
            if expected.get('kind')=='symlink':
                require(path.is_symlink() and os.readlink(path)==expected['target'], 'Source link changed')
            else:
                self.present_file(path, expected, role='execution_source')
        python_files = {str(path.relative_to(root)) for path in root.rglob('*.py') if path.is_file()}
        require(not python_files-set(source['files']), 'Unmanifested Python source is present')

    def source_amendment(self, original):
        status = self.json(self.root/'STATUS.json')
        actual = status.get('actual_execution_commit',self.plan['execution_commit'])
        if actual==self.plan['execution_commit']:
            require(not status.get('source_amendment_path'), 'Original-only receipt names a repair')
            return dict(actual_execution_commit=actual,source_count=1,source_amendment_identity=None,
                        source_amendment_binding=None,actual_source_identity=self.plan['source_identity'])
        path = Path(status['source_amendment_path'])
        receipt = self.json(path)
        receipt_identity = self.budget.hash(path)
        require(status['source_amendment_identity']==receipt_identity and
                receipt['schema']=='continuation_repair_fullrun_source_amendment_v1' and
                receipt['binding']==digest({k:v for k,v in receipt.items() if k!='binding'}),
                'Source amendment identity/seal differs')
        require(receipt['plan_binding']==self.plan['binding'] and
                receipt['execution_commit']==self.plan['execution_commit'] and
                receipt['actual_execution_commit']==actual and
                Path(receipt['original_plan_path'])==self.root/'sealed/PLAN.json' and
                receipt['original_plan_identity']==self.budget.hash(self.root/'sealed/PLAN.json'),
                'Source amendment does not preserve the original plan')
        require(receipt['approved_production_changes']==AMENDMENT_FILES, 'Infrastructure whitelist changed')
        source_root = Path(receipt['source_root'])
        require(source_root.is_relative_to(self.root/'infrastructure_sources') and
                source_root!=Path(self.plan['source_root']), 'Repaired source is not separately charged')
        repaired = self.json(receipt['source_manifest'])
        require(self.budget.hash(receipt['source_manifest'])==receipt['source_identity'] and
                repaired['commit']==actual and repaired['clean'] is True, 'Repaired source identity differs')
        comparison = compare_sources(original,repaired)
        require(comparison==receipt['source_comparison'], 'Source-difference receipt differs')
        self.verify_source_tree(source_root,repaired)
        proof = receipt['resume_evidence']
        require(self.budget.hash(proof['path'])==proof['identity'], 'CPU recovery proof changed')
        proof_data = self.json(proof['path'])
        require(proof_data['status'] in ('passed','completed') and proof_data['new_model_work']==0,
                'Repair recovery introduced model work')
        resume_files = sorted((self.root/'infrastructure_resumes').glob('resume_*/RESUME.json'))
        if not resume_files:
            raise Incomplete('Repaired execution has no preserved original process receipts')
        preserved = {}
        for resume_path in resume_files:
            resume = self.json(resume_path)
            require(resume['status']=='preserved' and resume['plan_binding']==self.plan['binding']
                    and resume['new_model_work']==0 and resume['source_amendment_identity']==receipt_identity
                    and resume['actual_execution_commit']==actual and resume['source_amendment_binding']==receipt['binding'],
                    'Resume provenance differs')
            for name,expected in resume['prior_files'].items():
                require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'Unsafe preserved receipt path')
                backup = resume_path.parent/'prior'/name
                require(self.budget.hash(backup)==expected, 'Original failed/resume receipt changed')
                preserved.setdefault(name,[]).append(expected)
                if name.startswith(('processes/','phases/')):
                    require((self.root/name).is_file() and self.budget.hash(self.root/name)==expected,
                            'Original process/phase receipt was removed or overwritten')
            prior_clock = self.json(resume_path.parent/'prior/CLOCK.json')
            current_clock = self.json(self.root/'CLOCK.json')
            require(prior_clock['first_model_load_epoch']==resume['original_model_load_epoch']==
                    current_clock['first_model_load_epoch'], 'Resume reset the original clock')
            require(not prior_clock['active'] and all(row in current_clock['finished'] for row in prior_clock['finished']),
                    'Resume dropped consumed card intervals')
            prior_work = self.json(resume_path.parent/'prior/WORK.json')
            current_work = self.json(self.root/'WORK.json')
            require(not prior_work['pending'] and all(current_work['attempted'].get(key,0)>=value
                    for key,value in prior_work['attempted'].items()), 'Resume dropped prior physical work')
        execution_receipts = [self.root/'STATUS.json',*self.root.glob('processes/*.json'),*self.root.glob('phases/*/STATUS.json')]
        for evidence in execution_receipts:
            row = self.json(evidence)
            executed = row.get('actual_execution_commit',self.plan['execution_commit'])
            require(row.get('execution_commit',self.plan['execution_commit'])==self.plan['execution_commit'],
                    'Scientific execution binding changed')
            if executed==self.plan['execution_commit']:
                require(self.budget.hash(evidence) in preserved.get(str(evidence.relative_to(self.root)),[]),
                        'Untagged original receipt lacks pre-resume byte identity')
            else:
                require(executed==actual and row['source_amendment_identity']==receipt_identity
                        and row['source_amendment_binding']==receipt['binding'] and
                        Path(row['actual_source_root'])==source_root, 'Actual phase/source binding differs')
        self.private['source_amendment'] = dict(path=str(path),identity=receipt_identity,comparison=comparison)
        return dict(actual_execution_commit=actual,source_count=2,source_amendment_identity=receipt_identity,
                    source_amendment_binding=receipt['binding'],actual_source_identity=receipt['source_identity'],
                    infrastructure_only_production_changes=comparison['changed_production_paths'],
                    original_plan_unchanged=True,scientific_code_unchanged=True,repaired_source_entries=len(repaired['files']),
                    preserved_resume_receipts=len(resume_files),phase_process_receipts_verified=len(execution_receipts))

    def sources_and_protected(self):
        source = self.json(self.plan['source_manifest'])
        require(source['commit']==self.plan['execution_commit'] and source['clean'] is True,
                'Execution source identity/clean checkout mismatch')
        require(self.budget.hash(self.plan['source_manifest'])==self.plan['source_identity'],
                'Sealed source manifest changed')
        source_root = Path(self.plan['source_root'])
        self.verify_source_tree(source_root,source)
        for name, expected in self.plan['formula_code_identities'].items():
            self.present_file(source_root/name, expected, role='frozen_objective')
        expected = dict(self.plan['input_identities'])
        for row in self.plan['authenticated_files']:
            if row['path'] in expected:
                require(expected[row['path']]==row['expected_identity'], 'Conflicting input seals')
            expected[row['path']] = row['expected_identity']
        for endpoint in self.plan['retained_endpoints'].values():
            expected[str(Path(endpoint['checkpoint'])/'model.safetensors')] = endpoint['checkpoint_identity']
            expected.update(endpoint['companion_files'])
        protected = self.json(self.root/'PROTECTED_INPUTS.json')
        require(protected['status']=='sealed' and len(protected['files'])==len(expected), 'Protected seal differs')
        require({row['path']:{k:row[k] for k in ('bytes','sha256')} for row in protected['files']}==expected,
                'Protected inputs differ from sealed plan')
        for row in protected['files']:
            self.present_file(row['path'], row, role='historical_or_sealed_input')
            require(Path(row['path']).stat().st_mtime_ns==row['mtime_ns'], 'Protected input mtime changed')
        chain = self.source_amendment(source)
        return dict(source_entries=len(source['files']), protected_files=len(protected['files']),
                    full_streamed_hashes=True, tensors_deserialized=False, **chain)

    def retention(self):
        counts, final_states, final_exports = Counter(), 0, 0
        for arm in ARMS:
            base = self.root/'campaign_001/arms'/arm
            status = self.json(base/'STATUS.json')
            if status['status']!='completed':
                raise Incomplete('Arm is not complete')
            require(status['events_completed']==58 and status['global_step']==7424
                    and status['presentations']==14848, 'Completed arm counters differ')
            require(status['binding']==self.plan['binding'] and status['execution_commit']==self.plan['execution_commit'],
                    'Arm provenance mismatch')
            markers = {}
            for event in range(1,59):
                marker = self.json(base/f'boundaries/event_{event:03d}.json')
                require(marker['status']=='passed' and marker['arm']==arm and marker['event']==event,
                        'Unvalidated event boundary')
                proof = marker['scientific_check']
                require(proof['status']=='passed' and proof['global_step']==128*event
                        and proof['presentations']==256*event and proof['export_exact']
                        and proof['live']['exact_resume'], 'Boundary scientific integrity not passed')
                markers[event] = marker
                if event>=57:
                    self.present_file(marker['state'], marker['state_identity'], role='final_complete_recovery')
                    final_states += 1
                if event==58:
                    self.present_file(marker['weights'], marker['weights_identity'], role='final_serving_export')
                    self.present_file(base/'trainer/training_state.pt', marker['state_identity'], role='current_recovery_alias')
                    require(status['checkpoint_identity']==marker['weights_identity'] and
                            Path(status['checkpoint'])/'model.safetensors'==Path(marker['weights']),
                            'Final status points to a different export')
                    final_exports += 1
            for event in range(1,59):
                metadata = self.archive(arm,event)
                prefix = f'campaign_001/arms/{arm}/events/event_{event:03d}/'
                feature = metadata[prefix+'RETIREMENT.json']
                require(len(feature['removed'])==128 and all(row['kind']=='prepared_feature' for row in feature['removed']),
                        'Prepared response retirement count changed')
                counts.update(self.retirement(feature))
                prep = [value for name,value in metadata.items()
                        if name.startswith(prefix+'preparation_retirement/') and name.endswith('.json')
                        and not name.endswith(('.plan.json','.equivalence.json'))]
                require(len(prep)==128, 'Missing incremental response retirement')
                for receipt in prep:
                    require(Counter(row['kind'] for row in receipt['removed'])==Counter(native_capture=1,terminal_feature=1),
                            'Incremental feature retirement differs')
                    counts.update(self.retirement(receipt))
                if event>=3:
                    retired = metadata[prefix+'OLDER_BOUNDARY_RETIREMENT.json']
                    require([row['path'] for row in retired['removed']]==[markers[event-2]['state']],
                            'Wrong rolling recovery retired')
                    require(retired['retained']['current']==markers[event] and
                            retired['retained']['previous']==markers[event-1], 'Missing two-successor proof')
                    counts.update(self.retirement(retired))
                if event<=57:
                    export = self.json(base/f'boundaries/event_{event:03d}_EXPORT_RETIREMENT.json')
                    fingerprint = self.json(base/f'boundaries/event_{event:03d}_TENSOR_FINGERPRINTS.json')
                    require([row['path'] for row in export['removed']]==[markers[event]['weights']],
                            'Wrong duplicate export retired')
                    require(fingerprint['status']=='passed' and fingerprint['exact_export_tensors'] is True
                            and fingerprint['identity']==markers[event]['state_identity']
                            and fingerprint['export_identity']==markers[event]['weights_identity'],
                            'Retired export lacks exact recovery tensor proof')
                    counts.update(self.retirement(export))
        witness = self.json(self.root/'witness/BINDINGS.json')
        require(witness['status']=='sealed' and len(witness['features'])==8, 'Shared witness binding missing')
        for row in witness['features']:
            self.present_file(row['feature_path'],row['feature_identity'],role='shared_witness_feature')
        disposable = self.json(self.root/'qualification/DISPOSABLE_RETIREMENT.json')
        # This receipt has its own explicit identity rows, not native feature-retirement schema.
        require(disposable['status']=='completed' and disposable['retired']==disposable['candidates']
                and {row['arm'] for row in disposable['retired']}==set(ARMS),
                'Disposable qualification retention unfinished')
        for row in disposable['retired']:
            path = Path(row['path'])
            require(path.is_relative_to(self.root/'qualification'), 'Disposable retirement outside qualification')
            require(not os.path.lexists(path), 'Retired qualification export remains')
            case = self.json(row['retained_receipt'])
            require(case['status']=='completed' and case['restoration_exact']
                    and case['export']['identity']==row['identity'], 'Disposable compact proof differs')
            serialization = self.json(Path(row['retained_receipt']).parent/'SERIALIZATION.json')
            require(serialization['retired_after_exact_readback'] is True
                    and not os.path.lexists(Path(row['retained_receipt']).parent/'disposable_state.pt'),
                    'Disposable complete state retirement not established')
        self.private['qualification_disposable_retirement'] = disposable
        return dict(final_complete_states=final_states, final_serving_exports=final_exports,
                    retained_witness_features=8, retired_by_kind=dict(counts),
                    lost_capability='Retired tensors are deleted, not archived or directly resumable')

    def processes_and_clock(self, *, original_clock_epoch, gpu_probe=None, process_probe=None):
        status, clock, work = [self.json(self.root/name) for name in ('STATUS.json','CLOCK.json','WORK.json')]
        if status['status']!='completed':
            raise Incomplete('Campaign is not complete')
        require(status['execution_commit']==self.plan['execution_commit'], 'Campaign source mismatch')
        require(clock['first_model_load_epoch']==original_clock_epoch, 'Original model-load clock changed')
        require(not clock['active'], 'A physical-card lease remains open')
        if work['pending']:
            raise Incomplete('Unresolved physical work remains pending')
        require(all(work['attempted'].get(key,0)==count for key,count in work['completed'].items())
                and work['attempted']==work['completed'], 'Attempted/completed work differs')
        elapsed = status['finished_epoch']-clock['first_model_load_epoch']
        occupied = sum(row['occupied_s'] for row in clock['finished'])
        require(0<=elapsed<=LIMITS['elapsed_s'] and math.isfinite(elapsed), 'Elapsed ceiling violated')
        require(0<=occupied<=LIMITS['occupied_card_s'] and math.isfinite(occupied)
                and abs(occupied-clock['completed_card_s'])<=1e-6, 'Occupied-card accounting differs')
        leases = {row['task']:row for row in clock['finished']}
        require(len(leases)==len(clock['finished']), 'Duplicate card leases')
        by_gpu = {}
        for row in clock['finished']:
            require(0<=row['occupied_s'] and row['started_epoch']>=original_clock_epoch and
                    row['finished_epoch']>=row['started_epoch'] and
                    abs(row['finished_epoch']-row['started_epoch']-row['occupied_s'])<.1,
                    'Invalid card interval')
            by_gpu.setdefault(row['physical_gpu_uuid'],[]).append(row)
        for rows in by_gpu.values():
            rows.sort(key=lambda row:row['started_epoch'])
            require(all(a['finished_epoch']<=b['started_epoch'] for a,b in zip(rows,rows[1:])),
                    'Overlapping ownership of one physical card')
        processes = list((self.root/'processes').glob('*.json'))
        if not processes:
            raise Incomplete('No process receipts')
        expected_leases = set()
        process_probe = process_probe or pid_state
        process_rows = []
        for path in processes:
            row = self.json(path)
            require(row['termination_confirmed'] and row['finished_epoch'] is not None
                    and (not row.get('process_created',True) or row['exit_code'] is not None),
                    'Unconfirmed worker termination')
            if row.get('process_created',True):
                observed = process_probe(row['pid'])
                process_rows.append(dict(receipt=path.name,pid=row['pid'],**observed))
                if observed['exists']:
                    raise Incomplete('Owned PID is live or reused; inspect rather than assuming termination')
            slots = ('A','B') if row['phase']=='train' else ('B',) if row['phase']=='qualification_training' else ('A',)
            for slot in slots:
                key = path.stem+'_'+slot
                expected_leases.add(key)
                require(key in leases, 'Process has no charged physical-card lease')
                lease = leases[key]
                require(lease['started_epoch']<=row.get('started_epoch',lease['started_epoch'])
                        <=row['finished_epoch']<=lease['finished_epoch']+.01, 'Process escaped charged interval')
        require(expected_leases==set(leases), 'Unmatched occupied-card lease')
        observed = (gpu_probe or gpu_state)()
        expected_gpus = set(self.plan['devices'].values())
        require(set(observed['gpu_uuids'])==expected_gpus, 'Physical GPUs differ from seal')
        if observed['compute_processes']:
            raise Incomplete('A physical GPU has live compute work; no idle claim')
        self.private.update(processes=process_rows,gpu_observation=observed)
        return dict(original_clock_epoch=original_clock_epoch, campaign_elapsed_s=elapsed,
                    occupied_card_s=occupied, completed_card_leases=len(leases),
                    process_receipts=len(processes), both_cards_idle=True, unresolved_operations=0,
                    preserved_work_failures=len(work['failures']),
                    timing_scope='Ends at campaign finish; this CPU postflight is timed separately')

    def storage(self, external, *, now=None):
        now = time.time() if now is None else now
        require(external['status']=='passed' and external['plan_binding']==self.plan['binding']
                and external['execution_commit']==self.plan['execution_commit'], 'External storage provenance differs')
        require(external['host']!=socket.gethostname(), 'External receipt must measure the peer location')
        require(0<=now-external['checked_epoch']<=3600, 'External location receipt is stale')
        require(external['all_project_allocations_included'] is True and bool(external['measurements']),
                'Peer storage coverage is not explicit')
        measured = sum(row['allocated_bytes'] for row in external['measurements'])
        require(measured==external['allocated_bytes'] and measured>=0, 'External allocation total differs')
        allowance = external['remaining_write_allowance_bytes']
        require(type(allowance) is int and allowance>=0, 'Publication/retrieval allowance missing')
        require(external['free_bytes']>=external['reserve_bytes']+allowance,
                'Peer location reserve does not cover planned writes')
        remote = allocation(self.plan['allocation_roots'], self.budget)
        baseline = self.plan['compiler_baseline']
        directories = sorted(set(baseline['directories'].values()))
        cache_roots = [path for path in directories if Path(path).exists()]
        cache = allocation(cache_roots,self.budget) if cache_roots else dict(allocated_bytes=0,entries=0,roots=[])
        growth = max(0,cache['allocated_bytes']-baseline['allocated_bytes'])
        external_charge = max(measured,self.plan['storage']['external_charge_bytes'])
        charged = remote['allocated_bytes']+growth+external_charge
        free = shutil.disk_usage(self.root).free
        # Receipt output is small but itself a new write; reserve an explicit 64MiB.
        receipt_allowance = 64*1024*1024
        remaining = allowance+receipt_allowance
        peak = max(self.plan['storage']['projected_peak_bytes'],charged+remaining)
        require(charged+remaining<=LIMITS['new_allocated_bytes'] and peak<=LIMITS['new_allocated_bytes'],
                'All-location storage ceiling exceeded')
        require(free>=LIMITS['free_reserve_bytes']+receipt_allowance, 'A800 free reserve violated')
        snapshots = []
        for path in sorted(set(self.root.glob('STORAGE.*.json')) | set((self.root/'forecasts').rglob('*.json'))):
            row = self.json(path)
            storage = row.get('storage',row)
            if 'charged_bytes' not in storage:
                raise Incomplete('Unrecognized storage/forecast receipt: '+str(path))
            require(storage['charged_bytes']+storage.get('remaining_peak_bytes',0)<=LIMITS['new_allocated_bytes']
                    and storage['free_bytes']>=storage.get('required_free_bytes',LIMITS['free_reserve_bytes'])
                    and storage['projected_peak_bytes']<=LIMITS['new_allocated_bytes'],
                    'Historical storage admission violation')
            snapshots.append(storage)
        if not snapshots:
            raise Incomplete('Historical resource admission receipts missing')
        self.private['allocation'] = dict(remote=remote,compiler=cache,external=external)
        return dict(remote_allocated_bytes=remote['allocated_bytes'],compiler_growth_bytes=growth,
                    external_measured_bytes=measured,external_charge_bytes=external_charge,
                    charged_bytes=charged,free_bytes=free,projected_peak_bytes=peak,
                    remaining_peak_bytes=remaining,required_free_bytes=LIMITS['free_reserve_bytes']+receipt_allowance,
                    historical_snapshots=len(snapshots),
                    recorded_max_charged_bytes=max(row['charged_bytes'] for row in snapshots),
                    recorded_min_free_bytes=min(row['free_bytes'] for row in snapshots),
                    coverage='Measured declared roots and pinned cache growth; snapshot evidence, not continuous usage proof')


def pid_state(pid):
    path = Path('/proc')/str(int(pid))
    if not path.exists():
        return dict(exists=False)
    try:
        return dict(exists=True,stat=(path/'stat').read_text(),command=(path/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace'))
    except FileNotFoundError:
        return dict(exists=False)


def gpu_state():
    """Management query only: does not import CUDA, initialize a model or allocate VRAM."""
    ids = subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader,nounits'],text=True,timeout=30)
    processes = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory',
                                         '--format=csv,noheader,nounits'],text=True,timeout=30)
    return dict(checked_epoch=time.time(),gpu_uuids=[line.strip() for line in ids.splitlines() if line.strip()],
                compute_processes=list(csv.reader(io.StringIO(processes.strip()))) if processes.strip() else [])


def run(plan_path, *, plan_sha256, original_clock_epoch, external_storage, output, budget=None,
        gpu_probe=None,process_probe=None):
    output = Path(output)
    require(not output.exists(), 'Use a new postflight destination; preserve prior attempts')
    output.mkdir(parents=True)
    budget = budget or Budget()
    plan = read(plan_path)
    audit = Postflight(plan,budget=budget)
    def seal():
        require(budget.hash(plan_path)['sha256']==plan_sha256, 'Original plan bytes changed')
        require(plan['binding']==digest({k:v for k,v in plan.items() if k!='binding'}), 'Plan seal mismatch')
        require(plan['limits']==LIMITS, 'Fixed resource limits changed')
        require(plan['arms']==list(ARMS) and plan['events']==58, 'Fixed campaign shape changed')
        return dict(plan_identity=budget.hash(plan_path))
    audit.section('plan',seal)
    # Do not follow paths supplied by an unauthenticated execution plan.
    if audit.sections['plan']['status']=='passed':
        audit.section('sources_and_protected',audit.sources_and_protected)
        audit.section('retention',audit.retention)
        audit.section('processes_and_clock',lambda:audit.processes_and_clock(
            original_clock_epoch=original_clock_epoch,gpu_probe=gpu_probe,process_probe=process_probe))
        audit.section('storage',lambda:audit.storage(audit.json(external_storage)))
    result_status = ('failed' if any(row['status']=='FAILED' for row in audit.errors) else
                     'incomplete' if audit.errors else 'passed')
    checked = time.time()
    inventory = dict(status=result_status,checked_epoch=checked,
                     files=sorted(audit.present.values(),key=lambda row:row['path']),absent_paths=sorted(audit.absent))
    receipt = dict(status=result_status,plan_binding=plan['binding'],execution_commit=plan['execution_commit'],
                   checked_epoch=checked,sections=audit.sections,storage=audit.sections.get('storage',{}),
                   source_chain=dict(original_execution_commit=plan['execution_commit'],
                       **audit.sections.get('sources_and_protected',{'status':'incomplete'})),
                   streamed_hash_bytes=budget.hash_bytes,filesystem_entries=budget.entries,
                   cpu_postflight_wall_s=time.monotonic()-budget.started,
                   limitations=['No model, optimizer, tensor deserialization or GPU computation is executed.',
                                'Source/retention/resource audit supplements, and does not replace, outcome/statistical audits.',
                                'Peer-location receipt is independently measured input; no network query is performed.',
                                'Shared implementation authorship/context is disclosed; this is not blind external review.'])
    write(output/'FINAL_TENSOR_INVENTORY.json',inventory)
    write(output/'PRIVATE_DETAILS.json',dict(errors=audit.errors,inputs=audit.inputs,details=audit.private))
    receipt_budget = Budget(max_hash_bytes=64*1024*1024,max_entries=10,max_seconds=60)
    try:
        receipt['tensor_inventory_identity'] = receipt_budget.hash(output/'FINAL_TENSOR_INVENTORY.json')
        receipt['private_details_identity'] = receipt_budget.hash(output/'PRIVATE_DETAILS.json')
        receipt['auditor_identity'] = receipt_budget.hash(__file__)
        receipt['receipt_hash_bytes'] = receipt_budget.hash_bytes
    except Incomplete:
        receipt['status'] = 'incomplete' if result_status!='failed' else 'failed'
        receipt['receipt_hash_status'] = 'incomplete'
    write(output/'POSTFLIGHT.json',receipt)
    return receipt


def measure_external(roots, *, binding,commit,remaining_bytes,reserve_bytes,output,budget=None):
    budget = budget or Budget()
    roots = list(map(lambda p:str(Path(p).absolute()),roots))
    require(bool(roots), 'Specify every project allocation root, including shared Git/publication charge')
    # A single inode-deduplicated measurement avoids counting nested roots twice.
    measured = allocation(roots,budget)
    devices = {Path(path).stat().st_dev for path in roots}
    require(len(devices)==1, 'Measure additional filesystems separately; this bounded helper admits one peer filesystem')
    free = shutil.disk_usage(roots[0]).free
    require(type(remaining_bytes) is int and remaining_bytes>=0 and reserve_bytes>=0,'Invalid planned allocation')
    require(free>=reserve_bytes+remaining_bytes,'Peer reserve insufficient')
    receipt = dict(status='passed',plan_binding=binding,execution_commit=commit,host=socket.gethostname(),
                   checked_epoch=time.time(),all_project_allocations_included=True,
                   measurements=[measured],allocated_bytes=measured['allocated_bytes'],free_bytes=free,
                   reserve_bytes=reserve_bytes,remaining_write_allowance_bytes=remaining_bytes,
                   scope='Caller-declared complete roots; includes planned retrieval/publication allowance')
    require(not Path(output).exists(),'Preserve prior measurement')
    write(output,receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command',required=True)
    native = commands.add_parser('postflight')
    for key in ('plan','plan-sha256','external-storage','output'):
        native.add_argument('--'+key,required=True)
    native.add_argument('--original-clock-epoch',type=float,required=True)
    external = commands.add_parser('measure-external')
    external.add_argument('--root',action='append',required=True)
    for key in ('binding','commit','output'):
        external.add_argument('--'+key,required=True)
    external.add_argument('--remaining-bytes',type=int,required=True)
    external.add_argument('--reserve-bytes',type=int,default=20_000_000_000)
    args = parser.parse_args()
    if args.command=='postflight':
        receipt=run(args.plan,plan_sha256=args.plan_sha256,original_clock_epoch=args.original_clock_epoch,
                    external_storage=args.external_storage,output=args.output)
    else:
        receipt=measure_external(args.root,binding=args.binding,commit=args.commit,remaining_bytes=args.remaining_bytes,
                                 reserve_bytes=args.reserve_bytes,output=args.output)
    print(json.dumps(dict(status=receipt['status'],output=args.output)))
    raise SystemExit(0 if receipt['status']=='passed' else 2 if receipt['status']=='incomplete' else 1)


if __name__=='__main__':
    main()
