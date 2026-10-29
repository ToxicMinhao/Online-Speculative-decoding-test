"""Copy explicit idle-boundary artifacts from A800 and acknowledge verified bytes.

Only small REQUEST/STATUS receipts are read while waiting. Bulk transfer starts
after REQUEST, while the campaign is blocked on ACK. Failed transfers retain
partial files and logs; this process never restarts the scientific campaign.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time

from online_feedback.storage import write_json
from .storage import GIB, identity, validate_ack

SSH = ('ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=60', 'A800')
SEPARATE_SSH = (*SSH[:-1], '-o', 'ControlMaster=no', '-o', 'ControlPath=none', SSH[-1])
SCIENTIFIC_EXECUTION_COMMIT = '04fac901f7af5478b62ff730833c026637ddd2cd'


def remote_campaign(remote_root):
    path = PurePosixPath(remote_root)
    if (not path.is_absolute() or '..' in path.parts or str(path) != remote_root
            or not path.is_relative_to('/data/sources/minhao')):
        raise ValueError('Use the exact absolute authorized A800 run root')
    return str(path / 'campaign_001')


def read_remote_json(path):
    quoted = shlex.quote(str(path))
    command = f'if test -f {quoted}; then cat -- {quoted}; fi'
    result = subprocess.run([*SSH, command], check=True, capture_output=True,
                            text=True, timeout=180)
    return json.loads(result.stdout) if result.stdout.strip() else None


def validate_request(request, *, event, campaign, destination):
    if (type(request.get('event')) is not int or request['event'] != event
            or request.get('campaign_root') != campaign
            or request.get('destination') != str(destination)
            or request.get('phase') != 'idle_after_training_and_evaluation'):
        raise ValueError('Offload request differs from the expected idle event/destination')
    files = request.get('files')
    if not isinstance(files, dict) or not files:
        raise ValueError('Require a nonempty explicit file manifest')
    for relative, expected in files.items():
        path = PurePosixPath(relative)
        if (not relative or path.is_absolute() or '..' in path.parts or str(path) != relative
                or '\x00' in relative or relative == '.' or path.parts[0].startswith('_courier')):
            raise ValueError('Unsafe relative offload path: ' + relative)
        if (not isinstance(expected, dict) or set(expected) != {'bytes', 'sha256'}
                or type(expected['bytes']) is not int or expected['bytes'] < 0
                or not isinstance(expected['sha256'], str)
                or re.fullmatch('[0-9a-f]{64}', expected['sha256']) is None):
            raise ValueError('Malformed file identity: ' + relative)


def local_path(destination, relative):
    path = destination / relative
    root = destination.resolve()
    if not path.resolve().is_relative_to(root):
        raise ValueError('Local offload path escaped the destination: ' + relative)
    for parent in (path, *path.parents):
        if parent == destination:
            break
        if parent.is_symlink():
            raise ValueError('Local offload path contains a symlink: ' + relative)
    return path


def stop_process(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def wait_process(process, deadline, cancel):
    while True:
        if cancel is not None and cancel.is_set():
            raise InterruptedError('Another transfer worker failed; retaining partial files')
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Bounded artifact transfer timed out')
        try:
            return process.wait(timeout=min(1, remaining))
        except subprocess.TimeoutExpired:
            pass


def transfer_files(campaign, destination, relative_files, control, event, *, timeout,
                   separate_connection=False, cancel=None):
    """Stream tar through SSH; no remote rsync or intermediate archive required."""
    file_list = control / f'event_{event:03d}_FILES.bin'
    file_list.write_bytes(b''.join(name.encode('utf-8') + b'\0' for name in relative_files))
    command = ('tar --create --file=- --no-recursion --null --verbatim-files-from '
               '--directory=' + shlex.quote(campaign) + ' --files-from=-')
    remote_command = [*(SEPARATE_SSH if separate_connection else SSH), command]
    local_command = ['tar', '--extract', '--file=-', '--no-same-owner',
        '--no-same-permissions', '--keep-old-files', '--directory=' + str(destination),
        '--null', '--verbatim-files-from', '--files-from=' + str(file_list)]
    started = time.perf_counter()
    receipt = dict(status='running', event=event, files=relative_files,
                   remote_command=remote_command, local_command=local_command,
                   separate_tcp_connection=separate_connection, timeout_s=timeout)
    write_json(control / f'event_{event:03d}_TRANSFER.json', receipt)
    remote = local = None
    with (control / f'event_{event:03d}_SSH.stderr').open('wb') as remote_error, \
            (control / f'event_{event:03d}_TAR.stderr').open('wb') as local_error:
        try:
            remote = subprocess.Popen(remote_command, stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE, stderr=remote_error)
            local = subprocess.Popen(local_command,
                stdin=remote.stdout, stdout=subprocess.DEVNULL, stderr=local_error)
            remote.stdout.close()
            remote.stdin.write(file_list.read_bytes())
            remote.stdin.close()
            deadline = time.monotonic() + timeout
            remote_code = wait_process(remote, deadline, cancel)
            local_code = wait_process(local, deadline, cancel)
            receipt.update(remote_exit_code=remote_code, local_exit_code=local_code)
            if remote_code or local_code:
                raise RuntimeError(f'Transfer failed: SSH/tar={remote_code}, local tar={local_code}')
            receipt['status'] = 'completed'
        except BaseException as error:
            receipt.update(status='failed', error=repr(error))
            if cancel is not None:
                cancel.set()
            raise
        finally:
            for process in (local, remote):
                if process is not None:
                    stop_process(process)
            receipt['wall_s'] = time.perf_counter() - started
            write_json(control / f'event_{event:03d}_TRANSFER.json', receipt)


def balanced_groups(files, missing, workers):
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError('Use between one and four bounded transfer workers')
    groups, totals = [[] for _ in range(workers)], [0] * workers
    for name in sorted(missing, key=lambda name: (-files[name]['bytes'], name)):
        index = min(range(workers), key=lambda index: (totals[index], index))
        groups[index].append(name)
        totals[index] += files[name]['bytes']
    return [group for group in groups if group]


def transfer_parallel(request, destination, missing, control, *, workers, timeout):
    event, campaign = request['event'], request['campaign_root']
    groups = balanced_groups(request['files'], missing, workers)
    folders = []
    for index, group in enumerate(groups):
        folder = control / f'event_{event:03d}_workers' / f'worker_{index:02d}'
        folder.mkdir(parents=True)
        folders.append(folder)
    write_json(control / f'event_{event:03d}_BATCH.json', dict(groups=groups,
        group_bytes=[sum(request['files'][name]['bytes'] for name in group) for group in groups],
        workers=len(groups), separate_tcp_connections=True, automatic_retry=False))
    cancel = threading.Event()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        jobs = [executor.submit(transfer_files, campaign, destination, group, folder, event,
                    timeout=timeout, separate_connection=True, cancel=cancel)
                for group, folder in zip(groups, folders)]
        try:
            for job in as_completed(jobs):
                job.result()
        except BaseException:
            cancel.set()
            raise


def publish_ack(campaign, event, ack):
    path = str(PurePosixPath(campaign) / 'retention' / f'event_{event:03d}_ACK.json')
    temporary = path + '.courier.pending'
    # Payload travels on stdin, never interpolated as shell code. Do not replace
    # an existing ACK or pending file after an ambiguous transport failure.
    command = (f'test ! -e {shlex.quote(path)} && '
               f'(set -C; cat > {shlex.quote(temporary)}) && '
               f'mv -- {shlex.quote(temporary)} {shlex.quote(path)}')
    subprocess.run([*SSH, command], input=json.dumps(ack, allow_nan=False) + '\n',
                   text=True, capture_output=True, check=True, timeout=180)


def inspect_existing(request, destination, *, allow_quarantine):
    missing, partials, complete = [], [], []
    for relative, expected in request['files'].items():
        path = local_path(destination, relative)
        if not path.exists():
            missing.append(relative)
            continue
        if not path.is_file():
            raise ValueError('Existing offload member is not a regular file: ' + relative)
        actual = identity(path)
        if actual == expected:
            complete.append(relative)
        elif allow_quarantine:
            partials.append(dict(relative=relative, identity=actual, expected_complete_identity=expected))
            missing.append(relative)
        else:
            raise ValueError('Existing local file differs; preserved without overwrite: ' + relative)
    return missing, partials, complete


def quarantine_partials(partials, destination, control, event):
    receipt = dict(status='preserving', files=partials,
                   reason='Explicit manual operational resume; original request-bound partial bytes retained')
    receipt_path = control / f'event_{event:03d}_QUARANTINE.json'
    write_json(receipt_path, receipt)
    for row in partials:
        source = local_path(destination, row['relative'])
        target = control / 'quarantine' / row['relative']
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or identity(source) != row['identity']:
            raise ValueError('Partial artifact changed before quarantine')
        row.update(original_path=str(source), quarantine_path=str(target), status='move_planned')
        write_json(receipt_path, receipt)
        os.rename(source, target)
        row['status'] = 'preserved'
        write_json(receipt_path, receipt)
    receipt['status'] = 'preserved'
    write_json(receipt_path, receipt)


def retire_quarantined(partials, destination, control, event):
    """Retire only identified failed-transfer bytes after verified ACK delivery."""
    rows = []
    started = time.perf_counter()
    receipt = dict(status='retiring', files=rows, before_free_bytes=shutil.disk_usage(destination).free,
        reason='Partial local transfer superseded by complete SHA256-verified requested artifact; scientific source and complete offload retained')
    path = control / f'event_{event:03d}_PARTIAL_RETIREMENT.json'
    write_json(path, receipt)
    for item in partials:
        partial = Path(item['quarantine_path'])
        retained = local_path(destination, item['relative'])
        if identity(partial) != item['identity'] or identity(retained) != item['expected_complete_identity']:
            raise ValueError('Partial or retained complete artifact changed; no retirement')
        row = dict(retired_path=str(partial), retired_identity=item['identity'],
                   retained_path=str(retained), retained_identity=item['expected_complete_identity'], status='planned')
        rows.append(row)
        write_json(path, receipt)
        partial.unlink()
        row['status'] = 'retired'
        write_json(path, receipt)
    receipt.update(status='completed', after_free_bytes=shutil.disk_usage(destination).free,
                   wall_s=time.perf_counter()-started)
    write_json(path, receipt)


def copy_event(request, destination, control, *, transfer_timeout=3600, workers=1,
               allow_quarantine=False):
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError('Use between one and four bounded transfer workers')
    event, campaign = request['event'], request['campaign_root']
    payload = sum(value['bytes'] for value in request['files'].values())
    started = time.perf_counter()
    missing, partials, complete = inspect_existing(request, destination, allow_quarantine=allow_quarantine)
    existing_hash_s = time.perf_counter() - started
    remaining = sum(request['files'][name]['bytes'] for name in missing)
    free = shutil.disk_usage(destination).free
    space = dict(free_bytes=free, payload_bytes=payload, remaining_payload_bytes=remaining,
                 verified_existing_files=complete, reserve_bytes=8*GIB,
                 workspace_bytes=GIB, required_bytes=remaining + 9*GIB,
                 quarantine_scope='Existing partial occupancy already reduces measured free space; rename adds no payload copy')
    write_json(control / f'event_{event:03d}_SPACE.json', space)
    if free < space['required_bytes']:
        raise RuntimeError('Insufficient local space for payload plus 8GiB reserve and 1GiB workspace')
    if partials:
        quarantine_partials(partials, destination, control, event)
    started = time.perf_counter()
    if missing:
        if workers == 1:
            transfer_files(campaign, destination, sorted(missing), control, event, timeout=transfer_timeout)
        else:
            transfer_parallel(request, destination, missing, control, workers=workers, timeout=transfer_timeout)
    transfer_s = time.perf_counter() - started
    started = time.perf_counter()
    for relative, expected in request['files'].items():
        path = local_path(destination, relative)
        if not path.is_file() or identity(path) != expected:
            raise ValueError('Transferred bytes differ; partial files retained: ' + relative)
    free_after = shutil.disk_usage(destination).free
    if free_after < 8*GIB:
        raise RuntimeError('Local reserve fell below 8GiB; verified files retained without ACK')
    ack = dict(status='verified', event=event, campaign_root=campaign,
        destination=str(destination), files=request['files'], transfer_s=transfer_s,
        local_hash_s=time.perf_counter()-started, transferred_files=len(missing),
        existing_hash_s=existing_hash_s, verified_existing_files=len(complete), workers=workers,
        free_bytes_before=free, free_bytes_after=free_after,
        timing_scope='Campaign idle boundary; no transfer during timed serving')
    write_json(control / f'event_{event:03d}_VERIFIED.json', ack)
    publish_ack(campaign, event, ack)
    write_json(control / f'event_{event:03d}_ACK_SENT.json', ack)
    if partials:
        retire_quarantined(partials, destination, control, event)
    return ack


def create_control(destination, campaign, *, resume_manual, courier_execution_commit):
    previous = []
    if resume_manual:
        if (not isinstance(courier_execution_commit, str)
                or re.fullmatch('[0-9a-f]{40}', courier_execution_commit) is None):
            raise ValueError('Manual courier recovery requires its separate complete Git revision')
        original = destination / '_courier'
        if not original.is_dir() or original.is_symlink():
            raise ValueError('Manual recovery requires original courier records')
        previous = [original, *sorted(destination.glob('_courier_recovery_[0-9][0-9][0-9]'))]
        for folder in previous:
            if folder.is_symlink():
                raise ValueError('Prior courier control directory cannot be a symlink')
            state = json.loads((folder / 'STATUS.json').read_text())
            if state.get('campaign_root') != campaign or state.get('destination') != str(destination):
                raise ValueError('Prior courier records belong to a different campaign/destination')
        for number in range(1, 1000):
            control = destination / f'_courier_recovery_{number:03d}'
            if not control.exists():
                control.mkdir()
                break
        else:
            raise ValueError('Courier recovery directory limit reached')
    else:
        control = destination / '_courier'
        control.mkdir(exist_ok=False)
    receipt = dict(manual_operational_resume=resume_manual, automatic_retry=False,
        operator_requirement='Previous courier and its transfer children must be stopped before manual resume',
        scientific_execution_commit=SCIENTIFIC_EXECUTION_COMMIT,
        courier_execution_commit=courier_execution_commit, courier_source_identity=identity(Path(__file__)),
        previous_control_files={str(path): identity(path) for folder in previous
            for path in sorted(folder.rglob('*')) if path.is_file() and 'quarantine' not in path.parts},
        scope='Local artifact transport only; no scientific campaign/source/configuration restart or change')
    write_json(control / 'RECOVERY_PROVENANCE.json', receipt)
    return control, previous


def bind_previous_requests(request, previous):
    matched = []
    for folder in previous:
        path = folder / f"event_{request['event']:03d}_REQUEST.json"
        if path.exists():
            if json.loads(path.read_text()) != request:
                raise ValueError('Remote REQUEST differs from preserved original courier request')
            matched.append(str(path))
    return matched


def manual_event_state(request, destination, previous, control):
    event, campaign = request['event'], request['campaign_root']
    matched = bind_previous_requests(request, previous)
    ack = read_remote_json(f'{campaign}/retention/event_{event:03d}_ACK.json')
    if ack is not None:
        validate_ack(request, ack)
        missing, _, _ = inspect_existing(request, destination, allow_quarantine=False)
        if missing:
            raise ValueError('Previously acknowledged offload is missing a complete local artifact')
        write_json(control / f'event_{event:03d}_PRIOR_ACK_VERIFIED.json', ack)
        return True, False
    status = read_remote_json(f'{campaign}/STATUS.json')
    if (not status or status.get('status') != 'running'
            or status.get('phase') != f'retention_{event:03d}' or status.get('event') != event):
        raise ValueError('Unacknowledged campaign is not paused at the exact requested retention boundary')
    write_json(control / f'event_{event:03d}_IDLE_BINDING.json',
               dict(status=status, previous_request_receipts=matched, allow_partial_quarantine=bool(matched)))
    return False, bool(matched)


def run(remote_root, destination, *, wait_timeout=21600, transfer_timeout=3600,
        workers=1, resume_manual=False, courier_execution_commit=None):
    if type(workers) is not int or not 1 <= workers <= 4 or wait_timeout <= 0 or transfer_timeout <= 0:
        raise ValueError('Require positive timeouts and one to four transfer workers')
    campaign = remote_campaign(remote_root)
    destination = Path(destination).absolute()
    destination.mkdir(parents=True, exist_ok=True)
    control, previous = create_control(destination, campaign, resume_manual=resume_manual,
                                      courier_execution_commit=courier_execution_commit)
    started_total = time.perf_counter()
    progress = dict(status='running', campaign_root=campaign, destination=str(destination),
                    acknowledged_events=[], automatic_retry=False, poll_seconds=15,
                    manual_operational_resume=resume_manual, workers=workers,
                    courier_execution_commit=courier_execution_commit,
                    scientific_execution_commit=SCIENTIFIC_EXECUTION_COMMIT,
                    settings=dict(wait_timeout_s=wait_timeout, transfer_timeout_s=transfer_timeout),
                    entrypoint=' '.join(map(shlex.quote, [sys.executable, '-m', 'experiments.ok_events_5_8.courier',
                        '--remote-root', str(remote_root), '--destination', str(destination), '--workers', str(workers),
                        '--wait-timeout-s', str(wait_timeout), '--transfer-timeout-s', str(transfer_timeout),
                        *(['--resume-manual', '--courier-execution-commit', courier_execution_commit] if resume_manual else [])])))
    write_json(control / 'STATUS.json', progress)
    try:
        for event in (5, 6, 7, 8):
            progress['event'] = event
            write_json(control / 'STATUS.json', progress)
            started = time.monotonic()
            while True:
                request = read_remote_json(f'{campaign}/retention/event_{event:03d}_REQUEST.json')
                if request is not None:
                    break
                status = read_remote_json(f'{campaign}/STATUS.json')
                if status and status.get('status') in ('failed', 'completed'):
                    raise RuntimeError('Campaign stopped before required offload request: ' + str(status.get('status')))
                if time.monotonic() - started >= wait_timeout:
                    raise TimeoutError('No offload request within the bounded courier wait')
                time.sleep(15)
            write_json(control / f'event_{event:03d}_REQUEST.json', request)
            validate_request(request, event=event, campaign=campaign, destination=destination)
            already_acknowledged, allow_quarantine = (manual_event_state(request, destination, previous, control)
                                                       if resume_manual else (False, False))
            if not already_acknowledged:
                copy_event(request, destination, control, transfer_timeout=transfer_timeout,
                           workers=workers, allow_quarantine=allow_quarantine)
            progress['acknowledged_events'].append(event)
            write_json(control / 'STATUS.json', progress)
        progress['status'] = 'completed'
    except BaseException as error:
        progress.update(status='failed', error_type=type(error).__name__, error=str(error))
        event = progress.get('event')
        progress['ack_delivery_uncertain'] = bool(event is not None
            and (control / f'event_{event:03d}_VERIFIED.json').exists()
            and not (control / f'event_{event:03d}_ACK_SENT.json').exists())
        write_json(control / 'FAILED.json', progress)
        raise
    finally:
        progress['courier_wall_s'] = time.perf_counter() - started_total
        write_json(control / 'STATUS.json', progress)
    return progress


if __name__ == '__main__':
    import signal
    def interrupted(signum, frame):
        raise InterruptedError('Courier supervisor signal ' + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--remote-root', required=True)
    parser.add_argument('--destination', required=True)
    parser.add_argument('--wait-timeout-s', type=int, default=21600)
    parser.add_argument('--transfer-timeout-s', type=int, default=3600)
    parser.add_argument('--workers', type=int, choices=range(1, 5), default=1)
    parser.add_argument('--resume-manual', action='store_true')
    parser.add_argument('--courier-execution-commit')
    args = parser.parse_args()
    run(args.remote_root, args.destination, wait_timeout=args.wait_timeout_s,
        transfer_timeout=args.transfer_timeout_s, workers=args.workers,
        resume_manual=args.resume_manual, courier_execution_commit=args.courier_execution_commit)
