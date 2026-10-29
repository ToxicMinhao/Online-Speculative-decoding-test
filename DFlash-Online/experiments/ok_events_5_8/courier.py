"""Copy explicit idle-boundary artifacts from A800 and acknowledge verified bytes.

Only small REQUEST/STATUS receipts are read while waiting. Bulk transfer starts
after REQUEST, while the campaign is blocked on ACK. Failed transfers retain
partial files and logs; this process never restarts the scientific campaign.
"""
import argparse
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import time

from online_feedback.storage import write_json
from .storage import GIB, identity

SSH = ('ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=60', 'A800')


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
                or '\x00' in relative or relative == '.' or relative.startswith('_courier/')):
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


def transfer_files(campaign, destination, relative_files, control, event, *, timeout):
    """Stream tar through SSH; no remote rsync or intermediate archive required."""
    file_list = control / f'event_{event:03d}_FILES.bin'
    file_list.write_bytes(b''.join(name.encode('utf-8') + b'\0' for name in relative_files))
    command = ('tar --create --file=- --no-recursion --null --verbatim-files-from '
               '--directory=' + shlex.quote(campaign) + ' --files-from=-')
    remote = local = None
    with (control / f'event_{event:03d}_SSH.stderr').open('wb') as remote_error, \
            (control / f'event_{event:03d}_TAR.stderr').open('wb') as local_error:
        try:
            remote = subprocess.Popen([*SSH, command], stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE, stderr=remote_error)
            local = subprocess.Popen(['tar', '--extract', '--file=-', '--no-same-owner',
                '--no-same-permissions', '--keep-old-files', '--directory=' + str(destination),
                '--null', '--verbatim-files-from', '--files-from=' + str(file_list)],
                stdin=remote.stdout, stdout=subprocess.DEVNULL, stderr=local_error)
            remote.stdout.close()
            remote.stdin.write(file_list.read_bytes())
            remote.stdin.close()
            started = time.monotonic()
            remote_code = remote.wait(timeout=timeout)
            local_code = local.wait(timeout=max(1, timeout - (time.monotonic() - started)))
            if remote_code or local_code:
                raise RuntimeError(f'Transfer failed: SSH/tar={remote_code}, local tar={local_code}')
        finally:
            for process in (local, remote):
                if process is not None:
                    stop_process(process)


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


def copy_event(request, destination, control, *, transfer_timeout=3600):
    event, campaign = request['event'], request['campaign_root']
    payload = sum(value['bytes'] for value in request['files'].values())
    free = shutil.disk_usage(destination).free
    space = dict(free_bytes=free, payload_bytes=payload, reserve_bytes=8*GIB,
                 workspace_bytes=GIB, required_bytes=payload + 9*GIB)
    write_json(control / f'event_{event:03d}_SPACE.json', space)
    if free < space['required_bytes']:
        raise RuntimeError('Insufficient local space for payload plus 8GiB reserve and 1GiB workspace')
    missing = []
    for relative, expected in request['files'].items():
        path = local_path(destination, relative)
        if path.exists():
            if not path.is_file() or identity(path) != expected:
                raise ValueError('Existing local file differs; preserved without overwrite: ' + relative)
        else:
            missing.append(relative)
    started = time.perf_counter()
    if missing:
        transfer_files(campaign, destination, sorted(missing), control, event, timeout=transfer_timeout)
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
        free_bytes_before=free, free_bytes_after=free_after,
        timing_scope='Campaign idle boundary; no transfer during timed serving')
    write_json(control / f'event_{event:03d}_VERIFIED.json', ack)
    publish_ack(campaign, event, ack)
    write_json(control / f'event_{event:03d}_ACK_SENT.json', ack)
    return ack


def run(remote_root, destination, *, wait_timeout=21600, transfer_timeout=3600):
    campaign = remote_campaign(remote_root)
    destination = Path(destination).absolute()
    destination.mkdir(parents=True, exist_ok=True)
    control = destination / '_courier'
    control.mkdir(exist_ok=False)
    progress = dict(status='running', campaign_root=campaign, destination=str(destination),
                    acknowledged_events=[], automatic_retry=False, poll_seconds=15)
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
            copy_event(request, destination, control, transfer_timeout=transfer_timeout)
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
        write_json(control / 'STATUS.json', progress)
    return progress


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--remote-root', required=True)
    parser.add_argument('--destination', required=True)
    parser.add_argument('--wait-timeout-s', type=int, default=21600)
    parser.add_argument('--transfer-timeout-s', type=int, default=3600)
    args = parser.parse_args()
    run(args.remote_root, args.destination, wait_timeout=args.wait_timeout_s,
        transfer_timeout=args.transfer_timeout_s)
