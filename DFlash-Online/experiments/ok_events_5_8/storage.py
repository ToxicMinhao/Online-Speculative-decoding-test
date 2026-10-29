"""Phase admission and verified, idle-boundary offload to Fenrir.

The experiment waits while a separate courier copies the explicit file list.
Only a receipt covering every byte permits removal of disposable local copies.
Latest restart states and current serving exports remain available on A800.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

from online_feedback.storage import write_json

GIB = 1024 ** 3
WEIGHTS = 2097259104
FEATURE_ROW_BYTES = 4096 * 6 * 2 + 12


def identity(path):
    path = Path(path)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(bytes=path.stat().st_size, sha256=digest)


def required_bytes(payload):
    # 10% tensor slack, 2GiB compact evidence, 2GiB workspace, 8GiB reserve.
    return int(payload * 1.10) + 12 * GIB


def validate_ack(request, ack):
    if (ack.get('status') != 'verified' or ack.get('event') != request['event']
            or ack.get('files') != request['files']
            or ack.get('campaign_root') != request['campaign_root']
            or ack.get('destination') != request['destination']):
        raise ValueError('Offload receipt does not verify the entire requested file set')


class StorageLifecycle:
    def __init__(self, plan, output):
        self.output = Path(output).resolve()
        self.root = self.output / 'retention'
        self.root.mkdir()
        self.settings = plan['storage']
        if self.settings['reserve_bytes'] != 8 * GIB:
            raise ValueError('Preserve the established 8GiB storage reserve')
        self.archived = {}

    def admit(self, phase, payload, basis):
        free = shutil.disk_usage(self.output).free
        row = dict(phase=phase, payload_bytes=payload, required_bytes=required_bytes(payload),
                   free_bytes=free, reserve_bytes=8*GIB, basis=basis,
                   scope='Only this next phase; existing occupied bytes already reduce free space')
        row['admitted'] = free >= row['required_bytes']
        write_json(self.root / (phase + '_STORAGE.json'), row)
        if not row['admitted']:
            raise RuntimeError('Insufficient peak storage: ' + phase)
        return row

    def admit_collection(self, window):
        rows = sum(len(r['prompt_ids']) + r['max_new_tokens'] for r in window['requests'])
        return self.admit(f"event_{window['event_index']:03d}_collection",
                          rows * FEATURE_ROW_BYTES, 'All128 maximum-length native feature captures')

    def admit_preparation(self, window, records):
        rows = sum(len(r['prompt_ids']) + len(r['response_ids']) for r in records)
        return self.admit(f"event_{window['event_index']:03d}_preparation",
            rows * FEATURE_ROW_BYTES, 'Prepared feature copies; captured originals already occupy disk')

    def admit_training(self, arm, event, pool, parents):
        # After O finishes a new event, its previous export is no longer a
        # dependency. Retire that verified remote copy before admitting K's
        # pending checkpoint; K's own parent export remains protected.
        self.retire_obsolete_exports(parents)
        # training.py writes one 3W pending state, then exports W, then atomically
        # replaces latest. Add another W for export temporaries. Failure does
        # not write another model snapshot. Existing parents are already charged.
        return self.admit(f'event_{event:03d}_{arm}_training', 5 * WEIGHTS,
                         '3W pending Adam state + W export + W export workspace; original parents retained')

    def retire_obsolete_exports(self, parents):
        before_free_bytes = shutil.disk_usage(self.output).free
        protected = {Path(parent['export']).resolve() for parent in parents.values()}
        removed = []
        for arm in ('O', 'K'):
            for export in (self.output / arm / 'events').glob('event_*/export'):
                if export.resolve() in protected:
                    continue
                for path in export.iterdir():
                    if not path.is_file():
                        continue
                    relative = str(path.relative_to(self.output))
                    if relative not in self.archived:
                        continue
                    if path.is_symlink() or identity(path) != self.archived[relative]:
                        raise ValueError('Obsolete export changed after verified offload: ' + str(path))
                    path.unlink(); removed.append(str(path))
        if removed:
            ledger = self.root / 'OBSOLETE_EXPORTS.json'
            prior = json.loads(ledger.read_text()) if ledger.exists() else []
            write_json(ledger, prior + [dict(removed_local_copies=removed,
                reason='No active parent requires these byte-verified Fenrir copies',
                protected_exports=sorted(str(path) for path in protected),
                before_free_bytes=before_free_bytes,
                free_bytes=shutil.disk_usage(self.output).free)])

    def after_event(self, event, parents, pool, *, evaluated):
        if evaluated != (event in (6, 8)):
            raise ValueError('Offload must follow this event\'s complete required evaluation')
        started = time.perf_counter()
        before_free_bytes = shutil.disk_usage(self.output).free
        event_root = self.output / f'event_{event:03d}'
        removable = set(Path(path) for path in pool['features'])
        # Source capture and terminal teacher packets remain reproducible on Fenrir.
        for path in (event_root / 'responses').glob('*.json'):
            record = json.loads(path.read_text())
            removable.add(Path(record['metadata']['capture']['feature_path']))
        removable.update((event_root / 'features').glob('*.pt'))
        for path in (event_root / 'features').glob('*.terminal.json'):
            terminal = json.loads(path.read_text())
            removable.add(Path(terminal['tensor_path']))
        if any(path.is_symlink() for path in removable):
            raise ValueError('Offload refuses symlinked tensor artifacts')
        removable = {path.resolve() for path in removable}
        files = set(removable)
        for parent in parents.values():
            files.update(path for path in Path(parent['export']).iterdir() if path.is_file())
            if event in (6, 8):
                files.add(Path(parent['state']))
        for path in files:
            if path.is_symlink() or not path.resolve().is_relative_to(self.output):
                raise ValueError('Offload escaped this new campaign: ' + str(path))
        request = dict(event=event, campaign_root=str(self.output),
            files={str(p.relative_to(self.output)): identity(p) for p in sorted(files)},
            phase='idle_after_training_and_evaluation', destination=self.settings['offload_destination'])
        write_json(self.root / f'event_{event:03d}_REQUEST.json', request)
        ack_path = self.root / f'event_{event:03d}_ACK.json'
        deadline = time.monotonic() + self.settings['offload_timeout_s']
        while not ack_path.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError('Verified offload not completed; no local copies removed')
            time.sleep(5)
        ack = json.loads(ack_path.read_text())
        validate_ack(request, ack)
        # Reverify the remote copies before deletion, outside any timed serving.
        for relative, expected in request['files'].items():
            if identity(self.output / relative) != expected:
                raise ValueError('Artifact changed during offload: ' + relative)
        self.archived.update(request['files'])
        removed = []
        for path in sorted(removable):
            path.unlink(); removed.append(str(path))
        # Old exports are no longer used by either trainer or the native worker.
        # Every old export was copied at its own completed boundary first.
        for arm in parents:
            for previous in range(5, event):
                export = self.output / arm / 'events' / f'event_{previous:03d}' / 'export'
                for path in export.glob('*'):
                    if path.is_file():
                        relative = str(path.relative_to(self.output))
                        if relative not in self.archived or identity(path) != self.archived[relative]:
                            raise ValueError('Old export has no matching verified offload: ' + str(path))
                        path.unlink(); removed.append(str(path))
        # Retain event6 completely on Fenrir; latest remains the same inode on
        # A800 until event7 atomically replaces it. Resume from that live path.
        if event == 6:
            for arm, parent in parents.items():
                state, latest = Path(parent['state']), self.output / arm / 'training_state.pt'
                if not os.path.samefile(state, latest):
                    raise ValueError('Retained event6 link differs from current restart state')
                state.unlink(); removed.append(str(state)); parent['state'] = str(latest)
        receipt = dict(status='completed', event=event, destination=ack['destination'],
            verified_files=request['files'], removed_local_copies=removed,
            retained_latest={arm: parent['state'] for arm, parent in parents.items()},
            before_free_bytes=before_free_bytes, free_bytes=shutil.disk_usage(self.output).free,
            wall_s=time.perf_counter()-started,
            timing_scope='Idle between event phases; excluded from serving latency')
        write_json(self.root / f'event_{event:03d}_RETENTION.json', receipt)
        return receipt

    def finalize(self, parents):
        for event in (6, 8):
            for arm in ('O', 'K'):
                for relative in (f'{arm}/events/event_{event:03d}/training_state.pt',
                                 f'{arm}/events/event_{event:03d}/export/model.safetensors'):
                    if relative not in self.archived:
                        raise ValueError('Required endpoint is not retained: ' + relative)
        write_json(self.root/'FINAL.json', dict(status='passed', required_events=[6,8],
            verified_files=self.archived, destination=self.settings['offload_destination'],
            latest_states={arm: parent['state'] for arm,parent in parents.items()}))
