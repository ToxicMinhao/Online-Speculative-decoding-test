#!/usr/bin/env python3
"""List bounded compact PR60 artifacts; never transfer or read model tensors.

The operator may run this after process closure. It writes a NUL-delimited
relative file list plus source hashes for a separate, explicitly owned transfer.
No network, shell, experiment import, archive creation, or source-tree mutation.
"""

import argparse
import hashlib
import json
from pathlib import Path
import stat

COMMIT = '225802657d630c0985abd68d6e7453f3e47c5b74'
ARMS = ('CONTROL', 'FRONTIER_P')
PHASES = ('collect', 'train_CONTROL', 'train_FRONTIER_P', 'evaluate')
MAX_FILE_BYTES = 64_000_000
MAX_TOTAL_BYTES = 1_000_000_000


def required_paths():
    paths = {
        'sealed/PLAN.json',
        'sealed/INPUT_SELECTION.json',
        'sealed/CANONICALS.json',
        'sealed/WARMUP_CANONICALS.json',
        'LAUNCH.json',
        'OCCUPIED_CARDS.json',
        'COLLECTION_ADMISSION.json',
        'CONTINUATION_ADMISSION.json',
        'STORAGE_STAGE.json',
        'shared/WORK.json',
        'shared/FEATURE_IDENTITIES.json',
        'shared/PHYSICAL_WORK.json',
        'evaluation_physical_work.json',
        'evaluation_work.json',
        'evaluation/STATUS.json',
        'evaluation/INPUTS.json',
        'evaluation/PROTOCOL.json',
        'evaluation/REFERENCES.json',
        'evaluation/MEASUREMENTS.jsonl',
        'analysis/SERVING.json',
        'analysis/WITNESSES.json',
        'witness/BINDINGS.json',
        'witness/parent.json',
        'witness/CONTROL.json',
        'witness/FRONTIER_P.json',
    }
    for arm in ARMS:
        paths.update(
            'arms/' + arm + '/' + name for name in ('STATUS.json', 'WORK.json')
        )
    for window in range(1, 129):
        directory = f'arms/FRONTIER_P/trainer/events/event_002/allocation_{window:03d}'
        paths.update(directory + '/' + name for name in ('WORK.json', 'RESULT.json'))
    activation = 0
    for repeat in range(3):
        for group in range(2):
            for arm in (ARMS if (repeat + group) % 2 == 0 else ARMS[::-1]):
                for phase, ordinals in (
                    ('warmup', range(2)),
                    ('scored', range(group * 64, (group + 1) * 64)),
                ):
                    paths.update(
                        f'evaluation/raw/a{activation:03d}_{phase}_{ordinal:04d}_{arm}.json'
                        for ordinal in ordinals
                    )
                activation += 1
    return paths


def supplementary_paths():
    paths = {
        'CPU_PREFLIGHT.json',
        'RUN.sh',
        'PROCESS_PID',
        'PROCESS_EXIT_CODE',
        'supervisor.log',
        'analysis/COSTS_AND_WORK.json',
        'witness/SUMMARY.json',
        'shared/STATUS.json',
        'shared/INPUTS.json',
        'shared/PARENT_PUBLICATION.json',
        'evaluation/BOUNDARIES.jsonl',
        'evaluation/ACTIVATIONS.jsonl',
    }
    paths.update(
        'sealed/' + name
        for name in (
            'MODEL_PIN.json',
            'RUNTIME_PIN.json',
            'TRAINING_PIN.json',
            'ORIGINAL_CONFIG.yaml',
            'CONTROL.yaml',
            'FRONTIER_P.yaml',
            'SOURCE_CONTINUITY.json',
        )
    )
    for phase in PHASES:
        paths.update(
            (
                phase + '.log',
                'phases/' + phase + '.json',
                'phases/' + phase + '.launch.json',
            )
        )
    for arm in ARMS:
        paths.add('evaluation/WEIGHTS_' + arm + '.json')
        base = 'arms/' + arm + '/'
        paths.update(
            base + name
            for name in (
                'FINAL_STATE_CHECK.json',
                'RUNTIME_COMPATIBILITY.json',
                'trainer/FORK_BINDING.json',
                'trainer/PARENT_RESTORATION.json',
                'startup/RESULT.json',
                'startup/EXACT_RESTORATION.json',
                'startup/WORK.jsonl',
                'startup/response_observations/RESPONSE_OBSERVATIONS.jsonl',
            )
        )
        paths.update(
            base + 'trainer/events/event_002/' + name
            for name in (
                'RESULT.json',
                'PENDING.json',
                'FAILED.json',
                'RESPONSE_OBSERVATIONS.jsonl',
                'WINDOW_OBSERVATIONS.jsonl',
                'RECOVERY_SIZE_PLAN.json',
                'RECOVERY_SIZE_ACTUAL.json',
            )
        )
    return paths


def inspect_file(root, relative):
    """Reject links/devices and directory escapes before opening any content."""
    path = root
    for part in Path(relative).parts:
        if part in ('.', '..'):
            raise ValueError('Unsafe relative path')
        path = path / part
        if path.is_symlink():
            raise ValueError(
                'Symlink is outside compact retrieval contract: ' + relative
            )
    if not path.exists():
        return None
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_FILE_BYTES:
        raise ValueError('Not a bounded regular compact file: ' + relative)
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            hasher.update(block)
    after = path.stat()
    if (info.st_size, info.st_mtime_ns, info.st_ino) != (
        after.st_size,
        after.st_mtime_ns,
        after.st_ino,
    ):
        raise ValueError('File changed while inventorying: ' + relative)
    return dict(path=relative, bytes=info.st_size, sha256=hasher.hexdigest())


def inventory(root):
    root = Path(root).resolve()
    launch_info = inspect_file(root, 'LAUNCH.json')
    if launch_info is None:
        raise ValueError('No terminal launch receipt')
    launch = json.loads((root / 'LAUNCH.json').read_text())
    if launch.get('status') not in ('completed', 'failed'):
        raise ValueError('Wait for final process closure before retrieval')
    if inspect_file(root, 'PROCESS_EXIT_CODE') is None:
        raise ValueError('Supervisor exit receipt is still missing')
    int((root / 'PROCESS_EXIT_CODE').read_text().strip())
    if inspect_file(root, 'sealed/PLAN.json') is None:
        raise ValueError('Sealed plan is missing')
    plan = json.loads((root / 'sealed/PLAN.json').read_text())
    if plan.get('execution_commit') != COMMIT:
        raise ValueError('Unexpected production execution commit')
    required = required_paths()
    files, missing, total = [], [], 0
    for relative in sorted(required | supplementary_paths()):
        item = inspect_file(root, relative)
        if item is None:
            if relative in required:
                missing.append(relative)
            continue
        total += item['bytes']
        if total > MAX_TOTAL_BYTES:
            raise ValueError('Compact package exceeds the fixed 1GB retrieval bound')
        files.append(item)
    return dict(
        execution_commit=COMMIT,
        launch_status=launch['status'],
        status='complete_file_coverage' if not missing else 'incomplete_file_coverage',
        files=files,
        bytes=total,
        missing_required=missing,
        scope='Private compact evidence only; file coverage does not establish scientific validity',
        external_dependencies={
            key: plan.get(key)
            for key in (
                'source_manifest',
                'parent_plan',
                'manifest',
                'numerical_result',
                'integration_result',
                'phase_clock',
            )
        },
        retained_local_dependencies=[
            'PARENT_INPUTS.json',
            'PARENT_FEATURE_IDENTITIES.json',
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--expected',
        action='store_true',
        help='Print exact whitelist without accessing a run',
    )
    parser.add_argument(
        '--root', help='Terminal production root; local filesystem only'
    )
    parser.add_argument(
        '--output', help='New inventory directory outside the production root'
    )
    args = parser.parse_args()
    if args.expected:
        print(
            json.dumps(
                dict(
                    required=sorted(required_paths()),
                    supplementary=sorted(supplementary_paths()),
                ),
                indent=2,
            )
        )
        return
    if not args.root or not args.output:
        parser.error('--root and --output are required without --expected')
    root, output = Path(args.root).resolve(), Path(args.output).resolve()
    if output == root or output.is_relative_to(root):
        raise ValueError(
            'Inventory output must be outside the immutable production root'
        )
    receipt = inventory(root)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'COMPACT_MANIFEST.json').write_text(
        json.dumps(receipt, indent=2, allow_nan=False) + '\n'
    )
    (output / 'FILES.nul').write_bytes(
        b''.join(row['path'].encode() + b'\0' for row in receipt['files'])
    )
    print(
        json.dumps({key: receipt[key] for key in ('status', 'launch_status', 'bytes')})
    )


if __name__ == '__main__':
    main()
