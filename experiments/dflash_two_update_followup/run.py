#!/usr/bin/env python3
"""Review a CPU-only plan, or explicitly launch one reviewed native episode."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

from experiment import (MODES, load_dataset, load_reference, plan, require,
                        run_author_episode, run_checkpoint_comparison, validate_reference)
from report import summarize


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'run'))
    parser.add_argument('--comparison', choices=MODES, default='both')
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--streams', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--episode')
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--reference-sha256')
    parser.add_argument('--runtime-config', type=Path)
    parser.add_argument('--allow-gpu-execution', action='store_true')
    args = parser.parse_args()
    require(not args.out.exists(), 'Use a fresh output directory; preserve earlier attempts')
    require(not args.out.resolve().is_relative_to(Path(__file__).resolve().parents[2]),
            'Runtime artifacts belong outside the public code checkout')
    streams = load_dataset(args.selection, args.streams)
    protocol = plan(streams, mode=args.comparison)
    if args.action == 'plan':
        require(not any((args.episode, args.reference, args.reference_sha256, args.runtime_config, args.allow_gpu_execution)),
                'Plan is metadata-only and does not accept native execution arguments')
    else:
        require(args.allow_gpu_execution and args.runtime_config and args.episode and args.reference and args.reference_sha256,
                'Execution needs explicit approval flag, episode, runtime config and bound original result')
        require(args.episode in {e['episode_id'] for e in protocol['episodes']}, 'Only R1/R2 may execute')
    args.out.mkdir(parents=True)
    write(args.out / 'PLAN.json', protocol)
    write(args.out / 'SOURCE_MANIFEST.json', {
        'code_sha256': {p.name: sha(p) for p in sorted(Path(__file__).parent.glob('*')) if p.is_file()},
        'selection_sha256': sha(args.selection), 'streams_file_sha256': sha(args.streams)})
    if args.action == 'plan':
        print(json.dumps({'status': 'review_only', 'episodes': 10, 'gpu_started': False}))
        return 0
    rows = [r for r in streams if r['stream_id'] == args.episode]
    reference = load_reference(args.reference, args.reference_sha256)
    validate_reference(reference, rows)
    config = json.loads(args.runtime_config.read_text())
    config['output_root'] = str(args.out.resolve() / 'native')
    write(args.out / 'RUNTIME_CONFIG.json', config)
    write(args.out / 'REFERENCE_BINDING.json', reference['source_binding'])
    # All GPU imports and worker creation are behind this explicit run branch.
    from native import NativeBackend
    backend = NativeBackend(config)
    def emit(kind, data):
        with (args.out / 'EVENTS.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps({'event': kind, 'data': data}, ensure_ascii=False, allow_nan=False) + '\n')
            f.flush(); os.fsync(f.fileno())
    # The author final matrix already includes F/C1/C2; both never duplicates it.
    execute = run_checkpoint_comparison if args.comparison == 'checkpoints-only' else run_author_episode
    result = execute(rows, reference, backend, emit=emit)
    write(args.out / 'RESULT.json', result)
    write(args.out / 'REPORT.json', summarize([result]))
    print(json.dumps({'status': result['status'], 'episode': args.episode, 'report': str(args.out / 'REPORT.json')}))
    return 0 if result['status'] == 'complete' else 1


if __name__ == '__main__':
    sys.exit(main())
