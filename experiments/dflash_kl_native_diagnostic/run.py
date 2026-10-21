"""Plan, explicitly execute, or report the bounded retained-checkpoint diagnostic."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from experiments.dflash_baseline_learning_curve.run import write_json, _git_identity
from experiments.dflash_baseline_learning_curve.experiment import require, digest, preserve_invalid_numbers
from .evidence import load_evidence, file_binding

HERE = Path(__file__).resolve().parent


def plan(bundle):
    return dict(status='review_only', gpu_started=False, group_order=[g['group_id'] for g in bundle['groups']],
        checkpoints=['F', 'S8'], source_requests=24, heldout_requests=24, new_source_generations=24,
        new_response_kl_evaluations=48, optimizer_steps=0, explicit_warmups=0,
        extra_operations='Teacher preparation, projection chunks, loading and validation are separately accounted.',
        source='Reuse F/S8 KL and F native; generate S8 naturally from exact original prompts.',
        heldout='Reuse F/S8 native; evaluate saved canonical answers with paired read-only KL.',
        validity='First complete group validates correctness; unfavorable results continue unchanged.',
        interpretation='Better native drafting means fewer calls for identical tokens, not established serving-time improvement.',
        limit='Patterns locate further investigation; they do not identify cause or establish forward KL is generally unsuitable.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'run', 'report'))
    parser.add_argument('--baseline-root', type=Path)
    parser.add_argument('--result', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--reviewed-code-sha256')
    parser.add_argument('--allow-gpu-execution', action='store_true')
    args = parser.parse_args(argv)
    out = args.out.resolve()
    checkout = HERE.parents[1]
    require(not out.exists() and not out.is_relative_to(checkout) and not checkout.is_relative_to(out),
            'Use a fresh private diagnostic directory outside the public checkout')
    if args.action == 'report':
        require(args.result and not any((args.baseline_root, args.allow_gpu_execution, args.reviewed_code_sha256)),
                'Report only reads an existing diagnostic result')
        from .report import write_reports
        write_reports(json.loads(args.result.read_text()), out)
        return 0
    require(args.baseline_root is not None and args.result is None, 'Original baseline root required')
    require(not out.is_relative_to(args.baseline_root.resolve()) and not args.baseline_root.resolve().is_relative_to(out),
            'Diagnostic output must be separate from original evidence')
    bundle = load_evidence(args.baseline_root)
    from .native import source_pins
    from experiments.dflash_baseline_learning_curve.native import source_pins as baseline_sources
    require(bundle['runtime_config']['learning_curve_source_sha256'] == baseline_sources(),
            'Qualified original source implementation changed')
    inventory = source_pins()
    code_hash = digest(inventory)
    git = _git_identity()
    if args.action == 'run':
        require(args.allow_gpu_execution and args.reviewed_code_sha256 == code_hash and not git['dirty'],
                'Execution requires explicit approval flag and clean reviewed code identity')
    else:
        require(not args.allow_gpu_execution and args.reviewed_code_sha256 is None, 'Plan never starts native workers')
    out.mkdir(parents=True)
    write_json(out / 'PLAN.json', plan(bundle))
    write_json(out / 'EVIDENCE_BINDINGS.json', bundle['bindings'])
    write_json(out / 'SOURCE_MANIFEST.json', dict(git=git, diagnostic_source_sha256=inventory,
        reviewed_code_sha256=code_hash, original_baseline_commit=bundle['bindings']['baseline_commit'],
        command_argv=sys.argv if argv is None else argv))
    if args.action == 'plan':
        print(json.dumps(dict(status='review_only', gpu_started=False, reviewed_code_sha256=code_hash)))
        return 0
    from .native import NativeBackend, storage_preflight
    config = dict(bundle['runtime_config'], output_root=str(out / 'native'), diagnostic_source_sha256=inventory)
    write_json(out / 'STORAGE_PREFLIGHT.json', storage_preflight(config))
    write_json(out / 'RUNTIME_CONFIG.json', config)
    from .experiment import run_campaign
    from .report import write_reports
    def emit(kind, value):
        with (out / 'EVENTS.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(dict(event=kind, data=preserve_invalid_numbers(value)), allow_nan=False) + '\n')
            stream.flush(); os.fsync(stream.fileno())
    result = run_campaign(bundle['groups'], lambda group: NativeBackend(config), emit=emit,
        snapshot=lambda value: write_json(out / 'PROGRESS.json', value))
    result['evidence_bindings'] = file_binding(out / 'EVIDENCE_BINDINGS.json')
    result['source_manifest'] = file_binding(out / 'SOURCE_MANIFEST.json')
    write_json(out / 'RESULT.json', result)
    write_reports(result, out / 'report')
    print(json.dumps({key: result[key] for key in ('status', 'actual_source_generations', 'actual_response_kl_evaluations', 'optimizer_steps')}))
    return 0 if result['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
