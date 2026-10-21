"""Seal a CPU-only probe plan, run that exact plan after approval, or report."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from experiments.dflash_baseline_learning_curve.run import write_json, _git_identity
from experiments.dflash_baseline_learning_curve.experiment import preserve_invalid_numbers
from .protocol import require, digest
from .evidence import load_evidence, build_plan, validate_plan

HERE = Path(__file__).resolve().parent


def _separate(out, root):
    require(not out.is_relative_to(root) and not root.is_relative_to(out),
            'New diagnostic output must be separate from checkout and retained evidence')


def progress_index(result):
    """Compact durable status; exact tokens/rows remain in PLAN/EVENTS/native files.

    Rewriting complete copied histories on every stage would add needless disk
    traffic. This index keeps every selected location and attempted arm visible.
    The final RESULT contains the full raw records and bound plan.
    """
    value = {key: item for key, item in result.items() if key not in ('plan', 'groups')}
    value['plan_sha256'] = result['plan']['plan_sha256']
    value['groups'] = []
    for group in result['groups']:
        current = {key: item for key, item in group.items() if key not in ('requests', 'locations')}
        current['locations'] = [dict(anchor_id=loc['anchor']['anchor_id'], status=loc['status'],
            attempted_arms=loc['attempted_arms'], recorded_arms=list(loc['rows']), validations=loc['validations'],
            comparisons={arm: {'valid': pair['valid']} for arm, pair in loc['comparisons'].items()},
            error=loc.get('error')) for loc in group['locations']]
        value['groups'].append(current)
    return preserve_invalid_numbers(value)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'run', 'report'))
    parser.add_argument('--baseline-root', type=Path)
    parser.add_argument('--diagnostic-root', type=Path)
    parser.add_argument('--lower-lr-root', type=Path)
    parser.add_argument('--sealed-plan', type=Path)
    parser.add_argument('--result', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--approved-code-sha256')
    parser.add_argument('--allow-gpu-execution', action='store_true')
    args = parser.parse_args(argv)
    out = args.out.resolve()
    checkout = HERE.parents[1]
    require(not out.exists(), 'Never overwrite a plan, diagnostic attempt or report')
    _separate(out, checkout)
    roots = (args.baseline_root, args.diagnostic_root, args.lower_lr_root)
    if args.action == 'report':
        require(args.result and not any((*roots, args.sealed_plan, args.allow_gpu_execution,
                                        args.approved_code_sha256)), 'Reporting reads only an existing result')
        _separate(out, args.result.resolve().parent)
        from .report import write_reports
        write_reports(json.loads(args.result.read_text()), out)
        return 0
    require(all(roots) and args.result is None, 'All three original evidence roots are required')
    for root in roots:
        _separate(out, root.resolve())
    if args.action == 'run':
        require(args.allow_gpu_execution and args.approved_code_sha256 and args.sealed_plan,
                'Native probes require explicit execution authorization, reviewed code and a sealed plan')
        _separate(out, args.sealed_plan.resolve().parent)
    else:
        require(not any((args.allow_gpu_execution, args.approved_code_sha256, args.sealed_plan)),
                'Plan is CPU-only and cannot execute or silently reuse a plan')
    bundle = load_evidence(*roots)
    plan = build_plan(bundle)
    validate_plan(plan, bundle)
    from .native import source_pins
    from experiments.dflash_baseline_learning_curve.native import source_pins as baseline_sources
    from experiments.dflash_kl_native_diagnostic.native import source_pins as diagnostic_sources
    from experiments.dflash_lower_lr_control.native import source_pins as lower_lr_sources
    cfg = bundle['runtime_config']
    require(cfg['learning_curve_source_sha256'] == baseline_sources() and
            cfg['diagnostic_source_sha256'] == diagnostic_sources() and
            cfg['lower_lr_source_sha256'] == lower_lr_sources(),
            'Historical qualified source changed')
    inventory = source_pins()
    code_hash, git = digest(inventory), _git_identity()
    if args.action == 'run':
        sealed = json.loads(args.sealed_plan.read_text())
        validate_plan(sealed, bundle)
        require(sealed == plan, 'Sealed selection/artifact bindings differ; never reselect for execution')
        require(args.approved_code_sha256 == code_hash and not git['dirty'],
                'Native execution requires clean reviewed source')
    out.mkdir(parents=True)
    write_json(out/'PLAN.json', plan)
    write_json(out/'EVIDENCE_BINDINGS.json', bundle['bindings'])
    write_json(out/'SOURCE_MANIFEST.json', dict(git=git, matched_prefix_source_sha256=inventory,
        approved_code_sha256=code_hash, plan_sha256=plan['plan_sha256'],
        command_argv=sys.argv if argv is None else argv))
    if args.action == 'plan':
        print(json.dumps(dict(status='planned', gpu_started=False, plan_sha256=plan['plan_sha256'],
                              approved_code_sha256=code_hash)))
        return 0
    from .native import NativeBackend, storage_preflight
    from .experiment import run_campaign
    from .report import write_reports
    config = dict(cfg, output_root=str(out/'native'), matched_prefix_source_sha256=inventory,
                  matched_prefix_native_runtime=bundle['native_runtime'])
    write_json(out/'STORAGE_PREFLIGHT.json', storage_preflight(config))
    write_json(out/'RUNTIME_CONFIG.json', config)

    def emit(kind, value):
        with (out/'EVENTS.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(dict(event=kind, data=preserve_invalid_numbers(value)), allow_nan=False)+'\n')
            stream.flush()
            os.fsync(stream.fileno())

    result = run_campaign(plan, lambda group: NativeBackend(config), emit=emit,
                          snapshot=lambda value: write_json(out/'PROGRESS.json', progress_index(value)))
    result['source_manifest'] = str(out/'SOURCE_MANIFEST.json')
    result['evidence_bindings'] = str(out/'EVIDENCE_BINDINGS.json')
    write_json(out/'RESULT.json', result)
    write_reports(result, out/'report')
    print(json.dumps({key: result[key] for key in ('status', 'planned_arm_evaluations',
                                                  'attempted_arm_evaluations', 'actual_draft_block_evaluations')}))
    return 0 if result['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
