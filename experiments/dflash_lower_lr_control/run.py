"""Plan, execute or report the one prespecified lower-LR control."""
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
    return dict(status='planned', gpu_started=False, group_order=[g['group_id'] for g in bundle['groups']],
        new_arm='L8', reused_arms=['F','S8'], learning_rate=3e-5, seed=17,
        optimizer_steps=48, training_presentations=96, native_generations=48, response_kl_evaluations=48,
        explicit_warmups=0, checkpoints_to_export=6, optimizer_state_exports=0,
        initialization='Both S8 and L8 originally start from released DFlash and fresh AdamW; only retained S8 artifacts are reused now.',
        supervision='Own prompt supplies context; only each source response has loss. Held-out answers never train.',
        anchors='Original per-presentation training trajectory; separate PR9 source KL and PR10 held-out KL anchors.',
        continuation='First complete group correctness gate, then five more even when null or harmful.',
        timing='Fresh L8 costs only; historical/fresh timings never establish speedup.',
        paused='Agreement-filter and correction-bank campaigns remain paused.')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('plan','run','report'))
    parser.add_argument('--baseline-root',type=Path)
    parser.add_argument('--diagnostic-root',type=Path)
    parser.add_argument('--result',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--approved-code-sha256')
    parser.add_argument('--allow-gpu-execution',action='store_true')
    args=parser.parse_args(argv)
    out=args.out.resolve();checkout=HERE.parents[1]
    require(not out.exists() and not out.is_relative_to(checkout) and not checkout.is_relative_to(out),
        'Use a fresh private directory outside the public checkout')
    if args.action=='report':
        require(args.result and not any((args.baseline_root,args.diagnostic_root,args.allow_gpu_execution,args.approved_code_sha256)),
            'Report reads only an existing result')
        from .report import write_reports
        write_reports(json.loads(args.result.read_text()),out)
        return 0
    require(args.baseline_root and args.diagnostic_root and args.result is None,'Both retained evidence roots are required')
    for root in (args.baseline_root,args.diagnostic_root):
        require(not out.is_relative_to(root.resolve()) and not root.resolve().is_relative_to(out),
            'New work must remain separate from retained evidence')
    bundle=load_evidence(args.baseline_root,args.diagnostic_root)
    from .native import source_pins
    from experiments.dflash_baseline_learning_curve.native import source_pins as curve_sources
    from experiments.dflash_kl_native_diagnostic.native import source_pins as diagnostic_sources
    require(bundle['runtime_config']['learning_curve_source_sha256']==curve_sources() and
        bundle['runtime_config']['diagnostic_source_sha256']==diagnostic_sources(), 'Historical qualified sources changed')
    inventory=source_pins();code_hash=digest(inventory);git=_git_identity()
    if args.action=='run':
        require(args.allow_gpu_execution and args.approved_code_sha256==code_hash and not git['dirty'],
            'Execution requires explicit authorization and clean pinned source')
    else:
        require(not args.allow_gpu_execution and args.approved_code_sha256 is None,'Plan cannot start models')
    out.mkdir(parents=True)
    write_json(out/'PLAN.json',plan(bundle));write_json(out/'EVIDENCE_BINDINGS.json',bundle['bindings'])
    write_json(out/'SOURCE_MANIFEST.json',dict(git=git,lower_lr_source_sha256=inventory,
        approved_code_sha256=code_hash,original_baseline_commit=bundle['bindings']['baseline_commit'],
        diagnostic_commit=bundle['bindings']['diagnostic']['diagnostic_commit'],
        command_argv=sys.argv if argv is None else argv))
    if args.action=='plan':
        print(json.dumps(dict(status='planned',gpu_started=False,approved_code_sha256=code_hash)))
        return 0
    from .native import NativeBackend,storage_preflight
    config=dict(bundle['runtime_config'],output_root=str(out/'native'),lower_lr_source_sha256=inventory,
        lower_lr_learning_rate=3e-5,lower_lr_seed=17)
    write_json(out/'STORAGE_PREFLIGHT.json',storage_preflight(config));write_json(out/'RUNTIME_CONFIG.json',config)
    from .experiment import run_campaign
    from .report import write_reports
    def emit(kind,value):
        with (out/'EVENTS.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(dict(event=kind,data=preserve_invalid_numbers(value)),allow_nan=False)+'\n')
            stream.flush();os.fsync(stream.fileno())
    result=run_campaign(bundle['groups'],lambda group:NativeBackend(config),emit=emit,
        snapshot=lambda value:write_json(out/'PROGRESS.json',value))
    result['evidence_bindings']=file_binding(out/'EVIDENCE_BINDINGS.json')
    result['source_manifest']=file_binding(out/'SOURCE_MANIFEST.json')
    write_json(out/'RESULT.json',result);write_reports(result,out/'report')
    print(json.dumps({key:result[key] for key in ('status','optimizer_steps','training_presentations',
        'actual_native_generations','actual_response_kl_evaluations')}))
    return 0 if result['status']=='complete' else 1


if __name__=='__main__':
    raise SystemExit(main())
