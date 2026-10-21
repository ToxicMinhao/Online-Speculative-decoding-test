"""Seal, execute or report the prespecified three-pool P8 allocation control."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys

from experiments.dflash_baseline_learning_curve.run import write_json, _git_identity
from experiments.dflash_baseline_learning_curve.experiment import require, digest, preserve_invalid_numbers
from experiments.dflash_kl_native_diagnostic.evidence import file_binding
from .evidence import load_evidence
from .protocol import SCHEMA, PLANNED, SERVING_CONTRACT
from .storage import build_storage_plan

HERE = Path(__file__).resolve().parent


def plan(bundle):
    return dict(schema=SCHEMA,status='planned',gpu_started=False,
        pool_order=[p['pool_id'] for p in bundle['pools']],
        pool_bindings=bundle['bindings']['pools'],new_arm='P8',reused_arms=['F','S8'],
        learning_rate=3e-4,seed=17,**PLANNED,
        checkpoints_to_export=3,optimizer_state_exports=0,intermediate_checkpoints=0,
        order='A0,B0,A1,B1,A2,B2,A3,B3 repeated twice; A/B follow original manifest order',
        initialization='Three independent released DFlash/fresh AdamW/seed17 trajectories',
        supervision='Own prompt supplies context; only each saved source response has loss; held-out never trains',
        anchors='Fresh normal training RNG for interleaved order; original PR9 source KL and PR10 held-out KL anchors',
        serving_contract=dict(SERVING_CONTRACT),storage_budget=build_storage_plan(bundle['pools']),
        continuation='First complete pool correctness gate, then remaining two unchanged, including null/harmful results',
        timing='Fresh P8 measurements only; historical/fresh latency does not establish speedup',
        limits='Per-model steps/presentations matched, not pure diversity or equal compute; three trajectories, one seed',
        paused='Agreement-filter, correction-bank and new matched-prefix work remain paused')


def _separate(path, protected, description):
    for root in protected:
        require(not path.is_relative_to(root) and not root.is_relative_to(path), description)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('plan','run','report'))
    parser.add_argument('--baseline-root',type=Path)
    parser.add_argument('--diagnostic-root',type=Path)
    parser.add_argument('--sealed-plan',type=Path)
    parser.add_argument('--native-root',type=Path,help='Optional fresh checkpoint/worker destination on a qualified filesystem')
    parser.add_argument('--result',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--approved-code-sha256')
    parser.add_argument('--allow-gpu-execution',action='store_true')
    args=parser.parse_args(argv)
    out=args.out.resolve();checkout=HERE.parents[1]
    require(not out.exists(),'Use a fresh private output directory')
    _separate(out,[checkout],'Private results must remain outside the public checkout')
    if args.action=='report':
        require(args.result and not any((args.baseline_root,args.diagnostic_root,args.sealed_plan,args.native_root,
            args.allow_gpu_execution,args.approved_code_sha256)), 'Report only reads an existing result')
        from .report import write_reports
        write_reports(json.loads(args.result.read_text()),out)
        return 0
    require(args.baseline_root and args.diagnostic_root and args.result is None,'Both retained evidence roots are required')
    protected=[checkout,args.baseline_root.resolve(),args.diagnostic_root.resolve()]
    _separate(out,protected,'New work must remain separate from retained evidence and public checkout')
    bundle=load_evidence(args.baseline_root,args.diagnostic_root)
    from .loss import validate_pool_diagnostics
    for pool in bundle['pools']:
        validate_pool_diagnostics(pool)
    from .native import source_pins
    from experiments.dflash_baseline_learning_curve.native import source_pins as curve_sources
    from experiments.dflash_kl_native_diagnostic.native import source_pins as diagnostic_sources
    require(bundle['runtime_config']['learning_curve_source_sha256']==curve_sources() and
        bundle['runtime_config']['diagnostic_source_sha256']==diagnostic_sources(),'Historical qualified source changed')
    inventory=source_pins();code_hash=digest(inventory);git=_git_identity();allocation=plan(bundle)
    allocation['plan_sha256']=digest(allocation)
    if args.action=='run':
        require(args.allow_gpu_execution and args.approved_code_sha256==code_hash and not git['dirty'],
                'GPU execution requires explicit authorization and clean pinned source')
        require(args.sealed_plan is not None,'Seal pool membership, order and assignments before training')
        sealed=json.loads(args.sealed_plan.read_text())
        require(sealed==allocation,'Current pools/evaluation assignments differ from sealed plan')
    else:
        require(not args.allow_gpu_execution and args.approved_code_sha256 is None and
                args.sealed_plan is None and args.native_root is None,'Plan cannot start model work or select a runtime destination')
    out.mkdir(parents=True)
    write_json(out/'PLAN.json',allocation);write_json(out/'EVIDENCE_BINDINGS.json',bundle['bindings'])
    write_json(out/'SOURCE_MANIFEST.json',dict(git=git,feedback_allocation_source_sha256=inventory,
        approved_code_sha256=code_hash,original_baseline_commit=bundle['bindings']['baseline_commit'],
        diagnostic_commit=bundle['bindings']['diagnostic']['diagnostic_commit'],
        sealed_plan_binding=file_binding(args.sealed_plan) if args.sealed_plan else None,
        command_argv=sys.argv if argv is None else argv))
    if args.action=='plan':
        print(json.dumps(dict(status='planned',gpu_started=False,approved_code_sha256=code_hash,
                              plan_sha256=allocation['plan_sha256'])))
        return 0
    from .native import NativeBackend, storage_preflight
    native_root=args.native_root.resolve() if args.native_root else out/'native'
    require(not native_root.exists(),'Native destination must be fresh; preserve existing evidence')
    _separate(native_root,protected,'Native artifacts must not overlap retained evidence or public checkout')
    require(native_root != out and not out.is_relative_to(native_root),'Native destination cannot contain controller output')
    config=dict(bundle['runtime_config'],output_root=str(native_root),feedback_allocation_source_sha256=inventory,
        feedback_allocation_learning_rate=3e-4,feedback_allocation_seed=17,serving_contract=dict(SERVING_CONTRACT),
        storage_budget=allocation['storage_budget'],controller_root=str(out))
    try:
        preflight=storage_preflight(config)
    except Exception as error:
        write_json(out/'STORAGE_PREFLIGHT.json',dict(getattr(error,'storage_preflight',{}),
            status='failed',error=str(error),gpu_started=False,
            destination=str(native_root),original_evidence_preserved=True))
        raise
    write_json(out/'STORAGE_PREFLIGHT.json',preflight);write_json(out/'RUNTIME_CONFIG.json',config)
    from .experiment import run_campaign
    from .report import write_reports
    def emit(kind,value):
        with (out/'EVENTS.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(dict(event=kind,data=preserve_invalid_numbers(value)),allow_nan=False)+'\n')
            stream.flush();os.fsync(stream.fileno())
    result=run_campaign(bundle['pools'],lambda pool:NativeBackend(config),emit=emit,
        snapshot=lambda value:write_json(out/'PROGRESS.json',value))
    result['evidence_bindings']=file_binding(out/'EVIDENCE_BINDINGS.json')
    result['source_manifest']=file_binding(out/'SOURCE_MANIFEST.json')
    result['plan_sha256']=allocation['plan_sha256']
    write_json(out/'RESULT.json',result);write_reports(result,out/'report')
    print(json.dumps({key:result[key] for key in ('status','optimizer_steps','training_presentations',
        'actual_native_generations','actual_response_kl_evaluations')}))
    return 0 if result['status']=='complete' else 1


if __name__=='__main__':
    raise SystemExit(main())
