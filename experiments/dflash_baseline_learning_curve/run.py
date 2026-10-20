"""Prepare review metadata or explicitly execute the fixed six-group screen.

No implicit GPU launch, download, automatic retry or parameter substitution.
Dataset preparation is a separate CPU-only command in data.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent


def require(condition,message):
    if not condition: raise ValueError(message)


def write_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.partial')
    with temporary.open('x',encoding='utf-8') as handle:
        json.dump(value,handle,indent=2,ensure_ascii=False,allow_nan=False)
        handle.write('\n');handle.flush();os.fsync(handle.fileno())
    os.replace(temporary,path)


def file_binding(path):
    path=Path(path).resolve()
    return dict(path=str(path),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def source_inventory():
    # Same inventory in the controller and child native workers.
    from .native import source_pins
    return source_pins()


def code_identity(sources):
    from .native import digest
    return digest(sources)


def plan(manifest):
    return dict(status='review_only',gpu_started=False,groups=6,feedback_requests=24,
        held_out_requests=24,checkpoints=[0,2,8],optimizer_steps=48,
        training_response_presentations=96,core_response_generations=96,
        explicit_warmup_generations=18,source_loss_diagnostic_responses=72,
        source_loss_diagnostic_generations=0,training_seed=17,selection_seed=20261020,
        first_group='First of six, correctness-only gate, never repeated as an extra qualification group',
        evaluation='Training process group exited before any evaluation warmup/timing; one timed execution/checkpoint/request',
        latency='Preliminary serving-only with available checkpoints; not net online speedup',
        group_order=[g['group_id'] for g in manifest['groups']],
        stopping='Preserve and stop on mismatch/nonfinite/irreproducible binding. Null/harmful adaptation continues.',
        retention='Keep both trained checkpoints, optimizer/RNG records and compact evidence for every group; no automatic retirement')


def storage_preflight(config, *, free_bytes=None):
    from .native import base, group_storage_bound, campaign_storage_bound
    models=json.loads(base.checked_binding(config['model_bindings']).read_text())
    size=sum(v['bytes'] for name,v in models['released_draft']['checkpoint_files'].items() if name.endswith('.safetensors'))
    needed=campaign_storage_bound(size)
    reserve=config.get('reserve_bytes',8*2**30)
    require(reserve>=8*2**30,'Keep at least the existing 8 GiB reserve')
    require(config['episode_storage_budget_bytes']>=group_storage_bound(size),'Group storage maximum is insufficient')
    require(config['campaign_storage_budget_bytes']>=needed,'Campaign storage maximum omits retained groups')
    destination=Path(config['output_root']).resolve()
    ancestor=destination
    while not ancestor.exists(): ancestor=ancestor.parent
    available=shutil.disk_usage(ancestor).free if free_bytes is None else free_bytes
    require(available>=needed+reserve,'All retained groups plus staging and reserve do not fit; offload or prepare storage before launch')
    return dict(status='passed',free_bytes=available,model_bytes=size,groups=6,
        peak_new_writes_bytes=needed,reserve_bytes=reserve,total_required_free_bytes=needed+reserve,
        method='Six times (two exports + two optimizer states + staging + 512 MiB compact evidence); existing files charged through free space')


def _git_identity():
    root=HERE.parents[1]
    def git(*args):
        return subprocess.check_output(['git','-C',str(root),*args],text=True).strip()
    return dict(commit=git('rev-parse','HEAD'),branch=git('branch','--show-current'),dirty=bool(git('status','--porcelain')))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('plan','run','report'))
    parser.add_argument('--manifest',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--result',type=Path)
    parser.add_argument('--runtime-config',type=Path)
    parser.add_argument('--tokenizer',type=Path)
    parser.add_argument('--reviewed-code-sha256')
    parser.add_argument('--reviewed-manifest-sha256')
    parser.add_argument('--allow-gpu-execution',action='store_true')
    args=parser.parse_args(argv)
    require(not args.out.exists() and not args.out.resolve().is_relative_to(HERE.parents[1]),
            'Use a fresh artifacts directory outside the public checkout')
    if args.action=='report':
        require(args.result and not any((args.allow_gpu_execution,args.runtime_config,args.manifest)),
                'Report reads existing results only')
        from .report import write_reports
        write_reports(json.loads(args.result.read_text()),args.out)
        return 0
    require(args.manifest is not None and args.result is None,'Prepared immutable sample manifest required')
    from .data import load_manifest, validate_reencoded_inputs
    manifest=load_manifest(args.manifest)
    sources=source_inventory(); code_hash=code_identity(sources); sample=file_binding(args.manifest)
    if args.action=='plan':
        require(not any((args.allow_gpu_execution,args.runtime_config,args.reviewed_code_sha256,args.reviewed_manifest_sha256)),
                'Planning cannot execute GPU code')
    else:
        require(args.allow_gpu_execution and args.runtime_config and args.tokenizer,
                'Explicit reviewed execution flag, native config and pinned tokenizer required')
        require(args.reviewed_code_sha256==code_hash,'Code differs from reviewed inventory')
        require(args.reviewed_manifest_sha256==sample['sha256'],'Prepared requests differ from reviewed manifest')
    args.out.mkdir(parents=True)
    write_json(args.out/'PLAN.json',plan(manifest))
    write_json(args.out/'SOURCE_MANIFEST.json',dict(code_sha256=sources,reviewed_code_sha256=code_hash,
        sample_manifest=sample,git=_git_identity(),command_argv=sys.argv if argv is None else argv))
    if args.action=='plan':
        print(json.dumps(dict(status='review_only',gpu_started=False,reviewed_code_sha256=code_hash,
                              reviewed_manifest_sha256=sample['sha256'])))
        return 0
    from experiments.dflash_per_turn.framing import load_pinned_tokenizer
    tokenizer=load_pinned_tokenizer(args.tokenizer)
    validate_reencoded_inputs(manifest,tokenizer)
    write_json(args.out/'TEXT_QUALIFICATION.json',dict(status='passed',requests=48,
        exact_user_text_and_input_token_ids=True,tokenizer_binding=manifest['tokenizer']))
    del tokenizer
    from .native import NativeBackend, base
    config=json.loads(args.runtime_config.read_text())
    require(config.get('output_cap')==1024 and config.get('block_size')==10,'Qualified serving settings changed')
    # All artifacts, including native weights, remain under this one private run.
    config['output_root']=str(args.out/'native')
    config['learning_curve_source_sha256']=sources
    base.check_environment(config)
    write_json(args.out/'STORAGE_PREFLIGHT.json',storage_preflight(config))
    write_json(args.out/'RUNTIME_CONFIG.json',config)
    write_json(args.out/'SAMPLE_MANIFEST.json',manifest)
    from .experiment import run_campaign
    from .report import write_reports
    def emit(kind,value):
        with (args.out/'EVENTS.jsonl').open('a',encoding='utf-8') as handle:
            handle.write(json.dumps(dict(event=kind,data=value),ensure_ascii=False,allow_nan=False)+'\n')
            handle.flush();os.fsync(handle.fileno())
    def snapshot(value):write_json(args.out/'PROGRESS.json',value)
    started=time.monotonic()
    result=run_campaign(manifest['groups'],lambda group:NativeBackend(config),emit=emit,snapshot=snapshot)
    result['actual_campaign_wall_s']=time.monotonic()-started
    result['source_manifest']=file_binding(args.out/'SOURCE_MANIFEST.json')
    write_json(args.out/'RESULT.json',result)
    write_reports(result,args.out/'report')
    print(json.dumps(dict(status=result['status'],groups_completed=sum(g['status']=='complete' for g in result['groups']),
                          actual_campaign_wall_s=result['actual_campaign_wall_s'])))
    return 0 if result['status']=='complete' else 1


if __name__=='__main__':
    raise SystemExit(main())
